import asyncio
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.main import cors_origins, create_app
from app.sandbox import FakeSandbox, SessionManager


@pytest.mark.asyncio
async def test_upload_profiles_and_preloads_dataframe(tmp_path: Path):
    settings = Settings(sessions_dir=str(tmp_path / "sessions"))
    manager = SessionManager(
        settings,
        sandbox_factory=lambda sid: FakeSandbox(sid, tmp_path / "workspaces" / sid),
    )
    application = create_app(settings, manager)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://test",
    ) as client:
        created = await client.post("/api/sessions")
        assert created.status_code == 200
        session_id = created.json()["id"]
        response = await client.post(
            f"/api/sessions/{session_id}/files",
            files={"files": ("sales.csv", b"region,revenue\nNorth,1200\nSouth,950\n", "text/csv")},
        )

    assert response.status_code == 200
    schema = response.json()["schemas"][0]
    assert schema["file"] == "sales.csv"
    assert schema["rows"] == 2
    session = manager.get(session_id)
    assert session.dataset_profiles == response.json()["schemas"]
    execution = await session.sandbox.exec("print(df_1['revenue'].mean())")
    assert execution.stderr == ""
    assert execution.stdout.strip() == "1075.0"


@pytest.mark.asyncio
async def test_preview_serves_dashboard_when_static_server_directory_is_wrong(tmp_path: Path):
    settings = Settings(sessions_dir=str(tmp_path / "sessions"))
    manager = SessionManager(
        settings,
        sandbox_factory=lambda sid: FakeSandbox(sid, tmp_path / "workspaces" / sid),
    )
    application = create_app(settings, manager)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://test",
    ) as client:
        created = await client.post("/api/sessions")
        session_id = created.json()["id"]
        session = manager.get(session_id)
        await session.sandbox.upload("dashboard.html", b"<html><body>People Pulse</body></html>")
        await session.sandbox.start_webapp(
            "python -m http.server 18501 --directory /home/oai/share",
            18501,
        )
        try:
            await asyncio.sleep(0.1)
            response = await client.get(
                f"/api/sessions/{session_id}/preview/18501/",
            )
        finally:
            await session.sandbox.close()

    assert response.status_code == 200
    assert "People Pulse" in response.text


@pytest.mark.asyncio
async def test_preview_replaces_static_directory_listing_with_dashboard(tmp_path: Path):
    settings = Settings(sessions_dir=str(tmp_path / "sessions"))
    manager = SessionManager(
        settings,
        sandbox_factory=lambda sid: FakeSandbox(sid, tmp_path / "workspaces" / sid),
    )
    application = create_app(settings, manager)
    empty_directory = tmp_path / "empty-preview"
    empty_directory.mkdir()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://test",
    ) as client:
        created = await client.post("/api/sessions")
        session_id = created.json()["id"]
        session = manager.get(session_id)
        await session.sandbox.upload("dashboard.html", b"<html><body>People Pulse</body></html>")
        await session.sandbox.start_webapp(
            f"python -m http.server 18502 --directory {empty_directory}",
            18502,
        )
        try:
            await asyncio.sleep(0.1)
            response = await client.get(
                f"/api/sessions/{session_id}/preview/18502/",
            )
        finally:
            await session.sandbox.close()

    assert response.status_code == 200
    assert "Directory listing for" not in response.text
    assert "People Pulse" in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin",
    ["http://localhost:5173", "http://127.0.0.1:5173"],
)
async def test_preflight_allows_both_loopback_spellings_of_the_dev_origin(tmp_path: Path, origin: str):
    settings = Settings(
        sessions_dir=str(tmp_path / "sessions"),
        frontend_origin="http://localhost:5173",
    )
    manager = SessionManager(
        settings,
        sandbox_factory=lambda sid: FakeSandbox(sid, tmp_path / "workspaces" / sid),
    )
    application = create_app(settings, manager)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://test",
    ) as client:
        response = await client.options(
            "/api/sessions",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_production_cors_origins_are_not_expanded():
    settings = Settings(app_env="production", frontend_origin="http://localhost:5173")
    assert cors_origins(settings) == ["http://localhost:5173"]
