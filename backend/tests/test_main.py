from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.main import create_app
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
