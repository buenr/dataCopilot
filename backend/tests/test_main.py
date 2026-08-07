import asyncio
import json
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


@pytest.mark.asyncio
async def test_preview_rejects_unregistered_port(tmp_path: Path):
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
        # No webapp registered this port; the gateway must refuse to dial an
        # arbitrary loopback port on the client's behalf (SSRF guard).
        response = await client.get(f"/api/sessions/{session_id}/preview/8000/")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_upload_numbers_dataframes_consistently_with_profiles(tmp_path: Path):
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
        # An unsupported file that sorts first must not shift the df_N numbering
        # away from the profiles the agent is shown.
        response = await client.post(
            f"/api/sessions/{session_id}/files",
            files=[
                ("files", ("aaa-notes.txt", b"not a dataset", "text/plain")),
                ("files", ("sales.csv", b"region,revenue\nNorth,1200\n", "text/csv")),
            ],
        )

    assert response.status_code == 200
    assert [(s["name"], s["file"]) for s in response.json()["schemas"]] == [("df_1", "sales.csv")]
    session = manager.get(session_id)
    execution = await session.sandbox.exec("print(df_1['region'].iloc[0])")
    assert execution.stdout.strip() == "North"


def test_chat_socket_survives_malformed_json(tmp_path: Path):
    from fastapi.testclient import TestClient

    # Pin the mock provider so a developer .env cannot route this test to a real LLM.
    settings = Settings(sessions_dir=str(tmp_path / "sessions"), llm_provider="mock")
    manager = SessionManager(
        settings,
        sandbox_factory=lambda sid: FakeSandbox(sid, tmp_path / "workspaces" / sid),
    )
    application = create_app(settings, manager)

    with TestClient(application) as client:
        session_id = client.post("/api/sessions").json()["id"]
        with client.websocket_connect(f"/ws/sessions/{session_id}") as socket:
            assert socket.receive_json()["type"] == "session_ready"
            socket.send_text("this is not json")
            assert socket.receive_json() == {"type": "error", "message": "malformed JSON message"}
            # The socket (and its session) survives the bad frame.
            socket.send_json({"type": "user_message", "content": "hello"})
            types = [socket.receive_json()["type"] for _ in range(5)]
            assert types == [
                "user_message",
                "turn_step",
                "assistant_delta",
                "assistant_message",
                "done",
            ]


def test_cancel_with_no_active_turn_sends_cancelled_and_done(tmp_path: Path):
    from fastapi.testclient import TestClient

    settings = Settings(sessions_dir=str(tmp_path / "sessions"), llm_provider="mock")
    manager = SessionManager(
        settings,
        sandbox_factory=lambda sid: FakeSandbox(sid, tmp_path / "workspaces" / sid),
    )
    application = create_app(settings, manager)

    with TestClient(application) as client:
        session_id = client.post("/api/sessions").json()["id"]
        with client.websocket_connect(f"/ws/sessions/{session_id}") as socket:
            socket.receive_json()  # session_ready
            # A cancel with nothing running still acks so the UI can settle.
            socket.send_json({"type": "cancel"})
            events = [socket.receive_json() for _ in range(2)]
            assert events[0]["type"] == "cancelled"
            assert events[1]["type"] == "done"
            # The session remains usable after the cancel.
            socket.send_json({"type": "user_message", "content": "hello"})
            types = [socket.receive_json()["type"] for _ in range(5)]
            assert types == [
                "user_message",
                "turn_step",
                "assistant_delta",
                "assistant_message",
                "done",
            ]


def test_session_resume_replays_transcript(tmp_path: Path):
    from fastapi.testclient import TestClient

    settings = Settings(sessions_dir=str(tmp_path / "sessions"), llm_provider="mock")
    manager = SessionManager(
        settings,
        sandbox_factory=lambda sid: FakeSandbox(sid, tmp_path / "workspaces" / sid),
    )
    application = create_app(settings, manager)

    with TestClient(application) as client:
        session_id = client.post("/api/sessions").json()["id"]
        # First connection: build up a transcript.
        with client.websocket_connect(f"/ws/sessions/{session_id}") as socket:
            ready = socket.receive_json()
            assert ready["type"] == "session_ready"
            assert ready["messages"] == []
            socket.send_json({"type": "user_message", "content": "hello"})
            while True:
                event = socket.receive_json()
                if event["type"] == "done":
                    break
        # Second connection: the transcript is replayed in session_ready.
        with client.websocket_connect(f"/ws/sessions/{session_id}") as socket:
            ready = socket.receive_json()
            assert ready["type"] == "session_ready"
            messages = ready["messages"]
            assert any(m["role"] == "user" and "hello" in m["content"] for m in messages)
            assert any(m["role"] == "assistant" for m in messages)


def test_session_resume_replays_artifacts(tmp_path: Path):
    from fastapi.testclient import TestClient

    settings = Settings(sessions_dir=str(tmp_path / "sessions"), llm_provider="mock")
    manager = SessionManager(
        settings,
        sandbox_factory=lambda sid: FakeSandbox(sid, tmp_path / "workspaces" / sid),
    )
    application = create_app(settings, manager)

    with TestClient(application) as client:
        session_id = client.post("/api/sessions").json()["id"]
        # A generated dashboard exists in the sandbox workspace.
        (tmp_path / "workspaces" / session_id / "dashboard.html").write_text("<!doctype html>")
        # A reconnecting browser gets the artifact list in session_ready, so a
        # refresh does not empty the canvas.
        with client.websocket_connect(f"/ws/sessions/{session_id}") as socket:
            ready = socket.receive_json()
            assert ready["type"] == "session_ready"
            assert any(a["name"] == "dashboard.html" for a in ready["artifacts"])


def test_session_resume_restores_registered_artifact_metadata(tmp_path: Path):
    from fastapi.testclient import TestClient

    settings = Settings(sessions_dir=str(tmp_path / "sessions"), llm_provider="mock")
    manager = SessionManager(
        settings,
        sandbox_factory=lambda sid: FakeSandbox(sid, tmp_path / "workspaces" / sid),
    )
    application = create_app(settings, manager)

    with TestClient(application) as client:
        session_id = client.post("/api/sessions").json()["id"]
        workspace = tmp_path / "workspaces" / session_id
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "dashboard.html").write_text("<!doctype html>")
        (workspace / "report.pdf").write_bytes(b"%PDF-1.4 fake")
        (workspace / "chart.png").write_bytes(b"\x89PNG fake")
        (workspace / "recovered.html").write_text("<!doctype html>")
        # The sandbox scan reports name/path/size only; the trajectory remembers
        # the type and port the model declared for each artifact.
        trajectory = tmp_path / "sessions" / session_id / "trajectory.jsonl"
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        trajectory.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "tool_result",
                            "tool": "register_artifact",
                            "result": {
                                "artifacts": [
                                    {
                                        "name": "dashboard.html",
                                        "path": "dashboard.html",
                                        "size": 15,
                                        "port": 8501,
                                        "type": "webapp",
                                    }
                                ]
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "tool_result",
                            "tool": "register_artifact",
                            "result": {
                                "artifacts": [
                                    {
                                        "name": "report.pdf",
                                        "path": "report.pdf",
                                        "size": 12,
                                        "type": "pdf",
                                    }
                                ]
                            },
                        }
                    ),
                    # The dashboard recovery path records its own event rather
                    # than a register_artifact tool result.
                    json.dumps(
                        {
                            "type": "artifact_recovered",
                            "artifact": {
                                "name": "recovered.html",
                                "path": "recovered.html",
                                "port": 8502,
                                "type": "webapp",
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )

        with client.websocket_connect(f"/ws/sessions/{session_id}") as socket:
            ready = socket.receive_json()
            artifacts = {a["name"]: a for a in ready["artifacts"]}
            assert artifacts["dashboard.html"]["type"] == "webapp"
            assert artifacts["dashboard.html"]["port"] == 8501
            assert artifacts["report.pdf"]["type"] == "pdf"
            assert artifacts["recovered.html"]["type"] == "webapp"
            assert artifacts["recovered.html"]["port"] == 8502
            # An unregistered scan hit still lands on a sensible tab.
            assert artifacts["chart.png"]["type"] == "image"
            # Registered artifacts replay last so the canvas re-selects the
            # artifact the user was looking at.
            assert ready["artifacts"][-1]["name"] == "recovered.html"


@pytest.mark.asyncio
async def test_dataset_delete_renumbers_remaining_frames(tmp_path: Path):
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
        # Upload two datasets so the second is numbered df_2.
        await client.post(
            f"/api/sessions/{session_id}/files",
            files=[
                ("files", ("sales.csv", b"region,revenue\nNorth,1200\n", "text/csv")),
                ("files", ("costs.csv", b"item,cost\nA,50\n", "text/csv")),
            ],
        )
        # Delete the first; the remaining one must be renumbered to df_1.
        response = await client.delete(f"/api/sessions/{session_id}/files/sales.csv")
        assert response.status_code == 200
        schemas = response.json()["schemas"]
        assert len(schemas) == 1
        assert schemas[0]["name"] == "df_1"
        assert schemas[0]["file"] == "costs.csv"
        # The kernel reflects the renumbering.
        session = manager.get(session_id)
        execution = await session.sandbox.exec("print(df_1['item'].iloc[0])")
        assert execution.stdout.strip() == "A"


@pytest.mark.asyncio
async def test_delete_missing_dataset_returns_404(tmp_path: Path):
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
        response = await client.delete(f"/api/sessions/{session_id}/files/nonexistent.csv")
        assert response.status_code == 404


