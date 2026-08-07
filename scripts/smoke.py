"""Lightweight live-server smoke test.

Run with ``python scripts/smoke.py`` while the gateway is running. Docker-specific
checks are reported as skipped when the daemon is not reachable.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx


def docker_available() -> bool:
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


async def collect_turn(websocket: Any, prompt: str) -> list[dict[str, object]]:
    await websocket.send(json.dumps({"type": "user_message", "content": prompt}))
    events: list[dict[str, object]] = []
    while True:
        event = json.loads(await asyncio.wait_for(websocket.recv(), timeout=60))
        events.append(event)
        if event.get("type") in {"done", "error"}:
            return events


async def docker_smoke(base: str) -> int:
    from websockets.asyncio.client import connect

    # Blocking Path calls are fine here: the smoke script's loop serves nothing else.
    sample = Path(__file__).resolve().parents[1] / "sample_data" / "sales.csv"  # noqa: ASYNC240
    async with httpx.AsyncClient(base_url=base, timeout=120) as client:
        session = await client.post("/api/sessions")
        session.raise_for_status()
        session_id = session.json()["id"]
        exit_code = 0
        try:
            with sample.open("rb") as handle:
                upload = await client.post(
                    f"/api/sessions/{session_id}/files",
                    files={"files": ("sales.csv", handle, "text/csv")},
                )
            upload.raise_for_status()
            schemas = upload.json()["schemas"]
            assert schemas and schemas[0]["name"] == "df_1" and schemas[0]["rows"] > 0
            print("PASS upload/profile: df_1")

            websocket_url = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
            async with connect(f"{websocket_url}/ws/sessions/{session_id}") as websocket:
                ready = json.loads(await asyncio.wait_for(websocket.recv(), timeout=10))
                assert ready["type"] == "session_ready"

                dashboard = await collect_turn(websocket, "Build a dashboard")
                assert dashboard[-1]["type"] == "done"
                artifact = next(
                    event for event in dashboard
                    if event.get("type") == "artifact" and event.get("name") == "dashboard.html"
                )
                assert artifact.get("port") == 8501
                preview = await client.get(f"/api/sessions/{session_id}/preview/8501/")
                preview.raise_for_status()
                assert b"Data Copilot Dashboard" in preview.content
                print("PASS dashboard artifact/preview")

                report_events = await collect_turn(websocket, "Create a PDF whitepaper report")
                assert report_events[-1]["type"] == "done"
                assert any(
                    event.get("type") == "artifact" and event.get("name") == "report.pdf"
                    for event in report_events
                )
                report = await client.get(f"/api/sessions/{session_id}/artifacts/report.pdf")
                report.raise_for_status()
                assert report.content.startswith(b"%PDF") and b"/Count 2" in report.content
                print("PASS PDF artifact/download")
        finally:
            # Cleanup must run even when an assertion fails, but a return here
            # would swallow the original failure, so defer the exit code.
            deleted = await client.delete(f"/api/sessions/{session_id}")
            if deleted.status_code != 204:
                print(f"FAIL session delete: {deleted.status_code}")
                exit_code = 1
            else:
                print("PASS session delete")
    return exit_code


def main() -> int:
    base = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
    try:
        response = httpx.get(f"{base}/health", timeout=5)
        response.raise_for_status()
    except Exception as exc:
        print(f"SKIP server checks: backend unavailable ({exc})")
        return 0
    print(f"PASS health: {response.json()}")
    if not docker_available():
        print("SKIP Docker checks: Docker daemon unavailable (expected for local backend-only runs)")
        return 0
    try:
        return asyncio.run(docker_smoke(base))
    except Exception as exc:
        print(f"FAIL Docker-backed session checks: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
