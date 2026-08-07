from pathlib import Path

import pytest

from app.config import Settings
from app.sandbox import FakeSandbox, SessionManager


@pytest.mark.asyncio
async def test_session_manager_uses_fake_and_reaps(tmp_path: Path):
    settings = Settings(session_ttl_minutes=0)
    manager = SessionManager(settings, sandbox_factory=lambda sid: FakeSandbox(sid, tmp_path / sid))
    session = await manager.create()
    assert manager.get(session.id).status == "ready"
    # A zero-minute TTL makes this session immediately eligible.
    assert await manager.reap() == 1
    assert session.id not in manager.sessions
