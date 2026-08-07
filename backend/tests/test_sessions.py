import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.config import Settings
from app.sandbox import DockerSandbox, FakeSandbox, SessionManager


@pytest.mark.asyncio
async def test_session_manager_uses_fake_and_reaps(tmp_path: Path):
    settings = Settings(session_ttl_minutes=0)
    manager = SessionManager(settings, sandbox_factory=lambda sid: FakeSandbox(sid, tmp_path / sid))
    session = await manager.create()
    assert manager.get(session.id).status == "ready"
    # A zero-minute TTL makes this session immediately eligible.
    assert await manager.reap() == 1
    assert session.id not in manager.sessions


@pytest.mark.asyncio
async def test_touch_keeps_active_session_alive(tmp_path: Path):
    settings = Settings(session_ttl_minutes=30)
    manager = SessionManager(settings, sandbox_factory=lambda sid: FakeSandbox(sid, tmp_path / sid))
    session = await manager.create()
    # Simulate a long-lived websocket conversation: the connection is old, but
    # messages keep arriving, so each one refreshes the idle clock.
    session.last_activity -= 31 * 60
    manager.touch(session.id)
    assert await manager.reap() == 0
    # Once traffic stops, the same session becomes eligible again.
    session.last_activity -= 31 * 60
    assert await manager.reap() == 1
    manager.touch("missing-session")  # unknown ids are ignored


@pytest.mark.asyncio
async def test_create_runs_blocking_sandbox_factory_off_the_event_loop(tmp_path: Path):
    settings = Settings()
    loop_thread = threading.get_ident()
    factory_threads: list[int] = []

    def factory(sid: str) -> FakeSandbox:
        factory_threads.append(threading.get_ident())
        return FakeSandbox(sid, tmp_path / sid)

    manager = SessionManager(settings, sandbox_factory=factory)
    await manager.create()

    # Docker provisioning blocks for seconds; it must not run on the loop thread.
    assert factory_threads and factory_threads[0] != loop_thread


def test_docker_provision_failure_removes_the_session_volume():
    class Volumes:
        def __init__(self) -> None:
            self.created: list[str] = []
            self.removed: list[str] = []

        def create(self, name: str):
            self.created.append(name)
            return SimpleNamespace(remove=lambda: self.removed.append(name))

        def get(self, name: str):
            return SimpleNamespace(remove=lambda: self.removed.append(name))

    class Containers:
        @staticmethod
        def run(*args, **kwargs):
            raise RuntimeError("image not found")

    class Client:
        def __init__(self) -> None:
            self.volumes = Volumes()
            self.containers = Containers()

        @staticmethod
        def ping() -> bool:
            return True

    # Egress mode skips network provisioning so the failure stays at container run.
    settings = Settings(sandbox_allow_egress=True)
    client = Client()
    with pytest.raises(RuntimeError, match="image not found"):
        DockerSandbox.create("session-1", settings, docker_client=client)
    assert client.volumes.created == ["dc-session-1"]
    assert client.volumes.removed == ["dc-session-1"]
