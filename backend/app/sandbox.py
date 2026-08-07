"""Sandbox abstractions and Docker-backed session containers."""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

# Called with {"type": "stdout"|"stderr"|"result", "data": str} while code runs.
EventCallback = Callable[[dict[str, Any]], None]


@dataclass
class Execution:
    stdout: str = ""
    stderr: str = ""
    result: Any = None
    events: list[dict[str, Any]] = field(default_factory=list)


class Sandbox(Protocol):
    session_id: str
    preview_ports: dict[int, int]
    preview_host: str

    async def exec(self, code: str, on_event: EventCallback | None = None) -> Execution: ...
    async def upload(self, name: str, content: bytes) -> None: ...
    async def read_file(self, name: str) -> bytes: ...
    async def delete_file(self, name: str) -> None: ...
    async def list_files(self, path: str = ".") -> list[dict[str, Any]]: ...
    async def start_webapp(self, command: str, port: int) -> int: ...
    async def stop_webapp(self, port: int) -> None: ...
    async def artifacts(self) -> list[dict[str, Any]]: ...
    async def vars(self) -> list[dict[str, Any]]: ...
    async def interrupt(self) -> bool: ...
    async def close(self) -> None: ...


class FakeSandbox:
    """In-memory, persistent Python sandbox for unit tests and offline demos."""

    def __init__(self, session_id: str | None = None, root: str | Path | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.root = Path(root or Path("sessions") / self.session_id / "workspace").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.preview_ports: dict[int, int] = {}
        self.preview_host = "127.0.0.1"
        self._globals: dict[str, Any] = {"__name__": "__main__", "WORKSPACE": self.root}
        self._processes: dict[int, subprocess.Popen[Any]] = {}

    async def exec(self, code: str, on_event: EventCallback | None = None) -> Execution:
        output, errors = io.StringIO(), io.StringIO()
        result: Any = None
        try:
            with redirect_stdout(output), redirect_stderr(errors):
                compiled = compile(code, "<sandbox>", "exec")
                exec(compiled, self._globals, self._globals)
        except Exception as exc:  # stderr is intentionally returned to agent for retry
            errors.write(f"{type(exc).__name__}: {exc}")
        events: list[dict[str, Any]] = []
        for name, text in (("stdout", output.getvalue()), ("stderr", errors.getvalue())):
            if text:
                events.append({"type": name, "data": text})
        if on_event is not None:
            # This sandbox runs code inline, so output can only be replayed once
            # the cell finishes; the callback contract stays the same as Docker's.
            for event in events:
                on_event(event)
        return Execution(
            stdout=output.getvalue(),
            stderr=errors.getvalue(),
            result=result,
            events=events,
        )

    async def upload(self, name: str, content: bytes) -> None:
        from .paths import safe_path

        target = safe_path(self.root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    async def read_file(self, name: str) -> bytes:
        from .paths import safe_path

        return safe_path(self.root, name, allow_missing=False).read_bytes()

    async def delete_file(self, name: str) -> None:
        from .paths import safe_path

        safe_path(self.root, name, allow_missing=False).unlink()

    async def list_files(self, path: str = ".") -> list[dict[str, Any]]:
        from .paths import safe_path

        directory = safe_path(self.root, path, allow_missing=False)
        if not directory.is_dir():
            return [{"path": str(directory.relative_to(self.root)), "size": directory.stat().st_size}]
        return [
            {"path": str(item.relative_to(self.root)), "size": item.stat().st_size, "directory": item.is_dir()}
            for item in sorted(directory.rglob("*"))
            if item.is_file()
        ]

    async def start_webapp(self, command: str, port: int) -> int:
        process = subprocess.Popen(command, shell=True, cwd=self.root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._processes[port] = process
        self.preview_ports[port] = port
        return port

    async def stop_webapp(self, port: int) -> None:
        process = self._processes.pop(port, None)
        if process:
            process.terminate()
            process.wait(timeout=2)
        self.preview_ports.pop(port, None)

    async def artifacts(self) -> list[dict[str, Any]]:
        return [
            {"name": path.name, "path": str(path.relative_to(self.root)), "size": path.stat().st_size}
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pdf", ".html", ".png", ".jpg", ".jpeg"}
        ]

    async def vars(self) -> list[dict[str, Any]]:
        import types

        variables: list[dict[str, Any]] = []
        for name in sorted(self._globals):
            if name.startswith("_"):
                continue
            value = self._globals[name]
            if isinstance(value, types.ModuleType) or callable(value):
                continue
            try:
                rendered = repr(value)
            except Exception:
                rendered = "<unrepresentable>"
            variables.append({"name": name, "type": type(value).__name__, "value": rendered[:120]})
        return variables

    async def interrupt(self) -> bool:
        # Inline execution blocks the event loop, so there is nothing to interrupt.
        return False

    async def close(self) -> None:
        for port in list(self._processes):
            await self.stop_webapp(port)


class DockerSandbox:
    """Client for one hardened sandboxd container."""

    def __init__(
        self,
        session_id: str,
        container: Any,
        *,
        client: Any,
        preview_ports: dict[int, int],
        preview_host: str,
        volume_name: str = "",
    ):
        self.session_id = session_id
        self.container = container
        self.client = client
        self.preview_ports = preview_ports
        self.preview_host = preview_host
        self.volume_name = volume_name
        self._base_url = f"http://{preview_host}:{preview_ports[7000]}"

    @classmethod
    def create(cls, session_id: str, settings: Any, docker_client: Any | None = None) -> "DockerSandbox":
        try:
            if docker_client is None:
                try:
                    import docker
                except ImportError as exc:
                    raise RuntimeError("Docker SDK is not installed; install the project dependencies") from exc
                client = docker.DockerClient(base_url=settings.docker_socket)
            else:
                client = docker_client
            client.ping()
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "Docker SDK" in str(exc):
                raise
            raise RuntimeError(
                "Docker is unavailable. Start Docker Desktop/daemon and retry; "
                "Data Copilot will not fall back to an unsandboxed local process."
            ) from exc
        volume_name = f"{settings.session_volume_prefix}-{session_id}"
        client.volumes.create(name=volume_name)
        # Publish ephemeral loopback ports for the gateway. The custom bridge
        # keeps masquerading disabled, so publishing does not grant the
        # sandbox general outbound NAT access.
        ports = {7000: 0, 8501: 0, 8502: 0, 8000: 0, 8080: 0}
        network_name = settings.sandbox_network if not settings.sandbox_allow_egress else None
        if network_name:
            try:
                client.networks.get(network_name)
            except Exception:
                client.networks.create(
                    network_name,
                    driver="bridge",
                    options={"com.docker.network.bridge.enable_ip_masquerade": "false"},
                )
        container = None
        try:
            container = client.containers.run(
                settings.sandbox_image,
                detach=True,
                name=f"dc-sandbox-{session_id}",
                volumes={volume_name: {"bind": "/workspace", "mode": "rw"}},
                ports={f"{port}/tcp": ("127.0.0.1", 0) for port in ports},
                nano_cpus=2_000_000_000,
                mem_limit="4g",
                pids_limit=512,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                tmpfs={"/tmp": "rw,noexec,nosuid,size=512m"},
                network=network_name,
                environment={"SANDBOX_ALLOW_EGRESS": str(settings.sandbox_allow_egress).lower()},
            )
            container.reload()
            bindings = container.attrs.get("NetworkSettings", {}).get("Ports", {})
            mapped = {}
            for port in ports:
                values = bindings.get(f"{port}/tcp") or []
                mapped[port] = int(values[0]["HostPort"]) if values else port
            sandbox = cls(
                session_id,
                container,
                client=client,
                preview_ports=mapped,
                preview_host="127.0.0.1",
                volume_name=volume_name,
            )
            sandbox.wait_ready()
        except Exception:
            # A partial provision must not leak the container or the
            # per-session volume (previously only wait_ready failures cleaned up).
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
            try:
                client.volumes.get(volume_name).remove()
            except Exception:
                pass
            raise
        return sandbox

    def wait_ready(self, timeout: float = 90) -> None:
        """Wait until sandboxd has completed its IPython kernel startup."""
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                response = httpx.get(f"{self._base_url}/health", timeout=2)
                if response.is_success:
                    payload = response.json()
                    if payload.get("kernel", True):
                        return
                    last_error = RuntimeError("sandboxd is serving but its kernel is not ready")
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
            time.sleep(0.5)
        raise RuntimeError(f"Sandbox did not become ready within {timeout:.0f}s: {last_error}")

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.request(method, self._base_url + path, **kwargs)
            response.raise_for_status()
            return response.json()

    async def exec(self, code: str, on_event: EventCallback | None = None) -> Execution:
        if on_event is None:
            payload = await self._request("POST", "/exec", json={"code": code})
            return Execution(
                stdout=payload.get("stdout", ""),
                stderr=payload.get("stderr", ""),
                result=payload.get("result"),
                events=payload.get("events", []),
            )
        stdout: list[str] = []
        stderr: list[str] = []
        events: list[dict[str, Any]] = []
        summary: dict[str, Any] = {}
        # No read timeout: output arrives as the cell runs, and a long silent
        # computation is bounded by sandboxd's own per-message kernel timeout.
        timeout = httpx.Timeout(None, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", self._base_url + "/exec/stream", json={"code": code}) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "summary":
                        summary = event.get("result", {})
                        continue
                    events.append(event)
                    data = event.get("data")
                    if isinstance(data, str) and data:
                        kind = event.get("type")
                        if kind == "stderr":
                            stderr.append(data)
                        elif kind == "stdout":
                            stdout.append(data)
                        # `result` reprs are worth showing live but are not stdout,
                        # matching how the buffered endpoint reports them.
                        on_event(event)
        return Execution(
            stdout=summary.get("stdout", "".join(stdout)),
            stderr=summary.get("stderr", "".join(stderr)),
            result=summary.get("result"),
            events=summary.get("events", events),
        )

    async def upload(self, name: str, content: bytes) -> None:
        await self._request("POST", "/files", files={"file": (name, content)})

    async def read_file(self, name: str) -> bytes:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(self._base_url + "/files/" + name)
            response.raise_for_status()
            return response.content

    async def delete_file(self, name: str) -> None:
        try:
            await self._request("DELETE", "/files/" + name)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise FileNotFoundError(name) from exc
            raise

    async def list_files(self, path: str = ".") -> list[dict[str, Any]]:
        result = await self._request("GET", "/files", params={"path": path})
        return result.get("files", result)

    async def start_webapp(self, command: str, port: int) -> int:
        await self._request("POST", "/procs", json={"action": "start", "command": command, "port": port})
        # Keep preview_ports as the logical-to-host mapping established at
        # container creation. The process inside the sandbox still listens on
        # the logical port, while the gateway reaches its ephemeral host port.
        return port

    async def stop_webapp(self, port: int) -> None:
        await self._request("POST", "/procs", json={"action": "stop", "port": port})

    async def artifacts(self) -> list[dict[str, Any]]:
        result = await self._request("GET", "/artifacts")
        return result.get("artifacts", result)

    async def vars(self) -> list[dict[str, Any]]:
        result = await self._request("GET", "/vars")
        return result.get("vars", [])

    async def interrupt(self) -> bool:
        result = await self._request("POST", "/interrupt")
        return bool(result.get("interrupted"))

    async def close(self) -> None:
        try:
            self.container.remove(force=True)
            self.client.volumes.get(self.volume_name).remove()
        except Exception:
            # Teardown should be best-effort after an already-dead container.
            pass


@dataclass
class ManagedSession:
    id: str
    sandbox: Sandbox
    created_at: str
    last_activity: float
    status: str = "ready"
    dataset_profiles: list[dict[str, Any]] = field(default_factory=list)
    # The transcript belongs to the session, not to a websocket: a page reload or
    # dropped connection creates a new Agent that must keep the conversation.
    messages: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)


class SessionManager:
    def __init__(self, settings: Any, sandbox_factory: Any | None = None):
        self.settings = settings
        self.sandbox_factory = sandbox_factory or (lambda sid: DockerSandbox.create(sid, settings))
        self.sessions: dict[str, ManagedSession] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> ManagedSession:
        from datetime import datetime, timezone
        sid = str(uuid.uuid4())
        # Provisioning blocks on the Docker SDK (container start plus kernel
        # health polling can take many seconds); keep it off the event loop so
        # one session creation cannot stall every other session.
        sandbox = await asyncio.to_thread(self.sandbox_factory, sid)
        if asyncio.iscoroutine(sandbox):
            sandbox = await sandbox
        session = ManagedSession(sid, sandbox, datetime.now(timezone.utc).isoformat(), asyncio.get_running_loop().time())
        async with self._lock:
            self.sessions[sid] = session
        return session

    def get(self, session_id: str) -> ManagedSession:
        try:
            session = self.sessions[session_id]
        except KeyError as exc:
            raise KeyError("session not found") from exc
        session.last_activity = asyncio.get_running_loop().time()
        return session

    def touch(self, session_id: str) -> None:
        """Refresh idle timing for long-lived channels (e.g. the chat websocket)
        that hold a session reference without going through get()."""
        session = self.sessions.get(session_id)
        if session is not None:
            session.last_activity = asyncio.get_running_loop().time()

    async def delete(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session:
            await session.sandbox.close()

    async def reap(self) -> int:
        now = asyncio.get_running_loop().time()
        ttl = self.settings.session_ttl_minutes * 60
        expired = [sid for sid, item in self.sessions.items() if now - item.last_activity >= ttl]
        for sid in expired:
            await self.delete(sid)
        return len(expired)
