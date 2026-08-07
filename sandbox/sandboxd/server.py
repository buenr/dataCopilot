"""Persistent IPython sandbox control service."""

from __future__ import annotations

import asyncio
import io
import json
import os
import socket
import subprocess
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

WORKSPACE = Path(os.environ.get("SANDBOX_WORKSPACE", "/workspace")).resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)


def safe_workspace_path(name: str, *, allow_missing: bool = True) -> Path:
    if not name or "\x00" in name or Path(name).is_absolute():
        raise ValueError("invalid workspace path")
    path = (WORKSPACE / name).resolve()
    try:
        path.relative_to(WORKSPACE)
    except ValueError as exc:
        raise ValueError("path escapes workspace") from exc
    if not allow_missing and not path.exists():
        raise FileNotFoundError(name)
    return path


class Kernel:
    def __init__(self) -> None:
        self.manager: Any = None
        self.client: Any = None
        self.lock = asyncio.Lock()
        self.fallback_globals: dict[str, Any] = {"WORKSPACE": WORKSPACE}

    def start(self) -> None:
        try:
            from jupyter_client import KernelManager
        except ImportError:
            os.chdir(WORKSPACE)
            return
        self.manager = KernelManager()
        # Agent code routinely writes to the current directory. Only the mounted
        # workspace is persisted and downloadable, so the kernel must start there
        # rather than in the image's build directory.
        self.manager.start_kernel(cwd=str(WORKSPACE))
        self.client = self.manager.client()
        self.client.start_channels()
        self.client.wait_for_ready(timeout=30)
        # Make the workspace explicit and available to all agent code.
        self.client.execute(f"from pathlib import Path\nWORKSPACE = Path({str(WORKSPACE)!r})")

    async def stream(self, code: str) -> AsyncIterator[dict[str, Any]]:
        """Execute code and yield IOPub events as the kernel produces them.

        The last event is always ``{"type": "summary", "result": {...}}`` carrying
        the aggregate payload, so a caller can render live output and still end up
        with the same result the buffered ``execute`` returns.
        """
        async with self.lock:
            if not self.client:
                output, errors = io.StringIO(), io.StringIO()
                try:
                    with redirect_stdout(output), redirect_stderr(errors):
                        # Kernel fallback: sandboxd exists to execute agent code.
                        exec(code, self.fallback_globals, self.fallback_globals)  # nosemgrep: python.lang.security.audit.exec-detected.exec-detected
                except Exception as exc:
                    errors.write(f"{type(exc).__name__}: {exc}")
                events = []
                for name, text in (("stdout", output.getvalue()), ("stderr", errors.getvalue())):
                    if text:
                        events.append({"type": name, "data": text})
                        yield events[-1]
                yield {
                    "type": "summary",
                    "result": {
                        "stdout": output.getvalue(),
                        "stderr": errors.getvalue(),
                        "result": None,
                        "events": events,
                    },
                }
                return
            msg_id = self.client.execute(code, allow_stdin=False)
            stdout, stderr, events = [], [], []
            while True:
                message = await asyncio.to_thread(self.client.get_iopub_msg, timeout=120)
                if message["parent_header"].get("msg_id") != msg_id:
                    continue
                content = message["content"]
                kind = message["msg_type"]
                event: dict[str, Any] | None = None
                if kind == "stream":
                    (stdout if content["name"] == "stdout" else stderr).append(content["text"])
                    event = {"type": content["name"], "data": content["text"]}
                elif kind == "error":
                    stderr.append("\n".join(content.get("traceback", [])))
                    event = {"type": "stderr", "data": stderr[-1]}
                elif kind in {"execute_result", "display_data"}:
                    data = content.get("data", {})
                    text = data.get("text/plain", "")
                    if text:
                        event = {"type": "result", "data": text}
                elif kind == "status" and content.get("execution_state") == "idle":
                    break
                if event is not None:
                    events.append(event)
                    yield event
            yield {
                "type": "summary",
                "result": {
                    "stdout": "".join(stdout),
                    "stderr": "".join(stderr),
                    "result": None,
                    "events": events,
                },
            }

    async def execute(self, code: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"stdout": "", "stderr": "", "result": None, "events": []}
        async for event in self.stream(code):
            if event.get("type") == "summary":
                payload = event["result"]
        return payload

    async def interrupt(self) -> bool:
        """Interrupt the running cell. Must not take the lock: the execution that
        needs interrupting is holding it."""
        if self.manager is None:
            return False
        await asyncio.to_thread(self.manager.interrupt_kernel)
        return True

    def close(self) -> None:
        if self.client:
            self.client.stop_channels()
        if self.manager:
            self.manager.shutdown_kernel(now=True)


class ExecRequest(BaseModel):
    code: str = Field(min_length=1)


class ProcessRequest(BaseModel):
    action: str
    command: str | None = None
    port: int | None = None
    pid: int | None = None


kernel = Kernel()
processes: dict[int, subprocess.Popen[Any]] = {}


def wait_for_port(process: subprocess.Popen[Any], port: int, timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"webapp exited before port {port} became ready")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"webapp did not listen on port {port} within {timeout:.0f}s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    kernel.start()
    yield
    kernel.close()
    for process in processes.values():
        process.terminate()


app = FastAPI(title="Data Copilot sandboxd", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "kernel": kernel.client is not None}


@app.post("/exec")
async def execute(request: ExecRequest) -> dict[str, Any]:
    # This is the IPython kernel's execute, not a DB cursor — semgrep matches the name.
    return await kernel.execute(request.code)  # nosemgrep: python.django.security.injection.sql.sql-injection-using-db-cursor-execute.sql-injection-db-cursor-execute


@app.post("/exec/stream")
async def execute_stream(request: ExecRequest) -> StreamingResponse:
    """Newline-delimited JSON so the gateway can forward output while code runs."""

    async def body() -> AsyncIterator[bytes]:
        async for event in kernel.stream(request.code):
            yield (json.dumps(event, default=str) + "\n").encode()

    return StreamingResponse(body(), media_type="application/x-ndjson")


@app.post("/interrupt")
async def interrupt() -> dict[str, Any]:
    return {"interrupted": await kernel.interrupt()}


@app.get("/vars")
async def variables() -> dict[str, Any]:
    # Querying through the kernel keeps state persistent and avoids exposing object data.
    # Modules and callables are noise for the inspector; reprs are truncated so
    # large frames do not flood the response.
    snippet = (
        "import json as _json, types as _types\n"
        "_vars = []\n"
        "for _name in sorted(globals()):\n"
        "    if _name.startswith('_'):\n"
        "        continue\n"
        "    _value = globals()[_name]\n"
        "    if isinstance(_value, _types.ModuleType) or callable(_value):\n"
        "        continue\n"
        "    try:\n"
        "        _rendered = repr(_value)\n"
        "    except Exception:\n"
        "        _rendered = '<unrepresentable>'\n"
        "    _vars.append({'name': _name, 'type': type(_value).__name__, 'value': _rendered[:120]})\n"
        "print(_json.dumps(_vars))\n"
    )
    result = await kernel.execute(snippet)
    try:
        variables = json.loads(result["stdout"].strip())
    except (json.JSONDecodeError, AttributeError):
        variables = []
    return {"vars": variables}


@app.post("/files")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        target = safe_workspace_path(file.filename or "upload")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(await file.read())
    return {"file": target.name, "path": str(target.relative_to(WORKSPACE))}


@app.get("/files")
async def list_files(path: str = ".") -> dict[str, Any]:
    try:
        root = safe_workspace_path(path, allow_missing=False)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    files = [
        {"path": str(item.relative_to(WORKSPACE)), "size": item.stat().st_size, "directory": item.is_dir()}
        for item in sorted(root.rglob("*"))
        if item.is_file()
    ]
    return {"files": files}


@app.delete("/files/{path:path}")
async def remove_file(path: str) -> dict[str, Any]:
    try:
        target = safe_workspace_path(path, allow_missing=False)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    if not target.is_file():
        raise HTTPException(404, "not a file")
    target.unlink()
    return {"deleted": str(target.relative_to(WORKSPACE))}


@app.get("/files/{path:path}")
async def download(path: str) -> FileResponse:
    try:
        target = safe_workspace_path(path, allow_missing=False)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    if not target.is_file():
        raise HTTPException(404, "not a file")
    return FileResponse(target)


@app.get("/procs")
async def list_processes() -> dict[str, Any]:
    return {"processes": [{"pid": process.pid, "port": port, "running": process.poll() is None} for port, process in processes.items()]}


@app.post("/procs")
async def process(request: ProcessRequest) -> dict[str, Any]:
    if request.action == "start":
        if not request.command or not request.port:
            raise HTTPException(400, "command and port are required")
        # Running agent-supplied commands is this service's purpose; Popen itself
        # returns immediately, so the event loop is not blocked here.
        child = subprocess.Popen(request.command, shell=True, cwd=WORKSPACE, start_new_session=True)  # noqa: ASYNC220  # nosemgrep: python.django.security.injection.command.subprocess-injection.subprocess-injection, python.lang.security.audit.subprocess-shell-true.subprocess-shell-true
        processes[request.port] = child
        try:
            await asyncio.to_thread(wait_for_port, child, request.port)
        except RuntimeError as exc:
            processes.pop(request.port, None)
            child.terminate()
            raise HTTPException(502, str(exc)) from exc
        return {"pid": child.pid, "port": request.port}
    if request.action == "stop":
        port = request.port
        child = processes.pop(port, None) if port is not None else None
        if child:
            child.terminate()
        return {"stopped": bool(child), "port": port}
    if request.action == "list":
        return await list_processes()
    raise HTTPException(400, "action must be start, stop, or list")


@app.get("/artifacts")
async def artifacts() -> dict[str, Any]:
    extensions = {".pdf", ".html", ".png", ".jpg", ".jpeg", ".svg"}
    return {
        "artifacts": [
            {"name": path.name, "path": str(path.relative_to(WORKSPACE)), "size": path.stat().st_size}
            for path in WORKSPACE.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        ]
    }
