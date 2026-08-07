"""FastAPI gateway for sessions, uploads, agent events, and previews."""

from __future__ import annotations

import asyncio
import mimetypes
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .agent import Agent, AnthropicProvider, MockProvider, OpenAIProvider
from .config import Settings, get_settings
from .profiling import profile_files
from .proxy import parse_preview_path, preview_target
from .sandbox import SessionManager


class Message(BaseModel):
    type: str = "user_message"
    content: str = Field(min_length=1)


async def _preview_file_fallback(session: Any, path: str) -> Response | None:
    """Serve workspace files when a static server points at a wrong directory."""
    requested = path.strip("/")
    candidates = [requested] if requested else []
    if not candidates:
        try:
            files = await session.sandbox.list_files(".")
        except Exception:
            return None
        html_files = [
            str(item.get("path", ""))
            for item in files
            if isinstance(item, dict)
            and not item.get("directory")
            and str(item.get("path", "")).lower().endswith((".html", ".htm", ".xhtml"))
        ]
        candidates = sorted(
            html_files,
            key=lambda item: (
                0 if PurePosixPath(item).name == "dashboard.html" else
                1 if PurePosixPath(item).name == "index.html" else 2,
                item,
            ),
        )

    for candidate in candidates:
        if not candidate or candidate.startswith("../") or "/../" in candidate:
            continue
        try:
            content = await session.sandbox.read_file(candidate)
        except Exception:
            continue
        media_type = mimetypes.guess_type(candidate)[0] or "application/octet-stream"
        return Response(content, media_type=media_type)
    return None


_LOOPBACK_ALIASES = {"localhost": "127.0.0.1", "127.0.0.1": "localhost"}


def cors_origins(settings: Settings) -> list[str]:
    """Resolve the browser origins allowed to call the gateway."""
    origins = [item.strip() for item in settings.frontend_origin.split(",") if item.strip()]
    if settings.app_env.lower() == "development":
        # Browsers treat http://localhost:5173 and http://127.0.0.1:5173 as different
        # origins, so a single configured dev origin has to cover both spellings.
        for origin in list(origins):
            parts = urlsplit(origin)
            alias = _LOOPBACK_ALIASES.get(parts.hostname or "")
            if alias is None:
                continue
            netloc = alias if parts.port is None else f"{alias}:{parts.port}"
            candidate = urlunsplit((parts.scheme, netloc, parts.path, "", ""))
            if candidate not in origins:
                origins.append(candidate)
    return origins


def make_provider(settings: Settings) -> Any:
    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
        return AnthropicProvider(settings.anthropic_api_key, settings.anthropic_model)
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
        return OpenAIProvider(settings.openai_api_key, settings.openai_model)
    return MockProvider()


async def reap_loop(manager: SessionManager, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=60)
        except asyncio.TimeoutError:
            await manager.reap()


def create_app(settings: Settings | None = None, manager: SessionManager | None = None) -> FastAPI:
    settings = settings or get_settings()
    session_manager = manager or SessionManager(settings)
    stop = asyncio.Event()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(reap_loop(session_manager, stop))
        yield
        stop.set()
        await task
        for session_id in list(session_manager.sessions):
            await session_manager.delete(session_id)

    application = FastAPI(title="Data Copilot", version="0.1.0", lifespan=lifespan)
    application.state.session_manager = session_manager
    application.state.settings = settings
    origins = cors_origins(settings)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/api/sessions")
    async def create_session() -> dict[str, Any]:
        try:
            session = await session_manager.create()
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"id": session.id, "status": session.status, "created_at": session.created_at}

    @application.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        try:
            session = session_manager.get(session_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {
            "id": session.id,
            "status": session.status,
            "created_at": session.created_at,
            "preview_ports": session.sandbox.preview_ports,
            "preview_host": session.sandbox.preview_host,
        }

    @application.post("/api/sessions/{session_id}/files")
    async def upload_files(session_id: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
        try:
            session = session_manager.get(session_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        uploaded: list[str] = []
        for item in files:
            name = (item.filename or "upload").replace("\\", "/").split("/")[-1]
            if not name or name in {".", ".."}:
                raise HTTPException(400, "invalid filename")
            await session.sandbox.upload(f"data/{name}", await item.read())
            uploaded.append(f"data/{name}")
        schemas: list[dict[str, Any]] = []
        root = getattr(session.sandbox, "root", None)
        if root:
            try:
                # Re-profile the complete data folder so multiple uploads stay
                # available to later turns as df_1, df_2, ... .
                schemas = profile_files(root / "data")
            except (ValueError, ImportError, OSError) as exc:
                raise HTTPException(400, f"could not profile upload: {exc}") from exc
            bootstrap = (
                "from pathlib import Path\n"
                "import pandas as pd\n"
                "data_root = Path(WORKSPACE) / 'data'\n"
                "for key in list(globals()):\n"
                "    if key.startswith('df_'):\n"
                "        del globals()[key]\n"
                "for i, path in enumerate(sorted(data_root.iterdir()), 1):\n"
                "    suffix = path.suffix.lower()\n"
                "    if suffix == '.csv':\n"
                "        frame = pd.read_csv(path)\n"
                "    elif suffix in ('.xls', '.xlsx'):\n"
                "        frame = pd.read_excel(path)\n"
                "    elif suffix == '.parquet':\n"
                "        frame = pd.read_parquet(path)\n"
                "    elif suffix == '.json':\n"
                "        frame = pd.read_json(path)\n"
                "    else:\n"
                "        continue\n"
                "    globals()[f'df_{i}'] = frame\n"
            )
            result = await session.sandbox.exec(bootstrap)
            if result.stderr:
                raise HTTPException(400, f"could not load upload into sandbox: {result.stderr[-1000:]}")
        else:
            # Docker sandboxes own their volume, so profiling runs in the same
            # persistent kernel rather than copying tenant data to the gateway.
            code = (
                "import json\n"
                "from pathlib import Path\n"
                "import pandas as pd\n"
                "data_root = Path(WORKSPACE) / 'data'\n"
                "for key in list(globals()):\n"
                "    if key.startswith('df_'):\n"
                "        del globals()[key]\n"
                "out = []\n"
                "for i, path in enumerate(sorted(data_root.iterdir()), 1):\n"
                "    suffix = path.suffix.lower()\n"
                "    if suffix == '.csv':\n"
                "        frame = pd.read_csv(path)\n"
                "    elif suffix in ('.xls', '.xlsx'):\n"
                "        frame = pd.read_excel(path)\n"
                "    elif suffix == '.parquet':\n"
                "        frame = pd.read_parquet(path)\n"
                "    elif suffix == '.json':\n"
                "        frame = pd.read_json(path)\n"
                "    else:\n"
                "        continue\n"
                "    globals()[f'df_{i}'] = frame\n"
                "    numeric = frame.select_dtypes(include='number')\n"
                "    out.append({'name': f'df_{i}', 'file': path.name, 'rows': int(frame.shape[0]), 'columns': int(frame.shape[1]), 'dtypes': {str(k): str(v) for k, v in frame.dtypes.items()}, 'null_percentages': {str(k): float(frame[k].isna().mean()*100) for k in frame.columns}, 'numeric_stats': {str(k): {'min': float(v.min()) if not v.dropna().empty else None, 'max': float(v.max()) if not v.dropna().empty else None, 'mean': float(v.mean()) if not v.dropna().empty else None, 'median': float(v.median()) if not v.dropna().empty else None} for k, v in numeric.items()}, 'sample_rows': frame.head(5).to_dict(orient='records')})\n"
                "print(json.dumps(out, default=str))\n"
            )
            result = await session.sandbox.exec(code)
            if result.stderr:
                raise HTTPException(400, f"could not profile upload: {result.stderr[-1000:]}")
            try:
                import json

                schemas = json.loads(result.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError) as exc:
                raise HTTPException(400, "sandbox returned an invalid profile") from exc
        session.dataset_profiles = schemas
        return {"files": uploaded, "schemas": schemas}

    @application.get("/api/sessions/{session_id}/artifacts/{name:path}")
    async def download_artifact(session_id: str, name: str) -> Response:
        try:
            session = session_manager.get(session_id)
            content = await session.sandbox.read_file(name)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (FileNotFoundError, ValueError, httpx.HTTPError) as exc:
            raise HTTPException(404, "artifact not found") from exc
        media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        return Response(content=content, media_type=media_type)

    @application.delete("/api/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str) -> Response:
        if session_id not in session_manager.sessions:
            raise HTTPException(404, "session not found")
        await session_manager.delete(session_id)
        return Response(status_code=204)

    @application.api_route("/api/sessions/{session_id}/preview/{port}/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def preview(request: Request, session_id: str, port: int, path: str = "") -> Response:
        try:
            session = session_manager.get(session_id)
            mapped_port = session.sandbox.preview_ports.get(port, port)
            target = preview_target(
                mapped_port,
                path,
                request.url.query,
                host=session.sandbox.preview_host,
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(404, "preview unavailable") from exc
        body = await request.body()
        headers = {key: value for key, value in request.headers.items() if key.lower() in {"accept", "content-type", "range", "user-agent"}}
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                upstream = await client.request(request.method, target, content=body, headers=headers)
            except httpx.HTTPError as exc:
                raise HTTPException(502, f"preview upstream unavailable: {exc}") from exc
        upstream_is_directory_listing = (
            upstream.status_code == 200
            and not path.strip("/")
            and "text/html" in upstream.headers.get("content-type", "")
            and b"Directory listing for" in upstream.content
        )
        if request.method == "GET" and (
            upstream.status_code == 404 or upstream_is_directory_listing
        ):
            fallback = await _preview_file_fallback(session, path)
            if fallback is not None:
                return fallback
        response_headers = {key: value for key, value in upstream.headers.items() if key.lower() not in {"content-length", "transfer-encoding", "connection"}}
        return Response(upstream.content, status_code=upstream.status_code, headers=response_headers, media_type=upstream.headers.get("content-type"))

    @application.websocket("/api/sessions/{session_id}/preview/{port}/{path:path}")
    async def preview_websocket(websocket: WebSocket, session_id: str, port: int, path: str = "") -> None:
        """Relay app websocket traffic (for example Streamlit's _stcore stream)."""
        try:
            session = session_manager.get(session_id)
            mapped = session.sandbox.preview_ports.get(port, port)
            target = preview_target(
                mapped,
                path,
                websocket.url.query,
                host=session.sandbox.preview_host,
            ).replace("http://", "ws://", 1)
            import websockets
        except (KeyError, ValueError, ImportError):
            await websocket.close(code=4404)
            return
        await websocket.accept()
        try:
            async with websockets.connect(target) as upstream:
                async def client_to_upstream() -> None:
                    while True:
                        message = await websocket.receive()
                        if message.get("text") is not None:
                            await upstream.send(message["text"])
                        elif message.get("bytes") is not None:
                            await upstream.send(message["bytes"])
                        else:
                            break

                async def upstream_to_client() -> None:
                    async for message in upstream:
                        if isinstance(message, bytes):
                            await websocket.send_bytes(message)
                        else:
                            await websocket.send_text(message)

                await asyncio.gather(client_to_upstream(), upstream_to_client())
        except Exception:
            await websocket.close(code=1011)

    @application.websocket("/ws/sessions/{session_id}")
    async def session_socket(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        try:
            session = session_manager.get(session_id)
        except KeyError:
            await websocket.send_json({"type": "error", "message": "session not found"})
            await websocket.close(code=4404)
            return
        await websocket.send_json({"type": "session_ready", "session_id": session_id})
        agent = Agent(
            session.sandbox,
            make_provider(settings),
            session_id,
            settings.sessions_dir,
            dataset_context=lambda: session.dataset_profiles,
        )
        try:
            while True:
                payload = await websocket.receive_json()
                if payload.get("type") != "user_message" or not isinstance(payload.get("content"), str):
                    await websocket.send_json({"type": "error", "message": "expected a user_message"})
                    continue
                await websocket.send_json({"type": "user_message", "content": payload["content"]})
                async for event in agent.turn(payload["content"]):
                    await websocket.send_json(event)
        except WebSocketDisconnect:
            return

    return application


app = create_app()
