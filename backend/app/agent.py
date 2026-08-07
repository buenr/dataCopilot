"""Provider-neutral agent loop and sandbox tool dispatch."""

from __future__ import annotations

import asyncio
import html
import json
import re
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Protocol

from .sandbox import EventCallback, Execution, Sandbox

TOOLS = {"run_python", "write_file", "read_file", "list_files", "start_webapp", "stop_webapp", "register_artifact"}
PROVIDER_WORKSPACE = "/home/oai/share"


def _workspace_relative(path: Any) -> str:
    """Models sometimes pass kernel-absolute paths; the sandbox file API is workspace-relative."""
    text = str(path)
    if text.startswith("/workspace/"):
        return text[len("/workspace/"):]
    return text
# The kernel preloads WORKSPACE for convenience, but agent code can reassign it,
# so this snippet resolves the real workspace from the sandbox environment. A
# wrong destination would silently copy a rescued PDF back onto itself.
PDF_RESCUE_CODE = """import os
from pathlib import Path
workspace = Path(os.environ.get("SANDBOX_WORKSPACE", "/workspace")).resolve()
candidates = []
for source_root in (Path.cwd(), Path("/app"), Path("/mnt/data"), Path("/tmp")):
    try:
        candidates.extend(source_root.glob("*.pdf"))
    except OSError:
        pass
valid = []
for source in candidates:
    try:
        source = source.resolve()
        if source.is_relative_to(workspace):
            continue
        data = source.read_bytes()
        if data.startswith(b"%PDF-"):
            valid.append((source.stat().st_mtime, source, data))
    except OSError:
        pass
if valid:
    _, source, data = max(valid, key=lambda item: item[0])
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / source.name).write_bytes(data)
"""
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run Python in the persistent analysis sandbox. Use this for data analysis and calculations.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a generated artifact or source file inside the sandbox workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the sandbox workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the sandbox workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_webapp",
            "description": "Start a generated web application in the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "port": {"type": "integer"},
                },
                "required": ["command", "port"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_webapp",
            "description": "Stop a sandbox web application by port.",
            "parameters": {
                "type": "object",
                "properties": {"port": {"type": "integer"}},
                "required": ["port"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "register_artifact",
            "description": "Register a generated HTML, PDF, image, or other artifact for the canvas. "
            "For a webapp, register the served HTML file (for example dashboard.html or site/index.html).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["webapp", "pdf", "document", "image"]},
                    "port": {"type": "integer"},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
]


def system_prompt(dataset_profiles: list[dict[str, Any]]) -> str:
    prompt = """You are Data Copilot, a data analysis agent.

Uploaded datasets are authoritative. Do not invent facts or produce a generic
business summary when data is available. For analytical requests, use the
persistent Python sandbox and the preloaded dataframe variables (df_1, df_2,
...) to calculate the requested results. Cite concrete values, comparisons,
row counts, and the relevant dataset or columns in your response. Separate
computed facts from reasonable interpretation. If the request asks for an
executive summary, first inspect or calculate from the data, then summarize
the actual findings for an executive audience.

For dashboard or web-app requests, the task is not complete until a runnable
app is started inside the sandbox with start_webapp and a renderable artifact
is registered with register_artifact. Use port 8501 unless there is a good
reason not to. When data is uploaded, render concrete dataset-derived values
in the initial page HTML, including at least one KPI and one grouped or
comparative insight. Do not rely only on a JavaScript data constant, blank
placeholders, or values that appear only after a client-side failure. Read the
generated HTML before handoff and verify that the visible page contains real
values from the uploaded data. Do not claim completion after only writing
source code.

For PDF or report requests, write the final PDF inside the sandbox workspace
using Path(WORKSPACE) (for example, Path(WORKSPACE) / "report.pdf"), then call
register_artifact with the PDF filename and type "pdf". WORKSPACE is already
defined for you; never reassign it, and never substitute os.getcwd() for it.
Only files inside the workspace can be previewed or downloaded, so do not save
the only copy under /mnt/data or /tmp, and do not claim completion without
registering the PDF artifact. Keep exploration to one or two steps, then write
and register the PDF right away so a step limit cannot cut the report off.

For chart or visualization requests, compute the chart from the uploaded data
and save it as a PNG or SVG inside the workspace (for example,
Path(WORKSPACE) / "chart.png"), then call register_artifact with the filename
and type "image" so it renders on the canvas.
"""
    if not dataset_profiles:
        return prompt + "\nNo dataset has been uploaded yet."
    return (
        prompt
        + "\nUploaded dataset profiles are included below. They are a compact "
        "context; use run_python for calculations that are not fully represented.\n"
        + "DATASET_PROFILES_JSON:\n"
        + json.dumps(dataset_profiles, default=str)
    )


def _http_server_directory(command: str) -> str | None:
    if "http.server" not in command:
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    for index, token in enumerate(tokens):
        if token == "--directory" and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith("--directory="):
            return token.split("=", 1)[1]
    return None


def mock_executive_summary(messages: list[dict[str, Any]]) -> str:
    """Produce a grounded offline response when the mock provider is enabled."""
    profiles: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "system":
            continue
        marker = "DATASET_PROFILES_JSON:\n"
        content = str(message.get("content", ""))
        if marker in content:
            try:
                profiles = json.loads(content.split(marker, 1)[1])
            except (TypeError, ValueError):
                profiles = []
    if not profiles:
        return "No uploaded dataset is available to summarize yet."
    lines = ["Executive summary grounded in the uploaded data:"]
    for profile in profiles:
        lines.append(
            f"- {profile.get('file', profile.get('name', 'dataset'))}: "
            f"{profile.get('rows', 0):,} rows and {profile.get('columns', 0)} columns."
        )
        for column, stats in (profile.get("numeric_stats") or {}).items():
            if stats.get("mean") is not None:
                lines.append(
                    f"  - {column}: average {stats['mean']:.2f}, "
                    f"range {stats.get('min')} to {stats.get('max')}."
                )
    return "\n".join(lines)


class LLMProvider(Protocol):
    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AsyncIterator[str | dict[str, Any]]: ...


class MockProvider:
    """Deterministic offline provider used by the POC and smoke tests."""

    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AsyncIterator[str | dict[str, Any]]:
        last = messages[-1]
        if last.get("role") == "tool":
            # Tool results just came back; wrap up instead of pattern-matching
            # the result payload (which mentions e.g. "dashboard.html") into
            # another round of identical tool calls until the turn cap.
            yield "Done. The requested output is ready in the canvas."
            return
        prompt = str(last.get("content") or "").lower()
        if "dashboard" in prompt or "web app" in prompt or "webapp" in prompt:
            yield {"tool": "run_python", "arguments": {"code": dashboard_code()}}
            yield {
                "tool": "start_webapp",
                "arguments": {
                    "command": "python -m http.server 8501 --bind 0.0.0.0",
                    "port": 8501,
                },
            }
            yield {
                "tool": "register_artifact",
                "arguments": {"name": "dashboard.html", "port": 8501, "type": "webapp"},
            }
        elif "pdf" in prompt or "report" in prompt or "whitepaper" in prompt:
            yield {"tool": "run_python", "arguments": {"code": pdf_code()}}
            yield {"tool": "register_artifact", "arguments": {"name": "report.pdf", "type": "pdf"}}
        elif any(term in prompt for term in ("chart", "graph", "plot", "visuali")):
            yield {"tool": "run_python", "arguments": {"code": chart_code()}}
            yield {"tool": "register_artifact", "arguments": {"name": "chart.svg", "type": "image"}}
        elif any(term in prompt for term in ("executive summary", "summary", "insight", "analy", "explore")):
            yield mock_executive_summary(messages)
        else:
            yield "I can analyze the uploaded data, build a dashboard, or write a PDF report."


def dashboard_code() -> str:
    # The canvas quality gate only publishes dashboards whose visible text
    # carries concrete dataset values, so the mock computes real metrics from
    # df_1 the way a competent model would.
    return """from pathlib import Path
workspace = Path(WORKSPACE)
frame = globals().get("df_1")
rows = len(frame) if frame is not None else 0
cards = ""
if frame is not None:
    for name in frame.select_dtypes(include="number").columns[:4]:
        value = frame[name].mean()
        if value == value:  # skip NaN averages
            cards += f'<p class="metric">{name}: {value:,.2f}</p>'
html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Data Copilot Dashboard</title>
<style>body{{font-family:system-ui;margin:2rem;background:#f8fafc}}main{{max-width:900px;margin:auto}}
.card{{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:1.5rem;box-shadow:0 2px 8px #0001}}
h1{{color:#0f172a}}.metric{{font-size:2rem;font-weight:700;color:#2563eb}}</style></head>
<body><main><div class="card"><h1>Data Copilot Dashboard</h1>
<p class="metric">{rows:,} rows analyzed</p>{cards}
<p>Interactive dashboard generated in the session sandbox.</p>
</div></main></body></html>'''
(workspace / "dashboard.html").write_text(html, encoding="utf-8")
(workspace / "index.html").write_text(html, encoding="utf-8")
"""


def pdf_code() -> str:
    # A dependency-free, valid two-page PDF keeps the offline mock usable.
    return r"""from pathlib import Path
workspace = Path(WORKSPACE)
def stream(text):
    body = ("BT /F1 24 Tf 72 700 Td (" + text + ") Tj ET\n").encode()
    return b"<< /Length " + str(len(body)).encode() + b" >>\nstream\n" + body + b"endstream"
objects = [
 b"<< /Type /Catalog /Pages 2 0 R >>",
 b"<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>",
 b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 7 0 R >> >> >>",
 stream("Data Copilot Report - Executive summary"),
 b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R /Resources << /Font << /F1 7 0 R >> >> >>",
 stream("Detailed Findings - Generated analysis"),
 b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
]
pdf = bytearray(b"%PDF-1.4\n")
offsets = [0]
for i, obj in enumerate(objects, 1):
    offsets.append(len(pdf)); pdf.extend(f"{i} 0 obj\n".encode()); pdf.extend(obj); pdf.extend(b"\nendobj\n")
xref = len(pdf); pdf.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
for offset in offsets[1:]: pdf.extend(f"{offset:010d} 00000 n \n".encode())
pdf.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
(workspace / "report.pdf").write_bytes(pdf)
"""


def chart_code() -> str:
    # Dependency-free SVG bar chart so the offline mock needs no matplotlib.
    return r"""from pathlib import Path
import html
workspace = Path(WORKSPACE)
frame = globals().get("df_1")
means = []
if frame is not None:
    numeric = frame.select_dtypes(include="number")
    for name, value in numeric.mean().head(8).items():
        value = float(value)
        if value == value:  # drop NaN averages
            means.append((str(name), value))
top = max((abs(value) for _, value in means), default=0.0) or 1.0
bars = [(name, value, max(4, round(260 * abs(value) / top))) for name, value in means]
if not bars:
    bars = [("no numeric columns", 0.0, 4)]
width = 80 + 110 * len(bars)
parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="380" font-family="system-ui, sans-serif">',
    '<rect width="100%" height="100%" fill="#ffffff"/>',
    '<text x="24" y="36" font-size="18" font-weight="700" fill="#0f172a">Column averages</text>',
]
for i, (name, value, height) in enumerate(bars):
    x = 40 + 110 * i
    y = 320 - height
    label = html.escape(name if len(name) <= 12 else name[:11] + "...")
    parts.append(f'<rect x="{x}" y="{y}" width="70" height="{height}" rx="6" fill="#2563eb"/>')
    parts.append(f'<text x="{x + 35}" y="{y - 8}" font-size="12" text-anchor="middle" fill="#334155">{value:,.1f}</text>')
    parts.append(f'<text x="{x + 35}" y="342" font-size="11" text-anchor="middle" fill="#64748b">{label}</text>')
parts.append("</svg>")
(workspace / "chart.svg").write_text("\n".join(parts), encoding="utf-8")
"""


class AnthropicProvider:
    def __init__(self, api_key: str, model: str):
        self.api_key, self.model = api_key, model

    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AsyncIterator[str | dict[str, Any]]:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise RuntimeError("Anthropic provider requires the optional providers dependency") from exc
        client = AsyncAnthropic(api_key=self.api_key)
        system = next((item["content"] for item in messages if item.get("role") == "system"), None)
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": self._convert_messages(messages),
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [
                {
                    "name": item["function"]["name"],
                    "description": item["function"].get("description", ""),
                    "input_schema": item["function"]["parameters"],
                }
                for item in tools
            ]
        async with client.messages.stream(**request) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
        for block in final.content:
            if getattr(block, "type", None) == "tool_use":
                yield {
                    "tool": getattr(block, "name", ""),
                    "arguments": block.input if isinstance(block.input, dict) else {},
                    "call_id": getattr(block, "id", ""),
                }

    @staticmethod
    def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate the OpenAI-shaped running history into Anthropic content blocks.

        Assistant tool calls become tool_use blocks; tool results become
        tool_result blocks inside a user message, with consecutive results
        merged so the conversation keeps the user/assistant alternation the
        API expects.
        """
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            if role == "assistant" and message.get("tool_calls"):
                blocks: list[dict[str, Any]] = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": message["content"]})
                for call in message["tool_calls"]:
                    function = call["function"]
                    try:
                        inputs = json.loads(function.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        inputs = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call["id"],
                            "name": function["name"],
                            "input": inputs,
                        }
                    )
                converted.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message["tool_call_id"],
                    "content": message.get("content", ""),
                }
                previous = converted[-1] if converted else None
                if previous and previous["role"] == "user" and isinstance(previous["content"], list):
                    previous["content"].append(block)
                else:
                    converted.append({"role": "user", "content": [block]})
            elif role in {"user", "assistant"}:
                content = message.get("content") or ""
                if content:
                    converted.append({"role": role, "content": content})
        return converted


class OpenAIProvider:
    def __init__(self, api_key: str, model: str):
        self.api_key, self.model = api_key, model

    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AsyncIterator[str | dict[str, Any]]:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI provider requires the optional providers dependency") from exc
        client = AsyncOpenAI(api_key=self.api_key)
        if self.model.lower().startswith(("gpt-5", "o")):
            async for item in self._stream_responses(client, messages, tools):
                yield item
            return
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        response = await client.chat.completions.create(**request)
        tool_calls: dict[int, dict[str, str]] = {}
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = delta.content
            if text:
                yield text
            for call in delta.tool_calls or []:
                index = int(call.index or 0)
                item = tool_calls.setdefault(
                    index,
                    {"call_id": call.id or f"call-{index}", "name": "", "arguments": ""},
                )
                if call.id:
                    item["call_id"] = call.id
                if call.function and call.function.name:
                    item["name"] += call.function.name
                if call.function and call.function.arguments:
                    item["arguments"] += call.function.arguments
        for item in tool_calls.values():
            try:
                arguments = json.loads(item["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            yield {
                "tool": item["name"],
                "arguments": arguments,
                "call_id": item["call_id"],
            }

    async def _stream_responses(
        self,
        client: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[str | dict[str, Any]]:
        system = next((item["content"] for item in messages if item.get("role") == "system"), None)
        input_items: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            if role == "assistant" and message.get("tool_calls"):
                for call in message["tool_calls"]:
                    function = call["function"]
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": call["id"],
                            "name": function["name"],
                            "arguments": function["arguments"],
                        }
                    )
                if message.get("content"):
                    input_items.append({"role": "assistant", "content": message["content"]})
            elif role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message["tool_call_id"],
                        "output": message["content"],
                    }
                )
            else:
                input_items.append({"role": role, "content": message.get("content", "")})

        response_tools = [
            {
                "type": "function",
                "name": item["function"]["name"],
                "description": item["function"].get("description", ""),
                "parameters": item["function"]["parameters"],
            }
            for item in tools
        ]
        request: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "stream": True,
            # Always run reasoning models at maximum effort.
            "reasoning": {"effort": "max"},
        }
        if system:
            request["instructions"] = system
        if response_tools:
            request["tools"] = response_tools
            request["tool_choice"] = "auto"
        response = await client.responses.create(**request)
        async for event in response:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                yield getattr(event, "delta", "")
            elif event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) == "function_call":
                    try:
                        arguments = json.loads(getattr(item, "arguments", "") or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    yield {
                        "tool": getattr(item, "name", ""),
                        "arguments": arguments,
                        "call_id": getattr(item, "call_id", ""),
                    }


@dataclass
class ToolDispatcher:
    sandbox: Sandbox
    session_id: str
    trajectory_root: Path = Path("sessions")

    def __post_init__(self) -> None:
        self.trajectory_path = self.trajectory_root / self.session_id / "trajectory.jsonl"
        self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: dict[str, Any]) -> None:
        event = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        with self.trajectory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")

    @staticmethod
    def _normalize_webapp_command(command: str) -> str:
        """Keep static servers inside the sandbox workspace.

        Providers sometimes use paths from their own runtime, such as
        ``/home/oai/share``. The sandbox process already starts in its
        workspace, so those paths make an otherwise valid dashboard return
        404. Other absolute paths, such as the sandbox's own ``/workspace``,
        must remain unchanged.
        """
        directory = _http_server_directory(command)
        provider_prefix = f"{PROVIDER_WORKSPACE}/"
        if directory is None or (
            directory != PROVIDER_WORKSPACE and not directory.startswith(provider_prefix)
        ):
            return command
        try:
            tokens = shlex.split(command)
        except ValueError:
            return command

        replacement = directory[len(PROVIDER_WORKSPACE):].lstrip("/")
        normalized: list[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == "--directory" and index + 1 < len(tokens):
                if replacement:
                    normalized.extend((token, replacement))
                index += 2
                continue
            if token.startswith("--directory="):
                if replacement:
                    normalized.append(f"--directory={replacement}")
                index += 1
                continue
            normalized.append(token)
            index += 1
        return shlex.join(normalized)

    async def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        on_event: EventCallback | None = None,
    ) -> dict[str, Any]:
        if name not in TOOLS:
            raise ValueError(f"unknown tool: {name}")
        self.record({"type": "tool_start", "tool": name, "arguments": {k: v for k, v in arguments.items() if k != "content"}})
        if name == "run_python":
            execution = await self._run_with_retries(str(arguments.get("code", "")), on_event)
            result = {
                "stdout": execution.stdout,
                "stderr": execution.stderr,
                "events": execution.events,
                "code": str(arguments.get("code", "")),
            }
        elif name == "write_file":
            path = _workspace_relative(arguments["path"])
            await self.sandbox.upload(path, str(arguments.get("content", "")).encode())
            result = {"path": path}
        elif name == "read_file":
            path = _workspace_relative(arguments["path"])
            data = await self.sandbox.read_file(path)
            result = {"path": path, "content": data.decode("utf-8", errors="replace")}
        elif name == "list_files":
            path = _workspace_relative(arguments.get("path", "."))
            result = {"files": await self.sandbox.list_files("." if path == "/" else path)}
        elif name == "start_webapp":
            port = int(arguments.get("port", 8501))
            command = self._normalize_webapp_command(str(arguments["command"]))
            result = {"port": await self.sandbox.start_webapp(command, port)}
        elif name == "stop_webapp":
            await self.sandbox.stop_webapp(int(arguments["port"]))
            result = {"port": arguments["port"], "stopped": True}
        elif name == "register_artifact":
            artifacts = await self.sandbox.artifacts()
            requested = arguments.get("name")
            requested = _workspace_relative(requested).rstrip("/") if requested else None
            selected = [a for a in artifacts if not requested or a.get("name") == requested]
            if requested and not selected:
                # Models sometimes register the directory they serve (for
                # example attrition_dashboard/) instead of the HTML page inside
                # it; resolve the directory to its served page so the canvas
                # still receives the artifact.
                prefix = requested + "/"
                nested = [a for a in artifacts if str(a.get("path", "")).startswith(prefix)]
                served = next((a for a in nested if a.get("name") == "index.html"), None)
                if served is None:
                    served = next(
                        (a for a in nested if str(a.get("path", "")).lower().endswith(".html")),
                        None,
                    )
                selected = [served] if served is not None else []
            for artifact in selected:
                if arguments.get("port"):
                    artifact["port"] = int(arguments["port"])
                if arguments.get("type"):
                    artifact["type"] = arguments["type"]
            result = {"artifacts": selected}
        else:
            raise ValueError(f"unhandled tool: {name}")
        self.record({"type": "tool_result", "tool": name, "result": result})
        return result

    async def _run_with_retries(self, code: str, on_event: EventCallback | None = None) -> Execution:
        execution = await self.sandbox.exec(code, on_event)
        retries = 0
        # Retry only a bare failure (stderr with no stdout and no result), the
        # signature of a transient kernel error. Code that already produced
        # output must not be re-executed: stderr also carries harmless warnings,
        # and re-running would duplicate side effects such as file writes.
        while execution.stderr and not execution.stdout and execution.result is None and retries < 3:
            retries += 1
            execution = await self.sandbox.exec(code, on_event)
        return execution


class Agent:
    def __init__(
        self,
        sandbox: Sandbox,
        provider: LLMProvider,
        session_id: str,
        sessions_dir: str | Path = "sessions",
        dataset_context: Callable[[], list[dict[str, Any]]] | None = None,
        history: list[dict[str, Any]] | None = None,
    ):
        self.provider = provider
        self.dispatcher = ToolDispatcher(sandbox, session_id, Path(sessions_dir))
        self.dataset_context = dataset_context or (lambda: [])
        # A caller can own the transcript (the gateway keeps it on the session) so
        # conversation memory survives a reconnect, which builds a new Agent.
        self.messages: list[dict[str, Any]] = history if history is not None else []

    @staticmethod
    def _is_dashboard_request(content: str) -> bool:
        lowered = content.lower()
        return "dashboard" in lowered or "web app" in lowered or "webapp" in lowered

    @staticmethod
    def _is_report_request(content: str) -> bool:
        lowered = content.lower()
        # "whitepaper" and "pdf" are strong signals on their own; "report" needs
        # an action verb nearby to avoid matching e.g. "what does the report say".
        if "pdf" in lowered or "whitepaper" in lowered:
            return True
        if lowered.strip() == "report":
            return True
        return re.search(
            r"\b(?:create|generate|write|build|make|produce|export|draft)\b.{0,60}\breport\b",
            lowered,
        ) is not None

    @staticmethod
    def _is_pdf_artifact(artifact: dict[str, Any]) -> bool:
        path = str(artifact.get("path") or artifact.get("name") or "").lower()
        return artifact.get("type") == "pdf" or path.endswith(".pdf")

    @staticmethod
    def _fallback_pdf_code(content: str) -> str:
        """Return sandbox code that builds a simple PDF from analysis text.

        Dependency-free (same raw-PDF approach as the mock provider) so it
        works in both the Docker sandbox and the in-process FakeSandbox.
        Content is base64-encoded to avoid all escaping issues.
        """
        import base64

        content_b64 = base64.b64encode(content.encode()).decode()
        return '''
import base64
import os
from pathlib import Path

# The kernel preloads WORKSPACE, but agent code can clobber it; the sandbox
# image always sets SANDBOX_WORKSPACE, so prefer it (same as the rescue sweep).
workspace = Path(os.environ.get("SANDBOX_WORKSPACE") or WORKSPACE)
content = base64.b64decode("''' + content_b64 + '''").decode()

lines = content.split("\\n")[:35]
objects = [
    b"<< /Type /Catalog /Pages 2 0 R >>",
    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
]
parts = []
y = 700
for line in lines:
    esc = line.replace("\\\\", "\\\\\\\\").replace("(", "\\\\(").replace(")", "\\\\)")
    parts.append("BT /F1 11 Tf 72 " + str(y) + " Td (" + esc + ") Tj ET")
    y -= 16
data = "\\n".join(parts).encode()
objects.append(b"<< /Length " + str(len(data)).encode() + b" >>\\nstream\\n" + data + b"endstream")
objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

pdf = bytearray(b"%PDF-1.4\\n")
offsets = [0]
for i, obj in enumerate(objects, 1):
    offsets.append(len(pdf))
    pdf.extend(str(i).encode() + b" 0 obj\\n" + obj + b"\\nendobj\\n")
xref = len(pdf)
pdf.extend(b"xref\\n0 " + str(len(objects) + 1).encode() + b"\\n0000000000 65535 f \\n")
for off in offsets[1:]:
    pdf.extend(str(off).zfill(10).encode() + b" 00000 n \\n")
pdf.extend(b"trailer\\n<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R >>\\nstartxref\\n" + str(xref).encode() + b"\\n%%EOF\\n")
(workspace / "report.pdf").write_bytes(bytes(pdf))
print("Fallback PDF written")
'''

    @staticmethod
    def _dataset_summary_text(profiles: list[dict[str, Any]] | None) -> str:
        """Minimal report content when a provider produced no text at all."""
        lines = ["Data Copilot report", ""]
        for profile in profiles or []:
            lines.append(
                f"{profile.get('name', 'dataset')} ({profile.get('file', 'unknown')}): "
                f"{profile.get('rows', '?')} rows x {profile.get('columns', '?')} columns"
            )
        return "\n".join(lines)

    async def _recover_pdf_artifact(
        self, fallback_content: str = ""
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Recover a PDF when a provider wrote it but omitted the handoff.

        If no PDF exists after the rescue sweep, generate one from the
        assistant's analysis text so the user always gets a canvas artifact.
        """
        async def find_pdf() -> str | None:
            listing = await self.dispatcher.dispatch("list_files", {"path": "."})
            files = listing.get("files", [])
            for item in files:
                if (
                    not isinstance(item, dict)
                    or item.get("directory")
                    or int(item.get("size", 0) or 0) < 5
                    or not str(item.get("path", "")).lower().endswith(".pdf")
                ):
                    continue
                try:
                    result = await self.dispatcher.dispatch(
                        "read_file",
                        {"path": str(item["path"])},
                    )
                    if str(result.get("content", "")).startswith("%PDF-"):
                        return str(item["path"])
                except Exception:
                    continue
            return None

        try:
            pdf_path = await find_pdf()
            # Providers occasionally write their output to the working directory,
            # /mnt/data, or /tmp. Copy only those known runtime output locations,
            # never arbitrary host paths, into the persisted workspace.
            if not pdf_path and not hasattr(self.dispatcher.sandbox, "root"):
                await self.dispatcher.dispatch("run_python", {"code": PDF_RESCUE_CODE})
                pdf_path = await find_pdf()
            fallback_result: dict[str, Any] = {}
            if not pdf_path:
                # The provider ran analysis but never wrote a PDF. Build one
                # from the collected output so the canvas is not empty.
                fallback_result = await self.dispatcher.dispatch(
                    "run_python", {"code": self._fallback_pdf_code(fallback_content)}
                )
                pdf_path = await find_pdf()
            if not pdf_path:
                # run_python reports code failures as stderr, not exceptions, so
                # surface the real reason instead of a generic "not found".
                stderr = str(fallback_result.get("stderr") or "").strip()
                if stderr:
                    return None, f"the fallback PDF generator failed: {stderr.splitlines()[-1]}"
                return None, "no PDF artifact was found in the sandbox workspace"
            artifact_result = await self.dispatcher.dispatch(
                "register_artifact",
                {"name": Path(pdf_path).name, "type": "pdf"},
            )
            artifact = next(
                (
                    item
                    for item in artifact_result.get("artifacts", [])
                    if self._is_pdf_artifact(item)
                ),
                None,
            )
            if not artifact:
                return None, "the recovered PDF could not be registered"
            artifact["type"] = "pdf"
            return artifact, None
        except Exception as exc:
            return None, str(exc)

    async def _recover_dashboard_artifact(
        self,
        started_ports: dict[int, str],
        dataset_profiles: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Finish the canvas handoff when a provider stopped after writing code."""
        try:
            listing = await self.dispatcher.dispatch("list_files", {"path": "."})
            files = listing.get("files", [])
            paths = [
                str(item.get("path", ""))
                for item in files
                if isinstance(item, dict) and not item.get("directory")
            ]
        except Exception as exc:
            return None, str(exc)

        html_path = next(
            (
                path
                for path in paths
                if Path(path).name in {"dashboard.html", "index.html"}
            ),
            next((path for path in paths if path.lower().endswith(".html")), None),
        )
        source_path = next(
            (
                path
                for path in paths
                if Path(path).name in {"app.py", "dashboard.py", "main.py"}
            ),
            None,
        )

        profiles = dataset_profiles or []
        source: str | None = None
        if profiles and html_path:
            try:
                html_result = await self.dispatcher.dispatch(
                    "read_file",
                    {"path": html_path},
                )
                content = str(html_result["content"])
                if not self._dashboard_has_visible_data(content, profiles):
                    return await self._create_grounded_fallback(profiles, started_ports)
            except Exception as exc:
                return None, str(exc)

        if profiles and source_path and not html_path:
            return await self._create_grounded_fallback(profiles, started_ports)

        if profiles and not html_path and not source_path and started_ports:
            return await self._create_grounded_fallback(profiles, started_ports)

        # A successful start_webapp call already waits for the sandbox process
        # to bind its port. Use a generated HTML path when one exists; app
        # frameworks such as Flask and Streamlit serve their preview at root.
        if started_ports:
            port, command = next(iter(started_ports.items()))
            path = self._static_preview_path(html_path, command)
            return {
                "name": html_path or "dashboard",
                "path": path,
                "port": port,
                "type": "webapp",
            }, None

        if source_path:
            try:
                if source is None:
                    source_result = await self.dispatcher.dispatch(
                        "read_file",
                        {"path": source_path},
                    )
                    source = str(source_result["content"])
                quoted_path = shlex.quote(source_path)
                if "streamlit" in source.lower():
                    command = (
                        f"streamlit run {quoted_path} --server.address=0.0.0.0 "
                        "--server.port=8501 --server.headless=true"
                    )
                else:
                    command = f"python {quoted_path}"
                await self.dispatcher.dispatch(
                    "start_webapp",
                    {"command": command, "port": 8501},
                )
                return {
                    "name": source_path,
                    "path": None,
                    "port": 8501,
                    "type": "webapp",
                }, None
            except Exception as exc:
                return None, str(exc)

        if html_path:
            try:
                await self.dispatcher.dispatch(
                    "start_webapp",
                    {
                        "command": "python -m http.server 8501 --bind 0.0.0.0",
                        "port": 8501,
                    },
                )
                return {
                    "name": html_path,
                    "path": html_path,
                    "port": 8501,
                    "type": "webapp",
                }, None
            except Exception as exc:
                return None, str(exc)

        return None, "no runnable dashboard source or HTML file was found"

    async def _create_grounded_fallback(
        self,
        profiles: list[dict[str, Any]],
        started_ports: dict[int, str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Serve a static, data-grounded view when generated UI content is empty."""
        try:
            path = "data_grounded_dashboard.html"
            await self.dispatcher.dispatch(
                "write_file",
                {"path": path, "content": self._grounded_dashboard_html(profiles)},
            )
            port = 8502 if 8502 not in started_ports else 8501
            if port == 8501 and 8501 in started_ports:
                return None, "all supported preview ports are already in use"
            await self.dispatcher.dispatch(
                "start_webapp",
                {
                    "command": f"python -m http.server {port} --bind 0.0.0.0",
                    "port": port,
                },
            )
            return {
                "name": path,
                "path": path,
                "port": port,
                "type": "webapp",
            }, None
        except Exception as exc:
            return None, str(exc)

    @staticmethod
    def _dashboard_has_visible_data(
        content: str,
        profiles: list[dict[str, Any]],
    ) -> bool:
        """Require concrete profile values outside scripts and styles."""
        visible = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", content, flags=re.I | re.S)
        visible = re.sub(r"<[^>]+>", " ", visible)
        visible = html.unescape(visible).lower()
        if not visible.strip():
            return False

        metadata_markers: set[str] = set()
        value_markers: set[str] = set()
        for profile in profiles:
            file_name = str(profile.get("file", "")).strip().lower()
            if file_name:
                metadata_markers.add(file_name)
            rows = profile.get("rows")
            if rows is not None:
                metadata_markers.update({str(rows).lower(), f"{int(rows):,}".lower()})
            for column in (profile.get("dtypes") or {}):
                if str(column).strip():
                    metadata_markers.add(str(column).strip().lower())
            for stats in (profile.get("numeric_stats") or {}).values():
                if not isinstance(stats, dict):
                    continue
                for key in ("min", "max", "mean", "median"):
                    value = stats.get(key)
                    if value is None:
                        continue
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        continue
                    value_markers.update(
                        {
                            str(value).lower(),
                            f"{number:g}".lower(),
                            f"{number:,.2f}".lower(),
                        }
                    )
            for row in profile.get("sample_rows") or []:
                if not isinstance(row, dict):
                    continue
                for value in row.values():
                    if value is None:
                        continue
                    marker = str(value).strip().lower()
                    if len(marker) >= 3:
                        value_markers.add(marker)

        def contains_marker(marker: str) -> bool:
            if not marker:
                return False
            if any(character.isalnum() for character in marker):
                pattern = rf"(?<!\w){re.escape(marker)}(?!\w)"
                return re.search(pattern, visible) is not None
            return marker in visible

        has_metadata = any(contains_marker(marker) for marker in metadata_markers)
        value_matches = {marker for marker in value_markers if contains_marker(marker)}
        return has_metadata and len(value_matches) >= 2

    @staticmethod
    def _grounded_dashboard_html(profiles: list[dict[str, Any]]) -> str:
        cards: list[str] = []
        insights: list[str] = []
        samples: list[tuple[str, str]] = []
        chart_rows: list[str] = []
        for profile in profiles:
            file_name = html.escape(str(profile.get("file", profile.get("name", "dataset"))))
            rows = int(profile.get("rows", 0) or 0)
            columns = int(profile.get("columns", 0) or 0)
            cards.append(
                f'<article class="card"><span class="label">{file_name}</span>'
                f'<strong>{rows:,}</strong><small>{columns} columns</small></article>'
            )
            insights.append(
                f"<li><b>{file_name}</b> contains {rows:,} rows across "
                f"{columns} columns.</li>"
            )
            for column, stats in (profile.get("numeric_stats") or {}).items():
                if not isinstance(stats, dict) or stats.get("mean") is None:
                    continue
                label = html.escape(str(column))
                try:
                    mean = float(stats["mean"])
                except (TypeError, ValueError):
                    continue
                minimum = stats.get("min")
                maximum = stats.get("max")
                range_text = ""
                if minimum is not None and maximum is not None:
                    try:
                        range_text = f" (range {float(minimum):g} to {float(maximum):g})"
                    except (TypeError, ValueError):
                        range_text = ""
                try:
                    upper_bound = max(abs(float(minimum or 0)), abs(float(maximum or 0)), abs(mean), 1)
                    bar_width = min(100, max(4, abs(mean) / upper_bound * 100))
                    chart_rows.append(
                        f'<div class="chart-row"><span>{label}</span>'
                        f'<div class="track"><i style="width:{bar_width:.1f}%"></i></div>'
                        f'<b>{mean:,.2f}</b></div>'
                    )
                except (TypeError, ValueError):
                    pass
                insights.append(
                    f"<li><b>{label}</b> averages {mean:,.2f}"
                    f"{html.escape(range_text)}.</li>"
                )
            for row in profile.get("sample_rows") or []:
                if isinstance(row, dict) and len(samples) < 8:
                    for key, value in list(row.items())[:4]:
                        samples.append((str(key), str(value)))

        insight_html = "".join(insights[:8]) or "<li>Profiled records are ready for exploration.</li>"
        sample_html = "".join(
            f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>"
            for key, value in samples
        )
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data-grounded dashboard</title>
<style>
:root{{--ink:#172033;--muted:#667085;--line:#e5eaf1;--accent:#0d9488;--bg:#f5f7fb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:40px 28px 56px}}.eyebrow{{color:var(--accent);font-size:12px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase}}
h1{{font-size:32px;margin:8px 0}}.subtitle{{color:var(--muted);margin:0 0 26px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;margin:24px 0}}
.card,section{{background:white;border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 8px 24px #1720330d}}
.label,small{{display:block;color:var(--muted);font-size:12px}}strong{{display:block;font-size:30px;margin:9px 0 2px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.wide{{grid-column:1/-1}}
h2{{font-size:18px;margin:0 0 15px}}li{{margin:12px 0;line-height:1.5}}
.chart-row{{display:grid;grid-template-columns:170px 1fr 90px;gap:12px;align-items:center;margin:13px 0;font-size:12px}}
.chart-row span{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.track{{height:10px;background:#edf1f6;border-radius:9px;overflow:hidden}}
.track i{{display:block;height:100%;background:linear-gradient(90deg,#0d9488,#5eead4);border-radius:9px}}
.chart-row b{{text-align:right;color:var(--muted)}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid var(--line);padding:8px;text-align:left}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="eyebrow">Data Copilot · grounded fallback</div><h1>Dataset insights</h1>
<p class="subtitle">Concrete values computed from the uploaded data.</p>
<div class="cards">{''.join(cards)}</div>
<div class="grid"><section><h2>Key insights</h2><ul>{insight_html}</ul></section>
<section><h2>Sample values from the dataset</h2><table>{sample_html}</table></section></div>
<section class="wide"><h2>Numeric averages</h2>{''.join(chart_rows) or '<p>No numeric metrics were detected.</p>'}</section>
</main></body></html>"""

    @staticmethod
    def _static_preview_path(html_path: str | None, command: str) -> str | None:
        if not html_path or "http.server" not in command:
            return None
        directory = _http_server_directory(command)
        if directory:
            # html_path is usually workspace-relative while --directory often
            # names the served root absolutely (for example
            # --directory /workspace/attrition_dashboard); compare both in
            # workspace-relative form so the preview path is relative to the
            # server's document root instead of 404ing behind the proxy.
            relative_dir = _workspace_relative(directory).strip("/\\")
            normalized = _workspace_relative(html_path)
            if relative_dir in {"", ".", "workspace"}:
                # Serving the workspace root: the relative path already matches
                # the server's document root.
                return normalized
            prefix = relative_dir + "/"
            if normalized.startswith(prefix):
                return normalized[len(prefix):]
        return html_path

    async def _dispatch(
        self, name: str, arguments: dict[str, Any]
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Dispatch a tool, forwarding kernel output while the code is still running.

        Yields ``("event", output_event)`` zero or more times and always ends with
        ``("result", tool_result)``. The sandbox reports output through a callback,
        so a queue is what lets an async generator re-emit it as it arrives.
        """
        if name != "run_python":
            yield "result", await self.dispatcher.dispatch(name, arguments)
            return
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        task = asyncio.create_task(self.dispatcher.dispatch(name, arguments, queue.put_nowait))
        task.add_done_callback(lambda _: queue.put_nowait(None))
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield "event", event
            yield "result", await task
        finally:
            # A cancelled turn (user pressed stop) must not leave the dispatch running.
            if not task.done():
                task.cancel()

    async def turn(self, content: str) -> AsyncIterator[dict[str, Any]]:
        self.messages.append({"role": "user", "content": content})
        self.dispatcher.record({"type": "user_message", "content": content})
        try:
            dataset_profiles = self.dataset_context()
            request_messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt(dataset_profiles)},
                *self.messages,
            ]
            final_text = ""
            dashboard_request = self._is_dashboard_request(content)
            report_request = self._is_report_request(content)
            # Tracked per category so a combined dashboard+report turn gates each
            # recovery path below on its own artifact kind.
            published_pdf = False
            published_webapp = False
            started_ports: dict[int, str] = {}
            all_text: list[str] = []
            run_python_outputs: list[str] = []
            # Eight rounds give slower providers room to explore and still write
            # the artifact; the fallback below covers turns that exhaust it.
            for _ in range(8):
                text_parts: list[str] = []
                tool_calls: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
                async for item in self.provider.stream(request_messages, TOOL_DEFINITIONS):
                    if isinstance(item, str):
                        text_parts.append(item)
                        yield {"type": "assistant_delta", "content": item}
                        continue
                    tool_id = f"tool-{datetime.now(timezone.utc).timestamp():.6f}"
                    provider_call_id = str(item.get("call_id", tool_id))
                    arguments = item.get("arguments", {})
                    yield {
                        "type": "tool_start",
                        "id": tool_id,
                        "tool": item["tool"],
                        "input": arguments,
                    }
                    streamed = False
                    if item["tool"] == "run_python":
                        # Publish the code before it runs so the inspector is not
                        # empty for the duration of the execution.
                        yield {
                            "type": "execution",
                            "status": "running",
                            "code": str(arguments.get("code", "")),
                        }
                    tool_result: dict[str, Any] = {}
                    try:
                        async for kind, chunk in self._dispatch(item["tool"], arguments):
                            if kind == "result":
                                tool_result = chunk
                                continue
                            data = chunk.get("data")
                            if not isinstance(data, str) or not data:
                                continue
                            streamed = True
                            yield {
                                "type": "execution",
                                "status": "running",
                                "stream": str(chunk.get("type") or "stdout"),
                                "data": data,
                            }
                    except Exception as exc:
                        # A failing tool (e.g. read_file on a path sandboxd
                        # rejects) must not kill the turn: hand the error back
                        # to the model so it can recover.
                        tool_result = {"error": str(exc)}
                    tool_calls.append((provider_call_id, item["tool"], arguments, tool_result))
                    yield {"type": "tool_result", "id": tool_id, "tool": item["tool"], **tool_result}
                    if item["tool"] == "run_python":
                        stdout = str(tool_result.get("stdout", ""))
                        if stdout.strip():
                            run_python_outputs.append(stdout.strip())
                        try:
                            variables = await self.dispatcher.sandbox.vars()
                        except Exception:
                            # Variable inspection is best-effort and must never break a turn.
                            variables = []
                        completed = dict(tool_result)
                        if streamed:
                            # Output already reached the client incrementally; resending
                            # the aggregate would duplicate it.
                            completed.pop("stdout", None)
                            completed.pop("stderr", None)
                        yield {"type": "execution", "status": "complete", "variables": variables, **completed}
                    if item["tool"] == "start_webapp" and "port" in tool_result:
                        normalized_command = ToolDispatcher._normalize_webapp_command(
                            str(arguments.get("command", ""))
                        )
                        started_ports[int(tool_result["port"])] = str(
                            normalized_command
                        )
                    if item["tool"] == "stop_webapp" and "port" in tool_result:
                        started_ports.pop(int(tool_result["port"]), None)
                    if item["tool"] == "register_artifact" and "artifacts" in tool_result:
                        artifacts = tool_result["artifacts"]
                        for artifact in artifacts:
                            is_webapp = bool(
                                artifact.get("type") == "webapp" and artifact.get("port")
                            )
                            is_pdf = self._is_pdf_artifact(artifact)
                            # A turn can ask for a dashboard AND a report: match
                            # each artifact to its own category rather than
                            # requiring every artifact to satisfy both gates.
                            if dashboard_request and report_request:
                                if not (is_webapp or is_pdf):
                                    continue
                            elif dashboard_request and not is_webapp:
                                continue
                            elif report_request and not is_pdf:
                                continue
                            artifact = dict(artifact)
                            if report_request and is_pdf:
                                artifact["type"] = "pdf"
                            artifact_source_path = artifact.get("path")
                            if dashboard_request and is_webapp and artifact.get("path"):
                                command = started_ports.get(int(artifact["port"]), "")
                                artifact["path"] = self._static_preview_path(
                                    str(artifact["path"]),
                                    command,
                                )
                            if dashboard_request and is_webapp and dataset_profiles:
                                artifact_path = artifact_source_path
                                if artifact_path:
                                    try:
                                        content_result = await self.dispatcher.dispatch(
                                            "read_file",
                                            {"path": str(artifact_path)},
                                        )
                                        if not self._dashboard_has_visible_data(
                                            str(content_result["content"]),
                                            dataset_profiles,
                                        ):
                                            continue
                                    except Exception:
                                        # A read hiccup says nothing about the
                                        # dashboard's content; still publish it.
                                        pass
                            if is_webapp:
                                published_webapp = True
                            if is_pdf:
                                published_pdf = True
                            yield {
                                "type": "artifact",
                                "name": artifact.get("name"),
                                "path": artifact.get("path"),
                                "port": artifact.get("port"),
                                "artifact": artifact,
                            }
                final_text = "".join(text_parts)
                if final_text:
                    all_text.append(final_text)
                if not tool_calls:
                    break
                request_messages.append(
                    {
                        "role": "assistant",
                        "content": final_text or None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                            for call_id, name, arguments, _ in tool_calls
                        ],
                    }
                )
                for call_id, name, _, result in tool_calls:
                    request_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": json.dumps(result, default=str),
                        }
                    )
            if report_request and not published_pdf:
                fallback_content = "\n\n".join(all_text or run_python_outputs)
                if not fallback_content:
                    # The provider hit the step cap without saying anything;
                    # fall back to a dataset summary so the PDF still has content.
                    fallback_content = self._dataset_summary_text(dataset_profiles)
                recovered, recovery_error = await self._recover_pdf_artifact(
                    fallback_content
                )
                if recovered:
                    self.dispatcher.record(
                        {"type": "artifact_recovered", "artifact": recovered}
                    )
                    yield {
                        "type": "artifact",
                        "name": recovered["name"],
                        "path": recovered["path"],
                        "artifact": recovered,
                    }
                    final_text = (
                        f"{final_text}\n\nThe PDF report is ready in the canvas."
                        if final_text
                        else "The PDF report is ready in the canvas."
                    )
                else:
                    notice = (
                        "I generated report work, but could not put a PDF on the "
                        f"canvas: {recovery_error}."
                    )
                    final_text = f"{final_text}\n\n{notice}" if final_text else notice
            if dashboard_request and not published_webapp:
                recovered, recovery_error = await self._recover_dashboard_artifact(
                    started_ports,
                    dataset_profiles,
                )
                if recovered:
                    self.dispatcher.record(
                        {"type": "artifact_recovered", "artifact": recovered}
                    )
                    yield {
                        "type": "artifact",
                        "name": recovered["name"],
                        "path": recovered.get("path"),
                        "port": recovered["port"],
                        "artifact": recovered,
                    }
                    final_text = (
                        f"{final_text}\n\nThe dashboard is now running in the canvas."
                        if final_text
                        else "The dashboard is running in the canvas."
                    )
                else:
                    final_text = (
                        "I generated dashboard work, but could not put a runnable "
                        f"artifact on the canvas: {recovery_error}."
                    )
            if final_text:
                self.messages.append({"role": "assistant", "content": final_text})
                yield {"type": "assistant_message", "content": final_text}
            else:
                yield {"type": "assistant_message", "content": "The requested sandbox work finished."}
            yield {"type": "done"}
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}
