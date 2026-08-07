"""Provider-neutral agent loop and sandbox tool dispatch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Protocol

from .paths import safe_path
from .sandbox import Execution, Sandbox

TOOLS = {"run_python", "write_file", "read_file", "list_files", "start_webapp", "stop_webapp", "register_artifact"}
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
            "description": "Register a generated HTML, PDF, image, or other artifact for the canvas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["webapp", "pdf", "document"]},
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
        prompt = messages[-1]["content"].lower()
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
        elif any(term in prompt for term in ("executive summary", "summary", "insight", "analy")):
            yield mock_executive_summary(messages)
        else:
            yield "I can analyze the uploaded data, build a dashboard, or write a PDF report."


def dashboard_code() -> str:
    return """from pathlib import Path
workspace = Path(WORKSPACE)
html = '''<!doctype html><html><head><meta charset="utf-8"><title>Data Copilot Dashboard</title>
<style>body{font-family:system-ui;margin:2rem;background:#f8fafc}main{max-width:900px;margin:auto}
.card{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:1.5rem;box-shadow:0 2px 8px #0001}
h1{color:#0f172a}.metric{font-size:2rem;font-weight:700;color:#2563eb}</style></head>
<body><main><div class="card"><h1>Data Copilot Dashboard</h1>
<p class="metric">Analysis complete</p><p>Interactive dashboard generated in the session sandbox.</p>
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


class AnthropicProvider:
    def __init__(self, api_key: str, model: str):
        self.api_key, self.model = api_key, model

    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AsyncIterator[str]:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise RuntimeError("Anthropic provider requires the optional providers dependency") from exc
        client = AsyncAnthropic(api_key=self.api_key)
        system = next((item["content"] for item in messages if item.get("role") == "system"), None)
        user_messages = [item for item in messages if item.get("role") != "system"]
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": user_messages,
        }
        if system:
            request["system"] = system
        async with client.messages.stream(**request) as stream:
            async for text in stream.text_stream:
                yield text


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

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in TOOLS:
            raise ValueError(f"unknown tool: {name}")
        self.record({"type": "tool_start", "tool": name, "arguments": {k: v for k, v in arguments.items() if k != "content"}})
        if name == "run_python":
            execution = await self._run_with_retries(str(arguments.get("code", "")))
            result = {
                "stdout": execution.stdout,
                "stderr": execution.stderr,
                "events": execution.events,
                "code": str(arguments.get("code", "")),
            }
        elif name == "write_file":
            await self.sandbox.upload(str(arguments["path"]), str(arguments.get("content", "")).encode())
            result = {"path": arguments["path"]}
        elif name == "read_file":
            data = await self.sandbox.read_file(str(arguments["path"]))
            result = {"path": arguments["path"], "content": data.decode("utf-8", errors="replace")}
        elif name == "list_files":
            result = {"files": await self.sandbox.list_files(str(arguments.get("path", ".")))}
        elif name == "start_webapp":
            port = int(arguments.get("port", 8501))
            result = {"port": await self.sandbox.start_webapp(str(arguments["command"]), port)}
        elif name == "stop_webapp":
            await self.sandbox.stop_webapp(int(arguments["port"]))
            result = {"port": arguments["port"], "stopped": True}
        else:
            artifacts = await self.sandbox.artifacts()
            requested = arguments.get("name")
            selected = [a for a in artifacts if not requested or a.get("name") == requested]
            for artifact in selected:
                if arguments.get("port"):
                    artifact["port"] = int(arguments["port"])
                if arguments.get("type"):
                    artifact["type"] = arguments["type"]
            result = {"artifacts": selected}
        self.record({"type": "tool_result", "tool": name, "result": result})
        return result

    async def _run_with_retries(self, code: str) -> Execution:
        execution = await self.sandbox.exec(code)
        retries = 0
        while execution.stderr and retries < 3:
            retries += 1
            # A provider would receive this context; retrying the same deterministic
            # code is useful for transient fake-kernel errors without hiding stderr.
            execution = await self.sandbox.exec(code)
        return execution


class Agent:
    def __init__(
        self,
        sandbox: Sandbox,
        provider: LLMProvider,
        session_id: str,
        sessions_dir: str | Path = "sessions",
        dataset_context: Callable[[], list[dict[str, Any]]] | None = None,
    ):
        self.provider = provider
        self.dispatcher = ToolDispatcher(sandbox, session_id, Path(sessions_dir))
        self.dataset_context = dataset_context or (lambda: [])
        self.messages: list[dict[str, Any]] = []

    async def turn(self, content: str) -> AsyncIterator[dict[str, Any]]:
        self.messages.append({"role": "user", "content": content})
        self.dispatcher.record({"type": "user_message", "content": content})
        try:
            request_messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt(self.dataset_context())},
                *self.messages,
            ]
            final_text = ""
            for _ in range(4):
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
                    tool_result = await self.dispatcher.dispatch(item["tool"], arguments)
                    tool_calls.append((provider_call_id, item["tool"], arguments, tool_result))
                    yield {"type": "tool_result", "id": tool_id, "tool": item["tool"], **tool_result}
                    if item["tool"] == "run_python":
                        yield {"type": "execution", "status": "complete", **tool_result}
                    if item["tool"] == "register_artifact":
                        for artifact in tool_result["artifacts"]:
                            yield {
                                "type": "artifact",
                                "name": artifact.get("name"),
                                "path": artifact.get("path"),
                                "port": artifact.get("port"),
                                "artifact": artifact,
                            }
                final_text = "".join(text_parts)
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
            if final_text:
                self.messages.append({"role": "assistant", "content": final_text})
                yield {"type": "assistant_message", "content": final_text}
            else:
                yield {"type": "assistant_message", "content": "Completed the requested sandbox work."}
            yield {"type": "done"}
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}
