"""Provider-neutral agent loop and sandbox tool dispatch."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .sandbox import EventCallback, Execution, Sandbox

TOOLS = {"run_python", "write_file", "read_file", "list_files", "start_webapp", "stop_webapp", "register_artifact"}
PROVIDER_WORKSPACE = "/home/oai/share"


def _workspace_relative(path: Any) -> str:
    """Models sometimes pass kernel-absolute paths; the sandbox file API is workspace-relative."""
    text = str(path)
    if text.startswith("/workspace/"):
        return text[len("/workspace/"):]
    return text
# Tool-call rounds allowed per turn, for every request. Sixteen gives slower
# providers room to explore and still author the artifact; a few rounds before
# the cap a wrap-up nudge is folded into the running history so the model
# spends its last steps writing instead of exploring. Four remaining rounds
# leaves enough budget to write, register, and summarize an artifact.
MAX_TURN_STEPS = 16
WRAP_UP_REMAINING_STEPS = 4
WRAP_UP_NUDGE = (
    f"Step budget nearly exhausted: only {WRAP_UP_REMAINING_STEPS} tool "
    "rounds remain. Stop exploring and finish with what you have: write the "
    "requested artifact to the workspace, call register_artifact, then "
    "summarize your findings."
)

# History hygiene: the running prompt keeps the signal from each tool result,
# not its bulk. run_python stream events duplicate stdout chunk-by-chunk for
# the client, and the executed code already sits in the assistant's tool_call
# arguments, so neither is re-sent to the provider. Long outputs are elided
# head+tail with a marker so the model knows material was dropped.
HISTORY_OUTPUT_LIMIT = 4_000
HISTORY_FILE_LIMIT = 8_000


def _elide(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n[... {omitted:,} chars elided ...]\n{text[-half:]}"


def _history_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    # write_file payloads are pure bulk: once the write succeeds, the path and
    # byte count carry all the signal the model needs for the rest of the turn.
    if name == "write_file" and "content" in arguments:
        size = len(str(arguments.get("content", "")).encode("utf-8", errors="replace"))
        return {
            "path": arguments.get("path"),
            "content": f"<written to the sandbox: {size:,} bytes>",
        }
    return arguments


def _history_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
    if name == "run_python":
        trimmed: dict[str, Any] = {
            "stdout": _elide(str(result.get("stdout", "")), HISTORY_OUTPUT_LIMIT),
            "stderr": _elide(str(result.get("stderr", "")), HISTORY_OUTPUT_LIMIT),
        }
        if result.get("error"):
            trimmed["error"] = result["error"]
        return trimmed
    if name == "read_file" and "content" in result:
        return {**result, "content": _elide(str(result["content"]), HISTORY_FILE_LIMIT)}
    return result


@dataclass(frozen=True)
class _FileRecovery:
    """One recoverable output kind: which files qualify (extension plus magic
    bytes where the format has a reliable signature), the artifact type they
    register under, and the wording for user-facing recovery notices."""

    artifact_type: str
    signatures: dict[str, bytes | None]
    ready: str
    failed: str
    exhausted: str
    missing: str


_FILE_RECOVERIES: dict[str, _FileRecovery] = {
    "pdf": _FileRecovery(
        artifact_type="pdf",
        signatures={".pdf": b"%PDF-"},
        ready="The PDF report is ready in the canvas.",
        failed="I generated report work, but could not put a PDF on the canvas",
        exhausted="I ran out of steps before I could finish the PDF report",
        missing="no PDF artifact was found in the sandbox workspace",
    ),
    "slides": _FileRecovery(
        artifact_type="slides",
        signatures={".pptx": b"PK\x03\x04"},
        ready="The slide deck is ready in the canvas.",
        failed="I generated slide deck work, but could not put a deck on the canvas",
        exhausted="I ran out of steps before I could finish the slide deck",
        missing="no slide deck file was found in the sandbox workspace",
    ),
    "image": _FileRecovery(
        artifact_type="image",
        # PNG's leading byte is not valid UTF-8, so it cannot be verified
        # through read_file's decoded content; the extension is the contract.
        signatures={".png": None, ".svg": None},
        ready="The chart is ready in the canvas.",
        failed="I generated chart work, but could not put an image on the canvas",
        exhausted="I ran out of steps before I could finish the chart",
        missing="no chart image was found in the sandbox workspace",
    ),
    "data": _FileRecovery(
        artifact_type="data",
        signatures={".xlsx": b"PK\x03\x04", ".csv": None},
        ready="The data export is ready in the canvas.",
        failed="I generated export work, but could not put a file on the canvas",
        exhausted="I ran out of steps before I could finish the data export",
        missing="no export file was found in the sandbox workspace",
    ),
}

# The kernel preloads WORKSPACE for convenience, but agent code can reassign it,
# so this snippet resolves the real workspace from the sandbox environment. A
# wrong destination would silently copy a rescued file back onto itself.
# Scan roots default to known provider output locations (some providers
# scratch-write under /tmp or /mnt/data); SANDBOX_RESCUE_ROOTS overrides the
# list so tests never glob the host's real temp directory.
_RESCUE_TEMPLATE = """import os
from pathlib import Path
workspace = Path(os.environ.get("SANDBOX_WORKSPACE", "/workspace")).resolve()
roots_override = os.environ.get("SANDBOX_RESCUE_ROOTS")
if roots_override:
    source_roots = [Path(part) for part in roots_override.split(os.pathsep) if part]
else:
    source_roots = [Path.cwd(), Path("/app"), Path("/mnt/data"), Path("/tmp")]
signatures = __SIGNATURES__
candidates = []
for source_root in source_roots:
    for extension in signatures:
        try:
            candidates.extend(source_root.glob("*" + extension))
        except OSError:
            pass
valid = []
for source in candidates:
    try:
        source = source.resolve()
        if source.is_relative_to(workspace):
            continue
        data = source.read_bytes()
        magic = signatures.get(source.suffix.lower())
        if magic is not None and not data.startswith(magic):
            continue
        valid.append((source.stat().st_mtime, source, data))
    except OSError:
        pass
if valid:
    _, source, data = max(valid, key=lambda item: item[0])
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / source.name).write_bytes(data)
"""


def _rescue_code(recovery: _FileRecovery) -> str:
    """Kernel snippet that copies a stray output file into the workspace.

    Only files of the requested kind are copied (extension, plus magic bytes
    when the format has a reliable signature), never arbitrary host paths.
    """
    return _RESCUE_TEMPLATE.replace("__SIGNATURES__", repr(recovery.signatures))


# Kept under its original name: the script exporter and tests recognize this
# exact cell when replaying a trajectory.
PDF_RESCUE_CODE = _rescue_code(_FILE_RECOVERIES["pdf"])
# Every rescue variant, so the script exporter can skip internal plumbing
# regardless of which output kind was recovered.
RESCUE_CODES = frozenset(_rescue_code(recovery) for recovery in _FILE_RECOVERIES.values())
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
            "description": "Register a generated HTML, PDF, image, spreadsheet, slide deck, or other artifact for the canvas. "
            "For a webapp, register the served HTML file (for example dashboard.html or site/index.html). "
            "For an Excel or CSV export, register the file with type data so the canvas offers it for download. "
            "For a PowerPoint deck, register the .pptx file with type slides.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["webapp", "pdf", "document", "image", "data", "slides"]},
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

The sandbox ships a report toolkit: import reportkit (it is already on the
import path, no installation needed). Prefer it over hand-writing reportlab,
weasyprint, or CSS boilerplate. For PDFs use reportkit.PdfReport(title,
subtitle, theme="light", accent="#2f6fed"): title_page, kpi_row, section,
prose, bullets, table, callout, chart, figure, page_break, then
build(Path(WORKSPACE) / "name.pdf"). For single-file HTML pages use
reportkit.HtmlPage(title, ...): hero, kpi_row, section, prose, bullets,
table, callout, chart, figure, raw, then save(Path(WORKSPACE) / "name.html").
You still decide the structure, order, length, topics, copy, colors, and
which charts and tables to include; an appendix is just more sections. The
toolkit only handles rendering. chart() covers quick bar/line/scatter/hist
plots straight from the data; figure(fig) embeds any custom matplotlib
figure for full chart control. Raw reportlab and raw(html) blocks remain
available for layouts the toolkit cannot express.

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

For spreadsheet or data-export requests, compute the result from the uploaded
data and save it inside the workspace as Excel (for example,
Path(WORKSPACE) / "summary.xlsx" with DataFrame.to_excel; openpyxl is
preinstalled) or CSV (with DataFrame.to_csv), then call register_artifact
with the filename and type "data" so the canvas offers the file for download.
Prefer Excel for polished multi-column deliverables and CSV when the user
asks for raw data. As with every artifact, only files inside the workspace
can be downloaded, and the task is not complete until the file is registered.

For slide-deck or presentation requests, build the deck with
reportkit.Deck(title, subtitle, theme="light", accent="#2f6fed"): title_slide
and section for structure, then slide(title) to start each content slide
followed by kpi_row, prose, bullets, table, callout, chart, or figure, and
finally build(Path(WORKSPACE) / "deck.pptx"). Then call register_artifact
with the .pptx filename and type "slides". Compute every number on the slides
from the uploaded data first, and keep slides terse: a few bullets or one
visual each. As with every artifact, only files inside the workspace can be
downloaded, and the task is not complete until the deck is registered.
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
    # Implementations are async generator functions, so a call returns the
    # AsyncIterator directly; declaring `async def` here would type the call
    # as a coroutine and mislead both checkers and future implementations.
    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AsyncIterator[str | dict[str, Any]]: ...


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
        elif any(term in prompt for term in ("pptx", "powerpoint", "slide", "deck", "presentation")):
            yield {"tool": "run_python", "arguments": {"code": pptx_code()}}
            yield {"tool": "register_artifact", "arguments": {"name": "deck.pptx", "type": "slides"}}
        elif "pdf" in prompt or "report" in prompt or "whitepaper" in prompt:
            yield {"tool": "run_python", "arguments": {"code": pdf_code()}}
            yield {"tool": "register_artifact", "arguments": {"name": "report.pdf", "type": "pdf"}}
        elif any(term in prompt for term in ("chart", "graph", "plot", "visuali")):
            yield {"tool": "run_python", "arguments": {"code": chart_code()}}
            yield {"tool": "register_artifact", "arguments": {"name": "chart.svg", "type": "image"}}
        elif any(
            term in prompt
            for term in ("excel", "xlsx", "spreadsheet", "to csv", "as csv", "csv export", "export csv")
        ):
            yield {"tool": "run_python", "arguments": {"code": excel_code()}}
            yield {"tool": "register_artifact", "arguments": {"name": "summary.xlsx", "type": "data"}}
        elif any(term in prompt for term in ("executive summary", "summary", "insight", "analy", "explore")):
            yield mock_executive_summary(messages)
        else:
            yield "I can analyze the uploaded data, build a dashboard, write a PDF report, or draft a slide deck."


def excel_code() -> str:
    # Mirrors a competent model: derive a real summary from df_1 and register
    # the workbook as a downloadable data artifact.
    return """from pathlib import Path
import pandas as pd
frame = globals().get("df_1")
summary = frame.describe() if frame is not None else pd.DataFrame({"note": ["no dataset loaded"]})
summary.to_excel(Path(WORKSPACE) / "summary.xlsx")
"""


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


def pptx_code() -> str:
    # A dependency-free, valid two-slide PPTX keeps the offline mock usable in
    # any environment, mirroring pdf_code(): a .pptx is a zip of OOXML parts,
    # so zipfile plus string XML is enough.
    return r'''import html
import zipfile
from pathlib import Path

workspace = Path(WORKSPACE)
frame = globals().get("df_1")
rows = len(frame) if frame is not None else 0
bullets = [f"{rows:,} rows analyzed"]
if frame is not None:
    for name in frame.select_dtypes(include="number").columns[:3]:
        value = frame[name].mean()
        if value == value:  # skip NaN averages
            bullets.append(f"Average {name}: {value:,.2f}")

NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')
RT = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
EMPTY_GROUP = ('<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
               '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
               '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>')


def text_shape(sid, name, x, y, cx, cy, lines, size, color, bold=False, align="l"):
    runs = ""
    for line in lines:
        b = ' b="1"' if bold else ""
        runs += (f'<a:p><a:pPr algn="{align}"/><a:r>'
                 f'<a:rPr lang="en-US" sz="{size}"{b}>'
                 f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:rPr>'
                 f'<a:t>{html.escape(line)}</a:t></a:r></a:p>')
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{name}"/>'
            '<p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>{runs}</p:txBody></p:sp>')


def fill_shape(sid, name, x, y, cx, cy, color):
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{name}"/>'
            '<p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill></p:spPr>'
            '<p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody></p:sp>')


def slide_xml(shapes):
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<p:sld {NS}><p:cSld><p:spTree>{EMPTY_GROUP}{shapes}</p:spTree></p:cSld></p:sld>')


def rels_xml(entries):
    body = "".join(
        f'<Relationship Id="{rid}" Type="{RT}/{kind}" Target="{target}"/>'
        for rid, kind, target in entries
    )
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<Relationships xmlns="{PKG}">{body}</Relationships>')


title_slide = slide_xml(
    fill_shape(2, "AccentBar", 5000445, 2010000, 2194560, 36576, "2F6FED")
    + text_shape(3, "Title", 914400, 2286000, 10383520, 1280160,
                 ["Data Copilot Slide Deck"], 4000, "1C2430", bold=True, align="ctr")
    + text_shape(4, "Subtitle", 914400, 3620000, 10383520, 457200,
                 ["Generated analysis summary"], 1800, "5B6672", align="ctr")
    + text_shape(5, "Meta", 914400, 5181600, 10383520, 365760,
                 ["Prepared by Data Copilot"], 1100, "5B6672", align="ctr")
)
findings_slide = slide_xml(
    text_shape(2, "Title", 548640, 320040, 11094720, 731520,
               ["Key findings"], 2600, "1C2430", bold=True)
    + fill_shape(3, "AccentBar", 566928, 1024128, 1463040, 41148, "2F6FED")
    + text_shape(4, "Bullets", 548640, 1280160, 11094720, 4572000,
                 ["\u2022 " + item for item in bullets], 1600, "1C2430")
)

content_types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
    '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>'
    '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
    '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
    '<Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
    '<Override PartName="/ppt/slides/slide2.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
    '</Types>')

presentation = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    f'<p:presentation {NS}>'
    '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
    '<p:sldIdLst><p:sldId id="256" r:id="rId2"/><p:sldId id="257" r:id="rId3"/></p:sldIdLst>'
    '<p:sldSz cx="12192000" cy="6858000"/><p:notesSz cx="6858000" cy="9144000"/>'
    '</p:presentation>')

master = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    f'<p:sldMaster {NS}><p:cSld><p:spTree>{EMPTY_GROUP}</p:spTree></p:cSld>'
    '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" '
    'accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" '
    'hlink="hlink" folHlink="folHlink"/>'
    '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
    '<p:txStyles>'
    '<p:titleStyle><a:lvl1pPr><a:defRPr sz="3200"/></a:lvl1pPr></p:titleStyle>'
    '<p:bodyStyle><a:lvl1pPr><a:defRPr sz="1800"/></a:lvl1pPr></p:bodyStyle>'
    '<p:otherStyle><a:lvl1pPr><a:defRPr sz="1800"/></a:lvl1pPr></p:otherStyle>'
    '</p:txStyles></p:sldMaster>')

layout = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    f'<p:sldLayout {NS} type="blank" preserve="1">'
    f'<p:cSld name="Blank"><p:spTree>{EMPTY_GROUP}</p:spTree></p:cSld>'
    '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')

theme = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Data Copilot">'
    '<a:themeElements><a:clrScheme name="DataCopilot">'
    '<a:dk1><a:srgbClr val="1C2430"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>'
    '<a:dk2><a:srgbClr val="5B6672"/></a:dk2><a:lt2><a:srgbClr val="F2F5FA"/></a:lt2>'
    '<a:accent1><a:srgbClr val="2F6FED"/></a:accent1><a:accent2><a:srgbClr val="5B6672"/></a:accent2>'
    '<a:accent3><a:srgbClr val="D9DEE6"/></a:accent3><a:accent4><a:srgbClr val="1C2430"/></a:accent4>'
    '<a:accent5><a:srgbClr val="2F6FED"/></a:accent5><a:accent6><a:srgbClr val="5B6672"/></a:accent6>'
    '<a:hlink><a:srgbClr val="2F6FED"/></a:hlink><a:folHlink><a:srgbClr val="5B6672"/></a:folHlink>'
    '</a:clrScheme>'
    '<a:fontScheme name="DataCopilot">'
    '<a:majorFont><a:latin typeface="Helvetica"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>'
    '<a:minorFont><a:latin typeface="Helvetica"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont>'
    '</a:fontScheme>'
    '<a:fmtScheme name="DataCopilot">'
    '<a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>'
    '<a:lnStyleLst>'
    '<a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
    '<a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
    '<a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
    '</a:lnStyleLst>'
    '<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle>'
    '<a:effectStyle><a:effectLst/></a:effectStyle>'
    '<a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>'
    '<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>'
    '</a:fmtScheme></a:themeElements></a:theme>')

parts = {
    "[Content_Types].xml": content_types,
    "_rels/.rels": rels_xml([("rId1", "officeDocument", "ppt/presentation.xml")]),
    "ppt/presentation.xml": presentation,
    "ppt/_rels/presentation.xml.rels": rels_xml([
        ("rId1", "slideMaster", "slideMasters/slideMaster1.xml"),
        ("rId2", "slide", "slides/slide1.xml"),
        ("rId3", "slide", "slides/slide2.xml"),
        ("rId4", "theme", "theme/theme1.xml"),
    ]),
    "ppt/slideMasters/slideMaster1.xml": master,
    "ppt/slideMasters/_rels/slideMaster1.xml.rels": rels_xml([
        ("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml"),
        ("rId2", "theme", "../theme/theme1.xml"),
    ]),
    "ppt/slideLayouts/slideLayout1.xml": layout,
    "ppt/slideLayouts/_rels/slideLayout1.xml.rels": rels_xml([
        ("rId1", "slideMaster", "../slideMasters/slideMaster1.xml"),
    ]),
    "ppt/theme/theme1.xml": theme,
    "ppt/slides/slide1.xml": title_slide,
    "ppt/slides/_rels/slide1.xml.rels": rels_xml([
        ("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml"),
    ]),
    "ppt/slides/slide2.xml": findings_slide,
    "ppt/slides/_rels/slide2.xml.rels": rels_xml([
        ("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml"),
    ]),
}
with zipfile.ZipFile(workspace / "deck.pptx", "w", zipfile.ZIP_DEFLATED) as bundle:
    for part_name, part_xml in parts.items():
        bundle.writestr(part_name, part_xml)
'''


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
                # Only ToolUseBlock carries input; getattr keeps this safe across
                # the SDK's growing union of content block types.
                arguments = getattr(block, "input", {})
                yield {
                    "tool": getattr(block, "name", ""),
                    "arguments": arguments if isinstance(arguments, dict) else {},
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
            # xhigh keeps near-maximum quality on long authoring turns without
            # the latency tax that maximum effort adds to every round.
            "reasoning": {"effort": "xhigh"},
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
        event = {"timestamp": datetime.now(UTC).isoformat(), **event}
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
        self.dataset_context = dataset_context or (list)
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

    # The slides/chart/data classifiers mirror MockProvider's routing keywords,
    # so the offline provider and the recovery gates agree on what was asked for.
    @staticmethod
    def _is_slides_request(content: str) -> bool:
        lowered = content.lower()
        return any(
            term in lowered
            for term in ("pptx", "powerpoint", "slide", "deck", "presentation")
        )

    @staticmethod
    def _is_chart_request(content: str) -> bool:
        lowered = content.lower()
        return any(term in lowered for term in ("chart", "graph", "plot", "visuali"))

    @staticmethod
    def _is_data_export_request(content: str) -> bool:
        lowered = content.lower()
        return any(
            term in lowered
            for term in (
                "excel",
                "xlsx",
                "spreadsheet",
                "to csv",
                "as csv",
                "csv export",
                "export csv",
            )
        )

    @staticmethod
    def _artifact_kind(artifact: dict[str, Any]) -> str | None:
        """Map a registered artifact to its recovery category: explicit type
        first, file extension second."""
        path = str(artifact.get("path") or artifact.get("name") or "").lower()
        for kind, recovery in _FILE_RECOVERIES.items():
            if artifact.get("type") == recovery.artifact_type or path.endswith(
                tuple(recovery.signatures)
            ):
                return kind
        return None

    async def _recover_file_artifact(
        self, kind: str
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Recover a file artifact when a provider wrote it but omitted the handoff.

        Only a real file of the requested kind found in the sandbox is
        published. When the provider never wrote one, the caller says so
        instead of fabricating an artifact from raw tool output.
        """
        recovery = _FILE_RECOVERIES[kind]

        async def find_file() -> str | None:
            listing = await self.dispatcher.dispatch("list_files", {"path": "."})
            files = listing.get("files", [])
            for extension, magic in recovery.signatures.items():
                for item in files:
                    if (
                        not isinstance(item, dict)
                        or item.get("directory")
                        or int(item.get("size", 0) or 0) < 5
                    ):
                        continue
                    path = str(item.get("path", ""))
                    # Uploaded datasets live in data/ and are inputs, not deliverables.
                    if path.startswith("data/") or not path.lower().endswith(extension):
                        continue
                    if magic is None:
                        return path
                    try:
                        result = await self.dispatcher.dispatch(
                            "read_file",
                            {"path": path},
                        )
                        if str(result.get("content", "")).startswith(
                            magic.decode("utf-8", errors="replace")
                        ):
                            return path
                    except Exception:
                        continue
            return None

        try:
            path = await find_file()
            # Providers occasionally write their output to the working directory,
            # /mnt/data, or /tmp. Copy only those known runtime output locations,
            # never arbitrary host paths, into the persisted workspace.
            if not path and not hasattr(self.dispatcher.sandbox, "root"):
                await self.dispatcher.dispatch("run_python", {"code": _rescue_code(recovery)})
                path = await find_file()
            if not path:
                return None, recovery.missing
            artifact_result = await self.dispatcher.dispatch(
                "register_artifact",
                {"name": Path(path).name, "type": recovery.artifact_type},
            )
            artifact = next(
                (
                    item
                    for item in artifact_result.get("artifacts", [])
                    if self._artifact_kind(item) == kind
                ),
                None,
            )
            if not artifact:
                return None, f"the recovered {recovery.artifact_type} file could not be registered"
            artifact["type"] = recovery.artifact_type
            return artifact, None
        except Exception as exc:
            return None, str(exc)

    async def _recover_dashboard_artifact(
        self,
        started_ports: dict[int, str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Finish the canvas handoff when a provider stopped after writing code.

        Only files the provider actually wrote are served. When nothing usable
        exists, the caller says so instead of serving a generated placeholder.
        """
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

        source: str | None = None

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
            # File-artifact kinds the user asked for; a combined turn (say,
            # dashboard + slide deck) gates and recovers each kind on its own.
            requested_kinds = [
                kind
                for kind, requested in (
                    ("pdf", self._is_report_request(content)),
                    ("slides", self._is_slides_request(content)),
                    ("image", self._is_chart_request(content)),
                    ("data", self._is_data_export_request(content)),
                )
                if requested
            ]
            published_kinds: set[str] = set()
            published_webapp = False
            started_ports: dict[int, str] = {}
            # The nudge is folded into the latest tool result rather than sent
            # as its own message: mid-conversation system messages are dropped
            # by the Anthropic and OpenAI Responses adapters, and a bare user
            # message would break Anthropic's role alternation.
            steps_exhausted = True
            for step in range(MAX_TURN_STEPS):
                if (
                    step == MAX_TURN_STEPS - WRAP_UP_REMAINING_STEPS
                    and request_messages[-1].get("role") == "tool"
                ):
                    request_messages[-1]["content"] += "\n\n" + WRAP_UP_NUDGE
                    self.dispatcher.record({"type": "wrap_up_nudge", "step": step + 1})
                    # Surfaced to the chat status line as well as the
                    # trajectory, so the wind-down is visible while it happens.
                    yield {"type": "wrap_up_nudge", "step": step + 1}
                # Drives the chat status line ("round 9 of 16 · …") during
                # long turns; the inspector already shows the tool trail.
                yield {"type": "turn_step", "step": step + 1, "max_steps": MAX_TURN_STEPS}
                text_parts: list[str] = []
                tool_calls: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
                async for item in self.provider.stream(request_messages, TOOL_DEFINITIONS):
                    if isinstance(item, str):
                        text_parts.append(item)
                        yield {"type": "assistant_delta", "content": item}
                        continue
                    tool_id = f"tool-{datetime.now(UTC).timestamp():.6f}"
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
                            kind = self._artifact_kind(artifact)
                            # A turn can ask for several outputs at once: match
                            # each artifact to its own category and keep
                            # intermediate files (charts embedded in a deck,
                            # scratch CSVs) off the canvas.
                            if dashboard_request or requested_kinds:
                                wanted = (dashboard_request and is_webapp) or (
                                    kind is not None and kind in requested_kinds
                                )
                                if not wanted:
                                    continue
                            artifact = dict(artifact)
                            if kind is not None:
                                artifact["type"] = _FILE_RECOVERIES[kind].artifact_type
                            if dashboard_request and is_webapp and artifact.get("path"):
                                command = started_ports.get(int(artifact["port"]), "")
                                artifact["path"] = self._static_preview_path(
                                    str(artifact["path"]),
                                    command,
                                )
                            if is_webapp:
                                published_webapp = True
                            if kind is not None:
                                published_kinds.add(kind)
                            yield {
                                "type": "artifact",
                                "name": artifact.get("name"),
                                "path": artifact.get("path"),
                                "port": artifact.get("port"),
                                "artifact": artifact,
                            }
                final_text = "".join(text_parts)
                if not tool_calls:
                    steps_exhausted = False
                    break
                # History hygiene: the client stream and trajectory already
                # carry the full payloads; the running prompt keeps only the
                # compacted arguments and elided results from here on.
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
                                    "arguments": json.dumps(
                                        _history_arguments(name, arguments)
                                    ),
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
                            "content": json.dumps(
                                _history_result(name, result), default=str
                            ),
                        }
                    )
            unrecovered_kinds = [
                kind for kind in requested_kinds if kind not in published_kinds
            ]
            for kind in unrecovered_kinds:
                recovery = _FILE_RECOVERIES[kind]
                recovered, recovery_error = await self._recover_file_artifact(kind)
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
                        f"{final_text}\n\n{recovery.ready}"
                        if final_text
                        else recovery.ready
                    )
                elif steps_exhausted:
                    notice = (
                        f"{recovery.exhausted}, "
                        "so nothing was published to the canvas. Ask me to "
                        "continue and I'll pick up where I left off."
                    )
                    final_text = f"{final_text}\n\n{notice}" if final_text else notice
                else:
                    notice = f"{recovery.failed}: {recovery_error}."
                    final_text = f"{final_text}\n\n{notice}" if final_text else notice
            if steps_exhausted and not dashboard_request and not unrecovered_kinds:
                notice = (
                    "I ran out of steps before finishing, so this answer may be "
                    "incomplete. Ask me to continue and I'll pick up where I "
                    "left off."
                )
                final_text = f"{final_text}\n\n{notice}" if final_text else notice
            if dashboard_request and not published_webapp:
                recovered, recovery_error = await self._recover_dashboard_artifact(
                    started_ports,
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
                elif steps_exhausted:
                    notice = (
                        "I ran out of steps before I could finish the dashboard, "
                        "so nothing was published to the canvas. Ask me to "
                        "continue and I'll pick up where I left off."
                    )
                    final_text = f"{final_text}\n\n{notice}" if final_text else notice
                else:
                    notice = (
                        "I generated dashboard work, but could not put a runnable "
                        f"artifact on the canvas: {recovery_error}."
                    )
                    final_text = f"{final_text}\n\n{notice}" if final_text else notice
            if not final_text:
                final_text = "The requested sandbox work finished."
            self.messages.append({"role": "assistant", "content": final_text})
            self.dispatcher.record({"type": "assistant_message", "content": final_text})
            yield {"type": "assistant_message", "content": final_text}
            yield {"type": "done"}
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}
