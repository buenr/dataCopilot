import os
import sys
import types
from pathlib import Path

import pytest
from app.agent import (
    MAX_TURN_STEPS,
    PDF_RESCUE_CODE,
    TOOL_DEFINITIONS,
    WRAP_UP_NUDGE,
    WRAP_UP_REMAINING_STEPS,
    Agent,
    AnthropicProvider,
    MockProvider,
    OpenAIProvider,
    ToolDispatcher,
)
from app.sandbox import FakeSandbox


@pytest.mark.asyncio
async def test_mock_dashboard_and_pdf_are_real_artifacts(tmp_path: Path):
    sandbox = FakeSandbox("test", tmp_path / "workspace")
    agent = Agent(sandbox, MockProvider(), "test", tmp_path / "sessions")
    try:
        dashboard_events = [event async for event in agent.turn("Build a dashboard")]
        assert any(event["type"] == "artifact" and event["name"] == "dashboard.html" for event in dashboard_events)
        pdf_events = [event async for event in agent.turn("Write a PDF report")]
        assert any(event["type"] == "artifact" and event["name"] == "report.pdf" for event in pdf_events)
    finally:
        await sandbox.close()
    assert (tmp_path / "workspace" / "dashboard.html").read_text().startswith("<!doctype html>")
    assert (tmp_path / "workspace" / "report.pdf").read_bytes().startswith(b"%PDF")


@pytest.mark.asyncio
async def test_tool_dispatch_persists_state_and_trajectory(tmp_path: Path):
    sandbox = FakeSandbox("state", tmp_path / "workspace")
    agent = Agent(sandbox, MockProvider(), "state", tmp_path / "sessions")
    try:
        result = await sandbox.exec("counter = 41")
        assert not result.stderr
        result = await sandbox.exec("counter += 1")
        assert not result.stderr
        await sandbox.upload("note.txt", b"hello")
        assert (await sandbox.read_file("note.txt")) == b"hello"
        assert (tmp_path / "sessions" / "state" / "trajectory.jsonl").exists() is False
        # A provider turn records its tool trajectory.
        [event async for event in agent.turn("Build a dashboard")]
        assert (tmp_path / "sessions" / "state" / "trajectory.jsonl").exists()
    finally:
        await sandbox.close()


@pytest.mark.asyncio
async def test_mock_executive_summary_uses_dataset_profile(tmp_path: Path):
    sandbox = FakeSandbox("summary", tmp_path / "workspace")
    profiles = [
        {
            "name": "df_1",
            "file": "sales.csv",
            "rows": 4,
            "columns": 4,
            "numeric_stats": {
                "revenue": {"min": 950.0, "max": 1430.0, "mean": 1172.5, "median": 1155.0},
            },
        }
    ]
    agent = Agent(
        sandbox,
        MockProvider(),
        "summary",
        tmp_path / "sessions",
        dataset_context=lambda: profiles,
    )

    events = [event async for event in agent.turn("Create an executive summary")]

    message = next(event["content"] for event in events if event["type"] == "assistant_message")
    assert "sales.csv" in message
    assert "4 rows" in message
    assert "1172.50" in message
    assert "950.0 to 1430.0" in message


class ToolThenTextProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            yield {
                "tool": "run_python",
                "call_id": "call-1",
                "arguments": {"code": "print(df_1['revenue'].mean())"},
            }
        else:
            assert any(message["role"] == "tool" for message in messages)
            yield "The calculated average is 1190.0."


@pytest.mark.asyncio
async def test_agent_returns_tool_results_to_provider_once(tmp_path: Path):
    sandbox = FakeSandbox("loop", tmp_path / "workspace")
    await sandbox.upload("data/sales.csv", b"revenue\n950\n1430\n")
    await sandbox.exec("import pandas as pd\ndf_1 = pd.read_csv(WORKSPACE / 'data/sales.csv')")
    provider = ToolThenTextProvider()
    agent = Agent(sandbox, provider, "loop", tmp_path / "sessions")

    events = [event async for event in agent.turn("Calculate the average revenue")]

    assert provider.calls == 2
    assert any(
        event["type"] == "execution" and event.get("status") == "running" and "1190.0" in event.get("data", "")
        for event in events
    )
    assert any(event["type"] == "assistant_message" and "1190.0" in event["content"] for event in events)
    trajectory = (tmp_path / "sessions" / "loop" / "trajectory.jsonl").read_text()
    assert trajectory.count('"type": "tool_start"') == 1
    # The closing message must land in the audit trail, not only on the wire.
    assert '"type": "assistant_message"' in trajectory
    assert "1190.0" in trajectory


class SetVariableProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            yield {"tool": "run_python", "call_id": "call-1", "arguments": {"code": "answer = 41"}}
        else:
            yield "Done."


@pytest.mark.asyncio
async def test_execution_event_includes_kernel_variables(tmp_path: Path):
    sandbox = FakeSandbox("vars", tmp_path / "workspace")
    agent = Agent(sandbox, SetVariableProvider(), "vars", tmp_path / "sessions")

    events = [event async for event in agent.turn("Set a variable")]

    execution = next(event for event in events if event["type"] == "execution" and event.get("status") == "complete")
    variables = {variable["name"]: variable for variable in execution["variables"]}
    assert variables["answer"]["type"] == "int"
    assert variables["answer"]["value"] == "41"
    assert all(not name.startswith("_") for name in variables)


class DashboardSourceOnlyProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            yield {
                "tool": "write_file",
                "arguments": {
                    "path": "app.py",
                    "content": (
                        "from http.server import HTTPServer, SimpleHTTPRequestHandler\n"
                        "HTTPServer(('0.0.0.0', 8501), SimpleHTTPRequestHandler).serve_forever()\n"
                    ),
                },
            }


@pytest.mark.asyncio
async def test_dashboard_source_is_recovered_when_provider_omits_handoff(tmp_path: Path):
    sandbox = FakeSandbox("recovery", tmp_path / "workspace")
    agent = Agent(
        sandbox,
        DashboardSourceOnlyProvider(),
        "recovery",
        tmp_path / "sessions",
    )

    try:
        events = [event async for event in agent.turn("Build a dashboard")]
    finally:
        await sandbox.close()

    artifact = next(event for event in events if event["type"] == "artifact")
    assert artifact["name"] == "app.py"
    assert artifact["port"] == 8501
    assert any(
        event["type"] == "assistant_message"
        and "running in the canvas" in event["content"]
        for event in events
    )
    assert not any(
        event["type"] == "assistant_message"
        and "Completed the requested sandbox work" in event["content"]
        for event in events
    )


class DashboardPdfOnlyProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            yield {
                "tool": "write_file",
                "arguments": {"path": "report.pdf", "content": "%PDF-1.4"},
            }
        elif self.calls == 2:
            yield {
                "tool": "register_artifact",
                "arguments": {"name": "report.pdf", "type": "pdf"},
            }


class PdfWithoutHandoffProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            yield {
                "tool": "write_file",
                "arguments": {"path": "generated-report.pdf", "content": "%PDF-1.4"},
            }


@pytest.mark.asyncio
async def test_root_file_listing_is_scoped_to_workspace(tmp_path: Path):
    sandbox = FakeSandbox("root-listing", tmp_path / "workspace")
    dispatcher = ToolDispatcher(sandbox, "root-listing", tmp_path / "sessions")

    try:
        result = await dispatcher.dispatch("list_files", {"path": "/"})
    finally:
        await sandbox.close()

    assert result["files"] == []


@pytest.mark.asyncio
async def test_report_pdf_is_recovered_when_provider_omits_handoff(tmp_path: Path):
    sandbox = FakeSandbox("report-recovery", tmp_path / "workspace")
    agent = Agent(
        sandbox,
        PdfWithoutHandoffProvider(),
        "report-recovery",
        tmp_path / "sessions",
    )

    try:
        events = [event async for event in agent.turn("Create a PDF report")]
    finally:
        await sandbox.close()

    artifact = next(event for event in events if event["type"] == "artifact")
    assert artifact["name"] == "generated-report.pdf"
    assert artifact["path"] == "generated-report.pdf"
    assert artifact["artifact"]["type"] == "pdf"
    assert any(
        event["type"] == "assistant_message"
        and "ready in the canvas" in event["content"]
        for event in events
    )


@pytest.mark.asyncio
async def test_dashboard_does_not_treat_pdf_as_canvas_artifact(tmp_path: Path):
    sandbox = FakeSandbox("pdf-only", tmp_path / "workspace")
    agent = Agent(
        sandbox,
        DashboardPdfOnlyProvider(),
        "pdf-only",
        tmp_path / "sessions",
    )

    events = [event async for event in agent.turn("Build a dashboard")]

    assert not any(event["type"] == "artifact" for event in events)
    assert any(
        event["type"] == "assistant_message"
        and "could not put a runnable artifact" in event["content"]
        for event in events
    )


class DashboardNestedHtmlProvider:
    async def stream(self, messages, tools):
        yield {
            "tool": "write_file",
            "arguments": {
                "path": "dashboard/index.html",
                "content": "<!doctype html><title>Sales</title>",
            },
        }
        yield {
            "tool": "start_webapp",
            "arguments": {
                "command": "python3 -m http.server 8501 --directory dashboard",
                "port": 8501,
            },
        }


@pytest.mark.asyncio
async def test_dashboard_static_subdirectory_path_is_rewritten_for_preview(tmp_path: Path):
    sandbox = FakeSandbox("nested-html", tmp_path / "workspace")
    agent = Agent(
        sandbox,
        DashboardNestedHtmlProvider(),
        "nested-html",
        tmp_path / "sessions",
    )

    try:
        events = [event async for event in agent.turn("Build a dashboard")]
    finally:
        await sandbox.close()

    artifact = next(event for event in events if event["type"] == "artifact")
    assert artifact["path"] == "index.html"
    assert artifact["port"] == 8501


class DashboardPlaceholderProvider:
    async def stream(self, messages, tools):
        yield {
            "tool": "write_file",
            "arguments": {
                "path": "dashboard.html",
                "content": (
                    "<!doctype html><h1>Sales dashboard</h1>"
                    "<p>Analysis complete</p><script>const DATA = [];</script>"
                ),
            },
        }
        yield {
            "tool": "start_webapp",
            "arguments": {
                "command": "python3 -m http.server 8501 --bind 0.0.0.0",
                "port": 8501,
            },
        }


class DashboardGroundedProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, messages, tools):
        self.calls += 1
        if self.calls > 1:
            yield "The grounded dashboard is ready."
            return
        yield {
            "tool": "write_file",
            "arguments": {
                "path": "dashboard.html",
                "content": (
                    "<!doctype html><h1>Sales dashboard</h1>"
                    "<p>sales.csv · 4 rows · revenue</p>"
                    "<p>Average revenue: 1,172.50. Range: 950 to 1,430.</p>"
                ),
            },
        }
        yield {
            "tool": "start_webapp",
            "arguments": {
                "command": "python3 -m http.server 8501 --bind 0.0.0.0",
                "port": 8501,
            },
        }
        yield {
            "tool": "register_artifact",
            "arguments": {"name": "dashboard.html", "type": "webapp", "port": 8501},
        }


def sales_profile() -> list[dict]:
    return [
        {
            "name": "df_1",
            "file": "sales.csv",
            "rows": 4,
            "columns": 2,
            "dtypes": {"revenue": "int64"},
            "numeric_stats": {
                "revenue": {"min": 950.0, "max": 1430.0, "mean": 1172.5, "median": 1155.0},
            },
            "sample_rows": [{"revenue": 950}],
        }
    ]


@pytest.mark.asyncio
async def test_placeholder_dashboard_is_served_as_written(tmp_path: Path):
    """An empty-shell dashboard is the model's own output: publish it, never swap it."""
    sandbox = FakeSandbox("placeholder", tmp_path / "workspace")
    agent = Agent(
        sandbox,
        DashboardPlaceholderProvider(),
        "placeholder",
        tmp_path / "sessions",
        dataset_context=sales_profile,
    )

    try:
        events = [event async for event in agent.turn("Build a dashboard")]
        artifact = next(event for event in events if event["type"] == "artifact")
        content = (tmp_path / "workspace" / artifact["path"]).read_text()
    finally:
        await sandbox.close()

    assert artifact["name"] == "dashboard.html"
    assert artifact["port"] == 8501
    assert "Analysis complete" in content
    assert not (tmp_path / "workspace" / "data_grounded_dashboard.html").exists()


@pytest.mark.asyncio
async def test_grounded_dashboard_is_preserved(tmp_path: Path):
    sandbox = FakeSandbox("already-grounded", tmp_path / "workspace")
    agent = Agent(
        sandbox,
        DashboardGroundedProvider(),
        "already-grounded",
        tmp_path / "sessions",
        dataset_context=sales_profile,
    )

    try:
        events = [event async for event in agent.turn("Build a dashboard")]
    finally:
        await sandbox.close()

    artifacts = [event for event in events if event["type"] == "artifact"]
    assert [artifact["name"] for artifact in artifacts] == ["dashboard.html"]
    assert artifacts[0]["port"] == 8501


def test_static_dashboard_preview_keeps_workspace_relative_html_path():
    command = "python3 -m http.server 8501 --directory /workspace"

    assert Agent._static_preview_path("data/hr_dashboard.html", command) == "data/hr_dashboard.html"
    assert Agent._static_preview_path("/workspace/hr_dashboard.html", command) == "hr_dashboard.html"


def test_static_dashboard_preview_strips_absolute_workspace_subdirectory():
    # http.server --directory re-roots the server: the preview path must be
    # relative to that directory, not to the workspace, or the proxy 404s.
    command = "python3 -m http.server 8501 --directory /workspace/attrition_dashboard"

    assert Agent._static_preview_path("attrition_dashboard/index.html", command) == "index.html"
    assert Agent._static_preview_path("/workspace/attrition_dashboard/index.html", command) == "index.html"


def test_static_server_ignores_provider_runtime_directory():
    command = "python3 -m http.server 8501 --directory /home/oai/share"

    assert ToolDispatcher._normalize_webapp_command(command) == "python3 -m http.server 8501"
    assert ToolDispatcher._normalize_webapp_command(
        "python3 -m http.server 8501 --directory /workspace"
    ) == "python3 -m http.server 8501 --directory /workspace"


class ReportWithoutPdfProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            yield {"tool": "run_python", "arguments": {"code": "analysis = 1"}}


@pytest.mark.asyncio
async def test_report_request_explains_why_no_pdf_reached_the_canvas(tmp_path: Path):
    sandbox = FakeSandbox("report-no-pdf", tmp_path / "workspace")
    agent = Agent(
        sandbox,
        ReportWithoutPdfProvider(),
        "report-no-pdf",
        tmp_path / "sessions",
    )

    try:
        events = [event async for event in agent.turn("Create a pdf whitepaper")]
    finally:
        await sandbox.close()

    assert not any(event["type"] == "artifact" for event in events)
    message = next(event for event in events if event["type"] == "assistant_message")
    assert "could not put a PDF on the canvas" in message["content"]
    assert "no PDF artifact was found in the sandbox workspace" in message["content"]


def run_pdf_rescue(
    workspace: Path,
    working_directory: Path,
    monkeypatch,
    scan_roots: tuple[Path, ...] = (),
) -> None:
    monkeypatch.setenv("SANDBOX_WORKSPACE", str(workspace))
    monkeypatch.chdir(working_directory)
    # Keep the scan hermetic: without this override the snippet would glob the
    # host's real /tmp and pick up unrelated PDFs from outside the test.
    roots = scan_roots or (working_directory,)
    monkeypatch.setenv("SANDBOX_RESCUE_ROOTS", os.pathsep.join(str(root) for root in roots))
    # A clobbered WORKSPACE global is exactly the failure the snippet must survive.
    exec(compile(PDF_RESCUE_CODE, "<rescue>", "exec"), {"WORKSPACE": "/app"})


def test_pdf_rescue_copies_working_directory_output_into_the_workspace(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    working_directory = tmp_path / "app"
    working_directory.mkdir()
    (working_directory / "whitepaper.pdf").write_bytes(b"%PDF-1.4 generated")

    run_pdf_rescue(workspace, working_directory, monkeypatch)

    assert (workspace / "whitepaper.pdf").read_bytes() == b"%PDF-1.4 generated"


def test_pdf_rescue_leaves_workspace_files_untouched(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    (workspace / "reports").mkdir(parents=True)
    (workspace / "reports" / "whitepaper.pdf").write_bytes(b"%PDF-1.4 generated")

    run_pdf_rescue(workspace, workspace / "reports", monkeypatch)

    assert not (workspace / "whitepaper.pdf").exists()
    assert sorted(path.name for path in workspace.rglob("*.pdf")) == ["whitepaper.pdf"]


def test_pdf_rescue_ignores_pdfs_outside_the_configured_scan_roots(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    working_directory = tmp_path / "app"
    working_directory.mkdir()
    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / "scratch.pdf").write_bytes(b"%PDF-1.4 unrelated")

    run_pdf_rescue(workspace, working_directory, monkeypatch)

    assert not (workspace / "scratch.pdf").exists()
    assert list(workspace.rglob("*.pdf")) == []


def test_anthropic_history_maps_openai_tool_flow_to_content_blocks():
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "analyze the data"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "run_python", "arguments": "{\"code\": \"print(1)\"}"},
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "list_files", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "run_python", "content": "{\"stdout\": \"1\"}"},
        {"role": "tool", "tool_call_id": "call-2", "name": "list_files", "content": "{\"files\": []}"},
        {"role": "assistant", "content": "The answer is 1."},
    ]

    converted = AnthropicProvider._convert_messages(messages)

    assert converted[0] == {"role": "user", "content": "analyze the data"}
    assistant = converted[1]
    assert assistant["role"] == "assistant"
    assert [block["type"] for block in assistant["content"]] == ["tool_use", "tool_use"]
    assert assistant["content"][0]["id"] == "call-1"
    assert assistant["content"][0]["input"] == {"code": "print(1)"}
    tool_results = converted[2]
    assert tool_results["role"] == "user"
    assert [block["type"] for block in tool_results["content"]] == ["tool_result", "tool_result"]
    assert [block["tool_use_id"] for block in tool_results["content"]] == ["call-1", "call-2"]
    assert converted[3] == {"role": "assistant", "content": "The answer is 1."}


@pytest.mark.asyncio
async def test_mock_provider_wraps_up_after_tool_results(tmp_path: Path):
    sandbox = FakeSandbox("mock-loop", tmp_path / "workspace")

    class CountingMockProvider(MockProvider):
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, messages, tools):
            self.calls += 1
            async for item in super().stream(messages, tools):
                yield item

    provider = CountingMockProvider()
    agent = Agent(sandbox, provider, "mock-loop", tmp_path / "sessions")
    try:
        events = [event async for event in agent.turn("build a dashboard")]
    finally:
        await sandbox.close()

    # One tool round plus one wrap-up call; previously the tool result payload
    # re-matched "dashboard" and re-ran every tool until the turn cap.
    assert provider.calls == 2
    starts = [
        event
        for event in events
        if event.get("type") == "tool_start" and event.get("tool") == "run_python"
    ]
    assert len(starts) == 1


@pytest.mark.asyncio
async def test_run_with_retries_does_not_repeat_code_that_produced_output(tmp_path: Path):
    sandbox = FakeSandbox("retry", tmp_path / "workspace")
    dispatcher = ToolDispatcher(sandbox, "retry", tmp_path / "sessions")

    execution = await dispatcher._run_with_retries(
        "import sys\n"
        "with open(WORKSPACE / 'marker.txt', 'a') as handle:\n"
        "    handle.write('x')\n"
        "print('done')\n"
        "sys.stderr.write('DeprecationWarning: noisy library\\n')\n"
    )

    assert execution.stdout.strip() == "done"
    # A warning on stderr must not cause side-effecting code to run again.
    assert (sandbox.root / "marker.txt").read_text() == "x"


@pytest.mark.asyncio
async def test_run_with_retries_still_retries_bare_failures(tmp_path: Path):
    sandbox = FakeSandbox("retry-bare", tmp_path / "workspace")
    dispatcher = ToolDispatcher(sandbox, "retry-bare", tmp_path / "sessions")

    execution = await dispatcher._run_with_retries(
        "with open(WORKSPACE / 'attempts.txt', 'a') as handle:\n"
        "    handle.write('x')\n"
        "raise RuntimeError('boom')\n"
    )

    assert "RuntimeError" in execution.stderr
    assert (sandbox.root / "attempts.txt").read_text() == "xxxx"  # initial run + 3 retries


class PrintAndSetProvider:
    """One round: run_python that prints and sets a variable, then wrap-up text."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            yield {"tool": "run_python", "call_id": "call-1", "arguments": {"code": "answer = 42\nprint(answer)"}}
        else:
            yield "Done."


@pytest.mark.asyncio
async def test_streaming_events_emit_code_then_deltas_then_complete_without_stdout(tmp_path: Path):
    sandbox = FakeSandbox("stream-order", tmp_path / "workspace")
    agent = Agent(sandbox, PrintAndSetProvider(), "stream-order", tmp_path / "sessions")

    events = [event async for event in agent.turn("Compute and print")]
    exec_events = [event for event in events if event["type"] == "execution"]

    # 1. An initial "running" event publishes the code before the cell runs.
    assert exec_events[0]["status"] == "running"
    assert "answer = 42" in exec_events[0].get("code", "")

    # 2. Streaming deltas carry the printed output.
    deltas = [event for event in exec_events if event.get("stream") and event.get("data")]
    assert len(deltas) >= 1
    assert any("42" in event["data"] for event in deltas)

    # 3. The "complete" event carries variables but not the already-streamed stdout.
    complete = next(event for event in exec_events if event.get("status") == "complete")
    assert "variables" in complete
    assert any(v["name"] == "answer" for v in complete["variables"])
    assert "stdout" not in complete
    assert "stderr" not in complete


def test_report_request_detects_whitepaper_with_long_parenthetical():
    """Regression: a 31-char gap between verb and 'whitepaper' was not detected."""
    assert Agent._is_report_request(
        "create a (insightful but funny tone) whitepaper with analysis of how each player did"
    )
    assert Agent._is_report_request("create a whitepaper")
    assert Agent._is_report_request("write a PDF report")
    assert Agent._is_report_request("generate a report")
    assert not Agent._is_report_request("build a dashboard")
    assert not Agent._is_report_request("what does the report say")


class AnalysisNoPdfProvider:
    """Provider that runs analysis but never writes a PDF or calls register_artifact."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            yield {
                "tool": "run_python",
                "arguments": {"code": "print('Player A: 28 pts, 12 ast\\nPlayer B: 24 pts, 8 reb')"},
            }
        elif self.calls == 2:
            yield "Analysis complete. Player A leads in scoring and assists."


@pytest.mark.asyncio
async def test_report_without_pdf_is_explained_not_fabricated(tmp_path: Path):
    """When the LLM analyzes data but never writes a PDF, no fake is published."""
    sandbox = FakeSandbox("no-pdf", tmp_path / "workspace")
    agent = Agent(
        sandbox,
        AnalysisNoPdfProvider(),
        "no-pdf",
        tmp_path / "sessions",
    )

    try:
        events = [event async for event in agent.turn("Create a whitepaper with analysis")]
    finally:
        await sandbox.close()

    assert not any(event["type"] == "artifact" for event in events)
    assert not (sandbox.root / "report.pdf").exists()
    assert any(
        event["type"] == "assistant_message"
        and "could not put a PDF on the canvas" in event["content"]
        for event in events
    )


class SilentNoPdfProvider:
    """One tool call with no output, then no further content at all."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            yield {"tool": "run_python", "arguments": {"code": "x = 1"}}
        # Round two yields nothing: no text, no tool calls.


@pytest.mark.asyncio
async def test_silent_report_turn_is_explained_not_fabricated(tmp_path: Path):
    """A report request publishes nothing when the provider never wrote a PDF."""
    sandbox = FakeSandbox("silent", tmp_path / "workspace")
    agent = Agent(sandbox, SilentNoPdfProvider(), "silent", tmp_path / "sessions")

    try:
        events = [event async for event in agent.turn("Create a report")]
    finally:
        await sandbox.close()

    assert not any(event["type"] == "artifact" for event in events)
    assert not (sandbox.root / "report.pdf").exists()
    assert any(
        event["type"] == "assistant_message"
        and "could not put a PDF on the canvas" in event["content"]
        for event in events
    )


@pytest.mark.asyncio
async def test_mock_explore_request_returns_grounded_summary(tmp_path: Path):
    """The welcome screen's 'Explore my data' suggestion must not dead-end."""
    sandbox = FakeSandbox("explore", tmp_path / "workspace")
    profiles = [
        {
            "name": "df_1",
            "file": "sales.csv",
            "rows": 4,
            "columns": 2,
            "numeric_stats": {},
        }
    ]
    agent = Agent(
        sandbox,
        MockProvider(),
        "explore",
        tmp_path / "sessions",
        dataset_context=lambda: profiles,
    )

    try:
        events = [event async for event in agent.turn("Explore my data")]
    finally:
        await sandbox.close()

    message = next(event["content"] for event in events if event["type"] == "assistant_message")
    assert "sales.csv" in message
    assert "I can analyze" not in message


@pytest.mark.asyncio
async def test_mock_chart_request_registers_image_artifact(tmp_path: Path):
    sandbox = FakeSandbox("chart", tmp_path / "workspace")
    await sandbox.upload("data/sales.csv", b"region,revenue\nNorth,1200\nSouth,950\n")
    await sandbox.exec("import pandas as pd\ndf_1 = pd.read_csv(WORKSPACE / 'data/sales.csv')")
    agent = Agent(sandbox, MockProvider(), "chart", tmp_path / "sessions")

    try:
        events = [event async for event in agent.turn("Show me a chart of the data")]
    finally:
        await sandbox.close()

    artifact = next(event for event in events if event["type"] == "artifact")
    assert artifact["name"] == "chart.svg"
    assert artifact["artifact"]["type"] == "image"
    svg = (tmp_path / "workspace" / "chart.svg").read_text()
    assert svg.startswith("<svg")
    assert "revenue" in svg


def test_register_artifact_tool_accepts_data_type():
    schema = next(tool for tool in TOOL_DEFINITIONS if tool["function"]["name"] == "register_artifact")
    assert "data" in schema["function"]["parameters"]["properties"]["type"]["enum"]


@pytest.mark.asyncio
async def test_mock_excel_request_registers_data_artifact(tmp_path: Path):
    sandbox = FakeSandbox("excel", tmp_path / "workspace")
    await sandbox.upload("data/sales.csv", b"region,revenue\nNorth,1200\nSouth,950\n")
    await sandbox.exec("import pandas as pd\ndf_1 = pd.read_csv(WORKSPACE / 'data/sales.csv')")
    agent = Agent(sandbox, MockProvider(), "excel", tmp_path / "sessions")

    try:
        events = [event async for event in agent.turn("Export the summary to Excel")]
    finally:
        await sandbox.close()

    artifact = next(event for event in events if event["type"] == "artifact")
    assert artifact["name"] == "summary.xlsx"
    assert artifact["artifact"]["type"] == "data"
    # An .xlsx is a zip container; the workbook must actually exist.
    assert (tmp_path / "workspace" / "summary.xlsx").read_bytes()[:2] == b"PK"


class _EmptyStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _FakeResponsesAPI:
    """Captures Responses API requests and returns an empty stream."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def create(self, **request):
        self.requests.append(request)
        return _EmptyStream()


def _install_fake_openai(monkeypatch) -> _FakeResponsesAPI:
    fake = _FakeResponsesAPI()
    module = types.SimpleNamespace(AsyncOpenAI=lambda api_key: types.SimpleNamespace(responses=fake))
    monkeypatch.setitem(sys.modules, "openai", module)
    return fake


@pytest.mark.asyncio
async def test_openai_provider_uses_xhigh_reasoning_effort(monkeypatch):
    fake = _install_fake_openai(monkeypatch)
    provider = OpenAIProvider("key", "gpt-5.6-luna")

    _ = [item async for item in provider.stream([{"role": "user", "content": "hi"}], [])]

    assert fake.requests[0]["reasoning"] == {"effort": "xhigh"}


class _BothArtifactsProvider:
    """Registers a dashboard and a PDF across two rounds, then wraps up."""

    async def stream(self, messages, tools):
        tool_results = sum(1 for m in messages if m.get("role") == "tool")
        if tool_results == 0:
            yield {
                "tool": "register_artifact",
                "arguments": {"name": "dashboard.html", "type": "webapp", "port": 8501},
                "call_id": "c1",
            }
        elif tool_results == 1:
            yield {
                "tool": "register_artifact",
                "arguments": {"name": "report.pdf", "type": "pdf"},
                "call_id": "c2",
            }
        else:
            yield "Both artifacts are ready."


@pytest.mark.asyncio
async def test_dashboard_and_report_turn_publishes_both_artifacts(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "dashboard.html").write_text("<!doctype html><p>data</p>")
    (workspace / "report.pdf").write_bytes(b"%PDF-1.4 fake")
    sandbox = FakeSandbox("both", workspace)
    agent = Agent(sandbox, _BothArtifactsProvider(), "both", tmp_path / "sessions")

    try:
        events = [event async for event in agent.turn("Build a dashboard and write a PDF report")]
    finally:
        await sandbox.close()

    # A combined dashboard+report request must publish each artifact to the
    # canvas; the per-request gates used to filter out both.
    names = [event.get("name") for event in events if event["type"] == "artifact"]
    assert "dashboard.html" in names
    assert "report.pdf" in names


class _SubdirectoryDashboardProvider:
    """Serves a dashboard from a subdirectory and registers the directory name."""

    async def stream(self, messages, tools):
        tool_results = sum(1 for m in messages if m.get("role") == "tool")
        if tool_results == 0:
            yield {
                "tool": "run_python",
                "arguments": {
                    "code": (
                        "from pathlib import Path\n"
                        "site = Path(WORKSPACE) / 'site'\n"
                        "site.mkdir(exist_ok=True)\n"
                        "(site / 'index.html').write_text('<!doctype html><p>4 rows</p>')"
                    )
                },
                "call_id": "c1",
            }
        elif tool_results == 1:
            yield {
                "tool": "start_webapp",
                "arguments": {
                    "command": "python3 -m http.server 8501 --directory /workspace/site",
                    "port": 8501,
                },
                "call_id": "c2",
            }
        elif tool_results == 2:
            yield {
                "tool": "register_artifact",
                "arguments": {"name": "site", "type": "webapp", "port": 8501},
                "call_id": "c3",
            }
        else:
            yield "The dashboard is live."


@pytest.mark.asyncio
async def test_dashboard_registered_by_directory_publishes_served_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The live OpenAI model did exactly this: register_artifact with the
    served directory name returned no artifacts, so nothing published until
    recovery stepped in — with a workspace-relative path that 404s behind a
    server rooted at the subdirectory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sandbox = FakeSandbox("subdir", workspace)

    async def fake_start_webapp(command: str, port: int) -> int:
        sandbox.preview_ports[port] = port
        return port

    monkeypatch.setattr(sandbox, "start_webapp", fake_start_webapp)
    agent = Agent(sandbox, _SubdirectoryDashboardProvider(), "subdir", tmp_path / "sessions")

    try:
        events = [event async for event in agent.turn("Build a dashboard")]
    finally:
        await sandbox.close()

    artifacts = [event for event in events if event["type"] == "artifact"]
    assert len(artifacts) == 1
    assert artifacts[0]["path"] == "index.html"
    assert artifacts[0]["port"] == 8501


class _BadReadProvider:
    """Calls read_file on a missing file, then wraps up after the error."""

    async def stream(self, messages, tools):
        if any(m.get("role") == "tool" for m in messages):
            yield "Recovered from the failed read."
            return
        yield {"tool": "read_file", "arguments": {"path": "missing.txt"}, "call_id": "bad"}


@pytest.mark.asyncio
async def test_tool_failure_returns_error_result_instead_of_killing_turn(tmp_path: Path):
    sandbox = FakeSandbox("err", tmp_path / "workspace")
    agent = Agent(sandbox, _BadReadProvider(), "err", tmp_path / "sessions")

    try:
        events = [event async for event in agent.turn("Read the missing file")]
    finally:
        await sandbox.close()

    tool_results = [event for event in events if event["type"] == "tool_result"]
    assert tool_results and "error" in tool_results[0]
    assert any(event["type"] == "assistant_message" for event in events)
    assert not any(event["type"] == "error" for event in events)
    assert events[-1]["type"] == "done"


class _DashboardOnlyProvider:
    """Registers just the dashboard on a combined dashboard+report prompt."""

    async def stream(self, messages, tools):
        if any(m.get("role") == "tool" for m in messages):
            yield "Dashboard done."
            return
        yield {
            "tool": "register_artifact",
            "arguments": {"name": "dashboard.html", "type": "webapp", "port": 8501},
            "call_id": "c1",
        }


@pytest.mark.asyncio
async def test_combined_turn_reports_the_missing_pdf_honestly(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "dashboard.html").write_text("<!doctype html><p>data</p>")
    sandbox = FakeSandbox("combo", workspace)
    agent = Agent(sandbox, _DashboardOnlyProvider(), "combo", tmp_path / "sessions")

    try:
        events = [event async for event in agent.turn("Build a dashboard and write a PDF report")]
    finally:
        await sandbox.close()

    names = [event.get("name") for event in events if event["type"] == "artifact"]
    assert "dashboard.html" in names
    # Publishing the dashboard must not fabricate a PDF that was never written.
    assert not any(str(name).endswith(".pdf") for name in names)
    assert any(
        event["type"] == "assistant_message"
        and "could not put a PDF on the canvas" in event["content"]
        for event in events
    )


class EndlessExplorerProvider:
    """Calls run_python forever, never writing or registering an artifact."""

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[list[dict]] = []

    async def stream(self, messages, tools):
        self.calls += 1
        self.requests.append([dict(message) for message in messages])
        yield {"tool": "run_python", "arguments": {"code": f"print({self.calls})"}}


@pytest.mark.asyncio
async def test_turn_stops_after_sixteen_rounds_and_says_so(tmp_path: Path):
    provider = EndlessExplorerProvider()
    sandbox = FakeSandbox("cap", tmp_path / "workspace")
    agent = Agent(sandbox, provider, "cap", tmp_path / "sessions")

    try:
        events = [event async for event in agent.turn("Analyze the data")]
    finally:
        await sandbox.close()

    assert provider.calls == MAX_TURN_STEPS
    message = next(event for event in events if event["type"] == "assistant_message")
    assert "ran out of steps" in message["content"]
    assert "continue" in message["content"]


@pytest.mark.asyncio
async def test_wrap_up_nudge_reaches_the_provider_before_the_cap(tmp_path: Path):
    provider = EndlessExplorerProvider()
    sandbox = FakeSandbox("nudge", tmp_path / "workspace")
    agent = Agent(sandbox, provider, "nudge", tmp_path / "sessions")

    try:
        [event async for event in agent.turn("Analyze the data")]
    finally:
        await sandbox.close()

    assert f"only {WRAP_UP_REMAINING_STEPS} tool rounds remain" in WRAP_UP_NUDGE
    before = provider.requests[: MAX_TURN_STEPS - WRAP_UP_REMAINING_STEPS]
    nudged = provider.requests[MAX_TURN_STEPS - WRAP_UP_REMAINING_STEPS]
    assert not any(WRAP_UP_NUDGE in str(message) for call in before for message in call)
    assert any(
        message.get("role") == "tool" and WRAP_UP_NUDGE in str(message.get("content", ""))
        for message in nudged
    )


@pytest.mark.asyncio
async def test_turn_step_events_track_every_round_and_the_nudge(tmp_path: Path):
    provider = EndlessExplorerProvider()
    sandbox = FakeSandbox("progress", tmp_path / "workspace")
    agent = Agent(sandbox, provider, "progress", tmp_path / "sessions")

    try:
        events = [event async for event in agent.turn("Analyze the data")]
    finally:
        await sandbox.close()

    # The chat status line ("round N of M · …") is built from these events, so
    # every provider round must announce itself, in order, against the cap.
    steps = [event for event in events if event["type"] == "turn_step"]
    assert [event["step"] for event in steps] == list(range(1, MAX_TURN_STEPS + 1))
    assert all(event["max_steps"] == MAX_TURN_STEPS for event in steps)
    nudges = [event for event in events if event["type"] == "wrap_up_nudge"]
    assert [event["step"] for event in nudges] == [
        MAX_TURN_STEPS - WRAP_UP_REMAINING_STEPS + 1
    ]


@pytest.mark.asyncio
async def test_exhausted_report_turn_says_so_instead_of_faking_a_pdf(tmp_path: Path):
    provider = EndlessExplorerProvider()
    sandbox = FakeSandbox("exhausted", tmp_path / "workspace")
    agent = Agent(sandbox, provider, "exhausted", tmp_path / "sessions")

    try:
        events = [event async for event in agent.turn("Create a PDF report")]
    finally:
        await sandbox.close()

    assert not any(event["type"] == "artifact" for event in events)
    assert not (sandbox.root / "report.pdf").exists()
    message = next(event for event in events if event["type"] == "assistant_message")
    assert "ran out of steps" in message["content"]
    assert "continue" in message["content"]


@pytest.mark.asyncio
async def test_exhausted_dashboard_turn_says_so_instead_of_faking_html(tmp_path: Path):
    provider = EndlessExplorerProvider()
    sandbox = FakeSandbox("exhausted-dash", tmp_path / "workspace")
    agent = Agent(sandbox, provider, "exhausted-dash", tmp_path / "sessions")

    try:
        events = [event async for event in agent.turn("Build a dashboard")]
    finally:
        await sandbox.close()

    assert not any(event["type"] == "artifact" for event in events)
    assert not (sandbox.root / "data_grounded_dashboard.html").exists()
    message = next(event for event in events if event["type"] == "assistant_message")
    assert "ran out of steps" in message["content"]
    assert "continue" in message["content"]
