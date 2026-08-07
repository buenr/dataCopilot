from pathlib import Path

from pathlib import Path

import pytest

from app.agent import PDF_RESCUE_CODE, Agent, AnthropicProvider, MockProvider, ToolDispatcher
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
async def test_dashboard_fallback_contains_profiled_values(tmp_path: Path):
    sandbox = FakeSandbox("grounded", tmp_path / "workspace")
    agent = Agent(
        sandbox,
        DashboardPlaceholderProvider(),
        "grounded",
        tmp_path / "sessions",
        dataset_context=sales_profile,
    )

    try:
        events = [event async for event in agent.turn("Build a dashboard")]
        artifact = next(event for event in events if event["type"] == "artifact")
        content = (tmp_path / "workspace" / artifact["path"]).read_text()
    finally:
        await sandbox.close()

    assert artifact["name"] == "data_grounded_dashboard.html"
    assert artifact["port"] == 8502
    assert "4" in content
    assert "1,172.50" in content
    assert "950" in content
    assert "Key insights" in content
    assert "chart-row" in content


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


def test_dashboard_data_check_ignores_script_only_values():
    html = (
        "<h1>Sales</h1><script>const DATA = "
        '{"file":"sales.csv","rows":4,"revenue":950,"mean":1172.5}</script>'
    )

    assert not Agent._dashboard_has_visible_data(html, sales_profile())


def test_static_dashboard_preview_keeps_workspace_relative_html_path():
    command = "python3 -m http.server 8501 --directory /workspace"

    assert Agent._static_preview_path("data/hr_dashboard.html", command) == "data/hr_dashboard.html"
    assert Agent._static_preview_path("/workspace/hr_dashboard.html", command) == "hr_dashboard.html"


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
async def test_report_request_explains_why_no_pdf_reached_the_canvas(
    tmp_path: Path, monkeypatch
):
    sandbox = FakeSandbox("report-no-pdf", tmp_path / "workspace")
    # Point the fallback PDF writer at a missing directory so the write fails
    # and the turn has to explain itself with the real kernel error.
    monkeypatch.setenv("SANDBOX_WORKSPACE", str(tmp_path / "missing"))
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
    assert "fallback PDF generator failed" in message["content"]
    assert "FileNotFoundError" in message["content"]


def run_pdf_rescue(workspace: Path, working_directory: Path, monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_WORKSPACE", str(workspace))
    monkeypatch.chdir(working_directory)
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
async def test_fallback_pdf_generated_when_provider_omits_pdf(tmp_path: Path):
    """When the LLM analyzes data but never writes a PDF, the agent builds one."""
    sandbox = FakeSandbox("fallback-pdf", tmp_path / "workspace")
    agent = Agent(
        sandbox,
        AnalysisNoPdfProvider(),
        "fallback-pdf",
        tmp_path / "sessions",
    )

    try:
        events = [event async for event in agent.turn("Create a whitepaper with analysis")]
    finally:
        await sandbox.close()

    artifact = next(event for event in events if event["type"] == "artifact")
    assert artifact["name"] == "report.pdf"
    assert artifact["artifact"]["type"] == "pdf"
    assert any(
        event["type"] == "assistant_message"
        and "ready in the canvas" in event["content"]
        for event in events
    )
    # The fallback PDF must be a valid PDF file in the workspace.
    pdf_path = sandbox.root / "report.pdf"
    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF-")


class ClobberingNoPdfProvider:
    """Runs analysis, clobbers the WORKSPACE global, never writes a PDF."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            yield {
                "tool": "run_python",
                "arguments": {"code": "WORKSPACE = '/app'\nprint('rows: 150')"},
            }
        elif self.calls == 2:
            yield "Analysis complete."


@pytest.mark.asyncio
async def test_fallback_pdf_survives_clobbered_workspace(tmp_path: Path, monkeypatch):
    """Agent code can reassign WORKSPACE; the fallback must use the real one."""
    sandbox = FakeSandbox("clobber", tmp_path / "workspace")
    monkeypatch.setenv("SANDBOX_WORKSPACE", str(sandbox.root))
    agent = Agent(sandbox, ClobberingNoPdfProvider(), "clobber", tmp_path / "sessions")

    try:
        events = [event async for event in agent.turn("Create a report")]
    finally:
        await sandbox.close()

    artifact = next(event for event in events if event["type"] == "artifact")
    assert artifact["name"] == "report.pdf"
    assert (sandbox.root / "report.pdf").read_bytes().startswith(b"%PDF-")


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
async def test_fallback_pdf_generated_even_without_any_text(tmp_path: Path):
    """A report request must produce a PDF even when the provider says nothing."""
    sandbox = FakeSandbox("silent", tmp_path / "workspace")
    agent = Agent(sandbox, SilentNoPdfProvider(), "silent", tmp_path / "sessions")

    try:
        events = [event async for event in agent.turn("Create a report")]
    finally:
        await sandbox.close()

    artifact = next(event for event in events if event["type"] == "artifact")
    assert artifact["name"] == "report.pdf"
    assert (sandbox.root / "report.pdf").read_bytes().startswith(b"%PDF-")


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
