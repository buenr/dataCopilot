from pathlib import Path

from pathlib import Path

import pytest

from app.agent import PDF_RESCUE_CODE, Agent, MockProvider, ToolDispatcher
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
    assert any(event["type"] == "execution" and "1190.0" in event["stdout"] for event in events)
    assert any(event["type"] == "assistant_message" and "1190.0" in event["content"] for event in events)
    trajectory = (tmp_path / "sessions" / "loop" / "trajectory.jsonl").read_text()
    assert trajectory.count('"type": "tool_start"') == 1


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
    assert "no PDF artifact was found" in message["content"]


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
