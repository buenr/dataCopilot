from pathlib import Path

import pytest

from app.agent import Agent, MockProvider
from app.sandbox import FakeSandbox


@pytest.mark.asyncio
async def test_mock_dashboard_and_pdf_are_real_artifacts(tmp_path: Path):
    sandbox = FakeSandbox("test", tmp_path / "workspace")
    agent = Agent(sandbox, MockProvider(), "test", tmp_path / "sessions")
    dashboard_events = [event async for event in agent.turn("Build a dashboard")]
    assert any(event["type"] == "artifact" and event["name"] == "dashboard.html" for event in dashboard_events)
    pdf_events = [event async for event in agent.turn("Write a PDF report")]
    assert any(event["type"] == "artifact" and event["name"] == "report.pdf" for event in pdf_events)
    assert (tmp_path / "workspace" / "dashboard.html").read_text().startswith("<!doctype html>")
    assert (tmp_path / "workspace" / "report.pdf").read_bytes().startswith(b"%PDF")


@pytest.mark.asyncio
async def test_tool_dispatch_persists_state_and_trajectory(tmp_path: Path):
    sandbox = FakeSandbox("state", tmp_path / "workspace")
    agent = Agent(sandbox, MockProvider(), "state", tmp_path / "sessions")
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
