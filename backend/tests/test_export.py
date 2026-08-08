"""Tests for the analysis-script export built from session trajectories."""

import json
from pathlib import Path

from app.agent import PDF_RESCUE_CODE
from app.export import build_analysis_script


def _event(**fields):
    return json.dumps(fields)


def test_script_replays_cells_in_order_with_turn_headers(tmp_path: Path):
    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.write_text(
        "\n".join(
            [
                _event(type="user_message", content="analyze churn"),
                _event(
                    type="tool_result",
                    tool="run_python",
                    timestamp="t1",
                    result={"code": "print(df_1.shape)", "stdout": "(10, 2)", "stderr": ""},
                ),
                _event(type="tool_result", tool="write_file", result={"path": "app.py"}),
                _event(type="user_message", content="now a chart"),
                _event(
                    type="tool_result",
                    tool="run_python",
                    timestamp="t2",
                    result={"code": "df_1.plot()", "stdout": "", "stderr": "boom"},
                ),
            ]
        ),
        encoding="utf-8",
    )

    script = build_analysis_script(trajectory, [{"name": "df_1", "file": "hr.csv"}], "session123")

    assert "df_1 = pd.read_csv(DATA_DIR / 'hr.csv')" in script
    assert script.index("print(df_1.shape)") < script.index("df_1.plot()")
    assert "# ==== Turn 1: 'analyze churn' ====" in script
    assert "# ==== Turn 2: 'now a chart' ====" in script
    assert "app.py" not in script  # write_file sources are not replayed
    assert "failed on its last recorded run" in script  # stderr-only cell is flagged


def test_script_skips_rescue_code_and_consecutive_duplicates(tmp_path: Path):
    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.write_text(
        "\n".join(
            [
                _event(
                    type="tool_result",
                    tool="run_python",
                    result={"code": PDF_RESCUE_CODE, "stdout": "", "stderr": ""},
                ),
                _event(
                    type="tool_result",
                    tool="run_python",
                    result={"code": "x = 1", "stdout": "", "stderr": ""},
                ),
                _event(
                    type="tool_result",
                    tool="run_python",
                    result={"code": "x = 1", "stdout": "", "stderr": ""},
                ),
            ]
        ),
        encoding="utf-8",
    )

    script = build_analysis_script(trajectory, [], "session123")

    assert "SANDBOX_RESCUE_ROOTS" not in script
    assert script.count("x = 1") == 1


def test_script_without_cells_or_trajectory(tmp_path: Path):
    script = build_analysis_script(tmp_path / "missing.jsonl", [], "session123")
    assert "No analysis code has run yet" in script


def test_preamble_readers_match_upload_suffixes(tmp_path: Path):
    profiles = [
        {"name": "df_1", "file": "a.csv"},
        {"name": "df_2", "file": "b.parquet"},
        {"name": "df_3", "file": "c.xlsx"},
    ]

    script = build_analysis_script(tmp_path / "missing.jsonl", profiles, "s")

    assert "df_1 = pd.read_csv(DATA_DIR / 'a.csv')" in script
    assert "df_2 = pd.read_parquet(DATA_DIR / 'b.parquet')" in script
    assert "df_3 = pd.read_excel(DATA_DIR / 'c.xlsx')" in script


def test_exported_script_is_valid_python(tmp_path: Path):
    trajectory = tmp_path / "trajectory.jsonl"
    trajectory.write_text(
        _event(
            type="tool_result",
            tool="run_python",
            result={"code": "total = df_1['revenue'].sum()\nprint(total)", "stdout": "1", "stderr": ""},
        ),
        encoding="utf-8",
    )

    script = build_analysis_script(trajectory, [{"name": "df_1", "file": "sales.csv"}], "session123")

    compile(script, "exported.py", "exec")
