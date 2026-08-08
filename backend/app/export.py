"""Export a session's analysis as a runnable Python script.

The trajectory log already records every run_python cell the agent executed;
this module stitches those cells into a standalone script the user can rerun
outside Data Copilot, preceded by a data-loading preamble that rebuilds the
df_1..df_N variables from the original upload names.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .agent import RESCUE_CODES

# Trajectory cells are compared stripped, so normalize the rescue snippets once.
_RESCUE_CELL_CODES = frozenset(code.strip() for code in RESCUE_CODES)

_READERS = {
    ".csv": "pd.read_csv",
    ".xls": "pd.read_excel",
    ".xlsx": "pd.read_excel",
    ".parquet": "pd.read_parquet",
    ".json": "pd.read_json",
}


def _preamble(profiles: list[dict[str, Any]]) -> list[str]:
    lines = [
        "from pathlib import Path",
        "",
        "import pandas as pd",
        "",
        "# Original uploads, loaded exactly as the session loaded them.",
        'DATA_DIR = Path(__file__).resolve().parent / "data"',
        "",
    ]
    for index, profile in enumerate(profiles, 1):
        filename = str(profile.get("file") or "")
        reader = _READERS.get(Path(filename).suffix.lower())
        if not filename or reader is None:
            continue
        variable = str(profile.get("name") or f"df_{index}")
        lines.append(f"{variable} = {reader}(DATA_DIR / {filename!r})")
    return lines


def build_analysis_script(
    trajectory_path: Path,
    profiles: list[dict[str, Any]],
    session_id: str,
) -> str:
    """Assemble a runnable .py from a session trajectory's run_python cells."""
    header = [
        "#!/usr/bin/env python3",
        '"""',
        "Data Copilot analysis script",
        f"Exported {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from session {session_id}.",
        "",
        "To repeat this analysis:",
        "  1. Place the original dataset files in a data/ folder next to this script.",
    ]
    for profile in profiles:
        filename = str(profile.get("file") or "")
        variable = str(profile.get("name") or "df_?")
        if filename:
            header.append(f"       data/{filename}  ->  {variable}")
    header += [
        "  2. Install the libraries the cells use (pandas, openpyxl, pyarrow, ...).",
        f"  3. Run: python data-copilot-analysis-{session_id[:8]}.py",
        "",
        "Cells appear in the order the agent ran them, grouped by turn. Files the",
        "agent wrote with write_file (web app sources) are not replayed here.",
        '"""',
        "",
    ]
    body = _preamble(profiles)
    cells = 0
    turn = 0
    previous_code = ""
    try:
        lines = trajectory_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "user_message":
            turn += 1
            prompt = str(event.get("content") or "").replace("\n", " ").strip()
            if len(prompt) > 70:
                prompt = prompt[:67] + "..."
            body += ["", f"# ==== Turn {turn}: {prompt!r} ====", ""]
            continue
        if event.get("type") != "tool_result" or event.get("tool") != "run_python":
            continue
        result = event.get("result") or {}
        code = str(result.get("code") or "").strip()
        # Artifact rescue snippets are internal plumbing, not analysis;
        # back-to-back retries of an identical cell add noise, not information.
        if not code or code in _RESCUE_CELL_CODES or code == previous_code:
            continue
        previous_code = code
        cells += 1
        stamp = str(event.get("timestamp") or "")
        body.append(f"# ---- Cell {cells} · {stamp} ----")
        if result.get("stderr") and not result.get("stdout"):
            body.append("# NOTE: this cell failed on its last recorded run; kept for fidelity.")
        body += [code, ""]
    if not cells:
        body += ["", "# No analysis code has run yet in this session.", ""]
    return "\n".join(header + body).rstrip() + "\n"
