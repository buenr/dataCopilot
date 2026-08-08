import json
from pathlib import Path

import pandas as pd
import pytest
from app.profiling import profile_dataframe, profile_files


def test_profile_dataframe_has_schema_and_stats():
    summary = profile_dataframe(pd.DataFrame({"amount": [1, 3, None], "label": ["a", "b", "c"]}))
    assert summary["rows"] == 3
    assert summary["columns"] == 2
    assert summary["null_percentages"]["amount"] == pytest.approx(100 / 3)
    assert summary["numeric_stats"]["amount"]["max"] == 3


def test_profile_dataframe_emits_strict_json_for_sparse_frames():
    # Sparse survey-style sheets carry NaN cells into the sample rows; bare
    # NaN/Infinity tokens are not valid JSON and browsers refuse to parse any
    # frame containing them, so profiles must normalize them to null.
    frame = pd.DataFrame(
        {
            "rating": [5, None, 3],
            "comment": ["good", None, "bad"],
            "empty_score": pd.Series([float("nan"), float("nan"), float("nan")]),
            "runaway": [float("inf"), 2.0, 1.0],
        }
    )
    summary = profile_dataframe(frame)
    # allow_nan=False raises on any bare NaN/Infinity token.
    json.dumps(summary, allow_nan=False)
    assert summary["sample_rows"][1]["rating"] is None
    assert summary["sample_rows"][1]["comment"] is None
    # All-NaN numeric columns have no stats; infinities are nulled too.
    assert summary["numeric_stats"]["empty_score"]["mean"] is None
    assert summary["numeric_stats"]["runaway"]["mean"] is None


def test_profile_files_assigns_sequential_names(tmp_path: Path):
    pd.DataFrame({"x": [1, 2]}).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame({"x": [3]}).to_json(tmp_path / "b.json", orient="records")
    result = profile_files(tmp_path)
    assert [item["name"] for item in result] == ["df_1", "df_2"]
