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


def test_profile_files_assigns_sequential_names(tmp_path: Path):
    pd.DataFrame({"x": [1, 2]}).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame({"x": [3]}).to_json(tmp_path / "b.json", orient="records")
    result = profile_files(tmp_path)
    assert [item["name"] for item in result] == ["df_1", "df_2"]
