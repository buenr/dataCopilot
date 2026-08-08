"""Small, deterministic dataframe ingestion and profiling helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import safe_path

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".parquet", ".json"}


def _strict_json(value: Any) -> Any:
    """Round-trip that yields strictly valid JSON.

    ``json.dumps`` happily emits bare ``NaN``/``Infinity`` tokens for non-finite
    floats, which browsers reject outright (``JSON.parse`` throws). Sparse
    survey-style sheets produce plenty of NaN cells, so every profile is
    normalized here: non-finite floats become ``None`` and non-serializable
    pandas/numpy scalars are stringified.
    """
    return json.loads(json.dumps(value, default=str), parse_constant=lambda _constant: None)


def profile_dataframe(frame: pd.DataFrame) -> dict[str, Any]:
    numeric = frame.select_dtypes(include="number")
    numeric_stats: dict[str, dict[str, float | None]] = {}
    for column in numeric.columns:
        values = numeric[column].dropna()
        numeric_stats[str(column)] = {
            "min": float(values.min()) if not values.empty else None,
            "max": float(values.max()) if not values.empty else None,
            "mean": float(values.mean()) if not values.empty else None,
            "median": float(values.median()) if not values.empty else None,
        }
    return _strict_json(
        {
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "dtypes": {str(name): str(dtype) for name, dtype in frame.dtypes.items()},
            "null_percentages": {
                str(name): float(frame[name].isna().mean() * 100) for name in frame.columns
            },
            "numeric_stats": numeric_stats,
            "sample_rows": frame.head(5).to_dict(orient="records"),
        }
    )


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"unsupported data format: {suffix or 'none'}")


def profile_files(root: str | Path, names: list[str] | None = None) -> list[dict[str, Any]]:
    """Load supported files in stable order and assign df_1, df_2, ... names."""
    root = Path(root)
    paths = []
    requested = names if names is not None else [p.name for p in root.iterdir() if p.is_file()]
    for name in sorted(requested):
        path = safe_path(root, name, allow_missing=False)
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            paths.append(path)
    summaries = []
    for index, path in enumerate(paths, start=1):
        frame = load_dataframe(path)
        summary = profile_dataframe(frame)
        summary.update({"name": f"df_{index}", "file": path.name})
        summaries.append(summary)
    return summaries
