"""
profiling — dataset profiling: time-series detection, primary-key detection,
column typing.

Extracted (and lightly optimized) from
``helpers/utils.py::analyze_csv_time_series_df`` and ``::detect_primary_key``.
Pure: takes a DataFrame, returns a JSON-serializable dict. No I/O, no Airflow.
"""
from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd

__all__ = [
    "analyze_time_series",
    "classify_dataset",
    "detect_primary_key",
    "profile",
    "METADATA",
]


def analyze_time_series(
    df: pd.DataFrame,
    configured_name: Optional[str] = None,
    sample_ratio: float = 0.9,
) -> dict:
    """Decide whether ``df`` is a time series and which column is the timestamp.

    Detection order: configured column → datetime-parseable string → Unix-epoch
    numeric → monotonic step index (weak signal).
    """
    result = {
        "is_time_series": False,
        "timestamp_column": None,
        "detected_type": "Not Time Series",
    }

    if configured_name and configured_name in df.columns:
        return {
            "is_time_series": True,
            "timestamp_column": configured_name,
            "detected_type": "Configured Timestamp Column",
        }

    step_candidates = []
    n = len(df)

    for col in df.columns:
        series = df[col]

        # A. datetime-parseable string column. Probing arbitrary columns with no
        # known format is expected to be noisy — silence the dateutil fallback
        # warning rather than leak it to callers.
        if series.dtype == "object":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() >= sample_ratio:
                return {
                    "is_time_series": True,
                    "timestamp_column": col,
                    "detected_type": "Datetime String",
                }

        # B / C. numeric: Unix epoch, or monotonic step index
        if pd.api.types.is_numeric_dtype(series):
            nonnull = series.dropna()
            if not nonnull.empty and 1e9 < float(nonnull.mean()) < 3e9:
                return {
                    "is_time_series": True,
                    "timestamp_column": col,
                    "detected_type": "Unix Timestamp",
                }
            if (
                series.is_monotonic_increasing
                and series.nunique() > n * 0.9
            ):
                step_candidates.append(col)

    if step_candidates:
        return {
            "is_time_series": True,
            "timestamp_column": step_candidates[0],
            "detected_type": "Step Index",
        }
    return result


def classify_dataset(
    df: pd.DataFrame,
    timestamp_col: Optional[str] = None,
    sample_ratio: float = 0.9,
    *,
    allow_step_index_timestamp: bool = False,
) -> dict:
    """Classify a dataframe as ``time_series`` or ``tabular``.

    ``analyze_time_series`` intentionally reports weak step-index signals for
    legacy callers. Classification is stricter: a monotonic integer id is common
    in tabular data, so step-index detection is treated as tabular unless the
    caller explicitly opts in.
    """
    ts = analyze_time_series(df, configured_name=timestamp_col, sample_ratio=sample_ratio)
    if not ts["is_time_series"]:
        return {
            "data_type": "tabular",
            "timestamp_column": None,
            "detected_type": ts["detected_type"],
            "classification_reason": "no timestamp-like column detected",
        }
    if ts["detected_type"] == "Step Index" and not allow_step_index_timestamp:
        return {
            "data_type": "tabular",
            "timestamp_column": None,
            "detected_type": ts["detected_type"],
            "classification_reason": (
                "monotonic step-index signal treated as tabular by default"
            ),
        }
    return {
        "data_type": "time_series",
        "timestamp_column": ts["timestamp_column"],
        "detected_type": ts["detected_type"],
        "classification_reason": f"time-series signal: {ts['detected_type']}",
    }


def detect_primary_key(
    df: pd.DataFrame,
    max_combo: int = 2,
    uniqueness_threshold: float = 0.999,
) -> dict:
    """Heuristic primary-key detection (single column, then 2-column composite)."""
    n = len(df)
    denom = max(n, 1)

    # 1. single-column candidates
    for col in df.columns:
        null_ratio = df[col].isna().mean()
        uniq_ratio = df[col].nunique(dropna=True) / denom
        if uniq_ratio >= uniqueness_threshold and null_ratio < 0.01:
            return {
                "type": "single",
                "columns": [col],
                "uniqueness_ratio": round(float(uniq_ratio), 4),
                "null_ratio": round(float(null_ratio), 4),
                "is_hard_pk": uniq_ratio > 0.999,
            }

    # 2. two-column composites
    if max_combo >= 2:
        cols = list(df.columns)
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                combo = df[[c1, c2]].dropna()
                uniq_ratio = len(combo.drop_duplicates()) / max(len(combo), 1)
                if uniq_ratio >= uniqueness_threshold:
                    return {
                        "type": "composite",
                        "columns": [c1, c2],
                        "uniqueness_ratio": round(float(uniq_ratio), 4),
                        "null_ratio": round(
                            float(max(df[c1].isna().mean(), df[c2].isna().mean())), 4
                        ),
                        "is_hard_pk": uniq_ratio > 0.999,
                    }

    return {
        "type": "none",
        "columns": [],
        "uniqueness_ratio": 0.0,
        "null_ratio": None,
        "is_hard_pk": False,
    }


def profile(
    df: pd.DataFrame,
    timestamp_col: Optional[str] = None,
    sample_ratio: float = 0.9,
    preview_rows: int = 5,
    allow_step_index_timestamp: bool = False,
) -> dict:
    """Full dataset profile combining TS detection, PK detection and typing.

    Mirrors the payload the DAG's ``load_raw_data`` pushed to XCom, minus the S3
    handle.
    """
    classification = classify_dataset(
        df,
        timestamp_col=timestamp_col,
        sample_ratio=sample_ratio,
        allow_step_index_timestamp=allow_step_index_timestamp,
    )
    pk = detect_primary_key(df)
    ts_col = classification["timestamp_column"]
    target_cols = [c for c in df.columns if c != ts_col]

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return {
        "data_type": classification["data_type"],
        "is_time_series": classification["data_type"] == "time_series",
        "timestamp_column": ts_col,
        "detected_type": classification["detected_type"],
        "classification_reason": classification["classification_reason"],
        "target_cols": target_cols,
        "primary_key": pk,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "shape": {"rows": int(len(df)), "cols": int(df.shape[1])},
        "columns": list(df.columns),
        "preview": df.head(preview_rows).to_dict(orient="records"),
    }


METADATA = {
    "name": "profiling",
    "version": "0.1.0",
    "category": "profiling",
    "summary": "Time-series detection, primary-key detection and column typing.",
    "entrypoint": "dataops.profiling:profile",
    "gpu": False,
    "dependencies": ["pandas", "numpy"],
    "inputs": {
        "df": {"type": "DataFrame", "required": True},
        "timestamp_col": {"type": "str", "default": None},
        "sample_ratio": {"type": "float", "default": 0.9},
        "preview_rows": {"type": "int", "default": 5},
        "allow_step_index_timestamp": {"type": "bool", "default": False},
    },
    "outputs": {
        "report": {
            "type": "dict",
            "schema": "profile",
            "keys": [
                "data_type", "is_time_series", "timestamp_column", "detected_type",
                "classification_reason", "target_cols", "primary_key", "numeric_cols",
                "categorical_cols", "shape", "columns", "preview",
            ],
        },
    },
    "artifacts": ["profile.json"],
}
