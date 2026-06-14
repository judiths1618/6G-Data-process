"""
ts_checks — time-series quality checks: timestamp integrity, gaps, missingness,
outliers.

Extracted from the DAG ``ts_qc()`` task + ``helpers/ts_utils.py::detect_time_gaps``.
GX runs through the no-Airflow :func:`dataops.gx.get_gx_context` shim. Pure:
``run(df, ts_col) -> dict`` (field-compatible with the old ``ts_result``).
"""
from __future__ import annotations

from uuid import uuid4
from typing import Optional

import numpy as np
import pandas as pd

from .gx import get_gx_context, summarize_gx as _summarize_gx

__all__ = ["detect_time_gaps", "run", "METADATA"]


# --------------------------------------------------------------------------- #
# Gap detection (ported verbatim from helpers/ts_utils.py)
# --------------------------------------------------------------------------- #

def _diffs_in_seconds(ts: pd.Series) -> pd.Series:
    """diff(ts) in seconds whether ts is datetime64, epoch s, or epoch ms."""
    if pd.api.types.is_datetime64_any_dtype(ts):
        return ts.diff().dropna().dt.total_seconds()

    s = pd.to_numeric(ts, errors="coerce").dropna()
    diffs = s.diff().dropna()
    if len(diffs) == 0:
        return diffs
    med = float(s.median())
    if 1e12 <= med < 1e14:  # milliseconds since epoch
        return diffs / 1000.0
    return diffs


def detect_time_gaps(
    df: pd.DataFrame,
    ts_col: str,
    *,
    gap_factor: float = 1.5,
    min_gap_seconds: Optional[float] = None,
) -> dict:
    """Detect timeline irregularities in ``df[ts_col]``.

    Returns expected cadence, gap count/positions, and an estimate of how many
    grid points a regularized timeline would need to synthesize.
    """
    info: dict = {
        "expected_dt_seconds": None,
        "num_gaps": 0,
        "has_gaps": False,
        "gap_pct": 0.0,
        "total_missing_rows": 0,
        "largest_gap_seconds": 0.0,
        "sample_gap_indices": [],
        "notes": [],
    }

    ts = df[ts_col].dropna().sort_values()
    if len(ts) < 2:
        info["notes"].append("fewer than 2 timestamps — cannot infer cadence")
        return info

    diffs = _diffs_in_seconds(ts)
    diffs = diffs[diffs > 0]
    if diffs.empty:
        info["notes"].append("no positive-duration intervals")
        return info

    modes = diffs.mode()
    expected_dt = float(modes.iloc[0]) if len(modes) else float(diffs.median())
    if expected_dt <= 0:
        info["notes"].append("non-positive modal interval; falling back to median")
        expected_dt = float(diffs.median())
    info["expected_dt_seconds"] = expected_dt

    threshold = min_gap_seconds if min_gap_seconds is not None else expected_dt * gap_factor
    flagged = diffs[diffs > threshold]
    info["num_gaps"] = int(len(flagged))
    info["has_gaps"] = info["num_gaps"] > 0
    info["sample_gap_indices"] = flagged.index[:5].tolist()
    info["largest_gap_seconds"] = float(diffs.max())

    if info["has_gaps"]:
        extra = ((flagged / expected_dt) - 1).clip(lower=0).round().astype(int)
        info["total_missing_rows"] = int(extra.sum())

    grid_rows = int(np.ceil((float(diffs.sum()) + expected_dt) / expected_dt))
    if grid_rows > 0:
        info["gap_pct"] = max(0.0, 1.0 - len(ts) / grid_rows)
    return info


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _normalize_dtype(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def _gx_safe_dataframe(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Return a copy with timestamp values represented safely for GX checks."""
    gx_df = df.copy()
    if pd.api.types.is_datetime64_any_dtype(gx_df[ts_col]):
        gx_df[ts_col] = (gx_df[ts_col] - pd.Timestamp("1970-01-01")).dt.total_seconds()
    return gx_df


# --------------------------------------------------------------------------- #
# Public check
# --------------------------------------------------------------------------- #

def run(
    df: pd.DataFrame,
    ts_col: str,
    *,
    gap_factor: float = 1.5,
    min_gap_seconds: Optional[float] = None,
    miss_threshold: float = 0.98,
    outlier_q: float = 0.01,
    outlier_mostly: float = 0.95,
    gx_context_root: Optional[str] = None,
) -> dict:
    """Run the full time-series quality check suite.

    Returns a dict compatible with the DAG's ``ts_result``:
    ``{mode, gx_passed, issues{ts_gaps,missing,outliers}, recommendations, summary}``.
    """
    if ts_col not in df.columns:
        raise KeyError(f"timestamp column {ts_col!r} not in dataframe")

    # --- GX structural checks ------------------------------------------------
    gx_df = _gx_safe_dataframe(df, ts_col)
    ctx = get_gx_context(gx_context_root)
    ds_name = "pipeline_ts_source"
    try:
        ds = ctx.sources.add_pandas(name=ds_name)
    except Exception:
        ds = ctx.get_datasource(ds_name)
    asset = ds.add_dataframe_asset(name=f"ts_asset_{uuid4().hex}", dataframe=gx_df)
    batch_request = asset.build_batch_request()
    ctx.add_or_update_expectation_suite(expectation_suite_name="ts_quality")
    validator = ctx.get_validator(
        batch_request=batch_request, expectation_suite_name="ts_quality"
    )

    validator.expect_column_values_to_not_be_null(ts_col)
    validator.expect_column_values_to_be_unique(ts_col)
    validator.expect_column_values_to_be_increasing(ts_col)
    for col in gx_df.columns:
        validator.expect_column_values_to_not_be_null(column=col, mostly=miss_threshold)

    # numeric range expectations (also drives the outlier list)
    numeric_cols = [
        c for c in gx_df.select_dtypes(include="number").columns if c != ts_col
    ]
    outlier_cols = []
    for col in numeric_cols:
        lower = gx_df[col].quantile(outlier_q)
        upper = gx_df[col].quantile(1 - outlier_q)
        # ``mostly`` gets margin over the ~2*outlier_q tail the quantile band
        # inevitably leaves outside, so this stays a soft sentinel instead of
        # tripping by construction on every continuous column.
        validator.expect_column_values_to_be_between(
            column=col, min_value=lower, max_value=upper,
            mostly=min(outlier_mostly, 1 - 2 * outlier_q),
        )
        if (gx_df[col] < lower).any() or (gx_df[col] > upper).any():
            outlier_cols.append(col)

    gx_result = validator.validate()
    gx_passed = bool(gx_result["success"])
    gx_detail = _summarize_gx(gx_result)

    # --- type-aware missingness ---------------------------------------------
    missing_info = {}
    for col in df.columns:
        miss_ratio = df[col].isna().mean()
        if miss_ratio > 0:
            missing_info[col] = {
                "dtype": _normalize_dtype(df[col]),
                "missing_ratio": round(float(miss_ratio), 4),
            }

    # --- gaps ----------------------------------------------------------------
    ts_diag = detect_time_gaps(
        df, ts_col, gap_factor=gap_factor, min_gap_seconds=min_gap_seconds
    )

    recommendations = {
        "ts_imputation": bool(ts_diag.get("has_gaps")),
        "tabular_imputation": bool(missing_info),
        "outlier_handling": bool(outlier_cols),
        "structural_fix": not gx_passed,
    }

    return {
        "mode": "time_series",
        "gx_passed": gx_passed,
        "gx": gx_detail,
        "issues": {
            "ts_gaps": ts_diag,
            "missing": missing_info,
            "outliers": sorted(set(outlier_cols)),
        },
        "recommendations": recommendations,
        "summary": {
            "total_records": int(len(df)),
            "start_date": str(df[ts_col].min()),
            "end_date": str(df[ts_col].max()),
        },
    }


METADATA = {
    "name": "ts_checks",
    "version": "0.1.0",
    "category": "quality_check",
    "summary": "Time-series quality checks: timestamp integrity, gaps, missingness, outliers.",
    "entrypoint": "dataops.ts_checks:run",
    "gpu": False,
    "dependencies": ["pandas", "numpy", "great_expectations"],
    "inputs": {
        "df": {"type": "DataFrame", "required": True},
        "ts_col": {"type": "str", "required": True},
        "gap_factor": {"type": "float", "default": 1.5},
        "min_gap_seconds": {"type": "float", "default": None},
        "miss_threshold": {"type": "float", "default": 0.98},
        "outlier_q": {"type": "float", "default": 0.01},
        "outlier_mostly": {"type": "float", "default": 0.95},
        "gx_context_root": {"type": "str", "default": None},
    },
    "outputs": {
        "report": {
            "type": "dict",
            "schema": "ts_quality_report",
            "keys": ["mode", "gx_passed", "issues", "recommendations", "summary"],
        },
    },
    "artifacts": ["qc_report.json"],
}
