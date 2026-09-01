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

from .gx import (
    get_gx_context,
    summarize_gx as _summarize_gx,
    suppress_validator_result_format_warning,
)
from .timeline import (
    _within_run_diffs,
    detect_runs,
    diffs_in_seconds,
    estimate_cadence,
    infer_key_columns,
)

__all__ = [
    "detect_time_gaps",
    "inspect_timestamp_order",
    "infer_key_columns",
    "run",
    "METADATA",
]


# --------------------------------------------------------------------------- #
# Gap detection (ported verbatim from helpers/ts_utils.py)
# --------------------------------------------------------------------------- #

_diffs_in_seconds = diffs_in_seconds  # backward-compatible private alias


def detect_time_gaps(
    df: pd.DataFrame,
    ts_col: str,
    *,
    gap_factor: float = 1.5,
    min_gap_seconds: Optional[float] = None,
    run_id: Optional[np.ndarray] = None,
) -> dict:
    """Detect timeline irregularities in ``df[ts_col]``.

    Returns expected cadence, gap count/positions, and an estimate of how many
    grid points a regularized timeline would need to synthesize.

    ``expected_dt_seconds`` is the **median** positive step, which is what
    ``transform.preprocess`` uses to build the grid. The modal step is still
    reported under ``cadence`` — it is the estimator this function used to use,
    and it is unstable: timestamp collisions leave short residual steps that can
    win the mode on a ~10% plurality and inflate ``total_missing_rows`` by
    orders of magnitude.

    Pass ``run_id`` (from :func:`dataops.timeline.detect_runs`) so steps are
    never measured across an acquisition-run boundary.
    """
    info: dict = {
        "expected_dt_seconds": None,
        "num_gaps": 0,
        "has_gaps": False,
        "gap_pct": 0.0,
        "total_missing_rows": 0,
        "largest_gap_seconds": 0.0,
        "sample_gap_indices": [],
        "cadence": {},
        "notes": [],
    }

    ts = df[ts_col].dropna()
    if len(ts) < 2:
        info["notes"].append("fewer than 2 timestamps — cannot infer cadence")
        return info

    cadence = estimate_cadence(df, ts_col, run_id=run_id)
    info["cadence"] = cadence
    info["notes"].extend(cadence.get("notes", []))
    expected_dt = cadence.get("expected_dt_seconds")
    if not expected_dt or expected_dt <= 0:
        info["notes"].append("no positive-duration intervals")
        return info
    info["expected_dt_seconds"] = expected_dt

    diffs = _within_run_diffs(df, ts_col, run_id)
    diffs = diffs[diffs > 0]
    if diffs.empty:
        info["notes"].append("no positive-duration intervals")
        return info

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


def inspect_timestamp_order(
    df: pd.DataFrame,
    ts_col: str,
    *,
    key_columns: Optional[list] = None,
    sweep_aware: bool = False,
) -> dict:
    """Inspect timestamp contract issues without reordering the source frame.

    Backward steps are reported both as raw step counts and as **acquisition
    runs**: a single backward jump that is followed by a long ascending block
    overlapping the previous span is a restarted run, not one bad cell, and
    sorting it into the previous run interleaves two independent experiments.

    ``num_duplicate_timestamps`` counts collisions on the timestamp alone;
    ``num_duplicate_rows_on_key`` counts what survives after ``key_columns``,
    which is the number that actually justifies dropping rows.
    """
    ts = df[ts_col]
    diffs = diffs_in_seconds(ts)
    backward_steps = diffs[diffs < 0]
    runs = detect_runs(df, ts_col)

    info = {
        "is_monotonic_increasing": bool(ts.is_monotonic_increasing),
        "num_non_monotonic_steps": int(len(backward_steps)),
        "num_duplicate_timestamps": int(ts.duplicated().sum()),
        "num_null_timestamps": int(ts.isna().sum()),
        "sample_non_monotonic_indices": backward_steps.index[:5].tolist(),
        "sweep_aware": bool(sweep_aware),
        "num_runs": runs["num_runs"],
        "run_sizes": runs["run_sizes"],
        "run_boundaries": runs["boundaries"],
        "rows_out_of_order": runs["rows_out_of_order"],
        "has_overlapping_runs": runs["has_overlapping_runs"],
    }

    key_columns = [c for c in (key_columns or []) if c in df.columns]
    info["key_columns"] = key_columns
    if key_columns:
        info["num_duplicate_rows_on_key"] = int(
            df.duplicated(subset=[ts_col, *key_columns]).sum()
        )
    else:
        info["num_duplicate_rows_on_key"] = info["num_duplicate_timestamps"]
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
    key_columns: Optional[list] = None,
    sweep_aware: bool = False,
) -> dict:
    """Run the full time-series quality check suite.

    Returns a dict compatible with the DAG's ``ts_result``:
    ``{mode, gx_passed, issues{ts_gaps,missing,outliers}, recommendations, summary}``.

    ``key_columns`` names the columns that, together with ``ts_col``, identify a
    row. Pass ``None`` to infer them (see :func:`dataops.timeline.infer_key_columns`)
    or ``[]`` to force timestamp-only identity.
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

    with suppress_validator_result_format_warning():
        validator.expect_column_values_to_not_be_null(ts_col)
        validator.expect_column_values_to_be_unique(ts_col)
        validator.expect_column_values_to_be_increasing(ts_col)
        for col in gx_df.columns:
            validator.expect_column_values_to_not_be_null(
                column=col, mostly=miss_threshold
            )

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

    # --- timestamp integrity and gaps ----------------------------------------
    # Resolve the row-identity key first: in parameter-sweep datasets the swept
    # factors distinguish rows that share a timestamp, so "duplicate timestamp"
    # and "duplicate row" are different questions with very different answers.
    if key_columns is None:
        key_columns = (
            infer_key_columns(df, ts_col)["key_columns"] if sweep_aware else []
        )
    runs = detect_runs(df, ts_col)
    timestamp_order = inspect_timestamp_order(
        df, ts_col, key_columns=key_columns, sweep_aware=sweep_aware
    )
    ts_diag = detect_time_gaps(
        df,
        ts_col,
        gap_factor=gap_factor,
        min_gap_seconds=min_gap_seconds,
        run_id=runs["run_id"],
    )

    recommendations = {
        "ts_imputation": bool(ts_diag.get("has_gaps")),
        "tabular_imputation": bool(missing_info),
        "outlier_handling": bool(outlier_cols),
        # Only genuinely indistinguishable rows and null timestamps are
        # structural defects. Colliding timestamps that a key separates, and
        # backward jumps that are run boundaries, are handled by segmentation.
        "structural_fix": (
            not gx_passed
            or timestamp_order["num_duplicate_rows_on_key"] > 0
            or timestamp_order["num_null_timestamps"] > 0
        ),
        "run_segmentation": bool(timestamp_order["has_overlapping_runs"]),
    }

    return {
        "mode": "time_series",
        "gx_passed": gx_passed,
        "gx": gx_detail,
        "issues": {
            "timestamp_order": timestamp_order,
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
