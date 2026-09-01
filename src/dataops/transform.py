"""
transform — data transformation: time-column coercion, optional unit
conversion, timeline regularization, cond-feature engineering, robust scaler
stats, outlier reporting.

This wraps the canonical, pure preprocessor (`_preprocess_impl.preprocess_csv`,
copied verbatim from `helpers/preprocess.py`) so the transformation logic stays
**identical** to what produced the existing `prepared_<subset>/` bundles. The
preprocessor is end-to-end (it also writes the train/test split via
:mod:`dataops.split`'s primitives); use it when you want the full bundle in one
call, or use :func:`dataops.split.train_test` separately on an already
regularized frame.

Two surfaces:
  * :func:`preprocess_csv` — file/dir based, returns the meta dict (thin pass-through).
  * helper transforms (`coerce_time_column`, `regularize_timeline`, …) re-exported
    for callers that want to compose steps in-memory.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .profiling import profile
from . import _preprocess_impl as _impl

# Re-export the composable in-memory transform helpers for advanced callers.
from ._preprocess_impl import (  # noqa: F401
    add_gap_structure_features,
    add_time_features,
    analyze_outliers,
    coerce_time_column,
    compute_scaler_stats,
    preprocess_csv as _preprocess_csv_impl,
    regularize as regularize_timeline,
)

__all__ = [
    "preprocess",
    "preprocess_csv",
    "METADATA",
    # The composable helpers the module docstring advertises. They must be named
    # here or they do not survive `from dataops.transform import *`, which is
    # exactly how data_process_modules.transform re-exports this module — so the
    # documented surface was missing through the compat shim.
    "add_gap_structure_features",
    "add_time_features",
    "analyze_outliers",
    "coerce_time_column",
    "compute_scaler_stats",
    "regularize_timeline",
]


def _existing_columns(columns: Iterable[str], df: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in df.columns]


def _tabular_target_cols(
    df: pd.DataFrame,
    target_cols: Optional[List[str]],
) -> list[str]:
    if target_cols is not None:
        return _existing_columns(target_cols, df)
    return df.select_dtypes(include="number").columns.tolist()


def _numeric_target_cols(
    df: pd.DataFrame,
    target_cols: Optional[List[str]],
    timestamp_col: str,
) -> tuple[list[str], list[str]]:
    candidates = target_cols
    if candidates is None:
        candidates = [
            col for col in df.select_dtypes(include="number").columns
            if col != timestamp_col
        ]

    selected: list[str] = []
    dropped: list[str] = []
    for col in _existing_columns(candidates, df):
        if col == timestamp_col:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if pd.api.types.is_numeric_dtype(df[col]) or numeric.notna().any():
            selected.append(col)
        else:
            dropped.append(col)
    return selected, dropped


def _conditioning_columns(df: pd.DataFrame) -> list[str]:
    candidates = (
        "t_norm",
        "sin_day",
        "cos_day",
        "sin_hour",
        "cos_hour",
        "is_gap",
        "time_since_last_obs",
        "time_to_next_obs",
    )
    return [col for col in candidates if col in df.columns]


def preprocess(
    df: pd.DataFrame,
    timestamp_col: Optional[str] = None,
    *,
    target_cols: Optional[List[str]] = None,
    regularize: bool = True,
    add_cond_features: bool = True,
    base_dt=None,
    allow_step_index_timestamp: bool = False,
    sparse_skip_pct: float = 80.0,
    time_unit_seconds: Optional[float] = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """Preprocess an arbitrary tabular or time-series DataFrame in memory.

    This is the library-first API for orchestration systems. Tabular data is
    returned in a cleaned, type-aware form without inventing a timestamp. If a
    real timestamp is configured or detected, numeric target columns are
    time-sorted, optionally regularized, and augmented with conditioning
    features.
    """
    raw = df.copy()
    prof = profile(
        raw,
        timestamp_col=timestamp_col,
        allow_step_index_timestamp=allow_step_index_timestamp,
    )
    ts_col = prof["timestamp_column"]

    if not ts_col:
        selected_targets = _tabular_target_cols(raw, target_cols)
        meta = {
            "data_type": "tabular",
            "time_col": None,
            "target_cols": selected_targets,
            "cond_cols": [c for c in raw.columns if c not in selected_targets],
            "original_rows": int(len(raw)),
            "regularized_rows": int(len(raw)),
            "regularized": False,
            "notes": [prof["classification_reason"]],
        }
        return raw, meta

    work = raw.copy()
    coerced_time, time_kind = coerce_time_column(work[ts_col])
    work[ts_col] = coerced_time
    work = work.dropna(subset=[ts_col]).sort_values(ts_col)
    work = work.drop_duplicates(subset=[ts_col], keep="last")

    numeric_targets, dropped_targets = _numeric_target_cols(work, target_cols, ts_col)
    for col in numeric_targets:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # Regularization is intentionally limited to numeric targets. Non-numeric
    # columns stay out of the numeric grid path and are reported in metadata.
    transformed = work[[ts_col] + numeric_targets].copy()
    observed_row_mask = np.ones(len(transformed), dtype=bool)
    observed_col_mask = {c: transformed[c].notna().to_numpy() for c in numeric_targets}
    actual_base_dt = base_dt
    was_regularized = False
    regularization: dict = {"regularized": False, "skip_reason": "not requested"}
    if regularize and len(transformed) >= 2:
        (transformed, observed_row_mask, observed_col_mask,
         actual_base_dt, regularization) = _impl.regularize(
            transformed,
            time_col=ts_col,
            base_dt=base_dt,
            skip_if_sparse=True,
            sparse_skip_pct=sparse_skip_pct,
        )
        # Ask the regularizer whether it actually built the grid rather than
        # inferring it from a row count — the sparsity guard can return the
        # original irregular frame unchanged.
        was_regularized = bool(regularization["regularized"])

    cond_cols: list[str] = []
    if add_cond_features and len(transformed) > 0:
        transformed = add_time_features(
            transformed,
            time_col=ts_col,
            time_unit_seconds=1.0 if time_kind == "datetime" else time_unit_seconds,
        )
        transformed = add_gap_structure_features(transformed, observed_row_mask)
        cond_cols = _conditioning_columns(transformed)

    scaler_stats = (
        compute_scaler_stats(transformed, numeric_targets, observed_col_mask)
        if numeric_targets else {}
    )

    meta = {
        "data_type": "time_series",
        "time_col": ts_col,
        "time_kind": time_kind,
        "base_dt": actual_base_dt,
        "target_cols": numeric_targets,
        "cond_cols": cond_cols,
        "dropped_non_numeric_targets": dropped_targets,
        "original_rows": int(len(raw)),
        "regularized_rows": int(len(transformed)),
        "regularized": was_regularized,
        "regularization": regularization,
        "scaler": scaler_stats,
        "notes": [prof["classification_reason"]],
    }
    return transformed, meta


def preprocess_csv(
    input_csv: str,
    output_dir: str,
    *,
    time_col: Optional[str] = None,
    base_dt=None,
    split_ratio: float = 0.8,
    holdout_frac: float = 0.15,
    holdout_block_size: int = 5,
    seed: int = 0,
    add_cond_features: bool = True,
    convert_units: bool = False,
    extract_main_segment: bool = False,
    extract_all_segments_flag: bool = False,
    min_segment_length: int = 50,
    skip_regularize_if_sparse: bool = True,
    segment_regularization: bool = True,
    segment_gap_seconds: float = 86400.0,
    min_segment_rows: int = 32,
    require_all_segments: bool = True,
    gap_threshold: float = 1000.0,
    sparse_skip_pct: float = 80.0,
    time_unit_seconds: Optional[float] = 1.0,
    keep_categoricals: bool = True,
) -> dict:
    """End-to-end preprocessing → writes a full ``prepared/`` bundle, returns meta.

    Thin pass-through to the verbatim canonical implementation so output is
    bit-compatible with the existing pipeline. See module docstring for the
    in-memory alternative.
    """
    return _impl.preprocess_csv(
        input_csv=input_csv,
        output_dir=output_dir,
        time_col=time_col,
        base_dt=base_dt,
        split_ratio=split_ratio,
        holdout_frac=holdout_frac,
        holdout_block_size=holdout_block_size,
        seed=seed,
        add_cond_features=add_cond_features,
        convert_units=convert_units,
        extract_main_segment=extract_main_segment,
        extract_all_segments_flag=extract_all_segments_flag,
        min_segment_length=min_segment_length,
        skip_regularize_if_sparse=skip_regularize_if_sparse,
        segment_regularization=segment_regularization,
        segment_gap_seconds=segment_gap_seconds,
        min_segment_rows=min_segment_rows,
        require_all_segments=require_all_segments,
        gap_threshold=gap_threshold,
        sparse_skip_pct=sparse_skip_pct,
        time_unit_seconds=time_unit_seconds,
        keep_categoricals=keep_categoricals,
    )


METADATA = {
    "name": "transform",
    "version": "0.1.0",
    "category": "transform",
    "summary": "In-memory tabular/time-series preprocessing plus file-based prepared-bundle writer.",
    "entrypoint": "dataops.transform:preprocess",
    "gpu": False,
    "dependencies": ["pandas", "numpy"],
    "inputs": {
        "df": {"type": "DataFrame", "required": True},
        "timestamp_col": {"type": "str", "default": None},
        "target_cols": {"type": "list[str]", "default": None},
        "regularize": {"type": "bool", "default": True},
        "add_cond_features": {"type": "bool", "default": True},
        "base_dt": {"type": "float", "default": None},
        "allow_step_index_timestamp": {"type": "bool", "default": False},
        "sparse_skip_pct": {"type": "float", "default": 80.0},
        "time_unit_seconds": {"type": "float", "default": 1.0},
    },
    "outputs": {
        "df": {"type": "DataFrame", "schema": "preprocessed_dataframe"},
        "meta": {
            "type": "dict",
            "schema": "preprocess_meta",
            "keys": [
                "data_type", "time_col", "base_dt", "target_cols", "cond_cols",
                "original_rows", "regularized_rows", "regularized", "notes",
            ],
        },
    },
    "artifacts": [
        "meta.json", "scaler/", "col_masks/", "outlier_report.json",
        "train.csv", "test_input.csv", "test_gt.csv", "eval_holdout_mask.npy",
    ],
}
