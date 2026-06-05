"""
transform — data transformation: time-column coercion, optional unit
conversion, timeline regularization, cond-feature engineering, robust scaler
stats, outlier reporting.

This wraps the canonical, pure preprocessor (`_preprocess_impl.preprocess_csv`,
copied verbatim from `helpers/preprocess.py`) so the transformation logic stays
**identical** to what produced the existing `prepared_<subset>/` bundles. The
preprocessor is end-to-end (it also writes the train/test split via
:mod:`pipeline_modules.split`'s primitives); use it when you want the full bundle in one
call, or use :func:`pipeline_modules.split.train_test` separately on an already
regularized frame.

Two surfaces:
  * :func:`preprocess_csv` — file/dir based, returns the meta dict (thin pass-through).
  * helper transforms (`coerce_time_column`, `regularize_timeline`, …) re-exported
    for callers that want to compose steps in-memory.
"""
from __future__ import annotations

from typing import List, Optional

from . import _preprocess_impl as _impl

# Re-export the composable in-memory transform helpers for advanced callers.
from ._preprocess_impl import (  # noqa: F401
    analyze_outliers,
    coerce_time_column,
    compute_scaler_stats,
    preprocess_csv as _preprocess_csv_impl,
)

__all__ = ["preprocess_csv", "METADATA"]


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
        gap_threshold=gap_threshold,
        sparse_skip_pct=sparse_skip_pct,
        time_unit_seconds=time_unit_seconds,
        keep_categoricals=keep_categoricals,
    )


METADATA = {
    "name": "transform",
    "version": "0.1.0",
    "category": "transform",
    "summary": "Time coercion, optional unit conversion, timeline regularization, "
               "cond-feature engineering and robust scaler stats (end-to-end "
               "preprocessor, bit-compatible with the existing pipeline).",
    "entrypoint": "pipeline_modules.transform:preprocess_csv",
    "gpu": False,
    "dependencies": ["pandas", "numpy"],
    "inputs": {
        "input_csv": {"type": "path", "required": True},
        "output_dir": {"type": "path", "required": True},
        "time_col": {"type": "str", "default": None},
        "split_ratio": {"type": "float", "default": 0.8},
        "holdout_frac": {"type": "float", "default": 0.15},
        "holdout_block_size": {"type": "int", "default": 5},
        "seed": {"type": "int", "default": 0},
        "add_cond_features": {"type": "bool", "default": True},
        "convert_units": {"type": "bool", "default": False},
        "skip_regularize_if_sparse": {"type": "bool", "default": True},
        "gap_threshold": {"type": "float", "default": 1000.0},
        "sparse_skip_pct": {"type": "float", "default": 80.0},
        "time_unit_seconds": {"type": "float", "default": 1.0},
        "keep_categoricals": {"type": "bool", "default": True},
    },
    "outputs": {
        "meta": {
            "type": "dict",
            "schema": "preprocess_meta",
            "keys": [
                "time_col", "base_dt", "target_cols", "cond_cols",
                "all_model_cols", "units_converted", "split_ratio",
                "holdout_frac", "original_rows", "regularized_rows",
                "train_rows", "test_rows", "clip_recommendation",
                "preprocessing_version", "notes",
            ],
        },
    },
    "artifacts": [
        "meta.json", "scaler/", "col_masks/", "outlier_report.json",
        "train.csv", "test_input.csv", "test_gt.csv", "eval_holdout_mask.npy",
    ],
}
