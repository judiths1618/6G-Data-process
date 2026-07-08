"""
tabular_checks — quality checks for non-time-series (tabular) data: missingness,
numeric outliers, primary-key awareness.

Extracted from the DAG ``qc()`` task. GX runs through the no-Airflow
:func:`dataops.gx.get_gx_context` shim. Pure: ``run(df) -> dict`` (field-compatible
with the old ``qc_result``).
"""
from __future__ import annotations

from uuid import uuid4
from typing import Optional

import pandas as pd

from .gx import (
    get_gx_context,
    summarize_gx as _summarize_gx,
    suppress_validator_result_format_warning,
)
from .profiling import detect_primary_key

__all__ = ["run", "METADATA"]


def run(
    df: pd.DataFrame,
    *,
    miss_threshold_numeric: float = 0.95,
    miss_threshold_cat: float = 0.90,
    outlier_q: float = 0.01,
    outlier_mostly: float = 0.95,
    missing_report_threshold: float = 0.05,
    min_unique_for_outlier: int = 10,
    pk_info: Optional[dict] = None,
    gx_context_root: Optional[str] = None,
) -> dict:
    """Run the tabular quality check suite.

    Returns a dict compatible with the DAG's ``qc_result``:
    ``{mode, gx_passed, missing_columns, outlier_columns, failed_columns,
    primary_key, recommendations, summary}``.
    """
    if pk_info is None:
        pk_info = detect_primary_key(df)
    pk_cols = pk_info.get("columns", [])

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    ctx = get_gx_context(gx_context_root)
    ds_name = "pipeline_tabular_source"
    try:
        ds = ctx.sources.add_pandas(name=ds_name)
    except Exception:
        ds = ctx.get_datasource(ds_name)
    asset = ds.add_dataframe_asset(name=f"tabular_asset_{uuid4().hex}", dataframe=df)
    batch_request = asset.build_batch_request()
    ctx.add_or_update_expectation_suite("tabular_quality")
    validator = ctx.get_validator(
        batch_request=batch_request, expectation_suite_name="tabular_quality"
    )

    with suppress_validator_result_format_warning():
        # --- missingness (soft) ----------------------------------------------
        missing_cols = []
        for col in df.columns:
            if col in pk_cols:
                continue
            mostly = miss_threshold_numeric if col in numeric_cols else miss_threshold_cat
            validator.expect_column_values_to_not_be_null(column=col, mostly=mostly)
            if df[col].isna().mean() > missing_report_threshold:
                missing_cols.append(col)

        # --- numeric sanity (skip constants / id-like) -----------------------
        outlier_cols = []
        for col in numeric_cols:
            if col in pk_cols or df[col].nunique() < min_unique_for_outlier:
                continue
            q_low = df[col].quantile(outlier_q)
            q_high = df[col].quantile(1 - outlier_q)
            if q_low == q_high:
                continue
            validator.expect_column_values_to_be_between(
                column=col, min_value=q_low, max_value=q_high,
                mostly=min(outlier_mostly, 1 - 2 * outlier_q),
            )
            outlier_cols.append(col)

        gx_result = validator.validate()

    failed_cols = {
        r["expectation_config"]["kwargs"].get("column")
        for r in gx_result["results"]
        if not r["success"]
    }

    recommendations = []
    if missing_cols:
        recommendations.append(
            f"Missing values detected in columns: {missing_cols}. "
            "Recommend tabular imputation."
        )
    if outlier_cols:
        recommendations.append(
            f"Potential outliers detected in numeric columns: {outlier_cols}."
        )
    if pk_info.get("type") == "none":
        recommendations.append(
            "No primary key detected. Treat as fact table; avoid row-wise imputation."
        )

    return {
        "mode": "tabular",
        "gx_passed": bool(gx_result["success"]),
        "gx": _summarize_gx(gx_result),
        "missing_columns": missing_cols,
        "outlier_columns": outlier_cols,
        "failed_columns": sorted(c for c in failed_cols if c is not None),
        "primary_key": pk_info,
        "recommendations": recommendations,
        "summary": {"rows": int(len(df)), "columns": df.columns.tolist()},
    }


METADATA = {
    "name": "tabular_checks",
    "version": "0.1.0",
    "category": "quality_check",
    "summary": "Tabular quality checks: missingness, numeric outliers, primary-key awareness.",
    "entrypoint": "dataops.tabular_checks:run",
    "gpu": False,
    "dependencies": ["pandas", "numpy", "great_expectations"],
    "inputs": {
        "df": {"type": "DataFrame", "required": True},
        "miss_threshold_numeric": {"type": "float", "default": 0.95},
        "miss_threshold_cat": {"type": "float", "default": 0.90},
        "outlier_q": {"type": "float", "default": 0.01},
        "outlier_mostly": {"type": "float", "default": 0.95},
        "missing_report_threshold": {"type": "float", "default": 0.05},
        "min_unique_for_outlier": {"type": "int", "default": 10},
        "pk_info": {"type": "dict", "default": None},
        "gx_context_root": {"type": "str", "default": None},
    },
    "outputs": {
        "report": {
            "type": "dict",
            "schema": "tabular_quality_report",
            "keys": [
                "mode", "gx_passed", "missing_columns", "outlier_columns",
                "failed_columns", "primary_key", "recommendations", "summary",
            ],
        },
    },
    "artifacts": ["qc_report.json"],
}
