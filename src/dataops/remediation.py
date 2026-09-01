"""
remediation — apply a concrete fix for each detected quality issue, *after* the
quality checks have run.

This closes the loop between detection and cleaning. The conservative first-pass
``cleaning.clean_dataframe`` runs *before* the checks (column names, empty rows,
duplicates, datetime coercion); ``remediate`` runs *after* and acts on the
issues the checks actually found:

  * numeric **outliers** → *reported by default*, and winsorized only when
    ``clip_outliers=True``. Clipping rewrites real measurements, so it is
    opt-in: the default records how many cells sit outside the band (and in
    which columns) and leaves the values alone;
  * ordinary **missing** values → type-aware fill, **tabular mode only**
    (numeric → median, categorical → mode);
  * **time-series gaps / target missingness** → *not* filled here. Those are
    deferred to the imputation handoff (``regularize`` the timeline, then route
    to an imputation app), because filling them is the imputation step's job.

Pure: ``remediate(df, quality_report, ...) -> (df_out, RemediationReport)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

__all__ = ["RemediationReport", "remediate", "METADATA"]


@dataclass
class RemediationReport:
    """Summary of the fixes applied per issue family."""

    mode: str
    actions: list[dict] = field(default_factory=list)
    missing_cells_before: int = 0
    missing_cells_after: int = 0
    outlier_cells_clipped: int = 0
    #: Cells outside the quantile band. Equals ``outlier_cells_clipped`` when
    #: clipping is on; when it is off this is the count that *would* be clipped.
    outlier_cells_flagged: int = 0


def _missing_cells(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())


def _scan_outliers(df: pd.DataFrame, cols: list[str], outlier_q: float,
                   *, apply: bool) -> tuple[int, list[str], list[dict]]:
    """Count cells outside the ``[outlier_q, 1 - outlier_q]`` band per column.

    With ``apply=True`` the columns are winsorized to that band in place; with
    ``apply=False`` nothing is modified and the same counts are returned, so a
    report can state exactly what clipping *would* have changed.
    """
    total = 0
    hit_cols: list[str] = []
    detail: list[dict] = []
    for col in cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        lower = df[col].quantile(outlier_q)
        upper = df[col].quantile(1 - outlier_q)
        if pd.isna(lower) or pd.isna(upper) or lower == upper:
            continue
        below = int((df[col] < lower).sum())
        above = int((df[col] > upper).sum())
        if below + above == 0:
            continue
        if apply:
            df[col] = df[col].clip(lower=lower, upper=upper)
        total += below + above
        hit_cols.append(col)
        detail.append({"column": col, "cells": below + above,
                       "below": below, "above": above,
                       "lower": float(lower), "upper": float(upper)})
    return total, hit_cols, detail


def _outlier_action(df: pd.DataFrame, cols: list[str], outlier_q: float,
                    *, clip: bool) -> dict | None:
    """The remediation action for numeric outliers, or None if there are none."""
    cells, hit_cols, detail = _scan_outliers(df, cols, outlier_q, apply=clip)
    if not hit_cols:
        return None
    action = {
        "issue": "numeric_outliers",
        "action": "winsorize" if clip else "report_only",
        "columns": hit_cols,
        "detail": detail,
        "quantile_band": [outlier_q, 1 - outlier_q],
        "status": "applied" if clip else "reported_not_applied",
    }
    if clip:
        action["clipped_cells"] = cells
    else:
        action["flagged_cells"] = cells
        action["would_clip_cells"] = cells
        action["solution"] = (
            "Values outside the quantile band are reported, not modified. "
            "Set validation.clip_outliers: true (or pass clip_outliers=True) to "
            "winsorize them to the band."
        )
    return action


def _fill_missing_tabular(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """Type-aware fill: numeric → median, otherwise → mode."""
    filled: list[dict] = []
    for col in cols:
        if col not in df.columns or df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            value = df[col].median()
            strategy = "median"
        else:
            modes = df[col].mode(dropna=True)
            if modes.empty:
                continue
            value = modes.iloc[0]
            strategy = "mode"
        n = int(df[col].isna().sum())
        df[col] = df[col].fillna(value)
        filled.append({"column": col, "strategy": strategy, "filled_cells": n})
    return filled


def remediate(
    df: pd.DataFrame,
    quality_report: dict,
    *,
    outlier_q: float = 0.01,
    fill_tabular_missing: bool = True,
    clip_outliers: bool = False,
) -> tuple[pd.DataFrame, RemediationReport]:
    """Apply per-issue fixes and return ``(df_out, report)``.

    ``quality_report`` is the dict produced by :mod:`dataops.ts_checks` or
    :mod:`dataops.tabular_checks`. Time-series gaps are intentionally left for
    the imputation handoff (see module docstring).

    ``clip_outliers`` is off by default: winsorizing rewrites real measurements,
    and on benchmark data a genuine latency spike is signal, not noise. The
    outliers are always *detected and reported* either way — only the rewrite is
    optional.
    """
    mode = quality_report.get("mode") or "unknown"
    out = df.copy()
    report = RemediationReport(mode=mode, missing_cells_before=_missing_cells(out))

    if mode == "time_series":
        outlier_cols = list(quality_report.get("issues", {}).get("outliers", []))
        action = _outlier_action(out, outlier_cols, outlier_q, clip=clip_outliers)
        if action:
            report.actions.append(action)
        gaps = quality_report.get("issues", {}).get("ts_gaps", {})
        if gaps.get("has_gaps"):
            report.actions.append({
                "issue": "time_gaps",
                "action": "regularize_then_impute",
                "detail": {
                    "num_gaps": gaps.get("num_gaps", 0),
                    "missing_rows_estimate": gaps.get("total_missing_rows", 0),
                },
                "status": "deferred_to_imputation",
            })
        missing = quality_report.get("issues", {}).get("missing", {})
        if missing:
            report.actions.append({
                "issue": "missing_values",
                "action": "leave_for_imputation",
                "columns": sorted(missing.keys()),
                "status": "deferred_to_imputation",
            })

    elif mode == "tabular":
        outlier_cols = list(quality_report.get("outlier_columns", []))
        action = _outlier_action(out, outlier_cols, outlier_q, clip=clip_outliers)
        if action:
            report.actions.append(action)
        missing_cols = list(quality_report.get("missing_columns", []))
        if missing_cols and fill_tabular_missing:
            filled = _fill_missing_tabular(out, missing_cols)
            if filled:
                report.actions.append({
                    "issue": "missing_values",
                    "action": "type_aware_fill",
                    "detail": filled,
                    "status": "applied",
                })
        elif missing_cols:
            report.actions.append({
                "issue": "missing_values",
                "action": "type_aware_fill",
                "columns": missing_cols,
                "status": "skipped_by_config",
            })

    report.outlier_cells_clipped = sum(
        a.get("clipped_cells", 0) for a in report.actions
    )
    report.outlier_cells_flagged = sum(
        a.get("clipped_cells", a.get("flagged_cells", 0)) for a in report.actions
    )
    report.missing_cells_after = _missing_cells(out)
    return out, report


METADATA = {
    "name": "remediation",
    "version": "0.1.0",
    "category": "cleaning",
    "summary": "Per-issue remediation after quality checks: report (optionally "
               "winsorize) outliers, type-aware tabular fill; defers time-series "
               "gaps to imputation.",
    "entrypoint": "dataops.remediation:remediate",
    "gpu": False,
    "dependencies": ["pandas"],
    "inputs": {
        "df": {"type": "DataFrame", "required": True},
        "quality_report": {"type": "dict", "required": True},
        "outlier_q": {"type": "float", "default": 0.01},
        "fill_tabular_missing": {"type": "bool", "default": True},
        "clip_outliers": {"type": "bool", "default": False},
    },
    "outputs": {
        "df": {"type": "DataFrame", "schema": "remediated_dataframe"},
        "report": {
            "type": "dict",
            "schema": "remediation_report",
            "keys": ["mode", "actions", "missing_cells_before",
                     "missing_cells_after", "outlier_cells_clipped",
                     "outlier_cells_flagged"],
        },
    },
    "artifacts": [],
}
