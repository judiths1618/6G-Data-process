# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
from airflow.models import Variable

# Reuse IO + time utilities you already have
from helpers.dqc_utils import (
    load_df_from_minio, save_df_to_minio, _s3,
    detect_timestamp_column, normalize_ts_for_gap, compute_time_gaps_smart,
    build_schema_profile,
    PROJECT, DATASET_NAME, TARGET, S3_BUCKET,
    REPORT_PREFIX, CURATED_PREFIX,
    DEFAULT_TZ, TS_STD_COL, TS_EXPECTED_FREQ, TS_GAP_TOL_MULT, TS_GROUP_KEYS
)

# --------------------------------------------------------------------------------------
# CONFIG (can be overridden at runtime by Airflow Variable: N2N_DQC_CONFIG_JSON)
# --------------------------------------------------------------------------------------
# Example JSON you can put into Variable "N2N_DQC_CONFIG_JSON":
# {
#   "dataset_kind": "generic",              // or "kpi", "beam", ...
#   "tabular_metrics": ["completeness", "pk_duplicates", "numeric_outliers_iqr"],
#   "ts_metrics": ["completeness", "pk_duplicates", "numeric_outliers_iqr",
#                  "time_standardize", "time_gaps_adaptive"],
#   "ranges": {"RSRP": [-150,-40], "SINR": [-10,40], "BLER": [0,1]},
#   "primary_key": "id",                    // "col1,col2"
#   "group_keys": "site_id,device_id",
#   "expected_freq": "",                    // "1S" if known, else ""
#   "gap_tol_mult": 1.8
# }
#
DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset_kind": "generic",
    "tabular_metrics": ["completeness", "pk_duplicates", "numeric_outliers_iqr"],
    "ts_metrics": [
        "completeness", "pk_duplicates",
        "numeric_outliers_iqr",
        "time_standardize", "time_gaps_adaptive"
    ],
    "ranges": {},              # domain ranges (optional)
    "primary_key": "",         # comma-joined
    "group_keys": "",          # comma-joined
    "expected_freq": "",       # e.g. "1S"
    "gap_tol_mult": TS_GAP_TOL_MULT,
}

def _load_config() -> Dict[str, Any]:
    try:
        cfg = json.loads(Variable.get("N2N_DQC_CONFIG_JSON", default_var="{}"))
        base = DEFAULT_CONFIG.copy()
        base.update({k: v for k, v in cfg.items() if v is not None})
        return base
    except Exception:
        return DEFAULT_CONFIG.copy()

# --------------------------------------------------------------------------------------
# Context object passed to metrics
# --------------------------------------------------------------------------------------
def _split_cols(csv: str) -> List[str]:
    return [c.strip() for c in (csv or "").split(",") if c.strip()]

def make_context(df: pd.DataFrame, is_time_series: bool, profile: Dict[str, Any] | None) -> Dict[str, Any]:
    cfg = _load_config()
    pk_cols = _split_cols(cfg.get("primary_key", ""))

    # If PK not configured, try top candidate from profile
    if not pk_cols and profile:
        for cand in profile.get("primary_key_candidates", []):
            c = cand.get("column")
            if c in df.columns:
                pk_cols = [c]
                break

    ctx: Dict[str, Any] = {
        "cfg": cfg,
        "pk_cols": [c for c in pk_cols if c in df.columns],
        "group_keys": [c for c in _split_cols(cfg.get("group_keys", "")) if c in df.columns],
        "ranges": cfg.get("ranges", {}),
        "expected_freq": pd.to_timedelta(cfg["expected_freq"]) if cfg.get("expected_freq") else None,
        "gap_tol_mult": float(cfg.get("gap_tol_mult", TS_GAP_TOL_MULT)),
        "is_time_series": bool(is_time_series),
        "ts_col_detected": None,       # filled by time_standardize
        "ts_std_present": False,       # ditto
        "std_meta": {},
    }
    return ctx

# --------------------------------------------------------------------------------------
# Metric registry
# Each metric: fn(df: DataFrame, ctx: dict) -> dict payload with:
#   {
#     "name": "<metric_name>",
#     "ok": True|False|None,
#     "metrics": {...},     # numbers you want in the report
#     "actions": [ ... ],   # suggested downstream actions
#     "notes": [ ... ],     # free text
#   }
# --------------------------------------------------------------------------------------

MetricFn = Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, Any]]
REGISTRY: Dict[str, MetricFn] = {}

def metric(name: str) -> Callable[[MetricFn], MetricFn]:
    def _wrap(fn: MetricFn) -> MetricFn:
        REGISTRY[name] = fn
        return fn
    return _wrap

# -------------------- Tabular / common metrics --------------------

# @metric("completeness")
# def m_completeness(df: pd.DataFrame, ctx: Dict[str, Any]) -> Dict[str, Any]:
#     miss = {c: int(df[c].isna().sum()) for c in df.columns}
#     rate = float(df.isna().mean().mean())
#     return {
#         "name": "completeness", 
#         "ok": None,
#         "metrics": {"missing_rate": rate, "missing_by_col": miss},
#         "actions": ["impute_missing"] if any(v > 0 for v in miss.values()) else [],
#         "notes": [],
#     }
@metric("completeness")
def m_completeness(df: pd.DataFrame, ctx: Dict[str, Any]) -> Dict[str, Any]:
    if df.empty:
        return {
            "name": "completeness", "ok": None,
            "metrics": {"missing_rate": 1.0, "missing_by_col": {c: 0 for c in df.columns}},
            "actions": [],
            "notes": ["DataFrame is empty."],
        }

    miss = {c: int(df[c].isna().sum()) for c in df.columns}
    rate = float(df.isna().mean().mean())
    return {
        "name": "completeness", "ok": None,
        "metrics": {"missing_rate": rate, "missing_by_col": miss},
        "actions": ["impute_missing"] if any(v > 0 for v in miss.values()) else [],
        "notes": [],
    }


@metric("pk_duplicates")
def m_pk_duplicates(df: pd.DataFrame, ctx: Dict[str, Any]) -> Dict[str, Any]:
    pk = ctx.get("pk_cols") or []
    if pk:
        dup_pk = int(df.duplicated(subset=pk).sum())
        return {
            "name": "pk_duplicates",
            "ok": dup_pk == 0,
            "metrics": {"primary_key": pk, "duplicate_pk_rows": dup_pk},
            "actions": (["drop_duplicate_pk"] if dup_pk > 0 else []),
            "notes": [],
        }
    dup_all = int(df.duplicated().sum())
    return {
        "name": "pk_duplicates",
        "ok": dup_all == 0,
        "metrics": {"primary_key": [], "duplicate_rows_fallback": dup_all, "note": "No PK configured/detected."},
        "actions": (["drop_duplicates"] if dup_all > 0 else []),
        "notes": [],
    }

@metric("numeric_outliers_iqr")
def m_numeric_outliers_iqr(df: pd.DataFrame, ctx: Dict[str, Any]) -> Dict[str, Any]:
    outliers: Dict[str, int] = {}
    for c in df.select_dtypes(include=["number"]).columns:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty: 
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outliers[c] = int(((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum())
    total = int(sum(outliers.values()))
    return {
        "name": "numeric_outliers_iqr",
        "ok": None,
        "metrics": {"per_column": outliers, "total": total},
        "actions": (["clip_outliers_iqr"] if total > 0 else []),
        "notes": [],
    }

@metric("range_checks")
def m_range_checks(df: pd.DataFrame, ctx: Dict[str, Any]) -> Dict[str, Any]:
    rules = ctx.get("ranges") or {}
    viol = {}
    for k, bounds in rules.items():
        if k not in df.columns: 
            continue
        lo, hi = float(bounds[0]), float(bounds[1])
        s = pd.to_numeric(df[k], errors="coerce")
        bad = int(((s < lo) | (s > hi)).sum())
        viol[k] = {"range": [lo, hi], "out_of_range": bad}
    total = sum(v["out_of_range"] for v in viol.values()) if viol else 0
    return {
        "name": "range_checks",
        "ok": None if not rules else (total == 0),
        "metrics": viol,
        "actions": (["cap_ranges"] if total > 0 else []),
        "notes": [],
    }

# -------------------- Time-series metrics --------------------

@metric("time_standardize")
def m_time_standardize(df: pd.DataFrame, ctx: Dict[str, Any]) -> Dict[str, Any]:
    # detect & standardize only once per task run
    ts_detected, _ = detect_timestamp_column(df, configured_name=None)
    ctx["ts_col_detected"] = ts_detected
    if not ts_detected:
        return {
            "name": "time_standardize",
            "ok": False,
            "metrics": {"detected_ts_col": None},
            "actions": [],
            "notes": ["No timestamp column detected."],
        }

    df_std, std_meta = normalize_ts_for_gap(
        ti=None,  # not needed for normalization here (we skip anchor-by-ti)
        df=df, dataset_name=DATASET_NAME,
        configured_ts_col=ts_detected, out_col=TS_STD_COL,
    )
    ctx["ts_std_present"] = (TS_STD_COL in df_std.columns)
    ctx["std_meta"] = std_meta
    # return only metadata; caller can reuse df_std if needed (we avoid xcom huge)
    return {
        "name": "time_standardize",
        "ok": ctx["ts_std_present"],
        "metrics": {"detected_ts_col": ts_detected, "standardized_col": TS_STD_COL},
        "actions": [],
        "notes": [],
    }

@metric("time_gaps_adaptive")
def m_time_gaps_adaptive(df: pd.DataFrame, ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not ctx.get("ts_std_present"):
        return {
            "name": "time_gaps_adaptive", "ok": None,
            "metrics": {"skipped": True}, "actions": [], "notes": ["no standardized time"]
        }
    # Recompute standardized df quickly (tiny cost; avoids big state passing)
    ts_detected = ctx.get("ts_col_detected")
    df_std, _ = normalize_ts_for_gap(
        ti=None, df=df, dataset_name=DATASET_NAME,
        configured_ts_col=ts_detected, out_col=TS_STD_COL
    )
    expected = ctx.get("expected_freq") or (pd.to_timedelta(TS_EXPECTED_FREQ) if TS_EXPECTED_FREQ else None)
    tol_mult = float(ctx.get("gap_tol_mult", TS_GAP_TOL_MULT))
    results: Dict[str, Any] = {}
    missing_windows = missing_points = 0

    group_keys = ctx.get("group_keys") or TS_GROUP_KEYS
    if group_keys:
        for keys, g in df_std.groupby(group_keys, dropna=False, sort=False):
            r = compute_time_gaps_smart(
                df=g, ts_col=TS_STD_COL, expected_delta=expected,
                tol_mult=tol_mult, window=200, respect_calendar=True
            )
            results[str(keys)] = r
            missing_windows += r["counts"]["missing_windows"]
            missing_points  += r["counts"]["missing_points"]
        results["_summary"] = {"groups": len([k for k in results if k != "_summary"]),
                               "total_missing_windows": int(missing_windows),
                               "total_missing_points": int(missing_points)}
    else:
        r = compute_time_gaps_smart(
            df=df_std, ts_col=TS_STD_COL, expected_delta=expected,
            tol_mult=tol_mult, window=200, respect_calendar=True
        )
        results["_all"] = r
        missing_windows = r["counts"]["missing_windows"]
        missing_points  = r["counts"]["missing_points"]

    actions = []
    if (missing_windows > 0) or (missing_points > 0):
        actions.append("forward_fill_gaps")

    return {
        "name": "time_gaps_adaptive",
        "ok": (missing_windows == 0),
        "metrics": results,
        "actions": actions,
        "notes": [],
    }

# --------------------------------------------------------------------------------------
# Runner: executes a set of metrics and merges results into a unified QC block
# --------------------------------------------------------------------------------------
def run_metrics(df: pd.DataFrame, is_time_series: bool, profile: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Execute configured metrics for tabular or time-series.
    Returns a block shaped for your DQC report:
    {
      "checks": { "<metric_name>": <payload>, ... },
      "recommended_actions": [... (deduped, ordered) ...]
    }
    """
    cfg = _load_config()
    ctx = make_context(df, is_time_series=is_time_series, profile=profile)
    metric_names = (cfg["ts_metrics"] if is_time_series else cfg["tabular_metrics"])

    checks: Dict[str, Any] = {}
    actions_ordered: List[str] = []

    for name in metric_names:
        fn = REGISTRY.get(name)
        if not fn:
            continue
        try:
            payload = fn(df, ctx)
        except Exception as e:
            payload = {"name": name, "ok": None, "metrics": {}, "actions": [], "notes": [f"error: {e}"]}
        checks[name] = payload
        for act in payload.get("actions", []):
            if act not in actions_ordered:
                actions_ordered.append(act)

    return {"checks": checks, "recommended_actions": actions_ordered, "context": ctx}
