"""Minimal DataOps pipeline: clean, validate, profile, and write artifacts."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_process_modules.cleaning import clean_dataframe, snake_case
from data_process_modules.config import load_config
from data_process_modules.imputation_catalog import get_catalog, validate_selection
from data_process_modules.profiling import profile
from data_process_modules.remediation import remediate
from data_process_modules import tabular_checks, timeline, ts_checks
from data_process_modules.validation import (
    validate_numeric_timeseries,
    validate_tabular_dataframe,
)

LOGGER = logging.getLogger("dataops.pipeline")


def _missing_cells(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())


def _duplicate_rows(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def _issue_count(value: Any) -> int:
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, list):
        return len(value)
    if value:
        return 1
    return 0


def _quality_issue_summary(quality_report: dict) -> dict:
    if quality_report.get("mode") == "time_series":
        issues = quality_report.get("issues", {})
        timestamp_order = issues.get("timestamp_order", {})
        timestamp_order_issues = (
            int(timestamp_order.get("num_non_monotonic_steps", 0))
            + int(timestamp_order.get("num_duplicate_timestamps", 0))
            + int(timestamp_order.get("num_null_timestamps", 0))
        )
        return {
            "timestamp_order": timestamp_order_issues,
            "ts_gaps": int(issues.get("ts_gaps", {}).get("num_gaps", 0)),
            "missing": _issue_count(issues.get("missing", {})),
            "outliers": _issue_count(issues.get("outliers", [])),
            "failed_columns": 0 if quality_report.get("gx_passed") else 1,
        }
    return {
        "ts_gaps": 0,
        "missing": _issue_count(quality_report.get("missing_columns", [])),
        "outliers": _issue_count(quality_report.get("outlier_columns", [])),
        "failed_columns": _issue_count(quality_report.get("failed_columns", [])),
    }


def _quality_action_plan(quality_report: dict) -> list[dict]:
    """Map each quality issue family to the intended cleaning/transform action."""
    actions: list[dict] = []
    mode = quality_report.get("mode")
    if mode == "time_series":
        issues = quality_report.get("issues", {})
        timestamp_order = issues.get("timestamp_order", {})
        # Separate the three things a backward step or a shared timestamp can
        # mean. Reporting them as one "not monotonic" error hid the fact that
        # the pipeline was deleting real rows to make the error go away.
        if timestamp_order.get("num_runs", 1) > 1 and timestamp_order.get(
            "sweep_aware", False
        ):
            actions.append({
                "issue": "multiple_acquisition_runs",
                "severity": "warning",
                "status": "applied_by_remediation",
                "detected": {
                    "num_runs": timestamp_order.get("num_runs"),
                    "run_sizes": timestamp_order.get("run_sizes"),
                    "rows_out_of_order": timestamp_order.get("rows_out_of_order"),
                    "has_overlapping_runs": timestamp_order.get("has_overlapping_runs"),
                    "run_boundaries": timestamp_order.get("run_boundaries", []),
                },
                "solution": "A backward timestamp jump marks a new acquisition run, "
                            "not a corrupt cell. Rows are ordered within each run and "
                            "runs are kept in acquisition order, so independent runs "
                            "are never interleaved. Set timeline.sweep_aware: true to "
                            "carry the run id into the prepared bundle.",
                "module": "data_process_modules.timeline:detect_runs",
            })
        if timestamp_order.get("num_duplicate_timestamps", 0):
            key_cols = timestamp_order.get("key_columns", [])
            actions.append({
                "issue": "duplicate_primary_key",
                "severity": "warning",
                "status": "applied_by_remediation",
                "detected": {
                    "primary_key": ["<timestamp>"] + list(key_cols),
                    "duplicate_rows": timestamp_order.get("num_duplicate_timestamps"),
                    "duplicates_on_full_key": timestamp_order.get(
                        "num_duplicate_rows_on_key"),
                },
                "solution": "The timestamp is the primary key, so rows sharing one are "
                            "duplicates and are reduced by timeline.collision_policy. "
                            "Set timeline.sweep_aware: true to co-identify rows by the "
                            "swept factors instead of deduplicating them.",
                "module": "data_process_modules.timeline:resolve_collisions",
            })
        if timestamp_order.get("num_null_timestamps", 0) or not timestamp_order.get(
            "is_monotonic_increasing", True
        ):
            actions.append({
                "issue": "timestamp_not_monotonic",
                "severity": "error",
                "status": "applied_by_remediation",
                "detected": timestamp_order,
                "solution": "Out-of-order rows are dropped by a forward scan "
                            "(timeline.disorder_policy: drop), leaving a strictly "
                            "increasing timeline. Use disorder_policy: sort to reorder "
                            "them into the series instead of removing them.",
                "module": "data_process_modules.timeline:enforce_monotonic",
            })
        cadence = issues.get("ts_gaps", {}).get("cadence", {})
        if cadence.get("estimators_disagree"):
            actions.append({
                "issue": "ambiguous_cadence",
                "severity": "warning",
                "status": "applied_by_remediation",
                "detected": {
                    "median_dt_seconds": cadence.get("median_dt_seconds"),
                    "modal_dt_seconds": cadence.get("modal_dt_seconds"),
                    "modal_support": cadence.get("modal_support"),
                    "disagreement_ratio": cadence.get("disagreement_ratio"),
                    "estimator_used": cadence.get("estimator"),
                },
                "solution": "The modal and median sampling intervals disagree, usually "
                            "because collision residue leaves short steps behind. The "
                            "median is used, matching transform.preprocess:infer_base_dt, "
                            "so the gap estimate and the regularization grid agree.",
                "module": "data_process_modules.timeline:estimate_cadence",
            })
        gaps = issues.get("ts_gaps", {})
        if gaps.get("has_gaps"):
            actions.append({
                "issue": "time_gaps",
                "severity": "warning",
                "status": "deferred_to_imputation",
                "detected": {
                    "num_gaps": gaps.get("num_gaps", 0),
                    "missing_rows_estimate": gaps.get("total_missing_rows", 0),
                    "expected_dt_seconds": gaps.get("expected_dt_seconds"),
                },
                "solution": "Regularize the timeline with data_process_modules.transform.preprocess; then route long or structured gaps to the imputation apps.",
                "module": "data_process_modules.transform:preprocess",
            })
        missing = issues.get("missing", {})
        if missing:
            actions.append({
                "issue": "missing_values",
                "severity": "warning",
                "status": "deferred_to_imputation",
                "detected": missing,
                "solution": "Use tabular imputation for ordinary missing cells; use time-series imputation when missingness aligns with timeline gaps.",
                "module": "data_process_modules.transform:preprocess",
            })
        outliers = issues.get("outliers", [])
        if outliers:
            actions.append({
                "issue": "numeric_outliers",
                "severity": "info",
                "status": "applied_by_remediation",
                "detected": outliers,
                "solution": "Compute robust scaler stats and apply bounded clipping or downstream soft-clipping before model training.",
                "module": "data_process_modules.remediation:remediate",
            })
    else:
        missing_cols = quality_report.get("missing_columns", [])
        if missing_cols:
            actions.append({
                "issue": "missing_values",
                "severity": "warning",
                "status": "applied_by_remediation",
                "detected": missing_cols,
                "solution": "Apply type-aware tabular imputation or drop columns above the configured missingness threshold.",
                "module": "data_process_modules.remediation:remediate",
            })
        outlier_cols = quality_report.get("outlier_columns", [])
        if outlier_cols:
            actions.append({
                "issue": "numeric_outliers",
                "severity": "info",
                "status": "applied_by_remediation",
                "detected": outlier_cols,
                "solution": "Review numeric bounds; apply clipping, winsorization, or robust scaling before downstream modeling.",
                "module": "data_process_modules.remediation:remediate",
            })
        if quality_report.get("primary_key", {}).get("type") == "none":
            actions.append({
                "issue": "primary_key_missing",
                "severity": "info",
                "status": "manual",
                "detected": quality_report.get("primary_key"),
                "solution": "Treat the dataset as a fact table or configure a business key before row-level deduplication.",
                "module": "data_process_modules.profiling:detect_primary_key",
            })

    if not quality_report.get("gx_passed", True):
        gx_detail = quality_report.get("gx", {})
        actions.append({
            "issue": "gx_expectation_failures",
            "severity": "error",
            "status": "manual",
            "detected": {
                "passed": gx_detail.get("passed"),
                "evaluated": gx_detail.get("evaluated"),
                "failed_expectations": gx_detail.get("failed_expectations", []),
            } if gx_detail else _quality_issue_summary(quality_report),
            "solution": "Inspect the failed GX expectations below. The quantile-based "
                        "range checks are self-referential (re-flag near the clip "
                        "boundary), so winsorizing reduces but rarely zeroes them; "
                        "adjust schema, clean offending values, or relax thresholds deliberately.",
            "module": "great_expectations",
        })
    return actions


def _run_quality_checks(
    cleaned: pd.DataFrame,
    *,
    mode: str,
    timestamp_col: str | None,
    validation_config: dict[str, Any],
    timestamp_order_override: dict | None = None,
    key_columns: list[str] | None = None,
    sweep_aware: bool = False,
) -> dict:
    try:
        if mode == "time_series" and timestamp_col:
            report = ts_checks.run(
                cleaned,
                ts_col=timestamp_col,
                miss_threshold=float(validation_config.get("gx_missing_mostly", 0.98)),
                gap_factor=float(validation_config.get("gap_factor", 1.5)),
                min_gap_seconds=validation_config.get("min_gap_seconds"),
                outlier_q=float(validation_config.get("outlier_q", 0.01)),
                outlier_mostly=float(validation_config.get("outlier_mostly", 0.95)),
                gx_context_root=validation_config.get("gx_context_root"),
                key_columns=key_columns,
                sweep_aware=sweep_aware,
            )
        elif mode == "tabular":
            report = tabular_checks.run(
                cleaned,
                miss_threshold_numeric=float(
                    validation_config.get("gx_missing_mostly_numeric", 0.95)
                ),
                miss_threshold_cat=float(
                    validation_config.get("gx_missing_mostly_cat", 0.90)
                ),
                outlier_q=float(validation_config.get("outlier_q", 0.01)),
                outlier_mostly=float(validation_config.get("outlier_mostly", 0.95)),
                gx_context_root=validation_config.get("gx_context_root"),
            )
        else:
            report = {
                "mode": mode,
                "gx_passed": None,
                "summary": {},
                "skipped_reason": "quality checks require tabular or time_series mode",
            }
    except Exception as exc:
        report = {
            "mode": mode,
            "gx_passed": False,
            "summary": {},
            "error": str(exc),
        }

    if timestamp_order_override and report.get("mode") == "time_series":
        report.setdefault("issues", {})["timestamp_order"] = timestamp_order_override
        report.setdefault("recommendations", {})["structural_fix"] = True

    return {
        "mode": report.get("mode"),
        "gx_passed": report.get("gx_passed"),
        "report": report,
        "issue_summary": _quality_issue_summary(report),
        "action_plan": _quality_action_plan(report),
    }


def _is_timestamp_contract_quality_issue(
    error: Exception | None,
    *,
    validation_mode: str,
    timestamp_col: str | None,
) -> bool:
    """Return True for timestamp ordering/integrity errors we report as quality issues."""
    if error is None or validation_mode != "time_series" or not timestamp_col:
        return False
    message = str(error)
    if f"timestamp column {timestamp_col!r}" not in message:
        return False
    quality_markers = (
        "is not monotonic increasing",
        "contains duplicate values",
        "contains null values",
    )
    return any(marker in message for marker in quality_markers)


def _build_handoff(
    quality: dict,
    *,
    output_csv: str,
    ts_col: str | None,
    imputation_cfg: dict[str, Any],
    timeline_cfg: dict[str, Any] | None = None,
) -> dict:
    """Build the imputation handoff: regularize (bundle) + advertise the catalog.

    ``timeline_cfg`` controls per-campaign regularization; the bundle contract
    itself is unchanged, so every imputation runner reads it as before.

    The pipeline never runs imputation; it produces the prepared-dir bundle the
    apps consume and records the user's configured ``(app, method)`` selection
    so an external orchestrator can invoke the chosen app.
    """
    report = quality.get("report", {})
    is_ts = quality.get("mode") == "time_series"
    needs = bool(report.get("recommendations", {}).get("ts_imputation")) if is_ts else False

    selection = validate_selection(
        imputation_cfg.get("app"), imputation_cfg.get("method")
    )
    handoff: dict[str, Any] = {
        "needs_ts_imputation": needs,
        "reason": "time_gaps_detected" if needs
        else ("no_time_gaps" if is_ts else "not_time_series"),
        "prepared_dir": None,
        "bundle_written": False,
        "bundle_error": None,
        "target_cols": [],
        "imputation_catalog": get_catalog(),
        "selection": selection,
        "invoke_hint": None,
    }
    if not needs:
        return handoff

    timeline_cfg = timeline_cfg or {}
    if imputation_cfg.get("build_bundle", True) and ts_col:
        _base = Path(output_csv).stem.removesuffix("_remediated")
        prepared_dir = imputation_cfg.get("prepared_dir") or str(
            Path(output_csv).with_name(f"{_base}_regularized")
        )
        try:
            from data_process_modules.transform import preprocess_csv

            meta = preprocess_csv(
                input_csv=output_csv, output_dir=prepared_dir, time_col=ts_col,
                # 6G-schema unit conversions (ram_limit→ram_limit_mb, ram_usage→
                # ram_usage_mb, latency μs→ms, cpu_usage %→fraction). A no-op on
                # non-6G inputs, and matches the Airflow/container + training paths
                # so the local raw→processed bundle carries the same columns.
                convert_units=True,
                # Per-campaign grids: a collection pause must not stretch one
                # grid across it, and each campaign keeps its own cadence.
                segment_regularization=bool(
                    timeline_cfg.get("segment_regularization", True)),
                segment_gap_seconds=float(
                    timeline_cfg.get("segment_gap_seconds", 86400.0)),
                min_segment_rows=int(timeline_cfg.get("min_segment_rows", 32)),
                # Never ship a bundle with two sampling regimes: mixed
                # regularization splits them across train/test.
                require_all_segments=bool(
                    timeline_cfg.get("require_all_segments", True)),
            )
            handoff["prepared_dir"] = prepared_dir
            handoff["bundle_written"] = True
            handoff["target_cols"] = meta.get("target_cols", [])
            if selection["status"] in {"ok", "known_failing"}:
                out_dir = str(Path(prepared_dir) / "imputed")
                handoff["invoke_hint"] = (
                    f"# app: {selection['app']}\n"
                    f"python run_imputation.py --prepared-dir {prepared_dir} "
                    f"--output-dir {out_dir} --method {selection['method']} "
                    f"--inputs train test"
                )
        except Exception as exc:  # noqa: BLE001 - record, never fail the pipeline
            handoff["bundle_error"] = str(exc)
            LOGGER.warning("imputation bundle generation failed: %s", exc)

    return handoff


def _validation_comparison(
    raw: pd.DataFrame,
    soft_cleaned: pd.DataFrame,
    remediated: pd.DataFrame,
    *,
    cleaning_report: Any,
    remediation_report: Any,
    quality: dict,
    validation: dict,
    duplicate_timestamps: int = 0,
    non_monotonic_timestamps: int = 0,
    timeline_diag: dict | None = None,
) -> dict:
    """Compact payload for dashboard charts comparing raw, soft-cleaned, remediated,
    GX and Pandera."""
    quality_summary = quality.get("issue_summary", {})
    timeline_diag = timeline_diag or {}
    return {
        "dataset_shape": {
            "raw": {"rows": int(len(raw)), "cols": int(raw.shape[1])},
            "soft_cleaned": {
                "rows": int(len(soft_cleaned)),
                "cols": int(soft_cleaned.shape[1]),
            },
            # Backward-compatible alias for existing dashboard/report consumers.
            "cleaned": {
                "rows": int(len(soft_cleaned)),
                "cols": int(soft_cleaned.shape[1]),
            },
            "remediated": {"rows": int(len(remediated)), "cols": int(remediated.shape[1])},
        },
        "cleaning_effect": {
            "dropped_rows": int(getattr(cleaning_report, "dropped_empty_rows", 0))
            + int(getattr(cleaning_report, "dropped_duplicate_rows", 0)),
            "duplicate_rows_before": _duplicate_rows(raw),
            "duplicate_rows_after": _duplicate_rows(soft_cleaned),
            "duplicate_timestamps_collapsed": int(duplicate_timestamps),
            "non_monotonic_timestamps_sorted": int(non_monotonic_timestamps),
            # Collisions that a key separated are *preserved*, not collapsed —
            # the number that used to be silently folded into the line above.
            "timestamp_collisions_detected": int(
                (timeline_diag.get("collision") or {}).get(
                    "collisions_on_timestamp", 0)
            ),
            "timestamp_collisions_preserved": int(
                (timeline_diag.get("collision") or {}).get(
                    "distinct_conditions_preserved", 0)
            ),
            "acquisition_runs": int(
                (timeline_diag.get("runs") or {}).get("num_runs", 1) or 1
            ),
            "rows_out_of_order": int(
                (timeline_diag.get("runs") or {}).get("rows_out_of_order", 0)
            ),
            "out_of_order_rows_dropped": int(
                (timeline_diag.get("disorder") or {}).get("rows_dropped", 0)
            ),
            "disorder_policy": (timeline_diag.get("disorder") or {}).get("policy"),
            "missing_cells_before": _missing_cells(raw),
            "missing_cells_after": _missing_cells(soft_cleaned),
        },
        "remediation_effect": {
            "missing_cells_before": int(getattr(remediation_report, "missing_cells_before", 0)),
            "missing_cells_after": int(getattr(remediation_report, "missing_cells_after", 0)),
            "outlier_cells_clipped": int(getattr(remediation_report, "outlier_cells_clipped", 0)),
            "outlier_cells_flagged": int(getattr(remediation_report, "outlier_cells_flagged", 0)),
            "actions": [a.get("issue") for a in getattr(remediation_report, "actions", [])],
        },
        "validation_status": {
            "gx_passed": quality.get("gx_passed"),
            "pandera_passed": validation.get("pandera_passed"),
            "mode": validation.get("mode"),
        },
        "issue_counts": quality_summary,
        "chart_ready": [
            {"stage": "raw", "metric": "missing_cells", "value": _missing_cells(raw)},
            {
                "stage": "soft_cleaned",
                "metric": "missing_cells",
                "value": _missing_cells(soft_cleaned),
            },
            {"stage": "remediated", "metric": "missing_cells", "value": _missing_cells(remediated)},
            {"stage": "raw", "metric": "duplicate_rows", "value": _duplicate_rows(raw)},
            {
                "stage": "soft_cleaned",
                "metric": "duplicate_rows",
                "value": _duplicate_rows(soft_cleaned),
            },
            {"stage": "remediated", "metric": "outlier_cells_clipped",
             "value": int(getattr(remediation_report, "outlier_cells_clipped", 0))},
            {"stage": "gx", "metric": "failed_issue_groups", "value": sum(quality_summary.values())},
            {"stage": "pandera", "metric": "errors", "value": len(validation.get("errors", []))},
        ],
    }


def configure_logging(log_file: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=os.environ.get("DATAOPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def _log_raw_quality(input_csv: str,
                     raw: pd.DataFrame,
                     cleaning_report: Any,
                     quality: dict,
                     *,
                     ts_col: str | None,
                     timeline_diag: dict,
                     duplicate_timestamps: int,
                     non_monotonic_timestamps: int) -> None:
    """Log every quality issue detected on the raw data, one issue per line.

    The report JSON has always carried this; the log did not, so a run that
    silently dropped thousands of rows looked identical in the log to a clean
    one. Written at WARNING for anything that loses or rewrites data and INFO
    for the rest, so ``grep WARNING`` over a log is a usable triage pass.
    """
    rep = quality.get("report", {})
    issues = rep.get("issues", {}) or {}
    mode = quality.get("mode") or rep.get("mode") or "unknown"

    LOGGER.info("── raw data quality: %s ──", input_csv)
    LOGGER.info("shape: %d rows × %d columns · mode=%s",
                len(raw), len(raw.columns), mode)

    # --- structural -------------------------------------------------------
    dupe_rows = _duplicate_rows(raw)
    if dupe_rows:
        LOGGER.warning("duplicate rows: %d fully identical row(s)", dupe_rows)
    dropped = int(getattr(cleaning_report, "input_rows", len(raw))
                  - getattr(cleaning_report, "output_rows", len(raw)))
    if dropped:
        LOGGER.warning("cleaning dropped %d row(s) (empty/duplicate)", dropped)

    # --- missing ----------------------------------------------------------
    total_missing = _missing_cells(raw)
    if total_missing:
        per_col = raw.isna().sum()
        worst = per_col[per_col > 0].sort_values(ascending=False)
        LOGGER.warning("missing values: %d cell(s) across %d column(s)",
                       total_missing, len(worst))
        for col, n in worst.head(10).items():
            LOGGER.warning("    %-22s %d missing (%.1f%%)", col, int(n), 100 * n / len(raw))
        if len(worst) > 10:
            LOGGER.warning("    … and %d more column(s)", len(worst) - 10)
    else:
        LOGGER.info("missing values: none in the raw file")

    # --- timestamps -------------------------------------------------------
    if ts_col:
        if duplicate_timestamps:
            collision = timeline_diag.get("collision") or {}
            LOGGER.warning(
                "timestamp collisions: %d row(s) share a '%s' value — policy '%s' "
                "removed %d row(s)",
                collision.get("collisions_on_timestamp", duplicate_timestamps), ts_col,
                collision.get("policy", "?"), collision.get("rows_removed", 0))
            advisory = timeline_diag.get("key_advisory") or {}
            if advisory.get("candidate_key"):
                LOGGER.warning(
                    "    a composite key %s would resolve %d of them to distinct rows "
                    "(residual %d) — set timeline.sweep_aware: true to keep them",
                    advisory["candidate_key"],
                    advisory.get("collisions_on_timestamp", 0)
                    - advisory.get("residual_collisions", 0),
                    advisory.get("residual_collisions", 0))
        if non_monotonic_timestamps:
            disorder = timeline_diag.get("disorder") or {}
            LOGGER.warning(
                "timestamp order: %d backward step(s) on '%s' — policy '%s' dropped "
                "%d row(s)",
                disorder.get("backward_steps", non_monotonic_timestamps), ts_col,
                disorder.get("policy", "?"), disorder.get("rows_dropped", 0))
        gaps = issues.get("ts_gaps") or {}
        if gaps.get("has_gaps"):
            LOGGER.warning(
                "timeline gaps: %d gap(s), %.1f%% of the grid, ~%d missing row(s); "
                "cadence %ss, largest gap %ss",
                gaps.get("num_gaps", 0), 100 * float(gaps.get("gap_pct", 0)),
                gaps.get("total_missing_rows", 0),
                gaps.get("expected_dt_seconds"), gaps.get("largest_gap_seconds"))

    # --- outliers (detected always; rewritten only when clip_outliers) -----
    outlier_cols = (issues.get("outliers") if mode == "time_series"
                    else rep.get("outlier_columns")) or []
    if outlier_cols:
        LOGGER.info("numeric outliers: %d column(s) outside the quantile band: %s",
                    len(outlier_cols), ", ".join(map(str, outlier_cols)))

    # --- validation -------------------------------------------------------
    gx = rep.get("gx") or {}
    if gx:
        failed = gx.get("failed", 0)
        (LOGGER.warning if failed else LOGGER.info)(
            "expectations: %d/%d passed, %d failed",
            gx.get("passed", 0), gx.get("evaluated", 0), failed)
        for item in (gx.get("failed_expectations") or [])[:10]:
            LOGGER.warning("    failed: %s", item)

    recs = [k for k, v in (rep.get("recommendations") or {}).items() if v]
    if recs:
        LOGGER.info("recommendations: %s", ", ".join(recs))


def _log_remediation(remediation_report: Any) -> None:
    """Log what remediation actually changed, and what it only reported."""
    LOGGER.info("── remediation ──")
    for action in getattr(remediation_report, "actions", []):
        status = action.get("status", "?")
        line = f"{action.get('issue')}: {action.get('action')} [{status}]"
        if action.get("clipped_cells"):
            line += f" — {action['clipped_cells']} cell(s) winsorized"
        elif action.get("flagged_cells"):
            line += (f" — {action['flagged_cells']} cell(s) outside the band, "
                     "left unchanged (validation.clip_outliers: false)")
        elif action.get("columns"):
            line += f" — {len(action['columns'])} column(s)"
        (LOGGER.info if status in ("applied", "reported_not_applied")
         else LOGGER.warning)("%s", line)
    LOGGER.info("missing cells: %d → %d",
                getattr(remediation_report, "missing_cells_before", 0),
                getattr(remediation_report, "missing_cells_after", 0))


def notify_failure(message: str) -> None:
    """Send a simple failure notification when DATAOPS_WEBHOOK_URL is configured."""
    webhook_url = os.environ.get("DATAOPS_WEBHOOK_URL")
    if not webhook_url:
        return
    try:
        requests.post(webhook_url, json={"text": message}, timeout=10).raise_for_status()
    except requests.RequestException:
        LOGGER.exception("failed to send failure notification")


def run_pipeline(
    input_csv: str,
    output_csv: str,
    report_json: str,
    *,
    timestamp_col: str | None = None,
    validation_config: dict[str, Any] | None = None,
    imputation_config: dict[str, Any] | None = None,
    timeline_config: dict[str, Any] | None = None,
    soft_cleaned_csv: str | None = None,
    cleaned_csv: str | None = None,
) -> dict:
    raw = pd.read_csv(input_csv)
    soft_cleaned, cleaning_report = clean_dataframe(raw, datetime_column=timestamp_col)
    validation_cfg = validation_config or {}
    configured_mode = validation_cfg.get("mode", "auto")
    detected_profile = profile(
        soft_cleaned,
        timestamp_col=timestamp_col,
        allow_step_index_timestamp=bool(
            validation_cfg.get("allow_step_index_timestamp", False)
        ),
    )
    configured_ts_col = snake_case(timestamp_col) if timestamp_col else None
    detected_ts_col = detected_profile.get("timestamp_column")

    ts_col = None
    if configured_mode == "time_series":
        ts_col = configured_ts_col or detected_ts_col
    elif configured_mode == "auto":
        if configured_ts_col and configured_ts_col in soft_cleaned.columns:
            ts_col = configured_ts_col
        elif detected_profile.get("data_type") == "time_series":
            ts_col = detected_ts_col

    validation_mode = "time_series" if configured_mode == "auto" and ts_col else configured_mode
    if configured_mode == "auto" and not ts_col:
        validation_mode = "tabular"

    # Resolve row identity and timestamp ordering.
    #
    # Default model: **the timestamp is the primary key**. Rows sharing a
    # timestamp are duplicates and are reduced by ``collision_policy``; rows that
    # go backwards in time are dropped by a forward scan (``disorder_policy:
    # drop``) rather than sorted into the middle of the series.
    #
    # ``timeline.sweep_aware: true`` opts into the alternative model for
    # parameter-sweep datasets, where the swept factors co-identify a row and a
    # backward jump starts a new acquisition run. The key inference still runs
    # either way so the report can show what a composite key *would* resolve —
    # that is advisory only and never changes the default behaviour.
    duplicate_timestamps = int(getattr(cleaning_report, "duplicate_timestamps", 0))
    timestamp_order_before: dict | None = None
    non_monotonic_timestamps = 0
    timeline_cfg = timeline_config or {}
    timeline_diag: dict[str, Any] = {}
    key_columns: list[str] = []
    if validation_mode == "time_series" and ts_col and ts_col in soft_cleaned.columns:
        sweep_aware = bool(timeline_cfg.get("sweep_aware", False))
        configured_key = [
            snake_case(c) for c in (timeline_cfg.get("key_columns") or [])
        ]
        configured_key = [c for c in configured_key if c in soft_cleaned.columns]

        # Always computed, for the report; only *used* as the key in sweep mode.
        key_info = timeline.infer_key_columns(soft_cleaned, ts_col)
        if configured_key:
            key_columns, key_source = configured_key, "configured"
        elif sweep_aware or timeline_cfg.get("auto_key_columns", False):
            key_columns, key_source = key_info["key_columns"], "inferred"
        else:
            key_columns, key_source = [], "timestamp_primary_key"

        min_overlap = int(timeline_cfg.get("min_run_overlap_rows", 8))
        run_detection = bool(timeline_cfg.get("run_detection", False)) or sweep_aware
        runs = (
            timeline.detect_runs(soft_cleaned, ts_col, min_overlap_rows=min_overlap)
            if run_detection else None
        )
        run_id = runs["run_id"] if runs else None

        # Snapshot the timestamp contract as *detected*, before anything is
        # removed — this is what the action plan reports, so the duplicates and
        # out-of-order rows the pipeline removes stay on the record.
        timestamp_order_before = ts_checks.inspect_timestamp_order(
            soft_cleaned, ts_col, key_columns=key_columns, sweep_aware=sweep_aware
        )

        # --- 1. Duplicates, on the primary key ---------------------------
        policy = timeline_cfg.get("collision_policy", "keep_last")
        soft_cleaned, collision = timeline.resolve_collisions(
            soft_cleaned, ts_col,
            key_columns=key_columns, policy=policy, run_id=run_id,
        )
        soft_cleaned = soft_cleaned.reset_index(drop=True)
        duplicate_timestamps += int(collision["rows_removed"])
        if collision["collisions_on_timestamp"]:
            LOGGER.warning(
                "%s duplicate timestamp(s) on primary key %r: %s row(s) removed "
                "by policy %r%s",
                collision["collisions_on_timestamp"], ts_col,
                collision["rows_removed"], policy,
                f" (key {key_columns})" if key_columns else "",
            )

        # --- 2. Time disorder --------------------------------------------
        non_monotonic_timestamps = int(
            ts_checks.inspect_timestamp_order(soft_cleaned, ts_col)
            .get("num_non_monotonic_steps", 0)
        )
        disorder_policy = timeline_cfg.get("disorder_policy", "drop")
        if runs and runs["num_runs"] > 1 and sweep_aware:
            # Sweep mode keeps runs separate instead of dropping the later one.
            disorder_policy = "sort"
        runs_after = (
            timeline.detect_runs(soft_cleaned, ts_col, min_overlap_rows=min_overlap)
            if run_detection else None
        )
        soft_cleaned, disorder = timeline.enforce_monotonic(
            soft_cleaned, ts_col, policy=disorder_policy,
            run_id=runs_after["run_id"] if runs_after else None,
        )
        if disorder["rows_dropped"]:
            LOGGER.warning(
                "dropped %s out-of-order row(s) on %r (%s backward step(s); "
                "forward scan keeps a strictly increasing timeline)",
                disorder["rows_dropped"], ts_col, disorder["backward_steps"],
            )
        elif not disorder["was_monotonic"] and disorder_policy == "sort":
            LOGGER.warning(
                "sorted %s non-monotonic timestamp step(s) on %r",
                disorder["backward_steps"], ts_col,
            )

        if runs_after and sweep_aware:
            # Sweep-aware mode: expose the run as data so downstream splits,
            # regularization, and the dashboard can segment on it.
            soft_cleaned = soft_cleaned.assign(
                run=timeline.detect_runs(
                    soft_cleaned, ts_col, min_overlap_rows=min_overlap)["run_id"]
            )

        timeline_diag = {
            "primary_key": [ts_col] + list(key_columns),
            "key_columns": key_columns,
            "key_source": key_source,
            # Advisory: what a composite sweep key would resolve, even when the
            # timestamp alone is the key. Lets the dashboard show how many of the
            # removed rows were distinct experimental conditions.
            "key_advisory": {
                "candidate_key": key_info.get("key_columns", []),
                "collisions_on_timestamp": key_info.get("collisions_on_timestamp", 0),
                "residual_collisions": key_info.get("residual_collisions", 0),
                "candidates": key_info.get("candidates", [])[:8],
            },
            "collision": collision,
            "disorder": disorder,
            "runs": {k: v for k, v in (runs_after or {}).items() if k != "run_id"},
            "sweep_aware": sweep_aware,
        }

    validation = {
        "configured_mode": configured_mode,
        "mode": validation_mode,
        "pandera_passed": None,
        "timestamp_column": ts_col,
        "errors": [],
    }
    validation_error: Exception | None = None
    if validation_mode == "none":
        validation["pandera_passed"] = None
    elif validation_mode == "time_series":
        if not ts_col:
            validation_error = ValueError("time_series validation requires a timestamp column")
            validation["pandera_passed"] = False
            validation["errors"].append(str(validation_error))
        else:
            try:
                validate_numeric_timeseries(
                    soft_cleaned,
                    timestamp_col=ts_col,
                    expected_columns=validation_cfg.get("expected_columns") or None,
                    numeric_bounds=validation_cfg.get("numeric_bounds") or None,
                    missing_threshold=float(validation_cfg.get("missing_threshold", 0.0)),
                    require_timestamp_unique=bool(
                        validation_cfg.get("require_timestamp_unique", True)
                    ),
                    require_timestamp_monotonic=bool(
                        validation_cfg.get("require_timestamp_monotonic", True)
                    ),
                )
                validation["pandera_passed"] = True
            except Exception as exc:
                validation["pandera_passed"] = False
                validation["errors"].append(str(exc))
                validation_error = exc
    elif validation_mode == "tabular":
        try:
            validate_tabular_dataframe(
                soft_cleaned,
                expected_columns=validation_cfg.get("expected_columns") or None,
                numeric_bounds=validation_cfg.get("numeric_bounds") or None,
                missing_threshold=float(validation_cfg.get("missing_threshold", 0.0)),
            )
            validation["pandera_passed"] = True
        except Exception as exc:
            validation["pandera_passed"] = False
            validation["errors"].append(str(exc))
            validation_error = exc
    else:
        validation_error = ValueError(
            "validation.mode must be one of: auto, time_series, tabular, none"
        )
        validation["pandera_passed"] = False
        validation["errors"].append(str(validation_error))

    output_path = Path(output_csv)
    report_path = Path(report_json)
    soft_cleaned_csv = soft_cleaned_csv or cleaned_csv
    _base = output_path.stem.removesuffix("_remediated")
    soft_cleaned_path = Path(soft_cleaned_csv) if soft_cleaned_csv else output_path.with_name(
        f"{_base}_soft_cleaned{output_path.suffix}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    soft_cleaned_path.parent.mkdir(parents=True, exist_ok=True)

    quality = _run_quality_checks(
        soft_cleaned,
        mode=validation_mode,
        timestamp_col=ts_col,
        validation_config=validation_cfg,
        # Report the timestamp contract as *detected*, not as left behind by the
        # fixes above — otherwise the action plan silently omits the duplicates
        # and out-of-order rows the pipeline just removed. ``quality_after``
        # carries the post-fix state, which is the other half of the before/after.
        timestamp_order_override=timestamp_order_before
        if (non_monotonic_timestamps or duplicate_timestamps)
        else None,
        key_columns=key_columns,
        sweep_aware=bool(timeline_cfg.get("sweep_aware", False)),
    )

    _log_raw_quality(
        input_csv, raw, cleaning_report, quality,
        ts_col=ts_col, timeline_diag=timeline_diag,
        duplicate_timestamps=duplicate_timestamps,
        non_monotonic_timestamps=non_monotonic_timestamps,
    )

    # Remediation: act on each detected issue (report — or, when enabled,
    # winsorize — outliers; type-aware fill for tabular missing). Time-series
    # gaps are left for the imputation handoff below.
    remediated, remediation_report = remediate(
        soft_cleaned,
        quality["report"],
        outlier_q=float(validation_cfg.get("outlier_q", 0.01)),
        # Off by default: outliers are always detected and reported, but
        # winsorizing rewrites real measurements — on benchmark data a genuine
        # latency spike is signal. Opt in with validation.clip_outliers: true.
        clip_outliers=bool(validation_cfg.get("clip_outliers", False)),
    )

    _log_remediation(remediation_report)

    # Re-run the same checks on the remediated frame so the report carries a
    # genuine before/after for the dashboard. Time-series gaps remain (deferred
    # to imputation), so "after" reflects only what remediation actually fixed.
    quality_after = None
    if validation_cfg.get("recheck_after_remediation", True):
        quality_after = _run_quality_checks(
            remediated,
            mode=validation_mode,
            timestamp_col=ts_col,
            validation_config=validation_cfg,
            key_columns=key_columns,
            sweep_aware=bool(timeline_cfg.get("sweep_aware", False)),
        )

    comparison = _validation_comparison(
        raw,
        soft_cleaned,
        remediated,
        cleaning_report=cleaning_report,
        remediation_report=remediation_report,
        quality=quality,
        validation=validation,
        duplicate_timestamps=duplicate_timestamps,
        non_monotonic_timestamps=non_monotonic_timestamps,
        timeline_diag=timeline_diag,
    )

    # Persist both stages as artifacts: the soft-cleaned frame (before
    # per-issue remediation) and the remediated frame (the pipeline's output).
    soft_cleaned.to_csv(soft_cleaned_path, index=False)
    remediated.to_csv(output_path, index=False)

    # Handoff: regularize a gappy timeline into the prepared-dir bundle and
    # advertise the imputation catalog + configured selection. No imputation runs.
    imputation_cfg = imputation_config or {}
    handoff = _build_handoff(
        quality,
        output_csv=output_csv,
        ts_col=ts_col,
        imputation_cfg=imputation_cfg,
        timeline_cfg=timeline_cfg,
    )

    report = {
        "input": input_csv,
        "output": output_csv,
        "soft_cleaned_output": str(soft_cleaned_path),
        # Backward-compatible alias for older dashboard/scripts/tests.
        "cleaned_output": str(soft_cleaned_path),
        "report_path": report_json,
        "data_type": validation_mode if validation_mode in {"tabular", "time_series"} else None,
        "cleaning": {
            **asdict(cleaning_report),
            "duplicate_timestamps": duplicate_timestamps,
            "non_monotonic_timestamps": non_monotonic_timestamps,
        },
        "remediation": asdict(remediation_report),
        "profile": detected_profile,
        "timeline": timeline_diag,
        "quality": quality,
        "quality_after": quality_after,
        "validation": validation,
        "validation_comparison": comparison,
        "handoff": handoff,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    if validation_error and not _is_timestamp_contract_quality_issue(
        validation_error,
        validation_mode=validation_mode,
        timestamp_col=ts_col,
    ):
        raise validation_error
    return report


def run_from_config(config_path: str | None = None, **overrides: str | None) -> dict:
    """Run the pipeline from YAML config with optional CLI/env overrides."""
    cfg = load_config(config_path)
    for key in (
        "input",
        "output",
        "report",
        "log_file",
        "timestamp_col",
        "soft_cleaned_output",
        "cleaned_output",
    ):
        if overrides.get(key) is not None:
            cfg[key] = overrides[key]
    soft_cleaned_output = cfg.get("soft_cleaned_output") or cfg.get("cleaned_output")
    return run_pipeline(
        cfg["input"],
        cfg["output"],
        cfg["report"],
        timestamp_col=cfg.get("timestamp_col"),
        validation_config=cfg.get("validation", {}),
        imputation_config=cfg.get("imputation", {}),
        timeline_config=cfg.get("timeline", {}),
        soft_cleaned_csv=soft_cleaned_output,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal DataOps pipeline")
    parser.add_argument("--config", default="config/dataops.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--timestamp-col", default=None)
    parser.add_argument("--log-file", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    log_file = args.log_file if args.log_file is not None else cfg.get("log_file")
    configure_logging(log_file)
    try:
        report = run_from_config(
            args.config,
            input=args.input,
            output=args.output,
            report=args.report,
            log_file=args.log_file,
            timestamp_col=args.timestamp_col,
        )
    except Exception as exc:
        LOGGER.exception("minimal DataOps pipeline failed")
        notify_failure(f"minimal DataOps pipeline failed: {exc}")
        raise
    # cleaning.output_rows predates the timeline collision/disorder stages, so it
    # overstates what was written (golang: 58763 vs 51591 rows actually on disk).
    # Report the shape the pipeline actually produced.
    shape = ((report.get("validation_comparison") or {}).get("dataset_shape") or {})
    written = (shape.get("remediated") or {}).get("rows")
    LOGGER.info(
        "pipeline complete: rows=%s (raw %s) output=%s report=%s",
        written if written is not None else report["cleaning"]["output_rows"],
        report["cleaning"]["input_rows"],
        report["output"],
        report.get("report_path", args.report),
    )


if __name__ == "__main__":
    main()
