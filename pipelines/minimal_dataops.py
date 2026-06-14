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
from data_process_modules import tabular_checks, ts_checks
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
        return {
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

    return {
        "mode": report.get("mode"),
        "gx_passed": report.get("gx_passed"),
        "report": report,
        "issue_summary": _quality_issue_summary(report),
        "action_plan": _quality_action_plan(report),
    }


def _build_handoff(
    quality: dict,
    *,
    output_csv: str,
    ts_col: str | None,
    imputation_cfg: dict[str, Any],
) -> dict:
    """Build the imputation handoff: regularize (bundle) + advertise the catalog.

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

    if imputation_cfg.get("build_bundle", True) and ts_col:
        prepared_dir = imputation_cfg.get("prepared_dir") or (
            str(Path(output_csv).with_suffix("")) + "_prepared"
        )
        try:
            from data_process_modules.transform import preprocess_csv

            meta = preprocess_csv(
                input_csv=output_csv, output_dir=prepared_dir, time_col=ts_col
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
    cleaned: pd.DataFrame,
    remediated: pd.DataFrame,
    *,
    cleaning_report: Any,
    remediation_report: Any,
    quality: dict,
    validation: dict,
) -> dict:
    """Compact payload for dashboard charts comparing raw, cleaned, remediated,
    GX and Pandera."""
    quality_summary = quality.get("issue_summary", {})
    return {
        "dataset_shape": {
            "raw": {"rows": int(len(raw)), "cols": int(raw.shape[1])},
            "cleaned": {"rows": int(len(cleaned)), "cols": int(cleaned.shape[1])},
            "remediated": {"rows": int(len(remediated)), "cols": int(remediated.shape[1])},
        },
        "cleaning_effect": {
            "dropped_rows": int(getattr(cleaning_report, "dropped_empty_rows", 0))
            + int(getattr(cleaning_report, "dropped_duplicate_rows", 0)),
            "duplicate_rows_before": _duplicate_rows(raw),
            "duplicate_rows_after": _duplicate_rows(cleaned),
            "missing_cells_before": _missing_cells(raw),
            "missing_cells_after": _missing_cells(cleaned),
        },
        "remediation_effect": {
            "missing_cells_before": int(getattr(remediation_report, "missing_cells_before", 0)),
            "missing_cells_after": int(getattr(remediation_report, "missing_cells_after", 0)),
            "outlier_cells_clipped": int(getattr(remediation_report, "outlier_cells_clipped", 0)),
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
            {"stage": "cleaned", "metric": "missing_cells", "value": _missing_cells(cleaned)},
            {"stage": "remediated", "metric": "missing_cells", "value": _missing_cells(remediated)},
            {"stage": "raw", "metric": "duplicate_rows", "value": _duplicate_rows(raw)},
            {"stage": "cleaned", "metric": "duplicate_rows", "value": _duplicate_rows(cleaned)},
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
    cleaned_csv: str | None = None,
) -> dict:
    raw = pd.read_csv(input_csv)
    cleaned, cleaning_report = clean_dataframe(raw, datetime_column=timestamp_col)
    validation_cfg = validation_config or {}
    configured_mode = validation_cfg.get("mode", "auto")
    detected_profile = profile(
        cleaned,
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
        if configured_ts_col and configured_ts_col in cleaned.columns:
            ts_col = configured_ts_col
        elif detected_profile.get("data_type") == "time_series":
            ts_col = detected_ts_col

    validation_mode = "time_series" if configured_mode == "auto" and ts_col else configured_mode
    if configured_mode == "auto" and not ts_col:
        validation_mode = "tabular"
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
                    cleaned,
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
                cleaned,
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
    cleaned_path = Path(cleaned_csv) if cleaned_csv else output_path.with_name(
        output_path.stem + "_cleaned" + output_path.suffix
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)

    quality = _run_quality_checks(
        cleaned,
        mode=validation_mode,
        timestamp_col=ts_col,
        validation_config=validation_cfg,
    )

    # Remediation: act on each detected issue (winsorize outliers, type-aware
    # fill for tabular missing). Time-series gaps are left for the imputation
    # handoff below.
    remediated, remediation_report = remediate(
        cleaned,
        quality["report"],
        outlier_q=float(validation_cfg.get("outlier_q", 0.01)),
    )

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
        )

    comparison = _validation_comparison(
        raw,
        cleaned,
        remediated,
        cleaning_report=cleaning_report,
        remediation_report=remediation_report,
        quality=quality,
        validation=validation,
    )

    # Persist both stages as artifacts: the conservative-clean frame (before
    # per-issue remediation) and the remediated frame (the pipeline's output).
    cleaned.to_csv(cleaned_path, index=False)
    remediated.to_csv(output_path, index=False)

    # Handoff: regularize a gappy timeline into the prepared-dir bundle and
    # advertise the imputation catalog + configured selection. No imputation runs.
    imputation_cfg = imputation_config or {}
    handoff = _build_handoff(
        quality,
        output_csv=output_csv,
        ts_col=ts_col,
        imputation_cfg=imputation_cfg,
    )

    report = {
        "input": input_csv,
        "output": output_csv,
        "cleaned_output": str(cleaned_path),
        "report_path": report_json,
        "data_type": validation_mode if validation_mode in {"tabular", "time_series"} else None,
        "cleaning": asdict(cleaning_report),
        "remediation": asdict(remediation_report),
        "profile": detected_profile,
        "quality": quality,
        "quality_after": quality_after,
        "validation": validation,
        "validation_comparison": comparison,
        "handoff": handoff,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    if validation_error:
        raise validation_error
    return report


def run_from_config(config_path: str | None = None, **overrides: str | None) -> dict:
    """Run the pipeline from YAML config with optional CLI/env overrides."""
    cfg = load_config(config_path)
    for key in ("input", "output", "report", "log_file", "timestamp_col", "cleaned_output"):
        if overrides.get(key) is not None:
            cfg[key] = overrides[key]
    return run_pipeline(
        cfg["input"],
        cfg["output"],
        cfg["report"],
        timestamp_col=cfg.get("timestamp_col"),
        validation_config=cfg.get("validation", {}),
        imputation_config=cfg.get("imputation", {}),
        cleaned_csv=cfg.get("cleaned_output"),
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
    LOGGER.info(
        "pipeline complete: rows=%s output=%s report=%s",
        report["cleaning"]["output_rows"],
        report["output"],
        report.get("report_path", args.report),
    )


if __name__ == "__main__":
    main()
