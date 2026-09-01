from __future__ import annotations

import json

import pandas as pd

from dataops.imputation_catalog import validate_selection
from dataops.remediation import remediate
from pipelines.minimal_dataops import run_pipeline


def _outlier_frame() -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame(
        {
            "x": list(range(100)) + [10_000],          # one extreme outlier
            "label": ["a"] * 90 + ["b"] * 10 + [None],  # one missing categorical
        }
    )
    return df, {
        "mode": "tabular",
        "outlier_columns": ["x"],
        "missing_columns": ["label"],
    }


def test_remediate_reports_outliers_without_clipping_by_default():
    """Clipping is opt-in: the outlier is reported, the value is left alone."""
    df, quality_report = _outlier_frame()
    out, report = remediate(df, quality_report, outlier_q=0.01)

    assert out["x"].max() == 10_000                  # extreme value NOT rewritten
    assert report.outlier_cells_clipped == 0
    assert report.outlier_cells_flagged >= 1         # but it is counted
    assert out["label"].isna().sum() == 0            # categorical still filled (mode)
    assert report.missing_cells_after < report.missing_cells_before

    action = next(a for a in report.actions if a["issue"] == "numeric_outliers")
    assert action["action"] == "report_only"
    assert action["status"] == "reported_not_applied"
    assert action["would_clip_cells"] == report.outlier_cells_flagged
    assert "x" in action["columns"]


def test_remediate_clips_tabular_outliers_when_enabled():
    df, quality_report = _outlier_frame()
    out, report = remediate(df, quality_report, outlier_q=0.01, clip_outliers=True)

    assert out["x"].max() < 10_000                   # extreme value winsorized
    assert out["label"].isna().sum() == 0            # categorical filled (mode)
    assert report.outlier_cells_clipped >= 1
    assert report.outlier_cells_flagged == report.outlier_cells_clipped
    assert report.missing_cells_after < report.missing_cells_before
    issues = {a["issue"] for a in report.actions}
    assert {"numeric_outliers", "missing_values"} <= issues


def test_remediate_defers_timeseries_gaps_to_imputation():
    quality_report = {
        "mode": "time_series",
        "issues": {
            "ts_gaps": {"has_gaps": True, "num_gaps": 2, "total_missing_rows": 5},
            "missing": {"value": {"missing_ratio": 0.1}},
            "outliers": [],
        },
    }
    df = pd.DataFrame({"time": [1, 2, 5], "value": [1.0, 2.0, 3.0]})
    _, report = remediate(df, quality_report)

    statuses = {a["issue"]: a["status"] for a in report.actions}
    assert statuses["time_gaps"] == "deferred_to_imputation"
    assert statuses["missing_values"] == "deferred_to_imputation"


def test_validate_selection_statuses():
    assert validate_selection(None, None)["status"] == "none_configured"
    assert validate_selection("Nope", "x")["status"] == "invalid"
    assert validate_selection("PyPOTS", "not_a_method")["status"] == "invalid"
    assert validate_selection("PyPOTS", "csdi")["status"] == "ok"
    assert validate_selection("PyPOTS", "saits")["status"] == "ok"


def test_validate_selection_flags_known_failing(monkeypatch):
    """The known_failing status resolves — without pinning which method carries it.

    ``known_failing`` records what failed in one *environment*, so its contents
    are expected to change as installs are re-verified (PyPOTS/saits was listed
    from the autofeat-6g env and runs clean in wavestitchplus-repro). Asserting
    against the shipped entry made correcting that data a test failure, so the
    flag is injected here and only the mechanism is under test.
    """
    from dataops import imputation_catalog

    patched = {app: dict(spec) for app, spec in imputation_catalog.CATALOG.items()}
    patched["PyPOTS"]["known_failing"] = ["csdi"]
    monkeypatch.setattr(imputation_catalog, "CATALOG", patched)

    result = imputation_catalog.validate_selection("PyPOTS", "csdi")
    assert result["status"] == "known_failing"
    assert "csdi" in result["message"]
    assert imputation_catalog.validate_selection("PyPOTS", "saits")["status"] == "ok"


def test_pipeline_emits_handoff_with_catalog_and_selection(tmp_path):
    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "processed" / "clean.csv"
    report_json = tmp_path / "reports" / "report.json"
    pd.DataFrame(
        {
            "time": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:01:00",
                "2026-01-01 00:04:00",  # 3-min gap → time gap
                "2026-01-01 00:05:00",
            ],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(input_csv, index=False)

    report = run_pipeline(
        str(input_csv),
        str(output_csv),
        str(report_json),
        timestamp_col="time",
        validation_config={"mode": "time_series", "missing_threshold": 0.5},
        imputation_config={
            "app": "PyPOTS",
            "method": "csdi",
            "build_bundle": True,
        },
    )

    handoff = report["handoff"]
    assert handoff["needs_ts_imputation"] is True
    assert handoff["reason"] == "time_gaps_detected"
    assert "PyPOTS" in handoff["imputation_catalog"]
    assert handoff["selection"]["status"] == "ok"
    assert handoff["selection"]["app"] == "PyPOTS"
    # bundle should be produced and the invoke hint wired
    assert handoff["bundle_written"] is True
    assert handoff["prepared_dir"]
    assert "run_imputation.py" in (handoff["invoke_hint"] or "")
    # report persisted with the new sections
    saved = json.loads(report_json.read_text(encoding="utf-8"))
    assert "remediation" in saved
    assert "handoff" in saved


def test_pipeline_handoff_when_no_gaps(tmp_path):
    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "processed" / "clean.csv"
    report_json = tmp_path / "reports" / "report.json"
    pd.DataFrame(
        {"time": [1, 2, 3, 4], "value": [1.0, 2.0, 3.0, 4.0]}
    ).to_csv(input_csv, index=False)

    report = run_pipeline(
        str(input_csv),
        str(output_csv),
        str(report_json),
        timestamp_col="time",
        validation_config={"mode": "time_series"},
        imputation_config={"app": None, "method": None},
    )

    handoff = report["handoff"]
    assert handoff["needs_ts_imputation"] is False
    assert handoff["bundle_written"] is False
    assert handoff["selection"]["status"] == "none_configured"
