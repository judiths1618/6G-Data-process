from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipelines.minimal_dataops import run_from_config, run_pipeline


def test_minimal_pipeline_writes_clean_data_and_report(tmp_path):
    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "processed" / "clean.csv"
    report_json = tmp_path / "reports" / "report.json"
    pd.DataFrame(
        {
            "Time": [1, 1, 2, None],
            "CPU Usage": [0.5, 0.5, 0.7, None],
        }
    ).to_csv(input_csv, index=False)

    report = run_pipeline(
        str(input_csv),
        str(output_csv),
        str(report_json),
        timestamp_col="time",
    )

    cleaned = pd.read_csv(output_csv)
    assert list(cleaned.columns) == ["time", "cpu_usage"]
    assert len(cleaned) == 2
    assert report["cleaning"]["dropped_duplicate_rows"] == 1
    assert report["cleaning"]["column_mapping"] == {"Time": "time", "CPU Usage": "cpu_usage"}
    assert report_json.exists()
    # the conservative-clean frame is also persisted as its own artifact
    soft_cleaned_artifact = Path(report["soft_cleaned_output"])
    assert soft_cleaned_artifact.exists()
    assert soft_cleaned_artifact.name == "clean_soft_cleaned.csv"
    assert report["cleaned_output"] == report["soft_cleaned_output"]
    assert list(pd.read_csv(soft_cleaned_artifact).columns) == ["time", "cpu_usage"]


def test_minimal_pipeline_can_run_from_config(tmp_path):
    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "processed" / "clean.csv"
    report_json = tmp_path / "reports" / "report.json"
    config_yaml = tmp_path / "dataops.yaml"
    pd.DataFrame({"time": [1, 2], "value": [0.5, 0.7]}).to_csv(input_csv, index=False)
    config_yaml.write_text(
        f"""
input: {input_csv}
output: {output_csv}
report: {report_json}
timestamp_col: time
validation:
  expected_columns: [time, value]
  numeric_bounds:
    value:
      min: 0.0
      max: 1.0
""",
        encoding="utf-8",
    )

    report = run_from_config(str(config_yaml))

    assert report["validation"]["pandera_passed"] is True
    assert output_csv.exists()


def test_minimal_pipeline_writes_report_on_validation_failure(tmp_path):
    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "processed" / "clean.csv"
    report_json = tmp_path / "reports" / "report.json"
    pd.DataFrame({"time": [1, 2], "value": [0.5, 3.0]}).to_csv(input_csv, index=False)

    with pytest.raises(Exception):
        run_pipeline(
            str(input_csv),
            str(output_csv),
            str(report_json),
            timestamp_col="time",
            validation_config={"numeric_bounds": {"value": {"max": 1.0}}},
        )

    assert report_json.exists()
    assert '"pandera_passed": false' in report_json.read_text(encoding="utf-8")


def test_minimal_pipeline_marks_non_monotonic_timestamp_as_quality_issue(tmp_path):
    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "processed" / "clean.csv"
    report_json = tmp_path / "reports" / "report.json"
    pd.DataFrame({"time": [1, 3, 2, 4], "value": [0.5, 0.7, 0.6, 0.8]}).to_csv(
        input_csv, index=False
    )

    report = run_pipeline(
        str(input_csv),
        str(output_csv),
        str(report_json),
        timestamp_col="time",
        validation_config={
            "mode": "time_series",
            "require_timestamp_monotonic": True,
        },
    )

    assert output_csv.exists()
    assert report_json.exists()
    assert report["validation"]["pandera_passed"] is True
    assert report["validation"]["errors"] == []
    assert report["cleaning"]["non_monotonic_timestamps"] == 1
    assert report["quality"]["issue_summary"]["timestamp_order"] >= 1
    assert any(
        action["issue"] == "timestamp_not_monotonic"
        for action in report["quality"]["action_plan"]
    )
    assert report["validation_comparison"]["cleaning_effect"][
        "non_monotonic_timestamps_sorted"
    ] == 1
    assert report["validation_comparison"]["validation_status"]["pandera_passed"] is True


def test_minimal_pipeline_handles_arbitrary_tabular_data(tmp_path):
    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "processed" / "clean.csv"
    report_json = tmp_path / "reports" / "report.json"
    pd.DataFrame(
        {
            "Customer ID": [1, 2, 3],
            "Plan": ["basic", "pro", "enterprise"],
            "Spend": [12.5, 55.0, 120.0],
        }
    ).to_csv(input_csv, index=False)

    report = run_pipeline(
        str(input_csv),
        str(output_csv),
        str(report_json),
        validation_config={
            "mode": "auto",
            "expected_columns": ["customer_id", "plan", "spend"],
            "numeric_bounds": {"spend": {"min": 0.0}},
        },
    )

    cleaned = pd.read_csv(output_csv)
    assert list(cleaned.columns) == ["customer_id", "plan", "spend"]
    assert report["validation"]["mode"] == "tabular"
    assert report["validation"]["pandera_passed"] is True
    assert report["quality"]["mode"] == "tabular"
    assert "action_plan" in report["quality"]
    assert report["validation_comparison"]["validation_status"]["pandera_passed"] is True


def test_auto_mode_does_not_treat_monotonic_id_as_timestamp(tmp_path):
    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "processed" / "clean.csv"
    report_json = tmp_path / "reports" / "report.json"
    pd.DataFrame({"id": [1, 2, 3, 4], "label": ["a", "b", "c", "d"]}).to_csv(
        input_csv, index=False
    )

    report = run_pipeline(
        str(input_csv),
        str(output_csv),
        str(report_json),
        validation_config={"mode": "auto"},
    )

    assert report["profile"]["data_type"] == "tabular"
    assert report["profile"]["detected_type"] == "Step Index"
    assert report["validation"]["mode"] == "tabular"
    assert report["validation"]["timestamp_column"] is None


def test_minimal_pipeline_handles_arbitrary_tabular_without_timestamp(tmp_path):
    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "processed" / "clean.csv"
    report_json = tmp_path / "reports" / "report.json"
    pd.DataFrame(
        {
            "Customer ID": [1, 2, 3],
            "Segment": ["a", "b", "a"],
            "Score": [0.1, 0.7, 0.4],
        }
    ).to_csv(input_csv, index=False)

    report = run_pipeline(
        str(input_csv),
        str(output_csv),
        str(report_json),
        validation_config={
            "mode": "auto",
            "expected_columns": ["customer_id", "segment", "score"],
            "numeric_bounds": {"score": {"min": 0.0, "max": 1.0}},
        },
    )

    cleaned = pd.read_csv(output_csv)
    assert list(cleaned.columns) == ["customer_id", "segment", "score"]
    assert report["validation"]["mode"] == "tabular"
    assert report["validation"]["pandera_passed"] is True
    assert report["validation"]["timestamp_column"] is None


def test_minimal_pipeline_rejects_unknown_validation_mode(tmp_path):
    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "processed" / "clean.csv"
    report_json = tmp_path / "reports" / "report.json"
    pd.DataFrame({"name": ["a"], "score": [0.4]}).to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="validation.mode"):
        run_pipeline(
            str(input_csv),
            str(output_csv),
            str(report_json),
            validation_config={"mode": "mystery"},
        )

    assert report_json.exists()


def test_minimal_pipeline_reports_quality_actions_and_visual_comparison(tmp_path):
    input_csv = tmp_path / "raw.csv"
    output_csv = tmp_path / "processed" / "clean.csv"
    report_json = tmp_path / "reports" / "report.json"
    pd.DataFrame(
        {
            "time": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:01:00",
                "2026-01-01 00:04:00",
                "2026-01-01 00:05:00",
            ],
            "value": [1.0, 2.0, 100.0, 3.0],
        }
    ).to_csv(input_csv, index=False)

    report = run_pipeline(
        str(input_csv),
        str(output_csv),
        str(report_json),
        timestamp_col="time",
        validation_config={
            "mode": "time_series",
            "require_timestamp_unique": True,
            "require_timestamp_monotonic": True,
            "missing_threshold": 0.5,
        },
    )

    assert report["quality"]["mode"] == "time_series"
    assert report["quality"]["issue_summary"]["ts_gaps"] >= 1
    assert any(
        action["issue"] == "time_gaps"
        for action in report["quality"]["action_plan"]
    )
    assert report["validation_comparison"]["dataset_shape"]["raw"]["rows"] == 4
    assert report["validation_comparison"]["dataset_shape"]["soft_cleaned"]["rows"] == 4
    assert report["validation_comparison"]["dataset_shape"]["cleaned"]["rows"] == 4
    assert any(
        row["stage"] == "soft_cleaned"
        for row in report["validation_comparison"]["chart_ready"]
    )
    assert report["validation_comparison"]["chart_ready"]
