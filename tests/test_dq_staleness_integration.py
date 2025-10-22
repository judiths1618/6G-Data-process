import datetime as dt
import math
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dq_local_beam import (
    DEFAULT_RULE,
    RowCtx,
    _accumulate_numeric_values,
    validate_row_against_rule,
)


def _simple_row(header, data):
    return RowCtx(file="example.csv", rownum=1, header=header, data=data)


def test_validate_row_attaches_default_staleness_score() -> None:
    reference_time = dt.datetime(2024, 3, 1, 12, tzinfo=dt.timezone.utc)
    event_time = reference_time - dt.timedelta(hours=3)

    rule = dict(DEFAULT_RULE)
    rule["event_time_col"] = "time"
    rule["event_time_format"] = "iso"
    rule["freshness_slo_hours"] = 24

    rc = _simple_row(["time"], {"time": event_time.isoformat()})

    valid, issues = validate_row_against_rule(rc, rule, None, reference_time=reference_time)

    assert issues == []
    assert valid is rc

    metrics = getattr(rc, "_dq_computed_metrics", {})
    assert "staleness_score" in metrics
    expected = 1 - (3 / 24)
    assert math.isclose(metrics["staleness_score"], expected, rel_tol=1e-6)


def test_staleness_uses_custom_columns_and_units() -> None:
    reference_time = dt.datetime(2024, 3, 1, tzinfo=dt.timezone.utc)
    event_time = dt.datetime(2024, 2, 29, 21, tzinfo=dt.timezone.utc)
    ingest_time = event_time + dt.timedelta(hours=2)
    delivery_time = ingest_time + dt.timedelta(hours=1)

    rule = dict(DEFAULT_RULE)
    rule["event_time_col"] = "event_time"
    rule["event_time_format"] = "iso"
    rule["staleness"] = {
        "input_time_col": "ingest",
        "input_time_format": "iso",
        "delivery_time_col": "delivery",
        "delivery_time_format": "iso",
        "age_col": "age_hours",
        "age_unit": "hours",
        "validity_duration_hours": 24,
        "score_column": "freshness_score",
    }

    rc = _simple_row(
        ["event_time", "ingest", "delivery", "age_hours"],
        {
            "event_time": event_time.isoformat(),
            "ingest": ingest_time.isoformat(),
            "delivery": delivery_time.isoformat(),
            "age_hours": "2",
        },
    )

    valid, issues = validate_row_against_rule(rc, rule, None, reference_time=reference_time)
    assert issues == []
    assert valid is rc

    metrics = getattr(rc, "_dq_computed_metrics", {})
    assert "freshness_score" in metrics
    expected = 1 - (3 / 24)
    assert math.isclose(metrics["freshness_score"], expected, rel_tol=1e-6)


def test_accumulate_numeric_values_infers_staleness() -> None:
    reference_time = dt.datetime(2024, 3, 2, tzinfo=dt.timezone.utc)
    event_time = reference_time - dt.timedelta(hours=6)

    rule = dict(DEFAULT_RULE)
    rule["event_time_col"] = "time"
    rule["event_time_format"] = "iso"
    rule["freshness_slo_hours"] = 12

    rc = _simple_row(["time"], {"time": event_time.isoformat()})
    alias = {"time": "time"}
    setattr(rc, "_dq_header_alias", alias)

    global_values = {}
    per_file_values = {"example.csv": {}}
    per_file_series = {"example.csv": {}}

    _accumulate_numeric_values(
        rc,
        rule,
        global_values=global_values,
        per_file_values=per_file_values,
        file_path="example.csv",
        per_file_series=per_file_series,
        event_time_col="time",
        event_time_format="iso",
        reference_time=reference_time,
    )

    assert "staleness_score" in global_values
    assert global_values["staleness_score"]
    assert per_file_values["example.csv"]["staleness_score"]
    assert per_file_series["example.csv"]["staleness_score"]
