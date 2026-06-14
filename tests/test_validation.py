from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("pandera")

from dataops.validation import validate_numeric_timeseries, validate_tabular_dataframe


def test_validate_numeric_timeseries_accepts_numeric_frame():
    df = pd.DataFrame({"time": [1, 2, 3], "cpu": [0.1, 0.2, 0.3]})
    validated = validate_numeric_timeseries(df, timestamp_col="time")
    assert list(validated.columns) == ["time", "cpu"]


def test_validate_numeric_timeseries_rejects_duplicate_timestamps():
    df = pd.DataFrame({"time": [1, 1, 2], "cpu": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_numeric_timeseries(df, timestamp_col="time")


def test_validate_numeric_timeseries_applies_numeric_bounds():
    df = pd.DataFrame({"time": [1, 2, 3], "cpu": [0.1, 5.0, 0.3]})
    with pytest.raises(Exception):
        validate_numeric_timeseries(
            df,
            timestamp_col="time",
            numeric_bounds={"cpu": {"max": 1.0}},
        )


def test_validate_numeric_timeseries_requires_timestamp_column():
    df = pd.DataFrame({"cpu": [0.1, 0.2, 0.3]})
    with pytest.raises(KeyError):
        validate_numeric_timeseries(df, timestamp_col="time")


def test_validate_tabular_dataframe_accepts_no_timestamp():
    df = pd.DataFrame({"city": ["a", "b"], "score": [0.2, 0.8]})
    validated = validate_tabular_dataframe(
        df,
        expected_columns=["city", "score"],
        numeric_bounds={"score": {"min": 0.0, "max": 1.0}},
    )
    assert list(validated.columns) == ["city", "score"]


def test_validate_tabular_dataframe_rejects_missing_expected_columns():
    df = pd.DataFrame({"city": ["a", "b"]})
    with pytest.raises(ValueError, match="missing expected"):
        validate_tabular_dataframe(df, expected_columns=["city", "score"])
