from __future__ import annotations

import pandas as pd

from dataops.cleaning import clean_dataframe, coerce_datetime, snake_case, standardize_columns


def test_snake_case_normalizes_names():
    assert snake_case(" CPU Usage (%) ") == "cpu_usage"
    assert snake_case("") == "column"


def test_standardize_columns_makes_duplicate_names_unique():
    df = pd.DataFrame([[1, 2]], columns=["CPU Usage", "cpu-usage"])
    cleaned = standardize_columns(df)
    assert list(cleaned.columns) == ["cpu_usage", "cpu_usage_2"]


def test_clean_dataframe_drops_empty_and_duplicate_rows():
    df = pd.DataFrame(
        {
            " Time ": ["2026-01-01", "2026-01-01", None],
            "Value": [1.0, 1.0, None],
        }
    )
    cleaned, report = clean_dataframe(df, datetime_column="time")
    assert len(cleaned) == 1
    assert report.dropped_empty_rows == 1
    assert report.dropped_duplicate_rows == 1
    assert report.column_mapping == {" Time ": "time", "Value": "value"}
    assert pd.api.types.is_datetime64_any_dtype(cleaned["time"])


def test_coerce_datetime_parses_epoch_seconds_not_nanoseconds():
    # 1636553178 is Nov 2021 in epoch *seconds*; without unit-awareness
    # pd.to_datetime would read it as nanoseconds → 1970.
    df = pd.DataFrame({"time": [1636553178, 1636553188, 1636553198]})
    out = coerce_datetime(df, "time")
    assert pd.api.types.is_datetime64_any_dtype(out["time"])
    assert out["time"].dt.year.eq(2021).all()


def test_coerce_datetime_parses_epoch_milliseconds():
    df = pd.DataFrame({"time": [1636553178000, 1636553188000]})
    out = coerce_datetime(df, "time")
    assert out["time"].dt.year.eq(2021).all()


def test_coerce_datetime_still_parses_iso_strings():
    df = pd.DataFrame({"time": ["2026-01-01 00:00:00", "2026-01-01 00:01:00"]})
    out = coerce_datetime(df, "time")
    assert out["time"].dt.year.eq(2026).all()
