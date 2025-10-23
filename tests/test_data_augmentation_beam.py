import datetime as dt

import pytest

from methods.data_augmentation_beam import (
    augment_with_time,
    augment_without_time,
    load_and_align_time_series,
)


@pytest.fixture()
def sample_tables(tmp_path):
    table_a = """time,throughput,latency
2024-01-01 00:00:00,100,10.0
2024-01-01 00:01:00,120,11.0
2024-01-01 00:02:00,150,9.5
"""

    table_b = """time,cpu,memory
2024-01-01 00:00:00,0.10,512
2024-01-01 00:01:00,0.15,520
2024-01-01 00:02:00,0.20,530
"""

    path_a = tmp_path / "performance.csv"
    path_a.write_text(table_a)
    path_b = tmp_path / "system.csv"
    path_b.write_text(table_b)

    return [path_a, path_b]


def test_load_and_align_time_series_returns_prefixed_columns(sample_tables):
    rows = load_and_align_time_series(sample_tables)

    assert len(rows) == 3
    first_row = rows[0]
    assert set(first_row.keys()) == {
        "performance_throughput",
        "performance_latency",
        "system_cpu",
        "system_memory",
        "time",
    }
    assert isinstance(first_row["time"], dt.datetime)


def test_augment_without_time_excludes_timestamp(sample_tables):
    rows = augment_without_time(sample_tables)

    assert len(rows) == 3
    assert all("time" not in row for row in rows)
    assert rows[0]["performance_throughput"] == 100
    assert rows[0]["system_cpu"] == 0.1


def test_augment_with_time_adds_temporal_features(sample_tables):
    rows = augment_with_time(sample_tables)

    expected_columns = {
        "time",
        "performance_throughput",
        "performance_latency",
        "system_cpu",
        "system_memory",
        "time_unix",
        "time_year",
        "time_month",
        "time_day",
        "time_hour",
        "time_minute",
        "time_second",
        "time_dayofweek",
        "time_dayofyear",
        "time_iso_week",
        "time_iso_year",
        "time_hour_sin",
        "time_hour_cos",
        "time_minute_sin",
        "time_minute_cos",
        "time_second_sin",
        "time_second_cos",
    }

    assert expected_columns.issubset(rows[0].keys())
    assert rows[0]["time"].startswith("2024-01-01 00:00:00")
    assert rows[0]["time_unix"] < rows[-1]["time_unix"]


def test_load_and_align_time_series_rejects_duplicate_timestamps(tmp_path):
    table = """time,value\n2024-01-01 00:00:00,1\n2024-01-01 00:00:00,2\n"""
    path = tmp_path / "duplicate.csv"
    path.write_text(table)

    with pytest.raises(ValueError, match="Duplicate timestamp"):
        load_and_align_time_series([path])
