"""Unit tests for the pure gap-detection logic in pipeline_modules.ts_checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline_modules.ts_checks import detect_time_gaps, _diffs_in_seconds


def test_regular_grid_has_no_gaps(regular_ts_df):
    info = detect_time_gaps(regular_ts_df, "time")
    assert info["has_gaps"] is False
    assert info["num_gaps"] == 0
    assert info["expected_dt_seconds"] == 1.0


def test_single_gap_detected(gappy_ts_df):
    info = detect_time_gaps(gappy_ts_df, "time")
    assert info["has_gaps"] is True
    assert info["num_gaps"] == 1
    assert info["expected_dt_seconds"] == 1.0
    # last point before hole is 49, first after is 55 -> interval 6s, 5 missing points
    assert info["total_missing_rows"] == 5
    assert info["largest_gap_seconds"] == 6.0


def test_fewer_than_two_timestamps():
    df = pd.DataFrame({"time": [42], "v": [1.0]})
    info = detect_time_gaps(df, "time")
    assert info["has_gaps"] is False
    assert any("fewer than 2" in n for n in info["notes"])


def test_datetime_column_cadence():
    ts = pd.date_range("2024-01-01", periods=100, freq="5min")
    df = pd.DataFrame({"time": ts, "v": range(100)})
    info = detect_time_gaps(df, "time")
    assert info["expected_dt_seconds"] == 300.0
    assert info["has_gaps"] is False


def test_epoch_milliseconds_unit_detection():
    # values ~1.7e12 -> milliseconds; 1000ms cadence -> 1.0s
    base = 1_700_000_000_000
    ts = pd.Series(np.arange(base, base + 50 * 1000, 1000))
    diffs = _diffs_in_seconds(ts)
    assert np.allclose(diffs, 1.0)


def test_gap_factor_threshold():
    # cadence 1; a single 2-step interval is flagged only at a low gap_factor
    t = np.array([0, 1, 2, 4, 5, 6])  # one interval of size 2
    df = pd.DataFrame({"time": t, "v": t.astype(float)})
    assert detect_time_gaps(df, "time", gap_factor=1.5)["num_gaps"] == 1
    assert detect_time_gaps(df, "time", gap_factor=3.0)["num_gaps"] == 0
