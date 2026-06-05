"""Unit tests for pipeline_modules.profiling (pure — no GX, no I/O)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline_modules import profiling


# --------------------------------------------------------------------------- #
# analyze_time_series
# --------------------------------------------------------------------------- #

def test_configured_column_short_circuits():
    df = pd.DataFrame({"ts": [1, 2, 3], "x": [9, 9, 9]})
    r = profiling.analyze_time_series(df, configured_name="ts")
    assert r["is_time_series"] is True
    assert r["timestamp_column"] == "ts"
    assert r["detected_type"] == "Configured Timestamp Column"


def test_configured_column_missing_falls_through():
    df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    r = profiling.analyze_time_series(df, configured_name="not_here")
    # 'a' is not datetime-parseable, 'b' is a tiny monotonic step index
    assert r["timestamp_column"] in (None, "b")


def test_datetime_string_detection():
    df = pd.DataFrame({
        "when": pd.date_range("2024-01-01", periods=50, freq="h").astype(str),
        "v": range(50),
    })
    r = profiling.analyze_time_series(df)
    assert r["is_time_series"] is True
    assert r["timestamp_column"] == "when"
    assert r["detected_type"] == "Datetime String"


def test_unix_timestamp_detection():
    base = 1_700_000_000  # ~2023, within (1e9, 3e9)
    df = pd.DataFrame({"epoch": np.arange(base, base + 30), "v": range(30)})
    r = profiling.analyze_time_series(df)
    assert r["timestamp_column"] == "epoch"
    assert r["detected_type"] == "Unix Timestamp"


def test_step_index_detection():
    df = pd.DataFrame({"step": np.arange(1000), "v": np.random.rand(1000)})
    r = profiling.analyze_time_series(df)
    assert r["is_time_series"] is True
    assert r["timestamp_column"] == "step"
    assert r["detected_type"] == "Step Index"


def test_not_time_series():
    df = pd.DataFrame({"a": ["p", "q", "r"], "b": [3, 1, 2]})  # non-monotonic, non-date
    r = profiling.analyze_time_series(df)
    assert r["is_time_series"] is False
    assert r["timestamp_column"] is None


# --------------------------------------------------------------------------- #
# detect_primary_key
# --------------------------------------------------------------------------- #

def test_single_column_primary_key():
    df = pd.DataFrame({"id": range(100), "v": [1] * 100})
    pk = profiling.detect_primary_key(df)
    assert pk["type"] == "single"
    assert pk["columns"] == ["id"]
    assert pk["is_hard_pk"] is True


def test_composite_primary_key():
    a = np.repeat(np.arange(10), 10)
    b = np.tile(np.arange(10), 10)
    # 'v' is intentionally non-unique so it can't be picked as a single-column PK
    df = pd.DataFrame({"a": a, "b": b, "v": np.zeros(100)})
    pk = profiling.detect_primary_key(df)
    # neither a nor b alone is unique, but (a,b) is
    assert pk["type"] == "composite"
    assert set(pk["columns"]) == {"a", "b"}


def test_no_primary_key():
    df = pd.DataFrame({"x": [1, 1, 2, 2], "y": ["a", "a", "b", "b"]})
    pk = profiling.detect_primary_key(df)
    assert pk["type"] == "none"
    assert pk["columns"] == []


# --------------------------------------------------------------------------- #
# profile
# --------------------------------------------------------------------------- #

def test_profile_excludes_ts_from_targets(regular_ts_df):
    p = profiling.profile(regular_ts_df, timestamp_col="time")
    assert p["timestamp_column"] == "time"
    assert "time" not in p["target_cols"]
    assert set(p["target_cols"]) == {"cpu", "mem"}
    assert p["shape"] == {"rows": 200, "cols": 3}
    assert p["columns"] == ["time", "cpu", "mem"]
    assert len(p["preview"]) == 5
    assert "cpu" in p["numeric_cols"]


def test_profile_typing_split(tabular_df):
    p = profiling.profile(tabular_df, timestamp_col=None)
    assert "category" in p["categorical_cols"]
    assert "value" in p["numeric_cols"]
    assert p["primary_key"]["columns"] == ["id"]
