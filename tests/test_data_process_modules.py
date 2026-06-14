from __future__ import annotations

import pandas as pd

from data_process_modules import MANIFEST, profiling, transform
from data_process_modules.registry import get


def test_data_process_modules_exposes_public_manifest_entrypoints():
    assert "profiling" in MANIFEST
    assert get("profiling")["entrypoint"] == "data_process_modules.profiling:profile"
    assert get("transform")["entrypoint"] == "data_process_modules.transform:preprocess"


def test_data_process_modules_profile_uses_shared_implementation():
    df = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=3, freq="h").astype(str),
            "value": [1.0, 2.0, 3.0],
        }
    )

    report = profiling.profile(df, timestamp_col="time")

    assert report["data_type"] == "time_series"
    assert report["timestamp_column"] == "time"


def test_transform_preprocess_keeps_arbitrary_tabular_data_tabular():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "segment": ["a", "b", "a"],
            "score": [0.1, 0.7, 0.4],
        }
    )

    transformed, meta = transform.preprocess(df)

    assert transformed.equals(df)
    assert meta["data_type"] == "tabular"
    assert meta["time_col"] is None
    assert meta["target_cols"] == ["customer_id", "score"]


def test_transform_preprocess_regularizes_time_series_safely():
    df = pd.DataFrame(
        {
            "time": ["2026-01-01 00:00:00", "2026-01-01 00:02:00"],
            "value": [1.0, 3.0],
            "label": ["first", "second"],
        }
    )

    transformed, meta = transform.preprocess(
        df,
        timestamp_col="time",
        base_dt=60,
    )

    assert meta["data_type"] == "time_series"
    assert meta["target_cols"] == ["value"]
    assert meta["dropped_non_numeric_targets"] == []
    assert len(transformed) == 3
    assert transformed["value"].isna().sum() == 1
    assert "is_gap" in transformed.columns


def test_transform_preprocess_reports_non_numeric_time_series_targets():
    df = pd.DataFrame(
        {
            "time": ["2026-01-01 00:00:00", "2026-01-01 00:01:00"],
            "value": [1.0, 2.0],
            "label": ["first", "second"],
        }
    )

    transformed, meta = transform.preprocess(
        df,
        timestamp_col="time",
        target_cols=["value", "label"],
    )

    assert list(transformed.columns[:2]) == ["time", "value"]
    assert meta["target_cols"] == ["value"]
    assert meta["dropped_non_numeric_targets"] == ["label"]
