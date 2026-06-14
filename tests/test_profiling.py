from __future__ import annotations

import pandas as pd

from dataops.profiling import classify_dataset, profile


def test_classify_dataset_detects_time_series_from_datetime_column():
    df = pd.DataFrame(
        {
            "when": pd.date_range("2026-01-01", periods=3, freq="h").astype(str),
            "value": [1.0, 2.0, 3.0],
        }
    )

    result = classify_dataset(df)

    assert result["data_type"] == "time_series"
    assert result["timestamp_column"] == "when"


def test_classify_dataset_treats_step_index_as_tabular_by_default():
    df = pd.DataFrame({"id": [1, 2, 3], "label": ["a", "b", "c"]})

    result = classify_dataset(df)

    assert result["data_type"] == "tabular"
    assert result["timestamp_column"] is None
    assert result["detected_type"] == "Step Index"


def test_profile_records_explicit_data_type_and_reason():
    df = pd.DataFrame({"id": [1, 2, 3], "label": ["a", "b", "c"]})

    result = profile(df)

    assert result["data_type"] == "tabular"
    assert result["is_time_series"] is False
    assert "classification_reason" in result
