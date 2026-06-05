"""
End-to-end checks of the modules against the four real EUR datasets
(amf / golang / python / rabbitmq).

These read the committed raw CSVs under ``6GDALI_Datasets/EUR/6907619/``. If the
datasets are absent (e.g. a slim checkout), each case skips. A 5000-row head is
used so the suite stays fast while still exercising real schemas.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipeline_modules import profiling
from pipeline_modules.ts_checks import detect_time_gaps

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "6GDALI_Datasets" / "EUR" / "6907619"

RAW_CSV = {
    "amf": "amf-performance.csv",
    "golang": "golang-web-server-performance.csv",
    "python": "python-web-server-performance.csv",
    "rabbitmq": "rabbitmq-performance.csv",
}
SUBSETS = sorted(RAW_CSV)
SAMPLE_ROWS = 5000


def _load(subset: str) -> pd.DataFrame:
    path = DATA_DIR / RAW_CSV[subset]
    if not path.exists():
        pytest.skip(f"raw dataset for {subset} not present ({path})")
    return pd.read_csv(path, nrows=SAMPLE_ROWS)


@pytest.mark.parametrize("subset", SUBSETS)
def test_profile_detects_time_series(subset):
    df = _load(subset)
    p = profiling.profile(df)
    assert p["is_time_series"] is True
    assert p["timestamp_column"] == "time"
    assert "time" not in p["target_cols"]
    assert p["shape"]["rows"] == len(df)
    assert len(p["target_cols"]) == df.shape[1] - 1


@pytest.mark.parametrize("subset", SUBSETS)
def test_profile_with_configured_time_col(subset):
    df = _load(subset)
    p = profiling.profile(df, timestamp_col="time")
    assert p["timestamp_column"] == "time"
    assert p["detected_type"] == "Configured Timestamp Column"


@pytest.mark.parametrize("subset", SUBSETS)
def test_gap_detection_runs(subset):
    df = _load(subset)
    info = detect_time_gaps(df, "time")
    # cadence must be inferable and self-consistent on every real dataset
    assert info["expected_dt_seconds"] is not None
    assert info["expected_dt_seconds"] > 0
    assert info["num_gaps"] >= 0
    assert isinstance(info["has_gaps"], bool)
