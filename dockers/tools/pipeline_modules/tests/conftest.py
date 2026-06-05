"""
Shared pytest fixtures + path setup for pipeline_modules tests.

Adds ``dockers/tools`` to ``sys.path`` so ``import pipeline_modules`` resolves
regardless of the working directory pytest is launched from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# .../dockers/tools/pipeline_modules/tests/conftest.py  -> parents[2] == dockers/tools
TOOLS_DIR = Path(__file__).resolve().parents[2]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

REPO_ROOT = Path(__file__).resolve().parents[4]
PREPARED_AMF = REPO_ROOT / "experiments" / "EUR" / "prepared_amf"


@pytest.fixture
def regular_ts_df() -> pd.DataFrame:
    """A clean, gap-free time series on a 1-second grid with two target cols."""
    n = 200
    t = np.arange(n, dtype=np.int64)  # unitless/seconds grid
    return pd.DataFrame({
        "time": t,
        "cpu": np.sin(t / 10.0) + 1.0,
        "mem": np.cos(t / 10.0) + 1.0,
    })


@pytest.fixture
def gappy_ts_df() -> pd.DataFrame:
    """A 1-second grid with a single 5-step hole in the timeline."""
    t = np.concatenate([np.arange(0, 50), np.arange(55, 100)]).astype(np.int64)
    return pd.DataFrame({"time": t, "v": t.astype(float)})


@pytest.fixture
def tabular_df() -> pd.DataFrame:
    """Non-TS tabular frame with a unique id, a NaN-laden col, and an outlier."""
    n = 120
    rng = np.random.default_rng(0)
    val = rng.normal(10.0, 1.0, size=n)
    val[0] = 9999.0  # clear outlier
    miss = rng.normal(5.0, 1.0, size=n)
    miss[:20] = np.nan  # ~17% missing
    return pd.DataFrame({
        "id": np.arange(n),                 # unique → primary key
        "category": (["a", "b", "c"] * n)[:n],
        "value": val,
        "sometimes_missing": miss,
    })


@pytest.fixture(scope="session")
def prepared_amf_available() -> bool:
    return (PREPARED_AMF / "meta.json").exists()
