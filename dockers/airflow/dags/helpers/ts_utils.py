# helpers/ts_utils.py
"""
Time-series quality helpers used by the DAG.

`detect_time_gaps` is the single source of truth for "are the timestamps
regular?". It works on either a datetime64 series or a numeric series (any
unit) and returns metrics the cleaning step uses to decide whether to route
through WaveStitchPlus's preprocess_csv for regularization.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _diffs_in_seconds(ts: pd.Series) -> pd.Series:
    """
    Return diff(ts) expressed in seconds, regardless of whether ``ts`` is
    datetime64, numeric epoch seconds, numeric epoch milliseconds, or some
    other unitless numeric index.
    """
    if pd.api.types.is_datetime64_any_dtype(ts):
        return ts.diff().dropna().dt.total_seconds()

    s = pd.to_numeric(ts, errors="coerce").dropna()
    diffs = s.diff().dropna()
    if len(diffs) == 0:
        return diffs
    # Auto-detect unit from the median value (same heuristic as the dashboard).
    med = float(s.median())
    if 1e12 <= med < 1e14:
        # milliseconds since epoch
        return diffs / 1000.0
    # Already in seconds (or unitless — caller treats output as the timeline's
    # native unit; we just label it "seconds" for downstream code).
    return diffs


def detect_time_gaps(
    df: pd.DataFrame,
    ts_col: str,
    *,
    gap_factor: float = 1.5,
    min_gap_seconds: float | None = None,
) -> dict:
    """
    Detect timeline irregularities in ``df[ts_col]``.

    A "gap" is an interval between consecutive timestamps that exceeds either
    ``min_gap_seconds`` (when provided) or ``expected_dt * gap_factor``.

    Returns a dict with:
        expected_dt_seconds   modal interval (in seconds)
        num_gaps              count of intervals classified as gaps
        has_gaps              True iff num_gaps > 0
        gap_pct               estimated fraction of the regular grid that is
                              missing (0..1). Useful to decide if
                              regularization is worth the row-count blow-up.
        total_missing_rows    estimated count of grid points the regularized
                              timeline would have to synthesize
        largest_gap_seconds   biggest single interval observed
        sample_gap_indices    first 5 row indices flagged as gap starts
        notes                 informational notes (e.g. "fewer than 2 points")

    Robust to a single-row dataset (returns has_gaps=False with a note rather
    than raising).
    """
    info: dict = {
        "expected_dt_seconds": None,
        "num_gaps": 0,
        "has_gaps": False,
        "gap_pct": 0.0,
        "total_missing_rows": 0,
        "largest_gap_seconds": 0.0,
        "sample_gap_indices": [],
        "notes": [],
    }

    ts = df[ts_col].dropna().sort_values()
    if len(ts) < 2:
        info["notes"].append("fewer than 2 timestamps — cannot infer cadence")
        return info

    diffs = _diffs_in_seconds(ts)
    diffs = diffs[diffs > 0]
    if diffs.empty:
        info["notes"].append("no positive-duration intervals")
        return info

    # Modal interval is the most common cadence — robust against a handful of
    # outlier gaps. ``.mode()`` returns a Series; pick the first.
    modes = diffs.mode()
    expected_dt = float(modes.iloc[0]) if len(modes) else float(diffs.median())
    if expected_dt <= 0:
        info["notes"].append("non-positive modal interval; falling back to median")
        expected_dt = float(diffs.median())
    info["expected_dt_seconds"] = expected_dt

    threshold = min_gap_seconds if min_gap_seconds is not None else expected_dt * gap_factor
    flagged = diffs[diffs > threshold]
    info["num_gaps"] = int(len(flagged))
    info["has_gaps"] = info["num_gaps"] > 0
    info["sample_gap_indices"] = flagged.index[:5].tolist()
    info["largest_gap_seconds"] = float(diffs.max())

    # Estimate how many grid points the regularized timeline would have to
    # synthesize. For each flagged gap, (gap / expected_dt) - 1 cells are missing.
    if info["has_gaps"]:
        extra = ((flagged / expected_dt) - 1).clip(lower=0).round().astype(int)
        info["total_missing_rows"] = int(extra.sum())

    # Fraction of the regular grid that's missing.
    grid_rows = int(np.ceil((float(diffs.sum()) + expected_dt) / expected_dt))
    if grid_rows > 0:
        info["gap_pct"] = max(0.0, 1.0 - len(ts) / grid_rows)

    return info
