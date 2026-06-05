"""Unit tests for the WaveStitch+ v2 local-anchoring layer (``wsp_v2``)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wsp_v2 import (
    _distance_to_observed, anchor_blend, build_prior, default_monotone_groups,
    enforce_monotone_groups, score_holdout,
)

TARGETS = ["a", "b"]


def _toy_test_input():
    # 6 rows; row 2 and rows 4-5 missing in 'a'; row 3 missing in 'b'.
    return pd.DataFrame({
        "time": np.arange(6.0),
        "a": [10.0, 20.0, np.nan, 40.0, np.nan, np.nan],
        "b": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0],
    })


def test_distance_to_observed_with_left_context():
    missing = np.array([False, True, True, False])
    # No left context: leading run measures distance within the array.
    d = _distance_to_observed(missing, left_context=0)
    assert d.tolist() == [0.0, 1.0, 1.0, 0.0]
    # With a virtual observed row at index -1, the first missing cell is closer.
    missing2 = np.array([True, True, False])
    d2 = _distance_to_observed(missing2, left_context=1)
    assert d2[0] == 1.0  # one step from the train boundary at -1


def test_build_prior_uses_train_context():
    train = pd.DataFrame({"a": [0.0, 5.0], "b": [0.0, 0.0]})
    ti = _toy_test_input()
    prior = build_prior(train, ti, TARGETS, method="linear")
    assert len(prior) == len(ti)
    # Leading/internal gaps are filled (no NaN remains in the prior).
    assert not prior[TARGETS].isna().any().any()
    # Internal linear gap in 'a' (between 20 and 40) interpolates to 30.
    assert prior["a"].iloc[2] == 30.0


def test_anchor_blend_preserves_observed_cells():
    ti = _toy_test_input()
    # Diffusion proposes absurd values everywhere.
    diff = ti.copy()
    for c in TARGETS:
        diff[c] = 999.0
    prior = build_prior(None, ti, TARGETS, method="linear")
    out = anchor_blend(ti, diff, prior, TARGETS, tau=20, hard_prior=8,
                       has_left_context=False)
    # Observed cells are byte-for-byte preserved.
    for c in TARGETS:
        obs = ti[c].notna().to_numpy()
        assert np.allclose(out[c].to_numpy()[obs], ti[c].to_numpy()[obs])
    # No NaNs left in the targets after filling.
    assert not out[TARGETS].isna().any().any()


def test_hard_prior_follows_prior_near_observations():
    ti = _toy_test_input()
    diff = ti.copy()
    for c in TARGETS:
        diff[c] = 999.0
    prior = build_prior(None, ti, TARGETS, method="linear")
    # hard_prior covers the whole short series -> output equals prior at gaps.
    out = anchor_blend(ti, diff, prior, TARGETS, tau=1, hard_prior=8,
                       has_left_context=False)
    miss = ti["a"].isna().to_numpy()
    assert np.allclose(out["a"].to_numpy()[miss], prior["a"].to_numpy()[miss])


def test_default_monotone_groups_detects_latency_percentiles():
    cols = ["cpu_usage", "lat100_ms", "lat50_ms", "lat99_ms", "n", "lat75"]
    groups = default_monotone_groups(cols)
    assert groups == [["lat50_ms", "lat75", "lat99_ms", "lat100_ms"]]
    # No latency cols -> no groups.
    assert default_monotone_groups(["cpu_usage", "ram_mb"]) == []


def test_enforce_monotone_fully_imputed_row_is_sorted():
    cols = ["lat50_ms", "lat75_ms", "lat100_ms"]
    df = pd.DataFrame({"lat50_ms": [90.0], "lat75_ms": [10.0], "lat100_ms": [50.0]})
    miss = pd.DataFrame({c: [True] for c in cols})
    enforce_monotone_groups(df, miss, [cols])
    assert df.iloc[0].tolist() == [10.0, 50.0, 90.0]  # ascending


def test_enforce_monotone_preserves_observed_anchors():
    cols = ["lat50_ms", "lat75_ms", "lat100_ms"]
    # middle observed=60; ends imputed and out of order (200, _, 5)
    df = pd.DataFrame({"lat50_ms": [200.0], "lat75_ms": [60.0], "lat100_ms": [5.0]})
    miss = pd.DataFrame({"lat50_ms": [True], "lat75_ms": [False], "lat100_ms": [True]})
    enforce_monotone_groups(df, miss, [cols])
    row = df.iloc[0].tolist()
    assert row[1] == 60.0                       # observed anchor untouched
    assert row[0] <= row[1] <= row[2]           # non-decreasing
    assert row[0] == 60.0 and row[2] == 60.0    # clamped to the anchor


def test_anchor_blend_monotone_groups_removes_violations():
    ti = pd.DataFrame({
        "time": np.arange(4.0),
        "lat50_ms": [1.0, np.nan, np.nan, 4.0],
        "lat75_ms": [2.0, np.nan, np.nan, 5.0],
        "lat100_ms": [3.0, np.nan, np.nan, 6.0],
    })
    tcols = ["lat50_ms", "lat75_ms", "lat100_ms"]
    diff = ti.copy()
    diff["lat50_ms"] = [1, 99, 80, 4]   # gap rows non-monotone
    diff["lat75_ms"] = [2, 10, 81, 5]
    diff["lat100_ms"] = [3, 50, 2, 6]
    prior = build_prior(None, ti, tcols, method="linear")
    out = anchor_blend(ti, diff, prior, tcols, tau=1e-9, hard_prior=0,
                       has_left_context=False,
                       monotone_groups=[["lat50_ms", "lat75_ms", "lat100_ms"]])
    arr = out[tcols].to_numpy(float)
    assert (np.diff(arr, axis=1) >= -1e-9).all()  # every row non-decreasing


def test_score_holdout_only_counts_masked_observed_cells():
    ti = _toy_test_input()
    gt = ti.copy()
    gt["a"] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]  # truth known at gaps
    gt["b"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    perfect = gt.copy()
    s = score_holdout(ti, gt, perfect, TARGETS)
    # 3 missing-in-a + 1 missing-in-b = 4 scored cells, zero error.
    assert s["n_cells"] == 4
    assert s["MAE"] == 0.0
