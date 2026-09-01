"""Tests for dataops.timeline — run segmentation, collisions, cadence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataops import timeline


def _sweep_frame() -> pd.DataFrame:
    """Two rows per second at three sweep levels — collisions, but no duplicates."""
    return pd.DataFrame({
        "time":      [10, 10, 20, 20, 30, 40],
        "ram_limit": ["1024M", "2048M", "1024M", "2048M", "1024M", "2048M"],
        "latency":   [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })


# --------------------------------------------------------------------------- #
# Key inference
# --------------------------------------------------------------------------- #

def test_infer_key_columns_finds_the_sweep_factor():
    info = timeline.infer_key_columns(_sweep_frame(), "time")
    assert info["key_columns"] == ["ram_limit"]
    assert info["collisions_on_timestamp"] == 2
    assert info["residual_collisions"] == 0


def test_infer_key_columns_noop_when_timestamps_are_unique():
    df = pd.DataFrame({"time": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
    info = timeline.infer_key_columns(df, "time")
    assert info["key_columns"] == []
    assert info["collisions_on_timestamp"] == 0


# --------------------------------------------------------------------------- #
# Collision resolution
# --------------------------------------------------------------------------- #

def test_distinct_sweep_conditions_are_preserved_not_collapsed():
    df = _sweep_frame()
    out, report = timeline.resolve_collisions(df, "time", key_columns=["ram_limit"])
    assert len(out) == len(df)
    assert report["rows_removed"] == 0
    assert report["collisions_on_timestamp"] == 2
    assert report["distinct_conditions_preserved"] == 2
    # every (time, ram_limit) pair survives
    assert set(zip(out["time"], out["ram_limit"])) == set(zip(df["time"], df["ram_limit"]))


def test_genuine_duplicates_are_averaged_under_the_aggregate_policy():
    df = pd.DataFrame({
        "time":      [10, 10, 20],
        "ram_limit": ["1024M", "1024M", "2048M"],
        "latency":   [2.0, 4.0, 9.0],
    })
    out, report = timeline.resolve_collisions(df, "time", key_columns=["ram_limit"])
    assert len(out) == 2
    assert report["rows_removed"] == 1
    assert report["groups_aggregated"] == 1
    assert out.loc[out["time"] == 10, "latency"].iloc[0] == pytest.approx(3.0)


def test_keep_last_reproduces_the_legacy_behaviour():
    df = _sweep_frame()
    out, report = timeline.resolve_collisions(
        df, "time", key_columns=[], policy="keep_last"
    )
    assert len(out) == 4          # one row per distinct timestamp
    assert report["rows_removed"] == 2


def test_collision_resolution_preserves_input_row_order():
    df = pd.DataFrame({
        "time":  [30, 10, 10, 20],
        "group": ["b", "a", "a", "c"],
        "v":     [1.0, 2.0, 4.0, 8.0],
    })
    out, _ = timeline.resolve_collisions(df, "time", key_columns=["group"])
    assert out["time"].tolist() == [30, 10, 20]


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown collision policy"):
        timeline.resolve_collisions(_sweep_frame(), "time", policy="nope")


# --------------------------------------------------------------------------- #
# Run detection
# --------------------------------------------------------------------------- #

def test_restarted_run_is_segmented_not_sorted():
    # 12 rows re-covering the first run's span → a genuine second run.
    first = list(range(100, 160, 5))
    second = list(range(110, 170, 5))
    df = pd.DataFrame({"time": first + second})
    runs = timeline.detect_runs(df, "time", min_overlap_rows=8)
    assert runs["num_runs"] == 2
    assert runs["has_overlapping_runs"]
    assert runs["boundaries"][0]["row_index"] == len(first)
    assert runs["run_id"][: len(first)].tolist() == [0] * len(first)


def test_single_out_of_order_row_is_not_a_run():
    df = pd.DataFrame({"time": [1, 3, 2, 4]})
    runs = timeline.detect_runs(df, "time", min_overlap_rows=8)
    assert runs["num_runs"] == 1
    assert runs["ignored_backward_steps"] == 1
    assert not runs["is_monotonic_increasing"]


def test_monotonic_frame_has_one_run():
    df = pd.DataFrame({"time": [1, 2, 3, 4]})
    runs = timeline.detect_runs(df, "time")
    assert runs["num_runs"] == 1
    assert runs["boundaries"] == []
    assert runs["rows_out_of_order"] == 0


# --------------------------------------------------------------------------- #
# Time disorder
# --------------------------------------------------------------------------- #

def test_drop_removes_out_of_order_rows_and_leaves_a_rising_timeline():
    df = pd.DataFrame({"time": [1, 3, 2, 4], "v": [10, 30, 20, 40]})
    out, report = timeline.enforce_monotonic(df, "time", policy="drop")
    assert out["time"].tolist() == [1, 3, 4]
    assert report["rows_dropped"] == 1
    assert report["backward_steps"] == 1
    assert not report["was_monotonic"]
    assert out["time"].is_monotonic_increasing


def test_drop_removes_a_whole_overlapping_block():
    # Second acquisition re-covers the first one's span: the conventional
    # treatment removes it rather than interleaving it.
    df = pd.DataFrame({"time": [100, 110, 120, 130, 105, 115, 125, 140]})
    out, report = timeline.enforce_monotonic(df, "time", policy="drop")
    assert out["time"].tolist() == [100, 110, 120, 130, 140]
    assert report["rows_dropped"] == 3


def test_repeated_timestamps_alone_are_left_to_the_collision_policy():
    # [1, 2, 2, 3] is monotonic *increasing*; duplicates are a primary-key
    # concern, not a disorder one, and resolve_collisions runs first.
    df = pd.DataFrame({"time": [1, 2, 2, 3]})
    out, report = timeline.enforce_monotonic(df, "time", policy="drop")
    assert out["time"].tolist() == [1, 2, 2, 3]
    assert report["rows_dropped"] == 0
    assert report["was_monotonic"]


def test_drop_enforces_strict_increase_once_it_engages():
    # A genuine backward step engages the scan, which then also collapses the
    # repeated timestamp it walks over.
    df = pd.DataFrame({"time": [1, 5, 3, 3, 6]})
    out, report = timeline.enforce_monotonic(df, "time", policy="drop")
    assert out["time"].tolist() == [1, 5, 6]
    assert report["rows_dropped"] == 2
    assert out["time"].is_monotonic_increasing


def test_sort_policy_reorders_instead_of_dropping():
    df = pd.DataFrame({"time": [1, 3, 2, 4]})
    out, report = timeline.enforce_monotonic(df, "time", policy="sort")
    assert out["time"].tolist() == [1, 2, 3, 4]
    assert report["rows_dropped"] == 0


def test_none_policy_reports_without_changing_the_frame():
    df = pd.DataFrame({"time": [1, 3, 2, 4]})
    out, report = timeline.enforce_monotonic(df, "time", policy="none")
    assert out["time"].tolist() == [1, 3, 2, 4]
    assert report["backward_steps"] == 1
    assert not report["was_monotonic"]


def test_already_monotonic_frame_is_untouched():
    df = pd.DataFrame({"time": [1, 2, 3]})
    out, report = timeline.enforce_monotonic(df, "time", policy="drop")
    assert out["time"].tolist() == [1, 2, 3]
    assert report["rows_dropped"] == 0
    assert report["was_monotonic"]


def test_unknown_disorder_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown disorder policy"):
        timeline.enforce_monotonic(pd.DataFrame({"time": [1, 2]}), "time", policy="nope")


# --------------------------------------------------------------------------- #
# Cadence
# --------------------------------------------------------------------------- #

def test_cadence_prefers_the_median_over_a_minority_mode():
    # Collision residue: eight 1s steps against a genuine 20s cadence.
    steps = [1] * 8 + [20] * 12
    times = np.cumsum([0] + steps)
    df = pd.DataFrame({"time": times})
    cad = timeline.estimate_cadence(df, "time")
    assert cad["expected_dt_seconds"] == 20.0
    assert cad["modal_dt_seconds"] == 20.0 or cad["median_dt_seconds"] == 20.0
    assert cad["estimator"] == "median"


def test_cadence_flags_estimator_disagreement():
    # 1s wins the mode on a 43% plurality, but 39 of 69 steps are ~20s, so the
    # median lands on the real cadence. This is the golang/python shape.
    steps = [1] * 30 + [20] * 20 + [21] * 19
    df = pd.DataFrame({"time": np.cumsum([0] + steps)})
    cad = timeline.estimate_cadence(df, "time")
    assert cad["modal_dt_seconds"] == 1.0
    assert cad["median_dt_seconds"] == 20.0
    assert cad["expected_dt_seconds"] == 20.0
    assert cad["estimators_disagree"]
    assert cad["disagreement_ratio"] == pytest.approx(20.0)


def test_cadence_never_measures_across_a_run_boundary():
    # Two runs of 10s cadence; the boundary jumps back 1000s.
    df = pd.DataFrame({"time": list(range(0, 100, 10)) + list(range(50, 150, 10))})
    runs = timeline.detect_runs(df, "time", min_overlap_rows=1)
    cad = timeline.estimate_cadence(df, "time", run_id=runs["run_id"])
    assert cad["expected_dt_seconds"] == 10.0


# --------------------------------------------------------------------------- #
# Per-campaign regularization
# --------------------------------------------------------------------------- #

def _two_campaigns() -> pd.DataFrame:
    """10s cadence, then a 30-day pause, then 60s cadence."""
    a = np.arange(0, 1000, 10)
    b = np.arange(0, 6000, 60) + a[-1] + 30 * 86400
    t = np.concatenate([a, b])
    return pd.DataFrame({"time": t, "v": np.arange(len(t), dtype=float)})


def test_each_campaign_gets_its_own_grid_and_cadence():
    from dataops import _preprocess_impl as impl
    df = _two_campaigns()
    _, _, _, base_dt, diag = impl.regularize_segments(
        df, "time", segment_gap_seconds=86400.0, min_segment_rows=8
    )
    assert diag["strategy"] == "per_segment"
    assert diag["num_segments"] == 2
    assert diag["segment_base_dts"] == [10.0, 60.0]
    assert diag["regularized"] and diag["segments_regularized"] == 2
    assert base_dt == pytest.approx(35.0)   # median of the per-campaign cadences


def test_segmenting_avoids_the_sparsity_guard_a_single_grid_would_trip():
    from dataops import _preprocess_impl as impl
    df = _two_campaigns()
    # One grid stretched across the 30-day pause is overwhelmingly empty.
    _, _, _, _, single = impl.regularize(df, "time", sparse_skip_pct=80.0)
    assert not single["regularized"]
    assert single["expected_gap_pct"] > 80
    # Per campaign, both fit comfortably inside the guard.
    _, _, _, _, seg = impl.regularize_segments(
        df, "time", segment_gap_seconds=86400.0, min_segment_rows=8
    )
    assert seg["regularized"]
    assert all(s["expected_gap_pct"] < 80 for s in seg["segments"])


def test_segment_output_is_uniform_within_each_campaign():
    from dataops import _preprocess_impl as impl
    out, row_mask, col_mask, _, diag = impl.regularize_segments(
        _two_campaigns(), "time", segment_gap_seconds=86400.0, min_segment_rows=8
    )
    steps = np.unique(np.diff(out["time"].to_numpy()))
    # two campaign cadences plus the single jump between them
    assert sorted(steps)[:2] == [10.0, 60.0]
    assert len(steps) == 3
    assert len(row_mask) == len(out)
    assert all(len(m) == len(out) for m in col_mask.values())


def test_short_campaigns_are_dropped_and_reported():
    from dataops import _preprocess_impl as impl
    a = np.arange(0, 1000, 10)
    stub = np.array([a[-1] + 30 * 86400, a[-1] + 30 * 86400 + 10])
    df = pd.DataFrame({"time": np.concatenate([a, stub]),
                       "v": np.arange(len(a) + 2, dtype=float)})
    _, _, _, _, diag = impl.regularize_segments(
        df, "time", segment_gap_seconds=86400.0, min_segment_rows=8
    )
    assert diag["num_segments"] == 1
    assert diag["rows_dropped_short_segments"] == 2


def test_single_campaign_matches_the_unsegmented_grid():
    from dataops import _preprocess_impl as impl
    df = pd.DataFrame({"time": np.arange(0, 1000, 10),
                       "v": np.arange(100, dtype=float)})
    seg = impl.regularize_segments(df, "time", segment_gap_seconds=86400.0)
    one = impl.regularize(df, "time")
    pd.testing.assert_frame_equal(seg[0], one[0])
    assert seg[3] == one[3]


def test_mixed_regimes_are_rejected_so_train_and_test_match():
    """One gridded + one too-sparse campaign must not ship as one bundle.

    The chronological split would put the gridded campaign in train and the
    irregular one in test, training on one sampling regime and scoring on
    another.
    """
    from dataops import _preprocess_impl as impl
    dense = np.arange(0, 2000, 10)          # regularizes cleanly
    # Bursty second campaign: 20 points 2s apart, then 10 points ~50000s apart.
    # The median step is 2s but the span is ~500ks, so a 2s grid is >99% empty —
    # yet no single gap exceeds the 86400s campaign-split threshold.
    burst = np.arange(0, 40, 2)
    tail = burst[-1] + np.arange(1, 11) * 50_000
    sparse = np.concatenate([burst, tail]) + dense[-1] + 30 * 86400
    t = np.concatenate([dense, sparse])
    df = pd.DataFrame({"time": t, "v": np.arange(len(t), dtype=float)})

    _, _, _, _, strict = impl.regularize_segments(
        df, "time", segment_gap_seconds=86400.0, min_segment_rows=8,
        require_all_segments=True,
    )
    assert strict["strategy"] == "per_segment_rejected"
    assert not strict["regularized"]
    assert strict["segments_regularized"] < strict["num_segments"]
    assert "two sampling regimes" in strict["skip_reason"]

    # Opting out keeps the mixed bundle, and says so.
    _, _, _, _, mixed = impl.regularize_segments(
        df, "time", segment_gap_seconds=86400.0, min_segment_rows=8,
        require_all_segments=False,
    )
    assert mixed["strategy"] == "per_segment"
    assert mixed["segments_regularized"] < mixed["num_segments"]


def test_all_segments_passing_still_regularizes_under_the_strict_rule():
    from dataops import _preprocess_impl as impl
    _, _, _, _, diag = impl.regularize_segments(
        _two_campaigns(), "time", segment_gap_seconds=86400.0,
        min_segment_rows=8, require_all_segments=True,
    )
    assert diag["strategy"] == "per_segment"
    assert diag["regularized"]


# --------------------------------------------------------------------------- #
# Grid occupancy: projected vs achieved emptiness, and dropped source rows
# --------------------------------------------------------------------------- #

def test_regularize_measures_achieved_emptiness_and_dropped_rows():
    """A uniform grid takes one row per slot, so rows spaced closer than
    ``base_dt`` lose theirs. ``expected_gap_pct`` is projected before the grid
    exists and cannot see that; the measured fields must."""
    from dataops import _preprocess_impl as impl

    # Cadence is 100s, but every third sample arrives 1s after its predecessor,
    # so those stragglers have to fight for a slot they cannot both hold.
    base = np.arange(0, 100 * 60, 100, dtype=float)
    crowded = np.sort(np.concatenate([base, base[::3] + 1.0]))
    df = pd.DataFrame({"time": crowded, "v": np.arange(len(crowded), dtype=float)})

    _, row_mask, _, base_dt, diag = impl.regularize(
        df, "time", skip_if_sparse=False, sparse_skip_pct=100.0
    )

    placed = int(np.asarray(row_mask, dtype=bool).sum())
    assert diag["source_rows_placed"] == placed
    assert diag["source_rows_dropped"] == diag["source_rows"] - placed
    assert diag["source_rows_dropped"] > 0, "crowded timestamps must lose slots"
    # The projection assumes one slot per row, so it understates the emptiness.
    assert diag["achieved_gap_pct"] > diag["expected_gap_pct"]
    assert diag["guard_basis"] == "expected_gap_pct"


def test_achieved_emptiness_flags_a_grid_the_guard_let_through():
    """The guard decides on the projection; when the built grid misses the
    budget anyway, that has to be visible rather than silently accepted."""
    from dataops import _preprocess_impl as impl

    # Blocks of 60 occupied slots then 40 empty keep the median step at 100s
    # while leaving the grid 37.5% empty; a straggler beside every tenth point
    # adds source rows that cannot win a slot, so the projection undercounts.
    slots = [blk * 100 + i for blk in range(10) for i in range(60)]
    on_grid = np.array(slots, dtype=float) * 100.0
    stragglers = on_grid[::10] + 25.0
    crowded = np.sort(np.concatenate([on_grid, stragglers]))
    df = pd.DataFrame({"time": crowded, "v": np.arange(len(crowded), dtype=float)})

    _, _, _, _, diag = impl.regularize(df, "time", sparse_skip_pct=35.0)

    assert diag["regularized"], "the projection must pass the guard for this case"
    assert diag["expected_gap_pct"] <= 35.0 < diag["achieved_gap_pct"]
    assert diag["achieved_exceeds_guard"] is True
    assert "guard_note" in diag["decision"]


def test_uniform_timeline_drops_nothing():
    """The measurement must not manufacture drops on a clean grid."""
    from dataops import _preprocess_impl as impl

    df = pd.DataFrame({"time": np.arange(0, 1000, 10, dtype=float),
                       "v": np.arange(100, dtype=float)})
    _, _, _, _, diag = impl.regularize(df, "time")

    assert diag["source_rows_dropped"] == 0
    assert diag["achieved_gap_pct"] == 0.0
    assert diag["achieved_exceeds_guard"] is False
