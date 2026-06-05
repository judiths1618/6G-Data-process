"""Unit tests for pipeline_modules.split — focus on the 1:1 reproducibility claim."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline_modules import split
from pipeline_modules import _preprocess_impl as impl

# .../dockers/tools/pipeline_modules/tests/test_split.py -> parents[4] == repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
WORK_EUR = REPO_ROOT / "experiments" / "EUR"
SUBSETS = ["amf", "golang", "python", "rabbitmq"]


@pytest.fixture
def reg_df() -> pd.DataFrame:
    n = 500
    t = np.arange(n)
    return pd.DataFrame({
        "time": t,
        "a": t.astype(float),
        "b": (t * 2).astype(float),
    })


META = {"time_col": "time", "target_cols": ["a", "b"]}


# --------------------------------------------------------------------------- #
# Split mechanics
# --------------------------------------------------------------------------- #

def test_row_counts_match_time_cut(reg_df):
    parts = split.train_test(reg_df, META, split_ratio=0.8)
    cut = int(len(reg_df) * 0.8)
    assert parts.meta["train_rows"] == cut
    assert parts.meta["test_rows"] == len(reg_df) - cut
    assert len(parts.train) == cut
    assert len(parts.test_gt) == len(reg_df) - cut


def test_test_gt_is_unmodified_tail(reg_df):
    parts = split.train_test(reg_df, META, split_ratio=0.8)
    cut = int(len(reg_df) * 0.8)
    expected = reg_df.iloc[cut:].reset_index(drop=True)
    pd.testing.assert_frame_equal(parts.test_gt.reset_index(drop=True), expected)


def test_eval_mask_shape_and_dtype(reg_df):
    parts = split.train_test(reg_df, META)
    assert parts.eval_mask.dtype == np.bool_
    assert parts.eval_mask.shape == (parts.meta["test_rows"],)
    assert 0 < parts.eval_mask.sum() <= parts.meta["test_rows"]


def test_masking_semantics_hide_only_observed(reg_df):
    parts = split.train_test(reg_df, META)
    mask = parts.eval_mask
    for c in META["target_cols"]:
        gt = parts.test_gt[c].to_numpy()
        ti = parts.test_input[c].to_numpy()
        observed = ~np.isnan(gt)
        hidden = mask & observed
        # hidden positions become NaN ...
        assert np.isnan(ti[hidden]).all()
        # ... and every other position is preserved exactly
        keep = ~hidden
        np.testing.assert_array_equal(ti[keep], gt[keep])
    assert parts.meta["hidden_values"] > 0


def test_non_target_columns_untouched(reg_df):
    parts = split.train_test(reg_df, META)
    cut = int(len(reg_df) * 0.8)
    expected_time = reg_df["time"].iloc[cut:].to_numpy()
    np.testing.assert_array_equal(parts.test_input["time"].to_numpy(), expected_time)


def test_target_cols_default_from_meta(reg_df):
    parts = split.train_test(reg_df, {"time_col": "time", "target_cols": ["a"]})
    # only 'a' should ever be masked; 'b' stays fully intact
    np.testing.assert_array_equal(
        parts.test_input["b"].to_numpy(), parts.test_gt["b"].to_numpy()
    )


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

def test_same_seed_identical_mask(reg_df):
    m1 = split.train_test(reg_df, META, seed=0).eval_mask
    m2 = split.train_test(reg_df, META, seed=0).eval_mask
    np.testing.assert_array_equal(m1, m2)


def test_different_seed_changes_mask(reg_df):
    m1 = split.train_test(reg_df, META, seed=0).eval_mask
    m2 = split.train_test(reg_df, META, seed=12345).eval_mask
    assert not np.array_equal(m1, m2)


# --------------------------------------------------------------------------- #
# 1:1 with the canonical primitives
# --------------------------------------------------------------------------- #

def test_matches_canonical_primitives(reg_df):
    """split.train_test must equal impl.train_test_split_by_time + make_eval_holdout_mask."""
    ratio, frac, block, seed = 0.8, 0.15, 5, 0
    train_ref, test_ref = impl.train_test_split_by_time(reg_df, ratio)
    mask_ref = impl.make_eval_holdout_mask(len(test_ref), frac, block, seed)

    parts = split.train_test(
        reg_df, META, split_ratio=ratio, holdout_frac=frac,
        holdout_block_size=block, seed=seed,
    )
    np.testing.assert_array_equal(parts.eval_mask, mask_ref.astype(np.bool_))
    pd.testing.assert_frame_equal(parts.train, train_ref)
    pd.testing.assert_frame_equal(parts.test_gt, test_ref)


# --------------------------------------------------------------------------- #
# 1:1 against the real prepared_amf artifacts (skipped if data absent)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("subset", SUBSETS)
def test_reproduces_prepared_holdout(subset):
    prepared = WORK_EUR / f"prepared_{subset}"
    if not (prepared / "eval_holdout_mask.npy").exists():
        pytest.skip(f"prepared_{subset} artifacts not present")

    meta = json.loads((prepared / "meta.json").read_text())
    train = pd.read_csv(prepared / "train.csv")
    test_gt = pd.read_csv(prepared / "test_gt.csv")
    saved_mask = np.load(prepared / "eval_holdout_mask.npy")
    saved_test_input = pd.read_csv(prepared / "test_input.csv")

    # Reconstruct the regularized frame the preprocessor split.
    df_reg = pd.concat([train, test_gt], ignore_index=True)

    parts = split.train_test(
        df_reg, meta,
        split_ratio=meta["split_ratio"], holdout_frac=meta["holdout_frac"],
        holdout_block_size=5, seed=0,
    )

    # Mask reproduced bit-for-bit.
    np.testing.assert_array_equal(parts.eval_mask, saved_mask)
    # NaN-position layout of the masked test input reproduced bit-for-bit.
    tcols = meta["target_cols"]
    got_nan = parts.test_input[tcols].isna().to_numpy()
    exp_nan = saved_test_input[tcols].isna().to_numpy()
    np.testing.assert_array_equal(got_nan, exp_nan)
