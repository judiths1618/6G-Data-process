"""Tests for wsp_final.build_wsp_final — the vendored WaveStitch+ final stitcher.

Mirrors dataops.imputation_runner.build_final_dataset: the final is the imputed
train split + the imputed test split, keyed on time, gap-free.
"""
from __future__ import annotations

import json

import pandas as pd

from wsp_final import build_wsp_final


def _bundle(tmp_path, variant):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "meta.json").write_text(json.dumps({
        "time_col": "time", "target_cols": ["a", "b"],
    }))
    gen = tmp_path / "generated"
    gen.mkdir()
    train = pd.DataFrame({"time": [0, 1, 2, 3],
                          "a": [0.0, 1.0, 2.0, 3.0], "b": [10.0, 11.0, 12.0, 13.0]})
    test = pd.DataFrame({"time": [4, 5, 6],
                         "a": [40.0, 50.0, 60.0], "b": [14.0, 15.0, 16.0]})
    train.to_csv(gen / f"wavestitchplus_{variant}_train_imputed.csv", index=False)
    test.to_csv(gen / f"wavestitchplus_{variant}_test_imputed.csv", index=False)
    return prepared, gen, train, test


def test_build_wsp_final_stitches_imputed_train_and_test(tmp_path):
    prepared, gen, train, test = _bundle(tmp_path, "v2")
    out = build_wsp_final(prepared, gen, "v2")

    assert out == gen / "wavestitchplus_v2_final.csv"
    df = pd.read_csv(out)
    # imputed TRAIN included: final == both imputed splits concatenated
    assert len(df) == len(train) + len(test)
    assert df.loc[df["time"] == 1, "a"].iloc[0] == 1.0     # from imputed train
    assert df.loc[df["time"] == 5, "a"].iloc[0] == 50.0    # from imputed test
    assert list(df.columns) == ["time", "split", "a", "b"]  # time + split + targets
    assert df[["a", "b"]].isna().sum().sum() == 0          # gap-free
    assert df["time"].is_monotonic_increasing
    # split column marks the train/test boundary per row
    assert df.loc[df["time"] <= 3, "split"].eq("train").all()
    assert df.loc[df["time"] >= 4, "split"].eq("test").all()


def test_build_wsp_final_skips_when_no_splits(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "meta.json").write_text(json.dumps({
        "time_col": "time", "target_cols": ["a"],
    }))
    gen = tmp_path / "generated"
    gen.mkdir()
    assert build_wsp_final(prepared, gen, "harpoon") is None


def test_build_wsp_final_train_only(tmp_path):
    """A run that produced only the train split still yields a (train-only) final."""
    prepared, gen, train, test = _bundle(tmp_path, "v1")
    (gen / "wavestitchplus_v1_test_imputed.csv").unlink()
    df = pd.read_csv(build_wsp_final(prepared, gen, "v1"))
    assert len(df) == len(train)
    assert df["a"].tolist() == [0.0, 1.0, 2.0, 3.0]
