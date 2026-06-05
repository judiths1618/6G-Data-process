"""
Unit tests for the pure helpers in WaveStitch+'s baseline-compatible runner
(``run_imputation.py``).

Only the deterministic, dependency-light parts are covered here — output-name
construction, the observed-cell-preserving merge, and the train-output publisher.
The actual train/synthesis steps shell out to GPU scripts and are out of scope
for a unit test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import run_imputation as ri


# ---------------------------------------------------------------------------
# _output_name
# ---------------------------------------------------------------------------

def test_output_name_full_and_v1_use_v1_tag():
    # Default ``full`` is now an alias for ``v1`` so both produce a v1-tagged file
    # (parallel to the v2 runner's ``wavestitchplus_v2_<split>_imputed.csv``).
    assert ri._output_name("full", "train") == "wavestitchplus_v1_train_imputed.csv"
    assert ri._output_name("full", "test") == "wavestitchplus_v1_test_imputed.csv"
    assert ri._output_name("v1", "test") == "wavestitchplus_v1_test_imputed.csv"


def test_output_name_explicit_methods_keep_tag():
    assert ri._output_name("em", "test") == "wavestitchplus_em_test_imputed.csv"
    assert ri._output_name("standard", "train") == "wavestitchplus_standard_train_imputed.csv"


# ---------------------------------------------------------------------------
# _fill_missing_only
# ---------------------------------------------------------------------------

def test_fill_missing_only_preserves_observed_and_fills_gaps():
    inp = pd.DataFrame({
        "time": [0, 1, 2, 3],
        "v": [10.0, np.nan, 30.0, np.nan],
    })
    # Imputed frame disagrees everywhere (whole-series reconstruction).
    imp = pd.DataFrame({
        "time": [0, 1, 2, 3],
        "v": [-1.0, 20.0, -3.0, 40.0],
    })
    out = ri._fill_missing_only(inp, imp)

    # Observed cells preserved exactly; only the NaNs are filled from `imp`.
    assert out["v"].tolist() == [10.0, 20.0, 30.0, 40.0]
    # Schema and row count match the input.
    assert list(out.columns) == ["time", "v"]
    assert len(out) == len(inp)


def test_fill_missing_only_restores_missing_column_schema():
    # train_imputed_denorm.csv may omit the timestamp; the input's columns win.
    inp = pd.DataFrame({"time": [0, 1, 2], "v": [np.nan, 5.0, np.nan]})
    imp = pd.DataFrame({"v": [1.0, 2.0, 3.0]})  # no "time" column
    out = ri._fill_missing_only(inp, imp)

    assert "time" in out.columns
    assert out["time"].tolist() == [0, 1, 2]
    assert out["v"].tolist() == [1.0, 5.0, 3.0]


def test_fill_missing_only_handles_shorter_imputed_without_crashing():
    # Imputed shorter than input → trailing missing cells stay NaN, no IndexError.
    inp = pd.DataFrame({"v": [np.nan, 2.0, np.nan, np.nan]})
    imp = pd.DataFrame({"v": [99.0, 88.0]})  # only 2 rows
    out = ri._fill_missing_only(inp, imp)

    assert len(out) == 4
    assert out["v"].iloc[0] == 99.0      # filled from imp
    assert out["v"].iloc[1] == 2.0       # observed preserved
    assert np.isnan(out["v"].iloc[2])    # beyond imp length → still missing
    assert np.isnan(out["v"].iloc[3])


def test_fill_missing_only_zero_missing_is_noop():
    inp = pd.DataFrame({"time": [0, 1, 2], "v": [1.0, 2.0, 3.0]})
    imp = pd.DataFrame({"time": [0, 1, 2], "v": [-1.0, -2.0, -3.0]})
    out = ri._fill_missing_only(inp, imp)

    # Nothing was missing → output equals the input.
    pd.testing.assert_frame_equal(out, inp.reset_index(drop=True), check_dtype=False)


def test_fill_missing_only_ignores_extra_imputed_columns():
    inp = pd.DataFrame({"v": [np.nan, 2.0]})
    imp = pd.DataFrame({"v": [1.0, 9.0], "extra": [7.0, 7.0]})
    out = ri._fill_missing_only(inp, imp)

    assert list(out.columns) == ["v"]      # extra column not introduced
    assert out["v"].tolist() == [1.0, 2.0]


# ---------------------------------------------------------------------------
# _publish_train_output (file-based; uses tmp_path)
# ---------------------------------------------------------------------------

def test_publish_train_output_returns_none_when_artifact_absent(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    out_dir = tmp_path / "generated"
    # No train_imputed_denorm.csv present.
    assert ri._publish_train_output(prepared, out_dir, "full") is None


def test_publish_train_output_preserves_observed(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    out_dir = tmp_path / "generated"

    pd.DataFrame({
        "time": [0, 1, 2],
        "v": [10.0, np.nan, 30.0],
    }).to_csv(prepared / "train.csv", index=False)
    # Denorm artifact disagrees on observed cells and omits the timestamp.
    pd.DataFrame({"v": [-1.0, 20.0, -3.0]}).to_csv(
        prepared / "train_imputed_denorm.csv", index=False)

    out_csv = ri._publish_train_output(prepared, out_dir, "full")

    assert out_csv is not None
    assert out_csv.name == "wavestitchplus_v1_train_imputed.csv"
    written = pd.read_csv(out_csv)
    # Timestamp restored from train.csv; observed v kept; gap filled from denorm.
    assert written["time"].tolist() == [0, 1, 2]
    assert written["v"].tolist() == [10.0, 20.0, 30.0]
