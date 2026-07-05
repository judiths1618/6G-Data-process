"""Unit tests for the WaveStitch+ v2 runner glue."""
from __future__ import annotations

import argparse

import pandas as pd

import run_imputation_v2 as riv2


def test_output_name_tags_tuned_results_separately():
    assert riv2._output_name("anchored", "test") == "wavestitchplus_v2_test_imputed.csv"
    assert riv2._output_name("tuned", "test") == "wavestitchplus_v2_tuned_test_imputed.csv"
    assert riv2._output_name("tuned", "train") == "wavestitchplus_v2_tuned_train_imputed.csv"


def test_select_tuned_params_can_choose_non_nearest_prior():
    test_input = pd.DataFrame({"time": [0, 1, 2, 3], "a": [0.0, None, None, 30.0]})
    gt = pd.DataFrame({"time": [0, 1, 2, 3], "a": [0.0, 10.0, 20.0, 30.0]})
    diffusion = pd.DataFrame({"time": [0, 1, 2, 3], "a": [0.0, 100.0, 100.0, 30.0]})
    args = argparse.Namespace(
        tune_priors=["nearest", "linear"],
        tune_taus=[1.0],
        tune_hard_priors=[0, 8],
    )

    prior, tau, hard_prior, score = riv2._select_tuned_params(
        None, test_input, gt, diffusion, ["a"], args, None,
    )

    assert (prior, tau) == ("linear", 1.0)
    assert hard_prior in {0, 8}
    assert score["MAE"] == 0.0
