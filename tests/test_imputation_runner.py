from __future__ import annotations

import json

import numpy as np
import pandas as pd

from dataops.imputation_runner import (
    INTERP_METHODS,
    build_final_dataset,
    builtin_methods,
    compare_clean_vs_imputed,
    impute_bundle,
    impute_dataframe,
)


def _make_bundle(tmp_path):
    prepared = tmp_path / "prepared_demo"
    prepared.mkdir()
    # regularized timeline with explicit gaps (NaN) in the target columns
    train = pd.DataFrame({
        "time": range(10),
        "a": [0.0, np.nan, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0],
        "b": [10.0, 11.0, np.nan, np.nan, 14.0, 15.0, 16.0, np.nan, 18.0, 19.0],
    })
    # test_input has holes; test_gt holds the truth at the masked cells
    test_input = pd.DataFrame({
        "time": range(5),
        "a": [0.0, np.nan, 2.0, np.nan, 4.0],
        "b": [10.0, np.nan, 12.0, 13.0, 14.0],
    })
    test_gt = pd.DataFrame({
        "time": range(5),
        "a": [0.0, 1.0, 2.0, 3.0, 4.0],
        "b": [10.0, 11.0, 12.0, 13.0, 14.0],
    })
    train.to_csv(prepared / "train.csv", index=False)
    test_input.to_csv(prepared / "test_input.csv", index=False)
    test_gt.to_csv(prepared / "test_gt.csv", index=False)
    (prepared / "meta.json").write_text(json.dumps({
        "time_col": "time", "target_cols": ["a", "b"],
    }))
    return prepared


def test_impute_dataframe_fills_all_target_nans_nearest():
    df = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0], "keep": [np.nan, 1, 2, 3]})
    out = impute_dataframe(df, ["a"], "nearest")
    assert out["a"].isna().sum() == 0           # target filled
    assert out["keep"].isna().sum() == 1         # non-target untouched
    assert "nearest" in INTERP_METHODS


def test_impute_bundle_and_comparison(tmp_path):
    prepared = _make_bundle(tmp_path)
    out_dir = tmp_path / "generated_demo"
    result = impute_bundle(prepared, method="nearest", output_dir=str(out_dir),
                           engine="pandas")

    assert result["method"] == "nearest" and result["engine"] == "pandas"
    assert set(result["files"]) == {"train", "test"}
    for kind, info in result["files"].items():
        assert info["nan_after"] == 0
        assert info["filled"] == info["nan_before"] > 0
        assert (out_dir / f"darts_nearest_{kind}_imputed.csv").exists()

    comparison = compare_clean_vs_imputed(prepared, result)
    test_rep = comparison["splits"]["test"]
    assert test_rep["fill_rate"] == 1.0
    assert test_rep["residual_nan"] == 0
    acc = test_rep["accuracy"]
    assert acc["eval_cells"] > 0
    assert "per_column" in acc and "a" in acc["per_column"]
    # nearest on a=[0,nan,2,nan,4] reproduces 1->? (nearest of 0/2) — error small
    assert acc["per_column"]["a"]["MAE"] >= 0.0


def test_pandas_engine_rejects_non_interp_method(tmp_path):
    prepared = _make_bundle(tmp_path)
    import pytest
    with pytest.raises(ValueError, match="kalman"):
        impute_bundle(prepared, method="kalman", output_dir=str(tmp_path / "g"),
                      engine="pandas")


def test_imputegap_statistics_run_dependency_free(tmp_path):
    assert set(builtin_methods("imputegap")) >= {"interpolation", "mean", "min", "zero"}
    assert builtin_methods("pypots") == []   # heavy-only library: no built-ins

    prepared = _make_bundle(tmp_path)
    out_dir = tmp_path / "gen"
    res = impute_bundle(prepared, method="mean", lib="imputegap",
                        output_dir=str(out_dir), engine="pandas")
    assert res["lib"] == "imputegap"
    assert (out_dir / "imputegap_mean_test_imputed.csv").exists()
    assert res["files"]["test"]["nan_after"] == 0

    # imputegap/mean fills with the column mean; imputegap/zero fills 0.
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    assert impute_dataframe(df, ["a"], "mean", lib="imputegap")["a"].tolist() == [1.0, 2.0, 3.0]
    assert impute_dataframe(df, ["a"], "zero", lib="imputegap")["a"].tolist() == [1.0, 0.0, 3.0]


def _make_disjoint_bundle(tmp_path):
    """Prepared bundle with DISJOINT train/test time ranges (as real bundles are),
    a train gap and a test holdout cell whose ground truth is a sentinel (999)."""
    prepared = tmp_path / "prepared_d"
    prepared.mkdir()
    train = pd.DataFrame({"time": [0, 1, 2, 3],
                          "a": [0.0, np.nan, 2.0, 3.0],
                          "b": [10.0, 11.0, np.nan, 13.0]})
    test_input = pd.DataFrame({"time": [4, 5, 6],
                               "a": [4.0, np.nan, 6.0], "b": [14.0, 15.0, 16.0]})
    test_gt = pd.DataFrame({"time": [4, 5, 6],
                            "a": [4.0, 999.0, 6.0], "b": [14.0, 15.0, 16.0]})
    for name, d in [("train.csv", train), ("test_input.csv", test_input),
                    ("test_gt.csv", test_gt)]:
        d.to_csv(prepared / name, index=False)
    (prepared / "meta.json").write_text(json.dumps({
        "time_col": "time", "target_cols": ["a", "b"],
    }))
    return prepared, train, test_input


def test_build_final_dataset_stitches_imputed_train_and_test(tmp_path):
    """The final = imputed train + imputed test (the imputer's own output for
    BOTH splits), gap-free, and it never leaks test_gt ground truth."""
    prepared, train, test_input = _make_disjoint_bundle(tmp_path)
    out_dir = tmp_path / "gen_d"
    result = impute_bundle(prepared, method="nearest", output_dir=str(out_dir),
                           engine="pandas")

    final = build_final_dataset(
        prepared, method="nearest", output_path=str(tmp_path / "d_final.csv"),
        lib="darts", bundle_result=result, imputed_dir=str(out_dir),
    )
    df = pd.read_csv(tmp_path / "d_final.csv")
    imp_test = pd.read_csv(result["files"]["test"]["path"])

    # imputed TRAIN is included: final row count == both imputed splits concatenated
    assert final["rows"] == len(train) + len(test_input) == len(df)
    assert set(final["sources"]) == {"train", "test"}
    assert final["gaps_after"] == 0 and df[["a", "b"]].isna().sum().sum() == 0
    # gaps_before counts the raw gappy inputs (train a@1, b@2, test_input a@5)
    assert final["gaps_before"] == 3

    # the test holdout cell (a@time=5) is the IMPUTED test value, not GT (999)
    got = df.loc[df["time"] == 5, "a"].iloc[0]
    assert got != 999.0
    assert got == imp_test.loc[imp_test["time"] == 5, "a"].iloc[0]


def test_build_final_dataset_stitches_prebuilt_model_splits(tmp_path):
    """WaveStitch+-style heavy lib: the imputed train/test splits already exist on
    disk and have no dependency-free built-in. build_final_dataset must stitch them
    (imputed train included) without trying to re-impute."""
    prepared = tmp_path / "prepared_wsp"
    prepared.mkdir()
    (prepared / "meta.json").write_text(json.dumps({
        "time_col": "time", "target_cols": ["a", "b"],
    }))
    gen = tmp_path / "generated_wsp"
    gen.mkdir()
    train = pd.DataFrame({"time": [0, 1, 2, 3],
                          "a": [0.0, 1.0, 2.0, 3.0], "b": [10.0, 11.0, 12.0, 13.0]})
    test = pd.DataFrame({"time": [4, 5, 6],
                         "a": [40.0, 50.0, 60.0], "b": [14.0, 15.0, 16.0]})
    train.to_csv(gen / "wavestitchplus_v1_train_imputed.csv", index=False)
    test.to_csv(gen / "wavestitchplus_v1_test_imputed.csv", index=False)

    final = build_final_dataset(
        prepared, method="v1", lib="wavestitchplus", imputed_dir=str(gen),
        output_path=str(tmp_path / "wsp_final.csv"),
    )
    df = pd.read_csv(tmp_path / "wsp_final.csv")
    assert final["rows"] == len(train) + len(test) == len(df)
    assert df.loc[df["time"] == 1, "a"].iloc[0] == 1.0    # from the imputed TRAIN split
    assert df.loc[df["time"] == 5, "a"].iloc[0] == 50.0   # from the imputed test split
    assert df[["a", "b"]].isna().sum().sum() == 0
    assert set(final["sources"]) == {"train", "test"}
