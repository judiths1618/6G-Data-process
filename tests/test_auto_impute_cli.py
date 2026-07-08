from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.auto_impute import ALL_BUILTIN_SPECS, main


def _make_report_and_bundle(tmp_path):
    prepared = tmp_path / "demo_regularized"
    prepared.mkdir()
    train = pd.DataFrame({
        "time": range(8),
        "a": [0.0, np.nan, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0],
        "b": [10.0, 11.0, np.nan, 13.0, 14.0, np.nan, 16.0, 17.0],
    })
    test_input = pd.DataFrame({
        "time": range(8, 14),
        "a": [8.0, np.nan, 10.0, np.nan, 12.0, 13.0],
        "b": [18.0, np.nan, 20.0, 21.0, np.nan, 23.0],
    })
    test_gt = pd.DataFrame({
        "time": range(8, 14),
        "a": [8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
        "b": [18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
    })
    train.to_csv(prepared / "train.csv", index=False)
    test_input.to_csv(prepared / "test_input.csv", index=False)
    test_gt.to_csv(prepared / "test_gt.csv", index=False)
    (prepared / "meta.json").write_text(json.dumps({
        "time_col": "time",
        "target_cols": ["a", "b"],
    }))

    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    output_dir = tmp_path / "data" / "processed"
    output_dir.mkdir(parents=True)
    report = {
        "output": str(output_dir / "demo_remediated.csv"),
        "report_path": str(report_dir / "demo_report.json"),
        "handoff": {
            "needs_ts_imputation": True,
            "prepared_dir": str(prepared),
            "selection": {"app": "Darts", "method": "nearest"},
        },
    }
    report_path = report_dir / "demo_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, output_dir


def test_auto_impute_all_writes_method_outputs_and_canonical_final(tmp_path):
    report_path, output_dir = _make_report_and_bundle(tmp_path)

    assert main(["--report", str(report_path), "--method", "all"]) == 0

    generated = tmp_path / "demo_generated"
    for lib, method in ALL_BUILTIN_SPECS:
        assert (generated / f"{lib}_{method}_train_imputed.csv").exists()
        assert (generated / f"{lib}_{method}_test_imputed.csv").exists()
        assert (generated / f"{lib}_{method}_final.csv").exists()

    canonical = output_dir / "demo_final.csv"
    assert canonical.exists()
    assert pd.read_csv(canonical)[["a", "b"]].isna().sum().sum() == 0

    compare = json.loads((tmp_path / "reports" / "demo_imputation_compare.json").read_text())
    assert len(compare["runs"]) == len(ALL_BUILTIN_SPECS)
    assert compare["canonical_final_dataset"]["path"] == str(canonical)
