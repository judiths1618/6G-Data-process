#!/usr/bin/env python3
"""
Local baseline-imputation experiment runner.

Mirrors the Airflow ``clean_dirty_data`` task's logic — but bypasses Airflow
for fast local iteration. Uses the *exact same* preprocess args the DAG uses
(and that WaveStitchPlus uses), so any cross-method comparison produced here
is apples-to-apples with what the DAG would produce.

Two input modes:

  ``--input-csv <path>``        run preprocess_csv on a raw CSV then impute
                                with each selected method.

  ``--prepared-dir <path>``     skip preprocess; reuse an existing
                                prepared_<subset>/ directory (the same files
                                the DAG writes to S3).

Outputs land in:

  ``experiments/<dataset>/prepared_<run_id>/``
  ``experiments/<dataset>/generated_<run_id>/``
                                  ├── <method>_test_imputed.csv
                                  └── results.csv

(``run_id`` defaults to a timestamp so re-runs accumulate as separate
"subsets" of the same dataset.) Point the dashboard's *Work root* at the
top-level ``experiments/`` dir to see every method + run side-by-side.

Example::

    conda activate myenv
    python scripts/compare_baselines.py \\
        --input-csv 6GDALI_Datasets/EUR/6907619/amf-performance.csv \\
        --methods darts_linear,darts_cubic,darts_kalman,pypots_saits

    streamlit run dashboard/app.py
    # in the sidebar, set Work root = experiments/
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Stub airflow.models + docker so we can import the helpers from this script
# without an Airflow runtime. Variable.get falls back to os.environ so users
# can still do `N2N_PYPOTS_EPOCHS=10 python scripts/compare_baselines.py ...`.
# ---------------------------------------------------------------------------
if "airflow" not in sys.modules:
    _airflow = types.ModuleType("airflow")
    _models = types.ModuleType("airflow.models")

    class _Variable:
        @staticmethod
        def get(key, default_var=None):
            return os.environ.get(key, default_var)

    _models.Variable = _Variable
    sys.modules["airflow"] = _airflow
    sys.modules["airflow.models"] = _models
if "docker" not in sys.modules:
    sys.modules["docker"] = types.ModuleType("docker")

REPO_ROOT = Path(__file__).resolve().parent.parent
HELPERS_PARENT = REPO_ROOT / "dockers" / "airflow" / "dags"
if str(HELPERS_PARENT) not in sys.path:
    sys.path.insert(0, str(HELPERS_PARENT))

from helpers.preprocess import preprocess_csv  # noqa: E402
from helpers.clean_dirty_data import (  # noqa: E402
    _BASELINE_METHODS,
    _PREPROCESS_ARGS_MATCHING_WAVESTITCHPLUS,
    _ts_imputation_baseline,
)


def _maybe_dataset_name(path: Path) -> str:
    stem = path.stem
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "dataset"


def _resolve_methods(arg: Optional[str]) -> List[str]:
    if not arg:
        # Order is stable so the output table is deterministic.
        return sorted(_BASELINE_METHODS)
    requested = [m.strip() for m in arg.split(",") if m.strip()]
    unknown = [m for m in requested if m not in _BASELINE_METHODS]
    if unknown:
        raise SystemExit(
            f"Unknown method(s): {unknown}. "
            f"Choose from: {sorted(_BASELINE_METHODS)}"
        )
    return requested


def _build_prepared_dir(input_csv: Path, prepared_dir: Path) -> dict:
    """Run preprocess_csv with the WaveStitchPlus-matching args."""
    prepared_dir.mkdir(parents=True, exist_ok=True)
    meta = preprocess_csv(
        input_csv=str(input_csv),
        output_dir=str(prepared_dir),
        time_col=None,
        **_PREPROCESS_ARGS_MATCHING_WAVESTITCHPLUS,
    )
    return meta


def _impute_one_method(
    method: str,
    train_df: pd.DataFrame,
    test_input_df: pd.DataFrame,
    test_gt_df: pd.DataFrame,
    target_cols: List[str],
    cat_encoded_cols: List[str],
    ts_col: str,
    out_dir: Path,
) -> tuple[dict, dict]:
    """Run one imputer; save train_imputed + test_imputed; return (paths, timing)."""
    t0 = time.perf_counter()
    full_in = pd.concat([train_df, test_input_df], ignore_index=True)
    full_out, _ = _ts_imputation_baseline(
        full_in, ts_col=ts_col, target_cols=target_cols, method=method,
    )
    # Mirror the DAG: ffill/bfill categorical-encoded cond columns.
    for c in cat_encoded_cols:
        if c in full_out.columns:
            full_out[c] = full_out[c].ffill().bfill()

    n_train = len(train_df)
    train_imputed = full_out.iloc[:n_train].reset_index(drop=True)
    test_imputed = full_out.iloc[n_train:].reset_index(drop=True)
    # Dashboard regex expects <lib>_<method>_<split>_imputed.csv → method
    # names like 'darts_linear' already follow that shape.
    train_path = out_dir / f"{method}_train_imputed.csv"
    test_path = out_dir / f"{method}_test_imputed.csv"
    train_imputed.to_csv(train_path, index=False)
    test_imputed.to_csv(test_path, index=False)
    return ({"train": train_path, "test": test_path},
            {"elapsed_sec": time.perf_counter() - t0})


def _score(
    test_input: pd.DataFrame,
    test_gt: pd.DataFrame,
    test_imputed: pd.DataFrame,
    target_cols: List[str],
) -> tuple[pd.DataFrame, dict]:
    """Per-target MAE/RMSE/MAPE on cells where input is NaN and GT is not."""
    rows = []
    overall_err = []
    overall_ratios = []
    for c in target_cols:
        if c not in test_imputed.columns:
            continue
        cell = test_input[c].isna().to_numpy() & test_gt[c].notna().to_numpy()
        if cell.sum() == 0:
            continue
        gt = test_gt[c][cell].to_numpy(dtype=float)
        pr = test_imputed[c][cell].to_numpy(dtype=float)
        keep = ~np.isnan(gt) & ~np.isnan(pr)
        if keep.sum() == 0:
            continue
        gt, pr = gt[keep], pr[keep]
        err = pr - gt
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        nonzero = np.abs(gt) > 1e-12
        mape = (float(np.mean(np.abs(err[nonzero] / gt[nonzero])) * 100)
                if nonzero.any() else float("nan"))
        rows.append({
            "target": c, "MAE": mae, "RMSE": rmse,
            "MAPE_%": mape, "n_cells": int(len(err)),
        })
        overall_err.extend(err.tolist())
        if nonzero.any():
            overall_ratios.extend(
                np.abs(err[nonzero] / gt[nonzero]).tolist()
            )
    per_target = pd.DataFrame(rows)
    if overall_err:
        a = np.array(overall_err)
        overall = {
            "MAE": float(np.mean(np.abs(a))),
            "RMSE": float(np.sqrt(np.mean(a ** 2))),
            "MAPE_%": (float(np.mean(np.array(overall_ratios)) * 100)
                       if overall_ratios else float("nan")),
            "n_cells": int(len(a)),
        }
    else:
        overall = {"MAE": float("nan"), "RMSE": float("nan"),
                   "MAPE_%": float("nan"), "n_cells": 0}
    return per_target, overall


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-csv", type=Path,
                     help="Raw CSV to preprocess and impute.")
    src.add_argument("--prepared-dir", type=Path,
                     help="Existing prepared_<subset>/ to reuse "
                          "(skips preprocess).")

    p.add_argument("--methods", type=str, default=None,
                   help="Comma-separated subset of baselines. "
                        "Default: all available.")
    p.add_argument("--output-root", type=Path,
                   default=REPO_ROOT / "experiments",
                   help="Top-level output directory (default: ./experiments).")
    p.add_argument("--dataset-name", type=str, default=None,
                   help="Override dataset name (default: derived from input).")
    p.add_argument("--run-id", type=str, default=None,
                   help="Run identifier (becomes the prepared_<run_id>/ + "
                        "generated_<run_id>/ suffix). Defaults to a timestamp "
                        "so repeated runs accumulate side-by-side under the "
                        "same dataset dir.")
    p.add_argument("--keep-prepared", action="store_true",
                   help="If --prepared-dir was given, copy it into the "
                        "experiment dir. Default is to symlink.")
    args = p.parse_args()

    methods = _resolve_methods(args.methods)
    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.input_csv:
        dataset = args.dataset_name or _maybe_dataset_name(args.input_csv)
        ds_dir = args.output_root / dataset
        prepared_dir = ds_dir / f"prepared_{run_id}"
        generated_dir = ds_dir / f"generated_{run_id}"
        generated_dir.mkdir(parents=True, exist_ok=True)
        print(f"[STEP 1/3] preprocess_csv on {args.input_csv}")
        prep_meta = _build_prepared_dir(args.input_csv, prepared_dir)
    else:
        prep_src = args.prepared_dir.resolve()
        if not (prep_src / "meta.json").exists():
            raise SystemExit(f"{prep_src} is not a prepared_<subset>/ dir "
                             f"(missing meta.json).")
        dataset = args.dataset_name or _maybe_dataset_name(prep_src.parent)
        ds_dir = args.output_root / dataset
        prepared_dir = ds_dir / f"prepared_{run_id}"
        generated_dir = ds_dir / f"generated_{run_id}"
        ds_dir.mkdir(parents=True, exist_ok=True)
        generated_dir.mkdir(parents=True, exist_ok=True)
        if args.keep_prepared:
            print(f"[STEP 1/3] copying {prep_src} -> {prepared_dir}")
            shutil.copytree(prep_src, prepared_dir)
        else:
            print(f"[STEP 1/3] symlinking prepared dir from {prep_src}")
            prepared_dir.symlink_to(prep_src, target_is_directory=True)
        prep_meta = json.loads((prepared_dir / "meta.json").read_text())

    target_cols = prep_meta["target_cols"]
    cat_encoded = prep_meta.get("categorical_encoded_cols") or []
    ts_col = prep_meta.get("time_col", "time")

    train_df = pd.read_csv(prepared_dir / "train.csv")
    test_input_df = pd.read_csv(prepared_dir / "test_input.csv")
    test_gt_df = pd.read_csv(prepared_dir / "test_gt.csv")

    print(f"\n[INFO] dataset={dataset}  run_id={run_id}")
    print(f"[INFO] target_cols ({len(target_cols)}): {target_cols}")
    print(f"[INFO] train={len(train_df)}  test={len(test_input_df)}  "
          f"holdout_cells={int(test_input_df[target_cols].isna().sum().sum())}")
    print(f"[INFO] output: {ds_dir}")

    print(f"\n[STEP 2/3] imputing with {len(methods)} method(s): {methods}")
    timings: Dict[str, float] = {}
    for i, m in enumerate(methods, start=1):
        print(f"  [{i}/{len(methods)}] {m} ...")
        try:
            _, t_stats = _impute_one_method(
                m, train_df, test_input_df, test_gt_df,
                target_cols, cat_encoded, ts_col, generated_dir,
            )
            timings[m] = t_stats["elapsed_sec"]
            print(f"      done in {t_stats['elapsed_sec']:.1f}s")
        except Exception as e:
            print(f"      FAILED: {e}")
            timings[m] = float("nan")

    print("\n[STEP 3/3] scoring against test_gt at holdout cells")
    overall_rows = []
    per_target_frames = []
    for m in methods:
        test_imputed_path = generated_dir / f"{m}_test_imputed.csv"
        if not test_imputed_path.exists():
            continue
        test_imputed_df = pd.read_csv(test_imputed_path)
        per_target, overall = _score(
            test_input_df, test_gt_df, test_imputed_df, target_cols
        )
        per_target["method"] = m
        per_target_frames.append(per_target)
        overall_rows.append({
            "method": m,
            **overall,
            "elapsed_sec": timings.get(m, float("nan")),
        })

    overall_df = pd.DataFrame(overall_rows).sort_values("MAE")
    per_target_df = (pd.concat(per_target_frames, ignore_index=True)
                     if per_target_frames else pd.DataFrame())

    results_csv = generated_dir / "results.csv"
    if not per_target_df.empty:
        per_target_df.to_csv(results_csv, index=False)
    (generated_dir / "results_overall.csv").write_text(
        overall_df.to_csv(index=False)
    )

    print("\n" + "=" * 78)
    print(f"OVERALL  (held-out cells across {len(target_cols)} targets)")
    print("=" * 78)
    with pd.option_context("display.float_format", "{:.4g}".format,
                           "display.max_rows", None, "display.width", 120):
        print(overall_df.to_string(index=False))

    print(f"\n→ per-target results:  {results_csv}")
    print(f"→ overall results:     {generated_dir / 'results_overall.csv'}")
    print(f"\nTo browse in the dashboard:")
    print(f"  streamlit run dashboard/app.py")
    print(f"  set sidebar 'Work root' = {args.output_root.resolve()}")
    print(f"  pick Dataset = {dataset} · Subset = {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
