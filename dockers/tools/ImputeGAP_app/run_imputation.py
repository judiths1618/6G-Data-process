#!/usr/bin/env python3
"""
ImputeGAP baseline imputation runner (train.csv + test_input.csv).

Reads `prepared_<subset>/` and writes one imputed CSV per input into
`--output-dir`:

    imputegap_<method>_train_imputed.csv
    imputegap_<method>_test_imputed.csv

ImputeGAP works on a (n_series, n_timestamps) numeric matrix with NaNs.
We feed only `target_cols` (the columns that may contain gaps), keeping
`time_col` and `cond_cols` untouched in the output.

Method registry is resolved against the installed `imputegap` version
(`imputegap.recovery.imputation.Imputation`); pass `--list` to print what
your install actually exposes.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

INPUT_FILES = {"train": "train.csv", "test": "test_input.csv"}


def _resolve_imputation():
    try:
        from imputegap.recovery.imputation import Imputation
        return Imputation
    except Exception as exc:
        raise ImportError(
            "Install imputegap >=1.1 (`pip install imputegap`)."
        ) from exc


def _build_registry(Imputation) -> Dict[str, Callable]:
    """Map flat method-name -> factory(matrix) -> imputer instance.

    Mirrors the v1.1.x layout (groups: Statistics, MachineLearning,
    PatternSearch, MatrixCompletion, DeepLearning, LLMs)."""
    candidates = {
        # --- Statistics -----------------------------------------------------
        "mean":          "Statistics.MeanImpute",
        "mean_by_series":"Statistics.MeanImputeBySeries",
        "min":           "Statistics.MinImpute",
        "zero":          "Statistics.ZeroImpute",
        "interpolation": "Statistics.Interpolation",
        "knn":           "Statistics.KNNImpute",
        # --- Matrix completion ---------------------------------------------
        "cdrec":         "MatrixCompletion.CDRec",
        "grouse":        "MatrixCompletion.GROUSE",
        "iterative_svd": "MatrixCompletion.IterativeSVD",
        "rosl":          "MatrixCompletion.ROSL",
        "spirit":        "MatrixCompletion.SPIRIT",
        "svt":           "MatrixCompletion.SVT",
        "soft_impute":   "MatrixCompletion.SoftImpute",
        "trmf":          "MatrixCompletion.TRMF",
        # --- Pattern search -------------------------------------------------
        "dynammo":       "PatternSearch.DynaMMo",
        "stmvl":         "PatternSearch.STMVL",
        "tkcm":          "PatternSearch.TKCM",
        # --- Machine learning ----------------------------------------------
        "iim":           "MachineLearning.IIM",
        "mice":          "MachineLearning.MICE",
        "miss_forest":   "MachineLearning.MissForest",
        "xgboost":       "MachineLearning.XGBOOST",
        # --- Deep learning --------------------------------------------------
        "brits":         "DeepLearning.BRITS",
        "mrnn":          "DeepLearning.MRNN",
        "gain":          "DeepLearning.GAIN",
        "deep_mvi":      "DeepLearning.DeepMVI",
        "miss_net":      "DeepLearning.MissNet",
        "pristi":        "DeepLearning.PRISTI",
    }
    registry: Dict[str, Callable] = {}
    for name, path in candidates.items():
        cur = Imputation
        try:
            for part in path.split("."):
                cur = getattr(cur, part)
            registry[name] = cur
        except AttributeError:
            continue
    return registry


def load_meta(prepared_dir: Path) -> dict:
    with (prepared_dir / "meta.json").open() as f:
        return json.load(f)


def _impute_matrix(factory: Callable, matrix: np.ndarray) -> np.ndarray:
    """Run an ImputeGAP imputer on a (n_series, n_steps) matrix and return the
    filled matrix. Falls back to per-column mean if the imputer leaves NaNs."""
    imputer = factory(matrix)
    imputer.impute()
    out = np.asarray(imputer.recov_data, dtype=np.float64)
    if np.isnan(out).any():
        # Some methods (e.g. BRITS with very high missing ratio) may leave gaps.
        col_means = np.nanmean(matrix, axis=1, keepdims=True)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_mask = np.isnan(out)
        out = np.where(nan_mask, np.broadcast_to(col_means, out.shape), out)
    return out


def run(args: argparse.Namespace) -> List[Path]:
    Imputation = _resolve_imputation()
    registry = _build_registry(Imputation)

    if args.list:
        print("Available ImputeGAP methods (resolved from this install):")
        for k in sorted(registry):
            print(f"  - {k}")
        sys.exit(0)

    if args.method not in registry:
        raise SystemExit(
            f"Method '{args.method}' not available. Run with --list to see options."
        )

    prepared = Path(args.prepared_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(prepared)
    target_cols = meta["target_cols"]
    factory = registry[args.method]

    written: List[Path] = []
    for kind in args.inputs:
        src = prepared / INPUT_FILES[kind]
        if not src.exists():
            print(f"[ImputeGAP]   skip: {src} not found")
            continue
        df = pd.read_csv(src)
        targets = [c for c in target_cols if c in df.columns]
        if not targets:
            print(f"[ImputeGAP]   skip: no target columns in {src}")
            continue

        # Rows = series, columns = timestamps (ImputeGAP convention).
        matrix = df[targets].to_numpy(dtype=np.float64).T
        nan_before = int(np.isnan(matrix).sum())
        all_nan_rows = int((np.isnan(matrix).sum(axis=1) == matrix.shape[1]).sum())
        if all_nan_rows:
            print(f"[ImputeGAP]   WARN: {all_nan_rows} target columns are all-NaN in {kind}")
        print(f"[ImputeGAP] {kind}: matrix={matrix.shape}  NaN={nan_before}  method={args.method}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                filled = _impute_matrix(factory, matrix)
            except Exception as exc:
                msg = str(exc).splitlines()[0]
                print(f"[ImputeGAP]   ✗ {args.method} failed on {kind}: {msg}")
                continue

        if filled.shape != matrix.shape:
            # ImputeGAP 1.1.x has a few methods (e.g. MICE) whose `recov_data`
            # returns a partial/holdout shape rather than the full matrix.
            # We refuse to write a CSV in that case rather than guess.
            print(f"[ImputeGAP]   ✗ {args.method} on {kind}: returned shape "
                  f"{filled.shape} != input {matrix.shape}; skipping")
            continue

        nan_after = int(np.isnan(filled).sum())
        print(f"[ImputeGAP] {kind}: NaN {nan_before} -> {nan_after}")

        out_df = df.copy()
        out_df[targets] = filled.T
        out_path = output_dir / f"imputegap_{args.method}_{kind}_imputed.csv"
        out_df.to_csv(out_path, index=False)
        print(f"[ImputeGAP] wrote {out_path}")
        written.append(out_path)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="ImputeGAP baseline imputation runner")
    p.add_argument("--prepared-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--method", default="iim")
    p.add_argument("--inputs", nargs="+", default=["train", "test"],
                   choices=list(INPUT_FILES.keys()))
    p.add_argument("--list", action="store_true",
                   help="List the imputation methods resolved from the installed ImputeGAP")
    run(p.parse_args())


if __name__ == "__main__":
    main()
