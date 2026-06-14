"""
imputation_runner — automated time-series imputation over a prepared-dir bundle,
plus a clean-vs-imputed comparison.

This closes the handoff loop *executably* for the interpolation family (the
Darts baselines). It is dependency-light by default: the ``pandas`` engine
reproduces Darts' ``MissingValuesFiller`` exactly — that filler forwards to
``pandas.Series.interpolate(method=...)`` and ffills/bfills the edges, so
``darts/nearest`` here is bit-faithful to the Docker runner without importing
``darts``. Set ``engine="darts"`` to subprocess the real
``Darts_app/run_imputation.py`` where it is installed (the ``autofeat-6g`` env /
the Darts image).

Produces files named like the Docker runners so the dashboard discovers them:

    <output_dir>/darts_<method>_<train|test>_imputed.csv
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

__all__ = [
    "INTERP_METHODS",
    "impute_dataframe",
    "impute_bundle",
    "compare_clean_vs_imputed",
    "build_final_dataset",
    "METADATA",
]

# Mirrors Darts_app/run_imputation.py INTERP_METHODS (forwarded to pandas).
INTERP_METHODS = {"auto", "linear", "quadratic", "cubic", "nearest", "slinear", "zero"}
INPUT_FILES = {"train": "train.csv", "test": "test_input.csv"}
GT_FILES = {"train": None, "test": "test_gt.csv"}


def _load_meta(prepared_dir: Path) -> dict:
    with (prepared_dir / "meta.json").open() as f:
        return json.load(f)


def impute_dataframe(df: pd.DataFrame, target_cols: list[str], method: str) -> pd.DataFrame:
    """Fill NaNs in ``target_cols`` per column — faithful to Darts MissingValuesFiller.

    ``method`` is forwarded to ``pandas.Series.interpolate``; ``auto`` uses the
    pandas default (linear). Edge NaNs left by some methods are ff/bf filled,
    matching the Darts runner.
    """
    out = df.copy()
    pandas_method = "linear" if method == "auto" else method
    for col in target_cols:
        if col not in out.columns or out[col].notna().sum() == 0:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        # interpolate over integer position, matching Darts' synthetic RangeIndex.
        filled = series.interpolate(method=pandas_method, limit_direction="both")
        if filled.isna().any():
            filled = filled.ffill().bfill()
        out[col] = filled
    return out


def _run_darts_subprocess(
    prepared_dir: Path, output_dir: Path, method: str, inputs: list[str],
    python_exe: str, runner: Path,
) -> None:
    cmd = [
        python_exe, str(runner),
        "--prepared-dir", str(prepared_dir),
        "--output-dir", str(output_dir),
        "--method", method,
        "--inputs", *inputs,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"darts runner failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )


def impute_bundle(
    prepared_dir: str | Path,
    *,
    method: str = "nearest",
    output_dir: str | Path | None = None,
    inputs: Iterable[str] = ("train", "test"),
    engine: str = "pandas",
    python_exe: str | None = None,
    runner_path: str | Path | None = None,
) -> dict:
    """Impute the bundle's input CSVs and write ``darts_<method>_<kind>_imputed.csv``.

    Returns ``{method, engine, output_dir, files:{kind: {path, rows, nan_before,
    nan_after, filled}}}``.
    """
    prepared = Path(prepared_dir)
    out_dir = Path(output_dir) if output_dir else prepared.parent / f"generated_{prepared.name.removeprefix('prepared_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _load_meta(prepared)
    target_cols = meta["target_cols"]
    inputs = list(inputs)

    if engine == "darts":
        runner = Path(runner_path) if runner_path else (
            Path(__file__).resolve().parents[2]
            / "dockers" / "tools" / "Darts_app" / "run_imputation.py"
        )
        _run_darts_subprocess(prepared, out_dir, method, inputs,
                              python_exe or sys.executable, runner)
    elif engine != "pandas":
        raise ValueError(f"engine must be 'pandas' or 'darts', got {engine!r}")
    if engine == "pandas" and method not in INTERP_METHODS:
        raise ValueError(
            f"method {method!r} needs engine='darts' (pandas engine supports "
            f"{sorted(INTERP_METHODS)})"
        )

    files: dict[str, dict] = {}
    for kind in inputs:
        src = prepared / INPUT_FILES[kind]
        if not src.exists():
            continue
        out_path = out_dir / f"darts_{method}_{kind}_imputed.csv"
        df = pd.read_csv(src)
        nan_before = int(df[target_cols].isna().sum().sum())
        if engine == "pandas":
            out_df = impute_dataframe(df, target_cols, method)
            out_df.to_csv(out_path, index=False)
        else:  # darts engine already wrote the file
            out_df = pd.read_csv(out_path)
        nan_after = int(out_df[target_cols].isna().sum().sum())
        files[kind] = {
            "path": str(out_path),
            "rows": int(len(out_df)),
            "nan_before": nan_before,
            "nan_after": nan_after,
            "filled": nan_before - nan_after,
        }
    return {
        "method": method,
        "engine": engine,
        "output_dir": str(out_dir),
        "target_cols": target_cols,
        "files": files,
    }


def _err_metrics(err: np.ndarray, truth: np.ndarray) -> dict:
    denom = np.where(np.abs(truth) < 1e-9, 1e-9, np.abs(truth))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(np.mean(np.abs(err / denom)) * 100),
    }


def _score_eval_cells(
    input_arr: np.ndarray, gt_arr: np.ndarray, imputed_arr: np.ndarray,
    target_cols: list[str],
) -> dict:
    """MAE / RMSE / MAPE on cells NaN in input but known in GT (the eval mask).

    Reports per-column metrics too — the pooled figure mixes columns of wildly
    different scale (microseconds vs bytes vs ratios) and is not interpretable on
    its own.
    """
    miss = np.isnan(input_arr)
    eval_mask = miss & ~np.isnan(gt_arr) & ~np.isnan(imputed_arr)
    n_eval = int(eval_mask.sum())
    if n_eval == 0:
        return {"eval_cells": 0, "pooled": None, "per_column": {}}
    err = imputed_arr[eval_mask] - gt_arr[eval_mask]
    per_column: dict[str, dict] = {}
    for j, col in enumerate(target_cols):
        cm = eval_mask[:, j]
        n = int(cm.sum())
        if n == 0:
            continue
        col_err = imputed_arr[cm, j] - gt_arr[cm, j]
        per_column[col] = {"eval_cells": n, **_err_metrics(col_err, gt_arr[cm, j])}
    return {
        "eval_cells": n_eval,
        "pooled": {**_err_metrics(err, gt_arr[eval_mask]),
                   "note": "pooled across scales — see per_column for interpretable metrics"},
        "per_column": per_column,
    }


def compare_clean_vs_imputed(
    prepared_dir: str | Path, impute_result: dict
) -> dict:
    """Compare the regularized (gappy) input against the imputed output.

    Reports fill coverage per split + per column, and — on the ``test`` split,
    where ``test_gt.csv`` holds the masked-out truth — accuracy on the eval cells.
    """
    prepared = Path(prepared_dir)
    meta = _load_meta(prepared)
    target_cols = meta["target_cols"]
    comparison: dict = {"method": impute_result["method"], "splits": {}}

    for kind, info in impute_result["files"].items():
        src = prepared / INPUT_FILES[kind]
        inp = pd.read_csv(src)
        imp = pd.read_csv(info["path"])
        in_arr = inp[target_cols].to_numpy(dtype=float)
        imp_arr = imp[target_cols].to_numpy(dtype=float)
        miss = np.isnan(in_arr)
        n_miss = int(miss.sum())
        filled = int((miss & ~np.isnan(imp_arr)).sum())

        split_report: dict = {
            "rows": int(len(inp)),
            "missing_cells": n_miss,
            "filled_cells": filled,
            "fill_rate": (filled / n_miss) if n_miss else 1.0,
            "residual_nan": int(np.isnan(imp_arr).sum()),
            "per_column": {
                col: {
                    "missing": int(np.isnan(in_arr[:, j]).sum()),
                    "filled": int((np.isnan(in_arr[:, j]) & ~np.isnan(imp_arr[:, j])).sum()),
                }
                for j, col in enumerate(target_cols)
            },
        }
        gt_name = GT_FILES.get(kind)
        gt_path = prepared / gt_name if gt_name else None
        if gt_path and gt_path.exists():
            gt = pd.read_csv(gt_path)
            split_report["accuracy"] = _score_eval_cells(
                in_arr, gt[target_cols].to_numpy(dtype=float), imp_arr, target_cols
            )
        comparison["splits"][kind] = split_report
    return comparison


def build_final_dataset(
    prepared_dir: str | Path,
    *,
    method: str = "nearest",
    output_path: str | Path,
    engine: str = "pandas",
    keep_cond_features: bool = False,
    python_exe: str | None = None,
) -> dict:
    """Produce THE final cleaned dataset: the full regularized timeline, gap-filled.

    This is the analysis-ready endpoint of the pipeline. It reconstructs the
    *complete* regularized timeline — ``train.csv`` + ``test_gt.csv`` (the true
    grid; NOT ``test_input.csv``, which carries the artificial eval holdout) —
    keeps the real columns (``time`` + ``target_cols``; engineered conditioning
    features are dropped unless ``keep_cond_features``), imputes the genuine gaps
    with ``method``, and writes a single gap-free CSV.

    Returns ``{path, rows, columns, gaps_before, gaps_after, fill_rate, method}``.
    """
    prepared = Path(prepared_dir)
    meta = _load_meta(prepared)
    time_col = meta.get("time_col", "time")
    target_cols = meta["target_cols"]
    cond_cols = meta.get("cond_cols", []) if keep_cond_features else []

    if engine == "pandas" and method not in INTERP_METHODS:
        raise ValueError(
            f"method {method!r} needs engine='darts'; pandas supports {sorted(INTERP_METHODS)}"
        )

    frames = []
    for name in ("train.csv", "test_gt.csv"):
        fp = prepared / name
        if fp.exists():
            frames.append(pd.read_csv(fp))
    if not frames:
        raise FileNotFoundError(f"no train.csv/test_gt.csv under {prepared}")

    full = pd.concat(frames, ignore_index=True)
    keep = [c for c in [time_col, *target_cols, *cond_cols] if c in full.columns]
    full = full[keep].sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")
    full = full.reset_index(drop=True)

    gaps_before = int(full[target_cols].isna().sum().sum())
    if engine == "darts":
        # Reuse the real runner via a temp single-input bundle is overkill; the
        # interp families are identical, so route engine='darts' only for kalman
        # etc. by deferring to impute_dataframe's pandas path is not possible.
        raise NotImplementedError(
            "build_final_dataset currently supports the pandas (interp) engine; "
            "for the full Darts/kalman path run impute_bundle(engine='darts') and "
            "concatenate train + test_gt outputs."
        )
    filled = impute_dataframe(full, target_cols, method)
    gaps_after = int(filled[target_cols].isna().sum().sum())

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    filled.to_csv(out_path, index=False)
    return {
        "path": str(out_path),
        "rows": int(len(filled)),
        "columns": list(filled.columns),
        "method": method,
        "gaps_before": gaps_before,
        "gaps_after": gaps_after,
        "fill_rate": (gaps_before - gaps_after) / gaps_before if gaps_before else 1.0,
    }


METADATA = {
    "name": "imputation_runner",
    "version": "0.1.0",
    "category": "imputation",
    "summary": "Automated interpolation-family imputation over a prepared bundle "
               "(Darts-faithful pandas engine or real Darts subprocess) + clean-vs-imputed comparison.",
    "entrypoint": "dataops.imputation_runner:impute_bundle",
    "gpu": False,
    "dependencies": ["pandas", "numpy", "scipy"],
    "inputs": {
        "prepared_dir": {"type": "str", "required": True},
        "method": {"type": "str", "default": "nearest"},
        "output_dir": {"type": "str", "default": None},
        "inputs": {"type": "list[str]", "default": ["train", "test"]},
        "engine": {"type": "str", "default": "pandas"},
    },
    "outputs": {
        "result": {"type": "dict", "schema": "impute_bundle_result",
                   "keys": ["method", "engine", "output_dir", "files"]},
    },
    "artifacts": ["darts_<method>_<split>_imputed.csv"],
}
