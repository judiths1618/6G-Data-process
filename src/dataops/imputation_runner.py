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
    "builtin_methods",
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

# Dependency-free, **library-scoped** built-in methods. Standard, unambiguous ops
# only — each maps to ``(kind, arg)``. Library-specific algorithms (darts kalman,
# PyPOTS SAITS, ImputeGAP CDRec/BRITS, …) are NOT here; they still need the real
# library (run it via the app runner / ``DATAOPS_IMPUTE_CONDA_ENV``). Note that the
# same name can differ across libraries (darts ``zero`` = zero-order-hold interp;
# ImputeGAP ``zero`` = fill 0), which is exactly why the table is keyed by library.
_BUILTIN: dict[str, dict[str, tuple]] = {
    # darts interpolation family — bit-faithful to Darts' MissingValuesFiller.
    "darts": {m: ("interp", "linear" if m == "auto" else m) for m in INTERP_METHODS},
    # ImputeGAP statistics family — standard equivalents (not the ImputeGAP lib).
    "imputegap": {
        "interpolation": ("interp", "linear"),
        "mean": ("fill", "mean"),
        "mean_by_series": ("fill", "mean"),
        "min": ("fill", "min"),
        "zero": ("fill", "zero"),
    },
}


def builtin_methods(lib: str = "darts") -> list[str]:
    """Dependency-free method names available for ``lib`` (empty for heavy-only libs)."""
    return sorted(_BUILTIN.get(lib, {}))


def _generated_dirname(prepared: Path) -> str:
    """Sibling ``<base>_generated`` dir for a regularized bundle. Handles the
    stage-based name (``<base>_regularized``) and the legacy ``prepared_<base>`` /
    ``<base>_prepared`` layouts."""
    base = (prepared.name
            .removesuffix("_regularized")
            .removesuffix("_prepared")
            .removeprefix("prepared_"))
    return f"{base}_generated"


def _load_meta(prepared_dir: Path) -> dict:
    with (prepared_dir / "meta.json").open() as f:
        return json.load(f)


def _fill_series(series: pd.Series, kind: str, arg) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if kind == "interp":
        # interpolate over integer position, matching Darts' synthetic RangeIndex.
        filled = s.interpolate(method=arg, limit_direction="both")
        return filled.ffill().bfill() if filled.isna().any() else filled
    if kind == "fill":
        value = {"mean": s.mean(), "min": s.min(), "zero": 0.0}[arg]
        return s.fillna(value)
    raise ValueError(f"unknown built-in op {kind!r}")


def impute_dataframe(
    df: pd.DataFrame, target_cols: list[str], method: str, lib: str = "darts"
) -> pd.DataFrame:
    """Fill NaNs in ``target_cols`` per column using ``lib``'s built-in ``method``.

    ``darts`` interpolation is bit-faithful to Darts' MissingValuesFiller; the
    ``imputegap`` statistics methods are standard equivalents (not the ImputeGAP
    library). Raises ``ValueError`` for a non-built-in ``(lib, method)``.
    """
    spec = _BUILTIN.get(lib, {}).get(method)
    if spec is None:
        raise ValueError(
            f"{lib}/{method} is not a dependency-free built-in "
            f"(built-ins for {lib}: {builtin_methods(lib)})"
        )
    kind, arg = spec
    out = df.copy()
    for col in target_cols:
        if col not in out.columns:
            continue
        if pd.to_numeric(out[col], errors="coerce").notna().sum() == 0:
            continue
        out[col] = _fill_series(out[col], kind, arg)
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
    lib: str = "darts",
    output_dir: str | Path | None = None,
    inputs: Iterable[str] = ("train", "test"),
    engine: str = "pandas",
    python_exe: str | None = None,
    runner_path: str | Path | None = None,
) -> dict:
    """Impute the bundle's input CSVs and write ``<lib>_<method>_<kind>_imputed.csv``.

    With ``engine="pandas"`` the dependency-free built-ins for ``lib`` are used
    (see :data:`_BUILTIN`); ``engine="darts"`` subprocesses the real Darts runner.
    Returns ``{method, lib, engine, output_dir, files:{kind: {path, rows,
    nan_before, nan_after, filled}}}``.
    """
    prepared = Path(prepared_dir)
    out_dir = Path(output_dir) if output_dir else prepared.parent / _generated_dirname(prepared)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _load_meta(prepared)
    target_cols = meta["target_cols"]
    inputs = list(inputs)

    if engine == "darts":
        if lib != "darts":
            raise ValueError("engine='darts' only applies to lib='darts'")
        runner = Path(runner_path) if runner_path else (
            Path(__file__).resolve().parents[2]
            / "dockers" / "tools" / "Darts_app" / "run_imputation.py"
        )
        _run_darts_subprocess(prepared, out_dir, method, inputs,
                              python_exe or sys.executable, runner)
    elif engine != "pandas":
        raise ValueError(f"engine must be 'pandas' or 'darts', got {engine!r}")
    if engine == "pandas" and method not in _BUILTIN.get(lib, {}):
        raise ValueError(
            f"{lib}/{method} needs the real library (pandas built-ins for "
            f"{lib}: {builtin_methods(lib)})"
        )

    files: dict[str, dict] = {}
    for kind in inputs:
        src = prepared / INPUT_FILES[kind]
        if not src.exists():
            continue
        out_path = out_dir / f"{lib}_{method}_{kind}_imputed.csv"
        df = pd.read_csv(src)
        nan_before = int(df[target_cols].isna().sum().sum())
        if engine == "pandas":
            out_df = impute_dataframe(df, target_cols, method, lib=lib)
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
        "lib": lib,
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
    lib: str = "darts",
    bundle_result: dict | None = None,
    imputed_dir: str | Path | None = None,
    keep_cond_features: bool = False,
    python_exe: str | None = None,
) -> dict:
    """Produce THE final cleaned dataset: the full timeline, imputed & gap-free.

    This is the analysis-ready endpoint of the pipeline. It stitches together the
    *imputed* train and test splits — ``<lib>_<method>_train_imputed.csv`` and
    ``<lib>_<method>_test_imputed.csv`` as written by :func:`impute_bundle` — so
    the model/method's own imputation of **both** splits (including the imputed
    training data) lands in the final. It keeps the real columns (``time`` +
    ``target_cols``; engineered conditioning features are dropped unless
    ``keep_cond_features``) and writes a single gap-free CSV.

    The imputed split CSVs are located from ``bundle_result`` (an
    :func:`impute_bundle` return dict), then from ``imputed_dir`` /the default
    ``generated_<subset>`` dir; any split that is still missing is generated by
    calling :func:`impute_bundle` here. Because it consumes already-imputed files,
    this works for every engine (``pandas`` interp *and* the real ``darts``
    subprocess / model runners) — not just the pandas family.

    Returns ``{path, rows, columns, gaps_before, gaps_after, fill_rate, method}``.
    """
    prepared = Path(prepared_dir)
    meta = _load_meta(prepared)
    time_col = meta.get("time_col", "time")
    target_cols = meta["target_cols"]
    cond_cols = meta.get("cond_cols", []) if keep_cond_features else []

    splits = ("train", "test")
    out_dir = (
        Path(imputed_dir) if imputed_dir
        else prepared.parent / _generated_dirname(prepared)
    )

    # 1) Resolve the imputed split CSVs: prefer the bundle result, then look on
    #    disk, then generate whatever is still missing via impute_bundle.
    imputed_paths: dict[str, Path] = {}
    for kind in splits:
        info = (bundle_result or {}).get("files", {}).get(kind)
        if info and Path(info["path"]).exists():
            imputed_paths[kind] = Path(info["path"])
        elif (cand := out_dir / f"{lib}_{method}_{kind}_imputed.csv").exists():
            imputed_paths[kind] = cand
    missing = [k for k in splits if k not in imputed_paths]
    if missing:
        gen = impute_bundle(
            prepared, method=method, lib=lib, output_dir=out_dir,
            inputs=missing, engine=engine, python_exe=python_exe,
        )
        for kind in missing:
            imputed_paths[kind] = Path(gen["files"][kind]["path"])

    # 2) Stitch the imputed splits into one timeline (time + targets [+ cond]).
    frames = [pd.read_csv(imputed_paths[k]) for k in splits if k in imputed_paths]
    if not frames:
        raise FileNotFoundError(f"no imputed train/test splits found under {out_dir}")
    full = pd.concat(frames, ignore_index=True)
    keep = [c for c in [time_col, *target_cols, *cond_cols] if c in full.columns]
    full = full[keep].sort_values(time_col).drop_duplicates(subset=[time_col], keep="last")
    full = full.reset_index(drop=True)
    gaps_after = int(full[target_cols].isna().sum().sum())

    # 3) "gaps filled" is measured against the raw gappy inputs (train.csv +
    #    test_input.csv) restricted to the same rows.
    gaps_before = 0
    for kind in splits:
        raw = prepared / INPUT_FILES[kind]
        if raw.exists():
            gaps_before += int(pd.read_csv(raw)[target_cols].isna().sum().sum())

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)
    return {
        "path": str(out_path),
        "rows": int(len(full)),
        "columns": list(full.columns),
        "method": method,
        "sources": {k: str(v) for k, v in imputed_paths.items()},
        "gaps_before": gaps_before,
        "gaps_after": gaps_after,
        "fill_rate": (gaps_before - gaps_after) / gaps_before if gaps_before else 1.0,
    }


METADATA = {
    "name": "imputation_runner",
    "version": "0.1.0",
    "category": "imputation",
    "summary": "Automated dependency-free imputation over a prepared bundle "
               "(darts interpolation + imputegap statistics built-ins, or real Darts "
               "subprocess) + clean-vs-imputed comparison + final dataset.",
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
