"""Complete DataOps pipeline: raw CSV → final gap-free dataset, imputation included.

:mod:`pipelines.minimal_dataops` deliberately stops at the imputation *handoff*.
It cleans, remediates, regularizes a gappy timeline into ``<name>_regularized/``
and advertises a method catalog — but it never fills a cell (see that module's
``_build_handoff``: *"The pipeline never runs imputation"*). Closing the loop has
so far needed three more entrypoints: ``scripts/auto_impute.py`` for the
dependency-free built-ins, the per-app ``dockers/tools/*/run_imputation.py``
runners for everything else, and ``scripts/reproduce_all.sh`` to sequence them.

This module is the single entrypoint that runs all of it, end to end::

    raw CSV
      → <name>_soft_cleaned.csv → <name>_remediated.csv      stage 1  clean
      → <name>_regularized/  (train, test_input, test_gt, …) stage 2  regularize
      → <name>_generated/<key>_<split>_imputed.csv           stage 3  impute
      → out-of-band filled cells reported (opt-in: clipped)  stage 4a outliers
      → fill coverage + shared-holdout leaderboard           stage 4b score
      → <name>_generated/<key>_final.csv + <name>_final.csv  stage 5  finalize
      → reports/<name>_imputation_compare.json               stage 6  report

Stages 1–2 delegate to :func:`pipelines.minimal_dataops.run_pipeline`, so the
cleaning contract is unchanged and a report written here is the same report the
dashboard already reads. Stage 3 dispatches each method to whichever engine can
run it: the in-process pandas built-ins (darts interpolation, imputegap
statistics) or the app runner as a subprocess (darts kalman, imputegap ML,
PyPOTS, WaveStitch+ v1/v2/harpoon).

Usage::

    # every dependency-free method on one dataset
    python -m pipelines.dataops_imputation_completes --dataset rabbitmq-performance

    # the full comparison, WaveStitch+ included, on all four bundled datasets
    python -m pipelines.dataops_imputation_completes \
        --dataset amf-performance,golang-web-server-performance \
        --methods all --device cpu --fast

    # reuse an existing bundle (skip cleaning) and add one method
    python -m pipelines.dataops_imputation_completes --dataset amf-performance \
        --skip-clean --methods pypots/saits --pypots-epochs 15

Outlier handling is opt-in at both ends and always reported: ``--clip-outliers``
winsorizes the raw data during cleaning, ``--clip-imputed-outliers`` clips *filled*
cells that land outside the observed plausible band (observed cells are never
modified). Neither runs by default — rewriting a measurement, or a model's output,
should be a decision rather than a side effect.

A method that fails is recorded and the run continues (``--strict`` aborts the
dataset instead), matching ``reproduce_all.sh``. Heavy libraries can live in
another env: ``DATAOPS_IMPUTE_PYTHON`` / ``DATAOPS_IMPUTE_CONDA_ENV`` are honoured
exactly as the dashboard honours them.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_process_modules.config import load_config
from dataops.imputation_runner import (
    build_final_dataset,
    builtin_methods,
    compare_clean_vs_imputed,
    impute_bundle,
)

from pipelines.minimal_dataops import configure_logging, run_pipeline

LOGGER = logging.getLogger("dataops.pipeline.complete")

# Bundle contract (mirrors dataops.imputation_runner / the app runners).
INPUT_FILES = {"train": "train.csv", "test": "test_input.csv"}

APP_RUNNERS: dict[str, Path] = {
    "darts": PROJECT_ROOT / "dockers" / "tools" / "Darts_app" / "run_imputation.py",
    "imputegap": PROJECT_ROOT / "dockers" / "tools" / "ImputeGAP_app" / "run_imputation.py",
    "pypots": PROJECT_ROOT / "dockers" / "tools" / "PyPOTS_app" / "run_imputation.py",
    "wavestitchplus": PROJECT_ROOT / "dockers" / "tools" / "WaveStitchPlus_app" / "run_imputation.py",
    "wavestitchplus_v2": PROJECT_ROOT / "dockers" / "tools" / "WaveStitchPlus_app" / "run_imputation_v2.py",
    "wavestitchplus_harpoon": PROJECT_ROOT / "dockers" / "tools" / "WaveStitchPlus_app" / "run_imputation_harpoon.py",
}

# Libraries with a dependency-free pandas engine for *some* of their methods.
BUILTIN_LIBS = ("darts", "imputegap")

# Method used when a spec names only a library (e.g. ``--methods pypots``).
# These are the runnable-here defaults, which is deliberately not always the
# catalog's advertised default: ImputeGAP advertises ``iim``, but a bare
# ``--methods imputegap`` should not require the real library to be installed,
# so it resolves to the dependency-free ``interpolation`` instead.
DEFAULT_METHOD = {
    "darts": "auto",
    "imputegap": "interpolation",
    "pypots": "saits",
    "wavestitchplus": "v1",
    "wavestitchplus_v2": "anchored",
    "wavestitchplus_harpoon": "harpoon",
}

# config/dataops.yaml names apps by their catalog label; map to a runner key.
APP_TO_LIB = {
    "darts": "darts",
    "imputegap": "imputegap",
    "pypots": "pypots",
    "wavestitchplus": "wavestitchplus",
}

# WaveStitch+ v2/harpoon read the v1 diffusion output, so v1 must run first and
# the two dependants must run after it. Stable sort keeps user order within a rank.
_RUN_ORDER = {"wavestitchplus": 0, "wavestitchplus_v2": 2, "wavestitchplus_harpoon": 2}


# ---------------------------------------------------------------------------
# Method specs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MethodSpec:
    """One ``(library, method)`` pair to run against a bundle."""

    lib: str
    method: str

    @property
    def label(self) -> str:
        return f"{self.lib}/{self.method}"

    @property
    def key(self) -> str:
        """Filename stem the runner writes, e.g. ``darts_linear``, ``wavestitchplus_v2``."""
        return output_name(self.lib, self.method, "test").removesuffix("_test_imputed.csv")

    @property
    def is_builtin(self) -> bool:
        return self.lib in BUILTIN_LIBS and self.method in builtin_methods(self.lib)


def output_name(lib: str, method: str, split: str) -> str:
    """Filename each runner writes for ``(lib, method, split)``.

    Most runners follow ``<lib>_<method>_<split>_imputed.csv``. The WaveStitch+
    family does not: v2/harpoon carry the variant in the *library* half of the
    name and ignore the method token (``--method`` exists there only for CLI
    compatibility), so the mapping is spelled out rather than derived.
    """
    if lib == "wavestitchplus":
        tag = "v1" if method == "full" else method       # ``full`` is a v1 alias
        return f"wavestitchplus_{tag}_{split}_imputed.csv"
    if lib == "wavestitchplus_v2":
        tag = "v2_tuned" if method == "tuned" else "v2"
        return f"wavestitchplus_{tag}_{split}_imputed.csv"
    if lib == "wavestitchplus_harpoon":
        return f"wavestitchplus_harpoon_{split}_imputed.csv"
    return f"{lib}_{method}_{split}_imputed.csv"


BUILTIN_SPECS = [
    *[MethodSpec("darts", m) for m in builtin_methods("darts")],
    *[MethodSpec("imputegap", m) for m in builtin_methods("imputegap")],
]
WSP_SPECS = [
    MethodSpec("wavestitchplus", "v1"),
    MethodSpec("wavestitchplus_v2", "anchored"),
    MethodSpec("wavestitchplus_harpoon", "harpoon"),
]
DEEP_SPECS = [MethodSpec("pypots", "saits"), MethodSpec("pypots", "brits")]

PRESETS: dict[str, list[MethodSpec]] = {
    "interp": [MethodSpec("darts", m) for m in builtin_methods("darts")],
    "builtin": list(BUILTIN_SPECS),
    "wsp": list(WSP_SPECS),
    "deep": list(DEEP_SPECS),
    "smoke": [MethodSpec("darts", "linear"), MethodSpec("imputegap", "mean"),
              MethodSpec("wavestitchplus", "v1")],
    "all": [*BUILTIN_SPECS, *WSP_SPECS, *DEEP_SPECS],
}


def parse_spec(raw: str) -> MethodSpec:
    """Parse one method spec.

    Accepted forms::

        darts/linear   pypots/saits   wavestitchplus_v2/anchored   (explicit)
        darts_linear   pypots_saits                                (underscore)
        pypots         wavestitchplus_v2                           (library default)
        v1  v2  harpoon                                            (WaveStitch+)
        linear                                                     (bare → darts)
    """
    s = raw.strip()
    if not s:
        raise ValueError("empty method spec")

    shorthand = {"v1": ("wavestitchplus", "v1"),
                 "v2": ("wavestitchplus_v2", "anchored"),
                 "harpoon": ("wavestitchplus_harpoon", "harpoon")}
    if s in shorthand:
        return MethodSpec(*shorthand[s])
    if s in APP_RUNNERS:                       # library name alone
        return MethodSpec(s, DEFAULT_METHOD[s])
    if "/" in s:
        lib, method = (part.strip() for part in s.split("/", 1))
    else:
        for lib_name in sorted(APP_RUNNERS, key=len, reverse=True):
            if s.startswith(f"{lib_name}_"):
                lib, method = lib_name, s[len(lib_name) + 1:]
                break
        else:
            lib, method = "darts", s
    if lib not in APP_RUNNERS:
        raise ValueError(f"unknown library {lib!r} in spec {raw!r} "
                         f"(known: {', '.join(APP_RUNNERS)})")
    if not method:
        raise ValueError(f"spec {raw!r} names a library but no method")
    return MethodSpec(lib, method)


def resolve_specs(items: Iterable[str]) -> list[MethodSpec]:
    """Expand presets and specs into an ordered, de-duplicated method list."""
    out: list[MethodSpec] = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        out.extend(PRESETS[item] if item in PRESETS else [parse_spec(item)])
    seen: set[MethodSpec] = set()
    unique = [s for s in out if not (s in seen or seen.add(s))]
    return sorted(unique, key=lambda s: _RUN_ORDER.get(s.lib, 1))


# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

@dataclass
class DatasetPaths:
    """Stage-based artifact paths for one dataset (see config/dataops.yaml)."""

    name: str
    raw_csv: Path
    remediated_csv: Path
    report_json: Path
    log_file: Path | None = None

    @classmethod
    def from_name(cls, name: str, *, root: Path = PROJECT_ROOT) -> "DatasetPaths":
        return cls(
            name=name,
            raw_csv=root / "data" / "raw" / f"{name}.csv",
            remediated_csv=root / "data" / "processed" / f"{name}_remediated.csv",
            report_json=root / "reports" / f"{name}_report.json",
            log_file=root / "logs" / f"{name}-dataops.log",
        )

    @classmethod
    def from_config(cls, cfg: dict, *, root: Path = PROJECT_ROOT) -> "DatasetPaths":
        raw = Path(cfg["input"])
        return cls(
            name=raw.stem,
            raw_csv=_abs(raw, root),
            remediated_csv=_abs(Path(cfg["output"]), root),
            report_json=_abs(Path(cfg["report"]), root),
            log_file=_abs(Path(cfg["log_file"]), root) if cfg.get("log_file") else None,
        )

    @property
    def final_csv(self) -> Path:
        """Canonical gap-free endpoint: ``data/processed/<name>_final.csv``."""
        stem = self.remediated_csv.stem.removesuffix("_remediated")
        return self.remediated_csv.with_name(f"{stem}_final.csv")


def _abs(path: Path, root: Path = PROJECT_ROOT) -> Path:
    return path if path.is_absolute() else root / path


def generated_dir_for(prepared: Path) -> Path:
    """Sibling ``<base>_generated`` dir for a bundle (all three layouts)."""
    n = prepared.name
    if n.endswith("_regularized"):
        return prepared.parent / f"{n.removesuffix('_regularized')}_generated"
    if n.startswith("prepared_"):
        return prepared.parent / f"generated_{n.removeprefix('prepared_')}"
    return prepared.parent / f"generated_{n.removesuffix('_prepared')}"


# ---------------------------------------------------------------------------
# Runner options + execution
# ---------------------------------------------------------------------------

@dataclass
class RunnerOptions:
    """Hyperparameters forwarded to the app runners that accept them."""

    device: str = "auto"
    fast: bool = False
    em_iterations: int = 3
    epochs_per_em: int = 50
    ddim_steps: int = 30
    repaint_rounds: int | None = None
    pypots_epochs: int = 50
    pypots_window: int = 100
    pypots_batch_size: int = 32
    timeout: float | None = None

    @property
    def synthesis_ddim(self) -> int:
        """DDIM steps for the synthesis-only runners (v2 fallback, harpoon).

        ``--fast`` sets v1's own hyperparams through its ``--fast`` flag, which
        those two runners do not have — without this they would keep running a
        full 50-step synthesis inside a smoke run (the same 5 steps
        ``reproduce_all.sh`` uses for its smoke HARPOON_ARGS).
        """
        return 5 if self.fast else self.ddim_steps


def impute_python() -> tuple[list[str], bool]:
    """Interpreter that runs the app runners; ``(argv_prefix, is_custom)``.

    Mirrors the dashboard's resolution so one env setting drives both:
    ``DATAOPS_IMPUTE_PYTHON`` → that interpreter, ``DATAOPS_IMPUTE_CONDA_ENV`` →
    ``conda run -n <env> python``, neither → this interpreter.
    """
    py = os.environ.get("DATAOPS_IMPUTE_PYTHON")
    if py:
        return [py], True
    env = os.environ.get("DATAOPS_IMPUTE_CONDA_ENV")
    if env:
        return ["conda", "run", "--no-capture-output", "-n", env, "python"], True
    return [sys.executable], False


def _extra_args(spec: MethodSpec, generated: Path, opts: RunnerOptions) -> list[str]:
    """Per-library CLI arguments for one method."""
    if spec.lib == "pypots":
        return ["--epochs", str(opts.pypots_epochs),
                "--window", str(opts.pypots_window),
                "--batch-size", str(opts.pypots_batch_size)]

    if spec.lib == "wavestitchplus":
        args = ["--device", opts.device]
        if opts.fast:
            args.append("--fast")
        else:
            args += ["--em-iterations", str(opts.em_iterations),
                     "--epochs-per-em", str(opts.epochs_per_em),
                     "--ddim-steps", str(opts.ddim_steps)]
        if opts.repaint_rounds is not None:
            args += ["--repaint-rounds", str(opts.repaint_rounds)]
        return args

    if spec.lib == "wavestitchplus_v2":
        args = ["--device", opts.device, "--ddim-steps", str(opts.synthesis_ddim)]
        # Anchoring is pure-python: reusing the v1 test output skips synthesis
        # entirely (no torch, no GPU). Same shortcut the dashboard takes.
        v1_test = generated / "wavestitchplus_v1_test_imputed.csv"
        if v1_test.exists():
            args += ["--reuse-diffusion", str(v1_test)]
        return args

    if spec.lib == "wavestitchplus_harpoon":
        args = ["--device", opts.device, "--ddim-steps", str(opts.synthesis_ddim)]
        repaint = 1 if opts.fast and opts.repaint_rounds is None else opts.repaint_rounds
        if repaint is not None:
            args += ["--repaint-rounds", str(repaint)]
        return args

    return []


def _run_app_runner(spec: MethodSpec, prepared: Path, generated: Path,
                    splits: Sequence[str], opts: RunnerOptions) -> tuple[int, str]:
    runner = APP_RUNNERS[spec.lib]
    if not runner.exists():
        return 127, f"runner not found: {runner}"
    cmd = [*impute_python()[0], str(runner),
           "--prepared-dir", str(prepared),
           "--output-dir", str(generated),
           "--method", spec.method,
           "--inputs", *splits,
           *_extra_args(spec, generated, opts)]
    LOGGER.debug("running %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=opts.timeout)
    except FileNotFoundError as exc:
        return 127, f"interpreter not found: {exc}"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {opts.timeout}s"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


# ---------------------------------------------------------------------------
# Output validation + result assembly
# ---------------------------------------------------------------------------

@dataclass
class OutlierPolicy:
    """Post-imputation outlier handling for the *filled* cells.

    Imputation invents values, and some methods invent impossible ones: polynomial
    interpolation overshoots (darts cubic writes 2,026 negative ``min_ms`` cells on
    rabbitmq), and neural imputers can undershoot below zero. This is the same
    report-by-default / clip-on-request contract the cleaning stage uses, applied
    one stage later and to the opposite set of cells.

    ``clip`` rewrites only cells that were NaN in the prepared input — observed
    measurements are never touched, matching scripts/enforce_monotone.py.
    """
    clip: bool = False
    #: Fallback band quantile when the bundle stores no bounds.
    quantile: float = 0.005


def load_target_bounds(prepared: Path, target_cols: Sequence[str],
                       quantile: float = 0.005) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Per-column ``(lower, upper)`` plausible band, derived from OBSERVED data.

    Prefers the bundle's stored bounds (``scaler/lower_bound.npy`` and
    ``scaler/upper_bound_p995.npy``, aligned to ``meta['target_cols']``) — the same
    band the WaveStitch+ runners clamp to. Falls back to observed quantiles over
    train + test_input. Never derived from the imputed frame, which would let the
    outliers widen their own band.
    """
    scaler = prepared / "scaler"
    lo_p, hi_p = scaler / "lower_bound.npy", scaler / "upper_bound_p995.npy"
    if lo_p.exists() and hi_p.exists():
        lo, hi = np.load(lo_p), np.load(hi_p)
        if lo.shape == hi.shape == (len(target_cols),):
            return lo.astype(float), hi.astype(float)
        LOGGER.warning("bundle bounds are %s for %d target cols — using quantiles",
                       lo.shape, len(target_cols))

    frames = [pd.read_csv(prepared / f) for f in INPUT_FILES.values()
              if (prepared / f).exists()]
    if not frames:
        return None
    obs = pd.concat(frames, ignore_index=True)
    cols = [c for c in target_cols if c in obs.columns]
    if len(cols) != len(target_cols):
        return None
    arr = obs[list(target_cols)].to_numpy(float)
    return (np.nanquantile(arr, quantile, axis=0),
            np.nanquantile(arr, 1 - quantile, axis=0))


def scan_imputed_outliers(prepared: Path, result: dict, target_cols: Sequence[str],
                          bounds: Optional[Tuple[np.ndarray, np.ndarray]],
                          policy: OutlierPolicy) -> Optional[dict]:
    """Report — and with ``policy.clip``, fix — out-of-band *imputed* cells.

    Returns ``{clipped, splits: {split: {...}}, per_column: {...}}`` or None when
    no band is available. When clipping, the imputed CSV is rewritten in place so
    the scoring and final-dataset stages downstream see the corrected values.
    """
    if bounds is None:
        return None
    lo, hi = bounds
    # The bundle stores bounds as float32 and the imputed frames round-trip
    # through CSV, so a cell clipped exactly TO a bound reads back ~1e-14 outside
    # it. Without this tolerance a second run re-flags (and re-"clips") cells the
    # first run already fixed, and the stage never reaches a fixed point.
    tol = np.maximum(np.maximum(np.abs(lo), np.abs(hi)), 1.0) * 1e-9
    tcols = list(target_cols)
    report: dict = {"clipped": policy.clip, "splits": {}, "per_column": {},
                    "total_cells": 0, "total_flagged": 0, "negative_flagged": 0}

    for kind, info in result.get("files", {}).items():
        src_path = prepared / INPUT_FILES[kind]
        out_path = Path(info["path"])
        if not src_path.exists() or not out_path.exists():
            continue
        gappy = pd.read_csv(src_path)
        imputed = pd.read_csv(out_path)
        filled = gappy[tcols].isna().to_numpy()
        arr = imputed[tcols].to_numpy(float)

        oob = filled & ~np.isnan(arr) & ((arr < lo - tol) | (arr > hi + tol))
        n_oob = int(oob.sum())
        n_neg = int((filled & ~np.isnan(arr) & (arr < 0) & (lo >= 0)).sum())

        if policy.clip and n_oob:
            arr = np.where(oob, np.clip(arr, lo, hi), arr)
            # Write back only the columns that actually changed: reassigning every
            # target column would re-serialize (and could re-type) columns this
            # stage never touched.
            for j, col in enumerate(tcols):
                if oob[:, j].any():
                    imputed[col] = arr[:, j]
            imputed.to_csv(out_path, index=False)

        report["splits"][kind] = {
            "imputed_cells": int(filled.sum()),
            "out_of_band": n_oob,
            "negative": n_neg,
            "rewritten": n_oob if policy.clip else 0,
        }
        report["total_cells"] += int(filled.sum())
        report["total_flagged"] += n_oob
        report["negative_flagged"] += n_neg
        for j, col in enumerate(tcols):
            if int(oob[:, j].sum()):
                entry = report["per_column"].setdefault(
                    col, {"cells": 0, "lower": float(lo[j]), "upper": float(hi[j])})
                entry["cells"] += int(oob[:, j].sum())
    return report


def mismatch_reason(df: pd.DataFrame, target_cols: Sequence[str],
                    n_rows: int) -> str | None:
    """Why ``df`` cannot be scored on the current bundle, or None if it can.

    A frame produced against an older bundle either lacks a current target column
    (a units rename: ``ram_usage`` vs ``ram_usage_mb``) or carries a different row
    count. Scoring the first silently drops columns and reports an artificially
    low MAE on a smaller n_cells; the second cannot be scored at all. Both are
    caught here so a stale file never reaches the leaderboard.
    """
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        preview = ", ".join(missing[:5]) + (" …" if len(missing) > 5 else "")
        return f"missing {len(missing)}/{len(target_cols)} target cols ({preview})"
    if len(df) != n_rows:
        return f"{len(df)} rows, but this bundle has {n_rows}"
    return None


def collect_result(spec: MethodSpec, prepared: Path, generated: Path,
                   splits: Sequence[str], target_cols: Sequence[str],
                   engine: str) -> dict:
    """Build an :func:`impute_bundle`-shaped result from files already on disk."""
    files: dict[str, dict] = {}
    for kind in splits:
        src = prepared / INPUT_FILES[kind]
        out_path = generated / output_name(spec.lib, spec.method, kind)
        if not src.exists() or not out_path.exists():
            continue
        before = pd.read_csv(src)
        after = pd.read_csv(out_path)
        reason = mismatch_reason(after, target_cols, len(before))
        if reason:
            raise ValueError(f"{out_path.name} does not match the bundle: {reason}")
        nan_before = int(before[list(target_cols)].isna().sum().sum())
        nan_after = int(after[list(target_cols)].isna().sum().sum())
        files[kind] = {
            "path": str(out_path),
            "rows": int(len(after)),
            "nan_before": nan_before,
            "nan_after": nan_after,
            "filled": nan_before - nan_after,
        }
    if not files:
        raise FileNotFoundError(
            f"{spec.label} produced no output under {generated} "
            f"(expected {output_name(spec.lib, spec.method, splits[0])})"
        )
    return {
        "method": spec.method,
        "lib": spec.lib,
        "engine": engine,
        "output_dir": str(generated),
        "target_cols": list(target_cols),
        "files": files,
    }


def stale_outputs(prepared: Path, generated: Path,
                  target_cols: Sequence[str]) -> list[tuple[Path, str]]:
    """Imputed CSVs in ``generated`` that were produced against another bundle.

    Re-running the cleaning stage can change the regularized row count (a raw-data
    edit, a timeline-config change), which silently invalidates every previously
    generated file. Reporting them is what keeps the leaderboard comparable.
    """
    test_input = prepared / INPUT_FILES["test"]
    if not generated.exists() or not test_input.exists():
        return []
    n_rows = len(pd.read_csv(test_input))
    stale: list[tuple[Path, str]] = []
    for path in sorted(generated.glob("*_test_imputed.csv")):
        try:
            reason = mismatch_reason(pd.read_csv(path), target_cols, n_rows)
        except Exception as exc:  # noqa: BLE001 - unreadable is stale enough
            reason = f"unreadable ({exc})"
        if reason:
            stale.append((path, reason))
    return stale


def prune_stale(generated: Path, paths: Iterable[Path]) -> list[Path]:
    """Delete a stale ``*_test_imputed.csv`` plus its train/final siblings."""
    removed: list[Path] = []
    for test_path in paths:
        key = test_path.name.removesuffix("_test_imputed.csv")
        for sibling in (test_path,
                        generated / f"{key}_train_imputed.csv",
                        generated / f"{key}_final.csv"):
            if sibling.exists():
                sibling.unlink()
                removed.append(sibling)
    return removed


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_clean(paths: DatasetPaths, cfg: dict, *, timestamp_col: str | None = None) -> dict:
    """Stages 1–2: raw → remediated → regularized bundle (via minimal_dataops)."""
    if not paths.raw_csv.exists():
        raise FileNotFoundError(f"raw CSV not found: {paths.raw_csv}")
    paths.report_json.parent.mkdir(parents=True, exist_ok=True)
    return run_pipeline(
        str(paths.raw_csv),
        str(paths.remediated_csv),
        str(paths.report_json),
        timestamp_col=timestamp_col if timestamp_col is not None else cfg.get("timestamp_col"),
        validation_config=cfg.get("validation", {}),
        imputation_config=cfg.get("imputation", {}),
        timeline_config=cfg.get("timeline", {}),
        soft_cleaned_csv=cfg.get("soft_cleaned_output") or cfg.get("cleaned_output"),
    )


def stage_impute(spec: MethodSpec, prepared: Path, generated: Path,
                 splits: Sequence[str], target_cols: Sequence[str],
                 opts: RunnerOptions) -> dict:
    """Stage 3: run one method and return its :func:`impute_bundle`-shaped result."""
    if spec.is_builtin:
        result = impute_bundle(prepared, method=spec.method, lib=spec.lib,
                               output_dir=generated, inputs=list(splits), engine="pandas")
        # Re-read from disk so built-in and runner results are validated alike.
        return collect_result(spec, prepared, generated, splits, target_cols,
                              engine=result["engine"])

    code, log = _run_app_runner(spec, prepared, generated, splits, opts)
    if code != 0:
        raise RuntimeError(f"{spec.label} runner exited with code {code}\n{log[-2000:]}")
    try:
        return collect_result(spec, prepared, generated, splits, target_cols, engine="runner")
    except (FileNotFoundError, ValueError) as exc:
        # A runner that exits 0 without usable output (an uninstalled optional
        # dependency reported as a warning, say) is still a failed method — but
        # the reason is only in its log, so carry the tail into the error.
        raise RuntimeError(f"{exc}\n--- {spec.label} runner log (tail) ---\n"
                           f"{log[-1500:].strip()}") from exc


def stage_finalize(spec: MethodSpec, prepared: Path, generated: Path,
                   result: dict, *, output_path: Path | None = None) -> dict | None:
    """Stage 5: stitch a method's imputed train+test into one gap-free CSV."""
    if not {"train", "test"} <= set(result["files"]):
        LOGGER.info("skip final for %s: needs both splits, have %s",
                    spec.label, sorted(result["files"]))
        return None
    return build_final_dataset(
        prepared,
        method=spec.method,
        lib=spec.lib,
        output_path=output_path or (generated / f"{spec.key}_final.csv"),
        # bundle_result wins over the <lib>_<method> name guess, which is what
        # makes this work for the WaveStitch+ v2/harpoon filenames.
        bundle_result=result,
        imputed_dir=generated,
    )


def build_leaderboard(runs: list[dict]) -> list[dict]:
    """Stage 4: rank every method on the shared test holdout (pooled MAE/RMSE)."""
    rows: list[dict] = []
    for run in runs:
        acc = (((run.get("comparison") or {}).get("splits", {}).get("test") or {})
               .get("accuracy") or {})
        pooled = acc.get("pooled") or {}
        mae = pooled.get("MAE")
        # NaN would sort into an arbitrary position and read as a valid rank, so
        # a method that could not be scored is left off the board entirely.
        if mae is None or not np.isfinite(mae):
            continue
        rows.append({
            "method": run["key"],
            # The runner's own --method token, kept beside the file-stem key so a
            # consumer can rebuild its own label (the dashboard keys on lib/method).
            "method_name": run["imputation"]["method"],
            "lib": run["imputation"]["lib"],
            "MAE": pooled.get("MAE"),
            "RMSE": pooled.get("RMSE"),
            "MAPE_%": pooled.get("MAPE_%"),
            "eval_cells": acc.get("eval_cells", 0),
        })
    return sorted(rows, key=lambda r: r["MAE"])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_dataset(paths: DatasetPaths, specs: Sequence[MethodSpec], cfg: dict, *,
                opts: RunnerOptions | None = None,
                splits: Sequence[str] = ("train", "test"),
                timestamp_col: str | None = None,
                skip_clean: bool = False,
                skip_existing: bool = False,
                canonical: MethodSpec | None = None,
                prune: bool = False,
                strict: bool = False,
                outlier_policy: OutlierPolicy | None = None) -> dict:
    """Run the complete pipeline for one dataset and return the run manifest."""
    opts = opts or RunnerOptions()
    outlier_policy = outlier_policy or OutlierPolicy()
    started = time.time()
    failures: list[dict] = []

    # ---- stages 1-2: clean + regularize ----------------------------------
    if skip_clean and paths.report_json.exists():
        report = json.loads(paths.report_json.read_text(encoding="utf-8"))
        LOGGER.info("[%s] reusing existing report %s", paths.name, paths.report_json)
    else:
        LOGGER.info("[%s] stage 1-2: cleaning %s", paths.name, paths.raw_csv)
        report = stage_clean(paths, cfg, timestamp_col=timestamp_col)

    handoff = report.get("handoff") or {}
    if not handoff.get("needs_ts_imputation"):
        LOGGER.info("[%s] no time-series imputation needed (%s)",
                    paths.name, handoff.get("reason"))
        return {"dataset": paths.name, "report_path": str(paths.report_json),
                "skipped": handoff.get("reason", "no_imputation_needed"),
                "runs": [], "leaderboard": [], "failures": [],
                "seconds": round(time.time() - started, 1)}

    prepared_raw = handoff.get("prepared_dir")
    if not prepared_raw:
        raise RuntimeError(f"[{paths.name}] handoff has no prepared_dir "
                           f"(bundle_error: {handoff.get('bundle_error')})")
    prepared = _abs(Path(prepared_raw))
    if not prepared.exists():
        raise FileNotFoundError(f"[{paths.name}] prepared bundle missing: {prepared}")
    generated = generated_dir_for(prepared)
    generated.mkdir(parents=True, exist_ok=True)
    target_cols = json.loads((prepared / "meta.json").read_text())["target_cols"]
    bounds = load_target_bounds(prepared, target_cols, outlier_policy.quantile)
    if bounds is None:
        LOGGER.warning("[%s] no plausible-value band available — skipping the "
                       "post-imputation outlier stage", paths.name)

    # Outputs from a previous bundle cannot be scored against this one.
    stale = stale_outputs(prepared, generated, target_cols)
    if stale:
        LOGGER.warning("[%s] %d imputed file(s) predate this bundle:",
                       paths.name, len(stale))
        for path, reason in stale:
            LOGGER.warning("    %s — %s", path.name, reason)
        if prune:
            removed = prune_stale(generated, [p for p, _ in stale])
            LOGGER.warning("[%s] pruned %d stale file(s)", paths.name, len(removed))

    # ---- stages 3-5: impute, score, finalize ------------------------------
    runs: list[dict] = []
    for spec in specs:
        test_out = generated / output_name(spec.lib, spec.method, "test")
        if skip_existing and test_out.exists():
            LOGGER.info("[%s] %-34s skip (output exists)", paths.name, spec.label)
            try:
                result = collect_result(spec, prepared, generated, splits,
                                        target_cols, engine="existing")
            except Exception as exc:  # noqa: BLE001 - a stale reuse is a failure
                LOGGER.error("[%s] %s cannot reuse existing output: %s",
                             paths.name, spec.label, exc)
                failures.append({"method": spec.label, "stage": "reuse", "error": str(exc)})
                if strict:
                    raise
                continue
        else:
            t0 = time.time()
            LOGGER.info("[%s] %-34s %s", paths.name, spec.label,
                        "built-in" if spec.is_builtin else "runner")
            try:
                result = stage_impute(spec, prepared, generated, splits, target_cols, opts)
            except Exception as exc:  # noqa: BLE001 - one method must not sink the run
                LOGGER.error("[%s] %s FAILED: %s", paths.name, spec.label, exc)
                failures.append({"method": spec.label, "stage": "impute", "error": str(exc)})
                if strict:
                    raise
                continue
            LOGGER.info("[%s] %-34s ok (%.1fs)", paths.name, spec.label, time.time() - t0)

        try:
            # Stage 4a — outliers in the FILLED cells. Runs before scoring and
            # before the final is stitched, so both see the corrected values when
            # clipping is on (and the leaderboard reflects the fix).
            outliers = scan_imputed_outliers(prepared, result, target_cols,
                                             bounds, outlier_policy)
            if outliers and outliers["total_flagged"]:
                verb = "clipped" if outlier_policy.clip else "flagged (kept)"
                LOGGER.warning(
                    "[%s] %-34s %d/%d imputed cell(s) outside the observed band "
                    "%s%s", paths.name, spec.label, outliers["total_flagged"],
                    outliers["total_cells"], verb,
                    f", {outliers['negative_flagged']} negative"
                    if outliers["negative_flagged"] else "")
            comparison = compare_clean_vs_imputed(prepared, result)
            final = stage_finalize(spec, prepared, generated, result)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("[%s] %s scoring/final FAILED: %s", paths.name, spec.label, exc)
            failures.append({"method": spec.label, "stage": "score", "error": str(exc)})
            if strict:
                raise
            continue

        runs.append({"key": spec.key, "spec": spec.label,
                     "imputation": result, "comparison": comparison,
                     "outliers": outliers, "final_dataset": final})

    leaderboard = build_leaderboard(runs)

    # A file that was stale going in is no longer stale if this run regenerated it.
    produced = {run["key"] for run in runs}
    stale = [(path, reason) for path, reason in stale
             if path.name.removesuffix("_test_imputed.csv") not in produced]

    # ---- the canonical final: <name>_final.csv ----------------------------
    canonical_run = _pick_canonical(runs, canonical, leaderboard)
    canonical_final = None
    if canonical_run is not None:
        spec = parse_spec(canonical_run["spec"])
        canonical_final = stage_finalize(spec, prepared, generated,
                                         canonical_run["imputation"],
                                         output_path=paths.final_csv)
        LOGGER.info("[%s] canonical final ← %s → %s",
                    paths.name, canonical_run["spec"], paths.final_csv)

    # ---- stage 6: the compare report --------------------------------------
    manifest = {
        "dataset": paths.name,
        "report_path": str(paths.report_json),
        "prepared_dir": str(prepared),
        "generated_dir": str(generated),
        "target_cols": target_cols,
        "stale_inputs": [{"file": p.name, "reason": r} for p, r in stale],
        "runs": runs,
        "leaderboard": leaderboard,
        "outlier_handling": {
            "clip_imputed_outliers": outlier_policy.clip,
            "band": "bundle scaler bounds" if bounds is not None else None,
            "flagged_by_method": {r["key"]: (r["outliers"] or {}).get("total_flagged", 0)
                                  for r in runs},
        },
        "failures": failures,
        "seconds": round(time.time() - started, 1),
        # Flat views kept for scripts/auto_impute.py compatibility.
        "imputations": [r["imputation"] for r in runs],
        "comparisons": [r["comparison"] for r in runs],
        "final_datasets": [r["final_dataset"] for r in runs if r["final_dataset"]],
        "canonical_final_dataset": canonical_final,
    }
    if canonical_run is not None:
        # The dashboard reads these three top-level keys to locate the generated
        # dir and the final CSV; promoting the canonical run keeps it working
        # on a multi-method manifest (auto_impute's multi-run shape omits them).
        manifest["imputation"] = canonical_run["imputation"]
        manifest["comparison"] = canonical_run["comparison"]
        manifest["final_dataset"] = canonical_final or canonical_run["final_dataset"]

    compare_path = paths.report_json.with_name(
        paths.report_json.stem.removesuffix("_report") + "_imputation_compare.json")
    compare_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n",
                            encoding="utf-8")
    manifest["compare_path"] = str(compare_path)
    return manifest


def _pick_canonical(runs: list[dict], canonical: MethodSpec | None,
                    leaderboard: list[dict]) -> dict | None:
    """The run that becomes ``<name>_final.csv``: the requested one, else best MAE."""
    by_key = {r["key"]: r for r in runs if r["final_dataset"]}
    if not by_key:
        return None
    if canonical is not None and canonical.key in by_key:
        return by_key[canonical.key]
    if canonical is not None:
        LOGGER.warning("canonical method %s is not among this run's finals; "
                       "falling back to the best holdout MAE", canonical.label)
    for row in leaderboard:
        if row["method"] in by_key:
            return by_key[row["method"]]
    return next(iter(by_key.values()))


def canonical_from_config(cfg: dict) -> MethodSpec | None:
    """The ``imputation: {app, method}`` selection in config/dataops.yaml, if usable."""
    sel = cfg.get("imputation") or {}
    app, method = sel.get("app"), sel.get("method")
    if not app or not method:
        return None
    lib = APP_TO_LIB.get(str(app).strip().lower())
    if lib is None:
        return None
    if lib == "wavestitchplus" and method in ("v2", "anchored", "tuned"):
        return MethodSpec("wavestitchplus_v2", "anchored" if method == "v2" else method)
    if lib == "wavestitchplus" and method == "harpoon":
        return MethodSpec("wavestitchplus_harpoon", "harpoon")
    return MethodSpec(lib, method)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(manifest: dict) -> None:
    name = manifest["dataset"]
    print(f"\n{'=' * 78}\n== {name}  ({manifest['seconds']}s)\n{'=' * 78}")
    if manifest.get("skipped"):
        print(f"  no imputation needed ({manifest['skipped']})")
        return

    print(f"  bundle    {manifest['prepared_dir']}")
    print(f"  generated {manifest['generated_dir']}")

    board = manifest["leaderboard"]
    if board:
        table = pd.DataFrame(board)[["method", "MAE", "RMSE", "MAPE_%", "eval_cells"]]
        print("\n  holdout leaderboard (pooled over target columns, lower is better):")
        print("\n".join("    " + line for line in
                        table.to_string(index=False,
                                        float_format=lambda x: f"{x:.4g}").splitlines()))
        counts = {row["eval_cells"] for row in board}
        if len(counts) > 1:
            print(f"    ! methods scored on differing eval-cell counts {sorted(counts)} — "
                  "not directly comparable")

    oh = manifest.get("outlier_handling") or {}
    flagged = {k: v for k, v in (oh.get("flagged_by_method") or {}).items() if v}
    if flagged:
        verb = "clipped" if oh.get("clip_imputed_outliers") else "flagged, kept"
        print(f"\n  imputed cells outside the observed band ({verb}):")
        for key, n in sorted(flagged.items(), key=lambda kv: -kv[1])[:8]:
            print(f"      {key:28s} {n:,}")
        if not oh.get("clip_imputed_outliers"):
            print("      → pass --clip-imputed-outliers to clip them to the band")

    finals = [r for r in manifest["runs"] if r["final_dataset"]]
    print(f"\n  finals written: {len(finals)}/{len(manifest['runs'])}")
    canonical = manifest.get("canonical_final_dataset")
    if canonical:
        filled = canonical["gaps_before"] - canonical["gaps_after"]
        print(f"  ★ FINAL → {canonical['path']}")
        print(f"    {canonical['rows']:,} rows · {len(canonical['columns'])} cols · "
              f"gaps filled {filled:,}/{canonical['gaps_before']:,} "
              f"(residual {canonical['gaps_after']:,})")

    if manifest["stale_inputs"]:
        print(f"\n  ! {len(manifest['stale_inputs'])} pre-existing file(s) predate this "
              "bundle (re-run them, or pass --prune-stale):")
        for item in manifest["stale_inputs"]:
            print(f"      - {item['file']}: {item['reason']}")

    if manifest["failures"]:
        print(f"\n  ! {len(manifest['failures'])} method(s) failed:")
        for f in manifest["failures"]:
            first = f["error"].strip().splitlines()[0] if f["error"].strip() else "?"
            print(f"      - {f['method']} ({f['stage']}): {first[:150]}")

    print(f"\n  compare report → {manifest.get('compare_path')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="presets for --methods: " + ", ".join(PRESETS),
    )
    p.add_argument("--config", default="config/dataops.yaml",
                   help="YAML config supplying validation/timeline/imputation blocks")
    p.add_argument("--dataset", default=None,
                   help="dataset name(s), comma-separated; paths follow the "
                        "data/raw/<name>.csv convention. Default: the --config dataset.")
    p.add_argument("--methods", default="builtin",
                   help="comma-separated presets and/or <lib>/<method> specs "
                        f"(presets: {', '.join(PRESETS)})")
    p.add_argument("--inputs", nargs="+", default=["train", "test"],
                   choices=["train", "test"], help="which splits to impute")
    p.add_argument("--canonical", default=None,
                   help="method whose final becomes <name>_final.csv "
                        "(default: the config's imputation selection, else best MAE)")
    p.add_argument("--timestamp-col", default=None)
    p.add_argument("--clip-outliers", action="store_true",
                   help="CLEANING stage: winsorize flagged outliers in the raw data "
                        "to the [q, 1-q] band. Off by default: outliers are detected "
                        "and reported either way, but clipping rewrites real "
                        "measurements.")
    p.add_argument("--clip-imputed-outliers", action="store_true",
                   help="POST-IMPUTATION stage: clip *filled* cells that land outside "
                        "the observed plausible band (e.g. darts cubic's negative "
                        "latencies). Off by default — reported either way. Observed "
                        "cells are never modified.")
    p.add_argument("--outlier-band-quantile", type=float, default=0.005,
                   help="fallback band quantile when the bundle stores no bounds "
                        "(default 0.005 → the [0.5%%, 99.5%%] observed range)")
    p.add_argument("--skip-clean", action="store_true",
                   help="reuse the existing report + bundle instead of re-cleaning")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip a method whose test output is already on disk")
    p.add_argument("--prune-stale", action="store_true",
                   help="delete imputed files produced against an older bundle")
    p.add_argument("--strict", action="store_true",
                   help="abort on the first method failure instead of continuing")
    p.add_argument("--log-file", default=None)
    p.add_argument("--verbose", "-v", action="store_true")

    g = p.add_argument_group("runner hyperparameters")
    g.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    g.add_argument("--fast", action="store_true",
                   help="WaveStitch+ smoke hyperparams (em=1, epochs/em=5, ddim=5)")
    g.add_argument("--em-iterations", type=int, default=3)
    g.add_argument("--epochs-per-em", type=int, default=50)
    g.add_argument("--ddim-steps", type=int, default=30)
    g.add_argument("--repaint-rounds", type=int, default=None)
    g.add_argument("--pypots-epochs", type=int, default=50)
    g.add_argument("--pypots-window", type=int, default=100)
    g.add_argument("--pypots-batch-size", type=int, default=32)
    g.add_argument("--timeout", type=float, default=None,
                   help="per-method runner timeout in seconds")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    # --log-file wins, then the config's log_file — same precedence as
    # minimal_dataops.main, so the raw-data quality lines land in the same file
    # whichever entrypoint ran the cleaning stage.
    configure_logging(args.log_file if args.log_file is not None else cfg.get("log_file"))
    logging.getLogger("dataops").setLevel(logging.DEBUG if args.verbose else logging.INFO)

    if args.clip_outliers:
        cfg.setdefault("validation", {})["clip_outliers"] = True
    datasets = ([DatasetPaths.from_name(n.strip())
                 for n in args.dataset.split(",") if n.strip()]
                if args.dataset else [DatasetPaths.from_config(cfg)])

    try:
        specs = resolve_specs(args.methods.split(","))
        canonical = parse_spec(args.canonical) if args.canonical else canonical_from_config(cfg)
    except (ValueError, KeyError) as exc:
        print(f"bad --methods/--canonical: {exc}", file=sys.stderr)
        return 2
    if not specs:
        print("no methods selected", file=sys.stderr)
        return 2

    outlier_policy = OutlierPolicy(clip=args.clip_imputed_outliers,
                                   quantile=args.outlier_band_quantile)
    opts = RunnerOptions(
        device=args.device, fast=args.fast,
        em_iterations=args.em_iterations, epochs_per_em=args.epochs_per_em,
        ddim_steps=args.ddim_steps, repaint_rounds=args.repaint_rounds,
        pypots_epochs=args.pypots_epochs, pypots_window=args.pypots_window,
        pypots_batch_size=args.pypots_batch_size, timeout=args.timeout,
    )

    print(f"datasets: {', '.join(d.name for d in datasets)}")
    print(f"methods ({len(specs)}): {', '.join(s.label for s in specs)}")
    if canonical:
        print(f"canonical final: {canonical.label}")

    manifests: list[dict] = []
    broken: list[str] = []
    for paths in datasets:
        try:
            manifest = run_dataset(
                paths, specs, cfg, opts=opts, splits=args.inputs,
                timestamp_col=args.timestamp_col, skip_clean=args.skip_clean,
                skip_existing=args.skip_existing, canonical=canonical,
                prune=args.prune_stale, strict=args.strict,
                outlier_policy=outlier_policy,
            )
        except Exception as exc:  # noqa: BLE001 - one dataset must not sink the rest
            LOGGER.exception("[%s] pipeline failed", paths.name)
            print(f"\n!! {paths.name}: {exc}", file=sys.stderr)
            broken.append(paths.name)
            if args.strict:
                return 1
            continue
        manifests.append(manifest)
        print_summary(manifest)

    failed_methods = sum(len(m["failures"]) for m in manifests)
    print(f"\n{'=' * 78}")
    print(f"== {len(manifests)}/{len(datasets)} dataset(s) completed"
          + (f", {failed_methods} method failure(s)" if failed_methods else "")
          + (f", {len(broken)} dataset failure(s)" if broken else ""))
    for name in broken:
        print(f"  - {name}: pipeline failed (see log)")
    if broken:
        return 1
    return 1 if (failed_methods and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
