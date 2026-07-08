"""
Streamlit dashboard for comparing time-series imputation methods.

Discovers `prepared_<subset>/` and `generated_<subset>/` folders under
`experiments/<group>/`, lets you pick a subset / split / methods /
feature, and plots:

  * original observed values  (solid markers, blue)
  * masked-for-eval positions (where ground truth is known)
  * truly-missing positions   (no ground truth available)
  * imputed values per method (one color per method)

Tabs:
  1. Time series   — interactive per-feature comparison plot
  2. Metrics       — MAE / RMSE / MAPE / fill-rate per method
  3. Distribution  — histogram per method (observed-only vs imputed-only)
  4. Run experiment — invoke any of the three runners as a subprocess

Launch from the repo root:

    conda activate myenv
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# Canonical results live under data/processed: the pipeline writes each stage as
# <name>_{soft_cleaned,remediated,regularized/,generated/,final}, and every method's
# outputs + finals land in <name>_generated/. The dashboard reads this one place.
DEFAULT_WORK_ROOT = REPO_ROOT / "data" / "processed"

RUNNERS = {
    "darts": REPO_ROOT / "dockers" / "tools" / "Darts_app" / "run_imputation.py",
    "imputegap": REPO_ROOT / "dockers" / "tools" / "ImputeGAP_app" / "run_imputation.py",
    "pypots": REPO_ROOT / "dockers" / "tools" / "PyPOTS_app" / "run_imputation.py",
    "wavestitchplus": REPO_ROOT / "dockers" / "tools" / "WaveStitchPlus_app" / "run_imputation.py",
    "wavestitchplus_v2": REPO_ROOT / "dockers" / "tools" / "WaveStitchPlus_app" / "run_imputation_v2.py",
    "wavestitchplus_harpoon": REPO_ROOT / "dockers" / "tools" / "WaveStitchPlus_app" / "run_imputation_harpoon.py",
}

# Method palettes shown in the "Run experiment" form.
RUNNER_METHODS = {
    "darts": ["auto", "linear", "quadratic", "cubic", "nearest", "slinear", "zero", "kalman"],
    "imputegap": [
        "mean", "mean_by_series", "min", "zero", "interpolation", "knn",
        "cdrec", "iterative_svd", "soft_impute", "svt",
        "iim", "mice", "miss_forest", "xgboost",
        "brits", "mrnn", "gain",
    ],
    "pypots": ["saits", "brits", "transformer", "gpvae", "mrnn", "csdi", "usgan", "timesnet"],
    "wavestitchplus": ["v1", "em", "standard"],
    "wavestitchplus_v2": ["anchored", "tuned"],
    "wavestitchplus_harpoon": ["harpoon"],
}

RUNNER_IMPORT_CHECKS = {
    "darts": ("darts", "python -m pip install -r dockers/tools/Darts_app/requirements.txt"),
    "imputegap": ("imputegap", "python -m pip install -r dockers/tools/ImputeGAP_app/requirements.txt"),
    "pypots": ("pypots", "python -m pip install -r dockers/tools/PyPOTS_app/requirements.txt"),
    "wavestitchplus": ("torch", "python -m pip install -r dockers/tools/requirements.txt"),
    # wavestitchplus_v2 is NOT torch-gated: its anchoring is pure-python (wsp_v2),
    # so `--reuse-diffusion <v1 output>` runs torch-free. Only the (optional)
    # synthesis fallback needs torch, and that surfaces from the runner itself.
    "wavestitchplus_harpoon": ("torch", "python -m pip install -r dockers/tools/requirements.txt"),
}

# Prebuilt Docker images per app (tags from each app's build_image.sh). Override
# any with ``DATAOPS_IMPUTE_IMAGE_<LIB>``. Running a method in its image means the
# dashboard env needs no heavy deps at all.
RUNNER_IMAGES = {
    "darts": "darts-baseline:latest",
    "imputegap": "imputegap-baseline:latest",
    "pypots": "pypots-baseline:latest",
    # CPU default so it runs without nvidia-docker (e.g. on a Mac). Override with
    # DATAOPS_IMPUTE_IMAGE_WAVESTITCHPLUS=wavestitchplus-gpu:latest where a GPU exists.
    "wavestitchplus": "wavestitchplus-cpu:latest",
    "wavestitchplus_v2": "wavestitchplus-cpu:latest",
    "wavestitchplus_harpoon": "wavestitchplus-cpu:latest",
}
# The WaveStitch+ image's CMD is run_pipeline.py, so the imputation runner is
# invoked by overriding the entrypoint with the in-image script path.
WSP_SCRIPTS = {
    "wavestitchplus": "/app/WaveStitchPlus_app/run_imputation.py",
    "wavestitchplus_v2": "/app/WaveStitchPlus_app/run_imputation_v2.py",
    "wavestitchplus_harpoon": "/app/WaveStitchPlus_app/run_imputation_harpoon.py",
}

# Human-readable description of what each heavy library actually needs — so the
# "not available" message is accurate per method, not just "needs <module>".
RUNNER_NEEDS = {
    "darts": "the `darts` library (only the `kalman` method; interpolation runs here built-in)",
    "imputegap": "the `imputegap` library (only the matrix/ML/deep methods; statistics run here built-in)",
    "pypots": "`pypots` + `torch` (trains a neural imputer; GPU recommended)",
    "wavestitchplus": "`torch`, and it retrains a diffusion model (slow; GPU recommended)",
    "wavestitchplus_harpoon": "`torch` + a pretrained WaveStitch+ model (inference-time guidance)",
}

INPUT_FILES = {"train": "train.csv", "test": "test_input.csv"}
GT_FILES = {"train": None, "test": "test_gt.csv"}

# Defaults for the "Pipeline run" tab — point at the SeaweedFS S3 endpoint that
# the Airflow stack publishes. Override per-session in the sidebar.
PIPELINE_S3_DEFAULTS = {
    "endpoint": os.environ.get("PIPELINE_S3_ENDPOINT", "http://localhost:8333"),
    "bucket":   os.environ.get("PIPELINE_S3_BUCKET",   "6gdali-lake2026"),
    "access":   os.environ.get("PIPELINE_S3_ACCESS",   "anykey"),
    "secret":   os.environ.get("PIPELINE_S3_SECRET",   "anysecret"),
}

# Filenames written by our runners (canonical):
#   <lib>_<method>_<split>_imputed.csv  e.g. wavestitchplus_v1_test_imputed.csv,
#                                            wavestitchplus_v2_train_imputed.csv
IMPUTED_NEW_RE = re.compile(r"^(?P<lib>[A-Za-z0-9]+)_(?P<method>[A-Za-z0-9_]+?)_(?P<split>train|test)_imputed\.csv$")
# Back-compat: very old WaveStitch+ outputs omitted the method token; discovered
# as plain ``wavestitchplus`` so historical files still surface in the dashboard.
IMPUTED_WS_SPLIT_RE = re.compile(r"^wavestitchplus_(?P<split>train|test)_imputed\.csv$", re.IGNORECASE)
# Legacy test-only mixed-case output (``wavestitchPlus_<method>_imputed.csv``)
# from the original container/dev scripts; discovered under library ``wavestitch+``.
IMPUTED_WS_RE = re.compile(r"^wavestitchPlus_(?P<method>[A-Za-z0-9_]+)_imputed\.csv$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data classes & discovery
# ---------------------------------------------------------------------------

@dataclass
class ImputedFile:
    library: str
    method: str          # empty for method-free default WaveStitch+ split outputs
    split: str          # "train" or "test"
    path: Path
    label: str          # "darts/linear", "imputegap/iim", "wavestitchplus", ...

    @property
    def key(self) -> str:
        return f"{self.library}/{self.method}" if self.method else self.library


@dataclass
class Subset:
    dataset: str       # e.g. "EUR"
    name: str          # e.g. "amf" (the suffix after `prepared_`)
    prepared_dir: Path
    generated_dir: Optional[Path]

    @property
    def label(self) -> str:
        return f"{self.dataset} / {self.name}"


@dataclass
class RawDataset:
    """A selectable raw CSV plus the DataOps runs generated from it."""
    name: str
    path: Path
    runs: List["DataOpsRun"]

    @property
    def label(self) -> str:
        suffix = f" · {len(self.runs)} run(s)" if self.runs else " · no runs yet"
        return f"{self.name}{suffix}"


def _generated_dir_for(prepared: Path) -> Path:
    """Sibling generated dir for a regularized/prepared bundle. Handles the
    stage-based ``<base>_regularized`` → ``<base>_generated`` layout and the
    legacy ``prepared_<base>`` / ``<base>_prepared`` → ``generated_...`` ones."""
    n = prepared.name
    if n.endswith("_regularized"):
        return prepared.parent / f"{n.removesuffix('_regularized')}_generated"
    if n.startswith("prepared_"):
        return prepared.parent / f"generated_{n.removeprefix('prepared_')}"
    return prepared.parent / f"generated_{n}"


# NOT @st.cache_data: returns script-defined Subset dataclasses, which st.cache_data
# cannot pickle (UnserializableReturnValueError). Directory scan is cheap.
def discover_subsets(work_root: Path) -> List[Subset]:
    """Discover prepared bundles under ``work_root`` (fallback for the run picker).

    Handles both naming conventions:
      * ``prepared_<name>`` + ``generated_<name>``         (experiments/EUR tree)
      * ``<name>_prepared`` + ``generated_<name>_prepared`` (pipeline output under
        ``data/processed``)

    Scans ``work_root`` itself and one level of sub-directories, so the default
    can point straight at ``data/processed``.
    """
    out: List[Subset] = []
    if not work_root.exists():
        return out
    candidates = [work_root] + sorted(p for p in work_root.iterdir() if p.is_dir())
    seen: set = set()
    for ds_dir in candidates:
        bundles = sorted({*ds_dir.glob("*regularized*"), *ds_dir.glob("*prepared*")})
        for prep in bundles:
            if prep in seen or not prep.is_dir() or not (prep / "meta.json").exists():
                continue
            seen.add(prep)
            if prep.name.endswith("_regularized"):        # stage-based layout
                name = prep.name.removesuffix("_regularized")
            elif prep.name.startswith("prepared_"):       # experiments/EUR layout
                name = prep.name.removeprefix("prepared_")
            else:                                         # legacy data/processed layout
                name = prep.name.removesuffix("_prepared")
            gen = _generated_dir_for(prep)
            out.append(Subset(
                dataset=ds_dir.name,
                name=name,
                prepared_dir=prep,
                generated_dir=gen,   # may not exist yet; a Run creates it here
            ))
    return out


@dataclass
class DataOpsRun:
    """One end-to-end DataOps run, discovered from a pipeline report JSON.

    Unifies the cleaning lineage (raw→soft-cleaned→remediated→regularized→final) and
    the imputation bundle (prepared/generated dirs) behind a single selectable
    run, so the dashboard no longer needs two separate data-source mental models.
    """
    name: str
    report_path: Optional[Path]
    report: Optional[dict]
    compare: Optional[dict]         # *_imputation_compare.json, if present
    raw_csv: Optional[Path]
    soft_cleaned_csv: Optional[Path]
    remediated_csv: Optional[Path]
    prepared_dir: Optional[Path]    # regularized bundle (drives the imputation views)
    generated_dir: Optional[Path]
    final_csv: Optional[Path]

    @property
    def data_type(self) -> str:
        rep = self.report or {}
        return (rep.get("data_type")
                or rep.get("profile", {}).get("data_type")
                or "unknown")

    def to_subset(self) -> Optional["Subset"]:
        """Expose the regularized bundle as a Subset for the imputation workbench."""
        if not self.prepared_dir or not (self.prepared_dir / "meta.json").exists():
            return None
        gen = self.generated_dir or _generated_dir_for(self.prepared_dir)
        return Subset(dataset=self.name, name="run", prepared_dir=self.prepared_dir,
                      generated_dir=gen)


@st.cache_data(show_spinner=False)
def discover_dataops_runs(reports_dir: Path, repo_root: Path) -> List[DataOpsRun]:
    """Discover DataOps runs from ``reports/*.json`` pipeline reports.

    A file counts as a pipeline report if it carries ``validation_comparison``.
    Paths inside the report (relative to the repo root) are resolved, and the
    sibling ``*_imputation_compare.json`` (final dataset + comparison) is attached.
    """
    out: List[DataOpsRun] = []
    if not reports_dir.exists():
        return out

    def _abs(p) -> Optional[Path]:
        if not p:
            return None
        pp = Path(p)
        return pp if pp.is_absolute() else repo_root / pp

    for jp in sorted(reports_dir.glob("*.json")):
        if jp.name.endswith("_imputation_compare.json"):
            continue
        try:
            rep = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rep, dict) or "validation_comparison" not in rep:
            continue
        cmp_path = jp.with_name(jp.stem + "_imputation_compare.json")
        compare = None
        if cmp_path.exists():
            try:
                compare = json.loads(cmp_path.read_text(encoding="utf-8"))
            except Exception:
                compare = None
        generated = final_csv = None
        if compare:
            generated = _abs((compare.get("imputation") or {}).get("output_dir"))
            final_csv = _abs((compare.get("final_dataset") or {}).get("path"))
        out.append(DataOpsRun(
            name=jp.stem,
            report_path=jp,
            report=rep,
            compare=compare,
            raw_csv=_abs(rep.get("input")),
            soft_cleaned_csv=_abs(rep.get("soft_cleaned_output") or rep.get("cleaned_output")),
            remediated_csv=_abs(rep.get("output")),
            prepared_dir=_abs((rep.get("handoff") or {}).get("prepared_dir")),
            generated_dir=generated,
            final_csv=final_csv,
        ))
    return out


def _path_key(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve())
    except Exception:
        return str(path.absolute())


def discover_raw_datasets(raw_dir: Path, runs: List[DataOpsRun]) -> List[RawDataset]:
    """Discover raw CSVs and attach all pipeline reports that used each one."""
    paths: dict[str, Path] = {}
    if raw_dir.exists():
        for path in sorted(raw_dir.glob("*.csv")):
            paths[_path_key(path)] = path
    for run in runs:
        if run.raw_csv:
            paths.setdefault(_path_key(run.raw_csv), run.raw_csv)

    out: List[RawDataset] = []
    for key, path in sorted(paths.items(), key=lambda kv: kv[1].name.lower()):
        linked = [r for r in runs if _path_key(r.raw_csv) == key]
        out.append(RawDataset(name=path.stem, path=path, runs=linked))
    return out


def _label_for(file: ImputedFile) -> str:
    # Compact label used in dropdowns / legends.
    return f"{file.library}/{file.method}" if file.method else file.library


def _produced_key(lib: str, method: str) -> str:
    """Mirror the discovery filename → key mapping so the Run tab can auto-select
    the method it just produced. The default WaveStitch+ ``full`` method now
    writes ``wavestitchplus_v1_<split>_imputed.csv`` (parallel to v2), so its
    discovered key is ``wavestitchplus/v1``. The v2 runner is registered under
    library ``wavestitchplus_v2`` but its file's filename-derived library is
    ``wavestitchplus`` with method ``v2`` → key ``wavestitchplus/v2``."""
    if lib == "wavestitchplus":
        return "wavestitchplus/v1" if method == "full" else f"wavestitchplus/{method}"
    if lib == "wavestitchplus_v2":
        return "wavestitchplus/v2"
    if lib == "wavestitchplus_harpoon":
        return "wavestitchplus/harpoon"
    return f"{lib}/{method}"


# NOT @st.cache_data: it pickles the return, and pickling the script-defined
# ImputedFile dataclass fails (UnserializableReturnValueError). The glob is cheap
# and staying uncached means newly-produced method outputs appear immediately.
def _discover_imputed_files_by_dir(generated_dir: Optional[Path], split: str) -> Dict[str, "ImputedFile"]:
    out: Dict[str, ImputedFile] = {}
    if not generated_dir or not generated_dir.exists():
        return out
    for path in sorted(generated_dir.glob("*_imputed.csv")):
        name = path.name
        m = IMPUTED_NEW_RE.match(name)
        if m and m.group("split") == split:
            f = ImputedFile(
                library=m.group("lib").lower(),
                method=m.group("method"),
                split=split,
                path=path,
                label="",
            )
            f.label = _label_for(f)
            out[f.key] = f
            continue
        mws_split = IMPUTED_WS_SPLIT_RE.match(name)
        if mws_split:
            # New-format WaveStitch+ split file. Only register it for its own
            # split; either way skip the legacy branch below, whose
            # case-insensitive regex would otherwise match these lowercase names
            # (e.g. discover the 1970-row train file under the test split).
            if mws_split.group("split").lower() == split:
                f = ImputedFile(
                    library="wavestitchplus",
                    method="",
                    split=split,
                    path=path,
                    label="",
                )
                f.label = _label_for(f)
                out[f.key] = f
            continue
        # Legacy WaveStitch+ test-only filename
        if split == "test":
            mws = IMPUTED_WS_RE.match(name)
            if mws:
                if (
                    mws.group("method").lower() == "full"
                    and (generated_dir / "wavestitchplus_test_imputed.csv").exists()
                ):
                    # Prefer the renamed split output when both current and
                    # legacy default WaveStitch+ test files are present.
                    continue
                f = ImputedFile(
                    library="wavestitch+",
                    method=mws.group("method"),
                    split="test",
                    path=path,
                    label="",
                )
                f.label = _label_for(f)
                out[f.key] = f
    return out


def discover_imputed_files(subset: Subset, split: str) -> Dict[str, ImputedFile]:
    """Public wrapper around the cached discoverer (keyed on the path, not the
    unhashable Subset dataclass)."""
    return _discover_imputed_files_by_dir(subset.generated_dir, split)


# Gap-free final datasets: ``wavestitchplus_<variant>_final.csv`` (imputed train +
# imputed test) plus the DataOps report's own ``*_final.csv``.
FINAL_WSP_RE = re.compile(r"^wavestitchplus_(?P<variant>.+)_final\.csv$")


def discover_final_files(generated_dir: Optional[Path],
                         run: Optional["DataOpsRun"] = None) -> Dict[str, Path]:
    """Map label → path for gap-free ``*_final.csv`` datasets found in the
    generated dir, plus the pipeline run's own final (auto_impute)."""
    out: Dict[str, Path] = {}
    if generated_dir and generated_dir.exists():
        for p in sorted(generated_dir.glob("*_final.csv")):
            m = FINAL_WSP_RE.match(p.name)
            label = f"wavestitchplus/{m.group('variant')}" if m else p.stem
            out[label] = p
    final_csv = getattr(run, "final_csv", None)
    if final_csv and Path(final_csv).exists():
        out.setdefault(f"pipeline · {Path(final_csv).name}", Path(final_csv))
    return out


@st.cache_data(show_spinner=False)
def _load_raw_timeline_cached(prepared_dir: Path, feature: str, time_col: str,
                              sig: Tuple[float, ...]
                              ) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """Concat the RAW gappy inputs (train.csv + test_input.csv) on time+feature so
    a final can be diffed against the cells it filled. Returns (df, train→test
    boundary time)."""
    frames: List[pd.DataFrame] = []
    boundary: Optional[float] = None
    for split in ("train", "test"):
        fp = prepared_dir / INPUT_FILES[split]
        if not fp.exists():
            continue
        try:
            d = pd.read_csv(fp, usecols=[time_col, feature])
        except (ValueError, KeyError):
            continue
        if split == "train" and len(d):
            boundary = float(pd.to_numeric(d[time_col], errors="coerce").max())
        frames.append(d)
    if not frames:
        return None, None
    raw = pd.concat(frames, ignore_index=True).drop_duplicates(subset=[time_col], keep="last")
    return raw, boundary


def load_raw_timeline(prepared_dir: Path, feature: str, time_col: str
                      ) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    sig = _file_sig(prepared_dir / INPUT_FILES["train"],
                    prepared_dir / INPUT_FILES["test"])
    return _load_raw_timeline_cached(prepared_dir, feature, time_col, sig)


load_raw_timeline.clear = _load_raw_timeline_cached.clear


def _file_sig(*paths: Path) -> Tuple[float, ...]:
    """Modification-time signature for cache keying. Passing this into a
    ``@st.cache_data`` function makes it reload when the file on disk changes
    (e.g. a pipeline re-run rewrites meta.json / the imputed CSVs) instead of
    serving a stale parse — the source of ``KeyError`` on renamed columns when
    the dashboard outlived a regeneration. Must be a non-underscore arg so
    Streamlit includes it in the cache key.
    """
    sig = []
    for p in paths:
        try:
            sig.append(Path(p).stat().st_mtime)
        except OSError:
            sig.append(0.0)
    return tuple(sig)


@st.cache_data(show_spinner=False)
def _load_meta_cached(prepared_dir: Path, sig: Tuple[float, ...]) -> dict:
    with (prepared_dir / "meta.json").open() as f:
        return json.load(f)


def load_meta(prepared_dir: Path) -> dict:
    return _load_meta_cached(prepared_dir, _file_sig(prepared_dir / "meta.json"))


# Forward the @st.cache_data ``.clear`` from the inner cached fn to the thin
# wrapper, so existing ``load_*.clear()`` cache-busting call sites keep working.
load_meta.clear = _load_meta_cached.clear


@st.cache_data(show_spinner=False)
def _load_csv_cached(path: Path, sig: Tuple[float, ...]) -> pd.DataFrame:
    return pd.read_csv(path)


def load_csv(path: Path) -> pd.DataFrame:
    return _load_csv_cached(path, _file_sig(path))


load_csv.clear = _load_csv_cached.clear


@st.cache_data(show_spinner=False)
def _load_csv_subset_cached(path: Path, columns: Tuple[str, ...],
                            sig: Tuple[float, ...]) -> pd.DataFrame:
    """Read only the columns we actually plot — much faster for wide CSVs."""
    try:
        return pd.read_csv(path, usecols=list(columns))
    except ValueError:
        # Fall back to full read if a requested column isn't in the file.
        df = pd.read_csv(path)
        keep = [c for c in columns if c in df.columns]
        return df[keep]


def load_csv_subset(path: Path, columns: Tuple[str, ...]) -> pd.DataFrame:
    return _load_csv_subset_cached(path, columns, _file_sig(path))


load_csv_subset.clear = _load_csv_subset_cached.clear


@st.cache_data(show_spinner=False)
def _load_holdout_mask_cached(prepared_dir: Path,
                              sig: Tuple[float, ...]) -> Optional[np.ndarray]:
    p = prepared_dir / "eval_holdout_mask.npy"
    return np.load(p) if p.exists() else None


def load_holdout_mask(prepared_dir: Path) -> Optional[np.ndarray]:
    return _load_holdout_mask_cached(
        prepared_dir, _file_sig(prepared_dir / "eval_holdout_mask.npy"))


load_holdout_mask.clear = _load_holdout_mask_cached.clear


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(input_arr: np.ndarray,
                    gt_arr: Optional[np.ndarray],
                    imputed_arr: np.ndarray) -> Dict[str, float]:
    """Score imputation quality on cells that were NaN in input but known in GT."""
    miss = np.isnan(input_arr)
    n_miss = int(miss.sum())
    # No missing cells → nothing to impute; report as vacuously fully filled
    # (1.0) rather than 0/0 collapsing to 0.0%.
    fill_rate = float(((~np.isnan(imputed_arr)) & miss).sum()) / n_miss if n_miss else 1.0
    metrics = {"missing_cells": n_miss, "fill_rate": fill_rate}
    if gt_arr is None:
        return metrics
    eval_mask = miss & ~np.isnan(gt_arr) & ~np.isnan(imputed_arr)
    n_eval = int(eval_mask.sum())
    metrics["eval_cells"] = n_eval
    if n_eval == 0:
        metrics.update({"MAE": np.nan, "RMSE": np.nan, "MAPE_%": np.nan})
        return metrics
    pred = imputed_arr[eval_mask]
    truth = gt_arr[eval_mask]
    err = pred - truth
    metrics["MAE"] = float(np.mean(np.abs(err)))
    metrics["RMSE"] = float(np.sqrt(np.mean(err ** 2)))
    denom = np.where(np.abs(truth) < 1e-9, 1e-9, np.abs(truth))
    metrics["MAPE_%"] = float(np.mean(np.abs(err / denom)) * 100)
    return metrics


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
]


def plot_feature_comparison(input_df: pd.DataFrame,
                            gt_df: Optional[pd.DataFrame],
                            imputed_dfs: Dict[str, pd.DataFrame],
                            feature: str,
                            time_col: str,
                            holdout_mask: Optional[np.ndarray] = None,
                            show_context_lines: bool = False,
                            max_bands: int = 30,
                            final_overlays: Optional[Dict[str, pd.DataFrame]] = None,
                            ) -> go.Figure:
    fig = go.Figure()

    # Render the time axis as wall-clock dates whenever the column is datetime
    # or epoch seconds/ms (shared, robust coercion — handles s and ms).
    if time_col in input_df.columns:
        x, x_label = _coerce_time_axis(input_df[time_col])
    else:
        x, x_label = pd.Series(input_df.index), "row index"

    input_vals = input_df[feature].to_numpy(dtype=float)
    miss_mask = np.isnan(input_vals)

    gt_vals = None
    if gt_df is not None and feature in gt_df.columns:
        gt_vals = gt_df[feature].to_numpy(dtype=float)

    # 1) Shade the "truly missing" regions (NaN in input, NaN in GT too)
    if gt_vals is not None:
        unknown_mask = miss_mask & np.isnan(gt_vals)
    else:
        unknown_mask = miss_mask
    _add_missing_bands(fig, x, unknown_mask, label="truly missing", max_bands=max_bands)

    # 2) Original observed values (solid blue with markers) — WebGL for speed.
    fig.add_trace(go.Scattergl(
        x=x, y=input_vals,
        mode="lines+markers",
        name="observed",
        connectgaps=False,
        line=dict(color="#1f3b73", width=2),
        marker=dict(size=5, color="#1f3b73"),
        hovertemplate="<b>observed</b><br>t=%{x}<br>y=%{y}<extra></extra>",
    ))

    # 3) Ground-truth values at masked-for-eval positions
    if gt_vals is not None:
        eval_mask = miss_mask & ~np.isnan(gt_vals)
        if eval_mask.any():
            fig.add_trace(go.Scattergl(
                x=np.asarray(x)[eval_mask],
                y=gt_vals[eval_mask],
                mode="markers",
                name="ground truth (eval)",
                marker=dict(symbol="diamond-open", size=10, color="#2ca02c",
                            line=dict(width=2, color="#2ca02c")),
                hovertemplate="<b>GT</b><br>t=%{x}<br>y=%{y}<extra></extra>",
            ))

    # 4) Imputed values per method — only at the previously-NaN positions.
    x_arr = np.asarray(x)
    for i, (key, df) in enumerate(imputed_dfs.items()):
        if feature not in df.columns:
            continue
        color = PALETTE[i % len(PALETTE)]
        ivals = df[feature].to_numpy(dtype=float)
        if len(ivals) != len(miss_mask):
            # Row count disagrees with the input split (e.g. a stray train-sized
            # file matched to the test split) — skip rather than crash on the
            # boolean index.
            st.warning(f"Skipping '{key}': {len(ivals)} rows ≠ {len(miss_mask)} "
                       f"input rows for split.")
            continue
        if miss_mask.any():
            fig.add_trace(go.Scattergl(
                x=x_arr[miss_mask],
                y=ivals[miss_mask],
                mode="markers",
                name=f"{key}",
                marker=dict(symbol="x", size=8, color=color,
                            line=dict(width=1, color=color)),
                hovertemplate=f"<b>{key}</b><br>t=%{{x}}<br>y=%{{y}}<extra></extra>",
            ))
        if show_context_lines:
            fig.add_trace(go.Scattergl(
                x=x, y=ivals,
                mode="lines",
                name=f"{key} (line)",
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.45,
                showlegend=False,
                hoverinfo="skip",
            ))

    # 5) Optional row-level holdout mask annotation (test split only)
    if holdout_mask is not None and len(holdout_mask) == len(input_vals):
        ho_idx = np.where(holdout_mask)[0]
        if ho_idx.size:
            fig.add_trace(go.Scattergl(
                x=x_arr[ho_idx],
                y=np.full(ho_idx.shape, np.nan),
                mode="markers",
                name="holdout rows",
                marker=dict(symbol="line-ns-open", size=14, color="#888"),
                showlegend=True, hoverinfo="skip", visible="legendonly",
            ))

    # 6) Final overlay(s): the gap-free final's values aligned to this split by
    #    time — a solid line + filled circles at the fill positions, so on the
    #    train split you can see the final carries this split's imputed values.
    if final_overlays:
        fin_colors = ["#111111", "#5b2c6f", "#0b5345", "#8c6d31"]
        for i, (key, fdf) in enumerate(final_overlays.items()):
            if feature not in fdf.columns or len(fdf) != len(miss_mask):
                continue
            fvals = fdf[feature].to_numpy(dtype=float)
            c = fin_colors[i % len(fin_colors)]
            fig.add_trace(go.Scattergl(
                x=x, y=fvals, mode="lines", name=f"{key} (final)",
                line=dict(color=c, width=1.5), opacity=0.8, connectgaps=False,
                hovertemplate=f"<b>{key} final</b><br>t=%{{x}}<br>y=%{{y}}<extra></extra>",
            ))
            if miss_mask.any():
                fig.add_trace(go.Scattergl(
                    x=x_arr[miss_mask], y=fvals[miss_mask], mode="markers",
                    name=f"{key} (final) · fill", marker=dict(symbol="circle", size=6, color=c),
                    showlegend=False,
                    hovertemplate=f"<b>{key} final fill</b><br>t=%{{x}}<br>y=%{{y}}<extra></extra>",
                ))

    fig.update_layout(
        height=520,
        margin=dict(l=40, r=20, t=40, b=40),
        title=f"{feature}  —  observed vs imputed",
        xaxis_title=x_label,
        yaxis_title=feature,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="closest",
        uirevision="ts-plot",
    )
    return fig


def plot_final_timeline(final_df: pd.DataFrame,
                        raw_df: Optional[pd.DataFrame],
                        feature: str,
                        time_col: str,
                        boundary_x: Optional[float] = None,
                        max_bands: int = 30) -> go.Figure:
    """Full stitched timeline for a FINAL dataset (imputed train + imputed test).

    Observed cells (kept from the raw input) draw as the connected series; cells
    that were NaN in the raw ``train.csv``/``test_input.csv`` draw as imputed-fill
    ×-markers over a "was gap" band. A dashed line marks the train→test boundary.
    """
    fig = go.Figure()
    fdf = (final_df.sort_values(time_col).reset_index(drop=True)
           if time_col in final_df.columns else final_df.reset_index(drop=True))
    if time_col in fdf.columns:
        x, x_label = _coerce_time_axis(fdf[time_col])
    else:
        x, x_label = pd.Series(fdf.index), "row index"
    x_arr = np.asarray(x)
    final_vals = fdf[feature].to_numpy(dtype=float)

    split_boundary_idx: Optional[int] = None
    split_boundary_next_idx: Optional[int] = None
    if "split" in fdf.columns:
        split_labels = fdf["split"].astype(str).str.lower()
        train_idx = np.where(split_labels.eq("train").to_numpy())[0]
        test_idx = np.where(split_labels.eq("test").to_numpy())[0]
        if train_idx.size and test_idx.size:
            split_boundary_idx = int(train_idx.max())
            later_test = test_idx[test_idx > split_boundary_idx]
            split_boundary_next_idx = int(later_test.min()) if later_test.size else int(test_idx.min())

    # "was gap" = NaN in the raw input at this time, present in the final.
    was_gap = np.zeros(len(fdf), dtype=bool)
    if raw_df is not None and feature in raw_df.columns and time_col in raw_df.columns:
        merged = fdf[[time_col]].merge(
            raw_df[[time_col, feature]].rename(columns={feature: "_raw"}),
            on=time_col, how="left")
        raw_vals = merged["_raw"].to_numpy(dtype=float)
        was_gap = np.isnan(raw_vals) & ~np.isnan(final_vals)
        _add_missing_bands(fig, x, np.isnan(raw_vals), label="was gap", max_bands=max_bands)

    # Observed series (break the line at the imputed cells so fills read as ×).
    obs_vals = final_vals.copy()
    obs_vals[was_gap] = np.nan
    fig.add_trace(go.Scattergl(
        x=x, y=obs_vals, mode="lines+markers", name="observed",
        connectgaps=False, line=dict(color="#1f3b73", width=2),
        marker=dict(size=4, color="#1f3b73"),
        hovertemplate="<b>observed</b><br>t=%{x}<br>y=%{y}<extra></extra>",
    ))
    if was_gap.any():
        fig.add_trace(go.Scattergl(
            x=x_arr[was_gap], y=final_vals[was_gap], mode="markers",
            name="imputed (final)",
            marker=dict(symbol="x", size=7, color="#d62728",
                        line=dict(width=1, color="#d62728")),
            hovertemplate="<b>imputed</b><br>t=%{x}<br>y=%{y}<extra></extra>",
        ))
    if boundary_x is not None or split_boundary_idx is not None:
        if split_boundary_idx is not None:
            bxv = x_arr[split_boundary_idx]
            if split_boundary_next_idx is not None:
                bxv = _axis_midpoint(x_arr[split_boundary_idx],
                                      x_arr[split_boundary_next_idx])
        else:
            bx, _ = _coerce_time_axis(pd.Series([boundary_x]))
            bxv = np.asarray(bx)[0]
            if time_col in fdf.columns:
                tnum = pd.to_numeric(fdf[time_col], errors="coerce").to_numpy()
                left = np.where(tnum <= float(boundary_x))[0]
                right = np.where(tnum > float(boundary_x))[0]
                if left.size and right.size:
                    bxv = _axis_midpoint(x_arr[int(left.max())],
                                          x_arr[int(right.min())])
        # add_shape/add_annotation avoid add_vline's annotation-center math, which
        # does (x0+x1)/2 and blows up on a datetime axis (datetime + datetime).
        # A prominent divider (not a faint hairline) so the train→test split reads
        # at a glance; colour is distinct from observed (blue) and imputed (red).
        fig.add_shape(type="line", x0=bxv, x1=bxv, y0=0, y1=1,
                      xref="x", yref="paper", layer="above",
                      line=dict(color="#7a3fb0", width=2, dash="dash"))
        finite_vals = final_vals[np.isfinite(final_vals)]
        if finite_vals.size:
            ymin, ymax = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))
            pad = max((ymax - ymin) * 0.02, 1e-9)
            yline = [ymin - pad, ymax + pad]
        else:
            yline = [0.0, 1.0]
        # Scattergl markers can visually dominate SVG/layout shapes in dense
        # plots. Add the divider as the last WebGL trace too, so it stays in the
        # same rendering layer as the data points.
        fig.add_trace(go.Scattergl(
            x=[bxv, bxv], y=yline, mode="lines", name="train/test split",
            line=dict(color="#7a3fb0", width=2, dash="dash"),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_annotation(x=bxv, y=1.0, xref="x", yref="paper", showarrow=False,
                           text="train ┆ test", xanchor="center", yanchor="bottom",
                           font=dict(size=11, color="#7a3fb0"),
                           bgcolor="rgba(255,255,255,0.65)")

    fig.update_layout(
        height=520, margin=dict(l=40, r=20, t=40, b=40),
        title=f"{feature}  —  final (imputed train + imputed test)",
        xaxis_title=x_label, yaxis_title=feature,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="closest", uirevision="final-plot",
    )
    return fig


def _axis_midpoint(left, right):
    """Midpoint for Plotly x-axis values; falls back to ``left`` for categories."""
    try:
        if pd.isna(left) or pd.isna(right):
            return left
    except (TypeError, ValueError):
        pass
    try:
        if isinstance(left, (pd.Timestamp, np.datetime64)) or isinstance(right, (pd.Timestamp, np.datetime64)):
            lts, rts = pd.Timestamp(left), pd.Timestamp(right)
            return lts + (rts - lts) / 2
    except (TypeError, ValueError, pd.errors.OutOfBoundsDatetime):
        pass
    try:
        return (float(left) + float(right)) / 2.0
    except (TypeError, ValueError):
        return left


def _add_missing_bands(fig: go.Figure, x, mask: np.ndarray, label: str = "",
                       max_bands: int = 30) -> None:
    """Draw faint gray rectangles over contiguous True runs in `mask`.
    Keeps at most `max_bands` largest runs so very-fragmented gap patterns
    don't drown Plotly in thousands of shapes."""
    if not mask.any():
        return
    # Vectorized run detection — much faster than the Python loop for long arrays.
    m = mask.astype(np.int8)
    diff = np.diff(np.concatenate(([0], m, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    runs = list(zip(starts.tolist(), ends.tolist()))
    if len(runs) > max_bands:
        runs.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
        runs = runs[:max_bands]
        runs.sort(key=lambda ab: ab[0])

    xv = np.asarray(x)
    for j, (a, b) in enumerate(runs):
        x0 = xv[a]
        x1 = xv[b] if b + 1 >= len(xv) else xv[b + 1]
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor="#cccccc", opacity=0.25, line_width=0,
            annotation_text=label if j == 0 else None,
            annotation_position="top left",
            annotation_font_size=10,
        )


def plot_distribution(input_df: pd.DataFrame,
                      imputed_dfs: Dict[str, pd.DataFrame],
                      feature: str,
                      gt_df: Optional[pd.DataFrame] = None) -> go.Figure:
    fig = go.Figure()
    obs = input_df[feature].dropna()
    if not obs.empty:
        fig.add_trace(go.Histogram(
            x=obs, name="observed", nbinsx=40,
            marker=dict(color="#1f3b73"), opacity=0.55,
        ))
    if gt_df is not None and feature in gt_df.columns:
        miss = input_df[feature].isna() & gt_df[feature].notna()
        if miss.any():
            fig.add_trace(go.Histogram(
                x=gt_df.loc[miss, feature], name="ground truth (eval)",
                nbinsx=40, marker=dict(color="#2ca02c"), opacity=0.55,
            ))
    miss_all = input_df[feature].isna()
    for i, (key, df) in enumerate(imputed_dfs.items()):
        if feature not in df.columns or not miss_all.any():
            continue
        if len(df) != len(input_df):
            # Row count disagrees with the input split — skip rather than raise
            # on the unalignable boolean mask.
            continue
        fig.add_trace(go.Histogram(
            x=df.loc[miss_all, feature], name=f"{key} (imputed)",
            nbinsx=40, marker=dict(color=PALETTE[i % len(PALETTE)]),
            opacity=0.45,
        ))
    fig.update_layout(
        barmode="overlay",
        title=f"Distribution of '{feature}' — observed vs imputed",
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Long-gap regime tab — reads artifacts from scripts/eval_long_gap.py
# ---------------------------------------------------------------------------

# Stable colours so the same method keeps its colour across both charts.
_LONGGAP_COLORS = {
    "nearest": "#1f77b4", "linear": "#17becf",
    "wsp_v1": "#d62728", "wsp_v2": "#2ca02c",
}
_DEPTH_ORDER = ["d1-1", "d2-4", "d5-8", "d9-16", "d17-+"]


@st.cache_data(show_spinner=False)
def load_long_gap(generated_dir: Optional[Path]) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    if not generated_dir:
        return None
    base = generated_dir / "long_gap"
    res, depth = base / "long_gap_results.csv", base / "long_gap_depth.csv"
    if not res.exists() or not depth.exists():
        return None
    return pd.read_csv(res), pd.read_csv(depth)


def _import_wsp_v2():
    """Lazy import of the pure-python (numpy/pandas) v2 anchoring helpers."""
    app = REPO_ROOT / "dockers" / "tools" / "WaveStitchPlus_app"
    if str(app) not in sys.path:
        sys.path.insert(0, str(app))
    import wsp_v2  # noqa: E402
    return wsp_v2


@st.cache_data(show_spinner=False)
def load_long_gap_fills(generated_dir: Optional[Path], prepared_dir: Path, gap_len: int
                        ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], List[str]]]:
    """For one gap length, return (masked_input, gt, {method: imputed}, targets).

    Reuses the saved masked input + diffusion output; recomputes the
    interpolation priors and v2 anchoring on the fly (cheap, pure-python).
    """
    if not generated_dir:
        return None
    base = generated_dir / "long_gap"
    ti_path = base / f"test_input_L{gap_len}.csv"
    diff_path = base / f"diffusion_L{gap_len}.csv"
    if not ti_path.exists() or not diff_path.exists():
        return None
    v2 = _import_wsp_v2()
    meta = load_meta(prepared_dir)
    tcols = meta["target_cols"]
    ti = pd.read_csv(ti_path)
    diff = pd.read_csv(diff_path)
    gt = pd.read_csv(prepared_dir / "test_gt.csv")
    train = pd.read_csv(prepared_dir / "train.csv") if (prepared_dir / "train.csv").exists() else None
    near = v2.build_prior(train, ti, tcols, method="nearest")
    lin = v2.build_prior(train, ti, tcols, method="linear")
    wsp2 = v2.anchor_blend(ti, diff, near, tcols, tau=4.0, hard_prior=1,
                           has_left_context=train is not None)
    preds = {"nearest": near, "linear": lin, "wsp_v1": diff, "wsp_v2": wsp2}
    return ti, gt, preds, tcols


def plot_long_gap_feature(ti: pd.DataFrame, gt: pd.DataFrame,
                          preds: Dict[str, pd.DataFrame], feature: str,
                          time_col: str) -> go.Figure:
    """Per-method reconstruction of one feature over the masked long gaps."""
    if time_col in ti.columns:
        xser, _ = _coerce_time_axis(ti[time_col])
    else:
        xser = pd.Series(np.arange(len(ti)))
    x = np.asarray(xser)
    inp = ti[feature].to_numpy(float)
    miss = np.isnan(inp)
    gtv = gt[feature].to_numpy(float) if feature in gt.columns else None

    fig = go.Figure()
    _add_missing_bands(fig, x, miss, label="masked gap", max_bands=60)
    if gtv is not None:
        fig.add_trace(go.Scattergl(
            x=x, y=gtv, mode="lines", name="ground truth",
            line=dict(color="#2ca02c", width=2), connectgaps=False,
            hovertemplate="GT t=%{x}<br>y=%{y}<extra></extra>",
        ))
    fig.add_trace(go.Scattergl(
        x=x[~miss], y=inp[~miss], mode="markers", name="observed",
        marker=dict(size=4, color="#1f3b73"),
        hovertemplate="obs t=%{x}<br>y=%{y}<extra></extra>",
    ))
    for m in ["nearest", "linear", "wsp_v1", "wsp_v2"]:
        if m not in preds or feature not in preds[m].columns:
            continue
        full = inp.copy()
        full[miss] = preds[m][feature].to_numpy(float)[miss]
        y = np.where(miss, full, np.nan)  # draw methods only inside the gaps
        fig.add_trace(go.Scattergl(
            x=x, y=y, mode="lines", name=m, connectgaps=False,
            line=dict(color=_LONGGAP_COLORS.get(m), width=1.6),
            hovertemplate=m + " t=%{x}<br>y=%{y}<extra></extra>",
        ))
    fig.update_layout(
        title=f"Long-gap fill of '{feature}' — methods drawn inside masked gaps",
        height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def render_long_gap_tab(subset: Subset) -> None:
    st.write(
        "Does the WaveStitch+ **diffusion** pay off in *its* regime — deep inside "
        "long gaps, where interpolation can only draw a straight line? This view "
        "carves contiguous gaps out of fully-observed runs and scores each method "
        "on the masked cells, overall and by **depth** (distance to the nearest "
        "observed point)."
    )
    loaded = load_long_gap(subset.generated_dir)
    if loaded is None:
        st.info(
            "No long-gap artifacts for this subset yet. Generate them with:\n\n"
            f"```\npython scripts/eval_long_gap.py \\\n"
            f"  --prepared-dir {subset.prepared_dir} \\\n"
            f"  --gap-lengths 16,32,64,128,256 --context 8\n```\n\n"
            "Feasible only where the test split has long observed runs (e.g. EUR/python). "
            "Subsets with isolated observations (golang/rabbitmq) can't form long gaps.",
            icon="📂",
        )
        return

    results, depth = loaded
    methods = [m for m in ["nearest", "linear", "wsp_v1", "wsp_v2"] if m in results["method"].unique()]

    # Chart 1 — overall MAE vs gap length.
    fig1 = go.Figure()
    for m in methods:
        d = results[results["method"] == m].sort_values("gap_len")
        fig1.add_trace(go.Scatter(
            x=d["gap_len"], y=d["MAE"], mode="lines+markers", name=m,
            line=dict(color=_LONGGAP_COLORS.get(m), width=2),
        ))
    fig1.update_layout(
        title="Overall MAE vs gap length (lower = better)",
        xaxis_title="gap length L (rows masked per block)", yaxis_title="MAE",
        xaxis_type="log", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )

    # Chart 2 — MAE vs depth-into-gap for a chosen gap length (the crossover).
    gap_opts = sorted(depth["gap_len"].unique())
    if len(gap_opts) <= 1:
        # st.select_slider renders as a 0..N-1 range and crashes when min == max,
        # so it needs >= 2 options. With a single gap length there is nothing to
        # slide: pin the value and show a hint instead of the widget.
        sel_L = gap_opts[0] if gap_opts else None
        if sel_L is not None:
            st.caption(
                f"Gap length for the depth view: **{sel_L}** — re-run "
                "`eval_long_gap.py` with more `--gap-lengths` to compare across lengths."
            )
    else:
        default_idx = gap_opts.index(32) if 32 in gap_opts else len(gap_opts) - 1
        sel_L = st.select_slider("Gap length for the depth view", options=gap_opts,
                                 value=gap_opts[default_idx])
    if sel_L is None:
        st.info("No depth-bucketed long-gap results to plot yet.")
        return
    sub = depth[depth["gap_len"] == sel_L]
    fig2 = go.Figure()
    for m in methods:
        d = sub[sub["method"] == m].set_index("depth_bucket").reindex(_DEPTH_ORDER).reset_index()
        fig2.add_trace(go.Scatter(
            x=d["depth_bucket"], y=d["MAE"], mode="lines+markers", name=m,
            line=dict(color=_LONGGAP_COLORS.get(m), width=2), connectgaps=False,
        ))
    fig2.update_layout(
        title=f"MAE by depth into gap  (L={sel_L})  —  watch interp ↔ diffusion cross over",
        xaxis_title="depth bucket (distance to nearest observed)", yaxis_title="MAE",
        height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )

    c1, c2 = st.columns(2)
    c1.plotly_chart(fig1, width="stretch")
    c2.plotly_chart(fig2, width="stretch")

    with st.expander("Reading this", expanded=False):
        st.markdown(
            "- **Shallow cells (d≤8):** interpolation wins — a masked point sits next "
            "to a real neighbour.\n"
            "- **Deep cells (d≈9–32, within the model's ~32-step window):** the "
            "diffusion (`wsp_v1`) overtakes interpolation, whose error explodes with "
            "depth while the diffusion stays roughly flat.\n"
            "- **`wsp_v2`** (locally anchored) takes interpolation where it's strong and "
            "the diffusion where it's strong — lowest overall MAE for mid-range gaps.\n"
            "- **Beyond the receptive field (very large L):** even the diffusion loses "
            "context, so no method recovers deep cells."
        )
    st.dataframe(
        results.pivot(index="gap_len", columns="method", values="MAE").round(0),
        width="stretch",
    )

    # ---- Per-feature gap inspection --------------------------------------
    st.divider()
    st.subheader(f"Inspect a single gap fill (L={sel_L})")
    fills = load_long_gap_fills(subset.generated_dir, subset.prepared_dir, int(sel_L))
    if fills is None:
        st.caption(
            "Saved masked input / diffusion output for this gap length aren't "
            "present (only the score CSVs were kept). Re-run the harness for this L "
            "to enable the curve view."
        )
    else:
        ti_lg, gt_lg, preds_lg, tcols_lg = fills
        feat = st.selectbox("Feature", options=tcols_lg, index=0, key="longgap_feat")
        st.plotly_chart(
            plot_long_gap_feature(ti_lg, gt_lg, preds_lg, feat,
                                  load_meta(subset.prepared_dir).get("time_col", "time")),
            width="stretch",
        )
        st.caption(
            "Green = ground truth · blue dots = observed · coloured lines = each "
            "method's fill drawn only inside the shaded masked gaps. In long gaps "
            "interpolation collapses to a straight segment while `wsp_v1`/`wsp_v2` "
            "track the structural shape."
        )


def latency_violation_rates(
    input_arr: np.ndarray,
    imputed_by_key: Dict[str, np.ndarray],
    target_cols: List[str],
) -> Optional[pd.DataFrame]:
    """Per-method constraint-violation rate for auto-detected monotone groups.

    A row counts as a violation if its group values are not non-decreasing
    (e.g. lat50 ≤ … ≤ lat100). The rate is over *imputed* rows (rows with ≥1
    NaN in the group in the input), since observed rows aren't the method's
    doing. Returns ``None`` if no ordered group is present.
    """
    groups = _import_wsp_v2().default_monotone_groups(target_cols)
    if not groups:
        return None
    rows = []
    for cols in groups:
        idx = [target_cols.index(c) for c in cols if c in target_cols]
        if len(idx) < 2:
            continue
        miss_any = np.isnan(input_arr[:, idx]).any(axis=1)
        denom = int(miss_any.sum())
        gname = f"{target_cols[idx[0]]}≤…≤{target_cols[idx[-1]]}"
        for key, arr in imputed_by_key.items():
            sub = arr[:, idx]
            nonmono = (np.diff(sub, axis=1) < -1e-6).any(axis=1)
            viol = int((nonmono & miss_any).sum())
            rows.append({
                "method": key, "group": gname,
                "violations": viol, "imputed_rows": denom,
                "violation_rate_%": (100.0 * viol / denom) if denom else 0.0,
            })
    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------
# Run-experiment subprocess
# ---------------------------------------------------------------------------

def _find_v1_output(generated_dir: Optional[Path], split: str = "test") -> Optional[Path]:
    """Locate an existing WaveStitch+ v1 diffusion output to anchor for v2 reuse."""
    if not generated_dir or not generated_dir.exists():
        return None
    for name in (f"wavestitchplus_v1_{split}_imputed.csv",
                 f"wavestitchplus_{split}_imputed.csv"):
        cand = generated_dir / name
        if cand.exists():
            return cand
    return None


def _runner_image(library: str) -> Optional[str]:
    """Docker image for ``library`` (env override wins over the built-in default)."""
    return os.environ.get(f"DATAOPS_IMPUTE_IMAGE_{library.upper()}") or RUNNER_IMAGES.get(library)


def _run_in_docker(library: str, method: str, subset: Subset, splits: List[str],
                   extra_args: List[str], output_dir: Path, gpu: bool,
                   image: Optional[str] = None) -> Tuple[int, str]:
    """Run an imputation method inside its prebuilt Docker image.

    The bundle's parent dir is mounted at ``/work`` and host paths (prepared dir,
    output dir, and any path-valued ``extra_args`` such as ``--reuse-diffusion``)
    are rewritten to container paths. WaveStitch+ images override the entrypoint
    to reach the in-image runner script.
    """
    image = image or _runner_image(library)
    if not image:
        return 127, (f"No Docker image configured for `{library}`. Set "
                     f"`DATAOPS_IMPUTE_IMAGE_{library.upper()}=<image:tag>`.")

    # Fail fast & clearly if the image isn't built locally — otherwise `docker run`
    # tries to pull it and reports a misleading "pull access denied".
    try:
        chk = subprocess.run(["docker", "image", "inspect", image],
                             capture_output=True, text=True)
    except FileNotFoundError:
        return 127, "`docker` not found on PATH. Install Docker, or uncheck 'Run in Docker image'."
    if chk.returncode != 0:
        return 127, (
            f"Docker image `{image}` is not built locally.\n\n"
            f"• List your images:   docker images\n"
            f"• Use the right tag:  set the **Docker image** field in the Run tab "
            f"(or `export DATAOPS_IMPUTE_IMAGE_{library.upper()}=<repo:tag>`)\n"
            f"• Or build it:        bash dockers/tools/<App>/build_image.sh\n\n"
            f"(Docker did not attempt a registry pull.)"
        )

    host_root = subset.prepared_dir.parent.resolve()

    def cpath(p) -> str:
        # Relative to the mounted /work (the container workdir), so every run arg
        # is a relative path — portable across machines and mount points.
        rp = Path(p).resolve()
        try:
            return str(rp.relative_to(host_root))
        except ValueError:
            return str(p)

    def maybe_path(arg: str) -> str:
        try:
            if os.sep in arg and Path(arg).exists():
                return cpath(arg)
        except Exception:
            pass
        return arg

    runner_args = [
        "--prepared-dir", cpath(subset.prepared_dir),
        "--output-dir", cpath(output_dir),
        "--method", method,
        "--inputs", *splits,
    ] + [maybe_path(a) for a in extra_args]

    docker = ["docker", "run", "--rm", "-v", f"{host_root}:/work", "-w", "/work"]
    if gpu or os.environ.get("DATAOPS_IMPUTE_GPU"):
        docker += ["--gpus", "all"]
    if os.environ.get("DATAOPS_IMPUTE_DOCKER_USER", "1").lower() not in ("0", "", "false", "no"):
        try:
            docker += ["--user", f"{os.getuid()}:{os.getgid()}"]   # avoid root-owned outputs
        except AttributeError:
            pass
    if library.startswith("wavestitchplus"):
        docker += ["--entrypoint", "python", image, WSP_SCRIPTS[library]] + runner_args
    else:
        docker += [image] + runner_args

    try:
        proc = subprocess.run(docker, capture_output=True, text=True)
    except FileNotFoundError:
        return 127, "`docker` not found on PATH. Install Docker, or uncheck 'Run in Docker image'."
    head = (f"ran {library}/{method} in Docker image `{image}`:\n"
            f"  {' '.join(docker)}\n\n")
    return proc.returncode, head + (proc.stdout or "") + "\n" + (proc.stderr or "")


def _impute_python() -> Tuple[List[str], bool]:
    """Resolve the interpreter that runs the imputation app runners.

    The heavy imputation libraries (PyPOTS/torch, ImputeGAP, WaveStitch+) can live
    in a different env than the light dashboard env. Returns ``(argv_prefix, is_custom)``:

      * ``DATAOPS_IMPUTE_PYTHON=/path/to/python`` → that interpreter;
      * ``DATAOPS_IMPUTE_CONDA_ENV=autofeat-6g``  → ``conda run -n <env> python``;
      * neither                                   → the dashboard's own ``python``.
    """
    py = os.environ.get("DATAOPS_IMPUTE_PYTHON")
    if py:
        return [py], True
    env = os.environ.get("DATAOPS_IMPUTE_CONDA_ENV")
    if env:
        return ["conda", "run", "--no-capture-output", "-n", env, "python"], True
    return [sys.executable], False


def run_experiment(library: str,
                   method: str,
                   subset: Subset,
                   splits: List[str],
                   extra_args: List[str],
                   *,
                   use_docker: bool = False,
                   gpu: bool = False,
                   image: Optional[str] = None) -> Tuple[int, str]:
    """Invoke an imputation runner and return (exit_code, combined_output).

    Works across all listed libraries via three execution paths:
      * **Dependency-free built-ins** — darts interpolation + ImputeGAP statistics
        (pandas engine), and WaveStitch+ v2 reusing an existing v1 output.
      * **Docker** (``use_docker``) — run the method inside its prebuilt image
        (the deps live there; the dashboard env needs nothing).
      * **Subprocess** — the app runner via :func:`_impute_python` (the dashboard
        env, or one pointed to by ``DATAOPS_IMPUTE_CONDA_ENV`` / ``_PYTHON``).
    """
    output_dir = subset.generated_dir or (
        subset.prepared_dir.parent / f"generated_{subset.prepared_dir.name}"
    )

    # Dependency-free built-ins (skipped when Docker is explicitly requested).
    from dataops.imputation_runner import builtin_methods, impute_bundle
    if not use_docker and method in builtin_methods(library):
        try:
            res = impute_bundle(subset.prepared_dir, method=method, lib=library,
                                output_dir=str(output_dir), inputs=splits, engine="pandas")
        except Exception as exc:  # noqa: BLE001 - surface as a run failure
            return 1, f"{library}/{method} (built-in) failed: {exc}"
        note = "" if library == "darts" else f" — standard equivalent, not the {library} library"
        lines = [f"{library}/{method} via built-in engine (no `{library}` dependency{note}):"]
        for kind, info in res["files"].items():
            lines.append(f"  {kind}: filled {info['filled']:,}/{info['nan_before']:,} "
                         f"NaN target cells → {Path(info['path']).name}")
        return 0, "\n".join(lines)

    runner = RUNNERS[library]
    if not use_docker and not runner.exists():
        return 127, f"runner not found: {runner}"

    # WaveStitch+ v2 anchoring is pure-python (wsp_v2). When a v1 diffusion output
    # already exists, reuse it (`--reuse-diffusion`) so v2 runs torch-free / no
    # retrain. Applies to both the Docker and subprocess paths.
    reuse_note = ""
    if library == "wavestitchplus_v2" and "--reuse-diffusion" not in extra_args:
        # v2 always anchors the TEST split internally, so --reuse-diffusion must be
        # the v1 TEST output regardless of which split(s) the user selected.
        v1 = _find_v1_output(output_dir, "test")
        if v1 is not None:
            extra_args = list(extra_args) + ["--reuse-diffusion", str(v1)]
            reuse_note = (f"v2 anchored an existing v1 output (`{v1.name}`) — "
                          "no diffusion synthesis.\n")

    if use_docker:
        rc, out = _run_in_docker(library, method, subset, splits, extra_args,
                                 output_dir, gpu, image=image)
        return rc, reuse_note + out

    impute_py, custom_env = _impute_python()
    module_name, install_hint = RUNNER_IMPORT_CHECKS.get(library, (None, None))
    # Only gate on the *dashboard* env when we'd run in it. With a custom
    # imputation env the dep lives there, so let the subprocess report for real.
    if module_name and not custom_env:
        import importlib.util

        if importlib.util.find_spec(module_name) is None:
            needs = RUNNER_NEEDS.get(library, f"`{module_name}`")
            return 127, (
                f"`{library}/{method}` isn't available in the dashboard Python env — "
                f"it needs {needs}.\n\n"
                f"Run it without changing this env by ticking **Run in Docker image**, "
                f"or point imputation at an env that has the deps:\n"
                f"  export DATAOPS_IMPUTE_CONDA_ENV=autofeat-6g    # conda env with the deps\n"
                f"  export DATAOPS_IMPUTE_PYTHON=/path/to/python   # or an explicit interpreter\n"
                f"or install them here:\n"
                f"  {install_hint}\n\n"
                f"Dependency-free in this env: darts interpolation, imputegap statistics, "
                f"and WaveStitch+ v2 (when a v1 output exists to anchor)."
            )
    cmd = impute_py + [
        str(runner),
        "--prepared-dir", str(subset.prepared_dir),
        "--output-dir", str(output_dir),
        "--method", method,
        "--inputs", *splits,
    ] + extra_args
    env = os.environ.copy()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return proc.returncode, reuse_note + (proc.stdout or "") + "\n" + (proc.stderr or "")


# ---------------------------------------------------------------------------
# Pipeline-run discovery (SeaweedFS S3)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _s3_client(endpoint: str, access: str, secret: str):
    """Cached boto3 client pointing at the SeaweedFS endpoint."""
    import boto3
    from botocore.client import Config
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name="us-east-1",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


@st.cache_data(show_spinner=False)
def _s3_list_runs(endpoint: str, bucket: str, access: str, secret: str
                  ) -> Dict[str, List[str]]:
    """Discover ``cleaned/<dataset>/<run_id>/`` prefixes -> {dataset: [run_ids]}."""
    client = _s3_client(endpoint, access, secret)
    runs: Dict[str, List[str]] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="cleaned/"):
        for obj in page.get("Contents", []):
            parts = obj["Key"].split("/")
            if len(parts) < 3:
                continue
            dataset, run_id = parts[1], parts[2]
            runs.setdefault(dataset, set()).add(run_id)  # type: ignore[arg-type]
    return {ds: sorted(ids, reverse=True) for ds, ids in runs.items()}  # type: ignore[misc]


@st.cache_data(show_spinner=False)
def _s3_get_csv(endpoint: str, bucket: str, access: str, secret: str,
                key: str) -> Optional[pd.DataFrame]:
    """Download a CSV object as DataFrame; returns None if missing."""
    client = _s3_client(endpoint, access, secret)
    try:
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except client.exceptions.NoSuchKey:
        return None
    except Exception:
        return None
    from io import BytesIO
    return pd.read_csv(BytesIO(body))


@st.cache_data(show_spinner=False)
def _s3_get_json(endpoint: str, bucket: str, access: str, secret: str,
                 key: str) -> Optional[dict]:
    client = _s3_client(endpoint, access, secret)
    try:
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:
        return None
    try:
        return json.loads(body)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _s3_resolve_raw_key(endpoint: str, bucket: str, access: str, secret: str,
                        dataset: str) -> Optional[str]:
    """Guess the raw input key for a dataset.

    Looks for ``test/<dataset>*.csv`` first (the DAG's default location), then
    falls back to any ``*<dataset>*.csv`` in the bucket. Returns the first hit.
    """
    client = _s3_client(endpoint, access, secret)
    candidates: List[str] = []
    for prefix in ("test/", ""):
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        ):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if k.endswith(".csv") and dataset.lower() in k.lower():
                    candidates.append(k)
        if candidates:
            break
    return candidates[0] if candidates else None


def _coerce_time_axis(series: pd.Series) -> Tuple[pd.Series, str]:
    """
    Return ``(x_values, axis_label)`` for plotting.

    If the series is already datetime, returns it untouched. If it looks like
    Unix-epoch seconds / milliseconds (large positive numerics), converts to
    datetime so the x-axis reads as wall-clock time. Otherwise returns the
    series as-is.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series, series.name or "time"
    if pd.api.types.is_numeric_dtype(series):
        s = pd.to_numeric(series, errors="coerce")
        med = float(s.dropna().median()) if s.notna().any() else 0.0
        # Heuristic: 10^9 ≤ |t| < 10^11   → seconds since 1970   (1970..5138)
        #            10^12 ≤ |t| < 10^14  → milliseconds         (1970..5138)
        if 1e9 <= med < 1e11:
            return pd.to_datetime(s, unit="s", errors="coerce"), f"{series.name} (UTC)"
        if 1e12 <= med < 1e14:
            return pd.to_datetime(s, unit="ms", errors="coerce"), f"{series.name} (UTC)"
    return series, series.name or "time"


def _pipeline_compare_plot(raw: pd.DataFrame, soft_cleaned: pd.DataFrame,
                           curated: Optional[pd.DataFrame], column: str,
                           time_col: str) -> go.Figure:
    """Three-line plot: raw vs soft-cleaned vs curated for one numeric column."""
    fig = go.Figure()
    x_label = "row index"
    # Use row index when both frames share length; otherwise fall back to time
    # column. Different row counts (e.g. dedup removed rows) are tolerated.
    def _x(df: pd.DataFrame):
        nonlocal x_label
        if time_col in df.columns:
            x, x_label = _coerce_time_axis(df[time_col])
            return x
        return pd.Series(np.arange(len(df)))

    fig.add_trace(go.Scattergl(
        x=_x(raw), y=raw[column].astype(float),
        name="raw (source)", mode="lines",
        line=dict(color="#1f77b4", width=1),
        opacity=0.55,
    ))
    fig.add_trace(go.Scattergl(
        x=_x(soft_cleaned), y=soft_cleaned[column].astype(float),
        name="soft-cleaned", mode="lines",
        line=dict(color="#ff7f0e", width=1.4),
    ))
    if curated is not None and column in curated.columns:
        fig.add_trace(go.Scattergl(
            x=_x(curated), y=curated[column].astype(float),
            name="curated", mode="lines",
            line=dict(color="#2ca02c", width=1, dash="dot"),
        ))
    fig.update_layout(
        title=f"{column} — raw vs soft-cleaned vs curated",
        xaxis_title=x_label,
        yaxis_title=column,
        height=420, margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


_SEVERITY_COLORS = {"error": "#d62728", "warning": "#e8820c", "info": "#1f77b4"}
_STATUS_BADGES = {
    "applied_by_remediation": ("#2ca02c", "✓ applied"),
    "deferred_to_imputation": ("#7e57c2", "→ imputation"),
    "marked_quality_issue": ("#e8820c", "marked"),
    "manual": ("#6c757d", "manual"),
}


def _badge(text: str, color: str) -> str:
    return (
        f"<span style='background:{color};color:#fff;padding:2px 9px;"
        f"border-radius:11px;font-size:0.78em;font-weight:600;white-space:nowrap'>"
        f"{text}</span>"
    )


def _passed_badge(value: Optional[bool]) -> str:
    if value is True:
        return _badge("PASSED", "#2ca02c")
    if value is False:
        return _badge("FAILED", "#d62728")
    return _badge("not run", "#6c757d")


def _bool_badge(value: Optional[bool], yes: str, no: str) -> str:
    if value is True:
        return _badge(yes, "#e8820c")
    if value is False:
        return _badge(no, "#2ca02c")
    return _badge("—", "#6c757d")


def _issue_counts_chart(before: dict, after: Optional[dict]) -> Optional[go.Figure]:
    families = ["timestamp_order", "ts_gaps", "missing", "outliers", "failed_columns"]
    nice = {"ts_gaps": "time gaps", "missing": "missing",
            "outliers": "outliers", "failed_columns": "GX failed cols",
            "timestamp_order": "timestamp order"}
    bvals = [int(before.get(f, 0)) for f in families]
    if not any(bvals) and not after:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[nice[f] for f in families], y=bvals, name="detected",
        marker_color="#e8820c",
        text=bvals, textposition="outside",
    ))
    if after is not None:
        avals = [int(after.get(f, 0)) for f in families]
        fig.add_trace(go.Bar(
            x=[nice[f] for f in families], y=avals, name="after remediation",
            marker_color="#2ca02c", text=avals, textposition="outside",
        ))
    fig.update_layout(
        barmode="group", height=320,
        title="Quality issues — detected vs after remediation",
        yaxis_title="count", margin=dict(l=30, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def _lifecycle_chart(comparison: dict) -> Optional[go.Figure]:
    cleaning = comparison.get("cleaning_effect", {})
    remed = comparison.get("remediation_effect", {})
    miss = [
        cleaning.get("missing_cells_before"),
        cleaning.get("missing_cells_after"),
        remed.get("missing_cells_after"),
    ]
    if all(v is None for v in miss):
        return None
    stages = ["raw", "soft-cleaned", "remediated"]
    vals = [int(v) if v is not None else 0 for v in miss]
    fig = go.Figure(go.Bar(
        x=stages, y=vals, marker_color=["#1f77b4", "#e8820c", "#2ca02c"],
        text=vals, textposition="outside",
    ))
    fig.update_layout(
        height=320, title="Missing cells across pipeline stages",
        yaxis_title="missing cells", margin=dict(l=30, r=20, t=50, b=30),
    )
    return fig


def _render_action_plan(actions: List[dict]) -> None:
    """Issue → solution plan, grouped into auto-handled vs needs-attention."""
    auto = [a for a in actions if a.get("status") == "applied_by_remediation"]
    deferred = [a for a in actions if a.get("status") == "deferred_to_imputation"]
    manual = [a for a in actions if a.get("status") not in
              {"applied_by_remediation", "deferred_to_imputation"}]

    def _line(a: dict) -> str:
        sev = a.get("severity", "info")
        status = a.get("status", "manual")
        color, label = _STATUS_BADGES.get(status, ("#6c757d", status))
        return (
            f"<div style='margin:6px 0;padding:8px 12px;border-left:4px solid "
            f"{_SEVERITY_COLORS.get(sev, '#1f77b4')};background:#fafafa;border-radius:4px'>"
            f"{_badge(label, color)} &nbsp;<b>{a.get('issue')}</b> "
            f"<span style='color:#888;font-size:0.85em'>({sev})</span><br>"
            f"<span style='font-size:0.9em'>{a.get('solution', '')}</span><br>"
            f"<code style='font-size:0.8em;color:#555'>{a.get('module', '')}</code>"
            f"</div>"
        )

    if auto:
        st.markdown("**Auto-handled by remediation**")
        st.markdown("".join(_line(a) for a in auto), unsafe_allow_html=True)
    if deferred:
        st.markdown("**Deferred to imputation**")
        st.markdown("".join(_line(a) for a in deferred), unsafe_allow_html=True)
    if manual:
        st.markdown("**Needs manual attention**")
        st.markdown("".join(_line(a) for a in manual), unsafe_allow_html=True)


def _render_handoff(handoff: dict) -> None:
    needs = handoff.get("needs_ts_imputation")
    st.markdown(
        f"Needs time-series imputation: {_bool_badge(needs, 'YES', 'NO')} "
        f"&nbsp;<span style='color:#888;font-size:0.85em'>({handoff.get('reason', '')})</span>",
        unsafe_allow_html=True,
    )
    if not needs:
        st.caption("No timeline gaps detected — nothing routed to imputation.")
        return

    sel = handoff.get("selection", {}) or {}
    sel_status = sel.get("status")
    status_color = {
        "ok": "#2ca02c", "known_failing": "#e8820c",
        "invalid": "#d62728", "none_configured": "#6c757d",
    }.get(sel_status, "#6c757d")
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected app", sel.get("app") or "—")
    c2.metric("Method", sel.get("method") or "—")
    c3.markdown(
        f"<div style='padding-top:14px'>{_badge(sel_status or '—', status_color)}</div>",
        unsafe_allow_html=True,
    )
    if sel.get("message"):
        st.caption(sel["message"])

    if handoff.get("bundle_written"):
        st.success(f"Imputation-ready bundle written to `{handoff.get('prepared_dir')}` "
                   f"· targets: {', '.join(handoff.get('target_cols', [])) or '—'}")
    elif handoff.get("bundle_error"):
        st.warning(f"Bundle not written: {handoff['bundle_error']}")

    hint = handoff.get("invoke_hint")
    if hint:
        st.caption("Next step (run by the external orchestrator):")
        st.code(hint, language="bash")

    catalog = handoff.get("imputation_catalog", {})
    if catalog:
        with st.expander("Imputation method catalog", expanded=False):
            rows = []
            for app, spec in catalog.items():
                rows.append({
                    "app": app,
                    "default": spec.get("default"),
                    "methods": len(spec.get("methods", [])),
                    "known_failing": ", ".join(spec.get("known_failing", [])) or "—",
                    "available": ", ".join(spec.get("methods", [])),
                })
            st.dataframe(pd.DataFrame(rows).set_index("app"), width="stretch")


def _render_gx_failures(quality: dict, quality_after: Optional[dict]) -> None:
    """Surface the concrete failed GX expectations so 'manual attention' is actionable."""
    src = quality_after if (quality_after and (quality_after.get("report") or {}).get("gx")) else quality
    gx = (src.get("report") or {}).get("gx", {})
    failed = gx.get("failed_expectations", [])
    if not failed:
        return
    which = "after remediation" if src is quality_after else "as detected"
    st.markdown(f"**Failed GX expectations ({which})** — {gx.get('failed', len(failed))} "
                f"of {gx.get('evaluated', '?')} evaluated")
    fdf = pd.DataFrame(failed).rename(columns={
        "expectation": "expectation", "column": "column",
        "unexpected_percent": "unexpected %",
    })
    st.dataframe(fdf, width="stretch", hide_index=True)
    types = {f.get("expectation") for f in failed}
    if types == {"expect_column_values_to_be_between"}:
        st.caption(
            "ℹ️ All failures are the quantile-based range sentinel "
            "`expect_column_values_to_be_between(q1, q99, mostly=0.98)`. By "
            "construction ~2% of a continuous column lies outside its own 1st/99th "
            "percentile, so this check trips by design and is **not** a hard schema "
            "break. Winsorizing doesn't clear it — the after-check recomputes the "
            "same quantiles on the clipped data (a fixed point). Treat these as soft "
            "outlier flags, or relax `outlier_q` / `mostly` if you want them green."
        )


def _render_imputation_compare(impute_compare: dict) -> None:
    """Final cleaned dataset callout + clean-vs-imputed fill rate / per-column MAE."""
    final = impute_compare.get("final_dataset") or {}
    comp = impute_compare.get("comparison") or {}
    imp = impute_compare.get("imputation") or {}
    method = imp.get("method") or comp.get("method") or "?"
    lib = imp.get("lib") or "darts"

    if final:
        filled = final.get("gaps_before", 0) - final.get("gaps_after", 0)
        st.success(
            f"**Final cleaned data** (imputed train + imputed test) → "
            f"`{final.get('path')}`  ·  "
            f"{final.get('rows', 0):,} rows × {len(final.get('columns', []))} cols  ·  "
            f"gaps filled {filled:,}/{final.get('gaps_before', 0):,} via "
            f"{lib}/{method} (residual {final.get('gaps_after', 0):,})"
        )

    splits = comp.get("splits", {})
    cols = st.columns(2)
    if splits:
        fr = go.Figure(go.Bar(
            x=list(splits.keys()),
            y=[s.get("fill_rate", 0) * 100 for s in splits.values()],
            marker_color="#2ca02c",
            text=[f"{s.get('fill_rate', 0) * 100:.0f}%" for s in splits.values()],
            textposition="outside",
        ))
        fr.update_layout(
            title=f"Gap fill rate by split (darts/{method})", yaxis_title="% filled",
            height=320, yaxis_range=[0, 110], margin=dict(l=30, r=20, t=50, b=30),
        )
        cols[0].plotly_chart(fr, width="stretch")

    acc = (splits.get("test") or {}).get("accuracy") or {}
    per_col = acc.get("per_column") or {}
    if per_col:
        items = sorted(per_col.items(), key=lambda kv: kv[1]["MAE"], reverse=True)
        mae = go.Figure(go.Bar(
            x=[v["MAE"] for _, v in items], y=[k for k, _ in items],
            orientation="h", marker_color="#1f77b4",
        ))
        mae.update_layout(
            title="Per-column MAE on eval cells (test)", xaxis_type="log",
            xaxis_title="MAE (log scale)", height=320, margin=dict(l=30, r=20, t=50, b=30),
        )
        cols[1].plotly_chart(mae, width="stretch")
        cols[1].caption(
            "Eval cells = holdout-masked positions with known truth. Log scale "
            "because columns span microseconds → bytes; `nearest` is a weak "
            "baseline on spiky latency (high MAE) but exact on constants like `cpu_limit`."
        )


def _guess_time_column(df: pd.DataFrame) -> str:
    for name in df.columns:
        low = str(name).lower()
        if low in {"time", "timestamp", "date", "datetime"} or "time" in low:
            return str(name)
    return str(df.columns[0]) if len(df.columns) else "index"


def render_raw_data_view(raw: Optional[RawDataset], selected_run: Optional["DataOpsRun"]) -> None:
    """Raw-data landing view: source preview, quick quality, and linked runs."""
    if raw is None:
        st.info("No raw CSV found under `data/raw/`, and no selected report references a raw input.")
        return
    st.markdown(f"#### Raw data `{raw.name}`")
    st.caption(f"Source: `{raw.path}`")
    if not raw.path.exists():
        st.warning("The selected raw CSV path is referenced by a report but is not present locally.")
        return

    df = load_csv(raw.path)
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    m = st.columns(4)
    m[0].metric("Rows", f"{len(df):,}")
    m[1].metric("Columns", f"{df.shape[1]:,}")
    m[2].metric("Missing cells", f"{missing_cells:,}", delta_color="inverse")
    m[3].metric("Duplicate rows", f"{duplicate_rows:,}", delta_color="inverse")

    if raw.runs:
        rows = []
        for r in raw.runs:
            status = ((r.report or {}).get("validation_comparison") or {}).get(
                "validation_status", {}
            )
            qsum = ((r.report or {}).get("quality") or {}).get("issue_summary", {})
            rows.append({
                "run": r.name,
                "type": r.data_type,
                "pandera": status.get("pandera_passed"),
                "gx": ((r.report or {}).get("quality") or {}).get("gx_passed"),
                "issues": int(sum(int(v) for v in qsum.values())),
                "final": bool(r.final_csv and r.final_csv.exists()),
                "report": str(r.report_path) if r.report_path else "",
            })
        st.markdown("##### Runs for this raw data")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if selected_run:
            st.caption(f"Current dashboard context: `{selected_run.name}`")
    else:
        st.info("No DataOps run has been generated for this raw dataset yet.")

    st.markdown("##### Raw preview & visualization")
    p1, p2 = st.columns([1, 2])
    with p1:
        st.dataframe(df.head(100), width="stretch")
    with p2:
        if not numeric_cols:
            st.info("No numeric columns available for quick visualization.")
        else:
            time_col = _guess_time_column(df)
            feature = st.selectbox(
                "Raw feature", options=[c for c in numeric_cols if c != time_col] or numeric_cols,
                key=f"raw_feature_{raw.name}",
            )
            x = df[time_col] if time_col in df.columns else pd.Series(np.arange(len(df)))
            x, x_label = _coerce_time_axis(x)
            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=x,
                y=pd.to_numeric(df[feature], errors="coerce"),
                mode="lines",
                name=feature,
                line=dict(color="#1f77b4", width=1.2),
            ))
            fig.update_layout(
                title=f"Raw {feature}",
                xaxis_title=x_label,
                yaxis_title=feature,
                height=360,
                margin=dict(l=35, r=20, t=45, b=35),
            )
            st.plotly_chart(fig, width="stretch")


def render_validation_comparison(report: dict, data_type: str,
                                 impute_compare: Optional[dict] = None) -> None:
    comparison = report.get("validation_comparison", {})
    quality = report.get("quality", {})
    quality_after = report.get("quality_after") or {}
    handoff = report.get("handoff", {})
    status = comparison.get("validation_status", {})

    if not comparison and not quality:
        st.info("No validation comparison payload found in the report. Re-run "
                "`python -m pipelines.minimal_dataops` to populate it.")
        return

    st.caption("Lineage:  raw → soft-cleaned → remediated → regularized (gaps explicit) "
               "→ **final** (imputed, gap-free)")

    # ---- Validation status badges ----------------------------------------
    st.markdown("##### Validation status")
    mode = status.get("mode", data_type)
    gx_b = (quality.get("report") or {}).get("gx", {})
    gx_a = (quality_after.get("report") or {}).get("gx", {}) if quality_after else {}

    def _gx_label(passed: Optional[bool], detail: dict) -> str:
        badge = _passed_badge(passed)
        if detail.get("evaluated"):
            badge += (f" <span style='color:#888;font-size:0.85em'>"
                      f"{detail.get('passed')}/{detail.get('evaluated')} expectations</span>")
        return badge

    st.markdown(
        f"GX detected {_gx_label(quality.get('gx_passed'), gx_b)} "
        f"&nbsp;→&nbsp; after remediation "
        f"{_gx_label(quality_after.get('gx_passed') if quality_after else None, gx_a)} "
        f"&nbsp;&nbsp;·&nbsp;&nbsp; Pandera "
        f"{_passed_badge(status.get('pandera_passed'))} "
        f"&nbsp;&nbsp;·&nbsp;&nbsp; mode <code>{mode}</code>",
        unsafe_allow_html=True,
    )

    # ---- Lifecycle metrics -----------------------------------------------
    shape = comparison.get("dataset_shape", {})
    cleaning = comparison.get("cleaning_effect", {})
    remed = comparison.get("remediation_effect", {})
    raw_rows = shape.get("raw", {}).get("rows")
    soft_shape = shape.get("soft_cleaned") or shape.get("cleaned", {})
    remed_rows = shape.get("remediated", {}).get("rows", soft_shape.get("rows"))
    miss_raw = cleaning.get("missing_cells_before")
    miss_remed = remed.get("missing_cells_after", cleaning.get("missing_cells_after"))

    m = st.columns(4)
    m[0].metric(
        "Final rows", f"{remed_rows:,}" if remed_rows is not None else "—",
        delta=(f"{remed_rows - raw_rows:+,}" if raw_rows is not None and remed_rows is not None else None),
        help="raw → soft-cleaned → remediated",
    )
    m[1].metric(
        "Missing cells", f"{miss_remed:,}" if miss_remed is not None else "—",
        delta=(f"{miss_remed - miss_raw:+,}" if miss_raw is not None and miss_remed is not None else None),
        delta_color="inverse",
    )
    m[2].metric("Outliers clipped", f"{remed.get('outlier_cells_clipped', 0):,}")
    m[3].metric("Rows dropped (cleaning)", f"{cleaning.get('dropped_rows', 0):,}",
                help="empty + duplicate rows removed by the first-pass cleaning")

    # ---- Two charts side by side -----------------------------------------
    c1, c2 = st.columns(2)
    life = _lifecycle_chart(comparison)
    if life is not None:
        c1.plotly_chart(life, width="stretch")
    issues = _issue_counts_chart(
        quality.get("issue_summary", {}),
        quality_after.get("issue_summary") if quality_after else None,
    )
    if issues is not None:
        c2.plotly_chart(issues, width="stretch")

    # ---- Failed GX expectations ------------------------------------------
    _render_gx_failures(quality, quality_after)

    # ---- Issue → solution plan -------------------------------------------
    actions = quality.get("action_plan", [])
    if actions:
        st.divider()
        st.markdown("##### Issue → solution plan")
        _render_action_plan(actions)

    # ---- Imputation handoff ----------------------------------------------
    if handoff:
        st.divider()
        st.markdown("##### Imputation handoff")
        _render_handoff(handoff)

    # ---- Final cleaned data & clean-vs-imputed comparison ----------------
    if impute_compare:
        st.divider()
        st.markdown("##### Final cleaned data & imputation comparison")
        _render_imputation_compare(impute_compare)
    elif handoff.get("needs_ts_imputation"):
        st.divider()
        st.caption(
            "No imputation comparison yet. Produce the final cleaned dataset with:  "
            "`python scripts/auto_impute.py --report <this report> --method nearest`"
        )


@st.cache_data(show_spinner=False)
def _row_count(path: Path) -> Optional[int]:
    """Row count of a CSV (reads one column; cached)."""
    try:
        return int(pd.read_csv(path, usecols=[0]).shape[0])
    except Exception:
        return None


def render_overview(run: Optional["DataOpsRun"]) -> None:
    """Cleaning-first landing view: the raw→...→final lineage for one run."""
    if run is None or not run.report:
        st.info("No DataOps pipeline run selected. Pick a run (`reports/*.json`) in the "
                "sidebar, or generate one with "
                "`python -m pipelines.minimal_dataops --config config/dataops.yaml`.")
        return
    rep = run.report
    vc = rep.get("validation_comparison", {})
    shape = vc.get("dataset_shape", {})
    cleaning = vc.get("cleaning_effect", {})
    remed = vc.get("remediation_effect", {})
    compare = run.compare or {}
    final = compare.get("final_dataset") or {}
    q = rep.get("quality", {})
    qa = rep.get("quality_after") or {}

    st.markdown(f"#### Run `{run.name}` · type `{run.data_type}`")
    st.caption("Lineage:  raw → soft-cleaned → remediated → regularized (gaps explicit) "
               "→ **final** (imputed, gap-free)")

    # ---- pick the DELIVERED final among all this run's gap-free finals ----------
    finals_all = discover_final_files(run.generated_dir, run)
    pipeline_final_name = Path(final["path"]).name if final.get("path") else None
    canon_label, canon_path = None, None
    if finals_all:
        labels = list(finals_all)
        default_ix = next((i for i, k in enumerate(labels)
                           if pipeline_final_name and Path(finals_all[k]).name == pipeline_final_name), 0)
        key = f"delivered_final::{run.name}"
        if key not in st.session_state:
            st.session_state[key] = labels[default_ix]
        canon_label = st.selectbox(
            "Delivered final (交付版)", options=labels, key=key,
            help="Which gap-free final to treat as THE delivered dataset for this run — "
                 "it drives the `final` stage below. All finals live under "
                 "data/processed; WaveStitch+ v2 usually beats the pipeline `nearest` default.",
        )
        canon_path = Path(finals_all[canon_label])
    elif final.get("path"):
        canon_path = Path(final["path"])
    elif run.final_csv:
        canon_path = Path(run.final_csv)

    canon_exists = bool(canon_path and Path(canon_path).exists())
    canon_rows = (_row_count(canon_path) if canon_exists else None) or final.get("rows")
    is_pipeline_canon = bool(pipeline_final_name and canon_path
                             and Path(canon_path).name == pipeline_final_name)

    reg_rows = None
    if run.prepared_dir and (run.prepared_dir / "meta.json").exists():
        try:
            reg_rows = load_meta(run.prepared_dir).get("regularized_rows")
        except Exception:
            reg_rows = None

    soft_shape = shape.get("soft_cleaned") or shape.get("cleaned", {})
    cards = [
        ("raw", run.raw_csv, shape.get("raw", {}).get("rows"), None),
        ("soft-cleaned", run.soft_cleaned_csv, soft_shape.get("rows"), None),
        ("remediated", run.remediated_csv, shape.get("remediated", {}).get("rows"),
         f"{remed.get('outlier_cells_clipped', 0):,} clipped"),
        ("regularized", run.prepared_dir, reg_rows, "gaps explicit"),
        ("final", canon_path, canon_rows, f"gap-free · {canon_label}" if canon_exists
         else "not built yet"),
    ]
    cols = st.columns(5)
    for c, (stage, path, rows, note) in zip(cols, cards):
        c.metric(stage, f"{rows:,}" if isinstance(rows, int) else "—")
        exists = bool(path and Path(path).exists())
        label = Path(path).name if path else "—"
        c.caption(("✓ " if exists else "✗ ") + label + (f" · {note}" if note else ""))

    gx_b = (q.get("report") or {}).get("gx", {})
    gx_a = (qa.get("report") or {}).get("gx", {}) if qa else {}
    pandera = vc.get("validation_status", {}).get("pandera_passed")
    line = (f"GX detected {_passed_badge(q.get('gx_passed'))}"
            + (f" <span style='color:#888;font-size:.85em'>{gx_b.get('passed')}/{gx_b.get('evaluated')}</span>"
               if gx_b.get("evaluated") else "")
            + f" → after remediation {_passed_badge(qa.get('gx_passed') if qa else None)}"
            + (f" <span style='color:#888;font-size:.85em'>{gx_a.get('passed')}/{gx_a.get('evaluated')}</span>"
               if gx_a.get("evaluated") else "")
            + f" &nbsp;·&nbsp; Pandera {_passed_badge(pandera)}")
    st.markdown(line, unsafe_allow_html=True)

    handoff = rep.get("handoff", {})
    st.markdown(
        f"Needs ts-imputation: {_bool_badge(handoff.get('needs_ts_imputation'), 'YES', 'NO')} "
        f"<span style='color:#888;font-size:.85em'>({handoff.get('reason', '')})</span>",
        unsafe_allow_html=True,
    )
    if canon_exists:
        if is_pipeline_canon and final.get("path"):
            filled = final.get("gaps_before", 0) - final.get("gaps_after", 0)
            st.success(f"★ Delivered final [{canon_label or 'pipeline'}]: `{canon_path}` · "
                       f"{canon_rows:,} rows · gaps filled {filled:,} "
                       f"(residual {final.get('gaps_after', 0):,})")
        else:
            st.success(f"★ Delivered final [{canon_label}]: `{canon_path}` · "
                       f"{canon_rows:,} rows · gap-free")

    # Every gap-free FINAL for this run, all under data/processed; the selected one
    # (★ delivered) drives the `final` stage card above.
    if finals_all:
        with st.expander(f"All final datasets for this run ({len(finals_all)}) "
                         f"— unified under `data/processed/`", expanded=True):
            for label, p in finals_all.items():
                n = _row_count(Path(p))
                star = " ★ delivered" if label == canon_label else ""
                st.caption(f"　✓ **{label}**{star} → `{p}`"
                           + (f" · {n:,} rows · gap-free" if n else ""))
    st.caption("→ full breakdown in **Quality & remediation**; per-method comparison and the "
               "stitched-timeline plot in the **Imputation → Final** tab.")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="6G-DALI Imputation Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("6G-DALI DataOps — data cleaning & imputation")
    st.caption("One run, end to end: raw → soft-cleaned → remediated → regularized → final (imputed). "
               "Imputation method comparison (Darts / ImputeGAP / PyPOTS / WaveStitch+) is the "
               "detail of the final step.")

    # ---- Sidebar: pick raw data first, then the run derived from it ----------
    if st.sidebar.button("Refresh runs", key="refresh_runs"):
        discover_dataops_runs.clear()
        # discover_subsets is no longer cached (returns script-defined dataclasses).
        load_csv.clear()
        load_csv_subset.clear()
    runs = discover_dataops_runs(REPO_ROOT / "reports", REPO_ROOT)
    raw_datasets = discover_raw_datasets(REPO_ROOT / "data" / "raw", runs)
    raw: Optional[RawDataset] = None
    if raw_datasets:
        raw_labels = [r.label for r in raw_datasets]
        raw_sel = st.sidebar.selectbox(
            "Raw data",
            raw_labels,
            index=0,
            help="CSV sources discovered from `data/raw/*.csv` and pipeline reports.",
        )
        raw = raw_datasets[raw_labels.index(raw_sel)]
        scoped_runs = raw.runs
    else:
        st.sidebar.caption("No raw CSVs found under `data/raw/`.")
        scoped_runs = runs

    run_names = [r.name for r in scoped_runs]
    run: Optional[DataOpsRun] = None
    if run_names:
        sel = st.sidebar.selectbox(
            "DataOps run",
            run_names,
            index=0,
            help="A pipeline report for the selected raw data.",
        )
        run = scoped_runs[run_names.index(sel)]
    else:
        st.sidebar.caption("No pipeline reports found for this raw data.")

    st.sidebar.divider()

    # ---- Imputation workbench source: the run's bundle, else an experiment subset
    work_root = Path(st.sidebar.text_input(
        "Bundle root (fallback)", value=str(DEFAULT_WORK_ROOT),
        help="Folder holding prepared bundles (`*_prepared/` under data/processed, or "
             "`prepared_<subset>/` under an experiments tree). Used only when no "
             "DataOps run is selected above.",
    ))
    subsets = discover_subsets(work_root)
    subset: Optional[Subset] = run.to_subset() if run else None
    if subset is None and subsets:
        labels = [s.label for s in subsets]
        chosen = st.sidebar.selectbox("Subset (experiment)", labels, index=0)
        subset = subsets[labels.index(chosen)]

    # No imputation source (no bundle, no experiment subset) → cleaning-only view.
    if subset is None:
        t_raw, t_ov, t_q = st.tabs(["Raw data", "Overview", "Quality & remediation"])
        with t_raw:
            render_raw_data_view(raw, run)
        with t_ov:
            render_overview(run)
        with t_q:
            if run and run.report:
                render_validation_comparison(run.report, run.data_type, impute_compare=run.compare)
            else:
                st.info("Select a DataOps run with a report to see quality & remediation.")
        return

    # ---- Top-level sections (cleaning-first) -------------------------------
    tab_raw, tab_overview, tab_quality, tab_imp, tab_run_sec = st.tabs(
        ["Raw data", "Overview", "Quality & remediation", "Imputation", "Run"]
    )

    with tab_raw:
        render_raw_data_view(raw, run)

    with tab_overview:
        render_overview(run)

    with tab_quality:
        if run and run.report:
            render_validation_comparison(run.report, run.data_type, impute_compare=run.compare)
        else:
            st.info("No DataOps report for this source (experiment-only subset). Pick a "
                    "pipeline run in the sidebar to see quality & remediation.")

    # ---- Imputation: method comparison on the run's regularized bundle -----
    #      (pickers live here, not in the sidebar; sub-tabs are children of the
    #       Imputation section so the bodies below need no re-indentation).
    with tab_imp:
        meta = load_meta(subset.prepared_dir)
        target_cols: List[str] = list(meta.get("target_cols", []))
        time_col: str = meta.get("time_col", "time")

        pick = st.columns([1, 3, 2])
        split = pick[0].radio("Split", options=["test", "train"], horizontal=True, index=0)
        input_path = subset.prepared_dir / INPUT_FILES[split]
        if not input_path.exists():
            st.error(f"Missing input file: {input_path}")
            return

        available = discover_imputed_files(subset, split)
        method_keys = sorted(available.keys())

        # Selection lives in session_state so it survives reruns. We do NOT pass
        # ``default=`` (Streamlit warns when a key has both state and a default);
        # we (re)initialise on first render / when subset|split changes, then inject
        # the method a Run just produced so it auto-shows on rerun.
        multi_key = "methods_multiselect"
        sig = (str(subset.prepared_dir), split)
        if multi_key not in st.session_state or st.session_state.get("_methods_sig") != sig:
            st.session_state[multi_key] = method_keys[: min(3, len(method_keys))]
            st.session_state["_methods_sig"] = sig
        cur = [k for k in st.session_state[multi_key] if k in method_keys]
        just_ran = st.session_state.pop("last_run_key", None)
        if just_ran and just_ran in method_keys and just_ran not in cur:
            cur.append(just_ran)
        st.session_state[multi_key] = cur
        selected = pick[1].multiselect(
            "Methods to compare", options=method_keys, key=multi_key,
            help="≤5 renders the time-series plot fastest; more is fine for Metrics.",
        )
        feature_options = target_cols or [c for c in pd.read_csv(input_path, nrows=0).columns]
        feature = pick[2].selectbox("Feature", options=feature_options, index=0)
        if method_keys:
            st.caption(f"{len(method_keys)} imputed file(s) discovered for split='{split}'.")
        else:
            st.caption(f"No imputed CSVs in {subset.generated_dir} for split='{split}'. "
                       "Produce some in the **Run** tab.")

        with st.expander("Plot options", expanded=False):
            show_context_lines = st.checkbox(
                "Draw per-method context line", value=False,
                help="Faint dotted line through the full imputed series; off by default for speed.",
            )
            max_bands = st.slider("Max gap bands", 0, 200, 30, step=10,
                                  help="Only the N longest 'truly missing' runs are shaded.")

        needed_for_plot = tuple(dict.fromkeys([time_col, feature]))
        needed_for_metrics = tuple(dict.fromkeys([time_col, *target_cols]))
        input_df_plot = load_csv_subset(input_path, needed_for_plot)
        gt_df_plot: Optional[pd.DataFrame] = None
        gt_name = GT_FILES.get(split)
        gt_path = (subset.prepared_dir / gt_name) if gt_name else None
        if gt_path and gt_path.exists():
            gt_df_plot = load_csv_subset(gt_path, needed_for_plot)
        imputed_dfs_plot = {k: load_csv_subset(available[k].path, needed_for_plot) for k in selected}

        sub_ts, sub_metrics, sub_dist, sub_final, sub_longgap = st.tabs(
            ["Time series", "Metrics", "Distribution", "Final", "Long-gap"]
        )

    sub_run, sub_pipe = tab_run_sec.tabs(["Run experiment", "Pipeline run"])

    # ---- Imputation › Time series -----------------------------------------
    with sub_ts:
        # Optionally overlay the gap-free final(s), aligned to THIS split by time —
        # on the train split this shows the final carries the imputed-train values.
        final_overlays_plot: Dict[str, pd.DataFrame] = {}
        finals_ts = discover_final_files(subset.generated_dir, run)
        if finals_ts:
            overlay_keys = st.multiselect(
                f"Overlay final(s) on the '{split}' split", options=list(finals_ts),
                default=[], key="final_overlay_pick",
                help="Aligns each gap-free *_final.csv to this split by timestamp — "
                     "confirms the final carries this split's imputed values.",
            )
            for k in overlay_keys:
                fdf = load_csv_subset(finals_ts[k], needed_for_plot)
                if time_col in fdf.columns and time_col in input_df_plot.columns:
                    aligned = input_df_plot[[time_col]].merge(
                        fdf[[time_col, feature]], on=time_col, how="left")
                    final_overlays_plot[k] = aligned

        holdout = load_holdout_mask(subset.prepared_dir) if split == "test" else None
        fig = plot_feature_comparison(
            input_df_plot, gt_df_plot, imputed_dfs_plot, feature, time_col,
            holdout_mask=holdout,
            show_context_lines=show_context_lines,
            max_bands=max_bands,
            final_overlays=final_overlays_plot,
        )
        st.plotly_chart(fig, width="stretch")

        with st.expander("How to read this plot"):
            st.markdown("""
This view is **per split** (`test` or `train`). It compares each method's fill
against the *observed* input and, on `test`, the held-out *ground truth*:

- **Dark-blue line + markers** — values *observed* in the input (no imputation applied).
- **Green open diamonds** — ground-truth values at *masked-for-evaluation* positions (only available on the `test` split).
- **Colored ×-markers** — values produced by each selected method at positions that were NaN in the input.
- **Light dotted lines** — the full per-method imputed series for context (off by default; enable in *Plot options*).
- **Gray vertical bands** — positions that are NaN in both input and GT (truly missing — no GT score possible).
- **Dark solid line + filled circles** — an overlaid **final** dataset aligned to this split (enable via *Overlay final(s)*); circles mark its fill at the missing cells. On the `train` split this is how the imputed-train data surfaces.

The **Final** sub-tab shows the same final as one gap-free `train`+`test`
timeline; its `test`-range fill is the *imputed* value, **not** the ground truth.
""")

    # ---- Imputation › Final (imputed train + imputed test) ----------------
    with sub_final:
        finals = discover_final_files(subset.generated_dir, run)
        if not finals:
            st.info(
                "No `*_final.csv` yet. The pipeline writes one via "
                "`scripts/auto_impute.py`, and each WaveStitch+ run writes "
                "`wavestitchplus_<variant>_final.csv`. The final stitches the "
                "**imputed train + imputed test** into one gap-free timeline."
            )
        else:
            fc = st.columns([2, 2])
            fkey = fc[0].selectbox("Final dataset", options=list(finals),
                                   key="final_pick")
            f_idx = feature_options.index(feature) if feature in feature_options else 0
            ffeat = fc[1].selectbox("Feature", options=feature_options, index=f_idx,
                                    key="final_feature")
            fpath = finals[fkey]
            fdf = load_csv_subset(fpath, tuple(dict.fromkeys([time_col, "split", ffeat])))
            # Match the train→test divider to THIS dataset dynamically: the boundary
            # is the selected subset's own last-train timestamp (load_raw_timeline
            # reads its prepared bundle), so it moves per dataset with no baked-in
            # value. Fall back to the final's ``split`` column only if the raw
            # bundle is unavailable (e.g. a standalone final without its bundle).
            raw_df, boundary = load_raw_timeline(subset.prepared_dir, ffeat, time_col)
            if boundary is None and "split" in fdf.columns and fdf["split"].eq("train").any():
                boundary = float(pd.to_numeric(
                    fdf.loc[fdf["split"] == "train", time_col], errors="coerce").max())
            st.plotly_chart(
                plot_final_timeline(fdf, raw_df, ffeat, time_col,
                                    boundary_x=boundary, max_bands=max_bands),
                width="stretch",
            )
            if raw_df is not None and ffeat in raw_df.columns:
                merged = fdf[[time_col]].merge(
                    raw_df[[time_col, ffeat]].rename(columns={ffeat: "_r"}),
                    on=time_col, how="left")
                n_imp = int(merged["_r"].isna().sum())
                residual = int(fdf[ffeat].isna().sum())
                st.caption(
                    f"`{fpath.name}` · {len(fdf):,} rows · **{ffeat}**: "
                    f"{len(fdf) - n_imp:,} observed + {n_imp:,} imputed · "
                    f"{residual} gaps remaining. The `test`-range fill is the "
                    f"**imputed** value, not `test_gt` ground truth."
                )
            else:
                st.caption(f"`{fpath.name}` · {len(fdf):,} rows "
                           "(raw inputs unavailable to mark imputed cells).")

    # ---- Imputation › Metrics ---------------------------------------------
    with sub_metrics:
        if not target_cols:
            st.info("No target columns declared in meta.json.")
        else:
            # Read full target-cols only when this tab is active.
            input_df_full = load_csv_subset(input_path, needed_for_metrics)
            # Schema-drift guard: only score target columns actually present in
            # this split's input (e.g. a bundle regenerated with renamed unit
            # columns while an older imputed file still lags). Prevents a KeyError
            # from crashing the whole tab.
            tcols_present = [c for c in target_cols if c in input_df_full.columns]
            if len(tcols_present) != len(target_cols):
                _absent = [c for c in target_cols if c not in tcols_present]
                st.warning(
                    f"Scoring {len(tcols_present)}/{len(target_cols)} target columns "
                    f"present in this split; absent: {', '.join(_absent[:8])}"
                    + (" …" if len(_absent) > 8 else "")
                )
            input_arr = input_df_full[tcols_present].to_numpy(dtype=float)
            gt_arr: Optional[np.ndarray] = None
            if gt_path and gt_path.exists():
                gt_df_full = load_csv_subset(gt_path, needed_for_metrics)
                if all(c in gt_df_full.columns for c in tcols_present):
                    gt_arr = gt_df_full[tcols_present].to_numpy(dtype=float)
            rows = []
            imputed_full_by_key: Dict[str, np.ndarray] = {}
            for key in selected:
                idf = load_csv_subset(available[key].path, needed_for_metrics)
                if not all(c in idf.columns for c in tcols_present):
                    st.warning(f"Skipping '{key}': missing target column(s) in this split.")
                    continue
                arr = idf[tcols_present].to_numpy(dtype=float)
                if arr.shape[0] != input_arr.shape[0]:
                    # Row count disagrees with the input split — skip rather than
                    # broadcast-error in compute_metrics.
                    st.warning(f"Skipping '{key}': {arr.shape[0]} rows ≠ "
                               f"{input_arr.shape[0]} input rows for split.")
                    continue
                imputed_full_by_key[key] = arr
                m = compute_metrics(input_arr, gt_arr, arr)
                m = {"method": key, **m}
                rows.append(m)
            if rows:
                mdf = pd.DataFrame(rows).set_index("method")
                cols_order = [c for c in ["MAE", "RMSE", "MAPE_%", "fill_rate", "eval_cells", "missing_cells"] if c in mdf.columns]
                mdf = mdf[cols_order]
                st.dataframe(
                    mdf.style.format({
                        "MAE": "{:.4g}", "RMSE": "{:.4g}", "MAPE_%": "{:.2f}",
                        "fill_rate": "{:.1%}",
                    }),
                    width="stretch",
                )
                if st.checkbox("Show per-feature MAE breakdown", value=False, key="show_per_feat_mae"):
                    if gt_arr is None:
                        st.info("No ground truth available for this split.")
                    else:
                        miss = np.isnan(input_arr)
                        eval_mask = miss & ~np.isnan(gt_arr)
                        per_feat = {}
                        for key in imputed_full_by_key:
                            imp_arr = imputed_full_by_key[key]
                            mae_per_col = []
                            for j, col in enumerate(tcols_present):
                                m = eval_mask[:, j] & ~np.isnan(imp_arr[:, j])
                                if m.sum() == 0:
                                    mae_per_col.append(np.nan)
                                else:
                                    mae_per_col.append(float(np.mean(np.abs(imp_arr[m, j] - gt_arr[m, j]))))
                            per_feat[key] = mae_per_col
                        st.dataframe(
                            pd.DataFrame(per_feat, index=tcols_present).style.format("{:.4g}"),
                            width="stretch",
                        )

                # ---- Constraint violations (data-integrity check) ---------
                cvr = latency_violation_rates(input_arr, imputed_full_by_key, tcols_present)
                if cvr is not None:
                    st.markdown(
                        "**Constraint violations** — ordered groups that must hold "
                        "by definition (e.g. `lat50 ≤ … ≤ lat100`). Rate is over "
                        "imputed rows; lower is better. Generative imputers "
                        "(WaveStitch+ v1, PyPOTS) routinely break these; v2's "
                        "monotone projection drives them to 0."
                    )
                    show = cvr.pivot(index="method", columns="group",
                                     values="violation_rate_%")
                    st.dataframe(show.style.format("{:.1f}%"), width="stretch")
                    fig_cvr = go.Figure()
                    for g in cvr["group"].unique():
                        d = cvr[cvr["group"] == g]
                        fig_cvr.add_trace(go.Bar(x=d["method"], y=d["violation_rate_%"], name=g))
                    fig_cvr.update_layout(
                        title="Constraint-violation rate by method (lower = better)",
                        yaxis_title="% of imputed rows violating", height=340,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    )
                    st.plotly_chart(fig_cvr, width="stretch")
            else:
                st.info("Select at least one method to see metrics.")

    # ---- Imputation › Distribution ----------------------------------------
    with sub_dist:
        fig = plot_distribution(input_df_plot, imputed_dfs_plot, feature, gt_df=gt_df_plot)
        st.plotly_chart(fig, width="stretch")

    # ---- Imputation › Long-gap regime -------------------------------------
    with sub_longgap:
        render_long_gap_tab(subset)

    # ---- Run › Run experiment ---------------------------------------------
    with sub_run:
        st.write("Invoke a runner on the **currently selected subset** to add new imputed CSVs to "
                 f"`{subset.generated_dir}`. The dashboard will pick them up after the run finishes.")
        _impy, _custom = _impute_python()
        if _custom:
            st.caption(f"Imputation runs via `{' '.join(_impy)}` "
                       "(`DATAOPS_IMPUTE_PYTHON` / `DATAOPS_IMPUTE_CONDA_ENV`).")
        else:
            st.caption("Dependency-free here: **darts** interpolation, **imputegap** statistics "
                       "(interpolation / mean / min / zero), and **WaveStitch+ v2** when a v1 output "
                       "exists to anchor. For methods needing a trained model / GPU libs "
                       "(WaveStitch+ v1 & harpoon, PyPOTS, darts-kalman, imputegap cdrec/brits/…), "
                       "tick **Run in Docker image** below, or set "
                       "`DATAOPS_IMPUTE_CONDA_ENV=<env-with-deps>` / `DATAOPS_IMPUTE_PYTHON=…`.")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            lib = st.selectbox("Library", options=list(RUNNERS.keys()), index=0)
        with c2:
            method = st.selectbox("Method", options=RUNNER_METHODS[lib], index=0)
        with c3:
            splits = st.multiselect("Inputs", options=["train", "test"], default=["train", "test"])

        extra_args: List[str] = []
        if lib == "pypots":
            ce1, ce2 = st.columns(2)
            epochs = ce1.number_input("epochs", min_value=1, max_value=2000, value=50, step=10)
            batch = ce2.number_input("batch size", min_value=1, max_value=64, value=1)
            extra_args = ["--epochs", str(int(epochs)), "--batch-size", str(int(batch))]
            weights = st.radio(
                "Model weights", options=["train fresh", "save", "load", "load + save"],
                index=0, horizontal=True,
                help="save: persist trained weights · load: reuse a saved checkpoint and "
                     "skip training when one matches (else train) · load + save: reuse if "
                     "present, else train and save. Checkpoints are per method + window shape.",
            )
            if weights != "train fresh":
                model_dir = st.text_input(
                    "Model dir (--model-path)",
                    value=str(subset.prepared_dir / "pypots_models"),
                    key="pypots_model_dir",
                )
                extra_args += ["--model-path", model_dir]
                if "save" in weights:
                    extra_args.append("--save-model")
                if "load" in weights:
                    extra_args.append("--load-model")
        elif lib == "wavestitchplus":
            st.warning(
                "WaveStitch+ **retrains** on run — it overwrites this subset's saved "
                f"model and `prepared_{subset.name}/` artifacts (scaler, train_imputed, "
                "training_completed.json). Unlike the read-only baselines, a run mutates "
                "the prepared dir in place.",
                icon="⚠️",
            )
            device = st.radio(
                "Device", options=["auto", "cpu", "gpu"], index=0, horizontal=True,
                help="cpu forces CUDA off; auto/gpu use the GPU when available "
                     "(falls back to CPU if none). WaveStitch+ publishes requested split outputs.",
            )
            fast = st.checkbox(
                "Fast smoke test (tiny hyperparams)", value=False,
                help="em=1, epochs/em=5, ddim=5, repaint=1 — finishes in ~1–2 min on CPU "
                     "(longer for bigger subsets) to verify the run end-to-end. Not for quality.",
            )
            cw1, cw2, cw3 = st.columns(3)
            em_iters = cw1.number_input("EM iterations", min_value=1, max_value=20, value=3,
                                        disabled=fast)
            epochs_em = cw2.number_input("epochs / EM", min_value=5, max_value=500, value=50,
                                         step=5, disabled=fast)
            ddim = cw3.number_input("DDIM steps", min_value=5, max_value=200, value=30,
                                    step=5, disabled=fast)
            extra_args = ["--device", device]
            if fast:
                extra_args.append("--fast")
            else:
                extra_args += [
                    "--em-iterations", str(int(em_iters)),
                    "--epochs-per-em", str(int(epochs_em)),
                    "--ddim-steps", str(int(ddim)),
                ]
            if device != "cpu" and not fast:
                st.caption("⚠ GPU/auto runs use CUDA only if this host has a GPU; "
                           "training on CPU can take many minutes.")
        elif lib == "wavestitchplus_v2":
            st.info(
                "WaveStitch+ **v2** reuses the existing trained model (synthesis only, "
                "no retrain) and **locally anchors** the diffusion output to a "
                "context-aware interpolation prior. It reads the saved model for this "
                "subset and writes `wavestitchplus_v2_<split>_imputed.csv` (or "
                "`wavestitchplus_v2_tuned_<split>_imputed.csv` for tuned mode) — "
                "the prepared dir is not mutated.",
                icon="🧭",
            )
            if method == "tuned":
                st.caption(
                    "`tuned` scans prior/tau/hard-prior on this subset's `test_gt.csv`; "
                    "use it as an evaluation preset, not an unsupervised production default."
                )
            device = st.radio(
                "Device", options=["auto", "cpu", "gpu"], index=0, horizontal=True,
                key="wsp_v2_device",
                help="cpu forces CUDA off; auto/gpu use the GPU when available.",
            )
            cv1, cv2, cv3, cv4 = st.columns(4)
            prior = cv1.selectbox(
                "Prior", options=["auto", "nearest", "linear"], index=0,
                help="interpolation prior on concat(train,test); 'auto' picks nearest vs "
                     "linear per column (unsupervised) — this is what lets v2 beat the "
                     "single nearest/linear baselines.",
            )
            ddim = cv2.number_input("DDIM steps", min_value=5, max_value=200, value=50,
                                    step=5, help="synthesis steps for the diffusion output")
            tau = cv3.number_input(
                "tau (prior reach)", min_value=1.0, max_value=200.0, value=8.0, step=1.0,
                help="prior-weight decay length; larger = trust the prior deeper into gaps. "
                     "Conservative by default — on these holdouts the diffusion is weaker "
                     "than nearest even at depth.",
            )
            hard_prior = cv4.number_input(
                "hard-prior dist", min_value=0, max_value=64, value=32, step=1,
                help="cells within this distance of an observation follow the prior exactly; "
                     "diffusion blends in only beyond it. Default 32 (≈ receptive field) makes "
                     "v2 ≈ the auto interpolation on natural holdouts; lower it to expose more "
                     "diffusion in long gaps.",
            )
            extra_args = [
                "--device", device,
                "--prior", prior,
                "--ddim-steps", str(int(ddim)),
                "--tau", str(float(tau)),
                "--hard-prior", str(int(hard_prior)),
            ]

        elif lib == "wavestitchplus_harpoon":
            st.info(
                "WaveStitch+ **HARPOON** runs inference-time manifold guidance "
                "on the pre-trained model (no retrain): value bounds plus local "
                "prior, smoothness, and latency-percentile monotonicity penalties.",
                icon="🎯",
            )
            device = st.radio(
                "Device", options=["auto", "cpu", "gpu"], index=0, horizontal=True,
                key="harpoon_device",
                help="cpu forces CUDA off; auto/gpu use the GPU when available.",
            )
            ch1, ch2, ch3, ch4 = st.columns(4)
            bound_lambda = ch1.number_input(
                "bound_lambda", min_value=0.0, max_value=10.0, value=0.3, step=0.05,
                help="weight on the manifold-bound penalty in the guidance loss; "
                     "0 falls back to vanilla v1.",
            )
            ddim = ch2.number_input(
                "DDIM steps", min_value=5, max_value=200, value=50, step=5,
                key="harpoon_ddim",
            )
            auto_ub_q = ch3.number_input(
                "auto_ub_q", min_value=0.5, max_value=1.0, value=0.99, step=0.01,
                format="%.2f",
                help="upper-bound observed-data quantile.",
            )
            auto_ub_pad = ch4.number_input(
                "auto_ub_pad", min_value=0.0, max_value=1.0, value=0.05, step=0.01,
                format="%.2f",
                help="multiplicative padding above the quantile.",
            )
            hard_pos = st.checkbox(
                "hard-project positive (final pass: floor at pos_eps)",
                value=False, key="harpoon_hard_pos",
            )
            hp1, hp2, hp3, hp4 = st.columns(4)
            prior_lambda = hp1.number_input(
                "prior_lambda", min_value=0.0, max_value=10.0, value=0.25, step=0.05,
                help="soft guidance toward a context-aware nearest/linear prior.",
            )
            prior_method = hp2.selectbox(
                "prior", options=["nearest", "linear"], index=0,
                help="local prior used inside HARPOON guidance.",
            )
            smooth_lambda = hp3.number_input(
                "smooth_lambda", min_value=0.0, max_value=10.0, value=0.02, step=0.01,
                help="soft temporal smoothness guidance.",
            )
            monotone_lambda = hp4.number_input(
                "monotone_lambda", min_value=0.0, max_value=10.0, value=0.05, step=0.01,
                help="soft monotonicity guidance for latency percentile groups.",
            )
            extra_args = [
                "--device", device,
                "--ddim-steps", str(int(ddim)),
                "--bound-lambda", str(float(bound_lambda)),
                "--prior-lambda", str(float(prior_lambda)),
                "--prior-method", prior_method,
                "--smooth-lambda", str(float(smooth_lambda)),
                "--monotone-lambda", str(float(monotone_lambda)),
                "--auto-ub-q", str(float(auto_ub_q)),
                "--auto-ub-pad", str(float(auto_ub_pad)),
            ]
            if hard_pos:
                extra_args.append("--hard-project-positive")

        # Persisted result of the previous run (survives the st.rerun() below
        # that re-renders the sidebar with the newly-produced method discovered).
        # ---- Execution backend: Docker image vs local/conda subprocess --------
        use_docker = st.checkbox(
            "Run in Docker image", value=bool(os.environ.get("DATAOPS_IMPUTE_DOCKER")),
            help="Run the method inside its prebuilt image — no heavy deps needed "
                 "in the dashboard env. The image must be built locally.",
        )
        docker_image: Optional[str] = None
        gpu = False
        if use_docker:
            di1, di2 = st.columns([3, 1])
            docker_image = di1.text_input(
                "Docker image", value=_runner_image(lib) or "",
                key=f"docker_image_{lib}",
                help="Your local image tag — run `docker images` to find it. Default comes "
                     "from each app's build_image.sh; override here or via "
                     f"DATAOPS_IMPUTE_IMAGE_{lib.upper()}.",
            ) or None
            gpu = di2.checkbox("GPU", value=bool(os.environ.get("DATAOPS_IMPUTE_GPU")),
                               help="`--gpus all` (needs nvidia-docker).")

        prev = st.session_state.pop("last_run_result", None)
        if prev is not None:
            (st.success if prev["rc"] == 0 else st.error)(prev["msg"])
            with st.expander("Run log", expanded=(prev["rc"] != 0)):
                st.code(prev["log"], language="text")

        run_btn = st.button("Run on selected subset", type="primary",
                            disabled=(len(splits) == 0))
        if run_btn:
            where = f"Docker `{docker_image or _runner_image(lib)}`" if use_docker else "subprocess"
            with st.spinner(f"Running {lib}/{method} on {subset.label} "
                            f"({', '.join(splits)}) via {where}..."):
                rc, out = run_experiment(lib, method, subset, splits, extra_args,
                                         use_docker=use_docker, gpu=gpu, image=docker_image)
            st.session_state["last_run_result"] = {
                "rc": rc,
                "msg": (f"{lib}/{method} done." if rc == 0
                        else f"{lib}/{method} exited with code {rc}."),
                "log": out[-8000:] if out else "(no output)",
            }
            if rc == 0:
                # Drop cached lookups and rerun so the sidebar's multiselect picks
                # up the new file in the same click; auto-select the produced key.
                st.session_state["last_run_key"] = _produced_key(lib, method)
                # _discover_imputed_files_by_dir is no longer cached (always fresh);
                # only the CSV-content caches need clearing to pick up the new file.
                load_csv.clear()
                load_csv_subset.clear()
            st.rerun()

    # ---- Run › Pipeline run (S3 / Airflow source comparison) -------------
    with sub_pipe:
        st.write(
            "Compare a DAG-pipeline run against its **original source**. "
            "Pulls artifacts directly from the SeaweedFS S3 endpoint that the "
            "Airflow stack publishes."
        )
        with st.expander("S3 connection", expanded=False):
            ep   = st.text_input("Endpoint",  value=PIPELINE_S3_DEFAULTS["endpoint"], key="pipe_ep")
            buc  = st.text_input("Bucket",    value=PIPELINE_S3_DEFAULTS["bucket"],   key="pipe_buc")
            akey = st.text_input("Access key",value=PIPELINE_S3_DEFAULTS["access"],   key="pipe_ak")
            skey = st.text_input("Secret key",value=PIPELINE_S3_DEFAULTS["secret"],   key="pipe_sk", type="password")

        if st.button("Refresh runs", key="pipe_refresh"):
            _s3_list_runs.clear()
            _s3_get_csv.clear()
            _s3_get_json.clear()
            _s3_resolve_raw_key.clear()

        try:
            runs_map = _s3_list_runs(ep, buc, akey, skey)
        except Exception as e:
            st.error(f"Could not list runs from `{ep}` / bucket `{buc}`: {e}")
            runs_map = {}

        if not runs_map:
            st.info(f"No DAG runs found under `cleaned/` in bucket `{buc}`. "
                    "Trigger the Airflow pipeline once, then click *Refresh runs*.")
        else:
            cdata, crun = st.columns([1, 2])
            with cdata:
                dataset = st.selectbox("Dataset", options=sorted(runs_map), key="pipe_dataset")
            with crun:
                run_id = st.selectbox("Run ID", options=runs_map[dataset], key="pipe_run_id")

            cleaned_key  = f"cleaned/{dataset}/{run_id}/cleaned.csv"
            report_key   = f"cleaned/{dataset}/{run_id}/cleaning_report.json"
            curated_key  = f"curated/{dataset}/{run_id}/data.csv"
            curated_meta = f"curated/{dataset}/{run_id}/meta.json"

            cleaned_df  = _s3_get_csv(ep, buc, akey, skey, cleaned_key)
            curated_df  = _s3_get_csv(ep, buc, akey, skey, curated_key)
            report      = _s3_get_json(ep, buc, akey, skey, report_key) or {}
            curated_meta_dict = _s3_get_json(ep, buc, akey, skey, curated_meta) or {}

            raw_key = _s3_resolve_raw_key(ep, buc, akey, skey, dataset)
            raw_df  = _s3_get_csv(ep, buc, akey, skey, raw_key) if raw_key else None

            if cleaned_df is None:
                st.error(f"Missing `{cleaned_key}`. Cannot compare.")
            elif raw_df is None:
                st.error(f"Could not resolve a raw source CSV for dataset `{dataset}`. "
                         "Place the input under `test/<file>.csv` or set N2N_INPUT_KEY explicitly.")
            else:
                # ---- High-level summary -----------------------------------
                summary_cols = st.columns(4)
                summary_cols[0].metric("Raw rows",     f"{len(raw_df):,}")
                summary_cols[1].metric("Soft-cleaned rows", f"{len(cleaned_df):,}",
                                       delta=f"{len(cleaned_df)-len(raw_df):+d}")
                summary_cols[2].metric("Curated rows",
                                       f"{len(curated_df):,}" if curated_df is not None else "—")
                imputer = (report.get("ts_imputation") or {}).get("method", "—")
                summary_cols[3].metric("Imputer", imputer)
                st.caption(f"Raw key: `{raw_key}` · Soft-cleaned: `{cleaned_key}` · "
                           f"Curated: `{curated_key}`")

                # ---- Column picker + plot ---------------------------------
                numeric_cols_raw = [c for c in raw_df.columns
                                    if pd.api.types.is_numeric_dtype(raw_df[c])]
                numeric_cols_cleaned = [c for c in cleaned_df.columns
                                        if pd.api.types.is_numeric_dtype(cleaned_df[c])]
                shared = [c for c in numeric_cols_raw if c in numeric_cols_cleaned]
                if not shared:
                    st.warning("No numeric columns in common between raw and soft-cleaned data.")
                else:
                    time_guess = (curated_meta_dict.get("time_col") or
                                  curated_meta_dict.get("timestamp_column") or
                                  raw_df.columns[0])
                    feature_pipe = st.selectbox(
                        "Feature",
                        options=[c for c in shared if c != time_guess] or shared,
                        index=0, key="pipe_feature",
                    )
                    fig = _pipeline_compare_plot(
                        raw_df, cleaned_df, curated_df, feature_pipe, time_guess
                    )
                    st.plotly_chart(fig, width="stretch")

                    # Per-column delta vs raw
                    diff_rows = []
                    for c in shared:
                        r = pd.to_numeric(raw_df[c], errors="coerce")
                        cl = pd.to_numeric(cleaned_df[c], errors="coerce")
                        # Length may differ if dedup removed rows; truncate to min.
                        n = min(len(r), len(cl))
                        r2, cl2 = r.iloc[:n], cl.iloc[:n]
                        diff = (cl2 - r2).dropna()
                        changed = int((cl2 != r2).fillna(False).sum())
                        diff_rows.append({
                            "column": c,
                            "raw_NaN": int(r.isna().sum()),
                            "soft_cleaned_NaN": int(cl.isna().sum()),
                            "cells_changed": changed,
                            "mean_delta": float(diff.mean()) if len(diff) else float("nan"),
                            "max_abs_delta": float(diff.abs().max()) if len(diff) else float("nan"),
                        })
                    st.subheader("Per-column impact (soft-cleaned vs raw)")
                    diff_df = pd.DataFrame(diff_rows).set_index("column")
                    st.dataframe(
                        diff_df.style.format({
                            "raw_NaN": "{:,}", "soft_cleaned_NaN": "{:,}",
                            "cells_changed": "{:,}",
                            "mean_delta": "{:.4g}", "max_abs_delta": "{:.4g}",
                        }),
                        width="stretch",
                    )

                # ---- Cleaning report --------------------------------------
                with st.expander("Cleaning report (raw)", expanded=False):
                    st.json(report)


if __name__ == "__main__":
    main()
