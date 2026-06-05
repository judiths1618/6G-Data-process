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
# All demo results live under experiments/ (local-demo convention). The canonical
# imputation tree is experiments/EUR/{prepared_<subset>, generated_<subset>}.
DEFAULT_WORK_ROOT = REPO_ROOT / "experiments" / "EUR"

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
    "wavestitchplus_v2": ["anchored"],
    "wavestitchplus_harpoon": ["harpoon"],
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


@st.cache_data(show_spinner=False)
def discover_subsets(work_root: Path) -> List[Subset]:
    """Discover ``prepared_<subset>`` folders under ``work_root``.

    Accepts either layout: ``work_root/<dataset>/prepared_<subset>`` (a tree of
    dataset groups) or ``work_root/prepared_<subset>`` (work_root *is* the group,
    e.g. ``experiments/EUR``). Both are scanned so the default points straight at
    the consolidated ``experiments/EUR`` tree.
    """
    out: List[Subset] = []
    if not work_root.exists():
        return out
    candidates = [work_root] + sorted(p for p in work_root.iterdir() if p.is_dir())
    seen: set = set()
    for ds_dir in candidates:
        for prep in sorted(ds_dir.glob("prepared_*")):
            if prep in seen or not (prep / "meta.json").exists():
                continue
            seen.add(prep)
            name = prep.name.removeprefix("prepared_")
            gen = ds_dir / f"generated_{name}"
            out.append(Subset(
                dataset=ds_dir.name,
                name=name,
                prepared_dir=prep,
                generated_dir=gen if gen.exists() else None,
            ))
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


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def load_meta(prepared_dir: Path) -> dict:
    with (prepared_dir / "meta.json").open() as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_csv_subset(path: Path, columns: Tuple[str, ...]) -> pd.DataFrame:
    """Read only the columns we actually plot — much faster for wide CSVs."""
    try:
        return pd.read_csv(path, usecols=list(columns))
    except ValueError:
        # Fall back to full read if a requested column isn't in the file.
        df = pd.read_csv(path)
        keep = [c for c in columns if c in df.columns]
        return df[keep]


@st.cache_data(show_spinner=False)
def load_holdout_mask(prepared_dir: Path) -> Optional[np.ndarray]:
    p = prepared_dir / "eval_holdout_mask.npy"
    return np.load(p) if p.exists() else None


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
                            max_bands: int = 30) -> go.Figure:
    fig = go.Figure()

    x = input_df[time_col] if time_col in input_df.columns else input_df.index
    if time_col == "time":
        # Treat unix-seconds as a datetime axis when it looks like one.
        try:
            xv = pd.to_numeric(x, errors="coerce")
            if xv.notna().all() and xv.min() > 1_000_000_000:
                x = pd.to_datetime(xv, unit="s")
        except Exception:
            pass

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

    fig.update_layout(
        height=520,
        margin=dict(l=40, r=20, t=40, b=40),
        title=f"{feature}  —  observed vs imputed",
        xaxis_title=time_col,
        yaxis_title=feature,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="closest",
        uirevision="ts-plot",
    )
    return fig


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
    wsp2 = v2.anchor_blend(ti, diff, near, tcols, tau=20.0, hard_prior=8,
                           has_left_context=train is not None)
    preds = {"nearest": near, "linear": lin, "wsp_v1": diff, "wsp_v2": wsp2}
    return ti, gt, preds, tcols


def plot_long_gap_feature(ti: pd.DataFrame, gt: pd.DataFrame,
                          preds: Dict[str, pd.DataFrame], feature: str,
                          time_col: str) -> go.Figure:
    """Per-method reconstruction of one feature over the masked long gaps."""
    x = ti[time_col] if time_col in ti.columns else pd.Series(np.arange(len(ti)))
    if time_col == "time":
        try:
            xv = pd.to_numeric(x, errors="coerce")
            if xv.notna().all() and xv.min() > 1_000_000_000:
                x = pd.to_datetime(xv, unit="s")
        except Exception:
            pass
    x = np.asarray(x)
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
    default_idx = gap_opts.index(32) if 32 in gap_opts else len(gap_opts) - 1
    sel_L = st.select_slider("Gap length for the depth view", options=gap_opts, value=gap_opts[default_idx])
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

def run_experiment(library: str,
                   method: str,
                   subset: Subset,
                   splits: List[str],
                   extra_args: List[str]) -> Tuple[int, str]:
    """Invoke a runner script and return (exit_code, combined_output)."""
    runner = RUNNERS[library]
    if not runner.exists():
        return 127, f"runner not found: {runner}"
    cmd = [
        sys.executable, str(runner),
        "--prepared-dir", str(subset.prepared_dir),
        "--output-dir", str(subset.generated_dir or (subset.prepared_dir.parent / f"generated_{subset.name}")),
        "--method", method,
        "--inputs", *splits,
    ] + extra_args
    env = os.environ.copy()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or "")


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


def _pipeline_compare_plot(raw: pd.DataFrame, cleaned: pd.DataFrame,
                           curated: Optional[pd.DataFrame], column: str,
                           time_col: str) -> go.Figure:
    """Three-line plot: raw vs cleaned vs curated for one numeric column."""
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
        x=_x(cleaned), y=cleaned[column].astype(float),
        name="cleaned", mode="lines",
        line=dict(color="#ff7f0e", width=1.4),
    ))
    if curated is not None and column in curated.columns:
        fig.add_trace(go.Scattergl(
            x=_x(curated), y=curated[column].astype(float),
            name="curated", mode="lines",
            line=dict(color="#2ca02c", width=1, dash="dot"),
        ))
    fig.update_layout(
        title=f"{column} — raw vs cleaned vs curated",
        xaxis_title=x_label,
        yaxis_title=column,
        height=420, margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="6G-DALI Imputation Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Time-series imputation — method comparison")
    st.caption("Compare Darts / ImputeGAP / PyPOTS / WaveStitch+ outputs on the 6G-DALI EUR datasets.")

    # ---- Sidebar: dataset / subset / split ---------------------------------
    work_root = Path(st.sidebar.text_input(
        "Work root", value=str(DEFAULT_WORK_ROOT),
        help="Folder containing `<dataset>/prepared_<subset>/` and `<dataset>/generated_<subset>/`.",
    ))
    subsets = discover_subsets(work_root)
    if not subsets:
        st.error(f"No `prepared_*` folders found under {work_root}.")
        return

    labels = [s.label for s in subsets]
    chosen = st.sidebar.selectbox("Subset", labels, index=0)
    subset = subsets[labels.index(chosen)]

    split = st.sidebar.radio("Split", options=["test", "train"], horizontal=True, index=0)

    meta = load_meta(subset.prepared_dir)
    target_cols: List[str] = list(meta.get("target_cols", []))
    time_col: str = meta.get("time_col", "time")

    input_path = subset.prepared_dir / INPUT_FILES[split]
    if not input_path.exists():
        st.error(f"Missing input file: {input_path}")
        return

    # ---- Sidebar: imputed-method picker -----------------------------------
    available = discover_imputed_files(subset, split)
    method_keys = sorted(available.keys())
    if not method_keys:
        st.sidebar.warning(f"No imputed CSVs found in {subset.generated_dir} for split='{split}'.")
    else:
        st.sidebar.caption(f"{len(method_keys)} imputed file(s) discovered.")

    # Selection lives in session_state so it survives reruns. We do NOT pass
    # ``default=`` to the widget (Streamlit warns when a key both has session
    # state and a default). Instead, we (re)initialise the state on first render
    # or when the subset/split changes, then optionally inject the method that a
    # Run just produced so it auto-shows on rerun.
    multi_key = "methods_multiselect"
    sig = (str(subset.prepared_dir), split)
    if multi_key not in st.session_state or st.session_state.get("_methods_sig") != sig:
        st.session_state[multi_key] = method_keys[: min(3, len(method_keys))]
        st.session_state["_methods_sig"] = sig
    # Drop any stale selections that aren't valid for the current method_keys.
    cur = [k for k in st.session_state[multi_key] if k in method_keys]
    just_ran = st.session_state.pop("last_run_key", None)
    if just_ran and just_ran in method_keys and just_ran not in cur:
        cur.append(just_ran)
    st.session_state[multi_key] = cur
    selected = st.sidebar.multiselect(
        "Methods to compare",
        options=method_keys,
        key=multi_key,
        help="Loading 8+ methods is fine for the metrics tab but the time-series "
             "plot renders fastest with ≤5.",
    )

    # ---- Sidebar: feature picker ------------------------------------------
    feature_options = target_cols or [c for c in pd.read_csv(input_path, nrows=0).columns]
    feature = st.sidebar.selectbox(
        "Feature", options=feature_options, index=0,
    )

    # ---- Sidebar: plot tweaks ---------------------------------------------
    with st.sidebar.expander("Plot options", expanded=False):
        show_context_lines = st.checkbox(
            "Draw per-method context line", value=False,
            help="Faint dotted line through the full imputed series. Off by default to speed up rendering.",
        )
        max_bands = st.slider("Max gap bands", 0, 200, 30, step=10,
                              help="Only the N longest 'truly missing' runs are shaded.")

    # ---- Targeted column reads (only the cols we'll plot/score) -----------
    needed_for_plot = tuple(dict.fromkeys([time_col, feature]))
    needed_for_metrics = tuple(dict.fromkeys([time_col, *target_cols]))

    input_df_plot = load_csv_subset(input_path, needed_for_plot)
    gt_df_plot: Optional[pd.DataFrame] = None
    gt_name = GT_FILES.get(split)
    gt_path = (subset.prepared_dir / gt_name) if gt_name else None
    if gt_path and gt_path.exists():
        gt_df_plot = load_csv_subset(gt_path, needed_for_plot)

    imputed_dfs_plot = {k: load_csv_subset(available[k].path, needed_for_plot) for k in selected}

    # ---- Header info -------------------------------------------------------
    cols = st.columns(4)
    cols[0].metric("Subset", subset.label)
    cols[1].metric("Split", split)
    miss_feature = int(input_df_plot[feature].isna().sum()) if feature in input_df_plot.columns else 0
    cols[2].metric(f"NaN in '{feature}'", f"{miss_feature:,}",
                   delta=f"{miss_feature / max(len(input_df_plot), 1) * 100:.1f}%", delta_color="off")
    cols[3].metric("Methods loaded", str(len(selected)))

    # ---- Tabs --------------------------------------------------------------
    tab_ts, tab_metrics, tab_dist, tab_longgap, tab_run, tab_pipeline = st.tabs(
        ["Time series", "Metrics", "Distribution", "Long-gap", "Run experiment", "Pipeline run"]
    )

    # ---- Tab 1: Time series ------------------------------------------------
    with tab_ts:
        holdout = load_holdout_mask(subset.prepared_dir) if split == "test" else None
        fig = plot_feature_comparison(
            input_df_plot, gt_df_plot, imputed_dfs_plot, feature, time_col,
            holdout_mask=holdout,
            show_context_lines=show_context_lines,
            max_bands=max_bands,
        )
        st.plotly_chart(fig, width="stretch")

        with st.expander("How to read this plot"):
            st.markdown("""
- **Dark-blue line + markers** — values *observed* in the input (no imputation applied).
- **Green open diamonds** — ground-truth values at *masked-for-evaluation* positions (only available on the `test` split).
- **Colored ×-markers** — values produced by each selected method at positions that were NaN in the input.
- **Light dotted lines** — the full per-method imputed series for context (off by default; enable in *Plot options*).
- **Gray vertical bands** — positions that are NaN in both input and GT (truly missing — no GT score possible).
""")

    # ---- Tab 2: Metrics ----------------------------------------------------
    with tab_metrics:
        if not target_cols:
            st.info("No target columns declared in meta.json.")
        else:
            # Read full target-cols only when this tab is active.
            input_df_full = load_csv_subset(input_path, needed_for_metrics)
            input_arr = input_df_full[target_cols].to_numpy(dtype=float)
            gt_arr: Optional[np.ndarray] = None
            if gt_path and gt_path.exists():
                gt_df_full = load_csv_subset(gt_path, needed_for_metrics)
                gt_arr = gt_df_full[target_cols].to_numpy(dtype=float)
            rows = []
            imputed_full_by_key: Dict[str, np.ndarray] = {}
            for key in selected:
                arr = load_csv_subset(available[key].path, needed_for_metrics)[target_cols].to_numpy(dtype=float)
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
                            for j, col in enumerate(target_cols):
                                m = eval_mask[:, j] & ~np.isnan(imp_arr[:, j])
                                if m.sum() == 0:
                                    mae_per_col.append(np.nan)
                                else:
                                    mae_per_col.append(float(np.mean(np.abs(imp_arr[m, j] - gt_arr[m, j]))))
                            per_feat[key] = mae_per_col
                        st.dataframe(
                            pd.DataFrame(per_feat, index=target_cols).style.format("{:.4g}"),
                            width="stretch",
                        )

                # ---- Constraint violations (data-integrity check) ---------
                cvr = latency_violation_rates(input_arr, imputed_full_by_key, target_cols)
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

    # ---- Tab 3: Distribution -----------------------------------------------
    with tab_dist:
        fig = plot_distribution(input_df_plot, imputed_dfs_plot, feature, gt_df=gt_df_plot)
        st.plotly_chart(fig, width="stretch")

    # ---- Tab 4: Long-gap regime -------------------------------------------
    with tab_longgap:
        render_long_gap_tab(subset)

    # ---- Tab 5: Run experiment --------------------------------------------
    with tab_run:
        st.write("Invoke a runner on the **currently selected subset** to add new imputed CSVs to "
                 f"`{subset.generated_dir}`. The dashboard will pick them up after the run finishes.")
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
                "subset and writes `wavestitchplus_v2_<split>_imputed.csv` — the prepared "
                "dir is not mutated.",
                icon="🧭",
            )
            device = st.radio(
                "Device", options=["auto", "cpu", "gpu"], index=0, horizontal=True,
                key="wsp_v2_device",
                help="cpu forces CUDA off; auto/gpu use the GPU when available.",
            )
            cv1, cv2, cv3, cv4 = st.columns(4)
            prior = cv1.selectbox(
                "Prior", options=["nearest", "linear"], index=0,
                help="interpolation prior on concat(train,test); 'nearest' matches the "
                     "strongest darts baseline.",
            )
            ddim = cv2.number_input("DDIM steps", min_value=5, max_value=200, value=50,
                                    step=5, help="synthesis steps for the diffusion output")
            tau = cv3.number_input(
                "tau (prior reach)", min_value=1.0, max_value=200.0, value=20.0, step=1.0,
                help="prior-weight decay length; larger = trust the prior deeper into gaps. "
                     "The smooth 6G series favour a large tau.",
            )
            hard_prior = cv4.number_input(
                "hard-prior dist", min_value=0, max_value=64, value=8, step=1,
                help="cells within this distance of an observation follow the prior exactly "
                     "(where interpolation is near-optimal and the diffusion adds only noise).",
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
                "WaveStitch+ **HARPOON** runs inference-time *manifold-bound* "
                "guidance on the pre-trained model (no retrain). The penalty "
                "is auto-derived from observed data: `[pos_eps, "
                "quantile(observed, auto_ub_q) * (1 + auto_ub_pad)]` per target.",
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
            extra_args = [
                "--device", device,
                "--ddim-steps", str(int(ddim)),
                "--bound-lambda", str(float(bound_lambda)),
                "--auto-ub-q", str(float(auto_ub_q)),
                "--auto-ub-pad", str(float(auto_ub_pad)),
            ]
            if hard_pos:
                extra_args.append("--hard-project-positive")

        # Persisted result of the previous run (survives the st.rerun() below
        # that re-renders the sidebar with the newly-produced method discovered).
        prev = st.session_state.pop("last_run_result", None)
        if prev is not None:
            (st.success if prev["rc"] == 0 else st.error)(prev["msg"])
            with st.expander("Run log", expanded=(prev["rc"] != 0)):
                st.code(prev["log"], language="text")

        run_btn = st.button("Run on selected subset", type="primary",
                            disabled=(len(splits) == 0))
        if run_btn:
            with st.spinner(f"Running {lib}/{method} on {subset.label} ({', '.join(splits)})..."):
                rc, out = run_experiment(lib, method, subset, splits, extra_args)
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
                _discover_imputed_files_by_dir.clear()
                load_csv.clear()
                load_csv_subset.clear()
            st.rerun()

    # ---- Tab 5: Pipeline run (compare with original source) ---------------
    with tab_pipeline:
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
                summary_cols[1].metric("Cleaned rows", f"{len(cleaned_df):,}",
                                       delta=f"{len(cleaned_df)-len(raw_df):+d}")
                summary_cols[2].metric("Curated rows",
                                       f"{len(curated_df):,}" if curated_df is not None else "—")
                imputer = (report.get("ts_imputation") or {}).get("method", "—")
                summary_cols[3].metric("Imputer", imputer)
                st.caption(f"Raw key: `{raw_key}` · Cleaned: `{cleaned_key}` · "
                           f"Curated: `{curated_key}`")

                # ---- Column picker + plot ---------------------------------
                numeric_cols_raw = [c for c in raw_df.columns
                                    if pd.api.types.is_numeric_dtype(raw_df[c])]
                numeric_cols_cleaned = [c for c in cleaned_df.columns
                                        if pd.api.types.is_numeric_dtype(cleaned_df[c])]
                shared = [c for c in numeric_cols_raw if c in numeric_cols_cleaned]
                if not shared:
                    st.warning("No numeric columns in common between raw and cleaned.")
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
                            "cleaned_NaN": int(cl.isna().sum()),
                            "cells_changed": changed,
                            "mean_delta": float(diff.mean()) if len(diff) else float("nan"),
                            "max_abs_delta": float(diff.abs().max()) if len(diff) else float("nan"),
                        })
                    st.subheader("Per-column impact (cleaned vs raw)")
                    diff_df = pd.DataFrame(diff_rows).set_index("column")
                    st.dataframe(
                        diff_df.style.format({
                            "raw_NaN": "{:,}", "cleaned_NaN": "{:,}",
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
