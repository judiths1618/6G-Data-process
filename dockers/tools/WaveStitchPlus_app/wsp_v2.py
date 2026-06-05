#!/usr/bin/env python3
"""WaveStitch+ v2 — inference-time *local anchoring* of the diffusion output.

Why v2 exists
-------------
Diagnostics on the 6G perf datasets showed WaveStitch+ v1 scores far worse than
trivial interpolation on the point-wise holdout metric (amf: v1 MAE≈7.7k vs
darts_nearest≈1.2k). Two structural causes, both fixable *without retraining*:

1. **No local anchor.** At a held-out cell the reverse-diffusion samples a
   distributionally-plausible value that is essentially uncorrelated with the
   truth (measured corr≈0), so its error ≈ the marginal spread. These series
   are smooth and strongly autocorrelated, so a held-out point sits very close
   to its observed neighbours — exactly what interpolation nails and a free
   generative sample misses.
2. **No train context at synthesis time.** v1 synthesises the test split in
   isolation, so leading/short test gaps have no left-context to lean on (this
   is also why the very first test cell was wildly off).

v2 keeps the WaveStitch+ diffusion output but *anchors* it to a context-aware
interpolation prior, blending the two per cell by how far the cell is from the
nearest observed value:

    out = w(d)·prior + (1 - w(d))·diffusion,   w(d) = exp(-(d-1)/tau), d≥1

Cells adjacent to an observation (d=1 → w≈1) follow the prior (where it is
near-optimal); cells deep inside a long gap (large d → w→0) fall back to the
diffusion, which is where its multivariate/long-range structure is the only
signal available. ``hard_prior`` forces full-prior up to a distance, and a
large ``tau`` makes the prior dominate everywhere (best on these smooth series).

This module is import-friendly (no side effects) so the runner and the
comparison harness can both call :func:`anchor_blend` and :func:`build_prior`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def default_monotone_groups(target_cols: List[str]) -> List[List[str]]:
    """Auto-detect ordered constraint groups from column names.

    Currently recognises latency-percentile columns (``lat<pct>_ms`` /
    ``lat<pct>``): by definition lat50 ≤ lat75 ≤ … ≤ lat100 within a row.
    Returns the column list ordered by percentile (ascending), or ``[]`` if
    fewer than two are present.
    """
    lat = []
    for c in target_cols:
        m = re.match(r"^lat(\d+)(?:_ms)?$", c)
        if m:
            lat.append((int(m.group(1)), c))
    lat.sort()
    cols = [c for _, c in lat]
    return [cols] if len(cols) >= 2 else []


def enforce_monotone_groups(
    df: pd.DataFrame,
    missing_mask: pd.DataFrame,
    groups: List[List[str]],
) -> pd.DataFrame:
    """Project each row's ordered group to be non-decreasing, in place.

    Only **imputed** cells (``missing_mask`` True) are modified; observed cells
    are kept exactly. For a fully-imputed group the row is replaced by its
    ascending sort (the multiset-preserving monotone projection); for a mixed
    row, imputed cells are clamped left-to-right into ``[prev, next observed]``
    so the whole group stays non-decreasing around the fixed observed anchors.
    """
    for cols in groups:
        cols = [c for c in cols if c in df.columns and c in missing_mask.columns]
        if len(cols) < 2:
            continue
        V = df[cols].to_numpy(dtype=float)
        M = missing_mask[cols].to_numpy(dtype=bool)
        n = len(cols)
        for r in range(V.shape[0]):
            row, mrow = V[r], M[r]
            if not mrow.any():
                continue
            if mrow.all():
                V[r] = np.sort(row)
                continue
            # Mixed: nearest observed value to the right of each position.
            next_obs = np.full(n, np.inf)
            nv = np.inf
            for j in range(n - 1, -1, -1):
                if not mrow[j] and not np.isnan(row[j]):
                    nv = row[j]
                next_obs[j] = nv
            prev = -np.inf
            for j in range(n):
                if mrow[j]:
                    hi = max(next_obs[j], prev)
                    row[j] = min(max(row[j], prev), hi)
                prev = row[j]
            V[r] = row
        df[cols] = V
    return df


def load_meta(prepared_dir: Path) -> dict:
    with (Path(prepared_dir) / "meta.json").open() as f:
        return json.load(f)


def build_prior(
    train_df: Optional[pd.DataFrame],
    test_input: pd.DataFrame,
    target_cols: List[str],
    method: str = "linear",
) -> pd.DataFrame:
    """Context-aware interpolation prior for the *test* split.

    Interpolates over ``concat(train, test_input)`` so leading/short test gaps
    are anchored by the trailing train history (the context v1 synthesis lacks),
    then returns only the test rows. Observed test cells are preserved exactly.

    ``method``: "linear" (pandas interpolate) or "nearest" (nearest observed).
    """
    n_train = 0 if train_df is None else len(train_df)
    if train_df is None:
        full = test_input[target_cols].copy().reset_index(drop=True)
    else:
        full = pd.concat(
            [train_df[target_cols], test_input[target_cols]], ignore_index=True
        )

    filled = full.copy()
    for c in target_cols:
        s = full[c]
        if method == "nearest":
            f = s.interpolate(method="nearest", limit_direction="both")
        else:
            f = s.interpolate(method="linear", limit_direction="both")
        # Edges / all-NaN columns: ffill+bfill, then column mean, then 0.
        f = f.ffill().bfill()
        if f.isna().any():
            m = s.mean()
            f = f.fillna(m if not np.isnan(m) else 0.0)
        filled[c] = f

    prior_test = filled.iloc[n_train:].reset_index(drop=True)
    return prior_test


def _distance_to_observed(mask_missing: np.ndarray, left_context: int = 0) -> np.ndarray:
    """Per-row distance (in index steps) to the nearest observed row.

    ``mask_missing`` is the test-split boolean mask (True = missing). When
    ``left_context`` > 0 the series is treated as if that many observed rows
    precede it (the trailing train history), so leading test gaps measure their
    true distance to the last train observation rather than +inf.
    """
    n = len(mask_missing)
    obs = np.where(~mask_missing)[0]
    if left_context > 0:
        # A virtual observed row sits at index -1 (the train boundary).
        obs = np.concatenate([[-1], obs])
    if len(obs) == 0:
        return np.full(n, 1e9)
    # Nearest observed index for every row, vectorised.
    pos = np.searchsorted(obs, np.arange(n))
    left = np.clip(pos - 1, 0, len(obs) - 1)
    right = np.clip(pos, 0, len(obs) - 1)
    dl = np.abs(np.arange(n) - obs[left])
    dr = np.abs(np.arange(n) - obs[right])
    return np.minimum(dl, dr).astype(float)


def anchor_blend(
    test_input: pd.DataFrame,
    diffusion: pd.DataFrame,
    prior: pd.DataFrame,
    target_cols: List[str],
    tau: float = 12.0,
    hard_prior: int = 4,
    has_left_context: bool = True,
    monotone_groups: Optional[List[List[str]]] = None,
) -> pd.DataFrame:
    """Blend diffusion output with the interpolation prior, per missing cell.

    Observed test cells are preserved exactly. For each originally-missing cell
    the weight on the prior is ``w = exp(-(d-1)/tau)`` (``=1`` for ``d ≤
    hard_prior``), where ``d`` is the distance to the nearest observed row.

    When ``monotone_groups`` is given, each group's imputed cells are projected
    to be non-decreasing per row (e.g. lat50 ≤ … ≤ lat100), a constraint the
    diffusion does not otherwise enforce. Observed cells are still preserved.

    Returns a frame with ``test_input``'s schema (time column + targets), with
    only originally-missing cells changed.
    """
    out = test_input.copy().reset_index(drop=True)
    diff = diffusion.reset_index(drop=True)
    pri = prior.reset_index(drop=True)

    for c in target_cols:
        if c not in out.columns:
            continue
        missing = out[c].isna().to_numpy()
        if not missing.any():
            continue
        d = _distance_to_observed(missing, left_context=1 if has_left_context else 0)
        w = np.exp(-(np.maximum(d - 1.0, 0.0)) / max(tau, 1e-6))
        if hard_prior > 0:
            w[d <= hard_prior] = 1.0

        p = pri[c].to_numpy(float) if c in pri.columns else np.zeros(len(out))
        f = diff[c].to_numpy(float) if c in diff.columns else p
        # Where diffusion is NaN (column absent / dropped), fall back to prior.
        f = np.where(np.isnan(f), p, f)
        blended = w * p + (1.0 - w) * f

        col = out[c].to_numpy(float)
        col[missing] = blended[missing]
        out[c] = col

    if monotone_groups:
        missing_mask = test_input.reset_index(drop=True)[target_cols].isna()
        enforce_monotone_groups(out, missing_mask, monotone_groups)

    return out


# ---------------------------------------------------------------------------
# Scoring (shared with the baseline harness: cells where input is NaN & GT is
# not). Kept here so v2 can be compared continuously without importing the
# Airflow-stubbed compare_baselines module.
# ---------------------------------------------------------------------------

def score_holdout(
    test_input: pd.DataFrame,
    test_gt: pd.DataFrame,
    imputed: pd.DataFrame,
    target_cols: List[str],
) -> Dict[str, float]:
    errs: List[float] = []
    for c in target_cols:
        if c not in imputed.columns:
            continue
        cell = test_input[c].isna().to_numpy() & test_gt[c].notna().to_numpy()
        if not cell.any():
            continue
        g = test_gt[c].to_numpy(float)
        p = imputed[c].to_numpy(float)
        k = cell & ~np.isnan(g) & ~np.isnan(p)
        errs.extend((p[k] - g[k]).tolist())
    if not errs:
        return {"MAE": float("nan"), "RMSE": float("nan"), "n_cells": 0}
    a = np.asarray(errs)
    return {
        "MAE": float(np.mean(np.abs(a))),
        "RMSE": float(np.sqrt(np.mean(a ** 2))),
        "n_cells": int(a.size),
    }
