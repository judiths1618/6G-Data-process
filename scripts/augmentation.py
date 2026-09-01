"""
Time-series augmentation primitives for the imputation comparison experiments.

These are intentionally minimal — just enough to test whether augmentation
helps the neural imputers (pypots_saits / pypots_brits) converge on the EUR
telemetry data. Each augmenter operates on a (T, F) numpy array and returns
either a perturbed copy or a stack of windows.

Strategy composition is expressed as a ``--strategy`` string, e.g.::

    "sliding32+jitter"   sliding window W=32, stride=W/4, + 5% gaussian noise
    "sliding64"          sliding window W=64, stride=W/4, no perturbation
    "none"               no augmentation: one window covering the whole sequence

See ``parse_strategy`` for the supported tokens.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

def make_windows(arr: np.ndarray, window: int, stride: int) -> np.ndarray:
    """
    Slice ``arr`` (shape ``(T, F)``) into overlapping windows of shape
    ``(N, window, F)`` with the given stride. Returns at least one window;
    short inputs are NaN-padded on the right.
    """
    arr = np.asarray(arr, dtype=np.float32)
    T, F = arr.shape
    if T <= window:
        pad = np.full((window - T, F), np.nan, dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)[None, ...]  # (1, W, F)
    n = (T - window) // stride + 1
    out = np.empty((n, window, F), dtype=np.float32)
    for i in range(n):
        out[i] = arr[i * stride : i * stride + window]
    return out


# ---------------------------------------------------------------------------
# Per-window perturbations
# ---------------------------------------------------------------------------

def jitter(windows: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add per-channel Gaussian noise scaled by ``sigma * std(channel)``.

    NaN cells are left untouched (they're masked-as-missing for the imputer).
    """
    if sigma <= 0:
        return windows
    out = windows.copy()
    # Per-channel std across all train cells (ignore NaN).
    std = np.nanstd(out, axis=(0, 1), keepdims=True)
    noise = rng.standard_normal(out.shape).astype(out.dtype) * std * sigma
    observed = ~np.isnan(out)
    out[observed] = out[observed] + noise[observed]
    return out


def scaling(
    windows: np.ndarray, low: float, high: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Multiply each window by a per-channel scalar drawn uniformly in [low, high].
    Useful to teach the model invariance to overall amplitude (cpu_usage,
    ram_usage, latency scales vary across deployments).
    """
    if low == 1.0 and high == 1.0:
        return windows
    N, W, F = windows.shape
    factors = rng.uniform(low, high, size=(N, 1, F)).astype(windows.dtype)
    out = windows * factors
    return out


def time_warp(
    windows: np.ndarray, sigma: float, n_knots: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Non-uniform resampling: each window's time axis is warped via a smooth
    random function. ``n_knots`` control points, knot offsets ~ N(0, sigma).
    """
    if sigma <= 0:
        return windows
    N, W, F = windows.shape
    out = np.empty_like(windows)
    orig_t = np.linspace(0, 1, W)
    for i in range(N):
        # Random warp curve: monotone non-decreasing
        knots = np.linspace(0, 1, n_knots)
        offsets = rng.normal(0, sigma, size=n_knots)
        offsets[0] = offsets[-1] = 0
        new_knots = np.clip(knots + offsets, 0, 1)
        # Force monotone
        new_knots = np.maximum.accumulate(new_knots)
        new_t = np.interp(orig_t, knots, new_knots)
        for f in range(F):
            out[i, :, f] = np.interp(new_t, orig_t, windows[i, :, f])
    return out


# ---------------------------------------------------------------------------
# Strategy parsing
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    """A parsed augmentation strategy."""
    name: str
    window: int  # 0 = no windowing (whole sequence as one window)
    stride: int  # only meaningful when window > 0
    jitter_sigma: float = 0.0
    scaling_range: Optional[tuple] = None  # (low, high)
    timewarp_sigma: float = 0.0
    timewarp_knots: int = 4
    seed: int = 0

    def apply(self, arr: np.ndarray) -> np.ndarray:
        """Apply window extraction + perturbations to a (T, F) array."""
        rng = np.random.default_rng(self.seed)
        if self.window == 0:
            # Wrap the full sequence as one big window.
            windows = arr[None, ...].astype(np.float32)
        else:
            windows = make_windows(arr, self.window, self.stride)
        if self.jitter_sigma > 0:
            windows = jitter(windows, self.jitter_sigma, rng)
        if self.scaling_range is not None:
            lo, hi = self.scaling_range
            windows = scaling(windows, lo, hi, rng)
        if self.timewarp_sigma > 0:
            windows = time_warp(
                windows, self.timewarp_sigma, self.timewarp_knots, rng
            )
        return windows


_WINDOW_RE = re.compile(r"^sliding(\d+)(?:s(\d+))?$")


def parse_strategy(spec: str, seed: int = 0) -> Strategy:
    """
    Parse a ``--strategy`` token into a :class:`Strategy`.

    Grammar (tokens joined by ``+``):
      * ``none``                        — no augmentation (single window)
      * ``slidingW`` or ``slidingWsS``  — sliding window W, stride S (S=W/4 default)
      * ``jitter[F]``                   — Gaussian noise σ=0.05 (or F/100 if specified)
      * ``scale[L-H]``                  — per-channel scaling in [L/100, H/100] (default 90-110)
      * ``warp[F]``                     — time warp σ=0.1 (or F/100 if specified)
    """
    spec = spec.strip().lower()
    if spec in ("", "none"):
        return Strategy(name="none", window=0, stride=0, seed=seed)

    window = 0
    stride = 0
    jitter_sigma = 0.0
    scaling_range = None
    timewarp_sigma = 0.0
    timewarp_knots = 4

    for tok in spec.split("+"):
        tok = tok.strip()
        m = _WINDOW_RE.match(tok)
        if m:
            window = int(m.group(1))
            stride = int(m.group(2)) if m.group(2) else max(1, window // 4)
            continue
        if tok.startswith("jitter"):
            tail = tok[len("jitter"):]
            jitter_sigma = (int(tail) / 100.0) if tail else 0.05
            continue
        if tok.startswith("scale"):
            tail = tok[len("scale"):]
            if tail and "-" in tail:
                lo, hi = tail.split("-", 1)
                scaling_range = (int(lo) / 100.0, int(hi) / 100.0)
            else:
                scaling_range = (0.9, 1.1)
            continue
        if tok.startswith("warp"):
            tail = tok[len("warp"):]
            timewarp_sigma = (int(tail) / 100.0) if tail else 0.1
            continue
        raise ValueError(f"unknown augmentation token: {tok!r} in {spec!r}")

    return Strategy(
        name=spec, window=window, stride=stride,
        jitter_sigma=jitter_sigma,
        scaling_range=scaling_range,
        timewarp_sigma=timewarp_sigma,
        timewarp_knots=timewarp_knots,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Tile-based inference helper
# ---------------------------------------------------------------------------

def chunked_predict(
    impute_fn: Callable[[np.ndarray], np.ndarray],
    full_arr: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Tile ``full_arr`` (shape ``(T, F)``) into non-overlapping size-``window``
    chunks, run ``impute_fn`` on the stacked chunks ``(N, window, F)``, and
    splice the predictions back to ``(T, F)``. Right-padding is dropped.

    Used at inference time when the model was trained with ``n_steps=window``.
    """
    arr = np.asarray(full_arr, dtype=np.float32)
    T, F = arr.shape
    if window <= 0:
        # No windowing — process the whole sequence as one window.
        return impute_fn(arr[None, ...])[0, :T, :]
    pad = (window - T % window) % window
    if pad:
        arr = np.concatenate(
            [arr, np.full((pad, F), np.nan, dtype=np.float32)], axis=0
        )
    chunks = arr.reshape(-1, window, F)
    out = impute_fn(chunks)  # (N, W, F)
    out = np.asarray(out).reshape(-1, F)
    return out[:T]
