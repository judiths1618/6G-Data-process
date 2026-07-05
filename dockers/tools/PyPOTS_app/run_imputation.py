#!/usr/bin/env python3
"""
PyPOTS baseline imputation runner (train.csv + test_input.csv).

Reads `prepared_<subset>/` (meta.json, train.csv, test_input.csv) and writes:

    pypots_<method>_train_imputed.csv
    pypots_<method>_test_imputed.csv

Each model fits on `train.csv` (target columns only, mean/std-standardized
using train statistics so train and test live in the same space), then imputes
both `train.csv` and `test_input.csv` and de-standardizes back. Original
observed cells are kept exactly as-is so the comparison reflects imputation
quality, not reconstruction noise.

Long series are handled by **windowed inference** (``--window``): the series is
tiled into fixed-length windows batched along the sample axis, so attention cost
stays O(window**2) instead of O(series_len**2). Without this a 100k-step series
OOMs. Overlapping windows (``--stride`` < window) are averaged on stitch.

Methods: saits, brits, transformer, gpvae, mrnn, csdi, usgan, timesnet
(only those available in your installed PyPOTS will work).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

INPUT_FILES = {"train": "train.csv", "test": "test_input.csv"}


def load_meta(prepared_dir: Path) -> dict:
    with (prepared_dir / "meta.json").open() as f:
        return json.load(f)


def fit_train_scaler(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    # Replace zero/NaN std (constant or all-NaN columns) with 1.0 so we don't divide by 0.
    std = np.where((std < 1e-8) | np.isnan(std), 1.0, std)
    mean = np.where(np.isnan(mean), 0.0, mean)
    return mean, std


def to_pypots_sample(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32)[None, ...]


# --- Windowed inference ---------------------------------------------------
# PyPOTS models fix ``n_steps`` at build time and attention is O(n_steps**2),
# so feeding a whole long series as one (1, n_steps, F) window OOMs. Instead we
# tile the series into fixed-length windows, batch them along the sample axis
# (n_windows, window, F), and stitch overlaps back by averaging.

def _window_starts(n_rows: int, window: int, stride: int) -> List[int]:
    """Window start indices covering ``[0, n_rows)``; the last is aligned to the
    end so the tail is never dropped."""
    if n_rows <= window:
        return [0]
    starts = list(range(0, n_rows - window + 1, stride))
    if starts[-1] != n_rows - window:
        starts.append(n_rows - window)
    return starts


def _to_windows(arr: np.ndarray, starts: List[int], window: int) -> np.ndarray:
    """Stack ``(n_windows, window, F)`` float32 windows; the array is padded with
    NaN (treated as missing by PyPOTS) when shorter than one window."""
    if arr.shape[0] < window:
        pad = np.full((window - arr.shape[0], arr.shape[1]), np.nan, dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    return np.stack([arr[s:s + window] for s in starts]).astype(np.float32)


def _stitch(win_arr: np.ndarray, starts: List[int], window: int,
            n_rows: int, n_features: int) -> np.ndarray:
    """Reconstruct ``(n_rows, F)`` from windows, averaging overlapping cells."""
    acc = np.zeros((n_rows, n_features), dtype=np.float64)
    cnt = np.zeros((n_rows, 1), dtype=np.float64)
    for i, s in enumerate(starts):
        end = min(s + window, n_rows)
        acc[s:end] += win_arr[i, : end - s]
        cnt[s:end] += 1.0
    cnt[cnt == 0] = 1.0
    return acc / cnt


def _impute_batch(model, windows: np.ndarray) -> np.ndarray:
    """Impute a batch of windows → ``(n_windows, window, F)``."""
    out = model.impute({"X": windows.astype(np.float32)})
    if isinstance(out, dict):
        out = out.get("imputation", out.get("X"))
    out = np.asarray(out)
    if out.ndim == 2:
        out = out[None, ...]
    return out


def _impute_series(model, series_norm: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Window → impute → stitch a normalized ``(L, F)`` series (same L back)."""
    L, F = series_norm.shape
    starts = _window_starts(L, window, stride)
    wins = _to_windows(series_norm, starts, window)
    out = _impute_batch(model, wins)
    if L <= window:
        return out[0, :L]
    return _stitch(out, starts, window, L, F)


def build_model(method: str, n_steps: int, n_features: int, args: argparse.Namespace):
    method = method.lower()
    if method == "saits":
        from pypots.imputation import SAITS
        return SAITS(
            n_steps=n_steps, n_features=n_features,
            n_layers=2, d_model=128, d_ffn=128,
            n_heads=4, d_k=32, d_v=32, dropout=0.1,
            epochs=args.epochs, batch_size=args.batch_size,
        )
    if method == "brits":
        from pypots.imputation import BRITS
        return BRITS(
            n_steps=n_steps, n_features=n_features, rnn_hidden_size=64,
            epochs=args.epochs, batch_size=args.batch_size,
        )
    if method == "transformer":
        from pypots.imputation import Transformer
        return Transformer(
            n_steps=n_steps, n_features=n_features,
            n_layers=2, d_model=128, d_ffn=128, n_heads=4, d_k=32, d_v=32,
            dropout=0.1, epochs=args.epochs, batch_size=args.batch_size,
        )
    if method == "gpvae":
        from pypots.imputation import GPVAE
        return GPVAE(
            n_steps=n_steps, n_features=n_features, latent_size=8,
            epochs=args.epochs, batch_size=args.batch_size,
        )
    if method == "mrnn":
        from pypots.imputation import MRNN
        return MRNN(
            n_steps=n_steps, n_features=n_features, rnn_hidden_size=64,
            epochs=args.epochs, batch_size=args.batch_size,
        )
    if method == "csdi":
        from pypots.imputation import CSDI
        return CSDI(
            n_steps=n_steps, n_features=n_features,
            n_layers=4, n_heads=4, n_channels=64, d_time_embedding=32,
            d_feature_embedding=16, d_diffusion_embedding=128,
            n_diffusion_steps=50,
            epochs=args.epochs, batch_size=args.batch_size,
        )
    if method == "usgan":
        from pypots.imputation import USGAN
        return USGAN(
            n_steps=n_steps, n_features=n_features, rnn_hidden_size=64,
            epochs=args.epochs, batch_size=args.batch_size,
        )
    if method == "timesnet":
        from pypots.imputation import TimesNet
        return TimesNet(
            n_steps=n_steps, n_features=n_features,
            n_layers=2, top_k=3, d_model=64, d_ffn=64, n_kernels=6,
            dropout=0.1, epochs=args.epochs, batch_size=args.batch_size,
        )
    raise ValueError(f"Unknown PyPOTS method '{method}'")


def _ckpt_path(model_path: str, method: str, n_steps: int, n_features: int) -> Path:
    """Checkpoint file for a (method, window-length, feature-count) combination.

    Keyed by n_steps + n_features because PyPOTS ``load`` requires the rebuilt
    model to match the saved architecture/shape exactly. The ``.pypots`` suffix
    is what ``model.save`` writes.
    """
    return Path(model_path) / f"pypots_{method}_n{n_steps}_f{n_features}.pypots"


def _prepare_model(method: str, n_steps: int, n_features: int,
                   args: argparse.Namespace, train_windows: np.ndarray):
    """Build a model and either load saved weights (skipping training) or fit it
    on ``train_windows`` — a ``(n_windows, n_steps, n_features)`` batch.

    With ``--load-model`` and an existing checkpoint for this exact shape, the
    weights are loaded and training is skipped — the WaveStitch+-style
    train-once / reuse pattern. With ``--save-model`` the freshly-fitted weights
    are persisted for next time.
    """
    model = build_model(method, n_steps=n_steps, n_features=n_features, args=args)
    ckpt = _ckpt_path(args.model_path, method, n_steps, n_features) if args.model_path else None

    if args.load_model and ckpt is not None and ckpt.exists():
        model.load(str(ckpt))
        print(f"[PyPOTS] loaded {ckpt} (training skipped)")
        return model

    model.fit({"X": train_windows.astype(np.float32)})
    if args.save_model and ckpt is not None:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(ckpt), overwrite=True)
        print(f"[PyPOTS] saved {ckpt}")
    return model


def run(args: argparse.Namespace) -> List[Path]:
    prepared = Path(args.prepared_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(prepared)
    target_cols = meta["target_cols"]

    train_df = pd.read_csv(prepared / INPUT_FILES["train"])
    targets = [c for c in target_cols if c in train_df.columns]
    if not targets:
        raise RuntimeError("No target columns found in train.csv")
    train_arr = train_df[targets].to_numpy(dtype=np.float64)
    mean, std = fit_train_scaler(train_arr)
    train_norm = (train_arr - mean) / std

    # Window length caps attention cost; the model is built with n_steps=window,
    # so train and every inference series are tiled into equal windows.
    n_rows, n_features = train_arr.shape
    window = args.window if args.window and args.window > 0 else n_rows
    window = min(window, n_rows)
    stride = args.stride if args.stride and args.stride > 0 else window
    stride = min(stride, window)
    train_starts = _window_starts(n_rows, window, stride)
    train_windows = _to_windows(train_norm, train_starts, window)

    print(f"[PyPOTS] train shape={train_arr.shape}  window={window}  "
          f"n_windows={len(train_starts)}  method={args.method}  "
          f"epochs={args.epochs}  batch_size={args.batch_size}")
    model = _prepare_model(args.method, window, n_features, args, train_windows)

    written: List[Path] = []
    for kind in args.inputs:
        src = prepared / INPUT_FILES[kind]
        if not src.exists():
            print(f"[PyPOTS]   skip: {src} not found")
            continue
        df = pd.read_csv(src)
        cur_targets = [c for c in targets if c in df.columns]
        if not cur_targets:
            print(f"[PyPOTS]   skip: no shared target columns in {src}")
            continue

        arr = df[cur_targets].to_numpy(dtype=np.float64)
        nan_before = int(np.isnan(arr).sum())

        # Same window length as training; tile → impute → stitch.
        arr_norm = (arr - mean) / std
        imputed_norm = _impute_series(model, arr_norm, window, stride)
        filled = imputed_norm * std + mean

        # Only fill cells that were originally missing.
        miss_mask = np.isnan(arr)
        out_arr = np.where(miss_mask, filled, arr)
        # Catch any residual NaNs (e.g. all-NaN columns) with column means.
        if np.isnan(out_arr).any():
            col_means = np.where(np.isnan(mean), 0.0, mean)
            still = np.isnan(out_arr)
            out_arr = np.where(still, np.broadcast_to(col_means, out_arr.shape), out_arr)
        nan_after = int(np.isnan(out_arr).sum())
        print(f"[PyPOTS] {kind}: shape={arr.shape}  NaN {nan_before} -> {nan_after}")

        out_df = df.copy()
        out_df[cur_targets] = out_arr
        out_path = output_dir / f"pypots_{args.method}_{kind}_imputed.csv"
        out_df.to_csv(out_path, index=False)
        print(f"[PyPOTS] wrote {out_path}")
        written.append(out_path)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="PyPOTS baseline imputation runner")
    p.add_argument("--prepared-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--method", default="saits",
                   choices=["saits", "brits", "transformer", "gpvae",
                            "mrnn", "csdi", "usgan", "timesnet"])
    p.add_argument("--inputs", nargs="+", default=["train", "test"],
                   choices=list(INPUT_FILES.keys()))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--window", type=int, default=100,
                   help="window length (n_steps) the model is built for; the series is "
                        "tiled into windows of this size. 0 = whole series (legacy; OOMs "
                        "on long series because attention is O(n_steps**2)).")
    p.add_argument("--stride", type=int, default=0,
                   help="window stride; 0 → = window (non-overlapping). Smaller = more "
                        "overlap, averaged on stitch.")
    # Optional checkpointing — the WaveStitch+-style train-once / reuse pattern.
    p.add_argument("--model-path", default=None,
                   help="directory for checkpoints; required by --save-model/--load-model")
    p.add_argument("--save-model", action="store_true",
                   help="save trained weights under --model-path")
    p.add_argument("--load-model", action="store_true",
                   help="load weights from --model-path if present, skipping training")
    args = p.parse_args()

    if (args.save_model or args.load_model) and not args.model_path:
        p.error("--save-model/--load-model require --model-path")

    try:
        run(args)
    except ImportError as exc:
        print(f"[PyPOTS] missing dependency: {exc}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
