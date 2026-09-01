#!/usr/bin/env python3
"""
Compare data-augmentation strategies for neural imputers (pypots_saits /
pypots_brits).

Reuses an existing ``prepared_<run_id>/`` directory produced by
``compare_baselines.py`` — no need to re-preprocess. For each requested
strategy, the script:

    1. Builds an augmented training set from ``train.csv``.
       The strategy controls windowing (sliding W, stride S) and optional
       per-window perturbations (jitter, per-channel scaling, time-warp).
    2. Trains the selected neural imputer with ``n_steps = window``.
    3. Imputes the FULL sequence (``train.csv + test_input.csv``) by tiling
       it into size-W chunks and stitching back.
    4. Splits the imputed sequence back into a train slice (saved as
       ``<method>_<strategy>_train_imputed.csv``) and a test slice
       (saved as ``<method>_<strategy>_test_imputed.csv``).
    5. Scores the test slice against ``test_gt.csv`` at the cells flagged by
       ``eval_holdout_mask.npy`` — IDENTICAL to ``compare_baselines.py``'s
       scoring so the numbers are directly comparable.

Outputs land in ``experiments/<dataset>/augmented_<run_id>/`` so the
dashboard can pick them up (Subset = ``<run_id>``) alongside the
non-augmented baseline.

Example::

    python scripts/augment_experiment.py \\
        --prepared-dir experiments/amf-performance/prepared_all_baselines \\
        --method pypots_saits \\
        --strategies none,sliding32,sliding32+jitter,sliding32+scale,sliding64 \\
        --epochs 30
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "dockers" / "airflow" / "dags"))

from augmentation import chunked_predict, parse_strategy  # noqa: E402


SUPPORTED_METHODS = ("pypots_saits", "pypots_brits")


def _fit_and_impute(
    method: str,
    train_windows: np.ndarray,
    full_arr: np.ndarray,
    window: int,
    epochs: int,
) -> tuple[np.ndarray, float]:
    """
    Train the specified pypots model with ``n_steps=train_windows.shape[1]``
    and impute the full timeline.

    Returns ``(imputed_array (T,F), elapsed_seconds)``.
    """
    import torch
    t0 = time.perf_counter()

    n_steps = train_windows.shape[1]
    n_features = train_windows.shape[2]
    common = dict(
        n_steps=n_steps, n_features=n_features,
        epochs=epochs, batch_size=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    if method == "pypots_saits":
        from pypots.imputation import SAITS
        model = SAITS(n_layers=2, d_model=64, n_heads=2,
                      d_k=32, d_v=32, d_ffn=128, dropout=0.0, **common)
    elif method == "pypots_brits":
        from pypots.imputation import BRITS
        model = BRITS(rnn_hidden_size=64, **common)
    else:
        raise ValueError(f"Unsupported method: {method!r}")

    model.fit({"X": train_windows})

    def _do_impute(chunks: np.ndarray) -> np.ndarray:
        out = model.impute({"X": chunks})
        if isinstance(out, dict):
            out = out.get("imputation", out)
        return np.asarray(out)

    imputed = chunked_predict(_do_impute, full_arr, window=window)
    return imputed, time.perf_counter() - t0


def _score(
    test_input: pd.DataFrame,
    test_gt: pd.DataFrame,
    test_imputed: pd.DataFrame,
    target_cols: List[str],
) -> dict:
    """Per-target → overall MAE/RMSE/MAPE on the held-out cells."""
    errs = []
    ratios = []
    n = 0
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
        errs.extend((pr - gt).tolist())
        nz = np.abs(gt) > 1e-12
        if nz.any():
            ratios.extend(np.abs((pr[nz] - gt[nz]) / gt[nz]).tolist())
        n += len(gt)
    if not errs:
        return {"MAE": float("nan"), "RMSE": float("nan"),
                "MAPE_%": float("nan"), "n_cells": 0}
    a = np.array(errs)
    return {
        "MAE": float(np.mean(np.abs(a))),
        "RMSE": float(np.sqrt(np.mean(a ** 2))),
        "MAPE_%": (float(np.mean(np.array(ratios)) * 100) if ratios
                   else float("nan")),
        "n_cells": int(n),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--prepared-dir", type=Path, required=True,
                   help="An existing prepared_<run_id>/ directory.")
    p.add_argument("--method", choices=SUPPORTED_METHODS, default="pypots_saits")
    p.add_argument(
        "--strategies", type=str,
        default="none,sliding32,sliding32+jitter,sliding64",
        help="Comma-separated strategies. See augmentation.parse_strategy.",
    )
    p.add_argument("--epochs", type=int,
                   default=int(os.environ.get("AUGMENT_EPOCHS", "30")))
    p.add_argument("--output-root", type=Path,
                   default=REPO_ROOT / "experiments")
    p.add_argument("--run-id", type=str, default=None,
                   help="Output dir suffix. Default: augmented_<timestamp>.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    prep = args.prepared_dir.resolve()
    if not (prep / "meta.json").exists():
        raise SystemExit(f"{prep} is not a prepared_<run_id>/ dir.")
    prep_meta = json.loads((prep / "meta.json").read_text())
    target_cols = prep_meta["target_cols"]
    cat_encoded = prep_meta.get("categorical_encoded_cols") or []

    train_df = pd.read_csv(prep / "train.csv")
    test_input_df = pd.read_csv(prep / "test_input.csv")
    test_gt_df = pd.read_csv(prep / "test_gt.csv")

    dataset = prep.parent.name  # experiments/<dataset>/prepared_<run_id>/
    run_id = args.run_id or f"augmented_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    out_dir = args.output_root / dataset / f"generated_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies = [parse_strategy(s, seed=args.seed)
                  for s in args.strategies.split(",")]

    train_arr = train_df[target_cols].to_numpy(dtype=np.float32)
    test_input_arr = test_input_df[target_cols].to_numpy(dtype=np.float32)
    full_arr = np.concatenate([train_arr, test_input_arr], axis=0)
    n_train = len(train_arr)

    print(f"[INFO] prepared_dir = {prep}")
    print(f"[INFO] method       = {args.method}  epochs = {args.epochs}")
    print(f"[INFO] target_cols  ({len(target_cols)}): {target_cols}")
    print(f"[INFO] train={len(train_df)}  test={len(test_input_df)}  "
          f"holdout_cells="
          f"{int(test_input_df[target_cols].isna().sum().sum())}")
    print(f"[INFO] output       = {out_dir}\n")

    rows = []
    for strat in strategies:
        print(f"=== strategy: {strat.name} ===")
        print(f"    window={strat.window}  stride={strat.stride}  "
              f"jitter={strat.jitter_sigma}  scale={strat.scaling_range}  "
              f"warp={strat.timewarp_sigma}")
        try:
            train_windows = strat.apply(train_arr)
            # SAITS/BRITS need at least one window. Make sure the window-size
            # we use for inference matches what we trained on.
            window_size = train_windows.shape[1]
            print(f"    train_windows: {train_windows.shape}")
            # Always chunk inference at the same width the model was trained
            # with — even the ``none`` strategy, where the training "window"
            # is the full train sequence and inference (which is longer:
            # train + test) gets tiled into multiples of that width.
            imputed_full, elapsed = _fit_and_impute(
                args.method, train_windows, full_arr,
                window=window_size,
                epochs=args.epochs,
            )
            assert imputed_full.shape == full_arr.shape, \
                f"shape mismatch: {imputed_full.shape} vs {full_arr.shape}"

            # Rebuild a DataFrame keeping the original schema so the dashboard
            # can pick it up; replace target_cols with imputed values, ffill
            # categorical-encoded cond columns.
            train_imp = train_df.copy()
            test_imp = test_input_df.copy()
            train_imp[target_cols] = imputed_full[:n_train]
            test_imp[target_cols] = imputed_full[n_train:]
            for c in cat_encoded:
                if c in train_imp.columns:
                    combined = pd.concat([train_imp[c], test_imp[c]]).ffill().bfill()
                    train_imp[c] = combined.iloc[:n_train].values
                    test_imp[c] = combined.iloc[n_train:].values

            tag = f"{args.method}_{strat.name}"
            train_imp.to_csv(out_dir / f"{tag}_train_imputed.csv", index=False)
            test_imp.to_csv(out_dir / f"{tag}_test_imputed.csv", index=False)

            score = _score(test_input_df, test_gt_df, test_imp, target_cols)
            score.update({"strategy": strat.name, "elapsed_sec": elapsed,
                          "window": strat.window, "stride": strat.stride,
                          "train_windows": int(train_windows.shape[0])})
            print(f"    MAE={score['MAE']:.2f}  RMSE={score['RMSE']:.2f}  "
                  f"elapsed={elapsed:.1f}s  windows={train_windows.shape[0]}")
            rows.append(score)
        except Exception as e:
            print(f"    FAILED: {e}")
            rows.append({"strategy": strat.name, "MAE": float("nan"),
                         "RMSE": float("nan"), "MAPE_%": float("nan"),
                         "n_cells": 0, "elapsed_sec": float("nan"),
                         "error": str(e)[:100]})

    results = pd.DataFrame(rows).sort_values("MAE")
    results.to_csv(out_dir / "results_overall.csv", index=False)

    print("\n" + "=" * 78)
    print(f"RESULTS  ({args.method} on {dataset}, {args.epochs} epochs)")
    print("=" * 78)
    with pd.option_context("display.float_format", "{:.4g}".format,
                           "display.max_rows", None,
                           "display.width", 140):
        print(results.to_string(index=False))
    print(f"\n→ {out_dir / 'results_overall.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
