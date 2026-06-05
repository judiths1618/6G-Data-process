#!/usr/bin/env python3
"""WaveStitch+ **v2** imputation runner — baseline-compatible CLI wrapper.

Same interface as ``run_imputation.py`` (v1), but the synthesized values are
*locally anchored* to a context-aware interpolation prior before being written
(see :mod:`wsp_v2` for the rationale). Writes baseline-compatible outputs:

    wavestitchplus_v2_train_imputed.csv
    wavestitchplus_v2_test_imputed.csv

Two ways to obtain the underlying diffusion output:

  --reuse-diffusion <csv>   anchor an existing v1 test output (fast; reuses a
                            checkpoint's synthesis, no GPU/re-run needed).
  (default)                 run ``synthesis_improved.py`` on the prepared dir
                            to produce the diffusion output, then anchor it.
                            Requires the prepared dir to map to a trained model
                            (see directory_manager).

Observed cells are preserved exactly; only originally-missing cells carry v2
values, matching the v1 runner and the PyPOTS baseline.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from wsp_v2 import (  # noqa: E402
    anchor_blend, build_prior, default_monotone_groups, load_meta, score_holdout,
)


def _env_for_device(device: str) -> dict:
    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _run_diffusion(prepared: Path, env: dict, args: argparse.Namespace) -> Path:
    """Run v1 synthesis to produce the diffusion test CSV; return its path.

    Publishes directly as ``wavestitchplus_v1_test_imputed.csv`` (the dashboard
    name for v1, parallel to v2's ``wavestitchplus_v2_test_imputed.csv``) so a
    fresh v2 run also surfaces v1 in the dashboard — no orphan temp files left
    behind. If you want to anchor an existing v1 output without re-running
    synthesis, use ``--reuse-diffusion <path>``.
    """
    out_csv = Path(args.output_dir) / "wavestitchplus_v1_test_imputed.csv"
    cmd = [
        sys.executable, str(APP_DIR / "synthesis_improved.py"),
        "-d", "custom_csv",
        "-prepared_dir", str(prepared),
        "-out_csv", str(out_csv),
        "-model_type", "em",
        "-clamp_mode", "bounds",
        "-repaint_rounds", str(args.repaint_rounds),
        "-guidance_scale", str(args.guidance_scale),
        "-n_trials", "1",
        "-ddim_steps", str(args.ddim_steps),
        "-bound_headroom", "1.2",
    ]
    print(f"[WaveStitch+ v2] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"synthesis failed (exit {proc.returncode})")
    return out_csv


def run(args: argparse.Namespace) -> List[Path]:
    prepared = Path(args.prepared_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(prepared)
    time_col = meta.get("time_col", "time")
    target_cols = meta.get("target_cols", [])
    if not target_cols:
        raise SystemExit("meta.json has no target_cols")

    test_input = pd.read_csv(prepared / "test_input.csv")
    train_df = (
        pd.read_csv(prepared / "train.csv")
        if (prepared / "train.csv").exists() else None
    )

    # ---- diffusion output (reuse or synthesize) -----------------------------
    if args.reuse_diffusion:
        diff_path = Path(args.reuse_diffusion)
        if not diff_path.exists():
            raise SystemExit(f"--reuse-diffusion not found: {diff_path}")
        print(f"[WaveStitch+ v2] reusing diffusion output: {diff_path}")
    else:
        diff_path = _run_diffusion(prepared, _env_for_device(args.device), args)
    diffusion = pd.read_csv(diff_path)

    # ---- context-aware prior + local anchoring ------------------------------
    prior = build_prior(train_df, test_input, target_cols, method=args.prior)
    mono_groups = None if args.no_monotone else default_monotone_groups(target_cols)
    if mono_groups:
        print(f"[WaveStitch+ v2] enforcing monotone group(s): {mono_groups}")
    merged = anchor_blend(
        test_input, diffusion, prior, target_cols,
        tau=args.tau, hard_prior=args.hard_prior,
        has_left_context=train_df is not None,
        monotone_groups=mono_groups,
    )
    # Keep only schema columns present in test_input.
    merged = merged[[c for c in test_input.columns if c in merged.columns]]

    test_out = output_dir / "wavestitchplus_v2_test_imputed.csv"
    merged.to_csv(test_out, index=False)
    print(f"[WaveStitch+ v2] wrote {test_out}")
    written = [test_out]

    # ---- optional self-scoring against test_gt ------------------------------
    gt_path = prepared / "test_gt.csv"
    if gt_path.exists():
        gt = pd.read_csv(gt_path)
        s_v2 = score_holdout(test_input, gt, merged, target_cols)
        s_v1 = score_holdout(test_input, gt, diffusion, target_cols)
        print(f"[WaveStitch+ v2] holdout score  v1: MAE={s_v1['MAE']:.1f} "
              f"RMSE={s_v1['RMSE']:.1f}  →  v2: MAE={s_v2['MAE']:.1f} "
              f"RMSE={s_v2['RMSE']:.1f}  (n={s_v2['n_cells']})")

    # ---- train split: apply the SAME v2 anchoring to the EM train-imputation
    # The EM train-imputation (train_imputed_denorm.csv) is the train-side analog
    # of the test diffusion output, so v2 anchors it to a train-only interpolation
    # prior and enforces the monotone groups — observed train cells preserved,
    # only the natural-gap cells carry blended (and constraint-satisfying) values.
    train_imputed = prepared / "train_imputed_denorm.csv"
    if "train" in args.inputs and train_df is not None and train_imputed.exists():
        train_diff = pd.read_csv(train_imputed)
        train_prior = build_prior(None, train_df, target_cols, method=args.prior)
        train_merged = anchor_blend(
            train_df, train_diff, train_prior, target_cols,
            tau=args.tau, hard_prior=args.hard_prior,
            has_left_context=False, monotone_groups=mono_groups,
        )
        train_merged = train_merged[[c for c in train_df.columns if c in train_merged.columns]]
        train_out = output_dir / "wavestitchplus_v2_train_imputed.csv"
        train_merged.to_csv(train_out, index=False)
        print(f"[WaveStitch+ v2] wrote {train_out}")
        written.append(train_out)
    elif "train" in args.inputs:
        print("[WaveStitch+ v2] skip train output: train_imputed_denorm.csv not found")

    return written


def main() -> None:
    p = argparse.ArgumentParser(
        description="WaveStitch+ v2 imputation runner (locally-anchored)")
    p.add_argument("--prepared-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--method", default="anchored", choices=["anchored"],
                   help="accepted for CLI compatibility with the dashboard's "
                        "generic runner invocation; v2 has a single mode")
    p.add_argument("--inputs", nargs="+", default=["train", "test"],
                   choices=["train", "test"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--reuse-diffusion", default=None,
                   help="path to an existing v1 diffusion test CSV to anchor "
                        "(skips re-running synthesis)")
    # Anchoring knobs.
    p.add_argument("--prior", default="nearest", choices=["linear", "nearest"],
                   help="interpolation prior built on concat(train,test); "
                        "'nearest' matches the strongest darts baseline")
    p.add_argument("--tau", type=float, default=20.0,
                   help="prior-weight decay length; larger = trust prior deeper "
                        "into gaps. Default lets the diffusion contribute only in "
                        "gaps longer than ~20 steps (its structural regime); "
                        "these smooth 6G series favour an even larger tau.")
    p.add_argument("--hard-prior", type=int, default=8,
                   help="cells within this distance of an observation follow the "
                        "prior exactly (the regime where interpolation is "
                        "near-optimal and the diffusion adds only noise)")
    p.add_argument("--no-monotone", action="store_true",
                   help="disable the per-row monotone projection of ordered groups "
                        "(e.g. lat50≤…≤lat100); on by default")
    # Synthesis passthrough (only used when not reusing a diffusion CSV).
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--repaint-rounds", type=int, default=3)
    p.add_argument("--guidance-scale", type=float, default=0.1)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
