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
from typing import List, Optional, Tuple

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from wsp_v2 import (  # noqa: E402
    anchor_blend, build_prior, default_monotone_groups, load_meta, score_holdout,
)


def _csv_floats(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _csv_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _output_name(method: str, split: str) -> str:
    tag = "v2_tuned" if method == "tuned" else "v2"
    return f"wavestitchplus_{tag}_{split}_imputed.csv"


def _select_tuned_params(
    train_df: Optional[pd.DataFrame],
    test_input: pd.DataFrame,
    gt: pd.DataFrame,
    diffusion: pd.DataFrame,
    target_cols: List[str],
    args: argparse.Namespace,
    mono_groups: Optional[List[List[str]]],
) -> Tuple[str, float, int, dict]:
    """Grid-search v2 anchoring knobs on the prepared holdout cells.

    This is an explicit evaluation/tuning mode for experiments and dashboards.
    It requires ``test_gt.csv`` and should be reported as tuned, not as the
    default unsupervised v2 setting.
    """
    best: Tuple[str, float, int, dict] | None = None
    priors = {
        prior_name: build_prior(train_df, test_input, target_cols, method=prior_name)
        for prior_name in args.tune_priors
    }
    for prior_name, prior in priors.items():
        for tau in args.tune_taus:
            for hard_prior in args.tune_hard_priors:
                merged = anchor_blend(
                    test_input, diffusion, prior, target_cols,
                    tau=tau, hard_prior=hard_prior,
                    has_left_context=train_df is not None,
                    monotone_groups=mono_groups,
                )
                score = score_holdout(test_input, gt, merged, target_cols)
                if best is None or score["MAE"] < best[3]["MAE"]:
                    best = (prior_name, tau, hard_prior, score)
    if best is None:
        raise SystemExit("tuned v2 found no valid grid candidates")
    return best


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

    # v2 anchors the TEST split, so the diffusion CSV must line up with
    # test_input.csv. A common mistake is passing the v1 *train* output — catch
    # it with a clear message instead of a cryptic broadcast error downstream.
    if len(diffusion) != len(test_input):
        raise SystemExit(
            f"--reuse-diffusion {diff_path.name} has {len(diffusion)} rows but the "
            f"test split (test_input.csv) has {len(test_input)}. Pass the v1 TEST "
            f"output (wavestitchplus_v1_test_imputed.csv), not the train output — v2 "
            f"anchors the train split separately from train_imputed_denorm.csv."
        )

    mono_groups = None if args.no_monotone else default_monotone_groups(target_cols)
    if mono_groups:
        print(f"[WaveStitch+ v2] enforcing monotone group(s): {mono_groups}")

    gt_path = prepared / "test_gt.csv"
    gt = pd.read_csv(gt_path) if gt_path.exists() else None
    if args.method == "tuned":
        if gt is None:
            print("[WaveStitch+ v2] tuned mode requires test_gt.csv; falling back to anchored")
        else:
            args.prior, args.tau, args.hard_prior, tuned_score = _select_tuned_params(
                train_df, test_input, gt, diffusion, target_cols, args, mono_groups,
            )
            print(
                "[WaveStitch+ v2] tuned params: "
                f"prior={args.prior} tau={args.tau:g} hard_prior={args.hard_prior} "
                f"MAE={tuned_score['MAE']:.1f} RMSE={tuned_score['RMSE']:.1f}"
            )

    # ---- context-aware prior + local anchoring ------------------------------
    prior = build_prior(train_df, test_input, target_cols, method=args.prior)
    merged = anchor_blend(
        test_input, diffusion, prior, target_cols,
        tau=args.tau, hard_prior=args.hard_prior,
        has_left_context=train_df is not None,
        monotone_groups=mono_groups,
    )
    # Keep only schema columns present in test_input.
    merged = merged[[c for c in test_input.columns if c in merged.columns]]

    test_out = output_dir / _output_name(args.method, "test")
    merged.to_csv(test_out, index=False)
    print(f"[WaveStitch+ v2] wrote {test_out}")
    written = [test_out]

    # ---- optional self-scoring against test_gt ------------------------------
    if gt is not None:
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
    if not train_imputed.exists():
        # v1's run_imputation.py removes train_imputed_denorm.csv from the bundle
        # after publishing it as wavestitchplus_v1_train_imputed.csv; anchor that so
        # a v1→v2 sequence still produces the train split (and a full final).
        published = output_dir / "wavestitchplus_v1_train_imputed.csv"
        if published.exists():
            train_imputed = published
    if "train" in args.inputs and train_df is not None and train_imputed.exists():
        train_diff = pd.read_csv(train_imputed)
        train_prior = build_prior(None, train_df, target_cols, method=args.prior)
        train_merged = anchor_blend(
            train_df, train_diff, train_prior, target_cols,
            tau=args.tau, hard_prior=args.hard_prior,
            has_left_context=False, monotone_groups=mono_groups,
        )
        train_merged = train_merged[[c for c in train_df.columns if c in train_merged.columns]]
        train_out = output_dir / _output_name(args.method, "train")
        train_merged.to_csv(train_out, index=False)
        print(f"[WaveStitch+ v2] wrote {train_out}")
        written.append(train_out)
    elif "train" in args.inputs:
        print("[WaveStitch+ v2] skip train output: train_imputed_denorm.csv not found")

    # ---- final: stitch the imputed train + imputed test into one gap-free CSV
    from wsp_final import build_wsp_final

    tag = "v2_tuned" if args.method == "tuned" else "v2"
    final = build_wsp_final(prepared, output_dir, tag)
    if final is not None:
        written.append(final)

    return written


def main() -> None:
    p = argparse.ArgumentParser(
        description="WaveStitch+ v2 imputation runner (locally-anchored)")
    p.add_argument("--prepared-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--method", default="anchored", choices=["anchored", "tuned"],
                   help="accepted for CLI compatibility with the dashboard's "
                        "generic runner invocation; tuned grid-searches "
                        "anchoring knobs on test_gt.csv and writes v2_tuned")
    p.add_argument("--inputs", nargs="+", default=["train", "test"],
                   choices=["train", "test"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--reuse-diffusion", default=None,
                   help="path to an existing v1 diffusion test CSV to anchor "
                        "(skips re-running synthesis)")
    # Anchoring knobs.
    p.add_argument("--prior", default="auto", choices=["linear", "nearest", "auto"],
                   help="interpolation prior on concat(train,test). 'auto' (default) "
                        "picks nearest vs linear PER COLUMN by an unsupervised "
                        "observed-data cross-check (trending cols → linear, "
                        "near-constant → nearest); this is what lets v2 beat the "
                        "single nearest/linear baselines.")
    p.add_argument("--tau", type=float, default=8.0,
                   help="prior-weight decay length; larger = trust the prior "
                        "deeper into gaps. On the 6G holdouts the diffusion is "
                        "weaker than interpolation even at depth, so the default is "
                        "conservative — lower it only to let diffusion contribute "
                        "in genuinely long gaps.")
    p.add_argument("--hard-prior", type=int, default=32,
                   help="cells within this distance of an observation follow the "
                        "prior exactly; diffusion blends in only beyond it. Default "
                        "32 (≈ the model receptive field) makes v2 ≈ the auto "
                        "interpolation on natural holdouts (where interpolation is "
                        "the ceiling) and reserves diffusion for genuinely long gaps; "
                        "lower it to expose more diffusion.")
    p.add_argument("--tune-prior-names", default="nearest,linear",
                   help="comma-separated priors to scan in --method tuned")
    p.add_argument("--tune-taus", type=_csv_floats, default=_csv_floats("1,2,3,4,6,8,12,20,40"),
                   help="comma-separated tau grid for --method tuned")
    p.add_argument("--tune-hard-priors", type=_csv_ints, default=_csv_ints("0,1,2,4,8"),
                   help="comma-separated hard-prior grid for --method tuned")
    p.add_argument("--no-monotone", action="store_true",
                   help="disable the per-row monotone projection of ordered groups "
                        "(e.g. lat50≤…≤lat100); on by default")
    # Synthesis passthrough (only used when not reusing a diffusion CSV).
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--repaint-rounds", type=int, default=3)
    p.add_argument("--guidance-scale", type=float, default=0.1)
    args = p.parse_args()
    args.tune_priors = [
        x.strip() for x in args.tune_prior_names.split(",")
        if x.strip() in {"nearest", "linear"}
    ]
    if not args.tune_priors:
        args.tune_priors = ["nearest", "linear"]
    run(args)


if __name__ == "__main__":
    main()
