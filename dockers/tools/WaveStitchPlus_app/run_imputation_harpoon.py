#!/usr/bin/env python3
"""WaveStitch+ **HARPOON** imputation runner — baseline-compatible CLI wrapper.

Same interface as ``run_imputation.py`` (v1) / ``run_imputation_v2.py`` (v2); the
underlying synthesis is :mod:`synthesis_harpoon` (RePaint-DDIM + HARPOON
manifold-bound guidance) on the pre-trained WaveStitch+ model.

Writes baseline-compatible outputs (parallel to v1/v2):

    wavestitchplus_harpoon_train_imputed.csv   (copied from train_imputed_denorm,
                                                with the prepared train.csv's
                                                observed cells preserved)
    wavestitchplus_harpoon_test_imputed.csv    (HARPOON synthesis, observed cells
                                                preserved)

Discovered by the dashboard as ``wavestitchplus/harpoon`` (NEW_RE: lib=
wavestitchplus, method=harpoon). HARPOON requires only inference on a pre-trained
checkpoint — no retraining — so it's fast (~the same as v1 synthesis).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import pandas as pd

APP_DIR = Path(__file__).resolve().parent


def _env_for_device(device: str) -> dict:
    env = os.environ.copy()
    cache_root = Path(tempfile.gettempdir())
    mpl_cache = cache_root / "wavestitchplus_matplotlib"
    keops_cache = cache_root / "wavestitchplus_keops"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    keops_cache.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(mpl_cache))
    env.setdefault("PYKEOPS_CACHE_FOLDER", str(keops_cache))
    env.setdefault("KEOPS_CACHE_FOLDER", str(keops_cache))
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _fill_missing_only(input_df: pd.DataFrame, imputed_df: pd.DataFrame) -> pd.DataFrame:
    """Return ``input_df`` with NaN cells filled from ``imputed_df`` (positional).
    Observed input cells are preserved exactly; the input's schema/row count win.
    """
    out = input_df.reset_index(drop=True).copy()
    imp = imputed_df.reset_index(drop=True).reindex(out.index)
    for col in out.columns:
        if col in imp.columns:
            out[col] = out[col].where(out[col].notna(), imp[col])
    return out


def _publish_train_output(prepared: Path, output_dir: Path) -> Path | None:
    """Publish HARPOON's train output from the EM train-imputation artifact.

    HARPOON is inference-only on a pre-trained model — the train-side imputation
    comes from ``prepared/train_imputed_denorm.csv`` (produced by EM training).
    Mirrors the v1/v2 runners' train path.
    """
    train_imputed = prepared / "train_imputed_denorm.csv"
    train_input = prepared / "train.csv"
    if not train_imputed.exists():
        # v1's run_imputation.py removes train_imputed_denorm.csv from the bundle
        # after publishing it as wavestitchplus_v1_train_imputed.csv; reuse that so
        # a v1→harpoon sequence still emits the train split (and a full final).
        published = output_dir / "wavestitchplus_v1_train_imputed.csv"
        if published.exists():
            train_imputed = published
        else:
            print("[HARPOON] skip train output: train_imputed_denorm.csv not present")
            return None
    imp = pd.read_csv(train_imputed)
    if train_input.exists():
        imp = _fill_missing_only(pd.read_csv(train_input), imp)
    out_csv = output_dir / "wavestitchplus_harpoon_train_imputed.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out_csv, index=False)
    print(f"[HARPOON] wrote {out_csv}")
    return out_csv


def _run_synthesis(prepared: Path, out_csv: Path, args: argparse.Namespace,
                   env: dict) -> None:
    cmd = [
        sys.executable, str(APP_DIR / "synthesis_harpoon.py"),
        "-d", "custom_csv",
        "-prepared_dir", str(prepared.resolve()),
        "-out_csv", str(out_csv.resolve()),
        "-model_type", "em",
        "-model_tag", args.model_tag,
        "-clamp_mode", "bounds",
        "-repaint_rounds", str(args.repaint_rounds),
        "-guidance_scale", str(args.guidance_scale),
        "-ddim_steps", str(args.ddim_steps),
        "-n_trials", "1",
        "-bound_lambda", str(args.bound_lambda),
        "-bound_power", str(args.bound_power),
        "-project_bounds", str(args.project_bounds),
        "-prior_lambda", str(args.prior_lambda),
        "-prior_method", str(args.prior_method),
        "-smooth_lambda", str(args.smooth_lambda),
        "-monotone_lambda", str(args.monotone_lambda),
        "-pos_eps", str(args.pos_eps),
        "-auto_ub_q", str(args.auto_ub_q),
        "-auto_ub_pad", str(args.auto_ub_pad),
        "-bound_headroom", "1.2",
    ]
    if args.hard_project_positive:
        cmd.append("-hard_project_positive")
    print(f"[HARPOON] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, env=env, cwd=str(APP_DIR))
    if proc.returncode != 0:
        raise SystemExit(f"synthesis_harpoon failed (exit {proc.returncode})")


def run(args: argparse.Namespace) -> List[Path]:
    prepared = Path(args.prepared_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    if "test" in args.inputs:
        test_input = prepared / "test_input.csv"
        if not test_input.exists():
            raise SystemExit(f"test_input not found: {test_input}")
        out_csv = output_dir / "wavestitchplus_harpoon_test_imputed.csv"
        _run_synthesis(prepared, out_csv, args, _env_for_device(args.device))
        # Preserve observed cells (synthesis writes raw reconstruction).
        merged = _fill_missing_only(pd.read_csv(test_input), pd.read_csv(out_csv))
        merged.to_csv(out_csv, index=False)
        print(f"[HARPOON] wrote {out_csv}")
        written.append(out_csv)

    if "train" in args.inputs:
        tr = _publish_train_output(prepared, output_dir)
        if tr is not None:
            written.append(tr)

    # ---- final: stitch the imputed train + imputed test into one gap-free CSV
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    from wsp_final import build_wsp_final

    final = build_wsp_final(prepared, output_dir, "harpoon")
    if final is not None:
        written.append(final)

    return written


def main() -> None:
    p = argparse.ArgumentParser(description="WaveStitch+ HARPOON imputation runner")
    p.add_argument("--prepared-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--method", default="harpoon", choices=["harpoon"],
                   help="accepted for CLI compatibility with the dashboard's "
                        "generic runner invocation")
    p.add_argument("--inputs", nargs="+", default=["train", "test"],
                   choices=["train", "test"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    # Sampling knobs (parallel to v1/v2 defaults).
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--repaint-rounds", type=int, default=3)
    p.add_argument("--guidance-scale", type=float, default=0.1)
    p.add_argument("--model-tag", default="v1",
                   help="checkpoint tag produced by WaveStitch+ v1 training")
    # HARPOON-specific knobs.
    p.add_argument("--bound-lambda", type=float, default=0.3,
                   help="weight on the manifold-bound penalty (0 = vanilla v1)")
    p.add_argument("--bound-power", type=float, default=2.0,
                   help="hinge-penalty exponent (HARPOON p)")
    p.add_argument("--project-bounds", type=str, default="True",
                   choices=["True", "False", "true", "false"],
                   help="project imputed x0 cells into HARPOON bounds after "
                        "the gradient step")
    p.add_argument("--prior-lambda", type=float, default=0.25,
                   help="soft HARPOON guidance weight toward a local prior")
    p.add_argument("--prior-method", default="nearest", choices=["nearest", "linear"],
                   help="local prior used by HARPOON's manifold guidance")
    p.add_argument("--smooth-lambda", type=float, default=0.02,
                   help="soft temporal smoothness guidance weight")
    p.add_argument("--monotone-lambda", type=float, default=0.05,
                   help="soft latency-percentile monotonicity guidance weight")
    p.add_argument("--pos-eps", type=float, default=1e-6,
                   help="positivity epsilon (raw-scale lb floor)")
    p.add_argument("--auto-ub-q", type=float, default=0.99,
                   help="observed-data quantile used as the upper bound")
    p.add_argument("--auto-ub-pad", type=float, default=0.05,
                   help="multiplicative padding above the quantile")
    p.add_argument("--hard-project-positive", action="store_true",
                   help="final pass: floor target values at pos_eps (raw scale)")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
