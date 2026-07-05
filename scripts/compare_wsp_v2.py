#!/usr/bin/env python3
"""Continuous comparison harness: WaveStitch+ v1 vs v2 vs interpolation baselines.

Scores every method on the *same* prepared holdout (cells where test_input is
NaN and test_gt is not — identical to scripts/compare_baselines.py) so v2 can be
tracked against v1 and the darts baselines on each iteration.

Usage:
    python scripts/compare_wsp_v2.py \
        --prepared-dir experiments/EUR/prepared_amf \
        --baseline-dir experiments/amf-performance/generated_all_baselines \
        --v1-csv  .../imputed_em_ddim50_..._trial_0.csv \
        [--tau 4 --hard-prior 1 --prior nearest]

Prints a sorted MAE/RMSE table and writes ``wsp_v2_comparison.csv`` next to the
v2 output. Reuses the v1 diffusion CSV (no re-synthesis), so it runs in seconds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

APP_DIR = Path(__file__).resolve().parent.parent / "dockers" / "tools" / "WaveStitchPlus_app"
sys.path.insert(0, str(APP_DIR))

from wsp_v2 import (  # noqa: E402
    anchor_blend, build_prior, default_monotone_groups, load_meta, score_holdout,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepared-dir", required=True)
    p.add_argument("--baseline-dir", required=True,
                   help="dir with <method>_test_imputed.csv from compare_baselines.py")
    p.add_argument("--v1-csv", required=True,
                   help="WaveStitch+ v1 diffusion test CSV to anchor for v2")
    p.add_argument("--out-csv", default=None,
                   help="where to write the comparison table (default: alongside v1)")
    # Match run_imputation_v2.py defaults: 'auto' per-column prior (unsupervised),
    # anchored hard/wide because the diffusion is weaker than interpolation even at
    # depth on these holdouts.
    p.add_argument("--prior", default="auto", choices=["linear", "nearest", "auto"])
    p.add_argument("--tau", type=float, default=8.0)
    p.add_argument("--hard-prior", type=int, default=32)
    args = p.parse_args()

    prepared = Path(args.prepared_dir)
    meta = load_meta(prepared)
    tcols = meta["target_cols"]
    ti = pd.read_csv(prepared / "test_input.csv")
    gt = pd.read_csv(prepared / "test_gt.csv")
    train = pd.read_csv(prepared / "train.csv") if (prepared / "train.csv").exists() else None

    rows: List[Dict] = []

    # darts / pypots / imputegap baselines already on disk
    bdir = Path(args.baseline_dir)
    for f in sorted(bdir.glob("*_test_imputed.csv")):
        method = f.name[: -len("_test_imputed.csv")]
        df = pd.read_csv(f)
        s = score_holdout(ti, gt, df, tcols)
        rows.append({"method": method, **s})

    # WaveStitch+ v1 (raw diffusion)
    v1 = pd.read_csv(args.v1_csv)
    rows.append({"method": "wavestitchplus_v1", **score_holdout(ti, gt, v1, tcols)})

    # WaveStitch+ v2 (locally anchored)
    prior = build_prior(train, ti, tcols, method=args.prior)
    v2 = anchor_blend(ti, v1, prior, tcols, tau=args.tau,
                      hard_prior=args.hard_prior, has_left_context=train is not None,
                      monotone_groups=default_monotone_groups(tcols))
    rows.append({"method": f"wavestitchplus_v2(tau={args.tau:g},hp={args.hard_prior})",
                 **score_holdout(ti, gt, v2, tcols)})

    table = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    pd.set_option("display.width", 120)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    out = Path(args.out_csv) if args.out_csv else Path(args.v1_csv).parent / "wsp_v2_comparison.csv"
    table.to_csv(out, index=False)
    print(f"\n[wrote] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
