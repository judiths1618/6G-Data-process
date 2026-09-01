#!/usr/bin/env python3
"""Continuous comparison harness: WaveStitch+ v1 vs v2 vs interpolation baselines.

Scores every method on the *same* prepared holdout (cells where test_input is
NaN and test_gt is not — identical to scripts/compare_baselines.py) so v2 can be
tracked against v1 and the darts baselines on each iteration.

Usage:
    python scripts/compare_wsp_v2.py \
        --prepared-dir data/processed/amf-performance_regularized \
        --baseline-dir data/processed/amf-performance_generated \
        --v1-csv data/processed/amf-performance_generated/wavestitchplus_v1_test_imputed.csv \
        [--tau 4 --hard-prior 1 --prior nearest]

Prints a sorted MAE/RMSE table and writes ``wsp_v2_comparison.csv`` next to the
v2 output. Reuses the v1 diffusion CSV (no re-synthesis), so it runs in seconds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

APP_DIR = Path(__file__).resolve().parent.parent / "dockers" / "tools" / "WaveStitchPlus_app"
sys.path.insert(0, str(APP_DIR))

from wsp_v2 import (  # noqa: E402
    anchor_blend, build_prior, default_monotone_groups, load_meta, score_holdout,
)


def stale_reason(df: pd.DataFrame, target_cols: List[str], n_rows: int) -> Optional[str]:
    """Why ``df`` cannot be scored on the current holdout, or None if it can.

    A file produced on an older bundle either lacks a current target column
    (e.g. after a units rename: ram_usage vs ram_usage_mb) or carries a
    different number of rows. The first would be silently scored on just the
    surviving columns — a smaller n_cells and an artificially low MAE that is
    not comparable to the others. The second cannot be scored at all: the
    holdout mask and the predictions no longer line up, and numpy raises a
    broadcast error that takes the whole comparison down with it.
    """
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        preview = ", ".join(missing[:5]) + (" …" if len(missing) > 5 else "")
        return f"missing {len(missing)}/{len(target_cols)} target cols ({preview})"
    if len(df) != n_rows:
        return f"{len(df)} rows, but this holdout has {n_rows}"
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepared-dir", required=True)
    p.add_argument("--baseline-dir", required=True,
                   help="dir with *_test_imputed.csv files from processed generated outputs")
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
    v1_path = Path(args.v1_csv)
    if not v1_path.exists():
        raise SystemExit(
            f"missing WaveStitch+ v1 test output: {v1_path}\n"
            "Run Track C step 2 first, for example:\n"
            "  python dockers/tools/WaveStitchPlus_app/run_imputation.py "
            "--prepared-dir data/processed/<name>_regularized "
            "--output-dir data/processed/<name>_generated --fast --device cpu"
        )
    meta = load_meta(prepared)
    tcols = meta["target_cols"]
    ti = pd.read_csv(prepared / "test_input.csv")
    gt = pd.read_csv(prepared / "test_gt.csv")
    train = pd.read_csv(prepared / "train.csv") if (prepared / "train.csv").exists() else None

    rows: List[Dict] = []

    # darts / pypots / imputegap baselines already on disk
    bdir = Path(args.baseline_dir)
    stale: List[tuple] = []
    for f in sorted(bdir.glob("*_test_imputed.csv")):
        method = f.name[: -len("_test_imputed.csv")]
        if method in {"wavestitchplus_v1", "wavestitchplus_v2"}:
            continue
        df = pd.read_csv(f)
        # Only compare methods scored on the SAME holdout — same target columns
        # and same rows. See stale_reason() for why mixing the others is worse
        # than skipping them.
        reason = stale_reason(df, tcols, len(ti))
        if reason:
            stale.append((method, reason))
            continue
        s = score_holdout(ti, gt, df, tcols)
        rows.append({"method": method, **s})

    if stale:
        print(f"[compare_wsp_v2] skipped {len(stale)} method(s) produced on a different "
              f"bundle (re-run them on this one):", file=sys.stderr)
        for method, reason in stale:
            print(f"  - {method}: {reason}", file=sys.stderr)

    # WaveStitch+ v1 (raw diffusion). This one is not skippable — v2 is built
    # from it — so a bundle mismatch here is a hard error with a fix to run.
    v1 = pd.read_csv(v1_path)
    reason = stale_reason(v1, tcols, len(ti))
    if reason:
        raise SystemExit(
            f"{v1_path} does not match {prepared}: {reason}.\n"
            "The v1 output was produced on an older bundle. Re-run Track C step 2:\n"
            "  python dockers/tools/WaveStitchPlus_app/run_imputation.py "
            f"--prepared-dir {prepared} --output-dir {v1_path.parent} --fast --device cpu"
        )
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
