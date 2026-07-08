#!/usr/bin/env python3
"""Long-gap evaluation — does the WaveStitch+ diffusion add value in its regime?

The default holdout masks *scattered single* observed points, which sit one step
from a real neighbour — a regime where interpolation is near-optimal and the
diffusion only adds noise. This harness instead carves **contiguous interior
gaps** out of fully-observed runs, so the scored cells lie deep inside a gap
(far from any observation) — the regime where a generative model's multivariate
/ long-range structure is the only signal interpolation can't supply.

For a sweep of gap lengths L it builds a re-masked test input (periodic
[context observed | L masked] blocks), re-runs every method on it, and scores on
the masked cells vs ground truth, both overall and bucketed by *depth* (distance
to the nearest observed row). Methods: nearest / linear interpolation (the strong
baselines), WaveStitch+ v1 (raw diffusion) and v2 (locally anchored).

Only feasible where the test split has long observed runs (e.g. python; amf
partially). golang/rabbitmq have isolated observations — no long observed gap
can be constructed, which is itself the finding for those subsets.

Usage:
    python scripts/eval_long_gap.py --prepared-dir experiments/EUR/prepared_python \
        --gap-lengths 4,8,16,32,64 --context 16 --ddim-steps 50
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = REPO_ROOT / "dockers" / "tools" / "WaveStitchPlus_app"
sys.path.insert(0, str(APP_DIR))

from wsp_v2 import (  # noqa: E402
    _distance_to_observed, anchor_blend, build_prior, default_monotone_groups,
    load_meta, score_holdout,
)
from custom_pipeline.directory_manager import get_generated_root  # noqa: E402


def build_longgap_input(
    gt: pd.DataFrame, target_cols: List[str], gap_len: int, context: int
) -> pd.DataFrame:
    """Re-mask ``gt`` into periodic [context observed | gap_len masked] blocks.

    Only originally-observed rows (all targets present in gt) are eligible to be
    masked, so every masked cell has a ground-truth value to score against.
    """
    obs_row = gt[target_cols].notna().all(axis=1).to_numpy()
    period = context + gap_len
    pos = np.arange(len(gt)) % period
    mask_row = (pos >= context) & obs_row  # mask the tail of each period
    out = gt.copy()
    tcols = [c for c in target_cols if c in out.columns]
    arr = out[tcols].to_numpy(dtype=float)
    arr[mask_row, :] = np.nan
    out[tcols] = arr
    return out


def run_diffusion(prepared: Path, test_csv: Path, out_csv: Path,
                  ddim_steps: int, repaint: int) -> pd.DataFrame:
    cmd = [
        sys.executable, str(APP_DIR / "synthesis_improved.py"),
        "-d", "custom_csv",
        "-prepared_dir", str(Path(prepared).resolve()),
        "-test_csv", str(Path(test_csv).resolve()),
        "-ignore_col_masks",
        "-out_csv", str(Path(out_csv).resolve()),
        "-model_type", "em",
        # The v1 runner trains/saves the checkpoint under the "v1" tag
        # (model_v1_best.pth); match it so synthesis loads that file instead of
        # looking for the legacy model_em.pth name.
        "-model_tag", "v1",
        "-clamp_mode", "bounds",
        "-repaint_rounds", str(repaint),
        "-ddim_steps", str(ddim_steps),
        "-n_trials", "1",
        "-bound_headroom", "1.2",
    ]
    proc = subprocess.run(cmd, cwd=str(APP_DIR), capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-2000:] + "\n")
        raise SystemExit(f"synthesis failed (exit {proc.returncode})")
    return pd.read_csv(out_csv)


DEPTH_EDGES = [(1, 1), (2, 4), (5, 8), (9, 16), (17, 9999)]


def _bucket_label(lo: int, hi: int) -> str:
    return f"d{lo}-{hi if hi < 9999 else '+'}"


def depth_buckets(ti: pd.DataFrame, gt: pd.DataFrame, preds: Dict[str, pd.DataFrame],
                  target_cols: List[str], has_left_context: bool) -> pd.DataFrame:
    """Per-depth-bucket MAE for each method (depth = dist to nearest observed)."""
    rows = []
    edges = DEPTH_EDGES
    # Collect (depth, abs_err) per method across all targets.
    per_method_depth: Dict[str, List[np.ndarray]] = {k: [[], []] for k in preds}
    for c in target_cols:
        miss = ti[c].isna().to_numpy()
        if not miss.any():
            continue
        d = _distance_to_observed(miss, left_context=1 if has_left_context else 0)
        g = gt[c].to_numpy(float)
        for name, df in preds.items():
            if c not in df.columns:
                continue
            p = df[c].to_numpy(float)
            sel = miss & ~np.isnan(g) & ~np.isnan(p)
            per_method_depth[name][0].append(d[sel])
            per_method_depth[name][1].append(np.abs(p[sel] - g[sel]))
    for name in preds:
        dd = np.concatenate(per_method_depth[name][0]) if per_method_depth[name][0] else np.array([])
        ee = np.concatenate(per_method_depth[name][1]) if per_method_depth[name][1] else np.array([])
        row = {"method": name}
        for lo, hi in edges:
            sel = (dd >= lo) & (dd <= hi)
            row[_bucket_label(lo, hi)] = (
                float(np.mean(ee[sel])) if sel.any() else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepared-dir", required=True)
    p.add_argument("--gap-lengths", default="4,8,16,32,64")
    p.add_argument("--context", type=int, default=16,
                   help="observed rows kept between consecutive masked blocks")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--repaint-rounds", type=int, default=3)
    p.add_argument("--tau", type=float, default=20.0)
    p.add_argument("--hard-prior", type=int, default=8)
    p.add_argument("--work-dir", default=None,
                   help="where to write masked inputs, diffusion outputs and the "
                        "result CSVs (default: <name>_generated/long_gap next to the "
                        "prepared bundle, so the dashboard's Long-gap tab can discover them)")
    p.add_argument("--reuse", action="store_true",
                   help="reuse an existing diffusion_L<L>.csv in the work dir "
                        "instead of re-running synthesis (instant re-scoring)")
    args = p.parse_args()

    prepared = Path(args.prepared_dir)
    meta = load_meta(prepared)
    tcols = meta["target_cols"]
    gt = pd.read_csv(prepared / "test_gt.csv")
    train = pd.read_csv(prepared / "train.csv") if (prepared / "train.csv").exists() else None
    gap_lengths = [int(x) for x in args.gap_lengths.split(",") if x.strip()]

    if args.work_dir:
        work = Path(args.work_dir)
    else:
        work = Path(get_generated_root(prepared)) / "long_gap"
    work.mkdir(parents=True, exist_ok=True)
    print(f"[long-gap] subset={prepared.name}  context={args.context}  "
          f"ddim={args.ddim_steps}  work={work}")

    overall_rows = []
    depth_rows = []
    for L in gap_lengths:
        ti = build_longgap_input(gt, tcols, gap_len=L, context=args.context)
        n_masked = int(ti[tcols].isna().sum().sum() - gt[tcols].isna().sum().sum())
        if n_masked <= 0:
            print(f"  [L={L}] no maskable cells (no observed runs ≥ {args.context}+{L}); skipping")
            continue
        ti_path = work / f"test_input_L{L}.csv"
        ti.to_csv(ti_path, index=False)

        diff_csv = work / f"diffusion_L{L}.csv"
        if args.reuse and diff_csv.exists():
            print(f"  [L={L}] reusing {diff_csv.name}")
            diff = pd.read_csv(diff_csv)
        else:
            diff = run_diffusion(prepared, ti_path, diff_csv,
                                 args.ddim_steps, args.repaint_rounds)
        prior_n = build_prior(train, ti, tcols, method="nearest")
        prior_l = build_prior(train, ti, tcols, method="linear")
        v2 = anchor_blend(ti, diff, prior_n, tcols, tau=args.tau,
                          hard_prior=args.hard_prior, has_left_context=train is not None,
                          monotone_groups=default_monotone_groups(tcols))

        preds = {
            "nearest": prior_n, "linear": prior_l,
            "wsp_v1": diff, "wsp_v2": v2,
        }
        scored = {k: score_holdout(ti, gt, v, tcols) for k, v in preds.items()}
        n_cells = scored["nearest"]["n_cells"]
        print(f"\n========== gap length L={L}  (masked≈{n_cells} cells) ==========")
        tbl = pd.DataFrame([{"method": k, **s} for k, s in scored.items()]).sort_values("MAE")
        print(tbl.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
        print("  -- MAE by depth-into-gap --")
        db = depth_buckets(ti, gt, preds, tcols, has_left_context=train is not None)
        print(db.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

        for k, s in scored.items():
            overall_rows.append({"gap_len": L, "method": k, **s})
        # Tidy (long-form) depth rows for the dashboard.
        for _, r in db.iterrows():
            for lo, hi in DEPTH_EDGES:
                lbl = _bucket_label(lo, hi)
                depth_rows.append({
                    "gap_len": L, "method": r["method"],
                    "depth_bucket": lbl, "depth_lo": lo,
                    "MAE": r[lbl],
                })

    if overall_rows:
        out = work / "long_gap_results.csv"
        pd.DataFrame(overall_rows).to_csv(out, index=False)
        pd.DataFrame(depth_rows).to_csv(work / "long_gap_depth.csv", index=False)
        print(f"\n[wrote] {out}")
        print(f"[wrote] {work / 'long_gap_depth.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
