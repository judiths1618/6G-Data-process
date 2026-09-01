#!/usr/bin/env python3
"""
Render a cross-dataset comparison table from one or more local experiments.

Walks ``experiments/<dataset>/generated_<run_id>/results_overall.csv`` and
produces a single matrix: rows=method, columns=dataset, cell=MAE (or any
metric chosen with ``--metric``).

Example::

    python scripts/consolidate_experiments.py
    python scripts/consolidate_experiments.py --metric RMSE --run-id all_baselines
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def discover(experiments_root: Path, run_id: str | None) -> dict[str, Path]:
    """Find one results_overall.csv per dataset. Prefer the requested run_id;
    fall back to the most recently modified generated_* dir per dataset."""
    out: dict[str, Path] = {}
    if not experiments_root.exists():
        return out
    for ds_dir in sorted(p for p in experiments_root.iterdir() if p.is_dir()):
        if run_id:
            match = ds_dir / f"generated_{run_id}" / "results_overall.csv"
            if match.exists():
                out[ds_dir.name] = match
                continue
        cands = sorted(
            (g for g in ds_dir.glob("generated_*/results_overall.csv")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if cands:
            out[ds_dir.name] = cands[0]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-root", type=Path,
                    default=REPO_ROOT / "experiments")
    ap.add_argument("--run-id", type=str, default=None,
                    help="Pick this run_id from each dataset (default: latest).")
    ap.add_argument("--metric", choices=["MAE", "RMSE", "MAPE_%", "elapsed_sec"],
                    default="MAE")
    args = ap.parse_args()

    files = discover(args.experiments_root, args.run_id)
    if not files:
        print(f"No experiments found under {args.experiments_root}", file=sys.stderr)
        return 1

    frames = []
    for ds, path in files.items():
        df = pd.read_csv(path)
        df["dataset"] = ds
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)

    print(f"=== {args.metric} per (method, dataset) ===\n")
    pivot = long.pivot_table(
        index="method", columns="dataset",
        values=args.metric, aggfunc="first",
    )
    # Order columns by dataset name; order rows by mean of the metric.
    pivot = pivot.reindex(
        index=pivot.mean(axis=1).sort_values().index,
        columns=sorted(pivot.columns),
    )
    with pd.option_context("display.float_format", "{:.4g}".format,
                            "display.max_rows", None,
                            "display.width", 140):
        print(pivot.to_string())

    print("\nrow-mean (across datasets):")
    rm = pivot.mean(axis=1).round(2).sort_values()
    print(rm.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
