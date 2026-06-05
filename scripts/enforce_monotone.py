#!/usr/bin/env python3
"""Apply the per-row monotone-group projection to any imputed CSV (optional v1 pass).

WaveStitch+ v2 enforces monotone groups (e.g. lat50 ≤ … ≤ lat100) by default;
this standalone pass applies the same projection to an *existing* imputed CSV —
e.g. the raw v1 ``wavestitchplus_test_imputed.csv`` — without re-running anything.
Only originally-missing cells (NaN in the prepared ``test_input.csv``) are
modified; observed cells are preserved exactly.

    python scripts/enforce_monotone.py \
        --prepared-dir experiments/EUR/prepared_amf \
        --imputed-csv  experiments/EUR/generated_amf/wavestitchplus_test_imputed.csv \
        [--out <csv>]   # default: overwrite in place
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent.parent / "dockers" / "tools" / "WaveStitchPlus_app"
sys.path.insert(0, str(APP_DIR))

from wsp_v2 import default_monotone_groups, enforce_monotone_groups, load_meta  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepared-dir", required=True)
    p.add_argument("--imputed-csv", required=True)
    p.add_argument("--out", default=None, help="output CSV (default: overwrite input)")
    args = p.parse_args()

    prepared = Path(args.prepared_dir)
    meta = load_meta(prepared)
    tcols = meta["target_cols"]
    groups = default_monotone_groups(tcols)
    if not groups:
        print("[enforce_monotone] no ordered groups detected; nothing to do")
        return 0

    ti = pd.read_csv(prepared / "test_input.csv")
    df = pd.read_csv(args.imputed_csv)
    group_cols = [c for g in groups for c in g if c in df.columns]
    before = int((np.diff(df[group_cols].to_numpy(float), axis=1) < -1e-6).any(axis=1).sum())

    missing = ti[tcols].isna().reset_index(drop=True)
    enforce_monotone_groups(df, missing, groups)
    after = int((np.diff(df[group_cols].to_numpy(float), axis=1) < -1e-6).any(axis=1).sum())

    out = Path(args.out) if args.out else Path(args.imputed_csv)
    df.to_csv(out, index=False)
    print(f"[enforce_monotone] groups={groups}")
    print(f"[enforce_monotone] non-monotone rows: {before} → {after}   wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
