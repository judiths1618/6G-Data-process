#!/usr/bin/env python3
"""Stitch a WaveStitch+ run's imputed splits into ONE gap-free final dataset.

The final = the imputed **train** split + the imputed **test** split (the model's
own output for both), keeping ``time`` + a ``split`` label ("train"/"test") +
target columns. The ``split`` column marks the train/test boundary. This mirrors
``dataops.imputation_runner.build_final_dataset`` but is **vendored** here,
pandas-only, because the WaveStitch+ Docker image ships ``WaveStitchPlus_app/``
without ``src/dataops`` — so the runners can build a final natively and
in-container without that dependency.

Given ``output_dir`` holding
``wavestitchplus_<variant>_{train,test}_imputed.csv``, writes
``wavestitchplus_<variant>_final.csv`` next to them.

CLI:
    python wsp_final.py --prepared-dir <prepared_dir> --output-dir <gen_dir> \
        --variant v1
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SPLITS = ("train", "test")


def build_wsp_final(
    prepared_dir: str | Path,
    output_dir: str | Path,
    variant: str = "v1",
    *,
    keep_cond_features: bool = False,
) -> Path | None:
    """Concatenate the imputed train+test splits into a gap-free final CSV.

    Returns the final path, or ``None`` if no imputed split was found.
    """
    prepared = Path(prepared_dir)
    out_dir = Path(output_dir)
    meta = json.loads((prepared / "meta.json").read_text())
    time_col = meta.get("time_col", "time")
    target_cols = list(meta.get("target_cols", []))
    cond_cols = list(meta.get("cond_cols", [])) if keep_cond_features else []

    frames = []
    for split in SPLITS:
        fp = out_dir / f"wavestitchplus_{variant}_{split}_imputed.csv"
        if fp.exists():
            fr = pd.read_csv(fp)
            fr["split"] = split  # keep the train/test boundary explicit in the final
            frames.append(fr)
    if not frames:
        print(f"[WaveStitch+ final] no imputed splits for variant={variant!r} "
              f"in {out_dir}; skip")
        return None

    full = pd.concat(frames, ignore_index=True)
    keep = [c for c in [time_col, "split", *target_cols, *cond_cols] if c in full.columns]
    full = (full[keep]
            .sort_values(time_col)
            .drop_duplicates(subset=[time_col], keep="last")
            .reset_index(drop=True))

    final_path = out_dir / f"wavestitchplus_{variant}_final.csv"
    full.to_csv(final_path, index=False)
    present_targets = [c for c in target_cols if c in full.columns]
    residual = int(full[present_targets].isna().sum().sum()) if present_targets else 0
    print(f"[WaveStitch+ final] {variant}: {len(full):,} rows · {len(keep)} cols · "
          f"residual NaN {residual:,} → {final_path}")
    return final_path


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepared-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--variant", default="v1",
                   help="split tag, e.g. v1 / v2 / v2_tuned / harpoon")
    p.add_argument("--keep-cond-features", action="store_true")
    a = p.parse_args()
    build_wsp_final(a.prepared_dir, a.output_dir, a.variant,
                    keep_cond_features=a.keep_cond_features)


if __name__ == "__main__":
    main()
