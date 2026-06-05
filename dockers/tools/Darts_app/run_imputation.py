#!/usr/bin/env python3
"""
Darts baseline imputation runner (train.csv + test_input.csv).

Reads a `prepared_<subset>/` directory produced by the WaveStitch+ pipeline
(meta.json + train.csv + test_input.csv) and writes one imputed CSV per input
into `--output-dir`:

    darts_<method>_train_imputed.csv
    darts_<method>_test_imputed.csv

The output schema matches `wavestitchplus_v1_test_imputed.csv` so the comparison
notebooks can pick them up automatically.

Methods (univariate, applied per target column):
    auto                                              -> Darts default heuristic
    linear, quadratic, cubic, nearest, slinear, zero  -> Darts MissingValuesFiller
                                                         (forwarded to pandas.interpolate)
    kalman                                            -> KalmanFilter fit on observed pts
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller

INTERP_METHODS = {"auto", "linear", "quadratic", "cubic", "nearest", "slinear", "zero"}
KALMAN_METHODS = {"kalman"}
SUPPORTED = INTERP_METHODS | KALMAN_METHODS

INPUT_FILES = {"train": "train.csv", "test": "test_input.csv"}


def load_meta(prepared_dir: Path) -> dict:
    with (prepared_dir / "meta.json").open() as f:
        return json.load(f)


def to_series(df: pd.DataFrame, value_col: str) -> TimeSeries:
    sub = df[[value_col]].copy()
    sub["__idx__"] = np.arange(len(sub))
    return TimeSeries.from_dataframe(sub, time_col="__idx__", value_cols=value_col)


def impute_interp(series: TimeSeries, method: str) -> TimeSeries:
    # In darts 0.33: MissingValuesFiller(fill='auto') uses pandas.interpolate;
    # extra kwargs (e.g. method='cubic') are forwarded to pandas.interpolate.
    filler = MissingValuesFiller(fill="auto")
    if method == "auto":
        return filler.transform(series)
    return filler.transform(series, method=method)


def impute_kalman(series: TimeSeries) -> TimeSeries:
    from darts.models import KalmanFilter

    values = series.values().squeeze()
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return impute_interp(series, "auto")

    # Darts 0.33 requires the TimeSeries index to be a contiguous RangeIndex,
    # so we re-index the observed values densely and map back afterwards.
    observed = TimeSeries.from_values(values[mask].reshape(-1, 1))
    kf = KalmanFilter(dim_x=1)
    kf.fit(observed)
    smoothed = kf.filter(observed).values().squeeze()

    out = values.copy()
    obs_positions = np.where(mask)[0]
    out[obs_positions] = smoothed
    holes = np.where(~mask)[0]
    if holes.size:
        out[holes] = np.interp(holes, obs_positions, out[obs_positions])
    return TimeSeries.from_values(out.reshape(-1, 1))


def impute_dataframe(df: pd.DataFrame, target_cols: List[str], method: str) -> pd.DataFrame:
    out = df.copy()
    for col in target_cols:
        if col not in out.columns:
            print(f"[Darts]   WARN: target column '{col}' missing, skipped")
            continue
        if out[col].notna().sum() == 0:
            print(f"[Darts]   WARN: column '{col}' is all-NaN, skipped")
            continue
        series = to_series(out, col)
        if method in INTERP_METHODS:
            filled = impute_interp(series, method)
        else:
            filled = impute_kalman(series)
        out[col] = filled.values().squeeze()
        # Edge NaNs left by some interpolation methods → forward/back fill.
        if out[col].isna().any():
            out[col] = out[col].ffill().bfill()
    return out


def run(args: argparse.Namespace) -> List[Path]:
    prepared = Path(args.prepared_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(prepared)
    target_cols = meta["target_cols"]

    written: List[Path] = []
    for kind in args.inputs:
        src = prepared / INPUT_FILES[kind]
        if not src.exists():
            print(f"[Darts]   skip: {src} not found")
            continue
        df = pd.read_csv(src)
        nan_before = int(df[target_cols].isna().sum().sum())
        print(f"[Darts] {kind}: shape={df.shape}  NaN target cells={nan_before}")
        out_df = impute_dataframe(df, target_cols, args.method)
        nan_after = int(out_df[target_cols].isna().sum().sum())
        print(f"[Darts] {kind}: NaN target cells {nan_before} -> {nan_after}")

        out_path = output_dir / f"darts_{args.method}_{kind}_imputed.csv"
        out_df.to_csv(out_path, index=False)
        print(f"[Darts] wrote {out_path}")
        written.append(out_path)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Darts baseline imputation runner")
    p.add_argument("--prepared-dir", required=True,
                   help="prepared_<subset>/ folder (must contain meta.json, train.csv, test_input.csv)")
    p.add_argument("--output-dir", required=True,
                   help="Where to write darts_<method>_<train|test>_imputed.csv")
    p.add_argument("--method", default="auto", choices=sorted(SUPPORTED))
    p.add_argument("--inputs", nargs="+", default=["train", "test"],
                   choices=list(INPUT_FILES.keys()),
                   help="Which CSVs to impute (default: both)")
    args = p.parse_args()

    if args.method not in SUPPORTED:
        print(f"Unsupported method: {args.method}; choose from {sorted(SUPPORTED)}",
              file=sys.stderr)
        sys.exit(2)

    run(args)


if __name__ == "__main__":
    main()
