#!/usr/bin/env python3
"""
Time-series **forecasting** experiment: does WaveStitchPlus-imputed training
data help a downstream forecaster compared to simple linear interpolation?

Pattern A from the design discussion: WSP is used as a synthetic-data
*generator*. The "synthesis" here is the train_imputed_denorm.csv it produced
during its own training/imputation pass — the same train rows, but with
regularize-induced gaps filled by the diffusion model instead of a 1-D
interpolator. We then train an identical Darts forecaster under two fill
strategies and compare forecasts of the held-out tail.

Pipeline (per dataset, per target column)::

    train.csv  ─┬─→ [linear interp + ffill]  ──→ train_ts_linear ──┐
                │                                                   ├─→ same forecaster (LinearRegressionModel)
    train_imputed_denorm.csv (WSP fill) ──→ train_ts_wsp ──────────┘
                ↓ (same time index)
    Hold out last h rows = test_tail
    Train each forecaster on first T-h rows
    Forecast h steps → score against test_tail cells that were
    NON-NaN in the original train.csv (so we use ground-truth only,
    never any imputed value as a "truth label")

Outputs land in ``experiments/<dataset>/forecast_<run_id>/`` so the layout is
consistent with the imputation experiments. Two CSVs per variant per column
are saved alongside a per-target results.csv.

Example::

    python scripts/forecast_experiment.py \\
        --dataset amf \\
        --horizon 32 --lags 24

    # multi-dataset:
    for ds in amf golang python rabbitmq; do
        python scripts/forecast_experiment.py --dataset $ds
    done
    python scripts/consolidate_experiments.py --run-id forecast_default
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# Darts is available in the myenv env.
from darts import TimeSeries
from darts.models import LinearRegressionModel, NaiveDrift


def _load_prepared(dataset: str, wsp_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (raw_train, wsp_filled_train, meta) for a dataset."""
    pdir = wsp_root / f"prepared_{dataset}"
    if not (pdir / "meta.json").exists():
        raise SystemExit(f"missing prepared dir: {pdir}")
    raw = pd.read_csv(pdir / "train.csv")
    wsp = pd.read_csv(pdir / "train_imputed_denorm.csv")
    meta = json.loads((pdir / "meta.json").read_text())
    return raw, wsp, meta


def _fill_linear(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Linear interpolate per column, then forward/back-fill any edges."""
    out = df.copy()
    out[cols] = (
        out[cols]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )
    return out


def _forecast_one_column(
    train_values: np.ndarray,
    horizon: int,
    lags: int,
    method: str,
) -> np.ndarray:
    """Fit a model on ``train_values`` and forecast ``horizon`` steps ahead.

    Returns predictions of length ``horizon``. Falls back to repeating the
    last observed value if the model fails (e.g., constant series).
    """
    # Darts expects a 2-D dataframe with a time index.
    s = pd.Series(train_values, name="y")
    if s.isna().any():
        # Should not happen after fill; defensive fallback.
        s = s.interpolate(limit_direction="both").ffill().bfill()
    ts = TimeSeries.from_series(s)

    try:
        if method == "linreg":
            # Cap lags at train length // 4 so the model has data to fit.
            eff_lags = max(1, min(lags, len(ts) // 4))
            model = LinearRegressionModel(lags=eff_lags)
            model.fit(ts)
            pred = model.predict(horizon)
        elif method == "naive_drift":
            model = NaiveDrift()
            model.fit(ts)
            pred = model.predict(horizon)
        else:
            raise ValueError(f"unknown forecaster method: {method!r}")
        return np.asarray(pred.values().squeeze(), dtype=float).reshape(horizon)
    except Exception as e:
        # Fallback: last-value-carried-forward
        last = float(train_values[-1]) if len(train_values) else 0.0
        return np.full(horizon, last)


def _score(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    """MAE/RMSE/MAPE on positions where ``mask`` is True (cell had ground truth)."""
    if mask.sum() == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"),
                "MAPE_%": float("nan"), "n_cells": 0}
    err = pred[mask] - truth[mask]
    err = err[~np.isnan(err)]
    if len(err) == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"),
                "MAPE_%": float("nan"), "n_cells": 0}
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    gt = truth[mask]
    pr = pred[mask]
    nz = ~np.isnan(gt) & ~np.isnan(pr) & (np.abs(gt) > 1e-12)
    mape = float(np.mean(np.abs((pr[nz] - gt[nz]) / gt[nz])) * 100) if nz.any() else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape, "n_cells": int(len(err))}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True,
                   choices=["amf", "golang", "python", "rabbitmq"])
    p.add_argument("--wsp-root", type=Path,
                   default=REPO_ROOT / "experiments" / "EUR")
    p.add_argument("--output-root", type=Path,
                   default=REPO_ROOT / "experiments")
    p.add_argument("--run-id", type=str, default="forecast_default")
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--lags", type=int, default=24)
    p.add_argument("--forecaster", choices=["linreg", "naive_drift"],
                   default="linreg")
    args = p.parse_args()

    raw_df, wsp_df, meta = _load_prepared(args.dataset, args.wsp_root)
    ts_col = meta.get("time_col", "time")
    target_cols = meta["target_cols"]

    if len(raw_df) <= args.horizon + args.lags * 2:
        raise SystemExit(
            f"dataset too short: T={len(raw_df)} < lags*2 + horizon "
            f"({args.lags*2 + args.horizon}). Pick a smaller --lags or --horizon."
        )

    T = len(raw_df)
    n_train = T - args.horizon
    # Names of the dataset/run dirs follow the same pattern as the
    # imputation experiments so consolidate_experiments.py can pick them up.
    dataset_name = {
        "amf": "amf-performance",
        "golang": "golang-web-server-performance",
        "python": "python-web-server-performance",
        "rabbitmq": "rabbitmq-performance",
    }[args.dataset]
    out_dir = args.output_root / dataset_name / f"generated_{args.run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fill strategies — both produce dense T-row frames with the same time axis.
    fills = {
        "linear": _fill_linear(raw_df, target_cols),
        "wsp":    wsp_df,
    }

    # Ground truth for the held-out tail = the ORIGINAL train.csv values.
    # Score only on cells that were non-NaN in the raw data — never compare
    # against any imputed value (linear or WSP) as if it were ground truth.
    tail_truth = raw_df.iloc[n_train:][target_cols].to_numpy(dtype=float)
    tail_mask = ~np.isnan(tail_truth)  # (h, F)

    print(f"\n[INFO] dataset={dataset_name}")
    print(f"[INFO] T={T}  n_train={n_train}  horizon={args.horizon}  lags={args.lags}")
    print(f"[INFO] target_cols ({len(target_cols)}): {target_cols}")
    print(f"[INFO] tail ground-truth cells: {int(tail_mask.sum())} / {tail_mask.size}")
    print(f"[INFO] forecaster: {args.forecaster}")
    print(f"[INFO] output: {out_dir}\n")

    rows = []
    overall_per_variant: dict = {v: {"err": [], "ratios": [], "n": 0,
                                     "elapsed": 0.0}
                                 for v in fills}

    # For each (variant, column) train a forecaster and predict h steps.
    for variant, df_filled in fills.items():
        print(f"=== variant: {variant} ===")
        t0 = time.perf_counter()
        # Persist the "training" filled CSV (useful for the dashboard).
        train_csv_path = out_dir / f"{args.forecaster}_{variant}_train.csv"
        df_filled.iloc[:n_train].to_csv(train_csv_path, index=False)

        # Persist forecasts (same shape as test slice) for the dashboard.
        pred_df = raw_df.iloc[n_train:].copy()
        for col in target_cols:
            train_vals = df_filled[col].to_numpy(dtype=float)[:n_train]
            preds = _forecast_one_column(
                train_vals, args.horizon, args.lags, args.forecaster,
            )
            pred_df[col] = preds

            # Score
            j = target_cols.index(col)
            truth = tail_truth[:, j]
            mask = tail_mask[:, j]
            s = _score(preds, truth, mask)
            rows.append({"variant": variant, "method": args.forecaster,
                         "target": col, **s})

            # Accumulate overall stats
            valid_err = (preds[mask] - truth[mask])
            valid_err = valid_err[~np.isnan(valid_err)]
            overall_per_variant[variant]["err"].extend(valid_err.tolist())
            gt = truth[mask]; pr = preds[mask]
            nz = ~np.isnan(gt) & ~np.isnan(pr) & (np.abs(gt) > 1e-12)
            ratios = np.abs((pr[nz] - gt[nz]) / gt[nz])
            overall_per_variant[variant]["ratios"].extend(ratios.tolist())
            overall_per_variant[variant]["n"] += int(mask.sum())

        # Save the prediction CSV (mirrors dashboard's <method>_test_imputed.csv)
        pred_df.to_csv(
            out_dir / f"{args.forecaster}_{variant}_forecast.csv",
            index=False,
        )
        overall_per_variant[variant]["elapsed"] = time.perf_counter() - t0
        print(f"    columns processed: {len(target_cols)}  "
              f"elapsed={overall_per_variant[variant]['elapsed']:.1f}s")

    # Aggregate
    per_target_df = pd.DataFrame(rows)
    per_target_df.to_csv(out_dir / "results.csv", index=False)

    overall_rows = []
    for variant, agg in overall_per_variant.items():
        e = np.asarray(agg["err"])
        r = np.asarray(agg["ratios"])
        overall_rows.append({
            "variant": variant,
            "method":  args.forecaster,
            "MAE":     float(np.mean(np.abs(e))) if len(e) else float("nan"),
            "RMSE":    float(np.sqrt(np.mean(e**2))) if len(e) else float("nan"),
            "MAPE_%":  float(np.mean(r) * 100) if len(r) else float("nan"),
            "n_cells": int(agg["n"]),
            "elapsed_sec": agg["elapsed"],
        })
    overall_df = pd.DataFrame(overall_rows).sort_values("MAE")
    overall_df.to_csv(out_dir / "results_overall.csv", index=False)

    print("\n" + "=" * 78)
    print(f"FORECAST RESULTS  ({args.forecaster}, h={args.horizon}, "
          f"dataset={dataset_name})")
    print("=" * 78)
    with pd.option_context("display.float_format", "{:.4g}".format,
                           "display.max_rows", None, "display.width", 130):
        print(overall_df.to_string(index=False))

    # Per-target winners
    if not per_target_df.empty:
        print("\n--- per-target MAE: linear vs wsp ---")
        pivot = per_target_df.pivot_table(
            index="target", columns="variant", values="MAE", aggfunc="first"
        )
        if {"linear", "wsp"} <= set(pivot.columns):
            pivot["delta(wsp-linear)"] = pivot["wsp"] - pivot["linear"]
            pivot["wsp_better"] = pivot["delta(wsp-linear)"] < 0
        with pd.option_context("display.float_format", "{:.4g}".format,
                               "display.max_rows", None, "display.width", 130):
            print(pivot.to_string())

    print(f"\n→ {out_dir / 'results.csv'}")
    print(f"→ {out_dir / 'results_overall.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
