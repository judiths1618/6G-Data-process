from __future__ import annotations

import json

import numpy as np
import pandas as pd
from airflow.models import Variable
from helpers.object_store import load_df_from_object_store, save_df_to_object_store


# helpers/clean_dirty_data.py

def _get_s3_client():
    """Shared S3 client, kept in sync with run_pipeline.py."""
    import boto3
    from botocore.client import Config

    return boto3.client(
        "s3",
        endpoint_url=Variable.get("S3_ENDPOINT",   default_var="http://seaweed-s3:8333"),
        aws_access_key_id=Variable.get("S3_ACCESS_KEY", default_var="anykey"),
        aws_secret_access_key=Variable.get("S3_SECRET_KEY", default_var="anysecret"),
        region_name="us-east-1",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"}
        ),
    )


def _s3_upload_string(content: str, bucket: str, key: str) -> None:
    """上传字符串内容（替代 S3Hook.load_string）"""
    client = _get_s3_client()
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=content.encode("utf-8"),
    )
    print(f"  [S3] ↑ s3://{bucket}/{key}")


def _s3_upload_bytes(data: bytes, bucket: str, key: str) -> None:
    """上传二进制内容（替代 S3Hook.load_bytes，用于图片）"""
    client = _get_s3_client()
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
    )
    print(f"  [S3] ↑ s3://{bucket}/{key}")

# ─────────────────────────────────────────────
# Sub-module 1: Deduplication
# ─────────────────────────────────────────────

def _clean_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove exact duplicate rows, keeping last occurrence."""
    before = len(df)
    df = df.drop_duplicates(keep="last").reset_index(drop=True)
    after = len(df)
    stats = {"removed": before - after, "remaining": after}
    print(f"  [DEDUP] Removed {before - after} duplicate rows ({before} → {after})")
    return df, stats


# ─────────────────────────────────────────────
# Sub-module 2: Structural fixes
# ─────────────────────────────────────────────

def _fix_structural(
    df: pd.DataFrame,
    ts_col: str | None,
    failed_cols: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Fix structural issues:
    - Drop rows with null timestamps
    - Enforce timestamp monotonicity
    - Coerce failed columns to numeric
    """
    fixes = []

    if ts_col and ts_col in df.columns:
        null_ts = int(df[ts_col].isna().sum())
        if null_ts > 0:
            df = df.dropna(subset=[ts_col])
            fixes.append(f"Dropped {null_ts} rows with null timestamps")

        df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce")
        df = df.sort_values(ts_col).reset_index(drop=True)
        fixes.append("Sorted by timestamp (enforced monotonicity)")

    for col in failed_cols:
        if col and col in df.columns and col != ts_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            fixes.append(f"Coerced '{col}' to numeric")

    print(f"  [STRUCTURAL] Applied: {fixes}")
    return df, {"fixes_applied": fixes}


# ─────────────────────────────────────────────
# Sub-module 3: Outlier handling
# ─────────────────────────────────────────────

def _handle_outliers(
    df: pd.DataFrame,
    outlier_cols: list[str],
    ts_col: str | None,
) -> tuple[pd.DataFrame, dict]:
    """Clip numeric outlier columns to [1st, 99th] percentile."""
    stats = {}
    for col in outlier_cols:
        if col not in df.columns or col == ts_col:
            continue
        q_low  = df[col].quantile(0.01)
        q_high = df[col].quantile(0.99)
        n_clipped = int(((df[col] < q_low) | (df[col] > q_high)).sum())
        df[col] = df[col].clip(lower=q_low, upper=q_high)
        stats[col] = {
            "clipped": n_clipped,
            "range": [float(q_low), float(q_high)],
        }
    print(f"  [OUTLIERS] Stats: {stats}")
    return df, stats


# ─────────────────────────────────────────────
# Sub-module 4: Tabular imputation (CPU)
# ─────────────────────────────────────────────

def _tabular_imputation(
    df: pd.DataFrame,
    missing_info: dict,
    ts_col: str | None,
) -> tuple[pd.DataFrame, dict]:
    """
    CPU-based imputation for non-TS missing values:
    - Numeric  → median
    - Categorical / boolean → mode
    """
    stats = {}
    for col, info in missing_info.items():
        if col not in df.columns or col == ts_col:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue

        dtype = info.get("dtype", "numeric")

        if dtype == "numeric":
            fill_val = float(df[col].median())
            df[col]  = df[col].fillna(fill_val)
            stats[col] = {"method": "median", "fill_value": fill_val, "imputed": n_missing}

        elif dtype in ("categorical", "boolean"):
            mode = df[col].mode()
            fill_val = mode.iloc[0] if not mode.empty else ("UNKNOWN" if dtype == "categorical" else False)
            df[col]  = df[col].fillna(fill_val)
            stats[col] = {"method": "mode", "fill_value": str(fill_val), "imputed": n_missing}

    print(f"  [TABULAR_IMPUTE] Stats: {stats}")
    return df, stats


# ─────────────────────────────────────────────
# Sub-module 5(pre): regularize + split via preprocess_csv (shared with WaveStitchPlus)
# ─────────────────────────────────────────────
#
# ``preprocess_csv`` lives next to us in ``helpers/preprocess.py``. We invoke
# it with the **same arguments WaveStitchPlus's training scripts use** so the
# train/test split and holdout mask the DAG produces match exactly what
# WaveStitchPlus would produce on the same input. That keeps any subsequent
# method comparison (WaveStitchPlus vs darts_linear vs pypots_*) honest:
# every method sees the same train/test/holdout cells.

# Defaults mirroring the WaveStitchPlus callsites in
# train_improved.py / train_wavestitchPlus_customdata.py / train_wavestitch_customdata.py.
# Keep this dict as the single source of truth — if WaveStitchPlus changes its
# preprocess args, update here too.
_PREPROCESS_ARGS_MATCHING_WAVESTITCHPLUS = dict(
    base_dt=None,                       # auto-infer
    extract_main_segment=True,          # WSP uses longest segment only
    skip_regularize_if_sparse=True,
    convert_units=True,                 # 6G-schema renames (no-op on non-6G)
    add_cond_features=True,             # t_norm / sin_day / gap-depth conds
    # preprocess_csv defaults: split_ratio=0.8, holdout_frac=0.15, seed=0
)


def _prepare_via_preprocess_csv(
    df: pd.DataFrame,
    ts_col: str,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, "np.ndarray", dict, Path]:
    """
    Run :func:`helpers.preprocess.preprocess_csv` with WaveStitchPlus's exact
    args and return the artifacts the imputer + downstream comparison need.

    Returns
    -------
    train_df : DataFrame             same as ``prepared_<subset>/train.csv``
    test_input_df : DataFrame        test rows with holdout cells set to NaN
    test_gt_df : DataFrame           ground-truth values at holdout positions
    holdout_mask : np.ndarray[bool]  row-level holdout indicator, length=T_test
    prep_meta : dict                 preprocess_csv's ``meta.json`` payload
    prepared_dir : Path              tmp dir on disk (kept alive by caller)

    The caller is responsible for the lifetime of ``prepared_dir`` — wrap the
    whole imputation+save block in a ``TemporaryDirectory`` context.
    """
    import tempfile
    from pathlib import Path
    import numpy as np
    from helpers.preprocess import preprocess_csv

    tmp = Path(tempfile.mkdtemp(prefix=f"prep_{dataset_name}_"))
    in_csv = tmp / "input.csv"
    out_dir = tmp / "prepared"
    df.to_csv(in_csv, index=False)

    prep_meta = preprocess_csv(
        input_csv=str(in_csv),
        output_dir=str(out_dir),
        time_col=ts_col,
        **_PREPROCESS_ARGS_MATCHING_WAVESTITCHPLUS,
    )

    train_df = pd.read_csv(out_dir / "train.csv")
    test_input_df = pd.read_csv(out_dir / "test_input.csv")
    test_gt_df = pd.read_csv(out_dir / "test_gt.csv")
    holdout_mask = np.load(out_dir / "eval_holdout_mask.npy")
    return train_df, test_input_df, test_gt_df, holdout_mask, prep_meta, out_dir


# ─────────────────────────────────────────────
# Sub-module 5a: TS imputation via in-process baselines
# ─────────────────────────────────────────────

# Methods chosen for the in-process path. Selected via Airflow Variable
# ``N2N_IMPUTER`` (default ``darts_linear``).
#
# ImputeGAP is intentionally NOT included here: it transitively requires PyQt5,
# whose source build needs Qt/qmake (and no aarch64 prebuilt wheels exist on
# the Airflow base image). Use the standalone ImputeGAP container via
# DockerOperator if you need ``imputegap_*`` methods.
_BASELINE_METHODS = {
    "darts_linear",
    "darts_cubic",
    "darts_nearest",
    "darts_kalman",
    "darts_auto",
    "pypots_saits",
    "pypots_brits",
}


def _impute_darts(df: pd.DataFrame, target_cols: list, method: str) -> pd.DataFrame:
    """One Darts method, applied per target column."""
    from darts import TimeSeries
    from darts.dataprocessing.transformers import MissingValuesFiller

    out = df.copy()
    for c in target_cols:
        if c not in out.columns:
            continue
        sub = out[[c]].copy()
        sub["__idx__"] = np.arange(len(sub))
        series = TimeSeries.from_dataframe(sub, time_col="__idx__", value_cols=c)

        if method == "darts_kalman":
            from darts.models import KalmanFilter

            vals = series.values().squeeze()
            mask = ~np.isnan(vals)
            if mask.sum() < 2:
                filler = MissingValuesFiller(fill="auto")
                filled = filler.transform(series).values().squeeze()
            else:
                observed = TimeSeries.from_values(vals[mask].reshape(-1, 1))
                kf = KalmanFilter(dim_x=1)
                kf.fit(observed)
                smoothed = kf.filter(observed).values().squeeze()
                filled = vals.copy()
                filled[mask] = smoothed
                holes = np.where(~mask)[0]
                if holes.size:
                    # Linearly interpolate at hole positions over the smoothed obs.
                    obs_idx = np.where(mask)[0]
                    filled[holes] = np.interp(holes, obs_idx, smoothed)
        else:
            kind = method.split("_", 1)[1]  # 'linear', 'cubic', 'nearest', 'auto'
            filler = MissingValuesFiller(fill="auto")
            filled = (filler.transform(series)
                      if kind == "auto"
                      else filler.transform(series, method=kind)).values().squeeze()
        out[c] = filled
    return out


def _impute_pypots(df: pd.DataFrame, target_cols: list, method: str,
                   epochs: int = 30) -> pd.DataFrame:
    """PyPOTS SAITS / BRITS, fit-then-impute on a single window."""
    import torch

    cols = [c for c in target_cols if c in df.columns]
    X = df[cols].to_numpy(dtype=float)
    # PyPOTS expects shape (n_samples, n_steps, n_features). Treat the whole
    # series as one sample.
    X3 = X.reshape(1, X.shape[0], X.shape[1])
    n_steps = X3.shape[1]
    n_features = X3.shape[2]
    kind = method.split("_", 1)[1]
    common = dict(n_steps=n_steps, n_features=n_features,
                  epochs=epochs, batch_size=1,
                  device="cuda" if torch.cuda.is_available() else "cpu")
    if kind == "saits":
        from pypots.imputation import SAITS
        model = SAITS(n_layers=2, d_model=64, n_heads=2,
                      d_k=32, d_v=32, d_ffn=128, dropout=0.0, **common)
    elif kind == "brits":
        from pypots.imputation import BRITS
        model = BRITS(rnn_hidden_size=64, **common)
    else:
        raise ValueError(f"Unsupported pypots method: {method}")
    model.fit({"X": X3})
    recovered = model.impute({"X": X3})
    if isinstance(recovered, dict):
        recovered = recovered.get("imputation", recovered)
    recovered = np.asarray(recovered).reshape(n_steps, n_features)
    out = df.copy()
    out[cols] = recovered
    return out


def _ts_imputation_baseline(df: pd.DataFrame, ts_col: str,
                            target_cols: list, method: str) -> tuple[pd.DataFrame, dict]:
    """
    In-process baseline imputation. Returns (imputed_df, stats).

    ``method`` is one of the keys in ``_BASELINE_METHODS``.
    """
    if method not in _BASELINE_METHODS:
        raise ValueError(
            f"Unknown imputer '{method}'. Pick one of: {sorted(_BASELINE_METHODS)}"
        )

    # Limit to numeric target columns; non-numeric / ts column passes through.
    numeric_targets = [
        c for c in target_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    missing_before = int(df[numeric_targets].isna().sum().sum())
    print(f"  [BASELINE_IMPUTE] method={method} cols={len(numeric_targets)} "
          f"NaN cells before={missing_before}")

    if method.startswith("darts_"):
        out = _impute_darts(df, numeric_targets, method)
    elif method.startswith("pypots_"):
        epochs = int(Variable.get("N2N_PYPOTS_EPOCHS", default_var="30"))
        out = _impute_pypots(df, numeric_targets, method, epochs=epochs)
    else:
        raise ValueError(f"Method dispatch missing for '{method}'")

    missing_after = int(out[numeric_targets].isna().sum().sum())
    stats = {
        "method": method,
        "columns_imputed": numeric_targets,
        "cells_filled": missing_before - missing_after,
        "cells_remaining_nan": missing_after,
    }
    print(f"  [BASELINE_IMPUTE] {method}: filled {stats['cells_filled']} cells, "
          f"{missing_after} remain NaN")
    return out, stats


# ─────────────────────────────────────────────
# Sub-module 5b: TS imputation via WaveStitch+
# (invoked as subprocess / API call to GPU container)
# ─────────────────────────────────────────────

def _ts_imputation_via_docker(
    handle: dict,
    meta: dict,
    run_id: str,
) -> dict:
    """
    Trigger GPU-based TS imputation by calling the WaveStitch+ container
    via Docker SDK. Returns the S3 handle of the imputed output.
    """
    import docker

    client = docker.from_env()

    dataset_name = meta.get("dataset_name", "dataset")
    ts_col       = meta.get("timestamp_column", "time")
    target_cols  = meta.get("target_cols", [])

    print(f"  [TS_IMPUTE] Launching WaveStitch+ container for dataset '{dataset_name}'...")

    # logs = client.containers.run(
    #     image="wavestitchplus-gpu:latest",
    #     command=[
    #         "python", "/app/run_pipeline.py",
    #         "--mode",         "full",
    #         "--dataset-name", dataset_name,
    #         "--version",      run_id,
    #         "--input-s3-key", handle["key"],
    #         "--time-col",     ts_col,
    #         "--target-cols",  ",".join(target_cols),
    #         "--epochs",       "500",
    #         "--batch-size",   "512",
    #         "--window-size",  "32",
    #         "--n-trials",     "1",
    #         "--guidance-scale", "0.1",
    #         "--use-em",
    #         "--em-iterations",  "5",
    #         "--epochs-per-em",  "200",
    #     ],
    logs = client.containers.run(
    image="wavestitchplus-gpu:latest",
    command=[
        "python", "/app/run_pipeline.py",
        "--mode", "full",
        "--dataset-name", dataset_name,
        "--version", run_id,
        "--input-s3-key", handle["key"],
        "--time-col", ts_col,
        "--target-cols", ",".join(target_cols),
        "--use-em",
        "--em-iterations", "5",
        "--epochs-per-em", "200",
        "--model-type", "em",
        "--clamp-mode", "bounds",
    ],
        environment={
            "S3_ENDPOINT":   "http://seaweed-s3:8333",
            "S3_ACCESS_KEY": "anykey",
            "S3_SECRET_KEY": "anysecret",
            "S3_BUCKET":     handle["bucket"],
        },
        device_requests=[
            docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])
        ],
        network="dockers_airflow_net",
        remove=True,
        stdout=True,
        stderr=True,
    )

    for line in logs.decode("utf-8").splitlines():
        print(f"    [WaveStitch+] {line}")

    imputed_key = f"wavestitchplus/{dataset_name}/latest_inference/imputed.csv"
    print(f"  [TS_IMPUTE] Done. Output: s3://{handle['bucket']}/{imputed_key}")

    return {"bucket": handle["bucket"], "key": imputed_key, "format": "csv"}


# ─────────────────────────────────────────────
# Main clean_dirty_data callable (single DAG task)
# ─────────────────────────────────────────────

def clean_dirty_data(**context):
    """
    Single DAG task that internally routes cleaning actions
    based on QC recommendations:

        1. drop_duplicates       → always
        2. fix_structural        → if structural_fix == True
        3. handle_outliers       → if outlier_handling == True
        4. ts_imputation         → if ts_imputation == True  (GPU via Docker)
        5. tabular_imputation    → elif tabular_imputation == True (CPU)
    """
    ti = context["ti"]

    # ── Pull context ─────────────────────────
    meta    = ti.xcom_pull(task_ids="load_raw_data", key="dataset_meta")
    handle  = ti.xcom_pull(task_ids="load_raw_data", key="raw_handle")
    report  = ti.xcom_pull(task_ids="report_dqc",   key="dqc_report")
    run_id  = context["run_id"]

    recs         = report.get("quality_report", {}).get("recommendations", {})
    diagnostics  = report.get("full_diagnostics", {})
    ts_col       = meta.get("timestamp_column")
    dataset_name = meta.get("dataset_name", "dataset")
    bucket       = Variable.get("S3_BUCKET", default_var="6gdali-lake2026")

    print(f"\n{'='*60}")
    print(f"[CLEAN] Starting cleaning for: {dataset_name}")
    print(f"[CLEAN] Recommendations: {json.dumps(recs, indent=2)}")
    print(f"{'='*60}\n")

    # ── Load data ────────────────────────────
    df, fmt = load_df_from_object_store(key=handle["key"], bucket=handle["bucket"])
    cleaning_report = {}

    # ── Step 1: Always deduplicate ───────────
    print("[CLEAN] Step 1/5: Deduplication")
    df, dedup_stats = _clean_duplicates(df)
    cleaning_report["deduplication"] = dedup_stats

    # ── Step 2: Structural fixes ─────────────
    if recs.get("structural_fix", False):
        print("[CLEAN] Step 2/5: Structural fixes")
        failed_cols = diagnostics.get("failed_columns", [])
        df, struct_stats = _fix_structural(df, ts_col, failed_cols)
        cleaning_report["structural_fix"] = struct_stats
    else:
        print("[CLEAN] Step 2/5: Structural fixes — skipped (no issues detected)")
        cleaning_report["structural_fix"] = {"skipped": True}

    # ── Step 3: Outlier handling ─────────────
    if recs.get("outlier_handling", False):
        print("[CLEAN] Step 3/5: Outlier handling")
        outlier_cols = (
            diagnostics.get("issues", {}).get("outliers", [])
            or diagnostics.get("outlier_columns", [])
        )
        df, outlier_stats = _handle_outliers(df, outlier_cols, ts_col)
        cleaning_report["outlier_handling"] = outlier_stats
    else:
        print("[CLEAN] Step 3/5: Outlier handling — skipped (no outliers detected)")
        cleaning_report["outlier_handling"] = {"skipped": True}

    # ── Step 4/5: Imputation ─────────────────
    if recs.get("ts_imputation", False):
        imputer = Variable.get("N2N_IMPUTER", default_var="darts_linear")
        target_cols = meta.get("target_cols", [])
        ts_gaps = diagnostics.get("issues", {}).get("ts_gaps", {}) or {}
        print(f"[CLEAN] Step 4: Time-series imputation "
              f"(imputer={imputer}, has_gaps={ts_gaps.get('has_gaps')}, "
              f"num_gaps={ts_gaps.get('num_gaps')})")

        if imputer == "wavestitchplus":
            # WaveStitchPlus does its own preprocess_csv internally inside the
            # GPU container — we only hand it the raw interim CSV.
            interim_key = f"interim/{dataset_name}/{run_id}/pre_impute.csv"
            save_df_to_object_store(df, key=interim_key, bucket=bucket, fmt="csv")
            imputed_handle = _ts_imputation_via_docker(
                handle={"bucket": bucket, "key": interim_key, "format": "csv"},
                meta=meta,
                run_id=run_id,
            )
            df, _ = load_df_from_object_store(
                key=imputed_handle["key"], bucket=imputed_handle["bucket"]
            )
            cleaning_report["ts_imputation"] = {
                "method": "WaveStitch+ (diffusion EM)",
                "output_key": imputed_handle["key"],
            }
            cleaning_report["prepared_artifacts"] = {"applied": False}
        else:
            # ── 4a: SAME preprocess_csv args as WaveStitchPlus ───────
            # The DAG runs preprocess_csv with the identical settings the
            # WaveStitchPlus app uses (extract_main_segment, holdout_frac=0.15,
            # split_ratio=0.8, etc.). Any baseline imputer driven from this
            # task therefore sees the exact same train/test/holdout cells
            # that WaveStitchPlus would see — comparisons stay fair.
            try:
                (train_df, test_input_df, test_gt_df, holdout_mask,
                 prep_meta, prepared_dir) = _prepare_via_preprocess_csv(
                    df, ts_col=ts_col, dataset_name=dataset_name,
                )
            except Exception as e:
                print(f"[CLEAN] Step 4a: preprocess_csv FAILED ({e}). "
                      f"Falling back to imputing on raw irregular timeline.")
                cleaning_report["prepared_artifacts"] = {
                    "applied": False, "error": str(e),
                }
                df, impute_stats = _ts_imputation_baseline(
                    df, ts_col=ts_col, target_cols=target_cols, method=imputer,
                )
                cleaning_report["ts_imputation"] = impute_stats
            else:
                target_cols = prep_meta.get("target_cols", target_cols)
                # Time-order: train rows come first, then test rows.
                # We impute the *concatenation* of train + test_input. The
                # imputer fills both the regularize-induced NaNs (in train
                # AND test) and the holdout-hidden NaNs (in test_input only).
                full_input = pd.concat(
                    [train_df, test_input_df], ignore_index=True
                )
                nan_before = int(full_input[target_cols].isna().sum().sum())
                print(f"[CLEAN]   train={len(train_df)}  "
                      f"test={len(test_input_df)}  "
                      f"holdout_rows={int(holdout_mask.sum())}  "
                      f"NaN cells to fill={nan_before:,}")

                # ── 4b: Run the in-process baseline imputer ──────────
                full_imputed, impute_stats = _ts_imputation_baseline(
                    full_input, ts_col=ts_col,
                    target_cols=target_cols, method=imputer,
                )
                # Forward/back-fill categorical-encoded cond columns (e.g.
                # ram_limit_cat) — they aren't in target_cols so the imputer
                # ignored them, but we don't want NaN leaking into curated.
                for c in (prep_meta.get("categorical_encoded_cols") or []):
                    if c in full_imputed.columns:
                        full_imputed[c] = full_imputed[c].ffill().bfill()

                # Split back so we can persist per-method train_imputed.csv +
                # test_imputed.csv next to the prepared/ dir. The dashboard
                # picks up both filenames; test_imputed is the one we score
                # against test_gt.csv at eval_holdout_mask positions.
                n_train = len(train_df)
                train_imputed_df = full_imputed.iloc[:n_train].reset_index(drop=True)
                test_imputed_df = full_imputed.iloc[n_train:].reset_index(drop=True)

                # ── 4c: Persist the comparable artifacts to S3 ───────
                # These are the SAME files WaveStitchPlus's prepared/<subset>/
                # folder produces, so the dashboard's Time series / Metrics
                # tabs can score this run apples-to-apples against any other
                # method that lands its own test_imputed.csv in the same dir.
                prep_prefix = f"cleaned/{dataset_name}/{run_id}/prepared"
                save_df_to_object_store(train_df,
                    key=f"{prep_prefix}/train.csv", bucket=bucket, fmt="csv")
                save_df_to_object_store(test_input_df,
                    key=f"{prep_prefix}/test_input.csv", bucket=bucket, fmt="csv")
                save_df_to_object_store(test_gt_df,
                    key=f"{prep_prefix}/test_gt.csv", bucket=bucket, fmt="csv")
                save_df_to_object_store(train_imputed_df,
                    key=f"{prep_prefix}/{imputer}_train_imputed.csv",
                    bucket=bucket, fmt="csv")
                save_df_to_object_store(test_imputed_df,
                    key=f"{prep_prefix}/{imputer}_test_imputed.csv",
                    bucket=bucket, fmt="csv")
                # eval_holdout_mask.npy — boolean numpy array, save via bytes
                from io import BytesIO
                import numpy as np
                buf = BytesIO()
                np.save(buf, holdout_mask.astype(np.bool_))
                _s3_upload_bytes(buf.getvalue(),
                    bucket=bucket, key=f"{prep_prefix}/eval_holdout_mask.npy")
                _s3_upload_string(json.dumps(prep_meta, indent=2, default=str),
                    bucket=bucket, key=f"{prep_prefix}/meta.json")

                cleaning_report["ts_imputation"] = impute_stats
                cleaning_report["prepared_artifacts"] = {
                    "applied": True,
                    "s3_prefix": prep_prefix,
                    "train_rows": len(train_df),
                    "test_rows": len(test_input_df),
                    "holdout_rows": int(holdout_mask.sum()),
                    "preprocess_args": _PREPROCESS_ARGS_MATCHING_WAVESTITCHPLUS,
                }

                # The curated CSV (downstream) gets the full imputed timeline.
                df = full_imputed.sort_values(ts_col).reset_index(drop=True) \
                     if ts_col in full_imputed.columns else full_imputed

                # Cleanup tmp prepared dir (we've already persisted to S3)
                import shutil
                shutil.rmtree(prepared_dir.parent, ignore_errors=True)

    elif recs.get("tabular_imputation", False):
        print("[CLEAN] Step 4/5: Tabular imputation (CPU / median+mode)")
        missing_info = (
            diagnostics.get("issues", {}).get("missing", {})
            or {c: {"dtype": "numeric"}
                for c in diagnostics.get("missing_columns", [])}
        )
        df, impute_stats = _tabular_imputation(df, missing_info, ts_col)
        cleaning_report["tabular_imputation"] = impute_stats

    else:
        print("[CLEAN] Step 4/5: Imputation — skipped (no missing values detected)")
        cleaning_report["imputation"] = {"skipped": True}

    # ── Step 5/5: Post-imputation outlier removal ──
    # Re-run after imputation because:
    # 1. Diffusion model may synthesize values outside the observed distribution
    # 2. All target metrics (latency, CPU, RAM, etc.) are physically non-negative
    print("[CLEAN] Step 5/5: Post-imputation outlier removal")

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c != ts_col
    ]

    if numeric_cols:
        # ── Enforce non-negativity first ─────────
        negative_stats = {}
        for col in numeric_cols:
            n_negative = int((df[col] < 0).sum())
            if n_negative > 0:
                df[col] = df[col].clip(lower=0)
                negative_stats[col] = {"clipped_to_zero": n_negative}
                print(f"  [NON-NEG] '{col}': clipped {n_negative} negative values to 0")

        cleaning_report["non_negativity_enforcement"] = (
            negative_stats if negative_stats else {"skipped": "no negative values found"}
        )

        # ── Then clip upper outliers (99th percentile) ──
        df, post_outlier_stats = _handle_outliers(df, numeric_cols, ts_col)
        cleaning_report["post_imputation_outlier_removal"] = post_outlier_stats

    else:
        print("  [POST_OUTLIER] No numeric columns — skipped")
        cleaning_report["post_imputation_outlier_removal"] = {"skipped": True}
        cleaning_report["non_negativity_enforcement"]      = {"skipped": True}

    # ── Save final cleaned data ───────────────
    cleaned_key = f"cleaned/{dataset_name}/{run_id}/cleaned.csv"
    save_df_to_object_store(df, key=cleaned_key, bucket=bucket, fmt="csv")

    # ── Save cleaning report sidecar ─────────
    report_key = f"cleaned/{dataset_name}/{run_id}/cleaning_report.json"
    _s3_upload_string(                              # ← 修复：传入正确参数
        content=json.dumps(cleaning_report, indent=2),
        bucket=bucket,
        key=report_key,
    )

    print(f"\n{'='*60}")
    print(f"[CLEAN] Completed. Output:  s3://{bucket}/{cleaned_key}")
    print(f"[CLEAN] Report:             s3://{bucket}/{report_key}")
    print(f"[CLEAN] Cleaning report: {json.dumps(cleaning_report, indent=2)}")
    print(f"{'='*60}\n")

    ti.xcom_push(
        key="cleaned_handle",
        value={"bucket": bucket, "key": cleaned_key, "format": "csv"},
    )
    ti.xcom_push(key="cleaning_report", value=cleaning_report)
    return cleaning_report
