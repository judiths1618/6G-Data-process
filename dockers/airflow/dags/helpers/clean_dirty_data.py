from __future__ import annotations  # ADD THIS — enables PEP 604

import pandas as pd
import numpy as np
import json
import docker
from airflow.models import Variable
from helpers.object_store import load_df_from_object_store, save_df_to_object_store


import boto3
from botocore.client import Config
from airflow.models import Variable

# helpers/clean_dirty_data.py

def _get_s3_client():
    """统一的 S3 客户端，与 run_pipeline.py 保持一致。"""
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
# Sub-module 5: TS imputation via WaveStitch+
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

    logs = client.containers.run(
        image="wavestitchplus-gpu:latest",
        command=[
            "python", "/app/run_pipeline.py",
            "--mode",         "full",
            "--dataset-name", dataset_name,
            "--version",      run_id,
            "--input-s3-key", handle["key"],
            "--time-col",     ts_col,
            "--target-cols",  ",".join(target_cols),
            "--epochs",       "500",
            "--batch-size",   "512",
            "--window-size",  "32",
            "--n-trials",     "1",
            "--guidance-scale", "0.1",
            "--use-em",
            "--em-iterations",  "5",
            "--epochs-per-em",  "200",
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
        print("[CLEAN] Step 4/5: Time-series imputation (GPU / WaveStitch+)")
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