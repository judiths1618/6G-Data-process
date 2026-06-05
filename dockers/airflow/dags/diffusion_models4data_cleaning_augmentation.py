from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path


import json
import pandas as pd
import re


from helpers.object_store import load_df_from_object_store, save_df_to_object_store
from helpers.utils import analyze_csv_time_series_df, detect_primary_key
from helpers.clean_dirty_data import clean_dirty_data, _get_s3_client, _s3_upload_string, _s3_upload_bytes

def make_dataset_name_from_key(key: str) -> str:
    stem = Path(key).name  # e.g., "amfperformance.csv"
    stem = stem.rsplit(".", 1)[0]  # "amfperformance"
    stem = stem.strip()

    # 变成安全的 id：a-zA-Z0-9_-，其他都替换成 _
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", stem)
    safe = safe.strip("_")
    return safe or "dataset"


def load_raw_data(**context):
    ti = context['ti']

    input_key = Variable.get("N2N_INPUT_KEY", default_var = "test/amf-performance.csv")
    bucket = Variable.get("S3_BUCKET", default_var="6gdali-lake2026")
    print(f"[LOAD_RAW_DATA] Reading s3://{bucket}/{input_key}")

    df, fmt = load_df_from_object_store(
        key=input_key,
        bucket=bucket,
    )
    pk_info = detect_primary_key(df)
    

    # ---------- TS detection ----------
    TIMESTAMP_COL = Variable.get("N2N_TIMESTAMP_COL", default_var="time")

    ts_analysis = analyze_csv_time_series_df(
        df,
        configured_name=TIMESTAMP_COL,
    )
    is_ts = ts_analysis["is_time_series"]
    ts_col = ts_analysis["timestamp_column"]
    
    dataset_name = Variable.get(
        "N2N_DATASET_NAME",
        default_var=make_dataset_name_from_key(input_key),
    )
    target_cols = [c for c in df.columns if c != ts_col]  # ts_col 后面才算出来，所以放到 ts_analysis 后


    info = {
        "source": {
            "bucket": bucket,
            "key": input_key,
            "format": fmt,
        },
        "dataset_name": dataset_name,
        "target_cols": target_cols,

        "shape": {
            "rows": len(df),
            "cols": df.shape[1],
        },
        "columns": list(df.columns),
        "is_time_series": is_ts,
        "timestamp_column": ts_col,
        "time_series_analysis": ts_analysis,
        "preview": df.head(5).to_dict(orient="records"),
    }

    print("[LOAD_RAW_DATA]")
    print(json.dumps(info, indent=2, ensure_ascii=False))

    # ---------- XCom contract ----------
    ti.xcom_push(key="dataset_meta", value=info)
    ti.xcom_push(
        key="raw_handle",
        value={
            "bucket": bucket,
            "key": input_key,
            "format": fmt,
        },
    )
    ti.xcom_push(key="pk_info", value=pk_info)
    return info

def is_time_series(**context):
    ti = context["ti"]
    meta = ti.xcom_pull(
        task_ids="load_raw_data",
        key="dataset_meta",
    )
    if not meta:
        raise ValueError(f"Task 'load_raw_data' returned no data. Check upstream logs.")

    if meta.get("is_time_series"):
        return "ts_qc"
    return "qc"


def qc(**context):
    import pandas as pd
    from helpers.object_store import load_df_from_object_store
    from helpers.gx_utils import get_gx_context

    ti = context["ti"]

    meta = ti.xcom_pull(task_ids="load_raw_data", key="dataset_meta")
    handle = ti.xcom_pull(task_ids="load_raw_data", key="raw_handle")
    pk_info = ti.xcom_pull(task_ids="load_raw_data", key="pk_info") or {}

    df, _ = load_df_from_object_store(
        key=handle["key"],
        bucket=handle["bucket"],
    )

    gx_context = get_gx_context()

    # ----------------------------
    # GX datasource & validator
    # ----------------------------
    ds_name = "pandas_tabular"
    try:
        ds = gx_context.sources.add_pandas(name=ds_name)
    except Exception:
        ds = gx_context.sources.get(ds_name)

    asset = ds.add_dataframe_asset(
        name=f"raw_tabular_{ti.run_id}",
        dataframe=df,
    )
    batch_request = asset.build_batch_request()

    suite_name = "tabular_quality"
    suite = gx_context.add_or_update_expectation_suite(suite_name)

    # ----------------------------
    # Column classification
    # ----------------------------
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    pk_cols = pk_info.get("columns", [])

    # ----------------------------
    # Expectations
    # ----------------------------

    # Missingness (soft, not fatal)
    missing_cols = []

    for col in df.columns:
        if col in pk_cols:
            continue

        missing_ratio = df[col].isna().mean()

        suite.expect_column_values_to_not_be_null(
            column=col,
            mostly=0.95 if col in numeric_cols else 0.90,
        )

        if missing_ratio > 0.05:
            missing_cols.append(col)

    # Numeric sanity (skip constants / ids)
    outlier_cols = []

    for col in numeric_cols:
        if col in pk_cols:
            continue

        if df[col].nunique() < 10:
            continue  # likely categorical id

        q_low = df[col].quantile(0.01)
        q_high = df[col].quantile(0.99)

        if q_low == q_high:
            continue

        suite.expect_column_values_to_be_between(
            column=col,
            min_value=q_low,
            max_value=q_high,
            mostly=0.98,
        )

        outlier_cols.append(col)

    # ----------------------------
    # Run validation
    # ----------------------------
    validator = gx_context.get_validator(
        batch_request=batch_request,
        expectation_suite=suite,
    )

    gx_result = validator.validate()

    # ----------------------------
    # Interpret results (this is the key part)
    # ----------------------------
    failed_cols = set(
        r["expectation_config"]["kwargs"].get("column")
        for r in gx_result["results"]
        if not r["success"]
    )

    recommendations = []

    if missing_cols:
        recommendations.append(
            f"Missing values detected in columns: {missing_cols}. "
            "Recommend tabular imputation."
        )

    if outlier_cols:
        recommendations.append(
            f"Potential outliers detected in numeric columns: {outlier_cols}."
        )

    if pk_info.get("type") == "none":
        recommendations.append(
            "No primary key detected. Treat as fact table; avoid row-wise imputation."
        )

    # ----------------------------
    # Final QC result (router-friendly)
    # ----------------------------
    qc_result = {
        "mode": "tabular",
        "gx_passed": bool(gx_result["success"]),
        "missing_columns": missing_cols,
        "outlier_columns": outlier_cols,
        "failed_columns": list(failed_cols),
        "primary_key": pk_info,
        "recommendations": recommendations,
        "summary": {
            "rows": len(df),
            "columns": df.columns.tolist(),
        },
    }

    ti.xcom_push(key="qc_result", value=qc_result)
    return qc_result

def ts_qc(**context):
    import pandas as pd
    from helpers.object_store import load_df_from_object_store
    from helpers.gx_utils import get_gx_context
    from helpers.ts_utils import detect_time_gaps

    ti = context["ti"]
    meta = ti.xcom_pull(task_ids="load_raw_data", key="dataset_meta")
    handle = ti.xcom_pull(task_ids="load_raw_data", key="raw_handle")

    # --- Load data ---
    s3_key = handle["key"].strip()
    df, _ = load_df_from_object_store(key=s3_key, bucket=handle["bucket"])

    ts_col = meta["timestamp_column"]

    # =========================
    # 1. GX Structural Checks
    # =========================
    gx_context = get_gx_context()
    datasource_name = "pandas_s3_source"

    try:
        ds = gx_context.sources.add_pandas(name=datasource_name)
    except Exception:
        ds = gx_context.get_datasource(datasource_name)

    asset = ds.add_dataframe_asset(name=f"asset_{ti.run_id}", dataframe=df)
    batch_request = asset.build_batch_request()

    suite = gx_context.add_or_update_expectation_suite(
        expectation_suite_name="ts_quality"
    )
    # --- Prepare GX-safe DataFrame ---
    df_gx = df.copy()

    df_gx[ts_col] = pd.to_datetime(
        df_gx[ts_col],
        errors="coerce",
        infer_datetime_format=True,
    )
    validator = gx_context.get_validator(
        batch_request=batch_request,
        expectation_suite=suite
    )

    # --- Timestamp structural integrity ---
    validator.expect_column_values_to_not_be_null(ts_col)
    validator.expect_column_values_to_be_unique(ts_col)
    validator.expect_column_values_to_be_increasing(ts_col)

    # --- General missingness (column-level, type-agnostic) ---
    for col in df.columns:
        validator.expect_column_values_to_not_be_null(column=col, mostly=0.98)

    gx_result = validator.validate()

    # =========================
    # 2. Type-aware Missing Detection
    # =========================
    def normalize_dtype(series: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        return "categorical"

    missing_info = {}
    for col in df.columns:
        miss_ratio = df[col].isna().mean()
        if miss_ratio > 0:
            missing_info[col] = {
                "dtype": normalize_dtype(df[col]),
                "missing_ratio": round(float(miss_ratio), 4)
            }

    # =====================================
    # 3. Outlier Detection (Numeric only)
    # =====================================
    outlier_cols = []

    numeric_cols = [
        c for c in df.select_dtypes(include=["number"]).columns
        if c != ts_col
    ]

    for col in numeric_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)

        validator.expect_column_values_to_be_between(
            column=col,
            min_value=lower,
            max_value=upper,
            mostly=0.95
        )

        if (df[col] < lower).any() or (df[col] > upper).any():
            outlier_cols.append(col)

    # ===============================
    # 4. Time-series Gap Detection
    # ===============================
    # ``detect_time_gaps`` is dtype-aware: numeric epoch seconds / ms /
    # datetime64 / parseable string columns are all handled. We pass the
    # original column so the detector can pick the right unit; no need to
    # force a conversion here.
    ts_diag = detect_time_gaps(df, ts_col)
    print(f"[TS_QC] gap detection: has_gaps={ts_diag.get('has_gaps')} "
          f"num_gaps={ts_diag.get('num_gaps')} "
          f"expected_dt={ts_diag.get('expected_dt_seconds')}s "
          f"missing_rows~={ts_diag.get('total_missing_rows')} "
          f"gap_pct={100 * (ts_diag.get('gap_pct') or 0):.1f}%")

    # ===============================
    # 5. Recommendations (Decoupled!)
    # ===============================
    recommendations = {
        "ts_imputation": False,
        "tabular_imputation": False,
        "outlier_handling": False,
        "structural_fix": False
    }

    if ts_diag.get("has_gaps"):
        recommendations["ts_imputation"] = True

    if missing_info:
        recommendations["tabular_imputation"] = True

    if outlier_cols:
        recommendations["outlier_handling"] = True

    if not gx_result["success"]:
        recommendations["structural_fix"] = True

    # =========================
    # 6. Final TS QC Result
    # =========================
    ts_result = {
        "mode": "time_series",
        "gx_passed": bool(gx_result["success"]),
        "issues": {
            "ts_gaps": ts_diag,
            "missing": missing_info,
            "outliers": list(set(outlier_cols)),
        },
        "recommendations": recommendations,
        "summary": {
            "total_records": len(df),
            "start_date": str(df[ts_col].min()),
            "end_date": str(df[ts_col].max()),
        }
    }

    ti.xcom_push(key="qc_result", value=ts_result)
    return ts_result

def report_dqc(ti, **context):
    import datetime
    import json

    dataset = ti.xcom_pull(task_ids="load_raw_data", key="dataset_meta")
    
    # 拉取可能的 QC 结果
    qc_pull = ti.xcom_pull(task_ids=["qc", "ts_qc"], key="qc_result")
    qc_result = next((r for r in qc_pull if r is not None), {})

    # 状态逻辑
    status = "PASS"
    issues = []
    recs = qc_result.get("recommendations", [])

    if not qc_result.get("gx_passed", True):
        status = "FAIL"
        issues.append("Data quality expectations failed (GX).")

    ts_diag = qc_result.get("issues", {}).get("ts_gaps", {})
    if ts_diag.get("has_gaps"):
        if status != "FAIL":
            status = "WARN"
        issues.append(
            f"Time-series gaps detected: {ts_diag.get('num_gaps')} gaps, "
            f"expected dt={ts_diag.get('expected_dt_seconds')}s, "
            f"~{ts_diag.get('total_missing_rows')} missing rows "
            f"({100 * (ts_diag.get('gap_pct') or 0):.1f}% of grid)."
        )

    report = {
        "metadata": {
            "dag_id": context["dag"].dag_id,
            "run_id": context["run_id"],
            "timestamp": datetime.datetime.utcnow().isoformat(),
        },
        "dataset": {
            "source": dataset.get("source", {}).get("key", "").strip(),
            "rows": dataset.get("shape", {}).get("rows"),
        },
        "quality_report": {
            "status": status,
            "gx_passed": qc_result.get("gx_passed"),
            "issues": issues,
            "recommendations": recs  # 建议现在在这里
        },
        "full_diagnostics": qc_result
    }

    ti.xcom_push(key="dqc_report", value=report)
    print(json.dumps(report, indent=2))
    return report

# def clean_dirty_data():
    # =================== Apply cleaning actions ===================
    # 1) Basic cleaning actions for drop_duplicates, impute_missing, clip_outliers_iqr
        
# def validate_cleaned():
#     # GX validation on cleaned dataset
#     pass

# def store_curated():
#     # Write cleaned dataset back to SeaweedFS
#     pass

# def visualize():
#     # Generate plots, save to SeaweedFS
#     pass
def validate_cleaned(**context):
    """
    Re-run GX expectations on cleaned dataset to confirm issues are resolved.
    """
    from helpers.gx_utils import get_gx_context

    ti     = context["ti"]
    meta   = ti.xcom_pull(task_ids="load_raw_data", key="dataset_meta")
    report = ti.xcom_pull(task_ids="report_dqc",    key="dqc_report")
    ts_col = meta.get("timestamp_column")

    # Pull cleaned handle from clean_dirty_data
    cleaned_handle = ti.xcom_pull(task_ids="clean_dirty_data", key="cleaned_handle")
    if not cleaned_handle:
        raise ValueError("[VALIDATE] No cleaned_handle found. Check clean_dirty_data task.")

    df, _ = load_df_from_object_store(
        key=cleaned_handle["key"], bucket=cleaned_handle["bucket"]
    )

    print(f"\n{'='*60}")
    print(f"[VALIDATE] Running post-clean GX validation")
    print(f"[VALIDATE] Dataset shape: {df.shape}")
    print(f"{'='*60}\n")

    # ── GX setup ─────────────────────────────
    gx_context = get_gx_context()
    try:
        ds = gx_context.sources.add_pandas(name="pandas_post_clean")
    except Exception:
        ds = gx_context.sources.get("pandas_post_clean")

    asset = ds.add_dataframe_asset(
        name=f"cleaned_{ti.run_id}", dataframe=df
    )
    batch_request = asset.build_batch_request()
    suite = gx_context.add_or_update_expectation_suite("post_clean_quality")
    validator = gx_context.get_validator(
        batch_request=batch_request, expectation_suite=suite
    )

    # ── Expectations ─────────────────────────
    # 1. No nulls (relaxed threshold post-cleaning)
    for col in df.columns:
        validator.expect_column_values_to_not_be_null(column=col, mostly=0.99)

    # 2. Timestamp integrity (if TS data)
    if ts_col and ts_col in df.columns:
        validator.expect_column_values_to_not_be_null(ts_col)
        validator.expect_column_values_to_be_unique(ts_col)
        validator.expect_column_values_to_be_increasing(ts_col)

    # 3. Numeric range sanity (no extreme outliers post-clipping)
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c != ts_col
    ]
    for col in numeric_cols:
        q_low  = df[col].quantile(0.001)
        q_high = df[col].quantile(0.999)
        if q_low < q_high:
            validator.expect_column_values_to_be_between(
                column=col, min_value=q_low, max_value=q_high, mostly=0.999
            )

    # ── Run ──────────────────────────────────
    gx_result = validator.validate()

    failed = [
        r["expectation_config"]["kwargs"].get("column")
        for r in gx_result["results"]
        if not r["success"]
    ]

    validation_result = {
        "passed":       bool(gx_result["success"]),
        "failed_cols":  failed,
        "statistics":   gx_result.get("statistics", {}),
        "cleaned_key":  cleaned_handle["key"],
    }

    if not gx_result["success"]:
        print(f"[VALIDATE] ⚠ Some expectations still failing: {failed}")
    else:
        print(f"[VALIDATE] ✓ All expectations passed.")

    ti.xcom_push(key="validation_result",      value=validation_result)
    ti.xcom_push(key="final_cleaned_handle",   value=cleaned_handle)
    return validation_result


def store_curated(**context):
    """
    Write final cleaned dataset to curated S3 prefix with JSON metadata sidecar.
    """
    import datetime

    ti     = context["ti"]
    meta   = ti.xcom_pull(task_ids="load_raw_data",        key="dataset_meta")
    report = ti.xcom_pull(task_ids="report_dqc",           key="dqc_report")
    val    = ti.xcom_pull(task_ids="validate_cleaned_data", key="validation_result") or {}
    cleaning_report = ti.xcom_pull(task_ids="clean_dirty_data", key="cleaning_report") or {}

    final_handle = ti.xcom_pull(
        task_ids="validate_cleaned_data", key="final_cleaned_handle"
    )
    if not final_handle:
        raise ValueError("[STORE] No final_cleaned_handle found. Check validate_cleaned_data task.")

    df, _ = load_df_from_object_store(
        key=final_handle["key"], bucket=final_handle["bucket"]
    )

    dataset_name = meta.get("dataset_name", "dataset")
    run_id       = context["run_id"]
    bucket       = Variable.get("S3_BUCKET", default_var="6gdali-lake2026")

    curated_key  = f"curated/{dataset_name}/{run_id}/data.csv"
    sidecar_key  = f"curated/{dataset_name}/{run_id}/meta.json"

    print(f"\n{'='*60}")
    print(f"[STORE] Saving curated data to s3://{bucket}/{curated_key}")
    print(f"{'='*60}\n")

    # ── Save curated CSV ──────────────────────
    save_df_to_object_store(df, key=curated_key, bucket=bucket, fmt="csv")

    # ── Save metadata sidecar ─────────────────
    sidecar = {
        "dataset_name":      dataset_name,
        "run_id":            run_id,
        "stored_at":         datetime.datetime.utcnow().isoformat(),
        "rows":              len(df),
        "columns":           list(df.columns),
        "source_key":        meta.get("source", {}).get("key", ""),
        "quality_status":    report.get("quality_report", {}).get("status"),
        "validation_passed": val.get("passed"),
        "cleaning_summary":  cleaning_report,
        "curated_key":       curated_key,
    }

    _s3_upload_string(json.dumps(sidecar, indent=2), bucket, sidecar_key)


    print(f"[STORE] ✓ Curated data:     s3://{bucket}/{curated_key}")
    print(f"[STORE] ✓ Metadata sidecar: s3://{bucket}/{sidecar_key}")
    print(json.dumps(sidecar, indent=2))

    ti.xcom_push(key="curated_handle", value={"bucket": bucket, "key": curated_key})
    ti.xcom_push(key="sidecar",        value=sidecar)
    return sidecar


def visualize(**context):
    """
    Generate and save diagnostic plots to S3:
    1. Missing value heatmap
    2. Numeric distributions
    3. Time-series line plots (if TS data)
    4. Before/after cleaning comparison (null rate per column)
    """
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    ti             = context["ti"]
    meta           = ti.xcom_pull(task_ids="load_raw_data",   key="dataset_meta")
    cleaning_report = ti.xcom_pull(task_ids="clean_dirty_data", key="cleaning_report") or {}
    curated_handle = ti.xcom_pull(task_ids="store_curated_data", key="curated_handle")
    raw_handle     = ti.xcom_pull(task_ids="load_raw_data",   key="raw_handle")

    if not curated_handle:
        raise ValueError("[VIZ] No curated_handle found. Check store_curated_data task.")

    ts_col       = meta.get("timestamp_column")
    dataset_name = meta.get("dataset_name", "dataset")
    run_id       = context["run_id"]
    bucket       = Variable.get("S3_BUCKET", default_var="6gdali-lake2026")

    plot_keys = []

    # Load curated (cleaned) data
    df_clean, _ = load_df_from_object_store(
        key=curated_handle["key"], bucket=curated_handle["bucket"]
    )
    # Load raw data for before/after comparison
    df_raw, _ = load_df_from_object_store(
        key=raw_handle["key"], bucket=raw_handle["bucket"]
    )

    numeric_cols = [
        c for c in df_clean.select_dtypes(include="number").columns
        if c != ts_col
    ]

    def _save_plot(fig, name: str) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        buf.seek(0)
        key = f"viz/{dataset_name}/{run_id}/{name}.png"
        # hook.load_bytes(buf.read(), key=key, bucket_name=bucket, replace=True)
        _s3_upload_bytes(buf.read(), bucket, key)
        print(f"  [VIZ] Saved: s3://{bucket}/{key}")
        return key

    print(f"\n{'='*60}")
    print(f"[VIZ] Generating plots for: {dataset_name}")
    print(f"{'='*60}\n")

    # ── Plot 1: Missing value heatmap (cleaned) ───
    if df_clean.isna().any().any():
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.imshow(
            df_clean.isna().astype(int).T,
            aspect="auto", cmap="Reds", interpolation="none"
        )
        ax.set_yticks(range(len(df_clean.columns)))
        ax.set_yticklabels(df_clean.columns, fontsize=8)
        ax.set_title(f"Missing Value Map — {dataset_name} (post-clean)", fontsize=11)
        ax.set_xlabel("Row index")
        plot_keys.append(_save_plot(fig, "missing_heatmap"))
    else:
        print("  [VIZ] No missing values in cleaned data — heatmap skipped.")

    # ── Plot 2: Before/after null rate per column ─
    common_cols = [c for c in df_raw.columns if c in df_clean.columns and c != ts_col]
    if common_cols:
        raw_null_rate   = df_raw[common_cols].isna().mean() * 100
        clean_null_rate = df_clean[common_cols].isna().mean() * 100

        x = range(len(common_cols))
        fig, ax = plt.subplots(figsize=(max(10, len(common_cols) * 0.8), 4))
        ax.bar([i - 0.2 for i in x], raw_null_rate,   width=0.4,
               label="Before cleaning", color="tomato",    alpha=0.8)
        ax.bar([i + 0.2 for i in x], clean_null_rate, width=0.4,
               label="After cleaning",  color="steelblue", alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(common_cols, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Missing %")
        ax.set_title(f"Missing Rate Before vs After Cleaning — {dataset_name}", fontsize=11)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plot_keys.append(_save_plot(fig, "before_after_missing"))

    # ── Plot 3: Numeric distributions ────────────
    if numeric_cols:
        n      = min(len(numeric_cols), 6)
        cols   = numeric_cols[:n]
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            df_clean[col].dropna().hist(
                bins=30, ax=ax, color="steelblue", edgecolor="white"
            )
            ax.set_title(col, fontsize=9)
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plot_keys.append(_save_plot(fig, "distributions"))

    # ── Plot 4: Time-series line plots ────────────
    if ts_col and ts_col in df_clean.columns and numeric_cols:
        plot_cols = numeric_cols[:4]
        df_plot   = df_clean[[ts_col] + plot_cols].copy()
        df_plot[ts_col] = pd.to_numeric(df_plot[ts_col], errors="coerce")
        df_plot   = df_plot.sort_values(ts_col)

        fig = plt.figure(figsize=(14, 3 * len(plot_cols)))
        gs  = gridspec.GridSpec(len(plot_cols), 1, hspace=0.4)

        for i, col in enumerate(plot_cols):
            ax = fig.add_subplot(gs[i])
            ax.plot(
                df_plot[ts_col], df_plot[col],
                linewidth=0.8, color="steelblue"
            )
            ax.set_ylabel(col, fontsize=9)
            ax.grid(True, alpha=0.3)
            if i < len(plot_cols) - 1:
                ax.set_xticklabels([])

        axes_list = fig.get_axes()
        if axes_list:
            axes_list[-1].set_xlabel("Timestamp")

        fig.suptitle(f"Time Series (cleaned) — {dataset_name}", fontsize=11)
        plot_keys.append(_save_plot(fig, "timeseries"))

    print(f"\n[VIZ] ✓ Generated {len(plot_keys)} plots:")
    for k in plot_keys:
        print(f"      s3://{bucket}/{k}")

    ti.xcom_push(key="plot_keys", value=plot_keys)
    return plot_keys

# ---------------- DAG ----------------
default_args = {
    "owner": "yd data-platform-pipeline",
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}


with DAG(
    dag_id="data_quality_and_cleaning_pipeline",
    description="N2N ELT: Load → Profile → Branch(QC) → Report → Clean (duplicates, missing, TS gaps, Outliers) → Store → Viz",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["6G Time Series", "GX", "SeaweedFS", "WaveStitchPlus", "GPU", "Diffusion Models"],
) as dag:

    # 1) Load + Profile
    load = PythonOperator(
        task_id="load_raw_data",
        python_callable=load_raw_data,
    )
    # 2) Branch
    branch = BranchPythonOperator(
        task_id="is_time_series",
        python_callable=is_time_series,
    )

    qc_task = PythonOperator(
        task_id="qc",
        python_callable=qc,
    )

    ts_qc_task = PythonOperator(
        task_id="ts_qc",
        python_callable=ts_qc,
    )

    report = PythonOperator(
        task_id="report_dqc",
        python_callable=report_dqc,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # 🔥 AI-based methods for cleaning (GPU isolated, correct)
    # from airflow.providers.docker.operators.docker import DockerOperator

    # 在 DAG 中使用 DockerOperator
    # clean = DockerOperator(
    #     task_id="clean_dirty_data",
    #     image="wavestitchplus-gpu:latest",
    #     api_version="auto",
    #     auto_remove=True,
    #     docker_url="unix://var/run/docker.sock",
    #     network_mode="dockers_airflow_net",

    #     command=[
    #         "python", "/app/run_pipeline.py",

    #         # 运行模式：full=train+inference
    #         "--mode", "full",

    #         # run_pipeline.py 需要 dataset-name
    #         "--dataset-name",
    #         "{{ ti.xcom_pull(task_ids='load_raw_data', key='dataset_meta')['dataset_name'] }}",

    #         # 用 run_id 做版本号，方便追踪（你的 run_pipeline.py 支持 --version）
    #         "--version", "{{ run_id }}",

    #         # 输入数据 key：从 load_raw_data 的 raw_handle 拿
    #         "--input-s3-key",
    #         "{{ ti.xcom_pull(task_ids='load_raw_data', key='raw_handle')['key'] }}",

    #         # 时间列：从 ts detection 拿
    #         "--time-col",
    #         "{{ ti.xcom_pull(task_ids='load_raw_data', key='dataset_meta')['timestamp_column'] }}",

    #         # 目标列：list -> csv string
    #         "--target-cols",
    #         "{{ ti.xcom_pull(task_ids='load_raw_data', key='dataset_meta')['target_cols'] | join(',') }}",

    #         # 训练参数
    #         "--epochs", "500",
    #         "--batch-size", "512",
    #         "--window-size", "32",

    #         # 推理参数
    #         "--n-trials", "1",
    #         "--guidance-scale", "0.1",

    #         # EM（你想用就保留）
    #         "--use-em",
    #         "--em-iterations", "5",
    #         "--epochs-per-em", "200",
    #     ],

    #     environment={
    #         "S3_ENDPOINT": "http://seaweed-s3:8333",
    #         "S3_ACCESS_KEY": "anykey",
    #         "S3_SECRET_KEY": "anysecret",
    #         "S3_BUCKET": "airflow-bucket",
    #     },

    #     device_requests=[{
    #         "Driver": "nvidia",
    #         "Count": 1,
    #         "Capabilities": [["gpu"]],
    #     }],

    #     mount_tmp_dir=False,
    #     tmp_dir="/tmp",
    #     execution_timeout=timedelta(hours=4),
    # )
    
    clean = PythonOperator(
        task_id="clean_dirty_data",
        python_callable=clean_dirty_data,
        execution_timeout=timedelta(hours=4),
    )
    validate = PythonOperator(
        task_id="validate_cleaned_data",
        python_callable=validate_cleaned,
    )

    store = PythonOperator(
        task_id="store_curated_data",
        python_callable=store_curated,
    )

    # viz = PythonOperator(
    #     task_id="visualization",
    #     python_callable=visualize,
    # )

    # -------- Dependencies (mirrors your diagram) --------

    load >> branch
    branch >> qc_task >> report
    branch >> ts_qc_task >> report
    report >> clean >> validate >> store 
    # >> viz
