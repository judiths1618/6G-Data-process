from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta


import json
import pandas as pd
from io import BytesIO
from airflow.models import Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


from helpers.object_store import load_df_from_object_store
from helpers.utils import analyze_csv_time_series_df, detect_primary_key

def load_raw_data(**context):
    ti = context['ti']

    input_key = Variable.get("N2N_INPUT_KEY", default_var = "test/amfperformance.csv")
    bucket = Variable.get("S3_BUCKET", default_var="airflow-bucket")

    # --- ADD THIS PRINT BLOCK ---
    # print(f"DEBUG INFO:")
    # print(f"   Target Bucket: {bucket}")
    # print(f"   Target Key:    {input_key}")
    # print(f"   Full URI:      s3://{bucket}/{input_key}")
    # # ----------------------------
    # print("[DEBUG] input_key repr:", repr(input_key))
    # print("[DEBUG] bucket repr:", repr(bucket))

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

    info = {
        "source": {
            "bucket": bucket,
            "key": input_key,
            "format": fmt,
        },
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
    df_ts = df[[ts_col]].copy()
    df_ts[ts_col] = pd.to_datetime(df_ts[ts_col], unit="s", errors="coerce")

    ts_diag = detect_time_gaps(df_ts, ts_col)

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
            "start_date": str(df_ts[ts_col].min()),
            "end_date": str(df_ts[ts_col].max())
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

    ts_diag = qc_result.get("time_series_diagnostics", {})
    if ts_diag.get("has_gaps"):
        if status != "FAIL": status = "WARN"
        issues.append(f"Time-series gaps detected ({ts_diag.get('total_gaps')} gaps).")

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

def validate_cleaned():
    # GX validation on cleaned dataset
    pass

def store_curated():
    # Write cleaned dataset back to SeaweedFS
    pass

def visualize():
    # Generate plots, save to SeaweedFS
    pass

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
    tags=["GX", "SeaweedFS", "GPU", "Diffusion Models"],
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

    # 🔥 AI-based methods for cleaning and augmentation (GPU isolated, correct)
    clean = DockerOperator(
        task_id="clean_dirty_data_gpu",
        image="diffusion-models-gpu-cleaning-image:latest",
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        command="python clean.py",
        environment={
            "INPUT_S3": "s3://raw-data/...",
            "OUTPUT_S3": "s3://clean-data/...",
            "AWS_ENDPOINT_URL": "http://seaweed-s3:8333",
        },
        device_requests=[{
            "Driver": "nvidia",
            "Count": 1,
            "Capabilities": [["gpu"]],
        }],
    )

    validate = PythonOperator(
        task_id="validate_cleaned_data",
        python_callable=validate_cleaned,
    )

    store = PythonOperator(
        task_id="store_curated_data",
        python_callable=store_curated,
    )

    viz = PythonOperator(
        task_id="visualization",
        python_callable=visualize,
    )

    # -------- Dependencies (mirrors your diagram) --------

    load >> branch
    branch >> qc_task >> report
    branch >> ts_qc_task >> report
    report >> clean >> validate >> store >> viz
