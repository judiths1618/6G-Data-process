"""
WaveStitch Airflow DAG - 保持原有 pipeline 结构
clean -> validate -> store -> visualization
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta

import json
import os
import shutil
import subprocess
import glob
import pandas as pd
from pathlib import Path
from airflow.models import Variable

from helpers.object_store import (
    load_df_from_object_store,
    save_df_to_object_store,
    upload_directory_to_s3,
    create_wavestitch_manifest,
    save_json,
    S3_BUCKET,
    _prepare_data_manually
)
from helpers.utils import analyze_csv_time_series_df, detect_primary_key

import os

# 获取当前 DAG 文件所在目录
DAG_DIR = os.path.dirname(os.path.abspath(__file__))
WAVESTITCH_DIR = os.path.join(DAG_DIR, "WaveStitchPlus_app")  # 你的文件夹名


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

# ============ Clean Task（WaveStitch 整合）============

def clean_dirty_data(**context):
    """清洗任务"""
    ti = context['ti']
    run_id = context['run_id']
    dag_id = context['dag'].dag_id
    
    meta = ti.xcom_pull(task_ids="load_raw_data", key="dataset_meta")
    handle = ti.xcom_pull(task_ids="load_raw_data", key="raw_handle")
    
    qc_result = ti.xcom_pull(task_ids="ts_qc", key="qc_result")
    if not qc_result:
        qc_result = ti.xcom_pull(task_ids="qc", key="qc_result")
    qc_result = qc_result or {}
    
    # 工作目录
    work_dir = os.path.join("/tmp/wavestitch", dag_id, run_id)
    prepared_dir = os.path.join(work_dir, "prepared")
    generated_dir = os.path.join(work_dir, "generated")
    model_dir = os.path.join(work_dir, "models")
    logs_dir = os.path.join(work_dir, "logs")
    
    for d in [work_dir, prepared_dir, generated_dir, model_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 下载数据
    df, fmt = load_df_from_object_store(
        key=handle["key"],
        bucket=handle["bucket"],
    )
    
    input_csv = os.path.join(work_dir, "input.csv")
    df.to_csv(input_csv, index=False)
    
    original_shape = df.shape
    original_missing = df.isna().sum().sum()
    
    print(f"[CLEAN] Input: {original_shape}, Missing: {original_missing}")
    
    # 配置
    mode = qc_result.get("mode", "tabular")
    use_wavestitch = (mode == "time_series")
    
    ts_col = meta.get("timestamp_column", "time")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    target_cols = [c for c in numeric_cols if c != ts_col]
    
    config = {
        "run_id": run_id,
        "dag_id": dag_id,
        "work_dir": work_dir,
        "prepared_dir": prepared_dir,
        "generated_dir": generated_dir,
        "model_dir": model_dir,
        "logs_dir": logs_dir,
        "input_csv": input_csv,
        "source": handle,
        "original_shape": list(original_shape),
        "original_missing": int(original_missing),
        "time_col": ts_col,
        "target_cols": target_cols,
    }
    
    if use_wavestitch:
        print(f"[CLEAN] Using WaveStitch")
        config["method"] = "wavestitch"
        
        # Step 1: Preprocess
        print(f"[CLEAN] Step 1/3: Preprocessing...")
        preprocess_result = _run_wavestitch_preprocess(config, logs_dir)
        
        # 🔥 检查并手动创建
        meta_path = os.path.join(prepared_dir, "meta.json")
        if not os.path.exists(meta_path):
            print(f"[CLEAN] Preprocess didn't create meta.json, creating manually...")
            _prepare_data_manually(config, df, ts_col, target_cols)
        
        # 验证
        if not os.path.exists(meta_path):
            raise RuntimeError(f"Failed to create {meta_path}")
        print(f"[CLEAN] ✓ meta.json ready")
        
        # Step 2: Training
        print(f"[CLEAN] Step 2/3: Training...")
        train_result = _run_wavestitch_training(config, logs_dir)
        
        if train_result["status"] != "success":
            raise RuntimeError(f"Training failed: {train_result.get('error')}")
        
        # Step 3: Synthesis
        print(f"[CLEAN] Step 3/3: Synthesis...")
        synthesis_result = _run_wavestitch_synthesis(config, logs_dir)
        
        if synthesis_result["status"] != "success":
            raise RuntimeError(f"Synthesis failed: {synthesis_result.get('error')}")
        
        output_csv = synthesis_result.get("output_csv")
        
        if output_csv and os.path.exists(output_csv):
            df_cleaned = pd.read_csv(output_csv)
            final_csv = os.path.join(generated_dir, "final_imputed.csv")
            df_cleaned.to_csv(final_csv, index=False)
            config["output_csv"] = final_csv
            config["cleaned_shape"] = list(df_cleaned.shape)
            config["cleaned_missing"] = int(df_cleaned.isna().sum().sum())
        else:
            # Fallback
            print(f"[CLEAN] No synthesis output, using interpolation")
            df_cleaned = df.copy()
            for col in target_cols:
                df_cleaned[col] = df_cleaned[col].interpolate(method='linear', limit_direction='both').fillna(df_cleaned[col].median())
            
            final_csv = os.path.join(generated_dir, "final_fallback.csv")
            df_cleaned.to_csv(final_csv, index=False)
            config["output_csv"] = final_csv
            config["cleaned_shape"] = list(df_cleaned.shape)
            config["cleaned_missing"] = int(df_cleaned.isna().sum().sum())
    
    else:
        # Traditional
        print(f"[CLEAN] Using traditional cleaning")
        config["method"] = "traditional"
        
        df_cleaned = df.drop_duplicates()
        for col in target_cols:
            df_cleaned[col] = df_cleaned[col].interpolate(method='linear', limit_direction='both').fillna(df_cleaned[col].median())
        
        final_csv = os.path.join(generated_dir, "final_cleaned.csv")
        df_cleaned.to_csv(final_csv, index=False)
        config["output_csv"] = final_csv
        config["cleaned_shape"] = list(df_cleaned.shape)
        config["cleaned_missing"] = int(df_cleaned.isna().sum().sum())
    
    # 保存
    with open(os.path.join(work_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    ti.xcom_push(key="clean_config", value=config)
    ti.xcom_push(key="clean_result", value={
        "status": "success",
        "method": config["method"],
        "output_csv": config.get("output_csv"),
    })
    
    print(f"[CLEAN] Done!")
    return config
def _run_wavestitch_preprocess(config: dict, logs_dir: str) -> dict:
    """运行 WaveStitch 预处理"""
    cmd = [
        "python", os.path.join(WAVESTITCH_DIR, "custom_pipeline", "preprocess.py"),
        "-i", config["input_csv"],
        "-o", config["prepared_dir"],
        "-time_col", config["time_col"],
        "-target_cols", ",".join(config["target_cols"]),
        "-extract_main_segment", "True",
        "-skip_regularize_if_sparse", "True",
    ]
    
    log_file = os.path.join(logs_dir, "preprocess.log")
    
    print(f"[PREPROCESS] {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=WAVESTITCH_DIR)
    
    with open(log_file, 'w') as f:
        f.write(f"CWD: {WAVESTITCH_DIR}\n")
        f.write(f"Command: {' '.join(cmd)}\n\n")
        f.write(f"Return: {result.returncode}\n\n")
        f.write(f"STDOUT:\n{result.stdout}\n\n")
        f.write(f"STDERR:\n{result.stderr}\n")
    
    if result.returncode != 0:
        return {"status": "failed", "error": result.stderr}
    
    return {"status": "success"}


def _run_wavestitch_training(config: dict, logs_dir: str) -> dict:
    """运行 WaveStitch 训练"""
    use_em = os.environ.get("WAVESTITCH_USE_EM", "false").lower() == "true"
    epochs = int(os.environ.get("WAVESTITCH_EPOCHS", "500"))
    batch_size = int(os.environ.get("WAVESTITCH_BATCH_SIZE", "512"))
    
    cmd = [
        "python", os.path.join(WAVESTITCH_DIR, "train_wavestitch_customdata.py"),
        "-d", "custom_csv",
        "-prepared_dir", config["prepared_dir"],
        "-epochs", str(epochs),
        "-batch_size", str(batch_size),
        "-window_size", "32",
        "-stride", "1",
    ]
    
    if use_em:
        em_iterations = int(os.environ.get("WAVESTITCH_EM_ITERATIONS", "5"))
        cmd.extend([
            "-use_em",
            "-em_iterations", str(em_iterations),
            "-epochs_per_em", str(epochs // em_iterations),
            "-ddim_steps", "50",
        ])
    
    log_file = os.path.join(logs_dir, "training.log")
    
    print(f"[TRAINING] {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=WAVESTITCH_DIR)
    
    with open(log_file, 'w') as f:
        f.write(f"CWD: {WAVESTITCH_DIR}\n")
        f.write(f"Command: {' '.join(cmd)}\n\n")
        f.write(f"Return: {result.returncode}\n\n")
        f.write(f"STDOUT:\n{result.stdout}\n\n")
        f.write(f"STDERR:\n{result.stderr}\n")
    
    if result.returncode != 0:
        return {"status": "failed", "error": result.stderr}
    
    # 复制模型
    save_dir = os.path.join(config["prepared_dir"], "saved_model")
    if os.path.exists(save_dir):
        for f_name in os.listdir(save_dir):
            src = os.path.join(save_dir, f_name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(config["model_dir"], f_name))
    
    return {"status": "success"}


def _run_wavestitch_synthesis(config: dict, logs_dir: str) -> dict:
    """运行 WaveStitch 推理"""
    n_trials = int(os.environ.get("WAVESTITCH_N_TRIALS", "1"))
    guidance = float(os.environ.get("WAVESTITCH_GUIDANCE_SCALE", "0.1"))
    
    cmd = [
        "python", os.path.join(WAVESTITCH_DIR, "synthesis_wavestitch_pipeline_strided_preconditioning_customdata.py"),
        "-d", "custom_csv",
        "-prepared_dir", config["prepared_dir"],
        "-n_trials", str(n_trials),
        "-guidance_scale", str(guidance),
        "-synth_mask", "gap_imputation",
        "-stride", "1",
    ]
    
    log_file = os.path.join(logs_dir, "synthesis.log")
    
    print(f"[SYNTHESIS] {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=WAVESTITCH_DIR)
    
    with open(log_file, 'w') as f:
        f.write(f"CWD: {WAVESTITCH_DIR}\n")
        f.write(f"Command: {' '.join(cmd)}\n\n")
        f.write(f"Return: {result.returncode}\n\n")
        f.write(f"STDOUT:\n{result.stdout}\n\n")
        f.write(f"STDERR:\n{result.stderr}\n")
    
    if result.returncode != 0:
        return {"status": "failed", "error": result.stderr}
    
    # 查找输出文件
    output_csv = None
    patterns = [
        os.path.join(config["prepared_dir"], "..", "generated_*", "**", "*.csv"),
        os.path.join(config["generated_dir"], "**", "*.csv"),
    ]
    
    for pattern in patterns:
        candidates = glob.glob(pattern, recursive=True)
        candidates = [c for c in candidates if 'config' not in c.lower()]
        if candidates:
            output_csv = max(candidates, key=os.path.getmtime)
            break
    
    if output_csv:
        dst = os.path.join(config["generated_dir"], "wavestitch_output.csv")
        shutil.copy2(output_csv, dst)
        output_csv = dst
    
    return {"status": "success", "output_csv": output_csv}


def _run_outlier_removal(input_csv: str, output_csv: str, logs_dir: str) -> dict:
    """运行异常值移除"""
    cmd = [
        "python", os.path.join(WAVESTITCH_DIR, "outlier_removal.py"),
        "-i", input_csv,
        "-o", output_csv,
        "-methods", "negative", "physical", "iqr",
        "-fill", "interpolate",
    ]
    
    log_file = os.path.join(logs_dir, "outlier_removal.log")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=WAVESTITCH_DIR)
    
    with open(log_file, 'w') as f:
        f.write(f"Command: {' '.join(cmd)}\n\n")
        f.write(f"Return: {result.returncode}\n\n")
        f.write(f"STDOUT:\n{result.stdout}\n\n")
        f.write(f"STDERR:\n{result.stderr}\n")
    
    if result.returncode != 0:
        return {"status": "failed", "error": result.stderr}
    
    return {"status": "success"}

# ============ Validate Task ============

def validate_cleaned_data(**context):
    """
    验证清洗后的数据质量
    """
    ti = context['ti']
    
    clean_config = ti.xcom_pull(task_ids="clean_dirty_data", key="clean_config")
    clean_result = ti.xcom_pull(task_ids="clean_dirty_data", key="clean_result")
    
    if not clean_config or not clean_result:
        raise ValueError("No clean result found")
    
    output_csv = clean_config.get("output_csv")
    
    if not output_csv or not os.path.exists(output_csv):
        raise FileNotFoundError(f"Cleaned data not found: {output_csv}")
    
    # 加载清洗后的数据
    df = pd.read_csv(output_csv)
    
    # 验证检查
    validation = {
        "timestamp": datetime.utcnow().isoformat(),
        "input_file": output_csv,
        "checks": {},
        "passed": True,
    }
    
    # 1. 缺失值检查
    missing_total = df.isna().sum().sum()
    missing_by_col = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0}
    
    validation["checks"]["missing_values"] = {
        "total": int(missing_total),
        "by_column": missing_by_col,
        "passed": missing_total == 0,
    }
    
    if missing_total > 0:
        validation["passed"] = False
    
    # 2. 重复行检查
    duplicates = df.duplicated().sum()
    
    validation["checks"]["duplicates"] = {
        "count": int(duplicates),
        "passed": duplicates == 0,
    }
    
    # 3. 数据形状检查
    validation["checks"]["shape"] = {
        "rows": len(df),
        "columns": len(df.columns),
        "original_rows": clean_config.get("original_shape", [0])[0],
    }
    
    # 4. 数值范围检查（针对数值列）
    numeric_cols = df.select_dtypes(include=["number"]).columns
    range_issues = []
    
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        
        # 检查是否有无穷值
        if pd.isna(col_min) or pd.isna(col_max):
            range_issues.append(f"{col}: contains NaN")
        elif col_min == float('-inf') or col_max == float('inf'):
            range_issues.append(f"{col}: contains infinity")
    
    validation["checks"]["numeric_range"] = {
        "issues": range_issues,
        "passed": len(range_issues) == 0,
    }
    
    if range_issues:
        validation["passed"] = False
    
    # 5. 与原始数据对比
    original_missing = clean_config.get("original_missing", 0)
    cleaned_missing = clean_config.get("cleaned_missing", 0)
    
    validation["checks"]["improvement"] = {
        "original_missing": original_missing,
        "cleaned_missing": cleaned_missing,
        "missing_reduced": original_missing - cleaned_missing,
        "improvement_rate": round((original_missing - cleaned_missing) / max(original_missing, 1) * 100, 2),
    }
    
    # 保存验证报告
    work_dir = clean_config.get("work_dir")
    if work_dir:
        report_path = os.path.join(work_dir, "validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(validation, f, indent=2, default=str)
    
    ti.xcom_push(key="validation_result", value=validation)
    
    print(f"[VALIDATE] Result: {json.dumps(validation, indent=2, default=str)}")
    
    # 如果验证失败，可以选择抛出异常或继续
    if not validation["passed"]:
        print(f"[VALIDATE] WARNING: Validation failed!")
        # raise ValueError("Data validation failed")
    
    return validation


# ============ Store Task ============

def store_curated_data(**context):
    """
    存储清洗后的数据和相关文件到 Data Lake
    """
    ti = context['ti']
    
    clean_config = ti.xcom_pull(task_ids="clean_dirty_data", key="clean_config")
    clean_result = ti.xcom_pull(task_ids="clean_dirty_data", key="clean_result")
    validation = ti.xcom_pull(task_ids="validate_cleaned_data", key="validation_result")
    
    if not clean_config:
        raise ValueError("No clean config found")
    
    work_dir = clean_config.get("work_dir")
    output_csv = clean_config.get("output_csv")
    source = clean_config.get("source", {})
    run_id = clean_config.get("run_id")
    method = clean_config.get("method", "unknown")
    
    # S3 配置
    bucket = Variable.get("S3_BUCKET", default_var="airflow-bucket")
    datalake_prefix = Variable.get("DATALAKE_PREFIX", default_var="datalake/cleaned")
    
    # 生成存储路径
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    original_filename = os.path.basename(source.get("key", "unknown"))
    original_name = os.path.splitext(original_filename)[0]
    
    s3_base_prefix = f"{datalake_prefix}/{original_name}/{timestamp}_{run_id}"
    
    print(f"[STORE] Destination: s3://{bucket}/{s3_base_prefix}/")
    
    store_result = {
        "timestamp": timestamp,
        "source": source,
        "method": method,
        "destination": {
            "bucket": bucket,
            "prefix": s3_base_prefix,
            "uri": f"s3://{bucket}/{s3_base_prefix}/",
        },
        "files": {},
    }
    
    # 1. 上传最终清洗数据（快速访问）
    if output_csv and os.path.exists(output_csv):
        quick_key = f"{datalake_prefix}/results/{original_name}_cleaned_{timestamp}.csv"
        
        df = pd.read_csv(output_csv)
        save_df_to_object_store(
            df=df,
            key=quick_key,
            bucket=bucket,
            fmt="csv",
        )
        
        store_result["files"]["quick_access"] = {
            "key": quick_key,
            "uri": f"s3://{bucket}/{quick_key}",
        }
        
        print(f"[STORE] Quick access: s3://{bucket}/{quick_key}")
    
    # 2. 上传完整工作目录（包括模型、日志等）
    if work_dir and os.path.exists(work_dir):
        # 创建清单
        manifest_metadata = {
            "run_id": run_id,
            "method": method,
            "source": source,
            "validation": validation,
            "clean_result": clean_result,
        }
        
        manifest = create_wavestitch_manifest(work_dir, manifest_metadata)
        
        # 上传整个目录
        upload_result = upload_directory_to_s3(
            local_dir=work_dir,
            s3_prefix=s3_base_prefix,
            bucket=bucket,
            exclude_patterns=['__pycache__', '.pyc', '.git', 'input.csv'],  # 排除原始输入
        )
        
        store_result["files"]["full_archive"] = {
            "prefix": s3_base_prefix,
            "uri": f"s3://{bucket}/{s3_base_prefix}/",
            "files_uploaded": upload_result.get("files_uploaded", 0),
            "total_size_mb": upload_result.get("total_size_mb", 0),
        }
        
        print(f"[STORE] Archive: {upload_result.get('files_uploaded', 0)} files, "
              f"{upload_result.get('total_size_mb', 0)} MB")
    
    # 3. 保存元数据
    metadata_key = f"{s3_base_prefix}/metadata.json"
    
    metadata = {
        "created_at": timestamp,
        "run_id": run_id,
        "method": method,
        "source": source,
        "clean_result": clean_result,
        "validation": validation,
        "store_result": store_result,
    }
    
    save_json(metadata, key=metadata_key, bucket=bucket)
    store_result["files"]["metadata"] = {"key": metadata_key}
    
    ti.xcom_push(key="store_result", value=store_result)
    
    print(f"[STORE] Complete: {json.dumps(store_result, indent=2, default=str)}")
    
    return store_result


# ============ Visualization Task ============

def visualization(**context):
    """
    生成可视化报告
    """
    ti = context['ti']
    
    clean_config = ti.xcom_pull(task_ids="clean_dirty_data", key="clean_config")
    clean_result = ti.xcom_pull(task_ids="clean_dirty_data", key="clean_result")
    validation = ti.xcom_pull(task_ids="validate_cleaned_data", key="validation_result")
    store_result = ti.xcom_pull(task_ids="store_curated_data", key="store_result")
    
    work_dir = clean_config.get("work_dir") if clean_config else None
    output_csv = clean_config.get("output_csv") if clean_config else None
    
    viz_result = {
        "timestamp": datetime.utcnow().isoformat(),
        "charts": [],
        "summary": {},
    }
    
    # 生成摘要
    viz_result["summary"] = {
        "method": clean_config.get("method") if clean_config else None,
        "input_shape": clean_config.get("original_shape") if clean_config else None,
        "output_shape": clean_config.get("cleaned_shape") if clean_config else None,
        "input_missing": clean_config.get("original_missing") if clean_config else None,
        "output_missing": clean_config.get("cleaned_missing") if clean_config else None,
        "validation_passed": validation.get("passed") if validation else None,
        "stored_at": store_result.get("destination", {}).get("uri") if store_result else None,
    }
    
    # 如果有数据，生成图表
    if output_csv and os.path.exists(output_csv) and work_dir:
        try:
            import matplotlib
            matplotlib.use('Agg')  # 无头模式
            import matplotlib.pyplot as plt
            
            df = pd.read_csv(output_csv)
            viz_dir = os.path.join(work_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # 1. 数据概览图
            numeric_cols = df.select_dtypes(include=["number"]).columns[:6]  # 最多6列
            
            if len(numeric_cols) > 0:
                n_cols = min(3, len(numeric_cols))
                n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
                
                for idx, col in enumerate(numeric_cols):
                    ax = axes[idx]
                    ax.plot(df[col].values, linewidth=0.5, alpha=0.7)
                    ax.set_title(col, fontsize=10)
                    ax.set_xlabel('Index')
                    ax.grid(True, alpha=0.3)
                
                for idx in range(len(numeric_cols), len(axes)):
                    axes[idx].set_visible(False)
                
                plt.suptitle('Cleaned Data Overview', fontsize=12)
                plt.tight_layout()
                
                overview_path = os.path.join(viz_dir, "data_overview.png")
                plt.savefig(overview_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                viz_result["charts"].append({
                    "name": "data_overview",
                    "path": overview_path,
                })
            
            # 2. 缺失值热力图（如果有缺失）
            if df.isna().sum().sum() > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                missing_matrix = df.isna().astype(int)
                ax.imshow(missing_matrix.T, aspect='auto', cmap='Reds')
                ax.set_yticks(range(len(df.columns)))
                ax.set_yticklabels(df.columns, fontsize=8)
                ax.set_xlabel('Row Index')
                ax.set_title('Missing Values Heatmap')
                
                plt.tight_layout()
                
                missing_path = os.path.join(viz_dir, "missing_heatmap.png")
                plt.savefig(missing_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                viz_result["charts"].append({
                    "name": "missing_heatmap",
                    "path": missing_path,
                })
            
            print(f"[VIZ] Generated {len(viz_result['charts'])} charts")
            
            # 上传图表到 S3
            if store_result and viz_result["charts"]:
                bucket = Variable.get("S3_BUCKET", default_var="airflow-bucket")
                base_prefix = store_result.get("destination", {}).get("prefix", "")
                
                for chart in viz_result["charts"]:
                    chart_key = f"{base_prefix}/visualizations/{os.path.basename(chart['path'])}"
                    
                    from helpers.object_store import upload_file_to_s3
                    upload_file_to_s3(
                        local_path=chart["path"],
                        key=chart_key,
                        bucket=bucket,
                    )
                    
                    chart["s3_key"] = chart_key
                    chart["s3_uri"] = f"s3://{bucket}/{chart_key}"
            
        except Exception as e:
            print(f"[VIZ] Warning: Failed to generate charts: {e}")
            viz_result["error"] = str(e)
    
    # 清理本地文件（可选）
    if Variable.get("WAVESTITCH_CLEANUP_LOCAL", default_var="false").lower() == "true":
        if work_dir and os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            viz_result["cleanup"] = {"removed": work_dir}
            print(f"[VIZ] Cleaned up: {work_dir}")
    
    ti.xcom_push(key="viz_result", value=viz_result)
    
    print(f"[VIZ] Summary: {json.dumps(viz_result['summary'], indent=2, default=str)}")
    
    return viz_result


# ============ DAG 定义 ============

default_args = {
    "owner": "yd data-platform-pipeline",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="data_quality_and_cleaning_pipeline_v2",
    description="N2N ELT: Load → QC → Report → Clean (WaveStitch) → Validate → Store → Viz",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["GX", "WaveStitch", "TimeSeries", "DataLake"],
) as dag:

    # 1. Load & Profile
    load = PythonOperator(
        task_id="load_raw_data",
        python_callable=load_raw_data,
    )

    # 2. Branch: TS vs Tabular
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

    # 3. Report
    report = PythonOperator(
        task_id="report_dqc",
        python_callable=report_dqc,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # 4. Clean (整合 WaveStitch)
    clean = PythonOperator(
        task_id="clean_dirty_data",
        python_callable=clean_dirty_data,
        execution_timeout=timedelta(hours=6),  # 允许长时间训练
    )

    # 5. Validate
    validate = PythonOperator(
        task_id="validate_cleaned_data",
        python_callable=validate_cleaned_data,
    )

    # 6. Store
    store = PythonOperator(
        task_id="store_curated_data",
        python_callable=store_curated_data,
    )

    # 7. Visualization
    viz = PythonOperator(
        task_id="visualization",
        python_callable=visualization,
    )

    # ============ Dependencies ============
    load >> branch
    branch >> qc_task >> report
    branch >> ts_qc_task >> report
    report >> clean >> validate >> store >> viz