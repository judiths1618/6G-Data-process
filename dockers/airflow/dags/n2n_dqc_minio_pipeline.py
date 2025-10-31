# dags/n2n_dqc_minio_pipeline.py
from __future__ import annotations
import json
from datetime import timedelta
from typing import Optional, Dict, Any

import pandas as pd
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

# ---- helpers (public API from helpers/dqc_utils.py) ----
from helpers.dqc_utils import (
    # IO
    load_df_from_minio, save_df_to_minio, _s3,
    PRIMARY_KEY_RAW,
    # Detection / normalization / QC
    detect_timestamp_column, build_schema_profile,
    normalize_ts_for_gap, compute_time_gaps_smart,
    # Config exported by helpers (single source of truth)
    PROJECT, DATASET_NAME, TARGET, S3_BUCKET,
    REPORT_PREFIX, CURATED_PREFIX,
    DEFAULT_TZ, TS_STD_COL,
    TS_EXPECTED_FREQ, TS_GAP_TOL_MULT, TS_GROUP_KEYS
)
from helpers.dqc_metrics_methods import run_metrics
from helpers.dqc_utils import detect_primary_key

# ---------------- DAG ----------------
default_args = {
    "owner": "data-platform",
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="n2n_dqc_minio_pipeline",
    description="N2N ELT: Load → Profile → Branch(QC) → Report → Clean → Store → Viz",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["n2n","minio","dqc","elt"],
) as dag:

    # 1) Load + Profile
    def load_raw_data(ti, **_):
        input_key = Variable.get("N2N_INPUT_KEY")  # required
        assert input_key, "Airflow Variable N2N_INPUT_KEY is required (e.g., raw/data.csv)"

        df, fmt = load_df_from_minio(input_key)

        # detect primary key candidates
        # detect PK (explicit wins; else heuristic)
        pk_info = detect_primary_key(df, configured_pk=PRIMARY_KEY_RAW or None)
        pk_cols = pk_info.get("primary_key", [])

        # info = {
        #     "project": PROJECT,
        #     "dataset": DATASET_NAME,
        #     "bucket": S3_BUCKET,
        #     "input_key": INPUT_KEY,
        #     "format": fmt,
        #     "rows": int(len(df)),
        #     "cols": int(df.shape[1]),
        #     "columns": list(df.columns),
        #     "has_timestamp_col": TIMESTAMP_COL in df.columns,
        #     "preview": df.head(5).to_dict(orient="records"),
        #     "pk": {"columns": pk_cols, "uniqueness": pk_info.get("uniqueness"), "null_rows": pk_info.get("null_rows")},
        # }
        # print("[LOAD] from s3://%s/%s\n%s" % (S3_BUCKET, INPUT_KEY, json.dumps(info, ensure_ascii=False, indent=2)))

        # ti.xcom_push(key="raw_handle", value={"bucket": S3_BUCKET, "key": INPUT_KEY, "format": fmt})
        # ti.xcom_push(key="schema", value={"columns": info["columns"], "target": TARGET, "timestamp_col": TIMESTAMP_COL})
        # ti.xcom_push(key="preview", value=info["preview"])
        # ti.xcom_push(key="is_time_series", value=info["has_timestamp_col"])
        # ti.xcom_push(key="pk_cols", value=pk_cols)
        # ti.xcom_push(key="pk_detect_info", value=pk_info)
        # return info
        # schema profile (auto-detect timestamp)
        ts_col, _ = detect_timestamp_column(df, configured_name=Variable.get("N2N_TIMESTAMP_COL", default_var=""))
        profile = build_schema_profile(df, configured_ts=ts_col, target=TARGET)

        info = {
            "project": PROJECT,
            "dataset": DATASET_NAME,
            "bucket": S3_BUCKET,
            "input_key": input_key,
            "format": fmt,
            "rows": int(len(df)),
            "cols": int(df.shape[1]),
            "columns": list(df.columns),
            "detected_timestamp_col": ts_col,
            "preview": df.head(5).to_dict(orient="records"),
            "profile": profile,
        }

        print("[LOAD] ", json.dumps({
            "s3": f"s3://{S3_BUCKET}/{input_key}", "fmt": fmt,
            "rows": info["rows"], "cols": info["cols"],
            "detected_ts": ts_col
        }, ensure_ascii=False))

        # XCom (keep DF off XCom)
        ti.xcom_push(key="raw_handle", value={"bucket": S3_BUCKET, "key": input_key, "format": fmt})
        ti.xcom_push(key="schema_profile", value=profile)
        ti.xcom_push(key="detected_ts_col", value=ts_col)
        ti.xcom_push(key="preview", value=info["preview"])
        ti.xcom_push(key="is_time_series", value=bool(ts_col))
        return info

    t_load = PythonOperator(task_id="load_raw_data", python_callable=load_raw_data)

    # 2) Branch: Is Time Series?
    def branch_ts_or_tabular(ti, **_):
        is_ts = bool(ti.xcom_pull(task_ids="load_raw_data", key="is_time_series"))
        next_task = "ts_quality" if is_ts else "tabular_quality"
        print(f"[BRANCH] is_time_series={is_ts} → {next_task}")
        return next_task

    t_branch = BranchPythonOperator(task_id="is_time_series", python_callable=branch_ts_or_tabular)

    # 3a) Time-series QC
    # def ts_quality(ti, **_):
    #     handle = ti.xcom_pull(task_ids="load_raw_data", key="raw_handle")
    #     schema = ti.xcom_pull(task_ids="load_raw_data", key="schema_profile") or {}
    #     ts_detected = ti.xcom_pull(task_ids="load_raw_data", key="detected_ts_col")

    #     df, _ = load_df_from_minio(handle["key"])

    #     # Normalize to UTC column for robust analysis
    #     df_std, std_meta = normalize_ts_for_gap(
    #         ti=ti,
    #         df=df,
    #         dataset_name=DATASET_NAME,
    #         configured_ts_col=ts_detected,
    #         out_col=TS_STD_COL,
    #     )

    #     # Basic metrics
    #     duplicate_rows = int(df_std.duplicated().sum())
    #     missing_rate = float(df_std.isna().mean().mean())

    #     # Numeric IQR outliers
    #     outlier_count = 0
    #     for c in df_std.select_dtypes(include=["number"]).columns:
    #         s = pd.to_numeric(df_std[c], errors="coerce").dropna()
    #         if s.empty: continue
    #         q1, q3 = s.quantile(0.25), s.quantile(0.75)
    #         iqr = q3 - q1
    #         if iqr > 0:
    #             outlier_count += int(((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum())

    #     # Gap analysis (group-aware)
    #     gap_results: Dict[str, Any] = {}
    #     expected = pd.to_timedelta(TS_EXPECTED_FREQ) if TS_EXPECTED_FREQ else None
    #     if TS_STD_COL in df_std.columns:
    #         if TS_GROUP_KEYS:
    #             total_windows = total_points = 0
    #             for keys, g in df_std.groupby(TS_GROUP_KEYS, dropna=False, sort=False):
    #                 res = compute_time_gaps_smart(
    #                     df=g, ts_col=TS_STD_COL, expected_delta=expected,
    #                     tol_mult=TS_GAP_TOL_MULT, window=200, respect_calendar=True
    #                 )
    #                 gap_results[str(keys)] = res
    #                 total_windows += res["counts"]["missing_windows"]
    #                 total_points  += res["counts"]["missing_points"]
    #             gap_results["_summary"] = {
    #                 "groups": len([k for k in gap_results.keys() if k != "_summary"]),
    #                 "total_missing_windows": int(total_windows),
    #                 "total_missing_points": int(total_points),
    #             }
    #         else:
    #             gap_results["_all"] = compute_time_gaps_smart(
    #                 df=df_std, ts_col=TS_STD_COL, expected_delta=expected,
    #                 tol_mult=TS_GAP_TOL_MULT, window=200, respect_calendar=True
    #             )

    #     qc = {
    #         "dimension": "time_series",
    #         "timestamp_col": ts_detected,
    #         "standardized_col": TS_STD_COL if TS_STD_COL in df_std.columns else None,
    #         "standardize_meta": std_meta,
    #         "checks": {
    #             "completeness": {"missing_rate": missing_rate},
    #             "duplications": {
    #                 "duplicate_rows": duplicate_rows,
    #                 "duplicate_timestamps": int(df_std[TS_STD_COL].duplicated().sum()) if TS_STD_COL in df_std.columns else 0,
    #             },
    #             "outliers": {"iqr_suspects": int(outlier_count)},
    #             "time_gaps": gap_results or {"note": "no standardized time column"},
    #         },
    #         "summary": (
    #             f"TSQC on '{TS_STD_COL}' (tol={TS_GAP_TOL_MULT}×; expected={TS_EXPECTED_FREQ or 'auto'})."
    #             if TS_STD_COL in df_std.columns else
    #             "No standardized time column; gap analysis skipped."
    #         ),
    #     }

    #     # optional: persist detailed gaps
    #     try:
    #         # Use load time for a tag to keep ordering; or include execution date
    #         tag = ti.xcom_pull(task_ids="report_dqc", key="report") or {}
    #         tag = tag.get("ts") or pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    #         if gap_results:
    #             gaps_key = f"{REPORT_PREFIX}/{DATASET_NAME}/{tag}_gaps.json"
    #             _s3().put_object(
    #                 Bucket=S3_BUCKET, Key=gaps_key,
    #                 Body=json.dumps(gap_results, ensure_ascii=False, indent=2).encode("utf-8"),
    #                 ContentType="application/json"
    #             )
    #             qc["checks"]["time_gaps_location"] = f"s3://{S3_BUCKET}/{gaps_key}"
    #     except Exception as e:
    #         print(f"[TSQC] warn: failed to persist gap details: {e}")

    #     print("[TSQC] " + json.dumps(qc, ensure_ascii=False, indent=2))
    #     ti.xcom_push(key="qc_result", value=qc)

    #     # tiny preview for UI
    #     if ts_detected and TS_STD_COL in df_std.columns and ts_detected in df_std.columns:
    #         ti.xcom_push(
    #             key="ts_preview",
    #             value=df_std[[ts_detected, TS_STD_COL]].head(5).astype(str).to_dict("records"),
    #         )
    #     # pass normalized handle forward (for potential later steps)
    #     ti.xcom_push(key="normalized_time_present", value=TS_STD_COL in df_std.columns)
    #     return qc
    
    def ts_quality(ti, **_):
        handle = ti.xcom_pull(task_ids="load_raw_data", key="raw_handle")
        schema = ti.xcom_pull(task_ids="load_raw_data", key="schema_profile") or {}

        df, _ = load_df_from_minio(handle["key"])
        res = run_metrics(df, is_time_series=True, profile=schema)

        qc = {
            "dimension": "time_series",
            "checks": res["checks"],
            "summary": "TSQC (metrics registry)",
        }
        # surface some common aggregates for gates
        comp = res["checks"].get("completeness", {}).get("metrics", {})
        dup  = res["checks"].get("pk_duplicates", {}).get("metrics", {})
        gaps = res["checks"].get("time_gaps_adaptive", {}).get("metrics", {})
        qc["aggregates"] = {
            "missing_rate": comp.get("missing_rate"),
            "duplicate_pk_rows": dup.get("duplicate_pk_rows", dup.get("duplicate_rows_fallback", 0)),
            "gap_windows": (gaps.get("_all", {}).get("counts", {}).get("missing_windows", 0)
                            if isinstance(gaps, dict) else gaps.get("_summary", {}).get("total_missing_windows", 0)),
        }
        ti.xcom_push(key="qc_result", value=qc)
        ti.xcom_push(key="recommended_actions", value=res["recommended_actions"])
        return qc

    t_tsqc = PythonOperator(task_id="ts_quality", python_callable=ts_quality)

    # 3b) Tabular QC
    # def tabular_quality(ti, **_):
    #     handle = ti.xcom_pull(task_ids="load_raw_data", key="raw_handle")
    #     df, _ = load_df_from_minio(handle["key"])

    #     dup = int(df.duplicated().sum())
    #     miss = {c: int(df[c].isna().sum()) for c in df.columns}

    #     outliers: Dict[str, int] = {}
    #     for c in df.select_dtypes(include=["number"]).columns:
    #         s = pd.to_numeric(df[c], errors="coerce").dropna()
    #         if s.empty: continue
    #         q1, q3 = s.quantile(0.25), s.quantile(0.75)
    #         iqr = q3 - q1
    #         if iqr > 0:
    #             outliers[c] = int(((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum())

    #     qc = {
    #         "dimension": "tabular",
    #         "checks": {
    #             "completeness": {"missing_by_col": miss},
    #             "duplications": {"duplicate_rows": dup},
    #             "outliers": outliers,
    #         },
    #         "summary": "Tabular QC done.",
    #     }
    #     print("[QC] " + json.dumps(qc, ensure_ascii=False, indent=2))
    #     ti.xcom_push(key="qc_result", value=qc)
    #     return qc

    def tabular_quality(ti, **_):
        handle = ti.xcom_pull(task_ids="load_raw_data", key="raw_handle")
        schema = ti.xcom_pull(task_ids="load_raw_data", key="schema_profile") or {}

        df, _ = load_df_from_minio(handle["key"])
        res = run_metrics(df, is_time_series=False, profile=schema)

        qc = {
            "dimension": "tabular",
            "checks": res["checks"],
            "summary": "Tabular QC (metrics registry)",
        }
        comp = res["checks"].get("completeness", {}).get("metrics", {})
        dup  = res["checks"].get("pk_duplicates", {}).get("metrics", {})
        qc["aggregates"] = {
            "missing_rate": comp.get("missing_rate"),
            "duplicate_pk_rows": dup.get("duplicate_pk_rows", dup.get("duplicate_rows_fallback", 0)),
        }
        ti.xcom_push(key="qc_result", value=qc)
        ti.xcom_push(key="recommended_actions", value=res["recommended_actions"])
        return qc
    t_qc = PythonOperator(task_id="tabular_quality", python_callable=tabular_quality)

    # 4) Report DQC
    # def report_dqc(ti, **_):
    #     # pull whichever produced a QC result
    #     ts_qc = ti.xcom_pull(task_ids="ts_quality", key="qc_result")
    #     tb_qc = ti.xcom_pull(task_ids="tabular_quality", key="qc_result")
    #     qc = ts_qc or tb_qc
    #     assert qc, "No QC result found."

    #     # pick recommended actions
    #     actions = []
    #     if qc["dimension"] == "time_series":
    #         checks = qc["checks"]
    #         gaps = checks.get("time_gaps", {})
    #         has_gap = bool(gaps and (("_all" in gaps and gaps["_all"]["counts"]["missing_windows"] > 0) or
    #                                  ("_summary" in gaps and gaps["_summary"]["total_missing_windows"] > 0)))
    #         actions = []
    #         if has_gap:
    #             actions.append("forward_fill_gaps")
    #         if checks["duplications"]["duplicate_timestamps"] > 0:
    #             actions.append("drop_duplicate_timestamps")
    #         actions.append("clip_outliers_iqr")
    #     else:
    #         checks = qc["checks"]
    #         if any(v > 0 for v in checks["completeness"]["missing_by_col"].values()):
    #             actions.append("impute_missing")
    #         if checks["duplications"]["duplicate_rows"] > 0:
    #             actions.append("drop_duplicates")
    #         actions.append("clip_outliers_iqr")

    #     report = {
    #         "project": PROJECT,
    #         "dataset": DATASET_NAME,
    #         "data_type": qc["dimension"],
    #         "qc": qc,
    #         "recommended_actions": actions or ["noop"],
    #         "ts": pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ"),
    #     }

    #     report_key = f"{REPORT_PREFIX}/{DATASET_NAME}/{report['ts']}_dqc.json"
    #     _s3().put_object(
    #         Bucket=S3_BUCKET, Key=report_key,
    #         Body=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
    #         ContentType="application/json"
    #     )
    #     print(f"[REPORT] s3://{S3_BUCKET}/{report_key}")
    #     ti.xcom_push(key="report", value=report)
    #     ti.xcom_push(key="report_key", value=report_key)
        
    #     return report

    # --- Report DQC (drop-in) ---

    def report_dqc(ti, **_):
        # 1) Gather inputs (works for either branch)
        qc_ts  = ti.xcom_pull(task_ids="ts_quality", key="qc_result")
        qc_tb  = ti.xcom_pull(task_ids="tabular_quality", key="qc_result")
        qc     = qc_ts or qc_tb
        assert qc, "No QC result found."
        actions = (
            ti.xcom_pull(task_ids="ts_quality", key="recommended_actions") or
            ti.xcom_pull(task_ids="tabular_quality", key="recommended_actions") or
            []
        )
        profile = ti.xcom_pull(task_ids="load_raw_data", key="schema_profile") or {}
        raw_handle = ti.xcom_pull(task_ids="load_raw_data", key="raw_handle") or {}
        detected_ts_col = ti.xcom_pull(task_ids="load_raw_data", key="detected_ts_col")

        # 2) Params / gates (UI-overridable)
        p = dag.params
        max_missing_rate = float(p.get("max_missing_rate", 0.20))
        max_gap_windows  = int(p.get("max_gap_windows", 1000))
        allow_dup_pk     = bool(p.get("allow_dup_pk", False))
        hard_fail        = bool(p.get("hard_fail_on_violation", False))

        # 3) Evaluate gates from QC aggregates (set by QC tasks)
        agg = qc.get("aggregates", {})
        status = "pass"; violations = []

        def _v(flag: bool, msg: str):
            nonlocal status
            if flag:
                status = "fail"; violations.append(msg)

        # Missing rate (both paths)
        if agg.get("missing_rate") is not None:
            _v(agg["missing_rate"] > max_missing_rate,
            f"missing_rate>{max_missing_rate:.2f} (got {agg['missing_rate']:.3f})")

        # Duplicates (prefer PK-based)
        dup_val = agg.get("duplicate_pk_rows", 0)
        _v((dup_val > 0) and (not allow_dup_pk),
        f"duplicate_pk_rows>0 (got {dup_val})")

        # Time gaps (TS path only)
        if qc["dimension"] == "time_series":
            gap_w = int(agg.get("gap_windows", 0) or 0)
            _v(gap_w > max_gap_windows,
            f"gap_windows>{max_gap_windows} (got {gap_w})")

        # 4) Build report payload
        run_ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
        report = {
            "project": PROJECT,
            "dataset": DATASET_NAME,
            "data_type": qc["dimension"],
            "profile_head": {
                "rows": profile.get("table", {}).get("rows"),
                "cols": profile.get("table", {}).get("cols"),
                "pk_candidates": profile.get("primary_key_candidates", [])[:5],
            },
            "source": {
                "bucket": S3_BUCKET,
                "input_key": raw_handle.get("key"),
                "format": raw_handle.get("format"),
                "detected_timestamp_col": detected_ts_col,
            },
            "qc": qc,  # full checks tree from metrics registry
            "aggregates": agg,
            "recommended_actions": actions or ["noop"],
            "status": status,
            "violations": violations,
            "ts": run_ts,
        }

        # 5) Persist report + manifest
        report_key = f"{REPORT_PREFIX}/{DATASET_NAME}/{run_ts}_dqc.json"
        _s3().put_object(
            Bucket=S3_BUCKET, Key=report_key,
            Body=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json"
        )

        manifest = {
            "project": PROJECT,
            "dataset": DATASET_NAME,
            "run_ts": run_ts,
            "status": status,
            "violations": violations,
            "input": raw_handle,
            "detected_ts_col": detected_ts_col,
            "recommended_actions": actions,
            "outputs": {}  # to be filled by store_curated_data & visualization
        }
        manifest_key = f"{REPORT_PREFIX}/{DATASET_NAME}/{run_ts}_manifest.json"
        _s3().put_object(
            Bucket=S3_BUCKET, Key=manifest_key,
            Body=json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json"
        )

        # 6) XCom for downstream + optional gate fail
        ti.xcom_push(key="report", value=report)
        ti.xcom_push(key="report_key", value=report_key)
        ti.xcom_push(key="manifest_key", value=manifest_key)
        ti.xcom_push(key="recommended_actions", value=actions)

        print(f"[REPORT] s3://{S3_BUCKET}/{report_key}  (status={status}, violations={violations})")

        if hard_fail and status == "fail":
            from airflow.exceptions import AirflowException
            raise AirflowException(f"Quality gates failed: {violations}")

        return report

    # It depends on one of the two QC tasks; set a permissive trigger rule
    t_report = PythonOperator(
        task_id="report_dqc",
        python_callable=report_dqc,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # 5) Clean (very simplified; applies to either path)
    def clean_dirty_data(ti, **_):
        handle = ti.xcom_pull(task_ids="load_raw_data", key="raw_handle")
        report = ti.xcom_pull(task_ids="report_dqc", key="report")
        actions = report["recommended_actions"]

        df, fmt = load_df_from_minio(handle["key"])

        if "drop_duplicates" in actions:
            df = df.drop_duplicates()

        if "impute_missing" in actions:
            num_cols = df.select_dtypes("number").columns
            cat_cols = [c for c in df.columns if c not in num_cols]
            for c in num_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
            for c in cat_cols:
                mode = df[c].mode()
                df[c] = df[c].fillna(mode.iloc[0] if not mode.empty else df[c].ffill().bfill())

        if "clip_outliers_iqr" in actions:
            for c in df.select_dtypes("number").columns:
                s = pd.to_numeric(df[c], errors="coerce")
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    df[c] = s.clip(lower=lb, upper=ub)

        if "drop_duplicate_timestamps" in actions or "forward_fill_gaps" in actions:
            # If TS, normalize time and optionally forward-fill per group
            ts_col_detected = ti.xcom_pull(task_ids="load_raw_data", key="detected_ts_col")
            if ts_col_detected:
                df_std, _ = normalize_ts_for_gap(
                    ti=ti, df=df, dataset_name=DATASET_NAME,
                    configured_ts_col=ts_col_detected, out_col=TS_STD_COL,
                )
                if "drop_duplicate_timestamps" in actions:
                    df_std = df_std.drop_duplicates(subset=[TS_STD_COL])
                if "forward_fill_gaps" in actions:
                    # forward fill numeric per group or globally
                    if TS_GROUP_KEYS:
                        df_std = (
                            df_std
                            .sort_values(TS_STD_COL)
                            .groupby(TS_GROUP_KEYS)
                            .apply(lambda g: g.ffill().bfill())
                            .reset_index(drop=True)
                        )
                    else:
                        df_std = df_std.sort_values(TS_STD_COL).ffill().bfill()
                df = df_std

        # write curated
        ts_tag = report["ts"]
        out_fmt = "parquet" if fmt == "parquet" else "csv"
        curated_key = f"{CURATED_PREFIX}/{DATASET_NAME}/{ts_tag}_clean.{ 'parquet' if out_fmt=='parquet' else 'csv'}"
        save_df_to_minio(df, curated_key, fmt=out_fmt, index=False)
        print(f"[STORE] cleaned → s3://{S3_BUCKET}/{curated_key}")

        ti.xcom_push(key="curated_key", value=curated_key)
        ti.xcom_push(key="curated_rows", value=int(len(df)))
        ti.xcom_push(key="curated_cols", value=int(df.shape[1]))
        return {"curated_key": curated_key, "rows": int(len(df)), "cols": int(df.shape[1])}

    t_clean = PythonOperator(task_id="clean_dirty_data", python_callable=clean_dirty_data)

    # 6) Store summary (already stored in clean step; provide a manifest)
    def store_curated_data(ti, **_):
        curated_key = ti.xcom_pull(task_ids="clean_dirty_data", key="curated_key")
        curated_rows = ti.xcom_pull(task_ids="clean_dirty_data", key="curated_rows")
        curated_cols = ti.xcom_pull(task_ids="clean_dirty_data", key="curated_cols")
        summary = {
            "bucket": S3_BUCKET,
            "curated_key": curated_key,
            "rows": curated_rows,
            "cols": curated_cols,
        }
        print("[STORE] Summary: ", json.dumps(summary, ensure_ascii=False))
        ti.xcom_push(key="store_summary", value=summary)
        return summary

    t_store = PythonOperator(task_id="store_curated_data", python_callable=store_curated_data)

    # 7) Visualization stub (persist quick stats JSON next to curated)
    def visualization(ti, **_):
        summary = ti.xcom_pull(task_ids="store_curated_data", key="store_summary")
        curated_key = summary["curated_key"]
        df, _ = load_df_from_minio(curated_key)

        stats = {
            "rows": int(len(df)),
            "cols": int(df.shape[1]),
            "numeric_cols": df.select_dtypes("number").columns.tolist(),
            "sample": df.head(5).to_dict(orient="records"),
        }
        base_name = curated_key.split("/")[-1]
        viz_key = curated_key.replace(base_name, f"viz_{base_name.split('.')[0]}.json")
        _s3().put_object(
            Bucket=S3_BUCKET, Key=viz_key,
            Body=json.dumps(stats, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json"
        )
        print(f"[VIZ] s3://{S3_BUCKET}/{viz_key}")
        ti.xcom_push(key="viz_key", value=viz_key)
        return stats

    t_viz = PythonOperator(task_id="visualization", python_callable=visualization)

    # ----- Dependencies -----
    t_load >> t_branch
    t_branch >> t_qc >> t_report
    t_branch >> t_tsqc >> t_report
    t_report >> t_clean >> t_store >> t_viz
