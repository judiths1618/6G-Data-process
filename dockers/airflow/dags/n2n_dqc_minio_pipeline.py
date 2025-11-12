# dags/n2n_dqc_minio_pipeline.py
from __future__ import annotations
import os
import json
from datetime import timedelta
from typing import Optional, Dict, Any

import pandas as pd
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

import io
# from dockers.airflow.dags.sampling_remasker import X_test_eval
import torch
import numpy as np
from tqdm import tqdm
from imputers.remasker.remasker_impute import ReMasker
from hyperimpute.utils.serialization import load, save
from imputers.generate_mask import generate_mask
from helpers.preprocessor import split_and_store_dataset


# ---- helpers (public API from helpers/dqc_utils.py) ----
from helpers.dqc_utils import (
    # IO
    load_df_from_minio, save_df_to_minio, _s3,
    PRIMARY_KEY_RAW,TIMESTAMP_COL,DATASET_NAME,
    # Detection / normalization / QC
    detect_timestamp_column, build_schema_profile,
    normalize_ts_for_gap, compute_time_gaps_smart,
    # Config exported by helpers (single source of truth)
    PROJECT, TARGET, S3_BUCKET,
    REPORT_PREFIX, CURATED_PREFIX,
    DEFAULT_TZ, TS_STD_COL,
    TS_EXPECTED_FREQ, TS_GAP_TOL_MULT, TS_GROUP_KEYS
)
from helpers.dqc_metrics_methods import run_metrics
from helpers.dqc_utils import detect_primary_key, build_report_key, build_curated_key
from helpers.preprocessor import Preprocessor, get_eval, load_imputer_from_s3
# from helpers.preprocessor import split_complete_incomplete_minio

# ---------------- DAG ----------------
default_args = {
    "owner": "yd data-platform",
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="n2n_dqc_minio_pipeline",
    description="N2N ELT: Load → Profile → Branch(QC) → Report → Clean (duplicates, missing, TS gaps) → Store → Viz",
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

        ts_col, ts_info = detect_timestamp_column(df, configured_name=(TIMESTAMP_COL if TIMESTAMP_COL in df.columns else None))
        is_ts = ts_col is not None

        pk_info = detect_primary_key(df, configured_pk=PRIMARY_KEY_RAW or None)
        pk_cols = pk_info.get("primary_key", [])
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
            "pk": {"columns": pk_cols, "uniqueness": pk_info.get("uniqueness"), "null_rows": pk_info.get("null_rows")},
            "preview": df.head(5).to_dict(orient="records"),
            "profile": profile,
        }

        print("[LOAD]", json.dumps(info, ensure_ascii=False, indent=2))

        # XCom (keep DF off XCom)
        ti.xcom_push(key="raw_handle", value={"bucket": S3_BUCKET, "key": input_key, "format": fmt})
        ti.xcom_push(key="schema", value={"columns": info["columns"], "target": TARGET})
        ti.xcom_push(key="preview", value=info["preview"])
        ti.xcom_push(key="is_time_series", value=is_ts)
        ti.xcom_push(key="detected_ts_col", value=ts_col)
        ti.xcom_push(key="pk_cols", value=pk_cols)
        ti.xcom_push(key="pk_detect_info", value=pk_info)
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
    def report_dqc(ti, ts_nodash, **_):
        qc = ti.xcom_pull(task_ids=["ts_quality", "tabular_quality"], key="qc_result")
        qc = next((x for x in qc if x is not None), None)

        schema = ti.xcom_pull(task_ids="load_raw_data", key="schema")
        is_ts  = bool(ti.xcom_pull(task_ids="load_raw_data", key="is_time_series"))
        handle = ti.xcom_pull(task_ids="load_raw_data", key="raw_handle")  # {"bucket":..., "key":..., "format":...}
        input_key = handle["key"]

        actions = []
        if is_ts:
            # 这里可按你的逻辑生成动作
            actions = ["drop_duplicates", "impute_missing", "remove_ts_outliers", "forward_fill_gaps"]
        if not is_ts:
            # checks = qc.get("checks", {})

            # # (1) check missing_by_col
            # completeness = checks.get("completeness", {})
            # # missing_by_col = completeness.get("missing_by_col", {})
            # missing_by_col = qc.get("checks", {}).get("completeness", {}).get("metrics", {}).get("missing_by_col", {})

            # miss_any = any(v > 0 for v in missing_by_col.values()) if isinstance(missing_by_col, dict) else False

            # # (2) fallback to missing_rate if available
            # missing_rate = completeness.get("missing_rate", 0)
            # if not miss_any and isinstance(missing_rate, (float, int)):
            #     miss_any = missing_rate > 0.0

            # if miss_any:
            #     actions.append("impute_missing")

            # # (3) check duplicates
            # duplicate_rows = checks.get("duplications", {}).get("duplicate_rows", 0)
            # if isinstance(duplicate_rows, int) and duplicate_rows > 0:
            #     actions.append("drop_duplicates")

            # actions.append("clip_outliers_iqr")
            actions = ["drop_duplicates", "impute_missing", "clip_outliers_iqr"]
        
        report = {
            "project": PROJECT,
            "dataset": DATASET_NAME,
            "data_type": "time_series" if is_ts else "tabular",
            "schema": schema,
            "qc": qc,
            "recommended_actions": actions or ["noop"],
            "ts": ts_nodash,
            "input": {
                "bucket": S3_BUCKET,
                "key": input_key,
            },
            "output_layout": {
                "root": REPORT_PREFIX,
                "resolved_key": None,  # 稍后填入
            },
        }

        # <<< 关键：构造报告 Key >>>
        report_key = build_report_key(input_key=input_key, run_ts=ts_nodash, suffix="dqc.json")
        report["output_layout"]["resolved_key"] = report_key

        _s3().put_object(
            Bucket=S3_BUCKET,
            Key=report_key,
            Body=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

        print(f"[REPORT] saved to s3://{S3_BUCKET}/{report_key}\n{json.dumps(report, ensure_ascii=False, indent=2)}")
        ti.xcom_push(key="report", value=report)
        ti.xcom_push(key="report_key", value=report_key)
        ti.xcom_push(key="recommended_actions", value=report["recommended_actions"])
        ti.xcom_push(key="detected_ts_col", value=report["ts"])
        print(f"[REPORT] recommended_actions: {report['recommended_actions']}")
        print(f"[REPORT] recommended_actions: {actions}")
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
        print(f"[CLEAN] recommended actions: {actions}")
        
    
        # 0) Load raw CSV from MinIO
        df, fmt = load_df_from_minio(handle["key"])

        # =================== Apply cleaning actions ===================
        # 1) Basic cleaning actions for drop_duplicates, impute_missing, clip_outliers_iqr
        if "drop_duplicates" in actions:
            df = df.drop_duplicates()
            print(f"[CLEAN] dropped duplicates, new row count: {len(df)}")

        if "impute_missing" in actions:
            # num_cols = df.select_dtypes("number").columns           # recognize numeric columns
            # cat_cols = [c for c in df.columns if c not in num_cols] # recognize categorical columns
            # # basic imputation: median for numeric, mode for categorical
            # for c in num_cols:
            #     df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
            # for c in cat_cols:
            #     mode = df[c].mode()
            #     df[c] = df[c].fillna(mode.iloc[0] if not mode.empty else df[c].ffill().bfill())
            
            # AI-based imputation could be added here: placeholder for advanced methods

            # 1) Derive curated keys/prefix and prepare dataset root
            curated_data_key = build_curated_key(input_key=handle["key"], leaf="data.csv")
            curated_key_parent = curated_data_key[: curated_data_key.rfind("/") + 1]
            print(f"[CURATED] parent prefix: s3://{S3_BUCKET}/{curated_key_parent}")
            ti.xcom_push(key="curated_key_parent", value=curated_key_parent)

            # Make a dataset root that matches Preprocessor's expectation
            # IMPORTANT: DATASET_NAME must match the folder name (case-sensitive).
            curated_dataset_root = f"{curated_key_parent}"

            # 2a) write data.csv
            save_df_to_minio(df, f"{curated_dataset_root}data.csv", fmt="csv", index=False)
            print(f"[STORE] data.csv       → s3://{S3_BUCKET}/{curated_dataset_root}data.csv")
            ts_col_detected = ti.xcom_pull(task_ids="load_raw_data", key="detected_ts_col") or None

            curated_train_key, curated_test_key, curated_incomplete_key, df_incomplete = split_and_store_dataset(
                df,
                prefix=curated_dataset_root,
                test_size=0.2,
                time_col=ts_col_detected,  # or set your time column name
                stratify_col=None,  # or set your stratification column
                exclude_cols=[],  # optionally provide columns to exclude
                auto_exclude_path_like=True,
            )

            print(f"[PREP] Preprocessor initialized with curated root s3://{S3_BUCKET}/{curated_dataset_root}, {DATASET_NAME}")

            
            # B) Prepare + fit 
            # 1) Build the preprocessor from the curated prefix
            # data_dir = "s3://" + S3_BUCKET + "/" + curated_dataset_root
            # print(f'dataset name: {DATASET_NAME}, data dir: {data_dir}')
            # prepper = Preprocessor(DATASET_NAME, data_dir)

            # 1) Build the preprocessor from the curated prefix
            data_dir = "s3://" + S3_BUCKET + "/" + curated_dataset_root
            dataset_name = os.path.basename(handle["key"]).split(".")[0]
            print(f'dataset name: {dataset_name}, data dir: {data_dir}')
            prepper = Preprocessor(dataset_name, data_dir)


            # 2) Encode train/test in Ordinal layout: [ numeric | categorical-as-ordinal ]
            train_X = prepper.encodeDf("Ordinal", prepper.df_train)   # np.ndarray
            test_X  = prepper.encodeDf("Ordinal", prepper.df_test)    # np.ndarray
            num_numeric = prepper.numerical_indices_np_end            # split index for numeric slice

            # 3) Normalize (train stats only). Guard zeros in std
            mean_X = np.mean(train_X, axis=0)
            std_X  = np.std(train_X,  axis=0)
            std_X[std_X == 0] = 1.0

            X_train = (train_X - mean_X) / std_X
            X_test  = (test_X  - mean_X) / std_X

            # 4) Fit your imputer on normalized training features
            # imputer  = ReMasker(max_epochs=25, batch_size=128)   # tweak as you like
            # remasker = imputer.fit(torch.as_tensor(X_train.copy(), dtype=torch.float32))

            models_prefix = f"{curated_key_parent}saved_models/"
            # model
            # buf_model = save(remasker)  # get the model as bytes
            # _s3().put_object(
            #     Bucket=S3_BUCKET,
            #     Key=f"{models_prefix}remasker.pkl",
            #     Body=buf_model,
            #     ContentType="application/octet-stream",
            # )
            print(f"[STORE] model          → s3://{S3_BUCKET}/{models_prefix}remasker.pkl")
            print(train_X.shape, test_X.shape)
            print(np.isnan(train_X).sum(), np.isnan(test_X).sum())  # 应该为 0
            print(train_X[:5])


            # C) Evaluate on test set with synthetic missingness

            # 5) Build decoded ground-truth for test set
            # X_test_eval = X_test.copy()
            # # de-normalize numeric slice ONLY
            # X_test_eval[:, :num_numeric] = (
            #     X_test_eval[:, :num_numeric] * std_X[:num_numeric]
            # ) + mean_X[:num_numeric]
            # # decode back to original domain (nums + categorical strings)
            # X_test_eval = prepper.decodeNp("Ordinal", X_test_eval)

            # De-normalize NUMERIC slice for both truth and prediction
            X_test_eval = X_test.copy()
            X_test_eval[:, :num_numeric] = X_test_eval[:, :num_numeric] * std_X[:num_numeric] + mean_X[:num_numeric]
            X_test_eval = prepper.decodeNp('Ordinal', X_test_eval)
     
            print("[DEBUG] train_X stats", train_X.mean(), train_X.std())
            print("[DEBUG] test_X stats", test_X.mean(), test_X.std())
            print("[DEBUG] X_test_eval sample")
            print(X_test_eval[:2])

            


            # 6) Generate masks on the ORIGINAL (unencoded) test frame (True==MISSING)
            mask_type  = "MAR"    # or "MCAR" / "MNAR_logistic_T2"
            num_trials = 5
            ratio      = 0.25
            excluded   = prepper.info.get("excluded_path_like_cols", [])

            orig_masks = generate_mask(
                prepper.df_test,              # ✅ original DF
                mask_type=mask_type,
                mask_num=num_trials,
                p=ratio,
                exclude_cols=excluded,
                return_observed=False,        # True == MISSING
            )

            # 7) Convert those masks to the ENCODED (Ordinal) order (num+cat only)
            encoded_order = prepper.num_idx + prepper.cat_idx           # indices in original df.columns
            # Build a boolean indexer over original width for each mask, then slice columns in encoded order:
            masks_enc = np.stack([m[:, encoded_order] for m in orig_masks], axis=0)  # (T, N, D_encoded)

            # (Optional) sanity logs on the mask actually used downstream
            for t, m in enumerate(masks_enc):
                # "maskable" columns = any True in that column
                maskable_idx = np.where(np.any(m, axis=0))[0]
                missing_rate = m[:, maskable_idx].mean() if maskable_idx.size else 0.0
                print(f"[Mask {t}] {mask_type} actual missing rate on maskable columns: {missing_rate:.4f}")

            # 8) Score the imputer (normalize → impute → de-normalize numeric → decode → metrics)
            imputer = load_imputer_from_s3(S3_BUCKET, f"{models_prefix}remasker.pkl")  # reload to ensure no leakage

            MSEs, ACCs = [], []
            with torch.no_grad():
                for t in range(num_trials):
                    mask_enc = masks_enc[t]             # ✅ True==MISSING on ENCODED layout (num | cat_ordinal)
                    Xm = X_test.copy()                  # X_test is normalized encoded features
                    Xm[mask_enc] = np.nan              # mask in the SAME space/order

                    # impute in normalized space
                    Xm_t = torch.tensor(Xm, dtype=torch.float32)
                    imputed = imputer.transform(Xm_t).cpu().numpy()

                    # de-normalize numeric slice ONLY
                    imputed[:, :num_numeric] = (
                        imputed[:, :num_numeric] * std_X[:num_numeric]
                    ) + mean_X[:num_numeric]

                    # decode to original domain but only for encoded columns (nums + cats)
                    imputed_decoded = prepper.decodeNp("Ordinal", imputed)

                    # mse, acc = get_eval(imputed_decoded, X_test_eval, mask_test, num_numeric)
                    # Evaluate ONLY on the masked (missing) entries; mask must match encoded columns
                    mse, acc, nrmse = get_eval(
                        imputed_decoded,    # X_pred (decoded: nums + cat strings)
                        X_test_eval,        # X_true (decoded same way)
                        mask_enc,           # ✅ encoded-order mask
                        num_numeric
                    )
                    print(f"[EVAL] Trial {t} - ReMasker {mask_type} p={ratio}: MSE={mse:.6f}, ACC={acc:.4f}, {nrmse}")
                    MSEs.append(mse); ACCs.append(acc)

            avg_mse, std_mse = float(np.mean(MSEs)), float(np.std(MSEs))
            avg_acc, std_acc = float(np.mean(ACCs)), float(np.std(ACCs))
            print(f"[EVAL] ReMasker {mask_type} p={ratio}: MSE={avg_mse:.6f}±{std_mse:.6f}, ACC={avg_acc:.4f}±{std_acc:.4f}")


            # 4) Append eval summary to MinIO: curated/.../metrics/imputation.csv
            exp_key = f"{curated_key_parent}metrics/imputation.csv"
            try:
                obj = _s3().get_object(Bucket=S3_BUCKET, Key=exp_key)
                prev = pd.read_csv(io.BytesIO(obj["Body"].read()))
                if "Unnamed: 0" in prev.columns:
                    prev = prev.drop(columns=["Unnamed: 0"])
            except Exception:
                prev = pd.DataFrame(columns=["Dataset","Method","Mask Type","Ratio","Avg MSE","STD of MSE","Avg Acc","STD of Acc"])

            row = pd.DataFrame([{
                "Dataset": handle["key"],  # dataset name from path
                "Method": "Remasker",
                "Mask Type": mask_type,
                "Ratio": ratio,
                "Avg MSE": avg_mse,
                "STD of MSE": std_mse,
                "Avg Acc": avg_acc,
                "STD of Acc": std_acc,
            }])
            out_df = pd.concat([prev, row], ignore_index=True)
            buf = io.BytesIO(); out_df.to_csv(buf, index=False); buf.seek(0)
            _s3().put_object(Bucket=S3_BUCKET, Key=exp_key, Body=buf.getvalue(), ContentType="text/csv")
            print(f"[STORE] eval → s3://{S3_BUCKET}/{exp_key}")

            # before encoding incomplete
            if df_incomplete.empty:
                print("[IMPUTE] incomplete.csv has 0 rows; writing through unchanged.")
                imputed_inc_key = f"{curated_key_parent}imputed_incomplete.csv"
                save_df_to_minio(df_incomplete, imputed_inc_key, fmt="csv", index=False)
                # XComs (still push keys)
                ti.xcom_push(key="imputed_incomplete_key", value=imputed_inc_key)
                # you can return or just skip the transform block below
            else:
                # proceed to encode/normalize/transform/merge/save
                # 5) Impute curated incomplete.csv using the same normalization logic
                curated_incomplete_key = f"{curated_key_parent}incomplete.csv"
                if curated_incomplete_key is not None:
                    df_incomplete, _ = load_df_from_minio(curated_incomplete_key)

                    # encode incomplete to encoded (Ordinal)
                    X_incomplete = prepper.encodeDf("Ordinal", df_incomplete)
                    # normalize entire encoded vector
                    X_incomplete_norm = (X_incomplete - mean_X) / std_X
                    X_incomplete_norm_t = torch.tensor(X_incomplete_norm, dtype=torch.float32)

                    with torch.no_grad():
                        imputed_incomplete_norm = imputer.transform(X_incomplete_norm_t).cpu().numpy()

                    # denormalize NUMERIC block only
                    imputed_incomplete_norm[:, :num_numeric] = (
                        imputed_incomplete_norm[:, :num_numeric] * std_X[:num_numeric]
                    ) + mean_X[:num_numeric]

                    # decode to original domain
                    imputed_inc_decoded = prepper.decodeNp("Ordinal", imputed_incomplete_norm)
                    
                    # columns that were encoded (order matches the encoded layout)
                    enc_cols = [prepper.df.columns[i] for i in (prepper.num_idx + prepper.cat_idx)]

                    # build a partial DF using only encoded columns
                    imputed_part = pd.DataFrame(imputed_inc_decoded, columns=enc_cols)

                    # merge back into the original incomplete DF (preserve excluded columns as-is)
                    out_df = df_incomplete.copy()
                    out_df[enc_cols] = imputed_part[enc_cols]

                    imputed_inc_key = f"{curated_key_parent}imputed_incomplete.csv"
                    save_df_to_minio(out_df, imputed_inc_key, fmt="csv", index=False)
                    print(f"[STORE] imputed_incomplete.csv → s3://{S3_BUCKET}/{imputed_inc_key}")
            
           



            # XComs for downstream
            ti.xcom_push(key="curated_train", value=curated_train_key)
            ti.xcom_push(key="curated_test", value=curated_test_key)
            ti.xcom_push(key="curated_incomplete", value=curated_incomplete_key)
            ti.xcom_push(key="imputation_eval_key", value=exp_key)
            ti.xcom_push(key="imputed_incomplete_key", value=imputed_inc_key)
            ti.xcom_push(key="remasker_avg_mse", value=avg_mse)
            ti.xcom_push(key="remasker_avg_acc", value=avg_acc)



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

        # curated_key = f"{CURATED_PREFIX}/{DATASET_NAME}/{ts_tag}_clean.{ 'parquet' if out_fmt=='parquet' else 'csv'}"
        curated_key = build_curated_key(
            input_key=handle["key"],
            leaf="curated_data.csv",
            # run_ts=ts_tag,
            # suffix=f"clean.{ 'parquet' if out_fmt=='parquet' else 'csv'}"
        )
        
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
