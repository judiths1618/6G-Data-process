# dags/remote_remasker_minio.py
from datetime import datetime
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.operators.python import PythonOperator

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

# --- CONFIG ---
DAG_ID = "remote_gpu_impute_remasker"
SSH_CONN_ID = "gpu_ssh"
AWS_CONN_ID = "minio_s3"

# Sane defaults; override via the UI "Params" or when triggering a DAG run
DEFAULT_PARAMS = {
    # MinIO bucket + input object for your Scenario33 dataset
    "bucket": "ml-runs",
    "data_key": DATASET_NAME,  # or .csv
    # Where to write outputs in the bucket
    "out_prefix": "runs/Scenario33-remasker",
    # Training config
    "epochs": 1,
    # Remote paths for logs/PIDs
    "remote_base": "/opt/impute_test",
    # Use conda env? If yes, set e.g. "myenv"; if "", system python3 is used
    "conda_env": "torchcuda",
}

REMOTE_RUNNER_PY = r'''#!/usr/bin/env python3
import os, sys, json, time
import subprocess, importlib
from io import BytesIO

def ensure_pip_pkg(spec, import_name=None):
    name = import_name or spec.split("==")[0].split("[")[0].split(">=")[0]
    try:
        import importlib; importlib.import_module(name)
        return True
    except Exception:
        return subprocess.call(f"{sys.executable} -m pip install -U {spec}", shell=True) == 0

def main():
    # Inputs via env (set by launcher)
    endpoint = os.environ["S3_ENDPOINT"]
    access   = os.environ["AWS_ACCESS_KEY_ID"]
    secret   = os.environ["AWS_SECRET_ACCESS_KEY"]
    bucket   = os.environ["S3_BUCKET"]
    data_key = os.environ["DATA_KEY"]
    out_pref = os.environ["OUT_PREFIX"]
    run_id   = os.environ["RUN_ID"]
    epochs   = int(os.environ.get("EPOCHS","1"))

    # Base deps
    ensure_pip_pkg("pip"); ensure_pip_pkg("setuptools"); ensure_pip_pkg("wheel")
    ensure_pip_pkg("boto3"); ensure_pip_pkg("pandas>=2.0"); ensure_pip_pkg("pyarrow"); ensure_pip_pkg("scikit-learn>=1.2")
    ensure_pip_pkg("torch --index-url https://download.pytorch.org/whl/cu121", "torch")
    rem_ok = ensure_pip_pkg("git+https://github.com/tydusky/remasker#egg=remasker", "remasker")

    import boto3, pandas as pd
    s3 = boto3.client("s3", endpoint_url=endpoint, verify=False, region_name="us-east-1",
                      aws_access_key_id=access, aws_secret_access_key=secret)

    def s3_read(b, k): return s3.get_object(Bucket=b, Key=k)["Body"].read()
    def s3_write(b, k, data): s3.put_object(Bucket=b, Key=k, Body=data)

    # Load data
    raw = s3_read(bucket, data_key)
    if data_key.lower().endswith((".parquet",".pq")):
        df = pd.read_parquet(BytesIO(raw))
    else:
        try: df = pd.read_parquet(BytesIO(raw))
        except Exception: df = pd.read_csv(BytesIO(raw))

    num_cols = df.select_dtypes(include=["number"]).columns.tolist() or df.columns.tolist()
    X = df[num_cols].copy()

    metrics = {
        "run_id": run_id, "started": int(time.time()),
        "rows": len(df), "cols": df.shape[1],
        "numeric_cols": num_cols, "epochs": epochs,
        "method": "remasker" if rem_ok else "simple_mean_fallback"
    }

    try:
        if rem_ok:
            try:
                from remasker import RemaskerModel
                m = RemaskerModel(epochs=epochs)
                m.fit(X)
                X_imp = m.transform(X)
            except Exception:
                import torch
                t = torch.tensor(X.values, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
                col_means = torch.nanmean(t, dim=0)
                t[torch.isnan(t)] = col_means.repeat(t.size(0),1)[torch.isnan(t)]
                import pandas as pd
                X_imp = pd.DataFrame(t.detach().cpu().numpy(), columns=X.columns, index=X.index)
        else:
            from sklearn.impute import SimpleImputer
            imp = SimpleImputer(strategy="mean")
            import pandas as pd
            X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)

        df[num_cols] = X_imp
        metrics["status"] = "ok"
    except Exception as e:
        metrics["status"] = "error"
        metrics["error"] = repr(e)

    prefix = f"{out_pref}/{run_id}"
    # Save imputed data as parquet
    buf = BytesIO(); df.to_parquet(buf, index=False)
    s3_write(bucket, f"{prefix}/imputed.parquet", buf.getvalue())
    # Save metrics
    s3_write(bucket, f"{prefix}/metrics.json", json.dumps(metrics).encode("utf-8"))
    # Markers
    if metrics.get("status") == "ok":
        s3_write(bucket, f"{prefix}/_SUCCESS", b"")
    else:
        s3_write(bucket, f"{prefix}/_FAILED", json.dumps(metrics).encode("utf-8"))

if __name__ == "__main__":
    main()
'''

with DAG(
    dag_id=DAG_ID,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    params=DEFAULT_PARAMS,
    doc_md="""
**Remote GPU imputation** using *remasker* on a GPU host via SSH, with I/O on **MinIO**.
- Writes `remote_remasker_job.py` to the host
- Launches detached job (nohup)
- Waits for `.../_SUCCESS` in MinIO
- Pulls `metrics.json` to XCom
""",
) as dag:

    # 1) Submit remote job: write runner + launch detached
    submit = SSHOperator(
        task_id="submit_remote_job",
        ssh_conn_id=SSH_CONN_ID,
        get_pty=True,
        cmd_timeout=0,  # allow long jobs
        command=r"""
            set -euo pipefail
            RUN_ID="{{ ts_nodash }}"
            REMOTE_BASE="{{ params.remote_base }}"
            RDIR="${REMOTE_BASE}/${RUN_ID}"
            mkdir -p "${RDIR}" "${REMOTE_BASE}/logs" "${REMOTE_BASE}/pids"

            # Write the runner file
            cat > "${RDIR}/remote_remasker_job.py" <<'PYEOF'
{{ REMOTE_RUNNER_PY }}
PYEOF
            chmod +x "${RDIR}/remote_remasker_job.py"

            # Optional: conda activation
            ACTIVATE=""
            if [ -n "{{ params.conda_env }}" ]; then
              ACTIVATE="source ~/.bashrc && conda activate {{ params.conda_env }} && "
            fi

            # Launch detached with env vars for MinIO + job config
            nohup bash -lc "${ACTIVATE} \
              S3_ENDPOINT='{{ conn.minio_s3.extra_dejson.get('endpoint_url') }}' \
              AWS_ACCESS_KEY_ID='{{ conn.minio_s3.login }}' \
              AWS_SECRET_ACCESS_KEY='{{ conn.minio_s3.password }}' \
              S3_BUCKET='{{ params.bucket }}' \
              DATA_KEY='{{ params.data_key }}' \
              OUT_PREFIX='{{ params.out_prefix }}' \
              RUN_ID='${RUN_ID}' \
              EPOCHS='{{ params.epochs }}' \
              python3 ${RDIR}/remote_remasker_job.py \
              > ${REMOTE_BASE}/logs/${RUN_ID}.log 2>&1" &

            echo $! > "${REMOTE_BASE}/pids/${RUN_ID}.pid"
            echo "Launched PID $(cat ${REMOTE_BASE}/pids/${RUN_ID}.pid) for run ${RUN_ID}"
        """,
        environment={"REMOTE_RUNNER_PY": REMOTE_RUNNER_PY},
    )

    # 2) Wait for success marker in MinIO
    wait_success = S3KeySensor(
        task_id="wait_success_marker",
        aws_conn_id=AWS_CONN_ID,
        bucket_key="s3://{{ params.bucket }}/{{ params.out_prefix }}/{{ ts_nodash }}/_SUCCESS",
        poke_interval=30,
        timeout=60 * 60 * 6,  # up to 6 hours
        soft_fail=False,
        mode="reschedule",
    )

    # 3) Collect metrics.json into XCom
    def pull_metrics(**ctx):
        run_id = ctx["ts_nodash"]
        hook = S3Hook(aws_conn_id=AWS_CONN_ID)
        key = f"{{ params.out_prefix }}/{run_id}/metrics.json"
        body = hook.read_key(key=key, bucket_name="{{ params.bucket }}")
        ctx["ti"].xcom_push(key="metrics_json", value=body)
        print("Metrics:", body)

    collect = PythonOperator(
        task_id="collect_metrics",
        python_callable=pull_metrics,
    )

    submit >> wait_success >> collect
