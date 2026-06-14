"""Airflow DAG for the minimal DataOps pipeline.

Set these env vars in Airflow when paths differ from the repo defaults:
DATAOPS_INPUT_CSV, DATAOPS_OUTPUT_CSV, DATAOPS_REPORT_JSON, DATAOPS_TIMESTAMP_COL.
"""
from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from pipelines.minimal_dataops import configure_logging, notify_failure, run_from_config


def _notify_airflow_failure(context) -> None:
    task = context.get("task_instance")
    exc = context.get("exception")
    task_id = getattr(task, "task_id", "unknown")
    notify_failure(f"Airflow DAG minimal_dataops task {task_id} failed: {exc}")


def _run_minimal_dataops() -> None:
    configure_logging(os.environ.get("DATAOPS_LOG_FILE", "logs/dataops.log"))
    run_from_config(
        os.environ.get("DATAOPS_CONFIG", "config/dataops.yaml"),
        input=os.environ.get("DATAOPS_INPUT_CSV"),
        output=os.environ.get("DATAOPS_OUTPUT_CSV"),
        report=os.environ.get("DATAOPS_REPORT_JSON"),
        log_file=os.environ.get("DATAOPS_LOG_FILE"),
        timestamp_col=os.environ.get("DATAOPS_TIMESTAMP_COL"),
    )


with DAG(
    dag_id="minimal_dataops",
    description="Clean, validate, profile, and write minimal DataOps artifacts.",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    on_failure_callback=_notify_airflow_failure,
    tags=["dataops", "minimal"],
) as dag:
    PythonOperator(
        task_id="clean_validate_profile",
        python_callable=_run_minimal_dataops,
    )
