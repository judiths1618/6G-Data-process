
# from __future__ import annotations

# from datetime import datetime
# import os

# from airflow import DAG
# from airflow.models import Variable
# from airflow.operators.bash import BashOperator

# # ----------
# # Variables (override in Airflow UI > Admin > Variables)
# # ----------
# # S3 / MinIO
# S3_ENDPOINT_URL = Variable.get("DQ_S3_ENDPOINT_URL", default_var=os.getenv("S3_ENDPOINT_URL", "http://minio:9000"))
# AWS_ACCESS_KEY_ID = Variable.get("DQ_AWS_ACCESS_KEY_ID", default_var=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"))
# AWS_SECRET_ACCESS_KEY = Variable.get("DQ_AWS_SECRET_ACCESS_KEY", default_var=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"))
# AWS_REGION = Variable.get("DQ_AWS_REGION", default_var=os.getenv("AWS_REGION", "us-east-1"))

# # IO patterns (can be local or s3://)
# DQ_INPUT_PATTERN = Variable.get(
#     "DQ_INPUT_PATTERN",
#     default_var="s3://6gdali-lake2025/DeepSense/Scenario33/*.csv",
# )
# DQ_OUTPUT_ROOT = Variable.get(
#     "DQ_OUTPUT_ROOT",
#     default_var="s3://6gdali-lake2025/gold/dq_reports/scenario33",
# )

# # Presigned URL expiry (seconds) when writing to S3/MinIO
# DQ_PRESIGN_EXPIRES = int(Variable.get("DQ_PRESIGN_EXPIRES", default_var="86400"))  # 24h

# # Engine for dq_local_beam: "sequential" (recommended) | "beam" | "auto"
# DQ_ENGINE = Variable.get("DQ_ENGINE", default_var="sequential")

# # Path to the dashboard script inside the scheduler/worker container
# DASHBOARD_SCRIPT = Variable.get(
#     "DQ_DASHBOARD_SCRIPT",
#     default_var="/opt/airflow/dags/quality_dashboard.py",
# )

# with DAG(
#     dag_id="dq_run_dashboard_s3",
#     start_date=datetime(2025, 1, 1),
#     schedule_interval=None,
#     catchup=False,
#     tags=["dq", "s3", "minio", "auto-rules"],
# ) as dag:

#     # We pass --config AUTO to enable the "no rules" flow.
#     dq_task = BashOperator(
#         task_id="run_dq_dashboard_s3",
#         bash_command=(
#             "set -euo pipefail\n"
#             "python {{ var.value.get('DQ_DASHBOARD_SCRIPT', '" + DASHBOARD_SCRIPT + "') }} "
#             "--input_pattern \"{{ var.value.get('DQ_INPUT_PATTERN', '" + DQ_INPUT_PATTERN + "') }}\" "
#             "--config \"AUTO\" "
#             "--output_root \"{{ var.value.get('DQ_OUTPUT_ROOT', '" + DQ_OUTPUT_ROOT + "') }}\" "
#             "--engine \"{{ var.value.get('DQ_ENGINE', '" + DQ_ENGINE + "') }}\" "
#             "--open-browser \"false\" "
#             "--expires {{ var.value.get('DQ_PRESIGN_EXPIRES', '" + str(DQ_PRESIGN_EXPIRES) + "') }} "
#         ),
#         env={
#             "S3_ENDPOINT_URL": S3_ENDPOINT_URL,
#             "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
#             "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
#             "AWS_REGION": AWS_REGION,
#         },
#     )

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="dq_run_dashboard_s3",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["dq", "s3"],
) as dag:

    dq_task = BashOperator(
        task_id="run_dq_dashboard_s3",
        bash_command=r'''
set -euo pipefail
# 运行并抓取脚本打印的预签名链接（形如：[DQ REPORT URL] https://...）
OUT=$(python /opt/airflow/dags/quality_dashboard.py \
  --input_pattern "s3://6gdali-lake2025/DeepSense/Scenario33/*.csv" \
  --config "AUTO" \
  --output_root "s3://6gdali-lake2025/gold/dq_reports/scenario33" \
  --engine "sequential" \
  --open-browser "false" \
  --expires 86400 \
  --embed-viz "true" \
  --iframe-height 560 \
  --time-column time_stamp \
#   --ts-limit 4000
  )

echo "$OUT" >&2  # 全量日志保留到 Log
# 只把 URL 回显到 stdout 作为 XCom
echo "$OUT" | sed -n 's/^\[DQ REPORT URL\] //p' | tail -n1
''',
        do_xcom_push=True,   # 让 stdout 进入 XCom
    )
