# Test upload file to the bucket
import logging
from datetime import datetime
from airflow import DAG
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.operators.python import PythonOperator

# 1. Initialize the Task Logger
logger = logging.getLogger("airflow.task")

# Define constants to avoid typos
BUCKET_NAME = "airflow-bucket"
CONN_ID = "seaweed_s3"

def upload_to_datalake(**context):
    """
    Uploads a string to SeaweedFS with detailed logging and error tracing.
    """
    # Using 'context' allows you to log specific run details
    run_id = context.get('run_id')
    logger.info(f"Starting upload task for Run ID: {run_id}")

    try:
        logger.info(f"Connecting to SeaweedFS via connection ID: {CONN_ID}")
        s3 = S3Hook(aws_conn_id=CONN_ID)
        
        target_key = "test/hello_world.txt"
        data = f"Hello from Airflow to SeaweedFS! Run Time: {datetime.now()}"

        logger.info(f"Attempting to upload file to s3://{BUCKET_NAME}/{target_key}")
        
        s3.load_string(
            string_data=data,
            key=target_key,
            bucket_name=BUCKET_NAME,
            replace=True
        )
        
        logger.info("Upload successful!")

    except Exception as e:
        # This captures the full traceback and sends it to Airflow logs
        logger.error(f"Failed to upload to SeaweedFS.")
        logger.error(f"Error Details: {str(e)}")
        raise  # Re-raise the error so the task is marked as 'failed' in the UI

with DAG(
    dag_id="seaweedfs_datalake_test_v2",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=['debug', 'seaweedfs']
) as dag:

    # 1. Ensure the bucket exists
    # If this fails, it will log the reason automatically in the UI
    create_bucket = S3CreateBucketOperator(
        task_id="create_datalake_bucket",
        bucket_name=BUCKET_NAME,
        aws_conn_id=CONN_ID
    )

    # 2. Upload data
    upload_file = PythonOperator(
        task_id="upload_test_file",
        python_callable=upload_to_datalake,
        provide_context=True # Ensures the function gets the 'context' dictionary
    )

    create_bucket >> upload_file