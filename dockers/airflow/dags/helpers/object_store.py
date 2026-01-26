import json
from functools import lru_cache
from io import BytesIO
from typing import Tuple

import pandas as pd
from airflow.models import Variable
from botocore.client import Config
from botocore.exceptions import ClientError
import boto3

# ---------------------------------------------------------------------
# Configuration (Airflow Variables preferred)
# ---------------------------------------------------------------------

S3_ENDPOINT = Variable.get("N2N_S3_ENDPOINT", default_var="http://seaweed-s3:8333")
S3_ACCESS_KEY = Variable.get("N2N_S3_ACCESS_KEY", default_var="anykey")
S3_SECRET_KEY = Variable.get("N2N_S3_SECRET_KEY", default_var="anysecret")
S3_BUCKET = Variable.get("S3_BUCKET", default_var="airflow-bucket")

DEFAULT_REGION = "us-east-1"
# ---------------------------------------------------------------------
# S3 Client (SeaweedFS-compatible)
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_s3_client():
    """
    SeaweedFS-compatible S3 client (path-style addressing).
    Cached to avoid re-creation in Airflow workers.
    """
    if not S3_ENDPOINT:
        raise RuntimeError("Airflow Variable N2N_S3_ENDPOINT is required")

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=DEFAULT_REGION,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},  # ⭐ REQUIRED for SeaweedFS
        ),
    )

# ---------------------------------------------------------------------
# Object existence (SeaweedFS-safe)
# ---------------------------------------------------------------------

def key_exists(bucket: str, key: str) -> bool:
    """
    Robust object existence check for SeaweedFS.
    """
    s3 = get_s3_client()
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey"}:
            return False
        if code in {"403", "AccessDenied"}:
            # SeaweedFS sometimes returns 403 for existing objects
            try:
                s3.get_object(Bucket=bucket, Key=key, Range="bytes=0-0")
                return True
            except Exception:
                return False
        raise

# ---------------------------------------------------------------------
# DataFrame IO
# ---------------------------------------------------------------------
def load_df_from_object_store(key: str, bucket: str = S3_BUCKET) -> Tuple[pd.DataFrame, str]:
    s3 = get_s3_client()
    # 彻底清理 Key
    clean_key = key.strip().lstrip('/')
    
    print(f"Attempting to read: s3://{bucket}/{clean_key}")

    try:
        obj = s3.get_object(Bucket=bucket, Key=clean_key)
        body = obj["Body"].read()
        
        if clean_key.endswith(".csv"):
            return pd.read_csv(BytesIO(body)), "csv"
        if clean_key.endswith(".parquet"):
            return pd.read_parquet(BytesIO(body)), "parquet"
            
        raise ValueError(f"Unsupported format: {clean_key}")
        
    except s3.exceptions.NoSuchKey:
        # 如果报错，打印出当前桶里的前 5 个 Key 帮助调试
        print(f"Error: NoSuchKey. Available keys in {bucket} starting with '{clean_key[:4]}':")
        res = s3.list_objects_v2(Bucket=bucket, MaxKeys=5)
        for o in res.get('Contents', []):
            print(f" - Found existing key: {o['Key']}")
        raise FileNotFoundError(f"Object not found: s3://{bucket}/{clean_key}")


def save_df_to_object_store(
    df: pd.DataFrame,
    key: str,
    bucket: str = S3_BUCKET,
    fmt: str = "csv",
    overwrite: bool = True,
):
    """
    Save DataFrame to SeaweedFS.
    """
    s3 = get_s3_client()

    if not overwrite and key_exists(bucket, key):
        raise FileExistsError(f"s3://{bucket}/{key} already exists")

    if fmt == "csv":
        data = df.to_csv(index=False).encode("utf-8")
        content_type = "text/csv"
    elif fmt == "parquet":
        buf = BytesIO()
        df.to_parquet(buf, index=False)
        data = buf.getvalue()
        content_type = "application/octet-stream"
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )

# ---------------------------------------------------------------------
# JSON helpers (GX / DQC reports)
# ---------------------------------------------------------------------

def save_json(
    obj: dict,
    key: str,
    bucket: str = S3_BUCKET,
):
    s3 = get_s3_client()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

def load_json(
    key: str,
    bucket: str = S3_BUCKET,
) -> dict:
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())
