"""
io_utils — optional thin local|S3 adapter for the CLI.

The library modules never import this; they only see DataFrames. The CLI uses
these helpers to read inputs and persist outputs from either the local
filesystem or an ``s3://bucket/key`` reference, mirroring the env-var contract
used elsewhere in the repo (``S3_ENDPOINT/S3_ACCESS_KEY/S3_SECRET_KEY``).
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Tuple

import pandas as pd

__all__ = [
    "is_s3", "split_s3", "read_csv", "write_csv",
    "read_json", "write_json", "write_npy",
]


def is_s3(ref: str) -> bool:
    return str(ref).startswith("s3://")


def split_s3(ref: str) -> Tuple[str, str]:
    """``s3://bucket/some/key`` -> ``("bucket", "some/key")``."""
    rest = ref[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def _client():
    import boto3
    from botocore.client import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def read_csv(ref: str) -> pd.DataFrame:
    if is_s3(ref):
        bucket, key = split_s3(ref)
        body = _client().get_object(Bucket=bucket, Key=key)["Body"].read()
        return pd.read_csv(io.BytesIO(body))
    return pd.read_csv(ref)


def write_csv(df: pd.DataFrame, ref: str) -> None:
    if is_s3(ref):
        bucket, key = split_s3(ref)
        _client().put_object(Bucket=bucket, Key=key, Body=df.to_csv(index=False).encode())
    else:
        Path(ref).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ref, index=False)


def read_json(ref: str) -> dict:
    if is_s3(ref):
        bucket, key = split_s3(ref)
        body = _client().get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    with open(ref) as f:
        return json.load(f)


def write_json(obj, ref: str) -> None:
    payload = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    if is_s3(ref):
        bucket, key = split_s3(ref)
        _client().put_object(Bucket=bucket, Key=key, Body=payload.encode())
    else:
        Path(ref).parent.mkdir(parents=True, exist_ok=True)
        with open(ref, "w") as f:
            f.write(payload)


def write_npy(arr, ref: str) -> None:
    import numpy as np

    if is_s3(ref):
        bucket, key = split_s3(ref)
        buf = io.BytesIO()
        np.save(buf, arr)
        _client().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    else:
        Path(ref).parent.mkdir(parents=True, exist_ok=True)
        np.save(ref, arr)
