import boto3
import os
from botocore.client import Config

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:8333",
    aws_access_key_id="anykey",
    aws_secret_access_key="anysecret",
    region_name="us-east-1",
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"}
    ),
)

def download_folder(bucket: str, prefix: str, local_dir: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(prefix):].lstrip("/")
            local_path = os.path.join(local_dir, relative)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)
            print(f"↓ {key}")

# Usage
SeawwedFS_folders =  [
            # "test/",
            "wavestitchplus/",
            # "interim/"
            "cleaned/",
            "curated/"
            ]  
       
for prefix_item in SeawwedFS_folders: 
    download_folder(
        bucket="airflow-bucket",
        prefix=prefix_item,
        local_dir=f"/home/Yuandou/Desktop/projects/6G-Data-process/notebooks/EUR/{prefix_item}",
    )