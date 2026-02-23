"""
helpers/object_store.py - 扩展版
支持 DataFrame IO + 文件夹上传下载
"""

import json
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from datetime import datetime

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
            s3={"addressing_style": "path"},
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
        
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey"}:
            print(f"Error: NoSuchKey. Available keys in {bucket}:")
            res = s3.list_objects_v2(Bucket=bucket, MaxKeys=10)
            for o in res.get('Contents', []):
                print(f" - {o['Key']}")
            raise FileNotFoundError(f"Object not found: s3://{bucket}/{clean_key}")
        raise


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
    clean_key = key.strip().lstrip('/')

    if not overwrite and key_exists(bucket, clean_key):
        raise FileExistsError(f"s3://{bucket}/{clean_key} already exists")

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
        Key=clean_key,
        Body=data,
        ContentType=content_type,
    )
    
    print(f"Saved: s3://{bucket}/{clean_key}")


# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------

def save_json(
    obj: dict,
    key: str,
    bucket: str = S3_BUCKET,
):
    s3 = get_s3_client()
    clean_key = key.strip().lstrip('/')
    
    s3.put_object(
        Bucket=bucket,
        Key=clean_key,
        Body=json.dumps(obj, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )


def load_json(
    key: str,
    bucket: str = S3_BUCKET,
) -> dict:
    s3 = get_s3_client()
    clean_key = key.strip().lstrip('/')
    
    obj = s3.get_object(Bucket=bucket, Key=clean_key)
    return json.loads(obj["Body"].read())


# ---------------------------------------------------------------------
# 🆕 文件夹上传/下载 (新增)
# ---------------------------------------------------------------------

def upload_file_to_s3(
    local_path: str,
    key: str,
    bucket: str = S3_BUCKET,
    content_type: str = None,
) -> Dict:
    """
    上传单个文件到 S3
    """
    s3 = get_s3_client()
    clean_key = key.strip().lstrip('/')
    
    # 自动检测 content type
    if content_type is None:
        ext = os.path.splitext(local_path)[1].lower()
        content_type_map = {
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.txt': 'text/plain',
            '.log': 'text/plain',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.pdf': 'application/pdf',
            '.parquet': 'application/octet-stream',
            '.npy': 'application/octet-stream',
            '.pth': 'application/octet-stream',
            '.pt': 'application/octet-stream',
        }
        content_type = content_type_map.get(ext, 'application/octet-stream')
    
    file_size = os.path.getsize(local_path)
    
    with open(local_path, 'rb') as f:
        s3.put_object(
            Bucket=bucket,
            Key=clean_key,
            Body=f,
            ContentType=content_type,
        )
    
    return {
        "local": local_path,
        "s3_key": clean_key,
        "size": file_size,
        "content_type": content_type,
    }


def upload_directory_to_s3(
    local_dir: str,
    s3_prefix: str,
    bucket: str = S3_BUCKET,
    exclude_patterns: List[str] = None,
    include_patterns: List[str] = None,
) -> Dict:
    """
    递归上传整个文件夹到 S3
    
    Args:
        local_dir: 本地目录路径
        s3_prefix: S3 前缀（目录路径）
        bucket: S3 bucket 名称
        exclude_patterns: 排除的文件/目录模式
        include_patterns: 只包含的文件模式（如果指定）
    
    Returns:
        上传结果摘要
    """
    if exclude_patterns is None:
        exclude_patterns = [
            '__pycache__',
            '.pyc',
            '.git',
            '.DS_Store',
            '.gitignore',
            '*.tmp',
            '*.temp',
        ]
    
    s3_prefix = s3_prefix.strip().lstrip('/')
    local_path = Path(local_dir)
    
    if not local_path.exists():
        raise FileNotFoundError(f"Directory not found: {local_dir}")
    
    uploaded_files = []
    failed_files = []
    total_size = 0
    
    print(f"[UPLOAD] Starting upload: {local_dir} -> s3://{bucket}/{s3_prefix}/")
    
    for file_path in local_path.rglob('*'):
        if not file_path.is_file():
            continue
        
        # 检查排除模式
        relative_str = str(file_path.relative_to(local_path))
        skip = False
        
        for pattern in exclude_patterns:
            if pattern.startswith('*'):
                if relative_str.endswith(pattern[1:]):
                    skip = True
                    break
            elif pattern in relative_str:
                skip = True
                break
        
        if skip:
            continue
        
        # 检查包含模式
        if include_patterns:
            include = False
            for pattern in include_patterns:
                if pattern.startswith('*'):
                    if relative_str.endswith(pattern[1:]):
                        include = True
                        break
                elif pattern in relative_str:
                    include = True
                    break
            if not include:
                continue
        
        # 计算 S3 key
        relative_path = file_path.relative_to(local_path)
        s3_key = f"{s3_prefix}/{relative_path}".replace("\\", "/")
        
        # 上传
        try:
            result = upload_file_to_s3(
                local_path=str(file_path),
                key=s3_key,
                bucket=bucket,
            )
            uploaded_files.append(result)
            total_size += result["size"]
            print(f"  ✓ {relative_path} ({result['size']} bytes)")
        except Exception as e:
            failed_files.append({
                "local": str(file_path),
                "error": str(e),
            })
            print(f"  ✗ {relative_path}: {e}")
    
    summary = {
        "status": "success" if not failed_files else "partial",
        "source": str(local_dir),
        "destination": f"s3://{bucket}/{s3_prefix}/",
        "files_uploaded": len(uploaded_files),
        "files_failed": len(failed_files),
        "total_size": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "uploaded": uploaded_files,
        "failed": failed_files,
    }
    
    print(f"[UPLOAD] Complete: {len(uploaded_files)} files, {summary['total_size_mb']} MB")
    
    return summary


def download_file_from_s3(
    key: str,
    local_path: str,
    bucket: str = S3_BUCKET,
) -> Dict:
    """
    从 S3 下载单个文件
    """
    s3 = get_s3_client()
    clean_key = key.strip().lstrip('/')
    
    # 创建目录
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    obj = s3.get_object(Bucket=bucket, Key=clean_key)
    
    with open(local_path, 'wb') as f:
        f.write(obj['Body'].read())
    
    file_size = os.path.getsize(local_path)
    
    return {
        "s3_key": clean_key,
        "local": local_path,
        "size": file_size,
    }


def download_directory_from_s3(
    s3_prefix: str,
    local_dir: str,
    bucket: str = S3_BUCKET,
    include_patterns: List[str] = None,
) -> Dict:
    """
    从 S3 下载整个目录
    
    Args:
        s3_prefix: S3 前缀（目录路径）
        local_dir: 本地目录路径
        bucket: S3 bucket 名称
        include_patterns: 只包含的文件模式
    
    Returns:
        下载结果摘要
    """
    s3 = get_s3_client()
    s3_prefix = s3_prefix.strip().lstrip('/').rstrip('/') + '/'
    
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[DOWNLOAD] Starting download: s3://{bucket}/{s3_prefix} -> {local_dir}")
    
    # 列出所有文件
    downloaded_files = []
    failed_files = []
    total_size = 0
    
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            
            # 跳过目录标记
            if key.endswith('/'):
                continue
            
            # 计算相对路径
            relative_path = key[len(s3_prefix):]
            if not relative_path:
                continue
            
            # 检查包含模式
            if include_patterns:
                include = False
                for pattern in include_patterns:
                    if pattern.startswith('*'):
                        if relative_path.endswith(pattern[1:]):
                            include = True
                            break
                    elif pattern in relative_path:
                        include = True
                        break
                if not include:
                    continue
            
            # 本地文件路径
            local_file = local_path / relative_path
            
            try:
                result = download_file_from_s3(
                    key=key,
                    local_path=str(local_file),
                    bucket=bucket,
                )
                downloaded_files.append(result)
                total_size += result["size"]
                print(f"  ✓ {relative_path} ({result['size']} bytes)")
            except Exception as e:
                failed_files.append({
                    "s3_key": key,
                    "error": str(e),
                })
                print(f"  ✗ {relative_path}: {e}")
    
    summary = {
        "status": "success" if not failed_files else "partial",
        "source": f"s3://{bucket}/{s3_prefix}",
        "destination": str(local_dir),
        "files_downloaded": len(downloaded_files),
        "files_failed": len(failed_files),
        "total_size": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "downloaded": downloaded_files,
        "failed": failed_files,
    }
    
    print(f"[DOWNLOAD] Complete: {len(downloaded_files)} files, {summary['total_size_mb']} MB")
    
    return summary


def list_s3_directory(
    s3_prefix: str,
    bucket: str = S3_BUCKET,
    recursive: bool = True,
) -> List[Dict]:
    """
    列出 S3 目录内容
    """
    s3 = get_s3_client()
    s3_prefix = s3_prefix.strip().lstrip('/').rstrip('/') + '/'
    
    files = []
    
    if recursive:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
            for obj in page.get('Contents', []):
                if not obj['Key'].endswith('/'):
                    files.append({
                        "key": obj['Key'],
                        "size": obj['Size'],
                        "last_modified": obj['LastModified'].isoformat(),
                    })
    else:
        response = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=s3_prefix,
            Delimiter='/',
        )
        
        # 文件
        for obj in response.get('Contents', []):
            if obj['Key'] != s3_prefix:
                files.append({
                    "key": obj['Key'],
                    "size": obj['Size'],
                    "type": "file",
                })
        
        # 子目录
        for prefix in response.get('CommonPrefixes', []):
            files.append({
                "key": prefix['Prefix'],
                "type": "directory",
            })
    
    return files


def delete_s3_directory(
    s3_prefix: str,
    bucket: str = S3_BUCKET,
) -> Dict:
    """
    删除 S3 目录（包括所有内容）
    """
    s3 = get_s3_client()
    s3_prefix = s3_prefix.strip().lstrip('/').rstrip('/') + '/'
    
    deleted = []
    
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        objects = page.get('Contents', [])
        if not objects:
            continue
        
        delete_keys = [{'Key': obj['Key']} for obj in objects]
        
        response = s3.delete_objects(
            Bucket=bucket,
            Delete={'Objects': delete_keys},
        )
        
        deleted.extend([d['Key'] for d in response.get('Deleted', [])])
    
    return {
        "status": "success",
        "prefix": s3_prefix,
        "files_deleted": len(deleted),
    }


# ---------------------------------------------------------------------
# 🆕 WaveStitch 专用工具函数
# ---------------------------------------------------------------------

def create_wavestitch_manifest(
    work_dir: str,
    metadata: Dict,
) -> Dict:
    """
    创建 WaveStitch 工作目录的清单文件
    """
    manifest = {
        "created_at": datetime.utcnow().isoformat(),
        "wavestitch_version": "1.0",
        "metadata": metadata,
        "files": [],
        "directories": [],
    }
    
    work_path = Path(work_dir)
    
    for item in work_path.rglob('*'):
        relative = str(item.relative_to(work_path))
        
        if item.is_file():
            manifest["files"].append({
                "path": relative,
                "size": item.stat().st_size,
                "extension": item.suffix,
            })
        elif item.is_dir():
            manifest["directories"].append(relative)
    
    # 统计
    manifest["summary"] = {
        "total_files": len(manifest["files"]),
        "total_directories": len(manifest["directories"]),
        "total_size": sum(f["size"] for f in manifest["files"]),
        "total_size_mb": round(sum(f["size"] for f in manifest["files"]) / (1024 * 1024), 2),
        "file_types": {},
    }
    
    for f in manifest["files"]:
        ext = f["extension"] or "no_extension"
        manifest["summary"]["file_types"][ext] = manifest["summary"]["file_types"].get(ext, 0) + 1
    
    # 保存清单
    manifest_path = work_path / "MANIFEST.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    return manifest


def upload_wavestitch_results(
    work_dir: str,
    source_info: Dict,
    run_id: str,
    bucket: str = S3_BUCKET,
    datalake_prefix: str = "datalake/wavestitch",
) -> Dict:
    """
    上传 WaveStitch 完整结果到 Data Lake
    
    Args:
        work_dir: WaveStitch 工作目录
        source_info: 原始数据源信息 {"bucket": ..., "key": ...}
        run_id: Airflow run_id
        bucket: 目标 bucket
        datalake_prefix: Data Lake 路径前缀
    
    Returns:
        上传结果摘要
    """
    # 生成存储路径
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    original_filename = os.path.basename(source_info.get("key", "unknown"))
    original_name = os.path.splitext(original_filename)[0]
    
    s3_base_prefix = f"{datalake_prefix}/{original_name}/{timestamp}_{run_id}"
    
    # 创建清单
    manifest_metadata = {
        "run_id": run_id,
        "source": source_info,
        "timestamp": timestamp,
    }
    
    manifest = create_wavestitch_manifest(work_dir, manifest_metadata)
    
    # 上传整个目录
    upload_result = upload_directory_to_s3(
        local_dir=work_dir,
        s3_prefix=s3_base_prefix,
        bucket=bucket,
    )
    
    # 查找并单独上传最终结果到快速访问位置
    quick_access_key = None
    generated_dir = os.path.join(work_dir, "generated")
    
    if os.path.exists(generated_dir):
        # 查找 final_imputed.csv 或类似文件
        for name in ["final_imputed.csv", "full_imputed_cleaned.csv", "full_imputed.csv"]:
            final_csv = os.path.join(generated_dir, name)
            if os.path.exists(final_csv):
                quick_access_key = f"{datalake_prefix}/results/{original_name}_imputed_{timestamp}.csv"
                upload_file_to_s3(
                    local_path=final_csv,
                    key=quick_access_key,
                    bucket=bucket,
                )
                print(f"[QUICK ACCESS] s3://{bucket}/{quick_access_key}")
                break
    
    result = {
        "status": upload_result["status"],
        "source": source_info,
        "destination": {
            "bucket": bucket,
            "prefix": s3_base_prefix,
            "full_uri": f"s3://{bucket}/{s3_base_prefix}/",
        },
        "quick_access": {
            "bucket": bucket,
            "key": quick_access_key,
            "uri": f"s3://{bucket}/{quick_access_key}" if quick_access_key else None,
        },
        "upload_summary": upload_result,
        "manifest": manifest["summary"],
    }
    
    return result


def download_wavestitch_model(
    model_uri: str,
    local_dir: str,
) -> Dict:
    """
    从 Data Lake 下载 WaveStitch 模型
    
    Args:
        model_uri: S3 URI (s3://bucket/prefix)
        local_dir: 本地目录
    
    Returns:
        下载结果
    """
    if not model_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {model_uri}")
    
    parts = model_uri[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    
    result = download_directory_from_s3(
        s3_prefix=prefix,
        local_dir=local_dir,
        bucket=bucket,
    )
    
    # 加载配置
    config_path = os.path.join(local_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        result["config"] = config
    
    return result

def _prepare_data_manually(config: dict, df: pd.DataFrame, time_col: str, target_cols: list):
    """
    手动准备 WaveStitch 输入数据
    """
    import json
    import numpy as np
    
    prepared_dir = config["prepared_dir"]
    os.makedirs(prepared_dir, exist_ok=True)
    
    print(f"[PREPARE] Creating data manually in {prepared_dir}")
    
    # 所有数值列
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cond_cols = [c for c in numeric_cols if c not in target_cols and c != time_col]
    
    all_model_cols = target_cols + cond_cols
    
    # 1. 创建 meta.json
    meta = {
        "time_col": time_col,
        "target_cols": target_cols,
        "cond_cols": cond_cols,
        "all_model_cols": all_model_cols,
        "n_rows": len(df),
        "n_target_cols": len(target_cols),
        "n_cond_cols": len(cond_cols),
    }
    
    meta_path = os.path.join(prepared_dir, "meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"[PREPARE] ✓ Created meta.json")
    
    # 2. 分割数据 (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # 3. 保存 train.csv
    train_path = os.path.join(prepared_dir, "train.csv")
    train_df.to_csv(train_path, index=False)
    print(f"[PREPARE] ✓ Created train.csv ({len(train_df)} rows)")
    
    # 4. 保存 test_gt.csv
    test_gt_path = os.path.join(prepared_dir, "test_gt.csv")
    test_df.to_csv(test_gt_path, index=False)
    print(f"[PREPARE] ✓ Created test_gt.csv ({len(test_df)} rows)")
    
    # 5. 保存 test_input.csv
    test_input_path = os.path.join(prepared_dir, "test_input.csv")
    test_df.to_csv(test_input_path, index=False)
    print(f"[PREPARE] ✓ Created test_input.csv")
    
    # 6. 计算并保存 scaler
    scaler_dir = os.path.join(prepared_dir, "scaler")
    os.makedirs(scaler_dir, exist_ok=True)
    
    train_target = train_df[target_cols].values.astype(np.float64)
    
    col_means = np.nanmean(train_target, axis=0)
    col_stds = np.nanstd(train_target, axis=0)
    col_stds = np.where(col_stds < 1e-8, 1.0, col_stds)
    
    np.save(os.path.join(scaler_dir, "mean.npy"), col_means)
    np.save(os.path.join(scaler_dir, "std.npy"), col_stds)
    print(f"[PREPARE] ✓ Created scaler (mean, std)")
    
    # 7. 创建 saved_model 目录
    save_dir = os.path.join(prepared_dir, "saved_model")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[PREPARE] ✓ Created saved_model dir")
    
    # 列出创建的文件
    files = os.listdir(prepared_dir)
    print(f"[PREPARE] Files created: {files}")