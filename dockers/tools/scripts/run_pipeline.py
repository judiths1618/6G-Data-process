#!/usr/bin/env python3
"""
run_pipeline.py
WaveStitchPlus Pipeline - S3 作为数据湖存储
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import glob
from datetime import datetime
from pathlib import Path
from io import BytesIO

import pandas as pd
import numpy as np
import boto3
from botocore.client import Config

sys.path.insert(0, '/app/WaveStitchPlus_app')


# ============ S3 客户端 ============

class S3Client:
    def __init__(self):
        self.client = boto3.client(
            's3',
            endpoint_url=os.environ.get('S3_ENDPOINT', 'http://seaweed-s3:8333'),
            aws_access_key_id=os.environ.get('S3_ACCESS_KEY', 'anykey'),
            aws_secret_access_key=os.environ.get('S3_SECRET_KEY', 'anysecret'),
            region_name='us-east-1',
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}),
        )
        self.bucket = os.environ.get('S3_BUCKET', 'airflow-bucket')
    
    def download_file(self, key: str, local_path: str):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.client.download_file(self.bucket, key, local_path)
        print(f"[S3] ↓ {key}")
    
    def upload_file(self, local_path: str, key: str):
        self.client.upload_file(local_path, self.bucket, key)
        print(f"[S3] ↑ {key}")
    
    def upload_directory(self, local_dir: str, s3_prefix: str):
        """上传整个目录到 S3"""
        exclude = ['__pycache__', '.pyc', '.git']
        uploaded = []
        
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                
                if any(ex in local_path for ex in exclude):
                    continue
                
                relative = os.path.relpath(local_path, local_dir)
                s3_key = f"{s3_prefix}/{relative}".replace('\\', '/')
                
                self.client.upload_file(local_path, self.bucket, s3_key)
                uploaded.append(s3_key)
        
        print(f"[S3] ↑ {len(uploaded)} files -> {s3_prefix}/")
        return uploaded
    
    def download_directory(self, s3_prefix: str, local_dir: str):
        """从 S3 下载整个目录"""
        os.makedirs(local_dir, exist_ok=True)
        downloaded = []
        
        paginator = self.client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('/'):
                    continue
                
                relative = key[len(s3_prefix):].lstrip('/')
                local_path = os.path.join(local_dir, relative)
                
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                self.client.download_file(self.bucket, key, local_path)
                downloaded.append(local_path)
        
        print(f"[S3] ↓ {len(downloaded)} files <- {s3_prefix}/")
        return downloaded
    
    def list_versions(self, prefix: str) -> list:
        """列出某个前缀下的所有版本"""
        versions = set()
        paginator = self.client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter='/'):
            for cp in page.get('CommonPrefixes', []):
                version = cp['Prefix'].rstrip('/').split('/')[-1]
                versions.add(version)
        
        return sorted(versions)


# ============ 查找生成的文件 ============

def find_all_generated_files(base_dir: str) -> dict:
    """
    查找训练脚本生成的所有文件
    
    返回:
        {
            'saved_model': path or None,
            'scaler': path or None,
            'meta': path or None,
            'train_imputed': path or None,
            'train_csv': path or None,
            'all_files': [list of all files]
        }
    """
    result = {
        'saved_model': None,
        'scaler': None,
        'meta': None,
        'train_imputed': None,
        'train_csv': None,
        'all_files': []
    }
    
    # 搜索目录列表
    search_dirs = [
        base_dir,
        os.path.join(base_dir, 'prepared'),
        '/tmp',
        '/app/WaveStitchPlus_app',
    ]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        for root, dirs, files in os.walk(search_dir):
            # 查找 saved_model 目录
            if 'saved_model' in dirs:
                candidate = os.path.join(root, 'saved_model')
                if os.listdir(candidate):  # 非空
                    result['saved_model'] = candidate
                    print(f"[FIND] saved_model: {candidate}")
            
            # 查找 scaler 目录
            if 'scaler' in dirs:
                candidate = os.path.join(root, 'scaler')
                if os.listdir(candidate):
                    result['scaler'] = candidate
                    print(f"[FIND] scaler: {candidate}")
            
            # 查找文件
            for f in files:
                fpath = os.path.join(root, f)
                result['all_files'].append(fpath)
                
                if f == 'meta.json' and result['meta'] is None:
                    result['meta'] = fpath
                    print(f"[FIND] meta.json: {fpath}")
                
                if f == 'train_imputed.npy' and result['train_imputed'] is None:
                    result['train_imputed'] = fpath
                    print(f"[FIND] train_imputed.npy: {fpath}")
                
                if f == 'train.csv' and result['train_csv'] is None:
                    result['train_csv'] = fpath
                    print(f"[FIND] train.csv: {fpath}")
    
    return result


# ============ 训练阶段 ============

def run_train(
    s3: S3Client,
    input_key: str,
    dataset_name: str,
    version: str,
    time_col: str,
    target_cols: str,
    epochs: int,
    batch_size: int,
    window_size: int,
    use_em: bool,
    em_iterations: int,
    epochs_per_em: int,
):
    """训练阶段"""
    # 本地工作目录
    local_work = f"/tmp/wavestitchplus/{dataset_name}/{version}"
    local_prepared = os.path.join(local_work, "prepared")
    os.makedirs(local_prepared, exist_ok=True)
    
    # S3 目标路径
    s3_prefix = f"cleaned/{dataset_name}/{version}"
    
    # Step 1: 下载输入数据
    print(f"\n[TRAIN] Step 1: Download input data")
    local_input = os.path.join(local_work, "input.csv")
    s3.download_file(input_key, local_input)
    
    # Step 2: 运行训练脚本
    print(f"\n[TRAIN] Step 2: Run training script")
    
    cmd = [
        'python', '/app/WaveStitchPlus_app/train_wavestitchPlus_customdata.py', # WaveStitch+EM methods
        '-d', 'custom_csv',
        '-input_csv', local_input,
        '-prepared_dir', local_prepared,
        '-epochs', str(epochs),
        '-batch_size', str(batch_size),
        '-window_size', str(window_size),
        '-stride', '1',
    ]
    
    if use_em:
        cmd.extend([
            '-use_em',
            '-em_iterations', str(em_iterations),
            '-epochs_per_em', str(epochs_per_em),
            '-ddim_steps', '50',
        ])    
    print(f"[TRAIN] CMD: {' '.join(cmd)}")
    
    start_time = datetime.utcnow()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='/app/WaveStitchPlus_app')
    end_time = datetime.utcnow()
    
    # 保存日志
    log_path = os.path.join(local_work, "training.log")
    with open(log_path, 'w') as f:
        f.write(f"Command: {' '.join(cmd)}\n\n")
        f.write(f"Return code: {result.returncode}\n\n")
        f.write(f"STDOUT:\n{result.stdout}\n\n")
        f.write(f"STDERR:\n{result.stderr}\n")
    
    print(f"[TRAIN] STDOUT (last 3000 chars):\n{result.stdout[-3000:]}")
    
    if result.returncode != 0:
        print(f"[TRAIN] STDERR:\n{result.stderr}")
        raise RuntimeError(f"Training failed with code {result.returncode}")
    
    # Step 3: 查找所有生成的文件
    print(f"\n[TRAIN] Step 3: Find generated files")
    generated = find_all_generated_files(local_work)
    
    # 确保关键文件都在 local_prepared 下
    # 复制 saved_model
    if generated['saved_model']:
        dst = os.path.join(local_prepared, 'saved_model')
        if generated['saved_model'] != dst:
            print(f"[TRAIN] Copying saved_model to {dst}")
            shutil.copytree(generated['saved_model'], dst, dirs_exist_ok=True)
    
    # 复制 scaler
    if generated['scaler']:
        dst = os.path.join(local_prepared, 'scaler')
        if generated['scaler'] != dst:
            print(f"[TRAIN] Copying scaler to {dst}")
            shutil.copytree(generated['scaler'], dst, dirs_exist_ok=True)
    
    # 复制其他文件
    if generated['meta'] and os.path.dirname(generated['meta']) != local_prepared:
        shutil.copy(generated['meta'], os.path.join(local_prepared, 'meta.json'))
    
    if generated['train_imputed'] and os.path.dirname(generated['train_imputed']) != local_prepared:
        shutil.copy(generated['train_imputed'], os.path.join(local_prepared, 'train_imputed.npy'))
    
    # Step 4: 列出要上传的文件
    print(f"\n[TRAIN] Step 4: Files to upload")
    print(f"Contents of {local_work}:")
    for root, dirs, files in os.walk(local_work):
        level = root.replace(local_work, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = '  ' * (level + 1)
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath) / 1024
            print(f"{sub_indent}{f} ({size:.1f} KB)")
    
    # Step 5: 保存训练信息
    training_info = {
        'dataset_name': dataset_name,
        'version': version,
        'input_key': input_key,
        's3_prefix': s3_prefix,
        'params': {
            'time_col': time_col,
            'target_cols': target_cols,
            'epochs': epochs,
            'batch_size': batch_size,
            'window_size': window_size,
            'use_em': use_em,
            'em_iterations': em_iterations,
            'epochs_per_em': epochs_per_em,
        },
        'started_at': start_time.isoformat(),
        'completed_at': end_time.isoformat(),
        'duration_seconds': (end_time - start_time).total_seconds(),
        'files_found': {
            'saved_model': generated['saved_model'] is not None,
            'scaler': generated['scaler'] is not None,
            'meta': generated['meta'] is not None,
            'train_imputed': generated['train_imputed'] is not None,
        }
    }
    
    info_path = os.path.join(local_work, "training_info.json")
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    # Step 6: 上传到 S3
    print(f"\n[TRAIN] Step 5: Upload to S3")
    s3.upload_directory(local_work, s3_prefix)
    
    # 验证关键文件
    print(f"\n[TRAIN] Verifying uploaded files...")
    key_checks = [
        f"{s3_prefix}/prepared/saved_model/",
        f"{s3_prefix}/prepared/scaler/",
        f"{s3_prefix}/prepared/meta.json",
    ]
    
    for check in key_checks:
        # 简单检查（列出前缀）
        found = bool(s3.client.list_objects_v2(Bucket=s3.bucket, Prefix=check, MaxKeys=1).get('Contents'))
        status = "✓" if found else "✗"
        print(f"  {status} {check}")
    
    print(f"\n[TRAIN] ✓ Complete!")
    print(f"[TRAIN] S3: s3://{s3.bucket}/{s3_prefix}/")
    
    return training_info


# ============ 推理阶段 ============
def run_inference(
    s3: S3Client,
    input_key: str,
    dataset_name: str,
    model_version: str,
    output_version: str,
    n_trials: int,
    guidance_scale: float,
):
    """推理阶段 - 使用训练时的测试集"""
    
    # 确定模型版本
    if model_version is None:
        versions = s3.list_versions(f"wavestitchplus/{dataset_name}/")
        if not versions:
            raise FileNotFoundError(f"No models found for dataset: {dataset_name}")
        model_version = versions[-1]
        print(f"[INFERENCE] Auto-selected latest model: {model_version}")
    
    # 本地工作目录
    local_work = f"/tmp/wavestitchplus/{dataset_name}/inference_{output_version}"
    local_prepared = os.path.join(local_work, "prepared")
    os.makedirs(local_prepared, exist_ok=True)
    
    # S3 路径 - 🔥 修改：推理结果保存在模型版本目录的上一级
    model_s3_prefix = f"wavestitchplus/{dataset_name}/{model_version}/prepared"
    output_s3_prefix = f"wavestitchplus/{dataset_name}/{model_version}"  # 🔥 去掉 /results/
    
    print(f"\n{'='*60}")
    print(f"[INFERENCE] Starting inference")
    print(f"{'='*60}")
    print(f"Dataset:        {dataset_name}")
    print(f"Model version:  {model_version}")
    print(f"Output version: {output_version}")
    print(f"{'='*60}\n")
    
    # Step 1: 从 S3 下载完整的 prepared 目录
    print(f"[INFERENCE] Step 1/5: Download prepared directory from S3")
    print(f"  S3 prefix: s3://{s3.bucket}/{model_s3_prefix}")
    
    s3.download_directory(model_s3_prefix, local_prepared)
    
    print(f"\n[INFERENCE] Downloaded files:")
    for root, dirs, files in os.walk(local_prepared):
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath) / 1024
            rel = os.path.relpath(fpath, local_prepared)
            print(f"  {rel} ({size:.1f} KB)")
    
    # 验证必需文件
    print(f"\n[INFERENCE] Step 2/5: Verify required files")
    
    required_files = {
        'saved_model/': 'Model directory',
        'scaler/mean.npy': 'Scaler mean',
        'scaler/std.npy': 'Scaler std',
        'meta.json': 'Dataset metadata',
        'test_input.csv': 'Test input (from training)',
        'test_gt.csv': 'Test ground truth (from training)',
    }
    
    all_ok = True
    for rel_path, description in required_files.items():
        full_path = os.path.join(local_prepared, rel_path.rstrip('/'))
        exists = os.path.exists(full_path)
        
        if exists:
            if os.path.isdir(full_path):
                file_count = len([f for f in os.listdir(full_path) 
                                 if os.path.isfile(os.path.join(full_path, f))])
                print(f"  ✓ {rel_path:30s} ({file_count} files)")
            else:
                size = os.path.getsize(full_path)
                print(f"  ✓ {rel_path:30s} ({size:,} bytes)")
        else:
            print(f"  ❌ {rel_path:30s} (MISSING)")
            all_ok = False
    
    if not all_ok:
        raise FileNotFoundError(f"Required files missing in {local_prepared}")
    
    # 验证 test_input.csv 格式
    test_input_path = os.path.join(local_prepared, "test_input.csv")
    df_test_input = pd.read_csv(test_input_path)
    
    print(f"\n  test_input.csv:")
    print(f"    Shape: {df_test_input.shape}")
    print(f"    Columns: {list(df_test_input.columns)[:5]}..." if len(df_test_input.columns) > 5 else f"    Columns: {list(df_test_input.columns)}")
    
    # 读取 meta.json
    with open(os.path.join(local_prepared, 'meta.json'), 'r') as f:
        meta = json.load(f)
    
    target_cols = meta.get('target_cols', [])
    
    if target_cols:
        nan_count = df_test_input[target_cols].isna().sum().sum()
        total = len(df_test_input) * len(target_cols)
        print(f"    Missing in target cols: {nan_count}/{total} ({nan_count/total*100:.1f}%)")
    
    # Step 3: 运行推理脚本
    print(f"\n[INFERENCE] Step 3/5: Run inference script")
    
    cmd = [
        'python', '/app/WaveStitchPlus_app/synthesis_wavestitchPlus_pipeline_strided_preconditioning_customdata.py',
        '-d', 'custom_csv',
        '-prepared_dir', local_prepared,
        '-n_trials', str(n_trials),
        '-guidance_scale', str(guidance_scale),
        '-synth_mask', 'gap_imputation',
        '-stride', '1',
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    
    start_time = datetime.utcnow()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='/app/WaveStitchPlus_app')
    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()
    
    print(f"  Completed in {duration:.1f}s")
    
    # 保存日志
    log_path = os.path.join(local_work, "inference.log")
    with open(log_path, 'w') as f:
        f.write(f"Command: {' '.join(cmd)}\n\n")
        f.write(f"Return code: {result.returncode}\n")
        f.write(f"Duration: {duration:.1f}s\n\n")
        f.write(f"STDOUT:\n{result.stdout}\n\n")
        f.write(f"STDERR:\n{result.stderr}\n")
    
    print(f"\n  STDOUT (last 2000 chars):\n{result.stdout[-2000:]}")
    
    if result.returncode != 0:
        print(f"\n  ❌ STDERR:\n{result.stderr}")
        raise RuntimeError(f"Inference failed with code {result.returncode}")
    
    # Step 4: 🔥 修改：查找输出文件（扩展搜索范围）
    print(f"\n[INFERENCE] Step 4/5: Find output files")
    
    # 搜索多个可能的位置
    search_dirs = [
        
        '/app/WaveStitchPlus_app',  # 推理脚本的工作目录
        '/app/WaveStitchPlus_app/generated',  # 默认输出位置
        local_work,  # /tmp/wavestitchplus/.../inference_xxx/
        os.path.dirname(local_work),  # /tmp/wavestitchplus/.../
    ]
    
    output_csv = None
    all_found_csvs = []
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        
        print(f"  Searching: {search_dir}")
        
        for root, dirs, files in os.walk(search_dir):
            for f in files:
                if not f.endswith('.csv'):
                    continue
                
                # 排除输入文件
                if any(x in f.lower() for x in ['test_input', 'test_gt', 'train', 'config', 'timing']):
                    continue
                
                # 查找包含 imputed 的文件
                if 'imputed' not in f.lower():
                    continue
                
                fpath = os.path.join(root, f)
                fsize = os.path.getsize(fpath)
                
                if fsize < 100:
                    continue
                
                rel_path = os.path.relpath(fpath, search_dir)
                print(f"    Found: {rel_path} ({fsize:,} bytes)")
                all_found_csvs.append(fpath)
                
                # 选择最新的
                if output_csv is None or os.path.getmtime(fpath) > os.path.getmtime(output_csv):
                    output_csv = fpath
    
    if not output_csv:
        print(f"\n  ⚠ No output CSV found in any search location")
        print(f"\n  Listing all files in /app/WaveStitchPlus_app/generated:")
        
        generated_dir = '/app/WaveStitchPlus_app/generated'
        if os.path.exists(generated_dir):
            for root, dirs, files in os.walk(generated_dir):
                for f in files:
                    fpath = os.path.join(root, f)
                    size = os.path.getsize(fpath)
                    rel = os.path.relpath(fpath, generated_dir)
                    print(f"    {rel} ({size:,} bytes)")
        
        raise FileNotFoundError(f"Inference output not found. Check inference.log")
    
    print(f"\n  ✓ Selected output: {output_csv}")
    
    # Step 5: 🔥 修改：处理和上传所有推理结果
    print(f"\n[INFERENCE] Step 5/5: Process and upload results")
    
    df_result = pd.read_csv(output_csv)
    print(f"  Output shape: {df_result.shape}")
    
    if target_cols:
        result_nan = df_result[target_cols].isna().sum().sum()
        print(f"  Remaining NaN: {result_nan}")
    
    # 🔥 创建推理结果目录结构
    inference_output_dir = os.path.join(local_work, "inference_results")
    os.makedirs(inference_output_dir, exist_ok=True)
    
    # 保存主要结果
    final_csv = os.path.join(inference_output_dir, "imputed.csv")
    df_result.to_csv(final_csv, index=False)
    print(f"  ✓ Saved: imputed.csv")
    
    # 复制所有生成的文件
    if all_found_csvs:
        trials_dir = os.path.join(inference_output_dir, "trials")
        os.makedirs(trials_dir, exist_ok=True)
        
        for csv_path in all_found_csvs:
            filename = os.path.basename(csv_path)
            dst = os.path.join(trials_dir, filename)
            shutil.copy2(csv_path, dst)
            print(f"  ✓ Copied: {filename}")
    
    # 保存元数据
    inference_info = {
        'dataset_name': dataset_name,
        'model_version': model_version,
        'output_version': output_version,
        'model_s3_prefix': model_s3_prefix,
        'inference_timestamp': datetime.utcnow().isoformat(),
        'shapes': {
            'test_input': list(df_test_input.shape),
            'output': list(df_result.shape),
        },
        'missing_values': {
            'input': int(nan_count) if target_cols else 0,
            'output': int(result_nan) if target_cols else 0,
            'filled': int(nan_count - result_nan) if target_cols else 0,
        },
        'params': {
            'n_trials': n_trials,
            'guidance_scale': guidance_scale,
        },
        'timing': {
            'started_at': start_time.isoformat(),
            'completed_at': end_time.isoformat(),
            'duration_seconds': duration,
        },
        'files': {
            'main_output': 'imputed.csv',
            'trials': [os.path.basename(f) for f in all_found_csvs],
            'log': 'inference.log',
        }
    }
    
    info_path = os.path.join(inference_output_dir, "inference_info.json")
    with open(info_path, 'w') as f:
        json.dump(inference_info, f, indent=2)
    print(f"  ✓ Saved: inference_info.json")
    
    # 复制日志
    shutil.copy2(log_path, os.path.join(inference_output_dir, "inference.log"))
    
    # 🔥 上传整个推理结果目录到 S3（在 prepared 的同级）
    print(f"\n  Uploading inference results to S3...")
    s3_inference_prefix = f"{output_s3_prefix}/inference_results_{output_version}"
    
    s3.upload_directory(inference_output_dir, s3_inference_prefix)
    
    print(f"  ✓ Uploaded to: s3://{s3.bucket}/{s3_inference_prefix}/")
    
    # 🔥 同时上传到 latest（便于快速访问）
    latest_prefix = f"wavestitchplus/{dataset_name}/latest_inference"
    
    s3.upload_file(final_csv, f"{latest_prefix}/imputed.csv")
    s3.upload_file(info_path, f"{latest_prefix}/inference_info.json")
    
    print(f"  ✓ Latest: s3://{s3.bucket}/{latest_prefix}/")
    
    # 打印最终的 S3 结构
    print(f"\n  Final S3 structure:")
    print(f"    s3://{s3.bucket}/wavestitchplus/{dataset_name}/{model_version}/")
    print(f"      ├── prepared/              (training artifacts)")
    print(f"      └── inference_results_{output_version}/  (inference outputs)")
    print(f"           ├── imputed.csv")
    print(f"           ├── inference_info.json")
    print(f"           ├── inference.log")
    print(f"           └── trials/")
    
    print(f"\n{'='*60}")
    print(f"[INFERENCE] ✓ Complete!")
    print(f"{'='*60}")
    print(f"Test input:     {df_test_input.shape}")
    print(f"Output:         {df_result.shape}")
    if target_cols:
        print(f"Input missing:  {nan_count}")
        print(f"Output missing: {result_nan}")
        print(f"Filled:         {nan_count - result_nan}")
        print(f"Success rate:   {(nan_count - result_nan) / nan_count * 100 if nan_count > 0 else 0:.1f}%")
    print(f"{'='*60}\n")
    
    return inference_info

# ============ 主程序 ============

def main():
    parser = argparse.ArgumentParser(description='WaveStitchPlus Pipeline')
    
    parser.add_argument('--mode', choices=['train', 'inference', 'full'], default='full')
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--input-s3-key', type=str, required=True)
    
    # 训练参数
    parser.add_argument('--time-col', type=str, default='time')
    parser.add_argument('--target-cols', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--window-size', type=int, default=32)
    parser.add_argument('--use-em', action='store_true')
    parser.add_argument('--em-iterations', type=int, default=5)
    parser.add_argument('--epochs-per-em', type=int, default=200)
    
    # 推理参数
    parser.add_argument('--model-version', type=str, default=None)
    parser.add_argument('--n-trials', type=int, default=1)
    parser.add_argument('--guidance-scale', type=float, default=0.1)
    
    args = parser.parse_args()
    
    version = args.version or datetime.now().strftime('%Y%m%d_%H%M%S')
    s3 = S3Client()
    
    # GPU 检查
    import torch
    print(f"\n{'='*60}")
    print(f"WaveStitch Pipeline - {args.mode.upper()}")
    print(f"{'='*60}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Version: {version}")
    print(f"Input: s3://{s3.bucket}/{args.input_s3_key}")
    print(f"{'='*60}\n")
    
    results = {}
    
    try:
        if args.mode in ['train', 'full']:
            print(f"\n{'='*60}")
            print(f"TRAINING PHASE")
            print(f"{'='*60}")
            
            results['training'] = run_train(
                s3=s3,
                input_key=args.input_s3_key,
                dataset_name=args.dataset_name,
                version=version,
                time_col=args.time_col,
                target_cols=args.target_cols,
                epochs=args.epochs,
                batch_size=args.batch_size,
                window_size=args.window_size,
                use_em=args.use_em,
                em_iterations=args.em_iterations,
                epochs_per_em=args.epochs_per_em,
            )
        
        if args.mode in ['inference', 'full']:
            print(f"\n{'='*60}")
            print(f"INFERENCE PHASE")
            print(f"{'='*60}")
            
            model_ver = version if args.mode == 'full' else args.model_version
            
            results['inference'] = run_inference(
                s3=s3,
                input_key=args.input_s3_key,
                dataset_name=args.dataset_name,
                model_version=model_ver,
                output_version=version,
                n_trials=args.n_trials,
                guidance_scale=args.guidance_scale,
            )
        
        print(f"\n{'='*60}")
        print(f"✓ PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(json.dumps(results, indent=2, default=str))
        print(f"\n__RESULT_JSON__:{json.dumps(results, default=str)}")
        
    finally:
        # 清理本地文件
        work_dir = f"/tmp/wavestitchplus/{args.dataset_name}"
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
            print(f"\n[CLEANUP] Removed {work_dir}")


if __name__ == '__main__':
    main()