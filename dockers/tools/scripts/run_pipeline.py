#!/usr/bin/env python3
"""
run_pipeline.py
WaveStitchPlus Pipeline - S3-backed data lake storage: load --> train (save models) + inference --> store curated data
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from datetime import datetime

import pandas as pd
import boto3
from botocore.client import Config

sys.path.insert(0, '/app/WaveStitchPlus_app')


# ============ Helpers ============

def normalize_target_cols(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [x.strip() for x in value.split(',') if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def safe_makedirs_for_file(path: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


# ============ S3 Client ============

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
        safe_makedirs_for_file(local_path)
        self.client.download_file(self.bucket, key, local_path)
        print(f"[S3] ↓ {key}")

    def upload_file(self, local_path: str, key: str):
        self.client.upload_file(local_path, self.bucket, key)
        print(f"[S3] ↑ {key}")

    def upload_directory(self, local_dir: str, s3_prefix: str):
        exclude = ['__pycache__', '.pyc', '.git']
        uploaded = []

        for root, dirs, files in os.walk(local_dir):
            dirs[:] = [d for d in dirs if d not in {'__pycache__', '.git'}]
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
                safe_makedirs_for_file(local_path)
                self.client.download_file(self.bucket, key, local_path)
                downloaded.append(local_path)

        print(f"[S3] ↓ {len(downloaded)} files <- {s3_prefix}/")
        return downloaded

    def prefix_exists(self, prefix: str) -> bool:
        resp = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix, MaxKeys=1)
        return bool(resp.get('Contents'))

    def object_exists(self, key: str) -> bool:
        resp = self.client.list_objects_v2(Bucket=self.bucket, Prefix=key, MaxKeys=1)
        return any(obj.get('Key') == key for obj in resp.get('Contents', []))

    def list_candidate_versions(self, dataset_name: str) -> list:
        prefix = f"wavestitchplus/{dataset_name}/"
        versions = set()
        paginator = self.client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter='/'):
            for cp in page.get('CommonPrefixes', []):
                version = cp['Prefix'].rstrip('/').split('/')[-1]
                versions.add(version)

        return sorted(versions)

    def list_valid_model_versions(self, dataset_name: str) -> list:
        valid = []
        for version in self.list_candidate_versions(dataset_name):
            prepared_prefix = f"wavestitchplus/{dataset_name}/{version}/prepared/"
            meta_key = prepared_prefix + "meta.json"
            model_prefix = prepared_prefix + "saved_model/"
            if self.object_exists(meta_key) and self.prefix_exists(model_prefix):
                valid.append(version)
        return sorted(valid)


# ============ Artifact Discovery ============

def find_generated_files(base_dir: str) -> dict:
    result = {
        'saved_model': None,
        'scaler': None,
        'meta': None,
        'train_imputed': None,
        'train_imputed_denorm': None,
        'train_csv': None,
        'all_files': [],
    }

    for root, dirs, files in os.walk(base_dir):
        if 'saved_model' in dirs:
            candidate = os.path.join(root, 'saved_model')
            if os.listdir(candidate):
                result['saved_model'] = candidate
                print(f"[FIND] saved_model: {candidate}")

        if 'scaler' in dirs:
            candidate = os.path.join(root, 'scaler')
            if os.listdir(candidate):
                result['scaler'] = candidate
                print(f"[FIND] scaler: {candidate}")

        for f in files:
            fpath = os.path.join(root, f)
            result['all_files'].append(fpath)

            if f == 'meta.json' and result['meta'] is None:
                result['meta'] = fpath
                print(f"[FIND] meta.json: {fpath}")
            elif f == 'train_imputed.npy' and result['train_imputed'] is None:
                result['train_imputed'] = fpath
                print(f"[FIND] train_imputed.npy: {fpath}")
            elif f == 'train_imputed_denorm.npy' and result['train_imputed_denorm'] is None:
                result['train_imputed_denorm'] = fpath
                print(f"[FIND] train_imputed_denorm.npy: {fpath}")
            elif f == 'train.csv' and result['train_csv'] is None:
                result['train_csv'] = fpath
                print(f"[FIND] train.csv: {fpath}")

    return result


# ============ Training ============

def run_train(
    s3: S3Client,
    input_key: str,
    dataset_name: str,
    version: str,
    time_col: str,
    target_cols,
    epochs: int,
    batch_size: int,
    window_size: int,
    use_em: bool,
    em_iterations: int,
    epochs_per_em: int,
    repaint_rounds: int,
    clamp_mode: str = 'bounds',
):
    local_work = f"/tmp/wavestitchplus/{dataset_name}/{version}"
    local_prepared = os.path.join(local_work, "prepared")
    os.makedirs(local_prepared, exist_ok=True)

    s3_prefix = f"wavestitchplus/{dataset_name}/{version}"
    normalized_target_cols = normalize_target_cols(target_cols)

    print(f"\n[TRAIN] Step 1: Download input data")
    local_input = os.path.join(local_work, "input.csv")
    s3.download_file(input_key, local_input)

    print(f"\n[TRAIN] Step 2: Run training script")
    cmd = [
        'python', '/app/WaveStitchPlus_app/train_improved.py',
        '-d', 'custom_csv',
        '-input_csv', local_input,
        '-prepared_dir', local_prepared,
        '-repaint_rounds', str(repaint_rounds),
        '-save_train_imputed_denorm',
        '-train_imputed_clamp', clamp_mode,
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

    log_path = os.path.join(local_work, "training.log")
    with open(log_path, 'w') as f:
        f.write(f"Command: {' '.join(cmd)}\n\n")
        f.write(f"Return code: {result.returncode}\n\n")
        f.write(f"STDOUT:\n{result.stdout}\n\n")
        f.write(f"STDERR:\n{result.stderr}\n")

    print(f"[TRAIN] STDOUT (last 3000 chars):\n{result.stdout[-3000:]}")

    if result.returncode != 0:
        print(f"[TRAIN] ❌ Training script failed (exit code {result.returncode})")
        print(f"[TRAIN] STDERR (full):\n{result.stderr}")
        print(f"[TRAIN] STDOUT (full):\n{result.stdout}")
        print(f"[TRAIN] Log saved to: {log_path}")
        raise RuntimeError(
            f"Training failed with code {result.returncode}. "
            f"Re-run with --keep-workdir to inspect logs locally."
        )

    print(f"\n[TRAIN] Step 3: Find generated files")
    generated = find_generated_files(local_work)

    print(f"\n[TRAIN] Step 4: Consolidate files to prepared dir")
    if generated['saved_model']:
        dst = os.path.join(local_prepared, 'saved_model')
        if generated['saved_model'] != dst:
            print(f"[TRAIN] Copying saved_model to {dst}")
            shutil.copytree(generated['saved_model'], dst, dirs_exist_ok=True)

    if generated['scaler']:
        dst = os.path.join(local_prepared, 'scaler')
        if generated['scaler'] != dst:
            print(f"[TRAIN] Copying scaler to {dst}")
            shutil.copytree(generated['scaler'], dst, dirs_exist_ok=True)

    if generated['meta'] and os.path.dirname(generated['meta']) != local_prepared:
        shutil.copy2(generated['meta'], os.path.join(local_prepared, 'meta.json'))
    if generated['train_imputed'] and os.path.dirname(generated['train_imputed']) != local_prepared:
        shutil.copy2(generated['train_imputed'], os.path.join(local_prepared, 'train_imputed.npy'))
    if generated['train_imputed_denorm'] and os.path.dirname(generated['train_imputed_denorm']) != local_prepared:
        shutil.copy2(generated['train_imputed_denorm'], os.path.join(local_prepared, 'train_imputed_denorm.npy'))

    required_after_train = [
        os.path.join(local_prepared, 'saved_model'),
        os.path.join(local_prepared, 'scaler'),
        os.path.join(local_prepared, 'meta.json'),
    ]
    for path in required_after_train:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required training artifact missing: {path}")

    print(f"\n[TRAIN] Step 5: Files to upload")
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

    print(f"\n[TRAIN] Step 6: Save training info")
    training_info = {
        'dataset_name': dataset_name,
        'version': version,
        'input_key': input_key,
        's3_prefix': s3_prefix,
        'requested_params': {
            'time_col': time_col,
            'target_cols': normalized_target_cols,
            'epochs': epochs,
            'batch_size': batch_size,
            'window_size': window_size,
            'use_em': use_em,
            'em_iterations': em_iterations,
            'epochs_per_em': epochs_per_em,
            'repaint_rounds': repaint_rounds,
            'clamp_mode': clamp_mode,
        },
        'started_at': start_time.isoformat(),
        'completed_at': end_time.isoformat(),
        'duration_seconds': (end_time - start_time).total_seconds(),
        'files_found': {
            'saved_model': generated['saved_model'] is not None,
            'scaler': generated['scaler'] is not None,
            'meta': generated['meta'] is not None,
            'train_imputed': generated['train_imputed'] is not None,
            'train_imputed_denorm': generated['train_imputed_denorm'] is not None,
        }
    }

    info_path = os.path.join(local_work, "training_info.json")
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)

    print(f"\n[TRAIN] Step 7: Upload to S3")
    s3.upload_directory(local_work, s3_prefix)

    print(f"\n[TRAIN] Verifying uploaded files...")
    key_checks = [
        f"{s3_prefix}/prepared/saved_model/",
        f"{s3_prefix}/prepared/scaler/",
        f"{s3_prefix}/prepared/meta.json",
        f"{s3_prefix}/prepared/train_imputed_denorm.npy",
    ]

    for check in key_checks:
        found = s3.prefix_exists(check) if check.endswith('/') else s3.object_exists(check)
        status = "✓" if found else "✗"
        print(f"  {status} {check}")
        if not found:
            raise FileNotFoundError(f"Uploaded artifact verification failed: {check}")

    print(f"\n[TRAIN] ✓ Complete!")
    print(f"[TRAIN] S3: s3://{s3.bucket}/{s3_prefix}/")
    return training_info


# ============ Inference ============

def run_inference(
    s3: S3Client,
    dataset_name: str,
    model_version: str,
    output_version: str,
    model_type: str = 'em',
    n_trials: int = 1,
    guidance_scale: float = 0.1,
    repaint_rounds: int = 5,
    clamp_mode: str = 'bounds',
    ddim_steps: int = 50,
    use_ddpm: bool = False,
    bound_headroom: float = 1.2,
    nonneg_cols: list = None,
    upper_bounds: str = None,
    lower_bounds: str = None,
):
    if model_version is None:
        versions = s3.list_valid_model_versions(dataset_name)
        if not versions:
            raise FileNotFoundError(f"No valid trained model versions found for dataset: {dataset_name}")
        model_version = versions[-1]
        print(f"[INFERENCE] Auto-selected latest valid model: {model_version}")

    local_work = f"/tmp/wavestitchplus/{dataset_name}/inference_{output_version}"
    local_prepared = os.path.join(local_work, "prepared")
    os.makedirs(local_prepared, exist_ok=True)

    model_s3_prefix = f"wavestitchplus/{dataset_name}/{model_version}/prepared"
    output_s3_prefix = f"wavestitchplus/{dataset_name}/{model_version}"

    print(f"\n{'='*60}")
    print(f"[INFERENCE] Starting inference")
    print(f"{'='*60}")
    print(f"Dataset:        {dataset_name}")
    print(f"Model version:  {model_version}")
    print(f"Output version: {output_version}")
    print(f"Model type:     {model_type}")
    print(f"Clamp mode:     {clamp_mode}")
    print(f"Repaint rounds: {repaint_rounds}")
    print(f"DDIM steps:     {ddim_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"{'='*60}\n")

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
                file_count = len([f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))])
                print(f"  ✓ {rel_path:30s} ({file_count} files)")
            else:
                size = os.path.getsize(full_path)
                print(f"  ✓ {rel_path:30s} ({size:,} bytes)")
        else:
            print(f"  ❌ {rel_path:30s} (MISSING) — {description}")
            all_ok = False

    if not all_ok:
        raise FileNotFoundError(f"Required files missing in {local_prepared}")

    test_input_path = os.path.join(local_prepared, "test_input.csv")
    df_test_input = pd.read_csv(test_input_path)

    print(f"\n  test_input.csv:")
    print(f"    Shape: {df_test_input.shape}")
    cols_preview = list(df_test_input.columns)
    print(f"    Columns: {cols_preview[:5]}..." if len(cols_preview) > 5 else f"    Columns: {cols_preview}")

    with open(os.path.join(local_prepared, 'meta.json'), 'r') as f:
        meta = json.load(f)

    target_cols = normalize_target_cols(meta.get('target_cols', []))
    nan_count = 0
    if target_cols:
        existing_target_cols = [c for c in target_cols if c in df_test_input.columns]
        if existing_target_cols:
            nan_count = int(df_test_input[existing_target_cols].isna().sum().sum())
            total = len(df_test_input) * len(existing_target_cols)
            print(f"    Missing in target cols: {nan_count}/{total} ({nan_count / total * 100:.1f}%)")

    print(f"\n[INFERENCE] Step 3/5: Run inference script")
    local_generated_dir = os.path.join(local_work, "generated")
    os.makedirs(local_generated_dir, exist_ok=True)
    local_output_csv = os.path.join(local_generated_dir, "wavestitchplus_v1_test_imputed.csv")

    _synth_cwd = local_prepared
    _run_name = os.path.basename(os.path.dirname(local_prepared))
    _save_rel = os.path.join("saved_models", _run_name, os.path.basename(local_prepared))
    _save_abs = os.path.join(_synth_cwd, _save_rel)
    os.makedirs(_save_abs, exist_ok=True)

    _sm_src = os.path.join(local_prepared, "saved_model")
    if not os.path.isdir(_sm_src):
        raise FileNotFoundError(f"saved_model directory not found at {_sm_src}")

    linked = []
    for fname in os.listdir(_sm_src):
        src_file = os.path.join(_sm_src, fname)
        dst_file = os.path.join(_save_abs, fname)
        if os.path.lexists(dst_file):
            continue
        try:
            os.symlink(src_file, dst_file)
        except OSError:
            shutil.copy2(src_file, dst_file)
        linked.append(fname)

    pth_files = [f for f in os.listdir(_save_abs) if f.endswith('.pth')]
    print(f"[INFERENCE] Prepared {len(pth_files)} model files in {_save_abs}")
    for f in pth_files:
        print(f"  → {f}")

    cmd = [
        'python', '/app/WaveStitchPlus_app/synthesis_improved.py',
        '-d', 'custom_csv',
        '-prepared_dir', local_prepared,
        '-out_csv', local_output_csv,
        '-model_type', model_type,
        '-clamp_mode', clamp_mode,
        '-repaint_rounds', str(repaint_rounds),
        '-guidance_scale', str(guidance_scale),
        '-n_trials', str(n_trials),
        '-ddim_steps', str(ddim_steps),
        '-bound_headroom', str(bound_headroom),
    ]

    if use_ddpm:
        cmd.append('-use_ddpm')
    if nonneg_cols:
        cmd += ['-nonneg_cols'] + list(nonneg_cols)
    if upper_bounds:
        cmd += ['-upper_bounds', upper_bounds]
    if lower_bounds:
        cmd += ['-lower_bounds', lower_bounds]

    print(f"  Command: {' '.join(cmd)}")

    start_time = datetime.utcnow()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_synth_cwd)
    end_time = datetime.utcnow()
    duration = (end_time - start_time).total_seconds()

    print(f"  Completed in {duration:.1f}s")

    log_path = os.path.join(local_work, "inference.log")
    with open(log_path, 'w') as f:
        f.write(f"Command: {' '.join(cmd)}\n\n")
        f.write(f"Return code: {result.returncode}\n")
        f.write(f"Duration: {duration:.1f}s\n\n")
        f.write(f"STDOUT:\n{result.stdout}\n\n")
        f.write(f"STDERR:\n{result.stderr}\n")

    print(f"\n  STDOUT (last 2000 chars):\n{result.stdout[-2000:]}")

    if result.returncode != 0:
        print(f"\n  ❌ Inference script failed (exit code {result.returncode})")
        print(f"  STDERR (full):\n{result.stderr}")
        print(f"  STDOUT (full):\n{result.stdout}")
        print(f"  Log saved to: {log_path}")
        raise RuntimeError(
            f"Inference failed with code {result.returncode}. "
            f"Re-run with --keep-workdir to inspect logs locally."
        )

    print(f"\n[INFERENCE] Step 4/5: Find output files")
    output_csv = None
    if os.path.exists(local_output_csv) and os.path.getsize(local_output_csv) > 100:
        output_csv = local_output_csv
        print(f"  ✓ Found output at expected path: {output_csv}")

    all_found_csvs = []
    search_dirs = [local_work, _synth_cwd, '/app/WaveStitchPlus_app', '/app/WaveStitchPlus_app/generated']
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            if os.path.commonpath([os.path.abspath(root), os.path.abspath(local_prepared)]) == os.path.abspath(local_prepared):
                continue
            for f in files:
                if not f.endswith('.csv'):
                    continue
                if any(x in f.lower() for x in ['test_input', 'test_gt', 'train', 'config', 'timing']):
                    continue
                if 'imputed' not in f.lower():
                    continue
                fpath = os.path.join(root, f)
                if os.path.getsize(fpath) <= 100:
                    continue
                if output_csv is None or (fpath == local_output_csv):
                    output_csv = fpath
                elif fpath != output_csv:
                    all_found_csvs.append(fpath)

    if not output_csv:
        raise FileNotFoundError("Inference output not found. Check inference.log")

    print(f"\n  ✓ Selected output: {output_csv}")

    print(f"\n[INFERENCE] Step 5/5: Process and upload results")
    df_result = pd.read_csv(output_csv)
    print(f"  Output shape: {df_result.shape}")

    result_nan = 0
    if target_cols:
        existing_target_cols = [c for c in target_cols if c in df_result.columns]
        if existing_target_cols:
            result_nan = int(df_result[existing_target_cols].isna().sum().sum())
            print(f"  Remaining NaN: {result_nan}")

    inference_output_dir = os.path.join(local_work, "inference_results")
    os.makedirs(inference_output_dir, exist_ok=True)

    final_csv = os.path.join(inference_output_dir, "imputed.csv")
    df_result.to_csv(final_csv, index=False)
    print(f"  ✓ Saved: imputed.csv")

    if all_found_csvs:
        trials_dir = os.path.join(inference_output_dir, "trials")
        os.makedirs(trials_dir, exist_ok=True)
        for csv_path in sorted(set(all_found_csvs)):
            filename = os.path.basename(csv_path)
            shutil.copy2(csv_path, os.path.join(trials_dir, filename))
            print(f"  ✓ Copied trial: {filename}")

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
            'input': nan_count,
            'output': result_nan,
            'filled': nan_count - result_nan,
        },
        'params': {
            'model_type': model_type,
            'n_trials': n_trials,
            'guidance_scale': guidance_scale,
            'repaint_rounds': repaint_rounds,
            'clamp_mode': clamp_mode,
            'ddim_steps': ddim_steps,
            'use_ddpm': use_ddpm,
            'bound_headroom': bound_headroom,
            'nonneg_cols': nonneg_cols,
            'upper_bounds': upper_bounds,
            'lower_bounds': lower_bounds,
        },
        'timing': {
            'started_at': start_time.isoformat(),
            'completed_at': end_time.isoformat(),
            'duration_seconds': duration,
        },
        'files': {
            'main_output': 'imputed.csv',
            'trials': [os.path.basename(f) for f in sorted(set(all_found_csvs))],
            'log': 'inference.log',
        }
    }

    info_path = os.path.join(inference_output_dir, "inference_info.json")
    with open(info_path, 'w') as f:
        json.dump(inference_info, f, indent=2)
    print(f"  ✓ Saved: inference_info.json")

    shutil.copy2(log_path, os.path.join(inference_output_dir, "inference.log"))

    s3_inference_prefix = f"{output_s3_prefix}/inference_results_{output_version}"
    s3.upload_directory(inference_output_dir, s3_inference_prefix)
    print(f"  ✓ Uploaded to: s3://{s3.bucket}/{s3_inference_prefix}/")

    latest_prefix = f"wavestitchplus/{dataset_name}/latest_inference"
    s3.upload_file(final_csv, f"{latest_prefix}/imputed.csv")
    s3.upload_file(info_path, f"{latest_prefix}/inference_info.json")
    print(f"  ✓ Latest: s3://{s3.bucket}/{latest_prefix}/")

    print(f"\n{'='*60}")
    print(f"[INFERENCE] ✓ Complete!")
    print(f"{'='*60}")
    print(f"Test input:     {df_test_input.shape}")
    print(f"Output:         {df_result.shape}")
    if target_cols:
        print(f"Input missing:  {nan_count}")
        print(f"Output missing: {result_nan}")
        print(f"Filled:         {nan_count - result_nan}")
        fill_rate = (nan_count - result_nan) / nan_count * 100 if nan_count > 0 else 0.0
        print(f"Success rate:   {fill_rate:.1f}%")
    print(f"{'='*60}\n")

    return inference_info


# ============ Main ============

def upload_run_logs(s3: S3Client, local_run_dir: str, error_prefix: str):
    if not os.path.exists(local_run_dir):
        return
    for root, dirs, files in os.walk(local_run_dir):
        for fname in files:
            if fname.endswith(('.log', '.json')):
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, local_run_dir)
                s3.upload_file(fpath, f"{error_prefix}/{rel}")


def main():
    parser = argparse.ArgumentParser(description='WaveStitchPlus Pipeline')

    parser.add_argument('--mode', choices=['train', 'inference', 'full'], default='full')
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--input-s3-key', type=str, default=None)

    parser.add_argument('--time-col', type=str, default='time')
    parser.add_argument('--target-cols', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--window-size', type=int, default=32)
    parser.add_argument('--use-em', action='store_true')
    parser.add_argument('--em-iterations', type=int, default=5)
    parser.add_argument('--epochs-per-em', type=int, default=200)

    parser.add_argument('--model-version', type=str, default=None)
    parser.add_argument('--model-type', type=str, default='em', choices=['auto', 'em', 'standard'])
    parser.add_argument('--n-trials', type=int, default=1)
    parser.add_argument('--guidance-scale', type=float, default=0.1)
    parser.add_argument('--repaint-rounds', type=int, default=5)
    parser.add_argument('--clamp-mode', type=str, default='bounds', choices=['none', 'nonneg', 'bounds'])
    parser.add_argument('--ddim-steps', type=int, default=50)
    parser.add_argument('--use-ddpm', action='store_true')
    parser.add_argument('--bound-headroom', type=float, default=1.2)
    parser.add_argument('--nonneg-cols', type=str, nargs='*', default=None)
    parser.add_argument('--upper-bounds', type=str, default=None)
    parser.add_argument('--lower-bounds', type=str, default=None)
    parser.add_argument('--keep-workdir', action='store_true')

    args = parser.parse_args()

    if args.mode in ['train', 'full'] and not args.input_s3_key:
        parser.error('--input-s3-key is required for train/full mode')

    version = args.version or datetime.now().strftime('%Y%m%d_%H%M%S')
    s3 = S3Client()

    import torch
    print(f"\n{'='*60}")
    print(f"WaveStitch Pipeline - {args.mode.upper()}")
    print(f"{'='*60}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Version: {version}")
    if args.input_s3_key:
        print(f"Input:   s3://{s3.bucket}/{args.input_s3_key}")
    print(f"{'='*60}\n")

    results = {}
    train_run_dir = f"/tmp/wavestitchplus/{args.dataset_name}/{version}"
    inference_run_dir = f"/tmp/wavestitchplus/{args.dataset_name}/inference_{version}"

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
                repaint_rounds=args.repaint_rounds,
                clamp_mode=args.clamp_mode,
            )

        if args.mode in ['inference', 'full']:
            print(f"\n{'='*60}")
            print(f"INFERENCE PHASE")
            print(f"{'='*60}")
            model_ver = version if args.mode == 'full' else args.model_version
            effective_model_type = args.model_type
            if args.mode == 'full' and args.model_type == 'em' and not args.use_em:
                effective_model_type = 'standard'
                print('[INFERENCE] Adjusted model_type to standard because training did not use EM')

            results['inference'] = run_inference(
                s3=s3,
                dataset_name=args.dataset_name,
                model_version=model_ver,
                output_version=version,
                model_type=effective_model_type,
                n_trials=args.n_trials,
                guidance_scale=args.guidance_scale,
                repaint_rounds=args.repaint_rounds,
                clamp_mode=args.clamp_mode,
                ddim_steps=args.ddim_steps,
                use_ddpm=args.use_ddpm,
                bound_headroom=args.bound_headroom,
                nonneg_cols=args.nonneg_cols,
                upper_bounds=args.upper_bounds,
                lower_bounds=args.lower_bounds,
            )

        print(f"\n{'='*60}")
        print(f"✓ PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(json.dumps(results, indent=2, default=str))
        print(f"\n__RESULT_JSON__:{json.dumps(results, default=str)}")

    except Exception as exc:
        print(f"\n[ERROR] Pipeline failed: {exc}")
        print(f"[ERROR] Attempting to upload logs before cleanup...")
        try:
            if os.path.exists(train_run_dir):
                upload_run_logs(
                    s3,
                    train_run_dir,
                    f"wavestitchplus/{args.dataset_name}/{version}/error_logs/train",
                )
            if os.path.exists(inference_run_dir):
                upload_run_logs(
                    s3,
                    inference_run_dir,
                    f"wavestitchplus/{args.dataset_name}/{version}/error_logs/inference",
                )
            print(f"[ERROR] Logs uploaded to s3://{s3.bucket}/wavestitchplus/{args.dataset_name}/{version}/error_logs/")
        except Exception as upload_err:
            print(f"[ERROR] Could not upload logs: {upload_err}")

        if not args.keep_workdir:
            for run_dir in [train_run_dir, inference_run_dir]:
                if os.path.exists(run_dir):
                    shutil.rmtree(run_dir, ignore_errors=True)
                    print(f"[CLEANUP] Removed {run_dir}")
        else:
            for run_dir in [train_run_dir, inference_run_dir]:
                if os.path.exists(run_dir):
                    print(f"[DEBUG] Work dir preserved at: {run_dir}")
        raise

    else:
        if not args.keep_workdir:
            for run_dir in [train_run_dir, inference_run_dir]:
                if os.path.exists(run_dir):
                    shutil.rmtree(run_dir, ignore_errors=True)
                    print(f"\n[CLEANUP] Removed {run_dir}")
        else:
            for run_dir in [train_run_dir, inference_run_dir]:
                if os.path.exists(run_dir):
                    print(f"\n[DEBUG] Work dir preserved at: {run_dir}")


if __name__ == '__main__':
    main()
