#!/usr/bin/env bash
# Build the CPU WaveStitch+ image (no GPU / nvidia-docker required — works on a Mac).
# Tag: wavestitchplus-cpu:latest  → the dashboard's default for the WaveStitch+ methods.
#
# Build context is this directory (dockers/tools); Dockerfile.wavestitchplus-cpu
# COPYs requirements.txt, WaveStitchPlus_app/, and scripts/run_pipeline.py from here.
#
# It also bundles the canonical preprocess helper that the WaveStitchPlus shim
# (custom_pipeline/preprocess.py) needs at runtime. That helper lives outside this
# build context (dockers/airflow/dags/helpers/preprocess.py), so we vendor a copy
# into ./_wsp_helpers/ for the build and remove it afterwards.
set -euo pipefail
cd "$(dirname "$0")"

HELPER_SRC="../airflow/dags/helpers/preprocess.py"
if [[ ! -f "$HELPER_SRC" ]]; then
    echo "ERROR: canonical helper not found at $HELPER_SRC" >&2
    exit 1
fi
mkdir -p _wsp_helpers
cp "$HELPER_SRC" _wsp_helpers/preprocess.py
trap 'rm -rf _wsp_helpers' EXIT

docker build -f Dockerfile.wavestitchplus-cpu -t wavestitchplus-cpu:latest .
echo "Built wavestitchplus-cpu:latest"
