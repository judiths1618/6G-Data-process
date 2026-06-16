#!/usr/bin/env bash
# Build the GPU WaveStitch+ image (CUDA 11.8). Run on a host with nvidia-docker.
#     bash build_image.sh            # from dockers/tools/
# On a Linux GPU host you may need `sudo` (or membership in the `docker` group).
#
# Bundles the canonical preprocess helper that the WaveStitchPlus shim
# (custom_pipeline/preprocess.py) needs at runtime. That helper lives outside this
# build context (dockers/airflow/dags/helpers/preprocess.py), so we vendor a copy
# into ./_wsp_helpers/ for the build and remove it afterwards (same as build_image_cpu.sh).
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

# NOTE: the previous `docker system prune -a -f` was removed — it deletes ALL
# images not used by a running container (a global footgun). Reclaim space
# manually if needed, e.g.  docker image prune   (dangling layers only).
docker build --no-cache -f Dockerfile.wavestitchplus-gpu -t wavestitchplus-gpu:latest .
echo "Built wavestitchplus-gpu:latest"
