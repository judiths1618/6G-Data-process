#!/usr/bin/env bash
# Build the PyPOTS GPU baseline image. Run from dockers/tools/.
set -euo pipefail
sudo docker build --no-cache -f Dockerfile.pypots -t pypots-baseline:latest .
