#!/usr/bin/env bash
# Build the Darts baseline image. Run from dockers/tools/.
set -euo pipefail
sudo docker build --no-cache -f Dockerfile.darts -t darts-baseline:latest .
