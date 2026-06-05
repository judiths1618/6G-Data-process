#!/usr/bin/env bash
# Build the ImputeGAP baseline image. Run from dockers/tools/.
set -euo pipefail
sudo docker build --no-cache -f Dockerfile.imputegap -t imputegap-baseline:latest .
