#!/usr/bin/env bash
# Launch the DataOps cleaning-first dashboard.
# Use:  conda activate myenv && bash dashboard/run.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-8502}"
streamlit run "$REPO_ROOT/dashboard/app.py" --server.port "$PORT" --server.headless false
