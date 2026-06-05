#!/usr/bin/env bash
# Run ImputeGAP methods across the four EUR subsets.
# Imputes both train.csv and test_input.csv into experiments/EUR/generated_<subset>/.
#
# Use:  conda activate myenv && bash run_local.sh
#
# Notes on this conda env (myenv, imputegap 1.1.1):
#   - cdrec / stmvl need the native armadillo shared lib, which is NOT installed.
#     The runner gracefully skips a method that fails; we just omit them here.
#   - miss_forest fails on subsets where some target columns are entirely NaN.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/experiments/EUR}"
SUBSETS=(amf golang python rabbitmq)
METHODS=(mean interpolation knn iim)

for subset in "${SUBSETS[@]}"; do
  prepared="$WORK_ROOT/prepared_${subset}"
  generated="$WORK_ROOT/generated_${subset}"
  [[ -d "$prepared" ]] || { echo "skip: $prepared not found"; continue; }
  mkdir -p "$generated"
  for m in "${METHODS[@]}"; do
    echo ">>> ImputeGAP/$m on $subset"
    python "$(dirname "$0")/run_imputation.py" \
      --prepared-dir "$prepared" \
      --output-dir   "$generated" \
      --method       "$m" || echo "  (method $m failed on $subset, continuing)"
  done
done
