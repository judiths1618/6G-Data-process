#!/usr/bin/env bash
# Run Darts imputation methods across the four EUR subsets.
# Imputes both train.csv and test_input.csv and writes outputs into
# experiments/EUR/generated_<subset>/.
#
# Use:  conda activate myenv && bash run_local.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/experiments/EUR}"
SUBSETS=(amf golang python rabbitmq)
METHODS=(linear cubic nearest kalman auto)

for subset in "${SUBSETS[@]}"; do
  prepared="$WORK_ROOT/prepared_${subset}"
  generated="$WORK_ROOT/generated_${subset}"
  [[ -d "$prepared" ]] || { echo "skip: $prepared not found"; continue; }
  mkdir -p "$generated"
  for m in "${METHODS[@]}"; do
    echo ">>> Darts/$m on $subset"
    python "$(dirname "$0")/run_imputation.py" \
      --prepared-dir "$prepared" \
      --output-dir   "$generated" \
      --method       "$m"
  done
done
