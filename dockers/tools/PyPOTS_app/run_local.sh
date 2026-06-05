#!/usr/bin/env bash
# Run PyPOTS methods across the four EUR subsets.
# Fits each model on train.csv (target columns), then imputes BOTH train.csv
# and test_input.csv. Outputs land in experiments/EUR/generated_<subset>/.
#
# Use:  conda activate myenv && bash run_local.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/experiments/EUR}"
SUBSETS=(amf golang python rabbitmq)
METHODS=(saits brits)
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"

for subset in "${SUBSETS[@]}"; do
  prepared="$WORK_ROOT/prepared_${subset}"
  generated="$WORK_ROOT/generated_${subset}"
  [[ -d "$prepared" ]] || { echo "skip: $prepared not found"; continue; }
  mkdir -p "$generated"
  for m in "${METHODS[@]}"; do
    echo ">>> PyPOTS/$m on $subset"
    # Persist the trained model under generated_<subset>/saved_models/ — the
    # WaveStitch+-style train-once / reuse pattern. ``|| echo`` keeps the loop
    # going if one method fails (e.g. SAITS OOM on the largest subset).
    python "$(dirname "$0")/run_imputation.py" \
      --prepared-dir "$prepared" \
      --output-dir   "$generated" \
      --method       "$m" \
      --epochs       "$EPOCHS" \
      --batch-size   "$BATCH_SIZE" \
      --save-model \
      --model-path   "$generated/saved_models" \
      || echo "  (PyPOTS/$m on $subset failed, continuing)"
  done
done
