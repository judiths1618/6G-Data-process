#!/usr/bin/env bash
set -euo pipefail

# =========================
# Global settings
# =========================
DATA_ROOT="/home/Yuandou/Desktop/projects/6G-Data-process/6GDALI_Datasets/EUR/6907619"
WORK_ROOT="./work"
GROUP="EUR"

USE_EM=true
EM_ITERATIONS=5
EPOCHS_PER_EM=200
REPAINT_ROUNDS=5

MODEL_TYPE="em"
GUIDANCE_SCALE="0.1"   # e.g. 0.1
N_TRIALS="1"         # e.g. 3

# =========================
# Dataset list
# Format: name:input_csv
# =========================
DATASETS=(
  # "python:${DATA_ROOT}/python-web-server-performance.csv"
  # "golang:${DATA_ROOT}/golang-web-server-performance.csv"
  # "amf:${DATA_ROOT}/amf-performance.csv"
  "rabbitmq:${DATA_ROOT}/rabbitmq-performance.csv"
)

# =========================
# Main loop
# =========================
for item in "${DATASETS[@]}"; do
  NAME="${item%%:*}"
  INPUT_CSV="${item#*:}"

  PREPARED_DIR="${WORK_ROOT}/${GROUP}/prepared_${NAME}"
  GENERATED_DIR="${WORK_ROOT}/${GROUP}/generated_${NAME}"
  OUT_CSV="${GENERATED_DIR}/wavestitchPlus_full_imputed.csv"

  mkdir -p "${PREPARED_DIR}"
  mkdir -p "${GENERATED_DIR}"

  echo "=========================================="
  echo "Processing dataset: ${GROUP}/${NAME}"
  echo "INPUT_CSV    = ${INPUT_CSV}"
  echo "PREPARED_DIR = ${PREPARED_DIR}"
  echo "OUT_CSV      = ${OUT_CSV}"
  echo "=========================================="

  # -------------------------
  # Step 1: train / prepare
  # -------------------------
  TRAIN_CMD=(
    python train_improved.py
    -d custom_csv
    -input_csv "${INPUT_CSV}"
    -prepared_dir "${PREPARED_DIR}"
    -repaint_rounds "${REPAINT_ROUNDS}"
    -save_train_imputed_denorm \
    -train_imputed_clamp bounds \
  )

  if [ "${USE_EM}" = true ]; then
    TRAIN_CMD+=(
      -use_em
      -em_iterations "${EM_ITERATIONS}"
      -epochs_per_em "${EPOCHS_PER_EM}"
    )
  fi

  echo "[Running train] ${TRAIN_CMD[*]}"
  "${TRAIN_CMD[@]}"

  # -------------------------
  # Step 2: synthesis / imputation
  # -------------------------
  SYNTH_CMD=(
    python synthesis_improved.py \
    -d custom_csv \
    -prepared_dir "${PREPARED_DIR}"
    -out_csv "${OUT_CSV}"
    -model_type "${MODEL_TYPE}"
    -clamp_mode bounds \
  )
  if [ -n "${GUIDANCE_SCALE}" ]; then
    SYNTH_CMD+=(-guidance_scale "${GUIDANCE_SCALE}")
  fi

  if [ -n "${N_TRIALS}" ]; then
    SYNTH_CMD+=(-n_trials "${N_TRIALS}")
  fi

  echo "[Running synthesis] ${SYNTH_CMD[*]}"
  "${SYNTH_CMD[@]}"

done

echo "All datasets finished successfully."