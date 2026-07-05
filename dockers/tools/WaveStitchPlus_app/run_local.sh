#!/usr/bin/env bash
# Native (no-Docker) WaveStitch+ launcher.
#
# Trains + synthesizes locally using the existing prepared_<subset>/ folders
# under experiments/EUR/.  PyTorch picks CPU automatically when no CUDA
# device is available, so this script works on Mac/Linux without a GPU.
#
# Use:
#   conda activate myenv
#   bash dockers/tools/WaveStitchPlus_app/run_local.sh                   # all subsets
#   SUBSETS=python bash .../run_local.sh                                 # one subset
#   SKIP_PREPROCESS=1 bash .../run_local.sh                              # reuse existing prepared_<subset>/
#   FAST=1 bash .../run_local.sh                                         # CPU-friendly tiny hyperparams
#
# Environment knobs (all optional):
#   SUBSETS          space-separated list (default: "python")
#   SKIP_PREPROCESS  if set, reuse the existing prepared_<subset>/ instead of
#                    re-running preprocess from the raw CSV. Useful when the
#                    raw CSV is unavailable; the in-train fallback computes
#                    iqr/1.349 on the fly so the scaler fix still applies.
#   FAST             if set, use tiny hyperparams (em=2, epochs=30, ddim=20)
#                    for a quick CPU smoke test.
#   EM_ITERS, EPOCHS_PER_EM, DDIM_STEPS, REPAINT_ROUNDS
#                    override individual hyperparams.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
APP_DIR="$REPO_ROOT/dockers/tools/WaveStitchPlus_app"
WORK_ROOT="${WORK_ROOT:-$REPO_ROOT/experiments/EUR}"
RAW_ROOT="${RAW_ROOT:-$REPO_ROOT/6GDALI_Datasets/EUR/6907619}"

# Bash 3.2 (macOS default) doesn't have associative arrays — use a function.
raw_csv_for() {
  case "$1" in
    amf)      echo "amf-performance.csv" ;;
    golang)   echo "golang-web-server-performance.csv" ;;
    python)   echo "python-web-server-performance.csv" ;;
    rabbitmq) echo "rabbitmq-performance.csv" ;;
    *)        echo "" ;;
  esac
}

SUBSETS="${SUBSETS:-python}"

if [[ -n "${FAST:-}" ]]; then
  EM_ITERS="${EM_ITERS:-2}"
  EPOCHS_PER_EM="${EPOCHS_PER_EM:-30}"
  DDIM_STEPS="${DDIM_STEPS:-20}"
  REPAINT_ROUNDS="${REPAINT_ROUNDS:-3}"
else
  EM_ITERS="${EM_ITERS:-5}"
  EPOCHS_PER_EM="${EPOCHS_PER_EM:-200}"
  DDIM_STEPS="${DDIM_STEPS:-50}"
  REPAINT_ROUNDS="${REPAINT_ROUNDS:-5}"
fi

cd "$APP_DIR"

for subset in $SUBSETS; do
  prepared="$WORK_ROOT/prepared_${subset}"
  generated="$WORK_ROOT/generated_${subset}"
  mkdir -p "$prepared" "$generated"

  echo "============================================================"
  echo "WaveStitch+ on subset=$subset  (CPU/auto)"
  echo "  prepared_dir=$prepared"
  echo "  generated_dir=$generated"
  echo "  hyperparams: em=$EM_ITERS  epochs/em=$EPOCHS_PER_EM  ddim=$DDIM_STEPS  repaint=$REPAINT_ROUNDS"
  echo "============================================================"

  TRAIN_ARGS=(
    -d custom_csv
    -prepared_dir "$prepared"
    -repaint_rounds "$REPAINT_ROUNDS"
    -save_train_imputed_denorm
    -train_imputed_clamp bounds
    -use_em
    -em_iterations "$EM_ITERS"
    -epochs_per_em "$EPOCHS_PER_EM"
    -ddim_steps "$DDIM_STEPS"
  )

  if [[ -z "${SKIP_PREPROCESS:-}" ]]; then
    raw="$(raw_csv_for "$subset")"
    if [[ -n "$raw" && -f "$RAW_ROOT/$raw" ]]; then
      TRAIN_ARGS+=( -input_csv "$RAW_ROOT/$raw" )
      echo "  raw csv=$RAW_ROOT/$raw  (re-preprocess to pick up the robust_std stats)"
    else
      echo "  raw csv not found at $RAW_ROOT/$raw — falling back to SKIP_PREPROCESS"
    fi
  else
    echo "  SKIP_PREPROCESS=1 → reusing existing prepared/ (in-train fallback computes iqr/1.349)"
  fi

  echo ">>> Training"
  python "$APP_DIR/train_improved.py" "${TRAIN_ARGS[@]}" 2>&1 | tail -40

  if [[ -d "$prepared/saved_model" ]]; then
    model_group="$generated/saved_models/wavestitchplus/full"
    mkdir -p "$model_group"
    cp "$prepared"/saved_model/*.pth "$model_group"/ 2>/dev/null || true
    echo "  -> grouped model artifacts under $model_group"
  fi

  if [[ -f "$prepared/train_imputed_denorm.csv" ]]; then
    train_out_csv="$generated/wavestitchplus_v1_train_imputed.csv"
    cp "$prepared/train_imputed_denorm.csv" "$train_out_csv"
    echo "  → $train_out_csv"
  fi

  echo ">>> Synthesis"
  out_csv="$generated/wavestitchplus_v1_test_imputed.csv"
  python "$APP_DIR/synthesis_improved.py" \
    -d custom_csv \
    -prepared_dir "$prepared" \
    -out_csv "$out_csv" \
    -model_type em \
    -clamp_mode bounds \
    -repaint_rounds "$REPAINT_ROUNDS" \
    -guidance_scale 0.1 \
    -n_trials 1 \
    -ddim_steps "$DDIM_STEPS" \
    -bound_headroom 1.2 2>&1 | tail -20

  echo "  → $out_csv"

  echo ">>> Final (imputed train + imputed test)"
  python "$APP_DIR/wsp_final.py" \
    --prepared-dir "$prepared" \
    --output-dir "$generated" \
    --variant v1 2>&1 | tail -3
done
