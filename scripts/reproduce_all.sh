#!/usr/bin/env bash
# Reproduce the local DataOps + imputation results from a clean artifact state.
#
# Usage:
#   conda activate wavestitchplus-repro
#   bash scripts/reproduce_all.sh smoke
#   bash scripts/reproduce_all.sh full
#
# smoke: CPU-friendly verification on golang/python/rabbitmq with the core methods.
# full:  all bundled performance datasets (incl. AMF) and every runner method; GPU recommended.
#
# Tracks (printed as "== Track X: ... =="):
#   A  cleaning         raw CSV -> remediated -> regularized + built-in imputation
#   B  baselines        shared-holdout MAE/RMSE table on $BENCH_DATASET
#   C  WaveStitch+      v1 -> v2 -> harpoon (runs first among the model runners; WSP_ONLY stops here)
#   D  Darts            per-column interpolation / kalman
#   E  other libraries  ImputeGAP + PyPOTS
#
# Optional overrides:
#   DATASETS="amf-performance rabbitmq-performance" bash scripts/reproduce_all.sh full
#   STRICT=1 bash scripts/reproduce_all.sh full      # abort on first method failure
#   PYPOTS_EPOCHS=50 WINDOW=100 bash scripts/reproduce_all.sh full
#   WSP_ONLY=1 DATASETS="amf-performance" bash scripts/reproduce_all.sh smoke
#   WSP_METHODS="v1 v2 harpoon" bash scripts/reproduce_all.sh smoke
set -euo pipefail

MODE="${1:-smoke}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "usage: bash scripts/reproduce_all.sh [smoke|full]" >&2
  exit 2
fi

cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
APP="dockers/tools/WaveStitchPlus_app"
STRICT="${STRICT:-0}"
WSP_ONLY="${WSP_ONLY:-0}"
FAILED=()

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/wavestitchplus-mpl}"
export PYKEOPS_CACHE_FOLDER="${PYKEOPS_CACHE_FOLDER:-/tmp/wavestitchplus-keops}"
export KEOPS_CACHE_FOLDER="${KEOPS_CACHE_FOLDER:-$PYKEOPS_CACHE_FOLDER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/wavestitchplus-cache}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
mkdir -p "$MPLCONFIGDIR" "$PYKEOPS_CACHE_FOLDER" "$KEOPS_CACHE_FOLDER" "$XDG_CACHE_HOME"

if [[ "$MODE" == "smoke" ]]; then
  DEFAULT_DATASETS="amf-performance golang-web-server-performance python-web-server-performance rabbitmq-performance"
  DARTS_METHODS="${DARTS_METHODS:-linear}"
  IMPUTEGAP_METHODS="${IMPUTEGAP_METHODS:-knn}"
  PYPOTS_METHODS="${PYPOTS_METHODS:-saits brits}"
  PYPOTS_EPOCHS="${PYPOTS_EPOCHS:-1}"
  WSP_METHODS="${WSP_METHODS:-v1 v2 harpoon}"
  WSP_ARGS=(--fast --device cpu)
  HARPOON_ARGS=(--ddim-steps 5 --repaint-rounds 1 --device cpu)
else
  DEFAULT_DATASETS="amf-performance golang-web-server-performance python-web-server-performance rabbitmq-performance"
  DARTS_METHODS="${DARTS_METHODS:-auto cubic kalman linear nearest quadratic slinear zero}"
  IMPUTEGAP_METHODS="${IMPUTEGAP_METHODS:-}"
  PYPOTS_METHODS="${PYPOTS_METHODS:-saits brits transformer gpvae mrnn csdi usgan timesnet}"
  PYPOTS_EPOCHS="${PYPOTS_EPOCHS:-15}"
  WSP_METHODS="${WSP_METHODS:-v1 v2 harpoon}"
  WSP_ARGS=(--em-iterations "${EM_ITERS:-5}" --epochs-per-em "${EPOCHS:-200}" \
            --ddim-steps "${DDIM:-50}" --repaint-rounds "${REPAINT:-5}" --device auto)
  HARPOON_ARGS=(--ddim-steps "${DDIM:-50}" --repaint-rounds "${REPAINT:-5}" --device auto)
fi

DATASETS="${DATASETS:-$DEFAULT_DATASETS}"
BENCH_DATASET="${BENCH_DATASET:-rabbitmq-performance}"
WINDOW="${WINDOW:-100}"

run_step() {
  local label="$1"
  shift
  echo "---- ${label}"
  if "$@"; then
    echo "  ok: ${label}"
  else
    local code=$?
    echo "  FAILED(${code}): ${label}" >&2
    FAILED+=("${label}")
    if [[ "$STRICT" == "1" ]]; then
      return "$code"
    fi
    return 0
  fi
}

quiet_wsp() {
  "$@" 2> >(grep -v -E '^\[pyKeOps\] Warning|FutureWarning|UserWarning' >&2)
}

has_method() {
  local needle="$1"
  local method
  shift
  for method in "$@"; do
    [[ "$method" == "$needle" ]] && return 0
  done
  return 1
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    echo "  ok: ${label} produced ${path}"
  else
    echo "  FAILED: ${label} did not produce ${path}" >&2
    FAILED+=("${label}/missing_output")
    if [[ "$STRICT" == "1" ]]; then
      return 1
    fi
    return 0
  fi
}

run_wavestitchplus_methods() {
  local name="$1"
  local B="$2"
  local G="$3"
  local methods=($WSP_METHODS)

  echo "== ${name}: WaveStitch+ (${WSP_METHODS}) =="

  if has_method "v1" "${methods[@]}"; then
    run_step "wavestitchplus/${name}/v1" \
      quiet_wsp "$PY" "$APP/run_imputation.py" \
        --prepared-dir "$B" --output-dir "$G" "${WSP_ARGS[@]}"
    require_file "wavestitchplus/${name}/v1/test" "$G/wavestitchplus_v1_test_imputed.csv"
    require_file "wavestitchplus/${name}/v1/final" "$G/wavestitchplus_v1_final.csv"
  elif [[ ! -f "$G/wavestitchplus_v1_test_imputed.csv" ]]; then
    echo "skip WaveStitch+ v2/harpoon for ${name}: WSP_METHODS excludes v1 and no existing v1 test output is present" >&2
    FAILED+=("wavestitchplus/${name}/missing_v1")
    return 0
  fi

  if has_method "v2" "${methods[@]}"; then
    if [[ -f "$G/wavestitchplus_v1_test_imputed.csv" ]]; then
      run_step "wavestitchplus/${name}/v2" \
        quiet_wsp "$PY" "$APP/run_imputation_v2.py" \
          --prepared-dir "$B" --output-dir "$G" \
          --reuse-diffusion "$G/wavestitchplus_v1_test_imputed.csv"
      require_file "wavestitchplus/${name}/v2/test" "$G/wavestitchplus_v2_test_imputed.csv"
      require_file "wavestitchplus/${name}/v2/final" "$G/wavestitchplus_v2_final.csv"
    else
      echo "skip WaveStitch+ v2 for ${name}: missing $G/wavestitchplus_v1_test_imputed.csv" >&2
      FAILED+=("wavestitchplus/${name}/v2/missing_v1")
    fi
  fi

  if has_method "harpoon" "${methods[@]}"; then
    if [[ -f "$G/wavestitchplus_v1_test_imputed.csv" ]]; then
      run_step "wavestitchplus/${name}/harpoon" \
        quiet_wsp "$PY" "$APP/run_imputation_harpoon.py" \
          --prepared-dir "$B" --output-dir "$G" "${HARPOON_ARGS[@]}"
      require_file "wavestitchplus/${name}/harpoon/test" "$G/wavestitchplus_harpoon_test_imputed.csv"
      require_file "wavestitchplus/${name}/harpoon/final" "$G/wavestitchplus_harpoon_final.csv"
    else
      echo "skip WaveStitch+ harpoon for ${name}: missing $G/wavestitchplus_v1_test_imputed.csv" >&2
      FAILED+=("wavestitchplus/${name}/harpoon/missing_v1")
    fi
  fi

  if [[ -f "$G/wavestitchplus_v1_test_imputed.csv" ]]; then
    run_step "compare_wsp_v2/${name}" \
      "$PY" scripts/compare_wsp_v2.py \
        --prepared-dir "$B" \
        --baseline-dir "$G" \
        --v1-csv "$G/wavestitchplus_v1_test_imputed.csv" \
        --out-csv "$G/wsp_v2_comparison.csv"
  fi
}

resolve_imputegap_methods() {
  if [[ -n "$IMPUTEGAP_METHODS" ]]; then
    printf "%s\n" "$IMPUTEGAP_METHODS"
    return 0
  fi
  "$PY" dockers/tools/ImputeGAP_app/run_imputation.py \
    --prepared-dir . --output-dir . --list \
    | awk '/  - /{print $2}' \
    | tr '\n' ' '
}

echo "== environment =="
"$PY" --version
"$PY" - <<'PY'
import importlib
for name in ["torch", "darts", "pypots", "imputegap", "pandas", "numpy"]:
    mod = importlib.import_module(name)
    print(f"{name}=={getattr(mod, '__version__', 'ok')}")
PY

if [[ "$WSP_ONLY" != "1" ]]; then

echo "== Track A: raw CSVs -> data/processed/ with all built-in imputation =="
for name in $DATASETS; do
  run_step "pipeline/${name}" \
    "$PY" -m pipelines.minimal_dataops \
      --input "data/raw/${name}.csv" \
      --output "data/processed/${name}_remediated.csv" \
      --report "reports/${name}_report.json" \
      --log-file "logs/${name}-dataops.log"
  run_step "auto_impute_all/${name}" \
    "$PY" scripts/auto_impute.py \
      --report "reports/${name}_report.json" \
      --method all
done

echo "== Track B: shared-holdout baselines on ${BENCH_DATASET} =="
run_step "compare_baselines/${BENCH_DATASET}" \
  "$PY" scripts/compare_baselines.py \
    --input-csv "data/raw/${BENCH_DATASET}.csv" \
    --methods darts_linear,darts_nearest,pypots_saits \
    --run-id "reproduce_${MODE}"

fi

# Resolve which datasets have a prepared bundle once, so the "missing_bundle"
# skip is recorded a single time per dataset regardless of how many tracks run.
echo "== resolve prepared bundles -> data/processed/<name>_generated/ =="
VALID_DATASETS=""
for name in $DATASETS; do
  B="data/processed/${name}_regularized"
  G="data/processed/${name}_generated"
  if [[ ! -d "$B" ]]; then
    echo "skip imputation runners for ${name}: missing ${B}" >&2
    FAILED+=("missing_bundle/${name}")
    continue
  fi
  mkdir -p "$G"
  VALID_DATASETS="${VALID_DATASETS}${VALID_DATASETS:+ }${name}"
done

echo "== Track C: WaveStitch+ -> data/processed/<name>_generated/ =="
echo "WaveStitch+ methods: ${WSP_METHODS}"
for name in $VALID_DATASETS; do
  B="data/processed/${name}_regularized"
  G="data/processed/${name}_generated"
  run_wavestitchplus_methods "$name" "$B" "$G"
done

if [[ "$WSP_ONLY" != "1" ]]; then
  echo "== Track D: Darts -> data/processed/<name>_generated/ =="
  echo "Darts methods: ${DARTS_METHODS}"
  for name in $VALID_DATASETS; do
    B="data/processed/${name}_regularized"
    G="data/processed/${name}_generated"
    echo "== ${name}: Darts =="
    for method in $DARTS_METHODS; do
      run_step "darts/${name}/${method}" \
        "$PY" dockers/tools/Darts_app/run_imputation.py \
          --prepared-dir "$B" --output-dir "$G" --method "$method"
    done
  done

  echo "== Track E: other libraries (ImputeGAP + PyPOTS) -> data/processed/<name>_generated/ =="
  RESOLVED_IMPUTEGAP_METHODS="$(resolve_imputegap_methods)"
  echo "ImputeGAP methods: ${RESOLVED_IMPUTEGAP_METHODS}"
  echo "PyPOTS methods: ${PYPOTS_METHODS}"
  for name in $VALID_DATASETS; do
    B="data/processed/${name}_regularized"
    G="data/processed/${name}_generated"

    echo "== ${name}: ImputeGAP =="
    for method in $RESOLVED_IMPUTEGAP_METHODS; do
      run_step "imputegap/${name}/${method}" \
        "$PY" dockers/tools/ImputeGAP_app/run_imputation.py \
          --prepared-dir "$B" --output-dir "$G" --method "$method"
    done

    echo "== ${name}: PyPOTS =="
    for method in $PYPOTS_METHODS; do
      run_step "pypots/${name}/${method}" \
        "$PY" dockers/tools/PyPOTS_app/run_imputation.py \
          --prepared-dir "$B" --output-dir "$G" \
          --method "$method" --window "$WINDOW" --epochs "$PYPOTS_EPOCHS"
    done
  done
fi

echo "== final artifact sanity =="
"$PY" - <<'PY'
from pathlib import Path
import pandas as pd

for path in sorted(Path("data/processed").glob("*_final.csv")):
    df = pd.read_csv(path)
    print(f"{path}: shape={df.shape} nan={int(df.isna().sum().sum())}")

for path in sorted(Path("data/processed").glob("*_generated/*_final.csv")):
    df = pd.read_csv(path)
    print(f"{path}: shape={df.shape} nan={int(df.isna().sum().sum())}")
PY

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "== completed with ${#FAILED[@]} failed/skipped step(s) ==" >&2
  for item in "${FAILED[@]}"; do
    echo "  - $item" >&2
  done
  if [[ "$STRICT" == "1" ]]; then
    exit 1
  fi
else
  echo "== all requested steps completed =="
fi
