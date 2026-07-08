#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# End-to-end reproduction on a (GPU) server, one command after clone.
#
#   conda activate autofeat-6g          # see README → Reproduce → step 0
#   bash scripts/gpu_reproduce.sh
#
# Runs: DataOps pipeline (stage-based) + final  →  every imputation method into
# data/processed/<name>_generated/ at FULL hyperparameters  →  long-gap depth
# eval on python  →  a v2-vs-nearest summary. Everything auto-uses CUDA when
# torch sees it (no code change). Error-tolerant: one method failing does not
# abort the rest.
#
# Quick smoke run (minutes, CPU-friendly):
#   EM_ITERS=2 EPOCHS=30 DDIM=20 REPAINT=3 PYPOTS_EPOCHS=15 bash scripts/gpu_reproduce.sh
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."                 # repo root

PY="${PYTHON:-python}"
APP=dockers/tools/WaveStitchPlus_app
EUR="amf-performance golang-web-server-performance python-web-server-performance rabbitmq-performance"
KUL="user_0_sample_0_antenna_0"

# Full hyperparameters (override via env for a quicker run).
EM_ITERS="${EM_ITERS:-5}"; EPOCHS="${EPOCHS:-200}"; DDIM="${DDIM:-50}"; REPAINT="${REPAINT:-5}"
PYPOTS_EPOCHS="${PYPOTS_EPOCHS:-100}"; WINDOW="${WINDOW:-100}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

echo "############################################################"
$PY -c "import torch; print('device:', 'CUDA '+torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (no GPU visible)')"
echo "hyperparams: em=$EM_ITERS epochs/em=$EPOCHS ddim=$DDIM repaint=$REPAINT | pypots epochs=$PYPOTS_EPOCHS window=$WINDOW"
echo "############################################################"

echo; echo "########## 1) DataOps pipeline (stage-based) + final ##########"
for name in $EUR $KUL; do
  echo "----- $name -----"
  $PY -m pipelines.minimal_dataops --input "data/raw/${name}.csv" \
      --output "data/processed/${name}_remediated.csv" \
      --report "reports/${name}_report.json" 2>&1 | tail -1
  $PY scripts/auto_impute.py --report "reports/${name}_report.json" --method all 2>&1 | grep -E "FINAL|Nothing|canonical" || true
done

echo; echo "########## 2) all imputation methods → <name>_generated (full HP) ##########"
for name in $EUR; do
  B="data/processed/${name}_regularized"; G="data/processed/${name}_generated"
  echo "===== $name ====="
  for m in linear nearest cubic auto kalman; do
    $PY dockers/tools/Darts_app/run_imputation.py --prepared-dir "$B" --output-dir "$G" --method "$m" \
      >/dev/null 2>&1 && echo "  darts/$m ok" || echo "  darts/$m FAILED"
  done
  for m in mean_by_series interpolation knn; do   # mean_by_series = per-column mean (NOT the global 'mean')
    $PY dockers/tools/ImputeGAP_app/run_imputation.py --prepared-dir "$B" --output-dir "$G" --method "$m" \
      >/dev/null 2>&1 && echo "  imputegap/$m ok" || echo "  imputegap/$m FAILED"
  done
  for m in saits brits; do
    $PY dockers/tools/PyPOTS_app/run_imputation.py --prepared-dir "$B" --output-dir "$G" \
      --method "$m" --window "$WINDOW" --epochs "$PYPOTS_EPOCHS" \
      >/dev/null 2>&1 && echo "  pypots/$m ok" || echo "  pypots/$m FAILED"
  done
  # WaveStitch+ v1 (train+synth) → v2 (auto per-column anchor) → harpoon
  $PY "$APP/run_imputation.py" --prepared-dir "$B" --output-dir "$G" \
      --em-iterations "$EM_ITERS" --epochs-per-em "$EPOCHS" --ddim-steps "$DDIM" --repaint-rounds "$REPAINT" \
      >/dev/null 2>&1 && echo "  wsp/v1 ok" || echo "  wsp/v1 FAILED"
  $PY "$APP/run_imputation_v2.py" --prepared-dir "$B" --output-dir "$G" \
      --reuse-diffusion "$G/wavestitchplus_v1_test_imputed.csv" \
      >/dev/null 2>&1 && echo "  wsp/v2 ok" || echo "  wsp/v2 FAILED"
  $PY "$APP/run_imputation_harpoon.py" --prepared-dir "$B" --output-dir "$G" \
      --ddim-steps "$DDIM" --repaint-rounds "$REPAINT" \
      >/dev/null 2>&1 && echo "  wsp/harpoon ok" || echo "  wsp/harpoon FAILED"
done

echo; echo "########## 3) long-gap depth eval (python; does diffusion beat interp at depth?) ##########"
# run_imputation.py cleans the checkpoint from the bundle, so retrain here KEEPING it, then carve gaps.
PB=data/processed/python-web-server-performance_regularized
$PY "$APP/train_improved.py" -d custom_csv -prepared_dir "$PB" \
    -use_em -em_iterations "$EM_ITERS" -epochs_per_em "$EPOCHS" -ddim_steps "$DDIM" -repaint_rounds "$REPAINT" \
    -save_train_imputed_denorm -train_imputed_clamp bounds 2>&1 | grep -E "M-step|EM iter|DONE" || echo "  (retrain failed)"
$PY scripts/eval_long_gap.py --prepared-dir "$PB" --gap-lengths 4,8,16,32,64 --ddim-steps "$DDIM" 2>&1 | tail -30 || echo "  (long-gap failed)"

echo; echo "########## 4) summary: v2(auto) vs nearest vs v1, per dataset ##########"
$PY - <<'PYEOF' 2>/dev/null || true
import json, numpy as np, pandas as pd
def mae(ti, gt, imp, tc):
    ia=ti[tc].to_numpy(float); ga=gt[tc].to_numpy(float); pa=imp[tc].to_numpy(float)
    m=np.isnan(ia)&~np.isnan(ga)&~np.isnan(pa)
    return float(np.mean(np.abs(pa[m]-ga[m]))) if m.any() else float("nan")
print(f"{'dataset':<26}{'nearest':>11}{'v2(auto)':>11}{'ratio':>7}{'v1':>12}")
for name in ["amf-performance","golang-web-server-performance","python-web-server-performance","rabbitmq-performance"]:
    B=f"data/processed/{name}_regularized"; G=f"data/processed/{name}_generated"
    try:
        tc=json.load(open(f"{B}/meta.json"))["target_cols"]
        ti=pd.read_csv(f"{B}/test_input.csv"); gt=pd.read_csv(f"{B}/test_gt.csv")
        mn=mae(ti,gt,pd.read_csv(f"{G}/darts_nearest_test_imputed.csv"),tc)
        v2=mae(ti,gt,pd.read_csv(f"{G}/wavestitchplus_v2_test_imputed.csv"),tc)
        v1=mae(ti,gt,pd.read_csv(f"{G}/wavestitchplus_v1_test_imputed.csv"),tc)
        print(f"{name:<26}{mn:>11.4g}{v2:>11.4g}{v2/mn:>7.3f}{v1:>12.4g}")
    except Exception as e:
        print(f"{name:<26}  (missing outputs: {e})")
print("ratio<1 → v2 beats nearest; v2 should be << v1")
PYEOF

echo; echo "########## DONE — explore in the dashboard:  bash dashboard/run.sh  (→ http://localhost:8502) ##########"
