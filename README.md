# WaveStitchPlus

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?style=flat-square&logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?style=flat-square&logo=nvidia)
![Airflow](https://img.shields.io/badge/Apache%20Airflow-2.9.3-017CEE?style=flat-square&logo=apacheairflow)
![SeaweedFS](https://img.shields.io/badge/SeaweedFS-S3%20API-46A2F1?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

This repository provides an end-to-end implementation for (time-series) data cleaning, validation, remediation, and imputation handoff using Apache Airflow, SeaweedFS, and ML models.

## Highlights

**Time-series data cleaning & AI-assisted imputation** — `Python · PyTorch · Apache Airflow · SeaweedFS · Docker`

- **WaveStitch+**, a **diffusion-based imputation model** for 5G/6G telemetry (RePaint + Hann-weighted
  windowing + DDIM sampling + EM training), benchmarked against Darts, ImputeGAP, and PyPOTS. The inference-time **v2** local-anchoring layer substantially reduces holdout MAE versus raw v1 with no retraining and — anchored to a per-column **`auto`** interpolation prior — **matches or beats the strong `nearest` baseline** across the benchmark subsets, while reserving its diffusion contribution for genuinely long gaps.
- A **reproducible, end-to-end data-cleaning pipeline** (validation → remediation → imputation) with
  GitHub Actions CI, `pytest`, DVC versioning, and a Streamlit **monitoring dashboard** that makes every
  run stage-by-stage auditable.

Jump to **[Reproduce the results](#reproduce-the-results)** for a clean-room, copy-paste run.

## Table of Contents

- [Overview](#overview)
- [Reproduce the results](#reproduce-the-results)
- [Configuration & data lineage](#configuration--data-lineage)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Pipeline Modules](#pipeline-modules)
- [Baseline Imputation Methods](#baseline-imputation-methods)
- [WaveStitch+ v2 & Method Comparison](#wavestitch-v2--method-comparison)
- [Comparison Dashboard](#comparison-dashboard)
- [Testing](#testing)
- [Pipeline Artifacts on S3](#pipeline-artifacts-on-s3)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

Diffusion model-based time-series imputation: **WaveStitch+** extends WaveStitch[add reference] by combining **RePaint**, **Hann-weighted windowing**, and **DDIM sampling** with an **Expectation-Maximization (EM)** training loop. It repairs conditional gaps in test-time series, even when the training data also contains missing values. A lightweight **v2** variant adds inference-time **local anchoring** of the diffusion output, recovering interpolation-grade accuracy on short gaps while preserving the diffusion model's advantage deep inside long gaps. This repository packages WaveStitch+ as a reusable and reproducible component in an **end-to-end data-cleaning pipeline**, adopted for 5G/6G time-series data.

Architecture overview with WaveStitch+:

![Architecture overview](design/Arch.png)

> Full-resolution diagram: [design/Arch.pdf](design/Arch.pdf)



The system has four loosely coupled layers:

| Layer | What it does | Tech |
|---|---|---|
| **Orchestration** | Profiles the input, branches on time-series vs. tabular data, runs duplicate, missing-value, outlier, and time-series gap checks, validates with Great Expectations, and persists curated outputs. | Apache Airflow (DAG `data_quality_and_cleaning_pipeline`) |
| **Cleaning engine** | Launches the WaveStitch+ imputation container when time-series repair is recommended. It runs `train`, `inference`, or `full` (train + inference) and stores artifacts back in the data lake. | PyTorch / CUDA, packaged as `wavestitchplus-gpu:latest` |
| **Reusable modules** | Provides library-first profiling, quality-check, transform, and split components for local Python, Airflow, or external orchestrators. | `src/data_process_modules/` (`src/dataops/` is kept as a compatibility API) |
| **Storage** | Provides an S3-compatible object store used as a versioned data lake (`raw/`, `prepared/`, `inference_results_*/`, `latest_inference/`). | SeaweedFS (master / filer / volume / S3) + Postgres for Airflow metadata |

---

## Reproduce the results

Complete, copy-paste instructions to reproduce the local pipeline and imputation comparisons
from a fresh clone. Four sample datasets ship in [`data/raw/`](data/raw/), so the local
tracks run **without Docker or external data**. **Track A** runs in the plain `.venv` from
step 0. **Tracks B–E** call the real Darts / ImputeGAP / PyPOTS / PyTorch libraries, so use
the dedicated conda environment below. The **Airflow + SeaweedFS stack** (final section) adds
the full Docker service stack.

The reproduction is organized into five tracks — the same labels `scripts/reproduce_all.sh`
prints — plus two optional sections:

| Track | Section | What runs |
|---|---|---|
| **A** | [A. Raw CSVs → `data/processed/`](#a-raw-csvs--dataprocessed-results) | cleaning + built-in imputation |
| **B** | [B. Baseline benchmark](#b-baseline-benchmark-on-a-shared-holdout-cpu) | shared-holdout MAE/RMSE table |
| **C** | [C. WaveStitch+ v1 → v2](#c-wavestitch-v1--v2-and-the-top-2-result-cpu-or-gpu) | diffusion runners (first model track) |
| **D** | [D. Darts](#d-darts-runners) | interpolation / kalman |
| **E** | [E. Other libraries](#e-other-library-runners--imputegap--pypots) | ImputeGAP + PyPOTS |
| — | [Dashboard](#explore-every-run-in-the-dashboard) · [Airflow stack](#full-airflow--seaweedfs-stack-docker-optional) | explore + orchestrate (optional) |

### 0. Clone and install

```bash
git clone https://github.com/judiths1618/WaveStitchPlus.git
cd WaveStitchPlus

python -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -m pytest                              # sanity check: unit tests pass
```

The plain `.venv` above covers **Track A** (dependency-free). For **Tracks B–E** (real
Darts / ImputeGAP / PyPOTS / PyTorch), and especially a **GPU server**, use a clean conda env
that installs the pinned all-method stack. Avoid reusing a broad research env: `pip check`
should be clean in the dedicated env, while mixed envs can silently carry incompatible
transitive packages.

```bash
git clone https://github.com/judiths1618/WaveStitchPlus.git && cd WaveStitchPlus

conda env create -f environment.yml
conda activate wavestitchplus-repro

# equivalent manual setup:
# conda create -y -n wavestitchplus-repro python=3.10 && conda activate wavestitchplus-repro
# python -m pip install -r requirements.txt
# python -m pip install -e ".[dev]"

# confirm the GPU is visible (WaveStitch+ & PyPOTS auto-use it — no code change)
python -c "import torch; print('CUDA', torch.cuda.is_available(), \
  torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python -m pip check
python -m pytest tests dockers/tools    # full suite (CPU-only; ~seconds)
```

The all-method stack is pinned in [`requirements.txt`](requirements.txt). `pandas==2.0.3` is
intentional because `imputegap==1.1.1` declares that exact dependency; `pandera==0.20.4` is
pinned because newer Pandera releases require newer pandas and break the schema API used by
Track A validation. The package metadata in [`pyproject.toml`](pyproject.toml) allows these
versions so `pip install -e ".[dev]"` does not upgrade pandas behind ImputeGAP's back.

Everything selects the device with `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.
On a CUDA host, WaveStitch+ training/synthesis and PyPOTS use the GPU automatically. On CPU,
WaveStitch+ `--fast` is practical for the smaller AMF prepared bundle, but the large
`rabbitmq-performance_regularized` bundle can still take several minutes; use GPU for full
all-dataset reproduction.

**One command smoke reproduction** — a CPU-friendly pass over the three smaller performance
datasets (`golang-web-server-performance`, `python-web-server-performance`,
`rabbitmq-performance`; AMF is reserved for the full run). Track A cleans all three, Track B
scores the shared-holdout baseline on `rabbitmq-performance`, then the runner methods execute on
each bundle — **WaveStitch+ first**, then Darts (`linear`), ImputeGAP (`knn`), and PyPOTS
(`saits`/`brits`, 1 epoch):

```bash
conda activate wavestitchplus-repro
bash scripts/reproduce_all.sh smoke
# writes data/processed/{golang,python,rabbitmq}-*_generated/,
# each with wavestitchplus_{v1,v2,harpoon}_final.csv at 0 residual NaNs.
```

**Full GPU reproduction** — all four performance datasets, all method families, higher PyPOTS
epochs, and full WaveStitch+ hyperparameters:

```bash
conda activate wavestitchplus-repro
bash scripts/reproduce_all.sh full

# Legacy one-command GPU benchmark, including long-gap depth evaluation:
bash scripts/gpu_reproduce.sh
```

### A. Raw CSVs → `data/processed/` results

Track A is the local DataOps path that turns files in [`data/raw/`](data/raw/) into
stage-named artifacts under [`data/processed/`](data/processed/). For every performance
time-series CSV, the expected lineage is:

```text
data/raw/<name>.csv
  → data/processed/<name>_soft_cleaned.csv
  → data/processed/<name>_remediated.csv
  → data/processed/<name>_regularized/
  → data/processed/<name>_generated/
  → data/processed/<name>_final.csv
```

Run all four bundled performance raw files into `data/processed/`:

```bash
conda activate wavestitchplus-repro

for NAME in \
  amf-performance \
  golang-web-server-performance \
  python-web-server-performance \
  rabbitmq-performance
do
  python -m pipelines.minimal_dataops \
    --input "data/raw/${NAME}.csv" \
    --output "data/processed/${NAME}_remediated.csv" \
    --report "reports/${NAME}_report.json" \
    --log-file "logs/${NAME}-dataops.log"

  python scripts/auto_impute.py \
    --report "reports/${NAME}_report.json" \
    --method all
done
```

After this loop, `data/processed/` should contain, for each `<name>` above:

```text
<name>_soft_cleaned.csv
<name>_remediated.csv
<name>_regularized/
<name>_generated/                 # all built-in imputed splits + method-specific finals
<name>_final.csv
```

`--method all` runs every dependency-free built-in method:

```text
darts/{auto,cubic,linear,nearest,quadratic,slinear,zero}
imputegap/{interpolation,mean,mean_by_series,min,zero}
```

The canonical `<name>_final.csv` is still built from `darts/nearest` for backward-compatible
lineage and dashboard use. Method-specific finals are written under `<name>_generated/`, for
example `darts_linear_final.csv`, `darts_nearest_final.csv`, and `imputegap_mean_final.csv`.
PyPOTS and WaveStitch+ are heavier model runners; generate those with the model tracks
(WaveStitch+ = **Track C**, PyPOTS = **Track E**).

Check the final files:

```bash
python -c "from pathlib import Path; import pandas as pd
for p in sorted(Path('data/processed').glob('*_final.csv')):
    df = pd.read_csv(p)
    print(f'{p}: shape={df.shape}, nan={int(df.isna().sum().sum())}')"
```

The small KUL antenna sample, `data/raw/user_0_sample_0_antenna_0.csv`, is a tabular sample
rather than the EUR performance time-series format. It still goes through cleaning and
validation, but it does not normally produce a `_regularized/`, `_generated/`, or `_final.csv`
time-series imputation lineage:

```bash
NAME=user_0_sample_0_antenna_0
python -m pipelines.minimal_dataops \
  --input "data/raw/${NAME}.csv" \
  --output "data/processed/${NAME}_remediated.csv" \
  --report "reports/${NAME}_report.json" \
  --log-file "logs/${NAME}-dataops.log"
```

Single-dataset example, using the default rabbitmq config:

```bash
# 1) Validate + remediate + regularize + emit the imputation handoff
python -m pipelines.minimal_dataops --config config/dataops.yaml
#   → data/processed/rabbitmq-performance_soft_cleaned.csv     (conservative cleaning)
#   → data/processed/rabbitmq-performance_remediated.csv       (per-issue fixes)
#   → data/processed/rabbitmq-performance_regularized/         (prepared bundle; gaps explicit)
#   → reports/rabbitmq-performance_report.json                 (GX before/after + handoff)

# 2) Run the handoff method and build the gap-free final dataset
python scripts/auto_impute.py \
    --report reports/rabbitmq-performance_report.json --method all
#   → data/processed/rabbitmq-performance_generated/           (imputed train + test splits)
#   → data/processed/rabbitmq-performance_final.csv            (0 NaN, imputed train + test)
#   → reports/rabbitmq-performance_imputation_compare.json     (clean-vs-imputed MAE)
```

### B. Baseline benchmark on a shared holdout (CPU)

Scores the **Darts** and **PyPOTS** baselines (`compare_baselines.py` supports `darts_*` and
`pypots_*`; run ImputeGAP and WaveStitch+ via the app runners in **Tracks C and E**) on the *same* masked
holdout WaveStitch+ is evaluated on, and writes a sorted MAE/RMSE table. It calls the real libraries,
so run it in the `wavestitchplus-repro` env (`darts 0.33`, `pypots 1.5`, `torch 2.5.1` — see
[Reproduce → 0](#0-clone-and-install)).
PyPOTS uses windowed inference internally, so it scales to the long series here.

```bash
conda activate wavestitchplus-repro
python scripts/compare_baselines.py \
    --input-csv data/raw/amf-performance.csv \
    --methods   darts_linear,darts_nearest,pypots_saits
#   → experiments/amf-performance/generated_<run_id>/results.csv  (sorted MAE/RMSE)
```

> Trim `--methods` to `darts_linear,darts_nearest` for a fast, interpolation-only run. The
> dependency-free imputation path (no darts/pypots install) is **Track A** (`auto_impute.py`) and
> the dashboard's built-in methods — not this benchmarking script.

### C. WaveStitch+ v1 → v2, and the top-2 result (CPU or GPU)

Requires the all-method conda env with `darts`, `imputegap`, `pypots`, and `torch`
(see [Reproduce → 0](#0-clone-and-install)). WaveStitch+ is the **first model track** and uses
the same processed layout as every other runner: read `data/processed/<name>_regularized/` and
write every WaveStitch+ artifact into `data/processed/<name>_generated/`.

```bash
conda activate wavestitchplus-repro

# Pick any performance dataset that Track A has already regularized.
NAME=amf-performance
B=data/processed/${NAME}_regularized
G=data/processed/${NAME}_generated
mkdir -p "$G"

# Keep reproducibility logs readable. This hides known library warnings
# (PyKeOps deprecated dtype, torch.load FutureWarning, etc.) but keeps errors.
export PYTHONWARNINGS="ignore"
quiet_wsp() {
  "$@" 2> >(grep -v -E '^\[pyKeOps\] Warning|FutureWarning|UserWarning' >&2)
}

# 1) Optional: add baselines to the SAME processed generated dir for comparison.
#    Track A --method all already writes built-in Darts/ImputeGAP outputs here;
#    run these only if you also want real Darts kalman or PyPOTS in the table.
for m in linear nearest cubic auto kalman; do
  python dockers/tools/Darts_app/run_imputation.py --prepared-dir "$B" --output-dir "$G" --method "$m"
done
python dockers/tools/PyPOTS_app/run_imputation.py --prepared-dir "$B" --output-dir "$G" \
  --method saits --window 100 --epochs 15

# 2) WaveStitch+ v1 — train + synthesize.
#    CPU smoke: keep --fast --device cpu. GPU/full: remove --fast and use the full HP line below.
quiet_wsp python dockers/tools/WaveStitchPlus_app/run_imputation.py \
  --prepared-dir "$B" --output-dir "$G" --fast --device cpu
# GPU/full alternative:
# quiet_wsp python dockers/tools/WaveStitchPlus_app/run_imputation.py \
#   --prepared-dir "$B" --output-dir "$G" \
#   --em-iterations 5 --epochs-per-em 200 --ddim-steps 50 --repaint-rounds 5 --device auto
test -f "$G/wavestitchplus_v1_test_imputed.csv" || {
  echo "WaveStitch+ v1 did not produce $G/wavestitchplus_v1_test_imputed.csv" >&2
  exit 1
}

# 3) WaveStitch+ v2 — per-column auto anchoring; reuses the v1 diffusion test CSV.
quiet_wsp python dockers/tools/WaveStitchPlus_app/run_imputation_v2.py \
  --prepared-dir "$B" --output-dir "$G" \
  --reuse-diffusion "$G/wavestitchplus_v1_test_imputed.csv"

# 4) HARPOON — manifold-bound guidance using the v1 checkpoint copied under $G/saved_models/.
quiet_wsp python dockers/tools/WaveStitchPlus_app/run_imputation_harpoon.py \
  --prepared-dir "$B" --output-dir "$G" \
  --ddim-steps 5 --repaint-rounds 1 --device cpu
# GPU/full alternative:
# quiet_wsp python dockers/tools/WaveStitchPlus_app/run_imputation_harpoon.py \
#   --prepared-dir "$B" --output-dir "$G" --ddim-steps 50 --repaint-rounds 5 --device auto

# 5) Score v1/v2/HARPOON against every *_test_imputed.csv in the same generated dir.
python scripts/compare_wsp_v2.py \
  --prepared-dir "$B" \
  --baseline-dir "$G" \
  --v1-csv "$G/wavestitchplus_v1_test_imputed.csv" \
  --out-csv "$G/wsp_v2_comparison.csv"
#   → data/processed/<name>_generated/wavestitchplus_{v1,v2,harpoon}_{train,test}_imputed.csv
#   → data/processed/<name>_generated/wavestitchplus_{v1,v2,harpoon}_final.csv
#   → data/processed/<name>_generated/wsp_v2_comparison.csv
```

Optional long-gap regime (where diffusion overtakes interpolation):

```bash
python scripts/eval_long_gap.py \
    --prepared-dir data/processed/python-web-server-performance_regularized --reuse
```

Tracks D and E fill in the heavier real-library methods beside WaveStitch+ so the dashboard can
compare them all in one processed run. Both keep the shared convention: read the Track A bundle
from `data/processed/<name>_regularized/` (`$B`) and write results to
`data/processed/<name>_generated/` (`$G`). Track A with `--method all` already wrote the
dependency-free Darts/ImputeGAP built-ins there; use these two tracks only for the real-library
methods those built-ins do not cover.

```bash
conda activate wavestitchplus-repro
NAME=amf-performance
B=data/processed/${NAME}_regularized
G=data/processed/${NAME}_generated
mkdir -p "$G"
export PYTHONWARNINGS="ignore"
```

### D. Darts runners

Real-library Darts methods beyond Track A's built-in interpolation path — chiefly `kalman`,
which fits `darts.models.KalmanFilter` per column:

```bash
python dockers/tools/Darts_app/run_imputation.py \
  --prepared-dir "$B" --output-dir "$G" --method kalman
```

`kalman` is **slow on long series** (its cost grows super-linearly with row count, so a 40k+ row
bundle takes minutes); the `reproduce_all.sh smoke` path therefore defaults Darts to `linear`.
See the full method list in [Baseline Imputation Methods](#baseline-imputation-methods).

### E. Other library runners — ImputeGAP + PyPOTS

```bash
# ImputeGAP method not covered by Track A's built-in statistics.
python dockers/tools/ImputeGAP_app/run_imputation.py \
  --prepared-dir "$B" --output-dir "$G" --method knn

# PyPOTS — WINDOWED (--window keeps attention O(window²); whole-series inference OOMs)
for m in saits brits; do
  python dockers/tools/PyPOTS_app/run_imputation.py \
    --prepared-dir "$B" --output-dir "$G" \
    --method "$m" --window 100 --epochs 15
done
#   → $G/<lib>_<method>_{train,test}_imputed.csv
#   → $G/<lib>_<method>_final.csv
```

Every runner shares the same `--prepared-dir/--output-dir/--method` interface and the dashboard
auto-discovers files in `$G` for the run whose report points at `$B`. PyPOTS **requires**
`--window` because single-window whole-series attention is O(steps²). Repeat any track by changing
only `NAME` to `golang-web-server-performance`, `python-web-server-performance`, or
`rabbitmq-performance`.

### Explore every run in the dashboard

```bash
conda activate wavestitchplus-repro           # streamlit + plotly live here
bash dashboard/run.sh                         # → http://localhost:8502
```

Pick a raw CSV, then a DataOps run, to walk the `raw → … → final` lineage with GX
before/after metrics and the per-method imputation comparison. See
[Comparison Dashboard](#comparison-dashboard) for the section-by-section tour.

### Full Airflow + SeaweedFS stack (Docker, optional)

To reproduce the orchestrated pipeline and S3 data lake instead of the local runners, follow
[Installation & Setup](#installation--setup) → `bash dockers/start.sh`, then open the Airflow UI
at **http://localhost:8088** (`admin / admin`) and trigger the
`data_quality_and_cleaning_pipeline` DAG. The stack exposes:

| Service | Port (host) | Purpose |
|---|---|---|
| Airflow webserver | `8088` | UI / DAG runs (admin / admin) |
| DataOps dashboard | `8502` | Cleaning-first run overview, remediation, and imputation comparison |
| SeaweedFS S3 API | `8333` | S3-compatible endpoint |
| SeaweedFS filer | `8888`, `18888` | Filer + admin |
| SeaweedFS master | `9333` | Cluster master |
| SeaweedFS volume | `8080` | Volume server |
| Postgres | internal | Airflow metadata DB |

---

## Configuration & data lineage

The local pipeline lives in `src/data_process_modules/` (with `src/dataops/` kept as a
compatibility API), backed by Pandera validation, Great Expectations checks, `pytest`, GitHub
Actions CI (`.github/workflows/ci.yml`), and DVC (`dvc.yaml`). To run it, see
[Reproduce the results → Track A](#a-raw-csvs--dataprocessed-results).

`config/dataops.yaml` controls input and output paths, validation mode, optional
timestamp column, expected columns, numeric bounds, missingness threshold,
timestamp uniqueness, and timestamp ordering. With `validation.mode: auto`, the
pipeline classifies each dataset as `time_series` or `tabular`, records the
classification reason in the report, validates as time-series when a real
timestamp column is configured or detected, and otherwise validates as arbitrary
tabular data. Monotonic numeric IDs are treated as tabular by default; set
`allow_step_index_timestamp: true` only when a step index really is your time
axis. CLI flags such as `--input`, `--output`, `--report`, and
`--timestamp-col` can still override the config for one-off runs.

The pipeline records a five-stage lineage and writes an artifact for each stage:

```
raw → soft-cleaned → remediated → regularized (gaps explicit) → final (imputed, gap-free)
```

Artifacts are named per stage from the dataset `<name>` (e.g. `rabbitmq-performance`):

- **soft-cleaned** (`<name>_soft_cleaned.csv`, key `report.soft_cleaned_output`;
  legacy alias `report.cleaned_output`) — conservative cleaning: snake_case columns, empty or
  duplicate row removal, timestamp ordering fixes, and epoch-aware datetime coercion.
- **remediated** (`<name>_remediated.csv`, the configured `output`) — issue-specific fixes from
  `data_process_modules.remediation`, including numeric outlier winsorization and type-aware
  tabular filling. Time-series gaps are *deferred* to imputation.
- **regularized** (`<name>_regularized/`) — when a time-series gap is detected, the timeline is
  regularized onto a uniform grid. Gaps become explicit NaN rows, written as a prepared bundle.
- **final** (`<name>_final.csv`) — the gap-free, analysis-ready dataset (built by the imputation
  step, see below), stitched from the imputed train + imputed test splits under `<name>_generated/`.

The report's `quality` section is produced by Great Expectations-backed checks and includes an
`action_plan` tagging each issue with its `status` (`applied_by_remediation`,
`deferred_to_imputation`, or `manual`) plus the concrete failed GX expectations.
`quality_after` re-runs the checks on the remediated frame, so the report contains a genuine GX
**before/after** comparison. `handoff` advertises the imputation method catalog and the
configured `(app, method)`. `validation_comparison` provides chart-ready dashboard data for
raw → soft-cleaned → remediated status across GX and Pandera.

**Imputation handoff → final dataset.** The pipeline does not run imputation directly; it
regularizes the dataset and emits the handoff. `scripts/auto_impute.py`
([Track A](#a-raw-csvs--dataprocessed-results)) runs the selected method, or `--method all` for
all dependency-free built-ins, and writes `<name>_final.csv` (gap-free) plus
`reports/<name>_imputation_compare.json`.
`darts/<interp>` runs without external Darts dependencies through `dataops.imputation_runner`
(bit-faithful to Darts' `MissingValuesFiller`); use `--engine darts` for the real Darts runner.

Run the DVC stage after placing data at `data/raw/input.csv`:

```bash
python -m pip install -e ".[dvc]"
dvc repro
```

Failure notifications are opt-in:

```bash
export DATAOPS_WEBHOOK_URL="https://hooks.slack.com/services/..."
python -m pipelines.minimal_dataops --config config/dataops.yaml
```

## Installation & Setup

### Requirements

- Docker Engine and `docker compose` on a host with `sudo` access. This is only
  needed for the Airflow + SeaweedFS orchestration path; laptop-only workflows
  can skip Docker entirely (see [No-GPU / no-Docker workflow](#no-gpu--no-docker-workflow)).
- NVIDIA GPU with the **NVIDIA Container Toolkit**. This is recommended for
  WaveStitch+ training, but not required. PyTorch falls back to CPU
  automatically, and `Dockerfile.wavestitchplus-cpu` provides a CPU image.
- About 10 GB of free disk space for images and SeaweedFS volumes.

After cloning (see [Reproduce → 0](#0-clone-and-install)):

### 1. Build the WaveStitch+ GPU image

```bash
cd dockers/tools
bash build_image.sh            # runs: docker build -f Dockerfile.wavestitchplus-gpu -t wavestitchplus-gpu:latest .
```

### 2. Start Airflow + SeaweedFS + dashboard

```bash
cd ../                         # back to dockers/
bash start.sh                  # docker compose build && docker compose up -d
```

On first boot, the Airflow container automatically:

- runs `airflow db migrate`;
- creates an `admin / admin` user;
- waits for SeaweedFS S3 and creates the `6gdali-lake2026` bucket;
- registers the `seaweed_s3` Airflow connection;
- starts the scheduler and webserver.

Open **http://localhost:8088** and sign in as `admin / admin`.
Open **http://localhost:8502** for the DataOps dashboard.

### 3. (Optional) Local Python env for the notebooks

```bash
python -m venv venv
source venv/bin/activate                    # Linux/macOS
# venv\Scripts\activate                     # Windows
pip install -r dockers/tools/requirements.txt
```

---

## Usage

### A. Run the End-to-End Airflow Pipeline

1. In the Airflow UI, open the DAG **`data_quality_and_cleaning_pipeline`**.
2. (Optional) Override defaults via Airflow **Variables**:
   - `N2N_INPUT_KEY` (default `test/amf-performance.csv`)
   - `S3_BUCKET` (default `6gdali-lake2026`)
   - `N2N_TIMESTAMP_COL` (default `time`)
   - `N2N_DATASET_NAME` (default: derived from the input filename)
3. Click **Trigger DAG**. The run executes:
   `load_raw_data → is_time_series → qc / ts_qc → report_dqc → clean_dirty_data`.

A second debug DAG, `seaweedfs_datalake_test_v2`, verifies the SeaweedFS
connection by writing a `hello_world.txt` test object.

### B. Run the WaveStitch+ Container Directly

The container reads from and writes to the lake through S3 environment variables
and exposes a single CLI entry point.

```bash
docker network inspect dockers_airflow_net >/dev/null   # confirms the compose network exists

sudo docker run --rm --gpus all \
    --network dockers_airflow_net \
    -e S3_ENDPOINT=http://seaweed-s3:8333 \
    -e S3_ACCESS_KEY=anykey \
    -e S3_SECRET_KEY=anysecret \
    -e S3_BUCKET=6gdali-lake2026 \
    wavestitchplus-gpu:latest \
    python /app/run_pipeline.py \
      --mode full \
      --dataset-name amf-performance \
      --input-s3-key test/amf-performance.csv \
      --use-em --em-iterations 5 --epochs-per-em 200 \
      --model-type em --clamp-mode bounds
```

A ready-to-edit script with `train`-only and `inference`-only variants lives at [dockers/test_docker.sh](dockers/test_docker.sh).

#### No-GPU / No-Docker Workflow

PyTorch in [WaveStitchPlus_app](dockers/tools/WaveStitchPlus_app) already
selects the device with `torch.device("cuda" if torch.cuda.is_available() else
"cpu")`, so the same training and synthesis scripts can run natively on a CPU
laptop. There are two options:

**A. Native, in the `wavestitchplus-repro` conda env** — the recommended dev path, driven by
`run_local.sh` (see [Reproduce → Track C](#c-wavestitch-v1--v2-and-the-top-2-result-cpu-or-gpu)).
Useful knobs: `SUBSETS`, `SKIP_PREPROCESS=1` (reuse existing `prepared_*/` folders and let the
in-training fallback compute `iqr/1.349` on the fly), `FAST=1` (`em=2`, `epochs=30`, `ddim=20`),
`EM_ITERS`, `EPOCHS_PER_EM`, `DDIM_STEPS`, and `REPAINT_ROUNDS`. Outputs land in
`experiments/EUR/generated_<subset>/wavestitchplus_v1_{train,test}_imputed.csv`, which the
dashboard discovers automatically.

**B. CPU Docker image** (for reproducibility without a GPU):

```bash
cd dockers/tools
docker build -f Dockerfile.wavestitchplus-cpu -t wavestitchplus-cpu:latest .
docker run --rm \
    -v "$(pwd)/../../notebooks/work:/work" \
    -v "$(pwd)/../../6GDALI_Datasets:/data:ro" \
    wavestitchplus-cpu:latest \
    python /app/WaveStitchPlus_app/train_improved.py -d custom_csv \
        -input_csv /data/EUR/6907619/python-web-server-performance.csv \
        -prepared_dir /work/EUR/prepared_python \
        -use_em -em_iterations 5 -epochs_per_em 200 -ddim_steps 50 \
        -repaint_rounds 5 -save_train_imputed_denorm -train_imputed_clamp bounds
```

Expect CPU training to be **5-20x slower** than GPU with the same
hyperparameters. For example, a python-subset run with `em=5` and
`epochs/em=200` takes minutes on GPU and roughly 1-2 hours on a modern laptop
CPU. For interactive experimentation, use `FAST=1` first to verify the pipeline
end to end, then scale up only the configurations you want to benchmark.

#### `run_pipeline.py` modes

| Mode | What it does |
|---|---|
| `train` | Downloads `--input-s3-key`, runs `train_improved.py`, uploads `prepared/{saved_model, scaler, meta.json, train_imputed*.npy}` to `s3://<bucket>/wavestitchplus/<dataset>/<version>/`. |
| `inference` | Downloads a previously trained `prepared/` directory (auto-selects the latest valid version if `--model-version` is omitted), runs `synthesis_improved.py`, uploads `inference_results_<version>/imputed.csv` and refreshes `latest_inference/`. |
| `full` | Runs `train` and then `inference` in one command. `model_type` falls back to `standard` when `--use-em` is not set. |

#### Key flags

| Flag | Default | Notes |
|---|---|---|
| `--mode` | `full` | `train` \| `inference` \| `full` |
| `--dataset-name` | required | Logical dataset id used as the S3 prefix |
| `--input-s3-key` | required for `train` / `full` | CSV key inside the bucket |
| `--version` | timestamp | Run version; auto-generated if omitted |
| `--use-em` / `--em-iterations` / `--epochs-per-em` | off / 5 / 200 | Enables the EM training loop |
| `--repaint-rounds` | 5 | RePaint resampling rounds |
| `--ddim-steps` | 50 | Ignored when `--use-ddpm` is set |
| `--clamp-mode` | `bounds` | `none` \| `nonneg` \| `bounds` |
| `--guidance-scale`, `--n-trials`, `--bound-headroom` | 0.1 / 1 / 1.2 | Inference controls |
| `--keep-workdir` | off | Preserve `/tmp/wavestitchplus/...` for debugging |

### C. Evaluate Repair Quality in Notebooks

```bash
cd notebooks
jupyter lab
```

| Notebook | Purpose |
|---|---|
| `analysis of the ts imputation.ipynb` | Diagnostic plots over imputed series |
| `comparisons.ipynb` | Side-by-side method / version comparisons |
| `visual.ipynb` | Grid plots across all features |

`download_folders.py` helps pull artifact directories out of the lake for
offline analysis.

---

## Pipeline Modules

The Airflow DAG remains the reference orchestration path, but the non-imputation
pipeline logic is also exposed as reusable modules under
[src/data_process_modules](src/data_process_modules). Use these
modules when another system needs profiling, tabular or time-series quality
checks, preprocessing, or train/test splitting without importing Airflow tasks.

```python
from data_process_modules import profiling, split, ts_checks

profile = profiling.profile(df, timestamp_col="time")
ts_report = ts_checks.run(df, ts_col=profile["timestamp_column"])
parts = split.train_test(df, profile, seed=0)
```

`profiling`, `ts_checks`, `tabular_checks`, `transform.preprocess(...)`, and
`split` operate on DataFrames. `transform.preprocess_csv(...)` is still
available when you need the existing `prepared_<subset>/` artifact bundle
consumed by the imputation apps and dashboard. The same CLI is available with
`python -m data_process_modules ...` or the installed `data-process-modules`
script.

The cleaning and imputation loop adds three more modules:

- `remediation.remediate(df, quality_report)` — applies a concrete fix for each detected
  issue, such as outlier winsorization or type-aware tabular filling, and defers time-series
  gaps to imputation.
- `imputation_catalog` — lists the imputation apps and methods advertised in the handoff.
  `validate_selection(app, method)` never raises; it reports `ok`, `invalid`, or `known_failing`.
- `imputation_runner` — provides `impute_bundle(...)` through a Darts-faithful pandas engine
  or a real-Darts subprocess, plus `compare_clean_vs_imputed(...)` and
  `build_final_dataset(...)` for the gap-free final cleaned CSV.

Every module exposes a `METADATA` descriptor aggregated into
`data_process_modules.registry.MANIFEST` for host-system discovery.

---

## Baseline Imputation Methods

Three baseline runners live next to `WaveStitchPlus_app/` so their outputs can
be compared directly with WaveStitch+. Each runner is packaged in the same
style, with its own `Dockerfile.<lib>`, `requirements.txt`, and
`run_imputation.py`, but the runners are intentionally independent: no shared
module and no leakage of WaveStitch+ internals.

| Folder | Library | Built-in methods (`--method`) |
|---|---|---|
| [dockers/tools/Darts_app](dockers/tools/Darts_app) | [Darts](https://unit8co.github.io/darts/) | `auto`, `linear`, `quadratic`, `cubic`, `nearest`, `slinear`, `zero`, `kalman` |
| [dockers/tools/ImputeGAP_app](dockers/tools/ImputeGAP_app) | [ImputeGAP](https://github.com/eXascaleInfolab/ImputeGAP) | `mean`, `mean_by_series`, `min`, `zero`, `interpolation`, `knn`, `cdrec`, `iterative_svd`, `soft_impute`, `svt`, `iim`, `mice`, `miss_forest`, `brits`, `mrnn`, `gain`, ... (run with `--list` to see what your installation exposes) |
| [dockers/tools/PyPOTS_app](dockers/tools/PyPOTS_app) | [PyPOTS](https://github.com/WenjieDu/PyPOTS) | `saits`, `brits`, `transformer`, `gpvae`, `mrnn`, `csdi`, `usgan`, `timesnet` |

### Common interface

All three runners read a `prepared_<subset>/` folder produced by the
WaveStitch+ pipeline (`meta.json`, `train.csv`, `test_input.csv`, ...) and
write **one CSV per input**:

```
<output-dir>/<lib>_<method>_train_imputed.csv
<output-dir>/<lib>_<method>_test_imputed.csv
```

The split filenames match the WaveStitch+ wrapper convention, so
[comparisons.ipynb](notebooks/comparisons.ipynb) and the dashboard can discover
train and test outputs consistently. Pass `--inputs train test` to choose which
splits to impute. The default is both.

```bash
# Generic invocation; works for any of the three runners
python run_imputation.py \
    --prepared-dir notebooks/work/EUR/prepared_amf \
    --output-dir   notebooks/work/EUR/generated_amf \
    --method       <method-name>
```

### Run Locally With The Repro Env

The `wavestitchplus-repro` conda environment contains compatible versions of all three
baseline libraries plus WaveStitch+: `darts 0.33`, `imputegap 1.1.1`, `pypots 1.5`,
and `torch 2.5.1`. Create it with `conda env create -f environment.yml` from
[Reproduce → 0](#0-clone-and-install).

Each app includes a `run_local.sh` script that loops through the curated method
list across the four EUR subsets (`amf`, `golang`, `python`, `rabbitmq`) and
writes both train and test outputs:

```bash
conda activate wavestitchplus-repro

bash dockers/tools/Darts_app/run_local.sh        # linear, cubic, nearest, kalman, auto
bash dockers/tools/ImputeGAP_app/run_local.sh    # mean, interpolation, knn, iim
EPOCHS=50 bash dockers/tools/PyPOTS_app/run_local.sh   # saits, brits
```

Each runner writes `<lib>_<method>_{train,test}_imputed.csv` next to the WaveStitch+ result in
`generated_<subset>/`, so [comparisons.ipynb](notebooks/comparisons.ipynb) and the dashboard
discover every method side by side.

### Run in Docker

```bash
cd dockers/tools

bash Darts_app/build_image.sh        # → darts-baseline:latest
bash ImputeGAP_app/build_image.sh    # → imputegap-baseline:latest
bash PyPOTS_app/build_image.sh       # → pypots-baseline:latest    (needs --gpus all)
bash build_image_cpu.sh              # → wavestitchplus-cpu:latest (no GPU; dashboard default)
# bash build_image.sh                # → wavestitchplus-gpu:latest (needs nvidia-docker)

# Example: SAITS on the amf subset, mounting the local work tree.
sudo docker run --rm --gpus all \
    -v "$(pwd)/../../notebooks/work:/work" \
    pypots-baseline:latest \
    --prepared-dir /work/EUR/prepared_amf \
    --output-dir   /work/EUR/generated_amf \
    --method       saits --epochs 200
```

### Notes on each library

- **Darts** — interpolation methods (`linear` ... `auto`) run through
  `MissingValuesFiller`. `kalman` uses `darts.models.KalmanFilter` fitted on
  observed points, re-indexed to a contiguous range because Darts 0.33 requires a
  regular `RangeIndex`. It falls back to interpolation when a column has fewer
  than two observations. Any edge NaNs left by `quadratic` or `cubic` are
  forward- and back-filled.
- **ImputeGAP** — the runner probes the installed
  `imputegap.recovery.imputation.Imputation` registry and only registers methods
  that exist in your version. Pass `--list` to see what is available. The runner
  refuses to write a CSV if a method returns a shape that does not match the
  input matrix. Some 1.1.1 methods, such as MICE, return a holdout-only matrix
  that cannot be reassembled into the original CSV. On macOS, `cdrec` and
  `stmvl` need the native `libarmadillo` shared library; install it with
  `brew install armadillo`, or skip those methods.
- **PyPOTS** — fits each model on `train.csv` using target columns only,
  standardized by train-set mean and standard deviation, then imputes both
  `train.csv` and `test_input.csv`. **Windowed inference** (`--window`, default 100)
  tiles each series into fixed-length windows batched along the sample axis, so
  attention stays O(window²) — whole-series inference (`--window 0`) is O(steps²)
  and **OOMs on long series** (a 100k-step subset gets killed). Overlapping windows
  (`--stride` < window) are averaged on stitch; series shorter than one window are
  NaN-padded then cropped. The model is built with `n_steps = window`, so train and
  test share one shape. Original observed cells are preserved exactly, so comparisons
  measure imputation quality rather than reconstruction noise. **Optional checkpointing** follows a
  train-once/reuse pattern: `--model-path <dir> --save-model` persists trained
  weights as `pypots_<method>_n<steps>_f<feats>.pypots`; a later run with
  `--model-path <dir> --load-model` reloads them and skips training. Each window
  length and feature count keeps its own checkpoint; a missing checkpoint falls
  back to training.

---

## WaveStitch+ v2 & Method Comparison

On the default holdout, where scattered single points are usually one step from
a real neighbor, raw diffusion is easily beaten by interpolation. **v2** keeps
the WaveStitch+ diffusion but **anchors it to a context-aware interpolation
prior**, blending the two per cell by distance to the nearest observation. Short
gaps follow the prior, while deep-gap cells fall back to the diffusion model's
structural regime. No retraining is required; v2 reuses synthesis from an
existing checkpoint.

**Regenerate WaveStitch+ (v1 → v2 → harpoon) for one processed dataset** (e.g.
`amf-performance`; each writes `wavestitchplus_<variant>_{train,test}_imputed.csv`
+ `_final.csv` into `data/processed/<name>_generated/`):

```bash
conda activate wavestitchplus-repro

NAME=amf-performance
B=data/processed/${NAME}_regularized
G=data/processed/${NAME}_generated
mkdir -p "$G"

export PYTHONWARNINGS="ignore"
quiet_wsp() {
  "$@" 2> >(grep -v -E '^\[pyKeOps\] Warning|FutureWarning|UserWarning' >&2)
}

# 1) v1 — train + synthesize (GPU: full HP; laptop CPU: replace the flags with --fast)
quiet_wsp python dockers/tools/WaveStitchPlus_app/run_imputation.py \
  --prepared-dir "$B" --output-dir "$G" \
  --em-iterations 5 --epochs-per-em 200 --ddim-steps 50 --repaint-rounds 5 --device auto
# CPU smoke alternative:
# quiet_wsp python dockers/tools/WaveStitchPlus_app/run_imputation.py \
#   --prepared-dir "$B" --output-dir "$G" --fast --device cpu
test -f "$G/wavestitchplus_v1_test_imputed.csv" || {
  echo "WaveStitch+ v1 did not produce $G/wavestitchplus_v1_test_imputed.csv" >&2
  exit 1
}

# 2) v2 — per-column `auto` anchoring, reuses the v1 test diffusion (seconds, torch-free)
quiet_wsp python dockers/tools/WaveStitchPlus_app/run_imputation_v2.py \
  --prepared-dir "$B" --output-dir "$G" \
  --reuse-diffusion "$G/wavestitchplus_v1_test_imputed.csv"

# 3) harpoon — manifold-bound guidance on the pretrained checkpoint
quiet_wsp python dockers/tools/WaveStitchPlus_app/run_imputation_harpoon.py \
  --prepared-dir "$B" --output-dir "$G" \
  --ddim-steps 50 --repaint-rounds 5 --device auto
```

Order matters: **v2/harpoon depend on v1** — step 1 trains the model and writes
`wavestitchplus_v1_test_imputed.csv` (v2 reuses it via `--reuse-diffusion`; harpoon uses the v1
checkpoint). `run_imputation.py` clears the checkpoint from the bundle afterward but syncs it to
`$G/saved_models/` and publishes the v1 train output, so v2/harpoon still emit their train split
(and a full train+test `_final.csv`). amf is 111k rows — **run v1 on a GPU** (`--fast` for a CPU
smoke test). The `quiet_wsp` wrapper only filters known noisy warnings; a non-zero process exit
still fails normally.

Three helper scripts under [scripts/](scripts) score methods on the **same
holdout** (`test_input` is NaN and `test_gt` is known):

| Script | Purpose |
|---|---|
| [compare_baselines.py](scripts/compare_baselines.py) | Preprocess + run the Darts / ImputeGAP / PyPOTS baselines and write a sorted MAE/RMSE table. |
| [compare_wsp_v2.py](scripts/compare_wsp_v2.py) | Score WaveStitch+ **v1 vs v2** against the baselines (reuses a v1 diffusion CSV; runs in seconds). |
| [eval_long_gap.py](scripts/eval_long_gap.py) | **Long-gap regime**: carves contiguous gaps out of fully observed runs and scores by *depth*, where the diffusion model overtakes interpolation. Feasible only where the test split has long observed runs, such as EUR/python. |

**Honest reading of the numbers.** Across the four EUR subsets v2 cuts WaveStitch+'s holdout MAE
**~1.5–3× relative to raw v1** — a large, robust win over the diffusion baseline. On the scattered
point-holdout the **diffusion adds no positive signal** (it has ~zero correlation with the truth even
at depth: a sweep drives the MAE-optimum to a pure interpolation prior); so v2's job is really to
regularize v1 back onto the best *interpolation*. The lever that beats `nearest` is the **prior**,
not the diffusion: `nearest` flat-holds across a gap, but these series trend, so **`linear`
interpolation is usually better** (amf `linear` 0.92× `nearest`). The default anchor is therefore
**`prior=auto, hard_prior=32, tau=8`** — `auto` picks nearest-vs-linear **per column** by an
*unsupervised* observed-data cross-check (trending cols → linear, near-constant → nearest; it never
looks at `test_gt`), and the wide hard-prior keeps the (useless) diffusion out of the natural holdout.
Result: v2 **beats `nearest`** on amf (0.93×), golang (0.99×) and rabbitmq (0.97×) and ties python
(1.00×), matching the dashboard's per-run Metrics tab exactly. The diffusion is reserved for the
long-gap regime — where flat-hold interpolation finally breaks and the diffusion's structure helps —
which [eval_long_gap.py](scripts/eval_long_gap.py) isolates; it is underrepresented in the natural
scattered holdout, which is why the aggregate point-holdout still favours `nearest`.

---

## Comparison Dashboard

A Streamlit dashboard under [dashboard/](dashboard) discovers each DataOps run
from `reports/*.json` and walks through its `raw → ... → final` lineage.
Imputation method comparison is shown as the final-step detail.

```bash
conda activate wavestitchplus-repro
bash dashboard/run.sh                    # opens http://localhost:8502
# or:  streamlit run dashboard/app.py
```

### What the dashboard exposes

The dashboard is **raw-data first**: pick a source CSV, then pick one of the
DataOps runs produced from that raw data. Every section follows the same
selection, and imputation method comparison appears as the detail for the final
step.

| Sidebar control | What it does |
|---|---|
| **Raw data** | Auto-discovered from `data/raw/*.csv` plus any raw inputs referenced by `reports/*.json`; filters quality, visualization, and run views to one source dataset. |
| **DataOps run** | Auto-discovered from `reports/*.json` pipeline reports for the selected raw data; resolves raw / soft-cleaned / remediated / regularized / final + the imputation comparison. |
| **Bundle root (fallback)** | Folder holding regularized bundles (`*_regularized/` under `data/processed`; legacy `*_prepared/` or `prepared_<subset>/` under an experiments tree). Used only when no report run is selected. Defaults to `data/processed`. |

### Sections

1. **Raw data** — source preview, quick quality counters, numeric visualization, and all DataOps runs linked to that CSV.
2. **Overview** — the `raw → soft-cleaned → remediated → regularized → final` lineage as stage cards, including row counts, artifact presence, key metrics, GX **before/after**, the imputation handoff, and the final cleaned-data callout.
3. **Quality & remediation** — GX detected issues versus after-remediation results, concrete failed expectations, issue → solution plan grouped by status (auto-handled / deferred-to-imputation / manual), the handoff, and the **clean-vs-imputed** comparison (fill rate + per-column MAE).
4. **Imputation** — the method workbench; **split, method, and feature pickers live here**. Sub-tabs:
   - **Time series** — observed, ground-truth-on-eval, per-method imputed, and truly-missing bands on a datetime x-axis;
   - **Metrics** — MAE, RMSE, MAPE, and fill rate per method, with per-feature breakdowns and constraint violations;
   - **Distribution** — observed vs GT vs imputed histograms;
   - **Long-gap** — the [eval_long_gap.py](scripts/eval_long_gap.py) depth regime, including the interpolation/diffusion crossover.
5. **Run** — sub-tabs **Run experiment** (invoke any listed runner) and **Pipeline run** (compare a DAG run against its S3 source). Two execution paths:

   - **Built-in, dependency-free** (no install needed): **darts** interpolation (`auto / linear / cubic / nearest / slinear / quadratic / zero`) and **imputegap** statistics (`interpolation / mean / min / zero`). Darts interpolation is bit-faithful to Darts' `MissingValuesFiller`; the ImputeGAP statistics are standard equivalents, not calls into the ImputeGAP library. **WaveStitch+ v2** is also dependency-free **when a v1 diffusion output already exists**. Its local anchoring (`wsp_v2`) is pure NumPy/pandas, so the dashboard auto-passes `--reuse-diffusion` and runs it without Torch, GPU, or retraining.
   - **Heavier methods** that need a trained diffusion model or GPU libraries — **WaveStitch+ v1** (retrains), **harpoon** (inference-time guidance on the pretrained model), **PyPOTS**, **darts-kalman**, and **ImputeGAP cdrec/brits/...** — run through their own app runners in two ways:

     - **Docker image** (recommended; no dependencies in the dashboard environment) — tick **"Run in Docker image"** in the Run tab. Each method runs inside its prebuilt image (`darts-baseline` / `imputegap-baseline` / `pypots-baseline` / `wavestitchplus-cpu` by default). The bundle directory is mounted at `/work`, and **run arguments are relative to it**. The dashboard checks that the image exists locally before running, so it will not unexpectedly pull from a registry. Edit the per-run **Docker image** field, or set `DATAOPS_IMPUTE_IMAGE_<LIB>=<repo:tag>`. Docker is enabled by default with `DATAOPS_IMPUTE_DOCKER=1`; for GPU use, set `DATAOPS_IMPUTE_GPU=1` and point the WaveStitch+ image to `wavestitchplus-gpu:latest`.
     - **Subprocess in another environment** — point imputation at a Python environment that already has the dependencies:

       ```bash
       export DATAOPS_IMPUTE_CONDA_ENV=wavestitchplus-repro    # conda environment with the imputation dependencies
       # or: export DATAOPS_IMPUTE_PYTHON=/path/to/python
       ```

### Screen at a glance

```
┌─ Sidebar ────────────┐  ┌─ Sections ─────────────────────────────────────┐
│ DataOps run:         │  │ [Overview] [Quality & remediation]             │
│  amf-performance_…   │  │ [Imputation] [Run]                             │
│ Bundle root          │  │ ┌─ Overview ───────────────────────────────┐   │
│  data/processed      │  │ │ raw → soft-cleaned → remediated →       │   │
│                      │  │ │     regularized → final                 │   │
│                      │  │ │     (stage cards + GX before/after +    │   │
│                      │  │ │      handoff + final callout)           │   │
│                      │  │ └──────────────────────────────────────────┘   │
└──────────────────────┘  └─────────────────────────────────────────────────┘
```

### Requirements

`streamlit`, `plotly`, `pandas`, and `numpy` are already installed in `wavestitchplus-repro`.
To install them elsewhere:

```bash
pip install -r dashboard/requirements.txt
```

---

## Testing

Unit tests cover the reusable pipeline modules and the imputation-runner helpers.
They use `pytest` and run on plain DataFrames and temporary directories; no
Airflow, GPU, or S3 service is required.

```bash
conda activate wavestitchplus-repro
pip install pytest                 # not bundled in the app requirements

pytest dockers/tools                                 # everything
pytest tests                                         # local data process modules
pytest dockers/tools/WaveStitchPlus_app/tests        # WaveStitch+ runner only
```

| Test suite | Covers |
|---|---|
| [tests](tests) | local DataOps/data process modules, minimal pipeline, profiling, validation, and module registry |
| [WaveStitchPlus_app/tests](dockers/tools/WaveStitchPlus_app/tests) | WaveStitch+ runner helpers (output naming, observed-cell-preserving merge, train-output publisher) and the v2 anchoring layer (prior, distance weighting, gap blend, scoring) |
| [airflow/dags/test](dockers/airflow/dags/test) | imputation mask correctness |

Each `tests/` folder includes a `conftest.py` that adds its package to
`sys.path`, so the suites can run from any working directory.

---

## Pipeline Artifacts on S3

After a successful run, the data lake layout is:

```
s3://<bucket>/wavestitchplus/<dataset>/
├── <version>/
│   ├── input.csv
│   ├── training.log
│   ├── training_info.json
│   ├── prepared/
│   │   ├── saved_model/                 # *.pth checkpoints
│   │   ├── scaler/{mean.npy, std.npy}
│   │   ├── meta.json
│   │   ├── train_imputed.npy
│   │   ├── train_imputed_denorm.npy
│   │   ├── test_input.csv
│   │   └── test_gt.csv
│   └── inference_results_<version>/
│       ├── imputed.csv
│       ├── trials/                      # extra n_trials outputs (if any)
│       ├── inference.log
│       └── inference_info.json
└── latest_inference/
    ├── imputed.csv
    └── inference_info.json
```

Failed runs upload partial logs to
`wavestitchplus/<dataset>/<version>/error_logs/`.

---

## Project Structure

```
WaveStitchPlus/
├── 6GDALI_Datasets/                     # Raw datasets (gitignored)
│   ├── EUR/
│   └── KUL/
├── dockers/
│   ├── airflow/                         # Airflow image + DAGs
│   │   ├── Dockerfile                   #   apache/airflow:2.9.3-python3.10
│   │   ├── requirements.txt
│   │   └── dags/
│   │       ├── diffusion_models4data_cleaning_augmentation.py   # main DAG
│   │       ├── dq_run_dashboard.py                              # SeaweedFS smoke test
│   │       ├── configs/                 # entry_point.json
│   │       ├── helpers/                 # object_store, dqc_metrics, gx_utils, ...
│   │       ├── imputers/                # mask generation, ReMasker baselines
│   │       └── test/
│   ├── tools/                           # Imputation engines and container assets
│   │   ├── Dockerfile.wavestitchplus-gpu
│   │   ├── Dockerfile.wavestitchplus-cpu     # no-GPU build of the same code
│   │   ├── Dockerfile.darts
│   │   ├── Dockerfile.imputegap
│   │   ├── Dockerfile.pypots
│   │   ├── requirements.txt             # shared base dependencies for WaveStitch+
│   │   ├── build_image.sh
│   │   ├── scripts/run_pipeline.py      # WaveStitch+ container entry point
│   │   ├── WaveStitchPlus_app/          # training / synthesis code (diffusion)
│   │   │   ├── train_improved.py
│   │   │   ├── synthesis_improved.py    # -test_csv / -ignore_col_masks for re-masked eval
│   │   │   ├── run_imputation.py        # v1 runner (baseline-compatible CLI)
│   │   │   ├── run_imputation_v2.py     # v2 runner (locally-anchored)
│   │   │   ├── wsp_v2.py                # v2 anchoring helpers (prior + gap blend)
│   │   │   ├── run_local.sh             # CPU-friendly native launcher
│   │   │   ├── custom_pipeline/         # preprocess, dataset, features, eval
│   │   │   ├── TSImputers/              # SSSDS4Imputer, S4Model, TimeGAN
│   │   │   └── helper/                  # data_utils, training_utils, ...
│   │   ├── Darts_app/                   # Darts baselines (linear/cubic/kalman/...)
│   │   │   ├── run_imputation.py
│   │   │   ├── requirements.txt
│   │   │   ├── build_image.sh
│   │   │   └── run_local.sh
│   │   ├── ImputeGAP_app/               # ImputeGAP baselines (cdrec/iim/stmvl/...)
│   │   │   ├── run_imputation.py
│   │   │   ├── requirements.txt
│   │   │   ├── build_image.sh
│   │   │   └── run_local.sh
│   │   └── PyPOTS_app/                  # PyPOTS baselines (saits/brits/transformer/...)
│   │       ├── run_imputation.py
│   │       ├── requirements.txt
│   │       ├── build_image.sh
│   │       └── run_local.sh
│   ├── docker-compose.yml               # airflow + postgres + seaweedfs (master/filer/s3/volume)
│   ├── start.sh                         # docker compose build && up -d
│   └── test_docker.sh                   # standalone WaveStitch+ run examples
├── notebooks/                           # Jupyter analysis notebooks
│   ├── analysis of the ts imputation.ipynb
│   ├── comparisons.ipynb
│   ├── visual.ipynb
│   └── download_folders.py
├── scripts/                             # Method-comparison + evaluation harnesses
│   ├── auto_impute.py                   #   handoff → impute → final cleaned dataset
│   ├── compare_baselines.py             #   score Darts/PyPOTS on a holdout
│   ├── compare_wsp_v2.py                #   WaveStitch+ v1 vs v2 vs baselines
│   ├── eval_long_gap.py                 #   long-gap regime (depth-bucketed MAE)
│   └── gpu_reproduce.sh                 #   one-command end-to-end run (GPU server)
├── dashboard/                           # Streamlit comparison dashboard
│   ├── app.py                           #   discovers prepared/generated dirs
│   ├── requirements.txt                 #   streamlit + plotly
│   └── run.sh                           #   `bash dashboard/run.sh`
├── LICENSE                              # Apache 2.0
└── README.md
```

---

## License

This project is licensed under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.

---

<!-- ## Citation

If you use WaveStitch+ in your research, please cite:

```bibtex
@software{wavestitchplus_6g_dataops,
  title  = {From Raw to Clean: An End-to-End Pipeline for Time-Series Cleaning with Diffusion Models},
  author = {6G-DALI DataOps},
  year   = {2026},
  url    = {https://github.com/judiths1618/WaveStitchPlus}
}
``` -->
