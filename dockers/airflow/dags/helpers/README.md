# Airflow DAG helpers

This package holds support code for
`diffusion_models4data_cleaning_augmentation.py`. Keep orchestration decisions
in the DAG and reusable low-level behavior here.

| File | Responsibility |
|---|---|
| `object_store.py` | Read and write DataFrames through the configured object store. |
| `utils.py` | Dataset profiling helpers, including time-series and primary-key detection. |
| `gx_utils.py` | Great Expectations context helpers used by DAG checks. |
| `ts_utils.py` | Time-series diagnostics such as timestamp gap detection. |
| `dqc_utils.py`, `dqc_metrics_methods.py` | Data-quality report shaping and metrics. |
| `clean_dirty_data.py` | Cleaning task router, baseline imputation helpers, and WaveStitch+ container launch. |
| `preprocess.py`, `preprocessor.py` | Prepared-bundle preprocessing code used by the current pipeline paths. |

For new orchestration-independent integrations, prefer the extracted
`dockers/tools/pipeline_modules/` package where it covers the needed behavior.
