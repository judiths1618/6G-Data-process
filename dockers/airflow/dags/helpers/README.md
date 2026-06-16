# Airflow DAG helpers

This package contains support code for
`diffusion_models4data_cleaning_augmentation.py`. Keep orchestration decisions
inside the DAG, and keep reusable low-level behavior in these helper modules.

| File | Responsibility |
|---|---|
| `object_store.py` | Reads and writes DataFrames through the configured object store. |
| `utils.py` | Provides dataset profiling helpers, including time-series and primary-key detection. |
| `gx_utils.py` | Great Expectations context helpers used by DAG checks. |
| `ts_utils.py` | Time-series diagnostics such as timestamp gap detection. |
| `dqc_utils.py`, `dqc_metrics_methods.py` | Data-quality report shaping and metrics. |
| `clean_dirty_data.py` | Cleaning task router, baseline imputation helpers, and WaveStitch+ container launch. |
| `preprocess.py`, `preprocessor.py` | Prepared-bundle preprocessing code used by the current pipeline paths. |

For new integrations that should not depend on Airflow, prefer the extracted
`dockers/tools/pipeline_modules/` package whenever it covers the behavior you
need.
