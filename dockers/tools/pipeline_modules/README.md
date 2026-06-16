# pipeline_modules

Reusable **data-pipeline modules** for the 6G-DALI stack: profiling, quality
checks, preprocessing/transformation, and train/test splitting. These modules
are the non-imputation building blocks for an end-to-end pipeline. The
imputation step is provided by sibling apps (`Darts_app`, `ImputeGAP_app`, `PyPOTS_app`,
`WaveStitchPlus_app`).

**Not a packaged app.** There is no Docker image and no build script. These are
plain importable modules meant to be embedded in a larger system. The checks,
profiling, and split APIs are DataFrame-first and do not import Airflow or
storage clients. `transform.preprocess_csv(...)` intentionally wraps the
existing local prepared-bundle writer so the current imputation apps keep their
artifact contract.

## Modules

| Module | Function | Returns |
|---|---|---|
| `profiling` | `profile(df, timestamp_col=None)` | dict: TS detection, primary key, column typing |
| `ts_checks` | `run(df, ts_col)` | dict: gaps / missingness / outliers / GX result (`ts_result`) |
| `tabular_checks` | `run(df)` | dict: missing / outliers / PK / GX result (`qc_result`) |
| `transform` | `preprocess_csv(input_csv, output_dir, ...)` | meta dict; writes the full local `prepared/` bundle (bit-compatible with the existing pipeline) |
| `split` | `train_test(df, meta, ...)` | `Splits(train, test_input, test_gt, eval_mask, meta)`; reproduces the existing holdout **1:1** |

## Use as a library (the integration surface)

```python
from pipeline_modules import profiling, split, ts_checks

prof = profiling.profile(df, timestamp_col="time")
report = ts_checks.run(df, ts_col=prof["timestamp_column"])
parts = split.train_test(df, prof, seed=0)
# parts.train / parts.test_input / parts.test_gt / parts.eval_mask
```

To reproduce the prepared artifacts used by the baseline apps and dashboard,
use the path-based transform wrapper:

```python
from pipeline_modules import transform

meta = transform.preprocess_csv(
    "raw.csv",
    "./prepared_amf",
    time_col="time",
)
```

## Discovery / metadata

Every module exposes a `METADATA` descriptor. `pipeline_modules.registry`
aggregates those descriptors so a host system can enumerate modules, validate
parameters, and chain them.

```python
from pipeline_modules.registry import MANIFEST, list_modules, get
list_modules()        # ['profiling', 'ts_checks', 'tabular_checks', 'transform', 'split']
get("ts_checks")      # the METADATA dict
```

## Optional CLI (standalone runs only)

The report commands and `split` can use local paths or `s3://bucket/key`
through `io_utils` (environment variables:
`S3_ENDPOINT/S3_ACCESS_KEY/S3_SECRET_KEY`). `preprocess` is local-path only for
now because it writes the complete prepared
bundle in one call.

```bash
python -m pipeline_modules manifest
python -m pipeline_modules profile     --input data.csv      --output profile.json
python -m pipeline_modules ts-qc       --input data.csv      --output report.json --ts-col time
python -m pipeline_modules tabular-qc  --input data.csv      --output report.json
python -m pipeline_modules preprocess  --input raw.csv       --prepared-dir ./prepared_amf --ts-col time
python -m pipeline_modules split       --prepared-dir ./prepared_amf
```

The CLI is a convenience wrapper. The integration surface remains
`import pipeline_modules`.

## Tests

```bash
conda run -n autofeat-6g python -m pytest dockers/tools/pipeline_modules/tests -q
```

`tests/` covers the pure logic in depth: profiling (time-series and primary-key
detection), `split` (including a **bit-for-bit reproduction** check against the
real `eval_holdout_mask.npy` for **all four** EUR subsets), gap detection, the
metadata manifest, and local I/O. `test_subsets.py` also runs profiling and gap
detection against the four real raw datasets
(`6GDALI_Datasets/EUR/6907619/{amf,golang,python,rabbitmq}-*.csv`), skipping if
the data is absent. The two GX-backed checks (`tests/test_checks_gx.py`) are
gated on the **GX 0.18** fluent API targeted by these modules (the project pin),
so they skip in environments with GX 1.x and run in the Airflow image or a
GX-0.18 virtual environment.

## Dependencies

`pandas`, `numpy` (profiling/split/transform); `great_expectations` (the two
check modules); `boto3` (only the optional S3 I/O path). See `requirements.txt`.

## Provenance

The logic is extracted from the existing pipeline so behavior remains aligned:

- `profiling` ← `helpers/utils.py`
- `ts_checks` ← DAG `ts_qc()` + `helpers/ts_utils.py`
- `tabular_checks` ← DAG `qc()`
- `transform` / `split` <- `helpers/preprocess.py` (`_preprocess_impl.py` is a
  verbatim copy; `split` reuses its `make_eval_holdout_mask` /
  `train_test_split_by_time` for 1:1 reproducibility)

The original Airflow pipeline is left intact.
