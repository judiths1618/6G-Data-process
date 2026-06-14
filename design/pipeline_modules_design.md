# pipeline_modules — design doc

Status: **implemented (v0.1)** · Date: 2026-05-21

**Scope:** `src/data_process_modules/` is a set of reusable
*data-pipeline modules* — profiling, quality checks, preprocessing/transformation,
and splitting — the **non-imputation** building blocks for an end-to-end pipeline.
The implementation is shared with the backward-compatible `src/dataops/` package;
the Docker-local `dockers/tools/pipeline_modules/` copy is no longer the target
integration surface.

The imputation step is the sibling apps (`Darts_app`, `ImputeGAP_app`,
`PyPOTS_app`, `WaveStitchPlus_app`); these modules are their **peers**, not a
layer above them. They will be **integrated into a larger external system** that
owns orchestration, scheduling, and storage. The Airflow DAG under
`dockers/airflow/` is a local reference where this logic currently lives; it is
left **intact** and is not the integration target.

## 1. Design principles

1. **Library-first, orchestration-agnostic.** Each method is an importable pure
   function. No Airflow, no XCom, no mandatory S3, no global context. The host
   system feeds data in and decides what to do with the output.
2. **DataFrames in, (DataFrame + report) out.** The canonical signature is
   `transform(df, **params) -> (df_out, report_dict)` (or `-> report_dict` for
   pure checks). Reports are plain JSON-serializable dicts.
3. **I/O is a thin, separable layer.** Logic never reads/writes storage itself.
   A small `io_utils` adapter wraps a function so it *can* run from a path or
   `s3://…`, but the function underneath only sees DataFrames.
4. **CLI is a convenience, not the contract.** `python -m data_process_modules <method>` exists
   for standalone runs and quick testing; the integration surface is `import data_process_modules`.
5. **No hidden coupling between methods.** Each is usable on its own; they
   compose by passing DataFrames/artifacts, not by a shared runtime.

---

## 2. Package layout

```
src/data_process_modules/              # importable package — NOT a Docker app
├── __init__.py            # re-exports the modules
├── profiling.py           # is_time_series / primary-key / column typing
├── ts_checks.py           # time-series quality checks (gaps, missing, outliers)
├── tabular_checks.py      # tabular quality checks
├── transform.py           # coerce / regularize / scale / cond-features (wraps _preprocess_impl)
├── split.py               # train/test split + eval-holdout masking (1:1)
├── _preprocess_impl.py    # verbatim copy of helpers/preprocess.py (pure)
├── registry.py            # MANIFEST: aggregates each module's METADATA
├── io_utils.py            # OPTIONAL local|S3 adapter (not imported by logic)
├── gx.py                  # OPTIONAL GX context shim (no Airflow)
├── cli.py / __main__.py   # OPTIONAL argparse dispatcher (python -m data_process_modules)
└── validation/            # Pandera validation helpers shared with the minimal pipeline
```

Logic is **extracted**, from the top design, which can be used for Airflow data pipeline implementation and deployment. 

| Module            | Lift logic from |
|-------------------|-----------------|
| `profiling`       | `helpers/utils.py::analyze_csv_time_series_df`, `::detect_primary_key` |
| `ts_checks`       | DAG `ts_qc()` + `helpers/ts_utils.py::detect_time_gaps` |
| `tabular_checks`  | DAG `qc()` |
| `transform`       | `helpers/preprocess.py` (canonical) — retire the WaveStitchPlus copy |
| `split`           | the split/holdout portion of `preprocess.py` |

---

## 3. Public Python API (the integration surface)

The host system imports and calls these directly; no files required.

```python
from data_process_modules import profiling, ts_checks, tabular_checks, transform, split

prof = profiling.profile(df, timestamp_col="time")          # -> dict
report = ts_checks.run(df, ts_col="time", gap_factor=1.5)    # -> dict
report = tabular_checks.run(df)                              # -> dict

df_t, meta = transform.preprocess(                           # -> (DataFrame, dict)
    df, timestamp_col="time", target_cols=[...], regularize=True)

parts = split.train_test(                                    # -> dataclass / dict
    df_t, meta, split_ratio=0.8, holdout_frac=0.15, holdout_block_size=5, seed=0)
# parts.train, parts.test_input, parts.test_gt, parts.eval_mask
```

Signatures (params shown are the tunables; all have sane defaults):

| Method | Signature → output |
|---|---|
| `profiling.profile` | `(df, timestamp_col=None, sample_ratio=0.9) -> dict{is_time_series, timestamp_column, detected_type, target_cols, primary_key, shape, columns}` |
| `ts_checks.run` | `(df, ts_col, gap_factor=1.5, min_gap_seconds=None, miss_threshold=0.98, outlier_q=0.01) -> dict{mode, gx_passed, issues{ts_gaps,missing,outliers}, recommendations, summary}` |
| `tabular_checks.run` | `(df, miss_threshold_numeric=0.95, miss_threshold_cat=0.90, outlier_q=0.01) -> dict{mode, gx_passed, missing_columns, outlier_columns, failed_columns, primary_key, recommendations, summary}` |
| `transform.preprocess` | `(df, timestamp_col=None, target_cols=None, regularize=True) -> (DataFrame, meta_dict)` |
| `split.train_test` | `(df, meta, split_ratio=0.8, holdout_frac=0.15, holdout_block_size=5, seed=0) -> Splits{train, test_input, test_gt, eval_mask, meta}` — ports `train_test_split_by_time` + `make_eval_holdout_mask` verbatim |

Return dicts are kept **field-compatible with today's** `qc_result` / `ts_result`
/ `meta.json` so existing consumers (dashboard, baseline apps) need no changes.

---

## 4. Artifact shape (only when persisted)

When a consumer *chooses* to persist, the layout is today's `prepared_<subset>/`
bundle (so the baseline apps + dashboard keep working). This is a serialization
detail of the I/O adapter, **not** part of the method contract.

```
<work-dir>/
├── meta.json            # transform.preprocess  (keys: time_col, base_dt,
│                        #   target_cols, cond_cols, all_model_cols,
│                        #   units_converted, split_ratio, holdout_frac,
│                        #   original_rows, regularized_rows, train_rows,
│                        #   test_rows, clip_recommendation,
│                        #   preprocessing_version, notes)
├── scaler/{scale.npy,std.npy}
├── col_masks/  ·  outlier_report.json
├── train.csv  ·  test_input.csv  ·  test_gt.csv  ·  eval_holdout_mask.npy
└── *_report.json        # ts_checks / tabular_checks / profiling output
```

---

## 5. Optional CLI + I/O adapter (`cli.py`, `io_utils.py`)

For standalone runs only. The CLI wraps a method with the local|S3 adapter:

```bash
python -m data_process_modules ts-qc      --input data.csv            --output report.json
python -m data_process_modules preprocess --input s3://bucket/raw.csv --prepared-dir s3://bucket/prep/
python -m data_process_modules split      --prepared-dir ./prepared_amf
```

`io_utils` resolves `--input/--output/--prepared-dir` to local FS or boto3 by
`s3://` prefix (env: `S3_ENDPOINT/S3_ACCESS_KEY/S3_SECRET_KEY/S3_BUCKET`, reusing
the client style in `helpers/object_store.py`). **Methods never import
`io_utils`** — the CLI calls `io_utils.read_*`, hands a DataFrame to the method,
and `io_utils.write_*` the result. This keeps the library pure.

---

## 6. Status (v0.1)

Built and verified (import / `profiling` / `split` 1:1 determinism / `transform`
wiring / `manifest`). The two GX-backed checks (`ts_checks`, `tabular_checks`)
are verbatim ports of DAG `ts_qc()` / `qc()` and run wherever
`great_expectations` is installed (the Airflow image; declared in
`requirements.txt`).

Remaining / nice-to-have:
- Parity tests pinning each module's output against a captured baseline on the
  four EUR subsets (no Airflow needed).
- Retire the duplicate preprocessors (`WaveStitchPlus_app/custom_pipeline/preprocess.py`,
  `helpers/preprocessor.py`) once parity holds. The original Airflow DAG stays intact.

---

## 7. Module metadata (for host-system integration)

Because the host system must discover and wire these tools, **every module
exposes a `METADATA` descriptor** and the package aggregates them into a
`data_process_modules.registry.MANIFEST` (also dumpable via
`python -m data_process_modules manifest`). Schema per module:

```python
METADATA = {
  "name": "ts_checks",
  "version": "0.1.0",
  "category": "quality_check",        # quality_check | profiling | transform | split
  "summary": "Time-series quality checks: gaps, missingness, outliers.",
  "entrypoint": "data_process_modules.ts_checks:run",
  "gpu": False,
  "dependencies": ["pandas", "numpy", "great_expectations"],
  "inputs": {
     "df":      {"type": "DataFrame", "required": True},
     "ts_col":  {"type": "str", "required": True},
     "gap_factor": {"type": "float", "default": 1.5},
     # ... one entry per tunable
  },
  "outputs": {
     "report": {"type": "dict", "schema": "ts_quality_report",
                "keys": ["mode","gx_passed","issues","recommendations","summary"]},
  },
  "artifacts": ["qc_report.json"],    # what the I/O adapter persists, if used
}
```

`data_process_modules.registry` exposes `MANIFEST: dict[str, METADATA]`, `get(name)`, and
`list_modules()`. This is the contract the larger system reads to enumerate
available tools, validate params, and chain them.

## 8. Open questions

1. **`profiling` + `ts_checks` merge?** Both load the CSV and inspect timestamps.
   Kept separate to match the current functions; could be one call. *(Only
   remaining open item — `split` 1:1 and GX-kept are now decided above.)*
```
