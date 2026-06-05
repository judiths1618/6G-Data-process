"""
Compatibility shim.

The canonical ``preprocess_csv`` implementation has been moved into the
Airflow pipeline at ``dockers/airflow/dags/helpers/preprocess.py`` so it can
be reused by the data-quality DAG and every in-process baseline imputer
(darts_*, pypots_*) — not just WaveStitchPlus.

This shim re-exports the same public surface so existing
``from custom_pipeline.preprocess import preprocess_csv`` callsites keep
working when the WaveStitchPlus app is run from the repo root.

Note for the WaveStitchPlus Docker image: this shim relies on the airflow
helpers being reachable on the host filesystem (4 levels up). If you build
the ``wavestitchplus-cpu``/``wavestitchplus-gpu`` images and want them to
keep working standalone, also COPY ``dockers/airflow/dags/helpers/preprocess.py``
into the image at a known location and prepend it to ``PYTHONPATH``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
# dockers/tools/WaveStitchPlus_app/custom_pipeline/preprocess.py
#   parents[0] = custom_pipeline
#   parents[1] = WaveStitchPlus_app
#   parents[2] = tools
#   parents[3] = dockers
_AIRFLOW_HELPERS = _HERE.parents[3] / "airflow" / "dags" / "helpers"
if _AIRFLOW_HELPERS.is_dir() and str(_AIRFLOW_HELPERS) not in sys.path:
    sys.path.insert(0, str(_AIRFLOW_HELPERS))

try:
    # The canonical module is now `helpers/preprocess.py` (when the airflow
    # helpers dir is on sys.path, the package shadowing means we import as
    # just `preprocess`).
    from preprocess import *  # type: ignore  # noqa: F401,F403
    from preprocess import (  # noqa: F401
        preprocess_csv,
        coerce_time_column,
        regularize,
        infer_base_dt,
        diagnose_time_range,
        find_segments,
        extract_all_segments,
        extract_longest_segment,
        add_time_features,
        add_gap_structure_features,
        add_per_column_gap_features,
        analyze_outliers,
        compute_scaler_stats,
        train_test_split_by_time,
        make_eval_holdout_mask,
        split_numeric_and_categorical,
        encode_categoricals,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "preprocess_csv is no longer co-located with the WaveStitchPlus app. "
        "Canonical location: dockers/airflow/dags/helpers/preprocess.py. "
        "Either run from the repo root (so this shim can locate the airflow "
        "helpers dir) or bundle that file into the WaveStitchPlus image and "
        "add it to PYTHONPATH."
    ) from e
