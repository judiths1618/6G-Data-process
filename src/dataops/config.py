"""Configuration loading for the minimal DataOps pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "input": "data/raw/input.csv",
    "output": "data/processed/clean.csv",
    # Soft-cleaned frame (before remediation). null → <output_stem>_cleaned.csv.
    "soft_cleaned_output": None,
    # Backward-compatible alias for older configs/reports.
    "cleaned_output": None,
    "report": "reports/dataops_report.json",
    "log_file": "logs/dataops.log",
    "timestamp_col": None,
    "validation": {
        "mode": "auto",
        "expected_columns": [],
        "numeric_bounds": {},
        "missing_threshold": 0.0,
        "require_timestamp_unique": True,
        "require_timestamp_monotonic": True,
        "allow_step_index_timestamp": False,
        # Outlier sentinel band: flag values outside [q, 1-q]; ``outlier_mostly``
        # is the GX pass threshold (give margin over the ~2q tail so the
        # quantile band doesn't trip by construction on continuous columns).
        "outlier_q": 0.01,
        "outlier_mostly": 0.95,
        # Detection is unconditional; this only controls whether remediation
        # rewrites the flagged cells. Off by default — winsorizing replaces real
        # measurements, and a genuine benchmark spike is signal, not noise.
        "clip_outliers": False,
    },
    # Row identity and acquisition-run handling. Datasets like the EUR container
    # benchmarks are parameter sweeps: several rows legitimately share a
    # timestamp because they are different swept conditions, and a backward jump
    # marks a new sweep run rather than a corrupt cell. Treating the timestamp as
    # the sole key deletes real observations and interleaves independent runs.
    "timeline": {
        # The timestamp is the primary key: rows sharing a timestamp are
        # duplicates and are deduplicated, and rows that break monotonicity are
        # dropped. ``sweep_aware`` opts into the alternative model in which the
        # swept factors co-identify a row and a backward jump starts a new run.
        #
        # Extra columns that, with the timestamp, identify a row (sweep mode).
        "key_columns": [],
        # Infer sweep-factor keys when timestamps collide. Off by default:
        # the timestamp alone is the key. Implied by ``sweep_aware``.
        "auto_key_columns": False,
        # aggregate | keep_last | keep_first | none — how tied rows are reduced.
        "collision_policy": "keep_last",
        # drop | sort | none — what happens to rows that go backwards in time.
        # ``drop`` keeps a forward scan (the conventional treatment); ``sort``
        # reorders them into the series instead.
        "disorder_policy": "drop",
        # Treat a backward jump as a new acquisition run (sweep mode only).
        # Implied by ``sweep_aware``.
        "run_detection": False,
        # A backward jump only starts a new run when at least this many rows
        # re-cover the previous run's span.
        "min_run_overlap_rows": 8,
        # Per-campaign regularization: split the timeline where collection
        # paused and give each campaign its own uniform grid + cadence, instead
        # of stretching one grid across the pause. The bundle contract is
        # unchanged, so the imputation runners need no modification.
        "segment_regularization": True,
        "segment_gap_seconds": 86400.0,   # a pause this long starts a campaign
        "min_segment_rows": 32,           # shorter campaigns are dropped
        # Regularize only when EVERY campaign fits a grid. Mixing gridded and
        # irregular campaigns puts two sampling regimes on opposite sides of the
        # chronological train/test split.
        "require_all_segments": True,
        # Sweep-aware mode: co-identify rows by the swept factors, segment
        # acquisition runs, and emit a ``run`` column instead of presenting one
        # flattened time axis. Turns on auto_key_columns + run_detection.
        "sweep_aware": False,
    },
    "imputation": {
        # The pipeline never runs imputation itself; it regularizes the timeline
        # and emits a handoff signal. These pick the (app, method) advertised to
        # the external orchestrator. Leave app/method null to defer the choice.
        "app": None,          # Darts | ImputeGAP | PyPOTS | WaveStitchPlus
        "method": None,       # validated against dataops.imputation_catalog
        "build_bundle": True,  # write the prepared-dir bundle for the apps
        "prepared_dir": None,  # default: alongside output as <output_stem>_prepared/
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` recursively updated by ``override``."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load pipeline config from YAML, falling back to defaults."""
    if not path:
        return deep_merge({}, DEFAULT_CONFIG)

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config file must contain a mapping: {path}")
    return deep_merge(DEFAULT_CONFIG, loaded)


METADATA = {
    "name": "config",
    "version": "0.1.0",
    "category": "configuration",
    "summary": "YAML config loading for DataOps pipeline paths and validation contracts.",
    "entrypoint": "dataops.config:load_config",
    "gpu": False,
    "dependencies": ["pyyaml"],
}
