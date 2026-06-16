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
