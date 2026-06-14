"""
dataops — core implementation for the 6G-DALI data process modules.

These orchestration-agnostic building blocks are pure functions over pandas
DataFrames, each exposing a ``METADATA`` descriptor. The public package name for
new integrations is ``data_process_modules``; ``dataops`` remains the core and
backward-compatible import path.

Modules:
  * cleaning        — small pandas cleaning helpers
  * profiling       — TS detection, primary-key detection, column typing
  * ts_checks       — time-series quality checks (gaps / missing / outliers)
  * tabular_checks  — tabular quality checks
  * transform       — preprocessing (coerce / regularize / scale / features)
  * split           — train/test split + eval-holdout masking (1:1 reproducible)
  * remediation     — per-issue cleaning after the quality checks
  * imputation_catalog — methods advertised in the imputation handoff
  * registry        — MANIFEST of all module metadata (host-system discovery)
"""
from __future__ import annotations

from . import (
    cleaning,
    config,
    imputation_catalog,
    imputation_runner,
    profiling,
    ts_checks,
    tabular_checks,
    transform,
    split,
    remediation,
    registry,
    validation,
)
from .registry import MANIFEST

__version__ = "0.2.0"

__all__ = [
    "cleaning", "config", "imputation_catalog", "imputation_runner", "profiling",
    "ts_checks", "tabular_checks", "transform", "split", "remediation",
    "registry", "validation", "MANIFEST", "__version__",
]
