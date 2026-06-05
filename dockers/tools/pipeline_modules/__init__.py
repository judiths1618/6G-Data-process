"""
pipeline_modules — reusable data-transformation methods and data-pipeline modules for
the 6G-DALI stack.

These are orchestration-agnostic building blocks to be embedded in a larger
system: pure functions over pandas DataFrames, each exposing a ``METADATA``
descriptor. File/S3 I/O and the CLI are optional thin adapters; the integration
surface is ``import pipeline_modules``.

Modules:
  * profiling       — TS detection, primary-key detection, column typing
  * ts_checks       — time-series quality checks (gaps / missing / outliers)
  * tabular_checks  — tabular quality checks
  * transform       — preprocessing (coerce / regularize / scale / features)
  * split           — train/test split + eval-holdout masking (1:1 reproducible)
  * registry        — MANIFEST of all module metadata (host-system discovery)
"""
from __future__ import annotations

from . import profiling, ts_checks, tabular_checks, transform, split, registry
from .registry import MANIFEST

__version__ = "0.1.0"

__all__ = [
    "profiling", "ts_checks", "tabular_checks", "transform", "split",
    "registry", "MANIFEST", "__version__",
]
