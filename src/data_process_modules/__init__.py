"""
data_process_modules — reusable, orchestration-agnostic data processing modules.

This package is the public "data process modules" layout requested for the
project. It re-exports the implementation from :mod:`dataops` so existing code
can keep importing ``dataops`` while new local Python, Airflow, and Docker
integrations can depend on ``data_process_modules``.
"""
from __future__ import annotations

from . import (
    cleaning,
    config,
    imputation_catalog,
    imputation_runner,
    profiling,
    registry,
    remediation,
    split,
    tabular_checks,
    transform,
    ts_checks,
    validation,
)
from .registry import MANIFEST

__version__ = "0.2.0"

__all__ = [
    "MANIFEST",
    "__version__",
    "cleaning",
    "config",
    "imputation_catalog",
    "imputation_runner",
    "profiling",
    "registry",
    "remediation",
    "split",
    "tabular_checks",
    "transform",
    "ts_checks",
    "validation",
]
