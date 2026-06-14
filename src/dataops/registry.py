"""
registry — the discovery surface the host system reads.

Aggregates every module's ``METADATA`` into ``MANIFEST`` so an external
orchestrator can enumerate the available tools, validate parameters, and chain
them. Dump it as JSON with ``python -m dataops manifest``.
"""
from __future__ import annotations

from typing import Dict, List

from . import (
    cleaning, config, imputation_catalog, imputation_runner, profiling,
    ts_checks, tabular_checks, transform, split, remediation,
)
from .validation import pandera_schemas

_MODULES = [
    cleaning,
    config,
    profiling,
    ts_checks,
    tabular_checks,
    remediation,
    pandera_schemas,
    transform,
    split,
    imputation_catalog,
    imputation_runner,
]

# name -> METADATA descriptor
MANIFEST: Dict[str, dict] = {m.METADATA["name"]: m.METADATA for m in _MODULES}


def list_modules() -> List[str]:
    """Names of all registered modules."""
    return list(MANIFEST.keys())


def get(name: str) -> dict:
    """Return the METADATA descriptor for ``name`` (KeyError if unknown)."""
    return MANIFEST[name]
