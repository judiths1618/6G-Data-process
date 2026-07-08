"""
Great Expectations context shim — no Airflow, no on-disk project required.

The Airflow DAG used ``gx.get_context(context_root_dir="/opt/airflow/great_expectations")``
(see ``helpers/gx_utils.py``). That path only exists inside the Airflow image.
For a standalone tool we use an **ephemeral** GE context, which keeps suites and
validation results in memory — exactly what these one-shot checks need.

A ``context_root_dir`` can still be passed (or set via ``DATAOPS_GX_ROOT``) to
reuse a persisted GE project when one is available.
"""
from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from functools import lru_cache


@lru_cache(maxsize=4)
def get_gx_context(context_root_dir: str | None = None):
    """Return a GE context: file-backed if a root dir is given, else ephemeral.

    Cached so repeated checks in one process reuse a single context (building a
    context is the slow part of a GE run).
    """
    import great_expectations as gx

    root = context_root_dir or os.environ.get("DATAOPS_GX_ROOT")
    if root and os.path.isdir(root):
        return gx.get_context(context_root_dir=root)
    # Ephemeral, in-memory context — the default for standalone tool runs.
    # ``mode=`` is accepted by GX >=0.16; fall back for older signatures.
    try:
        return gx.get_context(mode="ephemeral")
    except TypeError:
        return gx.get_context()


@contextmanager
def suppress_validator_result_format_warning():
    """Silence GX's checkpoint warning for immediate in-memory validations."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`result_format` configured at the Validator-level.*",
            category=UserWarning,
            module=r"great_expectations\.expectations\.expectation",
        )
        yield


def summarize_gx(gx_result, *, max_failed: int = 25) -> dict:
    """Condense a GX validation result into a JSON-friendly pass/fail summary.

    Returns ``{evaluated, passed, failed, failed_expectations:[{expectation,
    column, unexpected_percent}]}`` so consumers (report, dashboard) can show
    *which* expectations failed and by how much, not just an overall boolean.
    """
    results = gx_result.get("results", []) if hasattr(gx_result, "get") else gx_result["results"]
    failed: list[dict] = []
    for r in results:
        if r.get("success"):
            continue
        cfg = r.get("expectation_config", {}) or {}
        kwargs = cfg.get("kwargs", {}) or {}
        res = r.get("result", {}) or {}
        pct = res.get("unexpected_percent")
        failed.append({
            "expectation": cfg.get("expectation_type"),
            "column": kwargs.get("column"),
            "unexpected_percent": round(float(pct), 2) if isinstance(pct, (int, float)) else None,
        })
    failed.sort(key=lambda f: (f["unexpected_percent"] is None, -(f["unexpected_percent"] or 0)))
    return {
        "evaluated": len(results),
        "passed": sum(1 for r in results if r.get("success")),
        "failed": len(failed),
        "failed_expectations": failed[:max_failed],
    }
