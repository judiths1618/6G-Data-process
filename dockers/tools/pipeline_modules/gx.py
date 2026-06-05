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
