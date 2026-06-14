"""Discovery manifest for the public data_process_modules package."""
from __future__ import annotations

from typing import Dict, List

from dataops.registry import MANIFEST as _DATAOPS_MANIFEST


def _rewrite_entrypoint(metadata: dict) -> dict:
    rewritten = dict(metadata)
    entrypoint = rewritten.get("entrypoint")
    if isinstance(entrypoint, str) and entrypoint.startswith("dataops."):
        rewritten["entrypoint"] = "data_process_modules." + entrypoint[len("dataops."):]
    return rewritten


MANIFEST: Dict[str, dict] = {
    name: _rewrite_entrypoint(metadata)
    for name, metadata in _DATAOPS_MANIFEST.items()
}


def list_modules() -> List[str]:
    """Names of all registered modules."""
    return list(MANIFEST.keys())


def get(name: str) -> dict:
    """Return the METADATA descriptor for ``name`` (KeyError if unknown)."""
    return MANIFEST[name]
