"""Compatibility wrapper exposing :mod:`evaluation_pipeline` under ``methods``."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_module = import_module("evaluation_pipeline")

__all__ = getattr(_module, "__all__", []) or [
    name for name in dir(_module) if not name.startswith("_")
]

globals().update({name: getattr(_module, name) for name in __all__})


def __getattr__(name: str) -> Any:  # pragma: no cover - convenience passthrough
    return getattr(_module, name)
