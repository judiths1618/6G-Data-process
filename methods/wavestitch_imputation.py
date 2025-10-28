"""Compatibility wrapper for the top-level :mod:`wavestitch_imputation` module."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_module = import_module("wavestitch_imputation")

__all__ = getattr(_module, "__all__", []) or [
    name for name in dir(_module) if not name.startswith("_")
]

globals().update({name: getattr(_module, name) for name in __all__})


def __getattr__(name: str) -> Any:  # pragma: no cover - convenience passthrough
    return getattr(_module, name)


if __name__ == "__main__":  # pragma: no cover - CLI passthrough
    _module.main()
