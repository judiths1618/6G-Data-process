"""Unit tests for the metadata manifest (host-system discovery surface)."""
from __future__ import annotations

import importlib
import json

import pytest

from pipeline_modules import registry

EXPECTED_MODULES = {"profiling", "ts_checks", "tabular_checks", "transform", "split"}
VALID_CATEGORIES = {"profiling", "quality_check", "transform", "split"}
REQUIRED_KEYS = {
    "name", "version", "category", "summary", "entrypoint",
    "gpu", "dependencies", "inputs", "outputs",
}


def test_manifest_lists_all_modules():
    assert set(registry.list_modules()) == EXPECTED_MODULES
    assert set(registry.MANIFEST) == EXPECTED_MODULES


def test_get_returns_descriptor():
    md = registry.get("split")
    assert md["name"] == "split"
    with pytest.raises(KeyError):
        registry.get("does_not_exist")


@pytest.mark.parametrize("name", sorted(EXPECTED_MODULES))
def test_metadata_schema(name):
    md = registry.MANIFEST[name]
    assert REQUIRED_KEYS <= set(md), f"{name} missing keys: {REQUIRED_KEYS - set(md)}"
    assert md["name"] == name
    assert md["category"] in VALID_CATEGORIES
    assert isinstance(md["gpu"], bool)
    assert isinstance(md["dependencies"], list) and md["dependencies"]
    assert isinstance(md["inputs"], dict) and md["inputs"]
    assert isinstance(md["outputs"], dict) and md["outputs"]


@pytest.mark.parametrize("name", sorted(EXPECTED_MODULES))
def test_entrypoint_resolves_to_callable(name):
    """'pkg.module:function' must import and be callable."""
    mod_path, _, func = registry.MANIFEST[name]["entrypoint"].partition(":")
    assert mod_path.startswith("pipeline_modules.")
    mod = importlib.import_module(mod_path)
    assert callable(getattr(mod, func))


def test_manifest_is_json_serializable():
    json.dumps(registry.MANIFEST, default=str)
