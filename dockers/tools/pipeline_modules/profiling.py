"""Compatibility shim for the package-level DataOps profiling helpers."""
from dataops.profiling import *  # noqa: F401,F403
from dataops.profiling import METADATA as _DATAOPS_METADATA

METADATA = {**_DATAOPS_METADATA, "entrypoint": "pipeline_modules.profiling:profile"}
