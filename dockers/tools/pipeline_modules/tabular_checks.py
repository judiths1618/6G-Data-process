"""Compatibility shim for the package-level DataOps tabular checks."""
from dataops.tabular_checks import *  # noqa: F401,F403
from dataops.tabular_checks import METADATA as _DATAOPS_METADATA

METADATA = {**_DATAOPS_METADATA, "entrypoint": "pipeline_modules.tabular_checks:run"}
