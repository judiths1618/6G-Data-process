"""Compatibility shim for the package-level DataOps time-series checks."""
from dataops.ts_checks import *  # noqa: F401,F403
from dataops.ts_checks import _diffs_in_seconds  # noqa: F401
from dataops.ts_checks import METADATA as _DATAOPS_METADATA

METADATA = {**_DATAOPS_METADATA, "entrypoint": "pipeline_modules.ts_checks:run"}
