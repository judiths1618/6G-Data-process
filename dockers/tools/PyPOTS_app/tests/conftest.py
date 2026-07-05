"""Path setup for PyPOTS_app tests.

``run_imputation.py`` exists in several apps (Darts_app, PyPOTS_app,
WaveStitchPlus_app). Under pytest's default import mode, modules are cached by
basename, so collecting two of those test suites together would make the second
``import run_imputation`` resolve to the first. We therefore load *this* app's
module under a unique name (``pypots_run_imputation``) via importlib. The
windowing helpers are pure NumPy, so no pypots import is triggered.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

_spec = importlib.util.spec_from_file_location(
    "pypots_run_imputation", APP_DIR / "run_imputation.py"
)
pypots_run_imputation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pypots_run_imputation)
sys.modules["pypots_run_imputation"] = pypots_run_imputation
