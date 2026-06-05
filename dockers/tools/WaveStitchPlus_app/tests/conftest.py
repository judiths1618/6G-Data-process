"""
Path setup for WaveStitchPlus_app tests.

Adds ``dockers/tools/WaveStitchPlus_app`` to ``sys.path`` so ``import
run_imputation`` resolves regardless of the directory pytest is launched from.
Importing the module is cheap — the heavy train/synthesis scripts are only
referenced inside functions, not at import time.
"""
from __future__ import annotations

import sys
from pathlib import Path

# .../WaveStitchPlus_app/tests/conftest.py -> parents[1] == WaveStitchPlus_app
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
