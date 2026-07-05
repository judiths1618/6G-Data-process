"""Path setup for the Airflow DAG tests — adds ``dockers/airflow/dags`` to
sys.path so ``from helpers... import`` / ``from imputers... import`` resolve
regardless of the directory pytest is launched from."""
from __future__ import annotations

import sys
from pathlib import Path

DAGS_DIR = Path(__file__).resolve().parents[1]
if str(DAGS_DIR) not in sys.path:
    sys.path.insert(0, str(DAGS_DIR))
