# wavestitch_app/io.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def write_imputed_csv(
    *,
    original_csv: str,
    time_col: str,
    time_index: np.ndarray,        # string timestamps aligned to regularized grid
    value_cols: List[str],
    imputed_values: np.ndarray,    # [T, D] unscaled
    output_csv: str,
) -> None:
    """
    Writes a clean time-indexed CSV:
      time_col + value_cols
    We don't preserve all original columns (because arbitrary CSV → regular grid).
    If you need to preserve extra columns, we can merge later.
    """
    df_out = pd.DataFrame(imputed_values, columns=value_cols)
    df_out.insert(0, time_col, pd.to_datetime(time_index))
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)


def write_report(report: dict, report_json: str) -> None:
    Path(report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(report_json).write_text(json.dumps(report, indent=2))
