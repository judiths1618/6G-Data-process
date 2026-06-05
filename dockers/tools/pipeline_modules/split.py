"""
split — train/test split + evaluation-holdout masking.

Reproduces the existing ``prepared_<subset>/`` split **1:1** by reusing the
exact primitives from the canonical preprocessor (``train_test_split_by_time``
and ``make_eval_holdout_mask``) and the same masking loop. Given the same
``split_ratio / holdout_frac / holdout_block_size / seed`` and the same
(regularized) input, ``train_test`` yields byte-identical ``train.csv`` /
``test_input.csv`` / ``test_gt.csv`` / ``eval_holdout_mask.npy``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

# Reuse the verbatim primitives so the split stays bit-identical to preprocess_csv.
from ._preprocess_impl import make_eval_holdout_mask, train_test_split_by_time

__all__ = ["Splits", "train_test", "METADATA"]


@dataclass
class Splits:
    train: pd.DataFrame        # full train rows (== train.csv)
    test_input: pd.DataFrame   # test rows with eval cells set to NaN (== test_input.csv)
    test_gt: pd.DataFrame      # test rows, ground truth (== test_gt.csv)
    eval_mask: np.ndarray      # 1-D bool holdout mask over test rows (== eval_holdout_mask.npy)
    meta: dict                 # split params + row counts


def train_test(
    df: pd.DataFrame,
    meta: Optional[dict] = None,
    *,
    target_cols: Optional[List[str]] = None,
    split_ratio: float = 0.8,
    holdout_frac: float = 0.15,
    holdout_block_size: int = 5,
    seed: int = 0,
) -> Splits:
    """Time-ordered split + block-structured eval holdout on the test segment.

    ``target_cols`` defaults to ``meta['target_cols']`` (then to every non-time
    column). The holdout hides only *observed* target cells, exactly as
    ``preprocess_csv`` does.
    """
    if target_cols is None:
        target_cols = (meta or {}).get("target_cols")
    if target_cols is None:
        time_col = (meta or {}).get("time_col")
        target_cols = [c for c in df.columns if c != time_col]

    train_df, test_df = train_test_split_by_time(df, split_ratio)

    t_test = len(test_df)
    # Match preprocess_csv: sample the holdout over OBSERVED test rows (rows with
    # at least one non-NaN target cell), so the recorded mask equals the rows that
    # actually hide a value. Passing this keeps split 1:1 with the preprocessor.
    tcols_present = [c for c in target_cols if c in test_df.columns]
    observed_row = test_df[tcols_present].notna().any(axis=1).to_numpy()
    holdout_1d = make_eval_holdout_mask(
        t_test, holdout_frac, holdout_block_size, seed,
        observed_row_mask=observed_row,
    )

    test_input = test_df.copy()
    hidden_count = 0
    for c in target_cols:
        if c not in test_input.columns:
            continue
        observed = test_input[c].notna().to_numpy()
        hide = holdout_1d & observed
        test_input.loc[test_input.index[hide], c] = np.nan
        hidden_count += int(hide.sum())

    out_meta = dict(meta or {})
    out_meta.update({
        "split_ratio": split_ratio,
        "holdout_frac": holdout_frac,
        "holdout_block_size": holdout_block_size,
        "seed": seed,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "holdout_rows": int(holdout_1d.sum()),
        "hidden_values": hidden_count,
    })

    return Splits(
        train=train_df,
        test_input=test_input,
        test_gt=test_df,
        eval_mask=holdout_1d.astype(np.bool_),
        meta=out_meta,
    )


METADATA = {
    "name": "split",
    "version": "0.1.0",
    "category": "split",
    "summary": "Time-ordered train/test split with block-structured eval holdout (1:1 with preprocess_csv).",
    "entrypoint": "pipeline_modules.split:train_test",
    "gpu": False,
    "dependencies": ["pandas", "numpy"],
    "inputs": {
        "df": {"type": "DataFrame", "required": True},
        "meta": {"type": "dict", "default": None},
        "target_cols": {"type": "list[str]", "default": None},
        "split_ratio": {"type": "float", "default": 0.8},
        "holdout_frac": {"type": "float", "default": 0.15},
        "holdout_block_size": {"type": "int", "default": 5},
        "seed": {"type": "int", "default": 0},
    },
    "outputs": {
        "result": {
            "type": "Splits",
            "schema": "splits",
            "keys": ["train", "test_input", "test_gt", "eval_mask", "meta"],
        },
    },
    "artifacts": [
        "train.csv", "test_input.csv", "test_gt.csv", "eval_holdout_mask.npy",
    ],
}
