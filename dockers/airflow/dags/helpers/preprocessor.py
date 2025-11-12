# """
# Preprocessing utilities for mixed-type tabular datasets.

# Key features
# ------------
# - Data loader that reads CSVs and infers schema from Info/<dataset>.json or heuristics.
# - Cleaning utilities that drop all-null columns.
# - Dataset splitter that separates complete vs. incomplete rows and writes train/test CSVs.
# - Robust schema inference with automatic exclusion of path-like columns from categoricals.
# - Preprocessor that fits/encodes categoricals (Ordinal or One-Hot) on the TRAIN SPLIT ONLY.
# - Safe inverse-decoding of ordinal predictions (clips to valid categories; keeps <UNK> for unknowns).
# - Mask expansion helper (original -> encoded width) for MCAR/MNAR evaluation.
# - Evaluation helpers for numeric MSE and categorical accuracy with a **clear mask convention**.

# Mask convention (IMPORTANT)
# ---------------------------
# Throughout this module we use: **mask == True means OBSERVED (i.e., not missing)**.
# This matches the downstream evaluation helper and avoids confusion.

# If your upstream generator produces masks with True == MISSING, call `invert_mask(mask)` first.

# Dependencies
# ------------
# - numpy, pandas, scikit-learn

# Author: Yuandou, refined by ChatGPT (2025-11-03)
# """
# from __future__ import annotations

# import os
# import json
# import re
# from dataclasses import dataclass
# from typing import Dict, List, Tuple, Optional, Any

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
# from sklearn.model_selection import train_test_split
# # from dqc_utils import load_df_from_minio, save_df_to_minio, s3_join, curated_keys, S3_BUCKET
# # from helpers.preprocessor import prepare_from_csv

# from helpers.dqc_utils import (
#     # IO
#     load_df_from_minio, save_df_to_minio, _s3,
#     PRIMARY_KEY_RAW,TIMESTAMP_COL,DATASET_NAME,
#     # Detection / normalization / QC
#     detect_timestamp_column, build_schema_profile,
#     normalize_ts_for_gap, compute_time_gaps_smart,
#     # Config exported by helpers (single source of truth)
#     PROJECT, TARGET, S3_BUCKET,
#     REPORT_PREFIX, CURATED_PREFIX,
#     DEFAULT_TZ, TS_STD_COL,
#     TS_EXPECTED_FREQ, TS_GAP_TOL_MULT, TS_GROUP_KEYS
# )

# # ---------------- Constants ----------------
# # Prefer env override; fall back to a sane project-relative cache dir
# DATA_DIR = os.getenv(
#     "DATA_DIR",
#     os.path.join(os.getcwd(), "helpers", "datasets")  # local cache dir
# )
# INFO_DIR = "Info"

# # ---------------- Loader ----------------

# def load_dataset(
#     dataname: str,
#     idx: int = 0,
#     mask_type: str = "MCAR",
#     ratio: str = "30",
#     scheme: str = "Ordinal",
#     data_dir: str = DATA_DIR,
# ) -> Tuple[
#     np.ndarray, np.ndarray,  # train_X, test_X
#     np.ndarray, np.ndarray,  # train_mask_orig, test_mask_orig
#     np.ndarray, np.ndarray,  # train_num, test_num
#     Optional[np.ndarray], Optional[np.ndarray],  # train_cat_idx (ordinal), test_cat_idx
#     np.ndarray, np.ndarray,  # extend_train_mask, extend_test_mask
#     Optional[np.ndarray],  # cat_bin_num (legacy placeholder)
#     Dict[str, Any],  # meta
# ]:
#     root = os.path.join(data_dir, dataname)
#     info_path = os.path.join(data_dir, INFO_DIR, f"{dataname}.json")

#     # schema
#     if os.path.exists(info_path):
#         info = _read_json(info_path)
#     else:
#         info = _infer_schema_from_df(_read_csv(os.path.join(root, "data.csv")))
#         os.makedirs(os.path.join(data_dir, INFO_DIR), exist_ok=True)
#         _write_json(info_path, info)

#     # read splits
#     data_df = _read_csv(os.path.join(root, "data.csv"))
#     train_df = _read_csv(os.path.join(root, "train.csv"))
#     test_df = _read_csv(os.path.join(root, "test.csv"))

#     # masks (shape [N, D_original]) — convention: True == observed
#     train_mask_path = os.path.join(root, f"masks/rate{ratio}/{mask_type}/train_mask_{idx}.npy")
#     test_mask_path = os.path.join(root, f"masks/rate{ratio}/{mask_type}/test_mask_{idx}.npy")
#     if not (os.path.exists(train_mask_path) and os.path.exists(test_mask_path)):
#         raise FileNotFoundError(f"Missing mask files under {os.path.dirname(train_mask_path)}")

#     train_mask = np.load(train_mask_path).astype(bool)
#     test_mask = np.load(test_mask_path).astype(bool)

#     # build preprocessor and (re-)use its filtered cat_idx
#     pp = Preprocessor(dataname, data_dir=data_dir)
#     cat_idx = pp.cat_idx  # ensure path-like exclusions take effect
#     num_idx = pp.num_idx

#     # Encoded X
#     train_X = pp.encodeDf(scheme=scheme, df=train_df)
#     test_X = pp.encodeDf(scheme=scheme, df=test_df)

#     # Raw numeric blocks for convenience
#     train_num = (
#         train_df.iloc[:, num_idx].to_numpy(dtype=np.float32) if num_idx else np.empty((len(train_df), 0), dtype=np.float32)
#     )
#     test_num = (
#         test_df.iloc[:, num_idx].to_numpy(dtype=np.float32) if num_idx else np.empty((len(test_df), 0), dtype=np.float32)
#     )

#     # Ordinal integer labels for cats (handy for some pipelines)
#     train_cat_idx_arr = None
#     test_cat_idx_arr = None
#     if cat_idx:
#         train_cats = train_df.iloc[:, cat_idx].astype(str).applymap(lambda x: x.strip())
#         test_cats = test_df.iloc[:, cat_idx].astype(str).applymap(lambda x: x.strip())
#         train_cat_idx_arr = pp._ord.transform(train_cats).astype(np.float32)
#         test_cat_idx_arr = pp._ord.transform(test_cats).astype(np.float32)

#     # Expand masks to encoded width
#     extend_train_mask = pp.extend_mask(train_mask, encoding=scheme)
#     extend_test_mask = pp.extend_mask(test_mask, encoding=scheme)

#     meta = {
#         "columns": list(train_df.columns),
#         "num_idx": num_idx,
#         "cat_idx": cat_idx,
#         "tgt_idx": info.get("target_col_idx", []),
#         "scheme": scheme,
#         "num_features": int(train_num.shape[1]),
#         "encoded_features": int(train_X.shape[1]),
#         "path_like_idx": pp.path_like_idx,
#         "path_like_cols": [data_df.columns[i] for i in pp.path_like_idx],
#         "ohe_cats_per_col": [len(c) for c in pp._ohe.categories_] if cat_idx and scheme.strip().lower() == "ohe" else None,
#     }

#     return (
#         train_X,
#         test_X,
#         train_mask,
#         test_mask,
#         train_num,
#         test_num,
#         train_cat_idx_arr,
#         test_cat_idx_arr,
#         extend_train_mask,
#         extend_test_mask,
#         None,  # cat_bin_num (legacy placeholder)
#         meta,
#     )


# # -------------------- Metrics helpers --------------------

# def invert_mask(mask: np.ndarray) -> np.ndarray:
#     """Utility: flip mask booleans.
#     If your generator uses True==missing, do: observed_mask = invert_mask(missing_mask).
#     """
#     return ~mask


# def mean_std(data: np.ndarray, observed_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Compute mean/std **over observed entries only**.

#     Parameters
#     ----------
#     data : np.ndarray, shape [N, D]
#     observed_mask : np.ndarray, shape [N, D], dtype bool, where True means OBSERVED
#     """
#     obs = observed_mask.astype(np.float32)
#     denom = obs.sum(0)
#     denom[denom == 0] = 1
#     mean = (data * obs).sum(0) / denom
#     var = ((data - mean) ** 2 * obs).sum(0) / denom
#     std = np.sqrt(var)
#     return mean, std


# def get_eval(
#     X_pred: np.ndarray,
#     X_true: np.ndarray,
#     observed_mask: np.ndarray,
#     num_numeric: int,
#     *,
#     cat_edge_case: bool = False,
# ) -> Tuple[float, float]:
#     """
#     Evaluate MSE on numeric cols, accuracy on categorical cols **where observed_mask == True**.
#     Assumes X_pred/X_true are in the SAME encoded space (both Ordinal or both OHE) and the mask was
#     expanded to encoded width when needed (see `Preprocessor.extend_mask`).

#     Returns
#     -------
#     mse : float
#         Mean squared error over observed numeric entries.
#     acc : float
#         Categorical accuracy (%) over observed categorical entries.
#     """
#     if X_true.shape != X_pred.shape or X_true.shape != observed_mask.shape:
#         raise ValueError("Shapes of X_true, X_pred, and observed_mask must match.")

#     # numeric mse
#     num_true = X_true[:, :num_numeric]
#     num_pred = X_pred[:, :num_numeric]
#     num_mask = observed_mask[:, :num_numeric]

#     num_vals = (num_true[num_mask] - num_pred[num_mask]) ** 2
#     mse = float(num_vals.mean()) if num_vals.size else 0.0

#     # categorical acc
#     cat_true = X_true[:, num_numeric:]
#     cat_pred = X_pred[:, num_numeric:]
#     if cat_true.size == 0:
#         return mse, 100.0

#     cat_mask = observed_mask[:, num_numeric:]
#     if cat_edge_case:
#         # Trim whitespace (for decoded string comparisons)
#         cat_true_comp = np.char.strip(cat_true[cat_mask].astype(str))
#         acc_arr = (cat_pred[cat_mask] == cat_true_comp)
#     else:
#         acc_arr = (cat_pred[cat_mask] == cat_true[cat_mask])

#     acc = float(np.sum(acc_arr) * 100.0 / max(1, len(acc_arr)))
#     return mse, acc


# # -------------------- Quick CSV -> dataset helper & CLI --------------------
# # Convenience wrapper to prepare dataset folder from a single CSV file input to 
# def prepare_from_csv(
#     csv_path: str,
#     dataname: str = "Scenario33",
#     *,
#     output_root: str = DATA_DIR,        
#     exclude_cols: Optional[List[str]] = None,
#     time_col: Optional[str] = None,
#     stratify_col: Optional[str] = None,
#     test_size: float = 0.2,
#     random_state: int = 42,
# ) -> Dict[str, Any]:
#     """
#     Convenience wrapper to take a *single* CSV (e.g., Scenario33.csv) and
#     create the folder structure expected by `Preprocessor`:

#     datasets/<dataname>/
#       ├── data.csv           (copy of the input)
#       ├── train.csv          (complete rows split)
#       ├── test.csv
#       └── incomplete.csv

#     Returns a dict with paths and basic stats.
#     """
#     df = pd.read_csv(csv_path)
#     root = os.path.join(output_root, dataname)
#     os.makedirs(root, exist_ok=True)

#     # Always keep a full copy as data.csv
#     data_csv = os.path.join(root, "data.csv")
#     df.to_csv(data_csv, index=False)

#     train_p, test_p, inc_p, meta = split_complete_incomplete(
#         df,
#         exclude_cols=exclude_cols,
#         time_col=time_col,
#         stratify_col=stratify_col,
#         test_size=test_size,
#         random_state=random_state,
#         output_dir=root,
#     )

#     return {
#         "root": root,
#         "data_csv": data_csv,
#         "train_csv": train_p,
#         "test_csv": test_p,
#         "incomplete_csv": inc_p,
#         "meta": meta,
#     }

# # ---------------- Public API ----------------
# __all__ = [
#     "split_complete_incomplete",
#     "drop_all_null_columns",
#     "Preprocessor",
#     "load_dataset",
#     "mean_std",
#     "get_eval",
#     "invert_mask",
# ]

# # ---------------- Utilities ----------------

# _PATH_PATTERNS = [
#     r"^(?:\.{0,2}/).+",  # ./foo  ../foo  /abs/posix
#     r"^[A-Za-z]:\\.+",   # C:\Windows\...
#     r".*/.+\.[A-Za-z0-9]{1,6}$",  # posix with extension
#     r".*\\.+\.[A-Za-z0-9]{1,6}$",  # windows with extension
#     r"^https?://.+",  # URLs
# ]


# def _is_path_like_string(s: str) -> bool:
#     s = (s or "").strip()
#     if not s:
#         return False
#     return any(re.search(p, s) for p in _PATH_PATTERNS)


# def _is_path_like_series(series: pd.Series, sample: int = 200, thresh: float = 0.6) -> bool:
#     s = series.dropna().astype(str)
#     if s.empty:
#         return False
#     if len(s) > sample:
#         s = s.sample(sample, random_state=42)
#     ratio = s.map(_is_path_like_string).mean()
#     return bool(ratio >= thresh)


# # ---------------- IO helpers ----------------
# def _read_csv(path: str) -> pd.DataFrame:
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Missing file: {path}")
#     lower = path.lower()
#     kwargs = dict(encoding="utf-8", low_memory=False)
#     if lower.endswith(".gz"):
#         kwargs["compression"] = "gzip"
#     # pandas engine fallback for tricky CSVs
#     try:
#         return pd.read_csv(path, **kwargs)
#     except Exception:
#         return pd.read_csv(path, engine="python", **kwargs)

# def _read_json(path: str) -> dict:
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Missing file: {path}")
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

# def _write_json(path: str, obj: dict) -> None:
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     tmp = f"{path}.tmp"
#     with open(tmp, "w", encoding="utf-8", newline="\n") as f:
#         json.dump(obj, f, indent=2, ensure_ascii=False)
#         f.flush()
#         os.fsync(f.fileno())
#     os.replace(tmp, path)

# def _write_csv_atomic(df: pd.DataFrame, path: str) -> None:
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     tmp = f"{path}.tmp"
#     df.to_csv(tmp, index=False, encoding="utf-8", line_terminator="\n")
#     os.replace(tmp, path)

# # ---------------- Schema inference (fallback) ----------------
# def _infer_schema_from_df(df: pd.DataFrame, max_enum: int = 100) -> Dict[str, List[int]]:
#     """
#     Heuristic fallback when Info/<dataset>.json is absent.
#     - numeric: integer/float dtypes
#     - categorical: non-numeric with low cardinality AND not path-like
#     - target: none by default
#     Returns column **indices** (stable across splits when using same header order).
#     """
#     num_idx: List[int] = []
#     cat_idx: List[int] = []
#     for i, c in enumerate(df.columns):
#         s = df[c]
#         if pd.api.types.is_numeric_dtype(s):
#             num_idx.append(i)
#         else:
#             if _is_path_like_series(s):
#                 continue
#             nunique = int(s.astype(str).nunique(dropna=True))
#             if nunique <= max_enum:
#                 cat_idx.append(i)
#     return {"num_col_idx": num_idx, "cat_col_idx": cat_idx, "target_col_idx": []}



# # ---------------- Dataset split ----------------
# def split_complete_incomplete(
#     df: pd.DataFrame,
#     *,
#     exclude_cols: Optional[List[str]] = None,  # columns to ignore when checking completeness (e.g., 'time_stamp', file paths)
#     time_col: Optional[str] = None,            # if set, do a chronological split for train/test
#     stratify_col: Optional[str] = None,        # for classification on tabular data (ignored if time_col is used)
#     test_size: float = 0.2,
#     random_state: int = 42,
#     output_dir: str = "datasets/myset",        # directory to write train.csv, test.csv, incomplete.csv, for minio, it's a curated key prefix
#     auto_exclude_path_like: bool = True        # NEW: auto-ignore obvious path/URL columns
# ) -> Tuple[str, str, str, Dict[str, Any]]:
#     """
#     Separates complete vs. incomplete rows (w.r.t. non-excluded columns) and splits the complete set
#     into train/test, writing CSVs + a meta JSON.

#     Returns: (train_path, test_path, incomplete_path, meta_dict)
#     """
#     # 0) Drop fully-null columns up-front
#     df_clean, dropped_all_null = _drop_all_null_columns(df)
#     # df_clean, dropped_all_constant = _drop_constant_columns(df_clean)

#     # Build final exclude set
#     exclude = set(exclude_cols or [])
#     exclude = {c for c in exclude if c in df_clean.columns}

#     # Auto-exclude path/URL columns (they’re “metadata-ish”, not data completeness)
#     if auto_exclude_path_like:
#         for c in df_clean.columns:
#             try:
#                 if _is_path_like_series(df_clean[c]):
#                     exclude.add(c)
#             except Exception:
#                 pass

#     # Guard: ensure time_col exists (if provided)
#     use_time = bool(time_col) and (time_col in df_clean.columns)

#     # Completeness columns = everything except excluded (and optionally time)
#     cols_to_check = [c for c in df_clean.columns if c not in exclude]

#     # Save normalized data.csv that matches the split columns (after drops)
#     # os.makedirs(output_dir, exist_ok=True)
#     data_csv = os.path.join(output_dir, "data.csv")
#     df_clean.to_csv(data_csv, index=False)

#     # 1) separate complete vs incomplete
#     if cols_to_check:
#         complete_mask = df_clean[cols_to_check].notna().all(axis=1)
#     else:
#         # If we excluded everything, consider all rows complete
#         complete_mask = pd.Series(True, index=df_clean.index)


#     df_complete = df_clean.loc[complete_mask].copy()
#     df_incomplete = df_clean.loc[~complete_mask].copy()
    
#     # drop constant columns again for the complete set
#     # df_complete, dropped_all_constant = _drop_constant_columns(df_complete)

#     # 2) ensure time dtype if chronological split
#     if use_time and not np.issubdtype(df_complete[time_col].dtype, np.datetime64):
#         df_complete[time_col] = pd.to_datetime(df_complete[time_col], errors="coerce")

#     # 3) split complete subset
#     if use_time:
#         # sort by time and cut chronologically on non-NaT
#         df_nonat = df_complete.dropna(subset=[time_col]).sort_values(time_col, kind="stable")
#         df_nat   = df_complete[df_complete[time_col].isna()]
#         n = len(df_nonat)
#         if n == 0:
#             # No valid timestamps → fallback to random split
#             train_df, test_df = train_test_split(
#                 df_complete, test_size=test_size, random_state=random_state, shuffle=True
#             )
#         else:
#             split_idx = int((1.0 - test_size) * n)
#             train_df = df_nonat.iloc[:max(0, split_idx)].copy()
#             test_df  = df_nonat.iloc[max(0, split_idx):].copy()
#             # Put NaT rows into train by default so models can still use them if needed
#             if not df_nat.empty:
#                 train_df = pd.concat([train_df, df_nat], axis=0, ignore_index=True)
#     else:
#         # stratified split if feasible, else safe fallback
#         strat = None
#         use_strat = bool(stratify_col) and (stratify_col in df_complete.columns)
#         if use_strat:
#             # Must have >=2 classes and each class must have at least 2 samples for typical splits
#             vc = df_complete[stratify_col].value_counts(dropna=False)
#             if len(vc) >= 2 and (vc.min() >= 2) and (len(df_complete) >= 4):
#                 strat = df_complete[stratify_col]
#         if len(df_complete) <= 1:
#             # Too small to split meaningfully → all train, empty test
#             train_df = df_complete.copy()
#             test_df  = df_complete.iloc[0:0].copy()
#         else:
#             train_df, test_df = train_test_split(
#                 df_complete, test_size=test_size, random_state=random_state, stratify=strat
#             )

#     # 4) write outputs
#     train_path = os.path.join(output_dir, "train.csv")
#     test_path  = os.path.join(output_dir, "test.csv")
#     inc_path   = os.path.join(output_dir, "incomplete.csv")
#     meta_path  = os.path.join(output_dir, "split_meta.json")

#     train_df.to_csv(train_path, index=False)
#     test_df.to_csv(test_path, index=False)
#     df_incomplete.to_csv(inc_path, index=False)

#     meta: Dict[str, Any] = {
#         "rows_total": int(len(df)),
#         "rows_complete": int(len(df_complete)),
#         "rows_incomplete": int(len(df_incomplete)),
#         "dropped_all_null_cols": sorted(dropped_all_null),
#         # "dropped_constant_cols": sorted(dropped_all_constant),
#         "test_size": float(test_size),
#         "exclude_cols": sorted(list(exclude)),
#         "split_mode": "chronological" if use_time else ("stratified" if stratify_col and strat is not None else "random"),
#         "time_col": time_col if use_time else None,
#         "stratify_col": (stratify_col if (stratify_col and not use_time and strat is not None) else None),
#         "outputs": {
#             "data_csv": data_csv,
#             "train_csv": train_path,
#             "test_csv": test_path,
#             "incomplete_csv": inc_path
#         },
#     }
#     # _write_json(meta_path, meta)
#     return train_path, test_path, inc_path, meta

# def split_complete_incomplete_minio(
#     df: pd.DataFrame,
#     *,
#     exclude_cols: Optional[List[str]] = None,   # columns to ignore when checking completeness
#     time_col: Optional[str] = None,             # chronological split if provided
#     stratify_col: Optional[str] = None,         # used only when time_col is None
#     test_size: float = 0.2,
#     random_state: int = 42,
#     output_dir: str = "curated/myset/",         # MinIO key prefix, e.g. "curated/DeepSense/scenario33/"
#     auto_exclude_path_like: bool = True
# ) -> Tuple[str, str, str, Dict[str, Any]]:
#     """
#     Split rows into complete vs incomplete (ignoring excluded cols), then split complete into train/test.
#     Persist CSVs + a split meta JSON to MinIO.

#     Returns (train_key, test_key, incomplete_key, meta_dict)
#     """

#     # --- Normalize prefix (MinIO key, trailing slash) ---
#     prefix = output_dir.strip().split("?", 1)[0].rstrip("/") + "/"

#     # 0) Drop fully-null columns up-front
#     df_clean, dropped_all_null = _drop_all_null_columns(df)

#     # Build exclude set
#     exclude = set(exclude_cols or [])
#     exclude = {c for c in exclude if c in df_clean.columns}

#     # Auto-exclude obvious path/URL columns
#     if auto_exclude_path_like:
#         for c in df_clean.columns:
#             try:
#                 if _is_path_like_series(df_clean[c]):
#                     exclude.add(c)
#             except Exception:
#                 pass

#     # Guard: ensure time_col exists (if provided)
#     use_time = bool(time_col) and (time_col in df_clean.columns)

#     # Completeness columns = everything except excluded
#     cols_to_check = [c for c in df_clean.columns if c not in exclude]

#     # 1) separate complete vs incomplete
#     if cols_to_check:
#         complete_mask = df_clean[cols_to_check].notna().all(axis=1)
#     else:
#         complete_mask = pd.Series(True, index=df_clean.index)

#     df_complete   = df_clean.loc[complete_mask].copy()
#     df_incomplete = df_clean.loc[~complete_mask].copy()

#     # 2) ensure time dtype if chronological split
#     if use_time and not np.issubdtype(df_complete[time_col].dtype, np.datetime64):
#         df_complete[time_col] = pd.to_datetime(df_complete[time_col], errors="coerce")

#     # 3) split complete subset
#     if use_time:
#         # chronological split on valid timestamps
#         df_nonat = df_complete.dropna(subset=[time_col]).sort_values(time_col, kind="stable")
#         df_nat   = df_complete[df_complete[time_col].isna()]
#         n = len(df_nonat)
#         if n == 0 or len(df_complete) <= 1:
#             train_df, test_df = train_test_split(
#                 df_complete, test_size=test_size, random_state=random_state, shuffle=True
#             )
#         else:
#             split_idx = int((1.0 - test_size) * n)
#             train_df = df_nonat.iloc[:max(0, split_idx)].copy()
#             test_df  = df_nonat.iloc[max(0, split_idx):].copy()
#             if not df_nat.empty:
#                 train_df = pd.concat([train_df, df_nat], axis=0, ignore_index=True)
#     else:
#         # stratified if feasible
#         strat = None
#         use_strat = bool(stratify_col) and (stratify_col in df_complete.columns)
#         if use_strat:
#             vc = df_complete[stratify_col].value_counts(dropna=False)
#             if len(vc) >= 2 and (vc.min() >= 2) and (len(df_complete) >= 4):
#                 strat = df_complete[stratify_col]
#         if len(df_complete) <= 1:
#             train_df = df_complete.copy()
#             test_df  = df_complete.iloc[0:0].copy()
#         else:
#             train_df, test_df = train_test_split(
#                 df_complete, test_size=test_size, random_state=random_state, stratify=strat
#             )

#     # --- MinIO keys ---
#     data_key       = f"{prefix}data.csv"
#     train_key      = f"{prefix}train.csv"
#     test_key       = f"{prefix}test.csv"
#     incomplete_key = f"{prefix}incomplete.csv"
#     meta_key       = f"{prefix}split_meta.json"

#     # 4) write outputs to MinIO
#     save_df_to_minio(df_clean,   data_key,       fmt="csv", index=False)
#     save_df_to_minio(train_df,   train_key,      fmt="csv", index=False)
#     save_df_to_minio(test_df,    test_key,       fmt="csv", index=False)
#     save_df_to_minio(df_incomplete, incomplete_key, fmt="csv", index=False)

#     meta: Dict[str, Any] = {
#         "rows_total": int(len(df)),
#         "rows_complete": int(len(df_complete)),
#         "rows_incomplete": int(len(df_incomplete)),
#         "dropped_all_null_cols": sorted(dropped_all_null),
#         "test_size": float(test_size),
#         "exclude_cols": sorted(list(exclude)),
#         "split_mode": "chronological" if use_time else (
#             "stratified" if (stratify_col and (stratify_col in df_clean.columns)) else "random"
#         ),
#         "time_col": time_col if use_time else None,
#         "stratify_col": (stratify_col if (stratify_col and not use_time) else None),
#         "outputs": {
#             "data_csv": data_key,
#             "train_csv": train_key,
#             "test_csv": test_key,
#             "incomplete_csv": incomplete_key,
#             "meta_json": meta_key,
#         },
#     }

#     _s3().put_object(
#         Bucket=S3_BUCKET,
#         Key=meta_key,
#         Body=json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
#         ContentType="application/json",
#     )

#     return train_key, test_key, incomplete_key, meta

# # ---------------- Column cleaning helpers ----------------

# def _drop_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
#     """
#     Drop columns in a DataFrame where all rows have the same value.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input DataFrame to process.

#     Returns
#     -------
#     Tuple[pd.DataFrame, List[str]]
#         - The cleaned DataFrame with constant columns removed.
#         - List of column names that were dropped.
#     """
#     # Identify columns with a single unique value (including NaN)
#     constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]

#     # Drop constant columns
#     df_clean = df.drop(columns=constant_cols)

#     if constant_cols:
#         print(f"Dropped {len(constant_cols)} constant column(s): {constant_cols}")
#     else:
#         print("No constant columns found.")

#     return df_clean, constant_cols


# def _drop_all_null_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
#     """
#     Drop columns that are entirely null/NaN.
#     Returns a new DataFrame and the list of dropped column names.
#     """
#     if df.empty:
#         return df.copy(), []
#     null_cols = [c for c in df.columns if df[c].isna().all()]
#     if not null_cols:
#         return df.copy(), []
#     return df.drop(columns=null_cols), null_cols


# # ---------------- Preprocessor ----------------

# @dataclass
# class _FittedState:
#     num_idx: List[int]
#     cat_idx: List[int]
#     tgt_idx: List[int]
#     path_like_idx: List[int]

# from typing import List, Optional, Dict, Any
# import os, json
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# class Preprocessor:
#     """
#     Robust mixed-type preprocessor for tabular data.
#     - Works with local folders OR MinIO prefixes (e.g., 'curated/.../scenario33/').
#     - Fits encoders on train split only (avoid leakage).
#     - Supports 'OHE' and 'Ordinal'.
#     - Excludes path-like columns from categorical set automatically.
#     """

#     def __init__(self, dataname: str, data_dir: str = DATA_DIR):
#         self.dataset = dataname
#         self.data_dir = data_dir

#         # ---------- Detect local vs MinIO ----------
#         # Local mode if we find a real folder with data.csv; otherwise treat data_dir as MinIO prefix.
#         local_root = os.path.join(data_dir, dataname)
#         local_mode = os.path.isdir(local_root) and os.path.exists(os.path.join(local_root, "data.csv"))

#         if local_mode:
#             # ---------- Local ----------
#             root = local_root
#             info_path = os.path.join(self.data_dir, INFO_DIR, f"{dataname}.json")

#             raw_df    = _read_csv(os.path.join(root, "data.csv"))
#             raw_train = _read_csv(os.path.join(root, "train.csv"))
#             raw_test  = _read_csv(os.path.join(root, "test.csv"))

#             if os.path.exists(info_path):
#                 self.info = _read_json(info_path)
#             else:
#                 self.info = _infer_schema_from_df(raw_df)
#                 os.makedirs(os.path.dirname(info_path), exist_ok=True)
#                 _write_json(info_path, self.info)

#             self._is_minio = False
#             self._info_loc = info_path  # file path

#         else:
#             # ---------- MinIO ----------
#             # Expect keys:
#             #   <prefix>data.csv, train.csv, test.csv, and <prefix>Info/<dataname>.json (optional)
#             prefix = data_dir.rstrip("/") + "/"
#             def _get_df(leaf: str) -> pd.DataFrame:
#                 df, _ = load_df_from_minio(prefix + leaf)
#                 return df

#             raw_df    = _get_df("data.csv")
#             raw_train = _get_df("train.csv")
#             raw_test  = _get_df("test.csv")

#             info_key = f"{prefix}{INFO_DIR.rstrip('/')}/{dataname}.json"
#             try:
#                 obj = _s3().get_object(Bucket=S3_BUCKET, Key=info_key)
#                 self.info = json.loads(obj["Body"].read().decode("utf-8"))
#             except Exception:
#                 self.info = _infer_schema_from_df(raw_df)
#                 _s3().put_object(
#                     Bucket=S3_BUCKET,
#                     Key=info_key,
#                     Body=json.dumps(self.info, ensure_ascii=False, indent=2).encode("utf-8"),
#                     ContentType="application/json",
#                 )

#             self._is_minio = True
#             self._prefix = prefix
#             self._info_loc = info_key  # s3 key

#         # ---------- Harmonize columns & constants ----------
#         df, dropped_all_null = _drop_all_null_columns(raw_df)

#         self.dropped_all_train_constant = [
#             c for c in raw_train.columns if raw_train[c].nunique(dropna=False) == 1
#         ]
#         self.dropped_all_test_constant = [
#             c for c in raw_test.columns if raw_test[c].nunique(dropna=False) == 1
#         ]

#         # Align train/test to data.csv columns
#         self.df = df
#         self.df_train = raw_train.reindex(columns=self.df.columns).copy()
#         self.df_test  = raw_test.reindex(columns=self.df.columns).copy()

#         # Column indices (stable across splits)
#         self.num_idx: List[int] = self.info.get("num_col_idx", [])
#         self.cat_idx: List[int] = self.info.get("cat_col_idx", [])
#         self.tgt_idx: List[int] = self.info.get("target_col_idx", [])
#         self.excluded_idx: List[int] = self.info.get("excluded_col_idx", [])

#         # Exclude path-like from categorical
#         self.path_like_idx: List[int] = []
#         if self.cat_idx:
#             keep_cat = []
#             for i in self.cat_idx:
#                 col = self.df.columns[i]
#                 if _is_path_like_series(self.df[col]):
#                     self.path_like_idx.append(i)
#                 else:
#                     keep_cat.append(i)
#             if self.path_like_idx:
#                 self.cat_idx = keep_cat
#                 self.info["cat_col_idx"] = self.cat_idx

#         # Exclude constant columns from indices (keep in frames)
#         const_names = set(self.dropped_all_train_constant) | set(self.dropped_all_test_constant)
#         if const_names:
#             const_idx = {i for i, n in enumerate(self.df.columns) if n in const_names}
#             self.num_idx = [i for i in self.num_idx if i not in const_idx]
#             self.cat_idx = [i for i in self.cat_idx if i not in const_idx]
#             self.info.setdefault("excluded_constant_cols", sorted(list(const_names)))
#             self.excluded_idx.extend(const_idx)

#         # Also exclude path-like from numeric indices if misclassified
#         bad_num_idx = [i for i in self.num_idx if _is_path_like_series(self.df[self.df.columns[i]])]

#         if bad_num_idx:
#             self.num_idx = [i for i in self.num_idx if i not in bad_num_idx]
#             self.info.setdefault("excluded_path_like_cols", [self.df.columns[i] for i in bad_num_idx])
#             self.excluded_idx.extend(bad_num_idx)

#         # Persist updated info (local file or MinIO object)
#         try:
#             excluded_idx_sorted = sorted(set(self.excluded_idx))
#             self.info["excluded_col_idx"] = excluded_idx_sorted
#             self.info["excluded_col_names"] = [self.df.columns[i] for i in excluded_idx_sorted]

#             if self._is_minio:
#                 _s3().put_object(
#                     Bucket=S3_BUCKET,
#                     Key=self._info_loc,
#                     Body=json.dumps(self.info, ensure_ascii=False, indent=2).encode("utf-8"),
#                     ContentType="application/json",
#                 )
#             else:
#                 os.makedirs(os.path.dirname(self._info_loc), exist_ok=True)
#                 _write_json(self._info_loc, self.info)
#         except Exception:
#             pass

#         # Sanity checks
#         cols = self.df.columns
#         for idx in self.num_idx + self.cat_idx + self.tgt_idx:
#             if idx < 0 or idx >= len(cols):
#                 raise ValueError(f"Column index {idx} out of range for dataset '{dataname}'.")

#         # ---- build encoders (fit on train only) ----
#         self._ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
#         self._ord = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)

#         if self.cat_idx:
#             train_cats = self.df_train.iloc[:, self.cat_idx].astype(str).applymap(lambda x: x.strip())
#             self._ohe.fit(train_cats)
#             self._ord.fit(train_cats)

#         self.numerical_indices_np_end: Optional[int] = None
#         self._cached_cat_feature_counts: Optional[np.ndarray] = None  # for OHE mask expansion

#     # ---------------- Encoding/Decoding on DataFrames (unchanged) ----------------
#     def encodeDf(self, scheme: str = "Ordinal", df: Optional[pd.DataFrame] = None) -> np.ndarray:
#         if df is None:
#             df = self.df
#         if self.num_idx:
#             nums_df = df.iloc[:, self.num_idx].copy()
#             nums_df = nums_df.apply(lambda col: pd.to_numeric(col, errors="coerce"))
#         else:
#             nums_df = pd.DataFrame(index=df.index)
#         cats = df.iloc[:, self.cat_idx].astype(str).applymap(lambda x: x.strip()) if self.cat_idx else None
#         self.numerical_indices_np_end = nums_df.shape[1]
#         scheme_norm = scheme.strip().lower()
#         if scheme_norm == "ohe":
#             if cats is None:
#                 return nums_df.to_numpy(dtype=np.float32)
#             cats_enc = self._ohe.transform(cats)
#             self._cached_cat_feature_counts = np.array([len(c) for c in self._ohe.categories_], dtype=int)
#             return np.concatenate([nums_df.to_numpy(dtype=np.float32), cats_enc], axis=1)
#         if scheme_norm == "ordinal":
#             if cats is None:
#                 return nums_df.to_numpy(dtype=np.float32)
#             cats_enc = self._ord.transform(cats)
#             return np.concatenate([nums_df.to_numpy(dtype=np.float32), cats_enc], axis=1)
#         raise ValueError("Invalid scheme; use 'OHE' or 'Ordinal'.")

#     def encodeNp(self, scheme: str = "Ordinal", arr: Optional[np.ndarray] = None) -> np.ndarray:
#         if arr is None:
#             raise ValueError("arr is required")
#         if self.numerical_indices_np_end is None:
#             raise RuntimeError("numerical_indices_np_end not set. Call encodeDf() first.")
#         nums = arr[:, : self.numerical_indices_np_end]
#         cats = arr[:, self.numerical_indices_np_end :]
#         scheme_norm = scheme.strip().lower()
#         if scheme_norm == "ohe":
#             cats_rounded = np.rint(cats).astype(int)
#             cats_rounded = np.clip(cats_rounded, -1, None)
#             cats_strings = self._inverse_from_ordinal_indices(cats_rounded)
#             cats_strings = pd.DataFrame(cats_strings).astype(str).values
#             cats_ohe = self._ohe.transform(cats_strings)
#             return np.concatenate([nums, cats_ohe], axis=1)
#         if scheme_norm == "ordinal":
#             cats_strings = self._ohe.inverse_transform(cats)
#             cats_ord = self._ord.transform(cats_strings)
#             return np.concatenate([nums, cats_ord], axis=1)
#         raise ValueError("Invalid scheme; use 'OHE' or 'Ordinal'.")

#     def decodeNp(self, scheme: str = "Ordinal", arr: Optional[np.ndarray] = None) -> np.ndarray:
#         if arr is None:
#             raise ValueError("arr is required")
#         if self.numerical_indices_np_end is None:
#             raise RuntimeError("numerical_indices_np_end not set. Call encodeDf() first.")
#         nums = arr[:, : self.numerical_indices_np_end]
#         if arr.shape[1] == self.numerical_indices_np_end or not self.cat_idx:
#             return nums.copy()
#         scheme_norm = scheme.strip().lower()
#         if scheme_norm == "ohe":
#             cats = self._ohe.inverse_transform(arr[:, self.numerical_indices_np_end :])
#             return np.concatenate([nums, cats], axis=1)
#         if scheme_norm == "ordinal":
#             raw = arr[:, self.numerical_indices_np_end :]
#             cats_idx = np.rint(raw).astype(int)
#             for j, cats_j in enumerate(self._ord.categories_):
#                 vmax = len(cats_j) - 1
#                 col = cats_idx[:, j]
#                 mask_ok = col >= 0
#                 col[mask_ok] = np.clip(col[mask_ok], 0, vmax)
#                 cats_idx[:, j] = col
#             cats = self._inverse_from_ordinal_indices(cats_idx)
#             return np.concatenate([nums, cats], axis=1)
#         raise ValueError("Invalid scheme; use 'OHE' or 'Ordinal'.")

#     def _inverse_from_ordinal_indices(self, idx_arr: np.ndarray) -> np.ndarray:
#         out = np.empty_like(idx_arr, dtype=object)
#         for j, cats in enumerate(self._ord.categories_):
#             col = idx_arr[:, j]
#             out_col = np.empty(col.shape[0], dtype=object)
#             unk = col < 0
#             out_col[unk] = "<UNK>"
#             ok = ~unk
#             if ok.any():
#                 ok_idx = np.clip(col[ok], 0, len(cats) - 1)
#                 out_col[ok] = np.asarray(cats, dtype=object)[ok_idx]
#             out[:, j] = out_col
#         return out

#     def extend_mask(self, ori_mask: np.ndarray, encoding: str = "Ordinal", input_mask_is_observed: bool = True) -> np.ndarray:
#         if ori_mask is None or ori_mask.ndim != 2:
#             raise ValueError("ori_mask must be a 2D boolean numpy array with shape (N, D_original).")
#         expected_D = len(self.df.columns)
#         if ori_mask.shape[1] != expected_D:
#             raise ValueError(f"ori_mask has wrong width: got {ori_mask.shape[1]}, expected {expected_D}.")
#         mask = ori_mask.copy()
#         if not input_mask_is_observed:
#             mask = ~mask
#         enc = encoding.strip().lower()
#         if enc == "ordinal" or not self.cat_idx:
#             return mask.copy()
#         if self.numerical_indices_np_end is None:
#             raise RuntimeError("numerical_indices_np_end not set. Call encodeDf() first.")
#         if self._cached_cat_feature_counts is None:
#             self._cached_cat_feature_counts = np.array([len(c) for c in self._ohe.categories_], dtype=int)
#         num_mask = mask[:, self.num_idx] if self.num_idx else np.zeros((mask.shape[0], 0), dtype=bool)
#         cat_mask = mask[:, self.cat_idx] if self.cat_idx else np.zeros((mask.shape[0], 0), dtype=bool)
#         pieces = []
#         for j, width in enumerate(self._cached_cat_feature_counts):
#             mj = cat_mask[:, j][:, None]
#             pieces.append(np.repeat(mj, width, axis=1))
#         cat_mask_expanded = np.concatenate(pieces, axis=1) if pieces else np.zeros((ori_mask.shape[0], 0), dtype=bool)
#         return np.concatenate([num_mask, cat_mask_expanded], axis=1)

# # -------------------- CLI for quick testing --------------------

# if __name__ == "__main__":
#     import argparse
#     import tempfile
#     import time
#     import io
#     import json as _json

#     # Optional: reuse your existing MinIO utils if you have them
#     try:
#         from dqc_utils import (
#             _s3, S3_BUCKET, RAW_PREFIX, CURATED_PREFIX,  # Airflow Variables-backed
#             # If you don't have RAW_PREFIX/CURATED_PREFIX, fall back to 'raw'/'curated'
#         )
#         _HAS_DQC = True
#     except Exception:
#         _HAS_DQC = False
#         S3_BUCKET = None
#         RAW_PREFIX = "raw"
#         CURATED_PREFIX = "curated"

#     parser = argparse.ArgumentParser(description="Preprocess CSV locally or via MinIO data lake.")
#     sub = parser.add_subparsers(dest="mode", required=True)

#     # ---------- Local mode (existing behavior) ----------
#     p_local = sub.add_parser("local", help="Read a local CSV and write local curated folder.")
#     p_local.add_argument("csv", help="Path to input CSV (single table)")
#     p_local.add_argument("--dataname", default="Scenario33")
#     p_local.add_argument("--data_dir", default=os.getenv("DATA_DIR", os.path.join(os.getcwd(), "helpers", "datasets")))
#     p_local.add_argument("--scheme", default="Ordinal", choices=["Ordinal", "OHE"])
#     p_local.add_argument("--time_col", default=None)
#     p_local.add_argument("--stratify_col", default=None)
#     p_local.add_argument("--exclude_cols", default=None, help="Comma-separated")
#     p_local.add_argument("--test_size", type=float, default=0.2)

#     # ---------- MinIO mode ----------
#     p_s3 = sub.add_parser("minio", help="Pull raw from MinIO and push curated outputs back.")
#     p_s3.add_argument("--datasetname", required=True, help="<datasetname> part")
#     p_s3.add_argument("--dataname", required=True, help="<dataname> part (without .csv)")
#     p_s3.add_argument("--bucket", default=None, help="Override bucket (defaults to dqc_utils.S3_BUCKET)")
#     p_s3.add_argument("--raw_prefix", default=None, help="Override raw prefix (defaults to dqc_utils.RAW_PREFIX or 'raw')")
#     p_s3.add_argument("--curated_prefix", default=None, help="Override curated prefix (defaults to dqc_utils.CURATED_PREFIX or 'curated')")
#     p_s3.add_argument("--exclude_cols", default=None, help="Comma-separated")
#     p_s3.add_argument("--time_col", default=None)
#     p_s3.add_argument("--stratify_col", default=None)
#     p_s3.add_argument("--test_size", type=float, default=0.2)
#     p_s3.add_argument("--random_state", type=int, default=42)

#     args = parser.parse_args()

#     if args.mode == "local":
#         exclude_cols = [c.strip() for c in args.exclude_cols.split(",")] if args.exclude_cols else None
#         info = prepare_from_csv(
#             args.csv,
#             dataname=args.dataname,
#             output_root=args.data_dir,
#             exclude_cols=exclude_cols,
#             time_col=args.time_col,
#             stratify_col=args.stratify_col,
#             test_size=args.test_size,
#         )
#         # quick smoke test with Preprocessor
#         pp = Preprocessor(args.dataname, data_dir=args.data_dir)
#         X_train = pp.encodeDf(scheme=args.scheme, df=pp.df_train)
#         X_test  = pp.encodeDf(scheme=args.scheme, df=pp.df_test)
#         print("Prepared dataset at:", info["root"])
#         print("Shapes:", {"train": X_train.shape, "test": X_test.shape})
#         print("Numeric cols:", pp.num_idx, "Categorical cols:", pp.cat_idx)
#         raise SystemExit(0)

#     # ---------------- MinIO mode ----------------
#     if args.mode == "minio":
        
#         if not _HAS_DQC:
#             raise RuntimeError("MinIO mode requires dqc_utils with configured _s3(), S3_BUCKET, RAW_PREFIX, CURATED_PREFIX.")

#         bucket = args.bucket or S3_BUCKET
#         raw_prefix = args.raw_prefix or RAW_PREFIX or "DeepSense"
#         curated_prefix = args.curated_prefix or CURATED_PREFIX or "curated"

#         # Build keys
#         raw_key = "/".join([raw_prefix.strip("/"), args.datasetname.strip("/"), f"{args.dataname}.csv"])
#         curated_root = "/".join([curated_prefix.strip("/"), args.datasetname.strip("/"), args.dataname.strip("/")])

#         s3 = _s3()  # boto3 client from your dqc_utils
#         # 1) download raw CSV to temp
#         with tempfile.TemporaryDirectory() as td:
#             local_raw = os.path.join(td, f"{args.dataname}.csv")
#             try:
#                 obj = s3.get_object(Bucket=bucket, Key=raw_key)
#             except Exception as e:
#                 raise FileNotFoundError(f"Cannot read s3://{bucket}/{raw_key}: {e}")
#             body = obj["Body"].read()
#             with open(local_raw, "wb") as f:
#                 f.write(body)

#             # 2) run local prepare_from_csv using your existing pipeline
#             local_out_root = os.path.join(td, "curated")
#             os.makedirs(local_out_root, exist_ok=True)

#             exclude_cols = [c.strip() for c in (args.exclude_cols or "").split(",")] if args.exclude_cols else None
#             info = prepare_from_csv(
#                 csv_path=local_raw,
#                 dataname=args.dataname,
#                 output_root=local_out_root,
#                 exclude_cols=exclude_cols,
#                 time_col=args.time_col,
#                 stratify_col=args.stratify_col,
#                 test_size=args.test_size,
#                 random_state=args.random_state,
#             )

#             # 3) upload curated outputs back to MinIO
#             def _put(path_local: str, key: str, ctype: str = "text/csv"):
#                 with open(path_local, "rb") as f:
#                     s3.put_object(Bucket=bucket, Key=key, Body=f.read(), ContentType=ctype)

#             # CSVs
#             _put(os.path.join(info["root"], "data.csv"),        f"{curated_root}/data.csv")
#             _put(os.path.join(info["root"], "train.csv"),       f"{curated_root}/train.csv")
#             _put(os.path.join(info["root"], "test.csv"),        f"{curated_root}/test.csv")
#             _put(os.path.join(info["root"], "incomplete.csv"),  f"{curated_root}/incomplete.csv")

#             # Info JSON — use the schema file produced by Preprocessor or infer if absent
#             info_dir = os.path.join(info["root"], "Info")
#             os.makedirs(info_dir, exist_ok=True)
#             info_json_local = os.path.join(info_dir, f"{args.dataname}.json")
#             if not os.path.exists(info_json_local):
#                 # ensure schema exists by instantiating Preprocessor once, which writes Info JSON if missing
#                 pp = Preprocessor(args.dataname, data_dir=os.path.dirname(info["root"]))
#                 # It should have created the Info file under your dataset folder;
#                 # for our temp output root, create and save a minimal JSON
#                 with open(info_json_local, "w", encoding="utf-8") as f:
#                     json.dump(pp.info, f, indent=2, ensure_ascii=False)
#             _put(info_json_local, f"{curated_root}/Info/{args.dataname}.json", ctype="application/json")

#             print("Uploaded curated outputs:")
#             print("  ", f"s3://{bucket}/{curated_root}/data.csv")
#             print("  ", f"s3://{bucket}/{curated_root}/train.csv")
#             print("  ", f"s3://{bucket}/{curated_root}/test.csv")
#             print("  ", f"s3://{bucket}/{curated_root}/incomplete.csv")
#             print("  ", f"s3://{bucket}/{curated_root}/Info/{args.dataname}.json")


# -*- coding: utf-8 -*-
"""
Preprocessor (robust, self-contained)

- Works with local folders and S3/MinIO prefixes (requires fsspec+s3fs).
- Loads data.csv / train.csv / test.csv and an optional Info/<name>.json schema.
- Builds stable numeric/categorical indices; auto-excludes path-like columns.
- Provides reliable encode/decode (Ordinal or OHE), mask expansion, and helpers
  to map decoded arrays back into original DataFrames safely.

Author: refactored for robustness
"""

from __future__ import annotations

import io
import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional: fsspec/s3fs for MinIO or S3 paths
try:
    import fsspec  # type: ignore
except Exception:  # pragma: no cover
    fsspec = None  # We'll fall back to vanilla pandas for local files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from helpers.dqc_utils import (
    # IO
    load_df_from_minio, save_df_to_minio, _s3,
    PRIMARY_KEY_RAW,TIMESTAMP_COL,
    # DATASET_NAME,
    # Detection / normalization / QC
    detect_timestamp_column, build_schema_profile,
    normalize_ts_for_gap, compute_time_gaps_smart,
    # Config exported by helpers (single source of truth)
    PROJECT, TARGET, S3_BUCKET,
    REPORT_PREFIX, CURATED_PREFIX,
    DEFAULT_TZ, TS_STD_COL,
    TS_EXPECTED_FREQ, TS_GAP_TOL_MULT, TS_GROUP_KEYS
)

# -------- Defaults --------
DATA_DIR = "./helpers/datasets"
DATASET_NAME = "Scenario33"
INFO_DIR = "Info"  # schema JSON lives at <data_dir>/<INFO_DIR>/<dataname>.json


# --------- IO helpers (local or remote via fsspec) ---------
def _open(path: str, mode: str = "rt"):
    """Open path with fsspec if available and the path looks remote (s3://...)."""
    if path.startswith("s3://") and fsspec is not None:
        fs, _, paths = fsspec.core.get_fs_token_paths(path)
        return fs.open(paths[0], mode=mode)
    # local fallback
    return open(path, mode)


def _exists(path: str) -> bool:
    if path.startswith("s3://") and fsspec is not None:
        fs, _, paths = fsspec.core.get_fs_token_paths(path)
        return fs.exists(paths[0])
    return os.path.exists(path)


def _makedirs_for(path: str) -> None:
    """Create parent dirs for local paths. For remote (s3), noop (put operations create keys)."""
    if path.startswith("s3://"):
        return
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# --- S3 URL helpers -------------------------------------------------
def _split_s3_url(url: str) -> Dict[str, str]:
    # url like: s3://<bucket>/<key...>
    parts = url.split("/", 3)
    if len(parts) < 4:
        raise ValueError(f"Bad S3 URL: {url}")
    return {"bucket": parts[2], "key": parts[3]}

def _s3_key_from_url(url: str) -> str:
    return _split_s3_url(url)["key"]

def _s3_bucket_from_url(url: str) -> str:
    return _split_s3_url(url)["bucket"]

# --- IO shims that work with local paths and s3:// -------------------
def _read_csv(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        # PASS *KEY* ONLY to load_df_from_minio
        bucket = _s3_bucket_from_url(path)
        key    = _s3_key_from_url(path)
        if bucket != S3_BUCKET:
            # read directly via boto if bucket differs from configured one
            body = _s3().get_object(Bucket=bucket, Key=key)["Body"].read()
            return pd.read_csv(io.BytesIO(body))
        # same-bucket fast path
        df, _ = load_df_from_minio(key)
        return df
    return pd.read_csv(path)

def _write_csv(df: pd.DataFrame, path: str) -> None:
    if path.startswith("s3://"):
        bucket = _s3_bucket_from_url(path)
        key    = _s3_key_from_url(path)
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        _s3().put_object(Bucket=bucket, Key=key, Body=buf.getvalue(), ContentType="text/csv")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def _write_json(path: str, obj: Dict[str, Any]) -> None:
    if path.startswith("s3://"):
        bucket = _s3_bucket_from_url(path)
        key    = _s3_key_from_url(path)
        body = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        _s3().put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

from botocore.exceptions import ClientError

def _read_json(path: str) -> dict:
    """Read JSON from local or s3:// and return a dict.
       If the S3 object is missing, raise FileNotFoundError."""
    if path.startswith("s3://"):
        bucket = _s3_bucket_from_url(path)
        key    = _s3_key_from_url(path)
        try:
            body = _s3().get_object(Bucket=bucket, Key=key)["Body"].read()
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NotFound"):
                raise FileNotFoundError(path)
            raise
        return json.loads(body.decode("utf-8"))

    with open(path, "rb") as f:
        return json.load(f)

# -------------------- split_and_store_dataset --------------------
def split_and_store_dataset(
    df: pd.DataFrame,
    prefix: str,
    test_size: float = 0.2,
    random_state: int = 42,
    time_col: Optional[str] = None,
    stratify_col: Optional[str] = None,
    exclude_cols: Optional[List[str]] = None,
    auto_exclude_path_like: bool = True,
) -> None:
    """
    Split dataset into train/test/incomplete and store to MinIO with metadata.
    """
    # 0) Drop fully-null columns up-front
    df_clean, dropped_all_null = _drop_all_null_columns(df)

    # 1) Build exclude set
    exclude = set(exclude_cols or [])
    exclude = {c for c in exclude if c in df_clean.columns}

    # Auto-exclude path-like columns
    if auto_exclude_path_like:
        for c in df_clean.columns:
            try:
                if _is_path_like_series(df_clean[c]):
                    exclude.add(c)
            except Exception:
                pass

    use_time = bool(time_col) and (time_col in df_clean.columns)
    cols_to_check = [c for c in df_clean.columns if c not in exclude]

    if cols_to_check:
        complete_mask = df_clean[cols_to_check].notna().all(axis=1)
    else:
        complete_mask = pd.Series(True, index=df_clean.index)

    df_complete   = df_clean.loc[complete_mask].copy()
    df_incomplete = df_clean.loc[~complete_mask].copy()

    # If time_col is used, ensure datetime dtype
    if use_time and not np.issubdtype(df_complete[time_col].dtype, np.datetime64):
        df_complete[time_col] = pd.to_datetime(df_complete[time_col], errors="coerce")

    # 3) split
    if use_time:
        df_nonat = df_complete.dropna(subset=[time_col]).sort_values(time_col, kind="stable")
        df_nat   = df_complete[df_complete[time_col].isna()]
        n = len(df_nonat)
        if n == 0 or len(df_complete) <= 1:
            train_df, test_df = train_test_split(df_complete, test_size=test_size, random_state=random_state)
        else:
            split_idx = int((1.0 - test_size) * n)
            train_df = df_nonat.iloc[:max(0, split_idx)].copy()
            test_df  = df_nonat.iloc[max(0, split_idx):].copy()
            if not df_nat.empty:
                train_df = pd.concat([train_df, df_nat], axis=0, ignore_index=True)
    else:
        strat = None
        use_strat = bool(stratify_col) and (stratify_col in df_complete.columns)
        if use_strat:
            vc = df_complete[stratify_col].value_counts(dropna=False)
            if len(vc) >= 2 and (vc.min() >= 2) and (len(df_complete) >= 4):
                strat = df_complete[stratify_col]
        if len(df_complete) <= 1:
            train_df = df_complete.copy()
            test_df  = df_complete.iloc[0:0].copy()
        else:
            train_df, test_df = train_test_split(
                df_complete, test_size=test_size, random_state=random_state, stratify=strat
            )

    # Build keys
    data_key       = f"{prefix}data.csv"
    train_key      = f"{prefix}train.csv"
    test_key       = f"{prefix}test.csv"
    incomplete_key = f"{prefix}incomplete.csv"
    meta_key       = f"{prefix}split_meta.json"

    # Write
    save_df_to_minio(df_clean,     data_key,       fmt="csv", index=False)
    save_df_to_minio(train_df,     train_key,      fmt="csv", index=False)
    save_df_to_minio(test_df,      test_key,       fmt="csv", index=False)
    save_df_to_minio(df_incomplete, incomplete_key, fmt="csv", index=False)

    meta = {
        "rows_total": len(df),
        "rows_complete": len(df_complete),
        "rows_incomplete": len(df_incomplete),
        "dropped_all_null_cols": sorted(dropped_all_null),
        "test_size": float(test_size),
        "exclude_cols": sorted(list(exclude)),
        "split_mode": "chronological" if use_time else (
            "stratified" if (stratify_col and stratify_col in df_clean.columns) else "random"
        ),
        "time_col": time_col if use_time else None,
        "stratify_col": stratify_col if not use_time else None,
        "outputs": {
            "data_csv": data_key,
            "train_csv": train_key,
            "test_csv": test_key,
            "incomplete_csv": incomplete_key,
            "meta_json": meta_key,
        }
    }

    _s3().put_object(
        Bucket=S3_BUCKET,
        Key=meta_key,
        Body=json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    return train_key, test_key, incomplete_key, df_incomplete

# -------------------- Metrics helpers --------------------

def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Utility: flip mask booleans.
    If your generator uses True==missing, do: observed_mask = invert_mask(missing_mask).
    """
    return ~mask


def mean_std(data: np.ndarray, observed_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean/std **over observed entries only**.

    Parameters
    ----------
    data : np.ndarray, shape [N, D]
    observed_mask : np.ndarray, shape [N, D], dtype bool, where True means OBSERVED
    """
    obs = observed_mask.astype(np.float32)
    denom = obs.sum(0)
    denom[denom == 0] = 1
    mean = (data * obs).sum(0) / denom
    var = ((data - mean) ** 2 * obs).sum(0) / denom
    std = np.sqrt(var)
    return mean, std


# def get_eval(
#     X_pred: np.ndarray,
#     X_true: np.ndarray,
#     observed_mask: np.ndarray,
#     num_numeric: int,
#     *,
#     cat_edge_case: bool = False,
# ) -> Tuple[float, float]:
#     """
#     Evaluate MSE on numeric cols, accuracy on categorical cols **where observed_mask == True**.
#     Assumes X_pred/X_true are in the SAME encoded space (both Ordinal or both OHE) and the mask was
#     expanded to encoded width when needed (see `Preprocessor.extend_mask`).

#     Returns
#     -------
#     mse : float
#         Mean squared error over observed numeric entries.
#     acc : float
#         Categorical accuracy (%) over observed categorical entries.
#     """
#     if X_true.shape != X_pred.shape or X_true.shape != observed_mask.shape:
#         raise ValueError("Shapes of X_true, X_pred, and observed_mask must match.")

#     # numeric mse
#     num_true = X_true[:, :num_numeric]
#     num_pred = X_pred[:, :num_numeric]
#     num_mask = observed_mask[:, :num_numeric]

#     num_vals = (num_true[num_mask] - num_pred[num_mask]) ** 2
#     mse = float(num_vals.mean()) if num_vals.size else 0.0

#     # categorical acc
#     cat_true = X_true[:, num_numeric:]
#     cat_pred = X_pred[:, num_numeric:]
#     if cat_true.size == 0:
#         return mse, 100.0

#     cat_mask = observed_mask[:, num_numeric:]
#     if cat_edge_case:
#         # Trim whitespace (for decoded string comparisons)
#         cat_true_comp = np.char.strip(cat_true[cat_mask].astype(str))
#         acc_arr = (cat_pred[cat_mask] == cat_true_comp)
#     else:
#         acc_arr = (cat_pred[cat_mask] == cat_true[cat_mask])

#     acc = float(np.sum(acc_arr) * 100.0 / max(1, len(acc_arr)))
#     return mse, acc

# import numpy as np

def get_eval(
    X_pred: np.ndarray,
    X_true: np.ndarray,
    mask_missing: np.ndarray,   # True == MISSING (encoded order)
    num_numeric: int,
    *,
    cat_edge_case: bool = True,         # strip whitespace on both sides
    train_numeric_std: np.ndarray = None,  # optional for NRMSE reporting
):
    # ---- split
    num_true = X_true[:, :num_numeric].astype(float, copy=False)
    num_pred = X_pred[:, :num_numeric].astype(float, copy=False)
    cat_true = X_true[:, num_numeric:]
    cat_pred = X_pred[:, num_numeric:]

    num_mask = mask_missing[:, :num_numeric].astype(bool, copy=False)
    cat_mask = mask_missing[:, num_numeric:].astype(bool, copy=False)

    # ---- numeric MSE over missing cells only
    if num_numeric > 0 and num_mask.any():
        diffs = num_true[num_mask] - num_pred[num_mask]
        mse = float(np.mean(diffs ** 2))
    else:
        mse = 0.0

    # ---- categorical accuracy over missing cells only
    if cat_true.size == 0:
        acc = 100.0
    else:
        if cat_mask.any():
            t = cat_true[cat_mask].astype(str)
            p = cat_pred[cat_mask].astype(str)
            if cat_edge_case:
                t = np.char.strip(t)
                p = np.char.strip(p)
            acc = float((t == p).mean() * 100.0)
        else:
            acc = 100.0

    # ---- optional normalized error (helps interpretability)
    nrmse = None
    if train_numeric_std is not None and num_numeric > 0 and num_mask.any():
        # per-cell normalization using column stds
        stds = np.asarray(train_numeric_std, dtype=float)
        stds[stds == 0] = 1.0
        # build a std array aligned to the masked numeric cells
        # gather std per column for masked entries
        col_ids = np.repeat(np.arange(num_numeric)[None, :], num_true.shape[0], axis=0)[num_mask]
        nrmse = float(np.sqrt(np.mean(((num_true[num_mask] - num_pred[num_mask]) / stds[col_ids]) ** 2)))

    return mse, acc, {"NRMSE": nrmse}

# ------------- Simple utilities -------------
# _PATH_LIKE_PAT = re.compile(r"""(^(/|[A-Za-z]:\\))|(^\w+://)""")
# _PATH_LIKE_PAT = [
#     r"^(?:\.{0,2}/).+",  # ./foo  ../foo  /abs/posix
#     r"^[A-Za-z]:\\.+",   # C:\Windows\...
#     r".*/.+\.[A-Za-z0-9]{1,6}$",  # posix with extension
#     r".*\\.+\.[A-Za-z0-9]{1,6}$",  # windows with extension
#     r"^https?://.+",  # URLs
# ]

# use ONE compiled regex (case-insensitive, verbose)
_PATH_LIKE_RE = re.compile(
    r"""(?ix)
    ^(                                  # start
        (?:[a-z]:)?[\\/]                # windows drive or absolute path
      | \.{1,2}[\\/]                    # ./
      | (?:https?|s3|gs|ftp)://         # urls / object stores
    ).+                                 # at least something after
    """
)

def _is_path_like_series(s: pd.Series, thresh: float = 0.25) -> bool:
    """Heuristic: True if >= thresh fraction of non-null values look like paths/URLs."""
    if s is None or len(s) == 0:
        return False
    sub = s.astype("string", errors="ignore")
    # ensure string dtype to avoid .str errors
    sub = sub.astype(str)
    hits = sub.str.match(_PATH_LIKE_RE, na=False)
    return bool((hits.sum() / max(len(sub), 1)) >= thresh)


def _drop_all_null_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Drop columns that are entirely NaN; return (df_clean, dropped_col_names)."""
    mask = df.isna().all(axis=0)
    dropped = df.columns[mask].tolist()
    return df.loc[:, ~mask].copy(), dropped


def _infer_schema_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    """Basic heuristic schema inference: numeric vs categorical, exclude path-like."""
    df0, dropped_all_null = _drop_all_null_columns(df)

    num_cols, cat_cols = [], []
    for c in df0.columns:
        if _is_path_like_series(df0[c]):
            continue  # exclude from modeling
        if pd.api.types.is_numeric_dtype(df0[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    info = {
        "num_col_names": num_cols,
        "cat_col_names": cat_cols,
        "target_col_names": [],
        "excluded_col_names": [],
        "dropped_all_null_cols": dropped_all_null,
        # indices will be filled by Preprocessor after sanitization
        "num_col_idx": [],
        "cat_col_idx": [],
        "target_col_idx": [],
        "excluded_col_idx": [],
    }
    return info

import pickle

def load_imputer_from_s3(bucket: str, key: str):
    s3_client = _s3()  # your wrapper around boto3.client("s3")
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    buf = obj["Body"].read()
    imputer = pickle.loads(buf)
    return imputer

# ------------------------ Preprocessor ------------------------
class Preprocessor:
    """
    Robust mixed-type preprocessor for tabular data.

    - Fits encoders on train split only (to avoid leakage).
    - Supports 'OHE' and 'Ordinal' (default).
    - Inverse decode handles off-grid predictions.
    - Excludes path-like columns (files/URLs) from categorical set automatically.
    - Works with local folders or s3://<bucket>/... prefixes.
    """

    def __init__(self, dataname: str = DATASET_NAME, data_dir: str = DATA_DIR):
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

        self.dataset = dataname
        print(f'initializing Preprocessor for dataset: {dataname}')
        self.data_dir = data_dir.rstrip("/")

        # ---- paths ----
        root = f"{self.data_dir}"
        info_path = f"{root}/{INFO_DIR}/{dataname}.json"
        raw_path  = f"{root}/data.csv"
        train_path= f"{root}/train.csv"
        test_path = f"{root}/test.csv"

        # ---- load splits ----
        raw_df    = _read_csv(raw_path)
        raw_train = _read_csv(train_path)
        raw_test  = _read_csv(test_path)

        # Harmonize to data.csv (drop all-null cols once, then align)
        df, _dropped_all_null = _drop_all_null_columns(raw_df)
        self.df = df
        self.df_train = raw_train.reindex(columns=self.df.columns).copy()
        self.df_test  = raw_test.reindex(columns=self.df.columns).copy()

        # ---- schema: read or infer on first run ----
        try:
            self.info = _read_json(info_path)  # supports s3:// and local
        except FileNotFoundError:
            self.info = _infer_schema_from_df(self.df)
            _write_json(info_path, self.info)

        # ---- constant columns (record only; keep DF intact) ----
        dropped_all_train_constant = [c for c in self.df_train.columns if self.df_train[c].nunique(dropna=False) == 1]
        dropped_all_test_constant  = [c for c in self.df_test.columns  if self.df_test[c].nunique(dropna=False)  == 1]
        self.dropped_all_train_constant: List[str] = dropped_all_train_constant
        self.dropped_all_test_constant:  List[str] = dropped_all_test_constant

        # ---- sanitize schema indices/names against current df ----
        cols  = list(self.df.columns)
        ncols = len(cols)
        name2idx = {c: i for i, c in enumerate(cols)}

        def _clamp_int_idxs(idxs):
            out = []
            for x in (idxs or []):
                try:
                    i = int(x)
                    if 0 <= i < ncols:
                        out.append(i)
                except Exception:
                    pass
            return out

        def _names_to_idx(names):
            return [name2idx[c] for c in (names or []) if c in name2idx]

        num_idx = _names_to_idx(self.info.get("num_col_names")) or _clamp_int_idxs(self.info.get("num_col_idx"))
        cat_idx = _names_to_idx(self.info.get("cat_col_names")) or _clamp_int_idxs(self.info.get("cat_col_idx"))
        tgt_idx = _names_to_idx(self.info.get("target_col_names")) or _clamp_int_idxs(self.info.get("target_col_idx"))
        exc_idx = _names_to_idx(self.info.get("excluded_col_names")) or _clamp_int_idxs(self.info.get("excluded_col_idx"))

        self.num_idx: List[int] = num_idx
        self.cat_idx: List[int] = cat_idx
        self.tgt_idx: List[int] = tgt_idx
        self.excluded_idx: List[int] = exc_idx

        # ---- exclude path-like columns from cat_idx ----
        keep_cat_idx: List[int] = []
        self.path_like_idx: List[int] = []
        for i in self.cat_idx:
            if 0 <= i < ncols and _is_path_like_series(self.df[cols[i]]):
                self.path_like_idx.append(i)
            else:
                keep_cat_idx.append(i)
        if self.path_like_idx:
            self.cat_idx = keep_cat_idx

        # ---- exclude constant columns from indices (do not drop from DF) ----
        const_names = set(self.dropped_all_train_constant) | set(self.dropped_all_test_constant)
        if const_names:
            const_idx = {i for i, n in enumerate(self.df.columns) if n in const_names}
            self.num_idx = [i for i in self.num_idx if i not in const_idx]
            self.cat_idx = [i for i in self.cat_idx if i not in const_idx]
            self.info.setdefault("excluded_constant_cols", sorted(list(const_names)))
            self.excluded_idx.extend(list(const_idx))

        # Also exclude path-like mistakenly in numeric
        bad_num_idx = [i for i in self.num_idx if 0 <= i < ncols and _is_path_like_series(self.df[cols[i]])]
        if bad_num_idx:
            self.num_idx = [i for i in self.num_idx if i not in bad_num_idx]
            self.info.setdefault("excluded_path_like_cols", [])
            self.info["excluded_path_like_cols"] = sorted(
                set(self.info["excluded_path_like_cols"] + [cols[i] for i in bad_num_idx])
            )
            self.excluded_idx.extend(bad_num_idx)

        # ---- persist sanitized schema (both idx & names) ----
        try:
            excluded_idx_sorted = sorted(set(self.excluded_idx))
            self.info["excluded_col_idx"]   = excluded_idx_sorted
            self.info["excluded_col_names"] = [self.df.columns[i] for i in excluded_idx_sorted]
            self.info["num_col_idx"]        = self.num_idx
            self.info["cat_col_idx"]        = self.cat_idx
            self.info["target_col_idx"]     = self.tgt_idx
            self.info["num_col_names"]      = [self.df.columns[i] for i in self.num_idx]
            self.info["cat_col_names"]      = [self.df.columns[i] for i in self.cat_idx]
            self.info["target_col_names"]   = [self.df.columns[i] for i in self.tgt_idx]
            _write_json(info_path, self.info)
        except Exception:
            pass

        # ---- sanity checks ----
        for idx in self.num_idx + self.cat_idx + self.tgt_idx:
            if idx < 0 or idx >= len(cols):
                raise ValueError(f"Column index {idx} out of range for dataset '{dataname}'.")

        # ---- encoders (fit on train only) ----
        self._ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
        self._ord = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)

        if self.cat_idx:
            train_cats = self.df_train.iloc[:, self.cat_idx].astype(str).applymap(lambda x: x.strip())
            self._ohe.fit(train_cats)
            self._ord.fit(train_cats)

        self.numerical_indices_np_end: Optional[int] = None
        self._cached_cat_feature_counts: Optional[np.ndarray] = None


    # ---------------- Encoding/Decoding on DataFrames ----------------
    def encodeDf(self, scheme: str = "Ordinal", df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Encode a DataFrame to numpy features [num | cats_enc].
        Sets self.numerical_indices_np_end to split index for later decoding/mask ops.
        """
        if df is None:
            df = self.df

        # Ensure identical column set/order as data.csv
        if list(df.columns) != list(self.df.columns):
            df = df.reindex(columns=self.df.columns)

        # numeric block
        if self.num_idx:
            nums_df = df.iloc[:, self.num_idx].copy().apply(lambda c: pd.to_numeric(c, errors="coerce"))
        else:
            nums_df = pd.DataFrame(index=df.index)

        cats_df = df.iloc[:, self.cat_idx].astype(str).applymap(lambda x: x.strip()) if self.cat_idx else None
        self.numerical_indices_np_end = len(self.num_idx)  # explicit & robust

        scheme_norm = scheme.strip().lower()
        if scheme_norm == "ohe":
            if cats_df is None:
                return nums_df.to_numpy(dtype=np.float32)
            cats_enc = self._ohe.transform(cats_df)
            self._cached_cat_feature_counts = np.array([len(c) for c in self._ohe.categories_], dtype=int)
            return np.concatenate([nums_df.to_numpy(dtype=np.float32), cats_enc], axis=1)

        if scheme_norm == "ordinal":
            if cats_df is None:
                return nums_df.to_numpy(dtype=np.float32)
            cats_enc = self._ord.transform(cats_df).astype(np.float32)
            return np.concatenate([nums_df.to_numpy(dtype=np.float32), cats_enc], axis=1)

        raise ValueError("Invalid scheme; use 'OHE' or 'Ordinal'.")

    def encodeNp(self, scheme: str = "Ordinal", arr: Optional[np.ndarray] = None) -> np.ndarray:
        """Re-encode an already concatenated [num | cats] numpy array with a different scheme.
        Assumes self.numerical_indices_np_end was set by a prior encode call.
        """
        if arr is None:
            raise ValueError("arr is required")
        if self.numerical_indices_np_end is None:
            raise RuntimeError("numerical_indices_np_end not set. Call encodeDf() first.")

        nums = arr[:, : self.numerical_indices_np_end]
        cats = arr[:, self.numerical_indices_np_end :]
        scheme_norm = scheme.strip().lower()

        if scheme_norm == "ohe":
            cats_rounded = np.rint(cats).astype(int)
            cats_rounded = np.clip(cats_rounded, -1, None)
            cats_strings = self._inverse_from_ordinal_indices(cats_rounded)
            cats_strings = pd.DataFrame(cats_strings).astype(str).values
            cats_ohe = self._ohe.transform(cats_strings)
            return np.concatenate([nums, cats_ohe], axis=1)

        if scheme_norm == "ordinal":
            cats_strings = self._ohe.inverse_transform(cats)
            cats_ord = self._ord.transform(cats_strings)
            return np.concatenate([nums, cats_ord], axis=1)

        raise ValueError("Invalid scheme; use 'OHE' or 'Ordinal'.")

    def decodeNp(self, scheme: str = "Ordinal", arr: Optional[np.ndarray] = None) -> np.ndarray:
        """Inverse-transform back to [num | cats_as_strings] (np.array; cats object dtype).
        For 'Ordinal', rounds/clips off-grid predictions to nearest valid category index (keeps -1 as <UNK>).
        Returns ONLY the encoded columns (num + cat) in the encoded order.
        """
        if arr is None:
            raise ValueError("arr is required")
        if self.numerical_indices_np_end is None:
            raise RuntimeError("numerical_indices_np_end not set. Call encodeDf() first.")
        if arr.shape[0] == 0:
            return arr.copy()

        expected = len(self.num_idx) + len(self.cat_idx)
        if arr.shape[1] != expected:
            raise ValueError(f"decode width mismatch: got {arr.shape[1]}, expected {expected} (num+cat).")

        nums = arr[:, : self.numerical_indices_np_end]
        if expected == self.numerical_indices_np_end or not self.cat_idx:
            return nums.copy()

        scheme_norm = scheme.strip().lower()

        if scheme_norm == "ohe":
            cats = self._ohe.inverse_transform(arr[:, self.numerical_indices_np_end :])
            return np.concatenate([nums, cats], axis=1)

        if scheme_norm == "ordinal":
            raw = arr[:, self.numerical_indices_np_end :]
            cats_idx = np.rint(raw).astype(int)
            # clip each column separately; keep -1 as unknown
            for j, cats_j in enumerate(self._ord.categories_):
                vmax = len(cats_j) - 1
                col = cats_idx[:, j]
                mask_ok = col >= 0
                col[mask_ok] = np.clip(col[mask_ok], 0, vmax)
                cats_idx[:, j] = col
            cats = self._inverse_from_ordinal_indices(cats_idx)
            return np.concatenate([nums, cats], axis=1)

        raise ValueError("Invalid scheme; use 'OHE' or 'Ordinal'.")

    def _inverse_from_ordinal_indices(self, idx_arr: np.ndarray) -> np.ndarray:
        """Convert integer indices (per-cat col) to original strings; -1 → '<UNK>'."""
        out = np.empty_like(idx_arr, dtype=object)
        for j, cats in enumerate(self._ord.categories_):
            col = idx_arr[:, j]
            out_col = np.empty(col.shape[0], dtype=object)
            unk = col < 0
            out_col[unk] = "<UNK>"
            ok = ~unk
            if ok.any():
                ok_idx = np.clip(col[ok], 0, len(cats) - 1)
                out_col[ok] = np.asarray(cats, dtype=object)[ok_idx]
            out[:, j] = out_col
        return out

    def extend_mask(self, ori_mask: np.ndarray, encoding: str = "Ordinal", input_mask_is_observed: bool = True) -> np.ndarray:
        """
        Expand a boolean mask shaped [N, D_original] to match encoded feature width.
        - For Ordinal: shape unchanged (1 mask per categorical col).
        - For OHE: expand each categorical col to its number of OHE bits.

        Parameters
        ----------
        ori_mask : np.ndarray, shape (N, D_original)
            Mask aligned to original DataFrame columns.
        encoding : str
            'Ordinal' or 'OHE' target encoding width to expand to.
        input_mask_is_observed : bool, default True
            If True, input uses convention True==OBSERVED (project default).
            If False, input uses True==MISSING and will be inverted before expansion.

        Returns
        -------
        expanded_mask : np.ndarray of shape (N, D_encoded)
        """
        if ori_mask is None or ori_mask.ndim != 2:
            raise ValueError("ori_mask must be a 2D boolean numpy array with shape (N, D_original).")
        expected_D = len(self.df.columns)
        if ori_mask.shape[1] != expected_D:
            raise ValueError(f"ori_mask has wrong width: got {ori_mask.shape[1]}, expected {expected_D} (number of original columns).")

        # Normalize semantics: ensure 'mask' is True==OBSERVED for internal logic
        mask = ori_mask.copy()
        if not input_mask_is_observed:
            mask = ~mask

        enc = encoding.strip().lower()
        if enc == "ordinal" or not self.cat_idx:
            return mask.copy()

        if self.numerical_indices_np_end is None:
            raise RuntimeError("numerical_indices_np_end not set. Call encodeDf() first.")
        if self._cached_cat_feature_counts is None:
            self._cached_cat_feature_counts = np.array([len(c) for c in self._ohe.categories_], dtype=int)

        num_mask = mask[:, self.num_idx] if self.num_idx else np.zeros((mask.shape[0], 0), dtype=bool)
        cat_mask = mask[:, self.cat_idx] if self.cat_idx else np.zeros((mask.shape[0], 0), dtype=bool)

        pieces = []
        for j, width in enumerate(self._cached_cat_feature_counts):
            mj = cat_mask[:, j][:, None]  # (N,1)
            pieces.append(np.repeat(mj, width, axis=1))
        cat_mask_expanded = (
            np.concatenate(pieces, axis=1) if pieces else np.zeros((ori_mask.shape[0], 0), dtype=bool)
        )
        return np.concatenate([num_mask, cat_mask_expanded], axis=1)

    # -------- Convenience helpers for DAGs --------
    def encoded_indices(self) -> List[int]:
        return list(self.num_idx) + list(self.cat_idx)

    def encoded_columns(self) -> List[str]:
        cols = list(self.df.columns)
        return [cols[i] for i in self.encoded_indices()]

    def decoded_df(self, encoded_arr: np.ndarray, scheme: str = "Ordinal") -> pd.DataFrame:
        """Return a DataFrame with ONLY the encoded columns (right names, right order)."""
        dec = self.decodeNp(scheme, encoded_arr)
        return pd.DataFrame(dec, columns=self.encoded_columns())
