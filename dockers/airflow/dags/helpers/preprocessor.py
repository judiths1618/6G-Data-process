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
# from helpers.dqc_utils import (
#     # IO
#     # load_df_from_minio, save_df_to_minio, _s3,
#     PRIMARY_KEY_RAW,TIMESTAMP_COL,
#     # DATASET_NAME,
#     # Detection / normalization / QC
#     detect_timestamp_column, build_schema_profile,
#     normalize_ts_for_gap, compute_time_gaps_smart,
#     # Config exported by helpers (single source of truth)
#     PROJECT, TARGET, S3_BUCKET,
#     REPORT_PREFIX, CURATED_PREFIX,
#     DEFAULT_TZ, TS_STD_COL,
#     TS_EXPECTED_FREQ, TS_GAP_TOL_MULT, TS_GROUP_KEYS
# )

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
