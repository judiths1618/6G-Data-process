# -*- coding: utf-8 -*-
from __future__ import annotations

import io, os, json, re
from functools import lru_cache
from typing import Any, Optional, Tuple, Dict, Iterable, List, Set

import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype, is_numeric_dtype, is_string_dtype,
    is_bool_dtype, is_integer_dtype, is_float_dtype
)
from airflow.models import Variable
from botocore.exceptions import ClientError


# ============================ Airflow Variables (single source of truth) ============================
PROJECT        = Variable.get("N2N_PROJECT",        default_var="demo")
DATASET_NAME   = Variable.get("N2N_DATASET",        default_var="default")
TARGET         = Variable.get("N2N_TARGET",         default_var="label")

S3_ENDPOINT    = Variable.get("N2N_S3_ENDPOINT")            # e.g. http://minio:9000
S3_ACCESS_KEY  = Variable.get("N2N_S3_ACCESS_KEY")
S3_SECRET_KEY  = Variable.get("N2N_S3_SECRET_KEY")
S3_BUCKET      = Variable.get("N2N_S3_BUCKET")
INPUT_KEY      = Variable.get("N2N_INPUT_KEY")              # e.g., raw/adult.csv

CURATED_PREFIX = Variable.get("N2N_CURATED_PREFIX", default_var="curated")
REPORT_PREFIX  = Variable.get("N2N_REPORT_PREFIX",  default_var="dq_reports")
SAVED_MODELS_PREFIX = Variable.get("N2N_SAVED_MODEL_PREFIX", default_var="saved_models")
# dqc_utils.py（或你的共享配置处）


# Time normalization / detection config
DEFAULT_TZ     = Variable.get("N2N_DEFAULT_TZ",     default_var="UTC")       # e.g. "Europe/Amsterdam"
TS_STD_COL     = Variable.get("N2N_TS_STD_COL",     default_var="ts_utc")    # normalized UTC column

# Cross-dataset time unification
TS_DATASET_TZ_MAP = Variable.get("N2N_TS_DATASET_TZ_MAP", default_var="{}")  # JSON {"datasetA":"Europe/Amsterdam",...}
TS_SNAP_UNIT      = Variable.get("N2N_TS_SNAP_UNIT",      default_var="")    # "", "S", "L", "T", "H"
TS_ANCHOR_DATE    = Variable.get("N2N_TS_ANCHOR_DATE",    default_var="")    # optional YYYY-MM-DD

# TS QC config (kept in helpers so DAGs can import from here consistently)
TIMESTAMP_COL = Variable.get("N2N_TIMESTAMP_COL", default_var="timestamp")
TS_EXPECTED_FREQ  = Variable.get("N2N_TS_EXPECTED_FREQ",  default_var="10S")    # e.g. "1S"; empty = auto
TS_GAP_TOL_MULT   = float(Variable.get("N2N_TS_GAP_TOL_MULT", default_var="1.8"))
TS_GROUP_KEYS_RAW = Variable.get("N2N_TS_GROUP_KEYS",     default_var="")    # e.g. "site_id,device_id"
TS_GROUP_KEYS     = [k.strip() for k in TS_GROUP_KEYS_RAW.split(",") if k.strip()]

# Optional business-hours mask (adaptive gap)
TS_ACTIVE_CRON    = Variable.get("N2N_TS_ACTIVE_CRON",    default_var="")    # e.g. "mon-fri 08:00-18:00"
TS_ACTIVE_TZ      = Variable.get("N2N_TS_ACTIVE_TZ",      default_var=DEFAULT_TZ)


# ---- Primary Key detection config ----
PRIMARY_KEY_RAW     = Variable.get("N2N_PRIMARY_KEY", default_var="")     # e.g. "id" or "site_id,device_id"
PK_ALLOW_TIME       = Variable.get("N2N_PK_ALLOW_TIME", default_var="true").lower() == "true"
PK_MAX_WIDTH        = int(Variable.get("N2N_PK_MAX_WIDTH", default_var="3"))  # try up to 3 columns
PK_NULL_RATE_MAX    = float(Variable.get("N2N_PK_NULL_RATE_MAX", default_var="0.20"))  # prefer cols with <=20% nulls
PK_TOPK_UNI_COLS    = int(Variable.get("N2N_PK_TOPK_UNI_COLS", default_var="8"))       # breadth for combos

# ID-like column name patterns (case-insensitive)
PK_NAME_PATTERNS = [
    r"^index", r"_index$", r"id$",
    r"^id$", r".*_id$", r"^gnb_id$", r"^cell_id$", r"^ue_id$",
    r"^imsi$", r"^imei$", r"^device_id$", r"^session_id$", r"^trace_id$",
    r"^beam_id$", r"^ssb_index$", r"^node_id$", r"^enb_id$", r"^nr_cell_id$",
]

# ============================ Key Builder ============================
# REPORT_PREFIX = "dq_reports"  # or load from your config

def build_report_key(
    input_key: str,
    run_ts: str,
    suffix: str = "dqc.json",
    report_prefix: Optional[str] = None,
) -> str:
    """
    Convert an input object key into a normalized report key:

      input:  DeepSense/Scenario33/scenario33.csv
      output: dq_reports/DeepSense/scenario33/<run_ts>_dqc.json

    Rules:
    - Strip query string and trailing slash.
    - Keep the *first* path segment (dataset group) and the file stem (basename without extension).
    - Drop any middle folders.
    """
    prefix = (report_prefix or REPORT_PREFIX).strip("/")

    # 1) strip query & trailing slash
    clean_key = input_key.strip().split("?", 1)[0].rstrip("/")

    # 2) split and extract
    parts = [p for p in clean_key.split("/") if p]
    KNOWN_PREFIXES = {"raw", "curated"}
    if parts and parts[0] in KNOWN_PREFIXES:
        parts = parts[1:]
    if not parts:
        raise ValueError(f"Invalid input_key: {input_key!r}")

    fname = parts[-1]
    stem = fname.rsplit(".", 1)[0] if "." in fname else fname
    top = parts[0]  # keep only the first segment

    # 3) build: <REPORT_PREFIX>/<top>/<stem>/<run_ts>_<suffix>
    return f"{prefix}/{top}/{stem}/{run_ts}_{suffix}"

def build_curated_key(input_key: str, leaf: str, curated_prefix: str | None = None) -> str:
    """
    将原始对象 Key 映射到 curated 下的标准目录，并去掉原始文件扩展名：
      input_key: DeepSense/Scenario33/scenario33.csv
      leaf:      'data.csv'  (或 'train.csv' / 'test.csv' / 'Info/Scenario33.json' 等)
      => curated/DeepSense/Scenario33/scenario33/data.csv

    参数
    ----
    input_key : str
        原始对象的 Key（可以是 raw 下的路径，函数只做路径重写，不访问存储）
    leaf : str
        需要落在该目录下的文件名或子路径（会自动去掉前导斜杠）
    curated_prefix : str | None
        覆盖默认的 curated 前缀；不传则从配置读取（默认 'curated'）
    """
    # 去掉 query/尾部斜杠
    clean_key = input_key.strip().split("?")[0].rstrip("/")

    # 去掉最后一段文件名的扩展名
    parts = clean_key.split("/")
    KNOWN_PREFIXES = {"raw", "curated"}
    if parts and parts[0] in KNOWN_PREFIXES:
        parts = parts[1:]
    fname = parts[-1]
    stem = fname.rsplit(".", 1)[0] if "." in fname else fname
    rebuilt = "/".join(parts[:-1] + [stem])

    # 读取配置中的 curated 前缀（若提供了参数则用参数）
    try:
        from dqc_utils import get_config_value  # 如果本函数就放在 dqc_utils 内部，直接调用本地函数即可
        _cur_prefix = curated_prefix or get_config_value("prefix", "curated", default="curated")
    except Exception:
        _cur_prefix = curated_prefix or "curated"

    leaf = leaf.lstrip("/")  # 规整 leaf
    return f"{_cur_prefix}/{rebuilt}/{leaf}"


def build_saved_models_key(input_key: str, run_ts: str, suffix: str = "model.pkl") -> str:
    """
    将输入对象 Key 原样嵌入到 saved_models/ 下，并去掉文件扩展名：
      input_key: DeepSense/Scenario33/scenario33.csv
      => saved_models/DeepSense/Scenario33/scenario33/<run_ts>_model.pkl
    """
    # 去掉 query/尾部斜杠
    clean_key = input_key.strip().split("?")[0].rstrip("/")

    # 去掉最后一段文件名的扩展名
    parts = clean_key.split("/")
    fname = parts[-1]
    stem = fname.rsplit(".", 1)[0] if "." in fname else fname
    rebuilt = "/".join(parts[:-1] + [stem])

    return f"{SAVED_MODELS_PREFIX}/{rebuilt}/{run_ts}_{suffix}"

# ============================ S3 / MinIO Helpers ============================
@lru_cache(maxsize=1)
def _s3():
    import boto3
    if not (S3_ENDPOINT and S3_BUCKET):
        raise RuntimeError("N2N_S3_ENDPOINT and N2N_S3_BUCKET must be set.")
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )

def key_exists(bucket: str, key: str) -> bool:
    try:
        _s3().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
            return False
        raise

def list_under_prefix(bucket: str, prefix: str) -> Iterable[Dict[str, Any]]:
    s3 = _s3()
    token: Optional[str] = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            yield {"Key": obj["Key"], "LastModified": obj["LastModified"], "Size": obj["Size"]}
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

def _read_bytes(bucket: str, key: str) -> bytes:
    return _s3().get_object(Bucket=bucket, Key=key)["Body"].read()

def load_df_from_minio(key: str) -> Tuple[pd.DataFrame, str]:
    ext = os.path.splitext(key)[1].lower()
    body = _read_bytes(S3_BUCKET, key)
    bio = io.BytesIO(body)
    lower_key = key.lower()

    if lower_key.endswith(".csv") or lower_key.endswith(".txt"):
        return pd.read_csv(bio), "csv"
    if lower_key.endswith(".csv.gz") or lower_key.endswith(".txt.gz"):
        return pd.read_csv(bio, compression="gzip"), "csv"

    if lower_key.endswith(".parquet") or lower_key.endswith(".pq"):
        return pd.read_parquet(bio), "parquet"

    if lower_key.endswith(".ndjson") or lower_key.endswith(".ndjson.gz"):
        return pd.read_json(bio, lines=True, compression=("gzip" if lower_key.endswith(".gz") else None)), "json"
    if lower_key.endswith(".json") or lower_key.endswith(".json.gz"):
        return pd.read_json(bio, lines=False, compression=("gzip" if lower_key.endswith(".gz") else None)), "json"

    # default: try CSV without/with gzip
    try:
        return pd.read_csv(io.BytesIO(body)), "csv"
    except Exception:
        return pd.read_csv(io.BytesIO(body), compression="gzip"), "csv"

def save_df_to_minio(df: pd.DataFrame, key: str, fmt: Optional[str] = None, index: bool = False) -> None:
    fmt = fmt or os.path.splitext(key)[1].lstrip(".").lower() or "csv"
    lower_key = key.lower()
    if fmt == "csv" or lower_key.endswith(".csv") or lower_key.endswith(".csv.gz"):
        buf = io.BytesIO()
        comp = "gzip" if lower_key.endswith(".gz") else None
        df.to_csv(buf, index=index, compression=comp)
        _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue(), ContentType="text/csv"); return
    if fmt == "parquet" or lower_key.endswith(".parquet") or lower_key.endswith(".pq"):
        buf = io.BytesIO(); df.to_parquet(buf, index=index)
        _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue(), ContentType="application/octet-stream"); return
    # json
    buf = io.BytesIO(json.dumps(df.to_dict(orient="records")).encode("utf-8"))
    _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue(), ContentType="application/json")


# dqc_utils.py (add)
def s3_join(*parts: str) -> str:
    return "/".join(str(p).strip("/") for p in parts if str(p).strip("/"))

def get_curated_root(datasetname: str, dataname: str) -> str:
    return s3_join(CURATED_PREFIX, datasetname, dataname)

def curated_keys(datasetname: str, dataname: str) -> Dict[str, str]:
    root = get_curated_root(datasetname, dataname)
    return {
        "root":        root + "/",
        "data_csv":    s3_join(root, "data.csv"),
        "train_csv":   s3_join(root, "train.csv"),
        "test_csv":    s3_join(root, "test.csv"),
        "incomplete":  s3_join(root, "incomplete.csv"),
        "info_json":   s3_join(root, "Info", f"{dataname}.json"),
        # "models_dir":  s3_join(root, "models"),
        # "metrics_dir": s3_join(root, "metrics"),
        "preds_dir":   s3_join(root, "predictions"),
        "logs_dir":    s3_join(root, "logs"),
        "masks_dir":   s3_join(root, "masks"),
    }

# ============================ Timestamp Inference & Normalization ============================

NAME_PATTERNS = [
    r"^time(?:_?stamp)?(?:\[.*\])?$",   # time, timestamp, time_stamp, time_stamp[UTC]
    r"^datetime$",
    r"^date$",
    r"^(?:event[_-]?(?:time|date))$",
    r"^(?:created|updated)_at$",
    # 次要模糊匹配（放后面）
    r"\btime[_-]?stamp\b",
    r"\bdatetime\b",
    r"\bevent[_-]?time\b",
]

# preclean the time format strings
def _preclean_time_strings(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()

    # 去掉像 "['...']" 的包裹
    s = s.str.replace(r"^\[\s*['\"]?(.*?)['\"]?\s*\]$", r"\1", regex=True)

    # HH:MM:SS-ffffff -> HH:MM:SS.ffffff
    s = s.str.replace(r"^(\d{1,2}):(\d{2}):(\d{2})-(\d{1,6})$", r"\1:\2:\3.\4", regex=True)

    # HH-MM-SS-ffffff 或 HH-MM-SS.ffffff -> HH:MM:SS.ffffff
    s = s.str.replace(r"^(\d{1,2})-(\d{2})-(\d{2})[.\-](\d{1,6})$", r"\1:\2:\3.\4", regex=True)

    # HH-MM-SS -> HH:MM:SS
    s = s.str.replace(r"^(\d{1,2})-(\d{2})-(\d{2})$", r"\1:\2:\3", regex=True)

    # 末尾的 .毫秒 补齐到 ≥3 位，最多保留6位（微秒）
    def _pad_frac(m):
        ms = m.group(1)[:6]             # 最多 6 位
        if 1 <= len(ms) <= 2:
            ms = ms.ljust(3, "0")       # 至少 3 位
        return "." + ms
    s = s.str.replace(r"\.(\d{1,6})$", lambda m: _pad_frac(m), regex=True)

    # 空字串统一为 NaN
    s = s.mask(s.str.len() == 0)
    return s


EXPLICIT_TIME_FORMATS = [
    # 日期+时间
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%d-%m-%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
    "%Y%m%d%H%M%S", "%Y%m%d",
    "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S%z",

    # 只有时间（冒号/短横，带/不带毫秒/微秒）
    "%H:%M:%S.%f", "%H:%M:%S",
    "%H-%M-%S.%f", "%H-%M-%S-%f", "%H-%M-%S",
]



def _preclean_time_strings(series: pd.Series) -> pd.Series:
    """
    Normalize common odd formats:
    - "['00-42-15-500']" -> "00-42-15-500"
    - "HH-MM-SS.mmm" or "HH-MM-SS-mmm" -> "HH:MM:SS.mmm"
    - pad milliseconds to at least 3 digits
    """
    s = series.astype(str).str.strip()
    # strip wrappers like ['...']
    s = s.str.replace(r"^\[\s*['\"]?(.*?)['\"]?\s*\]$", r"\1", regex=True)
    # HH-MM-SS[.-]fff -> HH:MM:SS.fff
    s = s.str.replace(r"^(\d{1,2})-(\d{2})-(\d{2})[.\-](\d{1,6})$", r"\1:\2:\3.\4", regex=True)
    # HH-MM-SS       -> HH:MM:SS
    s = s.str.replace(r"^(\d{1,2})-(\d{2})-(\d{2})$", r"\1:\2:\3", regex=True)
    # 必须包含：HH:MM:SS-ffffff -> HH:MM:SS.ffffff
    s = s.str.replace(r"^(\d{1,2}):(\d{2}):(\d{2})-(\d{1,6})$", r"\1:\2:\3.\4", regex=True)
    # 以及 HH-MM-SS[-|.]fff -> HH:MM:SS.fff
    s = s.str.replace(r"^(\d{1,2})-(\d{2})-(\d{2})[.\-](\d{1,6})$", r"\1:\2:\3.\4", regex=True)

    # pad trailing milliseconds to ≥3 digits
    def _pad_ms(m):
        ms = m.group(1)
        if 1 <= len(ms) <= 2:
            ms = ms.ljust(3, "0")
        return "." + ms
    s = s.str.replace(r"\.(\d{1,6})$", lambda m: _pad_ms(m), regex=True)
    return s

# def _try_parse_datetime(series, sample=1000, tz_aware=False):
#     s = series.dropna().astype(str).str.strip()
#     s = s[~s.str.lower().isin({"", "nan", "none", "null"})]
#     if s.empty:
#         return 0.0
#     s = s.sample(min(sample, len(s)), random_state=42)
#     s_clean = _preclean_time_strings(s)
#     parsed = pd.to_datetime(s_clean, errors="coerce", utc=tz_aware)
#     return float(parsed.notna().mean())

def _looks_like_epoch(series: pd.Series) -> tuple[bool, Optional[str]]:
    """Detect epoch s/ms/us/ns by magnitude + quasi-monotonicity."""
    if not is_numeric_dtype(series):
        return False, None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False, None
    med = s.median()
    span = (s.quantile(0.9) - s.quantile(0.1)) if len(s) > 1 else 0
    if span <= 0:  # constant series → not time
        return False, None
    candidates: List[str] = []
    for unit, lo, hi in [("s", 1e9, 1e10), ("ms", 1e12, 1e13), ("us", 1e15, 1e16), ("ns", 1e18, 1e19)]:
        if lo <= med < hi:
            candidates.append(unit)
    if not candidates:
        return False, None
    unit = next(u for u in ["s", "ms", "us", "ns"] if u in candidates)
    diffs = s.diff().dropna()
    if not diffs.empty and (diffs < 0).mean() <= 0.2:
        return True, unit
    return False, None
def detect_timestamp_column(df: pd.DataFrame, configured_name: Optional[str] = None) -> tuple[Optional[str], Dict[str, Any]]:
    """
    Detect a single timestamp column.
    Returns (column_name_or_None, details={"strategy","unit","confidence"})
    """
    details: Dict[str, Any] = {"strategy": None, "unit": None, "confidence": 0.0}

    # ---- local helpers ------------------------------------------------------
    TIME_ONLY_REGEX = re.compile(r"^\s*\d{1,2}[:\-]\d{2}[:\-]\d{2}(?:[.\-]\d{1,6})?\s*$")
    # rename-friendly patterns: time / timestamp / time_stamp / time_stamp[UTC]
    EXTRA_NAME_PATTERNS = [r"^time(?:_?stamp)?(?:\[.*\])?$"]

    def _score_series_as_time(s: pd.Series) -> Tuple[float, Optional[str]]:
        """
        Score a candidate text series as timestamp.
        Returns (score in [0,1], epoch_unit_if_used).
        """
        epoch_unit_used: Optional[str] = None

        # 0) preclean (strip ['...'], unify -, pad fraction)
        s_clean = _preclean_time_strings(s.astype(str))

        # 1) quick regex-based time-only score (no parsing)
        #    e.g. "04:09:06-325991", "12-00-06.900", "00-42-15-0"
        regex_rate = float(s_clean.str.match(TIME_ONLY_REGEX, na=False).mean())

        # 2) explicit formats first (robust)
        best = regex_rate  # start from regex score so time-only won't be 0
        for fmt in EXPLICIT_TIME_FORMATS:
            try:
                r = pd.to_datetime(s_clean, format=fmt, errors="coerce", utc=True).notna().mean()
                best = max(best, float(r))
                if best >= 0.95:
                    return best, None
            except Exception:
                pass

        # 3) generic parser on cleaned strings
        try:
            r2 = pd.to_datetime(s_clean, errors="coerce", utc=True).notna().mean()
            best = max(best, float(r2))
            if best >= 0.95:
                return best, None
        except Exception:
            pass

        # 4) epoch digits embedded or pure numeric (10–19 digits)
        try:
            digits = s_clean.str.extract(r"(\d{10,19})")[0]
            if digits is not None:
                x = pd.to_numeric(digits, errors="coerce")
                med = x.median()
                unit = None
                if pd.notna(med):
                    if 1e18 <= med < 1e19: unit = "ns"
                    elif 1e15 <= med < 1e16: unit = "us"
                    elif 1e12 <= med < 1e13: unit = "ms"
                    elif 1e9  <= med < 1e10: unit = "s"   # <- 覆盖 1636553178
                if unit:
                    p = pd.to_datetime(x, unit=unit, errors="coerce", utc=True)
                    r3 = float(p.notna().mean())
                    if r3 > best:
                        best, epoch_unit_used = r3, unit
        except Exception:
            pass

        return best, epoch_unit_used

    # 合并列名模式（保序去重）
    combined_name_patterns = []
    for pat in (list(NAME_PATTERNS) if 'NAME_PATTERNS' in globals() else []) + EXTRA_NAME_PATTERNS:
        if pat not in combined_name_patterns:
            combined_name_patterns.append(pat)

    # ---- 0) configured name -------------------------------------------------
    if configured_name and configured_name in df.columns:
        col = configured_name
        if is_datetime64_any_dtype(df[col]):
            details.update({"strategy": "configured", "confidence": 1.0})
            return col, details
        score, unit_used = _score_series_as_time(df[col])
        details.update({"strategy": "configured", "unit": unit_used, "confidence": float(score)})
        if score >= 0.6:
            return col, details
        # else fallthrough

    # ---- 1) name regex shortlist -------------------------------------------
    lowers = {c.lower(): c for c in df.columns}
    for pat in combined_name_patterns:
        prog = re.compile(pat, re.IGNORECASE)
        for lc, orig in lowers.items():
            # also try stripping bracket suffix like [UTC] for matching
            lc_stripped = re.sub(r"\[.*?\]$", "", lc)
            if prog.match(lc) or prog.search(lc) or prog.match(lc_stripped) or prog.search(lc_stripped):
                col = orig
                if is_datetime64_any_dtype(df[col]):
                    details.update({"strategy": "name_regex", "confidence": 1.0})
                    return col, details
                score, unit_used = _score_series_as_time(df[col])
                if score >= 0.6:
                    details.update({"strategy": "name_regex", "unit": unit_used, "confidence": float(score)})
                    return col, details

    # ---- 2) dtype is already datetime --------------------------------------
    for c in df.columns:
        if is_datetime64_any_dtype(df[c]):
            details.update({"strategy": "dtype_datetime", "confidence": 1.0})
            return c, details

    # ---- 3) free scan over string columns ----------------------------------
    best_col, best_score, best_unit = None, 0.0, None
    for c in df.columns:
        if is_string_dtype(df[c]):
            score, unit_used = _score_series_as_time(df[c])
            if score > best_score:
                best_col, best_score, best_unit = c, score, unit_used
    if best_col and best_score >= 0.6:
        details.update({"strategy": "explicit_formats_scan", "unit": best_unit, "confidence": float(best_score)})
        return best_col, details

    # ---- 4) epoch numeric (pure numeric dtype) ------------------------------
    for c in df.columns:
        ok, unit = _looks_like_epoch(df[c])
        if ok:
            details.update({"strategy": "epoch_numeric", "unit": unit, "confidence": 0.8})
            return c, details

    # ---- none ---------------------------------------------------------------
    return None, {"strategy": "none", "unit": None, "confidence": 0.0}

def _coerce_to_datetime_robust(series: pd.Series, default_tz: str = "UTC") -> tuple[pd.Series, dict]:
    """
    Parse heterogeneous timestamp columns into pandas datetime64[ns, UTC].

    - Uses pre-cleaning for odd strings (e.g., HH-MM-SS-fff, ['...']).
    - Tries: (A) fast generic parse on cleaned strings,
             (B) explicit formats (incl. time-only),
             (C) numeric epoch (s/ms/us/ns) by magnitude,
             (D) epoch digits embedded in strings,
             (E) last-resort localize->UTC.
    - Flags time-only detections so the caller can anchor a date if desired.

    Returns
    -------
    (parsed_series_utc, meta_dict)
      parsed_series_utc : pd.Series[datetime64[ns, UTC]]
      meta_dict         : {
        "strategy": str,
        "unit": Optional[str],             # epoch unit if used: "s"|"ms"|"us"|"ns"
        "parse_rate": float,               # fraction of non-NaT after parse
        "notes": list[str],
        "time_only": bool                  # True when values look like HH:MM:SS[.fff] without date
      }
    """
    s = series.copy()
    meta = {"strategy": None, "unit": None, "parse_rate": 0.0, "notes": [], "time_only": False}

    # --- Case 0: already datetime-like ---
    if is_datetime64_any_dtype(s):
        try:
            # localize naïve → default_tz, then convert to UTC
            if s.dt.tz is None:
                s = s.dt.tz_localize(default_tz)
            s = s.dt.tz_convert("UTC")
            meta.update({"strategy": "already_datetime", "parse_rate": 1.0})
            return s, meta
        except Exception as e:
            meta["notes"].append(f"already_datetime_convert_failed: {e}")

    # Prepare string & numeric views once
    is_str_like = (s.dtype == object) or pd.api.types.is_string_dtype(s)
    s_str = None
    if is_str_like:
        s_str = s.astype(str).str.strip()
        # drop common empties before scoring to avoid depressing parse_rate
        mask_valid = ~s_str.str.lower().isin({"", "nan", "none", "null"})
        s_str = s_str.where(mask_valid, None)

    # --- Case A: fast generic parse on pre-cleaned strings ---
    try:
        if is_str_like:
            s0 = _preclean_time_strings(s_str)
            parsed = pd.to_datetime(s0, errors="coerce", utc=True)
            rate = float(parsed.notna().mean())
            if rate >= 0.90:
                # mark time-only if many rows normalize to 1900-01-01 (pandas default for time-only)
                try:
                    if (parsed.dt.normalize() == pd.Timestamp("1900-01-01", tz="UTC")).mean() > 0.5:
                        meta["time_only"] = True
                except Exception:
                    pass
                meta.update({"strategy": "generic_cleaned", "parse_rate": rate})
                return parsed, meta
            meta["notes"].append(f"generic_cleaned_rate={rate:.2f}")
    except Exception as e:
        meta["notes"].append(f"generic_cleaned_error: {e}")

    # --- Case B: explicit formats (try strongest first) ---
    best = None
    best_rate = 0.0
    tried_time_only = False
    if is_str_like:
        s0 = _preclean_time_strings(s_str)
        for fmt in EXPLICIT_TIME_FORMATS:
            try:
                p = pd.to_datetime(s0, format=fmt, errors="coerce", utc=True)
                r = float(p.notna().mean())
                # identify time-only formats
                if fmt in ("%H:%M:%S.%f", "%H:%M:%S", "%H-%M-%S.%f", "%H-%M-%S-%f", "%H-%M-%S"):
                    tried_time_only = True
                if r > best_rate:
                    best, best_rate = p, r
                if r >= 0.95:
                    best, best_rate = p, r
                    break
            except Exception:
                continue
    if best is not None and best_rate >= 0.60:
        meta["time_only"] = meta["time_only"] or tried_time_only
        meta.update({"strategy": "explicit_formats", "parse_rate": float(best_rate)})
        return best, meta
    if best is not None:
        meta["notes"].append(f"explicit_formats_rate={best_rate:.2f}")

    # --- Case C: numeric epoch by magnitude (s/ms/us/ns) ---
    try:
        x = pd.to_numeric(s, errors="coerce")
        med = x.median()
        if pd.notna(med):
            unit = None
            # infer unit by magnitude (robust rough bands)
            if 1e18 <= med < 1e19:
                unit = "ns"
            elif 1e15 <= med < 1e16:
                unit = "us"
            elif 1e12 <= med < 1e13:
                unit = "ms"
            elif 1e9 <= med < 1e10:
                unit = "s"
            if unit:
                p = pd.to_datetime(x, unit=unit, errors="coerce", utc=True)
                rate = float(p.notna().mean())
                if rate >= 0.90:
                    meta.update({"strategy": "epoch_numeric", "unit": unit, "parse_rate": rate})
                    return p, meta
                meta["notes"].append(f"epoch_numeric_rate={rate:.2f} (unit={unit})")
    except Exception as e:
        meta["notes"].append(f"epoch_numeric_error: {e}")

    # --- Case D: epoch digits embedded inside strings ---
    if is_str_like:
        try:
            digits = s_str.str.extract(r"(\d{10,19})")[0]
            if digits is not None:
                vals = pd.to_numeric(digits, errors="coerce")
                med = vals.median()
                unit = None
                if pd.notna(med):
                    if 1e18 <= med < 1e19: unit = "ns"
                    elif 1e15 <= med < 1e16: unit = "us"
                    elif 1e12 <= med < 1e13: unit = "ms"
                    elif 1e9 <= med < 1e10:  unit = "s"
                if unit:
                    p = pd.to_datetime(vals, unit=unit, errors="coerce", utc=True)
                    rate = float(p.notna().mean())
                    if rate >= 0.90:
                        meta.update({"strategy": "epoch_in_string", "unit": unit, "parse_rate": rate})
                        return p, meta
                    meta["notes"].append(f"epoch_in_string_rate={rate:.2f} (unit={unit})")
        except Exception as e:
            meta["notes"].append(f"epoch_in_string_error: {e}")

    # --- Case E: last resort: naive parse → localize(default_tz) → UTC ---
    try:
        # Try on raw strings if available; else on original
        base = _preclean_time_strings(s_str) if is_str_like else s
        parsed = pd.to_datetime(base, errors="coerce")  # tz-naive
        rate = float(parsed.notna().mean())
        if rate > 0.0:
            parsed = parsed.dt.tz_localize(default_tz, nonexistent="NaT", ambiguous="NaT").dt.tz_convert("UTC")
            # mark time-only by 1900-01-01 heuristic if applicable
            try:
                if (parsed.dt.normalize() == pd.Timestamp("1900-01-01", tz="UTC")).mean() > 0.5:
                    meta["time_only"] = True
            except Exception:
                pass
            meta.update({"strategy": "last_resort_localize", "parse_rate": rate})
            return parsed, meta
        meta["notes"].append("last_resort_rate=0.00")
    except Exception as e:
        meta["notes"].append(f"last_resort_error: {e}")

    # --- Fail: return all-NaT UTC series with diagnostics ---
    meta.update({"strategy": "failed", "parse_rate": 0.0})
    return pd.Series([pd.NaT] * len(s), dtype="datetime64[ns, UTC]"), meta

def standardize_timestamp_column(
    df: pd.DataFrame, ts_col: str, out_col: str, default_tz: str = "UTC"
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    g = df.copy()
    out, parse_meta = _coerce_to_datetime_robust(g[ts_col], default_tz=default_tz)
    g[out_col] = out
    g = g.sort_values(out_col, kind="stable").reset_index(drop=True)
    return g, {"source_col": ts_col, "out_col": out_col, "parse": parse_meta, "sorted": True}

# ============================ Schema profiling ============================
def build_schema_profile(
    df: pd.DataFrame,
    configured_ts: str | None = None,
    target: str | None = None,
    max_enum: int = 50
) -> Dict[str, Any]:
    prof: Dict[str, Any] = {
        "table": {"rows": int(len(df)), "cols": int(df.shape[1])},
        "timestamp": {},
        "columns": {},
        "primary_key_candidates": [],
        "hints": [],
    }

    ts_col, ts_info = detect_timestamp_column(df, configured_ts)
    prof["timestamp"] = {"column": ts_col, **ts_info}

    pk_scores: List[tuple[str, float]] = []
    path_like_cols: List[str] = []
    for c in df.columns:
        s = df[c]
        null_rate = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))
        is_unique = (nunique == len(df) - int(s.isna().sum()))
        if is_bool_dtype(s):   logical = "boolean"
        elif is_datetime64_any_dtype(s): logical = "datetime"
        elif is_integer_dtype(s): logical = "integer"
        elif is_float_dtype(s):   logical = "float"
        elif is_numeric_dtype(s): logical = "numeric"
        elif is_string_dtype(s) and nunique <= max_enum: logical = "categorical"
        else: logical = "text"

        looks_path = bool(s.dropna().astype(str).str.match(r"^\.?/|[A-Za-z]:\\").head(20).any())
        if looks_path: path_like_cols.append(c)

        colp: Dict[str, Any] = {
            "logical_type": logical,
            "pandas_dtype": str(s.dtype),
            "nullable": null_rate > 0.0,
            "null_rate": null_rate,
            "cardinality": nunique,
            "is_unique": bool(is_unique),
            "uniqueness_ratio": float(nunique / max(1, len(df))),
            "example_values": s.dropna().astype(str).head(3).tolist(),
            "path_like": looks_path,
        }
        if logical in {"integer","float","numeric"}:
            sn = pd.to_numeric(s, errors="coerce")
            if not sn.dropna().empty:
                colp.update({
                    "min": float(sn.min()),
                    "max": float(sn.max()),
                    "mean": float(sn.mean()),
                    "std": float(sn.std()),
                })
        prof["columns"][c] = colp
        score = (1.0 if is_unique else colp["uniqueness_ratio"]) * (1.0 - null_rate)
        pk_scores.append((c, float(score)))

    prof["primary_key_candidates"] = [
        {"column": c, "score": round(sc, 4)} for c, sc in sorted(pk_scores, key=lambda x: x[1], reverse=True)[:5]
    ]
    if "unit1_pwr_60ghz" in df.columns and df["unit1_pwr_60ghz"].isna().any():
        prof["hints"].append({"column":"unit1_pwr_60ghz","rule":"nans_as_zero","note":"Replace NaNs with 0 for received-power vectors."})
    if target and target in df.columns:
        prof["target"] = {"column": target, "classes": int(df[target].nunique(dropna=True))}
    if path_like_cols:
        prof["hints"].append({"paths_in_columns": path_like_cols})
    return prof

# ============================ Cross-dataset time normalization ============================
def _get_dataset_tz(dataset_name: str) -> str:
    try:
        tz_map = json.loads(TS_DATASET_TZ_MAP or "{}")
        tz = tz_map.get(dataset_name)
        return tz or DEFAULT_TZ
    except Exception:
        return DEFAULT_TZ

def _extract_date_from_key_like(s: str) -> str | None:
    """Extract YYYY-MM-DD or YYYYMMDD from a path-like string."""
    if not s:
        return None
    m = re.search(r"(\d{4})[-_/]?(0[1-9]|1[0-2])[-_/]?([0-2]\d|3[01])", s)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

def _choose_anchor_date(ti, df: pd.DataFrame, ts_col: str | None) -> str:
    if TS_ANCHOR_DATE:
        return TS_ANCHOR_DATE
    key = ti.xcom_pull(task_ids="resolve_input_key", key="resolved_key") \
          or (ti.xcom_pull(task_ids="load_raw_data", key="raw_handle") or {}).get("key")
    extracted = _extract_date_from_key_like(key or "")
    if extracted:
        return extracted
    try:
        return str(ti.logical_date.date())
    except Exception:
        return ""

def _snap_timestamp_if_needed(ts: pd.Series) -> pd.Series:
    return ts if not TS_SNAP_UNIT else ts.dt.floor(TS_SNAP_UNIT)

def normalize_ts_for_gap(
    ti,
    df: pd.DataFrame,
    dataset_name: str,
    configured_ts_col: str | None,
    out_col: str = TS_STD_COL,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normalize time for gap analysis:
      - find timestamp column (configured or auto)
      - robust parse (strings/epoch/time-only)
      - localize to dataset tz then convert to UTC
      - optional snap to a unit (S/L/T/H)
    """
    meta: Dict[str, Any] = {"source_col": None, "dataset_tz": None, "anchored": False, "snap": TS_SNAP_UNIT or "", "notes": []}

    ts_col = configured_ts_col if configured_ts_col in df.columns else None
    if not ts_col:
        ts_col, _ = detect_timestamp_column(df, configured_name=None)
    if not ts_col or ts_col not in df.columns:
        meta["notes"].append("no_timestamp_col_detected")
        g = df.copy(); g[out_col] = pd.NaT
        return g, meta
    meta["source_col"] = ts_col

    g = df.copy()
    ts_parsed, parse_meta = _coerce_to_datetime_robust(g[ts_col], default_tz="UTC")
    meta["parse"] = parse_meta

    # time-only anchor (1900-01-01 heuristic or flagged)
    try:
        time_only_flag = bool(parse_meta.get("time_only"))
        naive_1900 = (ts_parsed.dt.normalize() == pd.Timestamp("1900-01-01", tz="UTC")).mean() > 0.5
    except Exception:
        time_only_flag, naive_1900 = False, False

    if time_only_flag or naive_1900:
        anchor = _choose_anchor_date(ti, g, ts_col)
        if anchor:
            combo = anchor + " " + g[ts_col].astype(str).str.strip().replace("nan", pd.NA).fillna("")
            ts2, meta2 = _coerce_to_datetime_robust(combo, default_tz=_get_dataset_tz(dataset_name))
            ts_parsed = ts2.dt.tz_convert("UTC")
            meta["anchored"] = True
            meta["parse_after_anchor"] = meta2
        else:
            meta["notes"].append("time_only_no_anchor")

    if ts_parsed.dt.tz is None:
        local_tz = _get_dataset_tz(dataset_name)
        meta["dataset_tz"] = local_tz
        ts_parsed = ts_parsed.dt.tz_localize(local_tz, nonexistent="NaT", ambiguous="NaT").dt.tz_convert("UTC")
    else:
        meta["dataset_tz"] = _get_dataset_tz(dataset_name)

    ts_parsed = _snap_timestamp_if_needed(ts_parsed)
    g[out_col] = ts_parsed
    g = g.sort_values(out_col, kind="stable").reset_index(drop=True)
    return g, meta

# ============================ Adaptive time-gap helpers ============================
_week_map = {"mon":0,"tue":1,"wed":2,"thu":3,"fri":4,"sat":5,"sun":6}

def _parse_active_cron(cron_expr: str) -> List[tuple[Set[int], int, int]]:
    """
    Parse patterns like: 'mon-fri 08:00-18:00; sat 09:00-12:00' (case-insensitive).
    Supports cross-midnight: 'fri 22:00-06:00'. Returns [(dayset, start_min, end_min)].
    """
    if not cron_expr:
        return []
    parts = re.split(r"[;,]", cron_expr)
    windows: List[tuple[Set[int], int, int]] = []
    for raw in parts:
        p = re.sub(r"\s+", " ", raw.strip().lower())
        if not p:
            continue
        m = re.match(r"([a-z]{3}(?:-[a-z]{3})?)\s+(\d{1,2}:\d{2})-(\d{1,2}:\d{2})$", p)
        if not m:
            return []
        days, start, end = m.group(1), m.group(2), m.group(3)
        if "-" in days:
            a,b = days.split("-"); ai, bi = _week_map[a], _week_map[b]
            dayset = set(range(ai, bi+1)) if ai <= bi else set(list(range(ai,7))+list(range(0,bi+1)))
        else:
            dayset = { _week_map[days] }
        h1,m1 = map(int, start.split(":")); h2,m2 = map(int, end.split(":"))
        windows.append((dayset, h1*60+m1, h2*60+m2))
    return windows

_ACTIVE_WINDOWS = _parse_active_cron(TS_ACTIVE_CRON)

def _in_active_window(ts: pd.Series) -> pd.Series:
    if not _ACTIVE_WINDOWS:
        return pd.Series(True, index=ts.index)
    # ensure tz-aware UTC series
    if not is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts, errors="coerce", utc=True)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    local = ts.dt.tz_convert(TS_ACTIVE_TZ)
    wk = local.dt.weekday
    mins = local.dt.hour*60 + local.dt.minute
    mask = pd.Series(False, index=ts.index)
    for wset, smin, emin in _ACTIVE_WINDOWS:
        if emin >= smin:
            mask |= (wk.isin(wset)) & (mins >= smin) & (mins < emin)
        else:
            # cross-midnight window
            mask |= (wk.isin(wset) & (mins >= smin)) | (wk.isin(wset) & (mins < emin))
    return mask

def _fallback_expected_from_diffs(diffs: pd.Series) -> Optional[pd.Timedelta]:
    if isinstance(diffs, pd.Index):
        diffs = diffs.to_series()
    diffs = pd.to_timedelta(diffs, errors="coerce").dropna()
    if diffs.empty:
        return None
    try:
        mode = diffs.mode().iloc[0]
        if pd.notna(mode) and mode > pd.Timedelta(0):
            return pd.Timedelta(mode)
    except Exception:
        pass
    med = diffs.median()
    if pd.isna(med) or med <= pd.Timedelta(0):
        return None
    seconds = med.total_seconds()
    if abs(round(seconds) - seconds) <= 0.05:
        return pd.to_timedelta(int(round(seconds)), unit="s")
    return pd.Timedelta(med)

def _infer_local_expected_delta(ts: pd.Series, window: int = 200) -> pd.Series:
    """Rolling expected inter-arrival using median & 25% quantile; falls back to constant if needed."""
    s = ts.dropna().sort_values()
    diffs = s.diff().dropna()

    if isinstance(diffs, pd.Index):
        diffs = diffs.to_series()

    if diffs.empty:
        return pd.Series(dtype="timedelta64[ns]")

    # --- cast to numeric nanoseconds for rolling ops ---
    diffs_ns = pd.to_timedelta(diffs).astype("timedelta64[ns]").view("int64")
    diffs_ns = pd.Series(diffs_ns, index=diffs.index, dtype="float64")

    med_ns = diffs_ns.rolling(window, min_periods=max(10, window // 5)).median()
    q25_ns = diffs_ns.rolling(window, min_periods=max(10, window // 5)).quantile(0.25)

    exp_ns = med_ns.combine(q25_ns, lambda a, b: b if pd.notna(b) else a).bfill().ffill()

    if exp_ns.isna().all():
        fallback = _fallback_expected_from_diffs(diffs)
        if fallback is not None:
            return pd.Series([fallback] * len(diffs), index=diffs.index)

    # convert numeric nanoseconds back to timedeltas
    exp_local = pd.to_timedelta(exp_ns, unit="ns")
    return exp_local


def compute_time_gaps_smart(
    df: pd.DataFrame,
    ts_col: str,
    expected_delta: Optional[pd.Timedelta],   # if None, infer locally
    tol_mult: float = TS_GAP_TOL_MULT,
    window: int = 200,
    respect_calendar: bool = True,
) -> Dict[str, Any]:
    """
    Thorough gap scan with adaptive local cadence & optional calendar mask.
    Returns:
      {
        "expected_mode_seconds": float|None,
        "method": "local-rolling"|"fixed",
        "gaps": [ {start,end,delta_seconds,expected_local_seconds,missing_points} ],
        "counts": {total_points, unique_timestamps, duplicate_timestamps, missing_windows, missing_points},
      }
    """
    out: Dict[str, Any] = {
        "expected_mode_seconds": None,
        "method": "local-rolling" if expected_delta is None else "fixed",
        "gaps": [],
        "counts": {
            "total_points": int(len(df)),
            "unique_timestamps": 0,
            "duplicate_timestamps": 0,
            "missing_windows": 0,
            "missing_points": 0,
        },
    }
    if ts_col not in df.columns:
        return out

    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True).dropna().sort_values()
    if ts.empty:
        return out

    if respect_calendar and _ACTIVE_WINDOWS:
        ts = ts[_in_active_window(ts)]
        if ts.empty:
            return out

    dup = int(ts.duplicated().sum())
    ts = ts.drop_duplicates()
    out["counts"]["unique_timestamps"] = int(len(ts))
    out["counts"]["duplicate_timestamps"] = dup

    diffs = ts.diff().dropna()
    if isinstance(diffs, pd.Index):
        diffs = diffs.to_series()
    diffs = pd.to_timedelta(diffs, errors="coerce").dropna()
    if diffs.empty:
        return out

    try:
        mode = diffs.mode().iloc[0]
        if pd.notna(mode) and mode > pd.Timedelta(0):
            out["expected_mode_seconds"] = float(pd.Timedelta(mode).total_seconds())
    except Exception:
        pass

    if expected_delta is None:
        exp_local = _infer_local_expected_delta(ts, window=window)
        if exp_local.empty or exp_local.isna().all():
            fb = _fallback_expected_from_diffs(diffs)
            if fb is None:
                return out
            exp = pd.Series([fb]*len(diffs), index=diffs.index)
        else:
            exp = exp_local.reindex(diffs.index, method="nearest").bfill().ffill()
    else:
        exp = pd.Series(expected_delta, index=diffs.index)

    tol = exp * tol_mult
    if tol.isna().any():
        tol = tol.fillna(value=pd.Timedelta(seconds=float(exp.dropna().max().total_seconds())))
    gap_mask = diffs > tol
    if not bool(gap_mask.any()):
        return out

    idx = gap_mask[gap_mask].index
    for i in idx:
        delta = diffs.loc[i]
        expd  = exp.loc[i]
        missing = max(0, int(delta / expd) - 1)
        pos = ts.index.get_loc(i)
        start_ts = ts.iloc[pos - 1]; end_ts = ts.loc[i]
        out["gaps"].append({
            "start": start_ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "end":   end_ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "delta_seconds": float(delta.total_seconds()),
            "expected_local_seconds": float(expd.total_seconds()),
            "missing_points": int(missing),
        })

    out["counts"]["missing_windows"] = len(out["gaps"])
    out["counts"]["missing_points"]  = int(sum(g["missing_points"] for g in out["gaps"]))
    return out

# =========================== helper: human-readable primary key ===========================
import itertools
from pandas.api.types import is_string_dtype, is_integer_dtype, is_float_dtype, is_datetime64_any_dtype

def _is_texty(series: pd.Series, sample: int = 500) -> bool:
    """Heuristic: long strings / high token entropy → not suitable for PK."""
    s = series.dropna().astype(str)
    if s.empty: return False
    s = s.sample(min(sample, len(s)), random_state=42)
    # very long average length or many spaces/punctuations → texty
    avg_len = s.str.len().mean()
    space_rate = (s.str.contains(r"\s", regex=True)).mean()
    return bool((avg_len and avg_len > 64) or (space_rate > 0.15))

def _name_boost(col: str) -> float:
    lc = col.lower()
    for pat in PK_NAME_PATTERNS:
        if re.match(pat, lc):
            return 0.15  # small positive boost for id-like names
    return 0.0

def _dtype_boost(s: pd.Series) -> float:
    if is_integer_dtype(s): return 0.10
    if is_string_dtype(s):  return 0.05
    if is_float_dtype(s):   return -0.05
    return 0.0

def _time_like(series: pd.Series, colname: str) -> bool:
    if is_datetime64_any_dtype(series): return True
    lc = colname.lower()
    if any(k in lc for k in ("time", "timestamp", "datetime", "ts")): return True
    if is_integer_dtype(series) or is_float_dtype(series):
        try:
            med = pd.to_numeric(series, errors="coerce").median()
            return pd.notna(med) and (1e9 <= med < 1e19)  # epoch-ish
        except Exception:
            return False
    if is_string_dtype(series):
        samp = series.dropna().astype(str).head(10)
        if samp.empty: return False
        try:
            ok = pd.to_datetime(samp, errors="coerce", utc=True).notna().mean()
            return ok >= 0.6
        except Exception:
            return False
    return False

def detect_primary_key(
    df: pd.DataFrame,
    configured_pk: Optional[str | list[str]] = None,
    allow_time: Optional[bool] = None,
    max_width: Optional[int] = None,
) -> dict:
    """
    Returns:
      {
        "primary_key": [cols],
        "uniqueness": float,        # unique non-null rows / total rows
        "null_rows": int,           # rows with any PK field null (for chosen PK)
        "used_time_col": bool,
        "candidates_scored": [ {"column":..., "uniqueness":..., "null_rate":..., "score":...}, ... ],
        "tested": {"single":[...], "pairs":[...], "triples":[...]}
      }
    """
    allow_time = PK_ALLOW_TIME if allow_time is None else bool(allow_time)
    max_width = PK_MAX_WIDTH if max_width is None else int(max_width)

    n = len(df)
    if n == 0:
        return {"primary_key": [], "uniqueness": 0.0, "null_rows": 0, "used_time_col": False,
                "candidates_scored": [], "tested": {}}

    # 0) explicit config
    if configured_pk:
        cols = [c.strip() for c in (configured_pk if isinstance(configured_pk, list) else configured_pk.split(",")) if c.strip()]
        cols = [c for c in cols if c in df.columns]
        if cols:
            nn_mask = ~df[cols].isna().any(axis=1)
            uniq = float(df.loc[nn_mask, cols].drop_duplicates().shape[0] / max(1, nn_mask.sum()))
            return {
                "primary_key": cols,
                "uniqueness": uniq,
                "null_rows": int((~nn_mask).sum()),
                "used_time_col": any(_time_like(df[c], c) for c in cols),
                "candidates_scored": [],
                "tested": {"configured": cols},
            }

    # 1) score single columns
    scored = []
    for c in df.columns:
        s = df[c]
        null_rate = float(s.isna().mean())
        if null_rate > PK_NULL_RATE_MAX: 
            # keep but penalize heavily (still may be used in combos)
            penalty = -0.3
        else:
            penalty = 0.0
        nunique = int(s.nunique(dropna=True))
        nonnull = n - int(s.isna().sum())
        uniq_ratio = float(nunique / max(1, nonnull))
        # base score = uniqueness
        score = uniq_ratio
        score += _name_boost(c) + _dtype_boost(s) + penalty
        if _is_texty(s): score -= 0.2
        if _time_like(s, c) and not allow_time: score -= 0.25
        scored.append({"column": c, "uniqueness": uniq_ratio, "null_rate": null_rate, "score": round(score, 6)})

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Take top-K as pool for combos
    pool = [x["column"] for x in scored[:PK_TOPK_UNI_COLS]]

    tested: dict[str, list] = {"single": [], "pairs": [], "triples": []}

    # 2) try singles (best-first)
    for col in pool:
        s = df[col]
        nn_mask = ~s.isna()
        uniq = float(s[nn_mask].nunique(dropna=True) / max(1, nn_mask.sum()))
        tested["single"].append({"cols":[col], "uniqueness": uniq})
        used_time = _time_like(s, col)
        if uniq >= 0.999 and (allow_time or not used_time):
            return {
                "primary_key": [col],
                "uniqueness": uniq,
                "null_rows": int((~nn_mask).sum()),
                "used_time_col": used_time,
                "candidates_scored": scored,
                "tested": tested
            }

    # 3) try pairs
    if max_width >= 2:
        for a, b in itertools.combinations(pool, 2):
            nn_mask = ~df[[a, b]].isna().any(axis=1)
            uniq = float(df.loc[nn_mask, [a, b]].drop_duplicates().shape[0] / max(1, nn_mask.sum()))
            tested["pairs"].append({"cols":[a,b], "uniqueness": uniq})
            used_time = _time_like(df[a], a) or _time_like(df[b], b)
            if uniq >= 0.999 and (allow_time or not used_time):
                return {
                    "primary_key": [a, b],
                    "uniqueness": uniq,
                    "null_rows": int((~nn_mask).sum()),
                    "used_time_col": used_time,
                    "candidates_scored": scored,
                    "tested": tested
                }

    # 4) try triples
    if max_width >= 3:
        for a, b, c in itertools.combinations(pool, 3):
            nn_mask = ~df[[a, b, c]].isna().any(axis=1)
            uniq = float(df.loc[nn_mask, [a, b, c]].drop_duplicates().shape[0] / max(1, nn_mask.sum()))
            tested["triples"].append({"cols":[a,b,c], "uniqueness": uniq})
            used_time = any(_time_like(df[x], x) for x in (a,b,c))
            if uniq >= 0.999 and (allow_time or not used_time):
                return {
                    "primary_key": [a, b, c],
                    "uniqueness": uniq,
                    "null_rows": int((~nn_mask).sum()),
                    "used_time_col": used_time,
                    "candidates_scored": scored,
                    "tested": tested
                }

    # 5) fallback: best single by score (even if not perfectly unique)
    best = scored[0] if scored else None
    if best:
        col = best["column"]
        s = df[col]
        nn_mask = ~s.isna()
        uniq = float(s[nn_mask].nunique(dropna=True) / max(1, nn_mask.sum()))
        return {
            "primary_key": [col] if col in df.columns else [],
            "uniqueness": uniq,
            "null_rows": int((~nn_mask).sum()),
            "used_time_col": _time_like(s, col),
            "candidates_scored": scored,
            "tested": tested
        }

    return {"primary_key": [], "uniqueness": 0.0, "null_rows": 0, "used_time_col": False,
            "candidates_scored": [], "tested": tested}
