from __future__ import annotations
# dq_local_beam.py
# 本地（DirectRunner）CSV 数据质量校验：
# - 规则驱动（YAML）：必备列/数值/范围/枚举/PK/外键/新鲜度
# - 支持带单位数值（如 ram_limit=2048M），解析后进行校验与统计
# - 统计阶段按“每条记录匹配到的规则”解析，避免统计为空
# - 输出：
#     GOOD -> CSV
#     BAD  -> JSONL（含 file/row/reason）
#     DQ   -> 每文件计数、数值分布、表头 union/intersection

import os
import io
import csv
import re
import json
import html
import datetime as dt
import math
import shutil
import fnmatch
import statistics
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Iterable, Set, Sequence
from string import Template

import glob

from augmentation import AUGMENTATION_STRATEGIES, generate_augmented_dataset
from staleness import staleness_score

try:
    import plotly.graph_objects as go  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    go = None  # type: ignore

try:
    import apache_beam as beam  # type: ignore
    from apache_beam.options.pipeline_options import PipelineOptions
    from apache_beam.io import fileio
    from apache_beam.metrics.metric import Metrics
    try:
        from apache_beam.transforms.combiners import ApproximateQuantiles
    except ImportError:
        try:
            from apache_beam.transforms.combiners.approximate_quantiles import (  # type: ignore
                ApproximateQuantiles,
            )
        except ImportError:  # apache-beam>=2.66 relocated the combiner symbol
            from apache_beam.transforms import combiners as _beam_combiners

            ApproximateQuantiles = getattr(  # type: ignore[assignment]
                _beam_combiners,
                "ApproximateQuantiles",
                None,
            )
            if ApproximateQuantiles is None:
                raise

    _HAVE_BEAM = True
except ModuleNotFoundError:
    beam = None  # type: ignore
    PipelineOptions = None  # type: ignore
    fileio = None  # type: ignore

    class _DummyCounter:
        def inc(self, unused_amount: int = 1) -> None:
            return None

    class _DummyMetrics:
        @staticmethod
        def counter(*_args, **_kwargs):
            return _DummyCounter()

    Metrics = _DummyMetrics()  # type: ignore

    class _DummyApproximateQuantiles:
        @staticmethod
        def Globally(*_args, **_kwargs):
            raise RuntimeError("ApproximateQuantiles requires apache_beam. Install apache-beam or use the sequential engine.")

    ApproximateQuantiles = _DummyApproximateQuantiles()  # type: ignore

    _HAVE_BEAM = False

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None  # type: ignore
try:
    from dateutil import parser as dtparse  # type: ignore
except ModuleNotFoundError:
    dtparse = None  # type: ignore

BAD_TAG = "bad"
HEADERS_TAG = "headers"

DIMENSION_DESCRIPTIONS = {
    "Accuracy": (
        "Data are accurate when data values stored in the database correspond to real-world values."
    ),
    "Completeness": (
        "The ability of an information system to represent every meaningful state of the represented real-world system."
    ),
    "Staleness": (
        "The extent to which the age of the data is appropriate for the task at hand."
    ),
    "Consistency": (
        "The extent to which data is presented in the same format and compatible with previous data while respecting semantic rules."
    ),
    "Duplication": (
        "A measure of unwanted duplication existing within or across systems for a particular field, record, or data set."
    ),
    "Other": "Issues that do not map to a predefined data-quality dimension.",
}


STALENESS_SCORE_DEFAULT_COLUMN = "staleness_score"
_NUMERIC_INT_RE = re.compile(r"^[-+]?\d+$")
_NUMERIC_FLOAT_RE = re.compile(r"^[-+]?\d+\.\d+$")
_DEEPSENSE_SCEN1_TIME_RE = re.compile(
    r"^(?P<hour>\d{1,2})\D(?P<minute>\d{1,2})\D(?P<second>\d{1,2})(?:\D(?P<fraction>\d+))?$"
)


def _canonicalize_column_name(name: Optional[str]) -> str:
    """Normalize a column identifier for case-insensitive comparisons."""

    if name is None:
        return ""
    text = str(name).strip().lower()
    if not text:
        return ""
    return re.sub(r"\s+", " ", text)


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return True
        if stripped.lower() == "nan":
            return True
    return False


def _resolve_column_from_header(
    header: Sequence[str], alias_map: Dict[str, str], column_name: Optional[str]
) -> Optional[str]:
    if not column_name:
        return None
    canon = _canonicalize_column_name(column_name)
    actual = alias_map.get(canon)
    if actual:
        return actual
    if column_name in header:
        return column_name
    for candidate in header:
        if _canonicalize_column_name(candidate) == canon:
            return candidate
    return None


def _format_numeric_like(original: Any, value: float) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    text = "" if original is None else str(original).strip()
    if not text:
        formatted = f"{value:.6f}"
    elif _NUMERIC_INT_RE.match(text):
        formatted = str(int(round(value)))
    elif _NUMERIC_FLOAT_RE.match(text):
        decimals = len(text.split(".", 1)[1])
        formatted = f"{value:.{decimals}f}"
    else:
        formatted = f"{value:.6f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    if formatted in {"-0", "-0.0"}:
        formatted = "0"
    return formatted


def _format_event_time_value(timestamp: dt.datetime, fmt: str, original_value: Any) -> str:
    ts_utc = timestamp.astimezone(dt.timezone.utc)
    fmt_lower = (fmt or "auto").lower()
    if fmt_lower == "epoch_s":
        return str(int(round(ts_utc.timestamp())))
    if fmt_lower == "epoch_ms":
        return str(int(round(ts_utc.timestamp() * 1000)))
    iso_value = ts_utc.isoformat()
    if iso_value.endswith("+00:00"):
        iso_value = iso_value[:-6] + "Z"
    if fmt_lower == "deepsense_scen1":
        return iso_value
    if fmt_lower == "iso":
        return iso_value
    text = "" if original_value is None else str(original_value).strip()
    if not text:
        return iso_value
    try:
        base = float(text)
    except ValueError:
        return iso_value
    magnitude = abs(base)
    if magnitude >= 1e18:
        return str(int(round(ts_utc.timestamp() * 1_000_000_000)))
    if magnitude >= 1e15:
        return str(int(round(ts_utc.timestamp() * 1_000_000)))
    if magnitude >= 1e12:
        return str(int(round(ts_utc.timestamp() * 1_000)))
    if '.' in text:
        decimals = text.split('.', 1)[1]
        trimmed = decimals.rstrip('0')
        precision = len(trimmed) if trimmed else len(decimals)
        formatted = f"{ts_utc.timestamp():.{precision}f}"
        formatted = formatted.rstrip('0').rstrip('.')
        return formatted or '0'
    if 'e' in text.lower():
        formatted = f"{ts_utc.timestamp():.6f}"
        return formatted.rstrip('0').rstrip('.')
    return str(int(round(ts_utc.timestamp())))


def _shift_staleness_time_columns(
    row: Dict[str, Any],
    header: Sequence[str],
    alias_map: Dict[str, str],
    staleness_spec: Dict[str, Any],
    delta_seconds: Optional[float],
) -> None:
    if not staleness_spec or delta_seconds is None:
        return
    for key, fmt_key in (("input_time_col", "input_time_format"), ("delivery_time_col", "delivery_time_format")):
        col_name = staleness_spec.get(key)
        if not col_name:
            continue
        actual_col = _resolve_column_from_header(header, alias_map, col_name)
        if not actual_col:
            continue
        raw_value = row.get(actual_col)
        if _is_missing_value(raw_value):
            continue
        fmt = (staleness_spec.get(fmt_key) or "auto").lower()
        try:
            base_ts = parse_event_time(raw_value, fmt)
        except Exception:
            continue
        shifted = base_ts + dt.timedelta(seconds=delta_seconds)
        row[actual_col] = _format_event_time_value(shifted, fmt, raw_value)


def _maybe_attach_staleness_metric(
    rc: RowCtx,
    rule: Dict[str, Any],
    alias_map: Dict[str, str],
    event_time_col: Optional[str],
    event_time_format: Optional[str],
    reference_time: Optional[dt.datetime],
    event_timestamp: Optional[dt.datetime] = None,
) -> Optional[Tuple[str, float]]:
    staleness_spec = rule.get("staleness") or {}
    if not staleness_spec and rule.get("freshness_slo_hours") is None:
        return None
    score_col = staleness_spec.get("score_column") or STALENESS_SCORE_DEFAULT_COLUMN
    computed_metrics = getattr(rc, "_dq_computed_metrics", None)
    if computed_metrics and score_col in computed_metrics:
        return score_col, computed_metrics[score_col]
    header = rc.header or []
    fmt = (event_time_format or rule.get("event_time_format") or "auto").lower()
    actual_event_col = event_time_col
    if not actual_event_col:
        actual_event_col = _resolve_column_from_header(header, alias_map, rule.get("event_time_col"))
    if not actual_event_col:
        return None
    raw_event = rc.data.get(actual_event_col)
    if _is_missing_value(raw_event):
        return None
    try:
        event_ts = event_timestamp or parse_event_time(raw_event, fmt)
    except Exception:
        return None
    now = reference_time or dt.datetime.now(dt.timezone.utc)
    input_time = event_ts
    input_col = staleness_spec.get("input_time_col")
    if input_col:
        actual_input_col = _resolve_column_from_header(header, alias_map, input_col)
        if actual_input_col:
            raw_input = rc.data.get(actual_input_col)
            if raw_input is not None and not _is_missing_value(raw_input):
                try:
                    input_fmt = (staleness_spec.get("input_time_format") or "auto").lower()
                    input_time = parse_event_time(raw_input, input_fmt)
                except Exception:
                    input_time = event_ts
    delivery_time = now
    delivery_col = staleness_spec.get("delivery_time_col")
    if delivery_col:
        actual_delivery_col = _resolve_column_from_header(header, alias_map, delivery_col)
        if actual_delivery_col:
            raw_delivery = rc.data.get(actual_delivery_col)
            if raw_delivery is not None and not _is_missing_value(raw_delivery):
                try:
                    delivery_fmt = (staleness_spec.get("delivery_time_format") or "auto").lower()
                    delivery_time = parse_event_time(raw_delivery, delivery_fmt)
                except Exception:
                    delivery_time = now
    age_seconds: Optional[float] = None
    age_col = staleness_spec.get("age_col")
    if age_col:
        actual_age_col = _resolve_column_from_header(header, alias_map, age_col)
        if actual_age_col:
            raw_age = rc.data.get(actual_age_col)
            if raw_age is not None and not _is_missing_value(raw_age):
                try:
                    age_value = float(raw_age)
                except Exception:
                    age_value = None
                if age_value is not None:
                    unit = (staleness_spec.get("age_unit") or "seconds").lower()
                    if unit == "hours":
                        age_seconds = age_value * 3600.0
                    elif unit == "minutes":
                        age_seconds = age_value * 60.0
                    else:
                        age_seconds = age_value
    if age_seconds is None:
        try:
            age_seconds = max(0.0, (input_time - event_ts).total_seconds())
        except Exception:
            age_seconds = None
    duration_seconds: Optional[float] = None
    duration_col = staleness_spec.get("validity_duration_col")
    if duration_col:
        actual_duration_col = _resolve_column_from_header(header, alias_map, duration_col)
        if actual_duration_col:
            raw_duration = rc.data.get(actual_duration_col)
            if raw_duration is not None and not _is_missing_value(raw_duration):
                try:
                    duration_value = float(raw_duration)
                except Exception:
                    duration_value = None
                if duration_value is not None:
                    unit = (staleness_spec.get("validity_duration_unit") or "hours").lower()
                    if unit == "seconds":
                        duration_seconds = duration_value
                    elif unit == "minutes":
                        duration_seconds = duration_value * 60.0
                    else:
                        duration_seconds = duration_value * 3600.0
    if duration_seconds is None:
        validity_hours = staleness_spec.get("validity_duration_hours")
        if validity_hours is None:
            validity_hours = rule.get("freshness_slo_hours")
        if validity_hours is not None:
            duration_seconds = float(validity_hours) * 3600.0
    if age_seconds is None or duration_seconds is None or duration_seconds <= 0:
        return None
    try:
        score = staleness_score(
            age=dt.timedelta(seconds=age_seconds),
            delivery_time=delivery_time,
            input_time=input_time,
            validity_duration=dt.timedelta(seconds=duration_seconds),
        )
    except Exception:
        return None
    if computed_metrics is None:
        computed_metrics = {}
        setattr(rc, "_dq_computed_metrics", computed_metrics)
    computed_metrics[score_col] = score
    return score_col, score


def complete_issue_dimensions(issue_summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure that all predefined data quality dimensions are represented."""

    summary_by_dim: Dict[str, Dict[str, Any]] = {}
    for entry in issue_summary:
        dim = entry.get("dimension")
        if not dim:
            continue
        summary = dict(entry)
        summary.setdefault("description", DIMENSION_DESCRIPTIONS.get(dim, DIMENSION_DESCRIPTIONS["Other"]))
        summary.setdefault("issue_count", 0)
        summary.setdefault("scenarios", [])
        summary.setdefault("examples", [])
        summary_by_dim[dim] = summary

    for dim, desc in DIMENSION_DESCRIPTIONS.items():
        if dim not in summary_by_dim:
            summary_by_dim[dim] = {
                "dimension": dim,
                "description": desc,
                "issue_count": 0,
                "scenarios": [],
                "examples": [],
            }

    return [summary_by_dim[dim] for dim in sorted(summary_by_dim.keys())]

# -----------------------------
# 数据结构
# -----------------------------
@dataclass
class RowCtx:
    file: str
    rownum: int
    header: List[str]
    data: Dict[str, Any]

# -----------------------------
# 规则配置
# -----------------------------
DEFAULT_RULE = {
    "patterns": [".*"],
    "required_cols": [],
    "required_col_prefixes": [],
    "numeric_cols": [],
    "enums": {},
    "ranges": {},
    "primary_key": None,
    "event_time_col": None,
    "event_time_format": "auto",     # iso | epoch_s | epoch_ms | deepsense_scen1 | auto
    "freshness_slo_hours": None,
    "max_future_hours": None,
    "time_epoch_bounds": None,       # {"min": <epoch_s>, "max": <epoch_s>}
    "reference_keys": None,          # legacy dict or list of {"path": ..., "column": ..., "target_col": ...}
    "numeric_unit_parsers": {},      # 列 -> {type:'mem', base:1024, out_unit:'MiB'}
    "metadata_path": None,
    "metadata_targets": [],          # 文件名（或相对路径）列表，用于从元数据中提取列说明
    "metadata_base_dir": None,
}

PRIMARY_KEY_FILE_TOKEN = "__file__"
PRIMARY_KEY_ROWNUM_TOKEN = "__rownum__"


def _normalize_primary_key_columns(value: Any) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        cols: List[str] = []
        for col in value:
            if col is None:
                continue
            if isinstance(col, str):
                cols.append(col)
            else:
                cols.append(str(col))
        return tuple(cols)
    return (str(value),)


def _get_primary_key_columns(rule: Dict[str, Any]) -> Tuple[str, ...]:
    cached = rule.get("_primary_key_columns")
    if isinstance(cached, tuple):
        return cached
    cols = _normalize_primary_key_columns(rule.get("primary_key"))
    rule["_primary_key_columns"] = cols
    return cols


def _primary_key_value(rule: Dict[str, Any], rc: RowCtx) -> Optional[str]:
    columns = _get_primary_key_columns(rule)
    if not columns:
        return None
    values: List[str] = []
    alias = getattr(rc, "_dq_header_alias", None)
    for col in columns:
        if col == PRIMARY_KEY_FILE_TOKEN:
            values.append(rc.file)
            continue
        if col == PRIMARY_KEY_ROWNUM_TOKEN:
            values.append(str(rc.rownum))
            continue
        actual_col = None
        if isinstance(alias, dict):
            actual_col = alias.get(_canonicalize_column_name(col))
        if not actual_col and col in rc.header:
            actual_col = col
        if not actual_col:
            return None
        values.append(rc.data.get(actual_col))
    normalized: List[str] = []
    for val in values:
        if _is_missing_value(val):
            return None
        normalized.append(str(val))
    if len(normalized) == 1:
        return normalized[0]
    return json.dumps(normalized, ensure_ascii=False)

_VISUALIZATION_FILE_LIMIT_ENV = "DQ_VISUALIZATION_MAX_FILE_BYTES"
_VISUALIZATION_TOTAL_LIMIT_ENV = "DQ_VISUALIZATION_MAX_TOTAL_BYTES"
_DEFAULT_VISUALIZATION_MAX_FILE_BYTES = 200 * 1024 * 1024  # 200 MiB
_DEFAULT_VISUALIZATION_MAX_TOTAL_BYTES = 500 * 1024 * 1024  # 500 MiB


def _parse_size_limit(raw_value: Optional[str], default: Optional[int]) -> Optional[int]:
    if raw_value is None:
        return default
    text = raw_value.strip()
    if not text:
        return None
    multiplier = 1
    suffix_multipliers = {
        "k": 1024,
        "m": 1024 ** 2,
        "g": 1024 ** 3,
        "t": 1024 ** 4,
    }
    suffix = text[-1].lower()
    if suffix in suffix_multipliers and text[:-1].strip():
        multiplier = suffix_multipliers[suffix]
        text = text[:-1]
    try:
        value = float(text)
    except ValueError:
        return default
    limit = int(value * multiplier)
    if limit <= 0:
        return None
    return limit


def _format_size(num_bytes: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def _visualizations_allowed(file_paths: Iterable[str]) -> Tuple[bool, Optional[str]]:
    file_limit = _parse_size_limit(
        os.environ.get(_VISUALIZATION_FILE_LIMIT_ENV),
        _DEFAULT_VISUALIZATION_MAX_FILE_BYTES,
    )
    total_limit = _parse_size_limit(
        os.environ.get(_VISUALIZATION_TOTAL_LIMIT_ENV),
        _DEFAULT_VISUALIZATION_MAX_TOTAL_BYTES,
    )

    total_size = 0
    largest_size = 0
    largest_path: Optional[str] = None

    for path in file_paths:
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        total_size += size
        if size > largest_size:
            largest_size = size
            largest_path = path

    if file_limit is not None and largest_path and largest_size > file_limit:
        note = (
            "Visualizations skipped because "
            f"{os.path.basename(largest_path)} is {_format_size(largest_size)} "
            f"which exceeds the per-file limit of {_format_size(file_limit)}."
        )
        return False, note

    if total_limit is not None and total_size > total_limit:
        note = (
            "Visualizations skipped because total input size "
            f"{_format_size(total_size)} exceeds the limit of {_format_size(total_limit)}."
        )
        return False, note

    return True, None

# -----------------------------
# 元数据解析
# -----------------------------
_COLUMN_NAME_RE = re.compile(r"^[A-Za-z0-9_ ]+$")


def _parse_metadata_text(lines: Iterable[str]) -> Dict[str, Dict[str, str]]:
    """解析元数据文本，返回 {文件名或模式: {列名: 描述}}。列名统一为小写。"""

    mapping: Dict[str, Dict[str, str]] = {}
    current_keys: List[str] = []
    current_key_set: Set[str] = set()

    def _register_literal(target: str) -> None:
        normalized = target.strip().replace("\\", "/").lower()
        if not normalized:
            return
        group = mapping.setdefault(normalized, {})
        if normalized not in current_key_set:
            current_keys.append(normalized)
            current_key_set.add(normalized)
        base = os.path.basename(normalized)
        if base and base not in mapping:
            mapping[base] = group
        if base and base not in current_key_set:
            current_keys.append(base)
            current_key_set.add(base)

    def _register_pattern(target: str) -> None:
        normalized = target.strip().replace("\\", "/").lower()
        if not normalized:
            return
        pattern_key = f"__pattern__:{normalized}"
        group = mapping.setdefault(pattern_key, {})
        if pattern_key not in current_key_set:
            current_keys.append(pattern_key)
            current_key_set.add(pattern_key)
        base = os.path.basename(normalized)
        if base:
            base_key = f"__pattern__:{base}"
            if base_key not in mapping:
                mapping[base_key] = group
            if base_key not in current_key_set:
                current_keys.append(base_key)
                current_key_set.add(base_key)

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        lower_key = key.lower()

        if lower_key in {"file", "files"}:
            current_keys = []
            current_key_set = set()
            parts = [part for part in (p.strip() for p in value.split(",")) if part]
            for part in parts:
                _register_literal(part)
            continue

        if lower_key in {"file_pattern", "files_pattern", "pattern", "patterns"}:
            current_keys = []
            current_key_set = set()
            parts = [part for part in (p.strip() for p in value.split(",")) if part]
            for part in parts:
                _register_pattern(part)
            continue

        if not current_keys:
            # 还未遇到 file(s) / pattern(s) 说明，此处多为段落描述
            continue

        if not value:
            # 章节标题或其他说明
            continue

        if not _COLUMN_NAME_RE.match(key):
            # 过滤掉“非列名”的行（例如长句子）
            continue

        col_name = key.strip().lower()
        for tgt in current_keys:
            mapping.setdefault(tgt, {})[col_name] = value

    return mapping


def parse_metadata_descriptions(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as fp:
        return _parse_metadata_text(fp.readlines())


def _split_inline_items(text: str) -> List[str]:
    items: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in text:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            items.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        items.append("".join(buf).strip())
    return [item for item in items if item]


def _parse_simple_yaml_value(token: str) -> Any:
    if token == "":
        return None
    lower = token.lower()
    if lower in {"null", "none"}:
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_parse_simple_yaml_value(part) for part in _split_inline_items(inner)]
    if token.startswith("{") and token.endswith("}"):
        inner = token[1:-1].strip()
        result: Dict[str, Any] = {}
        if inner:
            for part in _split_inline_items(inner):
                if ":" not in part:
                    raise ValueError(f"Invalid inline mapping segment: {part}")
                key, value = part.split(":", 1)
                result[key.strip()] = _parse_simple_yaml_value(value.strip())
        return result
    if token.startswith("\"") and token.endswith("\""):
        return json.loads(token)
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    try:
        if any(c in token for c in ".eE"):
            return float(token)
        return int(token)
    except ValueError:
        return token


def _simple_yaml_parse(lines: List[str], start: int, indent: int) -> Tuple[Any, int]:
    result: Any = None
    idx = start
    while idx < len(lines):
        raw = lines[idx]
        if not raw.strip():
            idx += 1
            continue
        current_indent = len(raw) - len(raw.lstrip(" "))
        if current_indent < indent:
            break
        stripped = raw.strip()
        if stripped.startswith("- "):
            if result is None:
                result = []
            elif not isinstance(result, list):
                raise ValueError("Mixed list/dict structures are not supported in simple YAML parser")
            value_part = stripped[2:].strip()
            if value_part:
                if ":" in value_part and not value_part.startswith("{"):
                    key, remainder = value_part.split(":", 1)
                    child_dict: Dict[str, Any] = {}
                    remainder = remainder.strip()
                    if remainder:
                        child_dict[key.strip()] = _parse_simple_yaml_value(remainder)
                        idx += 1
                    else:
                        child_value, idx = _simple_yaml_parse(
                            lines, idx + 1, current_indent + 2
                        )
                        child_dict[key.strip()] = child_value

                    # merge additional nested keys at deeper indent
                    next_idx = idx
                    while next_idx < len(lines) and not lines[next_idx].strip():
                        next_idx += 1
                    if next_idx < len(lines):
                        next_indent = len(lines[next_idx]) - len(lines[next_idx].lstrip(" "))
                    else:
                        next_indent = None
                    if next_indent is not None and next_indent >= current_indent + 2:
                        child_rest, idx = _simple_yaml_parse(lines, idx, current_indent + 2)
                        if isinstance(child_rest, dict):
                            child_dict.update(child_rest)
                    result.append(child_dict)
                else:
                    result.append(_parse_simple_yaml_value(value_part))
                    idx += 1
            else:
                child, idx = _simple_yaml_parse(lines, idx + 1, current_indent + 2)
                result.append(child)
        else:
            if ":" not in stripped:
                raise ValueError(f"Invalid YAML mapping line: {stripped}")
            key, remainder = stripped.split(":", 1)
            key = key.strip()
            remainder = remainder.strip()
            if result is None:
                result = {}
            elif not isinstance(result, dict):
                raise ValueError("Mixed list/dict structures are not supported in simple YAML parser")
            if remainder:
                result[key] = _parse_simple_yaml_value(remainder)
                idx += 1
            else:
                child, idx = _simple_yaml_parse(lines, idx + 1, current_indent + 2)
                result[key] = child
    if result is None:
        return {}, idx
    return result, idx


def _simple_yaml_load(text: str) -> Any:
    lines = [line.rstrip("\n") for line in text.splitlines()]
    parsed, _ = _simple_yaml_parse(lines, 0, 0)
    return parsed


def _load_config_text(text: str) -> Any:
    if yaml is not None:
        return yaml.safe_load(text)
    return _simple_yaml_load(text)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = _load_config_text(f.read())
    if isinstance(cfg, dict) and "rules" in cfg:
        rules = cfg["rules"]
    elif isinstance(cfg, list):
        rules = cfg
    else:
        rules = [cfg]
    norm: List[Dict[str, Any]] = []
    metadata_cache: Dict[str, Dict[str, Dict[str, str]]] = {}
    reference_cache: Dict[Tuple[str, str], set] = {}
    for r in rules:
        rr = dict(DEFAULT_RULE)
        if r:
            rr.update(r)
        pats = rr.get("patterns") or rr.get("pattern") or [".*"]
        rr["patterns"] = [pats] if isinstance(pats, str) else list(pats)
        rr["required_cols"] = rr.get("required_cols", []) or []
        rr["required_col_prefixes"] = rr.get("required_col_prefixes", []) or []
        rr["numeric_cols"]  = rr.get("numeric_cols", [])  or []
        rr["enums"]         = rr.get("enums", {})         or {}
        rr["ranges"]        = rr.get("ranges", {})        or {}
        rr["numeric_unit_parsers"] = rr.get("numeric_unit_parsers", {}) or {}

        meta_path = rr.get("metadata_path")
        targets = rr.get("metadata_targets") or []
        metadata_by_file: Dict[str, Dict[str, str]] = {}
        metadata_base_dir: Optional[str] = None
        if meta_path:
            if meta_path not in metadata_cache:
                metadata_cache[meta_path] = parse_metadata_descriptions(meta_path)
            meta_map = metadata_cache[meta_path]
            metadata_base_dir = os.path.abspath(os.path.dirname(meta_path) or ".")
            selected = targets or list(meta_map.keys())
            for target in selected:
                if not target:
                    continue
                if target.startswith("__pattern__:") and target in meta_map:
                    metadata_by_file[target] = meta_map[target]
                    continue
                normalized = target.strip().replace("\\", "/").lower()
                candidates: List[str] = []
                if normalized:
                    candidates.append(normalized)
                if metadata_base_dir:
                    try:
                        rel = os.path.relpath(os.path.normpath(target), metadata_base_dir)
                        rel_norm = rel.replace("\\", "/").lower()
                        if rel_norm:
                            candidates.append(rel_norm)
                    except ValueError:
                        pass
                base = os.path.basename(normalized)
                if base:
                    candidates.append(base)
                for cand in candidates:
                    if not cand:
                        continue
                    pattern_key = f"__pattern__:{cand}"
                    if cand in meta_map:
                        metadata_by_file[cand] = meta_map[cand]
                    if pattern_key in meta_map:
                        metadata_by_file[pattern_key] = meta_map[pattern_key]
            if not targets:
                for key, cols in meta_map.items():
                    if key not in metadata_by_file:
                        metadata_by_file[key] = cols
        if metadata_by_file:
            rr["metadata_by_file"] = metadata_by_file
            rr["metadata_base_dir"] = metadata_base_dir
            # 将元数据列合并进 required_cols，避免遗漏
            meta_cols = sorted({col for cols in metadata_by_file.values() for col in cols.keys()})
            for col in meta_cols:
                if col not in rr["required_cols"]:
                    rr["required_cols"].append(col)
        else:
            rr["metadata_by_file"] = {}
            rr["metadata_base_dir"] = metadata_base_dir
        norm.append(rr)
    return {"rules": norm}


def _resolve_metadata_columns(rule: Dict[str, Any], path: str) -> Optional[Dict[str, str]]:
    metadata_by_file: Dict[str, Dict[str, str]] = rule.get("metadata_by_file") or {}
    if not metadata_by_file:
        return None

    base_dir = rule.get("metadata_base_dir")
    candidates: List[str] = []

    norm_path = path.replace("\\", "/").lower()
    candidates.append(norm_path)

    if base_dir:
        try:
            rel = os.path.relpath(path, base_dir)
            rel_norm = rel.replace("\\", "/").lower()
            candidates.append(rel_norm)
        except ValueError:
            pass

    base = os.path.basename(norm_path)
    if base:
        candidates.append(base)

    seen: Set[str] = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        cols = metadata_by_file.get(cand)
        if cols:
            return cols

    for key, cols in metadata_by_file.items():
        if not key.startswith("__pattern__:"):
            continue
        pattern = key.split(":", 1)[1]
        for cand in candidates:
            if cand and fnmatch.fnmatch(cand, pattern):
                return cols

    return None


def _prepare_header_alias(header: List[str]) -> Tuple[Dict[str, str], List[str]]:
    alias: Dict[str, str] = {}
    canonical: List[str] = []
    for raw in header:
        cleaned = (raw or "").strip()
        canon = _canonicalize_column_name(cleaned)
        canonical.append(canon)
        if canon not in alias:
            alias[canon] = cleaned
    return alias, canonical


def _ensure_header_state(rule: Dict[str, Any], path: str, header: List[str]) -> Dict[str, Any]:
    cache: Dict[str, Dict[str, Any]] = rule.setdefault("_header_cache", {})
    cached = cache.get(path)
    if cached is not None:
        return cached

    alias, canonical_list = _prepare_header_alias(header)
    canonical_set = set(canonical_list)

    event_time_actual: Optional[str] = None
    et_col = rule.get("event_time_col")
    if et_col:
        canon_et = _canonicalize_column_name(et_col)
        event_time_actual = alias.get(canon_et)
        if not event_time_actual and et_col in header:
            event_time_actual = et_col

    if not header:
        state = {
            "valid": False,
            "issues": [],
            "alias": alias,
            "metadata_columns": {},
            "metadata_canonical": {},
            "reported": False,
            "canonical_header": canonical_list,
            "event_time_actual": event_time_actual,
        }
        cache[path] = state
        return state

    metadata_cols = _resolve_metadata_columns(rule, path) or {}
    metadata_canonical = {
        _canonicalize_column_name(col): col for col in metadata_cols.keys()
    }

    issues: List[Dict[str, Any]] = []

    for canon, original in metadata_canonical.items():
        if not canon:
            continue
        if canon not in canonical_set:
            issues.append({"file": path, "row": 0, "reason": f"missing_metadata_column:{original}"})

    for col in rule.get("required_cols", []):
        canon = _canonicalize_column_name(col)
        if not canon:
            continue
        if canon in metadata_canonical:
            continue
        if canon not in canonical_set:
            issues.append({"file": path, "row": 0, "reason": f"missing_required_column:{col}"})

    canonical_header_for_prefix = [c for c in canonical_list if c]
    for prefix in rule.get("required_col_prefixes", []):
        prefix_canon = _canonicalize_column_name(prefix)
        if not prefix_canon:
            continue
        if not any(col.startswith(prefix_canon) for col in canonical_header_for_prefix):
            issues.append({"file": path, "row": 0, "reason": f"missing_required_prefix:{prefix}"})

    state = {
        "valid": not issues,
        "issues": issues,
        "alias": alias,
        "metadata_columns": metadata_cols,
        "metadata_canonical": metadata_canonical,
        "reported": False,
        "canonical_header": canonical_list,
        "event_time_actual": event_time_actual,
    }
    cache[path] = state
    return state


def pick_rule(rules: List[Dict[str, Any]], path: str) -> Dict[str, Any]:
    for r in rules:
        if any(re.search(p, path) for p in r["patterns"]):
            return r
    return DEFAULT_RULE

# -----------------------------
# 时间解析
# -----------------------------
def _strip_brackets(value: str) -> str:
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if text and text[0] in {'"', "'"} and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text


def _parse_deepsense_scen1_time(value: Any) -> dt.datetime:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError("empty timestamp")
    text = _strip_brackets(text)
    match = _DEEPSENSE_SCEN1_TIME_RE.match(text)
    if not match:
        raise ValueError(f"invalid deepsense_scen1 timestamp: {value!r}")
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second"))
    fraction = match.group("fraction") or "0"
    fraction = (fraction + "000000")[:6]
    microsecond = int(fraction)
    base = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    return base.replace(hour=hour, minute=minute, second=second, microsecond=microsecond)


def parse_event_time(val, fmt: str = "auto") -> dt.datetime:
    if val is None or str(val).strip() == "":
        raise ValueError("empty timestamp")

    fmt_lower = (fmt or "auto").lower()
    if fmt_lower == "deepsense_scen1":
        return _parse_deepsense_scen1_time(val)

    x = 0.0
    try:
        x = float(val)
        is_numeric = True
    except Exception:
        is_numeric = False

    if fmt_lower == "iso" or (fmt_lower == "auto" and not is_numeric):
        return _parse_iso_datetime(str(val))

    if fmt_lower == "epoch_s":
        seconds = x
    elif fmt_lower == "epoch_ms":
        seconds = x / 1000.0
    else:
        ax = abs(x)
        if ax >= 1e18:
            seconds = x / 1_000_000_000.0  # ns
        elif ax >= 1e15:
            seconds = x / 1_000_000.0  # µs
        elif ax >= 1e12:
            seconds = x / 1_000.0  # ms
        else:
            seconds = x  # s
    return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)


def _parse_iso_datetime(value: str) -> dt.datetime:
    if dtparse is not None:
        t = dtparse.parse(value)
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        else:
            t = t.astimezone(dt.timezone.utc)
        return t

    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        parsed = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                parsed = dt.datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            raise
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    else:
        parsed = parsed.astimezone(dt.timezone.utc)
    return parsed

# -----------------------------
# 带单位数值解析（返回 float，统一 MiB）
# 支持 mem: "512K"/"2048M"/"2G"/"1T"，base=1024 或 1000
# -----------------------------
_MEM_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KkMmGgTt])?\s*$")

def parse_numeric_with_units(val: Any, col: str, rule: Dict[str, Any]) -> float:
    parsers = rule.get("numeric_unit_parsers") or {}
    spec = parsers.get(col)
    if not spec:
        if val in (None, "", "NaN", "nan"):
            raise ValueError("empty")
        return float(val)

    typ = (spec.get("type") or "").lower()
    if typ != "mem":
        if val in (None, "", "NaN", "nan"):
            raise ValueError("empty")
        return float(val)

    base = int(spec.get("base", 1024))
    s = "" if val is None else str(val).strip()
    m = _MEM_RE.match(s)
    if not m:
        raise ValueError(f"bad_mem_format:{val}")
    num = float(m.group(1))
    suf = m.group(2).upper() if m.group(2) else ""  # 无后缀→按 MiB
    if suf == "K":
        mib = num / base
    elif suf == "M" or suf == "":
        mib = num
    elif suf == "G":
        mib = num * base
    elif suf == "T":
        mib = num * (base ** 2)
    else:
        raise ValueError(f"unsupported_suffix:{suf}")
    return float(mib)

# -----------------------------
# CSV 解析（Beam 与本地模式共享）
# -----------------------------


def _parse_csv_text(path: str, text: str) -> Tuple[List[str], List[RowCtx], List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    if not text.strip():
        issues.append({"file": path, "row": 0, "reason": "empty_file"})
        return [], [], issues

    buf = io.StringIO(text)
    sample = buf.read(2048)
    buf.seek(0)
    try:
        # Restrict sniffed delimiters so that spaces inside column names (e.g. "abc 12")
        # are not misinterpreted as separators.
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except Exception:
        dialect = csv.excel

    reader = csv.DictReader(buf, dialect=dialect)
    if not reader.fieldnames:
        issues.append({"file": path, "row": 0, "reason": "no_header"})
        return [], [], issues

    header = [h.strip() for h in reader.fieldnames]
    rows: List[RowCtx] = []
    for i, row in enumerate(reader, start=1):
        try:
            cleaned = {
                (k.strip() if k else k): (v.strip() if isinstance(v, str) else v)
                for k, v in row.items()
            }
            rows.append(RowCtx(file=path, rownum=i, header=header, data=cleaned))
        except Exception as exc:
            issues.append({
                "file": path,
                "row": i,
                "reason": f"parse_error:{exc}",
            })
    return header, rows, issues


def read_csv_file(path: str) -> Tuple[List[str], List[RowCtx], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as fp:
        text = fp.read()
    return _parse_csv_text(path, text)

# -----------------------------
# 读 CSV（带表头）+ 发 headers
# -----------------------------
if _HAVE_BEAM:

    class ReadCSVWithHeader(beam.DoFn):  # type: ignore[attr-defined]
        bad_parse = Metrics.counter("dq", "bad_parse")
        empty_files = Metrics.counter("dq", "empty_files")

        def process(self, rf: fileio.ReadableFile):  # type: ignore[valid-type]
            path = rf.metadata.path
            header, rows, issues = _parse_csv_text(path, rf.read_utf8())
            if not header and issues:
                for issue in issues:
                    if issue["reason"] == "empty_file":
                        self.empty_files.inc()
                    else:
                        self.bad_parse.inc()
                    yield beam.pvalue.TaggedOutput(BAD_TAG, issue)
                return

            if not header:
                return

            yield beam.pvalue.TaggedOutput(HEADERS_TAG, {"file": path, "header": header})
            for row in rows:
                yield row
            for issue in issues:
                self.bad_parse.inc()
                yield beam.pvalue.TaggedOutput(BAD_TAG, issue)

# -----------------------------
# 校验逻辑（Beam/本地共享）
# -----------------------------


def validate_row_against_rule(
    rc: RowCtx,
    rule: Dict[str, Any],
    refset: Optional[set] = None,
    increment: Optional[Any] = None,
    reference_time: Optional[dt.datetime] = None,
) -> Tuple[Optional[RowCtx], List[Dict[str, Any]]]:
    inc = increment or (lambda _name: None)
    issues: List[Dict[str, Any]] = []
    data = rc.data

    header_state = _ensure_header_state(rule, rc.file, rc.header)
    alias_map: Dict[str, str] = header_state.get("alias") or {}
    setattr(rc, "_dq_header_alias", alias_map)

    if not header_state.get("valid", True):
        if not header_state.get("reported") and header_state.get("issues"):
            header_state["reported"] = True
            for issue in header_state["issues"]:
                reason = issue.get("reason", "")
                if isinstance(reason, str) and reason.startswith("missing_metadata_column"):
                    inc("metadata_mismatch")
            return None, list(header_state["issues"])
        return None, []

    def _resolve_actual_column(col: str) -> Tuple[str, Optional[str]]:
        canon = _canonicalize_column_name(col)
        actual = alias_map.get(canon)
        if not actual and isinstance(col, str) and col in rc.header:
            actual = col
        if not actual:
            for candidate in rc.header:
                if _canonicalize_column_name(candidate) == canon:
                    actual = candidate
                    break
        return canon, actual

    resolved_required: Dict[str, str] = {}
    for col in rule.get("required_cols", []):
        canon, actual_col = _resolve_actual_column(col)
        if not actual_col:
            continue
        resolved_required[canon or col] = actual_col

    for actual_col in resolved_required.values():
        if _is_missing_value(data.get(actual_col)):
            inc("nulls")
            issues.append(
                {
                    "file": rc.file,
                    "row": rc.rownum,
                    "reason": f"null_required:{actual_col}",
                }
            )
            return None, issues

    canonical_ranges = rule.get("_canonical_ranges", {}) or {}
    for col in rule.get("numeric_cols", []):
        canon, actual_col = _resolve_actual_column(col)
        if not actual_col:
            continue
        val = data.get(actual_col, "")
        try:
            parsed_val = parse_numeric_with_units(val, actual_col, rule)
        except Exception:
            inc("bad_numeric")
            issues.append(
                {
                    "file": rc.file,
                    "row": rc.rownum,
                    "reason": f"bad_numeric:{actual_col}={val}",
                }
            )
            return None, issues
        rng = canonical_ranges.get(canon)
        if rng:
            lo, hi = rng
            if (lo is not None and parsed_val < lo) or (hi is not None and parsed_val > hi):
                inc("bad_range")
                issues.append(
                    {
                        "file": rc.file,
                        "row": rc.rownum,
                        "reason": f"out_of_range:{actual_col}={parsed_val} not [{lo},{hi}]",
                    }
                )
                return None, issues

    canonical_enums = rule.get("_canonical_enums", {}) or {}
    for canon_col, allowed in canonical_enums.items():
        actual_col = alias_map.get(canon_col)
        if not actual_col:
            for candidate in rc.header:
                if _canonicalize_column_name(candidate) == canon_col:
                    actual_col = candidate
                    break
        if not actual_col:
            continue
        raw_val = data.get(actual_col)
        val = "" if raw_val is None else raw_val
        if isinstance(val, str):
            candidate = val.strip()
        else:
            candidate = str(val)
        if candidate == "":
            continue
        if candidate not in allowed:
            inc("bad_enum")
            issues.append(
                {
                    "file": rc.file,
                    "row": rc.rownum,
                    "reason": f"bad_enum:{actual_col}={candidate}",
                }
            )
            return None, issues

    ref_entries = rule.get("reference_keys") or []
    if isinstance(ref_entries, dict):
        ref_entries = [ref_entries]
    ref_map: Dict[str, set] = {}
    cached_sets = rule.get("_canonical_reference_sets") or {}
    if isinstance(cached_sets, dict):
        ref_map.update(cached_sets)
    if isinstance(refset, dict):
        for key, values in refset.items():
            ref_map[_canonicalize_column_name(key)] = values
    elif isinstance(refset, set):
        legacy_target = rule.get("_legacy_reference_target")
        if legacy_target:
            ref_map.setdefault(_canonicalize_column_name(legacy_target), refset)
    for entry in ref_entries:
        target_col = entry.get("target_col") if isinstance(entry, dict) else None
        if not target_col:
            continue
        canon_target, actual_col = _resolve_actual_column(target_col)
        if not actual_col:
            continue
        ref_values = ref_map.get(canon_target)
        if ref_values is None:
            continue
        val = data.get(actual_col)
        candidate = ("" if val is None else str(val).strip())
        if candidate not in ref_values:
            inc("bad_ref")
            issues.append(
                {
                    "file": rc.file,
                    "row": rc.rownum,
                    "reason": f"fk_missing:{actual_col}={candidate}",
                }
            )
            return None, issues

    et_col = rule.get("event_time_col")
    slo_h = rule.get("freshness_slo_hours")
    actual_et_col = None
    if et_col:
        _, actual_et_col = _resolve_actual_column(et_col)
    if actual_et_col and (
        slo_h
        or rule.get("max_future_hours")
        or rule.get("time_epoch_bounds")
        or rule.get("staleness")
    ):
        val = data.get(actual_et_col)
        try:
            et_fmt = (rule.get("event_time_format") or "auto").lower()
            ts = parse_event_time(val, et_fmt)
            now = reference_time or dt.datetime.now(dt.timezone.utc)

            if slo_h is not None:
                age_h = (now - ts).total_seconds() / 3600.0
                if age_h > float(slo_h):
                    inc("bad_freshness")
                    issues.append(
                        {
                            "file": rc.file,
                            "row": rc.rownum,
                            "reason": f"stale_event:{actual_et_col} age_h={age_h:.2f} max={slo_h}",
                        }
                    )
                    return None, issues

            mf = rule.get("max_future_hours")
            if mf is not None:
                future_h = (ts - now).total_seconds() / 3600.0
                if future_h > float(mf):
                    inc("bad_freshness")
                    issues.append(
                        {
                            "file": rc.file,
                            "row": rc.rownum,
                            "reason": f"future_event:{actual_et_col} future_h={future_h:.2f} max={mf}",
                        }
                    )
                    return None, issues

            bounds = rule.get("time_epoch_bounds")
            if bounds and str(val).strip() != "":
                try:
                    num_val = float(val)
                    lo = bounds.get("min")
                    hi = bounds.get("max")
                    if (lo is not None and num_val < float(lo)) or (
                        hi is not None and num_val > float(hi)
                    ):
                        inc("bad_range")
                        issues.append(
                            {
                                "file": rc.file,
                                "row": rc.rownum,
                                "reason": f"time_epoch_out_of_range:{val} not [{lo},{hi}]",
                            }
                        )
                        return None, issues
                except Exception:
                    pass

            _maybe_attach_staleness_metric(
                rc,
                rule,
                alias_map,
                actual_et_col,
                rule.get("event_time_format"),
                reference_time,
                event_timestamp=ts,
            )

        except Exception:
            inc("bad_numeric")
            issues.append(
                {
                    "file": rc.file,
                    "row": rc.rownum,
                    "reason": f"bad_timestamp:{actual_et_col}={val}",
                }
            )
            return None, issues

    return rc, issues


if _HAVE_BEAM:

    class ValidateRow(beam.DoFn):  # type: ignore[attr-defined]
        nulls = Metrics.counter("dq", "null_required")
        bad_enum = Metrics.counter("dq", "bad_enum")
        bad_range = Metrics.counter("dq", "bad_range")
        bad_numeric = Metrics.counter("dq", "bad_numeric")
        bad_ref = Metrics.counter("dq", "bad_ref_integrity")
        bad_freshness = Metrics.counter("dq", "bad_freshness")
        metadata_mismatch = Metrics.counter("dq", "metadata_mismatch")

        def __init__(self, rules: Dict[str, Any]):
            self.rules = rules

        def process(self, rc: RowCtx, refset: Optional[set] = None):
            def inc(name: str) -> None:
                counter = getattr(self, name, None)
                if counter is not None:
                    counter.inc()

            valid, issues = validate_row_against_rule(rc, self.rules, refset, inc)
            if valid is not None:
                yield valid
            for issue in issues:
                yield beam.pvalue.TaggedOutput(BAD_TAG, issue)

    class ValidateHeader(beam.DoFn):  # type: ignore[attr-defined]
        metadata_mismatch = Metrics.counter("dq", "metadata_mismatch")

        def __init__(self, rules: List[Dict[str, Any]]):
            self.rules = rules

        def process(self, info: Dict[str, Any]):
            path = info.get("file")
            header = info.get("header") or []
            if not path:
                return
            rule = pick_rule(self.rules, path)
            state = _ensure_header_state(rule, path, header)
            if state.get("issues") and not state.get("reported"):
                state["reported"] = True
                for issue in state["issues"]:
                    reason = issue.get("reason", "")
                    if isinstance(reason, str) and reason.startswith("missing_metadata_column"):
                        self.metadata_mismatch.inc()
                    yield issue

# -----------------------------
# 工具
# -----------------------------
if _HAVE_BEAM:

    class ToCsvLine(beam.DoFn):  # type: ignore[attr-defined]
        def process(self, rc: RowCtx):
            yield ",".join(str(rc.data.get(c, "")) for c in rc.header)

def load_reference_keys(path: str, column: str) -> set:
    with open(path, "r") as f:
        rdr = csv.DictReader(f)
        return { (row.get(column) or "").strip() for row in rdr }

# === 关键修复：按“每条记录对应的规则”做数值统计 ===
def numeric_profiles_from_pairs(pairs_pcoll, col: str):
    if not _HAVE_BEAM:
        raise RuntimeError("numeric_profiles_from_pairs requires apache_beam. Use summarize_numeric_values for the sequential engine.")
    """
    输入: PCollection[(RowCtx, rule_dict)]
    对指定列做统计；解析单位使用每条记录对应的 rule_dict。
    """
    def cast_with_rule(pair):
        rc, rule = pair
        header_state = _ensure_header_state(rule, rc.file, rc.header)
        alias = header_state.get("alias") or {}
        setattr(rc, "_dq_header_alias", alias)
        canon = _canonicalize_column_name(col)
        numeric_cols = {_canonicalize_column_name(c) for c in (rule.get("numeric_cols") or [])}
        if numeric_cols and canon not in numeric_cols:
            return None
        actual_col = alias.get(canon) or (col if col in rc.header else None)
        if not actual_col:
            return None
        v = rc.data.get(actual_col)
        if _is_missing_value(v):
            return None
        try:
            return parse_numeric_with_units(v, actual_col, rule)
        except Exception:
            return None

    vals = (
        pairs_pcoll
        | f"Cast_{col}_by_rule" >> beam.Map(cast_with_rule)
        | f"DropNone_{col}" >> beam.Filter(lambda x: x is not None)
    )
    count = vals | f"Count_{col}" >> beam.combiners.Count.Globally()
    minv = vals | f"Min_{col}" >> beam.CombineGlobally(min).without_defaults()
    maxv = vals | f"Max_{col}" >> beam.CombineGlobally(max).without_defaults()
    mean = vals | f"Mean_{col}" >> beam.combiners.Mean.Globally().without_defaults()
    qtls = (
        vals
        | f"Qtls_{col}" >> ApproximateQuantiles.Globally(num_quantiles=5)
    )

    def _qtls_to_bounds(qs: List[float]) -> Dict[str, Any]:
        if not qs:
            return {
                "quantiles": [],
                "q1": None,
                "q3": None,
                "iqr": None,
                "lower": None,
                "upper": None,
            }
        q1 = qs[1] if len(qs) >= 4 else None
        q3 = qs[3] if len(qs) >= 4 else None
        iqr = (q3 - q1) if (q1 is not None and q3 is not None) else None
        lower = (q1 - 1.5 * iqr) if iqr is not None else None
        upper = (q3 + 1.5 * iqr) if iqr is not None else None
        return {
            "quantiles": qs,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower": lower,
            "upper": upper,
        }

    bounds = qtls | f"Bounds_{col}" >> beam.Map(_qtls_to_bounds)

    default_bounds = {
        "quantiles": [],
        "q1": None,
        "q3": None,
        "iqr": None,
        "lower": None,
        "upper": None,
    }

    outlier_count = (
        vals
        | f"FlagOutliers_{col}" >> beam.Map(
            lambda x, info: 1
            if info and info.get("lower") is not None and (x < info["lower"] or x > info["upper"])
            else 0,
            info=beam.pvalue.AsSingleton(bounds, default_value=default_bounds),
        )
        | f"SumOutliers_{col}" >> beam.CombineGlobally(sum).without_defaults()
    )

    pipeline = pairs_pcoll.pipeline
    profile = (
        pipeline
        | f"BuildProfile_{col}" >> beam.Create([None])
        | beam.Map(
            lambda _, cnt, min_value, max_value, mean_value, info, out_cnt: {
                "count": cnt,
                "min": min_value,
                "max": max_value,
                "mean": mean_value,
                "quantiles": info.get("quantiles") or [],
                "outlier_bounds": info,
                "outlier_count": out_cnt if out_cnt is not None else 0,
            },
            cnt=beam.pvalue.AsSingleton(count, default_value=0),
            min_value=beam.pvalue.AsSingleton(minv, default_value=None),
            max_value=beam.pvalue.AsSingleton(maxv, default_value=None),
            mean_value=beam.pvalue.AsSingleton(mean, default_value=None),
            info=beam.pvalue.AsSingleton(bounds, default_value=default_bounds),
            out_cnt=beam.pvalue.AsSingleton(outlier_count, default_value=0),
        )
    )

    return profile


def _quantiles_from_values(values: List[float]) -> List[float]:
    if not values:
        return []
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    result: List[float] = []
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        if n == 1:
            result.append(sorted_vals[0])
            continue
        pos = q * (n - 1)
        lower = math.floor(pos)
        upper = math.ceil(pos)
        if lower == upper:
            result.append(sorted_vals[int(pos)])
        else:
            weight = pos - lower
            interpolated = sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight
            result.append(interpolated)
    return result


def summarize_numeric_values(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "stddev": None,
            "quantiles": [],
            "outlier_bounds": {
                "quantiles": [],
                "q1": None,
                "q3": None,
                "iqr": None,
                "lower": None,
                "upper": None,
            },
            "outlier_count": 0,
            "histogram": {"edges": [], "counts": []},
        }

    count = len(values)
    min_val = min(values)
    max_val = max(values)
    mean_val = sum(values) / count
    variance = sum((v - mean_val) ** 2 for v in values) / count if count > 0 else 0.0
    stddev = math.sqrt(variance)
    quantiles = _quantiles_from_values(values)

    q1 = quantiles[1] if len(quantiles) >= 4 else None
    q3 = quantiles[3] if len(quantiles) >= 4 else None
    iqr = (q3 - q1) if (q1 is not None and q3 is not None) else None
    lower = (q1 - 1.5 * iqr) if iqr is not None else None
    upper = (q3 + 1.5 * iqr) if iqr is not None else None

    outlier_count = 0
    if lower is not None or upper is not None:
        for v in values:
            if (lower is not None and v < lower) or (upper is not None and v > upper):
                outlier_count += 1

    def _histogram_from_values() -> Dict[str, Any]:
        bin_count = 20
        if min_val == max_val:
            return {"edges": [], "counts": []}
        span = max_val - min_val
        if span <= 0:
            return {"edges": [], "counts": []}
        step = span / bin_count
        if step <= 0:
            return {"edges": [], "counts": []}
        edges = [min_val + i * step for i in range(bin_count + 1)]
        counts = [0 for _ in range(bin_count)]
        for value in values:
            if value is None:
                continue
            try:
                position = (value - min_val) / step
            except (TypeError, ValueError):
                continue
            if position < 0:
                index = 0
            elif position >= bin_count:
                index = bin_count - 1
            else:
                index = int(position)
            counts[index] += 1
        return {"edges": edges, "counts": counts}

    histogram = _histogram_from_values()

    return {
        "count": count,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "stddev": stddev,
        "quantiles": quantiles,
        "outlier_bounds": {
            "quantiles": quantiles,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower": lower,
            "upper": upper,
        },
        "outlier_count": outlier_count,
        "histogram": histogram,
    }


def _write_single_shard(prefix: str, suffix: str, lines: Iterable[str]) -> str:
    path = f"{prefix}-00000-of-00001{suffix}"
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for line in lines:
            fp.write(line)
            if not line.endswith("\n"):
                fp.write("\n")
    return path


def _single_shard_path(prefix: str, suffix: str) -> str:
    return f"{prefix}-00000-of-00001{suffix}"


def _safe_component_name(name: str) -> str:
    base = os.path.basename(name)
    if not base:
        base = "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base)


def cleanup_outputs(patterns: Iterable[str]) -> None:
    for pattern in patterns:
        for candidate in glob.glob(pattern):
            if os.path.isdir(candidate):
                shutil.rmtree(candidate, ignore_errors=True)
            else:
                try:
                    os.remove(candidate)
                except FileNotFoundError:
                    continue


def _accumulate_numeric_values(
    rc: RowCtx,
    rule: Dict[str, Any],
    global_values: Optional[Dict[str, List[float]]] = None,
    per_file_values: Optional[Dict[str, Dict[str, List[float]]]] = None,
    file_path: Optional[str] = None,
    per_file_series: Optional[Dict[str, Dict[str, List[Tuple[dt.datetime, float]]]]] = None,
    event_time_col: Optional[str] = None,
    event_time_format: Optional[str] = None,
    reference_time: Optional[dt.datetime] = None,
) -> None:
    alias = getattr(rc, "_dq_header_alias", None)
    alias_map = alias if isinstance(alias, dict) else {}
    for col in rule.get("numeric_cols", []):
        canon = _canonicalize_column_name(col)
        actual_col = alias_map.get(canon)
        if not actual_col and col in rc.header:
            actual_col = col
        if not actual_col:
            continue
        val = rc.data.get(actual_col)
        if _is_missing_value(val):
            continue
        try:
            parsed_val = parse_numeric_with_units(val, actual_col, rule)
        except Exception:
            continue
        if global_values is not None:
            global_values.setdefault(actual_col, []).append(parsed_val)
        if per_file_values is not None and file_path is not None:
            per_file_values.setdefault(file_path, {}).setdefault(actual_col, []).append(parsed_val)
        if (
            per_file_series is not None
            and file_path is not None
            and event_time_col
            and not _is_missing_value(rc.data.get(event_time_col))
        ):
            try:
                ts = parse_event_time(rc.data.get(event_time_col), (event_time_format or "auto").lower())
            except Exception:
                continue
            per_file_series.setdefault(file_path, {}).setdefault(actual_col, []).append((ts, parsed_val))

    actual_event_col = event_time_col or _resolve_column_from_header(rc.header, alias_map, rule.get("event_time_col"))
    _maybe_attach_staleness_metric(
        rc,
        rule,
        alias_map,
        actual_event_col,
        event_time_format,
        reference_time,
    )
    computed_metrics = getattr(rc, "_dq_computed_metrics", None)
    if not computed_metrics:
        return
    for col_name, raw_value in computed_metrics.items():
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if global_values is not None:
            global_values.setdefault(col_name, []).append(numeric_value)
        if per_file_values is not None and file_path is not None:
            per_file_values.setdefault(file_path, {}).setdefault(col_name, []).append(numeric_value)
        if (
            per_file_series is not None
            and file_path is not None
            and event_time_col
            and not _is_missing_value(rc.data.get(event_time_col))
        ):
            try:
                ts = parse_event_time(rc.data.get(event_time_col), (event_time_format or "auto").lower())
            except Exception:
                continue
            per_file_series.setdefault(file_path, {}).setdefault(col_name, []).append((ts, numeric_value))


def _format_stat_value(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1000 or magnitude < 0.01:
        return f"{value:.3g}"
    return f"{value:.3f}"


def _plot_histogram(values: List[float], output_path: str, title: str, xlabel: str) -> None:
    if not values:
        return
    if go is None:
        raise RuntimeError(
            "Plotly is required to generate visualizations. Install plotly to enable this feature."
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bins = max(10, min(50, int(math.sqrt(len(values)))))
    mean_val = statistics.fmean(values)
    median_val = statistics.median(values)

    fig = go.Figure(
        data=[
            go.Histogram(
                x=values,
                nbinsx=bins,
                marker=dict(color="#2a9d8f", line=dict(color="#0b3d3f", width=0.6)),
                opacity=0.85,
                hovertemplate="Value: %{x}<br>Frequency: %{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title="Frequency",
        bargap=0.05,
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=60),
    )
    fig.add_vline(
        x=mean_val,
        line=dict(color="#e76f51", dash="dash"),
        annotation=dict(
            text="Mean",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(231,111,81,0.1)",
        ),
    )
    fig.add_vline(
        x=median_val,
        line=dict(color="#264653", dash="dot"),
        annotation=dict(
            text="Median",
            showarrow=False,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(38,70,83,0.1)",
        ),
    )
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)


def _plot_boxplot(
    values: List[float],
    output_path: str,
    title: str,
    xlabel: str,
    summary: Dict[str, Any],
) -> None:
    if not values:
        return
    if go is None:
        raise RuntimeError(
            "Plotly is required to generate visualizations. Install plotly to enable this feature."
        )

    bounds = summary.get("outlier_bounds", {}) if summary else {}
    text_lines = [
        f"Q1: {_format_stat_value(bounds.get('q1'))}",
        f"Q3: {_format_stat_value(bounds.get('q3'))}",
        f"IQR: {_format_stat_value(bounds.get('iqr'))}",
        f"Lower fence: {_format_stat_value(bounds.get('lower'))}",
        f"Upper fence: {_format_stat_value(bounds.get('upper'))}",
        f"Outliers: {summary.get('outlier_count', 0)}",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            x=values,
            orientation="h",
            name="distribution",
            boxpoints="outliers",
            marker=dict(color="#2a9d8f", opacity=0.7),
            line=dict(color="#264653"),
            hovertemplate="Value: %{x}<extra></extra>",
        )
    )
    lower = bounds.get("lower")
    upper = bounds.get("upper")
    if lower is not None:
        fig.add_vline(
            x=lower,
            line=dict(color="#e9c46a", dash="dash"),
            annotation=dict(
                text="Lower fence",
                showarrow=False,
                bgcolor="rgba(233,196,106,0.15)",
                xanchor="right",
            ),
        )
    if upper is not None:
        fig.add_vline(
            x=upper,
            line=dict(color="#e9c46a", dash="dash"),
            annotation=dict(
                text="Upper fence",
                showarrow=False,
                bgcolor="rgba(233,196,106,0.15)",
                xanchor="left",
            ),
        )

    fig.update_layout(
        title=f"{title} – outlier analysis",
        xaxis_title=xlabel,
        template="plotly_white",
        margin=dict(l=60, r=200, t=60, b=60),
        showlegend=False,
    )
    fig.add_annotation(
        x=1.02,
        y=0.5,
        xref="paper",
        yref="paper",
        text="<br>".join(text_lines),
        showarrow=False,
        align="left",
        bgcolor="rgba(248,250,252,0.95)",
        bordercolor="#d0d7de",
        borderwidth=1,
        font=dict(size=11),
    )
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)


def _sample_for_interactive(values: List[float], limit: int = 1000) -> List[float]:
    if not values:
        return []
    sorted_values = sorted(values)
    if len(sorted_values) <= limit:
        return list(sorted_values)
    step = len(sorted_values) / float(limit)
    sampled: List[float] = []
    cursor = 0.0
    while len(sampled) < limit and int(cursor) < len(sorted_values):
        sampled.append(sorted_values[int(cursor)])
        cursor += step
    if sampled[-1] != sorted_values[-1]:
        sampled[-1] = sorted_values[-1]
    return sampled


def _prepare_time_series_payload(
    series: List[Tuple[dt.datetime, float]], limit: int = 1000
) -> Optional[Dict[str, List[Any]]]:
    if not series:
        return None
    valid = [item for item in series if isinstance(item[0], dt.datetime)]
    if not valid:
        return None
    valid.sort(key=lambda item: item[0])
    if len(valid) > limit:
        step = len(valid) / float(limit)
        reduced: List[Tuple[dt.datetime, float]] = []
        cursor = 0.0
        while len(reduced) < limit and int(cursor) < len(valid):
            reduced.append(valid[int(cursor)])
            cursor += step
        if reduced[-1] != valid[-1]:
            reduced[-1] = valid[-1]
        selected = reduced
    else:
        selected = valid
    times = [ts.isoformat() for ts, _ in selected]
    values = [float(val) for _, val in selected]
    return {"time": times, "values": values}


def _collect_outlier_examples(
    values: List[float], lower: Optional[float], upper: Optional[float], limit: int = 15
) -> List[float]:
    if not values:
        return []
    if lower is None and upper is None:
        return []
    examples: List[float] = []
    for val in sorted(values):
        if (lower is not None and val < lower) or (upper is not None and val > upper):
            examples.append(val)
            if len(examples) >= limit:
                break
    return examples


def _render_combined_outlier_dashboard(payload: Dict[str, Any], output_path: str) -> None:
    if not payload.get("columns"):
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    page_title = f"Outlier overview – {payload.get('file_label', 'dataset')}"
    payload_json = json.dumps(payload, ensure_ascii=False)
    chart_height = max(300, 80 * len(payload.get("columns", [])))
    template = Template(
        """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>$page_title</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {
      font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 1.5rem;
      background: #f6f8fa;
      color: #24292f;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }
    h1 {
      margin-top: 0;
      font-size: 1.75rem;
    }
    .section {
      border: 1px solid #d0d7de;
      background: #fff;
      border-radius: 8px;
      padding: 1rem;
      box-shadow: 0 1px 2px rgba(27, 31, 35, 0.05);
      display: flex;
      flex-direction: column;
      gap: 1rem;
      min-height: 0;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      align-items: center;
      margin-bottom: 1rem;
    }
    select {
      padding: 0.35rem 0.5rem;
      border-radius: 4px;
      border: 1px solid #d0d7de;
      font-size: 0.95rem;
    }
    .feature-list {
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      margin: 1rem 0;
      padding: 0;
      list-style: none;
    }
    .feature-list label {
      display: flex;
      align-items: center;
      gap: 0.35rem;
      background: #e9f5f2;
      border: 1px solid #b7e4d9;
      border-radius: 6px;
      padding: 0.35rem 0.6rem;
      cursor: pointer;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 2px rgba(27, 31, 35, 0.05);
    }
    th, td {
      padding: 0.6rem 0.75rem;
      border-bottom: 1px solid #d8dee4;
      text-align: left;
      vertical-align: top;
    }
    th {
      background: #f1f5f9;
      width: 12rem;
    }
    tbody tr:nth-child(odd) td {
      background: #f9fbfc;
    }
    .note {
      font-size: 0.85rem;
      color: #57606a;
    }
    h2 {
      font-size: 1.35rem;
      margin: 0;
    }
    .feature-note {
      font-size: 0.85rem;
      color: #57606a;
    }
    .viz-layout {
      display: grid;
      grid-template-rows: minmax(0, 1fr) minmax(0, 1fr);
      gap: 1.5rem;
      flex: 1 1 auto;
      min-height: 0;
    }
    .chart-container {
      flex: 1 1 auto;
      min-height: 0;
      display: flex;
    }
    .chart {
      width: 100%;
      flex: 1 1 auto;
      min-height: 0;
    }
    .table-wrapper {
      flex: 1 1 auto;
      min-height: 0;
      overflow: auto;
    }
    .table-wrapper table {
      margin-top: 0;
    }
    .note {
      margin: 0;
    }
    .feature-note {
      margin: 0;
    }
  </style>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
  <h1>$page_title</h1>
  <div class="viz-layout">
    <div class="section" id="distribution-section">
      <h2>Statistics overview</h2>
      <div class="chart-container">
        <div class="chart" id="distribution-chart" style="min-height: ${chart_height}px;"></div>
      </div>
      <ul class="feature-list" id="feature-list"></ul>
      <div class="note" id="sampling-note"></div>
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Feature</th>
              <th>Statistics</th>
            </tr>
          </thead>
          <tbody id="stats-body"></tbody>
        </table>
      </div>
    </div>
    <div class="section" id="outlier-section">
      <h2>Outlier analysis</h2>
      <div class="toolbar">
        <label>Scale
          <select id="scale-mode">
            <option value="raw">Raw values</option>
            <option value="normalized">Z-score (mean ± std)</option>
          </select>
        </label>
      </div>
      <div class="chart-container">
        <div class="chart" id="outlier-chart" style="min-height: ${chart_height}px;"></div>
      </div>
      <div class="feature-note">Use the checkboxes above to toggle columns in both charts.</div>
    </div>
  </div>
  <script>
    const payload = $payload_json;
    const chartId = 'outlier-chart';
    const distributionChartId = 'distribution-chart';
    const featureList = document.getElementById('feature-list');
    const statsBody = document.getElementById('stats-body');
    const scaleSelect = document.getElementById('scale-mode');
    const samplingNote = document.getElementById('sampling-note');

    const colors = ['#2a9d8f', '#264653', '#e76f51', '#f4a261', '#457b9d', '#ef476f', '#118ab2', '#073b4c'];

    function formatNumber(value) {
      if (value === null || value === undefined || Number.isNaN(value)) {
        return 'N/A';
      }
      if (typeof value === 'number') {
        const magnitude = Math.abs(value);
        if (magnitude === 0) {
          return '0';
        }
        if (magnitude >= 1000 || magnitude < 0.01) {
          return value.toExponential(3);
        }
        return value.toFixed(3);
      }
      return String(value);
    }

    const distributionTraces = payload.columns.map((column, index) => {
      const label = column.description ? (column.name + ' — ' + column.description) : column.name;
      return {
        type: 'histogram',
        name: label,
        x: column.samples.raw,
        opacity: 0.65,
        marker: { color: colors[index % colors.length] },
        hovertemplate: label + '<br>value=%{x}<br>count=%{y}<extra></extra>'
      };
    });

    const distributionLayout = {
      barmode: 'overlay',
      margin: { l: 80, r: 40, t: 10, b: 60 },
      hovermode: 'closest',
      xaxis: { title: 'Value', zeroline: true, zerolinecolor: '#adb5bd' },
      yaxis: { title: 'Frequency', automargin: true },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff'
    };

    Plotly.newPlot(distributionChartId, distributionTraces, distributionLayout, { displaylogo: false, responsive: true });

    const traces = payload.columns.map((column, index) => {
      const label = column.description ? (column.name + ' — ' + column.description) : column.name;
      return {
        type: 'box',
        name: label,
        x: column.samples.raw,
        orientation: 'h',
        boxpoints: 'outliers',
        jitter: 0.4,
        whiskerwidth: 0.2,
        marker: { color: colors[index % colors.length], opacity: 0.7 },
        line: { color: '#264653' },
        hovertemplate: label + '<br>value=%{x}<extra></extra>'
      };
    });

    const layout = {
      margin: { l: 120, r: 40, t: 20, b: 60 },
      showlegend: false,
      hovermode: 'closest',
      xaxis: { title: 'Value', zeroline: true, zerolinecolor: '#adb5bd' },
      yaxis: { automargin: true },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff'
    };

    Plotly.newPlot(chartId, traces, layout, { displaylogo: false, responsive: true });

    function applyScale(mode) {
      payload.columns.forEach((column, index) => {
        const values = mode === 'normalized' ? column.samples.normalized : column.samples.raw;
        Plotly.restyle(chartId, { x: [values] }, [index]);
      });
      const xTitle = mode === 'normalized' ? 'Z-score (mean = 0, std = 1)' : 'Value';
      Plotly.relayout(chartId, { 'xaxis.title': xTitle });
    }

    scaleSelect.addEventListener('change', (event) => {
      applyScale(event.target.value);
    });

    payload.columns.forEach((column, index) => {
      const item = document.createElement('li');
      const label = document.createElement('label');
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.checked = true;
      checkbox.addEventListener('change', () => {
        Plotly.restyle(chartId, { visible: checkbox.checked ? true : 'legendonly' }, [index]);
        Plotly.restyle(distributionChartId, { visible: checkbox.checked ? true : 'legendonly' }, [index]);
      });
      const text = document.createElement('span');
      text.textContent = column.name;
      label.appendChild(checkbox);
      label.appendChild(text);
      if (column.description) {
        const desc = document.createElement('small');
        desc.textContent = ' ' + column.description;
        desc.style.fontSize = '0.75rem';
        desc.style.color = '#4b5563';
        label.appendChild(desc);
      }
      item.appendChild(label);
      featureList.appendChild(item);

      const statsRow = document.createElement('tr');
      const featureCell = document.createElement('th');
      featureCell.textContent = column.name;
      if (column.description) {
        const desc = document.createElement('div');
        desc.textContent = column.description;
        desc.style.fontSize = '0.8rem';
        desc.style.color = '#4b5563';
        featureCell.appendChild(desc);
      }
      const statsCell = document.createElement('td');
      const statsLines = [
        'Count: ' + column.stats.count,
        'Min: ' + formatNumber(column.stats.min),
        'Max: ' + formatNumber(column.stats.max),
        'Mean: ' + formatNumber(column.stats.mean),
        'Std dev: ' + formatNumber(column.stats.stddev),
        'Q1: ' + formatNumber(column.stats.q1),
        'Median: ' + formatNumber(column.stats.q2),
        'Q3: ' + formatNumber(column.stats.q3),
        'IQR: ' + formatNumber(column.stats.iqr),
        'Lower fence: ' + formatNumber(column.stats.lower),
        'Upper fence: ' + formatNumber(column.stats.upper),
        'Outliers detected: ' + column.stats.outlier_count
      ];
      if (column.outliers.examples.length) {
        statsLines.push('Outlier samples: ' + column.outliers.examples.map(formatNumber).join(', '));
      }
      statsCell.textContent = statsLines.join('\n');
      statsCell.style.whiteSpace = 'pre-line';
      statsRow.appendChild(featureCell);
      statsRow.appendChild(statsCell);
      statsBody.appendChild(statsRow);
    });

    if (payload.notes && payload.notes.sampled) {
      samplingNote.textContent = payload.notes.sampled;
    }

    applyScale('raw');
  </script>
</body>
</html>
"""
    )
    html_doc = template.safe_substitute(
        page_title=html.escape(page_title),
        chart_height=chart_height,
        payload_json=payload_json,
    )
    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write(html_doc)

def create_feature_visualizations(
    per_file_numeric_values: Dict[str, Dict[str, List[float]]],
    dq_out: str,
    column_descriptions: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, str]], Optional[str]]:
    if not per_file_numeric_values:
        return {}, "No numeric columns detected; skipping feature visualizations."
    if go is None:
        return {}, "Plotly is not installed; feature visualizations were skipped."

    visualization_root = os.path.join(dq_out, "visualizations")
    generated: Dict[str, Dict[str, str]] = {}

    for file_path, col_values in sorted(per_file_numeric_values.items()):
        safe_file = _safe_component_name(file_path)
        interactive_columns: List[Dict[str, Any]] = []
        sampled_flag = False
        for col, values in sorted(col_values.items()):
            if not values:
                continue
            safe_col = _safe_component_name(col)
            title = f"{os.path.basename(file_path)} – {col} distribution"
            desc = column_descriptions.get(col)
            if desc:
                title = f"{title}\n{desc}"
            output_path = os.path.join(visualization_root, safe_file, f"{safe_col}.html")
            summary = summarize_numeric_values(values)
            _plot_histogram(values, output_path, title, col)
            generated.setdefault(file_path, {})[col] = os.path.relpath(output_path, dq_out)

            outlier_path = os.path.join(
                visualization_root, safe_file, f"{safe_col}_outliers.html"
            )
            _plot_boxplot(values, outlier_path, title, col, summary)
            generated[file_path][f"{col}_outliers"] = os.path.relpath(outlier_path, dq_out)

            samples_raw = _sample_for_interactive(values)
            if samples_raw and len(samples_raw) < len(values):
                sampled_flag = True
            stddev = summary.get("stddev") or 0.0
            mean_val = summary.get("mean") or 0.0
            if stddev:
                samples_norm = [(v - mean_val) / stddev for v in samples_raw]
            else:
                samples_norm = [0.0 for _ in samples_raw]
            bounds = summary.get("outlier_bounds", {}) if summary else {}
            outlier_examples = _collect_outlier_examples(
                values, bounds.get("lower"), bounds.get("upper")
            )
            interactive_columns.append(
                {
                    "name": col,
                    "description": desc,
                    "samples": {"raw": samples_raw, "normalized": samples_norm},
                    "stats": {
                        "count": summary.get("count"),
                        "min": summary.get("min"),
                        "max": summary.get("max"),
                        "mean": summary.get("mean"),
                        "stddev": summary.get("stddev"),
                        "q1": bounds.get("q1"),
                        "q2": (summary.get("quantiles") or [None, None, None])[2]
                        if summary.get("quantiles")
                        else None,
                        "q3": bounds.get("q3"),
                        "iqr": bounds.get("iqr"),
                        "lower": bounds.get("lower"),
                        "upper": bounds.get("upper"),
                        "outlier_count": summary.get("outlier_count", 0),
                    },
                    "outliers": {
                        "examples": outlier_examples,
                    },
                }
            )

        if interactive_columns:
            dashboard_payload: Dict[str, Any] = {
                "file": file_path,
                "file_label": os.path.basename(file_path) or file_path,
                "columns": interactive_columns,
                "notes": {},
            }
            if sampled_flag:
                dashboard_payload["notes"][
                    "sampled"
                ] = "Interactive view shows an evenly spaced sample of up to 1,000 values per feature."
            dashboard_path = os.path.join(
                visualization_root, safe_file, "combined_outliers.html"
            )
            _render_combined_outlier_dashboard(dashboard_payload, dashboard_path)
            generated.setdefault(file_path, {})[
                "combined_outliers_dashboard"
            ] = os.path.relpath(dashboard_path, dq_out)

    return generated, None


def write_execution_log(
    dq_out: str,
    input_pattern: str,
    matched_files: List[str],
    per_file_counts: Dict[str, int],
    bad_issue_count: int,
    issue_summary: List[Dict[str, Any]],
    visualization_index: Dict[str, Dict[str, str]],
    visualization_note: Optional[str],
) -> None:
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    total_good = sum(per_file_counts.values())
    lines: List[str] = []
    header = f"[{timestamp}] Data quality pipeline completed"
    lines.append(header)
    lines.append(f"Input pattern: {input_pattern}")
    lines.append(f"Files processed: {len(matched_files)}")
    lines.append(f"Total good rows: {total_good}")
    lines.append(f"Total issues detected: {bad_issue_count}")
    lines.append("")
    lines.append("Per-file good row counts:")
    if per_file_counts:
        for file_path, count in sorted(per_file_counts.items()):
            lines.append(f"  - {file_path}: {count}")
    else:
        lines.append("  (no good rows emitted)")
    lines.append("")
    lines.append("Issue summary by dimension:")
    if issue_summary:
        for entry in issue_summary:
            lines.append(
                f"  - {entry.get('dimension')}: {entry.get('issue_count', 0)} issues across {len(entry.get('scenarios', []))} scenarios"
            )
    else:
        lines.append("  (no issues detected)")
    lines.append("")
    if visualization_note:
        lines.append(f"Visualizations: {visualization_note}")
    else:
        chart_count = sum(len(cols) for cols in visualization_index.values())
        lines.append(f"Visualizations generated: {chart_count} charts across {len(visualization_index)} files")
    lines.append("")
    lines.append("Output directory: {0}".format(dq_out))

    _write_single_shard(os.path.join(dq_out, "logs", "pipeline"), ".log", lines)


def write_quality_report(
    dq_out: str,
    input_pattern: str,
    matched_files: List[str],
    per_file_counts: Dict[str, int],
    bad_issue_count: int,
    bad_issue_samples: List[Dict[str, Any]],
    issue_summary: List[Dict[str, Any]],
    per_file_numeric_values: Dict[str, Dict[str, List[float]]],
    per_file_time_series: Dict[str, Dict[str, List[Tuple[dt.datetime, float]]]],
    column_descriptions: Dict[str, str],
    visualization_index: Dict[str, Dict[str, str]],
    visualization_note: Optional[str],
) -> None:
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    total_good = sum(per_file_counts.values())
    issue_summary = complete_issue_dimensions(issue_summary)

    per_file_section: List[Dict[str, Any]] = []
    for file_path in matched_files:
        numeric_summary: Dict[str, Any] = {}
        file_series = per_file_time_series.get(file_path) or {}
        for col, values in sorted((per_file_numeric_values.get(file_path) or {}).items()):
            stats = summarize_numeric_values(values)
            outlier_bounds = stats.get("outlier_bounds") or {}
            numeric_summary[col] = {
                "count": stats.get("count"),
                "min": stats.get("min"),
                "max": stats.get("max"),
                "mean": stats.get("mean"),
                "stddev": stats.get("stddev"),
                "quantiles": stats.get("quantiles"),
                "distribution": stats.get("histogram"),
                "outliers": {
                    "count": stats.get("outlier_count"),
                    "lower_fence": outlier_bounds.get("lower"),
                    "upper_fence": outlier_bounds.get("upper"),
                    "iqr": outlier_bounds.get("iqr"),
                    "q1": outlier_bounds.get("q1"),
                    "q3": outlier_bounds.get("q3"),
                },
                "description": column_descriptions.get(col),
            }
            series_payload = _prepare_time_series_payload(file_series.get(col) or [])
            if series_payload:
                numeric_summary[col]["time_series"] = series_payload

        per_file_section.append(
            {
                "file": file_path,
                "good_rows": per_file_counts.get(file_path, 0),
                "numeric_columns": numeric_summary,
                "visualizations": visualization_index.get(file_path, {}),
            }
        )

    report = {
        "generated_at": generated_at,
        "input_pattern": input_pattern,
        "files_processed": len(matched_files),
        "total_good_rows": total_good,
        "total_issue_records": bad_issue_count,
        "issue_summary": issue_summary,
        "bad_issue_samples": bad_issue_samples,
        "per_file": per_file_section,
    }
    if visualization_note:
        report["visualizations"] = {"note": visualization_note}

    _write_single_shard(
        os.path.join(dq_out, "quality_report"),
        ".json",
        [json.dumps(report, ensure_ascii=False, indent=2)],
    )


def collect_numeric_values_by_file(
    files: List[str], rules: List[Dict[str, Any]], reference_time: Optional[dt.datetime] = None
) -> Tuple[
    Dict[str, Dict[str, List[float]]],
    Dict[str, Dict[str, List[Tuple[dt.datetime, float]]]],
]:
    per_file: Dict[str, Dict[str, List[float]]] = {}
    per_file_series: Dict[str, Dict[str, List[Tuple[dt.datetime, float]]]] = {}
    ref_time = reference_time or dt.datetime.now(dt.timezone.utc)
    for path in files:
        header, rows, _ = read_csv_file(path)
        rule = pick_rule(rules, path)
        header_state = _ensure_header_state(rule, path, header)
        if not header_state.get("valid", True):
            continue
        alias = header_state.get("alias") or {}
        event_time_actual = header_state.get("event_time_actual")
        event_time_format = rule.get("event_time_format")
        for rc in rows:
            setattr(rc, "_dq_header_alias", alias)
            _accumulate_numeric_values(
                rc,
                rule,
                None,
                per_file,
                path,
                per_file_series=per_file_series,
                event_time_col=event_time_actual,
                event_time_format=event_time_format,
                reference_time=ref_time,
            )
    return per_file, per_file_series


def _load_jsonl(path: str, limit: Optional[int] = None) -> Tuple[int, List[Dict[str, Any]]]:
    count = 0
    samples: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return count, samples
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            count += 1
            if limit is None or len(samples) < limit:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    samples.append({"raw": line, "error": "json_decode_error"})
    return count, samples

def union_numeric_cols(rules: List[Dict[str, Any]]) -> List[str]:
    cols = set()
    for r in rules:
        cols.update(r.get("numeric_cols", []))
    return sorted(cols)


def gather_column_descriptions(rules: List[Dict[str, Any]]) -> Dict[str, str]:
    descriptions: Dict[str, str] = {}
    for r in rules:
        for _, cols in (r.get("metadata_by_file") or {}).items():
            for col, desc in cols.items():
                descriptions.setdefault(col, desc)
    return descriptions


def evaluate_metadata_prerequisites(
    rules: List[Dict[str, Any]], files: List[str]
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Validate that every input file has metadata available.

    Returns a tuple of (metadata_complete, notifications). When metadata is
    missing for any file, ``metadata_complete`` will be ``False`` and
    ``notifications`` will contain structured messages describing the missing
    metadata. When all files have metadata definitions the function returns
    ``(True, [])``.
    """

    if not files:
        return True, []

    missing: List[str] = []
    has_metadata = False

    for path in files:
        rule = pick_rule(rules, path)
        metadata_cols = _resolve_metadata_columns(rule, path)
        if metadata_cols:
            has_metadata = True
        else:
            missing.append(path)

    if not missing:
        return True, []

    notifications: List[Dict[str, Any]] = []

    if len(missing) == len(files) and not has_metadata:
        notifications.append(
            {
                "type": "missing_metadata",
                "scope": "dataset",
                "message": "No metadata descriptions were found for any of the input files.",
                "files": missing,
            }
        )
    else:
        for path in missing:
            notifications.append(
                {
                    "type": "missing_metadata",
                    "scope": "file",
                    "file": path,
                    "message": "Metadata description is missing; data quality verification was skipped.",
                }
            )

    return False, notifications


def _write_metadata_notifications(
    dq_out: str, notifications: List[Dict[str, Any]]
) -> Optional[str]:
    if not notifications:
        return None
    os.makedirs(dq_out, exist_ok=True)
    lines = [json.dumps(note, ensure_ascii=False) for note in notifications]
    return _write_single_shard(
        os.path.join(dq_out, "metadata_notifications"),
        ".jsonl",
        lines,
    )


def _write_metadata_block_report(
    dq_out: str,
    input_pattern: str,
    matched_files: List[str],
    notifications: List[Dict[str, Any]],
) -> None:
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_metadata_notifications(dq_out, notifications)

    summary = {
        "input_pattern": input_pattern,
        "files_processed": 0,
        "total_good_rows": 0,
        "total_issue_records": 0,
        "issue_summary": complete_issue_dimensions([]),
        "bad_issue_samples": [],
        "per_file": [],
        "status": "blocked",
        "blocked_reason": "missing_metadata",
        "metadata_notifications": notifications,
        "generated_at": timestamp,
    }

    _write_single_shard(
        os.path.join(dq_out, "quality_report"),
        ".json",
        [json.dumps(summary, ensure_ascii=False, indent=2)],
    )

    execution_log = {
        "input_pattern": input_pattern,
        "matched_files": matched_files,
        "status": "blocked",
        "blocked_reason": "missing_metadata",
        "metadata_notifications": notifications,
        "generated_at": timestamp,
    }

    _write_single_shard(
        os.path.join(dq_out, "execution_log"),
        ".json",
        [json.dumps(execution_log, ensure_ascii=False, indent=2)],
    )


def classify_issue(reason: str) -> Tuple[str, str]:
    """Map a failure reason to a (dimension, scenario) tuple."""
    if not reason:
        return "Other", "Uncategorized issue"

    r = reason.lower()
    if r.startswith("missing_required_column"):
        return "Completeness", "Required column missing from header"
    if r.startswith("missing_metadata_column"):
        return "Completeness", "Column listed in metadata is missing from header"
    if r.startswith("missing_required_prefix"):
        return "Completeness", "Required column prefix missing from header"
    if r.startswith("null_required"):
        return "Completeness", "Required field is empty"
    if r.startswith("bad_enum"):
        return "Accuracy", "Value not in allowed enumeration"
    if r.startswith("out_of_range"):
        return "Accuracy", "Numeric value outside configured range"
    if r.startswith("bad_numeric"):
        return "Accuracy", "Numeric value could not be parsed"
    if r.startswith("bad_timestamp"):
        return "Accuracy", "Timestamp could not be parsed"
    if r.startswith("time_epoch_out_of_range"):
        return "Staleness", "Epoch timestamp outside allowed bounds"
    if r.startswith("stale_event"):
        return "Staleness", "Event is older than freshness SLO"
    if r.startswith("future_event"):
        return "Staleness", "Event timestamp is too far in the future"
    if r.startswith("fk_missing"):
        return "Consistency", "Referenced value missing from reference data"
    if r.startswith("parse_error"):
        return "Accuracy", "CSV row could not be parsed"
    if r in ("empty_file",):
        return "Completeness", "File contains no data"
    if r.startswith("no_header"):
        return "Completeness", "Header row missing"
    if r.startswith("duplicate_primary_key"):
        return "Duplication", "Duplicate primary key values detected"
    return "Other", "Uncategorized issue"


class DimensionIssueSummary(beam.CombineFn if _HAVE_BEAM else object):  # type: ignore[misc]
    """Aggregate issue information by dimension."""

    def create_accumulator(self):
        return {"count": 0, "scenarios": {}, "examples": []}

    def add_input(self, accumulator, issue: Dict[str, Any]):
        weight = issue.get("occurrence_count", 1)
        scenario = issue.get("scenario") or "Uncategorized issue"
        accumulator["count"] += weight
        accumulator["scenarios"][scenario] = accumulator["scenarios"].get(scenario, 0) + weight
        if len(accumulator["examples"]) < 5:
            example = dict(issue)
            example.pop("occurrence_count", None)
            accumulator["examples"].append(example)
        return accumulator

    def merge_accumulators(self, accumulators):
        merged = self.create_accumulator()
        for acc in accumulators:
            merged["count"] += acc["count"]
            for scenario, cnt in acc["scenarios"].items():
                merged["scenarios"][scenario] = merged["scenarios"].get(scenario, 0) + cnt
            if len(merged["examples"]) < 5:
                merged["examples"].extend(acc["examples"][: 5 - len(merged["examples"])])
        return merged

    def extract_output(self, accumulator):
        scenarios = [
            {"scenario": scen, "count": cnt}
            for scen, cnt in sorted(accumulator["scenarios"].items(), key=lambda item: item[1], reverse=True)
        ]
        return {
            "issue_count": accumulator["count"],
            "scenarios": scenarios,
            "examples": accumulator["examples"],
        }


def attach_dimension(issue: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    dimension, scenario = classify_issue(issue.get("reason", ""))
    enriched = dict(issue)
    enriched["dimension"] = dimension
    enriched["scenario"] = scenario
    if "occurrence_count" not in enriched:
        enriched["occurrence_count"] = 1
    return dimension, enriched

# -----------------------------
# 主流程
# -----------------------------
def _run_with_beam(
    input_pattern: str, good_out: str, bad_out: str, dq_out: str, config_path: str, config: Optional[Dict[str, Any]] = None
):
    cfg = config if config is not None else load_config(config_path)
    rules = cfg["rules"]
    column_descriptions = gather_column_descriptions(rules)

    matched_files = sorted(glob.glob(input_pattern, recursive=True))
    if not matched_files:
        raise FileNotFoundError(f"No files matched input pattern: {input_pattern}")

    metadata_ready, notifications = evaluate_metadata_prerequisites(rules, matched_files)
    if not metadata_ready:
        _write_metadata_block_report(dq_out, input_pattern, matched_files, notifications)
        return

    opts = PipelineOptions(runner="DirectRunner", save_main_session=True, streaming=False)
    with beam.Pipeline(options=opts) as p:
        files = (
            p
            | "MatchFiles" >> fileio.MatchFiles(input_pattern)
            | "ReadMatches" >> fileio.ReadMatches()
        )

        parsed = files | "ParseCSV" >> beam.ParDo(ReadCSVWithHeader()).with_outputs(BAD_TAG, HEADERS_TAG, main="good")

        # 表头输出
        _ = (
            parsed.headers
            | "HeadersJSON" >> beam.Map(json.dumps)
            | "WriteHeaders" >> beam.io.WriteToText(os.path.join(dq_out, "headers", "per_file"),
                                                    file_name_suffix=".jsonl", num_shards=1)
        )
        union = (
            parsed.headers
            | "HdrToSetUnion" >> beam.Map(lambda h: set(h["header"]))
            | "Union" >> beam.CombineGlobally(lambda sets: set().union(*sets) if sets else set())
        )
        _ = (
            union
            | "UnionStr" >> beam.Map(lambda s: json.dumps(sorted(list(s))))
            | "WriteUnion" >> beam.io.WriteToText(os.path.join(dq_out, "headers", "union"),
                                                  file_name_suffix=".json", num_shards=1)
        )
        inter = (
            parsed.headers
            | "HdrToSetInter" >> beam.Map(lambda h: set(h["header"]))
            | "Inter" >> beam.CombineGlobally(lambda sets: set.intersection(*sets) if sets else set())
        )
        _ = (
            inter
            | "InterStr" >> beam.Map(lambda s: json.dumps(sorted(list(s))))
            | "WriteInter" >> beam.io.WriteToText(os.path.join(dq_out, "headers", "intersection"),
                                                  file_name_suffix=".json", num_shards=1)
        )

        header_issues = parsed.headers | "ValidateHeaders" >> beam.ParDo(ValidateHeader(rules))

        with_rules = parsed.good | "AttachRule" >> beam.Map(lambda rc: (rc, pick_rule(rules, rc.file)))

        class ValidateWrapper(beam.DoFn):
            def process(self, pair):
                rc, rule = pair
                validator = ValidateRow(rule)
                yield from validator.process(rc, None)

        validated = (
            with_rules
            | "ValidateRows" >> beam.ParDo(ValidateWrapper()).with_outputs(BAD_TAG, main="good")
        )

        # BAD 输出（JSONL）
        bad = (parsed.bad, validated.bad, header_issues) | "FlattenBad" >> beam.Flatten()
        _ = (
            bad
            | "ToJSON" >> beam.Map(json.dumps, ensure_ascii=False)
            | "WriteBad" >> beam.io.WriteToText(bad_out, file_name_suffix=".jsonl", num_shards=1)
        )

        issue_pairs = bad | "AttachIssueDimensions" >> beam.Map(attach_dimension)

        # 文件级 GOOD 行数
        per_file_counts = (
            validated.good
            | "PerFileKey" >> beam.Map(lambda rc: (rc.file, 1))
            | "PerFileCount" >> beam.CombinePerKey(sum)
        )
        _ = (
            per_file_counts
            | "CountsToJSON" >> beam.Map(lambda kv: json.dumps({"file": kv[0], "good_rows": kv[1]}))
            | "WriteFileCounts" >> beam.io.WriteToText(os.path.join(dq_out, "good_counts"),
                                                       file_name_suffix=".jsonl", num_shards=1)
        )

        # 主键重复统计（若配置）
        dup_issue_pairs = p | "EmptyDupIssues" >> beam.Create([])
        if any(_get_primary_key_columns(r) for r in rules):
            def pk_pairs(pair: Tuple[RowCtx, Dict[str, Any]]):
                rc, rule = pair
                cols = _get_primary_key_columns(rule)
                if not cols:
                    return
                key = _primary_key_value(rule, rc)
                if key is None:
                    return
                yield ((cols, key), rc.file)

            pkp = with_rules | "PKPairs" >> beam.FlatMap(pk_pairs)
            dup_keys = (
                pkp
                | "PKCount" >> beam.combiners.Count.PerKey()
                | "PKOnlyDup" >> beam.Filter(lambda kv: kv[1] > 1)
            )
            dup_pk = dup_keys | "PKDupCount" >> beam.combiners.Count.Globally()
            # _ = (
            #     dup_pk
            #     | "WritePKDup" >> beam.io.WriteToText(os.path.join(dq_out, "pk_duplicates"),
            #                                           file_name_suffix=".txt", num_shards=1)
            # )
            dup_issue_pairs = (
                dup_keys
                | "DupIssues" >> beam.Map(
                    lambda kv: (
                        "Duplication",
                        {
                            "dimension": "Duplication",
                            "reason": f"duplicate_primary_key:{kv[0][1]}",
                            "scenario": "Duplicate primary key values detected",
                            "detail": {
                                "primary_key_columns": list(kv[0][0]),
                                "key_value": kv[0][1],
                                "occurrences": kv[1],
                            },
                            "occurrence_count": max(kv[1] - 1, 1),
                        },
                    )
                )
            )

        all_issue_pairs = (issue_pairs, dup_issue_pairs) | "MergeIssuePairs" >> beam.Flatten()
        _ = (
            all_issue_pairs
            | "SummarizeIssues" >> beam.CombinePerKey(DimensionIssueSummary())
            | "FormatIssueSummary" >> beam.Map(
                lambda kv: json.dumps(
                    {
                        "dimension": kv[0],
                        "description": DIMENSION_DESCRIPTIONS.get(kv[0], DIMENSION_DESCRIPTIONS["Other"]),
                        **kv[1],
                    },
                    ensure_ascii=False,
                )
            )
            | "WriteIssueSummary" >> beam.io.WriteToText(
                os.path.join(dq_out, "issue_summary"), file_name_suffix=".jsonl", num_shards=1
            )
        )


    per_file_counts: Dict[str, int] = {}
    counts_path = _single_shard_path(os.path.join(dq_out, "good_counts"), ".jsonl")
    if os.path.exists(counts_path):
        with open(counts_path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                file_key = payload.get("file")
                if not file_key:
                    continue
                per_file_counts[file_key] = payload.get("good_rows", 0)

    issue_summary_path = _single_shard_path(os.path.join(dq_out, "issue_summary"), ".jsonl")
    issue_summary: List[Dict[str, Any]] = []
    if os.path.exists(issue_summary_path):
        with open(issue_summary_path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    issue_summary.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    issue_summary = complete_issue_dimensions(issue_summary)

    bad_path = _single_shard_path(bad_out, ".jsonl")
    _, bad_issue_samples = _load_jsonl(bad_path, limit=10)
    bad_issue_count = sum(entry.get("issue_count", 0) for entry in issue_summary)

    per_file_numeric_values: Dict[str, Dict[str, List[float]]] = {}
    per_file_time_series: Dict[str, Dict[str, List[Tuple[dt.datetime, float]]]] = {}
    visualization_index: Dict[str, Dict[str, str]] = {}
    visualization_note: Optional[str] = None
    visualizations_enabled, size_note = _visualizations_allowed(matched_files)
    if not visualizations_enabled:
        visualization_note = size_note
    else:
        (
            per_file_numeric_values,
            per_file_time_series,
        ) = collect_numeric_values_by_file(matched_files, rules, reference_time=reference_time)
        visualization_index, visualization_note = create_feature_visualizations(
            per_file_numeric_values, dq_out, column_descriptions
        )
    write_execution_log(
        dq_out,
        display_pattern,
        matched_files,
        per_file_counts,
        bad_issue_count,
        issue_summary,
        visualization_index,
        visualization_note,
    )
    write_quality_report(
        dq_out,
        display_pattern,
        matched_files,
        per_file_counts,
        bad_issue_count,
        bad_issue_samples,
        issue_summary,
        per_file_numeric_values,
        per_file_time_series,
        column_descriptions,
        visualization_index,
        visualization_note,
    )

    cleanup_outputs(
        [
            f"{good_out}-*",
            f"{bad_out}-*",
            os.path.join(dq_out, "good_counts") + "-*",
            os.path.join(dq_out, "issue_summary") + "-*",
            os.path.join(dq_out, "metadata_columns") + "-*",
        ]
    )


def _run_without_beam(
    input_pattern: str,
    good_out: str,
    bad_out: str,
    dq_out: str,
    config_path: str,
    config: Optional[Dict[str, Any]] = None,
    prefetched_files: Optional[List[str]] = None,
    input_display_pattern: Optional[str] = None,
    reference_time: Optional[dt.datetime] = None,
) -> None:
    cfg = config if config is not None else load_config(config_path)
    rules = cfg["rules"]
    column_descriptions = gather_column_descriptions(rules)

    display_pattern = input_display_pattern or input_pattern
    matched_files = prefetched_files if prefetched_files is not None else sorted(glob.glob(input_pattern, recursive=True))
    if not matched_files:
        raise FileNotFoundError(f"No files matched input pattern: {input_pattern}")

    metadata_ready, notifications = evaluate_metadata_prerequisites(rules, matched_files)
    if not metadata_ready:
        _write_metadata_block_report(dq_out, display_pattern, matched_files, notifications)
        return

    os.makedirs(dq_out, exist_ok=True)

    reference_time = reference_time or dt.datetime.now(dt.timezone.utc)

    visualizations_enabled, visualization_note = _visualizations_allowed(matched_files)

    headers_info: List[Dict[str, Any]] = []
    header_union: set = set()
    header_intersection: Optional[set] = None
    bad_issues: List[Dict[str, Any]] = []
    per_file_counts: Dict[str, int] = {}
    per_file_numeric_values: Dict[str, Dict[str, List[float]]] = {}
    per_file_time_series: Dict[str, Dict[str, List[Tuple[dt.datetime, float]]]] = {}
    pk_counts: Dict[Tuple[Tuple[str, ...], str], int] = {}

    for path in matched_files:
        header, rows, parse_issues = read_csv_file(path)
        headers_info.append({"file": path, "header": header})

        if header:
            header_union.update(header)
            if header_intersection is None:
                header_intersection = set(header)
            else:
                header_intersection &= set(header)
        else:
            if header_intersection is None:
                header_intersection = set()

        bad_issues.extend(parse_issues)

        rule = pick_rule(rules, path)

        header_state = _ensure_header_state(rule, path, header)
        if header_state.get("issues") and not header_state.get("reported"):
            bad_issues.extend(header_state["issues"])
            header_state["reported"] = True
        if not header_state.get("valid", True):
            continue

        event_time_actual = header_state.get("event_time_actual")
        event_time_format = rule.get("event_time_format")

        for rc in rows:
            valid, issues = validate_row_against_rule(rc, rule, None, reference_time=reference_time)
            if issues:
                bad_issues.extend(issues)

            if visualizations_enabled:
                _accumulate_numeric_values(
                    rc,
                    rule,
                    global_values=None,
                    per_file_values=per_file_numeric_values,
                    file_path=path,
                    per_file_series=per_file_time_series,
                    event_time_col=event_time_actual,
                    event_time_format=event_time_format,
                    reference_time=reference_time,
                )

            if valid is None:
                continue

            per_file_counts[path] = per_file_counts.get(path, 0) + 1

            pk_cols = _get_primary_key_columns(rule)
            if pk_cols:
                key_val = _primary_key_value(rule, valid)
                if key_val is not None:
                    pk_key = (pk_cols, key_val)
                    pk_counts[pk_key] = pk_counts.get(pk_key, 0) + 1

    header_intersection = header_intersection or set()

    header_lines = [json.dumps(info, ensure_ascii=False) for info in headers_info]
    _write_single_shard(os.path.join(dq_out, "headers", "per_file"), ".jsonl", header_lines)
    _write_single_shard(
        os.path.join(dq_out, "headers", "union"),
        ".json",
        [json.dumps(sorted(header_union))],
    )
    _write_single_shard(
        os.path.join(dq_out, "headers", "intersection"),
        ".json",
        [json.dumps(sorted(header_intersection))],
    )

    issue_pairs = [attach_dimension(issue) for issue in bad_issues]

    has_primary_key = any(_get_primary_key_columns(r) for r in rules)
    if has_primary_key:
        dup_keys = {k: v for k, v in pk_counts.items() if v > 1}
        for key, occ in sorted(dup_keys.items()):
            columns, key_value = key
            issue_pairs.append(
                (
                    "Duplication",
                    {
                        "dimension": "Duplication",
                        "reason": f"duplicate_primary_key:{key_value}",
                        "scenario": "Duplicate primary key values detected",
                        "detail": {
                            "primary_key_columns": list(columns),
                            "key_value": key_value,
                            "occurrences": occ,
                        },
                        "occurrence_count": max(occ - 1, 1),
                    },
                )
            )

    combiner = DimensionIssueSummary()
    accumulators: Dict[str, Any] = {}
    for dim, issue in issue_pairs:
        acc = accumulators.get(dim)
        if acc is None:
            acc = combiner.create_accumulator()
        acc = combiner.add_input(acc, issue)
        accumulators[dim] = acc

    summary_lines = [
        json.dumps(
            {
                "dimension": dim,
                "description": DIMENSION_DESCRIPTIONS.get(dim, DIMENSION_DESCRIPTIONS["Other"]),
                **combiner.extract_output(acc),
            },
            ensure_ascii=False,
        )
        for dim, acc in sorted(accumulators.items())
    ]
    issue_summary = complete_issue_dimensions([json.loads(line) for line in summary_lines])
    bad_issue_count = sum(entry.get("issue_count", 0) for entry in issue_summary)
    bad_issue_samples = bad_issues[:10]

    visualization_index: Dict[str, Dict[str, str]] = {}
    if visualizations_enabled:
        visualization_index, visualization_note = create_feature_visualizations(
            per_file_numeric_values, dq_out, column_descriptions
        )
    write_execution_log(
        dq_out,
        input_pattern,
        matched_files,
        per_file_counts,
        bad_issue_count,
        issue_summary,
        visualization_index,
        visualization_note,
    )
    write_quality_report(
        dq_out,
        input_pattern,
        matched_files,
        per_file_counts,
        bad_issue_count,
        bad_issue_samples,
        issue_summary,
        per_file_numeric_values,
        per_file_time_series,
        column_descriptions,
        visualization_index,
        visualization_note,
    )


def run(
    input_pattern: str,
    good_out: str,
    bad_out: str,
    dq_out: str,
    config_path: str,
    engine: str = "auto",
    augmentation_strategy: Optional[str] = None,
    augmentation_output: Optional[str] = None,
    augmentation_repeat: int = 1,
    augmentation_seed: Optional[int] = None,
) -> None:
    selected = (engine or "auto").lower()
    if selected not in {"auto", "beam", "sequential"}:
        raise ValueError("engine must be one of 'auto', 'beam', or 'sequential'")

    preloaded_config: Optional[Dict[str, Any]] = None
    augmented_files: Optional[List[str]] = None
    display_pattern = input_pattern

    if augmentation_strategy:
        if augmentation_strategy not in AUGMENTATION_STRATEGIES:
            raise ValueError(f"Unsupported augmentation strategy: {augmentation_strategy}")
        if not augmentation_output:
            raise ValueError("augmentation_output must be provided when augmentation_strategy is specified")
        if augmentation_repeat <= 0:
            raise ValueError("augmentation_repeat must be a positive integer")

        preloaded_config = load_config(config_path)
        rules = preloaded_config["rules"]
        original_files = sorted(glob.glob(input_pattern, recursive=True))
        if not original_files:
            raise FileNotFoundError(f"No files matched input pattern: {input_pattern}")
        augmented_files = generate_augmented_dataset(
            original_files,
            rules,
            augmentation_output,
            augmentation_strategy,
            repeat=max(1, augmentation_repeat),
            seed=augmentation_seed,
        )
        if not augmented_files:
            raise RuntimeError("Augmentation did not generate any rows; verify the strategy settings.")
        display_pattern = f"{input_pattern} [augmented:{augmentation_strategy}]"
        selected = "sequential"

    if selected == "beam":
        if not _HAVE_BEAM:
            raise RuntimeError("apache_beam is not installed; cannot run Beam engine")
        _run_with_beam(
            input_pattern,
            good_out,
            bad_out,
            dq_out,
            config_path,
            config=preloaded_config,
        )
        return

    if selected == "sequential" or not _HAVE_BEAM:
        _run_without_beam(
            input_pattern,
            good_out,
            bad_out,
            dq_out,
            config_path,
            config=preloaded_config,
            prefetched_files=augmented_files,
            input_display_pattern=display_pattern,
        )
        return

    _run_with_beam(
        input_pattern,
        good_out,
        bad_out,
        dq_out,
        config_path,
        config=preloaded_config,
    )


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_pattern", required=True,
        help='单个 glob，例如 "6GDALI_Datasets/EUR/6907619/*.csv" 或 "**/*performance*.csv"')
    ap.add_argument("--config", required=True, help="YAML 规则文件路径")
    ap.add_argument("--good_out", required=True, help="GOOD 输出前缀")
    ap.add_argument("--bad_out", required=True, help="BAD 输出前缀（JSONL）")
    ap.add_argument("--dq_out", required=True, help="DQ 汇总输出目录")
    ap.add_argument(
        "--engine",
        choices=["auto", "beam", "sequential"],
        default="auto",
        help="执行引擎：auto(默认，优先 Beam)、beam(强制 Beam)、sequential(无需 apache-beam)",
    )
    ap.add_argument(
        "--augmentation_strategy",
        choices=sorted(AUGMENTATION_STRATEGIES.keys()),
        help="可选的数据增强策略，生成合成数据后再执行 DQ",
    )
    ap.add_argument(
        "--augmentation_output",
        help="当指定增强策略时用于存放增强后 CSV 的目录",
    )
    ap.add_argument(
        "--augmentation_repeat",
        type=int,
        default=1,
        help="每个输入文件生成的增强副本次数（默认 1）",
    )
    ap.add_argument(
        "--augmentation_seed",
        type=int,
        help="数据增强的随机种子，便于复现",
    )
    args, _ = ap.parse_known_args()
    run(
        args.input_pattern,
        args.good_out,
        args.bad_out,
        args.dq_out,
        args.config,
        engine=args.engine,
        augmentation_strategy=args.augmentation_strategy,
        augmentation_output=args.augmentation_output,
        augmentation_repeat=args.augmentation_repeat,
        augmentation_seed=args.augmentation_seed,
    )