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
import datetime as dt
import math
import shutil
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Iterable

import glob

try:
    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    matplotlib = None  # type: ignore
    plt = None  # type: ignore

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
    "numeric_cols": [],
    "enums": {},
    "ranges": {},
    "primary_key": None,
    "event_time_col": None,
    "event_time_format": "auto",     # iso | epoch_s | epoch_ms | auto
    "freshness_slo_hours": None,
    "max_future_hours": None,
    "time_epoch_bounds": None,       # {"min": <epoch_s>, "max": <epoch_s>}
    "reference_keys": {},            # {"path": "...csv", "column": "...", "target_col": "..."}
    "numeric_unit_parsers": {},      # 列 -> {type:'mem', base:1024, out_unit:'MiB'}
    "metadata_path": None,
    "metadata_targets": [],          # 文件名（或相对路径）列表，用于从元数据中提取列说明
}

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
_COLUMN_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _parse_metadata_text(lines: Iterable[str]) -> Dict[str, Dict[str, str]]:
    """解析元数据文本，返回 {文件名: {列名: 描述}}。列名统一为小写。"""

    files: List[str] = []
    mapping: Dict[str, Dict[str, str]] = {}
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
            files = [os.path.basename(part.strip()) for part in value.split(",") if part.strip()]
            for f in files:
                mapping.setdefault(f, {})
            continue

        if not files:
            # 还未遇到 file(s) 说明，此处多为段落描述
            continue

        if not value:
            # 章节标题或其他说明
            continue

        if not _COLUMN_NAME_RE.match(key):
            # 过滤掉“非列名”的行（例如长句子）
            continue

        col_name = key.strip().lower()
        for f in files:
            mapping.setdefault(f, {})[col_name] = value

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
    for r in rules:
        rr = dict(DEFAULT_RULE)
        if r:
            rr.update(r)
        pats = rr.get("patterns") or rr.get("pattern") or [".*"]
        rr["patterns"] = [pats] if isinstance(pats, str) else list(pats)
        rr["required_cols"] = rr.get("required_cols", []) or []
        rr["numeric_cols"]  = rr.get("numeric_cols", [])  or []
        rr["enums"]         = rr.get("enums", {})         or {}
        rr["ranges"]        = rr.get("ranges", {})        or {}
        rr["numeric_unit_parsers"] = rr.get("numeric_unit_parsers", {}) or {}

        meta_path = rr.get("metadata_path")
        targets = rr.get("metadata_targets") or []
        metadata_by_file: Dict[str, Dict[str, str]] = {}
        if meta_path:
            if meta_path not in metadata_cache:
                metadata_cache[meta_path] = parse_metadata_descriptions(meta_path)
            meta_map = metadata_cache[meta_path]
            selected = targets or list(meta_map.keys())
            for target in selected:
                base = os.path.basename(target)
                if base in meta_map:
                    metadata_by_file[base] = meta_map[base]
        if metadata_by_file:
            rr["metadata_by_file"] = metadata_by_file
            # 将元数据列合并进 required_cols，避免遗漏
            meta_cols = sorted({col for cols in metadata_by_file.values() for col in cols.keys()})
            for col in meta_cols:
                if col not in rr["required_cols"]:
                    rr["required_cols"].append(col)
        else:
            rr["metadata_by_file"] = {}
        norm.append(rr)
    return {"rules": norm}

def pick_rule(rules: List[Dict[str, Any]], path: str) -> Dict[str, Any]:
    for r in rules:
        if any(re.search(p, path) for p in r["patterns"]):
            return r
    return DEFAULT_RULE

# -----------------------------
# 时间解析
# -----------------------------
def parse_event_time(val, fmt: str = "auto") -> dt.datetime:
    if val is None or str(val).strip() == "":
        raise ValueError("empty timestamp")
    try:
        x = float(val)
        is_numeric = True
    except Exception:
        is_numeric = False

    if fmt == "iso" or (fmt == "auto" and not is_numeric):
        return _parse_iso_datetime(str(val))

    if fmt == "epoch_s":
        seconds = x
    elif fmt == "epoch_ms":
        seconds = x / 1000.0
    else:
        ax = abs(x)
        if ax >= 1e18: seconds = x / 1_000_000_000.0  # ns
        elif ax >= 1e15: seconds = x / 1_000_000.0    # µs
        elif ax >= 1e12: seconds = x / 1_000.0        # ms
        else: seconds = x                              # s
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
        # dialect = csv.Sniffer().sniff(sample)
        # Restrict sniffed delimiters so that spaces inside column names (e.g. "abc 12")
        # are not misinterpreted as separators.
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
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
) -> Tuple[Optional[RowCtx], List[Dict[str, Any]]]:
    inc = increment or (lambda _name: None)
    issues: List[Dict[str, Any]] = []
    data = rc.data

    metadata_by_file = rule.get("metadata_by_file") or {}
    meta_cols = metadata_by_file.get(os.path.basename(rc.file))
    if meta_cols:
        missing = [col for col in meta_cols.keys() if col not in rc.header]
        if missing:
            for col in missing:
                inc("metadata_mismatch")
                issues.append(
                    {
                        "file": rc.file,
                        "row": 0,
                        "reason": f"missing_metadata_column:{col}",
                    }
                )
            return None, issues

    for col in rule["required_cols"]:
        if col not in rc.header:
            issues.append(
                {
                    "file": rc.file,
                    "row": 0,
                    "reason": f"missing_required_column:{col}",
                }
            )
            return None, issues

    for col in rule["required_cols"]:
        if data.get(col) in (None, "", "NaN"):
            inc("nulls")
            issues.append(
                {
                    "file": rc.file,
                    "row": rc.rownum,
                    "reason": f"null_required:{col}",
                }
            )
            return None, issues

    for col in rule["numeric_cols"]:
        if col not in rc.header:
            continue
        val = data.get(col, "")
        try:
            parsed_val = parse_numeric_with_units(val, col, rule)
        except Exception:
            inc("bad_numeric")
            issues.append(
                {
                    "file": rc.file,
                    "row": rc.rownum,
                    "reason": f"bad_numeric:{col}={val}",
                }
            )
            return None, issues
        rng = rule["ranges"].get(col)
        if rng:
            lo, hi = rng
            if (lo is not None and parsed_val < lo) or (hi is not None and parsed_val > hi):
                inc("bad_range")
                issues.append(
                    {
                        "file": rc.file,
                        "row": rc.rownum,
                        "reason": f"out_of_range:{col}={parsed_val} not [{lo},{hi}]",
                    }
                )
                return None, issues

    for col, allowed in rule["enums"].items():
        if col not in rc.header:
            continue
        val = data.get(col, "")
        if val != "" and val not in allowed:
            inc("bad_enum")
            issues.append(
                {
                    "file": rc.file,
                    "row": rc.rownum,
                    "reason": f"bad_enum:{col}={val}",
                }
            )
            return None, issues

    rconf = rule.get("reference_keys") or {}
    target_col = rconf.get("target_col")
    if target_col and refset is not None and target_col in rc.header:
        val = data.get(target_col)
        if val not in refset:
            inc("bad_ref")
            issues.append(
                {
                    "file": rc.file,
                    "row": rc.rownum,
                    "reason": f"fk_missing:{target_col}={val}",
                }
            )
            return None, issues

    et_col = rule.get("event_time_col")
    slo_h = rule.get("freshness_slo_hours")
    if et_col and et_col in rc.header and (
        slo_h or rule.get("max_future_hours") or rule.get("time_epoch_bounds")
    ):
        val = data.get(et_col)
        try:
            et_fmt = (rule.get("event_time_format") or "auto").lower()
            ts = parse_event_time(val, et_fmt)
            now = dt.datetime.now(dt.timezone.utc)

            if slo_h is not None:
                age_h = (now - ts).total_seconds() / 3600.0
                if age_h > float(slo_h):
                    inc("bad_freshness")
                    issues.append(
                        {
                            "file": rc.file,
                            "row": rc.rownum,
                            "reason": f"stale_event:{et_col} age_h={age_h:.2f} slo_h={slo_h}",
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
                            "reason": f"future_event:{et_col} future_h={future_h:.2f} max={mf}",
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

        except Exception:
            inc("bad_numeric")
            issues.append(
                {
                    "file": rc.file,
                    "row": rc.rownum,
                    "reason": f"bad_timestamp:{et_col}={val}",
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
        numeric_cols = rule.get("numeric_cols") or []
        if numeric_cols and col not in numeric_cols:
            return None
        v = rc.data.get(col)
        if v in (None, "", "NaN"):
            return None
        try:
            return parse_numeric_with_units(v, col, rule)
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
        }

    count = len(values)
    min_val = min(values)
    max_val = max(values)
    mean_val = sum(values) / count
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

    return {
        "count": count,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
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
) -> None:
    for col in rule.get("numeric_cols", []):
        if col not in rc.header:
            continue
        val = rc.data.get(col)
        if val in (None, "", "NaN"):
            continue
        try:
            parsed_val = parse_numeric_with_units(val, col, rule)
        except Exception:
            continue
        if global_values is not None:
            global_values.setdefault(col, []).append(parsed_val)
        if per_file_values is not None and file_path is not None:
            per_file_values.setdefault(file_path, {}).setdefault(col, []).append(parsed_val)


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
    if plt is None:
        raise RuntimeError(
            "matplotlib is required to generate visualizations. Install matplotlib to enable this feature."
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = max(10, min(50, int(math.sqrt(len(values)))))
    ax.hist(values, bins=bins, color="#2a9d8f", edgecolor="black", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_boxplot(
    values: List[float],
    output_path: str,
    title: str,
    xlabel: str,
    summary: Dict[str, Any],
) -> None:
    if not values:
        return
    if plt is None:
        raise RuntimeError(
            "matplotlib is required to generate visualizations. Install matplotlib to enable this feature."
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
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(
        values,
        vert=False,
        patch_artist=True,
        boxprops={"facecolor": "#2a9d8f", "alpha": 0.6},
        medianprops={"color": "#e76f51", "linewidth": 2},
        whiskerprops={"color": "#264653"},
        capprops={"color": "#264653"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#e76f51",
            "markeredgecolor": "#264653",
            "alpha": 0.6,
        },
    )
    ax.set_title(f"{title} – outlier analysis")
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.text(
        0.98,
        0.02,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def create_feature_visualizations(
    per_file_numeric_values: Dict[str, Dict[str, List[float]]],
    dq_out: str,
    column_descriptions: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, str]], Optional[str]]:
    if not per_file_numeric_values:
        return {}, "No numeric columns detected; skipping feature visualizations."
    if plt is None:
        return {}, "matplotlib is not installed; feature visualizations were skipped."

    visualization_root = os.path.join(dq_out, "visualizations")
    generated: Dict[str, Dict[str, str]] = {}

    for file_path, col_values in sorted(per_file_numeric_values.items()):
        safe_file = _safe_component_name(file_path)
        for col, values in sorted(col_values.items()):
            if not values:
                continue
            safe_col = _safe_component_name(col)
            title = f"{os.path.basename(file_path)} – {col} distribution"
            desc = column_descriptions.get(col)
            if desc:
                title = f"{title}\n{desc}"
            output_path = os.path.join(visualization_root, safe_file, f"{safe_col}.png")
            summary = summarize_numeric_values(values)
            _plot_histogram(values, output_path, title, col)
            generated.setdefault(file_path, {})[col] = os.path.relpath(output_path, dq_out)

            outlier_path = os.path.join(
                visualization_root, safe_file, f"{safe_col}_outliers.png"
            )
            _plot_boxplot(values, outlier_path, title, col, summary)
            generated[file_path][f"{col}_outliers"] = os.path.relpath(outlier_path, dq_out)

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
        for col, values in sorted((per_file_numeric_values.get(file_path) or {}).items()):
            stats = summarize_numeric_values(values)
            outlier_bounds = stats.get("outlier_bounds") or {}
            numeric_summary[col] = {
                "count": stats.get("count"),
                "min": stats.get("min"),
                "max": stats.get("max"),
                "mean": stats.get("mean"),
                "quantiles": stats.get("quantiles"),
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
    files: List[str], rules: List[Dict[str, Any]]
) -> Dict[str, Dict[str, List[float]]]:
    per_file: Dict[str, Dict[str, List[float]]] = {}
    for path in files:
        _, rows, _ = read_csv_file(path)
        rule = pick_rule(rules, path)
        for rc in rows:
            _accumulate_numeric_values(rc, rule, None, per_file, path)
    return per_file


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


def classify_issue(reason: str) -> Tuple[str, str]:
    """Map a failure reason to a (dimension, scenario) tuple."""
    if not reason:
        return "Other", "Uncategorized issue"

    r = reason.lower()
    if r.startswith("missing_required_column"):
        return "Completeness", "Required column missing from header"
    if r.startswith("missing_metadata_column"):
        return "Completeness", "Column listed in metadata is missing from header"
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
def _run_with_beam(input_pattern: str, good_out: str, bad_out: str, dq_out: str, config_path: str):
    cfg = load_config(config_path)
    rules = cfg["rules"]
    column_descriptions = gather_column_descriptions(rules)

    matched_files = sorted(glob.glob(input_pattern, recursive=True))
    if not matched_files:
        raise FileNotFoundError(f"No files matched input pattern: {input_pattern}")

    # 外键集合（可按需扩展为多集合 keyed side input）
    refset_local: Optional[set] = None
    for r in rules:
        rconf = r.get("reference_keys") or {}
        if rconf.get("path") and rconf.get("column"):
            refset_local = load_reference_keys(rconf["path"], rconf["column"])
            break

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

        with_rules = parsed.good | "AttachRule" >> beam.Map(lambda rc: (rc, pick_rule(rules, rc.file)))

        # side input for FK
        if refset_local is not None:
            ref_pc = p | "CreateRefSet" >> beam.Create([refset_local])
            ref_view = beam.pvalue.AsSingleton(ref_pc)
        else:
            ref_view = None

        class ValidateWrapper(beam.DoFn):
            def process(self, pair, refset=None):
                rc, rule = pair
                validator = ValidateRow(rule)
                if refset is None:
                    yield from validator.process(rc, None)
                else:
                    yield from validator.process(rc, refset)

        if ref_view is None:
            validated = (
                with_rules
                | "ValidateRows" >> beam.ParDo(ValidateWrapper()).with_outputs(BAD_TAG, main="good")
            )
        else:
            validated = (
                with_rules
                | "ValidateRows" >> beam.ParDo(ValidateWrapper(), refset=ref_view).with_outputs(BAD_TAG, main="good")
            )

        # BAD 输出（JSONL）
        bad = (parsed.bad, validated.bad) | "FlattenBad" >> beam.Flatten()
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
        if any(r.get("primary_key") for r in rules):
            def pk_pairs(pair: Tuple[RowCtx, Dict[str, Any]]):
                rc, rule = pair
                pk = rule.get("primary_key")
                if not pk or pk not in rc.header:
                    return
                key = rc.data.get(pk)
                if key not in (None, "", "NaN"):
                    yield (key, rc.file)

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
                            "reason": f"duplicate_primary_key:{kv[0]}",
                            "scenario": "Duplicate primary key values detected",
                            "detail": {"primary_key": kv[0], "occurrences": kv[1]},
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
    visualization_index: Dict[str, Dict[str, str]] = {}
    visualization_note: Optional[str] = None
    visualizations_enabled, size_note = _visualizations_allowed(matched_files)
    if not visualizations_enabled:
        visualization_note = size_note
    else:
        per_file_numeric_values = collect_numeric_values_by_file(matched_files, rules)
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
    input_pattern: str, good_out: str, bad_out: str, dq_out: str, config_path: str
) -> None:
    cfg = load_config(config_path)
    rules = cfg["rules"]
    column_descriptions = gather_column_descriptions(rules)

    refset_local: Optional[set] = None
    for r in rules:
        rconf = r.get("reference_keys") or {}
        if rconf.get("path") and rconf.get("column"):
            refset_local = load_reference_keys(rconf["path"], rconf["column"])
            break

    matched_files = sorted(glob.glob(input_pattern, recursive=True))
    if not matched_files:
        raise FileNotFoundError(f"No files matched input pattern: {input_pattern}")

    os.makedirs(dq_out, exist_ok=True)

    visualizations_enabled, visualization_note = _visualizations_allowed(matched_files)

    headers_info: List[Dict[str, Any]] = []
    header_union: set = set()
    header_intersection: Optional[set] = None
    bad_issues: List[Dict[str, Any]] = []
    per_file_counts: Dict[str, int] = {}
    per_file_numeric_values: Dict[str, Dict[str, List[float]]] = {}
    pk_counts: Dict[str, int] = {}

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

        for rc in rows:
            valid, issues = validate_row_against_rule(rc, rule, refset_local)
            if issues:
                bad_issues.extend(issues)

            if visualizations_enabled:
                _accumulate_numeric_values(
                    rc,
                    rule,
                    global_values=None,
                    per_file_values=per_file_numeric_values,
                    file_path=path,
                )

            if valid is None:
                continue

            per_file_counts[path] = per_file_counts.get(path, 0) + 1

            pk = rule.get("primary_key")
            if pk and pk in valid.header:
                key_val = valid.data.get(pk)
                if key_val not in (None, "", "NaN"):
                    pk_counts[key_val] = pk_counts.get(key_val, 0) + 1

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

    has_primary_key = any(r.get("primary_key") for r in rules)
    if has_primary_key:
        dup_keys = {k: v for k, v in pk_counts.items() if v > 1}
        for key, occ in sorted(dup_keys.items()):
            issue_pairs.append(
                (
                    "Duplication",
                    {
                        "dimension": "Duplication",
                        "reason": f"duplicate_primary_key:{key}",
                        "scenario": "Duplicate primary key values detected",
                        "detail": {"primary_key": key, "occurrences": occ},
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
) -> None:
    selected = (engine or "auto").lower()
    if selected not in {"auto", "beam", "sequential"}:
        raise ValueError("engine must be one of 'auto', 'beam', or 'sequential'")

    if selected == "beam":
        if not _HAVE_BEAM:
            raise RuntimeError("apache_beam is not installed; cannot run Beam engine")
        _run_with_beam(input_pattern, good_out, bad_out, dq_out, config_path)
        return

    if selected == "sequential" or not _HAVE_BEAM:
        _run_without_beam(input_pattern, good_out, bad_out, dq_out, config_path)
        return

    _run_with_beam(input_pattern, good_out, bad_out, dq_out, config_path)


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
    args, _ = ap.parse_known_args()
    run(
        args.input_pattern,
        args.good_out,
        args.bad_out,
        args.dq_out,
        args.config,
        engine=args.engine,
    )