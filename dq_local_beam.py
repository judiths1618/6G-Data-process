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
import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Iterable

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io import fileio
from apache_beam.metrics.metric import Metrics
from apache_beam.transforms.combiners import Sample

import yaml
from dateutil import parser as dtparse

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

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
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
        t = dtparse.parse(str(val))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        else:
            t = t.astimezone(dt.timezone.utc)
        return t

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
# 读 CSV（带表头）+ 发 headers
# -----------------------------
class ReadCSVWithHeader(beam.DoFn):
    bad_parse   = Metrics.counter("dq", "bad_parse")
    empty_files = Metrics.counter("dq", "empty_files")

    def process(self, rf: fileio.ReadableFile):
        path = rf.metadata.path
        text = rf.read_utf8()
        if not text.strip():
            self.empty_files.inc()
            yield beam.pvalue.TaggedOutput(BAD_TAG, {"file": path, "row": 0, "reason": "empty_file"})
            return

        buf = io.StringIO(text)
        sample = buf.read(2048)
        buf.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(buf, dialect=dialect)

        if not reader.fieldnames:
            self.bad_parse.inc()
            yield beam.pvalue.TaggedOutput(BAD_TAG, {"file": path, "row": 0, "reason": "no_header"})
            return

        header = [h.strip() for h in reader.fieldnames]
        yield beam.pvalue.TaggedOutput(HEADERS_TAG, {"file": path, "header": header})

        for i, row in enumerate(reader, start=1):
            try:
                cleaned = { (k.strip() if k else k): (v.strip() if isinstance(v, str) else v)
                            for k, v in row.items() }
                yield RowCtx(file=path, rownum=i, header=header, data=cleaned)
            except Exception as e:
                self.bad_parse.inc()
                yield beam.pvalue.TaggedOutput(BAD_TAG, {
                    "file": path, "row": i, "reason": f"parse_error:{e}"
                })

# -----------------------------
# 校验器
# -----------------------------
class ValidateRow(beam.DoFn):
    nulls         = Metrics.counter("dq", "null_required")
    bad_enum      = Metrics.counter("dq", "bad_enum")
    bad_range     = Metrics.counter("dq", "bad_range")
    bad_numeric   = Metrics.counter("dq", "bad_numeric")
    bad_ref       = Metrics.counter("dq", "bad_ref_integrity")
    bad_freshness = Metrics.counter("dq", "bad_freshness")
    metadata_mismatch = Metrics.counter("dq", "metadata_mismatch")

    def __init__(self, rules: Dict[str, Any]):
        self.rules = rules

    def process(self, rc: RowCtx, refset: Optional[set] = None):
        d = rc.data
        rules = self.rules

        # -1) 元数据列校验（如有）
        metadata_by_file = rules.get("metadata_by_file") or {}
        meta_cols = metadata_by_file.get(os.path.basename(rc.file))
        if meta_cols:
            missing = [col for col in meta_cols.keys() if col not in rc.header]
            if missing:
                for col in missing:
                    self.metadata_mismatch.inc()
                    yield beam.pvalue.TaggedOutput(BAD_TAG, {
                        "file": rc.file,
                        "row": 0,
                        "reason": f"missing_metadata_column:{col}",
                    })
                return

        # 0) 必备列存在
        for col in rules["required_cols"]:
            if col not in rc.header:
                yield beam.pvalue.TaggedOutput(BAD_TAG, {
                    "file": rc.file, "row": 0, "reason": f"missing_required_column:{col}"
                })
                return

        # 1) 必备列非空
        for col in rules["required_cols"]:
            if d.get(col) in (None, "", "NaN"):
                self.nulls.inc()
                yield beam.pvalue.TaggedOutput(BAD_TAG, {
                    "file": rc.file, "row": rc.rownum, "reason": f"null_required:{col}"
                })
                return

        # 2) 数值解析 + 范围（numeric_cols 为“可选”）
        for col in rules["numeric_cols"]:
            if col not in rc.header:
                continue
            val = d.get(col, "")
            try:
                f = parse_numeric_with_units(val, col, rules)
            except Exception:
                self.bad_numeric.inc()
                yield beam.pvalue.TaggedOutput(BAD_TAG, {
                    "file": rc.file, "row": rc.rownum, "reason": f"bad_numeric:{col}={val}"
                })
                return
            rng = rules["ranges"].get(col)
            if rng:
                lo, hi = rng
                if (lo is not None and f < lo) or (hi is not None and f > hi):
                    self.bad_range.inc()
                    yield beam.pvalue.TaggedOutput(BAD_TAG, {
                        "file": rc.file, "row": rc.rownum,
                        "reason": f"out_of_range:{col}={f} not [{lo},{hi}]"
                    })
                    return

        # 3) 枚举
        for col, allowed in rules["enums"].items():
            if col not in rc.header:
                continue
            val = d.get(col, "")
            if val != "" and val not in allowed:
                self.bad_enum.inc()
                yield beam.pvalue.TaggedOutput(BAD_TAG, {
                    "file": rc.file, "row": rc.rownum, "reason": f"bad_enum:{col}={val}"
                })
                return

        # 4) 外键（如有）
        rconf = rules.get("reference_keys") or {}
        target_col = rconf.get("target_col")
        if target_col and refset is not None and target_col in rc.header:
            val = d.get(target_col)
            if val not in refset:
                self.bad_ref.inc()
                yield beam.pvalue.TaggedOutput(BAD_TAG, {
                    "file": rc.file, "row": rc.rownum, "reason": f"fk_missing:{target_col}={val}"
                })
                return

        # 5) 新鲜度/未来时间/epoch 边界
        et_col = rules.get("event_time_col")
        slo_h  = rules.get("freshness_slo_hours")
        if et_col and et_col in rc.header and (slo_h or rules.get("max_future_hours") or rules.get("time_epoch_bounds")):
            val = d.get(et_col)
            try:
                et_fmt = (rules.get("event_time_format") or "auto").lower()
                ts = parse_event_time(val, et_fmt)
                now = dt.datetime.now(dt.timezone.utc)

                if slo_h is not None:
                    age_h = (now - ts).total_seconds() / 3600.0
                    if age_h > float(slo_h):
                        self.bad_freshness.inc()
                        yield beam.pvalue.TaggedOutput(BAD_TAG, {
                            "file": rc.file, "row": rc.rownum,
                            "reason": f"stale_event:{et_col} age_h={age_h:.2f} slo_h={slo_h}"
                        })
                        return

                mf = rules.get("max_future_hours")
                if mf is not None:
                    future_h = (ts - now).total_seconds() / 3600.0
                    if future_h > float(mf):
                        self.bad_freshness.inc()
                        yield beam.pvalue.TaggedOutput(BAD_TAG, {
                            "file": rc.file, "row": rc.rownum,
                            "reason": f"future_event:{et_col} future_h={future_h:.2f} max={mf}"
                        })
                        return

                bounds = rules.get("time_epoch_bounds")
                if bounds and str(val).strip() != "":
                    try:
                        x = float(val)
                        lo = bounds.get("min"); hi = bounds.get("max")
                        if (lo is not None and x < float(lo)) or (hi is not None and x > float(hi)):
                            self.bad_range.inc()
                            yield beam.pvalue.TaggedOutput(BAD_TAG, {
                                "file": rc.file, "row": rc.rownum,
                                "reason": f"time_epoch_out_of_range:{val} not [{lo},{hi}]"
                            })
                            return
                    except Exception:
                        pass

            except Exception:
                self.bad_numeric.inc()
                yield beam.pvalue.TaggedOutput(BAD_TAG, {
                    "file": rc.file, "row": rc.rownum, "reason": f"bad_timestamp:{et_col}={val}"
                })
                return

        yield rc

# -----------------------------
# 工具
# -----------------------------
class ToCsvLine(beam.DoFn):
    def process(self, rc: RowCtx):
        yield ",".join(str(rc.data.get(c, "")) for c in rc.header)

def load_reference_keys(path: str, column: str) -> set:
    with open(path, "r") as f:
        rdr = csv.DictReader(f)
        return { (row.get(column) or "").strip() for row in rdr }

def _percentiles(xs: List[float], perc=(0, 25, 50, 75, 100)) -> List[float]:
    if not xs:
        return []
    xs = sorted(xs)
    n = len(xs)
    out = []
    for p in perc:
        k = (n - 1) * (p / 100.0)
        f = math.floor(k); c = math.ceil(k)
        out.append(xs[int(k)] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f))
    return out

# === 关键修复：按“每条记录对应的规则”做数值统计 ===
def numeric_profiles_from_pairs(pairs_pcoll: beam.PCollection, col: str):
    """
    输入: PCollection[(RowCtx, rule_dict)]
    对指定列做统计；解析单位使用每条记录对应的 rule_dict。
    """
    def cast_with_rule(pair):
        rc, rule = pair
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
    minv  = vals | f"Min_{col}"   >> beam.CombineGlobally(min).without_defaults()
    maxv  = vals | f"Max_{col}"   >> beam.CombineGlobally(max).without_defaults()
    mean  = vals | f"Mean_{col}"  >> beam.combiners.Mean.Globally().without_defaults()
    qtls  = (
        vals
        | f"Sample_{col}" >> Sample.FixedSizeGlobally(10_000)
        | f"Qtls_{col}"   >> beam.Map(lambda xs: _percentiles(xs or [], (0, 25, 50, 75, 100)))
    )
    bounds = (
        qtls
        | f"Bounds_{col}" >> beam.Map(
            lambda qs: {
                "quantiles": qs,
                "q1": qs[1] if len(qs) >= 4 else None,
                "q3": qs[3] if len(qs) >= 4 else None,
                "iqr": (qs[3] - qs[1]) if len(qs) >= 4 else None,
            }
        )
        | f"OutlierBounds_{col}" >> beam.Map(
            lambda info: {
                **info,
                "lower": (info["q1"] - 1.5 * info["iqr"]) if info["iqr"] is not None else None,
                "upper": (info["q3"] + 1.5 * info["iqr"]) if info["iqr"] is not None else None,
            }
        )
    )

    outlier_count = (
        vals
        | f"FlagOutliers_{col}" >> beam.Map(
            lambda x, info: 1
            if info and info.get("lower") is not None and (x < info["lower"] or x > info["upper"])
            else 0,
            info=beam.pvalue.AsSingleton(bounds),
        )
        | f"SumOutliers_{col}" >> beam.CombineGlobally(sum)
    )

    return {
        "count": count,
        "min": minv,
        "max": maxv,
        "mean": mean,
        "quantiles": qtls,
        "outlier_bounds": bounds,
        "outlier_count": outlier_count,
    }

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


class DimensionIssueSummary(beam.CombineFn):
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
def run(input_pattern: str, good_out: str, bad_out: str, dq_out: str, config_path: str):
    cfg = load_config(config_path)
    rules = cfg["rules"]
    column_descriptions = gather_column_descriptions(rules)

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

        # GOOD 输出
        _ = (
            validated.good
            | "ToCsv" >> beam.ParDo(ToCsvLine())
            | "WriteGood" >> beam.io.WriteToText(good_out, file_name_suffix=".csv", num_shards=1)
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
            _ = (
                dup_pk
                | "WritePKDup" >> beam.io.WriteToText(os.path.join(dq_out, "pk_duplicates"),
                                                      file_name_suffix=".txt", num_shards=1)
            )
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

        # 新鲜度汇总
        if any(r.get("event_time_col") and r.get("freshness_slo_hours") for r in rules):
            def to_age(pair):
                rc, rule = pair
                col = rule.get("event_time_col")
                if not col or col not in rc.header:
                    return
                v = rc.data.get(col)
                if not v:
                    return
                try:
                    ts = parse_event_time(v, (rule.get("event_time_format") or "auto"))
                    age_h = (dt.datetime.now(dt.timezone.utc) - ts).total_seconds() / 3600.0
                    yield (rc.file, age_h)
                except Exception:
                    return

            ages = with_rules | "ToAges" >> beam.FlatMap(to_age)
            max_age = ages | "MaxAgePerFile" >> beam.CombinePerKey(max)
            _ = (
                max_age
                | "AgeJSON" >> beam.Map(lambda kv: json.dumps({"file": kv[0], "max_age_hours": kv[1]}))
                | "WriteAge" >> beam.io.WriteToText(os.path.join(dq_out, "freshness"),
                                                    file_name_suffix=".jsonl", num_shards=1)
            )

        # === 数值分布：逐行附规则后统计（修复“统计为空”） ===
        stats_pairs = (
            validated.good
            | "AttachRuleForStats" >> beam.Map(lambda rc: (rc, pick_rule(rules, rc.file)))
        )
        num_cols = union_numeric_cols(rules)
        for col in num_cols:
            prof = numeric_profiles_from_pairs(stats_pairs, col)
            _ = (
                prof["count"]
                | f"FmtCount_{col}" >> beam.Map(
                    lambda x, column=col, desc=column_descriptions.get(col): json.dumps(
                        {
                            "column": column,
                            "metric": "count",
                            "value": x,
                            "description": desc,
                        }
                    )
                )
                | f"WriteCount_{col}" >> beam.io.WriteToText(
                    os.path.join(dq_out, f"num_{col}_count"), file_name_suffix=".txt", num_shards=1
                )
            )
            _ = (
                prof["min"]
                | f"FmtMin_{col}" >> beam.Map(
                    lambda x, column=col, desc=column_descriptions.get(col): json.dumps(
                        {
                            "column": column,
                            "metric": "min",
                            "value": x,
                            "description": desc,
                        }
                    )
                )
                | f"WriteMin_{col}" >> beam.io.WriteToText(
                    os.path.join(dq_out, f"num_{col}_min"), file_name_suffix=".txt", num_shards=1
                )
            )
            _ = (
                prof["max"]
                | f"FmtMax_{col}" >> beam.Map(
                    lambda x, column=col, desc=column_descriptions.get(col): json.dumps(
                        {
                            "column": column,
                            "metric": "max",
                            "value": x,
                            "description": desc,
                        }
                    )
                )
                | f"WriteMax_{col}" >> beam.io.WriteToText(
                    os.path.join(dq_out, f"num_{col}_max"), file_name_suffix=".txt", num_shards=1
                )
            )
            _ = (
                prof["mean"]
                | f"FmtMean_{col}" >> beam.Map(
                    lambda x, column=col, desc=column_descriptions.get(col): json.dumps(
                        {
                            "column": column,
                            "metric": "mean",
                            "value": x,
                            "description": desc,
                        }
                    )
                )
                | f"WriteMean_{col}" >> beam.io.WriteToText(
                    os.path.join(dq_out, f"num_{col}_mean"), file_name_suffix=".txt", num_shards=1
                )
            )
            _ = (
                prof["quantiles"]
                | f"QtlsToJSON_{col}" >> beam.Map(
                    lambda qs, column=col, desc=column_descriptions.get(col): json.dumps(
                        {
                            "column": column,
                            "metric": "quantiles",
                            "values": qs,
                            "description": desc,
                        }
                    )
                )
                | f"WriteQtls_{col}" >> beam.io.WriteToText(
                    os.path.join(dq_out, f"num_{col}_quantiles"), file_name_suffix=".jsonl", num_shards=1
                )
            )
            _ = (
                prof["outlier_bounds"]
                | f"OutlierBoundsJSON_{col}" >> beam.Map(
                    lambda info, column=col, desc=column_descriptions.get(col): json.dumps(
                        {
                            "column": column,
                            "metric": "outlier_bounds",
                            "description": desc,
                            "lower": info.get("lower"),
                            "upper": info.get("upper"),
                            "iqr": info.get("iqr"),
                            "q1": info.get("q1"),
                            "q3": info.get("q3"),
                            "quantiles": info.get("quantiles"),
                        }
                    )
                )
                | f"WriteOutlierBounds_{col}" >> beam.io.WriteToText(
                    os.path.join(dq_out, f"num_{col}_outlier_bounds"), file_name_suffix=".jsonl", num_shards=1
                )
            )
            _ = (
                prof["outlier_count"]
                | f"OutlierCountJSON_{col}" >> beam.Map(
                    lambda cnt, total, column=col, desc=column_descriptions.get(col): json.dumps(
                        {
                            "column": column,
                            "metric": "outliers",
                            "outlier_count": cnt,
                            "total_count": total,
                            "outlier_ratio": (cnt / total) if total else 0.0,
                            "description": desc,
                        }
                    ),
                    total=beam.pvalue.AsSingleton(prof["count"]),
                )
                | f"WriteOutlierCount_{col}" >> beam.io.WriteToText(
                    os.path.join(dq_out, f"num_{col}_outliers"), file_name_suffix=".jsonl", num_shards=1
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

        metadata_entries = []
        for r in rules:
            for fname, cols in (r.get("metadata_by_file") or {}).items():
                metadata_entries.append(
                    {
                        "pattern": r.get("patterns"),
                        "file": fname,
                        "columns": [
                            {"name": col, "description": desc} for col, desc in sorted(cols.items())
                        ],
                    }
                )
        if metadata_entries:
            _ = (
                p
                | "CreateMetadataEntries" >> beam.Create(metadata_entries)
                | "MetadataToJSON" >> beam.Map(lambda x: json.dumps(x, ensure_ascii=False))
                | "WriteMetadataSummary" >> beam.io.WriteToText(
                    os.path.join(dq_out, "metadata_columns"), file_name_suffix=".jsonl", num_shards=1
                )
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
    args, _ = ap.parse_known_args()
    run(args.input_pattern, args.good_out, args.bad_out, args.dq_out, args.config)
