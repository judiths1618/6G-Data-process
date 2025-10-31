
from __future__ import annotations

"""
dq_local_beam.py  — minimal, self-contained, NO external Beam dependency.

Features
- Auto-rules if config_path is None or "AUTO"
- Local CSV glob input
- Basic data quality report with:
  - Per-file good row counts
  - Numeric column detection + distribution (histogram) stats
  - Header union/intersection
  - Optional metadata descriptions (via env DQ_METADATA_PATH; simple "key: value" pairs per line with 'file:' sections)
- Outputs:
  - dq/headers/{per_file.jsonl, union.json, intersection.json}
  - dq/quality_report-00000-of-00001.json
  - logs/pipeline-00000-of-00001.log
  - (No "good"/"bad" partitions in this minimal version; counts only)

This is a simplified drop-in "sequential engine" replacement that your dashboard can call.
"""

import os
import io
import csv
import glob
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import datetime as dt

# -----------------------------
# Helpers
# -----------------------------

def _write_single_shard(prefix: str, suffix: str, lines: List[str]) -> str:
    path = f"{prefix}-00000-of-00001{suffix}"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fp:
        for line in lines:
            fp.write(line)
            if not line.endswith("\n"):
                fp.write("\n")
    return str(p)

def _single_shard_path(prefix: str, suffix: str) -> str:
    return f"{prefix}-00000-of-00001{suffix}"

def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and (v != v):  # NaN
        return True
    if isinstance(v, str) and (not v.strip() or v.strip().lower() == "nan"):
        return True
    return False

def _try_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None

# -----------------------------
# CSV parsing
# -----------------------------

@dataclass
class RowCtx:
    file: str
    rownum: int
    header: List[str]
    data: Dict[str, Any]

def _parse_csv_text(path: str, text: str) -> Tuple[List[str], List[RowCtx], List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    if not text.strip():
        issues.append({"file": path, "row": 0, "reason": "empty_file"})
        return [], [], issues
    buf = io.StringIO(text)
    sample = buf.read(2048)
    buf.seek(0)
    try:
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
        cleaned = { (k.strip() if k else k): (v.strip() if isinstance(v, str) else v) for k, v in row.items() }
        rows.append(RowCtx(file=path, rownum=i, header=header, data=cleaned))
    return header, rows, issues

def read_csv_file(path: str) -> Tuple[List[str], List[RowCtx], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as fp:
        return _parse_csv_text(path, fp.read())

# -----------------------------
# Metadata (very simple "file:/pattern" → "col: desc" parser)
# -----------------------------

def _parse_metadata_text(lines: List[str]) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    current_keys: List[str] = []
    def _norm(s: str) -> str:
        return s.strip().replace("\\", "/").lower()

    for raw in lines:
        line = raw.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        lk = k.lower()
        if lk in ("file", "files", "pattern", "file_pattern"):
            current_keys = []
            for seg in [seg.strip() for seg in v.split(",") if seg.strip()]:
                current_keys.append(_norm(seg))
            continue
        # treat other keys as column names with description value
        if not current_keys:
            current_keys = ["__default__"]
        for tgt in current_keys:
            mapping.setdefault(tgt, {})[k.strip()] = v
    return mapping

def load_metadata_descriptions(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return _parse_metadata_text(fp.readlines())
    except FileNotFoundError:
        return {}

def _select_metadata_for_file(meta: Dict[str, Dict[str, str]], file_path: str) -> Dict[str, str]:
    if not meta:
        return {}
    norm = file_path.replace("\\", "/").lower()
    base = os.path.basename(norm)
    if norm in meta:
        return meta[norm]
    if base in meta:
        return meta[base]
    return meta.get("__default__", {})

# -----------------------------
# Stats
# -----------------------------

def _quantiles(vals: List[float]) -> List[float]:
    if not vals:
        return []
    s = sorted(vals)
    n = len(s)
    qs = []
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        if n == 1:
            qs.append(s[0]); continue
        pos = q * (n - 1)
        lo = int(pos)
        hi = int(pos + 1e-9)
        if hi >= n: hi = n-1
        if lo == hi:
            qs.append(s[lo])
        else:
            w = pos - lo
            qs.append(s[lo]*(1-w) + s[hi]*w)
    return qs

def summarize_numeric(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0, "min": None, "max": None, "mean": None, "stddev": None,
            "quantiles": [],
            "distribution": {"edges": [], "counts": []},
            "outliers": {"lower_fence": None, "upper_fence": None, "iqr": None, "q1": None, "q3": None, "count": 0},
        }
    n = len(values)
    mn = min(values); mx = max(values)
    mean = sum(values)/n
    var = sum((v-mean)**2 for v in values)/n
    std = var ** 0.5
    qs = _quantiles(values)
    q1 = qs[1] if len(qs) >= 4 else None
    q3 = qs[3] if len(qs) >= 4 else None
    iqr = (q3 - q1) if (q1 is not None and q3 is not None) else None
    lower = (q1 - 1.5*iqr) if iqr is not None else None
    upper = (q3 + 1.5*iqr) if iqr is not None else None
    out_cnt = 0
    if lower is not None or upper is not None:
        for v in values:
            if (lower is not None and v < lower) or (upper is not None and v > upper):
                out_cnt += 1
    # histogram
    bin_count = 20
    if mx == mn:
        edges, counts = [], []
    else:
        step = (mx - mn)/bin_count
        edges = [mn + i*step for i in range(bin_count+1)]
        counts = [0]*bin_count
        for v in values:
            idx = int((v - mn)/step)
            if idx < 0: idx = 0
            if idx >= bin_count: idx = bin_count-1
            counts[idx] += 1
    return {
        "count": n, "min": mn, "max": mx, "mean": mean, "stddev": std,
        "quantiles": qs,
        "distribution": {"edges": edges, "counts": counts},
        "outliers": {"lower_fence": lower, "upper_fence": upper, "iqr": iqr, "q1": q1, "q3": q3, "count": out_cnt},
    }

# -----------------------------
# Auto rules
# -----------------------------

def detect_numeric_cols(rows: List[RowCtx], header: List[str], min_ok_ratio: float = 0.9) -> List[str]:
    numeric_cols: List[str] = []
    if not header or not rows:
        return numeric_cols
    samples = min(500, len(rows))
    for col in header:
        ok = 0
        seen = 0
        for rc in rows[:samples]:
            v = rc.data.get(col)
            if _is_missing(v):
                continue
            seen += 1
            if _try_float(v) is not None:
                ok += 1
        if seen == 0:
            continue
        if ok/seen >= min_ok_ratio:
            numeric_cols.append(col)
    return numeric_cols

TIME_CANDIDATES = {"time_stamp","timestamp","ts","time","event_time"}

def guess_event_time_col(header: List[str]) -> Optional[str]:
    lower = {h.strip().lower(): h for h in header}
    for key in TIME_CANDIDATES:
        if key in lower:
            return lower[key]
    return None

def build_auto_rules(files: List[str], metadata_path: Optional[str] = None) -> Dict[str, Any]:
    required_cols: List[str] = []
    numeric_cols_union: List[str] = []
    event_time_col: Optional[str] = None

    meta_all = load_metadata_descriptions(metadata_path)

    for path in files:
        header, rows, _ = read_csv_file(path)
        if not header:
            continue
        if not required_cols:
            required_cols = list(header)
        nums = detect_numeric_cols(rows, header)
        for c in nums:
            if c not in numeric_cols_union:
                numeric_cols_union.append(c)
        if event_time_col is None:
            event_time_col = guess_event_time_col(header)

    metadata_by_file: Dict[str, Dict[str, str]] = {}
    for f in files:
        md = _select_metadata_for_file(meta_all, f)
        if md:
            metadata_by_file[f] = md

    return {
        "rules": [{
            "patterns": [".*"],
            "required_cols": required_cols,
            "numeric_cols": numeric_cols_union,
            "event_time_col": event_time_col,
            "event_time_format": "auto",
            "metadata_by_file": metadata_by_file,
        }]
    }

# -----------------------------
# Main sequential run
# -----------------------------

def run(
    input_pattern: str,
    good_out: str,
    bad_out: str,
    dq_out: str,
    config_path: Optional[str],
    engine: str = "sequential",
    augmentation_strategy: Optional[str] = None,
    augmentation_output: Optional[str] = None,
    augmentation_repeat: int = 1,
    augmentation_seed: Optional[int] = None,
) -> None:
    if engine not in ("auto","sequential","beam"):
        raise ValueError("engine must be one of auto|sequential|beam")
    if engine == "beam":
        raise RuntimeError("This minimal module only implements the sequential engine. Use engine='sequential'.")

    matched_files = sorted(glob.glob(input_pattern, recursive=True))
    if not matched_files:
        raise FileNotFoundError(f"No files matched input pattern: {input_pattern}")

    Path(dq_out).mkdir(parents=True, exist_ok=True)

    # Build config (auto or from file)
    cfg: Dict[str, Any]
    if (config_path is None) or (str(config_path).strip().upper() == "AUTO"):
        meta_path = os.getenv("DQ_METADATA_PATH")
        cfg = build_auto_rules(matched_files, metadata_path=meta_path)
    else:
        text = Path(config_path).read_text(encoding="utf-8")
        try:
            import yaml  # type: ignore
            cfg = yaml.safe_load(text)
        except Exception:
            try:
                cfg = json.loads(text)
            except Exception:
                cfg = {"rules": [ {"patterns": [".*"]} ]}

    rules = cfg.get("rules") or []
    general_rule = rules[0] if rules else {"patterns":[".*"], "required_cols": [], "numeric_cols": []}

    headers_info: List[Dict[str, Any]] = []
    header_union: set = set()
    header_intersection: Optional[set] = None
    per_file_counts: Dict[str, int] = {}
    per_file_numeric: Dict[str, Dict[str, List[float]]] = {}

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

        per_file_counts[path] = len(rows)

        numeric_cols = general_rule.get("numeric_cols") or detect_numeric_cols(rows, header)
        for col in numeric_cols:
            vals = per_file_numeric.setdefault(path, {}).setdefault(col, [])
            for rc in rows:
                v = _try_float(rc.data.get(col))
                if v is not None:
                    vals.append(v)

    header_intersection = header_intersection or set()

    _write_single_shard(os.path.join(dq_out, "headers", "per_file"), ".jsonl",
        [json.dumps(item, ensure_ascii=False) for item in headers_info])
    _write_single_shard(os.path.join(dq_out, "headers", "union"), ".json",
        [json.dumps(sorted(list(header_union))) ])
    _write_single_shard(os.path.join(dq_out, "headers", "intersection"), ".json",
        [json.dumps(sorted(list(header_intersection))) ])

    per_file_numeric_stats: Dict[str, Dict[str, Any]] = {}
    for f, colmap in per_file_numeric.items():
        per_file_numeric_stats[f] = {}
        for col, values in colmap.items():
            per_file_numeric_stats[f][col] = summarize_numeric(values)

    per_file_section: List[Dict[str, Any]] = []
    meta_all = general_rule.get("metadata_by_file") or {}
    for f in matched_files:
        cols = per_file_numeric_stats.get(f, {})
        md = meta_all.get(f) or {}
        cols_with_desc: Dict[str, Any] = {}
        for name, stats in cols.items():
            v = dict(stats)
            desc = md.get(name) or md.get(name.lower())
            if desc:
                v["description"] = desc
            cols_with_desc[name] = v
        per_file_section.append({
            "file": f,
            "good_rows": per_file_counts.get(f,0),
            "numeric_columns": cols_with_desc,
            "visualizations": {},
        })

    issue_summary = []

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_pattern": input_pattern,
        "files_processed": len(matched_files),
        "total_good_rows": sum(per_file_counts.values()),
        "total_issue_records": 0,
        "issue_summary": issue_summary,
        "bad_issue_samples": [],
        "per_file": per_file_section,
    }

    _write_single_shard(os.path.join(dq_out, "quality_report"), ".json",
        [json.dumps(report, ensure_ascii=False, indent=2)])

    lines = [
        f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] Data quality pipeline completed",
        f"Input pattern: {input_pattern}",
        f"Files processed: {len(matched_files)}",
        f"Total good rows: {sum(per_file_counts.values())}",
        "Visualizations: (disabled in minimal module)",
        f"Output directory: {dq_out}",
    ]
    _write_single_shard(os.path.join(dq_out, "logs", "pipeline"), ".log", lines)
