"""Data augmentation strategies for local DQ workflows."""

from __future__ import annotations

import csv
import datetime as dt
import os
import random
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dq_local_beam import RowCtx


@lru_cache(maxsize=None)
def _helpers() -> Dict[str, Any]:
    """Lazily import helper utilities from :mod:`dq_local_beam`.

    The augmentation strategies share parsing and formatting helpers with the
    main DQ script.  Importing them lazily avoids creating a circular import
    when :mod:`dq_local_beam` pulls in :mod:`augmentation`.
    """

    from dq_local_beam import (  # type: ignore circular import at runtime
        _format_event_time_value,
        _format_numeric_like,
        _is_missing_value,
        _prepare_header_alias,
        _resolve_column_from_header,
        _shift_staleness_time_columns,
        parse_event_time,
        parse_numeric_with_units,
    )

    return {
        "format_event_time_value": _format_event_time_value,
        "format_numeric_like": _format_numeric_like,
        "is_missing_value": _is_missing_value,
        "prepare_header_alias": _prepare_header_alias,
        "resolve_column_from_header": _resolve_column_from_header,
        "shift_staleness_time_columns": _shift_staleness_time_columns,
        "parse_event_time": parse_event_time,
        "parse_numeric_with_units": parse_numeric_with_units,
    }


def _temporal_jitter_strategy(
    header: List[str],
    rows: Sequence["RowCtx"],
    rule: Dict[str, Any],
    rng: random.Random,
    iteration: int,
) -> List[Dict[str, Any]]:
    helpers = _helpers()
    alias, _ = helpers["prepare_header_alias"](header)
    staleness_spec = rule.get("staleness") or {}
    fmt = (rule.get("event_time_format") or "auto").lower()
    event_actual = helpers["resolve_column_from_header"](header, alias, rule.get("event_time_col"))
    results: List[Dict[str, Any]] = []
    base_shift = iteration * 120.0

    for rc in rows:
        row = dict(rc.data)
        delta_seconds = base_shift + rng.uniform(-300.0, 300.0)
        event_delta: Optional[float] = None
        if event_actual:
            raw_time = row.get(event_actual)
            if not helpers["is_missing_value"](raw_time):
                try:
                    base_ts = helpers["parse_event_time"](raw_time, fmt)
                    shifted_ts = base_ts + dt.timedelta(seconds=delta_seconds)
                    row[event_actual] = helpers["format_event_time_value"](shifted_ts, fmt, raw_time)
                    event_delta = (shifted_ts - base_ts).total_seconds()
                except Exception:
                    event_delta = None
        helpers["shift_staleness_time_columns"](
            row,
            header,
            alias,
            staleness_spec,
            event_delta if event_delta is not None else delta_seconds,
        )
        for col in rule.get("numeric_cols", []):
            actual_col = helpers["resolve_column_from_header"](header, alias, col)
            if not actual_col:
                continue
            raw_val = row.get(actual_col)
            if helpers["is_missing_value"](raw_val):
                continue
            try:
                base_val = helpers["parse_numeric_with_units"](raw_val, actual_col, rule)
            except Exception:
                continue
            noise = rng.uniform(-0.05, 0.05)
            updated = base_val * (1.0 + noise)
            lower = actual_col.lower()
            if "usage" in lower or lower.endswith("_pct"):
                updated = max(0.0, min(100.0, updated))
            else:
                updated = max(0.0, updated)
            row[actual_col] = helpers["format_numeric_like"](raw_val, updated)
        results.append(row)
    return results


def _load_scaling_strategy(
    header: List[str],
    rows: Sequence["RowCtx"],
    rule: Dict[str, Any],
    rng: random.Random,
    iteration: int,
) -> List[Dict[str, Any]]:
    helpers = _helpers()
    alias, _ = helpers["prepare_header_alias"](header)
    staleness_spec = rule.get("staleness") or {}
    fmt = (rule.get("event_time_format") or "auto").lower()
    event_actual = helpers["resolve_column_from_header"](header, alias, rule.get("event_time_col"))
    base_offset = (iteration + 1) * 600.0
    load_factor = 1.0 + rng.uniform(0.25, 0.6)
    latency_factor = 1.0 + (load_factor - 1.0) * rng.uniform(0.8, 1.2)
    results: List[Dict[str, Any]] = []

    for rc in rows:
        row = dict(rc.data)
        delta_seconds = base_offset + rng.uniform(0.0, 180.0)
        event_delta: Optional[float] = None
        if event_actual:
            raw_time = row.get(event_actual)
            if not helpers["is_missing_value"](raw_time):
                try:
                    base_ts = helpers["parse_event_time"](raw_time, fmt)
                    shifted_ts = base_ts + dt.timedelta(seconds=delta_seconds)
                    row[event_actual] = helpers["format_event_time_value"](shifted_ts, fmt, raw_time)
                    event_delta = (shifted_ts - base_ts).total_seconds()
                except Exception:
                    event_delta = None
        helpers["shift_staleness_time_columns"](
            row,
            header,
            alias,
            staleness_spec,
            event_delta if event_delta is not None else delta_seconds,
        )
        for col in rule.get("numeric_cols", []):
            actual_col = helpers["resolve_column_from_header"](header, alias, col)
            if not actual_col:
                continue
            raw_val = row.get(actual_col)
            if helpers["is_missing_value"](raw_val):
                continue
            try:
                base_val = helpers["parse_numeric_with_units"](raw_val, actual_col, rule)
            except Exception:
                continue
            lower = actual_col.lower()
            updated = base_val
            if lower in {"n", "c"} or lower.endswith("_n") or lower.endswith("_c"):
                updated = max(0.0, round(base_val * load_factor))
            elif lower.startswith("lat") or "latency" in lower or lower in {"mean", "min"}:
                updated = max(0.0, base_val * latency_factor)
            elif "usage" in lower or "util" in lower:
                updated = max(
                    0.0,
                    min(100.0, base_val * min(load_factor * 1.1, 1.0 + 0.6 * (load_factor - 1.0))),
                )
            elif "limit" in lower:
                updated = base_val
            else:
                updated = max(0.0, base_val * (1.0 + rng.uniform(-0.03, 0.03)))
            row[actual_col] = helpers["format_numeric_like"](raw_val, updated)
        results.append(row)
    return results


def _write_augmented_csv(path: str, header: Sequence[str], rows: List[Dict[str, Any]]) -> str:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in header})
    return path


def generate_augmented_dataset(
    files: Sequence[str],
    rules: List[Dict[str, Any]],
    output_dir: str,
    strategy: str,
    repeat: int = 1,
    seed: Optional[int] = None,
) -> List[str]:
    from dq_local_beam import pick_rule, read_csv_file  # local import to avoid circular dependency

    if strategy not in AUGMENTATION_STRATEGIES:
        raise ValueError(f"Unknown augmentation strategy: {strategy}")
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    augmented_files: List[str] = []
    for path in files:
        header, rows, issues = read_csv_file(path)
        if not header or not rows:
            continue
        if issues and all(issue.get("reason") == "empty_file" for issue in issues):
            continue
        rule = pick_rule(rules, path)
        aggregated: List[Dict[str, Any]] = []
        for iteration in range(max(1, repeat)):
            aggregated.extend(
                AUGMENTATION_STRATEGIES[strategy](header, rows, rule, rng, iteration)
            )
        if not aggregated:
            continue
        base = os.path.basename(path)
        stem, _ = os.path.splitext(base)
        suffix = strategy if repeat <= 1 else f"{strategy}.x{repeat}"
        out_path = os.path.join(output_dir, f"{stem}.{suffix}.csv")
        _write_augmented_csv(out_path, header, aggregated)
        augmented_files.append(out_path)
    return augmented_files


AUGMENTATION_STRATEGIES = {
    "temporal_jitter": _temporal_jitter_strategy,
    "load_scaling": _load_scaling_strategy,
}


__all__ = ["AUGMENTATION_STRATEGIES", "generate_augmented_dataset"]
