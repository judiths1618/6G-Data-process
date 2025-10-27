"""Lightweight CSV-based data augmentation helpers.

This module provides a small subset of the augmentation helpers that are used
by :mod:`dq_local_beam`.  The original project ships a much richer
implementation, but the tests that accompany this kata only require two
strategies: ``temporal_jitter`` and ``load_scaling``.  The goal of this module
is therefore to offer a deterministic and dependency-free implementation that
behaves similarly enough for unit testing while keeping the public API stable.

The central entry point is :func:`generate_augmented_dataset` which receives a
list of input CSV files and a list of rules describing how those files should
be processed.  Each rule mirrors the structure used in the data quality CLI –
in particular the ``patterns`` list that determines which files a rule applies
to, the ``event_time_col`` and ``event_time_format`` fields, and
``numeric_cols`` describing which columns contain numeric values.  The function
returns a list of paths to the generated CSV files.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence


__all__ = [
    "AUGMENTATION_STRATEGIES",
    "generate_augmented_dataset",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_input_files(paths: Sequence[str | os.PathLike[str]]) -> Iterator[Path]:
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            yield path
        elif path.is_dir():
            for csv_path in sorted(p for p in path.rglob("*.csv") if p.is_file()):
                yield csv_path
        else:
            raise FileNotFoundError(f"Input source '{path}' does not exist")


def _read_csv(path: Path) -> List[MutableMapping[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty augmented dataset")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _match_rule(path: Path, rules: Sequence[Mapping[str, object]]) -> Optional[Mapping[str, object]]:
    for rule in rules:
        patterns = rule.get("patterns")
        if not patterns:
            continue
        for pattern in patterns:
            if re.search(pattern, str(path)):
                return rule
    return None


def _parse_float(value: str) -> Optional[float]:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _is_int_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("-"):
        return stripped[1:].isdigit()
    return stripped.isdigit()


def _format_number(original: str, value: float, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> str:
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    if _is_int_like(original):
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _coerce_event_time(value: str, fmt: Optional[str]) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError("Empty event time value encountered during augmentation")
    if fmt in {None, "", "auto", "epoch_s"}:
        return float(text)
    if fmt == "epoch_ms":
        return float(text) / 1000.0
    if fmt == "iso":
        return dt.datetime.fromisoformat(text).timestamp()
    # Treat any other value as a strptime-compatible pattern.
    return dt.datetime.strptime(text, fmt).timestamp()


def _format_event_time(value: float, original: str, fmt: Optional[str]) -> str:
    if fmt in {None, "", "auto", "epoch_s"}:
        return str(int(round(value)))
    if fmt == "epoch_ms":
        return str(int(round(value * 1000)))
    if fmt == "iso":
        return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc).astimezone(dt.timezone.utc).isoformat()
    timestamp = dt.datetime.fromtimestamp(value, tz=dt.timezone.utc).astimezone(dt.timezone.utc)
    try:
        return timestamp.strftime(fmt)
    except Exception:
        # Fall back to the original string if formatting fails.
        return original


# ---------------------------------------------------------------------------
# Augmentation strategies
# ---------------------------------------------------------------------------


@dataclass
class _StrategyContext:
    path: Path
    rule: Mapping[str, object]
    repeat: int
    rng: random.Random


def _temporal_jitter(ctx: _StrategyContext) -> List[Mapping[str, object]]:
    rows = _read_csv(ctx.path)
    if not rows:
        return []

    event_col = str(ctx.rule.get("event_time_col")) if ctx.rule.get("event_time_col") else None
    fmt = str(ctx.rule.get("event_time_format") or "auto")
    numeric_cols = [str(col) for col in ctx.rule.get("numeric_cols", [])]

    augmented: List[Mapping[str, object]] = []
    for row in rows:
        base_time = row.get(event_col) if event_col else None
        base_seconds = _coerce_event_time(base_time, fmt) if (event_col and base_time is not None) else None
        for _ in range(max(1, ctx.repeat)):
            jittered = dict(row)
            for col in numeric_cols:
                raw_value = row.get(col)
                if raw_value is None:
                    continue
                numeric = _parse_float(raw_value)
                if numeric is None:
                    continue
                scale = max(abs(numeric), 1.0)
                delta = ctx.rng.gauss(0.0, 0.1 * scale)
                minimum = 0.0 if numeric >= 0 else None
                maximum = 100.0 if 0.0 <= numeric <= 100.0 else None
                jittered[col] = _format_number(raw_value, numeric + delta, minimum=minimum, maximum=maximum)
            if base_seconds is not None and event_col:
                offset = ctx.rng.uniform(-300.0, 300.0)
                jittered[event_col] = _format_event_time(max(0.0, base_seconds + offset), row[event_col], fmt)
            augmented.append(jittered)
    return augmented


def _load_scaling(ctx: _StrategyContext) -> List[Mapping[str, object]]:
    rows = _read_csv(ctx.path)
    if not rows:
        return []

    event_col = str(ctx.rule.get("event_time_col")) if ctx.rule.get("event_time_col") else None
    fmt = str(ctx.rule.get("event_time_format") or "auto")
    numeric_cols = [str(col) for col in ctx.rule.get("numeric_cols", [])]

    augmented: List[Mapping[str, object]] = []
    for row in rows:
        base_time = row.get(event_col) if event_col else None
        base_seconds = _coerce_event_time(base_time, fmt) if (event_col and base_time is not None) else None
        for index in range(max(1, ctx.repeat)):
            scaled = dict(row)
            factor = 1.0 + ctx.rng.uniform(0.0, 0.5)
            for col in numeric_cols:
                raw_value = row.get(col)
                if raw_value is None:
                    continue
                numeric = _parse_float(raw_value)
                if numeric is None:
                    continue
                minimum = 0.0 if numeric >= 0 else None
                scaled[col] = _format_number(raw_value, numeric * factor, minimum=minimum)
            if base_seconds is not None and event_col:
                scaled[event_col] = _format_event_time(base_seconds + index, row[event_col], fmt)
            augmented.append(scaled)
    return augmented


AUGMENTATION_STRATEGIES: Dict[str, Callable[[_StrategyContext], List[Mapping[str, object]]]] = {
    "temporal_jitter": _temporal_jitter,
    "load_scaling": _load_scaling,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_augmented_dataset(
    input_files: Sequence[str | os.PathLike[str]],
    rules: Sequence[Mapping[str, object]],
    output_directory: str | os.PathLike[str],
    strategy: str,
    *,
    repeat: int = 1,
    seed: Optional[int] = None,
) -> List[str]:
    """Generate augmented CSV files using the requested strategy.

    Parameters
    ----------
    input_files:
        A collection of CSV paths (or directories containing CSV files) that
        will be processed.
    rules:
        Rule dictionaries from the configuration.  The first rule whose
        ``patterns`` regular expression matches the file path will be applied.
    output_directory:
        Directory where the augmented CSV files will be written.
    strategy:
        The augmentation strategy identifier.  See :data:`AUGMENTATION_STRATEGIES`.
    repeat:
        Number of synthetic rows to generate per input row.
    seed:
        Optional random seed for deterministic behaviour.
    """

    if strategy not in AUGMENTATION_STRATEGIES:
        known = ", ".join(sorted(AUGMENTATION_STRATEGIES))
        raise ValueError(f"Unsupported augmentation strategy: {strategy!r}. Known strategies: {known}")

    if repeat <= 0:
        raise ValueError("repeat must be a positive integer")

    rng = random.Random(seed)
    out_dir = Path(output_directory)
    _ensure_directory(out_dir)

    generated_paths: List[str] = []
    seen_targets: set[Path] = set()

    for input_path in _iter_input_files(input_files):
        rule = _match_rule(input_path, rules)
        if not rule:
            continue
        ctx = _StrategyContext(path=input_path, rule=rule, repeat=repeat, rng=rng)
        rows = AUGMENTATION_STRATEGIES[strategy](ctx)
        if not rows:
            continue
        target_name = f"{input_path.stem}__{strategy}.csv"
        target_path = out_dir / target_name
        # Avoid overwriting files if multiple inputs would map to the same name.
        counter = 1
        while target_path in seen_targets:
            counter += 1
            target_path = out_dir / f"{input_path.stem}__{strategy}_{counter}.csv"
        _write_csv(target_path, rows)
        seen_targets.add(target_path)
        generated_paths.append(str(target_path))

    return generated_paths

