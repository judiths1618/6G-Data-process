"""Lightweight helpers for aligning and enriching 6G CSV bundles."""

from __future__ import annotations

import csv
import datetime as dt
import math
from pathlib import Path
from typing import Dict, Iterator, List, MutableMapping, Sequence, Tuple

__all__ = [
    "augment_with_time",
    "augment_without_time",
    "load_and_align_time_series",
]


def _iter_input_files(sources: Sequence[str | Path]) -> Iterator[Path]:
    for raw in sources:
        path = Path(raw)
        if path.is_file():
            if path.suffix.lower() != ".csv":
                raise ValueError(f"Input file '{path}' is not a CSV table")
            yield path
        elif path.is_dir():
            for csv_path in sorted(path.rglob("*.csv")):
                if csv_path.is_file():
                    yield csv_path
        else:
            raise FileNotFoundError(f"Input source '{path}' does not exist")


def _parse_timestamp(value: str) -> dt.datetime:
    text = str(value).strip()
    if not text:
        raise ValueError("Encountered empty timestamp while aligning tables")
    try:
        number = float(text)
    except ValueError:
        pass
    else:
        if abs(number) > 1_000_000_000_000:
            number /= 1_000.0
        return dt.datetime.fromtimestamp(number, tz=dt.timezone.utc)
    text = text.replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unable to parse timestamp value '{value}'") from exc


def _coerce_value(value: str) -> object:
    text = str(value).strip()
    if text == "":
        return None
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return value


def _load_table(
    path: Path,
    time_column: str,
    on_duplicate: str,
) -> Tuple[Dict[dt.datetime, Dict[str, object]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or time_column not in reader.fieldnames:
            raise ValueError(f"Column '{time_column}' not found in {path}")
        columns = [col for col in reader.fieldnames if col != time_column]
        rows: Dict[dt.datetime, Dict[str, object]] = {}
        order = on_duplicate.lower()
        for raw_row in reader:
            timestamp = _parse_timestamp(raw_row[time_column])
            payload = {col: _coerce_value(raw_row[col]) for col in columns}
            if timestamp in rows:
                if order == "error":
                    raise ValueError(f"Duplicate timestamp detected in {path}")
                if order == "last":
                    rows[timestamp] = payload
                elif order == "first":
                    continue
                else:
                    raise ValueError(f"Unsupported duplicate policy '{on_duplicate}'")
            else:
                rows[timestamp] = payload
    return rows, columns


def _combine_timestamps(
    tables: Sequence[Tuple[Dict[dt.datetime, Dict[str, object]], List[str]]],
    join: str,
) -> List[dt.datetime]:
    sets = [set(mapping.keys()) for mapping, _ in tables]
    if not sets:
        return []
    policy = join.lower()
    if policy == "outer":
        timeline = set.union(*sets)
    elif policy == "inner":
        timeline = set.intersection(*sets)
    elif policy == "left":
        timeline = sets[0]
    elif policy == "right":
        timeline = sets[-1]
    else:
        raise ValueError(f"Unsupported join policy '{join}'")
    return sorted(timeline)


def load_and_align_time_series(
    sources: Sequence[str | Path],
    *,
    time_column: str = "time",
    join: str = "outer",
    on_duplicate: str = "error",
) -> List[MutableMapping[str, object]]:
    """Return aligned rows from one or more CSV sources."""

    payloads: List[Tuple[str, Dict[dt.datetime, Dict[str, object]], List[str]]] = []
    for path in _iter_input_files(sources):
        mapping, columns = _load_table(path, time_column, on_duplicate)
        payloads.append((path.stem, mapping, columns))

    if not payloads:
        raise ValueError("No CSV files were discovered in the provided sources")

    timeline = _combine_timestamps([(mapping, columns) for _, mapping, columns in payloads], join)
    records: List[MutableMapping[str, object]] = []
    for timestamp in timeline:
        row: MutableMapping[str, object] = {time_column: timestamp}
        for prefix, mapping, columns in payloads:
            values = mapping.get(timestamp)
            for column in columns:
                key = f"{prefix}_{column}"
                row[key] = values.get(column) if values is not None else None
        records.append(row)
    return records


def augment_without_time(
    sources: Sequence[str | Path],
    *,
    time_column: str = "time",
    join: str = "outer",
    on_duplicate: str = "error",
) -> List[Dict[str, object]]:
    rows = load_and_align_time_series(
        sources,
        time_column=time_column,
        join=join,
        on_duplicate=on_duplicate,
    )
    return [
        {key: value for key, value in row.items() if key != time_column}
        for row in rows
    ]


def _encode_cycle(value: int, period: int) -> Tuple[float, float]:
    angle = 2.0 * math.pi * (value % period) / period
    return math.sin(angle), math.cos(angle)


def augment_with_time(
    sources: Sequence[str | Path],
    *,
    time_column: str = "time",
    join: str = "outer",
    on_duplicate: str = "error",
) -> List[Dict[str, object]]:
    rows = load_and_align_time_series(
        sources,
        time_column=time_column,
        join=join,
        on_duplicate=on_duplicate,
    )

    enriched: List[Dict[str, object]] = []
    for row in rows:
        timestamp = row[time_column]
        if not isinstance(timestamp, dt.datetime):
            timestamp = _parse_timestamp(str(timestamp))
        naive = timestamp.astimezone(dt.timezone.utc)
        payload: Dict[str, object] = dict(row)
        payload[time_column] = naive.strftime("%Y-%m-%d %H:%M:%S")
        payload["time_unix"] = naive.timestamp()
        payload["time_year"] = naive.year
        payload["time_month"] = naive.month
        payload["time_day"] = naive.day
        payload["time_hour"] = naive.hour
        payload["time_minute"] = naive.minute
        payload["time_second"] = naive.second
        payload["time_dayofweek"] = naive.weekday()
        payload["time_dayofyear"] = naive.timetuple().tm_yday
        iso = naive.isocalendar()
        payload["time_iso_week"] = iso.week
        payload["time_iso_year"] = iso.year
        hour_sin, hour_cos = _encode_cycle(naive.hour, 24)
        minute_sin, minute_cos = _encode_cycle(naive.minute, 60)
        second_sin, second_cos = _encode_cycle(naive.second, 60)
        payload["time_hour_sin"] = hour_sin
        payload["time_hour_cos"] = hour_cos
        payload["time_minute_sin"] = minute_sin
        payload["time_minute_cos"] = minute_cos
        payload["time_second_sin"] = second_sin
        payload["time_second_cos"] = second_cos
        enriched.append(payload)
    return enriched
