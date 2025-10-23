"""Augmentation helpers for multi-table time series datasets.

The functions implemented here are intentionally light-weight so that they can
be used inside Apache Beam transforms or simple Python scripts without pulling
in heavy dependencies.  Each helper operates on CSV like inputs and returns a
list of dictionaries representing the augmented rows.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union


__all__ = [
    "augment_with_time",
    "augment_without_time",
    "load_and_align_time_series",
]


DataRecord = Dict[str, object]
DataFrameLike = Union[str, Path]


@dataclass(frozen=True)
class _PreparedTable:
    """Representation of a single table ready for alignment."""

    label: str
    rows: Dict[dt.datetime, Dict[str, object]]


def _infer_label(source: DataFrameLike, index: int) -> str:
    path = Path(source)
    stem = path.stem
    return stem if stem else f"table{index + 1}"


def _parse_timestamp(value: object, *, fmt: str | None = None) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc).astimezone(dt.timezone.utc)
    text = str(value).strip()
    if not text:
        raise ValueError("Empty timestamp value")
    if fmt:
        return dt.datetime.strptime(text, fmt)
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        pass
    common_formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M",
    )
    for pattern in common_formats:
        try:
            return dt.datetime.strptime(text, pattern)
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp format: {text}")


def _maybe_convert(value: str) -> object:
    text = value.strip()
    if text == "":
        return ""
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return text


def _prepare_table(
    source: DataFrameLike,
    *,
    time_column: str,
    parse_dates: bool,
    time_format: str | None,
    index: int,
) -> _PreparedTable:
    label = _infer_label(source, index)
    rows: Dict[dt.datetime, Dict[str, object]] = {}
    with open(source, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if time_column not in reader.fieldnames:
            raise ValueError(f"Column '{time_column}' not found in {source}")
        for raw_row in reader:
            raw_time = raw_row.get(time_column)
            if raw_time is None:
                continue
            timestamp = (
                _parse_timestamp(raw_time, fmt=time_format) if parse_dates else str(raw_time).strip()
            )
            if timestamp in rows:
                raise ValueError(
                    f"Duplicate timestamp detected in table '{label}' for value {timestamp!r}. "
                    "Timestamps must be unique per table to align rows predictably."
                )
            feature_row: Dict[str, object] = {}
            for key, val in raw_row.items():
                if key == time_column:
                    continue
                feature_row[f"{label}_{key}"] = _maybe_convert(val)
            rows[timestamp] = feature_row
    return _PreparedTable(label=label, rows=rows)


def _resolve_join_keys(tables: Sequence[_PreparedTable], join: str) -> List[dt.datetime]:
    if not tables:
        return []
    key_sets = [set(table.rows.keys()) for table in tables]
    if join == "inner":
        keys = set.intersection(*key_sets)
    elif join == "outer":
        keys = set.union(*key_sets)
    else:
        raise ValueError(f"Unsupported join type: {join}")
    return sorted(keys)


def load_and_align_time_series(
    tables: Sequence[DataFrameLike],
    *,
    time_column: str = "time",
    parse_dates: bool = True,
    time_format: str | None = None,
    join: str = "inner",
) -> List[DataRecord]:
    """Load tables and align them on the timestamp column.

    Returns a list of dictionaries with the timestamp preserved under
    ``time_column`` and all other fields prefixed by their originating table
    label. Each input table must have unique timestamps so that rows can be
    matched without ambiguity.
    """

    if not tables:
        raise ValueError("No tables provided for augmentation")

    prepared = [
        _prepare_table(
            str(table),
            time_column=time_column,
            parse_dates=parse_dates,
            time_format=time_format,
            index=index,
        )
        for index, table in enumerate(tables)
    ]

    aligned: List[DataRecord] = []
    for key in _resolve_join_keys(prepared, join):
        record: DataRecord = {}
        include = True
        for table in prepared:
            features = table.rows.get(key)
            if features is None:
                if join == "inner":
                    include = False
                    break
                continue
            record.update(features)
        if include:
            record[time_column] = key
            aligned.append(record)

    aligned.sort(key=lambda row: row[time_column])
    return aligned


def augment_without_time(
    tables: Sequence[DataFrameLike],
    *,
    time_column: str = "time",
    parse_dates: bool = True,
    time_format: str | None = None,
    join: str = "inner",
) -> List[DataRecord]:
    """Concatenate feature columns from all tables, excluding the timestamp."""

    aligned = load_and_align_time_series(
        tables,
        time_column=time_column,
        parse_dates=parse_dates,
        time_format=time_format,
        join=join,
    )
    return [
        {key: value for key, value in row.items() if key != time_column}
        for row in aligned
    ]


def _time_feature_row(timestamp: dt.datetime, prefix: str) -> Dict[str, object]:
    if not isinstance(timestamp, dt.datetime):
        timestamp = _parse_timestamp(timestamp)

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)

    unix_seconds = timestamp.timestamp()
    iso = timestamp.isocalendar()
    two_pi = 2.0 * math.pi
    hour = timestamp.hour
    minute = timestamp.minute
    second = timestamp.second

    return {
        f"{prefix}_unix": unix_seconds,
        f"{prefix}_year": timestamp.year,
        f"{prefix}_month": timestamp.month,
        f"{prefix}_day": timestamp.day,
        f"{prefix}_hour": hour,
        f"{prefix}_minute": minute,
        f"{prefix}_second": second,
        f"{prefix}_dayofweek": timestamp.weekday(),
        f"{prefix}_dayofyear": timestamp.timetuple().tm_yday,
        f"{prefix}_iso_week": iso.week,
        f"{prefix}_iso_year": iso.year,
        f"{prefix}_hour_sin": math.sin(two_pi * hour / 24.0),
        f"{prefix}_hour_cos": math.cos(two_pi * hour / 24.0),
        f"{prefix}_minute_sin": math.sin(two_pi * minute / 60.0),
        f"{prefix}_minute_cos": math.cos(two_pi * minute / 60.0),
        f"{prefix}_second_sin": math.sin(two_pi * second / 60.0),
        f"{prefix}_second_cos": math.cos(two_pi * second / 60.0),
    }


def _format_timestamp(value: object) -> str:
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            return value.isoformat(sep=" ")
        return value.isoformat()
    return str(value)


def augment_with_time(
    tables: Sequence[DataFrameLike],
    *,
    time_column: str = "time",
    parse_dates: bool = True,
    time_format: str | None = None,
    join: str = "inner",
) -> List[DataRecord]:
    """Augment the dataset with engineered temporal features."""

    aligned = load_and_align_time_series(
        tables,
        time_column=time_column,
        parse_dates=parse_dates,
        time_format=time_format,
        join=join,
    )
    augmented: List[DataRecord] = []
    for row in aligned:
        timestamp = row[time_column]
        features = _time_feature_row(timestamp, prefix=time_column)
        enriched = dict(row)
        enriched.update(features)
        enriched[time_column] = _format_timestamp(timestamp)
        augmented.append(enriched)
    return augmented

