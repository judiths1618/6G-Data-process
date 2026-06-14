"""Small pandas cleaning functions for the minimal DataOps path."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class CleaningReport:
    """Lightweight summary of dataframe-level cleaning."""

    input_rows: int
    output_rows: int
    input_columns: int
    output_columns: int
    dropped_duplicate_rows: int
    dropped_empty_rows: int
    column_mapping: dict[str, str]


def snake_case(name: object) -> str:
    """Normalize a column name to ASCII-ish snake_case."""
    text = str(name).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "column"


def build_column_mapping(columns: Iterable[object]) -> dict[str, str]:
    """Build a deterministic original-name -> cleaned-name mapping."""
    counts: dict[str, int] = {}
    mapping: dict[str, str] = {}
    for raw in columns:
        base = snake_case(raw)
        counts[base] = counts.get(base, 0) + 1
        mapping[str(raw)] = base if counts[base] == 1 else f"{base}_{counts[base]}"
    return mapping


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with deterministic, unique snake_case column names."""
    mapping = build_column_mapping(df.columns)

    cleaned = df.copy()
    cleaned.columns = list(mapping.values())
    return cleaned


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where every field is missing."""
    return df.dropna(how="all").copy()


def drop_duplicate_rows(df: pd.DataFrame, subset: Iterable[str] | None = None) -> pd.DataFrame:
    """Drop exact duplicate rows, or duplicates on a named subset."""
    return df.drop_duplicates(subset=list(subset) if subset is not None else None).copy()


def coerce_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Coerce one column to pandas datetime when it exists.

    Epoch-aware: a numeric column whose magnitude looks like Unix time is parsed
    with the right ``unit`` (s / ms / us). Without this, ``pd.to_datetime`` treats
    integer epoch *seconds* as *nanoseconds* (1636553178 → 1970-01-01 00:00:01.6),
    silently corrupting the timeline. Non-epoch / string columns fall back to the
    general parser.
    """
    cleaned = df.copy()
    if column not in cleaned.columns:
        return cleaned

    col = cleaned[column]
    numeric = pd.to_numeric(col, errors="coerce")
    if numeric.notna().mean() > 0.9 and numeric.notna().any():
        med = float(numeric.dropna().abs().median())
        unit = None
        if 1e8 <= med < 1e11:        # seconds   (1973 … 5138)
            unit = "s"
        elif 1e11 <= med < 1e14:     # milliseconds
            unit = "ms"
        elif 1e14 <= med < 1e17:     # microseconds
            unit = "us"
        if unit is not None:
            cleaned[column] = pd.to_datetime(numeric, unit=unit, errors="coerce")
            return cleaned

    cleaned[column] = pd.to_datetime(col, errors="coerce")
    return cleaned


def clean_dataframe(
    df: pd.DataFrame,
    *,
    datetime_column: str | None = None,
    duplicate_subset: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Run conservative dataframe cleaning and return ``(clean_df, report)``."""
    input_rows, input_columns = df.shape
    column_mapping = build_column_mapping(df.columns)
    cleaned = standardize_columns(df)
    before_empty = len(cleaned)
    cleaned = drop_empty_rows(cleaned)
    after_empty = len(cleaned)
    before_dupes = len(cleaned)
    cleaned = drop_duplicate_rows(cleaned, subset=duplicate_subset)
    if datetime_column:
        cleaned = coerce_datetime(cleaned, snake_case(datetime_column))

    report = CleaningReport(
        input_rows=input_rows,
        output_rows=len(cleaned),
        input_columns=input_columns,
        output_columns=len(cleaned.columns),
        dropped_empty_rows=before_empty - after_empty,
        dropped_duplicate_rows=before_dupes - len(cleaned),
        column_mapping=column_mapping,
    )
    return cleaned.reset_index(drop=True), report


METADATA = {
    "name": "cleaning",
    "version": "0.1.0",
    "category": "cleaning",
    "summary": "Conservative pandas cleaning helpers for column names, empty rows, duplicates, and datetime coercion.",
    "entrypoint": "dataops.cleaning:clean_dataframe",
    "gpu": False,
    "dependencies": ["pandas"],
}
