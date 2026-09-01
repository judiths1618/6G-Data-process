"""Pandera schemas used for fast dataframe validation in tests and pipelines."""
from __future__ import annotations

import pandas as pd


def _pandera():
    try:
        import pandera.pandas as pa
    except ImportError:
        import pandera as pa
    return pa


def build_numeric_timeseries_schema(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    nullable_numeric: bool = True,
    expected_columns: list[str] | None = None,
    numeric_bounds: dict[str, dict[str, float]] | None = None,
):
    """Build a schema for a timestamp column plus numeric feature columns."""
    pa = _pandera()
    if timestamp_col not in df.columns:
        raise KeyError(f"timestamp column {timestamp_col!r} not in dataframe")
    missing_expected = sorted(set(expected_columns or []) - set(df.columns))
    if missing_expected:
        raise ValueError(f"missing expected columns: {missing_expected}")

    columns = {
        timestamp_col: pa.Column(
            pa.DateTime if pd.api.types.is_datetime64_any_dtype(df[timestamp_col]) else pa.Int64,
            nullable=False,
            coerce=True,
        )
    }
    for col in df.columns:
        if col == timestamp_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            checks = []
            bounds = (numeric_bounds or {}).get(col, {})
            if "min" in bounds:
                checks.append(pa.Check.ge(bounds["min"]))
            if "max" in bounds:
                checks.append(pa.Check.le(bounds["max"]))
            columns[col] = pa.Column(
                float,
                checks=checks,
                nullable=nullable_numeric,
                coerce=True,
            )

    return pa.DataFrameSchema(columns=columns, strict=False, coerce=True)


def _expected_columns_issues(
    df: pd.DataFrame,
    expected_columns: list[str] | None = None,
) -> list[str]:
    missing_expected = sorted(set(expected_columns or []) - set(df.columns))
    if missing_expected:
        return [f"missing expected columns: {missing_expected}"]
    return []


def build_tabular_schema(
    df: pd.DataFrame,
    *,
    expected_columns: list[str] | None = None,
    numeric_bounds: dict[str, dict[str, float]] | None = None,
    nullable_numeric: bool = True,
):
    """Build a schema for arbitrary tabular data, validating known numeric columns."""
    pa = _pandera()
    missing_expected = sorted(set(expected_columns or []) - set(df.columns))
    if missing_expected:
        raise ValueError(f"missing expected columns: {missing_expected}")

    columns = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            checks = []
            bounds = (numeric_bounds or {}).get(col, {})
            if "min" in bounds:
                checks.append(pa.Check.ge(bounds["min"]))
            if "max" in bounds:
                checks.append(pa.Check.le(bounds["max"]))
            columns[col] = pa.Column(
                float,
                checks=checks,
                nullable=nullable_numeric,
                coerce=True,
            )

    return pa.DataFrameSchema(columns=columns, strict=False, coerce=True)


def validate_timestamp_contract(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    require_unique: bool = True,
    require_monotonic: bool = True,
) -> list[str]:
    """Return timestamp contract violations."""
    issues: list[str] = []
    if timestamp_col not in df.columns:
        return [f"timestamp column {timestamp_col!r} not in dataframe"]
    ts = df[timestamp_col]
    if ts.isna().any():
        issues.append(f"timestamp column {timestamp_col!r} contains null values")
    if require_unique and ts.duplicated().any():
        issues.append(f"timestamp column {timestamp_col!r} contains duplicate values")
    if require_monotonic and not ts.is_monotonic_increasing:
        issues.append(f"timestamp column {timestamp_col!r} is not monotonic increasing")
    return issues


def validate_missingness(
    df: pd.DataFrame,
    *,
    threshold: float = 0.0,
) -> list[str]:
    """Return columns whose missing ratio exceeds ``threshold``."""
    issues: list[str] = []
    for col in df.columns:
        ratio = float(df[col].isna().mean())
        if ratio > threshold:
            issues.append(f"column {col!r} missing ratio {ratio:.4f} exceeds {threshold:.4f}")
    return issues


def validate_numeric_timeseries(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    nullable_numeric: bool = True,
    expected_columns: list[str] | None = None,
    numeric_bounds: dict[str, dict[str, float]] | None = None,
    missing_threshold: float = 0.0,
    require_timestamp_unique: bool = True,
    require_timestamp_monotonic: bool = True,
) -> pd.DataFrame:
    """Validate and return a dataframe using a Pandera numeric time-series schema."""
    if timestamp_col not in df.columns:
        raise KeyError(f"timestamp column {timestamp_col!r} not in dataframe")
    issues = validate_timestamp_contract(
        df,
        timestamp_col=timestamp_col,
        require_unique=require_timestamp_unique,
        require_monotonic=require_timestamp_monotonic,
    )
    issues.extend(validate_missingness(df, threshold=missing_threshold))
    if issues:
        raise ValueError("; ".join(issues))
    schema = build_numeric_timeseries_schema(
        df,
        timestamp_col=timestamp_col,
        nullable_numeric=nullable_numeric,
        expected_columns=expected_columns,
        numeric_bounds=numeric_bounds,
    )
    return schema.validate(df)


def validate_tabular_dataframe(
    df: pd.DataFrame,
    *,
    expected_columns: list[str] | None = None,
    numeric_bounds: dict[str, dict[str, float]] | None = None,
    missing_threshold: float = 0.0,
    nullable_numeric: bool = True,
) -> pd.DataFrame:
    """Validate arbitrary tabular data without requiring a timestamp column.

    Missing expected columns and missingness are collected together so a caller
    sees every problem at once. (An earlier duplicate definition of this function
    shadowed by this one did the same; the surviving copy had dropped the
    expected-columns check from the issue list and left it to raise separately
    inside the schema builder, hiding it whenever missingness also failed.)
    """
    issues = _expected_columns_issues(df, expected_columns)
    issues.extend(validate_missingness(df, threshold=missing_threshold))
    if issues:
        raise ValueError("; ".join(issues))
    schema = build_tabular_schema(
        df,
        expected_columns=expected_columns,
        numeric_bounds=numeric_bounds,
        nullable_numeric=nullable_numeric,
    )
    return schema.validate(df)


METADATA = {
    "name": "pandera_schemas",
    "version": "0.1.0",
    "category": "validation",
    "summary": "Pandera dataframe schemas for fast local time-series and tabular validation.",
    "entrypoint": "dataops.validation:validate_numeric_timeseries",
    "gpu": False,
    "dependencies": ["pandera", "pandas"],
}
