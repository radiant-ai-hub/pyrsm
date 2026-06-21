"""Shared utilities for exploratory data analysis helpers."""

from collections.abc import Iterable

import polars as pl

NUMERIC_DTYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
)

INTEGER_DTYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
)

CATEGORICAL_DTYPES = (pl.Utf8, pl.String, pl.Categorical, pl.Enum)

TEMPORAL_DTYPES = (pl.Date, pl.Datetime, pl.Time, pl.Duration)


def materialize(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Collect LazyFrames so downstream helpers work with concrete data."""
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    return df


def as_list(value, name: str) -> list[str]:
    """Normalize a string-or-sequence argument to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        values = list(value)
        if all(isinstance(v, str) for v in values):
            return values
    raise TypeError(f"{name} must be a string or a sequence of strings")


def columns_or_all(df: pl.DataFrame, cols: str | Iterable[str] | None) -> list[str]:
    """Return selected columns, validating that they exist in ``df``."""
    selected = list(df.columns) if cols is None else as_list(cols, "cols")
    missing = [c for c in selected if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {', '.join(missing)}")
    return selected


def is_numeric_dtype(dtype) -> bool:
    """Return True for Polars numeric dtypes."""
    return isinstance(dtype, NUMERIC_DTYPES) or dtype in NUMERIC_DTYPES


def is_integer_dtype(dtype) -> bool:
    """Return True for Polars integer dtypes."""
    return isinstance(dtype, INTEGER_DTYPES) or dtype in INTEGER_DTYPES


def is_categorical_dtype(dtype) -> bool:
    """Return True for string/categorical/enum dtypes."""
    return isinstance(dtype, CATEGORICAL_DTYPES) or dtype in CATEGORICAL_DTYPES


def is_temporal_dtype(dtype) -> bool:
    """Return True for Polars temporal dtypes."""
    return isinstance(dtype, TEMPORAL_DTYPES) or dtype in TEMPORAL_DTYPES


def is_categorical(df: pl.DataFrame, col: str, nint: int = 25) -> bool:
    """Classify strings/enums and low-cardinality integers as categorical."""
    dtype = df.schema.get(col)
    if dtype is None:
        return False
    if is_categorical_dtype(dtype):
        return True
    if is_integer_dtype(dtype):
        n_unique = df.select(pl.col(col).n_unique()).item()
        return n_unique < nint
    return False


def classify_column(df: pl.DataFrame, col: str, nint: int = 25) -> str:
    """Classify a column as numeric, categorical, temporal, or other."""
    dtype = df.schema.get(col)
    if dtype is None:
        return "other"
    if is_categorical(df, col, nint):
        return "categorical"
    if is_numeric_dtype(dtype):
        return "numeric"
    if is_temporal_dtype(dtype):
        return "temporal"
    return "other"
