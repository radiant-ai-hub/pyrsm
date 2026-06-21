"""Dataset profiling helpers."""

from collections.abc import Iterable

import polars as pl

from pyrsm.eda._utils import (
    classify_column,
    columns_or_all,
    is_numeric_dtype,
    is_temporal_dtype,
    materialize,
)


def _stringify(value) -> str | None:
    """Return a display-safe scalar string for mixed-type profile columns."""
    if value is None:
        return None
    return str(value)


def _value_counts(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Return value counts for a column sorted by frequency."""
    return df.select(pl.col(col).value_counts(sort=True)).unnest(col)


def _top_values(
    df: pl.DataFrame, col: str, top_n: int
) -> tuple[str | None, int | None, str]:
    """Return mode, mode count, and a compact top-values string."""
    counts = _value_counts(df, col)
    if counts.height == 0:
        return None, None, ""

    total = df.height
    mode = _stringify(counts[col][0])
    mode_count = counts["count"][0]
    top_parts = []
    for row in counts.head(top_n).iter_rows(named=True):
        value = "NA" if row[col] is None else str(row[col])
        pct = row["count"] / total if total else 0.0
        top_parts.append(f"{value}: {row['count']} ({pct:.1%})")
    return mode, mode_count, "; ".join(top_parts)


def profile(
    df: pl.DataFrame | pl.LazyFrame,
    cols: str | Iterable[str] | None = None,
    nint: int = 25,
    top_n: int = 3,
) -> pl.DataFrame:
    """
    Create a compact one-row-per-column profile table.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Data to profile.
    cols : str | Iterable[str] | None
        Columns to include. If None, all columns are profiled.
    nint : int
        Number of unique values below which integer columns are treated as
        categorical.
    top_n : int
        Number of top values to show in the ``top_values`` column.

    Returns
    -------
    pl.DataFrame
        One row per variable with dtype, type, missingness, uniqueness, numeric
        summaries, min/max, and top-value information.
    """
    df = materialize(df)
    selected = columns_or_all(df, cols)
    total = df.height

    rows = []
    for col in selected:
        dtype = df.schema[col]
        col_type = classify_column(df, col, nint)
        series = df[col]
        n_missing = series.null_count()
        n_complete = total - n_missing
        n_unique = series.n_unique()
        mode, mode_count, top_values = _top_values(df, col, top_n)

        row = {
            "variable": col,
            "dtype": str(dtype),
            "type": col_type,
            "n": total,
            "n_complete": n_complete,
            "n_missing": n_missing,
            "pct_missing": n_missing / total if total else 0.0,
            "n_unique": n_unique,
            "pct_unique": n_unique / total if total else 0.0,
            "mean": None,
            "median": None,
            "sd": None,
            "min": None,
            "max": None,
            "mode": mode,
            "mode_count": mode_count,
            "top_values": top_values,
        }

        if is_numeric_dtype(dtype):
            non_null = series.drop_nulls()
            if len(non_null) > 0:
                row["mean"] = float(non_null.mean())
                row["median"] = float(non_null.median())
                row["sd"] = float(non_null.std()) if len(non_null) > 1 else None
                row["min"] = _stringify(non_null.min())
                row["max"] = _stringify(non_null.max())
        elif is_temporal_dtype(dtype):
            non_null = series.drop_nulls()
            if len(non_null) > 0:
                row["min"] = _stringify(non_null.min())
                row["max"] = _stringify(non_null.max())

        rows.append(row)

    return pl.DataFrame(rows)
