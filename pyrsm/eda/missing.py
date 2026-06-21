"""Missing-data diagnostics."""

from collections.abc import Iterable

import polars as pl

from pyrsm.eda._utils import as_list, columns_or_all, materialize


def missing(
    df: pl.DataFrame | pl.LazyFrame,
    cols: str | Iterable[str] | None = None,
    by: str | Iterable[str] | None = None,
    patterns: bool = False,
) -> pl.DataFrame:
    """
    Summarize missing values by column, group, or missingness pattern.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Data to inspect.
    cols : str | Iterable[str] | None
        Columns to include. If None, all non-``by`` columns are inspected.
    by : str | Iterable[str] | None
        Optional grouping column(s). When set, returns one row per group with
        wide ``<col>_n_missing`` and ``<col>_pct_missing`` columns.
    patterns : bool
        If True, return one row per missingness pattern using 1 for missing and
        0 for observed.

    Returns
    -------
    pl.DataFrame
        Missing-value summary table.
    """
    df = materialize(df)
    by_cols = as_list(by, "by")
    selected = columns_or_all(df, cols)
    selected = [c for c in selected if c not in by_cols]
    if not selected:
        raise ValueError("No columns selected for missing-value diagnostics")

    if patterns:
        exprs = [pl.col(c).is_null().cast(pl.Int8).alias(c) for c in selected]
        pattern_df = (
            df.select(exprs)
            .group_by(selected, maintain_order=True)
            .agg(pl.len().alias("n"))
            .with_columns((pl.col("n") / df.height).alias("pct"))
            .sort("n", descending=True)
        )
        return pattern_df

    if by_cols:
        missing_exprs = [
            pl.col(c).null_count().alias(f"{c}_n_missing") for c in selected
        ]
        pct_exprs = [
            pl.col(c).is_null().mean().alias(f"{c}_pct_missing") for c in selected
        ]
        return df.group_by(by_cols, maintain_order=True).agg(
            [pl.len().alias("n"), *missing_exprs, *pct_exprs]
        )

    n = df.height
    rows = []
    for col in selected:
        n_missing = df[col].null_count()
        rows.append(
            {
                "variable": col,
                "dtype": str(df.schema[col]),
                "n": n,
                "n_missing": n_missing,
                "pct_missing": n_missing / n if n else 0.0,
                "n_complete": n - n_missing,
                "pct_complete": (n - n_missing) / n if n else 0.0,
            }
        )
    return pl.DataFrame(rows)
