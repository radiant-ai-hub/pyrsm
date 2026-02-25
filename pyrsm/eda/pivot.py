"""Pivot tables and crosstabs."""

import polars as pl

# Supported aggregation functions
AGG_FUNCTIONS = {
    "count": lambda col: pl.len(),
    "sum": lambda col: pl.col(col).sum(),
    "mean": lambda col: pl.col(col).mean(),
    "median": lambda col: pl.col(col).median(),
    "min": lambda col: pl.col(col).min(),
    "max": lambda col: pl.col(col).max(),
    "std": lambda col: pl.col(col).std(),
    "var": lambda col: pl.col(col).var(),
}

NORMALIZE_OPTIONS = {"row", "column", "total", "none", None}


def pivot(
    df: pl.DataFrame | pl.LazyFrame,
    rows: str | list[str],
    cols: str | None = None,
    values: str | None = None,
    agg: str = "count",
    normalize: str | None = None,
    totals: bool = False,
    fill: float | None = None,
) -> pl.DataFrame:
    """
    Create pivot tables and crosstabs.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Polars DataFrame or LazyFrame.
    rows : str | list[str]
        Row variable(s).
    cols : str | None
        Column variable for crosstab (optional).
    values : str | None
        Value column for aggregation (optional, uses count if None).
    agg : str
        Aggregation function. Default: ``"count"`` (or ``"mean"`` if values
        specified). Supported: count, sum, mean, median, min, max, std, var.
    normalize : str | None
        Normalization type: ``"row"``, ``"column"``, ``"total"``, or ``None``.
    totals : bool
        Whether to include row/column totals.
    fill : float | None
        Fill value for missing cells (only when no values variable). Default: None.

    Returns
    -------
    pl.DataFrame
        Pivot table or crosstab.

    Raises
    ------
    ValueError
        If ``agg`` or ``normalize`` is unknown.

    Examples
    --------
    >>> import polars as pl
    >>> import pyrsm as rsm
    >>> df = pl.DataFrame(
    ...     {"cut": ["A", "A", "B", "B"], "color": ["X", "Y", "X", "Y"], "price": [10, 20, 30, 40]}
    ... )
    >>> print(rsm.eda.pivot(df, rows="cut").sort("cut"))
    shape: (2, 2)
    ┌─────┬───────┐
    │ cut ┆ count │
    │ --- ┆ ---   │
    │ str ┆ u32   │
    ╞═════╪═══════╡
    │ A   ┆ 2     │
    │ B   ┆ 2     │
    └─────┴───────┘
    >>> out = rsm.eda.pivot(df, rows="cut", cols="color").sort("cut")
    >>> print(out.select(["cut", "X", "Y"]))
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ cut ┆ X   ┆ Y   │
    │ --- ┆ --- ┆ --- │
    │ str ┆ f64 ┆ f64 │
    ╞═════╪═════╪═════╡
    │ A   ┆ 1.0 ┆ 1.0 │
    │ B   ┆ 1.0 ┆ 1.0 │
    └─────┴─────┴─────┘
    """
    # Convert to LazyFrame for consistency
    if isinstance(df, pl.DataFrame):
        lf = df.lazy()
    else:
        lf = df

    # Normalize rows to list
    if isinstance(rows, str):
        rows_list = [rows]
    else:
        rows_list = list(rows)

    # Default agg based on whether values is specified
    if values and agg == "count":
        agg = "mean"

    # Validate agg
    if agg not in AGG_FUNCTIONS:
        raise ValueError(
            f"Unknown aggregation: {agg}\n"
            f"Supported: {', '.join(AGG_FUNCTIONS.keys())}"
        )

    # Validate normalize
    if normalize not in NORMALIZE_OPTIONS:
        raise ValueError(
            f"Unknown normalize option: {normalize}\n"
            f"Supported: row, column, total, none"
        )

    # Build aggregation expression
    if values:
        agg_expr = AGG_FUNCTIONS[agg](values).alias("value")
    else:
        agg_expr = pl.len().alias("value")

    # Frequency table (no cols variable)
    if not cols:
        result = lf.group_by(rows_list).agg(agg_expr)

        # Rename 'value' to more descriptive name (col_agg format, consistent with explore)
        value_col = f"{values}_{agg}" if values else "count"
        result = result.rename({"value": value_col})

        # Collect for further processing
        pivoted = result.collect()

        # Normalization for frequency table
        if normalize and normalize != "none":
            pivoted = pivoted.with_columns(
                (pl.col(value_col) / pl.col(value_col).sum() * 100).alias(
                    f"{value_col}_pct"
                )
            )

        # Totals for frequency table
        if totals:
            # Cast row columns to string to allow "Total" label
            for row_col in rows_list:
                pivoted = pivoted.with_columns(pl.col(row_col).cast(pl.Utf8))
            # Cast numeric columns to Float64
            for col in pivoted.columns:
                if col not in rows_list:
                    pivoted = pivoted.with_columns(pl.col(col).cast(pl.Float64))

            # Compute totals
            total_vals = {row_col: "Total" for row_col in rows_list}
            for col in pivoted.columns:
                if col not in rows_list:
                    total_vals[col] = float(pivoted[col].sum())

            total_row = pl.DataFrame([total_vals])
            pivoted = pl.concat([pivoted, total_row])

        # Fill missing values if specified (only when no values variable)
        if fill is not None and not values:
            pivoted = pivoted.fill_null(fill)

        return pivoted

    # Crosstab (rows + cols) - need to collect for pivot
    group_cols = rows_list + [cols]
    grouped = lf.group_by(group_cols).agg(agg_expr)

    # Polars LazyFrame.pivot requires collect first
    df_grouped = grouped.collect()

    # Pivot the data
    pivoted = df_grouped.pivot(
        on=cols,
        index=rows_list,
        values="value",
        aggregate_function=None,  # Already aggregated
    )

    # Get the column names (excluding index columns)
    data_cols = [c for c in pivoted.columns if c not in rows_list]

    # Cast row columns to string if needed (for totals "Total" label)
    if totals:
        for row_col in rows_list:
            if pivoted[row_col].dtype in (pl.Categorical, pl.Enum) or totals:
                pivoted = pivoted.with_columns(pl.col(row_col).cast(pl.Utf8))

    # Cast numeric columns to Float64 for consistency with totals
    for col in data_cols:
        pivoted = pivoted.with_columns(pl.col(col).cast(pl.Float64))

    # Add totals if requested
    if totals:
        # Row totals
        pivoted = pivoted.with_columns(pl.sum_horizontal(data_cols).alias("Total"))
        data_cols.append("Total")

        # Column totals (as a new row)
        col_totals = {row_col: "Total" for row_col in rows_list}
        for col in data_cols:
            col_totals[col] = float(pivoted[col].sum())

        total_row = pl.DataFrame([col_totals])
        pivoted = pl.concat([pivoted, total_row])

    # Normalize if requested
    if normalize and normalize != "none":
        numeric_cols = [c for c in data_cols if c != "Total"] if totals else data_cols

        if normalize == "row":
            # Each row sums to 100%
            row_sums = pivoted.select(pl.sum_horizontal(numeric_cols)).to_series()
            for col in numeric_cols:
                pivoted = pivoted.with_columns(
                    (pl.col(col) / row_sums * 100).alias(col)
                )
            if totals and "Total" in pivoted.columns:
                pivoted = pivoted.with_columns(pl.lit(100.0).alias("Total"))

        elif normalize == "column":
            # Each column sums to 100%
            for col in numeric_cols:
                col_sum = pivoted[col].sum()
                if col_sum > 0:
                    pivoted = pivoted.with_columns(
                        (pl.col(col) / col_sum * 100).alias(col)
                    )

        elif normalize == "total":
            # All cells sum to 100%
            grand_total = sum(pivoted[col].sum() for col in numeric_cols)
            if grand_total > 0:
                for col in numeric_cols:
                    pivoted = pivoted.with_columns(
                        (pl.col(col) / grand_total * 100).alias(col)
                    )

    # Fill missing values if specified (only when no values variable)
    if fill is not None and not values:
        pivoted = pivoted.fill_null(fill)

    return pivoted
