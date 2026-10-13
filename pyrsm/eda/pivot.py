"""Pivot tables and crosstabs."""

import polars as pl

from pyrsm.eda.agg_functions import AGG_FUNCTIONS, resolve_agg


NORMALIZE_OPTIONS = {"row", "column", "total", "none", None}


def _as_percent(df: pl.DataFrame, cols: list[str], dec: int) -> pl.DataFrame:
    """Format ``cols`` as percentage strings: ``0.2341`` → ``"23.41%"``."""
    if not cols:
        return df
    return df.with_columns(
        [
            pl.col(col)
            .map_elements(
                lambda v, _dec=dec: f"{v * 100:.{_dec}f}%",
                return_dtype=pl.Utf8,
            )
            .alias(col)
            for col in cols
        ]
    )


def pivot(
    df: pl.DataFrame | pl.LazyFrame | dict,
    rows: str | list[str],
    cols: str | None = None,
    values: str | None = None,
    agg: str = "count",
    normalize: str | None = None,
    totals: bool = False,
    fill: float | None = None,
    perc: bool = False,
    dec: int = 2,
    data_filter: str = "",
    sort: str = "",
    slice: str = "",  # noqa: A002 - the argument name every class publishes
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
        Normalized cells are **proportions** (0-1), matching R's
        ``radiant.data::pivotr``. Use ``perc=True`` to show them as
        percentages instead.
    totals : bool
        Whether to include row/column totals.
    fill : float | None
        Fill value for missing cells (only when no values variable). Default: None.
    perc : bool
        Display the value cells as percentages (``0.234`` → ``"23.40%"``)
        rather than as numbers. Opt-in, like the "Percentage" checkbox in
        Radiant. Raw counts are left alone; only cells that are already on
        a proportion scale (or any other non-count aggregate) are formatted.
    dec : int
        Number of decimals used when ``perc=True``. Default: 2.
    data_filter : str
        Rows to keep, as a Python expression (e.g., ``"price > 1000"``),
        applied before anything is computed. Then ``sort`` (e.g., ``"-price"``)
        and ``slice`` (e.g., ``"1:100"``), in that order, as in the model and
        basics classes.
    sort : str
        Columns to sort by, comma separated; a leading ``-`` sorts descending.
    slice : str
        Rows to keep after filtering and sorting (e.g., ``"1:100"``).

    Returns
    -------
    pl.DataFrame
        Pivot table or crosstab. Columns are numeric unless ``perc=True``,
        which formats the value cells as strings.

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

    Normalized cells are proportions ...

    >>> out = rsm.eda.pivot(df, rows="cut", cols="color", normalize="total").sort("cut")
    >>> print(out.select(["cut", "X", "Y"]))
    shape: (2, 3)
    ┌─────┬──────┬──────┐
    │ cut ┆ X    ┆ Y    │
    │ --- ┆ ---  ┆ ---  │
    │ str ┆ f64  ┆ f64  │
    ╞═════╪══════╪══════╡
    │ A   ┆ 0.25 ┆ 0.25 │
    │ B   ┆ 0.25 ┆ 0.25 │
    └─────┴──────┴──────┘

    ... unless you ask for percentages explicitly.

    >>> out = rsm.eda.pivot(
    ...     df, rows="cut", cols="color", normalize="total", perc=True
    ... ).sort("cut")
    >>> print(out.select(["cut", "X", "Y"]))
    shape: (2, 3)
    ┌─────┬────────┬────────┐
    │ cut ┆ X      ┆ Y      │
    │ --- ┆ ---    ┆ ---    │
    │ str ┆ str    ┆ str    │
    ╞═════╪════════╪════════╡
    │ A   ┆ 25.00% ┆ 25.00% │
    │ B   ┆ 25.00% ┆ 25.00% │
    └─────┴────────┴────────┘
    """
    from pyrsm.utils import named_data

    # ``{name: frame}`` names the dataset, as in the model and basics classes.
    _, df = named_data(df)
    if data_filter or sort or slice:
        from pyrsm.data_scope import apply_scope

        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        df = apply_scope(df, data_filter, sort, slice)

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

    # Validate agg (raises with the supported set)
    agg_builder = resolve_agg(agg)

    # Validate normalize
    if normalize not in NORMALIZE_OPTIONS:
        raise ValueError(
            f"Unknown normalize option: {normalize}\nSupported: row, column, total, none"
        )
    normalized = bool(normalize) and normalize != "none"

    # Build aggregation expression
    if values:
        agg_expr = agg_builder(values).alias("value")
    else:
        agg_expr = pl.len().alias("value")

    # Frequency table (no cols variable)
    if not cols:
        # In the order of the row variables' levels (radiant.data), not the
        # order the groups happen to come out of group_by.
        result = lf.group_by(rows_list).agg(agg_expr).sort(rows_list, nulls_last=True)

        # Rename 'value' to more descriptive name (col_agg format, consistent with explore)
        value_col = f"{values}_{agg}" if values else "count"
        result = result.rename({"value": value_col})

        # Collect for further processing
        pivoted = result.collect()

        # ``perc`` never applies to raw frequencies, so record which value
        # columns are plain integer counts before ``totals`` casts every
        # numeric column to Float64.
        perc_cols = [
            col
            for col in pivoted.columns
            if col not in rows_list and pivoted[col].dtype.is_float()
        ]

        # Normalization for frequency table. The share is a proportion, as
        # in radiant.data; ``perc`` is what turns it into a percentage.
        if normalized:
            share_col = f"{value_col}_perc" if perc else f"{value_col}_prop"
            pivoted = pivoted.with_columns(
                (pl.col(value_col) / pl.col(value_col).sum()).alias(share_col)
            )
            perc_cols.append(share_col)

        # Totals for frequency table
        if totals:
            # Cast row columns to string to allow "Total" label
            for row_col in rows_list:
                pivoted = pivoted.with_columns(pl.col(row_col).cast(pl.Utf8))
            # Cast numeric columns to Float64
            for col in pivoted.columns:
                if col not in rows_list:
                    pivoted = pivoted.with_columns(pl.col(col).cast(pl.Float64))

            # Compute totals: the aggregate over all rows (a mean of the
            # whole data, not the sum of the group means); shares add to 1.
            total_vals = {row_col: "Total" for row_col in rows_list}
            for col in pivoted.columns:
                if col not in rows_list:
                    total_vals[col] = float(pivoted[col].sum())
            total_vals[value_col] = float(lf.select(agg_expr).collect().item())

            total_row = pl.DataFrame([total_vals])
            pivoted = pl.concat([pivoted, total_row])

        # Fill missing values if specified (only when no values variable)
        if fill is not None and not values:
            pivoted = pivoted.fill_null(fill)

        if perc:
            pivoted = _as_percent(pivoted, perc_cols, dec)

        return pivoted

    # Crosstab (rows + cols) - need to collect for pivot
    group_cols = rows_list + [cols]
    grouped = lf.group_by(group_cols).agg(agg_expr)

    # Polars LazyFrame.pivot requires collect first; rows and columns in the
    # order of the variables' levels (radiant.data).
    df_grouped = grouped.collect().sort(group_cols, nulls_last=True)

    # Pivot the data
    pivoted = df_grouped.pivot(
        on=cols,
        index=rows_list,
        values="value",
        aggregate_function=None,  # Already aggregated
    )
    col_levels = [str(v) for v in df_grouped[cols].unique().sort(nulls_last=True).to_list()]
    pivoted = pivoted.select(rows_list + [c for c in col_levels if c in pivoted.columns]
                             + [c for c in pivoted.columns if c not in rows_list and c not in col_levels])

    # Get the column names (excluding index columns)
    data_cols = [c for c in pivoted.columns if c not in rows_list]

    # Cast row columns to string if needed (for totals "Total" label)
    if totals:
        for row_col in rows_list:
            if pivoted[row_col].dtype in (pl.Categorical, pl.Enum) or totals:
                pivoted = pivoted.with_columns(pl.col(row_col).cast(pl.Utf8))

    # ``perc`` never applies to raw frequencies, so note whether the cells
    # are plain counts before the Float64 cast erases that distinction.
    count_cells = all(pivoted[col].dtype.is_integer() for col in data_cols)

    # Cast numeric columns to Float64 for consistency with totals
    for col in data_cols:
        pivoted = pivoted.with_columns(pl.col(col).cast(pl.Float64))

    # Marginals: the same aggregate as the cells, applied to each row group,
    # each column level and the whole data (radiant.data). For a count (or a
    # sum) that is the sum of the cells; for a mean it is the mean of the
    # group, not the sum of the cell means. Computed before any totals row or
    # column is attached, so normalizing never counts an observation twice.
    row_marg = (
        lf.group_by(rows_list).agg(agg_expr).collect()
        .with_columns(pl.col(c).cast(pivoted[c].dtype) for c in rows_list)
    )
    raw_row_totals = (
        pivoted.select(rows_list)
        .join(row_marg, on=rows_list, how="left", nulls_equal=True, maintain_order="left")
        .get_column("value")
        .cast(pl.Float64)
    )
    col_marg = lf.group_by(cols).agg(agg_expr).collect()
    col_marg_map = {
        ("null" if k is None else str(k)): v
        for k, v in zip(col_marg[cols].to_list(), col_marg["value"].to_list())
    }
    raw_col_totals = {
        col: float(col_marg_map[col]) if col_marg_map.get(col) is not None else float("nan")
        for col in data_cols
    }
    grand_total = float(lf.select(agg_expr).collect().item())

    # Normalize the data cells. The result is a proportion (0-1), as in
    # radiant.data; ``perc`` below is the opt-in that renders percentages.
    if normalized:
        if normalize == "row":
            # Each row sums to 1
            for col in data_cols:
                pivoted = pivoted.with_columns(
                    (pl.col(col) / raw_row_totals).alias(col)
                )

        elif normalize == "column":
            # Each column sums to 1
            for col in data_cols:
                col_sum = raw_col_totals[col]
                if col_sum > 0:
                    pivoted = pivoted.with_columns((pl.col(col) / col_sum).alias(col))

        elif normalize == "total":
            # All cells sum to 1
            if grand_total > 0:
                for col in data_cols:
                    pivoted = pivoted.with_columns(
                        (pl.col(col) / grand_total).alias(col)
                    )

    # Add totals if requested. The totals summarize whatever the cells now
    # hold, so they are expressed on the same scale as the normalized cells:
    # the margin the normalization pins to 1 reads 1, and the other margin
    # shows that variable's share of the grand total.
    if totals:
        # Row totals (right-hand "Total" column)
        if normalize == "row":
            row_total_values = pl.Series("Total", [1.0] * pivoted.height)
        elif normalize in ("column", "total"):
            row_total_values = (
                (raw_row_totals / grand_total).alias("Total")
                if grand_total > 0
                else pl.Series("Total", [0.0] * pivoted.height)
            )
        else:
            row_total_values = raw_row_totals.alias("Total")
        pivoted = pivoted.with_columns(row_total_values)
        data_cols.append("Total")

        # Column totals (bottom "Total" row)
        col_totals: dict[str, object] = {row_col: "Total" for row_col in rows_list}
        for col in data_cols:
            if col == "Total":
                # Bottom-right corner: the whole table, so 1 once normalized.
                col_totals[col] = 1.0 if normalized else grand_total
            elif normalize == "column":
                col_totals[col] = 1.0
            elif normalize in ("row", "total"):
                col_totals[col] = (
                    raw_col_totals[col] / grand_total if grand_total > 0 else 0.0
                )
            else:
                col_totals[col] = raw_col_totals[col]

        total_row = pl.DataFrame([col_totals])
        pivoted = pl.concat([pivoted, total_row])

    # Fill missing values if specified (only when no values variable)
    if fill is not None and not values:
        pivoted = pivoted.fill_null(fill)

    if perc and (normalized or not count_cells):
        pivoted = _as_percent(pivoted, data_cols, dec)

    return pivoted
