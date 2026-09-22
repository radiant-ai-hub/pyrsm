"""Summary statistics for numeric columns."""

from typing import Literal

import polars as pl

from pyrsm.eda.agg_functions import AGG_FUNCTIONS, resolve_agg

#: Supported summary functions. Shared with ``pivot`` so the two tools can
#: never disagree about which metrics exist — see
#: :mod:`pyrsm.eda.agg_functions` for the radiant.data parity set.
EXPLORE_FUNCTIONS = AGG_FUNCTIONS

DEFAULT_AGG = ["mean", "median", "min", "max", "sd"]

# Numeric dtypes for auto-detection
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

# Categorical dtypes for to_dummies conversion
CATEGORICAL_DTYPES = (pl.Categorical, pl.Enum, pl.String, pl.Utf8)


def _grouped_result(
    lf: pl.LazyFrame,
    by_cols: list[str],
    cols: list[str],
    agg: list[str],
    exprs: list[pl.Expr],
    header: str,
) -> pl.DataFrame:
    """Summarize by group, laid out according to ``header``.

    ``group_by().agg()`` gives one wide row per group with ``{col}_{func}``
    columns — a shape that honours neither ``header`` option. Reshape it the
    way radiant.data does: a tidy ``by..., variable, <fun>...`` table for
    ``header="function"``, flipped to ``by..., statistic, <variable>...``
    for ``header="variable"``.
    """
    wide = lf.group_by(by_cols).agg(exprs).collect()

    # Map each generated column back to its (variable, function) pair rather
    # than parsing the name — a column called "unit_price" would make
    # "unit_price_mean" ambiguous to split on "_".
    keys = [f"{col}_{func}" for col in cols for func in agg]
    to_variable = {f"{col}_{func}": col for col in cols for func in agg}
    to_statistic = {f"{col}_{func}": func for col in cols for func in agg}
    var_order = {col: i for i, col in enumerate(cols)}
    fun_order = {func: i for i, func in enumerate(agg)}

    long = wide.unpivot(
        index=by_cols, on=keys, variable_name="_key", value_name="_value"
    ).with_columns(
        pl.col("_key").replace_strict(to_variable).alias("variable"),
        pl.col("_key").replace_strict(to_statistic).alias("statistic"),
        pl.col("_key")
        .replace_strict({k: var_order[v] for k, v in to_variable.items()})
        .alias("_var_ord"),
        pl.col("_key")
        .replace_strict({k: fun_order[v] for k, v in to_statistic.items()})
        .alias("_fun_ord"),
    )

    # ``pivot`` keeps first-appearance order, so sort first to get stable
    # groups and to keep variables / statistics in the order requested
    # rather than alphabetical.
    if header == "variable":
        long = long.sort([*by_cols, "_fun_ord", "_var_ord"])
        return long.pivot(
            on="variable", index=[*by_cols, "statistic"], values="_value"
        )
    long = long.sort([*by_cols, "_var_ord", "_fun_ord"])
    return long.pivot(on="statistic", index=[*by_cols, "variable"], values="_value")


def explore(
    df: pl.DataFrame | pl.LazyFrame,
    cols: list[str] | None = None,
    agg: list[str] | None = None,
    by: str | list[str] | None = None,
    to_dummies: bool = True,
    header: Literal["function", "variable"] = "function",
) -> pl.DataFrame:
    """
    Compute summary statistics for numeric columns.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Polars DataFrame or LazyFrame.
    cols : list[str] | None
        Column names to summarize. If None, uses all numeric columns (and
        dummy-encoded categorical columns when ``to_dummies=True``).
    agg : list[str] | None
        Aggregation functions to compute. Default: ``["mean", "median", "min",
        "max", "sd"]``. Supported: mean, median, sum, std, sd, var, min, max,
        count, n, n_unique, n_missing, null_count.
    by : str | list[str] | None
        Optional column(s) to group by. Pass a list to group by several
        variables at once.
    to_dummies : bool
        If True, convert categorical/Enum/String columns to dummy variables
        (drop_first=True) and include them in the summary.
    header : Literal["function", "variable"]
        Layout of the result table. ``"function"`` (default) puts statistic
        names across the top (variables as rows). ``"variable"`` puts variable
        names across the top (statistics as rows).

    Returns
    -------
    pl.DataFrame
        DataFrame with summary statistics.

    Raises
    ------
    ValueError
        If an aggregation function is unknown or if no numeric columns are
        detected.

    Examples
    --------
    >>> import polars as pl
    >>> import pyrsm as rsm
    >>> df = pl.DataFrame({"price": [10, 20, 30], "carat": [1.0, 2.0, 3.0]})
    >>> print(rsm.eda.explore(df, cols=["price", "carat"], agg=["mean", "min"]))
    shape: (2, 3)
    ┌──────────┬──────┬──────┐
    │ variable ┆ mean ┆ min  │
    │ ---      ┆ ---  ┆ ---  │
    │ str      ┆ f64  ┆ f64  │
    ╞══════════╪══════╪══════╡
    │ price    ┆ 20.0 ┆ 10.0 │
    │ carat    ┆ 2.0  ┆ 1.0  │
    └──────────┴──────┴──────┘
    >>> print(rsm.eda.explore(df, cols=["price"], agg=["mean", "max"], header="variable"))
    shape: (2, 2)
    ┌───────────┬───────┐
    │ statistic ┆ price │
    │ ---       ┆ ---   │
    │ str       ┆ f64   │
    ╞═══════════╪═══════╡
    │ mean      ┆ 20.0  │
    │ max       ┆ 30.0  │
    └───────────┴───────┘

    Grouped output is tidy, and ``header`` flips it just like it does for
    an ungrouped summary.

    >>> df2 = pl.DataFrame({"g": ["a", "a", "b"], "x": [1.0, 2.0, 3.0]})
    >>> print(rsm.eda.explore(df2, cols=["x"], by="g", agg=["mean", "max"]))
    shape: (2, 4)
    ┌─────┬──────────┬──────┬─────┐
    │ g   ┆ variable ┆ mean ┆ max │
    │ --- ┆ ---      ┆ ---  ┆ --- │
    │ str ┆ str      ┆ f64  ┆ f64 │
    ╞═════╪══════════╪══════╪═════╡
    │ a   ┆ x        ┆ 1.5  ┆ 2.0 │
    │ b   ┆ x        ┆ 3.0  ┆ 3.0 │
    └─────┴──────────┴──────┴─────┘
    >>> print(
    ...     rsm.eda.explore(
    ...         df2, cols=["x"], by="g", agg=["mean", "max"], header="variable"
    ...     )
    ... )
    shape: (4, 3)
    ┌─────┬───────────┬─────┐
    │ g   ┆ statistic ┆ x   │
    │ --- ┆ ---       ┆ --- │
    │ str ┆ str       ┆ f64 │
    ╞═════╪═══════════╪═════╡
    │ a   ┆ mean      ┆ 1.5 │
    │ a   ┆ max       ┆ 2.0 │
    │ b   ┆ mean      ┆ 3.0 │
    │ b   ┆ max       ┆ 3.0 │
    └─────┴───────────┴─────┘

    ``by`` also accepts several grouping variables.

    >>> df3 = pl.DataFrame(
    ...     {"g": ["a", "a", "b"], "h": ["x", "y", "x"], "v": [1.0, 2.0, 3.0]}
    ... )
    >>> print(rsm.eda.explore(df3, cols=["v"], by=["g", "h"], agg=["mean"]))
    shape: (3, 4)
    ┌─────┬─────┬──────────┬──────┐
    │ g   ┆ h   ┆ variable ┆ mean │
    │ --- ┆ --- ┆ ---      ┆ ---  │
    │ str ┆ str ┆ str      ┆ f64  │
    ╞═════╪═════╪══════════╪══════╡
    │ a   ┆ x   ┆ v        ┆ 1.0  │
    │ a   ┆ y   ┆ v        ┆ 2.0  │
    │ b   ┆ x   ┆ v        ┆ 3.0  │
    └─────┴─────┴──────────┴──────┘
    """
    # Normalize ``by`` to a list so one and many grouping variables follow
    # the same path.
    by_cols = [by] if isinstance(by, str) else list(by or [])

    # Materialize if LazyFrame
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # Convert categorical columns to dummies if requested
    if to_dummies:
        schema = df.schema
        # When cols is specified, only convert categoricals listed in cols
        candidates = cols if cols is not None else list(schema.keys())
        cat_cols = [
            name
            for name in candidates
            if name in schema
            and (
                isinstance(schema[name], CATEGORICAL_DTYPES)
                or schema[name] in CATEGORICAL_DTYPES
            )
            and name not in by_cols
        ]
        if cat_cols:
            df = df.to_dummies(columns=cat_cols, drop_first=True)
            # Cast dummy columns (UInt8) to Float64 for consistent stats
            df = df.cast(
                {
                    col: pl.Float64
                    for col, dtype in df.schema.items()
                    if dtype == pl.UInt8
                }
            )

    lf = df.lazy()

    # Default aggregation functions
    if agg is None:
        agg = DEFAULT_AGG

    # Validate aggregation functions (raises with the supported set)
    for func in agg:
        resolve_agg(func)

    # Auto-detect numeric columns if none specified
    if cols is None:
        schema = lf.collect_schema()
        cols = [
            name
            for name, dtype in schema.items()
            if isinstance(dtype, NUMERIC_DTYPES) or dtype in NUMERIC_DTYPES
        ]
        if not cols:
            raise ValueError("No numeric columns found in dataset")

    # Build aggregation expressions
    exprs = []
    for col in cols:
        for func in agg:
            expr = resolve_agg(func)(col).alias(f"{col}_{func}")
            exprs.append(expr)

    # Execute with or without grouping
    if by_cols:
        result = _grouped_result(lf, by_cols, cols, agg, exprs, header)
    elif header == "variable":
        # Statistics as rows, variables as columns
        wide_result = lf.select(exprs).collect()
        rows = []
        for func in agg:
            row = {"statistic": func}
            for col in cols:
                row[col] = wide_result[f"{col}_{func}"][0]
            rows.append(row)
        result = pl.DataFrame(rows)
    else:
        # Variables as rows, functions as columns (default)
        wide_result = lf.select(exprs).collect()
        rows = []
        for col in cols:
            row = {"variable": col}
            for func in agg:
                row[func] = wide_result[f"{col}_{func}"][0]
            rows.append(row)
        result = pl.DataFrame(rows)

    return result
