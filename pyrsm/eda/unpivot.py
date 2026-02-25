"""Convert wide format to long format (reverse of pivot)."""

import polars as pl


def unpivot(
    df: pl.DataFrame | pl.LazyFrame,
    on: str | list[str] | None = None,
    id_vars: str | list[str] | None = None,
    variable_name: str = "variable",
    value_name: str = "value",
) -> pl.DataFrame:
    """
    Convert wide format data to long format (reverse of pivot).

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Polars DataFrame or LazyFrame.
    on : str | list[str] | None
        Column(s) to unpivot. If None, uses all columns not in ``id_vars``.
    id_vars : str | list[str] | None
        Column(s) to keep as identifier variables.
    variable_name : str
        Name for the new variable column. Default: ``"variable"``.
    value_name : str
        Name for the new value column. Default: ``"value"``.

    Returns
    -------
    pl.DataFrame
        DataFrame in long format.

    Examples
    --------
    >>> import polars as pl
    >>> import pyrsm as rsm
    >>> df = pl.DataFrame({"name": ["A", "B"], "Q1": [10, 20], "Q2": [30, 40]})
    >>> out = rsm.eda.unpivot(df, on=["Q1", "Q2"], id_vars="name")
    >>> print(out.sort(["name", "variable"]))
    shape: (4, 3)
    ┌──────┬──────────┬───────┐
    │ name ┆ variable ┆ value │
    │ ---  ┆ ---      ┆ ---   │
    │ str  ┆ str      ┆ i64   │
    ╞══════╪══════════╪═══════╡
    │ A    ┆ Q1       ┆ 10    │
    │ A    ┆ Q2       ┆ 30    │
    │ B    ┆ Q1       ┆ 20    │
    │ B    ┆ Q2       ┆ 40    │
    └──────┴──────────┴───────┘
    """
    # Convert to DataFrame if LazyFrame
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # Normalize id_vars to list
    if id_vars is None:
        id_vars_list = []
    elif isinstance(id_vars, str):
        id_vars_list = [id_vars]
    else:
        id_vars_list = list(id_vars)

    # Normalize on to list (if specified)
    if on is None:
        value_vars_list = None
    elif isinstance(on, str):
        value_vars_list = [on]
    else:
        value_vars_list = list(on)

    # Call Polars unpivot
    result = df.unpivot(
        on=value_vars_list,
        index=id_vars_list,
        variable_name=variable_name,
        value_name=value_name,
    )

    return result
