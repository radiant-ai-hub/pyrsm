"""Outlier diagnostics for numeric columns."""

from collections.abc import Iterable

import polars as pl

from pyrsm.eda._utils import columns_or_all, is_numeric_dtype, materialize


def _numeric_cols(df: pl.DataFrame, cols: str | Iterable[str] | None) -> list[str]:
    """Return selected numeric columns."""
    selected = columns_or_all(df, cols)
    numeric = [c for c in selected if is_numeric_dtype(df.schema[c])]
    if not numeric:
        raise ValueError("No numeric columns selected")
    return numeric


def _bounds(
    series: pl.Series, method: str, threshold: float
) -> tuple[float | None, float | None]:
    """Compute lower and upper outlier bounds for a numeric series."""
    values = series.drop_nulls().cast(pl.Float64)
    if len(values) == 0:
        return None, None

    if method == "iqr":
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        return q1 - threshold * iqr, q3 + threshold * iqr

    if method == "zscore":
        mean = values.mean()
        sd = values.std()
        if sd in (None, 0):
            return None, None
        return mean - threshold * sd, mean + threshold * sd

    median = values.median()
    mad = (values - median).abs().median()
    if mad in (None, 0):
        return None, None
    spread = threshold * mad / 0.6745
    return median - spread, median + spread


def outliers(
    df: pl.DataFrame | pl.LazyFrame,
    cols: str | Iterable[str] | None = None,
    method: str = "iqr",
    threshold: float | None = None,
    ret: str = "summary",
) -> pl.DataFrame:
    """
    Flag or summarize numeric outliers without modifying the input data.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Data to inspect.
    cols : str | Iterable[str] | None
        Numeric columns to inspect. If None, all numeric columns are used.
    method : str
        One of ``"iqr"``, ``"zscore"``, or ``"robust_zscore"``.
    threshold : float | None
        Cutoff multiplier. Defaults to 1.5 for IQR, 3.0 for z-score, and 3.5
        for robust z-score.
    ret : str
        ``"summary"`` for one row per variable, or ``"flags"`` for row-level
        boolean outlier indicators.

    Returns
    -------
    pl.DataFrame
        Outlier summary table or flag matrix.
    """
    if method not in ("iqr", "zscore", "robust_zscore"):
        raise ValueError("method must be 'iqr', 'zscore', or 'robust_zscore'")
    if ret not in ("summary", "flags"):
        raise ValueError("ret must be 'summary' or 'flags'")

    if threshold is None:
        threshold = {"iqr": 1.5, "zscore": 3.0, "robust_zscore": 3.5}[method]

    df = materialize(df)
    numeric = _numeric_cols(df, cols)
    bounds = {col: _bounds(df[col], method, threshold) for col in numeric}

    if ret == "flags":
        exprs = []
        for col in numeric:
            lower, upper = bounds[col]
            if lower is None or upper is None:
                exprs.append(pl.lit(False).alias(f"{col}_outlier"))
            else:
                exprs.append(
                    ((pl.col(col) < lower) | (pl.col(col) > upper))
                    .fill_null(False)
                    .alias(f"{col}_outlier")
                )
        return df.with_row_index("row_nr").select(["row_nr", *exprs])

    rows = []
    for col in numeric:
        values = df[col].drop_nulls().cast(pl.Float64)
        n = len(values)
        lower, upper = bounds[col]
        if lower is None or upper is None:
            n_outliers = 0
        else:
            n_outliers = values.filter((values < lower) | (values > upper)).len()
        rows.append(
            {
                "variable": col,
                "method": method,
                "threshold": float(threshold),
                "n": n,
                "n_outliers": n_outliers,
                "pct_outliers": n_outliers / n if n else 0.0,
                "lower": lower,
                "upper": upper,
                "min": values.min() if n else None,
                "max": values.max() if n else None,
            }
        )

    return pl.DataFrame(rows)
