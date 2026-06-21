"""Association diagnostics across variable types."""

from collections.abc import Iterable
from itertools import combinations
from math import sqrt

import numpy as np
import polars as pl
from scipy import stats

from pyrsm.eda._utils import classify_column, columns_or_all, materialize


def _safe_float(value) -> float | None:
    """Return a finite float or None."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _numeric_numeric(pair: pl.DataFrame, var1: str, var2: str, method: str):
    """Compute numeric-vs-numeric correlation and p-value."""
    x = pair[var1].cast(pl.Float64).to_numpy()
    y = pair[var2].cast(pl.Float64).to_numpy()
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return None, None
    if method == "pearson":
        result = stats.pearsonr(x, y)
    else:
        result = stats.spearmanr(x, y)
    return _safe_float(result.statistic), _safe_float(result.pvalue)


def _categorical_categorical(pair: pl.DataFrame, var1: str, var2: str):
    """Compute Cramer's V and chi-square p-value."""
    table = pair.to_pandas()
    contingency = (
        table.groupby([var1, var2], observed=False).size().unstack(fill_value=0)
    )
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None, None
    chi2, p_value, _, _ = stats.chi2_contingency(contingency.to_numpy())
    n = contingency.to_numpy().sum()
    denom = n * min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = sqrt(chi2 / denom) if denom > 0 else None
    return _safe_float(cramers_v), _safe_float(p_value)


def _categorical_numeric(pair: pl.DataFrame, cat: str, num: str):
    """Compute eta-squared and one-way ANOVA p-value."""
    grouped = pair.group_by(cat, maintain_order=True).agg(pl.col(num).alias("values"))
    groups = [
        np.asarray(values, dtype=float)
        for values in grouped["values"].to_list()
        if len(values) > 0
    ]
    groups = [g for g in groups if np.isfinite(g).all()]
    if len(groups) < 2:
        return None, None

    y = np.concatenate(groups)
    if len(y) < 3 or np.nanstd(y) == 0:
        return None, None

    grand_mean = np.nanmean(y)
    ss_between = sum(len(g) * (np.nanmean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.nansum((y - grand_mean) ** 2)
    eta_squared = ss_between / ss_total if ss_total > 0 else None

    try:
        p_value = stats.f_oneway(*groups).pvalue
    except ValueError:
        p_value = None

    return _safe_float(eta_squared), _safe_float(p_value)


def associations(
    df: pl.DataFrame | pl.LazyFrame,
    cols: str | Iterable[str] | None = None,
    target: str | None = None,
    method: str = "pearson",
    nint: int = 25,
) -> pl.DataFrame:
    """
    Compute association diagnostics for numeric and categorical variables.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Data to inspect.
    cols : str | Iterable[str] | None
        Columns to include. If None, all columns are considered.
    target : str | None
        Optional target column. When set, associations are computed between the
        target and each other selected column.
    method : str
        Correlation method for numeric-numeric pairs: ``"pearson"`` or
        ``"spearman"``.
    nint : int
        Number of unique values below which integer columns are treated as
        categorical.

    Returns
    -------
    pl.DataFrame
        Association table with variables, type pairing, metric, value, p-value,
        and sample size.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")

    df = materialize(df)
    selected = columns_or_all(df, cols)
    if target is not None:
        if target not in df.columns:
            raise ValueError(f"Column not found: {target}")
        pair_specs = [(c, target) for c in selected if c != target]
    else:
        pair_specs = list(combinations(selected, 2))

    rows = []
    for var1, var2 in pair_specs:
        type1 = classify_column(df, var1, nint)
        type2 = classify_column(df, var2, nint)
        if "other" in (type1, type2) or "temporal" in (type1, type2):
            continue

        pair = df.select([var1, var2]).drop_nulls()
        n = pair.height
        if n == 0:
            continue

        if type1 == "numeric" and type2 == "numeric":
            association_type = "numeric_numeric"
            metric = method
            value, p_value = _numeric_numeric(pair, var1, var2, method)
        elif type1 == "categorical" and type2 == "categorical":
            association_type = "categorical_categorical"
            metric = "cramers_v"
            value, p_value = _categorical_categorical(pair, var1, var2)
        else:
            association_type = "categorical_numeric"
            metric = "eta_squared"
            cat, num = (var1, var2) if type1 == "categorical" else (var2, var1)
            value, p_value = _categorical_numeric(pair, cat, num)

        rows.append(
            {
                "var1": var1,
                "var2": var2,
                "type1": type1,
                "type2": type2,
                "association_type": association_type,
                "metric": metric,
                "value": value,
                "p_value": p_value,
                "n": n,
            }
        )

    result = pl.DataFrame(rows)
    if result.height == 0:
        return pl.DataFrame(
            schema={
                "var1": pl.String,
                "var2": pl.String,
                "type1": pl.String,
                "type2": pl.String,
                "association_type": pl.String,
                "metric": pl.String,
                "value": pl.Float64,
                "p_value": pl.Float64,
                "n": pl.Int64,
            }
        )
    return result.sort("value", descending=True, nulls_last=True)
