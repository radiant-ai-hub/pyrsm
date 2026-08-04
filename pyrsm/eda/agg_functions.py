"""Aggregation functions shared by ``explore`` and ``pivot``.

One registry, so the two tools can never disagree about which metrics
exist. The set and the semantics mirror R's ``radiant.data`` — the
``radiant.functions`` option in ``inst/app/global.R`` plus the metric
definitions in ``R/explore.R``.
"""

from functools import lru_cache

import polars as pl


@lru_cache(maxsize=1)
def _get_scipy_stats():
    """Lazy load scipy.stats — only needed for skew / kurtosis / me."""
    from scipy import stats

    return stats


# 97.5th percentile of the standard normal, for ``meprop`` (radiant uses
# ``qnorm(conf_lev / 2 + .5)`` with conf_lev = 0.95).
_Z_975 = 1.959963984540054


def _percentile_expr(col: str, p: float) -> pl.Expr:
    """Build a polars expression for the ``p`` percentile (0-1)."""
    return pl.col(col).quantile(p)


def _batch(col: str, fn) -> pl.Expr:
    """Run ``fn`` over a group's non-null values via numpy."""

    def _apply(s: pl.Series) -> float:
        arr = s.drop_nulls().to_numpy()
        if arr.size < 2:
            return float("nan")
        return float(fn(arr))

    return pl.col(col).map_batches(_apply, return_dtype=pl.Float64, returns_scalar=True)


def _skew_expr(col: str) -> pl.Expr:
    """Sample skewness (radiant: ``skew``)."""
    return _batch(col, lambda a: _get_scipy_stats().skew(a, bias=False))


def _kurtosis_expr(col: str) -> pl.Expr:
    """Excess kurtosis (radiant: ``kurtosi``)."""
    return _batch(col, lambda a: _get_scipy_stats().kurtosis(a, bias=False))


def _me_expr(col: str) -> pl.Expr:
    """Margin of error: ``se * qt(0.975, n - 1)`` (radiant: ``me``)."""

    def _me(arr):
        n = arr.size
        se = arr.std(ddof=1) / (n**0.5)
        return se * _get_scipy_stats().t.ppf(0.975, n - 1)

    return _batch(col, _me)


def _prop_expr(col: str) -> pl.Expr:
    """Proportion of the maximum value (radiant: ``prop``).

    radiant returns the share of the largest value for a numeric column
    (and of the first level for a factor). For a 0/1 column that is the
    share of 1s, which is the common case.
    """
    return (pl.col(col) == pl.col(col).max()).mean()


def _varprop_expr(col: str) -> pl.Expr:
    """``p * (1 - p)`` where ``p`` is :func:`_prop_expr`."""
    return _prop_expr(col) * (1 - _prop_expr(col))


def _varpop_expr(col: str) -> pl.Expr:
    """Population variance: ``var * (n - 1) / n`` (radiant: ``varpop``)."""
    n = pl.col(col).count()
    return pl.col(col).var() * (n - 1) / n


def _seprop_expr(col: str) -> pl.Expr:
    """``sqrt(varprop / n)`` (radiant: ``seprop``)."""
    return (_varprop_expr(col) / pl.col(col).count()).sqrt()


# Every metric radiant.data offers, keyed by the name radiant uses. Aliases
# for the same statistic are registered alongside (``sd``/``std``,
# ``kurtosi``/``kurtosis``, ``n_obs``/``count``/``n``, ...) so code written
# against either vocabulary works.
AGG_FUNCTIONS = {
    # --- Counts ----------------------------------------------------
    # radiant's n_obs is ``length(x)`` — rows in the group, nulls included.
    "n_obs": lambda col: pl.len(),
    "count": lambda col: pl.len(),
    "n": lambda col: pl.len(),
    "n_missing": lambda col: pl.col(col).null_count(),
    "null_count": lambda col: pl.col(col).null_count(),
    "n_distinct": lambda col: pl.col(col).n_unique(),
    "n_unique": lambda col: pl.col(col).n_unique(),
    # --- Central tendency ------------------------------------------
    "mean": lambda col: pl.col(col).mean(),
    "median": lambda col: pl.col(col).median(),
    "modal": lambda col: pl.col(col).mode().first(),
    "min": lambda col: pl.col(col).min(),
    "max": lambda col: pl.col(col).max(),
    "sum": lambda col: pl.col(col).sum(),
    # --- Spread ----------------------------------------------------
    "var": lambda col: pl.col(col).var(),
    "sd": lambda col: pl.col(col).std(),
    "std": lambda col: pl.col(col).std(),  # polars-style alias
    "se": lambda col: pl.col(col).std() / pl.col(col).count().cast(pl.Float64).sqrt(),
    "me": _me_expr,
    "cv": lambda col: pl.col(col).std() / pl.col(col).mean(),
    "varpop": _varpop_expr,
    "sdpop": lambda col: _varpop_expr(col).sqrt(),
    "IQR": lambda col: _percentile_expr(col, 0.75) - _percentile_expr(col, 0.25),
    "iqr": lambda col: _percentile_expr(col, 0.75) - _percentile_expr(col, 0.25),
    # --- Proportions -----------------------------------------------
    "prop": _prop_expr,
    "varprop": _varprop_expr,
    "sdprop": lambda col: _varprop_expr(col).sqrt(),
    "seprop": _seprop_expr,
    "meprop": lambda col: _seprop_expr(col) * _Z_975,
    # --- Distributional shape --------------------------------------
    "skew": _skew_expr,
    "kurtosi": _kurtosis_expr,  # radiant's spelling
    "kurtosis": _kurtosis_expr,
}


def _register_percentiles() -> None:
    """Add ``p01`` … ``p99`` plus radiant's ``p025`` / ``p975``."""
    for p in range(1, 100):
        AGG_FUNCTIONS[f"p{p:02d}"] = (
            lambda col, _p=p / 100.0: _percentile_expr(col, _p)
        )
    for key, value in (("p025", 0.025), ("p975", 0.975)):
        AGG_FUNCTIONS[key] = lambda col, _p=value: _percentile_expr(col, _p)


_register_percentiles()


# The menu radiant.data shows, in its order, as ``(key, label)`` pairs.
# Mirrors ``options(radiant.functions = ...)``. This is what a UI should
# offer; ``AGG_FUNCTIONS`` additionally accepts aliases and every ``pNN``.
RADIANT_FUNCTIONS: list[tuple[str, str]] = [
    ("n_obs", "n_obs"),
    ("n_missing", "n_missing"),
    ("n_distinct", "n_distinct"),
    ("mean", "mean"),
    ("median", "median"),
    ("modal", "modal"),
    ("min", "min"),
    ("max", "max"),
    ("sum", "sum"),
    ("var", "var"),
    ("sd", "sd"),
    ("se", "se"),
    ("me", "me"),
    ("cv", "cv"),
    ("prop", "prop"),
    ("varprop", "varprop"),
    ("sdprop", "sdprop"),
    ("seprop", "seprop"),
    ("meprop", "meprop"),
    ("varpop", "varpop"),
    ("sdpop", "sdpop"),
    ("p01", "1%"),
    ("p025", "2.5%"),
    ("p05", "5%"),
    ("p10", "10%"),
    ("p25", "25%"),
    ("p75", "75%"),
    ("p90", "90%"),
    ("p95", "95%"),
    ("p975", "97.5%"),
    ("p99", "99%"),
    ("skew", "skew"),
    ("kurtosi", "kurtosis"),
    ("IQR", "IQR"),
]

#: Ordered keys of :data:`RADIANT_FUNCTIONS` — the canonical UI menu.
RADIANT_FUNCTION_KEYS: list[str] = [key for key, _ in RADIANT_FUNCTIONS]


def resolve_agg(name: str):
    """Return the expression builder for ``name`` or raise a clear error."""
    try:
        return AGG_FUNCTIONS[name]
    except KeyError:
        raise ValueError(
            f"Unknown aggregation function: {name}\n"
            f"Supported: {', '.join(sorted(AGG_FUNCTIONS))}"
        ) from None
