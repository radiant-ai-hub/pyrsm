"""A loess fit line that draws for any data a scatter plot can show.

plotnine's ``geom_smooth(method="loess")`` fits a local quadratic to the
``span`` share of the points nearest each x (scikit-misc). When x has only a
few distinct values -- household size 3 to 9, a rating 1 to 5 -- the points
nearest x = 3 may hold only one or two of them, the local fit is singular and
the whole plot fails ("There are other near singularities as well"). R's
loess draws the line anyway (with a "pseudoinverse used" warning). Missing
values make this common: 167 complete rows of 1,000 leave few points at each
level.

``loess`` is a plotnine smoothing method (``geom_smooth(method=loess)``) that
widens the span, then fits local lines, then a straight line, until a fit
succeeds, and warns which one it drew.
"""

from __future__ import annotations

import warnings

import pandas as pd

__all__ = ["loess", "smooth_method"]

# (span, degree) tried after the requested span; scikit-misc allows span <= 1
_FALLBACKS = ((0.9, 2), (1.0, 2), (1.0, 1))


def loess(data: pd.DataFrame, xseq, params: dict) -> pd.DataFrame:
    """plotnine smoothing method: loess, falling back to a wider span, local
    lines, then a linear fit when x has too few distinct values."""
    from plotnine.stats.smoothers import lm
    from plotnine.stats.smoothers import loess as _loess

    span = params.get("span", 0.75)
    args = dict(params.get("method_args") or {})
    tries = [(span, args.get("degree", 2))] + [(s, d) for s, d in _FALLBACKS if s >= span]
    for i, (s, degree) in enumerate(tries):
        try:
            fit = _loess(data, xseq, {**params, "span": s, "method_args": {**args, "degree": degree}})
        except ValueError:
            continue
        if i:
            what = f"span {s}" + (" and local lines" if degree == 1 else "")
            warnings.warn(
                f"loess: x has too few distinct values for span {span}; the line uses {what}",
                stacklevel=2,
            )
        return fit
    warnings.warn(
        "loess: x has too few distinct values for a local fit; the line is a linear fit",
        stacklevel=2,
    )
    return lm(data, xseq, {**params, "method_args": {}, "formula": None})


def smooth_method(method: str):
    """The plotnine smoothing method for ``method`` ("loess" -> ``loess``)."""
    return loess if method == "loess" else method
