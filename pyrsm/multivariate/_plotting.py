"""Shared plotting helpers for the ``pyrsm.multivariate`` maps and factor plots.

Radiant draws brand/factor maps on square, origin-centered axes (so the
horizontal and vertical scales are comparable and the origin sits in the
middle). These helpers compute the symmetric limits and a fixed-aspect square
panel so the Python plots match.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "symmetric_limit",
    "square_origin_limits",
    "add_square_axes",
    "PlotList",
    "as_plot_result",
]


class PlotList(list):
    """A list of plots that renders each one inline in a Jupyter notebook.

    Plot methods that produce more than one figure (e.g. ``scree`` + ``change``,
    or one panel per dimension pair) return this. It behaves exactly like a
    ``list`` for indexing, iteration and ``len`` — so existing code and tests are
    unaffected — but when it is the result of a notebook cell each plot is
    displayed instead of the list's ``[<ggplot ...>, ...]`` text repr (plotnine
    only auto-renders a *single* ggplot result).
    """

    def _ipython_display_(self):
        from IPython.display import display

        for g in self:
            display(g)


def as_plot_result(out):
    """Return a single plot when there is one, else a render-friendly PlotList."""
    out = list(out)
    if not out:
        return None
    return out[0] if len(out) == 1 else PlotList(out)


def symmetric_limit(*arrays, pad: float = 1.05) -> float:
    """Return ``pad * max(|coordinate|)`` across all supplied arrays.

    Matches Radiant's ``lim <- max(abs(...))`` (with a small padding so points
    and labels are not clipped at the panel edge).
    """
    vals = []
    for a in arrays:
        a = np.asarray(a, dtype=float)
        if a.size:
            vals.append(np.nanmax(np.abs(a)))
    if not vals:
        return 1.0
    lim = max(vals) * pad
    return float(lim if lim > 0 else 1.0)


def square_origin_limits(*arrays, pad: float = 1.05) -> tuple[float, float]:
    """Return ``(-lim, lim)`` symmetric limits for both axes."""
    lim = symmetric_limit(*arrays, pad=pad)
    return (-lim, lim)


def add_square_axes(g, lim: float, zero_lines: bool = True):
    """Add square origin-centered axes (and zero lines) to a plotnine plot."""
    from plotnine import coord_fixed, geom_hline, geom_vline

    if zero_lines:
        g = (
            g
            + geom_vline(xintercept=0, color="grey", size=0.3)
            + geom_hline(yintercept=0, color="grey", size=0.3)
        )
    g = g + coord_fixed(ratio=1, xlim=(-lim, lim), ylim=(-lim, lim))
    return g
