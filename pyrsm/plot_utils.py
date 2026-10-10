"""Shared helpers for plotnine composition and layout."""

from math import ceil

from plotnine import theme
from plotnine.composition import plot_spacer


def compose_plots(
    plot_list: list,
    ncol: int = 2,
    width_per_col: float = 4,
    height_per_row: float = 3,
):
    """
    Compose multiple plotnine plots into a grid layout.

    An incomplete last row is padded with blank space so every plot keeps
    the same cell size, whatever the number of plots.

    Parameters
    ----------
    plot_list : list
        List of plotnine plots.
    ncol : int, default 2
        Number of columns in the grid.
    width_per_col, height_per_row : float
        Size of one grid cell in inches.

    Returns
    -------
    plotnine object or None
        Composed plot or None if no plots provided.
    """
    if not plot_list:
        return None
    if len(plot_list) == 1:
        return plot_list[0]

    nrow = ceil(len(plot_list) / ncol)

    # Build rows of plots
    rows = []
    for i in range(nrow):
        start = i * ncol
        end = min(start + ncol, len(plot_list))
        row_plots = list(plot_list[start:end])
        if nrow > 1 and len(row_plots) < ncol:
            row_plots += [plot_spacer() for _ in range(ncol - len(row_plots))]

        row = row_plots[0]
        for p in row_plots[1:]:
            row = row | p
        rows.append(row)

    # Stack rows vertically
    combined = rows[0]
    for row in rows[1:]:
        combined = combined / row

    # Auto-adjust figure size
    fig_width = width_per_col * min(ncol, len(plot_list))
    fig_height = height_per_row * nrow
    combined = combined + theme(figure_size=(fig_width, fig_height))

    return combined


def plot_each(plot_one, plots, *, ncol: int = 1, width: float = 7, height: float = 4.5):
    """
    Plot several plot types as one figure.

    ``plot_one(plot_type)`` draws a single plot. ``plots`` is a plot type or a
    list of them; one type returns its plot unchanged, several are composed
    with :func:`compose_plots` (one column by default, as radiant.basics stacks
    its plots), so ``obj.plot(plots=["hist", "sim"])`` shows both in a notebook
    or report chunk where only the last expression is displayed.

    Returns ``None`` when ``plots`` is a single type (the caller then draws it
    itself) so a class can use this as a guard at the top of ``plot``.
    """
    if isinstance(plots, str):
        return None
    plots = list(plots)
    if not plots:
        raise ValueError("Select at least one plot type.")
    if len(plots) == 1:
        return plot_one(plots[0])
    return compose_plots(
        [plot_one(p) for p in plots], ncol=ncol, width_per_col=width, height_per_row=height
    )
