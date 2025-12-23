"""Shared helpers for plotnine composition and layout."""

from math import ceil

from plotnine import theme


def compose_plots(plot_list: list, ncol: int = 2):
    """
    Compose multiple plotnine plots into a grid layout.

    Parameters
    ----------
    plot_list : list
        List of plotnine plots.
    ncol : int, default 2
        Number of columns in the grid.

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
        row_plots = plot_list[start:end]

        row = row_plots[0]
        for p in row_plots[1:]:
            row = row | p
        rows.append(row)

    # Stack rows vertically
    combined = rows[0]
    for row in rows[1:]:
        combined = combined / row

    # Auto-adjust figure size
    height_per_row = 3
    width_per_col = 4
    fig_width = width_per_col * min(ncol, len(plot_list))
    fig_height = height_per_row * nrow
    combined = combined + theme(figure_size=(fig_width, fig_height))

    return combined
