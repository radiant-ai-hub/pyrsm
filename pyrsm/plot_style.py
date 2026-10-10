"""The one pyrsm plot style.

Every pyrsm plot is a plotnine ``ggplot`` (or a composition / ``PlotList`` of
them) and takes its look from here, so a scree plot, a coefficient plot and a
histogram read as one system. Plot code imports these names instead of picking
its own theme or colours:

    from pyrsm.plot_style import FILL, LINE, REF, theme_pyrsm

    ggplot(...) + geom_col(fill=FILL) + geom_hline(yintercept=1, color=REF) + theme_pyrsm()

Changing the house style is a change to this file only.
"""

from __future__ import annotations

__all__ = [
    "FILL",
    "LINE",
    "POINT_FILL",
    "REF",
    "COMP",
    "MUTED",
    "FILL_ALT",
    "HIGHLIGHT",
    "ALPHA",
    "theme_pyrsm",
]

#: Primary colour for bars, histograms, areas and single-series marks.
FILL = "slateblue"
#: Primary colour for lines and point outlines of a single series.
LINE = "slateblue"
#: Fill for hollow points drawn on a LINE-coloured outline.
POINT_FILL = "white"
#: Reference lines and statistics that are part of the reading (eigenvalue = 1,
#: y = 0, the diagonal, a sample mean or a fitted line drawn over the data).
REF = "black"
#: A comparison or null value the student tests against.
COMP = "red"
#: De-emphasised marks (individual curves behind an average, grid of references).
MUTED = "gray"
#: Second fill when a plot contrasts two regions of one series, e.g. the
#: probability mass outside a range (FILL is the mass inside it).
FILL_ALT = "salmon"
#: The one mark singled out in a series, e.g. the bar at a probability bound.
#: At ALPHA on a white panel it renders as ~"#84f37f", the bound green of
#: radiant.basics ("green" at alpha 0.5 on the gray ggplot panel).
HIGHLIGHT = "#09E700"
#: Transparency for filled areas that may overlap.
ALPHA = 0.5


def theme_pyrsm():
    """The pyrsm plot theme: white panel, light grid, black frame (plotnine ``theme_bw``)."""
    from plotnine import theme_bw

    return theme_bw()
