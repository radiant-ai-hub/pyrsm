import polars as pl

from pyrsm.basics.cross_tabs import cross_tabs
from pyrsm.basics.goodness import goodness
from pyrsm.plot_utils import compose_plots


def test_compose_plots_returns_none_on_empty():
    assert compose_plots([]) is None


def test_cross_tabs_plot_composes_multiple():
    # Build a tiny cross-tab object
    data = pl.DataFrame(
        {
            "Income": ["Low", "Low", "High", "High"],
            "Newspaper": ["A", "B", "A", "B"],
        }
    )
    ct = cross_tabs(data, "Income", "Newspaper")
    composed = ct.plot(plots=["observed", "expected"])
    assert composed is not None


def test_goodness_plot_composes_multiple():
    # Create raw data that goodness will count
    data = pl.DataFrame({"category": ["A"] * 10 + ["B"] * 20 + ["C"] * 30})
    gd = goodness(data, var="category", probs=[0.2, 0.3, 0.5])
    composed = gd.plot(plots=["observed", "expected", "chisq"])
    assert composed is not None
