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


def test_list_of_plots_compose_for_every_basics_test():
    import polars as pl
    from plotnine.composition import Compose

    from pyrsm.basics.compare_means import compare_means
    from pyrsm.basics.compare_props import compare_props
    from pyrsm.basics.single_prop import single_prop

    data = pl.DataFrame({"g": ["a", "b"] * 20, "y": [float(i % 7) for i in range(40)],
                         "z": ["yes", "no", "no", "yes"] * 10})
    assert isinstance(single_prop({"d": data}, var="z", lev="yes").plot(["bar", "sim"]), Compose)
    assert isinstance(compare_props({"d": data}, var1="g", var2="z", lev="yes").plot(["bar", "dodge"]), Compose)
    assert isinstance(compare_means({"d": data}, var1="g", var2="y").plot(["scatter", "box"]), Compose)
