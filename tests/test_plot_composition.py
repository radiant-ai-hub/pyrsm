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


def test_bar_plots_show_confidence_and_standard_error_bars():
    """radiant.basics: compare means/props bars carry a CI (black) and an SE (blue) bar."""
    import pyrsm as rsm
    from plotnine import geom_errorbar

    salary = pl.read_parquet("examples/data/basics/salary.parquet")
    titanic = pl.read_parquet("examples/data/model/titanic.parquet")
    cm = rsm.basics.compare_means({"salary": salary}, var1="rank", var2="salary")
    cp = rsm.basics.compare_props({"titanic": titanic}, var1="pclass", var2="survived", lev="Yes")
    for p in (cm.plot(plots="bar"), cp.plot(plots="bar")):
        bars = [layer for layer in p.layers if isinstance(layer.geom, geom_errorbar)]
        assert len(bars) == 2
        assert bars[1].geom.aes_params.get("color") == "blue"
        p.draw()


def test_cross_tabs_plot_keeps_the_level_order():
    """Axis and legend follow the Enum levels, as the tables do (WS Journal before USA Today)."""
    newspaper = pl.read_parquet("examples/data/basics/newspaper.parquet")
    ct = cross_tabs({"newspaper": newspaper}, var1="Income", var2="Newspaper")
    for kind in ("dev_std", "observed", "perc_col"):
        p = ct.plot(plots=kind)
        assert p.data["Newspaper"].dtype.categories.to_list() == ["WS Journal", "USA Today"]
        assert p.data["Income"].dtype.categories.to_list() == ["Low Income", "High Income"]
        p.draw()
