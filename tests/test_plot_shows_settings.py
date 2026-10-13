"""Plots say what was calculated: the distribution's parameters and the bounds used."""

import pytest
from plotnine.geoms import geom_text

import pyrsm as rsm


def _texts(p):
    return [layer.geom.aes_params.get("label") for layer in p.layers if isinstance(layer.geom, geom_text)]


@pytest.mark.parametrize(
    "dist, kwargs, title, labels",
    [
        ("norm", {"mean": 100, "sd": 8, "lb": 90, "ub": 110}, "Normal Distribution (mean: 100, sd: 8)", ["lb = 90 ", " ub = 110"]),
        ("tdist", {"df": 101.286, "plb": 0.05}, "t Distribution (df: 101.286)", [" plb = 0.05 (x = -1.66)"]),
        ("binom", {"n": 1000, "p": 0.1, "pub": 0.05}, "Binomial Distribution (n: 1000, p: 0.1)", [" pub = 0.05 (x = 85)"]),
        ("chisq", {"df": 1, "pub": 0.95}, "Chi-square Distribution (df: 1)", [" pub = 0.95 (x = 3.841)"]),
        ("disc", {"v": [1, 2, 3, 4], "p": [0.4, 0.3, 0.2, 0.1], "lb": 2}, "Discrete Distribution\n(values: 1 2 3 4, probabilities: 0.4 0.3 0.2 0.1)", [" lb = 2"]),
        ("norm", {"mean": 0, "sd": 1}, "Normal Distribution (mean: 0, sd: 1)", []),
    ],
)
def test_prob_calc_plot_shows_parameters_and_bounds(dist, kwargs, title, labels):
    p = rsm.basics.prob_calc(dist, **kwargs).plot()
    assert p.labels.title == title
    assert p.labels.get("subtitle", None) is None
    assert p.theme.themeables["plot_title"].properties["ha"] == "center"
    assert _texts(p) == labels
    p.draw()


@pytest.mark.parametrize(
    "dist, kwargs",
    [
        ("norm", {"mean": 0, "sd": 1, "plb": 0.025, "pub": 0.975}),
        ("norm", {"mean": 100, "sd": 8, "lb": 90, "ub": 110}),
        ("binom", {"n": 10, "p": 0.3, "lb": 1, "ub": 3}),
        ("tdist", {"df": 10, "lb": -0.5, "ub": 0.5}),
    ],
)
def test_bound_labels_sit_at_one_height(dist, kwargs):
    p = rsm.basics.prob_calc(dist, **kwargs).plot()
    ys = {float(layer.geom.data["y"].iloc[0]) for layer in p.layers if isinstance(layer.geom, geom_text)}
    assert len(ys) == 1
    p.draw()


def test_a_bound_label_stays_visible_where_the_density_is_infinite():
    """chi-squared with df = 1 is infinite at 0: the label goes at the top of what is drawn."""
    p = rsm.basics.prob_calc("chisq", df=1, ub=3.841).plot()
    (layer,) = [layer for layer in p.layers if isinstance(layer.geom, geom_text)]
    y = float(layer.geom.data["y"].iloc[0])
    assert 0 < y < 10  # above the drawn curve (about 4), not at infinity
    p.draw()


def test_clt_plot_subtitles_carry_the_settings():
    c = rsm.basics.clt("Exponential", n=30, m=1000, expo_rate=1, seed=1234)
    comp = c.plot(stat="mean")
    subtitles = [plot.labels.subtitle for row in comp.items for plot in row.items]
    assert subtitles == ["rate = 1", "rate = 1"] + ["sample size n = 30, # of samples m = 1,000"] * 2
