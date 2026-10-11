"""Plots say what was calculated: the distribution's parameters and the bounds used."""

import pytest
from plotnine.geoms import geom_text

import pyrsm as rsm


def _texts(p):
    return [layer.geom.aes_params.get("label") for layer in p.layers if isinstance(layer.geom, geom_text)]


@pytest.mark.parametrize(
    "dist, kwargs, subtitle, labels",
    [
        ("norm", {"mean": 100, "sd": 8, "lb": 90, "ub": 110}, "mean = 100, sd = 8", [" lb = 90", "ub = 110 "]),
        ("tdist", {"df": 101.286, "plb": 0.05}, "df = 101.286", [" plb = 0.05 (x = -1.66)"]),
        ("binom", {"n": 1000, "p": 0.1, "pub": 0.05}, "n = 1000, p = 0.1", [" pub = 0.05 (x = 85)"]),
        ("chisq", {"df": 1, "pub": 0.95}, "df = 1", [" pub = 0.95 (x = 3.841)"]),
        ("disc", {"v": [1, 2, 3, 4], "p": [0.4, 0.3, 0.2, 0.1], "lb": 2}, "values = 1 2 3 4, probabilities = 0.4 0.3 0.2 0.1", [" lb = 2"]),
        ("norm", {"mean": 0, "sd": 1}, "mean = 0, sd = 1", []),
    ],
)
def test_prob_calc_plot_shows_parameters_and_bounds(dist, kwargs, subtitle, labels):
    p = rsm.basics.prob_calc(dist, **kwargs).plot()
    assert p.labels.subtitle == subtitle
    assert _texts(p) == labels
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
