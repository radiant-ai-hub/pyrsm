"""Loess lines and plots on data with missing values.

Data like this (most responses missing, an explanatory variable with a handful
of values) failed three ways: loess raised "There are other near
singularities as well", the regression residual plot raised "The number of
observations in the fitted model and the data frame must be the same", and a
visualize scatter sampled nobs rows before dropping the ones it cannot draw.
"""

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pyrsm.eda.visualize import visualize
from pyrsm.model.regress import regress
from pyrsm.plot_smooth import loess
from pyrsm.utils import complete_rows


@pytest.fixture(scope="module")
def sparse():
    """1,000 households; spending recorded for about 1 in 6."""
    rng = np.random.default_rng(7)
    n = 1000
    # most households have 3 or 4 members: loess at x = 3 sees only 3 and 4
    size = rng.choice(np.arange(3, 10), n, p=[0.3, 0.4, 0.15, 0.07, 0.04, 0.03, 0.01])
    age = rng.integers(18, 66, n)
    spend = 2 + 0.4 * size**2 + 0.1 * age + rng.normal(0, 3, n)
    spend = np.where(rng.random(n) < 1 / 6, spend, np.nan)
    return pl.DataFrame({"size": size, "age": age, "spend": spend}).with_columns(
        pl.col("spend").fill_nan(None)
    )


def _draw(p):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p.draw()
    return [str(w.message) for w in caught]


PARAMS = {"span": 0.75, "method_args": {}, "se": True, "level": 0.95, "formula": None}


def test_loess_widens_the_span_for_few_distinct_values():
    x = np.repeat([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [40, 60, 30, 15, 10, 8, 4])
    data = pd.DataFrame({"x": x, "y": 0.5 * x**2})
    with pytest.warns(UserWarning, match="the line uses span 0.9"):
        fit = loess(data, np.array([3.0, 6.0, 9.0]), dict(PARAMS))
    assert fit["y"].round(1).tolist() == [4.5, 18.0, 40.5]


def test_loess_falls_back_to_a_line():
    data = pd.DataFrame({"x": [1.0] * 10 + [2.0] * 10, "y": np.arange(20.0)})
    with pytest.warns(UserWarning, match="the line is a linear fit"):
        fit = loess(data, np.array([1.0, 2.0]), dict(PARAMS))
    assert fit["y"].round(1).tolist() == [4.5, 14.5]


def test_loess_unchanged_when_it_fits():
    x = np.linspace(0, 10, 100)
    data = pd.DataFrame({"x": x, "y": np.sin(x)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loess(data, x[:5], dict(PARAMS))


@pytest.mark.parametrize("plot", ["scatter", "residual", "dashboard"])
@pytest.mark.parametrize("lines", [["loess"], ["line", "loess"], []])
def test_regression_plots_with_missing_values(sparse, plot, lines):
    reg = regress({"sparse": sparse}, rvar="spend", evar=["size", "age"])
    caught = _draw(reg.plot(plot, lines=lines))
    # loess on size fails at span 0.75 (dashboard fits on fitted values)
    fallback = any("too few distinct values" in w for w in caught)
    assert fallback == ("loess" in lines and plot != "dashboard")


def test_regression_scatter_shows_the_estimation_rows(sparse):
    reg = regress({"sparse": sparse}, rvar="spend", evar=["size", "age"])
    p = reg.plot("scatter", nobs=-1)
    first = p[0] if hasattr(p, "__getitem__") else p.items[0]
    assert len(first.data) == sparse["spend"].is_not_null().sum() == reg.fitted.nobs


def test_visualize_samples_points_it_can_draw(sparse):
    p = visualize(sparse, x="size", y="spend", nobs=100, smooth="loess")
    assert len(p.data) == 100
    assert not p.data["spend"].is_null().any()
    _draw(p)


def test_complete_rows_counts_nan_as_missing():
    df = pl.DataFrame({"a": [1.0, float("nan"), None, 4.0], "b": [1, 2, 3, None]})
    assert df.filter(complete_rows(df, ["a", "b"])).height == 1
    assert df.filter(complete_rows(df, ["a"])).height == 2
