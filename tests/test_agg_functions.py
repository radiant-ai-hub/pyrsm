"""The shared aggregation registry used by ``explore`` and ``pivot``.

Regression guard for "Unknown aggregation function: n_obs": the two tools
used to keep separate hand-written lists, so a metric offered by one (and
by the Radiant UI) could be missing from the other.
"""

import math

import polars as pl
import pytest

from pyrsm.eda.agg_functions import (
    AGG_FUNCTIONS,
    RADIANT_FUNCTION_KEYS,
    RADIANT_FUNCTIONS,
    resolve_agg,
)
from pyrsm.eda.explore import EXPLORE_FUNCTIONS, explore
from pyrsm.eda.pivot import pivot

# The exact set R's radiant.data offers, from
# ``options(radiant.functions = ...)`` in inst/app/global.R.
RADIANT_DATA_FUNCTIONS = [
    "n_obs", "n_missing", "n_distinct",
    "mean", "median", "modal", "min", "max",
    "sum", "var", "sd", "se", "me", "cv",
    "prop", "varprop", "sdprop", "seprop", "meprop", "varpop", "sdpop",
    "p01", "p025", "p05", "p10", "p25", "p75", "p90", "p95", "p975", "p99",
    "skew", "kurtosi", "IQR",
]


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "g": ["a", "a", "b", "b", "a", "b"],
            "binary": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "value": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )


# ---- parity with radiant.data ---------------------------------------


def test_registry_covers_every_radiant_data_function():
    missing = [f for f in RADIANT_DATA_FUNCTIONS if f not in AGG_FUNCTIONS]
    assert not missing, f"missing radiant.data metrics: {missing}"


def test_radiant_menu_matches_radiant_data():
    """The offered menu is exactly radiant.data's, in radiant's order."""
    assert RADIANT_FUNCTION_KEYS == RADIANT_DATA_FUNCTIONS


def test_radiant_menu_entries_all_resolve():
    for key, label in RADIANT_FUNCTIONS:
        assert resolve_agg(key) is not None, key
        assert label, f"{key} has no display label"


def test_explore_and_pivot_share_one_registry():
    """A metric can never exist in one tool but not the other."""
    assert EXPLORE_FUNCTIONS is AGG_FUNCTIONS


# ---- every metric actually evaluates ---------------------------------


@pytest.mark.parametrize("fn", RADIANT_DATA_FUNCTIONS)
def test_explore_accepts_every_radiant_function(fn, df):
    result = explore(df, cols=["value"], agg=[fn])
    assert fn in result.columns
    assert result.height == 1


@pytest.mark.parametrize("fn", RADIANT_DATA_FUNCTIONS)
def test_explore_accepts_every_radiant_function_grouped(fn, df):
    result = explore(df, cols=["value"], agg=[fn], by="g")
    assert result.height == 2


@pytest.mark.parametrize("fn", RADIANT_DATA_FUNCTIONS)
def test_pivot_accepts_every_radiant_function(fn, df):
    result = pivot(df, rows="g", values="value", agg=fn)
    assert f"value_{fn}" in result.columns


def test_unknown_function_raises_with_the_supported_set(df):
    with pytest.raises(ValueError, match="Unknown aggregation function: nope"):
        explore(df, cols=["value"], agg=["nope"])
    # The message must list what IS available, so the fix is obvious.
    with pytest.raises(ValueError, match="n_obs"):
        explore(df, cols=["value"], agg=["nope"])


def test_radiant_default_functions_work(df):
    """radiant.data's default_funs — the exact combination that errored."""
    result = explore(df, cols=["value"], agg=["n_obs", "mean", "sd", "min", "max"])
    assert set(["n_obs", "mean", "sd", "min", "max"]) <= set(result.columns)


# ---- values match radiant.data's definitions -------------------------


def test_n_obs_counts_rows_including_missing():
    """radiant's ``n_obs`` is ``length(x)``, so nulls count."""
    df = pl.DataFrame({"x": [1.0, None, 3.0]})
    assert explore(df, cols=["x"], agg=["n_obs"])["n_obs"][0] == 3
    assert explore(df, cols=["x"], agg=["n_missing"])["n_missing"][0] == 1


def test_prop_is_share_of_the_maximum_value():
    """radiant: ``mean(x == max(x))`` — not the column mean."""
    df = pl.DataFrame({"x": [4.0, 4.0, 2.0, 2.0]})
    assert explore(df, cols=["x"], agg=["prop"])["prop"][0] == pytest.approx(0.5)

    binary = pl.DataFrame({"x": [1.0, 1.0, 1.0, 0.0]})
    assert explore(binary, cols=["x"], agg=["prop"])["prop"][0] == pytest.approx(0.75)


def test_prop_family_matches_radiant_formulas():
    df = pl.DataFrame({"x": [1.0, 1.0, 1.0, 0.0]})
    p, n = 0.75, 4
    got = explore(
        df, cols=["x"], agg=["prop", "varprop", "sdprop", "seprop", "meprop"]
    )
    assert got["varprop"][0] == pytest.approx(p * (1 - p))
    assert got["sdprop"][0] == pytest.approx(math.sqrt(p * (1 - p)))
    assert got["seprop"][0] == pytest.approx(math.sqrt(p * (1 - p) / n))
    # meprop = seprop * qnorm(0.975)
    assert got["meprop"][0] == pytest.approx(
        math.sqrt(p * (1 - p) / n) * 1.959963984540054
    )


def test_varpop_is_the_population_variance():
    """radiant: ``var(x) * (n - 1) / n``."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    got = explore(df, cols=["x"], agg=["var", "varpop", "sdpop"])
    sample_var, n = got["var"][0], 4
    assert got["varpop"][0] == pytest.approx(sample_var * (n - 1) / n)
    assert got["sdpop"][0] == pytest.approx(math.sqrt(sample_var * (n - 1) / n))


def test_me_uses_the_t_distribution():
    """radiant: ``se * qt(0.975, n - 1)`` — not a flat 1.96."""
    from scipy import stats

    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    got = explore(df, cols=["x"], agg=["se", "me"])
    n = 5
    assert got["me"][0] == pytest.approx(got["se"][0] * stats.t.ppf(0.975, n - 1))


def test_percentiles_include_radiants_two_sided_pairs():
    df = pl.DataFrame({"x": [float(i) for i in range(1, 101)]})
    got = explore(df, cols=["x"], agg=["p025", "p25", "p75", "p975"])
    assert got["p025"][0] < got["p25"][0] < got["p75"][0] < got["p975"][0]


def test_modal_returns_the_most_frequent_value():
    df = pl.DataFrame({"x": [5.0, 5.0, 5.0, 9.0]})
    assert explore(df, cols=["x"], agg=["modal"])["modal"][0] == 5.0


def test_aliases_agree_with_their_canonical_name(df):
    for alias, canonical in (
        ("sd", "std"),
        ("kurtosi", "kurtosis"),
        ("n_distinct", "n_unique"),
        ("IQR", "iqr"),
        ("n_obs", "count"),
    ):
        a = explore(df, cols=["value"], agg=[alias])[alias][0]
        b = explore(df, cols=["value"], agg=[canonical])[canonical][0]
        assert a == pytest.approx(b), f"{alias} != {canonical}"
