"""NaN is missing (as R's NA), and a summary says how many rows it dropped.

Every analysis goes through ``apply_scope``, which turns NaN into null; the
tools that drop incomplete rows (listwise, as radiant does with
``get_data(na.rm = TRUE)``) print ``Dropped : x of y rows (missing values)``
under the number of observations.
"""

import io
from contextlib import redirect_stdout

import numpy as np
import polars as pl
import pytest

import pyrsm as rsm
from pyrsm.data_scope import apply_scope, dropped_lines, nan_to_null


def _printed(obj, **kwargs) -> str:
    out = io.StringIO()
    with redirect_stdout(out):
        obj.summary(**kwargs)
    return out.getvalue()


@pytest.fixture
def survey():
    """60 rows, 4 correlated numeric variables; row 4 has a NaN, row 8 a null."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 4))
    X[:, 1] += X[:, 0]
    X[:, 3] += X[:, 2]
    a, c = X[:, 0].copy(), X[:, 2].tolist()
    a[3] = np.nan
    c[7] = None
    return pl.DataFrame({"a": a, "b": X[:, 1], "c": c, "d": X[:, 3]})


def test_nan_to_null_only_touches_floats():
    df = pl.DataFrame({"x": [1.0, float("nan"), None], "k": [1, 2, 3], "s": ["a", "b", "c"]})
    out = nan_to_null(df)
    assert out["x"].null_count() == 2
    assert out["k"].to_list() == [1, 2, 3] and out["s"].to_list() == ["a", "b", "c"]
    assert nan_to_null(df.select("k", "s")) is not None


def test_a_filter_drops_nan_as_r_drops_na():
    # polars: NaN > 0 is True; R: NA > 0 is NA, and filter() drops the row
    df = pl.DataFrame({"x": [1.0, float("nan"), -1.0]})
    assert apply_scope(df, "x > 0")["x"].to_list() == [1.0]


@pytest.mark.parametrize(
    "build, width",
    [
        (lambda d: rsm.multivariate.pre_factor(d, vars=list("abcd")), 12),
        (lambda d: rsm.multivariate.full_factor(d, vars=list("abcd"), nr_fact=2), 12),
        (lambda d: rsm.multivariate.kclus(d, vars=list("abcd"), nr_clus=2), 13),
        (lambda d: rsm.multivariate.hclus(d, vars=list("abcd")), 12),
        (lambda d: rsm.basics.correlation(d, vars=list("abcd")), 9),
    ],
    ids=["pre_factor", "full_factor", "kclus", "hclus", "correlation"],
)
def test_nan_and_null_rows_are_dropped_and_reported(survey, build, width):
    obj = build(survey)
    assert obj.nobs == 58 and obj.nobs_dropped == 2
    text = _printed(obj)
    assert f"{'Dropped'.ljust(width)}: 2 of 60 rows (missing values)" in text

    complete = build(survey.drop_nulls().filter(~pl.col("a").is_nan()))
    assert complete.nobs_dropped == 0
    assert "Dropped" not in _printed(complete)


def test_prmap_drops_brands_with_missing_attributes_or_preferences(survey):
    brands = survey.with_columns(pl.Series("brand", [f"B{i % 6}" for i in range(60)]))
    pm = brands.group_by("brand").agg(pl.col(list("abcd")).mean()).sort("brand")
    pm = pm.with_columns(pl.Series("pref", [1.0, 2.0, float("nan"), 3.0, 4.0, 5.0]))
    # B3 has the NaN in a (mean is NaN), B2 a NaN preference
    m = rsm.multivariate.prmap(pm, brand="brand", attr=list("abcd"), pref="pref", nr_dim=2)
    assert m.nobs == 4 and m.nobs_dropped == 2
    assert "Dropped     : 2 of 6 rows (missing values)" in _printed(m)


def test_mds_drops_pairs_without_a_distance():
    pairs = pl.DataFrame(
        {
            "from": ["A", "A", "A", "B", "B", "C", "C"],
            "to": ["B", "C", "D", "C", "D", "D", "A"],
            "dist": [3.0, 4.0, 5.0, 2.0, 4.0, 3.0, float("nan")],
        }
    )
    m = rsm.multivariate.mds(pairs, "from", "to", "dist")
    assert m.nobs == 6 and m.nobs_dropped == 1
    assert "Dropped     : 1 of 7 rows (missing values)" in _printed(m)


def test_dropped_lines_is_empty_without_dropped_rows():
    class Obj:
        nobs, nobs_dropped = 10, 0

    assert dropped_lines(Obj()) == []
    Obj.nobs_dropped = 1_200
    assert dropped_lines(Obj(), 8) == ["Dropped : 1,200 of 1,210 rows (missing values)"]
