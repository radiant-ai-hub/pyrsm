"""Radiant (R) parity for the data scope: filter, sort and slice.

The R radiant packages thread ``data_filter`` / ``arr`` / ``rows`` through
``get_data`` before an analysis sees the data; pyrsm does the same with
``data_filter`` / ``sort`` / ``slice``. These tests port the R tests that
filter, sort or slice (radiant.data ``test_funs.R``) and add the cases R does
not test directly but that matter: scoped data passed into analyses.

Reference numbers were computed with the installed R packages
(radiant.data, radiant.basics 1.6.6, radiant.model 1.6.12,
radiant.multivariate). The R call that produced each number is quoted next to
it. R analyses without ``arr``/``rows`` arguments were run on
``get_data(dataset, filt = ..., arr = ..., rows = ...)``, which is exactly what
the R UI does before calling them.

Notation used in the R snippets:
``d  <- diamonds;  d$rid  <- seq_len(nrow(d))``
``ti <- titanic;   ti$rid <- seq_len(nrow(ti))``
``rid`` is a row id used as a sort tie-breaker, so that a sort + slice picks a
well-defined set of rows (R's ``arrange`` is stable, so ``desc(x), rid`` is
the same as ``desc(x)`` in R).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import pyrsm as rsm
from pyrsm.model.logistic import logistic
from pyrsm.model.mlp import mlp
from pyrsm.model.regress import regress
from pyrsm.data_scope import apply_scope, get_data, sort_code

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.fixture(scope="module")
def diamonds() -> pl.DataFrame:
    return pl.read_parquet(DATA / "data" / "diamonds.parquet").with_row_index(
        "rid", offset=1
    )


@pytest.fixture(scope="module")
def titanic() -> pl.DataFrame:
    return pl.read_parquet(DATA / "data" / "titanic.parquet").with_row_index(
        "rid", offset=1
    )


@pytest.fixture(scope="module")
def shopping() -> pl.DataFrame:
    return pl.read_parquet(DATA / "multivariate" / "shopping.parquet")


def _manual(df, data_filter, sort_cols, descending, n):
    """The R semantics written out with plain polars: filter, *stable* sort, head."""
    out = df.filter(data_filter) if data_filter is not None else df
    if sort_cols:
        out = out.sort(sort_cols, descending=descending, maintain_order=True)
    return out.head(n) if n is not None else out


# ===========================================================================
# radiant.data: get_data / filter_data  (test_funs.R)
# ===========================================================================

# mtcars[, c("mpg", "cyl", "disp")] -- inlined, it is not in examples/data
_MTCARS = pl.DataFrame(
    {
        "name": [
            "Mazda RX4", "Mazda RX4 Wag", "Datsun 710", "Hornet 4 Drive",
            "Hornet Sportabout", "Valiant", "Duster 360", "Merc 240D",
            "Merc 230", "Merc 280", "Merc 280C", "Merc 450SE", "Merc 450SL",
            "Merc 450SLC", "Cadillac Fleetwood", "Lincoln Continental",
            "Chrysler Imperial", "Fiat 128", "Honda Civic", "Toyota Corolla",
            "Toyota Corona", "Dodge Challenger", "AMC Javelin", "Camaro Z28",
            "Pontiac Firebird", "Fiat X1-9", "Porsche 914-2", "Lotus Europa",
            "Ford Pantera L", "Ferrari Dino", "Maserati Bora", "Volvo 142E",
        ],
        "mpg": [
            21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8,
            16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5,
            15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4,
        ],
        "cyl": [
            6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 6, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 8,
            8, 8, 8, 4, 4, 4, 8, 6, 8, 4,
        ],
        "disp": [
            160.0, 160.0, 108.0, 258.0, 360.0, 225.0, 360.0, 146.7, 140.8,
            167.6, 167.6, 275.8, 275.8, 275.8, 472.0, 460.0, 440.0, 78.7,
            75.7, 71.1, 120.1, 318.0, 304.0, 350.0, 400.0, 79.0, 120.3, 95.1,
            351.0, 145.0, 301.0, 121.0,
        ],
    }
)


def test_get_data_filter_and_rows_mtcars():
    """R: get_data(mtcars, "mpg:disp", filt = "mpg > 20", rows = 1:5)
    == mtcars[mtcars$mpg > 20, c("mpg", "cyl", "disp")][1:5, ]"""
    name, out = get_data(
        {"mtcars": _MTCARS}, vars=["name", "mpg", "cyl", "disp"],
        data_filter="mpg > 20", slice="1:5",
    )
    assert name == "mtcars"
    assert out["name"].to_list() == [
        "Mazda RX4", "Mazda RX4 Wag", "Datsun 710", "Hornet 4 Drive", "Merc 240D"
    ]
    assert out["mpg"].to_list() == [21.0, 21.0, 22.8, 21.4, 24.4]
    assert out["disp"].to_list() == [160.0, 160.0, 108.0, 258.0, 146.7]
    assert out.equals(_MTCARS.filter(pl.col("mpg") > 20).head(5))


def test_filter_single_level(diamonds):
    """R: nrow(get_data(diamonds, filt = "cut == 'Very Good'")) == 677"""
    assert apply_scope(diamonds, "cut == 'Very Good'").height == 677


def test_filter_data_and(diamonds):
    """R: filter_data(diamonds, "cut == 'Very Good' & price > 5000")
    -> 187 rows, sum(price) == 1700078"""
    out = apply_scope(diamonds, "cut == 'Very Good' and price > 5000")
    assert out.height == 187
    assert out["price"].sum() == 1700078


def test_filter_data_factor_in(diamonds):
    """R: filter_data(diamonds, "clarity %in% c('SI2','SI1') & price > 18000")
    -> 14 rows, sum(price) == 256587"""
    out = apply_scope(diamonds, "clarity in ['SI2', 'SI1'] and price > 18000")
    assert out.height == 14
    assert out["price"].sum() == 256587


def test_filter_sort_slice_selects_the_rows_r_selects(diamonds):
    """R: get_data(d, "rid", filt = "price > 5000 & cut %in% c('Ideal','Premium')",
                   arr = "clarity, desc(price)", rows = 1:15)$rid"""
    out = apply_scope(
        diamonds, "price > 5000 and cut in ['Ideal', 'Premium']", "clarity -price", "1:15"
    )
    assert out["rid"].to_list() == [
        923, 574, 2216, 2379, 805, 2104, 1667, 1748, 835, 2062, 2065, 1181, 585, 2776, 489
    ]


def test_sort_on_one_column_with_ties_keeps_row_order(diamonds):
    """R: get_data(d, "rid", arr = "cut", rows = 1:10)$rid -- arrange() is
    stable, so tied rows keep their original order."""
    out = apply_scope(diamonds, "", "cut", "1:10")
    assert out["rid"].to_list() == [29, 72, 87, 120, 132, 141, 151, 238, 280, 283]


# R: get_data(d, "rid", arr = "cut, desc(color)", rows = 1:10)$rid
_R_CUT_DESC_COLOR = [72, 532, 565, 693, 986, 988, 1978, 132, 238, 334]

_UNSTABLE = (
    "data_scope.apply_scope sorts with DataFrame.sort(...) without "
    "maintain_order=True, so rows tied on the sort keys come back in an "
    "arbitrary order (polars' multi-column sort is not stable). R's arrange() "
    "is stable, so a slice taken after a sort on a column with ties picks "
    "different rows than radiant does."
)


def test_sort_on_two_columns_with_ties_is_stable_like_r(diamonds):
    out = apply_scope(diamonds, "", "cut -color", "1:10")
    assert out["rid"].to_list() == _R_CUT_DESC_COLOR


def test_generated_sort_code_is_stable_like_r(diamonds):
    ns = {"pl": pl, "data": diamonds}
    exec(sort_code("cut -color", "data"), ns)  # noqa: S102
    assert ns["data"].head(10)["rid"].to_list() == _R_CUT_DESC_COLOR


def test_sort_then_slice_differs_from_slice_alone(diamonds):
    """The interesting case: sort + slice and slice alone are different rows."""
    sliced = apply_scope(diamonds, "", "", "1:200")
    sorted_sliced = apply_scope(diamonds, "", "-price rid", "1:200")
    assert set(sliced["rid"]) != set(sorted_sliced["rid"])
    assert sorted_sliced["price"].min() == diamonds["price"].sort(descending=True)[199]


def test_slice_is_applied_after_the_filter(diamonds):
    """R: rows index the filtered data, not the original data."""
    out = apply_scope(diamonds, "cut == 'Fair'", "", "1:5")
    assert out.height == 5
    assert out["cut"].cast(pl.Utf8).unique().to_list() == ["Fair"]
    assert out["rid"].to_list() == diamonds.filter(pl.col("cut") == "Fair")["rid"].head(5).to_list()


# ===========================================================================
# radiant.data: explore with data_filter / arr / rows
# ===========================================================================


def _explore_scoped(df, by=None, **scope):
    scoped = apply_scope(df, **scope)
    return rsm.eda.explore(
        scoped, cols=["price", "carat"], agg=["n_obs", "mean", "sd"], by=by, to_dummies=False
    )


def test_explore_filter_sort_slice_matches_r(diamonds):
    """R: explore(diamonds, c("price","carat"), fun = c("n_obs","mean","sd"),
                  data_filter = "price > 5000", arr = "desc(carat)", rows = 1:200)$tab"""
    res = _explore_scoped(diamonds, data_filter="price > 5000", sort="-carat", slice="1:200")
    assert res["variable"].to_list() == ["price", "carat"]
    assert res["n_obs"].to_list() == [200, 200]
    np.testing.assert_allclose(res["mean"].to_numpy(), [13613.88, 1.96515], rtol=1e-10)
    np.testing.assert_allclose(
        res["sd"].to_numpy(), [3076.06078375032, 0.25663061756], rtol=1e-9
    )


def test_explore_filter_slice_without_sort_matches_r(diamonds):
    """Same call without arr: a different 200 rows.
    R: explore(diamonds, c("price","carat"), fun = c("n_obs","mean","sd"),
               data_filter = "price > 5000", rows = 1:200)$tab"""
    res = _explore_scoped(diamonds, data_filter="price > 5000", slice="1:200")
    np.testing.assert_allclose(res["mean"].to_numpy(), [8977.435, 1.3618], rtol=1e-10)
    np.testing.assert_allclose(
        res["sd"].to_numpy(), [3763.587943280859, 0.404637389629], rtol=1e-9
    )


def test_explore_grouped_filter_sort_slice_matches_r(diamonds):
    """R: explore(diamonds, c("price","carat"), byvar = "cut", fun = c("n_obs","mean","sd"),
                  data_filter = "price > 5000", arr = "desc(carat)", rows = 1:200)$tab"""
    res = _explore_scoped(
        diamonds, by="cut", data_filter="price > 5000", sort="-carat", slice="1:200"
    ).sort(["cut", "variable"], descending=[False, True])
    assert res["cut"].cast(pl.Utf8).to_list() == [
        "Fair", "Fair", "Good", "Good", "Very Good", "Very Good",
        "Premium", "Premium", "Ideal", "Ideal",
    ]
    assert res["n_obs"].to_list() == [12, 12, 18, 18, 48, 48, 71, 71, 51, 51]
    np.testing.assert_allclose(
        res["mean"].to_numpy(),
        [12351.41666666667, 2.0225, 13130.83333333333, 2.00777777778,
         13404.64583333333, 1.91729166667, 13706.3661971831, 1.98535211268,
         14149.58823529412, 1.95352941176],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        res["sd"].to_numpy(),
        [3086.302184531259, 0.15190457531, 2967.840435896306, 0.327621653302,
         3275.64855030094, 0.198169705731, 2929.953286996894, 0.280198047235,
         3106.61919891364, 0.263361527406],
        rtol=1e-9,
    )


# ===========================================================================
# radiant.basics: scoped data into the analysis
# ===========================================================================


def test_single_mean_filter_sort_slice_matches_r(diamonds):
    """R: single_mean(get_data(d, filt = "cut == 'Ideal'", arr = "desc(carat), rid",
                               rows = 1:200), "price", comp_value = 4000)$res"""
    sm = rsm.basics.single_mean(
        {"diamonds": diamonds}, "price", comp_value=4000,
        data_filter="cut == 'Ideal'", sort="-carat rid", slice="1:200",
    )
    assert sm.n == 200
    assert sm.df == 199
    assert sm.mean == pytest.approx(10088.865, rel=1e-12)
    assert sm.t_val == pytest.approx(22.3883338651, rel=1e-9)
    assert sm.p_val == pytest.approx(2.86868502567e-56, rel=1e-6)
    assert sm.ci.low == pytest.approx(9552.55992003, rel=1e-10)
    assert sm.ci.high == pytest.approx(10625.17008, rel=1e-10)


def test_single_mean_slice_without_sort_matches_r(diamonds):
    """R: single_mean(get_data(d, filt = "cut == 'Ideal'", rows = 1:200), "price",
                      comp_value = 4000)$res -- a different answer than the sorted one."""
    sm = rsm.basics.single_mean(
        {"diamonds": diamonds}, "price", comp_value=4000,
        data_filter="cut == 'Ideal'", slice="1:200",
    )
    assert sm.mean == pytest.approx(3443.87, rel=1e-12)
    assert sm.t_val == pytest.approx(-2.25762250829, rel=1e-9)
    assert sm.p_val == pytest.approx(0.0250550221161, rel=1e-8)


def test_single_mean_sort_with_ties_then_slice_matches_r(diamonds):
    """R: x <- get_data(d, "price", arr = "cut, desc(color)", rows = 1:10)
          single_mean(x, "price")$res  # estimate 4828.8, t 2.71670568627"""
    sm = rsm.basics.single_mean(
        {"diamonds": diamonds}, "price", sort="cut -color", slice="1:10"
    )
    assert sm.mean == pytest.approx(4828.8, rel=1e-12)
    assert sm.t_val == pytest.approx(2.71670568627, rel=1e-9)


def test_compare_means_filter_sort_slice_matches_r(diamonds):
    """R: compare_means(get_data(d, filt = "price > 5000", arr = "desc(carat), rid",
                                 rows = 1:300), "cut", "price")"""
    cm = rsm.basics.compare_means(
        {"diamonds": diamonds}, "cut", "price",
        data_filter="price > 5000", sort="-carat rid", slice="1:300",
    )
    ds = cm.descriptive_stats.with_columns(pl.col("cut").cast(pl.Utf8))
    ds = {r["cut"]: r for r in ds.iter_rows(named=True)}
    r_desc = {  # cut: (mean, n, sd)
        "Fair": (11460.6470588, 17, 2981.90851095),
        "Good": (11803.7575758, 33, 3111.11493309),
        "Very Good": (12609.6969697, 66, 3176.94660360),
        "Premium": (12567.0934579, 107, 3258.28008668),
        "Ideal": (13216.4805195, 77, 3272.26821225),
    }
    for cut, (mean, n, sd) in r_desc.items():
        assert ds[cut]["n"] == n
        assert ds[cut]["mean"] == pytest.approx(mean, rel=1e-10)
        assert ds[cut]["sd"] == pytest.approx(sd, rel=1e-10)

    cs = cm.comp_stats
    assert cs["Null hyp."].to_list()[0] == "Fair = Good"
    # R reports group1 - group2 t-values; pyrsm uses the same orientation
    np.testing.assert_allclose(
        cs["t.value"].to_numpy(),
        [-0.3797483147016, -1.3975753211799, -1.4026293112056, -2.1578394141236,
         -1.2064904048988, -1.2183814570933, -2.1484803938425, 0.0848442118149,
         -1.1229309582852, -1.3303309939273],
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        cs["p.value"].to_numpy(),
        [0.7065163484683, 0.1739756308857, 0.1743613461966, 0.0406391773945,
         0.2319796275694, 0.2282424811616, 0.0354942987231, 0.9325060629903,
         0.2634054120894, 0.1852621292773],
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        cs["df"].to_numpy(),
        [33.6802057499, 26.1728976242, 22.5236075746, 25.2628365276, 65.32507347,
         55.397931481, 63.5245105199, 140.4520504697, 138.8007828244, 163.4729968842],
        rtol=1e-9,
    )


def test_compare_means_filter_that_drops_a_level_matches_r(diamonds):
    """R: compare_means(get_data(d, filt = "cut != 'Fair'"), "cut", "price")
    -> 4 groups, 6 comparisons; Good vs Very Good t = 0.631018554923"""
    cm = rsm.basics.compare_means(
        {"diamonds": diamonds}, "cut", "price", data_filter="cut != 'Fair'"
    )
    assert cm.comp_stats.height == 6
    assert cm.comp_stats["Null hyp."][0] == "Good = Very Good"
    assert cm.comp_stats["t.value"][0] == pytest.approx(0.631018554923, rel=1e-9)
    assert cm.comp_stats["p.value"][-1] == pytest.approx(2.15802959941e-06, rel=1e-6)


def test_compare_props_filter_sort_slice_matches_r(titanic):
    """R: compare_props(get_data(ti, filt = "sex == 'female'", arr = "desc(fare), rid",
                                 rows = 1:300), "pclass", "survived", lev = "Yes")"""
    cp = rsm.basics.compare_props(
        {"titanic": titanic}, "pclass", "survived", lev="Yes",
        data_filter="sex == 'female'", sort="-fare rid", slice="1:300",
    )
    ds = cp.descriptive_stats
    assert ds["Yes"].to_list() == [126, 84, 33]
    assert ds["n"].to_list() == [131, 94, 75]
    np.testing.assert_allclose(
        ds["p"].to_numpy(), [0.961832061069, 0.893617021277, 0.44], rtol=1e-10
    )
    cs = cp.comp_stats
    np.testing.assert_allclose(
        cs["chisq.value"].to_numpy(), [4.09290238753, 73.75072839321, 40.29631205674],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        cs["p.value"].to_numpy(), [4.30636339776e-02, 8.86315717698e-18, 2.18220523619e-10],
        rtol=1e-7,
    )


def test_compare_props_filter_that_drops_a_level_matches_r(titanic):
    """R: compare_props(get_data(ti, filt = "pclass != '1st'"), "pclass", "survived",
                        lev = "Yes") -> one comparison, chisq 25.0080076018"""
    cp = rsm.basics.compare_props(
        {"titanic": titanic}, "pclass", "survived", lev="Yes", data_filter="pclass != '1st'"
    )
    assert cp.comp_stats.height == 1
    assert cp.comp_stats["chisq.value"][0] == pytest.approx(25.0080076018, rel=1e-9)


def test_cross_tabs_filter_sort_slice_matches_r(diamonds):
    """R: cross_tabs(get_data(d, filt = "price > 2000", arr = "desc(price), rid",
                              rows = 1:1000), "cut", "color")$cst"""
    ct = rsm.basics.cross_tabs(
        {"diamonds": diamonds}, "cut", "color",
        data_filter="price > 2000", sort="-price rid", slice="1:1000",
    )
    assert ct.chisq_test.statistic == pytest.approx(29.8435350244, rel=1e-9)
    assert ct.chisq_test.dof == 24
    assert ct.chisq_test.pvalue == pytest.approx(0.189991834377, rel=1e-8)


def test_cross_tabs_filter_that_drops_levels_matches_r(diamonds):
    """R: cross_tabs(get_data(d, filt = "cut != 'Fair' & clarity != 'I1'"),
                     "cut", "clarity")$cst -> X-squared 134.32681743, df 18"""
    ct = rsm.basics.cross_tabs(
        {"diamonds": diamonds}, "cut", "clarity",
        data_filter="cut != 'Fair' and clarity != 'I1'",
    )
    assert ct.chisq_test.statistic == pytest.approx(134.32681743, rel=1e-9)
    assert ct.chisq_test.dof == 18


def test_single_prop_filter_sort_slice_matches_r(titanic):
    """R: single_prop(get_data(ti, filt = "age > 30", arr = "fare, rid", rows = 1:250),
                      "survived", lev = "Yes")$res"""
    sp = rsm.basics.single_prop(
        {"titanic": titanic}, "survived", lev="Yes",
        data_filter="age > 30", sort="fare rid", slice="1:250",
    )
    assert sp.n == 250
    assert sp.ns == 71
    assert sp.p == pytest.approx(0.284)
    assert sp.p_val == pytest.approx(6.16846811362e-12, rel=1e-7)
    assert sp.ci.low == pytest.approx(0.228963799972, rel=1e-8)
    assert sp.ci.high == pytest.approx(0.344239498383, rel=1e-8)


def test_single_prop_after_filter_removes_a_level_matches_r(diamonds):
    """R: single_prop(get_data(d, filt = "color != 'D'"), "color")
    -> lev "E" (first remaining level), 554 of 2618."""
    sp = rsm.basics.single_prop(
        {"diamonds": diamonds}, "color", lev="E", data_filter="color != 'D'"
    )
    assert sp.n == 2618
    assert sp.ns == 554
    assert sp.p == pytest.approx(0.211611917494, rel=1e-10)
    assert sp.ci.low == pytest.approx(0.196104218065, rel=1e-8)
    assert sp.ci.high == pytest.approx(0.227767169777, rel=1e-8)


def test_correlation_scope_equals_prescoped_data(diamonds):
    """No R-side scope here (correlation has no arr/rows); compare against
    running the analysis on data scoped by hand with polars."""
    kwargs = {"vars": ["price", "carat", "depth", "table"]}
    scoped = rsm.basics.correlation(
        {"diamonds": diamonds}, data_filter="cut in ['Ideal', 'Premium']",
        sort="-carat rid", slice="1:400", **kwargs,
    )
    manual = _manual(
        diamonds, pl.col("cut").is_in(["Ideal", "Premium"]), ["carat", "rid"], [True, False], 400
    )
    plain = rsm.basics.correlation({"diamonds": manual}, **kwargs)
    np.testing.assert_allclose(np.asarray(scoped.cr), np.asarray(plain.cr), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(scoped.cp), np.asarray(plain.cp), rtol=1e-12)


# --- sorting every row (no slice) must not change an analysis ----------------


def test_sort_without_slice_does_not_change_single_mean(diamonds):
    a = rsm.basics.single_mean({"d": diamonds}, "price", data_filter="carat > 1")
    b = rsm.basics.single_mean({"d": diamonds}, "price", data_filter="carat > 1", sort="-price")
    assert (a.mean, a.sd, a.n) == pytest.approx((b.mean, b.sd, b.n), rel=1e-12)


def test_sort_without_slice_does_not_change_compare_means_tests(diamonds):
    a = rsm.basics.compare_means({"d": diamonds}, "cut", "price")
    b = rsm.basics.compare_means({"d": diamonds}, "cut", "price", sort="-price")
    np.testing.assert_allclose(
        a.comp_stats["t.value"].to_numpy(), b.comp_stats["t.value"].to_numpy(), rtol=1e-10
    )


def test_compare_means_descriptive_stats_in_level_order_regardless_of_sort(diamonds):
    levels = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    for sort in ("", "-price", "price"):
        cm = rsm.basics.compare_means({"d": diamonds}, "cut", "price", sort=sort)
        assert cm.descriptive_stats["cut"].cast(pl.Utf8).to_list() == levels


def test_sort_without_slice_does_not_change_cross_tabs(diamonds):
    a = rsm.basics.cross_tabs({"d": diamonds}, "cut", "color")
    b = rsm.basics.cross_tabs({"d": diamonds}, "cut", "color", sort="-price")
    assert a.chisq_test.statistic == pytest.approx(b.chisq_test.statistic, rel=1e-12)
    assert a.observed.equals(b.observed)


# ===========================================================================
# radiant.model: regress / logistic with data_filter, arr, rows
# ===========================================================================


def _coef(model, name):
    row = model.coef.filter(pl.col("index") == name)
    assert row.height == 1, f"{name} not in {model.coef['index'].to_list()}"
    return row.row(0, named=True)


def test_regress_filter_sort_slice_matches_r(diamonds):
    """R: regress(d, "price", c("carat", "clarity"),
                  data_filter = "cut %in% c('Ideal','Premium')",
                  arr = "desc(price), rid", rows = 1:500)"""
    reg = regress(
        {"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"],
        data_filter="cut in ['Ideal', 'Premium']", sort="-price rid", slice="1:500",
    )
    assert reg.fitted.nobs == 500
    r_coef = {  # label: (coefficient, std.error)
        "Intercept": (-8875.85290475, 799.973313807),
        "carat": (8933.56724115, 243.498497493),
        "clarity[SI2]": (4085.44886356, 689.504116106),
        "clarity[SI1]": (5297.99012188, 689.675768186),
        "clarity[VS2]": (6035.27824824, 689.792442910),
        "clarity[VS1]": (6873.16681476, 705.824173011),
        "clarity[VVS2]": (8433.16586981, 730.119411517),
        "clarity[VVS1]": (8694.37377447, 792.408927374),
        "clarity[IF]": (8843.67942801, 1018.309633666),
    }
    for name, (b, se) in r_coef.items():
        row = _coef(reg, name)
        assert row["coefficient"] == pytest.approx(b, rel=1e-9), name
        assert row["std.error"] == pytest.approx(se, rel=1e-9), name
    assert reg.fitted.rsquared == pytest.approx(0.744302572043, rel=1e-10)


def test_regress_slice_without_sort_matches_r(diamonds):
    """R: regress(d, "price", c("carat", "clarity"), rows = 1:500)"""
    reg = regress(
        {"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"], slice="1:500"
    )
    assert _coef(reg, "Intercept")["coefficient"] == pytest.approx(-5717.81617246, rel=1e-9)
    assert _coef(reg, "carat")["coefficient"] == pytest.approx(8596.07486769, rel=1e-9)
    assert _coef(reg, "carat")["std.error"] == pytest.approx(116.902423797, rel=1e-9)


def test_regress_sort_with_ties_then_slice_matches_r(diamonds):
    """R: regress(d, "price", "carat", arr = "cut, desc(color)", rows = 1:40)"""
    reg = regress(
        {"diamonds": diamonds}, rvar="price", evar=["carat"], sort="cut -color", slice="1:40"
    )
    assert _coef(reg, "Intercept")["coefficient"] == pytest.approx(-3779.47049412, rel=1e-9)
    assert _coef(reg, "carat")["coefficient"] == pytest.approx(7478.42748901, rel=1e-9)


def test_regress_filter_that_drops_a_level_keeps_slopes(diamonds):
    """R: regress(d, "price", c("carat", "clarity"), data_filter = "clarity != 'I1'")
    -> carat 8485.453828453 (se 51.1273517293). The numeric slope is
    identified however the empty level is handled."""
    reg = regress(
        {"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"],
        data_filter="clarity != 'I1'",
    )
    assert _coef(reg, "carat")["coefficient"] == pytest.approx(8485.453828453, rel=1e-8)


def test_regress_filter_that_drops_a_level_matches_r(diamonds):
    reg = regress(
        {"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"],
        data_filter="clarity != 'I1'",
    )
    assert reg.coef["index"].to_list() == [
        "Intercept", "clarity[SI1]", "clarity[VS2]", "clarity[VS1]",
        "clarity[VVS2]", "clarity[VVS1]", "clarity[IF]", "carat",
    ]
    assert _coef(reg, "Intercept")["coefficient"] == pytest.approx(-4041.323510086, rel=1e-8)
    assert _coef(reg, "clarity[SI1]")["coefficient"] == pytest.approx(828.559323059, rel=1e-8)
    assert _coef(reg, "clarity[SI1]")["std.error"] == pytest.approx(70.691759886, rel=1e-8)


def test_logistic_filter_sort_slice_matches_r(titanic):
    """R: logistic(ti, "survived", c("pclass", "sex", "age"), lev = "Yes",
                   data_filter = "age > 10", arr = "desc(fare), rid", rows = 1:600)"""
    lr = logistic(
        {"titanic": titanic}, rvar="survived", lev="Yes", evar=["pclass", "sex", "age"],
        data_filter="age > 10", sort="-fare rid", slice="1:600",
    )
    assert lr.fitted.nobs == 600
    r_coef = {  # label: (OR, coefficient, std.error)
        "Intercept": (56.9560688194417, 4.0422802482094, 0.49127767500007),
        "pclass[2nd]": (0.1815535921423, -1.7062043953509, 0.28969152594068),
        "pclass[3rd]": (0.0474436571031, -3.0482124382446, 0.38830269846620),
        "sex[male]": (0.0276243782064, -3.5890566277633, 0.27891292778524),
        "age": (0.97491564201, -0.0254043327428, 0.00953009671479),
    }
    for name, (odds, b, se) in r_coef.items():
        row = _coef(lr, name)
        assert row["OR"] == pytest.approx(odds, rel=1e-6), name
        assert row["coefficient"] == pytest.approx(b, rel=1e-6), name
        assert row["std.error"] == pytest.approx(se, rel=1e-5), name


def test_logistic_filter_that_drops_a_level_keeps_real_levels(titanic):
    """R: logistic(ti, "survived", c("pclass", "sex"), lev = "Yes",
                   data_filter = "pclass != '3rd'")"""
    lr = logistic(
        {"titanic": titanic}, rvar="survived", lev="Yes", evar=["pclass", "sex"],
        data_filter="pclass != '3rd'",
    )
    assert _coef(lr, "Intercept")["coefficient"] == pytest.approx(3.25488821375, rel=1e-6)
    assert _coef(lr, "pclass[2nd]")["coefficient"] == pytest.approx(-1.14447371013, rel=1e-6)
    assert _coef(lr, "sex[male]")["coefficient"] == pytest.approx(-3.87343655023, rel=1e-6)
    assert _coef(lr, "pclass[2nd]")["std.error"] == pytest.approx(0.252191382078, rel=1e-5)


def test_logistic_filter_that_drops_a_level_has_no_empty_level(titanic):
    lr = logistic(
        {"titanic": titanic}, rvar="survived", lev="Yes", evar=["pclass", "sex"],
        data_filter="pclass != '3rd'",
    )
    assert lr.coef["index"].to_list() == ["Intercept", "pclass[2nd]", "sex[male]"]


@pytest.mark.parametrize(
    ("build", "kwargs"),
    [
        (regress, {"rvar": "price", "evar": ["carat", "clarity", "depth"]}),
        (logistic, {"rvar": "cut_ideal", "lev": "yes", "evar": ["price", "carat"]}),
    ],
    ids=["regress", "logistic"],
)
def test_model_scope_equals_prescoped_data(build, kwargs, diamonds):
    df = diamonds.with_columns(
        pl.when(pl.col("cut") == "Ideal").then(pl.lit("yes")).otherwise(pl.lit("no")).alias("cut_ideal")
    )
    scoped = build(
        {"d": df}, data_filter="price > 1000 and color in ['D', 'E', 'F', 'G']",
        sort="-carat rid", slice="1:700", **kwargs,
    )
    manual = _manual(
        df, (pl.col("price") > 1000) & pl.col("color").is_in(["D", "E", "F", "G"]),
        ["carat", "rid"], [True, False], 700,
    )
    plain = build({"d": manual}, **kwargs)
    assert scoped.fitted.nobs == plain.fitted.nobs == 700
    np.testing.assert_allclose(
        scoped.fitted.params.to_numpy(), plain.fitted.params.to_numpy(), rtol=1e-10
    )


def test_mlp_scope_equals_prescoped_data(titanic):
    kwargs = {"rvar": "survived", "lev": "Yes", "evar": ["pclass", "sex", "age"],
              "hidden_layer_sizes": (2,), "random_state": 1234}
    scoped = mlp(
        {"t": titanic}, data_filter="age > 5", sort="-fare rid", slice="1:500", **kwargs
    )
    manual = _manual(titanic, pl.col("age") > 5, ["fare", "rid"], [True, False], 500)
    plain = mlp({"t": manual}, **kwargs)
    a = scoped.predict()["prediction"].to_numpy()
    b = plain.predict()["prediction"].to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-10)


# ===========================================================================
# radiant.multivariate: scoped data into the analysis
# ===========================================================================


def test_pre_factor_filter_sort_slice_matches_r(diamonds):
    """R: pre_factor(get_data(d, filt = "cut == 'Ideal'", arr = "desc(price), rid",
                              rows = 1:500), c("price", "carat", "table"))"""
    pf = rsm.multivariate.pre_factor(
        {"diamonds": diamonds}, ["price", "carat", "table"],
        data_filter="cut == 'Ideal'", sort="-price rid", slice="1:500",
    )
    assert pf.nobs == 500
    np.testing.assert_allclose(
        pf.pre_r2, [0.75395329383207, 0.75404895035794, 0.00335816962984], rtol=1e-9
    )
    np.testing.assert_allclose(
        pf.pre_eigen, [1.875364272572, 0.99293801344, 0.131697713989], rtol=1e-9
    )
    assert pf.pre_kmo == pytest.approx(0.502103813525, rel=1e-9)


def test_full_factor_filter_sort_slice_matches_r(diamonds):
    """R: full_factor(get_data(d, filt = "cut == 'Ideal'", arr = "desc(price), rid",
                               rows = 1:500), c("price", "carat", "table"))$floadings"""
    ff = rsm.multivariate.full_factor(
        {"diamonds": diamonds}, ["price", "carat", "table"],
        data_filter="cut == 'Ideal'", sort="-price rid", slice="1:500",
    )
    np.testing.assert_allclose(
        np.abs(np.asarray(ff.floadings).ravel()),
        [0.964330998258, 0.964575942301, 0.122569367725],
        rtol=1e-9,
    )


def test_hclus_filter_sort_slice_matches_r(shopping):
    """R: hclus(get_data(shopping, filt = "v1 > 1", arr = "desc(v2), id", rows = 1:15),
                "v1:v6")$hc_out$height"""
    hc = rsm.multivariate.hclus(
        {"shopping": shopping}, "v1:v6", data_filter="v1 > 1", sort="-v2 id", slice="1:15"
    )
    r_height = [
        0.734844271413, 1.073249362723, 1.31551854648, 1.884670326373, 2.208750523931,
        2.719093927496, 3.092698952692, 4.182987657451, 4.319459424451, 5.500393966572,
        6.805412271304, 13.642830404242, 36.361073009788, 84.159017355085,
    ]
    np.testing.assert_allclose(np.sort(hc.height), r_height, rtol=1e-9)


def test_kclus_filter_sort_slice_matches_r(shopping):
    """R: kclus(get_data(shopping, filt = "v1 > 1", arr = "desc(v2), id", rows = 1:15),
                "v1:v6")$clus_means"""
    km = rsm.multivariate.kclus(
        {"shopping": shopping}, "v1:v6", data_filter="v1 > 1", sort="-v2 id", slice="1:15"
    )
    r_means = np.array(
        [
            [3.125, 5.25, 3.0, 5.375, 4.0, 5.375],
            [5.57142857143, 3.85714285714, 6.0, 3.0, 2.0, 4.0],
        ]
    )
    got = km.clus_means.to_numpy()
    got = got[np.argsort(got[:, 0])]
    np.testing.assert_allclose(got, r_means, rtol=1e-9)


@pytest.mark.parametrize(
    ("build", "kwargs", "result"),
    [
        (rsm.multivariate.pre_factor, {"vars": "v1:v6"}, "pre_r2"),
        (rsm.multivariate.full_factor, {"vars": "v1:v6", "nr_fact": 2}, "floadings"),
        (rsm.multivariate.hclus, {"vars": "v1:v6"}, "height"),
    ],
    ids=["pre_factor", "full_factor", "hclus"],
)
def test_multivariate_scope_equals_prescoped_data(build, kwargs, result, shopping):
    scoped = build(
        {"shopping": shopping}, data_filter="v3 > 2", sort="-v4 id", slice="1:12", **kwargs
    )
    manual = _manual(shopping, pl.col("v3") > 2, ["v4", "id"], [True, False], 12)
    plain = build({"shopping": manual}, **kwargs)
    a, b = np.asarray(getattr(scoped, result)), np.asarray(getattr(plain, result))
    np.testing.assert_allclose(np.abs(a), np.abs(b), rtol=1e-10)


def test_multivariate_sort_then_slice_differs_from_slice_alone(shopping):
    """Sort order matters: the same slice of sorted and unsorted data are
    different respondents, so the analysis differs."""
    a = rsm.multivariate.pre_factor({"s": shopping}, "v1:v6", slice="1:12")
    b = rsm.multivariate.pre_factor({"s": shopping}, "v1:v6", sort="-v4 id", slice="1:12")
    assert not np.allclose(a.pre_r2, b.pre_r2)
