"""Radiant (R) parity: ports of R testthat blocks that had no pyrsm equivalent.

These are the non-scope gaps (no filter / sort / slice); the scope tests live
in ``test_radiant_parity_scope.py``. Every expected number comes from the R
test it ports (quoted in each docstring), or from running that R code with
the installed radiant packages.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import pyrsm as rsm
from pyrsm.model.logistic import logistic
from pyrsm.model.regress import regress
from pyrsm.stats import scale_df
from pyrsm.utils import sig_stars

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.fixture(scope="module")
def diamonds() -> pl.DataFrame:
    return pl.read_parquet(DATA / "data" / "diamonds.parquet")


@pytest.fixture(scope="module")
def titanic() -> pl.DataFrame:
    return pl.read_parquet(DATA / "data" / "titanic.parquet")


def _coef(model, name):
    row = model.coef.filter(pl.col("index") == name)
    assert row.height == 1, f"{name} not in {model.coef['index'].to_list()}"
    return row.row(0, named=True)


# ===========================================================================
# radiant.data  test_funs.R
# ===========================================================================


def test_sig_stars_r_vector():
    """R: sig_stars(c(.0009, .049, .009, .4, .09)) == c("***", "*", "**", "", ".")
    pyrsm pads "not significant" as " " (column alignment), so compare stripped."""
    out = sig_stars([0.0009, 0.049, 0.009, 0.4, 0.09]).to_list()
    assert [s.strip() for s in out] == ["***", "*", "**", "", "."]


def test_explore_numeric_columns_match_r(diamonds):
    """R: explore(diamonds, "price:x")$tab (fun = mean, sd). Factor columns are
    left out: R summarises them as a 0/1 indicator for the *first* level while
    pyrsm uses drop_first dummies, a deliberate difference."""
    res = rsm.eda.explore(
        diamonds, cols=["price", "carat", "depth", "table", "x"], agg=["mean", "sd"]
    )
    assert res.columns == ["variable", "mean", "sd"]
    np.testing.assert_allclose(
        res["mean"].to_numpy(),
        [3907.186, 0.794283333333333, 61.7526666666667, 57.4653333333333, 5.72182333333333],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        res["sd"].to_numpy(),
        [3956.91540005997, 0.473826329139292, 1.44602785395269, 2.24110219949434,
         1.12405453974662],
        rtol=1e-12,
    )


def test_explore_one_variable_matches_r(diamonds):
    """R: explore(diamonds, "price") -> mean 3907.186, sd 3956.91540005997"""
    res = rsm.eda.explore(diamonds, cols=["price"], agg=["mean", "sd"])
    assert res["variable"].to_list() == ["price"]
    assert res["mean"][0] == pytest.approx(3907.186, rel=1e-12)
    assert res["sd"][0] == pytest.approx(3956.91540005997, rel=1e-12)


def test_explore_column_layouts_match_r(diamonds):
    """R: colnames(explore(...)$tab) for the 1x1, 1x1x1, 1x1x2 and 2x2x2 cases."""
    assert rsm.eda.explore(diamonds, cols=["price"], agg=["n_obs"]).columns == [
        "variable", "n_obs"
    ]
    assert rsm.eda.explore(diamonds, cols=["price"], agg=["n_obs"], by="color").columns == [
        "color", "variable", "n_obs"
    ]
    assert rsm.eda.explore(
        diamonds, cols=["price"], agg=["n_obs"], by=["color", "cut"]
    ).columns == ["color", "cut", "variable", "n_obs"]
    assert rsm.eda.explore(
        diamonds, cols=["price", "carat"], agg=["n_obs", "mean"], by=["color", "cut"]
    ).columns == ["color", "cut", "variable", "n_obs", "mean"]


def test_explore_two_groupings_first_row_matches_r(diamonds):
    """R: explore(diamonds, "price", byvar = c("color", "cut"), fun = "n_obs")$tab[1, ]
    -> D / Fair / price / 15, and 35 rows in total."""
    res = rsm.eda.explore(diamonds, cols=["price"], agg=["n_obs"], by=["color", "cut"])
    assert res.height == 35
    first = res.row(0, named=True)
    assert (str(first["color"]), str(first["cut"]), first["variable"], first["n_obs"]) == (
        "D", "Fair", "price", 15
    )


# ===========================================================================
# radiant.basics  test_stats.R
# ===========================================================================


def test_compare_means_diamonds_matches_r(diamonds):
    """R: compare_means(diamonds, "cut", "price") -- 'compare_means 1' and '2'."""
    cm = rsm.basics.compare_means({"diamonds": diamonds}, "cut", "price")
    ds = {str(r["cut"]): r for r in cm.descriptive_stats.iter_rows(named=True)}
    r_desc = {  # mean, n, sd, se, me
        "Fair": (4505.238, 101, 3749.540, 373.093, 740.206),
        "Good": (4130.433, 275, 3730.354, 224.949, 442.848),
        "Very Good": (3959.916, 677, 3895.899, 149.732, 293.995),
        "Premium": (4369.409, 771, 4236.977, 152.591, 299.544),
        "Ideal": (3470.224, 1176, 3827.423, 111.610, 218.977),
    }
    for cut, (mean, n, sd, se, me) in r_desc.items():
        got = ds[cut]
        assert got["n"] == n
        assert (got["mean"], got["sd"], got["se"], got["me"]) == pytest.approx(
            (mean, sd, se, me), abs=5e-4
        )
    cs = cm.comp_stats
    assert cs["Null hyp."].to_list()[:2] == ["Fair = Good", "Fair = Very Good"]
    np.testing.assert_allclose(
        cs["diff"].to_numpy(),
        [374.805, 545.322, 135.829, 1035.014, 170.517, -238.976, 660.209, -409.493,
         489.692, 899.185],
        atol=5e-4,
    )
    np.testing.assert_allclose(
        cs["se"].to_numpy(),
        [435.661, 402.018, 403.091, 389.429, 270.225, 271.820, 251.115, 213.784,
         186.752, 189.052],
        atol=5e-4,
    )
    np.testing.assert_allclose(
        cs["df"].to_numpy(),
        [177.365, 134.291, 135.759, 118.618, 528.529, 543.242, 419.577, 1442.922,
         1389.163, 1527.729],
        atol=5e-4,
    )
    np.testing.assert_allclose(
        cs["2.5%"].to_numpy(),
        [-484.941, -249.783, -661.321, 263.879, -360.330, -772.922, 166.609, -828.853,
         123.346, 528.355],
        atol=5e-4,
    )
    stars = [s.strip() for s in cs[""].to_list()]
    assert stars == ["", "", "", "**", "", "", "**", ".", "**", "***"]


def test_compare_props_titanic_matches_r(titanic):
    """R: compare_props(titanic, "pclass", "survived") -- 'compare_props 1' and '2'."""
    cp = rsm.basics.compare_props({"titanic": titanic}, "pclass", "survived", lev="Yes")
    ds = cp.descriptive_stats
    assert ds["pclass"].cast(pl.Utf8).to_list() == ["1st", "2nd", "3rd"]
    assert ds["Yes"].to_list() == [179, 115, 131]
    assert ds["n"].to_list() == [282, 261, 500]
    np.testing.assert_allclose(ds["p"].to_numpy(), [0.635, 0.441, 0.262], atol=5e-4)
    np.testing.assert_allclose(ds["sd"].to_numpy(), [0.481, 0.496, 0.440], atol=5e-4)
    np.testing.assert_allclose(ds["se"].to_numpy(), [0.029, 0.031, 0.020], atol=5e-4)
    np.testing.assert_allclose(ds["me"].to_numpy(), [0.056, 0.060, 0.039], atol=5e-4)
    cs = cp.comp_stats
    np.testing.assert_allclose(cs["diff"].to_numpy(), [0.194, 0.373, 0.179], atol=5e-4)
    np.testing.assert_allclose(
        cs["chisq.value"].to_numpy(), [20.576, 104.704, 25.008], atol=5e-4
    )
    np.testing.assert_allclose(cs["2.5%"].to_numpy(), [0.112, 0.305, 0.107], atol=5e-4)
    np.testing.assert_allclose(cs["97.5%"].to_numpy(), [0.277, 0.441, 0.250], atol=5e-4)


def test_single_prop_diamonds_color_matches_r(diamonds):
    """R: single_prop(diamonds, "color") -> lev D: p 0.127, ns 382, n 3000,
    sd 0.333, se 0.006, me 0.012, diff -0.373, CI 0.116 - 0.140"""
    sp = rsm.basics.single_prop({"diamonds": diamonds}, "color", lev="D")
    assert (sp.ns, sp.n, sp.n_missing) == (382, 3000, 0)
    assert (sp.p, sp.sd, sp.se, sp.me, sp.diff) == pytest.approx(
        (0.127, 0.333, 0.006, 0.012, -0.373), abs=5e-4
    )
    assert (sp.ci.low, sp.ci.high) == pytest.approx((0.116, 0.140), abs=5e-4)
    assert sp.p_val < 0.001


def test_single_prop_default_level_is_first_level(diamonds):
    sp = rsm.basics.single_prop({"diamonds": diamonds}, "color")
    assert sp.lev == "D"
    assert sp.ns == 382


def test_single_prop_custom_level_and_comp_value_matches_r(diamonds):
    """R: single_prop(diamonds, "clarity", lev = "IF", comp_value = 0.05)
    -> p 0.033, ns 99, diff -0.017, CI 0.027 - 0.040"""
    sp = rsm.basics.single_prop(
        {"diamonds": diamonds}, "clarity", lev="IF", comp_value=0.05
    )
    assert sp.ns == 99
    assert (sp.p, sp.diff) == pytest.approx((0.033, -0.017), abs=5e-4)
    assert (sp.ci.low, sp.ci.high) == pytest.approx((0.027, 0.040), abs=5e-4)
    assert sp.p_val < 0.001


def test_single_mean_diamonds_carat_matches_r(diamonds):
    """R: single_mean(diamonds, "carat") -> "0.794 0.009 91.816 < .001 2999 0.777 0.811 ***" """
    sm = rsm.basics.single_mean({"diamonds": diamonds}, "carat")
    assert (sm.diff, sm.se, sm.t_val) == pytest.approx((0.794, 0.009, 91.816), abs=5e-4)
    assert sm.df == 2999
    assert (sm.ci.low, sm.ci.high) == pytest.approx((0.777, 0.811), abs=5e-4)
    assert sm.p_val < 0.001


def test_single_mean_titanic_age_matches_r(titanic):
    """R: single_mean(titanic, "age", comp_value = 40)
    -> "-10.187 0.445 -22.9 < .001 1042 28.94 30.686 ***" """
    sm = rsm.basics.single_mean({"titanic": titanic}, "age", comp_value=40)
    assert (sm.diff, sm.se) == pytest.approx((-10.187, 0.445), abs=5e-4)
    assert sm.t_val == pytest.approx(-22.9, abs=5e-3)
    assert sm.df == 1042
    assert (sm.ci.low, sm.ci.high) == pytest.approx((28.94, 30.686), abs=5e-4)


# ===========================================================================
# radiant.model  test_stats.R
# ===========================================================================


def test_regress_diamonds_carat_matches_r(diamonds):
    """R: regress(diamonds, "price", c("carat", "clarity"))
    -> "carat 8438.030 51.101 165.125 < .001 ***" """
    reg = regress({"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"])
    row = _coef(reg, "carat")
    assert (row["coefficient"], row["std.error"], row["t.value"]) == pytest.approx(
        (8438.030, 51.101, 165.125), abs=5e-4
    )
    assert row["p.value"] < 0.001


@pytest.mark.parametrize(
    ("ivar", "pred", "lo", "hi"),
    [
        # R: predict(regress(diamonds, "price", c("carat", "clarity")),
        #            pred_cmd = "carat = 1:10")[9, ]
        (None, 72769.811, 71948.301, 73591.322),
        # R: predict(regress(..., int = "I(carat^2)"), pred_cmd = "carat = 1:10")[9, ]
        (["I(carat**2)"], 114304.420, 104924.680, 123684.159),
    ],
    ids=["linear", "quadratic"],
)
def test_regress_predict_cmd_matches_r(diamonds, ivar, pred, lo, hi):
    reg = regress(
        {"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"], ivar=ivar
    )
    out = reg.predict(cmd={"carat": list(range(1, 11))}, ci=True)
    row = out.row(8, named=True)
    assert row["carat"] == 9
    assert str(row["clarity"]) == "SI1"
    assert row["prediction"] == pytest.approx(pred, abs=5e-4)
    # CI agrees to ~1e-4 relative; exact agreement is the xfail below
    assert (row["2.50%"], row["97.50%"]) == pytest.approx((lo, hi), rel=1e-4)


def test_regress_predict_ci_uses_t_distribution_like_r(diamonds):
    reg = regress({"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"])
    row = reg.predict(cmd={"carat": list(range(1, 11))}, ci=True).row(8, named=True)
    assert (row["2.50%"], row["97.50%"]) == pytest.approx((71948.301, 73591.322), abs=5e-4)


def test_regress_predict_with_date_matches_r(diamonds):
    """R: predict(regress(diamonds, "price", c("carat", "clarity", "date")),
                  pred_cmd = "carat = 1:10")[9, ] -> SI1 2012-03-19 9 72719.464"""
    reg = regress(
        {"diamonds": diamonds}, rvar="price", evar=["carat", "clarity", "date"]
    )
    assert "date" in reg.coef["index"].to_list()
    row = reg.predict(cmd={"carat": list(range(1, 11))}).row(8, named=True)
    assert str(row["date"]) == "2012-03-19"
    assert row["prediction"] == pytest.approx(72719.464, abs=5e-4)


def test_logistic_titanic_matches_r(titanic):
    """R: logistic(titanic, "survived", c("pclass", "sex"))
    -> "sex|male 0.080 -92.0% -2.522 0.163 -15.447 < .001 ***" """
    lr = logistic(
        {"titanic": titanic}, rvar="survived", lev="Yes", evar=["pclass", "sex"]
    )
    row = _coef(lr, "sex[male]")
    assert row["OR"] == pytest.approx(0.080, abs=5e-4)
    assert row["OR%"] == pytest.approx(-92.0, abs=5e-2)
    assert (row["coefficient"], row["std.error"], row["z.value"]) == pytest.approx(
        (-2.522, 0.163, -15.447), abs=5e-4
    )


def test_logistic_predict_cmd_and_data_match_r(titanic):
    """R: predict(logistic(titanic, "survived", c("pclass", "sex")),
                  pred_cmd = "pclass = levels(pclass); sex = 'female'")
       -> "2nd female 0.779 0.712 0.833"
       predict(..., pred_data = titanic)[1, ] -> "1st female 0.896 0.856 0.926" """
    lr = logistic(
        {"titanic": titanic}, rvar="survived", lev="Yes", evar=["pclass", "sex"]
    )
    out = lr.predict(cmd={"pclass": ["1st", "2nd", "3rd"], "sex": "female"}, ci=True)
    row = out.row(1, named=True)
    assert (str(row["pclass"]), str(row["sex"])) == ("2nd", "female")
    assert (row["prediction"], row["2.50%"], row["97.50%"]) == pytest.approx(
        (0.779, 0.712, 0.833), abs=5e-4
    )
    row = lr.predict(data=titanic, ci=True).row(0, named=True)
    assert (row["prediction"], row["2.50%"], row["97.50%"]) == pytest.approx(
        (0.896, 0.856, 0.926), abs=5e-4
    )


def test_logistic_predict_with_quadratic_term_matches_r(titanic):
    """R: logistic(titanic, "survived", c("pclass", "sex", "age"), int = "I(age^2)")
       pred_cmd = "pclass = levels(pclass); sex = 'female'; age = 1:100"
         -> "1st female 1 0.976 0.952 0.988"
       pred_data = titanic -> "1st female 29.000 0.919 0.880 0.945" """
    lr = logistic(
        {"titanic": titanic}, rvar="survived", lev="Yes",
        evar=["pclass", "sex", "age"], ivar=["I(age**2)"],
    )
    out = lr.predict(
        cmd={"pclass": ["1st", "2nd", "3rd"], "sex": "female", "age": list(range(1, 101))},
        ci=True,
    )
    row = out.row(0, named=True)
    assert (str(row["pclass"]), row["age"]) == ("1st", 1)
    assert (row["prediction"], row["2.50%"], row["97.50%"]) == pytest.approx(
        (0.976, 0.952, 0.988), abs=5e-4
    )
    row = lr.predict(data=titanic, ci=True).row(0, named=True)
    assert row["age"] == 29
    assert (row["prediction"], row["2.50%"], row["97.50%"]) == pytest.approx(
        (0.919, 0.880, 0.945), abs=5e-4
    )


# ===========================================================================
# radiant.model  test_standardize.R (scale_df part) / test_pdp_vip.R (varimp)
# ===========================================================================


def test_scale_df_scaling_factor_matches_r():
    """R: scale_df(dat) vs scale_df(dat, sf = 1) for x = c(1,2,3,4,10), y = c(2,4,6,8,10):
    (x - mean) / (sf * sd), so 1 X SD values are twice the 2 X SD values."""
    dat = pl.DataFrame({"x": [1.0, 2, 3, 4, 10], "y": [2.0, 4, 6, 8, 10]})
    s2 = scale_df(dat, sf=2, ddof=1)
    s1 = scale_df(dat, sf=1, ddof=1)
    x = dat["x"].to_numpy()
    np.testing.assert_allclose(s2["x"].to_numpy(), (x - x.mean()) / (2 * x.std(ddof=1)))
    np.testing.assert_allclose(s1["x"].to_numpy(), (x - x.mean()) / (1 * x.std(ddof=1)))
    np.testing.assert_allclose(s1["x"].to_numpy(), 2 * s2["x"].to_numpy())
    np.testing.assert_allclose(s1["y"].to_numpy(), 2 * s2["y"].to_numpy())

    # R stores the raw means/sds regardless of sf. pyrsm stores sf * sd (and
    # uses supplied stds as-is), a deliberate difference: means agree and the
    # stored "stds" differ by exactly the scaling factor.
    _, m1, sd1 = scale_df(dat, sf=1, ddof=1, stats=True)
    _, m2, sd2 = scale_df(dat, sf=2, ddof=1, stats=True)
    assert m1 == m2
    assert sd1["x"] == pytest.approx(x.std(ddof=1))
    assert {k: 2 * v for k, v in sd1.items()} == pytest.approx(sd2)


@pytest.mark.parametrize(
    ("which", "kwargs", "positive"),
    [
        ("regress", {"rvar": "price", "evar": ["carat", "clarity"]}, True),
        ("logistic", {"rvar": "survived", "lev": "Yes", "evar": ["pclass", "sex"]}, False),
    ],
)
def test_permutation_importance_structure_and_sorted(which, kwargs, positive, diamonds, titanic):
    """R: varimp(regress(...)) / varimp(logistic(...)): a data.frame of
    Variable / Importance, sorted with the most important first."""
    data = {"diamonds": diamonds} if which == "regress" else {"titanic": titanic}
    model = {"regress": regress, "logistic": logistic}[which](data, **kwargs)
    _, imp = model.plot(plots="pip", ret=True)
    assert isinstance(imp, pl.DataFrame)
    assert set(imp.columns) >= {"variable", "importance"}
    assert sorted(imp["variable"].to_list()) == sorted(kwargs["evar"])
    vals = imp["importance"].to_list()
    assert vals == sorted(vals, reverse=True)
    if positive:
        assert all(v > 0 for v in vals)
