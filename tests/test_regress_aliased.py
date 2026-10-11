"""Perfect multicollinearity: regress and logistic drop columns as R does.

R's ``lm()`` / ``glm()`` drop a column that is a linear combination of the
columns before it (LINPACK QR, left to right, in ``model.matrix`` order) and
report its coefficient as NA; statsmodels would spread the effect over the
collinear columns with a pseudo-inverse. Expected values are from R 4 with
radiant.model 1.6.x on the same data (carat2 = 2 * carat, cc = carat + depth).
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import pyrsm as rsm

DIAMONDS = Path(__file__).parents[1] / "examples" / "data" / "data" / "diamonds.parquet"
EVAR = ["carat", "depth", "carat2", "cc", "clarity"]


@pytest.fixture(scope="module")
def dia():
    d = pl.read_parquet(DIAMONDS)
    return d.with_columns(
        (2 * pl.col("carat")).alias("carat2"),
        (pl.col("carat") + pl.col("depth")).alias("cc"),
        (pl.col("clarity") == "SI2").cast(pl.Float64).alias("si2"),
        pl.when(pl.col("price") > 3000).then(pl.lit("yes")).otherwise(pl.lit("no")).alias("high"),
    )


def _coef(model, name):
    row = model.coef.filter(pl.col("index") == name).row(0, named=True)
    return row["coefficient"], row["std.error"]


def test_regress_drops_aliased_columns_like_lm(dia):
    r = rsm.model.regress({"d": dia}, rvar="price", evar=EVAR)
    assert r.aliased == ["carat2", "cc"]
    assert "carat2" not in r.coef["index"].to_list() and "cc" not in r.coef["index"].to_list()
    assert _coef(r, "carat") == pytest.approx((8437.08989367, 51.04414348), rel=1e-8)
    assert _coef(r, "depth") == pytest.approx((-43.59482203, 15.62207785), rel=1e-8)
    assert _coef(r, "Intercept") == pytest.approx((-4030.68020225, 1006.60413790), rel=1e-8)
    assert r.fitted.df_model == 9 and r.fitted.df_resid == 2990


def test_regress_summary_notes_like_radiant(dia, capsys):
    r = rsm.model.regress({"d": dia}, rvar="price", evar=EVAR)
    r.summary(vif=True, ci=True, plain=True)
    out = capsys.readouterr().out
    assert "perfect multicollinearity" in out
    assert "dropped from the estimation: carat2, cc" in out
    assert "Multicollinearity diagnostics were not calculated." in out
    assert "Confidence intervals were not calculated." in out


def test_regress_robust_drops_aliased_columns(dia):
    r = rsm.model.regress({"d": dia}, rvar="price", evar=EVAR, robust=True)
    assert _coef(r, "carat") == pytest.approx((8437.08989367, 83.57556210), rel=1e-8)
    assert _coef(r, "depth") == pytest.approx((-43.59482203, 17.74595069), rel=1e-8)
    assert _coef(r, "Intercept")[1] == pytest.approx(1167.03382185, rel=1e-8)


def test_regress_drops_in_r_column_order(dia):
    # R keeps si2 (listed first) and drops clarity|SI2; patsy puts the
    # categorical columns first, which would have dropped si2.
    r = rsm.model.regress({"d": dia}, rvar="price", evar=["si2", "carat", "clarity"])
    assert r.aliased == ["clarity[T.SI2]"]
    assert _coef(r, "si2") == pytest.approx((2790.759516, 201.3947112), rel=1e-8)
    assert _coef(r, "carat") == pytest.approx((8438.030310, 51.1009135), rel=1e-8)


def test_regress_aliased_interactions(dia):
    r = rsm.model.regress(
        {"d": dia}, rvar="price", evar=["carat", "carat2", "clarity"],
        ivar=["carat:clarity", "carat2:clarity"],
    )
    assert r.aliased[0] == "carat2" and len(r.aliased) == 8
    assert all(a.startswith("carat2") for a in r.aliased)
    assert _coef(r, "carat") == pytest.approx((5165.51923815, 409.9263327), rel=1e-8)
    assert _coef(r, "carat:clarity[IF]") == pytest.approx((5904.44929929, 597.4482475), rel=1e-8)


def test_regress_prediction_interval_uses_t_like_predict_lm(dia):
    r = rsm.model.regress({"d": dia}, rvar="price", evar=EVAR)
    p = r.predict(dia.head(3), ci=True)
    assert p["prediction"].to_list() == pytest.approx([411.02326455, -367.83320005, -63.68131664], rel=1e-8)
    assert p["2.50%"].to_list() == pytest.approx([288.0475521, -480.2784902, -176.8319194], rel=1e-8)
    assert p["97.50%"].to_list() == pytest.approx([533.99897700, -255.38790991, 49.46928612], rel=1e-8)


def test_regress_stepwise_drops_zero_df_terms_first(dia):
    r = rsm.model.regress({"d": dia}, rvar="price", evar=EVAR, stepwise="backward")
    assert r.removed == ["cc", "carat2"]
    assert r.step_trace.startswith(
        "Start:  AIC=42673.09\nprice ~ carat + depth + carat2 + cc + clarity\n\n\n"
        "Step:  AIC=42673.09\nprice ~ carat + depth + carat2 + clarity\n\n\n"
        "Step:  AIC=42673.09\nprice ~ carat + depth + clarity\n\n"
    )
    assert "- depth    1 1.1682e+07 4.4969e+09 42679" in r.step_trace
    assert r.aliased == []
    assert _coef(r, "carat") == pytest.approx((8437.08989367, 51.04414348), rel=1e-8)


def test_regress_f_test_compares_fits_like_anova(dia, capsys):
    r = rsm.model.regress({"d": dia}, rvar="price", evar=EVAR)
    r.f_test(["clarity"])
    out = capsys.readouterr().out
    assert "R-squared, Model 1 vs 2: 0.861 vs 0.904" in out
    assert "F-statistic: 192.936 df (7, 2990), p.value < .001" in out
    # dropping depth changes no df (cc takes its place): R prints nothing
    r.f_test(["depth"])
    assert capsys.readouterr().out == ""


def test_logistic_drops_aliased_columns_like_glm(dia):
    m = rsm.model.logistic({"d": dia}, rvar="high", lev="yes", evar=["carat", "depth", "carat2", "cc", "color"])
    assert m.aliased == ["carat2", "cc"]
    assert _coef(m, "carat") == pytest.approx((18.7829805792, 0.99467593481), rel=1e-6)
    assert _coef(m, "depth") == pytest.approx((-0.1041568929, 0.06436802752), rel=1e-6)
    assert _coef(m, "Intercept") == pytest.approx((-7.2509248867, 3.91592569156), rel=1e-6)


def test_logistic_robust_hc1_like_sandwich(dia):
    # Near-separation (fitted probabilities of 0 or 1): statsmodels' GLM
    # sandwich gave NaN; R's vcovHC(type = "HC1") is finite.
    for evar in (["carat", "depth", "carat2", "cc", "color"], ["carat", "depth", "color"]):
        m = rsm.model.logistic({"d": dia}, rvar="high", lev="yes", evar=evar, robust=True)
        assert _coef(m, "Intercept")[1] == pytest.approx(4.34760271413, rel=1e-6)
        assert _coef(m, "carat")[1] == pytest.approx(0.99128127654, rel=1e-6)
        assert _coef(m, "depth")[1] == pytest.approx(0.07042834253, rel=1e-6)
        assert _coef(m, "color[E]")[1] == pytest.approx(0.35038978267, rel=1e-6)


def test_logistic_summary_notes_like_radiant(dia, capsys):
    m = rsm.model.logistic({"d": dia}, rvar="high", lev="yes", evar=["carat", "depth", "carat2", "cc", "color"])
    m.summary(vif=True, ci=True, odds=True, plain=True)
    out = capsys.readouterr().out
    assert "dropped from the estimation: carat2, cc" in out
    assert "Multicollinearity diagnostics were not calculated." in out
    assert "Confidence intervals were not calculated." in out
    assert "Odds ratios were not calculated" in out
    p = m.predict(dia.head(3))["prediction"].to_numpy()
    assert np.all(np.isfinite(p))
