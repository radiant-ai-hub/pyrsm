"""Unused levels and collinear (aliased) terms are handled as in R.

Reference values from R 4.x (lm/glm, radiant.model::regress) shown next to each
test. statsmodels on its own would keep empty levels and split the effect of
collinear columns with a pseudo-inverse.
"""

import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# import the classes through the package: `from pyrsm.model.regress import ...`
# would bind the submodule over `pyrsm.model.regress` for later tests
from pyrsm.model import logistic, regress
from pyrsm.model.model import COLLINEARITY_NOTE, drop_unused_levels, r_qr_rank

DATA = Path(__file__).resolve().parents[1] / "examples" / "data" / "model"


@pytest.fixture(scope="module")
def diamonds():
    d = pl.read_parquet(DATA / "diamonds.parquet")
    return d.with_columns(
        (2 * pl.col("carat")).alias("carat2"), (pl.col("x") + pl.col("y")).alias("x3")
    )


@pytest.fixture(scope="module")
def titanic():
    t = pl.read_parquet(DATA / "titanic.parquet")
    return t.with_columns((2 * pl.col("age")).alias("age2"))


def test_qr_pivot_matches_r(diamonds):
    # R: lm(price ~ carat + carat2 + x + y + x3)$qr -> rank 4, pivot 1 2 4 5 3 6
    X = np.column_stack(
        [np.ones(diamonds.height)]
        + [diamonds[c].to_numpy() for c in ["carat", "carat2", "x", "y", "x3"]]
    )
    rank, piv = r_qr_rank(X)
    assert rank == 4
    assert list(piv) == [0, 1, 3, 4, 2, 5]


def test_regress_drops_collinear_terms_like_lm(diamonds):
    r = regress({"d": diamonds}, rvar="price", evar=["carat", "carat2", "x", "y", "x3"])
    assert r.aliased == ["carat2", "x3"]
    coef = dict(zip(r.coef["index"], r.coef["coefficient"]))
    se = dict(zip(r.coef["index"], r.coef["std.error"]))
    assert list(coef) == ["Intercept", "carat", "x", "y"]
    np.testing.assert_allclose(
        [coef[k] for k in coef], [2715.210, 10817.230, -4408.615, 3114.522], atol=1e-3
    )
    np.testing.assert_allclose(
        [se[k] for k in se], [443.7478, 268.6346, 482.6717, 477.1530], atol=1e-4
    )
    assert (r.fitted.df_model, r.fitted.df_resid) == (3, 2996)
    # R: predict(m, d[1:3, ], interval = "confidence")
    pr = r.fitted.get_prediction(diamonds.head(3).to_pandas()).summary_frame()
    np.testing.assert_allclose(
        pr[["mean", "mean_ci_lower", "mean_ci_upper"]].to_numpy()[0],
        [506.1843, 421.7926, 590.5761],
        atol=1e-4,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        r.summary(vif=True)
    out = buf.getvalue()
    assert COLLINEARITY_NOTE in out and "carat2, x3" in out
    assert " inf " not in out  # VIFs only for the estimated terms


def test_regress_f_test_matches_anova(diamonds):
    # R: anova(lm(price ~ carat2 + clarity), lm(price ~ carat + carat2 + clarity))
    # -> Df 0 (carat2 takes carat's place); and dropping clarity: F 197.58 df (7, 2991)
    r = regress({"d": diamonds}, rvar="price", evar=["carat", "carat2", "clarity"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        r.f_test(test=["carat"])
        r.f_test(test=["clarity"])
    out = buf.getvalue()
    assert "df (0, 2991)" in out
    assert "F-statistic: 197.585 df (7, 2991), p.value < .001" in out


def test_filtered_level_is_dropped_like_r(diamonds):
    # radiant: regress(diamonds, "price", c("carat", "clarity"),
    #                  data_filter = "clarity != 'I1'") -> base level SI2
    r = regress(
        {"d": diamonds}, rvar="price", evar=["carat", "clarity"],
        data_filter="clarity != 'I1'",
    )
    assert r.aliased == []
    coef = dict(zip(r.coef["index"], r.coef["coefficient"]))
    assert "clarity[SI2]" not in coef and "clarity[I1]" not in coef
    np.testing.assert_allclose(
        [coef["Intercept"], coef["carat"], coef["clarity[SI1]"], coef["clarity[IF]"]],
        [-4041.3235, 8485.4538, 828.5593, 2503.4274],
        atol=1e-3,
    )


def test_logistic_drops_collinear_terms_like_glm(titanic):
    # R: glm(survived == "Yes" ~ pclass + sex + age + age2, binomial)
    m = logistic({"t": titanic}, rvar="survived", lev="Yes",
                 evar=["pclass", "sex", "age", "age2"])
    assert m.aliased == ["age2"]
    np.testing.assert_allclose(
        m.coef["coefficient"].to_numpy(),
        [3.51472829, -1.27437885, -2.28279541, -2.49143673, -0.03448803],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        m.coef["std.error"].to_numpy(),
        [0.326872796, 0.225574176, 0.225926876, 0.166132572, 0.006342544],
        # R's summary.glm takes the standard errors from the weights of the
        # second-to-last IRLS iteration; statsmodels evaluates them at the final
        # estimates (same ~5e-5 relative gap for any logistic model, aliased or not)
        rtol=1e-4,
    )
    assert m.fitted.llf == pytest.approx(-490.864, abs=1e-3)
    assert m.fitted.aic == pytest.approx(991.7281, abs=1e-3)
    assert m.fitted.df_model == 4
    p = m.predict(cmd={"pclass": ["1st", "3rd"]})["prediction"]
    assert p.is_finite().all()


def test_logistic_filtered_level_is_dropped(titanic):
    m = logistic({"t": titanic}, rvar="survived", lev="Yes", evar=["pclass", "sex"],
                 data_filter="pclass != '3rd'")
    assert m.coef["index"].to_list() == ["Intercept", "pclass[2nd]", "sex[male]"]


def test_drop_unused_levels_keeps_order():
    df = pl.DataFrame(
        {"g": pl.Series(["b", "c", None]).cast(pl.Enum(["a", "b", "c"])), "x": [1.0, None, 2.0]}
    )
    # only row 1 is complete for (g, x): levels "a" and "c" go, order is kept
    out = drop_unused_levels(df, ["g", "x"])
    assert out["g"].dtype == pl.Enum(["b"])
    assert out["g"].to_list() == ["b", None, None]
