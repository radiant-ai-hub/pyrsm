"""radiant.model regress() options in pyrsm: standardize, center, stepwise,
robust, RMSE/RSD, confidence level, influence and lines plots.

Reference numbers come from radiant.model 1.x on the same diamonds data, e.g.
``regress(diamonds, "price", c("carat", "clarity", "cut"),
check = c("standardize-1sd", "robust"))``.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pyrsm.model import plot_predictions
from pyrsm.model.regress import regress

DIAMONDS = Path(__file__).parents[1] / "examples" / "data" / "data" / "diamonds.parquet"


@pytest.fixture(scope="module")
def diamonds():
    return pl.read_parquet(DIAMONDS)


def test_standardize_1sd_robust_matches_radiant(diamonds):
    reg = regress(
        {"diamonds": diamonds},
        rvar="price",
        evar=["carat", "clarity", "cut"],
        standardize=1,
        robust=True,
    )
    coef = dict(zip(reg.coef["index"], reg.coef["coefficient"]))
    se = dict(zip(reg.coef["index"], reg.coef["std.error"]))
    assert coef["Intercept"] == pytest.approx(-1.1468394, abs=1e-6)
    assert coef["carat"] == pytest.approx(1.0153364, abs=1e-6)
    assert coef["clarity[SI2]"] == pytest.approx(0.6530439, abs=1e-6)
    # HC1 standard errors, as sandwich::vcovHC(type = "HC1")
    assert se["Intercept"] == pytest.approx(0.07221197, abs=1e-7)
    assert se["carat"] == pytest.approx(0.01005354, abs=1e-7)
    assert se["clarity[SI1]"] == pytest.approx(0.07066783, abs=1e-7)


def test_standardized_prediction_is_in_original_units(diamonds):
    reg = regress(
        {"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"], standardize=True
    )
    assert reg.standardize == 2
    pred = reg.predict(cmd={"carat": [1]})
    assert pred["carat"][0] == 1.0
    assert pred["prediction"][0] == pytest.approx(5265.569, abs=1e-3)


def test_centered_prediction_and_ci_match_unscaled_model(diamonds):
    plain = regress({"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"])
    centered = regress(
        {"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"], center=True
    )
    a = plain.predict(cmd={"carat": [1, 2]}, ci=True)
    b = centered.predict(cmd={"carat": [1, 2]}, ci=True)
    for col in ["prediction", "2.50%", "97.50%"]:
        np.testing.assert_allclose(a[col].to_numpy(), b[col].to_numpy(), rtol=1e-9)
    # slopes do not change when centering
    assert dict(zip(plain.coef["index"], plain.coef["coefficient"]))[
        "carat"
    ] == pytest.approx(dict(zip(centered.coef["index"], centered.coef["coefficient"]))["carat"])


def test_rmse_rsd_and_confidence_level(diamonds, capsys):
    reg = regress({"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"])
    reg.summary(rmse=True, ci=True, conf=0.90)
    out = capsys.readouterr().out
    assert "Prediction error (RMSE): 1224.329" in out
    assert "Residual st.dev   (RSD): 1226.17" in out
    assert "5%" in out and "95%" in out
    # radiant.model: carat 8353.951 8522.110 at 90%
    assert "8353.951" in out and "8522.11" in out


def test_summary_header_flags(diamonds, capsys):
    reg = regress(
        {"diamonds": diamonds},
        rvar="price",
        evar=["carat", "clarity"],
        standardize=2,
        robust=True,
    )
    reg._summary_header()
    out = capsys.readouterr().out
    assert "**Standardized coefficients shown (2 X SD)**" in out
    assert "**Robust standard errors used**" in out


def _noisy(diamonds):
    rng = np.random.default_rng(1)
    return diamonds.with_columns(
        pl.Series("noise", rng.normal(size=diamonds.height)),
        pl.Series("n2", rng.normal(size=diamonds.height)),
    )


def test_stepwise_backward_drops_noise_and_respects_marginality(diamonds, capsys):
    df = _noisy(diamonds)
    reg = regress(
        {"diamonds": df},
        rvar="price",
        evar=["carat", "clarity", "noise", "n2", "depth"],
        ivar=["carat:clarity", "noise:n2"],
        stepwise=True,
    )
    # radiant.model selects price ~ carat + clarity + depth + carat:clarity
    assert set(reg.removed) == {"noise:n2", "noise", "n2"}
    assert reg.removed[0] == "noise:n2"  # interaction leaves before its main effects
    assert reg.ivar == ["carat:clarity"]
    assert reg._model_evar() == ["carat", "clarity", "depth"]
    full = regress(
        {"diamonds": df},
        rvar="price",
        evar=["carat", "clarity", "noise", "n2", "depth"],
        ivar=["carat:clarity", "noise:n2"],
    )
    assert reg.fitted.aic < full.fitted.aic
    reg.summary(test=["depth"])
    out = capsys.readouterr().out
    assert "removed noise:n2, noise, n2" in out
    assert "Model comparisons are not conducted when Stepwise has been selected." in out


def test_f_test_accepts_a_comma_separated_string(diamonds, capsys):
    reg = regress({"diamonds": diamonds}, rvar="price", evar=["carat", "clarity", "cut"])
    reg.f_test(test="clarity, cut")
    out = capsys.readouterr().out
    assert "Model 1: price ~ carat" in out


@pytest.mark.parametrize(
    "plots,kwargs",
    [
        ("influence", {}),
        ("scatter", {"lines": ["line", "loess", "jitter"]}),
        ("residual", {"lines": ["line"]}),
        ("dashboard", {"lines": ["line", "loess"]}),
        ("dashboard", {"lines": []}),
        ("coef", {"alpha": 0.1, "intercept": True}),
        ("pdp", {}),
        ("pip", {}),
    ],
)
def test_plots_render(diamonds, plots, kwargs, tmp_path):
    reg = regress(
        {"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"], robust=True
    )
    p = reg.plot(plots, nobs=500, **kwargs)
    p.save(tmp_path / f"{plots}.png", dpi=40, verbose=False)
    assert (tmp_path / f"{plots}.png").stat().st_size > 0


def test_plot_predictions(diamonds, tmp_path):
    reg = regress({"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"])
    pred = reg.predict(cmd={"carat": [0.5, 1, 2], "clarity": ["SI1", "IF"]}, ci=True)
    for kwargs in [
        {"xvar": "carat", "color": "clarity"},
        {"xvar": "clarity"},
        {"xvar": "carat", "facet_col": "clarity"},
    ]:
        p = plot_predictions(pred, **kwargs)
        p.save(tmp_path / "pred.png", dpi=40, verbose=False)
    with pytest.raises(ValueError):
        plot_predictions(pred, xvar="cut")
