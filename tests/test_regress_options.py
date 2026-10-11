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
    assert "removed: noise:n2, noise, n2" in out
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


REFERENCE = Path(__file__).parent / "reference" / "regress_step"


def _deterministic_noise(diamonds):
    # Same columns in R: noise <- sin(1:n); n2 <- cos(3 * (1:n))
    i = np.arange(1, diamonds.height + 1)
    return diamonds.with_columns(pl.Series("noise", np.sin(i)), pl.Series("n2", np.cos(3 * i)))


@pytest.mark.parametrize(
    "direction,evar,ivar",
    [
        ("backward", ["carat", "clarity", "noise", "n2", "depth"], ["carat:clarity", "noise:n2"]),
        ("forward", ["carat", "noise", "depth"], ["carat:depth"]),
        ("both", ["carat", "clarity", "noise", "depth"], None),
    ],
)
def test_step_trace_is_r_step_output(diamonds, direction, evar, ivar):
    """The trace is what R prints for regress(..., check = "stepwise-<direction>")."""
    reg = regress(
        {"diamonds": _deterministic_noise(diamonds)},
        rvar="price",
        evar=evar,
        ivar=ivar,
        stepwise=direction,
    )
    expected = (REFERENCE / f"{direction}.txt").read_text().rstrip("\n")
    assert reg.step_trace == expected


def test_stepwise_summary_banner_and_selected_model(diamonds, capsys):
    reg = regress(
        {"diamonds": _deterministic_noise(diamonds)},
        rvar="price",
        evar=["carat", "clarity", "noise", "depth"],
        stepwise="both",
    )
    reg.summary()
    out = capsys.readouterr().out
    banner = "-" * 52 + "\nForward and Backward stepwise selection of variables\n" + "-" * 52
    assert banner in out
    # radiant.model keeps carat, clarity and depth (see both_summary.txt)
    r_summary = (REFERENCE / "both_summary.txt").read_text()
    for term in ["carat", "clarity|IF", "depth"]:
        assert term in r_summary
    assert set(reg.formula.split("~")[1].replace(" ", "").split("+")) == {"carat", "clarity", "depth"}


def test_robust_keeps_classical_fit_statistics(diamonds):
    """As radiant.model: robust SEs for inference, classical R-squared and F."""
    plain = regress({"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"])
    robust = regress({"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"], robust=True)
    assert robust.fitted.fvalue == pytest.approx(plain.fitted.fvalue)
    assert robust.fitted.rsquared == pytest.approx(plain.fitted.rsquared)
    assert robust.coef["std.error"][1] != pytest.approx(plain.coef["std.error"][1])
    # prediction intervals are the classical ones (R's predict.lm)
    a = plain.predict(cmd={"carat": [1]}, ci=True)
    b = robust.predict(cmd={"carat": [1]}, ci=True)
    assert a["2.50%"][0] == pytest.approx(b["2.50%"][0])


@pytest.mark.parametrize("model", ["regress", "logistic", "rforest", "mlp", "xgboost"])
def test_prediction_with_missing_columns_says_which(model):
    from pyrsm.model.logistic import logistic
    from pyrsm.model.mlp import mlp
    from pyrsm.model.rforest import rforest
    from pyrsm.model.xgboost import xgboost

    rng = np.random.default_rng(1)
    df = pl.DataFrame(
        {
            "y": rng.integers(0, 2, 200).astype(float),
            "x1": rng.normal(size=200),
            "x2": rng.normal(size=200),
        }
    )
    if model == "regress":
        m = regress(df, rvar="y", evar=["x1", "x2"])
    elif model == "logistic":
        m = logistic(df.with_columns(pl.col("y").cast(pl.Int64).cast(pl.Utf8)), rvar="y", lev="1", evar=["x1", "x2"])
    elif model == "rforest":
        m = rforest(df, rvar="y", evar=["x1", "x2"], mod_type="regression")
    elif model == "mlp":
        m = mlp(df, rvar="y", evar=["x1", "x2"], mod_type="regression", max_iter=50)
    else:
        m = xgboost(df, rvar="y", evar=["x1", "x2"], mod_type="regression")
    with pytest.raises(ValueError) as err:
        m.predict(data=df.drop("x2"))
    assert str(err.value) == (
        "All variables in the model must also be in the prediction data\n"
        "Variables in the model: x1, x2\n"
        "Variables not available in prediction data: x2"
    )
    with pytest.raises(ValueError) as err:
        m.predict(cmd={"x3": [1]})
    assert "The command entered contains variable names that are not in the model" in str(err.value)
    assert str(err.value).endswith("Variables not available in prediction data: x3")


QTERMS = Path(__file__).parent / "reference" / "regress_qterms"


def test_squared_and_cubed_terms_match_radiant(diamonds):
    """radiant.model's qterms: I(carat^2), I(carat^3) as interaction terms."""
    import csv

    reg = regress(
        {"diamonds": diamonds},
        rvar="price",
        evar=["carat", "clarity"],
        ivar=["I(carat^2)", "I(carat^3)"],
    )
    coef = dict(zip(reg.coef["index"], reg.coef["coefficient"]))
    se = dict(zip(reg.coef["index"], reg.coef["std.error"]))
    with open(QTERMS / "coef.csv") as fh:
        for row in csv.DictReader(fh):
            label = row["label"].replace("(Intercept)", "Intercept").replace("|", "[")
            if "[" in label:
                label += "]"
            assert coef[label] == pytest.approx(float(row["coefficient"]), rel=1e-8), label
            assert se[label] == pytest.approx(float(row["std.error"]), rel=1e-8), label


def test_squared_terms_in_the_step_trace_read_as_r(diamonds):
    reg = regress(
        {"diamonds": _deterministic_noise(diamonds)},
        rvar="price",
        evar=["carat", "noise", "depth"],
        ivar=["I(carat^2)", "I(noise^2)", "carat:depth"],
        stepwise=True,
    )
    assert reg.step_trace == (QTERMS / "step.txt").read_text().rstrip("\n")


def test_f_test_on_a_squared_term(diamonds, capsys):
    reg = regress(
        {"diamonds": diamonds},
        rvar="price",
        evar=["carat", "clarity"],
        ivar=["I(carat^2)"],
    )
    reg.f_test(test=["I(carat^2)"])
    out = capsys.readouterr().out
    assert "F-statistic: 75.954 df (1, 2990)" in out
    assert "R-squared, Model 1 vs 2: 0.904 vs 0.907" in out


def test_squared_term_predictions_use_the_base_variable(diamonds):
    reg = regress(
        {"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"], ivar=["I(carat^2)"]
    )
    pred = reg.predict(cmd={"carat": [1, 2]})
    assert pred.columns[:2] == ["carat", "clarity"]
    assert pred["prediction"][1] > pred["prediction"][0]


def test_logistic_squared_term_and_chisq_test(capsys):
    from pyrsm.model.logistic import logistic

    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    y = (x + x**2 + rng.normal(size=400) > 1).astype(int)
    df = pl.DataFrame({"y": y.astype(str), "x": x})
    clf = logistic(df, rvar="y", lev="1", evar=["x"], ivar=["I(x^2)"])
    assert "I(x^2)" in clf.coef["index"].to_list()
    clf.chisq_test(test=["I(x^2)"])
    assert "Model 2: y ~ x + I(x^2)" in capsys.readouterr().out


def test_step_steps_are_the_trace_as_data(diamonds):
    """``step_steps`` holds what ``step_trace`` prints, as numbers (Radiant's tables)."""
    reg = regress({"diamonds": diamonds}, rvar="price", evar=["carat", "clarity", "cut", "x"], stepwise="backward")
    steps = reg.step_steps
    assert len(steps) == reg.step_trace.count("AIC=")
    assert [s["label"] for s in steps] == ["Start"] + ["Step"] * (len(steps) - 1)
    first = steps[0]
    assert first["columns"] == ["Df", "Sum of Sq", "RSS", "AIC"]
    assert first["rows"][0].keys() == {"term", "Df", "Sum of Sq", "RSS", "AIC"}
    # rows by AIC, the <none> row carries no Df / Sum of Sq
    aics = [r["AIC"] for r in first["rows"]]
    assert aics == sorted(aics)
    none = next(r for r in first["rows"] if r["term"] == "<none>")
    assert none["Df"] is None and none["AIC"] == pytest.approx(first["aic"])
    assert f"AIC={first['aic']:.2f}" in reg.step_trace.splitlines()[0]
    # every step after the first starts from the change chosen before it
    for prev, step in zip(steps, steps[1:]):
        assert step["aic"] == pytest.approx(prev["rows"][0]["AIC"])
