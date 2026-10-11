"""radiant.model logistic() options in pyrsm: standardize, center, stepwise,
robust, weights, VIF, profile-likelihood intervals, odds ratios, tests, plots.

Reference output comes from radiant.model on the same titanic data (see
tests/reference/logistic): noise = sin(1:n), w = 1 + (1:n %% 3).
"""

import csv
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pyrsm.model.logistic import logistic
from pyrsm.model.model import gvif

TITANIC = Path(__file__).parents[1] / "examples" / "data" / "model" / "titanic.parquet"
REF = Path(__file__).parent / "reference" / "logistic"


@pytest.fixture(scope="module")
def titanic():
    d = pl.read_parquet(TITANIC)
    i = np.arange(1, d.height + 1)
    return d.with_columns(pl.Series("noise", np.sin(i)), pl.Series("w", 1 + (i % 3)))


def _fit(titanic, **kw):
    kw.setdefault("evar", ["pclass", "sex", "age"])
    return logistic({"titanic": titanic}, rvar="survived", lev="Yes", **kw)


@pytest.mark.parametrize(
    "direction,evar,ivar,ref",
    [
        ("backward", ["pclass", "sex", "age", "noise"], ["pclass:sex"], "step_backward.txt"),
        ("both", ["pclass", "sex", "age", "noise"], None, "step_both.txt"),
    ],
)
def test_step_trace_is_r_step_output(titanic, direction, evar, ivar, ref):
    m = _fit(titanic, evar=evar, ivar=ivar, stepwise=direction)
    assert m.step_trace == (REF / ref).read_text().rstrip("\n")


def test_profile_likelihood_intervals_match_r_confint(titanic):
    m = _fit(titanic)
    assert m.ci_type == "profile"
    ci = m.conf_int().values
    with open(REF / "profile_ci.csv") as fh:
        rows = list(csv.reader(fh))[1:]
    expected = np.array([[float(r[1]), float(r[2])] for r in rows])
    # R's confint interpolates a spline through the profile; the exact root
    # agrees to about 1e-4.
    np.testing.assert_allclose(ci, expected, atol=2e-4)


def test_gvif_matches_car_vif(titanic):
    m = _fit(titanic)
    got = gvif(m.fitted, dec=None)["vif"].to_dict()
    with open(REF / "vif.csv") as fh:
        for row in csv.DictReader(fh):
            assert got[row["term"]] == pytest.approx(float(row["gvif"]), rel=1e-4)


def test_robust_hc1_and_wald_test(titanic, capsys):
    m = _fit(titanic, robust=True)
    m.summary(test=["age"])
    out = capsys.readouterr().out
    ref = (REF / "robust_test.txt").read_text()
    assert "Chi-squared: 28.138 df (1)" in out and "Chi-squared: 28.138 df(1)" in ref
    assert "**Robust standard errors used**" in out
    # standardized + robust, as radiant.model check = c("standardize-1sd", "robust")
    m = _fit(titanic, robust=True, standardize=1)
    np.testing.assert_allclose(
        m.fitted_inference.bse.values,
        [0.190562594, 0.198203770, 0.231949915, 0.162828331, 0.093403755],
        rtol=1e-6,
    )


def test_frequency_weights_match_r(titanic):
    m = _fit(titanic, weights="w")
    coef = dict(zip(m.coef["index"], m.coef["coefficient"]))
    with open(REF / "weights_coef.csv") as fh:
        for row in csv.DictReader(fh):
            label = row["label"].replace("(Intercept)", "Intercept").replace("|", "[")
            label = label + "]" if "[" in label else label
            assert coef[label] == pytest.approx(float(row["coefficient"]), rel=1e-6)
    assert "AIC: 1960.168" in (REF / "weights.txt").read_text()
    assert m.fitted.aic == pytest.approx(1960.168, abs=1e-3)


def test_sampling_weights_turn_on_robust(titanic):
    m = _fit(titanic.with_columns((pl.col("w") * 0 + 0.5 + (pl.col("w") > 1) * 0.5).alias("sw")), weights="sw")
    assert m.robust


def test_intercept_has_no_odds_ratio_and_odds_table(titanic, capsys):
    m = _fit(titanic)
    assert m.coef["OR"][0] is None
    m.summary(ci=True, odds=True, conf=0.9)
    out = capsys.readouterr().out
    assert "Odds ratios:" in out and "odds ratio" in out and "5%" in out and "95%" in out
    assert "Level                : Yes in survived" in out


def test_standardized_predictions_use_original_inputs(titanic):
    plain = _fit(titanic)
    std = _fit(titanic, standardize=2)
    a = plain.predict(cmd={"age": [20, 60]}, ci=True)
    b = std.predict(cmd={"age": [20, 60]}, ci=True)
    assert b["age"].to_list() == [20, 60]
    np.testing.assert_allclose(a["prediction"].to_numpy(), b["prediction"].to_numpy(), rtol=1e-8)


@pytest.mark.parametrize("plots", ["fit", "scatter", "influence", "or", "corr", "dist", "pdp", "pip"])
def test_plots_render(titanic, plots, tmp_path):
    m = _fit(titanic, robust=True)
    p = m.plot(plots, nobs=500)
    p.save(tmp_path / f"{plots}.png", dpi=40, verbose=False)


def test_step_steps_are_the_trace_as_data(titanic):
    clf = logistic({"titanic": titanic}, rvar="survived", lev="Yes", evar=["pclass", "sex", "age", "fare"], stepwise="backward")
    assert len(clf.step_steps) == clf.step_trace.count("AIC=")
    assert clf.step_steps[0]["columns"] == ["Df", "Deviance", "AIC"]
