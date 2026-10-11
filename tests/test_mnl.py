"""pyrsm.model.mnl against radiant.model::mnl (nnet::multinom).

Fixtures: scripts/generate_mnl_reference.R (radiant.model 1.6.12, nnet
7.3-21). ``nnet::multinom`` stops its optimizer at a relative tolerance of
1e-8 on the deviance, so its estimates sit slightly short of the maximum: the
log-likelihood here is never lower than R's, and estimates agree within a
small fraction of their standard error.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import pyrsm as rsm
from pyrsm._logit import clogit_fit

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "reference" / "radiant_model"
DATA = ROOT / "examples" / "data" / "model"

CASES = sorted(p.stem.removeprefix("mnl_") for p in REF.glob("mnl_*.json"))
PRICES = ["price.heinz28", "price.heinz32", "price.heinz41", "price.hunts32"]


def load(name: str) -> pl.DataFrame:
    if name == "titanic_agg":
        d = pl.read_csv(REF / "titanic_agg_mnl.csv")
        return d.with_columns(
            pl.col("pclass").cast(pl.Enum(["1st", "2nd", "3rd"])),
            pl.col("sex").cast(pl.Enum(["female", "male"])),
            pl.col("survived").cast(pl.Enum(["Yes", "No"])),
        )
    return pl.read_parquet(DATA / f"{name}.parquet")


def ref_case(case: str) -> dict:
    return json.loads((REF / f"mnl_{case}.json").read_text())


def fit_case(case: str):
    a = ref_case(case)["args"]
    check = a.get("check", "")
    return rsm.model.mnl(
        {a["dataset"]: load(a["dataset"])},
        rvar=a["rvar"],
        evar=a["evar"],
        lev=a.get("lev"),
        ivar=a.get("int") or None,
        weights=a.get("wts"),
        data_filter=a.get("data_filter", ""),
        standardize="standardize" in check,
        center="center" in check,
        intercept="no_int" not in check,
        stepwise="backward" if "stepwise" in check else False,
    )


def r_label(level: str, label: str) -> str:
    """R's 'heinz32 disp.heinz28|yes' as pyrsm's 'heinz32 disp.heinz28[yes]'."""
    label = "Intercept" if label == "(Intercept)" else label
    label = re.sub(r"\|([^:]+)", r"[\1]", label)
    return f"{level} {label}"


def printed(fn) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


@pytest.fixture(scope="module", params=CASES)
def case(request):
    return request.param, ref_case(request.param), fit_case(request.param)


def _tol(case: str) -> float:
    # R's stepwise fits stop furthest from the optimum
    return 0.12 if case == "ketchup_stepwise" else 0.03


def test_same_terms_as_r(case):
    name, ref, m = case
    rc = ref["coeff"]
    want = [r_label(lv, lab) for lv, lab in zip(rc["level"], rc["label"])]
    got = [f"{lv} {ix}" for lv, ix in zip(m.coef["level"], m.coef["index"])]
    assert got == want


def test_coefficients_match_r(case):
    name, ref, m = case
    rc = ref["coeff"]
    diff = np.abs(np.array(rc["coefficient"]) - m.coef["coefficient"].to_numpy())
    assert np.all(diff <= _tol(name) * np.array(rc["std.error"]) + 1e-6), name


def test_standard_errors_match_r(case):
    name, ref, m = case
    rtol = 2e-2 if name == "ketchup_stepwise" else 1e-3
    np.testing.assert_allclose(m.coef["std.error"].to_numpy(), ref["coeff"]["std.error"], rtol=rtol)


def test_loglik_at_least_r_and_close(case):
    """The engine reaches the maximum; nnet stops just short of it."""
    name, ref, m = case
    ll = m.fit_stats["loglik"]
    assert ll >= ref["loglik"] - 1e-6
    assert ll - ref["loglik"] < 1e-2


def test_fit_statistics_match_r(case):
    name, ref, m = case
    fs = m.fit_stats
    k = ref["edf"]
    nlev = len(m.levels)
    assert fs["nparams"] == k
    assert fs["nobs"] == pytest.approx(
        sum(load("titanic_agg")["n"]) if name == "titanic_agg_wts" else ref["nobs"]
    )
    assert fs["AIC"] == pytest.approx(ref["AIC"], abs=0.02)
    # radiant prints the chi-squared statistic against the intercept-only model
    lr = ref["null_deviance"] - ref["deviance"]
    assert fs["chisq"] == pytest.approx(lr, abs=0.02)
    # ...with df = edf - 1; the intercept-only model has nlev - 1 parameters
    assert fs["df"] == (k - (nlev - 1) if m.intercept else k)
    r2 = re.search(r"Pseudo R-squared: ([0-9.]+)", ref["summary"]).group(1)
    assert round(fs["pseudo_rsq"], 3) == pytest.approx(float(r2), abs=1e-3)
    bic = re.search(r"BIC: ([0-9.]+)", ref["summary"]).group(1)
    assert fs["BIC"] == pytest.approx(float(bic), abs=0.02)


def test_fitted_probabilities_match_r(case):
    name, ref, m = case
    atol = 2e-3 if name == "ketchup_stepwise" else 2e-4
    pred = m.predict().head(25)
    for lv, vals in ref["fitted"].items():
        np.testing.assert_allclose(pred[lv].to_numpy(), vals, atol=atol, err_msg=lv)


def test_predict_dataset_matches_r(case):
    name, ref, m = case
    atol = 2e-3 if name == "ketchup_stepwise" else 2e-4
    pd_ = ref["pred_data"]
    pred = m.predict(load(ref["args"]["dataset"])).head(25)
    for lv in m.levels:
        np.testing.assert_allclose(pred[lv].to_numpy(), pd_[lv], atol=atol, err_msg=lv)


def test_confusion_matrix_matches_r(case):
    name, ref, m = case
    if "confusion" not in ref["sum_check"]:
        pytest.skip("no confusion matrix in this fixture")
    if name == "titanic_agg_wts":
        # R counts the aggregated rows and ignores the weights; with frequency
        # weights each row stands for n people, as in the unaggregated data
        ref = ref_case("titanic_unagg")
    block = ref["summary"].split("Confusion matrix:")[1]
    rows = {}
    for line in block.splitlines()[3 : 3 + len(m.levels)]:
        parts = line.split()
        rows[parts[0]] = [float(x) for x in parts[1:]]
    tab = m.confusion()
    for lv, pred_row in zip(tab["predicted"], tab.select(m.levels).rows()):
        assert list(pred_row) == rows[lv], lv
    err = re.search(r"Misclassification error: ([0-9.]+)%", ref["summary"]).group(1)
    out = printed(lambda: m.summary(main=False, fit=False, confusion=True))
    assert f"Misclassification error: {err}%" in out


def test_variables_test_matches_r(case):
    name, ref, m = case
    if not ref["test_var"]:
        pytest.skip("no variables test in this fixture")
    res = m.lr_test(ref["test_var"])
    chi, df = re.search(r"Model 1.*?Chi-squared: ([0-9.]+) df\((\d+)\)", ref["summary"], re.S).groups()
    assert res["chisq"] == pytest.approx(float(chi), abs=0.02)
    assert res["df"] == int(df)


def test_predict_command_matches_r(case):
    name, ref, m = case
    if "pred_cmd" not in ref:
        pytest.skip("no prediction command in this fixture")
    cmds = {
        "ketchup_price": {"price.heinz28": [round(3.4 + 0.1 * i, 1) for i in range(19)]},
        "ketchup_all": {"disp.heinz41": ["yes", "no"], "feat.heinz41": ["yes", "no"]},
        "ketchup_std": {"price.heinz28": [4, 5]},
    }
    pred = m.predict(cmd=cmds[name])
    want = ref["pred_cmd_result"]
    keys = list(cmds[name])
    got = {tuple(str(r[k]) for k in keys): r for r in pred.iter_rows(named=True)}
    for i in range(len(want[m.levels[0]])):
        key = tuple(
            str(want[k][i]) if not isinstance(want[k][i], float) else str(float(want[k][i]))
            for k in keys
        )
        row = got[tuple(str(float(x)) if k.startswith("price") else x for k, x in zip(keys, key))]
        for lv in m.levels:
            assert row[lv] == pytest.approx(want[lv][i], abs=2e-4), (key, lv)


def test_stepwise_selects_r_terms():
    ref = ref_case("ketchup_stepwise")
    m = fit_case("ketchup_stepwise")
    r_terms = {re.sub(r"\|.*", "", lab) for lab in ref["coeff"]["label"]} - {"(Intercept)"}
    assert set(m.terms) == r_terms
    assert set(m.removed) == set(ref["args"]["evar"]) - r_terms
    out = printed(m.summary)
    assert out.startswith("Start:  AIC=")
    assert "Backward stepwise selection of variables" in out


def test_frequency_weights_equal_the_unaggregated_data():
    agg, full = fit_case("titanic_agg_wts"), fit_case("titanic_unagg")
    np.testing.assert_allclose(agg.coef["coefficient"], full.coef["coefficient"], atol=1e-7)
    np.testing.assert_allclose(agg.coef["std.error"], full.coef["std.error"], rtol=1e-6)
    assert agg.fit_stats["loglik"] == pytest.approx(full.fit_stats["loglik"], abs=1e-6)


def test_mnl_is_a_conditional_logit_on_a_long_design():
    """The shared engine's two likelihoods agree: one row per response level,
    level-specific intercepts and variables interacted with level indicators."""
    m = rsm.model.mnl({"ketchup": load("ketchup")}, rvar="choice", lev="heinz28", evar=PRICES)
    X, _ = m._design(m.data, m.terms)
    n, p = X.shape
    K = len(m.levels)
    rows, y, codes = [], [], []
    for k in range(K):
        block = np.zeros((n, p * (K - 1)))
        if k > 0:
            block[:, (k - 1) * p : k * p] = X
        rows.append(block)
        y.append((m._y == k).astype(float))
        codes.append(np.arange(n))
    order = np.argsort(np.concatenate(codes), kind="stable")
    Xl = np.vstack(rows)[order]
    yl = np.concatenate(y)[order]
    cl = np.concatenate(codes)[order]
    cf = clogit_fit(Xl, yl, cl, n)
    np.testing.assert_allclose(cf["params"], m.coef["coefficient"].to_numpy(), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(cf["bse"], m.coef["std.error"].to_numpy(), rtol=1e-5)
    assert cf["llf"] == pytest.approx(m.fit_stats["loglik"], abs=1e-8)


def test_probabilities_do_not_depend_on_the_base_level():
    k = load("ketchup")
    a = rsm.model.mnl({"k": k}, rvar="choice", lev="heinz28", evar=PRICES).predict(k)
    b = rsm.model.mnl({"k": k}, rvar="choice", lev="hunts32", evar=PRICES).predict(k)
    for lv in ["heinz28", "heinz32", "heinz41", "hunts32"]:
        np.testing.assert_allclose(a[lv], b[lv], atol=1e-7)


def test_predict_data_cmd_and_scaled_inputs_in_original_units():
    k = load("ketchup")
    m = rsm.model.mnl({"k": k}, rvar="choice", lev="heinz28", evar=PRICES, standardize=True)
    plain = rsm.model.mnl({"k": k}, rvar="choice", lev="heinz28", evar=PRICES)
    a = m.predict(k.head(5), data_cmd={"price.heinz28": 4.0})
    b = plain.predict(k.head(5), data_cmd={"price.heinz28": 4.0})
    assert a["price.heinz28"].to_list() == [4.0] * 5
    np.testing.assert_allclose(a["heinz32"], b["heinz32"], atol=1e-7)


def test_summary_prints_header_scope_and_tables():
    m = rsm.model.mnl(
        {"ketchup": load("ketchup")}, rvar="choice", lev="heinz28", evar=PRICES,
        data_filter="price.heinz28 > 4",
    )
    out = printed(lambda: m.summary(ci=True, rrr=True, confusion=True, test=["price.heinz28"]))
    for text in [
        "Multinomial logistic regression (MNL)",
        "Filter               : price.heinz28 > 4",
        "Base level           : heinz28 in choice",
        "Relative risk ratios:",
        "Confidence intervals:",
        "Model 1: choice ~ price.heinz32 + price.heinz41 + price.hunts32",
        "Misclassification error:",
        "Nr obs: 1,995",
    ]:
        assert text in out, text


@pytest.mark.parametrize("plots", ["coef", "pred", "dist", "corr"])
def test_plots_render(plots):
    import matplotlib

    matplotlib.use("Agg")
    m = rsm.model.mnl(
        {"ketchup": load("ketchup")}, rvar="choice", lev="heinz28",
        evar=["price.heinz28", "price.heinz32", "disp.heinz28"],
    )
    assert m.plot(plots) is not None


def test_errors_are_clear():
    k = load("ketchup")
    with pytest.raises(ValueError, match="not a level"):
        rsm.model.mnl(k, rvar="choice", lev="heinz99", evar=PRICES)
    with pytest.raises(ValueError, match="Response variable contained"):
        rsm.model.mnl(k, rvar="choice", evar=["choice"])
    with pytest.raises(ValueError, match="not in evar"):
        rsm.model.mnl(k, rvar="choice", evar=PRICES, ivar=["price.heinz28:disp.heinz28"])


def test_predict_dec_rounds_every_numeric_column():
    m = rsm.model.mnl({"k": load("ketchup")}, rvar="choice", lev="heinz28", evar=PRICES)
    pred = m.predict(cmd={"price.heinz28": [3.4, 5.2]}, dec=3)
    assert pred["price.heinz32"].to_list() == [3.143, 3.143]
    assert pred["heinz32"].to_list() == [0.393, 0.728]
