"""Every tool's ``summary(plain=False)`` shows Radiant's formatted report.

In a notebook (and so in a rendered Quarto report) each summary below displays
the report Radiant's Summary tab renders for that tool -- built from pyrsm's
printed summary (:func:`pyrsm.report.text_report`) or, for crtree, the random
forest, XGBoost, neural network and Model > Evaluate, from their own builders in
:mod:`pyrsm.model.report` -- with the plain summary as its text form.
"""

from pathlib import Path

import polars as pl
import pytest

import pyrsm as rsm
import pyrsm.report as report
from pyrsm.decide.simulate import repeat_simulate, simulate
from pyrsm.model.evaluate import confusion, evalbin, evalreg, uplift
from pyrsm.model.tune import tune

DATA = Path(__file__).resolve().parent.parent / "examples" / "data"
EVAL = Path(__file__).resolve().parent / "data" / "evaluate"


def _data(folder: str, name: str) -> pl.DataFrame:
    return pl.read_parquet(DATA / folder / f"{name}.parquet")


class _Kernel:
    """What ``du.is_notebook`` looks for: an IPython kernel app."""

    config = {"IPKernelApp": {}}


@pytest.fixture
def shown(monkeypatch):
    """Pretend to be a notebook and collect what would be displayed."""
    import IPython
    import IPython.display

    out = []
    monkeypatch.setattr(IPython, "get_ipython", lambda: _Kernel())
    monkeypatch.setattr(IPython.display, "display", lambda *objs, **kw: out.extend(objs))
    return out


MOVIE = """
name: Sign contract
variables:
    legal fees: 5000
type: decision
Sign with Movie Company:
    cost: legal fees
    type: chance
    Small Box Office:
        p: 0.3
        payoff: 200000
    Large Box Office:
        p: 0.7
        payoff: 1000000
Sign with TV Network:
    payoff: 900000
"""

MIX = {
    "meta": {"problem_name": "Production Mix", "objective_sense": "maximize"},
    "variables": {
        "product_a": {"label": "Units of A", "type": "continuous", "min": 0},
        "product_b": {"label": "Units of B", "type": "continuous", "min": 0},
    },
    "objective": {"product_a": 40.0, "product_b": 30.0},
    "constraints": [
        {"name": "labor", "coefficients": {"product_a": 2.0, "product_b": 1.0}, "sense": "L", "rhs": 400},
        {"name": "material", "coefficients": {"product_a": 1.0, "product_b": 2.0}, "sense": "L", "rhs": 500},
    ],
}


def _titanic():
    return {"titanic": _data("data", "titanic")}


def _rf():
    return rsm.model.rforest(_titanic(), rvar="survived", lev="Yes", evar=["pclass", "sex", "age"], n_estimators=50)


def _dvd():
    return {"dvd": pl.read_csv(EVAL / "dvd_preds.csv")}


#: name -> (build, summary kwargs, text the report must show)
CASES = {
    "conjoint": (lambda: rsm.multivariate.conjoint({"mp3": _data("multivariate", "mp3")}, rvar="Rating", evar="Memory:Shape"),
                 {}, ["Conjoint"]),
    "cbc": (lambda: rsm.multivariate.cbc({"choc": _data("multivariate", "choc")}, rvar="chosen",
                                         evar=["brand", "chocolate", "price", "nuts"], id="choice_id", alt="alt"),
            {}, ["Choice-based conjoint"]),
    "full_factor": (lambda: rsm.multivariate.full_factor({"shopping": _data("multivariate", "shopping")}, "v1:v6", nr_fact=2),
                    {}, ["shopping"]),
    "hclus": (lambda: rsm.multivariate.hclus({"shopping": _data("multivariate", "shopping")}, "v1:v6"), {}, ["shopping"]),
    "kclus": (lambda: rsm.multivariate.kclus({"shopping": _data("multivariate", "shopping")}, "v1:v6", nr_clus=3),
              {}, ["shopping"]),
    "mds": (lambda: rsm.multivariate.mds({"city": _data("multivariate", "city")}, "from", "to", "distance"), {}, ["city"]),
    "pre_factor": (lambda: rsm.multivariate.pre_factor({"shopping": _data("multivariate", "shopping")}, "v1:v6"),
                   {}, ["shopping"]),
    "prmap": (lambda: rsm.multivariate.prmap({"computer": _data("multivariate", "computer")}, "brand", "high_end:business"),
              {}, ["computer"]),
    "doe": (lambda: rsm.design.doe({"price": ["$10", "$13", "$16"], "food": ["popcorn", "gourmet", "no food"]}),
            {}, ["popcorn"]),
    "randomizer": (lambda: rsm.design.randomizer({"rndnames": _data("design", "rndnames")}, vars=["Names"],
                                                 conditions=["A", "B"], seed=1234), {}, ["rndnames"]),
    "sample_size": (lambda: rsm.design.sample_size("mean", err_mean=2, sd_mean=10), {}, ["Sample size"]),
    "sample_size_comp": (lambda: rsm.design.sample_size_comp("proportion", p1=0.008, p2=0.01, conf=0.95, power=0.9,
                                                             alt_hyp="less"), {}, ["Sample size"]),
    "sampling": (lambda: rsm.design.sampling({"rndnames": _data("design", "rndnames")}, vars=["Names"], sample_size=10,
                                             seed=1234), {}, ["rndnames"]),
    "dtree": (lambda: rsm.decide.dtree(MOVIE, opt="max"), {}, ["Sign contract"]),
    "optimize": (lambda: rsm.decide.optimize(MIX), {}, ["Production Mix"]),
    "robustness": (lambda: rsm.decide.optimize(MIX).robustness(trials=50, coef_noise=0.2, seed=1), {}, ["Robustness"]),
    "simulate": (lambda: simulate(rsm.decide.simulate_example("profit")["spec"]), {}, ["profit"]),
    "repeat": (lambda: repeat_simulate(rsm.decide.simulate_example("profit")["spec"], reps=10, vars=["E", "price"],
                                       sum_vars=["profit"], by="sim", fun="sum"), {}, ["profit"]),
    "tune": (lambda: tune(_rf(), {"max_features": [1, 2]}, method="grid", folds=2, n_jobs=1), {}, ["max_features"]),
    "crtree": (lambda: rsm.model.crtree(_titanic(), rvar="survived", lev="Yes", evar=["pclass", "sex", "age"]),
               {}, ["Tree", "All observations", "P(survived = Yes)"]),
    "rforest": (_rf, {}, ["Model fit", "AUC (OOB)", "Out-of-bag"]),
    "xgboost": (lambda: rsm.model.xgboost(_titanic(), rvar="survived", lev="Yes", evar=["pclass", "sex", "age"],
                                          n_estimators=20), {}, ["Model fit", "AUC"]),
    "mlp": (lambda: rsm.model.mlp(_titanic(), rvar="survived", lev="Yes", evar=["pclass", "sex", "age"]),
            {}, ["Model fit", "AUC"]),
    "evalreg": (lambda: evalreg({"diamonds": pl.read_csv(EVAL / "diamonds_preds.csv")}, "pred_lm", "price",
                                          train="All"), {}, ["Regression performance"]),
    "evalbin": (lambda: evalbin(_dvd(), ["pred_logit", "pred_rnd"], "buy", lev="yes", qnt=10, cost=1,
                                          margin=6), {}, ["Model performance by quantile", "cum_lift"]),
    "confusion": (lambda: confusion(_dvd(), ["pred_logit", "pred_rnd"], "buy", lev="yes", cost=1, margin=6),
                  {}, ["Confusion matrix and metrics", "kappa"]),
    "regress": (lambda: rsm.model.regress({"diamonds": _data("data", "diamonds")}, rvar="price", evar=["carat", "clarity"]),
                {"vif": True, "ci": True, "ssq": True}, ["Coefficient Estimates", "Variance inflation factors", "Sum of squares",
                                                         "Confidence intervals"]),
    "logistic": (lambda: rsm.model.logistic(_titanic(), rvar="survived", lev="Yes", evar=["pclass", "sex", "age"]),
                 {"odds": True}, ["Coefficient Estimates", "OR%", "Odds ratios"]),
    "mnl": (lambda: rsm.model.mnl(_titanic(), rvar="pclass", lev="1st", evar=["fare", "sex"]),
            {"confusion": True}, ["Coefficient Estimates", "RRR", "Confusion matrix"]),
    "conjoint_choice": (lambda: rsm.multivariate.conjoint_choice({"choc": _data("multivariate", "choc")}, rvar="chosen",
                                                                 evar=["brand", "chocolate", "price", "nuts"], id="choice_id",
                                                                 alt="alt"), {}, ["Conjoint part-worths", "4,800"]),
    "uplift": (lambda: uplift(_dvd(), ["pred_logit", "pred_rnd"], "buy", tvar="treat", lev="yes", tlev="yes",
                                        qnt=10, cost=1, margin=6), {}, ["Uplift by quantile"]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_summary_plain_false_shows_the_formatted_report(name, shown, capsys):
    build, kwargs, expected = CASES[name]
    model = build()
    capsys.readouterr()
    model.summary(plain=False, **kwargs)
    assert capsys.readouterr().out == "", "the formatted report replaces the printed text"
    assert len(shown) == 1
    html = shown[0]._repr_html_()
    assert html.startswith('<section class="pyrsm-report"') and "<style>" in html
    assert shown[0]._repr_markdown_().strip()
    for text in expected:
        assert text in html, f"{name}: {text!r} missing"
    # the text form is pyrsm's plain summary
    assert repr(shown[0]) == report.capture(model.summary, plain=True, **kwargs)


@pytest.mark.parametrize("name", ["sampling", "crtree", "evalreg"])
def test_outside_a_notebook_plain_false_prints_the_text(name, capsys):
    build, kwargs, _ = CASES[name]
    model = build()
    capsys.readouterr()
    model.summary(plain=False, **kwargs)
    assert capsys.readouterr().out.strip() == report.capture(model.summary, plain=True, **kwargs).strip()


def test_a_tool_without_a_builder_gets_the_text_report(shown):
    pf = CASES["pre_factor"][0]()
    pf.summary(plain=False)
    expected = report.text_report("pre_factor", report.capture(pf.summary, plain=True))
    assert shown[0]._repr_html_() == report.Report(expected)._repr_html_()


def test_rforest_fit_report_says_how_the_fit_was_measured():
    from pyrsm.model.report import fit_metrics, fit_report

    rf = _rf()
    metrics = fit_metrics(rf)
    assert metrics["auc_kind"] == "oob" and 0.5 < metrics["auc"] < 1
    out = fit_report("rforest", report.capture(rf.summary, plain=True), metrics)
    assert out["tables"][0]["title"] == "Model fit"
    assert "AUC" not in [m["label"] for m in out["meta"]]
