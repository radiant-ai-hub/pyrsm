"""pyrsm.model.evaluate against radiant.model (R) reference results.

The fixtures in tests/data/evaluate were produced by make_fixtures.R with
radiant.model 1.6.12: stored predictions on dvd (binary) and diamonds
(numeric), a training flag for the train/test split, and R's evalbin,
confusion, uplift and evalreg tables for those inputs.
"""

from pathlib import Path

import numpy as np
import plotnine
import polars as pl
import pytest

from pyrsm.model.evaluate import confusion, evalbin, evalreg, uplift, xtile

DATA = Path(__file__).parent / "data" / "evaluate"


@pytest.fixture(scope="module")
def dvd():
    return pl.read_csv(DATA / "dvd_preds.csv")


@pytest.fixture(scope="module")
def diamonds():
    return pl.read_csv(DATA / "diamonds_preds.csv")


def _matches_r(py: pl.DataFrame, name: str, key: list[str], cols: list[str]) -> None:
    r = pl.read_csv(DATA / f"r_{name}.csv")
    as_str = [pl.col(c).cast(pl.Utf8) for c in key]
    joined = py.with_columns(as_str).join(r.with_columns(as_str), on=key, suffix="_r")
    assert joined.height == r.height == py.height
    for c in cols:
        np.testing.assert_allclose(
            joined[c].cast(pl.Float64).to_numpy(),
            joined[f"{c}_r"].cast(pl.Float64).to_numpy(),
            rtol=1e-6, atol=1e-6, err_msg=f"{name}.{c}",
        )


def test_evalbin_matches_radiant(dvd):
    ev = evalbin({"dvd": dvd}, ["pred_logit", "pred_rnd"], "buy", lev="yes", qnt=10,
                 cost=1, margin=6, train="Both", data_filter="training == 1")
    _matches_r(ev.dataset, "evalbin", ["pred", "bins"], [
        "nr_obs", "nr_resp", "resp_rate", "gains", "profit", "ROME", "cum_prop",
        "cum_resp", "cum_resp_rate", "cum_lift", "cum_gains",
    ])


def test_confusion_matches_radiant(dvd):
    cnf = confusion({"dvd": dvd}, ["pred_logit", "pred_rnd"], "buy", lev="yes", cost=1,
                    margin=6, train="Both", data_filter="training == 1")
    _matches_r(cnf.dataset, "confusion", ["Type", "Predictor"], [
        "TP", "FP", "TN", "FN", "total", "TPR", "TNR", "precision", "Fscore", "RIG",
        "accuracy", "kappa", "profit", "index", "ROME", "contact", "AUC",
    ])


def test_uplift_matches_radiant(dvd):
    up = uplift({"dvd": dvd}, ["pred_logit", "pred_rnd"], "buy", tvar="treat", lev="yes",
                tlev="yes", qnt=10, cost=1, margin=6)
    _matches_r(up.dataset, "uplift", ["pred", "bins"], [
        "cum_prop", "T_resp", "T_n", "C_resp", "C_n", "incremental_resp",
        "incremental_profit", "inc_uplift", "uplift",
    ])


def test_evalreg_matches_radiant(diamonds):
    ev = evalreg({"diamonds": diamonds}, ["pred_lm", "pred_rnd"], "price", train="Both",
                 data_filter="training == 1")
    _matches_r(ev.dat, "evalreg", ["Type", "Predictor"], ["n", "Rsq", "RMSE", "MAE"])


def test_test_rows_are_the_complement_of_the_scope(diamonds):
    ev = evalreg({"diamonds": diamonds}, "pred_lm", "price", train="Both",
                 data_filter="training == 1")
    n = dict(zip(ev.dat["Type"], ev.dat["n"]))
    assert n["Training"] + n["Test"] == diamonds.height
    assert n["Training"] == diamonds["training"].sum()


def test_all_ignores_the_scope_like_radiant(diamonds):
    ev = evalreg({"diamonds": diamonds}, "pred_lm", "price", data_filter="training == 1")
    assert ev.dat["n"].to_list() == [diamonds.height]


@pytest.mark.parametrize("train", ["Training", "Test", "Both"])
def test_train_split_needs_a_filter_or_slice(diamonds, train):
    with pytest.raises(ValueError, match="filter or a slice"):
        evalreg(diamonds, "pred_lm", "price", train=train)


def test_slice_defines_training_rows(diamonds):
    ev = evalreg(diamonds, "pred_lm", "price", train="Both", slice="1:500")
    assert dict(zip(ev.dat["Type"], ev.dat["n"])) == {"Training": 500, "Test": 1000}


def test_errors_are_clear(dvd):
    with pytest.raises(ValueError, match="not found"):
        evalbin(dvd, "pred_logit", "buy", lev="maybe")
    with pytest.raises(ValueError, match="numeric"):
        evalbin(dvd, "treat", "buy")
    with pytest.raises(ValueError, match="Unknown plot"):
        evalbin(dvd, "pred_logit", "buy").plot("roc")
    with pytest.raises(ValueError, match="not found"):
        evalbin(dvd, "nope", "buy")


def test_default_level_is_the_first_sorted_level(dvd):
    assert evalbin(dvd, "pred_logit", "buy").lev == "no"


def test_xtile_bins_highest_scores_first_when_reversed():
    x = np.arange(1, 101)
    bins = xtile(x, 10, rev=True)
    assert bins[-1] == 1 and bins[0] == 10
    assert np.bincount(bins)[1:].tolist() == [10] * 10


def _ggplots(result):
    items = result if isinstance(result, list) else [result]
    assert items
    for g in items:
        assert isinstance(g, plotnine.ggplot)
        assert isinstance(g.theme, plotnine.themes.theme_bw)
    return items


@pytest.mark.parametrize("kind", ["lift", "gains", "profit", "expected_profit", "rome"])
@pytest.mark.parametrize("preds", [["pred_logit"], ["pred_logit", "pred_rnd"]])
def test_evalbin_plots_are_pyrsm_plotnine(dvd, kind, preds):
    g, = _ggplots(evalbin(dvd, preds, "buy", lev="yes", margin=6).plot(kind))
    g.draw()


def test_evalbin_plots_return_one_per_request(dvd):
    assert len(_ggplots(evalbin(dvd, "pred_logit", "buy").plot(["lift", "gains", "rome"]))) == 3


@pytest.mark.parametrize("kind", ["inc_uplift", "uplift", "inc_profit"])
def test_uplift_plots_are_pyrsm_plotnine(dvd, kind):
    g, = _ggplots(uplift(dvd, ["pred_logit", "pred_rnd"], "buy", tvar="treat", lev="yes",
                         tlev="yes").plot(kind))
    g.draw()


def test_metric_plots_are_pyrsm_plotnine(dvd, diamonds):
    for g in (
        evalreg(diamonds, ["pred_lm", "pred_rnd"], "price", train="Both", slice="1:500").plot(),
        confusion(dvd, ["pred_logit"], "buy", lev="yes").plot(),
    ):
        _ggplots(g)[0].draw()


def test_summaries_print(dvd, diamonds, capsys):
    evalbin(dvd, "pred_logit", "buy", lev="yes").summary()
    confusion(dvd, "pred_logit", "buy", lev="yes").summary()
    uplift(dvd, "pred_logit", "buy", tvar="treat", lev="yes", tlev="yes").summary()
    evalreg(diamonds, "pred_lm", "price").summary()
    out = capsys.readouterr().out
    for header in ("Evaluate predictions for binary response models", "Confusion matrix",
                   "Evaluate uplift for binary response models",
                   "Evaluate predictions for regression models"):
        assert header in out


def test_reachable_from_rsm_model():
    import pyrsm as rsm

    assert rsm.model.evaluate.evalbin is evalbin


def test_boolean_response(dvd):
    d = dvd.with_columns(bought=pl.col("buy") == "yes")
    ev = evalbin(d, "pred_logit", "bought", lev="true", margin=6)
    ref = evalbin(dvd, "pred_logit", "buy", lev="yes", margin=6)
    assert ev.lev == "true"
    np.testing.assert_allclose(ev.dataset["cum_gains"].to_numpy(), ref.dataset["cum_gains"].to_numpy())


def test_a_test_filter_picks_the_test_rows(diamonds):
    """Training rows from the filter, test rows from their own filter."""
    ev = evalreg(diamonds, pred=["pred_lm"], rvar="price", train="Both",
                 data_filter="training == 1", test_filter="training == 0 and price > 2000")
    assert ev.dat_list["Training"].height == (diamonds["training"] == 1).sum()
    assert ev.dat_list["Test"].height == ((diamonds["training"] == 0) & (diamonds["price"] > 2000)).sum()
    assert "Test filter : training == 0 and price > 2000" in _out(ev)


def test_test_rows_need_only_a_test_filter(diamonds):
    ev = evalreg(diamonds, pred=["pred_lm"], rvar="price", train="Test", test_filter="training == 0")
    assert ev.dat_list["Test"].height == (diamonds["training"] == 0).sum()
    with pytest.raises(ValueError, match="filter or a slice is required"):
        evalreg(diamonds, pred=["pred_lm"], rvar="price", train="Training", test_filter="training == 0")


def _out(ev) -> str:
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.summary()
    return buf.getvalue()
