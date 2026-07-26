"""Tests for pyrsm.multivariate.conjoint against radiant.multivariate."""

import numpy as np
import polars as pl
import pytest

from pyrsm.multivariate import conjoint

COEF_TOL = 1e-8
PW_TOL = 1e-6  # R stores PW/IW rounded to 3 decimals


@pytest.mark.parametrize(
    "case,dataset,rvar,evar,reverse",
    [
        ("mp3", "mp3", "Rating", "Memory:Shape", False),
        ("carpet", "carpet", "ranking", "design:money_back", True),
        ("movie", "movie", "Ranking", "price:food", True),
    ],
)
def test_conjoint_parity(mv_data, mv_ref, case, dataset, rvar, evar, reverse):
    ref = mv_ref(f"conjoint_{case}")
    cj = conjoint({dataset: mv_data(dataset)}, rvar=rvar, evar=evar, reverse=reverse)

    assert cj.evar == ref["evar"]

    # part-worths: compare to R's 3-decimal stored table
    assert cj.PW["Levels"].to_list() == ref["PW"]["Levels"]
    pw = np.round(cj.PW["PW"].to_numpy(), 3)
    assert np.abs(pw - np.array(ref["PW"]["PW"])).max() < PW_TOL

    iw = np.round(cj.IW["IW"].to_numpy(), 3)
    assert np.abs(iw - np.array(ref["IW"]["IW"])).max() < PW_TOL

    # regression coefficients: full precision
    assert cj.coeff["label"].to_list() == ref["coeff"]["label"]
    assert (
        np.abs(cj.coeff["coefficient"].to_numpy() - np.array(ref["coeff"]["coefficient"])).max()
        < COEF_TOL
    )
    assert (
        np.abs(cj.coeff["std.error"].to_numpy() - np.array(ref["coeff"]["std.error"])).max()
        < COEF_TOL
    )
    assert abs(cj.model_list["full"]["rsq"] - ref["rsq"]) < COEF_TOL


def test_conjoint_base_level_is_first_enum(mv_data):
    cj = conjoint({"mp3": mv_data("mp3")}, rvar="Rating", evar="Memory:Shape")
    # base level (first Enum level) has part-worth 0
    pw = cj.PW.filter(pl.col("Attributes") == "Memory")
    assert pw["Levels"].to_list() == ["4GB", "6GB", "8GB"]
    assert pw["PW"].to_list()[0] == 0.0


def test_conjoint_predict(mv_data):
    df = mv_data("mp3")
    cj = conjoint({"mp3": df}, rvar="Rating", evar="Memory:Shape")
    pred = cj.predict(df)
    assert "Prediction" in pred.columns
    assert pred.height == df.height


def test_conjoint_predict_intervals(mv_data):
    df = mv_data("mp3")
    cj = conjoint({"mp3": df}, rvar="Rating", evar="Memory:Shape")
    conf = cj.predict(df, se=True, interval="confidence")
    pred = cj.predict(df, se=True, interval="prediction")
    assert {"Prediction", "+/-"}.issubset(conf.columns)
    # prediction intervals are wider than confidence intervals
    assert (pred["+/-"].to_numpy() >= conf["+/-"].to_numpy() - 1e-9).all()
    assert pred["+/-"].to_numpy().mean() > conf["+/-"].to_numpy().mean()


def test_conjoint_predict_expand_grid(mv_data):
    cj = conjoint({"mp3": mv_data("mp3")}, rvar="Rating", evar="Memory:Shape")
    grid = cj.predict(
        pred_cmd={
            "Memory": ["4GB", "8GB"],
            "Radio": "Yes",
            "Size": "Small",
            "Price": "$50",
            "Shape": "Square",
        }
    )
    assert grid.height == 2
    # 8GB should outscore 4GB (positive Memory part-worths)
    assert grid["Prediction"][1] > grid["Prediction"][0]


def test_conjoint_interaction_parity(mv_data, mv_ref):
    ref = mv_ref("conjoint_mp3_int")
    cj = conjoint(
        {"mp3": mv_data("mp3")}, rvar="Rating", evar="Memory:Shape", int="Memory:Shape"
    )
    rco = dict(zip(ref["coeff"]["label"], ref["coeff"]["coefficient"]))
    pco = dict(zip(cj.coeff["label"].to_list(), cj.coeff["coefficient"].to_list()))
    assert set(rco) == set(pco)
    assert max(abs(rco[k] - pco[k]) for k in rco) < 1e-6


def test_conjoint_by_group_parity(mv_data, mv_ref):
    ref = mv_ref("conjoint_mp3_by_radio")
    cj = conjoint(
        {"mp3": mv_data("mp3")}, rvar="Rating", evar=["Memory", "Size", "Price"], by="Radio"
    )
    assert cj.bylevs == ref["bylevs"]
    norm = lambda s: s.replace("|", "")  # noqa: E731  (R uses Memory6GB, py Memory|6GB)
    for lev in ref["bylevs"]:
        g = ref["groups"][lev]
        rco = {norm(k): v for k, v in zip(g["coeff_label"], g["coeff"])}
        m = cj.model_list[lev]
        pco = {
            norm(k): v
            for k, v in zip(m["coeff"]["label"].to_list(), m["coeff"]["coefficient"].to_list())
        }
        assert set(rco) == set(pco)
        assert max(abs(rco[k] - pco[k]) for k in rco) < 1e-6


def test_conjoint_store_predictions(mv_data):
    from pyrsm.multivariate import store_predictions

    df = mv_data("mp3")
    cj = conjoint({"mp3": df}, rvar="Rating", evar="Memory:Shape")
    out = store_predictions(df, cj.predict(df, se=True), name="pref")
    assert {"pref", "pref_lb", "pref_ub", "pref_me"}.issubset(out.columns)
    assert out.height == df.height


def test_conjoint_scale_plot(mv_data):
    cj = conjoint({"mp3": mv_data("mp3")}, rvar="Rating", evar="Memory:Shape")
    plots = cj.plot(plots="pw", scale_plot=True)
    assert isinstance(plots, list) and len(plots) == 5  # one per attribute


def test_conjoint_plot_pw_and_iw_list(mv_data):
    cj = conjoint({"mp3": mv_data("mp3")}, rvar="Rating", evar="Memory:Shape")
    out = cj.plot(plots=["pw", "iw"])
    # 5 part-worth panels + 1 importance-weight plot
    assert isinstance(out, list) and len(out) == 6
    assert cj.plot(plots="iw") is not None
    with pytest.raises(ValueError):
        cj.plot(plots=["bogus"])


def test_conjoint_pred_cmd_string(mv_data):
    cj = conjoint({"mp3": mv_data("mp3")}, rvar="Rating", evar="Memory:Shape")
    grid = cj.predict(
        pred_cmd="Memory = c('4GB','8GB'); Radio = Yes; Size = Small; Price = $50; Shape = Square"
    )
    # equivalent to the dict expand-grid form
    grid2 = cj.predict(
        pred_cmd={"Memory": ["4GB", "8GB"], "Radio": "Yes", "Size": "Small",
                  "Price": "$50", "Shape": "Square"}
    )
    assert grid.height == 2
    assert np.allclose(grid["Prediction"].to_numpy(), grid2["Prediction"].to_numpy())


def test_conjoint_store(mv_data):
    cj = conjoint({"mp3": mv_data("mp3")}, rvar="Rating", evar="Memory:Shape")
    pw = cj.store("PW")
    iw = cj.store("IW")
    assert "PW" in pw.columns
    assert "IW" in iw.columns


def test_conjoint_reverse_effect(mv_data):
    df = mv_data("mp3")
    base = conjoint({"mp3": df}, rvar="Rating", evar="Memory:Shape")
    rev = conjoint({"mp3": df}, rvar="Rating", evar="Memory:Shape", reverse=True)
    # reversing the response flips the sign of every coefficient
    b = base.coeff.filter(pl.col("label") == "Memory|8GB")["coefficient"][0]
    r = rev.coeff.filter(pl.col("label") == "Memory|8GB")["coefficient"][0]
    assert abs(b + r) < 1e-8


def test_conjoint_summary_plot_smoke(mv_data, capsys):
    cj = conjoint({"mp3": mv_data("mp3")}, rvar="Rating", evar="Memory:Shape")
    cj.summary()
    out = capsys.readouterr().out
    assert "Conjoint analysis" in out
    assert "part-worths" in out
    assert "importance weights" in out
    assert cj.plot(plots="pw") is not None
    assert cj.plot(plots="iw") is not None
