"""Tests for pyrsm.multivariate.conjoint_choice against R's survival::clogit.

Fixtures: scripts/generate_conjoint_choice_reference.R
"""

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pyrsm.multivariate import conjoint_choice

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "reference" / "radiant_multivariate"
EVAR = ["brand", "chocolate", "price", "nuts"]


@pytest.fixture(scope="module")
def choc():
    return pl.read_parquet(ROOT / "examples" / "data" / "multivariate" / "choc.parquet")


def ref(name):
    return json.loads((REF / f"conjoint_choice_{name}.json").read_text())


def r_name(label: str) -> str:
    """pyrsm 'brand[Lindt]:price' -> R 'brandLindt:price'."""
    return label.replace("[", "").replace("]", "")


CASES = {
    "choc": dict(),
    "choc_int": dict(ivar=["brand:price"]),
    "choc_scope": dict(data_filter="resp <= 120", sort="-choice_id", slice="1:1500"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_matches_clogit(choc, case):
    r = ref(case)
    cc = conjoint_choice({"choc": choc}, rvar="chosen", evar=EVAR, id="choice_id",
                         **CASES[case])
    assert cc.nobs == r["nobs"] and cc.ntasks == r["ntasks"]
    assert [r_name(n) for n in cc.coef["index"]] == r["names"]
    assert np.allclose(cc.coef["coefficient"], r["coef"], atol=1e-6)
    assert np.allclose(cc.coef["std.error"], r["se"], rtol=1e-5)
    assert np.allclose(cc.coef["z.value"], r["z"], rtol=1e-5)
    assert np.allclose(cc.coef["p.value"], r["p"], rtol=1e-4, atol=1e-300)
    fs = cc.fit_stats
    assert np.allclose([fs["loglik_null"], fs["loglik"]], r["loglik"], atol=1e-6)
    assert np.isclose(fs["AIC"], r["AIC"], atol=1e-5)
    assert np.isclose(fs["BIC"], r["BIC"], atol=1e-5)
    pred = cc.predict()
    assert np.allclose(pred["prediction"], r["prob"], atol=1e-6)


def test_tiny_p_values_are_not_rounded_to_zero(choc):
    cc = conjoint_choice(choc, rvar="chosen", evar=EVAR, id="choice_id")
    assert (cc.coef["p.value"] > 0).all()


def test_part_worths_and_importance(choc):
    cc = conjoint_choice(choc, rvar="chosen", evar=EVAR, id="choice_id")
    pw = cc.PW
    assert pw.filter(pl.col("Levels") == "Hershey")["PW"][0] == 0
    lindt = cc.coef.filter(pl.col("index") == "brand[Lindt]")["coefficient"][0]
    assert pw.filter(pl.col("Levels") == "Lindt")["PW"][0] == pytest.approx(lindt)
    assert cc.IW["IW"].sum() == pytest.approx(1)
    # numeric attribute: |slope| x range of the attribute in the data
    bp = cc.coef.filter(pl.col("index") == "price")["coefficient"][0]
    ranges = {"brand": 1.203111, "chocolate": 0.493869 + 0.319996,
              "price": abs(bp) * 2, "nuts": 0.177863}
    total = sum(ranges.values())
    assert cc.IW.filter(pl.col("Attributes") == "price")["IW"][0] == pytest.approx(
        ranges["price"] / total, rel=1e-4
    )


def test_wtp(choc):
    cc = conjoint_choice(choc, rvar="chosen", evar=EVAR, id="choice_id")
    w = cc.wtp("price")
    bp = cc.coef.filter(pl.col("index") == "price")["coefficient"][0]
    assert np.allclose(w["WTP"], w["PW"] / -bp)
    with pytest.raises(ValueError, match="numeric attribute"):
        cc.wtp("brand")


def test_predict_market_simulation(choc):
    cc = conjoint_choice(choc, rvar="chosen", evar=EVAR, id="choice_id")
    sim = cc.predict(cmd={"brand": ["Hershey", "Lindt", "Godiva"]})
    assert sim.height == 3 and sim["prediction"].sum() == pytest.approx(1)
    # shares follow the brand part-worths
    assert sim["prediction"].to_list() == sorted(sim["prediction"].to_list())
    # profiles without an id form one choice set
    profiles = pl.DataFrame({
        "brand": ["Lindt", "Godiva"], "chocolate": ["Dark", "Milk"],
        "price": [3.99, 4.99], "nuts": ["No", "Yes"],
    })
    p = cc.predict(profiles)
    u = p["utility"].to_numpy()
    assert np.allclose(p["prediction"], np.exp(u) / np.exp(u).sum())
    # data_cmd changes an attribute for every profile
    p2 = cc.predict(profiles, data_cmd={"price": 2.99})
    assert (p2["price"] == 2.99).all()


def test_categorical_response_with_lev(choc):
    d = choc.with_columns(
        pl.when(pl.col("chosen") == 1).then(pl.lit("yes")).otherwise(pl.lit("no"))
        .cast(pl.Enum(["no", "yes"])).alias("choice")
    )
    a = conjoint_choice(d, rvar="choice", lev="yes", evar=EVAR, id="choice_id")
    b = conjoint_choice(d, rvar="chosen", evar=EVAR, id="choice_id")
    assert np.allclose(a.coef["coefficient"], b.coef["coefficient"])
    with pytest.raises(ValueError, match="Please specify the level"):
        conjoint_choice(d, rvar="choice", evar=EVAR, id="choice_id")


def test_errors(choc):
    with pytest.raises(ValueError, match="exactly one chosen"):
        conjoint_choice(choc.with_columns(pl.lit(1).alias("chosen")), rvar="chosen",
                        evar=EVAR, id="choice_id")
    with pytest.raises(ValueError, match="not in evar"):
        conjoint_choice(choc, rvar="chosen", evar=EVAR, id="choice_id", ivar=["brand:size"])
    with pytest.raises(ValueError, match="Response variable contained"):
        conjoint_choice(choc, rvar="chosen", evar=EVAR + ["chosen"], id="choice_id")
    cc = conjoint_choice(choc, rvar="chosen", evar=EVAR, id="choice_id")
    with pytest.raises(ValueError, match="Unknown level"):
        cc.predict(cmd={"brand": ["Mars"]})
    with pytest.raises(ValueError, match="Unknown plot"):
        cc.plot("dashboard")


def test_summary_and_plots(choc):
    cc = conjoint_choice({"choc": choc}, rvar="chosen", evar=EVAR, id="choice_id",
                         alt="alt", data_filter="resp <= 150")
    buf = io.StringIO()
    with redirect_stdout(buf):
        cc.summary()
    out = buf.getvalue()
    for s in ("Conjoint choice model", "Data                 : choc", "Filter",
              "brand[Lindt]", "Hit rate", "Conjoint part-worths", "importance weights"):
        assert s in out
    assert len(cc.plot("pw")) == 3  # one panel per categorical attribute
    assert cc.plot("iw") is not None and cc.plot("or") is not None


def test_store_and_plot_options(choc):
    cc = conjoint_choice({"choc": choc}, rvar="chosen", evar=EVAR, id="choice_id")
    assert cc.store("PW").columns == ["Attributes", "Levels", "PW"]
    assert cc.store("IW").height == len(EVAR)
    assert "WTP" in cc.store("WTP").columns
    assert cc.store("coef").height == 6
    with pytest.raises(ValueError):
        cc.store("bogus")
    assert len(cc.plot("pw", scale_plot=True)) == 3
    assert cc.plot(["iw", "or"]) is not None


def test_minivan_example_matches_clogit_and_converges():
    """The minivan example (price in $1,000s) converges and matches R's clogit."""
    mv = pl.read_parquet(ROOT / "examples" / "data" / "multivariate" / "minivan.parquet")
    cc = conjoint_choice({"minivan": mv}, rvar="choice", evar=["seat", "cargo", "eng", "price"],
                         id="task", alt="alt")
    assert cc.converged
    # survival::clogit(choice ~ seat + cargo + eng + price + strata(task))
    r_coef = [-0.570436, -0.445111, 0.420322, -0.780185, -1.585133, -0.153654]
    r_se = [0.061263, 0.060817, 0.050411, 0.059465, 0.068199, 0.006732]
    assert np.allclose(cc.coef["coefficient"], r_coef, atol=1e-6)
    assert np.allclose(cc.coef["std.error"], r_se, atol=1e-6)
    assert np.isclose(cc.fit_stats["loglik"], -2650.827, atol=1e-3)
    market = pl.read_parquet(ROOT / "examples" / "data" / "multivariate" / "minivan_market.parquet")
    shares = cc.predict(data=market)["prediction"]
    assert np.isclose(shares.sum(), 1.0)
