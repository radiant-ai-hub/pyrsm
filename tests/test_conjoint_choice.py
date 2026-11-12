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
                         price="price", **CASES[case])
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
    cc = conjoint_choice(choc, rvar="chosen", evar=EVAR, id="choice_id", price="price")
    assert (cc.coef["p.value"] > 0).all()


def test_part_worths_and_importance(choc):
    cc = conjoint_choice(choc, rvar="chosen", evar=EVAR, id="choice_id", price="price")
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
    cc = conjoint_choice(choc, rvar="chosen", evar=EVAR, id="choice_id", price="price")
    w = cc.wtp("price")
    bp = cc.coef.filter(pl.col("index") == "price")["coefficient"][0]
    assert np.allclose(w["WTP"], w["PW"] / -bp)
    with pytest.raises(ValueError, match="numeric attribute"):
        cc.wtp("brand")


def _dummies(choc):
    """choc with each categorical level as a 0/1 column."""
    return choc.with_columns(
        (pl.col("brand").cast(pl.Utf8) == "Lindt").cast(pl.Int64).alias("brand_Lindt"),
        (pl.col("brand").cast(pl.Utf8) == "Godiva").cast(pl.Int64).alias("brand_Godiva"),
        (pl.col("chocolate").cast(pl.Utf8) == "Dark").cast(pl.Int64).alias("dark"),
        (pl.col("chocolate").cast(pl.Utf8) == "White").cast(pl.Int64).alias("white"),
        (pl.col("nuts").cast(pl.Utf8) == "Yes").cast(pl.Int64).alias("nuts_yes"),
    )


DUMMIES = ["brand_Lindt", "brand_Godiva", "dark", "white", "nuts_yes", "price"]
LEVEL_OF = {"brand_Lindt": "brand:Lindt", "brand_Godiva": "brand:Godiva",
            "dark": "chocolate:Dark", "white": "chocolate:White", "nuts_yes": "nuts:Yes"}


def _cat_wtp(choc):
    cat = conjoint_choice(choc, rvar="chosen", evar=EVAR, id="choice_id", price="price").wtp()
    return dict(zip(cat["Attributes"] + ":" + cat["Levels"], cat["WTP"]))


def test_wtp_for_numeric_attributes(choc):
    """0/1 columns kept numeric (max_levels=0): one WTP per column (the value
    of a one-unit increase, here having the level), equal to the WTP of the
    same level coded as a categorical; only the price row is left out."""
    cc = conjoint_choice(_dummies(choc), rvar="chosen", evar=DUMMIES, id="choice_id",
                         price="price", max_levels=0)
    assert cc.as_categorical == []
    w = cc.wtp()
    assert w["Attributes"].to_list() == DUMMIES[:-1]
    assert w["WTP"].is_not_null().all()
    lookup = _cat_wtp(choc)
    assert np.allclose(w["WTP"], [lookup[LEVEL_OF[v]] for v in DUMMIES[:-1]], rtol=1e-5)


def test_few_valued_numeric_attributes_are_categorical(choc):
    """By default a numeric attribute with at most 4 values is categorical: a
    part-worth for each value, the lowest (0) the base. Price stays numeric."""
    d = _dummies(choc)
    cc = conjoint_choice({"d": d}, rvar="chosen", evar=DUMMIES, id="choice_id", price="price")
    assert cc.as_categorical == DUMMIES[:-1]
    assert cc.xnames == [f"{v}[1]" for v in DUMMIES[:-1]] + ["price"]
    assert cc.levels["dark"] == ["0", "1"]
    # the same model as the slopes: a 0/1 column's slope is its level-1 part-worth
    slopes = conjoint_choice(d, rvar="chosen", evar=DUMMIES, id="choice_id", price="price",
                             max_levels=0)
    assert np.allclose(cc.coef["coefficient"], slopes.coef["coefficient"], atol=1e-10)
    assert np.isclose(cc.fit_stats["loglik"], slopes.fit_stats["loglik"])
    w = cc.wtp()
    ones = w.filter(pl.col("Levels") == "1")
    lookup = _cat_wtp(choc)
    assert np.allclose(ones["WTP"], [lookup[LEVEL_OF[v]] for v in DUMMIES[:-1]], rtol=1e-5)
    assert (w.filter(pl.col("Levels") == "0")["WTP"] == 0).all()
    buf = io.StringIO()
    with redirect_stdout(buf):
        cc.summary()
    assert "Coded as categorical : brand_Lindt, brand_Godiva, dark, white, nuts_yes" in buf.getvalue()


def test_price_with_few_values_is_categorical_unless_named(choc):
    """choc's price has 3 values: without price= it gets a part-worth per
    value (lowest first), and WTP asks for price= instead of dividing by a
    slope that is not there."""
    cc = conjoint_choice(choc, rvar="chosen", evar=EVAR, id="choice_id")
    assert cc.as_categorical == ["price"]
    assert cc.levels["price"] == ["2.99", "3.99", "4.99"]
    assert "price[3.99]" in cc.xnames and "price" not in cc.xnames
    with pytest.raises(ValueError, match="price='price'"):
        cc.wtp()
    with pytest.raises(ValueError, match="numeric attribute in evar"):
        conjoint_choice(choc, rvar="chosen", evar=EVAR, id="choice_id", price="brand")
    with pytest.raises(ValueError, match="numeric attribute in evar"):
        conjoint_choice(choc, rvar="chosen", evar=EVAR, id="choice_id", price="cost")
    # more values than max_levels: one slope
    sl = conjoint_choice(choc, rvar="chosen", evar=EVAR, id="choice_id", max_levels=2)
    assert sl.as_categorical == [] and "price" in sl.xnames


def test_predict_with_numeric_levels(choc):
    """A level coded as a number is the same level as 1, 1.0 or "1"; a value
    the design did not test is refused, as for any categorical level."""
    d = _dummies(choc)
    cc = conjoint_choice(d, rvar="chosen", evar=DUMMIES, id="choice_id", price="price")
    a = cc.predict(cmd={"dark": [0, 1], "price": [3.99]})
    b = cc.predict(cmd={"dark": ["0", 1.0], "price": [3.99]})
    assert np.allclose(a["prediction"], b["prediction"])
    assert a["dark"].to_list() == ["0", "1"]
    profiles = pl.DataFrame({"brand_Lindt": [1, 0], "brand_Godiva": [0, 1], "dark": [1, 0],
                             "white": [0, 0], "nuts_yes": [0, 1], "price": [3.99, 4.99]})
    p = cc.predict(profiles)
    q = cc.predict(profiles.with_columns(pl.col("dark").cast(pl.Float64)))
    assert np.allclose(p["utility"], q["utility"])
    params = dict(zip(cc.xnames, cc._fit["params"]))
    u0 = params["brand_Lindt[1]"] + params["dark[1]"] + 3.99 * params["price"]
    assert np.isclose(p["utility"][0], u0)
    p2 = cc.predict(profiles, data_cmd={"dark": 0})
    assert p2["dark"].to_list() == ["0", "0"]
    with pytest.raises(ValueError, match="Unknown level"):
        cc.predict(cmd={"dark": [2]})
    with pytest.raises(ValueError, match="Unknown level"):
        cc.predict(profiles.with_columns(pl.lit(2).alias("white")))


def test_part_worth_plot_includes_numeric_attributes(choc):
    """Every attribute gets a part-worth panel, so the default plot is never
    empty: a numeric one shows the utility of each value relative to its
    lowest (slope x difference), the same range its importance weight uses."""
    cc = conjoint_choice(_dummies(choc), rvar="chosen", evar=DUMMIES, id="choice_id",
                         price="price", max_levels=0)
    assert cc.levels == {}
    assert len(cc.plot("pw")) == len(DUMMIES)
    assert len(cc.plot("pw", scale_plot=True)) == len(DUMMIES)
    curve = cc._numeric_pw("price")
    bp = float(cc.coef.filter(pl.col("index") == "price")["coefficient"][0])
    assert curve["value"].tolist() == [2.99, 3.99, 4.99]
    assert np.allclose(curve["PW"], [0, bp, 2 * bp])
    total = sum(abs(float(c)) * (1 if v != "price" else 2)
                for v, c in zip(cc.xnames, cc.coef["coefficient"]))
    iw = cc.IW.filter(pl.col("Attributes") == "price")["IW"][0]
    assert iw == pytest.approx(abs(bp) * 2 / total)
    # choc: one panel per attribute, price included
    mixed = conjoint_choice(choc, rvar="chosen", evar=EVAR, id="choice_id", price="price")
    assert len(mixed.plot("pw")) == len(EVAR)


def test_predict_market_simulation(choc):
    cc = conjoint_choice(choc, rvar="chosen", evar=EVAR, id="choice_id", price="price")
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
    a = conjoint_choice(d, rvar="choice", lev="yes", evar=EVAR, id="choice_id", price="price")
    b = conjoint_choice(d, rvar="chosen", evar=EVAR, id="choice_id", price="price")
    assert np.allclose(a.coef["coefficient"], b.coef["coefficient"])
    with pytest.raises(ValueError, match="Please specify the level"):
        conjoint_choice(d, rvar="choice", evar=EVAR, id="choice_id", price="price")


def test_errors(choc):
    with pytest.raises(ValueError, match="exactly one chosen"):
        conjoint_choice(choc.with_columns(pl.lit(1).alias("chosen")), rvar="chosen",
                        evar=EVAR, id="choice_id", price="price")
    with pytest.raises(ValueError, match="not in evar"):
        conjoint_choice(choc, rvar="chosen", evar=EVAR, id="choice_id", ivar=["brand:size"])
    with pytest.raises(ValueError, match="Response variable contained"):
        conjoint_choice(choc, rvar="chosen", evar=EVAR + ["chosen"], id="choice_id")
    cc = conjoint_choice(choc, rvar="chosen", evar=EVAR, id="choice_id", price="price")
    with pytest.raises(ValueError, match="Unknown level"):
        cc.predict(cmd={"brand": ["Mars"]})
    with pytest.raises(ValueError, match="Unknown plot"):
        cc.plot("dashboard")


def test_summary_and_plots(choc):
    cc = conjoint_choice({"choc": choc}, rvar="chosen", evar=EVAR, id="choice_id",
                         alt="alt", price="price", data_filter="resp <= 150")
    buf = io.StringIO()
    with redirect_stdout(buf):
        cc.summary()
    out = buf.getvalue()
    for s in ("Conjoint choice model", "Data                 : choc", "Filter",
              "brand[Lindt]", "Hit rate", "Conjoint part-worths", "importance weights"):
        assert s in out
    assert len(cc.plot("pw")) == 4  # one panel per attribute, price included
    assert cc.plot("iw") is not None and cc.plot("or") is not None


def test_store_and_plot_options(choc):
    cc = conjoint_choice({"choc": choc}, rvar="chosen", evar=EVAR, id="choice_id", price="price")
    assert cc.store("PW").columns == ["Attributes", "Levels", "PW"]
    assert cc.store("IW").height == len(EVAR)
    assert "WTP" in cc.store("WTP").columns
    assert cc.store("coef").height == 6
    with pytest.raises(ValueError):
        cc.store("bogus")
    assert len(cc.plot("pw", scale_plot=True)) == 4
    assert cc.plot(["iw", "or"]) is not None


def test_minivan_example_matches_clogit_and_converges():
    """The minivan example (price in $1,000s) converges and matches R's clogit."""
    mv = pl.read_parquet(ROOT / "examples" / "data" / "multivariate" / "minivan.parquet")
    cc = conjoint_choice({"minivan": mv}, rvar="choice", evar=["seat", "cargo", "eng", "price"],
                         id="task", alt="alt", price="price")
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


def test_alternative_id_can_also_be_an_attribute():
    """Brand as the alternative id *and* an attribute gives brand constants
    (ketchup in long format: one row per brand per purchase)."""
    k = pl.read_parquet(
        Path(__file__).resolve().parents[1] / "examples" / "data" / "model" / "ketchup.parquet"
    ).with_row_index("occasion", offset=1)
    brands = ["heinz28", "heinz32", "heinz41", "hunts32"]
    long = pl.concat(
        [
            k.select(
                "occasion",
                "choice",
                pl.lit(b).alias("brand"),
                pl.col(f"price.{b}").alias("price"),
            )
            for b in brands
        ]
    ).with_columns(
        pl.col("brand").cast(pl.Enum(brands)),
        (pl.col("choice").cast(pl.Utf8) == pl.col("brand").cast(pl.Utf8)).cast(pl.Int8).alias("chosen"),
    )
    cc = conjoint_choice(
        long, rvar="chosen", lev="1", evar=["brand", "price"], id="occasion", alt="brand"
    )
    assert cc.xnames == ["brand[heinz32]", "brand[heinz41]", "brand[hunts32]", "price"]
    assert cc.converged
