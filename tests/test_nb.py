"""Tests for pyrsm.model.nb against radiant.model::nb (e1071::naiveBayes).

Reference fixtures are produced by ``scripts/generate_nb_reference.R``.
"""

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pyrsm.model.nb import _dnorm, nb

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "reference" / "radiant_model"
DATA = ROOT / "examples" / "data" / "model"
TOL = 1e-10

CASES = sorted(
    p.stem.removeprefix("nb_") for p in REF.glob("nb_*.json")
)

WIDE_LEVELS = [
    f"fare_bracket_{s}"
    for s in ("lowest", "low", "middle_low", "middle_high", "high", "highest")
]


def load(name: str) -> pl.DataFrame:
    if name == "titanic_wide":
        d = pl.read_csv(
            REF / "nb_titanic_wide.csv",
            schema_overrides={"age": pl.Float64, "fare": pl.Float64, "age_yes": pl.Float64},
        )
        return d.with_columns(
            pl.col("survived").cast(pl.Enum(["Yes", "No"])),
            pl.col("pclass").cast(pl.Enum(["1st", "2nd", "3rd"])),
            pl.col("sex").cast(pl.Enum(["female", "male"])),
            pl.col("fare_bracket").cast(pl.Enum(WIDE_LEVELS)),
            pl.col("age_yes").cast(pl.Float64),
        )
    return pl.read_parquet(DATA / f"{name}.parquet")


def ref_case(case: str) -> dict:
    return json.loads((REF / f"nb_{case}.json").read_text())


def as_list(x):
    return x if isinstance(x, list) else [x]


def r_args_to_kwargs(a: dict) -> dict:
    """Translate the R call into pyrsm arguments."""
    kw = dict(rvar=a["rvar"], evar=as_list(a["evar"]))
    for k in ("laplace", "data_filter"):
        if k in a:
            kw[k] = a[k]
    if "arr" in a:  # R's arr = "desc(price)" is pyrsm's sort = "-price"
        kw["sort"] = a["arr"].replace("desc(", "-").replace(")", "")
    if "rows" in a:  # R's rows = 1:500 is pyrsm's slice = "1:500"
        kw["slice"] = f"{a['rows'][0]}:{a['rows'][-1]}"
    return kw


def r_cmd_to_dict(cmd, model: nb) -> dict:
    """The pred_cmd strings used in the fixtures, as pyrsm ``cmd`` dicts."""
    out = {}
    for c in as_list(cmd):
        var, expr = (s.strip() for s in c.split("=", 1))
        if expr.startswith("levels("):
            out[var] = model.xlevels[var]
        elif expr.startswith("seq("):
            lo, hi, by = (float(x) for x in expr[4:-1].split(","))
            out[var] = list(np.round(np.arange(lo, hi + by / 2, by), 10))
        elif ":" in expr:
            lo, hi = (int(x) for x in expr.split(":"))
            out[var] = list(range(lo, hi + 1))
        else:
            raise ValueError(c)
    return out


_FITS: dict = {}


def fit_case(case: str):
    if case not in _FITS:
        ref = ref_case(case)
        a = ref["args"]
        _FITS[case] = (nb({a["dataset"]: load(a["dataset"])}, **r_args_to_kwargs(a)), ref)
    return _FITS[case]


def capture(fun, *args, **kwargs) -> list[str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fun(*args, **kwargs)
    return buf.getvalue().splitlines()


SCOPE_LABELS = ("Filter ", "Arrange ", "Sort ", "Slice ")


# -- parity with Radiant -----------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_levels_and_nobs(case):
    m, ref = fit_case(case)
    assert m.levels == as_list(ref["lev"])
    assert m.evar == as_list(ref["evar"])
    assert m.nobs == ref["nobs"]


@pytest.mark.parametrize("case", CASES)
def test_apriori_matches_r(case):
    m, ref = fit_case(case)
    assert m.apriori[m.rvar].to_list() == as_list(ref["apriori"]["names"])
    np.testing.assert_array_equal(m.apriori["count"].to_numpy(), ref["apriori"]["values"])


@pytest.mark.parametrize("case", CASES)
def test_tables_match_r(case):
    m, ref = fit_case(case)
    assert [t["var"] for t in ref["tables"]] == m.evar
    for t in ref["tables"]:
        v = t["var"]
        assert m.isnumeric[v] == t["numeric"]
        want = np.array(
            [np.nan if x is None else x for x in t["values"]], dtype=float
        ).reshape(t["nrow"], t["ncol"])
        got = m.tables[v]
        assert got[m.rvar].to_list() == as_list(t["rownames"])
        if not t["numeric"]:
            assert got.columns[1:] == as_list(t["colnames"])
        np.testing.assert_allclose(
            got.drop(m.rvar).to_numpy(), want, rtol=TOL, atol=TOL, equal_nan=True
        )


@pytest.mark.parametrize("case", CASES)
def test_predictions_match_r(case):
    m, ref = fit_case(case)
    pred = m.predict(data=load(ref["args"]["dataset"]))
    want = ref["prediction"]
    assert pred.height == len(want[m.levels[0]])
    assert pred.columns == m.evar + m.levels
    for lv in m.levels:
        np.testing.assert_allclose(pred[lv].to_numpy(), want[lv], rtol=TOL, atol=TOL)
    rowsum = pred.select(m.levels).sum_horizontal().to_numpy()
    np.testing.assert_allclose(rowsum, 1.0, atol=1e-12)


@pytest.mark.parametrize(
    "case", [c for c in CASES if "pred_cmd" in ref_case(c)]
)
def test_pred_cmd_matches_r(case):
    m, ref = fit_case(case)
    pred = m.predict(cmd=r_cmd_to_dict(ref["pred_cmd"], m))
    want = pl.DataFrame(ref["pred_cmd_prediction"])
    assert pred.height == want.height
    # R's expand.grid varies the first variable fastest, pyrsm the last:
    # align the rows on the categorical explanatory variables
    keys = [v for v in m.evar if not m.isnumeric[v]]
    gotd = pred.with_columns([pl.col(k).cast(pl.Utf8) for k in keys])
    if keys:
        gotd = gotd.sort(keys)
        want = want.sort(keys)
    for v in m.evar:
        if m.isnumeric[v]:
            np.testing.assert_allclose(gotd[v].cast(pl.Float64).to_numpy(), want[v].to_numpy(), rtol=TOL)
        else:
            assert gotd[v].to_list() == want[v].to_list()
    for lv in m.levels:
        np.testing.assert_allclose(gotd[lv].to_numpy(), want[lv].to_numpy(), rtol=TOL, atol=TOL)


@pytest.mark.parametrize("case", CASES)
def test_summary_matches_r(case):
    """Radiant's summary, line for line (scope lines use pyrsm's own wording,
    so they are compared separately)."""
    m, ref = fit_case(case)
    got = [line.rstrip() for line in capture(m.summary)]
    want = [line.rstrip() for line in ref["summary"]]
    assert [x for x in got if not x.startswith(SCOPE_LABELS)] == [
        x for x in want if not x.startswith(SCOPE_LABELS)
    ]
    a = ref["args"]
    if "data_filter" in a:
        assert f"Filter               : {a['data_filter']}" in got
    if "arr" in a:
        assert f"Sort                 : {r_args_to_kwargs(a)['sort']}" in got
    if "rows" in a:
        assert f"Slice                : {r_args_to_kwargs(a)['slice']}" in got


@pytest.mark.parametrize("case", CASES)
def test_vimp_matches_r(case):
    m, ref = fit_case(case)

    def as_map(d):
        if "Predict" in d:
            return {(p, v): a for p, v, a in zip(d["Predict"], d["vars"], d["auc"])}
        return {v: a for v, a in zip(as_list(d["vars"]), as_list(d["auc"]))}

    got = m.vimp().to_dict(as_series=False)
    want = as_map(ref["vimp"])
    assert as_map(got).keys() == want.keys()
    for k, a in as_map(got).items():
        assert a == pytest.approx(want[k], abs=TOL)
    if "Predict" not in got:  # two levels: sorted by AUC, as Radiant's plot
        assert got["vars"] == as_list(ref["vimp"]["vars"])
    if "vimp_lev" in ref:
        got = m.vimp(lev=ref["vimp_lev"]).to_dict(as_series=False)
        assert got["vars"] == as_list(ref["vimp_lev_data"]["vars"])
        np.testing.assert_allclose(got["auc"], ref["vimp_lev_data"]["auc"], atol=TOL)


def test_summary_wraps_wide_tables():
    m, ref = fit_case("titanic_wide")
    out = capture(m.summary)
    # the fare_bracket table is printed in two blocks, each with the header
    assert sum(line == "        fare_bracket" for line in out) == 2
    assert max(len(line.rstrip()) for line in out) < 80


def test_zero_sd_uses_threshold():
    """age_yes is constant for survivors: e1071 replaces sd = 0 by 0.001."""
    m, _ = fit_case("titanic_wide")
    tab = m.tables["age_yes"]
    assert tab["st.dev"][0] == 0
    one = pl.DataFrame(
        {"pclass": ["1st"], "fare_bracket": ["fare_bracket_low"], "age_yes": [30.0005]}
    )
    p = m.predict(one)
    # likelihoods by hand
    from scipy.stats import norm

    t = m._tables
    lik = m._apriori * t["pclass"][:, 0] * t["fare_bracket"][:, 1]
    lik = lik * np.array(
        [norm.pdf(30.0005, 30, 0.001), norm.pdf(30.0005, t["age_yes"][1, 0], t["age_yes"][1, 1])]
    )
    np.testing.assert_allclose(p.select(m.levels).row(0), lik / lik.sum(), rtol=1e-9)


# -- e1071 semantics ------------------------------------------------------------


def test_dnorm_matches_r_reference_values():
    # values from R: dnorm(c(0, 1, 6.3, 38, 40), 0, 1); dnorm(1, 2, 0.5)
    np.testing.assert_allclose(
        _dnorm(np.array([0, 1, 6.3, 38, 40]), 0.0, 1.0),
        [3.9894228040143270e-01, 2.4197072451914337e-01, 9.6014333703123363e-10,
         1.0972210519949712e-314, 0.0],
        rtol=1e-13,
        atol=0,
    )
    assert _dnorm(1.0, 2.0, 0.5) == pytest.approx(0.10798193302637613, rel=1e-14)
    assert _dnorm(30.0005, 30.0, 0.001) == pytest.approx(352.06532676450462, rel=1e-13)
    assert np.isnan(_dnorm(np.nan, 0.0, 1.0))


def test_missing_and_unseen_values_are_ignored():
    titanic = load("titanic")
    m = nb(titanic, rvar="survived", evar=["pclass", "sex", "age"])
    base = pl.DataFrame({"pclass": ["1st"], "sex": ["male"], "age": [30.0]})
    full = m.predict(base)
    # an unseen level contributes nothing, same as leaving the variable out
    unseen = m.predict(base.with_columns(pl.lit("crew").alias("pclass")))
    m2 = nb(titanic, rvar="survived", evar=["sex", "age"])
    np.testing.assert_allclose(
        unseen.select(m.levels).row(0), m2.predict(base).select(m.levels).row(0), rtol=1e-12
    )
    assert full["Yes"][0] != unseen["Yes"][0]
    # rows with missing values are dropped from the predictions (na.omit)
    withna = pl.concat(
        [base, pl.DataFrame({"pclass": ["2nd"], "sex": ["female"], "age": [None]},
                            schema=base.schema)]
    )
    assert m.predict(withna).height == 1


def test_laplace_smoothing_formula():
    titanic = load("titanic")
    m0 = nb(titanic, rvar="survived", evar=["pclass"])
    m1 = nb(titanic, rvar="survived", evar=["pclass"], laplace=2)
    counts = m0._tables["pclass"] * m0._apriori[:, None]
    want = (counts + 2) / (m0._apriori[:, None] + 2 * 3)
    np.testing.assert_allclose(m1._tables["pclass"], want, rtol=1e-12)


def test_type_conversions():
    """Text, logical and integer columns, as e1071 treats them in R."""
    titanic = load("titanic")
    df = titanic.select(
        pl.col("survived").cast(pl.Utf8),
        pl.col("sex").cast(pl.Utf8),
        (pl.col("pclass") == "1st").alias("first"),
        pl.col("sibsp"),
    )
    m = nb(df, rvar="survived", evar=["sex", "first", "sibsp"])
    # text response: order of appearance (pyrsm policy; R would sort alphabetically)
    assert m.levels == ["Yes", "No"]
    assert m.xlevels["first"] == ["FALSE", "TRUE"]
    assert m.isnumeric == {"sex": False, "first": False, "sibsp": True}
    ref = nb(titanic.with_columns((pl.col("pclass") == "1st").alias("first")),
             rvar="survived", evar=["sex", "first", "sibsp"])
    np.testing.assert_allclose(
        m.predict()["Yes"].to_numpy(), ref.predict()["Yes"].to_numpy(), rtol=1e-12
    )
    # numeric response: levels ordered by value
    num = pl.DataFrame({"y": [10, 2, 1, 2, 10, 1], "x": [1.0, 2.0, 3.5, 2.5, 0.5, 3.0]})
    assert nb(num, rvar="y", evar=["x"]).levels == ["1", "2", "10"]


def test_pandas_input_and_lazy_import():
    titanic = load("titanic")
    m = nb(titanic.to_pandas(), rvar="survived", evar=["pclass", "sex"])
    assert m.name == "Not provided"
    assert m.predict().columns == ["pclass", "sex", "Yes", "No"]
    # rsm.model.nb resolves to the class through the lazy import (run in a fresh
    # interpreter: importing pyrsm.model.nb directly binds the module instead)
    import subprocess
    import sys

    code = "import pyrsm as rsm; import inspect; print(inspect.isclass(rsm.model.nb))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert out.stdout.strip() == "True", out.stderr


def test_data_cmd_and_dec():
    m, _ = fit_case("titanic")
    p = m.predict(data=load("titanic"), data_cmd={"sex": "male"}, dec=2)
    assert p["sex"].unique().to_list() == ["male"]
    assert p["Yes"].to_list() == [round(x, 2) for x in p["Yes"].to_list()]
    # plug values: mean for numeric (not truncated for integers), mode for categorical
    md = nb(load("titanic"), rvar="survived", evar=["pclass", "sibsp"])
    g = md.predict(cmd={"pclass": "1st"})
    assert g["sibsp"][0] == pytest.approx(load("titanic")["sibsp"].mean())


def test_lev_and_plot_level():
    m = nb(load("titanic"), rvar="pclass", evar=["survived", "age"], lev="2nd")
    assert m.lev == "2nd"
    assert m.vimp().equals(m.vimp(lev="2nd"))
    assert "Predict" in m.vimp(lev="All levels").columns


# -- errors and edge cases --------------------------------------------------------


def test_errors():
    titanic = load("titanic")
    with pytest.raises(ValueError, match="Response variable contained"):
        nb(titanic, rvar="survived", evar=["survived", "sex"])
    with pytest.raises(ValueError, match="no variation"):
        nb(titanic, rvar="survived", evar=["sex"], data_filter="sex == 'male'")
    with pytest.raises(ValueError, match="not a level"):
        nb(titanic, rvar="survived", evar=["sex"], lev="Maybe")
    with pytest.raises(ValueError, match="not in the data"):
        nb(titanic, rvar="survived", evar=["nope"])
    with pytest.raises(ValueError, match="laplace"):
        nb(titanic, rvar="survived", evar=["sex"], laplace=-1)
    with pytest.raises(ValueError, match="explanatory"):
        nb(titanic, rvar="survived", evar=[])
    m = nb(titanic, rvar="survived", evar=["sex", "age"])
    with pytest.raises(ValueError, match="must also be in the prediction data"):
        m.predict(titanic.drop("age"))
    with pytest.raises(ValueError, match="not in the model"):
        m.predict(cmd={"pclass": ["1st"]})
    with pytest.raises(ValueError, match="Type mismatch"):
        m.predict(titanic.with_columns(pl.col("age").cast(pl.Utf8)))
    with pytest.raises(ValueError, match="Unknown plot"):
        m.plot("nope")


def test_dropped_rows_reported():
    titanic = load("titanic").with_columns(
        pl.when(pl.int_range(pl.len()) < 3).then(None).otherwise(pl.col("age")).alias("age")
    )
    m = nb(titanic, rvar="survived", evar=["sex", "age"])
    assert m.nobs == 1040
    assert any("(3 obs. dropped)" in line for line in capture(m.summary))


# -- plots ------------------------------------------------------------------------


def test_plots_smoke():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plotnine import ggplot

    m, _ = fit_case("titanic")
    assert isinstance(m.plot("vimp"), ggplot)
    assert isinstance(m.plot("vimp", lev="No"), ggplot)
    m.plot("vimp").draw()
    m.plot("correlations", nobs=200)
    m.plot("correlations", lev="Yes", nobs=200)
    plt.close("all")
    md, _ = fit_case("diamonds_cut")
    p = md.plot("vimp")
    assert isinstance(p, ggplot)
    p.draw()
    assert md.plot("") is None
    plt.close("all")
