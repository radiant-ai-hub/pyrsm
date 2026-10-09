"""Tests for pyrsm.model.crtree against radiant.model::crtree (rpart 4.1.27).

Reference fixtures are produced by ``scripts/generate_crtree_reference.R``.
"""

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pyrsm.model._mermaid import MermaidDiagram
from pyrsm.model._rpart import RRandom, r_sample_int_replace, r_sample_xgroups
from pyrsm.model.crtree import crtree, cv_crtree

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "reference" / "radiant_model"
DATA = ROOT / "examples" / "data" / "model"
TOL = 1e-9

CASES = sorted(p.stem.removeprefix("crtree_") for p in REF.glob("crtree_*.json"))


def load(name: str) -> pl.DataFrame:
    if name == "titanic_agg":
        d = pl.read_csv(REF / "titanic_agg.csv")
        return d.with_columns(
            pl.col("pclass").cast(pl.Enum(["1st", "2nd", "3rd"])),
            pl.col("sex").cast(pl.Enum(["female", "male"])),
            pl.col("survived").cast(pl.Enum(["Yes", "No"])),
            pl.col("n").cast(pl.Int32),
        )
    return pl.read_parquet(DATA / f"{name}.parquet")


def ref_case(case: str) -> dict:
    return json.loads((REF / f"crtree_{case}.json").read_text())


def r_args_to_kwargs(a: dict) -> dict:
    """Translate the R call into pyrsm arguments."""
    evar = a["evar"] if isinstance(a["evar"], list) else [a["evar"]]
    kw = dict(rvar=a["rvar"], evar=evar)
    for k in ("lev", "cp", "minsplit", "prior", "cost", "margin", "split", "nodes",
              "pcp", "wts", "data_filter"):
        if k in a:
            kw[k] = a[k]
    if "type" in a:
        kw["mod_type"] = a["type"]
    if "arr" in a:  # R's arr = "desc(fare)" is pyrsm's sort = "-fare"
        kw["sort"] = a["arr"].replace("desc(", "-").replace(")", "")
    if "rows" in a:  # R's rows = 1:600 is pyrsm's slice = "1:600"
        kw["slice"] = f"{a['rows'][0]}:{a['rows'][-1]}"
    return kw


def mat(d: dict) -> np.ndarray:
    return np.array(d["values"], dtype=float).reshape(d["nrow"], d["ncol"])


_FITS: dict = {}


def fit_case(case: str):
    if case not in _FITS:
        ref = ref_case(case)
        a = ref["args"]
        _FITS[case] = (crtree({a["dataset"]: load(a["dataset"])}, **r_args_to_kwargs(a)), ref)
    return _FITS[case]


# -- R's random numbers --------------------------------------------------------


def test_r_rng_matches_r():
    ref = json.loads((REF / "r_rng.json").read_text())
    assert np.array_equal(r_sample_xgroups(1043, 10, 1234), ref["xgroups_1043_10"])
    assert np.array_equal(r_sample_int_replace(5, 1000, RRandom(1234)), ref["sample_5_1000"])


# -- the fitted tree -------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_frame_matches_r(case):
    ct, ref = fit_case(case)
    fr = ref["frame"]
    py = ct.fit.frame
    assert ct.nobs == ref["nobs"]
    assert [r["id"] for r in py] == fr["id"]
    assert [r["var"] for r in py] == fr["var"]
    for k in ("n", "wt", "dev", "yval", "complexity", "ncompete", "nsurrogate"):
        got = np.array([r[k] for r in py], dtype=float)
        want = np.array(fr[k], dtype=float)
        assert np.max(np.abs(got - want) / np.maximum(1, np.abs(want))) < TOL, k
    if "yval2" in fr:
        got = np.array([r["yval2"] for r in py])
        assert np.max(np.abs(got - mat(fr["yval2"]))) < TOL
    assert np.array_equal(ct.fit.where + 1, ref["where"])


@pytest.mark.parametrize("case", CASES)
def test_splits_match_r(case):
    ct, ref = fit_case(case)
    if ref["splits"] is None:
        assert ct.fit.splits_table() == []
        return
    tab = ct.fit.splits_table()
    S = mat(ref["splits"])
    assert [t["var"] for t in tab] == ref["splits"]["rownames"]
    for j, k in [(0, "count"), (1, "ncat"), (2, "improve"), (4, "adj")]:
        got = np.array([t[k] for t in tab], dtype=float)
        assert np.max(np.abs(got - S[:, j])) < TOL, k
    cont = np.array([t["index"] is not None for t in tab])
    if cont.any():
        got = np.array([t["index"] for t in tab if t["index"] is not None], dtype=float)
        assert np.max(np.abs(got - S[cont, 3])) < TOL


@pytest.mark.parametrize("case", CASES)
def test_cptable_matches_r(case):
    """Includes the cross-validated error, so the folds must match R's too."""
    ct, ref = fit_case(case)
    want = mat(ref["cptable"])
    assert ct.fit.cptable.shape == want.shape
    assert np.max(np.abs(ct.fit.cptable - want)) < TOL


@pytest.mark.parametrize("case", CASES)
def test_variable_importance_matches_r(case):
    ct, ref = fit_case(case)
    imp = ct.fit.variable_importance()
    assert list(imp) == ref["importance"]["names"]
    want = np.array(ref["importance"]["values"])
    assert np.max(np.abs(np.array(list(imp.values())) - want) / want) < TOL


@pytest.mark.parametrize("case", CASES)
def test_predictions_match_r(case):
    ct, ref = fit_case(case)
    d = load(ref["args"]["dataset"]).select(ct.evar).drop_nulls()
    pred = ct.predict(d)["prediction"].to_numpy()
    assert np.allclose(pred, ref["prediction"], atol=TOL)


def test_predict_cmd_matches_r():
    ct, ref = fit_case("titanic_pclass_sex")
    pred = ct.predict(cmd={"pclass": ["1st", "2nd", "3rd"]})
    assert np.allclose(pred["prediction"].to_numpy(), ref["pred_cmd_prediction"])


# -- printed output and the mermaid tree -----------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_summary_matches_r(case):
    """Radiant's summary, line for line (scope lines use pyrsm's own wording,
    and TRUE where Python prints True)."""
    ct, ref = fit_case(case)
    buf = io.StringIO()
    with redirect_stdout(buf):
        ct.summary(splits=True, cptab=True)
    got = [line.rstrip() for line in buf.getvalue().rstrip("\n").split("\n")]
    want = [line.rstrip() for line in ref["summary_extra"]]
    assert len(got) == len(want)
    for g, w in zip(got, want):
        if w.startswith(("Slice", "Arrange")):
            continue
        if w.startswith("Adjusted prob."):
            g, w = g.lower(), w.lower()
        assert g == w


@pytest.mark.parametrize("case", CASES)
def test_mermaid_matches_r(case):
    ct, ref = fit_case(case)
    if ct.mod_type == "classification" and len(ct.ylevels) > 2:
        # Radiant shows yval2[, 4] as "p", which is a class count (not a
        # probability) when there are more than two classes; pyrsm shows the
        # probability of `lev`. Everything else is the same.
        import re

        strip = lambda s: re.sub(r"<b>p:</b> [0-9.,]+%", "P", s)  # noqa: E731
        assert strip(ct.to_mermaid()) == strip(ref["diagram"])
    else:
        assert ct.to_mermaid() == ref["diagram"]


def test_plot_tree_returns_mermaid():
    ct, _ = fit_case("titanic_pclass_sex")
    m = ct.plot()
    assert isinstance(m, MermaidDiagram)
    assert m.source == ct.to_mermaid()
    assert "mermaid" in m._repr_html_()
    assert m._repr_markdown_().startswith("```mermaid\ngraph LR")
    assert ct.plot(orient="TD").source.startswith("graph TD")


def test_mermaid_save(tmp_path):
    ct, _ = fit_case("titanic_pclass_sex")
    m = ct.plot()
    m.save(tmp_path / "tree.html")
    m.save(tmp_path / "tree.mmd")
    assert "<pre class=\"mermaid\"" in (tmp_path / "tree.html").read_text()
    assert (tmp_path / "tree.mmd").read_text().startswith("graph LR")


def test_single_node_tree():
    d = load("titanic")
    ct = crtree(d, rvar="survived", evar=["sex"], lev="Yes", cp=0.9)
    assert len(ct.fit.frame) == 1
    assert ct.to_mermaid() == "graph LR\n A[Graph unavailable for single node tree]"


# -- cross-validation ------------------------------------------------------------


def test_cv_crtree_matches_r(capsys):
    ref = json.loads((REF / "cv_crtree.json").read_text())
    t, d = load("titanic"), load("diamonds")
    cases = [
        ("titanic", crtree(t, rvar="survived", evar=["pclass", "sex", "age"], lev="Yes"),
         dict(pcp=list(np.linspace(0, 0.01, 6)))),
        ("diamonds", crtree(d, rvar="price", evar=["carat", "clarity"],
                            mod_type="regression", cp=0.005),
         dict(K=3, repeats=2, pcp=[0, 0.002, 0.01])),
        ("prior", crtree(t, rvar="survived", evar=["pclass", "sex", "age"], lev="Yes",
                         prior=0.5),
         dict(pcp=[0, 0.005, 0.02])),
    ]
    for name, model, kw in cases:
        out = cv_crtree(model, trace=False, **kw)
        r = ref[name]
        want = np.column_stack([np.array(r[c], dtype=float) for c in r])
        assert list(out.columns) == list(r)
        assert np.allclose(out.to_numpy().astype(float), want, atol=1e-9), name
    assert "pruning complexity parameter" in capsys.readouterr().out


# -- data scope ------------------------------------------------------------------


def test_scope_equals_prescoped_data():
    """filter -> sort -> slice gives the same tree as scoping the data by hand."""
    d = load("titanic")
    scoped = crtree(d, rvar="survived", evar=["pclass", "sex", "age"], lev="Yes",
                    data_filter="age > 10", sort="-fare", slice="1:573")
    manual = (
        d.filter(pl.col("age") > 10)
        .sort("fare", descending=True, maintain_order=True)
        .head(573)
    )
    by_hand = crtree(manual, rvar="survived", evar=["pclass", "sex", "age"], lev="Yes")
    assert ct_equal(scoped, by_hand)


def ct_equal(a: crtree, b: crtree) -> bool:
    fa, fb = a.fit.frame, b.fit.frame
    return (
        [r["id"] for r in fa] == [r["id"] for r in fb]
        and np.allclose([r["dev"] for r in fa], [r["dev"] for r in fb])
        and np.allclose(a.fit.cptable, b.fit.cptable)
    )


# -- options and errors ----------------------------------------------------------


def test_standardize_scales_numeric_evar():
    d = load("titanic")
    ct1 = crtree(d, rvar="survived", evar=["pclass", "sex", "age"], lev="Yes",
                 check="standardize-1sd")
    ct2 = crtree(d, rvar="survived", evar=["pclass", "sex", "age"], lev="Yes",
                 check="standardize")
    assert ct1.sf == 1 and ct2.sf == 2
    age = d.drop_nulls(["survived", "pclass", "sex", "age"])["age"].to_numpy()
    X = ct1._encode(ct1.data.select(ct1.evar))
    assert np.allclose(X[:, 2], (age - age.mean()) / age.std(ddof=1))
    # a tree is invariant to standardization: same nodes and predictions
    base = crtree(d, rvar="survived", evar=["pclass", "sex", "age"], lev="Yes")
    assert [r["n"] for r in ct1.fit.frame] == [r["n"] for r in base.fit.frame]
    assert np.allclose(ct1.predict()["prediction"], base.predict()["prediction"])


def test_numeric_response_as_classification():
    d = load("titanic").with_columns(
        (pl.col("survived") == "Yes").cast(pl.Int32).alias("surv01")
    )
    ct = crtree(d, rvar="surv01", evar=["pclass", "sex"], lev="1", mod_type="classification")
    ref = crtree(d, rvar="survived", evar=["pclass", "sex"], lev="Yes")
    assert ct.ylevels == ["1", "0"]
    assert np.allclose(ct.predict()["prediction"], ref.predict()["prediction"])


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        (dict(evar=["survived", "sex"]), "Response variable contained"),
        (dict(nodes=1), "larger than or equal to 2"),
        (dict(lev="Maybe"), "not a level"),
        (dict(prior=0.5, cost=1, margin=5), "either a prior or cost"),
        (dict(cost=5, margin=5), "smaller than the specified margin"),
        (dict(prior=1.5), "not a valid probability"),
        (dict(mod_type="regression"), "Cannot estimate a regression"),
    ],
)
def test_errors(kwargs, msg):
    args = dict(rvar="survived", evar=["pclass", "sex"], lev="Yes")
    args.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        crtree(load("titanic"), **args)


def test_no_variation_raises():
    d = load("titanic").with_columns(pl.lit(1.0).alias("const"))
    with pytest.raises(ValueError, match="no variation"):
        crtree(d, rvar="survived", evar=["sex", "const"], lev="Yes")


def test_unknown_level_in_prediction_uses_surrogates():
    """A level not seen in estimation is treated as missing (surrogate splits)."""
    ct, _ = fit_case("titanic_age")
    new = pl.DataFrame({"pclass": ["1st"], "sex": ["unknown"], "age": [30.0]})
    pred = ct.predict(new)["prediction"]
    assert pred.len() == 1 and 0 <= pred[0] <= 1


# -- other plots -----------------------------------------------------------------


@pytest.mark.parametrize("plots", ["prune", "imp", "pip", "pred"])
def test_classification_plots(plots):
    ct, _ = fit_case("titanic_age")
    assert ct.plot(plots) is not None


def test_pdp_and_dashboard():
    ct, _ = fit_case("titanic_age")
    assert ct.plot("pdp", incl=["age"]) is not None
    rt, _ = fit_case("diamonds_reg_color")
    assert rt.plot("dashboard") is not None
    with pytest.raises(ValueError, match="only available for regression"):
        ct.plot("dashboard")


def test_pip_returns_table():
    ct, _ = fit_case("titanic_age")
    p, tab = ct.plot("pip", ret=True)
    assert set(tab["variable"]) == {"pclass", "sex", "age"}
    assert tab["importance"].to_list() == sorted(tab["importance"], reverse=True)


# -- rules: one row per terminal node ------------------------------------------
def test_rules_cover_every_leaf_once():
    d = load("titanic")
    ct = crtree({"titanic": d}, rvar="survived", lev="Yes", evar=["pclass", "sex", "age"],
                prior=0.5, cp=0.01)
    rules = ct.rules()
    leaves = [r for r in ct.fit.frame if r["leaf"]]
    assert sorted(rules["node"].to_list()) == sorted(r["id"] for r in leaves)
    assert rules["n"].sum() == ct.fit.frame[0]["n"]
    assert abs(rules["share"].sum() - 1) < 1e-12
    assert rules["prediction"].to_list() == sorted(rules["prediction"].to_list(), reverse=True)
    assert rules["rule"].to_list() == [
        "sex = female",
        "sex = male and age < 9.5",
        "sex = male and age >= 9.5",
    ]
    # the probabilities are the (prior-adjusted) ones the tree plot shows
    assert round(rules["prediction"][0], 4) == 0.7513


def test_rules_merge_bounds_on_one_variable():
    d = load("titanic")
    ct = crtree({"titanic": d}, rvar="survived", lev="Yes", evar=["pclass", "sex", "age"],
                prior=0.5, cp=0.001)
    rules = ct.rules()["rule"].to_list()
    # each variable appears at most once per rule
    for rule in rules:
        names = [part.split()[0] if not part[0].isdigit() else part.split()[2]
                 for part in rule.split(" and ")]
        assert len(names) == len(set(names)), rule
    assert any(" <= age < " in rule for rule in rules)
    assert any("pclass in 1st, 2nd" in rule for rule in rules)


def test_rules_regression_predicts_the_node_mean():
    d = load("catalog")
    ct = crtree({"catalog": d}, rvar="Sales", evar=["Income", "HH_size", "Age"], cp=0.02)
    rules = ct.rules()
    means = {r["id"]: r["yval"] for r in ct.fit.frame if r["leaf"]}
    for node, pred in zip(rules["node"], rules["prediction"]):
        assert pred == means[node]


def test_node_table_lists_every_node_in_print_order():
    d = load("titanic")
    ct = crtree({"titanic": d}, rvar="survived", lev="Yes", evar=["pclass", "sex", "age"],
                prior=0.5, cp=0.005)
    nodes = ct.node_table()
    assert nodes["node"].to_list() == [r["id"] for r in ct.fit.frame]
    assert nodes["depth"].to_list() == [0, 1, 2, 2, 3, 3, 1, 2, 2]
    assert nodes["split"].to_list()[:4] == ["", "sex = female", "pclass in 1st, 2nd", "pclass = 3rd"]
    assert nodes["last"].to_list() == [False, False, False, True, False, True, True, False, True]
    leaves = nodes.filter(pl.col("leaf"))
    assert leaves["n"].sum() == 1043
    # a leaf's prediction is the one rules() reports for it
    rules = dict(zip(ct.rules()["node"], ct.rules()["prediction"]))
    assert all(rules[n] == p for n, p in zip(leaves["node"], leaves["prediction"]))
