"""Prediction parity with radiant.model: R's ``predict(...)`` vs pyrsm's ``.predict(...)``.

Fixtures come from ``scripts/generate_predict_reference.R``: one
``tests/reference/radiant_model/predict_<model>_<case>.json`` per case, with
the R model call, the prediction call (pred_data / pred_cmd / data + pred_cmd)
and R's prediction table. Every case refits the same model in pyrsm and calls
``.predict`` the way a pyrsm user would (``data=``, ``cmd=`` or
``data=`` + ``data_cmd=``), then

* ``test_grid``: the explanatory-variable columns, the row count and the values
  of every non-prediction column (the grid of a command and how variables not
  mentioned are filled). An R error must be a pyrsm error.
* ``test_predictions``: the prediction column(s), and interval columns where
  there are any. Deterministic models to about 1e-6 relative; xgboost to 1e-5
  where pyrsm builds the same design as R; random forest (ranger vs sklearn)
  and neural network (nnet vs sklearn MLP) are different engines, so only the
  range of the predictions is checked (the numeric gap is in ``gap_report``).
* ``test_layout``: column order and row order exactly as R prints them.

pyrsm's naming conventions are mapped, not tested: ``Prediction`` is
``prediction``, R's interval labels ``2.5%`` are ``2.50%``, and R's ``+/-``
column (half the interval width) has no pyrsm counterpart.

A known discrepancy is marked ``xfail(strict=True)`` with its root cause, so
the suite passes now and fails once a fix lands (then remove the mark).
"""

from __future__ import annotations

import json
import math
import re
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import pyrsm as rsm

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "reference" / "radiant_model"
DATA = ROOT / "examples" / "data" / "model"

# Tolerances. regress, logistic, nb and crtree are deterministic: 1e-6
# relative (test_nb and test_crtree use 1e-10 and 1e-6). mnl: R's
# nnet::multinom stops at reltol 1e-8 on the log-likelihood, so its
# probabilities are only good to ~1e-4; test_mnl.py compares them with
# atol=2e-4. Logistic intervals: R's glm() takes the covariance from the IRLS
# weights of the iteration before its last update (titanic: the standard
# errors of the linear predictor are 4.5e-5 off the ones at the final
# coefficients; refitting in R with epsilon=1e-14 agrees with pyrsm to 1e-14),
# so the interval bounds get 1e-4 relative. xgboost: 1e-5 where pyrsm builds
# R's design (it then agrees to 1e-12: same XGBoost core, same data).
RTOL = 1e-6
MNL_ATOL = 2e-4
GLM_INTERVAL_RTOL = 1e-4
XGB_TOL = 1e-5
ENGINE_DIFFERS = {"nn", "rforest"}

CASES = sorted(
    p.stem.removeprefix("predict_")
    for p in REF.glob("predict_*.json")
    if not p.stem.startswith("predict_data_")
)


def as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, list) else [x]


@cache
def fixture(case: str) -> dict:
    return json.loads((REF / f"predict_{case}.json").read_text())


# -- data -------------------------------------------------------------------

SCHEMA = json.loads((REF / "predict_data_schema.json").read_text())


def polars_type(tp):
    if isinstance(tp, dict):
        return pl.Enum(as_list(tp["enum"]))
    return {"int": pl.Int32, "float": pl.Float64, "bool": pl.Boolean, "str": pl.String, "date": pl.Date}[tp]


def typed(name: str, values: list, tp) -> pl.Series:
    """A column with R's type: factor -> Enum (its levels), integer -> Int32,
    numeric -> Float64, logical -> Boolean, character -> String."""
    if tp == "bool":
        return pl.Series(name, values, dtype=pl.Boolean)
    if tp in ("int", "float"):
        return pl.Series(name, [None if v is None else float(v) for v in values], dtype=pl.Float64).cast(
            polars_type(tp)
        )
    return pl.Series(name, values, dtype=pl.String).cast(polars_type(tp))


@cache
def load_dataset(name: str) -> pl.DataFrame:
    if name in SCHEMA:  # synthetic data written by the generator
        raw = pl.read_csv(REF / f"predict_data_{name}.csv", infer_schema=False, null_values="")
        cols = []
        for c, tp in SCHEMA[name].items():
            vals = raw[c].to_list()
            if tp == "bool":
                vals = [None if v is None else v == "TRUE" for v in vals]
            cols.append(typed(c, vals, tp))
        return pl.DataFrame(cols)
    return pl.read_parquet(DATA / f"{name}.parquet")


def frame(spec: dict) -> pl.DataFrame:
    """A prediction dataset from a fixture: a package dataset by name, or a
    small frame written into the fixture with its R column types."""
    if "dataset" in spec:
        return load_dataset(spec["dataset"])
    n = spec["nrow"]
    cols = []
    for c in as_list(spec["columns"]):
        vals = as_list(spec["values"][c])
        assert len(vals) == n
        cols.append(typed(c, vals, spec["types"][c]))
    return pl.DataFrame(cols)


# -- the pyrsm model and prediction call ------------------------------------


def scope(a: dict) -> dict:
    kw = {}
    if a.get("data_filter"):
        kw["data_filter"] = a["data_filter"]
    if a.get("arr"):  # R's arr = "desc(fare)" is pyrsm's sort = "-fare"
        kw["sort"] = a["arr"].replace("desc(", "-").replace(")", "")
    if "rows" in a:  # R's rows = 1:500 is pyrsm's slice = "1:500"
        rows = as_list(a["rows"])
        kw["slice"] = f"{rows[0]}:{rows[-1]}"
    return kw


def fit_kwargs(model: str, a: dict) -> dict:
    """radiant.model's arguments as pyrsm's."""
    kw = dict(rvar=a["rvar"], evar=as_list(a["evar"]), **scope(a))
    check = as_list(a.get("check"))
    if model == "regress":
        kw.update(ivar=as_list(a.get("int")) or None, standardize="standardize" in check, center="center" in check)
    elif model == "logistic":
        kw.update(
            lev=a.get("lev"), ivar=as_list(a.get("int")) or None, weights=a.get("wts"),
            standardize="standardize" in check, center="center" in check,
        )
    elif model == "mnl":
        kw.update(
            lev=a.get("lev"), ivar=as_list(a.get("int")) or None, weights=a.get("wts"),
            standardize="standardize" in check, center="center" in check,
        )
    elif model == "nb":
        kw.update(laplace=a.get("laplace", 0))
    elif model == "crtree":
        kw.update(lev=a.get("lev"), mod_type=a.get("type") or None, wts=a.get("wts"), cp=a.get("cp", 0.001))
        for k in ("pcp", "nodes", "prior", "adjprob", "cost", "margin"):
            if k in a:
                kw[k] = a[k]
        if check:
            kw["check"] = check
    elif model == "nn":  # nnet(size, decay) ~ MLP(hidden_layer_sizes, alpha)
        kw.update(
            lev=a.get("lev"), mod_type=a.get("type", "classification"),
            hidden_layer_sizes=(a.get("size", 1),), alpha=a.get("decay", 0.5), random_state=1234,
        )
    elif model == "rforest":
        kw.update(lev=a.get("lev"), mod_type=a.get("type", "classification"), n_estimators=100, random_state=1234)
    elif model == "gbt":  # radiant.model's defaults, single-threaded
        kw.update(
            lev=a.get("lev"), mod_type=a.get("type", "classification"), n_estimators=100, max_depth=6,
            learning_rate=0.3, min_child_weight=1, subsample=1.0, random_state=1234, n_jobs=1,
        )
    return kw


PYRSM_CLASS = {"nn": "mlp", "gbt": "xgboost"}


@cache
def _fit(model: str, args_json: str):
    a = json.loads(args_json)
    cls = getattr(rsm.model, PYRSM_CLASS.get(model, model))
    return cls({a["dataset"]: load_dataset(a["dataset"])}, **fit_kwargs(model, a))


def fit(fx: dict):
    return _fit(fx["model"], json.dumps(fx["args"], sort_keys=True))


def predict_kwargs(fx: dict) -> dict:
    p, opts = fx["pred"], fx.get("opts") or {}
    kw = {}
    if "data" in p:
        kw["data"] = frame(p["data"])
    if "cmd" in p:
        cmd = {k: as_list(v) for k, v in p["cmd"].items()}
        if "data" in p:  # R: pred_data + pred_cmd changes columns of the data
            kw["data_cmd"] = {k: v[0] if len(v) == 1 else v for k, v in cmd.items()}
        else:
            kw["cmd"] = cmd
    if fx["model"] in ("regress", "logistic") and opts.get("se"):
        kw["ci"] = True
        kw["conf"] = opts.get("conf_lev", 0.95)
        if fx["model"] == "regress":
            kw["interval"] = opts.get("interval", "confidence")
    return kw


def predict(fx: dict) -> pl.DataFrame:
    return fit(fx).predict(**predict_kwargs(fx))


# -- R's table as pyrsm names it --------------------------------------------


def r_table(fx: dict) -> dict:
    return fx["table"]


def pred_columns(fx: dict) -> dict:
    """R's prediction columns -> pyrsm's column (None: no pyrsm counterpart)."""
    t = r_table(fx)
    evar = set(as_list(fx["r_evar"]))
    out = {}
    for c in as_list(t["columns"]):
        if c in evar:
            continue
        if c == "Prediction":
            out[c] = "prediction"
        elif c == "+/-":
            out[c] = None  # half the interval width
        elif re.fullmatch(r"[0-9.]+%", c):
            out[c] = f"{float(c[:-1]):.2f}%"
        elif fx["model"] == "rforest":  # a probability per level; pyrsm: lev only
            out[c] = "prediction" if c == fx["args"].get("lev") else None
        else:  # mnl / nb: one probability column per level
            out[c] = c
    return out


def evar_columns(fx: dict) -> list[str]:
    evar = set(as_list(fx["r_evar"]))
    return [c for c in as_list(r_table(fx)["columns"]) if c in evar]


def _values(s: pl.Series) -> list:
    if s.dtype.is_numeric() or s.dtype == pl.Boolean:
        return s.to_list()
    return s.cast(pl.String).to_list()


def _key(v) -> str:
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, int | float):
        return f"{float(v):.9g}"
    return str(v)


def grid_order(fx: dict, got: pl.DataFrame) -> list[int]:
    """pyrsm's row for each of R's rows. A command's grid is matched on the
    values of the command's variables (R's expand.grid varies the first
    variable fastest, pyrsm's grid the last); when the values differ (e.g.
    truncated) the grid positions are mapped. Other predictions keep the rows
    of the data."""
    n = got.height
    p = fx["pred"]
    if p["mode"] != "cmd" or len(p["cmd"]) < 2:
        return list(range(n))
    t = r_table(fx)
    keys = list(p["cmd"])
    if not t["truncated"] and all(k in got.columns for k in keys):
        r_rows = [tuple(_key(v) for v in vals) for vals in zip(*(as_list(t["values"][k]) for k in keys))]
        g_rows = [tuple(_key(v) for v in vals) for vals in zip(*(_values(got[k]) for k in keys))]
        if len(set(g_rows)) == n and set(g_rows) == set(r_rows):
            pos = {k: i for i, k in enumerate(g_rows)}
            return [pos[k] for k in r_rows]
    sizes = [len(as_list(v)) for v in p["cmd"].values()]
    if math.prod(sizes) != n:
        return list(range(n))
    order = []
    for r in range(n):  # decompose R's row (first variable fastest)
        idx, rest = [], r
        for s in sizes:
            idx.append(rest % s)
            rest //= s
        pos = 0
        for i, s in zip(idx, sizes):  # pyrsm's row (last variable fastest)
            pos = pos * s + i
        order.append(pos)
    return order


def aligned(fx: dict, got: pl.DataFrame) -> pl.DataFrame:
    """pyrsm's rows in R's order. When pyrsm kept rows R drops (a missing
    explanatory value), those rows are left out so the values can still be
    compared; ``test_grid`` reports the row count."""
    n = r_table(fx)["nrow"]
    if got.height != n:
        ev = [c for c in evar_columns(fx) if c in got.columns]
        got = got.filter(pl.all_horizontal(pl.col(ev).is_not_null()))
    if got.height != n:
        return got
    return got[grid_order(fx, got)]


def compare_column(name: str, got: pl.Series, fx: dict, want_name: str, rtol: float, atol: float | None = None):
    """pyrsm's column vs R's at the rows R kept (all rows, or the first and
    last 10 plus summary statistics of a long table)."""
    t = r_table(fx)
    rows = [r - 1 for r in as_list(t["rows"])]
    want = as_list(t["values"][want_name])
    tp = t["types"][want_name]
    g = got[rows]
    if tp in ("int", "float"):
        w = np.array([np.nan if v is None else v for v in want], dtype=float)
        gv = g.cast(pl.Float64).fill_null(np.nan).to_numpy()
        tol = atol if atol is not None else 1e-9 * max(1.0, float(np.nanmax(np.abs(w))) if w.size else 1.0)
        np.testing.assert_allclose(gv, w, rtol=rtol, atol=tol, equal_nan=True, err_msg=name)
        st = t["stats"].get(want_name)
        if t["truncated"] and st:
            full = got.cast(pl.Float64).drop_nulls().to_numpy()
            for k, v in (("mean", full.mean()), ("sd", full.std(ddof=1)), ("min", full.min()), ("max", full.max())):
                np.testing.assert_allclose(v, st[k], rtol=rtol, atol=tol, err_msg=f"{name} {k}")
    elif tp == "bool":
        assert g.to_list() == want, name
    else:  # factor / text: compare the labels
        assert g.cast(pl.String).to_list() == want, name


# -- known discrepancies ----------------------------------------------------
#
# Each root cause once, with the pyrsm file:line; ``KNOWN`` maps the cases it
# breaks (case -> {test: reason}). Line numbers are those of pyrsm at the
# commit that added this file.

INT_FILL = (
    "cmd: an integer column the command does not mention is filled with its mean "
    "cast back to the integer type, i.e. truncated (2000.94 -> 2000, 0.504 -> 0); "
    "R uses the mean. sim_prediction keeps each column's dtype "
    "(pyrsm/model/model.py:1893 schema, :1916 expand_grid(dct, schema)) and "
    "expand_grid casts to it (pyrsm/utils.py:203)"
)
INT_VALUES = (
    "cmd: fractions given for an integer column are truncated (1500.5 -> 1500, "
    "2.5 -> 2); R keeps them. Same cast: pyrsm/model/model.py:1893/1916 and "
    "pyrsm/utils.py:203"
)
DATACMD_INT = (
    "data + cmd: a fraction set for an integer column is cast to the column's "
    "integer type (2.5 -> 2); R's mutate keeps 2.5. pyrsm/model/nb.py:646 and "
    "pyrsm/model/crtree.py:774 (pl.lit(v).cast(<column dtype>))"
)
NA_ROWS = (
    "data: rows with a missing explanatory value are kept, with a NaN (regress, "
    "logistic) or a surrogate-split (crtree) prediction; R's predict_model drops "
    "them (select(vars) %>% na.omit()). No drop_nulls in regress.py:736-741, "
    "logistic.py:756-761, crtree.py:768-771 (nb.py:652 does drop them)"
)
NA_FILL = (
    "cmd: means and modes use every non-missing value of a column, also rows the "
    "model did not use (the response or another explanatory variable missing); R "
    "uses the model frame (complete rows). sim_prediction gets data_raw, the "
    "scoped data before rows with missing values are dropped: regress.py:737 -> "
    ":755, logistic.py:757 -> :775"
)
TEXT_MODE = (
    "cmd: tied modes in a text (String) column: pyrsm breaks the tie by first "
    "appearance (text becomes an Enum in order of appearance, "
    "pyrsm/model/model.py:141-146; sim_prediction then takes the first tied "
    "level, :1886-1890; nb.py:564-570 the same), R by its alphabetical "
    "table(): names(which.max(table(x)))"
)
BOOL_TIE = (
    "cmd: a bool column with exactly half TRUE: R fills FALSE (max_lfreq: "
    "mean(x) > 0.5), pyrsm the value that appears first (sim_prediction treats a "
    "Boolean as categorical: pyrsm/model/model.py:1882 is_numeric() is False for "
    "Boolean, ties at :1886-1890)"
)
NEW_LEVEL_CRTREE = (
    "a level not seen in estimation: R's predict refuses it (model.frame with "
    "xlev: 'factor dom has new levels zz'); pyrsm's crtree codes it as missing "
    "and predicts through surrogate splits (pyrsm/model/crtree.py:559, "
    "code.get(x, np.nan))"
)
NEW_LEVEL_MNL = (
    "a level not seen in estimation: R's mnl gives that row NA probabilities "
    "(predict.mnl re-levels with factor(x, levels)), pyrsm raises polars' "
    "InvalidOperationError from a strict cast (pyrsm/model/mnl.py:786)"
)
GBT_DESIGN = (
    "xgboost design: pyrsm one-hot codes a categorical with 3+ levels with a "
    "column for EVERY level (conditional_get_dummies, "
    "pyrsm/model/model.py:696-697 and :726 drop_first=False); radiant.model's gbt "
    "uses model.matrix treatment coding (first level dropped). With R's design "
    "pyrsm's own XGBoost call matches R to 1e-12"
)
CRTREE_ADJPROB = (
    "R bug, not pyrsm: radiant.model::crtree ignores adjprob = FALSE (it tests "
    "!is.empty(adjprob), which is TRUE for FALSE) and always rescales the "
    "probabilities for the prior; pyrsm honours adjprob=False "
    "(pyrsm/model/crtree.py:538)"
)
STD_CI = (
    "R refuses intervals for a standardized/centered regression ('Standard "
    "error calculations not supported ...') and returns the prediction only; "
    "pyrsm returns the interval in original units (pyrsm/model/regress.py:764-775)"
)
LAYOUT_COLS = (
    "cmd: R orders the table's columns by type (numeric, factor, text, date, "
    "logical; then the command's variables, radiant.model predict_model: "
    "plug_data + expand.grid); pyrsm keeps the model's variable order "
    "(pyrsm/model/model.py:1892 sim_prediction)"
)
LAYOUT_ROWS = (
    "cmd with 2+ variables: R's expand.grid varies the first variable fastest, "
    "pyrsm's itertools.product the last (pyrsm/utils.py:189)"
)

KNOWN: dict[str, dict[str, str]] = {}


def _known(reason: str, tests: str, *cases: str) -> None:
    for case in cases:
        for test in tests.split():
            have = KNOWN.setdefault(case, {}).get(test)
            KNOWN[case][test] = f"{have} || {reason}" if have else reason


# integer columns in a command's grid
_known(INT_FILL, "grid predictions",
       "logistic_dvd_cmd", "logistic_syn_cmd_fill", "logistic_syn_wts_cmd", "logistic_tit_int_cols_cmd",
       "mnl_syn_cmd_fill", "mnl_tit_cmd_age", "regress_catalog_cmd", "regress_hp_cmd_int_means",
       "regress_hp_cmd_int_values", "regress_hp_pi90_cmd", "regress_syn_cmd_bool", "regress_syn_cmd_fill",
       "regress_syn_filter_cmd", "regress_syn_xi_frac_cmd")
# (a tree's prediction does not change between 3 and 3.9 here; nn/rforest
# predictions are not compared)
_known(INT_FILL, "grid", "crtree_dvd_cmd", "crtree_syn_cmd_fill", "gbt_catalog_cmd", "gbt_dvd_cmd",
       "nn_dvd_cmd", "rforest_dvd_cmd")
_known(INT_VALUES, "grid predictions", "regress_hp_cmd_int_values", "regress_syn_xi_frac_cmd")
_known(DATACMD_INT, "grid predictions", "nb_dvd_datacmd_frac", "crtree_dvd_datacmd_frac")
# missing values
_known(NA_ROWS, "grid layout", "regress_syn_na_data", "logistic_syn_na_data", "crtree_syn_na_data")
_known(NA_FILL, "grid predictions", "regress_syn_na_x_cmd", "regress_syn_na_y_cmd", "logistic_syn_na_x_cmd")
# ties
_known(TEXT_MODE, "grid predictions", "regress_syn_tie_text", "nb_syn_tie_text")
_known(BOOL_TIE, "grid predictions", "regress_syn_flag50")
# levels not seen in estimation
_known(NEW_LEVEL_CRTREE, "grid", "crtree_syn_newlevel")
_known(NEW_LEVEL_MNL, "grid predictions layout", "mnl_syn_newlevel")
# model-specific
_known(GBT_DESIGN, "predictions", "gbt_dia_cmd", "gbt_dia_data", "gbt_tit_cmd", "gbt_tit_data",
       "gbt_tit_datacmd", "gbt_tit_levsub")
_known(CRTREE_ADJPROB, "predictions", "crtree_tit_prior_noadj")
_known(STD_CI, "grid", "regress_dia_std_ci_cmd")


def _layout_known() -> None:
    """The order of a command's table: columns by type, rows of a grid."""
    for case in CASES:
        fx = fixture(case)
        p = fx["pred"]
        if "error" in fx or p["mode"] != "cmd" or case in ("mnl_syn_newlevel",):
            continue
        cmd = list(p["cmd"])
        evar = as_list(fx["r_evar"])
        r_order = [c for c in as_list(fx["table"]["columns"]) if c in evar]
        if r_order != evar:
            _known(LAYOUT_COLS, "layout", case)
        if len(cmd) > 1 and math.prod(len(as_list(v)) for v in p["cmd"].values()) > 1:
            _known(LAYOUT_ROWS, "layout", case)


_layout_known()


def _xfails() -> dict[tuple[str, str], str]:
    return {(case, test): reason for case, tests in KNOWN.items() for test, reason in tests.items()}


def params(test: str, cases=None):
    xf = _xfails()
    out = []
    for case in cases if cases is not None else CASES:
        marks = []
        if (case, test) in xf:
            marks.append(pytest.mark.xfail(strict=True, reason=xf[(case, test)]))
        out.append(pytest.param(case, marks=marks, id=case))
    return out


# -- tests ------------------------------------------------------------------


def test_reference_data_is_pyrsm_data():
    """The package datasets R used are pyrsm's parquet files (rows, column
    types, factor levels in order, column sums)."""
    fp = json.loads((REF / "predict_data_fingerprint.json").read_text())
    for name, want in fp.items():
        d = load_dataset(name)
        assert d.height == want["nrow"], name
        for c, tp in want["types"].items():
            assert c in d.columns, (name, c)
            if isinstance(tp, dict):
                assert isinstance(d[c].dtype, pl.Enum), (name, c)
                assert d[c].dtype.categories.to_list() == as_list(tp["enum"]), (name, c)
            elif tp == "int":
                assert d[c].dtype.is_integer(), (name, c)
        for c, s in want["sums"].items():
            np.testing.assert_allclose(float(d[c].cast(pl.Float64).sum()), s, rtol=1e-12, err_msg=f"{name}.{c}")


@pytest.mark.parametrize("case", params("grid"))
def test_grid(case):
    fx = fixture(case)
    if "error" in fx:
        with pytest.raises(Exception):
            predict(fx)
        return
    got = predict(fx)
    t = r_table(fx)
    ev = evar_columns(fx)
    pmap = pred_columns(fx)
    want_cols = set(ev) | {v for v in pmap.values() if v}
    assert set(got.columns) == want_cols, f"pyrsm {got.columns} vs R {t['columns']}"
    assert got.height == t["nrow"], f"rows: pyrsm {got.height} vs R {t['nrow']}"
    got = got[grid_order(fx, got)]
    for c in ev:
        compare_column(c, got[c], fx, c, rtol=1e-9)


@pytest.mark.parametrize("case", params("predictions", [c for c in CASES if "error" not in fixture(c)]))
def test_predictions(case):
    fx = fixture(case)
    got = aligned(fx, predict(fx))
    t = r_table(fx)
    assert got.height == t["nrow"], f"rows: pyrsm {got.height} vs R {t['nrow']}"
    model = fx["model"]
    for rcol, pcol in pred_columns(fx).items():
        if pcol is None:
            continue
        if model in ENGINE_DIFFERS:
            # a different engine: no numeric closeness, only a sane range
            vals = got[pcol].to_numpy()
            assert np.isfinite(vals).all(), pcol
            if fx["args"].get("type", "classification") == "classification":
                assert ((vals >= 0) & (vals <= 1)).all(), pcol
            continue
        if model == "gbt":
            compare_column(pcol, got[pcol], fx, rcol, rtol=XGB_TOL, atol=XGB_TOL)
        elif model == "mnl":
            compare_column(pcol, got[pcol], fx, rcol, rtol=0, atol=MNL_ATOL)
        elif model == "logistic" and rcol != "Prediction":
            compare_column(pcol, got[pcol], fx, rcol, rtol=GLM_INTERVAL_RTOL)
        else:
            compare_column(pcol, got[pcol], fx, rcol, rtol=RTOL)


@pytest.mark.parametrize("case", params("layout", [c for c in CASES if "error" not in fixture(c)]))
def test_layout(case):
    """Column order and row order exactly as R's table (with pyrsm's names)."""
    fx = fixture(case)
    got = predict(fx)
    pmap = pred_columns(fx)
    want = [pmap.get(c, c) for c in as_list(r_table(fx)["columns"])]
    want = [c for i, c in enumerate(want) if c and c not in want[:i]]
    assert got.columns == want, f"columns: pyrsm {got.columns} vs R {want}"
    t = r_table(fx)
    assert got.height == t["nrow"], f"rows: pyrsm {got.height} vs R {t['nrow']}"
    for c in list(fx["pred"].get("cmd", {})):  # the grid's rows in R's order
        if fx["pred"]["mode"] == "cmd" and c in got.columns:
            rows = [r - 1 for r in as_list(t["rows"])]
            assert [_key(v) for v in _values(got[c][rows])] == [_key(v) for v in as_list(t["values"][c])], f"row order of {c}"


RF_OOB = (
    "random forest on its own estimation data: R's predict.rforest compares the "
    "prediction data with the model frame (all.equal) and then returns the "
    "out-of-bag predictions; pyrsm returns out-of-bag predictions only for "
    "predict() without data and in-sample predictions for predict(data=<the same "
    "data>) (pyrsm/model/rforest.py:291-296)"
)
RF_OOB_CASES = [
    c for c in CASES
    if fixture(c)["model"] == "rforest" and fixture(c)["pred"]["mode"] == "data"
    and "dataset" in fixture(c)["pred"]["data"] and not fixture(c)["args"].get("data_filter")
]


@pytest.mark.parametrize("case", [pytest.param(c, marks=pytest.mark.xfail(strict=True, reason=RF_OOB)) for c in RF_OOB_CASES])
def test_rforest_estimation_data_is_out_of_bag(case):
    """As R: predicting a forest's own estimation data gives out-of-bag
    predictions (the values differ by engine, so pyrsm is compared to itself)."""
    fx = fixture(case)
    m = fit(fx)
    np.testing.assert_allclose(predict(fx)["prediction"].to_numpy(), m.predict()["prediction"].to_numpy())


# -- numeric gap of the engines that differ ---------------------------------


def gap_report() -> list[dict]:
    """R vs pyrsm predictions for nn, rforest and gbt: max absolute difference
    and correlation over the rows the fixture keeps (all rows, or the first and
    last 10), and the mean of the predictions over all rows. For a random
    forest predicted on its estimation data R uses out-of-bag predictions;
    ``oob`` repeats the comparison with pyrsm's ``predict()`` (out-of-bag too).

    Run ``uv run python tests/test_predict_radiant_parity.py`` to print it.
    """
    out = []
    for case in CASES:
        fx = fixture(case)
        if fx["model"] not in ("nn", "rforest", "gbt") or "error" in fx:
            continue
        t = r_table(fx)
        rows = [r - 1 for r in as_list(t["rows"])]
        variants = [("predict", aligned(fx, predict(fx)))]
        if fx["model"] == "rforest" and fx["pred"]["mode"] == "data" and "dataset" in fx["pred"]["data"]:
            variants.append(("oob", fit(fx).predict()))
        for rcol, pcol in pred_columns(fx).items():
            if pcol is None:
                continue
            want = np.array(as_list(t["values"][rcol]), dtype=float)
            for label, got in variants:
                g = got[pcol].to_numpy()
                gr = g[rows]
                corr = float(np.corrcoef(gr, want)[0, 1]) if len(want) > 2 and want.std() > 0 and gr.std() > 0 else float("nan")
                out.append(dict(
                    case=case, call=label, column=rcol, n=t["nrow"],
                    max_abs_diff=float(np.max(np.abs(gr - want))),
                    corr=corr, r_mean=t["stats"][rcol]["mean"], pyrsm_mean=float(np.mean(g)),
                ))
    return out


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    print(f"{'case':30s} {'call':8s} {'col':6s} {'n':>6s} {'max|diff|':>10s} {'corr':>7s} {'R mean':>10s} {'py mean':>10s}")
    for r in gap_report():
        print(
            f"{r['case']:30s} {r['call']:8s} {r['column']:6s} {r['n']:6d} {r['max_abs_diff']:10.4g} "
            f"{r['corr']:7.3f} {r['r_mean']:10.4g} {r['pyrsm_mean']:10.4g}"
        )
