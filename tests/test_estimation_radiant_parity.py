"""Estimation parity with radiant.model: R's regress() / logistic() and their
summary() vs pyrsm's rsm.model.regress / rsm.model.logistic.

Fixtures come from ``scripts/generate_estimation_reference.R``: one
``tests/reference/radiant_model/est_<model>_<case>.json`` per case, with the
radiant.model call and R's estimation output. Every case refits the model in
pyrsm the way a pyrsm user would and compares, field by field, what
``summary()`` and the Radiant report (``pyrsm.model.report``) show:

* ``test_coefficients``: estimate, std.error, t/z, p.value (and the odds ratio
  for logistic) per term, matched by name, and the aliased (dropped) terms.
* ``test_coefficient_order``: the rows of the coefficient table in R's order.
* ``test_fit``: R-squared, adjusted R-squared, F, its df and p, RMSE, residual
  SE, the sums of squares and nobs (regress); pseudo R-squared, adjusted, AUC,
  log-likelihood, AIC, BIC, chi-squared, df, p and nobs (logistic) -- the
  numbers radiant prints (``model_fit``, ``rmse``).
* ``test_ci``: the coefficient confidence intervals radiant shows (confint,
  confint_robust, profile likelihood or confint.default); for logistic also the
  odds-ratio intervals.
* ``test_vif``: car::vif (GVIF) as radiant prints it, per term.
* ``test_variables_test``: the model comparison (``test_var``): F-test (regress),
  likelihood-ratio chi-squared or robust Wald (logistic).
* ``test_residuals``: residuals and fitted values (first and last 10 rows and
  summary statistics).
* ``test_stepwise``: the AIC path, the term changed at each step and the
  selected terms.
* ``test_error``: a model R refuses, pyrsm refuses.

Tolerances. regress: 1e-8 relative (``REG_RTOL``).

logistic: the reference is the tight refit. R's glm() takes its covariance
from the IRLS weights at the start of its last iteration, i.e. at the
coefficients before the last update (default epsilon = 1e-8, a relative
change in deviance), so radiant's printed standard errors are up to 5e-4 off
the converged ones (titanic with pclass:sex). Every logistic fixture therefore
also holds the same model refit in R with ``glm.control(epsilon = 1e-14,
maxit = 100)`` (``tight``), and pyrsm is compared to that refit:

* ``GLM_RTOL`` = 1e-6 for estimates, odds ratios, fit statistics (they depend
  on the deviance at the final coefficients), likelihood-ratio tests, profile
  intervals and residuals (observed agreement: 1e-10 standard errors);
* ``GLM_SE_RTOL`` = 1e-5 for standard errors and what is built from them (z,
  p, Wald and robust intervals, VIF, robust Wald tests). statsmodels' IRLS
  starts where R's does (mu = (y + 0.5) / 2) and has the same lag: its
  covariance also comes from the weights at the start of its last iteration.
  It stops on an absolute 1e-8 change in deviance, i.e. at R's iteration or one
  or two later, so in these fixtures pyrsm's standard errors equal either
  radiant's printed ones or the tight refit's to 12 digits, and are at most
  3.4e-6 from the converged values (``tit_arr_rows``, where they equal
  radiant's).

``test_logistic_radiant_output`` compares pyrsm to the numbers radiant prints
(default convergence), with the largest gaps observed: estimates to
``RADIANT_EST_SE`` = 1e-5 of their standard error (1.4e-6, titanic with
pclass:sex, where R stops an iteration early), standard errors, z, Wald
intervals and VIF to ``RADIANT_SE_RTOL`` = 1e-3 (4.9e-4, the same case), and
R's profile intervals to ``PROFILE_SE_ATOL`` = 1e-3 standard errors (6.5e-4,
the rare level): radiant's ``confint`` interpolates a spline through a grid of
profile points. The fixtures also hold the exact roots (uniroot on the profile
deviance), which pyrsm's ``profile_ci`` finds and ``test_ci`` compares at 1e-6.

p-values: a relative error e in a t or z statistic is a relative error of
about e * t^2 in its tail probability, so p-values get ``rtol * max(1, t^2)``.

A discrepancy goes in ``KNOWN`` (case -> {test: reason}, the root cause with
the pyrsm file:line), which marks it ``xfail(strict=True)``; the mark comes off
with the fix. A case without a finite maximum-likelihood estimate (perfect
separation) is in ``NO_MLE`` and a coefficient without one (quasi-separation)
in ``DIVERGING``: both programs stop wherever their iterations end, so only
the behaviour is tested (``test_perfect_separation``,
``test_quasi_separation_*``).
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import warnings
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import pyrsm as rsm
from pyrsm.model.model import coef_ci, gvif, model_fit, r_label
from pyrsm.model.model import vif as design_vif

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "reference" / "radiant_model"
DATA = ROOT / "examples" / "data" / "model"

REG_RTOL = 1e-8
GLM_RTOL = 1e-6
GLM_SE_RTOL = 1e-5
RADIANT_EST_SE = 1e-5
RADIANT_SE_RTOL = 1e-3
PROFILE_SE_ATOL = 1e-3

CASES = sorted(
    p.stem.removeprefix("est_")
    for p in REF.glob("est_*.json")
    if not p.stem.startswith("est_data_")
)


def as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, list) else [x]


@cache
def fixture(case: str) -> dict:
    return json.loads((REF / f"est_{case}.json").read_text())


# -- data -------------------------------------------------------------------

SCHEMA = json.loads((REF / "est_data_schema.json").read_text())


def polars_type(tp):
    if isinstance(tp, dict):
        return pl.Enum(as_list(tp["enum"]))
    return {"int": pl.Int32, "float": pl.Float64, "bool": pl.Boolean, "str": pl.String, "date": pl.Date}[tp]


@cache
def load_dataset(name: str) -> pl.DataFrame:
    """A package dataset (pyrsm's parquet) or a synthetic CSV the generator
    wrote, with R's column types: factor -> Enum (its levels), integer ->
    Int32, numeric -> Float64, logical -> Boolean, character -> String."""
    if name not in SCHEMA:
        return pl.read_parquet(DATA / f"{name}.parquet")
    raw = pl.read_csv(REF / f"est_data_{name}.csv", infer_schema=False, null_values="")
    cols = []
    for c, tp in SCHEMA[name].items():
        s = raw[c]
        if tp == "bool":
            s = pl.Series(c, [None if v is None else v == "TRUE" for v in s.to_list()], dtype=pl.Boolean)
        elif tp in ("int", "float"):
            s = s.cast(pl.Float64).cast(polars_type(tp))
        else:
            s = s.cast(polars_type(tp))
        cols.append(s.alias(c))
    return pl.DataFrame(cols)


# -- the pyrsm call ---------------------------------------------------------


def scope(a: dict, py: dict) -> dict:
    kw = {}
    if py.get("data_filter") or a.get("data_filter"):
        kw["data_filter"] = py.get("data_filter") or a["data_filter"]
    if a.get("arr"):  # R's arr = "desc(fare)" is pyrsm's sort = "-fare"
        kw["sort"] = a["arr"].replace("desc(", "-").replace(")", "")
    if "rows" in a:  # R's rows = 1:500 is pyrsm's slice = "1:500"
        rows = as_list(a["rows"])
        kw["slice"] = f"{rows[0]}:{rows[-1]}"
    return kw


def pyrsm_lev(lev: str, column: pl.Series):
    """radiant.model's lev as pyrsm's: "" is no level (None); for a number
    or logical response the level is that value ("1" -> 1, "TRUE" -> True)."""
    if lev in (None, ""):
        return None
    if column.dtype == pl.Boolean:
        return lev == "TRUE"
    if column.dtype.is_integer():
        return int(lev)
    if column.dtype.is_float():
        return float(lev)
    return lev


def fit_kwargs(fx: dict) -> dict:
    """radiant.model's arguments as pyrsm's."""
    a, py = fx["args"], fx.get("py") or {}
    kw = dict(rvar=a["rvar"], evar=as_list(a["evar"]), **scope(a, py))
    check = as_list(a.get("check"))
    kw["ivar"] = as_list(a.get("int")) or None
    kw["standardize"] = 1 if "standardize-1sd" in check else 2 if "standardize" in check else False
    kw["center"] = "center" in check
    kw["stepwise"] = next((c.split("-", 1)[1] for c in check if c.startswith("stepwise")), False)
    kw["robust"] = "robust" in check
    if fx["model"] == "logistic":
        kw["lev"] = pyrsm_lev(a.get("lev", ""), load_dataset(a["dataset"])[a["rvar"]])
        kw["weights"] = a.get("wts")
        if py.get("ci_type"):
            kw["ci_type"] = py["ci_type"]
    return kw


@cache
def _fit(case: str):
    fx = fixture(case)
    cls = getattr(rsm.model, fx["model"])
    a = fx["args"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = cls({a["dataset"]: load_dataset(a["dataset"])}, **fit_kwargs(fx))
    m._parity_warnings = [f"{w.category.__name__}: {w.message}" for w in caught]
    return m


def fit(case: str):
    return _fit(case)


# -- matching terms ---------------------------------------------------------


def r_key(term: str) -> frozenset:
    """An R coefficient name (``pclass2nd:sexmale``, ``I(age^2)``) as a key
    that ignores the order of the parts of an interaction."""
    return frozenset(term.split(":"))


_LEVEL = re.compile(r"(.+?)\[(?:T\.)?(.*)\]\Z")


def py_key(name: str) -> frozenset:
    """A pyrsm / statsmodels coefficient name (``pclass[T.2nd]:sex[male]``,
    ``I(age ** 2)``, ``flag[True]``) as R's key."""
    if name == "Intercept":
        return frozenset(["(Intercept)"])
    parts = []
    for p in name.split(":"):
        p = r_label(p)
        m = _LEVEL.fullmatch(p)
        if m:
            var, lev = m.groups()
            p = var + {"True": "TRUE", "False": "FALSE"}.get(lev, lev)
        parts.append(p)
    return frozenset(parts)


def term_key(name: str) -> frozenset:
    """A model term (``carat:clarity``, ``I(carat ** 2)``) as a key."""
    return frozenset(r_label(p).strip() for p in name.split(":"))


# -- comparisons ------------------------------------------------------------


def close(got, want, rtol, atol=0.0, msg=""):
    """|got - want| <= atol + rtol |want| elementwise (atol may be an array)."""
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    assert got.shape == want.shape, f"{msg}: shape {got.shape} vs R {want.shape}"
    ok = np.isclose(got, want, rtol=rtol, atol=0) | (np.abs(got - want) <= np.asarray(atol) + rtol * np.abs(want))
    ok |= np.isnan(got) & np.isnan(want)
    if not ok.all():
        bad = np.flatnonzero(~ok.ravel())
        g, w = got.ravel()[bad], want.ravel()[bad]
        rel = np.abs(g - w) / np.where(w == 0, 1, np.abs(w))
        raise AssertionError(
            f"{msg}: {len(bad)} of {got.size} differ (rtol {rtol:g}); pyrsm {g.tolist()} vs R {w.tolist()}; "
            f"max relative difference {rel.max():.3g}"
        )


def close_p(got, want, stat, rtol, msg=""):
    """p-values: a relative error e in the statistic is ~e * stat^2 in p."""
    stat = np.abs(np.asarray(stat, dtype=float))
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    tol = rtol * np.maximum(1.0, stat**2)
    bad = ~(np.isclose(got, want, rtol=0, atol=0) | (np.abs(got - want) <= tol * np.abs(want)))
    assert not bad.any(), f"{msg}: p-values {got[bad]} vs R {want[bad]} (stat {stat[bad]})"


def r_coef(fx: dict, block: str = "coef") -> dict:
    """R's coefficient rows by key."""
    c = fx[block] if block == "coef" else fx["tight"]
    names = ("estimate", "std_error", "statistic", "p_value")
    return {r_key(t): dict(zip(names, vals)) for t, *vals in zip(c["term"], *(as_list(c[n]) for n in names))}


def py_coef(m) -> dict:
    stat = "t.value" if "t.value" in m.coef.columns else "z.value"
    return {
        py_key(r["index"]): {
            "estimate": r["coefficient"], "std_error": r["std.error"], "statistic": r[stat],
            "p_value": r["p.value"], "OR": r.get("OR"), "OR_pct": r.get("OR%"),
        }
        for r in m.coef.iter_rows(named=True)
    }


def compare_coef(case: str, want: dict, got: dict, rtol: float, se_rtol: float):
    assert set(got) == set(want), f"terms: pyrsm {sorted(map(sorted, got))} vs R {sorted(map(sorted, want))}"
    keys = [k for k in want if k not in DIVERGING.get(case, ())]
    for k, rt in (("estimate", rtol), ("std_error", se_rtol), ("statistic", se_rtol)):
        close([got[t][k] for t in keys], [want[t][k] for t in keys], rtol=rt, atol=1e-300, msg=f"{case} {k}")
    close_p([got[t]["p_value"] for t in keys], [want[t]["p_value"] for t in keys],
            [want[t]["statistic"] for t in keys], se_rtol, msg=case)


def pyrsm_ci(m, conf: float):
    """The interval pyrsm's summary(ci=True) / the Radiant report show."""
    if isinstance(m, rsm.model.logistic):
        ci = m.conf_int(conf=conf)
        return {py_key(i): (lo, hi) for i, lo, hi in zip(ci.index, ci.iloc[:, 0], ci.iloc[:, 1])}
    ci = coef_ci(m.fitted_inference, alpha=1 - conf, dec=None)
    return {py_key(i): (row.iloc[1], row.iloc[2]) for i, row in ci.iterrows()}


def r_ci(spec: dict) -> dict:
    return {r_key(t): (lo, hi) for t, lo, hi in zip(spec["term"], as_list(spec["low"]), as_list(spec["high"]))}


def compare_ci(case: str, want: dict, got: dict, rtol: float, atol: float = 0.0):
    assert set(got) == set(want), f"terms: pyrsm {sorted(map(sorted, got))} vs R {sorted(map(sorted, want))}"
    keys = [k for k in want if want[k][0] is not None and want[k][1] is not None]
    for i, side in enumerate(("low", "high")):
        close([got[k][i] for k in keys], [want[k][i] for k in keys], rtol=rtol, atol=atol, msg=f"{case} {side}")


def pyrsm_vif(m) -> dict | None:
    """The VIF table summary(vif=True) / the report print, or None when pyrsm
    prints that it was not (or cannot be) calculated."""
    if m.aliased or m.evar is None or len(m.evar) < 2:
        return None
    if isinstance(m, rsm.model.logistic):
        tab = gvif(m.fitted, dec=None)
    else:
        tab = design_vif(m.fitted, dec=None)
    return {term_key(i): v for i, v in zip(tab.index, tab["vif"])}


def captured(f, *args, **kwargs) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f(*args, **kwargs)
    return buf.getvalue()


def parse_test(out: str) -> dict:
    """pyrsm's printed model comparison as numbers."""
    res = {}
    m = re.search(r"Model 1: (.+)", out)
    if m:
        res["model1"] = m.group(1).strip()
    m = re.search(r"R-squared, Model 1 vs 2: (-?[0-9.]+) vs (-?[0-9.]+)", out)
    if m:
        res["r2"] = (float(m.group(1)), float(m.group(2)))
    m = re.search(r"(F-statistic|Chi-squared): (\S+) df \(([0-9]+)(?:, ([0-9]+))?\), p.value (< \.001|\S+)", out)
    if m:
        res["stat"] = float(m.group(2))
        res["df1"] = int(m.group(3))
        res["df2"] = None if m.group(4) is None else int(m.group(4))
        res["p"] = m.group(5)
    return res


def model1_terms(formula: str) -> set:
    rhs = formula.split("~", 1)[1].replace(" ", "")
    return {term_key(t) for t in rhs.split("+") if t not in ("", "1", ".")}


# -- known discrepancies ----------------------------------------------------
#
# case -> {test: reason}: xfail(strict=True) with the root cause and its
# file:line ("R:" when the cause is on radiant.model's side and pyrsm's value
# is the defensible one). The mark comes off with the fix.

LEV_DEFAULT = (
    "no lev: pyrsm/model/logistic.py:186 skips the 0/1 conversion; statsmodels then models y = 1 for a 0/1 integer "
    "response and the first Enum level (order of appearance for text) for a categorical one; radiant's logistic() "
    "uses levels(rv)[1], i.e. the first factor level or the first sorted value (0; accept), so the signs flip; "
    "fix: pick radiant's level when lev is None"
)
NA_SCALE = (
    "standardize / center with missing values: _calc_scaling (pyrsm/model/regress.py:331, logistic.py:392) uses "
    "every non-missing value of each column; radiant's get_data(na.rm = TRUE) drops incomplete rows (response, "
    "explanatory variables, weights) first, so its means and sds are those of the estimation rows; fix: compute "
    "them on the estimation rows"
)
VIF_INT = (
    "regress VIF with interactions: vif() (pyrsm/model/model.py:1013-1033) groups dummy columns by the text "
    "before '[T.' and by substring, so clarity's GVIF also takes the carat:clarity columns (82 vs 45,023) and a "
    "categorical x categorical interaction is merged into its first factor; fix: gvif() (car::vif's term-wise "
    "GVIF; matches R to 8e-12 on every regress case), as logistic already does (regress.py:589, report.py:531)"
)
VIF_GATE = (
    "the VIF is gated on len(evar) < 2 (regress.py:585, logistic.py:590, report.py:527 / 611); radiant counts "
    "the model's terms (length(term.labels) > 1), so x + I(x^2) gets a VIF table in R and 'cannot be "
    "calculated' in pyrsm; fix: count the design's terms"
)
AUC_W = (
    "AUC with weights: get_mfit (pyrsm/model/model.py:817) passes the weights to perf.auc, which ranks a "
    "table2data expansion (pyrsm/utils.py:242, repeat_by truncates non-integer weights) but divides by the raw "
    "weight sums: 0.841 vs 0.840 for integer weights and above 1 (1.17, 1.11) for non-integer ones; radiant's "
    "summary.logistic computes auc(fitted, response) without weights (titanic_agg: R 0.5, pyrsm 0.821, the AUC of "
    "the expanded rows); fix: unweighted as radiant, or a correct weighted Mann-Whitney AUC"
)
LL_W = (
    "R: non-integer weights: radiant's log-likelihood, AIC and BIC come from binomial()$aic, which rounds the "
    "weights (dbinom(round(w y), round(w), mu)); pyrsm (statsmodels freq_weights, model.py:813-819) uses "
    "sum(w log-lik), consistent with the deviance and pseudo R-squared both programs report"
)
NOBS_SW = (
    "sampling weights (two values, the smallest below 1): radiant's Nr obs (and BIC) use the number of rows; "
    "model_fit (pyrsm/model/model.py:1308-1312) uses the sum of the weights whenever it is larger (241 vs 200)"
)
PROFILE_WARM = (
    "profile_ci (pyrsm/model/model.py:1126) warm-starts every constrained refit at the full-model estimates; for "
    "x1 >= 0.773 statsmodels' IRLS from there converges to a bad fit (deviance +5,598), the profile jumps and "
    "brentq returns the jump: x1's upper bound 0.7707 vs R 0.7756; a cold start gives 0.775586488566 (R's exact "
    "root 0.775586488566)"
)
PROFILE_SEP = (
    "profile_ci (pyrsm/model/model.py:1144) raises ValueError ('f(a) and f(b) must have different signs') when "
    "the profile never reaches the cutoff (quasi-separation: a one-sided interval); R's confint gives NA for that "
    "bound; summary(ci=True), summary(odds=True) and the report fail; fix: an infinite (or missing) bound"
)
PROFILE_XTOL = (
    "profile_ci's brentq xtol=1e-10 is absolute (pyrsm/model/model.py:1144): an upper bound of 3.5e-7 (ext, a "
    "coefficient on a 1e-7 scale) is off by 5.9e-13, 1.7e-6 relative; fix: xtol relative to the standard error"
)
CONST = (
    "a column without variation: radiant refuses the model ('The following variable(s) show no variation. Please "
    "select other variables.'); pyrsm (no does_vary check in regress.py / logistic.py __init__) fits it and drops "
    "the column as aliased"
)
LEV_TEXT = (
    "lev as radiant gives it, text, for a number or logical response: convert_binary (pyrsm/model/model.py:561) "
    "compares the column with the text, so lev='1' on an integer column raises polars' ComputeError ('cannot "
    "compare string with numeric type') and lev='TRUE' on a Boolean column matches nothing ('All converted "
    "response values are 0'); radiant's dataset[[rvar]] == lev coerces both; fix: compare as text, or cast lev "
    "to the column's type"
)
R_ROBUST_INT = (
    "R: robust standard errors and a test variable in an interaction: radiant's matchCf returns a list and "
    "car::linearHypothesis stops ('requires numeric/complex matrix/vector arguments'), so radiant shows no "
    "test; pyrsm gives the robust Wald test of the variable's coefficients"
)


def _known() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}

    def add(cases, test, reason):
        for c in cases.split():
            out.setdefault(c, {})[test] = reason

    for t in ("coefficients", "ci", "residuals", "radiant_output"):
        add("logistic_syn_resp_01_default", t, LEV_DEFAULT)
    for t in ("coefficients", "ci", "fit", "residuals"):
        add("regress_syn_na_std", t, NA_SCALE)
    for t in ("coefficients", "ci", "radiant_output"):
        add("logistic_syn_na_std", t, NA_SCALE)
    add("regress_dia_int_cat_cat regress_dia_int_num_cat regress_dia_robust_int regress_dia_std_int "
        "regress_dia_step_int", "vif", VIF_INT)
    add("regress_dia_quad_only logistic_tit_quad_only", "vif", VIF_GATE)
    add("logistic_tit_wts_agg logistic_tit_wts_int logistic_tit_wts_std logistic_syn_wts_na", "fit", AUC_W)
    add("logistic_syn_wts_frac", "fit", f"{AUC_W}. {LL_W}")
    add("logistic_syn_wts_sampling", "fit", f"{AUC_W}. {LL_W}. {NOBS_SW}")
    add("logistic_syn_na_x", "ci", PROFILE_WARM)
    add("logistic_syn_na_x", "radiant_output", PROFILE_WARM)
    add("logistic_syn_quasi_sep", "ci", PROFILE_SEP)
    add("logistic_syn_quasi_sep", "radiant_output", PROFILE_SEP)
    add("logistic_syn_ext", "ci", PROFILE_XTOL)
    add("regress_syn_const logistic_syn_const", "error", CONST)
    add("regress_dia_robust_int logistic_tit_robust_int", "variables_test", R_ROBUST_INT)
    return out


KNOWN: dict[str, dict[str, str]] = _known()

#: no finite maximum-likelihood estimate: only the behaviour is tested
#: (test_perfect_separation)
NO_MLE = {"logistic_sep"}

#: terms without a finite estimate (quasi-complete separation: the estimate
#: grows with every iteration, so it is wherever each program stops); they are
#: left out of the numeric comparisons and tested in test_quasi_separation
DIVERGING = {"logistic_syn_quasi_sep": {frozenset(["qseppure"])}}


def params(test: str, cases=None):
    xf = {(case, t): reason for case, tests in KNOWN.items() for t, reason in tests.items()}
    out = []
    for case in cases if cases is not None else CASES:
        marks = []
        if (case, test) in xf:
            marks.append(pytest.mark.xfail(strict=True, reason=xf[(case, test)]))
        out.append(pytest.param(case, marks=marks, id=case))
    return out


def ok_cases(pred=lambda fx: True) -> list[str]:
    return [c for c in CASES if "error" not in fixture(c) and c not in NO_MLE and pred(fixture(c))]


def is_logistic(fx):
    return fx["model"] == "logistic"


# -- tests ------------------------------------------------------------------


def test_reference_data_is_pyrsm_data():
    """The package datasets R used are pyrsm's parquet files (rows, column
    types, factor levels in order, column sums)."""
    fp = json.loads((REF / "est_data_fingerprint.json").read_text())
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
            elif tp == "date":
                assert d[c].dtype == pl.Date, (name, c)
        for c, s in want["sums"].items():
            np.testing.assert_allclose(float(d[c].cast(pl.Float64).sum()), s, rtol=1e-12, err_msg=f"{name}.{c}")


@pytest.mark.parametrize("case", params("error", [c for c in CASES if "error" in fixture(c)]))
def test_error(case):
    """radiant.model refuses the model (e.g. a column without variation)."""
    with pytest.raises(Exception):
        _fit.__wrapped__(case)


@pytest.mark.parametrize("case", params("coefficients", ok_cases()))
def test_coefficients(case):
    fx = fixture(case)
    m = fit(case)
    got = py_coef(m)
    assert {py_key(a) for a in m.aliased} == {r_key(a) for a in as_list(fx["aliased"])}, "aliased terms"
    if is_logistic(fx):
        want = r_coef(fx, "tight")
        compare_coef(case, want, got, GLM_RTOL, GLM_SE_RTOL)
        # odds ratios (none for the intercept): radiant's OR = exp(b) and
        # OR% = OR - 1 (shown as a percentage), at the converged estimates
        keys = [k for k in want if k != frozenset(["(Intercept)"]) and k not in DIVERGING.get(case, ())]
        odds = np.exp([want[k]["estimate"] for k in keys])
        close([got[k]["OR"] for k in keys], odds, rtol=GLM_RTOL, msg="OR")
        close([got[k]["OR_pct"] for k in keys], 100 * (odds - 1), rtol=GLM_RTOL, atol=1e-9, msg="OR%")
    else:
        compare_coef(case, r_coef(fx), got, REG_RTOL, REG_RTOL)


@pytest.mark.parametrize("case", params("coefficient_order", ok_cases()))
def test_coefficient_order(case):
    """The rows of the coefficient table in R's order (model.matrix: the
    formula's order, main effects before interactions)."""
    fx = fixture(case)
    got = [":".join(sorted(py_key(i))) for i in fit(case).coef["index"]]
    want = [":".join(sorted(r_key(t))) for t in fx["coef"]["term"]]
    assert got == want, f"pyrsm {got} vs R {want}"


@pytest.mark.parametrize("case", params("fit", ok_cases()))
def test_fit(case):
    fx = fixture(case)
    m = fit(case)
    want = fx["fit"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mf = model_fit(m.fitted, prn=False).iloc[0]
    if is_logistic(fx):
        # AUC is a step function of the predictions: two predictions that are
        # equal in exact arithmetic can tie in one program and differ by an
        # ulp in the other, which moves AUC by half a pair, 1 / (2 n1 n0)
        y = np.asarray(m.fitted.model.endog)
        auc_tol = 1.0 / (2 * y.sum() * (len(y) - y.sum())) + 1e-12
        pairs = [("pseudo_rsq_mcf", "r2", GLM_RTOL, 0), ("pseudo_rsq_mcf_adj", "r2_adj", GLM_RTOL, 0),
                 ("AUC", "auc", 0, auc_tol), ("log_likelihood", "loglik", GLM_RTOL, 0),
                 ("AIC", "aic", GLM_RTOL, 0), ("BIC", "bic", GLM_RTOL, 0), ("chisq", "chi2", GLM_RTOL, 0),
                 ("chisq_df", "df", 0, 0), ("nobs", "nobs", 0, 0)]
        bad = [
            f"{r}: pyrsm {mf[p]!r} vs R {want[r]!r}"
            for p, r, rt, at in pairs
            if not np.isclose(float(mf[p]), float(want[r]), rtol=rt, atol=at)
        ]
        assert not bad, "; ".join(bad)
        close_p([mf["chisq_pval"]], [want["p_value"]], [np.sqrt(want["chi2"])], GLM_RTOL, msg=f"{case} chi2 p")
    else:
        pairs = [("rsq", "r2"), ("rsq_adj", "adj_r2"), ("fvalue", "fstat"), ("ftest_df_model", "df_model"),
                 ("ftest_df_resid", "df_resid"), ("nobs", "nobs")]
        for p, r in pairs:
            close(mf[p], want[r], rtol=REG_RTOL, msg=f"{case} {r}")
        close_p([mf["ftest_pval"]], [want["p_value"]], [np.sqrt(want["fstat"])], REG_RTOL, msg=f"{case} F p")
        rmse, rsd = m.rmse()
        close(rmse, want["rmse"], rtol=REG_RTOL, msg="RMSE")
        close(rsd, want["sigma"], rtol=REG_RTOL, msg="RSD")
        f = m.fitted
        close([f.ess, f.ssr, f.centered_tss], [want["ss_reg"], want["ss_err"], want["ss_reg"] + want["ss_err"]],
              rtol=REG_RTOL, msg="sums of squares")
        assert [f.df_model, f.df_resid] == [want["ss_df_reg"], want["ss_df_err"]], "sum of squares df"


@pytest.mark.parametrize("case", params("ci", ok_cases(lambda fx: "ci" in fx)))
def test_ci(case):
    fx = fixture(case)
    m = fit(case)
    conf = fx["conf_lev"]
    got = pyrsm_ci(m, conf)
    # the table's rows in R's order
    assert [k for k in got if k in r_ci(fx["ci"])] == list(r_ci(fx["ci"])), "row order"
    if not is_logistic(fx):
        compare_ci(case, r_ci(fx["ci"]), got, rtol=REG_RTOL)
        return
    tight = fx["tight"]
    profile = fx["ci"]["type"] == "profile"
    want = r_ci(tight["ci_profile"] if profile else tight["ci_wald"])
    for k in DIVERGING.get(case, ()):
        want.pop(k, None), got.pop(k, None)
    compare_ci(case, want, got, rtol=GLM_RTOL if profile else GLM_SE_RTOL)
    # odds-ratio intervals (summary(odds=True)): exp of the same bounds
    keys = [k for k in want if k != frozenset(["(Intercept)"])]
    close(np.exp([got[k] for k in keys]), np.exp([want[k] for k in keys]), rtol=GLM_RTOL if profile else GLM_SE_RTOL,
          msg="odds-ratio CI")


@pytest.mark.parametrize("case", params("vif", ok_cases(lambda fx: "vif" in fx)))
def test_vif(case):
    fx = fixture(case)
    got = pyrsm_vif(fit(case))
    assert got is not None, "pyrsm does not calculate the VIF; radiant does"
    src = fx["tight"]["vif"] if is_logistic(fx) else fx["vif"]
    want = {term_key(t): v for t, v in zip(as_list(src["term"]), as_list(src["vif"]))}
    assert set(got) == set(want), f"terms: pyrsm {sorted(map(sorted, got))} vs R {sorted(map(sorted, want))}"
    keys = list(want)
    close([got[k] for k in keys], [want[k] for k in keys], rtol=GLM_SE_RTOL if is_logistic(fx) else REG_RTOL, msg="VIF")


@pytest.mark.parametrize("case", params("variables_test", ok_cases(lambda fx: "test" in fx and "skipped" not in fx["test"])))
def test_variables_test(case):
    fx = fixture(case)
    m = fit(case)
    want = fx["tight"]["test"] if is_logistic(fx) else fx["test"]
    tv = as_list(fx["test_var"])
    f = m.chisq_test if is_logistic(fx) else m.f_test
    got = parse_test(captured(f, test=tv, dec=12))
    if want.get("no_test") or want.get("error"):
        assert "stat" not in got, f"R gives no test ({want.get('error', 'no df')}); pyrsm: {got}"
        return
    assert "stat" in got, "pyrsm printed no test"
    heading = want["heading"].split("\n")[0].removeprefix("Model 1: ")
    if "~ ." not in heading:
        assert model1_terms(got["model1"]) == model1_terms(heading), f"Model 1: pyrsm {got['model1']} vs R {heading}"
    rtol = REG_RTOL
    if is_logistic(fx):  # a robust test is a Wald test on the covariance
        rtol = GLM_SE_RTOL if fx["robust"] else GLM_RTOL
    close(got["stat"], want["stat"], rtol=rtol, msg="statistic")
    if is_logistic(fx):
        assert got["df1"] == want["df"]
    else:
        assert (got["df1"], got["df2"]) == (want["df1"], want["df2"])
    if want["p_value"] < 0.001:
        assert got["p"] == "< .001"
    else:
        stat = np.sqrt(want["stat"]) if is_logistic(fx) else np.sqrt(want["stat"] * want["df1"])
        close_p([float(got["p"])], [want["p_value"]], [stat], rtol, msg="p-value")
    if want.get("r2") is not None:  # printed with 3 decimals
        r2 = as_list(want["r2"])
        assert abs(got["r2"][0] - r2[0]) <= 5e-4 + 1e-12, f"R-squared Model 1: pyrsm {got['r2'][0]} vs R {r2[0]}"
        assert abs(got["r2"][1] - r2[1]) <= 5e-4 + 1e-12, f"R-squared Model 2: pyrsm {got['r2'][1]} vs R {r2[1]}"


@pytest.mark.parametrize("case", params("residuals", ok_cases()))
def test_residuals(case):
    fx = fixture(case)
    m = fit(case)
    src = fx["tight"] if is_logistic(fx) else fx
    rtol = GLM_RTOL if is_logistic(fx) else REG_RTOL
    resid_model = m.fitted.resid_working if is_logistic(fx) else m.fitted.resid
    for name, got_all in (("resid", np.asarray(resid_model)), ("fitted_values", np.asarray(m.fitted.fittedvalues))):
        want = src[name]
        assert len(got_all) == want["n"], f"{name}: {len(got_all)} rows vs R {want['n']}"
        atol = rtol * max(1.0, want["stats"]["sd"])
        idx = [i - 1 for i in as_list(want["idx"])]
        close(got_all[idx], as_list(want["values"]), rtol=rtol, atol=atol, msg=name)
        st = want["stats"]
        close([got_all.mean(), got_all.std(ddof=1), got_all.min(), got_all.max(), (got_all**2).sum()],
              [st["mean"], st["sd"], st["min"], st["max"], st["sum_sq"]], rtol=rtol, atol=atol, msg=f"{name} stats")
    # the residuals property: one value per row of the (scoped) data, as Radiant stores them
    want = src["resid"]
    if want.get("data_rows"):
        rows = [r - 1 for r in as_list(want["data_rows"])]
        got = m.residuals.to_numpy()[rows]
        close(got, as_list(want["values"]), rtol=rtol, atol=rtol * max(1.0, want["stats"]["sd"]), msg="residuals property")


@pytest.mark.parametrize("case", params("stepwise", ok_cases(lambda fx: "step" in fx)))
def test_stepwise(case):
    fx = fixture(case)
    m = fit(case)
    an = fx["step"]["anova"]
    rtol = GLM_RTOL if is_logistic(fx) else REG_RTOL
    steps = m.step_steps
    assert len(steps) == len(as_list(an["AIC"])), f"{len(steps)} steps vs R {len(as_list(an['AIC']))}"
    close([s["aic"] for s in steps], as_list(an["AIC"]), rtol=rtol, msg="AIC path")
    changes = [s["rows"][0]["term"] for s in steps[:-1]]
    assert [c.replace(" ", "") for c in changes] == [c.replace(" ", "") for c in as_list(an["Step"])[1:]]
    design = m.fitted.model.data.design_info
    got = {term_key(t.name()) for t in design.terms if len(t.factors) > 0}
    assert got == {term_key(t) for t in as_list(fx["step"]["terms"])}


@pytest.mark.parametrize("case", ["regress_dia4", "regress_dia_int_num_cat", "logistic_tit_int", "logistic_tit_all"])
def test_coefficient_plot_order(case):
    """The coefficient (regress) and odds-ratio (logistic) plots list the
    terms top to bottom in R's order, without the intercept."""
    fx = fixture(case)
    m = fit(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = m.plot("coef")
    want = [":".join(sorted(r_key(t))) for t in fx["coef"]["term"] if t != "(Intercept)"]
    got = [":".join(sorted(py_key(str(i)))) for i in p.data["index"].cat.categories][::-1]
    assert got == want


# -- logistic: radiant's printed numbers (default glm convergence) ----------


@pytest.mark.parametrize("case", params("radiant_output", ok_cases(is_logistic)))
def test_logistic_radiant_output(case):
    """pyrsm vs the numbers radiant prints (glm's default epsilon = 1e-8):
    estimates to RADIANT_EST_SE of their standard error, standard errors, z, p, Wald /
    robust intervals and VIF to RADIANT_SE_RTOL (glm's covariance lags its last
    update), R's spline-based profile intervals to PROFILE_SE_ATOL standard
    errors."""
    fx = fixture(case)
    m = fit(case)
    want = r_coef(fx)
    got = py_coef(m)
    assert set(got) == set(want), f"terms: pyrsm {sorted(map(sorted, got))} vs R {sorted(map(sorted, want))}"
    keys = [k for k in want if k not in DIVERGING.get(case, ())]
    se = np.array([want[k]["std_error"] for k in keys])
    close([got[k]["estimate"] for k in keys], [want[k]["estimate"] for k in keys], rtol=0, atol=RADIANT_EST_SE * se,
          msg="estimate")
    close([got[k]["std_error"] for k in keys], se, rtol=RADIANT_SE_RTOL, msg="std.error")
    close([got[k]["statistic"] for k in keys], [want[k]["statistic"] for k in keys], rtol=RADIANT_SE_RTOL, msg="z")
    close_p([got[k]["p_value"] for k in keys], [want[k]["p_value"] for k in keys],
            [want[k]["statistic"] for k in keys], RADIANT_SE_RTOL, msg="p")
    if "ci" in fx:
        wci, gci = r_ci(fx["ci"]), pyrsm_ci(m, fx["conf_lev"])
        for k in DIVERGING.get(case, ()):
            wci.pop(k, None), gci.pop(k, None)
        if fx["ci"]["type"] == "profile":
            sek = np.array([want[k]["std_error"] for k in wci])
            compare_ci(case, wci, gci, rtol=0, atol=PROFILE_SE_ATOL * sek)
        else:
            compare_ci(case, wci, gci, rtol=RADIANT_SE_RTOL)
    if "vif" in fx:
        got_v = pyrsm_vif(m)
        if got_v is not None:
            want_v = {term_key(t): v for t, v in zip(as_list(fx["vif"]["term"]), as_list(fx["vif"]["vif"]))}
            if set(got_v) == set(want_v):
                close([got_v[k] for k in want_v], list(want_v.values()), rtol=RADIANT_SE_RTOL, msg="VIF")


# -- behaviour where no estimate exists --------------------------------------


@pytest.mark.xfail(strict=True, reason=LEV_TEXT)
@pytest.mark.parametrize("case", ["logistic_syn_resp_01", "logistic_syn_resp_bool"])
def test_lev_as_text_for_a_number_or_logical_response(case):
    """radiant.model's lev is text ("1", "TRUE"); the other tests pass pyrsm
    the column's own type (pyrsm_lev). Given the text, pyrsm should fit the
    same model."""
    fx = fixture(case)
    kw = fit_kwargs(fx)
    kw["lev"] = fx["args"]["lev"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = rsm.model.logistic({"synth": load_dataset("synth")}, **kw)
    compare_coef(case, r_coef(fx, "tight"), py_coef(m), GLM_RTOL, GLM_SE_RTOL)


@pytest.mark.xfail(strict=True, reason=PROFILE_SEP)
def test_quasi_separation_interval():
    """Every row of level pure is a yes: its coefficient has no finite
    estimate and its profile interval no upper bound. R's confint gives NA
    there and finite bounds for the other coefficients."""
    fx = fixture("logistic_syn_quasi_sep")
    assert fx["ci"]["high"][fx["ci"]["term"].index("qseppure")] is None
    ci = fit("logistic_syn_quasi_sep").conf_int()
    assert not np.isfinite(ci.loc["qsep[T.pure]"].iloc[1])


def test_quasi_separation_estimates():
    """The diverging coefficient is large in both programs (each stops where
    its convergence test does: R 16.2 after 15 iterations, the tight refit
    30.2, pyrsm 22.2); the other coefficients agree (test_coefficients)."""
    fx = fixture("logistic_syn_quasi_sep")
    m = fit("logistic_syn_quasi_sep")
    r = dict(zip(fx["coef"]["term"], fx["coef"]["estimate"]))
    p = dict(zip(m.coef["index"], m.coef["coefficient"]))
    assert r["qseppure"] > 10 and fx["tight"]["estimate"][fx["tight"]["term"].index("qseppure")] > 10
    assert p["qsep[pure]"] > 10


def test_perfect_separation():
    """y is "yes" exactly when x > 0: there is no finite estimate. R's glm
    stops after 25 iterations ("algorithm did not converge", "fitted
    probabilities numerically 0 or 1 occurred"; radiant's sshhr() hides both).
    pyrsm warns (statsmodels' PerfectSeparationWarning) and, as R, reports a
    perfect fit: deviance ~0, AUC 1."""
    fx = fixture("logistic_sep")
    assert "glm.fit: fitted probabilities numerically 0 or 1 occurred" in fx["glm_warnings"]
    assert not fx["converged"]
    m = fit("logistic_sep")
    assert any("PerfectSeparationWarning" in w for w in m._parity_warnings), m._parity_warnings
    assert fx["fit"]["auc"] == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mf = model_fit(m.fitted, prn=False).iloc[0]
    assert mf["AUC"] == 1
    assert m.fitted.deviance < 1e-6 and fx["fit"]["deviance"] < 1e-6
