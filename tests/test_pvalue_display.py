"""Consistent p-value display across pyrsm.

Rule: every p-value below 0.001 is displayed as ``"< .001"`` (less-than, space,
dot, 001); larger values are rounded to the requested number of decimals
(``pydisplay_utils.format_pval``). Stored p-values stay numeric.

Each test captures a summary for data with a highly significant effect and
checks the p-value cells (table columns named ``p.value``, the ``p.values:``
correlation matrix, and ``p.value ...`` text lines) contain ``"< .001"`` and
never ``0.0``/``0.000``/``<.001``/``< 0.001``/scientific notation.
"""

import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pyrsm.basics.compare_means import compare_means
from pyrsm.basics.compare_props import compare_props
from pyrsm.basics.correlation import correlation
from pyrsm.basics.cross_tabs import cross_tabs
from pyrsm.basics.display_utils import format_pval, format_pval_expr, format_pvals
from pyrsm.basics.goodness import goodness
from pyrsm.basics.single_mean import single_mean
from pyrsm.basics.single_prop import single_prop
from pyrsm.model.logistic import logistic
from pyrsm.model.regress import regress
from pyrsm.multivariate.cbc import cbc
from pyrsm.multivariate.conjoint_choice import conjoint_choice
from pyrsm.multivariate.pre_factor import pre_factor

ROOT = Path(__file__).resolve().parents[1]
CHOC = ROOT / "examples" / "data" / "multivariate" / "choc.parquet"
EVAR = ["brand", "chocolate", "price", "nuts"]
BAD = {"0", "0.0", "0.00", "0.000", "<.001", "< 0.001", "<0.001", "p < .001"}
VALID = re.compile(r"^(< \.001|0\.\d+|1\.0|1)$")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def capture(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("│").split("┆")]


def pvalue_cells(out: str) -> list[str]:
    """Extract every displayed p-value from captured summary output."""
    found = []
    col = None  # index of the p.value column in the current polars table
    corr = False  # inside the correlation "p.values:" matrix
    in_body = False
    for line in out.splitlines():
        s = line.strip()
        if s == "p.values:":
            corr, col, in_body = True, None, False
            continue
        if s.startswith("│") and "┆" in s and not in_body:
            parts = _cells(s)
            if "p.value" in parts:
                col = parts.index("p.value")
            continue
        if s.startswith("╞"):
            in_body = col is not None or corr
            continue
        if s.startswith("└"):
            if corr and in_body:
                corr = False
            col, in_body = None, False
            continue
        if in_body and s.startswith("│"):
            parts = _cells(s)
            if corr:
                found.extend(c for c in parts[1:] if c)
            elif col is not None:
                found.append(parts[col])
            continue
        m = re.search(r"p\.value (.+?)\s*$", line)
        if m:
            found.append(m.group(1))
    return found


def check_display(out: str, min_sig: int = 1) -> list[str]:
    cells = pvalue_cells(out)
    assert cells, f"no p-values found in output:\n{out}"
    assert sum(c == "< .001" for c in cells) >= min_sig, cells
    for c in cells:
        assert c not in BAD, f"bad p-value display {c!r} in:\n{out}"
        assert "e-" not in c and "E-" not in c, c
        assert VALID.match(c), f"unexpected p-value display {c!r}"
    for bad in ("<.001", "< 0.001", "<0.001"):
        assert bad not in out, f"{bad!r} found in output:\n{out}"
    return cells


@pytest.fixture(scope="module")
def sig_data():
    rng = np.random.default_rng(1234)
    n = 600
    x = rng.normal(size=n)
    grp = np.repeat(["a", "b"], n // 2)
    prob = np.where(grp == "a", 0.9, 0.1)
    return pl.DataFrame(
        {
            "x": x,
            "y": 2 * x + rng.normal(scale=0.5, size=n),
            "z": rng.normal(size=n),
            "w": x + rng.normal(scale=0.3, size=n),
            "num": np.where(grp == "a", 0.0, 3.0) + rng.normal(size=n),
            "grp": grp,
            "yes": np.where(rng.uniform(size=n) < prob, "yes", "no"),
            "g": np.where(x > -0.5, "hi", "lo"),
        }
    )


@pytest.fixture(scope="module")
def choc():
    return pl.read_parquet(CHOC)


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


def test_format_pval_rule():
    assert format_pval(0.0) == "< .001"
    assert format_pval(1e-300) == "< .001"
    assert format_pval(0.000999) == "< .001"
    assert format_pval(0.001) == "0.001"
    assert format_pval(0.0234567, 3) == "0.023"
    assert format_pval(0.0234567, 2) == "0.02"
    assert format_pval(np.float64(1e-20)) == "< .001"


def test_format_pval_expr_matches_scalar():
    vals = [0.0, 1e-300, 0.000999, 0.001, 0.0234567, 0.05, 0.12345, 0.99999, 1.0]
    for dec in (2, 3, 4):
        df = pl.DataFrame({"p.value": vals})
        vec = df.with_columns(format_pval_expr("p.value", dec))["p.value"].to_list()
        assert vec == [format_pval(p, dec) for p in vals]
        assert format_pvals(vals, dec) == [format_pval(p, dec) for p in vals]
    # stored values remain numeric
    assert df["p.value"].dtype == pl.Float64


def test_parser_finds_text_and_table_pvalues():
    out = (
        "┌─────┬─────────┐\n│ a   ┆ p.value │\n╞═════╪═════════╡\n"
        "│ 1   ┆ < .001  │\n│ 2   ┆ 0.21    │\n└─────┴─────────┘\n"
        "Chi-squared: 3.2 df(1), p.value 0.074\n"
    )
    assert pvalue_cells(out) == ["< .001", "0.21", "0.074"]


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------


def test_single_mean(sig_data):
    sm = single_mean(sig_data, var="num", comp_value=-1)
    check_display(capture(sm.summary))
    assert isinstance(sm.p_val, float) and sm.p_val < 0.001


@pytest.mark.parametrize("alt", ["two-sided", "greater"])
@pytest.mark.parametrize("test_type", ["z", "binomial"])
def test_single_prop(sig_data, alt, test_type):
    sp = single_prop(
        sig_data.filter(pl.col("grp") == "a"),
        var="yes",
        lev="yes",
        alt_hyp=alt,
        test_type=test_type,
    )
    check_display(capture(sp.summary))


def test_single_prop_z_pvalues_are_valid_and_do_not_underflow():
    """Two-sided p uses |z| (was 2 * cdf(z) > 1 for z > 0); the upper tail
    uses the survival function (1 - cdf underflowed to exactly 0)."""
    from scipy import stats

    df = pl.DataFrame({"x": ["yes"] * 900 + ["no"] * 100})
    kw = dict(var="x", lev="yes", test_type="z", comp_value=0.5)
    two = single_prop(df, alt_hyp="two-sided", **kw)
    gt = single_prop(df, alt_hyp="greater", **kw)
    lt = single_prop(df, alt_hyp="less", **kw)
    z = two.z_score
    assert 0 < two.p_val < 0.001
    assert two.p_val == pytest.approx(2 * stats.norm.sf(abs(z)))
    assert 0 < gt.p_val == pytest.approx(stats.norm.sf(z))
    assert lt.p_val == pytest.approx(1.0)
    # the same z test matches the uncorrected chi-squared (prop.test) p-value
    assert two.p_val == pytest.approx(stats.chi2.sf(z**2, 1), rel=1e-6)


def test_compare_means(sig_data):
    cm = compare_means(sig_data, var1="grp", var2="num")
    check_display(capture(cm.summary))
    check_display(capture(cm.summary, extra=True))
    assert cm.comp_stats["p.value"].dtype == pl.Float64


def test_compare_props(sig_data):
    cp = compare_props(sig_data, var1="grp", var2="yes", lev="yes")
    check_display(capture(cp.summary))
    check_display(capture(cp.summary, extra=True))
    assert cp.comp_stats["p.value"].dtype == pl.Float64


def test_cross_tabs(sig_data):
    ct = cross_tabs(sig_data, var1="grp", var2="yes")
    cells = check_display(capture(ct.summary))
    assert cells == ["< .001"]


def test_goodness(sig_data):
    gf = goodness(sig_data, var="g")
    assert check_display(capture(gf.summary)) == ["< .001"]


def test_correlation_pvalue_matrix(sig_data):
    cr = correlation(sig_data.select("x", "y", "z", "w"))
    cells = check_display(capture(cr.summary), min_sig=3)
    # x-y, x-w, y-w are highly significant; the z pairs are not
    assert cells.count("< .001") == 3
    # stored matrix stays numeric
    assert cr.cp.dtype == float


# ---------------------------------------------------------------------------
# multivariate
# ---------------------------------------------------------------------------


def test_pre_factor_bartlett(sig_data):
    pf = pre_factor(sig_data, ["x", "y", "w", "z"])
    assert check_display(capture(pf.summary)) == ["< .001"]
    assert isinstance(pf.btest["p.value"], float)


def test_cbc_coefficients(choc):
    m = cbc({"choc": choc}, rvar="chosen", evar=EVAR, id="choice_id", alt="alt")
    cells = check_display(capture(m.summary), min_sig=4)
    assert len(cells) == m.coeff.height
    # survival function keeps tiny p-values positive (1 - cdf gave 0.0 for |z|>8)
    assert (m.coeff["p.value"] > 0).all()
    assert m.coeff["p.value"].dtype == pl.Float64


def test_cbc_pvalues_do_not_underflow():
    from scipy.stats import norm

    from pyrsm.multivariate.cbc import _mnl_fit

    rng = np.random.default_rng(7)
    ntask, nalt = 3000, 3
    X = rng.normal(size=(ntask * nalt, 1))
    codes = np.repeat(np.arange(ntask), nalt).astype(np.intp)
    util = 3 * X[:, 0] + rng.gumbel(size=ntask * nalt)
    y = np.zeros(ntask * nalt)
    y[np.arange(ntask) * nalt + util.reshape(ntask, nalt).argmax(axis=1)] = 1
    fit = _mnl_fit(X, y, codes, ntask)
    assert abs(fit["zvalues"][0]) > 9  # 1 - norm.cdf(|z|) is exactly 0 here
    assert fit["pvalues"][0] > 0
    assert fit["pvalues"][0] == pytest.approx(2 * norm.sf(abs(fit["zvalues"][0])))


def test_conjoint_choice(choc):
    cc = conjoint_choice({"choc": choc}, rvar="chosen", evar=EVAR, id="choice_id")
    cells = check_display(capture(cc.summary), min_sig=5)
    # coefficient table plus the model chi-squared line
    assert len(cells) == cc.coef.height + 1
    assert cells[-1] == "< .001"


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["regress", "logistic"])
def test_model_summary(sig_data, kind):
    """Coefficient table, model-fit line (model_fit) and the F / chi-squared
    model comparison all show "< .001"."""
    if kind == "regress":
        m = regress(sig_data, rvar="y", evar=["x", "z"])
        out = capture(m.summary, test="x")
    else:
        m = logistic(sig_data, rvar="yes", lev="yes", evar=["grp", "x"])
        out = capture(m.summary, test="grp")
    # coefficients + model fit + model comparison
    cells = check_display(out, min_sig=3)
    assert cells[-2:] == ["< .001", "< .001"]
