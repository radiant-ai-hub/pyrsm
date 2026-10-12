"""``summary(plain=False)`` in a notebook shows Radiant's formatted report.

:mod:`pyrsm.report` builds the report Radiant's Summary tab renders and draws it
with the same markup and CSS; each Basics tool's ``summary(plain=False)``
displays it in a notebook (and so in a rendered Quarto report).
"""

from pathlib import Path

import polars as pl
import pytest

import pyrsm as rsm
import pyrsm.report as report

DATA = Path(__file__).resolve().parent.parent / "examples" / "data"
salary = pl.read_parquet(DATA / "basics" / "salary.parquet")
titanic = pl.read_parquet(next(DATA.rglob("titanic.parquet")))


class _Kernel:
    """What ``du.is_notebook`` looks for: an IPython kernel app."""

    config = {"IPKernelApp": {}}


@pytest.fixture
def shown(monkeypatch):
    """Pretend to be a notebook and collect what would be displayed.

    Patched at the IPython level, so it holds even when another test has
    re-imported pyrsm (test_import_performance clears it from sys.modules).
    """
    import IPython
    import IPython.display

    out = []
    monkeypatch.setattr(IPython, "get_ipython", lambda: _Kernel())
    monkeypatch.setattr(IPython.display, "display", lambda *objs, **kw: out.extend(objs))
    return out


CASES = {
    "single_mean": (lambda: rsm.basics.single_mean({"salary": salary}, var="salary", comp_value=100000),
                    {}, ["Descriptive statistics", "Test"]),
    "compare_means": (lambda: rsm.basics.compare_means({"salary": salary}, var1="rank", var2="salary"),
                      {"extra": True}, ["Descriptive Statistics", "Pairwise Comparisons (t-test)", "t.value"]),
    "single_prop": (lambda: rsm.basics.single_prop({"titanic": titanic}, var="survived", lev="Yes"),
                    {}, ["Descriptive statistics", "Test"]),
    "compare_props": (lambda: rsm.basics.compare_props({"titanic": titanic}, var1="pclass", var2="survived", lev="Yes"),
                      {}, ["Descriptive statistics", "Pairwise comparisons"]),
    "goodness": (lambda: rsm.basics.goodness({"titanic": titanic}, var="pclass"),
                 {"output": ["observed", "expected", "chisq"]}, ["Observed", "Expected", "Contribution to chi-squared", "Chi-squared"]),
    "cross_tabs": (lambda: rsm.basics.cross_tabs({"titanic": titanic}, var1="pclass", var2="survived"),
                   {"output": ["observed", "perc_row"]}, ["Observed", "Row percentages", "%"]),
    "correlation": (lambda: rsm.basics.correlation({"salary": salary}, vars=["salary", "yrs_since_phd", "yrs_service"]),
                    {}, ["Correlation matrix", "p.values"]),
    "prob_calc": (lambda: rsm.basics.prob_calc("norm", mean=0, sd=1, lb=-1.96, ub=1.96),
                  {}, ["Probabilities", "P(X &lt; -1.96)"]),
    "clt": (lambda: rsm.basics.clt("Uniform", n=10, m=200, seed=1), {"stat": "mean"}, ["Sample means", "Simulated", "CLT"]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_summary_plain_false_shows_the_formatted_report(name, shown, capsys):
    build, kwargs, expected = CASES[name]
    model = build()
    model.summary(plain=False, **kwargs)
    assert capsys.readouterr().out == "", "the formatted report replaces the printed text"
    assert len(shown) == 1
    html = shown[0]._repr_html_()
    assert "|" in shown[0]._repr_markdown_()
    assert html.startswith('<section class="pyrsm-report"') and "<style>" in html
    for text in expected:
        assert text in html, f"{name}: {text!r} missing"
    # the text form is pyrsm's plain summary
    plain = report.capture(model.summary, plain=True, **kwargs)
    assert repr(shown[0]) == plain


def test_compare_means_extra_adds_columns(shown):
    cm = rsm.basics.compare_means({"salary": salary}, var1="rank", var2="salary")
    from pyrsm.basics.report import compare_means_report

    brief = [c["key"] for c in compare_means_report(cm)["tables"][1]["columns"]]
    full = [c["key"] for c in compare_means_report(cm, extra=True)["tables"][1]["columns"]]
    assert brief == ["Null hyp.", "Alt. hyp.", "diff", "p.value", "stars"]
    assert {"se", "t.value", "df"} <= set(full)


def test_p_values_read_like_radiant():
    assert report.format_p_value(0.00001) == "< 0.001"
    assert report.format_p_value(0.0456) == "0.046"
    assert report.format_p_value("< .001") == "< 0.001"


def test_outside_a_notebook_plain_false_prints_text(capsys):
    rsm.basics.single_mean({"salary": salary}, var="salary").summary(plain=False)
    assert "Single mean test" in capsys.readouterr().out
