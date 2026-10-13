"""The formatted report of each Basics tool (see :mod:`pyrsm.report`).

Radiant shows these in its Summary tabs and Chat cards; ``summary(plain=False)``
shows the same report in a notebook or a rendered Quarto report. Each builder
reads the header pyrsm prints (so data scope lines come along) and formats the
tool's own result frames -- nothing is recomputed here.
"""

from __future__ import annotations

from typing import Any

from pyrsm.basics.utils import ci_label
from pyrsm.report import (
    build_report,
    capture,
    format_p_value,
    frame_table,
    model_header,
    table,
    text_header,
    text_lines,
)
from pyrsm.utils import sig_stars

#: pyrsm names the stars column with blanks; the report keys it "".
_STARS = {"", " ", "  "}

#: compare_means comparison columns without ``extra`` (``_summary_plain``).
_BRIEF = ("Null hyp.", "Alt. hyp.", "diff", "p.value", "")


def _plain(model: Any, **kwargs: Any) -> str:
    """The text ``summary(plain=True)`` prints, header included."""
    return capture(model.summary, plain=True, **kwargs)


def _item(value: Any) -> Any:
    """numpy scalars as plain Python numbers (a float stays a float)."""
    return value.item() if hasattr(value, "item") else value


def single_mean_report(sm: Any, dec: int = 3) -> dict[str, Any]:
    """Descriptive statistics and the test row, under pyrsm's header."""
    descriptive = {
        "mean": float(sm.mean), "n": int(sm.n), "n_missing": int(sm.n_missing),
        "sd": float(sm.sd), "se": float(sm.se), "me": float(sm.me),
    }
    # bound names follow the test: 5% / 100% for a one-sided 95% test
    low, high = ci_label(getattr(sm, "alt_hyp", "two-sided"), float(getattr(sm, "conf", 0.95)))
    comparison = {
        "diff": float(sm.diff), "se": float(sm.se), "t.value": float(sm.t_val),
        "p.value": float(sm.p_val), "df": int(sm.df),
        low: float(sm.ci[0]), high: float(sm.ci[1]), "": sig_stars([sm.p_val])[0],
    }
    return build_report(
        "single_mean",
        downloads=False,
        header=text_header(_plain(sm, dec=dec)),
        tables=[
            table("descriptive", [descriptive], title="Descriptive statistics",
                  subtitle=f"Variable: {sm.var}", dec=dec),
            table("comparison", [comparison], title="Test",
                  columns=["diff", "se", "t.value", "p.value", "df", low, high, ""], dec=dec),
        ],
    )


def compare_means_report(cm: Any, extra: bool = False, dec: int = 3) -> dict[str, Any]:
    """"Descriptive Statistics" and "Pairwise Comparisons", as pyrsm's styled view.

    Without ``extra`` the comparisons keep the brief columns ``summary()`` prints.
    """
    descriptive = cm.descriptive_stats
    desc = table(
        "descriptive",
        descriptive.iter_rows(named=True),
        title="Descriptive Statistics",
        subtitle=f"Variable: {cm.var2} by {cm.var1}",
        columns=list(descriptive.columns),
        terms=(descriptive.columns[0],),
        dec=dec,
    )
    comparisons = cm.comp_stats
    rows = [{("" if k in _STARS else k): v for k, v in row.items()} for row in comparisons.iter_rows(named=True)]
    columns = ["" if c in _STARS else c for c in comparisons.columns]
    if not extra:
        columns = [c for c in columns if c in _BRIEF]
    comp = table(
        "comparisons",
        rows,
        title=f"Pairwise Comparisons ({cm.test_type})",
        subtitle=f"Samples: {cm.sample_type}, Confidence: {cm.conf}",
        columns=columns,
        terms=("Null hyp.", "Alt. hyp."),
        dec=dec,
    )
    return build_report("compare_means", downloads=False, header=model_header(cm), tables=[desc, comp])


def single_prop_report(sp: Any, dec: int = 3) -> dict[str, Any]:
    """Descriptive statistics and the test row (binomial: ns; z-test: z.value)."""
    descriptive = {
        "p": float(sp.p), "ns": int(sp.ns), "n": int(sp.n), "n_missing": int(sp.n_missing),
        "sd": float(sp.sd), "se": float(sp.se), "me": float(sp.me),
    }
    comp: dict[str, Any] = {"diff": float(sp.diff)}
    if getattr(sp, "test_type", "binomial") == "binomial":
        comp["ns"] = int(sp.ns)
    else:
        comp["z.value"] = float(sp.z_score)
    comp["p.value"] = float(sp.p_val)
    low, high = ci_label(getattr(sp, "alt_hyp", "two-sided"), float(getattr(sp, "conf", 0.95)))
    comp[low] = float(sp.ci[0])
    comp[high] = float(sp.ci[1])
    return build_report(
        "single_prop",
        downloads=False,
        header=text_header(_plain(sp, dec=dec)),
        tables=[
            table("descriptive", [descriptive], title="Descriptive statistics",
                  subtitle=f'Variable: {sp.var}, level: "{sp.lev}"', dec=dec),
            table("comparison", [{**comp, "": sig_stars([sp.p_val])[0]}], title="Test",
                  columns=[*comp.keys(), ""], dec=dec),
        ],
    )


def compare_props_report(cp: Any, extra: bool = False, dec: int = 3) -> dict[str, Any]:
    """pyrsm's two tables. Without ``extra`` the test table keeps the
    hypotheses, diff, p.value and stars, as pyrsm's own summary does."""
    comp = cp.comp_stats
    names = list(comp.columns)
    stars = names[-1]  # pyrsm's stars column, the last one (blank header)
    keep = names if extra else [names[0], names[1], names[2], names[3], names[-1]]
    rows = [{("" if k == stars else k): v for k, v in row.items()} for row in comp.iter_rows(named=True)]
    test = table(
        "comparisons", rows, title="Pairwise comparisons",
        columns=["" if k == stars else k for k in keep],
        terms=("Null hyp.", "Alt. hyp."), dec=dec,
    )
    desc = frame_table("descriptive", cp.descriptive_stats, title="Descriptive statistics",
                       subtitle=f'Level: "{cp.lev}" in {cp.var2}', dec=dec)
    header = text_header(_plain(cp, extra=extra, dec=dec))
    return build_report("compare_props", downloads=False, header=header, tables=[desc, test])


#: goodness output option -> (attribute, title, subtitle) in pyrsm's order and words
GOODNESS_TABLES = [
    ("observed", "observed", "Observed", ""),
    ("expected", "expected", "Expected", "total x p"),
    ("chisq", "chisq", "Contribution to chi-squared", "(observed - expected) ^ 2 / expected"),
    ("dev_std", "stdev", "Deviation standardized", "(observed - expected) / sqrt(expected)"),
]


def goodness_report(gf: Any, output: list[str] | str = ("observed", "expected"), dec: int = 2) -> dict[str, Any]:
    """The tables selected in ``output``, one row per level; the test line below."""
    output = [output] if isinstance(output, str) else list(output)
    text = _plain(gf, output=output, dec=dec)
    tables = [
        frame_table(option, getattr(gf, attr), terms=(), title=title, subtitle=subtitle, dec=dec)
        for option, attr, title, subtitle in GOODNESS_TABLES
        if option in output
    ]
    return build_report("goodness", downloads=False, header=text_header(text), tables=tables,
                        fit=text_lines(text, "Chi-squared:"))


#: cross_tabs output option -> (title, subtitle, percentages?) in pyrsm's order and words
CROSS_TABS_TABLES = [
    ("observed", "Observed", "", False),
    ("expected", "Expected", "(row total x column total) / total", False),
    ("chisq", "Contribution to chi-squared", "(o - e)^2 / e", False),
    ("dev_std", "Deviation standardized", "(o - e) / sqrt(e)", False),
    ("perc_row", "Row percentages", "", True),
    ("perc_col", "Column percentages", "", True),
    ("perc", "Percentages", "", True),
]


def cross_tabs_report(ct: Any, output: list[str] | str = ("observed", "expected"), dec: int = 2) -> dict[str, Any]:
    """The tables selected in ``output``; pyrsm's chi-squared lines below them."""
    output = [output] if isinstance(output, str) else list(output)
    text = _plain(ct, output=output, dec=dec)
    tables = []
    for option, title, subtitle, pct in CROSS_TABS_TABLES:
        if option not in output:
            continue
        frame = getattr(ct, option)
        tables.append(frame_table(option, frame, title=title, subtitle=subtitle, dec=dec,
                                  percent=frame.columns[1:] if pct else ()))
    fit = text_lines(text, "Chi-squared:", contains=("of cells have expected values",))
    return build_report("cross_tabs", downloads=False, header=text_header(text), tables=tables, fit=fit)


def correlation_report(cr: Any, cov: bool = False, cutoff: float = 0, dec: int = 2) -> dict[str, Any]:
    """pyrsm's lower-triangle displays (cutoff applied, already strings)."""
    corr, _pvals, covs = cr._build_matrix_displays(cutoff, dec)
    names = list(corr.columns[1:])

    def matrix(id: str, rows: Any, title: str, subtitle: str = "") -> dict[str, Any]:
        cols = [{"key": "", "label": "", "kind": "term"}, *names]
        return table(id, rows, columns=cols, title=title, subtitle=subtitle, dec=dec)

    # Same cells as pyrsm's p-value display (lower triangle, cutoff applied),
    # styled like every other report: "< .001" rather than a rounded 0.0.
    p_rows = []
    for i, row in enumerate(_pvals.iter_rows(named=True)):
        out = {"": row[""]}
        for j, name in enumerate(names):
            out[name] = "" if row[name] == "" else format_p_value(float(cr.cp[i + 1, j]), 3)
        p_rows.append(out)

    tables = [matrix("correlation", corr.iter_rows(named=True), "Correlation matrix", f"Method: {cr.method}")]
    sections = []
    if cr.method != "polychoric" or getattr(cr, "hcor_se", False):
        tables.append(matrix("p_values", p_rows, "p.values"))
        if cov:
            tables.append(matrix("covariance", covs.iter_rows(named=True), "Covariance matrix"))
    elif cov:
        sections.append({"id": "covariance", "title": "Covariance matrix", "lines": [
            "Not available for the polychoric method: heterogeneous correlations are "
            "estimated from latent variables, so a covariance matrix on the original "
            "scale is not defined."]})
    header = text_header(_plain(cr, cov=cov, cutoff=cutoff, dec=dec))
    return build_report("correlation", downloads=False, header=header, tables=tables, sections=sections)


def prob_calc_report(pc: Any, dec: int = 3) -> dict[str, Any]:
    """Header as pyrsm prints it; the probability lines as a two-column table."""
    rows = [
        {"event": key, "probability": value}
        for key, value in pc.summary(dec=dec, ret=True).items()
        if key.startswith(("P(", "1 - P(")) and value is not None
    ]
    tables = [table("probabilities", rows, title="Probabilities", terms=("event",),
                    labels={"event": ""}, dec=dec)] if rows else []
    return build_report("prob_calc", downloads=False, header=text_header(_plain(pc, dec=dec)), tables=tables)


def clt_report(sim: Any, stat: str = "sum", dec: int = 3) -> dict[str, Any]:
    """Setup as pyrsm prints it; simulated vs. CLT mean and st. dev as a table."""
    text = _plain(sim, stat=stat, dec=dec)
    st = sim.stats(stat)
    label = "sums" if stat == "sum" else "means"
    rows = [
        {"": "Simulated", "Mean": _item(st["simulated"]["mean"]), "St. dev": _item(st["simulated"]["sd"])},
        {"": "CLT", "Mean": _item(st["clt"]["mean"]), "St. dev": _item(st["clt"]["sd"])},
    ]
    # pyrsm's one-line takeaway ("CLT: with a sample size of ...") under the table
    note = next((line for line in text.splitlines() if line.startswith("CLT:")), "")
    stats = table("stats", rows, title=f"Sample {label}", columns=[
        {"key": "", "label": "", "kind": "term"}, "Mean", "St. dev"], dec=dec, note=note)
    return build_report("clt", downloads=False, header=text_header(text), tables=[stats])
