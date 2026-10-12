"""Radiant's Summary reports for the Model tools.

The same builders Radiant's tool workspaces use, so ``summary(plain=False)``
in a notebook or a Quarto report shows the tables a student sees in the app
(:mod:`pyrsm.report` renders them):

- :func:`fit_report` -- random forest, XGBoost, neural network: pyrsm's header,
  then a "Model fit" table (:func:`fit_metrics`);
- :func:`crtree_report` -- the classification / regression tree, node by node;
- :func:`evaluate_report` -- Model > Evaluate: pyrsm's header, then its tables;
- :func:`linear_report`, :func:`logistic_report`, :func:`mnl_report` -- the
  regression summaries: coefficients, fit and every optional section.
"""

from __future__ import annotations

from typing import Any, Sequence

from pyrsm.report import SIG_CODES, build_report, column, format_p_value, frame_table, table, text_header
from pyrsm.report import capture as _capture
from pyrsm.report import model_header as _header

# ---------------------------------------------------------------------------
# Random forest, XGBoost, neural network
# ---------------------------------------------------------------------------

#: Header lines that are results, not settings: they move into the fit table.
_FIT_LINES = {"AUC", "Model fit"}

NOTES = {
    "oob": (
        "Out-of-bag (OOB): each row is scored by the trees that did not use it, a fair "
        "estimate of performance on new data. rf.predict() with no data returns these "
        "OOB predictions; predicting the estimation data itself would be overoptimistic."
    ),
    "in_sample": (
        "Computed on the estimation rows themselves, so it overstates performance on "
        "new data. Evaluate on a test set (Data > View filter, or Model > Evaluate) "
        "before reporting it."
    ),
}


def _regression_fit(y_true, y_pred, scale: float = 1.0) -> dict[str, float]:
    import numpy as np

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    mse = float(np.mean((y_true - y_pred) ** 2)) * scale**2
    return {"r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0, "mse": mse, "rmse": float(np.sqrt(mse))}


def _auc(y, prob) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, prob))
    except Exception:
        return None


def fit_metrics(model: Any) -> dict[str, Any]:
    """How well a fitted ``rforest``, ``xgboost`` or ``mlp`` fits: ``nobs`` and
    AUC (classification) or R², MSE and RMSE (regression), out-of-bag for a
    random forest fitted with ``oob_score=True`` and on the estimation rows
    otherwise (``auc_kind`` / ``fit_kind``)."""
    kind = type(model).__name__
    metrics: dict[str, Any] = {"nobs": int(model.nobs)}
    classification = model.mod_type == "classification"
    X = model.data_onehot.to_pandas()
    if kind == "mlp":
        # The network is fitted on standardized data: errors in the response's units.
        y = model.data_std[model.rvar].to_numpy()
        if classification:
            try:
                metrics["auc"] = _auc(y, model.fitted.predict_proba(X)[:, -1])
            except Exception:
                metrics["auc"] = None
        else:
            stds = getattr(model, "stds", None)
            scale = float(stds[model.rvar]) if stds is not None and model.rvar in stds else 1.0
            fit = _regression_fit(y, model.fitted.predict(X), scale)
            fit["r2"] = float(model.fitted.score(X, y))
            metrics.update(fit)
        return metrics

    y = model.data.get_column(model.rvar).to_numpy()
    oob = kind == "rforest" and bool(getattr(model, "oob_score", False))
    if classification:
        try:
            if oob and hasattr(model.fitted, "oob_decision_function_"):
                prob, how = model.fitted.oob_decision_function_[:, 1], "oob"
            else:
                prob, how = model.fitted.predict_proba(X)[:, -1], "in_sample"
            metrics["auc"] = _auc(y, prob)
            metrics["auc_kind"] = how if metrics["auc"] is not None else "unavailable"
        except Exception:
            metrics["auc"], metrics["auc_kind"] = None, "unavailable"
    else:
        if oob and hasattr(model.fitted, "oob_prediction_"):
            pred, metrics["fit_kind"] = model.fitted.oob_prediction_, "oob"
        else:
            pred, metrics["fit_kind"] = model.fitted.predict(X), "in_sample"
        metrics.update(_regression_fit(y, pred))
    if kind == "xgboost":
        metrics.setdefault("fit_kind", "in_sample")
    return metrics


def fit_report(kind: str, summary_text: str, metrics: dict[str, Any], *, dec: int = 3) -> dict[str, Any]:
    """The report for ``kind`` (``rforest``, ``xgboost``, ``mlp``) from pyrsm's
    text: its header -- data, filter, response, level, variables and every
    setting -- then a "Model fit" table, with the note that says how the fit
    was measured under it."""
    header = text_header(summary_text)
    header = {**header, "meta": [m for m in header["meta"] if m["label"] not in _FIT_LINES]}

    how = str(metrics.get("auc_kind") or metrics.get("fit_kind") or "in_sample")
    how = "oob" if how == "oob" else "in_sample"
    suffix = " (OOB)" if how == "oob" else ""
    row: dict[str, Any] = {"n": metrics.get("nobs")}
    if metrics.get("auc") is not None:
        row[f"AUC{suffix}"] = metrics["auc"]
    for key, label in (("r2", "R²"), ("rmse", "RMSE"), ("mse", "MSE"), ("mae", "MAE")):
        if metrics.get(key) is not None:
            row[f"{label}{suffix}"] = metrics[key]
    fit = table("fit", [row], title="Model fit", dec=dec, note=NOTES[how])
    return build_report(kind, header=header, tables=[fit], downloads=False)


# ---------------------------------------------------------------------------
# Classification and regression trees
# ---------------------------------------------------------------------------


def _pretty(split: str) -> str:
    return split.replace(" >= ", " ≥ ").replace(" <= ", " ≤ ")


def _number(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.2f}"


def _guides(rows: list[dict[str, Any]]) -> list[str]:
    """Connector prefix per node: a ``│`` for each open ancestor level, then
    ``├─`` or ``└─`` (second child). The root has none."""
    out: list[str] = []
    open_levels: list[bool] = []  # per depth: does that ancestor have a sibling below?
    for row in rows:
        depth = row["depth"]
        if depth == 0:
            out.append("")
            open_levels = []
            continue
        open_levels = open_levels[: depth - 1]
        prefix = "".join("│  " if is_open else "   " for is_open in open_levels)
        out.append(prefix + ("└─ " if row["last"] else "├─ "))
        open_levels.append(not row["last"])
    return out


#: A regression tree's SSE column: each node's sum of squared errors, total = 100.
SSE_LABEL = "SSE (total = 100)"


def _percent(value: float | None) -> str:
    return "" if value is None else f"{100 * value:.1f}%"


def crtree_report(model: Any, summary_text: str) -> dict[str, Any]:
    """The crtree Summary: pyrsm's header lines, then the tree itself.

    Every node in rpart's print order, indented under its parent with ``├─`` /
    ``└─`` connectors, so the hierarchy reads as in the tree plot: terminal
    nodes carry the plot's green marker and inner nodes are muted, and the
    prediction is drawn as a bar behind its number.
    """
    nodes = list(model.node_table().iter_rows(named=True))
    classification = model.mod_type == "classification"
    pred_label = (
        f"P({model.rvar} = {model.lev})" if classification else f"Predicted {model.rvar}"
    )
    preds = [row["prediction"] for row in nodes]
    low, high = min(preds), max(preds)

    def bar(value: float) -> str:
        if classification:
            frac = value
        else:  # a regression tree's mean, scaled across the nodes
            frac = 0.02 + 0.98 * (value - low) / (high - low) if high > low else 1.0
        return f"{max(0.0, min(1.0, frac)):.4f}"

    levels = [str(level) for level in (model.ylevels or [])] if classification else []
    counts_label = " / ".join(levels)
    rows = []
    for row in nodes:
        out: dict[str, Any] = {
            "Node": _pretty(row["split"]) if row["depth"] else "All observations",
            "n": f"{row['n']:,}",
        }
        if classification:
            out[counts_label] = " / ".join(f"{c:,}" for c in row["counts"])
        out["% of obs"] = f"{100 * row['share']:.1f}%"
        out[pred_label] = (
            f"{100 * row['prediction']:.2f}%" if classification else _number(row["prediction"])
        )
        if not classification:
            out[SSE_LABEL] = f"{100 * row['sse'] / nodes[0]['sse']:.1f}"
        out["Split removes"] = _percent(row["removed"])
        rows.append(out)

    prior = (
        classification
        and getattr(model, "prior", None) is not None
        and getattr(model, "cost", None) is None
    )
    if classification:
        note = (
            "Each node is indented under the node it splits from; green marks a "
            "terminal node, as in the tree plot. Split removes: how much splitting the "
            "node lowers the Gini impurity (n × Gini, parent minus its two "
            "children) as a share of the impurity at the root; Gini = 2 × p × (1 − p)."
            + (
                " With a prior, probabilities are adjusted back to the data's base "
                "rate and the impurity uses the prior-weighted shares, as rpart does."
                if prior
                else ""
            )
        )
    else:
        total = nodes[0]["sse"]
        total_text = f"{total:,.0f}" if abs(total) >= 100 else f"{total:,.4g}"
        note = (
            "Each node is indented under the node it splits from; green marks a "
            "terminal node, as in the tree plot. SSE: the node's "
            f"sum of squared errors, with the total SSE of {total_text} set to "
            "100. SSE adds up: a node's SSE is its two children's SSE plus what "
            "its split removes. Split removes: the drop in SSE from splitting the "
            "node; the splits add up to the tree's R²."
        )
    columns: list[Any] = [column("Node", "Node", "tree"), "n"]
    if classification:
        columns.append(counts_label)
    columns += ["% of obs", column(pred_label, pred_label, "bar")]
    if not classification:
        columns.append(SSE_LABEL)
    columns.append("Split removes")
    tree_table = table("tree", rows, title="Tree", columns=columns, note=note)
    # Row metadata the renderer reads but does not show as columns.
    for out, row, guide in zip(tree_table["rows"], nodes, _guides(nodes)):
        out["_guide"] = guide
        out["_leaf"] = "1" if row["leaf"] else "0"
        out[f"{pred_label}__bar"] = bar(row["prediction"])
    return build_report("crtree", header=text_header(summary_text), tables=[tree_table])


# ---------------------------------------------------------------------------
# Model > Evaluate
# ---------------------------------------------------------------------------


def evaluate_report(kind: str, summary_text: str, frames: Sequence[tuple[Any, str]]) -> dict[str, Any]:
    """One Model > Evaluate tab: pyrsm's printed header, then its tables from
    the ``(frame, title)`` pairs."""
    return build_report(
        kind,
        header=text_header(summary_text),
        tables=[frame_table(f"t{i + 1}", frame, title=title) for i, (frame, title) in enumerate(frames)
                if frame is not None],
    )


# ---------------------------------------------------------------------------
# Linear regression, logistic regression, multinomial logit
# ---------------------------------------------------------------------------
#
# The notebook layout of each model's summary, built from pyrsm's own objects
# and helpers -- the header pyrsm prints, ``coef``, ``format_p_value``,
# ``model_fit``, ``vif``, ``coef_ci`` / ``or_ci`` and the model tests -- so the
# app and a notebook show the same tables.


def _counted(report: dict[str, Any]) -> dict[str, Any]:
    """Every table says how many rows it has (as :func:`pyrsm.report.build_report`)."""
    tables = list(report.get("tables") or []) + [
        s["table"] for s in report.get("sections") or [] if isinstance(s, dict) and isinstance(s.get("table"), dict)
    ]
    for t in tables:
        t["total_rows"] = len(t.get("rows") or [])
    return report


STARS_COLUMN = "  "


def _fixed(value: Any, dec: int) -> str:
    """``dec`` fixed decimals with a comma as the thousands separator."""
    try:
        # + 0.0 turns a -0.0 into 0.0; then a tiny negative that rounds to
        # zero (a standardized intercept, -1e-17) reads 0.000, not -0.000
        return f"{round(float(value), dec) + 0.0:,.{dec}f}"
    except (TypeError, ValueError):
        return "" if value is None else str(value)


def _column(key: str, label: str, kind: str = "number") -> dict[str, str]:
    return {"key": key, "label": label, "kind": kind}


def _coefficient_table(model: Any, *, dec: int, logistic: bool) -> dict[str, Any]:
    if logistic:
        columns = [
            _column("index", "", "term"),
            _column("OR", "OR"),
            _column("OR%", "OR%"),
            _column("coefficient", "coefficient"),
            _column("std.error", "std.error"),
            _column("z.value", "z.value"),
            _column("p.value", "p.value"),
            _column("stars", "", "stars"),
        ]
        subtitle = f"Response: {model.rvar} (level: {model.lev})"
    else:
        columns = [
            _column("index", "", "term"),
            _column("coefficient", "coefficient"),
            _column("std.error", "std.error"),
            _column("t.value", "t.value"),
            _column("p.value", "p.value"),
            _column("stars", "", "stars"),
        ]
        subtitle = f"Response: {model.rvar}"

    rows = []
    for row in model.coef.iter_rows(named=True):
        out: dict[str, str] = {"index": str(row["index"])}
        for key in ("OR", "coefficient", "std.error", "t.value", "z.value"):
            if key in row:
                out[key] = _fixed(row[key], dec)
        if "OR%" in row:
            # Same rounding pyrsm uses for OR% in both of its presentations;
            # blank for the intercept, which has no odds ratio (radiant.model).
            if row["OR%"] is None:
                out["OR%"] = ""
            else:
                pct = round(float(row["OR%"]), max(dec - 2, 0))
                out["OR%"] = f"{pct}%"
        out["p.value"] = format_p_value(row["p.value"], dec)
        out["stars"] = str(row.get(STARS_COLUMN) or "").strip()
        rows.append(out)
    return {
        "id": "coefficients",
        "title": "Coefficient Estimates",
        "subtitle": subtitle,
        "columns": columns,
        "rows": rows,
        "note": f"Signif. codes:  {SIG_CODES}",
    }


def _pandas_table(frame: Any, *, id: str, title: str, dec: int) -> dict[str, Any]:
    """A pandas table pyrsm prints with ``to_string()``, as rows of strings.

    Floats get pyrsm's fixed decimals (``to_string`` pads 0.28 to 0.280);
    strings pyrsm already formatted (``< .001``, ``-72.0%``) pass through.
    """
    columns = [_column("index", "", "term")]
    for name in frame.columns:
        label = str(name)
        if not label.strip():
            columns.append(_column("stars", "", "stars"))
        else:
            # coef_ci/or_ci call it "p.values"; the coefficient table says "p.value".
            columns.append(_column(label, "p.value" if label == "p.values" else label))
    rows = []
    for index, values in zip(frame.index, frame.itertuples(index=False)):
        out = {"index": _clean_term(str(index))}
        for column, value in zip(columns[1:], values):
            if column["key"] in {"p.value", "p.values"}:
                out[column["key"]] = format_p_value(value, dec)
            elif isinstance(value, float):
                out[column["key"]] = _fixed(value, dec)
            else:
                out[column["key"]] = "" if value is None else str(value).strip()
        rows.append(out)
    return {"id": id, "title": title, "columns": columns, "rows": rows}


def _clean_term(term: str) -> str:
    """``clarity[T.SI2]`` -> ``clarity[SI2]``, matching pyrsm's coefficient table."""
    return term.replace("[T.", "[")


def _text_section(id: str, title: str, text: str) -> dict[str, Any] | None:
    lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None
    return {"id": id, "title": title, "lines": lines}


def _fit_lines(fitted: Any, dec: int, model: Any = None) -> list[str]:
    from pyrsm.model.model import model_fit

    lines = [line for line in str(model_fit(fitted, dec=dec)).splitlines() if line.strip()]
    # R's note when columns were dropped for perfect multicollinearity
    note = model.collinearity_note() if getattr(model, "aliased", None) else ""
    return lines + note.splitlines()


def _not_calculated(section_id: str, title: str, line: str) -> dict[str, Any]:
    """radiant.model skips VIF / intervals / odds ratios for an aliased model."""
    return {"id": section_id, "title": title, "lines": [line]}


def _step_cell(column: str, value: Any) -> str:
    """A number in the stepwise tables: Df as an integer, the rest with 2 decimals."""
    if value is None:
        return ""
    if column == "Df":
        return str(int(round(float(value))))
    return f"{float(value):,.2f}"


def _stepwise_section(model: Any) -> dict[str, Any] | None:
    """The stepwise selection, one table per step, collapsed under its title.

    pyrsm's ``step_steps``: the AIC of the model at each step and every change
    considered (``- term`` drops it, ``+ term`` adds it), best first. Older
    objects without it fall back to the text trace.
    """
    steps = getattr(model, "step_steps", None) or []
    if not steps:
        trace = getattr(model, "step_trace", "")
        if not trace:
            return None
        return {"id": "stepwise", "title": "Stepwise selection (AIC)", "lines": trace.splitlines(), "preformatted": True, "collapsed": True}
    tables = []
    for i, step in enumerate(steps):
        names = list(step["columns"])
        tables.append(
            {
                "id": f"stepwise-{i}",
                "title": f"{'Start' if i == 0 else f'Step {i}'}: AIC = {step['aic']:,.2f}",
                "subtitle": step["formula"],
                "columns": [_column("term", "Change", "term")] + [_column(n, n) for n in names],
                "rows": [
                    {"term": row["term"], **{n: _step_cell(n, row.get(n)) for n in names}}
                    for row in step["rows"]
                ],
            }
        )
    changes = len(steps) - 1
    return {
        "id": "stepwise",
        "title": "Stepwise selection (AIC)",
        "collapsed": True,
        "lines": [
            (
                "No change lowered the AIC, so the starting model is kept. "
                if changes == 0
                else f"{changes} change{'' if changes == 1 else 's'} lowered the AIC; then no change did. "
            )
            + "Each table lists the changes considered (- drops a term, + adds one) by the AIC "
            "after the change; <none> keeps the model as it is."
        ],
        "tables": tables,
    }


def linear_report(
    reg: Any,
    *,
    vif: bool = False,
    ssq: bool = False,
    rmse: bool = False,
    test: str | None = None,
    ci: bool = False,
    conf: float = 0.95,
    dec: int = 3,
) -> dict[str, Any]:
    """The notebook layout of ``pyrsm.model.regress.summary`` with the same options."""
    from pyrsm.model import model as model_utils
    from pyrsm.utils import format_nr

    report: dict[str, Any] = {
        "kind": "linear_regression",
        **_header(reg),
        "tables": [_coefficient_table(reg, dec=dec, logistic=False)],
        "fit": _fit_lines(reg.fitted, dec, reg),
        "sections": [],
    }
    sections = report["sections"]
    fitted = reg.fitted

    aliased = bool(getattr(reg, "aliased", None))
    # Same order as pyrsm's summary(): vif, ssq, rmse, test, ci.
    if vif:
        if aliased:
            sections.append(_not_calculated("vif", "Variance inflation factors", "Multicollinearity diagnostics were not calculated."))
        elif reg.evar is None or len(reg.evar) < 2:
            sections.append({"id": "vif", "title": "Variance inflation factors", "lines": ["Variance Inflation Factors cannot be calculated"]})
        else:
            sections.append(
                {"table": _pandas_table(model_utils.vif(fitted), id="vif", title="Variance inflation factors", dec=dec), "id": "vif", "title": "Variance inflation factors"}
            )
    if ssq:
        df = [fitted.df_model, fitted.df_resid, fitted.df_model + fitted.df_resid]
        ss = [fitted.ess, fitted.ssr, fitted.centered_tss]
        rows = [
            {"index": label, "df": d, "SS": s}
            for label, d, s in zip(
                ["Regression", "Error", "Total"], format_nr(df, dec=0), format_nr(ss, dec=0)
            )
        ]
        sections.append(
            {
                "id": "ssq",
                "title": "Sum of squares",
                "table": {
                    "id": "ssq",
                    "title": "Sum of squares",
                    "columns": [_column("index", "", "term"), _column("df", "df"), _column("SS", "SS")],
                    "rows": rows,
                },
            }
        )
    if rmse:
        rmse_val, rsd_val = reg.rmse()
        sections.append(
            {
                "id": "rmse",
                "title": "Prediction error",
                "lines": [
                    f"Prediction error (RMSE): {round(rmse_val, dec)}",
                    f"Residual st.dev   (RSD): {round(rsd_val, dec)}",
                ],
            }
        )
    step = _stepwise_section(reg)
    if step:
        sections.append(step)
    if test:
        section = _text_section("test", "Model comparison (F-test)", _capture(reg.f_test, test=test, dec=dec))
        if section:
            sections.append(section)
    if ci and aliased:
        sections.append(_not_calculated("ci", "Confidence intervals", "Confidence intervals were not calculated."))
    elif ci:
        # Robust (HC1) intervals when the model uses robust standard errors.
        inference = getattr(reg, "fitted_inference", fitted)
        frame = model_utils.coef_ci(inference, alpha=1 - conf, dec=dec)
        sections.append({"id": "ci", "title": "Confidence intervals", "table": _pandas_table(frame, id="ci", title="Confidence intervals", dec=dec)})
    return _counted(report)


def logistic_report(
    model: Any,
    *,
    vif: bool = False,
    test: Any = None,
    ci: bool = False,
    odds: bool = False,
    conf: float = 0.95,
    dec: int = 3,
) -> dict[str, Any]:
    """The notebook layout of ``pyrsm.model.logistic.summary`` with the same options."""
    import numpy as np
    import pandas as pd
    from pyrsm.model import model as model_utils

    report: dict[str, Any] = {
        "kind": "logistic_regression",
        **_header(model),
        "tables": [_coefficient_table(model, dec=dec, logistic=True)],
        "fit": _fit_lines(model.fitted, dec, model),
        "sections": [],
    }
    sections = report["sections"]
    aliased = bool(getattr(model, "aliased", None))
    # Same order as pyrsm's summary(): vif, ci, odds, test.
    if vif:
        if aliased:
            sections.append(_not_calculated("vif", "Variance inflation factors", "Multicollinearity diagnostics were not calculated."))
        elif model.evar is None or len(model.evar) < 2:
            sections.append({"id": "vif", "title": "Variance inflation factors", "lines": ["Variance Inflation Factors cannot be calculated"]})
        else:
            frame = model_utils.gvif(model.fitted, dec=dec)
            sections.append({"id": "vif", "title": "Variance inflation factors", "table": _pandas_table(frame, id="vif", title="Variance inflation factors", dec=dec)})
    if aliased:
        if ci:
            sections.append(_not_calculated("ci", "Confidence intervals", "Confidence intervals were not calculated."))
        if odds:
            sections.append(_not_calculated("odds", "Odds ratios", "Odds ratios were not calculated"))
    elif ci or odds:
        low, high = [f"{round(100 * p, 4):g}%" for p in ((1 - conf) / 2, 1 - (1 - conf) / 2)]
        cis = model.conf_int(conf=conf)
        coef = model.fitted.params
        labels = [model_utils.r_label(i).replace("[T.", "[") for i in coef.index]
        if ci:
            frame = pd.DataFrame(
                {
                    "coefficient": coef.values,
                    low: cis.iloc[:, 0].values,
                    high: cis.iloc[:, 1].values,
                    "+/-": cis.iloc[:, 1].values - coef.values,
                },
                index=labels,
            )
            sections.append({"id": "ci", "title": "Confidence intervals", "table": _pandas_table(frame, id="ci", title="Confidence intervals", dec=dec)})
        if odds:
            keep = [i != "Intercept" for i in coef.index]
            orlab = "std odds ratio" if getattr(model, "standardize", 0) else "odds ratio"
            frame = pd.DataFrame(
                {
                    orlab: np.exp(coef.values),
                    low: np.exp(cis.iloc[:, 0].values),
                    high: np.exp(cis.iloc[:, 1].values),
                },
                index=labels,
            )[keep]
            sections.append({"id": "odds", "title": "Odds ratios", "table": _pandas_table(frame, id="odds", title="Odds ratios", dec=dec)})
    step = _stepwise_section(model)
    if step:
        sections.append(step)
    if test:
        section = _text_section("test", "Model comparison (Chi-squared test)", _capture(model.chisq_test, test=test, dec=dec))
        if section:
            sections.append(section)
    return _counted(report)


def mnl_report(
    model: Any,
    *,
    ci: bool = False,
    rrr: bool = False,
    confusion: bool = False,
    test: Any = None,
    conf: float = 0.95,
    dec: int = 3,
) -> dict[str, Any]:
    """The layout of ``pyrsm.model.mnl.summary`` with the same options.

    One coefficient row per response level and term; every level is compared
    with the base level, which the subtitle names.
    """
    import numpy as np

    columns = [
        _column("level", "level", "term"),
        _column("index", "", "term"),
        _column("RRR", "RRR"),
        _column("RRR%", "RRR%"),
        _column("coefficient", "coefficient"),
        _column("std.error", "std.error"),
        _column("z.value", "z.value"),
        _column("p.value", "p.value"),
        _column("stars", "", "stars"),
    ]
    rows = []
    for row in model.coef.iter_rows(named=True):
        out = {"level": str(row["level"]), "index": str(row["index"])}
        for key in ("RRR", "coefficient", "std.error", "z.value"):
            out[key] = "" if row[key] is None else _fixed(row[key], dec)
        out["RRR%"] = "" if row["RRR%"] is None else f"{round(float(row['RRR%']), max(dec - 2, 0))}%"
        out["p.value"] = format_p_value(row["p.value"], dec)
        out["stars"] = str(row.get(STARS_COLUMN) or "").strip()
        rows.append(out)
    report: dict[str, Any] = {
        "kind": "mnl",
        **_header(model),
        "tables": [
            {
                "id": "coefficients",
                "title": "Coefficient Estimates",
                "subtitle": f"Response: {model.rvar} (base level: {model.lev})",
                "columns": columns,
                "rows": rows,
                "note": f"Signif. codes:  {SIG_CODES}",
            }
        ],
        "fit": [line for line in model._fit_text(dec).splitlines() if line.strip()],
        "sections": [],
    }
    sections = report["sections"]
    if ci or rrr:
        cis = model.conf_int(conf)
        low, high = cis.columns[3], cis.columns[4]

        def ci_table(id_: str, title: str, keys: list[tuple[str, str]], frame: Any, transform) -> dict[str, Any]:
            cols = [_column("level", "level", "term"), _column("index", "", "term")]
            cols += [_column(k, label) for k, label in keys]
            trows = []
            for r in frame.iter_rows(named=True):
                out = {"level": str(r["level"]), "index": str(r["index"])}
                for k, _ in keys:
                    out[k] = _fixed(transform(k, r[k]), dec)
                trows.append(out)
            return {"id": id_, "title": title, "columns": cols, "rows": trows}

        if ci:
            table = ci_table(
                "ci", "Confidence intervals",
                [("coefficient", "coefficient"), (low, low), (high, high), ("+/-", "+/-")],
                cis, lambda k, v: v,
            )
            sections.append({"id": "ci", "title": "Confidence intervals", "table": table})
        if rrr:
            lab = "std RRR" if getattr(model, "standardize", 0) else "RRR"
            frame = cis.filter(cis["index"] != "Intercept")
            table = ci_table(
                "rrr", "Relative risk ratios",
                [("coefficient", lab), (low, low), (high, high)],
                frame, lambda k, v: float(np.exp(v)),
            )
            sections.append({"id": "rrr", "title": "Relative risk ratios", "table": table})
    step = _stepwise_section(model)
    if step:
        sections.append(step)
    if test:
        section = _text_section("test", "Model comparison (Chi-squared test)", _capture(model.chisq_test, test=test, dec=dec))
        if section:
            sections.append(section)
    if confusion:
        tab = model.confusion()
        levels = list(model.levels)
        cols = [_column("predicted", "predicted", "term")] + [_column(lv, lv) for lv in levels]
        crows = []
        for r in tab.iter_rows(named=True):
            out = {"predicted": str(r["predicted"])}
            for lv in levels:
                v = r[lv]
                out[lv] = f"{v:,.0f}" if float(v).is_integer() else _fixed(v, dec)
            crows.append(out)
        m = tab.select(levels).to_numpy().astype(float)
        err = 1 - np.trace(m) / m.sum()
        sections.append(
            {
                "id": "confusion",
                "title": "Confusion matrix",
                "table": {
                    "id": "confusion",
                    "title": "Confusion matrix",
                    "subtitle": "Rows: predicted level (most likely); columns: actual level",
                    "columns": cols,
                    "rows": crows,
                    "note": f"Misclassification error: {100 * err:.{dec}f}%",
                },
            }
        )
    return _counted(report)
