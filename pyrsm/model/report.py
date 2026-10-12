"""Radiant's Summary reports for the Model tools without coefficients.

The same builders Radiant's tool workspaces use, so ``summary(plain=False)``
in a notebook or a Quarto report shows the tables a student sees in the app
(:mod:`pyrsm.report` renders them):

- :func:`fit_report` -- random forest, XGBoost, neural network: pyrsm's header,
  then a "Model fit" table (:func:`fit_metrics`);
- :func:`crtree_report` -- the classification / regression tree, node by node;
- :func:`evaluate_report` -- Model > Evaluate: pyrsm's header, then its tables.
"""

from __future__ import annotations

from typing import Any, Sequence

from pyrsm.report import build_report, column, frame_table, table, text_header

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
