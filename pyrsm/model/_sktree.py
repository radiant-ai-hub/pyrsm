"""Draw one scikit-learn decision tree with mermaid, as ``crtree.plot("tree")``.

A random forest has many trees and none of them is "the model", but looking at
one shows what the forest is made of: splits on (dummy-coded) variables, each
estimated on a bootstrap sample and choosing from a random subset of the
variables at every split. ``tree_to_mermaid`` turns ``estimator.tree_`` into
the same mermaid source ``crtree.to_mermaid`` writes -- variable names on the
splits, branch labels, green leaves with n and the prediction, and hover text
with the full branch labels -- so the two trees look alike in Radiant and in a
notebook.

Forest trees are deep (``min_samples_leaf=1`` grows them until the leaves are
pure), so ``depth`` limits how many levels are drawn. A split below that depth
is shown as a grey box with its n, its prediction and the number of splits not
drawn.
"""

from __future__ import annotations

import math

import numpy as np

from pyrsm.model.crtree import _format_nr, _r_num2str

_STYLE = (
    "classDef default fill:none, bg:none, stroke-width:0px;\n"
    "      classDef leaf fill:#9ACD32,stroke:#333,stroke-width:1px;\n"
    "      classDef more fill:#E9ECEF,stroke:#333,stroke-width:1px,stroke-dasharray:4,4;"
)


def _short(lab: str) -> str:
    return lab if len(lab) <= 20 else lab[:20] + " ..."


def tree_to_mermaid(
    estimator,
    feature_names: list[str],
    levels: dict[str, list[str]],
    int_vars: set[str] | None = None,
    mod_type: str = "classification",
    depth: int | None = 3,
    orient: str = "LR",
    dec: int = 2,
    classes: list | None = None,
) -> str:
    """Mermaid source for one fitted sklearn decision tree.

    Parameters
    ----------
    estimator : DecisionTreeClassifier | DecisionTreeRegressor
        A fitted tree, e.g. ``rf.fitted.estimators_[0]``.
    feature_names : list[str]
        The columns the tree was fitted on (dummy columns ``var_level``).
    levels : dict[str, list[str]]
        Every level of each categorical explanatory variable, in level order.
        A split on the dummy ``pclass_1st`` is labelled with levels rather than
        a threshold: ``1st`` on one branch, the other levels still possible on
        that path on the other.
    int_vars : set[str], optional
        Integer variables: their cut-offs are rounded up to whole numbers, as
        in ``crtree``.
    mod_type : {"classification", "regression"}
        Classification shows ``p:``, the share of the chosen level; regression
        shows ``b:``, the average response in the node.
    depth : int, optional
        Number of split levels to draw; None draws the whole tree.
    orient : {"LR", "TD", "RL", "BT"}
        Direction of the diagram.
    dec : int
        Decimals for the predictions and cut-offs.
    classes : list, optional
        The forest's ``classes_``; the share shown is that of class 1 (True).
        A forest's trees only know the classes' positions.
    """
    t = estimator.tree_
    int_vars = int_vars or set()
    depth = None if depth is None or depth < 1 else int(depth)
    if t.node_count == 1:
        return "graph LR\n A[Graph unavailable for single node tree]"

    # dummy column -> (variable, level)
    dummy = {}
    for var, levs in levels.items():
        for lev in levs:
            dummy[f"{var}_{lev}"] = (var, lev)

    def n_of(node: int) -> float:
        return float(t.weighted_n_node_samples[node])

    if mod_type == "classification":
        classes = list(classes if classes is not None else getattr(estimator, "classes_", [0, 1]))
        pos = classes.index(1) if 1 in classes else len(classes) - 1
        pre = "<b>p:</b> "

        def pred_of(node: int) -> str:
            v = np.asarray(t.value[node][0], dtype=float)
            share = v[pos] / v.sum() if v.sum() > 0 else float("nan")
            return _format_nr(share, dec, perc=True)
    else:
        pre = "<b>b:</b> "

        def pred_of(node: int) -> str:
            return _r_num2str(round(float(t.value[node][0][0]), dec))

    def fmt_cut(var: str, cut: float) -> str:
        if var in int_vars:
            return _r_num2str(math.ceil(cut))
        return _r_num2str(round(cut, max(dec, 2)))

    def n_splits(node: int) -> int:
        if t.children_left[node] == -1:
            return 0
        return 1 + n_splits(t.children_left[node]) + n_splits(t.children_right[node])

    lines: list[str] = []
    leafs: list[str] = []
    more: list[str] = []
    ttip: list[str] = []
    brn = ("top", "bottom") if orient in ("LR", "RL") else ("left", "right")
    brn = tuple(f"<b>{b}:</b> " for b in brn)

    def nid(node: int) -> str:
        return f"id{node + 1}"

    def split_of(node: int, allowed: dict[str, list[str]]):
        """(variable, left label, right label, allowed-left, allowed-right)."""
        col = feature_names[t.feature[node]]
        if col in dummy:
            var, lev = dummy[col]
            avail = allowed.get(var, levels[var])
            rest = [lv for lv in avail if lv != lev]
            left_allowed = {**allowed, var: rest}
            right_allowed = {**allowed, var: [lev]}
            return var, ",".join(rest), lev, left_allowed, right_allowed
        cut = fmt_cut(col, float(t.threshold[node]))
        return col, f"< {cut}", f">= {cut}", allowed, allowed

    def node_lab(node: int, level: int, var: str | None) -> str:
        head = f"{var}<br>" if var else ""
        tail = ""
        if var is None and t.children_left[node] != -1:
            tail = f"<br>+{n_splits(node)} splits"
        return f"{nid(node)}[{head}<b>n:</b> {_format_nr(n_of(node), 0)}<br>{pre}{pred_of(node)}{tail}]"

    def walk(node: int, level: int, allowed: dict[str, list[str]]) -> None:
        var, lab_l, lab_r, al_l, al_r = split_of(node, allowed)
        ttip_parts = []
        for child, lab, al, b in (
            (t.children_left[node], lab_l, al_l, brn[0]),
            (t.children_right[node], lab_r, al_r, brn[1]),
        ):
            is_leaf = t.children_left[child] == -1
            cut_off = depth is not None and level + 1 >= depth and not is_leaf
            child_var = None
            if not is_leaf and not cut_off:
                child_var = split_of(child, al)[0]
            lines.append(f"{nid(node)}[{var}] --- |{_short(lab)}|{node_lab(child, level + 1, child_var)}")
            if is_leaf:
                leafs.append(nid(child))
            elif cut_off:
                more.append(nid(child))
            else:
                walk(child, level + 1, al)
            ttip_parts.append("" if _short(lab) == lab else "<br>" + b + lab.replace(",", ", "))
        ttip.append(
            f'click {nid(node)} callback "<b>n:</b> {_format_nr(n_of(node), 0)}<br>'
            f'{pre}{pred_of(node)}{"".join(ttip_parts)}"'
        )

    walk(0, 0, {})
    style = _STYLE
    if leafs:
        style += "\n      class " + ",".join(leafs) + " leaf;"
    if more:
        style += "\n      class " + ",".join(more) + " more;"
    return "\n".join([f"graph {orient}", "\n".join(lines), style, "\n".join(ttip)])
