"""Classification and regression trees -- Python port of ``radiant.model::crtree``.

Radiant fits trees with ``rpart``; ``pyrsm.model._rpart`` ports that algorithm
so the same data and settings give the same tree, complexity table,
cross-validated error, variable importance and predictions as Radiant. The tree
is drawn with mermaid, like Radiant's "tree" plot.

Examples
--------
>>> import polars as pl
>>> import pyrsm as rsm
>>> titanic = pl.read_parquet("examples/data/model/titanic.parquet")
>>> ct = rsm.model.crtree({"titanic": titanic}, rvar="survived",
...                       evar=["pclass", "sex"], lev="Yes")
>>> ct.summary()
>>> ct.plot()              # mermaid tree
>>> ct.plot("prune")       # cross-validated pruning plot
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from pyrsm.data_scope import apply_scope, print_scope
from pyrsm.model._mermaid import MermaidDiagram
from pyrsm.model._rpart import (
    LEFT,
    RRandom,
    r_sample_int_replace,
    r_sample_xgroups,
    rpart_fit,
)
from pyrsm.model.model import convert_to_list, nobs_dropped, sim_prediction
from pyrsm.utils import cat_levels, check_dataframe

__all__ = ["crtree", "cv_crtree"]


# ---------------------------------------------------------------------------
# R-style number formatting (so the printed tree matches Radiant line by line)
# ---------------------------------------------------------------------------


def _r_sci(x: float, digits: int) -> tuple[bool, int, int]:
    """(negative, exponent, significant digits needed) -- R's ``scientific``."""
    if x == 0:
        return False, 0, 1
    mant, ex = f"{abs(x):.{digits - 1}e}".split("e")
    nsig = len(mant.replace(".", "").rstrip("0")) or 1
    return x < 0, int(ex), nsig


def _r_format(values, digits: int = 7, nsmall: int = 0) -> list[str]:
    """Format numbers together, like R's ``format()`` of a numeric vector.

    All values share one width and one number of decimals (or one scientific
    layout), the way R prints columns of numbers.
    """
    vals = [float(v) for v in values]
    fin = [v for v in vals if math.isfinite(v)]
    if not fin:
        return [str(v) for v in vals]
    neg = rgt = 0
    mxsl = mxns = -(10**9)
    mxe, mne = -(10**9), 10**9
    for v in fin:
        n, kp, nsig = _r_sci(v, digits)
        left = kp + 1
        sleft = int(n) + (left if left > 0 else 1)
        rgt = max(rgt, nsig - kp - 1)
        mxsl = max(mxsl, sleft)
        mxns = max(mxns, nsig)
        mxe, mne = max(mxe, kp), min(mne, kp)
        neg = neg or n
    rgt = max(rgt, 0)
    e = 2 if (mxe >= 100 or mne <= -99) else 1
    d = mxns - 1
    w_e = int(neg) + (d > 0) + d + 4 + e
    w_f = mxsl + rgt + (rgt != 0)
    if w_f <= w_e:
        if nsmall > rgt:
            rgt = nsmall
            w_f = mxsl + rgt + (rgt != 0)
        return [f"{v:>{w_f}.{rgt}f}" for v in vals]
    out = []
    for v in vals:
        s = f"{v:.{d}e}"
        mant, ex = s.split("e")
        sign, num = ex[0], ex[1:].lstrip("0") or "0"
        out.append(f"{mant}e{sign}{num.zfill(e + 1)}".rjust(w_e))
    return out


def _r_num2str(x: float) -> str:
    """``as.character(x)`` for a double (15 significant digits)."""
    return _r_format([x], digits=15)[0].strip()


def _signif(x: float, digits: int = 7) -> float:
    return float(f"{x:.{digits}g}") if math.isfinite(x) else x


def _formatg(x: float, digits: int) -> str:
    """rpart's ``formatg``: ``sprintf("%.<digits>g", x)``."""
    return f"{x:.{digits}g}"


def _format_nr(x: float, dec: int = 2, perc: bool = False) -> str:
    """radiant.data's ``format_nr`` (thousands separator, fixed decimals)."""
    if perc:
        return f"{100 * x:,.{dec}f}%"
    return f"{x:,.{dec}f}"


def _fmt_cut(x: float) -> str:
    """A split cut-off: whole numbers without a decimal point."""
    return str(int(x)) if x == int(x) else _r_num2str(x)


def _r_print_matrix(rownames, colnames, columns, digits: int = 7) -> str:
    """Print a numeric matrix the way R does (each column formatted on its own)."""
    cols = [_r_format(c, digits) for c in columns]
    widths = [max(len(cn), max((len(v) for v in c), default=0)) for cn, c in zip(colnames, cols)]
    rw = max((len(r) for r in rownames), default=0)
    lines = [" " * rw + " " + " ".join(cn.rjust(w) for cn, w in zip(colnames, widths))]
    for i, rn in enumerate(rownames):
        cells = " ".join(c[i].rjust(w) for c, w in zip(cols, widths))
        lines.append(rn.ljust(rw) + " " + cells)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------


def _is_categorical(s: pl.Series) -> bool:
    return isinstance(s.dtype, (pl.Enum, pl.Categorical)) or s.dtype in (
        pl.Utf8,
        pl.String,
        pl.Boolean,
    )


def _levels(s: pl.Series) -> list[str]:
    """Levels in Enum order (text: order of appearance), without levels that
    do not occur -- R drops unused levels before estimation."""
    if s.dtype == pl.Boolean:
        present = set(s.cast(pl.Utf8).drop_nulls().unique().to_list())
        return [c for c in ("false", "true") if c in present]
    return cat_levels(s, drop_unused=True)


def _numeric(s: pl.Series) -> np.ndarray:
    """Numeric values as R's model matrix sees them (dates as numbers)."""
    if s.dtype == pl.Date:
        return s.to_physical().cast(pl.Float64).to_numpy()
    if isinstance(s.dtype, pl.Datetime):
        div = {"ns": 1e9, "us": 1e6, "ms": 1e3}[s.dtype.time_unit or "us"]
        return s.to_physical().cast(pl.Float64).to_numpy() / div
    return s.cast(pl.Float64).to_numpy()


def _scale_factor(check) -> float:
    """radiant.model's ``scale_factor``: 'standardize-1sd' -> 1, else 2."""
    import re

    for c in convert_to_list(check):
        m = re.match(r"^standardize-([0-9.]+)sd$", str(c))
        if m:
            try:
                sf = float(m.group(1))
                return sf if sf > 0 else 2.0
            except ValueError:
                return 2.0
    return 2.0


class _TreePredictor:
    """Minimal sklearn-style view of a crtree so pyrsm's prediction and PDP plots
    can use it. Takes the original (not dummy-coded) explanatory variables."""

    def __init__(self, model: "crtree"):
        self._model = model
        self.feature_names_in_ = np.array(model.evar, dtype=object)
        if model.mod_type == "classification":
            self.classes_ = np.array([0, 1])

    def _pred(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = pl.from_pandas(X)
        return self._model._predict_values(X.select(self._model.evar))

    def predict_proba(self, X) -> np.ndarray:
        p = self._pred(X)
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return self._pred(X)


# ---------------------------------------------------------------------------
# crtree
# ---------------------------------------------------------------------------


class crtree:
    """
    Classification and regression trees (``rpart``), as in Radiant's crtree.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame]
        Data to estimate the tree on. Pass ``{"name": df}`` to show the name in
        the summary.
    rvar : str
        Response variable. A categorical response gives a classification tree,
        a numeric response a regression tree (unless ``mod_type`` says otherwise).
    evar : list[str]
        Explanatory variables.
    lev : str, optional
        Level of the response to treat as "success" in a classification tree.
        Defaults to the first level.
    mod_type : {"classification", "regression"}, optional
        Model type. Inferred from ``rvar`` when not given. Use "classification"
        to treat a numeric response (e.g. 0/1) as categorical.
    wts : str, optional
        Name of a column with (frequency) weights.
    minsplit : int, default 2
        Minimum number of observations in a node for a split to be attempted.
    minbucket : int, optional
        Minimum number of observations in a terminal node. Defaults to
        ``round(minsplit / 3)``.
    cp : float, default 0.001
        Complexity parameter: a split must decrease the overall lack of fit by
        at least this fraction of the root node's deviance.
    pcp : float, optional
        Complexity parameter used to prune the tree after estimation.
    nodes : int, optional
        Maximum number of terminal nodes; the tree is pruned to the largest
        tree in the cp table with at most this many nodes.
    K : int, default 10
        Number of folds used to cross-validate the cp table.
    seed : int, default 1234
        Random seed for the cross-validation folds (matches R's ``set.seed``).
    split : {"gini", "information"}, default "gini"
        Splitting criterion for classification trees.
    prior : float, optional
        Prior probability for ``lev`` (e.g. 0.5 for an unbalanced sample).
    adjprob : bool, default True
        With a prior, rescale predicted probabilities back to the original
        base rate.
    cost : float, optional
        Cost of a treatment (e.g. a mailing). Use with ``margin``.
    margin : float, optional
        Margin on a successful treatment (e.g. a purchase). Use with ``cost``.
    check : str | list[str], optional
        "standardize" (or "standardize-1sd" / "standardize-2sd") to
        standardize numeric explanatory variables by 2 (or 1) standard deviations.
    data_filter : str, default ""
        Expression used to filter the data before the analysis, e.g.,
        "price > 5000 and cut in ['Ideal', 'Premium']". A bare name is a column
        and a call is an operation on one (``is_null(price)``). See
        ``pyrsm.data_scope`` for the full list.
    sort : str, default ""
        Column(s) to sort on, "-" in front for descending, e.g.,
        "clarity -price". Applied after the filter.
    slice : str, default ""
        Rows to keep, counting from 1, e.g., "1:100". Applied after the filter
        and the sort.
    """

    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        rvar: str | None = None,
        evar: list[str] | str | None = None,
        lev: str | None = None,
        mod_type: Literal["classification", "regression"] | None = None,
        wts: str | None = None,
        minsplit: int = 2,
        minbucket: int | None = None,
        cp: float = 0.001,
        pcp: float | None = None,
        nodes: int | None = None,
        K: int = 10,
        seed: int | None = 1234,
        split: Literal["gini", "information"] = "gini",
        prior: float | None = None,
        adjprob: bool = True,
        cost: float | None = None,
        margin: float | None = None,
        check: str | list[str] | None = None,
        data_filter: str = "",
        sort: str = "",
        slice: str = "",
    ) -> None:
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            data = data[self.name]
        else:
            self.name = "Not provided"
        data = check_dataframe(data)

        # The data scope, applied before anything is measured: filter, then
        # sort, then slice. Kept on the object so ``summary`` can say which
        # rows the numbers describe.
        self.data_filter = data_filter
        self.sort = sort
        self.slice = slice
        data = apply_scope(data, data_filter, sort, slice)

        self.rvar = rvar
        self.evar = convert_to_list(evar)
        self.wts = wts
        self.minsplit = 2 if minsplit is None else minsplit
        self.minbucket = (
            int(round(self.minsplit / 3)) if minbucket is None else int(minbucket)
        )
        self.cp = cp
        self.pcp = pcp
        self.nodes = nodes
        self.K = K
        self.seed = seed
        self.split = split
        self.prior = prior
        self.adjprob = adjprob
        self.cost = cost
        self.margin = margin
        self.check = convert_to_list(check)

        if rvar in self.evar:
            raise ValueError(
                "Response variable contained in the set of explanatory variables. "
                "Please update model specification."
            )
        if cp is None:
            raise ValueError("Please provide a complexity parameter to split the data.")
        if nodes is not None and nodes < 2:
            raise ValueError(
                "The (maximum) number of nodes in the tree should be larger than "
                "or equal to 2."
            )

        cols = [rvar] + self.evar + ([wts] if wts else [])
        self.nobs_all = data.height
        data = data.select(cols).drop_nulls()
        self.nobs = data.height
        self.nobs_dropped = self.nobs_all - self.nobs

        not_vary = []
        for c in [rvar] + self.evar:
            s = data[c]
            if s.len() <= 1:
                not_vary.append(c)
            elif _is_categorical(s):
                if s.n_unique() <= 1:
                    not_vary.append(c)
            else:
                v = _numeric(s)
                if abs(np.max(v) - np.min(v)) <= np.finfo(float).eps ** 0.5:
                    not_vary.append(c)
        if not_vary:
            raise ValueError(
                "The following variable(s) show no variation. Please select other "
                f"variables: {', '.join(not_vary)}"
            )

        # response: a factor gives a classification tree
        rv = data[rvar]
        if mod_type == "classification" and not _is_categorical(rv):
            # R's factor() orders numeric levels by value, not as text
            levels = [_r_num2str(v) for v in sorted(np.unique(_numeric(rv)))]
            rv = pl.Series(rvar, [_r_num2str(v) for v in _numeric(rv)]).cast(pl.Enum(levels))
            data = data.with_columns(rv)
        if _is_categorical(rv):
            if mod_type == "regression":
                raise ValueError(
                    "Cannot estimate a regression when the response variable is "
                    "of type factor."
                )
            levels = _levels(rv)
            if lev is None or lev == "":
                lev = levels[0]
            elif lev not in levels:
                raise ValueError(f"Specified level is not a level in {rvar}")
            else:
                levels = [lev] + [lv for lv in levels if lv != lev]
            self.mod_type = "classification"
            self.ylevels = levels
            ystr = rv.cast(pl.Utf8).to_list()
            code = {lv: i + 1 for i, lv in enumerate(levels)}
            y = np.array([code[v] for v in ystr], dtype=float)
        else:
            self.mod_type = "regression"
            self.ylevels = None
            lev = None
            y = _numeric(rv).astype(float)
        self.lev = lev
        self.ml_model = {"model": "crtree", "mod_type": self.mod_type}

        if wts:
            self.wt = _numeric(data[wts]).astype(float)
            self.wts_integer = data[wts].dtype.is_integer()
        else:
            self.wt = np.ones(data.height)
            self.wts_integer = False

        # explanatory variables: factors become 1-based codes, as in R
        self.data = data.select([rvar] + self.evar)
        self.xlevels: dict[str, list[str]] = {}
        self.int_vars: list[str] = []
        self.means: dict[str, float] = {}
        self.sds: dict[str, float] = {}
        self.sf = _scale_factor(self.check)
        standardize = any(str(c).startswith("standardize") for c in self.check)
        for v in self.evar:
            s = data[v]
            if _is_categorical(s):
                self.xlevels[v] = _levels(s)
            else:
                if s.dtype.is_integer():
                    self.int_vars.append(v)
                if standardize:
                    x = _numeric(s)
                    if wts:
                        w = self.wt / self.wt.sum()
                        m = float(np.sum(w * x))
                        sd = float(np.sqrt(np.sum(w * (x - m) ** 2)))
                    else:
                        m = float(np.mean(x))
                        sd = float(np.std(x, ddof=1))
                    self.means[v], self.sds[v] = m, sd
        if standardize:
            # standardized integers are no longer integers (labels are not rounded)
            self.int_vars = []
        X = self._encode(data.select(self.evar))
        ncat = [len(self.xlevels[v]) if v in self.xlevels else 0 for v in self.evar]

        # priors, or a loss matrix from cost and margin
        loss = None
        prior_vec = None
        if self.mod_type == "classification":
            has_cm = cost is not None and margin is not None
            if prior is not None and has_cm:
                raise ValueError(
                    "Choose either a prior or cost and margin values but not both. "
                    "Please adjust your settings and try again"
                )
            if (prior is not None or has_cm) and len(self.ylevels) != 2:
                raise ValueError(
                    "A prior or cost and margin can only be used with a response "
                    "that has two levels"
                )
            if has_cm:
                loss2 = float(cost)
                loss1 = float(margin) - loss2
                if loss1 <= 0:
                    raise ValueError(
                        "Cost must be smaller than the specified margin. Please "
                        "adjust the settings and try again"
                    )
                if loss2 <= 0:
                    raise ValueError(
                        "Cost must be larger than zero. Please adjust the settings "
                        "and try again"
                    )
                loss = np.array([[0.0, loss1], [loss2, 0.0]])
            elif prior is not None:
                if not isinstance(prior, (int, float)):
                    raise ValueError("Prior did not resolve to a numeric factor")
                if prior > 1 or prior < 0:
                    raise ValueError("Prior is not a valid probability")
                prior_vec = np.array([prior, 1 - prior])

        if split not in ("gini", "information"):
            raise ValueError("split should be 'gini' or 'information'")

        K = int(K) if K else 0
        xgroups = None
        if K > 1:
            s = seed if seed is not None else int(np.random.randint(0, 2**31 - 1))
            xgroups = r_sample_xgroups(data.height, K, s)

        fit = rpart_fit(
            X,
            ncat,
            y,
            self.wt,
            "class" if self.mod_type == "classification" else "anova",
            self.evar,
            self.xlevels,
            ylevels=self.ylevels,
            prior=prior_vec,
            loss=loss,
            split=1 if split == "gini" else 2,
            minsplit=self.minsplit,
            minbucket=self.minbucket,
            cp=cp,
            xval=K,
            xgroups=xgroups,
        )

        self.unpruned = None
        if nodes is not None:
            self.unpruned = fit
            if len(fit.frame) > 1:
                cpt = fit.cptable
                ind = int(np.max(np.nonzero(cpt[:, 1] + 1 <= nodes)[0]))
                fit = fit.prune(cpt[ind, 0])
        elif pcp is not None:
            self.unpruned = fit
            if len(fit.frame) > 1:
                fit = fit.prune(pcp)

        # rpart does not return residuals by default (Pearson, as in Radiant)
        if self.mod_type == "classification":
            yprob = np.array([fit.frame[k]["yval2"][1 + len(self.ylevels) :][:-1] for k in fit.where])
            yhat = yprob[np.arange(y.size), y.astype(int) - 1]
            self.residuals = (1 - yhat) / yhat
        else:
            self.residuals = y - np.array([fit.frame[k]["yval"] for k in fit.where])

        if cost is None and margin is None and prior is not None and adjprob:
            # Elkan (2001), theorem 2: undo the prior so probabilities are on
            # the original scale
            bp = float(np.mean(y == 1))
            fit.yprob_adjust = (float(prior), bp)
            fit.build_frame()

        self.fit = fit
        self.y = y
        self.fitted = _TreePredictor(self)

    # -- encoding ----------------------------------------------------------
    def _encode(self, df: pl.DataFrame) -> np.ndarray:
        """Matrix rpart sees: level codes for factors, (standardized) numbers."""
        cols = []
        for v in self.evar:
            s = df[v]
            if v in self.xlevels:
                code = {lv: float(i + 1) for i, lv in enumerate(self.xlevels[v])}
                cols.append(
                    np.array(
                        [code.get(x, np.nan) if x is not None else np.nan
                         for x in s.cast(pl.Utf8).to_list()],
                        dtype=float,
                    )
                )
            else:
                x = _numeric(s).astype(float)
                if v in self.means:
                    x = (x - self.means[v]) / (self.sf * self.sds[v])
                cols.append(x)
        return np.column_stack(cols) if cols else np.empty((df.height, 0))

    # -- printing ----------------------------------------------------------
    def _split_labels(self, digits: int) -> dict[int, tuple[str, str]]:
        """Left and right labels of each internal node's primary split
        (``labels.rpart(minlength = 0, collapse = FALSE)``)."""
        out = {}
        for row in self.fit.frame:
            if row["leaf"]:
                continue
            s = row["node"].primary[0]
            if s.ncat == 0:
                cut = _formatg(s.spoint, digits)
                if s.direction < 0:
                    out[row["id"]] = ("< " + cut, ">=" + cut)
                else:
                    out[row["id"]] = (">=" + cut, "< " + cut)
            else:
                levels = self.xlevels[self.evar[s.var]]
                left = ",".join(lv for lv, d in zip(levels, s.csplit) if d == LEFT)
                right = ",".join(lv for lv, d in zip(levels, s.csplit) if d == -LEFT)
                out[row["id"]] = (left, right)
        return out

    def tree_text(self) -> str:
        """The tree as printed by ``print.rpart`` (without the "n=" header)."""
        frame = self.fit.frame
        digits = 7
        labels = self._split_labels(digits)
        ids = [r["id"] for r in frame]
        depth = [int(math.floor(math.log2(i))) for i in ids]
        node_str = _r_format(ids, digits)
        z = []
        byid = {r["id"]: r for r in frame}
        for r in frame:
            nid = r["id"]
            if nid == 1:
                z.append("root")
                continue
            parent = byid[nid // 2]
            pvar = parent["var"]
            lab = labels[nid // 2][nid % 2]
            if parent["node"].primary[0].ncat > 0:
                lab = "=" + lab
            z.append(pvar + lab)
        dev = _r_format([_signif(r["dev"]) for r in frame], digits)
        if self.mod_type == "classification":
            nclass = len(self.ylevels)
            probs = np.array([r["yval2"][1 + nclass : 1 + 2 * nclass] for r in frame])
            if nclass < 5:
                flat = _r_format(probs.ravel(), digits, nsmall=min(20, digits))
            else:
                flat = [_formatg(p, 2) for p in probs.ravel()]
            pm = np.array(flat, dtype=object).reshape(probs.shape)
            yval = []
            for k, r in enumerate(frame):
                cls = self.ylevels[int(r["yval2"][0]) - 1]
                yval.append(f"{cls} (" + " ".join(pm[k]) + ")")
            header = "node), split, n, loss, yval, (yprob)"
        else:
            yval = _r_format([_signif(r["yval"]) for r in frame], digits)
            header = "node), split, n, deviance, yval"
        lines = [header, "      * denotes terminal node", ""]
        for k, r in enumerate(frame):
            indent = "" if depth[k] == 0 else " " * (2 * depth[k])
            term = "*" if r["leaf"] else " "
            lines.append(
                f"{indent}{node_str[k]}) {z[k]} {r['n']} {dev[k]} {yval[k]} {term}"
            )
        return "\n".join(lines)

    def summary(self, prn: bool = True, splits: bool = False, cptab: bool = False) -> None:
        """
        Summarize a classification or regression tree.

        Parameters
        ----------
        prn : bool, default True
            Print the tree in text form.
        splits : bool, default False
            Print the splits table (primary, competing and surrogate splits).
        cptab : bool, default False
            Print the complexity parameter table.
        """
        print("Classification tree" if self.mod_type == "classification" else "Regression tree")
        print(f"Data                 : {self.name}")
        print_scope(self, 21)
        print(f"Response variable    : {self.rvar}")
        if self.mod_type == "classification":
            print(f"Level                : {self.lev} in {self.rvar}")
        print(f"Explanatory variables: {', '.join(self.evar)}")
        if self.wts:
            print(f"Weights used         : {self.wts}")
        print(f"Complexity parameter : {self.cp}")
        print(f"Minimum observations : {self.minsplit}")
        if self.nodes is not None:
            max_nodes = sum(r["leaf"] for r in self.unpruned.frame)
            print(f"Maximum nr. nodes    : {self.nodes} out of {max_nodes}")
        prior = self.prior
        if self.cost is not None and self.margin is not None and self.mod_type == "classification":
            print(f"Cost:Margin          : {self.cost} : {self.margin}")
        if prior is not None and self.mod_type == "classification":
            print(f"Priors               : {prior}")
            print(f"Adjusted prob.       : {self.adjprob}")
        if self.wts and self.wts_integer:
            print(f"Nr obs               : {self.wt.sum():,.0f}{nobs_dropped(self)}")
        else:
            print(f"Nr obs               : {self.nobs:,.0f}{nobs_dropped(self)}")
        print()

        if splits:
            tab = self.fit.splits_table()
            if tab:
                index = []
                crow = 0
                for t in tab:
                    if t["index"] is None:
                        crow += 1
                        index.append(crow)
                    else:
                        index.append(t["index"])
                print(
                    _r_print_matrix(
                        [t["var"] for t in tab],
                        ["count", "ncat", "improve", "index", "adj"],
                        [
                            [t["count"] for t in tab],
                            [t["ncat"] for t in tab],
                            [t["improve"] for t in tab],
                            index,
                            [t["adj"] for t in tab],
                        ],
                    )
                )
        if cptab:
            cpt = self.fit.cptable
            names = ["CP", "nsplit", "rel error", "xerror", "xstd"][: cpt.shape[1]]
            print(
                _r_print_matrix(
                    [str(i + 1) for i in range(cpt.shape[0])],
                    names,
                    [cpt[:, j] for j in range(cpt.shape[1])],
                )
            )
        if prn:
            print(self.tree_text())

    # -- results -----------------------------------------------------------
    @property
    def variable_importance(self) -> pl.DataFrame:
        """rpart's variable importance (primary and surrogate splits)."""
        imp = self.fit.variable_importance()
        return pl.DataFrame(
            {"variable": list(imp.keys()), "importance": [float(v) for v in imp.values()]}
        )

    @property
    def cptable(self) -> pl.DataFrame:
        """The complexity parameter table (cross-validated when K > 1)."""
        cpt = self.fit.cptable
        names = ["CP", "nsplit", "rel error", "xerror", "xstd"][: cpt.shape[1]]
        return pl.DataFrame({n: cpt[:, j] for j, n in enumerate(names)}).with_columns(
            pl.col("nsplit").cast(pl.Int64)
        )

    def _predict_values(self, df: pl.DataFrame) -> np.ndarray:
        X = self._encode(df)
        where = self.fit.predict_where(X)
        if self.mod_type == "classification":
            nclass = len(self.ylevels)
            return np.array([self.fit.frame[k]["yval2"][1 + nclass] for k in where])
        return np.array([self.fit.frame[k]["yval"] for k in where])

    def predict(self, data=None, cmd=None, data_cmd=None, dec=None) -> pl.DataFrame:
        """
        Predict probabilities (classification, for ``lev``) or values (regression).

        Parameters
        ----------
        data : pd.DataFrame | pl.DataFrame, optional
            Data to predict for. Defaults to the estimation data.
        cmd : dict, optional
            Values to vary, other variables held at their mean or most common
            value, e.g. ``{"pclass": ["1st", "2nd", "3rd"]}``.
        data_cmd : dict, optional
            Values to set for all rows of ``data``, e.g. ``{"sex": "male"}``.
        dec : int, optional
            Round float columns to this many decimals.
        """
        if data is None:
            pred_data = self.data.select(self.evar)
        else:
            pred_data = check_dataframe(data).select(self.evar)
        if data_cmd is not None and data_cmd != "":
            pred_data = pred_data.with_columns(
                [pl.lit(v).cast(pred_data[k].dtype).alias(k) for k, v in data_cmd.items()]
            )
        elif cmd is not None and cmd != "":
            cmd = {k: [v] if isinstance(v, str) else v for k, v in cmd.items()}
            pred_data = sim_prediction(data=pred_data, vary=cmd)
        pred = pred_data.with_columns(
            pl.Series("prediction", self._predict_values(pred_data))
        )
        if dec is not None:
            pred = pred.with_columns(
                [pl.col(c).round(dec) for c in pred.columns
                 if pred[c].dtype in (pl.Float64, pl.Float32)]
            )
        return pred

    def rules(self, dec: int = 4) -> pl.DataFrame:
        """
        One row per terminal node: the conditions that lead to it, its size
        and its prediction, highest prediction first.

        Conditions on the same variable are merged along the path, so a node
        reached by ``age >= 9.5`` and then ``age < 56.5`` reads
        ``9.5 <= age < 56.5``. Integer variables use whole-number cut-offs, as
        in the tree plot. For a classification tree the prediction is the
        probability of ``lev`` (adjusted for a prior, as in the plot); for a
        regression tree it is the mean response in the node.

        Parameters
        ----------
        dec : int, default 4
            Significant digits of the numeric cut-offs.

        Returns
        -------
        pl.DataFrame
            Columns ``node`` (rpart's node number), ``rule``, ``n``, ``share``
            (fraction of the estimation sample) and ``prediction``.
        """
        frame = self.fit.frame
        byid = {r["id"]: r for r in frame}
        labels = self._split_labels(dec)
        root_n = byid[1]["n"]
        nclass = len(self.ylevels) if self.mod_type == "classification" else 0


        rows = []
        for r in frame:
            if not r["leaf"]:
                continue
            path = []
            nid = r["id"]
            while nid > 1:
                parent = nid // 2
                path.append((byid[parent]["var"], labels[parent][nid % 2]))
                nid = parent
            path.reverse()
            bounds: dict[str, dict[str, float | str]] = {}
            for var, lab in path:
                kind, op, value = self._condition(var, lab)
                b = bounds.setdefault(var, {})
                if kind == "cat":
                    b["in"] = value  # deeper splits are subsets: keep the last
                elif op == ">=":
                    b[">="] = max(b.get(">=", -math.inf), value)
                else:
                    b["<"] = min(b.get("<", math.inf), value)
            parts = []
            for var, b in bounds.items():
                if "in" in b:
                    levels = str(b["in"]).split(",")
                    sep = " = " if len(levels) == 1 else " in "
                    parts.append(var + sep + ", ".join(levels))
                elif ">=" in b and "<" in b:
                    parts.append(f"{_fmt_cut(b['>='])} <= {var} < {_fmt_cut(b['<'])}")
                elif ">=" in b:
                    parts.append(f"{var} >= {_fmt_cut(b['>='])}")
                else:
                    parts.append(f"{var} < {_fmt_cut(b['<'])}")
            if nclass:
                pred = float(r["yval2"][1 + nclass])
            else:
                pred = float(r["yval"])
            rows.append(
                {
                    "node": int(r["id"]),
                    "rule": " and ".join(parts),
                    "n": int(r["n"]),
                    "share": r["n"] / root_n,
                    "prediction": pred,
                }
            )
        rows.sort(key=lambda row: -row["prediction"])
        return pl.DataFrame(rows)

    def node_table(self, dec: int = 4) -> pl.DataFrame:
        """
        Every node of the tree in rpart's print order (parent before its
        children, left before right), for an indented view of the hierarchy.

        Parameters
        ----------
        dec : int, default 4
            Significant digits of the numeric cut-offs.

        Returns
        -------
        pl.DataFrame
            Columns ``node`` (rpart's node number), ``depth`` (0 for the root),
            ``split`` (the condition that leads into the node from its parent,
            e.g. ``age >= 10``; empty for the root), ``last`` (True for the
            second child of a parent), ``leaf``, ``n``, ``share`` and
            ``prediction`` (probability of ``lev``, prior-adjusted as in the
            plot, or the node mean).
        """
        frame = self.fit.frame
        byid = {r["id"]: r for r in frame}
        labels = self._split_labels(dec)
        root_n = byid[1]["n"]
        nclass = len(self.ylevels) if self.mod_type == "classification" else 0

        def split_text(nid: int) -> str:
            if nid == 1:
                return ""
            parent = nid // 2
            var = byid[parent]["var"]
            kind, op, value = self._condition(var, labels[parent][nid % 2])
            if kind == "cat":
                levels = str(value).split(",")
                return var + (" = " if len(levels) == 1 else " in ") + ", ".join(levels)
            return f"{var} {op} {_fmt_cut(float(value))}"

        rows = []
        for r in frame:
            nid = r["id"]
            rows.append(
                {
                    "node": int(nid),
                    "depth": int(nid).bit_length() - 1,
                    "split": split_text(nid),
                    "last": nid > 1 and nid % 2 == 1,
                    "leaf": bool(r["leaf"]),
                    "n": int(r["n"]),
                    "share": r["n"] / root_n,
                    "prediction": float(r["yval2"][1 + nclass]) if nclass else float(r["yval"]),
                }
            )
        return pl.DataFrame(rows)

    def _condition(self, var: str, lab: str) -> tuple[str, str, float | str]:
        """A split label as (kind, operator, value); integer cut-offs rounded up
        to whole numbers, as in the tree plot."""
        for op in (">=", "<"):
            if lab.startswith(op):
                cut = float(lab[len(op):].strip())
                if var in self.int_vars:
                    cut = float(math.ceil(cut))
                return ("num", op, cut)
        return ("cat", "in", lab)

    # -- mermaid tree --------------------------------------------------------
    def to_mermaid(self, orient: str = "LR", dec: int = 2) -> str:
        """Mermaid source for the tree, as drawn by Radiant's ``plot.crtree``."""
        frame = self.fit.frame
        if len(frame) == 1:
            return "graph LR\n A[Graph unavailable for single node tree]"
        if dec is None:
            dec = 2
        if self.mod_type == "classification":
            nclass = len(self.ylevels)
            yv = {r["id"]: _format_nr(r["yval2"][1 + nclass], dec, perc=True) for r in frame}
            pre = "<b>p:</b> "
        else:
            yv = {r["id"]: _r_num2str(round(r["yval"], dec)) for r in frame}
            pre = "<b>b:</b> "
        byid = {r["id"]: r for r in frame}
        labels = self._split_labels(4)

        def int_labs(lab: str) -> str:
            import re

            m = re.match(r"^(>=|<)\s*(-?[0-9]+.*)$", lab)
            if not m:
                return lab
            return f"{m.group(1)} {_r_num2str(math.ceil(float(m.group(2))))}"

        rows = []  # (level, from-row, label, full label pair)
        internal = [r for r in frame if not r["leaf"]]
        full = {}
        short = {}
        for r in internal:
            s1, s2 = labels[r["id"]]
            if r["var"] in self.int_vars:
                s1, s2 = int_labs(s1), int_labs(s2)
            full[r["id"]] = (s1, s2)
            short[r["id"]] = tuple(s if len(s) <= 20 else s[:20] + " ..." for s in (s1, s2))
        for level in (0, 1):
            for r in internal:
                rows.append((level, r))

        def node_lab(nid: int) -> str:
            t = byid[nid]
            head = "" if t["leaf"] else f"{t['var']}<br>"
            return f"id{nid}[{head}<b>n:</b> {_format_nr(t['n'], 0)}<br>{pre}{yv[nid]}]"

        lines = []
        to_ids = []
        for level, r in rows:
            to = 2 * r["id"] + level
            to_ids.append(to)
            lab = short[r["id"]][level]
            lines.append(f"id{r['id']}[{r['var']}] --- |{lab}|{node_lab(to)}")
        internal_ids = {r["id"] for r in internal}
        leafs = []
        for t in to_ids:
            if t not in internal_ids and t not in leafs:
                leafs.append(t)
        style = (
            "classDef default fill:none, bg:none, stroke-width:0px;\n"
            "      classDef leaf fill:#9ACD32,stroke:#333,stroke-width:1px;\n"
            "      classDef chance fill:#FF8C00,stroke:#333,stroke-width:1px;\n"
            "      classDef chance_with_cost fill:#FF8C00,stroke:#333,stroke-width:3px,stroke-dasharray:4,5;\n"
            "      classDef decision fill:#9ACD32,stroke:#333,stroke-width:1px;\n"
            "      classDef decision_with_cost fill:#9ACD32,stroke:#333,stroke-width:3px,stroke-dasharray:4,5;\n"
            "      class " + ",".join(f"id{i}" for i in leafs) + " leaf;"
        )
        brn = ("top", "bottom") if orient in ("LR", "RL") else ("left", "right")
        brn = tuple(f"<b>{b}:</b> " for b in brn)
        ttip = []
        for r in internal:
            f1, f2 = full[r["id"]]
            s1, s2 = short[r["id"]]
            if f1 == s1 and f2 == s2:
                f1 = f2 = ""
            t1 = "" if f1 == "" else "<br>" + brn[0] + f1.replace(",", ", ")
            t2 = "" if f2 == "" else "<br>" + brn[1] + f2.replace(",", ", ")
            ttip.append(
                f'click id{r["id"]} callback "<b>n:</b> {_format_nr(r["n"], 0)}<br>'
                f'{pre}{yv[r["id"]]}{t1}{t2}"'
            )
        return "\n".join([f"graph {orient}", "\n".join(lines), style, "\n".join(ttip)])

    # -- plots ---------------------------------------------------------------
    def _plot_data(self) -> pl.DataFrame:
        """Estimation data with a 0/1 response, as pyrsm's plot helpers expect."""
        if self.mod_type == "classification":
            return self.data.select(self.evar).with_columns(
                pl.Series(self.rvar, (self.y == 1).astype(int))
            )
        return self.data

    def _prune_plot(self):
        from plotnine import (
            aes, geom_hline, geom_line, geom_vline, ggplot, labs, scale_x_continuous,
            theme_bw,
        )

        fit = self.unpruned if self.unpruned is not None else self.fit
        cpt = fit.cptable
        if cpt.shape[0] < 2:
            return "Evaluation of tree pruning not available for single node tree"
        if cpt.shape[1] < 5:
            return "Evaluation of tree pruning requires cross-validation (K > 1)"
        cp = cpt[:, 0]
        cp_mid = np.round(np.sqrt(cp * np.concatenate(([np.inf], cp[:-1]))), 5)
        nodes = (cpt[:, 1] + 1).astype(int)
        xerror, xstd = cpt[:, 3], cpt[:, 4]
        ind1 = int(np.nonzero(xerror == xerror.min())[0][0])
        size1 = (nodes[ind1], cp_mid[ind1])
        ind2 = int(np.nonzero(xerror < xerror[ind1] + xstd[ind1])[0][0])
        size2 = (nodes[ind2], cp_mid[ind2])
        df = pl.DataFrame({"nodes": nodes, "xerror": xerror})
        p = (
            ggplot(df, aes(x="nodes", y="xerror"))
            + geom_line()
            + geom_vline(xintercept=size1[0], linetype="dashed")
            + geom_hline(yintercept=xerror.min(), linetype="dashed")
            + geom_vline(xintercept=size2[0], linetype="dashdot", color="blue")
            + geom_hline(yintercept=xerror[ind1] + xstd[ind1], linetype="dashdot",
                         color="blue")
            + labs(title="Evaluate tree pruning based on cross-validation",
                   x="Number of nodes", y="Relative error")
            + theme_bw()
        )
        if cpt.shape[0] < 10:
            p = p + scale_x_continuous(breaks=list(nodes))
        cp1 = "NA" if math.isinf(size1[1]) else f"{size1[1]:g}"
        footnote = (
            f"Minimum error achieved at prune complexity {cp1} ({size1[0]} nodes)"
        )
        if size2[0] < size1[0]:
            footnote += (
                f".\nError at pruning complexity {size2[1]:g} ({size2[0]} nodes) is "
                "within one std. of minimum"
            )
        return p + labs(caption=footnote)

    def _imp_plot(self):
        from plotnine import aes, coord_flip, geom_bar, ggplot, labs, scale_x_discrete, theme_bw

        imp = self.fit.variable_importance()
        if not imp:
            return "Variable importance information not available for single node tree"
        tot = sum(imp.values())
        df = pl.DataFrame(
            {"vars": list(imp.keys()), "imp": [v / tot for v in imp.values()]}
        ).sort("imp")
        return (
            ggplot(df, aes(x="vars", y="imp"))
            + geom_bar(stat="identity", fill="slateblue")
            + coord_flip()
            + scale_x_discrete(limits=df["vars"].to_list())
            + labs(title="Variable importance", x="", y="Importance")
            + theme_bw()
        )

    def _pip(self, rep: int = 5):
        """Permutation importance (AUC or R-squared decrease)."""
        from plotnine import (
            aes, coord_flip, geom_bar, ggplot, ggtitle, labs, scale_x_discrete, theme,
            theme_bw,
        )

        from pyrsm.model.perf import auc

        data = self.data.select(self.evar)
        y = self.y

        def fit_stat(pred):
            if self.mod_type == "classification":
                return auc((y == 1).astype(int), pred, lev=1)
            return float(np.corrcoef(y, pred)[0, 1] ** 2) if np.std(pred) > 0 else 0.0

        base = fit_stat(self._predict_values(data))
        imp = {v: 0.0 for v in self.evar}
        for i in range(rep):
            for v in self.evar:
                permuted = data.with_columns(pl.col(v).shuffle(seed=i))
                imp[v] += base - fit_stat(self._predict_values(permuted))
        imp = {k: v / rep for k, v in imp.items()}
        df = pl.DataFrame(
            {"variable": list(imp.keys()), "importance": list(imp.values())}
        ).sort("importance")
        xlab = (
            "Importance (AUC decrease)"
            if self.mod_type == "classification"
            else "Importance (R-square decrease)"
        )
        p = (
            ggplot(df, aes(x="variable", y="importance"))
            + geom_bar(stat="identity", fill="slateblue")
            + coord_flip()
            + scale_x_discrete(limits=df["variable"].to_list())
            + labs(x="", y=xlab)
            + ggtitle("Permutation Importance")
            + theme_bw()
            + theme(figure_size=(6, max(5, len(df) * 0.4)))
        )
        return p, df.sort("importance", descending=True)

    def plot(
        self,
        plots: Literal["tree", "prune", "imp", "pip", "pred", "pdp", "dashboard"] = "tree",
        orient: Literal["LR", "TD"] = "LR",
        width: str = "900px",
        dec: int = 2,
        incl=None,
        excl=None,
        incl_int=None,
        nobs: int = 1000,
        fix=True,
        hline=True,
        ice=False,
        ice_nobs=100,
        nnv=20,
        minq=0.025,
        maxq=0.975,
        ret=None,
    ):
        """
        Plots for a classification or regression tree

        Parameters
        ----------
        plots : str, default "tree"
            "tree" draws the tree with mermaid; "prune" plots the cross-validated
            relative error by number of nodes; "imp" plots rpart's variable
            importance; "pip" (or "vip") plots permutation importance; "pred"
            and "pdp" show prediction and partial dependence plots; "dashboard"
            shows residual plots for a regression tree.
        orient : {"LR", "TD"}, default "LR"
            Direction of the tree: left-to-right or top-down.
        width : str, default "900px"
            Maximum width of the tree diagram.
        dec : int, default 2
            Decimals for the values shown in the tree.
        incl, excl, incl_int : list[str], optional
            Variables (and ":"-separated interactions) to include or exclude in
            "pred" and "pdp" plots.
        nobs : int, default 1000
            Number of points to show in the regression dashboard.
        fix, hline, ice, ice_nobs, nnv, minq, maxq
            Passed to the prediction and partial dependence plots.
        ret : bool, optional
            Also return the permutation importance table for "pip".
        """
        plots = convert_to_list(plots)
        if "tree" in plots:
            return MermaidDiagram(self.to_mermaid(orient=orient, dec=dec), width=width)
        if "prune" in plots:
            return self._prune_plot()
        if "imp" in plots:
            return self._imp_plot()
        if "pip" in plots or "vip" in plots:
            p, tab = self._pip()
            return (p, tab) if ret else p
        excl = convert_to_list(excl)
        incl = None if incl is None else convert_to_list(incl)
        incl_int = convert_to_list(incl_int)
        if "pred" in plots:
            from pyrsm.model.visualize import pred_plot_sk

            return pred_plot_sk(
                self.fitted, data=self._plot_data(), rvar=self.rvar, incl=incl,
                excl=excl, incl_int=incl_int, fix=fix, hline=hline, nnv=nnv,
                minq=minq, maxq=maxq,
            )
        if "pdp" in plots:
            from pyrsm.model.visualize import pdp_sk

            return pdp_sk(
                self.fitted, data=self._plot_data(), rvar=self.rvar, incl=incl,
                excl=excl, incl_int=incl_int, ice=ice, ice_nobs=ice_nobs,
                grid_resolution=nnv, minq=minq, maxq=maxq, fix=fix, hline=hline,
            )
        if "dashboard" in plots:
            if self.mod_type != "regression":
                raise ValueError("The dashboard plot is only available for regression trees")
            from types import SimpleNamespace

            from pyrsm.model.model import reg_dashboard

            fitted = self._predict_values(self.data.select(self.evar))
            obj = SimpleNamespace(
                fittedvalues=pl.Series(fitted),
                resid=pl.Series(self.y - fitted),
                model={"endog": self.y},
            )
            return reg_dashboard(obj, nobs=nobs)
        raise ValueError(f"Unknown plot type(s): {', '.join(map(str, plots))}")


# ---------------------------------------------------------------------------
# cross-validation (cv.crtree)
# ---------------------------------------------------------------------------


def cv_crtree(
    model: crtree,
    K: int = 5,
    repeats: int = 1,
    pcp=None,
    seed: int = 1234,
    trace: bool = True,
    fun=None,
    **kwargs,
) -> pl.DataFrame:
    """
    Cross-validate the pruning complexity of a tree, like Radiant's ``cv.crtree``.

    Each value in ``pcp`` is evaluated with ``repeats`` x ``K`` fold
    cross-validation (folds drawn with R's random number generator, so results
    match Radiant for the same seed). The tree is re-estimated with the
    settings of ``model`` on each training fold and pruned at ``pcp``.

    Parameters
    ----------
    model : crtree
        Estimated tree to use as a starting point.
    K : int, default 5
        Number of folds.
    repeats : int, default 1
        Number of times to repeat the K-fold cross-validation.
    pcp : list[float], optional
        Pruning complexity values to evaluate. Default ``0, 0.001, ..., 0.01``.
    seed : int, default 1234
        Random seed.
    trace : bool, default True
        Print progress.
    fun : callable, optional
        ``fun(pred, actual, **kwargs)`` returning a performance measure. Defaults
        to AUC (higher is better) for classification and RMSE (lower is better)
        for regression. For classification ``actual`` is 1 for ``lev``, else 0.

    Returns
    -------
    pl.DataFrame
        One row per ``pcp`` value with the mean, std, min and max of the
        performance measure, best first.
    """
    from pyrsm.model.perf import auc

    if pcp is None:
        pcp = list(np.linspace(0, 0.01, 11))
    pcp = convert_to_list(pcp)
    m = model
    X = m._encode(m.data.select(m.evar))
    ncat = [len(m.xlevels[v]) if v in m.xlevels else 0 for v in m.evar]
    y, wt = m.y, m.wt
    classification = m.mod_type == "classification"
    if fun is None:
        if classification:
            def fun(pred, actual):
                return auc(actual, pred, lev=1)
            cn = "AUC (mean)"
        else:
            def fun(pred, actual):
                return float(np.sqrt(np.mean((pred - actual) ** 2)))
            cn = "RMSE (mean)"
        higher_better = classification
    else:
        cn = f"{getattr(fun, '__name__', 'fun')} (mean)"
        higher_better = classification

    prior_vec = None
    loss = None
    if classification and m.cost is not None and m.margin is not None:
        loss = np.array([[0.0, m.margin - m.cost], [m.cost, 0.0]])
    elif classification and m.prior is not None:
        prior_vec = np.array([m.prior, 1 - m.prior])

    rng = RRandom(seed)
    n = y.size

    def fit_tree(idx):
        # rpart draws its own cross-validation groups on every call; those
        # results are not used here but the draws keep the random stream in
        # step with R
        if m.K and m.K > 1:
            r_sample_xgroups(idx.size, int(m.K), rng=rng)
        return rpart_fit(
            X[idx], ncat, y[idx], wt[idx],
            "class" if classification else "anova", m.evar, m.xlevels,
            ylevels=m.ylevels, prior=prior_vec, loss=loss,
            split=1 if m.split == "gini" else 2, minsplit=m.minsplit,
            minbucket=m.minbucket, cp=m.cp, xval=0,
        )

    out = []
    for p in pcp:
        perf = np.zeros(K * repeats)
        if trace:
            print(f"Working on cp {m.cp} pcp {p:g}")
        for j in range(repeats):
            rand = r_sample_int_replace(K, n, rng)
            for k in range(1, K + 1):
                train = np.nonzero(rand != k)[0]
                test = np.nonzero(rand == k)[0]
                fit = fit_tree(train)
                if len(fit.frame) > 1:
                    fit = fit.prune(p)
                if prior_vec is not None:
                    bp = float(np.mean(y[train] == 1))
                    fit.yprob_adjust = (float(m.prior), bp)
                    fit.build_frame()
                where = fit.predict_where(X[test])
                if classification:
                    nclass = len(m.ylevels)
                    pred = np.array([fit.frame[w]["yval2"][1 + nclass] for w in where])
                    perf[k - 1 + j * K] = fun(pred, (y[test] == 1).astype(int), **kwargs)
                else:
                    pred = np.array([fit.frame[w]["yval"] for w in where])
                    perf[k - 1 + j * K] = fun(pred, y[test], **kwargs)
        out.append(
            [perf.mean(), perf.std(ddof=1) if perf.size > 1 else np.nan,
             perf.min(), perf.max(), m.cp, p]
        )
    res = pl.DataFrame(
        out, schema=[cn, "std", "min", "max", "cp", "pcp"], orient="row"
    ).sort(cn, descending=higher_better, maintain_order=True)

    best = res.row(0, named=True)
    fit_tree(np.arange(n))  # keeps the random stream aligned with R's final refit
    full = rpart_fit(
        X, ncat, y, wt, "class" if classification else "anova", m.evar, m.xlevels,
        ylevels=m.ylevels, prior=prior_vec, loss=loss,
        split=1 if m.split == "gini" else 2, minsplit=m.minsplit,
        minbucket=m.minbucket, cp=m.cp, xval=0,
    )
    if len(full.frame) > 1:
        full = full.prune(best["pcp"])
    nnodes = int(full.cptable[:, 1].max()) + 1
    print(
        "\nGiven the provided tuning grid, the pruning complexity parameter\n"
        f"should be set to {best['pcp']:g} or the number of nodes set to {nnodes}"
    )
    return res
