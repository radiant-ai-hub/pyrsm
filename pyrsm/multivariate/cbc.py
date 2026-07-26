"""Choice-based conjoint (CBC) analysis.

Fits a McFadden conditional-logit (multinomial-logit) choice model to long-format
choice data — one row per alternative per choice task, with a binary ``chosen``
indicator. Categorical attributes are treatment-coded (first level as the base,
part-worth 0), so the output mirrors :class:`pyrsm.multivariate.conjoint`: a
part-worth table, importance weights, and regression coefficients, plus
choice-probability prediction for new product profiles.

This is a Python port of the ``ConjointChoice`` tooling in
``pyrsm_streamlit/modules/conjoint`` (which wrapped the ``choicemodels``
library). The model is fit directly with a small, robust conditional-logit
estimator (analytic gradient and Hessian via SciPy) so there is no extra
dependency and the result objects follow the ``pyrsm.multivariate`` conventions
(``summary`` / ``plot`` / ``predict`` / ``store``).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pyrsm.utils import format_nr

from ._plotting import as_plot_result
from ._utils import apply_filter, get_data, get_vars

__all__ = ["cbc"]


def _to_list(v):
    if v is None or v == "":
        return []
    if isinstance(v, str):
        return [v]
    return list(v)


def _segment_probs(xb: np.ndarray, codes: np.ndarray, ngroups: int):
    """Softmax of ``xb`` within each task (vectorized, numerically stable)."""
    seg_max = np.full(ngroups, -np.inf)
    np.maximum.at(seg_max, codes, xb)
    ex = np.exp(xb - seg_max[codes])
    seg_sum = np.bincount(codes, weights=ex, minlength=ngroups)
    return ex / seg_sum[codes], seg_max, seg_sum


def _mnl_fit(X: np.ndarray, y: np.ndarray, codes: np.ndarray, ngroups: int):
    """Fit a conditional (multinomial) logit by maximum likelihood.

    Parameters
    ----------
    X : (n, p) design matrix
    y : (n,) binary chosen indicator (exactly one 1 per task)
    codes : (n,) contiguous task code (0..ngroups-1) for each row
    ngroups : number of choice tasks

    Returns a dict with ``params``, ``bse``, ``zvalues``, ``pvalues``, ``llf``,
    ``llnull`` (uniform-within-task baseline) and ``prsquared`` (McFadden). The
    likelihood, gradient and Hessian are computed with segment (vectorized)
    operations — no per-task Python loop.
    """
    import warnings

    from scipy.optimize import minimize
    from scipy.stats import norm

    p = X.shape[1]
    chosen = y == 1
    Xc_sum = X[chosen].sum(0)  # sum of chosen rows' covariates

    def nll_grad(beta):
        xb = X @ beta
        prob, seg_max, seg_sum = _segment_probs(xb, codes, ngroups)
        # log-likelihood: sum_chosen xb - sum_tasks log-sum-exp
        f = -(xb[chosen].sum() - (seg_max + np.log(seg_sum)).sum())
        grad = -(Xc_sum - (X * prob[:, None]).sum(0))
        return f, grad

    with warnings.catch_warnings():
        # benign overflow can occur in BFGS's line-search step-norm; the
        # likelihood itself is max-stabilized, so ignore the noise.
        warnings.simplefilter("ignore", RuntimeWarning)
        res = minimize(nll_grad, np.zeros(p), jac=True, method="BFGS")
    beta = res.x

    # analytic Hessian: sum_n p_n x_n x_n' - sum_g xbar_g xbar_g'
    xb = X @ beta
    prob, _, _ = _segment_probs(xb, codes, ngroups)
    XB = np.zeros((ngroups, p))
    np.add.at(XB, codes, X * prob[:, None])
    H = (X * prob[:, None]).T @ X - XB.T @ XB
    try:
        bse = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
    except np.linalg.LinAlgError:
        bse = np.full(p, np.nan)

    zvalues = beta / bse
    pvalues = 2 * (1 - norm.cdf(np.abs(zvalues)))
    llf = -res.fun
    # null model: equal probability within each task
    llnull = -float(np.log(np.bincount(codes, minlength=ngroups)).sum())
    prsquared = 1 - llf / llnull if llnull != 0 else np.nan
    return {
        "params": beta,
        "bse": bse,
        "zvalues": zvalues,
        "pvalues": pvalues,
        "llf": float(llf),
        "llnull": float(llnull),
        "prsquared": float(prsquared),
        "converged": bool(res.success),
    }


class cbc:
    """Choice-based conjoint (conditional-logit) analysis.

    Parameters
    ----------
    data : dataset (or ``{name: df}``) in long form — one row per alternative
        per choice task.
    rvar : str
        Binary chosen indicator (1 if the alternative was chosen, else 0).
    evar : str | list[str]
        Explanatory attributes; supports ``"a:b"`` ranges.
    id : str
        Choice-task identifier (the grouping variable; each task has one chosen
        alternative).
    alt : str, default "none"
        Optional alternative identifier (used to label predictions).
    int : str | list[str], default ""
        Interaction terms, e.g. ``"brand:price"``.
    data_filter : str, default ""

    Attributes
    ----------
    coeff : pl.DataFrame
        Regression coefficients with radiant-style labels, std. errors, z- and
        p-values.
    PW : pl.DataFrame
        Part-worth table (Attributes, Levels, PW); base level = 0.
    IW : pl.DataFrame
        Importance weights (per-attribute part-worth range / total range).
    """

    def __init__(
        self,
        data,
        rvar: str,
        evar,
        id: str,
        alt: str = "none",
        int="",
        data_filter: str = "",
    ) -> None:
        self.name, self.data = get_data(data)
        self.rvar = rvar
        self.evar = get_vars(self.data.columns, evar)
        self.id = id
        self.alt = alt
        self.int = _to_list(int)
        self.data_filter = data_filter

        sel = [rvar, id] + self.evar + ([alt] if alt != "none" else [])
        sub = apply_filter(self.data, data_filter).select(sel).drop_nulls()
        self.nobs = sub.height
        self.ntasks = sub[id].n_unique()

        # capture each attribute's level order (Enum order, else order of
        # appearance); first level is the treatment-coding base
        self._levels = {}
        self._is_cat = {}
        for v in self.evar:
            dt = sub[v].dtype
            if isinstance(dt, pl.Enum):
                self._levels[v] = dt.categories.to_list()
                self._is_cat[v] = True
            elif dt in (pl.Utf8, pl.String, pl.Categorical):
                self._levels[v] = sub[v].cast(pl.Utf8).unique(maintain_order=True).to_list()
                self._is_cat[v] = True
            else:
                self._levels[v] = None
                self._is_cat[v] = False

        self._sub = sub
        X, self._xnames = self._design(sub)
        y = sub[rvar].cast(pl.Float64).to_numpy()
        _, codes = np.unique(sub[id].to_numpy(), return_inverse=True)
        codes = codes.astype(np.intp)
        ngroups = (codes.max() + 1) if len(codes) else 0

        self.fit = _mnl_fit(X, y, codes, ngroups)
        self.coeff = pl.DataFrame(
            {
                "label": self._xnames,
                "coefficient": self.fit["params"],
                "std.error": self.fit["bse"],
                "z.value": self.fit["zvalues"],
                "p.value": self.fit["pvalues"],
            }
        )
        self.PW, self.IW = self._part_worths()

    # -- design ---------------------------------------------------------------

    def _term_columns(self, sub, term):
        """Return ``(names, columns)`` for a single attribute (treatment coded)."""
        names, cols = [], []
        if self._is_cat[term]:
            s = sub[term].cast(pl.Utf8).to_numpy()
            for lev in self._levels[term][1:]:  # drop first level (base)
                names.append(f"{term}|{lev}")
                cols.append((s == str(lev)).astype(float))
        else:
            names.append(term)
            cols.append(sub[term].cast(pl.Float64).to_numpy())
        return names, cols

    def _design(self, sub):
        names, cols = [], []
        for v in self.evar:
            n, c = self._term_columns(sub, v)
            names += n
            cols += c
        # interactions: product of the coded columns of each operand
        lookup = dict(zip(names, cols))
        for term in self.int:
            parts = term.split(":")
            base_names = [[nm for nm in names if nm == p or nm.startswith(f"{p}|")] for p in parts]
            # cartesian product across operands
            combos = [[]]
            for grp in base_names:
                combos = [c + [nm] for c in combos for nm in grp]
            for combo in combos:
                prod = np.ones(sub.height)
                for nm in combo:
                    prod = prod * lookup[nm]
                iname = ":".join(combo)
                names.append(iname)
                cols.append(prod)
                lookup[iname] = prod
        return np.column_stack(cols), names

    def _part_worths(self):
        params = dict(zip(self._xnames, self.fit["params"]))
        attrs, levels, pws = [], [], []
        ranges = {}
        for v in self.evar:
            if self._is_cat[v]:
                pw_vals = []
                for k, lev in enumerate(self._levels[v]):
                    pw = 0.0 if k == 0 else params.get(f"{v}|{lev}", 0.0)
                    attrs.append(v)
                    levels.append(str(lev))
                    pws.append(pw)
                    pw_vals.append(pw)
                ranges[v] = max(pw_vals) - min(pw_vals)
            else:
                # numeric attribute: a single slope coefficient
                coef = params.get(v, 0.0)
                attrs.append(v)
                levels.append("(slope)")
                pws.append(coef)
                ranges[v] = abs(coef)
        PW = pl.DataFrame(
            {"Attributes": attrs, "Levels": levels, "PW": [round(x, 3) for x in pws]}
        )
        total = sum(ranges.values())
        IW = pl.DataFrame(
            {
                "Attributes": list(ranges.keys()),
                "IW": [round(ranges[v] / total, 3) if total else 0.0 for v in ranges],
            }
        )
        return PW, IW

    # -- output ---------------------------------------------------------------

    def summary(self, dec: int = 3) -> None:
        print("Choice-based conjoint analysis")
        print(f"Data                 : {self.name}")
        if self.data_filter:
            print(f"Filter               : {self.data_filter}")
        print(f"Response variable    : {self.rvar}")
        print(f"Choice-task id       : {self.id}")
        print(f"Explanatory variables: {', '.join(self.evar)}")
        print(f"Choice tasks         : {format_nr(self.ntasks, dec=0)}")
        print(f"Observations         : {format_nr(self.nobs, dec=0)}\n")

        print("Conjoint part-worths:")
        print(self.PW.with_columns(pl.col("PW").round(dec)))
        print("\nConjoint importance weights:")
        print(self.IW.with_columns(pl.col("IW").round(dec)))
        print("\nConjoint choice-model results:\n")
        print(
            self.coeff.with_columns(
                [pl.col(c).round(dec) for c in ("coefficient", "std.error", "z.value", "p.value")]
            )
        )
        print(
            f"\nLog-likelihood: {round(self.fit['llf'], dec)}, "
            f"McFadden pseudo-R-squared: {round(self.fit['prsquared'], dec)}"
        )
        if not self.fit["converged"]:
            print("** Warning: the optimizer did not fully converge **")

    def predict(self, data=None, pred_cmd=None, dec: int = 3):
        """Predict choice probabilities for product profiles.

        With ``data`` (or a ``pred_cmd`` expand-grid dict / string) the profiles
        are scored; probabilities are the conditional-logit softmax of the
        predicted utilities within each choice task (``id``). When neither is
        given the estimation data is scored. Returns a Polars DataFrame with the
        task id, optional alternative id, utility, and probability.
        """
        from .conjoint import _parse_pred_cmd

        if pred_cmd is not None:
            import itertools

            cmd = _parse_pred_cmd(pred_cmd) if isinstance(pred_cmd, str) else pred_cmd
            keys = list(cmd.keys())
            grids = [cmd[k] if isinstance(cmd[k], (list, tuple)) else [cmd[k]] for k in keys]
            combos = list(itertools.product(*grids))
            sub = pl.DataFrame({k: [c[i] for c in combos] for i, k in enumerate(keys)})
            # a single task containing all generated profiles
            if self.id not in sub.columns:
                sub = sub.with_columns(pl.lit(1).alias(self.id))
            if self.alt != "none" and self.alt not in sub.columns:
                sub = sub.with_columns(pl.Series(self.alt, list(range(1, sub.height + 1))))
            sub = self._coerce_levels(sub)
        elif data is None:
            sub = self._sub
        else:
            _, d = get_data(data)
            sub = self._coerce_levels(d)

        X, _ = self._design(sub)
        util = X @ self.fit["params"]
        _, codes = np.unique(sub[self.id].to_numpy(), return_inverse=True)
        codes = codes.astype(np.intp)
        prob, _, _ = _segment_probs(util, codes, codes.max() + 1)

        out = {self.id: sub[self.id].to_list()}
        if self.alt != "none" and self.alt in sub.columns:
            out[self.alt] = sub[self.alt].to_list()
        out["utility"] = util
        out["probability"] = prob
        return pl.DataFrame(out).with_columns(
            [pl.col("utility").round(dec), pl.col("probability").round(dec)]
        )

    def _coerce_levels(self, df: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        for v in self.evar:
            if self._is_cat[v] and v in df.columns:
                exprs.append(
                    pl.col(v).cast(pl.Utf8).cast(pl.Enum([str(x) for x in self._levels[v]]))
                )
        return df.with_columns(exprs) if exprs else df

    def store(self, what: str = "PW") -> pl.DataFrame:
        """Return the part-worth (``"PW"``) or importance-weight (``"IW"``) table.

        To store predicted choice probabilities, call :meth:`predict` (the result
        aligns row-for-row with the estimation data) and add the column, e.g.
        ``df.with_columns(cbc_model.predict()["probability"])``.
        """
        return self.PW if what.upper() == "PW" else self.IW

    def plot(self, plots="pw", scale_plot: bool = False, custom: bool = False):
        """Part-worth ("pw") and/or importance-weight ("iw") plots (plotnine).

        Mirrors :meth:`conjoint.plot`: one line/point panel per categorical
        attribute for ``"pw"`` and a bar plot for ``"iw"``. Returns a single
        plot when one is produced, otherwise a list.
        """
        from plotnine import (
            aes,
            element_text,
            geom_bar,
            geom_line,
            geom_point,
            ggplot,
            labs,
            theme,
            theme_classic,
        )

        if isinstance(plots, str):
            plots = [plots]
        out = []
        if "pw" in plots:
            for v in self.evar:
                if not self._is_cat[v]:
                    continue  # numeric attributes have a single slope, no curve
                dat = (
                    self.PW.filter(pl.col("Attributes") == v)
                    .with_columns(pl.col("Levels").cast(pl.Utf8))
                    .to_pandas()
                )
                out.append(
                    ggplot(dat, aes(x="Levels", y="PW", group=1))
                    + geom_line(color="blue", linetype="dashdot", size=0.7)
                    + geom_point(color="blue", size=4, fill="white")
                    + labs(title=f"Part-worths for {v}", x="", y="PW")
                    + theme_classic()
                    + theme(axis_text_x=element_text(angle=45, hjust=1))
                )
        if "iw" in plots:
            dat = self.IW.to_pandas()
            out.append(
                ggplot(dat, aes(x="Attributes", y="IW", fill="Attributes"))
                + geom_bar(stat="identity", alpha=0.5)
                + labs(title="Importance weights", x="", y="IW")
                + theme_classic()
                + theme(legend_position="none")
            )
        if not out:
            raise ValueError("plots must include 'pw' and/or 'iw'")
        return as_plot_result(out)
