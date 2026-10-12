"""Multinomial logistic regression (MNL), a port of ``radiant.model::mnl``.

A categorical response with two or more levels is explained by variables that
describe each observation (e.g., the prices and promotions a shopper saw).
Every non-base level of the response gets its own intercept and coefficient
for each explanatory variable; ``exp(coefficient)`` is the relative risk ratio
(RRR): the factor by which the odds of that level *relative to the base level*
change for a one-unit increase in the variable.

The model is estimated with pyrsm's logit engine (``pyrsm._logit``), the same
optimizer that fits the conditional logit in
``pyrsm.multivariate.conjoint_choice``: a multinomial logit is a conditional
logit on a long design with one row per response level. Estimates match
R's ``nnet::multinom`` (tests/test_mnl.py).

Examples
--------
>>> import polars as pl
>>> import pyrsm as rsm
>>> ketchup = pl.read_parquet("examples/data/model/ketchup.parquet")
>>> m = rsm.model.mnl(
...     {"ketchup": ketchup}, rvar="choice", lev="heinz28",
...     evar=["price.heinz28", "price.heinz32", "price.heinz41", "price.hunts32"],
... )
>>> m.summary(rrr=True)
>>> m.predict(cmd={"price.heinz28": [3.4, 4.3, 5.2]})
>>> m.plot("coef")
"""

from __future__ import annotations

import math
import re
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from pyrsm._logit import mnl_fit, mnl_probs
from pyrsm.basics import display_utils as du
from pyrsm.data_scope import apply_scope, print_scope
from pyrsm.model.model import sig_stars, sim_prediction
from pyrsm.utils import cat_levels, check_dataframe

__all__ = ["mnl"]

#: Where a categorical variable's levels sit on the shared x-axis of the
#: prediction and partial dependence plots (far from any data value).
_CAT_OFFSET = 1e9

_POWER = re.compile(r"^I\(\s*([^\s^*()]+)\s*(?:\^|\*\*)\s*(\d+)\s*\)$")


def _as_list(v) -> list:
    if v is None or (isinstance(v, str) and v in ("", "None")):
        return []
    if isinstance(v, str):
        return [v]
    return list(v)


def _is_categorical(s: pl.Series) -> bool:
    return isinstance(s.dtype, (pl.Enum, pl.Categorical)) or s.dtype in (
        pl.Utf8,
        pl.String,
        pl.Boolean,
    )


def _levels(s: pl.Series) -> list[str]:
    """Levels in Enum order (text: order of appearance), only levels that occur."""
    if s.dtype == pl.Boolean:
        present = set(s.cast(pl.Utf8).drop_nulls().unique().to_list())
        return [c for c in ("false", "true") if c in present]
    if s.dtype.is_numeric():
        return [str(v) for v in sorted(s.drop_nulls().unique().to_list())]
    return [str(v) for v in cat_levels(s, drop_unused=True)]


def _term_label(term: str) -> str:
    """A term as R prints it: ``I(x^2)`` rather than ``I(x ** 2)``."""
    m = _POWER.match(term.strip())
    return f"I({m.group(1)}^{m.group(2)})" if m else term.strip()


def _r_num(x: float) -> str:
    """A number as R's ``cat`` prints it (7 significant digits)."""
    return f"{float(f'{x:.7g}'):.7g}"


class mnl:
    """
    Multinomial logistic regression (MNL).

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame]
        Dataset; pass ``{"name": df}`` to show the name in the summary.
    rvar : str
        Categorical response variable with two or more levels.
    lev : str, optional
        Base level of ``rvar``: every coefficient compares a level with this
        one. Defaults to the first level.
    evar : list[str]
        Explanatory variables. Categorical variables get a coefficient per
        level (first level = base); numeric variables one slope.
    ivar : list[str], optional
        Interactions (``"price.heinz28:disp.heinz28"``) and powers of numeric
        variables (``"I(price.heinz28^2)"``).
    weights : str, optional
        Frequency (case) weights.
    data_filter : str, default ""
        Expression used to filter the data before the analysis, e.g.,
        "price.heinz28 > 4 and disp.heinz28 == 'yes'". A bare name is a column
        and a call is an operation on one (``is_null(price)``). See
        ``pyrsm.data_scope`` for the full list.
    sort : str, default ""
        Column(s) to sort on, "-" in front for descending, e.g.,
        "id -price.heinz28". Applied after the filter.
    slice : str, default ""
        Rows to keep, counting from 1, e.g., "1:100". Applied after the filter
        and the sort.
    standardize : bool or int, default False
        Standardize the numeric explanatory variables by 2 (``True``) or 1
        standard deviations (weighted when ``weights`` is set), as
        radiant.model. RRRs are then per 2 (or 1) standard deviations.
    center : bool, default False
        Mean-center the numeric explanatory variables. Ignored when
        ``standardize`` is set.
    intercept : bool, default True
        Include an intercept for every non-base level.
    stepwise : bool or str, default False
        Stepwise selection by AIC as R's ``step()``: ``"backward"`` (or
        ``True``), ``"forward"`` or ``"both"``. ``summary()`` prints the trace.

    Attributes
    ----------
    coef : pl.DataFrame
        One row per response level and term: RRR, coefficient, standard error,
        z- and p-value.
    fit_stats : dict
        Log-likelihood, pseudo R-squared, AIC, BIC and the chi-squared test
        against the intercept-only model.
    levels : list[str]
        Response levels, base level first.
    """

    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        rvar: str | None = None,
        lev: str | None = None,
        evar: list[str] | str | None = None,
        ivar: list[str] | str | None = None,
        weights: str | None = None,
        data_filter: str = "",
        sort: str = "",
        slice: str = "",
        standardize: bool | int | float = False,
        center: bool = False,
        intercept: bool = True,
        stepwise: bool | str = False,
    ) -> None:
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            data = data[self.name]
        else:
            self.name = "Not provided"
        data = check_dataframe(data)
        self.data_filter, self.sort, self.slice = data_filter, sort, slice
        data = apply_scope(data, data_filter, sort, slice)

        self.rvar = rvar
        self.evar = _as_list(evar)
        self.ivar = [t.strip() for t in _as_list(ivar)]
        if rvar is None or not self.evar:
            raise ValueError("Please specify rvar and evar")
        if rvar in self.evar:
            raise ValueError(
                "Response variable contained in the set of explanatory variables. "
                "Please update model specification."
            )
        for term in self.ivar:
            parts = self._term_vars(term)
            missing = [p for p in parts if p not in self.evar]
            if missing:
                raise ValueError(
                    f"Term '{term}' uses variables not in evar: {', '.join(missing)}"
                )

        self.weights_name = None if weights in (None, "", "None") else weights
        if standardize is True:
            standardize = 2
        self.standardize = standardize or 0
        self.center = bool(center) and not self.standardize
        self.intercept = bool(intercept)
        if stepwise is True:
            stepwise = "backward"
        if stepwise and stepwise not in ("backward", "forward", "both"):
            raise ValueError("stepwise must be 'backward', 'forward' or 'both'")
        self.stepwise = stepwise or ""

        cols = [rvar] + self.evar + ([self.weights_name] if self.weights_name else [])
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise ValueError(f"Variables not in the data: {', '.join(missing)}")
        self.nobs_all = data.height
        data = data.select(cols).drop_nulls()
        self.nobs_dropped = self.nobs_all - data.height

        not_vary = [c for c in [rvar] + self.evar if data[c].n_unique() < 2]
        if not_vary:
            raise ValueError(
                "The following variable(s) show no variation. Please select other "
                f"variables: {', '.join(not_vary)}"
            )

        # response levels, base level first (R's relevel)
        rv = data[rvar]
        levs = _levels(rv)
        lev = levs[0] if lev in (None, "") else str(lev)
        if lev not in levs:
            raise ValueError(f"Level '{lev}' is not a level of {rvar}: {', '.join(levs)}")
        self.lev = lev
        self.levels = [lev] + [x for x in levs if x != lev]
        y_str = rv.cast(pl.Utf8).to_numpy() if rv.dtype != pl.Boolean else (
            rv.cast(pl.Utf8).str.to_lowercase().to_numpy()
        )
        code = {x: i for i, x in enumerate(self.levels)}
        self._y = np.array([code[v] for v in y_str], dtype=np.intp)

        self.xlevels = {v: _levels(data[v]) for v in self.evar if _is_categorical(data[v])}
        if self.weights_name:
            w = data[self.weights_name].cast(pl.Float64).to_numpy()
            if np.any(w < 0):
                raise ValueError("Weights cannot be negative")
            self._w = w
        else:
            self._w = None

        self.data = data
        self.scale_means: dict[str, float] = {}
        self.scale_sds: dict[str, float] = {}
        if self.standardize or self.center:
            self._calc_scaling()

        # the model: every term (stepwise may drop some)
        self.terms = self.evar + self.ivar
        self.removed: list[str] = []
        self.added: list[str] = []
        self.step_trace = ""
        self.step_steps: list[dict] = []
        if self.stepwise:
            self.terms = self._step()
            self.ivar = [t for t in self.ivar if t in self.terms]
        self._fit_terms(self.terms)

    # -- design ----------------------------------------------------------------
    @staticmethod
    def _term_vars(term: str) -> list[str]:
        m = _POWER.match(term)
        if m:
            return [m.group(1)]
        return [p.strip() for p in term.split(":")]

    def _calc_scaling(self) -> None:
        """Means and standard deviations of the numeric explanatory variables.

        R's ``scale_df``: ``sd`` without weights, a weighted population standard
        deviation with them.
        """
        for c in self.evar:
            if c in self.xlevels:
                continue
            x = self.data[c].cast(pl.Float64).to_numpy()
            if self._w is not None:
                ww = self._w / self._w.sum()
                mean = float(np.sum(ww * x))
                sd = float(np.sqrt(np.sum(ww * (x - mean) ** 2)))
            else:
                mean = float(np.mean(x))
                sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
            self.scale_means[c] = mean
            self.scale_sds[c] = sd

    def _numeric(self, df: pl.DataFrame, v: str) -> np.ndarray:
        x = df[v].cast(pl.Float64).to_numpy()
        if v in self.scale_means:
            x = x - self.scale_means[v]
            sd = self.scale_sds[v]
            if self.standardize and sd > 0:
                x = x / (self.standardize * sd)
        return x

    def _var_columns(self, df: pl.DataFrame, v: str):
        """Treatment-coded columns for one explanatory variable."""
        if v in self.xlevels:
            s = df[v]
            vals = (
                s.cast(pl.Utf8).str.to_lowercase() if s.dtype == pl.Boolean else s.cast(pl.Utf8)
            ).to_numpy()
            unknown = set(vals) - set(self.xlevels[v])
            if unknown:
                raise ValueError(
                    f"Level(s) {', '.join(map(str, sorted(unknown)))} of {v} were not in "
                    "the estimation data"
                )
            return [(f"{v}[{lv}]", (vals == lv).astype(float)) for lv in self.xlevels[v][1:]]
        return [(v, self._numeric(df, v))]

    def _design(self, df: pl.DataFrame, terms: list[str]):
        names, cols = [], []
        if self.intercept:
            names.append("Intercept")
            cols.append(np.ones(df.height))
        coded = {}
        self._term_cols: dict[str, list[int]] = {}
        for term in terms:
            start = len(names)
            m = _POWER.match(term)
            if m:
                v, k = m.group(1), int(m.group(2))
                names.append(_term_label(term))
                cols.append(self._numeric(df, v) ** k)
            elif ":" in term:
                combos = [("", np.ones(df.height))]
                for part in self._term_vars(term):
                    if part not in coded:
                        coded[part] = self._var_columns(df, part)
                    combos = [
                        (f"{a}:{b}" if a else b, x * z) for a, x in combos for b, z in coded[part]
                    ]
                for n, c in combos:
                    names.append(n)
                    cols.append(c)
            else:
                if term not in coded:
                    coded[term] = self._var_columns(df, term)
                for n, c in coded[term]:
                    names.append(n)
                    cols.append(c)
            self._term_cols[term] = list(range(start, len(names)))
        X = np.column_stack(cols) if cols else np.zeros((df.height, 0))
        return X, names

    # -- estimation ------------------------------------------------------------
    def _estimate(self, terms: list[str]) -> dict:
        X, names = self._design(self.data, terms)
        if X.shape[1] == 0:
            # no intercept and no terms: every level equally likely
            k = len(self.levels)
            w = np.ones(len(self._y)) if self._w is None else self._w
            return {"llf": -float(w.sum() * math.log(k)), "params": np.zeros(0), "names": [],
                    "X": X, "B": np.zeros((0, k - 1)), "bse": np.zeros(0),
                    "cov": np.zeros((0, 0)), "llnull": None, "converged": True}
        fit = mnl_fit(X, self._y, len(self.levels), self._w)
        fit["names"] = names
        fit["X"] = X
        return fit

    def _nobs_for_bic(self) -> float:
        """Rows, or the sum of the weights for frequency weights (as radiant.model)."""
        if self._w is None:
            return float(len(self._y))
        w = self._w
        if len(np.unique(w)) > 2 or w.min() >= 1:
            return float(w.sum())
        return float(len(self._y))

    def _fit_terms(self, terms: list[str]) -> None:
        from scipy.stats import chi2, norm

        fit = self._estimate(terms)
        self._fit = fit
        self.converged = fit["converged"]
        names = fit["names"]
        p = len(names)
        M = len(self.levels) - 1
        params, bse = fit["params"], fit["bse"]
        z = params / bse
        pvals = 2 * norm.sf(np.abs(z))
        is_int = np.array([n == "Intercept" for n in names] * M, dtype=bool)
        rrr = np.exp(params)
        rrr_pct = 100 * (rrr - 1)
        self.coef = pl.DataFrame(
            {
                "level": [lv for lv in self.levels[1:] for _ in range(p)],
                "index": names * M,
                "RRR": pl.Series(np.where(is_int, np.nan, rrr), dtype=pl.Float64).fill_nan(None),
                "RRR%": pl.Series(np.where(is_int, np.nan, rrr_pct), dtype=pl.Float64).fill_nan(None),
                "coefficient": params,
                "std.error": bse,
                "z.value": z,
                "p.value": pvals,
                "  ": sig_stars(pvals),
            }
        )

        w = np.ones(len(self._y)) if self._w is None else self._w
        wk = np.bincount(self._y, weights=w, minlength=len(self.levels))
        share = wk[wk > 0] / wk.sum()
        llnull = float(np.sum(wk[wk > 0] * np.log(share)))
        llf = fit["llf"]
        k = p * M
        nobs = self._nobs_for_bic()
        lr = 2 * (llf - llnull)
        # the intercept-only model has one parameter per non-base level
        df = k - M if self.intercept else k
        self.fit_stats = {
            "loglik": llf,
            "loglik_null": llnull,
            "pseudo_rsq": 1 - llf / llnull,
            "pseudo_rsq_adj": 1 - (llf - k) / llnull,
            "AIC": -2 * llf + 2 * k,
            "BIC": -2 * llf + k * math.log(nobs),
            "chisq": lr,
            "df": df,
            "chisq_pvalue": float(chi2.sf(lr, df)) if df > 0 else float("nan"),
            "nparams": k,
            "nobs": nobs,
        }

    def _step(self) -> list[str]:
        """Stepwise selection by AIC with R's ``step()`` rules and trace."""
        from pyrsm.model.model import _step_block, _step_record

        upper = list(self.evar) + list(self.ivar)
        factors = {t: set(self._term_vars(t)) if ":" in t else {t} for t in upper}
        degree = {t: (len(factors[t]) if ":" in t else 1) for t in upper}
        order = {t: i for i, t in enumerate(upper)}

        def sort_terms(terms):
            return sorted(terms, key=lambda t: (degree[t], order[t]))

        def formula_for(terms):
            rhs = " + ".join(_term_label(t) for t in sort_terms(terms)) if terms else "1"
            if not self.intercept:
                rhs += " - 1"
            return f"{self.rvar} ~ {rhs}"

        def droppable(t, current):
            return not any(factors[t] < factors[o] for o in current if o != t)

        def addable(t, current):
            return all(o in current for o in upper if factors[o] < factors[t])

        def score(terms):
            fit = self._estimate(sort_terms(terms))
            k = len(fit["names"]) * (len(self.levels) - 1)
            return k, -2 * fit["llf"] + 2 * k

        current = list(upper) if self.stepwise == "backward" else []
        edf, aic = score(current)
        trace, label = [], "Start"
        self.step_steps = []
        while True:
            table = [("<none>", edf, aic, current)]
            if self.stepwise in ("backward", "both"):
                for t in current:
                    if droppable(t, current):
                        keep = [x for x in current if x != t]
                        table.append((f"- {_term_label(t)}", *score(keep), keep))
            if self.stepwise in ("forward", "both"):
                for t in upper:
                    if t not in current and addable(t, current):
                        keep = current + [t]
                        table.append((f"+ {_term_label(t)}", *score(keep), keep))
            table.sort(key=lambda r: r[2])
            trace.append(
                _step_block(label, aic, formula_for(current), ["Df"], [r[:3] for r in table])
            )
            self.step_steps.append(
                _step_record(label, aic, formula_for(current), ["Df"], [r[:3] for r in table])
            )
            label = "Step"
            best = table[0]
            if best[0] == "<none>" or best[2] >= aic - 1e-7:
                break
            changed = next(t for t in upper if _term_label(t) == best[0][2:])
            (self.removed if best[0].startswith("-") else self.added).append(changed)
            current = best[3]
            edf, aic = score(current)
        self.step_trace = "\n\n".join(trace)
        self.formula = formula_for(current)
        return sort_terms(current)

    # -- output ----------------------------------------------------------------
    def conf_int(self, conf: float = 0.95) -> pl.DataFrame:
        """Wald confidence intervals for the coefficients (as R's ``confint``)."""
        from scipy.stats import norm

        zc = norm.ppf(1 - (1 - conf) / 2)
        c = self.coef
        low = c["coefficient"] - zc * c["std.error"]
        high = c["coefficient"] + zc * c["std.error"]
        lo, hi = [f"{round(100 * q, 4):g}%" for q in ((1 - conf) / 2, 1 - (1 - conf) / 2)]
        return pl.DataFrame(
            {"level": c["level"], "index": c["index"], "coefficient": c["coefficient"],
             lo: low, hi: high, "+/-": high - c["coefficient"]}
        )

    def summary(
        self,
        main: bool = True,
        fit: bool = True,
        ci: bool = False,
        rrr: bool = False,
        confusion: bool = False,
        test: list[str] | str | None = None,
        conf: float = 0.95,
        dec: int = 3,
        plain: bool = True,
    ) -> None:
        """
        Summarize the multinomial logistic regression

        Parameters
        ----------
        main : bool, default True
            Print the header and the coefficient table.
        fit : bool, default True
            Print the fit statistics.
        ci : bool, default False
            Print confidence intervals for the coefficients.
        rrr : bool, default False
            Print the relative risk ratios with confidence intervals.
        confusion : bool, default False
            Print the confusion matrix of the most likely level against the
            observed level, and the misclassification error.
        test : list[str], optional
            Variables to test: likelihood-ratio chi-squared test of the model
            without them (and the interactions that use them).
        conf : float, default 0.95
            Confidence level for ``ci`` and ``rrr``.
        dec : int, default 3
            Number of decimals.
        plain : bool, default True
            Plain text output (styled tables in a notebook with ``False``).
        """
        if main:
            if self.stepwise:
                print(self.step_trace)
                title = {"backward": "Backward", "forward": "Forward",
                         "both": "Forward and Backward"}[self.stepwise]
                line = "-" * 52
                print(f"\n{line}\n{title} stepwise selection of variables\n{line}")
            self._summary_header()
            self._summary_table(dec, plain)
            du.print_sig_codes()
        if fit:
            print(f"\n{self._fit_text(dec)}")
        if ci or rrr:
            cis = self.conf_int(conf)
            lo, hi = cis.columns[3], cis.columns[4]
            if ci:
                tab = cis.with_columns(pl.col(pl.Float64).round(dec))
                print("\nConfidence intervals:\n")
                du.print_plain_tables(tab)
            if rrr:
                lab = "std RRR" if self.standardize else "RRR"
                tab = (
                    cis.filter(pl.col("index") != "Intercept")
                    .select(
                        "level",
                        "index",
                        pl.col("coefficient").exp().round(dec).alias(lab),
                        pl.col(lo).exp().round(dec),
                        pl.col(hi).exp().round(dec),
                    )
                )
                print("\nRelative risk ratios:\n")
                du.print_plain_tables(tab)
        if test:
            self.chisq_test(test=test, dec=dec)
        if confusion:
            self._print_confusion(dec)

    def _summary_header(self) -> None:
        print("Multinomial logistic regression (MNL)")
        print(f"Data                 : {self.name}")
        print_scope(self, 21)
        print(f"Response variable    : {self.rvar}")
        print(f"Base level           : {self.lev} in {self.rvar}")
        print(f"Explanatory variables: {', '.join(self.evar)}")
        if self.ivar:
            print(f"Interactions         : {', '.join(_term_label(t) for t in self.ivar)}")
        if self.weights_name:
            print(f"Weights used         : {self.weights_name}")
        if self.stepwise:
            if self.stepwise == "backward":
                change = "removed: " + (", ".join(_term_label(t) for t in self.removed) or "none")
            else:
                change = "selected: " + (" + ".join(_term_label(t) for t in self.terms) or "none")
            print(f"Stepwise selection   : {self.stepwise} (AIC), {change}")
        expl = self.evar[0] if len(self.evar) == 1 else "x"
        print(f"Null hyp.: there is no effect of {expl} on {self.rvar}")
        print(f"Alt. hyp.: there is an effect of {expl} on {self.rvar}")
        if self.standardize:
            print(f"**Standardized RRRs and coefficients shown ({self.standardize} X SD)**")
        elif self.center:
            print("**Centered RRRs and coefficients shown (x - mean(x))**")
        if not self.converged:
            print(
                "\n**Model did NOT converge. Consider standardizing the explanatory "
                "variables and/or simplifying your model**"
            )
        if self.nobs_dropped:
            print(f"Rows with missing values dropped: {self.nobs_dropped:,}")
        print()

    def _summary_table(self, dec: int, plain: bool) -> None:
        df = self.coef.with_columns(
            pl.col("RRR").round(dec).cast(pl.Utf8).fill_null(""),
            (pl.col("RRR%").round(max(dec - 2, 0)).cast(pl.Utf8) + "%").fill_null("").alias("RRR%"),
            pl.col("coefficient").round(dec),
            pl.col("std.error").round(dec),
            pl.col("z.value").round(dec),
            pl.when(pl.col("p.value") < 0.001)
            .then(pl.lit("< .001"))
            .otherwise(pl.col("p.value").round(dec).cast(pl.Utf8))
            .alias("p.value"),
        )
        if not plain and du.is_notebook() and du.has_great_tables():
            from IPython.display import display

            display(
                du.style_table(
                    df,
                    title="Coefficient Estimates",
                    subtitle=f"Response: {self.rvar} (base level: {self.lev})",
                    number_cols=["coefficient", "std.error", "z.value"],
                    dec=dec,
                )
            )
        else:
            du.print_plain_tables(df)

    def _fit_text(self, dec: int = 3) -> str:
        fs = self.fit_stats
        pval = fs["chisq_pvalue"]
        ptxt = "< .001" if pval < 0.001 else f"{round(pval, dec)}"
        return "\n".join(
            [
                f"Pseudo R-squared (McFadden): {fs['pseudo_rsq']:.{dec}f}",
                f"Pseudo R-squared (McFadden adjusted): {fs['pseudo_rsq_adj']:.{dec}f}",
                f"Log-likelihood: {fs['loglik']:.{dec}f}, AIC: {fs['AIC']:.{dec}f}, "
                f"BIC: {fs['BIC']:.{dec}f}",
                f"Chi-squared: {fs['chisq']:.{dec}f}, df({fs['df']}), p.value {ptxt}",
                f"Nr obs: {fs['nobs']:,.0f}",
            ]
        )

    def lr_test(self, test: list[str] | str) -> dict:
        """Likelihood-ratio test of the model without the ``test`` variables.

        Interactions and powers that use a tested variable are dropped too.
        Returns the two models' terms, pseudo R-squared, chi-squared, df and
        p-value.
        """
        from scipy.stats import chi2

        drop = set(_as_list(test))
        sub = [
            t for t in self.terms
            if t not in drop and not (set(self._term_vars(t)) & drop and t not in self.evar)
        ]
        fit_sub = self._estimate(sub)
        M = len(self.levels) - 1
        df = (len(self._fit["names"]) - len(fit_sub["names"])) * M
        lr = 2 * (self._fit["llf"] - fit_sub["llf"])
        return {
            "terms_sub": sub,
            "terms_full": list(self.terms),
            "pseudo_rsq_sub": 1 - fit_sub["llf"] / self.fit_stats["loglik_null"],
            "pseudo_rsq_full": self.fit_stats["pseudo_rsq"],
            "chisq": lr,
            "df": df,
            "p.value": float(chi2.sf(lr, df)) if df > 0 else float("nan"),
        }

    def chisq_test(self, test: list[str] | str, dec: int = 3) -> None:
        """Print the likelihood-ratio test of the model without ``test``."""
        if self.stepwise:
            print("\nModel comparisons are not conducted when Stepwise has been selected.")
            return
        res = self.lr_test(test)

        def form(terms):
            rhs = " + ".join(_term_label(t) for t in terms) if terms else "1"
            return f"{self.rvar} ~ {rhs}"

        pval = res["p.value"]
        ptxt = "< .001" if pval < 0.001 else f"{round(pval, dec)}"
        print(f"\nModel 1: {form(res['terms_sub'])}")
        print(f"Model 2: {form(res['terms_full'])}")
        print(
            f"Pseudo R-squared, Model 1 vs 2: {res['pseudo_rsq_sub']:.{dec}f} vs "
            f"{res['pseudo_rsq_full']:.{dec}f}"
        )
        print(f"Chi-squared: {res['chisq']:.{dec}f} df ({res['df']}), p.value {ptxt}")

    def _probs(self, df: pl.DataFrame) -> np.ndarray:
        X, _ = self._design(df, self.terms)
        return mnl_probs(X, self._fit["B"])

    def confusion(self) -> pl.DataFrame:
        """Confusion matrix: most likely level (rows) by observed level (columns)."""
        P = self._probs(self.data)
        pred = P.argmax(axis=1)
        k = len(self.levels)
        w = np.ones(len(self._y)) if self._w is None else self._w
        tab = np.zeros((k, k))
        np.add.at(tab, (pred, self._y), w)
        out = pl.DataFrame({"predicted": self.levels})
        for j, lv in enumerate(self.levels):
            vals = tab[:, j]
            out = out.with_columns(
                pl.Series(lv, vals.astype(np.int64) if np.allclose(vals, vals.round()) else vals)
            )
        return out

    def _print_confusion(self, dec: int) -> None:
        tab = self.confusion()
        m = tab.select(self.levels).to_numpy().astype(float)
        err = 1 - np.trace(m) / m.sum()
        print("\nConfusion matrix (rows: predicted, columns: actual):\n")
        du.print_plain_tables(tab)
        print(f"\nMisclassification error: {100 * err:.{dec}f}%")

    def predict(self, data=None, cmd=None, data_cmd=None, dec=None) -> pl.DataFrame:
        """
        Predicted probabilities of every response level

        Parameters
        ----------
        data : pd.DataFrame | pl.DataFrame, optional
            Data to predict for (default: the estimation data).
        cmd : dict, optional
            Values to predict for, e.g. ``{"price.heinz28": [3.4, 4.3, 5.2]}``:
            every combination is a row; variables left out are held at their
            mean (numeric) or most common level.
        data_cmd : dict, optional
            Values that replace columns of ``data`` before predicting.
        dec : int, optional
            Round the probabilities and the numeric inputs.

        Returns
        -------
        pl.DataFrame
            The explanatory variables and one probability column per response
            level (base level first).
        """
        from pyrsm.model.model import check_pred_vars

        check_pred_vars(
            self.evar,
            data=None if data is None else check_dataframe(data),
            cmd=cmd if isinstance(cmd, dict) else None,
            data_cmd=data_cmd if isinstance(data_cmd, dict) else None,
        )
        base = self.data if data is None else check_dataframe(data)
        pred_data = base.select(self.evar)
        if data_cmd:
            pred_data = pred_data.with_columns(
                [pl.lit(v).alias(k) for k, v in data_cmd.items()]
            )
        elif cmd:
            cmd = {k: [v] if isinstance(v, str) or not hasattr(v, "__len__") else list(v)
                   for k, v in cmd.items()}
            pred_data = sim_prediction(data=self.data.select(self.evar), vary=cmd)
        # keep the estimation data's column types (Enum levels, numeric types)
        pred_data = pred_data.with_columns(
            [pl.col(c).cast(self.data[c].dtype) for c in self.evar
             if pred_data[c].dtype != self.data[c].dtype and c in self.xlevels]
        )
        P = self._probs(pred_data)
        out = pred_data.with_columns(
            [pl.Series(lv, P[:, i]) for i, lv in enumerate(self.levels)]
        )
        if dec is not None:
            out = out.with_columns(
                [pl.col(c).round(dec) for c in out.columns if out[c].dtype in (pl.Float64, pl.Float32)]
            )
        return out

    def plot(
        self,
        plots: Literal["coef", "rrr", "pred", "pdp", "pip", "dist", "corr"] = "coef",
        alpha: float = 0.05,
        intercept: bool = False,
        incl: list[str] | str | None = None,
        excl: list[str] | str | None = None,
        incl_int: list[str] | str | None = None,
        nobs: int = 1000,
        nnv: int = 20,
        minq: float = 0.025,
        maxq: float = 0.975,
        fix: bool = True,
        hline: bool = True,
        n_sample: int = 2000,
        rep: int = 10,
        ret: bool = False,
        ice: bool = False,
        ice_nobs: int = 100,
        figsize=None,
    ):
        """
        Plots for a multinomial logistic regression

        Parameters
        ----------
        plots : str
            ``"coef"`` (``"rrr"``): relative risk ratios with 1 - ``alpha``
            confidence intervals, colored by response level.
            ``"pred"``: prediction plots, the probability of every response level
            as one explanatory variable changes with the others at their mean
            or most common level. ``"pdp"``: partial dependence plots, the same
            averaged over (a sample of ``n_sample``) rows of the data.
            ``"pip"``: permutation importance, the decrease in McFadden's pseudo
            R-squared when a variable is shuffled (``rep`` times).
            ``"dist"``: distributions; ``"corr"``: correlations (``nobs`` rows).
        incl, excl : list[str], optional
            Explanatory variables to plot (default all) or leave out.
        incl_int : list[str], optional
            Two-way interactions to plot (``"a:b"``): ``a`` on the x-axis, a
            panel for each level of ``b`` (numeric: its 10th, 50th and 90th
            percentile).
        nnv : int
            Number of values of a numeric variable, between the ``minq`` and
            ``maxq`` quantiles of the data.
        fix : bool
            The same y-axis for every prediction or partial dependence plot.
        hline : bool
            Dashed lines at each level's share of the data.
        ret : bool
            ``"pip"`` only: return ``(plot, importance table)``.
        ice : bool
            ``"pdp"`` only: individual conditional expectation lines -- each
            sampled row's probability of every level as the variable moves --
            drawn faintly behind the averages, in every panel of an interaction too.
        ice_nobs : int
            Rows drawn as ICE lines, from the partial dependence sample
            (-1: all of them).
        """
        plots = _as_list(plots)
        if "dist" in plots:
            from pyrsm.eda.distr import distr

            return distr(self.data.select([self.rvar] + self.evar)).plot()
        if "corr" in plots:
            from pyrsm.basics.correlation import correlation

            return correlation(self.data.select([self.rvar] + self.evar)).plot(
                nobs=nobs, figsize=figsize
            )
        if "pip" in plots or "vimp" in plots:
            imp, p = self._pip_plot(rep)
            return (p, imp) if ret else p
        if "pred" in plots or "pdp" in plots:
            mode = "pdp" if "pdp" in plots else "pred"
            return self._effect_plots(
                mode, incl, excl, incl_int, nnv, minq, maxq, fix, hline, n_sample,
                ice_nobs=ice_nobs if ice and mode == "pdp" else None,
            )
        return self._coef_plot(1 - alpha, intercept, incl, excl)

    def _coef_plot(self, conf, intercept, incl, excl):
        from plotnine import (
            aes, element_blank, geom_errorbarh, geom_point, geom_vline, ggplot, labs,
            scale_x_log10, scale_y_continuous, theme,
        )

        from pyrsm.plot_style import REF, theme_pyrsm

        cis = self.conf_int(conf)
        lo, hi = cis.columns[3], cis.columns[4]
        df = cis.with_columns(
            pl.col("coefficient").exp().alias("RRR"),
            pl.col(lo).exp().alias("low"),
            pl.col(hi).exp().alias("high"),
        )
        if not intercept:
            df = df.filter(pl.col("index") != "Intercept")
        if incl:
            keep = _as_list(incl)
            df = df.filter(pl.col("index").map_elements(
                lambda s: any(s == v or s.startswith(f"{v}[") for v in keep), return_dtype=pl.Boolean))
        if excl:
            drop = _as_list(excl)
            df = df.filter(~pl.col("index").map_elements(
                lambda s: any(s == v or s.startswith(f"{v}[") for v in drop), return_dtype=pl.Boolean))
        pdf = df.to_pandas()
        # one row per term, the response levels side by side within it
        terms = list(dict.fromkeys(pdf["index"]))
        nlev = len(self.levels) - 1
        width = 0.6 / max(nlev, 1)
        lev_pos = {lv: i for i, lv in enumerate(self.levels[1:])}
        pdf["y"] = [
            len(terms) - terms.index(t) + (nlev - 1) / 2 * width - lev_pos[lv] * width
            for t, lv in zip(pdf["index"], pdf["level"])
        ]
        pdf["level"] = pd.Categorical(pdf["level"], categories=self.levels[1:])
        yl = "RRR (standardized)" if self.standardize else (
            "RRR (centered)" if self.center else "RRR")
        return (
            ggplot(pdf, aes(x="RRR", y="y", color="level"))
            + geom_vline(xintercept=1, linetype="dashdot", color=REF)
            + geom_errorbarh(aes(xmin="low", xmax="high"), height=width * 0.6)
            + geom_point()
            + scale_x_log10()
            + scale_y_continuous(
                breaks=[len(terms) - i for i in range(len(terms))], labels=terms,
                limits=(0.5, len(terms) + 0.5),
            )
            + labs(x=f"{yl} ({100 * conf:g}% CI), relative to {self.lev}", y="",
                   color=self.rvar)
            + theme_pyrsm()
            + theme(panel_grid_minor_y=element_blank())
        )

    # -- prediction, partial dependence and permutation importance plots -------
    def _grid(self, v: str, nnv: int, minq: float, maxq: float) -> list:
        """Values of ``v`` to plot: its levels, its few values, or a range."""
        if v in self.xlevels:
            return list(self.xlevels[v])
        x = self.data[v].cast(pl.Float64)
        uniq = sorted(x.unique().to_list())
        if len(uniq) <= 5:
            return uniq
        lo, hi = float(x.quantile(minq)), float(x.quantile(maxq))
        return list(np.unique(np.linspace(lo, hi, nnv))) if hi > lo else [lo]

    def _slices(self, v: str) -> list:
        """Values of the second variable in an interaction plot."""
        if v in self.xlevels:
            return list(self.xlevels[v])
        x = self.data[v].cast(pl.Float64)
        uniq = sorted(x.unique().to_list())
        if len(uniq) <= 5:
            return uniq
        return [float(x.quantile(q)) for q in (0.1, 0.5, 0.9)]

    def _effect_frame(self, mode: str, vary: dict, n_sample: int) -> pl.DataFrame:
        """Probabilities of every level over the combinations in ``vary``.

        ``pred``: the other variables at their mean or most common level.
        ``pdp``: the average over a sample of rows with the varied columns set.
        """
        names = list(vary)
        if mode == "pred":
            pred = self.predict(cmd=vary)
            return pred.select(names + self.levels)
        base = self.data.select(self.evar)
        if 0 < n_sample < base.height:
            base = base.sample(n=n_sample, seed=1234)
        combos = [{}]
        for k in names:
            combos = [{**c, k: v} for c in combos for v in vary[k]]
        frames = []
        for i, combo in enumerate(combos):
            frames.append(
                base.with_columns(
                    [pl.lit(v).cast(self.data[k].dtype if k in self.xlevels else pl.Float64).alias(k)
                     for k, v in combo.items()]
                ).with_columns(pl.lit(i).alias("__combo"))
            )
        stacked = pl.concat(frames)
        P = self._probs(stacked.select(self.evar))
        probs = pl.DataFrame({lv: P[:, j] for j, lv in enumerate(self.levels)}).with_columns(
            stacked["__combo"]
        )
        means = probs.group_by("__combo").mean().sort("__combo")
        keys = pl.DataFrame([{k: c[k] for k in names} for c in combos])
        if any(k in self.xlevels for k in names):
            keys = keys.with_columns(
                [pl.col(k).cast(pl.Utf8) for k in names if k in self.xlevels]
            )
        return pl.concat([keys, means.drop("__combo")], how="horizontal")

    def _ice_frame(self, vary: dict, n_sample: int, ice_nobs: int) -> pl.DataFrame:
        """ICE: each sampled row's probability of every level over the combinations in ``vary``.

        The rows are a sample (``ice_nobs``; -1: all) of the partial dependence
        sample, so every line is one of the rows the average is taken over.
        """
        base = self.data.select(self.evar)
        if 0 < n_sample < base.height:
            base = base.sample(n=n_sample, seed=1234)
        if ice_nobs is not None and 0 <= ice_nobs < base.height:
            base = base.sample(n=ice_nobs, seed=5678)
        names = list(vary)
        combos = [{}]
        for k in names:
            combos = [{**c, k: v} for c in combos for v in vary[k]]
        obs = pl.int_range(base.height, eager=True).alias("__obs")
        stacked = pl.concat(
            [
                base.with_columns(
                    [pl.lit(v).cast(self.data[k].dtype if k in self.xlevels else pl.Float64).alias(k)
                     for k, v in combo.items()]
                    + [obs]
                )
                for combo in combos
            ]
        )
        P = self._probs(stacked.select(self.evar))
        keys = [stacked[k].cast(pl.Utf8) if k in self.xlevels else stacked[k] for k in names]
        return pl.DataFrame({lv: P[:, j] for j, lv in enumerate(self.levels)}).with_columns(
            *keys, stacked["__obs"]
        )

    def _shares(self) -> pd.DataFrame:
        w = np.ones(len(self._y)) if self._w is None else self._w
        wk = np.bincount(self._y, weights=w, minlength=len(self.levels))
        return pd.DataFrame(
            {
                "level": pd.Categorical(self.levels, categories=self.levels),
                "share": wk / wk.sum(),
            }
        )

    def _long(self, frame: pl.DataFrame, x: str, panel: str | None) -> pd.DataFrame:
        long = frame.unpivot(
            on=self.levels,
            index=[c for c in frame.columns if c not in self.levels],
            variable_name="level",
            value_name="prediction",
        ).to_pandas()
        long["level"] = pd.Categorical(long["level"], categories=self.levels)
        if x in self.xlevels:
            long[x] = pd.Categorical(long[x].astype(str), categories=self.xlevels[x])
        if panel is not None:
            if panel in self.xlevels:
                long["panel"] = pd.Categorical(
                    long[panel].astype(str), categories=self.xlevels[panel]
                )
            else:
                vals = sorted(long[panel].unique())
                labels = [f"{v:.4g}" for v in vals]
                long["panel"] = pd.Categorical(
                    [f"{v:.4g}" for v in long[panel]], categories=labels
                )
        return long

    def _effect_plots(self, mode, incl, excl, incl_int, nnv, minq, maxq, fix, hline, n_sample,
                      ice_nobs=None):
        from mizani.breaks import breaks_extended
        from plotnine import (
            aes, facet_wrap, geom_hline, geom_line, geom_point, ggplot, labs,
            scale_x_continuous, theme,
        )

        from pyrsm.plot_style import theme_pyrsm

        variables = list(self.evar) if incl is None else _as_list(incl)
        variables = [v for v in variables if v in self.evar and v not in _as_list(excl)]
        pairs = [t for t in _as_list(incl_int) if len(t.split(":")) == 2]
        if not variables and not pairs:
            return None

        frames = []
        ice_frames = {}
        for v in variables:
            grid = self._grid(v, nnv, minq, maxq)
            frames.append((v, v, None, self._effect_frame(mode, {v: grid}, n_sample)))
            if ice_nobs is not None:
                ice_frames[v] = self._ice_frame({v: grid}, n_sample, ice_nobs)
        for term in pairs:
            a, b = [t.strip() for t in term.split(":")]
            vary = {a: self._grid(a, nnv, minq, maxq), b: self._slices(b)}
            title = f"{a} by {b}"
            frames.append((title, a, b, self._effect_frame(mode, vary, n_sample)))
            if ice_nobs is not None:
                ice_frames[title] = self._ice_frame(vary, n_sample, ice_nobs)

        ylab = "Partial dependence" if mode == "pdp" else "Predicted probability"

        # One plot with a panel per variable (an interaction: a panel per slice
        # of its second variable), so the levels share a single legend and
        # every panel keeps its full height. Facets cannot mix numeric and
        # categorical x-axes, so a categorical variable's levels sit at
        # reserved positions far from any data and are labelled back.
        cat_names: dict[int, str] = {}
        parts, panels, ice_parts = [], [], []
        for j, (title, x, panel, frame) in enumerate(frames):
            long = self._long(frame, x, panel)
            if x in self.xlevels:
                offset = _CAT_OFFSET * (j + 1)
                cats = list(long[x].cat.categories)
                pos = {c: offset + i for i, c in enumerate(cats)}
                cat_names.update({offset + i: c for i, c in enumerate(cats)})
                long["x"] = [pos[str(v)] for v in long[x].astype(str)]
                long["categorical"] = True
            else:
                pos = None
                long["x"] = long[x].astype(float)
                long["categorical"] = False
            if title in ice_frames:
                ice = self._long(ice_frames[title], x, panel)
                ice["x"] = [pos[str(v)] for v in ice[x].astype(str)] if pos else ice[x].astype(float)
                ice["facet"] = (
                    title if panel is None
                    else [f"{x} | {panel} = {v}" for v in ice["panel"].astype(str)]
                )
                ice["ice_group"] = ice["__obs"].astype(str) + ":" + ice["level"].astype(str)
                ice_parts.append(ice[["facet", "x", "level", "prediction", "ice_group"]])
            if panel is None:
                long["facet"] = title
                panels.append(title)
            else:
                long["facet"] = [f"{x} | {panel} = {v}" for v in long["panel"].astype(str)]
                panels += [f"{x} | {panel} = {v}" for v in long["panel"].cat.categories]
            parts.append(long[["facet", "x", "level", "prediction", "categorical"]])
        data = pd.concat(parts, ignore_index=True)
        data["facet"] = pd.Categorical(data["facet"], categories=list(dict.fromkeys(panels)))
        data["level"] = pd.Categorical(data["level"].astype(str), categories=self.levels)

        def breaks(limits):
            lo, hi = limits
            if lo >= _CAT_OFFSET - 1:
                return [v for v in sorted(cat_names) if lo <= v <= hi]
            # the limits include the axis padding (expand): label only values
            # inside the data range, so no label sits on a panel's edge
            pad = 0.08 / 1.16 * (hi - lo)
            return [b for b in breaks_extended(n=6)(limits) if lo + pad <= b <= hi - pad]

        def labels(values):
            return [
                cat_names.get(int(round(v)), "") if v >= _CAT_OFFSET - 1 else f"{v:g}"
                for v in values
            ]

        npanel = len(data["facet"].cat.categories)
        ncol = 1 if npanel == 1 else 2
        nrow = -(-npanel // ncol)
        p = ggplot(data, aes(x="x", y="prediction", color="level", group="level"))
        if ice_parts:
            # Behind the averages, so each level's partial dependence stays readable.
            ice_data = pd.concat(ice_parts, ignore_index=True)
            ice_data["facet"] = pd.Categorical(ice_data["facet"], categories=data["facet"].cat.categories)
            ice_data["level"] = pd.Categorical(ice_data["level"].astype(str), categories=self.levels)
            p = p + geom_line(
                data=ice_data, mapping=aes(x="x", y="prediction", color="level", group="ice_group"),
                alpha=0.15, size=0.3,
            )
        p = (
            p
            + geom_line(size=1)
            + geom_point(data=data[data["categorical"]], size=2.5)
            + facet_wrap("~facet", ncol=ncol, scales="free_x" if fix else "free")
            + scale_x_continuous(breaks=breaks, labels=labels, expand=(0.08, 0))
        )
        if hline:
            p = p + geom_hline(
                self._shares(), aes(yintercept="share", color="level"), linetype="dashed",
                alpha=0.8, size=0.6,
            )
        return (
            p
            + labs(x="", y=ylab, color=self.rvar)
            + theme_pyrsm()
            + theme(
                legend_position="bottom",
                figure_size=(8 if ncol == 2 else 5, 3 * nrow + 0.6),
            )
        )

    def _pip_plot(self, rep: int = 10):
        """Permutation importance: the drop in McFadden's pseudo R-squared when
        one explanatory variable is shuffled (its interactions move with it)."""
        from plotnine import (
            aes, coord_flip, geom_bar, ggplot, ggtitle, labs, scale_x_discrete, theme,
        )

        from pyrsm.plot_style import FILL, theme_pyrsm

        w = np.ones(len(self._y)) if self._w is None else self._w
        llnull = self.fit_stats["loglik_null"]
        rows = np.arange(len(self._y))

        def pseudo_rsq(frame: pl.DataFrame) -> float:
            P = self._probs(frame)
            ll = float(np.sum(w * np.log(np.clip(P[rows, self._y], 1e-300, None))))
            return 1 - ll / llnull

        base = self.data.select(self.evar)
        baseline = pseudo_rsq(base)
        importance = {}
        for v in self.evar:
            total = 0.0
            for i in range(rep):
                total += baseline - pseudo_rsq(
                    base.with_columns(pl.col(v).shuffle(seed=i).alias(v))
                )
            importance[v] = total / rep
        imp = pl.DataFrame(
            {"variable": list(importance), "importance": list(importance.values())}
        ).sort("importance")
        p = (
            ggplot(imp, aes(x="variable", y="importance"))
            + geom_bar(stat="identity", fill=FILL)
            + coord_flip()
            + scale_x_discrete(limits=imp["variable"].to_list())
            + labs(x="", y="Importance (pseudo R-squared decrease)")
            + ggtitle("Permutation Importance")
            + theme_pyrsm()
            + theme(figure_size=(6, max(5, imp.height * 0.4)))
        )
        return imp.sort("importance", descending=True), p
