"""Choice-based conjoint analysis with the interface of pyrsm's modeling tools.

``conjoint_choice`` fits a conditional (McFadden multinomial) logit to choice
data in long format -- one row per alternative per choice task, with an
indicator for the alternative that was chosen. It has the same structure as
``pyrsm.model.logistic`` and the other modeling classes: ``rvar`` / ``lev`` /
``evar`` / ``ivar`` arguments, the data scope (``data_filter``, ``sort``,
``slice``), a ``summary`` with a coefficient table and fit statistics,
``predict(data, cmd, data_cmd, dec)`` and ``plot``. On top of that it reports
conjoint part-worths and importance weights (as ``pyrsm.multivariate.conjoint``
does for ratings-based conjoint), willingness to pay, and market-share
simulation through ``predict``.

Estimates match R's ``survival::clogit`` (see tests/test_conjoint_choice.py).
The estimator (``pyrsm._logit``) is shared with ``pyrsm.model.mnl`` and
``pyrsm.multivariate.cbc``, the earlier port of
``pyrsm_streamlit``'s ``ConjointChoice``.

Examples
--------
>>> import polars as pl
>>> from pyrsm.multivariate import conjoint_choice
>>> choc = pl.read_parquet("examples/data/multivariate/choc.parquet")
>>> cc = conjoint_choice({"choc": choc}, rvar="chosen", lev=1,
...                      evar=["brand", "chocolate", "price", "nuts"], id="choice_id")
>>> cc.summary()
>>> cc.predict(cmd={"brand": ["Hershey", "Lindt", "Godiva"]})  # market shares
>>> cc.plot("pw")
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from pyrsm.basics import display_utils as du
from pyrsm.data_scope import apply_scope, print_scope
from pyrsm.model.model import sig_stars, sim_prediction
from pyrsm.utils import cat_levels, check_dataframe

from ._plotting import as_plot_result
from pyrsm._logit import clogit_fit as _mnl_fit
from pyrsm._logit import segment_probs as _segment_probs

__all__ = ["conjoint_choice"]


def _as_list(v) -> list:
    if v is None or (isinstance(v, str) and v == ""):
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
    """Levels in Enum order (text: order of appearance), keeping only levels
    that occur."""
    if s.dtype == pl.Boolean:
        present = set(s.cast(pl.Utf8).drop_nulls().unique().to_list())
        return [c for c in ("false", "true") if c in present]
    return cat_levels(s, drop_unused=True)


class conjoint_choice:
    """
    Choice-based conjoint (conditional logit) analysis.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame]
        Choice data in long format: one row per alternative per choice task.
        Pass ``{"name": df}`` to show the name in the summary.
    rvar : str
        Column that indicates which alternative was chosen in each task.
    lev : str | int, optional
        Value of ``rvar`` that means "chosen". Defaults to 1 for a numeric
        ``rvar``; required for a categorical one (e.g. "yes").
    evar : list[str]
        Product attributes. Categorical attributes get a part-worth per level
        (first level = 0); numeric attributes (e.g. price) a single slope.
    ivar : list[str], optional
        Interactions between attributes, e.g. ``["brand:price"]``.
    id : str
        Choice-task identifier. Every task must have exactly one chosen
        alternative.
    alt : str, optional
        Alternative identifier within a task (used to label predictions).
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

    Attributes
    ----------
    coef : pl.DataFrame
        Coefficients with odds ratios, standard errors, z- and p-values.
    PW : pl.DataFrame
        Part-worths (Attributes, Levels, PW); the first level of each
        categorical attribute is the base (0).
    IW : pl.DataFrame
        Importance weights: each attribute's part-worth range over the total.
    fit_stats : dict
        Log-likelihoods, pseudo R-squared, AIC, BIC, likelihood-ratio test and
        hit rate.
    """

    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        rvar: str | None = None,
        lev: str | int | None = None,
        evar: list[str] | str | None = None,
        ivar: list[str] | str | None = None,
        id: str | None = None,
        alt: str | None = None,
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
        self.evar = _as_list(evar)
        self.ivar = _as_list(ivar)
        self.id = id
        self.alt = alt
        if rvar is None or not self.evar or id is None:
            raise ValueError("Please specify rvar, evar and id")
        if rvar in self.evar:
            raise ValueError(
                "Response variable contained in the set of explanatory variables. "
                "Please update model specification."
            )
        for term in self.ivar:
            missing = [p for p in term.split(":") if p not in self.evar]
            if missing:
                raise ValueError(
                    f"Interaction '{term}' uses variables not in evar: {', '.join(missing)}"
                )

        # the alternative id may also be an attribute (brand constants)
        cols = list(dict.fromkeys([id] + ([alt] if alt else []) + [rvar] + self.evar))
        self.nobs_all = data.height
        data = data.select(cols).drop_nulls()
        self.nobs = data.height
        self.nobs_dropped = self.nobs_all - self.nobs

        # which rows were chosen
        rv = data[rvar]
        if _is_categorical(rv):
            if lev is None:
                raise ValueError(
                    f"Please specify the level of {rvar} that means 'chosen' "
                    f"(lev). Levels: {', '.join(_levels(rv))}"
                )
            if str(lev) not in _levels(rv):
                raise ValueError(f"Specified level is not a level in {rvar}")
            chosen = (rv.cast(pl.Utf8) == str(lev)).to_numpy()
        else:
            lev = 1 if lev is None else lev
            chosen = (rv.cast(pl.Float64) == float(lev)).to_numpy()
        self.lev = lev

        _, codes = np.unique(data[id].to_numpy(), return_inverse=True)
        codes = codes.astype(np.intp)
        self.ntasks = int(codes.max()) + 1 if codes.size else 0
        nchosen = np.bincount(codes, weights=chosen.astype(float), minlength=self.ntasks)
        bad = np.nonzero(nchosen != 1)[0]
        if bad.size:
            raise ValueError(
                f"Each choice task ({id}) must have exactly one chosen alternative; "
                f"{bad.size} of {self.ntasks} tasks do not (e.g., "
                f"{id} = {data[id].unique(maintain_order=True).to_list()[bad[0]]})"
            )

        self.levels = {v: _levels(data[v]) for v in self.evar if _is_categorical(data[v])}
        self.data = data
        X, self.xnames = self._design(data)
        self._fit = _mnl_fit(X, chosen.astype(float), codes, self.ntasks)
        self.converged = self._fit["converged"]
        self._chosen = chosen
        self._codes = codes

        from scipy.stats import chi2, norm

        params = self._fit["params"]
        bse = self._fit["bse"]
        z = params / bse
        # norm.sf keeps tiny p-values (1 - cdf rounds them to 0)
        pvals = 2 * norm.sf(np.abs(z))
        self.coef = pl.DataFrame(
            {
                "index": self.xnames,
                "OR": np.exp(params),
                "OR%": 100 * (np.exp(params) - 1),
                "coefficient": params,
                "std.error": bse,
                "z.value": z,
                "p.value": pvals,
                "  ": sig_stars(pvals),
            }
        )

        k = len(params)
        llf, llnull = self._fit["llf"], self._fit["llnull"]
        prob, _, _ = _segment_probs(X @ params, codes, self.ntasks)
        best = np.full(self.ntasks, -1.0)
        np.maximum.at(best, codes, prob)
        hits = float(np.sum(chosen & (prob == best[codes]))) / self.ntasks
        lr = 2 * (llf - llnull)
        self.fit_stats = {
            "loglik": llf,
            "loglik_null": llnull,
            "pseudo_rsq": 1 - llf / llnull,
            "pseudo_rsq_adj": 1 - (llf - k) / llnull,
            "AIC": -2 * llf + 2 * k,
            "BIC": -2 * llf + k * math.log(self.ntasks),
            "chisq": lr,
            "df": k,
            "chisq_pvalue": float(chi2.sf(lr, k)),
            "hit_rate": hits,
            "ntasks": self.ntasks,
            "nobs": self.nobs,
        }
        self.PW, self.IW = self._part_worths()

    # -- design ----------------------------------------------------------------
    def _columns(self, df: pl.DataFrame, v: str):
        """Treatment-coded columns for one attribute."""
        if v in self.levels:
            s = df[v].cast(pl.Utf8).to_numpy()
            return (
                [f"{v}[{lev}]" for lev in self.levels[v][1:]],
                [(s == lev).astype(float) for lev in self.levels[v][1:]],
            )
        return [v], [df[v].cast(pl.Float64).to_numpy()]

    def _design(self, df: pl.DataFrame):
        names, cols = [], []
        coded = {}
        for v in self.evar:
            n, c = self._columns(df, v)
            coded[v] = list(zip(n, c))
            names += n
            cols += c
        for term in self.ivar:
            combos = [("", np.ones(df.height))]
            for part in term.split(":"):
                combos = [
                    (f"{a}:{b}" if a else b, x * y) for a, x in combos for b, y in coded[part]
                ]
            for n, c in combos:
                names.append(n)
                cols.append(c)
        return np.column_stack(cols), names

    def _part_worths(self):
        params = dict(zip(self.xnames, self._fit["params"]))
        attrs, levels, pws = [], [], []
        ranges = {}
        for v in self.evar:
            if v in self.levels:
                vals = [0.0] + [params[f"{v}[{lev}]"] for lev in self.levels[v][1:]]
                attrs += [v] * len(vals)
                levels += self.levels[v]
                pws += vals
                ranges[v] = max(vals) - min(vals)
            else:
                # a numeric attribute has one slope; its range over the data
                # is what it can move utility by
                x = self.data[v].cast(pl.Float64)
                attrs.append(v)
                levels.append("(slope)")
                pws.append(params[v])
                ranges[v] = abs(params[v]) * float(x.max() - x.min())
        total = sum(ranges.values())
        PW = pl.DataFrame({"Attributes": attrs, "Levels": levels, "PW": pws})
        IW = pl.DataFrame(
            {
                "Attributes": list(ranges),
                "IW": [r / total if total else 0.0 for r in ranges.values()],
            }
        )
        return PW, IW

    # -- output ----------------------------------------------------------------
    def summary(self, dec: int = 3, plain: bool = True) -> None:
        """
        Summarize the conjoint choice model.

        Parameters
        ----------
        dec : int, default 3
            Number of decimal places to round to.
        plain : bool, default True
            Plain text tables; set to False for styled tables in a notebook.
        """
        print("Conjoint choice model (conditional logit)")
        print(f"Data                 : {self.name}")
        print_scope(self, 21)
        print(f"Response variable    : {self.rvar}")
        print(f"Level                : {self.lev}")
        print(f"Choice task id       : {self.id}")
        if self.alt:
            print(f"Alternative id       : {self.alt}")
        print(f"Explanatory variables: {', '.join(self.evar)}")
        if self.ivar:
            print(f"Interactions         : {', '.join(self.ivar)}")
        print(f"Null hyp.: There is no effect of x on the choice of {self.rvar}")
        print(f"Alt. hyp.: There is an effect of x on the choice of {self.rvar}\n")

        df = self.coef.with_columns(
            du.format_pval_expr("p.value", dec),
            (pl.col("OR%").round(max(dec - 2, 0)).cast(pl.Utf8) + "%").alias("OR%"),
        )
        if not plain and du.is_notebook() and du.has_great_tables():
            from IPython.display import display

            display(
                du.style_table(
                    df,
                    title="Coefficient Estimates",
                    subtitle=f"Response: {self.rvar} (level: {self.lev})",
                    number_cols=["OR", "coefficient", "std.error", "z.value"],
                    dec=dec,
                )
            )
        else:
            du.print_plain_tables(
                df.with_columns(
                    [pl.col(c).round(dec) for c in ("OR", "coefficient", "std.error", "z.value")]
                )
            )
        du.print_sig_codes()

        fs = self.fit_stats
        pv = du.format_pval(fs["chisq_pvalue"], dec)
        print(f"\nPseudo R-squared (McFadden): {fs['pseudo_rsq']:.{dec}f}")
        print(f"Pseudo R-squared (McFadden adjusted): {fs['pseudo_rsq_adj']:.{dec}f}")
        print(
            f"Log-likelihood: {fs['loglik']:.{dec}f}, AIC: {fs['AIC']:.{dec}f}, "
            f"BIC: {fs['BIC']:.{dec}f}"
        )
        print(f"Chi-squared: {fs['chisq']:.{dec}f}, df({fs['df']}), p.value {pv}")
        print(f"Hit rate: {100 * fs['hit_rate']:.1f}% of tasks predicted correctly")
        dropped = f" ({self.nobs_dropped:,} obs. dropped)" if self.nobs_dropped else ""
        print(f"Nr tasks: {self.ntasks:,}, Nr obs: {self.nobs:,}{dropped}")
        if not self.converged:
            print("** Warning: the optimizer did not fully converge **")

        print("\nConjoint part-worths:")
        du.print_plain_tables(self.PW.with_columns(pl.col("PW").round(dec)))
        print("\nConjoint importance weights:")
        du.print_plain_tables(self.IW.with_columns(pl.col("IW").round(dec)))

    def wtp(self, price: str = "price", dec: int | None = None) -> pl.DataFrame:
        """
        Willingness to pay: each level's part-worth divided by -(price slope).

        Parameters
        ----------
        price : str, default "price"
            A numeric attribute in ``evar`` measured in money.
        dec : int, optional
            Round the result.
        """
        if price not in self.evar or price in self.levels:
            raise ValueError(f"'{price}' must be a numeric attribute in evar")
        bp = float(self.coef.filter(pl.col("index") == price)["coefficient"][0])
        out = self.PW.filter(pl.col("Levels") != "(slope)").with_columns(
            (pl.col("PW") / -bp).alias("WTP")
        )
        return out.with_columns(pl.col("PW", "WTP").round(dec)) if dec is not None else out

    def store(self, what: str = "PW", price: str = "price") -> pl.DataFrame:
        """
        Results as a data frame to store as a new dataset.

        Parameters
        ----------
        what : str, default "PW"
            "PW" part-worths, "IW" importance weights, "WTP" willingness to
            pay (needs the numeric ``price`` attribute) or "coef" the
            coefficient table.
        price : str, default "price"
            Price attribute used for "WTP".
        """
        what = what.upper() if what.lower() != "coef" else "coef"
        if what == "PW":
            return self.PW
        if what == "IW":
            return self.IW
        if what == "WTP":
            return self.wtp(price)
        if what == "coef":
            return self.coef
        raise ValueError("what must be 'PW', 'IW', 'WTP' or 'coef'")

    def predict(self, data=None, cmd=None, data_cmd=None, dec=None) -> pl.DataFrame:
        """
        Predict choice probabilities (shares) within each choice task.

        Parameters
        ----------
        data : pd.DataFrame | pl.DataFrame, optional
            Product profiles to predict for. Rows with the same ``id`` form a
            choice set; without an ``id`` column all rows form one choice set
            (a market simulation). Defaults to the estimation data.
        cmd : dict, optional
            Attribute values to combine into profiles, e.g.
            ``{"brand": ["Hershey", "Lindt"], "price": [2.99, 3.99]}``; other
            attributes are held at their mean or most common value. All the
            profiles form one choice set.
        data_cmd : dict, optional
            Values to set for every row of ``data``, e.g. ``{"nuts": "Yes"}``.
        dec : int, optional
            Round float columns to this many decimals.

        Returns
        -------
        pl.DataFrame
            The profiles with their ``utility`` and predicted choice
            probability (``prediction``).
        """
        if cmd is not None and cmd != "":
            cmd = {k: [v] if isinstance(v, (str, int, float)) else list(v) for k, v in cmd.items()}
            for k, vals in cmd.items():
                unknown = [x for x in vals if k in self.levels and str(x) not in self.levels[k]]
                if unknown:
                    raise ValueError(f"Unknown level(s) for {k}: {', '.join(map(str, unknown))}")
            pred = sim_prediction(self.data.select(self.evar), vary=cmd)
        elif data is None:
            keep = [self.id] + ([self.alt] if self.alt else []) + self.evar
            pred = self.data.select(keep)
        else:
            pred = check_dataframe(data)
            keep = [c for c in [self.id, self.alt] if c and c in pred.columns] + self.evar
            pred = pred.select(keep)
        if data_cmd is not None and data_cmd != "":
            pred = pred.with_columns(
                [pl.lit(v).cast(pred[k].dtype).alias(k) for k, v in data_cmd.items()]
            )
        for v, levs in self.levels.items():
            unknown = set(pred[v].cast(pl.Utf8).unique().to_list()) - set(levs)
            if unknown:
                raise ValueError(f"Unknown level(s) for {v}: {', '.join(map(str, unknown))}")

        X, _ = self._design(pred)
        util = X @ self._fit["params"]
        if self.id in pred.columns:
            _, codes = np.unique(pred[self.id].to_numpy(), return_inverse=True)
            codes = codes.astype(np.intp)
        else:
            codes = np.zeros(pred.height, dtype=np.intp)
        prob, _, _ = _segment_probs(util, codes, int(codes.max()) + 1)
        pred = pred.with_columns(pl.Series("utility", util), pl.Series("prediction", prob))
        if dec is not None:
            pred = pred.with_columns(
                [pl.col(c).round(dec) for c in pred.columns if pred[c].dtype in (pl.Float64, pl.Float32)]
            )
        return pred

    def plot(
        self,
        plots: Literal["pw", "iw", "or"] = "pw",
        conf: float = 0.95,
        scale_plot: bool = False,
    ):
        """
        Plots for a conjoint choice model

        Parameters
        ----------
        plots : str | list[str], default "pw"
            "pw" plots the part-worths of each categorical attribute, "iw" the
            importance weights, and "or" the odds ratios with confidence
            intervals.
        conf : float, default 0.95
            Confidence level for the "or" plot.
        scale_plot : bool, default False
            Give every part-worth plot the same y-axis range (as
            ``pyrsm.multivariate.conjoint``).
        """
        from plotnine import (
            aes, coord_cartesian, coord_flip, element_text, geom_col, geom_errorbar,
            geom_hline, geom_line, geom_point, ggplot, labs, scale_x_discrete,
            scale_y_log10, theme,
        )
        from scipy.stats import norm

        from pyrsm.plot_style import ALPHA, LINE, MUTED, POINT_FILL, theme_pyrsm

        plots = _as_list(plots)
        unknown = [p for p in plots if p not in ("pw", "iw", "or")]
        if unknown:
            raise ValueError(f"Unknown plot type(s): {', '.join(unknown)}. Use 'pw', 'iw' or 'or'")
        out = []
        if "pw" in plots:
            pw = self.PW.filter(pl.col("Levels") != "(slope)")["PW"]
            ylim = (min(pw.min(), 0.0), max(pw.max(), 0.0)) if pw.len() else None
            for v, levs in self.levels.items():
                dat = self.PW.filter(pl.col("Attributes") == v).to_pandas()
                g = (
                    ggplot(dat, aes(x="Levels", y="PW", group=1))
                    + geom_line(color=LINE, linetype="dashdot", size=0.7)
                    + geom_point(color=LINE, size=4, fill=POINT_FILL)
                    + scale_x_discrete(limits=levs)
                    + labs(title=f"Part-worths for {v}", x="", y="PW")
                    + theme_pyrsm()
                    + theme(axis_text_x=element_text(angle=45, hjust=1))
                )
                if scale_plot and ylim:
                    g = g + coord_cartesian(ylim=ylim)
                out.append(g)
        if "iw" in plots:
            dat = self.IW.to_pandas()
            out.append(
                ggplot(dat, aes(x="Attributes", y="IW", fill="Attributes"))
                + geom_col(alpha=ALPHA)
                + scale_x_discrete(limits=self.IW["Attributes"].to_list())
                + labs(title="Importance weights", x="", y="IW")
                + theme_pyrsm()
                + theme(legend_position="none")
            )
        if "or" in plots:
            zq = norm.ppf(1 - (1 - conf) / 2)
            dat = self.coef.select(
                "index",
                "OR",
                (pl.col("coefficient") - zq * pl.col("std.error")).exp().alias("low"),
                (pl.col("coefficient") + zq * pl.col("std.error")).exp().alias("high"),
            ).to_pandas()
            out.append(
                ggplot(dat, aes(x="index", y="OR", ymin="low", ymax="high"))
                + geom_hline(yintercept=1, linetype="dashed", color=MUTED)
                + geom_errorbar(width=0.2, color=LINE)
                + geom_point(color=LINE, size=3, fill=POINT_FILL)
                + scale_x_discrete(limits=self.coef["index"].to_list()[::-1])
                + scale_y_log10()
                + coord_flip()
                + labs(title=f"Odds ratios ({100 * conf:.0f}% CI)", x="", y="Odds ratio")
                + theme_pyrsm()
            )
        return as_plot_result(out)
