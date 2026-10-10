"""Conjoint analysis — port of ``radiant.multivariate::conjoint``.

Fits an OLS model with treatment-coded attributes (first level as base) and
derives part-worth utilities and importance weights, matching R's ``lm`` +
``radiant`` output. Supports a reversed response, interactions, by-group models,
prediction with intervals, and storing part-worths / importance weights.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl

from pyrsm.utils import cat_levels

from ._plotting import as_plot_result
from ._utils import get_data, get_vars
from pyrsm.data_scope import apply_scope, print_scope

__all__ = ["conjoint", "store_predictions"]


def store_predictions(dataset, prediction_result, name: str = "prediction"):
    """Append conjoint predictions to ``dataset`` as new column(s).

    ``prediction_result`` is the Polars DataFrame returned by
    :meth:`conjoint.predict`. As in Radiant's ``store``, ``name`` holds up to
    three comma-separated names: the first is used for the prediction, the
    second and third (when given, and when the predictions have intervals) for
    the lower and upper bound, e.g. ``name="pref, lb, ub"``. Row counts must
    match.
    """
    _, target = get_data(dataset)
    pred = prediction_result
    if target.height != pred.height:
        raise ValueError(
            "Cannot store predictions: row counts do not match. Predict on the "
            "same dataset you are storing into."
        )
    names = [n.strip() for n in str(name).split(",") if n.strip()]
    interval_cols = [c for c in pred.columns if c.endswith("%")]
    sources = ["Prediction"] + interval_cols[:2]
    cols = {nm: pred[src].to_numpy() for nm, src in zip(names, sources)}
    return target.with_columns([pl.Series(k, v) for k, v in cols.items()])


def _to_list(v):
    if v is None or v == "":
        return []
    if isinstance(v, str):
        return [v]
    return list(v)


def _parse_pred_cmd(cmd: str) -> dict:
    """Parse a Radiant-style ``pred_cmd`` string into an expand-grid dict.

    Assignments are separated by ``;``, newlines or ``,`` (a comma followed by
    ``name =``); values may be wrapped in R's ``c(...)`` with optional quotes::

        "Memory = c('4GB', '8GB'), Radio = 'Yes'; Price = $50, $100"

    becomes ``{"Memory": ["4GB", "8GB"], "Radio": ["Yes"], "Price": ["$50", "$100"]}``.

    Note: this is a deliberately small subset of Radiant's R-expression
    ``pred_cmd`` syntax (no numeric ranges like ``1:3`` or function calls) — pass
    a Python dict for full control.
    """
    out: dict[str, list[str]] = {}
    for part in re.split(r"[;\n]|,(?=\s*[A-Za-z_.][\w.]*\s*=)", cmd):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, rhs = part.split("=", 1)
        name = name.strip()
        rhs = rhs.strip()
        m = re.match(r"^c\((.*)\)$", rhs)
        if m:
            rhs = m.group(1)
        vals = [v.strip().strip("'\"") for v in rhs.split(",") if v.strip()]
        out[name] = vals
    return out


def _ci_label(p: float) -> str:
    """Radiant's interval label: 2.5% / 97.5% / 5% / 95%."""
    return f"{100 * p:g}%"


def _label(term: str) -> str:
    """Convert a statsmodels term to a radiant coefficient label.

    ``Intercept`` -> ``(Intercept)``; ``Memory[T.6GB]`` -> ``Memory|6GB``;
    interactions ``A[T.x]:B[T.y]`` -> ``A|x:B|y``.
    """
    if term == "Intercept":
        return "(Intercept)"
    parts = term.split(":")
    out = []
    for p in parts:
        m = re.match(r"^(.*?)\[T\.(.*)\]$", p)
        if m:
            out.append(f"{m.group(1)}|{m.group(2)}")
        else:
            out.append(p)
    return ":".join(out)


class conjoint:
    """Conjoint analysis.

    Parameters
    ----------
    data : dataset (or ``{name: df}``).
    rvar : str
        Response variable (ratings / rankings).
    evar : str | list[str]
        Explanatory attributes; supports ``"a:b"`` ranges.
    int : str | list[str], default ""
        Interaction terms, e.g. ``"Memory:Shape"``.
    by : str, default "none"
        Optional grouping variable; a separate model is fit per level.
    reverse : bool, default False
        Reverse the response as ``(max + 1) - x`` (useful for rankings).

    Attributes
    ----------
    PW : pl.DataFrame
        Part-worth table for the active model (Attributes, Levels, PW).
    IW : pl.DataFrame
        Importance-weight table (Attributes, IW).
    coeff : pl.DataFrame
        Regression coefficients with radiant labels.
    model_list : dict
        Per-group results (``"full"`` when ``by == "none"``).
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
        data,
        rvar: str,
        evar,
        int="",
        by: str = "none",
        reverse: bool = False,
        data_filter: str = "",
        sort: str = "",
        slice: str = "",
    ) -> None:
        self.name, self.data = get_data(data)

        # The data scope, applied before anything is measured: filter, then
        # sort, then slice. Kept on the object so ``summary`` can say which
        # rows the numbers describe.
        self.data_filter = data_filter
        self.sort = sort
        self.slice = slice
        self.data = apply_scope(self.data, data_filter, sort, slice)
        self.rvar = rvar
        self.evar = get_vars(self.data.columns, evar)
        # as in Radiant, a `by` variable is never also an attribute (it may be
        # inside an "a:b" range)
        if by != "none":
            self.evar = [v for v in self.evar if v != by]
        self.int = _to_list(int)
        self.by = by
        self.reverse = reverse

        sel = [rvar] + self.evar
        if by != "none":
            sel = sel + [by]
        sub = self.data.select(sel).drop_nulls()
        self.nobs = sub.height

        # Level order of each categorical attribute (Enum order, else order of
        # appearance); the first level is the treatment-coding base. Numeric
        # attributes stay numeric (one linear term), as in R's lm().
        self._levels = {}
        for v in self.evar:
            s = sub[v]
            if isinstance(s.dtype, (pl.Enum, pl.Categorical)) or s.dtype in (
                pl.Utf8, pl.String, pl.Boolean
            ):
                self._levels[v] = [str(x) for x in cat_levels(s.cast(pl.Utf8) if s.dtype == pl.Boolean else s)]
        self.numeric = [v for v in self.evar if v not in self._levels]
        # Radiant's part-worth table needs factors only
        self.pw_msg = "Only factors can be used." if self.numeric else ""

        import pandas as pd

        pdf = sub.to_pandas()
        # ensure attributes are ordered pandas categoricals with the base level
        # first, so patsy/statsmodels uses Enum/observed order (not alphabetical)
        for v in self._levels:
            pdf[v] = pd.Categorical(
                pdf[v].astype(str), categories=self._levels[v], ordered=True
            )

        self.model_list = {}
        if by == "none":
            self.bylevs = ["full"]
            self._fit(pdf, "full")
        else:
            self.bylevs = [str(x) for x in cat_levels(sub[by])]
            for lev in self.bylevs:
                self._fit(pdf[pdf[by].astype(str) == str(lev)].copy(), lev)

        active = self.model_list[self.bylevs[0]]
        self.PW = active["PW"]
        self.IW = active["IW"]
        self.coeff = active["coeff"]
        self.PW_wide, self.IW_wide = self._wide_tables() if by != "none" else (None, None)

    def _wide_tables(self):
        """Radiant's per-group tables for ``by``: one row per group with the
        R-squared and the coefficients (PW) or importance weights (IW)."""

        def clean(x):
            # R: gsub("[^A-z0-9_\\.]", "", ...); note A-z also keeps [ \ ] ^ `
            return re.sub(r"[^A-z0-9_.]", "", x.replace("|", "_"))

        first = self.model_list[self.bylevs[0]]
        labels = first["coeff"]["label"].to_list()
        pw = {self.by: self.bylevs, ".Rsq": [self.model_list[g]["rsq"] for g in self.bylevs]}
        for lab in labels:
            vals = []
            for g in self.bylevs:
                c = self.model_list[g]["coeff"]
                hit = c.filter(pl.col("label") == lab)["coefficient"]
                vals.append(float(hit[0]) if hit.len() else None)
            pw[clean(lab)] = vals
        PW_wide = pl.DataFrame(pw)
        if first["IW"] is None:
            return PW_wide, None
        iw = {self.by: self.bylevs, ".Rsq": pw[".Rsq"]}
        for attr in first["IW"]["Attributes"].to_list():
            iw[clean(attr)] = [
                float(self.model_list[g]["IW"].filter(pl.col("Attributes") == attr)["IW"][0])
                for g in self.bylevs
            ]
        return PW_wide, pl.DataFrame(iw)

    def _fit(self, pdf, group):
        import statsmodels.formula.api as smf

        rhs = list(self.evar) + list(self.int)
        if self.reverse:
            pdf = pdf.copy()
            pdf[self.rvar] = (pdf[self.rvar].max() + 1) - pdf[self.rvar]

        formula = f"{self.rvar} ~ " + (" + ".join(rhs) if rhs else "1")
        # As R's lm(): drop levels that do not occur (e.g. within a `by`
        # group) and drop aliased terms instead of splitting their effect with
        # a pseudo-inverse. Aliased terms get a coefficient (and part-worth) of
        # 0 and are left out of the coefficient table, as in Radiant.
        import pandas as pd

        from ._lm import fit_like_r

        pdf = pdf.copy()
        for c in self._levels:
            pdf[c] = pdf[c].cat.remove_unused_categories()
        model = fit_like_r(smf.ols(formula, data=pdf))

        est = ~model.params.index.isin(model.aliased)

        def term_pos(name):
            # R keeps the formula order; patsy moves numeric terms to the end
            if name == "Intercept":
                return -1
            base = ":".join(p.split("[")[0] for p in name.split(":"))
            return rhs.index(base) if base in rhs else len(rhs)

        order = sorted(range(len(model.params)), key=lambda i: term_pos(model.params.index[i]))
        est = np.array([est[i] for i in order])
        model_params = model.params.iloc[order]
        coeff = pl.DataFrame(
            {
                "label": [_label(t) for t in model_params.index[est]],
                "coefficient": model_params.values[est],
                "std.error": model.bse.iloc[order].values[est],
                "t.value": model.tvalues.iloc[order].values[est],
                "p.value": model.pvalues.iloc[order].values[est],
            }
        )
        if self.numeric:
            PW = IW = plot_ylim = None
        else:
            PW, IW, plot_ylim = self._part_worths(model)
        self.model_list[group] = {
            "model": model,
            "coeff": coeff,
            "PW": PW,
            "IW": IW,
            "plot_ylim": plot_ylim,
            "rsq": float(model.rsquared),
            "rsq_adj": float(model.rsquared_adj),
        }

    def _part_worths(self, model):
        params = dict(zip(model.params.index, model.params.values))
        attrs, levels, pws = [], [], []
        ranges, mins, maxs = {}, {}, {}
        for v in self.evar:
            levs = self._levels[v]
            pw_vals = []
            for k, lev in enumerate(levs):
                pw = 0.0 if k == 0 else params.get(f"{v}[T.{lev}]", 0.0)
                attrs.append(v)
                levels.append(str(lev))
                pws.append(pw)
                pw_vals.append(pw)
            ranges[v] = max(pw_vals) - min(pw_vals)
            mins[v] = min(pw_vals)
            maxs[v] = max(pw_vals)
        attrs.append("Base utility")
        levels.append("~")
        pws.append(params.get("Intercept", 0.0))

        PW = pl.DataFrame({"Attributes": attrs, "Levels": levels, "PW": [round(x, 3) for x in pws]})
        total = sum(ranges.values())
        IW = pl.DataFrame(
            {
                "Attributes": list(ranges.keys()),
                "IW": [round(ranges[v] / total, 3) if total else 0.0 for v in ranges],
            }
        )

        # shared y-limits for part-worth plots (Radiant's plot_ylim logic)
        maxrange = max(ranges.values()) if ranges else 0.0
        ylim = {}
        for v in self.evar:
            lo, hi = mins[v], maxs[v]
            if hi > abs(lo):
                hi = hi + maxrange - ranges[v]
            else:
                lo = lo - (maxrange - ranges[v])
            ylim[v] = (lo * 1.01, hi * 1.01)
        return PW, IW, ylim

    # -- output ---------------------------------------------------------------

    def _active(self, show=None):
        if show is None or show not in self.model_list:
            show = self.bylevs[0]
        return show, self.model_list[show]

    def summary(
        self, show=None, mc_diag: bool = False, additional: bool = False, dec: int = 3
    ) -> None:
        """
        Summarize the conjoint analysis.

        Parameters
        ----------
        show : str, optional
            Group to show when ``by`` was used (default: the first group).
        mc_diag : bool, default False
            Show multicollinearity diagnostics (generalized VIF per attribute).
        additional : bool, default False
            Show standard errors, t-values, p-values and model fit statistics.
        dec : int, default 3
            Number of decimals.
        """
        from pyrsm.basics import display_utils as du
        from pyrsm.model.model import model_fit, sig_stars

        from ._lm import collinearity_note

        show, m = self._active(show)
        print("Conjoint analysis")
        print(f"Data                 : {self.name}")
        print_scope(self, 21)
        if self.by != "none":
            print(f"Show                 : {self.by} == {show}")
        rv = f"{self.rvar} (reversed)" if self.reverse else self.rvar
        print(f"Response variable    : {rv}")
        print(f"Explanatory variables: {', '.join(self.evar)}\n")

        def fixed(df, cols):
            # fixed decimals (0.000, 8.500), as Radiant prints them
            return df.with_columns([pl.col(c).map_elements(
                lambda x: f"{x:.{dec}f}", return_dtype=pl.Utf8) for c in cols])

        if m["PW"] is None:
            print("Conjoint part-worths:")
            print(self.pw_msg)
            print("\nConjoint importance weights:")
            print(self.pw_msg)
        else:
            print("Conjoint part-worths:")
            du.print_plain_tables(fixed(m["PW"], ["PW"]))
            print("\nConjoint importance weights:")
            du.print_plain_tables(fixed(m["IW"], ["IW"]))
        print("\nConjoint regression results:\n")
        coeff = m["coeff"]
        if not additional:
            du.print_plain_tables(fixed(coeff.select(["label", "coefficient"]), ["coefficient"]))
        else:
            tab = fixed(coeff, ["coefficient", "std.error", "t.value"]).with_columns(
                # fixed decimals as Radiant prints them (0.100, 1.000)
                pl.col("p.value").map_elements(
                    lambda p: "< .001" if p < 0.001 else f"{p:.{dec}f}",
                    return_dtype=pl.Utf8,
                ),
                sig_stars(coeff["p.value"].to_list()).alias("  "),
            )
            du.print_plain_tables(tab)
            du.print_sig_codes()
            model = m["model"]
            model.nobs_dropped = 0
            print(f"\n{model_fit(model, dec=dec)}")
        if m["model"].aliased:
            print(f"\n{collinearity_note(m['model'])}")

        if mc_diag:
            if len(self.evar) > 1:
                print("\nMulticollinearity diagnostics:")
                du.print_plain_tables(self.vif(show=show, dec=dec))
            else:
                print(
                    "\nInsufficient number of attributes selected to calculate\n"
                    "multicollinearity diagnostics"
                )

    def vif(self, show=None, dec: int = 3) -> pl.DataFrame:
        """Generalized variance inflation factor per attribute (``car::vif``),
        sorted from high to low."""
        from pyrsm.model.model import gvif

        _, m = self._active(show)
        v = gvif(m["model"], dec=dec)
        out = pl.DataFrame(
            {"term": [str(t) for t in v.index], "VIF": v["vif"].astype(float).round(dec),
             "Rsq": v["Rsq"].astype(float).round(dec)}
        )
        return out.sort("VIF", descending=True, maintain_order=True)

    def _predict_one(self, model, pred_df, se, interval, conf_lev):
        if se:
            obs = interval == "prediction"
            pr = model.get_prediction(pred_df).summary_frame(alpha=1 - conf_lev)
            if obs:
                lo, hi = pr["obs_ci_lower"].values, pr["obs_ci_upper"].values
            else:
                lo, hi = pr["mean_ci_lower"].values, pr["mean_ci_upper"].values
            mean = pr["mean"].values
            return pl.DataFrame(
                {
                    "Prediction": mean,
                    _ci_label((1 - conf_lev) / 2): lo,
                    _ci_label((1 + conf_lev) / 2): hi,
                    "+/-": (hi - mean),
                }
            )
        return pl.DataFrame({"Prediction": np.asarray(model.predict(pred_df))})

    def _plug_values(self) -> dict:
        """Radiant's values for attributes not set in ``pred_cmd``: the most
        frequent level (the first one on ties) or the mean of a numeric."""
        sub = self.data.select(self.evar).drop_nulls()
        out = {}
        for v in self.evar:
            if v in self._levels:
                counts = sub[v].cast(pl.Utf8).value_counts()
                cnt = dict(zip(counts[v].to_list(), counts["count"].to_list()))
                best = max(cnt.values())
                out[v] = next(lv for lv in self._levels[v] if cnt.get(lv, 0) == best)
            else:
                out[v] = float(sub[v].mean())
        return out

    def predict(
        self,
        data=None,
        pred_cmd=None,
        se: bool = False,
        interval: str = "confidence",
        conf_lev: float = 0.95,
        dec: int | None = None,
        ci: bool | None = None,
        conf: float | None = None,
    ):
        """Predict the response for attribute profiles.

        Parameters
        ----------
        data : DataFrame | dict | None
            Profiles to score. If ``None`` (and no ``pred_cmd``) the estimation
            data is used.
        pred_cmd : dict | str | None
            ``{attribute: [levels]}`` expanded into all combinations (the first
            attribute varies fastest, as R's ``expand.grid``). Attributes that
            are not named get their most frequent level (the mean for a numeric
            attribute). A Radiant-style string such as
            ``"Memory = c('4GB','8GB'), Radio = 'Yes'"`` is accepted too. With
            ``data`` as well, ``pred_cmd`` sets those attributes in every row.
        se : bool
            Add intervals.
        interval : str
            ``"confidence"`` or ``"prediction"`` (only used when ``se`` is True).
        conf_lev : float
            Confidence level.
        dec : int, optional
            Round the predictions (default: full precision, as in Radiant).
        ci, conf : deprecated aliases for ``se``/``conf_lev``.

        Returns
        -------
        pl.DataFrame
            The attributes, ``Prediction`` and (with ``se``) the interval
            columns. For by-group models there is a block per group, with the
            group in the first column.
        """
        if ci is not None:
            se = ci
        if conf is not None:
            conf_lev = conf
        if not se:
            interval = "none"

        cmd = None
        if pred_cmd is not None:
            cmd = _parse_pred_cmd(pred_cmd) if isinstance(pred_cmd, str) else dict(pred_cmd)
            cmd = {k: list(v) if isinstance(v, (list, tuple)) else [v] for k, v in cmd.items()}
            unknown = [k for k in cmd if k not in self.evar]
            if unknown:
                raise ValueError(f"Not an attribute in the model: {', '.join(unknown)}")

        if cmd is not None and data is not None:
            # Radiant: pred_cmd changes the given data (one value per attribute)
            _, d = get_data(data)
            multi = [k for k, v in cmd.items() if len(v) != 1]
            if multi:
                raise ValueError(
                    "With data, pred_cmd can set a single value per attribute: "
                    + ", ".join(multi)
                )
            d = d.with_columns([pl.lit(v[0]).alias(k) for k, v in cmd.items()])
            pred_pl = self._coerce_levels(d).select(self.evar)
        elif cmd is not None:
            keys = list(cmd)
            # expand.grid: the first variable varies fastest
            import itertools

            rows = [r[::-1] for r in itertools.product(*[cmd[k] for k in reversed(keys)])]
            grid = {k: [r[i] for r in rows] for i, k in enumerate(keys)}
            plug = self._plug_values()
            others = [v for v in self.evar if v not in cmd]
            pred_pl = pl.DataFrame(
                {**{v: [plug[v]] * len(rows) for v in others}, **grid}
            )
            pred_pl = self._coerce_levels(pred_pl)
        elif data is None:
            pred_pl = None
        else:
            _, d = get_data(data)
            pred_pl = self._coerce_levels(d).select(self.evar)

        def one(m):
            if pred_pl is None:
                frame_pl = pl.from_pandas(m["model"].model.data.frame[self.evar])
                frame = m["model"].model.data.frame
            else:
                frame_pl, frame = pred_pl, pred_pl.to_pandas()
            out = self._predict_one(m["model"], frame, se, interval, conf_lev)
            out = pl.concat([frame_pl, out], how="horizontal")
            if dec is not None:
                out = out.with_columns(
                    [pl.col(c).round(dec) for c in out.columns if out[c].dtype == pl.Float64]
                )
            return out

        if self.by == "none":
            return one(self.model_list["full"])

        parts = []
        for lev in self.bylevs:
            out = one(self.model_list[lev]).with_columns(pl.lit(str(lev)).alias(self.by))
            parts.append(out.select([self.by, *[c for c in out.columns if c != self.by]]))
        return pl.concat(parts, how="vertical_relaxed")

    def _coerce_levels(self, df: pl.DataFrame) -> pl.DataFrame:
        """Cast attribute columns to the model's Enum level order."""
        exprs = []
        for v, levels in self._levels.items():
            if v in df.columns:
                exprs.append(pl.col(v).cast(pl.Utf8).cast(pl.Enum(levels)))
        return df.with_columns(exprs) if exprs else df

    def store(self, what: str = "PW", show=None) -> pl.DataFrame:
        """Return the part-worth (``"PW"``) or importance-weight (``"IW"``) table.

        With ``by``, as in Radiant, this is the table with one row per group
        (the group, R-squared and coefficients or importance weights) unless
        ``show`` picks a single group.
        """
        if self.by != "none" and show is None:
            out = self.PW_wide if what.upper() == "PW" else self.IW_wide
        else:
            _, m = self._active(show)
            out = m["PW"] if what.upper() == "PW" else m["IW"]
        if out is None:
            raise ValueError(self.pw_msg)
        return out

    def plot(
        self,
        plots="pw",
        show=None,
        scale_plot: bool = False,
        custom: bool = False,
    ):
        """Part-worth ("pw") and/or importance-weight ("iw") plots (plotnine).

        ``plots`` accepts a single string or a list (e.g. ``["pw", "iw"]``). For
        ``"pw"`` one line/point plot is produced per attribute (matching
        Radiant); ``"iw"`` adds an importance-weight bar plot. When
        ``scale_plot=True`` the part-worth panels share a common y-range
        (Radiant's ``plot_ylim`` logic). Returns a single plot when exactly one
        is produced, otherwise a list (part-worths first, then importance).
        """
        import pandas as pd
        from plotnine import (
            aes,
            coord_cartesian,
            element_text,
            geom_col,
            geom_line,
            geom_point,
            ggplot,
            labs,
            theme,
        )

        from pyrsm.plot_style import ALPHA, LINE, POINT_FILL, theme_pyrsm

        if isinstance(plots, str):
            plots = [plots]
        show, m = self._active(show)
        if m["PW"] is None:
            raise ValueError(self.pw_msg)
        # with by, Radiant adds the group to the titles
        grp = f" ({show})" if self.by != "none" else ""

        out = []
        if "pw" in plots:
            ylim = m["plot_ylim"]
            for v in self.evar:
                dat = (
                    m["PW"]
                    .filter(pl.col("Attributes") == v)
                    .with_columns(pl.col("Levels").cast(pl.Utf8))
                    .to_pandas()
                )
                # keep the attribute's level order on the x axis (plotnine
                # would sort a text column alphabetically)
                dat["Levels"] = pd.Categorical(
                    dat["Levels"], categories=list(dict.fromkeys(dat["Levels"])), ordered=True
                )
                g = (
                    ggplot(dat, aes(x="Levels", y="PW", group=1))
                    + geom_line(color=LINE, linetype="dashdot", size=0.7)
                    + geom_point(color=LINE, size=4, fill=POINT_FILL)
                    + labs(title=f"Part-worths for {v}{grp}", x="", y="PW")
                    + theme_pyrsm()
                    + theme(axis_text_x=element_text(angle=45, hjust=1))
                )
                if scale_plot:
                    g = g + coord_cartesian(ylim=ylim[v])
                out.append(g)

        if "iw" in plots:
            dat = m["IW"].to_pandas()
            out.append(
                ggplot(dat, aes(x="Attributes", y="IW", fill="Attributes"))
                + geom_col(alpha=ALPHA)
                + labs(title=f"Importance weights{grp}", x="", y="IW")
                + theme_pyrsm()
                + theme(legend_position="none")
            )

        if not out:
            raise ValueError("plots must include 'pw' and/or 'iw'")
        return as_plot_result(out)
