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

from ._plotting import as_plot_result
from ._utils import get_data, get_vars

__all__ = ["conjoint", "store_predictions"]


def store_predictions(dataset, prediction_result, name: str = "prediction"):
    """Append conjoint predictions to ``dataset`` as new column(s).

    ``prediction_result`` is the Polars DataFrame returned by
    :meth:`conjoint.predict`. When it has a single prediction column it is added
    as ``name``; interval columns are added as ``name_lb`` / ``name_ub`` and the
    margin as ``name_me``. Row counts must match.
    """
    _, target = get_data(dataset)
    pred = prediction_result
    if target.height != pred.height:
        raise ValueError(
            "Cannot store predictions: row counts do not match. Predict on the "
            "same dataset you are storing into."
        )
    cols = {name: pred["Prediction"].to_numpy()}
    interval_cols = [c for c in pred.columns if c.endswith("%")]
    if len(interval_cols) == 2:
        cols[f"{name}_lb"] = pred[interval_cols[0]].to_numpy()
        cols[f"{name}_ub"] = pred[interval_cols[1]].to_numpy()
    if "+/-" in pred.columns:
        cols[f"{name}_me"] = pred["+/-"].to_numpy()
    return target.with_columns([pl.Series(k, v) for k, v in cols.items()])


def _to_list(v):
    if v is None or v == "":
        return []
    if isinstance(v, str):
        return [v]
    return list(v)


def _parse_pred_cmd(cmd: str) -> dict:
    """Parse a Radiant-style ``pred_cmd`` string into an expand-grid dict.

    Supported assignment separator is ``;`` (or newlines); values are
    comma-separated and may be wrapped in R's ``c(...)`` with optional quotes::

        "Memory = c('4GB', '8GB'); Radio = Yes; Price = $50, $100"

    becomes ``{"Memory": ["4GB", "8GB"], "Radio": ["Yes"], "Price": ["$50", "$100"]}``.

    Note: this is a deliberately small subset of Radiant's R-expression
    ``pred_cmd`` syntax (no numeric ranges like ``1:3`` or function calls) — pass
    a Python dict for full control.
    """
    import re

    out: dict[str, list[str]] = {}
    for part in re.split(r"[;\n]", cmd):
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
    """

    def __init__(
        self,
        data,
        rvar: str,
        evar,
        int="",
        by: str = "none",
        reverse: bool = False,
    ) -> None:
        self.name, self.data = get_data(data)
        self.rvar = rvar
        self.evar = get_vars(self.data.columns, evar)
        self.int = _to_list(int)
        self.by = by
        self.reverse = reverse

        sel = [rvar] + self.evar
        if by != "none":
            sel = sel + [by]
        sub = self.data.select(sel).drop_nulls()
        self.nobs = sub.height

        # Capture each attribute's level order (Enum order, else order of
        # appearance) so the first level becomes the treatment-coding base.
        self._levels = {}
        for v in self.evar:
            if isinstance(sub[v].dtype, pl.Enum):
                self._levels[v] = sub[v].dtype.categories.to_list()
            else:
                self._levels[v] = sub[v].cast(pl.Utf8).unique(maintain_order=True).to_list()

        import pandas as pd

        pdf = sub.to_pandas()
        # ensure attributes are ordered pandas categoricals with the base level
        # first, so patsy/statsmodels uses Enum/observed order (not alphabetical)
        for v in self.evar:
            pdf[v] = pd.Categorical(
                pdf[v].astype(str), categories=[str(x) for x in self._levels[v]],
                ordered=True,
            )

        self.model_list = {}
        if by == "none":
            self.bylevs = ["full"]
            self._fit(pdf, "full")
        else:
            if isinstance(sub[by].dtype, pl.Enum):
                self.bylevs = sub[by].dtype.categories.to_list()
            else:
                self.bylevs = sub[by].cast(pl.Utf8).unique(maintain_order=True).to_list()
            for lev in self.bylevs:
                self._fit(pdf[pdf[by].astype(str) == str(lev)].copy(), lev)

        active = self.model_list[self.bylevs[0]]
        self.PW = active["PW"]
        self.IW = active["IW"]
        self.coeff = active["coeff"]

    def _fit(self, pdf, group):
        import statsmodels.formula.api as smf

        rhs = list(self.evar) + list(self.int)
        if self.reverse:
            pdf = pdf.copy()
            pdf[self.rvar] = (pdf[self.rvar].max() + 1) - pdf[self.rvar]

        formula = f"{self.rvar} ~ " + (" + ".join(rhs) if rhs else "1")
        model = smf.ols(formula, data=pdf).fit()

        coeff = pl.DataFrame(
            {
                "label": [_label(t) for t in model.params.index],
                "coefficient": model.params.values,
                "std.error": model.bse.values,
                "t.value": model.tvalues.values,
                "p.value": model.pvalues.values,
            }
        )
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

    def summary(self, show=None, mc_diag: bool = False, dec: int = 3) -> None:
        show, m = self._active(show)
        print("Conjoint analysis")
        print(f"Data                 : {self.name}")
        if self.by != "none":
            print(f"Show                 : {self.by} == {show}")
        rv = f"{self.rvar} (reversed)" if self.reverse else self.rvar
        print(f"Response variable    : {rv}")
        print(f"Explanatory variables: {', '.join(self.evar)}\n")

        print("Conjoint part-worths:")
        print(
            m["PW"].with_columns(pl.col("PW").round(dec))
        )
        print("\nConjoint importance weights:")
        print(m["IW"].with_columns(pl.col("IW").round(dec)))
        print("\nConjoint regression results:\n")
        print(m["coeff"].select(["label", "coefficient"]).with_columns(
            pl.col("coefficient").round(dec)
        ))

        if mc_diag and len(self.evar) > 1:
            print("\nMulticollinearity diagnostics:")
            print(self.vif(show=show, dec=dec))

    def vif(self, show=None, dec: int = 3) -> pl.DataFrame:
        """Generalized VIF-style Rsq per term (via auxiliary regressions)."""
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        _, m = self._active(show)
        exog = m["model"].model.exog
        names = m["model"].model.exog_names
        rows = []
        for i, nm in enumerate(names):
            if nm == "Intercept":
                continue
            v = variance_inflation_factor(exog, i)
            rows.append({"term": _label(nm), "VIF": round(v, dec),
                         "Rsq": round(1 - 1 / v, dec) if v else 0.0})
        return pl.DataFrame(rows)

    def _predict_one(self, model, pred_df, se, interval, conf_lev, dec):
        if se:
            obs = interval == "prediction"
            pr = model.get_prediction(pred_df).summary_frame(alpha=1 - conf_lev)
            if obs:
                lo, hi = pr["obs_ci_lower"].values, pr["obs_ci_upper"].values
            else:
                lo, hi = pr["mean_ci_lower"].values, pr["mean_ci_upper"].values
            mean = pr["mean"].values
            out = pl.DataFrame(
                {
                    "Prediction": mean,
                    f"{(1 - conf_lev) / 2:.1%}": lo,
                    f"{(1 + conf_lev) / 2:.1%}": hi,
                    "+/-": (hi - mean),
                }
            )
        else:
            out = pl.DataFrame({"Prediction": np.asarray(model.predict(pred_df))})
        return out.with_columns(pl.all().round(dec))

    def predict(
        self,
        data=None,
        pred_cmd=None,
        se: bool = False,
        interval: str = "confidence",
        conf_lev: float = 0.95,
        dec: int = 3,
        ci: bool | None = None,
        conf: float | None = None,
    ):
        """Predict response for new attribute profiles.

        Parameters
        ----------
        data : DataFrame | dict | None
            New profiles to score. If ``None`` and ``pred_cmd`` is ``None`` the
            estimation data is used.
        pred_cmd : dict | str | None
            Mapping of ``{attribute: [levels]}`` expanded into a full grid of
            profiles (an "expand-grid" convenience). A Radiant-style string such
            as ``"Memory = c('4GB','8GB'); Radio = Yes"`` is also accepted (a
            documented subset — see :func:`_parse_pred_cmd`).
        se : bool
            Return standard-error-based intervals.
        interval : str
            ``"confidence"`` or ``"prediction"`` (only used when ``se`` is True).
        conf_lev : float
            Confidence level.
        ci, conf : deprecated aliases for ``se``/``conf_lev``.

        For by-group models a prediction is returned for every group, with a
        leading column holding the group level.
        """
        if ci is not None:
            se = ci
        if conf is not None:
            conf_lev = conf
        if not se:
            interval = "none"

        if pred_cmd is not None:
            import itertools

            cmd = _parse_pred_cmd(pred_cmd) if isinstance(pred_cmd, str) else pred_cmd
            keys = list(cmd.keys())
            grids = [
                cmd[k] if isinstance(cmd[k], (list, tuple)) else [cmd[k]] for k in keys
            ]
            rows = list(itertools.product(*grids))
            pred_pl = pl.DataFrame({k: [r[i] for r in rows] for i, k in enumerate(keys)})
            pred_df = self._coerce_levels(pred_pl).to_pandas()
        elif data is None:
            pred_df = None
        else:
            _, d = get_data(data)
            pred_df = self._coerce_levels(d).to_pandas()

        if self.by == "none":
            m = self.model_list["full"]
            frame = m["model"].model.data.frame if pred_df is None else pred_df
            return self._predict_one(m["model"], frame, se, interval, conf_lev, dec)

        # by-group: one prediction block per group
        parts = []
        for lev in self.bylevs:
            m = self.model_list[lev]
            frame = m["model"].model.data.frame if pred_df is None else pred_df
            out = self._predict_one(m["model"], frame, se, interval, conf_lev, dec)
            out = out.with_columns(pl.lit(str(lev)).alias(self.by))
            parts.append(out.select([self.by, *[c for c in out.columns if c != self.by]]))
        return pl.concat(parts, how="vertical")

    def _coerce_levels(self, df: pl.DataFrame) -> pl.DataFrame:
        """Cast attribute columns to the model's Enum level order."""
        exprs = []
        for v in self.evar:
            if v in df.columns:
                exprs.append(
                    pl.col(v).cast(pl.Utf8).cast(pl.Enum([str(x) for x in self._levels[v]]))
                )
        return df.with_columns(exprs) if exprs else df

    def store(self, what: str = "PW", show=None) -> pl.DataFrame:
        """Return the part-worth (``"PW"``) or importance-weight (``"IW"``) table."""
        _, m = self._active(show)
        return m["PW"] if what.upper() == "PW" else m["IW"]

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
        from plotnine import (
            aes,
            coord_cartesian,
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
        _, m = self._active(show)

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
                # preserve level order on the x axis
                dat["Levels"] = pl.Series(dat["Levels"]).to_pandas()
                g = (
                    ggplot(dat, aes(x="Levels", y="PW", group=1))
                    + geom_line(color="blue", linetype="dashdot", size=0.7)
                    + geom_point(color="blue", size=4, fill="white")
                    + labs(title=f"Part-worths for {v}", x="", y="PW")
                    + theme_classic()
                    + theme(axis_text_x=element_text(angle=45, hjust=1))
                )
                if scale_plot:
                    g = g + coord_cartesian(ylim=ylim[v])
                out.append(g)

        if "iw" in plots:
            dat = m["IW"].to_pandas()
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
