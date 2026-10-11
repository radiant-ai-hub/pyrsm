"""Evaluate stored predictions — Python port of radiant.model's Model > Evaluate.

Each column in ``pred`` is one model's stored predictions (from a Predict tab,
or any numeric score such as an RFM index), evaluated against the response
``rvar`` in the same dataset:

* ``evalreg``   — regression: R-squared, RMSE and MAE per predictor
                  (``radiant.model::evalreg``).
* ``evalbin``   — binary classification: gains, lift, profit and ROME by
                  quantile of the score (``radiant.model::evalbin``).
* ``confusion`` — the confusion matrix at the break-even threshold
                  ``cost / margin`` with TPR, precision, kappa, AUC, ...
                  (``radiant.model::confusion``).
* ``uplift``    — incremental uplift from a treatment/control split
                  (``radiant.model::uplift``).

Training and test rows. ``train="All"`` uses every row (the scope is ignored,
as in Radiant). ``"Training"`` uses the rows the scope (``data_filter`` /
``sort`` / ``slice``) selects, ``"Test"`` the remaining rows, and ``"Both"``
reports each. Any value other than ``"All"`` needs a filter or a slice.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pyrsm.data_scope import apply_scope
from pyrsm.plot_style import FILL, LINE, REF, theme_pyrsm
from pyrsm.utils import check_dataframe

__all__ = ["evalreg", "evalbin", "confusion", "uplift", "auc", "rig", "xtile"]

TRAIN_OPTIONS = ("All", "Training", "Test", "Both")
_SUFFIX = {"All": "", "Training": " (train)", "Test": " (test)"}
EVALBIN_PLOTS = ("lift", "gains", "profit", "expected_profit", "rome")
UPLIFT_PLOTS = ("inc_uplift", "uplift", "inc_profit")
CONFUSION_METRICS = (
    "TP", "FP", "TN", "FN", "total", "TPR", "TNR", "precision", "Fscore",
    "RIG", "accuracy", "kappa", "profit", "index", "ROME", "contact", "AUC",
)


# -- shared helpers -----------------------------------------------------------


def _as_list(x) -> list[str]:
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


def _frame(data) -> tuple[str, pl.DataFrame]:
    if isinstance(data, dict):
        name = list(data.keys())[0]
        return name, check_dataframe(data[name])
    return "Not provided", check_dataframe(data)


def _splits(
    df: pl.DataFrame,
    train: str,
    data_filter: str,
    sort: str,
    slice: str,  # noqa: A002
    test_filter: str = "",
) -> dict[str, pl.DataFrame]:
    """``{"All"|"Training"|"Test": frame}`` following Radiant's train/test rules.

    Training rows are the scope (``data_filter`` / ``sort`` / ``slice``). Test rows
    are ``test_filter`` when it is given, else every row that is not a training row.
    """
    if train in ("", None):
        train = "All"
    if train == "Validation":  # Radiant's evalreg accepts it as Test
        train = "Test"
    if train not in TRAIN_OPTIONS:
        raise ValueError(f"train must be one of {', '.join(TRAIN_OPTIONS)}, not {train!r}")
    if train == "All":
        return {"All": df}
    has_training = bool(str(data_filter or "").strip() or str(slice or "").strip())
    has_test = bool(str(test_filter or "").strip())
    if not has_training and not (train == "Test" and has_test):
        raise ValueError(
            "A filter or a slice is required to tell training rows from test "
            "rows. Set data_filter (e.g. \"training == 1\") or slice, or use "
            "train=\"All\"."
        )
    indexed = df.with_row_index("__row__")
    scoped = apply_scope(indexed, data_filter, sort, slice)
    keep = set(scoped["__row__"].to_list())
    training = scoped.drop("__row__")
    if has_test:
        test = apply_scope(df, test_filter, "", "")
    else:
        test = indexed.filter(~pl.col("__row__").is_in(list(keep))).drop("__row__")
    out = {}
    if train in ("Training", "Both"):
        out["Training"] = training
    if train in ("Test", "Both"):
        out["Test"] = test
    for label, frame in out.items():
        if frame.height == 0:
            raise ValueError(
                f"Data for {label} has zero rows. Please correct the filter or "
                "slice and try again."
            )
    return out


def _levels(series: pl.Series) -> list[str]:
    """Response levels in Radiant's order: factor levels, else sorted strings."""
    if isinstance(series.dtype, (pl.Enum, pl.Categorical)):
        try:
            return [str(v) for v in series.cat.get_categories().to_list()]
        except Exception:  # pragma: no cover - older polars
            pass
    vals = series.drop_nulls().cast(pl.Utf8).unique().to_list()
    return sorted(vals)


def _is_level(series: pl.Series, lev: str) -> np.ndarray:
    return (series.cast(pl.Utf8) == str(lev)).fill_null(False).to_numpy()


def _check_lev(series: pl.Series, lev: str | None, what: str) -> str:
    levs = _levels(series)
    if lev in (None, ""):
        if not levs:
            raise ValueError(f"{what} has no non-missing values")
        return levs[0]
    if str(lev) not in levs:
        raise ValueError(f"Level {lev!r} not found in {what} (levels: {', '.join(levs[:20])})")
    return str(lev)


def xtile(x, n: int = 10, rev: bool = False) -> np.ndarray:
    """Quantile bins 1..n — radiant.data::xtile (type-7 quantiles, include lowest)."""
    x = np.asarray(x, dtype=float)
    breaks = np.nanquantile(x, np.linspace(0, 1, n + 1))
    bins = np.searchsorted(breaks, x, side="left")
    bins = np.clip(bins, 1, n)
    return (n + 1 - bins) if rev else bins


def auc(pred, rvar) -> float:
    """Area under the ROC curve (Mann-Whitney U), flipped when below 0.5."""
    from scipy.stats import rankdata

    pred = np.asarray(pred, dtype=float)
    rvar = np.asarray(rvar, dtype=bool)
    n1 = int((~rvar).sum())
    n2 = int(rvar.sum())
    if n1 == 0 or n2 == 0:
        return float("nan")
    u = rankdata(pred)[~rvar].sum() - n1 * (n1 + 1) / 2
    wt = u / n1 / n2
    return float(1 - wt if wt < 0.5 else wt)


def rig(pred, rvar, crv: float = 1e-7) -> float:
    """Relative information gain: 1 - logloss(pred) / logloss(base rate)."""
    pred = np.clip(np.asarray(pred, dtype=float), crv, 1 - crv)
    rvar = np.asarray(rvar, dtype=float)
    mo = rvar.mean()
    llpred = np.mean(-np.log(pred) * rvar - np.log(1 - pred) * (1 - rvar))
    llbase = np.mean(-np.log(mo) * rvar - np.log(1 - mo) * (1 - rvar))
    return float(round(1 - llpred / llbase, 6))


def _kappa(tn, fp, fn, tp) -> float:
    total = tn + fp + fn + tp
    if total == 0:
        return float("nan")
    po = (tn + tp) / total
    pe = ((tn + fp) * (tn + fn) + (fn + tp) * (fp + tp)) / total**2
    return float((po - pe) / (1 - pe)) if pe != 1 else float("nan")


def _header(obj, title: str, extra: list[tuple[str, str]]) -> None:
    print(title)
    print(f"Data        : {obj.name}")
    for label, value in (
        ("Filter", obj.data_filter),
        ("Sort", obj.sort),
        ("Slice", obj.slice),
        ("Test filter", getattr(obj, "test_filter", "")),
    ):
        if str(value or "").strip():
            print(f"{label.ljust(12)}: {value}")
    print(f"Results for : {obj.train}")
    print(f"Predictors  : {', '.join(obj.pred)}")
    print(f"Response    : {obj.rvar}")
    for label, value in extra:
        print(f"{label.ljust(12)}: {value}")
    print()


def _print_table(df: pl.DataFrame, dec: int) -> None:
    rounded = df.with_columns(
        [pl.col(c).round(dec) for c, t in df.schema.items() if t.is_float()]
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_hide_dataframe_shape=True,
                   tbl_hide_column_data_types=True, fmt_str_lengths=60,
                   tbl_width_chars=200):
        print(rounded)


def _single_series(obj) -> bool:
    return len(obj.pred) < 2 and obj.train != "Both"


def _line_plot(df, x, y, ylab, single: bool, ref=None, percent_y=False, comma_y=False,
               points=True, vlines=None, hlines=None):
    """A cumulative curve per predictor, in the pyrsm style."""
    from mizani.formatters import comma_format, percent_format
    from plotnine import (
        aes,
        geom_hline,
        geom_line,
        geom_point,
        geom_segment,
        geom_vline,
        ggplot,
        labs,
        scale_x_continuous,
        scale_y_continuous,
        theme,
    )

    pdf = df.to_pandas()
    pdf["pred"] = pdf["pred"].astype("category").cat.reorder_categories(
        list(dict.fromkeys(df["pred"].to_list()))
    )
    if single:
        g = ggplot(pdf, aes(x=x, y=y, group="pred")) + geom_line(color=LINE)
        if points:
            g += geom_point(color=LINE)
    else:
        g = ggplot(pdf, aes(x=x, y=y, color="pred", group="pred")) + geom_line()
        if points:
            g += geom_point()
    if ref is not None:
        x0, y0, x1, y1 = ref
        g += geom_segment(x=x0, y=y0, xend=x1, yend=y1, linetype="dashdot", color=REF, size=0.3)
    if vlines is not None and len(vlines):
        if single:
            g += geom_vline(xintercept=list(vlines["contact"]), color=LINE, linetype="dashdot", size=0.5)
        else:
            g += geom_vline(aes(xintercept="contact", color="pred"), data=vlines.to_pandas(),
                            linetype="dashdot", size=0.5, show_legend=False)
    if hlines is not None and len(hlines):
        if single:
            g += geom_hline(yintercept=list(hlines["max_profit"]), color=LINE, linetype="dashdot", size=0.5)
        else:
            g += geom_hline(aes(yintercept="max_profit", color="pred"), data=hlines.to_pandas(),
                            linetype="dashdot", size=0.5, show_legend=False)
    g += labs(x="Proportion of population targeted", y=ylab, color="Predictor")
    g += scale_x_continuous(labels=percent_format())
    if percent_y:
        g += scale_y_continuous(labels=percent_format())
    elif comma_y:
        g += scale_y_continuous(labels=comma_format())
    g += theme_pyrsm()
    if single:
        g += theme(legend_position="none")
    return g


def _metric_bars(df: pl.DataFrame, metrics: list[str], title: str, scale_y: bool = True):
    """Faceted bars: one row per metric, a bar per predictor (Radiant's evalreg/confusion plot)."""
    from plotnine import aes, facet_grid, geom_col, ggplot, labs, position_dodge, theme

    long = df.select(["Type", "Predictor", *metrics]).unpivot(
        index=["Type", "Predictor"], on=metrics, variable_name="Metric", value_name="Value"
    ).to_pandas()
    long["Metric"] = long["Metric"].astype("category").cat.reorder_categories(metrics)
    long["Predictor"] = long["Predictor"].astype("category").cat.reorder_categories(
        list(dict.fromkeys(df["Predictor"].to_list()))
    )
    types = list(dict.fromkeys(df["Type"].to_list()))
    if len(types) < 2:
        g = ggplot(long, aes(x="Predictor", y="Value")) + geom_col(fill=FILL)
    else:
        g = ggplot(long, aes(x="Predictor", y="Value", fill="Type")) + geom_col(
            position=position_dodge()
        )
    g += facet_grid("Metric ~ .", scales="free_y" if scale_y else "fixed")
    g += labs(title=f"{title} ({', '.join(types)})", x="Predictor", y="", fill="")
    g += theme_pyrsm()
    if len(types) < 2:
        g += theme(legend_position="none")
    return g


def _plot_result(plot_list: list):
    from pyrsm.multivariate._plotting import as_plot_result

    return as_plot_result(plot_list)


class _Evaluation:
    """Shared constructor bits: data, scope, train/test split."""

    def _setup(self, data, pred, rvar, train, data_filter, sort, slice, extra=(),  # noqa: A002
               test_filter=""):
        self.name, frame = _frame(data)
        self.pred = _as_list(pred)
        self.rvar = rvar
        if not self.pred:
            raise ValueError("Select at least one stored prediction to evaluate")
        missing = [v for v in [*self.pred, rvar, *extra] if v not in frame.columns]
        if missing:
            raise ValueError(f"Column(s) not found in {self.name}: {', '.join(missing)}")
        nonnum = [p for p in self.pred if not frame[p].dtype.is_numeric()]
        if nonnum:
            raise ValueError(f"Predictions must be numeric: {', '.join(nonnum)}")
        self.train = "All" if train in ("", None) else ("Test" if train == "Validation" else train)
        self.data_filter = data_filter or ""
        self.sort = sort or ""
        self.slice = slice or ""
        self.test_filter = test_filter or ""
        self.data = frame
        self.dat_list = _splits(
            frame, self.train, self.data_filter, self.sort, self.slice, self.test_filter
        )


# -- Evaluate regression ------------------------------------------------------


class evalreg(_Evaluation):
    """Evaluate stored predictions for a numeric response (R-squared, RMSE, MAE).

    Parameters
    ----------
    data : pl.DataFrame | pd.DataFrame | dict[str, DataFrame]
        Dataset (or ``{name: df}``) holding the response and the predictions.
    pred : str | list[str]
        Numeric prediction column(s); each is one model.
    rvar : str
        Numeric response variable.
    train : {"All", "Training", "Test", "Both"}, default "All"
        Which rows to evaluate (see the module docstring).
    data_filter : str, default ""
        Filter that selects the training rows, e.g. "training == 1"
        (``pyrsm.data_scope``). Ignored when ``train="All"``.
    sort : str, default ""
        Column(s) to sort on before slicing.
    slice : str, default ""
        Rows to keep as training rows, counting from 1, e.g. "1:1000".
    test_filter : str, default ""
        Filter that selects the test rows, e.g. "training == 0". Blank: every row
        that is not a training row, as in Radiant.

    Examples
    --------
    >>> ev = rsm.model.evaluate.evalreg({"diamonds": diamonds}, pred=["pred_reg", "pred_nn"], rvar="price")
    >>> ev.summary(); ev.plot()
    """

    def __init__(self, data, pred, rvar: str, train: str = "All", data_filter: str = "",
                 sort: str = "", slice: str = "", test_filter: str = "") -> None:  # noqa: A002
        self._setup(data, pred, rvar, train, data_filter, sort, slice, test_filter=test_filter)
        if not self.data[rvar].dtype.is_numeric():
            raise ValueError(f"The response variable {rvar} must be numeric")
        rows = []
        for label, frame in self.dat_list.items():
            rv = frame[rvar].cast(pl.Float64).to_numpy()
            for p in self.pred:
                pv = frame[p].cast(pl.Float64).to_numpy()
                ok = ~(np.isnan(rv) | np.isnan(pv))
                r = np.corrcoef(rv[ok], pv[ok])[0, 1] if ok.sum() > 1 else np.nan
                rows.append({
                    "Type": label,
                    "Predictor": p,
                    "n": int(frame.height),
                    "Rsq": float(r**2),
                    "RMSE": float(np.sqrt(np.mean((rv[ok] - pv[ok]) ** 2))),
                    "MAE": float(np.mean(np.abs(rv[ok] - pv[ok]))),
                })
        self.dat = pl.DataFrame(rows)

    def summary(self, dec: int = 3) -> None:
        """Print Radiant-style regression evaluation."""
        _header(self, "Evaluate predictions for regression models", [])
        _print_table(self.dat, dec)

    def plot(self, vars=("Rsq", "RMSE", "MAE")):
        """Bars per predictor, one facet row per metric."""
        metrics = _as_list(vars)
        unknown = [m for m in metrics if m not in ("Rsq", "RMSE", "MAE")]
        if unknown:
            raise ValueError(f"Unknown metric(s): {', '.join(unknown)}. Use Rsq, RMSE and/or MAE")
        return _metric_bars(self.dat, metrics, "Regression performance plots")


# -- Evaluate classification --------------------------------------------------


class confusion(_Evaluation):
    """Confusion matrix and classification metrics at the break-even threshold.

    A prediction above ``cost / margin`` (0.5 when either is missing; ``cost``
    when margin is 0) is classified as the selected level ``lev`` of ``rvar``.

    Parameters
    ----------
    data : pl.DataFrame | pd.DataFrame | dict[str, DataFrame]
        Dataset (or ``{name: df}``) holding the response and the predictions.
    pred : str | list[str]
        Numeric prediction column(s); each is one model.
    rvar : str
        Response variable.
    lev : str, default ""
        The "success" level of ``rvar`` (first level when empty).
    cost, margin, scale : float, default 1, 2 and 1
        Cost per contact, margin per response and a multiplier for profit.
    train : {"All", "Training", "Test", "Both"}, default "All"
    data_filter : str, default ""
        Filter that selects the training rows, e.g. "training == 1"
        (``pyrsm.data_scope``). Ignored when ``train="All"``.
    sort : str, default ""
        Column(s) to sort on before slicing.
    slice : str, default ""
        Rows to keep as training rows, counting from 1, e.g. "1:1000".
    test_filter : str, default ""
        Filter that selects the test rows, e.g. "training == 0". Blank: every row
        that is not a training row, as in Radiant.
    """

    def __init__(self, data, pred, rvar: str, lev: str = "", cost: float = 1, margin: float = 2,
                 scale: float = 1, train: str = "All", data_filter: str = "", sort: str = "",
                 slice: str = "", test_filter: str = "") -> None:  # noqa: A002
        self._setup(data, pred, rvar, train, data_filter, sort, slice, test_filter=test_filter)
        self.lev = _check_lev(self.data[rvar], lev, rvar)
        self.cost, self.margin, self.scale = cost, margin, scale
        if cost is None or margin is None or (isinstance(cost, float) and np.isnan(cost)) \
                or (isinstance(margin, float) and np.isnan(margin)):
            break_even = 0.5
        elif margin == 0:
            break_even = cost / 1
        else:
            break_even = cost / margin
        self.break_even = float(break_even)

        rows = []
        for label, frame in self.dat_list.items():
            rv = _is_level(frame[rvar], self.lev)
            for p in self.pred:
                pv = frame[p].cast(pl.Float64).to_numpy()
                pos = pv > break_even
                tp = int((pos & rv).sum())
                fp = int((pos & ~rv).sum())
                tn = int((~pos & ~rv).sum())
                fn = int((~pos & rv).sum())
                rows.append({
                    "Type": label, "Predictor": p, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                    "AUC": auc(pv, rv), "RIG": rig(pv, rv),
                    "kappa": _kappa(tn, fp, fn, tp),
                })
        df = pl.DataFrame(rows)
        m, c, s = float(margin or 0), float(cost or 0), float(scale or 1)
        df = df.with_columns(total=pl.col("TN") + pl.col("FN") + pl.col("FP") + pl.col("TP"))
        df = df.with_columns(
            TPR=pl.col("TP") / (pl.col("TP") + pl.col("FN")),
            TNR=pl.col("TN") / (pl.col("TN") + pl.col("FP")),
            precision=pl.col("TP") / (pl.col("TP") + pl.col("FP")),
            accuracy=(pl.col("TP") + pl.col("TN")) / pl.col("total"),
            profit=(m * pl.col("TP") - c * (pl.col("TP") + pl.col("FP"))) * s,
            ROME=(m * pl.col("TP") - c * (pl.col("TP") + pl.col("FP")))
            / (c * (pl.col("TP") + pl.col("FP"))),
            contact=(pl.col("TP") + pl.col("FP")) / pl.col("total"),
        )
        df = df.with_columns(
            Fscore=2 * (pl.col("precision") * pl.col("TPR")) / (pl.col("precision") + pl.col("TPR")),
            index=pl.col("profit") / pl.col("profit").max().over("Type"),
        )
        self.dataset = df.select(["Type", "Predictor", *CONFUSION_METRICS])

    def summary(self, dec: int = 3) -> None:
        """Print the confusion matrix counts and the metrics, as Radiant does."""
        _header(self, "Confusion matrix", [
            ("Level", f"{self.lev} in {self.rvar}"),
            ("Cost:Margin", f"{self.cost} : {self.margin}"),
            ("Scale", f"{self.scale}"),
        ])
        df = self.dataset
        _print_table(df.select(["Type", "Predictor", "TP", "FP", "TN", "FN", "total",
                                "TPR", "TNR", "precision", "Fscore"]), dec)
        print()
        _print_table(df.select(["Type", "Predictor", "RIG", "accuracy", "kappa", "profit",
                                "index", "ROME", "contact", "AUC"]), dec)

    def plot(self, vars=("kappa", "index", "ROME", "AUC"), scale_y: bool = True):
        """Bars per predictor, one facet row per metric (counts shown as shares of total)."""
        metrics = _as_list(vars)
        unknown = [m for m in metrics if m not in CONFUSION_METRICS]
        if unknown:
            raise ValueError(f"Unknown metric(s): {', '.join(unknown)}")
        df = self.dataset.with_columns(
            [(pl.col(c) / pl.col("total")).alias(c) for c in ("TN", "FN", "FP", "TP")]
        )
        return _metric_bars(df, metrics, "Classification performance plots", scale_y=scale_y)


class evalbin(_Evaluation):
    """Evaluate stored predictions for a binary response: gains, lift, profit, ROME.

    Parameters
    ----------
    data : pl.DataFrame | pd.DataFrame | dict[str, DataFrame]
        Dataset (or ``{name: df}``) holding the response and the predictions.
    pred : str | list[str]
        Numeric prediction column(s) (probabilities or scores); each is one model.
    rvar : str
        Response variable.
    lev : str, default ""
        The "success" level of ``rvar`` (first level when empty).
    qnt : int, default 10
        Number of quantile bins.
    cost, margin : float, default 1 and 2
        Cost per contact and margin per response, for profit and ROME.
    scale : float, default 1
        Multiplier for counts and profit (e.g. to scale a sample up to a population).
    train : {"All", "Training", "Test", "Both"}, default "All"
    data_filter : str, default ""
        Filter that selects the training rows, e.g. "training == 1"
        (``pyrsm.data_scope``). Ignored when ``train="All"``.
    sort : str, default ""
        Column(s) to sort on before slicing.
    slice : str, default ""
        Rows to keep as training rows, counting from 1, e.g. "1:1000".
    test_filter : str, default ""
        Filter that selects the test rows, e.g. "training == 0". Blank: every row
        that is not a training row, as in Radiant.

    Attributes
    ----------
    dataset : pl.DataFrame
        One row per predictor and bin with gains, lift, profit and ROME.
    cnf : confusion
        The confusion matrix at the break-even threshold, for the same inputs.

    Examples
    --------
    >>> ev = rsm.model.evaluate.evalbin({"dvd": dvd}, pred="pred_logit", rvar="buy", lev="yes")
    >>> ev.summary(); ev.plot(plots=["lift", "gains"])
    """

    def __init__(self, data, pred, rvar: str, lev: str = "", qnt: int = 10, cost: float = 1,
                 margin: float = 2, scale: float = 1, train: str = "All", data_filter: str = "",
                 sort: str = "", slice: str = "", test_filter: str = "") -> None:  # noqa: A002
        self._setup(data, pred, rvar, train, data_filter, sort, slice, test_filter=test_filter)
        cost = 0 if cost is None else cost
        margin = 0 if margin is None else margin
        scale = 1 if scale in (None, 0) else scale
        qnt = 10 if not qnt else int(qnt)
        if qnt < 2:
            raise ValueError("The number of quantiles (qnt) must be at least 2")
        self.lev = _check_lev(self.data[rvar], lev, rvar)
        self.qnt, self.cost, self.margin, self.scale = qnt, cost, margin, scale
        self.cnf = confusion(data, self.pred, rvar, lev=self.lev, cost=cost, margin=margin,
                             scale=scale, train=self.train, data_filter=self.data_filter,
                             sort=self.sort, slice=self.slice)

        tables = []
        for label, frame in self.dat_list.items():
            rv = _is_level(frame[rvar], self.lev)
            tot_resp = rv.sum() * scale
            tot_obs = frame.height * scale
            tot_rate = tot_resp / tot_obs
            for p in self.pred:
                pv = frame[p].cast(pl.Float64).to_numpy()
                bins = xtile(pv, qnt, rev=True)
                t = (
                    pl.DataFrame({"bins": bins, "resp": rv})
                    .group_by("bins").agg(
                        nr_obs=pl.len().cast(pl.Float64) * scale,
                        nr_resp=pl.col("resp").sum().cast(pl.Float64) * scale,
                    )
                    .sort("bins")
                    .with_columns(resp_rate=pl.col("nr_resp") / pl.col("nr_obs"),
                                  gains=pl.col("nr_resp") / tot_resp)
                )
                if t["resp_rate"][0] < t["resp_rate"][-1]:
                    t = t.reverse()
                t = t.with_columns(
                    profit=margin * pl.col("nr_resp").cum_sum() - cost * pl.col("nr_obs").cum_sum(),
                    ROME=(margin * pl.col("nr_resp").cum_sum() - cost * pl.col("nr_obs").cum_sum())
                    / (cost * pl.col("nr_obs").cum_sum()),
                    cum_prop=(pl.col("nr_obs") / tot_obs).cum_sum(),
                    cum_resp=pl.col("nr_resp").cum_sum(),
                ).with_columns(
                    cum_resp_rate=pl.col("cum_resp") / pl.col("nr_obs").cum_sum(),
                ).with_columns(
                    cum_lift=pl.col("cum_resp_rate") / tot_rate,
                    cum_gains=pl.col("cum_resp") / tot_resp,
                    ROME=pl.col("ROME").fill_nan(0).fill_null(0),
                ).with_columns(pred=pl.lit(p + _SUFFIX[label]))
                tables.append(t.select([
                    "pred", "bins", "nr_obs", "nr_resp", "resp_rate", "gains", "profit",
                    "ROME", "cum_prop", "cum_resp", "cum_resp_rate", "cum_lift", "cum_gains",
                ]))
        self.dataset = pl.concat(tables).with_columns(pl.col("bins").cast(pl.Int64))

    def summary(self, prn: bool = True, dec: int = 3) -> None:
        """Print the evaluation header and (when ``prn``) the per-bin table."""
        _header(self, "Evaluate predictions for binary response models", [
            ("Level", f"{self.lev} in {self.rvar}"),
            ("Bins", f"{self.qnt}"),
            ("Cost:Margin", f"{self.cost} : {self.margin}"),
            ("Scale", f"{self.scale}"),
        ])
        if prn:
            _print_table(self.dataset, dec)

    def _contact(self) -> pl.DataFrame:
        d = self.cnf.dataset
        return d.with_columns(
            pred=pl.col("Predictor") + pl.col("Type").replace_strict(_SUFFIX)
        ).select(["pred", "contact"])

    def _with_origin(self, cols: list[str]) -> pl.DataFrame:
        df = self.dataset.select(["pred", *cols])
        origin = df.group_by("pred", maintain_order=True).first().with_columns(
            [pl.lit(0.0).alias(c) for c in cols]
        )
        return pl.concat([origin, df.cast({c: pl.Float64 for c in cols})]).sort(
            "pred", maintain_order=True
        )

    def plot(self, plots=("lift", "gains")):
        """Lift, gains, profit, expected profit and/or ROME curves.

        Returns one plotnine plot, or a list of them when several are requested.
        """
        plots = _as_list(plots)
        unknown = [p for p in plots if p not in EVALBIN_PLOTS]
        if unknown:
            raise ValueError(f"Unknown plot type(s): {', '.join(unknown)}. Use {', '.join(EVALBIN_PLOTS)}")
        single = _single_series(self)
        out = []
        for kind in plots:
            if kind == "lift":
                out.append(_line_plot(self.dataset, "cum_prop", "cum_lift", "Cumulative lift",
                                      single, ref=(0, 1, 1, 1)))
            elif kind == "gains":
                out.append(_line_plot(self._with_origin(["cum_prop", "cum_gains"]), "cum_prop",
                                      "cum_gains", "Cumulative gains", single, ref=(0, 0, 1, 1),
                                      percent_y=True))
            elif kind == "profit":
                out.append(_line_plot(self._with_origin(["cum_prop", "profit"]), "cum_prop",
                                      "profit", "Profit", single, ref=(0, 0, 1, 0), comma_y=True,
                                      vlines=self._contact()))
            elif kind == "expected_profit":
                frames = []
                for label, frame in self.dat_list.items():
                    for p in self.pred:
                        pv = np.sort(frame[p].cast(pl.Float64).to_numpy())[::-1]
                        n = len(pv)
                        frames.append(pl.DataFrame({
                            "pred": [p + _SUFFIX[label]] * n,
                            "cum_prop": np.arange(1, n + 1) / n,
                            "cum_profit": np.cumsum(pv * self.margin - self.cost) * self.scale,
                        }))
                ep = pl.concat(frames)
                hl = ep.group_by("pred", maintain_order=True).agg(max_profit=pl.col("cum_profit").max())
                out.append(_line_plot(ep, "cum_prop", "cum_profit", "Expected profit", single,
                                      ref=(0, 0, 1, 0), comma_y=True, points=False,
                                      vlines=self._contact(), hlines=hl))
            elif kind == "rome":
                out.append(_line_plot(self.dataset, "cum_prop", "ROME",
                                      "Return on Marketing Expenditures (ROME)", single,
                                      ref=(0, 0, 1, 0), percent_y=True))
        return _plot_result(out)


class uplift(_Evaluation):
    """Evaluate uplift models from a treatment/control experiment.

    Bins are quantiles of each prediction among the treated rows (Radiant's
    ``uplift``); per bin the treated and control response counts give the
    uplift and the cumulative incremental response and profit.

    Parameters
    ----------
    data : pl.DataFrame | pd.DataFrame | dict[str, DataFrame]
        Dataset (or ``{name: df}``) with the response, treatment and predictions.
    pred : str | list[str]
        Numeric uplift prediction column(s); each is one model.
    rvar, tvar : str
        Response and treatment variables.
    lev, tlev : str, default ""
        The "success" level of ``rvar`` and the treated level of ``tvar``.
    qnt : int, default 10
        Number of quantile bins (computed on the treated rows).
    cost, margin, scale : float, default 1, 2 and 1
        Cost per treated contact, margin per response and a count multiplier.
    train : {"All", "Training", "Test", "Both"}, default "All"
    data_filter : str, default ""
        Filter that selects the training rows, e.g. "training == 1"
        (``pyrsm.data_scope``). Ignored when ``train="All"``.
    sort : str, default ""
        Column(s) to sort on before slicing.
    slice : str, default ""
        Rows to keep as training rows, counting from 1, e.g. "1:1000".
    test_filter : str, default ""
        Filter that selects the test rows, e.g. "training == 0". Blank: every row
        that is not a training row, as in Radiant.
    """

    def __init__(self, data, pred, rvar: str, tvar: str, lev: str = "", tlev: str = "",
                 qnt: int = 10, cost: float = 1, margin: float = 2, scale: float = 1,
                 train: str = "All", data_filter: str = "", sort: str = "",
                 slice: str = "", test_filter: str = "") -> None:  # noqa: A002
        self._setup(data, pred, rvar, train, data_filter, sort, slice, extra=(tvar,),
                    test_filter=test_filter)
        self.tvar = tvar
        qnt = 10 if not qnt else int(qnt)
        self.lev = _check_lev(self.data[rvar], lev, rvar)
        self.tlev = _check_lev(self.data[tvar], tlev, tvar)
        self.qnt, self.cost, self.margin, self.scale = qnt, cost, margin, scale
        self.cnf = confusion(data, self.pred, rvar, lev=self.lev, cost=cost, margin=margin,
                             scale=scale, train=self.train, data_filter=self.data_filter,
                             sort=self.sort, slice=self.slice)
        tables = []
        for label, frame in self.dat_list.items():
            rv = _is_level(frame[rvar], self.lev)
            tv = _is_level(frame[tvar], self.tlev)
            for p in self.pred:
                pv = frame[p].cast(pl.Float64).to_numpy()
                q = np.nanquantile(pv[tv], np.linspace(0, 1, qnt + 1))[1:qnt]
                breaks = np.concatenate([[-np.inf], q, [np.inf]])
                bins = np.clip(np.searchsorted(breaks, pv, side="left"), 1, qnt)
                bins = qnt + 1 - bins
                t = (
                    pl.DataFrame({"bins": bins, "t": tv, "r": rv})
                    .group_by("bins").agg(
                        T_resp=(pl.col("t") & pl.col("r")).sum().cast(pl.Float64) * scale,
                        T_n=pl.col("t").sum().cast(pl.Float64) * scale,
                        C_resp=(~pl.col("t") & pl.col("r")).sum().cast(pl.Float64) * scale,
                        C_n=(~pl.col("t")).sum().cast(pl.Float64) * scale,
                    )
                    .sort("bins")
                    .with_columns(uplift=pl.col("T_resp") / pl.col("T_n") - pl.col("C_resp") / pl.col("C_n"))
                    .with_columns(
                        cum_prop=pl.col("bins") / qnt,
                        T_resp=pl.col("T_resp").cum_sum(),
                        T_n=pl.col("T_n").cum_sum(),
                        C_resp=pl.col("C_resp").cum_sum(),
                        C_n=pl.col("C_n").cum_sum(),
                    )
                    .with_columns(
                        incremental_resp=pl.col("T_resp") - pl.col("C_resp") * pl.col("T_n") / pl.col("C_n"),
                    )
                    .with_columns(
                        incremental_profit=margin * pl.col("incremental_resp") - cost * pl.col("T_n"),
                        inc_uplift=pl.col("incremental_resp") / pl.col("T_n").last() * 100,
                        pred=pl.lit(p + _SUFFIX[label]),
                    )
                )
                tables.append(t.select([
                    "pred", "bins", "cum_prop", "T_resp", "T_n", "C_resp", "C_n",
                    "incremental_resp", "incremental_profit", "inc_uplift", "uplift",
                ]))
        self.dataset = pl.concat(tables).with_columns(pl.col("bins").cast(pl.Int64))

    def summary(self, prn: bool = True, dec: int = 3) -> None:
        """Print the uplift header and (when ``prn``) the per-bin table."""
        _header(self, "Evaluate uplift for binary response models", [
            ("Level", f"{self.lev} in {self.rvar}"),
            ("Treatment", self.tvar),
            ("Level", f"{self.tlev} in {self.tvar}"),
            ("Bins", f"{self.qnt}"),
            ("Cost:Margin", f"{self.cost} : {self.margin}"),
            ("Scale", f"{self.scale}"),
        ])
        if prn:
            _print_table(self.dataset, dec)

    def plot(self, plots=("inc_uplift", "uplift")):
        """Incremental uplift, uplift per bin and/or incremental profit."""
        from mizani.formatters import percent_format
        from plotnine import (
            aes,
            geom_col,
            ggplot,
            labs,
            position_dodge,
            scale_x_continuous,
            scale_y_continuous,
            theme,
        )

        plots = _as_list(plots)
        unknown = [p for p in plots if p not in UPLIFT_PLOTS]
        if unknown:
            raise ValueError(f"Unknown plot type(s): {', '.join(unknown)}. Use {', '.join(UPLIFT_PLOTS)}")
        single = _single_series(self)
        out = []
        for kind in plots:
            if kind == "inc_uplift":
                df = self.dataset.select(["pred", "cum_prop", "inc_uplift"])
                origin = df.group_by("pred", maintain_order=True).first().with_columns(
                    cum_prop=pl.lit(0.0), inc_uplift=pl.lit(0.0))
                df = pl.concat([origin, df]).sort("pred", maintain_order=True).with_columns(
                    inc_uplift=pl.col("inc_uplift") / 100)
                yend = float(self.dataset["inc_uplift"][-1]) / 100
                out.append(_line_plot(df, "cum_prop", "inc_uplift", "Incremental uplift", single,
                                      ref=(0, 0, 1, yend), percent_y=True))
            elif kind == "uplift":
                pdf = self.dataset.select(["pred", "cum_prop", "uplift"]).to_pandas()
                if single:
                    g = ggplot(pdf, aes(x="cum_prop", y="uplift")) + geom_col(fill=FILL)
                else:
                    g = ggplot(pdf, aes(x="cum_prop", y="uplift", fill="pred")) + geom_col(
                        position=position_dodge())
                g += labs(x="Proportion of population targeted", y="Uplift", fill="Predictor")
                g += scale_x_continuous(labels=percent_format())
                g += scale_y_continuous(labels=percent_format())
                g += theme_pyrsm()
                if single:
                    g += theme(legend_position="none")
                out.append(g)
            elif kind == "inc_profit":
                df = self.dataset.select(["pred", "cum_prop", "incremental_profit"])
                origin = df.group_by("pred", maintain_order=True).first().with_columns(
                    cum_prop=pl.lit(0.0), incremental_profit=pl.lit(0.0))
                df = pl.concat([origin, df]).sort("pred", maintain_order=True)
                cnf = self.cnf.dataset.with_columns(
                    pred=pl.col("Predictor") + pl.col("Type").replace_strict(_SUFFIX)
                ).select(["pred", "contact"])
                out.append(_line_plot(df, "cum_prop", "incremental_profit", "Incremental profit",
                                      single, ref=(0, 0, 1, 0), comma_y=True, vlines=cnf))
        return _plot_result(out)
