"""Naive Bayes classifier -- Python port of ``radiant.model::nb``.

Radiant estimates the model with ``e1071::naiveBayes``. This module ports that
estimator and ``e1071``'s ``predict.naiveBayes`` so the same data and settings
give the same a-priori counts, conditional probability tables and predicted
probabilities as Radiant:

* categorical explanatory variables get a table of conditional probabilities
  ``P(x = level | y)``, with optional Laplace smoothing
  ``(count + laplace) / (row total + laplace * number of levels)``;
* numeric explanatory variables get the mean and (sample) standard deviation of
  ``x`` for each level of the response, and a normal density at prediction time;
* predictions combine the (log) a-priori counts and the (log) likelihoods.
  Probabilities at or below ``eps`` (0) are replaced by ``threshold`` (0.001), as
  are standard deviations at or below ``eps``. A missing value, or a level that
  was not seen during estimation, is ignored for that variable.

scikit-learn's ``GaussianNB`` and ``CategoricalNB`` are not used: they use the
population variance plus a smoothing term and cannot mix variable types.

Examples
--------
>>> import polars as pl
>>> import pyrsm as rsm
>>> titanic = pl.read_parquet("examples/data/model/titanic.parquet")
>>> m = rsm.model.nb({"titanic": titanic}, rvar="survived",
...                  evar=["pclass", "sex", "age"])
>>> m.summary()
>>> m.predict(cmd={"pclass": ["1st", "2nd", "3rd"]})
>>> m.plot("vimp")
"""

from __future__ import annotations

import math
from itertools import combinations, product
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from pyrsm.data_scope import apply_scope, print_scope
from pyrsm.model.crtree import _r_format, _r_num2str
from pyrsm.model.model import convert_to_list, nobs_dropped
from pyrsm.utils import check_dataframe

__all__ = ["nb"]

_M_1_SQRT_2PI = 0.398942280401432677939946059934
# sqrt(-2 * ln(2) * (DBL_MIN_EXP + 1 - DBL_MANT_DIG)): beyond this dnorm underflows
_DNORM_UNDERFLOW = math.sqrt(-2 * math.log(2) * (-1021 + 1 - 53))
_R_WIDTH = 80  # R's default print width


# ---------------------------------------------------------------------------
# numerical helpers (R semantics)
# ---------------------------------------------------------------------------


def _dnorm(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """R's ``dnorm(x, mu, sigma)`` (``dnorm4`` in nmath, non-log scale).

    Ported operation for operation so densities in the tails, and the point at
    which they underflow to 0 (and so are replaced by e1071's threshold), match R.
    """
    x, mu, sigma = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    )
    out = np.full(x.shape, np.nan)
    with np.errstate(all="ignore"):
        ok = ~(np.isnan(x) | np.isnan(mu) | np.isnan(sigma))
        ok &= ~(~np.isfinite(x) & (mu == x))
        zero_sd = ok & (sigma == 0)
        out[zero_sd] = np.where(x[zero_sd] == mu[zero_sd], np.inf, 0.0)
        pos = ok & (sigma > 0)
        z = np.full(x.shape, np.nan)
        z[pos] = np.abs((x[pos] - mu[pos]) / sigma[pos])
        out[pos & ~np.isfinite(z)] = 0.0
        fin = pos & np.isfinite(z)
        out[fin & (z >= 2 * math.sqrt(np.finfo(float).max))] = 0.0
        small = fin & (z < 5)
        out[small] = _M_1_SQRT_2PI * np.exp(-0.5 * z[small] * z[small]) / sigma[small]
        out[fin & (z >= 5) & (z > _DNORM_UNDERFLOW)] = 0.0
        mid = fin & (z >= 5) & (z <= _DNORM_UNDERFLOW)
        x1 = np.ldexp(np.rint(np.ldexp(z[mid], 16)), -16)
        x2 = z[mid] - x1
        out[mid] = (
            _M_1_SQRT_2PI / sigma[mid] * (np.exp(-0.5 * x1 * x1) * np.exp((-0.5 * x2 - x1) * x2))
        )
    return out


def _auc(pred: np.ndarray, actual: np.ndarray) -> float:
    """radiant.model's ``auc``: Mann-Whitney U, folded so it is at least 0.5."""
    from scipy.stats import rankdata

    actual = np.asarray(actual, dtype=bool)
    n1 = float(np.sum(~actual))
    n2 = float(np.sum(actual))
    U = float(np.sum(rankdata(pred)[~actual])) - n1 * (n1 + 1) / 2
    wt = U / n1 / n2
    return 1 - wt if wt < 0.5 else wt


def _r_cat_num(x: float) -> str:
    """How R's ``cat`` shows a number (7 significant digits)."""
    return _r_format([x], digits=7)[0].strip()


def _fmt_column(values) -> list[str]:
    """Format a numeric column like R's ``print`` (shared decimals, NA for missing)."""
    vals = [float(v) for v in values]
    fin = [v for v in vals if math.isfinite(v)]
    fmt = iter(_r_format(fin, digits=7)) if fin else iter(())
    out = []
    for v in vals:
        if math.isfinite(v):
            out.append(next(fmt))
        elif math.isnan(v):
            out.append("NA")
        else:
            out.append("Inf" if v > 0 else "-Inf")
    w = max((len(s) for s in out), default=0)
    return [s.rjust(w) for s in out]


def _r_print_table(rname: str, cname: str, rownames, colnames, columns) -> list[str]:
    """Print a matrix with named dimnames like R (columns wrap at 80 characters).

    ``columns`` holds one list of numbers per column.
    """
    cells = [_fmt_column(c) for c in columns]
    widths = [max(len(cn), max((len(v) for v in c), default=0)) + 1 for cn, c in zip(colnames, cells)]
    rlabw = max((len(r) for r in rownames), default=0)
    if len(rname) < rlabw + 2:
        lbloff = 2
    else:
        lbloff = len(rname) - rlabw
    rlabw += lbloff
    lines = []
    jmin = 0
    ncol = len(colnames)
    while jmin < ncol:
        width = rlabw
        jmax = jmin
        while True:
            width += widths[jmax]
            jmax += 1
            if not (jmax < ncol and width + widths[jmax] < _R_WIDTH):
                break
        lines.append(" " * rlabw + cname)
        lines.append(
            rname.ljust(rlabw) + "".join(colnames[j].rjust(widths[j]) for j in range(jmin, jmax))
        )
        lab_w = rlabw - lbloff
        for i, rn in enumerate(rownames):
            lines.append(
                " " * lbloff
                + rn.ljust(lab_w)
                + "".join(cells[j][i].rjust(widths[j]) for j in range(jmin, jmax))
            )
        jmin = jmax
    return lines


def _r_print_named(header: str, names, values) -> list[str]:
    """Print a one-dimensional table with a named dimension like R."""
    vals = _fmt_column(values)
    w = max(max((len(n) for n in names), default=0), max((len(v) for v in vals), default=0))
    per_line = max(1, _R_WIDTH // (w + 1))
    lines = [header]
    for i in range(0, len(names), per_line):
        lines.append("".join(n.rjust(w) + " " for n in names[i : i + per_line]))
        lines.append("".join(v.rjust(w) + " " for v in vals[i : i + per_line]))
    return lines


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------


def _is_numeric(s: pl.Series) -> bool:
    """Numeric in R's sense (``is.numeric``): integers and doubles, not logicals or dates."""
    return s.dtype.is_numeric()


def _labels(s: pl.Series) -> list:
    """Values of a categorical variable as the labels R would show (None if missing)."""
    if s.dtype == pl.Boolean:
        return [None if v is None else ("TRUE" if v else "FALSE") for v in s.to_list()]
    if s.dtype.is_numeric():
        return [None if v is None else _r_num2str(float(v)) for v in s.to_list()]
    return s.cast(pl.Utf8).to_list()


def _levels(s: pl.Series) -> list[str]:
    """Levels of a categorical variable after ``droplevels``: Enum order, or order
    of appearance for text."""
    if s.dtype == pl.Boolean:
        # e1071 turns a logical into factor(x, levels = c("FALSE", "TRUE"))
        return ["FALSE", "TRUE"]
    present = set(v for v in _labels(s) if v is not None)
    if isinstance(s.dtype, pl.Enum):
        return [c for c in s.dtype.categories.to_list() if c in present]
    if s.dtype.is_temporal():
        uniq = s.drop_nulls().unique().sort()
        return [str(v) for v in uniq.to_list()]
    # text: order of appearance (pyrsm policy: categorical data should be an
    # Enum with a deliberate order; never sort alphabetically)
    return [v for v in s.cast(pl.Utf8).drop_nulls().unique(maintain_order=True).to_list()
            if v in present]


def _response_levels(s: pl.Series) -> list[str]:
    """Levels of the response, as R's ``factor()`` would order them."""
    if s.dtype.is_numeric():
        return [_r_num2str(float(v)) for v in np.unique(s.cast(pl.Float64).to_numpy())]
    return _levels(s)


def _as_numeric(s: pl.Series) -> np.ndarray:
    """radiant.data's ``as_numeric``, used for the variable importance plot."""
    if s.dtype == pl.Boolean:
        return s.cast(pl.Float64).to_numpy()
    if s.dtype.is_numeric():
        return s.cast(pl.Float64).to_numpy()
    if s.dtype.is_temporal():
        return s.to_physical().cast(pl.Float64).to_numpy()
    labels = _labels(s)
    num = []
    for v in labels:
        try:
            num.append(float(v))
        except (TypeError, ValueError):
            num.append(np.nan)
    num = np.array(num, dtype=float)
    if np.all(np.isnan(num)):
        code = {lv: i + 1.0 for i, lv in enumerate(_levels(s))}
        return np.array([code.get(v, np.nan) for v in labels], dtype=float)
    # labels that are not numbers are ranked last, as R's rank does with NA
    return np.where(np.isnan(num), np.inf, num)


# ---------------------------------------------------------------------------
# nb
# ---------------------------------------------------------------------------


class nb:
    """
    Naive Bayes classifier, as in Radiant's ``nb`` (``e1071::naiveBayes``).

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame]
        Data to estimate the model on. Pass ``{"name": df}`` to show the name in
        the summary.
    rvar : str
        Response variable. Converted to categorical if it is not already
        (numbers are ordered by value, text alphabetically, logicals FALSE, TRUE).
    evar : list[str]
        Explanatory variables. Numeric variables (integers and floats) get a
        normal distribution per level of the response, all other variables
        (categorical, text, logical, dates) a table of conditional probabilities.
    lev : str, optional
        Level of the response to focus the plots on (Radiant's "Choose level"
        for plots). The model itself always uses all levels of the response, as
        in Radiant. Defaults to "All levels".
    laplace : float, default 0
        Laplace smoothing for the conditional probability tables of categorical
        variables. 0 disables smoothing.
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
    levels : list[str]
        Levels of the response variable (one predicted probability per level).
    apriori : pl.DataFrame
        Count and proportion of each level of the response.
    tables : dict[str, pl.DataFrame]
        For each explanatory variable, the conditional probabilities (one column
        per level) or the mean and standard deviation ("mean", "st.dev") for
        each level of the response.
    isnumeric : dict[str, bool]
        Whether each explanatory variable was treated as numeric.
    data : pl.DataFrame
        Estimation data (response and explanatory variables, rows with missing
        values removed).
    """

    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        rvar: str | None = None,
        evar: list[str] | str | None = None,
        lev: str | None = None,
        laplace: float = 0,
        data_filter: str = "",
        sort: str = "",
        slice: str = "",
    ) -> None:
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            data = data[self.name]
        else:
            self.name = "Not provided"
        # keep the level order of pandas categoricals (R factors), which a plain
        # conversion to polars would lose
        cats = {}
        if isinstance(data, pd.DataFrame):
            cats = {
                c: [str(x) for x in data[c].cat.categories]
                for c in data.columns
                if isinstance(data[c].dtype, pd.CategoricalDtype)
            }
        data = check_dataframe(data)
        if cats:
            data = data.with_columns(
                [pl.col(c).cast(pl.Utf8).cast(pl.Enum(levs)) for c, levs in cats.items()]
            )

        # The data scope, applied before anything is measured: filter, then
        # sort, then slice. Kept on the object so ``summary`` can say which
        # rows the numbers describe.
        self.data_filter = data_filter
        self.sort = sort
        self.slice = slice
        data = apply_scope(data, data_filter, sort, slice)

        self.rvar = rvar
        self.evar = convert_to_list(evar)
        if rvar is None or rvar == "":
            raise ValueError("Please select a response variable.")
        if len(self.evar) == 0:
            raise ValueError("Please select one or more explanatory variables.")
        if rvar in self.evar:
            raise ValueError(
                "Response variable contained in the set of explanatory variables. "
                "Please update model specification."
            )
        missing = [c for c in [rvar] + self.evar if c not in data.columns]
        if missing:
            raise ValueError(f"Variable(s) not in the data: {', '.join(missing)}")
        if isinstance(laplace, bool) or not isinstance(laplace, int | float) or laplace < 0:
            raise ValueError("laplace should be a number larger than or equal to 0")
        self.laplace = laplace

        # get_data(..., na.rm = TRUE): drop rows with a missing value (R's
        # na.omit also drops NaN)
        self.nobs_all = data.height
        data = data.select([rvar] + self.evar).drop_nulls()
        floats = [c for c in data.columns if data[c].dtype in (pl.Float32, pl.Float64)]
        if floats:
            data = data.filter(~pl.any_horizontal([pl.col(c).is_nan() for c in floats]))
        self.nobs = data.height
        self.nobs_dropped = self.nobs_all - self.nobs

        not_vary = []
        for c in [rvar] + self.evar:
            s = data[c]
            if s.len() <= 1:
                not_vary.append(c)
            elif s.dtype.is_numeric() or s.dtype == pl.Boolean or s.dtype.is_temporal():
                v = s.to_physical().cast(pl.Float64).to_numpy()
                if abs(np.max(v) - np.min(v)) <= np.finfo(float).eps ** 0.5:
                    not_vary.append(c)
            elif s.n_unique() <= 1:
                not_vary.append(c)
        if not_vary:
            raise ValueError(
                "The following variable(s) show no variation. Please select other "
                f"variables: {', '.join(not_vary)}"
            )

        # response as a factor
        rv = data[rvar]
        self.levels = _response_levels(rv)
        if rv.dtype.is_numeric():
            ylab = [_r_num2str(float(v)) for v in rv.cast(pl.Float64).to_list()]
        else:
            ylab = _labels(rv)
        code = {lv: i for i, lv in enumerate(self.levels)}
        self.y = np.array([code[v] for v in ylab], dtype=int)
        if lev is not None and lev != "" and lev != "All levels":
            if lev not in self.levels:
                raise ValueError(f"Specified level is not a level in {rvar}")
        else:
            lev = None
        self.lev = lev
        self.data = data.select([rvar] + self.evar)
        self.ml_model = {"model": "nb", "mod_type": "classification"}

        # e1071::naiveBayes
        k = len(self.levels)
        counts = np.bincount(self.y, minlength=k).astype(float)
        self._apriori = counts
        self.isnumeric: dict[str, bool] = {}
        self.xlevels: dict[str, list[str]] = {}
        self._tables: dict[str, np.ndarray] = {}
        for v in self.evar:
            s = data[v]
            if _is_numeric(s):
                self.isnumeric[v] = True
                x = s.cast(pl.Float64).to_numpy()
                tab = np.full((k, 2), np.nan)
                for j in range(k):
                    xj = x[self.y == j]
                    if xj.size > 0:
                        tab[j, 0] = np.mean(xj)
                    if xj.size > 1:
                        tab[j, 1] = np.std(xj, ddof=1)
                self._tables[v] = tab
            else:
                self.isnumeric[v] = False
                levs = _levels(s)
                self.xlevels[v] = levs
                xc = {lv: i for i, lv in enumerate(levs)}
                xcode = np.array([xc[lv] for lv in _labels(s)], dtype=int)
                tab = np.zeros((k, len(levs)))
                np.add.at(tab, (self.y, xcode), 1.0)
                rs = tab.sum(axis=1, keepdims=True)
                with np.errstate(invalid="ignore", divide="ignore"):
                    self._tables[v] = (tab + laplace) / (rs + laplace * len(levs))

    # -- results -------------------------------------------------------------
    @property
    def apriori(self) -> pl.DataFrame:
        """Count and proportion of each level of the response."""
        return pl.DataFrame(
            {
                self.rvar: self.levels,
                "count": self._apriori,
                "probability": self._apriori / self._apriori.sum(),
            }
        )

    @property
    def tables(self) -> dict[str, pl.DataFrame]:
        """Conditional probabilities (categorical) or means and standard
        deviations (numeric) for each explanatory variable."""
        out = {}
        for v in self.evar:
            cols = ["mean", "st.dev"] if self.isnumeric[v] else self.xlevels[v]
            tab = self._tables[v]
            df = {self.rvar: self.levels}
            for j, c in enumerate(cols):
                df[c] = tab[:, j]
            out[v] = pl.DataFrame(df)
        return out

    # -- printing ------------------------------------------------------------
    def summary(self, dec: int = 3) -> None:
        """
        Summarize a naive Bayes model, as Radiant's ``summary.nb``.

        Parameters
        ----------
        dec : int, default 3
            Number of decimals for the conditional probabilities, means and
            standard deviations. A-priori probabilities are shown with 3 decimals.
        """
        print("Naive Bayes Classifier")
        print(f"Data                 : {self.name}")
        print_scope(self, 21)
        print(f"Response variable    : {self.rvar}")
        print(f"Levels               : {', '.join(self.levels)} in {self.rvar}")
        print(f"Explanatory variables: {', '.join(self.evar)}")
        print(f"Laplace              : {_r_cat_num(self.laplace)}")
        print(f"Nr obs               : {self.nobs:,.0f}{nobs_dropped(self)}")
        print()
        print("A-priori probabilities:")
        prob = self._apriori / self._apriori.sum()
        for line in _r_print_named(self.rvar, self.levels, [round(float(p), 3) for p in prob]):
            print(line)
        print()
        print("Conditional probabilities (categorical) or means & st.dev (numeric):")
        for v in self.evar:
            tab = self._tables[v]
            cols = ["mean", "st.dev"] if self.isnumeric[v] else self.xlevels[v]
            columns = [
                [round(float(x), dec) if math.isfinite(x) else x for x in tab[:, j]]
                for j in range(tab.shape[1])
            ]
            for line in _r_print_table(self.rvar, v, self.levels, cols, columns):
                print(line)
            print()

    # -- prediction ----------------------------------------------------------
    def _predict_proba(
        self, df: pl.DataFrame, threshold: float = 0.001, eps: float = 0.0
    ) -> np.ndarray:
        """Posterior probabilities (``predict.naiveBayes(type = "raw")``)."""
        n = df.height
        k = len(self.levels)
        logsum = np.zeros((n, k))
        for v in self.evar:
            s = df[v]
            tab = self._tables[v]
            if self.isnumeric[v]:
                if not _is_numeric(s):
                    raise ValueError(
                        f"Type mismatch between estimation and prediction data for "
                        f"variable '{v}': it should be numeric"
                    )
                x = s.cast(pl.Float64).to_numpy()
                sd = tab[:, 1].copy()
                sd[sd <= eps] = threshold
                prob = _dnorm(x[:, None], tab[None, :, 0], sd[None, :])
                prob[prob <= eps] = threshold
                miss = np.isnan(x)
            else:
                xc = {lv: i for i, lv in enumerate(self.xlevels[v])}
                codes = np.array(
                    [xc.get(lv, -1) if lv is not None else -1 for lv in _labels(s)], dtype=int
                )
                miss = codes < 0
                prob = tab[:, np.where(miss, 0, codes)].T.copy()
                prob[prob <= eps] = threshold
            prob[miss, :] = 1.0
            with np.errstate(divide="ignore", invalid="ignore"):
                logsum += np.log(prob)
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.log(self._apriori)[None, :] + logsum
            # 1 / sum(exp(L - lp)) for each level, as e1071 computes it
            post = 1.0 / np.exp(L[:, None, :] - L[:, :, None]).sum(axis=2)
        return post

    def _cmd_grid(self, base: pl.DataFrame, cmd: dict) -> pl.DataFrame:
        """Prediction data from a command: the variables in ``cmd`` take the
        values given, the others are fixed at their mean (numeric) or most
        common value (categorical), as in Radiant's ``predict_model``."""
        unknown = [k for k in cmd if k not in self.evar]
        if unknown:
            raise ValueError(
                "The command entered contains variable names that are not in the "
                f"model\nVariables in the model: {', '.join(self.evar)}\n"
                f"Variables not available in prediction data: {', '.join(unknown)}"
            )
        values = {}
        for v in self.evar:
            s = base[v]
            if v in cmd:
                val = cmd[v]
                if isinstance(val, str | bytes) or not hasattr(val, "__iter__"):
                    val = [val]
                values[v] = list(val)
            elif s.dtype.is_numeric():
                values[v] = [float(s.cast(pl.Float64).mean())]
            elif s.dtype == pl.Boolean:
                values[v] = [bool(s.cast(pl.Float64).mean() > 0.5)]
            else:
                # most frequent value; ties go to the first level, as which.max does
                labels = [x for x in _labels(s) if x is not None]
                levs = _levels(s)
                cnt = {lv: 0 for lv in levs}
                for x in labels:
                    cnt[x] += 1
                best = max(levs, key=lambda lv: cnt[lv])
                values[v] = [s.filter(pl.Series(_labels(s)) == best)[0]]
        rows = list(product(*values.values()))
        cols = {}
        for j, v in enumerate(self.evar):
            col = [r[j] for r in rows]
            dtype = base[v].dtype
            if self.isnumeric[v]:
                cols[v] = pl.Series(v, col, dtype=pl.Float64)
                if dtype.is_integer() and all(float(x).is_integer() for x in col):
                    cols[v] = cols[v].cast(dtype)
            else:
                ser = pl.Series(v, col)
                try:
                    ser = ser.cast(dtype)
                except Exception:  # values outside the categories stay as text
                    ser = ser.cast(pl.Utf8)
                cols[v] = ser
        return pl.DataFrame(cols)

    def predict(
        self,
        data: pd.DataFrame | pl.DataFrame | None = None,
        cmd: dict | None = None,
        data_cmd: dict | None = None,
        dec: int | None = None,
        threshold: float = 0.001,
        eps: float = 0.0,
    ) -> pl.DataFrame:
        """
        Predicted probabilities for each level of the response.

        Parameters
        ----------
        data : pd.DataFrame | pl.DataFrame, optional
            Data to predict for. Defaults to the estimation data. Rows with a
            missing value in an explanatory variable are dropped, as in Radiant.
        cmd : dict, optional
            Values to vary, other variables held at their mean or most common
            value, e.g. ``{"pclass": ["1st", "2nd", "3rd"]}``.
        data_cmd : dict, optional
            Values to set for all rows of ``data``, e.g. ``{"sex": "male"}``.
        dec : int, optional
            Round the probabilities (and other float columns) to this many decimals.
        threshold : float, default 0.001
            Value that replaces probabilities and standard deviations at or
            below ``eps`` (e1071's ``threshold``).
        eps : float, default 0
            e1071's ``eps``.

        Returns
        -------
        pl.DataFrame
            The explanatory variables plus one column of predicted probabilities
            per level of the response.
        """
        if data is None:
            base = self.data.select(self.evar)
        else:
            base = check_dataframe(data)
            missing = [v for v in self.evar if v not in base.columns]
            if missing:
                raise ValueError(
                    "All variables in the model must also be in the prediction data\n"
                    f"Variables in the model: {', '.join(self.evar)}\n"
                    f"Variables not available in prediction data: {', '.join(missing)}"
                )
            base = base.select(self.evar)
        if data_cmd is not None and data_cmd != "":
            unknown = [k for k in data_cmd if k not in self.evar]
            if unknown:
                raise ValueError(
                    "The command entered contains variable names that are not in "
                    f"the model: {', '.join(unknown)}"
                )
            pred_data = base.with_columns(
                [pl.lit(v).cast(base[k].dtype).alias(k) for k, v in data_cmd.items()]
            )
        elif cmd is not None and cmd != "":
            pred_data = self._cmd_grid(base, cmd)
        else:
            pred_data = base
        # na.omit, as Radiant's predict_model
        pred_data = pred_data.drop_nulls()
        floats = [c for c in pred_data.columns if pred_data[c].dtype in (pl.Float32, pl.Float64)]
        if floats:
            pred_data = pred_data.filter(
                ~pl.any_horizontal([pl.col(c).is_nan() for c in floats])
            )
        probs = self._predict_proba(pred_data, threshold=threshold, eps=eps)
        pred = pred_data.with_columns(
            [pl.Series(lv, probs[:, j]) for j, lv in enumerate(self.levels)]
        )
        if dec is not None:
            pred = pred.with_columns(
                [pl.col(c).round(dec) for c in pred.columns if pred[c].dtype in (pl.Float64, pl.Float32)]
            )
        return pred

    # -- plots ---------------------------------------------------------------
    def _plot_lev(self, lev: str | None) -> str:
        if lev is None:
            lev = self.lev if self.lev is not None else "All levels"
        if lev != "All levels" and lev not in self.levels:
            raise ValueError(f"Specified level is not a level in {self.rvar}")
        return lev

    def vimp(self, lev: str | None = None) -> pl.DataFrame:
        """
        Variable importance as in Radiant's nb plot: the AUC of each explanatory
        variable (as a number) for separating levels of the response.

        With two levels (or a chosen ``lev``, compared to all other levels)
        there is one AUC per variable, sorted from low to high. With more
        levels there is one AUC per variable for each pair of levels.

        Parameters
        ----------
        lev : str, optional
            Level of the response to compare to all other levels. Defaults to
            the ``lev`` given when the model was estimated, else "All levels".
        """
        lev = self._plot_lev(lev)
        X = {v: _as_numeric(self.data[v]) for v in self.evar}
        y = self.y
        levels = list(self.levels)
        if lev != "All levels":
            y = np.where(y == self.levels.index(lev), 0, 1)
            levels = [lev, f"not_{lev}"]
        if len(levels) == 2:
            df = pl.DataFrame(
                {"vars": self.evar, "auc": [_auc(X[v], y == 0) for v in self.evar]}
            )
            return df.sort("auc", maintain_order=True)
        rows = []
        for a, b in combinations(range(len(levels)), 2):
            ind = (y == a) | (y == b)
            for v in self.evar:
                rows.append((f"{levels[a]} vs {levels[b]}", v, _auc(X[v][ind], y[ind] == a)))
        return pl.DataFrame(rows, schema=["Predict", "vars", "auc"], orient="row")

    def plot(
        self,
        plots: Literal["correlations", "vimp"] = "correlations",
        lev: str | None = None,
        nobs: int = 1000,
    ):
        """
        Plots for a naive Bayes model, as Radiant's ``plot.nb``.

        Parameters
        ----------
        plots : {"correlations", "vimp"}, default "correlations"
            "correlations" shows a correlation (scatter) matrix of the variables
            in the model, to check the conditional independence assumption.
            "vimp" plots the variable importance (AUC) of each explanatory
            variable (see ``vimp``).
        lev : str, optional
            Level of the response the plot is based on. For "correlations" only
            the rows with that level are used (explanatory variables only); for
            "vimp" that level is compared to all other levels. Defaults to the
            ``lev`` given when the model was estimated, else "All levels".
        nobs : int, default 1000
            Number of data points to show in the scatter plots (-1 for all).
        """
        plots = convert_to_list(plots)
        lev = self._plot_lev(lev)
        if len(plots) == 0 or plots[0] in ("", "none", None):
            return None
        if "correlations" in plots:
            from pyrsm.basics.correlation import correlation

            if lev == "All levels":
                df = self.data
            else:
                df = self.data.filter(self.y == self.levels.index(lev)).select(self.evar)
            # logicals and dates are shown as categories
            df = df.with_columns(
                [
                    pl.Series(c, _labels(df[c])).cast(pl.Enum(_levels(df[c])))
                    for c in df.columns
                    if df[c].dtype == pl.Boolean or df[c].dtype.is_temporal()
                ]
            )
            cat = any(not df[c].dtype.is_numeric() for c in df.columns)
            cr = correlation(df, vars=df.columns, method="polychoric" if cat else "pearson")
            return cr.plot(nobs=nobs)
        if "vimp" in plots:
            from plotnine import (
                aes,
                coord_flip,
                element_text,
                geom_col,
                ggplot,
                guide_legend,
                guides,
                labs,
                scale_x_discrete,
                theme,
            )

            from pyrsm.plot_style import ALPHA, FILL, theme_pyrsm

            vimp = self.vimp(lev)
            ymax = float(vimp["auc"].max())
            if "Predict" not in vimp.columns:
                return (
                    ggplot(vimp, aes(x="vars", y="auc"))
                    + geom_col(fill=FILL, alpha=ALPHA)
                    + scale_x_discrete(limits=vimp["vars"].to_list())
                    + labs(x="", y="Variable Importance (AUC)")
                    + coord_flip(ylim=(0.5, ymax))
                    + theme_pyrsm()
                    + theme(axis_text_y=element_text(ha="left"))
                )
            pairs = list(dict.fromkeys(vimp["Predict"].to_list()))
            vimp = vimp.with_columns(pl.col("vars").cast(pl.Enum(self.evar)))
            return (
                ggplot(vimp, aes(x="Predict", y="auc", fill="vars"))
                + geom_col(position="dodge", alpha=ALPHA)
                + scale_x_discrete(limits=pairs[::-1])
                + guides(fill=guide_legend(title=""))
                + labs(x="", y="Variable Importance (AUC)")
                + coord_flip(ylim=(0.5, ymax))
                + theme_pyrsm()
                + theme(axis_text_y=element_text(ha="left"))
            )
        raise ValueError(f"Unknown plot type(s): {', '.join(map(str, plots))}")
