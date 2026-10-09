"""Pre-factor analysis diagnostics — Python port of ``radiant.multivariate::pre_factor``.

Produces the Bartlett test of sphericity, the Kaiser-Meyer-Olkin measure of
sampling adequacy (overall and per variable), the eigenvalues of the
correlation matrix, and per-variable collinearity R-squared values, plus scree
and eigenvalue-change plots.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pyrsm.basics import display_utils as du
from pyrsm.utils import format_nr

from ._correlation import heterogeneous_corr
from ._plotting import as_plot_result
from ._utils import apply_filter, does_vary, get_data, get_vars, is_categorical
from pyrsm.data_scope import apply_scope, print_scope

__all__ = ["pre_factor", "kmo"]


def kmo(cmat: np.ndarray):
    """Kaiser-Meyer-Olkin measure of sampling adequacy.

    Reproduces ``psych::KMO``. Returns ``(MSA, MSAi)`` where ``MSA`` is the
    overall measure and ``MSAi`` is the per-variable vector.
    """
    cmat = np.asarray(cmat, dtype=float)
    Qinv = np.linalg.inv(cmat)
    d = np.sqrt(np.diag(Qinv))
    # partial correlation matrix = cov2cor(inverse correlation matrix)
    P = Qinv / np.outer(d, d)
    R = cmat.copy()
    np.fill_diagonal(P, 0.0)
    np.fill_diagonal(R, 0.0)
    sumR2 = (R**2).sum()
    sumP2 = (P**2).sum()
    MSA = sumR2 / (sumR2 + sumP2)
    colR2 = (R**2).sum(axis=0)
    colP2 = (P**2).sum(axis=0)
    MSAi = colR2 / (colR2 + colP2)
    return float(MSA), MSAi


class pre_factor:
    """Pre-factor analysis diagnostics.

    Parameters
    ----------
    data : pl.DataFrame | pd.DataFrame | dict[str, DataFrame]
        Dataset (or ``{name: df}``) to analyze.
    vars : str | list[str]
        Variables to use; supports Radiant ranges such as ``"v1:v6"``.
    hcor : bool, default False
        Treat categorical (Enum/Categorical) variables as ordinal and build a
        heterogeneous (polychoric/polyserial) correlation matrix.

    Attributes
    ----------
    cmat : np.ndarray
        Correlation matrix.
    btest : dict
        Bartlett test with keys ``chisq``, ``df``, ``p.value``.
    pre_kmo : float
        Overall KMO/MSA.
    msai : np.ndarray
        Per-variable MSA.
    pre_eigen : np.ndarray
        Eigenvalues of the correlation matrix (descending).
    pre_r2 : np.ndarray
        Per-variable collinearity R-squared (``1 - 1/diag(inv(cmat))``).
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

    def __init__(self, data, vars, hcor: bool = False, data_filter: str = "", sort: str = "", slice: str = "") -> None:
        self.name, self.data = get_data(data)

        # The data scope, applied before anything is measured: filter, then
        # sort, then slice. Kept on the object so ``summary`` can say which
        # rows the numbers describe.
        self.data_filter = data_filter
        self.sort = sort
        self.slice = slice
        self.data = apply_scope(self.data, data_filter, sort, slice)
        self.vars = get_vars(self.data.columns, vars)
        self.hcor = hcor

        sub = self.data.select(self.vars).drop_nulls()
        self.nobs = sub.height
        p = len(self.vars)

        no_var = [v for v in self.vars if not does_vary(sub[v])]
        if no_var:
            raise ValueError(
                "The following variable(s) show no variation. Please select "
                f"other variables: {', '.join(no_var)}"
            )
        self.any_categorical = any(is_categorical(sub[v]) for v in self.vars)

        self.cmat, _ = heterogeneous_corr(sub, self.vars, hcor)

        # Bartlett test of sphericity (psych::cortest.bartlett)
        from scipy.stats import chi2

        sign, logdet = np.linalg.slogdet(self.cmat)
        detcm = np.exp(logdet) * sign
        chisq = -logdet * (self.nobs - 1 - (2 * p + 5) / 6) if sign > 0 else np.nan
        df = p * (p - 1) / 2
        pval = float(chi2.sf(chisq, df)) if np.isfinite(chisq) else np.nan
        self.btest = {"chisq": float(chisq), "df": float(df), "p.value": pval}

        # KMO / MSA
        self.pre_kmo, self.msai = kmo(self.cmat)

        # eigenvalues (descending)
        self.pre_eigen = np.sort(np.linalg.eigvalsh(self.cmat))[::-1]

        # per-variable collinearity R-squared
        if detcm > 0:
            try:
                scmat = np.linalg.inv(self.cmat)
                self.pre_r2 = 1.0 - (1.0 / np.diag(scmat))
            except np.linalg.LinAlgError:
                self.pre_r2 = np.full(p, np.nan)
        else:
            self.pre_r2 = np.full(p, np.nan)

    # -- output ---------------------------------------------------------------

    def summary(self, dec: int = 2) -> None:
        """Print Radiant-style pre-factor diagnostics."""
        print("Pre-factor analysis diagnostics")
        print(f"Data        : {self.name}")
        print_scope(self, 12)
        print(f"Variables   : {', '.join(self.vars)}")
        print(f"Observations: {format_nr(self.nobs, dec=0)}")
        if self.hcor and self.any_categorical:
            print("Correlation : Heterogeneous correlations using polychoric/polyserial")
            print("** Variables of type {factor} are assumed to be ordinal **\n")
        else:
            print("Correlation : Pearson\n")

        bt_str = du.format_pval(self.btest["p.value"], dec + 1)
        print("Bartlett test")
        print("Null hyp. : variables are not correlated")
        print("Alt. hyp. : variables are correlated")
        print(
            f"Chi-square: {round(self.btest['chisq'], 2)} "
            f"df({int(self.btest['df'])}), p.value {bt_str}"
        )

        print(f"\nKMO test: {round(self.pre_kmo, dec)}")

        print("\nVariable collinearity:")
        coll = pl.DataFrame(
            {
                " ": self.vars,
                "Rsq": [round(float(x), dec) for x in self.pre_r2],
                "KMO": [round(float(x), dec) for x in self.msai],
            }
        )
        print(coll)

        print("\nFit measures:")
        ev = self.pre_eigen
        tot = ev.sum()
        fit = pl.DataFrame(
            {
                " ": [f"PC{i + 1}" for i in range(len(ev))],
                "Eigenvalues": [round(float(x), dec) for x in ev],
                "Variance %": [round(100 * float(x) / tot, dec) for x in ev],
                "Cumulative %": [
                    round(100 * float(x), dec) for x in np.cumsum(ev / tot)
                ],
            }
        )
        print(fit)

    def plot(self, plots=("scree", "change"), cutoff: float = 0.2, custom: bool = False):
        """Scree and/or eigenvalue-change plots (plotnine objects).

        Returns a single plot when one type is requested, otherwise a list.
        """
        from plotnine import (
            aes,
            geom_bar,
            geom_hline,
            geom_line,
            geom_point,
            ggplot,
            labs,
            scale_x_continuous,
            theme_classic,
        )

        import pandas as pd

        if isinstance(plots, str):
            plots = [plots]
        unknown = [p for p in plots if p not in ("scree", "change")]
        if unknown:
            raise ValueError(
                f"Unknown plot type(s): {', '.join(unknown)}. Use 'scree' and/or 'change'"
            )

        ev = self.pre_eigen[self.pre_eigen > cutoff]
        if "scree" in plots and len(ev) < 1:
            raise ValueError(f"No eigenvalues above the cutoff ({cutoff}) to plot")
        if "change" in plots and len(ev) < 2:
            raise ValueError(
                f"The change plot needs at least 2 eigenvalues above the cutoff ({cutoff})"
            )

        out = []
        for ptype in plots:
            if ptype == "scree":
                dat = pl.DataFrame(
                    {"x": list(range(1, len(ev) + 1)), "y": [float(v) for v in ev]}
                ).to_pandas()
                g = (
                    ggplot(dat, aes(x="x", y="y", group=1))
                    + geom_line(color="blue", linetype="dashdot", size=0.7)
                    + geom_point(color="blue", size=4, shape="o", fill="white")
                    + geom_hline(yintercept=1, color="black", size=0.5)
                    + labs(title="Screeplot", x="# factors", y="Eigenvalues")
                    + scale_x_continuous(breaks=list(range(1, len(ev) + 1)))
                    + theme_classic()
                )
                out.append(g)
            elif ptype == "change":
                change = (ev[1:] - ev[:-1]) / ev[:-1]
                change = change / change.min()
                # change[i] is the drop from eigenvalue i+1 to i+2 (Radiant's
                # "1-2", "2-3", ...), kept in that order on the x-axis
                labels = [f"{i}-{i + 1}" for i in range(1, len(change) + 1)]
                dat = pd.DataFrame(
                    {
                        "nr_fact": pd.Categorical(labels, categories=labels, ordered=True),
                        "bump": change.astype(float),
                    }
                )
                g = (
                    ggplot(dat, aes(x="nr_fact", y="bump"))
                    + geom_bar(stat="identity", alpha=0.5, fill="blue")
                    + labs(
                        title="Change in Eigenvalues",
                        x="# factors",
                        y="Rate of change index",
                    )
                    + theme_classic()
                )
                out.append(g)

        return as_plot_result(out)
