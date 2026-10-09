from functools import lru_cache
from typing import Literal

import numpy as np
import polars as pl

import pyrsm.basics.display_utils as du
import pyrsm.basics.utils as bu
from pyrsm.utils import check_dataframe, ifelse, sig_stars
from pyrsm.data_scope import apply_scope, print_scope
from pyrsm.plot_utils import plot_each
from pyrsm.plot_style import COMP, FILL, REF, theme_pyrsm


@lru_cache(maxsize=1)
def _get_scipy_stats():
    """Lazy load scipy.stats."""
    from scipy import stats

    return stats


@lru_cache(maxsize=1)
def _get_plotting_utils():
    """Lazy load plotting_utils."""
    import pyrsm.basics.plotting_utils as pu

    return pu


class single_mean:
    """
    A class to perform single-mean hypothesis testing

    Attributes
    ----------
    data : pl.DataFrame
        The input data for the hypothesis test as a Polars DataFrame. If a dictionary is provided, the key should be the name of the dataframe.
    var : str
        The variable/column name to test.
    alt_hyp : str
        The alternative hypothesis ('two-sided', 'greater', 'less').
    conf : float
        The confidence level for the test.
    comp_value : float
        The comparison value for the test.
    t_val : float
        The t-statistic value.
    p_val : float
        The p-value of the test.
    ci : tuple
        The confidence interval of the test.
    mean : float
        The mean of the variable.
    n : int
        The number of observations.
    n_missing : int
        The number of missing observations.
    sd : float
        The standard deviation of the variable.
    se : float
        The standard error of the variable.
    me : float
        The margin of error.
    diff : float
        The difference between the mean and the comparison value.
    df : int
        The degrees of freedom.
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

    Methods
    -------
    __init__(data, var, alt_hyp='two-sided', conf=0.95, comp_value=0)
        Initializes the single_mean class with the provided data and parameters.
    summary(dec=3)
        Prints a summary of the hypothesis test.
    plot()
        Plots the results of the hypothesis test.

    Examples
    --------
    >>> import polars as pl
    >>> import pyrsm as rsm
    >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    >>> sm = rsm.basics.single_mean(df, var="x", comp_value=2)
    >>> float(sm.mean)
    2.0
    >>> sm = rsm.basics.single_mean(df, var="x", alt_hyp="greater", comp_value=1)
    >>> sm.alt_hyp
    'greater'
    """

    def __init__(
        self,
        data: pl.DataFrame | dict[str, pl.DataFrame],
        var: str,
        alt_hyp: str = "two-sided",
        conf: float = 0.95,
        comp_value: float = 0,
        data_filter: str = "",
        sort: str = "",
        slice: str = "",
    ):
        """
        Constructs all the necessary attributes for the single_mean object.

        Parameters
        ----------
        data : pl.DataFrame | dict[str, pl.DataFrame]
            The input data for the hypothesis test as a Polars DataFrame.
        var : str
            The variable/column name to test.
        alt_hyp : str, optional
            The alternative hypothesis ('two-sided', 'greater', 'less') (default is 'two-sided').
        conf : float, optional
            The confidence level for the test (default is 0.95).
        comp_value : float, optional
            The comparison value for the test (default is 0).
        data_filter : str, default ""
            Expression used to filter the data before the analysis, e.g.,
            "price > 5000 and cut in ['Ideal', 'Premium']". A bare name is a
            column and a call is an operation on one (``is_null(price)``). See
            ``pyrsm.data_scope`` for the full list.
        sort : str, default ""
            Column(s) to sort on, "-" in front for descending, e.g.,
            "clarity -price". Applied after the filter.
        slice : str, default ""
            Rows to keep, counting from 1, e.g., "1:100". Applied after the
            filter and the sort.
        """
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            incoming = data[self.name]
        else:
            incoming = data
            self.name = "Not provided"

        self.data = check_dataframe(incoming)

        # The data scope, applied before anything is measured: filter, then
        # sort, then slice. Kept on the object so ``summary`` can say which
        # rows the numbers describe -- a reader who cannot see the filter has
        # no way to know the n is not the whole dataset.
        self.data_filter = data_filter
        self.sort = sort
        self.slice = slice
        self.data = apply_scope(self.data, data_filter, sort, slice)
        self.var = var
        self.alt_hyp = alt_hyp
        self.conf = conf
        self.comp_value = comp_value

        series = self.data[self.var]
        values = series.to_numpy()

        stats = _get_scipy_stats()
        result = stats.ttest_1samp(
            a=values,
            popmean=self.comp_value,
            nan_policy="omit",
            alternative=self.alt_hyp,
        )

        self.t_val, self.p_val = result.statistic, result.pvalue
        self.ci = result.confidence_interval(confidence_level=conf)

        self.mean = np.nanmean(values)
        self.n = len(values)
        self.n_missing = int(series.null_count())

        self.sd = series.std(ddof=1)
        n_eff = self.n - self.n_missing
        self.se = self.sd / np.sqrt(n_eff) if n_eff > 0 else np.nan
        tscore = stats.t.ppf((1 + self.conf) / 2, n_eff - 1) if n_eff > 1 else np.nan

        self.me = (tscore * self.se).real
        self.diff = self.mean - self.comp_value
        self.df = n_eff - 1

    def summary(self, dec: int = 3, plain: bool = True) -> None:
        """
        Prints a summary of the hypothesis test.

        Parameters
        ----------
        dec : int, optional
            The number of decimal places to display (default is 3).
        plain : bool
            If True (default), print plain text output. If False and running
            in a Jupyter notebook, use styled table output.
        """
        display = du.SummaryDisplay(
            header_func=self._summary_header,
            plain_func=lambda extra, d: self._summary_plain(d),
            styled_func=lambda extra, d: self._style_tables(d),
        )
        display.display(extra=False, dec=dec, plain=plain)

    def _summary_header(self) -> None:
        """Print the summary header."""
        print("Single mean test")
        print(f"Data      : {self.name}")
        print_scope(self, 10)
        print(f"Variables : {self.var}")
        print(f"Confidence: {self.conf}")
        print(f"Comparison: {self.comp_value}\n")
        print(f"Null hyp. : the mean of {self.var} is equal to {self.comp_value}")

        if self.alt_hyp == "less":
            alt_hyp = "less than"
        elif self.alt_hyp == "two-sided":
            alt_hyp = "not equal to"
        else:
            alt_hyp = "greater than"

        print(f"Alt. hyp. : the mean of {self.var} is {alt_hyp} {self.comp_value}\n")

    def _summary_plain(self, dec: int = 3) -> None:
        """Print plain text tables."""
        cl = bu.ci_label(self.alt_hyp, self.conf, dec=dec)

        table1 = pl.DataFrame(
            {
                "mean": [round(self.mean, dec)],
                "n": [self.n],
                "n_missing": [self.n_missing],
                "sd": [round(self.sd, dec)],
                "se": [round(self.se, dec)],
                "me": [round(self.me, dec)],
            }
        )

        p_val_str = ifelse(self.p_val < 0.001, "< .001", round(self.p_val, dec))
        table2 = pl.DataFrame(
            {
                "diff": [round(self.diff, dec)],
                "se": [round(self.se, dec)],
                "t.value": [round(self.t_val, dec)],
                "p.value": [p_val_str],
                "df": [self.df],
                cl[0]: [round(self.ci[0], dec)],
                cl[1]: [round(self.ci[1], dec)],
                "": [sig_stars([self.p_val])[0]],
            }
        )

        du.print_plain_tables(table1, table2)

    def _style_tables(self, dec: int = 3) -> None:
        """Display styled tables using great_tables in Jupyter."""
        from IPython.display import display

        cl = bu.ci_label(self.alt_hyp, self.conf, dec=dec)

        table1 = pl.DataFrame(
            {
                "mean": [self.mean],
                "n": [self.n],
                "n_missing": [self.n_missing],
                "sd": [self.sd],
                "se": [self.se],
                "me": [self.me],
            }
        )

        table2 = pl.DataFrame(
            {
                "diff": [self.diff],
                "se": [self.se],
                "t.value": [self.t_val],
                "p.value": [du.format_pval(self.p_val, dec)],
                "df": [self.df],
                cl[0]: [self.ci[0]],
                cl[1]: [self.ci[1]],
                "": [sig_stars([self.p_val])[0]],
            }
        )

        gt1 = du.style_table(
            table1,
            title="Descriptive Statistics",
            subtitle=f"Variable: {self.var}",
            number_cols=["mean", "sd", "se", "me"],
            integer_cols=["n", "n_missing"],
            dec=dec,
        )

        gt2 = du.style_table(
            table2,
            title="Hypothesis Test",
            subtitle=f"Comparison value: {self.comp_value}",
            number_cols=["diff", "se", "t.value", cl[0], cl[1]],
            integer_cols=["df"],
            dec=dec,
        )

        display(gt1)
        display(gt2)

    def _simulate_means(self, nsim: int = 1000):
        """Simulate the sampling distribution of the mean under the null.

        Resamples the data with replacement ``nsim`` times, recentres the
        bootstrapped means on ``comp_value`` (so the simulation reflects the
        null hypothesis that the population mean equals ``comp_value``), and
        returns the simulated means together with the percentile cutoffs for
        the chosen alternative. Mirrors the ``"simulate"`` plot in
        radiant.basics ``plot.single_mean``.

        Parameters
        ----------
        nsim : int
            Number of bootstrap replications (default 1000).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The simulated means and the cutoff value(s) under the null.
        """
        values = (
            self.data.select(self.var).drop_nulls().to_series().to_numpy().astype(float)
        )
        nr = len(values)
        rng = np.random.default_rng()
        idx = rng.integers(0, nr, size=(nsim, nr))
        sims = values[idx].mean(axis=1)
        # Centre the bootstrap distribution on the comparison value so it
        # represents sampling variability *if the null hypothesis is true*.
        sims = (sims - sims.mean()) + self.comp_value
        if self.alt_hyp == "two-sided":
            probs = [(1 - self.conf) / 2, 1 - (1 - self.conf) / 2]
        elif self.alt_hyp == "less":
            probs = [1 - self.conf]
        else:  # greater
            probs = [self.conf]
        cutoffs = np.quantile(sims, probs)
        return sims, cutoffs

    def plot(
        self,
        plots: Literal["hist", "sim"] | list[str] = "hist",
        theme: Literal["modern", "publication", "minimal", "classic"] = "modern",
        backend: Literal["plotnine"] = "plotnine",
    ):
        """
        Plots the results of the hypothesis test. If the 'hist' is selected a histogram
        of the numeric variable will be shown. The solid black line in the histogram shows
        the sample mean. The dashed black lines show the confidence interval around the
        sample mean. The solid red line shows the comparison value (i.e., the value under
        the null-hypothesis). If the red line does not fall within the confidence interval
        we can reject the null-hypothesis in favor of the alternative at the specified
        confidence level (e.g., 0.95).

        Parameters
        ----------
        plots : str | list[str]
            The plot to generate, 'hist' (default) or 'sim', or a list such as
            ``["hist", "sim"]`` to show both in one figure.
        theme : str
            Kept for backward compatibility; every pyrsm plot uses the one
            pyrsm style (``pyrsm.plot_style.theme_pyrsm``).
        backend : str
            Kept for backward compatibility; pyrsm plots are always plotnine.
            ``"plotly"`` is no longer supported and falls back to plotnine
            with a warning.

        Returns
        -------
        plotnine.ggplot
            The plot object.
        """
        if backend != "plotnine":
            import warnings

            warnings.warn(
                f"backend={backend!r} is not supported; pyrsm plots are plotnine only.",
                UserWarning,
                stacklevel=2,
            )

        combined = plot_each(lambda p: self.plot(plots=p, theme=theme, backend=backend), plots)
        if combined is not None:
            return combined

        if plots == "hist":
            from plotnine import aes, geom_histogram, ggplot, labs

            pu = _get_plotting_utils()

            # Prepare data - convert polars to pandas for plotnine
            data = self.data.select(self.var).drop_nulls().to_pandas()

            # Create plotnine histogram
            p = (
                ggplot(data, aes(x=self.var))
                + geom_histogram(bins=10, fill=FILL)
                + labs(x="", y="Frequency")
                + theme_pyrsm()
            )

            # Add reference lines
            ref_lines = pu.ReferenceLine.ci_vlines(
                self.mean, self.ci[0], self.ci[1], self.comp_value
            )
            for line in ref_lines:
                p = p + line

            return p

        elif plots == "sim":
            import pandas as pd
            from plotnine import aes, geom_histogram, ggplot, labs

            pu = _get_plotting_utils()

            sims, cutoffs = self._simulate_means()
            data = pd.DataFrame({self.var: sims})
            spread = float(sims.max() - sims.min())
            bw = spread / 20 if spread > 0 else None

            hist = (
                geom_histogram(binwidth=bw, fill=FILL)
                if bw
                else geom_histogram(bins=20, fill=FILL)
            )
            p = (
                ggplot(data, aes(x=self.var))
                + hist
                + labs(
                    x="",
                    y="Frequency",
                    title=f"Simulated means if null hyp. is true ({self.var})",
                )
                + theme_pyrsm()
            )

            # red solid = comparison value (H0); black solid = observed mean;
            # red dashed = simulated cutoffs under the null.
            p = p + pu.ReferenceLine.vline(self.comp_value, color=COMP, linetype="solid")
            p = p + pu.ReferenceLine.vline(float(self.mean), color=REF, linetype="solid")
            for cut in cutoffs:
                p = p + pu.ReferenceLine.vline(float(cut), color=COMP, linetype="dashed")
            return p
        else:
            print("Invalid plot type")
            return None


# Ensure package attribute points to the class (not the module) when imported directly
import sys as _sys

_parent = _sys.modules.get(__name__.rsplit(".", 1)[0])
if _parent is not None:
    _parent.__dict__["single_mean"] = single_mean
