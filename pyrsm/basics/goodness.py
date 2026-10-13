from cmath import sqrt
from functools import lru_cache
from typing import Literal

import polars as pl

import pyrsm.basics.display_utils as du
from pyrsm.plot_utils import compose_plots
from pyrsm.utils import check_dataframe, ifelse
from pyrsm.data_scope import apply_scope, print_scope
from pyrsm.plot_style import ALPHA, FILL, theme_pyrsm


@lru_cache(maxsize=1)
def _get_scipy_chisquare():
    """Lazy load scipy.stats.chisquare."""
    from scipy.stats import chisquare

    return chisquare


@lru_cache(maxsize=1)
def _get_plotting_utils():
    """Lazy load plotting_utils."""
    import pyrsm.basics.plotting_utils as pu

    return pu


class goodness:
    """
    Chi-square goodness-of-fit test for a categorical variable.

    Parameters
    ----------
    data : pl.DataFrame | dict[str, pl.DataFrame]
        Input data as a Polars DataFrame or a dict with a single named DataFrame.
    var : str
        Categorical variable to test.
    probs : tuple[float, ...] | None
        Expected probabilities for each level, in level order: the category
        order of an Enum/Categorical variable (as R's factor levels), otherwise
        sorted. If None, uses a uniform distribution across levels.
    figsize : tuple[float, float] | None
        Figure size for plots.

    Attributes
    ----------
    observed : pl.DataFrame
        Observed counts with total.
    expected : pl.DataFrame
        Expected counts with total.
    chisq : pl.DataFrame
        Chi-square contributions with total.
    stdev : pl.DataFrame
        Standardized deviations without totals.
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

    Examples
    --------
    >>> import polars as pl
    >>> import pyrsm as rsm
    >>> df = pl.DataFrame({"resp": ["a", "a", "b", "b", "b"]})
    >>> gf = rsm.basics.goodness(df, var="resp")
    >>> print(gf.observed)
    shape: (1, 3)
    ┌─────┬─────┬───────┐
    │ a   ┆ b   ┆ Total │
    │ --- ┆ --- ┆ ---   │
    │ i64 ┆ i64 ┆ i32   │
    ╞═════╪═════╪═══════╡
    │ 2   ┆ 3   ┆ 5     │
    └─────┴─────┴───────┘
    >>> gf = rsm.basics.goodness(df, var="resp", probs=(0.4, 0.6))
    >>> gf.probs
    (0.4, 0.6)
    """

    def __init__(
        self,
        data: pl.DataFrame | dict[str, pl.DataFrame],
        var: str,
        probs: tuple[float, ...] | None = None,
        figsize: tuple[float, float] = None,
        data_filter: str = "",
        sort: str = "",
        slice: str = "",
    ) -> None:
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name]
        else:
            self.data = data
            self.name = "Not provided"

        self.data = check_dataframe(self.data)

        # The data scope, applied before anything is measured: filter, then
        # sort, then slice. Kept on the object so ``summary`` can say which
        # rows the numbers describe -- a reader who cannot see the filter has
        # no way to know the n is not the whole dataset.
        self.data_filter = data_filter
        self.sort = sort
        self.slice = slice
        self.data = apply_scope(self.data, data_filter, sort, slice)
        self.var = var
        self.figsize = figsize
        self.probs = probs

        # Get value counts using polars
        counts = self.data.group_by(self.var).len().sort(self.var)
        self.freq = dict(zip(counts[self.var].to_list(), counts["len"].to_list()))
        # Level order: sorting an Enum/Categorical keeps its category order, so
        # probs line up with the levels as declared (WS Journal before USA Today).
        self.levels = list(self.freq.keys())
        self.nlev = len(self.freq)
        if self.probs is None:
            self.probs = [1 / self.nlev] * self.nlev

        # Build observed DataFrame using polars
        sorted_keys = self.levels
        obs_data = {k: [self.freq[k]] for k in sorted_keys}
        self.observed = pl.DataFrame(obs_data)
        obs_total = sum(self.freq.values())
        self.observed = self.observed.with_columns(pl.lit(obs_total).alias("Total"))

        # Build expected DataFrame using polars
        exp_data = {
            sorted_keys[i]: [self.probs[i] * obs_total] for i in range(len(sorted_keys))
        }
        self.expected = pl.DataFrame(exp_data)
        exp_total = sum(exp_data[k][0] for k in sorted_keys)
        self.expected = self.expected.with_columns(pl.lit(exp_total).alias("Total"))

        # Build chisq DataFrame using polars
        chisq_data = {}
        for col in sorted_keys:
            obs_val = self.freq[col]
            exp_val = self.probs[sorted_keys.index(col)] * obs_total
            chisq_data[col] = [round(((obs_val - exp_val) ** 2) / exp_val, 2)]
        self.chisq = pl.DataFrame(chisq_data)
        chisq_total = sum(chisq_data[k][0] for k in sorted_keys)
        self.chisq = self.chisq.with_columns(pl.lit(chisq_total).alias("Total"))

        # Build stdev DataFrame using polars
        stdev_data = {}
        for col in sorted_keys:
            obs_val = self.freq[col]
            exp_val = self.probs[sorted_keys.index(col)] * obs_total
            stdev_data[col] = [round((obs_val - exp_val) / sqrt(exp_val).real, 2)]
        self.stdev = pl.DataFrame(stdev_data)

    def summary(
        self,
        output: list[str] = ["observed", "expected"],
        dec: int = 2,
        plain: bool = True,
    ) -> None:
        """
        Print summary of goodness of fit test.

        Parameters
        ----------
        output : list[str]
            Tables to display. Options: "observed", "expected", "chisq", "dev_std"
        dec : int
            Number of decimal places to display.
        plain : bool
            If True (default), print plain text output. If False and running
            in a notebook (or a rendered Quarto report), show the formatted
            report Radiant shows.
        """
        if not plain and du.is_notebook():
            # Radiant's formatted report (pyrsm.report), as its Summary tab shows it
            from pyrsm.basics.report import goodness_report

            return du.show_report(self, goodness_report(self, output=output, dec=dec), output=output, dec=dec)
        output = ifelse(isinstance(output, str), [output], output)
        self._validate_and_print_header()

        self._summary_plain(output=output, dec=dec)

        self._summary_footer(dec=dec)

    def _validate_and_print_header(self) -> None:
        """Validate inputs and print the summary header."""
        print("Goodness of fit test")
        print(f"Data         : {self.name}")
        print_scope(self, 13)
        if self.var not in self.data.columns:
            raise ValueError(f"{self.var} does not exist in chosen dataset")

        print(f"Variable     : {self.var}")
        if self.nlev != len(self.probs):
            raise ValueError(
                f'Number of elements in "probs" should match the number of levels in {self.var} ({self.nlev})'
            )

        if not 0.999 <= sum(self.probs) <= 1.001:
            raise ValueError("Probabilities do not sum to 1 ({sum(self.probs)})")

        print(f"Probabilities: {' '.join(map(str, self.probs))}")
        print(
            f"Null hyp.    : The distribution of {self.var} is consistent with the specified distribution"
        )
        print(
            f"Alt. hyp.    : The distribution of {self.var} is not consistent with the specified distribution"
        )

    def _summary_plain(self, output: list[str], dec: int = 2) -> None:
        """Print plain text tables."""
        with pl.Config(
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            fmt_str_lengths=100,
        ):
            if "observed" in output:
                print("\nObserved:")
                print(self.observed)

            if "expected" in output:
                print("\nExpected: total x p")
                print(self.expected)

            if "chisq" in output:
                print(
                    "\nContribution to chi-squared: (observed - expected) ^ 2 / expected"
                )
                print(self.chisq)

            if "dev_std" in output:
                print(
                    "\nDeviation standardized: (observed - expected) / sqrt(expected)\n"
                )
                print(self.stdev)

    def _summary_footer(self, dec: int = 2) -> None:
        """Print chi-squared test results."""
        sorted_keys = self.levels
        chisquare = _get_scipy_chisquare()
        chisq, p_val = chisquare(
            [self.freq[key] for key in sorted_keys],
            [self.expected[key].item() for key in sorted_keys],
        )

        # statistic and p-value with one more decimal than the tables, as radiant.basics
        p_val_str = du.format_pval(p_val, dec + 1)
        print(
            f"\nChi-squared: {round(chisq, dec + 1)} df ({self.nlev - 1}), p.value {p_val_str}"
        )

    def plot(
        self,
        plots: Literal["observed", "expected", "chisq", "dev_std"] = "observed",
    ):
        """
        Plot goodness of fit results.

        Parameters
        ----------
        plots : str
            The type of plot to create ('observed', 'expected', 'chisq', 'dev_std').

        Returns
        -------
        plotnine.ggplot
            A plotnine ggplot object.
        """
        from plotnine import aes, geom_col, ggplot, ggtitle, labs

        pu = _get_plotting_utils()
        plots = ifelse(isinstance(plots, str), [plots], plots)

        def _reshape_for_plot(df, value_name):
            """Reshape wide dataframe to long format for plotting."""
            cols = [c for c in df.columns if c != "Total"]
            values = [df[c].item() for c in cols]
            # An Enum keeps the bars in level order (plotnine sorts strings).
            return pl.DataFrame({self.var: pl.Series(cols, dtype=pl.Enum(cols)), value_name: values})

        plot_list = []

        if "observed" in plots:
            data = _reshape_for_plot(self.observed, "count")
            p = (
                ggplot(data, aes(x=self.var, y="count"))
                + geom_col(fill=FILL, alpha=ALPHA)
                + labs(x=self.var, y="Count")
                + ggtitle("Observed frequencies")
                + theme_pyrsm()
            )
            plot_list.append(p)

        if "expected" in plots:
            data = _reshape_for_plot(self.expected, "expected")
            p = (
                ggplot(data, aes(x=self.var, y="expected"))
                + geom_col(fill=FILL, alpha=ALPHA)
                + labs(x=self.var, y="Expected")
                + ggtitle("Expected frequencies")
                + theme_pyrsm()
            )
            plot_list.append(p)

        if "chisq" in plots:
            data = _reshape_for_plot(self.chisq, "chisq")
            p = (
                ggplot(data, aes(x=self.var, y="chisq"))
                + geom_col(fill=FILL, alpha=ALPHA)
                + labs(x=self.var, y="Chi-squared")
                + ggtitle("Contribution to chi-squared")
                + theme_pyrsm()
            )
            plot_list.append(p)

        if "dev_std" in plots:
            data = _reshape_for_plot(self.stdev, "stdev")
            p = (
                ggplot(data, aes(x=self.var, y="stdev"))
                + geom_col(fill=FILL, alpha=ALPHA)
                + labs(x=self.var, y="Std. Deviation")
                + ggtitle("Deviation standardized")
                + theme_pyrsm()
            )
            # Add significance reference lines
            for line in pu.ReferenceLine.significance_levels():
                p = p + line
            plot_list.append(p)

        # Return single plot or first plot if multiple requested
        combined = compose_plots(plot_list, ncol=1, width_per_col=7, height_per_row=4.5)  # stacked, as radiant.basics
        if combined is None:
            raise ValueError("No valid plot type specified.")
        return combined


if __name__ == "__main__":
    import pyrsm as rsm
    from pyrsm.radiant import utils as ru

    data_dct, descriptions_dct = ru.get_dfs(pkg="basics", name="newspaper")
    gf = rsm.basics.goodness(data=data_dct, var="Income", probs=[1 / 2, 1 / 2])
    gf.summary(output=["observed"])
    gf.plot(plots=["observed", "expected", "chisq", "dev_std"])
