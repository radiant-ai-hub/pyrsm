from cmath import sqrt
from functools import lru_cache
from typing import Literal

import polars as pl

import pyrsm.basics.display_utils as du
from pyrsm.plot_utils import compose_plots
from pyrsm.utils import check_dataframe, ifelse


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
    def __init__(
        self,
        data: pl.DataFrame | dict[str, pl.DataFrame],
        var: str,
        probs: tuple[float, ...] | None = None,
        figsize: tuple[float, float] = None,
    ) -> None:
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name]
        else:
            self.data = data
            self.name = "Not provided"

        self.data = check_dataframe(self.data)
        self.var = var
        self.figsize = figsize
        self.probs = probs

        # Get value counts using polars
        counts = self.data.group_by(self.var).len().sort(self.var)
        self.freq = dict(zip(counts[self.var].to_list(), counts["len"].to_list()))
        self.nlev = len(self.freq)
        if self.probs is None:
            self.probs = [1 / self.nlev] * self.nlev

        # Build observed DataFrame using polars
        sorted_keys = sorted(self.freq.keys())
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
        dec: int = 3,
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
            in a Jupyter notebook, use styled table output.
        """
        output = ifelse(isinstance(output, str), [output], output)
        self._validate_and_print_header()

        if not plain and du.is_notebook():
            self._style_tables(output=output, dec=dec)
        else:
            self._summary_plain(output=output, dec=dec)

        self._summary_footer(dec=dec)

    def _validate_and_print_header(self) -> None:
        """Validate inputs and print the summary header."""
        print("Goodness of fit test")
        print(f"Data         : {self.name}")
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

    def _summary_plain(self, output: list[str], dec: int = 3) -> None:
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

    def _style_tables(self, output: list[str], dec: int = 3) -> None:
        """Display styled tables using great_tables in Jupyter."""
        from IPython.display import display

        if "observed" in output:
            gt = du.style_table(
                self.observed, title="Observed Frequencies", subtitle=""
            )
            display(gt)

        if "expected" in output:
            gt = du.style_table(
                self.expected, title="Expected Frequencies", subtitle="total × p"
            )
            display(gt)

        if "chisq" in output:
            gt = du.style_table(
                self.chisq,
                title="Chi-squared Contributions",
                subtitle="(observed - expected)² / expected",
            )
            display(gt)

        if "dev_std" in output:
            gt = du.style_table(
                self.stdev,
                title="Standardized Deviations",
                subtitle="(observed - expected) / √expected",
            )
            display(gt)

    def _summary_footer(self, dec: int = 3) -> None:
        """Print chi-squared test results."""
        sorted_keys = sorted(self.freq.keys())
        chisquare = _get_scipy_chisquare()
        chisq, p_val = chisquare(
            [self.freq[key] for key in sorted_keys],
            [self.expected[key].item() for key in sorted_keys],
        )

        p_val_str = du.format_pval(p_val, dec)
        print(
            f"\nChi-squared: {round(chisq, dec)} df ({self.nlev - 1}), p.value {p_val_str}"
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
        from plotnine import aes, geom_col, ggplot, ggtitle, labs, theme_bw

        pu = _get_plotting_utils()
        plots = ifelse(isinstance(plots, str), [plots], plots)

        def _reshape_for_plot(df, value_name):
            """Reshape wide dataframe to long format for plotting."""
            cols = [c for c in df.columns if c != "Total"]
            values = [df[c].item() for c in cols]
            return pl.DataFrame({self.var: cols, value_name: values})

        plot_list = []

        if "observed" in plots:
            data = _reshape_for_plot(self.observed, "count")
            p = (
                ggplot(data, aes(x=self.var, y="count"))
                + geom_col(fill=pu.PlotConfig.FILL, alpha=0.8)
                + labs(x=self.var, y="Count")
                + ggtitle("Observed frequencies")
                + theme_bw()
            )
            plot_list.append(p)

        if "expected" in plots:
            data = _reshape_for_plot(self.expected, "expected")
            p = (
                ggplot(data, aes(x=self.var, y="expected"))
                + geom_col(fill=pu.PlotConfig.FILL, alpha=0.8)
                + labs(x=self.var, y="Expected")
                + ggtitle("Expected frequencies")
                + theme_bw()
            )
            plot_list.append(p)

        if "chisq" in plots:
            data = _reshape_for_plot(self.chisq, "chisq")
            p = (
                ggplot(data, aes(x=self.var, y="chisq"))
                + geom_col(fill=pu.PlotConfig.FILL, alpha=0.8)
                + labs(x=self.var, y="Chi-squared")
                + ggtitle("Contribution to chi-squared")
                + theme_bw()
            )
            plot_list.append(p)

        if "dev_std" in plots:
            data = _reshape_for_plot(self.stdev, "stdev")
            p = (
                ggplot(data, aes(x=self.var, y="stdev"))
                + geom_col(fill=pu.PlotConfig.FILL, alpha=0.8)
                + labs(x=self.var, y="Std. Deviation")
                + ggtitle("Deviation standardized")
                + theme_bw()
            )
            # Add significance reference lines
            for line in pu.ReferenceLine.significance_levels():
                p = p + line
            plot_list.append(p)

        # Return single plot or first plot if multiple requested
        combined = compose_plots(plot_list)
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
