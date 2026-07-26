from functools import lru_cache

import numpy as np
import polars as pl

import pyrsm.basics.display_utils as du
from pyrsm.utils import check_dataframe, sig_stars


@lru_cache(maxsize=1)
def _get_scipy_stats():
    """Lazy load scipy.stats."""
    from scipy import stats

    return stats


class correlation:
    """
    Calculate correlations between numeric variables in a Polars DataFrame.

    Parameters
    ----------
    data : pl.DataFrame | dict[str, pl.DataFrame]
        Input data with numeric columns.
    vars : list[str] | None
        Column names to include. If empty, all numeric columns are used.
    method : str
        Correlation method: ``"pearson"``, ``"spearman"``, ``"kendall"``,
        or ``"polychoric"``. Use ``"polychoric"`` for ordinal variables to
        estimate correlations between latent normal variables (equivalent
        to R's ``polycor::polychor``).

    Attributes
    ----------
    cr : np.ndarray
        Correlation matrix.
    cp : np.ndarray
        P-value matrix (not available for polychoric).
    cv : np.ndarray
        Covariance matrix (not available for polychoric).

    Examples
    --------
    >>> import polars as pl
    >>> import pyrsm as rsm
    >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    >>> cr = rsm.basics.correlation(df, vars=["x", "y"])
    >>> print(cr.cr)
    [[0. 1.]
     [1. 0.]]
    >>> cr = rsm.basics.correlation(df, vars=["x", "y"], method="spearman")
    >>> cr.method
    'spearman'
    """

    def __init__(
        self,
        data: pl.DataFrame | dict[str, pl.DataFrame],
        vars: list[str] | None = [],
        method: str = "pearson",
    ) -> None:
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name]
        else:
            self.data = data
            self.name = "Not provided"

        self.data = check_dataframe(self.data)
        self.vars = vars
        if len(self.vars) == 0:
            self.vars = [
                col for col, dtype in self.data.schema.items() if dtype.is_numeric()
            ]

        self.data = self.data.select(self.vars)
        self.method = method

        ncol = self.data.shape[1]
        cr = np.zeros([ncol, ncol])
        cp = cr.copy()
        cv = cr.copy()

        if method == "polychoric":
            # Heterogeneous correlations, matching radiant.basics'
            # polycor::hetcor: only factor x factor pairs use the (now
            # vectorized) polychoric estimator, numeric x factor pairs use the
            # fast closed-form polyserial, and numeric x numeric pairs use
            # Pearson. This avoids running continuous columns (which have
            # thousands of distinct "levels") through the polychoric optimizer.
            from pyrsm.utils import polychoric_corr, polyserial_corr

            codes, is_cat = [], []
            for col_name in self.vars:
                col = self.data[col_name]
                cat = col.dtype == pl.String or isinstance(
                    col.dtype, (pl.Categorical, pl.Enum)
                )
                if cat:
                    numeric = col.cast(pl.Categorical).to_physical().cast(pl.Float64)
                else:
                    numeric = col.cast(pl.Float64)
                codes.append(numeric.to_numpy())
                is_cat.append(cat)

            mat = np.column_stack(codes)
            cr = np.eye(ncol)
            for i in range(ncol - 1):
                for j in range(i + 1, ncol):
                    xi, xj = mat[:, i], mat[:, j]
                    ci, cj = is_cat[i], is_cat[j]
                    if ci and cj:
                        r = polychoric_corr(xi, xj)
                    elif ci:
                        r = polyserial_corr(xj, xi)
                    elif cj:
                        r = polyserial_corr(xi, xj)
                    else:
                        keep = ~(np.isnan(xi) | np.isnan(xj))
                        r = (
                            np.corrcoef(xi[keep], xj[keep])[0, 1]
                            if keep.sum() > 1
                            else np.nan
                        )
                    cr[i, j] = cr[j, i] = r
            # p-values and covariance are not available for polychoric
        else:
            # Vectorized matrix correlation with pairwise-complete handling and
            # p-values from the t-test approximation, matching radiant's
            # psych::corr.test + cov() instead of a Python pairwise loop (orders
            # of magnitude faster for wide/large data).
            import pandas as pd

            stats = _get_scipy_stats()
            pdf = self.data.to_pandas()
            cr = pdf.corr(method=method).to_numpy()
            cv = pdf.cov().to_numpy()
            notna = pdf.notna().to_numpy().astype(float)
            n_pair = notna.T @ notna  # pairwise-complete observation counts
            with np.errstate(divide="ignore", invalid="ignore"):
                dof = n_pair - 2.0
                tval = cr * np.sqrt(dof / (1.0 - cr**2))
                cp = 2.0 * stats.t.sf(np.abs(tval), dof)
            cp = np.where(np.isfinite(cp), cp, 0.0)
            np.fill_diagonal(cp, 0.0)

        self.cr = cr
        self.cp = cp
        self.cv = cv

    def summary(
        self, cov=False, cutoff: float = 0, dec: int = 2, plain: bool = True
    ) -> None:
        """
        Print correlations between numeric variables in a Polars dataframe

        Parameters
        ----------
        cov : bool
            Show the covariance matrix if set to True
        cutoff : float
            Only show correlations larger than a threshold in absolute value
        dec : int
            Number of decimal places to use in rounding
        plain : bool
            If True (default), print plain text output. If False and running
            in a Jupyter notebook, use styled table output.

        Examples
        --------
        import pyrsm as rsm
        salary = pl.read_parquet("https://github.com/radiant-ai-hub/pyrsm/raw/refs/heads/main/examples/data/basics/salary.parquet")
        cr = rsm.correlation(salary.select(["salary", "yrs.since.phd", "yrs.service"]))
        cr.summary()
        """
        self._summary_header()
        if len(self.vars) < 2:
            print("\n**Select two or more variables to calculate correlations**")
            return

        self._print_hypothesis_info(cutoff)

        if not plain and du.is_notebook():
            self._style_tables(cov=cov, cutoff=cutoff, dec=dec)
        else:
            self._summary_plain(cov=cov, cutoff=cutoff, dec=dec)

    def _summary_header(self) -> None:
        """Print the summary header."""
        prn = "Correlation\n"
        prn += f"Data     : {self.name}\n"
        prn += f"Method   : {self.method}"
        print(prn)

    def _print_hypothesis_info(self, cutoff: float) -> None:
        """Print hypothesis and variable information."""
        cn = list(self.data.columns)
        if len(cn) > 2:
            x, y = "x", "y"
        else:
            x, y = cn[0], cn[1]

        prn = f"Cutoff   : {cutoff}\n"
        prn += "Variables: " + ", ".join(cn) + "\n"
        prn += f"Null hyp.: variables {x} and {y} are not correlated\n"
        prn += f"Alt. hyp.: variables {x} and {y} are correlated\n"
        print(prn)

    def _build_matrix_displays(self, cutoff: float, dec: int):
        """Build correlation, p-value, and covariance display DataFrames."""
        ind = np.triu_indices(self.cr.shape[0])
        cn = list(self.data.columns)

        # Build correlation matrix
        crs_arr = self.cr.round(dec).astype(str)
        if cutoff > 0:
            crs_arr[np.abs(self.cr) < cutoff] = ""
        crs_arr[ind] = ""

        # Build p-value matrix
        cps_arr = self.cp.round(dec).astype(str)
        if cutoff > 0:
            cps_arr[np.abs(self.cr) < cutoff] = ""
        cps_arr[ind] = ""

        # Create polars DataFrames
        crs_display = pl.DataFrame(
            {cn[j]: crs_arr[1:, j] for j in range(len(cn) - 1)}
        ).with_columns(pl.Series("", cn[1:]).alias(""))
        crs_display = crs_display.select([""] + cn[:-1])

        cps_display = pl.DataFrame(
            {cn[j]: cps_arr[1:, j] for j in range(len(cn) - 1)}
        ).with_columns(pl.Series("", cn[1:]).alias(""))
        cps_display = cps_display.select([""] + cn[:-1])

        # Build covariance matrix
        cvs_arr = np.round(self.cv, dec)
        cvs_str = np.array([[f"{v:,}" for v in row] for row in cvs_arr])
        if cutoff > 0:
            cvs_str[np.abs(self.cr) < cutoff] = ""
        cvs_str[ind[0], ind[1]] = ""

        cvs_display = pl.DataFrame(
            {cn[j]: cvs_str[1:, j] for j in range(len(cn) - 1)}
        ).with_columns(pl.Series("", cn[1:]).alias(""))
        cvs_display = cvs_display.select([""] + cn[:-1])

        return crs_display, cps_display, cvs_display

    def _summary_plain(
        self, cov: bool = False, cutoff: float = 0, dec: int = 2
    ) -> None:
        """Print plain text tables."""
        crs_display, cps_display, cvs_display = self._build_matrix_displays(cutoff, dec)

        with pl.Config(
            tbl_rows=-1,
            tbl_cols=-1,
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            fmt_str_lengths=100,
        ):
            print("Correlation matrix:")
            print(crs_display)

            if self.method != "polychoric":
                print("\np.values:")
                print(cps_display)

                if cov:
                    print("\nCovariance matrix:")
                    print(cvs_display)
            elif cov:
                print("\nCovariance matrix:")
                print(
                    "Not available for the polychoric method. Heterogeneous "
                    "correlations are estimated from latent variables, so a "
                    "covariance matrix on the original scale is not defined."
                )

    def _style_tables(self, cov: bool = False, cutoff: float = 0, dec: int = 2) -> None:
        """Display styled tables using great_tables in Jupyter."""
        from IPython.display import display

        crs_display, cps_display, cvs_display = self._build_matrix_displays(cutoff, dec)

        gt1 = du.style_table(
            crs_display,
            title="Correlation Matrix",
            subtitle=f"Method: {self.method}",
        )
        display(gt1)

        if self.method != "polychoric":
            gt2 = du.style_table(
                cps_display,
                title="P-values",
                subtitle="",
            )
            display(gt2)

            if cov:
                gt3 = du.style_table(
                    cvs_display,
                    title="Covariance Matrix",
                    subtitle="",
                )
                display(gt3)
        elif cov:
            from IPython.display import Markdown

            display(
                Markdown(
                    "**Covariance Matrix:** Not available for the polychoric "
                    "method. Heterogeneous correlations are estimated from "
                    "latent variables, so a covariance matrix on the original "
                    "scale is not defined."
                )
            )

    def plot(self, nobs: int = 1000, dec: int = 2, figsize: tuple[float, float] = None):
        """
        Plot scatter matrix of correlations between numeric variables.

        Displays a matrix with:
        - Diagonal: variable names
        - Lower triangle: scatter plots with regression lines
        - Upper triangle: correlation coefficients with significance stars

        Parameters
        ----------
        nobs : int
            Number of observations to use for the scatter plots. The default
            value is 1,000. To use all observations in the plots, use nobs=-1
        dec : int
            Number of decimal places to use in rounding
        figsize : tuple
            A tuple that determines the figure size. If None, size is
            determined based on the number of numeric variables in the data

        Returns
        -------
        tuple
            (Figure, Axes) matplotlib objects

        Examples
        --------
        import pyrsm as rsm
        salary = pl.read_parquet("https://github.com/radiant-ai-hub/pyrsm/raw/refs/heads/main/examples/data/basics/salary.parquet")
        cr = rsm.correlation(salary.select(["salary", "yrs_since_phd", "yrs_service"]))
        cr.plot(figsize=(7, 7))
        """
        import matplotlib.pyplot as plt

        # Turn off interactive mode to prevent double display in Jupyter
        was_interactive = plt.isinteractive()
        plt.ioff()

        if figsize is None:
            figsize = (max(5, self.cr.shape[0]), max(self.cr.shape[0], 5))

        def cor_label(label, longest, ax_sub):
            ax_sub.axes.xaxis.set_visible(False)
            ax_sub.axes.yaxis.set_visible(False)
            fs = min(figsize[0], figsize[1])
            font = (80 * fs) / (len(longest) * self.cr.shape[0])
            ax_sub.text(
                0.5,
                0.5,
                label,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=font,
            )

        def cor_text(r, p, ax_sub, dec=2):
            if self.method == "polychoric" or np.isnan(p):
                p = 1  # polychoric correlations do not provide p-values
            p = round(p, dec)
            rt = round(r, dec)
            p_star = sig_stars([p])[0]

            fs = min(figsize[0], figsize[1])
            font = (50 * fs) / (len(str(rt)) * len(self.data.columns))
            font_star = (12 * fs) / len(self.data.columns)

            ax_sub.axes.xaxis.set_visible(False)
            ax_sub.axes.yaxis.set_visible(False)
            ax_sub.text(
                0.5,
                0.5,
                rt,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=font * abs(r),
            )
            ax_sub.text(
                0.8,
                0.8,
                p_star,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=font_star,
                color="blue",
            )

        def _as_str(arr):
            return np.array(["" if v is None else str(v) for v in arr], dtype=object)

        def _box_groups(cat_data, num_data):
            """Numeric values grouped by category level (order preserved)."""
            cats = _as_str(cat_data)
            levels = list(dict.fromkeys(cats))
            groups, kept = [], []
            for lev in levels:
                vals = num_data[cats == lev].astype(float)
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    groups.append(vals)
                    kept.append(lev)
            return kept, groups

        _box_style = dict(
            patch_artist=True,
            widths=0.6,
            boxprops=dict(facecolor="slateblue", alpha=0.4),
            medianprops=dict(color="blue"),
            flierprops=dict(marker="o", markersize=2, alpha=0.3),
        )

        def cor_plot(x_data, y_data, x_num, y_num, ax_sub, s_size):
            # Mixed-type panels, mirroring radiant.basics plot.correlation:
            #  numeric x numeric -> scatter + fitted line
            #  factor  x numeric -> boxplots
            #  factor  x factor  -> spineplot (proportional stacked bars)
            ax_sub.axes.xaxis.set_visible(False)
            ax_sub.axes.yaxis.set_visible(False)

            if x_num and y_num:
                x = x_data.astype(float)
                y = y_data.astype(float)
                mask = ~(np.isnan(x) | np.isnan(y))
                x, y = x[mask], y[mask]
                ax_sub.scatter(x, y, alpha=0.3, color="slateblue", s=s_size)
                if len(x) > 1:
                    coeffs = np.polyfit(x, y, 1)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    ax_sub.plot(x_line, np.polyval(coeffs, x_line), color="blue")
            elif y_num and not x_num:
                _, groups = _box_groups(x_data, y_data)
                if groups:
                    ax_sub.boxplot(groups, vert=True, **_box_style)
            elif x_num and not y_num:
                _, groups = _box_groups(y_data, x_data)
                if groups:
                    ax_sub.boxplot(groups, vert=False, **_box_style)
            else:
                # Both categorical: spineplot — bar width ∝ x-level share,
                # stacked height = conditional share of each y level.
                xc = _as_str(x_data)
                yc = _as_str(y_data)
                xlevels = list(dict.fromkeys(xc))
                ylevels = list(dict.fromkeys(yc))
                n = len(xc)
                shades = np.linspace(0.35, 0.85, max(len(ylevels), 1))
                left = 0.0
                for xl in xlevels:
                    sub = yc[xc == xl]
                    total = len(sub)
                    width = total / n if n else 0
                    bottom = 0.0
                    for yi, yl in enumerate(ylevels):
                        frac = (np.sum(sub == yl) / total) if total else 0
                        ax_sub.bar(
                            left, frac, width=width, bottom=bottom, align="edge",
                            color=plt.cm.Blues(shades[yi]),
                            edgecolor="white", linewidth=0.5,
                        )
                        bottom += frac
                    left += width
                ax_sub.set_xlim(0, 1)
                ax_sub.set_ylim(0, 1)

        # data = self.data.to_pandas()
        data = self.data
        cn = list(data.columns)
        ncol = len(cn)
        longest = max(cn, key=len)
        # Per-column type so the lower triangle can pick scatter / boxplot /
        # spineplot (mirrors radiant.basics handling of factor variables).
        is_num = {c: bool(data.schema[c].is_numeric()) for c in cn}

        fs = min(figsize[0], figsize[1])
        s_size = (5 * fs) / len(data.columns)

        fig, axes = plt.subplots(ncol, ncol, figsize=figsize)

        # Sample data if needed
        nrows = data.shape[0]
        if nobs < nrows and nobs != np.inf and nobs != -1:
            indices = np.random.choice(nrows, size=nobs, replace=False)
            data_np = {col: data[col].to_numpy()[indices] for col in cn}
        else:
            data_np = {col: data[col].to_numpy() for col in cn}

        for i in range(ncol):
            for j in range(ncol):
                if i == j:
                    cor_label(cn[i], longest, axes[i, j])
                elif i > j:
                    cor_plot(
                        data_np[cn[i]],
                        data_np[cn[j]],
                        is_num[cn[i]],
                        is_num[cn[j]],
                        axes[i, j],
                        s_size,
                    )
                else:
                    cor_text(self.cr[j, i], self.cp[j, i], axes[i, j], dec=dec)

        plt.subplots_adjust(wspace=0.04, hspace=0.04)

        if was_interactive:
            plt.ion()

        return fig
