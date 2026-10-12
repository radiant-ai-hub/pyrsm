from functools import lru_cache

import numpy as np
import polars as pl

import pyrsm.basics.display_utils as du
from pyrsm.utils import check_dataframe, sig_stars
from pyrsm.data_scope import apply_scope, scope_lines


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
    hcor_se : bool
        With ``method="polychoric"``: also compute standard errors and the
        adjusted p-values from them, as radiant.basics' "Calculate adjusted
        p.values" (``polycor::hetcor(std.err = TRUE)``; z = r / se, two-sided).
        As in R, numeric x categorical correlations are then re-optimized,
        so they can differ slightly from the estimates without it.

    Attributes
    ----------
    cr : np.ndarray
        Correlation matrix.
    cp : np.ndarray
        P-value matrix (polychoric: only with ``hcor_se=True``).
    cv : np.ndarray
        Covariance matrix (not available for polychoric).
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
        hcor_se: bool = False,
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
        self.vars = vars
        if len(self.vars) == 0:
            self.vars = [
                col for col, dtype in self.data.schema.items() if dtype.is_numeric()
            ]

        self.data = self.data.select(self.vars)
        self.method = method
        self.hcor_se = bool(hcor_se) and method == "polychoric"

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
            from pyrsm.utils import (
                pearson_se,
                polychoric_corr,
                polychoric_corr_se,
                polyserial_corr,
                polyserial_corr_se,
            )

            se_mat = np.full([ncol, ncol], np.nan)

            codes, is_cat = [], []
            for col_name in self.vars:
                col = self.data[col_name]
                cat = col.dtype == pl.String or isinstance(
                    col.dtype, (pl.Categorical, pl.Enum)
                )
                if isinstance(col.dtype, pl.Enum):
                    # the level order (I1 < SI2 < ... < IF), as R's factor codes
                    numeric = col.to_physical().cast(pl.Float64)
                elif cat:
                    # no level order: alphabetical, as R's factor() would set it
                    levels = sorted(col.drop_nulls().cast(pl.Utf8).unique().to_list())
                    numeric = (
                        col.cast(pl.Utf8)
                        .replace_strict(levels, list(range(len(levels))), default=None)
                        .cast(pl.Float64)
                    )
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
                    se = np.nan
                    if ci and cj:
                        if self.hcor_se:
                            r, se = polychoric_corr_se(xi, xj)
                        else:
                            r = polychoric_corr(xi, xj)
                    elif ci or cj:
                        num, cat = (xj, xi) if ci else (xi, xj)
                        if self.hcor_se:
                            r, se = polyserial_corr_se(num, cat)
                        else:
                            r = polyserial_corr(num, cat)
                    else:
                        keep = ~(np.isnan(xi) | np.isnan(xj))
                        r = (
                            np.corrcoef(xi[keep], xj[keep])[0, 1]
                            if keep.sum() > 1
                            else np.nan
                        )
                        if self.hcor_se and keep.sum() > 3:
                            se = pearson_se(r, int(keep.sum()))
                    cr[i, j] = cr[j, i] = r
                    se_mat[i, j] = se_mat[j, i] = se
            if self.hcor_se:
                # radiant.basics: z = r / se, two-sided normal p-values
                stats = _get_scipy_stats()
                with np.errstate(divide="ignore", invalid="ignore"):
                    cp = 2.0 * stats.norm.sf(np.abs(cr / se_mat))
                np.fill_diagonal(cp, 0.0)
            self.se = se_mat
            # covariance is not available for polychoric
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

        if not plain and du.is_notebook() and du.has_great_tables():
            self._style_tables(cov=cov, cutoff=cutoff, dec=dec)
        else:
            self._summary_plain(cov=cov, cutoff=cutoff, dec=dec)

    def _summary_header(self) -> None:
        """Print the summary header."""
        prn = "Correlation\n"
        prn += f"Data     : {self.name}\n"
        for line in scope_lines(self, 9):
            prn += f"{line}\n"
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

            if self.method != "polychoric" or self.hcor_se:
                print("\np.values:")
                print(cps_display)

            if cov and self.method != "polychoric":
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

        if self.method != "polychoric" or self.hcor_se:
            gt2 = du.style_table(
                cps_display,
                title="P-values",
                subtitle="",
            )
            display(gt2)

        if cov and self.method != "polychoric":
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

    def cor2df(self, labels: tuple[str, str] = ("label1", "label2")) -> pl.DataFrame:
        """
        The correlations as a long table: one row per pair of variables, as
        radiant.basics ``cor2df`` ("Store as data.frame").

        Columns are the two variable names, ``correlation`` and ``distance``
        (``0.5 * (1 - correlation)``), pairs in the order R's ``combn`` lists
        them (lower triangle by column).

        Parameters
        ----------
        labels : tuple[str, str]
            Names for the two variable-name columns.
        """
        names = list(self.data.columns)
        rows = []
        for j in range(len(names) - 1):
            for i in range(j + 1, len(names)):
                r = float(self.cr[i, j])
                rows.append({labels[0]: names[j], labels[1]: names[i],
                             "correlation": r, "distance": 0.5 * (1 - r)})
        return pl.DataFrame(rows, schema={labels[0]: pl.Utf8, labels[1]: pl.Utf8,
                                          "correlation": pl.Float64, "distance": pl.Float64})

    def plot(self, nobs: int = 1000, dec: int = 2, figsize: tuple[float, float] = None):
        """
        Plot a correlation matrix (plotnine, pyrsm plot style).

        A grid with one row and one column per variable (names in the panel
        strips), mirroring radiant.basics ``plot.correlation``:

        - Diagonal: distribution of the variable (histogram, or bars for a
          categorical variable)
        - Lower triangle: scatter plot with a fitted line (numeric x numeric),
          boxplots (categorical x numeric) or a spineplot (categorical x
          categorical)
        - Upper triangle: correlation coefficient, sized by its absolute
          value, with significance stars

        Parameters
        ----------
        nobs : int
            Number of observations to use for the scatter plots. The default
            value is 1,000. To use all observations in the plots, use nobs=-1
        dec : int
            Number of decimal places to use in rounding
        figsize : tuple
            A tuple that determines the figure size. If None, size is
            determined based on the number of variables in the data

        Returns
        -------
        plotnine.ggplot

        Examples
        --------
        import pyrsm as rsm
        salary = pl.read_parquet("https://github.com/radiant-ai-hub/pyrsm/raw/refs/heads/main/examples/data/basics/salary.parquet")
        cr = rsm.correlation(salary.select(["salary", "yrs_since_phd", "yrs_service"]))
        cr.plot(figsize=(7, 7))
        """
        import pandas as pd
        from plotnine import (
            aes,
            element_blank,
            element_text,
            facet_grid,
            geom_blank,
            geom_point,
            geom_rect,
            geom_segment,
            geom_text,
            ggplot,
            labs,
            scale_alpha_identity,
            scale_size_identity,
            theme,
        )

        from pyrsm.plot_style import ALPHA, FILL, LINE, REF, theme_pyrsm

        data = self.data
        cn = list(data.columns)
        ncol = len(cn)
        if figsize is None:
            figsize = (max(5, ncol), max(ncol, 5))

        # Sample data if needed (correlations themselves use all rows)
        nrows = data.shape[0]
        if nobs is not None and nobs < nrows and nobs != np.inf and nobs != -1:
            indices = np.random.choice(nrows, size=nobs, replace=False)
        else:
            indices = None

        # Every panel is drawn on continuous positions: a numeric variable
        # uses its values, a categorical one the codes 1..k of its levels.
        # facet_grid(scales="free") then shares the x range down a column and
        # the y range across a row, so all panels line up like a pairs plot.
        is_num = {c: bool(data.schema[c].is_numeric()) for c in cn}
        pos, rng, levels = {}, {}, {}
        for c in cn:
            col = data[c]
            if is_num[c]:
                v = col.cast(pl.Float64).to_numpy()
                if indices is not None:
                    v = v[indices]
                pos[c] = v
                ok = v[~np.isnan(v)]
                lo, hi = (float(ok.min()), float(ok.max())) if ok.size else (0.0, 1.0)
                if hi <= lo:
                    lo, hi = lo - 0.5, hi + 0.5
                rng[c] = (lo, hi)
            else:
                if isinstance(col.dtype, pl.Enum):
                    levs = [str(x) for x in col.dtype.categories.to_list()]
                else:
                    levs = [
                        str(x)
                        for x in col.drop_nulls().cast(pl.Utf8).unique(maintain_order=True).to_list()
                    ]
                code = {lev: i + 1 for i, lev in enumerate(levs)}
                raw = col.cast(pl.Utf8).to_list()
                v = np.array(
                    [np.nan if x is None else code.get(x, np.nan) for x in raw], dtype=float
                )
                if indices is not None:
                    v = v[indices]
                pos[c] = v
                levels[c] = levs
                rng[c] = (0.5, max(len(levs), 1) + 0.5)

        def panel(df, row, col):
            df["row_var"] = row
            df["col_var"] = col
            return df

        def box_stats(vals):
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            iqr = q3 - q1
            lo_w = vals[vals >= q1 - 1.5 * iqr].min()
            hi_w = vals[vals <= q3 + 1.5 * iqr].max()
            out = vals[(vals < lo_w) | (vals > hi_w)]
            return q1, med, q3, lo_w, hi_w, out

        blank, rects, segs, points, fits, texts, stars = [], [], [], [], [], [], []
        # text size (points) that makes "0.00" about half a panel wide
        size_base = 0.9 * 72 * min(figsize) / ncol / 4

        for i, rv in enumerate(cn):  # rows: y = rv
            ylo, yhi = rng[rv]
            for j, cv in enumerate(cn):  # columns: x = cv
                xlo, xhi = rng[cv]
                blank.append(panel(pd.DataFrame({"x": [xlo, xhi], "y": [ylo, yhi]}), rv, cv))
                if i == j:
                    # distribution of the variable, scaled to the row's range
                    v = pos[rv][~np.isnan(pos[rv])]
                    if v.size == 0:
                        continue
                    if is_num[rv]:
                        counts, edges = np.histogram(v, bins=min(20, max(5, int(np.sqrt(v.size)))))
                        x0, x1 = edges[:-1], edges[1:]
                    else:
                        k = len(levels[rv])
                        counts = np.array([np.sum(v == c) for c in range(1, k + 1)])
                        x0 = np.arange(1, k + 1) - 0.4
                        x1 = np.arange(1, k + 1) + 0.4
                    top = ylo + 0.95 * (yhi - ylo) * counts / max(counts.max(), 1)
                    rects.append(
                        panel(
                            pd.DataFrame(
                                {"xmin": x0, "xmax": x1, "ymin": ylo, "ymax": top, "alpha": ALPHA}
                            ),
                            rv,
                            cv,
                        )
                    )
                elif i > j:
                    x, y = pos[cv], pos[rv]
                    keep = ~(np.isnan(x) | np.isnan(y))
                    x, y = x[keep], y[keep]
                    if x.size == 0:
                        continue
                    if is_num[cv] and is_num[rv]:
                        points.append(panel(pd.DataFrame({"x": x, "y": y}), rv, cv))
                        if x.size > 1 and np.ptp(x) > 0:
                            b1, b0 = np.polyfit(x, y, 1)
                            fits.append(
                                panel(
                                    pd.DataFrame(
                                        {
                                            "x": [x.min()],
                                            "y": [b0 + b1 * x.min()],
                                            "xend": [x.max()],
                                            "yend": [b0 + b1 * x.max()],
                                        }
                                    ),
                                    rv,
                                    cv,
                                )
                            )
                    elif is_num[cv] != is_num[rv]:
                        # boxplots of the numeric variable per level of the other
                        vertical = not is_num[cv]  # categories along x
                        cat, num = (x, y) if vertical else (y, x)
                        for lev in np.unique(cat):
                            vals = num[cat == lev]
                            q1, med, q3, lo_w, hi_w, out = box_stats(vals)
                            if vertical:
                                r = dict(xmin=lev - 0.3, xmax=lev + 0.3, ymin=q1, ymax=q3)
                                sg = [
                                    (lev - 0.3, med, lev + 0.3, med),
                                    (lev, q3, lev, hi_w),
                                    (lev, lo_w, lev, q1),
                                ]
                                pts = pd.DataFrame({"x": lev, "y": out})
                            else:
                                r = dict(xmin=q1, xmax=q3, ymin=lev - 0.3, ymax=lev + 0.3)
                                sg = [
                                    (med, lev - 0.3, med, lev + 0.3),
                                    (q3, lev, hi_w, lev),
                                    (lo_w, lev, q1, lev),
                                ]
                                pts = pd.DataFrame({"x": out, "y": lev})
                            rects.append(panel(pd.DataFrame({**{k: [v] for k, v in r.items()}, "alpha": [ALPHA]}), rv, cv))
                            segs.append(panel(pd.DataFrame(sg, columns=["x", "y", "xend", "yend"]), rv, cv))
                            if len(pts):
                                points.append(panel(pts, rv, cv))
                    else:
                        # spineplot: column width = share of the x level, stacked
                        # height = share of each y level within it
                        kx, ky = len(levels[cv]), len(levels[rv])
                        left = 0.5
                        for xl in range(1, kx + 1):
                            sub = y[x == xl]
                            width = kx * sub.size / x.size
                            bottom = 0.5
                            for yl in range(1, ky + 1):
                                h = ky * (np.sum(sub == yl) / sub.size) if sub.size else 0.0
                                rects.append(
                                    panel(
                                        pd.DataFrame(
                                            {
                                                "xmin": [left],
                                                "xmax": [left + width],
                                                "ymin": [bottom],
                                                "ymax": [bottom + h],
                                                "alpha": [0.25 + 0.6 * yl / ky],
                                            }
                                        ),
                                        rv,
                                        cv,
                                    )
                                )
                                bottom += h
                            left += width
                else:
                    r = float(self.cr[j, i])
                    p = self.cp[j, i] if self.cp is not None else np.nan
                    if self.method == "polychoric" or p is None or np.isnan(p):
                        p = 1  # polychoric correlations do not provide p-values
                    rt = f"{round(r, dec):.{dec}f}" if np.isfinite(r) else "NA"
                    ar = abs(r) if np.isfinite(r) else 0.0
                    texts.append(
                        panel(
                            pd.DataFrame(
                                {
                                    "x": [(xlo + xhi) / 2],
                                    "y": [(ylo + yhi) / 2],
                                    "label": [rt],
                                    "size": [max(6.0, size_base * ar * 4 / max(len(rt), 4))],
                                }
                            ),
                            rv,
                            cv,
                        )
                    )
                    star = sig_stars([round(p, dec)])[0]
                    if star.strip():
                        stars.append(
                            panel(
                                pd.DataFrame(
                                    {
                                        "x": [xlo + 0.85 * (xhi - xlo)],
                                        "y": [ylo + 0.85 * (yhi - ylo)],
                                        "label": [star],
                                        "size": [max(7.0, size_base * 0.7)],
                                    }
                                ),
                                rv,
                                cv,
                            )
                        )

        def frame(parts):
            if not parts:
                return None
            df = pd.concat(parts, ignore_index=True)
            df["row_var"] = pd.Categorical(df["row_var"], categories=cn, ordered=True)
            df["col_var"] = pd.Categorical(df["col_var"], categories=cn, ordered=True)
            return df

        g = ggplot(frame(blank), aes(x="x", y="y")) + geom_blank()
        if (df := frame(rects)) is not None:
            g = g + geom_rect(
                df,
                aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax", alpha="alpha"),
                inherit_aes=False,
                fill=FILL,
                color="white",
                size=0.2,
            )
        if (df := frame(points)) is not None:
            g = g + geom_point(df, alpha=ALPHA, color=FILL, size=max(0.5, 6 / ncol), stroke=0)
        if (df := frame(fits)) is not None:
            g = g + geom_segment(df, aes(xend="xend", yend="yend"), color=REF, size=0.6)
        if (df := frame(segs)) is not None:
            g = g + geom_segment(df, aes(xend="xend", yend="yend"), color=REF, size=0.4)
        if (df := frame(texts)) is not None:
            g = g + geom_text(df, aes(label="label", size="size"))
        if (df := frame(stars)) is not None:
            g = g + geom_text(df, aes(label="label", size="size"), color=LINE)

        return (
            g
            + facet_grid("row_var ~ col_var", scales="free")
            + scale_size_identity()
            + scale_alpha_identity()
            + labs(x="", y="")
            + theme_pyrsm()
            + theme(
                figure_size=figsize,
                axis_text=element_blank(),
                axis_ticks=element_blank(),
                panel_grid=element_blank(),
                panel_spacing=0.01,
                strip_text=element_text(size=9),
            )
        )
