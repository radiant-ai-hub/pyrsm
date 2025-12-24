"""
distr - Distribution analysis for DataFrames.

Provides summary statistics and plots for numeric, categorical, and other variable types.

Examples:
    import pyrsm as rsm

    diamonds = pl.read_parquet("https://github.com/radiant-ai-hub/pyrsm/raw/refs/heads/main/examples/data/data/diamonds.parquet")

    # Basic usage
    d = rsm.distr(diamonds)
    d.summary()
    d.plot()

    # With grouping
    d = rsm.distr(diamonds, by="cut")
    d.summary()

    # Specific columns
    d = rsm.distr(diamonds, cols=["price", "cut", "color"])
    d.summary()
    d.plot()
"""

from math import ceil

import polars as pl
from plotnine import (
    aes,
    element_text,
    geom_bar,
    geom_histogram,
    ggplot,
    ggtitle,
    labs,
    scale_x_discrete,
    theme,
    theme_bw,
)

from pyrsm.eda.explore import explore


def _is_categorical(df: pl.DataFrame, col: str, nint: int = 25) -> bool:
    """Check if column should be treated as categorical."""
    dtype = df.schema.get(col)
    if dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Enum):
        return True
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        n_unique = df.select(pl.col(col).n_unique()).item()
        return n_unique < nint
    return False


def _compose_plots(plot_list: list, ncol: int = 2):
    """Compose multiple plotnine plots into a grid layout."""
    if len(plot_list) == 0:
        return None
    if len(plot_list) == 1:
        return plot_list[0]

    nrow = ceil(len(plot_list) / ncol)

    # Build rows of plots
    rows = []
    for i in range(nrow):
        start = i * ncol
        end = min(start + ncol, len(plot_list))
        row_plots = plot_list[start:end]

        if len(row_plots) == 1:
            row = row_plots[0]
        else:
            row = row_plots[0]
            for p in row_plots[1:]:
                row = row | p
        rows.append(row)

    # Stack rows vertically
    if len(rows) == 1:
        combined = rows[0]
    else:
        combined = rows[0]
        for row in rows[1:]:
            combined = combined / row

    # Auto-adjust figure size
    height_per_row = 3
    width_per_col = 4
    fig_width = width_per_col * min(ncol, len(plot_list))
    fig_height = height_per_row * nrow
    combined = combined + theme(figure_size=(fig_width, fig_height))

    return combined


def _is_notebook() -> bool:
    """Detect if running in a Jupyter/IPython notebook environment."""
    try:
        from IPython import get_ipython

        ipy = get_ipython()
        if ipy is not None and "IPKernelApp" in ipy.config:
            return True
    except (ImportError, AttributeError):
        pass
    return False


class distr:
    """
    Distribution analysis for DataFrames.

    Provides summary statistics and distribution plots for numeric, categorical,
    and other variable types.

    Parameters
    ----------
    data : pl.DataFrame or pd.DataFrame
        Input data
    cols : list, optional
        Column names to analyze. If None, all columns are used
    by : str, optional
        Column name to group by for summary statistics
    name : str
        Dataset name for display
    nint : int, default 25
        Number of unique values below which integers are treated as categorical

    Examples
    --------
    import pyrsm as rsm
    diamonds = pl.read_parquet("https://github.com/radiant-ai-hub/pyrsm/raw/refs/heads/main/examples/data/data/diamonds.parquet")
    d = rsm.distr(diamonds)
    d.summary()
    d.plot()
    """

    def __init__(
        self,
        data,
        cols: list | None = None,
        by: str | None = None,
        name: str = "Not provided",
        nint: int = 25,
    ):
        # Convert pandas to polars if needed
        if not isinstance(data, pl.DataFrame):
            import pandas as pd

            if isinstance(data, pd.DataFrame):
                data = pl.from_pandas(data)
            else:
                data = pl.DataFrame(data)

        self.data = data
        self.name = name
        self.by = by
        self.nint = nint

        # Determine columns to analyze
        if cols is None:
            cols = [c for c in data.columns if c != by]
        self.cols = cols

        # Classify columns by type
        self.numeric_cols = []
        self.categorical_cols = []
        self.other_cols = []

        for col in self.cols:
            dtype = data.schema.get(col)
            if dtype is None:
                continue
            if _is_categorical(data, col, nint):
                self.categorical_cols.append(col)
            elif dtype.is_numeric():
                self.numeric_cols.append(col)
            else:
                self.other_cols.append(col)

    def summary(self, dec: int = 3, plain: bool = True) -> None:
        """
        Print summary statistics for all variable types.

        Parameters
        ----------
        dec : int
            Number of decimal places to display
        plain : bool
            If True (default), print plain text output. If False and running
            in a Jupyter notebook, use styled table output.
        """
        self._summary_header()

        if not plain and _is_notebook():
            self._summary_styled(dec)
        else:
            self._summary_plain(dec)

    def _summary_header(self) -> None:
        """Print the summary header."""
        print("Distribution Analysis")
        print(f"Data    : {self.name}")
        if self.by:
            print(f"Group by: {self.by}")
        print(
            f"Columns : {len(self.cols)} ({len(self.numeric_cols)} numeric, "
            f"{len(self.categorical_cols)} categorical, {len(self.other_cols)} other)"
        )

    def _summary_plain(self, dec: int = 3) -> None:
        """Print plain text summary."""
        with pl.Config(
            tbl_rows=-1,
            tbl_cols=-1,
            fmt_str_lengths=100,
            tbl_width_chars=200,
            tbl_hide_dataframe_shape=True,
            tbl_hide_column_data_types=True,
        ):
            # Numeric variables
            if self.numeric_cols:
                print("\n--- Numeric Variables ---")
                numeric_stats = explore(self.data, cols=self.numeric_cols, by=self.by)
                # Round numeric columns
                float_cols = [
                    c
                    for c in numeric_stats.columns
                    if numeric_stats[c].dtype.is_float()
                ]
                if float_cols:
                    numeric_stats = numeric_stats.with_columns(
                        pl.col(float_cols).round(dec)
                    )
                print(numeric_stats)

            # Categorical variables
            if self.categorical_cols:
                print("\n--- Categorical Variables ---")
                for col in self.categorical_cols:
                    self._print_categorical_stats(col, dec)

            # Other variables (date, datetime, etc.)
            if self.other_cols:
                print("\n--- Other Variables ---")
                other_stats = self._compute_other_stats()
                print(other_stats)

    def _print_categorical_stats(self, col: str, dec: int = 3) -> None:
        """Print statistics for a single categorical variable."""
        if self.by:
            # Grouped categorical stats
            stats = (
                self.data.group_by([self.by, col])
                .agg(pl.len().alias("count"))
                .sort([self.by, "count"], descending=[False, True])
            )
            # Add proportion within each group
            stats = stats.with_columns(
                (pl.col("count") / pl.col("count").sum().over(self.by))
                .round(dec)
                .alias("proportion")
            )
            print(f"\n{col}:")
            print(stats)
        else:
            # Ungrouped categorical stats
            stats = (
                self.data.select(pl.col(col).value_counts(sort=True))
                .unnest(col)
                .rename({"count": "count"})
            )
            total = stats["count"].sum()
            stats = stats.with_columns(
                (pl.col("count") / total).round(dec).alias("proportion")
            )
            mode_val = stats[col][0]
            n_unique = len(stats)
            n_missing = self.data[col].null_count()

            print(
                f"\n{col} (n_unique: {n_unique}, mode: {mode_val}, n_missing: {n_missing}):"
            )
            print(stats)

    def _compute_other_stats(self) -> pl.DataFrame:
        """Compute statistics for other variable types (date, datetime, etc.)."""
        rows = []
        for col in self.other_cols:
            dtype = self.data.schema.get(col)
            row = {
                "variable": col,
                "type": str(dtype),
                "n_unique": self.data[col].n_unique(),
                "n_missing": self.data[col].null_count(),
                "min": str(self.data[col].min()),
                "max": str(self.data[col].max()),
            }
            rows.append(row)
        return pl.DataFrame(rows)

    def _summary_styled(self, dec: int = 3) -> None:
        """Display styled tables using great_tables in Jupyter."""
        from IPython.display import display

        # Numeric variables
        if self.numeric_cols:
            numeric_stats = explore(self.data, cols=self.numeric_cols, by=self.by)
            float_cols = [
                c for c in numeric_stats.columns if numeric_stats[c].dtype.is_float()
            ]
            if float_cols:
                numeric_stats = numeric_stats.with_columns(
                    pl.col(float_cols).round(dec)
                )
            gt = numeric_stats.style.tab_header(
                title="Numeric Variables",
                subtitle=f"Grouped by: {self.by}" if self.by else "",
            ).tab_options(table_margin_left="0px")
            display(gt)

        # Categorical variables
        if self.categorical_cols:
            for col in self.categorical_cols:
                if self.by:
                    stats = (
                        self.data.group_by([self.by, col])
                        .agg(pl.len().alias("count"))
                        .sort([self.by, "count"], descending=[False, True])
                    )
                    stats = stats.with_columns(
                        (pl.col("count") / pl.col("count").sum().over(self.by))
                        .round(dec)
                        .alias("proportion")
                    )
                else:
                    stats = self.data.select(
                        pl.col(col).value_counts(sort=True)
                    ).unnest(col)
                    total = stats["count"].sum()
                    stats = stats.with_columns(
                        (pl.col("count") / total).round(dec).alias("proportion")
                    )

                mode_val = stats[col][0]
                n_unique = self.data[col].n_unique()
                n_missing = self.data[col].null_count()

                gt = stats.style.tab_header(
                    title=f"Categorical: {col}",
                    subtitle=f"n_unique: {n_unique}, mode: {mode_val}, n_missing: {n_missing}",
                ).tab_options(table_margin_left="0px")
                display(gt)

        # Other variables
        if self.other_cols:
            other_stats = self._compute_other_stats()
            gt = other_stats.style.tab_header(
                title="Other Variables", subtitle="Date, Datetime, and other types"
            ).tab_options(table_margin_left="0px")
            display(gt)

    def plot(self, cols: list | None = None, bins: int = 25, ncol: int = 2):
        """
        Create distribution plots for variables.

        Parameters
        ----------
        cols : list, optional
            Column names to plot. If None, uses columns from initialization
        bins : int, default 25
            Number of bins for histograms
        ncol : int, default 2
            Number of columns in the plot grid

        Returns
        -------
        plotnine.ggplot or plotnine.composition.Compose
            Single plot or combined plot composition
        """
        if cols is None:
            cols = self.numeric_cols + self.categorical_cols

        if not cols:
            print("No plottable columns found")
            return None

        # Create individual plots for each column
        plot_list = []
        for c in cols:
            if c not in self.data.columns:
                print(f"Column {c} not found in data")
                continue

            is_cat = _is_categorical(self.data, c, self.nint)

            if is_cat:
                # Bar plot for categorical - preserve Enum ordering
                col_dtype = self.data[c].dtype
                col_data = self.data.select(pl.col(c).cast(pl.Utf8).alias("value"))

                p = (
                    ggplot(col_data, aes(x="value"))
                    + geom_bar(fill="slateblue", alpha=0.8)
                    + labs(x="", y="Count")
                    + ggtitle(c)
                    + theme_bw()
                    + theme(
                        plot_title=element_text(size=10, weight="bold"),
                        axis_text_x=element_text(rotation=45, ha="right"),
                    )
                )

                # Preserve Enum category order using scale_x_discrete
                if isinstance(col_dtype, pl.Enum):
                    cat_order = list(col_dtype.categories)
                    p = p + scale_x_discrete(limits=cat_order)
            else:
                # Histogram for numeric
                col_data = self.data.select(pl.col(c).cast(pl.Float64).alias("value"))
                p = (
                    ggplot(col_data, aes(x="value"))
                    + geom_histogram(bins=bins, fill="slateblue", alpha=0.8)
                    + labs(x="", y="Count")
                    + ggtitle(c)
                    + theme_bw()
                    + theme(plot_title=element_text(size=10, weight="bold"))
                )

            plot_list.append(p)

        return _compose_plots(plot_list, ncol)
