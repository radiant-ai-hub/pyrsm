"""Create plots using plotnine."""

import polars as pl

from pyrsm.plot_utils import compose_plots

# Geom configurations: required aesthetics and defaults
GEOM_CONFIG = {
    "dist": {
        "required": ["x"],
        "defaults": {"bins": 30, "fill": "slateblue", "alpha": 0.7},
    },
    "hist": {  # Alias for dist (numeric)
        "required": ["x"],
        "defaults": {"bins": 30, "fill": "slateblue", "alpha": 0.7},
    },
    "density": {
        "required": ["x"],
        "defaults": {"fill": "slateblue", "alpha": 0.5},
    },
    "scatter": {
        "required": ["x", "y"],
        "defaults": {"alpha": 0.7, "size": 2, "nobs": 1000},
    },
    "bar": {
        "required": ["x"],
        "defaults": {"fill": "slateblue", "alpha": 0.8},
    },
    "line": {
        "required": ["x", "y"],
        "defaults": {"size": 1},
    },
    "box": {
        "required": ["x", "y"],
        "defaults": {"fill": "slateblue", "alpha": 0.7},
    },
    "violin": {
        "required": ["x", "y"],
        "defaults": {"fill": "slateblue", "alpha": 0.7},
    },
}


def _is_categorical(df: pl.DataFrame, col: str) -> bool:
    """Check if column is categorical (string/enum or low-cardinality int)."""
    dtype = df.schema.get(col)
    if dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Enum):
        return True
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        # Treat integers with few unique values as categorical
        n_unique = df.select(pl.col(col).n_unique()).item()
        return n_unique <= 20
    return False


def _as_list(value, name: str) -> list[str]:
    """Normalize a string-or-sequence argument to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        values = list(value)
    except TypeError as err:
        raise TypeError(f"{name} must be a string or a sequence of strings") from err
    if not all(isinstance(v, str) for v in values):
        raise TypeError(f"{name} must be a string or a sequence of strings")
    return values


def visualize(
    df: pl.DataFrame | pl.LazyFrame,
    x: str | list[str] | tuple[str, ...],
    y: str | list[str] | tuple[str, ...] | None = None,
    geom: str | None = None,
    color: str | None = "slateblue",
    fill: str | None = None,
    shape: str | None = None,
    group: str | None = None,
    linetype: str | None = None,
    bins: int | None = None,
    alpha: float | None = None,
    size: int | float | None = None,
    position: str | None = None,
    smooth: str | None = None,
    jitter: bool = False,
    facet: str | None = None,
    facet_row: str | None = None,
    facet_col: str | None = None,
    title: str | None = None,
    nobs: int = 1000,
    agg: str | None = None,
    ncol: int = 2,
    ret: str = "compose",
):
    """
    Create one or more plots using plotnine.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Polars DataFrame or LazyFrame.
    x : str | list[str] | tuple[str, ...]
        Column name(s) for x-axis. Multiple values create one plot per x
        variable, or one plot per x/y pair when ``y`` is also multiple.
    y : str | list[str] | tuple[str, ...] | None
        Column name(s) for y-axis (required for scatter, line, box, violin).
    geom : str | None
        Plot type: dist, hist, density, scatter, bar, line, box, violin. Default:
        scatter if ``y`` provided, dist otherwise.
    color : str | None
        Column name for color aesthetic or literal color.
    fill : str | None
        Column name for fill aesthetic or literal color.
    shape : str | None
        Column name for shape aesthetic.
    group : str | None
        Column name for grouping.
    linetype : str | None
        Column name for linetype aesthetic.
    bins : int | None
        Number of bins for histogram (default: 30).
    alpha : float | None
        Transparency (0-1).
    size : int | float | None
        Point/line size.
    position : str | None
        Bar position: ``"stack"`` or ``"dodge"``.
    smooth : str | None
        Add smooth line to scatter: ``"lm"``, ``"loess"``, or ``"true"``.
    jitter : bool
        Add jitter to scatter plot points.
    facet : str | None
        Column for facet_wrap.
    facet_row : str | None
        Row faceting variable (for facet_grid).
    facet_col : str | None
        Column faceting variable (for facet_grid).
    title : str | None
        Plot title.
    nobs : int
        Max observations for scatter plots (default: 1000, -1 for all).
    agg : str | None
        Aggregation function for bar/scatter plots with categorical x: mean,
        median, sum, min, max. For bar plots, aggregates y by x. For scatter
        plots with categorical x, adds a line showing the aggregated value per
        category.
    ncol : int
        Number of columns in the composed plot grid when multiple plots are
        generated.
    ret : str
        Return mode for multiple plots: ``"compose"`` (default) returns a
        plotnine composition; ``"list"`` returns the individual ggplot objects.

    Returns
    -------
    plotnine.ggplot, plotnine composition, or list
        A single ggplot object, a composed plot grid, or a list of ggplot
        objects if ``ret="list"``.

    Raises
    ------
    ValueError
        If ``geom`` or ``agg`` is unknown, or required aesthetics are missing.

    Examples
    --------
    >>> import polars as pl
    >>> import pyrsm as rsm
    >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})
    >>> p = rsm.eda.visualize(df, x="x", y="y")
    >>> type(p).__name__
    'ggplot'
    """
    if ret not in ("compose", "list"):
        raise ValueError("ret must be 'compose' or 'list'")
    if ncol < 1:
        raise ValueError("ncol must be >= 1")

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    x_vars = _as_list(x, "x")
    y_vars = _as_list(y, "y")
    if not x_vars:
        raise ValueError("x is required")

    plot_list = []
    for x_var in x_vars:
        for y_var in y_vars or [None]:
            plot_list.append(
                _visualize_one(
                    df,
                    x=x_var,
                    y=y_var,
                    geom=geom,
                    color=color,
                    fill=fill,
                    shape=shape,
                    group=group,
                    linetype=linetype,
                    bins=bins,
                    alpha=alpha,
                    size=size,
                    position=position,
                    smooth=smooth,
                    jitter=jitter,
                    facet=facet,
                    facet_row=facet_row,
                    facet_col=facet_col,
                    title=title,
                    nobs=nobs,
                    agg=agg,
                )
            )

    if ret == "list":
        return plot_list

    return compose_plots(plot_list, ncol=ncol)


def _visualize_one(
    df: pl.DataFrame,
    x: str,
    y: str | None = None,
    geom: str | None = None,
    color: str | None = "slateblue",
    fill: str | None = None,
    shape: str | None = None,
    group: str | None = None,
    linetype: str | None = None,
    bins: int | None = None,
    alpha: float | None = None,
    size: int | float | None = None,
    position: str | None = None,
    smooth: str | None = None,
    jitter: bool = False,
    facet: str | None = None,
    facet_row: str | None = None,
    facet_col: str | None = None,
    title: str | None = None,
    nobs: int = 1000,
    agg: str | None = None,
):
    """Create a single plotnine plot."""
    from plotnine import (
        aes,
        facet_grid,
        facet_wrap,
        geom_bar,
        geom_boxplot,
        geom_density,
        geom_histogram,
        geom_jitter,
        geom_line,
        geom_point,
        geom_smooth,
        geom_violin,
        ggplot,
        labs,
        stat_summary,
        theme_bw,
    )

    # Aggregation function mapping using lambdas with Polars
    AGG_FUNCS = {
        "mean": lambda x: x.mean(),
        "median": lambda x: pl.Series(x).median(),
        "sum": lambda x: pl.Series(x).sum(),
        "min": lambda x: pl.Series(x).min(),
        "max": lambda x: pl.Series(x).max(),
    }
    AGG_EXPRS = {
        "mean": lambda col: pl.col(col).mean(),
        "median": lambda col: pl.col(col).median(),
        "sum": lambda col: pl.col(col).sum(),
        "min": lambda col: pl.col(col).min(),
        "max": lambda col: pl.col(col).max(),
    }

    # Validate agg argument
    if agg is not None and agg not in AGG_FUNCS:
        available = ", ".join(sorted(AGG_FUNCS.keys()))
        raise ValueError(f"Unknown agg: {agg}. Available: {available}")

    # Convert LazyFrame to DataFrame
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # Determine geom type
    if geom is None:
        geom = "scatter" if y else "dist"

    if geom not in GEOM_CONFIG:
        available = ", ".join(sorted(GEOM_CONFIG.keys()))
        raise ValueError(f"Unknown geom: {geom}. Available: {available}")

    # Validate required aesthetics
    config = GEOM_CONFIG[geom]
    if "x" in config["required"] and not x:
        raise ValueError(f"x is required for geom={geom}")
    if "y" in config["required"] and not y:
        raise ValueError(f"y is required for geom={geom}")

    # Build aesthetics dict (column mappings only)
    aes_kwargs = {"x": x}
    if y:
        aes_kwargs["y"] = y

    # Add color/fill if they're column names
    if color and color in df.columns:
        aes_kwargs["color"] = color
    if fill and fill in df.columns:
        aes_kwargs["fill"] = fill
    if shape and shape in df.columns:
        aes_kwargs["shape"] = shape
    if group and group in df.columns:
        aes_kwargs["group"] = group
    if linetype and linetype in df.columns:
        aes_kwargs["linetype"] = linetype

    # Build geom kwargs (non-aesthetic params)
    geom_kwargs = {}

    # Apply defaults then overrides
    for key, default in config["defaults"].items():
        geom_kwargs[key] = default

    if bins is not None:
        geom_kwargs["bins"] = bins
    if alpha is not None:
        geom_kwargs["alpha"] = alpha
    if size is not None:
        geom_kwargs["size"] = size

    # Handle literal colors (not column names)
    if color and color not in df.columns:
        geom_kwargs["color"] = color
    if fill and fill not in df.columns:
        geom_kwargs["fill"] = fill

    # Sample data for scatter plots if needed
    nobs_caption = None
    if geom == "scatter" and nobs != -1 and len(df) > nobs:
        df = df.sample(n=nobs, seed=1234)
        nobs_caption = f"nobs={nobs} used"

    # Build base plot
    p = ggplot(df, aes(**aes_kwargs))

    # Add geom layer
    if geom in ("dist", "hist"):
        if _is_categorical(df, x):
            # Categorical: use bar chart
            bar_kwargs = {k: v for k, v in geom_kwargs.items() if k != "bins"}
            p = p + geom_bar(**bar_kwargs)
        else:
            # Numeric: use histogram
            hist_bins = geom_kwargs.pop("bins", 30)
            p = p + geom_histogram(bins=hist_bins, **geom_kwargs)

    elif geom == "density":
        p = p + geom_density(**geom_kwargs)

    elif geom == "scatter":
        scatter_kwargs = {k: v for k, v in geom_kwargs.items() if k != "nobs"}
        if jitter:
            p = p + geom_jitter(width=0.2, height=0, **scatter_kwargs)
        else:
            p = p + geom_point(**scatter_kwargs)

        # Add smooth line if requested
        if smooth:
            if smooth == "lm":
                p = p + geom_smooth(method="lm", se=True, alpha=0.2)
            elif smooth == "loess":
                p = p + geom_smooth(method="loess", se=True, alpha=0.2)
            elif smooth in ("true", "True", True):
                p = p + geom_smooth(se=True, alpha=0.2)

        # Add aggregation line for categorical x
        if agg and _is_categorical(df, x):
            agg_func = AGG_FUNCS[agg]
            p = p + stat_summary(
                fun_y=agg_func,
                fun_ymin=agg_func,
                fun_ymax=agg_func,
                geom="crossbar",
                color="blue",
                linetype="solid",
                size=0.8,
                fatten=0,
            )

    elif geom == "bar":
        pos = position or "stack"
        bar_kwargs = {k: v for k, v in geom_kwargs.items() if k != "position"}
        if y:
            # Aggregate before plotting so duplicate x values become one bar
            # segment per group instead of many stacked row-level rectangles.
            agg_name = agg or "mean"
            group_cols = [x]
            for col in (fill, color, group, facet, facet_row, facet_col):
                if col and col in df.columns and col not in group_cols:
                    group_cols.append(col)
            plot_df = df.group_by(group_cols, maintain_order=True).agg(
                AGG_EXPRS[agg_name](y).alias(y)
            )
            p = ggplot(plot_df, aes(**aes_kwargs))
            p = p + geom_bar(stat="identity", position=pos, **bar_kwargs)
        else:
            p = p + geom_bar(stat="count", position=pos, **bar_kwargs)

    elif geom == "line":
        # For line plots, group by color if specified
        if "color" in aes_kwargs and "group" not in aes_kwargs:
            aes_kwargs["group"] = aes_kwargs["color"]
            p = ggplot(df, aes(**aes_kwargs))
        line_size = geom_kwargs.pop("size", 1)
        line_kwargs = {k: v for k, v in geom_kwargs.items() if k != "fill"}
        p = p + geom_line(size=line_size, **line_kwargs)

    elif geom == "box":
        p = p + geom_boxplot(**geom_kwargs)

    elif geom == "violin":
        p = p + geom_violin(**geom_kwargs)

    # Add faceting
    if facet:
        p = p + facet_wrap(f"~{facet}")
    elif facet_row or facet_col:
        row = facet_row or "."
        col = facet_col or "."
        p = p + facet_grid(f"{row}~{col}")

    # Add labels and theme
    x_lab = x
    y_lab = y or ""
    if geom in ("dist", "hist", "bar") and not y:
        y_lab = "Count"
    if geom == "density" and not y:
        y_lab = "Density"
    if y and geom == "bar":
        y_lab = f"{(agg or 'mean').capitalize()} of {y}"

    p = p + labs(x=x_lab, y=y_lab, title=title or "", caption=nobs_caption) + theme_bw()

    return p
