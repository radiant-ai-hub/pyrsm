from math import ceil, floor

import numpy as np
import polars as pl
from plotnine import (
    aes,
    geom_area,
    geom_col,
    geom_line,
    geom_vline,
    ggplot,
    labs,
    scale_fill_manual,
    scale_x_continuous,
    theme,
    theme_bw,
)


def iround(x, dec):
    return x if not isinstance(x, float) else round(x, dec)


def pretty_print_summary(summary_dict):
    """Print a summary dictionary in a formatted way."""
    for k, v in summary_dict.items():
        if v is not None:
            print(f"{k}: {v}")


def check(lb, ub, plb, pub):
    if (lb is not None and ub is not None and lb > ub) or (
        plb is not None and pub is not None and plb > pub
    ):
        raise ValueError(
            "Please ensure the lower bound is smaller than the upper bound"
        )


def plot_discrete(x_range, y_range, lb, ub, title=""):
    """
    Create a discrete distribution bar plot using plotnine.

    The bar at the lower and/or upper bound is highlighted in green, bars
    inside the bounds are slateblue, and bars outside the bounds are salmon
    (see make_colors_discrete in radiant.basics). When probability bounds are
    used, lb and ub are the *values* implied by those probabilities, so the
    bar at each of those values is the one shown in green.

    Parameters
    ----------
    x_range : array-like
        X values (discrete values)
    y_range : array-like
        Y values (probabilities)
    lb : float or None
        Lower bound value for highlighting
    ub : float or None
        Upper bound value for highlighting
    title : str
        Plot title

    Returns
    -------
    plotnine.ggplot
        The plot object
    """
    # Create DataFrame
    df = pl.DataFrame({"x": x_range, "prob": y_range})
    x = pl.col("x")

    # Determine fill colors based on bounds
    if lb is not None and ub is not None:
        # At a bound: green, between bounds: slateblue, outside: salmon
        fill_type = (
            pl.when((x == lb) | (x == ub))
            .then(pl.lit("at_bound"))
            .when((x > lb) & (x < ub))
            .then(pl.lit("in_range"))
            .otherwise(pl.lit("out_range"))
        )
    elif lb is not None:
        # At lb: green, > lb: slateblue, < lb: salmon
        fill_type = (
            pl.when(x == lb)
            .then(pl.lit("at_bound"))
            .when(x > lb)
            .then(pl.lit("in_range"))
            .otherwise(pl.lit("out_range"))
        )
    elif ub is not None:
        # At ub: green, < ub: slateblue, > ub: salmon
        fill_type = (
            pl.when(x == ub)
            .then(pl.lit("at_bound"))
            .when(x < ub)
            .then(pl.lit("in_range"))
            .otherwise(pl.lit("out_range"))
        )
    else:
        # All slateblue
        fill_type = pl.lit("in_range")

    df = df.with_columns(fill_type.alias("fill_type"))

    p = (
        ggplot(df, aes(x="x", y="prob", fill="fill_type"))
        + geom_col(alpha=0.7, width=0.8)
        + scale_fill_manual(
            # "#4FEE3B" at alpha 0.7 on a white panel renders as "#84f376", the
            # green used for the bound in radiant.basics (i.e., "green" at alpha
            # 0.5 on the gray ggplot panel)
            values={
                "at_bound": "#4FEE3B",
                "in_range": "slateblue",
                "out_range": "salmon",
            }
        )
        + labs(title=title, x="", y="Probability")
        + theme_bw()
        + theme(legend_position="none")
    )

    # Add x-axis breaks if few values
    if len(x_range) <= 20:
        p = p + scale_x_continuous(breaks=list(x_range))

    return p


def plot_continuous(x_range, y_range, lb, ub, title=""):
    """
    Create a continuous distribution plot using plotnine.

    Parameters
    ----------
    x_range : array-like
        X values
    y_range : array-like
        Y values (density)
    lb : float or None
        Lower bound for shading
    ub : float or None
        Upper bound for shading
    title : str
        Plot title

    Returns
    -------
    plotnine.ggplot
        The plot object
    """
    # Convert to numpy arrays if needed
    x_range = np.array(x_range)
    y_range = np.array(y_range)

    # Create DataFrame with shading regions
    df = pl.DataFrame({"x": x_range, "y": y_range})

    # Determine which region each point belongs to
    if lb is not None and ub is not None:
        df = df.with_columns(
            pl.when((pl.col("x") >= lb) & (pl.col("x") <= ub))
            .then(pl.col("y"))
            .otherwise(pl.lit(0.0))
            .alias("y_in"),
            pl.when((pl.col("x") < lb) | (pl.col("x") > ub))
            .then(pl.col("y"))
            .otherwise(pl.lit(0.0))
            .alias("y_out"),
        )
    elif lb is not None:
        df = df.with_columns(
            pl.when(pl.col("x") >= lb)
            .then(pl.col("y"))
            .otherwise(pl.lit(0.0))
            .alias("y_in"),
            pl.when(pl.col("x") < lb)
            .then(pl.col("y"))
            .otherwise(pl.lit(0.0))
            .alias("y_out"),
        )
    elif ub is not None:
        df = df.with_columns(
            pl.when(pl.col("x") <= ub)
            .then(pl.col("y"))
            .otherwise(pl.lit(0.0))
            .alias("y_in"),
            pl.when(pl.col("x") > ub)
            .then(pl.col("y"))
            .otherwise(pl.lit(0.0))
            .alias("y_out"),
        )
    else:
        df = df.with_columns(
            pl.col("y").alias("y_in"),
            pl.lit(0.0).alias("y_out"),
        )

    # Build the plot
    p = (
        ggplot(df, aes(x="x"))
        + geom_area(aes(y="y_in"), fill="slateblue", alpha=0.6)
        + geom_area(aes(y="y_out"), fill="salmon", alpha=0.6)
        + geom_line(aes(y="y"), color="black", size=0.5)
        + labs(title=title, x="", y="Density")
        + theme_bw()
    )

    # Add vertical lines at bounds
    if lb is not None:
        p = p + geom_vline(xintercept=lb, linetype="dashed", color="black", size=0.5)
    if ub is not None:
        p = p + geom_vline(xintercept=ub, linetype="dashed", color="black", size=0.5)

    return p
