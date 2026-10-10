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
)

from pyrsm.plot_style import ALPHA, FILL, FILL_ALT, HIGHLIGHT, REF, theme_pyrsm


def iround(x, dec):
    """Round floats to ``dec`` decimals; integral floats become ints (2.0 -> 2)."""
    if not isinstance(x, float):
        return x
    x = round(x, dec)
    return int(x) if x.is_integer() else x


def check_positive(**params):
    """Raise a ValueError naming the first parameter that is not positive."""
    for name, value in params.items():
        if value is None or not value > 0:
            raise ValueError(f"{name} must be positive")


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

    The bar at the lower and/or upper bound is highlighted in green
    (HIGHLIGHT), bars inside the bounds are FILL, and bars outside the bounds
    are FILL_ALT (pyrsm.plot_style)
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
        + geom_col(alpha=ALPHA, width=0.8)
        + scale_fill_manual(
            values={
                "at_bound": HIGHLIGHT,
                "in_range": FILL,
                "out_range": FILL_ALT,
            }
        )
        + labs(title=title, x="", y="Probability")
        + theme_pyrsm()
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
        + geom_area(aes(y="y_in"), fill=FILL, alpha=ALPHA)
        + geom_area(aes(y="y_out"), fill=FILL_ALT, alpha=ALPHA)
        + geom_line(aes(y="y"), color=REF, size=0.5)
        + labs(title=title, x="", y="Density")
        + theme_pyrsm()
    )

    # Add vertical lines at bounds
    if lb is not None:
        p = p + geom_vline(xintercept=lb, linetype="dashed", color=REF, size=0.5)
    if ub is not None:
        p = p + geom_vline(xintercept=ub, linetype="dashed", color=REF, size=0.5)

    return p
