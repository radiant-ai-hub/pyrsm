"""Shared plotting utilities for pyrsm.basics classes."""

from typing import Literal

from pyrsm import plot_style as ps


class PlotConfig:
    """Backward-compatible view of :mod:`pyrsm.plot_style` (the one pyrsm style)."""

    # Primary fill colors
    FILL = ps.FILL

    # Reference line colors
    COMP_COLOR = ps.COMP  # Null hypothesis / comparison value
    STAT_COLOR = ps.REF  # Sample statistics (mean, CI bounds)

    @staticmethod
    def theme():
        """The pyrsm theme (:func:`pyrsm.plot_style.theme_pyrsm`)."""
        return ps.theme_pyrsm()


class PlotTheme:
    """Backward-compatible theme accessor.

    pyrsm has one plot style, so every name returns
    :func:`pyrsm.plot_style.theme_pyrsm`. The ``theme_name`` argument is kept
    so existing calls keep working.
    """

    @staticmethod
    def get_theme(
        theme_name: Literal["modern", "publication", "minimal", "classic"] = "modern",
    ):
        """
        Return the pyrsm theme.

        Parameters
        ----------
        theme_name : str
            Ignored; kept for backward compatibility.

        Returns
        -------
        theme
            ``pyrsm.plot_style.theme_pyrsm()``
        """
        return ps.theme_pyrsm()


class ReferenceLine:
    """Helper class for adding reference lines to plots."""

    @staticmethod
    def vline(x, color=ps.REF, linetype="solid", alpha=1.0):
        """Add vertical reference line."""
        from plotnine import geom_vline

        return geom_vline(xintercept=x, color=color, linetype=linetype, alpha=alpha)

    @staticmethod
    def hline(y, color=ps.REF, linetype="solid", alpha=1.0):
        """Add horizontal reference line."""
        from plotnine import geom_hline

        return geom_hline(yintercept=y, color=color, linetype=linetype, alpha=alpha)

    @staticmethod
    def ci_vlines(mean, ci_lower, ci_upper, comp_value=None):
        """
        Add confidence interval vertical lines for single mean tests.

        Parameters
        ----------
        mean : float
            Sample mean
        ci_lower : float
            Lower confidence bound
        ci_upper : float
            Upper confidence bound
        comp_value : float, optional
            Comparison/null hypothesis value

        Returns
        -------
        list
            List of geom_vline objects
        """
        lines = [
            ReferenceLine.vline(mean, color=ps.REF, linetype="solid"),
            ReferenceLine.vline(ci_lower, color=ps.REF, linetype="dashed"),
            ReferenceLine.vline(ci_upper, color=ps.REF, linetype="dashed"),
        ]
        if comp_value is not None:
            lines.insert(
                0, ReferenceLine.vline(comp_value, color=ps.COMP, linetype="solid")
            )
        return lines

    @staticmethod
    def ci_hlines(mean, ci_lower, ci_upper, comp_value=None):
        """
        Add confidence interval horizontal lines for comparison tests.

        Parameters
        ----------
        mean : float
            Sample mean
        ci_lower : float
            Lower confidence bound
        ci_upper : float
            Upper confidence bound
        comp_value : float, optional
            Comparison/null hypothesis value

        Returns
        -------
        list
            List of geom_hline objects
        """
        lines = [
            ReferenceLine.hline(mean, color=ps.REF, linetype="solid"),
            ReferenceLine.hline(ci_lower, color=ps.REF, linetype="dashed"),
            ReferenceLine.hline(ci_upper, color=ps.REF, linetype="dashed"),
        ]
        if comp_value is not None:
            lines.insert(
                0, ReferenceLine.hline(comp_value, color=ps.COMP, linetype="solid")
            )
        return lines

    @staticmethod
    def significance_levels():
        """Add standard significance level reference lines."""
        return [
            ReferenceLine.hline(1.96, color=ps.REF, linetype="dotted", alpha=0.5),
            ReferenceLine.hline(-1.96, color=ps.REF, linetype="dotted", alpha=0.5),
            ReferenceLine.hline(1.64, color=ps.REF, linetype="dotted", alpha=0.3),
            ReferenceLine.hline(-1.64, color=ps.REF, linetype="dotted", alpha=0.3),
            ReferenceLine.hline(0, color=ps.REF, linetype="solid", alpha=0.7),
        ]


class PlotExporter:
    """Utilities for saving plots consistently."""

    @staticmethod
    def save(
        plot,
        filename: str,
        width: float = 8,
        height: float = 6,
        dpi: int = 300,
        format: str = "png",
    ):
        """
        Save a plotnine plot with consistent settings.

        Parameters
        ----------
        plot : plotnine.ggplot
            The plot object to save
        filename : str
            Output filename (with or without extension)
        width : float
            Width in inches
        height : float
            Height in inches
        dpi : int
            Resolution in dots per inch
        format : str
            File format (png, pdf, svg)
        """
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"

        plot.save(filename, width=width, height=height, dpi=dpi, verbose=False)


def ci_label(
    alt: str = "two-sided", conf: float = 0.95, dec: int = 3
) -> tuple[str, str]:
    """
    Generate confidence interval labels for hypothesis tests.

    This function is imported from pyrsm.basics.utils for backward compatibility.

    Parameters
    ----------
    alt : str
        Alternative hypothesis type ('two-sided', 'greater', 'less')
    conf : float
        Confidence level
    dec : int
        Decimal places for rounding

    Returns
    -------
    tuple[str, str]
        Lower and upper CI label strings
    """
    import pyrsm.basics.utils as bu

    return bu.ci_label(alt, conf, dec)
