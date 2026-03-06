import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from statsmodels.stats.power import TTestIndPower

from pyrsm.utils import format_nr


def _cohen_h(p1, p2):
    """Compute Cohen's h effect size for two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def _backout_es_h(h, p_known):
    """Reverse Cohen's h to recover unknown proportion given known proportion."""
    vals = sorted(
        [
            math.sin((h - 2 * math.asin(math.sqrt(p_known))) / 2) ** 2,
            math.sin((-h - 2 * math.asin(math.sqrt(p_known))) / 2) ** 2,
        ],
        reverse=True,
    )
    return vals


_ALT_MAP = {
    "two-sided": "two-sided",
    "less": "smaller",
    "greater": "larger",
}


# ---------------------------------------------------------------------------
# R-compatible power functions for two-proportion (arcsine) test
# These match R's pwr::pwr.2p.test and pwr::pwr.2p2n.test exactly.
# ---------------------------------------------------------------------------


def _prop_power_equal_n(h, n, sig_level, alt_hyp):
    """Power for equal sample sizes (R's pwr.2p.test power formula)."""
    if alt_hyp == "less":
        return norm.cdf(norm.ppf(sig_level) - h * math.sqrt(n / 2))
    elif alt_hyp == "greater":
        return norm.cdf(norm.ppf(sig_level) + h * math.sqrt(n / 2))
    else:  # two-sided
        h = abs(h)
        return norm.sf(norm.isf(sig_level / 2) - h * math.sqrt(n / 2)) + norm.cdf(
            norm.ppf(sig_level / 2) - h * math.sqrt(n / 2)
        )


def _prop_power_unequal_n(h, n1, n2, sig_level, alt_hyp):
    """Power for unequal sample sizes (R's pwr.2p2n.test power formula)."""
    # Effective n: harmonic-mean-like formula from R's pwr.2p2n.test
    # R uses: h * sqrt(n / 2) where n is the effective sample size
    # For unequal n: effective n = 2 / (1/n1 + 1/n2) = 2*n1*n2/(n1+n2)
    n_eff = 2 * n1 * n2 / (n1 + n2)
    return _prop_power_equal_n(h, n_eff, sig_level, alt_hyp)


def _prop_solve_n(h, sig_level, power, alt_hyp):
    """Solve for n (equal sample sizes) matching R's pwr.2p.test."""

    def objective(n):
        return _prop_power_equal_n(h, n, sig_level, alt_hyp) - power

    return brentq(objective, 2, 1e10)


def _prop_solve_n_unequal(h, n_known, sig_level, power, alt_hyp, known_is_n1=True):
    """Solve for unknown n given one known n, matching R's pwr.2p2n.test."""

    def objective(n_unknown):
        if known_is_n1:
            return (
                _prop_power_unequal_n(h, n_known, n_unknown, sig_level, alt_hyp) - power
            )
        else:
            return (
                _prop_power_unequal_n(h, n_unknown, n_known, sig_level, alt_hyp) - power
            )

    return brentq(objective, 2, 1e10)


def _prop_solve_sig_level(h, n1, n2, power, alt_hyp):
    """Solve for sig.level given n1, n2, power."""

    def objective(sig_level):
        if n1 == n2:
            return _prop_power_equal_n(h, n1, sig_level, alt_hyp) - power
        else:
            return _prop_power_unequal_n(h, n1, n2, sig_level, alt_hyp) - power

    return brentq(objective, 1e-10, 1 - 1e-10)


def _prop_solve_h(n1, n2, sig_level, power, alt_hyp):
    """Solve for effect size h given n1, n2, sig_level, power."""

    def objective(h):
        if n1 == n2:
            return _prop_power_equal_n(h, n1, sig_level, alt_hyp) - power
        else:
            return _prop_power_unequal_n(h, n1, n2, sig_level, alt_hyp) - power

    return brentq(objective, 1e-10, 10)


class sample_size_comp:
    """
    Sample size calculation for comparing means or proportions.

    Exactly one parameter among the group (n1/n2, p1/p2 or delta/sd, conf, power)
    must be None — that parameter will be solved for.

    Parameters
    ----------
    type : str
        Choose "mean" or "proportion"
    n1 : int or None
        Sample size for group 1
    n2 : int or None
        Sample size for group 2
    p1 : float or None
        Proportion 1 (proportion type only)
    p2 : float or None
        Proportion 2 (proportion type only)
    delta : float or None
        Difference in means (mean type only)
    sd : float or None
        Standard deviation (mean type only)
    conf : float or None
        Confidence level
    power : float or None
        Statistical power
    ratio : float
        Sampling ratio (n1 / n2)
    alt_hyp : str
        "two-sided", "less", or "greater". For proportions, "less" means
        p1 < p2 and "greater" means p1 > p2.

    Attributes
    ----------
    n1 : float
        Sample size for group 1
    n2 : float
        Sample size for group 2
    power : float
        Statistical power
    conf : float
        Confidence level
    effect_size : float
        Computed effect size

    Examples
    --------
    >>> ssc = sample_size_comp("proportion", p1=0.1, p2=0.15, conf=0.95, power=0.8)
    >>> math.ceil(ssc.n1)
    1006
    """

    def __init__(
        self,
        type: str = "mean",
        n1: int | None = None,
        n2: int | None = None,
        p1: float | None = None,
        p2: float | None = None,
        delta: float | None = None,
        sd: float | None = None,
        conf: float | None = None,
        power: float | None = None,
        ratio: float = 1,
        alt_hyp: str = "two-sided",
    ):
        self.type = type
        self.alt_hyp = alt_hyp
        self.ratio = ratio
        self.error = None

        sig_level = None if conf is None else 1 - conf
        alt = _ALT_MAP.get(alt_hyp, "two-sided")

        if type == "mean":
            if delta is not None:
                delta = abs(delta)

            if sd is not None and sd <= 0:
                self.error = "The standard deviation must be larger than 0"
                return

            # Count None parameters
            n_none = (n1 is None or n2 is None) + (delta is None) + (sd is None)
            n_none += (power is None) + (conf is None)
            if n_none == 0 or n_none > 1:
                self.error = (
                    "Exactly one of 'Sample size', 'Delta', 'Std. deviation',\n"
                    "'Confidence level', and 'Power' must be None"
                )
                return

            solver = TTestIndPower()

            if power is None or sig_level is None:
                # Solve power or conf with known n1, n2
                es = delta / sd
                result = solver.solve_power(
                    effect_size=es,
                    nobs1=n1,
                    ratio=n2 / n1 if (n1 and n2) else ratio,
                    alpha=sig_level,
                    power=power,
                    alternative=alt,
                )
                if power is None:
                    power = result
                else:
                    sig_level = result
                    conf = 1 - sig_level
            elif n1 is None and n2 is None:
                es = delta / sd
                n1 = solver.solve_power(
                    effect_size=es,
                    nobs1=None,
                    ratio=ratio,
                    alpha=sig_level,
                    power=power,
                    alternative=alt,
                )
                n2 = n1 * ratio
            elif n1 is None:
                es = delta / sd
                n1 = solver.solve_power(
                    effect_size=es,
                    nobs1=None,
                    ratio=ratio,
                    alpha=sig_level,
                    power=power,
                    alternative=alt,
                )
            elif n2 is None:
                es = delta / sd
                n2 = (
                    solver.solve_power(
                        effect_size=es,
                        nobs1=None,
                        ratio=ratio,
                        alpha=sig_level,
                        power=power,
                        alternative=alt,
                    )
                    * ratio
                )
            elif delta is None:
                es = solver.solve_power(
                    effect_size=None,
                    nobs1=n1,
                    ratio=n2 / n1,
                    alpha=sig_level,
                    power=power,
                    alternative=alt,
                )
                delta = abs(es * sd)
            else:
                # sd is None
                es = solver.solve_power(
                    effect_size=None,
                    nobs1=n1,
                    ratio=n2 / n1,
                    alpha=sig_level,
                    power=power,
                    alternative=alt,
                )
                sd = abs(delta / es)

            self.delta = delta
            self.sd = sd
            self.effect_size = delta / sd if (delta and sd) else None

        else:
            # proportion type — use R-compatible formulas directly
            if p1 is not None and p2 is not None:
                if p1 == p2:
                    self.error = "Proportion 1 and 2 should not be equal"
                    return
                if alt_hyp == "less" and p1 >= p2:
                    self.error = (
                        "Proportion 1 must be smaller than proportion 2 if the\n"
                        "alternative hypothesis is 'p1 less than p2'"
                    )
                    return
                if alt_hyp == "greater" and p1 <= p2:
                    self.error = (
                        "Proportion 1 must be larger than proportion 2 if the\n"
                        "alternative hypothesis is 'p1 greater than p2'"
                    )
                    return

            # Count None parameters
            n_none = (n1 is None or n2 is None) + (power is None) + (p1 is None)
            n_none += (p2 is None) + (conf is None)
            if n_none == 0 or n_none > 1:
                self.error = (
                    "Exactly one of 'Sample size', 'Proportion 1', 'Proportion 2',\n"
                    "'Confidence level', and 'Power' must be None"
                )
                return

            # Compute Cohen's h (sign is correct given validation above)
            if p1 is not None and p2 is not None:
                h = _cohen_h(p1, p2)

            if power is None:
                # Solve for power
                if n1 == n2:
                    power = _prop_power_equal_n(h, n1, sig_level, alt_hyp)
                else:
                    power = _prop_power_unequal_n(h, n1, n2, sig_level, alt_hyp)
            elif sig_level is None:
                # Solve for confidence level
                sig_level = _prop_solve_sig_level(h, n1, n2, power, alt_hyp)
                conf = 1 - sig_level
            elif n1 is None and n2 is None:
                # Solve for n (equal sample sizes)
                n1 = _prop_solve_n(h, sig_level, power, alt_hyp)
                n2 = n1 * ratio
            elif n1 is None:
                n1 = _prop_solve_n_unequal(
                    h, n2, sig_level, power, alt_hyp, known_is_n1=False
                )
            elif n2 is None:
                n2 = _prop_solve_n_unequal(
                    h, n1, sig_level, power, alt_hyp, known_is_n1=True
                )
            elif p1 is None:
                h_val = _prop_solve_h(n1, n2, sig_level, power, alt_hyp)
                candidates = _backout_es_h(h_val, p2)
                if alt_hyp == "two-sided":
                    p1 = candidates
                elif alt_hyp == "less":
                    p1 = candidates[1]  # smaller
                else:
                    p1 = candidates[0]  # larger
            else:
                # p2 is None
                h_val = _prop_solve_h(n1, n2, sig_level, power, alt_hyp)
                candidates = _backout_es_h(h_val, p1)
                if alt_hyp == "two-sided":
                    p2 = candidates
                elif alt_hyp == "less":
                    p2 = candidates[0]  # larger
                else:
                    p2 = candidates[1]  # smaller

            self.p1 = p1
            self.p2 = p2
            p1_val = p1[0] if isinstance(p1, list) else p1
            p2_val = p2[0] if isinstance(p2, list) else p2
            self.effect_size = abs(_cohen_h(p1_val, p2_val))

        self.n1 = n1
        self.n2 = n2
        self.power = power
        self.conf = conf

    def summary(self, dec: int = 3) -> None:
        """Print summary of sample size comparison calculation."""
        if self.error:
            print(self.error)
            return

        label = "proportions" if self.type == "proportion" else "means"
        print(f"Sample size calculation for comparison of {label}")
        print(f"Sample size 1    : {format_nr(math.ceil(self.n1), dec=0)}")
        print(f"Sample size 2    : {format_nr(math.ceil(self.n2), dec=0)}")
        print(
            f"Total sample size: {format_nr(math.ceil(self.n1) + math.ceil(self.n2), dec=0)}"
        )

        if self.type == "mean":
            print(f"Delta            : {self.delta}")
            print(f"Std. deviation   : {self.sd}")
            print(f"Effect size      : {round(self.delta / self.sd, dec)}")
        else:
            p1_val = self.p1[0] if isinstance(self.p1, list) else self.p1
            p2_val = self.p2[0] if isinstance(self.p2, list) else self.p2
            print(f"Proportion 1     : {p1_val}")
            print(f"Proportion 2     : {p2_val}")
            print(f"Effect size      : {round(abs(_cohen_h(p1_val, p2_val)), dec)}")

        print(f"Confidence level : {round(self.conf, dec)}")
        print(f"Power            : {round(self.power, dec)}")
        print(f"Alt. hyp.        : {self.alt_hyp}")

    def plot(self):
        """Plot power curve (power vs sample size)."""
        from plotnine import aes, geom_line, geom_vline, ggplot, labs, theme_minimal

        sig_level = 1 - self.conf

        n_range = np.arange(
            max(2, math.ceil(self.n1) // 5),
            math.ceil(self.n1) * 3,
            max(1, math.ceil(self.n1) // 50),
        )
        powers = []

        if self.type == "mean":
            alt = _ALT_MAP.get(self.alt_hyp, "two-sided")
            solver = TTestIndPower()
            es = self.delta / self.sd
            for n in n_range:
                p = solver.power(
                    effect_size=es,
                    nobs1=n,
                    ratio=self.ratio,
                    alpha=sig_level,
                    alternative=alt,
                )
                powers.append(p)
        else:
            p1_val = self.p1[0] if isinstance(self.p1, list) else self.p1
            p2_val = self.p2[0] if isinstance(self.p2, list) else self.p2
            h = _cohen_h(p1_val, p2_val)
            for n in n_range:
                p = _prop_power_equal_n(h, n, sig_level, self.alt_hyp)
                powers.append(p)

        import polars as pl

        plot_df = pl.DataFrame({"n": n_range, "power": powers}).to_pandas()

        p = (
            ggplot(plot_df, aes(x="n", y="power"))
            + geom_line()
            + geom_vline(xintercept=math.ceil(self.n1), linetype="dashed", color="red")
            + labs(x="Sample size (per group)", y="Power", title="Power curve")
            + theme_minimal()
        )
        return p
