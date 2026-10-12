import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import nct, norm
from scipy.stats import t as t_dist

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
    """Solve for effect size h given n1, n2, sig_level, power.

    h = ES.h(p1, p2) is negative when p1 < p2, so 'less' searches below 0
    (R's pwr.2p2n.test does the same).
    """

    def objective(h):
        return _prop_power_unequal_n(h, n1, n2, sig_level, alt_hyp) - power

    if alt_hyp == "less":
        return brentq(objective, -10, -1e-10)
    return brentq(objective, 1e-10, 10)


def _t_tail(q, nu, ncp, upper):
    """P(T > q) (upper) or P(T < q) for a noncentral t; scipy returns nan far
    in a tail, where the normal approximation is exact enough."""
    v = nct.sf(q, nu, ncp) if upper else nct.cdf(q, nu, ncp)
    if not np.isfinite(v):
        v = norm.sf(q - ncp) if upper else norm.cdf(q - ncp)
    return float(v)


def _mean_power(es, n1, n2, sig_level, alt_hyp):
    """Power of the two-sample t-test (R's pwr.t2n.test).

    ``es`` is |delta| / sd. Delta is the size of the difference to detect, so
    both one-sided alternatives use the upper tail.
    """
    nu = n1 + n2 - 2
    ncp = es / math.sqrt(1 / n1 + 1 / n2)
    if alt_hyp == "two-sided":
        q = t_dist.isf(sig_level / 2, nu)
        return _t_tail(q, nu, ncp, True) + _t_tail(-q, nu, ncp, False)
    return _t_tail(t_dist.isf(sig_level, nu), nu, ncp, True)


NO_SOLUTION = (
    "These inputs do not have a solution. Change one of the values and try again"
)


def _num(x, dec):
    """Round for display: 2.0 -> '2', 0.75376 -> '0.754' (dec=3)."""
    x = round(float(x), dec)
    return str(int(x)) if x.is_integer() else str(x)


class sample_size_comp:
    """
    Sample size calculation for comparing means or proportions.

    Exactly one parameter among the group (n1/n2, p1/p2 or delta/sd, conf, power)
    must be None — that parameter will be solved for. With n1 and n2 both None
    the sample size of each group is solved for (in the ratio ``ratio``); with
    only one of them None, the size of that group is solved for given the other.

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
        Difference in means to detect (mean type only)
    sd : float or None
        Standard deviation (mean type only)
    conf : float or None
        Confidence level
    power : float or None
        Statistical power
    ratio : float
        Group size ratio n1 / n2, used when n1 and n2 are both None (1 gives
        groups of equal size, 3 makes group 1 three times as large as group 2)
    alt_hyp : str
        "two-sided", "less", or "greater". For proportions, "less" means
        p1 < p2 and "greater" means p1 > p2. For means, delta is the size of
        the difference to detect, and "less" and "greater" both give the
        one-sided test.

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
    error : str or None
        Why the inputs could not be used (None when they could)

    Examples
    --------
    >>> ssc = sample_size_comp("proportion", p1=0.1, p2=0.15, conf=0.95, power=0.8)
    >>> math.ceil(ssc.n1)
    681
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

        if ratio is None or ratio <= 0:
            self.error = "The group ratio (n1 / n2) must be larger than 0"
            return
        for name, value in (("Confidence level", conf), ("Power", power)):
            if value is not None and not 0 < value < 1:
                self.error = f"{name} must be between 0 and 1"
                return
        for name, value in (("Sample size 1", n1), ("Sample size 2", n2)):
            if value is not None and value < 2:
                self.error = f"{name} must be at least 2"
                return

        sig_level = None if conf is None else 1 - conf
        # n1 and n2 both missing: solve n2 with n1 = ratio * n2 (both >= 2)
        n_low = max(2, 2 / ratio)

        if type == "mean":
            if delta is not None:
                delta = abs(delta)

            if sd is not None and sd <= 0:
                self.error = "The standard deviation must be larger than 0"
                return
            if delta is not None and delta == 0:
                self.error = "Delta must not be 0"
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

            es = delta / sd if (delta is not None and sd is not None) else None

            def power_at(es_, n1_, n2_, sig_):
                return _mean_power(es_, n1_, n2_, sig_, alt_hyp)

            try:
                if power is None:
                    power = power_at(es, n1, n2, sig_level)
                elif sig_level is None:
                    sig_level = brentq(
                        lambda a: power_at(es, n1, n2, a) - power, 1e-10, 1 - 1e-10
                    )
                    conf = 1 - sig_level
                elif n1 is None and n2 is None:
                    n2 = brentq(
                        lambda n: power_at(es, ratio * n, n, sig_level) - power,
                        n_low,
                        1e9,
                    )
                    n1 = ratio * n2
                elif n1 is None:
                    n1 = brentq(
                        lambda n: power_at(es, n, n2, sig_level) - power, 2, 1e9
                    )
                elif n2 is None:
                    n2 = brentq(
                        lambda n: power_at(es, n1, n, sig_level) - power, 2, 1e9
                    )
                else:
                    es = brentq(
                        lambda e: power_at(e, n1, n2, sig_level) - power, 1e-10, 100
                    )
                    if delta is None:
                        delta = es * sd
                    else:
                        sd = delta / es
            except (ValueError, RuntimeError, ZeroDivisionError):
                self.error = NO_SOLUTION
                return

            self.delta = delta
            self.sd = sd
            self.effect_size = delta / sd

        else:
            for name, value in (("Proportion 1", p1), ("Proportion 2", p2)):
                if value is not None and not 0 <= value <= 1:
                    self.error = f"{name} must be between 0 and 1"
                    return

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

            # Cohen's h (sign is correct given validation above)
            if p1 is not None and p2 is not None:
                h = _cohen_h(p1, p2)

            try:
                if power is None:
                    power = _prop_power_unequal_n(h, n1, n2, sig_level, alt_hyp)
                elif sig_level is None:
                    sig_level = _prop_solve_sig_level(h, n1, n2, power, alt_hyp)
                    conf = 1 - sig_level
                elif n1 is None and n2 is None:
                    if ratio == 1:
                        n1 = n2 = _prop_solve_n(h, sig_level, power, alt_hyp)
                    else:
                        n2 = brentq(
                            lambda n: _prop_power_unequal_n(
                                h, ratio * n, n, sig_level, alt_hyp
                            )
                            - power,
                            n_low,
                            1e10,
                        )
                        n1 = ratio * n2
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
            except (ValueError, RuntimeError, ZeroDivisionError):
                self.error = NO_SOLUTION
                return

            self.p1 = p1
            self.p2 = p2
            p1_val = p1[0] if isinstance(p1, list) else p1
            p2_val = p2[0] if isinstance(p2, list) else p2
            self.effect_size = abs(_cohen_h(p1_val, p2_val))

        self.n1 = float(n1)
        self.n2 = float(n2)
        self.power = float(power)
        self.conf = float(conf)

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
            print(f"Delta            : {_num(self.delta, dec)}")
            print(f"Std. deviation   : {_num(self.sd, dec)}")
        else:

            def props(p):
                vals = p if isinstance(p, list) else [p]
                return ", ".join(_num(v, dec) for v in vals)

            print(f"Proportion 1     : {props(self.p1)}")
            print(f"Proportion 2     : {props(self.p2)}")
        print(f"Effect size      : {_num(self.effect_size, dec)}")
        print(f"Confidence level : {_num(self.conf, dec)}")
        print(f"Power            : {_num(self.power, dec)}")
        print(f"Alt. hyp.        : {self.alt_hyp}")

    def plot(self):
        """Plot the power curve: power for a range of sample sizes, at the
        current effect size, confidence level and n2 / n1 ratio."""
        from plotnine import aes, geom_line, geom_vline, ggplot, labs

        from pyrsm.plot_style import COMP, LINE, theme_pyrsm

        sig_level = 1 - self.conf
        n1 = math.ceil(self.n1)
        share = self.n2 / self.n1
        n_range = np.arange(max(2, n1 // 5), n1 * 3, max(1, n1 // 50))

        if self.type == "mean":
            es = self.delta / self.sd
            powers = [
                _mean_power(es, n, n * share, sig_level, self.alt_hyp) for n in n_range
            ]
        else:
            p1_val = self.p1[0] if isinstance(self.p1, list) else self.p1
            p2_val = self.p2[0] if isinstance(self.p2, list) else self.p2
            h = _cohen_h(p1_val, p2_val)
            powers = [
                _prop_power_unequal_n(h, n, n * share, sig_level, self.alt_hyp)
                for n in n_range
            ]

        import polars as pl

        plot_df = pl.DataFrame({"n": n_range, "power": powers}).to_pandas()
        xlab = (
            "Sample size per group"
            if math.isclose(share, 1)
            else f"Sample size group 1 (n2 = {_num(share, 3)} x n1)"
        )
        p = (
            ggplot(plot_df, aes(x="n", y="power"))
            + geom_line(color=LINE)
            + geom_vline(xintercept=n1, linetype="dashed", color=COMP)
            + labs(x=xlab, y="Power", title="Power curve")
            + theme_pyrsm()
        )
        return p
