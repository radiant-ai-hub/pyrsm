import numpy as np
from plotnine import annotate, labs

from pyrsm.plot_style import REF

from .prob_calc import (
    plot_prob_binom,
    plot_prob_chisq,
    plot_prob_disc,
    plot_prob_expo,
    plot_prob_fdist,
    plot_prob_lnorm,
    plot_prob_norm,
    plot_prob_pois,
    plot_prob_tdist,
    plot_prob_unif,
    prob_binom,
    prob_chisq,
    prob_disc,
    prob_expo,
    prob_fdist,
    prob_lnorm,
    prob_norm,
    prob_pois,
    prob_tdist,
    prob_unif,
    summary_prob_binom,
    summary_prob_chisq,
    summary_prob_disc,
    summary_prob_expo,
    summary_prob_fdist,
    summary_prob_lnorm,
    summary_prob_norm,
    summary_prob_pois,
    summary_prob_tdist,
    summary_prob_unif,
)

pc_dist = {
    "Binomial": "binom",
    "Chi-squared": "chisq",
    "Discrete": "disc",
    "Exponential": "expo",
    "F": "fdist",
    "Log normal": "lnorm",
    "Normal": "norm",
    "Poisson": "pois",
    "t": "tdist",
    "Uniform": "unif",
}

#: Defaults from the radiant.basics probability calculator UI
#: (inst/app/tools/analysis/prob_calc_ui.R), per distribution:
#: ``params`` -- distribution parameters, filled in by ``prob_calc`` when omitted;
#: ``values`` / ``probs`` -- the bounds the R app pre-fills for each calculation
#: type. ``None`` is an empty input (R's NA, or -Inf for a lower bound).
#: Bounds are a starting point for an interactive UI and are never applied
#: implicitly: ``prob_calc("norm")`` computes no probabilities.
pc_defaults = {
    "binom": {"params": {"n": 10, "p": 0.2},
              "values": {"lb": None, "ub": 3}, "probs": {"plb": None, "pub": 0.3}},
    "chisq": {"params": {"df": 1},
              "values": {"lb": None, "ub": 3.841}, "probs": {"plb": None, "pub": 0.95}},
    "disc": {"params": {"v": "1 2 3 4 5 6", "p": "1/6"},
             "values": {"lb": None, "ub": 3}, "probs": {"plb": None, "pub": 0.95}},
    "expo": {"params": {"rate": 1},
             "values": {"lb": None, "ub": 2.996}, "probs": {"plb": None, "pub": 0.95}},
    "fdist": {"params": {"df1": 10, "df2": 10},
              "values": {"lb": None, "ub": 2.978}, "probs": {"plb": None, "pub": 0.95}},
    "lnorm": {"params": {"meanlog": 0, "sdlog": 1},
              "values": {"lb": 0, "ub": 1}, "probs": {"plb": 0.025, "pub": 0.975}},
    "norm": {"params": {"mean": 0, "sd": 1},
             "values": {"lb": None, "ub": 0}, "probs": {"plb": 0.025, "pub": 0.975}},
    "pois": {"params": {"lamb": 1},
             "values": {"lb": None, "ub": 3}, "probs": {"plb": None, "pub": 0.95}},
    "tdist": {"params": {"df": 10},
              "values": {"lb": None, "ub": 2.228}, "probs": {"plb": 0.025, "pub": 0.975}},
    "unif": {"params": {"min": 0, "max": 1},
             "values": {"lb": None, "ub": 0.3}, "probs": {"plb": None, "pub": 0.3}},
}


#: Plot subtitle: (argument, label) per distribution, labelled as in the UI.
_SPEC = {
    "binom": [("n", "n"), ("p", "p")],
    "chisq": [("df", "df")],
    "disc": [("v", "values"), ("p", "probabilities")],
    "expo": [("rate", "rate")],
    "fdist": [("df1", "df1"), ("df2", "df2")],
    "lnorm": [("meanlog", "mean log"), ("sdlog", "sd log")],
    "norm": [("mean", "mean"), ("sd", "sd")],
    "pois": [("lamb", "lambda")],
    "tdist": [("df", "df")],
    "unif": [("min", "min"), ("max", "max")],
}


def _fmt(x) -> str:
    """3 decimals at most, no trailing zeros: 0.050 -> 0.05, 2.0 -> 2."""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{v:.3f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))


class prob_calc:
    """
    Probability calculator for common distributions.

    Parameters
    ----------
    distribution : str
        Distribution key (e.g., ``"binom"``, ``"norm"``, ``"pois"``).
    **kwargs : dict
        Parameters passed to the selected distribution function.

    Attributes
    ----------
    dct : dict
        Computed distribution summary (values and probabilities).

    Examples
    --------
    >>> import pyrsm as rsm
    >>> pc = rsm.basics.prob_calc("binom", n=10, p=0.5, lb=2, ub=4)
    >>> pc.dct["lb"], pc.dct["ub"]
    (2, 4)
    >>> pc = rsm.basics.prob_calc("norm", mean=0, sd=1, plb=0.95)
    >>> pc.dct["plb"]
    0.95
    >>> rsm.basics.prob_calc("chisq", ub=3.841).args["df"]  # radiant.basics default
    1
    """

    def __init__(self, distribution: str, **kwargs) -> None:
        self.distribution = distribution
        # distribution parameters left out take the radiant.basics defaults
        defaults = pc_defaults.get(distribution, {}).get("params", {})
        kwargs = {**{k: v for k, v in defaults.items() if k not in kwargs}, **kwargs}
        self.args = kwargs
        if self.distribution == "binom":
            self.dct = prob_binom(**kwargs)
        elif self.distribution == "chisq":
            self.dct = prob_chisq(**kwargs)
        elif self.distribution == "disc":
            self.dct = prob_disc(**kwargs)
        elif self.distribution == "expo":
            self.dct = prob_expo(**kwargs)
        elif self.distribution == "fdist":
            self.dct = prob_fdist(**kwargs)
        elif self.distribution == "lnorm":
            self.dct = prob_lnorm(**kwargs)
        elif self.distribution == "norm":
            self.dct = prob_norm(**kwargs)
        elif self.distribution == "pois":
            self.dct = prob_pois(**kwargs)
        elif self.distribution == "tdist":
            self.dct = prob_tdist(**kwargs)
        elif self.distribution == "unif":
            self.dct = prob_unif(**kwargs)
        else:
            raise ValueError(f"Distribution must be one of {list(pc_dist.keys())}")

    def summary(self, dec=3, ret=False):
        """
        Print or return a formatted summary of probabilities.

        Parameters
        ----------
        dec : int
            Number of decimals for printed values.
        ret : bool
            If True, return the summary dict instead of printing.

        Returns
        -------
        dict | None
            Summary dictionary if ``ret=True``; otherwise None.
        """
        type = "probs" if ("plb" in self.args) or ("pub" in self.args) else "values"
        if self.distribution == "binom":
            pdict = summary_prob_binom(self.dct, type=type, dec=dec, ret=True)
        elif self.distribution == "chisq":
            pdict = summary_prob_chisq(self.dct, type=type, dec=dec, ret=True)
        elif self.distribution == "disc":
            pdict = summary_prob_disc(self.dct, type=type, dec=dec, ret=True)
        elif self.distribution == "expo":
            pdict = summary_prob_expo(self.dct, type=type, dec=dec, ret=True)
        elif self.distribution == "fdist":
            pdict = summary_prob_fdist(self.dct, type=type, dec=dec, ret=True)
        elif self.distribution == "lnorm":
            pdict = summary_prob_lnorm(self.dct, type=type, dec=dec, ret=True)
        elif self.distribution == "norm":
            pdict = summary_prob_norm(self.dct, type=type, dec=dec, ret=True)
        elif self.distribution == "pois":
            pdict = summary_prob_pois(self.dct, type=type, dec=dec, ret=True)
        elif self.distribution == "tdist":
            pdict = summary_prob_tdist(self.dct, type=type, dec=dec, ret=True)
        elif self.distribution == "unif":
            pdict = summary_prob_unif(self.dct, type=type, dec=dec, ret=True)
        if ret:
            return pdict
        else:
            self.pretty_print_summary(pdict)

    @staticmethod
    def pretty_print_summary(pdict):
        """Pretty-print a summary dictionary returned by ``summary``."""
        nonprob = {}
        prob = {}
        for k, v in pdict.items():
            # print 3 rather than 3.0, as radiant.basics does
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            if k.startswith("P(") or k.startswith("1 - P("):
                prob[k] = v
            else:
                nonprob[k] = v

        print("Probability Calculator")
        if nonprob:
            maxlen = max(len(k) for k in nonprob.keys())
            for k, v in nonprob.items():
                if v is not None:
                    pad = " " * (maxlen - len(k))
                    print(f"{k}{pad}: {v}")
        # Probability section: "P(...) =" with the values in one column, as in
        # radiant.basics
        # as in radiant.basics, only the interval lines are padded:
        # "P(1 <= X <= 3)     = 0.621" above "1 - P(1 <= X <= 3) = 0.379"
        lines = [(k, v) for k, v in prob.items() if v is not None]
        if lines:
            print()
            width = max(
                (len(k) for k, _ in lines if not k.startswith("P(X")), default=0
            )
            for k, v in lines:
                pad = "" if k.startswith("P(X") else " " * (width - len(k))
                print(f"{k}{pad} = {v}")

    def plot(self):
        """
        Plot the distribution using plotnine.

        Returns
        -------
        plotnine.ggplot
            Plotnine ggplot object for the selected distribution.
        """
        type = "probs" if ("plb" in self.args) or ("pub" in self.args) else "values"
        plot_fn = {
            "binom": plot_prob_binom,
            "chisq": plot_prob_chisq,
            "disc": plot_prob_disc,
            "expo": plot_prob_expo,
            "fdist": plot_prob_fdist,
            "lnorm": plot_prob_lnorm,
            "norm": plot_prob_norm,
            "pois": plot_prob_pois,
            "tdist": plot_prob_tdist,
            "unif": plot_prob_unif,
        }.get(self.distribution)
        if plot_fn is None:
            return None
        return self._annotate(plot_fn(self.dct, type=type), type)

    def spec(self) -> str:
        """The distribution's parameters as one line, e.g. ``"mean = 0, sd = 1"``."""
        parts = []
        for arg, label in _SPEC.get(self.distribution, []):
            value = self.args.get(arg)
            if value is None:
                continue
            if isinstance(value, (list, tuple, np.ndarray)):
                value = " ".join(_fmt(v) for v in value)
            parts.append(f"{label} = {_fmt(value)}")
        text = ", ".join(parts)
        return text if len(text) <= 80 else text[:77] + "..."

    def thresholds(self, type: str = "values") -> list[tuple[str, float, str]]:
        """The bounds drawn on the plot: ``(name, x, label)`` per bound.

        Values: the bound itself (``"lb = 90"``). Probabilities: the probability
        and the value it implies (``"plb = 0.05 (x = -1.66)"``).
        """
        out = []
        for side in ("lb", "ub"):
            if type == "values":
                x = self.dct.get(side)
                if x is not None:
                    out.append((side, float(x), f"{side} = {_fmt(x)}"))
            else:
                prob = self.dct.get(f"p{side}")
                x = self.dct.get(f"v_{side}", self.dct.get(f"v{side}"))
                if prob is not None and x is not None:
                    out.append((f"p{side}", float(x), f"p{side} = {_fmt(prob)} (x = {_fmt(x)})"))
        return out

    def _annotate(self, p, type: str):
        """Show what was calculated: the parameters as the subtitle and a
        label at each bound line (or highlighted bar)."""
        p = p + labs(subtitle=self.spec())
        bounds = self.thresholds(type)
        if not bounds:
            return p
        data = p.data
        ycol = "prob" if "prob" in data.columns else "y"
        # A chi-squared density with df = 1 is infinite at 0 (and not drawn).
        heights = data[ycol].to_numpy().astype(float)
        heights = heights[np.isfinite(heights)]
        top = float(heights.max()) * 1.08 if heights.size else 1.0
        xmin, xmax = float(data["x"].min()), float(data["x"].max())
        mid = (xmin + xmax) / 2
        for i, (_, x, label) in enumerate(bounds):
            # Text runs away from the middle so it stays on the canvas; a
            # second label sits a little lower so two bounds never overlap.
            right = x < mid
            p = p + annotate(
                "text",
                x=x,
                y=top * (1 - 0.07 * i),
                label=f" {label}" if right else f"{label} ",
                ha="left" if right else "right",
                va="bottom",
                size=9,
                color=REF,
            )
        return p


if __name__ == "__main__":
    pc = prob_calc("binom", n=10, p=0.3, lb=1, ub=3)
    pc.summary()
    pc.plot()
    pc = prob_calc("binom", n=10, p=0.1168, lb=4)
    pc.summary()
    pc.plot()
    pc = prob_calc("binom", n=10, p=0.1168, ub=2)
    pc.summary()
    pc.plot()
    pc = prob_calc("binom", n=10, p=0.1168, plb=0.3, pub=0.9)
    pc.summary()
    pc.plot()
    pc = prob_calc("binom", n=10, p=0.1168, pub=0.9)
    pc.summary()
    pc.plot()
    pc = prob_calc("binom", n=10, p=0.1168, plb=0.3)
    pc.summary()
    pc.plot()

    pc = prob_binom(n=10, p=0.3, lb=1, ub=3)
    summary_prob_binom(pc, type="values")
    plot_prob_binom(pc, type="values")
    pb = prob_binom(n=10, p=0.3, lb=1)
    summary_prob_binom(pb, type="values")
    plot_prob_binom(pb, type="values")
    pb = prob_binom(n=10, p=0.3, ub=3)
    summary_prob_binom(pb, type="values")
    plot_prob_binom(pb, type="values")

    pb = prob_binom(n=10, p=0.3, plb=0.1, pub=0.8)
    summary_prob_binom(pb, type="probs")
    plot_prob_binom(pb, type="probs")
    pb = prob_binom(n=10, p=0.3, plb=0.1)
    summary_prob_binom(pb, type="probs")
    plot_prob_binom(pb, type="probs")
    pb = prob_binom(n=10, p=0.3, pub=0.8)
    summary_prob_binom(pb, type="probs")
    plot_prob_binom(pb, type="probs")

    v = list(range(1, 7))
    p = [1 / 6] * 6
    pd = prob_disc(v, p, lb=2, ub=5)
    summary_prob_disc(pd, type="values")
    plot_prob_disc(pd, type="values")
    pd = prob_disc(v, p, lb=2)
    summary_prob_disc(pd, type="values")
    plot_prob_disc(pd, type="values")
    pd = prob_disc(v, p, ub=5)
    summary_prob_disc(pd, type="values")
    plot_prob_disc(pd, type="values")

    v = list(range(1, 7))
    p = [2 / 6, 2 / 6, 1 / 12, 1 / 12, 1 / 12, 1 / 12]
    pd = prob_disc(v, p, plb=0.5, pub=0.8)
    summary_prob_disc(pd, type="probs")
    plot_prob_disc(pd, type="probs")
    pd = prob_disc(v, p, plb=0.5)
    summary_prob_disc(pd, type="probs")
    plot_prob_disc(pd, type="probs")

    v = list(range(1, 7))
    p = [2 / 6, 2 / 6, 1 / 12, 1 / 12, 1 / 12, 1 / 12]
    pd = prob_disc(v, p, plb=0.05, pub=0.95)
    summary_prob_disc(pd, type="probs")
    plot_prob_disc(pd, type="probs")

    pd = prob_disc(v, p, pub=0.95)
    summary_prob_disc(pd, type="probs")
    plot_prob_disc(pd, type="probs")

    pd = prob_disc(v, p, plb=0.05)
    summary_prob_disc(pd, type="probs")
    plot_prob_disc(pd, type="probs")

    pf = prob_fdist(df1=10, df2=10, lb=0.5, ub=2.978)
    summary_prob_fdist(pf, type="values")
    plot_prob_fdist(pf, type="values")
    pf = prob_fdist(df1=10, df2=10, lb=0.5)
    summary_prob_fdist(pf, type="values")
    plot_prob_fdist(pf, type="values")
    pf = prob_fdist(df1=10, df2=10, ub=2.978)
    summary_prob_fdist(pf, type="values")
    plot_prob_fdist(pf, type="values")

    pf = prob_fdist(df1=10, df2=10, plb=0.05, pub=0.95)
    summary_prob_fdist(pf, type="probs")
    plot_prob_fdist(pf, type="probs")
    pf = prob_fdist(df1=10, df2=10, plb=0.05)
    summary_prob_fdist(pf, type="probs")
    pf = prob_fdist(df1=10, df2=10, pub=0.95)
    summary_prob_fdist(pf, type="probs")
    plot_prob_fdist(pf, type="probs")

    pn = prob_norm(mean=0, stdev=1, lb=-0.5, ub=0.5)
    summary_prob_norm(pn, type="values")
    plot_prob_norm(pn, type="values")
    pn = prob_norm(mean=0, stdev=1, lb=-0.5)
    summary_prob_norm(pn, type="values")
    plot_prob_norm(pn, type="values")
    pn = prob_norm(mean=0, stdev=1, ub=0.5)
    summary_prob_norm(pn, type="values")
    plot_prob_norm(pn, type="values")

    pn = prob_norm(mean=0, stdev=1, plb=0.025, pub=0.975)
    summary_prob_norm(pn, type="probs")
    plot_prob_norm(pn, type="probs")
    pn = prob_norm(mean=0, stdev=1, plb=0.025)
    summary_prob_norm(pn, type="probs")
    plot_prob_norm(pn, type="probs")
    pn = prob_norm(mean=0, stdev=1, pub=0.975)
    summary_prob_norm(pn, type="probs")
    plot_prob_norm(pn, type="probs")

    pc = prob_chisq(df=1, lb=1, ub=3.841)
    summary_prob_chisq(pc, type="values")
    plot_prob_chisq(pc, type="values")
    pc = prob_chisq(df=1, lb=1)
    summary_prob_chisq(pc, type="values")
    plot_prob_chisq(pc, type="values")
    pc = prob_chisq(df=1, ub=3.841)
    summary_prob_chisq(pc, type="values")
    plot_prob_chisq(pc, type="values")

    pc = prob_chisq(df=1, plb=0.05, pub=0.95)
    summary_prob_chisq(pc, type="probs")
    plot_prob_chisq(pc, type="probs")
    pc = prob_chisq(df=1, plb=0.05)
    summary_prob_chisq(pc, type="probs")
    plot_prob_chisq(pc, type="probs")
    pc = prob_chisq(df=1, pub=0.95)
    summary_prob_chisq(pc, type="probs")
    plot_prob_chisq(pc, type="probs")

    pu = prob_unif(min=0, max=1, lb=0.2, ub=0.8)
    summary_prob_unif(pu, type="values")
    plot_prob_unif(pu, type="values")
    pu = prob_unif(min=0, max=1, lb=0.2)
    summary_prob_unif(pu, type="values")
    plot_prob_unif(pu, type="values")
    pu = prob_unif(min=0, max=1, ub=0.8)
    summary_prob_unif(pu, type="values")
    plot_prob_unif(pu, type="values")

    pu = prob_unif(min=0, max=1, plb=0.2, pub=0.8)
    summary_prob_unif(pu, type="probs")
    plot_prob_unif(pu, type="probs")
    pu = prob_unif(min=0, max=1, plb=0.2)
    summary_prob_unif(pu, type="probs")
    plot_prob_unif(pu, type="probs")
    pu = prob_unif(min=0, max=1, pub=0.8)
    summary_prob_unif(pu, type="probs")
    plot_prob_unif(pu, type="probs")

    pt = prob_tdist(df=10, lb=-2.228, ub=2.228)
    summary_prob_tdist(pt, type="values")
    plot_prob_tdist(pt, type="values")
    pt = prob_tdist(df=10, ub=2.228)
    summary_prob_tdist(pt, type="values")
    plot_prob_tdist(pt, type="values")
    pt = prob_tdist(df=10, lb=-2.228)
    plot_prob_tdist(pt, type="values")
    pt = prob_tdist(df=10)
    summary_prob_tdist(pt, type="values")
    plot_prob_tdist(pt, type="values")

    pt = prob_tdist(df=10, plb=0.025, pub=0.975)
    summary_prob_tdist(pt, type="probs")
    plot_prob_tdist(pt, type="prob")
    pt = prob_tdist(df=10, pub=0.975)
    summary_prob_tdist(pt, type="probs")
    plot_prob_tdist(pt, type="probs")
    pt = prob_tdist(df=10, plb=0.025)
    summary_prob_tdist(pt, type="probs")
    plot_prob_tdist(pt, type="probs")
