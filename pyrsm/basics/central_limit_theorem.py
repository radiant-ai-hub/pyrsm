import numpy as np
import polars as pl
from plotnine import (
    aes,
    after_stat,
    geom_density,
    geom_histogram,
    ggplot,
    labs,
    stat_function,
)
from scipy import stats

from pyrsm.plot_style import ALPHA, FILL, REF, theme_pyrsm
from pyrsm.plot_utils import compose_plots

# Display name per distribution; keys are the accepted spellings
# (radiant.basics names plus the short prob_calc-style keys).
DISTRIBUTIONS = {
    "normal": "Normal",
    "norm": "Normal",
    "binomial": "Binomial",
    "binom": "Binomial",
    "uniform": "Uniform",
    "unif": "Uniform",
    "exponential": "Exponential",
    "expo": "Exponential",
}


class central_limit_theorem:
    """
    Simulate the Central Limit Theorem (radiant.basics ``clt``).

    Draws ``m`` samples of size ``n`` from the chosen distribution and shows
    how the distribution of the sample sums (or means) approaches a normal
    distribution, whatever the shape of the population.

    Parameters
    ----------
    dist : str
        ``"Normal"``, ``"Binomial"``, ``"Uniform"`` or ``"Exponential"``
        (case-insensitive; ``"norm"``, ``"binom"``, ``"unif"`` and ``"expo"``
        also work).
    n : int
        Sample size (observations per sample), at least 2.
    m : int
        Number of samples, at least 2.
    norm_mean, norm_sd : float
        Mean and standard deviation of the normal distribution.
    binom_size, binom_prob : int, float
        Number of trials and success probability of the binomial distribution.
    unif_min, unif_max : float
        Minimum and maximum of the uniform distribution.
    expo_rate : float
        Rate of the exponential distribution (mean = 1 / rate).
    seed : int | None
        Random seed. Use a seed to make the simulation reproducible.

    Attributes
    ----------
    sim : numpy.ndarray
        Simulated data with shape ``(n, m)``; each column is one sample.

    Examples
    --------
    >>> import pyrsm as rsm
    >>> clt = rsm.basics.central_limit_theorem("Uniform", n=10, m=20, unif_min=10, unif_max=20, seed=1)
    >>> clt.sim.shape
    (10, 20)
    >>> sums = clt.stat("sum")
    >>> len(sums)
    20
    """

    def __init__(
        self,
        dist: str = "Normal",
        n: int = 100,
        m: int = 100,
        norm_mean: float = 0,
        norm_sd: float = 1,
        binom_size: int = 10,
        binom_prob: float = 0.2,
        unif_min: float = 0,
        unif_max: float = 1,
        expo_rate: float = 1,
        seed: int | None = None,
    ) -> None:
        key = str(dist).strip().lower()
        if key not in DISTRIBUTIONS:
            raise ValueError(
                "Distribution must be one of Normal, Binomial, Uniform, or Exponential"
            )
        self.dist = DISTRIBUTIONS[key]
        self.n = int(n)
        self.m = int(m)
        self.norm_mean = norm_mean
        self.norm_sd = norm_sd
        self.binom_size = int(binom_size)
        self.binom_prob = binom_prob
        self.unif_min = unif_min
        self.unif_max = unif_max
        self.expo_rate = expo_rate
        self.seed = seed
        self._check()

        rng = np.random.default_rng(seed)
        size = (self.n, self.m)
        if self.dist == "Normal":
            self.sim = rng.normal(loc=norm_mean, scale=norm_sd, size=size)
        elif self.dist == "Binomial":
            self.sim = rng.binomial(n=self.binom_size, p=binom_prob, size=size)
        elif self.dist == "Uniform":
            self.sim = rng.uniform(low=unif_min, high=unif_max, size=size)
        else:
            # numpy parameterizes the exponential by its scale (1 / rate)
            self.sim = rng.exponential(scale=1 / expo_rate, size=size)

    def _check(self) -> None:
        """Input checks matching the radiant.basics CLT tool."""
        if self.n < 2:
            raise ValueError("Please choose a sample size of 2 or more")
        if self.m < 2:
            raise ValueError("Please choose 2 or more samples")
        if self.dist == "Uniform" and self.unif_max <= self.unif_min:
            raise ValueError(
                "The maximum value for the uniform distribution must be larger "
                "than the minimum value"
            )
        if self.dist == "Normal" and not self.norm_sd > 0:
            raise ValueError(
                "Please choose a positive standard deviation for the normal distribution"
            )
        if self.dist == "Exponential" and not self.expo_rate > 0:
            raise ValueError("Please choose a positive rate for the exponential distribution")
        if self.dist == "Binomial":
            if self.binom_size < 1:
                raise ValueError(
                    "Please choose a size of 1 or more for the binomial distribution"
                )
            if not 0 < self.binom_prob < 1:
                raise ValueError(
                    "Please choose a probability between 0 and 1 for the binomial distribution"
                )

    def params(self) -> dict[str, float]:
        """Parameters of the selected distribution."""
        if self.dist == "Normal":
            return {"Mean": self.norm_mean, "St. dev": self.norm_sd}
        if self.dist == "Binomial":
            return {"Size": self.binom_size, "Prob": self.binom_prob}
        if self.dist == "Uniform":
            return {"Min": self.unif_min, "Max": self.unif_max}
        return {"Rate": self.expo_rate}

    def population(self) -> tuple[float, float]:
        """Mean and standard deviation of the population distribution."""
        if self.dist == "Normal":
            return self.norm_mean, self.norm_sd
        if self.dist == "Binomial":
            n, p = self.binom_size, self.binom_prob
            return n * p, np.sqrt(n * p * (1 - p))
        if self.dist == "Uniform":
            a, b = self.unif_min, self.unif_max
            return (a + b) / 2, (b - a) / np.sqrt(12)
        return 1 / self.expo_rate, 1 / self.expo_rate

    def stat(self, stat: str = "sum") -> np.ndarray:
        """Sum or mean of each sample (one value per sample)."""
        if stat not in ("sum", "mean"):
            raise ValueError("stat must be 'sum' or 'mean'")
        return self.sim.sum(axis=0) if stat == "sum" else self.sim.mean(axis=0)

    def summary(self, stat: str = "sum", dec: int = 3) -> None:
        """
        Print the simulation setup and compare the simulated sample sums (or
        means) to what the Central Limit Theorem predicts.

        Parameters
        ----------
        stat : str
            ``"sum"`` or ``"mean"``.
        dec : int
            Number of decimals to show.
        """
        values = self.stat(stat)
        pop_mean, pop_sd = self.population()
        if stat == "sum":
            clt_mean, clt_sd = self.n * pop_mean, np.sqrt(self.n) * pop_sd
        else:
            clt_mean, clt_sd = pop_mean, pop_sd / np.sqrt(self.n)
        label = "sums" if stat == "sum" else "means"

        def fmt(x):
            # round floats and drop trailing zeros: 0.200 -> 0.2, 100.0 -> 100
            if isinstance(x, float):
                return f"{round(x, dec):,.{dec}f}".rstrip("0").rstrip(".")
            return f"{x:,}"

        print("Central Limit Theorem")
        rows = {"Distribution": self.dist}
        rows.update({k: fmt(v) for k, v in self.params().items()})
        rows["Sample size"] = fmt(self.n)
        rows["# of samples"] = fmt(self.m)
        rows["Statistic"] = f"sample {label}"
        if self.seed is not None:
            rows["Seed"] = str(self.seed)
        width = max(len(k) for k in rows)
        for k, v in rows.items():
            print(f"{k:<{width}}: {v}")

        print(f"\nSample {label}:")
        cells = [
            ["", "Mean", "St. dev"],
            ["Simulated", fmt(float(values.mean())), fmt(float(values.std(ddof=1)))],
            ["CLT", fmt(float(clt_mean)), fmt(float(clt_sd))],
        ]
        widths = [max(len(row[i]) for row in cells) for i in range(3)]
        for row in cells:
            print(f"{row[0]:<{widths[0]}}  {row[1]:>{widths[1]}}  {row[2]:>{widths[2]}}")
        print(
            f"\nCLT: with a sample size of {self.n}, the sample {label} are approximately "
            f"normal with mean {fmt(float(clt_mean))} and st. dev {fmt(float(clt_sd))}."
        )

    def plot(self, stat: str = "sum", bins: int = 15):
        """
        Four-panel CLT plot: histograms of the first and the last sample, a
        histogram of the sample sums (or means), and their density with a
        normal curve overlaid.

        Parameters
        ----------
        stat : str
            ``"sum"`` or ``"mean"``.
        bins : int
            Number of histogram bins (1 to 50).

        Returns
        -------
        plotnine composition of four ggplot objects.
        """
        bins = max(1, min(int(bins), 50))
        values = self.stat(stat)
        label = "sums" if stat == "sum" else "means"
        plots = [
            self._hist(self.sim[:, 0], bins, "Histogram of sample #1", f"CLT: {self.dist}"),
            self._hist(
                self.sim[:, -1], bins, f"Histogram of sample #{self.m}", f"CLT: {self.dist}"
            ),
            self._hist(values, bins, f"Histogram of sample {label}", f"Sample {label}"),
            self._density(values, f"Density of sample {label}", f"Sample {label}"),
        ]
        return compose_plots(plots, ncol=2, width_per_col=4.5, height_per_row=3.5)

    @staticmethod
    def _hist(x, bins, xlab, title):
        data = pl.DataFrame({"value": np.asarray(x, dtype=float)})
        return (
            ggplot(data, aes(x="value"))
            + geom_histogram(bins=bins, fill=FILL, alpha=ALPHA)
            + labs(x=xlab, y="Count", title=title)
            + theme_pyrsm()
        )

    @staticmethod
    def _density(x, xlab, title):
        x = np.asarray(x, dtype=float)
        data = pl.DataFrame({"value": x})
        mean, sd = float(x.mean()), float(x.std(ddof=1))
        p = (
            ggplot(data, aes(x="value"))
            + geom_density(aes(y=after_stat("density")), fill=FILL, alpha=ALPHA)
            + labs(x=xlab, y="Density", title=title)
            + theme_pyrsm()
        )
        if sd > 0:
            p = p + stat_function(
                fun=lambda v: stats.norm.pdf(v, mean, sd), color=REF, size=1
            )
        return p


# radiant.basics name
clt = central_limit_theorem
