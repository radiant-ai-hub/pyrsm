"""Tests for pyrsm.basics.central_limit_theorem (radiant.basics ``clt``)."""

import numpy as np
import pytest
from plotnine.composition import Compose

from pyrsm.basics import central_limit_theorem, clt


@pytest.mark.parametrize(
    "dist,kwargs,pop_mean",
    [
        ("Normal", dict(norm_mean=10, norm_sd=2), 10),
        ("Binomial", dict(binom_size=10, binom_prob=0.2), 2),
        ("Uniform", dict(unif_min=10, unif_max=20), 15),
        ("Exponential", dict(expo_rate=2), 0.5),
    ],
)
def test_simulates_n_by_m_from_the_population(dist, kwargs, pop_mean):
    c = central_limit_theorem(dist, n=200, m=300, seed=1, **kwargs)
    assert c.sim.shape == (200, 300)
    assert c.dist == dist
    assert c.population()[0] == pytest.approx(pop_mean)
    # every draw comes from the population: the grand mean is close to it
    assert c.sim.mean() == pytest.approx(pop_mean, rel=0.05)


def test_exponential_uses_rate_not_scale():
    # R's rexp(rate = 4) has mean 1 / 4
    c = central_limit_theorem("Exponential", n=1000, m=100, expo_rate=4, seed=2)
    assert c.sim.mean() == pytest.approx(0.25, rel=0.05)


def test_sum_and_mean_statistics():
    c = central_limit_theorem("Uniform", n=50, m=40, seed=3)
    assert np.allclose(c.stat("sum"), c.sim.sum(axis=0))
    assert np.allclose(c.stat("mean"), c.sim.mean(axis=0))
    with pytest.raises(ValueError):
        c.stat("median")


def test_clt_spread_matches_theory():
    c = central_limit_theorem("Binomial", n=100, m=2000, binom_size=10, binom_prob=0.2, seed=4)
    pop_mean, pop_sd = c.population()
    assert c.stat("sum").std(ddof=1) == pytest.approx(np.sqrt(100) * pop_sd, rel=0.1)
    assert c.stat("mean").std(ddof=1) == pytest.approx(pop_sd / np.sqrt(100), rel=0.1)


def test_seed_makes_runs_reproducible():
    a = central_limit_theorem("Normal", n=10, m=10, seed=7)
    b = central_limit_theorem("Normal", n=10, m=10, seed=7)
    assert np.array_equal(a.sim, b.sim)


def test_distribution_spellings_and_alias():
    assert clt is central_limit_theorem
    assert central_limit_theorem("norm", n=5, m=5).dist == "Normal"
    assert central_limit_theorem("EXPONENTIAL", n=5, m=5).dist == "Exponential"
    with pytest.raises(ValueError, match="Distribution must be one of"):
        central_limit_theorem("Poisson")


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(dist="Normal", n=1), "sample size"),
        (dict(dist="Normal", m=1), "2 or more samples"),
        (dict(dist="Uniform", unif_min=2, unif_max=1), "maximum value"),
        (dict(dist="Normal", norm_sd=0), "standard deviation"),
        (dict(dist="Exponential", expo_rate=0), "positive rate"),
        (dict(dist="Binomial", binom_size=0), "size of 1 or more"),
        (dict(dist="Binomial", binom_prob=1.2), "probability between 0 and 1"),
    ],
)
def test_input_checks(kwargs, message):
    with pytest.raises(ValueError, match=message):
        central_limit_theorem(**kwargs)


def test_summary_reports_simulated_and_clt_values(capsys):
    central_limit_theorem("Uniform", n=12, m=50, unif_min=0, unif_max=1, seed=5).summary(
        stat="mean"
    )
    out = capsys.readouterr().out
    assert "Distribution: Uniform" in out
    assert "Sample means:" in out
    assert "Simulated" in out and "CLT" in out
    # theoretical st. dev of the mean: (1 / sqrt(12)) / sqrt(12) = 1 / 12
    assert "0.083" in out


@pytest.mark.parametrize("stat", ["sum", "mean"])
def test_plot_has_four_panels(stat):
    c = central_limit_theorem("Exponential", n=30, m=40, seed=6)
    p = c.plot(stat=stat, bins=10)
    assert isinstance(p, Compose)


def test_stats_match_the_summary():
    c = central_limit_theorem("Uniform", n=12, m=50, seed=5)
    st = c.stats("mean")
    assert st["clt"]["sd"] == pytest.approx(1 / 12)
    assert st["simulated"]["mean"] == pytest.approx(c.stat("mean").mean())


def test_summary_does_not_claim_normality_at_any_sample_size(capsys):
    central_limit_theorem("Exponential", n=2, m=50, seed=1).summary(stat="mean")
    out = capsys.readouterr().out
    assert "approximately normal" not in out
    assert "gets closer to normal as the sample size grows" in out
