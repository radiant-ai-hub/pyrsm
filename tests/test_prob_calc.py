import math

from scipy import stats

from pyrsm.basics.prob_calc import binomial, normal, poisson


def test_binomial_basic():
    # compare pmf and cdf for a small example
    n = 10
    p = 0.3
    res = binomial.prob_binom(n=n, p=p, lb=2, ub=4)
    # manual checks
    assert res["n"] == n
    assert math.isclose(res["p_elb"], stats.binom.pmf(2, n, p), rel_tol=1e-9)
    assert math.isclose(res["p_eub"], stats.binom.pmf(4, n, p), rel_tol=1e-9)
    assert math.isclose(
        res["p_int"], sum(stats.binom.pmf(k, n, p) for k in range(2, 5)), rel_tol=1e-9
    )


def test_poisson_basic():
    lam = 2.5
    res = poisson.prob_pois(lamb=lam, lb=0, ub=3)
    assert res["lamb"] == lam
    # pmf at 0
    assert math.isclose(res["p_elb"], stats.poisson.pmf(0, lam), rel_tol=1e-9)
    assert math.isclose(res["p_eub"], stats.poisson.pmf(3, lam), rel_tol=1e-9)
    assert math.isclose(
        res["p_int"], sum(stats.poisson.pmf(k, lam) for k in range(0, 4)), rel_tol=1e-9
    )


def test_normal_basic():
    mean = 0
    sd = 1
    res = normal.prob_norm(mean=mean, sd=sd, lb=-1, ub=1)
    assert math.isclose(res["p_lb"], stats.norm.cdf(-1, mean, sd), rel_tol=1e-9)
    assert math.isclose(res["p_ub"], stats.norm.cdf(1, mean, sd), rel_tol=1e-9)
    assert math.isclose(
        res["p_int"],
        stats.norm.cdf(1, mean, sd) - stats.norm.cdf(-1, mean, sd),
        rel_tol=1e-9,
    )


# ---------------------------------------------------------------------------
# radiant.basics parity (values checked against R radiant.basics 2026-10-07)
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from pyrsm.basics.probability_calculator import prob_calc  # noqa: E402


def _summary(capsys, dist, **kwargs):
    prob_calc(dist, **kwargs).summary(dec=3)
    return capsys.readouterr().out


def test_lnorm_probability_bounds(capsys):
    out = _summary(capsys, "lnorm", meanlog=0, sdlog=1, plb=0.025, pub=0.975)
    assert "Lower bound : 0.025 (0.141)" in out
    assert "P(0.141 < X < 7.099) =" in out
    assert "1 - P(0.141 < X < 7.099) = 0.05" in out


def test_poisson_probability_bounds_give_discrete_table(capsys):
    # R: summary(prob_pois(lambda = 2, plb = .1, pub = .9), type = "probs")
    out = _summary(capsys, "pois", lamb=2, plb=0.1, pub=0.9)
    assert "Upper bound : 0.9 (4)" in out
    for line in [
        "P(X  = 0) =", "P(X  < 4) =          0.857", "P(X <= 4) =          0.947",
        "P(X >= 4) =          0.143", "P(0 <= X <= 4) =     0.947",
    ]:
        assert line in out


def test_poisson_float_value_bounds():
    # form inputs arrive as floats
    res = prob_calc("pois", lamb=2, lb=1.0, ub=3.0).dct
    assert res["p_int"] == pytest.approx(sum(stats.poisson.pmf(k, 2) for k in range(1, 4)))


def test_discrete_text_inputs_recycle_probabilities(capsys):
    out = _summary(capsys, "disc", v="1 2 3 4 5 6", p="1/6", lb=2.0, ub=5.0)
    assert "Values       : 1 2 3 4 5 6" in out
    assert "P(X = 2) =" in out and "P(2 <= X <= 5) =     0.667" in out


def test_discrete_does_not_mutate_the_callers_list():
    p = [0.5]
    prob_calc("disc", v=[1, 2], p=p, lb=1)
    assert p == [0.5]


def test_binomial_quantiles_print_as_integers(capsys):
    out = _summary(capsys, "binom", n=10, p=0.2, plb=0.1, pub=0.9)
    assert "Mean        : 2\n" in out
    assert "Lower bound : 0.1 (0)" in out and "P(X  = 4) =" in out


def test_one_minus_interval_is_a_probability_line(capsys):
    out = _summary(capsys, "norm", mean=0, sd=1, lb=-1.0, ub=1.0)
    head, probs = out.split("\n\n", 1)
    assert "1 - P(" not in head
    assert "1 - P(-1 < X < 1) = 0.317" in probs


@pytest.mark.parametrize(
    "dist,kwargs,message",
    [
        ("binom", dict(n=10, p=1.5), "p must be between 0 and 1"),
        ("norm", dict(mean=0, sd=0), "St. dev must be positive"),
        ("lnorm", dict(meanlog=0, sdlog=-1), "St. dev log must be positive"),
        ("tdist", dict(df=0), "Degrees of freedom must be positive"),
        ("expo", dict(rate=0), "Rate must be positive"),
        ("disc", dict(v="1 2", p="0.2 0.5"), "do not sum to 1"),
        ("pois", dict(lamb=2, pub=1.0), "smaller than 1"),
    ],
)
def test_invalid_parameters_raise(dist, kwargs, message):
    with pytest.raises(ValueError, match=message):
        prob_calc(dist, **kwargs)
