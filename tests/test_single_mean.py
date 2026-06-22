import numpy as np
import pandas as pd
import polars as pl

from pyrsm.basics.single_mean import single_mean


def test_single_mean_pandas_basic():
    df = pd.DataFrame({"values": [1, 2, 3, 4, 5]})
    sm = single_mean(df, var="values", comp_value=3)

    assert sm.n == 5
    assert sm.n_missing == 0
    assert sm.mean == 3
    assert sm.diff == 0
    assert sm.df == 4
    assert sm.t_val == 0
    assert sm.p_val == 1
    assert sm.me > 0


def test_single_mean_polars_with_missing():
    df = pl.DataFrame({"values": [10.0, 11.0, None, 9.0]})
    sm = single_mean(df, var="values", comp_value=10)

    assert sm.n == 4
    assert sm.n_missing == 1
    assert sm.mean == 10.0
    assert sm.diff == 0
    assert np.isfinite(sm.sd)
    assert np.isfinite(sm.se)
    assert sm.df == 2


def test_single_mean_alt_hyp_greater():
    df = pd.DataFrame({"values": [5, 6, 7, 8, 9]})
    sm = single_mean(df, var="values", comp_value=4, alt_hyp="greater", conf=0.9)

    assert sm.mean == 7
    assert sm.diff == 3
    assert sm.p_val < 0.01
    assert sm.t_val > 0


def test_single_mean_simulate_means_centered_on_comp_value():
    rng = np.random.default_rng(42)
    df = pl.DataFrame({"x": rng.normal(5.0, 2.0, 500)})
    sm = single_mean(df, var="x", comp_value=4.0, alt_hyp="two-sided", conf=0.95)

    sims, cutoffs = sm._simulate_means(nsim=2000)
    assert sims.shape == (2000,)
    # Simulated means are recentred on the comparison value (the null).
    assert abs(float(sims.mean()) - 4.0) < 1e-9
    # Two-sided alternative → two percentile cutoffs straddling comp_value.
    assert len(cutoffs) == 2
    assert cutoffs[0] < 4.0 < cutoffs[1]


def test_single_mean_simulate_cutoffs_match_alternative():
    rng = np.random.default_rng(7)
    df = pl.DataFrame({"x": rng.normal(0.0, 1.0, 300)})
    less = single_mean(df, var="x", comp_value=0.0, alt_hyp="less")
    greater = single_mean(df, var="x", comp_value=0.0, alt_hyp="greater")

    assert len(less._simulate_means()[1]) == 1
    assert len(greater._simulate_means()[1]) == 1


def test_single_mean_sim_plot_returns_object():
    df = pd.DataFrame({"values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    sm = single_mean(df, var="values", comp_value=3.0)
    plot = sm.plot(plots="sim")
    assert plot is not None
    assert plot.__class__.__name__ == "ggplot"
