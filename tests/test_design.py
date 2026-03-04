import math

import numpy as np
import polars as pl
import pytest

from pyrsm.design.doe import doe
from pyrsm.design.randomizer import randomizer
from pyrsm.design.sample_size import sample_size
from pyrsm.design.sample_size_comp import sample_size_comp
from pyrsm.design.sampling import sampling

# Load rndnames dataset for sampling/randomizer tests
rndnames = pl.read_parquet("examples/data/design/rndnames.parquet")


# ============================================================
# sample_size tests
# ============================================================


class TestSampleSize:
    def test_mean_basic(self):
        """R-validated: sample_size('mean', err_mean=2, sd_mean=10).n == 97"""
        ss = sample_size("mean", err_mean=2, sd_mean=10)
        assert ss.n == 97

    def test_proportion_basic(self):
        ss = sample_size("proportion", err_prop=0.1, p_prop=0.5)
        assert ss.n == 97

    def test_proportion_custom(self):
        ss = sample_size("proportion", err_prop=0.05, p_prop=0.3, conf=0.99)
        assert ss.n > 0
        # p=0.3 should give smaller n than p=0.5 for same error
        ss2 = sample_size("proportion", err_prop=0.05, p_prop=0.5, conf=0.99)
        assert ss.n < ss2.n

    def test_population_correction(self):
        ss_no_corr = sample_size("mean", err_mean=2, sd_mean=10, pop_correction=False)
        ss_corr = sample_size(
            "mean", err_mean=2, sd_mean=10, pop_correction=True, pop_size=500
        )
        # With population correction, n should be smaller
        assert ss_corr.n < ss_no_corr.n

    def test_incidence_response(self):
        ss = sample_size("mean", err_mean=2, sd_mean=10, incidence=0.5, response=0.5)
        assert ss.n == 97
        # Contact attempts should be 4x the sample size (1/0.5/0.5)
        assert ss.contact_attempts == math.ceil(97 / 0.5 / 0.5)

    def test_summary_runs(self):
        ss = sample_size("mean", err_mean=2, sd_mean=10)
        ss.summary()  # Should not raise

    def test_summary_proportion_runs(self):
        ss = sample_size("proportion", err_prop=0.1, p_prop=0.5)
        ss.summary()

    def test_summary_pop_correction_runs(self):
        ss = sample_size(
            "mean", err_mean=2, sd_mean=10, pop_correction=True, pop_size=1000
        )
        ss.summary()


# ============================================================
# sample_size_comp tests
# ============================================================


class TestSampleSizeComp:
    def test_proportion_solve_n_less(self):
        """R-validated: p1=0.008, p2=0.01, conf=0.95, power=0.9, alt='less' -> n1=n2=38073"""
        ssc = sample_size_comp(
            "proportion",
            p1=0.008,
            p2=0.01,
            conf=0.95,
            power=0.9,
            alt_hyp="less",
        )
        assert math.ceil(ssc.n1) == 38073
        assert math.ceil(ssc.n2) == 38073

    def test_proportion_solve_n_greater(self):
        """R-validated: p1=0.064, p2=0.035, alt='greater' -> total=1356"""
        ssc = sample_size_comp(
            "proportion",
            p1=0.064,
            p2=0.035,
            conf=0.95,
            power=0.8,
            alt_hyp="greater",
        )
        assert math.ceil(ssc.n1) == 678
        assert math.ceil(ssc.n2) == 678
        assert math.ceil(ssc.n1) + math.ceil(ssc.n2) == 1356

    def test_proportion_solve_n2(self):
        """R-validated: given n1=38073, solve n2 -> ceil(n2)==38073"""
        ssc = sample_size_comp(
            "proportion",
            n1=38073,
            p1=0.008,
            p2=0.01,
            conf=0.95,
            power=0.9,
            alt_hyp="less",
        )
        assert math.ceil(ssc.n2) == 38073

    def test_proportion_solve_power(self):
        """R-validated: given n1=n2=38073, conf=0.95 -> power ~ 0.9"""
        ssc = sample_size_comp(
            "proportion",
            n1=38073,
            n2=38073,
            p1=0.008,
            p2=0.01,
            conf=0.95,
            alt_hyp="less",
        )
        assert round(ssc.power, 1) == 0.9

    def test_proportion_solve_conf(self):
        """R-validated: given n1=n2=38073, power=0.9 -> conf ~ 0.95"""
        ssc = sample_size_comp(
            "proportion",
            n1=38073,
            n2=38073,
            p1=0.008,
            p2=0.01,
            power=0.9,
            alt_hyp="less",
        )
        assert round(ssc.conf, 2) == 0.95

    def test_proportion_two_sided(self):
        """Two-sided needs larger n than one-sided for same effect."""
        ssc = sample_size_comp(
            "proportion", p1=0.1, p2=0.15, conf=0.95, power=0.8, alt_hyp="two-sided"
        )
        ssc_one = sample_size_comp(
            "proportion", p1=0.1, p2=0.15, conf=0.95, power=0.8, alt_hyp="less"
        )
        assert ssc.n1 is not None
        assert math.ceil(ssc.n1) > math.ceil(ssc_one.n1)

    def test_proportion_less_and_greater_same_n(self):
        """One-sided 'less' and 'greater' give same n for same effect magnitude."""
        ssc_less = sample_size_comp(
            "proportion", p1=0.035, p2=0.064, conf=0.95, power=0.8, alt_hyp="less"
        )
        ssc_greater = sample_size_comp(
            "proportion", p1=0.064, p2=0.035, conf=0.95, power=0.8, alt_hyp="greater"
        )
        assert math.ceil(ssc_less.n1) == math.ceil(ssc_greater.n1)

    def test_error_less_p1_greater_than_p2(self):
        """alt_hyp='less' requires p1 < p2."""
        ssc = sample_size_comp(
            "proportion", p1=0.064, p2=0.035, conf=0.95, power=0.8, alt_hyp="less"
        )
        assert ssc.error is not None
        assert "smaller" in ssc.error

    def test_error_greater_p1_less_than_p2(self):
        """alt_hyp='greater' requires p1 > p2."""
        ssc = sample_size_comp(
            "proportion", p1=0.035, p2=0.064, conf=0.95, power=0.8, alt_hyp="greater"
        )
        assert ssc.error is not None
        assert "larger" in ssc.error

    def test_error_equal_proportions(self):
        ssc = sample_size_comp("proportion", p1=0.5, p2=0.5, conf=0.95, power=0.8)
        assert ssc.error is not None

    def test_mean_solve_n(self):
        ssc = sample_size_comp(
            "mean", delta=5, sd=10, conf=0.95, power=0.8, alt_hyp="two-sided"
        )
        assert ssc.n1 is not None
        assert math.ceil(ssc.n1) > 0

    def test_mean_solve_delta(self):
        ssc = sample_size_comp(
            "mean", n1=64, n2=64, sd=10, conf=0.95, power=0.8, alt_hyp="two-sided"
        )
        assert ssc.delta is not None
        assert ssc.delta > 0

    def test_mean_solve_sd(self):
        ssc = sample_size_comp(
            "mean",
            n1=64,
            n2=64,
            delta=5,
            conf=0.95,
            power=0.8,
            alt_hyp="two-sided",
        )
        assert ssc.sd is not None
        assert ssc.sd > 0

    def test_mean_solve_power(self):
        ssc = sample_size_comp(
            "mean", n1=64, n2=64, delta=5, sd=10, conf=0.95, alt_hyp="two-sided"
        )
        assert ssc.power is not None
        assert 0 < ssc.power < 1

    def test_mean_solve_conf(self):
        ssc = sample_size_comp(
            "mean", n1=64, n2=64, delta=5, sd=10, power=0.8, alt_hyp="two-sided"
        )
        assert ssc.conf is not None
        assert 0 < ssc.conf < 1

    def test_summary_runs(self):
        ssc = sample_size_comp(
            "proportion", p1=0.1, p2=0.15, conf=0.95, power=0.8
        )
        ssc.summary()  # Should not raise


# ============================================================
# sampling tests
# ============================================================


class TestSampling:
    def test_basic_sampling(self):
        s = sampling(rndnames, vars=["Names"], sample_size=10, seed=1234)
        assert s.selected.shape[0] == 10
        assert "rnd_number" in s.selected.columns

    def test_seed_reproducibility(self):
        s1 = sampling(rndnames, vars=["Names"], sample_size=10, seed=42)
        s2 = sampling(rndnames, vars=["Names"], sample_size=10, seed=42)
        assert s1.selected["Names"].to_list() == s2.selected["Names"].to_list()

    def test_different_seed_different_sample(self):
        s1 = sampling(rndnames, vars=["Names"], sample_size=10, seed=1)
        s2 = sampling(rndnames, vars=["Names"], sample_size=10, seed=2)
        assert s1.selected["Names"].to_list() != s2.selected["Names"].to_list()

    def test_vars_selection(self):
        s = sampling(rndnames, vars=["Names"], sample_size=5, seed=1234)
        # Should only have Names + rnd_number columns
        assert set(s.selected.columns) == {"Names", "rnd_number"}

    def test_all_vars(self):
        s = sampling(rndnames, sample_size=5, seed=1234)
        assert "Names" in s.selected.columns
        assert "Gender" in s.selected.columns

    def test_dict_input(self):
        s = sampling({"rndnames": rndnames}, vars=["Names"], sample_size=5, seed=1234)
        assert s.name == "rndnames"
        assert s.selected.shape[0] == 5

    def test_summary_runs(self):
        s = sampling(rndnames, vars=["Names"], sample_size=10, seed=1234)
        s.summary()  # Should not raise


# ============================================================
# randomizer tests
# ============================================================


class TestRandomizer:
    def test_simple_assignment(self):
        r = randomizer(rndnames, vars=["Names"], seed=1234)
        assert r.data.shape[0] == 100
        assert ".conditions" in r.data.columns

    def test_simple_assignment_balance(self):
        r = randomizer(rndnames, vars=["Names"], conditions=["A", "B"], seed=1234)
        counts = r.data[".conditions"].value_counts()
        a_count = counts.filter(pl.col(".conditions") == "A")["count"][0]
        b_count = counts.filter(pl.col(".conditions") == "B")["count"][0]
        assert a_count == 50
        assert b_count == 50

    def test_three_conditions(self):
        r = randomizer(
            rndnames,
            vars=["Names"],
            conditions=["A", "B", "C"],
            seed=1234,
        )
        counts = r.data[".conditions"].value_counts()
        # With 100 rows and 3 equal conditions, should be ~33/33/34
        for row in counts.iter_rows(named=True):
            assert 32 <= row["count"] <= 34

    def test_block_assignment(self):
        r = randomizer(
            rndnames,
            vars=["Names"],
            blocks="Gender",
            conditions=["test", "control"],
            seed=1234,
        )
        assert r.data.shape[0] == 100
        # Check balance within blocks
        for gender in ["Female", "Male"]:
            block_data = r.data.filter(pl.col("Gender") == gender)
            counts = block_data[".conditions"].value_counts()
            test_count = counts.filter(pl.col(".conditions") == "test")["count"][0]
            control_count = counts.filter(pl.col(".conditions") == "control")["count"][0]
            assert test_count == 25
            assert control_count == 25

    def test_custom_probs(self):
        r = randomizer(
            rndnames,
            vars=["Names"],
            conditions=["A", "B"],
            probs=[0.7, 0.3],
            seed=1234,
        )
        counts = r.data[".conditions"].value_counts()
        a_count = counts.filter(pl.col(".conditions") == "A")["count"][0]
        # Should be roughly 70
        assert 65 <= a_count <= 75

    def test_reproducibility(self):
        r1 = randomizer(rndnames, vars=["Names"], seed=42)
        r2 = randomizer(rndnames, vars=["Names"], seed=42)
        assert r1.data[".conditions"].to_list() == r2.data[".conditions"].to_list()

    def test_dict_input(self):
        r = randomizer({"rndnames": rndnames}, vars=["Names"], seed=1234)
        assert r.name == "rndnames"

    def test_summary_runs(self):
        r = randomizer(rndnames, vars=["Names"], seed=1234)
        r.summary()  # Should not raise

    def test_block_summary_runs(self):
        r = randomizer(rndnames, vars=["Names"], blocks="Gender", seed=1234)
        r.summary()  # Should not raise


# ============================================================
# doe tests
# ============================================================


_3x3 = {"price": ["$10", "$13", "$16"], "food": ["popcorn", "gourmet", "no food"]}
_3x2x2 = {"apr": ["14.9", "16.8", "19.8"], "annual_fee": ["20", "0"], "fixed_var": ["Fixed", "Variable"]}


class TestDoe:
    def test_basic_3x3(self):
        """R-validated: 9 trials -> D-efficiency==1.0, Balanced==True"""
        d = doe(_3x3)
        row_9 = d.eff.filter(pl.col("Trials") == 9)
        assert row_9["D-efficiency"][0] == 1.0
        assert row_9["Balanced"][0] is True

    def test_full_factorial_dea(self):
        d = doe(_3x3)
        assert d.Dea == 1.0

    def test_partial_factorial(self):
        d = doe(_3x3, trials=6, seed=1234)
        assert d.part.height == 6
        assert d.Dea > 0
        assert d.Dea < 1.0

    def test_partial_d_efficiency_positive(self):
        d = doe(_3x3, trials=6, seed=1234)
        assert d.Dea > 0.4  # R-compatible Dea ~0.449 for 3x3 with 6 trials

    def test_balance_enforcement(self):
        """Verify that when balance is possible, the design is actually balanced."""
        d = doe(_3x2x2, trials=6, seed=1234)
        # R-validated: Dea ~ 0.819, design is balanced
        assert abs(d.Dea - 0.819) < 0.01
        row = d.eff.filter(pl.col("Trials") == 6)
        assert row["Balanced"][0] is True
        # Verify actual balance: each level appears equally
        assert d.part["apr"].value_counts()["count"].to_list() == [2, 2, 2]
        assert d.part["annual_fee"].value_counts()["count"].to_list() == [3, 3]
        assert d.part["fixed_var"].value_counts()["count"].to_list() == [3, 3]

    def test_interactions(self):
        d = doe(_3x3, interactions="price:food", trials=9, seed=1234)
        assert d.Dea == 1.0

    def test_interactions_credit(self):
        """R-validated: 3x2x2x3 with 3 interactions, 18 trials -> Dea=0.819, balanced"""
        factors_credit = {
            "apr": ["14.9", "16.8", "19.8"],
            "annual_fee": ["20", "0"],
            "fixed_var": ["Fixed", "Variable"],
            "bk_score": ["150", "200", "250"],
        }
        d = doe(
            factors_credit,
            interactions=["apr:bk_score", "annual_fee:bk_score", "fixed_var:bk_score"],
            trials=18,
            seed=1234,
        )
        assert d.full.height == 36
        assert abs(d.Dea - 0.819) < 0.01
        row = d.eff.filter(pl.col("Trials") == 18)
        assert row["Balanced"][0] is True

    def test_estimable_full(self):
        d = doe(_3x3)
        effects = d.estimable()
        assert len(effects) > 0
        assert any("price" in e for e in effects)
        assert any("food" in e for e in effects)

    def test_estimable_partial(self):
        d = doe(_3x3, trials=6, seed=1234)
        effects = d.estimable()
        assert len(effects) > 0

    def test_too_few_factors(self):
        d = doe({"price": ["$10", "$13"]})
        assert d.error is not None

    def test_summary_runs(self):
        d = doe(_3x3)
        d.summary()  # Should not raise

    def test_summary_partial_runs(self):
        d = doe(_3x3, trials=6, seed=1234)
        d.summary()  # Should not raise

    def test_three_factors(self):
        d = doe({"A": ["1", "2"], "B": ["x", "y"], "C": ["low", "high"]})
        assert d.full.height == 8  # 2*2*2
        assert d.Dea == 1.0

    def test_full_height(self):
        d = doe(_3x3)
        assert d.full.height == 9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
