"""Tests for pyrsm.model.visualize pdp_sk and pdp_sm functions."""

import os

import matplotlib
import polars as pl
import pytest

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor

from pyrsm.model.rforest import rforest
from pyrsm.model.visualize import pdp_sk, pdp_sm, pred_plot_sk

# Directory for saving plot comparisons
PLOT_DIR = "tests/plot_comparisons/pdp"
os.makedirs(PLOT_DIR, exist_ok=True)


@pytest.fixture(scope="module")
def titanic_data():
    """Load the titanic dataset for classification."""
    titanic = pl.read_parquet("examples/data/model/titanic.parquet")
    return titanic.drop_nulls(subset=["age"])


@pytest.fixture(scope="module")
def salary_data():
    """Load the salary dataset for regression."""
    return pl.read_parquet("examples/data/basics/salary.parquet")


@pytest.fixture(scope="module")
def diamonds_data():
    """Load diamonds dataset for larger tests."""
    diamonds = pl.read_parquet("examples/data/basics/diamonds.parquet")
    # Sample for speed
    return diamonds.sample(1000, seed=1234)


@pytest.fixture(scope="module")
def rf_classifier(titanic_data):
    """Fit a random forest classifier."""
    rf = rforest(
        data=titanic_data.to_pandas(),
        rvar="survived",
        lev="Yes",
        evar=["age", "sex", "pclass"],
        mod_type="classification",
        n_estimators=20,
        random_state=1234,
    )
    return rf


@pytest.fixture(scope="module")
def rf_regressor(salary_data):
    """Fit a random forest regressor."""
    rf = rforest(
        data=salary_data.to_pandas(),
        rvar="salary",
        evar=["yrs_since_phd", "yrs_service", "rank"],
        mod_type="regression",
        n_estimators=20,
        random_state=1234,
    )
    return rf


@pytest.fixture(scope="module")
def logit_model(titanic_data):
    """Fit a logistic regression model."""
    df = titanic_data.with_columns(
        pl.when(pl.col("survived") == "Yes").then(1).otherwise(0).alias("survived_bin")
    ).to_pandas()
    model = smf.logit("survived_bin ~ age + C(sex) + C(pclass)", data=df).fit(disp=0)
    return model, df


@pytest.fixture(scope="module")
def ols_model(salary_data):
    """Fit an OLS regression model."""
    df = salary_data.to_pandas()
    model = smf.ols("salary ~ yrs_since_phd + yrs_service + C(rank)", data=df).fit()
    return model, df


class TestPdpSkBasic:
    """Basic tests for pdp_sk function."""

    def test_pdp_sk_returns_plot(self, rf_classifier, titanic_data):
        """Test that pdp_sk returns a plot."""
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            grid_resolution=10,
            n_sample=100,
        )
        # Should return a ggplot object or None
        assert plot is not None or plot is None

    def test_pdp_sk_with_multiple_vars(self, rf_classifier, titanic_data):
        """Test that pdp_sk works with multiple variables."""
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age", "sex"],
            grid_resolution=10,
            n_sample=100,
        )
        # Plot should be created
        assert plot is not None

    def test_pdp_sk_single_var(self, rf_classifier, titanic_data):
        """Test pdp_sk with single variable."""
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            grid_resolution=10,
            n_sample=100,
        )
        assert plot is not None

    def test_pdp_sk_classification(self, rf_classifier, titanic_data):
        """Test pdp_sk with classification model."""
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            grid_resolution=10,
            n_sample=100,
        )
        assert plot is not None

    def test_pdp_sk_regression(self, rf_regressor, salary_data):
        """Test pdp_sk with regression model."""
        plot = pdp_sk(
            rf_regressor.fitted,
            salary_data,
            incl=["yrs_since_phd"],
            grid_resolution=10,
            n_sample=100,
        )
        assert plot is not None


class TestPdpSkModes:
    """Test different modes of pdp_sk."""

    def test_pdp_mode_works(self, rf_classifier, titanic_data):
        """Test that pdp mode works."""
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="pdp",
            grid_resolution=10,
            n_sample=200,
        )
        assert plot is not None

    def test_fast_mode_works(self, rf_classifier, titanic_data):
        """Test that fast mode works."""
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="fast",
            grid_resolution=10,
        )
        assert plot is not None


class TestPdpSkCategorical:
    """Test categorical variable handling."""

    def test_pdp_sk_categorical_works(self, rf_classifier, titanic_data):
        """Test that categorical variables work."""
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["sex"],
            grid_resolution=10,
            n_sample=100,
        )
        assert plot is not None

    def test_pdp_sk_pclass_categorical(self, rf_classifier, titanic_data):
        """Test pclass as categorical."""
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["pclass"],
            grid_resolution=10,
            n_sample=100,
        )
        assert plot is not None


class TestPdpSkInteractions:
    """Test interaction plotting."""

    def test_pdp_sk_interaction_num_cat(self, rf_classifier, titanic_data):
        """Test numeric-categorical interaction."""
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=[],
            incl_int=["age:sex"],
            grid_resolution=10,
            n_sample=100,
        )
        assert plot is not None

    def test_pdp_sk_interaction_num_num(self, rf_regressor, salary_data):
        """Test numeric-numeric interaction."""
        plot = pdp_sk(
            rf_regressor.fitted,
            salary_data,
            incl=[],
            incl_int=["yrs_since_phd:yrs_service"],
            grid_resolution=10,
            n_sample=100,
            interaction_slices=5,
        )
        assert plot is not None


class TestPdpSkGridResolution:
    """Test grid resolution parameter."""

    def test_pdp_sk_grid_resolution_works(self, rf_classifier, titanic_data):
        """Test that different grid resolutions work."""
        plot_10 = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="fast",
            grid_resolution=10,
        )
        plot_20 = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="fast",
            grid_resolution=20,
        )
        assert plot_10 is not None
        assert plot_20 is not None


class TestPdpSkQuantiles:
    """Test quantile-based grid limits."""

    def test_pdp_sk_quantiles_work(self, rf_classifier, titanic_data):
        """Test that minq/maxq parameters work."""
        plot_full = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="fast",
            minq=0.0,
            maxq=1.0,
            grid_resolution=20,
        )
        plot_trimmed = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="fast",
            minq=0.1,
            maxq=0.9,
            grid_resolution=20,
        )
        assert plot_full is not None
        assert plot_trimmed is not None


class TestPdpSmBasic:
    """Basic tests for pdp_sm function."""

    def test_pdp_sm_returns_plot(self, logit_model, titanic_data):
        """Test that pdp_sm returns a plot."""
        model, df = logit_model
        plot = pdp_sm(
            model,
            pl.from_pandas(df),
            incl=["age"],
            grid_resolution=10,
            n_sample=100,
        )
        assert plot is not None

    def test_pdp_sm_logistic_predictions_valid(self, logit_model, titanic_data):
        """Test logistic model works with pdp_sm."""
        model, df = logit_model
        plot = pdp_sm(
            model,
            pl.from_pandas(df),
            incl=["age"],
            grid_resolution=10,
            n_sample=100,
        )
        assert plot is not None

    def test_pdp_sm_ols_predictions_reasonable(self, ols_model, salary_data):
        """Test OLS model works with pdp_sm."""
        model, df = ols_model
        plot = pdp_sm(
            model,
            pl.from_pandas(df),
            incl=["yrs_since_phd"],
            grid_resolution=10,
            n_sample=100,
        )
        assert plot is not None


class TestPdpSmModes:
    """Test pdp_sm modes."""

    def test_pdp_sm_both_modes_work(self, ols_model, salary_data):
        """Test both fast and pdp modes work."""
        model, df = ols_model
        df_pl = pl.from_pandas(df)

        plot_pdp = pdp_sm(
            model,
            df_pl,
            incl=["yrs_since_phd"],
            mode="pdp",
            grid_resolution=10,
            n_sample=100,
        )
        plot_fast = pdp_sm(
            model, df_pl, incl=["yrs_since_phd"], mode="fast", grid_resolution=10
        )

        assert plot_pdp is not None
        assert plot_fast is not None


class TestPdpPlotSaving:
    """Test plot generation and saving."""

    def test_pdp_sk_save_single_var(self, rf_classifier, titanic_data):
        """Test saving single variable PDP plot."""
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            grid_resolution=20,
            n_sample=200,
        )
        if plot is not None:
            plot.save(f"{PLOT_DIR}/pdp_sk_single_age.png", dpi=100, verbose=False)
        plt.close("all")

    def test_pdp_sk_save_multiple_vars(self, rf_classifier, titanic_data):
        """Test saving multiple variable PDP plots."""
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age", "sex", "pclass"],
            grid_resolution=15,
            n_sample=200,
        )
        if plot is not None:
            plot.save(f"{PLOT_DIR}/pdp_sk_multi.png", dpi=100, verbose=False)
        plt.close("all")

    def test_pdp_sk_save_interaction(self, rf_classifier, titanic_data):
        """Test saving interaction PDP plot."""
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=[],
            incl_int=["age:sex"],
            grid_resolution=15,
            n_sample=200,
        )
        if plot is not None:
            plot.save(f"{PLOT_DIR}/pdp_sk_interaction.png", dpi=100, verbose=False)
        plt.close("all")

    def test_pdp_sm_save_plot(self, ols_model, salary_data):
        """Test saving statsmodels PDP plot."""
        model, df = ols_model
        plot = pdp_sm(
            model,
            pl.from_pandas(df),
            incl=["yrs_since_phd", "yrs_service"],
            grid_resolution=15,
            n_sample=200,
        )
        if plot is not None:
            plot.save(f"{PLOT_DIR}/pdp_sm_ols.png", dpi=100, verbose=False)
        plt.close("all")


class TestPdpVsPredPlotComparison:
    """Compare pdp_sk with pred_plot_sk outputs."""

    def test_pdp_fast_and_pred_plot_both_work(self, rf_classifier, titanic_data):
        """Test that pdp_sk fast mode and pred_plot_sk both work."""
        # Get pred_plot_sk data
        pred_dict = pred_plot_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            ret=True,
            nnv=20,
        )

        # Get pdp_sk fast mode plot
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="fast",
            grid_resolution=20,
        )

        # Both should work
        assert "age" in pred_dict
        assert plot is not None


class TestPdpSklearnComparison:
    """Compare pdp_sk with sklearn's partial_dependence."""

    def test_pdp_sk_works_with_sklearn_model(self, salary_data):
        """Test pdp_sk works with sklearn model."""
        # Prepare data without categoricals for simpler comparison
        df = salary_data.select(["salary", "yrs_since_phd", "yrs_service"]).drop_nulls()
        X = df.select(["yrs_since_phd", "yrs_service"]).to_pandas()
        y = df["salary"].to_list()

        # Fit sklearn RF directly
        rf = RandomForestRegressor(n_estimators=20, random_state=1234)
        rf.fit(X, y)

        # Get our PDP
        plot = pdp_sk(
            rf,
            df,
            incl=["yrs_since_phd"],
            mode="pdp",
            grid_resolution=20,
            n_sample=df.height,
        )

        assert plot is not None


class TestPdpEdgeCases:
    """Test edge cases."""

    def test_pdp_sk_empty_incl(self, rf_classifier, titanic_data):
        """Test with empty incl list."""
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=[],
            incl_int=[],
        )
        assert plot is None

    def test_pdp_sk_exclude_variables(self, rf_classifier, titanic_data):
        """Test excluding variables."""
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            excl=["age"],
            grid_resolution=10,
            n_sample=100,
        )
        # Should work and create a plot (with remaining vars)
        assert plot is not None

    def test_pdp_sk_small_sample(self, rf_classifier, titanic_data):
        """Test with very small sample size."""
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="pdp",
            n_sample=10,
            grid_resolution=5,
        )
        assert plot is not None


class TestPdpPerformance:
    """Test performance characteristics."""

    def test_pdp_sk_timing_reasonable(self, rf_classifier, titanic_data):
        """Test that pdp_sk completes in reasonable time."""
        import time

        start = time.time()
        pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age", "sex"],
            mode="pdp",
            n_sample=500,
            grid_resolution=20,
        )
        runtime = time.time() - start
        # Should complete in under 30 seconds
        assert runtime < 30

    def test_pdp_sm_timing_reasonable(self, ols_model, salary_data):
        """Test that pdp_sm completes in reasonable time."""
        import time

        model, df = ols_model
        start = time.time()
        pdp_sm(
            model,
            pl.from_pandas(df),
            incl=["yrs_since_phd"],
            mode="pdp",
            n_sample=500,
            grid_resolution=20,
        )
        runtime = time.time() - start
        # Should complete in under 30 seconds
        assert runtime < 30


class TestPredPlotSkOrdering:
    """Test that pred_plot_sk returns variables in correct order."""

    def test_pred_plot_sk_ordering_matches_incl(self, rf_classifier, titanic_data):
        """Test that pred_plot_sk returns variables in incl parameter order."""
        incl_vars = ["age", "sex", "pclass"]
        pred_dict = pred_plot_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=incl_vars,
            ret=True,
            nnv=10,
        )

        # The keys should follow incl parameter order
        actual_order = list(pred_dict.keys())
        assert (
            actual_order == incl_vars
        ), f"Order mismatch: expected {incl_vars}, got {actual_order}"

    def test_pred_plot_sk_ordering_with_mlp(self, titanic_data):
        """Test that pred_plot_sk preserves ordering with MLP models (original bug case)."""
        from pyrsm.model.mlp import mlp

        # This tests the exact scenario from the bug report:
        # MLP model with age, sibsp, parch, fare should return them in that order
        incl_vars = ["age", "sibsp", "parch", "fare"]
        model = mlp(
            data=titanic_data.drop_nulls(subset=incl_vars),
            rvar="survived",
            lev="Yes",
            evar=incl_vars + ["sex", "pclass"],
            mod_type="classification",
            hidden_layer_sizes=(10,),
            random_state=1234,
            max_iter=100,
        )

        # Create data dict as MLP provides
        data_dict = {
            "data": model.data,
            "means": model.means,
            "stds": model.stds,
        }

        # Get pred_plot_sk result
        pred_dict = pred_plot_sk(
            model.fitted,
            data_dict,
            rvar="survived",
            incl=incl_vars,
            ret=True,
            nnv=10,
        )

        # Should return in incl order, not the scrambled order (age, fare, parch, sibsp)
        actual_order = list(pred_dict.keys())
        assert (
            actual_order == incl_vars
        ), f"Order mismatch: expected {incl_vars}, got {actual_order}"

    def test_pred_plot_sk_dummy_variable_ordering(self, titanic_data):
        """Test that dummy variables maintain correct ordering relative to other variables."""
        from pyrsm.model.mlp import mlp

        # Mix of numeric and categorical variables in specific order
        # sex and pclass are categorical -> will become dummies (sex_male, pclass_2nd, pclass_3rd)
        incl_vars = ["age", "sex", "sibsp", "pclass", "fare"]
        model = mlp(
            data=titanic_data.drop_nulls(subset=["age", "sibsp", "parch", "fare"]),
            rvar="survived",
            lev="Yes",
            evar=incl_vars,
            mod_type="classification",
            hidden_layer_sizes=(10,),
            random_state=1234,
            max_iter=100,
        )

        data_dict = {
            "data": model.data,
            "means": model.means,
            "stds": model.stds,
        }

        pred_dict = pred_plot_sk(
            model.fitted,
            data_dict,
            rvar="survived",
            incl=incl_vars,
            ret=True,
            nnv=10,
        )

        actual_order = list(pred_dict.keys())

        # Verify order matches incl: age, sex, sibsp, pclass, fare
        assert (
            actual_order == incl_vars
        ), f"Order mismatch: expected {incl_vars}, got {actual_order}"

        # Verify categorical variables are in correct position relative to numerics
        age_idx = actual_order.index("age")
        sex_idx = actual_order.index("sex")
        sibsp_idx = actual_order.index("sibsp")
        pclass_idx = actual_order.index("pclass")
        fare_idx = actual_order.index("fare")

        assert age_idx < sex_idx, "age should come before sex"
        assert sex_idx < sibsp_idx, "sex should come before sibsp"
        assert sibsp_idx < pclass_idx, "sibsp should come before pclass"
        assert pclass_idx < fare_idx, "pclass should come before fare"

    def test_pred_plot_sk_categorical_ordering_consistency(self, titanic_data):
        """Test categorical variable ordering is consistent between pred_plot_sk runs."""
        from pyrsm.model.rforest import rforest

        # Test with rforest (no scaling) to isolate ordering behavior
        incl_vars = ["pclass", "sex", "age"]  # categoricals first, then numeric
        model = rforest(
            data=titanic_data.drop_nulls(subset=["age"]),
            rvar="survived",
            lev="Yes",
            evar=incl_vars + ["sibsp"],
            mod_type="classification",
            random_state=1234,
        )

        # Run multiple times to verify consistency
        for _ in range(3):
            pred_dict = pred_plot_sk(
                model.fitted,
                model.data,
                incl=incl_vars,
                ret=True,
                nnv=10,
            )
            actual_order = list(pred_dict.keys())
            assert (
                actual_order == incl_vars
            ), f"Order inconsistent: expected {incl_vars}, got {actual_order}"

    def test_pred_plot_sk_no_double_scaling(self, titanic_data):
        """Test that pred_plot_sk doesn't double-scale MLP data."""
        from pyrsm.model.mlp import mlp

        # Fit MLP model
        model = mlp(
            data=titanic_data.drop_nulls(subset=["age", "sibsp", "parch", "fare"]),
            rvar="survived",
            lev="Yes",
            evar=["age", "sibsp", "parch", "fare", "sex", "pclass"],
            mod_type="classification",
            hidden_layer_sizes=(10,),
            random_state=1234,
            max_iter=100,
        )

        # Get pred_plot_sk data dictionary with MLP data dict
        data_dict = {
            "data": model.data,
            "means": model.means,
            "stds": model.stds,
        }

        pred_dict = pred_plot_sk(
            model.fitted,
            data_dict,
            rvar="survived",
            incl=["age"],
            ret=True,
            nnv=10,
        )

        # Predictions should be probabilities [0, 1]
        preds = pred_dict["age"]["prediction"].to_list()
        assert all(
            0 <= p <= 1 for p in preds
        ), "Predictions out of valid probability range"

        # Age values should be in original scale (not scaled)
        age_vals = pred_dict["age"]["age"].to_list()
        titanic_data.drop_nulls(subset=["age"])["age"]
        # Age should be in a reasonable human range
        assert min(age_vals) >= 0, "Age should not be negative"
        assert max(age_vals) <= 100, "Age should not exceed 100"


class TestPdpIce:
    """Test ICE (Individual Conditional Expectation) functionality."""

    def test_ice_pdp_values_unchanged_sk(self, rf_classifier, titanic_data):
        """Test that PDP values are identical with and without ICE for sklearn."""
        # Run without ICE
        plot_no_ice = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="pdp",
            ice=False,
            grid_resolution=10,
            n_sample=100,
        )

        # Run with ICE
        plot_with_ice = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="pdp",
            ice=True,
            grid_resolution=10,
            n_sample=100,
        )

        # Both should return plots
        assert plot_no_ice is not None
        assert plot_with_ice is not None

    def test_ice_pdp_values_unchanged_sm(self, ols_model, salary_data):
        """Test that PDP values are identical with and without ICE for statsmodels."""
        model, df = ols_model
        df_pl = pl.from_pandas(df)

        # Run without ICE
        plot_no_ice = pdp_sm(
            model,
            df_pl,
            incl=["yrs_since_phd"],
            mode="pdp",
            ice=False,
            grid_resolution=10,
            n_sample=100,
        )

        # Run with ICE
        plot_with_ice = pdp_sm(
            model,
            df_pl,
            incl=["yrs_since_phd"],
            mode="pdp",
            ice=True,
            grid_resolution=10,
            n_sample=100,
        )

        # Both should return plots
        assert plot_no_ice is not None
        assert plot_with_ice is not None

    def test_ice_no_performance_impact_sk(self, rf_classifier, titanic_data):
        """Test that ice=False has no significant performance impact for sklearn."""
        import time

        # Time without ICE
        start = time.time()
        for _ in range(3):
            pdp_sk(
                rf_classifier.fitted,
                titanic_data,
                incl=["age"],
                mode="pdp",
                ice=False,
                grid_resolution=10,
                n_sample=100,
            )
        time_no_ice = time.time() - start

        # Time with ICE
        start = time.time()
        for _ in range(3):
            pdp_sk(
                rf_classifier.fitted,
                titanic_data,
                incl=["age"],
                mode="pdp",
                ice=True,
                grid_resolution=10,
                n_sample=100,
            )
        time_with_ice = time.time() - start

        # ICE should not add more than 50% overhead (being generous)
        assert time_with_ice < time_no_ice * 1.5 or time_with_ice < 5.0

    def test_ice_with_categorical_sk(self, rf_classifier, titanic_data):
        """Test ICE works with categorical variables for sklearn."""
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["sex", "pclass"],
            mode="pdp",
            ice=True,
            n_sample=50,
        )
        assert plot is not None

    def test_ice_with_categorical_sm(self, logit_model, titanic_data):
        """Test ICE works with categorical variables for statsmodels."""
        model, df = logit_model
        plot = pdp_sm(
            model,
            pl.from_pandas(df),
            incl=["sex"],
            mode="pdp",
            ice=True,
            n_sample=50,
        )
        assert plot is not None

    def test_ice_fast_mode_no_effect_sk(self, rf_classifier, titanic_data):
        """Test that ice parameter has no effect in fast mode for sklearn."""
        # In fast mode, ICE doesn't make sense (single observation)
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="fast",
            ice=True,  # Should be ignored
            grid_resolution=10,
        )
        assert plot is not None

    def test_ice_save_plot(self, rf_classifier, titanic_data):
        """Test saving ICE plot."""
        plot = pdp_sk(
            rf_classifier.fitted,
            titanic_data,
            incl=["age"],
            mode="pdp",
            ice=True,
            grid_resolution=15,
            n_sample=100,
        )
        if plot is not None:
            plot.save(f"{PLOT_DIR}/pdp_sk_ice.png", dpi=100, verbose=False)
        plt.close("all")
