"""Tests for pyrsm.model.model utility functions."""

import numpy as np
import polars as pl
import pytest

from pyrsm.model.model import (
    coef_ci,
    convert_to_list,
    evalreg,
    get_dummies,
    make_train,
    model_fit,
    or_ci,
    sig_stars,
    sim_prediction,
    vif,
)


@pytest.fixture(scope="module")
def salary_data():
    """Load the salary dataset."""
    salary = pl.read_parquet("examples/data/basics/salary.parquet")
    return salary.to_pandas()


@pytest.fixture(scope="module")
def titanic_data():
    """Load the titanic dataset."""
    titanic = pl.read_parquet("examples/data/model/titanic.parquet")
    return titanic.drop_nulls(subset=["age"]).to_pandas()


class TestMakeTrain:
    """Tests for make_train function."""

    def test_make_train_basic(self, salary_data):
        """Test basic train/test split."""
        train_var = make_train(salary_data, test_size=0.2)
        assert len(train_var) == len(salary_data)
        # Should have approximately 80% training
        train_pct = train_var.mean()
        assert 0.7 < train_pct < 0.9

    def test_make_train_stratified(self, salary_data):
        """Test stratified train/test split."""
        train_var = make_train(salary_data, strat_var="sex", test_size=0.2)
        assert len(train_var) == len(salary_data)

    def test_make_train_polars_input(self):
        """Test with polars input."""
        salary = pl.read_parquet("examples/data/basics/salary.parquet")
        train_var = make_train(salary, test_size=0.2)
        assert len(train_var) == len(salary)

    def test_make_train_reproducible(self, salary_data):
        """Test that random_state makes it reproducible."""
        make_train(salary_data, test_size=0.2, random_state=42)
        make_train(salary_data, test_size=0.2, random_state=42)
        # Stratified splits should be identical
        train1_strat = make_train(
            salary_data, strat_var="sex", test_size=0.2, random_state=42
        )
        train2_strat = make_train(
            salary_data, strat_var="sex", test_size=0.2, random_state=42
        )
        assert np.array_equal(train1_strat, train2_strat)

    def test_make_train_multi_strat_vars(self, titanic_data):
        """Test stratified split with multiple stratification variables."""
        df = titanic_data.copy()
        train_var = make_train(
            df, strat_var=["sex", "pclass"], test_size=0.2, random_state=42
        )
        assert len(train_var) == len(df)
        # Check correct number of train/test
        train_pct = train_var.mean()
        assert 0.75 < train_pct < 0.85

    def test_make_train_multi_strat_preserves_proportions(self, titanic_data):
        """Test that multi-var stratification preserves group proportions."""
        df = titanic_data.copy()
        df_pl = pl.from_pandas(df)
        train_var = make_train(
            df, strat_var=["sex", "pclass"], test_size=0.2, random_state=42
        )
        df_pl = df_pl.with_columns(pl.Series("train", train_var))

        # For each combination, the train proportion should be close to 0.8
        for sex_val in df_pl["sex"].unique().to_list():
            for pclass_val in df_pl["pclass"].unique().to_list():
                mask = (df_pl["sex"] == sex_val) & (df_pl["pclass"] == pclass_val)
                group = df_pl.filter(mask)
                if len(group) >= 5:  # Only check groups with enough obs
                    group_train_pct = group["train"].mean()
                    assert 0.7 < group_train_pct < 0.9, (
                        f"Group sex={sex_val}, pclass={pclass_val} has "
                        f"train%={group_train_pct:.2f}, expected ~0.8"
                    )

    def test_make_train_multi_strat_reproducible(self, titanic_data):
        """Test reproducibility with multiple stratification variables."""
        df = titanic_data.copy()
        train1 = make_train(
            df, strat_var=["sex", "pclass"], test_size=0.2, random_state=99
        )
        train2 = make_train(
            df, strat_var=["sex", "pclass"], test_size=0.2, random_state=99
        )
        assert np.array_equal(train1, train2)

    def test_make_train_multi_strat_different_seeds(self, titanic_data):
        """Test that different random states produce different splits."""
        df = titanic_data.copy()
        train1 = make_train(
            df, strat_var=["sex", "pclass"], test_size=0.2, random_state=1
        )
        train2 = make_train(
            df, strat_var=["sex", "pclass"], test_size=0.2, random_state=2
        )
        assert not np.array_equal(train1, train2)

    def test_make_train_multi_strat_returns_int_series(self, titanic_data):
        """Test that result is a polars int Series with only 0/1 values."""
        df = titanic_data.copy()
        train_var = make_train(
            df, strat_var=["sex", "pclass"], test_size=0.3, random_state=42
        )
        assert isinstance(train_var, pl.Series)
        assert train_var.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64)
        unique_vals = set(train_var.to_list())
        assert unique_vals == {0, 1}

    def test_make_train_multi_strat_polars_input(self, titanic_data):
        """Test multi-var stratification with polars DataFrame input."""
        df_pl = pl.from_pandas(titanic_data)
        train_var = make_train(
            df_pl, strat_var=["sex", "pclass"], test_size=0.2, random_state=42
        )
        assert len(train_var) == len(df_pl)
        train_pct = train_var.mean()
        assert 0.75 < train_pct < 0.85

    def test_make_train_multi_strat_test_size(self, titanic_data):
        """Test that different test_size values are respected."""
        df = titanic_data.copy()
        for test_size in [0.1, 0.3, 0.5]:
            train_var = make_train(
                df, strat_var=["sex", "pclass"], test_size=test_size, random_state=42
            )
            train_pct = train_var.mean()
            expected = 1.0 - test_size
            assert abs(train_pct - expected) < 0.05, (
                f"test_size={test_size}: train%={train_pct:.3f}, "
                f"expected ~{expected:.1f}"
            )

    def test_make_train_three_strat_vars(self):
        """Test stratified split with three stratification variables."""
        np.random.seed(42)
        n = 500
        df = pl.DataFrame(
            {
                "a": np.random.choice(["x", "y"], n),
                "b": np.random.choice(["m", "f"], n),
                "c": np.random.choice(["low", "high"], n),
                "val": np.random.randn(n),
            }
        )
        train_var = make_train(
            df, strat_var=["a", "b", "c"], test_size=0.2, random_state=42
        )
        assert len(train_var) == n
        train_pct = train_var.mean()
        assert 0.75 < train_pct < 0.85

    def test_make_train_list_with_single_var(self, salary_data):
        """Test that passing a single var as a list works the same as a string."""
        train_list = make_train(
            salary_data, strat_var=["sex"], test_size=0.2, random_state=42
        )
        train_str = make_train(
            salary_data, strat_var="sex", test_size=0.2, random_state=42
        )
        assert np.array_equal(train_list, train_str)


class TestConvertToList:
    """Tests for convert_to_list function."""

    def test_convert_none(self):
        """Test converting None."""
        assert convert_to_list(None) == []

    def test_convert_string(self):
        """Test converting single string."""
        assert convert_to_list("test") == ["test"]

    def test_convert_list(self):
        """Test converting list (should return unchanged)."""
        assert convert_to_list(["a", "b"]) == ["a", "b"]

    def test_convert_tuple(self):
        """Test converting tuple."""
        result = convert_to_list(("a", "b"))
        assert result == ["a", "b"]


class TestSigStars:
    """Tests for sig_stars function."""

    def test_sig_stars_highly_significant(self):
        """Test very small p-values."""
        result = sig_stars([0.0001, 0.0005])
        assert result.to_list() == ["***", "***"]

    def test_sig_stars_significant(self):
        """Test significant p-values."""
        result = sig_stars([0.005, 0.03, 0.08])
        assert result.to_list() == ["**", "*", "."]

    def test_sig_stars_not_significant(self):
        """Test non-significant p-values."""
        result = sig_stars([0.5, 0.9])
        assert result.to_list() == [" ", " "]

    def test_sig_stars_nan(self):
        """Test handling NaN values."""
        result = sig_stars([np.nan, 0.005])
        assert result[0] == " "  # NaN treated as p=1
        assert result[1] == "**"


class TestGetDummies:
    """Tests for get_dummies function."""

    def test_get_dummies_basic(self, salary_data):
        """Test basic dummy variable creation."""
        df = salary_data[["salary", "rank", "sex"]].copy()
        result = get_dummies(df)
        # Numeric column should remain
        assert "salary" in result.columns
        # Categorical columns should be converted
        assert any("rank_" in col for col in result.columns)
        assert any("sex_" in col for col in result.columns)

    def test_get_dummies_drop_first(self, salary_data):
        """Test drop_first parameter."""
        df = salary_data[["salary", "sex"]].copy()
        result = get_dummies(df, drop_first=True)
        # Should have n-1 dummy columns for each categorical
        sex_dummies = [col for col in result.columns if "sex_" in col]
        assert len(sex_dummies) == 1  # Male or Female, not both


class TestEvalreg:
    """Tests for evalreg function."""

    def test_evalreg_basic(self, salary_data):
        """Test regression evaluation metrics."""
        df = salary_data.copy()
        # Create a simple prediction
        df["pred"] = df["salary"].mean()
        result = evalreg(df, rvar="salary", pred="pred")
        assert "r2" in result.columns
        assert "mse" in result.columns
        assert "mae" in result.columns

    def test_evalreg_dict_input(self, salary_data):
        """Test with dict input for train/test splits."""
        df = salary_data.copy()
        df["pred"] = df["salary"].mean()
        train = df.head(300)
        test = df.tail(97)
        dct = {"train": train, "test": test}
        result = evalreg(dct, rvar="salary", pred="pred")
        assert len(result) == 2
        assert "train" in result["Type"].to_list()
        assert "test" in result["Type"].to_list()


class TestSimPrediction:
    """Tests for sim_prediction function."""

    def test_sim_prediction_default(self, salary_data):
        """Test default simulation (mode/mean values)."""
        df = salary_data[["yrs_since_phd", "yrs_service", "rank"]].copy()
        result = sim_prediction(df)
        assert len(result) == 1
        # Should have mean for numeric columns
        assert "yrs_since_phd" in result.columns

    def test_sim_prediction_vary_numeric(self, salary_data):
        """Test varying a numeric variable."""
        df = salary_data[["yrs_since_phd", "yrs_service"]].copy()
        result = sim_prediction(df, vary=["yrs_since_phd"], nnv=5)
        assert len(result) == 5

    def test_sim_prediction_vary_dict(self, salary_data):
        """Test with explicit dict values."""
        df = salary_data[["yrs_since_phd", "yrs_service"]].copy()
        result = sim_prediction(df, vary={"yrs_since_phd": [5, 10, 15, 20]})
        assert len(result) == 4


class TestVIF:
    """Tests for VIF calculation."""

    def test_vif_basic(self, salary_data):
        """Test VIF calculation."""
        import statsmodels.formula.api as smf

        fitted = smf.ols("salary ~ yrs_since_phd + yrs_service", data=salary_data).fit()
        result = vif(fitted)
        assert "vif" in result.columns
        assert "Rsq" in result.columns
        # VIF should be >= 1
        assert all(result["vif"] >= 1)

    def test_vif_with_categorical(self, salary_data):
        """Test VIF with categorical variables."""
        import statsmodels.formula.api as smf

        fitted = smf.ols("salary ~ yrs_since_phd + rank", data=salary_data).fit()
        result = vif(fitted)
        assert len(result) >= 2


class TestCoefCI:
    """Tests for coefficient confidence intervals."""

    def test_coef_ci_basic(self, salary_data):
        """Test coefficient CI calculation."""
        import statsmodels.formula.api as smf

        fitted = smf.ols("salary ~ yrs_since_phd + yrs_service", data=salary_data).fit()
        result = coef_ci(fitted)
        assert "coefficient" in result.columns
        assert "2.5%" in result.columns
        assert "97.5%" in result.columns

    def test_coef_ci_no_intercept(self, salary_data):
        """Test excluding intercept."""
        import statsmodels.formula.api as smf

        fitted = smf.ols("salary ~ yrs_since_phd + yrs_service", data=salary_data).fit()
        result = coef_ci(fitted, intercept=False)
        assert "Intercept" not in result.index


class TestModelFit:
    """Tests for model_fit function."""

    def test_model_fit_regression(self, salary_data):
        """Test model fit stats for linear regression."""
        import statsmodels.formula.api as smf

        fitted = smf.ols("salary ~ yrs_since_phd + yrs_service", data=salary_data).fit()
        result = model_fit(fitted, prn=False)
        assert "rsq" in result.columns
        assert "rsq_adj" in result.columns
        assert "fvalue" in result.columns

    def test_model_fit_logistic(self, titanic_data):
        """Test model fit stats for logistic regression."""
        import statsmodels.formula.api as smf
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.families.links import Logit

        # Convert survived to binary
        titanic_data = titanic_data.copy()
        titanic_data["survived_bin"] = (titanic_data["survived"] == "Yes").astype(int)
        fitted = smf.glm(
            "survived_bin ~ age + sex", data=titanic_data, family=Binomial(link=Logit())
        ).fit()
        result = model_fit(fitted, prn=False)
        assert "pseudo_rsq_mcf" in result.columns
        assert "AUC" in result.columns
        assert "AIC" in result.columns


class TestOrCI:
    """Tests for odds ratio confidence intervals."""

    def test_or_ci_basic(self, titanic_data):
        """Test OR CI calculation."""
        import statsmodels.formula.api as smf
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.families.links import Logit

        titanic_data = titanic_data.copy()
        titanic_data["survived_bin"] = (titanic_data["survived"] == "Yes").astype(int)
        fitted = smf.glm(
            "survived_bin ~ age + sex", data=titanic_data, family=Binomial(link=Logit())
        ).fit()
        result = or_ci(fitted)
        assert "OR" in result.columns
        assert "OR%" in result.columns
        assert "2.5%" in result.columns
        assert "97.5%" in result.columns
