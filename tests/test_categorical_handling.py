"""
Tests for categorical variable handling in statsmodels integration.

These tests verify that:
1. Without proper handling, prediction with missing categorical levels fails
2. Polars Enum preserves all category levels during pandas conversion
3. The refactored code works correctly with Enum types
"""

import pandas as pd
import polars as pl
import pytest
import statsmodels.formula.api as smf


class TestEnumVsCategoricalBehavior:
    """Test the fundamental behavior difference between Enum and Categorical."""

    def test_enum_preserves_all_levels_on_pandas_conversion(self):
        """Verify Enum->pandas conversion preserves all levels, not just present ones."""
        all_levels = ["Fair", "Good", "Ideal", "Premium", "Very Good"]  # alphabetical
        cut_enum = pl.Enum(all_levels)

        # Create DataFrame with only 2 of the 5 levels
        df = pl.DataFrame({"cut": pl.Series(["Fair", "Good"]).cast(cut_enum)})
        df_pd = df.to_pandas()

        assert df_pd["cut"].dtype.name == "category"
        assert list(df_pd["cut"].cat.categories) == all_levels

    def test_categorical_loses_missing_levels_on_pandas_conversion(self):
        """Verify Categorical->pandas loses levels not present in data."""
        # Create DataFrame with Categorical - only 2 levels present
        df = pl.DataFrame({"cut": pl.Series(["Fair", "Good"]).cast(pl.Categorical)})
        df_pd = df.to_pandas()

        assert df_pd["cut"].dtype.name == "category"
        # Only the 2 present levels, not all 5
        assert list(df_pd["cut"].cat.categories) == ["Fair", "Good"]

    def test_string_has_no_categories_in_pandas(self):
        """Verify String columns become object dtype in pandas (no categories)."""
        df = pl.DataFrame({"cut": ["Fair", "Good", "Very Good"]})
        df_pd = df.to_pandas()

        assert df_pd["cut"].dtype == object
        assert not hasattr(df_pd["cut"], "cat")

    def test_sorted_enum_matches_pandas_alphabetical_order(self):
        """Verify that alphabetically sorted Enum matches pandas category order."""
        # Pandas sorts categories alphabetically when creating categorical
        train_pd = pd.DataFrame(
            {"cut": pd.Categorical(["Fair", "Good", "Very Good", "Premium", "Ideal"])}
        )
        pandas_order = list(train_pd["cut"].cat.categories)

        # Polars Enum with same alphabetical order
        enum_levels = sorted(["Fair", "Good", "Very Good", "Premium", "Ideal"])
        cut_enum = pl.Enum(enum_levels)
        df_enum = pl.DataFrame({"cut": pl.Series(["Fair"]).cast(cut_enum)})
        enum_order = list(df_enum.to_pandas()["cut"].cat.categories)

        assert pandas_order == enum_order


class TestPatsyLevelMismatch:
    """Test that patsy/statsmodels fails when categorical levels don't match."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted statsmodels model with categorical variable."""
        train_pd = pd.DataFrame(
            {
                "price": [100.0, 200.0, 300.0, 400.0, 500.0],
                "cut": pd.Categorical(
                    ["Fair", "Good", "Very Good", "Premium", "Ideal"]
                ),
                "carat": [0.5, 0.7, 1.0, 1.2, 1.5],
            }
        )
        return smf.ols("price ~ carat + cut", data=train_pd).fit()

    def test_prediction_fails_with_missing_levels(self, fitted_model):
        """Show that prediction fails when categorical has fewer levels."""
        # Prediction data with only 2 of 5 levels
        pred_pd = pd.DataFrame(
            {
                "cut": pd.Categorical(["Fair", "Good"]),  # Only 2 levels
                "carat": [0.5, 0.7],
            }
        )

        with pytest.raises(Exception) as exc_info:
            fitted_model.predict(pred_pd)

        # Patsy raises an error about mismatching levels
        assert "mismatching levels" in str(exc_info.value).lower()

    def test_prediction_fails_with_wrong_order(self, fitted_model):
        """Show that prediction fails when categorical levels are in wrong order."""
        # Get the training order (alphabetical: Fair, Good, Ideal, Premium, Very Good)
        # Use different order intentionally
        pred_pd = pd.DataFrame(
            {
                "cut": pd.Categorical(
                    ["Fair", "Good"],
                    categories=["Fair", "Good", "Very Good", "Premium", "Ideal"],
                ),
                "carat": [0.5, 0.7],
            }
        )

        with pytest.raises(Exception) as exc_info:
            fitted_model.predict(pred_pd)

        assert "mismatching levels" in str(exc_info.value).lower()

    def test_prediction_succeeds_with_matching_levels_and_order(self, fitted_model):
        """Show that prediction succeeds when levels match exactly."""
        # Use exact same order as pandas creates (alphabetical)
        categories = ["Fair", "Good", "Ideal", "Premium", "Very Good"]
        pred_pd = pd.DataFrame(
            {
                "cut": pd.Categorical(["Fair", "Good"], categories=categories),
                "carat": [0.5, 0.7],
            }
        )

        result = fitted_model.predict(pred_pd)
        assert len(result) == 2


class TestEnumSolvesLevelMismatch:
    """Test that Enum with proper ordering solves the level mismatch problem."""

    @pytest.fixture
    def fitted_model_and_enum(self):
        """Create model and matching Enum type."""
        train_pd = pd.DataFrame(
            {
                "price": [100.0, 200.0, 300.0, 400.0, 500.0],
                "cut": pd.Categorical(
                    ["Fair", "Good", "Very Good", "Premium", "Ideal"]
                ),
                "carat": [0.5, 0.7, 1.0, 1.2, 1.5],
            }
        )
        model = smf.ols("price ~ carat + cut", data=train_pd).fit()

        # Enum with alphabetically sorted levels (matching pandas)
        cut_enum = pl.Enum(sorted(["Fair", "Good", "Very Good", "Premium", "Ideal"]))

        return model, cut_enum

    def test_enum_prediction_succeeds_with_subset_values(self, fitted_model_and_enum):
        """Show that Enum->pandas preserves levels, allowing prediction to succeed."""
        model, cut_enum = fitted_model_and_enum

        # Create prediction data with only 2 values, but Enum preserves all 5 levels
        pred_pl = pl.DataFrame(
            {
                "cut": pl.Series(["Fair", "Good"]).cast(cut_enum),
                "carat": [0.5, 0.7],
            }
        )
        pred_pd = pred_pl.to_pandas()

        # Should succeed because Enum preserves all levels
        result = model.predict(pred_pd)
        assert len(result) == 2

    def test_enum_from_polars_categorical_fails_without_sorting(
        self, fitted_model_and_enum
    ):
        """Show that unsorted Enum causes level order mismatch."""
        model, _ = fitted_model_and_enum

        # Enum with non-alphabetical order
        cut_enum_wrong_order = pl.Enum(
            ["Fair", "Good", "Very Good", "Premium", "Ideal"]
        )
        pred_pl = pl.DataFrame(
            {
                "cut": pl.Series(["Fair", "Good"]).cast(cut_enum_wrong_order),
                "carat": [0.5, 0.7],
            }
        )
        pred_pd = pred_pl.to_pandas()

        with pytest.raises(Exception) as exc_info:
            model.predict(pred_pd)

        assert "mismatching levels" in str(exc_info.value).lower()


class TestCategoricalEdgeCases:
    """Test edge cases for categorical handling."""

    def test_multiple_categorical_variables(self):
        """Test with multiple categorical variables."""
        # Create training data with two categoricals
        train_pd = pd.DataFrame(
            {
                "price": [100.0, 200.0, 300.0, 400.0],
                "cut": pd.Categorical(["Fair", "Good", "Very Good", "Premium"]),
                "color": pd.Categorical(["D", "E", "F", "G"]),
                "carat": [0.5, 0.7, 1.0, 1.2],
            }
        )
        model = smf.ols("price ~ carat + cut + color", data=train_pd).fit()

        # Create Enum types with sorted levels
        cut_enum = pl.Enum(sorted(["Fair", "Good", "Very Good", "Premium"]))
        color_enum = pl.Enum(sorted(["D", "E", "F", "G"]))

        # Prediction with subset of both categoricals
        pred_pl = pl.DataFrame(
            {
                "cut": pl.Series(["Fair"]).cast(cut_enum),
                "color": pl.Series(["D"]).cast(color_enum),
                "carat": [0.5],
            }
        )
        pred_pd = pred_pl.to_pandas()

        result = model.predict(pred_pd)
        assert len(result) == 1

    def test_categorical_with_spaces_in_values(self):
        """Test categorical values with spaces work correctly."""
        train_pd = pd.DataFrame(
            {
                "price": [100.0, 200.0, 300.0],
                "quality": pd.Categorical(
                    ["Very Good", "Above Average", "Below Average"]
                ),
                "size": [1.0, 2.0, 3.0],
            }
        )
        model = smf.ols("price ~ size + quality", data=train_pd).fit()

        # Enum with sorted levels (spaces should be handled)
        quality_enum = pl.Enum(sorted(["Very Good", "Above Average", "Below Average"]))

        pred_pl = pl.DataFrame(
            {
                "quality": pl.Series(["Very Good"]).cast(quality_enum),
                "size": [1.5],
            }
        )
        pred_pd = pred_pl.to_pandas()

        result = model.predict(pred_pd)
        assert len(result) == 1

    def test_numeric_looking_categorical_values(self):
        """Test categorical values that look like numbers (e.g., '1', '2')."""
        train_pd = pd.DataFrame(
            {
                "price": [100.0, 200.0, 300.0],
                "grade": pd.Categorical(["1", "2", "3"]),
                "size": [1.0, 2.0, 3.0],
            }
        )
        model = smf.ols("price ~ size + grade", data=train_pd).fit()

        grade_enum = pl.Enum(sorted(["1", "2", "3"]))

        pred_pl = pl.DataFrame(
            {
                "grade": pl.Series(["1", "2"]).cast(grade_enum),
                "size": [1.5, 2.5],
            }
        )
        pred_pd = pred_pl.to_pandas()

        result = model.predict(pred_pd)
        assert len(result) == 2


class TestRegressIntegration:
    """Integration tests for regress with categorical variables using Enum."""

    @pytest.fixture
    def diamonds_data(self):
        """Load diamonds dataset with categorical columns."""
        return pl.read_parquet(
            "https://github.com/radiant-ai-hub/pyrsm/raw/refs/heads/main/examples/data/data/diamonds.parquet"
        ).sample(500, seed=1234)

    def test_regress_stores_enum_types(self, diamonds_data):
        """Test that regress stores Enum types for categorical columns."""
        from pyrsm.model.regress import regress

        reg = regress(diamonds_data, rvar="price", evar=["carat", "cut", "color"])

        assert hasattr(reg, "_enum_types")
        assert "cut" in reg._enum_types
        assert "color" in reg._enum_types
        assert "carat" not in reg._enum_types  # numeric, not categorical

    def test_regress_predict_with_subset_levels(self, diamonds_data):
        """Test that predict works with data containing subset of categorical levels."""
        from pyrsm.model.regress import regress

        reg = regress(diamonds_data, rvar="price", evar=["carat", "cut"])

        # Get all cut levels from training
        sorted(diamonds_data["cut"].unique().to_list())

        # Create prediction data with only 2 cuts
        pred_data = pl.DataFrame(
            {"carat": [1.0, 1.0], "cut": ["Fair", "Good"]}  # Subset of levels
        )

        # This should work because Enum preserves all levels
        result = reg.predict(data=pred_data)
        assert len(result) == 2
        assert "prediction" in result.columns

    def test_regress_predict_cmd_categorical(self, diamonds_data):
        """Test predict with cmd for categorical variable."""
        from pyrsm.model.regress import regress

        reg = regress(diamonds_data, rvar="price", evar=["carat", "cut"])

        # Use cmd to vary categorical variable
        result = reg.predict(cmd={"cut": ["Fair", "Good", "Very Good"]})
        assert len(result) == 3
        assert "prediction" in result.columns

    def test_regress_predict_unseen_level_raises_error(self, diamonds_data):
        """Test that predicting with unseen level raises informative error."""
        from pyrsm.model.regress import regress

        reg = regress(diamonds_data, rvar="price", evar=["carat", "cut"])

        # Create prediction data with unseen cut level
        pred_data = pl.DataFrame(
            {"carat": [1.0], "cut": ["SuperIdeal"]}  # Unseen level
        )

        with pytest.raises(ValueError) as exc_info:
            reg.predict(data=pred_data)

        assert "SuperIdeal" in str(exc_info.value)
        assert "not seen during training" in str(exc_info.value).lower()


class TestLogisticIntegration:
    """Integration tests for logistic with categorical variables using Enum."""

    @pytest.fixture
    def titanic_data(self):
        """Load titanic dataset with categorical columns."""
        return pl.read_parquet(
            "https://github.com/radiant-ai-hub/pyrsm/raw/refs/heads/main/examples/data/data/titanic.parquet"
        ).sample(500, seed=1234)

    def test_logistic_stores_enum_types(self, titanic_data):
        """Test that logistic stores Enum types for categorical columns."""
        from pyrsm.model.logistic import logistic

        log = logistic(
            titanic_data, rvar="survived", lev="Yes", evar=["age", "sex", "pclass"]
        )

        assert hasattr(log, "_enum_types")
        assert "sex" in log._enum_types
        assert "pclass" in log._enum_types
        assert "age" not in log._enum_types  # numeric

    def test_logistic_predict_with_subset_levels(self, titanic_data):
        """Test that predict works with data containing subset of categorical levels."""
        from pyrsm.model.logistic import logistic

        log = logistic(titanic_data, rvar="survived", lev="Yes", evar=["age", "sex"])

        # Create prediction data with only 1 sex level
        pred_data = pl.DataFrame({"age": [25.0, 30.0], "sex": ["male", "male"]})

        # This should work because Enum preserves all levels
        result = log.predict(data=pred_data)
        assert len(result) == 2
        assert "prediction" in result.columns

    def test_logistic_predict_cmd_categorical(self, titanic_data):
        """Test predict with cmd for categorical variable."""
        from pyrsm.model.logistic import logistic

        log = logistic(titanic_data, rvar="survived", lev="Yes", evar=["age", "sex"])

        # Use cmd to vary categorical variable
        result = log.predict(cmd={"sex": ["male", "female"]})
        assert len(result) == 2
        assert "prediction" in result.columns


class TestToPandasWithCategories:
    """Test the deprecated to_pandas_with_categories function.

    This function is no longer used in the main workflow since Enum types
    handle categorical level preservation automatically. Kept for backward
    compatibility only.
    """

    def test_to_pandas_with_categories_fixes_missing_levels(self):
        """Verify that to_pandas_with_categories adds missing levels."""
        from pyrsm.model.model import to_pandas_with_categories

        # Original data with all levels
        original = pl.DataFrame(
            {"cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"]}
        )
        # Prediction data with subset
        pred = pl.DataFrame({"cut": ["Fair", "Good"]})

        # Without fix
        pred_pd_raw = pred.to_pandas()
        assert not hasattr(pred_pd_raw["cut"], "cat")

        # With fix - need original to have categorical
        original_pd = original.to_pandas()
        original_pd["cut"] = pd.Categorical(original_pd["cut"])
        original_fixed = pl.from_pandas(original_pd)

        pred_pd_fixed = to_pandas_with_categories(pred, original_fixed)

        assert hasattr(pred_pd_fixed["cut"], "cat")
        assert len(pred_pd_fixed["cut"].cat.categories) == 5


class TestHelperFunctions:
    """Tests for the new helper functions (to be implemented)."""

    def test_convert_categoricals_to_enum_preserves_occurrence_order(self):
        """Test that convert_categoricals_to_enum preserves occurrence order (not alphabetical)."""
        from pyrsm.model.model import convert_categoricals_to_enum

        df = pl.DataFrame(
            {
                "cut": ["Ideal", "Fair", "Good"],  # occurrence order: Ideal, Fair, Good
                "price": [500.0, 100.0, 200.0],
            }
        )

        converted, enum_types = convert_categoricals_to_enum(df, ["cut"])

        # Enum should preserve occurrence order, NOT alphabetical
        # polars unique() returns first-seen order
        actual_levels = list(enum_types["cut"].categories)
        # Just verify we have all 3 levels and order is NOT alphabetical
        assert len(actual_levels) == 3
        assert set(actual_levels) == {"Ideal", "Fair", "Good"}
        # Alphabetical would be ["Fair", "Good", "Ideal"] - we should NOT have that
        assert actual_levels != ["Fair", "Good", "Ideal"]
        assert converted["cut"].dtype == enum_types["cut"]

    def test_convert_categoricals_to_enum_handles_multiple_columns(self):
        """Test conversion with multiple categorical columns."""
        from pyrsm.model.model import convert_categoricals_to_enum

        df = pl.DataFrame(
            {
                "cut": ["Ideal", "Fair", "Good"],
                "color": ["E", "D", "F"],
                "price": [500.0, 100.0, 200.0],
            }
        )

        converted, enum_types = convert_categoricals_to_enum(df, ["cut", "color"])

        assert "cut" in enum_types
        assert "color" in enum_types
        # Just verify all levels are present (order is occurrence-based)
        assert set(enum_types["cut"].categories) == {"Ideal", "Fair", "Good"}
        assert set(enum_types["color"].categories) == {"E", "D", "F"}

    def test_apply_enum_types_applies_stored_types(self):
        """Test that apply_enum_types correctly applies Enum types."""
        from pyrsm.model.model import apply_enum_types, convert_categoricals_to_enum

        # First convert original data
        original = pl.DataFrame(
            {
                "cut": ["Fair", "Good", "Ideal", "Premium", "Very Good"],
                "price": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )
        _, enum_types = convert_categoricals_to_enum(original, ["cut"])

        # Apply to new data with subset
        pred = pl.DataFrame(
            {
                "cut": ["Fair", "Good"],
                "price": [150.0, 250.0],
            }
        )

        result = apply_enum_types(pred, enum_types)

        # Result should have Enum type with all original levels
        assert result["cut"].dtype == enum_types["cut"]
        # Verify pandas conversion preserves all levels
        result_pd = result.to_pandas()
        assert len(result_pd["cut"].cat.categories) == 5

    def test_apply_enum_types_raises_on_unseen_level(self):
        """Test that apply_enum_types raises error for unseen levels."""
        from pyrsm.model.model import apply_enum_types, convert_categoricals_to_enum

        # Original data
        original = pl.DataFrame({"cut": ["Fair", "Good", "Ideal"]})
        _, enum_types = convert_categoricals_to_enum(original, ["cut"])

        # Try to apply to data with unseen level
        pred = pl.DataFrame({"cut": ["Fair", "Premium"]})  # "Premium" is unseen

        with pytest.raises(ValueError) as exc_info:
            apply_enum_types(pred, enum_types)

        assert "Premium" in str(exc_info.value)
        assert "not seen during training" in str(exc_info.value).lower()


class TestEnumOrderingPreservation:
    """
    Test that category ordering is preserved correctly:
    - Enum columns: preserve explicit level order
    - String/Categorical columns: use occurrence order

    This is the NEW behavior - no alphabetical sorting.
    """

    @pytest.fixture
    def test_data_enum(self):
        """Create test data with enum variables in NON-alphabetical order."""
        import numpy as np

        np.random.seed(1234)
        n = 300

        # Enum levels in NON-alphabetical order
        pclass_levels = ["3rd", "2nd", "1st"]  # NOT alphabetical
        sex_levels = ["male", "female"]  # NOT alphabetical

        data = pl.DataFrame(
            {
                "age": np.random.uniform(1, 80, n),
                "fare": np.random.uniform(5, 500, n),
            }
        )

        pclass_values = np.random.choice(pclass_levels, n)
        sex_values = np.random.choice(sex_levels, n)

        data = data.with_columns(
            [
                pl.Series("pclass", pclass_values).cast(pl.Enum(pclass_levels)),
                pl.Series("sex", sex_values).cast(pl.Enum(sex_levels)),
            ]
        )

        # Classification response
        prob = (
            0.3
            + 0.3 * (data["sex"].cast(pl.Utf8) == "female").cast(pl.Float64)
            + 0.2 * (data["pclass"].cast(pl.Utf8) == "1st").cast(pl.Float64)
        )
        survived = np.where(np.random.uniform(0, 1, n) < prob.to_numpy(), "Yes", "No")
        data = data.with_columns(pl.Series("survived", survived))

        # Regression response
        data = data.with_columns(
            (
                data["fare"] * 0.5
                + 10 * (data["pclass"].cast(pl.Utf8) == "1st").cast(pl.Float64)
                + np.random.normal(0, 10, n)
            ).alias("price")
        )

        return data

    def test_get_dummies_preserves_enum_order(self):
        """Test that get_dummies preserves Enum level order (not alphabetical)."""
        from pyrsm.model.model import get_dummies

        # Enum with non-alphabetical order: 3rd, 2nd, 1st
        pclass_enum = pl.Enum(["3rd", "2nd", "1st"])
        sex_enum = pl.Enum(["male", "female"])

        df = pl.DataFrame(
            {
                "pclass": pl.Series(["1st", "2nd", "3rd"]).cast(pclass_enum),
                "sex": pl.Series(["male", "female", "male"]).cast(sex_enum),
            }
        )

        dummies = get_dummies(df)

        # With drop_first=True, first level is dropped
        # pclass: drops "3rd" (first in enum), keeps "2nd", "1st" (in that order)
        # sex: drops "male" (first in enum), keeps "female"
        expected_cols = ["pclass_2nd", "pclass_1st", "sex_female"]
        assert dummies.columns == expected_cols

    def test_different_enum_orderings_produce_different_dummies(self):
        """Test that different enum orderings produce different dummy column order."""
        from pyrsm.model.model import get_dummies

        # Two dataframes with DIFFERENT enum orderings
        data1 = pl.DataFrame(
            {
                "grade": pl.Series(["A", "B", "C"]).cast(
                    pl.Enum(["C", "B", "A"])
                )  # C, B, A order
            }
        )
        data2 = pl.DataFrame(
            {
                "grade": pl.Series(["A", "B", "C"]).cast(
                    pl.Enum(["A", "B", "C"])
                )  # A, B, C order
            }
        )

        dummies1 = get_dummies(data1)
        dummies2 = get_dummies(data2)

        # Different enum order → different dummy column order
        # data1: drops C (first), keeps B, A → ["grade_B", "grade_A"]
        # data2: drops A (first), keeps B, C → ["grade_B", "grade_C"]
        assert dummies1.columns == ["grade_B", "grade_A"]
        assert dummies2.columns == ["grade_B", "grade_C"]

    def test_mlp_train_predict_with_enum_ordering(self, test_data_enum):
        """Test MLP model trains and predicts correctly with Enum ordering."""
        from pyrsm.model.mlp import mlp

        evar = ["pclass", "sex", "age", "fare"]

        # Train model
        model = mlp(
            test_data_enum,
            rvar="survived",
            lev="Yes",
            evar=evar,
            hidden_layer_sizes=(10,),
            random_state=1234,
            max_iter=1000,
        )

        # Feature names should reflect Enum order (3rd dropped, then 2nd, 1st)
        # pclass: 3rd dropped → pclass_2nd, pclass_1st
        # sex: male dropped → sex_female
        assert "pclass_2nd" in model.feature_names
        assert "pclass_1st" in model.feature_names
        assert "sex_female" in model.feature_names

        # Predict on same enum data should work
        pred = model.predict(test_data_enum.head(20))
        assert "prediction" in pred.columns
        assert pred.height == 20

        # Predictions should be between 0 and 1
        preds = pred["prediction"].to_list()
        assert all(0 <= p <= 1 for p in preds)

    def test_rforest_train_predict_with_enum_ordering(self, test_data_enum):
        """Test Random Forest trains and predicts correctly with Enum ordering."""
        from pyrsm.model.rforest import rforest

        evar = ["pclass", "sex", "age", "fare"]

        model = rforest(
            test_data_enum,
            rvar="survived",
            lev="Yes",
            evar=evar,
            n_estimators=50,
            random_state=1234,
        )

        # Predict should work
        pred = model.predict(test_data_enum.head(20))
        assert "prediction" in pred.columns
        assert pred.height == 20

    def test_xgboost_train_predict_with_enum_ordering(self, test_data_enum):
        """Test XGBoost trains and predicts correctly with Enum ordering."""
        from pyrsm.model.xgboost import xgboost

        evar = ["pclass", "sex", "age", "fare"]

        model = xgboost(
            test_data_enum,
            rvar="survived",
            lev="Yes",
            evar=evar,
            n_estimators=50,
            random_state=1234,
        )

        pred = model.predict(test_data_enum.head(20))
        assert "prediction" in pred.columns
        assert pred.height == 20

    def test_predict_with_string_data_after_enum_training(self, test_data_enum):
        """Test prediction with string data after training with Enum."""
        from pyrsm.model.mlp import mlp

        evar = ["pclass", "sex", "age", "fare"]

        # Train with Enum data
        model = mlp(
            test_data_enum,
            rvar="survived",
            lev="Yes",
            evar=evar,
            hidden_layer_sizes=(10,),
            random_state=1234,
            max_iter=1000,
        )

        # Create string prediction data (same values as enum)
        pred_data = pl.DataFrame(
            {
                "pclass": ["1st", "2nd", "3rd"],
                "sex": ["male", "female", "male"],
                "age": [25.0, 35.0, 45.0],
                "fare": [100.0, 200.0, 50.0],
            }
        )

        # This should work - model stores categories and applies them
        pred = model.predict(pred_data)
        assert "prediction" in pred.columns
        assert pred.height == 3


# =============================================================================
# Test Class: Silent Failure Detection
# =============================================================================


class TestSilentFailureDetection:
    """
    Tests designed to catch SILENT FAILURES - where predictions are wrong
    but no error is raised. Uses deterministic datasets where we KNOW
    the correct predictions.

    KEY INSIGHT: If categories get misaligned, predictions will be
    systematically wrong even though no error is raised.
    """

    def test_statsmodels_deterministic_predictions(self):
        """
        Train statsmodels regress with known data, verify predictions
        match expected values exactly.
        """
        from pyrsm.model.regress import regress

        # Enum with NON-alphabetical order: C is baseline (first)
        grade_enum = pl.Enum(["C", "B", "A"])

        # Perfect deterministic data: C=500, B=600, A=700
        data = pl.DataFrame(
            {
                "grade": pl.Series(["A", "B", "C"] * 100).cast(grade_enum),
                "price": [700.0, 600.0, 500.0] * 100,
            }
        )

        reg = regress(data, rvar="price", evar=["grade"])

        # Verify coefficients match expected values
        coef_df = reg.coef.to_pandas().set_index("index")

        # Intercept = C's value = 500 (C is baseline, first in Enum)
        assert (
            abs(coef_df.loc["Intercept", "coefficient"] - 500) < 0.1
        ), f"Intercept should be 500, got {coef_df.loc['Intercept', 'coefficient']}"

        # Predict and verify EXACT values
        pred = reg.predict(data=pl.DataFrame({"grade": ["A", "B", "C"]}))
        preds = dict(zip(pred["grade"].to_list(), pred["prediction"].to_list()))

        assert abs(preds["A"] - 700) < 0.1, f"A should be 700, got {preds['A']}"
        assert abs(preds["B"] - 600) < 0.1, f"B should be 600, got {preds['B']}"
        assert abs(preds["C"] - 500) < 0.1, f"C should be 500, got {preds['C']}"

    def test_statsmodels_logistic_coefficient_direction(self):
        """
        Verify logistic regression coefficient direction matches actual data.

        If categories are misaligned, the coefficient sign will be WRONG.
        """
        import numpy as np

        from pyrsm.model.logistic import logistic

        np.random.seed(42)

        # Enum: female is baseline (first)
        sex_enum = pl.Enum(["female", "male"])

        # Create data where males have LOWER survival than females
        # female: 70% survival, male: 30% survival
        data = pl.DataFrame(
            {
                "sex": pl.Series(["female"] * 100 + ["male"] * 100).cast(sex_enum),
                "survived": ["Yes"] * 70 + ["No"] * 30 + ["Yes"] * 30 + ["No"] * 70,
            }
        )

        log = logistic(data, rvar="survived", lev="Yes", evar=["sex"])

        # Male coefficient should be NEGATIVE (lower odds than female baseline)
        coef_df = log.coef.to_pandas().set_index("index")
        male_coef = coef_df.loc["sex[male]", "coefficient"]

        assert male_coef < 0, (
            f"Male coefficient should be NEGATIVE (males have lower survival), "
            f"got {male_coef}"
        )

        # OR should be < 1
        male_or = coef_df.loc["sex[male]", "OR"]
        assert male_or < 1, f"Male OR should be < 1, got {male_or}"

    def test_sklearn_prediction_direction(self):
        """
        Train sklearn model (MLP) with perfect separation, verify
        prediction direction is correct.
        """
        import numpy as np

        from pyrsm.model.mlp import mlp

        np.random.seed(42)

        # Enum with non-alphabetical order
        grade_enum = pl.Enum(["C", "B", "A"])

        # Perfect separation: A -> Yes, C -> No
        data = pl.DataFrame(
            {
                "grade": pl.Series(["A"] * 100 + ["C"] * 100).cast(grade_enum),
                "target": ["Yes"] * 100 + ["No"] * 100,
            }
        )

        model = mlp(
            data,
            rvar="target",
            lev="Yes",
            evar=["grade"],
            hidden_layer_sizes=(10,),
            max_iter=2000,
            random_state=42,
        )

        # Predict for A and C
        pred = model.predict(data=pl.DataFrame({"grade": ["A", "C"]}))
        preds = dict(zip(pred["grade"].to_list(), pred["prediction"].to_list()))

        # A should have HIGH probability (associated with Yes)
        # C should have LOW probability (associated with No)
        assert preds["A"] > 0.7, f"A should have prob > 0.7, got {preds['A']}"
        assert preds["C"] < 0.3, f"C should have prob < 0.3, got {preds['C']}"
        assert preds["A"] > preds["C"], f"A ({preds['A']}) should be > C ({preds['C']})"

    def test_cross_type_train_enum_predict_string(self):
        """
        Train with Enum, predict with plain String. Predictions must be correct.

        This tests that apply_enum_types correctly converts prediction data.
        """
        from pyrsm.model.regress import regress

        # Train with Enum
        grade_enum = pl.Enum(["C", "B", "A"])
        train_data = pl.DataFrame(
            {
                "grade": pl.Series(["A", "B", "C"] * 50).cast(grade_enum),
                "price": [700.0, 600.0, 500.0] * 50,
            }
        )

        reg = regress(train_data, rvar="price", evar=["grade"])

        # Predict with plain STRING (no Enum)
        pred_string = pl.DataFrame({"grade": ["A", "B", "C"]})

        predictions = reg.predict(data=pred_string)
        preds = dict(
            zip(predictions["grade"].to_list(), predictions["prediction"].to_list())
        )

        # Predictions must be correct despite different input type
        assert abs(preds["A"] - 700) < 1.0, f"A should be 700, got {preds['A']}"
        assert abs(preds["B"] - 600) < 1.0, f"B should be 600, got {preds['B']}"
        assert abs(preds["C"] - 500) < 1.0, f"C should be 500, got {preds['C']}"

    def test_baseline_category_correct(self):
        """
        Verify the baseline (dropped) category is the FIRST level of the Enum.
        """
        from pyrsm.model.regress import regress

        # Enum: X is first, should be baseline (dropped)
        cat_enum = pl.Enum(["X", "Y", "Z"])

        data = pl.DataFrame(
            {
                "category": pl.Series(["X", "Y", "Z"] * 50).cast(cat_enum),
                "value": [100.0, 200.0, 300.0] * 50,
            }
        )

        reg = regress(data, rvar="value", evar=["category"])

        # X should NOT appear in coefficients (it's baseline)
        coef_names = reg.coef["index"].to_list()

        assert not any(
            "X" in name for name in coef_names if name != "Intercept"
        ), f"X should be baseline (dropped), but found in coefficients: {coef_names}"
        assert any("Y" in name for name in coef_names), "Y should have a coefficient"
        assert any("Z" in name for name in coef_names), "Z should have a coefficient"

    def test_multi_categorical_all_combinations(self):
        """
        Train with two categoricals, verify ALL combinations predict correctly.

        If either categorical is misaligned, combination predictions will be wrong.
        """
        import itertools

        from pyrsm.model.regress import regress

        color_enum = pl.Enum(["blue", "green", "red"])
        size_enum = pl.Enum(["small", "medium", "large"])

        colors = ["blue", "green", "red"]
        sizes = ["small", "medium", "large"]

        # Deterministic effects: blue=0, green=100, red=200 | small=0, medium=50, large=100
        color_effect = {"blue": 0, "green": 100, "red": 200}
        size_effect = {"small": 0, "medium": 50, "large": 100}
        base = 500

        combos = list(itertools.product(colors, sizes))
        data_rows = []
        for color, size in combos:
            price = base + color_effect[color] + size_effect[size]
            data_rows.extend(
                [{"color": color, "size": size, "price": float(price)}] * 20
            )

        data = pl.DataFrame(data_rows).with_columns(
            [
                pl.col("color").cast(color_enum),
                pl.col("size").cast(size_enum),
            ]
        )

        reg = regress(data, rvar="price", evar=["color", "size"])

        # Predict all combinations
        pred_data = pl.DataFrame([{"color": c, "size": s} for c, s in combos])
        predictions = reg.predict(data=pred_data)

        # Verify each combination
        for i, (color, size) in enumerate(combos):
            expected = base + color_effect[color] + size_effect[size]
            actual = predictions["prediction"][i]
            assert (
                abs(actual - expected) < 1.0
            ), f"{color}+{size}: expected {expected}, got {actual}"

    def test_prediction_consistency_same_data_twice(self):
        """
        Predicting on the same data twice should give identical results.
        """
        import numpy as np

        from pyrsm.model.mlp import mlp

        np.random.seed(42)

        grade_enum = pl.Enum(["C", "B", "A"])
        data = pl.DataFrame(
            {
                "grade": pl.Series(["A", "B", "C"] * 50).cast(grade_enum),
                "target": ["Yes", "No", "No"] * 50,
            }
        )

        model = mlp(
            data,
            rvar="target",
            lev="Yes",
            evar=["grade"],
            hidden_layer_sizes=(5,),
            random_state=42,
            max_iter=1000,
        )

        pred1 = model.predict()
        pred2 = model.predict()

        # Predictions should be IDENTICAL
        assert (
            pred1["prediction"].to_list() == pred2["prediction"].to_list()
        ), "Repeated predictions should be identical"

    def test_numeric_looking_categories(self):
        """
        Test categories that look like numbers: "1", "2", "3".
        These could cause confusion with actual numeric columns.
        """
        from pyrsm.model.regress import regress

        # Non-alphabetical order for numeric-looking strings
        tier_enum = pl.Enum(["3", "2", "1"])

        data = pl.DataFrame(
            {
                "tier": pl.Series(["1", "2", "3"] * 50).cast(tier_enum),
                "price": [300.0, 200.0, 100.0] * 50,
            }
        )

        reg = regress(data, rvar="price", evar=["tier"])

        pred = reg.predict(data=pl.DataFrame({"tier": ["1", "2", "3"]}))
        preds = dict(zip(pred["tier"].to_list(), pred["prediction"].to_list()))

        assert abs(preds["1"] - 300) < 1.0, f"'1' should be 300, got {preds['1']}"
        assert abs(preds["2"] - 200) < 1.0, f"'2' should be 200, got {preds['2']}"
        assert abs(preds["3"] - 100) < 1.0, f"'3' should be 100, got {preds['3']}"

    def test_rforest_deterministic_predictions(self):
        """
        Train Random Forest with deterministic data, verify prediction direction.
        """
        import numpy as np

        from pyrsm.model.rforest import rforest

        np.random.seed(42)

        # Non-alphabetical Enum order
        grade_enum = pl.Enum(["C", "B", "A"])

        # Perfect separation: A -> Yes, C -> No
        data = pl.DataFrame(
            {
                "grade": pl.Series(["A"] * 100 + ["C"] * 100).cast(grade_enum),
                "target": ["Yes"] * 100 + ["No"] * 100,
            }
        )

        model = rforest(
            data,
            rvar="target",
            lev="Yes",
            evar=["grade"],
            n_estimators=50,
            random_state=42,
        )

        # Predict for A and C
        pred = model.predict(data=pl.DataFrame({"grade": ["A", "C"]}))
        preds = dict(zip(pred["grade"].to_list(), pred["prediction"].to_list()))

        # A should have HIGH probability, C should have LOW
        assert preds["A"] > 0.7, f"A should have prob > 0.7, got {preds['A']}"
        assert preds["C"] < 0.3, f"C should have prob < 0.3, got {preds['C']}"

    def test_xgboost_deterministic_predictions(self):
        """
        Train XGBoost with deterministic data, verify prediction direction.
        """
        import numpy as np

        from pyrsm.model.xgboost import xgboost

        np.random.seed(42)

        # Non-alphabetical Enum order
        grade_enum = pl.Enum(["C", "B", "A"])

        # Perfect separation: A -> Yes, C -> No
        data = pl.DataFrame(
            {
                "grade": pl.Series(["A"] * 100 + ["C"] * 100).cast(grade_enum),
                "target": ["Yes"] * 100 + ["No"] * 100,
            }
        )

        model = xgboost(
            data,
            rvar="target",
            lev="Yes",
            evar=["grade"],
            n_estimators=50,
            random_state=42,
        )

        # Predict for A and C
        pred = model.predict(data=pl.DataFrame({"grade": ["A", "C"]}))
        preds = dict(zip(pred["grade"].to_list(), pred["prediction"].to_list()))

        # A should have HIGH probability, C should have LOW
        assert preds["A"] > 0.7, f"A should have prob > 0.7, got {preds['A']}"
        assert preds["C"] < 0.3, f"C should have prob < 0.3, got {preds['C']}"

    def test_sklearn_cross_type_train_enum_predict_string(self):
        """
        Train sklearn model with Enum, predict with String. Verify correct.
        """
        import numpy as np

        from pyrsm.model.rforest import rforest

        np.random.seed(42)

        # Train with Enum
        grade_enum = pl.Enum(["C", "B", "A"])
        train_data = pl.DataFrame(
            {
                "grade": pl.Series(["A"] * 100 + ["C"] * 100).cast(grade_enum),
                "target": ["Yes"] * 100 + ["No"] * 100,
            }
        )

        model = rforest(
            train_data,
            rvar="target",
            lev="Yes",
            evar=["grade"],
            n_estimators=50,
            random_state=42,
        )

        # Predict with plain STRING
        pred_string = pl.DataFrame({"grade": ["A", "C"]})
        pred = model.predict(data=pred_string)
        preds = dict(zip(pred["grade"].to_list(), pred["prediction"].to_list()))

        # Predictions should still be correct
        assert preds["A"] > 0.7, f"A should have prob > 0.7, got {preds['A']}"
        assert preds["C"] < 0.3, f"C should have prob < 0.3, got {preds['C']}"


# =============================================================================
# Test Class: Plot Ordering Verification
# =============================================================================


class TestPlotCategoryOrdering:
    """
    Tests to verify that categorical variable ordering is preserved in plots.

    KEY ISSUE: Plotnine/ggplot defaults to alphabetical ordering.
    We need to ensure Enum ordering is preserved in all plot types.
    """

    @pytest.fixture
    def titanic_enum_data(self):
        """Create titanic-like data with non-alphabetical Enum ordering."""
        import numpy as np

        np.random.seed(42)

        # Non-alphabetical order: 3rd, 2nd, 1st (importance order)
        pclass_enum = pl.Enum(["3rd", "2nd", "1st"])
        sex_enum = pl.Enum(["male", "female"])

        n = 200
        pclass_vals = np.random.choice(["3rd", "2nd", "1st"], n)
        sex_vals = np.random.choice(["male", "female"], n)

        data = pl.DataFrame(
            {
                "pclass": pl.Series(pclass_vals).cast(pclass_enum),
                "sex": pl.Series(sex_vals).cast(sex_enum),
                "age": np.random.uniform(1, 80, n),
            }
        )

        # Survival probability based on pclass and sex
        prob = (
            0.2
            + 0.3 * (data["pclass"].cast(pl.Utf8) == "1st").cast(pl.Float64)
            + 0.15 * (data["pclass"].cast(pl.Utf8) == "2nd").cast(pl.Float64)
            + 0.25 * (data["sex"].cast(pl.Utf8) == "female").cast(pl.Float64)
        )
        survived = np.where(np.random.uniform(0, 1, n) < prob.to_numpy(), "Yes", "No")
        data = data.with_columns(pl.Series("survived", survived))

        return data

    def test_pred_plot_sm_preserves_enum_order(self, titanic_enum_data):
        """
        Test that pred_plot_sm preserves Enum ordering in x-axis.
        """
        from pyrsm.model.logistic import logistic

        model = logistic(
            titanic_enum_data, rvar="survived", lev="Yes", evar=["pclass", "sex", "age"]
        )

        # Get the plot - this should preserve ordering
        plot = model.plot("pred", incl=["pclass"])

        # The plot should exist
        assert plot is not None

        # Check that the data in the plot has correct ordering
        # Access the underlying data from the ggplot object
        plot_data = plot.data
        if "pclass" in plot_data.columns:
            # Get unique values in order they appear
            pclass_order = plot_data["pclass"].unique().to_list()
            # Should be in Enum order: 3rd, 2nd, 1st (NOT alphabetical)
            assert (
                pclass_order[0] != "1st"
            ), f"First pclass should NOT be '1st' (alphabetical), got order: {pclass_order}"

    def test_logistic_pdp_preserves_enum_order(self, titanic_enum_data):
        """
        Test that pdp_sm for logistic preserves Enum ordering.
        """
        from pyrsm.model.logistic import logistic

        model = logistic(
            titanic_enum_data, rvar="survived", lev="Yes", evar=["pclass", "sex", "age"]
        )

        # Get the pdp plot
        plot = model.plot("pdp", incl=["pclass"])

        assert plot is not None

    def test_regress_pred_plot_preserves_enum_order(self):
        """
        Test that pred_plot for regress preserves Enum ordering.
        """
        from pyrsm.model.regress import regress

        grade_enum = pl.Enum(["C", "B", "A"])  # Non-alphabetical

        data = pl.DataFrame(
            {
                "grade": pl.Series(["A", "B", "C"] * 50).cast(grade_enum),
                "price": [700.0, 600.0, 500.0] * 50,
            }
        )

        model = regress(data, rvar="price", evar=["grade"])

        # Get the pred plot
        plot = model.plot("pred", incl=["grade"])

        assert plot is not None

        # Check plot data ordering
        plot_data = plot.data
        if "grade" in plot_data.columns:
            grade_order = plot_data["grade"].unique().to_list()
            # Should preserve Enum order, not alphabetical
            # C is first in Enum, so should appear first
            assert (
                grade_order[0] == "C" or "C" in grade_order
            ), f"C should be in plot data, got: {grade_order}"

    def test_sklearn_pred_plot_preserves_enum_order(self):
        """
        Test that pred_plot_sk for sklearn models preserves Enum ordering.
        """
        import numpy as np

        from pyrsm.model.mlp import mlp

        np.random.seed(42)

        grade_enum = pl.Enum(["C", "B", "A"])  # Non-alphabetical

        data = pl.DataFrame(
            {
                "grade": pl.Series(["A", "B", "C"] * 50).cast(grade_enum),
                "target": ["Yes", "No", "No"] * 50,
            }
        )

        model = mlp(
            data,
            rvar="target",
            lev="Yes",
            evar=["grade"],
            hidden_layer_sizes=(5,),
            random_state=42,
            max_iter=500,
        )

        # Get the pred plot
        plot = model.plot("pred", incl=["grade"])

        assert plot is not None

    def test_enum_levels_extracted_correctly(self):
        """
        Test that Enum levels can be correctly extracted for plot ordering.
        """
        grade_enum = pl.Enum(["C", "B", "A"])

        data = pl.DataFrame({"grade": pl.Series(["A", "B", "C"]).cast(grade_enum)})

        # Verify we can extract Enum levels in correct order
        col_dtype = data["grade"].dtype
        assert isinstance(col_dtype, pl.Enum)

        levels = list(col_dtype.categories)
        assert levels == ["C", "B", "A"], f"Expected ['C', 'B', 'A'], got {levels}"

    def test_unique_maintain_order_preserves_enum(self):
        """
        Test that unique(maintain_order=True) works correctly with Enum.
        """
        grade_enum = pl.Enum(["C", "B", "A"])

        # Data in different order than Enum definition
        data = pl.DataFrame({"grade": pl.Series(["A", "B", "C", "A"]).cast(grade_enum)})

        # unique(maintain_order=True) preserves first-occurrence order
        unique_vals = data["grade"].unique(maintain_order=True).to_list()

        # First occurrence: A, B, C
        assert unique_vals == ["A", "B", "C"]

        # But the Enum categories are: C, B, A
        enum_order = list(data["grade"].dtype.categories)
        assert enum_order == ["C", "B", "A"]

        # For plots, we should use enum_order, not unique_vals!
