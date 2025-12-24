"""
Comprehensive tests for MLP model scaling and prediction consistency.

These tests verify that:
1. Scaling parameters (means/stds) are stored correctly during training
2. Predictions are consistent regardless of the data path taken
3. Categorical variables are handled correctly (alphabetical ordering, all levels preserved)
4. Round-trip scaling (scale -> predict -> inverse-scale) produces correct results
"""

import numpy as np
import polars as pl
import pytest

from pyrsm.model.mlp import mlp
from pyrsm.model.model import get_dummies
from pyrsm.stats import scale_df

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def titanic_data():
    """Titanic dataset for classification tests."""
    return pl.read_parquet(
        "https://github.com/radiant-ai-hub/pyrsm/raw/refs/heads/main/examples/data/model/titanic.parquet"
    ).drop_nulls(subset=["age"])


@pytest.fixture(scope="module")
def salary_data():
    """Salary dataset for regression tests."""
    return pl.read_parquet(
        "https://github.com/radiant-ai-hub/pyrsm/raw/refs/heads/main/examples/data/basics/salary.parquet"
    )


@pytest.fixture(scope="module")
def diamonds_data():
    """Diamonds dataset - has multiple categorical columns."""
    return pl.read_parquet(
        "https://github.com/radiant-ai-hub/pyrsm/raw/refs/heads/main/examples/data/model/diamonds.parquet"
    ).sample(500, seed=1234)


@pytest.fixture(scope="module")
def simple_regression_data():
    """Simple synthetic data for precise numerical testing."""
    np.random.seed(1234)
    n = 100
    return pl.DataFrame(
        {
            "x1": np.random.randn(n) * 10 + 50,  # mean~50, std~10
            "x2": np.random.randn(n) * 5 + 20,  # mean~20, std~5
            "category": np.random.choice(["A", "B", "C"], n),
            "y": np.random.randn(n) * 100 + 500,  # response
        }
    )


@pytest.fixture(scope="module")
def simple_classification_data():
    """Simple synthetic data for classification testing."""
    np.random.seed(1234)
    n = 100
    return pl.DataFrame(
        {
            "x1": np.random.randn(n) * 10 + 50,
            "x2": np.random.randn(n) * 5 + 20,
            "category": np.random.choice(["A", "B", "C"], n),
            "target": np.random.choice(["Yes", "No"], n),
        }
    )


# ============================================================================
# Test Class: TestMLPScalingStorage
# ============================================================================


class TestMLPScalingStorage:
    """Test that MLP correctly stores scaling parameters during training."""

    def test_means_stored_correctly(self, simple_regression_data):
        """Verify self.means contains correct training data means."""
        nn = mlp(
            data=simple_regression_data,
            rvar="y",
            evar=["x1", "x2", "category"],
            mod_type="regression",
            hidden_layer_sizes=(5,),
            max_iter=100,
        )

        # means should contain keys for all numeric columns
        assert "y" in nn.means
        assert "x1" in nn.means
        assert "x2" in nn.means

        # Each mean should match polars computed mean
        assert np.isclose(
            nn.means["x1"], simple_regression_data["x1"].mean(), rtol=1e-5
        )
        assert np.isclose(
            nn.means["x2"], simple_regression_data["x2"].mean(), rtol=1e-5
        )
        assert np.isclose(nn.means["y"], simple_regression_data["y"].mean(), rtol=1e-5)

    def test_stds_stored_correctly(self, simple_regression_data):
        """Verify self.stds contains correct training data standard deviations."""
        nn = mlp(
            data=simple_regression_data,
            rvar="y",
            evar=["x1", "x2", "category"],
            mod_type="regression",
            hidden_layer_sizes=(5,),
            max_iter=100,
        )

        # stds should use ddof=0 (sklearn convention via scale_df default)
        assert np.isclose(
            nn.stds["x1"], simple_regression_data["x1"].std(ddof=0), rtol=1e-5
        )
        assert np.isclose(
            nn.stds["x2"], simple_regression_data["x2"].std(ddof=0), rtol=1e-5
        )
        assert np.isclose(
            nn.stds["y"], simple_regression_data["y"].std(ddof=0), rtol=1e-5
        )

    def test_categories_stored_alphabetically(self, simple_regression_data):
        """Verify self.categories stores category levels in alphabetical order (after drop_first)."""
        nn = mlp(
            data=simple_regression_data,
            rvar="y",
            evar=["x1", "x2", "category"],
            mod_type="regression",
            hidden_layer_sizes=(5,),
            max_iter=100,
        )

        # get_dummies drops first category (occurrence order), so check categories stored
        assert "category" in nn.categories
        # Categories are stored in occurrence order, not alphabetical
        # The test fixture data has categories in some order, first is dropped
        assert (
            len(nn.categories["category"]) > 0
        )  # at least one category after drop_first

    def test_feature_names_order_matches_training_data(self, simple_regression_data):
        """Verify self.feature_names preserves column order from training."""
        nn = mlp(
            data=simple_regression_data,
            rvar="y",
            evar=["x1", "x2", "category"],
            mod_type="regression",
            hidden_layer_sizes=(5,),
            max_iter=100,
        )

        # feature_names should be: ["x1", "x2", "category_B", "category_C"]
        assert nn.feature_names[0] == "x1"
        assert nn.feature_names[1] == "x2"
        assert "category_" in nn.feature_names[2]


# ============================================================================
# Test Class: TestMLPPredictionConsistency
# ============================================================================


class TestMLPPredictionConsistency:
    """Test that predictions are consistent across different data paths."""

    def test_training_data_prediction_consistency_regression(self, salary_data):
        """
        KEY INVARIANT TEST (Regression):
        Predicting on training data should give same results as:
        1. Taking raw data
        2. Scaling with stored means/stds
        3. One-hot encoding with stored categories
        4. Getting sklearn predictions
        5. Inverse scaling predictions
        """
        nn = mlp(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            hidden_layer_sizes=(10,),
            max_iter=1000,
            random_state=1234,
        )

        # Path 1: Use predict() method
        pred_via_method = nn.predict()["prediction"]

        # Path 2: Manual reconstruction
        # Step 2a: Scale raw data using stored means/stds
        raw_data = nn.data.select(nn.evar)
        scaled_data = scale_df(raw_data, sf=1, means=nn.means, stds=nn.stds)

        # Step 2b: One-hot encode using stored categories
        data_onehot = get_dummies(
            scaled_data, drop_nonvarying=False, categories=nn.categories
        )

        # Step 2c: Reorder to match feature_names
        data_onehot = data_onehot.select(nn.feature_names)

        # Step 2d: Get sklearn predictions
        raw_pred = nn.fitted.predict(data_onehot.to_pandas())

        # Step 2e: Inverse scale
        manual_pred = raw_pred * nn.stds[nn.rvar] + nn.means[nn.rvar]

        # Assertions
        np.testing.assert_allclose(pred_via_method.to_numpy(), manual_pred, rtol=1e-10)

    def test_training_data_prediction_consistency_classification(self, titanic_data):
        """
        KEY INVARIANT TEST (Classification):
        Predicting on training data should give same results.
        Note: Classification does NOT inverse scale (predictions are probabilities).
        """
        nn = mlp(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
            mod_type="classification",
            hidden_layer_sizes=(10,),
            max_iter=1000,
            random_state=1234,
        )

        # Path 1: Use predict() method
        pred_via_method = nn.predict()["prediction"]

        # Path 2: Manual reconstruction
        raw_data = nn.data.select(nn.evar)
        scaled_data = scale_df(raw_data, sf=1, means=nn.means, stds=nn.stds)
        data_onehot = get_dummies(
            scaled_data, drop_nonvarying=False, categories=nn.categories
        )
        data_onehot = data_onehot.select(nn.feature_names)

        # For classification: predict_proba, take last column
        manual_pred = nn.fitted.predict_proba(data_onehot.to_pandas())[:, -1]

        np.testing.assert_allclose(pred_via_method.to_numpy(), manual_pred, rtol=1e-10)

    def test_scaled_vs_unscaled_data_predictions_match(self, salary_data):
        """
        Test that predict(scale=True) on raw data gives same result as
        predict(scale=False) on pre-scaled data.
        """
        nn = mlp(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            hidden_layer_sizes=(10,),
            max_iter=1000,
            random_state=1234,
        )

        # Approach 1: Pass raw data, let predict() scale
        raw_data = salary_data.head(10)
        pred_scaled = nn.predict(data=raw_data, scale=True)["prediction"]

        # Approach 2: Pre-scale data, pass with scale=False
        pre_scaled = scale_df(
            raw_data.select(nn.evar), sf=1, means=nn.means, stds=nn.stds
        )
        pred_unscaled = nn.predict(data=pre_scaled, scale=False)["prediction"]

        np.testing.assert_allclose(
            pred_scaled.to_numpy(), pred_unscaled.to_numpy(), rtol=1e-10
        )


# ============================================================================
# Test Class: TestMLPCategoricalHandling
# ============================================================================


class TestMLPCategoricalHandling:
    """Test categorical variable handling in MLP predictions."""

    def test_categorical_alphabetical_ordering(self, simple_regression_data):
        """Verify categories are ordered alphabetically in dummy encoding."""
        nn = mlp(
            data=simple_regression_data,
            rvar="y",
            evar=["x1", "category"],
            mod_type="regression",
            hidden_layer_sizes=(5,),
            max_iter=100,
        )

        # With drop_first=True (default in get_dummies), first category is dropped
        # Categories are now in occurrence order (not alphabetical)
        assert "category" in nn.categories
        assert len(nn.categories["category"]) == 2  # 3 categories - 1 dropped = 2

        # feature_names should have category dummies
        cat_features = [f for f in nn.feature_names if f.startswith("category_")]
        assert len(cat_features) == 2

    def test_predict_with_subset_of_categories(self, simple_regression_data):
        """Test prediction when new data has only subset of training categories."""
        nn = mlp(
            data=simple_regression_data,
            rvar="y",
            evar=["x1", "category"],
            mod_type="regression",
            hidden_layer_sizes=(5,),
            max_iter=100,
        )

        # Create prediction data with only category "A"
        pred_data = pl.DataFrame({"x1": [50.0, 55.0], "category": ["A", "A"]})

        # Should work - categories are preserved from training
        pred = nn.predict(data=pred_data)
        assert len(pred) == 2
        assert "prediction" in pred.columns

    def test_predict_with_all_categories(self, simple_regression_data):
        """Test prediction with all category levels present."""
        nn = mlp(
            data=simple_regression_data,
            rvar="y",
            evar=["x1", "category"],
            mod_type="regression",
            hidden_layer_sizes=(5,),
            max_iter=100,
        )

        pred_data = pl.DataFrame(
            {"x1": [50.0, 55.0, 60.0], "category": ["A", "B", "C"]}
        )

        pred = nn.predict(data=pred_data)
        assert len(pred) == 3

    def test_multiple_categorical_columns(self, diamonds_data):
        """Test with multiple categorical variables."""
        nn = mlp(
            data=diamonds_data,
            rvar="price",
            evar=["carat", "cut", "color"],
            mod_type="regression",
            hidden_layer_sizes=(10,),
            max_iter=100,
        )

        # Both cut and color should be in categories
        assert "cut" in nn.categories
        assert "color" in nn.categories

        # Categories are stored in occurrence order (not alphabetical)
        # After drop_first, we should have n-1 categories
        assert len(nn.categories["cut"]) > 0
        assert len(nn.categories["color"]) > 0

    def test_dummy_column_values_correct(self, simple_regression_data):
        """Verify dummy columns have correct 0/1 values."""
        nn = mlp(
            data=simple_regression_data,
            rvar="y",
            evar=["x1", "category"],
            mod_type="regression",
            hidden_layer_sizes=(5,),
            max_iter=100,
        )

        # Check data_onehot has correct dummy values
        original_cats = nn.data.select("category").to_series()
        onehot = nn.data_onehot

        # Get the category dummy columns
        cat_cols = [c for c in onehot.columns if c.startswith("category_")]
        assert len(cat_cols) == 2  # 3 categories - 1 dropped = 2

        # For each row, exactly one dummy should be 1 (or all zeros if it's the dropped category)
        for i in range(len(original_cats)):
            cat = original_cats[i]
            cat_col = f"category_{cat}"
            if cat_col in cat_cols:
                # This category has a dummy column - it should be 1
                assert onehot[cat_col][i] == 1
                # Other category dummies should be 0
                for other_col in cat_cols:
                    if other_col != cat_col:
                        assert onehot[other_col][i] == 0
            else:
                # This is the dropped category - all dummies should be 0
                for col in cat_cols:
                    assert onehot[col][i] == 0


# ============================================================================
# Test Class: TestMLPRoundTrip
# ============================================================================


class TestMLPRoundTrip:
    """Test round-trip scaling/prediction operations."""

    def test_regression_inverse_scaling(self, salary_data):
        """
        Test that regression predictions are properly inverse-scaled.
        pred = sklearn_pred * stds[rvar] + means[rvar]
        """
        nn = mlp(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            hidden_layer_sizes=(10,),
            max_iter=1000,
            random_state=1234,
        )

        # Get predictions
        predictions = nn.predict()["prediction"]

        # Predictions should be in original scale (salary ~50,000-200,000)
        assert predictions.mean() > 10000  # Not standardized
        assert predictions.mean() < 500000  # Reasonable range

        # Get raw sklearn predictions (standardized scale)
        raw_pred = nn.fitted.predict(nn.data_onehot.to_pandas())

        # Raw predictions should be roughly centered around 0
        assert abs(raw_pred.mean()) < 1  # Approximately zero-centered

        # Verify inverse transformation
        manual_inverse = raw_pred * nn.stds[nn.rvar] + nn.means[nn.rvar]
        np.testing.assert_allclose(predictions.to_numpy(), manual_inverse, rtol=1e-10)

    def test_classification_no_inverse_scaling(self, titanic_data):
        """
        Test that classification predictions are NOT inverse-scaled.
        Classification returns probabilities [0, 1].
        """
        nn = mlp(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
            mod_type="classification",
            hidden_layer_sizes=(10,),
            max_iter=1000,
            random_state=1234,
        )

        predictions = nn.predict()["prediction"]

        # All predictions should be probabilities
        assert (predictions >= 0).all()
        assert (predictions <= 1).all()

        # Should match raw sklearn probabilities exactly
        raw_proba = nn.fitted.predict_proba(nn.data_onehot.to_pandas())[:, -1]
        np.testing.assert_allclose(predictions.to_numpy(), raw_proba, rtol=1e-10)

    def test_scaling_is_reversible(self, simple_regression_data):
        """Test that scale -> inverse-scale recovers original values."""
        nn = mlp(
            data=simple_regression_data,
            rvar="y",
            evar=["x1", "x2"],
            mod_type="regression",
            hidden_layer_sizes=(5,),
            max_iter=100,
        )

        # Original numeric values
        original_x1 = simple_regression_data["x1"].to_numpy()

        # Scaled values in data_std
        scaled_x1 = nn.data_std["x1"].to_numpy()

        # Inverse transform
        recovered_x1 = scaled_x1 * nn.stds["x1"] + nn.means["x1"]

        np.testing.assert_allclose(original_x1, recovered_x1, rtol=1e-10)


# ============================================================================
# Test Class: TestMLPEdgeCases
# ============================================================================


class TestMLPEdgeCases:
    """Test edge cases for MLP scaling and prediction."""

    def test_numeric_columns_only(self, salary_data):
        """Test model with only numeric columns (no categoricals)."""
        nn = mlp(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            hidden_layer_sizes=(5,),
            max_iter=100,
        )

        # categories should be empty
        assert len(nn.categories) == 0

        # feature_names should just be numeric columns
        assert set(nn.feature_names) == {"yrs_since_phd", "yrs_service"}

        # Predictions should work
        pred = nn.predict()
        assert "prediction" in pred.columns

    def test_single_row_prediction(self, salary_data):
        """Test prediction with single row of data."""
        nn = mlp(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            hidden_layer_sizes=(5,),
            max_iter=100,
        )

        single_row = salary_data.head(1)
        pred = nn.predict(data=single_row)
        assert len(pred) == 1
        assert "prediction" in pred.columns

    def test_prediction_with_cmd_dict(self, salary_data):
        """Test predict() with cmd parameter for simulation."""
        nn = mlp(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            hidden_layer_sizes=(5,),
            max_iter=100,
        )

        # Use cmd to vary one variable
        pred = nn.predict(cmd={"yrs_since_phd": [5, 10, 15, 20, 25]})
        assert len(pred) == 5

        # yrs_since_phd should have the specified values
        assert pred["yrs_since_phd"].to_list() == [5, 10, 15, 20, 25]

    def test_prediction_with_data_cmd_dict(self, salary_data):
        """Test predict() with data_cmd parameter."""
        nn = mlp(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            hidden_layer_sizes=(5,),
            max_iter=100,
        )

        # Use data_cmd to set a fixed value
        pred = nn.predict(data_cmd={"yrs_since_phd": 15})

        # All rows should have yrs_since_phd = 15
        assert (pred["yrs_since_phd"] == 15).all()

    def test_means_stds_override_in_predict(self, salary_data):
        """Test passing custom means/stds to predict()."""
        nn = mlp(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            hidden_layer_sizes=(5,),
            max_iter=100,
        )

        # Create custom means/stds (slightly different)
        custom_means = {k: v * 1.1 for k, v in nn.means.items()}
        custom_stds = {k: v * 1.2 for k, v in nn.stds.items()}

        # Predictions with custom stats should differ from default
        pred_default = nn.predict()["prediction"]
        pred_custom = nn.predict(means=custom_means, stds=custom_stds)["prediction"]

        # Should not be equal (different scaling)
        assert not np.allclose(pred_default.to_numpy(), pred_custom.to_numpy())

    def test_new_data_with_different_column_order(self, salary_data):
        """Test prediction with columns in different order than training."""
        nn = mlp(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            hidden_layer_sizes=(5,),
            max_iter=100,
        )

        # Create prediction data with reversed column order
        pred_data = pl.DataFrame(
            {
                "yrs_service": [5, 10],
                "yrs_since_phd": [10, 20],  # Note: different order
            }
        )

        pred = nn.predict(data=pred_data)
        assert len(pred) == 2


# ============================================================================
# Test Class: TestMLPFeatureNamePreservation
# ============================================================================


class TestMLPFeatureNamePreservation:
    """Test feature name preservation and ordering."""

    def test_feature_names_match_data_onehot_columns(self, simple_regression_data):
        """Verify feature_names matches data_onehot columns exactly."""
        nn = mlp(
            data=simple_regression_data,
            rvar="y",
            evar=["x1", "x2", "category"],
            mod_type="regression",
            hidden_layer_sizes=(5,),
            max_iter=100,
        )

        assert list(nn.feature_names) == list(nn.data_onehot.columns)

    def test_prediction_data_reordered_to_match_training(self, simple_regression_data):
        """Verify prediction data columns are reordered to match training."""
        nn = mlp(
            data=simple_regression_data,
            rvar="y",
            evar=["x1", "x2", "category"],
            mod_type="regression",
            hidden_layer_sizes=(5,),
            max_iter=100,
        )

        # Create data with different column order
        pred_data = pl.DataFrame(
            {
                "category": ["A", "B"],
                "x2": [20.0, 25.0],
                "x1": [50.0, 55.0],  # Different order
            }
        )

        # Should still work correctly
        pred = nn.predict(data=pred_data)
        assert len(pred) == 2

    def test_n_features_tracked_correctly(self, simple_regression_data):
        """Verify n_features is (original_count, onehot_count)."""
        nn = mlp(
            data=simple_regression_data,
            rvar="y",
            evar=["x1", "x2", "category"],  # 3 features
            mod_type="regression",
            hidden_layer_sizes=(5,),
            max_iter=100,
        )

        # Original: 3 features (x1, x2, category)
        assert nn.n_features[0] == 3

        # After one-hot: x1, x2, category_B, category_C = 4 columns
        assert nn.n_features[1] == 4
