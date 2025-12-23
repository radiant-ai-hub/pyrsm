"""
Smoke tests for lazy import functionality.

These tests verify that deferred imports execute correctly when modules
are instantiated with minimal in-memory data.
"""

import pandas as pd
import polars as pl
import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tiny_df_pandas():
    """Create a tiny pandas DataFrame for testing."""
    return pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 5, 4, 5],
            "group": ["A", "A", "B", "B", "B"],
            "binary": ["yes", "no", "yes", "yes", "no"],
        }
    )


@pytest.fixture
def tiny_df_polars():
    """Create a tiny polars DataFrame for testing."""
    return pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 5, 4, 5],
            "group": ["A", "A", "B", "B", "B"],
            "binary": ["yes", "no", "yes", "yes", "no"],
        }
    )


# =============================================================================
# Basics Module Smoke Tests
# =============================================================================


class TestBasicsSmoke:
    """Smoke tests for pyrsm.basics modules with lazy imports."""

    def test_compare_means_instantiation(self, tiny_df_polars):
        """Test compare_means can be instantiated and lazy imports work."""
        from pyrsm.basics import compare_means

        cm = compare_means(
            data=tiny_df_polars,
            var1="group",
            var2="x",
        )
        assert cm is not None
        assert hasattr(cm, "descriptive_stats")
        assert hasattr(cm, "comp_stats")

    def test_compare_props_instantiation(self, tiny_df_polars):
        """Test compare_props can be instantiated and lazy imports work."""
        from pyrsm.basics import compare_props

        cp = compare_props(
            data=tiny_df_polars,
            var1="group",
            var2="binary",
            lev="yes",
        )
        assert cp is not None
        assert hasattr(cp, "descriptive_stats")
        assert hasattr(cp, "comp_stats")

    def test_correlation_instantiation(self, tiny_df_polars):
        """Test correlation can be instantiated and lazy imports work."""
        from pyrsm.basics import correlation

        cr = correlation(tiny_df_polars.select(["x", "y"]))
        assert cr is not None
        assert hasattr(cr, "cr")
        assert hasattr(cr, "cp")
        assert cr.cr.shape == (2, 2)

    def test_cross_tabs_instantiation(self, tiny_df_polars):
        """Test cross_tabs can be instantiated and lazy imports work."""
        from pyrsm.basics import cross_tabs

        ct = cross_tabs(tiny_df_polars, "group", "binary")
        assert ct is not None
        assert hasattr(ct, "observed")
        assert hasattr(ct, "expected")
        assert hasattr(ct, "chisq_test")

    def test_single_mean_instantiation(self, tiny_df_polars):
        """Test single_mean can be instantiated and lazy imports work."""
        from pyrsm.basics import single_mean

        sm = single_mean(tiny_df_polars, var="x", comp_value=3)
        assert sm is not None
        assert hasattr(sm, "t_val")
        assert hasattr(sm, "p_val")
        assert hasattr(sm, "ci")

    def test_single_prop_instantiation(self, tiny_df_polars):
        """Test single_prop can be instantiated and lazy imports work."""
        from pyrsm.basics import single_prop

        sp = single_prop(tiny_df_polars, var="binary", lev="yes", comp_value=0.5)
        assert sp is not None
        assert hasattr(sp, "p")
        assert hasattr(sp, "p_val")
        assert hasattr(sp, "ci")

    def test_goodness_instantiation(self, tiny_df_polars):
        """Test goodness can be instantiated and lazy imports work."""
        from pyrsm.basics import goodness

        gf = goodness(tiny_df_polars, var="group")
        assert gf is not None
        assert hasattr(gf, "observed")
        assert hasattr(gf, "expected")
        assert hasattr(gf, "chisq")


# =============================================================================
# Model Module Smoke Tests
# =============================================================================


class TestModelSmoke:
    """Smoke tests for pyrsm.model modules with lazy imports."""

    def test_regress_instantiation(self, tiny_df_polars):
        """Test regress can be instantiated and lazy imports work."""
        from pyrsm.model.regress import regress

        reg = regress(tiny_df_polars, rvar="y", evar=["x"])
        assert reg is not None
        assert hasattr(reg, "fitted")
        assert hasattr(reg, "coef")

    def test_logistic_instantiation(self, tiny_df_polars):
        """Test logistic can be instantiated and lazy imports work."""
        from pyrsm.model.logistic import logistic

        log = logistic(tiny_df_polars, rvar="binary", lev="yes", evar=["x"])
        assert log is not None
        assert hasattr(log, "fitted")
        assert hasattr(log, "coef")

    def test_rforest_instantiation(self, tiny_df_polars):
        """Test rforest can be instantiated and lazy imports work."""
        from pyrsm.model.rforest import rforest

        rf = rforest(
            tiny_df_polars,
            rvar="binary",
            lev="yes",
            evar=["x", "y"],
            n_estimators=10,
            oob_score=False,
        )
        assert rf is not None
        assert hasattr(rf, "fitted")

    def test_mlp_instantiation(self, tiny_df_polars):
        """Test mlp can be instantiated and lazy imports work."""
        from pyrsm.model.mlp import mlp

        nn = mlp(
            tiny_df_polars,
            rvar="binary",
            lev="yes",
            evar=["x", "y"],
            hidden_layer_sizes=(2,),
            max_iter=100,
        )
        assert nn is not None
        assert hasattr(nn, "fitted")

    def test_xgboost_instantiation(self, tiny_df_polars):
        """Test xgboost can be instantiated and lazy imports work."""
        from pyrsm.model.xgboost import xgboost

        xgb = xgboost(
            tiny_df_polars,
            rvar="binary",
            lev="yes",
            evar=["x", "y"],
            n_estimators=10,
        )
        assert xgb is not None
        assert hasattr(xgb, "fitted")


# =============================================================================
# Lazy Import Path Tests
# =============================================================================


class TestLazyImportPaths:
    """Test that lazy loading functions are called correctly."""

    def test_sig_stars_from_utils(self):
        """Test sig_stars is available from pyrsm.utils (not model.model)."""
        from pyrsm.utils import sig_stars

        result = sig_stars([0.0001, 0.005, 0.03, 0.08, 0.5])
        assert len(result) == 5
        assert result[0] == "***"  # < 0.001
        assert result[1] == "**"  # < 0.01
        assert result[2] == "*"  # < 0.05
        assert result[3] == "."  # < 0.1
        assert result[4] == " "  # >= 0.1

    def test_pandas_lazy_loading(self):
        """Test pandas is lazily loaded in utils."""
        from pyrsm.utils import check_dataframe

        # Should work with pandas DataFrame
        df_pd = pd.DataFrame({"x": [1, 2, 3]})
        result = check_dataframe(df_pd)
        assert isinstance(result, pl.DataFrame)


# =============================================================================
# Import Error Handling Tests
# =============================================================================


class TestImportErrorHandling:
    """Test graceful handling when optional deps are missing."""

    def test_core_import_without_plotnine(self):
        """Test core modules can be imported without plotnine.

        Note: This is a conceptual test - in CI we test this by
        installing only core deps and verifying import works.
        """
        # Just verify the module structure is correct
        from pyrsm.basics import correlation, cross_tabs

        assert correlation is not None
        assert cross_tabs is not None

    def test_core_import_without_xgboost(self):
        """Test core modules can be imported without xgboost.

        Note: This is a conceptual test - in CI we test this by
        installing only core deps and verifying import works.
        """
        from pyrsm.basics import compare_means, single_mean

        assert compare_means is not None
        assert single_mean is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
