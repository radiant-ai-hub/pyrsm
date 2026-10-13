"""Tests for pyrsm.model.rforest module (Random Forest)."""

import io
import os
import sys

import matplotlib.pyplot as plt
import polars as pl
import pytest

from pyrsm.model.rforest import rforest

# Directory for saving plot comparisons
PLOT_DIR = "tests/plot_comparisons/model"
os.makedirs(PLOT_DIR, exist_ok=True)


@pytest.fixture(scope="module")
def titanic_data():
    """Load the titanic dataset for classification."""
    titanic = pl.read_parquet("examples/data/model/titanic.parquet")
    return titanic.drop_nulls(subset=["age"]).to_pandas()


@pytest.fixture(scope="module")
def salary_data():
    """Load the salary dataset for regression."""
    salary = pl.read_parquet("examples/data/basics/salary.parquet")
    return salary.to_pandas()


@pytest.fixture(scope="module")
def titanic_polars():
    """Load titanic as polars for testing."""
    titanic = pl.read_parquet("examples/data/model/titanic.parquet")
    return titanic.drop_nulls(subset=["age"])


class TestRforestClassification:
    """Tests for random forest classification."""

    def test_rforest_classification_basic(self, titanic_data):
        """Test basic classification fitting."""
        rf = rforest(
            data={"titanic": titanic_data},
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,  # Small for speed
        )
        assert rf.name == "titanic"
        assert rf.rvar == "survived"
        assert rf.mod_type == "classification"
        assert rf.fitted is not None

    def test_rforest_classification_polars_input(self, titanic_polars):
        """Test with polars DataFrame input."""
        rf = rforest(
            data=titanic_polars,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
            mod_type="classification",
            n_estimators=10,
        )
        assert rf.fitted is not None

    def test_rforest_classification_oob_score(self, titanic_data):
        """Test OOB score calculation."""
        rf = rforest(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            oob_score=True,
            n_estimators=20,
        )
        assert hasattr(rf.fitted, "oob_score_")
        assert 0 <= rf.fitted.oob_score_ <= 1


class TestRforestRegression:
    """Tests for random forest regression."""

    def test_rforest_regression_basic(self, salary_data):
        """Test basic regression fitting."""
        rf = rforest(
            data={"salary": salary_data},
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            n_estimators=10,
        )
        assert rf.name == "salary"
        assert rf.rvar == "salary"
        assert rf.mod_type == "regression"
        assert rf.fitted is not None

    def test_rforest_regression_oob_score(self, salary_data):
        """Test OOB R² for regression."""
        rf = rforest(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            oob_score=True,
            n_estimators=20,
        )
        assert hasattr(rf.fitted, "oob_score_")


class TestRforestSummary:
    """Tests for summary output."""

    def test_summary_classification(self, titanic_data):
        """Test summary for classification."""
        rf = rforest(
            data={"titanic": titanic_data},
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,
        )
        captured = io.StringIO()
        sys.stdout = captured
        rf.summary()
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Random Forest" in output
        assert "Classification" in output or "classification" in output

    def test_summary_regression(self, salary_data):
        """Test summary for regression."""
        rf = rforest(
            data={"salary": salary_data},
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            n_estimators=10,
        )
        captured = io.StringIO()
        sys.stdout = captured
        rf.summary()
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Random Forest" in output


class TestRforestPredict:
    """Tests for prediction functionality."""

    def test_predict_classification(self, titanic_data):
        """Test classification predictions."""
        rf = rforest(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,
        )
        pred = rf.predict()
        assert "prediction" in pred.columns
        # Predictions should be probabilities
        assert all(pred["prediction"] >= 0)
        assert all(pred["prediction"] <= 1)

    def test_predict_regression(self, salary_data):
        """Test regression predictions."""
        rf = rforest(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            n_estimators=10,
        )
        pred = rf.predict()
        assert "prediction" in pred.columns
        # Predictions should be non-negative (salary)
        assert (pred["prediction"] >= 0).all()
        # Mean prediction should be in reasonable range
        assert pred["prediction"].mean() > 50000

    def test_predict_new_data(self, titanic_data):
        """Test prediction with new data."""
        rf = rforest(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,
        )
        new_data = titanic_data.head(10)
        pred = rf.predict(data=new_data)
        assert len(pred) == 10


class TestRforestPlot:
    """Tests for plotting functionality."""

    def test_plot_pred_classification(self, titanic_data):
        """Test prediction plot for classification."""
        rf = rforest(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
            mod_type="classification",
            n_estimators=10,
        )
        rf.plot(plots="pred", incl=["age"])
        plt.savefig(f"{PLOT_DIR}/rforest_pred_class.png", dpi=100, bbox_inches="tight")
        plt.close("all")

    def test_plot_pred_regression(self, salary_data):
        """Test prediction plot for regression."""
        rf = rforest(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            n_estimators=10,
        )
        rf.plot(plots="pred", incl=["yrs_since_phd"])
        plt.savefig(f"{PLOT_DIR}/rforest_pred_reg.png", dpi=100, bbox_inches="tight")
        plt.close("all")

    def test_plot_pip(self, titanic_data):
        """Test variable importance plot."""
        rf = rforest(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,
        )
        rf.plot(plots="pip")
        plt.savefig(f"{PLOT_DIR}/rforest_pip.png", dpi=100, bbox_inches="tight")
        plt.close("all")

    def test_plot_dashboard(self, salary_data):
        """Test dashboard plot for regression."""
        rf = rforest(
            data=salary_data,
            rvar="salary",
            evar=["yrs_since_phd", "yrs_service"],
            mod_type="regression",
            n_estimators=10,
        )
        rf.plot(plots="dashboard", nobs=100)
        plt.savefig(f"{PLOT_DIR}/rforest_dashboard.png", dpi=100, bbox_inches="tight")
        plt.close("all")


class TestRforestFeatureImportance:
    """Tests for feature importance."""

    def test_feature_importance_exists(self, titanic_data):
        """Test that feature importance is computed."""
        rf = rforest(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,
        )
        assert hasattr(rf.fitted, "feature_importances_")
        assert len(rf.fitted.feature_importances_) > 0

    def test_feature_importance_sums_to_one(self, titanic_data):
        """Test feature importances sum to approximately 1."""
        rf = rforest(
            data=titanic_data,
            rvar="survived",
            lev="Yes",
            evar=["age", "sex", "pclass"],
            mod_type="classification",
            n_estimators=10,
        )
        imp_sum = sum(rf.fitted.feature_importances_)
        assert 0.99 <= imp_sum <= 1.01


def test_ice_lines_stay_inside_the_y_axis():
    """ICE lines spread wider than the PDP; the axis must include them (not clip)."""
    import polars as pl

    import pyrsm as rsm

    t = pl.read_parquet("examples/data/model/titanic.parquet")
    rf = rsm.model.rforest({"titanic": t}, rvar="survived", lev="Yes", evar=["pclass", "sex", "age"], n_estimators=20)
    p = rf.plot("pdp", incl=["age"], ice=True, ice_nobs=50)
    ice = p.layers[0].geom.data
    low, high = [s for s in p.scales if "y" in s.aesthetics][0].limits
    assert low <= ice["prediction"].min() and high >= ice["prediction"].max()


class TestRforestTreePlot:
    """One tree of the forest, drawn with mermaid as crtree.plot("tree")."""

    @pytest.fixture(scope="class")
    def rf(self, titanic_polars):
        return rforest(
            {"titanic": titanic_polars}, rvar="survived", lev="Yes",
            evar=["pclass", "sex", "age"], mod_type="classification",
        )

    def test_levels_not_dummies(self, rf):
        src = rf.to_mermaid(tree=1, depth=3)
        assert src.startswith("graph LR")
        # dummy columns (sex_male, pclass_3rd) become variables and levels
        assert "sex_male" not in src and "pclass_3rd" not in src
        assert "|female|" in src and "|male|" in src
        assert "|1st,2nd|" in src and "|3rd|" in src
        # the root holds every (bootstrap) row and the share of survivors
        assert 'click id1 callback "<b>n:</b> 1,043<br><b>p:</b>' in src

    def test_depth_limits_the_drawing(self, rf):
        shallow = rf.to_mermaid(tree=1, depth=2)
        assert " more;" in shallow and "splits]" in shallow
        full = rf.to_mermaid(tree=1, depth=None)
        assert " more;" not in full and " leaf;" in full
        assert full.count("---") > shallow.count("---")

    def test_each_tree_differs(self, rf):
        assert rf.to_mermaid(tree=1) != rf.to_mermaid(tree=2)

    def test_tree_out_of_range(self, rf):
        with pytest.raises(ValueError, match="between 1 and 100"):
            rf.to_mermaid(tree=101)

    def test_plot_tree(self, rf):
        from pyrsm.model._mermaid import MermaidDiagram

        p = rf.plot("tree", tree=3, depth=2, orient="TD")
        assert isinstance(p, MermaidDiagram)
        assert str(p).startswith("graph TD")

    def test_regression_tree(self):
        catalog = pl.read_parquet("examples/data/model/catalog.parquet")
        rf = rforest({"catalog": catalog}, rvar="Sales", evar=["Income", "HH_size", "Age"], mod_type="regression")
        src = rf.to_mermaid(tree=1, depth=2)
        assert "<b>b:</b>" in src and "<b>p:</b>" not in src
        # HH_size is an integer: its cut-offs are whole numbers
        for line in src.splitlines():
            if line.startswith("id") and "[HH_size] ---" in line:
                cut = line.split("|")[1].split()[-1]
                assert float(cut).is_integer()


class TestTreeBoxes:
    """A large tree is drawn to fewer levels: at most MAX_BOXES boxes."""

    @pytest.fixture(scope="class")
    def big(self):
        d = pl.read_parquet(os.path.join(os.path.dirname(__file__), "..", "examples", "data", "model", "diamonds.parquet"))
        return rforest(
            {"diamonds": d}, rvar="price", evar=["carat", "clarity", "cut", "color"],
            mod_type="regression", n_estimators=3,
        )

    def test_boxes_count_levels(self, big):
        from pyrsm.model._sktree import boxes_drawn, node_levels

        est = big.fitted.estimators_[0]
        assert boxes_drawn(est, 1) == 3
        assert boxes_drawn(est, None) == est.tree_.node_count
        assert node_levels(est).max() == est.get_depth()

    def test_deep_tree_is_cut(self, big, capsys):
        from pyrsm.model._sktree import MAX_BOXES, boxes_drawn

        est = big.fitted.estimators_[0]
        assert boxes_drawn(est, 10) > MAX_BOXES  # diamonds at 10 levels is too big
        shown = big.tree_depth(1, 10)
        assert 1 <= shown < 10
        assert boxes_drawn(est, shown) <= MAX_BOXES < boxes_drawn(est, shown + 1)
        assert big.tree_depth(1, None) == shown
        big.plot("tree", tree=1, depth=10)
        assert f"Showing {shown} levels of tree 1: 10 levels would draw" in capsys.readouterr().out

    def test_small_tree_is_not_cut(self, titanic_polars, capsys):
        rf = rforest(
            {"titanic": titanic_polars}, rvar="survived", lev="Yes",
            evar=["pclass", "sex", "age"], mod_type="classification", n_estimators=3,
        )
        assert rf.tree_depth(1, 10) == 10
        assert rf.tree_depth(1, 10, max_boxes=20) < 10
        rf.plot("tree", tree=1, depth=10)
        assert capsys.readouterr().out == ""


class TestFitLines:
    """The plain summary prints the numbers the formatted report shows."""

    def test_rforest_regression_is_out_of_bag(self, capsys):
        catalog = pl.read_parquet("examples/data/model/catalog.parquet")
        rf = rforest({"catalog": catalog}, rvar="Sales", evar=["Income", "HH_size", "Age"], mod_type="regression")
        rf.summary()
        out = capsys.readouterr().out
        assert "Model fit            : out-of-bag (OOB)" in out
        from pyrsm.model.report import fit_metrics

        m = fit_metrics(rf)
        assert m["fit_kind"] == "oob" and m["r2"] < 0.5
        assert f"{m['r2']:.3f}" in out

    def test_rforest_classification_auc_is_out_of_bag(self, capsys, titanic_polars):
        rf = rforest({"titanic": titanic_polars}, rvar="survived", lev="Yes", evar=["pclass", "sex", "age"])
        rf.summary()
        assert "(OOB)" in capsys.readouterr().out

    def test_mlp_regression_in_response_units(self, capsys):
        from pyrsm.model.mlp import mlp

        catalog = pl.read_parquet("examples/data/model/catalog.parquet")
        nn = mlp({"catalog": catalog}, rvar="Sales", evar=["Income", "HH_size", "Age"], mod_type="regression")
        nn.summary()
        out = capsys.readouterr().out
        assert "Model fit            : estimation data" in out
        from pyrsm.model.report import fit_metrics

        # Sales is in dollars: the RMSE is tens of dollars, not a standardized ~0.7
        assert fit_metrics(nn)["rmse"] > 10
        assert fit_metrics(nn)["mae"] > 10
