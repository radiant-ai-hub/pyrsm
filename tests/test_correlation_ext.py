import warnings

import numpy as np
import polars as pl

from pyrsm.basics.correlation import correlation


class TestCorrelationBaseline:
    """Baseline tests to capture outputs before plotnine conversion."""

    def test_baseline_correlation_matrices(self, salary_data):
        """Capture baseline correlation, p-value, and covariance matrices."""
        numeric_cols = ["salary", "yrs_since_phd", "yrs_service"]
        cr = correlation({"salary": salary_data}, vars=numeric_cols)

        # Verify structure
        assert cr.cr.shape == (3, 3)
        assert cr.cp.shape == (3, 3)
        assert cr.cv.shape == (3, 3)

        # Verify correlations are in valid range
        assert np.all(np.abs(cr.cr) <= 1.0)

        # Verify p-values are in valid range
        assert np.all((cr.cp >= 0) & (cr.cp <= 1))

        # Verify symmetry
        assert np.allclose(cr.cr, cr.cr.T)
        assert np.allclose(cr.cp, cr.cp.T)

    def test_baseline_correlation_methods(self, salary_data):
        """Test different correlation methods."""
        numeric_cols = ["salary", "yrs_since_phd", "yrs_service"]

        cr_pearson = correlation(
            {"salary": salary_data}, vars=numeric_cols, method="pearson"
        )
        cr_spearman = correlation(
            {"salary": salary_data}, vars=numeric_cols, method="spearman"
        )
        cr_kendall = correlation(
            {"salary": salary_data}, vars=numeric_cols, method="kendall"
        )

        # All should produce valid correlation matrices
        for cr in [cr_pearson, cr_spearman, cr_kendall]:
            assert np.all(np.abs(cr.cr) <= 1.0)
            assert np.all((cr.cp >= 0) & (cr.cp <= 1))

    def test_baseline_plot_correlation_matrix(self, salary_data, baseline_plot_dir):
        """Save baseline correlation matrix plot."""
        numeric_cols = ["salary", "yrs_since_phd", "yrs_service"]
        cr = correlation({"salary": salary_data}, vars=numeric_cols)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            g = cr.plot(nobs=200, figsize=(6, 6))

        out_file = baseline_plot_dir / "correlation_matrix.png"
        g.save(out_file, dpi=100, verbose=False)

        assert out_file.exists()
        assert out_file.stat().st_size > 1000

    def test_plot_returns_ggplot(self, salary_data):
        """Test that plot returns a plotnine ggplot in the pyrsm style."""
        from plotnine import ggplot

        from pyrsm.plot_style import theme_pyrsm

        numeric_cols = ["salary", "yrs_since_phd", "yrs_service"]
        cr = correlation({"salary": salary_data}, vars=numeric_cols)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig = cr.plot()

        assert isinstance(fig, ggplot)
        assert type(fig.theme) is type(theme_pyrsm())
        # one row and one column of panels per variable
        assert set(fig.data["row_var"].cat.categories) == set(numeric_cols)
        fig.draw()


def test_correlation_auto_vars():
    """Test correlation with auto-detected numeric variables."""
    df = pl.DataFrame({"x": [0, 1, 1, 1, 0, 0, 0], "y": [2, 1, 1, 1, 2, 2, 10]})
    c = correlation(df)

    assert np.isclose(c.cr[1, 0], -0.493, atol=1e-3)
    assert c.cr[0, 1] == c.cr[1, 0]
    assert c.cp[0, 1] == c.cp[1, 0]


def test_correlation_polars_pairwise_missing():
    """Test correlation with missing values."""
    df = pl.DataFrame({"x": [0, 1, 1, 1, 0, 0, 0], "y": [2, 1, 1, None, 2, 2, 10]})
    c = correlation(df)

    assert np.isclose(c.cr[1, 0], -0.4472, atol=1e-3)
    assert c.cp[1, 0] <= 1
    assert np.isfinite(c.cv).all()


def test_correlation_spearman_monotonic():
    """Test spearman correlation with monotonic data."""
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    c = correlation(df, method="spearman")

    assert np.isclose(c.cr[1, 0], 1.0)
    assert c.cp[1, 0] <= 0.05


def test_plot_handles_categorical_variables():
    """The correlation plot renders with mixed numeric + categorical variables
    (scatter / boxplot / spineplot panels), replicating radiant.basics."""
    from plotnine import ggplot

    df = pl.DataFrame(
        {
            "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "carat": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
            "clarity": ["SI", "VS", "SI", "VS", "IF", "SI", "VS", "IF"],
            "cut": ["Fair", "Good", "Good", "Fair", "Ideal", "Ideal", "Good", "Fair"],
        }
    )
    cr = correlation({"d": df}, vars=["price", "carat", "clarity", "cut"], method="polychoric")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = cr.plot(nobs=-1)
    assert isinstance(fig, ggplot)
    fig.draw()


# --- radiant.basics parity: hetcor with standard errors, cor2df -------------
# Reference values: R radiant.basics::correlation(titanic, c("pclass",
# "survived", "age", "fare"), hcor = TRUE, hcor_se = TRUE / FALSE), polycor 0.8.

import glob as _glob

import numpy as _np
import polars as _pl
import pytest as _pytest


def _titanic():
    path = _glob.glob("examples/data/**/titanic.parquet", recursive=True) or _glob.glob(
        "/home/vnijs/gh/pyrsm/examples/data/**/titanic.parquet", recursive=True
    )
    if not path:
        _pytest.skip("titanic example data not available")
    return _pl.read_parquet(path[0])


_VARS = ["pclass", "survived", "age", "fare"]


def test_hetcor_two_step_matches_r():
    from pyrsm.basics.correlation import correlation

    c = correlation({"titanic": _titanic()}, vars=_VARS, method="polychoric")
    r = {(0, 1): 0.44409, (0, 2): -0.47134, (0, 3): -0.65048,
         (1, 2): 0.07268, (1, 3): -0.31375, (2, 3): 0.17721}
    for (i, j), value in r.items():
        assert c.cr[i, j] == _pytest.approx(value, abs=5e-4)
    assert not c.hcor_se


def test_hetcor_with_standard_errors_matches_r():
    from pyrsm.basics.correlation import correlation

    c = correlation({"titanic": _titanic()}, vars=_VARS, method="polychoric", hcor_se=True)
    r = {(0, 1): 0.44407, (0, 2): -0.45427, (0, 3): -0.81752,
         (1, 2): 0.07247, (1, 3): -0.35770, (2, 3): 0.17721}
    p = {(0, 1): 1.1420e-31, (0, 2): 7.9409e-63, (1, 2): 6.3235e-02,
         (1, 3): 8.2303e-19, (2, 3): 3.5166e-09}
    for (i, j), value in r.items():
        assert c.cr[i, j] == _pytest.approx(value, abs=1e-4)
    for (i, j), value in p.items():
        assert c.cp[i, j] == _pytest.approx(value, rel=0.01)


def test_hetcor_summary_shows_p_values_only_with_standard_errors(capsys):
    from pyrsm.basics.correlation import correlation

    correlation({"titanic": _titanic()}, vars=_VARS, method="polychoric").summary(cov=True)
    out = capsys.readouterr().out
    assert "p.values:" not in out and "Not available for the polychoric" in out
    correlation({"titanic": _titanic()}, vars=_VARS, method="polychoric", hcor_se=True).summary(cov=True)
    out = capsys.readouterr().out
    assert "p.values:" in out and "Not available for the polychoric" in out


def test_cor2df_matches_r_layout():
    from pyrsm.basics.correlation import correlation

    df = correlation({"titanic": _titanic()}, vars=["age", "fare", "survived"][:2]).cor2df()
    assert df.columns == ["label1", "label2", "correlation", "distance"]
    assert df.row(0)[:2] == ("age", "fare")
    assert df["distance"][0] == _pytest.approx(0.5 * (1 - df["correlation"][0]))
