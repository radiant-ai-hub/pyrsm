import warnings

import numpy as np
import polars as pl
import pytest

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


def test_correlation_drops_rows_with_missing_values_as_radiant_basics():
    """Listwise deletion, as radiant.basics (get_data(na.rm = TRUE)): a row
    missing c is dropped for a-b too. Values from radiant.basics::correlation."""
    import io
    from contextlib import redirect_stdout

    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8],
            "b": [2, 1, 4, 3, 6, 5, 8, 9],
            "c": [None, 3, 1, None, 2, 5, 4, 8],
        }
    )
    c = correlation(df)
    assert c.nobs == 6 and c.nobs_dropped == 2
    assert np.allclose(c.cr[[1, 2, 2], [0, 0, 1]], [0.9439504386, 0.7706422917, 0.6010411026])
    assert np.allclose(c.cp[[1, 2, 2], [0, 0, 1]], [0.004624288651, 0.072874761204, 0.20700151756])
    assert np.allclose(c.cv, [[5.366666667, 6.3, 4.433333333], [6.3, 8.3, 4.3], [4.433333333, 4.3, 6.166666667]])

    out = io.StringIO()
    with redirect_stdout(out):
        c.summary()
    assert "Dropped  : 2 of 8 rows (missing values)" in out.getvalue()

    out = io.StringIO()
    with redirect_stdout(out):
        correlation(df.drop_nulls()).summary()
    assert "Dropped" not in out.getvalue()


def test_correlation_drops_nan_as_missing():
    """R's na.omit drops NaN too."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [2.0, float("nan"), 3.0, 5.0, 4.0], "c": [1.0, 2.0, 2.0, 3.0, 1.0]})
    c = correlation(df)
    assert c.nobs_dropped == 1 and np.isfinite(c.cr).all()


def test_correlation_refuses_variables_without_variation():
    """As radiant.basics: a variable that is constant once incomplete rows are
    dropped is named, not reported as NaN correlations."""
    df = pl.DataFrame({"spend": [None, None, 3.0, 5.0, 4.0], "buyer": [0, 0, 1, 1, 1], "age": [30, 40, 35, 50, 45]})
    with pytest.raises(ValueError, match="no variation after dropping rows with missing values: buyer"):
        correlation(df)
    with pytest.raises(ValueError, match=r"no variation: k\."):
        correlation(pl.DataFrame({"k": [1, 1, 1], "x": [1, 2, 3]}))


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


def test_hetcor_with_standard_errors_keeps_the_estimates():
    """Asking for p.values must not change a correlation.

    polycor::hetcor(std.err = TRUE) re-optimizes numeric x categorical cells
    (R: pclass x fare -0.650 without, -0.818 with); pyrsm keeps the two-step
    estimates and evaluates the standard errors there. Reference SEs: R,
    optimHess() of polycor's polyserial likelihood at the two-step rho;
    polychor(std.err = TRUE) for pclass x survived; hetcor's Pearson SE for
    age x fare.
    """
    from scipy.stats import norm

    from pyrsm.basics.correlation import correlation

    plain = correlation({"titanic": _titanic()}, vars=_VARS, method="polychoric")
    c = correlation({"titanic": _titanic()}, vars=_VARS, method="polychoric", hcor_se=True)
    assert _np.array_equal(c.cr, plain.cr)
    r = {(0, 1): 0.44407, (0, 2): -0.47134, (0, 3): -0.65048,
         (1, 2): 0.07268, (1, 3): -0.31375, (2, 3): 0.17721}
    se = {(0, 1): 0.037925, (0, 2): 0.026014, (0, 3): 0.053518,
          (1, 2): 0.039013, (1, 3): 0.042383, (2, 3): 0.030007}
    for (i, j), value in r.items():
        assert c.cr[i, j] == _pytest.approx(value, abs=1e-4)
        assert c.se[i, j] == _pytest.approx(se[(i, j)], rel=0.01)
        assert c.cp[i, j] == _pytest.approx(2 * norm.sf(abs(value / se[(i, j)])), rel=0.05)


def test_polyserial_matches_r_on_salary_with_and_without_standard_errors(salary_data):
    """R: polycor::hetcor(salary[, c("salary", "yrs_since_phd", "yrs_service",
    "rank")]) gives rank 0.7705, 0.8750, 0.7594 (two-step polyserial); SEs from
    optimHess() of polyserial's likelihood at those values. pyrsm's
    polyserial once applied sqrt((n - 1) / n) twice (0.7695)."""
    from pyrsm.basics.correlation import correlation

    v = ["salary", "yrs_since_phd", "yrs_service", "rank"]
    plain = correlation({"salary": salary_data}, vars=v, method="polychoric")
    c = correlation({"salary": salary_data}, vars=v, method="polychoric", hcor_se=True)
    assert _np.array_equal(c.cr, plain.cr)
    for j, (rho, se) in enumerate([(0.770517, 0.060303), (0.875036, 0.012779), (0.759445, 0.030841)]):
        assert c.cr[3, j] == _pytest.approx(rho, abs=1e-5)
        assert c.se[3, j] == _pytest.approx(se, rel=1e-3)
    assert c.cr[1, 0] == _pytest.approx(0.4192311, abs=1e-6)  # Pearson, as hetcor


def test_hetcor_summary_says_which_pairs_used_which_method(salary_data, capsys):
    from pyrsm.basics.correlation import correlation

    v = ["salary", "yrs_since_phd", "rank"]
    correlation({"salary": salary_data}, vars=v, method="polychoric").summary()
    out = capsys.readouterr().out
    assert "Method   : polychoric (heterogeneous; categorical variables treated as ordinal)" in out
    assert "Pairs    : Pearson (numeric-numeric), polyserial (numeric-categorical)\n" in out
    correlation({"titanic": _titanic()}, vars=_VARS, method="polychoric").summary()
    assert "polychoric (categorical-categorical)" in capsys.readouterr().out
    correlation({"salary": salary_data}, vars=v[:2], method="spearman").summary()
    out = capsys.readouterr().out
    assert "Method   : spearman\n" in out and "Pairs" not in out


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
