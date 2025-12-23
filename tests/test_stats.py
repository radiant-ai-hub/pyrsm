import numpy as np
import polars as pl

from pyrsm.stats import (
    scale_df,
    seprop,
    varprop,
    weighted_mean,
    weighted_sd,
)


def test_varprop():
    assert varprop([1, 1, 1, 0, 0, 0]) == 0.25, "Proportion standard error incorrect"


def test_seprop():
    assert (
        seprop([1, 1, 1, 0, 0, 0]) == 0.2041241452319315
    ), "Proportion standard error incorrect"


# create example df and wt vector for testing
df = pl.DataFrame(
    {
        "x": [0, 1, 1, 1, 0, 0, 0],
        "y": [2, 1, 1, 1, 6, 2, 10],
        "z": [2, 1, 1, 1, 2, 2, 10],
    }
)
wt = [1, 10, 1, 10, 1, 10, 1]


def test_weighted_mean():
    result = weighted_mean(df, wt).round(5)
    expected = pl.Series([0.61765, 1.73529, 1.61765])
    assert result.equals(expected), "Weighted means incorrect"


def test_weighted_sd():
    result = weighted_sd(df, wt).round(5)
    expected = pl.Series([0.48596, 1.70309, 1.53421])
    assert result.equals(expected), "Weighted standard deviations incorrect"


def test_scale_df():
    scaled = scale_df(df, sf=2, ddof=1)
    row0 = scaled.row(0, named=True)
    row1 = scaled.row(1, named=True)
    assert np.allclose(
        [row0["x"], row0["y"], row0["z"]], [0, -0.18632, -0.10984], atol=1e-5
    ), "Scaled dataframe row 0 incorrect"
    assert np.allclose(
        [row1["x"], row1["y"], row1["z"]], [1, -0.33123, -0.26362], atol=1e-5
    ), "Scaled dataframe row 1 incorrect"


def test_weighted_scale_df():
    scaled = scale_df(df, wt, sf=2, ddof=0)
    row0 = scaled.row(0, named=True)
    row1 = scaled.row(1, named=True)
    assert np.allclose(
        [row0["x"], row0["y"], row0["z"]], [0, 0.07771, 0.12461], atol=1e-5
    ), "Weighted scaled dataframe row 0 incorrect"
    assert np.allclose(
        [row1["x"], row1["y"], row1["z"]], [1, -0.21587, -0.20129], atol=1e-5
    ), "Weighted scaled dataframe row 1 incorrect"


def test_correlation():
    from pyrsm.basics.correlation import correlation

    cr = correlation(df)
    assert round(cr.cr[2, 0], 3) == -0.493, "Correlations incorrect"
    df_nan = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 4)
        .then(None)
        .otherwise(pl.col("x"))
        .alias("x")
    )
    cr_nan = correlation(df_nan)
    assert round(cr_nan.cr[2, 0], 3) == -0.567, "Correlations with np.nan incorrect"


if __name__ == "__main__":
    test_varprop()
    test_seprop()
    test_weighted_mean()
    test_weighted_sd()
    test_scale_df()
    test_weighted_scale_df()
    test_correlation()
