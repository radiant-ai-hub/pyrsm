import numpy as np
import pandas as pd
import polars as pl
import pytest

from pyrsm.model.perf import (
    ROME_plot,
    auc,
    calc_qnt,
    evalbin,
    gains_plot,
    gains_tab,
    inc_profit_plot,
    inc_profit_tab,
    inc_uplift_plot,
    lift_plot,
    lift_tab,
    profit_plot,
    uplift_plot,
    uplift_tab,
)


@pytest.fixture(scope="module")
def perf_test_data():
    """Create Polars test data."""
    np.random.seed(1234)
    nr = 100
    x1 = np.random.uniform(0, 1, nr)
    response_prob = np.random.uniform(0, 1, nr)
    df = pl.DataFrame(
        {
            "x1": x1,
            "response_prob": response_prob,
            "training": np.concatenate([np.ones(80), np.zeros(20)]),
            "rnd_response": np.random.choice(["yes", "no"], nr),
        }
    ).with_columns(
        (1 - pl.col("x1")).alias("x2"),
        pl.when((pl.col("x1") > 0.5) & (pl.col("response_prob") > 0.5))
        .then(pl.lit("yes"))
        .otherwise(pl.lit("no"))
        .alias("response"),
    )
    return df


def test_calc_qnt(perf_test_data):
    df = perf_test_data
    ret = calc_qnt(df, "response", "yes", "x1", qnt=10)
    assert ret["cum_resp"].to_list() == [
        6,
        14,
        20,
        27,
        29,
        30,
        30,
        30,
        30,
        30,
    ], "Incorrect calculation of cum_resp with x1"


def test_calc_qnt_rev(perf_test_data):
    df = perf_test_data
    ret = calc_qnt(df, "response", "yes", "x2", qnt=10)
    assert ret["cum_resp"].to_list() == [
        6,
        14,
        20,
        27,
        29,
        30,
        30,
        30,
        30,
        30,
    ], "Incorrect calculation of cum_resp with x2"


def test_gains_tab(perf_test_data):
    df = perf_test_data
    ret = gains_tab(df, "response", "yes", "x1", qnt=10)
    expected = [0.0, 0.2, 0.467, 0.667, 0.9, 0.967, 1.0, 1.0, 1.0, 1.0, 1.0]
    actual = [round(v, 3) for v in ret["cum_gains"].to_list()]
    assert actual == expected, "Incorrect calculation of cum_gains"


def test_lift_tab(perf_test_data):
    df = perf_test_data
    ret = lift_tab(df, "response", "yes", "x1", qnt=10)
    expected = [2.0, 2.333, 2.222, 2.25, 1.933, 1.667, 1.429, 1.25, 1.111, 1.0]
    actual = [round(v, 3) for v in ret["cum_lift"].to_list()]
    assert actual == expected, "Incorrect calculation of cum_lift"


def test_evalbin(perf_test_data):
    df = perf_test_data
    dct = {
        "train": df.filter(pl.col("training") == 1),
        "test": df.filter(pl.col("training") == 0),
    }
    ret = evalbin(dct, "response", "yes", "x1", cost=1, margin=2, dec=3)

    assert ret.shape == (2, 18), "Incorrect dimensions"
    row0 = ret.row(0, named=True)
    assert [
        row0["profit"],
        row0["index"],
        row0["ROME"],
        row0["contact"],
        row0["AUC"],
    ] == [5, 1.0, 0.116, 0.538, 0.897], "Errors for profit:AUC in row 0"
    row1 = ret.row(1, named=True)
    assert [
        row1["profit"],
        row1["index"],
        row1["ROME"],
        row1["contact"],
        row1["AUC"],
    ] == [-2, 1.0, -0.143, 0.7, 0.845], "Errors for profit:AUC in row 1"


def test_lift_plot_single(perf_test_data):
    df = perf_test_data
    lift_plot(df, "response", "yes", "x1", qnt=10)


def test_lift_plot_mult(perf_test_data):
    df = perf_test_data
    lift_plot(df, "response", "yes", ["x1", "x2"], qnt=10)


def test_gains_plot_single(perf_test_data):
    df = perf_test_data
    gains_plot(df, "response", "yes", "x1", qnt=10)


def test_gains_plot_mult(perf_test_data):
    df = perf_test_data
    gains_plot(df, "response", "yes", ["x1", "x2"], qnt=10)


def test_profit_plot_single(perf_test_data):
    df = perf_test_data
    profit_plot(df, "response", "yes", "x1", qnt=10)


def test_profit_plot_mult(perf_test_data):
    df = perf_test_data
    profit_plot(df, "response", "yes", ["x1", "x2"], qnt=10, contact=True)


def test_rome_plot_single(perf_test_data):
    df = perf_test_data
    ROME_plot(df, "response", "yes", "x1", qnt=10)


def test_ROME_plot_mult(perf_test_data):
    df = perf_test_data
    ROME_plot(df, "response", "yes", ["x1", "x2"], qnt=10)


# ---- AUC tests with different input types ----


@pytest.fixture(scope="module")
def auc_test_data():
    """Create test data for AUC in numpy, pandas, and polars formats."""
    np.random.seed(42)
    n = 100
    pred = np.random.uniform(0, 1, n)
    rvar = np.where(pred + np.random.normal(0, 0.3, n) > 0.5, "yes", "no")
    return pred, rvar


def test_auc_numpy(auc_test_data):
    """Test auc with numpy arrays."""
    pred, rvar = auc_test_data
    result = auc(rvar, pred, lev="yes")
    assert isinstance(result, float)
    assert 0.5 <= result <= 1.0


def test_auc_pandas(auc_test_data):
    """Test auc with pandas Series."""
    pred, rvar = auc_test_data
    pred_pd = pd.Series(pred)
    rvar_pd = pd.Series(rvar)
    result = auc(rvar_pd, pred_pd, lev="yes")
    assert isinstance(result, float)
    assert 0.5 <= result <= 1.0


def test_auc_polars(auc_test_data):
    """Test auc with polars Series."""
    pred, rvar = auc_test_data
    pred_pl = pl.Series("pred", pred)
    rvar_pl = pl.Series("rvar", rvar)
    result = auc(rvar_pl, pred_pl, lev="yes")
    assert isinstance(result, float)
    assert 0.5 <= result <= 1.0


def test_auc_all_types_match(auc_test_data):
    """Test that auc produces identical results for numpy, pandas, and polars inputs."""
    pred, rvar = auc_test_data

    # Numpy
    result_np = auc(rvar, pred, lev="yes")

    # Pandas
    result_pd = auc(pd.Series(rvar), pd.Series(pred), lev="yes")

    # Polars
    result_pl = auc(pl.Series("rvar", rvar), pl.Series("pred", pred), lev="yes")

    assert np.isclose(
        result_np, result_pd
    ), f"numpy ({result_np}) != pandas ({result_pd})"
    assert np.isclose(
        result_np, result_pl
    ), f"numpy ({result_np}) != polars ({result_pl})"
    assert np.isclose(
        result_pd, result_pl
    ), f"pandas ({result_pd}) != polars ({result_pl})"


def test_auc_with_ties(auc_test_data):
    """Test auc handles ties correctly across all input types."""
    # Data with intentional ties
    pred = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.5])
    rvar = np.array(["yes", "no", "yes", "no", "yes", "no", "yes", "no"])

    result_np = auc(rvar, pred, lev="yes")
    result_pd = auc(pd.Series(rvar), pd.Series(pred), lev="yes")
    result_pl = auc(pl.Series("rvar", rvar), pl.Series("pred", pred), lev="yes")

    assert np.isclose(result_np, result_pd)
    assert np.isclose(result_np, result_pl)


# ---- Uplift tests ----


@pytest.fixture(scope="module")
def uplift_test_data():
    """Create deterministic test data for uplift calculations.

    Constructs a dataset where the treatment group has a higher response
    rate for high-prediction observations, allowing us to verify uplift
    calculations against hand-computed expected values.
    """
    np.random.seed(2024)

    # Treatment assignment (50/50 split)
    treatment = np.array(["treatment"] * 100 + ["control"] * 100)

    # Prediction scores - higher scores should correlate with uplift
    pred_treat = np.random.uniform(0.3, 1.0, 100)
    pred_control = np.random.uniform(0.0, 0.7, 100)
    pred = np.concatenate([pred_treat, pred_control])

    # Response: treatment group has higher response for high-pred individuals
    resp_treat = np.where(
        pred_treat > 0.6, "yes", np.where(np.random.random(100) > 0.7, "yes", "no")
    )
    resp_control = np.where(
        pred_control > 0.5, np.where(np.random.random(100) > 0.5, "yes", "no"), "no"
    )
    response = np.concatenate([resp_treat, resp_control])

    df = pl.DataFrame(
        {
            "pred_score": pred,
            "response": response,
            "treatment": treatment,
        }
    )
    return df


@pytest.fixture(scope="module")
def uplift_simple_data():
    """Small, fully deterministic dataset for exact uplift calculations."""
    df = pl.DataFrame(
        {
            "pred_score": [
                0.9,
                0.8,
                0.7,
                0.6,
                0.5,
                0.4,
                0.3,
                0.2,
                0.1,
                0.05,
                0.85,
                0.75,
                0.65,
                0.55,
                0.45,
                0.35,
                0.25,
                0.15,
                0.08,
                0.02,
            ],
            "response": [
                "yes",
                "yes",
                "yes",
                "no",
                "yes",
                "no",
                "no",
                "no",
                "no",
                "no",
                "yes",
                "no",
                "no",
                "no",
                "no",
                "no",
                "no",
                "no",
                "no",
                "no",
            ],
            "treatment": ["treatment"] * 10 + ["control"] * 10,
        }
    )
    return df


class TestUpliftTab:
    """Tests for uplift_tab function."""

    def test_uplift_tab_returns_polars_df(self, uplift_test_data):
        """Test that uplift_tab returns a Polars DataFrame."""
        result = uplift_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )
        assert isinstance(result, pl.DataFrame)

    def test_uplift_tab_expected_columns(self, uplift_test_data):
        """Test that output has all expected columns."""
        result = uplift_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )
        expected_cols = [
            "pred",
            "bins",
            "cum_prop",
            "T_resp",
            "T_n",
            "C_resp",
            "C_n",
            "incremental_resp",
            "inc_uplift",
            "uplift",
        ]
        assert result.columns == expected_cols

    def test_uplift_tab_num_rows_matches_qnt(self, uplift_test_data):
        """Test that the number of rows equals the number of quantiles."""
        for qnt in [5, 10, 20]:
            result = uplift_tab(
                uplift_test_data,
                "response",
                "yes",
                "pred_score",
                "treatment",
                "treatment",
                qnt=qnt,
            )
            assert result.height == qnt, f"Expected {qnt} rows, got {result.height}"

    def test_uplift_tab_cum_prop_reaches_one(self, uplift_test_data):
        """Test that cum_prop goes from 0.1 to 1.0 in tenths for qnt=10."""
        result = uplift_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )
        assert np.isclose(result["cum_prop"][-1], 1.0)
        assert result["cum_prop"][0] > 0

    def test_uplift_tab_cumulative_T_n_nondecreasing(self, uplift_test_data):
        """Test that cumulative treatment counts are non-decreasing."""
        result = uplift_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )
        t_n = result["T_n"].to_list()
        assert all(t_n[i] <= t_n[i + 1] for i in range(len(t_n) - 1))

    def test_uplift_tab_cumulative_C_n_nondecreasing(self, uplift_test_data):
        """Test that cumulative control counts are non-decreasing."""
        result = uplift_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )
        c_n = result["C_n"].to_list()
        assert all(c_n[i] <= c_n[i + 1] for i in range(len(c_n) - 1))

    def test_uplift_tab_final_T_n_equals_treatment_count(self, uplift_test_data):
        """Test that total T_n equals actual number of treatment observations."""
        result = uplift_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )
        expected_t_n = (uplift_test_data["treatment"] == "treatment").sum()
        assert result["T_n"][-1] == expected_t_n

    def test_uplift_tab_final_C_n_equals_control_count(self, uplift_test_data):
        """Test that total C_n equals actual number of control observations."""
        result = uplift_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )
        expected_c_n = (uplift_test_data["treatment"] != "treatment").sum()
        assert result["C_n"][-1] == expected_c_n

    def test_uplift_tab_final_T_resp_equals_treatment_responders(
        self, uplift_test_data
    ):
        """Test that total T_resp matches actual treatment responders."""
        result = uplift_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )
        expected = (
            (uplift_test_data["treatment"] == "treatment")
            & (uplift_test_data["response"] == "yes")
        ).sum()
        assert result["T_resp"][-1] == expected

    def test_uplift_tab_final_C_resp_equals_control_responders(self, uplift_test_data):
        """Test that total C_resp matches actual control responders."""
        result = uplift_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )
        expected = (
            (uplift_test_data["treatment"] != "treatment")
            & (uplift_test_data["response"] == "yes")
        ).sum()
        assert result["C_resp"][-1] == expected

    def test_uplift_tab_incremental_resp_formula(self, uplift_test_data):
        """Test that incremental_resp = T_resp - C_resp * T_n / C_n."""
        result = uplift_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )
        for i in range(result.height):
            row = result.row(i, named=True)
            if row["C_n"] == 0:
                # When there are no control obs, incremental_resp is NaN
                assert np.isnan(row["incremental_resp"])
                continue
            expected_inc = row["T_resp"] - row["C_resp"] * row["T_n"] / row["C_n"]
            assert np.isclose(row["incremental_resp"], expected_inc, atol=1e-10), (
                f"Row {i}: incremental_resp={row['incremental_resp']}, "
                f"expected={expected_inc}"
            )

    def test_uplift_tab_inc_uplift_formula(self, uplift_test_data):
        """Test that inc_uplift = incremental_resp / T_n_max * 100."""
        result = uplift_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )
        t_n_max = result["T_n"].max()
        for i in range(result.height):
            row = result.row(i, named=True)
            if np.isnan(row["incremental_resp"]):
                assert np.isnan(row["inc_uplift"])
                continue
            expected_inc_uplift = row["incremental_resp"] / t_n_max * 100
            assert np.isclose(row["inc_uplift"], expected_inc_uplift, atol=1e-10)

    def test_uplift_tab_pred_column(self, uplift_test_data):
        """Test that the pred column contains the predictor name."""
        result = uplift_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )
        assert all(v == "pred_score" for v in result["pred"].to_list())

    def test_uplift_tab_scale_factor(self, uplift_test_data):
        """Test that scale parameter multiplies cumulative counts correctly."""
        result_no_scale = uplift_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            scale=1,
            qnt=10,
        )
        result_scaled = uplift_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            scale=2,
            qnt=10,
        )
        # Cumulative T_resp, T_n, C_resp, C_n should be scaled by 2
        for col in ["T_resp", "T_n", "C_resp", "C_n"]:
            for i in range(result_no_scale.height):
                assert np.isclose(
                    result_scaled[col][i],
                    result_no_scale[col][i] * 2,
                    atol=1e-10,
                ), f"Scale mismatch in {col}, row {i}"

    def test_uplift_tab_pandas_input(self, uplift_test_data):
        """Test that uplift_tab works with pandas input."""
        df_pd = uplift_test_data.to_pandas()
        result = uplift_tab(
            df_pd,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )
        assert isinstance(result, pl.DataFrame)
        assert result.height == 10

    def test_uplift_tab_bins_sorted(self, uplift_test_data):
        """Test that bins are sorted in ascending order."""
        result = uplift_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )
        bins = result["bins"].to_list()
        assert bins == sorted(bins)

    def test_uplift_tab_uplift_per_bin_calculation(self, uplift_simple_data):
        """Test per-bin uplift = T_resp_rate - C_resp_rate using simple data."""
        result = uplift_tab(
            uplift_simple_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=5,
        )
        # uplift column should exist and contain numeric values
        assert "uplift" in result.columns
        uplift_vals = result["uplift"].to_list()
        assert all(isinstance(v, (int, float)) for v in uplift_vals)

    def test_uplift_tab_qnt5_rows(self, uplift_test_data):
        """Test with 5 quantiles."""
        result = uplift_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=5,
        )
        assert result.height == 5
        assert np.isclose(result["cum_prop"][-1], 1.0)

    def test_uplift_tab_consistent_results(self, uplift_test_data):
        """Test that calling uplift_tab twice gives the same result."""
        r1 = uplift_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )
        r2 = uplift_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )
        assert r1.equals(r2)


class TestUpliftPlots:
    """Tests for uplift-related plot functions (smoke tests)."""

    def test_uplift_plot_runs(self, uplift_test_data):
        """Test that uplift_plot runs without error."""
        uplift_plot(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )

    def test_inc_uplift_plot_runs(self, uplift_test_data):
        """Test that inc_uplift_plot runs without error."""
        inc_uplift_plot(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )

    def test_inc_profit_tab_runs(self, uplift_test_data):
        """Test that inc_profit_tab returns a DataFrame."""
        result = inc_profit_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            cost=1,
            margin=5,
            qnt=10,
        )
        assert isinstance(result, pl.DataFrame)
        assert "inc_profit" in result.columns

    def test_inc_profit_plot_runs(self, uplift_test_data):
        """Test that inc_profit_plot runs without error."""
        inc_profit_plot(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            cost=1,
            margin=5,
            qnt=10,
        )

    def test_uplift_plot_with_dict_input(self, uplift_test_data):
        """Test uplift_plot with a dict of DataFrames."""
        dct = {"model_a": uplift_test_data}
        uplift_plot(
            dct,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            qnt=10,
        )

    def test_inc_uplift_plot_multiple_preds(self, uplift_test_data):
        """Test inc_uplift_plot with multiple predictors."""
        df = uplift_test_data.with_columns((1 - pl.col("pred_score")).alias("pred2"))
        inc_uplift_plot(
            df,
            "response",
            "yes",
            ["pred_score", "pred2"],
            "treatment",
            "treatment",
            qnt=10,
        )


class TestIncProfitTab:
    """Tests for inc_profit_tab calculations."""

    def test_inc_profit_formula(self, uplift_test_data):
        """Test that inc_profit = incremental_resp * margin - T_n * cost."""
        cost, margin = 1, 5
        result = inc_profit_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            cost=cost,
            margin=margin,
            qnt=10,
        )
        for i in range(result.height):
            row = result.row(i, named=True)
            expected = row["incremental_resp"] * margin - row["T_n"] * cost
            if np.isnan(expected):
                assert np.isnan(row["inc_profit"])
                continue
            assert np.isclose(
                row["inc_profit"], expected, atol=1e-10
            ), f"Row {i}: inc_profit={row['inc_profit']}, expected={expected}"

    def test_inc_profit_starts_at_zero(self, uplift_test_data):
        """Test that inc_profit_tab includes a zero-origin row."""
        result = inc_profit_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            cost=1,
            margin=5,
            qnt=10,
        )
        # First row should have cum_prop=0.0 (zero origin)
        assert result["cum_prop"][0] == 0.0
        assert result["incremental_resp"][0] == 0.0

    def test_inc_profit_different_cost_margin(self, uplift_test_data):
        """Test that changing cost/margin changes the profit values."""
        r1 = inc_profit_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            cost=1,
            margin=2,
            qnt=10,
        )
        r2 = inc_profit_tab(
            uplift_test_data,
            "response",
            "yes",
            "pred_score",
            "treatment",
            "treatment",
            cost=1,
            margin=10,
            qnt=10,
        )
        # Higher margin should generally yield higher profit
        assert not r1["inc_profit"].equals(r2["inc_profit"])
