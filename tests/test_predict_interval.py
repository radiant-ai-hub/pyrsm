"""Prediction and confidence intervals, as radiant.model 1.6.13.

Reference numbers: R 4.x, radiant.model 1.6.12 on the same data, e.g.

    r <- regress(diamonds, "price", c("carat", "clarity"))
    predict(r$model, nd, interval = "prediction", level = 0.9)
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import pyrsm as rsm

DATA = Path(__file__).parents[1] / "examples" / "data" / "data"


@pytest.fixture(scope="module")
def diamonds():
    return pl.read_parquet(DATA / "diamonds.parquet")


@pytest.fixture(scope="module")
def titanic():
    return pl.read_parquet(DATA / "titanic.parquet")


def _new(diamonds):
    return pl.DataFrame({"carat": [0.5, 1.0, 1.5], "clarity": ["SI1"] * 3})


def _bounds(pred):
    return pred.select(pred.columns[-3:]).to_numpy()


def test_confidence_interval_matches_r(diamonds):
    reg = rsm.model.regress({"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"])
    got = _bounds(reg.predict(_new(diamonds), ci=True))
    ref = [
        [1046.553723, 950.3987247, 1142.708722],
        [5265.568878, 5174.7758080, 5356.361949],
        [9484.584034, 9373.8562211, 9595.311846],
    ]
    np.testing.assert_allclose(got, ref, rtol=1e-8)


def test_prediction_interval_matches_r(diamonds):
    reg = rsm.model.regress({"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"])
    pred = reg.predict(_new(diamonds), ci=True, conf=0.9, interval="prediction")
    assert pred.columns[-3:] == ["prediction", "5.00%", "95.00%"]
    ref = [
        [1046.553723, -972.5542681, 3065.661715],
        [5265.568878, 3246.6356892, 7284.502068],
        [9484.584034, 7464.9503909, 11504.217676],
    ]
    np.testing.assert_allclose(_bounds(pred), ref, rtol=1e-8)


def test_prediction_interval_is_wider_and_works_with_data_and_command(diamonds):
    reg = rsm.model.regress({"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"])
    ci = _bounds(reg.predict(diamonds.head(5), data_cmd={"carat": 1.0}, ci=True))
    pi = _bounds(reg.predict(diamonds.head(5), data_cmd={"carat": 1.0}, ci=True, interval="prediction"))
    np.testing.assert_allclose(ci[:, 0], pi[:, 0])
    assert (pi[:, 1] < ci[:, 1]).all() and (pi[:, 2] > ci[:, 2]).all()


@pytest.mark.parametrize("standardize", [1, 2])
def test_standardized_model_gives_the_same_intervals(diamonds, standardize):
    """Standardizing is linear: the same model, so the same intervals in dollars."""
    plain = rsm.model.regress({"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"])
    std = rsm.model.regress(
        {"diamonds": diamonds}, rvar="price", evar=["carat", "clarity"], standardize=standardize
    )
    for interval in ("confidence", "prediction"):
        a = _bounds(plain.predict(_new(diamonds), ci=True, interval=interval))
        b = _bounds(std.predict(_new(diamonds), ci=True, interval=interval))
        np.testing.assert_allclose(a, b, rtol=1e-8)


def test_unknown_interval_is_refused(diamonds):
    reg = rsm.model.regress({"diamonds": diamonds}, rvar="price", evar=["carat"])
    with pytest.raises(ValueError, match="interval"):
        reg.predict(ci=True, interval="credible")


def test_logistic_confidence_interval_matches_r(titanic):
    clf = rsm.model.logistic(
        {"titanic": titanic}, rvar="survived", lev="Yes", evar=["pclass", "sex", "age"]
    )
    pred = clf.predict(cmd={"pclass": ["1st", "2nd", "3rd"], "sex": "female", "age": [30]}, ci=True)
    ref = [
        [0.9227340279, 0.8886672447, 0.9469984054],
        [0.7695387422, 0.7006057690, 0.8265301975],
        [0.5491636251, 0.4752243767, 0.6209935951],
    ]
    # R's glm and statsmodels converge to slightly different points (~1e-6)
    np.testing.assert_allclose(_bounds(pred), ref, rtol=1e-5)


def test_logistic_has_no_prediction_interval(titanic):
    clf = rsm.model.logistic({"titanic": titanic}, rvar="survived", lev="Yes", evar=["sex"])
    with pytest.raises(ValueError, match="not available for logistic"):
        clf.predict(ci=True, interval="prediction")


@pytest.mark.parametrize("sd, age", [(1, -0.4954640453), (2, -0.9909280906)])
def test_logistic_standardizes_by_one_or_two_sd_as_r(titanic, sd, age):
    clf = rsm.model.logistic(
        {"titanic": titanic}, rvar="survived", lev="Yes", evar=["pclass", "sex", "age"],
        standardize=sd,
    )
    coef = dict(zip(clf.coef["index"], clf.coef["coefficient"]))
    assert coef["age"] == pytest.approx(age, rel=1e-6)
    assert coef["Intercept"] == pytest.approx(2.4865297733, rel=1e-6)
