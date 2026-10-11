"""rsm.model.tune: cross-validated hyper-parameter search for mlp, rforest and xgboost."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import pyrsm as rsm
from pyrsm.model.tune import TuneError, _dims, _grid, _narrow, tune

DATA = Path(__file__).resolve().parents[1] / "examples" / "data" / "model"


@pytest.fixture(scope="module")
def titanic():
    return pl.read_parquet(DATA / "titanic.parquet")


@pytest.fixture(scope="module")
def rf(titanic):
    return rsm.model.rforest({"titanic": titanic}, rvar="survived", lev="Yes", evar=["pclass", "sex", "age"], n_estimators=50)


SPACE = {"max_features": [1, 2, 3], "min_samples_leaf": {"min": 1, "max": 30, "n": 4}}


def test_grid_scores_every_combination(rf):
    res = tune(rf, SPACE, method="grid", folds=3, n_jobs=2)
    assert len(res.results) == 12 and res.n_fits == 36
    assert res.results["rank"].tolist() == list(range(1, 13))
    assert res.results["mean"].is_monotonic_decreasing
    assert res.best_params == {k: res.results.loc[0, k] for k in ("max_features", "min_samples_leaf")}
    assert res.metric == "auc" and 0.7 < res.best_score < 1


@pytest.mark.parametrize("method", ["random", "halving", "refine"])
def test_other_methods_stay_within_the_budget(rf, method):
    res = tune(rf, SPACE, method=method, n_iter=12, folds=3, max_fits=60, n_jobs=2)
    assert res.n_fits <= 60
    assert set(res.best_params) == {"max_features", "min_samples_leaf"}
    if method == "halving":
        assert res.results["rows"].nunique() > 1
    if method == "refine":
        assert {"coarse", "fine"} <= set(res.results["stage"])


def test_simplest_within_one_se_is_not_more_complex(rf):
    res = tune(rf, SPACE, method="grid", folds=3, n_jobs=2)
    assert res.simple_params["max_features"] <= res.best_params["max_features"] or (
        res.simple_params["min_samples_leaf"] >= res.best_params["min_samples_leaf"]
    )
    se = res.best_std / 3**0.5
    assert res.simple_score >= res.best_score - se - 1e-12


def test_progress_and_stop(rf):
    seen = []
    res = tune(rf, {"min_samples_leaf": [1, 5, 10, 20]}, method="grid", folds=2, n_jobs=1,
               progress=lambda d, t: seen.append((d, t)), should_stop=lambda: len(seen) >= 4)  # fmt: skip
    assert res.stopped and seen[0][1] == 8
    assert len(res.results) <= 2


def test_regression_and_mlp_target_scale():
    dia = pl.read_parquet(DATA / "diamonds.parquet")
    nn = rsm.model.mlp({"diamonds": dia}, rvar="price", evar=["carat", "clarity"], mod_type="regression")
    res = tune(nn, {"hidden_layer_sizes": ["2", "3"]}, method="grid", folds=3, n_jobs=2)
    # RMSE in dollars, not in standard deviations of price
    assert res.metric == "rmse" and 300 < res.best_score < 3000
    assert res.results["mean"].is_monotonic_increasing
    xg = rsm.model.xgboost({"diamonds": dia}, rvar="price", evar=["carat"], mod_type="regression", n_estimators=20)
    res = tune(xg, {"max_depth": [2, 4]}, method="grid", folds=3, metric="r2", n_jobs=2)
    assert res.metric == "r2" and 0.5 < res.best_score <= 1


def test_mlp_standardizes_inside_each_fold(tmp_path):
    """Each fold is scaled by its own training rows, like a scaler inside a pipeline."""
    import numpy as np
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from pyrsm.model.tune import _fit_score, _model_data

    def pipe(net, col):
        return make_pipeline(ColumnTransformer([("num", StandardScaler(), [col])], remainder="passthrough"), net)

    dia = pl.read_parquet(DATA / "diamonds.parquet")
    nn = rsm.model.mlp({"diamonds": dia}, rvar="price", evar=["carat", "clarity"], mod_type="regression", hidden_layer_sizes=(2,))
    data = _model_data(nn)
    # raw values go in; nothing was standardized with the held-out rows
    assert np.allclose(data.X["carat"].to_numpy(), nn.data["carat"].to_numpy())
    assert data.estimator.scale_cols == ["carat"] and data.estimator.scale_y
    assert clone(data.estimator).get_params()["alpha"] == nn.fitted.alpha
    train, test = next(KFold(n_splits=3, shuffle=True, random_state=1234).split(data.X, data.y))
    ours = _fit_score(data.estimator, {}, "rmse", data.X, data.y, train, test)
    net = MLPRegressor(**nn.fitted.get_params())
    ref = TransformedTargetRegressor(pipe(net, "carat"), transformer=StandardScaler()).fit(data.X.iloc[train], data.y[train])
    expected = np.sqrt(np.mean((ref.predict(data.X.iloc[test]) - data.y[test]) ** 2))
    assert ours == pytest.approx(expected, rel=1e-6)

    tit = pl.read_parquet(DATA / "titanic.parquet")
    clf = rsm.model.mlp({"titanic": tit}, rvar="survived", lev="Yes", evar=["age", "sex"], hidden_layer_sizes=(2,))
    data = _model_data(clf)
    assert data.estimator.scale_cols == ["age"] and not data.estimator.scale_y
    train, test = next(StratifiedKFold(n_splits=3, shuffle=True, random_state=1234).split(data.X, data.y))
    ours = _fit_score(data.estimator, {}, "auc", data.X, data.y, train, test)
    ref = pipe(MLPClassifier(**clf.fitted.get_params()), "age").fit(data.X.iloc[train], data.y[train])
    assert ours == pytest.approx(roc_auc_score(data.y[test], ref.predict_proba(data.X.iloc[test])[:, -1]), rel=1e-6)

    # cross_validation uses the same estimator: RMSE in dollars, and the best
    # settings go straight back into mlp
    cv = rsm.model.cross_validation(
        nn, "nn", {"alpha": [0.0001, 0.01]}, {"rmse": "neg_root_mean_squared_error"},
        prn=False, directory=str(tmp_path), n_splits=3, n_jobs=1, verbose=0,
    )  # fmt: skip
    assert 300 < -cv.best_score_ < 3000
    refit = rsm.model.mlp({"diamonds": dia}, rvar="price", evar=["carat", "clarity"], mod_type="regression", cv=cv)
    assert refit.alpha == cv.best_params_["alpha"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"params": {"nope": [1, 2]}}, "not a setting"),
        ({"params": {"max_features": [1, 2]}, "method": "magic"}, "method must be"),
        ({"params": {"max_features": [1, 2]}, "metric": "rmse"}, "metric for classification"),
        ({"params": {"min_samples_leaf": {"min": 0, "max": 5, "scale": "log"}}}, "log scale needs a min above 0"),
        ({"params": {"min_samples_leaf": list(range(1, 200))}, "method": "grid"}, "more than the 300 allowed"),
    ],
)
def test_input_errors(rf, kwargs, message):
    params = kwargs.pop("params")
    with pytest.raises(TuneError, match=message):
        tune(rf, params, **kwargs)


def test_ranges_lists_and_layers():
    dims = _dims({"alpha": {"min": 0.0001, "max": 1, "n": 5, "scale": "log"}, "hidden_layer_sizes": ["2", "10,5", (3,)]})
    grid = _grid(dims)
    assert [g["alpha"] for g in grid[::3]] == [0.0001, 0.001, 0.01, 0.1, 1.0]
    assert {g["hidden_layer_sizes"] for g in grid} == {(2,), (10, 5), (3,)}
    narrow = _narrow(dims, [{"alpha": 0.001, "hidden_layer_sizes": (2,)}, {"alpha": 0.01, "hidden_layer_sizes": (2,)}])
    assert narrow[1].values == [(2,)] and 0.0001 <= narrow[0].low < 0.001 and 0.01 < narrow[0].high < 1


def test_result_dict_and_summary(rf):
    res = tune(rf, {"min_samples_leaf": [5, 10]}, method="grid", folds=2, n_jobs=1)
    d = res.to_dict()
    assert d["best_params"] == res.best_params and d["columns"][0] == "rank"
    assert "Simplest within 1 SE" in res.summary_text()
    assert res.plot() is not None
    assert callable(rsm.model.tune)


def test_models_take_the_tuned_settings(rf, titanic):
    res = tune(rf, {"min_samples_leaf": [5, 10]}, method="grid", folds=2, n_jobs=1)
    refit = rsm.model.rforest({"titanic": titanic}, rvar="survived", lev="Yes", evar=["pclass", "sex", "age"], cv=res)
    assert refit.fitted.get_params()["min_samples_leaf"] == res.best_params["min_samples_leaf"]


def test_each_fit_uses_one_thread_and_threads_backend(titanic):
    from pyrsm.model.tune import _model_data

    xgb = rsm.model.xgboost(
        {"titanic": titanic}, rvar="survived", lev="Yes", evar=["pclass", "sex", "age"], n_estimators=20
    )
    assert _model_data(xgb).estimator.get_params()["n_jobs"] == 1
    res = tune(xgb, {"max_depth": [2, 3]}, method="grid", folds=3, backend="threading")
    assert len(res.results) == 2
