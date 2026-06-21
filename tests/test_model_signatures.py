"""Public-API signature snapshots for the five regression-family models.

The rsm-django-radiant component builds its catalog-driven model UI on top of
pyrsm's public parameter names (rvar/evar/lev/ivar/mod_type, plot ``plots``,
predict ``data``/``cmd``/``data_cmd``, ``figsize``, ...). These tests are a
tripwire: if a public parameter the UI relies on is renamed or removed, they
fail here in pyrsm rather than silently breaking the generated code or the
adapters downstream.

They assert *presence* of the expected parameters (a superset check), not the
exact full signature, so adding new optional params does not break them.
"""

import inspect

import pytest

from pyrsm.model.logistic import logistic
from pyrsm.model.mlp import mlp
from pyrsm.model.regress import regress
from pyrsm.model.rforest import rforest
from pyrsm.model.xgboost import xgboost


def params(func) -> set[str]:
    return set(inspect.signature(func).parameters)


# (class, expected __init__ params, expected predict params, expected plot params)
CONSTRUCTOR_PARAMS = {
    regress: {"data", "rvar", "evar", "ivar", "formula"},
    logistic: {"data", "rvar", "lev", "evar", "ivar", "formula", "weights"},
    rforest: {
        "data", "rvar", "lev", "evar", "n_estimators", "max_features",
        "min_samples_leaf", "max_samples", "oob_score", "random_state", "mod_type",
    },
    mlp: {
        "data", "rvar", "lev", "evar", "hidden_layer_sizes", "alpha", "activation",
        "solver", "batch_size", "learning_rate_init", "max_iter", "random_state",
        "mod_type",
    },
    xgboost: {
        "data", "rvar", "lev", "evar", "n_estimators", "max_depth",
        "min_child_weight", "learning_rate", "subsample", "colsample_bytree",
        "random_state", "mod_type",
    },
}

PREDICT_PARAMS = {
    regress: {"data", "cmd", "data_cmd", "ci", "conf", "dec"},
    logistic: {"data", "cmd", "data_cmd", "ci", "conf", "dec"},
    rforest: {"data", "cmd", "data_cmd", "dec"},
    mlp: {"data", "cmd", "data_cmd", "scale", "means", "stds", "dec"},
    xgboost: {"data", "cmd", "data_cmd", "dec"},
}

PLOT_PARAMS = {
    regress: {"plots", "data", "incl", "excl", "incl_int", "nobs", "fix", "hline",
              "ice", "ice_nobs", "nnv", "minq", "maxq", "figsize", "ret"},
    logistic: {"plots", "data", "incl", "excl", "incl_int", "nobs", "fix", "hline",
               "ice", "ice_nobs", "nnv", "minq", "maxq", "ret"},
    rforest: {"plots", "data", "incl", "excl", "incl_int", "nobs", "fix", "hline",
              "ice", "ice_nobs", "nnv", "minq", "maxq", "figsize", "ret"},
    mlp: {"plots", "data", "incl", "excl", "incl_int", "nobs", "fix", "hline",
          "ice", "ice_nobs", "nnv", "minq", "maxq", "figsize", "ret"},
    xgboost: {"plots", "data", "incl", "excl", "incl_int", "nobs", "fix", "hline",
              "ice", "ice_nobs", "nnv", "minq", "maxq", "figsize", "ret"},
}


@pytest.mark.parametrize("cls,expected", CONSTRUCTOR_PARAMS.items(), ids=lambda c: getattr(c, "__name__", c))
def test_constructor_params(cls, expected):
    missing = expected - params(cls.__init__)
    assert not missing, f"{cls.__name__}.__init__ missing params: {sorted(missing)}"


@pytest.mark.parametrize("cls,expected", PREDICT_PARAMS.items(), ids=lambda c: getattr(c, "__name__", c))
def test_predict_params(cls, expected):
    missing = expected - params(cls.predict)
    assert not missing, f"{cls.__name__}.predict missing params: {sorted(missing)}"


@pytest.mark.parametrize("cls,expected", PLOT_PARAMS.items(), ids=lambda c: getattr(c, "__name__", c))
def test_plot_params(cls, expected):
    missing = expected - params(cls.plot)
    assert not missing, f"{cls.__name__}.plot missing params: {sorted(missing)}"


def test_mlp_plot_has_figsize():
    """mlp.plot must expose figsize like regress/rforest/xgboost."""
    assert "figsize" in params(mlp.plot)


def test_summary_dec_param():
    """All five summaries accept a decimals (dec) control."""
    for cls in (regress, logistic, rforest, mlp, xgboost):
        assert "dec" in params(cls.summary), f"{cls.__name__}.summary missing dec"
