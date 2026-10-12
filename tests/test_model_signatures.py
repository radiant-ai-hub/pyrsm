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
    regress: {"data", "cmd", "data_cmd", "ci", "conf", "dec", "interval"},
    logistic: {"data", "cmd", "data_cmd", "ci", "conf", "dec", "interval"},
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


@pytest.mark.parametrize("model", [regress, logistic, rforest, mlp, xgboost])
def test_partial_dependence_defaults(model):
    """Every model: the average response line on, ICE lines off."""
    params = inspect.signature(model.plot).parameters
    assert params["hline"].default is True
    assert params["ice"].default is False


def test_logistic_conf_int_is_computed_once_per_level(monkeypatch):
    """Profile intervals refit the model per bound: summary, report and a later
    odds request reuse them, and a new confidence level computes new ones."""
    import numpy as np
    import polars as pl
    import pyrsm as rsm
    import pyrsm.model.model as mm

    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    y = np.where(rng.random(300) < 1 / (1 + np.exp(-x)), "yes", "no")
    m = rsm.model.logistic({"d": pl.DataFrame({"y": y, "x": x})}, rvar="y", lev="yes", evar=["x"])
    assert m.ci_type == "profile"
    calls = []
    real = mm.profile_ci
    monkeypatch.setattr(mm, "profile_ci", lambda *a, **k: calls.append(1) or real(*a, **k))
    first = m.conf_int()
    first.iloc[0, 0] = 99.0  # a caller's change must not leak into the cache
    again = m.conf_int(0.95)
    assert len(calls) == 1 and again.iloc[0, 0] != 99.0
    m.conf_int(0.9)
    assert len(calls) == 2


def test_logistic_distribution_plot_shows_the_response_levels():
    """The response is estimated as 0/1 but plotted with its own levels."""
    import matplotlib

    matplotlib.use("Agg")
    import polars as pl
    import pyrsm as rsm

    d = pl.DataFrame(
        {"y": pl.Series(["yes", "no", "no", "yes", "no", "yes", "no", "no"], dtype=pl.Enum(["yes", "no"])),
         "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}
    )
    m = rsm.model.logistic({"d": d}, rvar="y", lev="yes", evar=["x"])
    fig = m.plot("dist")
    fig = getattr(fig, "draw", lambda: fig)() if not hasattr(fig, "axes") else fig
    labels = {t.get_text() for ax in fig.axes for t in ax.get_xticklabels()}
    assert {"yes", "no"} <= labels and "0" not in labels


def test_logistic_dashboard_names_the_other_level(monkeypatch):
    """The dashboard legend reads Yes / No, not Yes / not Yes."""
    import polars as pl
    import pyrsm as rsm
    import pyrsm.model.model as mm

    d = pl.DataFrame(
        {"y": pl.Series(["yes", "no", "no", "yes", "no", "yes", "no", "no"], dtype=pl.Enum(["yes", "no"])),
         "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}
    )
    m = rsm.model.logistic({"d": d}, rvar="y", lev="yes", evar=["x"])
    seen = {}
    real = mm.logit_dashboard
    monkeypatch.setattr(mm, "logit_dashboard", lambda *a, **k: seen.update(k) or real(*a, **k))
    m.plot("dashboard")
    assert seen["lev"] == "yes" and seen["other"] == "no"
