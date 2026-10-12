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


def _titanic_with_missing():
    import polars as pl
    from pathlib import Path

    t = pl.read_parquet(Path(__file__).parents[1] / "examples/data/model/titanic.parquet")
    # five missing ages: rows a model drops, so residuals must leave them empty
    return t.with_columns(
        pl.when(pl.int_range(pl.len()) < 5).then(None).otherwise(pl.col("age")).alias("age")
    )


def test_model_correlation_plot_is_the_basics_plot_with_every_variable(monkeypatch):
    """Categorical explanatory variables are kept (Basics' categorical mode),
    not silently dropped; all-numeric data stay Pearson."""
    import matplotlib

    matplotlib.use("Agg")
    import polars as pl
    import pyrsm as rsm
    import sys

    import pyrsm.basics.correlation  # noqa: F401  (the module, for monkeypatch)

    bc = sys.modules["pyrsm.basics.correlation"]
    seen = []
    real = bc.correlation
    monkeypatch.setattr(bc, "correlation", lambda *a, **k: seen.append(k) or real(*a, **k))
    t = _titanic_with_missing().drop_nulls("age")
    rsm.model.logistic({"t": t}, rvar="survived", lev="Yes", evar=["pclass", "sex", "age"]).plot("corr")
    assert seen[-1]["method"] == "polychoric"
    assert seen[-1]["vars"] == ["survived", "pclass", "sex", "age"]
    num = pl.DataFrame({"y": [1.0, 2.0, 4.0, 3.0, 5.0], "x": [1.0, 3.0, 2.0, 5.0, 4.0]})
    rsm.model.regress({"d": num}, rvar="y", evar=["x"]).plot("corr")
    assert seen[-1]["method"] == "pearson"


def test_every_ml_model_has_a_dashboard_and_residuals_on_their_rows():
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    import pyrsm as rsm

    t = _titanic_with_missing()
    ev = ["pclass", "sex", "age"]
    for est in (rsm.model.rforest, rsm.model.mlp, rsm.model.xgboost):
        clf = est({"t": t}, rvar="survived", lev="Yes", evar=ev)
        assert clf.plot("dashboard") is not None
        r = clf.residuals
        assert len(r) == t.height and r.null_count() == 5 and r[:5].null_count() == 5
        # actual (1 = Yes) minus the predicted probability of Yes
        assert float(np.nanmax(np.abs(r.drop_nulls().to_numpy()))) <= 1.0
        reg = est({"t": t}, rvar="age", evar=["pclass", "sex"], mod_type="regression")
        assert reg.plot("dashboard") is not None
        assert len(reg.residuals) == t.height
    tree = rsm.model.crtree({"t": t}, rvar="survived", lev="Yes", evar=ev)
    assert tree.plot("dashboard") is not None
    assert tree._kept_rows.sum() == len(tree.residuals) == t.height - 5


def test_mnl_residuals_are_one_minus_the_probability_of_the_choice():
    import pyrsm as rsm

    t = _titanic_with_missing()
    m = rsm.model.mnl({"t": t}, rvar="pclass", lev="1st", evar=["sex", "age"])
    r = m.residuals
    assert len(r) == t.height and r.null_count() == 5
    assert 0.0 <= r.drop_nulls().min() and r.drop_nulls().max() <= 1.0


def test_categorical_correlations_follow_the_level_order():
    """A categorical variable is coded in its level order (Enum), not in the
    order its values first appear: bigger diamonds have worse clarity."""
    import polars as pl
    import pyrsm as rsm
    from pathlib import Path

    d = pl.read_parquet(Path(__file__).parents[1] / "examples/data/data/diamonds.parquet")
    cr = rsm.basics.correlation({"d": d}, vars=["carat", "clarity"], method="polychoric").cr
    assert cr[0, 1] < -0.3
    # the same answer when the rows come in another order
    shuffled = d.sample(fraction=1.0, shuffle=True, seed=1)
    cr2 = rsm.basics.correlation({"d": shuffled}, vars=["carat", "clarity"], method="polychoric").cr
    assert abs(cr[0, 1] - cr2[0, 1]) < 1e-9
