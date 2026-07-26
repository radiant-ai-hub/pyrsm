"""Tests for pyrsm.multivariate.pre_factor against radiant.multivariate."""

import numpy as np
import pytest
from conftest import mv_mat, mv_to_ordered_enum

from pyrsm.multivariate import pre_factor
from pyrsm.multivariate._utils import get_vars

TOL = 1e-8
POLY_TOL = 1e-4


# -- API ---------------------------------------------------------------------


def test_get_vars_range_expansion():
    cols = ["id", "v1", "v2", "v3", "v4", "v5", "v6"]
    assert get_vars(cols, "v1:v6") == ["v1", "v2", "v3", "v4", "v5", "v6"]
    assert get_vars(cols, ["v1", "v3"]) == ["v1", "v3"]
    assert get_vars(cols, "v3:v1") == ["v3", "v2", "v1"]
    with pytest.raises(ValueError):
        get_vars(cols, "v1:zz")


def test_pre_factor_construction(mv_data):
    pf = pre_factor({"shopping": mv_data("shopping")}, "v1:v6")
    assert pf.name == "shopping"
    assert pf.vars == [f"v{i}" for i in range(1, 7)]
    assert pf.cmat.shape == (6, 6)
    assert pf.pre_eigen.shape == (6,)
    assert pf.pre_r2.shape == (6,)
    assert set(pf.btest) == {"chisq", "df", "p.value"}


def test_pre_factor_accepts_pandas(mv_data):
    pf = pre_factor(mv_data("shopping").to_pandas(), "v1:v6")
    assert pf.cmat.shape == (6, 6)


# -- numeric parity ----------------------------------------------------------


@pytest.mark.parametrize(
    "case,dataset,vars",
    [
        ("shopping", "shopping", "v1:v6"),
        ("toothpaste", "toothpaste", "v1:v6"),
        ("diamonds", "diamonds", ["price", "carat", "table"]),
    ],
)
def test_pre_factor_parity(mv_data, mv_ref, case, dataset, vars):
    ref = mv_ref(f"pre_factor_{case}")
    pf = pre_factor({dataset: mv_data(dataset)}, vars)

    assert np.abs(pf.cmat - mv_mat(ref["cmat"])).max() < TOL
    assert abs(pf.btest["chisq"] - ref["bartlett"]["chisq"]) < 1e-6
    assert pf.btest["df"] == ref["bartlett"]["df"]
    assert abs(pf.btest["p.value"] - ref["bartlett"]["p.value"]) < TOL
    assert abs(pf.pre_kmo - ref["kmo"]) < TOL
    assert np.abs(pf.msai - np.array(ref["msai"]["values"])).max() < TOL
    assert np.abs(pf.pre_eigen - np.array(ref["eigen"])).max() < TOL
    assert np.abs(pf.pre_r2 - np.array(ref["pre_r2"]["values"]).ravel()).max() < TOL


def test_pre_factor_polychoric_hcor(mv_data, mv_ref):
    ref = mv_ref("pre_factor_toothpaste_hcor")
    df = mv_to_ordered_enum(mv_data("toothpaste"), [f"v{i}" for i in range(1, 7)])
    pf = pre_factor({"toothpaste": df}, "v1:v6", hcor=True)

    assert pf.any_categorical is True
    assert np.abs(pf.cmat - mv_mat(ref["cmat"])).max() < POLY_TOL
    assert abs(pf.pre_kmo - ref["kmo"]) < POLY_TOL
    assert np.abs(pf.pre_eigen - np.array(ref["eigen"])).max() < POLY_TOL
    assert np.abs(pf.pre_r2 - np.array(ref["pre_r2"]["values"]).ravel()).max() < POLY_TOL


def test_pre_factor_mixed_hcor(mv_data, mv_ref):
    """Mixed numeric + ordinal hcor uses polyserial/polychoric correlations.

    Compared against R's polycor::polychor / polycor::polyserial pairwise
    estimators; the 2-step optimizers differ slightly so a looser tolerance is
    used (1e-2)."""
    ref = mv_ref("pre_factor_toothpaste_mixed")
    pf = pre_factor(
        {"toothpaste": mv_data("toothpaste")},
        ["v1", "v2", "gender", "age.cat"],
        hcor=True,
    )
    assert pf.any_categorical is True
    assert np.abs(pf.cmat - mv_mat(ref["cmat"])).max() < 1e-2
    assert abs(pf.pre_kmo - ref["kmo"]) < 1e-2
    assert np.abs(pf.pre_eigen - np.array(ref["eigen"])).max() < 1e-2


def test_pre_factor_no_variation_raises(mv_data):
    import polars as pl

    df = mv_data("shopping").with_columns(pl.lit(5.0).alias("const"))
    with pytest.raises(ValueError, match="no variation"):
        pre_factor({"shopping": df}, ["v1", "const"])


def test_pre_factor_data_filter(mv_data):
    df = mv_data("shopping")
    pf = pre_factor({"shopping": df}, "v1:v6", data_filter="v1 > 2")
    assert pf.nobs < df.height


# -- output ------------------------------------------------------------------


def test_pre_factor_summary_smoke(mv_data, capsys):
    pre_factor({"shopping": mv_data("shopping")}, "v1:v6").summary()
    out = capsys.readouterr().out
    assert "Pre-factor analysis diagnostics" in out
    assert "Bartlett test" in out
    assert "KMO test" in out


def test_pre_factor_plot_smoke(mv_data):
    pf = pre_factor({"shopping": mv_data("shopping")}, "v1:v6")
    plots = pf.plot(plots=("scree", "change"))
    assert isinstance(plots, list) and len(plots) == 2
    assert pf.plot(plots="scree") is not None
