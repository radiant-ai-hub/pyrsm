"""Tests for pyrsm.multivariate.full_factor against radiant.multivariate."""

import numpy as np
import pytest
from conftest import mv_mat, mv_sign_align, mv_to_ordered_enum

from pyrsm.multivariate import clean_loadings, full_factor


def _align(A, B):
    """Permute/sign-align columns of A to B (rotation column order can vary)."""
    from itertools import permutations

    A = np.asarray(A, float)
    k = A.shape[1]
    best, out = None, None
    for p in permutations(range(k)):
        Ap = A[:, list(p)]
        Ap = np.column_stack(
            [Ap[:, j] * (1 if (Ap[:, j] @ B[:, j]) >= 0 else -1) for j in range(k)]
        )
        err = np.abs(Ap - B).max()
        if best is None or err < best:
            best, out = err, Ap
    return out, best


TOL = 1e-7
POLY_TOL = 1e-4


def test_full_factor_construction(mv_data):
    ff = full_factor({"shopping": mv_data("shopping")}, "v1:v6", nr_fact=2)
    assert ff.floadings.shape == (6, 2)
    assert ff.communality.shape == (6,)
    assert ff.scores.shape == (20, 2)
    assert ff.fnames == ["RC1", "RC2"]


@pytest.mark.parametrize("rot", ["quartimax", "oblimin", "simplimax"])
def test_full_factor_rotations_parity(mv_data, mv_ref, rot):
    ref = mv_ref(f"full_factor_shopping_2_{rot}")
    ff = full_factor({"shopping": mv_data("shopping")}, "v1:v6", nr_fact=2, rotation=rot)
    Lref = mv_mat(ref["floadings"])
    _, err = _align(ff.floadings, Lref)
    assert err < 1e-4


@pytest.mark.parametrize("case,dataset", [("shopping_ml2", "shopping"), ("toothpaste_ml2", "toothpaste")])
def test_full_factor_ml_parity(mv_data, mv_ref, case, dataset):
    ref = mv_ref(f"full_factor_{case}")
    ff = full_factor({dataset: mv_data(dataset)}, "v1:v6", nr_fact=2, method="ML")
    assert ff.method == "ML"
    Lref = mv_mat(ref["floadings"])
    _, err = _align(ff.floadings, Lref)
    assert err < 1e-3
    comm = np.array(ref["communality"]["values"])
    assert np.abs(np.sort(ff.communality) - np.sort(comm)).max() < 1e-3


def test_clean_loadings_cutoff_and_sort(mv_data):
    ff = full_factor({"shopping": mv_data("shopping")}, "v1:v6", nr_fact=2)
    # cutoff blanks small loadings (string output)
    cl = clean_loadings(ff.loadings_frame(), cutoff=0.5, repl="", dec=2)
    vals = [v for c in ff.fnames for v in cl[c].to_list()]
    assert "" in vals  # at least one blanked
    # fsort reorders rows so each variable's dominant factor is grouped
    cs = clean_loadings(ff.loadings_frame(), fsort=True, dec=3)
    assert cs.height == 6


@pytest.mark.parametrize(
    "case,dataset,vars,nf,rot",
    [
        ("shopping_2", "shopping", "v1:v6", 2, "varimax"),
        ("shopping_2_none", "shopping", "v1:v6", 2, "none"),
        ("shopping_3", "shopping", "v1:v6", 3, "varimax"),
        ("toothpaste_2", "toothpaste", "v1:v6", 2, "varimax"),
        ("diamonds_1", "diamonds", ["price", "carat", "table"], 1, "varimax"),
    ],
)
def test_full_factor_parity(mv_data, mv_ref, case, dataset, vars, nf, rot):
    ref = mv_ref(f"full_factor_{case}")
    ff = full_factor({dataset: mv_data(dataset)}, vars, nr_fact=nf, rotation=rot)

    Lref = mv_mat(ref["floadings"])
    assert np.abs(mv_sign_align(ff.floadings, Lref) - Lref).max() < TOL
    assert np.abs(ff.communality - np.array(ref["communality"]["values"])).max() < TOL
    assert np.abs(ff.eigen - np.array(ref["eigen"])).max() < TOL

    sref = mv_mat(ref["scores"])
    assert np.abs(mv_sign_align(ff.scores, sref) - sref).max() < 1e-6


def test_full_factor_polychoric_loadings(mv_data, mv_ref):
    ref = mv_ref("full_factor_toothpaste_hcor")
    df = mv_to_ordered_enum(mv_data("toothpaste"), [f"v{i}" for i in range(1, 7)])
    ff = full_factor({"toothpaste": df}, "v1:v6", nr_fact=2, hcor=True)

    Lref = mv_mat(ref["floadings"])
    assert np.abs(mv_sign_align(ff.floadings, Lref) - Lref).max() < POLY_TOL
    assert np.abs(ff.communality - np.array(ref["communality"]["values"])).max() < POLY_TOL


def test_full_factor_store(mv_data):
    df = mv_data("shopping")
    ff = full_factor({"shopping": df}, "v1:v6", nr_fact=2)
    out = ff.store(df)
    assert "factor1" in out.columns and "factor2" in out.columns
    assert out.height == df.height
    # stored values equal the score matrix
    assert np.abs(out["factor1"].to_numpy() - ff.scores[:, 0]).max() < 1e-12

    out2 = ff.store(df, name=["f_a", "f_b"])
    assert "f_a" in out2.columns and "f_b" in out2.columns


def test_full_factor_store_filter_alignment(mv_data):
    df = mv_data("shopping")
    ff = full_factor({"shopping": df}, "v1:v6", nr_fact=2, data_filter="v1 > 2")
    assert ff.nobs < df.height
    out = ff.store(df)  # store into the full (unfiltered) data
    assert out.height == df.height
    # filtered-out rows are null; the analysis rows carry the scores
    n_scores = ff.scores.shape[0]
    assert out["factor1"].null_count() == df.height - n_scores


def test_full_factor_summary_smoke(mv_data, capsys):
    full_factor({"shopping": mv_data("shopping")}, "v1:v6", nr_fact=2).summary()
    out = capsys.readouterr().out
    assert "Factor analysis" in out
    assert "Factor loadings" in out
    assert "communalities" in out


def test_full_factor_plot_attr_and_resp(mv_data):
    ff = full_factor({"shopping": mv_data("shopping")}, "v1:v6", nr_fact=2)
    assert ff.plot(plots="attr") is not None
    g = ff.plot(plots=["attr", "resp"])
    assert g is not None
    ff1 = full_factor({"shopping": mv_data("shopping")}, "v1:v6", nr_fact=1)
    assert ff1.plot() is None  # needs >= 2 factors


def test_full_factor_plot_square_limits(mv_data):
    from pyrsm.multivariate._plotting import square_origin_limits

    ff = full_factor({"shopping": mv_data("shopping")}, "v1:v6", nr_fact=2)
    lo, hi = square_origin_limits(ff.scores)
    assert abs(lo + hi) < 1e-9 and hi > 0


def test_full_factor_invalid_method_and_rotation(mv_data):
    df = mv_data("shopping")
    with pytest.raises(ValueError, match="method"):
        full_factor({"shopping": df}, "v1:v6", method="typo")
    with pytest.raises(ValueError, match="rotation"):
        full_factor({"shopping": df}, "v1:v6", nr_fact=2, rotation="typo")
    # aliases are accepted
    assert full_factor({"shopping": df}, "v1:v6", nr_fact=2, method="maximum likelihood").method == "ML"


def test_full_factor_resp_attr_limits_cover_loadings(mv_data):
    """When both attr and resp are shown the plot limit must span the loadings
    (which can extend beyond the respondent score cloud)."""
    from pyrsm.multivariate._plotting import symmetric_limit

    ff = full_factor({"shopping": mv_data("shopping")}, "v1:v6", nr_fact=2)
    lim_both = symmetric_limit(ff.scores, ff.floadings)
    assert lim_both >= symmetric_limit(ff.scores) - 1e-12
    assert lim_both >= symmetric_limit(ff.floadings) - 1e-12
    # the plot builds with both components
    assert ff.plot(plots=["attr", "resp"]) is not None
