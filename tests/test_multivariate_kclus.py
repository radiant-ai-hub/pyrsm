"""Tests for pyrsm.multivariate.kclus against radiant.multivariate."""

import numpy as np
import pytest
from conftest import mv_mat

from pyrsm.multivariate import kclus

TOL = 1e-6


def _same_partition(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return bool(((a[:, None] == a[None, :]) == (b[:, None] == b[None, :])).all())


def _row_match_err(A, B):
    used, err = set(), 0.0
    for a in A:
        d = [np.abs(a - b).max() if i not in used else 1e18 for i, b in enumerate(B)]
        j = int(np.argmin(d))
        used.add(j)
        err = max(err, d[j])
    return err


# -- K-means -----------------------------------------------------------------


@pytest.mark.parametrize(
    "case,dataset,nc",
    [
        ("shopping_2", "shopping", 2),
        ("shopping_3", "shopping", 3),
        ("toothpaste_3", "toothpaste", 3),
    ],
)
def test_kclus_kmeans_parity(mv_data, mv_ref, case, dataset, nc):
    ref = mv_ref(f"kclus_{case}")
    km = kclus({dataset: mv_data(dataset)}, "v1:v6", nr_clus=nc)
    cm_ref = mv_mat(ref["clus_means"])
    assert _row_match_err(km.clus_means.to_numpy(), cm_ref) < TOL
    assert abs(km.totss - ref["totss"]) < 1e-6
    assert abs(km.betweenss - ref["betweenss"]) < 1e-6
    assert sorted(km.sizes.tolist()) == sorted(ref["sizes"])
    assert np.abs(np.sort(km.withinss) - np.sort(np.array(ref["withinss"]))).max() < 1e-6


def test_kclus_kmeans_drops_categoricals(mv_data, capsys):
    km = kclus({"toothpaste": mv_data("toothpaste")}, ["v1", "v2", "gender"], nr_clus=2)
    out = capsys.readouterr().out
    assert "gender" not in km.vars
    assert "Categorical" in out


# -- K-prototypes ------------------------------------------------------------


def test_kclus_kproto_parity(mv_data, mv_ref):
    ref = mv_ref("kclus_toothpaste_kproto")
    kp = kclus(
        {"toothpaste": mv_data("toothpaste")},
        ["v1", "v2", "v3", "gender"],
        fun="kproto",
        nr_clus=3,
    )
    assert kp.fun == "kproto"
    assert abs(kp.lambda_ - ref["lambda"]) < 1e-6
    assert sorted(kp.sizes.tolist()) == sorted(ref["sizes"])
    assert abs(kp.totss - ref["totss"]) < 1e-5
    assert abs(kp.tot_withinss - ref["tot.withinss"]) < 1e-5
    assert abs(kp.betweenss - ref["betweenss"]) < 1e-5
    assert _same_partition(kp.cluster, ref["cluster"])


def test_kclus_kproto_categorical_modes(mv_data):
    kp = kclus(
        {"toothpaste": mv_data("toothpaste")},
        ["v1", "v2", "v3", "gender"],
        fun="kproto",
        nr_clus=3,
    )
    # categorical center column shows "<mode> (xx%)"
    gender_centers = kp.clus_means["gender"].to_list()
    assert all("(" in c and "%" in c for c in gender_centers)


def test_kclus_kproto_seed_is_used(mv_data):
    """With hc_init=False the seed drives the random starts (regression test:
    the seed used to be ignored / hard-coded)."""
    df = mv_data("toothpaste")
    common = dict(vars=["v1", "v2", "v3", "gender"], fun="kproto", nr_clus=3, hc_init=False)
    a = kclus({"toothpaste": df}, seed=1, **common)
    a2 = kclus({"toothpaste": df}, seed=1, **common)
    # same seed -> fully reproducible
    assert _same_partition(a.cluster, a2.cluster)
    # the seed actually changes the random starts: across several seeds at least
    # one yields a different partition (impossible if the seed were ignored)
    others = [kclus({"toothpaste": df}, seed=s, **common) for s in (7, 42, 123, 999)]
    assert any(not _same_partition(a.cluster, o.cluster) for o in others)


def test_kclus_kproto_falls_back_to_kmeans(mv_data, capsys):
    km = kclus({"shopping": mv_data("shopping")}, "v1:v6", fun="kproto", nr_clus=2)
    out = capsys.readouterr().out
    assert km.fun == "kmeans"
    assert "K-means used" in out


# -- output ------------------------------------------------------------------


def test_kclus_store_default_name(mv_data):
    df = mv_data("shopping")
    km = kclus({"shopping": df}, "v1:v6", nr_clus=2)
    out = km.store(df)
    assert "kclus2" in out.columns
    assert out["kclus2"].n_unique() == 2


def test_kclus_summary_smoke(mv_data, capsys):
    kclus({"shopping": mv_data("shopping")}, "v1:v6", nr_clus=2).summary()
    out = capsys.readouterr().out
    assert "cluster analysis" in out
    assert "within-cluster heterogeneity" in out.lower()


@pytest.mark.parametrize("ptype", ["density", "bar", "scatter"])
def test_kclus_plot_modes_numeric(mv_data, ptype):
    km = kclus({"shopping": mv_data("shopping")}, "v1:v6", nr_clus=2)
    out = km.plot(plots=ptype)
    assert out is not None


@pytest.mark.parametrize("ptype", ["density", "bar", "scatter"])
def test_kclus_plot_modes_categorical(mv_data, ptype):
    km = kclus(
        {"toothpaste": mv_data("toothpaste")},
        ["v1", "v2", "gender"],
        fun="kproto",
        nr_clus=2,
    )
    out = km.plot(plots=ptype)
    assert out is not None
