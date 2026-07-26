"""Tests for pyrsm.multivariate.hclus against radiant.multivariate."""

import numpy as np
import pytest

from pyrsm.multivariate import hclus


def _same_partition(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return bool(((a[:, None] == a[None, :]) == (b[:, None] == b[None, :])).all())


def test_hclus_construction(mv_data):
    hc = hclus({"shopping": mv_data("shopping")}, "v1:v6")
    assert hc.height.shape == (19,)
    assert hc.linkage.shape == (19, 4)


@pytest.mark.parametrize(
    "case,dataset,labels",
    [
        ("shopping", "shopping", "none"),
        ("toothpaste", "toothpaste", "none"),
        ("toothpaste_id", "toothpaste", "id"),
    ],
)
def test_hclus_numeric_parity(mv_data, mv_ref, case, dataset, labels):
    ref = mv_ref(f"hclus_{case}")
    hc = hclus({dataset: mv_data(dataset)}, "v1:v6", labels=labels)
    assert np.abs(np.sort(hc.height) - np.sort(np.array(ref["height"]))).max() < 1e-6
    for k in (2, 3):
        assert _same_partition(hc.cutree(k), ref["cutree"][str(k)])


def test_hclus_gower_parity(mv_data, mv_ref):
    """Mixed numeric/categorical data switches to Gower and matches R.

    Gower distances match R's ``gower::gower_dist`` exactly; the merge heights
    agree with R's ``ward.D`` to ~1e-2 (residual differences are tie-breaking
    in the agglomeration order on equal Gower distances), while the resulting
    partitions match exactly.
    """
    ref = mv_ref("hclus_toothpaste_gower")
    hc = hclus({"toothpaste": mv_data("toothpaste")}, ["v1", "v2", "v3", "gender"])
    assert hc.distance == "gower"
    assert hc.any_categorical is True
    for k in (2, 3):
        assert _same_partition(hc.cutree(k), ref["cutree"][str(k)])
    assert np.abs(np.sort(hc.height) - np.sort(np.array(ref["height"]))).max() < 5e-2


def test_hclus_auto_gower_switch(mv_data, capsys):
    hc = hclus({"toothpaste": mv_data("toothpaste")}, ["gender", "v1"])
    out = capsys.readouterr().out
    assert hc.distance == "gower"
    assert "Gower" in out


def test_hclus_store_default_name(mv_data):
    df = mv_data("shopping")
    hc = hclus({"shopping": df}, "v1:v6")
    out = hc.store(df, nr_clus=3)
    assert "hclus3" in out.columns
    assert out["hclus3"].n_unique() == 3


def test_hclus_label_uniqueness_fallback(mv_data, capsys):
    # gender is not unique -> falls back to row numbers
    hclus({"toothpaste": mv_data("toothpaste")}, "v1:v6", labels="gender")
    out = capsys.readouterr().out
    assert "not unique" in out


def test_hclus_scree_change_normalized(mv_data):
    hc = hclus({"shopping": mv_data("shopping")}, "v1:v6")
    g = hc.plot(plots="scree")
    ys = g.data["y"].to_numpy()
    assert ys.max() <= 1.0 + 1e-9  # normalized to [0, 1]


def test_hclus_scree_cutoff_filters(mv_data):
    hc = hclus({"shopping": mv_data("shopping")}, "v1:v6")
    full = hc.plot(plots="scree", cutoff=0.0).data
    cut = hc.plot(plots="scree", cutoff=0.2).data
    # the cutoff drops the smallest (many-cluster) merges from the plot
    assert cut.shape[0] < full.shape[0]
    assert (cut["y"].to_numpy() > 0.2).all()


def test_hclus_plots_smoke(mv_data):
    hc = hclus({"shopping": mv_data("shopping")}, "v1:v6")
    assert hc.plot(plots="dendro") is not None
    assert hc.plot(plots="scree") is not None
    assert hc.plot(plots="change") is not None
