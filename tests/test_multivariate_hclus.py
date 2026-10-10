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

    Gower distances reproduce ``gower::gower_dist`` to the last bit and the
    merges follow R's hclust, so tied distances merge in R's order too.
    """
    ref = mv_ref("hclus_toothpaste_gower")
    hc = hclus({"toothpaste": mv_data("toothpaste")}, ["v1", "v2", "v3", "gender"])
    assert hc.distance == "gower"
    assert hc.any_categorical is True
    for k in (2, 3):
        assert _same_partition(hc.cutree(k), ref["cutree"][str(k)])
    assert np.abs(hc.height - np.array(ref["height"])).max() < 1e-10


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


def test_hclus_dendro_is_plotnine(mv_data):
    from plotnine import ggplot

    hc = hclus({"shopping": mv_data("shopping")}, "v1:v6")
    full = hc.plot(plots="dendro", cutoff=0)
    assert isinstance(full, ggplot)
    # one inverted "U" (3 segments) per merge, leaves labelled in tree order
    assert full.data.shape[0] == 3 * (hc.nobs - 1)
    cut = hc.plot(plots="dendro")
    assert isinstance(cut, ggplot)
    assert cut.labels.title == "Cutoff dendrogram"
    with pytest.raises(ValueError, match="Unknown plot type"):
        hc.plot(plots="bogus")


def test_hclus_change_keeps_label_order(mv_data):
    hc = hclus({"shopping": mv_data("shopping")}, "v1:v6")
    g = hc.plot(plots="change", cutoff=0)
    cats = list(g.data["nr_clus"].cat.categories)
    assert cats[:3] == ["1-2", "2-3", "3-4"]


# -- every distance / method in Radiant's UI ------------------------------------

_HC_REF = "hclus_shopping_distances"


def _hc_cases():
    from conftest import mv_load_ref

    return sorted(mv_load_ref(_HC_REF)["cases"])


def _hc_data(mv_data, which):
    import polars as pl

    df = mv_data("shopping")
    if which == "shopping_bin":
        # zeros mixed with values 1..3: R's binary distance treats any non-zero
        # value as "on", SciPy's jaccard would compare the values
        df = df.with_columns([(pl.col(f"v{i}") - 4).clip(lower_bound=0) for i in range(1, 7)])
    return df


@pytest.mark.parametrize("key", _hc_cases())
def test_hclus_distance_method_matches_r(mv_data, mv_ref, key):
    from pyrsm.multivariate.hclus import _r_dissimilarity

    case = mv_ref(_HC_REF)["cases"][key]
    df = _hc_data(mv_data, case["data"])
    hc = hclus(
        {"shopping": df},
        "v1:v6",
        distance=case["distance"],
        method=case["method"],
        standardize=case["standardize"],
    )
    assert hc.distance == case["distance"] and hc.method == case["method"]

    # R's dist() semantics (JSON keeps 15 significant digits)
    if case["distance"] != "gower":
        Z = df.select([f"v{i}" for i in range(1, 7)]).to_numpy().astype(float)
        if case["standardize"]:
            Z = (Z - Z.mean(axis=0)) / Z.std(axis=0, ddof=1)
        dR = _r_dissimilarity(Z, case["distance"])
        assert np.abs(dR - np.array(case["dist"])).max() < 1e-12

    # same merges in the same order (ties included), so heights match in
    # merge order and cutree numbers clusters the way R does
    assert np.abs(hc.height - np.array(case["height"])).max() < 1e-10
    for k, ref in case["cutree"].items():
        assert hc.cutree(int(k)).tolist() == ref


def test_hclus_cutree_numbering_matches_r(mv_data, mv_ref):
    # R numbers clusters by first appearance in the data
    ref = mv_ref("hclus_shopping")
    hc = hclus({"shopping": mv_data("shopping")}, "v1:v6")
    for k in (2, 3):
        assert hc.cutree(k).tolist() == ref["cutree"][str(k)]


def test_hclus_ward_d2_still_supported(mv_data):
    hc = hclus({"shopping": mv_data("shopping")}, "v1:v6", distance="euclidean", method="ward.D2")
    hc2 = hclus({"shopping": mv_data("shopping")}, "v1:v6", method="ward.D")
    # ward.D2 on Euclidean distances is ward.D on squared distances, sqrt'ed
    assert np.allclose(hc.height**2, hc2.height)


def test_hclus_invalid_distance_and_method(mv_data):
    df = mv_data("shopping")
    with pytest.raises(ValueError, match="distance"):
        hclus({"shopping": df}, "v1:v6", distance="cosine")
    with pytest.raises(ValueError, match="method"):
        hclus({"shopping": df}, "v1:v6", method="ward")


@pytest.mark.parametrize("labels", ["id", "age", "gender", "age.cat"])
def test_hclus_labels_any_column(mv_data, labels):
    hc = hclus({"toothpaste": mv_data("toothpaste")}, "v1:v6", labels=labels)
    assert hc.vars == [f"v{i}" for i in range(1, 7)]
    assert hc.distance == "sq.euclidian"
    assert len(hc.row_labels) == 60
    assert hc.plot("dendro", cutoff=0) is not None
