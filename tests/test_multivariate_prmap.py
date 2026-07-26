"""Tests for pyrsm.multivariate.prmap against radiant.multivariate."""

import numpy as np
import pytest
from conftest import mv_mat

from pyrsm.multivariate import prmap

TOL = 1e-7


def _visible_label_boxes(g):
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox

    fig = g.draw(show=False)
    fig.canvas.draw()
    ax = fig.axes[0]
    renderer = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(renderer)
    boxes = {
        t.get_text(): t.get_window_extent(renderer)
        for t in ax.texts
        if t.get_visible()
    }
    contained = {
        label: (
            box.x0 >= ax_box.x0
            and box.x1 <= ax_box.x1
            and box.y0 >= ax_box.y0
            and box.y1 <= ax_box.y1
        )
        for label, box in boxes.items()
    }
    overlaps = []
    items = list(boxes.items())
    for i, (label_a, box_a) in enumerate(items):
        shrunk_a = Bbox.from_extents(
            box_a.x0 + 1, box_a.y0 + 1, box_a.x1 - 1, box_a.y1 - 1
        )
        for label_b, box_b in items[i + 1:]:
            shrunk_b = Bbox.from_extents(
                box_b.x0 + 1, box_b.y0 + 1, box_b.x1 - 1, box_b.y1 - 1
            )
            if shrunk_a.overlaps(shrunk_b):
                overlaps.append((label_a, label_b))
    legend = ax.get_legend()
    plt.close(fig)
    return set(boxes), contained, overlaps, legend


def test_prmap_construction(mv_data):
    pm = prmap({"computer": mv_data("computer")}, "brand", "high_end:business")
    assert pm.loadings.shape == (7, 2)  # high_end..business is 7 attributes
    assert pm.scores.height == 5
    assert pm.pref_cor is None


@pytest.mark.parametrize(
    "case,dataset,brand,attr,pref",
    [
        ("computer", "computer", "brand", "high_end:business", ""),
        (
            "retailers",
            "retailers",
            "retailer",
            "good_value:cluttered",
            ["segment1", "segment2"],
        ),
    ],
)
def test_prmap_parity(mv_data, mv_ref, case, dataset, brand, attr, pref):
    ref = mv_ref(f"prmap_{case}")
    pm = prmap({dataset: mv_data(dataset)}, brand, attr, pref=pref)

    assert pm.scores["brand"].to_list() == ref["scores"]["rownames"]
    sref = mv_mat(ref["scores"])
    assert np.abs(pm.scores.drop("brand").to_numpy() - sref).max() < TOL

    Lref = mv_mat(ref["loadings"])
    assert np.abs(pm.loadings - Lref).max() < TOL
    assert np.abs(pm.communality - np.array(ref["communality"]["values"])).max() < TOL
    assert np.abs(pm.eigen - np.array(ref["eigen"])).max() < TOL

    if pref:
        pcr = mv_mat(ref["pref_cor"])
        assert np.abs(pm.pref_cor.drop("pref").to_numpy() - pcr).max() < TOL


def test_prmap_3d_parity(mv_data, mv_ref):
    ref = mv_ref("prmap_retailers_3d")
    pm = prmap(
        {"retailers": mv_data("retailers")},
        "retailer",
        "good_value:cluttered",
        pref=["segment1", "segment2"],
        nr_dim=3,
    )
    assert pm.loadings.shape == (7, 3)
    Lref = mv_mat(ref["loadings"])
    # align columns by absolute value (varimax column order can differ in 3D)
    assert np.abs(np.sort(np.abs(pm.loadings), axis=None) - np.sort(np.abs(Lref), axis=None)).max() < 1e-5


def test_prmap_plot_components(mv_data):
    pm = prmap(
        {"retailers": mv_data("retailers")},
        "retailer",
        "good_value:cluttered",
        pref=["segment1", "segment2"],
    )
    def n_seg(g):
        return sum(type(la.geom).__name__ == "geom_segment" for la in g.layers)

    # brand only -> points + label leader lines, but no attribute arrows
    g_b = pm.plot(plots=["brand"])
    seg_b = n_seg(g_b)

    # brand + attr -> adds an arrow segment layer on top of the leader lines
    g_ba = pm.plot(plots=["brand", "attr"])
    assert n_seg(g_ba) == seg_b + 1
    assert any(type(la.geom).__name__ == "geom_point" for la in g_ba.layers)

    # brand + attr + pref -> still builds (preference arrows present)
    g_all = pm.plot(plots=["brand", "attr", "pref"])
    assert g_all is not None


def test_prmap_plot_draws_readable_labels(mv_data):
    pm = prmap(
        {"retailers": mv_data("retailers")},
        "retailer",
        "good_value:cluttered",
        pref=["segment1", "segment2"],
    )
    labels, contained, overlaps, legend = _visible_label_boxes(
        pm.plot(plots=["brand", "attr", "pref"])
    )
    expected = set(pm.scores["brand"].to_list()) | set(pm.attr) | set(pm.pref)
    assert expected <= labels
    assert [label for label, ok in contained.items() if not ok] == []
    assert overlaps == []
    assert legend is None


def test_prmap_plot_square_limits(mv_data):
    from pyrsm.multivariate._plotting import square_origin_limits

    pm = prmap({"computer": mv_data("computer")}, "brand", "high_end:business")
    brand = pm.scores.to_numpy()[:, 1:].astype(float)
    lo, hi = square_origin_limits(brand, pm.loadings * 2.0)
    assert abs(lo + hi) < 1e-9 and hi > 0
    assert pm.plot(plots=["brand", "attr"]) is not None


def test_prmap_summary_smoke(mv_data, capsys):
    pm = prmap({"computer": mv_data("computer")}, "brand", "high_end:business")
    pm.summary()
    out = capsys.readouterr().out
    assert "Attribute based brand map" in out
    assert "Attribute - Factor loadings" in out
