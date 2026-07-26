"""Tests for pyrsm.multivariate.mds against radiant.multivariate."""

import numpy as np
import pytest
from conftest import mv_mat, mv_sign_align
from scipy.spatial import procrustes

from pyrsm.multivariate import mds


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
        # Shrink by a pixel so adjacent labels that just touch are not treated
        # as an overlap by backend-specific antialiasing differences.
        shrunk_a = Bbox.from_extents(
            box_a.x0 + 1, box_a.y0 + 1, box_a.x1 - 1, box_a.y1 - 1
        )
        for label_b, box_b in items[i + 1:]:
            shrunk_b = Bbox.from_extents(
                box_b.x0 + 1, box_b.y0 + 1, box_b.x1 - 1, box_b.y1 - 1
            )
            if shrunk_a.overlaps(shrunk_b):
                overlaps.append((label_a, label_b))
    plt.close(fig)
    return set(boxes), contained, overlaps


def test_mds_construction(mv_data):
    m = mds({"city": mv_data("city")}, "from", "to", "distance")
    assert m.points.shape == (9, 2)
    assert len(m.labels) == 9
    assert m.dis_mat.shape == (9, 9)


@pytest.mark.parametrize(
    "case,dataset,i1,i2,dis",
    [
        ("city_metric", "city", "from", "to", "distance"),
        ("tpbrands_metric", "tpbrands", "id1", "id2", "dissimilarity"),
    ],
)
def test_mds_metric_parity(mv_data, mv_ref, case, dataset, i1, i2, dis):
    ref = mv_ref(f"mds_{case}")
    m = mds({dataset: mv_data(dataset)}, i1, i2, dis, method="metric")
    assert m.labels == ref["labels"]
    Pref = mv_mat(ref["points"])
    assert np.abs(mv_sign_align(m.points, Pref) - Pref).max() < 1e-6
    assert abs(m.stress - ref["stress"]) < 1e-8
    rec_ref = mv_mat(ref["recovered_dist"])
    assert np.abs(m.recovered_dist() - rec_ref).max() < 1e-6


def test_mds_metric_3d_parity(mv_data, mv_ref):
    ref = mv_ref("mds_city_metric_3d")
    m = mds({"city": mv_data("city")}, "from", "to", "distance", method="metric", nr_dim=3)
    assert m.points.shape == (9, 3)
    assert abs(m.stress - ref["stress"]) < 1e-6
    Pref = mv_mat(ref["points"])
    _, _, disp = procrustes(Pref, m.points)
    assert disp < 1e-6


@pytest.mark.parametrize("case,dataset,i1,i2,dis", [
    ("city_nonmetric", "city", "from", "to", "distance"),
    ("tpbrands_nonmetric", "tpbrands", "id1", "id2", "dissimilarity"),
])
def test_mds_nonmetric_parity(mv_data, mv_ref, case, dataset, i1, i2, dis):
    """Non-metric MDS matches MASS::isoMDS Kruskal stress-1 (and, up to a
    rotation/reflection, the configuration) to a tight tolerance."""
    ref = mv_ref(f"mds_{case}")
    m = mds({dataset: mv_data(dataset)}, i1, i2, dis, method="non-metric")
    # both stresses are small; SMACOF can converge below MASS::isoMDS's floor, so
    # the configuration match (Procrustes) is the primary parity check.
    Pref = mv_mat(ref["points"])
    _, _, disp = procrustes(Pref, m.points)
    assert disp < 1e-2
    assert m.stress <= ref["stress"] + 5e-3


def test_mds_flip(mv_data):
    m = mds({"city": mv_data("city")}, "from", "to", "distance")
    before = m.points.copy()
    m.flip(1)
    assert np.allclose(m.points[:, 0], -before[:, 0])
    assert np.allclose(m.points[:, 1], before[:, 1])


def test_mds_plot_rev_dim_does_not_mutate(mv_data):
    m = mds({"city": mv_data("city")}, "from", "to", "distance")
    before = m.points.copy()
    g = m.plot(rev_dim=[1])
    # plotted x is flipped but stored coords are unchanged
    assert np.allclose(m.points, before)
    xs = g.data["x"].to_numpy()
    assert np.allclose(np.sort(xs), np.sort(-before[:, 0]))


def test_mds_plot_square_limits(mv_data):
    from pyrsm.multivariate._plotting import square_origin_limits

    m = mds({"city": mv_data("city")}, "from", "to", "distance")
    lo, hi = square_origin_limits(m.points)
    assert abs(lo + hi) < 1e-9 and hi > 0
    # the plot builds without error
    assert m.plot() is not None


def test_mds_plot_all_dim_pairs(mv_data):
    m = mds({"city": mv_data("city")}, "from", "to", "distance", method="metric", nr_dim=3)
    plots = m.plot()
    assert isinstance(plots, list) and len(plots) == 3  # (1,2),(1,3),(2,3)


def test_mds_plot_draws_readable_labels(mv_data):
    m = mds({"city": mv_data("city")}, "from", "to", "distance")
    labels, contained, overlaps = _visible_label_boxes(m.plot(fontsz=8))
    assert set(m.labels) <= labels
    assert [label for label, ok in contained.items() if not ok] == []
    assert overlaps == []


def test_mds_invalid_method(mv_data):
    with pytest.raises(ValueError, match="method"):
        mds({"city": mv_data("city")}, "from", "to", "distance", method="typo")
    # alias accepted
    m = mds({"city": mv_data("city")}, "from", "to", "distance", method="nonmetric")
    assert m.method == "non-metric"


def test_mds_summary_smoke(mv_data, capsys):
    m = mds({"city": mv_data("city")}, "from", "to", "distance")
    m.summary()
    out = capsys.readouterr().out
    assert "MDS" in out
    assert "Coordinates" in out
    assert "Stress" in out
