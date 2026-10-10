"""Measured label placement for the multivariate maps and loadings plots."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pyrsm.multivariate import full_factor, mds, prmap
from pyrsm.multivariate._label_layout import layout_labels
from pyrsm.multivariate._plotting import add_square_axes

DATA = Path(__file__).resolve().parents[1] / "examples" / "data" / "multivariate"


def _rects(lay):
    return np.column_stack(
        [lay.lx - lay.width / 2, lay.lx + lay.width / 2, lay.ly - lay.height / 2, lay.ly + lay.height / 2]
    )


def _overlaps(rects):
    n = len(rects)
    out = []
    for a in range(n):
        for b in range(a + 1, n):
            w = min(rects[a, 1], rects[b, 1]) - max(rects[a, 0], rects[b, 0])
            h = min(rects[a, 3], rects[b, 3]) - max(rects[a, 2], rects[b, 2])
            if w > 1e-9 and h > 1e-9:
                out.append((a, b))
    return out


def _base(lim):
    from plotnine import ggplot

    from pyrsm.plot_style import theme_pyrsm

    return add_square_axes(ggplot(), lim) + theme_pyrsm()


def _layout(x, y, labels, lim=1.05, **kw):
    return layout_labels(_base, x, y, labels, size=10, lim=lim, **kw)


def test_loadings_labels_are_adjacent_inside_and_apart():
    tp = pl.read_parquet(DATA / "toothpaste.parquet")
    ff = full_factor({"tp": tp}, vars=["v1", "v2", "v3", "v4", "v5", "v6"], nr_fact=2)
    x, y = ff.floadings[:, 0], ff.floadings[:, 1]
    lay = _layout(x, y, ff.vars, segments=[(0, 0, a, b) for a, b in zip(x, y)],
                  segment_owner=list(range(6)))
    rects = _rects(lay)
    assert not _overlaps(rects)
    assert not lay.leader.any()  # room for every label next to its point
    # the label box is right next to its point: within one label height
    gap_x = np.maximum.reduce([rects[:, 0] - x, x - rects[:, 1], np.zeros(6)])
    gap_y = np.maximum.reduce([rects[:, 2] - y, y - rects[:, 3], np.zeros(6)])
    assert np.all(np.hypot(gap_x, gap_y) < lay.height)
    # the axes stay at Radiant's +/-1 range
    assert lay.lim == pytest.approx(1.05)


def test_crowded_points_get_every_label_and_no_overlap():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.05, 12)
    y = rng.normal(0, 0.05, 12)
    labels = [f"attribute_{k}" for k in range(12)]
    lay = _layout(x, y, labels, lim=0.3)
    assert len(lay.lx) == 12
    assert not _overlaps(_rects(lay))
    assert lay.leader.any()  # some labels needed a leader line
    # every label inside the (possibly widened) panel
    assert np.all(np.abs(_rects(lay)) <= lay.lim * 1.1 + 1e-9)


def test_layout_is_reproducible():
    x = np.array([0.5, 0.52, -0.3, 0.1])
    y = np.array([0.2, 0.22, -0.6, 0.9])
    a = _layout(x, y, ["one", "two", "three", "four"])
    b = _layout(x, y, ["one", "two", "three", "four"])
    assert np.allclose(a.lx, b.lx) and np.allclose(a.ly, b.ly)


def test_maps_draw_every_label_once():
    tp = pl.read_parquet(DATA / "tpbrands.parquet")
    m = mds({"tp": tp}, id1="id1", id2="id2", dis="dissimilarity")
    g = m.plot()
    texts = [layer for layer in g.layers if type(layer.geom).__name__ == "geom_text"]
    assert len(texts) == 1 and len(texts[0]._data) == 10
    rt = pl.read_parquet(DATA / "retailers.parquet")
    pm = prmap({"rt": rt}, brand="retailer",
               attr=["good_value", "quality_prod", "service", "convenience"],
               pref=["segment1", "segment2"])
    g = pm.plot(plots=["brand", "attr", "pref"])
    texts = [layer for layer in g.layers if type(layer.geom).__name__ == "geom_text"]
    assert len(texts[0]._data) == 6 + 4 + 2


def test_layout_works_with_a_backend_without_a_renderer():
    """Server processes may run a backend whose canvas has no renderer."""
    import os
    import subprocess
    import sys

    code = (
        "import polars as pl, pyrsm as rsm\n"
        f"tp = pl.read_parquet({str(DATA / 'toothpaste.parquet')!r})\n"
        "ff = rsm.multivariate.full_factor({'tp': tp}, vars=['v1','v2','v3','v4','v5','v6'], nr_fact=2)\n"
        "ff.plot(plots=['resp', 'attr'])\n"
        "print('ok')\n"
    )
    env = {**os.environ, "MPLBACKEND": "template"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.stdout.strip().endswith("ok"), out.stderr[-500:]
