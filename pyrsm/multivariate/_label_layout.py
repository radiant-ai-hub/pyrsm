"""Measured label placement for the multivariate maps and loadings plots.

plotnine has no ``geom_text_repel``. Repelling labels with forces (adjustText)
works on estimated text sizes and tends to push labels far from the point they
belong to. Here the plot is drawn once to *measure* the panel (inches, axis
range) and every label (its width and height in the plot's own font), and the
labels are then placed by search:

* each label gets candidate positions all around its point (16 directions),
  touching the point with a small gap, plus rings further out that need a
  leader line;
* a position costs more when it overlaps another label (heavily), covers a
  point (another label's point heavily, background points such as respondents
  lightly), crosses an arrow or loading line, leaves the panel, points away
  from the preferred side (outward from the origin), or needs a leader line;
* positions are chosen label by label and improved until nothing changes, from
  several seeded starts, keeping the cheapest layout;
* if a label still does not fit, the axes are widened and the layout repeated,
  so every label is shown in full.

All coordinates are data units on square, origin-centered axes (``coord_fixed``
with limits ``-lim..lim``), so one data unit has the same length on both axes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LabelLayout", "layout_labels", "measure_labels"]

#: Breathing room between a point and its label, and around each label (points)
GAP_PT = 2.0
PAD_PT = 1.0
#: Overlapping another label is the worst outcome; a label outside the panel is
#: next (the axes are then widened so it shows in full).
OVERLAP = 60.0
OUTSIDE = 20.0


@dataclass
class LabelLayout:
    """Where to draw each label (centers) and whether it needs a leader line."""

    x: np.ndarray
    y: np.ndarray
    lx: np.ndarray
    ly: np.ndarray
    leader: np.ndarray
    lim: float
    width: np.ndarray
    height: np.ndarray

    def leader_ends(self):
        """Leader line from each point to the nearest edge of its label."""
        hw, hh = self.width / 2, self.height / 2
        ex = np.clip(self.x, self.lx - hw, self.lx + hw)
        ey = np.clip(self.y, self.ly - hh, self.ly + hh)
        return ex, ey


def measure_labels(base, labels, size: float):
    """Draw ``base`` with the labels added and measure panel and text in inches.

    Returns ``(panel_w, panel_h, (xmin, xmax), (ymin, ymax), sizes)`` where
    ``sizes`` is an ``n x 2`` array of label width and height in inches.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from plotnine import aes, geom_text

    labels = [str(s) for s in labels]
    probe = pd.DataFrame({"x": [0.0] * len(labels), "y": [0.0] * len(labels), "label": labels})
    g = base + geom_text(probe, aes(x="x", y="y", label="label"), size=size)
    fig = g.draw()
    try:
        # Measure with an Agg canvas whatever the active backend is (a server
        # process may run a backend whose canvas has no renderer).
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        canvas = fig.canvas if isinstance(fig.canvas, FigureCanvasAgg) else FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        ax = fig.axes[0]
        dpi = fig.dpi
        bb = ax.get_window_extent(renderer)
        texts = ax.texts[-len(labels):] if labels else []
        sizes = np.array(
            [
                [e.width / dpi, e.height / dpi]
                for e in (t.get_window_extent(renderer) for t in texts)
            ]
        ).reshape(-1, 2)
        return bb.width / dpi, bb.height / dpi, ax.get_xlim(), ax.get_ylim(), sizes
    finally:
        plt.close(fig)


def _candidates(w, h, gap, step, rings=4, dirs=16):
    """Candidate offsets (center minus point) for one w x h label.

    Ring 0 touches the point (the box edge ``gap`` away); ring k moves the box
    k * ``step`` further out along the same direction.
    """
    ang = np.arange(dirs) * (2 * np.pi / dirs)
    ux, uy = np.cos(ang), np.sin(ang)
    with np.errstate(divide="ignore"):
        tx = np.where(np.abs(ux) > 1e-9, (w / 2) / np.abs(ux), np.inf)
        ty = np.where(np.abs(uy) > 1e-9, (h / 2) / np.abs(uy), np.inf)
    t0 = np.minimum(tx, ty) + gap
    offs, ring, theta = [], [], []
    for k in range(rings):
        t = t0 + k * step
        offs.append(np.column_stack([ux * t, uy * t]))
        ring.append(np.full(dirs, k))
        theta.append(ang)
    return np.vstack(offs), np.concatenate(ring), np.concatenate(theta)


def _overlap(rects, others):
    """Overlap areas between each rect (C x 4) and each other rect (M x 4)."""
    if others.size == 0:
        return np.zeros((rects.shape[0], 0))
    x0 = np.maximum(rects[:, None, 0], others[None, :, 0])
    x1 = np.minimum(rects[:, None, 1], others[None, :, 1])
    y0 = np.maximum(rects[:, None, 2], others[None, :, 2])
    y1 = np.minimum(rects[:, None, 3], others[None, :, 3])
    return np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)


def _segment_cut(rects, segs):
    """Length of each segment inside each rect (Liang-Barsky), summed (C,)."""
    total = np.zeros(rects.shape[0])
    for (sx0, sy0, sx1, sy1) in segs:
        dx, dy = sx1 - sx0, sy1 - sy0
        t0 = np.zeros(rects.shape[0])
        t1 = np.ones(rects.shape[0])
        ok = np.ones(rects.shape[0], dtype=bool)
        for p, q in (
            (-dx, sx0 - rects[:, 0]),
            (dx, rects[:, 1] - sx0),
            (-dy, sy0 - rects[:, 2]),
            (dy, rects[:, 3] - sy0),
        ):
            if abs(p) < 1e-12:
                ok &= q >= 0
                continue
            r = q / p
            if p < 0:
                t0 = np.maximum(t0, r)
            else:
                t1 = np.minimum(t1, r)
        inside = ok & (t1 > t0)
        total += np.where(inside, (t1 - t0) * np.hypot(dx, dy), 0.0)
    return total


def _solve(P, W, H, *, prefer, bound, gap, step, obstacles, segs, own_segs, seed, restarts=6,
           marks=None, rings=6):
    """Choose one candidate per label; returns centers (n x 2) and ring (n,)."""
    n = P.shape[0]
    cands, unary = [], []
    for i in range(n):
        off, ring, theta = _candidates(W[i], H[i], gap, step, rings=rings)
        c = P[i] + off
        rect = np.column_stack(
            [c[:, 0] - W[i] / 2, c[:, 0] + W[i] / 2, c[:, 1] - H[i] / 2, c[:, 1] + H[i] / 2]
        )
        area = W[i] * H[i]
        # outside the panel: the overhang, relative to the label's size
        over = (
            np.clip(-bound - rect[:, 0], 0, None) + np.clip(rect[:, 1] - bound, 0, None)
        ) / W[i] + (
            np.clip(-bound - rect[:, 2], 0, None) + np.clip(rect[:, 3] - bound, 0, None)
        ) / H[i]
        # points covered: every label's point (heavily), background points (lightly)
        cost = OUTSIDE * over
        if obstacles is not None and len(obstacles):
            pts, wts = obstacles
            inside = (
                (pts[None, :, 0] >= rect[:, None, 0])
                & (pts[None, :, 0] <= rect[:, None, 1])
                & (pts[None, :, 1] >= rect[:, None, 2])
                & (pts[None, :, 1] <= rect[:, None, 3])
            )
            cost = cost + (inside * wts[None, :]).sum(axis=1)
        if marks is not None and len(marks[0]):
            mp, mown = marks
            keep = mown != i
            if keep.any():
                mpk = mp[keep]
                inside = (
                    (mpk[None, :, 0] >= rect[:, None, 0])
                    & (mpk[None, :, 0] <= rect[:, None, 1])
                    & (mpk[None, :, 1] >= rect[:, None, 2])
                    & (mpk[None, :, 1] <= rect[:, None, 3])
                )
                cost = cost + 8.0 * inside.sum(axis=1)
        others = [k for k in range(n) if k != i]
        if others:
            inside = (
                (P[None, others, 0] >= rect[:, None, 0])
                & (P[None, others, 0] <= rect[:, None, 1])
                & (P[None, others, 1] >= rect[:, None, 2])
                & (P[None, others, 1] <= rect[:, None, 3])
            )
            cost = cost + 8.0 * inside.sum(axis=1)
        # lines crossed (not the label's own line), relative to the label's size
        other_segs = [s for k, s in enumerate(segs) if own_segs[k] != i]
        if other_segs:
            cost = cost + 4.0 * _segment_cut(rect, other_segs) / (W[i] + H[i])
        # side: outward along the preferred direction; further rings need a leader
        cost = cost + 1.0 * (1 - np.cos(theta - prefer[i])) / 2 + 2.0 * np.sqrt(ring)
        cands.append((c, rect, ring, area))
        unary.append(cost)

    def pair_cost(i, rect_i, area_i, choice):
        tot = np.zeros(rect_i.shape[0])
        for k in range(n):
            if k == i:
                continue
            rk = cands[k][1][choice[k]][None, :]
            ov = _overlap(rect_i, rk)[:, 0]
            tot += OVERLAP * ((ov > 1e-12) + ov / min(area_i, cands[k][3]))
        return tot

    def total(choice):
        t = sum(unary[i][choice[i]] for i in range(n))
        for i in range(n):
            for k in range(i + 1, n):
                ri = cands[i][1][choice[i]][None, :]
                rk = cands[k][1][choice[k]][None, :]
                ov = _overlap(ri, rk)[0, 0]
                t += OVERLAP * ((ov > 1e-12) + ov / min(cands[i][3], cands[k][3]))
        return t

    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    for start in range(restarts):
        if start == 0:
            choice = np.array([int(np.argmin(u)) for u in unary])
        else:
            # a random good start: one of each label's cheapest candidates
            choice = np.array([int(rng.choice(np.argsort(u)[:6])) for u in unary])
        for _ in range(60):
            changed = False
            for i in rng.permutation(n):
                c_i, rect_i, _, area_i = cands[i]
                cost = unary[i] + pair_cost(i, rect_i, area_i, choice)
                pick = int(np.argmin(cost))
                if cost[pick] < cost[choice[i]] - 1e-9:
                    choice[i] = pick
                    changed = True
            if not changed:
                break
        c = total(choice)
        if c < best_cost - 1e-9:
            best, best_cost = choice.copy(), c
    centers = np.array([cands[i][0][best[i]] for i in range(n)])
    rings = np.array([cands[i][2][best[i]] for i in range(n)])
    rects = np.array([cands[i][1][best[i]] for i in range(n)])
    return centers, rings, rects


def _any_overlap(rects) -> bool:
    if len(rects) < 2:
        return False
    ov = _overlap(rects, rects)
    np.fill_diagonal(ov, 0)
    return bool((ov > 1e-12).any())


def layout_labels(
    base_for_lim,
    x,
    y,
    labels,
    *,
    size: float,
    lim: float,
    prefer=None,
    background=None,
    segments=None,
    segment_owner=None,
    marks=None,
    mark_owner=None,
    seed: int = 1234,
    max_lim_factor: float = 2.0,
) -> LabelLayout:
    """Place ``labels`` next to the points ``(x, y)``.

    Parameters
    ----------
    base_for_lim : callable
        ``lim -> ggplot`` without the labels (same theme, titles and axes as
        the final plot), used once to measure the panel and the text.
    size : float
        Text size in points (``geom_text(size=...)``).
    lim : float
        Requested half-range of the square axes; widened when labels need room.
    prefer : array, optional
        Preferred direction (radians) for each label; default outward from
        the origin.
    background : array (m x 2), optional
        Points labels should avoid covering (e.g., respondents), lightly.
    segments : list of (x0, y0, x1, y1), optional
        Lines labels should not cross (arrows, loadings). ``segment_owner``
        gives the label index each line belongs to (or -1); a label may touch
        its own line.
    marks : array (k x 2), optional
        Other marks labels must not cover, such as arrowheads; ``mark_owner``
        gives the label each belongs to (or -1).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    P = np.column_stack([x, y])
    labels = [str(s) for s in labels]
    n = len(labels)
    if prefer is None:
        prefer = np.where(np.hypot(x, y) > 1e-9, np.arctan2(y, x), np.pi / 4)
    segs = list(segments or [])
    owner = list(segment_owner) if segment_owner is not None else [-1] * len(segs)
    mk = None
    if marks is not None and len(marks):
        mk = (
            np.asarray(marks, dtype=float).reshape(-1, 2),
            np.asarray(mark_owner if mark_owner is not None else [-1] * len(marks)),
        )
    bg = None
    if background is not None and len(background):
        bgp = np.asarray(background, dtype=float)
        bg = (bgp, np.full(bgp.shape[0], 0.6))

    panel_w, _, xlim, _, sizes = measure_labels(base_for_lim(lim), labels, size)
    # visible range per requested lim (plotnine pads the limits)
    expand = (xlim[1] - xlim[0]) / (2 * lim)
    pt = 1 / 72  # inches per point
    cur = lim
    for _ in range(6):
        span = 2 * cur * expand
        per_in = span / panel_w  # data units per inch (square panel)
        W = (sizes[:, 0] + 2 * PAD_PT * pt) * per_in
        H = (sizes[:, 1] + 2 * PAD_PT * pt) * per_in
        gap = GAP_PT * pt * per_in
        bound = cur * expand - gap
        # more and wider rings until no two labels overlap (crowded points)
        for nrings in (6, 12, 24):
            centers, rings, rects = _solve(
                P, W, H, prefer=np.asarray(prefer, dtype=float), bound=bound, gap=gap,
                step=float(np.median(H)) * 1.2, obstacles=bg, segs=segs, own_segs=owner,
                seed=seed, marks=mk, rings=nrings,
            )
            if not _any_overlap(rects):
                break
        need = max(
            float(np.max(np.abs(rects[:, :2]))) if n else 0.0,
            float(np.max(np.abs(rects[:, 2:]))) if n else 0.0,
        )
        if need <= bound * 1.0001 or cur >= lim * max_lim_factor:
            break
        # widen the axes so the labels fit (label sizes grow with the range too)
        cur = min(lim * max_lim_factor, cur * max(need / bound, 1.03))
    return LabelLayout(
        x=x, y=y, lx=centers[:, 0], ly=centers[:, 1], leader=rings > 0,
        lim=cur, width=W, height=H,
    )
