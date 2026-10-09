"""Label-placement helper — a lightweight ``ggrepel`` substitute.

plotnine has no ``geom_text_repel``, so the perceptual-map and factor plots
would otherwise stack labels on top of their points (the fixed-``nudge_y``
problem called out in the handoff). This module provides a small force-directed
placement routine that nudges labels apart and away from their anchor points
while keeping each label visually attached to its point/arrow endpoint.

The output is deterministic for a given ``seed`` so plots are reproducible.
"""

from __future__ import annotations

import numpy as np

from pyrsm.plot_style import MUTED

__all__ = ["label_positions", "arrow_label_positions", "text_repel_kwargs"]


def _seeded_rng(seed: int):
    return np.random.default_rng(seed)


def label_positions(
    points,
    lim: float | None = None,
    seed: int = 1234,
    iterations: int = 200,
    k_repel: float = 0.012,
    k_anchor: float = 0.008,
    margin: float = 0.08,
):
    """Return repelled label positions (n x 2) for a set of anchor points.

    Parameters
    ----------
    points : array-like (n x 2)
        Anchor coordinates (the points/arrow endpoints labels belong to).
    lim : float, optional
        Plot half-extent; used to scale the repulsion forces. Defaults to the
        max absolute coordinate.
    seed : int
        Seed for the tiny random initial offset (so labels reproducibly avoid
        perfectly overlapping start positions).

    The labels stay close to their anchors (a spring pulls them back) while a
    repulsion force separates labels that would otherwise overlap.
    """
    P = np.asarray(points, dtype=float)
    n = P.shape[0]
    if n == 0:
        return P.copy()
    if lim is None:
        lim = float(np.max(np.abs(P))) or 1.0
    lim = float(lim) or 1.0
    edge = max(lim * (1 - margin), lim * 0.5)

    rng = _seeded_rng(seed)
    # start labels slightly above-right of their points, with a tiny jitter
    offset = lim * 0.05
    L = P + offset + rng.uniform(-offset / 2, offset / 2, size=P.shape)
    L = np.clip(L, -edge, edge)

    rep = k_repel * lim * lim
    for _ in range(iterations):
        disp = np.zeros_like(L)
        # label-label repulsion
        for i in range(n):
            d = L[i] - L
            dist2 = (d**2).sum(axis=1)
            dist2[i] = np.inf
            f = rep / np.maximum(dist2, (lim * 0.02) ** 2)
            disp[i] += (d * f[:, None]).sum(axis=0)
        # label-point repulsion (so labels don't sit on other points)
        for i in range(n):
            d = L[i] - P
            dist2 = (d**2).sum(axis=1)
            dist2[i] = np.inf
            f = 0.5 * rep / np.maximum(dist2, (lim * 0.02) ** 2)
            disp[i] += (d * f[:, None]).sum(axis=0)
        # spring back to the label's own anchor
        disp += -k_anchor * (L - P)
        # damped step
        step = np.clip(disp, -lim * 0.05, lim * 0.05)
        L = L + step
        L = np.clip(L, -edge, edge)
    return L


def arrow_label_positions(x, y, scale: float = 1.08):
    """Place labels just beyond arrow endpoints (so they clear the arrowhead).

    Returns ``(lx, ly)`` arrays: each endpoint pushed radially outward from the
    origin by ``scale``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return x * scale, y * scale


def loading_label_positions(x, y, labels, lim: float = 1.0, size: float = 10):
    """Label positions just beyond each loading's endpoint, away from the origin.

    The label's center sits past the point by a small gap plus about half the
    label's width, so the text starts right next to its point and on the
    outside of the arrow. Overlaps are left to a light repel. Widths assume
    text of ``size`` points on pyrsm's default figure (a panel of about four
    inches spanning ``2 * lim``).
    """
    char = 0.0021 * size  # half-width of one character, share of ``lim``
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.hypot(x, y)
    r = np.where(r > 0, r, 1.0)
    ux, uy = x / r, y / r
    half_w = lim * char * np.array([len(str(lab)) for lab in labels], dtype=float)
    half_h = lim * 0.004 * size
    gap = lim * 0.03
    # A loading that points sideways (within 60 degrees of horizontal) gets its
    # label beside it, clear of the point; one that points up or down gets it
    # above or below, centered on the direction.
    side = np.abs(ux) >= 0.5
    dx = np.where(side, np.sign(ux) * (gap + half_w), ux * half_w)
    dy = np.where(side, uy * half_h, np.sign(uy) * (gap + half_h))
    return np.column_stack([x + dx, y + dy])


def text_repel_kwargs(
    anchors,
    arrow_color: str = MUTED,
    arrow_lw: float = 0.4,
    iter_lim: int = 800,
    tight: bool = False,
):
    """Keyword arguments for plotnine's ``geom_text(adjust_text=...)``.

    ``adjustText`` (used by plotnine's geom_text) is the equivalent of the behavior we
    relied on from R's ``ggrepel``. Passing the anchor coordinates as both
    avoidance points and arrow targets keeps labels visible inside the panel and
    connects them back to the correct point/arrow endpoint.
    """
    try:
        import adjustText  # noqa: F401
    except ImportError:
        return {}

    P = np.asarray(anchors, dtype=float)
    if tight:
        # Labels that start next to their point: only separate the ones that
        # overlap and keep them close (a strong pull back, little push apart).
        opts = {
            "force_text": (0.4, 0.5),
            "force_static": (0.1, 0.15),
            "force_pull": (0.15, 0.15),
            "force_explode": (0.1, 0.2),
            "expand": (1.15, 1.3),
            "max_move": (20, 20),
            # nudge up/down only: labels stay beside their point and inside
            "only_move": {"text": "y", "static": "y", "explode": "y", "pull": "y"},
            "ensure_inside_axes": True,
            "prevent_crossings": True,
            "iter_lim": iter_lim,
            "min_arrow_len": 8,
            "arrowprops": {"arrowstyle": "-", "color": arrow_color, "linewidth": arrow_lw},
        }
        if P.size:
            opts["x"] = P[:, 0]
            opts["y"] = P[:, 1]
            opts["target_x"] = P[:, 0]
            opts["target_y"] = P[:, 1]
        return {"adjust_text": opts}
    opts = {
        "force_text": (0.45, 0.70),
        "force_static": (0.15, 0.25),
        "force_pull": (0.01, 0.01),
        "force_explode": (0.35, 0.70),
        "expand": (1.45, 1.70),
        "max_move": (40, 40),
        "ensure_inside_axes": True,
        "prevent_crossings": True,
        "iter_lim": iter_lim,
        "min_arrow_len": 3,
        "arrowprops": {
            "arrowstyle": "-",
            "color": arrow_color,
            "linewidth": arrow_lw,
        },
    }
    if P.size:
        opts["x"] = P[:, 0]
        opts["y"] = P[:, 1]
        opts["target_x"] = P[:, 0]
        opts["target_y"] = P[:, 1]
    return {"adjust_text": opts}
