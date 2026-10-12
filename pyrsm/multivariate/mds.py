"""(Dis)similarity based brand maps (MDS) — port of ``radiant.multivariate::mds``.

Builds a symmetric dissimilarity matrix from a long-format
(id1, id2, dissimilarity) table and performs metric (classical, ``cmdscale``) or
non-metric (``MASS::isoMDS``-style Kruskal) multidimensional scaling.

Metric MDS reproduces R's ``cmdscale`` up to sign. Non-metric MDS uses SMACOF
with monotone (isotonic) disparities initialized from the classical solution; it
targets ``MASS::isoMDS`` and matches its Kruskal stress-1 closely (typically to
< 1e-2), though the coordinates can differ by a rotation/reflection because the
two optimizers can settle in different but equivalent minima.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pyrsm.utils import format_nr

from ._label_layout import layout_labels
from ._plotting import as_plot_result, symmetric_limit
from ._utils import apply_filter, get_data
from pyrsm.data_scope import apply_scope, print_scope
from pyrsm.basics import display_utils as du

__all__ = ["mds"]


def _build_dist_matrix(id1, id2, dis):
    """Reproduce radiant's lower-triangle (column-major) distance assembly."""
    lab = []
    seen = set()
    for v in list(id1) + list(id2):
        if v not in seen:
            seen.add(v)
            lab.append(v)
    n = len(lab)
    pos = {v: i for i, v in enumerate(lab)}
    d = np.asarray(dis, dtype=float)
    nobs = len(d)
    lower = n * (n - 1) // 2
    if nobs != lower and nobs != lower + n:
        raise ValueError(
            "Number of observations and unique IDs for the brand variable do "
            "not match. Please choose other ID variables or another dataset."
        )
    D = np.zeros((n, n))
    for a, b, val in zip(id1, id2, d):
        i, j = pos[a], pos[b]
        D[i, j] = D[j, i] = val
    return D, [str(x) for x in lab]


class mds:
    """(Dis)similarity-based multidimensional scaling.

    Parameters
    ----------
    data : dataset (or ``{name: df}``) in long (id1, id2, dissimilarity) form.
    id1, id2 : str
        Object identifier columns.
    dis : str
        Dissimilarity / distance column.
    method : str, default "metric"
        ``"metric"`` (classical MDS / cmdscale) or ``"non-metric"`` (SMACOF;
        match to ``MASS::isoMDS``).
    nr_dim : int, default 2
    seed : int, default 1234

    Attributes
    ----------
    points : np.ndarray
        Object coordinates (n x nr_dim).
    stress : float
    labels : list[str]
    dis_mat : np.ndarray
        Symmetric dissimilarity matrix used.
    data_filter : str, default ""
        Expression used to filter the data before the analysis, e.g.,
        "price > 5000 and cut in ['Ideal', 'Premium']". A bare name is a column
        and a call is an operation on one (``is_null(price)``). See
        ``pyrsm.data_scope`` for the full list.
    sort : str, default ""
        Column(s) to sort on, "-" in front for descending, e.g.,
        "clarity -price". Applied after the filter.
    slice : str, default ""
        Rows to keep, counting from 1, e.g., "1:100". Applied after the filter
        and the sort.
    """

    def __init__(
        self,
        data,
        id1: str,
        id2: str,
        dis: str,
        method: str = "metric",
        nr_dim: int = 2,
        seed: int = 1234,
        data_filter: str = "",
        sort: str = "",
        slice: str = "",
    ) -> None:
        _method_aliases = {
            "metric": "metric",
            "non-metric": "non-metric",
            "nonmetric": "non-metric",
            "non_metric": "non-metric",
        }
        key = str(method).strip().lower()
        if key not in _method_aliases:
            raise ValueError(
                f"Unknown method '{method}'. Use 'metric' or 'non-metric'."
            )
        method = _method_aliases[key]
        self.name, self.data = get_data(data)

        # The data scope, applied before anything is measured: filter, then
        # sort, then slice. Kept on the object so ``summary`` can say which
        # rows the numbers describe.
        self.data_filter = data_filter
        self.sort = sort
        self.slice = slice
        self.data = apply_scope(self.data, data_filter, sort, slice)
        self.id1, self.id2, self.dis = id1, id2, dis
        self.method = method
        self.nr_dim = int(nr_dim)
        self.seed = seed

        sub = self.data.select([id1, id2, dis]).drop_nulls()
        D, self.labels = _build_dist_matrix(
            sub[id1].cast(pl.Utf8).to_list(),
            sub[id2].cast(pl.Utf8).to_list(),
            sub[dis].to_numpy(),
        )
        self.dis_mat = D
        self.nobs = len(sub)

        if method == "metric":
            self.points = self._cmdscale(D, self.nr_dim)
            self.stress = self._metric_stress(self.points, D)
        else:
            self.points, self.stress = self._isomds(D, self.nr_dim, seed)

    @staticmethod
    def _cmdscale(D: np.ndarray, k: int) -> np.ndarray:
        n = D.shape[0]
        D2 = D**2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J
        vals, vecs = np.linalg.eigh(B)
        order = np.argsort(vals)[::-1]
        vals = vals[order][:k]
        vecs = vecs[:, order][:, :k]
        pos = np.clip(vals, 0, None)
        return vecs * np.sqrt(pos)

    @staticmethod
    def _metric_stress(points: np.ndarray, D: np.ndarray) -> float:
        """R's cmdscale stress: sqrt(sum((d_hat - d)^2) / sum(d^2))."""
        from scipy.spatial.distance import squareform

        rec = squareform(
            np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)),
            checks=False,
        )
        orig = squareform(D, checks=False)
        return float(np.sqrt(((rec - orig) ** 2).sum() / (orig**2).sum()))

    def _isomds(self, D: np.ndarray, k: int, seed: int):
        import warnings

        from sklearn.manifold import smacof

        init = self._cmdscale(D, k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            points, _ = smacof(
                D,
                metric=False,
                n_components=k,
                init=init,
                n_init=1,
                max_iter=1000,
                eps=1e-10,
                random_state=seed,
                normalized_stress=False,
            )
        return points, self._kruskal_stress(points, D)

    @staticmethod
    def _kruskal_stress(points: np.ndarray, D: np.ndarray) -> float:
        """Kruskal stress-1 with monotone (isotonic) disparities (isoMDS)."""
        from scipy.spatial.distance import squareform
        from sklearn.isotonic import IsotonicRegression

        rec = squareform(
            np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)),
            checks=False,
        )
        orig = squareform(D, checks=False)
        order = np.argsort(orig)
        dhat = np.empty_like(rec)
        dhat[order] = IsotonicRegression().fit_transform(
            np.arange(len(rec)), rec[order]
        )
        return float(np.sqrt(((rec - dhat) ** 2).sum() / (rec**2).sum()))

    def recovered_dist(self) -> np.ndarray:
        """Recovered distance matrix from the coordinates."""
        return np.sqrt(
            ((self.points[:, None, :] - self.points[None, :, :]) ** 2).sum(-1)
        )

    def flip(self, dims) -> None:
        """Flip the sign of one or more axes (1-based dimension indices)."""
        if isinstance(dims, int):
            dims = [dims]
        for d in dims:
            self.points[:, d - 1] *= -1

    def coordinates(self, dec: int | None = None) -> pl.DataFrame:
        cols = {
            f"Dim{j + 1}": [round(float(x), dec) if dec else float(x) for x in self.points[:, j]]
            for j in range(self.nr_dim)
        }
        return pl.DataFrame({"label": self.labels, **cols})

    def summary(self, dec: int = 2, plain: bool = True) -> None:
        """Print the summary. ``plain=False`` in a notebook (or a rendered Quarto report) shows the formatted report Radiant shows."""
        if not plain and du.is_notebook():
            # Radiant's formatted report (pyrsm.report), as its Summary tab shows it
            return du.show_text_report(self, "mds", dec=dec)

        print("(Dis)similarity based brand map (MDS)")
        print(f"Data        : {self.name}")
        print_scope(self, 12)
        print(f"Variables   : {self.id1}, {self.id2}, {self.dis}")
        print(f"Dimensions  : {self.nr_dim}")
        print(f"Method      : {'Non-metric' if self.method == 'non-metric' else 'Metric'}")
        print(f"Observations: {format_nr(self.nobs, dec=0)}")

        print("\nOriginal distance data:")
        print(self._labeled_matrix(self.dis_mat, dec))
        print("\nRecovered distance data:")
        print(self._labeled_matrix(self.recovered_dist(), dec))
        print("\nCoordinates:")
        print(self.coordinates(dec))
        print(f"\nStress: {round(self.stress, dec + 1)}")

    def _labeled_matrix(self, M, dec):
        return pl.DataFrame(
            {"label": self.labels, **{self.labels[j]: [round(float(x), dec) for x in M[:, j]] for j in range(len(self.labels))}}
        )

    def plot(
        self,
        rev_dim=None,
        fontsz: int = 5,
        seed: int = 1234,
        dim=None,
        custom: bool = False,
    ):
        """Brand-map scatter for every dimension pair (plotnine).

        Uses square, origin-centered axes and repelled labels. ``rev_dim`` flips
        the rendered coordinates for the listed (1-based) dimensions without
        mutating the stored coordinates. Returns a single plot for ``nr_dim==2``
        or a list of plots (one per dimension pair) otherwise.
        """
        pts = self.points.copy()
        if rev_dim is not None:
            if isinstance(rev_dim, int):
                rev_dim = [rev_dim]
            for d in rev_dim:
                pts[:, d - 1] *= -1

        lim = symmetric_limit(pts, pad=1.05)
        out = []
        pairs = [(dim[0] - 1, dim[1] - 1)] if dim else [
            (i, j)
            for i in range(self.nr_dim - 1)
            for j in range(i + 1, self.nr_dim)
        ]
        for i, j in pairs:
            out.append(self._plot_pair(pts, i, j, lim, fontsz, seed))
        return as_plot_result(out)

    def _plot_pair(self, pts, i, j, lim, fontsz, seed):
        from plotnine import (
            aes,
            geom_point,
            geom_segment,
            geom_text,
            ggplot,
            labs,
        )

        from pyrsm.plot_style import LINE, MUTED, theme_pyrsm

        from ._plotting import add_square_axes

        x, y = pts[:, i], pts[:, j]
        dat = pl.DataFrame({"x": x, "y": y}).to_pandas()

        def base(lim_):
            g = ggplot(dat) + geom_point(aes(x="x", y="y"), color=LINE)
            g = add_square_axes(g, lim_)
            return g + labs(
                title="Brand map", x=f"Dimension {i + 1}", y=f"Dimension {j + 1}"
            ) + theme_pyrsm()

        lay = layout_labels(base, x, y, self.labels, size=fontsz * 2, lim=lim, seed=seed)
        g = base(lay.lim)
        if lay.leader.any():
            ex, ey = lay.leader_ends()
            ldat = pl.DataFrame(
                {"x": lay.x[lay.leader], "y": lay.y[lay.leader],
                 "xend": ex[lay.leader], "yend": ey[lay.leader]}
            ).to_pandas()
            g = g + geom_segment(ldat, aes(x="x", y="y", xend="xend", yend="yend"),
                                 color=MUTED, size=0.3)
        tdat = pl.DataFrame({"x": lay.lx, "y": lay.ly, "label": self.labels}).to_pandas()
        return g + geom_text(tdat, aes(x="x", y="y", label="label"), size=fontsz * 2)
