"""Hierarchical cluster analysis — port of ``radiant.multivariate::hclus``.

For the Radiant default (``distance="sq.euclidian"``, ``method="ward.D"``) this
reproduces R's ``hclust(dist^2, "ward.D")`` exactly: SciPy's Ward linkage
implements ``ward.D2`` on Euclidean distances, and squaring those merge heights
yields the same tree and heights as R's ``ward.D`` on squared distances.

When any selected variable is categorical, Radiant switches the distance metric
to Gower (``cluster::daisy(metric = "gower")``); the same behavior is
reproduced here.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pyrsm.utils import format_nr

from ._plotting import as_plot_result
from ._utils import get_data, get_vars, gower_dist, is_categorical
from pyrsm.data_scope import apply_scope, print_dropped, print_scope
from ._utils import r_scale as _standardize
from pyrsm.basics import display_utils as du

__all__ = ["hclus"]

DISTANCES = (
    "sq.euclidian",
    "binary",
    "canberra",
    "euclidean",
    "gower",
    "manhattan",
    "maximum",
    "minkowski",
)
METHODS = ("ward.D", "single", "complete", "average", "mcquitty", "median", "centroid", "ward.D2")


def _r_canberra(Z: np.ndarray) -> np.ndarray:
    """R's ``dist(method = "canberra")``: terms with 0/0 are dropped and the
    sum is scaled up by ``p / (terms kept)``."""
    n, p = Z.shape
    out = []
    for i in range(n - 1):
        num = np.abs(Z[i] - Z[i + 1 :])
        den = np.abs(Z[i]) + np.abs(Z[i + 1 :])
        keep = den > 0
        terms = np.where(keep, num / np.where(keep, den, 1.0), 0.0)
        cnt = keep.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(cnt > 0, terms.sum(axis=1) * p / cnt, np.nan))
    return np.concatenate(out) if out else np.array([])


def _r_binary(Z: np.ndarray) -> np.ndarray:
    """R's ``dist(method = "binary")``: non-zero values are "on"; the share of
    variables with exactly one "on" among those with at least one."""
    from scipy.spatial.distance import pdist

    B = Z != 0
    either = pdist(B.astype(float), lambda u, v: np.sum((u > 0) | (v > 0)))
    one = pdist(B.astype(float), lambda u, v: np.sum((u > 0) != (v > 0)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(either > 0, one / np.where(either > 0, either, 1.0), 0.0)


def _r_dissimilarity(Z: np.ndarray, distance: str):
    """Return the dissimilarity vector R's ``dist()`` would produce.

    ``sq.euclidian`` -> squared Euclidean (R's ``dist(.)^2``); ``binary`` and
    ``canberra`` follow R's definitions (which differ from SciPy's jaccard and
    canberra); ``maximum`` is Chebyshev and ``minkowski`` uses R's default
    ``p = 2``.
    """
    from scipy.spatial.distance import pdist

    if distance in ("sq.euclidian", "sq.euclidean"):
        return pdist(Z, "euclidean") ** 2
    elif distance in ("euclidean", "minkowski"):
        return pdist(Z, "euclidean")
    elif distance == "manhattan":
        return pdist(Z, "cityblock")
    elif distance == "maximum":
        return pdist(Z, "chebyshev")
    elif distance == "canberra":
        return _r_canberra(Z)
    elif distance in ("binary", "jaccard"):
        return _r_binary(Z)
    raise ValueError(f"Unknown distance '{distance}'. Choose one of {DISTANCES}.")


_R_METHODS = {
    "ward.D": 1,
    "single": 2,
    "complete": 3,
    "average": 4,
    "mcquitty": 5,
    "median": 6,
    "centroid": 7,
    "ward.D2": 8,
}


def _r_hclust(dR: np.ndarray, method: str):
    """Port of R's ``hclust`` (Murtagh's nearest-neighbour-list algorithm in
    ``hclust.f``), including how it breaks ties: the first pair in index order
    with the smallest dissimilarity is merged. SciPy's linkage algorithms break
    ties differently, so with tied distances (e.g. unstandardized rating
    scales) they can build a different tree.

    Returns ``(ia, ib, crit)``: the 0-based representative observations merged
    at each step (``ia < ib``; the merged cluster keeps ``ia``) and the merge
    heights on R's scale.
    """
    from scipy.spatial.distance import squareform

    iopt = _R_METHODS[method]
    is_ward = iopt in (1, 8)
    D = squareform(np.asarray(dR, dtype=float), checks=False)
    if iopt == 8:
        D = D * D
    n = D.shape[0]
    inf = np.inf
    np.fill_diagonal(D, inf)
    membr = np.ones(n)
    flag = np.ones(n, dtype=bool)
    # nearest neighbour of each i to its right (j > i); argmin takes the first
    upper = np.triu(D, 1)
    upper[np.tril_indices(n)] = inf
    nn = np.argmin(upper, axis=1)
    disnn = upper[np.arange(n), nn]
    disnn[n - 1] = inf
    del upper
    idx = np.arange(n)

    ia = np.empty(n - 1, dtype=int)
    ib = np.empty(n - 1, dtype=int)
    crit = np.empty(n - 1)
    for step in range(n - 1):
        cand = np.where(flag[: n - 1], disnn[: n - 1], inf)
        im = int(np.argmin(cand))
        jm = int(nn[im])
        dmin = cand[im]
        i2, j2 = min(im, jm), max(im, jm)
        ia[step], ib[step] = i2, j2
        crit[step] = np.sqrt(dmin) if iopt == 8 else dmin
        flag[j2] = False

        # Lance-Williams update of the dissimilarities to the merged cluster
        k = flag.copy()
        k[i2] = False
        d1, d2, xx = D[i2], D[j2], D[i2, j2]
        mi, mj, mk = membr[i2], membr[j2], membr
        if is_ward:
            new = ((mi + mk) * d1 + (mj + mk) * d2 - mk * xx) / (mi + mj + mk)
        elif iopt == 2:
            new = np.minimum(d1, d2)
        elif iopt == 3:
            new = np.maximum(d1, d2)
        elif iopt == 4:
            new = (mi * d1 + mj * d2) / (mi + mj)
        elif iopt == 5:
            new = (d1 + d2) / 2
        elif iopt == 6:
            new = ((d1 + d2) - xx / 2) / 2
        else:
            new = (mi * d1 + mj * d2 - mi * mj * xx / (mi + mj)) / (mi + mj)
        new = np.where(k, new, inf)
        D[i2, :] = new
        D[:, i2] = new
        D[j2, :] = inf
        D[:, j2] = inf

        right = new[i2 + 1 :]
        if right.size and np.isfinite(right).any():
            jj = int(np.argmin(right)) + i2 + 1
            disnn[i2], nn[i2] = right[jj - i2 - 1], jj
        else:
            disnn[i2] = inf
        # rows to the left of i2 may now have i2 as their nearest neighbour
        left = k & (idx < i2) & (new < disnn)
        disnn[left] = new[left]
        nn[left] = i2
        membr[i2] = mi + mj

        # redetermine the nearest neighbour of rows that pointed at i2 or j2
        redo = np.flatnonzero(flag[: n - 1] & ((nn[: n - 1] == i2) | (nn[: n - 1] == j2)))
        for i in redo:
            row = D[i, i + 1 :]
            if row.size and np.isfinite(row).any():
                j = int(np.argmin(row))
                nn[i], disnn[i] = i + 1 + j, row[j]
            else:
                disnn[i] = inf
    return ia, ib, crit


def _linkage_from_dissim(dR: np.ndarray, method: str):
    """Build a SciPy-style linkage matrix matching R's ``hclust(dR, method)``.

    The merges come from ``_r_hclust`` so the tree, the merge order and the
    heights (on R's scale, in merge order) are R's. Returns
    ``(linkage_matrix, heights, merges)`` where ``merges`` holds the 0-based
    representative observations joined at each step (used by ``cutree``).
    """
    if method not in _R_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose one of {METHODS}.")
    ia, ib, crit = _r_hclust(dR, method)
    n = len(ia) + 1
    cid = np.arange(n)  # current SciPy cluster id of each representative
    size = np.ones(n)
    Z = np.empty((n - 1, 4))
    for s, (i, j) in enumerate(zip(ia, ib)):
        a, b = sorted((cid[i], cid[j]))
        size[i] += size[j]
        Z[s] = (a, b, crit[s], size[i])
        cid[i] = n + s
    return Z, crit.copy(), np.column_stack([ia, ib])


class hclus:
    """Hierarchical cluster analysis.

    Parameters
    ----------
    data, vars : data and variable selection (supports ``"v1:v6"`` ranges).
    labels : str, default "none"
        Column to use as row labels (and excluded from clustering).
    distance : str, default "sq.euclidian"
        ``"sq.euclidian"`` (squared Euclidean), ``"binary"``, ``"canberra"``,
        ``"euclidean"``, ``"gower"``, ``"manhattan"``, ``"maximum"`` or
        ``"minkowski"`` (p = 2), with R's ``dist()`` definitions. When
        categorical variables are present the distance is forced to
        ``"gower"`` (matching Radiant).
    method : str, default "ward.D"
        Linkage method: ``"ward.D"`` (Ward's), ``"single"``, ``"complete"``,
        ``"average"``, ``"mcquitty"``, ``"median"``, ``"centroid"`` (Radiant's
        choices) or ``"ward.D2"``.
    standardize : bool, default True
        Standardize numeric variables (R's ``scale``) before clustering. Only
        numeric variables are standardized; categoricals feed Gower as-is.

    Attributes
    ----------
    linkage : np.ndarray
        SciPy linkage matrix.
    height : np.ndarray
        Merge heights on the Radiant scale (squared for ``sq.euclidian``).
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
        vars,
        labels: str = "none",
        distance: str = "sq.euclidian",
        method: str = "ward.D",
        max_cases: int = 5000,
        standardize: bool = True,
        data_filter: str = "",
        sort: str = "",
        slice: str = "",
    ) -> None:
        if distance not in DISTANCES + ("sq.euclidean", "jaccard"):
            raise ValueError(f"Unknown distance '{distance}'. Choose one of {DISTANCES}.")
        if method not in METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose one of {METHODS}.")
        self.name, self.data = get_data(data)

        # The data scope, applied before anything is measured: filter, then
        # sort, then slice. Kept on the object so ``summary`` can say which
        # rows the numbers describe.
        self.data_filter = data_filter
        self.sort = sort
        self.slice = slice
        self.data = apply_scope(self.data, data_filter, sort, slice)
        self.vars = get_vars(self.data.columns, vars)
        self.labels = labels
        self.method = method
        self.standardize = standardize

        sel = self.vars if labels == "none" else [labels] + self.vars
        sub = self.data.select(sel).drop_nulls()
        # rows with a missing value in any selected variable (listwise, as R)
        self.nobs_dropped = self.data.height - sub.height
        if sub.height > max_cases:
            raise ValueError(
                f"The number of cases to cluster ({sub.height}) exceeds max_cases "
                f"({max_cases})."
            )

        if labels != "none":
            raw_labels = [str(x) for x in sub[labels].to_list()]
            # Radiant uses labels only when unique; otherwise falls back to rows.
            if len(set(raw_labels)) == len(raw_labels):
                self.row_labels = raw_labels
            else:
                print(
                    "** Label variable is not unique; using row numbers instead **"
                )
                self.row_labels = [str(i + 1) for i in range(sub.height)]
            sub = sub.select(self.vars)
        else:
            self.row_labels = [str(i + 1) for i in range(sub.height)]

        cat = [is_categorical(sub[v]) for v in self.vars]
        self.any_categorical = any(cat)
        # Radiant forces Gower distance when categorical variables are present.
        if self.any_categorical and distance != "gower":
            print(
                "** Categorical variables selected; using Gower distance **"
            )
            distance = "gower"
        self.distance = distance

        if distance == "gower":
            dR = gower_dist(sub, self.vars, standardize=standardize)
            self._Z = sub  # keep the (mixed) frame for reference
        else:
            M = sub.to_numpy().astype(float)
            Z = _standardize(M) if standardize else M
            self._Z = Z
            dR = _r_dissimilarity(Z, distance)

        self.linkage, self.height, self._merges = _linkage_from_dissim(dR, method)
        self.nobs = sub.height

    def cutree(self, nr_clus: int) -> np.ndarray:
        """Cluster assignments (1-based) for ``nr_clus`` clusters.

        Like R's ``cutree(hc, k)``: the first ``n - k`` merges are applied (so
        non-monotone centroid/median trees are cut by merge order, not by
        height) and clusters are numbered by first appearance in the data.
        """
        n = self.nobs
        k = int(max(1, min(nr_clus, n)))
        rep = np.arange(n)
        for i, j in self._merges[: n - k]:
            rep[rep == j] = i
        _, first, codes = np.unique(rep, return_index=True, return_inverse=True)
        rank = np.empty(len(first), dtype=int)
        rank[np.argsort(first)] = np.arange(1, len(first) + 1)
        return rank[codes]

    def store(self, data=None, nr_clus: int = 2, name: str | None = None):
        """Append cluster assignments as a categorical column.

        Default column name is ``hclus{nr_clus}`` (matching Radiant).
        """
        if name is None:
            name = f"hclus{nr_clus}"
        target = self.data if data is None else get_data(data)[1]
        assign = self.cutree(nr_clus)
        if target.height != len(assign):
            raise ValueError("Target row count does not match number of cases.")
        return target.with_columns(
            # An Enum, levels in cluster order ("1", "2", ..., "10").
            pl.Series(name, [str(x) for x in assign]).cast(
                pl.Enum([str(x) for x in sorted(set(assign))])
            )
        )

    def summary(self, dec: int = 2, plain: bool = True) -> None:
        """Print the summary. ``plain=False`` in a notebook (or a rendered Quarto report) shows the formatted report Radiant shows."""
        if not plain and du.is_notebook():
            # Radiant's formatted report (pyrsm.report), as its Summary tab shows it
            return du.show_text_report(self, "hclus", dec=dec)

        print("Hierarchical cluster analysis")
        print(f"Data        : {self.name}")
        print_scope(self, 12)
        print(f"Variables   : {', '.join(self.vars)}")
        print(f"Method      : {self.method}")
        print(f"Distance    : {self.distance}")
        print(f"Standardize : {self.standardize}")
        print(f"Observations: {format_nr(self.nobs, dec=0)}")
        print_dropped(self, 12)

    def plot(self, plots="dendro", dec: int = 2, cutoff: float = 0.05):
        """Cluster plots (plotnine, pyrsm plot style).

        ``"dendro"`` draws the dendrogram from SciPy's dendrogram coordinates.
        As in Radiant, a ``cutoff`` above 0 gives a "cutoff dendrogram" that
        shows only the merges with a (normalized) height of at least
        ``cutoff`` and no leaf labels; ``cutoff=0`` shows the full tree with
        a label per case. ``"scree"`` and ``"change"`` plot within-cluster
        heterogeneity vs. number of clusters. Heights are normalized to
        ``[0, 1]`` by ``max(height)`` (matching Radiant).
        """
        if isinstance(plots, str):
            plots = [plots]
        valid = ("dendro", "scree", "change")
        unknown = [p for p in plots if p not in valid]
        if unknown:
            raise ValueError(
                f"Unknown plot type(s) {unknown}. Choose from {list(valid)}."
            )
        out = []
        for ptype in plots:
            if ptype == "dendro":
                out.append(self._dendro(cutoff))
            elif ptype in ("scree", "change"):
                out.append(self._scree_change(ptype, cutoff))
        return as_plot_result(out)

    def _dendro(self, cutoff: float):
        import pandas as pd
        from plotnine import (
            aes,
            coord_cartesian,
            element_blank,
            element_text,
            geom_segment,
            ggplot,
            labs,
            scale_x_continuous,
            scale_y_continuous,
            theme,
        )
        from scipy.cluster.hierarchy import dendrogram

        from pyrsm.plot_style import LINE, theme_pyrsm

        # Heights are already on R's scale; normalize for display.
        lk = self.linkage.copy()
        hmax = lk[:, 2].max()
        if hmax > 0:
            lk[:, 2] = lk[:, 2] / hmax
        dd = dendrogram(lk, labels=self.row_labels, no_plot=True)

        # Each merge is an inverted "U": left leg, crossbar, right leg.
        rows = []
        for xs, ys in zip(dd["icoord"], dd["dcoord"]):
            for k in range(3):
                rows.append((xs[k], ys[k], xs[k + 1], ys[k + 1]))
        segs = pd.DataFrame(rows, columns=["x", "y", "xend", "yend"])

        n = len(dd["ivl"])
        leaf_x = [5 + 10 * i for i in range(n)]
        cutoff = 0.0 if not cutoff else float(cutoff)
        g = (
            ggplot(segs, aes(x="x", y="y", xend="xend", yend="yend"))
            + geom_segment(color=LINE, size=0.7)
            + labs(x="", y="Within-cluster heterogeneity")
            + theme_pyrsm()
            + theme(
                panel_grid_major_x=element_blank(),
                panel_grid_minor_x=element_blank(),
            )
        )
        if cutoff > 0:
            # Radiant: plot(hc, ylim = c(cutoff, 1), leaflab = "none")
            g = (
                g
                + labs(title="Cutoff dendrogram")
                + scale_x_continuous(breaks=[])
                # no expansion: the panel starts exactly at the cutoff
                + coord_cartesian(xlim=(0, 10 * n), ylim=(cutoff, 1.02), expand=False)
                + theme(axis_ticks_major_x=element_blank())
            )
        else:
            size = max(4.0, min(9.0, 360 / max(n, 1)))
            rotate = n > 25
            g = (
                g
                + labs(title="Dendrogram")
                + scale_x_continuous(breaks=leaf_x, labels=list(dd["ivl"]), expand=(0, 5))
                + scale_y_continuous(expand=(0, 0.02))
                + theme(
                    axis_text_x=element_text(
                        size=size,
                        rotation=90 if rotate else 0,
                        va="top",
                        ha="center",
                    )
                )
            )
        return g

    def _scree_change(self, ptype: str, cutoff: float):
        import pandas as pd
        from plotnine import (
            aes,
            geom_col,
            geom_line,
            geom_point,
            ggplot,
            labs,
            scale_x_continuous,
        )

        from pyrsm.plot_style import ALPHA, FILL, LINE, POINT_FILL, theme_pyrsm

        # normalize all merge heights to [0, 1] (Radiant divides by max), then
        # drop the earliest merges below the cutoff so dense trees stay readable.
        h = self.height
        hmax = h.max() if len(h) else 1.0
        hn = h / hmax if hmax > 0 else h
        if cutoff:
            hn = hn[hn > cutoff]
        # reverse so index 1 = largest height (= fewest clusters)
        heights_norm = hn[::-1]
        n = len(heights_norm)
        if ptype == "scree":
            dat = pl.DataFrame(
                {
                    "x": list(range(1, n + 1)),
                    "y": [float(v) for v in heights_norm],
                }
            ).to_pandas()
            return (
                ggplot(dat, aes(x="x", y="y", group=1))
                + geom_line(color=LINE, linetype="dashdot", size=0.7)
                + geom_point(color=LINE, size=4, shape="o", fill=POINT_FILL)
                + labs(
                    title="Screeplot",
                    x="# clusters",
                    y="Within-cluster heterogeneity",
                )
                + scale_x_continuous(breaks=list(range(1, n + 1)))
                + theme_pyrsm()
            )
        else:  # change
            change = np.abs((heights_norm[1:] - heights_norm[:-1]) / heights_norm[:-1])
            labels = [f"{i + 1}-{i + 2}" for i in range(len(change))]
            dat = pl.DataFrame(
                {"nr_clus": labels, "bump": [float(v) for v in change]}
            ).to_pandas()
            # keep the order of the labels ("10-11" must not sort before "2-3")
            dat["nr_clus"] = pd.Categorical(labels, categories=labels, ordered=True)
            return (
                ggplot(dat, aes(x="nr_clus", y="bump"))
                + geom_col(alpha=ALPHA, fill=FILL)
                + labs(
                    title="Change in heterogeneity",
                    x="# clusters",
                    y="Rate of change",
                )
                + theme_pyrsm()
            )
