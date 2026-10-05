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
from pyrsm.data_scope import apply_scope, print_scope
from ._utils import standardize as _standardize

__all__ = ["hclus"]

# SciPy linkage methods that match R's hclust Lance-Williams updates directly.
_SCIPY_METHODS = {
    "single": "single",
    "complete": "complete",
    "average": "average",
    "ward.D": "ward",
    "ward.D2": "ward",
    "centroid": "centroid",
    "median": "median",
}


def _r_dissimilarity(Z: np.ndarray, distance: str):
    """Return the dissimilarity vector R's ``dist()`` would produce.

    ``sq.euclidian`` -> squared Euclidean (R's ``dist(.)^2``); other metrics map
    to their SciPy equivalents.
    """
    from scipy.spatial.distance import pdist

    if distance in ("sq.euclidian", "sq.euclidean"):
        return pdist(Z, "euclidean") ** 2
    elif distance == "euclidean":
        return pdist(Z, "euclidean")
    elif distance == "manhattan":
        return pdist(Z, "cityblock")
    elif distance in ("binary", "jaccard"):
        return pdist(Z, "jaccard")
    else:
        return pdist(Z, distance)


def _linkage_from_dissim(dR: np.ndarray, method: str):
    """Build a SciPy linkage matching R's ``hclust(dR, method)``.

    For ``ward.D`` (Radiant's default) R applies the Ward Lance-Williams update
    directly to ``dR``; SciPy implements ``ward.D2`` (squares first, sqrt of
    heights). Feeding ``sqrt(dR)`` to SciPy's ward and squaring the resulting
    heights therefore reproduces R's ``ward.D`` heights exactly. Returns
    ``(linkage_matrix, heights)`` where ``heights`` are on R's scale.
    """
    from scipy.cluster.hierarchy import linkage as _linkage

    if method == "ward.D":
        Z = _linkage(np.sqrt(dR), method="ward")
        heights = Z[:, 2] ** 2
        Z = Z.copy()
        Z[:, 2] = heights
        return Z, heights
    if method == "ward.D2":
        Z = _linkage(dR, method="ward")
        return Z, Z[:, 2].copy()
    if method not in _SCIPY_METHODS:
        raise ValueError(f"Unsupported linkage method: {method}")
    Z = _linkage(dR, method=_SCIPY_METHODS[method])
    return Z, Z[:, 2].copy()


class hclus:
    """Hierarchical cluster analysis.

    Parameters
    ----------
    data, vars : data and variable selection (supports ``"v1:v6"`` ranges).
    labels : str, default "none"
        Column to use as row labels (and excluded from clustering).
    distance : str, default "sq.euclidian"
        ``"sq.euclidian"`` (squared Euclidean), ``"euclidean"``, ``"manhattan"``,
        ``"gower"``. When categorical variables are present the distance is
        forced to ``"gower"`` (matching Radiant).
    method : str, default "ward.D"
        Linkage method (``"ward.D"``, ``"single"``, ``"complete"``, ``"average"``).
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

        self.linkage, self.height = _linkage_from_dissim(dR, method)
        self.nobs = sub.height

    def cutree(self, nr_clus: int) -> np.ndarray:
        """Cluster assignments (1-based) for ``nr_clus`` clusters."""
        from scipy.cluster.hierarchy import fcluster

        return fcluster(self.linkage, t=nr_clus, criterion="maxclust")

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
            pl.Series(name, [str(x) for x in assign]).cast(pl.Categorical)
        )

    def summary(self, dec: int = 2) -> None:
        print("Hierarchical cluster analysis")
        print(f"Data        : {self.name}")
        print_scope(self, 12)
        print(f"Variables   : {', '.join(self.vars)}")
        print(f"Method      : {self.method}")
        print(f"Distance    : {self.distance}")
        print(f"Standardize : {self.standardize}")
        print(f"Observations: {format_nr(self.nobs, dec=0)}")

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
                + scale_x_continuous(breaks=[], expand=(0, 5))
                + scale_y_continuous(expand=(0, 0.02))
                + coord_cartesian(ylim=(cutoff, 1))
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
