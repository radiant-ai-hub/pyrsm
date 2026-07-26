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
    ) -> None:
        self.name, self.data = get_data(data)
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
        print(f"Variables   : {', '.join(self.vars)}")
        print(f"Method      : {self.method}")
        print(f"Distance    : {self.distance}")
        print(f"Standardize : {self.standardize}")
        print(f"Observations: {format_nr(self.nobs, dec=0)}")

    def plot(self, plots="dendro", dec: int = 2, cutoff: float = 0.05):
        """Cluster plots.

        ``"dendro"`` returns a matplotlib figure (SciPy dendrogram); ``"scree"``
        and ``"change"`` return plotnine objects of within-cluster
        heterogeneity vs. number of clusters. Heights are normalized to ``[0, 1]``
        by ``max(height)`` (matching Radiant).
        """
        if isinstance(plots, str):
            plots = [plots]
        out = []
        for ptype in plots:
            if ptype == "dendro":
                import matplotlib.pyplot as plt
                from scipy.cluster.hierarchy import dendrogram

                fig, ax = plt.subplots()
                # Heights are already on R's scale; normalize for display.
                lk = self.linkage.copy()
                hmax = lk[:, 2].max()
                if hmax > 0:
                    lk[:, 2] = lk[:, 2] / hmax
                dendrogram(lk, labels=self.row_labels, ax=ax)
                ax.set_title("Dendrogram")
                ax.set_ylabel("Within-cluster heterogeneity")
                if cutoff:
                    ax.axhline(cutoff, color="red", linestyle="dashed", linewidth=0.5)
                out.append(fig)
            elif ptype in ("scree", "change"):
                out.append(self._scree_change(ptype, cutoff))
        return as_plot_result(out)

    def _scree_change(self, ptype: str, cutoff: float):
        from plotnine import (
            aes,
            geom_bar,
            geom_line,
            geom_point,
            ggplot,
            labs,
            scale_x_continuous,
            theme_classic,
        )

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
                + geom_line(color="blue", linetype="dashdot", size=0.7)
                + geom_point(color="blue", size=4, shape="o", fill="white")
                + labs(
                    title="Screeplot",
                    x="# clusters",
                    y="Within-cluster heterogeneity",
                )
                + scale_x_continuous(breaks=list(range(1, n + 1)))
                + theme_classic()
            )
        else:  # change
            change = np.abs((heights_norm[1:] - heights_norm[:-1]) / heights_norm[:-1])
            labels = [f"{i + 1}-{i + 2}" for i in range(len(change))]
            dat = pl.DataFrame(
                {"nr_clus": labels, "bump": [float(v) for v in change]}
            ).to_pandas()
            return (
                ggplot(dat, aes(x="nr_clus", y="bump"))
                + geom_bar(stat="identity", alpha=0.5, fill="blue")
                + labs(
                    title="Change in heterogeneity",
                    x="# clusters",
                    y="Rate of change",
                )
                + theme_classic()
            )
