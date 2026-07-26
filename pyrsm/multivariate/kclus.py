"""K-clustering — port of ``radiant.multivariate::kclus``.

``fun="kmeans"`` performs K-means clustering, optionally initialized from the
hierarchical-clustering centers (Radiant's default), which makes the result
deterministic and matches R's ``kmeans(x, centers = hc_centers)``.

``fun="kproto"`` performs K-Prototypes clustering for mixed numeric/categorical
data (a faithful reimplementation of ``clustMixType::kproto`` with the Huang
distance: squared Euclidean for numeric variables plus ``lambda`` times the
simple-matching distance for categorical variables, ``lambda`` estimated as
``mean(numeric variance) / mean(categorical 1 - sum p^2)``).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pyrsm.utils import format_nr

from ._plotting import as_plot_result
from ._utils import does_vary, get_data, get_vars, is_categorical
from ._utils import standardize as _standardize

__all__ = ["kclus"]


def _relabel_first_appearance(labels: np.ndarray) -> np.ndarray:
    """Relabel clusters 1..k by order of first appearance (R's cutree order)."""
    labels = np.asarray(labels)
    mapping = {}
    nxt = 1
    out = np.empty_like(labels)
    for i, v in enumerate(labels):
        if v not in mapping:
            mapping[v] = nxt
            nxt += 1
        out[i] = mapping[v]
    return out


def _mode_level(values: np.ndarray, levels: list[str]) -> str:
    """Modal level, tie-broken by the first level in ``levels`` order."""
    counts = {lev: 0 for lev in levels}
    for v in values:
        counts[str(v)] = counts.get(str(v), 0) + 1
    best, best_n = levels[0], -1
    for lev in levels:
        if counts.get(lev, 0) > best_n:
            best, best_n = lev, counts.get(lev, 0)
    return best


class kclus:
    """K-means / K-prototypes cluster analysis.

    Parameters
    ----------
    data, vars : data and variable selection.
    fun : str, default "kmeans"
        ``"kmeans"`` (numeric only) or ``"kproto"`` (mixed-type K-prototypes).
    nr_clus : int, default 2
    hc_init : bool, default True
        Initialize from hierarchical-clustering centers (Radiant default; makes
        results deterministic).
    distance, method : passed to ``hclus`` for the initialization step.
    seed : int, default 1234
        Used only when ``hc_init`` is False (random init).
    lambda_ : float | None
        K-prototypes trade-off; estimated automatically when ``None``.
    standardize : bool, default True

    Attributes
    ----------
    clus_means : pl.DataFrame
        Cluster centers (numeric means on original scale; categorical modes
        with proportions for K-prototypes).
    cluster : np.ndarray
        1-based cluster assignment per observation.
    sizes, withinss : np.ndarray
    tot_withinss, betweenss, totss : float
    """

    def __init__(
        self,
        data,
        vars,
        fun: str = "kmeans",
        hc_init: bool = True,
        distance: str = "sq.euclidian",
        method: str = "ward.D",
        seed: int = 1234,
        nr_clus: int = 2,
        standardize: bool = True,
        lambda_: float | None = None,
    ) -> None:
        self.name, self.data = get_data(data)
        self.vars = get_vars(self.data.columns, vars)
        self.nr_clus = nr_clus
        self.hc_init = hc_init
        self.standardize = standardize
        self.method = method
        self.lambda_ = lambda_

        sub = self.data.select(self.vars).drop_nulls()
        cat_flags = {v: is_categorical(sub[v]) for v in self.vars}
        any_cat = any(cat_flags.values())

        # resolve clustering function (mirrors radiant's branching)
        if fun in ("mean", "kmeans"):
            fun = "kmeans"
            dropped = [v for v in self.vars if cat_flags[v]]
            if dropped:
                print(
                    "** Categorical variables cannot be used with K-means **\n"
                    "** Select the K-proto option instead **"
                )
            self.vars = [v for v in self.vars if not cat_flags[v]]
        elif fun == "kproto":
            if hc_init:
                distance = "gower"
            if not any_cat:
                fun = "kmeans"
                print("** K-means used when no categorical variables included **")
        self.fun = fun
        self.distance = distance

        sub = sub.select(self.vars)
        cat_flags = {v: is_categorical(sub[v]) for v in self.vars}

        # no-variation check
        no_var = [v for v in self.vars if not does_vary(sub[v])]
        if no_var:
            raise ValueError(
                "The following variable(s) show no variation. Please select "
                f"other variables: {', '.join(no_var)}"
            )

        self._sub = sub
        self._cat_flags = cat_flags
        self.num_vars = [v for v in self.vars if not cat_flags[v]]
        self.cat_vars = [v for v in self.vars if cat_flags[v]]

        if fun == "kmeans":
            self._fit_kmeans(sub, distance, method, seed, nr_clus, standardize, hc_init)
        else:
            self._fit_kproto(
                sub, distance, method, nr_clus, standardize, hc_init, lambda_, seed
            )

        self.nobs = len(self.cluster)
        self._build_clus_means(sub)

    # -- K-means (numeric) ----------------------------------------------------

    def _fit_kmeans(
        self, sub, distance, method, seed, nr_clus, standardize, hc_init
    ):
        from sklearn.cluster import KMeans

        from .hclus import hclus

        M = sub.to_numpy().astype(float)
        Z = _standardize(M) if standardize else M

        if hc_init:
            hc = hclus(
                {self.name: sub},
                self.num_vars,
                distance=distance if distance != "gower" else "sq.euclidian",
                method=method,
                max_cases=10**9,
                standardize=standardize,
            )
            init_labels = _relabel_first_appearance(hc.cutree(nr_clus))
            centers = np.vstack(
                [Z[init_labels == k].mean(axis=0) for k in range(1, nr_clus + 1)]
            )
            km = KMeans(n_clusters=nr_clus, init=centers, n_init=1, max_iter=500).fit(Z)
        else:
            km = KMeans(
                n_clusters=nr_clus, n_init=10, max_iter=500, random_state=seed
            ).fit(Z)

        self.cluster = km.labels_ + 1
        centers = km.cluster_centers_
        grand = Z.mean(axis=0)
        self.totss = float(((Z - grand) ** 2).sum())
        withinss = np.array(
            [((Z[km.labels_ == k] - centers[k]) ** 2).sum() for k in range(nr_clus)]
        )
        self.withinss = withinss
        self.tot_withinss = float(withinss.sum())
        self.betweenss = self.totss - self.tot_withinss
        self.sizes = np.array(
            [int((km.labels_ == k).sum()) for k in range(nr_clus)]
        )

    # -- K-prototypes (mixed) -------------------------------------------------

    def _kproto_levels(self, sub):
        levels = {}
        for v in self.cat_vars:
            s = sub[v]
            if isinstance(s.dtype, pl.Enum):
                levels[v] = s.dtype.categories.to_list()
            else:
                levels[v] = sorted(s.cast(pl.Utf8).unique().to_list())
        return levels

    def _kproto_lambda(self, num_std, cat_arrs):
        vnum = [arr.var(ddof=1) for arr in num_std] if num_std else [0.0]
        vcat = []
        for arr in cat_arrs:
            n = len(arr)
            _, counts = np.unique(arr, return_counts=True)
            p = counts / n
            vcat.append(1.0 - np.sum(p**2))
        vcat = vcat or [1.0]
        mean_num = np.mean(vnum)
        mean_cat = np.mean(vcat)
        return float(mean_num / mean_cat) if mean_cat > 0 else 1.0

    def _kproto_dist(self, num_std, cat_codes, proto_num, proto_cat, lam):
        n = num_std[0].shape[0] if num_std else cat_codes[0].shape[0]
        d = np.zeros(n)
        for j, arr in enumerate(num_std):
            d += (arr - proto_num[j]) ** 2
        for j, arr in enumerate(cat_codes):
            d += lam * (arr != proto_cat[j]).astype(float)
        return d

    def _fit_kproto(
        self, sub, distance, method, nr_clus, standardize, hc_init, lambda_, seed=1234
    ):
        from .hclus import hclus

        levels = self._kproto_levels(sub)
        self._levels = levels

        # numeric matrix (standardized) and categorical code arrays
        num_arrs = [sub[v].cast(pl.Float64).to_numpy().astype(float) for v in self.num_vars]
        if standardize and num_arrs:
            num_std = []
            for arr in num_arrs:
                mu, sd = arr.mean(), arr.std(ddof=1)
                num_std.append((arr - mu) / (sd if sd != 0 else 1.0))
        else:
            num_std = num_arrs
        cat_str = [sub[v].cast(pl.Utf8).to_numpy() for v in self.cat_vars]
        # encode categoricals to integer codes per level order
        cat_codes = []
        for v, arr in zip(self.cat_vars, cat_str):
            idx = {lev: i for i, lev in enumerate(levels[v])}
            cat_codes.append(np.array([idx[str(x)] for x in arr]))

        n = sub.height
        lam = lambda_ if lambda_ is not None else self._kproto_lambda(num_std, cat_str)
        self.lambda_ = lam

        # initial prototypes
        if hc_init:
            hc = hclus(
                {self.name: sub},
                self.vars,
                distance="gower",
                method=method,
                max_cases=10**9,
                standardize=standardize,
            )
            init_labels = hc.cutree(nr_clus)
            # R groups initial prototypes by cutree id 1..k (sorted), so mirror
            # that ordering rather than order of first appearance.
            uniq = sorted(set(init_labels.tolist()))
            proto_num = []
            proto_cat = []
            for lab in uniq:
                mask = init_labels == lab
                proto_num.append([arr[mask].mean() for arr in num_std])
                pc = []
                for v, codes in zip(self.cat_vars, cat_codes):
                    vals = [levels[v][c] for c in codes[mask]]
                    mode = _mode_level(np.array(vals), levels[v])
                    pc.append(levels[v].index(mode))
                proto_cat.append(pc)
            proto_num = np.array(proto_num) if num_std else np.zeros((nr_clus, 0))
            proto_cat = np.array(proto_cat) if cat_codes else np.zeros((nr_clus, 0), int)
        else:
            rng = np.random.default_rng(seed)
            ids = rng.choice(n, size=nr_clus, replace=False)
            proto_num = np.array([[arr[i] for arr in num_std] for i in ids]) if num_std else np.zeros((nr_clus, 0))
            proto_cat = np.array([[codes[i] for codes in cat_codes] for i in ids]) if cat_codes else np.zeros((nr_clus, 0), int)

        labels = np.zeros(n, dtype=int)
        for _ in range(500):
            # assign
            dmat = np.column_stack(
                [
                    self._kproto_dist(
                        num_std, cat_codes, proto_num[k], proto_cat[k], lam
                    )
                    for k in range(nr_clus)
                ]
            )
            new_labels = dmat.argmin(axis=1)
            moved = int((new_labels != labels).sum())
            labels = new_labels
            # update prototypes
            for k in range(nr_clus):
                mask = labels == k
                if not mask.any():
                    continue
                for j, arr in enumerate(num_std):
                    proto_num[k, j] = arr[mask].mean()
                for j, v in enumerate(self.cat_vars):
                    vals = [levels[v][c] for c in cat_codes[j][mask]]
                    mode = _mode_level(np.array(vals), levels[v])
                    proto_cat[k, j] = levels[v].index(mode)
            if moved == 0:
                break

        # final distances / within
        dmat = np.column_stack(
            [
                self._kproto_dist(num_std, cat_codes, proto_num[k], proto_cat[k], lam)
                for k in range(nr_clus)
            ]
        )
        min_d = dmat[np.arange(n), labels]
        self.withinss = np.array(
            [min_d[labels == k].sum() for k in range(nr_clus)]
        )
        self.tot_withinss = float(self.withinss.sum())

        # total heterogeneity = within of a single global prototype
        g_num = [arr.mean() for arr in num_std]
        g_cat = []
        for j, v in enumerate(self.cat_vars):
            vals = [levels[v][c] for c in cat_codes[j]]
            g_cat.append(levels[v].index(_mode_level(np.array(vals), levels[v])))
        self.totss = float(
            self._kproto_dist(num_std, cat_codes, g_num, g_cat, lam).sum()
        )
        self.betweenss = self.totss - self.tot_withinss

        self.cluster = labels + 1
        self.sizes = np.array([int((labels == k).sum()) for k in range(nr_clus)])

    # -- shared ---------------------------------------------------------------

    def _build_clus_means(self, sub):
        levels = getattr(self, "_levels", None) or self._kproto_levels(sub)
        rows = {}
        for v in self.vars:
            col = []
            for k in range(self.nr_clus):
                mask = self.cluster == (k + 1)
                if self._cat_flags[v]:
                    vals = sub[v].cast(pl.Utf8).to_numpy()[mask]
                    mode = _mode_level(vals, levels[v])
                    pct = round(100 * np.mean(vals == mode))
                    col.append(f"{mode} ({pct:.0f}%)")
                else:
                    col.append(float(sub[v].cast(pl.Float64).to_numpy()[mask].mean()))
            rows[v] = col
        self.clus_means = pl.DataFrame(rows)
        # numeric-only matrix for plotting/back-compat
        if self.num_vars:
            self._means = np.column_stack(
                [
                    [
                        float(sub[v].cast(pl.Float64).to_numpy()[self.cluster == (k + 1)].mean())
                        for k in range(self.nr_clus)
                    ]
                    for v in self.num_vars
                ]
            )
        else:
            self._means = np.zeros((self.nr_clus, 0))

    def store(self, data=None, name: str | None = None):
        if name is None:
            name = f"kclus{self.nr_clus}"
        target = self.data if data is None else get_data(data)[1]
        if target.height != len(self.cluster):
            raise ValueError("Target row count does not match number of cases.")
        return target.with_columns(
            pl.Series(name, [str(x) for x in self.cluster]).cast(pl.Categorical)
        )

    def summary(self, dec: int = 2) -> None:
        kind = "means" if self.fun == "kmeans" else "prototypes"
        print(f"K-{self.fun[1:]} cluster analysis")
        print(f"Data         : {self.name}")
        print(f"Variables    : {', '.join(self.vars)}")
        print(f"Clustering by: K-{self.fun[1:]}")
        if self.fun == "kproto":
            print(f"Lambda       : {round(self.lambda_, dec)}")
        if self.hc_init:
            print(f"HC method    : {self.method}")
            print(f"HC distance  : {self.distance}")
        print(f"Standardize  : {self.standardize}")
        print(f"Observations : {format_nr(self.nobs, dec=0)}")
        print(
            f"Generated    : {self.nr_clus} clusters of sizes "
            + " | ".join(str(int(x)) for x in self.sizes)
        )
        print(f"\nCluster {kind}:")
        cm = self.clus_means.with_columns(
            pl.Series("Cluster", [f"Cluster {k + 1}" for k in range(self.nr_clus)])
        ).select(["Cluster", *self.vars])
        print(cm)
        print(
            "\nPercentage of within-cluster heterogeneity accounted for "
            "by each cluster:"
        )
        wcv = self.withinss / self.tot_withinss
        print(
            pl.DataFrame(
                {
                    "Cluster": [f"Cluster {k + 1}" for k in range(self.nr_clus)],
                    "": [f"{round(100 * float(x), dec)}%" for x in wcv],
                }
            )
        )
        print(
            f"\nBetween-cluster heterogeneity accounts for "
            f"{round(100 * self.betweenss / self.totss, dec)}% of the total "
            f"heterogeneity in the data (higher is better)"
        )

    def plot(self, plots="density", dec: int = 2, custom: bool = False):
        """Cluster plots: ``"density"``, ``"bar"``, or ``"scatter"``.

        Numeric variables use density/bar(mean±ME)/jitter-scatter by cluster;
        categorical variables fall back to a proportional stacked bar (matching
        Radiant). Returns a single plotnine object or a list.
        """
        if isinstance(plots, str):
            plots = [plots]
        df = self._sub.with_columns(
            pl.Series("Cluster", [f"Cluster {c}" for c in self.cluster])
        ).to_pandas()

        out = []
        for ptype in plots:
            for v in self.vars:
                if self._cat_flags[v]:
                    out.append(self._fct_plot(df, v))
                elif ptype == "density":
                    out.append(self._density_plot(df, v))
                elif ptype == "bar":
                    out.append(self._bar_plot(df, v, dec))
                elif ptype == "scatter":
                    out.append(self._scatter_plot(df, v))
        return as_plot_result(out)

    def _fct_plot(self, df, var):
        from plotnine import (
            aes,
            geom_bar,
            ggplot,
            labs,
            theme_classic,
        )

        return (
            ggplot(df, aes(x="Cluster", fill=var))
            + geom_bar(position="fill", alpha=0.5, color="black")
            + labs(y="", x="Cluster", fill=var)
            + theme_classic()
        )

    def _density_plot(self, df, var):
        from plotnine import aes, geom_density, ggplot, labs, theme_classic

        return (
            ggplot(df, aes(x=var, fill="Cluster"))
            + geom_density(alpha=0.3)
            + labs(y="", x=var)
            + theme_classic()
        )

    def _bar_plot(self, df, var, dec):
        import pandas as pd
        from plotnine import (
            aes,
            geom_bar,
            geom_errorbar,
            ggplot,
            labs,
            theme_classic,
        )
        from scipy.stats import t as tdist

        g = df.groupby("Cluster")[var]
        summ = g.agg(["mean", "count", "std"]).reset_index()
        summ["se"] = summ["std"] / np.sqrt(summ["count"])
        summ["me"] = summ["se"] * tdist.ppf(0.975, summ["count"] - 1)
        summ = pd.DataFrame(summ)
        return (
            ggplot(summ, aes(x="Cluster", y="mean", fill="Cluster"))
            + geom_bar(stat="identity", alpha=0.5)
            + geom_errorbar(aes(ymin="mean - me", ymax="mean + me"), width=0.1)
            + labs(y=f"{var} (mean)", x="Cluster")
            + theme_classic()
        )

    def _scatter_plot(self, df, var):
        from plotnine import (
            aes,
            geom_jitter,
            ggplot,
            labs,
            theme_classic,
        )

        return (
            ggplot(df, aes(x="Cluster", y=var))
            + geom_jitter(width=0.2, color="black", alpha=0.5)
            + labs(x="Cluster", y=var)
            + theme_classic()
        )
