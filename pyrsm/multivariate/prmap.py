"""Attribute-based brand maps — port of ``radiant.multivariate::prmap``.

PCA with varimax on the attribute correlation matrix (reusing the same
``principal`` internals as ``full_factor``), brand coordinates from averaged
factor scores, optional preference correlations, and a perceptual-map plot with
brand points plus attribute / preference arrows.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pyrsm.utils import format_nr

from ._correlation import heterogeneous_corr
from ._labels import label_positions, text_repel_kwargs
from ._plotting import as_plot_result, symmetric_limit
from ._utils import (
    apply_filter,
    as_numeric_matrix,
    factor_scores,
    get_data,
    get_vars,
    is_categorical,
    principal,
    standardize,
)

__all__ = ["prmap"]


class prmap:
    """Attribute-based perceptual map (principal-components map).

    Parameters
    ----------
    data : dataset (or ``{name: df}``).
    brand : str
        Brand / entity identifier column.
    attr : str | list[str]
        Attribute variables (supports ``"a:b"`` ranges).
    pref : str | list[str], default ""
        Optional preference variables to correlate with the map dimensions.
    nr_dim : int, default 2
    hcor : bool, default False
        Treat categorical attributes/preferences as ordinal (heterogeneous
        correlations).
    data_filter : str, default ""

    Attributes
    ----------
    scores : pl.DataFrame
        Brand coordinates (averaged factor scores, sorted by brand).
    loadings : np.ndarray
        Attribute loadings (attr x nr_dim).
    communality, eigen : np.ndarray
    pref_cor : pl.DataFrame | None
        Preference correlations with communalities.
    """

    def __init__(
        self,
        data,
        brand: str,
        attr,
        pref="",
        nr_dim: int = 2,
        hcor: bool = False,
        data_filter: str = "",
    ) -> None:
        self.name, self.data = get_data(data)
        self.brand = brand
        self.attr = get_vars(self.data.columns, attr)
        self.pref = (
            get_vars(self.data.columns, pref) if pref not in ("", None, []) else []
        )
        self.nr_dim = int(nr_dim)
        self.hcor = hcor
        self.data_filter = data_filter

        sel = [brand] + self.attr + self.pref
        sub = apply_filter(self.data, data_filter).select(sel).drop_nulls()
        brands = [str(x).strip() for x in sub[brand].to_list()]
        self.nobs = sub.height

        if self.nr_dim > len(self.attr):
            raise ValueError(
                "The number of dimensions cannot exceed the number of attributes"
            )

        cmat, self.any_categorical = heterogeneous_corr(sub, self.attr, hcor)
        pr = principal(cmat, self.nr_dim, rotate="varimax")
        self.loadings = pr["loadings"]
        self.communality = pr["communality"]
        self.uniqueness = pr["uniqueness"]
        self.eigen = pr["values"]
        self.fnames = [
            f"RC{i + 1}" if self.nr_dim > 1 else "PC1" for i in range(self.nr_dim)
        ]

        f_data = as_numeric_matrix(sub, self.attr)
        row_scores = factor_scores(standardize(f_data), self.loadings)
        self._row_scores = row_scores

        score_df = pl.DataFrame(
            {
                "brand": brands,
                **{self.fnames[j]: row_scores[:, j] for j in range(self.nr_dim)},
            }
        )
        self.scores = score_df.group_by("brand").mean().sort("brand")

        self.pref_cor = None
        if self.pref:
            pref_cat = [is_categorical(sub[p]) for p in self.pref]
            if hcor and any(pref_cat):
                pc = self._pref_hetcor(sub, row_scores)
            else:
                p_data = as_numeric_matrix(sub, self.pref)
                pc = np.zeros((len(self.pref), self.nr_dim))
                for a in range(len(self.pref)):
                    for b in range(self.nr_dim):
                        pc[a, b] = np.corrcoef(p_data[:, a], row_scores[:, b])[0, 1]
            comm = (pc**2).sum(axis=1)
            self._pref_cor_mat = pc
            self.pref_cor = pl.DataFrame(
                {
                    "pref": self.pref,
                    **{self.fnames[j]: pc[:, j] for j in range(self.nr_dim)},
                    "communalities": comm,
                }
            )

    def _pref_hetcor(self, sub, row_scores):
        """Cross-correlations of (categorical) preferences with map scores.

        Ordinal preferences vs. the continuous map scores use the polyserial
        correlation (the relevant block of ``polycor::hetcor``); numeric
        preferences use Pearson.
        """
        from ._correlation import polyserial_corr

        pc = np.zeros((len(self.pref), self.nr_dim))
        p_data = as_numeric_matrix(sub, self.pref)
        pref_cat = [is_categorical(sub[p]) for p in self.pref]
        for a in range(len(self.pref)):
            for b in range(self.nr_dim):
                if pref_cat[a]:
                    pc[a, b] = polyserial_corr(row_scores[:, b], p_data[:, a])
                else:
                    pc[a, b] = np.corrcoef(p_data[:, a], row_scores[:, b])[0, 1]
        return pc

    def summary(self, cutoff: float = 0.0, dec: int = 2) -> None:
        print("Attribute based brand map")
        print(f"Data        : {self.name}")
        print(f"Attributes  : {', '.join(self.attr)}")
        if self.pref:
            print(f"Preferences : {', '.join(self.pref)}")
        print(f"Dimensions  : {self.nr_dim}")
        print("Rotation    : varimax")
        print(f"Observations: {format_nr(self.nobs, dec=0)}")
        corr = "Heterogeneous" if (self.hcor and self.any_categorical) else "Pearson"
        print(f"Correlation : {corr}\n")

        print("Brand - Factor scores:")
        print(self.scores.with_columns([pl.col(c).round(dec) for c in self.fnames]))

        print("\nAttribute - Factor loadings:")
        disp = {" ": self.attr}
        for j, fn in enumerate(self.fnames):
            if cutoff > 0:
                # blank small loadings -> render the whole column as strings so
                # Polars does not choke on a mixed float/empty-string column
                disp[fn] = [
                    "" if abs(v) < cutoff else f"{v:.{dec}f}"
                    for v in self.loadings[:, j]
                ]
            else:
                disp[fn] = [round(float(v), dec) for v in self.loadings[:, j]]
        print(pl.DataFrame(disp))

        if self.pref_cor is not None:
            print("\nPreference correlations:")
            print(
                self.pref_cor.with_columns(
                    [pl.col(c).round(dec) for c in self.fnames + ["communalities"]]
                )
            )

        print("\nFit measures:")
        ss = (self.loadings**2).sum(axis=0)
        var_pct = 100 * ss / len(self.attr)
        print(
            pl.DataFrame(
                {
                    " ": ["Eigenvalues", "Variance %", "Cumulative %"],
                    **{
                        fn: [
                            round(float(ss[j]), dec),
                            round(float(var_pct[j]), dec),
                            round(float(np.cumsum(var_pct)[j]), dec),
                        ]
                        for j, fn in enumerate(self.fnames)
                    },
                }
            )
        )

        print("\nAttribute communalities:")
        print(
            pl.DataFrame(
                {" ": self.attr, "": [round(float(x), dec) for x in self.communality]}
            )
        )

    def plot(
        self,
        plots=("brand", "attr"),
        scaling: float = 2.0,
        fontsz: int = 5,
        seed: int = 1234,
        dim=None,
        custom: bool = False,
    ):
        """Perceptual-map plot for every dimension pair (plotnine).

        Components: ``"brand"`` (black points + labels), ``"attr"`` (dark-blue
        dashed arrows scaled by ``scaling``), ``"pref"`` (red dashed arrows).
        Axes are square and origin-centered using all selected components;
        arrow segments are shortened to 0.9 of the endpoint (so labels clear the
        closed arrowheads). Returns a single plot or a list (one per dim pair).
        """
        if isinstance(plots, str):
            plots = [plots]
        plots = [p for p in plots if not (p == "pref" and self.pref_cor is None)]

        brand = self.scores.to_numpy()[:, 1:].astype(float)
        brand_lab = self.scores["brand"].to_list()
        attr = self.loadings * scaling
        pref = (
            self._pref_cor_mat * scaling if self.pref_cor is not None else None
        )

        # symmetric limits across all plotted numeric coordinates
        comps = []
        if "brand" in plots:
            comps.append(brand)
        if "attr" in plots:
            comps.append(attr)
        if "pref" in plots and pref is not None:
            comps.append(pref)
        lim = symmetric_limit(*comps, pad=1.7) if comps else symmetric_limit(brand, pad=1.7)

        pairs = [(dim[0] - 1, dim[1] - 1)] if dim else [
            (i, j)
            for i in range(self.nr_dim - 1)
            for j in range(i + 1, self.nr_dim)
        ]
        out = [
            self._plot_pair(
                plots, brand, brand_lab, attr, pref, i, j, lim, fontsz, seed
            )
            for i, j in pairs
        ]
        return as_plot_result(out)

    def _plot_pair(self, plots, brand, brand_lab, attr, pref, i, j, lim, fontsz, seed):
        from plotnine import (
            aes,
            arrow,
            geom_point,
            geom_segment,
            geom_text,
            ggplot,
            labs,
            scale_color_manual,
            theme,
            theme_classic,
        )

        from ._plotting import add_square_axes

        colors = {"brand": "black", "attr": "darkblue", "pref": "red"}
        g = ggplot()
        g = add_square_axes(g, lim)

        # gather labels for repel placement
        lab_pts, lab_txt, lab_col = [], [], []
        if "brand" in plots:
            for r, name in enumerate(brand_lab):
                lab_pts.append([brand[r, i], brand[r, j]])
                lab_txt.append(name)
                lab_col.append("brand")
        if "attr" in plots:
            for r, name in enumerate(self.attr):
                lab_pts.append([attr[r, i], attr[r, j]])
                lab_txt.append(name)
                lab_col.append("attr")
        if "pref" in plots and pref is not None:
            for r, name in enumerate(self.pref):
                lab_pts.append([pref[r, i], pref[r, j]])
                lab_txt.append(name)
                lab_col.append("pref")

        if "brand" in plots:
            bdat = pl.DataFrame(
                {"x": brand[:, i], "y": brand[:, j]}
            ).to_pandas()
            g = g + geom_point(bdat, aes(x="x", y="y"), color="black")

        # arrows for attr/pref (shortened to 0.9, closed arrowheads)
        arr_rows = []
        if "attr" in plots:
            for r in range(attr.shape[0]):
                arr_rows.append((0.9 * attr[r, i], 0.9 * attr[r, j], "attr"))
        if "pref" in plots and pref is not None:
            for r in range(pref.shape[0]):
                arr_rows.append((0.9 * pref[r, i], 0.9 * pref[r, j], "pref"))
        if arr_rows:
            adat = pl.DataFrame(
                {
                    "xend": [a[0] for a in arr_rows],
                    "yend": [a[1] for a in arr_rows],
                    "type": [a[2] for a in arr_rows],
                }
            ).to_pandas()
            g = g + geom_segment(
                adat,
                aes(x=0, y=0, xend="xend", yend="yend", color="type"),
                arrow=arrow(length=0.1, type="closed"),
                linetype="dashed",
                size=0.3,
            )

        if lab_pts:
            anchors = np.array(lab_pts)
            lab_xy = label_positions(anchors, lim=lim, seed=seed)
            ldat = pl.DataFrame(
                {
                    "ax": anchors[:, 0],
                    "ay": anchors[:, 1],
                    "lx": lab_xy[:, 0],
                    "ly": lab_xy[:, 1],
                    "label": lab_txt,
                    "type": lab_col,
                }
            ).to_pandas()
            repel = text_repel_kwargs(ldat[["ax", "ay"]].to_numpy())
            if repel:
                label_aes = aes(x="ax", y="ay", label="label", color="type", group=1)
            else:
                label_aes = aes(x="lx", y="ly", label="label", color="type")
                g = g + geom_segment(
                    ldat,
                    aes(x="ax", y="ay", xend="lx", yend="ly"),
                    color="grey",
                    size=0.2,
                )
            g = g + geom_text(
                ldat,
                label_aes,
                size=fontsz * 2,
                **repel,
            )

        g = (
            g
            + scale_color_manual(values=colors)
            + labs(x=f"Dimension {i + 1}", y=f"Dimension {j + 1}")
            + theme_classic()
            + theme(legend_position="none")
        )
        return g
