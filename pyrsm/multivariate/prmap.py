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
from ._label_layout import layout_labels
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
from pyrsm.data_scope import apply_scope, print_dropped, print_scope
from pyrsm.basics import display_utils as du

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

    Attributes
    ----------
    scores : pl.DataFrame
        Brand coordinates (averaged factor scores, sorted by brand).
    loadings : np.ndarray
        Attribute loadings (attr x nr_dim).
    communality, eigen : np.ndarray
    pref_cor : pl.DataFrame | None
        Preference correlations with communalities.
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
        brand: str,
        attr,
        pref="",
        nr_dim: int = 2,
        hcor: bool = False,
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
        unscoped = self.data
        self.data = apply_scope(self.data, data_filter, sort, slice)
        self.brand = brand
        self.attr = get_vars(self.data.columns, attr)
        self.pref = (
            get_vars(self.data.columns, pref) if pref not in ("", None, []) else []
        )
        self.nr_dim = int(nr_dim)
        self.hcor = hcor

        sel = [brand] + self.attr + self.pref
        # Original row positions (numbered before the scope) so ``store`` can
        # put each respondent's scores back on its own row, as in full_factor.
        work = apply_scope(
            unscoped.with_row_index("__row_idx__"), data_filter, sort, slice
        ).select(["__row_idx__", *sel]).drop_nulls()
        # rows with a missing value in any selected variable (listwise, as R)
        self.nobs_dropped = self.data.height - work.height
        self._orig_height = unscoped.height
        self._row_idx = work["__row_idx__"].to_numpy()
        sub = work.select(sel)
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
        # psych's labels, permuted along with the columns (e.g. RC1 RC3 RC2)
        self.fnames = pr["names"]

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

    def summary(self, cutoff: float = 0.0, dec: int = 2, plain: bool = True) -> None:
        """Print the summary. ``plain=False`` in a notebook (or a rendered Quarto report) shows the formatted report Radiant shows."""
        if not plain and du.is_notebook():
            # Radiant's formatted report (pyrsm.report), as its Summary tab shows it
            return du.show_text_report(self, "prmap", cutoff=cutoff, dec=dec)

        print("Attribute based brand map")
        print(f"Data        : {self.name}")
        print_scope(self, 12)
        print(f"Attributes  : {', '.join(self.attr)}")
        if self.pref:
            print(f"Preferences : {', '.join(self.pref)}")
        print(f"Dimensions  : {self.nr_dim}")
        print("Rotation    : varimax")
        print(f"Observations: {format_nr(self.nobs, dec=0)}")
        print_dropped(self, 12)
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

    def store(self, data=None, name: str | list[str] | None = None):
        """Append the factor scores to the data, returning a Polars DataFrame.

        Radiant's "Store factor scores" (``store.full_factor``, which prmap
        objects inherit): the scores of every analysed *row* (not the brand
        means in ``scores``) go into columns ``<name>1..<name>k`` (default
        ``factor1..factorK``), where the number is taken from the factor label,
        so with labels ``RC1 RC3 RC2`` the columns are ``name1 name3 name2``.
        Rows dropped by the filter or by missing values get ``null``. A list
        of ``nr_dim`` names is used as given.
        """
        if data is None:
            target = self.data
        else:
            _, target = get_data(data)
        if name is None or isinstance(name, str):
            prefix = name or "factor"
            name = [prefix + "".join(ch for ch in fn if ch.isdigit()) for fn in self.fnames]
        if len(name) != self.nr_dim:
            raise ValueError(f"Provide {self.nr_dim} names to store the factor scores.")

        scores = self._row_scores
        if target.height == scores.shape[0]:
            series = [pl.Series(nm, scores[:, j]) for j, nm in enumerate(name)]
        elif target.height == self._orig_height:
            full = np.full((self._orig_height, self.nr_dim), np.nan)
            full[self._row_idx] = scores
            series = [
                pl.Series(nm, full[:, j]).fill_nan(None) for j, nm in enumerate(name)
            ]
        else:
            raise ValueError(
                "Cannot store scores: row count of target data does not match "
                "the analysis sample."
            )
        return target.with_columns(series)

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
        lim = symmetric_limit(*comps, pad=1.05) if comps else symmetric_limit(brand, pad=1.05)

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
        )

        from pyrsm.plot_style import COMP, LINE, MUTED, REF, theme_pyrsm

        from ._plotting import add_square_axes

        # Radiant: brand black, attr darkblue, pref red -> REF / LINE / COMP
        colors = {"brand": REF, "attr": LINE, "pref": COMP}

        # the labelled points: brands, then attribute and preference arrow tips
        lab_pts, lab_txt, lab_col = [], [], []
        if "brand" in plots:
            for r, name in enumerate(brand_lab):
                lab_pts.append([brand[r, i], brand[r, j]])
                lab_txt.append(name)
                lab_col.append("brand")
        arr_rows = []
        if "attr" in plots:
            for r, name in enumerate(self.attr):
                lab_pts.append([attr[r, i], attr[r, j]])
                lab_txt.append(name)
                lab_col.append("attr")
                arr_rows.append((0.9 * attr[r, i], 0.9 * attr[r, j], "attr", len(lab_pts) - 1))
        if "pref" in plots and pref is not None:
            for r, name in enumerate(self.pref):
                lab_pts.append([pref[r, i], pref[r, j]])
                lab_txt.append(name)
                lab_col.append("pref")
                arr_rows.append((0.9 * pref[r, i], 0.9 * pref[r, j], "pref", len(lab_pts) - 1))

        def base(lim_):
            g = add_square_axes(ggplot(), lim_)
            if "brand" in plots:
                bdat = pl.DataFrame({"x": brand[:, i], "y": brand[:, j]}).to_pandas()
                g = g + geom_point(bdat, aes(x="x", y="y"), color=REF)
            # arrows for attr/pref (shortened to 0.9, closed arrowheads)
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
            return (
                g
                + scale_color_manual(values=colors)
                + labs(x=f"Dimension {i + 1}", y=f"Dimension {j + 1}")
                + theme_pyrsm()
                + theme(legend_position="none")
            )

        if not lab_pts:
            return base(lim)
        anchors = np.array(lab_pts)
        lay = layout_labels(
            base, anchors[:, 0], anchors[:, 1], lab_txt, size=fontsz * 2, lim=lim,
            segments=[(0.0, 0.0, a[0], a[1]) for a in arr_rows],
            segment_owner=[a[3] for a in arr_rows],
            # arrowheads: a label should not sit on another arrow's tip
            marks=[(a[0], a[1]) for a in arr_rows],
            mark_owner=[a[3] for a in arr_rows],
            seed=seed,
        )
        g = base(lay.lim)
        if lay.leader.any():
            ex, ey = lay.leader_ends()
            ldat = pl.DataFrame(
                {"x": lay.x[lay.leader], "y": lay.y[lay.leader],
                 "xend": ex[lay.leader], "yend": ey[lay.leader]}
            ).to_pandas()
            g = g + geom_segment(ldat, aes(x="x", y="y", xend="xend", yend="yend"),
                                 color=MUTED, size=0.3)
        tdat = pl.DataFrame(
            {"x": lay.lx, "y": lay.ly, "label": lab_txt, "type": lab_col}
        ).to_pandas()
        return g + geom_text(tdat, aes(x="x", y="y", label="label", color="type"), size=fontsz * 2)
