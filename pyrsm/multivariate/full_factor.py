"""Factor / principal-components analysis — port of ``radiant.multivariate::full_factor``.

Implements PCA and maximum-likelihood (ML) factor analysis on the (optionally
heterogeneous) correlation matrix with several rotations, reproducing
``psych::principal`` / ``psych::fa`` numerically: loadings, communalities,
variance explained, and factor scores. Factor scores can be stored back onto the
data.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pyrsm.utils import format_nr

from ._correlation import heterogeneous_corr
from ._label_layout import layout_labels
from ._plotting import as_plot_result, symmetric_limit
from ._utils import (
    ROTATIONS,
    apply_filter,
    as_numeric_matrix,
    does_vary,
    factor_scores,
    get_data,
    get_vars,
    is_categorical,
    ml_factor,
    principal,
    standardize,
)
from pyrsm.data_scope import apply_scope, print_scope

__all__ = ["full_factor", "clean_loadings"]


def clean_loadings(
    floadings,
    cutoff: float = 0.0,
    fsort: bool = False,
    dec: int = 8,
    repl=None,
):
    """Sort and clean a loadings table, reproducing R's ``clean_loadings``.

    Parameters
    ----------
    floadings : pl.DataFrame | np.ndarray
        Loadings (with or without a leading label column).
    cutoff : float
        Blank out loadings whose absolute value is below ``cutoff``.
    fsort : bool
        Sort rows by dominant factor then magnitude (``psych::fa.sort``).
    dec : int
        Rounding.
    repl : value
        Replacement for below-cutoff loadings (``None``/``np.nan`` or ``""``).
    """
    if isinstance(floadings, np.ndarray):
        df = pl.DataFrame(
            {" ": [f"V{i + 1}" for i in range(floadings.shape[0])]}
            | {f"F{j + 1}": floadings[:, j] for j in range(floadings.shape[1])}
        )
    else:
        df = floadings.clone() if isinstance(floadings, pl.DataFrame) else pl.DataFrame(floadings)

    label_col = df.columns[0] if df.dtypes[0] in (pl.Utf8, pl.String) else None
    num_cols = [c for c in df.columns if c != label_col]

    L = df.select(num_cols).to_numpy().astype(float)

    if fsort:
        # assign each row to the factor with the largest |loading|, then order
        dom = np.argmax(np.abs(L), axis=1)
        # within-factor order by descending |loading on dominant factor|
        order = sorted(
            range(L.shape[0]),
            key=lambda r: (dom[r], -abs(L[r, dom[r]])),
        )
        L = L[order]
        if label_col is not None:
            df = df[order]

    Lr = np.round(L, dec)
    as_str = repl == ""
    out = {}
    if label_col is not None:
        out[label_col] = df[label_col].to_list()
    for j, c in enumerate(num_cols):
        col = []
        for v in Lr[:, j]:
            below = cutoff > 0 and abs(v) < cutoff
            if as_str:
                col.append("" if below else f"{v:.{dec}f}")
            else:
                col.append(np.nan if below else float(v))
        out[c] = col
    return pl.DataFrame(out)


class full_factor:
    """Factor analysis using principal components or maximum likelihood.

    Parameters
    ----------
    data : pl.DataFrame | pd.DataFrame | dict[str, DataFrame]
    vars : str | list[str]
        Variables to use; supports Radiant ranges such as ``"v1:v6"``.
    method : str, default "PCA"
        ``"PCA"`` or ``"ML"`` (maximum-likelihood factor analysis; Radiant's
        ``"maxlik"`` is accepted as an alias).
    hcor : bool, default False
        Treat categorical variables as ordinal (polychoric/polyserial).
    nr_fact : int, default 1
        Number of factors to extract.
    rotation : str, default "varimax"
        ``"none"``, ``"varimax"``, ``"quartimax"``, ``"equamax"``,
        ``"promax"``, ``"oblimin"`` or ``"simplimax"`` (the choices in
        Radiant's UI).

    Attributes
    ----------
    floadings : np.ndarray
        Factor loadings (p x nr_fact), sign- and order-aligned to ``psych``.
        For oblique rotations these are the pattern loadings.
    phi : np.ndarray | None
        Factor correlations for the oblique rotations (promax, oblimin,
        simplimax), ``None`` otherwise.
    communality, uniqueness, eigen, scores : np.ndarray
    fnames : list[str]
        Factor column names as ``psych`` prints them: ``RC*`` for PCA with an
        orthogonal rotation or promax, ``PC*`` for PCA otherwise, ``ML*`` for
        maximum likelihood (in psych's, possibly permuted, order).
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
        method: str = "PCA",
        hcor: bool = False,
        nr_fact: int = 1,
        rotation: str = "varimax",
        data_filter: str = "",
        sort: str = "",
        slice: str = "",
    ) -> None:
        _method_aliases = {
            "pca": "PCA",
            "ml": "ML",
            "maximum likelihood": "ML",
            "maximum-likelihood": "ML",
            "mle": "ML",
            "maxlik": "ML",
        }
        key = str(method).strip().lower()
        if key not in _method_aliases:
            raise ValueError(
                f"Unknown method '{method}'. Use 'PCA' or 'ML' "
                "(aliases: 'maxlik', 'maximum likelihood')."
            )
        method = _method_aliases[key]
        valid_rotations = ROTATIONS
        if rotation not in valid_rotations:
            raise ValueError(
                f"Unknown rotation '{rotation}'. Choose one of {valid_rotations}."
            )
        self.name, self.data = get_data(data)

        # The data scope, applied before anything is measured: filter, then
        # sort, then slice. Kept on the object so ``summary`` can say which
        # rows the numbers describe.
        #
        # ``store`` writes a score per analysed row back onto the *original*
        # rows, padding the ones the scope dropped, so the row index has to be
        # taken before the scope narrows anything -- see below.
        self.data_filter = data_filter
        self.sort = sort
        self.slice = slice
        unscoped = self.data
        self.data = apply_scope(self.data, data_filter, sort, slice)
        self.vars = get_vars(self.data.columns, vars)
        p = len(self.vars)
        self.nr_fact = int(max(1, min(nr_fact, p)))
        self.rotation = rotation
        self.method = method
        self.hcor = hcor

        # Track original row positions so stored scores can be aligned back
        # (filtered / null rows get a missing value). The index is numbered on
        # the unscoped frame and then scoped, so a row that survives the scope
        # still knows where it came from.
        work = apply_scope(
            unscoped.with_row_index("__row_idx__"), data_filter, sort, slice
        ).select(["__row_idx__", *self.vars]).drop_nulls()
        self._orig_height = unscoped.height
        self._row_idx = work["__row_idx__"].to_numpy()
        sub = work.select(self.vars)
        self.nobs = sub.height

        no_var = [v for v in self.vars if not does_vary(sub[v])]
        if no_var:
            raise ValueError(
                "The following variable(s) show no variation. Please select "
                f"other variables: {', '.join(no_var)}"
            )

        self.any_categorical = any(is_categorical(sub[v]) for v in self.vars)
        self.cmat, _ = heterogeneous_corr(sub, self.vars, hcor)

        if method == "PCA":
            pr = principal(self.cmat, self.nr_fact, rotation)
        else:
            pr = ml_factor(self.cmat, self.nr_fact, rotation)
        self.floadings = pr["loadings"]
        self.communality = pr["communality"]
        self.uniqueness = pr["uniqueness"]
        self.eigen = pr["values"]
        self.phi = pr["Phi"]
        self.fnames = pr["names"]

        # factor scores
        M = as_numeric_matrix(sub, self.vars)
        Z = standardize(M)
        if method == "PCA":
            self.scores = factor_scores(Z, self.floadings)
        else:
            # Thurstone regression scores (psych::factor.scores):
            # scale(x) %*% solve(cor(x), loadings %*% Phi)
            S = self.floadings if self.phi is None else self.floadings @ self.phi
            self.scores = Z @ np.linalg.solve(np.corrcoef(M, rowvar=False), S)

    # -- helpers --------------------------------------------------------------

    def loadings_frame(self, dec: int | None = None) -> pl.DataFrame:
        """Loadings as a Polars DataFrame with a leading label column."""
        d = {" ": self.vars}
        for j, fn in enumerate(self.fnames):
            col = self.floadings[:, j]
            d[fn] = [round(float(x), dec) for x in col] if dec else col.tolist()
        return pl.DataFrame(d)

    # -- output ---------------------------------------------------------------

    def summary(self, cutoff: float = 0.0, fsort: bool = False, dec: int = 2) -> None:
        """Print Radiant-style factor-analysis output."""
        print("Factor analysis")
        print(f"Data        : {self.name}")
        print_scope(self, 12)
        print(f"Variables   : {', '.join(self.vars)}")
        print(f"Factors     : {self.nr_fact}")
        print(f"Method      : {self.method}")
        print(f"Rotation    : {self.rotation}")
        print(f"Observations: {format_nr(self.nobs, dec=0)}")
        if self.hcor and self.any_categorical:
            print("Correlation : Heterogeneous correlations (polychoric/polyserial)")
            print("** Categorical variables (Enum, Categorical or String) are treated as ordinal **\n")
        else:
            print("Correlation : Pearson\n")

        print("Factor loadings:")
        print(
            clean_loadings(
                self.loadings_frame(),
                cutoff=cutoff,
                fsort=fsort,
                dec=dec,
                repl="",
            )
        )

        print("\nFit measures:")
        ss = (self.floadings**2).sum(axis=0)
        var_pct = 100 * ss / len(self.vars)
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
                {
                    " ": self.vars,
                    "": [round(float(x), dec) for x in self.communality],
                }
            )
        )

        print("\nFactor scores (max 10 shown):")
        n = min(10, self.scores.shape[0])
        print(
            pl.DataFrame(
                {
                    fn: [round(float(x), dec) for x in self.scores[:n, j]]
                    for j, fn in enumerate(self.fnames)
                }
            )
        )

    def store(self, data=None, name=None):
        """Append factor scores to the data, returning a new Polars DataFrame.

        Default column names are ``factor1..factorK`` (matching Radiant). When a
        filter or dropped rows reduced the analysis sample, scores are placed at
        their original row positions and other rows are filled with ``null``.
        """
        if data is None:
            target = self.data
        else:
            _, target = get_data(data)
        if name is None:
            name = [f"factor{i + 1}" for i in range(self.nr_fact)]
        elif isinstance(name, str):
            name = [name] if self.nr_fact == 1 else [f"{name}{i + 1}" for i in range(self.nr_fact)]

        if target.height == self.scores.shape[0]:
            series = [pl.Series(nm, self.scores[:, j]) for j, nm in enumerate(name)]
        elif target.height == self._orig_height:
            idx = set(self._row_idx.tolist())
            pos = {int(r): k for k, r in enumerate(self._row_idx)}
            series = []
            for j, nm in enumerate(name):
                col = [
                    float(self.scores[pos[i], j]) if i in idx else None
                    for i in range(self._orig_height)
                ]
                series.append(pl.Series(nm, col, dtype=pl.Float64))
        else:
            raise ValueError(
                "Cannot store scores: row count of target data does not match "
                "the analysis sample."
            )
        return target.with_columns(series)

    def plot(
        self,
        plots="attr",
        fontsz: int = 5,
        seed: int = 1234,
        custom: bool = False,
        resp_scale: float | None = None,
    ):
        """Attribute ("attr") and/or respondent ("resp") loadings plots.

        Requires at least two factors. Uses square, origin-centered axes,
        dashed loading vectors, and a label next to each loading (overlapping
        labels are nudged apart). Returns a single plotnine object for two
        factors or a list (one per factor pair).

        resp_scale : float, optional
            Multiplier for the respondents' factor scores when they are shown
            with the loadings (scores run to about +/-2-3, loadings to +/-1).
            Default: shrink the scores so they fit the range of the loadings.
            1 plots the scores as they are.
        """
        if isinstance(plots, str):
            plots = [plots]
        if self.nr_fact < 2:
            print("Two or more factors required for a loadings plot")
            return None

        scores = np.asarray(self.scores, dtype=float)
        loadings = np.asarray(self.floadings, dtype=float)
        if "resp" in plots and "attr" in plots:
            if resp_scale is None:
                top = float(np.nanmax(np.abs(scores))) if scores.size else 0.0
                resp_scale = float(np.nanmax(np.abs(loadings))) / top if top else 1.0
            resp_scale = float(resp_scale)
        else:
            resp_scale = 1.0 if resp_scale is None else float(resp_scale)
        self._resp_scale = resp_scale

        # Axes hold the (scaled) scores and the loadings with room for labels.
        if "resp" in plots and "attr" in plots:
            lim = max(symmetric_limit(scores * resp_scale), symmetric_limit(loadings, pad=1.05))
        elif "resp" in plots:
            lim = symmetric_limit(scores * resp_scale)
        else:
            lim = max(1.0, symmetric_limit(loadings, pad=1.05))

        pairs = [
            (i, j)
            for i in range(self.nr_fact - 1)
            for j in range(i + 1, self.nr_fact)
        ]
        out = [self._plot_pair(plots, i, j, lim, fontsz, seed) for i, j in pairs]
        return as_plot_result(out)

    def _plot_pair(self, plots, i, j, lim, fontsz, seed):
        from plotnine import (
            aes,
            geom_point,
            geom_segment,
            geom_text,
            ggplot,
            labs,
        )

        from pyrsm.plot_style import ALPHA, LINE, MUTED, theme_pyrsm

        from ._plotting import add_square_axes

        scale = getattr(self, "_resp_scale", 1.0)
        x = self.floadings[:, i]
        y = self.floadings[:, j]
        adat = pl.DataFrame({"x": x, "y": y, "label": self.vars}).to_pandas()
        sx = self.scores[:, i] * scale
        sy = self.scores[:, j] * scale

        def base(lim_):
            g = add_square_axes(ggplot(), lim_)
            if "resp" in plots:
                sdat = pl.DataFrame({"x": sx, "y": sy}).to_pandas()
                g = g + geom_point(sdat, aes(x="x", y="y"), alpha=ALPHA, color=MUTED)
            if "attr" in plots:
                g = (
                    g
                    + geom_segment(
                        adat,
                        aes(x=0, y=0, xend="x", yend="y"),
                        color=LINE,
                        linetype="dashed",
                        alpha=ALPHA,
                    )
                    + geom_point(adat, aes(x="x", y="y"), color=LINE)
                )
            return g + labs(
                title="Attribute loadings",
                subtitle=(
                    f"Respondent scores x {scale:.2f}"
                    if "resp" in plots and abs(scale - 1) > 1e-9
                    else None
                ),
                x=self.fnames[i],
                y=self.fnames[j],
            ) + theme_pyrsm()

        if "attr" not in plots:
            return base(lim)

        lay = layout_labels(
            base, x, y, self.vars, size=fontsz * 2, lim=lim,
            background=np.column_stack([sx, sy]) if "resp" in plots else None,
            segments=[(0.0, 0.0, float(a), float(b)) for a, b in zip(x, y)],
            segment_owner=list(range(len(x))),
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
        tdat = pl.DataFrame({"x": lay.lx, "y": lay.ly, "label": self.vars}).to_pandas()
        return g + geom_text(tdat, aes(x="x", y="y", label="label"), size=fontsz * 2)
