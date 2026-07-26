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
from ._labels import label_positions, text_repel_kwargs
from ._plotting import as_plot_result, symmetric_limit
from ._utils import (
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
        ``"PCA"`` or ``"ML"`` (maximum-likelihood factor analysis).
    hcor : bool, default False
        Treat categorical variables as ordinal (polychoric/polyserial).
    nr_fact : int, default 1
        Number of factors to extract.
    rotation : str, default "varimax"
        ``"none"``, ``"varimax"``, ``"quartimax"``, ``"oblimin"`` or
        ``"simplimax"``.
    data_filter : str, default ""

    Attributes
    ----------
    floadings : np.ndarray
        Factor loadings (p x nr_fact), sign- and order-aligned to ``psych``.
    communality, uniqueness, eigen, scores : np.ndarray
    fnames : list[str]
        Factor column names (``RC*`` when rotated, ``PC*``/``ML*`` otherwise).
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
    ) -> None:
        _method_aliases = {
            "pca": "PCA",
            "ml": "ML",
            "maximum likelihood": "ML",
            "maximum-likelihood": "ML",
            "mle": "ML",
        }
        key = str(method).strip().lower()
        if key not in _method_aliases:
            raise ValueError(
                f"Unknown method '{method}'. Use 'PCA' or 'ML' "
                "(aliases: 'maximum likelihood')."
            )
        method = _method_aliases[key]
        valid_rotations = ("none", "varimax", "quartimax", "oblimin", "simplimax")
        if rotation not in valid_rotations:
            raise ValueError(
                f"Unknown rotation '{rotation}'. Choose one of {valid_rotations}."
            )
        self.name, self.data = get_data(data)
        self.vars = get_vars(self.data.columns, vars)
        p = len(self.vars)
        self.nr_fact = int(max(1, min(nr_fact, p)))
        self.rotation = rotation
        self.method = method
        self.hcor = hcor
        self.data_filter = data_filter

        # Track original row positions so stored scores can be aligned back
        # (filtered / null rows get a missing value).
        work = apply_filter(
            self.data.with_row_index("__row_idx__"), data_filter
        ).select(["__row_idx__", *self.vars]).drop_nulls()
        self._orig_height = self.data.height
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

        rotated = rotation not in ("none", None, "") and self.nr_fact > 1
        prefix = "RC" if rotated else ("PC" if method == "PCA" else "ML")
        self.fnames = [f"{prefix}{i + 1}" for i in range(self.nr_fact)]

        # factor scores
        M = as_numeric_matrix(sub, self.vars)
        Z = standardize(M)
        if method == "PCA":
            self.scores = factor_scores(Z, self.floadings)
        else:
            # Thurstone regression scores: Z %*% solve(R) %*% loadings
            self.scores = Z @ np.linalg.solve(self.cmat, self.floadings)

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
        print(f"Variables   : {', '.join(self.vars)}")
        print(f"Factors     : {self.nr_fact}")
        print(f"Method      : {self.method}")
        print(f"Rotation    : {self.rotation}")
        print(f"Observations: {format_nr(self.nobs, dec=0)}")
        if self.hcor and self.any_categorical:
            print("Correlation : Heterogeneous correlations (polychoric/polyserial)")
            print("** Variables of type {factor} are assumed to be ordinal **\n")
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

    def plot(self, plots="attr", fontsz: int = 5, seed: int = 1234, custom: bool = False):
        """Attribute ("attr") and/or respondent ("resp") loadings plots.

        Requires at least two factors. Uses square, origin-centered axes (scaled
        to respondent scores when ``"resp"`` is shown, otherwise ``[-1, 1]``),
        repelled attribute labels, and dashed loading vectors. Returns a single
        plotnine object for two factors or a list (one per factor pair).
        """
        if isinstance(plots, str):
            plots = [plots]
        if self.nr_fact < 2:
            print("Two or more factors required for a loadings plot")
            return None

        # Limits span both respondent scores (when shown) and the loadings, so
        # loading arrows are never clipped even when scores are smaller.
        if "resp" in plots and "attr" in plots:
            lim = symmetric_limit(self.scores, self.floadings)
        elif "resp" in plots:
            lim = symmetric_limit(self.scores)
        else:
            lim = 1.0

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
            theme_classic,
        )

        from ._plotting import add_square_axes

        g = ggplot()
        g = add_square_axes(g, lim)

        if "resp" in plots:
            sdat = pl.DataFrame(
                {"x": self.scores[:, i], "y": self.scores[:, j]}
            ).to_pandas()
            g = g + geom_point(sdat, aes(x="x", y="y"), alpha=0.4, color="grey")

        if "attr" in plots:
            lab_xy = label_positions(self.floadings[:, [i, j]], lim=lim, seed=seed)
            adat = pl.DataFrame(
                {
                    "x": self.floadings[:, i],
                    "y": self.floadings[:, j],
                    "lx": lab_xy[:, 0],
                    "ly": lab_xy[:, 1],
                    "label": self.vars,
                }
            ).to_pandas()
            repel = text_repel_kwargs(adat[["x", "y"]].to_numpy())
            g = (
                g
                + geom_segment(
                    adat,
                    aes(x=0, y=0, xend="x", yend="y"),
                    color="blue",
                    linetype="dashed",
                    alpha=0.5,
                )
                + geom_point(adat, aes(x="x", y="y"), color="blue")
            )
            label_aes = aes(x="lx", y="ly", label="label")
            if not repel:
                g = g + geom_segment(
                    adat, aes(x="x", y="y", xend="lx", yend="ly"),
                    color="grey", size=0.2,
                )
            g = g + geom_text(
                adat, label_aes, size=fontsz * 2, **repel
            )

        g = (
            g
            + labs(
                title="Attribute loadings",
                x=self.fnames[i],
                y=self.fnames[j],
            )
            + theme_classic()
        )
        return g
