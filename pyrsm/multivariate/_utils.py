"""Shared helpers for the ``pyrsm.multivariate`` modules.

These utilities centralize the bits of logic that the factor-analysis,
perceptual-map, clustering, and conjoint ports all need: Radiant-style variable
selection, R-compatible standardization, and a NumPy reproduction of
``psych::principal`` (PCA + varimax) that matches the R package numerically.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pyrsm.utils import check_dataframe

__all__ = [
    "get_data",
    "get_vars",
    "as_numeric_matrix",
    "standardize",
    "varimax",
    "rotate_loadings",
    "principal",
    "ml_factor",
    "factor_scores",
    "is_categorical",
    "is_date",
    "does_vary",
    "apply_filter",
    "gower_dist",
]


def apply_filter(df: pl.DataFrame, data_filter: str = "") -> pl.DataFrame:
    """Apply a Radiant-style string filter to a Polars DataFrame.

    The expression uses pandas/`query`-style syntax (e.g. ``"price > 10000"``).
    An empty filter returns the frame unchanged.
    """
    if data_filter is None or str(data_filter).strip() == "":
        return df
    pdf = df.to_pandas()
    try:
        filtered = pdf.query(data_filter)
    except Exception as e:  # pragma: no cover - surfaced to the user
        raise ValueError(f"Could not apply data_filter '{data_filter}': {e}") from e
    return pl.from_pandas(filtered)


def is_date(series: pl.Series) -> bool:
    """True for Date/Datetime/Time columns (treated as numeric, matching R)."""
    return series.dtype in (pl.Date, pl.Datetime, pl.Time, pl.Duration)


def does_vary(series: pl.Series) -> bool:
    """True if a column shows variation (more than one distinct non-null value)."""
    return series.drop_nulls().n_unique() > 1


def get_data(data):
    """Normalize a data argument into ``(name, polars.DataFrame)``.

    Accepts a Polars/pandas DataFrame or a ``{name: df}`` dict, mirroring the
    convention used throughout ``pyrsm`` (see ``pyrsm.model.regress``).
    """
    if isinstance(data, dict):
        name = list(data.keys())[0]
        df = data[name]
    else:
        df = data
        name = "Not provided"
    return name, check_dataframe(df)


def get_vars(columns, vars):
    """Expand Radiant-style variable selections into a flat column list.

    Supports plain names, lists of names, and colon ranges such as ``"v1:v6"``
    which select every column between the two endpoints by position (matching
    ``dplyr::select(v1:v6)`` as used inside ``radiant``). Order is preserved and
    duplicates are dropped.
    """
    columns = list(columns)
    if vars is None:
        return []
    if isinstance(vars, str):
        vars = [vars]

    out = []
    for v in vars:
        if isinstance(v, str) and ":" in v:
            lhs, rhs = v.split(":", 1)
            if lhs not in columns or rhs not in columns:
                raise ValueError(f"Variable range '{v}' not found in the data")
            ia, ib = columns.index(lhs), columns.index(rhs)
            sel = columns[ia : ib + 1] if ia <= ib else columns[ib : ia + 1][::-1]
            out.extend(sel)
        else:
            if v not in columns:
                raise ValueError(f"Variable '{v}' not found in the data")
            out.append(v)

    seen = set()
    deduped = []
    for c in out:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def is_categorical(series: pl.Series) -> bool:
    """True for Enum/Categorical/String columns (i.e. non-numeric factors)."""
    dtype = series.dtype
    return isinstance(dtype, (pl.Enum, pl.Categorical)) or dtype in (
        pl.Utf8,
        pl.String,
    )


def as_numeric_matrix(data: pl.DataFrame, vars: list[str]) -> np.ndarray:
    """Return a float matrix for ``vars``.

    Numeric columns pass through. Enum/Categorical/String columns are converted
    to 1-based integer codes, preserving Enum ordering (matching R's
    ``as.numeric(factor)``). This is location/scale-relevant only for
    correlations, where integer coding is the right behavior for ordinal data.
    """
    cols = []
    for v in vars:
        s = data[v]
        if is_categorical(s):
            if isinstance(s.dtype, pl.Enum):
                cats = s.dtype.categories.to_list()
                mapping = {c: i + 1 for i, c in enumerate(cats)}
                codes = s.cast(pl.Utf8).replace_strict(
                    mapping, default=None, return_dtype=pl.Float64
                )
            else:
                # occurrence order for plain strings / unordered categoricals
                cats = s.cast(pl.Utf8).unique(maintain_order=True).to_list()
                mapping = {c: i + 1 for i, c in enumerate(cats)}
                codes = s.cast(pl.Utf8).replace_strict(
                    mapping, default=None, return_dtype=pl.Float64
                )
            cols.append(codes.to_numpy().astype(float))
        elif is_date(s):
            # Dates are treated as numeric in R (days/seconds since epoch).
            cols.append(s.to_physical().cast(pl.Float64).to_numpy().astype(float))
        else:
            cols.append(s.cast(pl.Float64).to_numpy().astype(float))
    return np.column_stack(cols)


def standardize(M: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Z-score columns using the sample standard deviation (R's ``scale``)."""
    M = np.asarray(M, dtype=float)
    mu = M.mean(axis=0)
    sd = M.std(axis=0, ddof=ddof)
    sd = np.where(sd == 0, 1.0, sd)
    return (M - mu) / sd


def varimax(
    L: np.ndarray, normalize: bool = True, eps: float = 1e-5, max_iter: int = 1000
):
    """Varimax rotation, reproducing R's ``stats::varimax``.

    Uses Kaiser row normalization and the SVD-based update so results match the
    R implementation (and therefore ``psych::principal``) to numerical
    precision.

    Returns ``(rotated_loadings, rotation_matrix)``.
    """
    L = np.asarray(L, dtype=float).copy()
    p, nc = L.shape
    if nc < 2:
        return L, np.eye(nc)

    if normalize:
        sc = np.sqrt((L**2).sum(axis=1))
        sc = np.where(sc == 0, 1.0, sc)
        L = L / sc[:, None]

    TT = np.eye(nc)
    d = 0.0
    for _ in range(max_iter):
        z = L @ TT
        B = L.T @ (z**3 - z @ np.diag(np.ones(p) @ (z**2)) / p)
        u, s, vt = np.linalg.svd(B)
        TT = u @ vt
        dpast = d
        d = s.sum()
        if dpast != 0 and d < dpast * (1 + eps):
            break

    z = L @ TT
    if normalize:
        z = z * sc[:, None]
    return z, TT


def _gpa_orthogonal(L, vgQ, max_iter=1000, tol=1e-5):
    """Gradient-projection algorithm for orthogonal rotations (Jennrich 2001)."""
    p, k = L.shape
    Tmat = np.eye(k)
    al = 1.0
    Lt = L @ Tmat
    f, Gq = vgQ(Lt)
    G = L.T @ Gq
    for _ in range(max_iter):
        M = Tmat.T @ G
        S = (M + M.T) / 2
        Gp = G - Tmat @ S
        s = np.sqrt(np.sum(Gp * Gp))
        if s < tol:
            break
        al *= 2
        for _ in range(20):
            X = Tmat - al * Gp
            U, _D, Vt = np.linalg.svd(X, full_matrices=False)
            Tt = U @ Vt
            Lt = L @ Tt
            ft, Gq = vgQ(Lt)
            if ft < f - 0.5 * s * s * al:
                break
            al /= 2
        Tmat = Tt
        f = ft
        G = L.T @ Gq
    return L @ Tmat, Tmat


def _gpa_oblique(L, vgQ, max_iter=1000, tol=1e-5):
    """Gradient-projection algorithm for oblique rotations (Jennrich 2002)."""
    p, k = L.shape
    Tmat = np.eye(k)
    al = 1.0
    Tinv = np.linalg.inv(Tmat)
    Lt = L @ Tinv.T
    f, Gq = vgQ(Lt)
    G = -((Lt.T @ Gq @ Tinv).T)
    for _ in range(max_iter):
        Gp = G - Tmat @ np.diag(np.sum(Tmat * G, axis=0))
        s = np.sqrt(np.sum(Gp * Gp))
        if s < tol:
            break
        al *= 2
        for _ in range(20):
            X = Tmat - al * Gp
            v = 1 / np.sqrt(np.sum(X * X, axis=0))
            Tt = X * v
            Tinv = np.linalg.inv(Tt)
            Lt = L @ Tinv.T
            ft, Gq = vgQ(Lt)
            if ft < f - 0.5 * s * s * al:
                break
            al /= 2
        Tmat = Tt
        f = ft
        G = -((Lt.T @ Gq @ Tinv).T)
    Tinv = np.linalg.inv(Tmat)
    return L @ Tinv.T, Tmat


def _vgQ_quartimax(L):
    return -np.sum(L**4) / 4.0, -(L**3)


def _vgQ_oblimin(L, gam=0.0):
    p, k = L.shape
    L2 = L**2
    N = np.ones((k, k)) - np.eye(k)
    C = np.eye(p) - gam / p * np.ones((p, p))
    X = C @ L2 @ N
    f = np.sum(L2 * X) / 4.0
    Gq = L * X
    return f, Gq


def _vgQ_simplimax(L, k_target=None):
    p, k = L.shape
    if k_target is None:
        k_target = p
    L2 = L**2
    flat = np.sort(L2, axis=None)
    cut = flat[min(k_target, flat.size) - 1]
    Imat = (L2 <= cut).astype(float)
    IL = Imat * L
    f = np.sum(IL * L)
    Gq = 2 * IL
    return f, Gq


def rotate_loadings(loadings: np.ndarray, rotation: str = "varimax"):
    """Rotate a loadings matrix, reproducing ``psych``/``GPArotation`` defaults.

    Returns ``(rotated_loadings, is_oblique)``. ``varimax`` uses the exact
    ``stats::varimax`` replication; ``quartimax``/``oblimin``/``simplimax`` use
    the gradient-projection algorithm with GPArotation's default settings.
    """
    L = np.asarray(loadings, dtype=float)
    if L.shape[1] < 2 or rotation in ("none", None, ""):
        return L, False
    if rotation == "varimax":
        out, _ = varimax(L)
        return out, False
    if rotation == "quartimax":
        out, _ = _gpa_orthogonal(L, _vgQ_quartimax)
        return out, False
    if rotation == "oblimin":
        out, _ = _gpa_oblique(L, lambda x: _vgQ_oblimin(x, gam=0.0))
        return out, True
    if rotation == "simplimax":
        out, _ = _gpa_oblique(L, _vgQ_simplimax)
        return out, True
    raise ValueError(f"Unsupported rotation: {rotation}")


def principal(cmat: np.ndarray, nfactors: int = 1, rotate: str = "varimax") -> dict:
    """Principal-components factor analysis, reproducing ``psych::principal``.

    Parameters
    ----------
    cmat : np.ndarray
        Correlation matrix (p x p).
    nfactors : int
        Number of components to retain.
    rotate : str
        ``"varimax"`` or ``"none"``.

    Returns a dict with ``loadings`` (p x nfactors), ``communality``,
    ``uniqueness``, and ``values`` (all p eigenvalues of ``cmat``).
    """
    cmat = np.asarray(cmat, dtype=float)
    p = cmat.shape[0]
    nfactors = int(max(1, min(nfactors, p)))

    # eigen-decomposition, eigenvalues sorted descending (psych/R convention)
    vals, vecs = np.linalg.eigh(cmat)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # loadings = eigvec * sqrt(eigval) for the retained components
    pos = np.clip(vals[:nfactors], 0, None)
    loadings = vecs[:, :nfactors] * np.sqrt(pos)

    if nfactors > 1 and rotate not in ("none", None, ""):
        loadings, _ = rotate_loadings(loadings, rotate)

    # communalities are rotation invariant (and unaffected by sign/order); use
    # the pre-reorder loadings so oblique pattern matrices report correctly.
    communality = (loadings**2).sum(axis=1)
    uniqueness = 1.0 - communality

    # order retained components by (rotated) variance, descending
    ss = (loadings**2).sum(axis=0)
    col_order = np.argsort(ss)[::-1]
    loadings = loadings[:, col_order]

    # sign convention: flip each column so its column sum is non-negative
    signs = np.sign(loadings.sum(axis=0))
    signs[signs == 0] = 1.0
    loadings = loadings * signs

    return {
        "loadings": loadings,
        "communality": communality,
        "uniqueness": uniqueness,
        "values": vals,
    }


def ml_factor(cmat: np.ndarray, nfactors: int, rotate: str = "varimax") -> dict:
    """Maximum-likelihood factor analysis, reproducing ``psych::fa(fm="ml")``.

    Estimates uniquenesses by minimizing the ML objective (Lawley & Maxwell,
    as in ``stats::factanal``), forms unrotated loadings, then applies the
    requested rotation and ``psych``-style column ordering / sign convention.
    """
    from scipy.optimize import minimize

    R = np.asarray(cmat, dtype=float)
    p = R.shape[0]
    nf = int(max(1, min(nfactors, p - 1)))

    def loadings_from_psi(psi):
        sc = np.diag(1.0 / np.sqrt(psi))
        Sstar = sc @ R @ sc
        vals, vecs = np.linalg.eigh(Sstar)
        order = np.argsort(vals)[::-1]
        vals = vals[order][:nf]
        vecs = vecs[:, order][:, :nf]
        L = np.maximum(vals - 1.0, 0.0)
        loadings = np.sqrt(psi)[:, None] * vecs * np.sqrt(L)[None, :]
        return loadings, vals_full(psi)

    def vals_full(psi):
        sc = np.diag(1.0 / np.sqrt(psi))
        Sstar = sc @ R @ sc
        return np.sort(np.linalg.eigvalsh(Sstar))[::-1]

    def objective(psi):
        psi = np.clip(psi, 1e-4, 1.0)
        sc = np.diag(1.0 / np.sqrt(psi))
        Sstar = sc @ R @ sc
        e = np.sort(np.linalg.eigvalsh(Sstar))[::-1]
        e = e[nf:]
        # Lawley-Maxwell objective (factanal): sum(log(e) - e) over discarded
        return -np.sum(np.log(e) - e) - (p - nf)

    start = np.clip((1.0 - 0.5 * nf / p) / np.diag(np.linalg.inv(R)), 0.05, 1.0)
    res = minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=[(1e-4, 1.0)] * p,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    psi = np.clip(res.x, 1e-4, 1.0)
    loadings, allvals = loadings_from_psi(psi)

    # rescale uniquenesses to match diag(R) (factanal normalization)
    communality = (loadings**2).sum(axis=1)
    uniqueness = np.diag(R) - communality

    if nf > 1 and rotate not in ("none", None, ""):
        loadings, _ = rotate_loadings(loadings, rotate)
        communality = (loadings**2).sum(axis=1)
        uniqueness = np.diag(R) - communality

    ss = (loadings**2).sum(axis=0)
    col_order = np.argsort(ss)[::-1]
    loadings = loadings[:, col_order]
    signs = np.sign(loadings.sum(axis=0))
    signs[signs == 0] = 1.0
    loadings = loadings * signs

    return {
        "loadings": loadings,
        "communality": communality,
        "uniqueness": uniqueness,
        "values": allvals,
        "psi": psi,
    }


def gower_dist(
    data: pl.DataFrame, vars: list[str], standardize: bool = True
) -> np.ndarray:
    """Condensed Gower distance vector for mixed numeric/categorical data.

    Reproduces ``gower::gower_dist`` as used by ``radiant.multivariate::hclus``:
    numeric variables are range-normalized absolute differences (optionally
    standardized first, matching Radiant), categorical variables contribute the
    simple-matching distance (0 if equal, 1 otherwise), and the per-pair
    distance is the unweighted mean across variables.

    Returns a condensed (SciPy ``pdist``-style) distance vector.
    """
    from scipy.spatial.distance import squareform

    n = data.height
    num_cols = []
    cat_cols = []
    for v in vars:
        s = data[v]
        if is_categorical(s):
            cat_cols.append(s.cast(pl.Utf8).to_numpy())
        else:
            arr = (
                s.to_physical().cast(pl.Float64).to_numpy().astype(float)
                if is_date(s)
                else s.cast(pl.Float64).to_numpy().astype(float)
            )
            num_cols.append(arr)

    if standardize and num_cols:
        std = []
        for arr in num_cols:
            mu = arr.mean()
            sd = arr.std(ddof=1)
            sd = sd if sd != 0 else 1.0
            std.append((arr - mu) / sd)
        num_cols = std

    nvar = len(num_cols) + len(cat_cols)
    D = np.zeros((n, n))
    for arr in num_cols:
        rng = arr.max() - arr.min()
        if rng == 0:
            continue
        D += np.abs(arr[:, None] - arr[None, :]) / rng
    for arr in cat_cols:
        D += (arr[:, None] != arr[None, :]).astype(float)
    D /= nvar
    np.fill_diagonal(D, 0.0)
    return squareform(D, checks=False)


def factor_scores(std_data: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """PCA factor scores: ``std_data @ (L @ inv(L'L))`` (radiant's ``cscm``)."""
    L = np.asarray(loadings, dtype=float)
    cscm = L @ np.linalg.inv(L.T @ L)
    return np.asarray(std_data, dtype=float) @ cscm
