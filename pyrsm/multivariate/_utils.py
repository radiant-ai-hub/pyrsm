"""Shared helpers for the ``pyrsm.multivariate`` modules.

These utilities centralize the bits of logic that the factor-analysis,
perceptual-map, clustering, and conjoint ports all need: Radiant-style variable
selection, R-compatible standardization, and a NumPy reproduction of
``psych::principal`` (PCA + varimax) that matches the R package numerically.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pyrsm.data_scope import apply_scope
from pyrsm.utils import check_dataframe

__all__ = [
    "get_data",
    "get_vars",
    "as_numeric_matrix",
    "standardize",
    "r_scale",
    "varimax",
    "rotate_loadings",
    "ROTATIONS",
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
    """Apply a ``data_filter`` expression to a Polars DataFrame.

    Kept as a name because the multivariate modules call it; the language and
    the evaluation live in :mod:`pyrsm.data_scope`, so every class filters the
    same way. This used to convert to pandas and call ``query``, which meant
    the multivariate tools understood a different dialect than everything
    else and paid a full copy to do it.
    """
    return apply_scope(df, data_filter)


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


def r_scale(M: np.ndarray) -> np.ndarray:
    """Z-score columns exactly as R's ``scale()`` does, to the last bit.

    R sums sequentially in long double (``colMeans``, ``sum``), so tied
    distances computed from standardized data stay tied the same way they do
    in R, which decides the merge order of ``hclust``. Zero-variance columns
    are left centered (divided by 1).
    """
    M = np.asarray(M, dtype=float)
    if M.ndim == 1:
        return r_scale(M[:, None])[:, 0]
    n = M.shape[0]
    mu = (np.cumsum(M.astype(np.longdouble), axis=0)[-1] / n).astype(float)
    X = M - mu
    ss = np.cumsum((X * X).astype(np.longdouble), axis=0)[-1].astype(float)
    sd = np.sqrt(ss / max(1, n - 1))
    return X / np.where(sd == 0, 1.0, sd)


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


def _gpa(A, vgQ, oblique=False, eps=1e-5, maxit=2000, fwindow=10):
    """Gradient-projection rotation, reproducing ``GPArotation::GPForth`` /
    ``GPFoblq`` (Barzilai-Borwein step with a non-monotone line search, the
    package default ``algorithm = "bb"``).

    Returns ``(rotated_loadings, Tmat)``; for an oblique rotation the factor
    correlations are ``Tmat.T @ Tmat``.
    """
    A = np.asarray(A, dtype=float)
    k = A.shape[1]
    Tmat = np.eye(k)

    def evaluate(T):
        if oblique:
            Tinv = np.linalg.inv(T)
            L = A @ Tinv.T
            f, Gq = vgQ(L)
            return L, f, -((L.T @ Gq @ Tinv).T)
        L = A @ T
        f, Gq = vgQ(L)
        return L, f, A.T @ Gq

    L, f, G = evaluate(Tmat)
    alpha = 1.0
    T_prev = Gp_prev = None
    fs = []
    for _ in range(maxit + 1):
        if oblique:
            Gp = G - Tmat @ np.diag(np.sum(Tmat * G, axis=0))
        else:
            M = Tmat.T @ G
            Gp = G - Tmat @ ((M + M.T) / 2)
        s = np.sqrt(np.sum(Gp**2))
        fs.append(f)
        if s < eps:
            break
        if T_prev is not None:
            dT = Tmat - T_prev
            dGp = Gp - Gp_prev
            if np.sum(dGp**2) > 0:
                alpha = np.sum(dT**2) / abs(np.sum(dT * dGp))
                alpha = max(1e-10, min(alpha, 20.0))
        else:
            alpha = 2 * alpha
        target_f = max(fs[-fwindow:])
        for _ in range(11):
            X = Tmat - alpha * Gp
            if oblique:
                Tt = X / np.sqrt(np.sum(X**2, axis=0))
            else:
                U, _d, Vt = np.linalg.svd(X)
                Tt = U @ Vt
            Lt, ft, Gt = evaluate(Tt)
            if (target_f - ft) > 0.5 * s**2 * alpha:
                break
            alpha /= 2
        T_prev, Gp_prev = Tmat, Gp
        Tmat, L, f, G = Tt, Lt, ft, Gt
    return L, Tmat


def _vgQ_varimax(L):
    QL = L**2 - (L**2).mean(axis=0)
    return -np.sum(QL**2) / 4.0, -L * QL


def _vgQ_quartimax(L):
    return -np.sum(L**4) / 4.0, -(L**3)


def _vgQ_cf(L, kappa=0.0):
    """Crawford-Ferguson family (equamax is ``kappa = k / (2p)``)."""
    L2 = L**2
    L2N = L2.sum(axis=1, keepdims=True) - L2
    ML2 = L2.sum(axis=0, keepdims=True) - L2
    f = (1 - kappa) * np.sum(L2 * L2N) / 4 + kappa * np.sum(L2 * ML2) / 4
    return f, (1 - kappa) * L * L2N + kappa * L * ML2


def _vgQ_oblimin(L, gam=0.0):
    X = (L**2).sum(axis=1, keepdims=True) - L**2
    if gam != 0:
        X = X - (gam / L.shape[0]) * X.sum(axis=0)
    return np.sum(L**2 * X) / 4.0, L * X


def _vgQ_simplimax(L, k_target=None):
    p, k = L.shape
    if k_target is None:
        k_target = p * (k - 1)
    L2 = L**2
    # R's order() over the column-major matrix, ties broken by position
    idx = np.argsort(L2.flatten(order="F"), kind="stable")[:k_target]
    Imat = np.zeros(L2.size)
    Imat[idx] = 1.0
    Imat = Imat.reshape(L2.shape, order="F")
    return np.sum(L2 * Imat), 2 * Imat * L


def _promax(L, m=4, gpa_varimax=False):
    """Promax: varimax, then a least-squares fit to the power target.

    ``gpa_varimax=False`` reproduces ``stats::promax`` (used by
    ``psych::principal``); ``True`` reproduces ``psych::Promax`` (used by
    ``psych::fa``), which starts from ``GPArotation::Varimax``.
    Returns ``(loadings, Phi)``.
    """
    if gpa_varimax:
        x, Th = _gpa(L, _vgQ_varimax)
    else:
        x, Th = varimax(L)
    Q = x * np.abs(x) ** (m - 1)
    U = np.linalg.lstsq(x, Q, rcond=None)[0]
    d = np.diag(np.linalg.inv(U.T @ U))
    U = U @ np.diag(np.sqrt(d))
    z = x @ U
    ui = np.linalg.inv(Th @ U)
    Phi = ui @ ui.T
    sd = np.sqrt(np.diag(Phi))
    return z, Phi / np.outer(sd, sd)


ROTATIONS = ("none", "varimax", "quartimax", "equamax", "promax", "oblimin", "simplimax")


def rotate_loadings(loadings: np.ndarray, rotation: str = "varimax", fm: str = "pc"):
    """Rotate a loadings matrix the way ``psych`` does.

    ``fm="pc"`` follows ``psych::principal`` and ``fm="ml"`` follows
    ``psych::fa``; they differ only for promax (``stats::promax`` versus
    Kaiser-normalized ``psych::Promax``). Varimax is ``stats::varimax``; the
    other rotations use GPArotation's gradient-projection algorithm with its
    default settings. Returns ``(rotated_loadings, Phi)``, where ``Phi`` holds
    the factor correlations for oblique rotations and is ``None`` otherwise.
    """
    L = np.asarray(loadings, dtype=float)
    p, k = L.shape
    if k < 2 or rotation in ("none", None, ""):
        return L, None
    if rotation == "varimax":
        return varimax(L)[0], None
    if rotation == "quartimax":
        return _gpa(L, _vgQ_quartimax)[0], None
    if rotation == "equamax":
        return _gpa(L, lambda x: _vgQ_cf(x, kappa=k / (2 * p)))[0], None
    if rotation == "promax":
        if fm == "pc":
            return _promax(L)
        # psych::kaiser: row-normalize, Promax, then scale back
        h = np.sqrt((L**2).sum(axis=1))
        out, Phi = _promax(L / h[:, None], gpa_varimax=True)
        return out * h[:, None], Phi
    if rotation in ("oblimin", "simplimax"):
        vgQ = _vgQ_oblimin if rotation == "oblimin" else _vgQ_simplimax
        out, Tmat = _gpa(L, vgQ, oblique=True)
        return out, Tmat.T @ Tmat
    raise ValueError(f"Unsupported rotation: {rotation}")


def _sign_flip(L):
    """psych's sign convention: make every column sum non-negative."""
    signs = np.sign(L.sum(axis=0))
    signs[signs == 0] = 1.0
    return signs


def principal(cmat: np.ndarray, nfactors: int = 1, rotate: str = "varimax") -> dict:
    """Principal-components factor analysis, reproducing ``psych::principal``.

    Parameters
    ----------
    cmat : np.ndarray
        Correlation matrix (p x p).
    nfactors : int
        Number of components to retain.
    rotate : str
        One of ``ROTATIONS``.

    Returns a dict with ``loadings`` (p x nfactors), ``communality``,
    ``uniqueness``, ``values`` (all p eigenvalues of ``cmat``), ``Phi`` (factor
    correlations for oblique rotations, else ``None``) and ``names`` (psych's
    column names: ``RC*`` for orthogonal rotations and promax, ``PC*``
    otherwise, permuted along with the columns when they are reordered).
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
    pos = np.clip(vals[:nfactors], np.finfo(float).eps, None)
    loadings = vecs[:, :nfactors] * np.sqrt(pos)
    loadings = loadings * _sign_flip(loadings)

    # communalities come from the unrotated solution, so they are rotation
    # invariant even for oblique pattern matrices
    communality = (loadings**2).sum(axis=1)
    uniqueness = np.diag(cmat) - communality

    names = [f"PC{i + 1}" for i in range(nfactors)]
    Phi = None
    if nfactors > 1 and rotate not in ("none", None, ""):
        if rotate in ("varimax", "quartimax", "equamax", "promax"):
            names = [f"RC{i + 1}" for i in range(nfactors)]
        loadings, Phi = rotate_loadings(loadings, rotate, fm="pc")

    # order components by (rotated) sum of squared loadings, then flip signs
    col_order = np.argsort(-(loadings**2).sum(axis=0), kind="stable")
    loadings = loadings[:, col_order]
    names = [names[i] for i in col_order]
    signs = _sign_flip(loadings)
    loadings = loadings * signs
    if Phi is not None:
        Phi = Phi[np.ix_(col_order, col_order)] * np.outer(signs, signs)

    return {
        "loadings": loadings,
        "communality": communality,
        "uniqueness": uniqueness,
        "values": vals,
        "Phi": Phi,
        "names": names,
    }


def ml_factor(cmat: np.ndarray, nfactors: int, rotate: str = "varimax") -> dict:
    """Maximum-likelihood factor analysis, reproducing ``psych::fa(fm="ml")``.

    Uniquenesses minimize the ML objective with the same L-BFGS-B setup as
    psych (start ``1 - smc``, bounds ``[0.005, 1]``, analytic gradient,
    ``parscale = 0.01``). Loadings are sign-flipped, named ``ML1..MLk``,
    rotated, sign-flipped again and ordered by explained variance
    (``diag(Phi L'L)`` for oblique rotations), with the names following their
    columns -- exactly as psych prints them.
    """
    from scipy.optimize import minimize

    R = np.asarray(cmat, dtype=float)
    p = R.shape[0]
    nf = int(max(1, min(nfactors, p - 1)))

    def eig_scaled(psi):
        sc = 1.0 / np.sqrt(psi)
        vals, vecs = np.linalg.eigh(sc[:, None] * R * sc[None, :])
        order = np.argsort(vals)[::-1]
        return vals[order], vecs[:, order]

    def loadings_from_psi(psi):
        vals, vecs = eig_scaled(psi)
        L = vecs[:, :nf] * np.sqrt(np.maximum(vals[:nf] - 1.0, 0.0))
        return np.sqrt(psi)[:, None] * L

    # optimize over psi / parscale, as R's optim does
    scale = 0.01

    def objective(x):
        e = eig_scaled(x * scale)[0][nf:]
        return -(np.sum(np.log(e) - e) - nf + p)

    def gradient(x):
        psi = x * scale
        L = loadings_from_psi(psi)
        g = L @ L.T + np.diag(psi) - R
        return np.diag(g) / psi**2 * scale

    smc = 1.0 - 1.0 / np.diag(np.linalg.inv(R))
    start = np.diag(R) - smc
    upper = max(smc.max(), 1.0)
    res = minimize(
        objective,
        start / scale,
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(0.005 / scale, upper / scale)] * p,
        options={"maxiter": 100, "maxcor": 5, "ftol": 1e7 * np.finfo(float).eps, "gtol": 0.0},
    )
    psi = res.x * scale
    loadings = loadings_from_psi(psi)
    loadings = loadings * _sign_flip(loadings)

    # communalities/uniquenesses from the unrotated model (rotation invariant)
    communality = (loadings**2).sum(axis=1)
    uniqueness = np.diag(R) - communality
    S = R.copy()
    np.fill_diagonal(S, communality)
    allvals = np.sort(np.linalg.eigvalsh(S))[::-1]

    Phi = None
    if nf > 1 and rotate not in ("none", None, ""):
        loadings, Phi = rotate_loadings(loadings, rotate, fm="ml")

    signs = _sign_flip(loadings)
    loadings = loadings * signs
    names = [f"ML{i + 1}" for i in range(nf)]
    if Phi is not None:
        Phi = Phi * np.outer(signs, signs)
        ev = np.diag(Phi @ loadings.T @ loadings)
    else:
        ev = (loadings**2).sum(axis=0)
    col_order = np.argsort(-ev, kind="stable")
    loadings = loadings[:, col_order]
    names = [names[i] for i in col_order]
    if Phi is not None:
        Phi = Phi[np.ix_(col_order, col_order)]

    return {
        "loadings": loadings,
        "communality": communality,
        "uniqueness": uniqueness,
        "values": allvals,
        "psi": psi,
        "Phi": Phi,
        "names": names,
    }


def gower_dist(
    data: pl.DataFrame, vars: list[str], standardize: bool = True
) -> np.ndarray:
    """Condensed Gower distance vector for mixed numeric/categorical data.

    Reproduces ``gower::gower_dist`` as used by ``radiant.multivariate::hclus``:
    numeric variables are range-normalized absolute differences (optionally
    standardized first with R's ``scale``, matching Radiant), categorical
    variables contribute the simple-matching distance (0 if equal, 1
    otherwise), and the per-pair distance is the unweighted mean across
    variables. As in gower's C code the similarities are summed and the
    distance is ``1 - mean similarity``, so ties fall the same way as in R.

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
        num_cols = [r_scale(arr) for arr in num_cols]

    # variables are summed in their selected order, like gower's C loop
    nvar = 0
    S = np.zeros((n, n))
    for v in vars:
        s = data[v]
        if is_categorical(s):
            arr = cat_cols.pop(0)
            S += (arr[:, None] == arr[None, :]).astype(float)
            nvar += 1
        else:
            arr = num_cols.pop(0)
            rng = arr.max() - arr.min()
            # a constant variable: every pair is fully similar
            S += 1.0 if rng == 0 else 1.0 - np.abs(arr[:, None] - arr[None, :]) / rng
            nvar += 1
    D = 1.0 - S / max(nvar, 1)
    np.fill_diagonal(D, 0.0)
    return squareform(D, checks=False)


def factor_scores(std_data: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """PCA factor scores: ``std_data @ (L @ inv(L'L))`` (radiant's ``cscm``)."""
    L = np.asarray(loadings, dtype=float)
    cscm = L @ np.linalg.inv(L.T @ L)
    return np.asarray(std_data, dtype=float) @ cscm
