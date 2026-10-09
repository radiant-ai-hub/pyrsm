"""Correlation strategy for the multivariate factor / map ports.

Mirrors what ``radiant.multivariate`` does through ``polycor::hetcor``: when
``hcor=True`` categorical columns are treated as ordinal and a heterogeneous
correlation matrix is built. The ordinal/ordinal cells reuse the existing
``pyrsm.utils.polychoric_corr`` implementation (already validated against
``polycor::polychor`` and used by ``pyrsm.design.doe``) and the numeric/ordinal
cells reuse ``pyrsm.utils.polyserial_corr``, so there is a single canonical
routine for each across the package.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from pyrsm.utils import polychoric_corr, polyserial_corr

from ._utils import as_numeric_matrix, is_categorical

__all__ = ["heterogeneous_corr", "polyserial_corr", "near_pd_corr"]


def near_pd_corr(
    x: np.ndarray,
    eig_tol: float = 1e-6,
    conv_tol: float = 1e-7,
    posd_tol: float = 1e-8,
    maxit: int = 100,
):
    """Nearest positive-definite correlation matrix.

    Port of ``Matrix::nearPD(x, corr = TRUE)`` with its defaults (Higham's
    alternating projections with Dykstra's correction, then an eigenvalue
    floor), which is what ``polycor::hetcor`` applies when the pairwise
    estimates do not form a positive-definite matrix. Returns
    ``(matrix, converged)``.
    """
    X = (np.asarray(x, dtype=float) + np.asarray(x, dtype=float).T) / 2
    n = X.shape[0]
    D_S = np.zeros_like(X)
    converged = False
    for _ in range(maxit):
        Y = X
        R = Y - D_S
        d, Q = np.linalg.eigh(R)
        d, Q = d[::-1], Q[:, ::-1]
        keep = d > eig_tol * d[0]
        if not keep.any():
            raise ValueError("Matrix seems negative semi-definite")
        Q = Q[:, keep]
        X = (Q * d[keep]) @ Q.T
        D_S = X - R
        np.fill_diagonal(X, 1.0)
        conv = np.abs(Y - X).sum(axis=1).max() / np.abs(Y).sum(axis=1).max()
        if conv <= conv_tol:
            converged = True
            break

    d, Q = np.linalg.eigh(X)
    d, Q = d[::-1], Q[:, ::-1]
    eps = posd_tol * abs(d[0])
    if d[-1] < eps:
        d = np.where(d < eps, eps, d)
        o_diag = np.diag(X).copy()
        X = (Q * d) @ Q.T
        D = np.sqrt(np.maximum(eps, o_diag) / np.diag(X))
        X = D[:, None] * X * D[None, :]
    np.fill_diagonal(X, 1.0)
    return X, converged


def heterogeneous_corr(data: pl.DataFrame, vars: list[str], hcor: bool = False):
    """Build a correlation matrix for ``vars``.

    Parameters
    ----------
    data : pl.DataFrame
    vars : list[str]
    hcor : bool
        When False (or no categorical columns are present) a plain Pearson
        correlation matrix is returned. When True, categorical columns are
        treated as ordinal: ordinal/ordinal cells use polychoric correlation,
        numeric/ordinal cells use polyserial correlation, and numeric/numeric
        cells use Pearson.

    Returns
    -------
    (cmat, any_categorical) : tuple[np.ndarray, bool]
    """
    cat_flags = [is_categorical(data[v]) for v in vars]
    any_categorical = any(cat_flags)
    M = as_numeric_matrix(data, vars)

    if not hcor or not any_categorical:
        return np.corrcoef(M, rowvar=False), any_categorical

    p = len(vars)
    cmat = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            xi, xj = M[:, i], M[:, j]
            ci, cj = cat_flags[i], cat_flags[j]
            if ci and cj:
                r = polychoric_corr(xi, xj)
            elif ci and not cj:
                r = polyserial_corr(xj, xi)
            elif cj and not ci:
                r = polyserial_corr(xi, xj)
            else:
                r = np.corrcoef(xi, xj)[0, 1]
            cmat[i, j] = cmat[j, i] = r

    # pairwise estimates need not form a valid correlation matrix; repair it
    # the way polycor::hetcor does
    if np.linalg.eigvalsh(cmat).min() < 0:
        cmat, converged = near_pd_corr(cmat)
        warnings.warn(
            "the correlation matrix has been adjusted to make it positive-definite"
            if converged
            else "attempt to make correlation matrix positive-definite failed",
            stacklevel=3,
        )
    return cmat, any_categorical
