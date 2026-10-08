"""Correlation strategy for the multivariate factor / map ports.

Mirrors what ``radiant.multivariate`` does through ``polycor::hetcor``: when
``hcor=True`` categorical columns are treated as ordinal and a heterogeneous
correlation matrix is built. The ordinal/ordinal cells reuse the existing
``pyrsm.utils.polychoric_corr`` implementation (already validated against
``polycor::polychor`` and used by ``pyrsm.design.doe``), so there is a single
canonical polychoric routine across the package.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pyrsm.utils import polychoric_corr
from pyrsm.utils import polyserial_corr as _polyserial_corr

from ._utils import as_numeric_matrix, is_categorical

__all__ = ["heterogeneous_corr", "polyserial_corr"]


def polyserial_corr(x_num: np.ndarray, y_codes: np.ndarray) -> float:
    """Two-step polyserial correlation; the one routine in :mod:`pyrsm.utils`
    (``polycor::polyserial``, population sd of the codes as polycor uses)."""
    return _polyserial_corr(x_num, y_codes)


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
    return cmat, any_categorical
