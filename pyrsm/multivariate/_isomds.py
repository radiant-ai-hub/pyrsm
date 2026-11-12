"""Kruskal's non-metric MDS as ``MASS::isoMDS`` computes it.

A line-by-line port of MASS's C code (``VR_mds_fn``, ``VR_mds_dovm``) and of
R's BFGS optimizer ``vmmin`` (src/appl/optim.c, also ``optim(method = "BFGS")``),
which isoMDS calls with ``abstol = 1e-2`` (stress in percent) and the caller's
``maxit`` and ``tol``. The map starts from the classical solution and keeps its
scale, so the recovered distances are on the scale of the dissimilarities, as
in radiant.multivariate. Kruskal stress itself does not depend on the scale, so
an optimizer that rescales the map (sklearn's ``smacof`` normalizes the
disparities to a sum of squares of n(n-1)/2) gives the same stress for a map
whose distances are around 1, not in the data's units.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# vmmin's constants (src/appl/optim.c)
#
# Sums run in the C code's order (np.cumsum is a sequential sum, unlike
# np.sum's pairwise one) and the BFGS matrix is updated as its lower triangle,
# as vmmin stores it. Near isoMDS's stopping rule (stress below 0.01%, the
# city data) the last bits decide when it stops, so this is what makes the
# port give R's map, not just a map like it.
_STEPREDN = 0.2
_ACCTOL = 0.0001
_RELTEST = 10.0


def _seqsum(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """``s = 0; for v in a: s += v`` along ``axis``, as the C loops add."""
    return np.cumsum(a, axis=axis).take(-1, axis=axis)


def vmmin(
    b: np.ndarray,
    fn: Callable[[np.ndarray], float],
    gr: Callable[[np.ndarray], np.ndarray],
    maxit: int = 100,
    abstol: float = -np.inf,
    reltol: float = 1.490116e-08,
) -> tuple[np.ndarray, float, int, int, int]:
    """R's BFGS variable-metric minimizer (``vmmin``).

    Returns ``(par, value, fncount, grcount, fail)`` as ``optim`` reports
    them: ``fail`` is 1 when ``maxit`` iterations were used.
    """
    b = np.array(b, dtype=float)
    n = b.size
    if maxit <= 0:
        return b, float(fn(b)), 0, 0, 0
    f = float(fn(b))
    if not np.isfinite(f):
        raise ValueError("initial value in 'vmmin' is not finite")
    fmin = f
    funcount = gradcount = 1
    g = np.asarray(gr(b), dtype=float)
    it = 1
    ilast = gradcount
    L = np.eye(n)  # the lower triangle is the BFGS matrix

    def full(L):
        return np.tril(L) + np.tril(L, -1).T

    while True:
        if ilast == gradcount:
            L = np.eye(n)
        B = full(L)
        X = b.copy()
        c = g.copy()
        # s = 0; s -= B[i][j] * g[j] ... equals -(sum) exactly (negation is exact)
        t = -_seqsum(B * g[None, :], axis=1)
        gradproj = float(_seqsum(t * g))
        if gradproj < 0.0:  # downhill
            steplength = 1.0
            accpoint = False
            while True:
                b = X + steplength * t
                count = int(np.count_nonzero(_RELTEST + X == _RELTEST + b))
                if count < n:
                    f = float(fn(b))
                    funcount += 1
                    accpoint = bool(np.isfinite(f)) and f <= fmin + gradproj * steplength * _ACCTOL
                    if not accpoint:
                        steplength *= _STEPREDN
                if count == n or accpoint:
                    break
            enough = f > abstol and abs(f - fmin) > reltol * (abs(fmin) + reltol)
            if not enough:  # value small, or relative change low
                count = n
                fmin = f
            if count < n:  # making progress
                fmin = f
                g = np.asarray(gr(b), dtype=float)
                gradcount += 1
                it += 1
                t = steplength * t
                c = g - c
                D1 = float(_seqsum(t * c))
                if D1 > 0:
                    Xc = _seqsum(B * c[None, :], axis=1)
                    D2 = 1.0 + float(_seqsum(Xc * c)) / D1
                    # B[i][j] += (D2 * t[i] * t[j] - X[i] * t[j] - t[i] * X[j]) / D1, j <= i
                    upd = ((D2 * t)[:, None] * t[None, :] - Xc[:, None] * t[None, :]
                           - t[:, None] * Xc[None, :]) / D1
                    L = L + np.tril(upd)
                else:
                    ilast = gradcount
            elif ilast < gradcount:  # no progress
                count = 0
                ilast = gradcount
        else:  # uphill: reset unless just reset
            count = 0
            if ilast == gradcount:
                count = n
            else:
                ilast = gradcount
        if it >= maxit:
            break
        if gradcount - ilast > 2 * n:
            ilast = gradcount  # periodic restart
        if count == n and ilast == gradcount:
            break
    return b, fmin, funcount, gradcount, int(it >= maxit)


def isotonic_fit(y: np.ndarray) -> np.ndarray:
    """Least-squares non-decreasing fit to ``y`` (MASS's greatest convex
    minorant of the cumulative sums, first minimum on ties)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    yc = np.concatenate([[0.0], np.cumsum(y)])
    yf = np.empty(n)
    known = 0
    while known < n:
        idx = np.arange(known + 1, n + 1)
        slopes = (yc[idx] - yc[known]) / (idx - known)
        ip = int(idx[np.argmin(slopes)])
        yf[known:ip] = (yc[ip] - yc[known]) / (ip - known)
        known = ip
    return yf


class _Kruskal:
    """Stress (percent) and its gradient for a configuration, as VR_mds_fn."""

    def __init__(self, D: np.ndarray, k: int):
        self.nr = D.shape[0]
        self.k = k
        # R: dis <- x[row(x) > col(x)]; column-major lower triangle = the
        # (r1 < r2) row-major order calc_dist uses
        self.r1, self.r2 = np.triu_indices(self.nr, 1)
        dis = D[self.r1, self.r2]
        self.ord = np.argsort(dis, kind="stable")  # R's order() is stable
        self.rank = np.argsort(self.ord, kind="stable")  # order(ord) - 1

    def _config(self, x: np.ndarray) -> np.ndarray:
        return x.reshape((self.nr, self.k), order="F")  # R stores column-major

    def _parts(self, x: np.ndarray):
        X = self._config(x)
        diff = X[self.r1] - X[self.r2]
        tmp = np.zeros(diff.shape[0])
        for col in range(self.k):  # calc_dist: tmp += tmp1 * tmp1 per column
            tmp = tmp + diff[:, col] * diff[:, col]
        d = np.sqrt(tmp)
        y = d[self.ord]
        yf = isotonic_fit(y)
        res = y - yf
        sstar = float(_seqsum(res * res))
        tstar = float(_seqsum(y * y))
        return X, y, yf, sstar, tstar, 100 * np.sqrt(sstar / tstar)

    def stress(self, x: np.ndarray) -> float:
        return float(self._parts(x)[-1])

    def gradient(self, x: np.ndarray) -> np.ndarray:
        X, y, yf, sstar, tstar, ssq = self._parts(x)
        if sstar == 0.0:  # a perfect fit: nothing to improve
            return np.zeros_like(x)
        # VR_mds_fn: for each point u and dimension i, add over s != u (in
        # order) ((y - yf) / sstar - y / tstar) * sgn * (|x_u - x_s| / y)
        a = ((y - yf) / sstar - y / tstar)[self.rank]  # per pair
        yk = y[self.rank]
        A = np.zeros((self.nr, self.nr))
        Y = np.ones((self.nr, self.nr))
        A[self.r1, self.r2] = A[self.r2, self.r1] = a
        Y[self.r1, self.r2] = Y[self.r2, self.r1] = yk
        der = np.empty((self.nr, self.k))
        for col in range(self.k):
            diff = X[:, col][:, None] - X[:, col][None, :]
            sgn = np.where(diff >= 0, 1.0, -1.0)
            terms = A * sgn * (np.abs(diff) / Y)
            np.fill_diagonal(terms, 0.0)  # s == u is skipped; adding 0.0 is exact
            der[:, col] = _seqsum(terms, axis=1) * ssq
        return der.ravel(order="F")


def isomds(
    D: np.ndarray, init: np.ndarray, maxit: int = 50, tol: float = 1e-3
) -> tuple[np.ndarray, float]:
    """``MASS::isoMDS(D, y = init, k = ncol(init), maxit, tol)`` for a full
    symmetric dissimilarity matrix ``D``.

    Returns the configuration and the stress in percent, as isoMDS does
    (radiant divides it by 100).
    """
    D = np.asarray(D, dtype=float)
    init = np.asarray(init, dtype=float)
    nr, k = init.shape
    off = D[np.triu_indices(nr, 1)]
    if (off <= 0).any():
        i, j = (int(v[0]) + 1 for v in np.nonzero(np.triu(D <= 0, 1)))
        raise ValueError(f"zero or negative distance between objects {i} and {j}")
    obj = _Kruskal(D, k)
    x, val, *_ = vmmin(
        init.ravel(order="F"), obj.stress, obj.gradient, maxit=maxit, abstol=1e-2, reltol=tol
    )
    return obj._config(x).copy(), float(val)
