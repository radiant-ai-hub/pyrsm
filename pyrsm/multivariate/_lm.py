"""R-compatible linear-model fitting for the multivariate ports.

R's ``lm()`` drops aliased (perfectly collinear) columns -- e.g. a saturated
conjoint design within a ``by`` group -- and reports them as ``NA``, where
statsmodels would spread the effect over the columns with a pseudo-inverse.
These helpers reproduce R's choice of columns (LINPACK ``dqrdc2`` pivoting).
"""

from __future__ import annotations

import numpy as np

__all__ = ["r_qr_rank", "fit_like_r", "collinearity_note"]

COLLINEARITY_NOTE = (
    "The set of explanatory variables exhibit perfect multicollinearity.\n"
    "One or more variables were dropped from the estimation"
)


def collinearity_note(fitted) -> str:
    """R's message about perfect multicollinearity, naming the dropped terms."""
    aliased = getattr(fitted, "aliased", [])
    if not aliased:
        return ""
    names = ", ".join(a.replace("[T.", "[") for a in aliased)
    return f"{COLLINEARITY_NOTE}: {names}"


def r_qr_rank(x: np.ndarray, tol: float = 1e-7) -> tuple[int, np.ndarray]:
    """Rank and column pivot of ``x`` from R's LINPACK ``dqrdc2``.

    Columns are processed left to right; one whose norm, after removing the
    part explained by the columns kept so far, is below ``tol`` times its
    original norm is moved to the end. Returns ``(rank, pivot)``;
    ``pivot[rank:]`` are the aliased columns, exactly as ``lm()$qr$pivot``.
    """
    x = np.array(x, dtype=float, copy=True)
    n, p = x.shape
    qraux = np.linalg.norm(x, axis=0)
    work1 = qraux.copy()
    work2 = qraux.copy()
    work2[work2 == 0.0] = 1.0
    jpvt = np.arange(p)
    k = p + 1  # one past the last non-negligible column (1-based, as in Fortran)
    for l in range(min(n, p)):
        while (l + 1) < k and qraux[l] < work2[l] * tol:
            # cycle column l to the end
            x[:, l:] = np.concatenate([x[:, l + 1 :], x[:, l : l + 1]], axis=1)
            for arr in (jpvt, qraux, work1, work2):
                arr[l:] = np.concatenate([arr[l + 1 :], arr[l : l + 1]])
            k -= 1
        if l == n - 1:
            break
        nrmxl = np.linalg.norm(x[l:, l])
        if nrmxl == 0.0:
            continue
        if x[l, l] != 0.0:
            nrmxl = np.copysign(nrmxl, x[l, l])
        x[l:, l] /= nrmxl
        x[l, l] += 1.0
        for j in range(l + 1, p):
            t = -np.dot(x[l:, l], x[l:, j]) / x[l, l]
            x[l:, j] += t * x[l:, l]
            if qraux[j] != 0.0:
                tt = max(1.0 - (abs(x[l, j]) / qraux[j]) ** 2, 0.0)
                if abs(tt) < 1e-6:
                    qraux[j] = np.linalg.norm(x[l + 1 :, j])
                    work1[j] = qraux[j]
                else:
                    qraux[j] = qraux[j] * np.sqrt(tt)
        qraux[l] = x[l, l]
        x[l, l] = -nrmxl
    return min(k - 1, n), jpvt


def fit_like_r(model):
    """Fit a statsmodels OLS formula model, dropping aliased columns as R's
    ``lm()`` does (QR tolerance 1e-7).

    The columns R would keep are estimated on their own; the result is then
    expressed on the full design with a coefficient of 0 (and zero variance)
    for every aliased column. Predictions, confidence intervals and tests
    from statsmodels therefore work as usual and equal R's (R predicts with
    the NA coefficients treated as 0), while the coefficient table should
    leave out the names in ``results.aliased``.

    Parameters
    ----------
    model : statsmodels OLS model created with the formula API (smf.ols)

    Returns
    -------
    statsmodels results with an extra attribute ``aliased``: the names of the
    dropped columns (empty when the design has full rank).
    """
    import statsmodels.api as sma
    from statsmodels.regression.linear_model import (
        OLSResults,
        RegressionResultsWrapper,
    )

    X = np.asarray(model.exog, dtype=float)
    rank, piv = r_qr_rank(X, 1e-7)
    names = list(model.exog_names)
    if rank == X.shape[1]:
        full = model.fit()
        full.aliased = []
        return full
    import warnings

    with warnings.catch_warnings():
        # the full (singular) fit only sets up the model's internal state;
        # its coefficients are replaced below
        warnings.simplefilter("ignore")
        model.fit()

    drop = sorted(int(i) for i in piv[rank:])
    keep = [i for i in range(X.shape[1]) if i not in drop]
    k = X.shape[1]
    sub = sma.OLS(model.endog, X[:, keep]).fit()
    params = np.zeros(k)
    params[keep] = sub.params
    ncov = np.zeros((k, k))
    ncov[np.ix_(keep, keep)] = sub.normalized_cov_params
    model.rank = len(keep)
    model.df_model = len(keep) - model.k_constant
    model.df_resid = model.nobs - len(keep)
    res = RegressionResultsWrapper(OLSResults(model, params, normalized_cov_params=ncov))
    res.aliased = [names[i] for i in drop]
    return res
