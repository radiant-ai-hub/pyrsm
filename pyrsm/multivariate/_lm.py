"""R-compatible linear-model fitting for the multivariate ports.

R's ``lm()`` drops aliased (perfectly collinear) columns -- e.g. a saturated
conjoint design within a ``by`` group -- and reports them as ``NA``, where
statsmodels would spread the effect over the columns with a pseudo-inverse.
These helpers reproduce R's choice of columns (LINPACK ``dqrdc2`` pivoting).
"""

from __future__ import annotations

import numpy as np

__all__ = ["r_qr_rank", "r_column_order", "fit_like_r", "glm_like_r", "collinearity_note"]

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


def r_column_order(model) -> list[int]:
    """Columns of a statsmodels formula model in R's ``model.matrix`` order.

    patsy puts terms with only categorical factors before numeric ones
    (``price ~ carat + clarity`` gives Intercept, clarity[...], carat); R keeps
    the formula order and sorts terms by their order (main effects, then
    two-way interactions, ...). The order decides which of two collinear
    columns ``lm()`` drops, so the QR must see R's order. Falls back to the
    model's own order when there is no formula.
    """
    k = np.asarray(model.exog).shape[1]
    design = getattr(getattr(model, "data", None), "design_info", None)
    formula = getattr(model, "formula", None)
    if design is None or not formula:
        return list(range(k))
    try:
        from patsy import ModelDesc

        written = list(ModelDesc.from_formula(formula).rhs_termlist)
    except Exception:  # noqa: BLE001 - unusual formula: keep patsy's order
        return list(range(k))

    def position(term):
        return written.index(term) if term in written else len(written)

    terms = sorted(design.terms, key=lambda t: (len(t.factors), position(t)))
    order: list[int] = []
    for term in terms:
        sl = design.term_slices[term]
        order.extend(range(sl.start, sl.stop))
    return order if sorted(order) == list(range(k)) else list(range(k))


def fit_like_r(model, cov_type: str = "nonrobust"):
    """Fit a statsmodels OLS formula model, dropping aliased columns as R's
    ``lm()`` does (QR tolerance 1e-7, columns in R's ``model.matrix`` order).

    The columns R would keep are estimated on their own; the result is then
    expressed on the full design with a coefficient of 0 (and zero variance)
    for every aliased column. Predictions, confidence intervals and tests
    from statsmodels therefore work as usual and equal R's (R predicts with
    the NA coefficients treated as 0), while the coefficient table should
    leave out the names in ``results.aliased``.

    Parameters
    ----------
    model : statsmodels OLS model created with the formula API (smf.ols)
    cov_type : "nonrobust" (default) or a statsmodels robust type such as
        "HC1" (R's ``sandwich::vcovHC(type = "HC1")``), computed on the kept
        columns with t-based inference.

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
    names = list(model.exog_names)
    order = r_column_order(model)
    rank, piv = r_qr_rank(X[:, order], 1e-7)
    robust = cov_type != "nonrobust"
    if rank == X.shape[1]:
        full = model.fit(cov_type=cov_type, use_t=True) if robust else model.fit()
        full.aliased = []
        return full
    import warnings

    with warnings.catch_warnings():
        # the full (singular) fit only sets up the model's internal state;
        # its coefficients are replaced below
        warnings.simplefilter("ignore")
        model.fit()

    drop = sorted(order[int(i)] for i in piv[rank:])
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
    res = OLSResults(model, params, normalized_cov_params=ncov)
    if robust:
        cov = np.zeros((k, k))
        cov[np.ix_(keep, keep)] = np.asarray(
            sma.OLS(model.endog, X[:, keep]).fit(cov_type=cov_type, use_t=True).cov_params()
        )
        res.cov_params_default = cov
        res.cov_type = cov_type
        res.use_t = True
    res = RegressionResultsWrapper(res)
    res.aliased = [names[i] for i in drop]
    return res


def glm_like_r(model, robust: bool = False):
    """Fit a statsmodels GLM formula model, dropping aliased columns as R's
    ``glm()`` does (NA coefficients), in R's ``model.matrix`` column order.

    As :func:`fit_like_r`: the kept columns are estimated on their own and the
    result is expressed on the full design with 0 (and zero variance) for each
    aliased column, so predictions work as usual; ``results.aliased`` names
    the dropped columns. statsmodels' GLM already counts df from the rank.

    ``robust=True`` gives R's ``sandwich::vcovHC(type = "HC1")`` covariance for
    a canonical-link GLM with dispersion 1 (logistic regression), see
    :func:`_canonical_hc1`.
    """
    import statsmodels.api as sma
    from statsmodels.genmod.generalized_linear_model import (
        GLMResults,
        GLMResultsWrapper,
    )

    X = np.asarray(model.exog, dtype=float)
    names = list(model.exog_names)
    order = r_column_order(model)
    rank, piv = r_qr_rank(X[:, order], 1e-7)
    k = X.shape[1]
    drop = sorted(order[int(i)] for i in piv[rank:]) if rank < k else []
    keep = [i for i in range(k) if i not in drop]

    if not drop:
        full = model.fit()
        if robust:
            res = full._results
            res.cov_params_default = _canonical_hc1(model, X, full.mu, full.normalized_cov_params)
            res.cov_type = "HC1"
            res._cache = {}
        full.aliased = []
        return full

    import warnings

    with warnings.catch_warnings():
        # the full (singular) fit only sets up the model's internal state
        warnings.simplefilter("ignore")
        model.fit()

    weights = getattr(model, "freq_weights", None)
    sub = sma.GLM(
        model.endog,
        X[:, keep],
        family=model.family,
        freq_weights=weights,
        offset=getattr(model, "offset", None),
    ).fit()
    params = np.zeros(k)
    params[keep] = sub.params
    ncov = np.zeros((k, k))
    ncov[np.ix_(keep, keep)] = sub.normalized_cov_params
    res = GLMResults(model, params, ncov, sub.scale)
    res.converged = sub.converged
    res.fit_history = sub.fit_history
    if robust:
        cov = np.zeros((k, k))
        cov[np.ix_(keep, keep)] = _canonical_hc1(
            model, X[:, keep], sub.mu, sub.normalized_cov_params
        )
        res.cov_params_default = cov
        res.cov_type = "HC1"
        res._cache = {}
    res = GLMResultsWrapper(res)
    res.aliased = [names[i] for i in drop]
    return res


def _canonical_hc1(model, X, mu, bread) -> np.ndarray:
    """R's ``sandwich::vcovHC(glm, type = "HC1")`` for a canonical link.

    ``estfun.glm`` is ``w (y - mu) x`` (prior weights ``w``), the bread is the
    unscaled covariance, and HC1 multiplies by ``n / (n - k)``. Computed
    directly because statsmodels' GLM sandwich divides by ``mu (1 - mu)``,
    which is NaN once a fitted probability rounds to 0 or 1 -- where R still
    gives finite standard errors.
    """
    y = np.asarray(model.endog, dtype=float)
    w = getattr(model, "freq_weights", None)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    score = (w * (y - np.asarray(mu, dtype=float)))[:, None] * X
    bread = np.asarray(bread, dtype=float)
    n, k = X.shape
    return bread @ (score.T @ score) @ bread * n / (n - k)
