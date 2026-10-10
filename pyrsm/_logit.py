"""Maximum-likelihood engine for logit choice models.

One optimizer, two likelihoods:

* ``clogit_fit`` -- the conditional (McFadden) logit for choice data in long
  format: one row per alternative per choice task, the attributes vary across
  alternatives and share one coefficient each. Used by
  ``pyrsm.multivariate.conjoint_choice`` and ``pyrsm.multivariate.cbc``.
* ``mnl_fit`` -- the multinomial logit for a categorical response in wide
  format: one row per observation, the explanatory variables describe the
  observation and get a coefficient per response level (relative to the base
  level). Used by ``pyrsm.model.mnl``.

The multinomial logit is a conditional logit on a long design with one row per
response level, level-specific intercepts and the explanatory variables
interacted with the level indicators (tests/test_mnl.py checks this). The wide
likelihood here avoids building that larger design.

Both log-likelihoods are concave. ``maximize`` runs BFGS from zero and then
Newton-Raphson steps with the analytic Hessian (as R's ``clogit`` and
``mlogit``), so the optimum is reached even when BFGS stops early on badly
scaled data. Standard errors come from the inverse of the Hessian.
"""

from __future__ import annotations

import warnings

import numpy as np


def segment_probs(xb: np.ndarray, codes: np.ndarray, ngroups: int):
    """Softmax of ``xb`` within each task (vectorized, numerically stable)."""
    seg_max = np.full(ngroups, -np.inf)
    np.maximum.at(seg_max, codes, xb)
    ex = np.exp(xb - seg_max[codes])
    seg_sum = np.bincount(codes, weights=ex, minlength=ngroups)
    return ex / seg_sum[codes], seg_max, seg_sum


def maximize(nll_grad, hessian, p: int) -> dict:
    """Minimize a convex negative log-likelihood with ``p`` parameters.

    ``nll_grad(beta)`` returns ``(f, grad)``; ``hessian(beta)`` the Hessian of
    ``f``. Returns ``params``, ``fval`` (the minimum), ``hessian`` and
    ``converged`` (max |gradient| below 1e-6 relative to the likelihood).
    """
    from scipy.optimize import minimize

    with warnings.catch_warnings():
        # benign overflow can occur in BFGS's line-search step-norm; the
        # likelihood itself is max-stabilized, so ignore the noise.
        warnings.simplefilter("ignore", RuntimeWarning)
        res = minimize(nll_grad, np.zeros(p), jac=True, method="BFGS")
    beta = res.x

    # Newton-Raphson from the BFGS solution: the log-likelihood is concave, so
    # a few steps reach the optimum even when BFGS stops early.
    for _ in range(25):
        _, grad = nll_grad(beta)
        try:
            step = np.linalg.solve(hessian(beta), grad)
        except np.linalg.LinAlgError:
            break
        beta = beta - step
        if np.max(np.abs(step)) < 1e-10 * (1 + np.max(np.abs(beta))):
            break
    fval, grad = nll_grad(beta)
    converged = bool(np.max(np.abs(grad)) < 1e-6 * max(1.0, abs(fval)))
    return {"params": beta, "fval": fval, "hessian": hessian(beta), "converged": converged}


def std_errors(H: np.ndarray) -> np.ndarray:
    """Standard errors from the Hessian of the negative log-likelihood."""
    try:
        return np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
    except np.linalg.LinAlgError:
        return np.full(H.shape[0], np.nan)


def clogit_fit(X: np.ndarray, y: np.ndarray, codes: np.ndarray, ngroups: int) -> dict:
    """Fit a conditional (multinomial) logit by maximum likelihood.

    Parameters
    ----------
    X : (n, p) design matrix
    y : (n,) binary chosen indicator (exactly one 1 per task)
    codes : (n,) contiguous task code (0..ngroups-1) for each row
    ngroups : number of choice tasks

    Returns a dict with ``params``, ``bse``, ``zvalues``, ``pvalues``, ``llf``,
    ``llnull`` (uniform-within-task baseline), ``prsquared`` (McFadden) and
    ``converged``. The likelihood, gradient and Hessian are computed with
    segment (vectorized) operations -- no per-task Python loop.
    """
    from scipy.stats import norm

    p = X.shape[1]
    chosen = y == 1
    Xc_sum = X[chosen].sum(0)  # sum of chosen rows' covariates

    def nll_grad(beta):
        xb = X @ beta
        prob, seg_max, seg_sum = segment_probs(xb, codes, ngroups)
        # log-likelihood: sum_chosen xb - sum_tasks log-sum-exp
        f = -(xb[chosen].sum() - (seg_max + np.log(seg_sum)).sum())
        grad = -(Xc_sum - (X * prob[:, None]).sum(0))
        return f, grad

    def hessian(beta):
        # analytic Hessian: sum_n p_n x_n x_n' - sum_g xbar_g xbar_g'
        prob, _, _ = segment_probs(X @ beta, codes, ngroups)
        XB = np.zeros((ngroups, p))
        np.add.at(XB, codes, X * prob[:, None])
        return (X * prob[:, None]).T @ X - XB.T @ XB

    opt = maximize(nll_grad, hessian, p)
    beta = opt["params"]
    bse = std_errors(opt["hessian"])

    zvalues = beta / bse
    pvalues = 2 * (1 - norm.cdf(np.abs(zvalues)))
    llf = -opt["fval"]
    # null model: equal probability within each task
    llnull = -float(np.log(np.bincount(codes, minlength=ngroups)).sum())
    prsquared = 1 - llf / llnull if llnull != 0 else np.nan
    return {
        "params": beta,
        "bse": bse,
        "zvalues": zvalues,
        "pvalues": pvalues,
        "llf": float(llf),
        "llnull": float(llnull),
        "prsquared": float(prsquared),
        "converged": opt["converged"],
    }


def mnl_probs(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Probabilities of every response level, base level first.

    ``X`` is (n, p); ``B`` is (p, K - 1), one column per non-base level.
    """
    eta = np.column_stack([np.zeros(X.shape[0]), X @ B])
    eta -= eta.max(axis=1, keepdims=True)
    ex = np.exp(eta)
    return ex / ex.sum(axis=1, keepdims=True)


def mnl_fit(X: np.ndarray, y: np.ndarray, nlev: int, weights: np.ndarray | None = None) -> dict:
    """Fit a multinomial logit by maximum likelihood.

    Parameters
    ----------
    X : (n, p) design matrix (with the intercept column, if any)
    y : (n,) response level codes, 0 = base level, 1..nlev-1 the others
    nlev : number of response levels
    weights : (n,) frequency (case) weights, optional

    The parameters are ordered by response level: all coefficients for level
    1, then level 2, ... (as R's ``nnet::multinom``). Returns ``params``
    (flat), ``B`` ((p, nlev - 1)), ``bse``, ``cov``, ``llf``, ``llnull``
    (intercept-only model: each level's weighted share) and ``converged``.
    """
    n, p = X.shape
    M = nlev - 1
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    Y = np.zeros((n, M))
    rows = np.nonzero(y > 0)[0]
    Y[rows, y[rows] - 1] = 1.0

    def unpack(beta):
        return beta.reshape(M, p).T  # (p, M)

    def nll_grad(beta):
        B = unpack(beta)
        eta = X @ B
        mx = np.maximum(eta.max(axis=1), 0.0)
        ex = np.exp(eta - mx[:, None])
        denom = np.exp(-mx) + ex.sum(axis=1)
        lse = mx + np.log(denom)
        f = -float(np.sum(w * ((Y * eta).sum(axis=1) - lse)))
        P = ex / denom[:, None]
        G = -(X.T @ (w[:, None] * (Y - P)))  # (p, M)
        return f, G.T.ravel()

    def hessian(beta):
        B = unpack(beta)
        P = mnl_probs(X, B)[:, 1:]
        H = np.empty((M * p, M * p))
        for a in range(M):
            for b in range(a, M):
                d = w * (P[:, a] * ((a == b) - P[:, b]))
                blk = (X * d[:, None]).T @ X
                H[a * p : (a + 1) * p, b * p : (b + 1) * p] = blk
                H[b * p : (b + 1) * p, a * p : (a + 1) * p] = blk.T
        return H

    opt = maximize(nll_grad, hessian, M * p)
    beta = opt["params"]
    try:
        cov = np.linalg.inv(opt["hessian"])
    except np.linalg.LinAlgError:
        cov = np.full((M * p, M * p), np.nan)
    bse = np.sqrt(np.clip(np.diag(cov), 0, None))

    wk = np.bincount(y, weights=w, minlength=nlev)
    share = wk[wk > 0] / wk.sum()
    llnull = float(np.sum(wk[wk > 0] * np.log(share)))
    return {
        "params": beta,
        "B": unpack(beta),
        "bse": bse,
        "cov": cov,
        "llf": -float(opt["fval"]),
        "llnull": llnull,
        "converged": opt["converged"],
    }
