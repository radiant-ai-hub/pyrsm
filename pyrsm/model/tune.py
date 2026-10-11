"""Hyper-parameter tuning with cross-validation for ``mlp``, ``rforest`` and
``xgboost``.

:func:`tune` takes a fitted pyrsm model, the settings to try and a search
method, and cross-validates candidate settings on exactly the data the model
was fitted on. Four methods:

- ``"grid"``: every combination of the values given.
- ``"random"``: ``n_iter`` combinations drawn at random from the values or
  ranges given (log scale for ranges marked ``"log"``). For the same number
  of fits it usually finds better settings than a grid, because only a few
  settings tend to matter.
- ``"halving"``: successive halving. Many candidates are first scored on a
  small part of the data; the best third go on to a part three times larger,
  and so on, until a few are scored on all of it. Many candidates for the cost
  of a few full ones.
- ``"refine"``: coarse-to-fine. A random search first; then the ranges are
  narrowed to where the best 10% of candidates are and searched again with
  the rest of the budget ("look harder where it looks good").

>>> import pyrsm as rsm
>>> rf = rsm.model.rforest({"titanic": titanic}, rvar="survived", lev="Yes", evar=[...])
>>> tuned = rsm.model.tune(rf, {"max_features": [1, 2, 3, 4], "min_samples_leaf": [1, 5, 10]})
>>> tuned.summary()
>>> tuned.best_params
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import clone
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

METHODS = ("grid", "random", "halving", "refine")

#: Metrics by model type; the first is the default. Higher is better for all
#: but RMSE, which is shown as is and ranked lowest-first.
METRICS = {
    "classification": ("auc", "accuracy"),
    "regression": ("rmse", "r2"),
}

#: For "simplest within one standard error": settings where a larger value
#: makes a more complex model (+1) or a simpler one (-1).
COMPLEXITY = {
    "n_estimators": 1,
    "max_depth": 1,
    "learning_rate": 1,
    "max_features": 1,
    "max_samples": 1,
    "subsample": 1,
    "colsample_bytree": 1,
    "hidden_layer_sizes": 1,
    "alpha": -1,
    "min_samples_leaf": -1,
    "min_child_weight": -1,
}


class TuneError(ValueError):
    """Settings, ranges or budget that cannot be tuned as given."""


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------


def _layers(value: Any) -> tuple[int, ...]:
    """``(5,)``, ``"5"``, ``"10,5"``, ``[10, 5]`` -> a hidden-layer tuple."""
    if isinstance(value, int | np.integer):
        return (int(value),)
    if isinstance(value, list | tuple):
        return tuple(int(v) for v in value)
    text = str(value).strip().strip("()[]")
    parts = [p for p in text.replace(" ", ",").split(",") if p]
    if not parts:
        raise TuneError(f"cannot read hidden layer sizes from {value!r}")
    return tuple(int(p) for p in parts)


def _as_value(name: str, value: Any) -> Any:
    if name == "hidden_layer_sizes":
        return _layers(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"none", "all"}:
            return None
        try:
            number = float(text)
        except ValueError:
            return text
        return int(number) if number.is_integer() and "." not in text else number
    if isinstance(value, float) and value.is_integer() and name in _INTEGER:
        return int(value)
    return value


_INTEGER = {"n_estimators", "max_depth", "min_child_weight", "min_samples_leaf"}


@dataclass
class _Dim:
    """One setting to tune: a list of values or a range."""

    name: str
    values: list[Any] | None = None
    low: float | None = None
    high: float | None = None
    n: int = 5
    log: bool = False
    integer: bool = False

    def grid(self) -> list[Any]:
        if self.values is not None:
            return list(self.values)
        if self.log:
            pts = np.geomspace(self.low, self.high, self.n)
        else:
            pts = np.linspace(self.low, self.high, self.n)
        if self.integer:
            return sorted({int(round(p)) for p in pts})
        return [float(_round(p)) for p in pts]

    def sample(self, rng: np.random.Generator) -> Any:
        if self.values is not None:
            return self.values[int(rng.integers(len(self.values)))]
        if self.log:
            x = math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
        else:
            x = rng.uniform(self.low, self.high)
        return int(round(x)) if self.integer else float(_round(x))


def _round(x: float) -> float:
    """Four significant digits: readable settings, no false precision."""
    return float(f"{x:.4g}")


def _dims(params: dict[str, Any]) -> list[_Dim]:
    if not params:
        raise TuneError("give at least one setting to tune")
    dims = []
    for name, spec in params.items():
        if isinstance(spec, dict):
            if "values" in spec:
                vals = [_as_value(name, v) for v in spec["values"]]
                dims.append(_Dim(name, values=vals))
                continue
            try:
                low, high = float(spec["min"]), float(spec["max"])
            except (KeyError, TypeError, ValueError) as exc:
                raise TuneError(f"{name}: a range needs a min and a max") from exc
            if high < low:
                raise TuneError(f"{name}: max must be at least min")
            log = str(spec.get("scale", "linear")).lower() == "log"
            if log and low <= 0:
                raise TuneError(f"{name}: a log scale needs a min above 0")
            integer = bool(spec.get("integer", name in _INTEGER))
            dims.append(_Dim(name, low=low, high=high, n=max(int(spec.get("n", 5)), 1), log=log, integer=integer))
        else:
            vals = list(spec) if isinstance(spec, list | tuple) else [spec]
            if name == "hidden_layer_sizes" and isinstance(spec, tuple):
                vals = [spec]
            vals = [_as_value(name, v) for v in vals]
            if not vals:
                raise TuneError(f"{name}: give at least one value")
            dims.append(_Dim(name, values=vals))
    return dims


def _key(params: dict[str, Any]) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in params.items()))


def _unique(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen, out = set(), []
    for c in cands:
        k = _key(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _grid(dims: list[_Dim]) -> list[dict[str, Any]]:
    names = [d.name for d in dims]
    return [dict(zip(names, combo)) for combo in itertools.product(*(d.grid() for d in dims))]


def _random(dims: list[_Dim], n: int, rng: np.random.Generator, exclude: set | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set(exclude or ())
    space = math.prod(len(d.grid()) if d.values is not None else math.inf for d in dims)
    attempts = 0
    while len(out) < n and attempts < 50 * n:
        attempts += 1
        cand = {d.name: d.sample(rng) for d in dims}
        k = _key(cand)
        if k in seen:
            if len(seen) >= space:
                break
            continue
        seen.add(k)
        out.append(cand)
    return out


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


@dataclass
class _Data:
    X: pd.DataFrame
    y: np.ndarray
    estimator: Any
    mod_type: str


def _model_data(model: Any) -> _Data:
    for attr in ("fitted", "data_onehot", "mod_type", "rvar"):
        if not hasattr(model, attr):
            raise TuneError("tune() needs a fitted pyrsm mlp, rforest or xgboost model")
    if hasattr(model, "data_std"):
        # mlp: raw data and a network that standardizes inside every fit, so each
        # fold is scaled by its own training rows (see ``FoldScaledMLPRegressor``)
        from pyrsm.model.mlp import fold_scaled_data

        X, y, est = fold_scaled_data(model)
    else:
        X = model.data_onehot
        X = X.to_pandas() if isinstance(X, pl.DataFrame) else X
        y = model.data.get_column(model.rvar).to_numpy()
        est = clone(model.fitted)
    # Each fit runs on one thread: the parallelism is across fits (``n_jobs``), and
    # xgboost would otherwise use every core of the machine for every fit. The
    # out-of-bag score is not needed for cross-validation.
    own = est.get_params()
    fixed = {k: v for k, v in (("n_jobs", 1), ("oob_score", False)) if k in own}
    est.set_params(**fixed)
    return _Data(X=X, y=y, estimator=est, mod_type=model.mod_type)


def _score(metric: str, y: np.ndarray, pred: np.ndarray | None = None, prob: np.ndarray | None = None) -> float:
    if metric == "auc":
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, prob))
    if metric == "accuracy":
        return float(accuracy_score(y, pred))
    if metric == "rmse":
        return float(np.sqrt(np.mean((y - pred) ** 2)))
    if metric == "r2":
        return float(r2_score(y, pred))
    raise TuneError(f"unknown metric '{metric}'")


def _fit_score(estimator: Any, params: dict[str, Any], metric: str, X, y, train, test) -> float:
    est = clone(estimator).set_params(**params)
    est.fit(X.iloc[train], y[train])
    if metric == "auc":
        return _score(metric, y[test], prob=est.predict_proba(X.iloc[test])[:, -1])
    return _score(metric, y[test], pred=est.predict(X.iloc[test]))


def _higher_is_better(metric: str) -> bool:
    return metric != "rmse"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class TuneResult:
    """Cross-validated scores of each candidate, best first."""

    method: str
    metric: str
    folds: int
    results: pd.DataFrame
    best_params: dict[str, Any]
    best_score: float
    best_std: float
    simple_params: dict[str, Any]
    simple_score: float
    params: list[str]
    n_fits: int
    seconds: float
    model_type: str = ""
    stopped: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def best_params_(self) -> dict[str, Any]:
        """As sklearn's search objects: models take ``cv=tuned`` to use them."""
        return dict(self.best_params)

    def summary(self, dec: int = 4) -> None:
        print(self.summary_text(dec))

    def summary_text(self, dec: int = 4) -> str:
        better = "higher" if _higher_is_better(self.metric) else "lower"
        lines = [
            "Hyper-parameter tuning",
            f"Model        : {self.model_type}",
            f"Method       : {self.method}",
            f"Metric       : {self.metric.upper()} ({better} is better), {self.folds}-fold cross-validation",
            f"Candidates   : {len(self.results):,} ({self.n_fits:,} fits, {self.seconds:.0f} s)",
        ]
        if self.stopped:
            lines.append("Stopped early: the results are for the candidates scored so far")
        lines += [
            "",
            f"Best         : {_params_text(self.best_params)}",
            f"  {self.metric.upper()} {self.best_score:.{dec}f} (sd {self.best_std:.{dec}f})",
            f"Simplest within 1 SE: {_params_text(self.simple_params)}",
            f"  {self.metric.upper()} {self.simple_score:.{dec}f}",
            "",
            "Top candidates:",
            self.top(10).to_string(index=False, float_format=lambda v: f"{v:.{dec}f}"),
        ]
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)

    def top(self, n: int = 10) -> pd.DataFrame:
        return self.results.head(n)

    def plot(self):
        """The cross-validated score against each tuned setting (mean and
        +/- one standard deviation across the folds)."""
        from plotnine import aes, geom_errorbar, geom_point, ggplot, labs

        from pyrsm.plot_style import FILL, theme_pyrsm
        from pyrsm.plot_utils import compose_plots

        df = self.results.dropna(subset=["mean"])
        plots = []
        for p in self.params:
            d = df.assign(**{p: df[p].astype(str) if df[p].dtype == object else df[p]})
            plots.append(
                ggplot(d, aes(x=p, y="mean"))
                + geom_errorbar(aes(ymin="mean - std", ymax="mean + std"), width=0, alpha=0.4)
                + geom_point(color=FILL, size=2)
                + labs(x=p, y=self.metric.upper())
                + theme_pyrsm()
            )
        return compose_plots(plots, ncol=2, width_per_col=4, height_per_row=3)

    def to_dict(self, top: int = 50) -> dict[str, Any]:
        rows = self.results.head(top).replace({np.nan: None})
        return {
            "method": self.method,
            "metric": self.metric,
            "folds": self.folds,
            "higher_is_better": _higher_is_better(self.metric),
            "params": list(self.params),
            "best_params": _jsonable(self.best_params),
            "best_score": self.best_score,
            "best_std": self.best_std,
            "simple_params": _jsonable(self.simple_params),
            "simple_score": self.simple_score,
            "n_candidates": len(self.results),
            "n_fits": self.n_fits,
            "seconds": self.seconds,
            "stopped": self.stopped,
            "notes": list(self.notes),
            "model_type": self.model_type,
            "columns": list(rows.columns),
            "rows": [[_jsonable(v) for v in r] for r in rows.itertuples(index=False)],
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _params_text(params: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items()) or "(none)"


def _complexity(params: dict[str, Any]) -> tuple:
    out = []
    for name in sorted(params):
        sign = COMPLEXITY.get(name, 0)
        value = params[name]
        if name == "hidden_layer_sizes":
            value = sum(value) if isinstance(value, tuple | list) else value
        if isinstance(value, int | float) and sign:
            out.append(sign * float(value))
    return tuple(out)


# ---------------------------------------------------------------------------
# tune()
# ---------------------------------------------------------------------------


def tune(
    model: Any,
    params: dict[str, Any],
    *,
    method: str = "random",
    n_iter: int = 40,
    folds: int = 5,
    metric: str | None = None,
    seed: int = 1234,
    n_jobs: int = 4,
    backend: str | None = None,
    max_fits: int = 300,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> TuneResult:
    """Tune a fitted ``mlp``, ``rforest`` or ``xgboost`` model.

    Parameters
    ----------
    model : a fitted pyrsm ``mlp``, ``rforest`` or ``xgboost`` object; the
        settings not tuned keep their values, and the data are the rows it
        was fitted on.
    params : ``{setting: values}``. Values are a list (``[1, 2, 3]``) or a
        range ``{"min": 0.0001, "max": 1, "n": 5, "scale": "log"}`` (``n``
        is the number of grid points; random search draws from the range).
        Hidden layers can be given as ``(5,)``, ``"10,5"`` or ``[10, 5]``.
    method : "grid", "random", "halving" or "refine" (see the module docs).
    n_iter : candidates for "random", "halving" (the first round) and
        "refine" (both stages together).
    folds : number of cross-validation folds.
    metric : "auc" or "accuracy" (classification), "rmse" or "r2"
        (regression); the first is the default.
    max_fits : the budget: candidates x folds may not exceed it (halving
        counts the fits of all rounds).
    n_jobs, backend : fits run in parallel on ``n_jobs`` workers; ``backend``
        is a joblib backend ("threading" inside a daemon process, such as a
        Celery worker, which may not start processes of its own).
    progress, should_stop : called as fits finish (``progress(done, total)``)
        and checked between fits; stopping keeps the candidates scored so far.
    """
    from joblib import Parallel, delayed

    start = time.time()
    if method not in METHODS:
        raise TuneError(f"method must be one of {', '.join(METHODS)}")
    data = _model_data(model)
    allowed = METRICS[data.mod_type]
    metric = (metric or allowed[0]).lower()
    if metric not in allowed:
        raise TuneError(f"metric for {data.mod_type} must be one of {', '.join(allowed)}")
    folds = int(folds)
    if folds < 2:
        raise TuneError("use at least 2 folds")
    dims = _dims(params)
    valid = data.estimator.get_params()
    for d in dims:
        if d.name not in valid:
            raise TuneError(f"'{d.name}' is not a setting of this model")
    rng = np.random.default_rng(seed)
    splitter_cls = StratifiedKFold if data.mod_type == "classification" else KFold
    n_rows = len(data.y)
    notes: list[str] = []
    budget = int(max_fits)

    done = 0
    total_fits = 0
    stopped = False
    records: list[dict[str, Any]] = []

    def run(cands: list[dict[str, Any]], rows: np.ndarray, stage: str) -> list[dict[str, Any]]:
        """Cross-validate ``cands`` on ``rows``; return their records."""
        nonlocal done, stopped
        if stopped or not cands:
            return []
        X, y = data.X.iloc[rows], data.y[rows]
        splits = list(splitter_cls(n_splits=folds, shuffle=True, random_state=seed).split(X, y))
        tasks = [(ci, fi) for ci in range(len(cands)) for fi in range(len(splits))]
        scores: dict[int, list[float]] = {ci: [] for ci in range(len(cands))}
        import warnings

        warnings.filterwarnings("ignore", message=".*tasks have been successfully executed.*")
        par = Parallel(n_jobs=n_jobs, backend=backend, return_as="generator_unordered")
        gen = par(
            delayed(lambda ci, fi: (ci, _fit_score(data.estimator, cands[ci], metric, X, y, *splits[fi])))(ci, fi)
            for ci, fi in tasks
        )
        try:
            for ci, s in gen:
                scores[ci].append(s)
                done += 1
                if progress is not None:
                    progress(done, total_fits)
                if should_stop is not None and should_stop():
                    stopped = True
                    break
        finally:
            close = getattr(gen, "close", None)
            if close is not None:
                close()
        out = []
        for ci, cand in enumerate(cands):
            s = [v for v in scores[ci] if v is not None]
            if len(s) < folds:
                continue  # not every fold finished (stopped)
            arr = np.asarray(s, dtype=float)
            out.append({**cand, "mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr, ddof=1)), "rows": len(rows), "stage": stage})
        return out

    all_rows = np.arange(n_rows)
    if method == "grid":
        cands = _grid(dims)
        total_fits = len(cands) * folds
        if total_fits > budget:
            raise TuneError(
                f"the grid has {len(cands):,} combinations x {folds} folds = {total_fits:,} fits, "
                f"more than the {budget:,} allowed. Use fewer values, or the random, halving or "
                "refine method"
            )
        records = run(cands, all_rows, "grid")
    elif method == "random":
        n = min(int(n_iter), budget // folds)
        cands = _random(dims, n, rng)
        if len(cands) < int(n_iter):
            notes.append(f"{len(cands)} candidates (the budget or the values allow no more)")
        total_fits = len(cands) * folds
        records = run(cands, all_rows, "random")
    elif method == "refine":
        n_total = min(int(n_iter), budget // folds)
        n1 = max(n_total // 2, 3)
        first = _random(dims, n1, rng)
        total_fits = n_total * folds
        records = run(first, all_rows, "coarse")
        if records:
            ranked = sorted(records, key=lambda r: r["mean"], reverse=_higher_is_better(metric))
            top = ranked[: max(3, math.ceil(0.1 * len(ranked)))]
            narrow = _narrow(dims, top)
            notes.append("refined ranges: " + "; ".join(_dim_text(d) for d in narrow))
            second = _random(narrow, n_total - len(first), rng, exclude={_key(c) for c in first})
            total_fits = (len(first) + len(second)) * folds
            records += run(second, all_rows, "fine")
    else:  # halving
        factor = 3
        n0 = max(int(n_iter), factor)
        rounds = 1
        while True:
            # fits of all rounds when starting with n0 candidates
            k, fits, r = n0, 0, 0
            while True:
                fits += k * folds
                r += 1
                if k <= factor:
                    break
                k = math.ceil(k / factor)
            if fits <= budget or n0 <= factor:
                rounds = r
                break
            n0 -= 1
        cands = _random(dims, n0, rng)
        min_rows = max(folds * 20, n_rows // factor ** (rounds - 1))
        sizes = [min(n_rows, int(min_rows * factor**i)) for i in range(rounds)]
        sizes[-1] = n_rows
        total_fits = sum(math.ceil(len(cands) / factor**i) for i in range(rounds)) * folds
        notes.append(
            f"{rounds} rounds on {', '.join(f'{s:,}' for s in sizes)} rows, keeping the best third each round"
        )
        survivors = cands
        for i, size in enumerate(sizes):
            rows = np.sort(rng.choice(n_rows, size=size, replace=False)) if size < n_rows else all_rows
            recs = run(survivors, rows, f"round {i + 1}")
            records += recs
            if not recs or i == len(sizes) - 1:
                break
            ranked = sorted(recs, key=lambda r: r["mean"], reverse=_higher_is_better(metric))
            survivors = [{d.name: r[d.name] for d in dims} for r in ranked[: max(1, math.ceil(len(ranked) / factor))]]

    if not records:
        if stopped:
            raise TuneError("stopped before any candidate was scored on every fold")
        raise TuneError("no candidate finished; nothing to report")

    df = pd.DataFrame(records)
    # rank on the largest number of rows a candidate reached (halving)
    df = df.sort_values(["rows", "mean"], ascending=[False, not _higher_is_better(metric)]).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    names = [d.name for d in dims]
    best = df.iloc[0]
    best_params = {n: _native(best[n]) for n in names}
    full = df[df["rows"] == df["rows"].max()]
    se = float(best["std"]) / math.sqrt(folds) if math.isfinite(best["std"]) else 0.0
    if _higher_is_better(metric):
        close = full[full["mean"] >= best["mean"] - se]
    else:
        close = full[full["mean"] <= best["mean"] + se]
    simplest = min(
        (dict(r) for _, r in close.iterrows()),
        key=lambda r: _complexity({n: _native(r[n]) for n in names}),
    )
    return TuneResult(
        method=method,
        metric=metric,
        folds=folds,
        results=df,
        best_params=best_params,
        best_score=float(best["mean"]),
        best_std=float(best["std"]),
        simple_params={n: _native(simplest[n]) for n in names},
        simple_score=float(simplest["mean"]),
        params=names,
        n_fits=done,
        seconds=time.time() - start,
        model_type=type(model).__name__,
        stopped=stopped,
        notes=notes,
    )


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return tuple(value)
    return value


def _narrow(dims: list[_Dim], top: list[dict[str, Any]]) -> list[_Dim]:
    """New ranges around the best candidates: lists keep the values that
    appear among them, ranges shrink to their min..max (padded a little)."""
    out = []
    for d in dims:
        vals = [r[d.name] for r in top]
        if d.values is not None:
            keep = [v for v in d.values if any(str(v) == str(x) for x in vals)]
            out.append(_Dim(d.name, values=keep or d.values))
            continue
        lo, hi = min(vals), max(vals)
        if d.log:
            span = math.log(hi / lo) if hi > lo else math.log(d.high / d.low) / 4
            lo2, hi2 = lo / math.exp(0.25 * span), hi * math.exp(0.25 * span)
        else:
            span = (hi - lo) if hi > lo else (d.high - d.low) / 4
            lo2, hi2 = lo - 0.25 * span, hi + 0.25 * span
        out.append(_Dim(d.name, low=max(d.low, lo2), high=min(d.high, hi2), n=d.n, log=d.log, integer=d.integer))
    return out


def _dim_text(d: _Dim) -> str:
    if d.values is not None:
        return f"{d.name} in {d.values}"
    return f"{d.name} {_round(d.low)} to {_round(d.high)}"


__all__ = ["METHODS", "METRICS", "TuneError", "TuneResult", "tune"]


def _make_module_callable() -> None:
    """Keep ``pyrsm.model.tune(...)`` callable after the submodule is imported."""
    import sys
    import types

    class _CallableModule(types.ModuleType):
        def __call__(self, *args, **kwargs):
            return tune(*args, **kwargs)

    sys.modules[__name__].__class__ = _CallableModule


_make_module_callable()
