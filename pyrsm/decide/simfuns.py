"""Functions for simulation formulas and custom functions.

The names follow pyrsm's data filter and DSL (``mean``, ``std``, ``ceil``,
``first``, ``last``, ...) and polars (``max_horizontal``, ``cum_sum``,
``shift``). They are ordinary numpy functions, so a custom simulation
function can also be run and tested as plain Python::

    from pyrsm.decide.simfuns import zeros

    def calc_wealth(win, start, perc, mult):
        wealth = zeros(len(win))
        ...

    calc_wealth(np.array([1, 0, 1]), 1000, 0.05, 5)

Inside a simulation the same functions are used (and the random draws come
from the simulation's seeded generator).
"""

from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np

from pyrsm.utils import ifelse

#: Largest array a function may create
MAX_ARRAY = 10_000_000

_rng = np.random.default_rng()


def _arr(x: Any) -> np.ndarray:
    return x if isinstance(x, np.ndarray) else np.asarray(x, dtype=float)


def _flat(args: tuple) -> np.ndarray:
    parts = [np.ravel(np.asarray(a, dtype=float)) for a in args]
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def _one(value: Any) -> Any:
    """numpy scalars -> Python numbers."""
    return value.item() if isinstance(value, np.generic) else value


def _size(n: Any) -> int:
    n = int(n)
    if n < 0 or n > MAX_ARRAY:
        raise ValueError(f"arrays can have 0 to {MAX_ARRAY:,} values, not {n:,}")
    return n


def _elementwise(fn):
    def run(x: Any) -> Any:
        return _one(fn(x))

    run.__name__ = fn.__name__
    return run


# -- one value per row --------------------------------------------------------

abs = _elementwise(np.abs)  # noqa: A001
sqrt = _elementwise(np.sqrt)
exp = _elementwise(np.exp)
ln = _elementwise(np.log)
log10 = _elementwise(np.log10)
log2 = _elementwise(np.log2)
floor = _elementwise(np.floor)
ceil = _elementwise(np.ceil)
sign = _elementwise(np.sign)
sin = _elementwise(np.sin)
cos = _elementwise(np.cos)
tan = _elementwise(np.tan)


def log(x: Any, base: Any = None) -> Any:
    """Natural logarithm, or the logarithm with ``base``."""
    out = np.log(x) if base is None else np.log(x) / np.log(base)
    return _one(out)


def round(x: Any, decimals: Any = 0) -> Any:  # noqa: A001
    return _one(np.round(x, int(decimals)))


def clip(x: Any, lower: Any = None, upper: Any = None) -> Any:
    """``x`` limited to ``lower`` and ``upper``."""
    return _one(np.clip(x, lower, upper))


def _horizontal(fn):
    def run(*args: Any) -> Any:
        if not args:
            raise ValueError("give at least one variable")
        arrays = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in args])
        return _one(fn(np.stack(arrays), axis=0))

    return run


#: Per row: the smallest, largest, sum and mean of several variables
min_horizontal = _horizontal(np.min)
max_horizontal = _horizontal(np.max)
sum_horizontal = _horizontal(np.sum)
mean_horizontal = _horizontal(np.mean)


def cum_sum(x: Any) -> np.ndarray:
    """Running total."""
    return np.cumsum(_arr(x))


def cum_prod(x: Any) -> np.ndarray:
    """Running product."""
    return np.cumprod(_arr(x))


def cum_max(x: Any) -> np.ndarray:
    return np.maximum.accumulate(_arr(x))


def cum_min(x: Any) -> np.ndarray:
    return np.minimum.accumulate(_arr(x))


def shift(x: Any, n: Any = 1, fill_value: Any = 0.0) -> np.ndarray:
    """``x`` moved down ``n`` rows (up for negative ``n``); the rows left
    open get ``fill_value``. ``shift(x)`` is the previous row's value."""
    x = _arr(x)
    n = int(n)
    out = np.full(x.shape, float(fill_value))
    if n >= 0 and n < x.size:
        out[n:] = x[: x.size - n]
    elif n < 0 and -n < x.size:
        out[:n] = x[-n:]
    return out


# -- one value -------------------------------------------------------------------


def _aggregate(name: str, fn):
    def run(*args: Any) -> float:
        values = _flat(args)
        if values.size == 0:
            raise ValueError(f"{name}() of no values")
        return float(fn(values))

    run.__name__ = name
    return run


sum = _aggregate("sum", np.sum)  # noqa: A001
mean = _aggregate("mean", np.mean)
median = _aggregate("median", np.median)
min = _aggregate("min", np.min)  # noqa: A001
max = _aggregate("max", np.max)  # noqa: A001
std = _aggregate("std", lambda v: np.std(v, ddof=1) if v.size > 1 else math.nan)
var = _aggregate("var", lambda v: np.var(v, ddof=1) if v.size > 1 else math.nan)


def count(x: Any) -> int:
    """Number of values."""
    return int(np.size(x))


def first(x: Any) -> Any:
    return _one(_arr(x).ravel()[0])


def last(x: Any) -> Any:
    return _one(_arr(x).ravel()[-1])


def n_unique(x: Any) -> int:
    return int(np.unique(_arr(x)).size)


def quantile(x: Any, q: Any) -> float:
    return float(np.quantile(_arr(x), float(q)))


def corr(x: Any, y: Any) -> float:
    return float(np.corrcoef(_arr(x), _arr(y))[0, 1])


def cov(x: Any, y: Any) -> float:
    return float(np.cov(_arr(x), _arr(y), ddof=1)[0, 1])


def arg_max(x: Any) -> int:
    """Row of the largest value (from 0): ``price[arg_max(profit)]``."""
    return int(np.argmax(_arr(x)))


def arg_min(x: Any) -> int:
    """Row of the smallest value (from 0)."""
    return int(np.argmin(_arr(x)))


def weighted_sum_std(*args: Any) -> Any:
    """Standard deviation of a weighted sum of variables:
    ``weighted_sum_std(w1, w2, x1, x2)`` -- weights first, then as many
    variables. Weights can be numbers or one weight per row (a grid search);
    the result is one standard deviation per row of weights."""
    if len(args) < 2 or len(args) % 2:
        raise ValueError("give as many weights as variables: weighted_sum_std(w1, w2, x1, x2)")
    k = len(args) // 2
    series = np.column_stack([_arr(a) for a in args[k:]])
    covariance = np.atleast_2d(np.cov(series, rowvar=False, ddof=1))
    weights = np.column_stack(np.broadcast_arrays(*[_arr(a) for a in args[:k]]))
    out = np.sqrt(np.einsum("ij,jk,ik->i", weights, covariance, weights))
    return float(out[0]) if out.size == 1 else out


# -- queues -----------------------------------------------------------------------


def _queue(interarrival: Any, service: Any, servers: Any) -> tuple[np.ndarray, np.ndarray]:
    gaps, serv = np.broadcast_arrays(_arr(interarrival), _arr(service))
    c = int(servers)
    if c < 1:
        raise ValueError("queue functions need at least 1 server")
    if np.any(gaps < 0) or np.any(serv < 0):
        raise ValueError("time between arrivals and service times cannot be negative")
    arrive = np.cumsum(gaps)
    start = np.empty_like(arrive)
    free = [0.0] * c
    for i in range(arrive.size):
        first_free = heapq.heappop(free)
        start[i] = arrive[i] if arrive[i] > first_free else first_free
        heapq.heappush(free, start[i] + serv[i])
    return arrive, start


def queue_wait(interarrival: Any, service: Any, servers: Any = 1) -> np.ndarray:
    """Time each customer waits before service starts, in a first-come
    first-served queue with ``servers`` servers. ``interarrival`` is the time
    since the previous customer arrived, ``service`` the time service takes."""
    arrive, start = _queue(interarrival, service, servers)
    return start - arrive


def queue_length(interarrival: Any, service: Any, servers: Any = 1) -> np.ndarray:
    """Number of customers already waiting (not yet in service) when each
    customer arrives."""
    arrive, start = _queue(interarrival, service, servers)
    idx = np.arange(arrive.size)
    started = np.minimum(np.searchsorted(start, arrive, side="right"), idx)
    return (idx - started).astype(float)


# -- arrays (mostly for functions) -----------------------------------------------


def zeros(n: Any) -> np.ndarray:
    return np.zeros(_size(n))


def ones(n: Any) -> np.ndarray:
    return np.ones(_size(n))


def full(n: Any, value: Any) -> np.ndarray:
    return np.full(_size(n), float(value))


def arange(start: Any, stop: Any = None, step: Any = 1) -> np.ndarray:
    if stop is None:
        start, stop = 0, start
    count = math.ceil((float(stop) - float(start)) / float(step))
    _size(max(count, 0))
    return np.arange(float(start), float(stop), float(step))


def linspace(start: Any, stop: Any, num: Any) -> np.ndarray:
    return np.linspace(float(start), float(stop), _size(num))


# -- random draws -----------------------------------------------------------------


def make_random(rng: np.random.Generator) -> dict[str, Any]:
    """The random-draw functions, drawing from ``rng`` (the simulation's)."""

    def normal(size: Any, mean: Any = 0.0, sd: Any = 1.0) -> np.ndarray:
        return rng.normal(mean, sd, _size(size))

    def uniform(size: Any, min: Any = 0.0, max: Any = 1.0) -> np.ndarray:  # noqa: A002
        return rng.uniform(min, max, _size(size))

    def binomial(size: Any, trials: Any, prob: Any) -> np.ndarray:
        return rng.binomial(int(trials), prob, _size(size)).astype(float)

    def poisson(size: Any, rate: Any) -> np.ndarray:
        return rng.poisson(rate, _size(size)).astype(float)

    def exponential(size: Any, rate: Any = 1.0) -> np.ndarray:
        return rng.exponential(1.0 / float(rate), _size(size))

    def lognormal(size: Any, meanlog: Any = 0.0, sdlog: Any = 1.0) -> np.ndarray:
        return rng.lognormal(meanlog, sdlog, _size(size))

    def choice(values: Any, size: Any, probs: Any = None) -> np.ndarray:
        return rng.choice(_arr(values), _size(size), p=None if probs is None else _arr(probs))

    return {
        "normal": normal,
        "uniform": uniform,
        "binomial": binomial,
        "poisson": poisson,
        "exponential": exponential,
        "lognormal": lognormal,
        "choice": choice,
    }


_random = make_random(_rng)
normal = _random["normal"]
uniform = _random["uniform"]
binomial = _random["binomial"]
poisson = _random["poisson"]
exponential = _random["exponential"]
lognormal = _random["lognormal"]
choice = _random["choice"]

#: Everything formulas and functions can call, by name
FUNCTIONS = [
    "ifelse",
    "abs", "sqrt", "exp", "log", "ln", "log10", "log2", "floor", "ceil", "round", "sign",
    "sin", "cos", "tan", "clip",
    "min_horizontal", "max_horizontal", "sum_horizontal", "mean_horizontal",
    "cum_sum", "cum_prod", "cum_max", "cum_min", "shift",
    "sum", "mean", "median", "min", "max", "std", "var", "count", "first", "last",
    "n_unique", "quantile", "corr", "cov", "arg_max", "arg_min", "weighted_sum_std",
    "queue_wait", "queue_length",
    "zeros", "ones", "full", "arange", "linspace",
    "normal", "uniform", "binomial", "poisson", "exponential", "lognormal", "choice",
]  # fmt: skip

__all__ = [*FUNCTIONS, "make_random"]
