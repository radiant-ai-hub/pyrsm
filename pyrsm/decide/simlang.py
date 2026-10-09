"""The formula and function language of ``pyrsm.decide.simulate``.

Simulation formulas (``profit = demand * (price - cost)``) and custom
functions (``def wealth(win, start): ...``) are read by a small interpreter
over numpy arrays rather than by Python's ``eval``. It understands
arithmetic, comparisons, logic, function calls, indexing, and -- inside
functions -- assignments, ``if``, ``for``/``while`` loops and ``return``.
It has no attribute access, imports, strings, comprehensions or dunder names,
so a formula can only compute numbers, and every step counts against a budget
so a runaway loop stops with a message instead of hanging.

R habits are accepted where they cannot be misread: ``<-`` for ``=``,
``TRUE``/``FALSE``, ``^`` for powers, ``&``/``|``/``!`` for and/or/not,
``%%`` for the remainder, and R's function names (``ifelse``, ``pmax``,
``psum``, ``sd``, ``find_max``, ``rnorm``, ...).
"""

from __future__ import annotations

import ast
import heapq
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Steps (expression nodes and statements) one simulation may take
DEFAULT_BUDGET = 20_000_000
#: Largest array a function may create
MAX_ARRAY = 10_000_000
_MAX_DEPTH = 60


class SimLangError(ValueError):
    """A formula or function that cannot be read or evaluated."""


class BudgetExceeded(SimLangError):
    """The calculation took more steps than the budget allows."""


# ---------------------------------------------------------------------------
# R habits -> the Python syntax the parser reads
# ---------------------------------------------------------------------------


def r_to_python(expr: str) -> str:
    """Rewrite the R spellings a formula may use into Python syntax.

    R's ``&``, ``|`` and ``!`` bind more loosely than comparisons, like
    Python's ``and``, ``or`` and ``not`` (Python's ``&`` binds tighter, so
    ``x > 3 & y == 2`` would be misread); ``^`` is a power in R.
    """
    out = expr
    out = out.replace("%/%", "//").replace("%%", "%")
    out = re.sub(r"&&?", " and ", out)
    out = re.sub(r"\|\|?", " or ", out)
    out = re.sub(r"!(?!=)", " not ", out)
    out = out.replace("^", "**")
    out = re.sub(r"\bTRUE\b", "True", out)
    out = re.sub(r"\bFALSE\b", "False", out)
    return out


# ---------------------------------------------------------------------------
# Built-in functions
# ---------------------------------------------------------------------------


def _arr(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float) if not isinstance(x, np.ndarray) else x


def _flat(args: tuple) -> np.ndarray:
    parts = [np.ravel(np.asarray(a, dtype=float)) for a in args]
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def _need(name: str, args: tuple, n: int) -> None:
    if len(args) < n:
        raise SimLangError(f"{name}() needs {n} input{'s' if n > 1 else ''}")


def _parallel(name: str, fn: Callable[[np.ndarray], np.ndarray]) -> Callable[..., Any]:
    def run(*args: Any) -> Any:
        _need(name, args, 1)
        arrays = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in args])
        out = fn(np.stack(arrays))
        return out if np.ndim(out) else float(out)

    return run


def _aggregate(name: str, fn: Callable[[np.ndarray], float]) -> Callable[..., float]:
    def run(*args: Any) -> float:
        _need(name, args, 1)
        values = _flat(args)
        if values.size == 0:
            raise SimLangError(f"{name}() of no values")
        return float(fn(values))

    return run


def _sd(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if x.size > 1 else float("nan")


def _var(x: np.ndarray) -> float:
    return float(np.var(x, ddof=1)) if x.size > 1 else float("nan")


def find_max(x: Any, y: Any) -> float:
    """The value of ``y`` where ``x`` is largest (radiant.model's find_max)."""
    x, y = _arr(x), _arr(y)
    return float(np.broadcast_to(y, x.shape)[int(np.argmax(x))])


def find_min(x: Any, y: Any) -> float:
    """The value of ``y`` where ``x`` is smallest (radiant.model's find_min)."""
    x, y = _arr(x), _arr(y)
    return float(np.broadcast_to(y, x.shape)[int(np.argmin(x))])


def sdw(*args: Any) -> Any:
    """Standard deviation of a weighted sum: ``sdw(w1, w2, x1, x2)``.

    The first half of the inputs are weights (numbers, or one weight per row,
    e.g. from a grid search), the second half the series (e.g. data columns).
    Returns one standard deviation per row of weights, as radiant.model.
    """
    if len(args) < 2 or len(args) % 2:
        raise SimLangError("sdw() needs as many weights as variables, e.g. sdw(w1, w2, x1, x2)")
    k = len(args) // 2
    series = np.column_stack([_arr(a) for a in args[k:]])
    cov = np.cov(series, rowvar=False, ddof=1)
    weights = np.column_stack(np.broadcast_arrays(*[_arr(a) for a in args[:k]]))
    out = np.sqrt(np.einsum("ij,jk,ik->i", weights, np.atleast_2d(cov), weights))
    return float(out[0]) if out.size == 1 else out


def _queue(interarrival: Any, service: Any, servers: Any) -> tuple[np.ndarray, np.ndarray]:
    gaps, serv = np.broadcast_arrays(_arr(interarrival), _arr(service))
    c = int(servers)
    if c < 1:
        raise SimLangError("queue functions need at least 1 server")
    if np.any(gaps < 0) or np.any(serv < 0):
        raise SimLangError("time between arrivals and service times cannot be negative")
    arrive = np.cumsum(gaps)
    start = np.empty_like(arrive)
    free = [0.0] * c
    for i in range(arrive.size):
        first_free = heapq.heappop(free)
        start[i] = max(arrive[i], first_free)
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


def lag(x: Any, k: Any = 1, fill: Any = 0.0) -> np.ndarray:
    """``x`` shifted down ``k`` rows; the first ``k`` rows get ``fill``."""
    x = _arr(x)
    k = int(k)
    out = np.full(x.shape, float(fill))
    if k < x.size:
        out[k:] = x[: x.size - k]
    return out


def lead(x: Any, k: Any = 1, fill: Any = 0.0) -> np.ndarray:
    """``x`` shifted up ``k`` rows; the last ``k`` rows get ``fill``."""
    x = _arr(x)
    k = int(k)
    out = np.full(x.shape, float(fill))
    if k < x.size:
        out[: x.size - k] = x[k:]
    return out


def _size(n: Any) -> int:
    n = int(n)
    if n < 0 or n > MAX_ARRAY:
        raise SimLangError(f"arrays can have 0 to {MAX_ARRAY:,} values, not {n:,}")
    return n


def _seq(start: Any, stop: Any, by: Any = 1) -> np.ndarray:
    by = float(by)
    if by == 0:
        raise SimLangError("seq() needs a step that is not 0")
    count = math.floor((float(stop) - float(start)) / by + 1e-10) + 1
    return float(start) + by * np.arange(_size(max(count, 0)))


def _round(x: Any, digits: Any = 0) -> Any:
    out = np.round(x, int(digits))
    return out if np.ndim(out) else float(out)


def _log(x: Any, base: Any = None) -> Any:
    out = np.log(x) if base is None else np.log(x) / np.log(base)
    return out if np.ndim(out) else float(out)


def _elementwise(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def run(x: Any) -> Any:
        out = fn(x)
        return out if np.ndim(out) else (bool(out) if isinstance(out, np.bool_) else float(out))

    return run


def _make_builtins(rng: np.random.Generator, nrows: Callable[[], int]) -> dict[str, Callable]:
    def draws(n: Any) -> int:
        return _size(n)

    return {
        # one value per row
        "ifelse": lambda cond, a, b: np.where(cond, a, b),
        "pmin": _parallel("pmin", lambda s: np.min(s, axis=0)),
        "pmax": _parallel("pmax", lambda s: np.max(s, axis=0)),
        "psum": _parallel("psum", lambda s: np.sum(s, axis=0)),
        "pmean": _parallel("pmean", lambda s: np.mean(s, axis=0)),
        "pmedian": _parallel("pmedian", lambda s: np.median(s, axis=0)),
        "psd": _parallel("psd", lambda s: np.std(s, axis=0, ddof=1)),
        "pvar": _parallel("pvar", lambda s: np.var(s, axis=0, ddof=1)),
        "pprod": _parallel("pprod", lambda s: np.prod(s, axis=0)),
        "abs": _elementwise(np.abs),
        "sqrt": _elementwise(np.sqrt),
        "exp": _elementwise(np.exp),
        "log": _log,
        "ln": _elementwise(np.log),
        "log10": _elementwise(np.log10),
        "log2": _elementwise(np.log2),
        "square": _elementwise(np.square),
        "round": _round,
        "floor": _elementwise(np.floor),
        "ceiling": _elementwise(np.ceil),
        "ceil": _elementwise(np.ceil),
        "trunc": _elementwise(np.trunc),
        "sign": _elementwise(np.sign),
        "cumsum": lambda x: np.cumsum(_arr(x)),
        "cumprod": lambda x: np.cumprod(_arr(x)),
        "cummax": lambda x: np.maximum.accumulate(_arr(x)),
        "cummin": lambda x: np.minimum.accumulate(_arr(x)),
        "lag": lag,
        "lead": lead,
        "row_number": lambda: np.arange(1, nrows() + 1, dtype=float),
        # one value
        "sum": _aggregate("sum", np.sum),
        "mean": _aggregate("mean", np.mean),
        "median": _aggregate("median", np.median),
        "min": _aggregate("min", np.min),
        "max": _aggregate("max", np.max),
        "sd": _aggregate("sd", _sd),
        "var": _aggregate("var", _var),
        "prod": _aggregate("prod", np.prod),
        "prop": _aggregate("prop", np.mean),
        "quantile": lambda x, p: float(np.quantile(_arr(x), float(p))),
        "cor": lambda x, y: float(np.corrcoef(_arr(x), _arr(y))[0, 1]),
        "cov": lambda x, y: float(np.cov(_arr(x), _arr(y), ddof=1)[0, 1]),
        "any": lambda x: bool(np.any(x)),
        "all": lambda x: bool(np.all(x)),
        "length": lambda x: int(np.size(x)),
        "len": lambda x: int(np.size(x)),
        "n": nrows,
        "find_max": find_max,
        "find_min": find_min,
        "sdw": sdw,
        # queues
        "queue_wait": queue_wait,
        "queue_length": queue_length,
        # arrays (mostly for functions)
        "zeros": lambda n: np.zeros(_size(n)),
        "numeric": lambda n: np.zeros(_size(n)),
        "ones": lambda n: np.ones(_size(n)),
        "rep": lambda x, n: np.full(_size(n), float(x)),
        "seq": _seq,
        "seq_len": lambda n: np.arange(1, _size(n) + 1, dtype=float),
        "c": lambda *args: _flat(args),
        "range": lambda *args: range(*[int(a) for a in args]),
        "int": lambda x: int(x),
        "float": lambda x: float(x),
        # random draws, from the simulation's seeded generator
        "rnorm": lambda n, mean=0.0, sd=1.0: rng.normal(mean, sd, draws(n)),
        "runif": lambda n, min=0.0, max=1.0: rng.uniform(min, max, draws(n)),
        "rbinom": lambda n, size, prob: rng.binomial(int(size), prob, draws(n)).astype(float),
        "rpois": lambda n, rate: rng.poisson(rate, draws(n)).astype(float),
        "rexp": lambda n, rate=1.0: rng.exponential(1.0 / float(rate), draws(n)),
        "rlnorm": lambda n, meanlog=0.0, sdlog=1.0: rng.lognormal(meanlog, sdlog, draws(n)),
        "sample": lambda x, size, replace=True, prob=None: rng.choice(
            _arr(x), draws(size), replace=bool(replace), p=None if prob is None else _arr(prob)
        ),
    }


#: Names a simulation offers in formulas, for the help and error messages
BUILTIN_NAMES = sorted(_make_builtins(np.random.default_rng(0), lambda: 0))


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------


class _Return(Exception):
    def __init__(self, value: Any):
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


@dataclass
class UserFunction:
    name: str
    params: list[str]
    defaults: list[Any]
    body: list[ast.stmt]
    lineno: int = 0


_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: np.true_divide(a, b),
    ast.FloorDiv: lambda a, b: np.floor_divide(a, b),
    ast.Mod: lambda a, b: np.mod(a, b),
    ast.Pow: lambda a, b: np.power(np.asarray(a, dtype=float), b),
    # R: ^ is a power, & and | are and / or
    ast.BitXor: lambda a, b: np.power(np.asarray(a, dtype=float), b),
    ast.BitAnd: lambda a, b: np.logical_and(a, b),
    ast.BitOr: lambda a, b: np.logical_or(a, b),
}

_CMPOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


def _scalar(value: Any) -> Any:
    """numpy scalars -> Python numbers (so ``if`` and indexing work)."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


@dataclass
class Interpreter:
    """Evaluate formulas and custom functions over a namespace of arrays."""

    names: dict[str, Any]
    rng: np.random.Generator
    nrows: int = 0
    budget: int = DEFAULT_BUDGET
    functions: dict[str, UserFunction] = field(default_factory=dict)
    steps: int = 0
    _depth: int = 0

    def __post_init__(self) -> None:
        self.builtins = _make_builtins(self.rng, lambda: self.nrows)

    # -- budget -------------------------------------------------------
    def _tick(self) -> None:
        self.steps += 1
        if self.steps > self.budget:
            raise BudgetExceeded(
                f"the calculation took more than {self.budget:,} steps. Use functions that work "
                "on a whole column at once (cumsum, pmax, queue_wait, ...) instead of loops, "
                "or use fewer simulations."
            )

    # -- functions ----------------------------------------------------
    def define_functions(self, source: str) -> None:
        """Read ``def`` blocks (custom functions) into the interpreter."""
        if not source or not source.strip():
            return
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise SimLangError(f"functions, line {exc.lineno}: {exc.msg}") from exc
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue  # a docstring or comment string
            if not isinstance(node, ast.FunctionDef):
                raise SimLangError(
                    f"functions, line {node.lineno}: only function definitions "
                    "(def name(...):) belong here; put calculations in the formulas"
                )
            _check_function(node)
            args = node.args
            defaults = [self.eval(d, {}) for d in args.defaults]
            params = [a.arg for a in args.args]
            self.functions[node.name] = UserFunction(
                node.name, params, [None] * (len(params) - len(defaults)) + defaults, node.body, node.lineno
            )

    def call_user(self, fn: UserFunction, args: list[Any], kwargs: dict[str, Any]) -> Any:
        if len(args) > len(fn.params):
            raise SimLangError(f"{fn.name}() takes {len(fn.params)} inputs, got {len(args)}")
        local: dict[str, Any] = {}
        for i, p in enumerate(fn.params):
            if i < len(args):
                value = args[i]
            elif p in kwargs:
                value = kwargs.pop(p)
            elif fn.defaults[i] is not None:
                value = fn.defaults[i]
            else:
                raise SimLangError(f"{fn.name}() is missing input '{p}'")
            # a copy, so element assignments never change a simulated column
            local[p] = value.copy() if isinstance(value, np.ndarray) else value
        if kwargs:
            raise SimLangError(f"{fn.name}() has no input '{next(iter(kwargs))}'")
        self._depth += 1
        if self._depth > _MAX_DEPTH:
            raise SimLangError(f"{fn.name}() calls itself more than {_MAX_DEPTH} levels deep")
        try:
            self.exec_block(fn.body, local)
        except _Return as ret:
            return ret.value
        finally:
            self._depth -= 1
        return None

    # -- statements ---------------------------------------------------
    def exec_block(self, body: list[ast.stmt], local: dict[str, Any]) -> None:
        for stmt in body:
            self.exec(stmt, local)

    def exec(self, node: ast.stmt, local: dict[str, Any]) -> None:
        self._tick()
        if isinstance(node, ast.Return):
            raise _Return(None if node.value is None else self.eval(node.value, local))
        if isinstance(node, ast.Assign):
            value = self.eval(node.value, local)
            for target in node.targets:
                self._assign(target, value, local)
            return
        if isinstance(node, ast.AugAssign):
            op = _BINOPS.get(type(node.op))
            if op is None:
                raise SimLangError(f"line {node.lineno}: that kind of update is not supported")
            current = self.eval(_as_load(node.target), local)
            self._assign(node.target, op(current, self.eval(node.value, local)), local)
            return
        if isinstance(node, ast.If):
            branch = node.body if self._truth(self.eval(node.test, local), node) else node.orelse
            self.exec_block(branch, local)
            return
        if isinstance(node, ast.For):
            iterable = self.eval(node.iter, local)
            if isinstance(iterable, np.ndarray):
                iterable = iterable.tolist()
            if not isinstance(iterable, range | list):
                raise SimLangError(f"line {node.lineno}: loop over range(...) or an array")
            for item in iterable:
                self._tick()
                self._assign(node.target, item, local)
                try:
                    self.exec_block(node.body, local)
                except _Break:
                    break
                except _Continue:
                    continue
            return
        if isinstance(node, ast.While):
            while self._truth(self.eval(node.test, local), node):
                try:
                    self.exec_block(node.body, local)
                except _Break:
                    break
                except _Continue:
                    continue
            return
        if isinstance(node, ast.Break):
            raise _Break
        if isinstance(node, ast.Continue):
            raise _Continue
        if isinstance(node, ast.Pass):
            return
        if isinstance(node, ast.Expr):
            self.eval(node.value, local)
            return
        raise SimLangError(f"line {node.lineno}: {type(node).__name__} is not supported in functions")

    def _truth(self, value: Any, node: ast.AST) -> bool:
        value = _scalar(value)
        if isinstance(value, np.ndarray):
            raise SimLangError(
                f"line {node.lineno}: 'if' and 'while' need one True/False value, not a column; "
                "use ifelse() for a value per row, or any()/all()"
            )
        return bool(value)

    def _assign(self, target: ast.expr, value: Any, local: dict[str, Any]) -> None:
        if isinstance(target, ast.Name):
            local[target.id] = value
            return
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            name = target.value.id
            if name not in local:
                raise SimLangError(f"'{name}' must be created in the function before it is changed")
            arr = local[name]
            if not isinstance(arr, np.ndarray):
                raise SimLangError(f"'{name}' is not an array")
            arr[self._index(target.slice, local)] = value
            return
        if isinstance(target, ast.Tuple):
            values = list(value)
            if len(values) != len(target.elts):
                raise SimLangError("the number of names and values differ")
            for t, v in zip(target.elts, values):
                self._assign(t, v, local)
            return
        raise SimLangError("only names and array elements (x[i]) can be assigned")

    # -- expressions --------------------------------------------------
    def eval(self, node: ast.AST, local: dict[str, Any]) -> Any:
        self._tick()
        if isinstance(node, ast.Expression):
            return self.eval(node.body, local)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool | int | float) or node.value is None:
                return node.value
            raise SimLangError("text values are not supported in formulas")
        if isinstance(node, ast.Name):
            if node.id in local:
                return local[node.id]
            if node.id in self.names:
                return self.names[node.id]
            if node.id in ("True", "False"):
                return node.id == "True"
            if node.id in ("pi",):
                return math.pi
            raise SimLangError(f"unknown name '{node.id}'")
        if isinstance(node, ast.UnaryOp):
            operand = self.eval(node.operand, local)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.Not | ast.Invert):
                return np.logical_not(operand) if isinstance(operand, np.ndarray) else not operand
        if isinstance(node, ast.BinOp):
            op = _BINOPS.get(type(node.op))
            if op is None:
                raise SimLangError(f"operator {type(node.op).__name__} is not supported")
            left, right = self.eval(node.left, local), self.eval(node.right, local)
            try:
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    return _scalar(op(left, right))
            except ValueError as exc:
                raise SimLangError(_shape_message(left, right)) from exc
        if isinstance(node, ast.Compare):
            left = self.eval(node.left, local)
            result: Any = True
            for op_node, comp in zip(node.ops, node.comparators):
                op = _CMPOPS.get(type(op_node))
                if op is None:
                    raise SimLangError(f"comparison {type(op_node).__name__} is not supported")
                right = self.eval(comp, local)
                try:
                    part = op(left, right)
                except ValueError as exc:
                    raise SimLangError(_shape_message(left, right)) from exc
                result = part if result is True else np.logical_and(result, part)
                left = right
            return _scalar(result)
        if isinstance(node, ast.BoolOp):
            values = [self.eval(v, local) for v in node.values]
            if any(isinstance(v, np.ndarray) for v in values):
                fn = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
                acc = values[0]
                for v in values[1:]:
                    acc = fn(acc, v)
                return acc
            if isinstance(node.op, ast.And):
                return all(bool(v) for v in values)
            return any(bool(v) for v in values)
        if isinstance(node, ast.IfExp):
            test = self.eval(node.test, local)
            if isinstance(test, np.ndarray):
                return np.where(test, self.eval(node.body, local), self.eval(node.orelse, local))
            return self.eval(node.body if test else node.orelse, local)
        if isinstance(node, ast.Call):
            return self._call(node, local)
        if isinstance(node, ast.Subscript):
            value = self.eval(node.value, local)
            if not isinstance(value, np.ndarray):
                raise SimLangError("only arrays can be indexed with [ ]")
            return _scalar(value[self._index(node.slice, local)])
        if isinstance(node, ast.List | ast.Tuple):
            return np.asarray([self.eval(e, local) for e in node.elts], dtype=float)
        raise SimLangError(f"{type(node).__name__} is not supported in formulas")

    def _index(self, node: ast.AST, local: dict[str, Any]) -> Any:
        if isinstance(node, ast.Slice):
            parts = [None if p is None else int(_scalar(self.eval(p, local))) for p in (node.lower, node.upper, node.step)]
            return slice(*parts)
        index = _scalar(self.eval(node, local))
        if isinstance(index, float) and index.is_integer():
            return int(index)
        if isinstance(index, np.ndarray) and index.dtype != bool:
            return index.astype(int)
        return index

    def _call(self, node: ast.Call, local: dict[str, Any]) -> Any:
        if not isinstance(node.func, ast.Name):
            raise SimLangError("only functions called by name, like mean(x), are supported")
        name = node.func.id
        args = [self.eval(a, local) for a in node.args]
        kwargs = {k.arg: self.eval(k.value, local) for k in node.keywords if k.arg}
        if name in self.functions:
            return self.call_user(self.functions[name], args, kwargs)
        fn = self.builtins.get(name)
        if fn is None:
            raise SimLangError(f"unknown function '{name}'")
        try:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                return fn(*args, **kwargs)
        except SimLangError:
            raise
        except TypeError as exc:
            raise SimLangError(f"{name}(): wrong number or kind of inputs ({exc})") from exc
        except ValueError as exc:
            raise SimLangError(f"{name}(): {exc}") from exc

    def evaluate(self, expr: str) -> Any:
        """Evaluate one expression (R spellings allowed) in the namespace."""
        try:
            tree = ast.parse(r_to_python(expr).strip(), mode="eval")
        except SyntaxError as exc:
            raise SimLangError(f"cannot read '{expr}': {exc.msg}") from exc
        _check_expression(tree)
        return self.eval(tree, {})


def _shape_message(left: Any, right: Any) -> str:
    sizes = [np.size(x) for x in (left, right) if isinstance(x, np.ndarray)]
    return (
        f"cannot combine columns with {sizes[0]:,} and {sizes[-1]:,} values. Columns from the "
        "input data have one value per data row; summarise them (mean, sd, ...) or use them "
        "with a function such as sdw()"
    )


def _as_load(target: ast.expr) -> ast.expr:
    clone = ast.parse(ast.unparse(target), mode="eval").body
    return clone


# ---------------------------------------------------------------------------
# Static checks: what may appear at all
# ---------------------------------------------------------------------------

_EXPR_NODES = (
    ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Store, ast.UnaryOp, ast.BinOp,
    ast.Compare, ast.BoolOp, ast.IfExp, ast.Call, ast.keyword, ast.Subscript, ast.Slice,
    ast.List, ast.Tuple, ast.operator, ast.unaryop, ast.cmpop, ast.boolop,
)  # fmt: skip
_STMT_NODES = (
    ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign, ast.If,
    ast.For, ast.While, ast.Break, ast.Continue, ast.Pass, ast.Expr,
)  # fmt: skip


def _check_names(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        name = getattr(node, "id", None) or getattr(node, "arg", None) or getattr(node, "name", None)
        if isinstance(name, str) and name.startswith("__"):
            raise SimLangError(f"names starting with '__' are not allowed ('{name}')")
        if isinstance(node, ast.Attribute):
            raise SimLangError(
                f"'.{node.attr}' is not supported: use functions, like mean(x), and _ in names"
            )


def _check_expression(tree: ast.AST) -> None:
    _check_names(tree)
    for node in ast.walk(tree):
        if not isinstance(node, _EXPR_NODES):
            raise SimLangError(f"{type(node).__name__} is not supported in formulas")


def _check_function(fn: ast.FunctionDef) -> None:
    _check_names(fn)
    if fn.decorator_list or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.posonlyargs:
        raise SimLangError(f"{fn.name}(): use plain inputs, e.g. def {fn.name}(x, y=1):")
    for node in ast.walk(fn):
        if node is fn:
            continue
        if isinstance(node, ast.FunctionDef):
            raise SimLangError(f"{fn.name}(): define functions one after the other, not inside each other")
        if not isinstance(node, _EXPR_NODES + _STMT_NODES):
            raise SimLangError(
                f"{fn.name}(), line {getattr(node, 'lineno', fn.lineno)}: "
                f"{type(node).__name__} is not supported in functions"
            )


# ---------------------------------------------------------------------------
# Formula text
# ---------------------------------------------------------------------------

_ASSIGN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*(<-|=(?!=))\s*(.*)$", re.S)


def split_formulas(text: str) -> list[tuple[str, str, int]]:
    """``name = expression`` statements from a formulas text block.

    Comments (``#``) are dropped, ``;`` separates statements on one line, and
    a statement continues on the next line while a bracket is open. Returns
    ``(name, expression, line)`` tuples.
    """
    statements: list[tuple[str, int]] = []
    buffer, start, depth = "", 0, 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0]
        for part in line.split(";"):
            if not part.strip():
                continue
            if not buffer:
                start = lineno
            buffer = f"{buffer} {part.strip()}" if buffer else part.strip()
            depth = buffer.count("(") - buffer.count(")") + buffer.count("[") - buffer.count("]")
            if depth <= 0:
                statements.append((buffer, start))
                buffer = ""
    if buffer:
        statements.append((buffer, start))
    out: list[tuple[str, str, int]] = []
    for stmt, lineno in statements:
        m = _ASSIGN.match(stmt)
        if not m:
            raise SimLangError(
                f"formulas, line {lineno}: '{stmt}' needs a name and '=', e.g. profit = demand * price"
            )
        name, _, expr = m.groups()
        if "." in name:
            raise SimLangError(f"formulas, line {lineno}: use _ instead of . in names ('{name}')")
        if not expr.strip():
            raise SimLangError(f"formulas, line {lineno}: '{name} =' has nothing after the '='")
        out.append((name, expr.strip(), lineno))
    return out


__all__ = [
    "BUILTIN_NAMES",
    "BudgetExceeded",
    "Interpreter",
    "SimLangError",
    "UserFunction",
    "find_max",
    "find_min",
    "lag",
    "lead",
    "queue_length",
    "queue_wait",
    "r_to_python",
    "sdw",
    "split_formulas",
]
