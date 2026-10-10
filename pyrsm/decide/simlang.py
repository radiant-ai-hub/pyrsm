"""The formula and function language of ``pyrsm.decide.simulate``.

Simulation formulas (``profit = demand * (price - cost)``) and custom
functions (``def wealth(win, start): ...``) are Python, read by a small
interpreter over numpy arrays rather than by Python's ``eval``. Formulas are
expressions as in pyrsm's data filter (``and``/``or``/``not``, ``in [...]``,
``**``, ``x if c else y``) with the functions of
:mod:`pyrsm.decide.simfuns`. Functions add assignments, ``if``,
``for``/``while`` loops and ``return``. There is no attribute access,
imports, strings, comprehensions or dunder names, so a formula can only
compute numbers, and every step counts against a budget so a runaway loop
stops with a message instead of hanging.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Steps (expression nodes and statements) one simulation may take
DEFAULT_BUDGET = 20_000_000
_MAX_DEPTH = 60


class SimLangError(ValueError):
    """A formula or function that cannot be read or evaluated."""


class BudgetExceeded(SimLangError):
    """The calculation took more steps than the budget allows."""


# ---------------------------------------------------------------------------
# Habits from other languages, with the Python to use instead
# ---------------------------------------------------------------------------

_HABITS = [
    (re.compile(r"<-"), "use = to assign"),
    (re.compile(r"\^"), "use ** for a power (x ** 2)"),
    (re.compile(r"&&?"), "use and to combine conditions"),
    (re.compile(r"\|\|?"), "use or to combine conditions"),
    (re.compile(r"!(?!=)"), "use not to negate a condition"),
    (re.compile(r"%%"), "use % for the remainder"),
    (re.compile(r"\bTRUE\b|\bFALSE\b"), "use True and False"),
]


def check_habits(expr: str) -> None:
    """Explain syntax from other languages (R, Excel) in Python terms."""
    for pattern, advice in _HABITS:
        if pattern.search(expr):
            raise SimLangError(f"'{expr}': {advice}")


# ---------------------------------------------------------------------------
# Built-in functions
# ---------------------------------------------------------------------------


def _make_builtins(rng: np.random.Generator) -> dict[str, Callable]:
    from pyrsm.decide import simfuns

    out: dict[str, Callable] = {name: getattr(simfuns, name) for name in simfuns.FUNCTIONS}
    out.update(simfuns.make_random(rng))
    out.update({"len": len, "range": lambda *a: range(*[int(x) for x in a]), "int": int, "float": float})
    return out


#: Names a simulation offers in formulas, for the help and error messages
BUILTIN_NAMES = sorted(_make_builtins(np.random.default_rng(0)))


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
}

#: Operators that mean something else in R or Excel
_NOT_PYTHON = {
    ast.BitXor: "use ** for a power (x ** 2)",
    ast.BitAnd: "use and to combine conditions",
    ast.BitOr: "use or to combine conditions",
}

_CMPOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    # membership: price in [6, 8]
    ast.In: lambda a, b: _isin(a, b),
    ast.NotIn: lambda a, b: np.logical_not(_isin(a, b)) if isinstance(a, np.ndarray) else not _isin(a, b),
}


def _isin(a: Any, b: Any) -> Any:
    values = np.ravel(np.asarray(b, dtype=float))
    return np.isin(a, values) if isinstance(a, np.ndarray) else bool(np.isin(a, values))


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
        self.builtins = _make_builtins(self.rng)

    # -- budget -------------------------------------------------------
    def _tick(self) -> None:
        self.steps += 1
        if self.steps > self.budget:
            raise BudgetExceeded(
                f"the calculation took more than {self.budget:,} steps. Use functions that work "
                "on a whole column at once (cum_sum, max_horizontal, queue_wait, ...) instead of loops, "
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
                advice = _NOT_PYTHON.get(type(node.op), "that kind of update is not supported")
                raise SimLangError(f"line {node.lineno}: {advice}")
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
                raise SimLangError(_NOT_PYTHON.get(type(node.op), f"operator {type(node.op).__name__} is not supported"))
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
            check_habits(expr)
            tree = ast.parse(expr.strip(), mode="eval")
        except SyntaxError as exc:
            raise SimLangError(
                f"cannot read '{expr}': is something missing or mistyped? ({exc.msg})"
            ) from exc
        _check_expression(tree)
        return self.eval(tree, {})


def _shape_message(left: Any, right: Any) -> str:
    sizes = [np.size(x) for x in (left, right) if isinstance(x, np.ndarray)]
    return (
        f"cannot combine columns with {sizes[0]:,} and {sizes[-1]:,} values. Columns from the "
        "input data have one value per data row; summarise them (mean, sd, ...) or use them "
        "with a function such as weighted_sum_std()"
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

_ASSIGN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*=(?!=)\s*(.*)$", re.S)


def formula_list(text: str) -> list[str]:
    """The formulas in a text box, one string per formula.

    A formula is one line (or ``;``-separated part of a line); it continues on
    the next line while a bracket is open. Comment lines (``# ...``) are kept
    as entries of their own so they stay with the formulas they describe.
    """
    out: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        if not buffer and raw.strip().startswith("#"):
            out.append(raw.strip())
            continue
        code, _, note = raw.partition("#")
        for part in code.split(";"):
            if not part.strip():
                continue
            buffer = f"{buffer} {part.strip()}" if buffer else part.strip()
            depth = buffer.count("(") - buffer.count(")") + buffer.count("[") - buffer.count("]")
            if depth <= 0:
                out.append(f"{buffer}  # {note.strip()}" if note.strip() else buffer)
                buffer = ""
                note = ""
    if buffer:
        out.append(buffer)
    return out


def split_formulas(formulas: str | list[str]) -> list[tuple[str, str, int]]:
    """``(name, expression, number)`` for each formula (comments skipped).

    ``formulas`` is a list of ``"name = expression"`` strings, or text with
    one formula per line (see :func:`formula_list`).
    """
    entries = formula_list(formulas) if isinstance(formulas, str) else list(formulas)
    out: list[tuple[str, str, int]] = []
    for number, entry in enumerate(entries, start=1):
        stmt = str(entry).split("#", 1)[0].strip()
        if not stmt:
            continue
        if "<-" in stmt.split("=", 1)[0] or re.match(r"^\s*[A-Za-z_]\w*\s*<-", stmt):
            raise SimLangError(f"formula {number}, '{stmt}': use = to assign")
        m = _ASSIGN.match(stmt)
        if not m:
            raise SimLangError(
                f"formula {number}, '{stmt}': needs a name and '=', e.g. profit = demand * price"
            )
        name, expr = m.groups()
        if "." in name:
            raise SimLangError(f"formula {number}: use _ instead of . in names ('{name}')")
        if not expr.strip():
            raise SimLangError(f"formula {number}: '{name} =' has nothing after the '='")
        out.append((name, expr.strip(), number))
    return out


__all__ = [
    "BUILTIN_NAMES",
    "BudgetExceeded",
    "Interpreter",
    "SimLangError",
    "UserFunction",
    "check_habits",
    "formula_list",
    "split_formulas",
]
