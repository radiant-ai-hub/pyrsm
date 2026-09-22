"""The data scope: which rows an analysis runs on.

Radiant-for-R threads the scope through every analysis function --
``get_data(dataset, vars, filt = data_filter, arr = arr, rows = rows)`` -- so a
script reproduces what was on the screen, and the printed summary says which
rows it used. This is that, for pyrsm: every class takes ``data_filter``,
``sort`` and ``slice``, applies them in that order, and remembers them.

The three are ordinary strings, so an analysis is one call that carries its own
scope and can be copied, saved, re-run and read::

    rsm.model.regress(
        {"diamonds": diamonds},
        rvar="price", evar=["carat", "clarity"],
        data_filter="price > 5000 and cut in ['Ideal', 'Premium']",
        sort="clarity -price",
        slice="1:100",
    )

The filter is a Python expression evaluated against the data with polars. A
bare name is a column, a call is an operation on one:

===========================  =================================================
``price > 5000``             comparison: ``>`` ``<`` ``>=`` ``<=`` ``==`` ``!=``
``a and b``, ``a or b``      combine; ``not a`` negates
``cut in ['Ideal']``         membership; ``not in`` excludes
``is_null(price)``           missing; ``is_not_null(price)`` is present
``contains(cut, 'ood')``     text; also ``starts_with``, ``ends_with``
``price > mean(price)``      compare a row against a column statistic
``year(date) == 2024``       part of a date
===========================  =================================================

``sort`` is a space-separated list of columns, ``-`` for descending, as in
``"clarity -price"``. ``slice`` takes 1-based row numbers: ``"1:100"`` is the
first hundred, ``"100"`` is also the first hundred, ``"-100"`` the last
hundred, and ``"1 5 10"`` those three rows. Both match the pyrsm DSL's
``/sort`` and ``/slice``.
"""

from __future__ import annotations

import ast
import datetime as _dt
from typing import Any

import polars as pl

__all__ = [
    "ScopeError",
    "get_data",
    "apply_scope",
    "filter_expr",
    "filter_code",
    "sort_spec",
    "sort_code",
    "slice_code",
    "scope_lines",
    "print_scope",
]


class ScopeError(ValueError):
    """The filter, sort or slice could not be read."""


# ---------------------------------------------------------------------------
# Filter expressions
# ---------------------------------------------------------------------------

#: ``f(col)`` -> ``pl.col(...).f()``. One name per operation, polars' own.
_METHODS = {
    "log": "log",
    "exp": "exp",
    "sqrt": "sqrt",
    "abs": "abs",
    "floor": "floor",
    "ceil": "ceil",
    "is_null": "is_null",
    "is_not_null": "is_not_null",
    "mean": "mean",
    "median": "median",
    "min": "min",
    "max": "max",
    "sum": "sum",
    "std": "std",
    "var": "var",
    "n_unique": "n_unique",
    "first": "first",
    "last": "last",
    "lower": None,  # str accessor, below
    "upper": None,
}

#: ``f(col)`` -> ``pl.col(...).str.f()``
_STR_METHODS = {
    "lower": "to_lowercase",
    "upper": "to_uppercase",
}

#: ``f(col, 'text')`` -> ``pl.col(...).str.f('text')``
_STR_ARG_METHODS = {
    "contains": "contains",
    "starts_with": "starts_with",
    "ends_with": "ends_with",
}

#: ``f(col)`` / ``f(col, format)`` -> ``pl.col(...).str.f(...)``
_PARSE_METHODS = {
    "to_date": "to_date",
    "to_datetime": "to_datetime",
}

#: ``f(col)`` -> ``pl.col(...).dt.f()``
_DT_METHODS = {
    "year": "year",
    "month": "month",
    "day": "day",
    "week": "week",
    "weekday": "weekday",
    "hour": "hour",
    "minute": "minute",
    "second": "second",
}

_COMPARISONS = {
    ast.Gt: ">",
    ast.Lt: "<",
    ast.GtE: ">=",
    ast.LtE: "<=",
    ast.Eq: "==",
    ast.NotEq: "!=",
}

_ARITHMETIC = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Mod: "%",
    ast.Pow: "**",
}


def _known_functions() -> str:
    names = sorted(
        set(_METHODS)
        | set(_STR_ARG_METHODS)
        | set(_PARSE_METHODS)
        | set(_DT_METHODS)
        | {"round"}
    )
    return ", ".join(names)


def _parse(expression: str) -> ast.expr:
    try:
        return ast.parse(expression.strip(), mode="eval").body
    except SyntaxError as exc:
        raise ScopeError(f"Could not read the filter {expression!r}: {exc.msg}") from exc


def _columns(node: ast.expr, found: set[str]) -> set[str]:
    """Every column the expression names.

    A function's own name is an ``ast.Name`` too, so a plain walk reported
    ``is_null`` as a missing column and refused the filter that used it.
    """
    if isinstance(node, ast.Call):
        for argument in node.args:
            _columns(argument, found)
        return found
    if isinstance(node, ast.Name):
        found.add(node.id)
        return found
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr):
            _columns(child, found)
    return found


def _check_columns(node: ast.expr, schema: dict[str, Any] | None) -> None:
    if schema is None:
        return
    known = set(schema)
    for name in sorted(_columns(node, set())):
        if name not in known:
            raise ScopeError(f"Unknown column in filter: {name}")


def _literal(node: ast.expr) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal(node.operand)
    raise ScopeError("Expected a number, string or True/False")


def _list_values(node: ast.expr) -> list[Any]:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_literal(item) for item in node.elts]
    raise ScopeError("`in` needs a list, e.g. cut in ['Ideal', 'Premium']")


def _call_name(node: ast.Call) -> str:
    if not isinstance(node.func, ast.Name):
        raise ScopeError(
            "A filter is written with functions, not methods: "
            "is_null(price), not price.is_null()"
        )
    return node.func.id


def _column_dtype(node: ast.expr, schema: dict[str, Any] | None):
    """The dtype of ``node`` when it is a bare column name."""
    if schema is None or not isinstance(node, ast.Name):
        return None
    return schema.get(node.id)


def _is_temporal(dtype) -> bool:
    if dtype is None:
        return False
    try:
        return bool(dtype.is_temporal())
    except AttributeError:
        return dtype in {pl.Date, pl.Datetime, pl.Time}


def _as_literal(value: Any, dtype) -> Any:
    """Read a written value against the column it is compared with.

    Polars refuses to compare a date column with a string, so ``d <
    '2014-6-1'`` has to become a real ``date`` before it reaches the frame --
    and a date written out is the only form a student would use.
    """
    if not _is_temporal(dtype) or not isinstance(value, str):
        return value
    text = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            moment = _dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
        return moment.date() if dtype == pl.Date else moment
    raise ScopeError(f"{value!r} is not a date. Write it as YYYY-MM-DD.")


def _to_expr(node: ast.expr, schema: dict[str, Any] | None = None) -> pl.Expr:
    """The polars expression for one node."""
    if isinstance(node, ast.BoolOp):
        parts = [_to_expr(value, schema) for value in node.values]
        out = parts[0]
        for part in parts[1:]:
            out = out & part if isinstance(node.op, ast.And) else out | part
        return out

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return ~_to_expr(node.operand)
        if isinstance(node.op, ast.USub):
            return -_to_expr(node.operand)
        raise ScopeError("Only `not` and `-` may come before an expression")

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            # `1 < price < 10` reads as two comparisons; write them with `and`
            # so there is one way to say it.
            raise ScopeError(
                "Write a range as two comparisons joined by `and`, "
                "e.g. price > 1 and price < 10"
            )
        op, right = node.ops[0], node.comparators[0]
        left = _to_expr(node.left, schema)
        dtype = _column_dtype(node.left, schema)
        if isinstance(op, ast.In):
            return left.is_in([_as_literal(v, dtype) for v in _list_values(right)])
        if isinstance(op, ast.NotIn):
            return ~left.is_in([_as_literal(v, dtype) for v in _list_values(right)])
        symbol = _COMPARISONS.get(type(op))
        if symbol is None:
            raise ScopeError(f"Unsupported comparison: {type(op).__name__}")
        if dtype is not None and isinstance(right, ast.Constant):
            return _compare(left, symbol, pl.lit(_as_literal(right.value, dtype)))
        return _compare(left, symbol, _to_expr(right, schema))

    if isinstance(node, ast.BinOp):
        symbol = _ARITHMETIC.get(type(node.op))
        if symbol is None:
            raise ScopeError(f"Unsupported operator: {type(node.op).__name__}")
        return _arithmetic(_to_expr(node.left), symbol, _to_expr(node.right))

    if isinstance(node, ast.Call):
        return _call_expr(node, schema)

    if isinstance(node, ast.Name):
        return pl.col(node.id)

    if isinstance(node, ast.Constant):
        return pl.lit(node.value)

    raise ScopeError(f"Unsupported expression: {ast.dump(node)[:60]}")


def _compare(left: pl.Expr, symbol: str, right: pl.Expr) -> pl.Expr:
    return {
        ">": lambda: left > right,
        "<": lambda: left < right,
        ">=": lambda: left >= right,
        "<=": lambda: left <= right,
        "==": lambda: left == right,
        "!=": lambda: left != right,
    }[symbol]()


def _arithmetic(left: pl.Expr, symbol: str, right: pl.Expr) -> pl.Expr:
    return {
        "+": lambda: left + right,
        "-": lambda: left - right,
        "*": lambda: left * right,
        "/": lambda: left / right,
        "%": lambda: left % right,
        "**": lambda: left**right,
    }[symbol]()


def _call_expr(node: ast.Call, schema: dict[str, Any] | None = None) -> pl.Expr:
    name = _call_name(node)
    if name == "round":
        if not node.args:
            raise ScopeError("round() needs a column, e.g. round(price, 2)")
        decimals = _literal(node.args[1]) if len(node.args) > 1 else 0
        return _to_expr(node.args[0]).round(int(decimals))
    if name in _STR_ARG_METHODS:
        if len(node.args) != 2:
            raise ScopeError(f"{name}() needs a column and text, e.g. {name}(cut, 'ood')")
        target = _to_expr(node.args[0]).cast(pl.Utf8).str
        return getattr(target, _STR_ARG_METHODS[name])(_literal(node.args[1]))
    if name in _PARSE_METHODS:
        if not node.args:
            raise ScopeError(f"{name}() needs a column, e.g. {name}(date)")
        target = _to_expr(node.args[0]).cast(pl.Utf8).str
        method = getattr(target, _PARSE_METHODS[name])
        return method(_literal(node.args[1])) if len(node.args) > 1 else method()
    if len(node.args) != 1:
        raise ScopeError(f"{name}() takes one column")
    inner = _to_expr(node.args[0])
    if name in _STR_METHODS:
        return getattr(inner.cast(pl.Utf8).str, _STR_METHODS[name])()
    if name in _DT_METHODS:
        return getattr(inner.dt, _DT_METHODS[name])()
    method = _METHODS.get(name)
    if method is None:
        raise ScopeError(f"Unknown function {name}(). Available: {_known_functions()}")
    return getattr(inner, method)()


def _literal_code(node: ast.expr, dtype, schema: dict[str, Any] | None) -> str:
    """One value, written so the generated line runs.

    A date has to arrive at polars as a date, so the snippet parses it
    rather than passing the string the student wrote -- which polars refuses
    to compare with a date column.
    """
    if dtype is not None and isinstance(node, ast.Constant):
        value = _as_literal(node.value, dtype)
        if isinstance(value, (_dt.date, _dt.datetime)):
            method = "to_datetime" if isinstance(value, _dt.datetime) else "to_date"
            return f"pl.lit({node.value!r}).str.{method}()"
    return _to_code(node, schema)


def _literal_code_list(node: ast.expr, dtype) -> str:
    values = _list_values(node)
    if dtype is not None and any(
        isinstance(_as_literal(v, dtype), (_dt.date, _dt.datetime)) for v in values
    ):
        method = "to_datetime" if dtype == pl.Datetime else "to_date"
        rendered = ", ".join(repr(v) for v in values)
        return f"pl.Series([{rendered}]).str.{method}()"
    return repr(values)


def _to_code(node: ast.expr, schema: dict[str, Any] | None = None) -> str:
    """polars source for one node, for generated code."""
    if isinstance(node, ast.BoolOp):
        joiner = " & " if isinstance(node.op, ast.And) else " | "
        return joiner.join(f"({_to_code(value, schema)})" for value in node.values)

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return f"~({_to_code(node.operand, schema)})"
        return f"-{_to_code(node.operand, schema)}"

    if isinstance(node, ast.Compare):
        op, right = node.ops[0], node.comparators[0]
        left = _to_code(node.left, schema)
        dtype = _column_dtype(node.left, schema)
        if isinstance(op, ast.In):
            return f"{left}.is_in({_literal_code_list(right, dtype)})"
        if isinstance(op, ast.NotIn):
            return f"~{left}.is_in({_literal_code_list(right, dtype)})"
        return f"{left} {_COMPARISONS[type(op)]} {_literal_code(right, dtype, schema)}"

    if isinstance(node, ast.BinOp):
        return (
            f"{_to_code(node.left, schema)} {_ARITHMETIC[type(node.op)]} "
            f"{_to_code(node.right, schema)}"
        )

    if isinstance(node, ast.Call):
        name = _call_name(node)
        if name == "round":
            decimals = _literal(node.args[1]) if len(node.args) > 1 else 0
            return f"{_to_code(node.args[0], schema)}.round({int(decimals)})"
        if name in _STR_ARG_METHODS:
            text = _literal(node.args[1])
            method = _STR_ARG_METHODS[name]
            return f"{_to_code(node.args[0], schema)}.cast(pl.Utf8).str.{method}({text!r})"
        if name in _PARSE_METHODS:
            method = _PARSE_METHODS[name]
            inner_code = f"{_to_code(node.args[0], schema)}.cast(pl.Utf8).str.{method}"
            if len(node.args) > 1:
                return f"{inner_code}({_literal(node.args[1])!r})"
            return f"{inner_code}()"
        inner = _to_code(node.args[0], schema)
        if name in _STR_METHODS:
            return f"{inner}.cast(pl.Utf8).str.{_STR_METHODS[name]}()"
        if name in _DT_METHODS:
            return f"{inner}.dt.{_DT_METHODS[name]}()"
        return f"{inner}.{_METHODS[name]}()"

    if isinstance(node, ast.Name):
        return f"pl.col({node.id!r})"

    if isinstance(node, ast.Constant):
        return repr(node.value)

    raise ScopeError("Unsupported expression")


def filter_expr(data_filter: str, schema: dict[str, Any] | None = None) -> pl.Expr | None:
    """The polars predicate for ``data_filter``, or ``None`` when it is empty."""
    if not str(data_filter or "").strip():
        return None
    node = _parse(data_filter)
    _check_columns(node, schema)
    return _to_expr(node, schema)


def filter_code(data_filter: str, schema: dict[str, Any] | None = None) -> str:
    """``data_filter`` as polars source, for generated code."""
    if not str(data_filter or "").strip():
        return ""
    return _to_code(_parse(data_filter), schema)


# ---------------------------------------------------------------------------
# Sort
# ---------------------------------------------------------------------------


def sort_spec(sort: str) -> tuple[list[str], list[bool]]:
    """``"clarity -price"`` -> ``(["clarity", "price"], [False, True])``."""
    columns: list[str] = []
    descending: list[bool] = []
    for token in str(sort or "").replace(",", " ").split():
        if token.startswith("-"):
            columns.append(token[1:])
            descending.append(True)
        else:
            columns.append(token)
            descending.append(False)
    return columns, descending


def sort_code(sort: str, frame: str = "data") -> str:
    columns, descending = sort_spec(sort)
    if not columns:
        return ""
    return f"{frame} = {frame}.sort({columns!r}, descending={descending!r})"


# ---------------------------------------------------------------------------
# Slice
# ---------------------------------------------------------------------------


def _slice_tokens(spec: str) -> list[str]:
    return str(spec or "").replace(",", " ").split()


def _apply_slice(df: pl.DataFrame, spec: str) -> pl.DataFrame:
    """1-based row numbers, matching the DSL's ``/slice``."""
    tokens = _slice_tokens(spec)
    if not tokens:
        return df
    if ":" in tokens[0]:
        parts = tokens[0].split(":")
        if len(parts) > 3:
            raise ScopeError(f"Invalid row range: {tokens[0]}. Use start:stop or start:stop:step")
        start = int(parts[0]) if parts[0] else 1
        stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        start_0 = max(start - 1, 0)
        if stop is None and step == 1:
            return df.slice(start_0)
        stop = df.height if stop is None else stop
        return df[list(range(start_0, min(stop, df.height), step))]
    if len(tokens) >= 3:
        return df[[max(int(token) - 1, 0) for token in tokens]]
    if len(tokens) == 2:
        offset, length = int(tokens[0]), int(tokens[1])
        return df.slice(max(offset - 1, 0), length)
    count = int(tokens[0])
    return df.tail(abs(count)) if count < 0 else df.head(count)


def slice_code(spec: str, frame: str = "data") -> str:
    tokens = _slice_tokens(spec)
    if not tokens:
        return ""
    if ":" in tokens[0]:
        parts = tokens[0].split(":")
        start = int(parts[0]) if parts[0] else 1
        stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        start_0 = max(start - 1, 0)
        if stop is None and step == 1:
            return f"{frame} = {frame}.slice({start_0})"
        if step == 1:
            return f"{frame} = {frame}.slice({start_0}, {max(stop - start_0, 0)})"
        return f"{frame} = {frame}[list(range({start_0}, {stop}, {step}))]"
    if len(tokens) >= 3:
        return f"{frame} = {frame}[{[max(int(t) - 1, 0) for t in tokens]!r}]"
    if len(tokens) == 2:
        offset, length = int(tokens[0]), int(tokens[1])
        return f"{frame} = {frame}.slice({max(offset - 1, 0)}, {length})"
    count = int(tokens[0])
    if count < 0:
        return f"{frame} = {frame}.tail({abs(count)})"
    return f"{frame} = {frame}.head({count})"


# ---------------------------------------------------------------------------
# Putting the three together
# ---------------------------------------------------------------------------


def apply_scope(
    df: pl.DataFrame,
    data_filter: str = "",
    sort: str = "",
    slice: str = "",  # noqa: A002 - the argument name every class publishes
) -> pl.DataFrame:
    """Filter, then sort, then slice -- in that order, always.

    The order is the one thing about a scope that cannot be left to taste: a
    slice of unsorted rows and a slice of sorted rows are different rows, and
    a reader has to know which they got without running it.
    """
    out = df
    predicate = filter_expr(data_filter, dict(df.schema))
    if predicate is not None:
        try:
            out = out.filter(predicate)
        except Exception as exc:  # pragma: no cover - surfaced to the user
            raise ScopeError(f"Could not apply the filter {data_filter!r}: {exc}") from exc
    columns, descending = sort_spec(sort)
    if columns:
        missing = [column for column in columns if column not in out.columns]
        if missing:
            raise ScopeError(f"Unknown column in sort: {', '.join(missing)}")
        out = out.sort(columns, descending=descending)
    return _apply_slice(out, slice)


def get_data(
    data,
    vars: list[str] | None = None,
    data_filter: str = "",
    sort: str = "",
    slice: str = "",  # noqa: A002
) -> tuple[str, pl.DataFrame]:
    """``(name, frame)`` for an analysis: the scope applied, columns kept.

    The pyrsm counterpart of radiant.data's ``get_data``. ``data`` is a
    DataFrame or a ``{name: DataFrame}`` dict, as everywhere else in pyrsm.
    """
    from pyrsm.utils import check_dataframe

    if isinstance(data, dict):
        name = list(data.keys())[0]
        frame = data[name]
    else:
        name, frame = "Not provided", data
    frame = check_dataframe(frame)
    frame = apply_scope(frame, data_filter, sort, slice)
    if vars:
        keep = [column for column in vars if column in frame.columns]
        if keep:
            frame = frame.select(keep)
    return name, frame


def scope_lines(obj, width: int = 21) -> list[str]:
    """The scope lines a summary prints, padded to that summary's label width.

    A summary that does not say it was filtered is a summary a reader will
    take for the whole dataset. ``n = 100`` is not a fact about the data
    unless the filter that produced it is on the page next to it.
    """
    lines = []
    for label, value in (
        ("Filter", getattr(obj, "data_filter", "")),
        ("Sort", getattr(obj, "sort", "")),
        ("Slice", getattr(obj, "slice", "")),
    ):
        if str(value or "").strip():
            lines.append(f"{label.ljust(width)}: {value}")
    return lines


def print_scope(obj, width: int = 21) -> None:
    """Print the scope lines, if there are any."""
    for line in scope_lines(obj, width):
        print(line)
