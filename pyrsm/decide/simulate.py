"""pyrsm.decide.simulate -- Monte Carlo simulation (and repeated simulation).

A :class:`SimulationSpec` lists the inputs (constants, random variables,
sequences, grid search values), the formulas that combine them, optional
custom functions and an optional dataset. :func:`simulate` draws ``runs``
rows and evaluates the formulas; :func:`repeat_simulate` re-runs a simulation
many times and summarises the outputs per simulation or per repetition.

>>> import pyrsm as rsm
>>> sim = rsm.decide.simulate({
...     "runs": 1000, "seed": 1234,
...     "variables": [
...         {"name": "cost", "kind": "constant", "value": 3},
...         {"name": "demand", "kind": "normal", "mean": 2000, "sd": 1000},
...         {"name": "price", "kind": "discrete", "values": [5, 8], "probs": [0.3, 0.7]},
...     ],
...     "formulas": ["profit = demand * (price - cost)", "loss = profit < 0"],
... })
>>> sim.summary()          # doctest: +SKIP

Formulas are Python expressions, as in pyrsm's data filter (``and``,
``or``, ``not``, ``in [...]``, ``**``), with the functions of
:mod:`pyrsm.decide.simfuns` (``ifelse``, ``mean``, ``std``,
``max_horizontal``, ``cum_sum``, ``arg_max``, ``queue_wait``, random draws,
...). Custom functions are Python functions (``def``) that use the same
functions; they run in the same safe interpreter.
"""

from __future__ import annotations

import ast
import itertools
import pprint
from dataclasses import asdict, dataclass, field, replace
from io import StringIO
from typing import Any

import numpy as np
import polars as pl

from pyrsm.decide.simlang import (
    DEFAULT_BUDGET,
    Interpreter,
    SimLangError,
    _check_expression,
    check_habits,
    formula_list,
    split_formulas,
)

#: Variable types, in the order radiant.model lists them
KINDS = (
    "binomial",
    "constant",
    "discrete",
    "exponential",
    "lognormal",
    "normal",
    "poisson",
    "triangular",
    "uniform",
    "sequence",
    "grid",
)

#: The inputs each type needs (``probs`` is optional for discrete)
KIND_FIELDS: dict[str, tuple[str, ...]] = {
    "binomial": ("trials", "prob"),
    "constant": ("value",),
    "discrete": ("values", "probs"),
    "exponential": ("rate",),
    "lognormal": ("meanlog", "sdlog"),
    "normal": ("mean", "sd"),
    "poisson": ("rate",),
    "triangular": ("min", "mode", "max"),
    "uniform": ("min", "max"),
    "sequence": ("min", "max"),
    "grid": ("min", "max", "step"),
}

#: Functions a repeated simulation can apply (radiant.model's list)
REPEAT_FUNCTIONS = (
    "sum", "mean", "median", "min", "max", "std", "var", "std_prop", "var_prop",
    "p01", "p025", "p05", "p10", "p25", "p75", "p90", "p95", "p975", "p99",
    "first", "last",
)  # fmt: skip

# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

Number = float | int | str | None


@dataclass
class Variable:
    """One named simulation input.

    ``kind`` picks the type; only its inputs (see :data:`KIND_FIELDS`) are
    used. Inputs can be numbers or expressions of names defined earlier
    (constants are always evaluated first), e.g. ``mean="base_demand"``.

    - ``constant``: ``value``
    - ``normal``: ``mean``, ``sd``; ``lognormal``: ``meanlog``, ``sdlog``
    - ``uniform``: ``min``, ``max``; ``triangular``: ``min``, ``mode``, ``max``
    - ``binomial``: ``trials``, ``prob``; ``poisson``: ``rate`` (lambda)
    - ``exponential``: ``rate`` (the mean is 1 / rate), e.g. arrivals per minute
    - ``discrete``: ``values`` and ``probs`` (numbers or fractions like "1/3")
    - ``sequence``: ``min`` to ``max`` in as many steps as there are runs
    - ``grid``: ``min`` to ``max`` by ``step``; all grid variables are
      combined and the number of combinations sets the number of runs
    """

    name: str
    kind: str
    value: Number = None
    mean: Number = None
    sd: Number = None
    min: Number = None
    max: Number = None
    step: Number = None
    mode: Number = None
    trials: Number = None
    prob: Number = None
    rate: Number = None
    meanlog: Number = None
    sdlog: Number = None
    values: list[Any] = field(default_factory=list)
    probs: list[Any] = field(default_factory=list)
    description: str = ""


@dataclass
class Formula:
    """``name = expr``, evaluated in order after the variables."""

    name: str
    expr: str
    description: str = ""


@dataclass
class SimulationSpec:
    """A whole simulation.

    ``formulas`` is a list of ``"name = expression"`` strings, evaluated in
    order (``"# ..."`` entries are comments); text with one formula per line
    is accepted too. ``functions`` holds custom function definitions
    (``def ...``) as text; :func:`simulate` also takes them as Python
    functions.
    ``normal_exact`` makes normal variables have exactly the mean and
    standard deviation given; ``correlations`` (``[{"a", "b", "r"}]``)
    correlates normal variables. ``data`` names the dataset whose columns
    formulas can use (pass the data itself to :func:`simulate`).
    """

    name: str = ""
    runs: int = 1000
    seed: int | None = 1234
    variables: list[Variable] = field(default_factory=list)
    formulas: list[str] = field(default_factory=list)
    functions: str = ""
    normal_exact: bool = False
    correlations: list[dict[str, Any]] = field(default_factory=list)
    data: str = ""

    def __post_init__(self) -> None:
        # Formula objects, dicts or text become "name = expression" strings
        self.formulas = _formula_entries(self.formulas)
        if not isinstance(self.functions, str):
            self.functions = _functions_source(self.functions)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SimulationSpec:
        seed = payload.get("seed", 1234)
        spec = cls(
            name=str(payload.get("name", "") or ""),
            runs=int(payload.get("runs", 1000) or 0),
            seed=None if seed in (None, "") else int(seed),
            functions=_functions_source(payload.get("functions")),
            normal_exact=bool(payload.get("normal_exact", False)),
            correlations=[dict(c) for c in payload.get("correlations") or []],
            data=str(payload.get("data", "") or ""),
        )
        fields = set(Variable.__dataclass_fields__)
        for v in payload.get("variables") or []:
            spec.variables.append(Variable(**{k: val for k, val in v.items() if k in fields}))
        spec.formulas = _formula_entries(payload.get("formulas"))
        return spec

    def parsed_formulas(self) -> list[Formula]:
        """The formulas as ``Formula(name, expr)`` (comments left out)."""
        return [Formula(name=n, expr=e) for n, e, _ in split_formulas(self.formulas)]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "runs": self.runs, "seed": self.seed}
        out["variables"] = [_clean_dict(asdict(var)) for var in self.variables]
        out["formulas"] = list(self.formulas)
        if self.functions.strip():
            out["functions"] = self.functions
        if self.normal_exact:
            out["normal_exact"] = True
        if self.correlations:
            out["correlations"] = [dict(c) for c in self.correlations]
        if self.data:
            out["data"] = self.data
        return out


def _clean_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop None / empty-string / empty-list fields for compact JSON."""
    return {k: v for k, v in payload.items() if v is not None and v != "" and v != []}


def _formula_entries(value: Any) -> list[str]:
    """Formulas as a list of strings, from a list or from text."""
    if not value:
        return []
    if isinstance(value, str):
        return formula_list(value)
    out = []
    for item in value:
        if isinstance(item, Formula):
            out.append(f"{item.name} = {item.expr}")
        elif isinstance(item, dict):
            out.append(f"{item.get('name', '')} = {item.get('expr', '')}")
        else:
            out.extend(formula_list(str(item)))
    return out


def parse_formulas(formulas: str | list[str]) -> list[Formula]:
    """``Formula(name, expr)`` for each formula in a list or text."""
    return [Formula(name=n, expr=e) for n, e, _ in split_formulas(_formula_entries(formulas))]


def formulas_text(formulas: list[str]) -> str:
    """The formulas as the text of a formulas box (one per line)."""
    return "\n".join(formulas)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass
class Diagnostic:
    severity: str  # "error" | "warning"
    code: str
    message: str
    target: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FormulaError(ValueError):
    """A formula that cannot be read."""


def compile_formula(expr: str, known: set[str] | None = None) -> ast.Expression:
    """Check that ``expr`` is a valid formula and return its parsed form.

    Unknown names are reported when ``known`` is given. Raises
    :class:`FormulaError` for anything the formula language does not allow.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise FormulaError("formula expression cannot be empty")
    try:
        check_habits(expr)
        tree = ast.parse(expr.strip(), mode="eval")
        _check_expression(tree)
    except SyntaxError as exc:
        raise FormulaError(f"invalid syntax: {exc.msg}") from exc
    except SimLangError as exc:
        raise FormulaError(str(exc)) from exc
    if known is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in known and node.id not in ("True", "False", "pi"):
                parent_call = False
                for other in ast.walk(tree):
                    if isinstance(other, ast.Call) and other.func is node:
                        parent_call = True
                if not parent_call:
                    raise FormulaError(f"unknown name '{node.id}'")
    return tree


# ---------------------------------------------------------------------------
# Drawing the variables
# ---------------------------------------------------------------------------


def _param(interp: Interpreter, var: Variable, field_name: str) -> Any:
    """A variable's input: a number, a fraction ("1/3") or an expression."""
    raw = getattr(var, field_name)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raise SimLangError(f"{var.kind} variable '{var.name}' needs '{field_name}'")
    if isinstance(raw, bool):
        raise SimLangError(f"'{field_name}' of '{var.name}' must be a number")
    if isinstance(raw, int | float):
        return float(raw)
    value = interp.evaluate(str(raw))
    if isinstance(value, np.ndarray):
        return value.astype(float)
    return float(value)


def _list_param(interp: Interpreter, var: Variable, field_name: str) -> list[float]:
    raw = getattr(var, field_name) or []
    if isinstance(raw, str):
        raw = [p for p in raw.replace(",", " ").split() if p]
    out = []
    for item in raw:
        value = interp.evaluate(str(item)) if isinstance(item, str) else item
        out.append(float(value))
    return out


def _grid(interp: Interpreter, grid_vars: list[Variable]) -> dict[str, np.ndarray]:
    """All combinations of the grid variables; the first varies fastest (R's
    ``expand.grid``)."""
    axes = []
    for var in grid_vars:
        lo, hi = _param(interp, var, "min"), _param(interp, var, "max")
        step = 1.0 if var.step in (None, "") else _param(interp, var, "step")
        if step <= 0 or hi < lo:
            raise SimLangError(f"grid '{var.name}': max must be >= min and step > 0")
        count = int(np.floor((hi - lo) / step + 1e-9)) + 1
        axes.append(lo + step * np.arange(count))
    combos = list(itertools.product(*reversed(axes)))
    cols = np.array(combos, dtype=float).T[::-1] if combos else np.empty((len(axes), 0))
    return {var.name: cols[i] for i, var in enumerate(grid_vars)}


def _draw(interp: Interpreter, var: Variable, n: int, exact: bool) -> np.ndarray:
    rng = interp.rng
    kind = var.kind.lower()
    p = lambda f: _param(interp, var, f)  # noqa: E731
    if kind == "constant":
        return np.broadcast_to(p("value"), (n,)).astype(float)
    if kind == "normal":
        mean, sd = p("mean"), p("sd")
        if np.any(np.asarray(sd) <= 0):
            raise SimLangError(f"normal '{var.name}': the standard deviation must be larger than 0")
        if exact:
            z = rng.normal(0, 1, n)
            z = (z - z.mean()) / z.std(ddof=1) if n > 1 else z
            return mean + sd * z
        return rng.normal(mean, sd, n)
    if kind == "lognormal":
        sd = p("sdlog")
        if np.any(np.asarray(sd) <= 0):
            raise SimLangError(f"log normal '{var.name}': the standard deviation must be larger than 0")
        return rng.lognormal(p("meanlog"), sd, n)
    if kind == "uniform":
        lo, hi = p("min"), p("max")
        if np.any(np.asarray(hi) < np.asarray(lo)):
            raise SimLangError(f"uniform '{var.name}': max must be at least min")
        return rng.uniform(lo, hi, n)
    if kind == "triangular":
        lo, mode, hi = p("min"), p("mode"), p("max")
        if not (np.all(lo <= mode) and np.all(mode <= hi) and np.all(lo < hi)):
            raise SimLangError(f"triangular '{var.name}': needs min <= mode <= max and min < max")
        return rng.triangular(lo, mode, hi, n)
    if kind == "binomial":
        prob = p("prob")
        if np.any((np.asarray(prob) < 0) | (np.asarray(prob) > 1)):
            raise SimLangError(f"binomial '{var.name}': the probability must be between 0 and 1")
        return rng.binomial(np.asarray(p("trials")).astype(int), prob, n).astype(float)
    if kind == "poisson":
        rate = p("rate")
        if np.any(np.asarray(rate) < 0):
            raise SimLangError(f"Poisson '{var.name}': lambda cannot be negative")
        return rng.poisson(rate, n).astype(float)
    if kind == "exponential":
        rate = p("rate")
        if np.any(np.asarray(rate) <= 0):
            raise SimLangError(f"exponential '{var.name}': the rate must be larger than 0")
        return rng.exponential(1.0 / np.asarray(rate, dtype=float), n)
    if kind == "discrete":
        values = _list_param(interp, var, "values")
        probs = _list_param(interp, var, "probs") or [1.0 / len(values)] * len(values)
        if not values:
            raise SimLangError(f"discrete '{var.name}' needs values")
        if len(probs) != len(values):
            raise SimLangError(
                f"discrete '{var.name}': {len(values)} values but {len(probs)} probabilities"
            )
        if any(q < 0 for q in probs):
            raise SimLangError(f"probabilities for discrete variable '{var.name}' cannot be negative")
        if not np.isclose(sum(probs), 1.0, atol=1e-6):
            raise SimLangError(
                f"probabilities for discrete variable '{var.name}' do not sum to 1 ({sum(probs):g})"
            )
        return rng.choice(np.asarray(values, dtype=float), n, p=np.asarray(probs) / sum(probs))
    if kind == "sequence":
        return np.linspace(p("min"), p("max"), n)
    raise SimLangError(f"unknown type '{var.kind}' for '{var.name}'. Types: {', '.join(KINDS)}")


def _correlated(
    interp: Interpreter, normals: list[Variable], pairs: dict[frozenset, float], n: int, exact: bool
) -> dict[str, np.ndarray]:
    """Normal variables with the given pairwise correlations (R's sim_cor)."""
    k = len(normals)
    corr = np.eye(k)
    for i, j in itertools.combinations(range(k), 2):
        r = pairs.get(frozenset((normals[i].name, normals[j].name)), 0.0)
        corr[i, j] = corr[j, i] = r
    try:
        chol = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        raise SimLangError(
            "data with these correlations cannot be generated (the correlations are not "
            "consistent with each other). Review the correlations."
        ) from exc
    z = interp.rng.normal(0, 1, (n, k))
    if exact and n > k:
        # columns with mean 0, standard deviation 1 and correlation 0 exactly
        z = z - z.mean(axis=0)
        u, _, vt = np.linalg.svd(z, full_matrices=False)
        z = u @ vt * np.sqrt(n - 1)
    x = z @ chol.T
    out = {}
    for i, var in enumerate(normals):
        mean, sd = _param(interp, var, "mean"), _param(interp, var, "sd")
        if np.any(np.asarray(sd) <= 0):
            raise SimLangError(f"normal '{var.name}': the standard deviation must be larger than 0")
        out[var.name] = mean + sd * x[:, i]
    return out


def _as_column(value: Any, n: int, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 0 or arr.size == 1 and n != 1:
        return np.full(n, arr.reshape(-1)[0] if arr.ndim else arr.item())
    if arr.ndim != 1 or arr.size != n:
        raise SimLangError(
            f"'{name}' has {arr.size:,} values but the simulation has {n:,} rows. Summarise it "
            "(mean, sum, ...) or make it one value per row"
        )
    return arr


# ---------------------------------------------------------------------------
# One simulation
# ---------------------------------------------------------------------------


@dataclass
class _Run:
    columns: dict[str, np.ndarray]
    order: list[str]
    constants: set[str]


def _validate(spec: SimulationSpec, diagnostics: list[Diagnostic]) -> None:
    seen: set[str] = set()
    for var in spec.variables:
        if not var.name or not var.name.isidentifier() or var.name.startswith("__"):
            diagnostics.append(
                Diagnostic(
                    "error",
                    "bad_name",
                    f"variable name '{var.name}' can only have letters, numbers and _ "
                    "(and cannot start with a number)",
                    var.name,
                )
            )
        elif var.name in seen:
            diagnostics.append(
                Diagnostic("error", "duplicate_name", f"variable '{var.name}' is defined more than once", var.name)
            )
        seen.add(var.name)
        if (var.kind or "").lower() not in KINDS:
            diagnostics.append(
                Diagnostic(
                    "error", "bad_kind", f"unknown type '{var.kind}' for '{var.name}'. Types: {', '.join(KINDS)}", var.name
                )
            )
    try:
        parsed = spec.parsed_formulas()
    except SimLangError as exc:
        diagnostics.append(Diagnostic("error", "bad_formula", str(exc)))
        parsed = []
    for f in parsed:
        if not f.name or not f.name.isidentifier() or f.name.startswith("__"):
            diagnostics.append(
                Diagnostic("error", "bad_name", f"formula name '{f.name}' can only have letters, numbers and _", f.name)
            )
    for c in spec.correlations:
        names = {v.name for v in spec.variables if (v.kind or "").lower() == "normal"}
        for key in ("a", "b"):
            if c.get(key) not in names:
                diagnostics.append(
                    Diagnostic(
                        "error",
                        "bad_correlation",
                        f"correlation between '{c.get('a')}' and '{c.get('b')}': both must be normal variables",
                        str(c.get(key)),
                    )
                )
                break
        else:
            try:
                r = float(c.get("r"))
                if not -1 <= r <= 1:
                    raise ValueError
            except (TypeError, ValueError):
                diagnostics.append(
                    Diagnostic(
                        "error", "bad_correlation", f"correlation '{c.get('a')}'-'{c.get('b')}' must be between -1 and 1"
                    )
                )


def _run_once(
    spec: SimulationSpec,
    interp: Interpreter,
    n: int,
    *,
    base: _Run | None = None,
    resample: set[str] | None = None,
    overrides: dict[str, float] | None = None,
) -> _Run:
    """Draw the variables and evaluate the formulas.

    With ``base`` (a repeated simulation) only the variables in ``resample``
    are drawn again and the constants in ``overrides`` replaced; the rest keep
    the values of the base simulation. The formulas are always re-evaluated.
    """
    columns: dict[str, np.ndarray] = {}
    order: list[str] = []
    constants: set[str] = set()
    overrides = overrides or {}

    def put(name: str, arr: np.ndarray, scalar: Any = None) -> None:
        # one number (a constant, or a formula like mean(x)) stays one number
        # in the formulas, so it can be passed to a function as a number
        columns[name] = arr
        interp.names[name] = arr if scalar is None else scalar
        if name not in order:
            order.append(name)

    def one(arr: np.ndarray) -> Any:
        return float(arr[0]) if arr.size and np.all(arr == arr[0]) and arr.dtype != bool else None

    def reuse(var: Variable) -> bool:
        return base is not None and var.name not in (resample or set()) and var.name not in overrides

    variables = list(spec.variables)
    # grid variables first (they fix the rows), then constants, as radiant.model
    for var in variables:
        if var.kind.lower() == "grid":
            put(var.name, base.columns[var.name] if base is not None else interp.names[var.name])
    for var in variables:
        if var.kind.lower() == "constant":
            if var.name in overrides:
                arr = np.full(n, float(overrides[var.name]))
            elif reuse(var):
                arr = base.columns[var.name]
            else:
                arr = _draw(interp, var, n, spec.normal_exact)
            put(var.name, arr, one(arr))
            constants.add(var.name)

    pairs = {}
    for c in spec.correlations:
        pairs[frozenset((c["a"], c["b"]))] = float(c["r"])
    correlated = [
        v
        for v in variables
        if v.kind.lower() == "normal" and any(v.name in key for key in pairs)
    ]
    drawn_together = False
    for var in variables:
        kind = var.kind.lower()
        if kind in ("grid", "constant"):
            continue
        if var in correlated:
            if drawn_together:
                continue
            drawn_together = True
            if base is not None and not any(v.name in (resample or set()) for v in correlated):
                for v in correlated:
                    put(v.name, base.columns[v.name])
            else:
                for name, arr in _correlated(interp, correlated, pairs, n, spec.normal_exact).items():
                    put(name, arr)
            continue
        if reuse(var):
            put(var.name, base.columns[var.name])
        else:
            put(var.name, _draw(interp, var, n, spec.normal_exact))

    for f in spec.parsed_formulas():
        try:
            value = interp.evaluate(f.expr)
        except SimLangError as exc:
            raise SimLangError(f"formula '{f.name} = {f.expr}': {exc}") from exc
        column = _as_column(value, n, f.name)
        scalar = value if np.ndim(value) == 0 and not isinstance(value, bool | np.bool_) else None
        put(f.name, column, None if scalar is None else float(scalar))
        constants.discard(f.name)
    return _Run(columns, order, constants)


def _rows(spec: SimulationSpec, interp: Interpreter) -> int:
    grid_vars = [v for v in spec.variables if (v.kind or "").lower() == "grid"]
    if grid_vars:
        # constants may set the grid's bounds
        for var in spec.variables:
            if var.kind.lower() == "constant":
                interp.names[var.name] = _param(interp, var, "value")
        grid = _grid(interp, grid_vars)
        for name, arr in grid.items():
            interp.names[name] = arr
        for var in spec.variables:
            if var.kind.lower() == "constant":
                interp.names.pop(var.name, None)
        return len(next(iter(grid.values())))
    return int(spec.runs)


def _data_columns(data: Any) -> dict[str, np.ndarray]:
    if data is None:
        return {}
    if not isinstance(data, pl.DataFrame):
        data = pl.from_pandas(data) if hasattr(data, "to_dict") else pl.DataFrame(data)
    out = {}
    for name in data.columns:
        series = data[name]
        if series.dtype.is_numeric() or series.dtype == pl.Boolean:
            out[name] = series.cast(pl.Float64).to_numpy()
    return out


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _column_summary(name: str, arr: np.ndarray, constant: bool) -> dict[str, Any]:
    is_bool = arr.dtype == bool
    values = arr.astype(float)
    values = values[np.isfinite(values)]
    row: dict[str, Any] = {"name": name, "n": int(values.size), "kind": "logical" if is_bool else "numeric"}
    if values.size == 0:
        row.update(dict.fromkeys(["mean", "sd", "min", "p5", "p25", "median", "p75", "p95", "max"]))
        return row
    if not is_bool and (constant or np.all(values == values[0])):
        row["kind"] = "constant"
    q = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    row.update(
        mean=float(values.mean()),
        sd=float(values.std(ddof=1)) if values.size > 1 else 0.0,
        min=float(values.min()),
        p5=float(q[0]),
        p25=float(q[1]),
        median=float(q[2]),
        p75=float(q[3]),
        p95=float(q[4]),
        max=float(values.max()),
    )
    if is_bool:
        row["true_n"] = int(values.sum())
        row["true_prop"] = float(values.mean())
    return row


def _histogram(arr: np.ndarray, bins: int = 30) -> dict[str, list[float]]:
    if arr.dtype == bool:
        ones = int(arr.sum())
        return {"counts": [int(arr.size - ones), ones], "edges": [0.0, 0.5, 1.0]}
    values = arr.astype(float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"counts": [], "edges": []}
    if np.allclose(values.min(), values.max()):
        return {"counts": [int(values.size)], "edges": [float(values[0]) - 0.5, float(values[0]) + 0.5]}
    counts, edges = np.histogram(values, bins=bins)
    return {"counts": [int(c) for c in counts], "edges": [float(e) for e in edges]}


def _fmt(x: float, dec: int) -> str:
    return f"{x:,.{dec}f}"


def _table(header: list[str], rows: list[list[str]], left: int = 1) -> list[str]:
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    lines = []
    for r in [header, *rows]:
        cells = [r[i].ljust(widths[i]) if i < left else r[i].rjust(widths[i]) for i in range(len(r))]
        lines.append(" " + " ".join(cells))
    return lines


def sim_summary(rows: list[dict[str, Any]], dec: int = 4) -> str:
    """Constants, variables and logicals as radiant.model's ``sim_summary``."""
    lines: list[str] = []
    consts = [r for r in rows if r["kind"] == "constant" and r.get("mean") is not None]
    if consts:
        lines.append("Constants:")
        names = [r["name"] for r in consts]
        vals = [_fmt(r["mean"], dec) for r in consts]
        w = [max(len(a), len(b)) for a, b in zip(names, vals)]
        lines.append(" " + " ".join(nm.rjust(x) for nm, x in zip(names, w)))
        lines.append(" " + " ".join(v.rjust(x) for v, x in zip(vals, w)))
        lines.append("")
    nums = [r for r in rows if r["kind"] == "numeric" and r.get("mean") is not None]
    if nums:
        lines.append("Variables:")
        keys = ["n", "mean", "sd", "min", "p25", "median", "p75", "max"]
        body = [
            [r["name"]] + [f"{r['n']:,}"] + [_fmt(r[k], dec) for k in keys[1:]] for r in nums
        ]
        lines += _table(["", "n_obs", *keys[1:]], body)
        lines.append("")
    logic = [r for r in rows if r["kind"] == "logical" and r.get("mean") is not None]
    if logic:
        lines.append("Logicals:")
        body = [[r["name"], f"{r['true_n']:,}", _fmt(r["true_prop"], dec)] for r in logic]
        lines += _table(["", "True (n)", "True (prop)"], body)
        lines.append("")
    return "\n".join(lines).rstrip()


def _plot_columns(data: pl.DataFrame, rows: list[dict[str, Any]], bins: int):
    from plotnine import aes, geom_bar, geom_histogram, ggplot, labs

    from pyrsm.plot_style import FILL, theme_pyrsm
    from pyrsm.plot_utils import compose_plots

    plots = []
    for row in rows:
        name = row["name"]
        if row["kind"] == "constant" or row.get("mean") is None or name not in data.columns:
            continue
        if row["kind"] == "logical":
            df = data.select(pl.col(name).cast(pl.Utf8).str.to_uppercase()).to_pandas()
            p = ggplot(df, aes(x=name)) + geom_bar(fill=FILL, alpha=0.8)
        else:
            df = data.select(name).to_pandas()
            p = ggplot(df, aes(x=name)) + geom_histogram(bins=bins, fill=FILL, color="white", alpha=0.8)
        plots.append(p + labs(x=name, y="") + theme_pyrsm())
    if not plots:
        raise ValueError("Nothing to plot: every column is a constant")
    return compose_plots(plots, ncol=2, width_per_col=4, height_per_row=3)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    spec: SimulationSpec
    data: pl.DataFrame
    diagnostics: list[Diagnostic] = field(default_factory=list)
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    histograms: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    python_code: str = ""
    data_name: str = ""

    @property
    def has_errors(self) -> bool:
        return any(d.severity == "error" for d in self.diagnostics)

    def summary_text(self, dec: int = 4, name: str = "") -> str:
        """The summary as radiant.model prints it."""
        out = StringIO()
        if self.has_errors:
            out.write("Simulation could not run:\n")
            for d in self.diagnostics:
                if d.severity == "error":
                    out.write(f"  {d.message}\n")
            return out.getvalue().rstrip()
        spec = self.spec
        out.write("Simulation\n")
        out.write(f"Simulations: {self.data.height:,}\n")
        out.write(f"Random seed: {'' if spec.seed is None else spec.seed}\n")
        if name or spec.name:
            out.write(f"Sim data   : {name or spec.name}\n")
        labels = {
            "binomial": "Binomial   ", "discrete": "Discrete   ", "exponential": "Exponential",
            "lognormal": "Log normal ", "normal": "Normal     ", "poisson": "Poisson    ",
            "triangular": "Triangular ", "uniform": "Uniform    ", "constant": "Constant   ",
            "grid": "Grid search", "sequence": "Sequence   ",
        }  # fmt: skip
        for kind in KINDS:
            vars_ = [v for v in spec.variables if v.kind.lower() == kind]
            if vars_:
                parts = "; ".join(_variable_text(v) for v in vars_)
                exact = " (exact)" if kind == "normal" and spec.normal_exact else ""
                out.write(f"{labels[kind]}: {parts}{exact}\n")
        if self.data_name or spec.data:
            out.write(f"Data       : {self.data_name or spec.data}\n")
        names = _function_names(spec.functions)
        if names:
            out.write(f"Functions  : {', '.join(names)}\n")
        if spec.formulas:
            out.write("Formulas   :\n")
            for f in spec.formulas:
                out.write(f"\t{f}\n")
        if spec.correlations:
            out.write("Correlations:\n")
            for c in spec.correlations:
                out.write(f"\t{c['a']}-{c['b']}: {float(c['r']):g}\n")
        out.write("\n")
        out.write(sim_summary(self.summary_rows, dec))
        return out.getvalue().rstrip()

    def summary(self, dec: int = 4) -> None:
        """Print the summary (radiant.model's ``summary.simulater``)."""
        print(self.summary_text(dec))

    def plot(self, bins: int = 20):
        """Histograms of the variables that vary and bar charts of the
        logicals (radiant.model's ``plot.simulater``)."""
        return _plot_columns(self.data, self.summary_rows, bins)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "summary": list(self.summary_rows),
            "histograms": dict(self.histograms),
            "python_code": self.python_code,
            "has_errors": self.has_errors,
        }


def _variable_text(v: Variable) -> str:
    kind = v.kind.lower()
    if kind == "discrete":
        vals = " ".join(str(x) for x in v.values)
        probs = " ".join(str(x) for x in v.probs)
        return f"{v.name} {vals} {probs}".strip()
    if kind == "grid" and v.step in (None, ""):
        return f"{v.name} {v.min} {v.max}"
    parts = [getattr(v, f) for f in KIND_FIELDS.get(kind, ())]
    return " ".join([v.name, *(_num_text(p) for p in parts if p is not None)])


def _num_text(x: Any) -> str:
    return f"{x:.6g}" if isinstance(x, float) else str(x)


def _function_names(source: str) -> list[str]:
    if not source.strip():
        return []
    try:
        return [n.name for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
    except SyntaxError:
        return []


def _py(value: Any) -> str:
    """A Python literal with double quotes (as a student would type it)."""
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_py(k)}: {_py(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_py(v) for v in value) + "]"
    return repr(value)


def _list_lines(key: str, values: list[Any], indent: str = "    ") -> list[str]:
    if not values:
        return [f'{indent}"{key}": [],']
    return [f'{indent}"{key}": [', *[f"{indent}    {_py(v)}," for v in values], f"{indent}],"]


def _spec_literal(spec: SimulationSpec, function_names: list[str] | None = None) -> str:
    """The spec as a Python dict literal; functions by name (defined above it)."""
    payload = spec.to_dict()
    payload.pop("functions", None)
    if function_names:
        payload["functions"] = _Literal(f"[{', '.join(function_names)}]")
    lines = ["{"]
    for key, value in payload.items():
        if key in ("variables", "correlations", "formulas"):
            lines += _list_lines(key, value)
        elif isinstance(value, _Literal):
            lines.append(f'    "{key}": {value},')
        else:
            lines.append(f'    "{key}": {_py(value)},')
    lines.append("}")
    return "\n".join(lines)


def _functions_code(*sources: str) -> tuple[list[str], list[str], list[str]]:
    """Custom functions as ordinary Python: (import lines, def lines, names).

    The simulation functions they use (``zeros``, ``mean``, ...) are imported
    from :mod:`pyrsm.decide.simfuns`, so the functions also run (and can be
    tested) outside a simulation.
    """
    from pyrsm.decide import simfuns

    text = "\n\n\n".join(s.strip() for s in sources if s and s.strip())
    if not text:
        return [], [], []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [], [f'functions = """\n{text}\n"""'], []
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    used = sorted(
        {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in simfuns.FUNCTIONS}
        - set(names)
    )
    imports = [f"from pyrsm.decide.simfuns import {', '.join(used)}"] if used else []
    return imports, text.splitlines(), names


def _code_head(spec: SimulationSpec, data_name: str, *extra_functions: str) -> tuple[list[str], list[str]]:
    """Imports, function definitions and ``spec = {...}``; also the names of
    the extra functions (those of a repeated simulation)."""
    imports_sim, defs_sim, names_sim = _functions_code(spec.functions)
    imports_extra, defs_extra, names_extra = _functions_code(*extra_functions)
    used = sorted({n.strip() for ln in imports_sim + imports_extra for n in ln.split(" import ", 1)[1].split(",")})
    lines = ["import pyrsm as rsm"]
    if used:
        lines.append(f"from pyrsm.decide.simfuns import {', '.join(used)}")
    lines.append("")
    blocks = ["\n".join(d) for d in (defs_sim, defs_extra) if d]
    if blocks:
        lines += ["", *"\n\n\n".join(blocks).splitlines(), "", ""]
    if defs_sim and not names_sim:
        lines.append("spec = " + _spec_literal(spec) + '\nspec["functions"] = functions')
    else:
        lines.append(f"spec = {_spec_literal(spec, names_sim)}")
    if data_name:
        lines.append(f"# {data_name}: the input data (a polars or pandas DataFrame)")
    return lines, names_extra


def _generate_python_code(spec: SimulationSpec, data_name: str = "") -> str:
    lines, _ = _code_head(spec, data_name)
    data_arg = f", data={data_name}" if data_name else ""
    lines += [
        "",
        f"sim = rsm.decide.simulate(spec{data_arg})",
        "sim.summary()",
        "sim.plot()",
        "simdat = sim.data",
    ]
    return "\n".join(lines)


def _functions_source(functions: Any) -> str:
    """Custom functions as source text: from text, or from Python functions."""
    import inspect
    import textwrap

    if not functions:
        return ""
    if isinstance(functions, str):
        return functions
    parts = []
    for fn in functions if isinstance(functions, list | tuple) else [functions]:
        if isinstance(fn, str):
            parts.append(fn)
            continue
        try:
            source = textwrap.dedent(inspect.getsource(fn))
        except (OSError, TypeError) as exc:
            raise SimLangError(
                f"cannot read the source of {getattr(fn, '__name__', fn)!r}: define it with def "
                "in a file or notebook cell"
            ) from exc
        if not source.lstrip().startswith("def "):
            raise SimLangError(f"{getattr(fn, '__name__', fn)!r}: write it with def (not lambda)")
        parts.append(source)
    return "\n\n".join(p.strip() for p in parts)


# ---------------------------------------------------------------------------
# simulate()
# ---------------------------------------------------------------------------


def simulate(
    spec: SimulationSpec | dict[str, Any],
    data: Any = None,
    *,
    functions: Any = None,
    budget: int = DEFAULT_BUDGET,
) -> SimulationResult:
    """Run a simulation; see :class:`SimulationSpec` and :class:`Variable`.

    ``data`` is a polars or pandas DataFrame whose (numeric) columns the
    formulas can use ("Input data for calculations"); its columns are not
    part of the simulated data. ``functions`` are custom functions for the
    formulas: Python functions written with ``def`` (or their source text).
    The result keeps any input errors in ``diagnostics`` instead of raising.
    """
    if isinstance(spec, dict):
        spec = SimulationSpec.from_dict(spec)
    diagnostics: list[Diagnostic] = []
    if functions:
        try:
            extra = _functions_source(functions)
        except SimLangError as exc:
            diagnostics.append(Diagnostic("error", "bad_function", str(exc)))
            return SimulationResult(spec=spec, data=pl.DataFrame(), diagnostics=diagnostics)
        spec = replace(spec, functions="\n\n\n".join(f for f in (spec.functions.strip(), extra) if f))
    _validate(spec, diagnostics)
    if not any(v.kind.lower() == "grid" for v in spec.variables) and int(spec.runs) <= 0:
        diagnostics.append(Diagnostic("error", "bad_runs", "the number of simulations must be at least 1"))
    if any(d.severity == "error" for d in diagnostics):
        return SimulationResult(spec=spec, data=pl.DataFrame(), diagnostics=diagnostics)

    rng = np.random.default_rng(spec.seed)
    data_cols = _data_columns(data)
    interp = Interpreter(names=dict(data_cols), rng=rng, budget=budget)
    try:
        interp.define_functions(spec.functions)
        n = _rows(spec, interp)
        if n <= 0:
            raise SimLangError("the simulation has 0 rows")
        interp.nrows = n
        run = _run_once(spec, interp, n)
    except SimLangError as exc:
        diagnostics.append(Diagnostic("error", "simulation_error", str(exc)))
        return SimulationResult(spec=spec, data=pl.DataFrame(), diagnostics=diagnostics)

    data_out = pl.DataFrame({name: run.columns[name] for name in run.order})
    rows = [_column_summary(name, run.columns[name], name in run.constants) for name in run.order]
    hists = {r["name"]: _histogram(run.columns[r["name"]]) for r in rows if r["kind"] != "constant"}
    return SimulationResult(
        spec=spec,
        data=data_out,
        diagnostics=diagnostics,
        summary_rows=rows,
        histograms=hists,
        python_code=_generate_python_code(spec, spec.data),
        data_name=spec.data,
    )


# ---------------------------------------------------------------------------
# repeat_simulate()
# ---------------------------------------------------------------------------


def _apply(fn: str, m: np.ndarray, axis: int) -> np.ndarray:
    m = m.astype(float)
    if fn in ("sum", "mean", "median", "min", "max"):
        return getattr(np, fn)(m, axis=axis)
    if fn == "std":
        return np.std(m, axis=axis, ddof=1)
    if fn == "var":
        return np.var(m, axis=axis, ddof=1)
    if fn in ("std_prop", "var_prop"):
        p = m.mean(axis=axis)
        v = p * (1 - p)
        with np.errstate(invalid="ignore"):
            return np.sqrt(v) if fn == "std_prop" else v
    if fn == "first":
        return np.take(m, 0, axis=axis)
    if fn == "last":
        return np.take(m, -1, axis=axis)
    if fn.startswith("p") and fn[1:].isdigit():
        digits = fn[1:]
        q = int(digits) / (10 ** (len(digits) - 2) * 100) if len(digits) > 2 else int(digits) / 100
        return np.quantile(m, q, axis=axis)
    raise SimLangError(f"unknown function '{fn}'. Use: {', '.join(REPEAT_FUNCTIONS)}, or none")


@dataclass
class RepeatResult:
    """A repeated simulation: the summarised (or full) repeated data."""

    base_spec: SimulationSpec
    reps: int
    by: str
    funs: list[str]
    vars: list[str]
    sum_vars: list[str]
    formulas: list[str]
    grid: list[Variable]
    data: pl.DataFrame
    functions: str = ""
    diagnostics: list[Diagnostic] = field(default_factory=list)
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    histograms: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    python_code: str = ""
    seed: int | None = None
    sims: int = 0

    @property
    def has_errors(self) -> bool:
        return any(d.severity == "error" for d in self.diagnostics)

    def summary_text(self, dec: int = 4, name: str = "", sim_name: str = "") -> str:
        out = StringIO()
        if self.has_errors:
            out.write("Repeated simulation could not run:\n")
            for d in self.diagnostics:
                if d.severity == "error":
                    out.write(f"  {d.message}\n")
            return out.getvalue().rstrip()
        out.write("Repeated simulation\n")
        out.write(f"Simulations   : {self.sims:,}\n")
        out.write(f"Repetitions   : {self.reps:,}\n")
        if self.vars:
            out.write(f"Re-simulated  : {', '.join(self.vars)}\n")
        out.write(f"Group by      : {'Repeat' if self.by == 'rep' else 'Simulation'}\n")
        out.write(f"Function      : {', '.join(self.funs) or 'none'}\n")
        out.write(f"Random  seed  : {'' if self.seed is None else self.seed}\n")
        if sim_name or self.base_spec.name:
            out.write(f"Simulated data: {sim_name or self.base_spec.name}\n")
        if name:
            out.write(f"Repeat data   : {name}\n")
        if self.grid:
            out.write(f"Grid search   : {'; '.join(_variable_text(g) for g in self.grid)}\n")
        names = _function_names(self.functions)
        if names:
            out.write(f"Functions     : {', '.join(names)}\n")
        if self.formulas:
            out.write("Formulas      :\n")
            for f in self.formulas:
                out.write(f"\t{f}\n")
        out.write("\n")
        out.write(sim_summary(self.summary_rows, dec))
        return out.getvalue().rstrip()

    def summary(self, dec: int = 4) -> None:
        """Print the summary (radiant.model's ``summary.repeater``)."""
        print(self.summary_text(dec))

    def plot(self, bins: int = 20):
        """Histograms of the repeated-simulation outputs."""
        return _plot_columns(self.data, [r for r in self.summary_rows if r["name"] not in (".rep", ".sim")], bins)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_spec": self.base_spec.to_dict(),
            "reps": self.reps,
            "sims": self.sims,
            "by": self.by,
            "funs": list(self.funs),
            "vars": list(self.vars),
            "sum_vars": list(self.sum_vars),
            "formulas": list(self.formulas),
            "functions": self.functions,
            "grid": [_clean_dict(asdict(g)) for g in self.grid],
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "summary": list(self.summary_rows),
            "histograms": dict(self.histograms),
            "python_code": self.python_code,
            "has_errors": self.has_errors,
        }


def _repeat_code(rep: RepeatResult, kwargs: dict[str, Any], data_name: str) -> str:
    kwargs = dict(kwargs)
    lines, names = _code_head(rep.base_spec, data_name, kwargs.pop("functions", ""))
    if names:
        kwargs["functions"] = _Literal(f"[{', '.join(names)}]")
    args = []
    for key, value in kwargs.items():
        if key in ("formulas", "grid") and value:
            args += [f"    {key}=[", *[f"        {_py(v)}," for v in value], "    ],"]
        elif isinstance(value, _Literal):
            args.append(f"    {key}={value},")
        else:
            args.append(f"    {key}={_py(value)},")
    if data_name:
        args.append(f"    data={data_name},")
    return "\n".join(
        [
            *lines,
            "",
            "rep = rsm.decide.repeat_simulate(",
            "    spec,",
            *args,
            ")",
            "rep.summary()",
            "rep.plot()",
            "repdat = rep.data",
        ]
    )


class _Literal(str):
    """A name to print as is in generated code (no quotes)."""

    def __repr__(self) -> str:
        return str(self)


def repeat_simulate(
    spec: SimulationSpec | dict[str, Any],
    *,
    reps: int = 12,
    vars: list[str] | None = None,
    sum_vars: list[str] | None = None,
    by: str = "sim",
    fun: str | list[str] = "sum",
    formulas: str | list[str] = "",
    grid: list[Variable] | list[dict[str, Any]] | None = None,
    seed: int | None = None,
    functions: Any = "",
    data: Any = None,
    budget: int = DEFAULT_BUDGET,
) -> RepeatResult:
    """Repeat a simulation ``reps`` times (radiant.model's ``repeater``).

    Each repetition draws the variables in ``vars`` again (the others keep the
    values of the first simulation) and re-evaluates every formula. The
    outputs in ``sum_vars`` are then summarised with each function in ``fun``
    (``"sum"``, ``"mean"``, ``"p95"``, ... or ``"none"`` to keep every row):

    - ``by="sim"`` (group by Simulation): one value per simulation row,
      across the repetitions (e.g. the sum over 365 repeated days);
    - ``by="rep"`` (group by Repeat): one value per repetition, across the
      simulation rows.

    With ``by="rep"`` a ``grid`` (``{"name", "min", "max", "step"}``) replaces
    constants of the simulation in each repetition: every combination is one
    repetition, and the number of combinations sets ``reps``. ``formulas``
    (a list of ``"name = expression"`` strings) run on the summarised data,
    e.g. ``["loss = profit_sum < 0"]``. ``seed`` (default: the simulation's
    seed) makes the repetitions reproducible. ``functions`` are custom
    functions for those formulas (Python functions written with ``def``, or
    their source), next to the simulation's own. Output columns are named
    ``{variable}_{function}``.
    """
    if isinstance(spec, dict):
        spec = SimulationSpec.from_dict(spec)
    funs = [fun] if isinstance(fun, str) else list(fun)
    vars = list(vars or [])
    funs = [f for f in funs if f and f != "none"]
    by = "rep" if by in ("rep", ".rep", "repeat") else "sim"
    grid_vars = [g if isinstance(g, Variable) else Variable(kind="grid", **{k: v for k, v in g.items() if k != "kind"}) for g in grid or []]
    if by == "sim":
        grid_vars = []
    entries = _formula_entries(formulas)
    form_error: str | None = None
    try:
        rep_form = parse_formulas(entries)
    except SimLangError as exc:
        rep_form = []
        form_error = str(exc)
    try:
        functions = _functions_source(functions)
    except SimLangError as exc:
        form_error = str(exc)
        functions = ""
    seed = spec.seed if seed in (None, "") else int(seed)

    result = RepeatResult(
        base_spec=spec, reps=int(reps), by=by, funs=funs, vars=vars, sum_vars=list(sum_vars or []),
        formulas=entries, grid=grid_vars, data=pl.DataFrame(), seed=seed, functions=functions or "",
    )  # fmt: skip
    diagnostics = result.diagnostics
    if form_error:
        diagnostics.append(Diagnostic("error", "bad_repeat_formula", form_error))
    if functions and functions.strip():
        try:
            Interpreter(names={}, rng=np.random.default_rng(0)).define_functions(functions)
        except SimLangError as exc:
            diagnostics.append(Diagnostic("error", "bad_repeat_function", f"repeat functions: {exc}"))
    names = {v.name for v in spec.variables}
    unknown = [v for v in vars if v not in names]
    if unknown:
        diagnostics.append(
            Diagnostic("error", "unknown_variable", f"cannot re-simulate {', '.join(unknown)}: not a simulation variable")
        )
    consts = {v.name for v in spec.variables if v.kind.lower() == "constant"}
    bad_grid = [g.name for g in grid_vars if g.name not in consts]
    if bad_grid:
        diagnostics.append(
            Diagnostic(
                "error", "bad_grid",
                f"grid search in Repeat replaces constants; {', '.join(bad_grid)} is not a constant of the simulation",
            )
        )  # fmt: skip
    if not vars and not grid_vars:
        diagnostics.append(
            Diagnostic(
                "error", "nothing_to_repeat",
                "select variables to re-simulate and/or specify a constant to change with grid search "
                "(when Group by is set to Repeat)",
            )
        )  # fmt: skip
    if not grid_vars and int(reps) <= 0:
        diagnostics.append(Diagnostic("error", "bad_reps", "the number of repetitions must be at least 1"))

    base = simulate(spec, data, budget=budget)
    diagnostics.extend(d for d in base.diagnostics if d.severity == "error")
    if result.has_errors:
        return result
    outputs = result.sum_vars or [f.name for f in spec.formulas] or [v.name for v in spec.variables]
    missing = [v for v in outputs if v not in base.data.columns]
    if missing:
        diagnostics.append(Diagnostic("error", "unknown_output", f"{', '.join(missing)} is not in the simulated data"))
        return result
    result.sum_vars = outputs

    rng = np.random.default_rng(seed)
    interp = Interpreter(names=dict(_data_columns(data)), rng=rng, budget=budget)
    n = base.data.height
    result.sims = n
    base_run = _Run({c: base.data[c].to_numpy() for c in base.data.columns}, list(base.data.columns), set())
    grid_rows: list[dict[str, float]] = [{}]
    if grid_vars:
        g = _grid(interp, grid_vars)
        grid_rows = [{k: float(v[i]) for k, v in g.items()} for i in range(len(next(iter(g.values()))))]
        result.reps = len(grid_rows)
    total = result.reps
    try:
        interp.define_functions(spec.functions)
        interp.nrows = n
        if grid_vars:
            reps_out = [
                _run_once(spec, interp, n, base=base_run, resample=set(vars), overrides=row)
                for row in grid_rows
            ]
        else:
            reps_out = [_run_once(spec, interp, n, base=base_run, resample=set(vars)) for _ in range(total)]
    except SimLangError as exc:
        diagnostics.append(Diagnostic("error", "simulation_error", str(exc)))
        return result

    if not funs:
        cols: dict[str, Any] = {
            ".rep": np.repeat(np.arange(1, total + 1), n),
            ".sim": np.tile(np.arange(1, n + 1), total),
        }
        for v in outputs:
            cols[v] = np.concatenate([r.columns[v] for r in reps_out])
    else:
        key = ".sim" if by == "sim" else ".rep"
        cols = {key: np.arange(1, (n if by == "sim" else total) + 1)}
        if by == "rep":
            for g in grid_vars:
                cols[g.name] = np.array([row[g.name] for row in grid_rows])
        for f in funs:
            for v in outputs:
                m = np.vstack([r.columns[v] for r in reps_out])
                try:
                    cols[f"{v}_{f}"] = _apply(f, m, axis=0 if by == "sim" else 1)
                except SimLangError as exc:
                    diagnostics.append(Diagnostic("error", "bad_function", str(exc)))
                    return result
    if rep_form:
        interp2 = Interpreter(names=dict(cols), rng=rng, budget=budget)
        try:
            interp2.define_functions(spec.functions)
            interp2.define_functions(functions or "")
        except SimLangError as exc:
            diagnostics.append(Diagnostic("error", "bad_repeat_function", f"repeat functions: {exc}"))
            return result
        rows_out = len(next(iter(cols.values())))
        interp2.nrows = rows_out
        for f in rep_form:
            try:
                value = interp2.evaluate(f.expr)
                cols[f.name] = _as_column(value, rows_out, f.name)
                interp2.names[f.name] = cols[f.name]
            except SimLangError as exc:
                diagnostics.append(Diagnostic("error", "bad_repeat_formula", f"formula '{f.name} = {f.expr}': {exc}"))
                return result

    result.data = pl.DataFrame(cols)
    ids = {".rep", ".sim"}
    result.summary_rows = [_column_summary(c, np.asarray(cols[c]), False) for c in cols if c not in ids]
    result.histograms = {r["name"]: _histogram(np.asarray(cols[r["name"]])) for r in result.summary_rows if r["kind"] != "constant"}
    kwargs: dict[str, Any] = {"reps": int(reps), "vars": vars, "sum_vars": outputs, "by": by, "fun": funs or "none"}
    if entries:
        kwargs["formulas"] = entries
    if grid_vars:
        kwargs["grid"] = [_clean_dict(asdict(g)) for g in grid_vars]
    if seed != spec.seed:
        kwargs["seed"] = seed
    if functions and functions.strip():
        kwargs["functions"] = functions
    result.python_code = _repeat_code(result, kwargs, spec.data)
    return result


__all__ = [
    "KINDS",
    "KIND_FIELDS",
    "REPEAT_FUNCTIONS",
    "Diagnostic",
    "Formula",
    "FormulaError",
    "RepeatResult",
    "SimulationResult",
    "SimulationSpec",
    "Variable",
    "compile_formula",
    "parse_formulas",
    "repeat_simulate",
    "sim_summary",
    "simulate",
]


def _make_module_callable() -> None:
    """Keep ``pyrsm.decide.simulate(...)`` callable after submodule imports."""
    import sys
    import types

    class _CallableModule(types.ModuleType):
        def __call__(self, *args, **kwargs):
            return simulate(*args, **kwargs)

    sys.modules[__name__].__class__ = _CallableModule


_make_module_callable()
