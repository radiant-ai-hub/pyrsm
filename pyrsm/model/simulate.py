"""pyrsm.model.simulate — vectorized Monte Carlo simulation workbench.

Typed ``SimulationSpec`` + ``SimulationResult`` pair that turns a small
declarative spec (variables + formulas + seed + runs) into a reproducible
Polars DataFrame of simulated draws, plus per-column summaries and a
short generated-code snippet a student can paste into a notebook.

Formula evaluation is restricted to a small Polars-backed expression
grammar (arithmetic, comparisons, boolean ops, ``ifelse``, ``min``,
``max``, ``abs``, ``sqrt``, ``log``, ``exp``). Arbitrary Python is not
exposed.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Variable:
    """One named simulation input.

    ``kind`` picks the sampler. Only the fields relevant to the chosen
    kind need to be set; the rest stay None and are ignored.

    Supported kinds:
      - ``constant`` (``value``)
      - ``normal`` (``mean``, ``sd``)
      - ``uniform`` (``min``, ``max``)
      - ``binomial`` (``trials``, ``prob``)
      - ``poisson`` (``rate``)
      - ``lognormal`` (``meanlog``, ``sdlog``)
      - ``discrete`` (``values``, optional ``probs``)
      - ``sequence`` (``min``, ``max`` — linearly spaced over runs)
    """

    name: str
    kind: str
    value: float | None = None
    mean: float | None = None
    sd: float | None = None
    min: float | None = None
    max: float | None = None
    trials: int | None = None
    prob: float | None = None
    rate: float | None = None
    meanlog: float | None = None
    sdlog: float | None = None
    values: list[float] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)
    description: str = ""


@dataclass
class Formula:
    """A named formula evaluated row-wise across runs."""

    name: str
    expr: str
    description: str = ""


@dataclass
class SimulationSpec:
    """Whole simulation: a name, run count, seed, and ordered variables/formulas."""

    name: str = ""
    runs: int = 1000
    seed: int = 1234
    variables: list[Variable] = field(default_factory=list)
    formulas: list[Formula] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SimulationSpec:
        spec = cls(
            name=str(payload.get("name", "") or ""),
            runs=int(payload.get("runs", 1000)),
            seed=int(payload.get("seed", 1234)),
        )
        for v in payload.get("variables") or []:
            spec.variables.append(Variable(**v))
        for f in payload.get("formulas") or []:
            spec.formulas.append(Formula(**f))
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "runs": self.runs,
            "seed": self.seed,
            "variables": [_clean_dict(asdict(var)) for var in self.variables],
            "formulas": [_clean_dict(asdict(f)) for f in self.formulas],
        }


def _clean_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop None / empty-string / empty-list fields for compact JSON."""
    return {k: v for k, v in payload.items() if v is not None and v != "" and v != []}


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


# ---------------------------------------------------------------------------
# Safe formula compiler — formula string → Polars expression
# ---------------------------------------------------------------------------


class FormulaError(ValueError):
    """Raised when a formula uses unsupported syntax or unknown names."""


_BINOP_MAP = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}

_CMPOP_MAP = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}


def _compile_node(node: ast.AST, known: set[str]) -> Any:
    if isinstance(node, ast.Expression):
        return _compile_node(node.body, known)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return node.value
        if isinstance(node.value, (int, float)):
            return node.value
        raise FormulaError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in known:
            return pl.col(node.id)
        if node.id == "True":
            return True
        if node.id == "False":
            return False
        raise FormulaError(
            f"unknown name {node.id!r} — declare it as a variable or earlier formula"
        )
    if isinstance(node, ast.UnaryOp):
        operand = _compile_node(node.operand, known)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.Not):
            if hasattr(operand, "__invert__") and not isinstance(operand, (int, float)):
                return ~operand
            return not operand
        raise FormulaError(f"unsupported unary op: {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        left = _compile_node(node.left, known)
        right = _compile_node(node.right, known)
        op_cls = type(node.op)
        if op_cls not in _BINOP_MAP:
            raise FormulaError(f"unsupported binary op: {op_cls.__name__}")
        return _BINOP_MAP[op_cls](left, right)
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise FormulaError("chained comparisons (a < b < c) are not supported")
        left = _compile_node(node.left, known)
        right = _compile_node(node.comparators[0], known)
        op_cls = type(node.ops[0])
        if op_cls not in _CMPOP_MAP:
            raise FormulaError(f"unsupported comparison: {op_cls.__name__}")
        return _CMPOP_MAP[op_cls](left, right)
    if isinstance(node, ast.BoolOp):
        values = [_compile_node(v, known) for v in node.values]
        if isinstance(node.op, ast.And):
            acc = values[0]
            for v in values[1:]:
                acc = acc & v
            return acc
        if isinstance(node.op, ast.Or):
            acc = values[0]
            for v in values[1:]:
                acc = acc | v
            return acc
        raise FormulaError(f"unsupported boolean op: {type(node.op).__name__}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise FormulaError("only named function calls are allowed")
        fname = node.func.id
        if node.keywords:
            raise FormulaError(f"{fname}() does not accept keyword arguments")
        args = [_compile_node(a, known) for a in node.args]
        return _call_function(fname, args)
    raise FormulaError(f"unsupported syntax: {type(node).__name__}")


def _is_polars_expr(value: Any) -> bool:
    return isinstance(value, pl.Expr)


def _call_function(name: str, args: list[Any]) -> Any:
    if name == "ifelse":
        if len(args) != 3:
            raise FormulaError("ifelse(cond, a, b) requires 3 arguments")
        cond, a, b = args
        if not _is_polars_expr(cond):
            cond = pl.lit(cond)
        return pl.when(cond).then(a).otherwise(b)
    if name == "min":
        if not args:
            raise FormulaError("min() requires at least 1 argument")
        if len(args) == 1:
            return args[0]
        return pl.min_horizontal(*args)
    if name == "max":
        if not args:
            raise FormulaError("max() requires at least 1 argument")
        if len(args) == 1:
            return args[0]
        return pl.max_horizontal(*args)
    if name == "abs":
        if len(args) != 1:
            raise FormulaError("abs(x) requires 1 argument")
        x = args[0]
        return x.abs() if _is_polars_expr(x) else abs(x)
    if name == "sqrt":
        if len(args) != 1:
            raise FormulaError("sqrt(x) requires 1 argument")
        x = args[0]
        return x.sqrt() if _is_polars_expr(x) else x**0.5
    if name == "log":
        if len(args) != 1:
            raise FormulaError("log(x) requires 1 argument")
        x = args[0]
        if _is_polars_expr(x):
            return x.log()
        return float(np.log(x))
    if name == "exp":
        if len(args) != 1:
            raise FormulaError("exp(x) requires 1 argument")
        x = args[0]
        if _is_polars_expr(x):
            return x.exp()
        return float(np.exp(x))
    raise FormulaError(f"function {name!r} is not allowed in formulas")


def compile_formula(expr: str, known: set[str]) -> pl.Expr:
    """Compile a formula string into a Polars expression."""
    if not isinstance(expr, str) or not expr.strip():
        raise FormulaError("formula expression cannot be empty")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"invalid syntax: {exc.msg}") from exc
    result = _compile_node(tree, known)
    if not _is_polars_expr(result):
        result = pl.lit(result)
    return result


# ---------------------------------------------------------------------------
# Variable sampling
# ---------------------------------------------------------------------------


def _sample_variable(
    var: Variable,
    runs: int,
    rng: np.random.Generator,
    diagnostics: list[Diagnostic],
) -> np.ndarray | None:
    kind = (var.kind or "").lower()
    try:
        if kind == "constant":
            if var.value is None:
                raise ValueError("constant variable requires 'value'")
            return np.full(runs, float(var.value), dtype=float)
        if kind == "normal":
            if var.mean is None or var.sd is None:
                raise ValueError("normal requires 'mean' and 'sd'")
            if var.sd < 0:
                raise ValueError("normal sd must be non-negative")
            return rng.normal(loc=var.mean, scale=var.sd, size=runs)
        if kind == "uniform":
            if var.min is None or var.max is None:
                raise ValueError("uniform requires 'min' and 'max'")
            if var.max < var.min:
                raise ValueError("uniform max must be >= min")
            return rng.uniform(low=var.min, high=var.max, size=runs)
        if kind == "binomial":
            if var.trials is None or var.prob is None:
                raise ValueError("binomial requires 'trials' and 'prob'")
            if not (0.0 <= float(var.prob) <= 1.0):
                raise ValueError("binomial prob must be in [0, 1]")
            return rng.binomial(n=int(var.trials), p=float(var.prob), size=runs).astype(float)
        if kind == "poisson":
            if var.rate is None or var.rate < 0:
                raise ValueError("poisson requires non-negative 'rate'")
            return rng.poisson(lam=float(var.rate), size=runs).astype(float)
        if kind == "lognormal":
            if var.meanlog is None or var.sdlog is None:
                raise ValueError("lognormal requires 'meanlog' and 'sdlog'")
            if var.sdlog < 0:
                raise ValueError("lognormal sdlog must be non-negative")
            return rng.lognormal(mean=float(var.meanlog), sigma=float(var.sdlog), size=runs)
        if kind == "discrete":
            if not var.values:
                raise ValueError("discrete requires non-empty 'values'")
            probs = list(var.probs) if var.probs else [1.0 / len(var.values)] * len(var.values)
            if len(probs) != len(var.values):
                raise ValueError("discrete 'probs' length must match 'values'")
            if any(p < 0 for p in probs):
                raise ValueError("discrete probabilities must be non-negative")
            ssum = float(sum(probs))
            if ssum <= 0:
                raise ValueError("discrete probabilities must sum to a positive value")
            norm = [p / ssum for p in probs]
            choices = rng.choice(np.asarray(var.values, dtype=float), size=runs, p=norm)
            return np.asarray(choices, dtype=float)
        if kind == "sequence":
            lo = float(var.min) if var.min is not None else 1.0
            hi = float(var.max) if var.max is not None else float(runs)
            return np.linspace(lo, hi, runs)
        raise ValueError(f"unknown distribution kind: {var.kind!r}")
    except Exception as exc:
        diagnostics.append(
            Diagnostic(
                severity="error",
                code="bad_variable",
                message=str(exc),
                target=var.name,
            )
        )
        return None


# ---------------------------------------------------------------------------
# Summary + histogram helpers
# ---------------------------------------------------------------------------


def _column_summary(name: str, series: pl.Series) -> dict[str, Any]:
    is_bool = series.dtype == pl.Boolean
    cast = series.cast(pl.Float64, strict=False).drop_nulls()
    if cast.len() == 0:
        return {
            "name": name,
            "n": 0,
            "kind": "boolean" if is_bool else "numeric",
            "mean": None,
            "sd": None,
            "min": None,
            "p5": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    return {
        "name": name,
        "n": int(cast.len()),
        "kind": "boolean" if is_bool else "numeric",
        "mean": float(cast.mean()),
        "sd": float(cast.std()) if cast.len() > 1 else 0.0,
        "min": float(cast.min()),
        "p5": float(cast.quantile(0.05)),
        "p25": float(cast.quantile(0.25)),
        "median": float(cast.median()),
        "p75": float(cast.quantile(0.75)),
        "p95": float(cast.quantile(0.95)),
        "max": float(cast.max()),
    }


def _histogram(series: pl.Series, bins: int = 30) -> dict[str, list[float]]:
    if series.dtype == pl.Boolean:
        arr = series.cast(pl.Int64, strict=False).drop_nulls().to_numpy()
        zeros = int((arr == 0).sum())
        ones = int((arr != 0).sum())
        return {"counts": [zeros, ones], "edges": [0.0, 0.5, 1.0]}
    values = series.cast(pl.Float64, strict=False).drop_nulls().to_numpy()
    if values.size == 0:
        return {"counts": [], "edges": []}
    if np.allclose(values.min(), values.max()):
        edge = float(values[0])
        return {
            "counts": [int(values.size)],
            "edges": [edge - 0.5, edge + 0.5],
        }
    counts, edges = np.histogram(values, bins=bins)
    return {
        "counts": [int(c) for c in counts],
        "edges": [float(e) for e in edges],
    }


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    spec: SimulationSpec
    data: pl.DataFrame
    diagnostics: list[Diagnostic] = field(default_factory=list)
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    histograms: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    python_code: str = ""

    @property
    def has_errors(self) -> bool:
        return any(d.severity == "error" for d in self.diagnostics)

    def summary(self) -> str:
        if self.has_errors:
            return "Simulation could not run; see diagnostics."
        if not self.summary_rows:
            return "No variables or formulas defined."
        lines = [f"Runs: {self.spec.runs}  Seed: {self.spec.seed}", ""]
        width = max((len(r["name"]) for r in self.summary_rows), default=12)
        for row in self.summary_rows:
            if row["mean"] is None:
                lines.append(f"{row['name']:<{width}}  (no observations)")
                continue
            lines.append(
                f"{row['name']:<{width}}  mean={row['mean']:.3f}  "
                f"sd={row['sd']:.3f}  min={row['min']:.3f}  "
                f"median={row['median']:.3f}  max={row['max']:.3f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "summary": list(self.summary_rows),
            "histograms": dict(self.histograms),
            "python_code": self.python_code,
            "has_errors": self.has_errors,
        }


# ---------------------------------------------------------------------------
# Generated reproducibility code
# ---------------------------------------------------------------------------


def _format_variable_line(var: Variable) -> str | None:
    kind = (var.kind or "").lower()
    if kind == "constant":
        return f"    {var.name!r}: np.full(runs, {float(var.value or 0)}),"
    if kind == "normal":
        return f"    {var.name!r}: rng.normal({var.mean}, {var.sd}, runs),"
    if kind == "uniform":
        return f"    {var.name!r}: rng.uniform({var.min}, {var.max}, runs),"
    if kind == "binomial":
        return f"    {var.name!r}: rng.binomial({int(var.trials or 0)}, {var.prob}, runs),"
    if kind == "poisson":
        return f"    {var.name!r}: rng.poisson({var.rate}, runs),"
    if kind == "lognormal":
        return f"    {var.name!r}: rng.lognormal({var.meanlog}, {var.sdlog}, runs),"
    if kind == "discrete":
        values = list(var.values)
        probs = list(var.probs) if var.probs else [1.0 / len(values)] * len(values)
        return f"    {var.name!r}: rng.choice({values!r}, runs, p={probs!r}),"
    if kind == "sequence":
        lo = var.min if var.min is not None else 1
        hi = var.max if var.max is not None else "runs"
        return f"    {var.name!r}: np.linspace({lo}, {hi}, runs),"
    return None


_CMP_SYMBOL = {
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Gt: ">",
    ast.GtE: ">=",
}

_BIN_SYMBOL = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Mod: "%",
    ast.Pow: "**",
}


def _formula_to_polars_code(node: ast.AST, known: set[str]) -> str:
    """Translate a parsed formula AST into a Polars-native Python source string."""
    if isinstance(node, ast.Expression):
        return _formula_to_polars_code(node.body, known)
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        if node.id in known:
            return f"pl.col({node.id!r})"
        if node.id in {"True", "False"}:
            return node.id
        raise FormulaError(f"unknown name {node.id!r}")
    if isinstance(node, ast.UnaryOp):
        operand = _formula_to_polars_code(node.operand, known)
        if isinstance(node.op, ast.UAdd):
            return f"(+{operand})"
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        if isinstance(node.op, ast.Not):
            return f"(~({operand}))"
        raise FormulaError(f"unsupported unary op: {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        op_cls = type(node.op)
        if op_cls not in _BIN_SYMBOL:
            raise FormulaError(f"unsupported binary op: {op_cls.__name__}")
        left = _formula_to_polars_code(node.left, known)
        right = _formula_to_polars_code(node.right, known)
        return f"({left} {_BIN_SYMBOL[op_cls]} {right})"
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise FormulaError("chained comparisons are not supported")
        op_cls = type(node.ops[0])
        if op_cls not in _CMP_SYMBOL:
            raise FormulaError(f"unsupported comparison: {op_cls.__name__}")
        left = _formula_to_polars_code(node.left, known)
        right = _formula_to_polars_code(node.comparators[0], known)
        return f"({left} {_CMP_SYMBOL[op_cls]} {right})"
    if isinstance(node, ast.BoolOp):
        sep = " & " if isinstance(node.op, ast.And) else " | "
        parts = [_formula_to_polars_code(v, known) for v in node.values]
        return "(" + sep.join(parts) + ")"
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise FormulaError("only named function calls are allowed")
        fname = node.func.id
        args = [_formula_to_polars_code(a, known) for a in node.args]
        if fname == "ifelse":
            cond, a, b = args
            return f"pl.when({cond}).then({a}).otherwise({b})"
        if fname == "min":
            return f"pl.min_horizontal({', '.join(args)})"
        if fname == "max":
            return f"pl.max_horizontal({', '.join(args)})"
        if fname in {"abs", "sqrt", "log", "exp"}:
            if len(args) != 1:
                raise FormulaError(f"{fname}() requires 1 argument")
            return f"({args[0]}).{fname}()"
        raise FormulaError(f"function {fname!r} is not allowed")
    raise FormulaError(f"unsupported syntax: {type(node).__name__}")


def _generate_python_code(spec: SimulationSpec) -> str:
    """Produce a reproducible Polars-native snippet for the spec.

    The output uses chained ``with_columns(name=expr)`` kwargs (no
    ``.alias()``) so reading the code matches the workbook: each formula
    is a name = expression line.
    """
    header = [
        "import numpy as np",
        "import polars as pl",
        "",
        f"rng = np.random.default_rng({int(spec.seed)})",
        f"runs = {int(spec.runs)}",
        "",
    ]

    df_lines: list[str] = ["pl.DataFrame({"]
    for var in spec.variables:
        line = _format_variable_line(var)
        if line is not None:
            df_lines.append(line)
    df_lines.append("})")

    known: set[str] = {v.name for v in spec.variables if v.name}
    formula_lines: list[str] = []
    for formula in spec.formulas:
        if not formula.name:
            continue
        try:
            tree = ast.parse(formula.expr, mode="eval")
            code = _formula_to_polars_code(tree, known)
        except (SyntaxError, FormulaError):
            # Skip bad formulas in the generated snippet; the diagnostics
            # block already tells the user what went wrong.
            continue
        formula_lines.append(f".with_columns({formula.name}={code})")
        known.add(formula.name)

    if not formula_lines:
        body = ["data = " + df_lines[0]] + df_lines[1:]
    else:
        body = ["data = ("]
        body.extend("    " + line for line in df_lines)
        body.extend("    " + line for line in formula_lines)
        body.append(")")

    tail = ["", "print(data.describe())"]
    return "\n".join(header + body + tail)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def simulate(spec: SimulationSpec | dict[str, Any]) -> SimulationResult:
    """Run a vectorized Monte Carlo simulation and return a structured result.

    Always returns a result object even on validation failures so callers
    can surface diagnostics in the UI without exception handling.
    """
    if isinstance(spec, dict):
        spec = SimulationSpec.from_dict(spec)

    runs = int(spec.runs)
    diagnostics: list[Diagnostic] = []

    if runs <= 0:
        diagnostics.append(
            Diagnostic(
                severity="error",
                code="bad_runs",
                message="runs must be a positive integer",
            )
        )
        return SimulationResult(spec=spec, data=pl.DataFrame(), diagnostics=diagnostics)

    rng = np.random.default_rng(int(spec.seed))

    seen: set[str] = set()
    for var in spec.variables:
        if not var.name or not var.name.isidentifier():
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="bad_name",
                    message=(f"variable name {var.name!r} must be a valid Python identifier"),
                    target=var.name,
                )
            )
            continue
        if var.name in seen:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="duplicate_name",
                    message=f"variable {var.name!r} is defined more than once",
                    target=var.name,
                )
            )
        seen.add(var.name)
    for formula in spec.formulas:
        if not formula.name or not formula.name.isidentifier():
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="bad_name",
                    message=(f"formula name {formula.name!r} must be a valid Python identifier"),
                    target=formula.name,
                )
            )
            continue
        if formula.name in seen:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="duplicate_name",
                    message=f"formula {formula.name!r} collides with an existing name",
                    target=formula.name,
                )
            )
        seen.add(formula.name)

    columns: dict[str, np.ndarray] = {}
    for var in spec.variables:
        if not var.name or not var.name.isidentifier():
            continue
        arr = _sample_variable(var, runs, rng, diagnostics)
        if arr is not None:
            columns[var.name] = arr

    data = pl.DataFrame(columns) if columns else pl.DataFrame()
    known = set(columns.keys())

    for formula in spec.formulas:
        if not formula.name or not formula.name.isidentifier():
            continue
        try:
            expr = compile_formula(formula.expr, known)
            data = data.with_columns(expr.alias(formula.name))
            known.add(formula.name)
        except FormulaError as exc:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="bad_formula",
                    message=str(exc),
                    target=formula.name,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="formula_eval_failed",
                    message=str(exc),
                    target=formula.name,
                )
            )

    summary_rows: list[dict[str, Any]] = []
    histograms: dict[str, dict[str, list[float]]] = {}
    if data.width > 0:
        for col in data.columns:
            summary_rows.append(_column_summary(col, data[col]))
            histograms[col] = _histogram(data[col])

    return SimulationResult(
        spec=spec,
        data=data,
        diagnostics=diagnostics,
        summary_rows=summary_rows,
        histograms=histograms,
        python_code=_generate_python_code(spec),
    )


_AGG_FUNCS = {"sum", "mean", "min", "max", "median"}


@dataclass
class RepeatResult:
    """Aggregated outcomes from a repeated simulation (Radiant-style).

    The base spec's ``runs`` defines the per-period draw count
    (e.g. 365 days). ``reps`` is the number of repetitions
    (e.g. 1000 simulated years). For each repetition, the variables
    listed in ``resample`` get fresh draws; all other base-spec
    variables reuse the same per-period draws across every repetition.
    Numeric and boolean per-period columns are aggregated to one value
    per repetition (named ``{col}_{agg}``); optional ``repeat_formulas``
    then run on those aggregated columns.
    """

    base_spec: SimulationSpec
    reps: int
    agg: str
    resample: list[str]
    repeat_formulas: list[Formula]
    data: pl.DataFrame
    diagnostics: list[Diagnostic] = field(default_factory=list)
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    histograms: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    python_code: str = ""

    @property
    def has_errors(self) -> bool:
        return any(d.severity == "error" for d in self.diagnostics)

    @property
    def periods(self) -> int:
        """Per-period draw count for one repetition (= base spec ``runs``)."""
        return int(self.base_spec.runs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_spec": self.base_spec.to_dict(),
            "reps": self.reps,
            "periods": self.periods,
            "agg": self.agg,
            "resample": list(self.resample),
            "repeat_formulas": [{"name": f.name, "expr": f.expr} for f in self.repeat_formulas],
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "summary": list(self.summary_rows),
            "histograms": dict(self.histograms),
            "python_code": self.python_code,
            "has_errors": self.has_errors,
        }


def _aggregate(arr2d: np.ndarray, agg: str) -> np.ndarray:
    if agg == "sum":
        return arr2d.sum(axis=1)
    if agg == "mean":
        return arr2d.mean(axis=1)
    if agg == "min":
        return arr2d.min(axis=1)
    if agg == "max":
        return arr2d.max(axis=1)
    if agg == "median":
        return np.median(arr2d, axis=1)
    raise ValueError(f"unknown aggregation: {agg!r}")


def _generate_repeat_code(
    spec: SimulationSpec,
    reps: int,
    agg: str,
    resample: list[str],
    repeat_formulas: list[Formula],
) -> str:
    base_code = _generate_python_code(spec)
    base_code = base_code.replace("\nprint(data.describe())", "")
    resample_set = set(resample)
    quoted = ", ".join(repr(name) for name in resample)

    # Generate the fresh-draw lines for each resample variable.
    fresh_lines: list[str] = []
    for var in spec.variables:
        if not var.name or var.name not in resample_set:
            continue
        line = _format_variable_line(var)
        if line is None:
            continue
        # Base line says ``    'name': rng.X(args, runs),`` — swap runs for
        # the per-rep stacked size so the resample column gets fresh draws.
        line = line.replace(", runs)", ", reps * runs)").replace(", runs,", ", reps * runs,")
        fresh_lines.append(line)

    base_var_names = [v.name for v in spec.variables if v.name]
    base_formula_names = [f.name for f in spec.formulas if f.name]
    all_base_names = base_var_names + base_formula_names

    suffix_lines = [
        "",
        f"reps = {int(reps)}",
        f"resample = [{quoted}]",
        "",
        "# Build a (reps * runs,) stacked view per column. Resample-listed",
        "# variables get fresh draws; the rest tile the base per-period draws.",
        "_cols = {",
    ]
    for name in base_var_names:
        if name in resample_set:
            continue
        suffix_lines.append(
            f"    {name!r}: np.tile("
            f"data[{name!r}].cast(pl.Float64, strict=False).to_numpy(), reps),"
        )
    suffix_lines.append("}")
    if fresh_lines:
        suffix_lines.append("_fresh = {")
        for fl in fresh_lines:
            suffix_lines.append(fl)
        suffix_lines.append("}")
        suffix_lines.append("for _name, _arr in _fresh.items():")
        suffix_lines.append("    _cols[_name] = np.asarray(_arr, dtype=float)")
    suffix_lines.append("stacked = pl.DataFrame(_cols)")

    # Re-apply base formulas on the stacked dataframe so derived columns
    # (e.g. profit) reflect the fresh per-period draws.
    known_for_formulas: set[str] = set(base_var_names)
    for formula in spec.formulas:
        if not formula.name:
            continue
        try:
            tree = ast.parse(formula.expr, mode="eval")
            code = _formula_to_polars_code(tree, known_for_formulas)
        except (SyntaxError, FormulaError):
            continue
        suffix_lines.append(f"stacked = stacked.with_columns({formula.name}=({code}))")
        known_for_formulas.add(formula.name)

    suffix_lines.extend(
        [
            "",
            f"# Aggregate (reps, runs) → (reps,) by {agg!r}",
            "_agg_fn = "
            + (
                "lambda a: np.median(a, axis=1)"
                if agg == "median"
                else f"lambda a: a.{agg}(axis=1)"
            ),
            "aggregated = {",
            "    f'{col}_" + agg + "': _agg_fn(",
            "        stacked[col].cast(pl.Float64, strict=False).to_numpy().reshape(reps, runs)",
            "    )",
            "    for col in stacked.columns",
            "}",
            "horizon = pl.DataFrame(aggregated)",
        ]
    )
    # Reference the per-formula iteration variable explicitly so ruff
    # doesn't think it's unused after the for-block above runs.
    _ = all_base_names
    # Translate each repeat formula to Polars-native code (pl.col for
    # aggregated names) so the generated snippet is runnable as-is.
    agg_known = {f"{v.name}_{agg}" for v in spec.variables if v.name}
    agg_known.update({f"{f.name}_{agg}" for f in spec.formulas if f.name})
    for formula in repeat_formulas:
        try:
            tree = ast.parse(formula.expr, mode="eval")
            code = _formula_to_polars_code(tree, agg_known)
        except (SyntaxError, FormulaError):
            # Skip uncompilable repeat formulas in the snippet; the runtime
            # diagnostics already surface the error to the user.
            continue
        suffix_lines.append(
            f"horizon = horizon.with_columns({formula.name}=({code}))  # repeat-level"
        )
        agg_known.add(formula.name)
    suffix_lines.append("print(horizon.describe())")
    return base_code + "\n" + "\n".join(suffix_lines)


def repeat_simulate(
    spec: SimulationSpec | dict[str, Any],
    *,
    reps: int,
    agg: str = "sum",
    resample: list[str] | None = None,
    repeat_formulas: list[Formula] | list[dict[str, Any]] | None = None,
) -> RepeatResult:
    """Run a repeated simulation (Radiant-style) and aggregate per rep.

    The base spec's ``runs`` is the per-period draw count for one
    repetition. ``reps`` is the number of repetitions. ``agg`` (sum,
    mean, min, max, median) collapses each per-period column to one
    value per rep, named ``{col}_{agg}``. ``resample`` is the list of
    base-spec variable names that get fresh per-rep draws; everything
    else uses the single base set of per-period draws repeated across
    reps. ``repeat_formulas`` run on the aggregated columns and can
    reference them by their ``{col}_{agg}`` names.
    """
    if isinstance(spec, dict):
        spec = SimulationSpec.from_dict(spec)

    diagnostics: list[Diagnostic] = []
    if reps <= 0:
        diagnostics.append(
            Diagnostic(
                severity="error",
                code="bad_reps",
                message="reps must be a positive integer",
            )
        )
    if agg not in _AGG_FUNCS:
        diagnostics.append(
            Diagnostic(
                severity="error",
                code="bad_agg",
                message=f"agg must be one of {sorted(_AGG_FUNCS)}; got {agg!r}",
            )
        )
    if int(spec.runs) <= 0:
        diagnostics.append(
            Diagnostic(
                severity="error",
                code="bad_runs",
                message=(
                    "base spec runs must be a positive integer; the Simulate "
                    "tab's `runs` defines the per-period draw count."
                ),
            )
        )

    resample_names = [str(name) for name in (resample or []) if str(name)]
    base_var_names = {v.name for v in spec.variables}
    unknown = [name for name in resample_names if name not in base_var_names]
    if unknown:
        diagnostics.append(
            Diagnostic(
                severity="warning",
                code="unknown_resample",
                message=(
                    "resample contains names not in the base spec; ignoring: " + ", ".join(unknown)
                ),
            )
        )
    resample_set = {name for name in resample_names if name in base_var_names}

    # Normalize repeat_formulas to Formula instances
    norm_repeat_formulas: list[Formula] = []
    for entry in repeat_formulas or []:
        if isinstance(entry, Formula):
            norm_repeat_formulas.append(entry)
        elif isinstance(entry, dict):
            norm_repeat_formulas.append(
                Formula(
                    name=str(entry.get("name", "")),
                    expr=str(entry.get("expr", "")),
                )
            )

    if any(d.severity == "error" for d in diagnostics):
        return RepeatResult(
            base_spec=spec,
            reps=int(reps),
            agg=agg,
            resample=resample_names,
            repeat_formulas=norm_repeat_formulas,
            data=pl.DataFrame(),
            diagnostics=diagnostics,
        )

    runs = int(spec.runs)
    rng = np.random.default_rng(int(spec.seed))

    # Sample per-period columns. Non-resample: one set of `runs` draws,
    # tiled across all reps so each rep sees the same per-period values.
    # Resample: fresh `runs * reps` draws.
    columns: dict[str, np.ndarray] = {}
    for var in spec.variables:
        if not var.name or not var.name.isidentifier():
            continue
        if var.name in resample_set:
            arr = _sample_variable(var, runs * int(reps), rng, diagnostics)
        else:
            base_arr = _sample_variable(var, runs, rng, diagnostics)
            arr = np.tile(base_arr, int(reps)) if base_arr is not None else None
        if arr is not None:
            columns[var.name] = arr

    if any(d.severity == "error" for d in diagnostics):
        return RepeatResult(
            base_spec=spec,
            reps=int(reps),
            agg=agg,
            resample=resample_names,
            repeat_formulas=norm_repeat_formulas,
            data=pl.DataFrame(),
            diagnostics=diagnostics,
        )

    per_period = pl.DataFrame(columns) if columns else pl.DataFrame()

    # Apply base spec's per-period formulas vectorized on the (reps*runs,)
    # stacked dataframe. Polars formulas use the same compile path as the
    # single-period simulate.
    known = set(columns.keys())
    for formula in spec.formulas:
        if not formula.name or not formula.name.isidentifier():
            continue
        try:
            expr = compile_formula(formula.expr, known)
            per_period = per_period.with_columns(expr.alias(formula.name))
            known.add(formula.name)
        except FormulaError as exc:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="bad_formula",
                    message=str(exc),
                    target=formula.name,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="formula_eval_failed",
                    message=str(exc),
                    target=formula.name,
                )
            )

    if any(d.severity == "error" for d in diagnostics) or per_period.width == 0:
        return RepeatResult(
            base_spec=spec,
            reps=int(reps),
            agg=agg,
            resample=resample_names,
            repeat_formulas=norm_repeat_formulas,
            data=pl.DataFrame(),
            diagnostics=diagnostics,
            python_code=_generate_repeat_code(
                spec, int(reps), agg, resample_names, norm_repeat_formulas
            ),
        )

    # Reshape and aggregate to one value per rep; name aggregated cols
    # ``{col}_{agg}`` so repeat_formulas can reference them.
    aggregated: dict[str, np.ndarray] = {}
    for col in per_period.columns:
        arr = per_period[col].cast(pl.Float64, strict=False).to_numpy()
        arr2d = arr.reshape(int(reps), runs)
        aggregated[f"{col}_{agg}"] = _aggregate(arr2d, agg)

    data = pl.DataFrame(aggregated)

    # Apply repeat-level formulas — they reference the aggregated names
    # (e.g. ``annual_loss = profit_sum < 0``).
    agg_known = set(data.columns)
    for formula in norm_repeat_formulas:
        if not formula.name or not formula.name.isidentifier():
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="bad_name",
                    message=(
                        f"repeat formula name {formula.name!r} must be a valid Python identifier"
                    ),
                    target=formula.name,
                )
            )
            continue
        try:
            expr = compile_formula(formula.expr, agg_known)
            data = data.with_columns(expr.alias(formula.name))
            agg_known.add(formula.name)
        except FormulaError as exc:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="bad_repeat_formula",
                    message=str(exc),
                    target=formula.name,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="repeat_formula_eval_failed",
                    message=str(exc),
                    target=formula.name,
                )
            )

    summary_rows: list[dict[str, Any]] = []
    histograms: dict[str, dict[str, list[float]]] = {}
    for col in data.columns:
        summary_rows.append(_column_summary(col, data[col]))
        histograms[col] = _histogram(data[col])

    return RepeatResult(
        base_spec=spec,
        reps=int(reps),
        agg=agg,
        resample=resample_names,
        repeat_formulas=norm_repeat_formulas,
        data=data,
        diagnostics=diagnostics,
        summary_rows=summary_rows,
        histograms=histograms,
        python_code=_generate_repeat_code(
            spec, int(reps), agg, resample_names, norm_repeat_formulas
        ),
    )


__all__ = [
    "Variable",
    "Formula",
    "SimulationSpec",
    "SimulationResult",
    "RepeatResult",
    "Diagnostic",
    "FormulaError",
    "compile_formula",
    "simulate",
    "repeat_simulate",
]
