"""Linear and (mixed-)integer programming with sensitivity analysis.

A small, self-contained modeling layer on top of :mod:`pulp`. It ingests a
plain JSON/``dict`` problem definition (the same contract the Svelte/Django
"Optimization Studio" front-end exchanges with the backend), validates it into
typed dataclasses, solves it with the bundled CBC solver, and exposes the
results an analyst (or a UI) needs: the optimal decision vector, the objective
value, per-constraint **slack** and **shadow prices** (dual values), and
per-variable **reduced costs**.

The public surface mirrors the rest of ``pyrsm.decide`` (see :class:`dtree`):
structured :class:`Diagnostic` objects instead of raw exceptions, ``errors`` /
``warnings`` / ``is_solved`` convenience properties, a ``summary()`` method,
tabular views as Polars DataFrames, a ``to_dict()`` round-trip, and a
reproducible ``python_code`` string.

Examples
--------
>>> import pyrsm as rsm
>>> model = {
...     "meta": {"problem_name": "Production Mix", "objective_sense": "maximize"},
...     "variables": {
...         "product_a": {"label": "Units of A", "type": "continuous", "min": 0},
...         "product_b": {"label": "Units of B", "type": "continuous", "min": 0},
...     },
...     "objective": {"product_a": 40.0, "product_b": 30.0},
...     "constraints": [
...         {"name": "labor", "coefficients": {"product_a": 2, "product_b": 1}, "sense": "L", "rhs": 400},
...         {"name": "material", "coefficients": {"product_a": 1, "product_b": 2}, "sense": "L", "rhs": 500},
...     ],
... }
>>> opt = rsm.decide.optimize(model)
>>> opt.is_solved
True
>>> round(opt.objective_value, 2)
10000.0
>>> opt.solution["product_a"], opt.solution["product_b"]
(100.0, 200.0)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from io import StringIO
from typing import Any, Literal

import polars as pl

import pyrsm.basics.display_utils as du

Sense = Literal["L", "E", "G"]
VarType = Literal["continuous", "integer", "binary"]
Severity = Literal["error", "warning"]

# Tolerance used to decide whether a constraint is binding (|slack| < TOL).
_BINDING_TOL = 1e-6


@lru_cache(maxsize=1)
def _get_pulp():
    """Lazily import PuLP (keeps ``import pyrsm`` fast)."""
    import pulp

    return pulp


# ---------------------------------------------------------------------------
# Input/output normalization helpers
# ---------------------------------------------------------------------------

_SENSE_ALIASES: dict[str, Sense] = {
    "l": "L", "<=": "L", "<": "L", "le": "L", "leq": "L", "max<=": "L",
    "e": "E", "=": "E", "==": "E", "eq": "E",
    "g": "G", ">=": "G", ">": "G", "ge": "G", "geq": "G",
}

_VTYPE_ALIASES: dict[str, VarType] = {
    "continuous": "continuous", "cont": "continuous", "c": "continuous", "real": "continuous",
    "integer": "integer", "int": "integer", "i": "integer",
    "binary": "binary", "bin": "binary", "b": "binary", "bool": "binary",
}

_SENSE_SYMBOL = {"L": "<=", "E": "=", "G": ">="}


def _normalize_sense(value: Any) -> Sense | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    return _SENSE_ALIASES.get(key)


def _normalize_vtype(value: Any) -> VarType | None:
    if value is None:
        return "continuous"
    key = str(value).strip().lower()
    return _VTYPE_ALIASES.get(key)


def _normalize_objective_sense(value: Any) -> Literal["maximize", "minimize"] | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if key in {"maximize", "max", "maximise"}:
        return "maximize"
    if key in {"minimize", "min", "minimise"}:
        return "minimize"
    return None


def _coerce_float(value: Any) -> float | None:
    """Convert to float, returning ``None`` for unparseable / null input."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Structured result objects
# ---------------------------------------------------------------------------


@dataclass
class Diagnostic:
    """Structured validation/solver message (mirrors ``dtree.Diagnostic``)."""

    severity: Severity
    code: str
    message: str
    path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": list(self.path),
        }


@dataclass
class DecisionVariable:
    """A decision variable and (after solving) its optimal value.

    Attributes
    ----------
    name : str
        Identifier used in the objective and constraint coefficient maps.
    label : str
        Human-readable label for display.
    vtype : {"continuous", "integer", "binary"}
        Variable domain.
    lower, upper : float | None
        Bounds (``None`` means unbounded in that direction).
    value : float | None
        Optimal value after solving (``None`` if not solved).
    reduced_cost : float | None
        Reduced cost (only defined for pure LPs).
    """

    name: str
    label: str = ""
    vtype: VarType = "continuous"
    lower: float | None = 0.0
    upper: float | None = None
    value: float | None = None
    reduced_cost: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "type": self.vtype,
            "lower": self.lower,
            "upper": self.upper,
            "value": self.value,
            "reduced_cost": self.reduced_cost,
        }


@dataclass
class Constraint:
    """A linear constraint and (after solving) its slack/shadow price.

    Attributes
    ----------
    name : str
        Identifier.
    coefficients : dict[str, float]
        Map of variable name -> coefficient on the left-hand side.
    sense : {"L", "E", "G"}
        ``L`` is ``<=``, ``E`` is ``=``, ``G`` is ``>=``.
    rhs : float
        Right-hand side value.
    label : str
        Human-readable label for display.
    lhs : float | None
        Left-hand-side value at the optimum.
    slack : float | None
        Unused resource: ``rhs - lhs`` for ``L``, ``lhs - rhs`` for ``G``,
        ``0`` for ``E``. Non-negative for a feasible solution.
    shadow_price : float | None
        Dual value (marginal change in the objective per unit increase in
        ``rhs``). Only defined for pure LPs.
    binding : bool | None
        True when the constraint is satisfied with (near-)zero slack.
    """

    name: str
    coefficients: dict[str, float]
    sense: Sense
    rhs: float
    label: str = ""
    lhs: float | None = None
    slack: float | None = None
    shadow_price: float | None = None
    binding: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "coefficients": dict(self.coefficients),
            "sense": self.sense,
            "rhs": self.rhs,
            "lhs": self.lhs,
            "slack": self.slack,
            "shadow_price": self.shadow_price,
            "binding": self.binding,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class optimize:
    """Linear / (mixed-)integer programming with sensitivity analysis.

    Parameters
    ----------
    model : dict or str
        Problem definition. Either a ``dict`` matching the studio contract
        (keys ``meta``, ``variables``, ``objective``, ``constraints``) or a
        JSON string encoding the same structure.
    name : str, optional
        Override the problem name from ``meta.problem_name``.
    sense : {"maximize", "minimize"}, optional
        Override ``meta.objective_sense``.

    Attributes
    ----------
    name : str
        Problem name.
    sense : {"maximize", "minimize"}
        Optimization direction in effect.
    variables : dict[str, DecisionVariable]
        Declared variables (insertion order preserved).
    objective : dict[str, float]
        Objective coefficient per variable.
    constraints : list[Constraint]
        Declared constraints with solved slack/shadow prices.
    diagnostics : list[Diagnostic]
        Structured validation/solver messages.
    status : str
        Solver status ("Optimal", "Infeasible", "Unbounded", "Not Solved", ...).
    objective_value : float | None
        Optimal objective value (``None`` if not solved).
    solution : dict[str, float]
        Optimal value per variable (empty if not solved).
    solution_df : pl.DataFrame
        Per-variable solution table.
    constraints_df : pl.DataFrame
        Per-constraint slack/shadow-price table.
    python_code : str
        Reproducible Python snippet.

    Methods
    -------
    summary(dec=3)
        Print a Radiant-style summary of the model and solution.
    sensitivity(dec=3)
        Return the constraint sensitivity table (slack + shadow prices).
    robustness(trials=1000, ...)
        Monte Carlo robustness analysis on the objective coefficients.
    plot(...)
        Plot the feasible region (2 continuous variables only).
    to_dict()
        Serialize the model + solution to a JSON-ready dict.
    """

    def __init__(
        self,
        model: dict[str, Any] | str,
        *,
        name: str | None = None,
        sense: Literal["maximize", "minimize"] | None = None,
    ) -> None:
        if isinstance(model, str):
            try:
                model = json.loads(model)
            except json.JSONDecodeError as err:
                raise ValueError(f"model is not valid JSON: {err}") from None
        if not isinstance(model, dict):
            raise TypeError("model must be a dict or a JSON string")

        self._raw_model: dict[str, Any] = model
        self.diagnostics: list[Diagnostic] = []

        meta = model.get("meta") or {}
        self.name: str = name or str(meta.get("problem_name") or "optimization")

        resolved_sense = sense or _normalize_objective_sense(meta.get("objective_sense"))
        if resolved_sense is None:
            if meta.get("objective_sense") is not None:
                self._error(
                    "bad_objective_sense",
                    f"objective_sense {meta.get('objective_sense')!r} is not "
                    "'maximize' or 'minimize'; defaulting to 'maximize'",
                )
            resolved_sense = "maximize"
        self.sense: Literal["maximize", "minimize"] = resolved_sense

        self.variables: dict[str, DecisionVariable] = self._parse_variables(
            model.get("variables")
        )
        self.objective: dict[str, float] = self._parse_objective(model.get("objective"))
        self.constraints: list[Constraint] = self._parse_constraints(
            model.get("constraints")
        )

        # Solver outputs (populated by _solve)
        self.status: str = "Not Solved"
        self.objective_value: float | None = None
        self.solution: dict[str, float] = {}

        if not self.errors:
            self._solve()

        self.solution_df: pl.DataFrame = self._build_solution_df()
        self.constraints_df: pl.DataFrame = self._build_constraints_df()
        self.python_code: str = self._build_python_code()

    # ------------------------------------------------------------------
    # Parsing & validation
    # ------------------------------------------------------------------

    def _error(self, code: str, message: str, path: list[str] | None = None) -> None:
        self.diagnostics.append(Diagnostic("error", code, message, path or []))

    def _warn(self, code: str, message: str, path: list[str] | None = None) -> None:
        self.diagnostics.append(Diagnostic("warning", code, message, path or []))

    def _parse_variables(self, raw: Any) -> dict[str, DecisionVariable]:
        out: dict[str, DecisionVariable] = {}
        if not isinstance(raw, dict) or not raw:
            self._error("no_variables", "model must declare at least one variable")
            return out

        for vname, spec in raw.items():
            spec = spec or {}
            vtype = _normalize_vtype(spec.get("type"))
            if vtype is None:
                self._warn(
                    "bad_var_type",
                    f"unknown variable type {spec.get('type')!r}; using 'continuous'",
                    [vname],
                )
                vtype = "continuous"

            lower = _coerce_float(spec.get("min", spec.get("lower", 0.0)))
            upper = _coerce_float(spec.get("max", spec.get("upper")))

            if vtype == "binary":
                # Binary variables are pinned to {0, 1} regardless of bounds.
                lower, upper = 0.0, 1.0

            if lower is not None and upper is not None and lower > upper:
                self._error(
                    "bad_bounds",
                    f"lower bound {lower} exceeds upper bound {upper}",
                    [vname],
                )

            out[str(vname)] = DecisionVariable(
                name=str(vname),
                label=str(spec.get("label") or vname),
                vtype=vtype,
                lower=lower,
                upper=upper,
            )
        return out

    def _parse_objective(self, raw: Any) -> dict[str, float]:
        out: dict[str, float] = {}
        if not isinstance(raw, dict) or not raw:
            self._error("no_objective", "model must declare an objective function")
            return out

        for vname, coef in raw.items():
            vname = str(vname)
            fcoef = _coerce_float(coef)
            if fcoef is None:
                self._error(
                    "bad_objective_coef",
                    f"objective coefficient {coef!r} is not numeric",
                    [vname],
                )
                continue
            if vname not in self.variables:
                self._error(
                    "unknown_objective_var",
                    f"objective references undeclared variable {vname!r}",
                    [vname],
                )
                continue
            out[vname] = fcoef

        # Variables absent from the objective contribute 0 (note it once).
        missing = [v for v in self.variables if v not in out]
        if missing and not self.errors:
            self._warn(
                "zero_objective_coef",
                f"variables with no objective coefficient (treated as 0): "
                f"{', '.join(missing)}",
            )
        return out

    def _parse_constraints(self, raw: Any) -> list[Constraint]:
        out: list[Constraint] = []
        if raw is None:
            return out
        if not isinstance(raw, list):
            self._error("bad_constraints", "constraints must be a list")
            return out

        for i, spec in enumerate(raw):
            spec = spec or {}
            cname = str(spec.get("name") or f"constraint_{i + 1}")

            sense = _normalize_sense(spec.get("sense"))
            if sense is None:
                self._error(
                    "bad_sense",
                    f"constraint sense {spec.get('sense')!r} is not one of "
                    "L (<=), E (=), G (>=)",
                    [cname],
                )
                sense = "L"

            rhs = _coerce_float(spec.get("rhs"))
            if rhs is None:
                self._error(
                    "bad_rhs",
                    f"constraint rhs {spec.get('rhs')!r} is not numeric",
                    [cname],
                )
                rhs = 0.0

            raw_coefs = spec.get("coefficients") or {}
            coefs: dict[str, float] = {}
            if not isinstance(raw_coefs, dict) or not raw_coefs:
                self._error(
                    "empty_constraint",
                    "constraint has no coefficients",
                    [cname],
                )
            else:
                for vname, coef in raw_coefs.items():
                    vname = str(vname)
                    fcoef = _coerce_float(coef)
                    if fcoef is None:
                        self._error(
                            "bad_constraint_coef",
                            f"coefficient {coef!r} for {vname!r} is not numeric",
                            [cname],
                        )
                        continue
                    if vname not in self.variables:
                        self._error(
                            "unknown_constraint_var",
                            f"constraint references undeclared variable {vname!r}",
                            [cname],
                        )
                        continue
                    coefs[vname] = fcoef

            out.append(
                Constraint(
                    name=cname,
                    coefficients=coefs,
                    sense=sense,
                    rhs=rhs,
                    label=str(spec.get("label") or cname),
                )
            )
        return out

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    def _solve(self) -> None:
        pulp = _get_pulp()

        is_mip = any(v.vtype != "continuous" for v in self.variables.values())

        prob = pulp.LpProblem(
            self._safe_name(self.name),
            pulp.LpMaximize if self.sense == "maximize" else pulp.LpMinimize,
        )

        cat = {
            "continuous": pulp.LpContinuous,
            "integer": pulp.LpInteger,
            "binary": pulp.LpBinary,
        }
        lp_vars: dict[str, Any] = {}
        for name, var in self.variables.items():
            lp_vars[name] = pulp.LpVariable(
                self._safe_name(name),
                lowBound=var.lower,
                upBound=var.upper,
                cat=cat[var.vtype],
            )

        # Objective
        prob += pulp.lpSum(
            self.objective.get(name, 0.0) * lp_vars[name] for name in self.variables
        )

        # Constraints
        lp_cons: dict[str, Any] = {}
        sense_map = {
            "L": pulp.LpConstraintLE,
            "E": pulp.LpConstraintEQ,
            "G": pulp.LpConstraintGE,
        }
        for con in self.constraints:
            expr = pulp.lpSum(
                coef * lp_vars[vname] for vname, coef in con.coefficients.items()
            )
            lp_con = pulp.LpConstraint(
                expr, sense=sense_map[con.sense], rhs=con.rhs, name=self._safe_name(con.name)
            )
            prob += lp_con
            lp_cons[con.name] = lp_con

        try:
            code = prob.solve(pulp.PULP_CBC_CMD(msg=0))
        except Exception as err:  # pragma: no cover - solver/runtime failure
            self._error("solver_error", f"solver failed: {err}")
            self.status = "Solver Error"
            return

        self.status = pulp.LpStatus.get(code, "Unknown")

        if self.status != "Optimal":
            self._warn(
                "not_optimal",
                f"solver returned status {self.status!r}; no optimal solution stored",
            )
            return

        self.objective_value = _coerce_float(pulp.value(prob.objective))

        for name, var in self.variables.items():
            val = _coerce_float(lp_vars[name].varValue) or 0.0
            var.value = val
            self.solution[name] = val
            # Reduced costs are only meaningful for pure LPs.
            var.reduced_cost = None if is_mip else getattr(lp_vars[name], "dj", None)

        for con in self.constraints:
            lhs = sum(
                coef * self.solution.get(vname, 0.0)
                for vname, coef in con.coefficients.items()
            )
            con.lhs = lhs
            if con.sense == "L":
                con.slack = con.rhs - lhs
            elif con.sense == "G":
                con.slack = lhs - con.rhs
            else:
                con.slack = 0.0
            con.binding = abs(con.slack) < _BINDING_TOL
            # Shadow prices (duals) are only meaningful for pure LPs.
            con.shadow_price = None if is_mip else getattr(lp_cons[con.name], "pi", None)

        if is_mip:
            self._warn(
                "duals_unavailable",
                "model has integer/binary variables; shadow prices and reduced "
                "costs are not reported (duals are undefined for integer programs)",
            )

    @staticmethod
    def _safe_name(name: str) -> str:
        """PuLP rejects spaces and a few symbols in names; sanitize them."""
        out = []
        for ch in str(name):
            out.append(ch if (ch.isalnum() or ch in "_") else "_")
        cleaned = "".join(out).strip("_")
        return cleaned or "x"

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "warning"]

    @property
    def is_solved(self) -> bool:
        return self.status == "Optimal" and not self.errors

    def _build_solution_df(self) -> pl.DataFrame:
        schema = {
            "variable": pl.Utf8,
            "label": pl.Utf8,
            "type": pl.Utf8,
            "value": pl.Float64,
            "lower": pl.Float64,
            "upper": pl.Float64,
            "reduced_cost": pl.Float64,
        }
        if not self.variables:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(
            [
                {
                    "variable": v.name,
                    "label": v.label,
                    "type": v.vtype,
                    "value": v.value,
                    "lower": v.lower,
                    "upper": v.upper,
                    "reduced_cost": v.reduced_cost,
                }
                for v in self.variables.values()
            ],
            schema=schema,
        )

    def _build_constraints_df(self) -> pl.DataFrame:
        schema = {
            "constraint": pl.Utf8,
            "label": pl.Utf8,
            "lhs": pl.Float64,
            "sense": pl.Utf8,
            "rhs": pl.Float64,
            "slack": pl.Float64,
            "shadow_price": pl.Float64,
            "binding": pl.Boolean,
        }
        if not self.constraints:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(
            [
                {
                    "constraint": c.name,
                    "label": c.label,
                    "lhs": c.lhs,
                    "sense": _SENSE_SYMBOL[c.sense],
                    "rhs": c.rhs,
                    "slack": c.slack,
                    "shadow_price": c.shadow_price,
                    "binding": c.binding,
                }
                for c in self.constraints
            ],
            schema=schema,
        )

    def sensitivity(self, dec: int = 3) -> pl.DataFrame:
        """Return the constraint sensitivity table (slack + shadow prices).

        Parameters
        ----------
        dec : int
            Number of decimal places to round numeric columns to.
        """
        df = self.constraints_df
        if df.height == 0:
            return df
        num_cols = ["lhs", "rhs", "slack", "shadow_price"]
        return df.with_columns(
            [pl.col(c).round(dec) for c in num_cols if c in df.columns]
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _diagnostics_text(self, buf: StringIO) -> None:
        if self.errors:
            buf.write("Errors:\n")
            for d in self.errors:
                where = " > ".join(d.path) if d.path else "(model)"
                buf.write(f"  [{d.code}] {where}: {d.message}\n")
            buf.write("\n")
        if self.warnings:
            buf.write("Warnings:\n")
            for d in self.warnings:
                where = " > ".join(d.path) if d.path else "(model)"
                buf.write(f"  [{d.code}] {where}: {d.message}\n")
            buf.write("\n")

    def summary(self, dec: int = 3) -> None:
        """Print a Radiant-style summary of the model and its solution.

        Parameters
        ----------
        dec : int
            Number of decimal places to display.
        """
        buf = StringIO()
        buf.write("Linear/Integer programming optimization\n")
        buf.write(f"Problem    : {self.name}\n")
        buf.write(f"Direction  : {self.sense}\n")
        buf.write(f"Variables  : {len(self.variables)}\n")
        buf.write(f"Constraints: {len(self.constraints)}\n")
        buf.write(f"Status     : {self.status}\n")
        if self.objective_value is not None:
            buf.write(f"Objective  : {self.objective_value:,.{dec}f}\n")
        buf.write("\n")
        self._diagnostics_text(buf)
        print(buf.getvalue().rstrip())

        if not self.is_solved:
            return

        print("\nDecision variables:")
        du.print_plain_tables(
            self.solution_df.with_columns(
                [
                    pl.col(c).round(dec)
                    for c in ("value", "lower", "upper", "reduced_cost")
                    if c in self.solution_df.columns
                ]
            )
        )

        if self.constraints:
            print("\nConstraints (slack & shadow prices):")
            du.print_plain_tables(self.sensitivity(dec=dec))

    # ------------------------------------------------------------------
    # Robustness (Monte Carlo simulation bridge)
    # ------------------------------------------------------------------

    def robustness(
        self,
        trials: int = 1000,
        coef_noise: float = 0.1,
        rhs_noise: float = 0.0,
        noise: Literal["normal", "uniform"] = "normal",
        seed: int | None = 1234,
    ) -> RobustnessResult:
        """Monte Carlo robustness analysis.

        Re-solves the model ``trials`` times while perturbing the objective
        coefficients (and optionally the constraint right-hand sides) with
        multiplicative noise, to gauge how sensitive the optimum is to
        uncertain inputs.

        Parameters
        ----------
        trials : int
            Number of Monte Carlo re-solves.
        coef_noise : float
            Relative noise on objective coefficients. For ``noise="normal"``
            it is the standard deviation of a multiplier ``~ N(1, coef_noise)``;
            for ``noise="uniform"`` the multiplier is ``~ U(1 - coef_noise,
            1 + coef_noise)``. Set to 0 to leave coefficients unchanged.
        rhs_noise : float
            Same idea, applied to constraint right-hand sides.
        noise : {"normal", "uniform"}
            Distribution of the multiplicative noise.
        seed : int | None
            Seed for reproducibility.

        Returns
        -------
        RobustnessResult
            Per-trial outcomes plus summary statistics.
        """
        if not self.is_solved:
            raise ValueError(
                "model is not solved; cannot run robustness analysis "
                f"(status: {self.status})"
            )

        import numpy as np

        rng = np.random.default_rng(seed)
        var_names = list(self.variables)

        def _draw(size: int, spread: float) -> Any:
            if noise == "uniform":
                return rng.uniform(1 - spread, 1 + spread, size=size)
            return rng.normal(1.0, spread, size=size)

        rows: list[dict[str, Any]] = []
        base_obj = {name: self.objective.get(name, 0.0) for name in var_names}
        for t in range(trials):
            model = json.loads(json.dumps(self._raw_model))  # deep copy

            if coef_noise > 0:
                mult = _draw(len(var_names), coef_noise)
                model["objective"] = {
                    name: base_obj[name] * float(m)
                    for name, m in zip(var_names, mult)
                }
            if rhs_noise > 0 and self.constraints:
                mult = _draw(len(self.constraints), rhs_noise)
                for c_spec, m in zip(model.get("constraints", []), mult):
                    if c_spec.get("rhs") is not None:
                        c_spec["rhs"] = float(c_spec["rhs"]) * float(m)

            trial_opt = optimize(model, name=self.name, sense=self.sense)
            row: dict[str, Any] = {
                "trial": t,
                "status": trial_opt.status,
                "objective": trial_opt.objective_value,
            }
            for name in var_names:
                row[name] = trial_opt.solution.get(name)
            rows.append(row)

        df = pl.DataFrame(rows)
        return RobustnessResult(
            trials=trials,
            var_names=var_names,
            df=df,
            base_objective=self.objective_value,
        )

    # ------------------------------------------------------------------
    # Plotting (2 continuous variables)
    # ------------------------------------------------------------------

    def plot(self, resolution: int = 300):
        """Plot the feasible region and optimum for a 2-variable LP.

        Only supported for models with exactly two continuous decision
        variables. Returns a plotnine ``ggplot`` object.

        Parameters
        ----------
        resolution : int
            Grid resolution per axis used to shade the feasible region.
        """
        cont_vars = [v for v in self.variables.values() if v.vtype == "continuous"]
        if len(self.variables) != 2 or len(cont_vars) != 2:
            raise ValueError(
                "plot() supports exactly two continuous variables; "
                f"this model has {len(self.variables)} variable(s)"
            )

        import numpy as np
        import plotnine as p9

        xv, yv = list(self.variables.values())

        # Determine a sensible plotting window from bounds + constraint intercepts.
        def _axis_max(var: DecisionVariable, axis: int) -> float:
            if var.upper is not None:
                return float(var.upper)
            cand = [10.0]
            for c in self.constraints:
                coef = c.coefficients.get(var.name)
                if coef and coef > 0 and c.rhs > 0:
                    cand.append(c.rhs / coef)
            return max(cand) * 1.1

        x_lo = float(xv.lower or 0.0)
        y_lo = float(yv.lower or 0.0)
        x_hi = _axis_max(xv, 0)
        y_hi = _axis_max(yv, 1)

        xs = np.linspace(x_lo, x_hi, resolution)
        ys = np.linspace(y_lo, y_hi, resolution)
        gx, gy = np.meshgrid(xs, ys)
        feasible = np.ones_like(gx, dtype=bool)
        for c in self.constraints:
            a = c.coefficients.get(xv.name, 0.0)
            b = c.coefficients.get(yv.name, 0.0)
            lhs = a * gx + b * gy
            tol = 1e-9 * max(1.0, abs(c.rhs))
            if c.sense == "L":
                feasible &= lhs <= c.rhs + tol
            elif c.sense == "G":
                feasible &= lhs >= c.rhs - tol
            else:
                feasible &= np.abs(lhs - c.rhs) <= 1e-3 * max(1.0, abs(c.rhs))

        region = pl.DataFrame(
            {"x": gx[feasible], "y": gy[feasible]}
        ).to_pandas()

        plot = (
            p9.ggplot()
            + p9.geom_raster(
                region, p9.aes(x="x", y="y"), fill="#4c78a8", alpha=0.25
            )
            + p9.labs(
                x=xv.label or xv.name,
                y=yv.label or yv.name,
                title=f"Feasible region — {self.name}",
            )
        )

        # Constraint boundary lines.
        for c in self.constraints:
            a = c.coefficients.get(xv.name, 0.0)
            b = c.coefficients.get(yv.name, 0.0)
            if b != 0:
                line = pl.DataFrame(
                    {"x": xs, "y": (c.rhs - a * xs) / b}
                ).to_pandas()
                plot = plot + p9.geom_line(
                    line, p9.aes(x="x", y="y"), color="#555555", size=0.4
                )
            elif a != 0:
                plot = plot + p9.geom_vline(
                    xintercept=c.rhs / a, color="#555555", size=0.4
                )

        # Optimum point.
        if self.is_solved:
            opt_df = pl.DataFrame(
                {"x": [self.solution[xv.name]], "y": [self.solution[yv.name]]}
            ).to_pandas()
            plot = plot + p9.geom_point(
                opt_df, p9.aes(x="x", y="y"), color="#e45756", size=3
            )

        return (
            plot
            + p9.coord_cartesian(xlim=(x_lo, x_hi), ylim=(y_lo, y_hi))
            + p9.theme_minimal()
        )

    # ------------------------------------------------------------------
    # Serialization & reproducibility
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full model and solution to a JSON-ready dict."""
        return {
            "meta": {
                "problem_name": self.name,
                "objective_sense": self.sense,
            },
            "variables": {n: v.to_dict() for n, v in self.variables.items()},
            "objective": dict(self.objective),
            "constraints": [c.to_dict() for c in self.constraints],
            "solution": {
                "status": self.status,
                "objective_value": self.objective_value,
                "values": dict(self.solution),
                "is_solved": self.is_solved,
            },
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }

    def _build_python_code(self) -> str:
        import pprint

        # pformat (not json.dumps) so the snippet is valid Python:
        # None/True/False instead of null/true/false. sort_dicts=False
        # preserves the declared variable/constraint order.
        model_repr = pprint.pformat(self._raw_model, indent=4, sort_dicts=False, width=88)
        lines = [
            "import pyrsm as rsm",
            "",
            f"model = {model_repr}",
            "",
            "opt = rsm.decide.optimize(model)",
            "opt.summary()",
        ]
        return "\n".join(lines)


@dataclass
class RobustnessResult:
    """Outcome of :meth:`optimize.robustness`.

    Attributes
    ----------
    trials : int
        Number of Monte Carlo re-solves attempted.
    var_names : list[str]
        Decision variable names (column order in ``df``).
    df : pl.DataFrame
        Per-trial results: ``trial``, ``status``, ``objective`` and one column
        per decision variable.
    base_objective : float | None
        Objective value of the unperturbed (deterministic) optimum.
    """

    trials: int
    var_names: list[str]
    df: pl.DataFrame
    base_objective: float | None = None

    @property
    def solved(self) -> pl.DataFrame:
        """Trials that solved to optimality."""
        return self.df.filter(pl.col("status") == "Optimal")

    def stats(self, dec: int = 3) -> pl.DataFrame:
        """Summary statistics for the objective across solved trials."""
        solved = self.solved
        obj = solved["objective"].drop_nulls()
        if obj.len() == 0:
            return pl.DataFrame(schema={"metric": pl.Utf8, "value": pl.Float64})
        rows = {
            "trials": float(self.trials),
            "solved": float(solved.height),
            "mean": float(obj.mean()),
            "std": float(obj.std()) if obj.len() > 1 else 0.0,
            "min": float(obj.min()),
            "p05": float(obj.quantile(0.05)),
            "median": float(obj.median()),
            "p95": float(obj.quantile(0.95)),
            "max": float(obj.max()),
        }
        return pl.DataFrame(
            {"metric": list(rows), "value": [round(v, dec) for v in rows.values()]}
        )

    def summary(self, dec: int = 3) -> None:
        """Print summary statistics for the robustness analysis."""
        print("Robustness analysis (Monte Carlo)")
        if self.base_objective is not None:
            print(f"Deterministic objective: {self.base_objective:,.{dec}f}")
        n_failed = self.trials - self.solved.height
        if n_failed:
            print(f"Infeasible/unsolved trials: {n_failed} of {self.trials}")
        print()
        du.print_plain_tables(self.stats(dec=dec))

    def plot(self, bins: int = 30):
        """Histogram of the objective value across solved trials (ggplot)."""
        import plotnine as p9

        data = self.solved.select("objective").drop_nulls().to_pandas()
        plot = (
            p9.ggplot(data, p9.aes(x="objective"))
            + p9.geom_histogram(bins=bins, fill="#4c78a8", color="white")
            + p9.labs(x="Objective value", y="Count", title="Robustness of the optimum")
            + p9.theme_minimal()
        )
        if self.base_objective is not None:
            plot = plot + p9.geom_vline(
                xintercept=self.base_objective, color="#e45756", size=0.8
            )
        return plot


def _make_module_callable() -> None:
    """Keep ``pyrsm.decide.optimize(...)`` callable after submodule imports.

    Mirrors the pattern in ``dtree``/``simulate``: when the submodule itself is
    imported directly (e.g. ``from pyrsm.decide.optimize import ...``) the import
    machinery plants the *module* as ``pyrsm.decide.optimize``, shadowing the
    class. Making the module callable keeps ``rsm.decide.optimize(model)``
    working regardless of import order.
    """
    import sys
    import types

    class _CallableModule(types.ModuleType):
        def __call__(self, *args, **kwargs):
            return optimize(*args, **kwargs)

    sys.modules[__name__].__class__ = _CallableModule


_make_module_callable()
