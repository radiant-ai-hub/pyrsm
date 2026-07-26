"""Decision-tree analysis with YAML input, validation, and EV rollback.

Radiant-compatible decision tree engine. Builds typed node objects from
YAML, validates structure, resolves variables via a safe AST evaluator,
performs post-order expected-value rollback, and emits a solution table,
Mermaid graph, and reproducible Python code.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Literal

import pandas as pd
import yaml

NodeKind = Literal["decision", "chance", "terminal"]
Severity = Literal["error", "warning"]

_RESERVED_KEYS = {"name", "variables", "type", "p", "payoff", "cost"}


@dataclass
class Diagnostic:
    """Structured validation/solver diagnostic with a node path."""

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
class DecisionNode:
    """Typed node in a decision tree."""

    id: str
    label: str
    kind: NodeKind
    level: int = 1
    p: float | None = None
    cost: float | None = None
    payoff: float | None = None
    raw_p: str | None = None
    raw_cost: str | None = None
    raw_payoff: str | None = None
    children: list[DecisionNode] = field(default_factory=list)
    chosen_child_ids: list[str] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return not self.children

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "level": self.level,
            "p": self.p,
            "cost": self.cost,
            "payoff": self.payoff,
            "chosen_child_ids": list(self.chosen_child_ids),
            "children": [c.to_dict() for c in self.children],
        }


# ---------------------------------------------------------------------------
# Safe arithmetic evaluator
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class UnsafeExpressionError(ValueError):
    """Raised when an expression contains disallowed syntax."""


class UnresolvedVariableError(ValueError):
    """Raised when an expression references an unknown variable."""

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name


class UnknownSubtreeError(ValueError):
    """Raised when ``subtree("name")`` refers to a name not in the library."""

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name


def safe_eval(
    expr: str,
    variables: dict[str, float],
    *,
    subtree_resolver: Callable[[str], float] | None = None,
) -> float:
    """Evaluate ``expr`` using only safe arithmetic and known variables.

    Supports numeric literals, identifiers from ``variables``, parentheses,
    unary +/-, and binary +, -, *, /, **. Also allows the explicitly
    allowlisted ``subtree("name")`` call when a ``subtree_resolver`` is
    supplied; the resolver receives the literal string argument and must
    return a numeric payoff. Anything else (other function calls,
    attribute access, comparisons, etc.) raises ``UnsafeExpressionError``.
    """
    if not isinstance(expr, str):
        return float(expr)

    text = expr.strip()
    if not text:
        raise UnsafeExpressionError("empty expression")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"could not parse expression: {expr!r} ({exc.msg})")

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int | float) and not isinstance(node.value, bool):
                return float(node.value)
            raise UnsafeExpressionError(f"non-numeric constant {node.value!r}")
        if isinstance(node, ast.Name):
            name = node.id
            if name not in variables:
                raise UnresolvedVariableError(name)
            return float(variables[name])
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.Call):
            # Only ``subtree("name")`` is allowed, and only when a
            # resolver was provided. The keyword-args / star-args /
            # double-star-args paths stay rejected so we don't open a
            # back-door for arbitrary kwargs.
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "subtree"
                and subtree_resolver is not None
                and len(node.args) == 1
                and not node.keywords
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                return float(subtree_resolver(node.args[0].value))
        raise UnsafeExpressionError(f"unsupported expression element: {ast.dump(node)}")

    result = _eval(tree)
    return float(result)


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_identifier(name: str) -> bool:
    return bool(_IDENTIFIER_RE.match(name))


def _safe_var_name(name: str, index: int) -> str:
    """Map a free-form variable label to a Python identifier."""
    if _is_identifier(name):
        return name
    candidate = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    if not candidate or not _is_identifier(candidate):
        candidate = f"var_{index}"
    return candidate


def _resolve_variables(
    raw_vars: dict[str, Any] | None,
    *,
    subtree_resolver: Callable[[str], float] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, str], list[Diagnostic]]:
    """Resolve the ``variables:`` block.

    Returns ``(resolved, safe_vars, label_to_safe, diagnostics)``:

    - ``resolved``: original label -> numeric value (for display);
    - ``safe_vars``: safe identifier -> numeric value (for ``safe_eval``);
    - ``label_to_safe``: original label -> safe identifier (for the
      substitution pass on p/cost/payoff expressions);
    - ``diagnostics``: list of diagnostics for anything that failed.

    ``subtree_resolver`` is forwarded to ``safe_eval`` so a variable
    expression may include ``subtree("name")`` calls.
    """
    diagnostics: list[Diagnostic] = []
    resolved: dict[str, float] = {}
    label_to_safe: dict[str, str] = {}

    safe_vars: dict[str, float] = {}

    if not raw_vars:
        return resolved, safe_vars, label_to_safe, diagnostics

    if not isinstance(raw_vars, dict):
        diagnostics.append(
            Diagnostic(
                severity="error",
                code="variables_not_mapping",
                message="`variables:` must be a mapping of name -> value.",
                path=["variables"],
            )
        )
        return resolved, safe_vars, label_to_safe, diagnostics

    # First pass: assign safe identifiers (original label order matters
    # so later variables can reference earlier ones). Distinct labels can
    # collapse to the same safe name (e.g. "P(+|S)" and "P(S|+)" both
    # reduce to "P_S"); disambiguate with a trailing index so each label
    # keeps its own slot in the eval namespace.
    used: set[str] = set()
    for idx, label in enumerate(raw_vars.keys()):
        base = _safe_var_name(str(label), idx)
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        label_to_safe[str(label)] = candidate

    for label, expr in raw_vars.items():
        safe_name = label_to_safe[str(label)]
        try:
            substituted = _substitute_labels(expr, label_to_safe)
            value = safe_eval(substituted, safe_vars, subtree_resolver=subtree_resolver)
        except UnknownSubtreeError as exc:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="unknown_subtree",
                    message=(
                        f"Variable '{label}' references unknown subtree '{exc.name}'. "
                        "Add it to the project, or fix the name."
                    ),
                    path=["variables", str(label)],
                )
            )
            continue
        except UnresolvedVariableError as exc:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="unresolved_variable",
                    message=(f"Variable '{label}' references unknown name '{exc.name}'."),
                    path=["variables", str(label)],
                )
            )
            continue
        except UnsafeExpressionError as exc:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="unsafe_expression",
                    message=(f"Variable '{label}' has an unsafe or invalid expression: {exc}"),
                    path=["variables", str(label)],
                )
            )
            continue
        except ZeroDivisionError:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="division_by_zero",
                    message=f"Variable '{label}' divides by zero.",
                    path=["variables", str(label)],
                )
            )
            continue
        if not math.isfinite(value):
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="non_finite_variable",
                    message=f"Variable '{label}' evaluates to {value!r}.",
                    path=["variables", str(label)],
                )
            )
            continue
        safe_vars[safe_name] = value
        resolved[str(label)] = value

    return resolved, safe_vars, label_to_safe, diagnostics


def _substitute_labels(expr: Any, label_to_safe: dict[str, str]) -> str:
    """Substitute free-form variable labels in ``expr`` with safe names.

    Numeric inputs are passed through. Replacement uses longest-label-first
    word-boundary matching so a label like 'p_small' is not partially
    matched inside 'p_small_total'.
    """
    if isinstance(expr, bool):
        # bools are ints in Python; we explicitly reject them so YAML's
        # `true`/`false` doesn't get treated as 1/0.
        raise UnsafeExpressionError("boolean values are not allowed")
    if isinstance(expr, int | float):
        return repr(float(expr))
    if expr is None:
        raise UnsafeExpressionError("missing value")
    text = str(expr)
    # Longest first to avoid partial overlaps.
    for label in sorted(label_to_safe.keys(), key=len, reverse=True):
        safe = label_to_safe[label]
        if label == safe:
            continue
        if _is_identifier(label):
            text = re.sub(rf"\b{re.escape(label)}\b", safe, text)
        else:
            text = text.replace(label, safe)
    return text


def _eval_numeric_field(
    raw: Any,
    safe_vars: dict[str, float],
    label_to_safe: dict[str, str],
    path: list[str],
    field_name: str,
    *,
    subtree_resolver: Callable[[str], float] | None = None,
) -> tuple[float | None, Diagnostic | None]:
    """Evaluate a numeric field on a node (p / cost / payoff).

    Returns ``(value, None)`` on success or ``(None, Diagnostic)`` on
    failure. ``None`` raw input returns ``(None, None)``.
    """
    if raw is None:
        return None, None
    if isinstance(raw, bool):
        return None, Diagnostic(
            severity="error",
            code=f"{field_name}_not_numeric",
            message=f"`{field_name}` for '{path[-1]}' is a boolean, not a number.",
            path=list(path),
        )
    if isinstance(raw, int | float):
        value = float(raw)
        if not math.isfinite(value):
            return None, Diagnostic(
                severity="error",
                code=f"{field_name}_not_finite",
                message=f"`{field_name}` is not finite at {'/'.join(path)}.",
                path=list(path),
            )
        return value, None
    try:
        substituted = _substitute_labels(raw, label_to_safe)
        value = safe_eval(substituted, safe_vars, subtree_resolver=subtree_resolver)
    except UnknownSubtreeError as exc:
        return None, Diagnostic(
            severity="error",
            code="unknown_subtree",
            message=(
                f"`{field_name}` at {'/'.join(path)} references unknown subtree '{exc.name}'."
            ),
            path=list(path),
        )
    except UnresolvedVariableError as exc:
        return None, Diagnostic(
            severity="error",
            code="unresolved_variable",
            message=(
                f"`{field_name}` at {'/'.join(path)} references unknown variable '{exc.name}'."
            ),
            path=list(path),
        )
    except UnsafeExpressionError as exc:
        return None, Diagnostic(
            severity="error",
            code="unsafe_expression",
            message=(
                f"`{field_name}` at {'/'.join(path)} has an unsafe or invalid expression: {exc}"
            ),
            path=list(path),
        )
    except ZeroDivisionError:
        return None, Diagnostic(
            severity="error",
            code="division_by_zero",
            message=f"`{field_name}` at {'/'.join(path)} divides by zero.",
            path=list(path),
        )
    if not math.isfinite(value):
        return None, Diagnostic(
            severity="error",
            code=f"{field_name}_not_finite",
            message=f"`{field_name}` at {'/'.join(path)} is not finite.",
            path=list(path),
        )
    return value, None


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


class _IdAllocator:
    def __init__(self) -> None:
        self._next = 0

    def next(self) -> str:
        self._next += 1
        return f"n{self._next}"


def _yaml_to_dict(yl: str | dict[str, Any]) -> tuple[dict[str, Any] | None, list[Diagnostic]]:
    diagnostics: list[Diagnostic] = []
    if isinstance(yl, dict):
        return yl, diagnostics
    if not isinstance(yl, str):
        diagnostics.append(
            Diagnostic(
                severity="error",
                code="invalid_input",
                message=(
                    f"Decision tree input must be a YAML string or a dict, got {type(yl).__name__}."
                ),
            )
        )
        return None, diagnostics
    try:
        data = yaml.safe_load(yl)
    except yaml.YAMLError as exc:
        diagnostics.append(
            Diagnostic(
                severity="error",
                code="yaml_parse_error",
                message=f"YAML parse error: {exc}",
            )
        )
        return None, diagnostics
    if data is None:
        diagnostics.append(
            Diagnostic(
                severity="error",
                code="empty_tree",
                message="Decision tree input is empty.",
            )
        )
        return None, diagnostics
    if not isinstance(data, dict):
        diagnostics.append(
            Diagnostic(
                severity="error",
                code="root_not_mapping",
                message="Decision tree root must be a YAML mapping.",
            )
        )
        return None, diagnostics
    return data, diagnostics


def _build_nodes(
    raw: dict[str, Any],
    safe_vars: dict[str, float],
    label_to_safe: dict[str, str],
    *,
    id_alloc: _IdAllocator,
    label: str,
    path: list[str],
    level: int,
    parent_kind: NodeKind | None = None,
    inherited_p: Any = None,
    inherited_cost: Any = None,
    diagnostics: list[Diagnostic] | None = None,
    subtree_resolver: Callable[[str], float] | None = None,
) -> DecisionNode | None:
    """Recursively convert a nested YAML mapping into ``DecisionNode``s."""
    if diagnostics is None:
        diagnostics = []

    if not isinstance(raw, dict):
        diagnostics.append(
            Diagnostic(
                severity="error",
                code="node_not_mapping",
                message=f"Node '{label}' is not a mapping.",
                path=list(path),
            )
        )
        return None

    declared_type = raw.get("type")
    p_raw = raw.get("p", inherited_p)
    cost_raw = raw.get("cost", inherited_cost)
    payoff_raw = raw.get("payoff")

    child_keys = [k for k in raw.keys() if isinstance(k, str) and k not in _RESERVED_KEYS]

    # Duplicate sibling labels?
    seen: set[str] = set()
    for k in child_keys:
        if k in seen:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="duplicate_sibling_label",
                    message=f"Duplicate child label '{k}' under '{label}'.",
                    path=list(path),
                )
            )
        seen.add(k)

    # Decide kind.
    if declared_type in ("decision", "chance"):
        kind: NodeKind = declared_type
    elif declared_type in (None, "", "terminal"):
        if child_keys:
            kind = "decision" if declared_type is None else declared_type  # type: ignore[assignment]
            if declared_type is None:
                # If there are children and no type, assume it's a decision
                # ONLY for the root; otherwise we have no idea. Best to
                # raise a diagnostic and pick decision as a default.
                kind = "decision"
                diagnostics.append(
                    Diagnostic(
                        severity="warning",
                        code="missing_node_type",
                        message=(
                            f"Node '{label}' has children but no `type:`; defaulting to 'decision'."
                        ),
                        path=list(path),
                    )
                )
            else:
                kind = "terminal"
                diagnostics.append(
                    Diagnostic(
                        severity="error",
                        code="terminal_has_children",
                        message=(f"Node '{label}' is declared terminal but has child branches."),
                        path=list(path),
                    )
                )
        else:
            kind = "terminal"
    else:
        diagnostics.append(
            Diagnostic(
                severity="error",
                code="invalid_node_type",
                message=(
                    f"Node '{label}' has unsupported `type: {declared_type}`. "
                    "Use 'decision' or 'chance'."
                ),
                path=list(path),
            )
        )
        kind = "terminal" if not child_keys else "decision"

    node = DecisionNode(
        id=id_alloc.next(),
        label=label,
        kind=kind,
        level=level,
    )

    # Evaluate p / cost / payoff on this node.
    if p_raw is not None:
        value, diag = _eval_numeric_field(
            p_raw,
            safe_vars,
            label_to_safe,
            path,
            "p",
            subtree_resolver=subtree_resolver,
        )
        node.p = value
        node.raw_p = None if isinstance(p_raw, int | float) else str(p_raw)
        if diag is not None:
            diagnostics.append(diag)
    if cost_raw is not None:
        value, diag = _eval_numeric_field(
            cost_raw,
            safe_vars,
            label_to_safe,
            path,
            "cost",
            subtree_resolver=subtree_resolver,
        )
        node.cost = value
        node.raw_cost = None if isinstance(cost_raw, int | float) else str(cost_raw)
        if diag is not None:
            diagnostics.append(diag)
    if payoff_raw is not None:
        value, diag = _eval_numeric_field(
            payoff_raw,
            safe_vars,
            label_to_safe,
            path,
            "payoff",
            subtree_resolver=subtree_resolver,
        )
        node.payoff = value
        node.raw_payoff = None if isinstance(payoff_raw, int | float) else str(payoff_raw)
        if diag is not None:
            diagnostics.append(diag)

    if kind == "terminal":
        if node.payoff is None:
            diagnostics.append(
                Diagnostic(
                    severity="error",
                    code="terminal_missing_payoff",
                    message=(f"Terminal node '{label}' is missing a `payoff:`."),
                    path=list(path),
                )
            )
        if cost_raw is not None:
            diagnostics.append(
                Diagnostic(
                    severity="warning",
                    code="terminal_has_cost",
                    message=(
                        f"Terminal node '{label}' uses `cost:`. Prefer "
                        "folding the cost into the `payoff:`."
                    ),
                    path=list(path),
                )
            )
        return node

    if not child_keys:
        diagnostics.append(
            Diagnostic(
                severity="error",
                code="non_terminal_without_children",
                message=(f"{kind.title()} node '{label}' has no child branches."),
                path=list(path),
            )
        )
        return node

    for ck in child_keys:
        child = _build_nodes(
            raw[ck] if isinstance(raw[ck], dict) else {"payoff": raw[ck]},
            safe_vars,
            label_to_safe,
            id_alloc=id_alloc,
            label=ck,
            path=path + [ck],
            level=level + 1,
            parent_kind=kind,
            diagnostics=diagnostics,
            subtree_resolver=subtree_resolver,
        )
        if child is not None:
            node.children.append(child)

    return node


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def _validate(root: DecisionNode | None, diagnostics: list[Diagnostic]) -> None:
    if root is None:
        return

    def _walk(node: DecisionNode, path: list[str]) -> None:
        if node.kind == "chance":
            for child in node.children:
                if child.p is None:
                    diagnostics.append(
                        Diagnostic(
                            severity="error",
                            code="chance_child_missing_probability",
                            message=(
                                f"Chance child '{child.label}' under "
                                f"'{node.label}' is missing `p:`."
                            ),
                            path=path + [child.label],
                        )
                    )
                elif not (0.0 <= child.p <= 1.0):
                    diagnostics.append(
                        Diagnostic(
                            severity="error",
                            code="chance_probability_out_of_range",
                            message=(
                                f"Probability {child.p} for '{child.label}' "
                                f"under '{node.label}' is outside [0, 1]."
                            ),
                            path=path + [child.label],
                        )
                    )
            if all(c.p is not None for c in node.children) and node.children:
                total = sum(c.p for c in node.children)  # type: ignore[misc]
                if not math.isclose(total, 1.0, abs_tol=1e-6):
                    diagnostics.append(
                        Diagnostic(
                            severity="error",
                            code="chance_probabilities_not_one",
                            message=(
                                f"Probabilities under '{node.label}' sum to {total:g}, not 1."
                            ),
                            path=list(path),
                        )
                    )
        for child in node.children:
            _walk(child, path + [child.label])

    _walk(root, [root.label])


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _solve(
    node: DecisionNode,
    opt: Literal["max", "min"],
    diagnostics: list[Diagnostic],
    path: list[str],
) -> float | None:
    """Post-order EV rollback. Returns ``None`` if not solvable."""
    if node.kind == "terminal":
        if node.payoff is None:
            return None
        if node.cost is not None:
            node.payoff = node.payoff - node.cost
        return node.payoff

    child_payoffs: list[float] = []
    for child in node.children:
        cp = _solve(child, opt, diagnostics, path + [child.label])
        if cp is None:
            return None
        child_payoffs.append(cp)

    if not node.children:
        return None

    if node.kind == "chance":
        probs = [c.p for c in node.children]
        if any(p is None for p in probs):
            return None
        ev = sum(p * po for p, po in zip(probs, child_payoffs))  # type: ignore[arg-type]
        if node.cost is not None:
            ev -= node.cost
        node.payoff = ev
        return ev

    if node.kind == "decision":
        if opt == "max":
            best = max(child_payoffs)
        else:
            best = min(child_payoffs)
        chosen_ids: list[str] = []
        for child, cp in zip(node.children, child_payoffs):
            if math.isclose(cp, best, abs_tol=1e-9, rel_tol=1e-9):
                chosen_ids.append(child.id)
        node.chosen_child_ids = chosen_ids
        if node.cost is not None:
            best = best - node.cost
        node.payoff = best
        if len(chosen_ids) > 1:
            diagnostics.append(
                Diagnostic(
                    severity="warning",
                    code="decision_tie",
                    message=(
                        f"Decision '{node.label}' has a tie between {len(chosen_ids)} branches."
                    ),
                    path=list(path),
                )
            )
        return best

    return None


def _clone_tree(node: DecisionNode) -> DecisionNode:
    clone = DecisionNode(
        id=node.id,
        label=node.label,
        kind=node.kind,
        level=node.level,
        p=node.p,
        cost=node.cost,
        payoff=node.payoff,
        raw_p=node.raw_p,
        raw_cost=node.raw_cost,
        raw_payoff=node.raw_payoff,
        children=[_clone_tree(c) for c in node.children],
        chosen_child_ids=list(node.chosen_child_ids),
    )
    return clone


def _strip_solved_values(node: DecisionNode) -> None:
    """Reset rolled-up payoffs on non-terminal nodes (for the initial view)."""
    if node.kind != "terminal":
        node.payoff = None
        node.chosen_child_ids = []
    for child in node.children:
        _strip_solved_values(child)


# ---------------------------------------------------------------------------
# Solution table
# ---------------------------------------------------------------------------


def _solution_rows(node: DecisionNode) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _walk(n: DecisionNode) -> None:
        rows.append(
            {
                "level": n.level,
                "label": n.label,
                "type": n.kind,
                "p": n.p,
                "payoff": n.payoff,
                "cost": n.cost,
                "id": n.id,
                "chosen": list(n.chosen_child_ids),
            }
        )
        for c in n.children:
            _walk(c)

    _walk(node)
    return rows


# ---------------------------------------------------------------------------
# Mermaid
# ---------------------------------------------------------------------------


def _format_money(value: float | None, symbol: str, dec: int) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return f"{symbol}?"
    sign = "-" if value < 0 else ""
    return f"{sign}{symbol}{abs(value):,.{dec}f}"


def _mermaid_label(node: DecisionNode, final: bool, symbol: str, dec: int) -> str:
    if final or node.kind == "terminal":
        body = _format_money(node.payoff, symbol, dec)
    else:
        body = " "
    if node.kind == "decision":
        return f'{node.id}["{body}"]'
    if node.kind == "chance":
        return f'{node.id}(("{body}"))'
    return f'{node.id}["{body}"]'


def _mermaid_edge(parent: DecisionNode, child: DecisionNode, final: bool, dec: int) -> str:
    label_parts: list[str] = [child.label]
    if parent.kind == "chance" and child.p is not None:
        label_parts.append(f"p={child.p:.{dec + 2}f}")
    text = ": ".join(label_parts)
    text = text.replace('"', "'")
    if final and parent.kind == "decision" and child.id in parent.chosen_child_ids:
        return f'{parent.id} === |"{text}"| {child.id}'
    return f'{parent.id} --- |"{text}"| {child.id}'


def _to_mermaid(
    root: DecisionNode,
    *,
    final: bool,
    orient: str,
    dec: int,
    symbol: str,
) -> str:
    if orient not in {"LR", "TD"}:
        orient = "LR"
    lines: list[str] = [f"graph {orient}"]
    decisions_plain: list[str] = []
    decisions_cost: list[str] = []
    chance_plain: list[str] = []
    chance_cost: list[str] = []
    chosen_edges: list[tuple[str, str]] = []

    def _walk(n: DecisionNode) -> None:
        if n.kind == "decision":
            (decisions_cost if n.cost is not None else decisions_plain).append(n.id)
        elif n.kind == "chance":
            (chance_cost if n.cost is not None else chance_plain).append(n.id)
        lines.append(f"    {_mermaid_label(n, final, symbol, dec)}")
        for c in n.children:
            lines.append(f"    {_mermaid_edge(n, c, final, dec)}")
            if final and n.kind == "decision" and c.id in n.chosen_child_ids:
                chosen_edges.append((n.id, c.id))
            _walk(c)

    _walk(root)

    if decisions_plain:
        lines.append("    classDef decision fill:#9ACD32,stroke:#333,stroke-width:1px;")
        lines.append(f"    class {','.join(decisions_plain)} decision;")
    if decisions_cost:
        lines.append(
            "    classDef decision_with_cost fill:#9ACD32,stroke:#333,"
            "stroke-width:3px,stroke-dasharray:4 5;"
        )
        lines.append(f"    class {','.join(decisions_cost)} decision_with_cost;")
    if chance_plain:
        lines.append("    classDef chance fill:#FF8C00,stroke:#333,stroke-width:1px;")
        lines.append(f"    class {','.join(chance_plain)} chance;")
    if chance_cost:
        lines.append(
            "    classDef chance_with_cost fill:#FF8C00,stroke:#333,"
            "stroke-width:3px,stroke-dasharray:4 5;"
        )
        lines.append(f"    class {','.join(chance_cost)} chance_with_cost;")
    if final and chosen_edges:
        lines.append("    %% chosen branches drawn with === (highlighted in final tree)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# YAML serialization
# ---------------------------------------------------------------------------


def _node_to_yaml_dict(
    node: DecisionNode,
    *,
    is_root: bool,
    parent_kind: NodeKind | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if is_root:
        out["name"] = node.label
    if node.kind in ("decision", "chance"):
        out["type"] = node.kind
    if parent_kind == "chance" and node.p is not None:
        out["p"] = node.p
    if node.cost is not None and not is_root:
        # cost is allowed on the root too, but most trees don't use it.
        out["cost"] = node.cost
    if node.kind == "terminal" and node.payoff is not None:
        out["payoff"] = node.payoff
    for c in node.children:
        out[c.label] = _node_to_yaml_dict(c, is_root=False, parent_kind=node.kind)
    return out


def _yaml_dump(obj: dict[str, Any]) -> str:
    return yaml.safe_dump(obj, sort_keys=False, default_flow_style=False, indent=4)


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


@dataclass
class SensitivitySpec:
    """One varied variable in a sensitivity sweep.

    ``values`` is the list of points to evaluate. Each spec corresponds to
    one axis of the cartesian product across all varied variables.
    """

    name: str
    values: list[float]
    note: str = ""


@dataclass
class SensitivityCell:
    """One grid cell of a sensitivity sweep.

    Cells are preserved even when the solver fails — invalid probability
    combinations (e.g. negative residual probabilities) end up as cells
    with ``payoff=None`` and the diagnostics that explain why.
    """

    inputs: dict[str, float]
    payoff: float | None
    chosen_labels: list[str]
    errors: list[Diagnostic] = field(default_factory=list)
    warnings: list[Diagnostic] = field(default_factory=list)

    @property
    def is_solved(self) -> bool:
        return self.payoff is not None and not self.errors


@dataclass
class SensitivityResult:
    """Structured result of a 1-, 2-, or 3-variable sensitivity sweep."""

    specs: list[SensitivitySpec]
    cells: list[SensitivityCell]
    opt: Literal["max", "min"]
    base_payoff: float | None
    base_chosen: list[str]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(s.values) for s in self.specs)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    @property
    def valid_cells(self) -> list[SensitivityCell]:
        return [c for c in self.cells if c.is_solved]

    @property
    def invalid_cells(self) -> list[SensitivityCell]:
        return [c for c in self.cells if not c.is_solved]

    def to_frame(self) -> pd.DataFrame:
        """Flat tabular view: one row per cell, one column per varied var."""
        rows: list[dict[str, Any]] = []
        for cell in self.cells:
            row: dict[str, Any] = dict(cell.inputs)
            row["payoff"] = cell.payoff
            row["chosen"] = "; ".join(cell.chosen_labels)
            row["error_codes"] = "; ".join(d.code for d in cell.errors)
            rows.append(row)
        return pd.DataFrame(rows)

    def threshold_flips(self) -> list[dict[str, Any]]:
        """1-way grids only: adjacent cells where the chosen branch flips.

        Returns an empty list for 2- or 3-way grids; use ``to_frame()``
        and slice along an axis to compute conditional thresholds.
        """
        if len(self.specs) != 1:
            return []
        spec = self.specs[0]
        flips: list[dict[str, Any]] = []
        prev: SensitivityCell | None = None
        for cell in self.cells:
            if not cell.is_solved:
                prev = None
                continue
            if prev is not None and tuple(sorted(prev.chosen_labels)) != tuple(
                sorted(cell.chosen_labels)
            ):
                flips.append(
                    {
                        "variable": spec.name,
                        "from_value": prev.inputs[spec.name],
                        "to_value": cell.inputs[spec.name],
                        "from_chosen": list(prev.chosen_labels),
                        "to_chosen": list(cell.chosen_labels),
                    }
                )
            prev = cell
        return flips

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot (for serializing alongside the tree state)."""
        return {
            "opt": self.opt,
            "base_payoff": self.base_payoff,
            "base_chosen": list(self.base_chosen),
            "specs": [
                {"name": s.name, "values": list(s.values), "note": s.note} for s in self.specs
            ],
            "shape": list(self.shape),
            "cells": [
                {
                    "inputs": dict(c.inputs),
                    "payoff": c.payoff,
                    "chosen_labels": list(c.chosen_labels),
                    "errors": [d.to_dict() for d in c.errors],
                    "warnings": [d.to_dict() for d in c.warnings],
                    "is_solved": c.is_solved,
                }
                for c in self.cells
            ],
        }


def _make_sensitivity_spec(name: str, body: Any) -> SensitivitySpec:
    """Normalize one variable spec from dict/list input.

    Accepted shapes:
    - ``{"values": [0.3, 0.5, 0.7], "note": "..."}``
    - ``{"min": 0.3, "max": 0.7, "step": 0.1}``
    - a bare list ``[0.3, 0.5, 0.7]`` (treated as ``values``).
    """
    if isinstance(body, list | tuple):
        body = {"values": list(body)}
    if not isinstance(body, dict):
        raise TypeError(
            f"sensitivity spec for {name!r} must be a dict or list, got {type(body).__name__}"
        )
    values_raw = body.get("values")
    if values_raw:
        values = [float(v) for v in values_raw]
    else:
        missing = [k for k in ("min", "max", "step") if k not in body]
        if missing:
            raise ValueError(
                f"sensitivity spec for {name!r} needs 'values' or "
                f"'min'+'max'+'step'; missing {missing}"
            )
        lo = float(body["min"])
        hi = float(body["max"])
        step = float(body["step"])
        if step <= 0:
            raise ValueError(f"step for {name!r} must be positive, got {step}")
        if hi < lo:
            raise ValueError(f"max < min for {name!r}: {hi} < {lo}")
        n = int(round((hi - lo) / step)) + 1
        # round trims 0.1+0.2 style float jitter so axis labels stay clean.
        values = [round(lo + i * step, 12) for i in range(n)]
    if not values:
        raise ValueError(f"sensitivity spec for {name!r} produced no values")
    note = str(body.get("note", ""))
    return SensitivitySpec(name=name, values=values, note=note)


def _parse_sensitivity_specs(
    specs: dict[str, Any] | list[Any],
) -> list[SensitivitySpec]:
    """Accept dict-of-dicts, list-of-dicts, or list-of-SensitivitySpec."""
    if isinstance(specs, dict):
        return [_make_sensitivity_spec(name, body) for name, body in specs.items()]
    parsed: list[SensitivitySpec] = []
    for entry in specs:
        if isinstance(entry, SensitivitySpec):
            parsed.append(entry)
        elif isinstance(entry, dict):
            name = entry.get("name")
            if not name:
                raise ValueError("list-form sensitivity spec must include 'name'")
            parsed.append(_make_sensitivity_spec(str(name), entry))
        else:
            raise TypeError(f"unsupported sensitivity spec entry: {type(entry).__name__}")
    return parsed


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class dtree:
    """Decision-tree analysis with expected-value rollback.

    Parameters
    ----------
    yl : str or dict
        YAML text (Radiant-compatible) or a parsed nested dict describing
        the tree.
    opt : {"max", "min"}, default "max"
        Decision-node optimization direction.

    Attributes
    ----------
    yl : str
        Original YAML text (or pre-serialized dict input).
    normalized : DecisionNode
        Solved tree.
    initial_tree : DecisionNode
        Tree before rollback (payoffs only on terminals).
    vars : dict[str, float]
        Resolved variable values.
    diagnostics : list[Diagnostic]
        Structured validation/solver messages.
    solution_df : pandas.DataFrame
        Solution table (label, level, payoff, p, cost, type, id, chosen).
    payoff : float | None
        Root payoff (None if the tree did not solve).
    prob : dict[str, float | None]
        Mapping of label -> p for each node (None where not applicable).
    tree : DecisionNode
        Alias for ``normalized``.
    opt : str
        Optimization direction in effect.
    python_code : str
        Reproducible Python snippet for the solved tree.
    name : str
        Tree name (from the YAML ``name:`` field, or the root label).
    """

    def __init__(
        self,
        yl: str | dict[str, Any],
        opt: str = "max",
        *,
        subtree_library: dict[str, dtree] | None = None,
    ) -> None:
        if opt not in {"max", "min"}:
            raise ValueError(f"opt must be 'max' or 'min', got {opt!r}")
        self.opt: Literal["max", "min"] = opt  # type: ignore[assignment]

        if isinstance(yl, dict):
            self.yl: str = _yaml_dump(yl)
            raw_dict: dict[str, Any] | None = yl
            parse_diags: list[Diagnostic] = []
        else:
            self.yl = yl
            raw_dict, parse_diags = _yaml_to_dict(yl)

        diagnostics: list[Diagnostic] = list(parse_diags)
        self.vars: dict[str, float] = {}
        self.name: str = ""
        self.normalized: DecisionNode | None = None
        self.initial_tree: DecisionNode | None = None
        self.subtree_library: dict[str, dtree] = dict(subtree_library or {})
        # Track which subtree refs were actually consumed so the consumer
        # (and the UI) can show "this tree depends on lawsuit_tree".
        self.subtree_refs: dict[str, float] = {}

        def _resolve_subtree(name: str) -> float:
            entry = self.subtree_library.get(name)
            if entry is None:
                raise UnknownSubtreeError(name)
            value = getattr(entry, "payoff", None)
            if value is None:
                raise UnknownSubtreeError(name)
            self.subtree_refs[name] = float(value)
            return float(value)

        # Always install the resolver — even without a library — so a
        # ``subtree("name")`` call without a backing tree surfaces as the
        # specific ``unknown_subtree`` diagnostic rather than falling
        # through to the generic ``unsafe_expression`` path.
        resolver = _resolve_subtree

        if raw_dict is not None:
            # Pull out the tree-level metadata.
            self.name = str(raw_dict.get("name") or "")
            raw_vars = raw_dict.get("variables")
            resolved_vars, safe_vars, label_to_safe, var_diags = _resolve_variables(
                raw_vars, subtree_resolver=resolver
            )
            self.vars = resolved_vars
            diagnostics.extend(var_diags)

            # Identify the root label: prefer `name:`, else the first non-reserved key.
            child_keys = [
                k for k in raw_dict.keys() if isinstance(k, str) and k not in _RESERVED_KEYS
            ]
            root_label = self.name or (child_keys[0] if child_keys else "Tree")

            id_alloc = _IdAllocator()
            root = _build_nodes(
                raw_dict,
                safe_vars,
                label_to_safe,
                id_alloc=id_alloc,
                label=root_label,
                path=[root_label],
                level=1,
                diagnostics=diagnostics,
                subtree_resolver=resolver,
            )

            _validate(root, diagnostics)

            if root is not None:
                self.initial_tree = _clone_tree(root)
                _strip_solved_values(self.initial_tree)

                if not any(d.severity == "error" for d in diagnostics):
                    _solve(root, self.opt, diagnostics, [root.label])
                self.normalized = root

        self.diagnostics: list[Diagnostic] = diagnostics
        self.tree = self.normalized
        self.payoff: float | None = self.normalized.payoff if self.normalized is not None else None
        self.prob: dict[str, float | None] = self._build_prob_map()
        self.solution_df: pd.DataFrame = self._build_solution_df()
        self.python_code: str = self._build_python_code()

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
        return self.normalized is not None and not self.errors and self.payoff is not None

    def _build_prob_map(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        if self.normalized is None:
            return out

        def _walk(n: DecisionNode) -> None:
            out[n.label] = n.p
            for c in n.children:
                _walk(c)

        _walk(self.normalized)
        return out

    def _build_solution_df(self) -> pd.DataFrame:
        if self.normalized is None:
            return pd.DataFrame(columns=["level", "label", "type", "p", "payoff", "cost", "id"])
        rows = _solution_rows(self.normalized)
        df = pd.DataFrame(rows)
        return df

    def _build_python_code(self) -> str:
        lines = [
            "import pyrsm as rsm",
            "",
            "yaml_text = '''",
            self.yl.rstrip() + "\n",
            "'''",
            "",
            f"tree = rsm.decide.dtree(yaml_text, opt={self.opt!r})",
            "tree.summary(input=True, output=True)",
            "print(tree.solution_df)",
            "print(tree.to_mermaid(final=True))",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def summary(self, input: bool = True, output: bool = False, dec: int = 2) -> None:
        """Print a Radiant-style summary of the decision tree."""
        buf = StringIO()
        if self.errors:
            buf.write("Decision tree errors:\n")
            for d in self.errors:
                path = " > ".join(d.path) if d.path else "(root)"
                buf.write(f"  [{d.code}] {path}: {d.message}\n")
            buf.write("\n")
        if self.warnings:
            buf.write("Decision tree warnings:\n")
            for d in self.warnings:
                path = " > ".join(d.path) if d.path else "(root)"
                buf.write(f"  [{d.code}] {path}: {d.message}\n")
            buf.write("\n")
        if input:
            buf.write("Decision tree input:\n")
            buf.write(self.yl.rstrip() + "\n\n")
        if self.vars:
            buf.write("Variable values:\n")
            for k, v in self.vars.items():
                buf.write(f"  {k}: {v:.{dec}f}\n")
            buf.write("\n")
        if output and self.normalized is not None:
            buf.write(f"Optimization direction: {self.opt}\n")
            if self.payoff is not None:
                buf.write(f"Root payoff ('{self.normalized.label}'): {self.payoff:,.{dec}f}\n\n")

            buf.write("Initial decision tree:\n")
            self._print_tree(buf, self.initial_tree, dec=dec, show_payoff=False)
            buf.write("\n")

            buf.write("Final decision tree:\n")
            self._print_tree(buf, self.normalized, dec=dec, show_payoff=True)
            buf.write("\n")
        print(buf.getvalue().rstrip())

    def _print_tree(
        self,
        buf: StringIO,
        node: DecisionNode | None,
        *,
        dec: int,
        show_payoff: bool,
        prefix: str = "",
    ) -> None:
        if node is None:
            buf.write("  (empty)\n")
            return
        marker = ""
        if node.kind == "decision":
            marker = "[D]"
        elif node.kind == "chance":
            marker = "(C)"
        else:
            marker = " T "
        bits: list[str] = [f"{prefix}{marker} {node.label}"]
        if node.p is not None:
            bits.append(f"p={node.p:.{dec + 2}f}")
        if node.cost is not None:
            bits.append(f"cost={node.cost:,.{dec}f}")
        if show_payoff and node.payoff is not None:
            bits.append(f"payoff={node.payoff:,.{dec}f}")
        elif not show_payoff and node.kind == "terminal" and node.payoff is not None:
            bits.append(f"payoff={node.payoff:,.{dec}f}")
        buf.write("  " + "  ".join(bits) + "\n")
        for c in node.children:
            self._print_tree(
                buf,
                c,
                dec=dec,
                show_payoff=show_payoff,
                prefix=prefix + "  ",
            )

    def to_mermaid(
        self,
        final: bool = False,
        orient: str = "LR",
        dec: int = 2,
        symbol: str = "$",
    ) -> str:
        """Return Mermaid-flowchart source for the (initial or final) tree."""
        if self.normalized is None:
            return 'graph LR\n    err["Tree could not be parsed"]'
        tree = self.normalized if final else self.initial_tree
        if tree is None:
            return 'graph LR\n    err["Tree could not be parsed"]'
        return _to_mermaid(tree, final=final, orient=orient, dec=dec, symbol=symbol)

    def to_yaml(self) -> str:
        """Serialize the normalized tree back to YAML text."""
        if self.normalized is None:
            return self.yl
        out: dict[str, Any] = {}
        out["name"] = self.name or self.normalized.label
        if self.vars:
            out["variables"] = {k: v for k, v in self.vars.items()}
        body = _node_to_yaml_dict(self.normalized, is_root=True, parent_kind=None)
        # ``body`` already includes ``name``; drop the duplicate so we keep
        # ``name`` at the top of the file.
        body.pop("name", None)
        out.update(body)
        return _yaml_dump(out)

    # ------------------------------------------------------------------
    # JSON-safe state
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Sensitivity
    # ------------------------------------------------------------------

    def sensitivity(
        self,
        specs: dict[str, Any] | list[Any],
    ) -> SensitivityResult:
        """Run a 1-, 2-, or 3-variable sensitivity sweep over named variables.

        Each varied variable must already exist in this tree's ``variables:``
        block — sensitivity over a raw inline payoff isn't supported; lift
        it into a named variable first. Dependent formulas like
        ``p_failure = 1 - p_success`` recompute automatically per grid cell
        because the entire variable-resolution pipeline reruns for each
        combination. Cells whose probabilities go out of range or whose
        chance siblings stop summing to 1 are kept in the result with
        their diagnostics — they are not silently dropped.

        Parameters
        ----------
        specs : dict or list
            ``{"p_success": {"values": [0.3, 0.5, 0.7]}}`` or
            ``{"legal_fees": {"min": 2500, "max": 15000, "step": 2500}}``.
            A list form ``[{"name": "p_success", "values": [...]}, ...]``
            is also accepted, as is a bare list of values per variable.

        Returns
        -------
        SensitivityResult
            Structured grid result with payoff, chosen branch(es), and any
            diagnostics per cell.
        """
        parsed = _parse_sensitivity_specs(specs)
        if not parsed:
            raise ValueError("sensitivity requires at least one variable spec")
        if len(parsed) > 3:
            raise ValueError(f"sensitivity supports 1-3 variables, got {len(parsed)}")

        base_dict, parse_diags = _yaml_to_dict(self.yl)
        if base_dict is None or any(d.severity == "error" for d in parse_diags):
            raise ValueError(
                "sensitivity requires a parseable base tree; "
                f"got diagnostics: {[d.code for d in parse_diags]}"
            )

        base_vars = base_dict.get("variables") or {}
        if not isinstance(base_vars, dict):
            raise ValueError("sensitivity requires a `variables:` mapping in the tree YAML")
        missing = [s.name for s in parsed if s.name not in base_vars]
        if missing:
            raise ValueError(
                "sensitivity variables must already exist in the tree's "
                f"`variables:` block; missing: {missing}. "
                f"Available: {list(base_vars)}"
            )

        base_chosen: list[str] = []
        if self.normalized is not None:
            base_chosen = [
                c.label
                for c in self.normalized.children
                if c.id in self.normalized.chosen_child_ids
            ]

        import copy
        import itertools

        cells: list[SensitivityCell] = []
        for combo in itertools.product(*(s.values for s in parsed)):
            inputs = {s.name: v for s, v in zip(parsed, combo)}
            cell_dict = copy.deepcopy(base_dict)
            for name, value in inputs.items():
                cell_dict["variables"][name] = value
            cell_tree = dtree(cell_dict, opt=self.opt, subtree_library=self.subtree_library)
            chosen: list[str] = []
            if cell_tree.normalized is not None:
                chosen = [
                    c.label
                    for c in cell_tree.normalized.children
                    if c.id in cell_tree.normalized.chosen_child_ids
                ]
            cells.append(
                SensitivityCell(
                    inputs=inputs,
                    payoff=cell_tree.payoff if cell_tree.is_solved else None,
                    chosen_labels=chosen,
                    errors=list(cell_tree.errors),
                    warnings=list(cell_tree.warnings),
                )
            )

        return SensitivityResult(
            specs=parsed,
            cells=cells,
            opt=self.opt,
            base_payoff=self.payoff,
            base_chosen=base_chosen,
        )

    # ------------------------------------------------------------------
    # JSON-safe state
    # ------------------------------------------------------------------

    def to_state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of solver state.

        ``solution_table`` is sourced directly from the typed nodes rather
        than the pandas DataFrame to preserve ``None`` instead of ``NaN``;
        Postgres JSONB rejects ``NaN`` so the state must stay strict-JSON.
        """
        if self.normalized is not None:
            solution_table = _solution_rows(self.normalized)
        else:
            solution_table = []
        return {
            "name": self.name,
            "opt": self.opt,
            "yaml_text": self.yl,
            "normalized_yaml": self.to_yaml() if self.normalized else self.yl,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "solution_table": solution_table,
            "initial_tree": (self.initial_tree.to_dict() if self.initial_tree else None),
            "final_tree": (self.normalized.to_dict() if self.normalized else None),
            "mermaid_initial": self.to_mermaid(final=False),
            "mermaid_final": self.to_mermaid(final=True),
            "vars": dict(self.vars),
            "python_code": self.python_code,
            "payoff": self.payoff,
            "is_solved": self.is_solved,
        }


__all__ = [
    "DecisionNode",
    "Diagnostic",
    "SensitivityCell",
    "SensitivityResult",
    "SensitivitySpec",
    "UnknownSubtreeError",
    "UnresolvedVariableError",
    "UnsafeExpressionError",
    "dtree",
    "safe_eval",
]


def _make_module_callable() -> None:
    """Keep ``pyrsm.decide.dtree(...)`` callable after submodule imports."""
    import sys
    import types

    class _CallableModule(types.ModuleType):
        def __call__(self, *args, **kwargs):
            return dtree(*args, **kwargs)

    sys.modules[__name__].__class__ = _CallableModule


_make_module_callable()
