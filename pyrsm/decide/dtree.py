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

_RESERVED_KEYS = {"name", "variables", "type", "p", "payoff", "cost", "tree", "with"}


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
    #: The tree this branch grafts (``tree: name``), if any
    source_tree: str | None = None
    #: The ``with:`` values used for that tree, by variable name
    tree_with: dict[str, float] = field(default_factory=dict)

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
            "source_tree": self.source_tree,
            "tree_with": dict(self.tree_with),
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
    """Raised when ``dtree("name")`` refers to a name not in the library."""

    def __init__(self, name: str, detail: str = ""):
        super().__init__(name)
        self.name = name
        self.detail = detail


#: A reference to another tree: R's ``dtree("name")`` (pyrsm also accepts
#: ``subtree("name")``).
_TREE_REF_RE = re.compile(r"\b(dtree|subtree)\(\s*[\"']([^\"']+)[\"']\s*\)")


def _is_tree_ref(value: Any) -> bool:
    return isinstance(value, str) and bool(_TREE_REF_RE.search(value))


def _library_entry(entry: Any) -> tuple[Any, str]:
    """(YAML text or dict, opt) for a ``subtree_library`` value."""
    if isinstance(entry, str):
        return entry, "max"
    if isinstance(entry, dict) and ("yaml_text" in entry or "yl" in entry):
        opt = entry.get("opt", "max")
        return entry.get("yaml_text", entry.get("yl", "")), opt if opt in ("max", "min") else "max"
    if isinstance(entry, dict):
        return entry, "max"
    return getattr(entry, "yl", ""), getattr(entry, "opt", "max")


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

    # R's power operator (radiant.model formulas): ``2^3`` is ``2**3``. There
    # is no bitwise XOR in a decision tree.
    text = expr.strip().replace("^", "**")
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
                and node.func.id in ("subtree", "dtree")
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
    forward: bool = False,
) -> tuple[dict[str, float], dict[str, float], dict[str, str], list[Diagnostic]]:
    """Resolve the ``variables:`` block.

    Returns ``(resolved, safe_vars, label_to_safe, diagnostics)``:

    - ``resolved``: original label -> numeric value (for display);
    - ``safe_vars``: safe identifier -> numeric value (for ``safe_eval``);
    - ``label_to_safe``: original label -> safe identifier (for the
      substitution pass on p/cost/payoff expressions);
    - ``diagnostics``: list of diagnostics for anything that failed.

    ``subtree_resolver`` is forwarded to ``safe_eval`` so a variable
    expression may include ``subtree("name")`` calls. With ``forward`` a
    variable may use one defined further down (the merged variables of a
    grafted sub-tree are not in dependency order); otherwise they resolve in
    order, as in radiant.model.
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

    def _resolve_one(label: Any, expr: Any) -> Diagnostic | None:
        safe_name = label_to_safe[str(label)]
        try:
            if subtree_resolver is not None and _is_tree_ref(expr):
                # dtree("name") -> the EMV of that tree, before label
                # substitution can touch the quoted name
                expr = _TREE_REF_RE.sub(
                lambda m: repr(float(subtree_resolver(m.group(2)))), str(expr)
                )
            substituted = _substitute_labels(expr, label_to_safe)
            value = safe_eval(substituted, safe_vars, subtree_resolver=subtree_resolver)
        except UnknownSubtreeError as exc:
            return Diagnostic(
                severity="error",
                code="unknown_subtree",
                message=(
                    f"Variable '{label}': {exc.detail}"
                    if exc.detail
                    else f"Variable '{label}' references unknown tree '{exc.name}'. "
                    "Add a tree with that name, or fix the name."
                ),
                path=["variables", str(label)],
            )
        except UnresolvedVariableError as exc:
            return Diagnostic(
                severity="error",
                code="unresolved_variable",
                message=(f"Variable '{label}' references unknown name '{exc.name}'."),
                path=["variables", str(label)],
            )
        except UnsafeExpressionError as exc:
            return Diagnostic(
                severity="error",
                code="unsafe_expression",
                message=(f"Variable '{label}' has an unsafe or invalid expression: {exc}"),
                path=["variables", str(label)],
            )
        except ZeroDivisionError:
            return Diagnostic(
                severity="error",
                code="division_by_zero",
                message=f"Variable '{label}' divides by zero.",
                path=["variables", str(label)],
            )
        if not math.isfinite(value):
            return Diagnostic(
                severity="error",
                code="non_finite_variable",
                message=f"Variable '{label}' evaluates to {value!r}.",
                path=["variables", str(label)],
            )
        safe_vars[safe_name] = value
        resolved[str(label)] = value
        return None

    pending = list(raw_vars.items())
    while pending:
        retry: list[tuple[Any, Any]] = []
        failed: list[Diagnostic] = []
        for label, expr in pending:
            diag = _resolve_one(label, expr)
            if diag is None:
                continue
            if forward and diag.code == "unresolved_variable":
                retry.append((label, expr))
            failed.append(diag)
        if not retry or len(retry) == len(pending):
            diagnostics.extend(failed)
            break
        # a variable that uses one defined further down: try again now that
        # more names are known
        diagnostics.extend(d for d in failed if d.code != "unresolved_variable")
        pending = retry

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


def dtree_parser(yl: str) -> tuple[str, list[str]]:
    """Check and clean tree input as radiant.model's ``dtree_parser``.

    Returns the cleaned text and a list of problems, each a message that
    names the line(s) to fix: a missing ``:``, a node type other than
    chance/decision, a node type not followed by a node name, and
    indentation that does not match. Cleaning: tabs become four spaces,
    accented characters become plain ASCII, a space is added after ``:``,
    ``p: .4`` becomes ``p: 0.4``, the keywords are made lower case, and
    brackets and similar characters in node names become ``/``.
    """
    import unicodedata

    lines = str(yl).split("\n")
    lines = [
        unicodedata.normalize("NFKD", ln).encode("ascii", "ignore").decode("ascii").replace("\t", "    ")
        for ln in lines
    ]
    err: list[str] = []

    missing = [i + 1 for i, ln in enumerate(lines) if not (":" in ln or not ln.strip() or ln.lstrip().startswith("#"))]
    if missing:
        err.append(
            "Each line in the tree input must have a ':'. Add a ':' in line(s): "
            + ", ".join(map(str, missing))
        )

    lines = [re.sub(r":\s{2,}", ": ", re.sub(r":([^ $])", r": \1", ln)) for ln in lines]
    lines = [re.sub(r"(^\s*p\s*:)\s*(\.[0-9]+$)", r"\1 0\2", ln) for ln in lines]
    for key in ("name", "variables", "type", "p", "payoff", "cost"):
        lines = [re.sub(rf"(^\s*){key}(\s*:)", rf"\g<1>{key}\2", ln, flags=re.I) for ln in lines]

    type_id = [i for i, ln in enumerate(lines) if re.match(r"^\s*type\s*:\s*(.*)$", ln)]
    type_cid = [i for i, ln in enumerate(lines) if re.match(r"^\s*type\s*:\s*((chance)|(decision)|())\s*$", ln)]
    if type_id != type_cid:
        bad = [i + 1 for i in type_id if i not in type_cid]
        err.append("Node type should be 'type: chance', or 'type: decision' in line(s): " + ", ".join(map(str, bad)))

    nc_id = [i for i, ln in enumerate(lines) if not re.match(r"^\s*#", ln)]
    for i in nc_id:
        lines[i] = re.sub(r"(^\s*)[!`@%&*\-+]*\s*", r"\1", lines[i].replace("#", "//"), count=1)

    def _blank_value(ln: str) -> str:
        for key in ("p", "type", "cost", "payoff"):
            ln = re.sub(rf"(^\s*{key}\s*:\s*$)", r"\1 0", ln)
        return ln

    nn_id = [i for i, ln in enumerate(lines) if re.match(r"^\s*[^#]+:\s*$", _blank_value(ln))]
    for i in nn_id:
        lines[i] = re.sub(r"[(){}\[\]<>@;~]", "/", lines[i])

    def indent(i: int | None) -> int | None:
        if i is None or i >= len(lines):
            return None
        return len(lines[i]) - len(lines[i].lstrip(" "))

    after_type = [next((j for j in nc_id if j > i), None) for i in type_cid]
    not_named = [type_cid[k] + 1 for k, j in enumerate(after_type) if j not in nn_id]
    if not_named:
        err.append(
            "The node types defined on line(s) "
            + ", ".join(map(str, not_named))
            + " must be followed by a node name.\nA valid node name could be 'mud slide:'"
        )

    next_name = [next((j for j in nn_id if j > i), None) for i in type_cid]
    bad_type_indent = [
        type_cid[k] + 2
        for k, j in enumerate(next_name)
        if j is not None and indent(type_cid[k]) != indent(j)
    ]
    if bad_type_indent:
        err.append(
            "Indent issue in line(s): "
            + ", ".join(map(str, bad_type_indent))
            + "\nUse the tab key to ensure a node name is indented the same amount\n"
            "as the node type on the preceding line. Check the level of indentation\n"
            "on each line shown, as well as the indentation on the preceding lines"
        )

    bad_name_indent = []
    for i in nn_id:
        nxt = indent(i + 1)
        if nxt is None or indent(i) >= nxt:
            bad_name_indent.append(i + 2)
    if bad_name_indent:
        err.append(
            "Indent issue in line(s): "
            + ", ".join(map(str, bad_name_indent))
            + "\nAlways use the tab key to indent the line(s) after specifying a node name."
        )
    return "\n".join(lines), err


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
    cleaned, problems = dtree_parser(yl)
    if problems:
        for problem in problems:
            diagnostics.append(Diagnostic(severity="error", code="input_format", message=problem))
        return None, diagnostics
    try:
        data = yaml.safe_load(cleaned)
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
    graft: Callable[..., DecisionNode | None] | None = None,
    scope_vars: dict[str, Any] | None = None,
    stack: tuple[str, ...] = (),
) -> DecisionNode | None:
    """Recursively convert a nested YAML mapping into ``DecisionNode``s.

    A branch with ``tree: name`` is handed to ``graft``, which builds that
    tree in its place (in the variables of ``scope_vars`` and its ``with:``
    values); ``stack`` holds the trees grafted on the way here.
    """
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
        child_raw = raw[ck] if isinstance(raw[ck], dict) else {"payoff": raw[ck]}
        if graft is not None and "tree" in child_raw:
            child = graft(
                child_raw,
                safe_vars,
                label_to_safe,
                id_alloc=id_alloc,
                label=ck,
                path=path + [ck],
                level=level + 1,
                diagnostics=diagnostics,
                scope_vars=scope_vars or {},
                stack=stack,
            )
        else:
            if "with" in child_raw:
                diagnostics.append(
                    Diagnostic(
                        severity="error",
                        code="with_without_tree",
                        message=(
                            f"Branch '{ck}' has `with:` but no `tree:`. `with:` sets values "
                            "for the tree a branch uses."
                        ),
                        path=path + [ck],
                    )
                )
            child = _build_nodes(
                child_raw,
                safe_vars,
                label_to_safe,
                id_alloc=id_alloc,
                label=ck,
                path=path + [ck],
                level=level + 1,
                parent_kind=kind,
                diagnostics=diagnostics,
                subtree_resolver=subtree_resolver,
                graft=graft,
                scope_vars=scope_vars,
                stack=stack,
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
        source_tree=node.source_tree,
        tree_with=dict(node.tree_with),
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
# radiant.model's text layout
# ---------------------------------------------------------------------------


def _r_format_column(values: list, digits: int = 7, scipen: int = 0) -> list[str]:
    """Numbers formatted as R prints a column (``formatReal``)."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return ["" for _ in values]
    neg = any(v < 0 for v in finite)
    sig, left, right = [], [], []
    for v in finite:
        if v == 0:
            sig.append(1), left.append(1), right.append(0)
            continue
        mant, exp = f"{abs(v):.{digits - 1}e}".split("e")
        exp = int(exp)
        s_ = len(mant.replace(".", "").rstrip("0")) or 1
        sig.append(s_)
        left.append(max(1, exp + 1))
        right.append(max(0, s_ - 1 - exp))
    rgt = max(right)
    fixed_width = neg + max(left) + (rgt + 1 if rgt > 0 else 0)
    mf = max(sig) - 1
    sci_width = neg + (mf + 1 if mf > 0 else 0) + 5
    out = []
    for v in values:
        if v is None or not math.isfinite(v):
            out.append("")
        elif fixed_width <= sci_width + scipen:
            out.append(f"{v:.{rgt}f}")
        else:
            out.append(f"{v:.{mf}e}")
    return out


def _r_tree_table(root: DecisionNode | None, dec: int = 2) -> str:
    """A tree as radiant.model prints it (data.tree level names and columns)."""
    if root is None:
        return "  (empty)"
    rows: list[tuple[str, str, str, str, str]] = []

    def money(x: float | None) -> str:
        return "" if x is None else f"{x:,.{dec}f}"

    def walk(node: DecisionNode, prefix: str, is_last: bool, parent: DecisionNode | None) -> None:
        if parent is None:
            name = node.label
            child_prefix = ""
        else:
            name = prefix + (" \u00b0--" if is_last else " \u00a6--") + node.label
            child_prefix = prefix + ("    " if is_last else " \u00a6  ")
        prob = "" if node.p is None or parent is None or parent.kind != "chance" else f"{100 * node.p:.2f} %"
        ptype = "" if parent is None else parent.kind
        rows.append((name, prob, money(node.payoff), money(node.cost), ptype))
        for i, child in enumerate(node.children):
            walk(child, child_prefix, i == len(node.children) - 1, node)

    walk(root, "", True, None)
    header = ("", "Probability", "Payoff", "Cost", "Type")
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(5)]
    lines = [" " + " " * widths[0] + "".join(" " + header[i].rjust(widths[i]) for i in range(1, 5))]
    for r in rows:
        lines.append(" " + r[0].ljust(widths[0]) + "".join(" " + r[i].rjust(widths[i]) for i in range(1, 5)))
    return "\n".join(lines)


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
                "tree": n.source_tree,
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


def _tooltip_money(value: float | None, symbol: str, dec: int) -> str:
    return _format_money(value, symbol, dec).replace('"', "'")


def _node_tooltip(
    node: DecisionNode,
    parent: DecisionNode | None,
    *,
    final: bool,
    opt: str,
    symbol: str,
    dec: int,
) -> list[tuple[str, str]]:
    """Hover text for a node: (bold label, text) parts.

    The final tree shows how each value was folded back (a chance node's
    probability-weighted sum, a decision node's best option); both trees show
    the inputs as entered, with the values of any formulas.
    """

    def money(x: float | None) -> str:
        return _tooltip_money(x, symbol, dec)

    def num(x: float | None) -> str:
        return "" if x is None else f"{x:.{dec + 2}f}".rstrip("0").rstrip(".")

    kinds = {"decision": "Decision node", "chance": "Chance node", "terminal": "End node"}
    parts: list[tuple[str, str]] = [(node.label, kinds[node.kind])]
    if node.source_tree:
        parts.append(("Tree", node.source_tree))
        for k, v in node.tree_with.items():
            parts.append(("With", f"{k} = {num(v)}"))
    if parent is not None and parent.kind == "chance" and node.p is not None:
        raw = f"{node.raw_p} = " if node.raw_p else ""
        parts.append(("p", f"{raw}{num(node.p)}"))
    if node.cost is not None:
        raw = f"{node.raw_cost} = " if node.raw_cost else ""
        parts.append(("Cost", f"{raw}{money(node.cost)}"))

    if node.kind == "terminal":
        payoff = node.payoff
        if final and payoff is not None and node.cost is not None:
            # the final payoff already has the cost taken off
            payoff = payoff + node.cost
        raw = f"{node.raw_payoff} = " if node.raw_payoff else ""
        parts.append(("Payoff", f"{raw}{money(payoff)}"))
        if final and node.cost is not None:
            parts.append(("Net", f"{money(payoff)} − {money(node.cost)} = {money(node.payoff)}"))
        return parts
    if not final or node.payoff is None:
        return parts

    if node.kind == "chance":
        terms = " + ".join(f"{num(c.p)} × {money(c.payoff)}" for c in node.children)
        cost = f" − {money(node.cost)}" if node.cost is not None else ""
        parts.append(("EMV", f"{terms}{cost} = {money(node.payoff)}"))
        return parts

    best = "highest" if opt == "max" else "lowest"
    options = "; ".join(f"{c.label} {money(c.payoff)}" for c in node.children)
    chosen = [c.label for c in node.children if c.id in node.chosen_child_ids]
    parts.append(("Options", options))
    parts.append(("Choose", f"{' or '.join(chosen)} ({best} value)"))
    if node.cost is not None:
        gross = node.payoff + node.cost
        parts.append(("Value", f"{money(gross)} − {money(node.cost)} = {money(node.payoff)}"))
    return parts


def _to_mermaid(
    root: DecisionNode,
    *,
    final: bool,
    orient: str,
    dec: int,
    symbol: str,
    opt: str = "max",
    tooltips: bool = True,
    subgraphs: bool = True,
) -> str:
    if orient not in {"LR", "TD"}:
        orient = "LR"
    lines: list[str] = [f"graph {orient}"]
    decisions_plain: list[str] = []
    decisions_cost: list[str] = []
    chance_plain: list[str] = []
    chance_cost: list[str] = []
    chosen_edges: list[tuple[str, str]] = []
    tips: list[str] = []

    def _walk(n: DecisionNode, parent: DecisionNode | None = None) -> None:
        if n.kind == "decision":
            (decisions_cost if n.cost is not None else decisions_plain).append(n.id)
        elif n.kind == "chance":
            (chance_cost if n.cost is not None else chance_plain).append(n.id)
        lines.append(f"    {_mermaid_label(n, final, symbol, dec)}")
        if tooltips:
            parts = _node_tooltip(n, parent, final=final, opt=opt, symbol=symbol, dec=dec)
            text = "<br>".join(f"<b>{k}</b> {v}" for k, v in parts).replace('"', "'")
            tips.append(f'    click {n.id} callback "{text}"')
        for c in n.children:
            lines.append(f"    {_mermaid_edge(n, c, final, dec)}")
            if final and n.kind == "decision" and c.id in n.chosen_child_ids:
                chosen_edges.append((n.id, c.id))
            _walk(c, n)

    _walk(root)

    # A branch that uses another tree (``tree: name``) is drawn in a box
    # named after that tree, so the parts of a combined tree stay visible
    if subgraphs:
        counter = [0]
        boxes: list[str] = []

        def _box(n: DecisionNode, indent: str) -> None:
            counter[0] += 1
            box_id = f"tree{counter[0]}"
            title = n.source_tree or ""
            if n.tree_with:
                values = ", ".join(f"{k} = {v:.4g}" for k, v in n.tree_with.items())
                title = f"{title} · {values}"
            title = title.replace('"', "'")
            lines.append(f'{indent}subgraph {box_id} ["{title}"]')
            boxes.append(box_id)

            def _members(m: DecisionNode) -> None:
                lines.append(f"{indent}    {m.id}")
                for c in m.children:
                    if c.source_tree:
                        _box(c, indent + "    ")
                    else:
                        _members(c)

            _members(n)
            lines.append(f"{indent}end")

        def _find(n: DecisionNode) -> None:
            for c in n.children:
                if c.source_tree:
                    _box(c, "    ")
                else:
                    _find(c)

        _find(root)
        for box_id in boxes:
            lines.append(
                f"    style {box_id} fill:#f8f9fa,stroke:#6c757d,stroke-width:1px,"
                "stroke-dasharray:3 3"
            )

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
    # radiant.model's hover text: ``click id callback "text"`` lines
    lines.extend(tips)

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
    if parent_kind == "chance" and node.p is not None:
        out["p"] = node.p
    if node.cost is not None:
        out["cost"] = node.cost
    if node.kind == "terminal" and node.payoff is not None:
        out["payoff"] = node.payoff
    # ``type:`` comes last, right before the branches it describes, as
    # radiant.model's input rules require
    if node.kind in ("decision", "chance"):
        out["type"] = node.kind
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
    #: EMV of each node by label (first occurrence), e.g. each decision
    decision_payoffs: dict[str, float] = field(default_factory=dict)

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
    #: The options of the tree's first node (what ``plot()`` shows by default)
    base_options: list[str] = field(default_factory=list)

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

    def plot(self, decisions: list[str] | None = None, symbol: str = "$"):
        """EMV of each decision across a one-variable sweep, as radiant.model.

        Parameters
        ----------
        decisions : list[str], optional
            Node labels to show, e.g. the options of the first decision
            ("Sign with Movie Company", "Sign with TV Network"). Defaults to
            the options of the first decision in the tree.
        symbol : str, default "$"
            Currency symbol for the axis.

        Returns
        -------
        plotnine.ggplot
        """
        from plotnine import aes, geom_line, geom_point, ggplot, labs, scale_y_continuous

        from pyrsm.plot_style import theme_pyrsm

        if len(self.specs) != 1:
            raise ValueError("plot() shows a one-variable sweep; run one variable at a time")
        name = self.specs[0].name
        decisions = list(decisions or self.base_options or [])
        rows = [
            {name: c.inputs[name], "Decision": d, "EMV": c.decision_payoffs[d]}
            for c in self.cells
            for d in decisions
            if d in c.decision_payoffs
        ]
        if not rows:
            raise ValueError(f"None of {decisions} has a payoff in the sweep")
        df = pd.DataFrame(rows)
        return (
            ggplot(df, aes(x=name, y="EMV", color="Decision"))
            + geom_line()
            + geom_point()
            + scale_y_continuous(labels=lambda v: [f"{symbol}{x:,.0f}" for x in v])
            + labs(x=name, y="Payoff", color="Decision")
            + theme_pyrsm()
        )

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
                    "decision_payoffs": dict(c.decision_payoffs),
                }
                for c in self.cells
            ],
            "base_options": list(self.base_options),
        }


@dataclass
class TornadoResult:
    """One-at-a-time sensitivity: each variable at its low and high value
    with the others at their base values, ranked by the swing in EMV."""

    base_payoff: float | None
    base_chosen: list[str]
    rows: list[dict[str, Any]]

    def to_frame(self) -> pd.DataFrame:
        """One row per variable, largest swing first."""
        out = []
        for r in self.rows:
            out.append(
                {
                    "variable": r["name"],
                    "base": r["base_value"],
                    "low": r["low"]["value"],
                    "high": r["high"]["value"],
                    "EMV low": r["low"]["payoff"],
                    "EMV high": r["high"]["payoff"],
                    "swing": r["swing"],
                    "decision low": " / ".join(r["low"]["chosen_labels"]),
                    "decision high": " / ".join(r["high"]["chosen_labels"]),
                }
            )
        return pd.DataFrame(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_payoff": self.base_payoff,
            "base_chosen": list(self.base_chosen),
            "vars": [dict(r) for r in self.rows],
        }

    def plot(self, symbol: str = "$"):
        """Tornado chart: the change in EMV from the base value when each
        variable is at its low (red) and high (green) value."""
        from plotnine import (
            aes,
            coord_flip,
            geom_col,
            geom_hline,
            ggplot,
            labs,
            scale_fill_manual,
            scale_x_discrete,
            scale_y_continuous,
        )

        from pyrsm.plot_style import REF, theme_pyrsm

        base = self.base_payoff or 0.0
        data = []
        for r in self.rows:
            for side, label in (("low", "Low input"), ("high", "High input")):
                payoff = r[side]["payoff"]
                if payoff is not None:
                    data.append({"variable": r["name"], "input": label, "change": payoff - base})
        if not data:
            raise ValueError("No variable has a solved low or high value")
        df = pd.DataFrame(data)
        # largest swing at the top
        order = [r["name"] for r in reversed(self.rows)]
        df["variable"] = pd.Categorical(df["variable"], categories=order, ordered=True)
        df["input"] = pd.Categorical(df["input"], categories=["Low input", "High input"])
        return (
            ggplot(df, aes(x="variable", y="change", fill="input"))
            + geom_col(position="identity", width=0.6, color=REF)
            + geom_hline(yintercept=0)
            + coord_flip()
            + scale_x_discrete(limits=order)
            + scale_fill_manual(values={"Low input": "#d62728", "High input": "#2ca02c"})
            + scale_y_continuous(labels=lambda v: [f"{symbol}{x:,.0f}" for x in v])
            + labs(
                x="",
                y=f"Change in EMV from the base value ({symbol}{base:,.0f})",
                fill="",
            )
            + theme_pyrsm()
        )


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
    subtree_library : dict, optional
        Other trees by name, for ``tree: name`` branches, R's
        ``dtree("name")`` references and ``variables: name`` imports. Values can be YAML text, a parsed dict,
        or a ``dtree`` object (its YAML and ``opt`` are used).
    base : dict, optional
        Variables passed down by a tree that references this one (R's
        ``base``): they override this tree's variables of the same name and
        are added when missing. Used internally for ``dtree("name")``.

    Notes
    -----
    A branch can be another tree: ``tree: name`` puts tree ``name`` at that
    branch, so the combined tree is solved, printed and plotted as one. The
    branch keeps its label and ``p:``; a ``cost:`` on it is added to the
    cost of the other tree's first node. The other tree keeps its own
    variables as defaults; this tree's variables replace those with the same
    name, and a ``with:`` list on the branch sets values for that use only
    (evaluated in this tree), for example::

        Positive result:
            p: P(pos)
            type: decision
            Launch:
                tree: launch
                with:
                    P(high): P(high|pos)
            Do not launch:
                payoff: 0

    ``tree_uses`` lists every such branch with its values and EMV.

    The radiant.model ways to combine trees also work:

    * ``x: dtree("name")`` in the ``variables:`` section re-evaluates tree
      ``name`` with *this* tree's variables (except other ``dtree()``
      references) overriding the referenced tree's own, and uses its
      expected value. A sensitivity analysis on a variable therefore flows
      through to the sub-trees that use it.
    * ``variables: name`` (a tree name instead of a list) uses the variables
      of tree ``name``, except its ``dtree()`` references.

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
        subtree_library: dict[str, Any] | None = None,
        base: dict[str, Any] | None = None,
        _stack: tuple[str, ...] = (),
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
        self.subtree_library: dict[str, Any] = dict(subtree_library or {})
        # Track which subtree refs were actually consumed so the consumer
        # (and the UI) can show "this tree depends on lawsuit_tree".
        self.subtree_refs: dict[str, float] = {}
        #: Every tree this one needs, directly or through its sub-trees
        self.used_trees: list[str] = []
        #: The tree whose variables ``variables: name`` imported, if any
        self.variables_from: str = ""
        #: The variables as entered (after imports and base overrides)
        self.raw_vars: dict[str, Any] = {}
        self._stack = tuple(_stack)

        def _use(name: str) -> None:
            if name not in self.used_trees:
                self.used_trees.append(name)

        def _resolve_subtree(name: str) -> float:
            if name in self._stack:
                raise UnknownSubtreeError(
                    name,
                    f"tree '{name}' refers back to itself through "
                    f"{' -> '.join([*self._stack, name])}. Trees cannot reference each other in a circle.",
                )
            entry = self.subtree_library.get(name)
            if entry is None:
                raise UnknownSubtreeError(name)
            text, sub_opt = _library_entry(entry)
            child_base = {k: v for k, v in self.raw_vars.items() if not _is_tree_ref(v)}
            sub = dtree(
                text,
                opt=sub_opt,
                subtree_library=self.subtree_library,
                base=child_base,
                _stack=(*self._stack, name),
            )
            if sub.payoff is None:
                first = sub.errors[0].message if sub.errors else "it did not solve"
                raise UnknownSubtreeError(name, f"tree '{name}' could not be evaluated: {first}")
            _use(name)
            for used in sub.used_trees:
                _use(used)
            self.subtree_refs[name] = float(sub.payoff)
            return float(sub.payoff)

        # Always install the resolver — even without a library — so a
        # ``subtree("name")`` call without a backing tree surfaces as the
        # specific ``unknown_subtree`` diagnostic rather than falling
        # through to the generic ``unsafe_expression`` path.
        resolver = _resolve_subtree
        self._resolver = resolver
        self._use = _use
        #: Each ``tree:`` branch: the tree, where it is used, its ``with:``
        #: values and (once solved) the EMV it contributes
        self.tree_uses: list[dict[str, Any]] = []
        self._graft_nodes: list[DecisionNode | None] = []

        if raw_dict is not None:
            # Pull out the tree-level metadata.
            self.name = str(raw_dict.get("name") or "")
            raw_vars = raw_dict.get("variables")
            if isinstance(raw_vars, str) and raw_vars.strip():
                # radiant.model: ``variables: other_tree`` uses that tree's
                # variables, except its dtree() references
                source = raw_vars.strip()
                imported = self._import_variables(source)
                if imported is None:
                    diagnostics.append(
                        Diagnostic(
                            severity="error",
                            code="unknown_variables_tree",
                            message=(
                                f"The tree referenced in the 'variables:' section ('{source}') is "
                                "not available or has no variables. Correct the name and try again."
                            ),
                            path=["variables"],
                        )
                    )
                    raw_vars = None
                else:
                    raw_vars = imported
                    self.variables_from = source
                    _use(source)
            base_vars = {k: v for k, v in (base or {}).items() if not _is_tree_ref(v)}
            if base_vars:
                if isinstance(raw_vars, dict):
                    raw_vars = {**raw_vars}
                    raw_vars.update(base_vars)
                elif not raw_vars:
                    raw_vars = dict(base_vars)
            if isinstance(raw_vars, dict):
                self.raw_vars = dict(raw_vars)
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
                graft=self._graft,
                scope_vars=self.raw_vars,
                stack=self._stack,
            )

            _validate(root, diagnostics)

            if root is not None:
                self.initial_tree = _clone_tree(root)
                _strip_solved_values(self.initial_tree)

                if not any(d.severity == "error" for d in diagnostics):
                    _solve(root, self.opt, diagnostics, [root.label])
                self.normalized = root

        self.diagnostics: list[Diagnostic] = diagnostics
        for use, node in zip(self.tree_uses, self._graft_nodes):
            solved = node is not None and self.normalized is not None and not self.errors
            use["emv"] = node.payoff if solved else None
        self.tree = self.normalized
        self.payoff: float | None = self.normalized.payoff if self.normalized is not None else None
        self.prob: dict[str, float | None] = self._build_prob_map()
        self.solution_df: pd.DataFrame = self._build_solution_df()
        self.python_code: str = self._build_python_code()

    def _import_variables(self, source: str) -> dict[str, Any] | None:
        """The variables of tree ``source`` (``variables: source``), without
        its ``dtree()`` references; ``None`` when it has none."""
        if source not in self.subtree_library:
            return None
        src_dict, _ = _yaml_to_dict(_library_entry(self.subtree_library[source])[0])
        if src_dict is None or not isinstance(src_dict.get("variables"), dict):
            return None
        return {k: v for k, v in src_dict["variables"].items() if not _is_tree_ref(v)}

    def _graft(
        self,
        branch: dict[str, Any],
        safe_vars: dict[str, float],
        label_to_safe: dict[str, str],
        *,
        id_alloc: _IdAllocator,
        label: str,
        path: list[str],
        level: int,
        diagnostics: list[Diagnostic],
        scope_vars: dict[str, Any],
        stack: tuple[str, ...],
    ) -> DecisionNode | None:
        """Build the tree a ``tree: name`` branch uses in that branch's place.

        The other tree keeps its own variables as defaults; the variables of
        the tree that uses it replace those with the same name, and the
        branch's ``with:`` values (evaluated here, in the using tree) replace
        both. The branch keeps its label and its ``p:``; a ``cost:`` on the
        branch is added to the cost of the other tree's first node.
        """
        name = str(branch.get("tree") or "").strip()

        def fail(code: str, message: str) -> None:
            diagnostics.append(Diagnostic("error", code, message, list(path)))

        if not name:
            fail("tree_name_missing", f"Branch '{label}' has `tree:` without a tree name.")
            return None
        extra = [k for k in branch if isinstance(k, str) and k not in _RESERVED_KEYS]
        if extra:
            fail(
                "tree_branch_has_children",
                f"Branch '{label}' uses tree '{name}', so it cannot also have its own "
                f"branches ({', '.join(extra)}). Put them in tree '{name}', or remove `tree:`.",
            )
            return None
        for key in ("payoff", "type"):
            if key in branch:
                fail(
                    f"tree_branch_has_{key}",
                    f"Branch '{label}' uses tree '{name}', so its `{key}:` comes from that "
                    f"tree. Remove `{key}:` from the branch.",
                )
                return None
        if name in stack:
            fail(
                "tree_cycle",
                f"Tree '{name}' uses itself through {' -> '.join([*stack, name])}. "
                "Trees cannot use each other in a circle.",
            )
            return None
        entry = self.subtree_library.get(name)
        if entry is None:
            known = ", ".join(sorted(self.subtree_library)) or "none"
            fail(
                "unknown_tree",
                f"Branch '{label}' uses tree '{name}', which does not exist. "
                f"Trees in this project: {known}.",
            )
            return None
        text, _ = _library_entry(entry)
        sub_dict, sub_diags = (
            (text, []) if isinstance(text, dict) else _yaml_to_dict(text)
        )
        sub_errors = [d for d in sub_diags if d.severity == "error"]
        if sub_dict is None or sub_errors:
            first = sub_errors[0].message if sub_errors else "it is empty"
            fail("tree_has_errors", f"Tree '{name}' (used at '{label}') has errors: {first}")
            return None

        # Values set by this use, evaluated in the tree that uses it
        with_raw = branch.get("with")
        with_vals: dict[str, float] = {}
        if with_raw is not None and not isinstance(with_raw, dict):
            fail(
                "with_not_mapping",
                f"`with:` under '{label}' must list values, one per line "
                "(for example `P(high): 0.7`).",
            )
            return None
        for key, expr in (with_raw or {}).items():
            value, diag = _eval_numeric_field(
                expr,
                safe_vars,
                label_to_safe,
                path + ["with", str(key)],
                str(key),
                subtree_resolver=self._resolver,
            )
            if diag is not None:
                diagnostics.append(diag)
                return None
            with_vals[str(key)] = float(value)  # type: ignore[arg-type]

        own = sub_dict.get("variables")
        if isinstance(own, str) and own.strip():
            imported = self._import_variables(own.strip())
            if imported is None:
                fail(
                    "unknown_variables_tree",
                    f"Tree '{name}' (used at '{label}') takes its variables from "
                    f"'{own.strip()}', which is not available or has no variables.",
                )
                return None
            # the reproducible code needs that tree too
            self._use(own.strip())
            own = imported
        merged: dict[str, Any] = dict(own) if isinstance(own, dict) else {}
        merged.update({k: v for k, v in scope_vars.items() if not _is_tree_ref(v)})
        unknown_with = [k for k in with_vals if k not in merged]
        if unknown_with:
            fail(
                "unknown_with_variable",
                f"`with:` under '{label}' sets {', '.join(unknown_with)}, which tree '{name}' "
                "does not use as a variable.",
            )
            return None
        merged.update(with_vals)
        _, sub_safe, sub_l2s, var_diags = _resolve_variables(
            merged, subtree_resolver=self._resolver, forward=True
        )
        for d in var_diags:
            d.message = f"Tree '{name}' (used at '{label}'): {d.message}"
            d.path = list(path)
        diagnostics.extend(var_diags)
        if any(d.severity == "error" for d in var_diags):
            return None

        # The branch's own p and cost belong to the tree that uses it
        p_value = cost_value = None
        for key in ("p", "cost"):
            if key not in branch:
                continue
            value, diag = _eval_numeric_field(
                branch[key], safe_vars, label_to_safe, path, key, subtree_resolver=self._resolver
            )
            if diag is not None:
                diagnostics.append(diag)
                return None
            if key == "p":
                p_value = value
            else:
                cost_value = value

        body = {k: v for k, v in sub_dict.items() if k not in ("name", "variables", "p")}
        # listed in the order the branches appear (outer uses first)
        use: dict[str, Any] = {"tree": name, "at": list(path), "with": dict(with_vals)}
        slot = len(self.tree_uses)
        self.tree_uses.append(use)
        self._graft_nodes.append(None)
        node = _build_nodes(
            body,
            sub_safe,
            sub_l2s,
            id_alloc=id_alloc,
            label=label,
            path=path,
            level=level,
            diagnostics=diagnostics,
            subtree_resolver=self._resolver,
            graft=self._graft,
            scope_vars=merged,
            stack=(*stack, name),
        )
        if node is None:
            del self.tree_uses[slot]
            del self._graft_nodes[slot]
            return None
        self._graft_nodes[slot] = node
        node.source_tree = name
        node.tree_with = with_vals
        if p_value is not None:
            node.p = p_value
            raw_p = branch["p"]
            node.raw_p = None if isinstance(raw_p, int | float) else str(raw_p)
        if cost_value is not None:
            raw_cost = branch["cost"]
            raws = [r for r in (node.raw_cost, None if isinstance(raw_cost, int | float) else str(raw_cost)) if r]
            node.cost = (node.cost or 0.0) + cost_value
            node.raw_cost = " + ".join(raws) if raws else None
        self._use(name)
        return node

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
        lines = ["import pyrsm as rsm", ""]
        library = ""
        if self.used_trees:
            # The other trees this one uses (dtree("name") / variables: name)
            lines.append("trees = {")
            for name in self.used_trees:
                text, _ = _library_entry(self.subtree_library.get(name, ""))
                if isinstance(text, dict):
                    text = _yaml_dump(text)
                lines.append(f"    {name!r}: '''")
                lines.append(str(text).rstrip())
                lines.append("''',")
            lines.append("}")
            lines.append("")
            library = ", subtree_library=trees"
        lines += [
            "yaml_text = '''",
            self.yl.rstrip() + "\n",
            "'''",
            "",
            f"tree = rsm.decide.dtree(yaml_text, opt={self.opt!r}{library})",
            "tree.summary(input=True, output=True)",
            "print(tree.to_mermaid(final=True))",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def summary(self, input: bool = True, output: bool = False, dec: int = 2) -> None:
        """Print the decision tree as radiant.model's ``summary.dtree``.

        ``input`` prints the tree specification. ``output`` adds the
        resolved variable values and the initial and final trees as R
        prints them: one row per node with its probability, payoff, cost and
        the type of node it branches from.
        """
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
        if output and self.vars:
            buf.write("Variable input values:\n")
            names = list(self.vars)
            values = _r_format_column(list(self.vars.values()), digits=7, scipen=100)
            nw = max(len(n) for n in names)
            vw = max(len(v) for v in values)
            buf.write(" " * (nw + 1 + vw) + "\n")
            for n, v in zip(names, values):
                buf.write(f"{n.ljust(nw)} {v.rjust(vw)}\n")
        if output and self.tree_uses:
            buf.write("\nTrees used:\n")
            for use in self.tree_uses:
                at = " > ".join(use["at"][1:]) or use["at"][0]
                values = ", ".join(f"{k} = {v:.4g}" for k, v in use["with"].items())
                emv = "" if use.get("emv") is None else f": {use['emv']:,.{dec}f}"
                buf.write(f"  {at} uses '{use['tree']}'{f' with {values}' if values else ''}{emv}\n")
        if output and self.normalized is not None and not self.errors:
            buf.write("\nInitial decision tree:\n")
            buf.write(_r_tree_table(self.initial_tree, dec=dec) + "\n")
            buf.write("\nFinal decision tree:\n")
            buf.write(_r_tree_table(self.normalized, dec=dec) + "\n")
        print(buf.getvalue().rstrip())

    def to_mermaid(
        self,
        final: bool = False,
        orient: str = "LR",
        dec: int = 2,
        symbol: str = "$",
        tooltips: bool = True,
        subgraphs: bool = True,
    ) -> str:
        """Return Mermaid-flowchart source for the (initial or final) tree.

        ``tooltips`` adds hover text for each node as radiant.model does
        (``click id callback "text"`` lines): the inputs, and in the final
        tree how its value was calculated. ``subgraphs`` draws a branch that
        uses another tree (``tree: name``) in a box named after that tree.
        """
        if self.normalized is None:
            return 'graph LR\n    err["Tree could not be parsed"]'
        tree = self.normalized if final else self.initial_tree
        if tree is None:
            return 'graph LR\n    err["Tree could not be parsed"]'
        return _to_mermaid(
            tree,
            final=final,
            orient=orient,
            dec=dec,
            symbol=symbol,
            opt=self.opt,
            tooltips=tooltips,
            subgraphs=subgraphs,
        )

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

        # The variables in effect: the tree's own, those imported with
        # ``variables: other_tree``, and any passed down by a parent tree
        base_vars = dict(self.raw_vars)
        if not base_vars:
            raise ValueError(
                "sensitivity variables must already exist in a `variables:` section of the "
                "tree YAML (its own, or one it imports)"
            )
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
            cell_dict["variables"] = {**copy.deepcopy(base_vars), **inputs}
            cell_tree = dtree(cell_dict, opt=self.opt, subtree_library=self.subtree_library)
            chosen: list[str] = []
            if cell_tree.normalized is not None:
                chosen = [
                    c.label
                    for c in cell_tree.normalized.children
                    if c.id in cell_tree.normalized.chosen_child_ids
                ]
            node_payoffs: dict[str, float] = {}
            if cell_tree.is_solved and cell_tree.normalized is not None:

                # level by level, so a label repeated deeper in the tree (a
                # second "Do not launch") does not hide the first-level option
                queue = [cell_tree.normalized]
                while queue:
                    n = queue.pop(0)
                    if n.payoff is not None and n.label not in node_payoffs:
                        node_payoffs[n.label] = float(n.payoff)
                    queue.extend(n.children)
            cells.append(
                SensitivityCell(
                    inputs=inputs,
                    payoff=cell_tree.payoff if cell_tree.is_solved else None,
                    chosen_labels=chosen,
                    errors=list(cell_tree.errors),
                    warnings=list(cell_tree.warnings),
                    decision_payoffs=node_payoffs,
                )
            )

        return SensitivityResult(
            specs=parsed,
            cells=cells,
            opt=self.opt,
            base_payoff=self.payoff,
            base_chosen=base_chosen,
            base_options=[c.label for c in self.normalized.children] if self.normalized else [],
        )

    def tornado(self, specs: dict[str, Any] | list[Any]) -> TornadoResult:
        """Evaluate several variables at once, one at a time (a tornado chart).

        Each variable is set to its low and its high value while the others
        stay at their base values; the result ranks the variables by how far
        the EMV of the tree moves.

        Parameters
        ----------
        specs : dict or list
            ``{"P(high)": (0.3, 0.5), "margin": {"low": 40, "high": 60}}`` or
            ``[{"name": "P(high)", "low": 0.3, "high": 0.5}, ...]``.

        Returns
        -------
        TornadoResult
            ``to_frame()`` for the table, ``plot()`` for the chart.
        """
        if isinstance(specs, dict):
            items = []
            for name, body in specs.items():
                if isinstance(body, dict):
                    items.append({"name": name, "low": body.get("low"), "high": body.get("high")})
                else:
                    low, high = body
                    items.append({"name": name, "low": low, "high": high})
        else:
            items = [dict(x) for x in specs]
        if not items:
            raise ValueError("tornado requires at least one variable")
        rows: list[dict[str, Any]] = []
        for item in items:
            name = str(item["name"])
            low, high = float(item["low"]), float(item["high"])
            sweep = self.sensitivity({name: {"values": [low, high]}})
            row: dict[str, Any] = {"name": name, "base_value": self.vars.get(name)}
            for side, value in (("low", low), ("high", high)):
                cell = next((c for c in sweep.cells if c.inputs.get(name) == value), None)
                row[side] = {
                    "value": value,
                    "payoff": cell.payoff if cell else None,
                    "chosen_labels": list(cell.chosen_labels) if cell else [],
                    "error_codes": [d.code for d in (cell.errors if cell else [])],
                }
            lo, hi = row["low"]["payoff"], row["high"]["payoff"]
            row["swing"] = abs(hi - lo) if lo is not None and hi is not None else None
            rows.append(row)
        rows.sort(key=lambda r: -1.0 if r["swing"] is None else r["swing"], reverse=True)
        base_chosen = []
        if self.normalized is not None:
            base_chosen = [
                c.label for c in self.normalized.children if c.id in self.normalized.chosen_child_ids
            ]
        return TornadoResult(base_payoff=self.payoff, base_chosen=base_chosen, rows=rows)

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
            "tree_uses": [dict(u) for u in self.tree_uses],
            "python_code": self.python_code,
            "payoff": self.payoff,
            "is_solved": self.is_solved,
        }


__all__ = [
    "dtree_parser",
    "DecisionNode",
    "Diagnostic",
    "SensitivityCell",
    "SensitivityResult",
    "SensitivitySpec",
    "TornadoResult",
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
