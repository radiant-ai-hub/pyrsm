"""Tests for ``subtree("name")`` references in pyrsm.model.dtree."""

from __future__ import annotations

import pytest

from pyrsm.model.dtree import (
    UnknownSubtreeError,
    UnsafeExpressionError,
    dtree,
    safe_eval,
)

LAWSUIT_YAML = """
name: lawsuit
type: chance
settle:
    p: 0.7
    payoff: -200000
win:
    p: 0.2
    payoff: 0
lose:
    p: 0.1
    payoff: -1500000
"""


def _solved_lawsuit() -> dtree:
    return dtree(LAWSUIT_YAML, opt="max")


# ---------------------------------------------------------------------------
# safe_eval: subtree() is only allowed with a resolver
# ---------------------------------------------------------------------------


class TestSafeEvalSubtree:
    def test_resolver_supplies_value(self):
        seen: list[str] = []

        def resolver(name: str) -> float:
            seen.append(name)
            return 42.0

        assert safe_eval("subtree('foo') + 1", {}, subtree_resolver=resolver) == 43.0
        assert seen == ["foo"]

    def test_resolver_can_raise_unknown_subtree(self):
        def resolver(name: str) -> float:
            raise UnknownSubtreeError(name)

        with pytest.raises(UnknownSubtreeError) as exc:
            safe_eval("subtree('missing')", {}, subtree_resolver=resolver)
        assert exc.value.name == "missing"

    def test_subtree_call_rejected_without_resolver(self):
        # Without a resolver, function calls remain disallowed entirely.
        with pytest.raises(UnsafeExpressionError):
            safe_eval("subtree('foo')", {})

    def test_other_function_calls_still_rejected(self):
        # Make sure adding subtree() didn't open a door for arbitrary calls.
        def resolver(name: str) -> float:
            return 1.0

        with pytest.raises(UnsafeExpressionError):
            safe_eval("abs(-1)", {}, subtree_resolver=resolver)
        with pytest.raises(UnsafeExpressionError):
            safe_eval("__import__('os')", {}, subtree_resolver=resolver)
        with pytest.raises(UnsafeExpressionError):
            safe_eval("subtree(x)", {"x": 1}, subtree_resolver=resolver)  # not a literal
        with pytest.raises(UnsafeExpressionError):
            safe_eval("subtree('a', 'b')", {}, subtree_resolver=resolver)
        with pytest.raises(UnsafeExpressionError):
            safe_eval("subtree(name='a')", {}, subtree_resolver=resolver)


# ---------------------------------------------------------------------------
# dtree: subtree() in variables and node fields
# ---------------------------------------------------------------------------


PARENT_YAML = """
name: launch
variables:
    lawsuit_emv: subtree("lawsuit")
    revenue: 1000000
type: decision
launch:
    payoff: revenue + lawsuit_emv
hold:
    payoff: 0
"""


class TestSubtreeInDtree:
    def test_parent_uses_child_payoff(self):
        """parent.launch = revenue + lawsuit_emv = 1M + (-290k) = 710k."""
        lawsuit = _solved_lawsuit()
        # lawsuit EMV: 0.7*(-200k) + 0.2*0 + 0.1*(-1.5M) = -290k.
        assert lawsuit.payoff == pytest.approx(-290_000.0)

        parent = dtree(PARENT_YAML, opt="max", subtree_library={"lawsuit": lawsuit})
        assert parent.errors == []
        assert parent.vars["lawsuit_emv"] == pytest.approx(-290_000.0)
        assert parent.payoff == pytest.approx(710_000.0)
        # The chosen branch flips to "launch" because lawsuit_emv is small.
        root = parent.normalized
        chosen = [c.label for c in root.children if c.id in root.chosen_child_ids]
        assert chosen == ["launch"]

    def test_subtree_refs_tracked(self):
        """parent.subtree_refs records which library entries were consumed."""
        lawsuit = _solved_lawsuit()
        parent = dtree(PARENT_YAML, opt="max", subtree_library={"lawsuit": lawsuit})
        assert parent.subtree_refs == {"lawsuit": pytest.approx(-290_000.0)}

    def test_missing_subtree_yields_unknown_subtree_diagnostic(self):
        """No library at all → unknown_subtree, not unsafe_expression."""
        parent = dtree(PARENT_YAML, opt="max")
        codes = [d.code for d in parent.errors]
        assert "unknown_subtree" in codes
        # The variable couldn't resolve, so dependent payoff expressions
        # surface unresolved_variable downstream — that's fine, but the
        # primary diagnostic must be the subtree one for the variable.
        primary = next(d for d in parent.errors if d.code == "unknown_subtree")
        assert "lawsuit" in primary.message

    def test_missing_subtree_with_partial_library(self):
        """Library exists but is missing the referenced name."""
        parent = dtree(
            PARENT_YAML,
            opt="max",
            subtree_library={"other_tree": _solved_lawsuit()},
        )
        codes = [d.code for d in parent.errors]
        assert "unknown_subtree" in codes

    def test_unsolved_subtree_treated_as_unknown(self):
        """A library entry whose .payoff is None can't satisfy a reference."""
        bad_yaml = """
        name: bad
        type: chance
        x:
            p: 0.5
            payoff: 100
        y:
            p: 0.7
            payoff: 200
        """
        unsolved = dtree(bad_yaml)
        assert unsolved.payoff is None  # probs > 1, so unsolved
        parent = dtree(
            PARENT_YAML,
            opt="max",
            subtree_library={"lawsuit": unsolved},
        )
        assert any(d.code == "unknown_subtree" for d in parent.errors)

    def test_subtree_works_in_node_payoff_directly(self):
        """subtree() can appear directly in a node payoff expression."""
        lawsuit = _solved_lawsuit()
        yaml_text = """
        name: shorthand
        type: decision
        launch:
            payoff: subtree("lawsuit") + 1000000
        hold:
            payoff: 0
        """
        parent = dtree(yaml_text, opt="max", subtree_library={"lawsuit": lawsuit})
        assert parent.errors == []
        assert parent.payoff == pytest.approx(710_000.0)

    def test_chain_three_trees(self):
        """A→B→C: parent (A) refs B, B refs C. Consumer solves C first."""
        c_yaml = """
        name: c
        type: chance
        win:
            p: 0.5
            payoff: 100
        lose:
            p: 0.5
            payoff: -50
        """
        c = dtree(c_yaml, opt="max")  # EMV = 25

        b_yaml = """
        name: b
        variables:
            c_value: subtree("c")
        type: decision
        take:
            payoff: c_value + 200
        skip:
            payoff: 100
        """
        b = dtree(b_yaml, opt="max", subtree_library={"c": c})
        # take: 25 + 200 = 225; skip: 100; max -> 225
        assert b.payoff == pytest.approx(225.0)

        a_yaml = """
        name: a
        variables:
            b_value: subtree("b")
        type: decision
        go:
            payoff: b_value
        wait:
            payoff: 300
        """
        a = dtree(a_yaml, opt="max", subtree_library={"b": b, "c": c})
        # go: 225; wait: 300; max -> 300
        assert a.payoff == pytest.approx(300.0)
        # a only consumed the "b" subtree even though c was in the library.
        assert a.subtree_refs == {"b": pytest.approx(225.0)}

    def test_sensitivity_propagates_through_subtree(self):
        """Sweeping a variable in the parent recomputes its payoff; the
        subtree's payoff stays fixed (it was solved once at library build
        time, which is exactly what the consumer's snapshot semantics
        promise)."""
        lawsuit = _solved_lawsuit()
        # PARENT_YAML's `revenue` is a parent-side variable; the sweep
        # must show different launch payoffs but the lawsuit EMV stays at
        # -290k throughout.
        parent = dtree(PARENT_YAML, opt="max", subtree_library={"lawsuit": lawsuit})
        res = parent.sensitivity({"revenue": [500_000, 1_000_000, 2_000_000]})
        assert all(c.is_solved for c in res.cells)
        payoffs = sorted(c.payoff for c in res.cells)
        assert payoffs[0] == pytest.approx(210_000.0)  # 500k - 290k
        assert payoffs[1] == pytest.approx(710_000.0)
        assert payoffs[2] == pytest.approx(1_710_000.0)
