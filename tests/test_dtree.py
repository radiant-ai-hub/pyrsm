"""Tests for pyrsm.model.dtree."""

from __future__ import annotations

import pytest

from pyrsm.model.dtree import (
    UnresolvedVariableError,
    UnsafeExpressionError,
    dtree,
    safe_eval,
)

MOVIE_YAML = """
name: Sign contract
variables:
    legal fees: 5000
type: decision
Sign with Movie Company:
    cost: legal fees
    type: chance
    Small Box Office:
        p: 0.3
        payoff: 200000
    Medium Box Office:
        p: 0.6
        payoff: 1000000
    Large Box Office:
        p: 0.1
        payoff: 3000000
Sign with TV Network:
    payoff: 900000
"""


def test_movie_contract_max():
    tree = dtree(MOVIE_YAML, opt="max")
    assert tree.errors == []
    # Movie EV: 0.3*200k + 0.6*1M + 0.1*3M = 960k, minus 5k cost = 955k.
    assert tree.payoff == pytest.approx(955_000.0)
    # Root chose the Movie Company branch.
    root = tree.normalized
    assert root is not None
    assert root.kind == "decision"
    chosen_labels = {c.label for c in root.children if c.id in root.chosen_child_ids}
    assert chosen_labels == {"Sign with Movie Company"}


def test_movie_contract_min():
    tree = dtree(MOVIE_YAML, opt="min")
    assert tree.errors == []
    # Min of (955k, 900k) = 900k → TV Network.
    assert tree.payoff == pytest.approx(900_000.0)
    root = tree.normalized
    chosen = [c.label for c in root.children if c.id in root.chosen_child_ids]
    assert chosen == ["Sign with TV Network"]


def test_probabilities_must_sum_to_one():
    yaml_text = """
    name: bad probs
    type: chance
    A:
        p: 0.3
        payoff: 100
    B:
        p: 0.6
        payoff: 200
    """
    tree = dtree(yaml_text)
    codes = [d.code for d in tree.diagnostics]
    assert "chance_probabilities_not_one" in codes
    assert not tree.is_solved


def test_terminal_missing_payoff():
    yaml_text = """
    name: missing
    type: decision
    Option A:
        payoff: 100
    Option B: {}
    """
    tree = dtree(yaml_text)
    codes = [d.code for d in tree.diagnostics]
    assert "terminal_missing_payoff" in codes


def test_unresolved_variable_reports_name():
    yaml_text = """
    name: unresolved
    variables:
        a: 10
    type: decision
    A:
        payoff: a + missing_var
    B:
        payoff: 100
    """
    tree = dtree(yaml_text)
    msgs = [d.message for d in tree.errors if d.code == "unresolved_variable"]
    assert any("missing_var" in m for m in msgs)


def test_safe_arithmetic_resolves():
    yaml_text = """
    name: arithmetic
    variables:
        base: 100
        scale: 2
        total: base * scale + 50
    type: decision
    A:
        payoff: total
    B:
        payoff: 200
    """
    tree = dtree(yaml_text)
    assert tree.errors == []
    assert tree.vars["total"] == pytest.approx(250.0)
    assert tree.payoff == pytest.approx(250.0)


def test_unsafe_expression_rejected():
    with pytest.raises(UnsafeExpressionError):
        safe_eval("__import__('os')", {})
    with pytest.raises(UnsafeExpressionError):
        safe_eval("abs(-1)", {})
    with pytest.raises(UnsafeExpressionError):
        safe_eval("1 if True else 2", {})


def test_costs_subtract():
    yaml_text = """
    name: cost test
    type: decision
    A:
        cost: 100
        type: chance
        A1:
            p: 1.0
            payoff: 1000
    B:
        payoff: 800
    """
    tree = dtree(yaml_text)
    assert tree.errors == []
    # A: 1.0 * 1000 - 100 = 900
    # decision max(900, 800) = 900
    assert tree.payoff == pytest.approx(900.0)


def test_decision_ties_preserve_branches():
    yaml_text = """
    name: tie
    type: decision
    Option A:
        payoff: 100
    Option B:
        payoff: 100
    """
    tree = dtree(yaml_text)
    assert tree.payoff == pytest.approx(100.0)
    root = tree.normalized
    assert len(root.chosen_child_ids) == 2
    tie_warnings = [w for w in tree.warnings if w.code == "decision_tie"]
    assert tie_warnings


def test_mermaid_final_marks_chosen_with_double_equals():
    tree = dtree(MOVIE_YAML)
    final = tree.to_mermaid(final=True)
    initial = tree.to_mermaid(final=False)
    # The chosen branch is highlighted with `===` in the final tree only.
    assert " === " in final
    assert " === " not in initial


def test_to_yaml_roundtrips_payoff():
    tree = dtree(MOVIE_YAML)
    serialized = tree.to_yaml()
    re_tree = dtree(serialized)
    assert re_tree.errors == []
    assert re_tree.payoff == pytest.approx(tree.payoff)


def test_summary_runs_without_crashing(capsys):
    tree = dtree(MOVIE_YAML)
    tree.summary(input=True, output=True)
    captured = capsys.readouterr()
    assert "Sign contract" in captured.out
    assert "Final decision tree" in captured.out


def test_invalid_yaml_returns_diagnostic_not_exception():
    yaml_text = "name: bad\n  bad indentation: oops:"
    tree = dtree(yaml_text)
    codes = [d.code for d in tree.diagnostics]
    assert any(c in codes for c in ("yaml_parse_error", "root_not_mapping"))
    assert not tree.is_solved


def test_probability_out_of_range():
    yaml_text = """
    name: bad p
    type: chance
    A:
        p: 1.5
        payoff: 100
    B:
        p: -0.5
        payoff: 200
    """
    tree = dtree(yaml_text)
    codes = {d.code for d in tree.errors}
    assert "chance_probability_out_of_range" in codes


def test_state_roundtrip_includes_solution():
    tree = dtree(MOVIE_YAML)
    state = tree.to_state()
    assert state["is_solved"] is True
    assert state["payoff"] == pytest.approx(955_000.0)
    assert state["solution_table"][0]["label"] == "Sign contract"
    assert "graph LR" in state["mermaid_final"]
    assert "graph LR" in state["mermaid_initial"]


def test_safe_eval_basic():
    assert safe_eval("1 + 2 * 3", {}) == 7.0
    assert safe_eval("(1 + 2) * 3", {}) == 9.0
    assert safe_eval("-x", {"x": 5}) == -5.0
    assert safe_eval("2 ** 3", {}) == 8.0
    with pytest.raises(UnresolvedVariableError):
        safe_eval("x + 1", {})


def test_variables_with_dependencies():
    yaml_text = """
    name: depvars
    variables:
        p_small: 0.3
        p_medium: 0.6
        p_large: 1 - p_small - p_medium
    type: chance
    Small:
        p: p_small
        payoff: 100
    Medium:
        p: p_medium
        payoff: 200
    Large:
        p: p_large
        payoff: 300
    """
    tree = dtree(yaml_text)
    assert tree.errors == []
    assert tree.vars["p_large"] == pytest.approx(0.1)
    # EV = 0.3*100 + 0.6*200 + 0.1*300 = 180
    assert tree.payoff == pytest.approx(180.0)


def test_solution_df_columns():
    tree = dtree(MOVIE_YAML)
    df = tree.solution_df
    for col in ["level", "label", "type", "p", "payoff", "cost", "id"]:
        assert col in df.columns
