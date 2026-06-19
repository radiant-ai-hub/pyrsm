"""Golden-dataset tests against the MGT 403 decision-tree tutorial YAMLs.

These pin the expected behavior of ``pyrsm.model.dtree`` on the inputs
under ``rsm-mgt403/tutorial-decision-analysis/decision_tree/input/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyrsm.model.dtree import dtree

TUTORIAL_DIR = Path(
    "/home/vnijs/gh/rsm-teaching/rsm-mgt403/tutorial-decision-analysis/decision_tree/input"
)


def _load(name: str) -> str:
    return (TUTORIAL_DIR / name).read_text()


@pytest.mark.skipif(not TUTORIAL_DIR.is_dir(), reason="tutorial YAMLs not checked out")
def test_video2_open_a_new_store_solves_to_600k():
    """Decision-chance tree; expected EMV = 0.6*5M + 0.4*(-1M) - 2M = 600k."""
    tree = dtree(_load("video2_input.yaml"), opt="max")
    assert tree.errors == []
    assert tree.payoff == pytest.approx(600_000.0)
    root = tree.normalized
    chosen = [c.label for c in root.children if c.id in root.chosen_child_ids]
    assert chosen == ["open"]


@pytest.mark.skipif(not TUTORIAL_DIR.is_dir(), reason="tutorial YAMLs not checked out")
def test_video3_movie_contract_with_legal_fees_chooses_tv():
    """Movie EV = 0.3*200k + 0.6*800k + 0.1*3M - 5k = 835k; TV = 900k → TV wins."""
    tree = dtree(_load("video3_input.yaml"), opt="max")
    assert tree.errors == []
    assert tree.vars["legal fees"] == pytest.approx(5_000.0)
    assert tree.payoff == pytest.approx(900_000.0)
    root = tree.normalized
    chosen = [c.label for c in root.children if c.id in root.chosen_child_ids]
    assert chosen == ["Sign with TV Network"]


@pytest.mark.skipif(not TUTORIAL_DIR.is_dir(), reason="tutorial YAMLs not checked out")
def test_video7_popup_test_bayes_variables_resolve():
    """Bayes-rule variables must resolve correctly despite name collisions.

    P(+|S), P(S|+), P(-|S), P(S|-) all reduce to the safe Python name
    ``P_S`` after stripping non-word characters. Without dedup they
    overwrite each other and Bayes math silently collapses.
    """
    tree = dtree(_load("video7_input.yaml"), opt="max")
    assert tree.errors == []
    v = tree.vars
    assert v["P(S)"] == pytest.approx(0.6)
    assert v["P(F)"] == pytest.approx(0.4)
    assert v["P(+|S)"] == pytest.approx(0.8)
    assert v["P(-|S)"] == pytest.approx(0.2)
    assert v["P(-|F)"] == pytest.approx(0.9)
    assert v["P(+|F)"] == pytest.approx(0.1)
    # Marginal probabilities of the test result.
    assert v["P(+)"] == pytest.approx(0.52)
    assert v["P(-)"] == pytest.approx(0.48)
    # Bayes posteriors.
    assert v["P(S|+)"] == pytest.approx(0.8 * 0.6 / 0.52)
    assert v["P(F|+)"] == pytest.approx(1.0 - 0.8 * 0.6 / 0.52)
    assert v["P(F|-)"] == pytest.approx(0.9 * 0.4 / 0.48)
    assert v["P(S|-)"] == pytest.approx(1.0 - 0.9 * 0.4 / 0.48)


@pytest.mark.skipif(not TUTORIAL_DIR.is_dir(), reason="tutorial YAMLs not checked out")
def test_video7_popup_test_chooses_popup():
    """With a $500k information cost, the popup test still beats blind decisions."""
    tree = dtree(_load("video7_input.yaml"), opt="max")
    assert tree.errors == []
    # popup test EMV ≈ 0.52 * 2.538M + 0.48 * 0 - 0.5M = 820k.
    assert tree.payoff == pytest.approx(820_000.0, rel=1e-6)
    root = tree.normalized
    chosen = [c.label for c in root.children if c.id in root.chosen_child_ids]
    assert chosen == ["popup test"]


@pytest.mark.skipif(not TUTORIAL_DIR.is_dir(), reason="tutorial YAMLs not checked out")
def test_video5_malformed_input_returns_diagnostics_not_exception():
    """video5_input.yaml is intentionally broken; loader must not crash."""
    tree = dtree(_load("video5_input.yaml"), opt="max")
    assert tree.errors, "expected at least one error diagnostic"
    assert not tree.is_solved
