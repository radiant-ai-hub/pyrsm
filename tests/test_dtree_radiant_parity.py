"""Decision analysis as radiant.model: the tutorial examples, combining trees,
the input checks, and the text output.

References in tests/reference/dtree are radiant.model's ``summary(dtree(...),
input = FALSE, output = TRUE)`` for the example trees in
pyrsm/decide/dtree_examples.
"""

from pathlib import Path

import pytest

import pyrsm as rsm
from pyrsm.decide.dtree import dtree, dtree_parser

REF = Path(__file__).parent / "reference" / "dtree"


# radiant.model's EMV for every tree of every example
EXPECTED = {
    "open-store": {"open store": 600000},
    "sign-contract": {"sign contract": 955000, "fees 10000": 950000, "medium 800000": 900000},
    "sensitivity": {"fees 5000": 955000, "fees 8000": 952000, "fees variable": 955000},
    "imperfect-information": {"full": 820000, "parts": 820000, "popup": 820000},
    "subtrees": {"main_tree": 955000, "sub_tree": 955000},
}


@pytest.mark.parametrize("example_id", list(EXPECTED))
def test_examples_match_radiant(example_id):
    ex = rsm.decide.dtree_example(example_id)
    for name, emv in EXPECTED[example_id].items():
        tree = dtree(ex["trees"][name], subtree_library=ex["trees"])
        assert tree.errors == [], (name, tree.errors)
        assert tree.payoff == pytest.approx(emv), name


@pytest.mark.parametrize(
    "file",
    [
        "open_store",
        "sign_contract",
        "sign_contract_fees_10000",
        "sign_contract_medium_800000",
        "sign_contract_fees_5000",
        "sign_contract_fees_8000",
        "popup_full",
    ],
)
def test_summary_text_is_radiant_output(file, capsys):
    import importlib

    examples_pkg = importlib.import_module("pyrsm.decide.dtree_examples")

    text = (Path(examples_pkg.__file__).parent / f"{file}.yaml").read_text()
    dtree(text).summary(input=False, output=True)
    out = capsys.readouterr().out.strip("\n")
    assert out == (REF / f"{file}.txt").read_text().strip("\n")


def test_subtree_uses_the_main_trees_variables():
    """dtree("name") re-evaluates with the calling tree's variables."""
    ex = rsm.decide.dtree_example("subtrees")
    lib = dict(ex["trees"])
    lib["main_tree"] = lib["main_tree"].replace("legal fees: 5000", "legal fees: 80000")
    tree = dtree(lib["main_tree"], subtree_library=lib)
    assert tree.subtree_refs["sub_tree"] == pytest.approx(880000)
    assert tree.payoff == pytest.approx(900000)
    assert tree.used_trees == ["sub_tree", "main_tree"]


def test_sensitivity_flows_through_a_subtree():
    ex = rsm.decide.dtree_example("subtrees")
    tree = dtree(ex["trees"]["main_tree"], subtree_library=ex["trees"])
    res = tree.sensitivity({"legal fees": {"min": 0, "max": 100000, "step": 50000}})
    assert [c.payoff for c in res.cells] == [960000, 910000, 900000]
    assert res.cells[-1].chosen_labels == ["Sign with TV Network"]


def test_variables_imported_from_another_tree():
    ex = rsm.decide.dtree_example("imperfect-information")
    popup = dtree(ex["trees"]["popup"], subtree_library=ex["trees"])
    assert popup.variables_from == "parts"
    assert popup.vars["P(S|+)"] == pytest.approx(0.48 / 0.52)
    assert "payoff_test" not in popup.vars  # dtree() references are not imported


def test_reference_cycles_are_reported():
    lib = {
        "a": "name: a\nvariables:\n    x: dtree('b')\ntype: decision\ngo:\n    payoff: x\nstop:\n    payoff: 0\n",
        "b": "name: b\nvariables:\n    y: dtree('a')\ntype: decision\ngo:\n    payoff: y\nstop:\n    payoff: 0\n",
    }
    tree = dtree(lib["a"], subtree_library=lib)
    assert tree.payoff is None
    assert any("refers back to itself" in d.message for d in tree.errors)


def test_unknown_tree_reference():
    tree = dtree("name: a\nvariables:\n    x: dtree('nope')\ntype: decision\ngo:\n    payoff: x\nstop:\n    payoff: 0\n")
    assert [d.code for d in tree.errors][:1] == ["unknown_subtree"]


def test_input_checks_match_radiant():
    """radiant.model's messages for the debugging example (video #5)."""
    text = rsm.decide.dtree_example("debug")["trees"]["sign contract with errors"]
    _, problems = dtree_parser(text)
    assert problems[0] == "Each line in the tree input must have a ':'. Add a ':' in line(s): 13"
    assert problems[1].startswith("Indent issue in line(s): 5\nUse the tab key")
    assert problems[2].startswith("Indent issue in line(s): 4, 9\nAlways use the tab key")


def test_parser_cleans_like_radiant():
    cleaned, problems = dtree_parser("name: t\ntype: decision\na:\n\tPayoff:5\nb:\n\tpayoff: 1")
    assert problems == []
    assert "    payoff: 5" in cleaned
    assert dtree("name: t\ntype: chance\na:\n\tp: .4\n\tpayoff: 1\nb:\n\tp: .6\n\tpayoff: 2").payoff == pytest.approx(1.6)


def test_python_code_includes_the_referenced_trees():
    ex = rsm.decide.dtree_example("imperfect-information")
    tree = dtree(ex["trees"]["parts"], subtree_library=ex["trees"])
    code = tree.python_code
    assert "trees = {" in code and "'popup':" in code and "subtree_library=trees" in code
    ns: dict = {}
    exec(code.replace("tree.summary(input=True, output=True)", "").replace("print(tree.to_mermaid(final=True))", ""), ns)
    assert ns["tree"].payoff == pytest.approx(820000)


def test_sensitivity_decisions_and_plot(tmp_path):
    """radiant.model's sensitivity chart: each decision's EMV across the sweep."""
    ex = rsm.decide.dtree_example("sensitivity")
    tree = dtree(ex["trees"]["fees variable"])
    res = tree.sensitivity(ex["sensitivity"]["vars"])
    movie = [c.decision_payoffs["Sign with Movie Company"] for c in res.cells]
    tv = [c.decision_payoffs["Sign with TV Network"] for c in res.cells]
    assert movie[0] == pytest.approx(960000) and movie[-1] == pytest.approx(860000)
    assert set(tv) == {900000}
    assert res.to_dict()["base_options"] == ["Sign with Movie Company", "Sign with TV Network"]
    res.plot().save(tmp_path / "sens.png", dpi=40, verbose=False)
