"""``tree: name`` branches: another tree grafted in place of a branch."""

import importlib

import pytest

from pyrsm.decide.dtree import dtree, dtree_parser

examples = importlib.import_module("pyrsm.decide.dtree_examples")


@pytest.fixture
def launch_trees():
    return examples.dtree_example("product-launch")["trees"]


def _find(node, *labels):
    for label in labels:
        node = next(c for c in node.children if c.label == label)
    return node


def test_product_launch_solves_as_one_tree(launch_trees):
    tree = dtree(launch_trees["product launch"], subtree_library=launch_trees)
    assert tree.is_solved, tree.diagnostics
    assert tree.payoff == pytest.approx(1_380_000)
    root = tree.normalized
    assert [c.label for c in root.children if c.id in root.chosen_child_ids] == [
        "Test market first"
    ]
    assert _find(root, "Launch now").payoff == pytest.approx(600_000)
    positive = _find(root, "Test market first", "Positive result", "Launch")
    assert positive.payoff == pytest.approx(0.85 * 0.4 / 0.46 * 9e6 + 1e6 - 4e6)
    negative = _find(root, "Test market first", "Negative result", "Launch")
    assert negative.payoff == pytest.approx(-2_000_000)
    assert set(tree.used_trees) == {"product launch", "launch", "test market"}


def test_each_part_solves_on_its_own(launch_trees):
    launch = dtree(launch_trees["launch"], subtree_library=launch_trees)
    assert launch.payoff == pytest.approx(600_000)
    # without the test cost, which the main tree puts on its branch
    market = dtree(launch_trees["test market"], subtree_library=launch_trees)
    assert market.payoff == pytest.approx(1_680_000)


def test_grafted_branch_keeps_label_p_and_adds_cost(launch_trees):
    tree = dtree(launch_trees["product launch"], subtree_library=launch_trees)
    market = _find(tree.normalized, "Test market first")
    assert market.kind == "chance"
    assert market.source_tree == "test market"
    assert market.cost == pytest.approx(300_000)
    assert market.raw_cost == "test cost"
    launch = _find(market, "Positive result", "Launch")
    assert launch.source_tree == "launch"
    assert launch.tree_with == {"P(high)": pytest.approx(0.34 / 0.46)}
    assert launch.cost == pytest.approx(4_000_000)


def test_tree_uses_lists_every_use_in_order(launch_trees):
    tree = dtree(launch_trees["product launch"], subtree_library=launch_trees)
    uses = [(u["tree"], u["at"][-1], round(u["emv"])) for u in tree.tree_uses]
    assert uses == [
        ("launch", "Launch now", 600_000),
        ("test market", "Test market first", 1_380_000),
        ("launch", "Launch", 3_652_174),
        ("launch", "Launch", -2_000_000),
    ]
    state = tree.to_state()
    assert state["tree_uses"][2]["with"] == {"P(high)": pytest.approx(0.7391304)}


def test_calling_tree_variables_replace_the_sub_tree_defaults():
    lib = {
        "deal": "name: Deal\nvariables:\n    price: 10\n    p win: 0.5\ntype: chance\n"
        "Win:\n    p: p win\n    payoff: price\nLose:\n    p: 1 - p win\n    payoff: 0\n",
    }
    main = (
        "name: Main\nvariables:\n    price: 100\ntype: decision\n"
        "Bid:\n    tree: deal\nBid high:\n    tree: deal\n    with:\n        price: 2 * price\n"
        "        p win: 0.25\nPass:\n    payoff: 0\n"
    )
    tree = dtree(main, subtree_library=lib)
    assert tree.is_solved, tree.diagnostics
    assert _find(tree.normalized, "Bid").payoff == pytest.approx(50)
    assert _find(tree.normalized, "Bid high").payoff == pytest.approx(50)
    # sensitivity on the main tree's variable flows into both uses
    sens = tree.sensitivity({"price": {"values": [200]}})
    assert sens.cells[0].payoff == pytest.approx(100)


def test_sensitivity_options_are_first_level_labels(launch_trees):
    tree = dtree(launch_trees["product launch"], subtree_library=launch_trees)
    sens = tree.sensitivity({"P(high)": {"values": [0.1, 0.4, 0.7]}})
    chosen = [c.chosen_labels for c in sens.cells]
    assert chosen == [["Do not launch"], ["Test market first"], ["Launch now"]]
    # "Do not launch" also appears under each test result; the first-level
    # option is the one reported
    assert all(c.decision_payoffs["Do not launch"] == 0 for c in sens.cells)
    assert sens.cells[1].decision_payoffs["Launch now"] == pytest.approx(600_000)


def test_tornado_ranks_variables_by_swing(launch_trees):
    tree = dtree(launch_trees["product launch"], subtree_library=launch_trees)
    preset = examples.dtree_example("product-launch")["tornado"]["vars"]
    tor = tree.tornado(preset)
    frame = tor.to_frame()
    assert list(frame["variable"]) == ["margin", "P(high)", "P(pos|high)", "fixed cost", "test cost"]
    assert frame["swing"].iloc[0] == pytest.approx(1_408_000)
    assert tor.base_chosen == ["Test market first"]
    # tuple form
    assert tree.tornado({"test cost": (200_000, 400_000)}).rows[0]["swing"] == pytest.approx(
        200_000
    )


@pytest.mark.parametrize(
    ("branch", "code"),
    [
        ("    tree: nowhere\n", "unknown_tree"),
        ("    tree: deal\n    payoff: 5\n", "tree_branch_has_payoff"),
        ("    tree: deal\n    Extra:\n        payoff: 1\n", "tree_branch_has_children"),
        ("    tree: deal\n    with:\n        unknown var: 1\n", "unknown_with_variable"),
        ("    tree: deal\n    with: 5\n", "with_not_mapping"),
        ("    payoff: 1\n    with:\n        price: 1\n", "with_without_tree"),
    ],
)
def test_graft_input_errors_explain_the_fix(branch, code):
    lib = {"deal": "name: Deal\nvariables:\n    price: 10\ntype: decision\nTake:\n    payoff: price\n"}
    main = "name: Main\ntype: decision\nUse it:\n" + branch + "Pass:\n    payoff: 0\n"
    tree = dtree(main, subtree_library=lib)
    assert code in [d.code for d in tree.errors]
    assert not tree.is_solved


def test_a_tree_cannot_use_itself():
    lib = {
        "a": "name: A\ntype: decision\nGo:\n    tree: b\nStop:\n    payoff: 0\n",
        "b": "name: B\ntype: decision\nAgain:\n    tree: a\nStop:\n    payoff: 1\n",
    }
    tree = dtree(lib["a"], subtree_library=lib)
    errors = [d for d in tree.errors if d.code == "tree_cycle"]
    assert errors and "b -> a -> b" in errors[0].message


def test_mermaid_boxes_and_hover_text(launch_trees):
    tree = dtree(launch_trees["product launch"], subtree_library=launch_trees)
    final = tree.to_mermaid(final=True)
    assert 'subgraph tree1 ["launch"]' in final
    assert '["launch · P(high) = 0.7391"]' in final
    lines = [ln.strip() for ln in final.splitlines()]
    root_tip = next(ln for ln in lines if ln.startswith("click n1 "))
    assert "<b>Choose</b> Test market first (highest value)" in root_tip
    launch_tip = next(ln for ln in lines if ln.startswith("click n2 "))
    assert (
        "<b>EMV</b> 0.4 × $10,000,000.00 + 0.6 × $1,000,000.00 − $4,000,000.00 = $600,000.00"
        in launch_tip
    )
    leaf_tip = next(ln for ln in lines if ln.startswith("click n3 "))
    assert "<b>Payoff</b> margin * sales high = $10,000,000.00" in leaf_tip
    # the initial tree shows inputs, not results
    initial = tree.to_mermaid(final=False)
    assert "<b>EMV</b>" not in initial and "<b>p</b> P(high) = 0.4" in initial
    bare = tree.to_mermaid(final=True, tooltips=False, subgraphs=False)
    assert "click" not in bare and "subgraph" not in bare


def test_summary_lists_the_trees_used(launch_trees, capsys):
    tree = dtree(launch_trees["product launch"], subtree_library=launch_trees)
    tree.summary(input=False, output=True)
    out = capsys.readouterr().out
    assert "Trees used:" in out
    assert "Test market first > Positive result > Launch uses 'launch' with P(high) = 0.7391" in out


def test_examples_pass_the_input_checks(launch_trees):
    for text in launch_trees.values():
        assert dtree_parser(text)[1] == []
    assert all("video" not in e for e in examples.dtree_examples())


def test_forward_references_only_in_grafted_variables():
    # a tree's own variables still resolve in order, as in radiant.model
    tree = dtree("name: T\nvariables:\n    a: b + 1\n    b: 1\ntype: decision\nX:\n    payoff: a\n")
    assert "unresolved_variable" in [d.code for d in tree.errors]
