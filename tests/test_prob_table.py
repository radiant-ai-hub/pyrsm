"""Probability tables (prior, conditional, joint, posterior) for a test that
is not perfect."""

import pytest

import pyrsm as rsm

POPUP = dict(
    prior={"S": 0.6, "F": 0.4},
    likelihood={"S": {"Pos": 0.8, "Neg": 0.2}, "F": {"Pos": 0.1, "Neg": 0.9}},
)


def test_popup_tables():
    pt = rsm.decide.prob_table(**POPUP, state="store", signal="popup")
    v = pt.values
    assert v["P(Pos and S)"] == pytest.approx(0.48)
    assert v["P(Neg and F)"] == pytest.approx(0.36)
    assert v["P(Pos)"] == pytest.approx(0.52)
    assert v["P(Neg)"] == pytest.approx(0.48)
    assert v["P(S|Pos)"] == pytest.approx(0.9231, abs=1e-4)
    assert v["P(S|Neg)"] == pytest.approx(0.25)
    # the joint table's Total row is P(signal); its Total column is the prior
    joint = pt.joint
    assert joint.row(-1, named=True) == pytest.approx({"store": "Total", "Pos": 0.52, "Neg": 0.48, "Total": 1.0})
    assert joint["Total"].to_list()[:2] == pytest.approx([0.6, 0.4])
    # each posterior column adds up to 1
    post = pt.posterior
    assert post["Pos"][:2].sum() == pytest.approx(1.0)
    assert post["Neg"][:2].sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "prior, likelihood, key, expected",
    [
        # popup store (example 3/4) with + and - as signal names
        ({"S": 0.6, "F": 0.4}, {"S": {"+": 0.8, "-": 0.2}, "F": {"+": 0.1, "-": 0.9}}, "P(S|+)", 0.923),
        # product launch: P(high|pos) = 0.739
        (
            {"high": 0.4, "low": 0.6},
            {"high": {"pos": 0.85, "neg": 0.15}, "low": {"pos": 0.2, "neg": 0.8}},
            "P(high|pos)",
            0.739,
        ),
    ],
)
def test_examples_posteriors(prior, likelihood, key, expected):
    assert rsm.decide.prob_table(prior, likelihood).values[key] == pytest.approx(expected, abs=5e-4)


def test_three_states_three_signals():
    pt = rsm.decide.prob_table(
        {"low": 0.3, "mid": 0.5, "high": 0.2},
        {
            "low": {"weak": 0.7, "ok": 0.2, "strong": 0.1},
            "mid": {"weak": 0.2, "ok": 0.6, "strong": 0.2},
            "high": {"weak": 0.1, "ok": 0.2, "strong": 0.7},
        },
    )
    for g in pt.signals:
        assert sum(pt.values[f"P({s}|{g})"] for s in pt.states) == pytest.approx(1.0)
    assert sum(pt.values[f"P({g})"] for g in pt.signals) == pytest.approx(1.0)


def test_variables_reproduce_the_values_in_a_tree():
    pt = rsm.decide.prob_table(**POPUP)
    yaml = (
        "name: check\n"
        + pt.variables()
        + "type: chance\nPositive:\n    p: P(Pos)\n    payoff: P(S|Pos)\nNegative:\n    p: P(Neg)\n    payoff: P(S|Neg)\n"
    )
    tree = rsm.decide.dtree(yaml)
    assert not tree.errors
    # EMV = P(Pos) P(S|Pos) + P(Neg) P(S|Neg) = P(S)
    assert tree.payoff == pytest.approx(0.6)
    assert "P(F): 1 - P(S)" in pt.variables()
    assert "P(S|Pos): P(Pos and S) / P(Pos)" in pt.variables()
    assert pt.variables(header=False).startswith("    ## Given")


@pytest.mark.parametrize(
    "prior, likelihood, message",
    [
        ({"S": 0.6, "F": 0.5}, POPUP["likelihood"], "add up to 1.1"),
        (POPUP["prior"], {"S": {"Pos": 0.8, "Neg": 0.3}, "F": {"Pos": 0.1, "Neg": 0.9}}, "P(signal | S) adds up to 1.1"),
        (POPUP["prior"], {"S": {"Pos": 0.8, "Neg": 0.2}}, "no P(signal | state) row for F"),
        (POPUP["prior"], {"S": {"Pos": 0.8, "Neg": 0.2}, "F": {"Pos": 0.1, "Bad": 0.9}}, "every state needs Pos, Neg"),
        ({"S": 0.6, "F": 0.4}, {"S": {"Pos": 1, "Neg": 0}, "F": {"Pos": 1, "Neg": 0}}, "Signal Neg never happens"),
        ({"S|x": 0.6, "F": 0.4}, POPUP["likelihood"], "cannot contain"),
        ({"S": 1.2, "F": -0.2}, POPUP["likelihood"], "between 0 and 1"),
    ],
)
def test_input_errors_explain_the_fix(prior, likelihood, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)").replace("|", r"\|")):
        rsm.decide.prob_table(prior, likelihood)


def test_report_and_text():
    pt = rsm.decide.prob_table(**POPUP, state="store", signal="popup")
    rep = pt.report()
    assert [t["id"] for t in rep["tables"]] == ["conditional", "joint", "posterior"]
    assert rep["tables"][1]["rows"][-1] == {"store": "Total", "Pos": "0.5200", "Neg": "0.4800", "Total": "1.0000"}
    assert rep["tables"][2]["rows"][0]["Pos"] == "0.9231"
    d = pt.to_dict()
    assert d["values"]["P(S|Neg)"] == pytest.approx(0.25)
    assert "Posterior probabilities" in repr(pt)


@pytest.mark.parametrize("example_id, tree, key", [
    ("imperfect-information", "full", "P(S|+)"),
    ("product-launch", "product launch", "P(high|pos)"),
])
def test_example_tables_match_the_tree_variables(example_id, tree, key):
    ex = rsm.decide.dtree_example(example_id)
    spec = ex["prob_table"]
    pt = rsm.decide.prob_table(spec["prior"], spec["likelihood"])
    t = rsm.decide.dtree(ex["trees"][tree], subtree_library=ex["trees"])
    assert t.vars[key] == pytest.approx(pt.values[key])
