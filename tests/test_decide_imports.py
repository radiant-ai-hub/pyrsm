"""Import-surface tests for the ``pyrsm.decide`` namespace."""

from __future__ import annotations


def test_decide_namespace_exports_decision_tree():
    import pyrsm as rsm

    assert callable(rsm.decide.dtree)

    from pyrsm.decide import dtree
    from pyrsm.decide.dtree import dtree as direct_dtree
    from pyrsm.model.dtree import dtree as legacy_dtree

    exported_dtree = getattr(dtree, "dtree", dtree)
    assert exported_dtree is direct_dtree
    assert legacy_dtree is direct_dtree

    tree = rsm.decide.dtree(
        """
        name: import smoke
        type: decision
        A:
            payoff: 1
        B:
            payoff: 2
        """
    )
    assert tree.payoff == 2


def test_decide_namespace_exports_simulation():
    import pyrsm as rsm
    from pyrsm.decide import SimulationSpec, Variable, simulate
    from pyrsm.decide.simulate import simulate as direct_simulate
    from pyrsm.model.simulate import simulate as legacy_simulate

    assert callable(rsm.decide.simulate)
    exported_simulate = getattr(simulate, "simulate", simulate)
    assert exported_simulate is direct_simulate
    assert legacy_simulate is direct_simulate

    result = simulate(
        SimulationSpec(
            runs=3,
            seed=1,
            variables=[Variable(name="x", kind="constant", value=2)],
        )
    )
    assert result.data["x"].to_list() == [2.0, 2.0, 2.0]
