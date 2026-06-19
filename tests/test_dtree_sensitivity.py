"""Tests for ``pyrsm.model.dtree`` sensitivity sweeps.

Sensitivity is a first-class workflow: 1-, 2-, and 3-variable grids over
named ``variables:`` entries, with dependent-formula recomputation and
diagnostics for invalid probability combinations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pyrsm.model.dtree import SensitivitySpec, dtree

# ---------------------------------------------------------------------------
# Shared trees
# ---------------------------------------------------------------------------

# A decision-chance-decision tree with an EMV that crosses zero at
# p_success = 0.5 (when payoff_success = -payoff_failure and cost matches).
OPEN_STORE_YAML = """
name: open
variables:
    p_success: 0.6
    p_failure: 1 - p_success
    payoff_success: 5000000
    payoff_failure: -1000000
    fixed_cost: 2000000
type: decision
open:
    cost: fixed_cost
    type: chance
    success:
        p: p_success
        payoff: payoff_success
    failure:
        p: p_failure
        payoff: payoff_failure
not open:
    payoff: 0
"""


CONTRACT_YAML = """
name: sign
variables:
    p_small: 0.3
    p_medium: 0.6
    p_large: 1 - p_small - p_medium
    legal_fees: 5000
type: decision
Movie:
    cost: legal_fees
    type: chance
    Small:
        p: p_small
        payoff: 200000
    Medium:
        p: p_medium
        payoff: 800000
    Large:
        p: p_large
        payoff: 3000000
TV:
    payoff: 900000
"""


TUTORIAL_DIR = Path(
    "/home/vnijs/gh/rsm-teaching/rsm-mgt403/tutorial-decision-analysis/decision_tree/input"
)


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------


class TestSpecParsing:
    def test_values_list_form(self):
        t = dtree(OPEN_STORE_YAML)
        res = t.sensitivity({"p_success": {"values": [0.3, 0.5, 0.7]}})
        assert res.shape == (3,)
        assert res.names == ["p_success"]
        assert [c.inputs["p_success"] for c in res.cells] == [0.3, 0.5, 0.7]

    def test_min_max_step_form_inclusive(self):
        t = dtree(OPEN_STORE_YAML)
        res = t.sensitivity({"p_success": {"min": 0.2, "max": 0.8, "step": 0.2}})
        # Inclusive of both endpoints.
        assert [c.inputs["p_success"] for c in res.cells] == [0.2, 0.4, 0.6, 0.8]

    def test_bare_list_treated_as_values(self):
        t = dtree(OPEN_STORE_YAML)
        res = t.sensitivity({"p_success": [0.4, 0.6]})
        assert [c.inputs["p_success"] for c in res.cells] == [0.4, 0.6]

    def test_list_form_with_name(self):
        t = dtree(OPEN_STORE_YAML)
        res = t.sensitivity(
            [{"name": "p_success", "values": [0.3, 0.7]}],
        )
        assert res.shape == (2,)

    def test_sensitivityspec_passed_directly(self):
        t = dtree(OPEN_STORE_YAML)
        res = t.sensitivity([SensitivitySpec(name="p_success", values=[0.4, 0.6])])
        assert res.shape == (2,)

    def test_min_max_step_rejects_zero_step(self):
        t = dtree(OPEN_STORE_YAML)
        with pytest.raises(ValueError, match="step .* must be positive"):
            t.sensitivity({"p_success": {"min": 0.0, "max": 1.0, "step": 0.0}})

    def test_min_max_step_rejects_inverted_range(self):
        t = dtree(OPEN_STORE_YAML)
        with pytest.raises(ValueError, match="max < min"):
            t.sensitivity({"p_success": {"min": 0.8, "max": 0.2, "step": 0.1}})

    def test_spec_missing_required_keys_raises(self):
        t = dtree(OPEN_STORE_YAML)
        with pytest.raises(ValueError, match="needs 'values' or 'min'\\+'max'\\+'step'"):
            t.sensitivity({"p_success": {"max": 1.0}})

    def test_empty_specs_rejected(self):
        t = dtree(OPEN_STORE_YAML)
        with pytest.raises(ValueError, match="at least one variable"):
            t.sensitivity({})

    def test_too_many_specs_rejected(self):
        t = dtree(OPEN_STORE_YAML)
        with pytest.raises(ValueError, match="1-3 variables"):
            t.sensitivity(
                {
                    "p_success": [0.3, 0.5],
                    "payoff_success": [4e6, 5e6],
                    "payoff_failure": [-1e6, -2e6],
                    "fixed_cost": [1e6, 2e6],
                }
            )

    def test_unknown_variable_raises_with_available_list(self):
        t = dtree(OPEN_STORE_YAML)
        with pytest.raises(ValueError, match="missing.*not_a_real_var"):
            t.sensitivity({"not_a_real_var": [0.5]})

    def test_tree_without_variables_block_rejected(self):
        # video2_input.yaml has no `variables:` block.
        y = """
        name: x
        type: decision
        a:
            payoff: 1
        b:
            payoff: 2
        """
        t = dtree(y)
        with pytest.raises(ValueError, match="must already exist"):
            t.sensitivity({"foo": [0.5]})


# ---------------------------------------------------------------------------
# 1-way sweeps
# ---------------------------------------------------------------------------


class TestOneWay:
    def test_payoff_matches_hand_calculation(self):
        """EMV(open) = p*5M + (1-p)*(-1M) - 2M; threshold at p=0.5."""
        t = dtree(OPEN_STORE_YAML)
        res = t.sensitivity({"p_success": {"values": [0.3, 0.5, 0.7]}})
        # All cells should solve since p_failure = 1 - p_success keeps probs valid.
        assert all(c.is_solved for c in res.cells)
        by_p = {c.inputs["p_success"]: c for c in res.cells}
        # p=0.3: EMV(open) = 0.3*5M - 0.7*1M - 2M = 1.5M - 0.7M - 2M = -1.2M; max(_, 0) = 0
        assert by_p[0.3].payoff == pytest.approx(0.0)
        assert by_p[0.3].chosen_labels == ["not open"]
        # p=0.7: EMV(open) = 0.7*5M - 0.3*1M - 2M = 3.5M - 0.3M - 2M = 1.2M
        assert by_p[0.7].payoff == pytest.approx(1_200_000.0)
        assert by_p[0.7].chosen_labels == ["open"]

    def test_dependent_formula_recomputes_per_cell(self):
        """Changing p_success must drive p_failure = 1 - p_success."""
        t = dtree(OPEN_STORE_YAML)
        res = t.sensitivity({"p_success": [0.25, 0.75]})
        # If p_failure didn't track, the cells would all hit a
        # chance_probabilities_not_one diagnostic.
        assert all(c.is_solved for c in res.cells), [c.errors for c in res.cells]
        # And the payoffs must differ — proving the sweep actually
        # took effect rather than reading the base value.
        payoffs = sorted(c.payoff for c in res.cells)
        assert payoffs[0] < payoffs[1]

    def test_threshold_flip_detected_at_decision_boundary(self):
        t = dtree(OPEN_STORE_YAML)
        res = t.sensitivity({"p_success": [0.30, 0.45, 0.55, 0.70]})
        flips = res.threshold_flips()
        assert len(flips) == 1
        assert flips[0]["variable"] == "p_success"
        # Boundary lies somewhere between 0.45 and 0.55 (true threshold = 0.5).
        assert flips[0]["from_value"] == pytest.approx(0.45)
        assert flips[0]["to_value"] == pytest.approx(0.55)
        assert flips[0]["from_chosen"] == ["not open"]
        assert flips[0]["to_chosen"] == ["open"]

    def test_no_flip_when_choice_constant(self):
        t = dtree(OPEN_STORE_YAML)
        # Stay well above the threshold throughout.
        res = t.sensitivity({"p_success": [0.6, 0.7, 0.8]})
        assert res.threshold_flips() == []

    def test_threshold_flips_empty_for_multidim_grid(self):
        t = dtree(OPEN_STORE_YAML)
        res = t.sensitivity(
            {
                "p_success": [0.4, 0.7],
                "fixed_cost": [1_000_000, 3_000_000],
            }
        )
        assert res.threshold_flips() == []

    def test_base_result_preserved(self):
        t = dtree(OPEN_STORE_YAML)
        res = t.sensitivity({"p_success": [0.4, 0.7]})
        assert res.base_payoff == pytest.approx(t.payoff)
        # Base chose "open" (p_success default = 0.6).
        assert res.base_chosen == ["open"]
        # Running sensitivity must not mutate the parent tree.
        assert t.payoff == pytest.approx(600_000.0)
        assert t.vars["p_success"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 2-way sweeps
# ---------------------------------------------------------------------------


class TestTwoWay:
    def test_grid_shape_and_completeness(self):
        t = dtree(OPEN_STORE_YAML)
        res = t.sensitivity(
            {
                "p_success": [0.3, 0.5, 0.7],
                "fixed_cost": [1_500_000, 2_500_000],
            }
        )
        assert res.shape == (3, 2)
        assert len(res.cells) == 6  # prod(shape)
        # Every (p, cost) combination shows up exactly once.
        seen = {(c.inputs["p_success"], c.inputs["fixed_cost"]) for c in res.cells}
        expected = {(p, c) for p in (0.3, 0.5, 0.7) for c in (1_500_000, 2_500_000)}
        assert seen == expected

    def test_cartesian_order_is_first_axis_outer(self):
        """itertools.product order: leftmost var varies slowest."""
        t = dtree(OPEN_STORE_YAML)
        res = t.sensitivity(
            {
                "p_success": [0.3, 0.7],
                "fixed_cost": [1e6, 2e6],
            }
        )
        order = [(c.inputs["p_success"], c.inputs["fixed_cost"]) for c in res.cells]
        assert order == [(0.3, 1e6), (0.3, 2e6), (0.7, 1e6), (0.7, 2e6)]

    def test_lower_cost_makes_open_attractive(self):
        """At cost=0 every cell should choose open; at cost=10M, never."""
        t = dtree(OPEN_STORE_YAML)
        res = t.sensitivity(
            {
                "p_success": [0.3, 0.6, 0.9],
                "fixed_cost": [0, 10_000_000],
            }
        )
        by_key = {(c.inputs["p_success"], c.inputs["fixed_cost"]): c for c in res.cells}
        for p in (0.3, 0.6, 0.9):
            assert by_key[(p, 0)].chosen_labels == ["open"]
            assert by_key[(p, 10_000_000)].chosen_labels == ["not open"]


# ---------------------------------------------------------------------------
# 3-way sweeps
# ---------------------------------------------------------------------------


class TestThreeWay:
    def test_grid_shape(self):
        t = dtree(OPEN_STORE_YAML)
        res = t.sensitivity(
            {
                "p_success": [0.3, 0.6],
                "payoff_success": [4e6, 5e6],
                "payoff_failure": [-2e6, -1e6, 0],
            }
        )
        assert res.shape == (2, 2, 3)
        assert len(res.cells) == 12

    def test_each_axis_value_appears_in_every_combination(self):
        t = dtree(OPEN_STORE_YAML)
        res = t.sensitivity(
            {
                "p_success": [0.4, 0.7],
                "payoff_success": [4_000_000, 5_000_000],
                "fixed_cost": [1_000_000, 2_000_000],
            }
        )
        seen = {
            (c.inputs["p_success"], c.inputs["payoff_success"], c.inputs["fixed_cost"])
            for c in res.cells
        }
        expected = {
            (p, ps, fc)
            for p in (0.4, 0.7)
            for ps in (4_000_000, 5_000_000)
            for fc in (1_000_000, 2_000_000)
        }
        assert seen == expected


# ---------------------------------------------------------------------------
# Invalid-cell diagnostics
# ---------------------------------------------------------------------------


class TestInvalidCells:
    def test_three_branch_residual_breaks_when_independent_vars_clash(self):
        """If p_large = 1 - p_small - p_medium, varying p_small without
        moving p_medium produces cells where the residual probability
        falls outside [0, 1]. Those cells must remain in the result with
        their diagnostics, not be silently dropped."""
        t = dtree(CONTRACT_YAML)
        res = t.sensitivity({"p_small": [0.1, 0.3, 0.5]})
        assert len(res.cells) == 3
        # p_small=0.5 -> p_large = 1 - 0.5 - 0.6 = -0.1 (out of range).
        bad = [c for c in res.cells if c.inputs["p_small"] == 0.5]
        assert bad and bad[0].payoff is None
        codes = {d.code for d in bad[0].errors}
        assert "chance_probability_out_of_range" in codes
        # p_small=0.1 -> p_large = 0.3 (valid).
        good = [c for c in res.cells if c.inputs["p_small"] == 0.1]
        assert good and good[0].is_solved

    def test_independent_probability_violations_diagnosed_not_dropped(self):
        """Varying two unrelated chance siblings — many combinations
        won't sum to 1, but every cell must still be present in the
        result for the UI to render the heatmap."""
        y = """
        name: indep probs
        variables:
            p_a: 0.5
            p_b: 0.5
        type: chance
        A:
            p: p_a
            payoff: 100
        B:
            p: p_b
            payoff: 200
        """
        t = dtree(y)
        res = t.sensitivity(
            {
                "p_a": [0.2, 0.5],
                "p_b": [0.3, 0.5, 0.8],
            }
        )
        assert len(res.cells) == 6
        # Only (0.5, 0.5) and (0.2, 0.8) sum to 1.
        valid = res.valid_cells
        valid_keys = {(c.inputs["p_a"], c.inputs["p_b"]) for c in valid}
        assert valid_keys == {(0.5, 0.5), (0.2, 0.8)}
        # The other four must have a probability-sum diagnostic.
        for cell in res.invalid_cells:
            assert any(d.code == "chance_probabilities_not_one" for d in cell.errors)


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


class TestResultHelpers:
    def test_to_frame_one_row_per_cell(self):
        t = dtree(OPEN_STORE_YAML)
        res = t.sensitivity(
            {
                "p_success": [0.3, 0.7],
                "fixed_cost": [1e6, 2e6],
            }
        )
        df = res.to_frame()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert {"p_success", "fixed_cost", "payoff", "chosen", "error_codes"} <= set(df.columns)

    def test_to_frame_invalid_cells_have_error_codes(self):
        t = dtree(CONTRACT_YAML)
        res = t.sensitivity({"p_small": [0.5]})  # forces residual probability < 0
        df = res.to_frame()
        assert (df["error_codes"] != "").any()
        # Invalid cells still carry a row (None payoff is preserved).
        assert df["payoff"].isna().any()

    def test_to_dict_is_json_safe(self):
        import json

        t = dtree(OPEN_STORE_YAML)
        res = t.sensitivity({"p_success": [0.4, 0.6], "fixed_cost": [1e6, 2e6]})
        state = res.to_dict()
        # Round-trip through JSON without TypeError.
        text = json.dumps(state)
        assert json.loads(text) == state
        assert state["shape"] == [2, 2]
        assert len(state["cells"]) == 4
        # Each cell preserves its diagnostics shape.
        for cell in state["cells"]:
            assert isinstance(cell["errors"], list)
            assert isinstance(cell["warnings"], list)


# ---------------------------------------------------------------------------
# Tutorial integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TUTORIAL_DIR.is_dir(), reason="tutorial YAMLs not checked out")
class TestTutorialSensitivity:
    def test_video7_sensitivity_on_prior_changes_chosen_branch(self):
        """In the popup-test tree, dropping P(S) low enough should make
        the test no longer worth its $500k cost."""
        yl = (TUTORIAL_DIR / "video7_input.yaml").read_text()
        t = dtree(yl)
        res = t.sensitivity({"P(S)": [0.2, 0.4, 0.6, 0.8]})
        # All cells must solve — every dependent Bayes formula tracks P(S).
        assert all(c.is_solved for c in res.cells), [c.errors for c in res.cells]
        # And the chosen branch should change across the range.
        choices = {tuple(sorted(c.chosen_labels)) for c in res.cells}
        assert len(choices) > 1

    def test_video7_invalid_inputs_diagnosed_not_dropped(self):
        """A negative prior is not a valid probability — pyrsm should
        diagnose it but the cell still belongs in the result."""
        yl = (TUTORIAL_DIR / "video7_input.yaml").read_text()
        t = dtree(yl)
        # P(S) > 1 makes P(F) = 1 - P(S) negative, breaking downstream.
        res = t.sensitivity({"P(S)": [0.5, 1.5]})
        assert len(res.cells) == 2
        bad = [c for c in res.cells if c.inputs["P(S)"] == 1.5]
        assert bad and bad[0].payoff is None
        assert bad[0].errors  # at least one diagnostic
