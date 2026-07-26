"""Tests for pyrsm.decide.optimize (linear/integer programming)."""

import math

import pytest

import pyrsm as rsm
from pyrsm.decide.optimize import Constraint, DecisionVariable, RobustnessResult, optimize


@pytest.fixture
def production_mix():
    """The canonical 2-variable production-mix LP.

    maximize 40a + 30b
        2a + 1b <= 400   (labor)
        1a + 2b <= 500   (material)
        a, b >= 0
    Optimum: a=100, b=200, objective=10000.
    """
    return {
        "meta": {"problem_name": "Production Mix", "objective_sense": "maximize"},
        "variables": {
            "product_a": {"label": "Units of A", "type": "continuous", "min": 0},
            "product_b": {"label": "Units of B", "type": "continuous", "min": 0},
        },
        "objective": {"product_a": 40.0, "product_b": 30.0},
        "constraints": [
            {
                "name": "labor",
                "coefficients": {"product_a": 2.0, "product_b": 1.0},
                "sense": "L",
                "rhs": 400.0,
            },
            {
                "name": "material",
                "coefficients": {"product_a": 1.0, "product_b": 2.0},
                "sense": "L",
                "rhs": 500.0,
            },
        ],
    }


class TestSolveLP:
    """Core continuous LP behavior."""

    def test_optimal_solution(self, production_mix):
        opt = optimize(production_mix)
        assert opt.is_solved
        assert opt.status == "Optimal"
        assert math.isclose(opt.objective_value, 10000.0, abs_tol=1e-6)
        assert math.isclose(opt.solution["product_a"], 100.0, abs_tol=1e-6)
        assert math.isclose(opt.solution["product_b"], 200.0, abs_tol=1e-6)

    def test_shadow_prices(self, production_mix):
        opt = optimize(production_mix)
        sp = {c.name: c.shadow_price for c in opt.constraints}
        assert math.isclose(sp["labor"], 16.6667, abs_tol=1e-2)
        assert math.isclose(sp["material"], 6.6667, abs_tol=1e-2)

    def test_slack_and_binding(self, production_mix):
        opt = optimize(production_mix)
        for c in opt.constraints:
            assert math.isclose(c.slack, 0.0, abs_tol=1e-6)
            assert c.binding is True

    def test_nonbinding_constraint_has_slack(self, production_mix):
        # A slack constraint that never binds keeps positive slack and zero dual.
        production_mix["constraints"].append(
            {
                "name": "loose",
                "coefficients": {"product_a": 1.0, "product_b": 1.0},
                "sense": "L",
                "rhs": 10000.0,
            }
        )
        opt = optimize(production_mix)
        loose = next(c for c in opt.constraints if c.name == "loose")
        assert loose.slack > 0
        assert loose.binding is False
        assert math.isclose(loose.shadow_price, 0.0, abs_tol=1e-6)

    def test_reduced_costs_present_for_lp(self, production_mix):
        opt = optimize(production_mix)
        assert all(v.reduced_cost is not None for v in opt.variables.values())

    def test_minimize_direction(self):
        # minimize 2x + 3y s.t. x + y >= 10  ->  optimum x=10, y=0, obj=20
        model = {
            "meta": {"objective_sense": "minimize"},
            "variables": {"x": {"min": 0}, "y": {"min": 0}},
            "objective": {"x": 2, "y": 3},
            "constraints": [{"coefficients": {"x": 1, "y": 1}, "sense": "G", "rhs": 10}],
        }
        opt = optimize(model)
        assert opt.is_solved
        assert math.isclose(opt.objective_value, 20.0, abs_tol=1e-6)

    def test_variable_upper_bound(self):
        model = {
            "meta": {"objective_sense": "maximize"},
            "variables": {"x": {"min": 0, "max": 5}},
            "objective": {"x": 1},
            "constraints": [{"coefficients": {"x": 1}, "sense": "L", "rhs": 100}],
        }
        opt = optimize(model)
        assert math.isclose(opt.solution["x"], 5.0, abs_tol=1e-6)


class TestIntegerAndBinary:
    def test_integer_program(self):
        # maximize 5x + 4y; 6x+4y<=24; x+2y<=6; x,y integer -> obj 20 (x=4,y=0)
        model = {
            "meta": {"objective_sense": "maximize"},
            "variables": {"x": {"type": "integer", "min": 0}, "y": {"type": "integer", "min": 0}},
            "objective": {"x": 5, "y": 4},
            "constraints": [
                {"coefficients": {"x": 6, "y": 4}, "sense": "L", "rhs": 24},
                {"coefficients": {"x": 1, "y": 2}, "sense": "L", "rhs": 6},
            ],
        }
        opt = optimize(model)
        assert opt.is_solved
        assert math.isclose(opt.objective_value, 20.0, abs_tol=1e-6)
        assert float(opt.solution["x"]).is_integer()

    def test_duals_suppressed_for_mip(self):
        model = {
            "meta": {"objective_sense": "maximize"},
            "variables": {"x": {"type": "integer", "min": 0}},
            "objective": {"x": 1},
            "constraints": [{"coefficients": {"x": 1}, "sense": "L", "rhs": 3}],
        }
        opt = optimize(model)
        assert all(c.shadow_price is None for c in opt.constraints)
        assert all(v.reduced_cost is None for v in opt.variables.values())
        assert any(w.code == "duals_unavailable" for w in opt.warnings)

    def test_binary_variable_bounds_pinned(self):
        model = {
            "meta": {"objective_sense": "maximize"},
            "variables": {"x": {"type": "binary"}},
            "objective": {"x": 1},
            "constraints": [{"coefficients": {"x": 1}, "sense": "L", "rhs": 5}],
        }
        opt = optimize(model)
        assert opt.variables["x"].lower == 0.0
        assert opt.variables["x"].upper == 1.0
        assert math.isclose(opt.solution["x"], 1.0, abs_tol=1e-6)


class TestInfeasibleAndUnbounded:
    def test_infeasible(self):
        model = {
            "meta": {"objective_sense": "maximize"},
            "variables": {"x": {"min": 0}},
            "objective": {"x": 1},
            "constraints": [
                {"coefficients": {"x": 1}, "sense": "L", "rhs": 1},
                {"coefficients": {"x": 1}, "sense": "G", "rhs": 5},
            ],
        }
        opt = optimize(model)
        assert not opt.is_solved
        assert opt.status == "Infeasible"
        assert opt.objective_value is None

    def test_unbounded(self):
        model = {
            "meta": {"objective_sense": "maximize"},
            "variables": {"x": {"min": 0, "max": None}},
            "objective": {"x": 1},
            "constraints": [],
        }
        opt = optimize(model)
        assert not opt.is_solved
        assert opt.status in {"Unbounded", "Undefined"}


class TestValidation:
    def test_no_variables(self):
        opt = optimize({"meta": {}, "variables": {}, "objective": {}})
        assert any(e.code == "no_variables" for e in opt.errors)
        assert not opt.is_solved

    def test_no_objective(self):
        opt = optimize({"variables": {"x": {}}, "objective": {}})
        assert any(e.code == "no_objective" for e in opt.errors)

    def test_unknown_objective_variable(self):
        opt = optimize(
            {"variables": {"x": {}}, "objective": {"x": 1, "z": 2}, "constraints": []}
        )
        assert any(e.code == "unknown_objective_var" for e in opt.errors)

    def test_unknown_constraint_variable(self):
        opt = optimize(
            {
                "variables": {"x": {}},
                "objective": {"x": 1},
                "constraints": [{"coefficients": {"q": 1}, "sense": "L", "rhs": 3}],
            }
        )
        assert any(e.code == "unknown_constraint_var" for e in opt.errors)

    def test_bad_sense(self):
        opt = optimize(
            {
                "variables": {"x": {}},
                "objective": {"x": 1},
                "constraints": [{"coefficients": {"x": 1}, "sense": "≤≤", "rhs": 3}],
            }
        )
        assert any(e.code == "bad_sense" for e in opt.errors)

    def test_bad_bounds(self):
        opt = optimize(
            {"variables": {"x": {"min": 10, "max": 1}}, "objective": {"x": 1}}
        )
        assert any(e.code == "bad_bounds" for e in opt.errors)

    def test_errors_block_solving(self):
        opt = optimize({"variables": {}, "objective": {}})
        assert opt.status == "Not Solved"

    def test_sense_aliases_accepted(self):
        for alias in ("<=", "le", "L"):
            opt = optimize(
                {
                    "variables": {"x": {"min": 0}},
                    "objective": {"x": 1},
                    "constraints": [{"coefficients": {"x": 1}, "sense": alias, "rhs": 4}],
                }
            )
            assert not opt.errors
            assert math.isclose(opt.objective_value, 4.0, abs_tol=1e-6)


class TestInputForms:
    def test_json_string_input(self, production_mix):
        import json

        opt = optimize(json.dumps(production_mix))
        assert opt.is_solved
        assert math.isclose(opt.objective_value, 10000.0, abs_tol=1e-6)

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            optimize("{not valid json")

    def test_non_dict_raises(self):
        with pytest.raises(TypeError):
            optimize(42)

    def test_sense_override(self, production_mix):
        production_mix["meta"]["objective_sense"] = "minimize"
        opt = optimize(production_mix, sense="maximize")
        assert opt.sense == "maximize"
        assert math.isclose(opt.objective_value, 10000.0, abs_tol=1e-6)


class TestSerialization:
    def test_to_dict_roundtrips(self, production_mix):
        opt = optimize(production_mix)
        d = opt.to_dict()
        assert d["solution"]["is_solved"] is True
        assert math.isclose(d["solution"]["objective_value"], 10000.0, abs_tol=1e-6)
        assert set(d["variables"]) == {"product_a", "product_b"}
        assert len(d["constraints"]) == 2

    def test_python_code_is_valid_and_reproduces(self, production_mix):
        opt = optimize(production_mix)
        ns: dict = {}
        exec(opt.python_code.replace("opt.summary()", "RESULT = opt"), ns)
        assert math.isclose(ns["RESULT"].objective_value, 10000.0, abs_tol=1e-6)

    def test_dataframes(self, production_mix):
        opt = optimize(production_mix)
        assert opt.solution_df.height == 2
        assert opt.constraints_df.height == 2
        assert "shadow_price" in opt.constraints_df.columns
        assert opt.sensitivity().height == 2


class TestRobustness:
    def test_objective_noise_varies_outcome(self, production_mix):
        opt = optimize(production_mix)
        rr = opt.robustness(trials=200, coef_noise=0.15, seed=42)
        assert isinstance(rr, RobustnessResult)
        stats = {r["metric"]: r["value"] for r in rr.stats().to_dicts()}
        assert stats["std"] > 0
        assert stats["solved"] == 200

    def test_rhs_noise_varies_outcome(self, production_mix):
        opt = optimize(production_mix)
        rr = opt.robustness(trials=100, coef_noise=0.0, rhs_noise=0.1, seed=7)
        stats = {r["metric"]: r["value"] for r in rr.stats().to_dicts()}
        assert stats["std"] > 0

    def test_reproducible_with_seed(self, production_mix):
        opt = optimize(production_mix)
        a = opt.robustness(trials=50, coef_noise=0.2, seed=99).df["objective"].to_list()
        b = opt.robustness(trials=50, coef_noise=0.2, seed=99).df["objective"].to_list()
        assert a == b

    def test_raises_when_unsolved(self):
        opt = optimize(
            {
                "variables": {"x": {"min": 0}},
                "objective": {"x": 1},
                "constraints": [
                    {"coefficients": {"x": 1}, "sense": "L", "rhs": 1},
                    {"coefficients": {"x": 1}, "sense": "G", "rhs": 5},
                ],
            }
        )
        with pytest.raises(ValueError):
            opt.robustness(trials=10)


class TestPlot:
    def test_plot_two_vars(self, production_mix):
        import matplotlib

        matplotlib.use("Agg")
        opt = optimize(production_mix)
        p = opt.plot(resolution=60)
        assert type(p).__name__ == "ggplot"

    def test_plot_rejects_non_two_var(self):
        opt = optimize(
            {"variables": {"x": {"min": 0}}, "objective": {"x": 1}, "constraints": []}
        )
        with pytest.raises(ValueError):
            opt.plot()

    def test_robustness_plot(self, production_mix):
        import matplotlib

        matplotlib.use("Agg")
        opt = optimize(production_mix)
        rr = opt.robustness(trials=50, coef_noise=0.2, seed=1)
        assert type(rr.plot()).__name__ == "ggplot"


class TestLazyImport:
    def test_callable_via_decide_namespace(self, production_mix):
        # rsm.decide.optimize is callable whether it resolves to the class or
        # the callable module (see _make_module_callable), matching dtree.
        opt = rsm.decide.optimize(production_mix)
        assert isinstance(opt, optimize)
        assert opt.is_solved

    def test_dataclasses_exported(self):
        assert rsm.decide.DecisionVariable is DecisionVariable
        assert rsm.decide.Constraint is Constraint
        assert rsm.decide.RobustnessResult is RobustnessResult
