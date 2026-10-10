"""Tests for pyrsm.model.simulate Monte Carlo engine."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pyrsm.model.simulate import (
    Diagnostic,
    Formula,
    FormulaError,
    SimulationSpec,
    Variable,
    compile_formula,
    repeat_simulate,
    simulate,
)


# ---------------------------------------------------------------------------
# Deterministic seed
# ---------------------------------------------------------------------------


def test_same_seed_produces_same_data():
    spec = SimulationSpec(
        runs=200,
        seed=42,
        variables=[Variable(name="x", kind="normal", mean=0, sd=1)],
    )
    a = simulate(spec)
    b = simulate(spec)
    assert a.data["x"].to_list() == b.data["x"].to_list()


def test_different_seed_produces_different_data():
    base = SimulationSpec(
        runs=200,
        seed=1,
        variables=[Variable(name="x", kind="normal", mean=0, sd=1)],
    )
    other = SimulationSpec(
        runs=200,
        seed=2,
        variables=[Variable(name="x", kind="normal", mean=0, sd=1)],
    )
    assert simulate(base).data["x"].to_list() != simulate(other).data["x"].to_list()


# ---------------------------------------------------------------------------
# Distribution shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "var,expected_min,expected_max",
    [
        (Variable(name="c", kind="constant", value=5.0), 5.0, 5.0),
        (Variable(name="u", kind="uniform", min=10, max=20), 10.0, 20.0),
    ],
)
def test_bounded_distributions(var, expected_min, expected_max):
    spec = SimulationSpec(runs=500, seed=7, variables=[var])
    result = simulate(spec)
    assert not result.has_errors
    arr = result.data[var.name].to_numpy()
    assert arr.shape == (500,)
    assert arr.min() >= expected_min - 1e-9
    assert arr.max() <= expected_max + 1e-9


def test_normal_mean_is_close():
    spec = SimulationSpec(
        runs=20000,
        seed=11,
        variables=[Variable(name="x", kind="normal", mean=10.0, sd=2.0)],
    )
    result = simulate(spec)
    arr = result.data["x"].to_numpy()
    assert abs(arr.mean() - 10.0) < 0.1
    assert abs(arr.std(ddof=1) - 2.0) < 0.1


def test_binomial_in_range():
    spec = SimulationSpec(
        runs=2000,
        seed=3,
        variables=[Variable(name="k", kind="binomial", trials=10, prob=0.3)],
    )
    arr = simulate(spec).data["k"].to_numpy()
    assert arr.min() >= 0
    assert arr.max() <= 10


def test_poisson_non_negative():
    spec = SimulationSpec(
        runs=1000,
        seed=5,
        variables=[Variable(name="n", kind="poisson", rate=3.5)],
    )
    arr = simulate(spec).data["n"].to_numpy()
    assert arr.min() >= 0


def test_discrete_only_uses_declared_values():
    spec = SimulationSpec(
        runs=1000,
        seed=9,
        variables=[
            Variable(name="p", kind="discrete", values=[6, 8], probs=[0.3, 0.7]),
        ],
    )
    arr = simulate(spec).data["p"].to_numpy()
    unique = set(int(x) for x in np.unique(arr))
    assert unique <= {6, 8}


def test_discrete_probs_default_to_uniform():
    spec = SimulationSpec(
        runs=2000,
        seed=4,
        variables=[Variable(name="p", kind="discrete", values=[1, 2, 3, 4])],
    )
    arr = simulate(spec).data["p"].to_numpy()
    # Each should appear roughly 500 times +/- noise; loose check
    counts = {v: int((arr == v).sum()) for v in (1, 2, 3, 4)}
    for v, count in counts.items():
        assert 350 < count < 650, f"discrete value {v} count {count} outside loose band"


# ---------------------------------------------------------------------------
# Validation diagnostics
# ---------------------------------------------------------------------------


def test_invalid_runs_is_diagnostic():
    spec = SimulationSpec(runs=0, seed=1, variables=[])
    result = simulate(spec)
    assert result.has_errors
    assert any(d.code == "bad_runs" for d in result.diagnostics)


def test_unknown_kind_is_diagnostic():
    spec = SimulationSpec(runs=100, seed=1, variables=[Variable(name="x", kind="hyperbolic_wat")])
    result = simulate(spec)
    assert result.has_errors
    assert "x" in result.data.columns or True  # x should not be present
    assert "x" not in result.data.columns
    assert any(d.target == "x" for d in result.diagnostics)


def test_duplicate_names_flagged():
    spec = SimulationSpec(
        runs=10,
        seed=1,
        variables=[
            Variable(name="x", kind="constant", value=1),
            Variable(name="x", kind="constant", value=2),
        ],
    )
    result = simulate(spec)
    assert any(d.code == "duplicate_name" for d in result.diagnostics)


def test_bad_identifier_flagged():
    spec = SimulationSpec(
        runs=10,
        seed=1,
        variables=[Variable(name="1bad", kind="constant", value=1)],
    )
    result = simulate(spec)
    assert any(d.code == "bad_name" for d in result.diagnostics)


def test_discrete_probs_sum_validation():
    spec = SimulationSpec(
        runs=10,
        seed=1,
        variables=[
            Variable(name="x", kind="discrete", values=[1, 2], probs=[-0.1, 1.1]),
        ],
    )
    result = simulate(spec)
    assert any("cannot be negative" in d.message for d in result.diagnostics)


# ---------------------------------------------------------------------------
# Formula evaluation
# ---------------------------------------------------------------------------


def test_formula_arithmetic():
    spec = SimulationSpec(
        runs=10,
        seed=1,
        variables=[
            Variable(name="price", kind="constant", value=10),
            Variable(name="qty", kind="constant", value=3),
        ],
        formulas=[Formula(name="rev", expr="price * qty")],
    )
    result = simulate(spec)
    assert not result.has_errors
    assert result.data["rev"].to_list() == [30.0] * 10


def test_formula_dependencies_resolve_in_order():
    spec = SimulationSpec(
        runs=5,
        seed=1,
        variables=[Variable(name="x", kind="constant", value=2)],
        formulas=[
            Formula(name="y", expr="x * 3"),
            Formula(name="z", expr="y + 1"),
        ],
    )
    result = simulate(spec)
    assert result.data["z"].to_list() == [7.0] * 5


def test_formula_unknown_name_diagnostic():
    spec = SimulationSpec(
        runs=5,
        seed=1,
        variables=[Variable(name="x", kind="constant", value=1)],
        formulas=[Formula(name="bad", expr="x + nope")],
    )
    result = simulate(spec)
    assert any("unknown name 'nope'" in d.message for d in result.diagnostics)


def test_formula_comparison_produces_boolean():
    spec = SimulationSpec(
        runs=5,
        seed=1,
        variables=[Variable(name="profit", kind="constant", value=50)],
        formulas=[Formula(name="loss", expr="profit < 100")],
    )
    result = simulate(spec)
    assert result.data["loss"].to_list() == [True] * 5


def test_formula_ifelse():
    spec = SimulationSpec(
        runs=4,
        seed=1,
        variables=[
            Variable(name="x", kind="discrete", values=[1, 2, 3, 4]),
        ],
        formulas=[Formula(name="big", expr="ifelse(x > 2, 100, 0)")],
    )
    # Use a deterministic discrete by manually setting probs heavily on one
    # to avoid relying on a specific draw — but here we only check schema/range.
    result = simulate(spec)
    arr = set(result.data["big"].to_list())
    assert arr <= {0, 100}


def test_formula_rejects_function_call_security():
    """Arbitrary calls like __import__ must be rejected."""
    with pytest.raises(FormulaError):
        compile_formula("__import__('os')", known=set())


def test_formula_rejects_attribute_access():
    with pytest.raises(FormulaError):
        compile_formula("x.evil", known={"x"})


def test_formula_chained_compare_works():
    spec = SimulationSpec(
        runs=200, seed=1, variables=[Variable(name="x", kind="uniform", min=0, max=10)],
        formulas=[Formula(name="mid", expr="2 < x < 8")],
    )
    result = simulate(spec)
    x = result.data["x"].to_numpy()
    assert result.data["mid"].to_list() == list((x > 2) & (x < 8))


def test_formula_logical_negation():
    spec = SimulationSpec(
        runs=3,
        seed=1,
        variables=[Variable(name="x", kind="constant", value=5)],
        formulas=[Formula(name="big", expr="not (x < 3)")],
    )
    result = simulate(spec)
    assert result.data["big"].to_list() == [True, True, True]


# ---------------------------------------------------------------------------
# Logical probabilities (the killer feature)
# ---------------------------------------------------------------------------


def test_logical_formula_mean_matches_probability():
    # If profit ~ N(100, 50), P(profit < 100) should be ~ 0.5
    spec = SimulationSpec(
        runs=50000,
        seed=17,
        variables=[Variable(name="profit", kind="normal", mean=100, sd=50)],
        formulas=[Formula(name="below", expr="profit < 100")],
    )
    result = simulate(spec)
    prob = float(result.data["below"].cast(pl.Float64).mean())
    assert abs(prob - 0.5) < 0.02


# ---------------------------------------------------------------------------
# Output types and reproducible code
# ---------------------------------------------------------------------------


def test_output_is_polars_dataframe():
    spec = SimulationSpec(
        runs=10,
        seed=1,
        variables=[Variable(name="x", kind="normal", mean=0, sd=1)],
    )
    assert isinstance(simulate(spec).data, pl.DataFrame)


def test_summary_rows_present():
    spec = SimulationSpec(
        runs=200,
        seed=1,
        variables=[Variable(name="x", kind="normal", mean=0, sd=1)],
        formulas=[Formula(name="big", expr="x > 0")],
    )
    result = simulate(spec)
    names = {row["name"] for row in result.summary_rows}
    assert names == {"x", "big"}


def test_histograms_present():
    spec = SimulationSpec(
        runs=200,
        seed=1,
        variables=[Variable(name="x", kind="normal", mean=0, sd=1)],
    )
    hist = simulate(spec).histograms["x"]
    assert sum(hist["counts"]) == 200
    assert len(hist["edges"]) == len(hist["counts"]) + 1


def test_generated_python_code_runs_and_matches():
    spec = SimulationSpec(
        runs=50,
        seed=23,
        variables=[
            Variable(name="x", kind="normal", mean=0, sd=1),
            Variable(name="y", kind="uniform", min=0, max=10),
        ],
        formulas=[Formula(name="z", expr="x + y")],
    )
    result = simulate(spec)
    namespace: dict = {}
    code = result.python_code.replace("sim.summary()", "").replace("sim.plot()", "")
    exec(code, namespace)  # noqa: S102 - controlled generated code
    df = namespace["simdat"]
    assert isinstance(df, pl.DataFrame)
    assert {"x", "y", "z"} <= set(df.columns)
    # The generated code should reproduce identical draws under the same seed
    assert df["x"].to_list() == result.data["x"].to_list()


def test_spec_round_trip_dict():
    spec = SimulationSpec(
        runs=100,
        seed=99,
        variables=[Variable(name="x", kind="normal", mean=0, sd=1)],
        formulas=[Formula(name="big", expr="x > 0")],
    )
    payload = spec.to_dict()
    rebuilt = SimulationSpec.from_dict(payload)
    assert rebuilt.runs == 100
    assert rebuilt.seed == 99
    assert rebuilt.variables[0].name == "x"
    assert rebuilt.formulas[0] == "big = x > 0"


def test_simulate_accepts_dict_payload():
    payload = {
        "runs": 10,
        "seed": 1,
        "variables": [{"name": "x", "kind": "constant", "value": 7}],
        "formulas": [{"name": "y", "expr": "x + 1"}],
    }
    result = simulate(payload)
    assert result.data["y"].to_list() == [8.0] * 10


def test_diagnostic_is_serializable():
    d = Diagnostic(severity="error", code="x", message="m", target="t")
    assert d.to_dict() == {
        "severity": "error",
        "code": "x",
        "message": "m",
        "target": "t",
    }


# ---------------------------------------------------------------------------
# Repeated simulation
# ---------------------------------------------------------------------------


