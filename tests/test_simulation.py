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
    assert any(d.code == "bad_variable" and "non-negative" in d.message for d in result.diagnostics)


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
    assert any(d.code == "bad_formula" for d in result.diagnostics)


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


def test_formula_rejects_chained_compare():
    with pytest.raises(FormulaError):
        compile_formula("0 < x < 5", known={"x"})


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
    exec(result.python_code, namespace)  # noqa: S102 - controlled generated code
    df = namespace["data"]
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
    assert rebuilt.formulas[0].expr == "x > 0"


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


def test_repeat_simulate_shape_and_aggregation():
    # 50 reps × 10 per-period draws (= spec.runs), x is constant 2.
    spec = SimulationSpec(
        runs=10,
        seed=7,
        variables=[Variable(name="x", kind="constant", value=2)],
    )
    r = repeat_simulate(spec, reps=50, agg="sum")
    assert not r.has_errors
    assert r.data.shape == (50, 1)
    assert list(r.data.columns) == ["x_sum"]
    # Sum over 10 constants of 2 = 20 per rep
    assert r.data["x_sum"].to_list() == [20.0] * 50


def test_repeat_simulate_mean_aggregation():
    spec = SimulationSpec(
        runs=10,
        seed=7,
        variables=[Variable(name="x", kind="constant", value=2)],
    )
    r = repeat_simulate(spec, reps=50, agg="mean")
    assert r.data["x_mean"].to_list() == [2.0] * 50


def test_repeat_simulate_non_resample_reuses_same_draws():
    """Non-resample variables reuse the same per-period draws every rep."""
    spec = SimulationSpec(
        runs=20,
        seed=42,
        variables=[Variable(name="z", kind="normal", mean=0, sd=10)],
    )
    r = repeat_simulate(spec, reps=100, agg="sum", resample=[])
    # Every rep aggregates the SAME 20 draws → all reps get the same sum.
    sums = r.data["z_sum"].to_list()
    assert len(set(sums)) == 1


def test_repeat_simulate_resample_gives_different_per_rep():
    """When a variable IS in resample, each rep gets fresh draws → varying sums."""
    spec = SimulationSpec(
        runs=20,
        seed=42,
        variables=[Variable(name="z", kind="normal", mean=0, sd=10)],
    )
    r = repeat_simulate(spec, reps=100, agg="sum", resample=["z"])
    sums = r.data["z_sum"].to_list()
    assert len(set(sums)) > 50  # plenty of variation across reps


def test_repeat_simulate_boolean_sum_counts_true():
    spec = SimulationSpec(
        runs=100,
        seed=1,
        variables=[Variable(name="x", kind="constant", value=5)],
        formulas=[Formula(name="big", expr="x > 0")],
    )
    r = repeat_simulate(spec, reps=10, agg="sum")
    # Each rep: 100 per-period values of True → sum=100
    assert r.data["big_sum"].to_list() == [100.0] * 10


def test_repeat_simulate_diversification_reduces_tail():
    # Daily profit ~ N(10, 50), resampled every rep across 365 days.
    spec = SimulationSpec(
        runs=365,
        seed=11,
        variables=[Variable(name="profit", kind="normal", mean=10, sd=50)],
    )
    r = repeat_simulate(spec, reps=2000, agg="sum", resample=["profit"])
    annual_profit = r.data["profit_sum"].to_numpy()
    # 365 * 10 = 3650 mean, sd = 50 * sqrt(365) ≈ 955
    assert abs(annual_profit.mean() - 3650) < 60
    p_annual_loss = float((annual_profit < 0).mean())
    assert p_annual_loss < 0.05


def test_repeat_simulate_repeat_formula_on_aggregated_column():
    """Repeat formulas can reference {col}_{agg} aggregated columns."""
    spec = SimulationSpec(
        runs=365,
        seed=3,
        variables=[Variable(name="profit", kind="normal", mean=10, sd=50)],
    )
    r = repeat_simulate(
        spec,
        reps=2000,
        agg="sum",
        resample=["profit"],
        repeat_formulas=[Formula(name="annual_loss", expr="profit_sum < 0")],
    )
    assert "annual_loss" in r.data.columns
    p_annual_loss = float(r.data["annual_loss"].cast(pl.Float64).mean())
    assert p_annual_loss < 0.05


def test_repeat_simulate_bad_reps():
    spec = SimulationSpec(runs=10, variables=[Variable(name="x", kind="constant", value=1)])
    r = repeat_simulate(spec, reps=0)
    assert r.has_errors
    assert any(d.code == "bad_reps" for d in r.diagnostics)


def test_repeat_simulate_bad_agg():
    spec = SimulationSpec(runs=10, variables=[Variable(name="x", kind="constant", value=1)])
    r = repeat_simulate(spec, reps=5, agg="garbage")
    assert r.has_errors
    assert any(d.code == "bad_agg" for d in r.diagnostics)


def test_repeat_simulate_unknown_resample_is_warning():
    spec = SimulationSpec(runs=10, variables=[Variable(name="x", kind="constant", value=1)])
    r = repeat_simulate(spec, reps=5, resample=["does_not_exist"])
    assert not r.has_errors
    assert any(d.severity == "warning" and d.code == "unknown_resample" for d in r.diagnostics)


def test_repeat_simulate_generated_code_present():
    spec = SimulationSpec(
        runs=10,
        seed=1,
        variables=[Variable(name="x", kind="constant", value=1)],
    )
    r = repeat_simulate(spec, reps=5, agg="sum")
    assert "reps = 5" in r.python_code
    assert "_agg" in r.python_code
    assert "aggregated" in r.python_code
