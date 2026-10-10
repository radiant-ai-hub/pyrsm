"""Simulate (radiant.model's simulater/repeater) with Python formulas, plus
queues and custom functions."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import pyrsm as rsm
from pyrsm.decide.simfuns import queue_length, queue_wait
from pyrsm.decide.simlang import Interpreter, SimLangError, formula_list, split_formulas
from pyrsm.decide.simulate import repeat_simulate, simulate


def _sim(variables, formulas=(), **kw):
    return simulate(
        {"runs": kw.pop("runs", 1000), "seed": kw.pop("seed", 1234), "variables": variables, "formulas": list(formulas) if not isinstance(formulas, str) else formulas, **kw}
    )


def _run_code(code: str) -> dict:
    """Run generated code from a file, as a notebook would (functions need
    their source, which ``exec`` of a string does not keep)."""
    import os
    import runpy
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code)
    try:
        return runpy.run_path(fh.name)
    finally:
        os.unlink(fh.name)


# -- formulas: Python expressions and the simfuns functions -------------------


def test_formula_text_becomes_a_list_of_formulas():
    entries = formula_list("# daily\ndemand = 10 * x  # units\nprofit = demand - 2; big = profit > 5\nz = ifelse(x > 1,\n  1, 0)")
    assert entries == [
        "# daily",
        "demand = 10 * x  # units",
        "profit = demand - 2",
        "big = profit > 5",
        "z = ifelse(x > 1, 1, 0)",
    ]
    assert [(n, e) for n, e, _ in split_formulas(entries)] == [
        ("demand", "10 * x"),
        ("profit", "demand - 2"),
        ("big", "profit > 5"),
        ("z", "ifelse(x > 1, 1, 0)"),
    ]


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("x > 1 and y == 2", [False, True, False]),
        ("x > 1 or y == 2", [True, True, True]),
        ("not x > 1", [True, False, False]),
        ("x ** 2", [1.0, 4.0, 9.0]),
        ("x % 2", [1.0, 0.0, 1.0]),
        ("x in [1, 3]", [True, False, True]),
        ("x not in [1, 3]", [False, True, False]),
        ("True and x > 2", [False, False, True]),
        ("ifelse(x > 1, x, 0)", [0.0, 2.0, 3.0]),
        ("max_horizontal(x, y)", [2.0, 2.0, 3.0]),
        ("sum_horizontal(x, y)", [3.0, 4.0, 4.0]),
        ("cum_sum(x)", [1.0, 3.0, 6.0]),
        ("shift(x)", [0.0, 1.0, 2.0]),
        ("clip(x, 1.5, 2.5)", [1.5, 2.0, 2.5]),
    ],
)
def test_python_formulas(expr, expected):
    interp = Interpreter(names={"x": np.array([1.0, 2.0, 3.0]), "y": np.array([2.0, 2.0, 1.0])}, rng=np.random.default_rng(1), nrows=3)
    assert list(np.asarray(interp.evaluate(expr))) == expected


def test_one_value_functions_and_indexing():
    interp = Interpreter(names={"x": np.array([1.0, 5.0, 3.0]), "y": np.array([10.0, 20.0, 30.0])}, rng=np.random.default_rng(1), nrows=3)
    assert interp.evaluate("min(x, y)") == 1.0  # over all inputs
    assert interp.evaluate("std(x)") == pytest.approx(2.0)
    assert interp.evaluate("y[arg_max(x)]") == 20.0
    assert interp.evaluate("y[arg_min(x)]") == 10.0
    assert interp.evaluate("y[-1]") == 30.0
    assert interp.evaluate("quantile(y, 0.5)") == 20.0
    assert interp.evaluate("first(y) + last(y) + count(y)") == 43.0


@pytest.mark.parametrize(
    ("expr", "advice"),
    [
        ("x ^ 2", "use ** for a power"),
        ("x > 1 & y > 1", "use and"),
        ("x > 1 | y > 1", "use or"),
        ("!x", "use not"),
        ("x %% 2", "use % for the remainder"),
        ("TRUE", "use True and False"),
    ],
)
def test_other_languages_get_python_advice(expr, advice):
    interp = Interpreter(names={"x": np.ones(3), "y": np.ones(3)}, rng=np.random.default_rng(1))
    import re

    with pytest.raises(SimLangError, match=re.escape(advice)):
        interp.evaluate(expr)
    with pytest.raises(SimLangError, match="use = to assign"):
        split_formulas(["x <- 1"])


@pytest.mark.parametrize(
    "expr",
    ["__import__('os')", "x.__class__", "open('f')", "'text'", "[i for i in x]", "lambda: 1"],
)
def test_formulas_cannot_escape(expr):
    interp = Interpreter(names={"x": np.ones(3)}, rng=np.random.default_rng(1))
    with pytest.raises(SimLangError):
        interp.evaluate(expr)


def test_update_an_existing_variable_and_scalar_formulas():
    sim = _sim([{"name": "x", "kind": "normal", "mean": 10, "sd": 2}], ["x = x * 2", "avg = mean(x)", "above = x > avg"], runs=500)
    assert not sim.has_errors
    assert sim.data.columns == ["x", "avg", "above"]
    assert sim.data["x"].mean() == pytest.approx(20, abs=0.5)
    assert sim.data["avg"].n_unique() == 1
    rows = {r["name"]: r["kind"] for r in sim.summary_rows}
    assert rows == {"x": "numeric", "avg": "constant", "above": "logical"}


# -- variable types ------------------------------------------------------------


def test_grid_expands_all_combinations_first_fastest():
    sim = _sim(
        [{"name": "x", "kind": "grid", "min": 1, "max": 3, "step": 1}, {"name": "y", "kind": "grid", "min": 4, "max": 5, "step": 1}]
    )
    assert sim.data["x"].to_list() == [1, 2, 3, 1, 2, 3]
    assert sim.data["y"].to_list() == [4, 4, 4, 5, 5, 5]


def test_parameters_can_use_constants_and_fractions():
    sim = _sim(
        [
            {"name": "base", "kind": "constant", "value": 100},
            {"name": "d", "kind": "normal", "mean": "base * 2", "sd": "base / 10"},
            {"name": "k", "kind": "discrete", "values": [1, 2, 3], "probs": ["1/3", "1/3", "1/3"]},
        ],
        runs=4000,
    )
    assert not sim.has_errors, sim.diagnostics
    assert sim.data["d"].mean() == pytest.approx(200, abs=1)
    assert set(sim.data["k"].unique().to_list()) == {1.0, 2.0, 3.0}


def test_exact_normals_and_correlations():
    sim = _sim(
        [{"name": "a", "kind": "normal", "mean": 5, "sd": 2}, {"name": "b", "kind": "normal", "mean": 0, "sd": 1}],
        normal_exact=True,
        correlations=[{"a": "a", "b": "b", "r": -0.4}],
        runs=300,
    )
    a, b = sim.data["a"].to_numpy(), sim.data["b"].to_numpy()
    assert a.mean() == pytest.approx(5) and a.std(ddof=1) == pytest.approx(2)
    assert np.corrcoef(a, b)[0, 1] == pytest.approx(-0.4)


def test_exponential_and_triangular():
    sim = _sim(
        [{"name": "e", "kind": "exponential", "rate": 0.5}, {"name": "t", "kind": "triangular", "min": 1, "mode": 2, "max": 6}],
        runs=20000,
    )
    assert sim.data["e"].mean() == pytest.approx(2, rel=0.05)
    assert sim.data["t"].mean() == pytest.approx(3, rel=0.02)
    assert sim.data["t"].min() >= 1 and sim.data["t"].max() <= 6


def test_input_data_columns_and_sdw():
    rng = np.random.default_rng(5)
    data = pl.DataFrame({"A": rng.normal(0, 0.05, 200), "B": rng.normal(0, 0.1, 200)})
    sim = simulate(
        {"variables": [{"name": "w", "kind": "grid", "min": 0, "max": 1, "step": 0.5}], "formulas": ["risk = weighted_sum_std(w, 1 - w, A, B)"], "data": "stocks"},
        data=data,
    )
    assert not sim.has_errors, sim.diagnostics
    assert "A" not in sim.data.columns
    expected = [np.std(w * data["A"].to_numpy() + (1 - w) * data["B"].to_numpy(), ddof=1) for w in (0, 0.5, 1)]
    assert sim.data["risk"].to_list() == pytest.approx(expected)
    bad = simulate({"runs": 10, "variables": [{"name": "w", "kind": "constant", "value": 1}], "formulas": ["x = A + w"]}, data=data)
    assert "200 values" in bad.diagnostics[0].message


# -- custom functions -------------------------------------------------------------


FUNCS = """
def running_max(x):
    out = zeros(len(x))
    best = x[0]
    for i in range(len(x)):
        if x[i] > best:
            best = x[i]
        out[i] = best
    return out

def double(x, k=2):
    return x * k
"""


def test_custom_functions_with_loops_and_defaults():
    sim = _sim([{"name": "x", "kind": "normal", "mean": 0, "sd": 1}], ["m = running_max(x)", "d = double(x)", "t = double(x, 3)"], runs=50, functions=FUNCS)
    assert not sim.has_errors, sim.diagnostics
    x = sim.data["x"].to_numpy()
    assert sim.data["m"].to_list() == list(np.maximum.accumulate(x))
    assert sim.data["t"].to_list() == pytest.approx(list(3 * x))
    # the function's x is a copy: the simulated column is unchanged
    assert sim.data["x"].to_list() == list(x)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("import os", "only function definitions"),
        ("def f(x):\n    return x.mean()", "'.mean' is not supported"),
        ("def f(*args):\n    return 1", "plain inputs"),
        ("def f(x):\n    while True:\n        x = x + 1", "steps"),
        ("def f(x):\n    return x ^ 2", "use \\*\\* for a power"),
    ],
)
def test_custom_function_errors(source, message):
    sim = simulate({"runs": 5, "variables": [{"name": "x", "kind": "constant", "value": 1}], "functions": source, "formulas": ["y = f(x)"]}, budget=50_000)
    assert sim.has_errors
    import re

    assert re.search(message, sim.diagnostics[0].message)


# -- queues ---------------------------------------------------------------------------


def test_queue_wait_matches_hand_calculation():
    gaps = np.array([0.0, 1.0, 1.0, 1.0, 5.0])
    service = np.array([3.0, 3.0, 1.0, 1.0, 1.0])
    # one server: arrivals 0,1,2,3,8; starts 0,3,6,7,8
    assert list(queue_wait(gaps, service, 1)) == [0, 2, 4, 4, 0]
    # customer 4 arrives at 3: customer 2 is in service (started at 3), 3 waits
    assert list(queue_length(gaps, service, 1)) == [0, 0, 1, 1, 0]
    # two servers: starts 0,1,3,4,8
    assert list(queue_wait(gaps, service, 2)) == [0, 0, 1, 1, 0]


def test_queue_wait_agrees_with_a_loop_function():
    lindley = """
def wait_one(gap, service):
    w = zeros(len(gap))
    for i in range(1, len(gap)):
        w[i] = max(0, w[i - 1] + service[i - 1] - gap[i])
    return w
"""
    sim = _sim(
        [{"name": "gap", "kind": "exponential", "rate": 1}, {"name": "service", "kind": "exponential", "rate": 1.25}],
        ["a = queue_wait(gap, service, 1)", "b = wait_one(gap, service)"],
        runs=500,
        functions=lindley,
    )
    assert sim.data["a"].to_list() == pytest.approx(sim.data["b"].to_list())


# -- summary text ---------------------------------------------------------------------


def test_summary_text_has_rs_sections():
    ex = rsm.decide.simulate_example("profit")
    text = rsm.decide.simulate(ex["spec"]).summary_text()
    for part in ("Simulation\n", "Simulations: 1,000", "Random seed: 1234", "Constant   : var_cost 5; fixed_cost 1000",
                 "Discrete   : price 6 8 0.3 0.7", "Constants:", "Variables:", "n_obs", "Logicals:", "True (prop)"):  # fmt: skip
        assert part in text


# -- repeat ---------------------------------------------------------------------------


def test_repeat_by_simulation_sums_over_repetitions():
    ex = rsm.decide.simulate_example("profit")
    rep = repeat_simulate(ex["spec"], reps=365, vars=["E", "price"], sum_vars=["profit"], by="sim", fun="sum", formulas=["low = profit_sum < 36500"])
    assert not rep.has_errors, rep.diagnostics
    assert rep.data.columns == [".sim", "profit_sum", "low"]
    assert rep.data.height == 1000
    # expected daily profit: 0.3 x (700 x 1 - 1000) + 0.7 x (600 x 3 - 1000) = 470
    assert rep.data["profit_sum"].mean() == pytest.approx(365 * 470, rel=0.02)


def test_repeat_by_repeat_several_functions():
    spec = {"runs": 100, "seed": 1, "variables": [{"name": "x", "kind": "normal", "mean": 0, "sd": 1}], "formulas": ["y = x * 2"]}
    rep = repeat_simulate(spec, reps=20, vars=["x"], sum_vars=["y"], by="rep", fun=["mean", "p95", "std", "std_prop", "first"])
    assert rep.data.height == 20
    assert rep.data.columns == [".rep", "y_mean", "y_p95", "y_std", "y_std_prop", "y_first"]


def test_repeat_keeps_variables_not_resimulated():
    spec = {"runs": 10, "seed": 1, "variables": [{"name": "a", "kind": "normal", "mean": 0, "sd": 1}, {"name": "b", "kind": "normal", "mean": 0, "sd": 1}]}
    rep = repeat_simulate(spec, reps=5, vars=["a"], sum_vars=["a", "b"], by="rep", fun="sum")
    assert rep.data["b_sum"].n_unique() == 1
    assert rep.data["a_sum"].n_unique() == 5


def test_repeat_none_keeps_every_row():
    spec = {"runs": 4, "seed": 1, "variables": [{"name": "a", "kind": "normal", "mean": 0, "sd": 1}]}
    rep = repeat_simulate(spec, reps=3, vars=["a"], sum_vars=["a"], fun="none")
    assert rep.data.columns == [".rep", ".sim", "a"]
    assert rep.data.height == 12


def test_repeat_grid_replaces_constants():
    ex = rsm.decide.simulate_example("bank-queue")
    r = dict(ex["repeat"])
    r.pop("name")
    rep = repeat_simulate(ex["spec"], **r)
    assert not rep.has_errors, rep.diagnostics
    assert rep.data["tellers"].to_list() == [1, 2, 3, 4]
    waits = rep.data["wait_mean"].to_list()
    assert waits[0] > waits[1] > waits[2] > waits[3]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"vars": []}, "select variables to re-simulate"),
        ({"vars": ["nope"]}, "cannot re-simulate nope"),
        ({"vars": ["x"], "fun": "garbage"}, "unknown function"),
        ({"vars": ["x"], "by": "rep", "grid": [{"name": "x", "min": 1, "max": 2}]}, "replaces constants"),
        ({"vars": ["x"], "sum_vars": ["zz"]}, "not in the simulated data"),
    ],
)
def test_repeat_input_errors(kwargs, message):
    spec = {"runs": 10, "seed": 1, "variables": [{"name": "x", "kind": "normal", "mean": 0, "sd": 1}]}
    rep = repeat_simulate(spec, reps=3, **kwargs)
    assert rep.has_errors
    assert any(message in d.message for d in rep.diagnostics)


# -- examples and generated code ---------------------------------------------------------


@pytest.mark.parametrize("example", [e["id"] for e in rsm.decide.simulate_examples()])
def test_examples_run_and_their_code_reproduces_them(example):
    ex = rsm.decide.simulate_example(example)
    data = None
    if ex.get("requires"):
        data = simulate(rsm.decide.simulate_example(ex["requires"][0])["spec"]).data
    sim = simulate(ex["spec"], data=data)
    assert not sim.has_errors, sim.diagnostics
    code = sim.python_code.replace("sim.summary()", "").replace("sim.plot()", "")
    if data is not None:
        code = code.replace("spec = {", "import polars as pl\nimport pyrsm as rsm\nreturns = rsm.decide.simulate(rsm.decide.simulate_example('stock-returns')['spec']).data\nspec = {", 1)
    ns = _run_code(code)
    assert ns["simdat"].equals(sim.data)


def test_plots():
    ex = rsm.decide.simulate_example("profit")
    sim = simulate(ex["spec"])
    fig = sim.plot()
    assert fig is not None
    rep = repeat_simulate(ex["spec"], reps=20, vars=["E"], sum_vars=["profit", "profit_small"], fun="mean")
    assert rep.plot() is not None


def test_repeat_functions_of_their_own():
    spec = {"runs": 30, "seed": 1, "variables": [{"name": "x", "kind": "normal", "mean": 0, "sd": 1}],
            "functions": "def sq(v):\n    return v * v", "formulas": ["y = sq(x)"]}  # fmt: skip
    rep = repeat_simulate(
        spec, reps=5, vars=["x"], sum_vars=["y"], by="rep", fun="mean",
        functions="def big(v, k=1):\n    return v > k", formulas=["high = big(y_mean)", "same = sq(y_mean)"],
    )  # fmt: skip
    assert not rep.has_errors, rep.diagnostics
    assert {"high", "same"} <= set(rep.data.columns)
    assert "Functions     : big" in rep.summary_text()
    # the functions are written as Python functions and passed by name
    assert "def big(v, k=1):" in rep.python_code and "functions=[big]" in rep.python_code
    assert '"functions": [sq]' in rep.python_code
    ns = _run_code(rep.python_code.replace("rep.summary()", "").replace("rep.plot()", ""))
    assert ns["repdat"].equals(rep.data)
    bad = repeat_simulate(spec, reps=2, vars=["x"], functions="import os")
    assert bad.has_errors and "repeat functions" in bad.diagnostics[0].message
