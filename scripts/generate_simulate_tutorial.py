#!/usr/bin/env python3
"""Generate the simulation notebooks (examples/decide/simulate-tutorial).

One notebook per example in ``pyrsm/decide/simulate_examples.py`` (the same
specs as Radiant's Model > Simulate "Load example" menu), plus the overview
``examples/decide/decide-simulate``. Writes executed ``.ipynb`` notebooks
(cells run in-process, outputs stored) and matching ``.qmd`` files.

    python scripts/generate_simulate_tutorial.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "decide" / "simulate-tutorial"
QMD = ROOT / "examples" / "qmd" / "decide" / "simulate-tutorial"

# the notebook runner of the decision-tree tutorial
_spec = importlib.util.spec_from_file_location("_dtree_gen", ROOT / "scripts" / "generate_dtree_tutorial.py")
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)


def _example(eid: str) -> dict:
    import pyrsm as rsm

    return rsm.decide.simulate_example(eid)


def _spec_code(eid: str, var: str = "spec") -> str:
    """The example's functions (as Python) and spec, as the generated code
    writes them."""
    from pyrsm.decide.simulate import SimulationSpec, _code_head

    spec = SimulationSpec.from_dict(_example(eid)["spec"])
    lines, _ = _code_head(spec, "")
    lines = [ln for ln in lines if ln != "import pyrsm as rsm"]
    code = "\n".join(lines).strip()
    return code.replace("spec = {", f"{var} = {{", 1) if var != "spec" else code


def _repeat_code(eid: str, spec_var: str = "spec", data: str = "") -> str:
    from pyrsm.decide.simulate import _py

    rep = dict(_example(eid)["repeat"])
    rep.pop("name", None)
    args = []
    for key, value in rep.items():
        if key in ("formulas", "grid") and value:
            args += [f"    {key}=[", *[f"        {_py(v)}," for v in value], "    ],"]
        else:
            args.append(f"    {key}={_py(value)},")
    if data:
        args.append(f"    data={data},")
    return "\n".join(["rep = rsm.decide.repeat_simulate(", f"    {spec_var},", *args, ")", "rep.summary()"])


def question(eid: str) -> str:
    return "> " + _example(eid)["question"]


SETUP = "import pyrsm as rsm"

NOTEBOOKS: list[tuple[Path, Path, str, str, list[tuple[str, str]]]] = []


def nb(name: str, title: str, cells: list[tuple[str, str]], *, out: Path = OUT, qmd: Path = QMD) -> None:
    NOTEBOOKS.append((out, qmd, name, title, cells))


nb(
    "01-profit",
    "Daily profit and a year of it",
    [
        ("md", "In Radiant: _Model > Decide > Simulate_, **Load example > Daily profit**."),
        ("md", "## Question\n\n" + question("profit")),
        ("code", SETUP),
        ("md", "Constants (`var_cost`, `fixed_cost`) do not vary; `E` (a normal demand shock) and "
         "`price` (discrete: 6 or 8 with probabilities 0.3 and 0.7) are drawn for every "
         "simulation. The formulas are evaluated in order, so `profit` can use `demand`, and "
         "`profit_small` is a logical (`True`/`False`)."),
        ("code", _spec_code("profit")),
        ("code", "sim = rsm.decide.simulate(spec)\nsim.summary()"),
        ("md", "The summary lists the constants, the variables that vary and, for each logical, "
         "how often it is `True`: here the probability that daily profit is below $100."),
        ("code", "sim.plot()"),
        ("code", "sim.data.head()"),
        ("md", "## A year of profit: repeated simulation\n\nRe-simulate `E` and `price` 365 times "
         "(days) and sum `profit` for each of the 1,000 simulations (**group by Simulation**): "
         "1,000 simulated years. `profit_sum` is the sum of `profit`; a formula on the repeated "
         "data gives the probability that a year's profit is below $36,500."),
        ("code", _repeat_code("profit")),
        ("code", "rep.plot()"),
        ("md", "Daily profit is below $100 on more than a quarter of the days, but a whole year "
         "with less than $36,500 does not happen: good and bad days average out. A simulation "
         "should match the horizon of the decision."),
    ],
)

nb(
    "02-optimal-price",
    "Optimal price: grid search",
    [
        ("md", "In Radiant: **Load example > Optimal price (grid search)**."),
        ("md", "## Question\n\n" + question("optimal-price")),
        ("code", SETUP),
        ("md", "A **grid** variable runs from a minimum to a maximum in steps. With grid variables "
         "the number of rows is the number of grid values (601 prices), not the number of "
         "simulations. `arg_max(profit)` is the row with the highest profit, so "
         "`price[arg_max(profit)]` is the best price; `max(profit)` is one number, the highest "
         "profit."),
        ("code", _spec_code("optimal-price")),
        ("code", "sim = rsm.decide.simulate(spec)\nsim.summary()"),
        ("code", "import polars as pl\n\nsim.data.sort(\"profit\", descending=True).head(3)"),
        ("md", "Two grid variables give every combination: with `x` from 1 to 3 and `y` from 4 to "
         "5, the first varies fastest."),
        ("code", "rsm.decide.simulate({\"variables\": [\n"
                 "    {\"name\": \"x\", \"kind\": \"grid\", \"min\": 1, \"max\": 3, \"step\": 1},\n"
                 "    {\"name\": \"y\", \"kind\": \"grid\", \"min\": 4, \"max\": 5, \"step\": 1},\n"
                 "]}).data"),
    ],
)

nb(
    "03-bank-queue",
    "A bank queue",
    [
        ("md", "In Radiant: **Load example > Bank queue (waiting times)**."),
        ("md", "## Question\n\n" + question("bank-queue")),
        ("code", SETUP),
        ("md", "Each row is a customer. `gap` is the time since the previous customer arrived and "
         "`service` the time a teller needs; both are **exponential** with a rate (the mean is "
         "1 / rate). `queue_wait(gap, service, tellers)` works out, customer by customer, when a "
         "teller is free (first come, first served) and returns each customer's wait; "
         "`queue_length` returns how many customers are already waiting when a customer "
         "arrives. `in_bank[-1]` is the last customer (counting from the end, as in Python)."),
        ("code", _spec_code("bank-queue")),
        ("code", "sim = rsm.decide.simulate(spec)\nsim.summary(dec=2)"),
        ("code", "sim.plot()"),
        ("md", "## How many tellers? Grid search in a repeated simulation\n\nWith **group by "
         "Repeat**, a grid search replaces a constant of the simulation in each repetition: "
         "`tellers` from 1 to 4 gives four repetitions, each re-simulating the 2,000 customers. "
         "`mean` and `p95` summarise the waits of each repetition."),
        ("code", _repeat_code("bank-queue")),
        ("code", "rep.data"),
        ("md", "With one teller service is slower than arrivals and the line keeps growing. Two "
         "tellers are busy about 86% of the time and most customers wait; a third teller cuts the "
         "average wait to under a minute.\n\nThe same wait can be written as a custom function "
         "(Lindley's recursion for one server) -- `queue_wait` does this for any number of "
         "servers:"),
        ("code", "from pyrsm.decide.simfuns import zeros\n\n\n"
                 "def wait_one(gap, service):\n"
                 "    w = zeros(len(gap))\n"
                 "    for i in range(1, len(gap)):\n"
                 "        w[i] = max(0, w[i - 1] + service[i - 1] - gap[i])\n"
                 "    return w\n\n\n"
                 "one = dict(spec, functions=[wait_one], formulas=[\"w1 = queue_wait(gap, service, 1)\", \"w2 = wait_one(gap, service)\", \"same = abs(w1 - w2) < 1e-9\"])\n"
                 "rsm.decide.simulate(one).data[\"same\"].all()"),
    ],
)

nb(
    "04-gamblers-ruin",
    "Gambler's ruin: a custom function",
    [
        ("md", "In Radiant: **Load example > Gambler's ruin (custom function)**."),
        ("md", "## Question\n\n" + question("gamblers-ruin")),
        ("code", SETUP),
        ("md", "Each row is a day; `win` is 1 on a winning day. Wealth depends on yesterday's "
         "wealth, so it is calculated in a loop: a **custom function**, an ordinary Python "
         "function passed in the spec under `functions`. It uses the simulation functions "
         "(`zeros`, `ifelse`, `mean`, ...) imported from `pyrsm.decide.simfuns`, so it can be "
         "tried on its own; inside a simulation it runs with the same safety rules as in Radiant "
         "(no imports, files or `.` attributes). Constants are single numbers, so they can be "
         "passed in as they are."),
        ("code", _spec_code("gamblers-ruin")),
        ("code", "import numpy as np\n\ncalc_wealth(np.array([1, 0, 0, 1]), 1000, 0.05, 5)"),
        ("code", "sim = rsm.decide.simulate(spec)\nsim.summary()"),
        ("md", "One simulated year is one path. Repeat it 1,000 times (**group by Repeat**) and "
         "keep the `last` value (wealth at the end of the year) and the `min` (the lowest point "
         "of the year):"),
        ("code", _repeat_code("gamblers-ruin")),
        ("code", "rep.plot()"),
        ("md", "A loop is easy to read but slow for long simulations. The same wealth without a "
         "loop: each day multiplies wealth by `1 + 5 x 0.05` or `1 - 0.05`, so"),
        ("code", "fast = dict(spec, formulas=[\"wealth = start * cum_prod(ifelse(win == 1, 1 + win_mult * perc_invest, 1 - perc_invest))\"])\n"
                 "fast.pop(\"functions\")\n"
                 "(rsm.decide.simulate(fast).data[\"wealth\"] - sim.data[\"wealth\"]).abs().max()"),
    ],
)

nb(
    "05-portfolio",
    "Portfolio risk: correlated returns and input data",
    [
        ("md", "In Radiant: **Load example > Correlated stock returns**, press _Run simulation_ "
         "(saved as `returns`), then **Load example > Portfolio weights (input data)**."),
        ("md", "## Question\n\n" + question("stock-returns")),
        ("code", SETUP),
        ("md", "**Correlations** between normal variables are given per pair. With "
         "`normal_exact` the simulated returns have exactly the means, standard deviations and "
         "correlations given (\"Use exact specifications\" in Radiant)."),
        ("code", _spec_code("stock-returns")),
        ("code", "returns = rsm.decide.simulate(spec).data\nreturns.corr()"),
        ("md", "## Question\n\n" + question("portfolio")),
        ("md", "The simulated returns are the **input data**: their columns (`A`, `B`, `C`) can be "
         "used in the formulas but are not part of the result. A grid search over the weights "
         "of A and B gives 441 rows; `weighted_sum_std(wA, wB, wC, A, B, C)` is, for each row of weights, "
         "the standard deviation of the weighted sum of the returns."),
        ("code", _spec_code("portfolio", "pspec")),
        ("code", "port = rsm.decide.simulate(pspec, data=returns)\nport.summary(dec=3)"),
        ("code", "import polars as pl\n\nport.data.filter(pl.col(\"valid\")).sort(\"risk\").head(5)"),
    ],
)


def readme() -> None:
    import pyrsm as rsm

    lines = [
        "# Simulation tutorial (Python)",
        "",
        "Worked simulation examples with `pyrsm.decide.simulate` and "
        "`pyrsm.decide.repeat_simulate`. The same examples are in Radiant under "
        "_Model > Decide > Simulate > Load example_ and in Python with "
        "`rsm.decide.simulate_examples()`.",
        "",
        "| Notebook | Topic |",
        "|---|---|",
        "| [01-profit](01-profit.ipynb) | Constants, normal and discrete variables, formulas, repeated simulation (group by Simulation) |",
        "| [02-optimal-price](02-optimal-price.ipynb) | Grid search and `arg_max` |",
        "| [03-bank-queue](03-bank-queue.ipynb) | Exponential arrivals, `queue_wait`, repeat with a grid over a constant |",
        "| [04-gamblers-ruin](04-gamblers-ruin.ipynb) | A custom function with a loop; repeat with `last` and `min` |",
        "| [05-portfolio](05-portfolio.ipynb) | Correlated normals, input data, `weighted_sum_std` |",
        "",
        "## Questions and answers",
        "",
    ]
    for ex in rsm.decide.simulate_examples():
        lines += [f"**{ex['title']}.** {ex['question']}", "", f"_Answer:_ {ex['answer']}", ""]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(out: Path, qmd: Path, name: str, title: str, cells: list[tuple[str, str]]) -> None:
    _gen.OUT, _gen.QMD = out, qmd
    _gen.build(name, title, cells)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    QMD.mkdir(parents=True, exist_ok=True)
    for out, qmd, name, title, cells in NOTEBOOKS:
        build(out, qmd, name, title, cells)
        print("wrote", name)
    # the overview notebook next to decide-dtree
    overview = [
        ("md", "Monte Carlo simulation with `pyrsm.decide.simulate`: describe uncertain inputs by "
         "their distribution, combine them with formulas, and look at the range of outcomes. "
         "Worked examples, one per notebook, are in [simulate-tutorial](simulate-tutorial/README.md)."),
        *NOTEBOOKS[0][4][1:],
    ]
    build(ROOT / "examples" / "decide", ROOT / "examples" / "qmd" / "decide", "decide-simulate",
          "Decide > Simulate", overview)  # fmt: skip
    readme()
    print(json.dumps({"notebooks": len(NOTEBOOKS) + 1, "out": str(OUT)}))


if __name__ == "__main__":
    main()
