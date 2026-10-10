---
name: pyrsm-simulate
description: Build, run, and interpret Monte Carlo simulations with pyrsm.decide.simulate (radiant.model's simulater/repeater, plus queues and custom functions)
---

# Monte Carlo simulation (`pyrsm.decide.simulate`)

Express an uncertain business problem as named inputs (constants and
distributions) plus formulas, draw many rows, and read the summary. Use
`repeat_simulate` when the question is about a total or summary over many
periods or trials.

## When to use

- "What price should we charge if demand is uncertain?" (grid search)
- "What is the probability annual profit is below target?" (repeat)
- "How long do customers wait with 2 tellers? With 3?" (queue + repeat grid)
- "Which portfolio weights minimise risk?" (input data + grid + `sdw`)

Use `pyrsm.decide.dtree` instead for a few discrete outcomes with known
probabilities, and the model tools for fit/predict tasks.

## Spec

```python
spec = {
    "name": "simdat",          # the saved dataset
    "runs": 1000,              # rows (ignored when there are grid variables)
    "seed": 1234,
    "variables": [...],
    "formulas": [              # one "name = expression" string per formula
        "profit = demand * (price - cost)",
        "loss = profit < 0",
    ],
    "functions": [my_function],  # optional: Python functions (or their source text)
    "normal_exact": False,     # normal variables with exactly the mean / sd given
    "correlations": [],        # [{"a": "x", "b": "y", "r": 0.5}] between normal variables
    "data": "",                # dataset whose columns formulas can use (pass data=...)
}
```

### Variable types (`kind` and its fields)

| kind | fields | use |
|---|---|---|
| `constant` | `value` | knowns, decision levers (a single number in formulas) |
| `normal` | `mean`, `sd` | symmetric noise |
| `lognormal` | `meanlog`, `sdlog` | positive, right-skewed |
| `uniform` | `min`, `max` | anywhere in a range |
| `triangular` | `min`, `mode`, `max` | min / most likely / max estimates |
| `binomial` | `trials`, `prob` | successes in trials (`trials: 1` = a yes/no event) |
| `poisson` | `rate` | counts of events |
| `exponential` | `rate` | time between arrivals, service times (mean = 1 / rate) |
| `discrete` | `values`, `probs` | scenarios; probabilities may be fractions ("1/3") and must sum to 1 |
| `sequence` | `min`, `max` | evenly spaced over the rows |
| `grid` | `min`, `max`, `step` | every combination of all grid variables; sets the rows |

Fields can be numbers or expressions of earlier names (`"mean": "base * 2"`);
constants are evaluated first.

### Formulas

Python expressions, as in pyrsm's data filter: `+ - * / ** %`, comparisons,
`and`, `or`, `not`, `x in [1, 2]`, `a if cond else b`, `True`/`False`. They
run in order, can update an existing variable, and return one value per row or
a single value (`avg = mean(profit)`). R or Excel spellings (`<-`, `^`, `&`,
`|`, `!`, `TRUE`) are refused with the Python to use instead.

Functions (`pyrsm.decide.simfuns`, same names as the data filter and polars):

- Per row: `ifelse(cond, a, b)`, `abs`, `sqrt`, `exp`, `log`/`ln`, `log10`,
  `floor`, `ceil`, `round`, `clip`, `min_horizontal`, `max_horizontal`,
  `sum_horizontal`, `mean_horizontal`, `cum_sum`, `cum_prod`, `cum_max`,
  `cum_min`, `shift(x, n)`.
- One value: `sum`, `mean`, `median`, `min`, `max` (over all inputs), `std`,
  `var`, `count`, `first`, `last`, `n_unique`, `quantile(x, q)`, `corr`,
  `cov`, `arg_max`, `arg_min` (row number: `price[arg_max(profit)]`),
  `weighted_sum_std(w1, ..., x1, ...)` (sd of a weighted sum, per row of weights).
- Queues: `queue_wait(gap, service, servers)`, `queue_length(gap, service, servers)`.
- Random draws: `normal`, `uniform`, `binomial`, `poisson`, `exponential`,
  `lognormal`, `choice` (first argument: the number of draws).
- Indexing from 0: `wealth[-1]` is the last row.

### Custom functions

Ordinary Python functions written with `def` (`if`, `for`/`while`,
`return`, `x[i] = ...`) using the same function library; pass them in the
spec under `functions`. In a simulation they run in a safe interpreter (no
imports, attributes, strings or `__` names; a step budget stops runaway
loops). `from pyrsm.decide.simfuns import zeros, ifelse` lets them run on
their own for testing. Prefer whole-column formulas (`cum_prod`,
`queue_wait`) for long simulations; loops are for path-dependent logic.

```python
from pyrsm.decide.simfuns import zeros, ifelse

def calc_wealth(win, start, perc, mult):
    wealth = zeros(len(win))
    current = start
    for day in range(len(win)):
        bet = perc * current
        current = ifelse(win[day] == 1, current + mult * bet, current - bet)
        wealth[day] = current
    return wealth
```

## Repeated simulation

```python
rsm.decide.repeat_simulate(
    spec, reps=365, vars=["E", "price"], sum_vars=["profit"],
    by="sim", fun=["sum"], formulas=["low = profit_sum < 36500"],
)
```

- `vars`: re-simulated each repetition (others keep their first values); every
  formula is re-evaluated.
- `by="sim"`: one value per simulation row across repetitions (e.g. 1,000
  simulated years of 365 days); `by="rep"`: one value per repetition across rows.
- `fun`: any of sum, mean, median, min, max, std, var, std_prop, var_prop, p01,
  p025, p05, p10, p25, p75, p90, p95, p975, p99, first, last, or "none".
  Columns are `{variable}_{function}`.
- `functions`: Python functions for the repeat formulas (the simulation's are available too).
- `grid` (by="rep"): `[{"name": "tellers", "min": 1, "max": 4, "step": 1}]`
  replaces a constant in each repetition; the number of grid values sets `reps`.

In Radiant the actions are `simulate.run` (`spec`, `dec`) and
`simulate.repeat.run` (`spec`, `reps`, `vars`, `sum_vars`, `by`, `fun`,
`formulas`, `functions` (source text), `grid`, `seed`, `name`, `dec`); the
simulated and repeated data are saved as datasets (`spec.name`, `name`).

## Examples

`rsm.decide.simulate_examples()`: profit, optimal-price, bank-queue,
gamblers-ruin, stock-returns, portfolio (notebooks in
`examples/decide/simulate-tutorial`).

## Interpretation

- Lead with the decision-relevant number: expected profit, a probability
  (the share of `True` for a logical), or a tail quantile.
- Separate decision levers (constants, grid values) from uncertainty.
- Say what the model leaves out (competitor reactions, correlations assumed
  away, a queue's warm-up period).
- Match the horizon: daily risk can be large while a year's total is safe.
