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
    "name": "simdat",          # the saved dataset (radiant.model's "Simulated data")
    "runs": 1000,              # rows (ignored when there are grid variables)
    "seed": 1234,
    "variables": [...],
    "form": "profit = demand * (price - cost)\nloss = profit < 0",
    "functions": "",           # optional: def ... custom functions
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
| `poisson` | `rate` | counts of events (lambda) |
| `exponential` | `rate` | time between arrivals, service times (mean = 1 / rate) |
| `discrete` | `values`, `probs` | scenarios; probabilities may be fractions ("1/3") and must sum to 1 |
| `sequence` | `min`, `max` | evenly spaced over the rows |
| `grid` | `min`, `max`, `step` | every combination of all grid variables; sets the rows |

Fields can be numbers or expressions of earlier names (`"mean": "base * 2"`);
constants are evaluated first.

### Formulas (`form`)

One `name = expression` per line (`<-` works, `#` comments, `;` separates).
They run in order, can update an existing variable, and may return one value
per row or a single value (`avg = mean(profit)`).

- Operators: `+ - * / ^ %%`, comparisons, `&`, `|`, `!` (R style) or `and/or/not`.
- Per row: `ifelse`, `pmin`, `pmax`, `psum`, `pmean`, `pmedian`, `psd`, `pvar`,
  `abs`, `sqrt`, `exp`, `log`/`ln`, `log10`, `square`, `round`, `floor`,
  `ceiling`, `cumsum`, `cumprod`, `cummax`, `cummin`, `lag`, `lead`, `row_number()`.
- One value: `sum`, `mean`, `median`, `min`, `max` (over all inputs, as R),
  `sd`, `var`, `quantile(x, p)`, `prop`, `cor`, `cov`, `n()`,
  `find_max(x, y)` / `find_min(x, y)` (y where x is largest / smallest),
  `sdw(w1, ..., x1, ...)` (sd of a weighted sum, per row of weights).
- Queues: `queue_wait(gap, service, servers)` and
  `queue_length(gap, service, servers)` for a first-come-first-served queue.
- Random draws: `rnorm`, `runif`, `rbinom`, `rpois`, `rexp`, `rlnorm`, `sample`.
- Indexing: `wealth[n() - 1]` (from 0).

### Custom functions (`functions`)

Python syntax (`def`, `if`, `for`/`while`, `return`, `x[i] = ...`) with the
same function library; no imports, attributes (`.`), strings or `__` names.
A step budget stops runaway loops. Prefer vectorised formulas (`cumprod`,
`queue_wait`) for long simulations; loops are for path-dependent logic.

```python
def calc_wealth(win, start, perc, mult):
    wealth = zeros(len(win))
    current = start
    for day in range(len(win)):
        bet = perc * current
        current = current + mult * bet if win[day] == 1 else current - bet
        wealth[day] = current
    return wealth
```

## Repeated simulation

```python
rsm.decide.repeat_simulate(
    spec, reps=365, vars=["E", "price"], sum_vars=["profit"],
    by="sim", fun=["sum"], form="low = profit_sum < 36500",
)
```

- `vars`: re-simulated each repetition (others keep their first values); every
  formula is re-evaluated.
- `by="sim"`: one value per simulation row across repetitions (e.g. 1,000
  simulated years of 365 days); `by="rep"`: one value per repetition across rows.
- `fun`: any of sum, mean, median, min, max, sd, var, sdprop, varprop, p01,
  p025, p05, p10, p25, p75, p90, p95, p975, p99, first, last, or "none".
  Columns are `{variable}_{function}`.
- `grid` (by="rep"): `[{"name": "tellers", "min": 1, "max": 4, "step": 1}]`
  replaces a constant in each repetition; the number of grid values sets `reps`.

In Radiant the actions are `simulate.run` (`spec`, `dec`) and
`simulate.repeat.run` (`spec`, `reps`, `vars`, `sum_vars`, `by`, `fun`, `form`,
`grid`, `seed`, `name`, `dec`); the simulated and repeated data are saved as
datasets (`spec.name`, `name`).

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
