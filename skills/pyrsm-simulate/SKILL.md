---
name: pyrsm-simulate
description: Build, run, and interpret Monte Carlo simulations with pyrsm.model.simulate
---

# Monte Carlo simulation (`pyrsm.model.simulate`)

This skill teaches how to express an uncertain business problem as a
Monte Carlo workbook of named distributions plus derived formulas, run
thousands of draws, and turn the per-column summary into a decision-
grade interpretation. It is the standard tool whenever a Radiant student
needs to reason about uncertainty, risk, or repeated outcomes without
fitting a statistical model.

## When to use

Reach for `simulate` when the prompt names uncertainty and asks for a
probability, a distribution, or a robust choice:

- "What price should we charge if demand is uncertain?"
- "What is the probability annual profit is negative?"
- "Compare three staffing levels under random arrivals."
- "How much capital do we need to cover the worst week with 95%
  confidence?"

Do **not** use Monte Carlo when:

- The problem is a clean decision tree with a few discrete outcomes — use
  `pyrsm.model.dtree` and roll back expected values exactly.
- The problem is a fit/predict task — use `regress`, `logistic`,
  `rforest`, or `xgboost`.

## Workbook vocabulary

A simulation is a small JSON-safe workbook with two ordered sections.
The deterministic planner reasons about both sections so the question
cards reference variable / formula names directly.

### `variables`

Each entry is one named random (or constant) input. Use the smallest
distribution that is honest about the uncertainty:

| `kind` | Required fields | Use for |
| ------ | --------------- | ------- |
| `constant` | `value` | Knowns, base parameters, scenario flags |
| `normal` | `mean`, `sd` | Symmetric noise, additive errors |
| `uniform` | `low`, `high` | "Anywhere in this range, no preference" |
| `binomial` | `n`, `p` | Counts of successes in `n` independent trials |
| `poisson` | `mean` | Counts of independent arrivals/events |
| `lognormal` | `mean`, `sd` | Strictly positive, right-skewed magnitudes |
| `discrete` | `values`, `probs` | Named scenarios with explicit probabilities |
| `sequence` | `values` | Deterministic per-period sweeps in repeated runs |

Probabilities on a `discrete` variable should sum to 1.

### `formulas`

Each formula is `{name, expr}` and is evaluated in order against the
sampled columns. Expressions support `+ - * / **`, parentheses, the
standard math functions, comparisons (`< <= > >= == !=`) and logical
operators. Comparisons produce boolean columns that double as risk
flags. The compiler rejects attribute access, imports, and arbitrary
function calls, so formulas are safe to evaluate.

Pattern: lift every meaningful intermediate (demand, revenue, cost,
profit) into a named formula. Lifting makes the planner's cards
specific and lets the user click straight through to a sensitivity grid
on the right variable.

## Minimal example

```yaml
name: Pricing demo
runs: 5000
seed: 1234
variables:
  - {name: price, kind: discrete, values: [6, 8], probs: [0.3, 0.7]}
  - {name: demand_error, kind: normal, mean: 0, sd: 100}
  - {name: var_cost, kind: constant, value: 5}
  - {name: fixed_cost, kind: constant, value: 1000}
formulas:
  - {name: demand, expr: 1000 - 50 * price + demand_error}
  - {name: profit, expr: demand * (price - var_cost) - fixed_cost}
  - {name: loss,   expr: profit < 0}
```

After `simulate.run`:

- `summary_rows` shows `mean`, `sd`, `p5`, `p50`, `p95`, and (for
  boolean formulas) the probability of `True`.
- `histograms` provides per-column bin counts the UI renders directly.
- `python_code` is the Polars-native reproducibility text.

## Repeated simulation

Use `simulate.repeat.run` when the question mentions accumulation over
time or a portfolio of independent trials. Two questions tell the
planner what to do:

1. What is the period count? (12 months, 52 weeks, 100 trials.)
2. Which variables should resample each period vs. stay fixed for the
   whole horizon?

```json
{
  "type": "simulate.repeat.run",
  "spec": {"...": "same workbook"},
  "periods": 12,
  "resample": ["demand_error"]
}
```

Repeat aggregates each formula across periods so totals, averages, and
risk flags are first-class outputs.

## Probability of loss / risk flags

The canonical risk question is "What is the probability that profit is
negative?" Express it as a boolean formula and read the summary:

```yaml
formulas:
  - {name: profit, expr: revenue - cost}
  - {name: loss,   expr: profit < 0}
```

The pro-builder planner promotes this pattern automatically when the
prompt mentions risk, loss, or threshold language.

## Decision variables and grid search

When the workbook contains a controllable input (price, staffing,
capacity), surface it as a `constant` variable so the user can sweep it.
The planner proposes a sensitivity / grid action when:

- a controllable constant is present, and
- a risk metric or profit formula already exists.

Grid search is not yet a built-in pyrsm action; the planner asks for
realistic min/max/step and the host code stitches together a series of
`simulate.run` calls per cell.

## Queueing / process simulation (future submode)

If the prompt mentions arrivals, waiting, service times, capacity,
abandonment, or shifts, the planner should classify it as a future
process/queue submode and ask:

1. What is the arrival rate / pattern?
2. What is the service time distribution?
3. How many servers / what capacity?
4. What is the service target (avg wait, p90 wait, abandonment)?

Do not expose SimPy or Ciw abstractions directly; the future submode
will ship its own typed action and pyrsm helpers.

## Interpretation rules

- Lead with the most decision-relevant number — expected profit, risk
  probability, or a quantile that names the bad tail.
- Separate decision controls (constants the user can change) from
  exogenous uncertainty (distributions the user cannot change).
- Name explicitly what the simulation does NOT capture (competitor
  reactions, regime changes, correlation the workbook assumes away).
- Recommend repeated simulation when accumulation is implied even if
  the user did not ask for it directly.
