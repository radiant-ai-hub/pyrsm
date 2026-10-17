---
name: pyrsm-dtree
description: Build, validate, solve, and interpret decision trees with pyrsm.model.dtree
---

# Decision-tree analysis (`pyrsm.model.dtree`)

This skill teaches how to construct a decision tree in YAML, calculate
expected monetary value (EMV) via expected-value rollback, and interpret
the recommended decision. It mirrors Radiant's classic `dtree()` analysis
with one important difference: the Python engine is built around typed
node objects, structured diagnostics, and a safe expression evaluator
instead of free-form `eval`.

## When to use

Reach for `dtree` whenever a student needs to choose between several
courses of action whose outcomes depend on uncertain events. Typical
prompts include:

- "Should we sign with the movie company or the TV network?"
- "Compare drilling vs not drilling given a survey result"
- "Pick the option with the highest expected payoff"

Do **not** use a decision tree for prediction problems (use
`pyrsm.model.regress`, `logistic`, `rforest`, etc.) or for general
exploratory analysis.

## Tree vocabulary

| Node kind | YAML marker | Meaning |
| --------- | ----------- | ------- |
| Decision  | `type: decision` | A choice you control. Children are options. The decision node's payoff is `max` (or `min`) of its options. |
| Chance    | `type: chance`   | Random event. Children are outcomes with probabilities `p:`. The chance node's payoff is the probability-weighted average of its children. |
| Terminal  | (no `type:`, no children) | A leaf with a numeric `payoff:`. |

Each non-root node can also carry a `cost:`. Cost is subtracted from the
rolled-up payoff of that node.

## Minimal example

```yaml
name: Sign contract
type: decision
Sign with Movie Company:
    type: chance
    Small Box Office:
        p: 0.3
        payoff: 200000
    Medium Box Office:
        p: 0.6
        payoff: 1000000
    Large Box Office:
        p: 0.1
        payoff: 3000000
Sign with TV Network:
    payoff: 900000
```

```python
import pyrsm as rsm

tree = rsm.model.dtree(yaml_text, opt="max")
tree.summary(input=False, output=True)
print(tree.solution_df)
print(tree.to_mermaid(final=True))
```

`tree.payoff` gives the root EMV. `tree.normalized.chosen_child_ids`
gives the optimal branches. `tree.to_yaml()` re-serializes the parsed
tree for storage.

## Variables and arithmetic

Reusable constants go in a `variables:` block. Later variables may
reference earlier ones, and `p`, `cost`, and `payoff` may reference any
variable by name:

```yaml
variables:
    legal fees: 5000
    p_small: 0.3
    p_medium: 0.6
    p_large: 1 - p_small - p_medium
```

Allowed operators: `+`, `-`, `*`, `/`, `**`, parentheses, unary `+`/`-`.
Anything else (function calls, attribute access, list indexing) is
rejected. This is enforced by a small AST evaluator, not Python `eval`.

## Calculating

```python
tree = rsm.model.dtree(yaml_text, opt="max")
```

Roll-back rules:

- **Terminal payoff** = `payoff - cost` (if `cost` is set on the leaf;
  prefer folding cost into `payoff` for single-leaf situations).
- **Chance payoff** = `sum(p_i * child_payoff_i) - cost`.
- **Decision payoff** = `max(child_payoff)` (or `min` if `opt="min"`),
  minus any `cost` on the decision node itself.
- Ties at a decision node are preserved in `node.chosen_child_ids` and
  flagged with a `decision_tie` warning.

## Reading diagnostics

`tree.diagnostics` returns a list of `Diagnostic(severity, code, message,
path)` records. The common codes:

| Code | Severity | Meaning |
| ---- | -------- | ------- |
| `chance_probabilities_not_one` | error | A chance node's `p` values do not sum to 1. |
| `chance_probability_out_of_range` | error | A `p` is outside `[0, 1]`. |
| `chance_child_missing_probability` | error | A chance branch has no `p:`. |
| `terminal_missing_payoff` | error | A leaf has no `payoff:`. |
| `unresolved_variable` | error | A `p`/`cost`/`payoff` references a name that is not in `variables:`. |
| `unsafe_expression` | error | A `p`/`cost`/`payoff` uses unsupported syntax (e.g., a function call). |
| `duplicate_sibling_label` | error | Two children of the same parent share a label. |
| `missing_node_type` | warning | A node has children but no `type:`; defaulted to `decision`. |
| `terminal_has_cost` | warning | Terminal uses `cost:`. Prefer folding into `payoff`. |
| `decision_tie` | warning | A decision node has more than one optimal branch. |

A tree with any error-severity diagnostic does not roll back —
`tree.is_solved` is `False` and `tree.payoff` is `None`. Surface the
diagnostics in the UI rather than silently swallowing them.

## Probability tables (imperfect information)

When a test, survey or study is not perfect, the problem gives the prior
P(state) and how often the test reads each way in each state,
P(signal | state). The branches after the test need P(signal) and
P(state | signal). Fill them in with tables, not a formula from memory:

```python
pt = rsm.decide.prob_table(
    {"S": 0.6, "F": 0.4},                                            # P(state)
    {"S": {"Pos": 0.8, "Neg": 0.2}, "F": {"Pos": 0.1, "Neg": 0.9}},  # P(signal | state)
    state="store", signal="popup",
)
pt.summary()            # conditional, joint (Total row = P(signal)), posterior tables
pt.values["P(S|Pos)"]   # 0.9231
print(pt.variables())   # the variables: lines for the tree
```

1. **Joint table:** P(signal and state) = P(signal | state) x P(state).
   Its column totals are P(signal) (here P(Pos) = 0.52, P(Neg) = 0.48);
   its row totals are the prior.
2. **Posterior table:** divide each joint cell by its column total:
   P(S | Pos) = 0.48 / 0.52 = 0.9231. Each column adds up to 1.

`pt.variables()` writes the given probabilities as numbers and the rest
as formulas (`P(Pos and S): P(Pos|S) * P(S)`, `P(S|Pos): P(Pos and S) /
P(Pos)`), so a sensitivity analysis on `P(S)` updates the posteriors.
The usual mistake is to use P(Pos | S) (0.8, how good the test is) where
the tree needs P(S | Pos) (0.92, what a positive result tells you).

## Trees in parts (sub-trees)

A large tree repeats the same stage many times. Write each stage once as
its own tree and use it as a branch (the `product-launch` example):

```yaml
# tree "launch"
name: Launch
variables: product launch      # use the variables of the main tree
cost: fixed cost
type: chance
High demand:
    p: P(high)
    payoff: margin * sales high
Low demand:
    p: 1 - P(high)
    payoff: margin * sales low
```

```yaml
# in tree "test market": the same stage after a positive test
Positive result:
    p: P(pos)
    type: decision
    Launch:
        tree: launch
        with:
            P(high): P(high|pos)   # this use only
    Do not launch:
        payoff: 0
```

- `tree: name` puts tree `name` at that branch; the combined tree is
  solved, printed and plotted as one. The branch keeps its label and
  `p:`; a `cost:` on it is added to the sub-tree's first node. A branch
  with `tree:` has no `payoff:`, `type:` or branches of its own.
- `with:` sets values for that use only, evaluated in the tree that uses
  it. Use it for posteriors (`P(high): P(high|pos)`) and for a variant of
  a stage (a partner's share of the payoff).
- `variables: name` (a tree name instead of a list) uses another tree's
  variables, so every part can also be calculated on its own.
- `dtree("name")` in `variables:` is the EMV of another tree (Radiant's
  older way); `tree:` keeps the sub-tree visible in the solved tree.

Build the parts bottom-up: the last stage first, calculate it, check its
EMV by hand, then the stage that uses it. In Python, pass the other trees
as `subtree_library`:

```python
ex = rsm.decide.dtree_example("product-launch")
tree = rsm.decide.dtree(ex["trees"]["product launch"], subtree_library=ex["trees"])
tree.payoff     # 1380000
tree.tree_uses  # every tree: branch, where it is used and its with: values
```

## Interpretation

When explaining the result to a student:

1. **State the chosen branch.** "The optimal decision is *Sign with
   Movie Company* because its EMV is \$955,000 vs \$900,000 for the TV
   Network branch."
2. **Show the components.** Pull from `tree.solution_df` to show each
   branch's EMV and the probability-weighted contributions of chance
   outcomes.
3. **Mention sensitivity.** Emphasize that the result depends on the
   probabilities and payoffs the student entered. If small changes in a
   probability flip the decision, the recommendation is fragile.
4. **Do not over-claim.** EMV ignores risk preferences. Two branches
   with similar EMV but very different variance may warrant a
   different choice in practice.

## Common errors and fixes

- **Probabilities sum to 0.95.** Add the missing branch or correct the
  probabilities. If three outcomes share the remaining 5%, split it
  explicitly.
- **`p: .4`.** YAML accepts this, but for clarity prefer `0.4`.
- **`cost:` on a terminal.** Replace with an adjusted `payoff:`.
- **Variable name overlaps with text in a payoff expression.** Pick a
  more specific variable name (e.g., `legal_fees_pct` instead of `p`).

## Improving tree structure

- Use chance nodes with at most 4–5 mutually exclusive outcomes; trees
  with many branches per chance node are fragile and hard to validate.
- Push costs as high in the tree as possible. A `cost:` on a parent
  that applies to all children is more compact than repeating the same
  number on each terminal.
- Name decisions and chance outcomes the way a student would describe
  them aloud — these labels appear in `solution_df` and the Mermaid
  graph and serve as the analysis narrative.
