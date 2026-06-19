---
name: pyrsm-prob-calc
description: Compute tail probabilities, critical values, expected values, and standard deviations for common probability distributions in Python using the pyrsm library's `prob_calc` class. Use this skill whenever a student or analyst wants to find a probability given a value, find a value given a probability, look up a critical t / z / chi-squared / F for a hypothesis test, compute service-level inventory thresholds, derive p-values from a test statistic, calculate the mean and variance of a discrete distribution, or work with the pyrsm package for any distributional calculation. Triggers include phrases like "what's the probability that …", "find the critical t at α=0.05", "what value of X gives 95% probability below it", "compute the p-value for chi-sq = 32.84 on 1 df", "binomial probability of 15 or more successes", "normal critical value for service level 95%", "Poisson tail probability", "uniform between a and b", "inventory level for 95% service", or any mention of looking up probabilities or values from the Binomial, Chi-squared, Discrete, Exponential, F, Log-normal, Normal, Poisson, t, or Uniform distributions in a marketing/business analytics class.
---

# pyrsm probability-calculator workflow

This skill walks the user through computing tail probabilities, critical values, and distribution summaries using the `pyrsm.basics.prob_calc` class — across 10 distributions (Binomial, Chi-squared, Discrete, Exponential, F, Log-normal, Normal, Poisson, t, Uniform). It is designed for students learning probability and hypothesis-testing companions in a business/marketing analytics context.

For deep reference on each distribution's parameters, the `lb` vs `ub` vs `plb` vs `pub` mechanics, and worked examples for each distribution, see `references/prob-calc.md`.

## Output policy

Two rules apply to **every** pyrsm call you make in this skill:

1. **Show the full output verbatim.** When you call `.summary()`, `.predict()`, or any pyrsm method that prints results, paste the **complete printed output** into your response — every line of the coefficient table, every statistic, every row. Do NOT abbreviate. Do NOT replace it with a high-level "main takeaways" bullet list. Do NOT skip the table to "save space". The user is reading your response because they want to see the actual numbers, not your summary of them.

2. **Provide the full pedagogical interpretation, not lame bullets.** When walking through results, use the detailed templates in `references/<name>.md`: state hypotheses explicitly, walk every coefficient or cell with its real-world units from the sidecar description, give the significance verdict at the 5% level, use the "holding constant" qualifier (where applicable), and tie back to the business question. A response like "X is positive and significant; Y is positive and significant; Z is not" is exactly the failure mode this skill is built to prevent.

3. **Lead with the requested output.** When the user asks for a specific plot, table, or diagnostic (e.g., "create and interpret PDP", "show me the residual dashboard", "generate PIP", "give me the confusion matrix", "plot the gains chart"), your interpretation must focus on THAT output — not on a re-interpretation of the model's coefficient table or summary. Pi's failure pattern is to say *"I'll generate X and then interpret the fitted results"* and then quietly produce a coefficient walk-through. The user already saw the coefficient interpretation in the previous turn; do not repeat it unless they explicitly ask. Lead with the requested output's content, using the dedicated template for that output if the skill has one (Step 9 for PIP, Step 10 for PDP, etc.).

If your context is genuinely constrained and you cannot reproduce the full output, say so explicitly — never silently abbreviate.


---

## Step 1 — Identify the question type

`prob_calc` does not load data — it computes probabilities and values for parameterized distributions. Before any code, identify which of these the user is asking for:

- **Forward question (values → probability).** "Given a value, what's the probability of being above / below / in this range?" Use `lb` and/or `ub` parameters.
- **Inverse question (probability → values).** "Given a target probability (a service level, a significance threshold), what value cuts off this much probability?" Use `plb` and/or `pub` parameters.
- **Critical value lookup.** "What's the critical t / z / chi-squared / F for α = 0.05?" An inverse question with `plb` or `pub` set to `α` or `1 - α` depending on the test direction.
- **P-value derivation.** "Given an observed test statistic of `<value>`, what's the p-value?" A forward question with `lb` or `ub` set to the test statistic.
- **Distribution summary.** "What are the mean and variance of this discrete distribution?" Use the discrete branch with `v` and `p`.

If the user's question doesn't fit one of these, ask before calling pyrsm.

## Step 2 — Choose the right distribution

This is the **first critical concept** for prob_calc. The math gives sensible-looking output for *any* distribution you pick, so the choice has to come from the *underlying random process*, not from "let's try normal first":

| Distribution | Use when … | Required parameters |
| --- | --- | --- |
| `"binom"` | Count of successes in `n` independent yes/no trials with constant success probability `p`. | `n`, `p` |
| `"chisq"` | Sum of squared standard normals; test statistic for chi-squared independence / goodness-of-fit / variance. | `df` |
| `"disc"` | A discrete random variable with a finite specified support and probability vector. | `v` (list of values), `p` (list of probabilities, sums to 1) |
| `"expo"` | Time between events in a Poisson process; waiting times with constant rate. | `rate` (events per unit time) |
| `"fdist"` | Ratio of two scaled chi-squared variates; test statistic in ANOVA and regression F-tests. | `df1` (numerator), `df2` (denominator) |
| `"lnorm"` | Log of the variable is normal; right-skewed continuous variables (income, prices, returns). | `mean`, `sd` (of the underlying normal) |
| `"norm"` | Continuous variable, bell-shaped, symmetric, infinite support; sample means by CLT. | `mean`, `sd` |
| `"pois"` | Count of events in a fixed window with constant average rate. | `lamb` (mean rate) |
| `"tdist"` | t-statistic in single-sample / two-sample t-tests; small-sample inference about means. | `df` |
| `"unif"` | Continuous, all values equally likely on a bounded interval. | `min`, `max` |

If the user's described process doesn't match one of these (e.g., "rare events with varying rate" → not Poisson), say so before running anything.

### Common mismatches

- **"Successes out of n trials"** → almost always Binomial, **not** Normal. Students sometimes reach for Normal because it's familiar. For large `n` and moderate `p`, Normal is a CLT-based approximation to Binomial — but the exact Binomial is always available and usually correct.
- **"Time between events"** → Exponential. Students sometimes try Normal here; it gives nonsense for short waiting times and is mathematically wrong (Exponential is right-skewed, Normal is symmetric).
- **"Count of events in a window"** → Poisson. Normal approximation only works when `lamb` is large (> ~20).
- **"t-statistic" or "chi-squared statistic" lookup** → use `tdist` or `chisq`, not Normal. These are the right distributions for the test statistics.
- **"Variance / standard deviation test"** → chi-squared, not Normal.

## Step 3 — Lay out the question in plain English

Before writing code, state the question concretely:

> We have a `<distribution>` with parameters `<params>`. We want to compute `<the probability that X is < / > / between …>` **or** `<the value of X for which P(X < / >) = …>`.

This forces the user to commit to a direction and a known/unknown side, which determines which of `lb` / `ub` / `plb` / `pub` to use.

## Step 4 — Pick the right bounds parameter — `lb`/`ub` vs `plb`/`pub`

This is the **second critical concept** for prob_calc, and where most student errors occur.

### `lb` and `ub` — **value** bounds

You specify a value (an `x`); pyrsm computes the probability of being below / above / between.

```python
pc = rsm.basics.prob_calc("norm", mean=0, sd=1, lb=-1.96, ub=1.96)
pc.summary()
# Prints P(X < -1.96), P(X > -1.96), P(X < 1.96), P(X > 1.96), P(-1.96 < X < 1.96)
```

- `lb=<value>` only → "what's the probability below / above this value?"
- `ub=<value>` only → same, but above is the more natural read.
- `lb=<value>, ub=<value>` → between the two values.

### `plb` and `pub` — **probability** bounds

You specify a probability (a target, like 0.05 or 0.95); pyrsm computes the value of `x` for which the tail probability equals your target.

```python
pc = rsm.basics.prob_calc("norm", mean=0, sd=1, pub=0.975)
pc.summary()
# Prints the X value such that P(X < X) = 0.975, i.e., the upper-tail critical z = 1.96
```

- `plb=<prob>` only → "what value has this much probability below it?"
- `pub=<prob>` only → "what value has this much probability below it (interpreting from the upper-tail framing)?"
- `plb=<prob>, pub=<prob>` → two-sided: "what values have `<plb>` and `<pub>` probability below them?"

### The classic mistake

Asking "what value `x` has P(X < x) = 0.05?" with `lb=0.05` — wrong. `lb=0.05` says "compute P(X < 0.05)", which is a value question. To get the inverse, use `plb=0.05`.

Mnemonic:
- **value in → probability out** = `lb` / `ub`.
- **probability in → value out** = `plb` / `pub`.

The `p` prefix on `plb`/`pub` stands for "probability", reminding you the input is a probability.

### One-sided vs two-sided framings

- **One-sided** (e.g., critical value for a one-sided t-test at α=0.05 with `df=571`): `prob_calc("tdist", df=571, pub=0.95)` returns the upper t-cutoff (95% of probability below it = 5% above it).
- **Two-sided** (e.g., critical value for a two-sided t-test at α=0.05): `prob_calc("tdist", df=571, plb=0.025, pub=0.975)` returns both cutoffs symmetric around 0.

When in doubt, sketch the distribution, shade the rejection region, and read off whether you're naming the value(s) or the probability(ies).

## Step 5 — Run prob_calc

```python
import pyrsm as rsm

pc = rsm.basics.prob_calc(
    "<distribution>",         # one of: binom, chisq, disc, expo, fdist, lnorm, norm, pois, tdist, unif
    <distribution-params>,    # see Step 2 table
    lb=...,                   # value lower bound (optional)
    ub=...,                   # value upper bound (optional)
    plb=...,                  # probability lower bound (optional)
    pub=...,                  # probability upper bound (optional)
)

pc.summary()                  # printed table of values + probabilities
pc.plot()                     # plotnine ggplot
```

The summary prints:

- **Distribution metadata** — distribution name and its parameters.
- **Mean, st. dev., variance** where applicable.
- **Lower / Upper bound** — whichever the user supplied (value or probability).
- **Probability section** — one or more `P(X < a) = <p>` lines, automatically arranged for the question type.

For discrete distributions (binom, disc, pois), the printout includes both strict (`<`) and non-strict (`<=`) inequalities, since they differ for discrete cases.

## Step 6 — Interpret the output

This is the third critical concept: **read the summary as the answer to the original question, not as raw data to interpret further**.

### Forward (value → probability)

The relevant lines in the output are the `P(X ...)` rows. Pick the one that matches the question:

- "P(X > 15) = 0.957" — for "the probability that at least 15 succeed but not exactly 15" (use the `>` if you're asking about "more than", or `>=` for "at least").
- For the binomial-battery example (n=20, p=0.9, ub=15): the question was "P(15 or more succeed)" = `P(X >= 15) = 0.989`.

For discrete distributions, **be careful about `<` vs `<=`**. "At least 15" is `P(X >= 15)`, not `P(X > 15)`. The summary prints both.

### Inverse (probability → value)

The relevant lines are the `Lower bound` / `Upper bound` rows, which now report **values** (the inverse output), not probabilities.

- "Service level 95%, normal demand mean 3000 sd 800": `prob_calc("norm", mean=3000, sd=800, plb=0.95)` returns `Lower bound = 0.95` (the user input) and prints `P(X < 4315.883) = 0.95`. So 4316 units holds 95% of the demand-CDF below it — the service-level threshold.
- Critical t for two-sided α=0.05 with df=571: `prob_calc("tdist", df=571, plb=0.025, pub=0.975)` returns the two critical values (`-1.964` and `+1.964`).

### Distribution summary (discrete)

For discrete distributions specified via `v` and `p`, the summary also prints the mean and standard deviation. Cups-of-ice-cream: `prob_calc("disc", v=[1,2,3,4], p=[0.4,0.3,0.2,0.1])` → mean = 2.0, sd ≈ 1.0.

## Step 7 — Plot (offer, often useful)

```python
pc.plot()
```

Returns a plotnine ggplot showing the distribution's pdf/pmf with the relevant region shaded. For value-based questions, the shaded region is the probability mass to the left/right/between the named values. For probability-based questions, the shaded region equals the target probability and the cutoff values are marked.

The plot makes the bounds question concrete: students can see whether they meant "to the left of `lb`" vs "to the right". Always recommend looking at the plot when in doubt about which `P(X …)` line to read.

## Step 8 — Common applications

### Inventory / service-level threshold

> "Demand is normal with mean 3000, sd 800. Stock how many units to achieve a 95% service level?"

```python
pc = rsm.basics.prob_calc("norm", mean=3000, sd=800, plb=0.95)
pc.summary()
# Cutoff value 4316 — stocking 4316 units holds 95% probability below.
```

### Critical value lookup for a hypothesis test

> "What's the critical t for a one-sided test at α=0.05 with df=571?"

```python
pc = rsm.basics.prob_calc("tdist", df=571, pub=0.95)
pc.summary()
# Cutoff value 1.648 — observed t larger than 1.648 → reject.
```

For two-sided α=0.05 instead:

```python
pc = rsm.basics.prob_calc("tdist", df=571, plb=0.025, pub=0.975)
pc.summary()
# Two cutoffs at ±1.964
```

### P-value from observed test statistic

> "Given chi-squared = 32.84 on 1 df, what's the p-value?"

```python
pc = rsm.basics.prob_calc("chisq", df=1, ub=32.84)
pc.summary()
# P(X > 32.84) — the upper-tail probability — is the chi-squared p-value
```

### Binomial-style probability question

> "Manufacturer claims 90% of batteries last 12 hours. In a sample of 20, what's P(15 or more last)?"

```python
pc = rsm.basics.prob_calc("binom", n=20, p=0.9, ub=15)
pc.summary()
# P(X >= 15) = 0.989
```

### Poisson tail

> "On average 4 customer-complaints per day. What's P(2 or fewer tomorrow)?"

```python
pc = rsm.basics.prob_calc("pois", lamb=4, ub=2)
pc.summary()
# P(X <= 2) = 0.239
```

### Discrete summary

> "Customers buy 1/2/3/4 cups with probabilities 0.4/0.3/0.2/0.1. What's the expected value?"

```python
pc = rsm.basics.prob_calc("disc", v=[1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1], lb=2)
pc.summary()
# Mean = 2.0, sd = 1.0, P(X >= 2) = 0.6
```

## Step 9 — Related basics classes

`prob_calc` is most often used **alongside** other basics classes:

- **`single_mean` / `compare_means`** → look up critical t-values with `prob_calc("tdist", df=<df>, pub=1-alpha)` or one of the two-sided variants.
- **`single_prop` / `compare_props`** → critical z with `prob_calc("norm", mean=0, sd=1, pub=1-alpha)`; binomial critical values for exact tests with `prob_calc("binom", n=<n>, p=<p_0>, plb=alpha)`.
- **`goodness` / `cross_tabs`** → critical chi-squared with `prob_calc("chisq", df=<k-1 or (R-1)(C-1)>, pub=1-alpha)`.
- **`regress`** → critical t for individual coefficients and critical F for overall model.

When the user is doing one of those tests, the `prob_calc` task is corroboration / hand-calculation practice. The test itself produces the p-value directly.

---

## Style notes

- **First identify the question type.** Forward (value→probability) or inverse (probability→value)?
- **Then choose the right distribution from the underlying process**, not from familiarity.
- **Then choose `lb`/`ub` (value input) or `plb`/`pub` (probability input).** This is where most errors happen.
- **For discrete distributions, watch `<` vs `<=`.** "At least 15" is `>= 15`, not `> 15`.
- **One-sided vs two-sided framing.** A one-sided critical value for α=0.05 uses `pub=0.95` (or `plb=0.05`). A two-sided uses `plb=0.025, pub=0.975`. Sketch if unsure.
- **The plot makes the bounds concrete.** Recommend `pc.plot()` whenever the user is uncertain about the answer.
- **Use prob_calc as a companion**, not a substitute, for the tests in sibling skills.
- **Don't over-engineer.** Plain top-level code that the student can copy line by line.

## Growing this skill

This skill is one of a family of `pyrsm-basics-*` skills covering the `pyrsm.basics` module. `prob_calc` is the calculator companion for the hypothesis tests in the rest of the family. When the user is running an actual test (single_mean, single_prop, compare_means, compare_props, goodness, cross_tabs, regress), route them to the corresponding skill and use `prob_calc` only for critical-value / p-value lookups in support. The workflow shape — identify question → choose distribution → choose bounds parameter → run → interpret — is unique to this skill, since there's no data loading involved.
