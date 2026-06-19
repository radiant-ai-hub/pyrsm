# pyrsm.basics.prob_calc — reference

This file is the deeper reference for `pyrsm.basics.prob_calc`. The main `SKILL.md` walks the workflow at a high level; come here for per-distribution parameter details, worked examples, and edge cases.

## Table of contents

1. Constructor signature and the 10 distributions
2. `summary()` — what each option prints
3. `plot()` — what each plot shows
4. Output attributes
5. The four bounds parameters — `lb`, `ub`, `plb`, `pub`
6. Forward (value→probability) vs inverse (probability→value)
7. One-sided vs two-sided framings
8. Discrete `<` vs `<=` gotcha
9. Per-distribution parameter reference
10. Worked examples (from the notebook)
11. Common pitfalls

---

## 1. Constructor signature and the 10 distributions

```python
rsm.basics.prob_calc(
    distribution,           # one of: "binom", "chisq", "disc", "expo", "fdist",
                            #         "lnorm", "norm", "pois", "tdist", "unif"
    **kwargs,               # distribution-specific parameters + bounds
)
```

The distribution string is required. Everything else is keyword-only, passed via `**kwargs` to the appropriate distribution-specific function.

### The 10 distributions

| String | Distribution | Required params | Optional params |
| --- | --- | --- | --- |
| `"binom"` | Binomial | `n`, `p` | `lb`, `ub`, `plb`, `pub` |
| `"chisq"` | Chi-squared | `df` | `lb`, `ub`, `plb`, `pub` |
| `"disc"` | Discrete | `v` (values), `p` (probs) | `lb`, `ub`, `plb`, `pub` |
| `"expo"` | Exponential | `rate` | `lb`, `ub`, `plb`, `pub` |
| `"fdist"` | F | `df1`, `df2` | `lb`, `ub`, `plb`, `pub` |
| `"lnorm"` | Log-normal | `mean`, `sd` | `lb`, `ub`, `plb`, `pub` |
| `"norm"` | Normal | `mean`, `sd` | `lb`, `ub`, `plb`, `pub` |
| `"pois"` | Poisson | `lamb` | `lb`, `ub`, `plb`, `pub` |
| `"tdist"` | t | `df` | `lb`, `ub`, `plb`, `pub` |
| `"unif"` | Uniform | `min`, `max` | `lb`, `ub`, `plb`, `pub` |

For all distributions, **at least one** of `lb`, `ub`, `plb`, `pub` is typically passed; calling with none returns the distribution summary (mean, sd) only.

## 2. `summary()` — what each option prints

```python
pc.summary(
    dec=3,                 # decimal places
    ret=False,             # if True, returns the summary dict instead of printing
)
```

Output structure:

- **Header line**: `"Probability Calculator"`.
- **Distribution metadata**: distribution name, parameters (`n`, `p`, `df`, `mean`, `sd`, `lamb`, etc.).
- **Distribution moments**: `Mean` and `St. dev` (or `Variance`) where applicable.
- **Bound rows**: `Lower bound` and `Upper bound` echo the user input (value or probability).
- **Probability section**: one or more `P(X <op> <value>) = <prob>` lines, depending on the question type:
  - For discrete distributions: pairs of `<` / `<=` and `>` / `>=` lines.
  - For continuous distributions: just `<` and `>` (strict and non-strict are equal for continuous).
  - For value+value bounds: a `P(<lb> < X < <ub>)` line.
  - For probability+probability bounds: shows the cutoff values and `1 - P(bounds)`.

`ret=True` returns the underlying dict for programmatic use; `False` (default) prints.

## 3. `plot()` — what each plot shows

```python
pc.plot()
```

Returns a plotnine ggplot showing the distribution's pdf/pmf with the named region(s) shaded:

- For value bounds: the probability mass below `lb` and/or above `ub` is colored.
- For probability bounds: the cutoff value(s) are marked with vertical lines, and the region(s) corresponding to the probability are shaded.

For discrete distributions, the plot is a bar chart; for continuous, a density curve.

The plot is the easiest way to check whether the chosen bounds parameter answers the question the user asked.

## 4. Output attributes

| Attribute | Type | Meaning |
| --- | --- | --- |
| `pc.distribution` | str | The distribution string ("binom", "norm", etc.). |
| `pc.dct` | dict | The computed probability/value summary as returned by the distribution function. Keys include the input parameters, the bound values/probabilities, and computed probabilities. |
| `pc.args` | dict | The keyword arguments passed to the constructor. |

The `dct` keys vary by distribution but typically include: `mean`, `sd`/`stdev`/`variance`, `lb`, `ub`, `plb`, `pub`, and probability/value outputs.

## 5. The four bounds parameters — `lb`, `ub`, `plb`, `pub`

| Parameter | Type | Question type | What you specify | What pyrsm computes |
| --- | --- | --- | --- | --- |
| `lb` | value | forward | a value `x` | probabilities P(X < x), P(X > x) |
| `ub` | value | forward | a value `x` | probabilities P(X < x), P(X > x) |
| `plb` | probability | inverse | a probability `p` | value `x` such that P(X < x) = p |
| `pub` | probability | inverse | a probability `p` | value `x` such that P(X < x) = p (interpret as upper-tail cutoff) |
| `lb`, `ub` | both values | forward, between | two values | P(lb < X < ub) |
| `plb`, `pub` | both probs | inverse, two-sided | two probabilities | two cutoff values (typically symmetric for symmetric distributions) |

### Crucial mnemonic

The `p` prefix on `plb`/`pub` reminds you the input is a **probability**. With `lb` / `ub`, you pass a value; with `plb` / `pub`, you pass a probability.

A common mistake: asking "what value cuts off the lower 5% tail?" with `lb=0.05`. This says "compute the probability of being below 0.05", which is almost always not what was meant. The correct call is `plb=0.05`, which returns the value cutting off 5% below.

### Worked illustrations

```python
# Forward: P(X < 1.96) for standard normal
pc = rsm.basics.prob_calc("norm", mean=0, sd=1, ub=1.96)
# Output: P(X < 1.96) = 0.975
```

```python
# Inverse: value x with P(X < x) = 0.975 for standard normal
pc = rsm.basics.prob_calc("norm", mean=0, sd=1, pub=0.975)
# Output: cutoff value = 1.960
```

These are the same fact, asked two different ways. Pick the right parameter based on which side is known.

## 6. Forward (value→probability) vs inverse (probability→value)

### Forward — when you have a value and want a probability

Examples:
- "P(survived count >= 15 out of 20 trials at p=0.9)" → `binom`, `n=20`, `p=0.9`, `ub=15` (binomial with `>=`).
- "P-value for an observed chi-sq of 32.84 on 1 df" → `chisq`, `df=1`, `ub=32.84`, read the `P(X > 32.84)` line.
- "P(demand > 4000) for a normal with mean 3000, sd 800" → `norm`, `mean=3000`, `sd=800`, `lb=4000`, read `P(X > 4000)`.

### Inverse — when you have a probability and want a value

Examples:
- "What demand level gives 95% service?" → `norm`, `mean=3000`, `sd=800`, `plb=0.95`. (Holds 95% of probability *below* the cutoff.)
- "Critical t for one-sided test at α=0.05 with df=571" → `tdist`, `df=571`, `pub=0.95`.
- "Critical chi-sq at α=0.05 with df=1" → `chisq`, `df=1`, `pub=0.95`.

### Visual heuristic

If you sketch the distribution and shade the region your question asks about:
- The shaded region is given a **value** boundary → forward, use `lb`/`ub`.
- The shaded region is given a **probability** size → inverse, use `plb`/`pub`.

## 7. One-sided vs two-sided framings

When using `prob_calc` for hypothesis-test critical values, match the bounds parameter(s) to the test direction.

### One-sided "greater" test at α

```python
# Reject when test statistic exceeds upper cutoff
pc = rsm.basics.prob_calc(dist, ..., pub=1 - alpha)
# Returns the upper cutoff value
```

### One-sided "less" test at α

```python
# Reject when test statistic falls below lower cutoff
pc = rsm.basics.prob_calc(dist, ..., plb=alpha)
# Returns the lower cutoff value
```

### Two-sided test at α

```python
# Reject when |test statistic| exceeds the symmetric cutoffs
pc = rsm.basics.prob_calc(dist, ..., plb=alpha/2, pub=1 - alpha/2)
# Returns both cutoff values
```

### Chi-squared and F (always upper-tail rejection)

Chi-squared and F test statistics are always non-negative; rejection is one-sided upper-tail.

```python
pc = rsm.basics.prob_calc("chisq", df=df_val, pub=1 - alpha)
# Returns the upper cutoff
```

## 8. Discrete `<` vs `<=` gotcha

For **discrete** distributions (`binom`, `disc`, `pois`):

- `P(X < x)` ≠ `P(X <= x)`. They differ by exactly `P(X = x)`.
- The summary prints both lines.

For **continuous** distributions (`norm`, `chisq`, `tdist`, `fdist`, `expo`, `unif`, `lnorm`):

- `P(X < x) = P(X <= x)` because P(X = x) = 0 for continuous distributions.
- The summary prints just `<` and `>` (no `<=` / `>=`).

### Pedagogical example — binomial battery

The notebook example asks "P(15 or more laptops succeed | n=20, p=0.9)". "15 or more" is `>= 15`, not `> 15`. Read the `P(X >= 15) = 0.989` line, not `P(X > 15) = 0.957` (which would be "16 or more").

This is a frequent student error. Always confirm in plain English: "15 *or more*" = `>=`; "*more than* 15" = `>`.

## 9. Per-distribution parameter reference

### Binomial (`"binom"`)

```python
prob_calc("binom", n=<int>, p=<0..1>, [lb|ub|plb|pub])
```

Models the count of successes in `n` independent yes/no trials. `p` is the probability of success per trial.

Discrete; pmf has support `{0, 1, ..., n}`.

### Chi-squared (`"chisq"`)

```python
prob_calc("chisq", df=<int>, [lb|ub|plb|pub])
```

Continuous, non-negative support. Used for chi-squared independence / goodness-of-fit / variance tests.

### Discrete (`"disc"`)

```python
prob_calc("disc", v=<list>, p=<list>, [lb|ub|plb|pub])
```

User-specified discrete distribution. `v` is the list of possible values, `p` is the list of probabilities (must sum to ~1.0).

### Exponential (`"expo"`)

```python
prob_calc("expo", rate=<float>, [lb|ub|plb|pub])
```

Continuous, non-negative support. Models waiting times in a Poisson process with rate `rate`. Mean = 1/rate; sd = 1/rate.

### F (`"fdist"`)

```python
prob_calc("fdist", df1=<int>, df2=<int>, [lb|ub|plb|pub])
```

Continuous, non-negative support. Used for F-tests in ANOVA and regression. `df1` is numerator df, `df2` is denominator df.

### Log-normal (`"lnorm"`)

```python
prob_calc("lnorm", mean=<float>, sd=<float>, [lb|ub|plb|pub])
```

Continuous, positive support. `mean` and `sd` are of the **underlying normal** (i.e., `log(X) ~ Normal(mean, sd)`).

### Normal (`"norm"`)

```python
prob_calc("norm", mean=<float>, sd=<float>, [lb|ub|plb|pub])
```

Continuous, infinite support. The familiar bell curve. For standard normal, use `mean=0, sd=1`.

Some legacy code uses `stdev=` instead of `sd=`; both keywords are accepted in some branches but `sd` is the documented one.

### Poisson (`"pois"`)

```python
prob_calc("pois", lamb=<float>, [lb|ub|plb|pub])
```

Discrete, non-negative support. Models count of events in a fixed window with average rate `lamb`. Mean = `lamb`, variance = `lamb`.

Note the parameter is `lamb`, not `lambda` (which is a Python reserved word).

### t (`"tdist"`)

```python
prob_calc("tdist", df=<float>, [lb|ub|plb|pub])
```

Continuous, infinite support. Used for t-tests; converges to standard normal as df → ∞.

`df` can be non-integer (e.g., Welch's t-test gives non-integer df).

### Uniform (`"unif"`)

```python
prob_calc("unif", min=<float>, max=<float>, [lb|ub|plb|pub])
```

Continuous, bounded support `[min, max]`. All values equally likely.

## 10. Worked examples (from the notebook)

### Batteries — binomial forward

> "Manufacturer claims >90% of batteries last 12 hours. Test 20 batteries. What's the probability 15 or more succeed?"

```python
pc = rsm.basics.prob_calc("binom", n=20, p=0.9, ub=15)
pc.summary()
# Mean: 18.0, St. dev: 1.342
# P(X = 15) = 0.032, P(X < 15) = 0.011, P(X <= 15) = 0.043,
# P(X > 15) = 0.957, P(X >= 15) = 0.989
```

The answer is `P(X >= 15) = 0.989` (15 *or more* = `>=`).

### Headphones — normal inverse (service level)

> "Demand is normal(3000, 800). Stock how many to achieve 95% service level?"

```python
pc = rsm.basics.prob_calc("norm", mean=3000, sd=800, plb=0.95)
pc.summary()
# Cutoff value 4315.883
# P(X < 4315.883) = 0.95
```

Stocking 4316 units holds 95% of demand probability below the inventory level — i.e., a 95% service level.

### Ice cream — discrete summary + tail

> "Cup purchases follow {1:0.4, 2:0.3, 3:0.2, 4:0.1}. What's the mean, sd, and P(X >= 2)?"

```python
pc = rsm.basics.prob_calc("disc", v=[1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1], lb=2)
pc.summary()
# Mean: 2.0, St. dev: 1.0
# P(X >= 2) = 0.6
```

### Critical t for hypothesis test

> "One-sided t-test, α=0.05, df=571. Critical value?"

```python
pc = rsm.basics.prob_calc("tdist", df=571, pub=0.95)
pc.summary()
# Cutoff value 1.648
```

### Critical chi-squared

> "Chi-squared test, α=0.05, df=1. Critical value?"

```python
pc = rsm.basics.prob_calc("chisq", df=1, pub=0.95)
pc.summary()
# Cutoff value 3.841
```

### P-value from observed chi-squared

> "Observed chi-sq = 187.78 on 1 df. P-value?"

```python
pc = rsm.basics.prob_calc("chisq", df=1, ub=187.78)
pc.summary()
# P(X > 187.78) ≈ 0 (< .001)
```

## 11. Common pitfalls

- **`lb=0.05` vs `plb=0.05`.** The most common error. `lb=0.05` computes "probability of being below 0.05"; `plb=0.05` computes "value with 0.05 probability below it". Use the `p` prefix when your input is a probability.
- **Choosing the wrong distribution.** "Count of successes" → Binomial, not Normal (unless n is huge and you want the CLT approximation). "Waiting time" → Exponential, not Normal. "Test statistic for t-test" → tdist, not Normal.
- **`<` vs `<=` for discrete distributions.** "At least 15" is `>= 15`, not `> 15`. Pyrsm prints both; pick the right one.
- **One-sided vs two-sided critical-value framing.** One-sided at α uses `pub=1-α` (or `plb=α`). Two-sided at α uses `plb=α/2, pub=1-α/2`. Sketch the distribution if unsure.
- **Mixing up forward and inverse questions.** "Given X=4, what's P(X>4)?" → forward (`lb=4`). "Given 95% probability below, what's X?" → inverse (`plb=0.95`).
- **Poisson with `lambda=`.** `lambda` is a Python reserved word. Use `lamb=`.
- **Discrete `v`/`p` mismatched lengths or `p` not summing to 1.** Will raise or produce nonsense. Always double-check.
- **Log-normal `mean` and `sd` are of the underlying normal**, not of the log-normal variable itself. If you have the mean and sd of the variable directly, you have to convert: μ_underlying = ln(mean²/√(mean² + sd²)) and σ_underlying = √(ln(1 + sd²/mean²)).
- **Using prob_calc for hypothesis-test p-values when the test already prints them.** `single_mean`, `compare_means`, etc. all print p-values directly. prob_calc is for **practice** or **corroboration**, not the primary report.
- **Forgetting to call `.summary()`.** Just instantiating `pc = prob_calc(...)` does the math but prints nothing. Add `.summary()` (or `.plot()`) to see the result.
