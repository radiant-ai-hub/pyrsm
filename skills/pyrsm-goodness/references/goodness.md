# pyrsm.basics.goodness — reference

This file is the deeper reference for `pyrsm.basics.goodness`. The main `SKILL.md` walks the workflow at a high level; come here for API details, interpretation templates, and worked examples.

## Table of contents

1. Constructor signature
2. `summary()` — what each option prints
3. `plot()` — what each panel shows
4. Output attributes
5. Plain-English interpretation templates
6. Choosing the expected probabilities (`probs`)
7. The expected-count assumption (≥ 5 rule)
8. Standardized deviations and the 1.96 rule
9. Large-n inflation and effect-size context
10. Related basics classes — when to switch
11. Worked example (`newspaper`)
12. Common pitfalls

---

## 1. Constructor signature

```python
rsm.basics.goodness(
    data,                       # polars/pandas DataFrame, OR {"name": df}
    var,                        # categorical column to test
    probs=None,                 # tuple of expected probabilities (sums to ~1); None → uniform
    figsize=None,               # figure size for plots
)
```

**Always pass data as `{"name": df}`** — the `name` becomes the dataset label in the printed summary. Plain `df` works, but the summary prints `"Not provided"`.

`var` must reference a categorical / string / Enum / Categorical column. Pyrsm sorts the levels alphabetically before applying `probs` — so if `var` has levels `"High Income"` and `"Low Income"`, the first slot of `probs` applies to `"High Income"` (H < L). Verify the sorted order by inspecting `gf.observed` after the call: the column order is the sorted order.

`probs` must be a tuple/list of floats with one entry per level of `var` and `sum(probs) ≈ 1.0` (the class accepts `[0.999, 1.001]`). Passing percentages (e.g., `(30, 70)`) raises. Passing a mismatched length also raises.

If `probs=None`, the class defaults to `[1/k] * k` (uniform). This default is convenient but rarely the right substantive null — see §6.

`figsize` is for the plots; safe to ignore unless customizing.

## 2. `summary()` — what each option prints

```python
gf.summary(
    output=["observed", "expected"],   # which tables to print
    dec=3,                              # decimal places
    plain=True,                         # plain text vs styled great_tables (Jupyter)
)
```

`output` is a list of table-name strings:

| `output=` | Prints |
| --- | --- |
| `"observed"` | Observed counts per cell with a total column. |
| `"expected"` | Expected counts (`total × prob`) per cell with a total column. |
| `"chisq"` | Chi-squared contributions: `(observed − expected)² / expected` per cell, total = test statistic. |
| `"dev_std"` | Standardized deviations: `(observed − expected) / sqrt(expected)` per cell. No total. |

You can pass a string instead of a list — `output="chisq"` is fine — but the list form is clearer.

The footer always prints, regardless of `output`:

```
Chi-squared: <value> df (<k-1>), p.value <p>
```

The chi-squared p-value is computed via `scipy.stats.chisquare` using observed and expected counts.

For an initial fit, **pass all four** (`output=["observed", "expected", "chisq", "dev_std"]`) — the tables are small, and each teaches the student something different. Once the student is fluent, `output=["observed", "dev_std"]` is often the most informative compact view.

## 3. `plot()` — what each panel shows

```python
gf.plot(plots="observed")
```

`plots` can be a string (single panel) or a list (composed grid). Options:

| `plots=` | Shows |
| --- | --- |
| `"observed"` | Bar chart of observed counts per level. |
| `"expected"` | Bar chart of expected counts per level. |
| `"chisq"` | Bar chart of chi-squared contributions per level. |
| `"dev_std"` | Bar chart of standardized deviations with horizontal reference lines at ±1.96, ±2.58, etc. |

`plots=["observed", "expected", "chisq", "dev_std"]` returns a composed 4-panel grid.

The `dev_std` panel is the visually most informative for diagnosing *which* cells drive the rejection — any bar whose height exceeds ±1.96 (the outer dotted lines) is individually significant at α = 0.05.

## 4. Output attributes

After the constructor runs:

| Attribute | Type | Meaning |
| --- | --- | --- |
| `gf.observed` | `pl.DataFrame` (1 × (k+1)) | Counts per level + `Total` column. |
| `gf.expected` | `pl.DataFrame` (1 × (k+1)) | `total × probs[i]` per level + `Total`. |
| `gf.chisq` | `pl.DataFrame` (1 × (k+1)) | Chi-squared contributions per level + `Total` (= the test statistic). |
| `gf.stdev` | `pl.DataFrame` (1 × k) | Standardized deviations per level. |
| `gf.freq` | `dict[level, count]` | The raw value-counts dictionary. |
| `gf.nlev` | int | Number of distinct levels. |
| `gf.probs` | tuple[float] | Expected probabilities (uniform fill if input was None). |
| `gf.var`, `gf.data`, `gf.figsize`, `gf.name` | various | Echoes of inputs. |

Note: there is **no `p_val` or `chisq` scalar attribute** on the object — those are computed and printed inside `_summary_footer()` but not stored. If you need them programmatically:

```python
from scipy.stats import chisquare
import numpy as np

sorted_keys = sorted(gf.freq.keys())
observed_counts = [gf.freq[k] for k in sorted_keys]
expected_counts = [gf.expected[k].item() for k in sorted_keys]
chi2_stat, p_val = chisquare(observed_counts, expected_counts)
```

## 5. Plain-English interpretation templates

Use these templates verbatim (substituting variable names and units).

### Hypotheses

> H₀: the distribution of `<var>` in the population is consistent with the specified probabilities `<probs>` (i.e., `<level_1>` should make up `<probs[0]*100>`%, `<level_2>` should make up `<probs[1]*100>`%, …).
> Hₐ: the distribution of `<var>` in the population is **not** consistent with the specified probabilities.

### Sample description

> The sample contains `<n>` observations across `<k>` levels of `<var>`. Observed counts: `<level_1>` = `<count_1>`, `<level_2>` = `<count_2>`, …

### Chi-squared verdict

> Chi-squared = `<chi2>` on `<df = k-1>` degrees of freedom, p-value `<p>`. Because p `<is | is not>` smaller than 0.05, we `<reject | fail to reject>` the null hypothesis. The data `<do | do not>` provide statistically significant evidence that the distribution of `<var>` differs from `<probs>`.

### Identifying which cells drive the rejection

> The standardized deviations are: `<level_1>` = `<dev_1>`, `<level_2>` = `<dev_2>`, …
> Cells with `|dev_std| > 1.96` deviate significantly from expectation at α = 0.05. Here, `<list cells exceeding ±1.96>` are individually significant. Positive deviations mean the cell is over-represented relative to the expected distribution; negative means under-represented.

### Effect size in business terms

> The largest deviation is in `<level>`, where we observed `<obs>` versus the expected `<exp>` (`<dev_std>` standardized units). Whether this magnitude is decision-relevant depends on `<the business context>` — a statistically significant chi-squared with small standardized deviations is mathematically real but may not warrant action.

## 6. Choosing the expected probabilities (`probs`)

This is the **defining critical concept** for goodness-of-fit. The test is only as meaningful as the expected distribution it tests against — and the default uniform distribution is rarely the substantive null.

### When uniform is right

- **Fairness tests.** Testing a die (`probs=(1/6,)*6`), a coin (`probs=(0.5, 0.5)`), or random-assignment buckets.
- **Calendar-balanced samples.** Testing whether each day of the week appears equally often in a year of daily data.
- **Genuinely exploratory.** No substantive prior expectation; the test is a quick "anything sticking out?"

### When uniform is wrong

For almost everything else, the substantive baseline comes from outside the data:

- **Census or population data.** "What proportion of the US population is in each income bracket?" Use census values.
- **Historical baseline.** "What was the share of each product category in last year's sales?" Use last year.
- **Contract or design.** "The contract specifies a 25/25/25/25 split across operators." Use the contractual split.
- **Theoretical model.** "Under Mendelian inheritance we expect 3:1 dominant:recessive." Use `(0.75, 0.25)`.
- **Prior period.** "Did the new ad campaign change the customer mix?" Use the pre-campaign distribution.

If the user says "test whether `Income` is uniform" without a substantive reason for uniform, **push back**: where would the uniform expectation come from? Why is 50/50 the right null for income? It almost certainly isn't.

### Order of `probs`

Levels are sorted alphabetically before `probs` is applied. The easy way to verify:

```python
gf = rsm.basics.goodness({"<name>": df}, var="<col>", probs=(p1, p2, ...))
print(gf.observed)   # column order = sorted-level order = probs order
```

If the order is wrong, the test silently produces nonsense — the expected counts will be assigned to the wrong levels. Always check.

### `probs` must sum to 1

The class checks `0.999 ≤ sum(probs) ≤ 1.001` and raises otherwise. Round-off slop is fine; missing levels are not.

### Don't fit `probs` from the same data

If you derive `probs` from the same sample you're testing (e.g., fit a marginal distribution and then test against it), the degrees of freedom are wrong and the p-value is biased. The expected distribution must be set *a priori*, not estimated from the test sample.

## 7. The expected-count assumption (≥ 5 rule)

The chi-squared test is asymptotic — it relies on the test statistic converging to a chi-squared distribution under H₀. That convergence requires reasonably large expected counts in every cell.

### The rule of thumb

> Every expected cell count should be at least 5.

In pyrsm terms: check `min(gf.expected.drop("Total").row(0))` (the smallest expected cell count). If it's below 5, the chi-squared p-value is biased — typically biased away from rejection (the actual rejection rate under H₀ is higher than the nominal α), but the direction depends on the table shape.

### What to do when violated

- **Collapse adjacent categories** to merge low-count cells with neighbors. For ordinal categories (rating scales, age brackets) this is principled — combine "very poor" with "poor", etc.
- **Compute exact p-values manually.** Use `scipy.stats.multinomial` to enumerate the exact tail probabilities under H₀. This is computationally feasible for small k.
- **Get more data.** Sometimes works; often impossible.
- **Report with caveats.** If you must report the biased p-value, say so explicitly: "the expected count for `<level>` is `<count>`, below the conventional minimum of 5, so the chi-squared approximation may be inaccurate."

### Why pyrsm doesn't auto-collapse

Collapsing is a substantive decision — *which* categories to combine, and how to label the combined level — and pyrsm leaves it to the user. The class will happily compute a biased p-value if you ask it to.

## 8. Standardized deviations and the 1.96 rule

The `dev_std` table contains, for each cell:

```
dev_std = (observed - expected) / sqrt(expected)
```

Under H₀, these standardized deviations are approximately standard normal (mean 0, sd 1). So:

- `|dev_std| > 1.96` → cell is individually significant at α = 0.05.
- `|dev_std| > 2.58` → at α = 0.01.
- `|dev_std| > 3.29` → at α = 0.001.

The `gf.plot(plots="dev_std")` panel shows reference lines at exactly these thresholds, with the 1.96 line as the outermost prominent dashed line.

### Why standardized deviations are the right effect-size view

The omnibus chi-squared statistic only tells you "the distribution is off". It doesn't tell you *where* or *by how much*. The standardized-deviation view does both — large positive bars are over-represented cells, large negative bars are under-represented, and the magnitudes are comparable across cells (unlike raw `(observed − expected)`, which is scale-dependent).

### Multiple-comparisons caveat

If you have k cells and check all k for `|dev_std| > 1.96`, the family-wise error rate is higher than 5%. For exploratory diagnosis this is acceptable, but for *claims* of significance on individual cells (e.g., "the high-income cell is significantly over-represented at α = 0.05"), apply a Bonferroni-style correction: divide α by k. With k = 2, this is α/2 = 0.025 → threshold of |dev_std| ≈ 2.24. The newspaper example with 2 cells survives this comfortably.

## 9. Large-n inflation and effect-size context

The chi-squared test scales with sample size: doubling n approximately doubles the chi-squared statistic for the same deviation pattern. So:

- With n = 100, a 1-percentage-point deviation might give chi-sq ≈ 0.04 (n.s.).
- With n = 1,000, the same pattern gives chi-sq ≈ 0.4 (n.s.).
- With n = 100,000, chi-sq ≈ 40 (p < .001).

The deviation is the same; the p-value is wildly different. With large samples, the chi-squared test reliably rejects any deviation, including ones too small to matter for the business decision.

### Effect-size proxies

- **Standardized deviations.** Already discussed — the natural cell-level effect size.
- **Cramér's V** (for two-way tables, more relevant to `cross_tabs`). For 1-way goodness, a related quantity is `sqrt(chi2 / (n * (k - 1)))`, sometimes called "phi" or Cramér's V for 1-way tables. Values near 0 mean small effect; values near 1 mean large.
- **Practical deltas.** Just look at the raw percentage-point differences between observed and expected. A 5-percentage-point shift is large in most marketing contexts; a 0.5-percentage-point shift is tiny.

Always pair the p-value with at least one of these.

## 10. Related basics classes — when to switch

- **Two-level categorical with a single benchmark proportion** → `single_prop` (more direct API, supports binomial-exact).
- **Two categorical variables, testing for association** → `cross_tabs`.
- **Continuous variable's mean vs a benchmark** → `single_mean`.
- **Probability calculations for the chi-squared distribution** → `prob_calc("chisq", df=<k-1>, ...)`.

## 11. Worked example — `newspaper`

From `examples/basics/basics-goodness.ipynb`:

> "Data are from a sample of 580 newspaper readers indicating which newspaper they read most frequently and their level of income (Low income vs. High income). We will examine if the observed frequencies of income level match the expected (50/50) frequencies."

```python
import polars as pl
import pyrsm as rsm

newspaper = pl.read_parquet("<abs-path>/newspaper.parquet")
gf = rsm.basics.goodness({"newspaper": newspaper}, var="Income")  # probs=None → (0.5, 0.5)
gf.summary(output=["observed", "expected", "chisq", "dev_std"])
```

Output:

```
Goodness of fit test
Data         : newspaper
Variable     : Income
Probabilities: 0.5 0.5
Null hyp.    : The distribution of Income is consistent with the specified distribution
Alt. hyp.    : The distribution of Income is not consistent with the specified distribution

Observed:    High Income=221, Low Income=359, Total=580
Expected:    High Income=290.0, Low Income=290.0, Total=580.0
Chi-sq cells: High Income=16.42, Low Income=16.42, Total=32.84
dev_std:     High Income=-4.05, Low Income=+4.05

Chi-squared: 32.834 df (1), p.value < .001
```

Interpretation walkthrough:

1. **Hypotheses.** H₀: high and low income readers are equally common (50/50 split). Hₐ: not 50/50.
2. **Sample.** n = 580 readers. 221 high-income, 359 low-income.
3. **Tables.** Expected 290/290 under uniform. Observed 221/359 — high-income is 69 below expected, low-income 69 above.
4. **Chi-squared.** Each cell contributes 16.42; total = 32.84 on 1 df. P-value < .001.
5. **Standardized deviations.** Both cells at ±4.05 — well beyond the 1.96 threshold. The deviation is highly significant in both cells.
6. **Verdict.** Reject H₀.
7. **But — is 50/50 the right null?** Probably not. The sample is of newspaper readers, not the general US population, and median splits don't guarantee 50/50 in any sub-population. The substantively interesting null might be "matches US census 60/40" or "matches the previous year's mix". Re-running with a meaningful baseline is what should follow the rejection.

### Non-uniform example

If the US census says 30% high-income, 70% low-income (illustrative numbers, not real census):

```python
gf2 = rsm.basics.goodness({"newspaper": newspaper}, var="Income", probs=(0.3, 0.7))
gf2.summary(output=["observed", "expected", "chisq", "dev_std"])
# Observed:   High=221,    Low=359
# Expected:   High=174.0,  Low=406.0
# Chi-sq:     High=12.70,  Low=5.44, Total=18.14
# dev_std:    High=+3.56,  Low=-2.33
# Chi-squared: 18.14 df (1), p.value < .001
```

The sample still rejects the census null, but in the *opposite* direction: high-income readers are *over*-represented (positive dev_std) relative to the census 30%, while low-income readers are *under*-represented. The substantive conclusion is now meaningful: this newspaper's audience skews higher-income than the general population.

## 12. Common pitfalls

- **Defaulting to uniform `probs`.** Uniform is convenient but rarely substantive. Always ask "where does the expected distribution come from?" before defaulting.
- **Wrong `probs` order.** Levels are sorted alphabetically before `probs` is applied. Verify by printing `gf.observed` and confirming the column order matches what you expect.
- **`probs` not summing to 1.** The class raises; if you passed percentages, divide by 100.
- **Expected counts below 5.** The chi-squared approximation is biased. Either collapse categories or compute the exact p-value via `scipy.stats.multinomial`.
- **Large-n reject-on-anything.** With n in the thousands or more, the test rejects tiny deviations. Always pair the p-value with the standardized-deviation magnitudes and the practical effect size.
- **Reading the omnibus chi-squared without the `dev_std` table.** The omnibus says "something's off"; it doesn't say where. Always look at standardized deviations to identify the driving cells.
- **Deriving `probs` from the same sample.** Biases the p-value. The expected distribution must be set *a priori*.
- **Forgetting the dict wrapper.** `goodness(df, ...)` works but the summary shows `"Not provided"`. Use `goodness({"<name>": df}, ...)`.
- **Treating `goodness` as a test of independence.** It is not. For "are these two categorical variables independent?" use `cross_tabs`. Goodness tests one variable against an externally specified distribution; cross_tabs tests two variables for association.
