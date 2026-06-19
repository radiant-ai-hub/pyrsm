# pyrsm.basics.single_mean — reference

This file is the deeper reference for `pyrsm.basics.single_mean`. The main `SKILL.md` walks the workflow at a high level; come here for API details, interpretation templates, and worked examples.

## Table of contents

1. Constructor signature
2. `summary()` — what each option prints
3. `plot()` — what the histogram shows
4. Output attributes
5. Plain-English interpretation templates
6. Choosing `alt_hyp` (one-sided vs two-sided)
7. The three equivalences — p-value, CI, t-vs-critical
8. Sample size, power, and effect size
9. Related basics classes — when to switch
10. Worked example (`demand_uk`)
11. Common pitfalls

---

## 1. Constructor signature

```python
rsm.basics.single_mean(
    data,                    # polars/pandas DataFrame, OR {"name": df}
    var,                     # column to test (must be numeric)
    alt_hyp="two-sided",     # "two-sided", "greater", or "less"
    conf=0.95,               # confidence level (0 < conf < 1)
    comp_value=0,            # the population mean under H0
)
```

**Always pass data as `{"name": df}`** — the `name` becomes the dataset label in the printed summary. Plain `df` works, but the summary prints `"Not provided"`.

The class accepts polars or pandas DataFrames; it converts internally. `var` must reference a numeric column. Missing values are dropped — they show up as `n_missing` in the descriptive-statistics table, and the test is run on the non-missing rows only.

`alt_hyp` is a substantive choice (see §6). It changes the p-value, changes the CI from two-sided to one-sided, and changes which critical t-value applies.

`comp_value` defaults to 0, which is rarely what the user wants in a class setting — always confirm a non-zero benchmark.

## 2. `summary()` — what each option prints

```python
sm.summary(
    dec=3,                   # decimal places for floats
    plain=True,              # plain text vs styled great_tables output (Jupyter)
)
```

Output structure (plain mode):

- **Header** — dataset name, variable, confidence, comparison value, null and alternative hypothesis statements.
- **Descriptive-statistics table** — one row with `mean`, `n`, `n_missing`, `sd`, `se`, `me` (margin of error = `tscore * se`).
- **Hypothesis-test table** — one row with `diff`, `se`, `t.value`, `p.value`, `df`, lower CI bound, upper CI bound, significance stars.
- **Significance code legend** — `***` < 0.001, `**` < 0.01, `*` < 0.05, `.` < 0.1.

`plain=False` switches to styled `great_tables` output, which is nicer in Jupyter but harder to copy into a text writeup. The numbers are the same.

There is no `summary(extra=True)` variant for `single_mean` (unlike `compare_means`). The single-mean test has nothing extra to report.

## 3. `plot()` — what the histogram shows

```python
sm.plot(
    plots="hist",            # the only working plot type for single_mean
    theme="modern",          # "modern", "publication", "minimal", "classic"
    backend="plotnine",      # "plotnine" or "plotly"
)
```

`plots="hist"` returns a plotnine ggplot (or a plotly figure if `backend="plotly"`) of the sample distribution with four reference lines:

- **Solid black** at the sample mean.
- **Dashed black** at the lower and upper CI bounds.
- **Solid red** at the comparison value.

The visual decision rule is: **if the red line is outside the dashed-black interval, reject H₀**. This is the same conclusion as the p-value and the t-vs-critical comparisons — it's just easier to see at a glance.

`plots="sim"` is listed in the API but currently prints `"Plot type not available yet"` and returns `None`. Don't promise it to the user.

## 4. Output attributes

After the constructor runs, the following attributes are available on `sm`:

| Attribute | Type | Meaning |
| --- | --- | --- |
| `sm.mean` | float | Sample mean of `var`. |
| `sm.n` | int | Total rows (including missing). |
| `sm.n_missing` | int | Count of nulls in `var`. |
| `sm.sd` | float | Sample standard deviation (`ddof=1`). |
| `sm.se` | float | Standard error of the mean: `sd / sqrt(n_eff)`. |
| `sm.me` | float | Margin of error: `t_crit * se` at the user's `conf` level, df = `n_eff - 1`. |
| `sm.diff` | float | `mean - comp_value`. |
| `sm.t_val` | float | T-statistic: `diff / se`. |
| `sm.p_val` | float | P-value for the test, given `alt_hyp`. |
| `sm.ci` | tuple | Confidence interval for the mean, oriented by `alt_hyp`. |
| `sm.df` | int | Degrees of freedom (`n_eff - 1`). |
| `sm.data`, `sm.var`, `sm.alt_hyp`, `sm.conf`, `sm.comp_value`, `sm.name` | various | Echoes of the inputs, for reproducibility. |

Note that `sm.ci` is `(lower, upper)` for `alt_hyp="two-sided"`, `(-inf, upper)` for `"less"`, and `(lower, +inf)` for `"greater"`. The summary prints these correctly; don't write them as `[low, high]` for a one-sided test.

## 5. Plain-English interpretation templates

Use these templates verbatim (substituting variable names and units) when walking a student through results. **Always tie the units back to the sidecar description file** if available.

### Hypotheses

> H₀: the mean of `<var>` in the population is equal to `<comp_value>` `<unit>`.
> Hₐ: the mean of `<var>` in the population is `<less than | greater than | not equal to>` `<comp_value>` `<unit>`.

### Sample description

> The sample contains `<n>` observations (`<n_missing>` missing) with a sample mean of `<mean>` `<unit>` and a standard deviation of `<sd>` `<unit>`. The standard error of the mean is `<se>` `<unit>`.

### P-value verdict

> The p-value for the test is `<p.value>`. Because this is `<smaller | not smaller>` than the significance level (α = 0.05), we `<reject | fail to reject>` the null hypothesis. The data `<do | do not>` provide statistically significant evidence at the 5% level that the population mean differs from `<comp_value>` in the direction specified.

### Confidence-interval verdict

> The `<conf*100>`% confidence interval for the population mean is `<[lo, hi]>` `<unit>`. Because the comparison value `<comp_value>` `<is | is not>` contained in this interval, we `<reject | fail to reject>` the null hypothesis. If we were to repeat this sampling procedure many times, we would expect `<conf*100>`% of the resulting intervals to contain the true population mean.

### t-value-vs-critical verdict

> The observed t-statistic is `<t.value>` on `<df>` degrees of freedom. The critical t-value for a `<one-sided | two-sided>` test at α = 0.05 is approximately `<t_crit>` (computed via `prob_calc("tdist", df=<df>, pub=<...>)`). Because `|t|` is `<larger | smaller>` than `<t_crit>`, we `<reject | fail to reject>` the null hypothesis.

### Effect size in business terms

> The sample mean is `<diff>` `<unit>` `<above | below>` the comparison value of `<comp_value>` `<unit>` — a `<diff / comp_value * 100>`% relative difference. Whether this magnitude is large enough to drive the `<decision the user is making>` is a separate question from whether it is statistically significant.

## 6. Choosing `alt_hyp` (one-sided vs two-sided)

`alt_hyp` determines the direction of the test, and it changes three things at once: the p-value, the orientation of the CI, and the critical t-value. **Pick it from the business decision being made, not from the data.**

### One-sided "greater"

Use when the decision only triggers if the mean is *above* the comparison value. The classic example is the demand-UK case: management will enter the market if and only if average demand exceeds 1750 units per store. If actual demand is below 1750, the decision is to *not* enter — but the test does not need to detect that as a "significant finding" because no action would be taken differently.

- p-value: P(observed t or larger | H₀ true).
- CI: `[lower, +inf)` — only the lower bound matters.
- Reject H₀ ⇔ `lower bound > comp_value`.
- Critical t: `prob_calc("tdist", df=<df>, pub=1 - α)`.

### One-sided "less"

Symmetric. Use when the decision only triggers if the mean is *below* `comp_value` (e.g., "stop production if average defect rate is below 0.5%").

- p-value: P(observed t or smaller | H₀ true).
- CI: `(-inf, upper]`.
- Reject H₀ ⇔ `upper bound < comp_value`.
- Critical t: `prob_calc("tdist", df=<df>, plb=α)`.

### Two-sided

Use when either direction matters (e.g., "is this batch *different* from spec — too high or too low?"). The default.

- p-value: 2 × min(P(≥t), P(≤t)).
- CI: `[lower, upper]`.
- Reject H₀ ⇔ `comp_value < lower` OR `comp_value > upper`.
- Critical t: `prob_calc("tdist", df=<df>, pub=1 - α/2)` for the upper critical; the lower is its negative.

### Tradeoff

A one-sided test is roughly twice as powerful as a two-sided test at the same α — but you have to be willing to *not detect* an effect in the opposite direction. That tradeoff has to be honest: if you genuinely don't care about an effect in the opposite direction (because no action would change), one-sided is correct and more powerful. If you would want to know about either direction, two-sided is correct and you absorb the power cost.

**Never pick a one-sided test after seeing the data go in that direction.** That doubles the false-positive rate. The direction must be set by the question, not by the sample.

## 7. The three equivalences — p-value, CI, t-vs-critical

Every one-sample test admits three logically equivalent decision rules. The pedagogical purpose of teaching all three is to give the student multiple mental anchors and to make later concepts (confidence-interval inversion, Bayesian credible intervals) easier to grasp.

| View | The question | The answer |
| --- | --- | --- |
| p-value | What's the chance of data this extreme (or more) under H₀? | If `p < α`, reject. |
| CI | What range of population means is plausible given the data? | If `comp_value ∉ CI`, reject. |
| t-vs-critical | How many SEs is the sample mean from `comp_value`, and is that more than the rejection threshold? | If `|t| > t_crit`, reject. |

The three views must always agree by construction. If a student's writeup has them disagreeing, there is an arithmetic or specification error somewhere — check first that the alt_hyp is consistent across the three calculations.

## 8. Sample size, power, and effect size

The single-mean t-test is robust to non-normality for moderate-to-large n (say, n ≥ 30) by the central limit theorem. For small n with heavy skew or outliers, prefer a non-parametric alternative — the Wilcoxon signed-rank test, which `pyrsm.basics.compare_means` supports via `test_type="wilcox"` when run with one group.

### Effect-size note

`diff = mean - comp_value` is the absolute effect size in the original units. For a unit-free version, divide by `sd`:

```python
cohen_d = sm.diff / sm.sd
```

Conventionally:
- |d| ≈ 0.2 — small effect
- |d| ≈ 0.5 — medium
- |d| ≈ 0.8 — large

These thresholds are rough rules of thumb and are not a substitute for asking "is this effect large enough to matter for the decision?".

### Large-n caution

With n in the thousands, almost any `diff` will achieve statistical significance. Always report the magnitude of `diff` (and ideally its relation to a decision-relevant threshold) alongside the p-value. A statistically significant result that is too small to matter is a real category — common in customer-analytics datasets with millions of rows.

## 9. Related basics classes — when to switch

- **Response is 0/1 or yes/no** → use `single_prop` (`pyrsm-single-prop` skill). The t-test on 0/1 data works arithmetically but the test is named-for-proportions in that case, and `single_prop` exposes binomial-exact vs z-test options that single_mean does not.
- **Two or more groups to compare** → `compare_means` (`pyrsm-compare-means` skill).
- **Paired (before/after) measurements** → `compare_means` with `sample_type="paired"`.
- **Categorical distribution vs an expected distribution** → `goodness` (`pyrsm-goodness` skill).
- **Modeling how the mean depends on covariates** → `regress` (`pyrsm-regress` skill).
- **Finding a critical value or a tail probability** → `prob_calc` (`pyrsm-prob-calc` skill).

## 10. Worked example — `demand_uk`

From `examples/basics/basics-single-mean.ipynb`:

> "We have access to data from a random sample of grocery stores in the UK. Management will consider entering this market if consumer demand for the product category exceeds 100M units, or, approximately, 1750 units per store. The average demand per store in the sample is equal to 1953. While this number is larger than 1750 we need to determine if the difference could be attributed to sampling error."

```python
import polars as pl
import pyrsm as rsm

demand_uk = pl.read_parquet("<abs-path>/demand_uk.parquet")
sm = rsm.basics.single_mean(
    {"demand_uk": demand_uk},
    var="demand",
    alt_hyp="greater",
    conf=0.95,
    comp_value=1750,
)
sm.summary()
```

Output (paraphrased from the notebook):

```
Single mean test
Data      : demand_uk
Variables : demand
Confidence: 0.95
Comparison: 1750

Null hyp. : the mean of demand is equal to 1750
Alt. hyp. : the mean of demand is greater than 1750

Descriptive: mean=1953.393, n=572, n_missing=0, sd=815.266, se=34.088, me=66.953

Hypothesis: diff=203.393, se=34.088, t.value=5.967, p.value < .001,
            df=571, 5.0%-bound=1897.233, upper=inf, ***
```

Interpretation walkthrough:

1. **Hypotheses.** H₀: mean monthly demand per store = 1750 units. Hₐ: mean monthly demand per store > 1750 units.
2. **Sample.** 572 stores, average demand 1953 units per store, sd 815.
3. **p-value verdict.** p < .001 ≪ 0.05 → reject H₀.
4. **CI verdict.** One-sided 95% lower bound is 1897. Comparison value 1750 is below the lower bound → comparison value falls outside the (right-open) interval [1897, +inf) → reject H₀.
5. **t-vs-critical verdict.** Observed t = 5.967 on 571 df. Critical t (one-sided, α=0.05) = 1.648 from `prob_calc("tdist", df=571, pub=0.95)`. 5.967 ≫ 1.648 → reject H₀.
6. **Business conclusion.** Strong evidence that average monthly demand per store in the UK exceeds the 1750-unit threshold (point estimate 1953, lower 95% bound 1897). Management's decision rule says they should enter the market.

## 11. Common pitfalls

- **Picking `alt_hyp` after looking at the sample.** Doubles the effective false-positive rate. The direction must come from the decision being made, not from the data.
- **Forgetting the dict wrapper.** `single_mean(df, ...)` works but the summary shows `"Not provided"`. Use `single_mean({"<name>": df}, ...)`.
- **Defaulting `comp_value=0`.** The class default is 0, which is rarely the substantively interesting benchmark. Always confirm an explicit comp_value.
- **Reading p alone, ignoring `diff`.** A statistically significant result with a tiny `diff` (especially with very large n) is not the same as a practically important one. Report `diff` in the original units and discuss whether the magnitude matters.
- **Interpreting `comp_value ∈ CI` as "the mean equals comp_value".** "Fail to reject H₀" is not "accept H₀". The CI contains all the values of the population mean that are plausibly consistent with the data — `comp_value` being inside means it is consistent, not that it is correct.
- **Reading the one-sided CI as a two-sided one.** When `alt_hyp="greater"`, the printed interval is `[lower, +inf)` — the upper bound is genuinely infinity, not a missing value. The decision rule only uses the lower bound. Don't try to read the upper number as if it were a normal CI.
- **Running the t-test on a tiny, heavily-skewed sample.** The CLT-based t-test is robust for n ≥ 30, but with n < 30 and visible skew it under-covers. Run the histogram (`sm.plot("hist")`), eyeball the shape, and consider Wilcoxon (`compare_means` with `test_type="wilcox"`) as a robustness check.
- **Treating p = 0.04 and p = 0.06 as categorically different.** They are nearly identical pieces of evidence. The 0.05 threshold is a convention. Report the actual p-value and lead with the CI when explaining the result.
