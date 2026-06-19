# pyrsm.basics.single_prop — reference

This file is the deeper reference for `pyrsm.basics.single_prop`. The main `SKILL.md` walks the workflow at a high level; come here for API details, interpretation templates, and worked examples.

## Table of contents

1. Constructor signature
2. `summary()` — what each option prints
3. `plot()` — what the bar shows
4. Output attributes
5. Plain-English interpretation templates
6. Choosing `alt_hyp` (one-sided vs two-sided)
7. The three equivalences — p-value, CI, critical successes
8. Binomial-exact vs z-test — when to use each
9. Wilson confidence intervals (z-test) and Clopper–Pearson (binomial)
10. Related basics classes — when to switch
11. Worked example (`consider`)
12. Common pitfalls

---

## 1. Constructor signature

```python
rsm.basics.single_prop(
    data,                    # polars/pandas DataFrame, OR {"name": df}
    var,                     # column (categorical/string)
    lev=None,                # level of `var` treated as "success"
    alt_hyp="two-sided",     # "two-sided", "greater", or "less"
    conf=0.95,               # confidence level (0 < conf < 1)
    comp_value=0.5,          # population proportion under H0 (must be in (0,1))
    test_type="binomial",    # "binomial" (exact) or "z" (normal approximation)
)
```

**Always pass data as `{"name": df}`** — the `name` becomes the dataset label in the printed summary. Plain `df` works, but the summary prints `"Not provided"`.

`var` must be a categorical / string / Enum column. Internally `single_prop` counts rows where `var == lev` (`ns`, the number of successes) out of the non-missing rows (`n`). Missing values are reported as `n_missing` and excluded from `n`.

`comp_value` must be strictly between 0 and 1 — passing 0 or 1 raises an exception. The class default is 0.5, which is rarely the substantively interesting benchmark; always confirm an explicit comp_value.

`test_type` is the key API distinction not shared with `single_mean`. See §8.

## 2. `summary()` — what each option prints

```python
sp.summary(
    dec=3,                   # decimal places for floats
    plain=True,              # plain text vs styled great_tables output (Jupyter)
)
```

Output structure (plain mode):

- **Header** — test variant (binomial-exact or z-test), dataset, variable, level, confidence, null and alternative hypothesis statements.
- **Descriptive-statistics table** — one row with `p`, `ns`, `n`, `n_missing`, `sd`, `se`, `me`.
- **Hypothesis-test table** — one row with `diff`, then a test-specific statistic column (`ns` for binomial, `z.value` for z-test), `p.value`, lower CI bound, upper CI bound, significance stars.
- **Significance code legend** — `***` < 0.001, `**` < 0.01, `*` < 0.05, `.` < 0.1.

`plain=False` switches to styled great_tables output, which is nicer in Jupyter but harder to copy into a writeup. There is no `summary(extra=True)` variant.

## 3. `plot()` — what the bar shows

```python
sp.plot(plots="bar")
```

Returns a plotnine ggplot bar chart showing the proportion of each level in `var`. It is useful for sanity-checking the sample composition (did most respondents say "no"? "yes"? Is there a third level?) but does not visualize the test itself — there is no overlay of `comp_value` or the CI.

`plots="bar"` is the only supported plot type.

## 4. Output attributes

After the constructor runs, the following attributes are available on `sp`:

| Attribute | Type | Meaning |
| --- | --- | --- |
| `sp.p` | float | Sample proportion: `ns / n`. |
| `sp.ns` | int | Number of "successes" (rows where `var == lev`). |
| `sp.n` | int | Number of non-missing rows in `var`. |
| `sp.n_missing` | int | Count of nulls in `var`. |
| `sp.sd` | float | Binomial-style sd: `sqrt(p * (1 - p))`. |
| `sp.se` | float | Standard error of the sample proportion: `sd / sqrt(n)`. |
| `sp.se_p0` | float | SE under H₀: `sqrt(comp_value * (1 - comp_value) / n)`. Used by the z-test. |
| `sp.me` | float | Margin of error: `z_critical * se`. |
| `sp.diff` | float | `p - comp_value`. |
| `sp.p_val` | float | P-value (oriented by `alt_hyp` and `test_type`). |
| `sp.ci` | tuple | Confidence interval for the population proportion. |
| `sp.z_score` | float \| None | Z-statistic — `None` when `test_type="binomial"`. |
| `sp.z_critical` | float | Two-tailed critical z at `alpha = 1 - conf`. |
| `sp.alpha` | float | `1 - conf`. |
| `sp.data`, `sp.var`, `sp.lev`, `sp.alt_hyp`, `sp.conf`, `sp.comp_value`, `sp.test_type`, `sp.name` | various | Echoes of the inputs. |

The CI is `(lower, upper)` for two-sided. For `alt_hyp="less"` it is `(0, upper)` and for `alt_hyp="greater"` it is `(lower, 1)` — bounded by the [0, 1] interval of valid proportions, not by ±∞ as in `single_mean`.

## 5. Plain-English interpretation templates

Use these templates verbatim (substituting variable names and units) when walking a student through results.

### Hypotheses

> H₀: the proportion of `<lev>` in `<var>` in the population is equal to `<comp_value>`.
> Hₐ: the proportion of `<lev>` in `<var>` in the population is `<less than | greater than | not equal to>` `<comp_value>`.

### Sample description

> The sample contains `<n>` non-missing observations (`<n_missing>` missing). `<ns>` of them are `<lev>`, a sample proportion of `<p>` (i.e., `<p*100>`%), with standard error `<se>` (`<se*100>` percentage points).

### P-value verdict

> The p-value for the test is `<p.value>`. Because this is `<smaller | not smaller>` than the significance level (α = 0.05), we `<reject | fail to reject>` the null hypothesis. The data `<do | do not>` provide statistically significant evidence at the 5% level that the population proportion differs from `<comp_value>` in the direction specified.

### Confidence-interval verdict

> The `<conf*100>`% confidence interval for the population proportion is `<[lo, hi]>`. Because the comparison value `<comp_value>` `<is | is not>` contained in this interval, we `<reject | fail to reject>` the null hypothesis.

### Number-of-successes verdict (binomial only)

> We observed `<ns>` successes out of `<n>` trials. Under H₀, the number of successes follows a Binomial(`<n>`, `<comp_value>`) distribution. The critical number of successes for a `<one-sided>` test at α = 0.05 is `<crit>` (from `prob_calc("binom", n=<n>, p=<comp_value>, <plb|pub>=<α|1-α>)`). Because `<ns>` `<is | is not>` in the rejection region (`<below crit | above crit>`), we `<reject | fail to reject>` the null hypothesis.

### Effect size in business terms

> The sample proportion is `<diff*100>` percentage points `<above | below>` the comparison value of `<comp_value*100>`% — a `<diff / comp_value * 100>`% relative difference. Whether this magnitude is large enough to drive the `<decision the user is making>` is a separate question from whether it is statistically significant.

## 6. Choosing `alt_hyp` (one-sided vs two-sided)

`alt_hyp` determines the direction of the test, the orientation of the CI, and the rejection region. **Pick it from the business decision being made, not from the data.**

### One-sided "less"

Use when the decision only triggers if the proportion is *below* `comp_value` (e.g., consider example: spend more on advertising only if brand preference is below 10%).

- CI: `[0, upper]`.
- Reject H₀ ⇔ `upper < comp_value`.
- Binomial critical successes: `prob_calc("binom", n=<n>, p=<comp_value>, plb=alpha)`.

### One-sided "greater"

Symmetric. Use when the decision only triggers if the proportion is *above* `comp_value` (e.g., "launch product only if interest > 25%").

- CI: `[lower, 1]`.
- Reject H₀ ⇔ `lower > comp_value`.
- Binomial critical successes: `prob_calc("binom", n=<n>, p=<comp_value>, pub=1-alpha)`.

### Two-sided

Use when either direction matters.

- CI: `[lower, upper]`.
- Reject H₀ ⇔ `comp_value ∉ [lower, upper]`.

A one-sided test is roughly twice as powerful as a two-sided test at the same α — but you have to be willing to *not detect* an effect in the opposite direction. Make the choice based on the question, not the result.

## 7. The three equivalences — p-value, CI, critical successes

Every one-sample proportion test admits three logically equivalent decision rules:

| View | The question | The answer |
| --- | --- | --- |
| p-value | What's the chance of data this extreme (or more) under H₀? | If `p < α`, reject. |
| CI | What range of population proportions is plausible? | If `comp_value ∉ CI`, reject. |
| Critical successes (binomial) | Is `ns` in the rejection region under Binomial(n, comp_value)? | If yes, reject. |
| Critical z (z-test) | Is `|z|` larger than the z-critical value? | If yes, reject. |

All three views must agree by construction. If a writeup has them disagreeing, check first that the `alt_hyp` is the same across the three calculations.

## 8. Binomial-exact vs z-test — when to use each

This is the **defining critical concept** for `single_prop`.

### Binomial-exact (`test_type="binomial"`)

- Wraps `scipy.stats.binomtest`.
- P-value is the exact probability of observing `ns` or more extreme successes under Binomial(`n`, `comp_value`).
- CI is the Clopper–Pearson exact interval.
- **Always valid**, regardless of n or p₀.
- Can be slightly conservative (the discrete nature of the binomial means the actual coverage of the CI is at least `conf`, often a hair more).
- The test statistic column in the summary is `ns` (the observed count), not a `z`.

### Z-test (`test_type="z"`)

- Uses the normal approximation to the binomial.
- Z-statistic: `z = (p - comp_value) / sqrt(comp_value * (1 - comp_value) / n)`.
- P-value via the standard normal CDF.
- CI is the **Wilson score interval** (same as R's `prop.test`), not the naive normal-approximation `p ± z * se`.
- **Only reliable when both `n * comp_value ≥ ~5–10` and `n * (1 - comp_value) ≥ ~5–10`** — i.e., when the binomial is well-approximated by a normal.
- Gives a `z.value` for direct comparison to a critical-z.

### Decision rule

> Compute `n * comp_value` and `n * (1 - comp_value)`. If **both** are ≥ ~10, either test is fine — default to binomial for class assignments unless the user asks for a z-statistic. If **either** is < ~5, use the binomial. The gray zone (5–10) is the classic rule-of-thumb boundary; defaulting to the binomial in this range is the conservative choice.

### Worked check

For the consider example: n = 1000, comp_value = 0.10.
- `n * comp_value = 100`. ✓
- `n * (1 - comp_value) = 900`. ✓
- Either test is fine. The notebook uses the binomial.

For a hypothetical small pilot: n = 20, comp_value = 0.05.
- `n * comp_value = 1`. ✗ (well below 5)
- `n * (1 - comp_value) = 19`. ✓
- Only the binomial-exact test is reliable here. The z-test will produce numbers, but they will be misleading.

### Why students get this wrong

The z-test produces a tidy continuous test statistic and a familiar `±1.96` rejection region. It feels like the more "advanced" test. Students reach for it when the more conservative choice would be the binomial-exact. The skill should always make the sample-size check visible to the student before letting them proceed with `test_type="z"`.

## 9. Wilson confidence intervals (z-test) and Clopper–Pearson (binomial)

Two related but distinct CIs ship with this class.

### Wilson score interval (used when `test_type="z"`)

```
        p + z²/2n ± z * sqrt(p(1-p)/n + z²/4n²)
CI =   ───────────────────────────────────────
                    1 + z²/n
```

(Implementation in `single_prop._init_` via `wilson_ci`.) The Wilson interval has substantially better coverage than the naive `p ± z * se` interval, especially for small n or p close to 0 or 1. It is the same interval R's `prop.test` returns.

### Clopper–Pearson exact (used when `test_type="binomial"`)

Computed by `scipy.stats.binomtest`'s `.proportion_ci()`. Guaranteed coverage of at least the nominal level — slightly wider than Wilson, but valid for any n and p₀.

### Why neither is `p ± z * se`

The naive normal-approximation interval is taught in many textbooks but has bad coverage near 0 or 1. Pyrsm uses Wilson (z-test) or Clopper–Pearson (binomial) for both options. If a student's textbook formula disagrees with the printed CI, this is why.

## 10. Related basics classes — when to switch

- **Continuous numeric response** → `single_mean` (`pyrsm-single-mean` skill).
- **Proportion of one level across two or more groups** → `compare_props` (`pyrsm-compare-props` skill).
- **Three or more levels, testing whether the distribution matches an expected one** → `goodness` (`pyrsm-goodness` skill).
- **Two categorical variables, testing for association** → `cross_tabs` (`pyrsm-cross-tabs` skill).
- **Modeling how the probability of `lev` depends on covariates** → `logistic` (`pyrsm-logistic` skill).
- **Finding a critical value or tail probability for a distribution** → `prob_calc` (`pyrsm-prob-calc` skill).

## 11. Worked example — `consider`

From `examples/basics/basics-single-proportion.ipynb`:

> "A car manufacturer conducted a study by randomly sampling and interviewing 1,000 consumers. Management has already determined the company will enter this segment, but if brand preference is below 10%, additional resources will be committed to advertising. In the sample, 93 of 1,000 consumers exhibited strong brand liking."

```python
import polars as pl
import pyrsm as rsm

consider = pl.read_parquet("<abs-path>/consider.parquet")
sp = rsm.basics.single_prop(
    data={"consider": consider},
    var="consider",
    lev="yes",
    alt_hyp="less",
    conf=0.95,
    comp_value=0.1,
    test_type="binomial",
)
sp.summary()
```

Output (from the notebook):

```
Single proportion (binomial exact)
Data      : consider
Variable  : consider
Level     : "yes" in consider
Confidence: 0.95
Null hyp. : the proportion of "yes" in consider is equal to 0.1
Alt. hyp. : the proportion of "yes" in consider less than 0.1

Descriptive: p=0.093, ns=93, n=1000, n_missing=0, sd=0.290, se=0.009, me=0.018
Hypothesis: diff=-0.007, ns=93, p.value=0.249, 0%-bound=0, upper=0.110
```

Interpretation walkthrough:

1. **Hypotheses.** H₀: brand preference = 10%. Hₐ: brand preference < 10%.
2. **Sample.** 1,000 respondents, 93 said "yes" → sample proportion 9.3% with SE 0.9 percentage points.
3. **Sample-size check.** `n * p_0 = 100 ≥ 10`, `n * (1 - p_0) = 900 ≥ 10` — both tests would be valid. Notebook chose binomial.
4. **p-value verdict.** p = 0.249 ≫ 0.05 → fail to reject H₀.
5. **CI verdict.** One-sided 95% upper bound is 0.110. Comparison value 0.10 < 0.110 → comp_value is inside the (right-closed) interval [0, 0.110] → fail to reject H₀.
6. **Critical-successes verdict.** Under Binomial(1000, 0.10), the critical number of successes for the lower 5% rejection region is 85 (from `prob_calc("binom", n=1000, p=0.1, plb=0.05)`). Observed `ns = 93 > 85` → not in the rejection region → fail to reject H₀.
7. **Business conclusion.** The data are consistent with brand preference being at or above the 10% threshold. The company should not commit additional advertising resources on the basis of this survey.

### Z-test sibling on the same data

```python
sp_z = rsm.basics.single_prop(consider, var="consider", lev="yes",
                              test_type="z", comp_value=0.1)
sp_z.summary()
# Single proportion (z-test)
# Hypothesis: diff=-0.007, z.value=-0.738, p.value=0.461, 2.5%=0.077, 97.5%=0.113
```

Note that the z-test here defaulted to `alt_hyp="two-sided"`, so the p-value (0.461) and CI ([0.077, 0.113]) are two-sided — those would be 0.461/2 = 0.231 for a one-sided "less" p-value, in line with the binomial's 0.249. Both tests agree: fail to reject.

## 12. Common pitfalls

- **Forgetting `lev` or picking the wrong level.** Without `lev` the class doesn't know what counts as "success" and `ns` will be 0 or unpredictable. Always set `lev` explicitly, and check `df[var].value_counts()` if you're unsure of the casing/spelling.
- **`comp_value` of 0 or 1.** Raises an exception. Use a value strictly between 0 and 1; if the substantive benchmark really is "any successes at all", reframe the question (e.g., test against `comp_value=0.01` if 1% is the smallest rate the business would still care about).
- **Reaching for the z-test on a tiny sample with a rare outcome.** Classic failure mode: n × p₀ < 5 and/or n × (1−p₀) < 5. The z-test will produce a `z.value` and p-value anyway — they are just wrong. Always check before defaulting to `test_type="z"`.
- **Picking `alt_hyp` after looking at the data.** Same logic as `single_mean`: doubles the effective false-positive rate.
- **Picking `lev` after looking at the data** (e.g., "the more interesting level is the one with the smaller proportion — let's flip"). Same problem.
- **Forgetting the dict wrapper.** `single_prop(df, ...)` works but the summary shows `"Not provided"`. Use `single_prop({"<name>": df}, ...)`.
- **Reading the printed CI as a naive `p ± z * se`.** It isn't — the CI is Wilson (z-test) or Clopper–Pearson (binomial). The numbers will differ slightly from a textbook hand-calculation that uses the naive formula. Pyrsm is right; the textbook formula is the classroom simplification.
- **Treating p = 0.04 and p = 0.06 as categorically different.** Same warning as `single_mean`. Lead with the CI.
- **Reporting only `p` (the sample proportion) without `ns` and `n`.** "Brand preference is 9.3%" leaves the precision invisible. "93 of 1,000 respondents (9.3%, 95% CI [0%, 11.0%])" is the right level of detail.
