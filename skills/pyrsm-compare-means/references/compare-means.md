# pyrsm.basics.compare_means — reference

This file is the deeper reference for `pyrsm.basics.compare_means`. The main `SKILL.md` walks the workflow at a high level; come here for API details, interpretation templates, and worked examples.

## Table of contents

1. Constructor signature
2. `summary()` — what each option prints
3. `plot()` — what each plot type shows
4. Output attributes
5. Plain-English interpretation templates
6. Multiple-testing adjustment — the critical concept
7. t-test vs Wilcoxon
8. Independent vs paired samples
9. The `comb` parameter — choosing which pairs to compare
10. Related basics classes — when to switch
11. Worked example (`salary`)
12. Common pitfalls

---

## 1. Constructor signature

```python
rsm.basics.compare_means(
    data,                            # polars/pandas DataFrame, OR {"name": df}
    var1,                            # grouping variable (categorical OR numeric)
    var2,                            # numeric outcome (str or list[str])
    comb=[],                         # list of "level1:level2" pair strings; [] = all pairs
    alt_hyp="two-sided",             # "two-sided", "greater", "less"
    conf=0.95,                       # confidence level
    sample_type="independent",       # "independent" or "paired"
    adjust=None,                     # None, "bonferroni", or any statsmodels multipletests method
    test_type="t-test",              # "t-test" (Welch's / paired) or "wilcox" (rank-sum / signed-rank)
)
```

**Always pass data as `{"name": df}`** — the `name` becomes the dataset label in the printed summary. Plain `df` works, but the summary prints `"Not provided"`.

The `var1` / `var2` interface has two modes:

- **Categorical grouping variable (typical)**: `var1` is a string/Categorical/Enum column. `var2` is one numeric column. The class compares means of `var2` across the levels of `var1`. This is the documented and most common use.
- **Multiple numeric columns**: `var1` is numeric and `var2` is a list of numeric columns. The data are auto-melted to long form, with column name as the implicit grouping variable. Useful for comparing multiple measurements of the same kind (e.g., scores on multiple test items).

If `var1` is an Enum, the level order comes from the Enum's `dtype.categories`. Otherwise levels are sorted alphabetically.

## 2. `summary()` — what each option prints

```python
cm.summary(
    extra=False,                     # if True, add se, t.value, df, CI columns
    dec=3,                           # decimal places
    plain=True,                      # plain text vs great_tables in Jupyter
)
```

Output structure (plain mode):

- **Header** — test type, dataset, variables, sample type, confidence level, adjustment method.
- **Descriptive statistics** — per group: `mean`, `n`, `n_missing`, `sd`, `se`, `me`.
- **Pairwise comparisons** — one row per pair in `comb`:
  - Always: `Null hyp.`, `Alt. hyp.`, `diff`, `p.value`, significance stars.
  - With `extra=True`: also `se`, `t.value`, `df`, lower CI bound, upper CI bound.

For a first look, **pass `extra=True`** — the additional columns give the student context (degrees of freedom, CI bounds) that the bare table doesn't.

`p.value` formatting: `< .001` for tiny p-values, else rounded to `dec`. If `adjust` is set, the p-values are the *adjusted* values (and the stars reflect them).

## 3. `plot()` — what each plot type shows

```python
cm.plot(
    plots="scatter",                 # "scatter" (default), "box", "density", "bar"
    nobs=None,                       # subsample for scatter/box; None = all
)
```

| `plots=` | Shows |
| --- | --- |
| `"scatter"` | Jittered scatter of every (group, value) pair; horizontal crossbar at each group mean. Best for spotting outliers + getting an honest sense of spread. |
| `"box"` | Boxplot per group. Best for comparing medians and IQRs. |
| `"density"` | Overlaid density curves per group. Best for shape comparisons (bimodality, asymmetry). |
| `"bar"` | Bar of group means only. Loses spread information; use sparingly. |

All four return plotnine ggplot objects. In Jupyter they display inline; elsewhere print the returned object with `print(p)`.

## 4. Output attributes

| Attribute | Type | Meaning |
| --- | --- | --- |
| `cm.descriptive_stats` | `pl.DataFrame` | Per-group summary: `var1`, `mean`, `n`, `n_missing`, `sd`, `se`, `me`. |
| `cm.comp_stats` | `pl.DataFrame` | One row per pair: `Null hyp.`, `Alt. hyp.`, `diff`, `p.value`, `se`, `t.value`, `df`, CI bounds, stars. |
| `cm.levels` | list[str] | The (sorted or Enum-ordered) levels of `var1`. |
| `cm.comb` | list[str] | The pair-string list actually evaluated. |
| `cm.var1`, `cm.var2`, `cm.alt_hyp`, `cm.conf`, `cm.sample_type`, `cm.adjust`, `cm.test_type`, `cm.name` | various | Echoes of inputs. |
| `cm.data` | `pl.DataFrame` | The (possibly melted) data actually used. |

`cm.comp_stats["p.value"]` is the **adjusted** p-value when `adjust` is set; the original unadjusted p-values are not retained on the object. If you need the unadjusted ones, refit with `adjust=None`.

## 5. Plain-English interpretation templates

### Header

> We compared the mean of `<var2>` across the `<k>` levels of `<var1>` (`<level_1>`, `<level_2>`, …) using a `<test_type>`. Samples are `<independent | paired>`, with `<conf*100>`% confidence. Multiple-testing adjustment: `<None | bonferroni | …>`. The number of pairwise comparisons is `<k*(k-1)/2>`.

### Descriptive

> Group means: `<level_1>` = `<mean_1>` `<unit>` (n=`<n_1>`, sd=`<sd_1>`), `<level_2>` = `<mean_2>` `<unit>` (n=`<n_2>`, sd=`<sd_2>`), …

### Pairwise hypothesis

> For the comparison `<level_a>` vs `<level_b>`:
> H₀: the population mean of `<var2>` is equal in `<level_a>` and `<level_b>`.
> Hₐ: the population mean of `<var2>` in `<level_a>` is `<less than | greater than | not equal to>` that in `<level_b>`.

### Three-way verdict for a pair

> p = `<p>`. CI for `<level_a> − <level_b>` is [`<lo>`, `<hi>`] `<unit>`. Observed t = `<t>` on `<df>` df.
> Because p `<is | is not>` smaller than α = 0.05, and 0 `<is | is not>` inside the CI, and `|t|` `<exceeds | does not exceed>` the critical t, we `<reject | fail to reject>` H₀ for this pair.

### Effect size

> Group `<level_a>` has a mean `<diff>` `<unit>` `<higher | lower>` than group `<level_b>` (95% CI [`<lo>`, `<hi>`]). Relative to the mean in `<level_b>` (`<mean_b>` `<unit>`), this is a `<diff/mean_b*100>`% difference.

## 6. Multiple-testing adjustment — the critical concept

The pedagogical heart of compare-means is the **family-wise error rate**.

### The math

If you run `k` independent tests at α = 0.05 and all `k` true differences are zero, the probability of *at least one* false positive is:

```
FWER = 1 − (1 − α)^k
```

For α = 0.05:

| k tests | FWER |
| --- | --- |
| 1 | 0.05 |
| 3 | 0.143 |
| 5 | 0.226 |
| 10 | 0.401 |
| 20 | 0.642 |

So with k = 10, the chance of *at least one* false positive under all-true-null is 40%. Reporting unadjusted "look, this one was significant!" from a pool of 10 tests is roughly a coin flip.

### Bonferroni

The Bonferroni correction multiplies each p-value by `k` (capped at 1), or equivalently requires `p < α/k`. Conservative — it controls the FWER but at the cost of statistical power.

In `compare_means`, set `adjust="bonferroni"`. The displayed p-values are adjusted (the `comp_stats` table holds adjusted values).

### Other methods passed through

`adjust="holm"`, `"hochberg"`, `"hommel"`, `"fdr_bh"`, `"fdr_by"`, etc., are passed through to `statsmodels.stats.multitest.multipletests`. Holm is uniformly more powerful than Bonferroni while still controlling FWER. FDR (Benjamini–Hochberg) controls the false discovery rate instead, which is less stringent than FWER for many-test settings.

### When to use each

- **Class assignments, ≤ ~10 pairs** → Bonferroni. Easy to explain, conservative, defensible.
- **Many tests (genomics-style)** → FDR (`"fdr_bh"`).
- **Pre-specified single pair** → no adjustment needed.
- **Omnibus question** ("does the mean differ somewhere across these k groups?") → fit a regression with the categorical as a predictor and read the F-statistic. The omnibus F-test is the proper non-pairwise answer.

### When you don't need to adjust

- You decided *before* looking at the data to focus on a single pair. Set `comb=["that_pair"]` and run one test.
- The user explicitly says "I just want pairwise descriptive comparisons, not a hypothesis test" — but then call them descriptive, not significant.

### Interaction with `alt_hyp`

Internally, when `alt_hyp != "two-sided"` and `adjust` is set, the implementation doubles the working alpha before passing to `multipletests`. This keeps Bonferroni's logic consistent with the one-sided framing. Practically: use the adjusted p-values as printed.

## 7. t-test vs Wilcoxon

### t-test (`test_type="t-test"`)

- For `sample_type="independent"`: Welch's t-test (`scipy.stats.ttest_ind` with `equal_var=False`). Does **not** assume equal variances across groups. Robust to moderate variance differences and to moderate non-normality for n ≥ 15–30 per group.
- For `sample_type="paired"`: paired t-test (`scipy.stats.ttest_rel`). Tests whether the *within-pair differences* have mean zero.

P-value comes from a t-distribution. Welch's t-test uses an approximate (often non-integer) `df` via the Welch–Satterthwaite formula; pyrsm computes and displays this.

### Wilcoxon (`test_type="wilcox"`)

- For `sample_type="independent"`: Wilcoxon rank-sum test (`scipy.stats.ranksums`), equivalent to the Mann–Whitney U. Tests whether one distribution stochastically dominates the other (loosely: "are the medians equal?"). Doesn't assume normality. Robust to outliers.
- For `sample_type="paired"`: Wilcoxon signed-rank test (`scipy.stats.wilcoxon`). Tests whether the within-pair differences have a symmetric distribution around zero.

### Decision rule

- n ≥ 30 per group, distributions roughly symmetric, no extreme outliers → t-test.
- n < 15 per group, or skew, or outliers, or ordinal-ish data → Wilcoxon.
- Unsure → run both. Same conclusion → t-test is fine (probably). Different conclusions → trust the Wilcoxon, because the t-test's assumptions are likely violated.

### What Wilcoxon's printed `diff` means

Pyrsm prints `diff = mean(level1) − mean(level2)` even when `test_type="wilcox"`. The Wilcoxon test does *not* test the difference of means per se; it tests stochastic equality. When reporting a Wilcoxon result, prefer language like "the distribution of `<var2>` differs between groups (Wilcoxon p = `<p>`)" rather than "the means differ".

## 8. Independent vs paired samples

### Independent (default)

Different subjects in each group. Two-sample tests assume the observations across groups are independent.

### Paired

Same subjects measured twice, or naturally matched (twins, before-after, repeated measures). The test is on the *within-subject difference*.

### Power implications

Paired tests have higher power than independent tests on the same data, because they remove between-subject variance. But:

- **Misusing paired on truly independent data** inflates the false-positive rate (paired-test assumes a specific pairing that doesn't exist).
- **Misusing independent on truly paired data** wastes power (you'd reject more often with the correct paired test).

### Equal-size requirement

Paired samples require `len(x) == len(y)`. In long-form data, that means each level of `var1` has the same number of rows. The code raises a `ValueError` if not.

### Implementation note

Pyrsm computes `diff = mean(level1) − mean(level2)` regardless of `sample_type`. For paired samples, the t-statistic and p-value come from `scipy.stats.ttest_rel`, which uses the within-pair differences. For Wilcoxon paired, `scipy.stats.wilcoxon` is used.

## 9. The `comb` parameter — choosing which pairs to compare

`comb=[]` (default) → all pairwise combinations of `var1` levels are evaluated. With `k` levels that is `k*(k-1)/2` tests.

`comb=["level1:level2", "level1:level3"]` → only the listed pairs. The order matters: `"AsstProf:Prof"` computes `mean(AsstProf) - mean(Prof)`. Reversing the levels flips the sign of `diff` and the orientation of `alt_hyp`.

Use `comb` to:
- Focus on a small number of substantive contrasts (and reduce multiple-testing burden).
- Test specific hypotheses pre-registered before looking at the data.
- Compare against a single reference level (e.g., compare each treatment to control).

## 10. Related basics classes — when to switch

- **One sample, comparing its mean to a benchmark** → `single_mean`.
- **Comparing proportions (binary outcomes) across groups** → `compare_props`.
- **Three or more groups, want an omnibus F-test (does ANY level differ?)** → fit `rsm.model.regress` with the categorical as a predictor; read the F-statistic from the summary header.
- **Two categorical variables, association testing** → `cross_tabs`.
- **Continuous outcome, multiple predictors and/or covariates** → `regress`.
- **Probability calculations for the t-distribution** → `prob_calc("tdist", ...)`.

## 11. Worked example — `salary`

From `examples/basics/basics-compare-means.ipynb`:

> "We have the 2008-09 nine-month academic salary for Assistant, Associate, and Full Professors in a US college. Suppose we want to test if professors of lower rank earn lower salaries compared to those of higher rank."

### Basic call (one-sided "less", three pairs)

```python
import polars as pl
import pyrsm as rsm

salary = pl.read_parquet("<abs-path>/salary.parquet")
cm = rsm.basics.compare_means(
    {"salary": salary}, var1="rank", var2="salary", alt_hyp="less"
)
cm.summary(extra=True, dec=3)
```

Output (paraphrased):

```
Descriptive: Prof: mean=126772, n=266; AsstProf: mean=80776, n=67; AssocProf: mean=93876, n=64

Pairwise comparisons (3 pairs, alt_hyp="less"):
  AsstProf vs AssocProf: diff=-13100, p<.001, t=-6.561, df=101.3,  upper bound=-9786,  ***
  AsstProf vs Prof:      diff=-45996, p<.001, t=-23.334, df=324.3, upper bound=-42744, ***
  AssocProf vs Prof:     diff=-32896, p<.001, t=-13.569, df=199.3, upper bound=-28889, ***
```

Interpretation walkthrough:

1. **Setup.** 3 ranks, so 3 pairwise comparisons. Alt-hyp "less" means we're testing whether each lower rank earns *less* than the named higher rank.
2. **Descriptive.** Prof earns ~$127K on average (n=266, large). AssocProf ~$94K (n=64). AsstProf ~$81K (n=67).
3. **Pair AsstProf vs AssocProf.** diff = -$13,100, p < .001, 95% one-sided upper bound = -$9,786. Comparison value 0 is well above the upper bound → reject. AsstProf earn significantly less than AssocProf.
4. **Pair AsstProf vs Prof.** diff = -$45,996, p < .001. Even larger gap; same direction.
5. **Pair AssocProf vs Prof.** diff = -$32,896, p < .001. Same direction.
6. **Conclusion.** All three pairwise comparisons strongly support the hierarchical pay structure: salaries increase with rank. Effect sizes are large ($13K-$46K differences relative to a population mean around $113K).

### With Bonferroni adjustment

```python
cm_bonf = rsm.basics.compare_means(
    {"salary": salary}, var1="rank", var2="salary", alt_hyp="less", adjust="bonferroni"
)
cm_bonf.summary()
```

With 3 tests and all raw p < .001, the Bonferroni-adjusted p-values are still < .003 — all three pairs survive. The substantive conclusion is unchanged, but reviewers will appreciate that you did the check.

### Wilcoxon variant on salary by sex

From the notebook's "Additional examples":

```python
cm_w = rsm.basics.compare_means(
    salary, var1="sex", var2="salary", test_type="wilcox"
)
cm_w.summary()
# Female vs Male: diff = -$14,088, p = 0.008, **
```

Wilcoxon two-sided. Females in the sample earn about $14K less than males on average; the rank-sum test detects this at p ≈ 0.008. **Caveat**: the sample is heavily imbalanced (358 males vs 39 females), so the female group's standard error dominates inference — and confounders (rank, discipline, yrs_service) are not controlled. For a real pay-equity question, fit a regression with those controls.

## 12. Common pitfalls

- **Cherry-picking significant pairs.** With k levels you have k*(k-1)/2 pairs and a substantial family-wise false-positive rate. Always apply Bonferroni (or equivalent) before claiming individual pair significance from a pool of comparisons.
- **Confusing omnibus vs pairwise.** Pairwise tests answer "does this specific pair differ?". Omnibus F (from a regression with the categorical predictor) answers "does ANY level differ?". They are different questions with different p-values.
- **Picking `alt_hyp="greater"` or `"less"` after looking at the sample.** Doubles the effective false-positive rate. Choose direction from the research question, not the data.
- **One-sided pairwise with mixed-direction expectations.** If the research question is "rank determines pay" without a specified direction, use two-sided. If it is "higher rank ⇒ higher pay", one-sided "less" with pairs as "lower_rank:higher_rank" is correct.
- **Welch ≠ Student.** The default t-test in `compare_means` is Welch's (no equal-variance assumption). If the user expects a textbook "pooled-variance" Student's t-test, the numbers will differ slightly. Welch is preferred in most modern practice.
- **Wilcoxon when you really want a mean comparison.** Wilcoxon tests stochastic equality, not means. With non-normal data the printed `diff` (a mean difference) and the Wilcoxon p-value answer slightly different questions. State conclusions in distribution / location-shift language when using Wilcoxon.
- **Paired with unequal group sizes.** Raises a `ValueError`. If your two samples really are paired, they must have the same number of rows.
- **Unequal sample sizes.** Welch's t-test handles unequal n with unequal variance, but very unequal n (say, 10 vs 500) makes the small-group SE dominate. Note the imbalance explicitly in writeups.
- **Forgetting the dict wrapper.** `compare_means(df, ...)` works but the summary shows `"Not provided"`. Use `compare_means({"<name>": df}, ...)`.
- **Reading adjusted and unadjusted p-values interchangeably.** Once `adjust` is set, `comp_stats["p.value"]` holds the *adjusted* values. If you need both, fit twice.
