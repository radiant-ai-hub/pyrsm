# pyrsm.basics.compare_props — reference

This file is the deeper reference for `pyrsm.basics.compare_props`. The main `SKILL.md` walks the workflow at a high level; come here for API details, interpretation templates, and worked examples.

## Table of contents

1. Constructor signature
2. `summary()` — what each option prints
3. `plot()` — what each plot type shows
4. Output attributes
5. Plain-English interpretation templates
6. Choosing `lev` (the success level)
7. The chi-squared cell-count assumption — the critical concept
8. Multiple-testing adjustment
9. Wald confidence intervals — what `compare_props` reports
10. Related basics classes — when to switch
11. Worked example (`titanic`)
12. Common pitfalls

---

## 1. Constructor signature

```python
rsm.basics.compare_props(
    data,                            # polars/pandas DataFrame, OR {"name": df}
    var1,                            # grouping variable (categorical)
    var2,                            # response variable (categorical)
    lev,                             # level of var2 that counts as "success"
    comb=[],                         # list of "level1:level2" pair strings; [] = all pairs
    alt_hyp="two-sided",             # "two-sided", "greater", "less"
    conf=0.95,                       # confidence level
    adjust=None,                     # None, "bonferroni", or other statsmodels method
)
```

**Always pass data as `{"name": df}`** — the `name` becomes the dataset label in the printed summary.

`var1` and `var2` must both be categorical (string, Categorical, or Enum). `lev` must be one of the levels of `var2`. Level order for `var1` comes from the Enum's `dtype.categories` if it's an Enum, otherwise alphabetical sort of unique values.

The test runs `statsmodels.stats.proportion.proportions_ztest` for each pair. The chi-squared value reported is `z**2`. The CI is the **Wald interval** for the difference of proportions:

```
SE_diff = sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
CI = (p1 - p2) ± z_crit * SE_diff
```

This is what R's `prop.test` returns. It's not the pooled-SE used internally for the z-test, which is one reason the CI and the p-value occasionally seem to disagree at the boundary. For a class assignment, treat them as consistent.

## 2. `summary()` — what each option prints

```python
cp.summary(
    extra=False,                     # if True, add chisq.value, df, CI columns
    dec=3,                           # decimal places
    plain=True,                      # plain text vs great_tables in Jupyter
)
```

Output structure (plain mode):

- **Header** — dataset, variables, level, confidence, adjustment.
- **Descriptive statistics** — per group: `<var1>`, `<lev>` (number of successes), `p` (proportion), `n` (non-missing), `n_missing`, `sd`, `se`, `me`.
- **Pairwise comparisons** — one row per pair in `comb`:
  - Always: `Null hyp.`, `Alt. hyp.`, `diff`, `p.value`, significance stars.
  - With `extra=True`: also `chisq.value`, `df` (always 1), lower CI bound, upper CI bound.

For a first look, **pass `extra=True`** — the chi-squared and CI columns are informative.

## 3. `plot()` — what each plot type shows

```python
cp.plot(plots="bar")    # or "dodge"
```

| `plots=` | Shows |
| --- | --- |
| `"bar"` (default) | One bar per group, height = proportion of `lev`. Quick comparison of rates across groups. |
| `"dodge"` | Side-by-side bars for each level of `var2` within each `var1` group. Useful when `var2` has more than two levels, or for showing the full breakdown. |

Both return plotnine ggplot objects.

## 4. Output attributes

| Attribute | Type | Meaning |
| --- | --- | --- |
| `cp.descriptive_stats` | `pl.DataFrame` | Per-group: `var1`, `lev` count, `p`, `n`, `n_missing`, `sd`, `se`, `me`. |
| `cp.comp_stats` | `pl.DataFrame` | One row per pair: `Null hyp.`, `Alt. hyp.`, `diff`, `p.value`, `chisq.value`, `df`, CI bounds, stars. |
| `cp.levels` | list[str] | Level order of `var1`. |
| `cp.comb` | list[str] | Pair strings actually evaluated. |
| `cp.var1`, `cp.var2`, `cp.lev`, `cp.alt_hyp`, `cp.conf`, `cp.adjust`, `cp.name` | various | Echoes of inputs. |
| `cp.data` | `pl.DataFrame` | Underlying data. |
| `cp.alpha` | float | `1 - conf`. |

`cp.comp_stats["p.value"]` is the **adjusted** p-value when `adjust` is set; the original unadjusted p-values are not retained on the object.

## 5. Plain-English interpretation templates

### Header

> We compared the proportion of `<lev>` in `<var2>` across the `<k>` levels of `<var1>` (`<level_1>`, `<level_2>`, …). With `<k>` levels there are `<k*(k-1)/2>` pairwise comparisons. `<conf*100>`% confidence. Adjustment: `<None | bonferroni | …>`.

### Descriptive

> Per group: `<level_1>` had `<ns_1>` of `<n_1>` (`<p_1*100>`%); `<level_2>` had `<ns_2>` of `<n_2>` (`<p_2*100>`%); …

### Pairwise hypothesis

> For the comparison `<level_a>` vs `<level_b>`:
> H₀: the population proportion of `<lev>` is equal in `<level_a>` and `<level_b>`.
> Hₐ: the population proportion of `<lev>` in `<level_a>` is `<less than | greater than | not equal to>` that in `<level_b>`.

### Three-way verdict for a pair

> diff = `<diff>` (i.e., `<diff*100>` percentage points). p = `<p>`. 95% CI for the difference is [`<lo>`, `<hi>`]. chisq.value = `<chisq>` on 1 df, critical chisq at α=0.05 = 3.841.
> Because p `<is | is not>` smaller than α = 0.05, and 0 `<is | is not>` inside the CI, and chisq `<exceeds | does not exceed>` 3.841, we `<reject | fail to reject>` H₀ for this pair.

### Effect size

> The proportion of `<lev>` in `<level_a>` is `<diff*100>` percentage points `<higher | lower>` than in `<level_b>` (`<p_a*100>`% vs `<p_b*100>`%, 95% CI for the difference [`<lo*100>` pp, `<hi*100>` pp]). Relative risk: `<p_a / p_b>`x.

## 6. Choosing `lev` (the success level)

`lev` is the level of `var2` that counts as "success" in each group. It is a **substantive choice** tied to the research question.

### Statistical symmetry, substantive asymmetry

`lev="Yes"` and `lev="No"` give the **same** p-value and chi-squared statistic — the z-statistic flips sign but `z**2` is identical. So in a purely statistical sense, the choice doesn't matter.

In a **substantive** sense, it does:

- The reported `diff` flips sign when you flip `lev`.
- The CI flips orientation.
- The framing of the alt-hyp changes ("more likely to survive" vs "more likely to die").
- The plain-English interpretation changes: "1st class had a 19 pp higher survival rate than 2nd class" vs "1st class had a 19 pp lower death rate than 2nd class" — same fact, different emphasis.

Pick `lev` from the **business decision** the test informs: which framing is most natural for the audience? For survival analysis, `lev="Yes"` (survived) is the conventional positive framing.

### Don't pick `lev` after seeing the result

Switching `lev` after looking at the data because the other framing has a "more striking" `diff` is the proportion-equivalent of switching `alt_hyp` after the fact. The substantive interpretation should be set by the question, not by the data.

### Typos and casing

`lev` must exactly match a level of `var2`. `"Yes"` and `"yes"` are different. If the test silently returns `ns = 0` for every group, you've probably mis-cased. Check `df[var2].value_counts()`.

## 7. The chi-squared cell-count assumption — the critical concept

The two-proportion z-test (and equivalent chi-squared test) approximates a discrete sampling distribution by a continuous one. The approximation requires enough expected counts in every cell of the implicit 2×2 table.

### The implicit 2×2 table for a pair

For pair `(v1, v2)`:

|        | success | failure |
| ------ | ------- | ------- |
| `v1`   | c1      | n1 − c1 |
| `v2`   | c2      | n2 − c2 |

Pooled proportion under H₀:

```
p_pool = (c1 + c2) / (n1 + n2)
```

Expected cell counts under H₀:

```
e1_success = n1 * p_pool
e1_failure = n1 * (1 - p_pool)
e2_success = n2 * p_pool
e2_failure = n2 * (1 - p_pool)
```

### The rule of thumb

> All four expected cell counts should be at least 5.

If any is below 5, the chi-squared / z-test approximation is biased. With expected counts in single digits or low single digits, the p-value can be substantially wrong.

### What pyrsm does

`compare_props` does **not** check this assumption automatically. It will compute and print a p-value and CI regardless of cell counts. The skill needs to check.

### Quick Python check

```python
# Given a pair (n1, c1, n2, c2):
p_pool = (c1 + c2) / (n1 + n2)
expected = [n1*p_pool, n1*(1-p_pool), n2*p_pool, n2*(1-p_pool)]
min_expected = min(expected)
if min_expected < 5:
    print(f"WARNING: smallest expected cell = {min_expected:.2f}, chi-squared p-value is unreliable.")
```

### What to do when violated

- **Collapse categories**. For `var1`, merge a sparse level with an adjacent one. For `var2`, this usually doesn't apply since it's binary.
- **Fisher's exact test**. Not exposed by pyrsm. Compute via `scipy.stats.fisher_exact` on the 2×2 sub-table:
  ```python
  from scipy.stats import fisher_exact
  odds_ratio, p_fisher = fisher_exact([[c1, n1 - c1], [c2, n2 - c2]])
  ```
- **More data**. Sometimes feasible.

### Why this is the critical concept

Students reach for the proportion test because it produces tidy z-statistics, CIs, and stars. They trust the printed numbers. When the underlying assumption is violated — small groups with rare outcomes — those numbers are still printed, but they are misleading. The skill's job is to make the assumption check visible before letting the user proceed.

## 8. Multiple-testing adjustment

Same conceptual concern as in `compare_means`. With `k` levels of `var1`, you have `k*(k-1)/2` pairwise tests. The family-wise error rate (FWER) grows with the number of tests.

### Bonferroni

`adjust="bonferroni"` multiplies each raw p-value by the number of pairs (capped at 1). The displayed p-values are the adjusted ones.

### Other methods

`"holm"`, `"hochberg"`, `"fdr_bh"`, etc., are passed through to `statsmodels.stats.multitest.multipletests`. Bonferroni is the safest class default.

### When to skip

- Pre-specified single pair (`comb=["that_pair"]`).
- Omnibus association question ("is `var1` associated with `var2` at all?") — use `cross_tabs`.

### When to always adjust

- Multiple pairs run together.
- Reporting "the pairs that came back significant" from a wider fishing expedition.

### Interaction with `alt_hyp`

Internally, when `adjust` is set and `alt_hyp != "two-sided"`, the implementation passes `alpha = self.alpha * 2` to `multipletests` to keep the framework consistent. The displayed p-values are the adjusted values.

## 9. Wald confidence intervals — what `compare_props` reports

The CI for the difference of proportions is the **Wald interval**:

```
diff = p1 - p2
SE_diff = sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
CI = diff ± z_crit * SE_diff
```

This is the formula R's `prop.test` uses for the CI. It is **not** the same as the SE used internally for the z-test (which pools the variance under H₀). The result: the p-value and the CI use slightly different SE formulas. In practice the discrepancy is small, but it can produce edge cases where the CI just barely contains 0 while the p-value is just barely below 0.05 — or vice versa.

For class work, treat them as consistent. For more rigorous comparison, the CI is the more honest report (it doesn't assume H₀).

### One-sided CIs

`alt_hyp="less"` → CI = `[-1, diff + z_crit * SE_diff]`.
`alt_hyp="greater"` → CI = `[diff - z_crit * SE_diff, 1]`.

The −1 and +1 are the natural bounds of a difference of two proportions, each in [0, 1].

## 10. Related basics classes — when to switch

- **One group, comparing its proportion to a benchmark** → `single_prop` (`pyrsm-single-prop` skill; supports binomial-exact).
- **Two categorical variables, omnibus association test (any number of levels in either)** → `cross_tabs` (`pyrsm-cross-tabs` skill).
- **One categorical, testing against an expected distribution** → `goodness` (`pyrsm-goodness` skill).
- **Continuous outcome, group comparisons** → `compare_means`.
- **Modeling how the probability of success depends on covariates** → `logistic` (`pyrsm-logistic` skill).
- **Critical values / probabilities** → `prob_calc` (`pyrsm-prob-calc` skill).

## 11. Worked example — `titanic`

From `examples/basics/basics-compare-props.ipynb`:

> "We want to test if the proportion of people that survived the sinking of the Titanic differs across passenger classes (1st / 2nd / 3rd, a proxy for socio-economic status). `survived` is yes/no; `pclass` has three levels."

### Focused contrasts

```python
import polars as pl
import pyrsm as rsm

titanic = pl.read_parquet("<abs-path>/titanic.parquet")
cp = rsm.basics.compare_props(
    {"titanic": titanic},
    var1="pclass",
    var2="survived",
    comb=["1st:2nd", "1st:3rd"],   # the substantive contrasts of interest
    lev="Yes",
    alt_hyp="two-sided",
    conf=0.95,
)
cp.summary(extra=True)
```

Output (paraphrased):

```
Descriptive: 1st: ns=179, p=0.635, n=282; 2nd: ns=115, p=0.441, n=261; 3rd: ns=131, p=0.262, n=500

Pairwise comparisons (2 pairs, adjustment=None):
  1st vs 2nd: diff=0.194, p<.001, chisq=20.58, df=1, 95% CI [0.112, 0.277], ***
  1st vs 3rd: diff=0.373, p<.001, chisq=104.70, df=1, 95% CI [0.305, 0.441], ***
```

Interpretation walkthrough:

1. **Setup.** 3 levels of `pclass`; user requested 2 focused contrasts (vs all 3 pairwise). No adjustment chosen — with only 2 tests, Bonferroni would multiply p by 2, both still < .001.
2. **Cell-count check.** Pool for 1st vs 2nd: `p_pool = (179+115)/(282+261) = 0.542`. Expected cells: `282*0.542=152.8`, `282*0.458=129.2`, `261*0.542=141.5`, `261*0.458=119.5`. All ≫ 5. Same check for 1st vs 3rd: `p_pool = 0.396`; expected cells all > 100. Assumption clearly satisfied.
3. **Pair 1st vs 2nd.** 63.5% vs 44.1% — a 19.4 percentage-point gap. p < .001, 95% CI [11.2 pp, 27.7 pp] doesn't contain 0, chi-sq 20.58 ≫ 3.841 → reject. 1st-class passengers had significantly higher survival than 2nd class.
4. **Pair 1st vs 3rd.** 63.5% vs 26.2% — a 37.3 percentage-point gap. p < .001, 95% CI [30.5 pp, 44.1 pp], chi-sq 104.70 → reject.
5. **Business conclusion.** Socio-economic class (a known proxy for cabin location, lifeboat access, and survival opportunities on the Titanic) was strongly associated with survival.

### With Bonferroni and all 3 pairs

```python
cp_all = rsm.basics.compare_props(
    titanic, var1="pclass", var2="survived", lev="Yes", adjust="bonferroni"
)
cp_all.summary()
```

All 3 pairwise comparisons (including 2nd:3rd, which the focused call skipped) come back at adjusted p < .001 — substantive conclusion unchanged. 2nd:3rd: diff ≈ 0.179, also significant.

### `lev="No"` to see the mirror framing

```python
cp_no = rsm.basics.compare_props(
    titanic, var1="pclass", var2="survived", lev="No", comb=["1st:3rd"]
)
cp_no.summary()
# diff = -0.373 (note the sign flip)
# p < .001 (same)
# CI [-0.441, -0.305] (mirror of the Yes CI)
```

Same statistical conclusion; different framing ("1st class had a 37.3 pp lower death rate than 3rd class").

## 12. Common pitfalls

- **Picking `lev` after looking at the data.** The same numerical test; different substantive framing. Pick from the research question.
- **Cherry-picking significant pairs.** With multiple pairs, FWER inflates fast. Use `adjust="bonferroni"` or pre-specify pairs via `comb=`.
- **Picking `alt_hyp` after seeing direction.** Doubles the false-positive rate. Pick from the research question.
- **Ignoring the chi-squared cell-count assumption.** With small groups or rare outcomes, the printed p-value is unreliable. Always check expected cells; use Fisher's exact when violated.
- **Misreading `lev` for `var2` (typos and casing).** Silently returns `ns = 0` for every group. Check `df[var2].value_counts()`.
- **Forgetting the dict wrapper.** `compare_props(df, ...)` works but the summary shows `"Not provided"`. Use `compare_props({"<name>": df}, ...)`.
- **Reading `chisq.value` as if it's the omnibus chi-squared from cross_tabs.** Each pair's `chisq.value` is for that 2×2 sub-table, df=1. The cross_tabs omnibus chi-squared for the full `pclass × survived` table is different (different df, tests a different hypothesis).
- **Confusing the p-value and CI bases.** The CI is Wald (uses sample proportions); the p-value uses pooled-variance. Tiny edge-case discrepancies can occur; the CI is the more honest one to report.
- **Using `compare_props` when the omnibus question is the real one.** "Is there ANY association between pclass and survived?" is `cross_tabs`. "Is survival different in 1st vs 2nd?" is `compare_props`. Use the right tool.
- **Reporting only adjusted p-values without saying so.** Always state the adjustment method in the writeup. Bonferroni-adjusted p = 0.04 came from a raw p smaller than 0.04 / k.
