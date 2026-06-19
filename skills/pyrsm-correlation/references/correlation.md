# pyrsm.basics.correlation — reference

This file is the deeper reference for `pyrsm.basics.correlation`. The main `SKILL.md` walks the workflow at a high level; come here for API details, interpretation templates, and worked examples.

## Table of contents

1. Constructor signature
2. `summary()` — what each option prints
3. `plot()` — what the scatter matrix shows
4. Output attributes
5. Plain-English interpretation templates
6. Choosing the method (Pearson, Spearman, Kendall, polychoric)
7. When r=0 lies — non-linearity, outliers, sub-populations
8. Multicollinearity for downstream regression
9. Large-n significance inflation and multiple-testing
10. Related basics classes — when to switch
11. Worked example (`salary`)
12. Common pitfalls

---

## 1. Constructor signature

```python
rsm.basics.correlation(
    data,                       # polars/pandas DataFrame, OR {"name": df}
    vars=[],                    # list of column names; [] auto-selects numeric columns
    method="pearson",           # "pearson", "spearman", "kendall", "polychoric"
)
```

**Always pass data as `{"name": df}`** — the `name` becomes the dataset label in the printed summary. Plain `df` works, but the summary prints `"Not provided"`.

`vars` is a list of column names to include. The default `vars=[]` triggers auto-selection of all numeric columns. If you pass explicit names, every one must reference a numeric column for Pearson/Spearman/Kendall (polychoric will cast string/categorical columns to numeric codes — see §6).

`method` chooses the correlation coefficient. Spearman, Kendall, and Pearson share most of the API (correlation matrix, p-value matrix, covariance matrix); polychoric is more limited (only the correlation matrix is computed, no p-values).

Missing values are handled pairwise: for each pair, rows where either value is missing are dropped. This means different pairs may be computed on different subsets of the data — usually fine, occasionally surprising.

## 2. `summary()` — what each option prints

```python
cr.summary(
    cov=False,                  # also print the covariance matrix
    cutoff=0,                   # hide cells with |r| < cutoff
    dec=2,                      # decimal places
    plain=True,                 # plain text vs great_tables in Jupyter
)
```

Output structure (plain mode):

- **Header** — dataset name, method, cutoff value.
- **Hypothesis statements** — H₀: variables x and y are not correlated. Hₐ: variables x and y are correlated.
- **Correlation matrix** — lower-triangle display. Rows and columns are labeled by variable. The upper triangle (and the diagonal) is blank to avoid duplication.
- **P-values** — same lower-triangle layout, with the p-value for each pair. **Not printed for polychoric** (no p-values available).
- **Covariance matrix** — printed only if `cov=True`.

`cutoff` is a *display* filter: cells with `|r| < cutoff` print as empty strings. The full matrix is always computed; this just shrinks the printed view for large matrices. `cutoff` does not affect `cr.cr`, `cr.cp`, or `cr.cv`.

The summary header includes generic `x` and `y` placeholders in the hypothesis statement when more than two variables are involved; for two-variable tables it substitutes the actual names. The hypothesis applies pair-by-pair.

## 3. `plot()` — what the scatter matrix shows

```python
cr.plot(
    nobs=1000,                  # number of points to scatter; -1 for all
    dec=2,                      # decimal places for r in the upper triangle
    figsize=None,               # (width, height); auto-sized if None
)
```

The plot is a square grid of `k × k` panels for `k` variables:

- **Diagonal**: the variable name centered in the panel.
- **Lower triangle**: scatter of `var_j` (x) vs `var_i` (y), with a fitted simple linear regression line in blue.
- **Upper triangle**: the correlation coefficient as text, sized proportional to `|r|`, with significance stars (`***` < 0.001, `**` < 0.01, `*` < 0.05, `.` < 0.1).

The plot is a `matplotlib.figure.Figure` (not plotnine). In Jupyter it displays inline. Returned figure can be saved with `fig.savefig(...)`.

`nobs` controls only the scatter subsample, not the correlation computation. The correlations on the upper triangle are always computed on all non-missing data.

Why the plot is essential: a Pearson r of 0.0 can mean (a) the variables are unrelated, (b) the relationship is exactly U-shaped, or (c) two sub-populations cancel each other out. The scatter visually disambiguates. Always look.

## 4. Output attributes

After the constructor runs:

| Attribute | Type | Meaning |
| --- | --- | --- |
| `cr.cr` | `np.ndarray` (k × k) | Symmetric correlation matrix. Diagonal is 0 (not 1) by construction; off-diagonals are the pairwise correlations. |
| `cr.cp` | `np.ndarray` (k × k) | P-value matrix; symmetric, diagonal 0. Not populated for polychoric. |
| `cr.cv` | `np.ndarray` (k × k) | Covariance matrix; symmetric, diagonal 0. Not populated for polychoric. |
| `cr.vars` | list[str] | Variable names in the order they appear in the matrix. |
| `cr.method` | str | Echo of the method used. |
| `cr.data` | `pl.DataFrame` | The (possibly subsetted) data actually used. |
| `cr.name` | str | Dataset name (from the dict wrapper, or "Not provided"). |

Note: the diagonals of `cr.cr`, `cr.cp`, `cr.cv` are populated with 0, not 1 (and the p-values aren't NaN). This is a pyrsm choice to keep the display logic simple; it doesn't affect the printed lower-triangle.

## 5. Plain-English interpretation templates

Use these templates verbatim (substituting variable names and units).

### Header

> We computed `<method>` correlations among `<list-vars>`. For each pair, the null hypothesis is that the population correlation is zero; the alternative is that it is non-zero. With `n = <n>` observations, p-values come from a `<t-test for Pearson | rank-based test for Spearman | tau-based test for Kendall>`.

### Per-pair, magnitude language

> The correlation between `<var_a>` and `<var_b>` is `<r>`. This is a `<small | moderate | large | very large>` `<positive | negative>` correlation, indicating that higher values of `<var_a>` tend to go with `<higher | lower>` values of `<var_b>` in this sample. The relationship is `<statistically significant at p < .001 | significant at p = <p> | not significant>`.

### Effect-size context

> The squared correlation `<r²>` (about `<r² * 100>`%) is the share of variance one variable can explain about the other under a *linear* model. (Note: this is the same as R² in a simple regression of one on the other.)

### Multicollinearity flag

> `<var_a>` and `<var_b>` are very strongly correlated (r = `<r>`). If you intend to use both as predictors in a downstream regression, expect inflated standard errors and possibly counter-intuitive coefficient signs (multicollinearity). Consider dropping one, combining them into an index, or using regularization.

### Outliers / non-linearity flag

> The scatter for `<var_a>` vs `<var_b>` shows `<one or two extreme points | a curved pattern | two distinct clusters>`. The Pearson r of `<r>` is `<dominated by | misleadingly small because of | averaged across>` this pattern. Compare with the Spearman r of `<r_s>`: `<they agree → Pearson is fine | they differ noticeably → the pattern is the issue>`.

## 6. Choosing the method (Pearson, Spearman, Kendall, polychoric)

### Pearson — linear

```
r = cov(x, y) / (sd(x) * sd(y))
```

Measures *linear* association. Range −1 to +1. P-value comes from a t-statistic `t = r * sqrt((n - 2) / (1 - r²))` on `n − 2` df.

**Use when**: both variables are continuous, plausibly linear relationship, no extreme outliers.

**Pitfalls**: insensitive to non-linear relationships (perfect quadratic → r ≈ 0); sensitive to outliers; meaningful only on interval-scaled data.

### Spearman — monotonic

Computes Pearson r on the ranks of the data. Captures any monotonic relationship — exponential, logarithmic, sigmoidal — not just linear.

**Use when**: variables are skewed, have outliers, or are ordinal (Likert scales coded as integers); relationship may be monotonic but is unlikely to be exactly linear.

**Robustness**: a single outlier moves only the rank, not the value, so its impact on Spearman is bounded.

**Trade-off**: throws away the actual values, which can be informative when the data are well-behaved.

### Kendall — rank-based, more robust on small samples

Counts pairs of observations as "concordant" (same direction) or "discordant" (opposite direction):

```
τ = (concordant − discordant) / (n * (n - 1) / 2)
```

Similar interpretation to Spearman but typically smaller in magnitude. Preferred by some practitioners for small samples and for data with many ties.

**Use when**: small n (< 30) and you want a rank-based measure; many ties in the data; reporting standard requires Kendall's tau specifically.

### Polychoric — latent-normal for ordinal pairs

Estimates the correlation between two *latent* normal variables that, when discretized at fitted thresholds, would produce the observed ordinal categories. Implementation calls `pyrsm.utils.polychoric_corr`.

**Use when**: both variables are ordinal Likert-style with relatively few levels (e.g., 5-point or 7-point scales); the underlying construct is plausibly continuous.

**Properties**: typically gives a larger |r| than Pearson or Spearman on the same data, because it corrects for the information loss due to discretization. No p-values, no covariance.

### Decision tree

```
Two ordinal Likert-style categoricals?
  → polychoric
Both continuous, plausibly linear, well-behaved?
  → Pearson
Skewed, outliers, ordinal, or monotonic-but-not-linear?
  → Spearman (or Kendall for small n / many ties)
```

When in doubt, **run Pearson and Spearman both** and compare. If they agree, Pearson is fine. If they disagree noticeably (|Δ| > 0.1), look at the scatter — outliers or non-linearity is the cause.

## 7. When r=0 lies — non-linearity, outliers, sub-populations

`r = 0` does not mean "no relationship". It means "no linear association". Common counter-examples:

### Perfect non-linear relationship

`y = x²` on a symmetric range around 0 → r ≈ 0 (the linear best-fit slope is 0, but the relationship is deterministic). Always look at the scatter.

### Cancelling sub-populations

Two groups, one with positive slope and one with negative slope, plotted together. Overall r ≈ 0; within-group r is large and non-zero.

### Outlier-induced apparent independence

A bulk of points show a clear positive trend; one extreme point pulls the regression line so that r ≈ 0. Drop or downweight the outlier and the relationship reappears.

### Restricted range

A truncated x or y range can attenuate r toward 0. The relationship is real but invisible because the variance is artificially small.

### Threshold effects

`y = 0 if x < c, else f(x)` — half flat, half non-flat. Pearson r summarizes the average, missing the threshold.

All of these are reasons to **inspect the scatter matrix** alongside the correlation matrix. The numerical matrix is a summary; it is not the data.

## 8. Multicollinearity for downstream regression

If correlation is being computed as a screen before fitting a regression, the rule of thumb is:

- `|r| < 0.5` — usually fine.
- `0.5 ≤ |r| < 0.7` — be aware; check VIF after fitting.
- `0.7 ≤ |r| < 0.9` — likely multicollinearity; expect inflated SEs.
- `|r| ≥ 0.9` — severe; consider dropping one variable, combining them, or using regularization.

In the salary example, `yrs_since_phd` and `yrs_service` have r = 0.91 — severe collinearity. Fitting `regress(salary ~ yrs_since_phd + yrs_service + …)` will give unstable coefficients for those two; the F-test on the variable pair will be highly significant even when neither individual t-test is.

VIF (variance inflation factor) is the better post-fit diagnostic — pyrsm exposes it via `reg.summary(vif=True)` in `pyrsm-regress`.

## 9. Large-n significance inflation and multiple-testing

### Significance is cheap with large n

The p-value for Pearson r tests H₀: ρ = 0. With n in the thousands, p falls below 0.05 for `|r|` as small as `2 / sqrt(n)`. At n = 10,000 that's `|r| ≈ 0.02` — a correlation that explains 0.04% of the variance.

Always report magnitude alongside significance. r² is the easy effect-size translation: `r² × 100`% is the variance share under a linear model.

### Multiple-testing in matrices

A correlation matrix with k variables has `k * (k - 1) / 2` distinct pair tests. With k = 20, that's 190 pair tests. At α = 0.05 you'd expect about 9–10 false positives by chance alone if all population correlations were truly zero.

For exploratory analysis this is acceptable — you're not making formal claims about individual cells. For a writeup that *names* a specific pair as significant, apply a Bonferroni correction: divide α by the number of tests. With 20 vars, α' = 0.05 / 190 ≈ 0.00026, i.e., require p < .0003 to claim significance on one cell.

## 10. Related basics classes — when to switch

- **One continuous predictor and one continuous response, want to model (not just summarize)** → `regress` (`pyrsm-regress` skill).
- **Two categorical variables, association testing** → `cross_tabs` (`pyrsm-cross-tabs` skill).
- **One continuous, one categorical (two or more groups)** → `compare_means` (`pyrsm-compare-means` skill).
- **Single categorical distribution vs an expected one** → `goodness` (`pyrsm-goodness` skill).
- **Latent-factor structure across many variables (not just pairs)** → `pyrsm.multivariate` (PCA, factor analysis).
- **Probability calculations for the t-distribution to validate correlation p-values** → `prob_calc` (`pyrsm-prob-calc` skill).

## 11. Worked example — `salary`

From `examples/basics/basics-correlation.ipynb`:

> "The 2008–09 nine-month academic salary for Assistant, Associate, and Full Professors in a US college, with covariates rank, discipline, yrs_since_phd, yrs_service, sex, salary. Numeric columns are `salary`, `yrs_since_phd`, `yrs_service`."

### Pearson on numeric columns

```python
import polars as pl
import pyrsm as rsm

salary = pl.read_parquet("<abs-path>/salary.parquet")

cr = rsm.basics.correlation({"salary": salary})  # auto-selects numeric
cr.summary()
```

Output:

```
Correlation
Data     : salary
Method   : pearson
Cutoff   : 0
Variables: salary, yrs_since_phd, yrs_service
Null hyp.: variables x and y are not correlated
Alt. hyp.: variables x and y are correlated

Correlation matrix:
                 salary   yrs_since_phd
yrs_since_phd    0.42
yrs_service      0.33     0.91

p.values:
                 salary   yrs_since_phd
yrs_since_phd    0.00
yrs_service      0.00     0.00
```

Interpretation:

1. **Salary × yrs_since_phd**: r = 0.42 (moderate positive). Longer time since PhD → higher salary, on average. Plausible (seniority/tenure premium). p ≈ 0 → significant.
2. **Salary × yrs_service**: r = 0.33 (moderate positive). Same direction, slightly weaker.
3. **yrs_since_phd × yrs_service**: r = 0.91 (very large). The two tenure variables are nearly redundant — most professors got their PhDs and joined this college around the same time. **Multicollinearity flag** for any downstream regression.

### Subset call with explicit `vars`

```python
cr2 = rsm.basics.correlation(
    {"salary": salary[["salary", "yrs_since_phd", "yrs_service"]]}
)
cr2.cr  # same matrix
```

(Subsetting via column selection vs `vars=[...]` is interchangeable — both work.)

### Spearman comparison

```python
cr_s = rsm.basics.correlation({"salary": salary}, method="spearman")
cr_s.summary()
# Salary × yrs_since_phd: r = 0.48 (Spearman) vs 0.42 (Pearson)
```

Spearman is slightly larger than Pearson here — consistent with the relationship being monotonic but slightly non-linear (the salary scale is right-skewed and the yrs_since_phd–salary relationship probably has diminishing returns at very long tenures). Worth knowing, not dramatic.

### Cutoff for skimming

```python
cr.summary(cutoff=0.5)
# Only the yrs_since_phd × yrs_service pair (r = 0.91) survives.
```

### Plot

```python
cr.plot(figsize=(7, 7))
```

The lower triangle shows scatter plots: `salary × yrs_since_phd` has a positive cloud with a fitted line; `yrs_since_phd × yrs_service` is nearly a tight diagonal. The upper triangle echoes the correlations with significance stars.

## 12. Common pitfalls

- **Reporting r without looking at the scatter.** r is a one-number summary of a two-dimensional cloud. Always plot.
- **Treating r = 0 as "no relationship".** It means no *linear* relationship. The variables could still be perfectly related (quadratically, cyclically, in a sub-population). Plot.
- **Pearson on heavy-tailed or outlier-laden data.** A single extreme observation can swing r by 0.5+. If Pearson and Spearman disagree, outliers/non-linearity are why — switch methods or report both.
- **Multicollinearity ignored.** A correlation matrix is the cheapest screen for it. Flag `|r| > 0.7` pairs before fitting a regression.
- **Reading significance with large n.** With n ≥ 10,000, almost any r ≠ 0 is "significant". Pair with magnitude (`r²`).
- **Family-wise false positives in large matrices.** A 20-variable matrix has 190 pair tests; ~10 false positives at α = 0.05. Apply a multiple-testing correction if you're going to claim significance on individual cells.
- **Confusing correlation with causation.** r is symmetric; the analysis is correlational; without a research design that addresses confounding, "X correlates with Y" is not evidence for "X causes Y".
- **Using Pearson on ordinal Likert data with few levels.** Acceptable but inefficient. Spearman is better; polychoric is better still if both variables are ordinal.
- **Forgetting the dict wrapper.** `correlation(df, ...)` works but the summary shows `"Not provided"`. Use `correlation({"<name>": df}, ...)`.
- **Confusing `vars=[]` (auto-select numeric) with `vars=None` (would raise).** Pass an empty list, not None.
