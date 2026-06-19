# pyrsm.basics.cross_tabs — reference

This file is the deeper reference for `pyrsm.basics.cross_tabs`. The main `SKILL.md` walks the workflow at a high level; come here for API details, interpretation templates, and worked examples.

## Table of contents

1. Constructor signature
2. `summary()` — what each option prints
3. `plot()` — what each plot type shows
4. Output attributes
5. Plain-English interpretation templates
6. The expected-cell-count assumption — the critical concept
7. Standardized deviations as cell-level effect size
8. Row vs column vs total percentages
9. Relationship to `compare_props` and `goodness`
10. Worked example (`newspaper`)
11. Common pitfalls

---

## 1. Constructor signature

```python
rsm.basics.cross_tabs(
    data,                       # polars/pandas DataFrame, OR {"name": df}
    var1,                       # categorical (row variable)
    var2,                       # categorical (column variable)
)
```

**Always pass data as `{"name": df}`** — the `name` becomes the dataset label in the printed summary.

Both `var1` and `var2` must be categorical (string, Categorical, or Enum). Level order comes from the Enum's `dtype.categories` if it's an Enum, otherwise alphabetical sort of unique values.

The chi-squared test of independence is **symmetric** in `var1` and `var2`. Which one you call the "row variable" and which the "column variable" is conventional, not statistical. Pick the assignment that makes the printed table easiest to read — typically, fewer levels on the column axis.

Internally the constructor runs `scipy.stats.chi2_contingency(observed_matrix, correction=False)` and stores:
- The observed matrix and totals.
- The expected matrix under independence.
- The chi-squared contribution per cell.
- The standardized deviation per cell.
- The row, column, and total percentage tables.
- `expected_low`: a `[count_below_5, total_cells]` summary for the assumption check.

## 2. `summary()` — what each option prints

```python
ct.summary(
    output=["observed", "expected"],   # list of tables to print
    dec=2,                              # decimal places
    plain=True,                         # plain text vs great_tables in Jupyter
)
```

`output` is a list of table names:

| `output=` | Prints |
| --- | --- |
| `"observed"` | Observed counts with marginal totals. |
| `"expected"` | Expected counts under H₀ with marginal totals. |
| `"chisq"` | `(o - e)² / e` per cell with marginal totals (the row/column totals are themselves sums of chi-sq contributions, useful for spotting which row or column drives the test statistic). |
| `"dev_std"` | `(o - e) / √e` per cell. **No totals** — standardized deviations don't sum meaningfully. |
| `"perc_row"` | Row-conditional percentages (each row sums to 100%). |
| `"perc_col"` | Column-conditional percentages (each column sums to 100%). |
| `"perc"` | Cell as fraction of grand total. |

The summary footer always prints, regardless of `output`:

```
Chi-squared: <value> df(<R-1)*(C-1)>), p.value <p>
<X>% of cells have expected values below 5
```

The footer's "below 5" line is the **expected-count assumption check** — pay attention to it.

For an initial fit, **pass `output=["observed", "expected", "chisq", "dev_std"]`** — the diagnostic suite.

## 3. `plot()` — what each plot type shows

```python
ct.plot(plots="perc_col")   # default
ct.plot(plots=["observed", "expected", "dev_std"])   # composed
```

| `plots=` | Shows |
| --- | --- |
| `"observed"` | Stacked bar of counts. |
| `"expected"` | Stacked bar of expected counts. |
| `"chisq"` | Grouped bars of chi-sq contributions per cell. |
| `"dev_std"` | Grouped bars of standardized deviations with reference lines at ±1.96, ±2.58, ±3.29. **The single most informative diagnostic plot.** |
| `"perc_col"` (default) | Grouped bars of column percentages. |
| `"perc_row"` | Grouped bars of row percentages. |
| `"perc"` | Grouped bars of total percentages. |

Pass a list to compose multiple panels. All return plotnine ggplot (or composed) objects.

## 4. Output attributes

| Attribute | Type | Meaning |
| --- | --- | --- |
| `ct.observed` | `pl.DataFrame` | Observed counts with row, column, grand totals. |
| `ct.expected` | `pl.DataFrame` | Expected under H₀ with totals. |
| `ct.expected_low` | `[int, int]` | `[count_below_5, total_cells]` for the assumption check. |
| `ct.chisq` | `pl.DataFrame` | `(o − e)² / e` per cell with totals (the grand total is the test statistic). |
| `ct.dev_std` | `pl.DataFrame` | `(o − e) / √e` per cell. No totals. |
| `ct.perc_row` | `pl.DataFrame` | Row-conditional percentages. |
| `ct.perc_col` | `pl.DataFrame` | Column-conditional percentages. |
| `ct.perc` | `pl.DataFrame` | Cell-fraction-of-grand-total. |
| `ct.chisq_test` | tuple | Raw output of `scipy.stats.chi2_contingency` — `(chi2, p, dof, expected_array)`. |
| `ct.var1`, `ct.var2`, `ct.name` | various | Echoes of inputs. |

If you need the chi-squared statistic, df, or p-value programmatically:

```python
chi2, p_val, df_val, expected_arr = ct.chisq_test
```

## 5. Plain-English interpretation templates

### Hypotheses

> H₀: `<var1>` and `<var2>` are independent in the population (no association).
> Hₐ: `<var1>` and `<var2>` are associated (their joint distribution is not the product of the marginals).

### Walking the tables

> **Observed**: in our sample, `<n_grand>` observations are distributed across the `<R>×<C>` cells of the `<var1>×<var2>` table. Marginal totals are: `<row totals>` for `<var1>`, `<col totals>` for `<var2>`.

> **Expected under H₀**: each cell's expected count is `(row total × column total) / grand total`. Largest expected count is `<max>`; smallest is `<min>`.

> **Chi-squared contributions**: cell `(<i>,<j>)` contributes `<(o-e)²/e>` to the test statistic. The total chi-squared is `<sum>`.

> **Standardized deviations**: cells with `|dev_std| > 1.96` are individually over- or under-represented relative to independence at α = 0.05. Cells exceeding this threshold: `<list>`.

### Test verdict

> Chi-squared = `<chi2>` on `<df>` degrees of freedom, p-value `<p>`. Because p `<is | is not>` smaller than 0.05, we `<reject | fail to reject>` H₀. The data `<provide | do not provide>` statistically significant evidence of association between `<var1>` and `<var2>`.

### Assumption check

> The footer reports that `<X>%` of cells have expected counts below 5. `<None do | Some do>`, so the chi-squared approximation is `<reliable | possibly biased>`. `<No action needed | Consider collapsing categories or using Fisher's exact for 2×2>`.

### Cell-level story (after rejecting)

> The standardized-deviation table identifies which cells drive the rejection:
> - `<cell A>` has dev_std = `<value>` (positive) — `<over-represented>` relative to independence.
> - `<cell B>` has dev_std = `<value>` (negative) — `<under-represented>` relative to independence.
> Substantively: `<plain-English story tying cells back to the research question>`.

### Effect-size context

> The largest standardized deviation is `<value>` in the `(<var1>=<level>, <var2>=<level>)` cell — `<this cell is over/under-represented by this many standardized units>`. The total chi-squared scales with sample size, so a "highly significant" omnibus result can come from many tiny deviations (with large n) or a few large ones (with smaller n); the dev_std table tells you which.

## 6. The expected-cell-count assumption — the critical concept

The chi-squared test of independence is asymptotic. The test statistic follows a chi-squared distribution *under H₀ as n → ∞*; for finite n, the approximation requires enough expected counts in every cell.

### The rule of thumb

> Every expected cell count should be at least 5.

### What pyrsm reports

The footer prints `"<X>% of cells have expected values below 5"`. The `ct.expected_low = [count_below_5, total_cells]` attribute holds the underlying numbers.

For the newspaper example: `expected_low = [0, 4]` → 0% of cells below 5 → assumption fully satisfied. For a small 5×4 table with sparse cells, expected_low might be `[6, 20]` → 30% of cells below 5 → assumption substantively violated.

### Remedies when violated

- **Collapse adjacent categories** to merge low-count cells. For ordinal variables (rating scales, age brackets, income brackets) this is principled. For nominal variables, requires judgment about which merges are substantively defensible.
- **Use Fisher's exact test** for 2×2 tables. Not in pyrsm; compute via:
  ```python
  from scipy.stats import fisher_exact
  matrix = ct.observed.filter(pl.col(ct.var1) != "Total").select(ct._var2_levels).to_numpy()
  odds_ratio, p_fisher = fisher_exact(matrix)
  ```
- **For larger tables**, exact alternatives are computationally heavy. The chi-squared with caveats is often the most practical option; just state the caveat in the writeup.
- **Get more data.** Sometimes feasible.

### Why this matters more for cross_tabs than goodness

Both `goodness` and `cross_tabs` have this assumption. But `cross_tabs` has *more cells* (R×C vs k), and contingency tables tend to be sparser than 1-way frequency tables, so the violation is more common. The skill should always check.

## 7. Standardized deviations as cell-level effect size

The omnibus chi-squared statistic answers "the joint distribution is not the product of the marginals — somewhere". It does not tell you *where*. The standardized-deviation table does both:

```
dev_std = (observed - expected) / sqrt(expected)
```

Under H₀, dev_std is approximately standard normal for each cell. So:

- `|dev_std| > 1.96` → cell is individually significant at α = 0.05.
- `|dev_std| > 2.58` → at α = 0.01.
- `|dev_std| > 3.29` → at α = 0.001.

### Sign interpretation

- **Positive dev_std**: observed > expected. This cell is *more* common than independence predicts.
- **Negative dev_std**: observed < expected. This cell is *less* common than independence predicts.

For a 2×2 table, the four dev_std values are arithmetically tied to one another: if one cell is over-represented, the diagonal cell is too, and the off-diagonal cells are under-represented. So a 2×2 dev_std table mostly tells you *which direction* the association runs.

For R×C with R or C > 2, dev_std typically highlights a subset of cells as the driving ones — many cells will be close to zero while a few will be large.

### Multiple-comparisons caveat

If you have R×C cells and check every one for `|dev_std| > 1.96`, you have R×C family-wise tests. For exploratory diagnosis this is fine; for claims of cell-level significance, apply a Bonferroni-style threshold of `1.96 + ...` (or use `α / (R×C)`).

### Why this is the most important table

Students often stop at the omnibus chi-squared and report "they're associated". The dev_std table is what turns "associated" into a substantive story. Always look at it before writing up the result.

## 8. Row vs column vs total percentages

The three percentage tables answer three different questions. Mixing them up is a classic student mistake.

### `perc_row` — row-conditional

Each row sums to 100%. Answers: "given `var1=A`, what fraction is in each `var2` level?"

Example: "of low-income respondents (n=359), 23.1% read WSJ and 76.9% read USA Today."

### `perc_col` — column-conditional

Each column sums to 100%. Answers: "given `var2=X`, what fraction is in each `var1` level?"

Example: "of WSJ readers (n=263), 31.6% are low-income and 68.4% are high-income."

### `perc` — total

Each cell as a fraction of the grand total. The whole table sums to 100%. Answers: "what fraction of the entire sample is in this cell?"

Example: "14.3% of all respondents are low-income WSJ readers."

### Which to use

Depends on the research question:

- **"Does X predict / influence Y?"** → `perc_row` (with X as `var1`).
- **"Among observed Y, what's the composition of X?"** → `perc_col`.
- **"What's the joint distribution?"** → `perc`.

Always state which conditional you're using when reporting percentages.

## 9. Relationship to `compare_props` and `goodness`

`cross_tabs`, `compare_props`, and `goodness` all sit in the same statistical neighborhood. Which to use depends on the question:

- **`cross_tabs`** — two categorical variables; omnibus test of independence.
- **`compare_props`** — two categorical variables, but the question is *pairwise comparison of one level* across groups of the other variable (e.g., "is survival rate for 1st class different from 2nd class?"). More targeted; runs a separate z-test for each pair; supports Bonferroni adjustment.
- **`goodness`** — one categorical variable; test against a hypothesized distribution (uniform or specified).

For a 2×2 table, the omnibus chi-squared of `cross_tabs` and the single z-test of `compare_props` give equivalent results (chi² = z² and same p-value). For larger tables, they answer different questions: cross_tabs the omnibus, compare_props the per-pair.

## 10. Worked example — `newspaper`

From `examples/basics/basics-cross-tabs.ipynb`:

> "Data from a sample of 580 newspaper readers indicating which newspaper they read most frequently (USA Today or Wall Street Journal) and their level of income (Low / High). Does income level predict newspaper choice?"

```python
import polars as pl
import pyrsm as rsm

newspaper = pl.read_parquet("<abs-path>/newspaper.parquet")
ct = rsm.basics.cross_tabs({"newspaper": newspaper}, var1="Income", var2="Newspaper")
ct.summary(output=["observed", "expected", "chisq", "dev_std"])
```

Output:

```
Cross-tabs
Data     : newspaper
Variables: Income, Newspaper
Null hyp : There is no association between Income and Newspaper
Alt. hyp : There is an association between Income and Newspaper

Observed:
                WS Journal    USA Today   Total
Low Income            83          276     359
High Income          180           41     221
Total                263          317     580

Expected: (row total x column total) / total
                WS Journal    USA Today   Total
Low Income        162.79       196.21    359.0
High Income       100.21       120.79    221.0
Total              263.0        317.0    580.0

Contribution to chi-squared: (o - e)^2 / e
                WS Journal    USA Today   Total
Low Income         39.11        32.45    71.55
High Income        63.53         52.7   116.23
Total             102.63        85.15   187.78

Deviation standardized: (o - e) / sqrt(e)
                WS Journal    USA Today
Low Income          -6.25         5.70
High Income          7.97        -7.26

Chi-squared: 187.78 df(1), p.value < .001
0.0% of cells have expected values below 5
```

Interpretation walkthrough:

1. **Hypotheses.** H₀: Income and Newspaper are independent. Hₐ: they're associated.
2. **Sample.** n=580 across a 2×2 table. df = (2-1)(2-1) = 1.
3. **Observed vs expected.** Stark differences: low-income observed 83 WSJ vs 163 expected; high-income observed 180 WSJ vs 100 expected. Mirror in USA Today.
4. **Chi-squared.** 187.78 on 1 df, p < .001 → reject H₀. Critical chi-sq at α=0.05 with df=1 is 3.841; observed 187.78 ≫ 3.841.
5. **Assumption check.** 0% of cells below 5 → assumption fully satisfied. (Expected counts are 100+.)
6. **Standardized deviations.** All four cells exceed |dev_std| > 5 — much higher than 1.96. The 2×2 is constrained so the four are tied (if one is over, diagonally is too); the magnitudes are roughly symmetric.
7. **Direction.** WSJ over-represented among high-income (dev_std = +7.97), under-represented among low-income (-6.25). Mirror pattern for USA Today.
8. **Business conclusion.** Strong association: high-income readers favor WSJ; low-income readers favor USA Today. For a marketing decision, reaching high-income readers means buying WSJ ad space; reaching low-income, USA Today.

### Inspecting percentages

For the marketing question "what fraction of each income segment reads each paper?", use `perc_row`:

```python
ct.summary(output=["observed", "perc_row"])
```

`perc_row` shows, for low-income, ~23% read WSJ and ~77% read USA Today. For high-income, ~81% read WSJ and ~19% read USA Today. **State the row-conditional explicitly** when reporting — "of low-income readers, 77% chose USA Today" is the substantive sentence.

## 11. Common pitfalls

- **Stopping at the omnibus p-value.** The chi-squared says "associated somewhere"; the dev_std table says *where*. Always look at dev_std after rejecting.
- **Ignoring the expected-cell-count assumption.** Pyrsm prints the percentage automatically — pay attention. If any cells are below 5, the chi-squared p-value is biased. Collapse or use Fisher's exact (2×2).
- **Reading percentages without sample sizes.** "80% of group X read WSJ" is meaningless if group X has 5 people. Always pair percentages with `n`.
- **Confusing row and column percentages.** They answer different questions. State which conditional you're using.
- **Treating cross_tabs as a test of one variable's distribution.** That's `goodness`. cross_tabs is a test of *joint* distribution vs independence.
- **Treating cross_tabs as a directional / causal claim.** chi-squared is symmetric; it doesn't say "X causes Y" or even "X predicts Y". Causation is a research-design question, not a chi-squared question.
- **Forgetting the dict wrapper.** `cross_tabs(df, ...)` works but the summary shows `"Not provided"`. Use `cross_tabs({"<name>": df}, ...)`.
- **Reporting the chi-squared contribution table as if it were the cell-level effect size.** It's the *contribution to the omnibus statistic* — useful for diagnosing which row/column dominates the test, but not normalized to a "is this cell over-represented" interpretation. The dev_std table is the standardized cell-level effect-size view.
- **Confusing the 2×2 cross_tab and the corresponding 2-proportion test.** They give the same chi-squared and p-value, but compare_props's `lev` choice and CI for the proportion difference are not directly visible from cross_tabs. Use compare_props when you want the percentage-point difference and its CI.
