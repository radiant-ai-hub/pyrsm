---
name: pyrsm-cross-tabs
description: Build and interpret two-way contingency tables and chi-squared tests of independence in Python using the pyrsm library's `cross_tabs` class. Use this skill whenever a student or analyst wants to test whether two categorical variables are associated, build observed/expected/chi-squared/standardized-deviation tables, identify which cells drive an omnibus rejection, compute row/column/total percentages, validate the expected-cell-count assumption, or work with the pyrsm package for any contingency-table task — even if they don't explicitly say "pyrsm" or "cross_tabs". Triggers include phrases like "is X associated with Y", "chi-square test of independence", "cross-tab", "contingency table", "does income predict newspaper choice", "which cells contribute to the chi-squared", "standardized residuals", "expected vs observed frequencies", "row percentages vs column percentages", or any mention of analyzing the relationship between two categorical variables in a marketing/business analytics class.
---

# pyrsm cross-tabs workflow

This skill walks the user through a chi-squared test of independence on a two-way contingency table using the `pyrsm.basics.cross_tabs` class — from loading their data to translating the omnibus test into a plain-English statement of association *and* identifying which cells drive the result via standardized deviations. It is designed for students learning chi-squared tests in a business/marketing analytics context.

For deep reference on the `cross_tabs` API, the seven output tables and what each one is for, the expected-count assumption, standardized residuals as the cell-level effect size, and the relationship to `compare_props`, see `references/cross-tabs.md`.

## Output policy

Two rules apply to **every** pyrsm call you make in this skill:

1. **Show the full output verbatim.** When you call `.summary()`, `.predict()`, or any pyrsm method that prints results, paste the **complete printed output** into your response — every line of the coefficient table, every statistic, every row. Do NOT abbreviate. Do NOT replace it with a high-level "main takeaways" bullet list. Do NOT skip the table to "save space". The user is reading your response because they want to see the actual numbers, not your summary of them.

2. **Provide the full pedagogical interpretation, not lame bullets.** When walking through results, use the detailed templates in `references/<name>.md`: state hypotheses explicitly, walk every coefficient or cell with its real-world units from the sidecar description, give the significance verdict at the 5% level, use the "holding constant" qualifier (where applicable), and tie back to the business question. A response like "X is positive and significant; Y is positive and significant; Z is not" is exactly the failure mode this skill is built to prevent.

3. **Lead with the requested output.** When the user asks for a specific plot, table, or diagnostic (e.g., "create and interpret PDP", "show me the residual dashboard", "generate PIP", "give me the confusion matrix", "plot the gains chart"), your interpretation must focus on THAT output — not on a re-interpretation of the model's coefficient table or summary. Pi's failure pattern is to say *"I'll generate X and then interpret the fitted results"* and then quietly produce a coefficient walk-through. The user already saw the coefficient interpretation in the previous turn; do not repeat it unless they explicitly ask. Lead with the requested output's content, using the dedicated template for that output if the skill has one (Step 9 for PIP, Step 10 for PDP, etc.).

If your context is genuinely constrained and you cannot reproduce the full output, say so explicitly — never silently abbreviate.


---

## Step 1 — Ask for the data location

The very first thing to do (before importing anything, before writing any code) is ask the user where their data lives. Require an **absolute path**.

Example phrasing:

> Before we start, can you give me the **absolute path** to the data file you want to analyze? For example: `/Users/yourname/Downloads/newspaper.parquet`. Any of `.parquet`, `.csv`, `.tsv`, `.feather`, `.arrow`, `.xlsx`, or `.json` will work.

If the user gives you a relative path, a `~`-style path, or just a filename, ask again for an absolute path rather than guessing.

## Step 2 — Load the data into a polars DataFrame

Use `scripts/load_data.py` to do the loading:

```bash
python scripts/load_data.py "<absolute-path-to-data>"
```

The script prints a small JSON report describing the loaded dataset and the path of any sidecar `.md` it found.

### Sidecar description file

If a sidecar is found, **read it before proposing which variables to cross-tabulate**. The description tells you which columns are categorical, what the levels mean, and what associations the user might substantively expect. For the newspaper example, the sidecar tells you `Income` is High/Low (median split) and `Newspaper` is USA Today / WS Journal — useful context for interpreting the direction of any association.

If no sidecar is found, say so and ask the user to briefly describe the variables.

## Step 3 — Decide together what to test

Once the data and (optionally) the description are in hand, propose a test specification and confirm it with the user. You need two things:

- **First variable (`var1`)** — categorical column. Will be the rows of the contingency table.
- **Second variable (`var2`)** — categorical column. Will be the columns of the contingency table.

Both must be categorical (string, Categorical, or Enum). The chi-squared test of independence is symmetric in `var1` and `var2`: it tests "are these two variables associated?" regardless of which you think of as the predictor.

If the description file or the user's question makes the choice obvious (e.g., "are income and newspaper choice associated?"), state your proposed specification — `var1="Income"`, `var2="Newspaper"` — and ask the user to confirm.

Do a quick sanity check before running:

- How many levels in each? An R×C contingency table has `(R-1)*(C-1)` degrees of freedom. A 2×2 table has 1 df; a 3×4 table has 6 df.
- Are any levels very rare? `df[var1].value_counts()` and `df[var2].value_counts()` quickly show the marginals. If any level has < ~10 observations, you'll need to check the expected-cell-count assumption carefully.
- Do you actually want a *test of independence*, or a *test against a specified distribution*? The latter is `goodness`, not `cross_tabs`. If `var2` levels have substantively expected probabilities (census, theory, etc.), use `goodness` per variable.

## Step 4 — Run the test

Write a short, runnable Python script (or run it interactively if you have a Python REPL available):

```python
import polars as pl
import pyrsm as rsm

# 1. Load the data (absolute path)
data_path = "<absolute-path>"
df = pl.read_parquet(data_path)   # adjust reader for the actual file type

# 2. Build the cross-tab
ct = rsm.basics.cross_tabs(
    {"<dataset_name>": df},        # dict so the dataset name shows up in the summary
    var1="<row-variable>",
    var2="<column-variable>",
)

# 3. Inspect
ct.summary(output=["observed", "expected", "chisq", "dev_std"])
```

Notes:
- Pass the data as `{"name": df}` (a dict with one key) rather than just `df`, so the summary header prints a meaningful dataset name instead of `"Not provided"`.
- `pyrsm` accepts polars DataFrames directly.
- The `output` argument is a **list** controlling which tables print. The available tables are: `"observed"`, `"expected"`, `"chisq"`, `"dev_std"`, `"perc_row"`, `"perc_col"`, `"perc"`. For an initial fit, **pass all four diagnostic tables** (`observed`, `expected`, `chisq`, `dev_std`) — they are small and each teaches the student something different.
- The chi-squared statistic, df, p-value, and the percentage-of-cells-with-expected-below-5 print in the footer regardless of `output`.

## Step 5 — Interpret the output

Walk the user through the summary in roughly this order — see `references/cross-tabs.md` for detailed templates:

1. **State the hypotheses in words.** From the header:
   - H₀: there is no association between `<var1>` and `<var2>` (they are independent).
   - Hₐ: there is an association between `<var1>` and `<var2>`.
   Restate in the user's vocabulary using the sidecar units.

2. **Walk the tables, in order:**
   - **Observed** — actual counts, with marginal totals (row, column, grand).
   - **Expected** — under H₀, each cell = `(row_total × col_total) / grand_total`. Compare visually to observed.
   - **Chi-squared contributions** — `(observed - expected)² / expected` per cell. Their sum is the test statistic.
   - **Standardized deviations** — `(observed - expected) / sqrt(expected)`. Cells with `|dev_std| > 1.96` are individually deviating significantly from independence (rough rule of thumb).

3. **Check the expected-cell-count assumption.** The footer reports `"X% of cells have expected values below 5"`. If any cells are below 5, **say so explicitly and explain that the chi-squared p-value may be biased**. The 0%-below-5 case is the green light.

4. **Read the test result.** Chi-squared statistic on `(R-1)*(C-1)` df, p-value. State conclusion:
   - "Chi-squared = `<chi2>` on `<df>` df, p-value `<p>`. Because p `<is | is not>` smaller than 0.05, we `<reject | fail to reject>` H₀. The data `<provide | do not provide>` statistically significant evidence of an association between `<var1>` and `<var2>`."

5. **If you reject, identify which cells drive it.** Use the `dev_std` table. Cells with `|dev_std| > 1.96` are individually over- or under-represented relative to independence. State direction:
   - Positive `dev_std` → observed > expected for that cell (this combination is *more* common than independence would predict).
   - Negative `dev_std` → observed < expected (this combination is *less* common).
   This is **the** unique value-add of `cross_tabs` over a single omnibus p-value.

6. **Translate to a business conclusion.** Don't stop at "we reject H₀". For the newspaper example: "Income and newspaper preference are strongly associated. High-income readers are over-represented as WSJ readers (dev_std ≈ +8) and under-represented as USA Today readers (dev_std ≈ -7); the low-income pattern is the mirror. Reach the high-income segment via WSJ; reach the low-income segment via USA Today."

### Don't stop at the omnibus p-value — and don't read percentages without n

The classic cross-tabs failure modes:

- **Reporting "p < .001, so they're associated" and stopping.** The omnibus chi-squared says "the joint distribution is not the product of the marginals — somewhere". It does **not** tell you *which* cells drive the association, or by how much. The standardized-deviation table is what gives you the cell-level story. Always look at it.
- **Reading row/column percentages without sample sizes.** A column showing "this group is 80% high-income" is meaningless if the column only has 5 people in it. Always report `n` alongside any percentage.
- **Confusing row-conditional and column-conditional percentages.** `perc_row` answers "among 1st-class passengers, what fraction survived?" `perc_col` answers "among survivors, what fraction were 1st class?" Same data, very different questions. Pick the one that matches the research question.
- **Ignoring the expected-cell-count assumption.** With small samples or sparse contingency tables (many low-count cells), the chi-squared p-value is biased. Always read the `expected_low` line in the footer.

So: **the omnibus chi-squared is the first answer, not the only one.** The standardized residuals are where the substantive story is.

## Step 6 — The seven output tables — what each is for

`cross_tabs` computes seven distinct tables; pass them as a list to `output=`:

| Table | Formula | Purpose |
| --- | --- | --- |
| `"observed"` | Actual counts | Ground truth; always include. |
| `"expected"` | `(row × col) / total` per cell | What we'd see under independence; visual comparison to observed. |
| `"chisq"` | `(o - e)² / e` per cell | How each cell contributes to the test statistic. |
| `"dev_std"` | `(o - e) / √e` per cell | Cell-level standardized residuals; the effect-size view. |
| `"perc_row"` | Row-conditional percentages | "Given `var1=A`, what fraction is in each `var2` level?" |
| `"perc_col"` | Column-conditional percentages | "Given `var2=X`, what fraction is in each `var1` level?" |
| `"perc"` | Each cell / grand total | "What fraction of the entire sample is in this cell?" |

**For diagnostic walk-through, pass `["observed", "expected", "chisq", "dev_std"]`.** For descriptive reporting, pass `["observed", "perc_row"]` or `["observed", "perc_col"]` depending on which conditional makes more substantive sense.

## Step 7 — The expected-cell-count assumption (the critical concept)

The chi-squared test is asymptotic — it relies on the test statistic converging to a chi-squared distribution under H₀. That convergence requires large enough expected counts in every cell.

### The rule of thumb

> Every expected cell count should be at least 5.

`cross_tabs` reports this automatically in the footer: `"<X>% of cells have expected values below 5"`. The `expected_low` attribute is the underlying `[count_below_5, total_cells]`.

### When violated

- **Collapse rows or columns** to merge low-count cells with neighbors. Justified for ordinal categories (rating scales, age brackets); requires judgment for nominal categories.
- **Fisher's exact test** for 2×2 tables. Not exposed by pyrsm directly; compute with `scipy.stats.fisher_exact(observed_matrix)`.
- **For R×C with R or C > 2 and small counts**, exact alternatives exist (`scipy.stats.chi2_contingency` has a `lambda_` argument for likelihood-ratio variants, but exact tests are computationally heavier). For class work, collapse.
- **Report with caveats** if you must use the biased p-value. Say "X% of cells had expected counts below 5, so the chi-squared approximation may be inaccurate."

### Why pyrsm reports this automatically

Because the assumption is silently violated more often than students realize. Pyrsm prints the percentage so you have one less thing to compute by hand. For the newspaper example, `expected_low = [0, 4]` → 0 of 4 cells are below 5 → assumption clearly satisfied.

## Step 8 — Standardized deviations and the 1.96 rule

The `dev_std` table contains, for each cell:

```
dev_std = (observed - expected) / sqrt(expected)
```

Under H₀ (independence), these are approximately standard normal in each cell:

- `|dev_std| > 1.96` → cell individually significant at α = 0.05 (rough rule of thumb).
- `|dev_std| > 2.58` → at α = 0.01.
- `|dev_std| > 3.29` → at α = 0.001.

### Why this is the right cell-level effect-size view

The omnibus chi-squared statistic answers "the joint distribution is not the product of the marginals — somewhere". It does not tell you *where*. The standardized-deviation view does both:

- **Direction**: positive (over-represented) vs negative (under-represented).
- **Magnitude**: comparable across cells (unlike raw `(o - e)` which scales with cell size).

For the newspaper example, all four cells of the 2×2 have `|dev_std| > 5` — the association is driven by all cells together (which is what you'd expect in a 2×2 since the four cells are arithmetically tied to each other). In a 3×3 or larger table, dev_std typically highlights a subset of cells as the "driving" cells.

### Multiple-comparisons caveat

If you have R*C cells and check every one for `|dev_std| > 1.96`, you have a family-wise error problem. For exploratory cell-level diagnosis this is acceptable, but for *claims* of cell-level significance, apply a Bonferroni-style correction: divide α by `R*C`. With 4 cells, that's α/4 = 0.0125 → threshold of |dev_std| ≈ 2.50.

## Step 9 — Row vs column vs total percentages

The three percentage tables answer different questions:

- **`perc_row`** — conditioned on `var1`. Each row sums to 1 (or 100%). Answers: "given `var1=A`, what's the distribution of `var2`?" e.g., "of low-income readers, what fraction read WSJ?"
- **`perc_col`** — conditioned on `var2`. Each column sums to 1. Answers: "given `var2=X`, what's the distribution of `var1`?" e.g., "of WSJ readers, what fraction are low-income?"
- **`perc`** — fraction of the entire sample. Each cell is a small fraction; the table sums to 1.

Always state which conditional you're using. "80% of WSJ readers are high-income" (perc_col) is a different fact from "80% of high-income readers read WSJ" (perc_row).

For the newspaper example, the more substantive question is usually `perc_row` ("what does each income segment read?") — but the right answer depends on the user's question.

## Step 10 — Plot (offer, don't force)

After the basic interpretation, **offer** to run the plots:

```python
ct.plot(plots="perc_col")                                    # default; column percentages as bars
ct.plot(plots=["observed", "expected", "dev_std"])           # composed grid: useful diagnostic
ct.plot(plots="dev_std")                                     # standalone deviations with 1.96 reference
```

| `plots=` | Shows |
| --- | --- |
| `"observed"` / `"expected"` | Stacked bar charts of counts. |
| `"chisq"` | Grouped bars of chi-squared contributions per cell. |
| `"dev_std"` | Grouped bars of standardized deviations with horizontal reference lines at ±1.96, ±2.58, ±3.29. **The single most informative plot.** |
| `"perc_row"` / `"perc_col"` / `"perc"` | Grouped bars of conditional / total percentages. |

For class work, the `["observed", "expected", "dev_std"]` 3-panel composition is usually the best initial visualization.

## Step 11 — Probability-calculator companion (optional)

For the critical chi-squared value:

```python
df_val = (R - 1) * (C - 1)
pc = rsm.basics.prob_calc("chisq", df=df_val, pub=0.95)
pc.summary()
# For df=1, critical chi-sq = 3.841
# For df=6, critical chi-sq = 12.592
```

For the p-value from a specific chi-squared:

```python
pc = rsm.basics.prob_calc("chisq", df=df_val, ub=<observed_chi2>)
pc.summary()
```

If the user wants more on `prob_calc`, route them to the `pyrsm-prob-calc` skill.

## Step 12 — Related tests (when to switch)

`cross_tabs` is the right tool when you have two categorical variables and want an omnibus test of association. If the situation is different, suggest:

- One categorical variable, testing against a hypothesized distribution → `goodness`.
- Two categorical variables but you specifically want pairwise comparisons of one level across groups → `compare_props` (with `var1` as the grouping and `var2` as the response, picking `lev`).
- Two categorical variables, multi-level, with low cell counts → Fisher's exact via `scipy.stats.fisher_exact` (2×2) or careful collapsing.
- Modeling how a binary outcome depends on multiple covariates → `logistic` (`pyrsm-logistic` skill).

Don't switch silently — explain why the alternative is a better fit and confirm with the user before changing tools.

---

## Style notes

- **Pedagogical tone.** Always state H₀/Hₐ explicitly. Always tie significance back to a 5% threshold. Always interpret cells via dev_std, not just the omnibus chi-squared.
- **Use the units from the description file.** "Low-income readers are over-represented in USA Today" is much weaker than "276 of 359 low-income respondents read USA Today (77%), versus an expected 196 under independence — a substantial over-representation".
- **Always look at dev_std after rejecting.** The omnibus says "something's off"; dev_std says "here's where". This is the unique value of cross_tabs.
- **Always state which conditional percentage you mean.** Row vs column vs total percentages are different facts; mixing them up is a classic student mistake.
- **Check the expected-cell-count assumption.** Pyrsm prints it for you — pay attention.
- **Don't over-engineer.** Plain top-level code that the student can copy line by line.
- **One step at a time.** After Step 4 wait for the user to look at the tables before barreling into plots.

## Growing this skill

This skill is one of a family of `pyrsm-basics-*` skills covering the `pyrsm.basics` module. When `cross_tabs` isn't the right tool, point the user to the sibling skill named for the analysis they need (`pyrsm-goodness`, `pyrsm-compare-props`, `pyrsm-prob-calc`, etc.). The workflow shape — ask for absolute path → load via `scripts/load_data.py` → read sidecar → propose spec → run → interpret — is shared across the family.
