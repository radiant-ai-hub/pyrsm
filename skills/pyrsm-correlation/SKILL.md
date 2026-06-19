---
name: pyrsm-correlation
description: Compute and interpret correlation matrices in Python using the pyrsm library's `correlation` class. Use this skill whenever a student or analyst wants to measure the strength and direction of pairwise relationships among numeric variables, choose between Pearson, Spearman, Kendall, or polychoric correlation, build a scatter-matrix visualization with significance stars, screen for multicollinearity in a dataset before fitting a regression, or work with the pyrsm package for any correlation task — even if they don't explicitly say "pyrsm" or "correlation". Triggers include phrases like "how correlated are X and Y", "compute a correlation matrix", "is there a relationship between these variables", "check for multicollinearity", "Pearson vs Spearman", "what's the rank correlation", "scatter matrix with regression lines", or any mention of measuring pairwise linear/monotonic association among continuous variables in a marketing/business analytics class.
---

# pyrsm correlation workflow

This skill walks the user through computing and interpreting a correlation matrix using the `pyrsm.basics.correlation` class — from loading their data to translating the matrix into plain-English statements about which variables move together, which method is appropriate, and what the scatter matrix reveals beyond a single correlation number. It is designed for students learning correlation in a business/marketing analytics context.

For deep reference on the `correlation` API, Pearson vs Spearman vs Kendall vs polychoric, the limits of `r` as a summary of relationship, the cutoff filter, and how to read the scatter-matrix plot, see `references/correlation.md`.

## Output policy

Two rules apply to **every** pyrsm call you make in this skill:

1. **Show the full output verbatim.** When you call `.summary()`, `.predict()`, or any pyrsm method that prints results, paste the **complete printed output** into your response — every line of the coefficient table, every statistic, every row. Do NOT abbreviate. Do NOT replace it with a high-level "main takeaways" bullet list. Do NOT skip the table to "save space". The user is reading your response because they want to see the actual numbers, not your summary of them.

2. **Provide the full pedagogical interpretation, not lame bullets.** When walking through results, use the detailed templates in `references/<name>.md`: state hypotheses explicitly, walk every coefficient or cell with its real-world units from the sidecar description, give the significance verdict at the 5% level, use the "holding constant" qualifier (where applicable), and tie back to the business question. A response like "X is positive and significant; Y is positive and significant; Z is not" is exactly the failure mode this skill is built to prevent.

3. **Lead with the requested output.** When the user asks for a specific plot, table, or diagnostic (e.g., "create and interpret PDP", "show me the residual dashboard", "generate PIP", "give me the confusion matrix", "plot the gains chart"), your interpretation must focus on THAT output — not on a re-interpretation of the model's coefficient table or summary. Pi's failure pattern is to say *"I'll generate X and then interpret the fitted results"* and then quietly produce a coefficient walk-through. The user already saw the coefficient interpretation in the previous turn; do not repeat it unless they explicitly ask. Lead with the requested output's content, using the dedicated template for that output if the skill has one (Step 9 for PIP, Step 10 for PDP, etc.).

If your context is genuinely constrained and you cannot reproduce the full output, say so explicitly — never silently abbreviate.


---

## Step 1 — Ask for the data location

The very first thing to do (before importing anything, before writing any code) is ask the user where their data lives. Require an **absolute path**, because relative paths are a frequent source of "file not found" frustration for students who are not yet comfortable with the shell.

Example phrasing:

> Before we start, can you give me the **absolute path** to the data file you want to analyze? For example: `/Users/yourname/Downloads/salary.parquet`. Any of `.parquet`, `.csv`, `.tsv`, `.feather`, `.arrow`, `.xlsx`, or `.json` will work.

If the user gives you a relative path, a `~`-style path, or just a filename, ask again for an absolute path rather than guessing. If the path doesn't exist, say so plainly and ask them to double-check.

## Step 2 — Load the data into a polars DataFrame

Use `scripts/load_data.py` to do the loading. It handles file-type detection by extension, reads the file with the appropriate polars reader, and also looks for a sidecar markdown description file in the same folder.

Run it like this:

```bash
python scripts/load_data.py "<absolute-path-to-data>"
```

The script prints a small JSON report describing the loaded dataset (path, shape, column names with dtypes) and the path of any sidecar `.md` it found. Use that to identify which columns are numeric and likely candidates for the correlation analysis.

### Sidecar description file

In pyrsm-style course materials, every dataset is shipped alongside a markdown file that documents what the variables mean. The convention is `<dataset>_description.md` next to `<dataset>.parquet` (e.g., `salary.parquet` ↔ `salary_description.md`). The loader script searches for this and several similar patterns automatically.

If a sidecar is found, **read it before proposing which variables to correlate**. The description tells you which numeric columns are continuous measurements, which are coded categoricals (e.g., 0/1 dummies you probably shouldn't put in a Pearson matrix), what the units are, and how variables relate substantively. This is the difference between a generic "yrs_since_phd is highly correlated with yrs_service" and a useful "years since PhD and years of service are nearly collinear (r=0.91) — putting both in a regression model is a multicollinearity problem".

If no sidecar is found, say so and ask the user to briefly describe the variables. Don't invent meanings.

## Step 3 — Decide together what to correlate

Once the data and (optionally) the description are in hand, propose a correlation specification and confirm it with the user before running. You need three things:

- **Variables (`vars`)** — a list of column names. **All must be numeric.** Passing `vars=[]` (the default) auto-selects every numeric column in the DataFrame.
- **Method (`method`)** — `"pearson"` (default, linear association), `"spearman"` (rank-based, monotonic association, robust to outliers), `"kendall"` (rank-based, more robust on small samples), or `"polychoric"` (latent-normal correlation for ordinal pairs). See Step 6.
- **Cutoff for display** — a non-zero `cutoff` in `summary()` hides correlations below the threshold (absolute value), useful for skimming a large matrix. The default 0 shows everything.

If the description file or the user's question makes the choice obvious (e.g., "are years and salary correlated?"), state your proposed specification — `vars=["salary", "yrs_since_phd", "yrs_service"]`, `method="pearson"` — and ask the user to confirm.

Do a quick sanity check before running:

- Are the selected variables actually numeric? Pearson, Spearman, and Kendall all require numeric input; the class will skip non-numeric columns if you let it auto-select, but raise / behave oddly if you pass them explicitly.
- Are any of the "numeric" columns really 0/1 dummies, ordinals coded as ints, or counts with heavy ties? Pearson on these is interpretable but limited; consider Spearman or polychoric (see Step 6).
- How big is the matrix? A correlation matrix of k variables has k×(k−1)/2 unique pairs. With 20 variables that's 190 pairs — way too many to read individually. For large matrices, use a `cutoff` (e.g., `summary(cutoff=0.3)`) and focus on the cells that survive.

## Step 4 — Compute the correlation matrix

Write a short, runnable Python script (or run it interactively if you have a Python REPL available). Keep it minimal and readable — students will look at this code:

```python
import polars as pl
import pyrsm as rsm

# 1. Load the data (absolute path)
data_path = "<absolute-path>"
df = pl.read_parquet(data_path)   # adjust reader for the actual file type

# 2. Compute correlations
cr = rsm.basics.correlation(
    {"<dataset_name>": df},        # dict so the dataset name shows up in the summary
    vars=["<x1>", "<x2>", "<x3>"], # or [] to auto-select numeric columns
    method="pearson",              # or "spearman" / "kendall" / "polychoric"
)

# 3. Inspect
cr.summary()
```

Notes:
- Pass the data as `{"name": df}` (a dict with one key) rather than just `df`, so the summary header prints a meaningful dataset name instead of `"Not provided"`.
- `pyrsm` accepts polars DataFrames directly.
- `cr.cr`, `cr.cp`, `cr.cv` give the raw numpy matrices (correlations, p-values, covariances) if you need them programmatically. The summary prints lower-triangle views with significance stars.
- Add `summary(cov=True)` to also print the covariance matrix; rarely needed for class work but available.

## Step 5 — Interpret the output

This is where most of the value is. Walk the user through the matrix in roughly this order — see `references/correlation.md` for detailed templates and worked examples:

1. **State what the test reports.** One sentence: "We're measuring `<method>` correlation between each pair of `<list-vars>`. For each pair, H₀ is that the population correlation is zero; Hₐ is that it is non-zero."

2. **Walk the highest-magnitude correlations first.** Sort the off-diagonal entries by `|r|`. For each pair with `|r| > 0.3` (a rough rule of thumb for "noteworthy"):
   - Direction: "positively / negatively correlated".
   - Magnitude: small (0.1–0.3), moderate (0.3–0.5), large (0.5–0.7), very large (> 0.7).
   - Significance: from the p-value matrix (or the asterisks in the plot). With moderate-to-large n, almost any |r| > 0.1 will be "significant" — significance is necessary but not sufficient for "interesting".
   - Substantive read: tie back to the units. "Years since PhD and years of service are nearly perfectly correlated (r=0.91), which makes sense — long-tenured professors generally got their PhDs long ago."

3. **Flag multicollinearity warning signs.** Any pair with `|r| > 0.8` in a set of would-be predictors should be flagged as a problem for a downstream regression: keeping both inflates standard errors, can flip signs, and makes individual coefficients hard to interpret. (See `pyrsm-regress` skill for the VIF check.)

4. **Don't over-read significance with large n.** With n in the thousands, p-values of 0.000 are routine even for tiny correlations. Always pair the asterisks with the actual magnitude.

### Don't read r alone — and don't read significance off the p-value alone

The classic correlation failure modes:

- **r = 0 ≠ no relationship.** Pearson r measures *linear* association. A perfect quadratic relationship (y = x²) has r ≈ 0 over a symmetric range. A correlation of 0 in the printed matrix could mean (a) the variables are unrelated, (b) the relationship is non-linear, or (c) there is a sub-population where they're related but it cancels with another. **Always inspect the scatter plot before claiming "no relationship".**
- **Outliers dominate Pearson.** A single extreme observation can drag Pearson's r far from where the bulk of the data sit. Spearman (rank-based) is robust to outliers; if Pearson and Spearman disagree noticeably, that's a tell that outliers are driving the Pearson result.
- **Correlation is not causation.** Standard caveat — repeat it explicitly. r is symmetric: r(x, y) = r(y, x). Without a research design that addresses confounding, "X is correlated with Y" supports neither "X causes Y" nor "Y causes X".
- **Statistical significance is cheap with large n.** A correlation of r = 0.02 with n = 100,000 lands at p < .001 but explains 0.04% of the variance — almost certainly noise.
- **Matrix-level multiple testing.** With k variables you compute k×(k−1)/2 correlations. At α = 0.05 you'd expect 5% of zero-correlation pairs to look "significant" by chance. Don't claim individual significance on a single cell of a large matrix without thinking about this.

So: **always look at the scatter matrix (Step 7), not just the numbers.** Even when the matrix says r = 0, the plot can reveal a relationship that requires a different method or a transformation.

## Step 6 — Choosing the correlation method

`correlation` supports four methods. The right choice depends on the variable types and the relationship you want to detect.

### Pearson (default, linear)

- Measures *linear* association.
- Sensitive to outliers.
- Requires (approximately) interval-scaled variables.
- The default; right when both variables are continuous and the relationship is plausibly linear.

### Spearman (rank-based, monotonic)

- Computes Pearson on the *ranks* of the data.
- Captures *monotonic* (not just linear) associations.
- Robust to outliers and non-normal distributions.
- Right when:
  - Variables are skewed or have outliers (and you don't want to drop them).
  - One or both variables are ordinal (e.g., a 1–5 rating).
  - You want a relationship measure that survives monotonic transformations like log.

### Kendall (rank-based, more robust on small samples)

- Like Spearman, but based on concordant/discordant pairs.
- Slightly more conservative; preferred by some statisticians for small n.
- Computationally heavier than Spearman.

### Polychoric (latent-normal, for ordinal pairs)

- Estimates the correlation between *latent* normal variables underlying observed ordinal categories.
- Right when both variables are ordinal categoricals with relatively few levels (e.g., Likert scales).
- More principled than Pearson on ordinal data, more informative than Spearman when the underlying construct is plausibly continuous.
- P-values are not reported.

### Decision rule

> Continuous, plausibly linear, well-behaved → Pearson.
> Skewed, outliers, ordinal, or relationship may be monotonic but not linear → Spearman (or Kendall for small n).
> Both variables are ordinal Likert-scale with few levels → polychoric.

### When Pearson and Spearman disagree

If `r_pearson` and `r_spearman` are noticeably different (say, |Δ| > 0.1), something is going on:
- Outliers are pulling Pearson.
- The relationship is non-linear (monotonic but curved).
- The data have heavy ties (Spearman handles them, Pearson treats them as data).

In all of those cases, the scatter plot is essential. Don't pick one method and move on without looking.

## Step 7 — Plot (always recommended)

After the basic interpretation, **strongly suggest** running the scatter matrix:

```python
cr.plot()
```

The plot is a matrix of scatter plots:
- **Diagonal**: variable names.
- **Lower triangle**: scatter plot with a fitted regression line — the visual ground-truth for the relationship.
- **Upper triangle**: correlation coefficient, sized by magnitude, with significance stars.

Use the plot to catch the failure modes Step 5 warned about:
- **Non-linearity.** A scatter with a clear curve and a regression line that misses it — `r` is misleading.
- **Outliers.** A single point far from the cloud, with the regression line pulled toward it.
- **Heteroscedastic spread.** Cloud fans out — `r` summarizes the average relationship but misses the changing variance.
- **Sub-populations.** Two distinct clusters with different slopes inside the same scatter — `r` is an average over them.

Optional parameter: `nobs=` controls how many points are scattered (default 1000, set `-1` for all). The correlations themselves are always computed on all data; only the plotted points are sub-sampled to keep the chart legible for large datasets.

For matplotlib output the `plot()` method returns a `Figure` object; in Jupyter it displays inline. Tune `figsize` if the labels look squished.

## Step 8 — Cutoff filter (large matrices)

When the matrix has more than ~6 variables, the summary is hard to skim. Use the `cutoff` parameter to hide weak correlations:

```python
cr.summary(cutoff=0.3)
```

Cells with `|r| < 0.3` print as empty. The full matrix is still computed; this is purely a display filter.

Choose `cutoff` based on the question:
- 0.3 — "show me anything moderate or larger".
- 0.5 — "show me only the strong relationships".
- 0.8 — "show me only the multicollinearity warnings".

## Step 9 — Probability-calculator companion (optional)

For a single pair, you can validate the printed p-value by recomputing it:

```python
# For Pearson with df = n - 2:
pc = rsm.basics.prob_calc("tdist", df=<n-2>, ub=<observed_t>)
pc.summary()
```

where `observed_t = r * sqrt((n - 2) / (1 - r**2))`. The p-value (two-sided) should match the matrix. Mostly useful as a teaching tool to make the connection between r, t, and p concrete.

## Step 10 — Related tests (when to switch)

`correlation` is the right tool when you want pairwise association among numeric variables. If the situation is different, suggest:

- One predictor and one response, want to model the relationship (not just summarize) → `regress`.
- Two categorical variables, testing for association → `cross_tabs`.
- One continuous variable and one categorical with two or more levels → `compare_means`.
- One categorical's distribution vs an expected one → `goodness`.
- Multivariate dimensionality reduction (rather than pairwise) → tools in `pyrsm.multivariate` (PCA, factor analysis).

Don't switch silently — explain why the alternative is a better fit and confirm with the user before changing tools.

---

## Style notes

- **Pedagogical tone.** Always state what `r` measures (linear or monotonic association), what its range is (−1 to +1), and what magnitudes are conventionally called small/moderate/large. Always tie back to whether the relationship is plausibly linear before reporting Pearson.
- **Use the units and definitions from the description file.** "yrs_since_phd and yrs_service have r=0.91" is much weaker than "years since PhD and years of service in this faculty sample move almost perfectly together (r=0.91) — most professors got their PhDs and joined this college around the same time."
- **Always inspect the scatter plot.** The matrix is a summary; the plot is the data. r = 0 with a clear scatter pattern is a story; r = 0.9 with a tight linear scatter is a different story. Don't report just the numbers.
- **Don't over-interpret statistical significance.** With moderate-to-large n, almost any non-zero correlation is "significant". Magnitude carries the meaning.
- **Flag multicollinearity for downstream regression.** Pairs with `|r| > 0.8` among predictors are warnings.
- **Don't over-engineer.** Plain top-level code that the student can copy line by line.
- **One step at a time.** After Step 5 (matrix interpretation) wait before barreling into the plot or method-switch experiments.

## Growing this skill

This skill is one of a family of `pyrsm-basics-*` skills covering the `pyrsm.basics` module. When `correlation` isn't the right tool, point the user to the sibling skill named for the analysis they need (`pyrsm-regress`, `pyrsm-compare-means`, `pyrsm-cross-tabs`, `pyrsm-prob-calc`, etc.). The workflow shape — ask for absolute path → load via `scripts/load_data.py` → read sidecar → propose spec → run → interpret — is shared across the family.
