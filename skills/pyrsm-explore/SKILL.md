---
name: pyrsm-explore
description: Compute summary statistics for numeric columns in Python using the pyrsm library's `explore` function. Use this skill whenever a student or analyst wants a quick numeric summary (mean, median, min, max, sd, count, missing, n_unique) across one or many columns of a polars DataFrame, optionally grouped by a categorical column, with categorical predictors auto-expanded to dummies. The output is itself a polars DataFrame that the user can further chain with `.filter`, `.sort`, `.with_columns` — extensibility is a first-class part of this skill. Triggers include phrases like "describe these variables", "summary statistics", "group means by category", "what's the mean and median of price by cut", "give me a quick describe", "explore this dataset", "show me missingness per column", "n_unique per variable", "count rows per group with mean and sd", or any mention of wanting per-column numeric summaries in a marketing/business analytics context.
---

# pyrsm explore workflow

This skill walks the user through computing per-column summary statistics on a polars DataFrame using `pyrsm.eda.explore` — from loading their data to producing a tidy stats table they can read or chain further polars operations onto. It is designed for students learning exploratory data analysis in a business/marketing analytics context, so default to clear, pedagogical explanations and remind them that the returned table is itself a polars DataFrame for downstream work.

For deep reference on the `explore` API, the full list of supported aggregation functions, the `to_dummies` behavior, the two `header` layouts, and worked examples, see `references/explore.md`.

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

> Before we start, can you give me the **absolute path** to the data file you want to analyze? For example: `/Users/yourname/Downloads/diamonds.parquet`. Any of `.parquet`, `.csv`, `.tsv`, `.feather`, `.arrow`, `.xlsx`, or `.json` will work.

If the user gives you a relative path, a `~`-style path, or just a filename, ask again for an absolute path rather than guessing.

## Step 2 — Load the data into a polars DataFrame

Use `scripts/load_data.py` to do the loading:

```bash
python scripts/load_data.py "<absolute-path-to-data>"
```

The script prints a small JSON report describing the loaded dataset (path, shape, column names with dtypes) and the path of any sidecar `.md` it found. The dtype list is especially useful here — `explore` auto-selects numeric columns and (by default) expands categorical columns to dummies, so knowing which columns are which informs Step 3.

### Sidecar description file

If a sidecar is found, **read it before deciding which columns to summarize**. The description tells you what each column means and what units it's in — essential for translating raw means and medians into meaningful sentences.

If no sidecar is found, say so and ask the user to briefly describe the variables.

## Step 3 — Decide what to summarize

Once the data is in hand, propose a summary specification and confirm with the user. You need:

- **Columns (`cols`)** — list of column names. `None` (default) selects all numeric columns + dummy-encoded categoricals; pass an explicit list to focus.
- **Aggregation functions (`agg`)** — list of stat names. Default is `["mean", "median", "min", "max", "sd"]`. Supported: `mean`, `median`, `sum`, `std` (alias `sd`), `var`, `min`, `max`, `count` (alias `n`), `n_unique`, `n_missing` (alias `null_count`).
- **Grouping variable (`by`)** — optional column to group by. Output is one row per group when `by` is set.
- **Dummy expansion (`to_dummies`)** — `True` by default; categorical/Enum/String columns are dummy-encoded with `drop_first=True` and included in the summary. Set `False` to skip dummy expansion (then categoricals are excluded entirely).
- **Layout (`header`)** — `"function"` (default; variables as rows, statistics as columns) or `"variable"` (statistics as rows, variables as columns; transposes the table).

Confirm with the user, e.g.: "I'll summarize `price` and `carat` with mean, median, and sd, grouped by `cut`. Should I include any other variables?"

Do a quick sanity check before running:

- If the user wants per-group counts and missingness, include `count` and `n_unique` and `n_missing` in `agg`.
- If a "numeric" column is actually a 0/1 indicator or an ordinal code, the mean/sd may be misleading. Flag it.
- `by` must reference an existing column. With many groups (e.g., a high-cardinality ID), the result table can have thousands of rows — fine, but the user may want to filter further (see Step 6).

## Step 4 — Run explore

Write a short, runnable Python script (or run it interactively if you have a Python REPL available):

```python
import polars as pl
import pyrsm as rsm

# 1. Load the data (absolute path)
data_path = "<absolute-path>"
df = pl.read_parquet(data_path)   # adjust reader for the actual file type

# 2. Summarize
stats = rsm.eda.explore(
    df,
    cols=["<x1>", "<x2>"],         # or None for all numeric + dummies
    agg=["mean", "median", "sd"],  # or default
    by="<group>",                  # optional
    to_dummies=True,               # default; set False to skip categorical expansion
    header="function",             # or "variable"
)

# 3. Inspect (it's a polars DataFrame)
print(stats)
```

Notes:
- `explore` accepts polars DataFrames (and LazyFrames — it will collect internally). No dict wrapper is needed (this is a function, not a class).
- The output column names follow the pattern `{col}_{agg}` when `by` is set (e.g., `price_mean`) and just `{agg}` when there is no `by` (e.g., `mean`).
- For `header="variable"`, the first column is `statistic` and the remaining columns are the variables; rows are the aggregation functions.

## Step 5 — Interpret the output

Walk the user through the table in roughly this order — see `references/explore.md` for detailed templates:

1. **Describe the shape.** One sentence: "The output is `<R>×<C>`: one row per `<variable | group>`, one column per `<aggregation | variable>`, depending on `header=`."

2. **Read the central tendencies.** Mean vs median per variable:
   - **Mean ≈ median** → roughly symmetric distribution.
   - **Mean ≫ median** → right-skewed (long upper tail — common for prices, sales, incomes).
   - **Mean ≪ median** → left-skewed.
   - For business analytics work, a 30-50% gap between mean and median is a strong sign you should look at the distribution before modeling with a linear assumption — the `pyrsm-distr` skill or `pyrsm-visualize` can show you the histogram.

3. **Read the spread.** sd and the min/max range together. A sd that is large relative to the mean (a high coefficient of variation `sd/mean`) signals high variability; a sd near zero with `n_unique=2` is probably an effectively-binary indicator.

4. **Read counts and missingness if included.** `count` reports non-null observations, `n_missing` reports nulls, `n_unique` reports cardinality. Use `n_unique` to spot numeric columns that should really be treated as categorical (e.g., `n_unique = 5` on a "rating" column).

5. **Use the units from the sidecar.** Generic "price has mean 3907" is much weaker than "the mean diamond price is $3,907 (median $2,407), with a sd of $3,957 — a right-skewed distribution typical of prices."

### Don't read mean alone — and watch for misleading dummies

The classic explore failure modes:

- **Reporting mean without median for skewed variables.** Prices, sales, incomes, and counts are typically right-skewed; the mean is pulled by the upper tail and is *not* the typical observation. Always report both, and lead with the median for skewed variables.
- **Reading the mean of an auto-generated dummy as if it's a regular variable.** With `to_dummies=True`, a categorical column `cut` with 5 levels expands into 4 dummies (drop_first=True). The "mean" of `cut_Ideal` is the *proportion* of rows that are Ideal cut — interpret it as a proportion, not as an average. The `sd` similarly maps to `sqrt(p(1-p))`. If this is confusing, set `to_dummies=False` and summarize the numeric columns separately, then use `pyrsm-pivot` for categorical breakdowns.
- **Comparing groups via `by=` without comparing the n's.** A group with n=10 and another with n=1000 will have very different uncertainty around the same point estimate. Always include `count` in `agg` when using `by`.
- **Missing values silently dropped.** `mean` skips nulls. Always add `n_missing` to `agg` when missingness is a possible concern.

So: **when interpreting, do not lean on the mean alone**, always include the count when grouping, and watch for dummies masquerading as regular variables.

## Step 6 — Extend with polars

`rsm.eda.explore` returns a `pl.DataFrame`. **The output is data, and you can keep working on it.** This is the most common downstream pattern and a key piece of value in pyrsm.eda:

```python
stats = rsm.eda.explore(df, cols=["price", "carat"], by="cut", agg=["mean", "median", "count"])

# 1. Sort by a column
stats.sort("price_mean", descending=True)

# 2. Filter to groups with enough observations
stats.filter(pl.col("price_count") >= 100)

# 3. Add derived metrics (e.g., coefficient of variation)
stats.with_columns(
    cv=pl.col("price_sd") / pl.col("price_mean")
)

# 4. Stack two summaries
together = pl.concat([
    rsm.eda.explore(df, cols=["price"], by="cut", agg=["mean", "median"]).with_columns(pl.lit("cut").alias("group_var")),
    rsm.eda.explore(df, cols=["price"], by="color", agg=["mean", "median"]).rename({"color": "cut"}).with_columns(pl.lit("color").alias("group_var")),
])

# 5. Pivot the long form to wide for presentation
stats.pivot(values="price_mean", index="cut", on="color", aggregate_function="first")
```

When the user's task is "I want to see X and then do Y to it", structure your code as:
1. Run `explore` for X.
2. Chain polars operations for Y.
3. Print or pass to the next step.

The output of `explore` is the input to the next polars expression. Don't treat the printed table as the endpoint when the user's question implies follow-up work.

## Step 7 — Related EDA tools (when to switch)

`explore` is the right tool when you want a tidy table of numeric summaries. If the situation is different, suggest:

- **Single-column distribution shape (histogram, plot, mean+median in one go)** → `pyrsm-distr`.
- **Two-way cross-tabulation with aggregation** → `pyrsm-pivot`.
- **Wide-to-long reshaping** → `pyrsm-unpivot`.
- **Joining two DataFrames** → `pyrsm-combine`.
- **Custom plot of any single variable or relationship** → `pyrsm-visualize`.
- **Hypothesis test on a mean vs a benchmark** → `pyrsm-single-mean`.
- **Hypothesis test comparing group means** → `pyrsm-compare-means`.

Don't switch silently — explain why the alternative is a better fit and confirm with the user before changing tools.

---

## Style notes

- **Pedagogical tone.** Always interpret the stats back in business / domain terms; the output is a table of numbers, but the value is in the sentence the student writes next to it.
- **Use the units from the description file.** Mean price = $3,907, not "3907".
- **Always pair mean and median for skewed variables.** Skewness is the rule, not the exception, in business data.
- **Always include `count` when using `by`.** Group-size imbalance changes interpretation.
- **Always include `n_missing` or `null_count` if missingness might matter.** Quietly skipping nulls is a frequent source of bugs.
- **Watch for `to_dummies` artifacts.** "mean of `color_E`" is a proportion, not an average. If the user is confused, switch to `to_dummies=False` and use `pyrsm-pivot` for the categorical breakdown.
- **Extend with polars.** The returned table is data; the next polars expression is part of the workflow.
- **Don't over-engineer.** Plain top-level code that the student can copy line by line.
- **One step at a time.** After Step 4 wait for the user to look at the table before recommending follow-up transformations.

## Growing this skill

This skill is one of a family of `pyrsm-eda-*` skills covering the `pyrsm.eda` module. When `explore` isn't the right tool, point the user to the sibling skill named for the operation they need (`pyrsm-distr`, `pyrsm-pivot`, `pyrsm-unpivot`, `pyrsm-combine`, `pyrsm-visualize`). The workflow shape — ask for absolute path → load via `scripts/load_data.py` → read sidecar → propose spec → run → interpret → extend with polars — is shared across the family.
