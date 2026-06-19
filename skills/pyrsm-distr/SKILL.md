---
name: pyrsm-distr
description: Run distribution analysis on a polars DataFrame in Python using the pyrsm library's `distr` class. Use this skill whenever a student or analyst wants a one-shot exploratory summary of every column in a dataset — numeric variables get mean/median/min/max/sd, categorical variables get counts and proportions per level, other (date/datetime) get min/max/n_unique/n_missing — paired with histograms (numeric) and bar plots (categorical). The class also auto-classifies low-cardinality integers as categorical, and returns plotnine ggplot objects users can layer additional aesthetics onto. Triggers include phrases like "describe this dataset", "show me the distributions", "plot all the variables", "categorical vs numeric split", "histograms for every numeric column", "what's in this DataFrame", "EDA dump", or any request for a holistic first-look at a dataset in a marketing/business analytics context.
---

# pyrsm distr workflow

This skill walks the user through a holistic distribution analysis of a polars DataFrame using the `pyrsm.eda.distr` class — from loading the data to producing per-column summary statistics, frequency tables, and distribution plots that the user can extend with plotnine layers and chain further polars operations onto.

For deep reference on the `distr` API, the column-type classification heuristic, the underlying `explore` integration, and worked plotnine-layering examples, see `references/distr.md`.

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

The script prints a JSON report. **The dtype list is especially useful here** — `distr` classifies columns automatically (numeric / categorical / other), and seeing the raw dtypes tells you whether the classification is going to behave as expected.

### Sidecar description file

If a sidecar is found, **read it before proposing the analysis**. The description tells you what each column means and what units it's in — essential context for the per-column interpretation.

## Step 3 — Decide what to analyze

Confirm the analysis specification with the user. You need:

- **Columns (`cols`)** — list of column names to include. `None` (default) selects all columns of `data` except `by`.
- **Group-by (`by`)** — optional column to stratify the analysis. With `by`, numeric stats are computed per group, and categorical tables show counts within each group.
- **Dataset name (`name`)** — printed in the summary header (default `"Not provided"`).
- **Integer-cardinality threshold (`nint`)** — integer columns with fewer than `nint` unique values are classified as **categorical**, not numeric. Default 25. This is the most important parameter to be aware of: it controls whether a column like "rating" (1–5, n_unique=5) is treated as a numeric (statistics: mean=3.2) or a categorical (frequency table: 5 says 1, 12 says 2, …).

Confirm: "I'll run a distribution analysis on `<columns>` of `<dataset>`. Integer columns with fewer than `<nint>` unique values will be treated as categorical."

Do a quick sanity check before running:

- Look at the input dtypes: are there any int columns that should be categorical (e.g., ordinal codes, 0/1 indicators) or vice versa (e.g., year-as-string)?
- For very wide datasets (50+ columns), the printed summary can be overwhelming. Either subset `cols=[...]` or be ready to walk only the highlights.
- If `by` is set, make sure it's a categorical column with a manageable number of groups (≤ ~10 for clean output).

## Step 4 — Run distr

```python
import polars as pl
import pyrsm as rsm

# 1. Load the data (absolute path)
data_path = "<absolute-path>"
df = pl.read_parquet(data_path)

# 2. Run distribution analysis
d = rsm.eda.distr(
    df,
    cols=None,                  # or a list
    by=None,                    # or a column name
    name="<dataset>",           # for display
    nint=25,                    # int-cardinality threshold for categorical classification
)

# 3. Inspect
d.summary()                     # prints numeric stats + per-categorical frequency tables
d.plot()                        # returns a plotnine composition (histograms + bar charts)
```

Notes:
- `distr` is a class. The constructor classifies columns; `summary()` and `plot()` are methods.
- `summary()` prints — it doesn't return.
- `plot()` returns a `plotnine.ggplot` (or composition thereof) — assign it to a variable to layer plotnine extensions, or display it directly.
- `d.data` (polars DataFrame), `d.numeric_cols`, `d.categorical_cols`, `d.other_cols` are attributes; you can query them programmatically.

## Step 5 — Interpret the summary output

Walk the user through the printed summary in roughly this order:

1. **Header.** Dataset name + columns classified by type. "11 columns: 7 numeric, 3 categorical, 1 other." This is the first sanity check — were the classifications what you expected?

2. **Numeric variables table.** One row per numeric column, with mean / median / min / max / sd. Lead with the gap between mean and median: a large gap signals skew. Note which variables have potential outliers (sd ≫ |mean| / 3) or implausible ranges.

3. **Categorical variables — one section per column.** For each: n_unique, mode, n_missing, and a frequency table with `count` and `proportion`. Look for:
   - **Levels with extreme imbalance** (one level has 80%+ of the rows).
   - **Tiny levels** (counts in the single digits — may need to be collapsed for downstream modeling).
   - **Unexpected levels** (typos, encoding issues).

4. **Other variables.** Date / datetime columns. Min and max give the date range. n_unique = 1 means all-same; very high n_unique relative to row count means each row has a unique timestamp.

5. **Translate to data-quality flags.** Before any modeling, the `distr` summary surfaces issues to fix: missingness, skew, sparse categories, outliers, weird ranges. Make those explicit.

### Don't treat the summary as the final analysis — interpret it as a diagnostic

The classic `distr` failure modes:

- **Treating mean as "typical" for skewed variables.** Compare mean and median for every numeric column. If the gap is > 30% of the sd, treat median as the typical value and flag the skew as a modeling consideration (often calls for a log transform).
- **Missing the `nint` classification trick.** A column called `rating` with values 1–5 is an *ordinal categorical* in substance, but if it's stored as `Int64` with n_unique < `nint`, `distr` will (correctly) classify it as categorical and show a frequency table. If `nint` is too low and you have an integer column like "number_of_children" (0–10, n_unique=11), it may be classified as categorical when continuous interpretation is more useful. Adjust `nint` to taste, or pre-cast the column.
- **Ignoring low-count levels.** A 12-level categorical with most rows in two levels needs to be either combined or dropped before being used as a model predictor — the small-level estimates are noisy.
- **Treating high-n_unique categoricals as informative.** An ID column will have n_unique ≈ n_rows and be useless as a predictor. The `distr` summary shows this clearly.

## Step 6 — Extend the plots with plotnine

This is the **critical extensibility concept** for `distr`. `d.plot()` returns a `plotnine.ggplot` (or a composition of them). You can add any plotnine layer to customize:

```python
from plotnine import (
    aes, geom_density, geom_smooth, geom_vline,
    labs, scale_x_log10, theme_minimal, theme,
)

# Default
p = d.plot()

# Customize: log-scale x-axis on the histograms, add overall title
p_custom = p + scale_x_log10() + labs(title="Diamonds: distribution of all variables")

# Different theme
p_theme = p + theme_minimal()

# Add a vertical reference line at the mean (single column case)
p_single = d.plot(cols=["price"])
p_with_line = (
    p_single
    + geom_vline(xintercept=df["price"].mean(), color="red", linetype="dashed")
    + labs(title="Diamond Prices with Mean Reference")
)

# Suppress the auto-layout grid by plotting columns one-at-a-time
for col in d.numeric_cols:
    single_plot = d.plot(cols=[col])
    print(single_plot)   # render one chart at a time
```

The point: `distr.plot()` is the *starting point* for the visualization, not the endpoint. Anything plotnine supports — scales, themes, additional geoms, faceting, custom aesthetics — can be added afterward.

### When to use distr.plot() vs pyrsm-visualize

- `distr.plot()` — auto-pick of histogram vs bar based on column type, multi-column grid, default theme. Best for the **first look**.
- `pyrsm-visualize` — explicit choice of geom (scatter / line / box / violin / density / dist / bar), full control over color / fill / shape / facet / position. Best for **specific plots** when you know what you want.

If the user has a specific visualization in mind, route them to `pyrsm-visualize`. If they want a holistic overview, `distr.plot()` is the right tool.

## Step 7 — Extend with polars on `d.data`

The `distr` object stores the underlying DataFrame as `d.data`, and the column classifications as `d.numeric_cols` / `d.categorical_cols` / `d.other_cols`. You can use these to drive downstream polars operations:

```python
# Filter to non-missing rows on a specific column
clean = d.data.filter(pl.col("age").is_not_null())

# Compute correlations only across the numeric columns
import pyrsm as rsm
cr = rsm.basics.correlation(d.data, vars=d.numeric_cols)
cr.summary()

# Build a regression spec using d's classifications
# (numerics as covariates, the response variable separately)
predictors = [c for c in d.numeric_cols if c != "price"]
reg = rsm.model.regress({"diamonds": d.data}, rvar="price", evar=predictors)
reg.summary()

# Re-run distr after a transform
transformed = d.data.with_columns(price_log=pl.col("price").log())
rsm.eda.distr(transformed, cols=["price", "price_log"]).plot()
```

The `distr` object's classification is itself a useful piece of metadata that you can use downstream.

## Step 8 — Related EDA tools (when to switch)

`distr` is the right tool when you want a **holistic first-look** at every column. If the situation is different, suggest:

- **Just the numeric stats, no plots, custom aggregations** → `pyrsm-explore`.
- **Cross-tabulation of two variables** → `pyrsm-pivot` (descriptive) or `pyrsm-cross-tabs` (with chi-squared).
- **Custom plotting of a specific relationship** → `pyrsm-visualize`.
- **Joining two DataFrames** → `pyrsm-combine`.
- **Reshape wide ↔ long** → `pyrsm-pivot` / `pyrsm-unpivot`.

Don't switch silently — explain why the alternative is a better fit and confirm with the user before changing tools.

---

## Style notes

- **Pedagogical tone.** The summary is *diagnostic*, not conclusive. Always tie back to "what should we do about this finding".
- **Use the units from the description file.** "price mean 3907" is much weaker than "average diamond price is $3,907 (median $2,407 — strongly right-skewed)".
- **Always check mean vs median for numeric columns.** Skewed business variables (prices, sales, incomes, counts) are the norm. Flag the skew.
- **Always inspect the categorical levels.** Tiny levels, encoding errors, and high-cardinality ID columns surface in the categorical section.
- **Extend the plot with plotnine.** `d.plot()` is the starting point. Add scales, themes, reference lines as needed.
- **Use `d.data` and `d.numeric_cols` / `d.categorical_cols` downstream.** They're useful for building regression specs, correlation matrices, and follow-up subsets.
- **Don't over-engineer.** Plain top-level code that the student can copy line by line.
- **One step at a time.** After Step 4 wait for the user to scroll through the summary before barreling into plots or correlations.

## Growing this skill

This skill is one of a family of `pyrsm-eda-*` skills covering the `pyrsm.eda` module. When `distr` isn't the right tool, point the user to the sibling skill named for the operation they need (`pyrsm-explore`, `pyrsm-pivot`, `pyrsm-unpivot`, `pyrsm-combine`, `pyrsm-visualize`). The workflow shape — ask for absolute path → load via `scripts/load_data.py` → read sidecar → propose spec → run → interpret → extend with polars/plotnine — is shared across the family.
