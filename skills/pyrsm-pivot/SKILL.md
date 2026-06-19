---
name: pyrsm-pivot
description: Build pivot tables, frequency tables, and crosstabs in Python using the pyrsm library's `pivot` function. Use this skill whenever a student or analyst wants to count occurrences of categorical levels, cross-tabulate two categorical variables, compute mean/median/sum/sd of a numeric value across grouping variables, normalize a crosstab into row/column/total percentages, add row and column totals, or work with polars-first wide-format outputs. The output is a polars DataFrame, so it can be further chained with any polars expression — extensibility is a first-class part of this skill. Triggers include phrases like "make a frequency table", "crosstab cut by color", "pivot table of mean price by cut and color", "row percentages of survival by class", "count rows per group", "summarize sales by region and product", "category counts with totals", or any descriptive cross-tabulation request in a marketing/business analytics context.
---

# pyrsm pivot workflow

This skill walks the user through building pivot tables, frequency tables, and crosstabs using `pyrsm.eda.pivot` — from loading their data to producing a wide-format table they can read or chain further polars operations onto. It is designed for students learning exploratory data analysis in a business/marketing analytics context.

For deep reference on the `pivot` API, the 40+ supported aggregation functions, normalization mechanics, and worked examples, see `references/pivot.md`.

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

The script prints a small JSON report describing the loaded dataset and the path of any sidecar `.md` it found.

### Sidecar description file

If a sidecar is found, **read it before deciding what to pivot**. The description tells you what each column means and what the levels of any categorical variables represent.

## Step 3 — Decide what to pivot

Once the data is in hand, propose a pivot specification and confirm with the user. You need:

- **Rows (`rows`)** — single column name or list. Becomes the rows of the output. For a frequency table of one variable, just pass `rows="cut"`. For a multi-key pivot, pass a list like `rows=["region", "store"]`.
- **Columns (`cols`)** — optional second categorical column for a crosstab. If `None`, the output is a single-column frequency table.
- **Values (`values`)** — optional numeric column to aggregate. If `None`, counts are returned. If provided, the default `agg` switches from `"count"` to `"mean"`.
- **Aggregation (`agg`)** — string name of the aggregation function. Defaults: `"count"` (frequency) or `"mean"` (when `values` is set). 40+ supported: see Step 6.
- **Normalization (`normalize`)** — `None`, `"row"`, `"column"`, or `"total"`. Converts counts/values into percentages. See Step 5.
- **Totals (`totals`)** — `True` to append a Total row (and a Total column in a crosstab).
- **Fill (`fill`)** — optional fill value for missing cells when no `values` is set (e.g., `fill=0` turns empty crosstab cells from `null` into `0`).

Confirm the specification with the user before running.

Do a quick sanity check before running:

- For a frequency table: is `rows` truly categorical? A numeric `rows` will produce one row per distinct value — fine for an integer rating, less useful for a continuous price.
- For a crosstab: are both `rows` and `cols` categorical with a manageable number of levels? An R×C output with R=20, C=15 is hard to read.
- If `values` is set: is it numeric? Non-numeric values + a `"mean"` agg will fail.
- For percentages: which conditional matters? Row percentages condition on the row variable; column percentages condition on the column variable; total percentages give cell-as-fraction-of-grand-total. See Step 5.

## Step 4 — Run pivot

Write a short, runnable Python script:

```python
import polars as pl
import pyrsm as rsm

# 1. Load the data (absolute path)
data_path = "<absolute-path>"
df = pl.read_parquet(data_path)

# 2. Build the pivot
result = rsm.eda.pivot(
    df,
    rows="<row-variable>",
    cols="<col-variable>",      # None for a frequency table
    values="<numeric>",          # None for counts
    agg="mean",                  # or "count", "sum", "median", "std", "sd", "var", "p25", "p75", "skew", etc.
    normalize="row",             # or "column" / "total" / None
    totals=True,                 # add row/column totals
    fill=0,                      # optional fill for empty cells
)

# 3. Inspect (it's a polars DataFrame)
print(result)
```

Notes:
- `pivot` accepts polars DataFrames and LazyFrames.
- No dict wrapper — this is a function.
- Output is a wide table: the values of `cols` become columns; the values of `rows` become row labels.
- When `totals=True`, the row-variable column is cast to `Utf8` to allow the `"Total"` label.

## Step 5 — Choosing the right normalization

This is the **first critical concept** for pivot. The three normalization options answer three different questions; mixing them up is a classic student mistake.

### `normalize="row"`

Each row sums to 100%. Answers: "given the row level, what fraction is in each column?"

Example: `pivot(titanic, rows="pclass", cols="survived", normalize="row")` shows what fraction of each passenger class survived. 1st class: ~63% Yes / 37% No. This is the natural "survival rate by class" view.

### `normalize="column"`

Each column sums to 100%. Answers: "given the column level, what fraction is in each row?"

Example: `pivot(titanic, rows="pclass", cols="survived", normalize="column")` shows what fraction of survivors (and what fraction of deaths) came from each class. Useful for asking "of the people who survived, what was their class composition?"

### `normalize="total"`

Each cell as a fraction of the grand total. The whole table sums to 100%.

Example: `pivot(titanic, rows="pclass", cols="survived", normalize="total")` shows what fraction of all passengers fell in each (class, outcome) combination.

### `normalize=None`

Raw counts (or values). The default for a count table; the natural choice when reporting absolute frequencies.

### Decision rule

- "What's the rate of X within each group?" → row percentages, with the group as `rows`.
- "Within each level of X, what's the composition?" → column percentages, with X as `cols`.
- "What share of the data is in each cell?" → total percentages.
- "How many rows in each cell?" → no normalization.

Always state which conditional you're reporting. "73.8% of 3rd-class passengers died" (`normalize="row"`) is a different fact from "59.7% of deaths were 3rd-class passengers" (`normalize="column"`).

## Step 6 — Choosing the right aggregation

This is the **second critical concept**. `pivot` supports 40+ aggregation functions because the right one depends on the question. The supported set:

- **Counts**: `count`, `n_obs` (aliases), `n_distinct`, `n_missing`.
- **Central tendency**: `sum`, `mean`, `median`, `min`, `max`.
- **Spread**: `std`, `sd` (alias), `var`, `se` (standard error), `me` (margin of error at z=1.96), `cv` (coefficient of variation = sd/mean), `iqr` / `IQR` (75th − 25th percentile).
- **Proportions** (for 0/1 columns): `prop` (= mean), `varprop` (= p(1-p)), `sdprop` (= sqrt(p(1-p))), `seprop` (= sqrt(p(1-p)/n)).
- **Percentiles**: `p01` through `p99` (any integer percentile).
- **Shape**: `skew`, `kurtosis` (via scipy).

### Which agg for which question

| Question | agg |
| --- | --- |
| How many rows per group? | `count` |
| Total sales per region? | `sum` |
| Typical price per cut? (median for skewed) | `median` |
| Average price per cut? (only if symmetric) | `mean` |
| Best- and worst-case per group | `min` / `max` |
| Variability of price per cut | `std` / `sd` |
| Variability relative to level | `cv` |
| Spread without outlier influence | `iqr` |
| 90th-percentile cutoff per group | `p90` |
| Proportion of binary success per group | `prop` |
| Skewness check per group | `skew` |

### Defaults

- If `values` is `None`: `agg="count"` (frequency table).
- If `values` is set and `agg` is not changed: `agg="mean"`.

If the user passes `values="price"` and doesn't change `agg`, you'll get a table of mean prices — which is often *not* what they want for a skewed variable. **Always ask which aggregation makes sense for the question** when `values` is involved, and prefer `median` for prices, sales, incomes, counts.

## Step 7 — Interpret the output

Walk the user through the table:

1. **Describe the shape.** "The output is `<R>×<C>`: one row per `<row level>`, one column per `<col level>`. `<Total row | column | both>` `<included | not included>`."

2. **Read the cells.** Match each cell back to the question. Watch for the dimension of the entries: counts vs proportions (×100) vs values in the original units.

3. **Compare across rows and columns.** If the table answers "how does the joint distribution vary", scan for cells with much larger or smaller values than their row / column average.

4. **Translate to a business conclusion using the sidecar units.** Generic "Premium cut has 144 in column E" is much weaker than "Premium-cut diamonds are 144 of the sample (about 18.7% of Premium-cut diamonds are color E)".

### Pivot vs. cross_tabs — when each fits

`pivot` is a *descriptive* tool — it builds the table. It does not test any hypothesis.

If the user also wants a **chi-squared test of independence** with expected counts and standardized deviations, route them to **`pyrsm-cross-tabs`** instead. `cross_tabs` produces the same observed table that `pivot(..., cols=..., totals=True)` gives, plus the inferential machinery.

Rule of thumb:
- "Build a count / percentage / mean table" → `pivot`.
- "Test whether these two categorical variables are associated" → `cross_tabs`.

## Step 8 — Extend with polars

`rsm.eda.pivot` returns a `pl.DataFrame`. The output is data, and you can keep working on it:

```python
result = rsm.eda.pivot(diamonds, rows="cut", cols="color", values="price", agg="median")

# 1. Sort rows by an arbitrary criterion
result.sort("D", descending=True)

# 2. Compute a derived column (e.g., total per row)
result.with_columns(
    pl.sum_horizontal([c for c in result.columns if c != "cut"]).alias("row_total")
)

# 3. Filter rows
result.filter(pl.col("cut").is_in(["Premium", "Ideal"]))

# 4. Unpivot back to long form for plotting
long = rsm.eda.unpivot(result, id_vars=["cut"], variable_name="color", value_name="median_price")
# (Then feed `long` into pyrsm-visualize)

# 5. Round for presentation
result.with_columns(pl.col(pl.Float64).round(2))
```

The wide-format pivot output is sometimes inconvenient for plotting (most plotnine geoms want long-form data). The natural pipeline is `pivot` → `unpivot` → `visualize`. See `pyrsm-unpivot` and `pyrsm-visualize` for the next steps.

## Step 9 — Related EDA / basics tools (when to switch)

`pivot` is the right tool for descriptive cross-tabulation. If the situation is different, suggest:

- **Per-column summary statistics across many variables (not a single cross-tab)** → `pyrsm-explore`.
- **Distribution shape of one variable + histogram** → `pyrsm-distr`.
- **Wide-to-long reshaping (often after pivot)** → `pyrsm-unpivot`.
- **Joining two DataFrames** → `pyrsm-combine`.
- **Custom plotting** → `pyrsm-visualize`.
- **Chi-squared test of independence on the same table** → `pyrsm-cross-tabs`.
- **Pairwise proportion comparison with adjustment** → `pyrsm-compare-props`.
- **Goodness-of-fit against a hypothesized distribution** → `pyrsm-goodness`.

Don't switch silently — explain why the alternative is a better fit and confirm with the user before changing tools.

---

## Style notes

- **Pedagogical tone.** Always state which `normalize` you used and what conditional that gives. Always state which `agg` and why.
- **Use the units from the description file.** "Mean price is 4505" is much weaker than "Fair-cut diamonds have a mean price of $4,505, n=101".
- **Prefer median over mean for skewed values.** Prices, sales, incomes, counts — all default to right-skewed. The `pivot` API doesn't push back automatically.
- **Always include `totals=True` unless the user has a reason not to.** Reading a count table without totals is harder.
- **When you do percentages, NEVER lose track of n.** Report both: 73.8% of 3rd-class passengers died (n=500 for that cell). Without `n`, percentages can mislead.
- **Pivot is descriptive, not inferential.** For hypothesis tests, route to `pyrsm-cross-tabs` or `pyrsm-compare-props`.
- **Extend with polars.** The wide output is the start of an analysis, often followed by `unpivot` → `visualize`.
- **Don't over-engineer.** Plain top-level code that the student can copy line by line.
- **One step at a time.** After Step 4 wait for the user to look at the table before suggesting transformations.

## Growing this skill

This skill is one of a family of `pyrsm-eda-*` skills covering the `pyrsm.eda` module. When `pivot` isn't the right tool, point the user to the sibling skill named for the operation they need. The workflow shape — ask for absolute path → load via `scripts/load_data.py` → read sidecar → propose spec → run → interpret → extend with polars — is shared across the family.
