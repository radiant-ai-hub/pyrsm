---
name: pyrsm-unpivot
description: Convert wide-format data into long format (melt / gather) in Python using the pyrsm library's `unpivot` function. Use this skill whenever a student or analyst has columns that should really be rows of a categorical (Q1/Q2/Q3/Q4 → quarter; 2019/2020/2021 → year; product_a/product_b → product), needs to reshape data for plotting with plotnine, wants to feed a wide pivot table back into a long-format analysis, or generally needs to make the data "tidy" for a downstream tool. The output is a polars DataFrame the user can keep chaining onto. Triggers include phrases like "wide to long", "melt this table", "unpivot the quarters", "convert columns Q1-Q4 to a single quarter column", "reshape for plotting", "make this tidy", "tidy data", or any mention of needing long-format data after producing a wide-format summary.
---

# pyrsm unpivot workflow

This skill walks the user through converting wide-format data into long format using `pyrsm.eda.unpivot` — from loading their data to producing a tidy long-form DataFrame they can plot, model, or chain further polars operations onto. It is designed for students learning data reshaping in a business/marketing analytics context.

For deep reference on the `unpivot` API, the `on` and `id_vars` parameters, the relationship to `pivot`, and worked examples, see `references/unpivot.md`.

## Output policy

Two rules apply to **every** pyrsm call you make in this skill:

1. **Show the full output verbatim.** When you call `.summary()`, `.predict()`, or any pyrsm method that prints results, paste the **complete printed output** into your response — every line of the coefficient table, every statistic, every row. Do NOT abbreviate. Do NOT replace it with a high-level "main takeaways" bullet list. Do NOT skip the table to "save space". The user is reading your response because they want to see the actual numbers, not your summary of them.

2. **Provide the full pedagogical interpretation, not lame bullets.** When walking through results, use the detailed templates in `references/<name>.md`: state hypotheses explicitly, walk every coefficient or cell with its real-world units from the sidecar description, give the significance verdict at the 5% level, use the "holding constant" qualifier (where applicable), and tie back to the business question. A response like "X is positive and significant; Y is positive and significant; Z is not" is exactly the failure mode this skill is built to prevent.

3. **Lead with the requested output.** When the user asks for a specific plot, table, or diagnostic (e.g., "create and interpret PDP", "show me the residual dashboard", "generate PIP", "give me the confusion matrix", "plot the gains chart"), your interpretation must focus on THAT output — not on a re-interpretation of the model's coefficient table or summary. Pi's failure pattern is to say *"I'll generate X and then interpret the fitted results"* and then quietly produce a coefficient walk-through. The user already saw the coefficient interpretation in the previous turn; do not repeat it unless they explicitly ask. Lead with the requested output's content, using the dedicated template for that output if the skill has one (Step 9 for PIP, Step 10 for PDP, etc.).

If your context is genuinely constrained and you cannot reproduce the full output, say so explicitly — never silently abbreviate.


---

## Step 1 — Ask for the data location (or accept an in-memory DataFrame)

The very first thing to do is figure out where the data is coming from. Three common cases:

- **The user has a file.** Require an **absolute path** to a parquet / csv / etc. (same as the other eda skills).
- **The user has an in-memory polars DataFrame** that's the output of a previous step (e.g., a `pyrsm-pivot` result). Just accept it and proceed.
- **The user describes the wide shape in words** ("I have quarterly columns Q1-Q4 with one row per region"). Ask them for the absolute path to a file, or to give you the polars code that produces the DataFrame.

If they give a relative or `~` path, ask for an absolute one.

## Step 2 — Load the data (if needed)

If loading from a file, use `scripts/load_data.py`:

```bash
python scripts/load_data.py "<absolute-path-to-data>"
```

The script prints a JSON report. **Pay special attention to the column list** — for unpivot, the column names are the candidate "values" that become rows. If the wide columns share an obvious naming pattern (`Q1`/`Q2`/`Q3`/`Q4`, `price_2019`/`price_2020`, …), the pattern is the user's encoded categorical that we're about to surface as a column of its own.

## Step 3 — Decide what to unpivot

Confirm the reshape specification:

- **`id_vars`** — column(s) that stay as identifier columns (one per row in the long form). These typically encode the "primary keys" of the wide table — e.g., `region` in a `region × Q1/Q2/Q3/Q4` table.
- **`on`** — column(s) to unpivot into the long form. Each will contribute one row per input row to the output. If `on=None`, *all* columns not in `id_vars` are unpivoted (useful, but error-prone if you forget to set `id_vars`).
- **`variable_name`** — name of the new column that holds the *original column names*. Default `"variable"`. Pick something substantive (`"quarter"`, `"year"`, `"product"`).
- **`value_name`** — name of the new column that holds the *original cell values*. Default `"value"`. Pick something substantive (`"sales"`, `"revenue"`, `"pct"`).

Confirm the spec, e.g.: "I'll unpivot Q1–Q4 into a `quarter` column and put the cell values into a `sales` column; `region` stays as the id."

Do a quick sanity check before running:

- The columns in `on` should be **homogeneous in type** — i.e., the values that become the new `value_name` column should all be the same kind of thing (all numeric sales figures, all string survey responses, etc.). Polars will widen types if they differ, but it can be surprising.
- If the wide columns have *very different* meanings (e.g., one is a price and another is a count), unpivoting them into a single `value` column is rarely useful — those are conceptually different variables.
- For a roundtrip with `pivot`: the long form has one row per `(id × variable)` combination; the wide form has one row per `id` and one column per `variable`.

## Step 4 — Run unpivot

```python
import polars as pl
import pyrsm as rsm

# 1. Load the data (or get from previous step)
df = pl.read_parquet("<absolute-path>")   # or build / pass through

# 2. Unpivot
long = rsm.eda.unpivot(
    df,
    on=["<col_a>", "<col_b>", "<col_c>"],   # or None for "all non-id columns"
    id_vars=["<id_col>"],
    variable_name="<substantive-name>",     # e.g., "quarter"
    value_name="<substantive-name>",        # e.g., "sales"
)

# 3. Inspect
print(long)
```

Notes:
- The output is a `pl.DataFrame` in long format.
- Number of rows = `(rows in input) × (len(on))`.
- Number of columns = `len(id_vars) + 2` (the new `variable_name` and `value_name` columns).
- `on` can be a single string, a list, or `None` (which means "all non-id_vars columns").
- `id_vars` can be a single string, a list, or `None` (which means "no id columns" — every row becomes one cell).

## Step 5 — Interpret the output

The long form has one row per `(id × variable)` combination:

1. **Describe the shape transformation.** "We went from `<R>` rows × `<C>` columns wide to `<R × n_pivoted>` rows × `<len(id_vars) + 2>` columns long. The wide column names are now values in `<variable_name>`."

2. **Read the columns.** The `<variable_name>` column holds what *used to be* the column names — typically a discrete categorical (Q1/Q2/Q3/Q4, year-labels, product-names). The `<value_name>` column holds the cell values — typically numeric (sales, prices, counts).

3. **Why this matters.** Long form is the canonical "tidy" shape for plotting and modeling:
   - **Plotting**: `plotnine` (and most modern plotting libraries) want one row per observation with the grouping variable as a column, not as a column-name.
   - **Modeling**: a regression of sales on (region, quarter) needs `quarter` as a column, not 4 sales columns.
   - **Group-by**: `df.group_by("quarter").agg(pl.col("sales").mean())` only works after unpivoting.

## Step 6 — When wide-to-long is needed (the critical concept)

This is the **defining pedagogical concept** for `unpivot`: knowing *when* you need it.

### Symptoms that you need to unpivot

- **Plotting**: "I want a bar chart with quarter on the x-axis and sales on the y-axis, but my data has 4 separate `Q1`/`Q2`/`Q3`/`Q4` columns."
- **Faceting**: "I want one panel per product, but products are in columns."
- **Aggregating across a wide axis**: "What's the average sales across all quarters per region?" Wide: `df.with_columns(mean=pl.mean_horizontal(["Q1","Q2","Q3","Q4"]))`. Long: `df.group_by("region").agg(pl.col("sales").mean())`. The long form is easier and scales to more quarters without code changes.
- **Modeling**: "I want to regress sales on quarter and region." The regression needs `quarter` as a single column with 4 levels, not 4 different sales columns.

### Symptoms that you do NOT need to unpivot

- **Wide format is the final report shape.** A pivot table for a manager is usually best in wide form.
- **Columns are conceptually different.** `region`, `population`, `gdp` are three different variables, not three "values" of one variable. Don't unpivot them.
- **One-off arithmetic between two columns.** `df.with_columns(diff = pl.col("Q4") - pl.col("Q1"))` is simpler in wide form.

### Naming the new columns

`variable_name` and `value_name` default to `"variable"` and `"value"`. **These defaults are almost always wrong for a writeup.** Pick substantive names that describe what the column actually represents (`quarter`, `year`, `product`; `sales`, `revenue`, `count`, `pct`).

A reader of `df["variable"]` has to ask "variable of what?". A reader of `df["quarter"]` doesn't.

## Step 7 — Extend with polars (pivot → unpivot → analyze)

`rsm.eda.unpivot` returns a `pl.DataFrame`. The natural follow-ups:

### Filter

```python
long.filter(pl.col("quarter") != "Q1")        # exclude a level
long.filter(pl.col("sales") > 100)            # threshold filter
```

### Sort

```python
long.sort(["region", "quarter"])
```

### Aggregate after unpivoting

```python
long.group_by("quarter").agg(pl.col("sales").mean().alias("mean_sales"))
```

### Plot

```python
rsm.eda.visualize(long, x="quarter", y="sales", color="region", geom="line")
```

### Round-trip through pivot

```python
# Long → wide
back = rsm.eda.pivot(long, rows="region", cols="quarter", values="sales", agg="sum")
```

This roundtrip is exact when the original cells were single values; lossy if you aggregated during pivot.

### Convert numerical levels back to numeric

If the column names that became `variable` are actually numeric (e.g., years `2019`, `2020`, `2021`), the `variable_name` column will be `str` after unpivot. Cast it:

```python
long.with_columns(pl.col("year").cast(pl.Int32))
```

## Step 8 — Related EDA tools (when to switch)

`unpivot` is the right tool when going wide → long. If the situation is different, suggest:

- **Long → wide** → `pyrsm-pivot`.
- **Joining two DataFrames** → `pyrsm-combine`.
- **Per-column summary statistics** → `pyrsm-explore`.
- **Distribution analysis** → `pyrsm-distr`.
- **Plotting the long-form output** → `pyrsm-visualize`.

Don't switch silently — explain why the alternative is a better fit and confirm with the user before changing tools.

---

## Style notes

- **Pedagogical tone.** Always state *why* the user needs long form (plotting, modeling, group-by). Don't unpivot for its own sake.
- **Always rename `variable_name` and `value_name`.** Defaults `"variable"` and `"value"` are placeholders, not analysis-ready column names.
- **Mind the column types.** If the columns being unpivoted are heterogeneous (mix of numeric and string), polars will widen to a common type — usually `str`. If you wanted numeric, fix the input or cast the output.
- **Pivot → unpivot is the canonical pipeline for going from raw long → wide summary → plottable long.** Mention this when relevant.
- **Don't over-engineer.** Plain top-level code that the student can copy line by line.
- **One step at a time.** After Step 4 wait for the user to look at the table before suggesting plots or aggregations.

## Growing this skill

This skill is one of a family of `pyrsm-eda-*` skills covering the `pyrsm.eda` module. When `unpivot` isn't the right tool, point the user to the sibling skill named for the operation they need (`pyrsm-pivot`, `pyrsm-explore`, `pyrsm-combine`, `pyrsm-distr`, `pyrsm-visualize`). The workflow shape — ask for absolute path → load via `scripts/load_data.py` → read sidecar → propose spec → run → interpret → extend with polars — is shared across the family.
