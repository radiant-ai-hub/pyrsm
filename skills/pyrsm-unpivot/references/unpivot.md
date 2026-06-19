# pyrsm.eda.unpivot — reference

This file is the deeper reference for `pyrsm.eda.unpivot`. The main `SKILL.md` walks the workflow at a high level; come here for API details, the `pivot ↔ unpivot` relationship, and worked examples.

## Table of contents

1. Function signature
2. The `on` and `id_vars` semantics
3. The shape transformation
4. Plain-English interpretation templates
5. When wide-to-long is needed (and when it isn't)
6. The `pivot ↔ unpivot` roundtrip
7. Extending the output with polars
8. Worked examples
9. Common pitfalls

---

## 1. Function signature

```python
rsm.eda.unpivot(
    df,                            # pl.DataFrame or pl.LazyFrame
    on=None,                       # str | list[str] | None — columns to unpivot
    id_vars=None,                  # str | list[str] | None — columns to keep as identifiers
    variable_name="variable",      # name of the new "what column was this from" column
    value_name="value",            # name of the new "value" column
) -> pl.DataFrame
```

Returns a `pl.DataFrame`. Always.

Internally calls `df.unpivot()` (polars' built-in). The pyrsm wrapper exists for naming consistency with the rest of `pyrsm.eda` and to normalize `id_vars` / `on` arguments to lists.

Defaults:
- `on=None` → unpivot all columns *not* in `id_vars`.
- `id_vars=None` → no identifier columns; every cell of the input becomes one row.
- `variable_name="variable"` and `value_name="value"` are placeholders; replace them with substantive names.

## 2. The `on` and `id_vars` semantics

`id_vars` are columns that **survive** the unpivot — they appear in the output, once per input row that goes into the long form, repeated as needed.

`on` are columns that **disappear** from the column space and **appear** as values in the new `variable_name` column.

### Both specified

```python
df = pl.DataFrame({
    "region": ["N", "S", "E", "W"],
    "Q1": [100, 200, 150, 175],
    "Q2": [120, 220, 160, 180],
})
rsm.eda.unpivot(df, on=["Q1", "Q2"], id_vars="region", variable_name="quarter", value_name="sales")
# shape: (8, 3)
# region | quarter | sales
# N      | Q1      | 100
# S      | Q1      | 200
# E      | Q1      | 150
# W      | Q1      | 175
# N      | Q2      | 120
# S      | Q2      | 220
# E      | Q2      | 160
# W      | Q2      | 180
```

### `on=None`

When `on` is None, all columns *not* in `id_vars` are unpivoted:

```python
rsm.eda.unpivot(sales_wide, id_vars="region")
# All of Q1, Q2, Q3, Q4 (and any other non-id columns) are unpivoted.
```

This is convenient but error-prone if you forget to set `id_vars` — `on=None` and `id_vars=None` will try to unpivot every column, often resulting in a type-coercion warning or silent error.

### `id_vars=None`

When `id_vars` is None, no columns are kept as identifiers — every input cell becomes one row of `(variable, value)`. Useful for "flatten this 2D table into a long list of values" but rarely what you want.

### Multi-column `id_vars`

```python
rsm.eda.unpivot(df, on=["sales", "revenue"], id_vars=["region", "year"])
```

The output keeps `region` and `year` as identifier columns, and unpivots `sales` and `revenue` into `variable` and `value`.

## 3. The shape transformation

```
input:   shape (R, C)   where C = len(id_vars) + len(on)
output:  shape (R * len(on), len(id_vars) + 2)
```

- Rows multiply by the number of unpivoted columns.
- Output columns = `id_vars` + `variable_name` + `value_name`.
- Total cells: same (just rearranged).

### Type coercion

If the columns in `on` have different polars types, polars will pick a common type for the new `value` column:
- All-numeric: widens to `Float64` if any are floats; otherwise the widest integer type.
- Mixed numeric + string: widens to `str`.
- All-string: stays `str`.

If the unpivot produces a `str` column when you expected numeric, check that all `on` columns were really numeric in the input. A stray string column (or a `None`-filled column with no explicit dtype) can force the cast.

## 4. Plain-English interpretation templates

### Header

> We unpivoted `<len(on)>` columns of `<df name>` into a long-format table. Each row of the original wide format produces `<len(on)>` rows in the long format. The new columns are: `<id_vars>` (kept from the input), `<variable_name>` (holds the original column names), and `<value_name>` (holds the cell values).

### Why and what next

> The long form is the right shape for plotting (the `<variable_name>` becomes an aesthetic mapping), for group-by aggregation across the unpivoted axis, and for modeling (the `<variable_name>` becomes a categorical predictor). The wide form was better for human reading and for one-off arithmetic between specific columns.

### Naming reminder

> The defaults `variable_name="variable"` and `value_name="value"` are placeholders. In this analysis they should be `<substantive name>` (the dimension we're unpivoting, e.g., `quarter`) and `<substantive name>` (what the cell values represent, e.g., `sales`).

## 5. When wide-to-long is needed (and when it isn't)

### Need long

- **Plot a categorical on an axis with one row per category.** A bar chart "sales by quarter, faceted by region" wants a `quarter` column, not `Q1`/`Q2`/`Q3`/`Q4` columns.
- **Aggregate across a wide axis.** "Average sales across all quarters per region" — in long form, `df.group_by("region").agg(pl.col("sales").mean())`. In wide form, `df.with_columns(mean=pl.mean_horizontal(["Q1","Q2","Q3","Q4"]))` works but scales poorly to many quarters.
- **Regress on the wide axis as a categorical predictor.** `regress(sales ~ quarter + region)` needs `quarter` as a single column.
- **Join with another long-format table.** Two long-format tables join naturally on the shared `(id_vars, variable_name)` keys.

### Don't need long

- **Wide is the final report.** A manager-facing pivot table is best in wide form.
- **Conceptually different columns.** `population`, `gdp`, `area` are three different variables of a country, not three "values" of a single variable. Don't unpivot them.
- **One-off arithmetic between two specific columns.** `df.with_columns(growth = (pl.col("Q4") / pl.col("Q1") - 1) * 100)` is simpler in wide form.

### Heuristic: would the same row appear multiple times if you tidy?

If yes (each row of long form has different `(id × variable)`): unpivot is right.

If no (each row of long form is a different observation entirely): the data was probably already tidy — you may be looking for a different operation.

## 6. The `pivot ↔ unpivot` roundtrip

`pivot` and `unpivot` are inverse operations when the original wide table has no aggregation:

```python
# Start long
sales_long = pl.DataFrame({
    "region": ["N", "N", "S", "S"],
    "quarter": ["Q1", "Q2", "Q1", "Q2"],
    "sales": [100, 120, 200, 220],
})

# To wide
sales_wide = rsm.eda.pivot(sales_long, rows="region", cols="quarter", values="sales", agg="sum")
# region | Q1  | Q2
# N      | 100 | 120
# S      | 200 | 220

# Back to long
sales_back = rsm.eda.unpivot(sales_wide, on=["Q1", "Q2"], id_vars="region",
                             variable_name="quarter", value_name="sales")
# region | quarter | sales
# N      | Q1      | 100
# N      | Q2      | 120
# S      | Q1      | 200
# S      | Q2      | 220
```

This roundtrip is **lossless** when `pivot` had nothing to aggregate (one input row per `(rows, cols)` combination). When `pivot` did aggregate (`agg="mean"` over multiple rows per cell), the unpivot recovers the aggregated values, not the raw observations — `pivot → unpivot` is no longer an identity.

### The canonical pipeline

`raw long data → pivot to wide summary → unpivot to plottable long → visualize`

Each step has a single role. The "plottable long" form is the long form of the *summary statistics*, which is what plotnine wants.

## 7. Extending the output with polars

`rsm.eda.unpivot` returns a `pl.DataFrame`. Common next steps:

### Filter levels

```python
long.filter(pl.col("quarter") != "Q1")
```

### Sort

```python
long.sort(["region", "quarter"])
```

### Cast the variable column if it's actually numeric

If you unpivoted columns named `2019`, `2020`, `2021`, the new `year` column is `str`. Cast for further math:

```python
long.with_columns(pl.col("year").cast(pl.Int32))
```

### Group-by + aggregate

```python
long.group_by("quarter").agg(pl.col("sales").mean().alias("mean_sales"))
```

### Plot directly

```python
rsm.eda.visualize(long, x="quarter", y="sales", color="region", geom="line")
```

### Re-pivot for a different view

```python
# After unpivoting and filtering, pivot back to a different shape
long.filter(pl.col("quarter").is_in(["Q1", "Q4"])).pivot(
    on="quarter", index="region", values="sales", aggregate_function="first"
)
```

## 8. Worked examples

### Quarterly sales — basic unpivot

```python
import polars as pl
import pyrsm as rsm

sales_wide = pl.DataFrame({
    "region": ["N", "S", "E", "W"],
    "Q1": [100, 200, 150, 175],
    "Q2": [120, 220, 160, 180],
    "Q3": [140, 240, 170, 190],
    "Q4": [160, 260, 180, 200],
})

sales_long = rsm.eda.unpivot(
    sales_wide,
    on=["Q1", "Q2", "Q3", "Q4"],
    id_vars="region",
    variable_name="quarter",
    value_name="sales",
)
# shape: (16, 3)
# 16 = 4 regions × 4 quarters
```

The substantively renamed columns (`quarter`, `sales`) are immediately usable — no further cleanup needed.

### Plot the long-form data

```python
rsm.eda.visualize(sales_long, x="quarter", y="sales", color="region", geom="line")
```

A simple line chart of sales over quarters, one line per region — impossible without the long form.

### Pivot of diamonds price, then unpivot for plotting

```python
diamonds = pl.read_parquet("<abs-path>/diamonds.parquet")

# 1. Build a wide pivot of mean price by cut × color
price_pivot = rsm.eda.pivot(diamonds, rows="cut", cols="color", values="price", agg="mean")
# shape: (5, 8)

# 2. Unpivot for plotting
price_long = rsm.eda.unpivot(
    price_pivot,
    on=["D", "E", "F", "G", "H", "I", "J"],
    id_vars="cut",
    variable_name="color",
    value_name="mean_price",
)
# shape: (35, 3)

# 3. Plot
rsm.eda.visualize(price_long, x="color", y="mean_price", color="cut", geom="bar", position="dodge")
```

This is the canonical "pivot → unpivot → visualize" pipeline.

### Multi-key id_vars (longitudinal data)

```python
panel_wide = pl.DataFrame({
    "country": ["US", "US", "UK", "UK"],
    "year": [2020, 2021, 2020, 2021],
    "gdp": [21000, 22000, 2900, 3100],
    "pop": [330, 332, 67, 68],
})

# Unpivot 'gdp' and 'pop' (two SEPARATE measures) into long form
long = rsm.eda.unpivot(
    panel_wide,
    on=["gdp", "pop"],
    id_vars=["country", "year"],
    variable_name="measure",
    value_name="value",
)
# shape: (8, 4)
# country | year | measure | value
# US      | 2020 | gdp     | 21000
# US      | 2020 | pop     | 330
# ...
```

Note that `gdp` and `pop` are conceptually different variables — this unpivot collapses them into a single `value` column. That's useful for plotting both on the same axis (with `color="measure"`) or for a faceted view, but loses the unit context: gdp is in $B, pop is in millions. A reader of the long form has to consult the `measure` column to know which.

### `on=None` for "all non-id columns"

```python
rsm.eda.unpivot(sales_wide, id_vars="region")
# Unpivots Q1, Q2, Q3, Q4 (everything except region)
```

Convenient for quick reshaping when there are many columns to unpivot.

### Subset of columns

```python
rsm.eda.unpivot(sales_wide, on=["Q1", "Q2"], id_vars="region")
# Only Q1 and Q2 are unpivoted; Q3 and Q4 remain as wide columns in the output
```

Useful for "tidy half of the data and keep the rest wide" — rarely needed but possible.

## 9. Common pitfalls

- **Forgetting `id_vars`.** Without it, every cell becomes a row, and you lose the connection between rows and their identifiers. Always specify `id_vars`.
- **Leaving `variable_name="variable"` and `value_name="value"`.** Default placeholders are not analysis-ready. Pick substantive names.
- **Type coercion of mixed columns.** If `on` contains both numeric and string columns, the output `value` column is `str`. If you wanted numeric, fix the input.
- **Unpivoting conceptually different columns.** `gdp` and `population` are different variables, not levels of the same variable. Unpivoting them together can be useful for plotting but obscures the unit difference.
- **Forgetting to cast the new variable column if it's actually numeric.** If column names like `2019` get unpivoted into a `year` column, it's `str` after unpivot. Cast for arithmetic.
- **Treating `unpivot` as an aggregation.** It is *not* — it preserves all cells, just rearranges them. To aggregate after unpivoting, chain `.group_by(...).agg(...)`.
- **Assuming `pivot → unpivot` is always lossless.** Only when the pivot didn't aggregate (single input row per output cell). If `pivot` used `agg="mean"` over multiple rows per cell, you can't recover the original rows from the unpivot.
- **Re-running `unpivot` on already-long data.** No effect (except possibly renaming columns or stacking the value column oddly). Verify shape first.
- **Forgetting that LazyFrames are materialized.** Polars' `unpivot` is eager; pyrsm's wrapper calls `.collect()` on a LazyFrame input. If you want a lazy unpivot pipeline, use the polars API directly.
