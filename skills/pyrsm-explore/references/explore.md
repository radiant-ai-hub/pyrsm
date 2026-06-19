# pyrsm.eda.explore — reference

This file is the deeper reference for `pyrsm.eda.explore`. The main `SKILL.md` walks the workflow at a high level; come here for API details, interpretation templates, the full list of aggregation functions, and worked examples.

## Table of contents

1. Function signature
2. Supported aggregation functions
3. The `to_dummies` behavior
4. The two `header` layouts
5. Plain-English interpretation templates
6. Reading skew, missingness, and cardinality
7. Extending the output with polars
8. Related EDA / basics tools — when to switch
9. Worked examples (`diamonds`, `titanic`)
10. Common pitfalls

---

## 1. Function signature

```python
rsm.eda.explore(
    df,                          # pl.DataFrame or pl.LazyFrame
    cols=None,                   # list[str] | None — defaults to all numeric (+ dummies)
    agg=None,                    # list[str] | None — defaults to ["mean", "median", "min", "max", "sd"]
    by=None,                     # str | None — group-by column
    to_dummies=True,             # bool — auto-expand categoricals into dummies (drop_first=True)
    header="function",           # "function" or "variable"
) -> pl.DataFrame
```

Returns a `pl.DataFrame`. Always.

Key points:

- Accepts both DataFrames and LazyFrames; LazyFrames are materialized internally.
- No dict wrapper for naming the dataset (this is a function, not a class with a header that includes the name).
- `cols=None` auto-selects all numeric columns plus the dummy-expanded categorical columns (when `to_dummies=True`).
- `by` must reference an existing column. When set, the output column names become `{col}_{agg}` for each combination.
- `header="function"` is the row-by-variable layout (default); `header="variable"` transposes to stats-by-row.

## 2. Supported aggregation functions

The internal `EXPLORE_FUNCTIONS` dict maps aggregation-name strings to polars expressions:

| Name | Implementation | Notes |
| --- | --- | --- |
| `mean` | `pl.col(c).mean()` | Skips nulls. |
| `median` | `pl.col(c).median()` | Robust to outliers; lead with this for skewed variables. |
| `sum` | `pl.col(c).sum()` | Skips nulls. |
| `std` / `sd` | `pl.col(c).std()` | R-style `sd` alias for `std`. Uses `ddof=1` (sample sd). |
| `var` | `pl.col(c).var()` | Sample variance. |
| `min` | `pl.col(c).min()` | |
| `max` | `pl.col(c).max()` | |
| `count` / `n` | `pl.col(c).count()` | Non-null count. R-style `n` alias for `count`. |
| `n_unique` | `pl.col(c).n_unique()` | Distinct values, including nulls. |
| `n_missing` / `null_count` | `pl.col(c).null_count()` | Aliases. |

Default when `agg=None`: `["mean", "median", "min", "max", "sd"]`.

Unknown aggregation names raise `ValueError`.

## 3. The `to_dummies` behavior

By default, `to_dummies=True` and `explore` automatically converts categorical / Enum / String columns into dummy variables using `df.to_dummies(columns=cat_cols, drop_first=True)`. The resulting dummy columns are cast to `Float64` for consistent stats.

### What this looks like

A column `cut` with 5 levels (`Fair`, `Good`, `Very Good`, `Premium`, `Ideal`) becomes 4 dummies — say, `cut_Good`, `cut_Very Good`, `cut_Premium`, `cut_Ideal` (alphabetically first level dropped). Each dummy is 0/1 per row.

`explore` then includes these dummies in the summary alongside the genuinely numeric columns. The `mean` of `cut_Ideal` is the *proportion* of rows that are `Ideal` cut.

### When this is useful

- Quick check of categorical *distribution* without going through `pyrsm-pivot` — the dummy means give you the level frequencies.
- Input to a downstream regression where dummies are needed anyway.
- Mixed-type "describe everything" requests.

### When to turn it off

- The user wants a strict numeric-only summary. Set `to_dummies=False`.
- The user is confused by the auto-expansion (e.g., expected one row for `cut` and got 4).
- The categorical has dozens of levels and you don't want all of them in the summary. Either turn off and let `pyrsm-pivot` handle it, or filter `cols=` to only the dummies you want.

### Interpretation reminder

When `to_dummies=True`:

- `mean(<dummy>)` = `P(<level>)` = proportion in that level.
- `sd(<dummy>)` ≈ `sqrt(P(1−P))`.
- `min(<dummy>)` is always 0 if any row is not in the level.
- `max(<dummy>)` is always 1 if any row is in the level.

So `min=0, max=1` with `mean=0.3, sd≈0.46` is a "balanced-ish" two-level dummy — not anything weird about the data.

## 4. The two `header` layouts

### `header="function"` (default)

One row per variable, one column per aggregation. The first column is `variable`.

```
shape: (2, 6)
┌──────────┬──────────┬────────┬───────────┬───────┬─────────┐
│ variable ┆ mean     ┆ median ┆ sd        ┆ min   ┆ max     │
├──────────┼──────────┼────────┼───────────┼───────┼─────────┤
│ price    ┆ 3907.186 ┆ 2407.0 ┆ 3956.9154 ┆ 338.0 ┆ 18791.0 │
│ carat    ┆ 0.794283 ┆ 0.7    ┆ 0.473826  ┆ 0.2   ┆ 3.0     │
└──────────┴──────────┴────────┴───────────┴───────┴─────────┘
```

Best for: scanning many variables. Easy to sort by a single statistic.

### `header="variable"`

One row per aggregation, one column per variable. The first column is `statistic`.

```
shape: (2, 3)
┌───────────┬──────────┬──────────┐
│ statistic ┆ price    ┆ carat    │
├───────────┼──────────┼──────────┤
│ mean      ┆ 3907.186 ┆ 0.794283 │
│ median    ┆ 2407.0   ┆ 0.7      │
└───────────┴──────────┴──────────┘
```

Best for: comparing the same statistic across a small number of variables. Often easier to paste into a writeup.

### When `by` is set

The output is always one row per group; the columns become `{col}_{agg}`. `header` doesn't apply when `by` is set.

## 5. Plain-English interpretation templates

### Single-table description

> The summary table is shape `<R>×<C>`. Each row is a `<variable | statistic>`; each column is a `<statistic | variable>`. `<n>` observations were used for each summary (nulls dropped).

### Per-variable, lead with the right central tendency

> `<var>` has mean `<mean>` and median `<median>` `<unit>`. The mean is `<higher | lower | roughly equal to>` the median, suggesting `<right-skewed | left-skewed | symmetric>` distribution. The sd is `<sd>` `<unit>`, ranging from `<min>` to `<max>`. (`<n>` non-null observations; `<n_missing>` missing.)

### Grouped (with `by=`)

> `<var>` varies systematically across `<by>`. Group `<group_a>` (n=`<n_a>`) has mean `<mean_a>` `<unit>` and median `<median_a>`; group `<group_b>` (n=`<n_b>`) has mean `<mean_b>` and median `<median_b>`. The difference of `<mean_a - mean_b>` `<unit>` is `<discuss whether the difference is large relative to spread / business-meaningful>`. For a formal test of whether the group means differ, see the `pyrsm-compare-means` skill.

### Effect-size context

> sd / mean (the coefficient of variation) is `<cv>`. A CV near 0 indicates a tight distribution; CV near 1 indicates the sd is comparable to the mean (typical for right-skewed positive variables); CV > 1 indicates the spread exceeds the level (very high variability).

## 6. Reading skew, missingness, and cardinality

### Skew check

Compare mean and median:

- `|mean − median| / sd < 0.1` → roughly symmetric.
- `mean > median + 0.5 sd` → right-skew (long upper tail).
- `mean < median − 0.5 sd` → left-skew.

A right-skewed numeric variable (price, sales, income) is the rule, not the exception, in business data. A log transform often helps in modeling.

### Missingness check

Include `n_missing` (or its alias `null_count`) in `agg`:

```python
rsm.eda.explore(df, agg=["count", "null_count", "n_unique"])
```

Missingness is a frequent silent bug. A column with high `null_count` may need imputation, filtering, or special handling.

### Cardinality check

Include `n_unique`:

- Numeric columns with low `n_unique` (e.g., 2–10) are probably **really categorical**. Treating them as continuous in a model can give nonsense interpretations.
- Categorical columns with very high `n_unique` (hundreds or thousands) probably can't be one-hot encoded usefully — consider grouping or hashing.

## 7. Extending the output with polars

`explore` returns a `pl.DataFrame`. **The output is data, and you can keep working on it.** This is a first-class part of the workflow.

### Sort

```python
stats = rsm.eda.explore(df, cols=["price"], by="cut", agg=["mean", "count"])
stats.sort("price_mean", descending=True)
```

### Filter

```python
stats.filter(pl.col("price_count") >= 100)   # only groups with enough observations
```

### Add derived columns

```python
stats.with_columns(
    cv=pl.col("price_sd") / pl.col("price_mean"),                       # coefficient of variation
    n_below_median=pl.col("price_count") // 2,                          # rough below-median count
    rel_to_overall=pl.col("price_mean") / pl.lit(df["price"].mean()),   # relative to overall mean
)
```

### Pivot wide for presentation

```python
# Long-form: one row per group, mean and median in columns
# Wider: one row per cut, columns for each color's mean price
rsm.eda.pivot(df, rows="cut", cols="color", values="price", agg="mean")
# (this is also a pivot-table — see pyrsm-pivot)
```

### Concatenate summaries

```python
import polars as pl
together = pl.concat([
    rsm.eda.explore(df, cols=["price"], by="cut", agg=["mean", "median"]).rename({"cut": "group"}).with_columns(pl.lit("cut").alias("group_var")),
    rsm.eda.explore(df, cols=["price"], by="color", agg=["mean", "median"]).rename({"color": "group"}).with_columns(pl.lit("color").alias("group_var")),
])
```

### Pass downstream

The `stats` DataFrame can be the input to plotting (`pyrsm-visualize`), reshaping (`pyrsm-unpivot`), or joining with other data (`pyrsm-combine`). Treat it as a normal polars DataFrame — there is no special "stats object" wrapper.

## 8. Related EDA / basics tools — when to switch

- **One-variable distribution shape + plot** → `pyrsm-distr` (the `distr` class also classifies columns and plots histograms).
- **Two-variable cross-tabulation with aggregation** → `pyrsm-pivot`.
- **Wide → long reshaping** → `pyrsm-unpivot`.
- **Joining two DataFrames** → `pyrsm-combine`.
- **Custom plot of any variable** → `pyrsm-visualize`.
- **Hypothesis test on a mean vs benchmark** → `pyrsm-single-mean`.
- **Comparing group means** → `pyrsm-compare-means`.
- **Cross-tab + chi-squared independence test** → `pyrsm-cross-tabs`.

## 9. Worked examples (`diamonds`, `titanic`)

### Diamonds — default, all numeric columns

```python
import polars as pl
import pyrsm as rsm
diamonds = pl.read_parquet("<abs-path>/diamonds.parquet")
rsm.eda.explore(diamonds)
```

This computes mean / median / min / max / sd for all numeric columns (and all dummy-encoded categorical levels). The shape is around 24×6 because `clarity`, `cut`, and `color` expand to many dummy columns.

For a cleaner numeric-only view:

```python
rsm.eda.explore(diamonds, cols=["price", "carat", "depth", "table", "x", "y", "z"])
```

### Diamonds — focused, by group

```python
rsm.eda.explore(
    diamonds,
    cols=["price", "carat"],
    by="cut",
    agg=["mean", "median", "count", "sd"],
)
```

Output: one row per cut level, with `price_mean`, `price_median`, `price_count`, `price_sd`, then the same for `carat`. Useful for spotting that diamonds of `Ideal` cut have a lower mean price (despite being higher-quality cut) — a classic confounding pattern explained by smaller carats among Ideal-cut diamonds.

### Titanic — missingness check

```python
titanic = pl.read_parquet("<abs-path>/titanic.parquet")
rsm.eda.explore(titanic, agg=["count", "null_count", "n_unique"])
```

The `age` column shows `count=1043`, `null_count=263` — about 20% of passenger ages are missing in the source dataset. This is essential to know before modeling.

### Titanic — grouped means with count

```python
rsm.eda.explore(
    titanic,
    cols=["age", "fare"],
    by="pclass",
    agg=["mean", "median", "std", "count"],
)
```

Output: one row per pclass, with mean/median/sd/count for `age` and `fare`. Notice the dramatic differences in `fare_mean` across class (1st: $92, 2nd: $22, 3rd: $13) — exactly the substantive heterogeneity that drove the differential survival rates in the `pyrsm-cross-tabs` analysis of titanic.

### Transposed layout

```python
rsm.eda.explore(
    diamonds,
    cols=["price", "carat"],
    agg=["mean", "median"],
    header="variable",
)
```

Output: `statistic` column with rows `mean` and `median`; `price` and `carat` as columns. Easy to paste into a writeup as a 2-row, 2-column comparison.

## 10. Common pitfalls

- **Reading the mean of a heavy-tailed variable without the median.** Always pair them; lead with the median for skewed variables.
- **Misreading dummy means.** With `to_dummies=True`, the "mean" of `cut_Ideal` is the *proportion* of rows that are Ideal, not an average score. Either accept this and interpret as a proportion, or set `to_dummies=False` and reach for `pyrsm-pivot` for the categorical breakdown.
- **Forgetting `count` when using `by`.** Group-size imbalance changes how seriously to take a difference in means. Always include `count` in `agg` when grouping.
- **Forgetting `null_count`.** Means are computed on non-null observations only. If 30% of a column is null, the mean reported is of the 70% — be explicit about this in the writeup.
- **Trusting numeric columns with low `n_unique` as continuous.** A "rating" column with `n_unique=5` is ordinal — treat with care; `explore` will happily compute its mean.
- **Passing a non-existent column to `cols=` or `by=`.** Polars will raise a clear error, but it can be confusing for students. Always check `df.columns` first.
- **Passing an unknown `agg` name.** `explore` raises `ValueError` with the supported list — useful, but typing `"average"` instead of `"mean"` is a common mistake.
- **Trying to chain `.summary()` on the result.** `explore` returns a raw polars DataFrame, not a custom class. Use polars methods (`.sort`, `.filter`, `.with_columns`, etc.) for further work.
- **Two summaries with different `by=` columns concatenated without renaming.** Polars `pl.concat` is strict about column-name matches; rename the group column to a common name first.
- **Computing a sum when you really want a count.** `sum` adds the values; `count` counts the non-null rows. For 0/1 indicators they're numerically equal; for other columns they aren't.
