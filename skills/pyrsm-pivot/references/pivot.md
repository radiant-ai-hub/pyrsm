# pyrsm.eda.pivot — reference

This file is the deeper reference for `pyrsm.eda.pivot`. The main `SKILL.md` walks the workflow at a high level; come here for API details, the full aggregation registry, normalization mechanics, and worked examples.

## Table of contents

1. Function signature
2. Frequency tables (no `cols`)
3. Crosstabs (with `cols`)
4. Aggregation registry — all 40+ functions
5. Normalization: row vs column vs total
6. Totals and how they interact with normalization
7. Plain-English interpretation templates
8. Extending the output with polars (pivot → unpivot → visualize)
9. Pivot vs cross_tabs
10. Worked examples (`diamonds`, `titanic`)
11. Common pitfalls

---

## 1. Function signature

```python
rsm.eda.pivot(
    df,                          # pl.DataFrame or pl.LazyFrame
    rows,                        # str or list[str]
    cols=None,                   # str | None — column variable for crosstab
    values=None,                 # str | None — numeric column to aggregate
    agg="count",                 # one of the 40+ registered aggregations
    normalize=None,              # None, "row", "column", "total"
    totals=False,                # add row+column totals
    fill=None,                   # fill value for empty cells (only when values=None)
) -> pl.DataFrame
```

Returns a `pl.DataFrame`. Always.

Defaults:
- If `values=None`, `agg` defaults to `"count"`. If you pass `agg="count"` explicitly with `values` set, pivot still computes counts (ignores `values`).
- If `values` is set and you don't change `agg`, it switches to `"mean"`.
- `normalize=None` returns raw counts/values.
- `totals=False` skips the Total row/column.

`rows` can be a single string or a list of strings. Multi-key rows produce one row per unique combination of the key columns.

`cols` accepts only a single column (no multi-column pivoting on the column axis).

`fill` applies only when `values=None` (i.e., for crosstabs of counts where an `(row, col)` combination doesn't appear in the data; passing `fill=0` makes those cells `0` instead of `null`).

## 2. Frequency tables (no `cols`)

```python
rsm.eda.pivot(df, rows="cut")
# shape: (5, 2)
# ┌───────────┬───────┐
# │ cut       ┆ count │
# ╞═══════════╪═══════╡
# │ Ideal     ┆ 1176  │
# │ Premium   ┆ 771   │
# │ Very Good ┆ 677   │
# │ Good      ┆ 275   │
# │ Fair      ┆ 101   │
# └───────────┴───────┘
```

When `cols=None`, the output is a one-column long table with `count` (or `<values>_<agg>` when `values` is set). Sort order is data-dependent; chain `.sort()` for a stable display.

With `values` set:

```python
rsm.eda.pivot(df, rows="cut", values="price")
# Returns a 'price_mean' column per cut level (mean is the default when values is set)

rsm.eda.pivot(df, rows="cut", values="price", agg="median")
# Returns 'price_median' instead
```

With `normalize`:

```python
rsm.eda.pivot(df, rows="cut", normalize="total")
# Adds a 'count_pct' column with each cell as a fraction of the grand total × 100
```

With `totals`:

```python
rsm.eda.pivot(df, rows="cut", totals=True)
# Appends a 'Total' row at the bottom
# Cuts the row-variable column to Utf8 so the "Total" label fits
```

Multi-key:

```python
rsm.eda.pivot(df, rows=["cut", "color"], values="price", agg="mean")
# One row per (cut, color) combination
```

## 3. Crosstabs (with `cols`)

```python
rsm.eda.pivot(df, rows="cut", cols="color")
# shape: (5, 8)   # 5 cut levels × (color levels + cut column)
# Each cell is the count of diamonds with that (cut, color).
```

With `values`:

```python
rsm.eda.pivot(df, rows="cut", cols="color", values="price", agg="mean")
# Each cell is the mean price for that (cut, color).
```

With normalization:

```python
rsm.eda.pivot(df, rows="cut", cols="color", normalize="row")
# Each row sums to 100. Cell = P(color | cut) × 100.

rsm.eda.pivot(df, rows="cut", cols="color", normalize="column")
# Each column sums to 100. Cell = P(cut | color) × 100.

rsm.eda.pivot(df, rows="cut", cols="color", normalize="total")
# All cells sum to 100. Cell = P(cut, color) × 100.
```

With totals + normalization:

```python
rsm.eda.pivot(df, rows="cut", cols="color", normalize="row", totals=True)
# Row Total column = 100 (each row); Col Total row = overall column proportions
```

## 4. Aggregation registry — all 40+ functions

The internal `AGG_FUNCTIONS` dict registers these:

### Counts

| Key | What it computes |
| --- | --- |
| `count` | `pl.len()` — row count, ignores the `values` column |
| `n_obs` | same as `count` |
| `n_distinct` | `pl.col(values).n_unique()` |
| `n_missing` | `pl.col(values).null_count()` |

### Central tendency

| Key | What it computes |
| --- | --- |
| `sum` | `pl.col(values).sum()` |
| `mean` | `pl.col(values).mean()` |
| `median` | `pl.col(values).median()` |
| `min` | `pl.col(values).min()` |
| `max` | `pl.col(values).max()` |

### Spread

| Key | What it computes |
| --- | --- |
| `std` / `sd` | `pl.col(values).std()` (ddof=1) |
| `var` | `pl.col(values).var()` |
| `se` | `sd / sqrt(n)` (standard error of the mean) |
| `me` | `1.96 × se` (margin of error at z=1.96, ≈ 95%) |
| `cv` | `sd / mean` (coefficient of variation) |
| `iqr` / `IQR` | `p75 − p25` |

### Proportions (for 0/1 columns)

| Key | What it computes |
| --- | --- |
| `prop` | `mean(0/1)` — proportion of 1s |
| `varprop` | `p × (1 − p)` |
| `sdprop` | `sqrt(p × (1 − p))` |
| `seprop` | `sqrt(p × (1 − p) / n)` |

### Percentiles

`p01` through `p99` — any integer percentile from 1 to 99. E.g., `p10` is the 10th percentile, `p90` is the 90th.

### Shape

| Key | What it computes (via scipy) |
| --- | --- |
| `skew` | `scipy.stats.skew(..., bias=False)` |
| `kurtosis` | `scipy.stats.kurtosis(..., bias=False)` (excess kurtosis) |

### Choosing aggregations

- For a frequency table: `count` (default).
- For absolute totals: `sum`.
- For a "typical value" by group: `median` for skewed; `mean` for symmetric.
- For variability comparison: `cv` (scale-free) or `sd`.
- For tails: `p10` / `p90` / `iqr`.
- For binary success rates: `prop` (or `mean`, equivalent).
- For shape diagnostics: `skew`, `kurtosis`.

Unknown aggregation names raise `ValueError`.

## 5. Normalization: row vs column vs total

The three normalization options answer three different questions. They are the **defining critical concept** for crosstabs.

### `normalize="row"` — row-conditional

Each row sums to 100. Cell `(i, j)` = `P(col = j | row = i) × 100`.

Question answered: "Given the row level, what fraction is in each column level?"

Example: `pivot(titanic, rows="pclass", cols="survived", normalize="row")` shows survival rate by class. 1st class row: 36.5% No, 63.5% Yes — the survival rate for 1st class is 63.5%.

### `normalize="column"` — column-conditional

Each column sums to 100. Cell `(i, j)` = `P(row = i | col = j) × 100`.

Question answered: "Given the column level, what fraction is in each row level?"

Example: `pivot(titanic, rows="pclass", cols="survived", normalize="column")` shows the class breakdown of survivors. Yes column: 42.1% 1st, 27.1% 2nd, 30.8% 3rd. Of the people who survived, 42.1% were 1st-class passengers.

### `normalize="total"` — grand-total

Each cell as a fraction of the grand total. Cells sum to 100.

Question answered: "What share of the entire sample is in this cell?"

Example: `pivot(titanic, rows="pclass", cols="survived", normalize="total")` shows joint distribution percentages. 1st class × Yes ≈ 17.2% of the sample.

### `normalize=None`

Raw counts (or values). Use when reporting absolute frequencies.

### Which one to use

Match the conditional to the question:
- "Survival rate by class" → `row` (class as `rows`).
- "Class composition among survivors" → `column` (survival as `cols`).
- "Joint share of each (class, survival) combination" → `total`.
- "Absolute counts" → `None`.

State explicitly which conditional you're reporting. The numbers are different and the interpretation is different.

## 6. Totals and how they interact with normalization

`totals=True` appends a Total row and (for crosstabs) a Total column. The row-variable column is cast to `Utf8` to allow the `"Total"` string label.

When combined with normalization:

- `normalize="row"` + `totals=True` → row Total column = 100 (every row, including the Total row); the Total row shows overall column proportions.
- `normalize="column"` + `totals=True` → column Total row = 100 (every column); the Total column shows overall row proportions.
- `normalize="total"` + `totals=True` → grand total = 100; both Total row and Total column sum to 100.

In general the totals are computed *before* normalization where it makes sense, then renormalized for display. The displayed totals always sum to 100% on the relevant axis.

## 7. Plain-English interpretation templates

### Frequency table

> The frequency table shows the count of each level of `<var>`. `<level_1>`: `<count_1>` (`<pct_1>%`), `<level_2>`: `<count_2>` (`<pct_2>%`), …

### Crosstab of counts

> The crosstab shows the joint count of `<var1>` and `<var2>`. The grand total is `<n>`. `<Largest cell>` is the (`<level_a>`, `<level_b>`) combination with `<count>` observations.

### Crosstab of row percentages

> Within each `<var1>` level, the percentages show the conditional distribution of `<var2>`. For `<level_a>` of `<var1>`: `<pct_b_given_a>%` are `<level_b>` of `<var2>`. The conditioning is on `<var1>`, so each row sums to 100%.

### Crosstab of values (means / medians / etc.)

> The crosstab shows the `<agg>` of `<values>` for each (`<var1>`, `<var2>`) combination. The largest cell is `(<level_a>, <level_b>)` at `<value>`; the smallest is `(<level_c>, <level_d>)` at `<value>`. Note: with `agg="<agg>"` we are summarizing each cell by its `<central tendency interpretation>`. For skewed data consider `agg="median"` for a more robust summary.

## 8. Extending the output with polars (pivot → unpivot → visualize)

`pivot` returns a wide-format `pl.DataFrame`. Common next steps:

### Sort

```python
result.sort("Yes", descending=True)
```

### Filter rows

```python
result.filter(pl.col("cut") != "Total")
```

### Add a derived column

```python
result.with_columns(
    survival_diff=pl.col("Yes") - pl.col("No")   # if cols=survived
)
```

### Unpivot back to long form for plotting

```python
long = rsm.eda.unpivot(
    result.filter(pl.col("cut") != "Total"),
    id_vars=["cut"],
    variable_name="color",
    value_name="median_price",
)
rsm.eda.visualize(long, x="cut", y="median_price", color="color", geom="bar")
```

This pivot → unpivot → visualize is the canonical pipeline for going from raw long data to a grouped bar chart. Each step does one thing.

### Round for presentation

```python
result.with_columns(pl.col(pl.Float64).round(2))
```

### Join with another table

```python
result.join(
    other_df, on="cut", how="left"
)
```

## 9. Pivot vs cross_tabs

`pivot(rows, cols, totals=True)` and `cross_tabs(var1, var2)` produce the same observed-count table for the same inputs. The difference is in scope:

| Tool | Observed counts | Expected counts | Chi-squared statistic | Standardized deviations | Plot |
| --- | --- | --- | --- | --- | --- |
| `pivot` | ✓ | ✗ | ✗ | ✗ | ✗ |
| `cross_tabs` | ✓ | ✓ | ✓ | ✓ | ✓ |

Use `pivot` when:
- You want a descriptive table only (counts, percentages, mean/median of `values`).
- You're not testing a hypothesis.
- You'll feed the output into a polars pipeline (sort, filter, plot, model).

Use `cross_tabs` (`pyrsm-cross-tabs` skill) when:
- You want a chi-squared test of independence with assumption checks.
- You want to identify which cells drive the association via standardized residuals.

You can also use both: `pivot` to produce a custom view (e.g., medians by group), and `cross_tabs` separately for the hypothesis-test view.

## 10. Worked examples (`diamonds`, `titanic`)

### Diamonds — frequency table of cut

```python
import polars as pl
import pyrsm as rsm

diamonds = pl.read_parquet("<abs-path>/diamonds.parquet")
rsm.eda.pivot(diamonds, rows="cut", normalize="total", totals=True)
```

Output:
- Ideal: 1176 (39.2%), Premium: 771 (25.7%), Very Good: 677 (22.6%), Good: 275 (9.2%), Fair: 101 (3.4%), Total: 3000 (100%).

### Diamonds — count crosstab of cut × color with totals

```python
rsm.eda.pivot(diamonds, rows="cut", cols="color", totals=True)
```

Output is a 6×9 table (5 cuts + Total row, 7 colors + cut column + Total column).

### Diamonds — mean price by cut × color

```python
rsm.eda.pivot(diamonds, rows="cut", cols="color", values="price", agg="mean")
```

For Fair-cut, J-color: mean price ~$6,102. For Ideal-cut, D-color: ~$2,667. Note the *counter-intuitive* pattern that better cuts (Ideal, Premium) have lower mean prices for the same color — driven by smaller carats among the better-cut diamonds.

For the same view with `agg="median"`:

```python
rsm.eda.pivot(diamonds, rows="cut", cols="color", values="carat", agg="median")
```

This is the most-informative one-look at the cut-color-carat structure of the data.

### Titanic — survival rates by passenger class

```python
titanic = pl.read_parquet("<abs-path>/titanic.parquet")
rsm.eda.pivot(titanic, rows="pclass", cols="survived", normalize="row", totals=True)
```

Output:
- 1st: 36.5% No, 63.5% Yes (Total 100%)
- 2nd: 55.9% No, 44.1% Yes
- 3rd: 73.8% No, 26.2% Yes
- Total: 59.3% No, 40.7% Yes

This is the cleanest descriptive view of "survival rate by class" — a strong monotonic relationship. For a formal test, see `pyrsm-cross-tabs` or `pyrsm-compare-props`.

### Titanic — sex × survival

```python
rsm.eda.pivot(titanic, rows="sex", cols="survived", normalize="row", totals=True)
```

- Female: 24.9% No, 75.1% Yes
- Male: 79.5% No, 20.5% Yes
- Total: 59.3% No, 40.7% Yes

A massive gradient by sex.

### Titanic — mean fare by class and survival

```python
rsm.eda.pivot(titanic, rows="pclass", cols="survived", values="fare", agg="mean")
```

- 1st: Yes $102.5, No $74.7
- 2nd: Yes $23.2, No $20.8
- 3rd: Yes $12.4, No $13.0

Among 1st-class passengers, survivors paid more on average than non-survivors — possibly because higher-fare passengers had better cabin locations. 3rd-class shows the opposite (very small difference).

### Custom percentile aggregation

```python
rsm.eda.pivot(diamonds, rows="cut", values="price", agg="p90")
# 90th-percentile price by cut
```

Useful for setting per-group cutoffs or stratified pricing thresholds.

## 11. Common pitfalls

- **Reporting a row percentage as if it were a column percentage** (or vice versa). State the conditional explicitly. "73.8% of 3rd-class died" ≠ "of those who died, 59.7% were in 3rd class".
- **Forgetting `totals=True` for the count check.** Reading a crosstab without the marginal totals makes it hard to spot small-sample cells.
- **Using `mean` for a skewed `values` column.** Prices, sales, incomes, and counts are typically right-skewed. Prefer `median` (or `agg="p50"`, equivalent).
- **Forgetting `fill=0` when expecting zero cells.** Default for a missing cell in a count crosstab is `null`, which can break downstream arithmetic. Pass `fill=0` to make them 0s.
- **Treating `pivot` as a hypothesis test.** It's descriptive. For a test, use `pyrsm-cross-tabs`.
- **Passing `agg="count"` along with `values=`.** `count` ignores `values`. If you want "count of non-null `<values>`", use `n_obs` (alias) — or `n_distinct` for unique values.
- **Mixing absolute and percentage cells in one writeup.** Decide once whether you're reporting counts or percentages and stick with it.
- **Comparing percentages across very-different-size groups.** A row percentage based on n=10 is much noisier than one based on n=1000. Pair percentages with sample sizes.
- **Forgetting to chain `.sort()` for a stable display.** Polars doesn't guarantee row order in groupby; pivot inherits this. Sort by the row variable for reproducible printed output.
