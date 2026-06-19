# pyrsm.eda.distr — reference

This file is the deeper reference for `pyrsm.eda.distr`. The main `SKILL.md` walks the workflow at a high level; come here for API details, the column-type classification heuristic, plotnine layering examples, and worked walkthroughs.

## Table of contents

1. Class signature
2. The column-type classification (and the `nint` parameter)
3. `summary()` — what each block prints
4. `plot()` — auto-pick of histograms vs bar charts
5. Output attributes
6. Plain-English interpretation templates
7. Extending the plot with plotnine
8. Extending with polars via `d.data` and the classification attributes
9. Related EDA / basics tools — when to switch
10. Worked example (`diamonds`)
11. Common pitfalls

---

## 1. Class signature

```python
rsm.eda.distr(
    data,                       # pl.DataFrame, pd.DataFrame, or dict-like
    cols=None,                  # list[str] | None — columns to analyze
    by=None,                    # str | None — group-by column for stratified stats
    name="Not provided",        # str — for display
    nint=25,                    # int — threshold for int → categorical classification
)
```

`distr` is a **class**, not a function. The constructor classifies the columns; you call `summary()` to print statistics and `plot()` to get a plotnine ggplot.

`data` accepts polars DataFrames, pandas DataFrames (auto-converted), or anything `pl.DataFrame(data)` can ingest.

`cols=None` defaults to all columns except `by`.

`by=None` produces an ungrouped summary. With `by` set, numeric statistics are computed per group, and categorical frequency tables are shown within each group.

`name` is purely for display (the summary header prints `Data: <name>`).

`nint` is the **column-classification threshold** — see §2.

## 2. The column-type classification (and the `nint` parameter)

`distr` classifies each column into one of three buckets:

- **numeric** — floats and integers with `n_unique ≥ nint`. Get histograms + central-tendency statistics.
- **categorical** — strings, Categoricals, Enums, AND integers with `n_unique < nint`. Get frequency tables + bar plots.
- **other** — everything else (dates, datetimes, durations, lists, structs). Get min/max/n_unique/n_missing only; no plot.

### The `nint` heuristic

The motivation: an `Int64` column with 5 unique values is "really" a categorical (e.g., a 1-5 rating scale). Treating it as numeric would compute a mean rating, which is occasionally useful but obscures the distribution. By default (`nint=25`), low-cardinality integers become categoricals.

`nint=25` is generous: a categorical variable that uses integer codes for 20 levels still classifies as categorical. Below 25 is "probably ordinal".

When to change `nint`:
- **Increase** (e.g., `nint=100`) for datasets where low-cardinality integers are genuinely continuous counts (e.g., "number of dependents", which is 0–10 but still useful as numeric).
- **Decrease** (e.g., `nint=10`) for datasets where high-cardinality categoricals shouldn't be treated as categorical (rare).
- **Set to 0** to disable the heuristic entirely — all numeric dtypes become numeric, ignoring n_unique.

### How dtype maps

| polars dtype | Goes to |
| --- | --- |
| `String`, `Utf8`, `Categorical`, `Enum` | always `categorical` |
| `Int*`, `UInt*` (small) | `categorical` if `n_unique < nint`, else `numeric` |
| `Float*` | always `numeric` |
| `Bool` | `other` (not numeric, not categorical in this classification) |
| `Date`, `Datetime`, `Duration`, `List`, `Struct`, `Object` | `other` |

## 3. `summary()` — what each block prints

```python
d.summary(
    dec=3,           # decimal places for floats
    plain=True,      # plain text vs styled great_tables (Jupyter)
)
```

Output structure:

- **Header**:
  - `Distribution Analysis`
  - `Data: <name>`
  - `Group by: <by>` (only if `by` is set)
  - `Columns : <N> (<n_numeric> numeric, <n_categorical> categorical, <n_other> other)`

- **Numeric Variables section** (if any). One row per numeric column with `variable`, `mean`, `median`, `min`, `max`, `sd` (default `explore` aggregations). When `by` is set, the table is grouped — one row per (group × variable).

- **Categorical Variables section** (if any). One block per categorical column with:
  - Header line: `<col> (n_unique: <k>, mode: <mode>, n_missing: <m>)`.
  - Frequency table with `<col>`, `count`, `proportion`.
  - When `by` is set, frequency tables are computed within each group of `by`.

- **Other Variables section** (if any). One row per "other" column with `variable`, `type`, `n_unique`, `n_missing`, `min`, `max`. min/max are string-cast (for dates this gives "2012-02-26", for datetimes the full timestamp, etc.).

Internally, the numeric table is built by calling `pyrsm.eda.explore` on `self.numeric_cols`, so the agg set is `explore`'s default (`["mean", "median", "min", "max", "sd"]`).

`plain=False` switches to styled `great_tables` output in Jupyter — same content, prettier rendering.

## 4. `plot()` — auto-pick of histograms vs bar charts

```python
d.plot(
    cols=None,       # list[str] | None — defaults to numeric_cols + categorical_cols
    bins=25,         # int — histogram bins for numeric columns
    ncol=2,          # int — number of plots per row in the composition
)
```

Returns a `plotnine.ggplot` (single column) or a `plotnine.composition.Compose` (multiple columns).

For each column:

- **Categorical** → `ggplot + geom_bar + theme_bw`. Title is the column name. X-axis labels rotated 45° (helps with long category names).
- **Numeric** → `ggplot + geom_histogram(bins=bins) + theme_bw`. Title is the column name.

Both use `fill="slateblue"` with `alpha=0.8` and a 10-pt bold title.

Multiple plots are composed into a grid via `|` (horizontal) and `/` (vertical) plotnine composition operators. The grid is `ncol` wide; the auto-figure-size is `4 × min(ncol, len(plots))` wide by `3 × ceil(len/ncol)` tall.

When the underlying column is a pyrsm Enum, the bar plot uses `scale_x_discrete(limits=cat_order)` to preserve the Enum's category order (rather than alphabetical).

## 5. Output attributes

After the constructor runs:

| Attribute | Type | Meaning |
| --- | --- | --- |
| `d.data` | `pl.DataFrame` | The underlying DataFrame (after any pandas → polars conversion). |
| `d.name` | str | Dataset display name. |
| `d.by` | str \| None | The group-by column (if set). |
| `d.cols` | list[str] | All columns being analyzed (cols arg, or all-non-by columns). |
| `d.nint` | int | The int-cardinality threshold used for classification. |
| `d.numeric_cols` | list[str] | Columns classified as numeric. |
| `d.categorical_cols` | list[str] | Columns classified as categorical. |
| `d.other_cols` | list[str] | Columns classified as other. |

The class doesn't store the computed summary statistics — those are recomputed every time `summary()` is called. For programmatic access, call `pyrsm.eda.explore(d.data, cols=d.numeric_cols, by=d.by)` directly.

## 6. Plain-English interpretation templates

### Header

> Distribution analysis of `<n>` columns of `<dataset name>`: `<n_numeric>` numeric, `<n_categorical>` categorical, `<n_other>` other (date / datetime / etc.). `<If 'by' set: stratified by <by>.>`

### Numeric variable

> `<var>` is numeric with mean `<mean>` and median `<median>` `<unit>`. The mean is `<higher | lower | roughly equal to>` the median by `<gap>` `<unit>`, suggesting `<right-skewed | left-skewed | symmetric>` distribution. The standard deviation is `<sd>` `<unit>`, ranging from `<min>` to `<max>`. For modeling, consider `<log-transform if right-skewed | nothing if symmetric>`.

### Categorical variable

> `<var>` has `<k>` unique levels with mode `<mode>` (`<mode_count>` observations, `<mode_pct>%` of the sample). `<m>` rows have missing values. `<If imbalanced: 'The largest level holds <max_level_pct>% of the data; the smallest level has <min_level_count> observations'.>` `<Suggest collapsing tiny levels for downstream use.>`

### Other variable

> `<var>` is of type `<type>`, with `<n_unique>` unique values ranging from `<min>` to `<max>`. `<m>` rows are missing. `<If a date column: the range is <max - min> wide.>`

### Cross-cutting flags

> Data quality issues to address before modeling: `<list>`. For example: '`age` has 20% missing — consider imputation or exclusion. `price` is heavily right-skewed — consider a log transform. `cabin` has 950 unique levels — too sparse for direct use as a predictor.'

## 7. Extending the plot with plotnine

`d.plot()` returns a plotnine object. **This is the heart of `distr`'s extensibility** — anything plotnine supports can be added to the plot.

### Add a theme

```python
from plotnine import theme_minimal, theme_classic

p = d.plot()
p_minimal = p + theme_minimal()
p_classic = p + theme_classic()
```

### Add an overall title or axis labels

```python
from plotnine import labs

p_titled = p + labs(title="Diamonds — distribution of all variables")
```

### Change scales (e.g., log)

```python
from plotnine import scale_x_log10, scale_y_log10

p_single = d.plot(cols=["price"])
p_log = p_single + scale_x_log10() + labs(title="Diamond Prices (log scale)")
```

### Add reference lines

```python
from plotnine import geom_vline

p_single = d.plot(cols=["price"])
p_with_lines = (
    p_single
    + geom_vline(xintercept=df["price"].mean(), color="red", linetype="dashed")
    + geom_vline(xintercept=df["price"].median(), color="blue", linetype="dotted")
    + labs(title="Diamond Prices — mean (red) and median (blue)")
)
```

### Layer a density on a histogram

```python
from plotnine import aes, ggplot, geom_histogram, geom_density, after_stat

# distr.plot internal geom_histogram only — to add a density, build the plot from scratch
import polars as pl
data = df.select(pl.col("price").cast(pl.Float64).alias("value"))
p = (
    ggplot(data, aes(x="value"))
    + geom_histogram(aes(y=after_stat("density")), bins=25, fill="slateblue", alpha=0.8)
    + geom_density(color="red", size=1)
    + labs(title="Density overlaid on histogram")
)
```

(For more elaborate single-column plots, switch to `pyrsm-visualize` or build directly with plotnine — `distr.plot()` is for the multi-column overview.)

### Custom facet by a column

```python
# distr.plot doesn't natively support faceting (because it composes per-column),
# but you can call plotnine directly for a single-column faceted plot:
from plotnine import ggplot, aes, geom_histogram, facet_wrap

p_faceted = (
    ggplot(df.to_pandas(), aes(x="price"))
    + geom_histogram(bins=25, fill="slateblue", alpha=0.8)
    + facet_wrap("~cut")
)
```

### Save to file

```python
p = d.plot()
p.save("distr_plot.png", width=12, height=8, dpi=150)
```

## 8. Extending with polars via `d.data` and the classification attributes

The classification stored on the `distr` object is useful metadata for downstream work.

### Build a regression spec from the numeric columns

```python
import pyrsm as rsm

predictors = [c for c in d.numeric_cols if c != "price"]
reg = rsm.model.regress({"diamonds": d.data}, rvar="price", evar=predictors)
reg.summary()
```

### Correlation matrix on numeric columns only

```python
cr = rsm.basics.correlation(d.data, vars=d.numeric_cols)
cr.summary()
```

### Filter to clean rows on key columns

```python
clean = d.data.filter(
    pl.col("price").is_not_null() & pl.col("carat").is_not_null()
)
```

### Re-run distr after a transformation

```python
transformed = d.data.with_columns(price_log=pl.col("price").log())
rsm.eda.distr(transformed, cols=["price", "price_log"]).plot()
```

The before/after view makes the transform's effect concrete.

### Group-aware exploration

```python
# Compare numeric stats across cut levels
rsm.eda.distr(d.data, by="cut", cols=["price", "carat"]).summary()
```

## 9. Related EDA / basics tools — when to switch

- **Numeric stats only, custom aggregations** → `pyrsm-explore`.
- **Cross-tabulation of two variables** → `pyrsm-pivot` or `pyrsm-cross-tabs`.
- **Single specific plot (scatter, line, box, density)** → `pyrsm-visualize`.
- **Joining two DataFrames** → `pyrsm-combine`.
- **Reshape wide ↔ long** → `pyrsm-pivot` / `pyrsm-unpivot`.
- **Hypothesis test on a mean / proportion** → `pyrsm-single-mean` / `pyrsm-single-prop`.
- **Correlation matrix with significance** → `pyrsm-correlation`.

## 10. Worked example — `diamonds`

```python
import polars as pl
import pyrsm as rsm

diamonds = pl.read_parquet("<abs-path>/diamonds.parquet")
d = rsm.eda.distr(diamonds, name="diamonds")
d.summary()
```

Output (paraphrased):

```
Distribution Analysis
Data    : diamonds
Columns : 11 (7 numeric, 3 categorical, 1 other)

--- Numeric Variables ---
variable | mean    | median | min   | max     | sd
price    | 3907.19 | 2407.0 | 338.0 | 18791.0 | 3956.9   (heavily right-skewed: mean ≫ median)
carat    | 0.794   | 0.7    | 0.2   | 3.0     | 0.474    (right-skewed but milder)
depth    | 61.75   | 61.9   | 54.2  | 70.8    | 1.45     (symmetric, tight)
table    | 57.47   | 57.0   | 50.0  | 69.0    | 2.24     (mostly symmetric)
x        | 5.72    | 5.71   | 3.73  | 9.42    | 1.12     (symmetric)
y        | 5.72    | 5.72   | 3.71  | 9.26    | 1.11     (symmetric)
z        | 3.53    | 3.52   | 2.33  | 5.58    | 0.69     (symmetric)

--- Categorical Variables ---
clarity (n_unique: 8, mode: SI1): SI1 (24%), VS2 (22%), SI2 (18%), VS1 (15%), VVS2 (10%), VVS1 (7%), IF (3%), I1 (1%)
cut (n_unique: 5, mode: Ideal): Ideal (39%), Premium (26%), Very Good (23%), Good (9%), Fair (3%)
color (n_unique: 7, mode: G): G (20%), F (19%), E (18%), H (15%), D (13%), I (9%), J (5%)

--- Other Variables ---
date  Date  n_unique=30  range: 2012-02-26 to 2015-12-01
```

### Interpretation

1. **Numeric:** Price is heavily right-skewed (mean $3,907 vs median $2,407). Carat is also right-skewed. Depth / table / x / y / z are tight, symmetric (the physical-dimension variables).
2. **Categorical:** All three categoricals have well-populated levels — no tiny-level issues. Cut is Ideal-heavy. The clarity distribution has the right asymmetry (`I1` is the rarest, `IF` is rare, `SI1` is the mode — consistent with what we'd expect from a retail diamond inventory).
3. **Other:** `date` covers 4 years (2012–2015) with 30 distinct dates — likely monthly snapshots.

### Plot

```python
d.plot()
```

Returns a composition of 10 plots (7 histograms + 3 bar charts), arranged 2-wide. Price histogram clearly shows the right skew; cut bar chart shows the Ideal-heavy distribution.

### Extending the plot

```python
from plotnine import scale_x_log10, labs

# Just the price histogram, log-scale
p = d.plot(cols=["price"]) + scale_x_log10() + labs(title="Diamond Prices (log scale)")
```

Log-transforming makes the distribution much closer to symmetric — a hint that `log(price)` is a more natural modeling target than `price`.

### Stratified by cut

```python
rsm.eda.distr(diamonds, by="cut", cols=["price", "carat"]).summary()
```

Output: numeric stats now per-cut, revealing that Ideal-cut diamonds have lower mean price *and* lower mean carat than Premium — the cut–carat confounding pattern.

## 11. Common pitfalls

- **Forgetting `nint` controls the int → categorical threshold.** An `Int64` column with 8 unique values is classified categorical by default. Increase `nint` to keep it numeric (e.g., for a "number of bedrooms" 0–8 column).
- **Treating mean as "typical" for skewed variables.** Always compare to median. For right-skewed business variables (price, sales, income), the median is the more meaningful "typical" value.
- **Ignoring low-count categorical levels.** A 12-level categorical with most rows in 2 levels needs to be collapsed or dropped for modeling.
- **Treating high-`n_unique` categoricals as informative.** An ID column will have n_unique ≈ n_rows; useless as a predictor.
- **Calling `summary()` and expecting a return.** It prints; it doesn't return. Capture stdout or query attributes directly.
- **Calling `plot()` in a non-notebook environment.** The return value is a `plotnine.ggplot` — print it (`print(p)`) or save it (`p.save(...)`).
- **Not extending the plot with plotnine.** `distr.plot()` is the default look; customizing with scales, themes, and reference lines is the natural next step.
- **Building one regression per numeric column.** `d.numeric_cols` is convenient metadata, but use it to drive one principled regression specification, not to iterate naively.
- **Re-running distr to extract the numeric stats programmatically.** Call `pyrsm.eda.explore(d.data, cols=d.numeric_cols, by=d.by)` directly — same data, returned as a polars DataFrame for chaining.
- **Expecting Bool columns to be classified numeric or categorical.** They go to `other`. Cast to Int or String first if you want a different bucket.
