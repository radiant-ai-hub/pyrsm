# pyrsm.eda.visualize — reference

This file is the deeper reference for `pyrsm.eda.visualize`. The main `SKILL.md` walks the workflow at a high level; come here for the full parameter list, per-geom details, and worked plotnine-extension examples.

## Table of contents

1. Function signature
2. Multiple x/y variables
3. The 8 geoms in detail
4. Aesthetics — column mapping vs literal values
5. Faceting (`facet`, `facet_row`, `facet_col`)
6. Aggregation (`agg`) and smoothing (`smooth`)
7. The `nobs` sample cap for scatter plots
8. The categorical-vs-numeric auto-detection
9. Plain-English interpretation templates
10. Extending the plot with plotnine
11. Preprocessing the data with polars
12. Worked examples
13. Common pitfalls

---

## 1. Function signature

```python
rsm.eda.visualize(
    df,                       # pl.DataFrame or pl.LazyFrame
    x,                        # str | list[str] — column(s) for x-axis (required)
    y=None,                   # str | list[str] | None — y-axis column(s)
    geom=None,                # str | None — see §2
    color="slateblue",        # str | None — column for color aesthetic OR a literal color
    fill=None,                # str | None — column for fill OR literal color
    shape=None,               # str | None — column for shape aesthetic
    group=None,               # str | None — column for grouping (no separate color)
    linetype=None,            # str | None — column for linetype aesthetic
    bins=None,                # int | None — histogram bins (default 30)
    alpha=None,               # float | None
    size=None,                # int | float | None
    position=None,            # str | None — "stack" or "dodge" for bar
    smooth=None,              # str | None — "lm", "loess", or True
    jitter=False,             # bool — add jitter for scatter on discrete x
    facet=None,               # str | None — facet_wrap by this column
    facet_row=None,           # str | None — facet_grid row variable
    facet_col=None,           # str | None — facet_grid column variable
    title=None,               # str | None
    nobs=1000,                # int — sample cap for scatter (-1 for all)
    agg=None,                 # str | None — "mean", "median", "sum", "min", "max"
    ncol=2,                   # int — columns in the composed grid for multiple plots
    ret="compose",            # "compose" or "list"
) -> plotnine.ggplot | plotnine composition | list[plotnine.ggplot]
```

Returns a `plotnine.ggplot` object for one plot. If multiple plots are generated, returns a plotnine composition by default. Use `ret="list"` to get the individual ggplot objects.

If `geom` is `None`, the default depends on whether `y` is set:
- `y` set → `geom="scatter"`.
- `y` not set → `geom="dist"`.

Unknown `geom` or `agg` values raise `ValueError`.

## 2. Multiple x/y variables

`x` and `y` can each be a string or a list/tuple of strings.

For one-variable geoms (`dist`, `hist`, `density`, or `bar` without `y`), multiple `x` values create one plot per x variable:

```python
rsm.eda.visualize(df, x=["price", "carat", "depth"], geom="hist", ncol=2)
```

For two-variable geoms, pyrsm creates one plot for every `x` × `y` pair:

```python
rsm.eda.visualize(
    df,
    x=["carat", "depth"],
    y=["price", "quantity"],
    geom="scatter",
    ncol=2,
)
# carat vs price, carat vs quantity, depth vs price, depth vs quantity
```

By default these plots are composed into a grid using plotnine composition. To customize individual panels before composing, request a list:

```python
plots = rsm.eda.visualize(df, x=["price", "carat"], geom="hist", ret="list")
plots[0] + labs(title="Price distribution")
```

## 3. The 8 geoms in detail

```python
GEOM_CONFIG = {
    "dist":    {"required": ["x"],      "defaults": {"bins": 30, "fill": "slateblue", "alpha": 0.7}},
    "hist":    {"required": ["x"],      "defaults": {"bins": 30, "fill": "slateblue", "alpha": 0.7}},
    "density": {"required": ["x"],      "defaults": {"fill": "slateblue", "alpha": 0.5}},
    "scatter": {"required": ["x", "y"], "defaults": {"alpha": 0.7, "size": 2, "nobs": 1000}},
    "bar":     {"required": ["x"],      "defaults": {"fill": "slateblue", "alpha": 0.8}},
    "line":    {"required": ["x", "y"], "defaults": {"size": 1}},
    "box":     {"required": ["x", "y"], "defaults": {"fill": "slateblue", "alpha": 0.7}},
    "violin":  {"required": ["x", "y"], "defaults": {"fill": "slateblue", "alpha": 0.7}},
}
```

### `dist` / `hist` (aliases)

Auto-switches between histogram (numeric x) and bar chart (categorical x, i.e., string / Categorical / Enum dtype or an integer column with `n_unique <= 20`).

For numeric x: `geom_histogram(bins=30)`.
For categorical x: `geom_bar()`.

The split is convenient — passing `geom="dist"` gives you "the right one-variable plot" without picking. Use `geom="hist"` or `geom="bar"` if you want to be explicit.

### `density`

Smoothed kernel density estimate. `geom_density()` with `fill="slateblue"`, `alpha=0.5`.

Useful overlay if you have a `color` mapping to compare distributions across groups (each group's density is colored separately, with translucent fills).

### `scatter`

`geom_point()`. Use the `jitter=True` flag to switch to `geom_jitter` (useful when x is discrete and overlapping). Use `smooth="lm"` or `"loess"` to add a `geom_smooth()` fit.

For categorical x with `agg=` set, adds a `stat_summary` crossbar at the agg per category (e.g., a horizontal mean line per group).

### `bar`

`geom_bar()` with `stat="count"` when `y` is omitted (counts per x level). When `y` is set, pyrsm first groups the data with Polars and computes the selected aggregation per x level, then plots the aggregated values as bar heights. If `agg` is omitted, the default aggregation is `mean`.

`position` controls grouping: `"stack"` (default) or `"dodge"`.

### `line`

`geom_line()`. Auto-grouping behavior: if `color=` is mapped to a column but `group=` is not specified, the function adds `group=color` so that each color value gets a continuous line. Otherwise lines may zigzag between groups in unintended ways.

### `box`

`geom_boxplot()`. Requires categorical (or grouping) x and numeric y. Shows median (center line), IQR (box), 1.5× IQR whiskers (lines), and outlier points beyond the whiskers.

### `violin`

`geom_violin()`. Like box but shows the kernel-density envelope on either side. Useful when you want the full distribution shape per group, not just the summary statistics.

## 4. Aesthetics — column mapping vs literal values

The aesthetics `color`, `fill`, `shape`, `group`, `linetype` can each be:

1. **A column name** (string that exists in `df.columns`) → mapped aesthetic. Each unique value gets a different color / fill / shape / linetype.
2. **A literal value** (string not in `df.columns`, like `"red"` or `"slateblue"`) → fixed visual property. All points / lines get the same color/fill.

The detection rule is straightforward: `if value in df.columns`.

### `color` vs `fill`

- `color` is the *line* color (outlines of bars, line geom, point edges).
- `fill` is the *interior* color (bar interior, density fill, box interior).

For solid points and lines, `color` is usually what you want. For filled shapes (bars, density, violin, box), set `fill` for the interior and let `color` default.

### `group`

Use when you want to draw separate lines/elements per group without coloring them differently — useful for "spaghetti plot" style line charts where you want all lines in one color but distinct lines per group.

### `shape`

Maps a categorical to point shapes (only applies to scatter). Use sparingly: more than ~5 shapes becomes confusing.

### `linetype`

Maps to line styles (solid, dashed, dotted, etc.). Useful for distinguishing series when you can't use color.

## 5. Faceting (`facet`, `facet_row`, `facet_col`)

Faceting splits the plot into multiple sub-plots based on a categorical variable.

### `facet`

Equivalent to `plotnine.facet_wrap("~<col>")`. One panel per level of `<col>`, arranged in a grid (auto-shaped).

```python
rsm.eda.visualize(df, x="carat", y="price", geom="scatter", facet="cut")
# One panel per cut level
```

### `facet_row` and `facet_col`

Equivalent to `plotnine.facet_grid("<row>~<col>")`. Two-dimensional grid with `facet_row` on the y-direction and `facet_col` on the x-direction.

```python
rsm.eda.visualize(df, x="carat", y="price", geom="scatter",
                  facet_row="cut", facet_col="color")
# 5 cuts × 7 colors = 35 panels
```

Pass `facet_row="."` (with just `facet_col`) for column-only grids, or `facet_col="."` (with just `facet_row`) for row-only.

`facet` takes precedence over `facet_row`/`facet_col` if both are passed.

### When to facet vs color

- **Color**: groups overlay each other in the same panel. Use when you want direct comparison.
- **Facet**: each group gets its own panel. Use when overlay would be too cluttered (many groups, or distributions that don't fit on the same scale).

Rule of thumb: ≤ 6 groups → color; > 6 groups → facet.

## 6. Aggregation (`agg`) and smoothing (`smooth`)

### `agg`

Aggregates y by x. Available functions: `"mean"`, `"median"`, `"sum"`, `"min"`, `"max"`.

- For `geom="bar"` with `y=` set: each bar's height is the aggregation of y within that x level. Y-axis label becomes `"<Agg> of <y>"`.
- For `geom="scatter"` with categorical x: adds a horizontal crossbar at the aggregation per category.
- For other geoms: ignored.

### `smooth`

Adds a `geom_smooth()` to scatter plots:
- `"lm"` — linear regression line with 95% CI shading (`se=True`).
- `"loess"` — LOESS smoother with CI shading.
- `"true"` / `True` — default smoother (currently LOESS).

If a `color=` mapping is also set, the smoother is fit *per color group* (one smooth per group).

## 7. The `nobs` sample cap for scatter plots

For large datasets, plotting every scatter point hurts readability and rendering performance. `visualize` samples down to `nobs` points (default 1000) using `df.sample(n=nobs, seed=1234)` for reproducibility.

When the sample is taken:
- The plot adds a caption noting `nobs=<n> used`.
- The smoother (if any) is fit on the *sampled* data, not the full data. This is usually fine; for very large datasets where you want the smooth on all data, fit it manually and add via `+ geom_smooth(...)`.

Set `nobs=-1` to use all points (no sampling).

`nobs` only applies to `geom="scatter"`. Other geoms always use all data.

## 8. The categorical-vs-numeric auto-detection

The internal `_is_categorical(df, col)` function:

- Returns `True` if dtype is `String`, `Utf8`, `Categorical`, or `Enum`.
- Returns `True` if dtype is an integer type AND `n_unique <= 20`.
- Otherwise returns `False`.

This drives the `dist` / `hist` auto-switch (histogram for numeric x, bar for categorical x) and the `agg`-for-scatter behavior (crossbar added when categorical x).

The threshold is hardcoded at 20 (unlike `distr` where it's the `nint` parameter). If you have an integer column with 21-25 unique values that you want treated as categorical, cast to string first: `df.with_columns(pl.col("rating").cast(pl.Utf8))`.

## 9. Plain-English interpretation templates

### Spec announcement

> Building a `<geom>` plot of `<y>` (y-axis) vs `<x>` (x-axis)`<, colored by <color>>``<, faceted by <facet>>`. `<additional details: smooth, agg, nobs>`. The default theme is `theme_bw()`.

### Pattern callouts (by geom)

- **scatter**: "Look for the slope (positive / negative correlation), the spread (tight / loose), curvature, outliers, and color separation between groups."
- **histogram / density**: "Look at the skew (mean vs median), modality, and tails."
- **bar**: "Compare relative heights across categories. Watch for dominant levels and tiny levels."
- **line**: "Look at trend (rising / falling / flat), seasonality, breaks, and crossovers between groups."
- **box / violin**: "Compare medians (center line) across groups. The whiskers and dots flag outliers and skew."

### Cross-reference

> For a quantitative confirmation of this visual pattern, use `<pyrsm-correlation>` (scatter), `<pyrsm-distr>` (histogram), `<pyrsm-compare-means>` (box), `<pyrsm-pivot>` (bar), etc.

## 10. Extending the plot with plotnine

`visualize` returns a `plotnine.ggplot`. **Anything plotnine supports can be added.**

### Themes

```python
from plotnine import theme_minimal, theme_classic, theme_dark, theme_void, theme

p + theme_minimal()
p + theme_classic()
p + theme(figure_size=(10, 6), legend_position="bottom")
```

### Scales

```python
from plotnine import (
    scale_x_log10, scale_y_log10,
    scale_x_continuous, scale_y_continuous,
    scale_color_brewer, scale_color_manual, scale_color_gradient,
    scale_fill_brewer, scale_fill_manual,
    scale_x_date,
)

p + scale_x_log10()
p + scale_y_continuous(limits=(0, 20000), breaks=[0, 5000, 10000, 15000, 20000])
p + scale_color_brewer(type="qual", palette="Set2")
p + scale_color_manual(values={"Ideal": "blue", "Premium": "red", "Good": "green"})
```

### Additional geoms

```python
from plotnine import (
    geom_smooth, geom_vline, geom_hline, geom_abline,
    geom_rug, geom_text, geom_label, geom_density,
    annotate,
)

p + geom_smooth(method="loess", se=False, color="darkred", linetype="dashed")
p + geom_vline(xintercept=1.0, linetype="dashed", color="red")
p + geom_hline(yintercept=df["price"].median(), linetype="dotted", color="blue")
p + geom_rug()                  # marginal ticks
p + annotate("text", x=2.0, y=15000, label="High value cluster", color="black")
```

### Labels

```python
from plotnine import labs

p + labs(
    title="Diamond Prices vs Carat",
    subtitle="LM smoothing per cut group",
    caption="Source: diamonds (n=3,000)",
    x="Carat (weight)",
    y="Price ($)",
    color="Cut quality",
)
```

### Save

```python
p.save("plot.png", width=10, height=6, dpi=150)
p.save("plot.pdf", width=10, height=6)
p.save("plot.svg", width=10, height=6)
```

### Compose multiple plots

```python
# Built-in multi-variable composition
rsm.eda.visualize(df, x=["price", "carat", "depth"], geom="hist", ncol=2)
rsm.eda.visualize(df, x=["carat", "depth"], y=["price", "quantity"], geom="scatter")

# Raw plot list for per-panel customization
plots = rsm.eda.visualize(df, x=["price", "carat"], geom="hist", ret="list")

# Manual composition operators still work
p1 = rsm.eda.visualize(df, x="carat", y="price", geom="scatter")
p2 = rsm.eda.visualize(df, x="carat", geom="density")

# plotnine composition operators
composed = p1 | p2     # side by side
composed = p1 / p2     # stacked vertically
composed = (p1 | p2) / p3   # mixed
```

## 11. Preprocessing the data with polars

`visualize` doesn't transform data — it expects the right shape. Preprocess with polars first:

### Filter

```python
plot_data = df.filter(pl.col("cut").is_in(["Ideal", "Premium"]))
rsm.eda.visualize(plot_data, x="carat", y="price", color="cut", geom="scatter")
```

### Transform a column

```python
plot_data = df.with_columns(price_log=pl.col("price").log())
rsm.eda.visualize(plot_data, x="carat", y="price_log", geom="scatter", smooth="lm")
```

### Aggregate before plotting

```python
plot_data = df.group_by("date").agg(mean_price=pl.col("price").mean())
rsm.eda.visualize(plot_data, x="date", y="mean_price", geom="line")
```

### Pivot → unpivot → visualize

```python
# Build a pivot
wide = rsm.eda.pivot(df, rows="region", cols="quarter", values="sales", agg="sum")
# Unpivot for plotting
long = rsm.eda.unpivot(wide, on=["Q1","Q2","Q3","Q4"], id_vars="region",
                       variable_name="quarter", value_name="sales")
# Plot
rsm.eda.visualize(long, x="quarter", y="sales", color="region", geom="line")
```

## 12. Worked examples

### Histogram of a numeric column

```python
rsm.eda.visualize(df, x="price", geom="hist", bins=50, title="Diamond Prices")
```

### Histograms for several variables

```python
rsm.eda.visualize(df, x=["price", "carat", "depth"], geom="hist", bins=30, ncol=2)
```

### Scatter plots for several x/y pairs

```python
rsm.eda.visualize(
    df,
    x=["carat", "depth"],
    y=["price", "quantity"],
    geom="scatter",
    ncol=2,
)
```

### Density colored by cut

```python
rsm.eda.visualize(df, x="price", geom="density", color="cut", title="Price density by cut")
# Each cut gets a colored density curve
```

### Scatter with linear smooth, colored by cut

```python
rsm.eda.visualize(
    df, x="carat", y="price", geom="scatter",
    color="cut", smooth="lm", nobs=500,
    title="Price vs Carat (linear smooth per cut)",
)
```

### Scatter with categorical x and per-category mean

```python
rsm.eda.visualize(
    df, x="cut", y="price", geom="scatter",
    agg="median", jitter=True,
    title="Price distribution per cut (with median crossbar)",
)
```

### Bar chart of counts per cut

```python
rsm.eda.visualize(df, x="cut", geom="bar", title="Diamond count per cut")
```

### Bar chart with aggregated y (mean price per cut)

```python
rsm.eda.visualize(
    df, x="cut", y="price", geom="bar", agg="mean",
    title="Mean price per cut",
)
# y-axis label automatically becomes "Mean of price"
```

### Dodged bar chart with two groupings

```python
rsm.eda.visualize(
    df, x="cut", geom="bar", fill="color", position="dodge",
    title="Cut × Color counts (dodged)",
)
```

### Box plot by group

```python
rsm.eda.visualize(
    df, x="cut", y="price", geom="box",
    title="Price distribution by cut (boxplots)",
)
```

### Violin plot with color

```python
rsm.eda.visualize(
    df, x="cut", y="price", geom="violin", fill="cut",
    title="Price by cut (violin)",
)
```

### Line plot with color grouping

```python
plot_data = df.group_by(["date", "cut"]).agg(mean_price=pl.col("price").mean()).sort(["cut", "date"])
rsm.eda.visualize(
    plot_data, x="date", y="mean_price", geom="line", color="cut",
    title="Mean price over time, by cut",
)
# auto-groups by color so each cut gets a continuous line
```

### Faceted scatter

```python
rsm.eda.visualize(
    df, x="carat", y="price", geom="scatter", smooth="lm",
    facet="cut", title="Price vs Carat, faceted by cut",
)
```

### facet_grid (2D)

```python
rsm.eda.visualize(
    df, x="carat", y="price", geom="scatter",
    facet_row="cut", facet_col="color",
    title="Price vs Carat, grid by cut × color",
)
# Lots of small panels — useful for spotting cell-specific patterns
```

### Layered plotnine extension

```python
from plotnine import scale_y_log10, theme_minimal, labs, geom_smooth

p = rsm.eda.visualize(df, x="carat", y="price", geom="scatter", color="cut", nobs=500)

p_layered = (
    p
    + scale_y_log10()
    + geom_smooth(method="loess", se=False, color="black", linetype="dashed")
    + theme_minimal()
    + labs(
        title="Price vs Carat (log y)",
        subtitle="Linear smooth per cut (default) + overall LOESS (black dashed)",
        x="Carat (weight)",
        y="Price ($, log)",
        color="Cut quality",
    )
)
p_layered.save("price_vs_carat.png", width=10, height=6, dpi=150)
```

## 13. Common pitfalls

- **Wrong geom for the question.** `scatter` with two categoricals, `line` with unordered categorical x, `hist` of a string column (auto-switches to bar but is confusing). Pick the geom from the data types.
- **Passing a literal color where you meant a column mapping (or vice versa).** If `df` has a column called `"red"`, `color="red"` maps to that column, not the literal color red. Renaming or pre-checking helps.
- **High-cardinality color mapping.** A 50-level categorical mapped to `color` produces an unreadable legend. Use `facet` instead.
- **Scatter on large data with no `nobs`.** Defaults to 1000 sampled points. If you need all points, pass `nobs=-1` — but expect rendering to be slow.
- **Line plot on unordered x.** Sort by the x-axis variable first (with `polars` `sort`) or use `bar` instead.
- **Box plot with continuous x.** Polars / plotnine will try, and produce one box per unique x value, which is rarely useful. Bin the x first if you want grouped boxes (`df.with_columns(carat_bin=pl.col("carat").cut([0.5, 1.0, 1.5, 2.0]))`).
- **`agg=` only does anything for `bar` (with `y`) and `scatter` (with categorical x).** It's silently ignored otherwise.
- **Faceting AND coloring AND multiple smooths.** You can layer them, but the chart becomes hard to read. Pick ≤ 2 grouping dimensions.
- **Forgetting `position="dodge"` on a bar chart with `fill=`.** Default is `"stack"`, which stacks bars; `"dodge"` puts them side-by-side. For category × subcategory counts, "dodge" is usually clearer.
- **Saving without specifying dimensions.** `p.save("plot.png")` uses plotnine defaults which can be small. Pass `width=10, height=6, dpi=150` for a presentation-quality save.
- **Not extending with plotnine when the default isn't enough.** The whole point of `visualize` is that you keep the `ggplot` object and customize. Add scales, themes, reference lines, and additional geoms freely.
- **Expecting multiple x/y variables to overlay automatically.** Multiple variables create multiple panels by default. If you want overlayed series in one panel, reshape first with `pyrsm.eda.unpivot` or Polars `unpivot`, then map the variable-name column to `color` or `fill`.
