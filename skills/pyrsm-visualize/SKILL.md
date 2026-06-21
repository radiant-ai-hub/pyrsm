---
name: pyrsm-visualize
description: Create custom plots from a polars DataFrame in Python using the pyrsm library's `visualize` function — a plotnine wrapper supporting 8 geom types (dist / hist / density / scatter / bar / line / box / violin) with full control over aesthetics (color / fill / shape / group / linetype), faceting (facet_wrap / facet_grid), aggregation, smoothing (lm / loess), jitter, titles, and multiple x/y variables. The return value is a `plotnine.ggplot` object for one plot or a plotnine composition for multiple plots, and users can extend the result with plotnine layers — scales, themes, geom_smooth, geom_vline, custom labels — making plotnine extensibility a first-class part of this skill. Triggers include phrases like "scatter plot of x vs y", "histogram of price", "box plot of price by cut", "line chart over time, colored by region", "facet by category", "add a regression line", "use plotnine to plot", "density plot of carat by cut", or any request for a specific plot in a marketing/business analytics context.
---

# pyrsm visualize workflow

This skill walks the user through building plots with `pyrsm.eda.visualize` — from loading data to choosing the right geom, configuring aesthetics, adding aggregation or smoothing, faceting, and (critically) extending the returned `plotnine.ggplot` object with arbitrary plotnine layers.

For deep reference on all 8 geoms, the aesthetic mapping rules, the smooth / agg / jitter modifiers, faceting, and worked plotnine-extension examples, see `references/visualize.md`.

## Output policy

Two rules apply to **every** pyrsm call you make in this skill:

1. **Show the full output verbatim.** When you call `.summary()`, `.predict()`, or any pyrsm method that prints results, paste the **complete printed output** into your response — every line of the coefficient table, every statistic, every row. Do NOT abbreviate. Do NOT replace it with a high-level "main takeaways" bullet list. Do NOT skip the table to "save space". The user is reading your response because they want to see the actual numbers, not your summary of them.

2. **Provide the full pedagogical interpretation, not lame bullets.** When walking through results, use the detailed templates in `references/<name>.md`: state hypotheses explicitly, walk every coefficient or cell with its real-world units from the sidecar description, give the significance verdict at the 5% level, use the "holding constant" qualifier (where applicable), and tie back to the business question. A response like "X is positive and significant; Y is positive and significant; Z is not" is exactly the failure mode this skill is built to prevent.

3. **Lead with the requested output.** When the user asks for a specific plot, table, or diagnostic (e.g., "create and interpret PDP", "show me the residual dashboard", "generate PIP", "give me the confusion matrix", "plot the gains chart"), your interpretation must focus on THAT output — not on a re-interpretation of the model's coefficient table or summary. Pi's failure pattern is to say *"I'll generate X and then interpret the fitted results"* and then quietly produce a coefficient walk-through. The user already saw the coefficient interpretation in the previous turn; do not repeat it unless they explicitly ask. Lead with the requested output's content, using the dedicated template for that output if the skill has one (Step 9 for PIP, Step 10 for PDP, etc.).

If your context is genuinely constrained and you cannot reproduce the full output, say so explicitly — never silently abbreviate.


---

## Step 1 — Ask for the data location

Same as the other eda skills. Require an **absolute path**.

> Before we start, can you give me the **absolute path** to the data file you want to plot? For example: `/Users/yourname/Downloads/diamonds.parquet`. Any of `.parquet`, `.csv`, `.tsv`, `.feather`, `.arrow`, `.xlsx`, or `.json` will work.

If the data is already in memory from a previous step (e.g., the output of `pyrsm-pivot` or `pyrsm-unpivot`), accept it directly.

## Step 2 — Load the data into a polars DataFrame

Use `scripts/load_data.py`:

```bash
python scripts/load_data.py "<absolute-path-to-data>"
```

The dtype list matters here: numeric columns are candidates for the y-axis of a scatter / line / box / violin; categorical columns are candidates for color / fill / facet aesthetics or as the x of a bar chart.

### Sidecar description file

Read it. Knowing what each column means makes the difference between picking the right geom and picking a generic one.

## Step 3 — Decide what to plot

This is the **defining critical concept** of `visualize`: **choose the right geom for the question**.

| Question | Geom |
| --- | --- |
| "What's the distribution of `<numeric>`?" | `dist` or `hist` (histogram) |
| "What's the distribution of `<categorical>`?" | `dist` or `bar` (bar chart) |
| "Smoothed distribution shape?" | `density` |
| "Relationship between `<numeric_x>` and `<numeric_y>`?" | `scatter` |
| "How does `<numeric>` vary over time / ordered axis?" | `line` |
| "How does `<numeric_y>` differ across levels of `<categorical_x>`?" | `box` or `violin` |
| "Counts (or aggregated values) per level of `<categorical>`?" | `bar` |

Confirm the spec with the user. You need:

- **x** (required) — column or list of columns for x-axis. May be numeric or categorical.
- **y** (required for scatter / line / box / violin) — numeric column or list of columns for y-axis.
- **geom** — one of `dist`, `hist`, `density`, `scatter`, `bar`, `line`, `box`, `violin`. Default: `scatter` if `y` is provided, `dist` otherwise.
- **Aesthetics** (optional): `color` (line / point color or grouping), `fill` (fill color or grouping for filled shapes), `shape` (point shape), `group` (grouping without separate color), `linetype` (linetype grouping).
- **Modifiers** (geom-dependent):
  - `bins` (default 30) for histogram.
  - `alpha`, `size` — visual properties.
  - `position` — `"stack"` (default) or `"dodge"` for bar charts.
  - `smooth` — `"lm"` (linear regression line) or `"loess"` (LOESS smoother) — adds a fitted line to a scatter.
  - `jitter` — boolean; add jitter to scatter (useful for discrete x).
  - `agg` — `mean`, `median`, `sum`, `min`, `max` — for bar charts with `y`, aggregates y by x; for scatter with categorical x, adds a horizontal crossbar at the aggregated value per category.
- **Faceting**:
  - `facet` — `facet_wrap` by a column.
  - `facet_row` / `facet_col` — `facet_grid` by row / column.
- **Title** — optional.
- **nobs** — for scatter plots, cap the number of plotted points (default 1000). Set `-1` for all points; the data still uses all observations for computation, but only `nobs` are drawn to keep the chart legible.
- **ncol** — for multiple plots, number of columns in the composed grid.
- **ret** — `"compose"` returns a composed grid for multiple plots; `"list"` returns the individual ggplot objects.

State your proposed spec, e.g.: "I'll make a scatter of price vs carat, colored by cut, with a linear smooth and 500 sampled points for clarity."

Sanity checks:

- For `color=` / `fill=`: if the value is a column name, it becomes an aesthetic (mapped to data). If it's a CSS color literal (e.g., `"slateblue"`, `"red"`), it becomes a fixed color. The function checks `if value in df.columns` to decide.
- For high-cardinality color: avoid mapping a 100-level categorical to color — the legend becomes unreadable and the chart messy. Use facet instead.
- For scatter on large data: `nobs=1000` is the default sample cap; mention this when relevant.

## Step 4 — Run visualize

```python
import polars as pl
import pyrsm as rsm

df = pl.read_parquet("<absolute-path>")

p = rsm.eda.visualize(
    df,
    x="<x-column>",
    y="<y-column>",
    geom="scatter",        # or any of the 8
    color="<col-or-color>",
    facet="<col>",         # optional
    smooth="lm",           # optional smoother
    title="<title>",
    nobs=1000,             # scatter sample cap
)

# 3. Inspect (it's a plotnine ggplot, or a composition for multiple plots)
p   # in Jupyter; outside Jupyter, print(p) or p.save(...)
```

Notes:
- `visualize` returns a `plotnine.ggplot` object for one plot and a plotnine composition for multiple plots. In Jupyter it auto-renders; elsewhere use `print(p)` or `p.save("file.png", width=8, height=6)`.
- The default plot has `theme_bw()` applied. To switch themes, add `+ theme_minimal()` or `+ theme_classic()` to the returned ggplot.
- For aesthetic mappings, the value must be a column in `df`. For fixed visual properties (e.g., always-red points), pass the literal color string — `visualize` will detect that it's not a column name and use it as a fixed color.

## Step 5 — Interpret the plot

Most plot interpretation is visual — the user looks at the chart. Your job is to:

1. **Confirm what was plotted.** "This is a scatter of `price` (y, in USD) vs `carat` (x, in carats), colored by `cut`. A linear smoothing line is overlaid. 1,000 randomly-sampled points are shown out of 3,000 total."

2. **Highlight the patterns to look at.**
   - **For scatter**: direction (positive / negative), strength (tight / loose), shape (linear / curved / threshold), outliers, color separation between groups.
   - **For histogram / density**: skew, modality (unimodal / bimodal), heavy tails.
   - **For bar**: dominance of certain levels, gradient across an ordered axis.
   - **For box / violin**: median (center line), IQR (box), extreme values (whiskers and dots).
   - **For line**: trend, periodicity, breaks, gaps.

3. **Cross-reference with quantitative summaries.** A plot is useful but imprecise. Once you've seen the visual pattern, route the user to a quantitative skill: `pyrsm-correlation` for scatter, `pyrsm-distr` or `pyrsm-explore` for distributions, `pyrsm-pivot` for cross-tab summaries, `pyrsm-compare-means` for box-plot comparisons.

### Don't pick a geom that fights the question

Common geom-mismatches:

- **Histogram of a categorical**: `dist` and `hist` switch automatically to a bar chart for categorical x (n_unique < 20), but it's worth being explicit. For categoricals just call `geom="bar"`.
- **Scatter of two categoricals**: produces a grid of overlapping dots — almost always wrong. Use `geom="box"` or aggregate into a frequency table with `pyrsm-pivot`.
- **Line chart on unordered categorical x**: produces a zigzag through whatever the dataframe's row order is. Use bar instead, or sort first by the categorical.
- **Box plot with a continuous x**: forces polars to discretize the x; usually you wanted a scatter. Cast / bin x first if you really want a box per x-bin.

## Step 6 — Extend with plotnine (the critical concept)

`visualize` returns a `plotnine.ggplot` object. **You can add any plotnine layer to extend it.** This is the heart of `visualize`'s power.

### Add themes

```python
from plotnine import theme_minimal, theme_classic, theme

p_minimal = p + theme_minimal()
p_classic = p + theme_classic()

# Custom theme adjustments
p_custom = p + theme(
    figure_size=(10, 6),
    axis_text_x=element_text(rotation=45),
    legend_position="bottom",
)
```

### Change scales

```python
from plotnine import scale_x_log10, scale_y_log10, scale_color_brewer, scale_y_continuous

p_log = p + scale_x_log10() + scale_y_log10()             # log-log plot
p_color = p + scale_color_brewer(type="qual", palette="Set2")
p_y = p + scale_y_continuous(limits=(0, 20000), breaks=[0, 5000, 10000, 15000, 20000])
```

### Add reference lines

```python
from plotnine import geom_vline, geom_hline, geom_abline

p_with_refs = (
    p
    + geom_vline(xintercept=1.0, linetype="dashed", color="red")
    + geom_hline(yintercept=df["price"].median(), linetype="dotted", color="blue")
)
```

### Add additional geoms

```python
from plotnine import geom_smooth, geom_rug, geom_text, geom_label

# Layer a loess smooth on top of a scatter with linear smooth
p_layered = p + geom_smooth(method="loess", se=False, color="darkred", linetype="dashed")

# Add rug ticks to show 1d distribution
p_rug = p + geom_rug()
```

### Add labels and annotations

```python
from plotnine import labs

p_labeled = p + labs(
    title="Diamond Prices vs Carat by Cut",
    subtitle="Linear smoothing per cut group",
    caption="Source: diamonds dataset (n=3,000)",
    x="Carat (weight)",
    y="Price ($)",
    color="Cut quality",
)
```

### Save to file

```python
p.save("plot.png", width=10, height=6, dpi=150)
p.save("plot.pdf", width=10, height=6)
```

### Combine plots

```python
# Built-in multi-variable composition
rsm.eda.visualize(df, x=["price", "carat", "depth"], geom="hist", ncol=2)
rsm.eda.visualize(df, x=["carat", "depth"], y=["price", "quantity"], geom="scatter")

# Or request the individual ggplot objects
plots = rsm.eda.visualize(df, x=["price", "carat"], geom="hist", ret="list")

# Manual composition operators from plotnine also work
p1 = rsm.eda.visualize(df, x="carat", y="price")
p2 = rsm.eda.visualize(df, x="carat", geom="density")
composed = p1 | p2     # side by side
composed = p1 / p2     # vertically stacked
```

The point: `visualize` produces a sensible default plot; everything beyond that is plotnine. The user can keep all of pyrsm's polars-friendly conveniences AND get the full plotnine API for customization.

## Step 7 — Extend the data with polars (preprocess before plotting)

`visualize` doesn't transform the data — it expects the right shape. The natural pattern is:

```python
# Compute / reshape / filter first with polars
plot_data = (
    df
    .filter(pl.col("cut") != "Fair")              # drop a level
    .with_columns(price_log=pl.col("price").log())   # log transform
    .group_by("year")                              # aggregate if needed
    .agg(mean_price=pl.col("price").mean())
)

# Then plot
rsm.eda.visualize(plot_data, x="year", y="mean_price", geom="line")
```

Or use `pyrsm-pivot` → `pyrsm-unpivot` → `pyrsm-visualize` for the canonical pipeline:

```python
wide = rsm.eda.pivot(df, rows="region", cols="quarter", values="sales", agg="sum")
long = rsm.eda.unpivot(wide, on=["Q1","Q2","Q3","Q4"], id_vars="region",
                       variable_name="quarter", value_name="sales")
rsm.eda.visualize(long, x="quarter", y="sales", color="region", geom="line")
```

## Step 8 — Related EDA tools (when to switch)

`visualize` is the right tool when you have a specific plot in mind. If the situation is different, suggest:

- **Holistic first-look at all variables** → `pyrsm-distr` (auto-classifies + composes per-column plots).
- **Per-column numeric summaries** → `pyrsm-explore`.
- **Cross-tabulation** → `pyrsm-pivot`.
- **Reshape wide ↔ long before plotting** → `pyrsm-pivot` / `pyrsm-unpivot`.
- **Joining two DataFrames** → `pyrsm-combine`.

Don't switch silently — explain why the alternative is a better fit and confirm with the user before changing tools.

---

## Style notes

- **Pedagogical tone.** State which geom and why. Match the geom to the question, not to "what looks pretty".
- **Use the units from the description file.** "Plot of price by carat" is much weaker than "Plot of diamond price (USD) by weight (carats), colored by cut quality".
- **Use plotnine extensions liberally.** `visualize` is the convenient default; layered plotnine is the full customization. Show the user how to add scales, themes, reference lines, and additional geoms.
- **Preprocess with polars before plotting.** `visualize` expects the data in plot-ready shape — let polars (and/or other pyrsm.eda functions) get it there.
- **Watch the geom auto-pick rules.** `dist` and `hist` switch to bar for categorical x; `line` with `color=` auto-groups; bar with `agg=` uses `stat="summary"`.
- **For large scatter, use `nobs=`.** The default 1000 keeps charts legible; pass `-1` only when really wanted.
- **Don't over-engineer.** Plain top-level code that the student can copy line by line.
- **One step at a time.** After Step 4 wait for the user to look at the plot before suggesting more layers.

## Growing this skill

This skill is one of a family of `pyrsm-eda-*` skills covering the `pyrsm.eda` module. When `visualize` isn't the right tool, point the user to the sibling skill named for the operation they need (`pyrsm-distr`, `pyrsm-explore`, `pyrsm-pivot`, `pyrsm-unpivot`, `pyrsm-combine`). The workflow shape — ask for absolute path → load via `scripts/load_data.py` → read sidecar → propose spec → run → interpret → extend with polars/plotnine — is shared across the family.
