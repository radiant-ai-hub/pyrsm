---
name: pyrsm-logistic
description: Run and interpret logistic regression analyses in Python using the pyrsm library's `logistic` class. Use this skill whenever a student or analyst wants to fit a logistic or binary classification model, predict the probability of a binary or categorical outcome, interpret odds ratios, evaluate AUC or pseudo R-squared, compare predictor importance, generate prediction scenarios, or work with the pyrsm package for any classification task — even if they don't explicitly say "pyrsm" or "logistic". Triggers include phrases like "predict whether a customer will churn", "model a binary outcome", "what predicts survival", "fit a classification model", "interpret odds ratios", "which variables predict purchase", "binary dependent variable", "are these odds ratios significant", or any mention of modeling a yes/no, 0/1, or two-level categorical response in a business or analytics context.
---

# pyrsm logistic regression workflow

This skill walks the user through a complete logistic regression workflow using
the `pyrsm.model.logistic` class — from loading their data to interpreting
odds ratios and comparing predictor importance. It is designed for students
and analysts in business/marketing analytics contexts, so default to clear,
pedagogical explanations rather than terse statistical output.

For deep reference on the `logistic` API, odds-ratio templates, model-fit
metrics, plot options, and prediction patterns, see `references/logistic.md`.
Read it whenever the user asks a question that goes beyond the basic
fit-and-summary loop.

## Persistent kernel (optional)

If you're driving this skill from an agent CLI (Pi, Claude Code, etc.) that
runs Python via subprocess, every `python -c '...'` call is a fresh
interpreter — loaded data and fitted models do NOT persist across turns.
For analysis work where you fit once and then explore the model (predict,
plot, evaluate, score new data), re-fitting on every turn wastes minutes.

The `pyrsm-kernel` wrapper at `skills/_kernel_bridge/` solves this:

```bash
pyrsm-kernel start               # start kernel for this project (once)
pyrsm-kernel exec '<code>'       # run code, keep state in memory
```

State (loaded DataFrames, fitted models) survives between `exec` calls. See
`skills/_kernel_bridge/README.md` for setup and the `SYSTEM_kernel.md`
context-file fragment to tell your agent to use it. Strongly recommended
for any non-trivial fitting workflow (rforest, xgboost, mlp) and for
iterating on regression / logistic analyses.

## Output policy

Two rules apply to **every** pyrsm call you make in this skill:

1. **Show the full output verbatim.** When you call `.summary()`, `.predict()`, or any pyrsm method that prints results, paste the **complete printed output** into your response — every line of the coefficient table, every statistic, every row. Do NOT abbreviate. Do NOT replace it with a high-level "main takeaways" bullet list. Do NOT skip the table to "save space". The user is reading your response because they want to see the actual numbers, not your summary of them.

2. **Provide the full pedagogical interpretation, not lame bullets.** When walking through results, use the detailed templates in `references/<name>.md`: state hypotheses explicitly, walk every coefficient or cell with its real-world units from the sidecar description, give the significance verdict at the 5% level, use the "holding constant" qualifier (where applicable), and tie back to the business question. A response like "X is positive and significant; Y is positive and significant; Z is not" is exactly the failure mode this skill is built to prevent.

3. **Lead with the requested output.** When the user asks for a specific plot, table, or diagnostic (e.g., "create and interpret PDP", "show me the residual dashboard", "generate PIP", "give me the confusion matrix", "plot the gains chart"), your interpretation must focus on THAT output — not on a re-interpretation of the model's coefficient table or summary. Pi's failure pattern is to say *"I'll generate X and then interpret the fitted results"* and then quietly produce a coefficient walk-through. The user already saw the coefficient interpretation in the previous turn; do not repeat it unless they explicitly ask. Lead with the requested output's content, using the dedicated template for that output if the skill has one (Step 9 for PIP, Step 10 for PDP, etc.).

4. **Always close interpretation with the importance disclaimer below.** See the 'Closing disclaimer — relative importance' section near the end of this SKILL.md. It is MANDATORY at the end of any walkthrough or interpretation response — even if the user did not explicitly ask "which variable matters most". Do not skip it.

If your context is genuinely constrained and you cannot reproduce the full output, say so explicitly — never silently abbreviate.


---

## Step 1 — Ask for the data location

The very first thing to do (before importing anything, before writing any code)
is ask the user where their data lives. Require an **absolute path**, because
relative paths are a frequent source of "file not found" frustration.

Example phrasing:

> Before we start, can you give me the **absolute path** to the data file you
> want to analyze? For example: `/Users/yourname/Downloads/customers.parquet`.
> Any of `.parquet`, `.csv`, `.tsv`, `.feather`, `.arrow`, `.xlsx`, or `.json`
> will work.

If the user gives a relative path, a `~`-style path, or just a filename, ask
again for an absolute path. If the path doesn't exist, say so and ask them to
double-check.

## Step 2 — Load the data and read the sidecar description

Use the same `scripts/load_data.py` script as the regression skill to load the
data and surface any sidecar description file:

```bash
python scripts/load_data.py "<absolute-path-to-data>"
```

The script prints a JSON report (path, shape, column names with dtypes) and the
path of any sidecar `.md` it found. Use that to plan the next step.

### Sidecar description file

In pyrsm-style course materials, every dataset ships with a sidecar markdown
file documenting what the variables mean. The convention is
`<dataset>_description.md` next to `<dataset>.parquet`. The loader script
searches for these patterns automatically.

If a sidecar is found, **read it before proposing variable choices**. The
descriptions tell you what each column measures, what the response variable is
likely to be, what the levels of binary/categorical columns mean, and any
quirks. This is the difference between a generic interpretation ("the odds of
Y increase by...") and a useful one ("the odds of purchasing the product
increase by...").

If no sidecar is found, say so and ask the user to briefly describe their
variables. Don't invent meanings.

## Step 3 — Identify the response variable and check its type

Once the schema is in hand, identify the column to model:

- The response variable (`rvar`) must be **binary or categorical** — typically
  a 2-level column (string, Enum, or 0/1 integer). If it is numeric with many
  values, it is not appropriate for logistic regression; suggest linear
  regression instead.
- If the user's question makes the response obvious ("predict whether someone
  churns"), state the proposed `rvar` and confirm.
- If ambiguous (the dataset has several binary columns), list candidates and
  ask.

Quick sanity checks before proceeding:

- How many levels does `rvar` have? 2 = binary, fine. 3+ = multinomial, outside
  this skill.
- Any missing values in `rvar`? Note them; `pyrsm` drops rows with missing
  values in `rvar`.
- Is the positive level obvious from context (e.g., `Yes`, `1`, `Churn`,
  `Purchased`)? Confirm it in Step 4.

## Step 4 — Choose the positive level (`lev`)

This is the step most students skip and later regret. The `lev` parameter
tells pyrsm which level of the response variable represents the "positive"
or "success" outcome. All odds ratios and predicted probabilities will be
computed *for this level*.

Example: if `rvar = "churn"` and the levels are `Yes` / `No`, set `lev = "Yes"`
to model the probability of churning.

If the data use 0/1 integer coding, the typical convention is `lev = 1`. But
**always confirm with the user** — in some datasets 0 is the interesting
outcome.

State your proposed `lev` and ask the user to confirm before fitting.

## Step 5 — Decide on explanatory variables (`evar`)

Propose a list of explanatory variables and confirm with the user. The minimum
is one; there is no fixed maximum, but warn about small samples with many
predictors.

Things to check:

- Are any `evar` columns numeric and very skewed (e.g., monetary amounts)?
  Flag it — log-transforming them may help, though it is not required.
- Are there categorical `evar` columns with many levels? Each non-reference
  level becomes a separate row in the output. More than ~10 levels can make
  the summary unwieldy.
- Is there any `evar` that is nearly perfectly collinear with another?
  Mention it.
- Interactions: skip unless the user's question calls for them or the sidecar
  description implies a joint effect.

## Step 6 — Fit the model

Write a short, readable Python snippet:

```python
import polars as pl
import pyrsm as rsm

# 1. Load the data (absolute path)
df = pl.read_parquet("<absolute-path>")   # adjust reader for the file type

# 2. Fit the logistic regression
clf = rsm.model.logistic(
    {"<dataset_name>": df},    # dict so the dataset name shows up in the summary
    rvar="<response>",
    lev="<positive_level>",
    evar=["<x1>", "<x2>", "..."],
)

# 3. Inspect
clf.summary()
```

Notes:
- Pass the data as `{"name": df}` rather than just `df`, so the summary header
  prints the dataset name instead of `"Not provided"`.
- `pyrsm` accepts polars DataFrames directly.
- For categorical (string / Enum / Categorical) `evar` columns, `pyrsm`
  automatically dummy-codes them using the alphabetically first level as the
  reference. Each non-reference level appears as its own row in the output
  formatted as `<var>[<level>]`.
- For 0/1 integer `rvar` columns, pass `lev=1`.

## Step 7 — Interpret the output

This is where most of the value is. Walk the user through the summary in this
order — see `references/logistic.md` for detailed templates and the exact
interpretation language:

### 7a. State the model

> I fitted a logistic regression with `<rvar>` as the response variable,
> predicting the probability of `<rvar> = <lev>`. Explanatory variables:
> `<evar_list>`.

### 7b. Interpret overall model fit

Three pieces:

1. **Chi-squared test**: the null is that all slope coefficients are zero.
   Report the chi-squared statistic, degrees of freedom, and p-value.
   Conclude whether the model as a whole is statistically significant.
2. **Pseudo R-squared (McFadden)**: analogous to R-squared, but not directly
   comparable. Values above 0.2 are generally considered adequate.
3. **AUC**: how well the model ranks positive vs. negative cases. Report it as:
   > The AUC is `<value>`, meaning the model correctly ranks `<value*100>`% of
   > randomly chosen positive/negative pairs by predicted probability.

### 7c. Interpret each coefficient (odds ratios)

For every row in the summary table, interpret in plain English using the `OR`
and `OR%` columns. See `references/logistic.md` for the exact language
templates.

Key rules:

- **Continuous predictor**: "For a one-`<unit>` increase in `<var>`, the odds
  of `<rvar> = <lev>` are multiplied by `<OR>` (`<OR%>` change), holding all
  other variables constant."
- **Categorical predictor level**: "Compared to `<reference_level>`, the odds
  of `<rvar> = <lev>` for `<level>` are `<OR%>` `<higher|lower>`, holding all
  other variables constant."
- **Always state the reference level** for categorical predictors. It is the
  alphabetically first level unless the user recoded it.
- The sign of `OR%` tells the direction: positive = higher odds, negative =
  lower odds.
- Use the sidecar description for units and context (e.g., "per $1,000 of
  income" rather than "per unit").

### 7d. Statistical significance

> Predictors with p-values below 0.05 are statistically significant at the 5%
> level. We reject the null hypothesis that the odds ratio equals 1 for those
> predictors.

**Do not drop non-significant predictors.** The same logic that protects linear
regression from omitted variable bias applies here: a non-significant predictor
may still be controlling for confounding, and dropping it can shift the
remaining coefficients. See Step 8.

### 7e. Do not rank predictor importance from odds ratios or z-values

Odds ratios are on different scales for numeric vs. categorical predictors and
for predictors with different ranges. Do not say one predictor "matters more"
because its odds ratio is larger or its z-value is bigger. Use permutation
importance for that comparison — see Step 9.

## Step 8 — Model simplification (only when there's a real need)

The same protocol as linear regression applies:

1. Simplification is rarely needed for a class assignment. The default end-state
   is the model fit in Step 6.
2. Statistical insignificance alone is not a reason to drop a predictor.
3. If simplification is warranted, drop the highest-p-value predictor first
   (one at a time), snapshot the remaining coefficients, and check for large
   shifts before concluding the drop was safe.

The OVB check uses percent shifts in odds ratios:

```python
# Snapshot before dropping
baseline = clf.coef.select(["index", "OR", "p.value"]).clone()

# Drop and refit
new_evar = [v for v in clf.evar if v != "<dropped_var>"]
clf2 = rsm.model.logistic({clf.name: clf.data}, rvar=clf.rvar, lev=clf.lev, evar=new_evar)

# Compare
new_coef = clf2.coef.select(["index", "OR", "p.value"]).rename({"OR": "new_OR", "p.value": "new_p"})
comparison = baseline.rename({"OR": "old_OR", "p.value": "old_p"}).join(
    new_coef, on="index", how="inner"
).with_columns(
    pct_shift=((pl.col("new_OR") - pl.col("old_OR")) / pl.col("old_OR").abs() * 100).round(1),
    sign_flip=(pl.col("new_OR").sign() != pl.col("old_OR").sign()),
    sig_flip=((pl.col("old_p") < 0.05) != (pl.col("new_p") < 0.05)),
)
```

Triggers: `|pct_shift| > 10%`, sign flip, or significance flip. If any trigger
fires, present the four remediation options (re-include / relabel / combine /
acknowledge the bias) and ask the user to choose.

## Step 9 — Plots, predictions, and permutation importance

After the basic interpretation, **offer** to run the following — pick the one
that answers the user's actual question, not all of them by default:

```python
clf.plot("or")            # odds-ratio forest plot with CIs
clf.plot("dist")          # distributions of all variables in the model
clf.plot("corr")          # correlation matrix
clf.plot("pdp")           # partial dependence plots
clf.plot("pdp", ice=True) # PDP with individual conditional expectation lines
clf.plot("pip")           # permutation importance bar chart
pip_df = clf.plot("pip", ret=True)[1]  # importance scores as a DataFrame
```

### Permutation importance (`pip`)

This is the correct way to compare predictor importance. Use it whenever the
user asks "which variable matters most?" or "how do the predictors compare?"

- Permutation importance measures the drop in AUC when each predictor is
  randomly shuffled. A larger drop = more important predictor.
- It is immune to the scale and coding issues that make odds ratio size
  misleading for cross-predictor comparisons.
- It is model-agnostic: the same interpretation works whether or not
  predictors are standardized.

Narrate the result like this:

> The permutation importance plot shows `<top_var>` as the most important
> predictor (AUC drop ≈ `<value>`), followed by `<second_var>` (≈ `<value>`).
> This ranking reflects each variable's actual contribution to the model's
> discriminatory accuracy — it can differ substantially from the ranking
> suggested by odds ratios or p-values.

### Predictions

```python
# Score training data
pred = clf.predict(ci=True)

# Counterfactual / scenario predictions
pred = clf.predict(cmd={"<var>": [v1, v2, v3]})  # other vars held at mean/mode
```

Use `cmd` for "what-if" questions:

> What is the predicted probability of purchase for customers earning $50K,
> $75K, and $100K, holding all other variables at their means?

### Chi-squared test for a subset of variables

To test whether a categorical variable (with multiple dummies) contributes
as a group:

```python
clf.summary(test="<var_name>")
```

This compares the full model against the model without that variable and
reports the chi-squared difference test. Use it when the user asks "does
`<var>` matter overall?" rather than level by level.

## Step 10 — Evaluate classification performance

The summary's AUC and pseudo R-squared are *fit* metrics. For *decision-relevant*
performance — confusion matrix, profit at a cost/margin, gains/lift curves,
ROME (return on marketing expenditure) — use the `pyrsm.model.perf` submodule.
These functions take a DataFrame that includes the predictions; the standard
pattern is:

```python
import pyrsm as rsm

# 1. Attach predictions to the data
pred = clf.predict()
data_with_pred = clf.data.with_columns(pl.Series("prediction", pred["prediction"]))

# 2. Overall evaluation metrics (binary classification)
rsm.model.perf.evalbin(
    data_with_pred, rvar="<rvar>", lev="<lev>", pred="prediction",
    cost=1, margin=2, scale=1,
)
# Returns a polars DataFrame with: TP, FP, TN, FN, TPR (recall), TNR, precision,
# F-score, accuracy, kappa, profit, ROME, contact rate, and AUC.

# 3. Confusion matrix
TP, FP, TN, FN, contact = rsm.model.perf.confusion(
    data_with_pred, rvar="<rvar>", lev="<lev>", pred="prediction",
    cost=1, margin=2,
)

# 4. AUC directly
rsm.model.perf.auc(
    rvar=data_with_pred["<rvar>"], pred=data_with_pred["prediction"], lev="<lev>",
)

# 5. Profit-max threshold (and corresponding profit / ROME)
rsm.model.perf.profit_max(
    data_with_pred, rvar="<rvar>", lev="<lev>", pred="prediction",
    cost=1, margin=2, scale=1,
)
rsm.model.perf.ROME_max(
    data_with_pred, rvar="<rvar>", lev="<lev>", pred="prediction",
    cost=1, margin=2,
)
```

For deciles-based plots and tables (the marketing-analytics workhorses):

```python
# 6. Gains / lift / profit / ROME plots and tables (10-decile default)
rsm.model.perf.gains_plot(data_with_pred, rvar="<rvar>", lev="<lev>", pred="prediction")
rsm.model.perf.gains_tab(data_with_pred, rvar="<rvar>", lev="<lev>", pred="prediction")
rsm.model.perf.lift_plot(data_with_pred, rvar="<rvar>", lev="<lev>", pred="prediction")
rsm.model.perf.profit_plot(
    data_with_pred, rvar="<rvar>", lev="<lev>", pred="prediction",
    cost=1, margin=2, scale=1,
)
rsm.model.perf.ROME_plot(data_with_pred, rvar="<rvar>", lev="<lev>", pred="prediction",
                        cost=1, margin=2)
rsm.model.perf.expected_profit_plot(
    data_with_pred, rvar="<rvar>", lev="<lev>", pred="prediction",
    cost=1, margin=2, scale=1,
)
```

### Interpretation rules of thumb

- **AUC ≈ 0.5**: model is no better than chance at ranking. AUC ≈ 0.8 is
  typically considered "good" for marketing classification problems.
- **Profit-max threshold**: the cutoff probability where expected profit
  is maximized given the `cost` (per-contact cost) and `margin` (per-success
  margin). Below this threshold, the expected profit per customer is negative
  and you shouldn't contact them.
- **Gains chart**: shows cumulative true positives captured as you contact
  the top X% of customers by predicted probability. A diagonal = random
  baseline; a steeper curve = better model.
- **Lift**: ratio of the response rate in the top X% to the overall response
  rate. Lift = 1 means the model adds no value at that depth.
- **ROME** (return on marketing expenditure) = `(profit / cost)` per
  contacted customer. Plot it to find the depth that maximizes ROI per dollar
  spent rather than per-customer profit.

State the `cost` and `margin` explicitly when reporting profit, ROME, or
profit-max threshold — they are user-supplied business parameters, not
statistical constants.

## Step 9 — Interpreting permutation importance (PIP)

When the user asks for "PIP", "permutation importance", "feature importance", "which variable matters most", or anything similar, this is a **separate task from interpreting the regression output**. Do NOT respond by re-interpreting coefficients / odds ratios. The user wants the PIP scores.

### Canonical code (ALWAYS use `ret=True`)

```python
pip_plot, pip_df = clf.plot("pip", ret=True)
print(pip_df)                                              # required: print the scores verbatim
pip_plot.save("pip.png", width=8, height=5, dpi=120)       # save the plot for visual inspection
```

`clf.plot("pip")` alone (without `ret=True`) returns only the plot — the numeric importance scores are not exposed. **Always use `ret=True`** so you get both the plot AND the polars DataFrame of (variable, importance) pairs.

### What the printed PIP DataFrame looks like

The DataFrame is one row per predictor (in `evar`), with the **importance** column reporting the drop in model fit when that predictor's values are randomly shuffled. For regression: R² drop. For classification (logistic, rforest, xgboost, mlp): AUC drop.

```
shape: (5, 2)
┌──────────┬────────────┐
│ variable ┆ importance │
╞══════════╪════════════╡
│ carat    ┆ 0.920      │
│ clarity  ┆ 0.090      │
│ color    ┆ 0.036      │
│ table    ┆ 0.002      │
│ depth    ┆ 0.001      │
└──────────┴────────────┘
```

This **DataFrame is what you must show the user verbatim** (per the Output policy). Then save and display the plot.

### Interpretation template — focus on the PIP scores, NOT on coefficients

Walk through the scores in order from largest to smallest:

1. **Top predictor.** "The most important predictor by permutation importance is `<top_var>` with importance `<value>` — that is, randomly shuffling `<top_var>` reduces the model's fit by `<value>` `<units: R² or AUC>`."

2. **Subsequent predictors.** "`<2nd>` is next at `<value>`, then `<3rd>` at `<value>`, …"

3. **Ratios make this concrete.** "`<top>` is roughly `<ratio>`x more important than `<next>`, and `<top>` is `<larger ratio>`x more important than `<smallest>`. Most of the model's predictive power is concentrated in `<top>` (and any other predictors with importance > some practical threshold)."

4. **Tie-back to the model and the business context.** If a predictor that you EXPECTED to matter has low PIP, flag it: it may be confounded with a more important predictor, or it may genuinely not contribute much. Conversely, the top PIP predictor is the one you'd preserve if you were forced to simplify the model.

5. **What PIP does NOT tell you.** PIP gives importance, not direction. A high-PIP predictor could have a positive or negative effect on the response — for that you need the regression coefficient (or PDP shape). PIP + coefficient sign together tell the full story.

### What NOT to do

- **Do NOT re-interpret the regression coefficients in response to a PIP request.** "Carat has a positive coefficient of $8,438" is a regression-summary statement; "carat has importance 0.92" is a PIP statement. They answer different questions. If the user asked for PIP, lead with PIP.
- **Do NOT skip the verbatim DataFrame** — print the polars output exactly as it comes back, before any prose interpretation.
- **Do NOT skip the plot save** — call `.save("pip.png", ...)` so a human can also inspect it visually.
- **Do NOT skip the closing disclaimer** — the same disclaimer at the end of any interpretation still applies (it explains *why* PIP is the right tool).

## Step 10 — Interpreting partial dependence plots (PDP)

When the user asks for "PDP", "partial dependence", "create and interpret pdp plots", or similar, this is a **separate task from interpreting the regression output**. Do NOT respond by re-interpreting coefficients / odds ratios. The user wants to see the PDP curves and have their *shapes* described.

### Canonical code

```python
p = clf.plot("pdp")                                       # one panel per predictor
p.save("pdp.png", width=10, height=8, dpi=120)              # save the composed grid
print("saved pdp.png")
```

For ICE lines (individual predicted curves overlaid on the average): `clf.plot("pdp", ice=True)` — supported by `regress` and `logistic`; for `rforest` / `xgboost` / `mlp`, use `plot("pdp_sklearn")` for an alternative implementation.

### Required workflow

1. **Save the plot to disk.** PDP returns a plotnine ggplot object (or a composed grid for multi-panel). Save it before doing anything else: `p.save("pdp.png", width=10, height=8, dpi=120)`. Without this, neither you nor the user can see the curves.

2. **Read the saved image with your image-reading tool.** Pi has `read` (which supports images), Claude Code has `Read` (which renders images inline). After saving:

   ```bash
   # In your bash call:
   pyrsm-kernel exec '
   p = reg.plot("pdp")
   p.save("pdp.png", width=10, height=8, dpi=120)
   print("saved pdp.png")
   '
   # Then in your next tool call:
   read pdp.png
   ```

   This is what lets you actually see and describe the curves rather than guessing from the coefficients.

3. **Describe the shape of each predictor's curve, one panel at a time.** For each predictor in `evar`, state:
   - **Direction**: is the curve rising, falling, flat, or non-monotonic?
   - **Linearity / non-linearity**: is the slope roughly constant, or does it change (curved, threshold, saturating)?
   - **Range**: what's the predictor's natural range, and how much does the predicted response change across that range (in the units of the response)?
   - **Anomalies**: outlier panels, sharp jumps, U-shapes, plateaus.

4. **Tie back to the substantive question** — what does this curve mean for the business / research outcome? "Each additional carat raises predicted price by about $X, and the curve is roughly linear up to ~3 carats before flattening" is the right level of detail.

### PDP interpretation templates

> **Predictor `<var>`** (range `<min>` to `<max>` `<unit>`): the PDP curve is `<rising | falling | flat | U-shaped | inverted-U | non-monotonic>`. The predicted `<response>` changes from `<low_value>` to `<high_value>` `<units of response>` across the predictor's range, with `<linear / curved / threshold-at-X / saturating-after-Y>` shape. `<Plain-English implication>`.

> **Categorical predictor `<var>`**: the PDP shows discrete steps, one per level. The reference level (`<reference>`) is at `<value>`, and other levels are at `<list>`. The largest step is between `<level_a>` and `<level_b>`, a `<diff>` `<units>` shift in predicted `<response>`. Note: with a categorical PDP, you're looking at the *average* predicted response per level holding other predictors at their means; this is identical information to the regression coefficients on those dummies, just rendered visually.

> **Where two PDPs look similar, that's fine** — the slopes correspond to the partial effects in the (linear) model; the PDP just plots them.

### What NOT to do

- **Do NOT re-interpret the regression coefficients in response to a PDP request.** A PDP request means "show me what the model predicts as each input varies"; a coefficient is "the slope of that prediction surface in the linear model." They tell related but different stories — and the user already saw the coefficient interpretation in a previous turn.
- **Do NOT compare PDP steepness across predictors to claim relative importance.** That's the failure mode the closing disclaimer is designed to prevent. A steep PDP for one predictor and a flat PDP for another does NOT mean the steep one is "more important" — the steepness depends on the predictor's natural range. Use PIP for importance ranking.
- **Do NOT skip the `p.save(...)` call.** Without it the plot exists only in the kernel's memory and the user can't see it.
- **Do NOT skip the image read.** If you save a PDP and then never read the image, you're guessing at shapes from your prior knowledge of the data, not from the actual model. Read the saved PNG and describe what you see.
- **Do NOT skip the closing disclaimer.** The same disclaimer applies — PDP shows shape, not importance; PIP shows importance.

## Closing disclaimer — relative importance (MANDATORY)

**Every interpretation response in this skill MUST end with the following disclaimer (in the user's vocabulary — but covering all the points below). Do not skip it. Do not bury it. Do not paraphrase away the substance.**

The reason: students and analysts routinely (and incorrectly) read off "which variable matters most" from coefficient magnitudes, t-values, z-values, odds ratios, or PDP shapes. None of those are valid for relative-importance ranking.

> **Note on relative importance.** The predictors in this model are on
> different scales — a 1-unit change in one variable is not the same kind
> of change as a 1-unit change in another (e.g., income in $1,000s vs.
> age in years vs. a 0/1 sex dummy). The coefficients and odds ratios are
> *per-unit* effects on their own scale, so their magnitudes are **not
> directly comparable** for ranking predictors by importance. The same
> caveat applies to t-values / z-values and to PDP curves: a steep PDP is
> not "more important" than a flat one — the shape depends on the
> predictor's natural range. To rank predictors by their actual
> contribution to model fit, use the permutation importance plot:
>
> ```python
> reg.plot("pip")                # or clf.plot("pip") for logistic
> pip_df = reg.plot("pip", ret=True)[1]    # the importance scores as a DataFrame
> ```
>
> PIP measures the drop in model fit (R² for regression, AUC for
> classification) when each predictor is randomly shuffled. The scores
> are on a common scale and **are** comparable across predictors and even
> across model types.

This disclaimer is mandatory at the **end** of any walkthrough — even when the user did not explicitly ask "which variable matters most". Students often draw that conclusion implicitly from the coefficient table or the PDPs, and the disclaimer is what prevents the wrong takeaway.

---

## Style notes

- **Pedagogical tone.** Many users are students learning classification for the
  first time. State hypotheses explicitly. Explain what AUC means in plain
  English. Remind them what "holding other variables constant" means.
- **Use sidecar descriptions for units and context.** Generic "a one-unit
  increase in X" is much weaker than "a one-year increase in customer tenure."
- **Don't over-engineer.** Plain top-level code that the student can copy line
  by line. No try/except scaffolding, no helper functions.
- **One step at a time.** After Step 6 (fit), wait for the user to see the
  summary before offering diagnostics or plots. After Step 7 (interpret), ask
  whether they want predictions or importance before running them.
- **Titanic as an example.** The notebook `examples/model/model-logistic-regression.ipynb`
  uses the Titanic dataset to illustrate the API. Use it to ground any example
  code, but always frame this skill as applicable to any binary classification
  problem — churn, purchase, default, survival, adoption, etc.
