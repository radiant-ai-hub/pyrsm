---
name: pyrsm-regress
description: Run and interpret linear regression analyses in Python using the pyrsm library's `regress` class. Use this skill whenever a student or analyst wants to fit a linear regression, interpret coefficients, check residual diagnostics, run an F-test, compare linear vs log-log specifications, generate predictions, or work with the pyrsm package for an OLS task — even if they don't explicitly say "pyrsm" or "regress". Triggers include phrases like "regress y on x", "fit a linear model", "run an OLS", "interpret these coefficients", "are these residuals OK", "should I log-transform", or any mention of fitting a model to a parquet/csv dataset for a marketing/business analytics class.
---

# pyrsm linear regression workflow

This skill walks the user through a complete linear regression workflow using the `pyrsm.model.regress` class — from loading their data to interpreting the coefficients in plain English. It is designed for students learning regression in a business/marketing analytics context, so default to clear, pedagogical explanations rather than terse statistical output.

For deep reference on the `regress` API, residual diagnostics, log-log interpretation, categorical dummies, and prediction/PDP/PIP plots, see `references/regress.md`. Read it whenever the user asks a question that goes beyond the basic fit-and-summary loop.

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

The very first thing to do (before importing anything, before writing any code) is ask the user where their data lives. Require an **absolute path**, because relative paths are a frequent source of "file not found" frustration for students who are not yet comfortable with the shell.

Example phrasing:

> Before we start, can you give me the **absolute path** to the data file you want to analyze? For example: `/Users/yourname/Downloads/catalog.parquet`. Any of `.parquet`, `.csv`, `.tsv`, `.feather`, `.arrow`, `.xlsx`, or `.json` will work.

If the user gives you a relative path, a `~`-style path, or just a filename, ask again for an absolute path rather than guessing. If the path doesn't exist, say so plainly and ask them to double-check.

## Step 2 — Load the data into a polars DataFrame

Use `scripts/load_data.py` to do the loading. It handles file-type detection by extension, reads the file with the appropriate polars reader, and also looks for a sidecar markdown description file in the same folder.

Run it like this:

```bash
python scripts/load_data.py "<absolute-path-to-data>"
```

The script prints a small JSON report describing the loaded dataset (path, shape, column names with dtypes) and the path of any sidecar `.md` it found. Use that to plan your next step. After this initial probe, in the actual analysis code, you will do the equivalent inside the Python session you'll run the regression in — see Step 4.

### Sidecar description file

In pyrsm-style course materials, every dataset is shipped alongside a markdown file that documents what the variables mean. The convention is `<dataset>_description.md` next to `<dataset>.parquet` (e.g., `catalog.parquet` ↔ `catalog_description.md`), but other "very similar" filenames are also common in the wild (`catalog.md`, `catalog-readme.md`, `README_catalog.md`, etc.). The loader script searches for these patterns automatically.

If a sidecar is found, **read it before proposing variable choices to the user**. The descriptions tell you the units (e.g., "Income is measured in thousands of dollars"), what's categorical vs continuous, what the response variable is likely to be, and any quirks. This is the difference between a generic interpretation ("a one-unit increase in Income") and a useful one ("a one-thousand-dollar increase in household income").

If no sidecar is found, say so and ask the user to briefly describe their variables. Don't invent meanings.

## Step 3 — Decide together what to model

Once the data and (optionally) the description are in hand, propose a regression specification and confirm it with the user before fitting. The minimum you need is:

- **Response variable (`rvar`)** — one column. Must be numeric for linear regression.
- **Explanatory variables (`evar`)** — a list of one or more columns.
- **(Optional) Interactions (`ivar`)** — a list like `["Income:HH_size"]`. Skip unless the user asks or the question clearly calls for it.

If the description file or the user's question makes the choice obvious (e.g., "predict Sales from Income, household size, and age"), state your proposed specification and ask the user to confirm. If the choice is ambiguous, list candidate response variables (numeric columns that look like outcomes) and ask which one.

Do a quick sanity check before fitting:

- Is `rvar` numeric? If it's binary 0/1, mention that logistic regression (`rsm.model.logistic`) might fit better, but proceed with linear if the user wants it.
- Do any `evar` columns have a lot of missing values? Mention it.
- Are there obvious skew / log-candidate columns (e.g., `price`, `sales`, `income`)? Don't auto-transform — but flag it so the user can decide.

## Step 4 — Fit the model

Write a short, runnable Python script (or run it interactively if you have a Python REPL available). Keep it minimal and readable — students will look at this code:

```python
import polars as pl
import pyrsm as rsm

# 1. Load the data (absolute path)
data_path = "<absolute-path>"
df = pl.read_parquet(data_path)   # adjust reader for the actual file type

# 2. Fit the regression
reg = rsm.model.regress(
    {"<dataset_name>": df},        # dict so the dataset name shows up in the summary
    rvar="<response>",
    evar=["<x1>", "<x2>", "..."],
)

# 3. Inspect
reg.summary()
```

Notes:
- Pass the data as `{"name": df}` (a dict with one key) rather than just `df`, so the summary header prints a meaningful dataset name instead of `"Not provided"`.
- `pyrsm` accepts polars DataFrames directly — no need to convert to pandas first.
- For categorical (string / Enum / Categorical) columns, `pyrsm` automatically dummy-codes them; you'll see one row per non-reference level in the coefficient table.

**Use plain `reg.summary()` by default.** The plain output already includes the coefficient table, F-statistic, R², adjusted R², n, df, and significance stars — everything needed for interpretation. Only add optional flags when the user explicitly asks: `ssq=True` for the sum-of-squares decomposition, `rmse=True` for residual spread, `vif=True` for multicollinearity diagnostics, `ci=True` for coefficient confidence intervals, `test=[...]` for an F-test on a subset of coefficients.

## Step 5 — Interpret the output

This is where most of the value is. Walk the user through the summary in roughly this order — see `references/regress.md` for the detailed templates and worked examples:

1. **Model as a whole** — the F-statistic and its p-value answer "is this model doing anything at all?" State the null/alternative hypotheses, report the F-statistic and p-value, and conclude in one sentence.
2. **R² and adjusted R²** — what fraction of the variance in the response is explained. One sentence.
3. **Each coefficient** — for every explanatory variable, state in plain English:
   - the direction and magnitude of the effect, **in the units of the response and the predictor** (use the sidecar description for units!),
   - whether it is statistically significant at the 5% level,
   - the standard "holding all other variables in the model constant" qualifier.
4. **Anything notable** — a coefficient with a surprising sign, a non-significant variable the user expected to matter, a categorical variable expanded into many dummies, etc.

For categorical predictors, remember each non-reference level gets its own row formatted as `<var>[<level>]`. Interpret as "compared to the reference level …".

For log-log or semi-log specifications, switch to the elasticity / percentage-change interpretation — see `references/regress.md`.

### Don't drop variables on p-value alone

After a first fit, students very often look at a non-significant predictor (p > 0.05) and want to drop it. This is one of the most consequential mistakes in applied regression, and the skill should always push back on it. The reasons, stated for the student:

- **Non-significance is not the same as "no effect".** It says we don't have enough evidence to distinguish the coefficient from zero in *this* sample, not that the variable is irrelevant. Dropping it discards information.
- **Dropping a variable can introduce omitted variable bias (OVB) in the coefficients you keep.** If the dropped variable is correlated with a kept variable, the kept variable's coefficient will absorb part of the dropped variable's effect. The numbers and the interpretation both change — and not for the better.
- **The right time to think about simplification is *after* the model is essentially finalized**, not after the first fit. And even then, simplification is a judgment call that should be motivated by something other than "the p-value is high" — typically prediction-deployment cost, communication, or genuine dimensionality reduction.

So: **as part of the Step 5 interpretation, when a coefficient is non-significant, explicitly state that you are *keeping it in the model*** and walk the student through what the non-significance does and does not mean. Do not propose dropping it. If the student asks to drop it ("can I drop Age since p = 0.559?"), do not just remove it — go to Step 6 (Model simplification) and walk the protocol.

## Step 6 — Model simplification (only when there's a real need)

Most regressions for a class assignment do *not* need simplification. The default end-state is the model the student fit in Step 4, with the interpretation from Step 5. Only enter this step when there is a concrete reason to remove a variable: dimensionality concerns, predictive deployment cost, or a clear communication need. **Statistical insignificance alone is not a reason.**

When simplification is genuinely warranted, follow this protocol — it is what protects the student from silently introducing OVB:

### 6a. Confirm the purpose

Ask: "Why do we want to simplify? Is the goal prediction, communication, or dimensionality?" If the only answer is "this variable isn't significant", say so plainly and stop here — go back to Step 5 and report the model as fit.

### 6b. Snapshot the current coefficients

Before dropping anything, save the current coefficient table — variable name, coefficient, std.error, p-value. This is the **baseline** that lets you detect OVB after a drop. A simple way:

```python
baseline = reg.coef.select(["index", "coefficient", "std.error", "p.value"]).clone()
print(baseline)
```

### 6c. Drop the highest-p-value variable first (one at a time)

Identify the variable with the largest p-value among the **non-significant** ones (p > 0.05). Drop only that one. Refit:

```python
new_evar = [v for v in reg.evar if v != "<dropped_var>"]
reg2 = rsm.model.regress({"<name>": df}, rvar=reg.rvar, evar=new_evar)
reg2.summary()
```

For categorical predictors, drop the *whole variable* (all dummies together), not a single level.

### 6d. Compare the remaining coefficients to the baseline — the OVB check

For every variable that is still in the model, compare the new coefficient to the baseline:

```
percent_shift = (new_coef - old_coef) / |old_coef| * 100
```

**Trigger an OVB investigation if *any* remaining variable shows:**

- `|percent_shift| > 10%`, **OR**
- the **sign flips** (was positive, is now negative, or vice versa), **OR**
- the **significance status flips** across the 5% line (was p < 0.05, is now p ≥ 0.05, or vice versa).

If none of those triggers fire, the drop appears OVB-safe — the variable was approximately orthogonal to the rest of the model. Document this finding and (if needed) proceed to drop the next-highest-p-value variable, repeating from 6b.

### 6e. If OVB is triggered, investigate

Don't just shrug and accept the new model. Present *all four* of these options to the student as live alternatives, and ask them to pick one with a rationale:

1. **Re-include the dropped variable.** The simplest fix. The "cost" of carrying a non-significant predictor is usually small; the cost of a biased kept coefficient is usually larger.
2. **Relabel a remaining variable.** If a remaining predictor's coefficient now clearly captures the combined effect of itself plus the dropped variable, reframe its interpretation accordingly. ("`Income` here is now picking up an income-and-household-life-stage effect.")
3. **Combine the correlated variables.** Build a single composite predictor (a weighted average, an index, a principal component, or a domain-specific construct) and use that instead of either original variable.
4. **Keep the dropped-variable model, but acknowledge.** Only legitimate when the student can articulate *why* the bias is acceptable for the question at hand. This must be stated explicitly, not glossed over.

Whichever option the student picks, document the reasoning in the writeup. The goal is not to land on a "clean" model with three asterisks next to every coefficient — it is to land on a model whose coefficients can be honestly interpreted.

## Step 7 — Diagnostics (offer, don't force)

After the basic interpretation, **offer** to run residual diagnostics. Don't drown the user in plots they didn't ask for, but make clear that conclusions from a regression are only trustworthy if the residuals look reasonable.

The most useful diagnostic plots, in order:

- `reg.plot("dist")` — distributions of every variable in the model (spot skew that suggests a log transform).
- `reg.plot("corr")` — correlation matrix among response and explanatory variables (spot multicollinearity).
- `reg.plot("scatter")` — scatter of response against each predictor with a fitted line (spot non-linearity).
- `reg.plot("dashboard")` — six-panel residual dashboard (predicted-vs-actual, residuals, Q-Q, etc.). The single most important check.
- `reg.plot("residual")` — residuals vs each explanatory variable (spot heteroscedasticity, missed non-linearity).

If the dashboard looks bad — fanning residuals, curved predicted-vs-actual, heavy Q-Q tails — the conclusions from Step 5 are not trustworthy. Suggest a log-log specification (especially for `price`, `sales`, dollar amounts) and offer to refit. See the diamonds example in `references/regress.md`.

## Step 8 — Predictions, importance, partial dependence (optional)

If the user wants to use the model rather than just describe it, the relevant methods are:

- `reg.predict(data=..., ci=True, conf=0.95, dec=3)` — predictions with confidence intervals.
- `reg.predict(cmd={"Income": [50, 100, 150]})` — predictions at specific values, holding other variables at their means/modes.
- `reg.plot("pred")` — predicted-value plots across the range of each predictor.
- `reg.plot("pdp")` (add `ice=True` for ICE lines) — partial dependence plots.
- `reg.plot("pip")` — permutation importance (with `ret=True` to also get the importance scores back as a DataFrame).
- `reg.plot("coef")` — coefficient plot with confidence intervals.

Don't run all of these by default — pick the one that answers the user's actual question.

## Step 9 — Interpreting permutation importance (PIP)

When the user asks for "PIP", "permutation importance", "feature importance", "which variable matters most", or anything similar, this is a **separate task from interpreting the regression output**. Do NOT respond by re-interpreting coefficients / odds ratios. The user wants the PIP scores.

### Canonical code (ALWAYS use `ret=True`)

```python
pip_plot, pip_df = reg.plot("pip", ret=True)
print(pip_df)                                              # required: print the scores verbatim
pip_plot.save("pip.png", width=8, height=5, dpi=120)       # save the plot for visual inspection
```

`reg.plot("pip")` alone (without `ret=True`) returns only the plot — the numeric importance scores are not exposed. **Always use `ret=True`** so you get both the plot AND the polars DataFrame of (variable, importance) pairs.

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
p = reg.plot("pdp")                                       # one panel per predictor
p.save("pdp.png", width=10, height=8, dpi=120)              # save the composed grid
print("saved pdp.png")
```

For ICE lines (individual predicted curves overlaid on the average): `reg.plot("pdp", ice=True)` — supported by `regress` and `logistic`; for `rforest` / `xgboost` / `mlp`, use `plot("pdp_sklearn")` for an alternative implementation.

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

- **Pedagogical tone.** Many users will be students learning regression for the first time. Always state the null and alternative hypotheses explicitly. Always tie significance back to a 5% threshold (or whatever the user specified). Always remind them what "holding other variables constant" means when interpreting a coefficient.
- **Use the units from the description file.** Generic "a one-unit increase in X" is much weaker than "a one-thousand-dollar increase in household income". The sidecar `.md` exists for exactly this reason.
- **Don't over-engineer.** This is an interactive, exploratory workflow, not a production pipeline. Don't wrap things in functions, don't add try/except scaffolding, don't write `if __name__ == "__main__"`. Plain top-level code that the student can copy line by line.
- **One step at a time.** After Step 4 (fit) wait for the user to look at the summary before barreling into diagnostics. After Step 5 (interpretation) ask whether they want to check diagnostics before running them.

## Growing this skill

This skill is structured to grow into a broader pyrsm modeling toolkit. When other model types (logistic, random forest, xgboost, MLP) are added, drop a new file under `references/` (e.g., `references/logistic.md`) and add a short selection section near the top of this `SKILL.md` to route the user to the right reference based on the response variable type and their goal.
