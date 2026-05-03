---
name: pyrsm-regress
description: Run and interpret linear regression analyses in Python using the pyrsm library's `regress` class. Use this skill whenever a student or analyst wants to fit a linear regression, interpret coefficients, check residual diagnostics, run an F-test, compare linear vs log-log specifications, generate predictions, or work with the pyrsm package for an OLS task — even if they don't explicitly say "pyrsm" or "regress". Triggers include phrases like "regress y on x", "fit a linear model", "run an OLS", "interpret these coefficients", "are these residuals OK", "should I log-transform", or any mention of fitting a model to a parquet/csv dataset for a marketing/business analytics class.
---

# pyrsm linear regression workflow

This skill walks the user through a complete linear regression workflow using the `pyrsm.model.regress` class — from loading their data to interpreting the coefficients in plain English. It is designed for students learning regression in a business/marketing analytics context, so default to clear, pedagogical explanations rather than terse statistical output.

For deep reference on the `regress` API, residual diagnostics, log-log interpretation, categorical dummies, and prediction/PDP/PIP plots, see `references/regress.md`. Read it whenever the user asks a question that goes beyond the basic fit-and-summary loop.

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
reg.summary(rmse=True, ssq=True)
```

Notes:
- Pass the data as `{"name": df}` (a dict with one key) rather than just `df`, so the summary header prints a meaningful dataset name instead of `"Not provided"`.
- `pyrsm` accepts polars DataFrames directly — no need to convert to pandas first.
- For categorical (string / Enum / Categorical) columns, `pyrsm` automatically dummy-codes them; you'll see one row per non-reference level in the coefficient table.

Always pass `rmse=True, ssq=True` to `summary` for the first fit. The RMSE and sum-of-squares table give the student useful context (residual spread; how variance is partitioned).

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

---

## Style notes

- **Pedagogical tone.** Many users will be students learning regression for the first time. Always state the null and alternative hypotheses explicitly. Always tie significance back to a 5% threshold (or whatever the user specified). Always remind them what "holding other variables constant" means when interpreting a coefficient.
- **Use the units from the description file.** Generic "a one-unit increase in X" is much weaker than "a one-thousand-dollar increase in household income". The sidecar `.md` exists for exactly this reason.
- **Don't over-engineer.** This is an interactive, exploratory workflow, not a production pipeline. Don't wrap things in functions, don't add try/except scaffolding, don't write `if __name__ == "__main__"`. Plain top-level code that the student can copy line by line.
- **One step at a time.** After Step 4 (fit) wait for the user to look at the summary before barreling into diagnostics. After Step 5 (interpretation) ask whether they want to check diagnostics before running them.

## Growing this skill

This skill is structured to grow into a broader pyrsm modeling toolkit. When other model types (logistic, random forest, xgboost, MLP) are added, drop a new file under `references/` (e.g., `references/logistic.md`) and add a short selection section near the top of this `SKILL.md` to route the user to the right reference based on the response variable type and their goal.
