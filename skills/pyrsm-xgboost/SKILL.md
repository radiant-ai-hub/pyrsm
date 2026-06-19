---
name: pyrsm-xgboost
description: Fit and interpret XGBoost (gradient-boosted trees) models in Python using the pyrsm library's `xgboost` class — for either binary classification (`mod_type="classification"`, predict P(`lev`)) or regression (`mod_type="regression"`, predict a continuous outcome). Use this skill whenever a student or analyst wants to fit a gradient-boosted model, tune the learning_rate / n_estimators / max_depth / regularization tradeoff with cross-validation, examine feature importance (permutation or xgboost's built-in), look at partial-dependence plots, score new data, or evaluate classification performance (confusion, AUC, gains, lift, profit). Triggers include phrases like "fit an xgboost", "boosting model", "tune xgboost", "gradient boosted trees", "compare random forest vs xgboost", "what's the best learning rate", "xgboost feature importance", "PDP for xgboost", "GridSearchCV with pyrsm.model.xgboost", or any boosted-tree modeling request in a marketing/business analytics context.
---

# pyrsm XGBoost workflow

This skill walks the user through fitting, interpreting, tuning, and evaluating an XGBoost model using the `pyrsm.model.xgboost` class — from loading the data to producing AUC / R², permutation importance, PDP plots, and (for classification) profit-max / gains / lift / ROME performance metrics.

For deep reference on the `xgboost` API, the learning_rate / n_estimators tradeoff, hyperparameter tuning via the `cv=` argument or `pyrsm.model.cross_validation`, the importance flavors, and worked examples, see `references/xgboost.md`.

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

Same as the other model skills. Require an **absolute path**.

> Before we start, can you give me the **absolute path** to the data file you want to analyze?

If the user gives a relative or `~` path, ask for an absolute one.

## Step 2 — Load the data and read the sidecar

Use `scripts/load_data.py`:

```bash
python scripts/load_data.py "<absolute-path-to-data>"
```

The JSON report includes the column types. If a sidecar `.md` exists, read it before proposing predictors.

## Step 3 — Decide what to model

Confirm the spec with the user. You need:

- **Response (`rvar`)**: one column. Binary categorical → `mod_type="classification"`. Continuous numeric → `mod_type="regression"`.
- **Level (`lev`)** *(classification only)*: which level of `rvar` is the positive outcome.
- **Predictors (`evar`)**: list of one or more columns. Categoricals are auto-dummy-coded.
- **Hyperparameters** (key ones — see Step 7 for tuning):
  - `n_estimators` (default 100) — number of boosting rounds.
  - `max_depth` (default 6) — maximum tree depth. XGBoost trees are typically shallow (3–8) compared to Random Forest's unbounded trees.
  - `learning_rate` (default 0.3) — shrinkage factor applied to each tree's contribution. **The most consequential XGBoost hyperparameter.** Lower values (0.01–0.1) need more `n_estimators` but generalize better; higher values (0.3+) are faster but more prone to overfitting.
  - `min_child_weight` (default 1) — minimum sum-of-weights per leaf. Larger = more regularization.
  - `subsample` (default 1.0) — fraction of rows sampled per tree.
  - `colsample_bytree` (default 1.0) — fraction of features sampled per tree.
  - `random_state` (default 1234) — for reproducibility.
- **Train/test split**: **mandatory for XGBoost.** Unlike Random Forest, XGBoost has *no OOB equivalent* — the `summary()` AUC is in-sample on the training rows, and is dramatically overoptimistic. Use `rsm.model.make_train(strat_var=<rvar>, test_size=0.3, random_state=1234)` to add a `training` column.

Confirm: "I'll fit an XGBoost classifier predicting survived='Yes' from pclass, sex, age. 100 trees, max_depth=3, learning_rate=0.1, train on 70%, evaluate on the held-out 30%."

### XGBoost vs Random Forest

Mention this when relevant:

| Aspect | Random Forest | XGBoost |
| --- | --- | --- |
| Tree depth | Unbounded by default | Limited (default 6) |
| Tree fitting | Independent (parallel) | Sequential, each corrects the previous |
| Regularization | Implicit (averaging) | Explicit (learning_rate, min_child_weight, subsample) |
| Honest training-set estimate | OOB (built in) | None — **must use a held-out test set** |
| Typical AUC on tabular data | Good | Often higher with tuning |
| Tuning complexity | Few key knobs | Many interacting knobs |

XGBoost is more powerful but easier to overfit and requires honest validation. Random Forest is more forgiving with defaults.

## Step 4 — Fit the model

```python
import polars as pl
import pyrsm as rsm

# 1. Load and split (mandatory for XGBoost)
df = pl.read_parquet("<absolute-path>")
df = df.with_columns(
    training=rsm.model.make_train(df, strat_var="<rvar>", test_size=0.3, random_state=1234),
)

# 2. Fit on training data
xgb_model = rsm.model.xgboost(
    data={"<dataset> (train)": df.filter(pl.col("training") == 1)},
    rvar="<response>",
    lev="<positive_level>",          # for classification
    evar=["<x1>", "<x2>", "..."],
    n_estimators=100,
    max_depth=3,                     # shallower trees → less overfitting; tune later
    learning_rate=0.1,               # default 0.3 is often too high; 0.1 is safer
    random_state=1234,
    mod_type="classification",       # or "regression"
)

# 3. Inspect
xgb_model.summary()
```

Notes:
- The model is fit inside `__init__` — no separate `.fit()`. Pyrsm passes `eval_set=[(X_train, y_train)]` under the hood for the early-stopping interface (though `early_stopping_rounds` is not exposed by default).
- The dummy-encoding is done internally; categoricals don't need pre-processing.
- For regression, set `mod_type="regression"` and omit `lev`.

## Step 5 — Interpret the summary

Walk through the printed summary:

1. **Model setup**: data, rvar, lev, evar, classification vs regression.
2. **Feature counts and observations**: `(original, dummy-encoded)`. nobs after dropping rows with any missing predictor.
3. **Hyperparameters**: n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, random_state.
4. **Performance**:
   - Classification: **AUC** — but this is computed **on the training data, in-sample**. It is dramatically overoptimistic.
   - Regression: R² and RMSE, similarly in-sample.

### The in-sample AUC trap (the critical concept)

This is the **defining pedagogical concept** for XGBoost in pyrsm:

- **The AUC printed by `xgb_model.summary()` is computed on the training data**, using the fitted model. Every training row has been seen by the model during fitting; predictions on these rows reflect partial memorization.
- **XGBoost has NO OOB equivalent.** Unlike Random Forest (where each row has a natural "trees that didn't see it"), boosting fits sequentially on the full data. Every tree learns from every row that survived subsampling.
- **Always evaluate on a held-out test set.** The honest performance estimate is `xgb_model.predict(test_df)` → attach to test rows → compute AUC on the test rows only.

Explicit pattern:

```python
# In-sample AUC (overoptimistic)
print("In-sample AUC:", xgb_model.summary())   # whatever the summary prints

# Honest held-out test AUC
df = df.with_columns(pred_xgb=xgb_model.predict(df).get_column("prediction"))
test_df = df.filter(pl.col("training") == 0)
test_auc = rsm.model.perf.auc(rvar=test_df["<rvar>"], pred=test_df["pred_xgb"], lev="<lev>")
print("Held-out test AUC:", test_auc)
```

For Titanic with default hyperparameters: in-sample AUC ≈ 0.96+, held-out AUC ≈ 0.83–0.85. The gap of ~0.10+ is the memorization premium that disappears on test data. **Always report the test AUC, not the summary's AUC.**

## Step 6 — Feature importance and partial dependence

After the basic interpretation, **offer** these plots:

```python
xgb_model.plot("pip")              # permutation importance (recommended)
pip_df = xgb_model.plot("pip", ret=True)[1]

xgb_model.plot("pip_sklearn")      # xgboost's built-in importance (gain, weight, cover)

xgb_model.plot("pdp")              # partial dependence plots
xgb_model.plot("pdp_sklearn")      # alternative PDP via sklearn.inspection

xgb_model.plot("pred")             # predicted-value curves per predictor
```

### `pip` vs `pip_sklearn`

Same critical concept as for Random Forest:

- **`pip` (permutation importance)** — shuffles each predictor and measures the AUC drop. **Model-agnostic, comparable across predictors and model types.** Use this as the default.
- **`pip_sklearn`** — XGBoost's internal importance (default `"gain"` metric, which sums the gain in split quality). Biased toward features that appear early in trees and toward continuous predictors with many split candidates.

Default to `pip` for any cross-predictor or cross-model importance question. Use `pip_sklearn` only when comparing to an XGBoost-published baseline or for a quick within-model view.

### PDP — partial dependence

Shows the average prediction as a function of each predictor, with others marginalized. Works the same as for Random Forest. Especially useful for XGBoost where the trees can capture sharp non-linearities and thresholds the PDP makes visible.

## Step 7 — Hyperparameter tuning via cross-validation

XGBoost has many interacting hyperparameters. The most consequential pair:

- **`learning_rate` × `n_estimators`** — they trade off. Lower learning_rate needs more rounds to converge but generalizes better. A common heuristic: `learning_rate=0.01, n_estimators=1000` >> `learning_rate=0.3, n_estimators=100` for AUC on most datasets — at higher compute cost.

Other key knobs:
- **`max_depth`** — controls tree complexity. 3–6 is typical; deeper trees overfit faster with XGBoost than with Random Forest because boosting *amplifies* small fits.
- **`subsample` / `colsample_bytree`** — random subsampling of rows / columns per tree. Both at 0.7–0.8 add useful regularization.
- **`min_child_weight`** — minimum sum of instance weights in a leaf. Larger = more regularization.

### Grid search via `cross_validation`

```python
param_grid = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 300],
}
cv = rsm.model.cross_validation(xgb_model, "xgb-cv", param_grid, {"AUC": "roc_auc"})

xgb_tuned = rsm.model.xgboost(
    data={"<dataset> (train)": df.filter(pl.col("training") == 1)},
    rvar="<rvar>", lev="<lev>", evar=[...],
    random_state=1234,
    **cv.best_params_,
)
xgb_tuned.summary()
```

A `3 × 3 × 3 = 27` grid with 5-fold CV is `27 × 5 = 135` model fits — usually fast enough for class work.

### Practical tuning advice

1. **Start conservative**: `max_depth=3`, `learning_rate=0.1`, `n_estimators=100`. Run held-out AUC.
2. **If training–test gap is small**: try slightly more capacity (`max_depth=4` or more `n_estimators`).
3. **If training–test gap is large**: reduce capacity (`learning_rate=0.05`, `subsample=0.7`, `colsample_bytree=0.7`, `min_child_weight=5`).
4. **Tune `learning_rate` × `n_estimators` together**: don't fix `n_estimators` and only tune `learning_rate`.

## Step 8 — Predictions

```python
# Score new data (test set, scoring set, etc.)
pred = xgb_model.predict(test_df)
# Returns a pl.DataFrame with predictor columns + 'prediction' column

# Attach to master DataFrame
df = df.with_columns(pred_xgb=xgb_model.predict(df).get_column("prediction"))

# Counterfactual / scenario predictions
pred = xgb_model.predict(cmd={"age": [10, 30, 50]})
```

For classification, `prediction` is `P(rvar = lev)`. For regression, it's the predicted continuous value.

**Critical for honest evaluation**: predictions on training rows are overoptimistic (memorization). Use the test-set predictions for any reported AUC / RMSE / profit metric. Unlike Random Forest, there is no OOB workaround — you genuinely need a held-out split.

## Step 9 — Evaluate classification performance

```python
# Score everything and attach
df = df.with_columns(pred_xgb=xgb_model.predict(df).get_column("prediction"))

# Decision-relevant metrics — compute on TEST ROWS for honest numbers
test_df = df.filter(pl.col("training") == 0)
rsm.model.perf.evalbin(test_df, rvar="<rvar>", lev="<lev>", pred="pred_xgb",
                       cost=1, margin=10, scale=1)

# Confusion matrix at profit-max threshold (on test rows)
TP, FP, TN, FN, contact = rsm.model.perf.confusion(
    test_df, rvar="<rvar>", lev="<lev>", pred="pred_xgb", cost=1, margin=10
)

# Decile plots — compare train and test on one chart
dct = {"train": df.filter(pl.col("training") == 1),
       "test":  df.filter(pl.col("training") == 0)}
rsm.model.perf.gains_plot(dct, rvar="<rvar>", lev="<lev>", pred="pred_xgb")
rsm.model.perf.profit_plot(dct, rvar="<rvar>", lev="<lev>", pred="pred_xgb",
                          cost=1, margin=10)
rsm.model.perf.lift_plot(dct, rvar="<rvar>", lev="<lev>", pred="pred_xgb")
rsm.model.perf.ROME_plot(dct, rvar="<rvar>", lev="<lev>", pred="pred_xgb",
                        cost=1, margin=10)
```

**The train–test gap is the key diagnostic.** With XGBoost, expect some gap — if training is dramatically steeper than test, the model is overfit. Reduce `max_depth`, increase `min_child_weight`, lower `learning_rate`, or add row/column subsampling.

**Always state cost and margin** in any profit / ROME / threshold report.

For regression, use `rsm.model.evalreg` (or compute R² / RMSE / MAE manually) on the test rows.

## Step 10 — Related models (when to switch)

`xgboost` is the highest-capacity tree model and usually the strongest default for AUC on tabular classification. If the situation is different, suggest:

- **Interpretable coefficients** → `logistic` / `regress`.
- **Less tuning effort, robust default** → `rforest`.
- **Neural network for very large / complex data** → `mlp` (`pyrsm-mlp` skill).
- **Descriptive only, no model** → `pyrsm-eda` family.

XGBoost typically beats Random Forest by 0.01–0.05 AUC on standard tabular problems *after tuning*. Without tuning, the two are often comparable.

## Step 9 — Interpreting permutation importance (PIP)

When the user asks for "PIP", "permutation importance", "feature importance", "which variable matters most", or anything similar, this is a **separate task from interpreting the regression output**. Do NOT respond by re-interpreting coefficients / odds ratios. The user wants the PIP scores.

### Canonical code (ALWAYS use `ret=True`)

```python
pip_plot, pip_df = xgb_model.plot("pip", ret=True)
print(pip_df)                                              # required: print the scores verbatim
pip_plot.save("pip.png", width=8, height=5, dpi=120)       # save the plot for visual inspection
```

`xgb_model.plot("pip")` alone (without `ret=True`) returns only the plot — the numeric importance scores are not exposed. **Always use `ret=True`** so you get both the plot AND the polars DataFrame of (variable, importance) pairs.

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
p = xgb_model.plot("pdp")                                       # one panel per predictor
p.save("pdp.png", width=10, height=8, dpi=120)              # save the composed grid
print("saved pdp.png")
```

For ICE lines (individual predicted curves overlaid on the average): `xgb_model.plot("pdp", ice=True)` — supported by `regress` and `logistic`; for `rforest` / `xgboost` / `mlp`, use `plot("pdp_sklearn")` for an alternative implementation.

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

The reason: students and analysts routinely (and incorrectly) read off "which variable matters most" from PDP shapes, tree split counts, or the sklearn-style "feature importances". None of those are valid for relative-importance ranking.

> **Note on relative importance.** PDPs show the *shape* of each
> predictor's effect on the model output, not the *magnitude of its
> contribution relative to other predictors*. A predictor with a steep
> PDP is **not** automatically "more important" than one with a flat PDP
> — the steepness depends on the predictor's natural range and scale.
> The same caveat applies to sklearn's mean-decrease-in-impurity scores
> (`pip_sklearn` for rforest / xgboost): they are biased toward
> continuous and high-cardinality predictors. To rank predictors by
> their actual contribution to model fit, use permutation importance:
>
> ```python
> model.plot("pip")                # AUC drop (classification) or R² drop (regression) per predictor
> pip_df = model.plot("pip", ret=True)[1]    # the importance scores as a DataFrame
> ```
>
> PIP measures the drop in model performance when each predictor is
> randomly shuffled. The scores are on a common scale and **are**
> comparable across predictors and even across model types (rforest,
> xgboost, mlp, logistic).

This disclaimer is mandatory at the **end** of any walkthrough — even when the user did not explicitly ask "which variable matters most". Students often draw that conclusion implicitly from PDP shapes (saying things like *"X was most strongly associated with the outcome"* based on a steeper PDP), and this disclaimer is what prevents the wrong takeaway.

---

## Style notes

- **Always split into train/test** with `rsm.model.make_train(strat_var=<rvar>)`. XGBoost has no OOB; you need a held-out test for honest evaluation.
- **Don't trust the summary's AUC for classification (or R² for regression).** It is computed in-sample on the training data. Report the **test-set AUC** in any writeup.
- **Tune `learning_rate` × `n_estimators` together.** They trade off. Don't tune one in isolation.
- **Prefer permutation importance (`pip`) over xgboost's built-in (`pip_sklearn`).**
- **State cost and margin** for any profit / ROME / threshold report.
- **Don't over-engineer.** Plain top-level code the student can copy.
- **One step at a time.** Fit → summary → wait. Then offer importance / PDP. Then perf metrics on the test set.

## Growing this skill

This skill is one of a family of `pyrsm` model skills (`pyrsm-regress`, `pyrsm-logistic`, `pyrsm-rforest`, `pyrsm-mlp`). The workflow shape — absolute path → load → split → fit → summary → importance → predict on test → perf metrics — is shared. When `xgboost` isn't the right tool, route to the sibling skill.
