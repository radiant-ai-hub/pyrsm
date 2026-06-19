---
name: pyrsm-rforest
description: Fit and interpret Random Forest models in Python using the pyrsm library's `rforest` class — for either binary classification (`mod_type="classification"`, predict P(`lev`)) or regression (`mod_type="regression"`, predict a continuous outcome). Use this skill whenever a student or analyst wants to fit a Random Forest, get an OOB AUC / R² baseline, examine feature importance (permutation or sklearn-style mean-decrease-impurity), look at partial-dependence plots, tune `n_estimators` / `max_features` / `min_samples_leaf` with cross-validation, score new data, or evaluate classification performance (confusion, AUC, gains, lift, profit). Triggers include phrases like "fit a random forest", "tune a random forest", "feature importance from random forest", "OOB AUC", "predict churn with rforest", "compare logistic vs random forest", "PDP for the random forest", "GridSearchCV with pyrsm.model.rforest", or any tree-ensemble modeling request in a marketing/business analytics context.
---

# pyrsm random forest workflow

This skill walks the user through fitting, interpreting, tuning, and evaluating a Random Forest model using the `pyrsm.model.rforest` class — from loading the data to producing OOB AUC / R², permutation importance, PDP plots, and (for classification) profit-max / gains / lift / ROME performance metrics.

For deep reference on the `rforest` API, the OOB vs in-sample prediction trap, hyperparameter tuning via the `cv=` argument or `pyrsm.model.cross_validation`, the two feature-importance flavors, and worked examples, see `references/rforest.md`.

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

> Before we start, can you give me the **absolute path** to the data file you want to analyze? For example: `/Users/yourname/Downloads/titanic.parquet`. Any of `.parquet`, `.csv`, `.tsv`, `.feather`, `.arrow`, `.xlsx`, or `.json` will work.

If the user gives a relative or `~` path, ask for an absolute one. If the path doesn't exist, say so.

## Step 2 — Load the data and read the sidecar

Use `scripts/load_data.py`:

```bash
python scripts/load_data.py "<absolute-path-to-data>"
```

The JSON report includes the column types — critical for picking response and predictors and deciding whether to use `mod_type="classification"` (binary `rvar`) or `mod_type="regression"` (continuous `rvar`).

If a sidecar `.md` exists, read it. Generic "rforest of y on x" is much weaker than "rforest predicting Titanic survival (Yes / No) from passenger class, sex, and age".

## Step 3 — Decide what to model

Confirm the spec with the user. You need:

- **Response (`rvar`)**: one column. Binary categorical → `mod_type="classification"`. Continuous numeric → `mod_type="regression"`. Multi-class > 2 levels is not directly supported (treat as one-vs-rest in a wrapper).
- **Level (`lev`)** *(classification only)*: which level of `rvar` is the positive outcome (e.g., `"Yes"` for survival, `1` for a 0/1 column). All AUC / probability outputs will be for this level.
- **Predictors (`evar`)**: list of one or more columns. Categoricals are auto-dummy-coded internally.
- **Hyperparameters** (key ones):
  - `n_estimators` (default 100) — number of trees. More trees = more stable, slower; usually 100–500 is plenty.
  - `max_features` (default `"sqrt"`) — features considered at each split. Lower = more variance reduction across trees, higher = less. Common choices: `"sqrt"`, `"log2"`, an integer, or a float ∈ (0, 1).
  - `min_samples_leaf` (default 1) — minimum samples per leaf. Higher = more regularization, less overfitting.
  - `max_samples` (default 1.0) — fraction of training rows bootstrapped per tree.
  - `oob_score` (default `True`) — compute out-of-bag score. Always leave this on for classification.
  - `random_state` (default 1234) — for reproducibility.
- **Train/test split**: ideally always. Use `rsm.model.make_train(df, strat_var=<rvar>, test_size=0.3, random_state=1234)` to add a `training` column (1 for train, 0 for test). Fit on `df.filter(pl.col("training") == 1)`; evaluate honestly on `df.filter(pl.col("training") == 0)`.

Confirm: "I'll fit a Random Forest classifier predicting `survived = "Yes"` from pclass, sex, age. 100 trees, max_features=`sqrt`, with OOB AUC. Train on 70% (stratified by survived), evaluate on 30%."

Quick sanity checks:

- For classification: is `rvar` binary? `df[rvar].n_unique() == 2`. If not, switch tool or pre-collapse.
- For regression: is `rvar` numeric? Right-skewed responses (price, sales) may benefit from a log-transform before fitting — though Random Forest is invariant to monotonic transforms of `rvar` for ranking purposes, the metrics (R², RMSE) change scale.
- High-cardinality categoricals (>30 levels) blow up the dummified feature count. Consider grouping.
- Many missing values in `evar`: the class drops rows with any missing predictor before fitting. Note how many are dropped (`nobs_dropped` is reported in the summary).

## Step 4 — Fit the model

```python
import polars as pl
import pyrsm as rsm

# 1. Load and split
df = pl.read_parquet("<absolute-path>")
df = df.with_columns(
    training=rsm.model.make_train(df, strat_var="<rvar>", test_size=0.3, random_state=1234),
)

# 2. Fit on training data
rf = rsm.model.rforest(
    data={"<dataset> (train)": df.filter(pl.col("training") == 1)},
    rvar="<response>",
    lev="<positive_level>",          # for classification
    evar=["<x1>", "<x2>", "..."],
    n_estimators=100,
    max_features="sqrt",
    min_samples_leaf=1,
    random_state=1234,
    mod_type="classification",       # or "regression"
)

# 3. Inspect
rf.summary()
```

Notes:
- Pass the data as `{"name": df_train}` so the summary header prints `<name>`.
- For categorical `evar` columns: pyrsm dummy-codes them internally (one column per non-reference level). The summary's `Nr. of features` shows `(<original>, <after dummies>)`.
- The model is fit inside `__init__` — no separate `.fit()` call.
- For regression: omit `lev`, set `mod_type="regression"`; the summary reports R² / RMSE instead of AUC.

## Step 5 — Interpret the summary

The summary is much shorter than a regression's because there are no coefficients. Walk through:

1. **Model setup**: data, rvar, lev, evar, classification vs regression.
2. **Number of features and observations**: original-feature count vs after-dummy-encoded count. nobs and any nobs_dropped (due to missing values).
3. **Hyperparameters used**: max_features (with the effective integer count), n_estimators, min_samples_leaf, max_samples, random_state.
4. **Performance**:
   - Classification: **OOB AUC**. Interpretation: "an OOB AUC of `<value>` means the model correctly ranks `<value*100>`% of random survivor / non-survivor pairs *using OOB predictions* — i.e., for each tree, only the rows not in its bootstrap sample contribute to that tree's predictions. This is an honest training-set estimate that does not overfit."
   - Regression: R² and RMSE (also OOB-based when `oob_score=True`).

### The OOB vs in-sample prediction trap (the critical concept)

This is the **defining pedagogical concept** for Random Forest in pyrsm:

- **`rf.predict()` (no `data` argument) returns OOB predictions** — for each training row, the prediction averages only the trees in which that row was NOT in the bootstrap sample. This is the *honest* training-set prediction.
- **`rf.predict(data=training_df)` returns IN-SAMPLE predictions** — for each row, ALL trees vote, including the trees that saw the row during training. This is **dramatically overoptimistic** for Random Forest because each tree has near-perfect memory of its bootstrap sample.

**Implication**: if you score the training rows by passing the *same* DataFrame you fit on, the resulting gains / lift / AUC / confusion are not a faithful "training-set performance" — they reflect partial memorization. Use **OOB predictions** for the training-set evaluation.

The canonical pattern:

```python
# Fit on training subset
rf = rsm.model.rforest(data={"titanic (train)": df.filter(pl.col("training") == 1)}, ...)

# Score the test set normally (no overlap with training rows, so in-sample is honest)
test_pred = rf.predict(df.filter(pl.col("training") == 0))

# For the training rows: use OOB predictions (NOT rf.predict(df.filter(training==1)))
train_oob_pred = rf.predict()    # no arg → OOB predictions
```

Then attach both to the master DataFrame and pass the combined frame to `gains_plot` / `evalbin` etc.

The 30%-test baseline check: if the training and test gains curves are essentially the same shape, the model isn't overfit. If the training (in-sample) curve is much steeper than test, you're seeing the memorization artifact — switch to OOB for training.

## Step 6 — Feature importance and partial dependence

After the basic interpretation, **offer** these plots — pick what answers the user's question:

```python
rf.plot("pip")              # permutation importance (recommended; AUC-drop based)
pip_df = rf.plot("pip", ret=True)[1]   # importance as a DataFrame

rf.plot("pip_sklearn")      # sklearn's mean-decrease-in-impurity importance

rf.plot("pdp")              # partial dependence plots (one panel per evar)
rf.plot("pdp_sklearn")      # sklearn-flavored PDP (uses model.predict_proba)

rf.plot("pred")             # predicted-probability plots per predictor
```

### `pip` vs `pip_sklearn`

This is the **second critical concept** for Random Forest:

- **`pip` (permutation importance)** — shuffles each predictor's values and measures the AUC drop. **Model-agnostic, comparable across predictors and across model types.** *This is the right importance metric for cross-model comparisons.*
- **`pip_sklearn` (mean decrease in impurity, MDI)** — sums the impurity reduction at every split that uses each feature. **Biased toward high-cardinality and continuous predictors** (more splits available → more importance). *Useful as a quick within-model view, not for cross-comparison.*

Default to `pip`. Mention `pip_sklearn` only when the user explicitly asks or when comparing to a published baseline that uses MDI.

### PDP — partial dependence

Shows, for each predictor, the average prediction as a function of that predictor with all other predictors integrated out. Useful for:
- Identifying non-linear effects (a U-shape, a threshold, a saturation curve).
- Confirming the direction of effect.
- Spotting interaction-driven artifacts (if PDPs look flat but the model has high accuracy, there's likely an interaction).

For classification, the y-axis is the predicted probability of `lev`. The flatter the PDP, the less the model uses that predictor on average.

## Step 7 — Hyperparameter tuning via cross-validation

Random Forest has a handful of hyperparameters that interact:

- `n_estimators`: more is generally better up to a saturation point; rarely worth more than ~500.
- `max_features`: the most consequential. Lower = more variance reduction across trees (helps generalization), higher = more memorization.
- `min_samples_leaf`: a regularizer. Increase if overfitting.
- `max_samples`: bootstrap fraction. Default 1.0 (full bootstrap).

Use grid-search cross-validation to tune:

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {
    "max_features": list(range(1, 6)),
    "n_estimators": [100, 200, 300, 400, 500],
}

# The pyrsm helper:
cv = rsm.model.cross_validation(rf, "rf-cv-name", param_grid, {"AUC": "roc_auc"})

# Then refit with the best params:
rf_tuned = rsm.model.rforest(
    data={"<dataset> (train)": df.filter(pl.col("training") == 1)},
    rvar="<rvar>", lev="<lev>", evar=[...],
    random_state=1234,
    **cv.best_params_,         # inject the tuned hyperparameters
)
rf_tuned.summary()
```

Alternatively, fit `cv` separately (any sklearn-compatible GridSearchCV result) and pass it as `cv=cv` to the constructor — pyrsm will pull `n_estimators` / `max_features` / `min_samples_leaf` / `max_samples` from `cv.best_params_` automatically.

For class assignments, a small 5-fold grid over `max_features` (1, 2, 3, 4) is usually plenty. Always report what was searched and what was selected.

## Step 8 — Predictions

```python
# Score new data
pred = rf.predict(new_df)
# Returns a pl.DataFrame with predictor columns + 'prediction' column

# Attach predictions to the master DataFrame
df = df.with_columns(pred_rf=rf.predict(df).get_column("prediction"))

# Counterfactual / scenario predictions (held at means/modes for unspecified vars)
pred = rf.predict(cmd={"age": [10, 30, 50]})
```

`predict()` returns a polars DataFrame. For classification, the `prediction` column is `P(rvar = lev)`. For regression, it's the predicted continuous value.

**Critical**: do NOT pass the training rows to `predict(data=...)` to get the training-set predictions — that gives in-sample (memorized) predictions. Use `predict()` (no argument) for OOB predictions on training.

## Step 9 — Evaluate classification performance

For classification models, route through `pyrsm.model.perf` for decision-relevant metrics:

```python
# Step A: attach honest predictions to the master DataFrame
df = df.with_columns(pred_rf=rf.predict(df).get_column("prediction"))

# Replace training-row predictions with OOB predictions (for an honest training estimate)
train_idx = df.with_row_index().filter(pl.col("training") == 1).get_column("index")
df[train_idx, "pred_rf"] = rf.predict().get_column("prediction")   # OOB

# Step B: compute overall metrics
rsm.model.perf.evalbin(df, rvar="<rvar>", lev="<lev>", pred="pred_rf",
                       cost=1, margin=10, scale=1)
# Returns TP/FP/TN/FN, precision/recall/F-score, accuracy/kappa, profit, ROME, AUC.

# Step C: confusion matrix at the profit-max threshold
TP, FP, TN, FN, contact = rsm.model.perf.confusion(
    df, rvar="<rvar>", lev="<lev>", pred="pred_rf", cost=1, margin=10
)

# Step D: gains / lift / profit / ROME (decile-based, the marketing-analytics workhorses)
rsm.model.perf.gains_plot(df, rvar="<rvar>", lev="<lev>", pred="pred_rf")
rsm.model.perf.lift_plot(df, rvar="<rvar>", lev="<lev>", pred="pred_rf")
rsm.model.perf.profit_plot(df, rvar="<rvar>", lev="<lev>", pred="pred_rf", cost=1, margin=10)
rsm.model.perf.ROME_plot(df, rvar="<rvar>", lev="<lev>", pred="pred_rf", cost=1, margin=10)

# Step E: split-aware comparison (train vs test) on one chart
dct = {"train": df.filter(pl.col("training") == 1),
       "test":  df.filter(pl.col("training") == 0)}
rsm.model.perf.gains_plot(dct, rvar="<rvar>", lev="<lev>", pred="pred_rf")
```

**Always state cost and margin** in any profit / ROME / threshold report — they are user-supplied business parameters. Changing them changes the optimal threshold and the reported numbers.

For regression: use `rsm.model.evalreg` (or compute R² / RMSE / MAE manually) to report fit quality.

## Step 10 — Related models (when to switch)

`rforest` is a strong default for non-linear classification / regression with mixed-type predictors. If the situation is different, suggest:

- **Interpretable coefficients / odds ratios** → `logistic` (classification) or `regress` (regression).
- **Higher-capacity gradient boosting** → `xgboost` (`pyrsm-xgboost` skill). Usually higher AUC at the cost of more tuning effort.
- **Neural network for tabular** → `mlp` (`pyrsm-mlp` skill). Requires feature scaling; better when interactions are deep and the dataset is large.
- **Pure descriptive task** → `pyrsm-eda` family (no model).
- **Two-group means comparison** → `compare_means` or `compare_props`.

Don't switch silently — explain why the alternative is a better fit.

## Step 9 — Interpreting permutation importance (PIP)

When the user asks for "PIP", "permutation importance", "feature importance", "which variable matters most", or anything similar, this is a **separate task from interpreting the regression output**. Do NOT respond by re-interpreting coefficients / odds ratios. The user wants the PIP scores.

### Canonical code (ALWAYS use `ret=True`)

```python
pip_plot, pip_df = rf.plot("pip", ret=True)
print(pip_df)                                              # required: print the scores verbatim
pip_plot.save("pip.png", width=8, height=5, dpi=120)       # save the plot for visual inspection
```

`rf.plot("pip")` alone (without `ret=True`) returns only the plot — the numeric importance scores are not exposed. **Always use `ret=True`** so you get both the plot AND the polars DataFrame of (variable, importance) pairs.

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
p = rf.plot("pdp")                                       # one panel per predictor
p.save("pdp.png", width=10, height=8, dpi=120)              # save the composed grid
print("saved pdp.png")
```

For ICE lines (individual predicted curves overlaid on the average): `rf.plot("pdp", ice=True)` — supported by `regress` and `logistic`; for `rforest` / `xgboost` / `mlp`, use `plot("pdp_sklearn")` for an alternative implementation.

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

- **Always split into train/test** with `rsm.model.make_train(strat_var=<rvar>)` for classification. Evaluation on training data alone is misleading for any model, but especially so for Random Forest.
- **Use OOB predictions for training-row evaluation.** `rf.predict()` (no arg) is the right call; `rf.predict(training_df)` is the wrong one. State this explicitly when the gains chart for training looks "too good".
- **Prefer permutation importance (`pip`) over sklearn impurity-based (`pip_sklearn`).** MDI is biased toward continuous and high-cardinality predictors; permutation importance is comparable and fair.
- **State cost and margin** for any profit / ROME / threshold report.
- **Tune `max_features` first**, then `n_estimators`. `min_samples_leaf` and `max_samples` are second-order knobs.
- **Don't over-engineer.** Plain top-level code the student can copy.
- **One step at a time.** Fit → summary → wait for the user to look. Then offer importance / PDP. Then offer perf metrics.

## Growing this skill

This skill is one of a family of `pyrsm` model skills (`pyrsm-regress`, `pyrsm-logistic`, `pyrsm-mlp`, `pyrsm-xgboost`). The workflow shape — absolute path → load → split → fit → summary → importance → predict → perf metrics — is shared. When `rforest` isn't the right tool, route to the sibling skill that fits the user's task.
