---
name: pyrsm-mlp
description: Fit and interpret a Multi-Layer Perceptron (feed-forward neural network) in Python using the pyrsm library's `mlp` class — for either binary classification (`mod_type="classification"`, predict P(`lev`)) or regression (`mod_type="regression"`, predict a continuous outcome). Use this skill whenever a student or analyst wants to fit a neural network on tabular data, tune hidden layer sizes / regularization (alpha) / solver with cross-validation, examine feature importance (permutation), look at partial-dependence plots, score new data, or evaluate classification performance (confusion, AUC, gains, lift, profit). The class automatically z-scores features before training (NN models require scaling). Triggers include phrases like "fit a neural network", "MLP classifier", "tune hidden layers", "neural network feature importance", "compare neural net vs logistic", "PDP for MLP", "neural net for tabular data", or any feed-forward NN modeling request in a marketing/business analytics context.
---

# pyrsm MLP workflow

This skill walks the user through fitting, interpreting, tuning, and evaluating a multi-layer perceptron using the `pyrsm.model.mlp` class — from loading the data to producing AUC / R², permutation importance, PDP plots, and (for classification) profit-max / gains / lift / ROME performance metrics.

For deep reference on the `mlp` API, why feature scaling is required (and how pyrsm handles it automatically), the local-optima issue, hyperparameter tuning, and worked examples, see `references/mlp.md`.

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

If the user gives a relative or `~` path, ask for an absolute one.

## Step 2 — Load the data and read the sidecar

Use `scripts/load_data.py`. Read the sidecar `.md` description if found.

## Step 3 — Decide what to model

Confirm the spec with the user. You need:

- **Response (`rvar`)**: one column. Binary categorical → `mod_type="classification"`. Continuous numeric → `mod_type="regression"`.
- **Level (`lev`)** *(classification only)*: which level of `rvar` is the positive outcome.
- **Predictors (`evar`)**: list of columns. Categoricals are auto-dummy-coded; numerics are auto-z-scored (mean 0, sd 1). **Do not pre-scale your features** — pyrsm does this for you.
- **Hyperparameters** (key ones — see Step 7 for tuning):
  - `hidden_layer_sizes` (default `(5,)`) — tuple. `(5,)` is one hidden layer with 5 neurons. `(10, 10)` is two layers of 10 each. Start small.
  - `activation` (default `"tanh"`) — `"identity"`, `"logistic"`, `"tanh"`, or `"relu"`. `tanh` is a good default for small tabular data; `relu` for larger / deeper networks.
  - `solver` (default `"lbfgs"`) — `"lbfgs"`, `"sgd"`, or `"adam"`. **`lbfgs` is the right default for small datasets** (n < ~5000); `adam` is for larger data.
  - `alpha` (default `0.0001`) — L2 regularization strength. Larger = more regularization, smoother decision boundary.
  - `learning_rate_init` (default `0.001`) — initial learning rate (only used by `sgd` and `adam`; `lbfgs` is a quasi-Newton method and ignores this).
  - `max_iter` (default `1_000_000`) — the upper bound for convergence iterations. Very high so the solver runs to convergence.
  - `random_state` (default `1234`) — reproducibility. **MLP is sensitive to this** because the optimizer can get stuck in local minima — different seeds can yield meaningfully different models. Always set explicitly.
- **Train/test split**: mandatory. NN models can overfit, and the summary reports in-sample AUC. Use `rsm.model.make_train(strat_var=<rvar>, test_size=0.3, random_state=1234)`.

### Feature scaling — the critical concept

This is the **defining pedagogical concept** for MLP:

- **NN models are sensitive to the scale of input features.** A feature on a scale of [0, 10^6] dominates a feature on [0, 1] simply because of magnitude — the optimizer can't easily balance their gradients.
- **Pyrsm scales features automatically inside `mlp.__init__`.** Numeric features get z-scored (mean 0, sd 1) using `scale_df`. The scaled feature matrix is stored as `data_std`; categoricals are then dummy-encoded.
- **You don't need to pre-scale your features** — pyrsm handles it. But you DO need to know it's happening: when `mlp.predict(data=...)` is called on new data, pyrsm uses the *training-set means and sds* to scale the new data the same way. If your new data has very different distributional properties, the predictions may be unreliable.

If the user asks "do I need to standardize my features for the MLP?", the answer is "pyrsm does it for you, but yes — feature scaling is required for NN models and is the reason `mlp` works on raw inputs that would otherwise be incompatible."

## Step 4 — Fit the model

```python
import polars as pl
import pyrsm as rsm

# 1. Load and split (mandatory for MLP)
df = pl.read_parquet("<absolute-path>")
df = df.with_columns(
    training=rsm.model.make_train(df, strat_var="<rvar>", test_size=0.3, random_state=1234),
)

# 2. Fit on training data
mlp_model = rsm.model.mlp(
    data={"<dataset> (train)": df.filter(pl.col("training") == 1)},
    rvar="<response>",
    lev="<positive_level>",          # for classification
    evar=["<x1>", "<x2>", "..."],
    hidden_layer_sizes=(5,),         # start with one small hidden layer
    activation="tanh",
    solver="lbfgs",                  # good for small datasets
    alpha=0.0001,                    # L2 regularization
    random_state=1234,
    mod_type="classification",       # or "regression"
)

# 3. Inspect
mlp_model.summary()
```

Notes:
- pyrsm internally calls `MLPClassifier` or `MLPRegressor` from sklearn, after z-scoring numeric features.
- The summary's AUC is in-sample (overoptimistic). Use the test set for honest evaluation.
- The number of weights in the model is reported (e.g., `Nr. of weights: 31`). Each weight is a parameter the optimizer tunes — more weights = more capacity = more overfitting risk.

## Step 5 — Interpret the summary

Walk through:

1. **Model setup**: data, rvar, lev, evar, classification vs regression.
2. **Feature counts**: `(original, dummy-encoded)`. Numerics are scaled (one number each); categoricals expand to dummies (k-1 per k-level categorical).
3. **Number of weights**: total trainable parameters. For a `(5,)` MLP with 4 input features: 4·5 (input→hidden) + 5 (hidden bias) + 5·1 (hidden→output) + 1 (output bias) = 31. Larger architectures (e.g., `(10, 10)`) have many more weights.
4. **Hyperparameters**: hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate_init, max_iter, random_state.
5. **Performance** (in-sample on training data):
   - Classification: AUC.
   - Regression: R² and RMSE.
6. **Raw data preview** and **Estimation data preview** (the z-scored feature matrix) — useful to confirm scaling has happened.

### The in-sample-AUC trap (same as XGBoost)

Like XGBoost, MLP has no OOB equivalent. The AUC in the summary is on the training data. **Always report the test-set AUC** in any writeup.

```python
df = df.with_columns(pred_mlp=mlp_model.predict(df).get_column("prediction"))
test = df.filter(pl.col("training") == 0)
print("Test AUC:", rsm.model.perf.auc(rvar=test["<rvar>"], pred=test["pred_mlp"], lev="<lev>"))
```

## Step 6 — Feature importance and partial dependence

MLP has no interpretable coefficients. Use permutation importance and PDPs:

```python
mlp_model.plot("pip")              # permutation importance
pip_df = mlp_model.plot("pip", ret=True)[1]

mlp_model.plot("pdp")              # partial dependence plots
mlp_model.plot("pdp_sklearn")      # sklearn-flavored PDP

mlp_model.plot("pred")             # predicted-value curves per predictor
```

`pip_sklearn` is not available for MLP (sklearn's MLP doesn't expose `feature_importances_`). Use `pip` (permutation) — which is the correct importance metric anyway.

### Interpreting PDPs for MLP

MLP can capture sharp non-linearities. PDPs are especially useful for an MLP because:
- They reveal the shape the network learned (often non-monotonic).
- Compare them to PDPs from a Random Forest or XGBoost trained on the same data — if MLP's PDP is much smoother, the NN may be under-fitted; if much more curvy, it may be over-fitted to noise.
- For categorical predictors, the PDP shows the average prediction per level — useful for confirming the dominant effects.

## Step 7 — Hyperparameter tuning via cross-validation

MLP has several interacting knobs:

- **`hidden_layer_sizes`** — the main capacity knob. Start small. Try `(3,)`, `(5,)`, `(10,)`, then maybe `(10, 5)` or `(20, 10)`.
- **`alpha`** — L2 regularization. Try `1e-5`, `1e-4`, `1e-3`, `1e-2`. Larger = more regularization.
- **`activation`** — `"tanh"` and `"relu"` are the main contenders. For small tabular data, tanh often wins.
- **`solver`** — `"lbfgs"` for n < ~5000, `"adam"` for larger. Rarely worth grid-searching.
- **`random_state`** — **always try at least 2–3 seeds**. MLP gets stuck in local minima; what looks like a hyperparameter effect can be a seed effect.

### Grid search

```python
param_grid = {
    "hidden_layer_sizes": [(3,), (5,), (10,), (10, 5)],
    "alpha": [1e-4, 1e-3, 1e-2],
}
cv = rsm.model.cross_validation(mlp_model, "mlp-cv", param_grid, {"AUC": "roc_auc"})

mlp_tuned = rsm.model.mlp(
    data={"<dataset> (train)": df.filter(pl.col("training") == 1)},
    rvar="<rvar>", lev="<lev>", evar=[...],
    random_state=1234,
    **cv.best_params_,
)
```

### Seed sensitivity check

After tuning, re-fit with 3–5 different `random_state` values and compare test AUC. If results swing by more than ~0.02, the model is sensitive to initialization — average multiple runs or increase regularization (`alpha`).

```python
seeds = [1234, 42, 100, 2024]
results = []
for seed in seeds:
    m = rsm.model.mlp(data={...}, rvar=rvar, lev=lev, evar=evar,
                     random_state=seed, **cv.best_params_)
    pred = m.predict(df.filter(pl.col("training") == 0))
    auc = rsm.model.perf.auc(rvar=df.filter(pl.col("training") == 0)[rvar],
                            pred=pred["prediction"], lev=lev)
    results.append((seed, auc))
```

## Step 8 — Predictions

```python
# Score new data (test rows here)
pred = mlp_model.predict(test_df)
# Returns pl.DataFrame with predictor columns + 'prediction'

# Attach to master DataFrame
df = df.with_columns(pred_mlp=mlp_model.predict(df).get_column("prediction"))

# Counterfactual queries
pred = mlp_model.predict(cmd={"age": [10, 30, 50]})
```

`predict()` internally z-scores the new `data` using the *training-set* means and sds, then runs the network. For classification, returns `P(rvar = lev)`. For regression, the continuous value.

**No OOB option** — like XGBoost, MLP has no honest training-set estimate. Use the test set.

## Step 9 — Evaluate classification performance

```python
# Score and attach
df = df.with_columns(pred_mlp=mlp_model.predict(df).get_column("prediction"))

# Test-set metrics (honest)
test_df = df.filter(pl.col("training") == 0)
rsm.model.perf.evalbin(test_df, rvar="<rvar>", lev="<lev>", pred="pred_mlp",
                       cost=1, margin=10, scale=1)

# Train vs test gains comparison
dct = {"train": df.filter(pl.col("training") == 1),
       "test":  df.filter(pl.col("training") == 0)}
rsm.model.perf.gains_plot(dct, rvar="<rvar>", lev="<lev>", pred="pred_mlp")
rsm.model.perf.profit_plot(dct, rvar="<rvar>", lev="<lev>", pred="pred_mlp",
                          cost=1, margin=10)
rsm.model.perf.ROME_plot(dct, rvar="<rvar>", lev="<lev>", pred="pred_mlp",
                        cost=1, margin=10)
```

**Always state cost and margin.**

For regression, use `rsm.model.evalreg` or compute R² / RMSE / MAE on the test set.

## Step 10 — Related models (when to switch)

`mlp` is a flexible tabular-data model with good capacity but less interpretability than tree models. If the situation is different, suggest:

- **Interpretable coefficients / odds ratios** → `logistic` / `regress`.
- **Tree ensemble baseline** → `rforest`.
- **High-AUC tuned model on tabular data** → `xgboost`. Often outperforms MLP on small-to-medium tabular datasets after tuning.
- **Descriptive only** → `pyrsm-eda` family.

For tabular data with <10K rows and <100 predictors, **tree-based models (RForest, XGBoost) typically beat MLP after tuning** and are more interpretable. MLP shines on larger datasets, deep interactions, and when the user wants a baseline NN.

## Step 9 — Interpreting permutation importance (PIP)

When the user asks for "PIP", "permutation importance", "feature importance", "which variable matters most", or anything similar, this is a **separate task from interpreting the regression output**. Do NOT respond by re-interpreting coefficients / odds ratios. The user wants the PIP scores.

### Canonical code (ALWAYS use `ret=True`)

```python
pip_plot, pip_df = mlp_model.plot("pip", ret=True)
print(pip_df)                                              # required: print the scores verbatim
pip_plot.save("pip.png", width=8, height=5, dpi=120)       # save the plot for visual inspection
```

`mlp_model.plot("pip")` alone (without `ret=True`) returns only the plot — the numeric importance scores are not exposed. **Always use `ret=True`** so you get both the plot AND the polars DataFrame of (variable, importance) pairs.

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
p = mlp_model.plot("pdp")                                       # one panel per predictor
p.save("pdp.png", width=10, height=8, dpi=120)              # save the composed grid
print("saved pdp.png")
```

For ICE lines (individual predicted curves overlaid on the average): `mlp_model.plot("pdp", ice=True)` — supported by `regress` and `logistic`; for `rforest` / `xgboost` / `mlp`, use `plot("pdp_sklearn")` for an alternative implementation.

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

- **Feature scaling is mandatory.** Pyrsm does this automatically; mention it when relevant so the user understands what's happening internally.
- **Always train/test split.** MLP has no OOB; the summary's AUC is in-sample.
- **Start with small architectures.** `(5,)` is a sensible baseline. Grow only if held-out AUC indicates more capacity is needed.
- **Set and report `random_state`.** MLP is local-optima-sensitive; reproducibility matters.
- **Check seed sensitivity.** If results swing wildly between seeds, the model is underregularized or the architecture is too large.
- **`solver="lbfgs"` for small data; `"adam"` for large.**
- **Use `pip` for feature importance.** MLP doesn't expose a built-in importance.
- **State cost and margin** in any profit / ROME report.
- **Don't over-engineer.** Plain top-level code the student can copy.
- **One step at a time.** Fit → summary → wait. Then offer importance / PDP. Then perf metrics on test.

## Growing this skill

This skill is one of a family of `pyrsm` model skills (`pyrsm-regress`, `pyrsm-logistic`, `pyrsm-rforest`, `pyrsm-xgboost`). The workflow shape — absolute path → load → split → fit → summary → importance → predict on test → perf metrics — is shared. When `mlp` isn't the right tool, route to the sibling skill.
