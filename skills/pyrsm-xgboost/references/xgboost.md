# pyrsm.model.xgboost — reference

This file is the deeper reference for `pyrsm.model.xgboost`. The main `SKILL.md` walks the workflow at a high level; come here for API details, the in-sample-AUC trap, the learning_rate × n_estimators tradeoff, the hyperparameter-tuning playbook, and worked examples.

## Table of contents

1. Constructor signature
2. `summary()` — what each block prints (and the in-sample trap)
3. `predict()` — honest scoring
4. `plot()` — feature importance and PDP
5. Classification vs regression
6. The learning_rate × n_estimators tradeoff
7. Hyperparameter tuning playbook
8. Feature importance — `pip` vs `pip_sklearn`
9. Classification performance evaluation
10. Worked examples
11. Common pitfalls

---

## 1. Constructor signature

```python
rsm.model.xgboost(
    data,                            # pl.DataFrame, pd.DataFrame, or {"name": df}
    rvar=None,                       # response variable name
    lev=None,                        # positive level (classification only)
    evar=None,                       # list of predictor names
    n_estimators=100,                # number of boosting rounds
    max_depth=6,                     # max tree depth
    min_child_weight=1,              # min sum of weights per leaf
    learning_rate=0.3,               # shrinkage applied to each tree
    subsample=1.0,                   # row subsampling per tree
    colsample_bytree=1.0,            # column subsampling per tree
    random_state=1234,               # reproducibility
    mod_type="classification",       # or "regression"
    cv=None,                         # sklearn CV object — pull best_params_
    **kwargs,                        # forwarded to xgboost.XGBClassifier/Regressor
)
```

Returns a fitted `xgboost` instance — trained inside `__init__`, no separate `.fit()` call.

If `cv` is supplied with a `best_params_` attribute (e.g., a fitted `GridSearchCV` result), pyrsm automatically pulls `n_estimators`, `max_depth`, `min_child_weight`, `learning_rate`, `subsample`, and `colsample_bytree` from those best params.

Internally:
- Classification → `xgboost.XGBClassifier` with `objective="binary:logistic"`.
- Regression → `xgboost.XGBRegressor` with `objective="reg:squarederror"`.

The fit call passes `eval_set=[(X_train, y_train)]` to enable early stopping if the user wants to pass `early_stopping_rounds` via `**kwargs`. By default, no early stopping is used and all `n_estimators` rounds run.

## 2. `summary()` — what each block prints (and the in-sample trap)

```python
xgb_model.summary(dec=3)
```

Output structure:

- **Header**: data, rvar, lev (classification), evar, model type.
- **Feature counts**: `(<original>, <after dummies>)`.
- **Observations**: nobs, nobs_dropped.
- **Hyperparameters**: n_estimators, max_depth, min_child_weight, learning_rate, subsample, colsample_bytree, random_state.
- **Performance**:
  - Classification: **AUC** (computed in-sample on training data).
  - Regression: R² and RMSE (also in-sample).

### The in-sample-AUC trap (the critical pedagogical concept)

This is the **defining caveat** of using XGBoost in pyrsm:

The AUC the summary prints is `xgboost.XGBClassifier.score(X_train, y_train)` or equivalent — i.e., it uses the trained model to predict on the *training* data. Every training row has been seen during fitting; predictions on these rows reflect substantial memorization, especially with default hyperparameters (max_depth=6, learning_rate=0.3) and small datasets.

**Unlike Random Forest, XGBoost has no OOB equivalent.** Boosting is sequential — every tree learns from every row that survives subsampling. There is no natural "trees that didn't see this row" partition.

**Always evaluate on a held-out test set.** The pattern:

```python
# In-sample AUC from summary (overoptimistic — for diagnostic use only)
xgb_model.summary()

# Honest held-out test AUC
df = df.with_columns(pred_xgb=xgb_model.predict(df).get_column("prediction"))
test_df = df.filter(pl.col("training") == 0)
test_auc = rsm.model.perf.auc(rvar=test_df[rvar], pred=test_df["pred_xgb"], lev=lev)
print(f"Test AUC: {test_auc:.3f}")
```

For Titanic with defaults (n_estimators=100, max_depth=3, learning_rate=0.1): in-sample AUC ≈ 0.89; held-out test AUC ≈ 0.83–0.85. The 0.04–0.06 gap is the memorization premium. With `max_depth=6, learning_rate=0.3` (the constructor defaults), the gap is even larger.

**Always report the test AUC, not the summary's AUC, in any writeup.**

## 3. `predict()` — honest scoring

```python
xgb_model.predict(
    data=None,                       # new data; if None, uses training data (in-sample!)
    cmd=None,                        # counterfactual dict
    data_cmd=None,                   # row-wise overrides
    dec=None,                        # rounding
)
```

Returns a `pl.DataFrame` with predictor columns + `prediction` column.

**Critical difference from Random Forest**: there is no OOB option. If you pass training rows to `predict(data=...)`, you get in-sample (memorized) predictions. Always score the test rows separately and use those for evaluation:

```python
# Score everything
df = df.with_columns(pred_xgb=xgb_model.predict(df).get_column("prediction"))
# But: predictions on training rows are NOT honest. Only test-row predictions are.
```

If you want a single dataframe with honest predictions for all rows, you need to perform cross-validation manually (fit on N-1 folds, predict on the held-out fold, repeat) — much more involved than rforest's OOB.

For classification: `prediction` is `P(rvar = lev)`. For regression: predicted continuous value.

## 4. `plot()` — feature importance and PDP

```python
xgb_model.plot(
    plots,                           # str or list[str]: pip / pip_sklearn / pdp / pdp_sklearn / pred
    nobs=1000,                       # subsample size for some plots
    incl=None,                       # restrict to these predictors
    excl=None,                       # exclude these predictors
    fix=True,                        # fix y-axis across panels
    hline=False,                     # add baseline line
    nnv=20,                          # grid points for predictors
    minq=0.025, maxq=0.975,          # quantile bounds for the grid
    ret=False,                       # for "pip": also return the importance scores
)
```

| `plots=` | What it shows |
| --- | --- |
| `"pip"` | Permutation importance (AUC drop per shuffled feature). `ret=True` returns the importance DataFrame. **Recommended.** |
| `"pip_sklearn"` | XGBoost's built-in feature importance (default `"gain"`). Biased toward features that appear early in trees. |
| `"pdp"` | Partial dependence plots — one panel per predictor. |
| `"pdp_sklearn"` | sklearn's PDP implementation. |
| `"pred"` | Predicted-probability curves. |

## 5. Classification vs regression

Same `xgboost` class for both; differs only in `mod_type=` and `lev=`.

- Classification: `XGBClassifier`, `objective="binary:logistic"`. Predictions are P(`rvar = lev`).
- Regression: `XGBRegressor`, `objective="reg:squarederror"`. Predictions are continuous.

For multi-class (>2 levels), pyrsm doesn't directly support it — collapse to binary via `lev` or use the underlying xgboost API directly with `objective="multi:softprob"`.

## 6. The learning_rate × n_estimators tradeoff

The **most consequential XGBoost knob**:

- `learning_rate` (also called `eta`): shrinks each tree's contribution. Smaller values = slower learning, more iterations needed.
- `n_estimators`: number of boosting rounds.

They trade off: lowering `learning_rate` by 10× typically requires multiplying `n_estimators` by ~5–10× to reach similar training error, but the held-out performance is usually better with the lower learning rate (less aggressive moves means less overfitting per step).

| Setting | Behavior |
| --- | --- |
| `learning_rate=0.3, n_estimators=100` (defaults) | Fast, prone to overfitting on small data |
| `learning_rate=0.1, n_estimators=300` | Safer default; commonly used |
| `learning_rate=0.05, n_estimators=600` | Conservative, better generalization |
| `learning_rate=0.01, n_estimators=3000+` | Slow, expensive; gold-standard for AUC |

**The pyrsm default `learning_rate=0.3` is the original xgboost default and is often too high for modern practice.** For class assignments, recommend starting at `learning_rate=0.1` with `n_estimators=100–300`, then tune from there.

## 7. Hyperparameter tuning playbook

### The standard knobs

1. **`learning_rate × n_estimators`** — tune together. Start with `learning_rate=0.1` and try `n_estimators ∈ [100, 200, 300]`.
2. **`max_depth`** — controls tree complexity. 2–4 is conservative; 5–8 is aggressive. **Shallower than Random Forest** because boosting amplifies fits, so deep trees overfit fast.
3. **`min_child_weight`** — minimum sum-of-weights per leaf. Larger = more regularization. Typical values: 1, 3, 5, 10.
4. **`subsample` and `colsample_bytree`** — random subsampling per tree. Both at 0.7–0.8 is a common regularization choice.

### Grid search

```python
from sklearn.model_selection import StratifiedKFold

param_grid = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 300],
}

cv = rsm.model.cross_validation(xgb_model, "xgb-cv", param_grid, {"AUC": "roc_auc"})

# Refit with tuned params
xgb_tuned = rsm.model.xgboost(
    data={"<dataset> (train)": df.filter(pl.col("training") == 1)},
    rvar=rvar, lev=lev, evar=evar,
    random_state=1234,
    **cv.best_params_,
)
```

A `3 × 3 × 3 = 27` grid with 5-fold CV is 135 fits — usually fast.

### Diagnostic-driven tuning

Don't grid-search blindly. Run the baseline, check the train–test gap, then adjust:

| Train–test AUC gap | Action |
| --- | --- |
| < 0.02 | Model isn't overfit. Can try slightly more capacity. |
| 0.02–0.05 | Reasonable. Tune learning_rate and n_estimators. |
| 0.05–0.10 | Mild overfit. Lower max_depth, raise min_child_weight. |
| > 0.10 | Heavy overfit. Drop max_depth to 2–3, learning_rate to 0.05, subsample / colsample to 0.7. |

### Early stopping

XGBoost supports early stopping via `early_stopping_rounds`. Pyrsm doesn't expose this directly in the constructor, but you can pass it via `**kwargs`:

```python
xgb_with_es = rsm.model.xgboost(
    data={...}, rvar=..., lev=..., evar=[...],
    n_estimators=500, learning_rate=0.05,
    early_stopping_rounds=20,
)
```

Note: the `eval_set` in pyrsm is the training set itself, so early stopping based on it doesn't help. To use early stopping properly, you need to construct your own validation set and call xgboost directly — beyond the scope of this skill.

## 8. Feature importance — `pip` vs `pip_sklearn`

### `pip` — permutation importance (recommended)

Shuffles each predictor and measures the AUC drop. Same implementation as for `rforest`. Model-agnostic and comparable across models.

### `pip_sklearn` — XGBoost's built-in importance

Returns one of XGBoost's internal importance metrics. The default is `"gain"`:

- **`gain`**: average gain in split quality from splits using the feature.
- **`weight`**: number of times the feature is used in any split.
- **`cover`**: average number of training rows touched by splits using the feature.

These metrics have known biases:
- **`gain`** rewards features that produce early high-value splits.
- **`weight`** rewards features that appear in many trees (good for high-cardinality predictors).
- All of them are biased toward continuous features that offer many split candidates.

For cross-predictor and cross-model importance comparisons, use `pip`. For a quick "what features did XGBoost split on" diagnostic, `pip_sklearn` is fine.

## 9. Classification performance evaluation

```python
# Score everything
df = df.with_columns(pred_xgb=xgb_model.predict(df).get_column("prediction"))

# All decision-relevant metrics on the TEST set
test_df = df.filter(pl.col("training") == 0)
rsm.model.perf.evalbin(test_df, rvar=rvar, lev=lev, pred="pred_xgb",
                       cost=1, margin=10, scale=1)

# Train–test comparison
dct = {"train": df.filter(pl.col("training") == 1),
       "test":  df.filter(pl.col("training") == 0)}
rsm.model.perf.gains_plot(dct, rvar=rvar, lev=lev, pred="pred_xgb")
rsm.model.perf.profit_plot(dct, rvar=rvar, lev=lev, pred="pred_xgb",
                          cost=1, margin=10)
rsm.model.perf.ROME_plot(dct, rvar=rvar, lev=lev, pred="pred_xgb",
                        cost=1, margin=10)
```

**The train–test gap is the central diagnostic.** With XGBoost, expect *some* gap — if training looks dramatically better than test, the model is overfit. Re-tune with more regularization.

See `pyrsm-logistic` references §11 for the full perf API.

## 10. Worked examples

### Classification — Titanic, defaults vs tuned

```python
import polars as pl
import pyrsm as rsm

titanic = pl.read_parquet("<abs-path>/titanic.parquet")
titanic = titanic.with_columns(
    training=rsm.model.make_train(titanic, strat_var="survived", test_size=0.3, random_state=1234)
)

# Baseline with conservative defaults
xgb_baseline = rsm.model.xgboost(
    data={"titanic (train)": titanic.filter(pl.col("training") == 1)},
    rvar="survived", lev="Yes", evar=["pclass", "sex", "age"],
    n_estimators=100, max_depth=3, learning_rate=0.1,
)
xgb_baseline.summary()
# In-sample AUC: ~0.89 (overoptimistic!)

# Honest test AUC
titanic = titanic.with_columns(
    pred_xgb=xgb_baseline.predict(titanic).get_column("prediction")
)
test = titanic.filter(pl.col("training") == 0)
print(rsm.model.perf.auc(rvar=test["survived"], pred=test["pred_xgb"], lev="Yes"))
# Test AUC: ~0.83
```

Now tune:

```python
param_grid = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 300],
}
cv = rsm.model.cross_validation(xgb_baseline, "xgb-titanic-cv", param_grid, {"AUC": "roc_auc"})

xgb_tuned = rsm.model.xgboost(
    data={"titanic (train)": titanic.filter(pl.col("training") == 1)},
    rvar="survived", lev="Yes", evar=["pclass", "sex", "age"],
    random_state=1234,
    **cv.best_params_,
)
xgb_tuned.summary()
```

### Train–test gains comparison

```python
titanic = titanic.with_columns(
    pred_xgb=xgb_tuned.predict(titanic).get_column("prediction")
)
dct = {"train": titanic.filter(pl.col("training") == 1),
       "test":  titanic.filter(pl.col("training") == 0)}
rsm.model.perf.gains_plot(dct, rvar="survived", lev="Yes", pred="pred_xgb")
```

A well-tuned XGBoost should show train and test gains curves that are reasonably close. A big gap means overfit.

### Regression — diamonds

```python
diamonds = pl.read_parquet("<abs-path>/diamonds.parquet")
diamonds = diamonds.with_columns(
    training=rsm.model.make_train(diamonds, test_size=0.3, random_state=1234)
)

reg = rsm.model.xgboost(
    data={"diamonds (train)": diamonds.filter(pl.col("training") == 1)},
    rvar="price",
    evar=["carat", "cut", "color", "clarity", "depth", "table"],
    n_estimators=200, max_depth=4, learning_rate=0.1,
    mod_type="regression",
)
reg.summary()
# In-sample R² and RMSE — overoptimistic; report test-set RMSE for any writeup.
```

## 11. Common pitfalls

- **Reporting the summary's AUC as "the model's AUC".** It's in-sample. Always report the held-out test AUC.
- **No train/test split.** XGBoost has no OOB; you MUST split. Without a held-out test, you can't honestly evaluate.
- **Using the pyrsm-default `learning_rate=0.3`.** Often too high. Start with `0.1`.
- **Tuning `n_estimators` without `learning_rate`.** They trade off; lower learning_rate requires more estimators.
- **Tuning `max_depth` to RF-typical depths (unlimited or 10+).** XGBoost trees should be shallow (2–6) because boosting amplifies fits. Deep trees overfit fast.
- **Trusting `pip_sklearn` (xgboost gain) as importance.** Biased; use `pip` (permutation) for cross-predictor / cross-model comparison.
- **Reporting profit / ROME without `cost` and `margin`.** They are user-supplied business parameters.
- **Comparing in-sample AUC of XGBoost (0.96) against OOB AUC of Random Forest (0.81) and concluding XGBoost is better.** This is a category error — compare apples to apples. Use test-set AUC for both.
- **Forgetting the dict wrapper.** `xgboost(df, ...)` works but the summary prints `"Not provided"`. Use `xgboost({"<name>": df_train}, ...)`.
- **Setting `early_stopping_rounds` while using the default `eval_set`.** The default eval_set is the training set itself, so early stopping triggers on training loss — useless. Construct your own validation set for proper early stopping.
- **Treating the printed in-sample AUC ≈ 1.0 as "the model is perfect".** No — it's memorizing. Test AUC will be much lower.
