# pyrsm.model.rforest — reference

This file is the deeper reference for `pyrsm.model.rforest`. The main `SKILL.md` walks the workflow at a high level; come here for API details, OOB / bootstrap mechanics, hyperparameter tuning, the two flavors of feature importance, and worked examples.

## Table of contents

1. Constructor signature
2. `summary()` — what each block prints
3. `predict()` — OOB vs in-sample
4. `plot()` — what each option shows
5. Classification vs regression
6. Hyperparameter tuning (`cv=` and `cross_validation`)
7. Feature importance — `pip` vs `pip_sklearn`
8. PDP — partial dependence
9. Classification performance evaluation
10. Worked examples
11. Common pitfalls

---

## 1. Constructor signature

```python
rsm.model.rforest(
    data,                            # pl.DataFrame, pd.DataFrame, or {"name": df}
    rvar=None,                       # response variable name
    lev=None,                        # positive level (classification only)
    evar=None,                       # list of predictor names
    n_estimators=100,                # number of trees
    max_features="sqrt",             # features considered at each split
    min_samples_leaf=1,              # leaf-size regularization
    max_samples=1.0,                 # bootstrap fraction per tree
    sample_weight=None,              # per-row weights (optional)
    oob_score=True,                  # compute OOB estimate
    random_state=1234,               # reproducibility
    mod_type="classification",       # or "regression"
    cv=None,                         # sklearn CV object — pull best_params_
    **kwargs,                        # forwarded to sklearn
)
```

Returns a fitted `rforest` instance — the model is trained inside `__init__`, no separate `.fit()` call.

`max_features` options:
- `"sqrt"` (default) — `sqrt(n_features)` rounded down.
- `"log2"` — `log2(n_features)` rounded down.
- An integer — exact count of features considered per split.
- A float in (0, 1) — fraction of features.

If `cv` is supplied with a `best_params_` attribute (e.g., a fitted `GridSearchCV` result), pyrsm automatically pulls `n_estimators`, `max_features`, `min_samples_leaf`, and `max_samples` from those best params, overriding the explicit constructor arguments. This makes tune-then-refit a one-liner:

```python
rf_tuned = rsm.model.rforest(data={...}, rvar=..., lev=..., evar=..., random_state=1234, **cv.best_params_)
```

## 2. `summary()` — what each block prints

```python
rf.summary(dec=3)
```

Output structure:

- **Header**: data name, response variable, level (for classification), explanatory variables, OOB flag, model type.
- **Number of features**: `(<original_count>, <dummy-encoded_count>)`. If you have one 3-level categorical, the dummy count is +2 over the original.
- **Number of observations**: total used after dropping rows with any missing predictor. Reports `nobs_dropped` if any.
- **Hyperparameters used**: `max_features` (with effective integer), `n_estimators`, `min_samples_leaf`, `max_samples`, `random_state`.
- **Performance**:
  - Classification: **OOB AUC** computed from `rf.fitted.oob_decision_function_`.
  - Regression: R² (out-of-bag) and RMSE.
- **Estimation data preview**: first 5 rows of the dummy-encoded predictor matrix actually used.

The summary is intentionally compact — Random Forest has no coefficients to print. Importance and PDP plots provide the interpretive view.

## 3. `predict()` — OOB vs in-sample

```python
rf.predict(
    data=None,                       # new data — uses in-sample if it's training rows
    cmd=None,                        # dict for counterfactual queries
    data_cmd=None,                   # row-wise overrides
    dec=None,                        # rounding
)
```

Returns a `pl.DataFrame` with the input predictor columns and a `prediction` column.

**The critical distinction:**

- **`rf.predict()` (no `data` argument)** → returns OOB predictions for the *training rows*. For each row, only the trees that did NOT include it in their bootstrap sample contribute. This is the honest training-set prediction.
- **`rf.predict(data=<training_df>)`** → returns IN-SAMPLE predictions for the training rows. ALL trees vote, including those that memorized the row during training. **Dramatically overoptimistic.**

For test rows (rows not seen during training), `rf.predict(data=test_df)` is the right call — all trees vote, and none of them have seen these rows.

### The canonical workflow

```python
df = df.with_columns(training=rsm.model.make_train(df, strat_var=rvar, test_size=0.3, random_state=1234))

# Fit on training subset
rf = rsm.model.rforest(data={"... (train)": df.filter(pl.col("training") == 1)}, ...)

# Score test rows
df = df.with_columns(pred_rf=rf.predict(df).get_column("prediction"))

# Replace TRAINING-row predictions with OOB predictions
train_idx = df.with_row_index().filter(pl.col("training") == 1).get_column("index")
df[train_idx, "pred_rf"] = rf.predict().get_column("prediction")
```

Now `df["pred_rf"]` is OOB on training rows and in-sample on test rows — both honest for their roles.

The visible signal that you've made this mistake: the training-set gains chart looks dramatically better than the test-set one. With OOB on training, the two curves should be reasonably similar (small overfitting gap is fine; huge gap is a red flag).

## 4. `plot()` — what each option shows

```python
rf.plot(
    plots,                           # str or list[str]: pred / pdp / pip / pdp_sklearn / pip_sklearn
    nobs=1000,                       # subsample size for some plots
    incl=None,                       # restrict to these predictors
    excl=None,                       # exclude these predictors
    incl_int=None,                   # interactions (rare)
    fix=True,                        # fix y-axis across panels
    hline=False,                     # add baseline horizontal line
    nnv=20,                          # grid points for predictors
    minq=0.025, maxq=0.975,          # quantile bounds for the grid
    ret=False,                       # for "pip": also return the importance scores
)
```

| `plots=` | What it shows |
| --- | --- |
| `"pip"` | Permutation importance bar chart (AUC drop per shuffled feature). `ret=True` returns the importance DataFrame. **Default importance metric.** |
| `"pip_sklearn"` | sklearn's mean-decrease-in-impurity importance. Biased toward high-cardinality and continuous predictors. |
| `"pdp"` | Partial dependence plots — one panel per predictor in `evar`. |
| `"pdp_sklearn"` | sklearn's PDP implementation. Computed slightly differently (uses `model.predict_proba` directly). |
| `"pred"` | Predicted-probability curves — like PDP but per individual prediction, useful for showing prediction ranges. |

Pass a list (e.g., `plots=["pip", "pdp"]`) to compose multiple panels.

## 5. Classification vs regression

The same `rforest` class handles both via `mod_type=`. Internally:

- `mod_type="classification"` → `sklearn.ensemble.RandomForestClassifier`. Requires `lev` to be set. `oob_decision_function_` provides the OOB probabilities; AUC is computed from those vs the actual `rvar` (mapped to 0/1 via `lev`).
- `mod_type="regression"` → `sklearn.ensemble.RandomForestRegressor`. `lev` is ignored. `oob_prediction_` provides the OOB continuous predictions; R² and RMSE are computed from those.

For regression problems where the response is right-skewed (price, sales), Random Forest *predictions* are invariant to monotonic transforms of `rvar` — but the **reported R² and RMSE** are on the transformed scale. If you fit `log(y) ~ x`, the R² reflects fit to log y; convert back if you want dollar-units RMSE.

## 6. Hyperparameter tuning (`cv=` and `cross_validation`)

### Manual GridSearchCV

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
param_grid = {
    "max_features": [1, 2, 3, 4],
    "n_estimators": [100, 200, 300, 400, 500],
}
cv = GridSearchCV(
    rf.fitted,                            # the trained sklearn model from a base rforest
    param_grid,
    scoring={"AUC": "roc_auc"},
    cv=stratified_k_fold,
    n_jobs=4,
    refit="AUC",
    verbose=0,
).fit(rf.data_onehot, rf.data.get_column(rf.rvar))

print(cv.best_params_)
print(cv.best_score_)
```

### pyrsm helper

`rsm.model.cross_validation` wraps the above with caching to disk so re-running a notebook doesn't refit:

```python
cv = rsm.model.cross_validation(rf, "rf-cv-name", param_grid, {"AUC": "roc_auc"})
```

Behind the scenes it pickles the `GridSearchCV` object to `cv-objects/rf-cv-name-cross-validation-object.pkl`. Subsequent runs load from the cache. Delete the file to refit.

### Refit with best parameters

Once you have `cv`, pass it to the constructor:

```python
rf_tuned = rsm.model.rforest(
    data={"... (train)": df.filter(pl.col("training") == 1)},
    rvar=rvar, lev=lev, evar=evar,
    random_state=1234,
    cv=cv,                              # pyrsm pulls best_params_ from this
)
# Equivalent to:
# rf_tuned = rsm.model.rforest(... , **cv.best_params_)
```

### What to tune

In order of how much they matter:

1. **`max_features`** — the most important Random Forest knob. Lower values force more variance reduction across trees (and usually better generalization); higher values memorize more. Default `"sqrt"` is a reasonable starting point but a small grid (1, 2, 3, 4 for ~5 features; sqrt, log2, 0.3, 0.5 for many features) is worth trying.
2. **`n_estimators`** — more is better up to a saturation point. 100–500 covers most cases.
3. **`min_samples_leaf`** — increase if you observe overfitting. Default 1 (no regularization).
4. **`max_samples`** — bootstrap fraction. Default 1.0 (full bootstrap). Lower values (e.g., 0.7) add more diversity across trees.

A 5-fold grid over 4 values of `max_features` and 5 values of `n_estimators` is 20 fits × 5 folds = 100 model fits, which is usually fast enough for class assignments.

## 7. Feature importance — `pip` vs `pip_sklearn`

### `pip` — permutation importance (recommended)

Shuffles each predictor's values and measures the drop in AUC (classification) or R² (regression). Implementation: `pyrsm.model.visualize.pip_plot_sk`.

Properties:
- **Model-agnostic**: gives a comparable importance scale across model types (logistic, random forest, xgboost, mlp).
- **Honest**: uses the model's prediction quality on real data, not an internal proxy.
- **Comparable across predictors**: a continuous predictor with many splits doesn't get extra credit just for being continuous.

Use `pip` for any "which predictor matters most" comparison, especially when comparing models.

### `pip_sklearn` — mean decrease in impurity (MDI)

Sums the impurity reduction at every split that uses each feature, weighted by samples reaching that node. Implementation: sklearn's `feature_importances_`.

Properties:
- **Biased toward continuous and high-cardinality predictors** — they offer more potential splits, so accumulate more importance.
- **Computed for free at fit time** — fast.
- **Not comparable across model types** — only valid within the same model.

Use `pip_sklearn` only when:
- Speed matters (very large models).
- A published baseline uses it and you're matching that.
- You want an internal "which predictors did the tree splits use" view, not a model-agnostic importance.

### Reading the values

For `pip`:
- Importance is the *drop* in AUC (or R²) when the predictor is shuffled.
- Larger drop = more important.
- Negative or near-zero importance means shuffling didn't hurt performance — the predictor is uninformative for this model (could still be predictive in a different specification).

```python
pip_df = rf.plot("pip", ret=True)[1]
# Returns: pl.DataFrame with `variable` and `importance` columns
```

## 8. PDP — partial dependence

Partial dependence shows the average prediction as a function of a single predictor, with all other predictors averaged out. For classification, the y-axis is the predicted probability of `lev`.

`rf.plot("pdp")` produces one panel per `evar`, with `nnv=20` grid points across the predictor's range (quantile-trimmed by `minq` / `maxq` to avoid extrapolation).

### Reading PDPs

- **Monotonic curve** — the predictor has a consistent direction of effect (more = higher prob, less = lower).
- **Non-monotonic curve** — the predictor interacts with others, or has a threshold / U-shape.
- **Flat curve** — the predictor is not used much (consistent with low permutation importance).
- **Saturation** — past a certain value, the effect plateaus.

PDPs reveal the *shape* of effects that a logistic regression would force to be log-linear.

### Caveats

- PDPs assume independence between predictors. If two predictors are highly correlated, the PDP averaged over the marginal may not reflect realistic counterfactuals.
- For interactions, use 2D PDPs (sklearn supports them but pyrsm doesn't expose them directly — compute via `sklearn.inspection.partial_dependence`).

## 9. Classification performance evaluation

Same as for `logistic` — route through `pyrsm.model.perf`:

```python
# Attach honest predictions (OOB on training, in-sample on test)
df = df.with_columns(pred_rf=rf.predict(df).get_column("prediction"))
train_idx = df.with_row_index().filter(pl.col("training") == 1).get_column("index")
df[train_idx, "pred_rf"] = rf.predict().get_column("prediction")

# Single-frame overall metrics
rsm.model.perf.evalbin(df, rvar=rvar, lev=lev, pred="pred_rf", cost=1, margin=10)

# Compare train vs test gains
dct = {"train": df.filter(pl.col("training") == 1),
       "test":  df.filter(pl.col("training") == 0)}
rsm.model.perf.gains_plot(dct, rvar=rvar, lev=lev, pred="pred_rf")
```

See `pyrsm-logistic` references §11 for the full perf API.

## 10. Worked examples

### Classification — Titanic

```python
import polars as pl
import pyrsm as rsm
from sklearn.model_selection import GridSearchCV, StratifiedKFold

titanic = pl.read_parquet("<abs-path>/titanic.parquet")
titanic = titanic.with_columns(
    training=rsm.model.make_train(titanic, strat_var="survived", test_size=0.3, random_state=1234)
)

# Baseline Random Forest
clf = rsm.model.rforest(
    data={"titanic (train)": titanic.filter(pl.col("training") == 1)},
    rvar="survived", lev="Yes", evar=["pclass", "sex", "age"],
    max_features=2, n_estimators=100,
)
clf.summary()
# OOB AUC: 0.809
```

Permutation importance (sex > pclass ≈ age, all positive). Tune with a small grid:

```python
param_grid = {"max_features": list(range(1, 6)), "n_estimators": [100, 200, 300, 400, 500]}
cv = rsm.model.cross_validation(clf, "clf-rf", param_grid, {"AUC": "roc_auc"})
print(cv.best_params_)

clf_tuned = rsm.model.rforest(
    data={"titanic (train)": titanic.filter(pl.col("training") == 1)},
    rvar="survived", lev="Yes", evar=["pclass", "sex", "age"],
    random_state=1234, **cv.best_params_,
)
clf_tuned.summary()
```

Evaluate honestly with OOB on training, in-sample on test:

```python
titanic = titanic.with_columns(pred_rf=clf_tuned.predict(titanic).get_column("prediction"))
train_idx = titanic.with_row_index().filter(pl.col("training") == 1).get_column("index")
titanic[train_idx, "pred_rf"] = clf_tuned.predict().get_column("prediction")

dct = {"train": titanic.filter(pl.col("training") == 1),
       "test":  titanic.filter(pl.col("training") == 0)}
rsm.model.perf.gains_plot(dct, rvar="survived", lev="Yes", pred="pred_rf")
```

### Regression

For a continuous response (e.g., diamond price):

```python
diamonds = pl.read_parquet("<abs-path>/diamonds.parquet")
diamonds = diamonds.with_columns(
    training=rsm.model.make_train(diamonds, test_size=0.3, random_state=1234)
)

reg = rsm.model.rforest(
    data={"diamonds (train)": diamonds.filter(pl.col("training") == 1)},
    rvar="price",
    evar=["carat", "cut", "color", "clarity", "depth", "table"],
    n_estimators=200, max_features="sqrt",
    mod_type="regression",
)
reg.summary()
# Reports OOB R² and RMSE
```

## 11. Common pitfalls

- **Using `rf.predict(training_df)` to evaluate training-set performance.** Returns in-sample predictions where every tree has memorized the row → gains chart looks perfect, AUC near 1.0, then crashes on test. Use `rf.predict()` (no arg) for OOB predictions instead.
- **Reporting MDI (`pip_sklearn`) as "the" feature importance.** Biased toward continuous predictors. Use `pip` (permutation) for cross-model and cross-predictor comparisons.
- **No train/test split.** OOB is honest within the training set, but a held-out test is still essential for generalization assessment. Always use `rsm.model.make_train(strat_var=rvar)`.
- **Tuning over a huge grid for a small dataset.** A 4 × 5 grid with 5-fold CV = 100 fits — fast. A 10 × 10 × 5 × 5 grid = 2500 fits — overkill for a class assignment.
- **Forgetting `random_state`.** Without it, OOB scores fluctuate between runs and Cross-Validation results aren't reproducible.
- **Mistreating regression metrics on a log-transformed response.** R² and RMSE are on the scale of the fitted response. If you log-transformed `y`, the metrics describe fit to `log(y)`, not `y` — convert back if reporting in dollar units.
- **High `oob_score` with `max_samples << 1.0` and few trees.** Too few OOB samples per row → noisy OOB estimate. With `max_samples=0.5` and `n_estimators=10`, each row has only ~5 OOB trees on average. Increase `n_estimators` or set `max_samples=1.0`.
- **Treating the bootstrap as a substitute for cross-validation across hyperparameters.** OOB gives an honest *within-hyperparameter-setting* estimate, but you still need CV to compare across hyperparameter settings.
- **Forgetting the dict wrapper.** `rforest(df, ...)` works but the summary header prints `"Not provided"`. Use `rforest({"<name>": df_train}, ...)`.
- **Not stating cost / margin** for profit / ROME metrics. They are user-supplied business parameters; profit-max and threshold depend on them.
