# pyrsm.model.mlp — reference

This file is the deeper reference for `pyrsm.model.mlp`. The main `SKILL.md` walks the workflow at a high level; come here for API details, the feature-scaling story, hyperparameter tuning, the local-optima caveat, and worked examples.

## Table of contents

1. Constructor signature
2. Feature scaling — pyrsm's automatic z-scoring
3. `summary()` — what each block prints (and the in-sample trap)
4. `predict()` — scaled-input scoring
5. `plot()` — feature importance and PDP
6. Classification vs regression
7. Hyperparameter tuning
8. The local-optima issue and seed sensitivity
9. Classification performance evaluation
10. Worked examples
11. Common pitfalls

---

## 1. Constructor signature

```python
rsm.model.mlp(
    data,                            # pl.DataFrame, pd.DataFrame, or {"name": df}
    rvar=None,                       # response variable
    lev=None,                        # positive level (classification only)
    evar=None,                       # list of predictor columns
    hidden_layer_sizes=(5,),         # tuple — one hidden layer with 5 neurons
    alpha=0.0001,                    # L2 regularization
    activation="tanh",               # "identity", "logistic", "tanh", "relu"
    solver="lbfgs",                  # "lbfgs", "sgd", or "adam"
    batch_size="auto",               # used by sgd/adam
    learning_rate_init=0.001,        # initial learning rate (sgd/adam only)
    max_iter=1_000_000,              # convergence iterations cap
    random_state=1234,               # reproducibility (and initialization seed)
    mod_type="classification",       # or "regression"
    cv=None,                         # sklearn CV object — pull best_params_
    **kwargs,                        # forwarded to sklearn MLPClassifier/MLPRegressor
)
```

Returns a fitted `mlp` instance — trained inside `__init__`, no separate `.fit()` call.

If `cv` is supplied with a `best_params_` attribute, pyrsm pulls `hidden_layer_sizes`, `alpha`, `activation`, `solver`, and `learning_rate_init` from those best params.

### Internals

- Numeric columns are z-scored using `pyrsm.utils.scale_df` with the training set's means and sds (stored on the model object as `self.means` and `self.stds`).
- Categorical / string / Enum columns are dummy-encoded with `drop_first=True` (since the NN has a bias term).
- The fully prepared design matrix is stored as `self.data_onehot`.
- The scaled raw matrix (before dummies) is `self.data_std`.

## 2. Feature scaling — pyrsm's automatic z-scoring

This is the **defining MLP-specific concept**:

NN models are sensitive to the absolute scale of input features. With unscaled inputs:
- Features on a large numeric scale (e.g., price in dollars: 0–10⁶) dominate features on a small scale (e.g., age: 0–100) in the gradient updates.
- The optimizer struggles to balance their contributions.
- Convergence is slow or stuck in poor local minima.

Pyrsm fixes this automatically:

```python
self.data_std, self.means, self.stds = scale_df(
    self.data.select([self.rvar] + self.evar), sf=1, stats=True
)
```

All numeric columns become z-scored: `x_scaled = (x - mean) / sd`. The means and sds are computed on the training data and stored. When `predict()` is called on new data, the **training-set means and sds** are used to scale the new data — preserving the same scale.

### Practical implications

- **The user does NOT need to manually scale features** before calling `mlp(...)`. Pyrsm does it.
- **`mlp.predict(new_df)` will use the training-set scaling**, even for new data with a different empirical distribution.
- **The estimation data preview in the summary shows the scaled (z-scored) values.** If you see values like `-2.04` for age, that's z-scored age, not negative age.
- **Categoricals are not z-scored** — they're dummy-encoded (0/1).
- **If you compare two MLP runs on the same data**, you should get identical scaling. If you compare across different train/test splits, the scaling will differ slightly because the means and sds change.

### Why pyrsm does this and Random Forest / XGBoost don't

Tree-based models split on values (not weighted sums), so the scale doesn't matter — only the relative ordering of values. NN models multiply weights by inputs, so absolute scale matters.

## 3. `summary()` — what each block prints (and the in-sample trap)

```python
mlp_model.summary(dec=3)
```

Output:

- **Header**: data, rvar, lev (classification), evar, model type.
- **Feature counts**: `(original, dummy-encoded)`.
- **Number of weights**: total trainable parameters. For an MLP with `(5,)` and 4 input features (after dummies): 4·5 + 5 + 5·1 + 1 = 31 weights (input-to-hidden + hidden bias + hidden-to-output + output bias).
- **Hyperparameters**: hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate_init, max_iter, random_state.
- **Performance** (in-sample on training data):
  - Classification: AUC.
  - Regression: R² and RMSE.
- **Raw data preview** (5 rows from training data before scaling).
- **Estimation data preview** (5 rows after scaling and dummy encoding — useful to confirm scaling happened).

### The in-sample-AUC trap

Like XGBoost, the AUC printed in the summary is computed on the training data. **MLP has no OOB or natural held-out estimate.** Always evaluate on a held-out test set:

```python
df = df.with_columns(pred_mlp=mlp_model.predict(df).get_column("prediction"))
test_df = df.filter(pl.col("training") == 0)
print("Test AUC:", rsm.model.perf.auc(rvar=test_df[rvar], pred=test_df["pred_mlp"], lev=lev))
```

## 4. `predict()` — scaled-input scoring

```python
mlp_model.predict(
    data=None,                       # new data; if None, uses training data
    cmd=None,                        # counterfactual dict
    data_cmd=None,                   # row-wise overrides
    dec=None,                        # rounding
)
```

Returns a `pl.DataFrame` with predictor columns + `prediction` column.

Internally:
1. The input is z-scored using `self.means` and `self.stds` (the training-set statistics).
2. Categoricals are dummy-encoded using `self.categories` (the training-set categorical levels).
3. The scaled, dummified matrix is passed to the sklearn MLP for prediction.

This means:
- **Predicting on training rows gives in-sample predictions** (overfit-friendly).
- **Predicting on test rows gives an honest evaluation**.
- **Predicting on data with categorical levels not seen in training** can fail or produce NaN — make sure new data has the same levels.

For classification, `prediction` is `P(rvar = lev)`. For regression, the continuous value.

## 5. `plot()` — feature importance and PDP

```python
mlp_model.plot(
    plots,                           # "pip", "pdp", "pdp_sklearn", "pred"
    nobs=1000,
    incl=None, excl=None,
    fix=True,
    hline=False,
    nnv=20,
    minq=0.025, maxq=0.975,
    ret=False,
)
```

| `plots=` | What it shows |
| --- | --- |
| `"pip"` | Permutation importance (AUC drop per shuffled feature). `ret=True` returns the importance DataFrame. **Recommended.** |
| `"pdp"` | Partial dependence plots. |
| `"pdp_sklearn"` | sklearn.inspection PDP (sometimes computes faster). |
| `"pred"` | Predicted-value curves. |

**Note**: unlike `rforest` and `xgboost`, MLP does NOT support `pip_sklearn` — sklearn's MLP classes don't expose a `feature_importances_` attribute. Permutation importance (`pip`) is the only built-in option, which is fine because it's also the correct importance metric for cross-predictor comparisons.

## 6. Classification vs regression

Same `mlp` class; differs only in `mod_type=` and `lev=`.

- Classification → `MLPClassifier`. Output layer: 1 unit with sigmoid (binary). Predictions are `P(rvar = lev)`.
- Regression → `MLPRegressor`. Output layer: 1 unit (linear). Predictions are continuous.

For multi-class (>2), pyrsm doesn't directly support it; use sklearn directly.

## 7. Hyperparameter tuning

### The knobs

1. **`hidden_layer_sizes`** — main capacity knob. Try `(3,)`, `(5,)`, `(10,)`, `(20,)`, `(10, 5)`, `(20, 10)`. Default `(5,)`. Going wider before deeper is usually safer.
2. **`alpha`** — L2 regularization on the weights. Larger = stronger regularization. Try `1e-5, 1e-4, 1e-3, 1e-2`.
3. **`activation`** — `"tanh"` is a good default for small tabular data; `"relu"` is faster and better for larger / deeper networks.
4. **`solver`** — `"lbfgs"` for small data (< ~5000), `"adam"` for larger.
5. **`learning_rate_init`** — only for `sgd` / `adam`. Default `0.001` is fine.

### Grid search

```python
param_grid = {
    "hidden_layer_sizes": [(3,), (5,), (10,), (10, 5)],
    "alpha": [1e-4, 1e-3, 1e-2],
}
cv = rsm.model.cross_validation(mlp_model, "mlp-cv", param_grid, {"AUC": "roc_auc"})

mlp_tuned = rsm.model.mlp(
    data={"<dataset> (train)": df.filter(pl.col("training") == 1)},
    rvar=rvar, lev=lev, evar=evar,
    random_state=1234,
    **cv.best_params_,
)
```

### Tuning advice

- Start with the smallest architecture and weakest regularization (`(3,)`, `alpha=1e-5`). Build up only if test performance suggests more capacity is needed.
- **Avoid grid-searching over `random_state`.** Instead, fix the seed during tuning and afterward check seed sensitivity with a few rerolls (see Step 8 of SKILL.md).
- For class assignments, a `4 × 3 = 12` grid (4 hidden sizes × 3 alphas) with 5-fold CV is 60 fits — usually fast.

## 8. The local-optima issue and seed sensitivity

NN training is **non-convex** — the loss surface has multiple local minima. Different `random_state` values lead to different starting points and (potentially) different final models.

### Symptoms

- Test AUC varies between runs by 0.02+ even with the same hyperparameters.
- Predictions for individual rows differ noticeably between runs.
- The `pip` importance ranking shifts.

### Diagnostics

```python
seeds = [1234, 42, 100, 2024, 7777]
aucs = []
for seed in seeds:
    m = rsm.model.mlp(data={...}, rvar=rvar, lev=lev, evar=evar,
                     random_state=seed, **cv.best_params_)
    test_pred = m.predict(test_df).get_column("prediction")
    auc = rsm.model.perf.auc(rvar=test_df[rvar], pred=test_pred, lev=lev)
    aucs.append(auc)
print(f"Min: {min(aucs):.3f}, Max: {max(aucs):.3f}, Range: {max(aucs)-min(aucs):.3f}")
```

A range > 0.02 indicates substantial seed sensitivity. Mitigations:
- Increase `alpha` (more regularization → smoother loss surface).
- Increase `max_iter` (let the optimizer run longer to escape shallow minima — though with `lbfgs` and default `max_iter=1_000_000` this rarely binds).
- Average predictions across several seeds:

```python
preds = []
for seed in seeds:
    m = rsm.model.mlp(data={...}, random_state=seed, **cv.best_params_)
    preds.append(m.predict(df).get_column("prediction").to_numpy())
import numpy as np
mean_pred = np.mean(preds, axis=0)
```

For a class assignment, a single seed is fine if you state it. For a deployment-grade model, an ensemble of seeds is more reliable.

## 9. Classification performance evaluation

Same pattern as `rforest` / `xgboost`:

```python
df = df.with_columns(pred_mlp=mlp_model.predict(df).get_column("prediction"))

# Test-set metrics
test_df = df.filter(pl.col("training") == 0)
rsm.model.perf.evalbin(test_df, rvar=rvar, lev=lev, pred="pred_mlp",
                       cost=1, margin=10)

# Train vs test gains
dct = {"train": df.filter(pl.col("training") == 1),
       "test":  df.filter(pl.col("training") == 0)}
rsm.model.perf.gains_plot(dct, rvar=rvar, lev=lev, pred="pred_mlp")
```

See `pyrsm-logistic` references §11 for the full perf API.

## 10. Worked examples

### Classification — Titanic

```python
import polars as pl
import pyrsm as rsm

titanic = pl.read_parquet("<abs-path>/titanic.parquet")
titanic = titanic.with_columns(
    training=rsm.model.make_train(titanic, strat_var="survived", test_size=0.3, random_state=1234)
)

clf = rsm.model.mlp(
    data={"titanic (train)": titanic.filter(pl.col("training") == 1)},
    rvar="survived", lev="Yes", evar=["pclass", "sex", "age"],
    hidden_layer_sizes=(5,), activation="tanh", solver="lbfgs",
    random_state=1234,
)
clf.summary()
# Nr. of weights: 31
# In-sample AUC: ~0.87

# Test AUC
titanic = titanic.with_columns(pred=clf.predict(titanic).get_column("prediction"))
test = titanic.filter(pl.col("training") == 0)
print(rsm.model.perf.auc(rvar=test["survived"], pred=test["pred"], lev="Yes"))
# Test AUC: ~0.85
```

### Tune

```python
param_grid = {
    "hidden_layer_sizes": [(3,), (5,), (10,), (10, 5)],
    "alpha": [1e-4, 1e-3, 1e-2],
}
cv = rsm.model.cross_validation(clf, "mlp-titanic", param_grid, {"AUC": "roc_auc"})
print(cv.best_params_)

clf_tuned = rsm.model.mlp(
    data={"titanic (train)": titanic.filter(pl.col("training") == 1)},
    rvar="survived", lev="Yes", evar=["pclass", "sex", "age"],
    random_state=1234,
    **cv.best_params_,
)
clf_tuned.summary()
```

### Seed sensitivity check

```python
seeds = [1234, 42, 100, 7777, 9999]
aucs = []
for seed in seeds:
    m = rsm.model.mlp(data={"titanic (train)": titanic.filter(pl.col("training") == 1)},
                     rvar="survived", lev="Yes", evar=["pclass", "sex", "age"],
                     random_state=seed, **cv.best_params_)
    test_pred = m.predict(test_df).get_column("prediction")
    aucs.append(rsm.model.perf.auc(rvar=test_df["survived"], pred=test_pred, lev="Yes"))
print(f"Min: {min(aucs):.3f}, Max: {max(aucs):.3f}, Range: {max(aucs)-min(aucs):.3f}")
```

### Regression

```python
diamonds = pl.read_parquet("<abs-path>/diamonds.parquet")
diamonds = diamonds.with_columns(
    training=rsm.model.make_train(diamonds, test_size=0.3, random_state=1234)
)

reg = rsm.model.mlp(
    data={"diamonds (train)": diamonds.filter(pl.col("training") == 1)},
    rvar="price",
    evar=["carat", "cut", "color", "clarity", "depth", "table"],
    hidden_layer_sizes=(20, 10),
    activation="tanh", solver="lbfgs",
    alpha=1e-3,
    mod_type="regression",
    random_state=1234,
)
reg.summary()
```

## 11. Common pitfalls

- **Trying to manually scale features before passing to `mlp(...)`.** Pyrsm does this internally. Pre-scaling is harmless but redundant.
- **Not splitting into train/test.** MLP has no OOB; honest evaluation requires a held-out set.
- **Reporting the summary's in-sample AUC as the headline.** Same as XGBoost — always report test AUC.
- **Starting with a huge architecture** like `(100, 50, 25)`. With small data, this overfits trivially. Start with `(3,)` or `(5,)`.
- **Ignoring seed sensitivity.** NN training is non-convex; different seeds give different models. Always set `random_state` and consider checking 2–3 seeds.
- **Using `solver="adam"` on small data.** `adam` needs many gradient updates to converge; `lbfgs` is faster and more reliable for n < 5000.
- **Trying to use `pip_sklearn`.** Not supported for MLP (sklearn's MLP doesn't expose `feature_importances_`). Use `pip` (permutation) instead.
- **Forgetting that `predict()` z-scores new data using training-set stats.** If new data has substantially different distributional properties than training, the predictions may be unreliable.
- **Forgetting the dict wrapper.** `mlp(df, ...)` works but the summary prints `"Not provided"`. Use `mlp({"<name>": df_train}, ...)`.
- **Treating MLP as automatically the "best" model.** For small-to-medium tabular data, tree-based models (Random Forest, XGBoost) typically beat MLP after tuning and are more interpretable. Use MLP when you have lots of data and/or when you want a baseline NN.
- **Not stating cost / margin** for profit-based metrics.
