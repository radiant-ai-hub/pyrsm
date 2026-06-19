# pyrsm.model.logistic — reference

This file is the deeper reference for `pyrsm.model.logistic`. The main
`SKILL.md` walks the workflow at a high level; come here for API details,
interpretation templates, model-fit metrics, and worked examples.

## Table of contents

1. Constructor signature
2. `summary()` — what each option does
3. `predict()` — generating predictions
4. `plot()` — what each plot is for
5. Odds-ratio interpretation templates
6. Categorical predictors and dummy coding
7. Model-fit metrics — chi-squared, pseudo R-squared, AUC
8. Permutation importance — why and how
9. Model simplification and the OVB check
10. Common pitfalls

---

## 1. Constructor signature

```python
rsm.model.logistic(
    data,           # polars/pandas DataFrame, OR {"name": df}
    rvar=None,      # response variable (str, must be binary/categorical)
    lev=None,       # positive level to predict (str or int)
    evar=None,      # explanatory variables (list[str])
    ivar=None,      # interactions, e.g. ["Income:Age"]
    formula=None,   # alternative to rvar/evar
)
```

**Always pass data as `{"name": df}`** — the `name` becomes the dataset label
in the printed summary. Plain `df` works, but the summary prints `"Not provided"`.

`pyrsm` accepts polars DataFrames natively. Categorical / string / Enum columns
are auto-converted to sorted Enums so dummy levels are stable across runs.

`lev` must match exactly one level of `rvar`. For a 0/1 integer column, use
`lev=1`. For a string column like `"churn"` with levels `Yes` / `No`, use
`lev="Yes"`.

If both `formula` and (`rvar`, `evar`) are provided, `formula` wins.

---

## 2. `summary()` — what each option does

```python
clf.summary(
    test=None,    # str or list[str]: variable names to chi-squared test
    ci=False,     # include 95% confidence intervals for each OR
    vif=False,    # variance inflation factors (multicollinearity)
    dec=3,        # decimal places
)
```

Output structure:

- **Header**: dataset name, response variable, level, explanatory variables,
  null/alt hypothesis statements.
- **Coefficient table**: `index | OR | OR% | coefficient | std.error | z.value | p.value | stars`
  - `OR` — odds ratio = exp(coefficient)
  - `OR%` — percent change in odds = (OR − 1) × 100
  - `coefficient` — log-odds coefficient (the raw GLM output)
- **Significance codes**: `***` < 0.001, `**` < 0.01, `*` < 0.05, `.` < 0.1
- **Model-fit block**: Pseudo R-squared (McFadden), Pseudo R-squared (McFadden
  adjusted), AUC, Log-likelihood, AIC, BIC, Chi-squared test on the model,
  number of observations.

For a first look, `clf.summary()` with defaults is sufficient. Add:
- `test="<var>"` to run a chi-squared test comparing the full model vs. the
  model without that variable.
- `ci=True` if the user wants confidence intervals for each OR.
- `vif=True` if you suspect multicollinearity.

---

## 3. `predict()` — generating predictions

```python
clf.predict(
    data=None,      # new data; if None, uses the training data
    cmd=None,       # dict of values to vary, e.g. {"Income": [50, 100, 150]}
    data_cmd=None,  # dict to override columns in `data` row-wise
    ci=True,        # add 2.5% / 97.5% confidence interval columns
    conf=0.95,      # confidence level
    dec=3,          # rounding for float columns
)
```

Two main modes:

- **Score new data**: `data=<new_df>`. The new DataFrame must contain all
  `evar` columns. Returns predicted probability of `lev` for each row.
- **Counterfactual / scenario**: `cmd={"<var>": [v1, v2, v3]}`. Other
  variables are held at their means (numeric) or modes (categorical). Useful
  for "predict the probability of purchase for customers earning $50K, $75K,
  $100K."

`ci=True` is mutually exclusive with `data_cmd`.

Output columns: the `evar` columns used in the scenario, `prediction`
(probability of `lev`), `2.50%`, `97.50%` (if `ci=True`).

---

## 4. `plot()` — what each plot is for

```python
clf.plot(
    plots="or",     # see options below
    incl=None,      # include only these variables
    excl=None,      # exclude these variables
    nobs=1000,      # subsample for distribution / scatter plots
    ice=False,      # add ICE lines (pdp plots only)
    ice_nobs=100,
    nnv=20,         # number of x grid points for pred / pdp
    minq=0.025,     # x range = quantiles [minq, maxq]
    maxq=0.975,
    ret=None,       # for "pip": if True, return (figure, importance_df)
)
```

| `plots=` value | Purpose | Use when |
|----------------|---------|----------|
| `"or"` | Odds-ratio forest plot with CIs | Visualize direction and significance of each predictor |
| `"dist"` | Distribution of every variable in the model | First look; spot skew or unusual patterns |
| `"corr"` | Correlation matrix among response + predictors | Spot multicollinearity |
| `"pdp"` | Partial dependence plots (add `ice=True` for ICE) | How predicted probability changes across predictor values |
| `"pip"` | Permutation importance bar chart | Rank predictors by contribution to AUC |

**No residual diagnostic plots.** Logistic regression does not have the same
residual diagnostics as OLS. Use the `"dist"` and `"corr"` plots to understand
the data, and AUC + chi-squared for model fit.

---

## 5. Odds-ratio interpretation templates

Use these verbatim (substituting names and values) when walking a student
through the summary table. **Always tie the units back to the sidecar
description file** if available.

### Continuous predictor

> For a one-`<unit>` increase in `<predictor>`, the odds of
> `<rvar> = <lev>` are multiplied by `<OR>` (a `<|OR%|>`%
> `<increase|decrease>`), holding all other variables in the model constant.

Examples:

- "For a one-year increase in passenger age, the odds of surviving are
  multiplied by 0.966 (a 3.4% decrease), holding passenger class and sex
  constant."
- "For a $1,000 increase in annual income, the odds of purchasing the
  product increase by 12.3%, holding all other variables constant."

### Categorical predictor level

> Compared to `<reference_level>` `<unit>`, the odds of `<rvar> = <lev>`
> for `<level>` `<unit>` are `<|OR%|>`% `<higher|lower>`, holding all
> other variables in the model constant.

Examples:

- "Compared to 1st-class passengers, the odds of surviving for 3rd-class
  passengers were 89.8% lower, holding sex and age constant."
- "Compared to female passengers, the odds of surviving for male passengers
  were 91.7% lower, holding passenger class and age constant."

### Stating the reference level

The reference level for a categorical predictor is the alphabetically first
level (because `pyrsm` sorts levels into an Enum). Always state it explicitly.
If the user wants a different reference, they need to recode the column before
fitting.

### Significance verdict

> The odds ratio for `<predictor>` is `<significant|not significant>` at the
> 5% level (p = `<p.value>`). We `<reject|fail to reject>` the null hypothesis
> that the odds ratio equals 1.

---

## 6. Categorical predictors and dummy coding

Same as linear regression: a k-level categorical produces k−1 dummy rows in
the table, formatted `<var>[<level>]`.

**F-testing a categorical variable as a whole:**

```python
clf.summary(test="<var_name>")
```

This runs a chi-squared test comparing the full model vs. the model without
any of that variable's dummies. Use it when the question is "does `<var>`
matter as a whole?" — especially useful for categorical variables with many
levels where individual p-values are mixed.

---

## 7. Model-fit metrics

### Chi-squared test on the model

> H₀: All slope coefficients are zero.
> Hₐ: At least one slope coefficient is not zero.
> Chi-squared = `<X>`, df = `<k>`, p-value `<p>`.
> We `<reject|fail to reject>` H₀ and conclude the model as a whole
> `<does|does not>` explain a statistically significant amount of variation.

### Pseudo R-squared (McFadden)

McFadden's pseudo R-squared = 1 − (log-likelihood_full / log-likelihood_null).
Range: [0, 1]. Not directly comparable to the R-squared from linear regression.
Rough guide: values above 0.2 are considered adequate; values above 0.4 are
considered excellent.

Do not say "the model explains `<value * 100>`% of the variance." Instead:

> The McFadden pseudo R-squared is `<value>`. This is not the same as the
> R-squared from linear regression — it is based on log-likelihood ratios.
> A value of `<value>` is `<below/above>` the typical 0.2 threshold for
> adequate model fit.

### AUC (Area Under the ROC Curve)

AUC is the probability that a randomly chosen positive case receives a higher
predicted probability than a randomly chosen negative case.

- 0.5 = chance (no better than random).
- 0.7–0.8 = acceptable discrimination.
- 0.8–0.9 = excellent discrimination.
- Above 0.9 = outstanding (may indicate overfitting; check).

Interpretation template:

> The AUC is `<value>`. The model correctly ranks `<value * 100>`% of randomly
> chosen `<positive>`/`<negative>` pairs by predicted probability.

---

## 8. Permutation importance — why and how

### Why not odds ratios or z-values for importance?

Odds ratios are on different scales for predictors with different units and
coding. Three concrete reasons:

1. **Scale dependence.** A one-unit change in `age` (one year) is not
   comparable to a one-unit change in a categorical dummy. A large OR for a
   continuous predictor may just reflect a large range in the data.
2. **Reference level sensitivity.** Categorical predictors produce multiple
   ORs; a single OR describes one contrast, not the variable's overall effect.
3. **Multicollinearity.** When predictors are correlated, individual ORs
   understate each predictor's marginal contribution.

### How permutation importance works

For each predictor in turn:
1. Randomly shuffle that predictor's values.
2. Re-evaluate AUC with the shuffled predictor.
3. Importance = baseline AUC − shuffled AUC.

A larger drop = the predictor was contributing more information.

```python
clf.plot("pip")               # bar chart
pip_scores = clf.plot("pip", ret=True)[1]  # DataFrame: variable | importance
```

The importance DataFrame has columns `variable` and `importance`, sorted
descending by importance.

### Narration template

> The permutation importance plot shows `<top_var>` as the most important
> predictor (AUC drop ≈ `<value>`), followed by `<second_var>` (≈ `<value>`).
> `<bottom_var>` contributes the least. This ranking can differ substantially
> from what odds ratios or p-values suggest, especially when predictors are on
> different scales.

---

## 9. Model simplification and the OVB check

The same logic as linear regression applies. The short version:

- Do not drop a predictor because its p-value is high.
- If simplification is genuinely warranted, drop one predictor at a time,
  snapshot the odds ratios before and after, and check for large shifts.

OVB triggers for logistic regression:

- `|percent_shift_in_OR| > 10%`
- OR sign flips (from > 1 to < 1, or vice versa)
- Significance status flips across the 5% line

```python
import polars as pl

baseline = clf.coef.select(["index", "OR", "p.value"]).clone()
new_evar = [v for v in clf.evar if v != "<dropped_var>"]
clf2 = rsm.model.logistic({clf.name: clf.data}, rvar=clf.rvar, lev=clf.lev, evar=new_evar)
new_coef = clf2.coef.select(["index", "OR", "p.value"]).rename({"OR": "new_OR", "p.value": "new_p"})
comparison = (
    baseline.rename({"OR": "old_OR", "p.value": "old_p"})
    .join(new_coef, on="index", how="inner")
    .with_columns(
        pct_shift=((pl.col("new_OR") - pl.col("old_OR")) / pl.col("old_OR").abs() * 100).round(1),
        sign_flip=(pl.col("new_OR").sign() != pl.col("old_OR").sign()),
        sig_flip=((pl.col("old_p") < 0.05) != (pl.col("new_p") < 0.05)),
    )
)
triggers = comparison.filter(
    (pl.col("pct_shift").abs() > 10) | pl.col("sign_flip") | pl.col("sig_flip")
)
print("OVB triggers:" if triggers.height > 0 else "No OVB triggers — drop appears safe.")
print(triggers if triggers.height > 0 else comparison)
```

If any trigger fires, present the four options: re-include, relabel, combine,
or acknowledge the bias explicitly.

---

## 10. Common pitfalls

- **Numeric `rvar` with many values.** `logistic` is for binary/categorical
  responses. For a continuous `rvar`, use `rsm.model.regress` instead.
- **Not setting `lev`.** If `lev` is wrong, all odds ratios and predictions
  will be for the opposite outcome. Always confirm with the user before fitting.
- **Comparing odds ratios across predictors.** Different scales, different
  reference levels, multicollinearity — odds ratio size is not importance. Use
  `clf.plot("pip")`.
- **Forgetting the dict wrapper.** `logistic(df, ...)` works but the summary
  prints `"Not provided"`. Use `logistic({"<name>": df}, ...)` for clean output.
- **"The model is significant, so all predictors matter."** The chi-squared
  test tells you the model as a whole is useful; it says nothing about any
  individual predictor's importance or practical significance.
- **Treating pseudo R-squared like R-squared.** They are not on the same scale.
  A McFadden pseudo R-squared of 0.30 does not mean "explains 30% of variance."
  Report it with the caveat above.
- **Dropping non-significant predictors before the OVB check.** See Section 9.
  The same risks apply as in linear regression.

---

## 11. Classification performance evaluation (perf submodule)

The model's `summary()` reports *fit* metrics (AUC, pseudo R-squared, AIC, BIC).
For *decision-relevant* performance metrics, the `pyrsm.model.perf` submodule
exposes a rich set of functions for binary classification. All of them take a
DataFrame with the predictions attached and the response/level/prediction
column names as arguments.

### Attaching predictions

```python
pred = clf.predict()
data_with_pred = clf.data.with_columns(pl.Series("prediction", pred["prediction"]))
```

Then pass `data_with_pred` to any perf function as `df=`, with `rvar=<response>`,
`lev=<positive_level>`, `pred="prediction"`.

### Core functions

| Function | Returns | What it computes |
| --- | --- | --- |
| `evalbin(df, rvar, lev, pred, cost, margin, scale)` | `pl.DataFrame` (1 row) | All metrics in one row: TP, FP, TN, FN, total, TPR (recall), TNR, precision, F-score, accuracy, kappa, profit, lift index, ROME, contact rate, AUC |
| `confusion(df, rvar, lev, pred, cost, margin)` | `(TP, FP, TN, FN, contact)` tuple | Counts and contact rate at the profit-max threshold |
| `auc(rvar, pred, lev, weights)` | float | Area under the ROC curve |
| `profit_max(df, rvar, lev, pred, cost, margin, scale)` | float | Maximum profit at the optimal threshold |
| `ROME_max(df, rvar, lev, pred, cost, margin)` | float | Maximum ROME (return on marketing expenditure) at the optimal threshold |

### Decile-based table and plot functions

| Function | Returns | What it plots / shows |
| --- | --- | --- |
| `gains_tab(df, rvar, lev, pred, qnt=10)` | `pl.DataFrame` | Cumulative gains per decile of predicted probability |
| `gains_plot(df, rvar, lev, pred, qnt=10)` | plotnine ggplot | Gains chart — cumulative response captured by contact depth |
| `lift_tab(df, rvar, lev, pred, qnt=10)` | `pl.DataFrame` | Lift per decile |
| `lift_plot(df, rvar, lev, pred, qnt=10)` | plotnine ggplot | Lift chart — ratio of response rate at each depth vs. overall rate |
| `profit_tab(df, rvar, lev, pred, qnt=10, cost, margin, scale)` | `pl.DataFrame` | Profit per decile |
| `profit_plot(df, rvar, lev, pred, qnt, cost, margin, scale)` | plotnine ggplot | Profit chart — total profit at each contact depth |
| `expected_profit_plot(df, rvar, lev, pred, qnt, cost, margin, scale)` | plotnine ggplot | Expected profit per customer at each depth |
| `ROME_tab(df, rvar, lev, pred, qnt=10, cost, margin)` | `pl.DataFrame` | ROME per decile |
| `ROME_plot(df, rvar, lev, pred, qnt, cost, margin)` | plotnine ggplot | ROME chart — ROI per dollar spent at each depth |

`qnt` defaults to 10 (deciles); pass `qnt=20` for vigintiles, etc.

### Cost / margin / scale parameters

All profit-related functions take three business parameters:

- `cost` — per-contact cost (the cost of reaching out to one customer).
  Default 1.
- `margin` — per-success margin (the marginal revenue from one successful
  contact = one customer who responds). Default 2.
- `scale` — scale factor for the response rate (used when the test sample
  is a fraction of the full target population). Default 1.

The profit per contact is `margin × P(success) − cost`. Profit-max is the
probability threshold at which the marginal customer is exactly break-even;
above the threshold, expected profit is positive.

Always **state cost and margin explicitly** in any writeup — they are business
inputs, not statistical constants. Changing them changes the profit-max
threshold and the optimal contact depth.

### Interpretation templates

> **AUC**: The AUC of `<auc>` means the model correctly ranks `<auc * 100>`% of
> randomly chosen positive/negative pairs. An AUC of 0.5 is random; 0.7 is
> "ok"; 0.8 is "good"; 0.9 is "excellent" — these are rules of thumb.

> **Profit-max**: At the optimal threshold (predicted probability `≥ <p_max>`),
> contacting the resulting `<contact_rate * 100>`% of customers yields a total
> profit of `$<profit_max>` under `cost = $<cost>` and `margin = $<margin>` per
> contact / success.

> **Gains chart**: By contacting the top `<X>%` of customers by predicted
> probability, we capture `<Y>%` of the actual positives. The diagonal is the
> random baseline; a curve well above it indicates the model is useful.

> **Lift**: A lift of `<L>` at the top decile means customers in the top 10%
> by predicted probability are `<L>x` more likely to be positive than the
> average customer. Lift declines as we contact more customers; the profit
> question is when the marginal lift no longer covers the marginal cost.

> **ROME**: A ROME of `<R>` means we earn `$<R>` of profit per dollar of
> marketing spend at this contact depth. ROME-maximizing depth is typically
> shallower than profit-maximizing depth — ROME maximizes ROI per dollar,
> profit maximizes total dollars.

### Pitfalls specific to perf

- **Reporting profit-max profit without stating `cost` and `margin`.** Changing
  these changes the optimal threshold and the reported profit. Always state both.
- **Confusion matrix at a default 0.5 threshold instead of the profit-max
  threshold.** `confusion()` uses the *profit-max* threshold by default
  (the one that maximizes profit given cost and margin). For a fixed 0.5
  threshold or a custom cutoff, attach predicted classes manually first.
- **Treating AUC as a sufficient metric.** AUC measures ranking, not
  calibration or threshold choice. For a deployment decision you also need
  the threshold (from `profit_max`) and the resulting confusion matrix.
- **Comparing AUCs across different datasets.** AUC depends on the base rate
  and the cost / margin structure of the problem; cross-dataset comparisons
  need to control for those.
- **Forgetting that gains/lift/profit/ROME plots evaluate on the training
  data by default.** For an honest out-of-sample evaluation, split the data
  first (`pyrsm.model.make_train` is the standard helper), fit on the training
  set, predict on the test set, and pass the test-set DataFrame to the perf
  functions.
