# pyrsm.model.regress — reference

This file is the deeper reference for `pyrsm.model.regress`. The main `SKILL.md` walks the workflow at a high level; come here for API details, interpretation templates, and worked examples.

## Table of contents

1. Constructor signature
2. `summary()` — what each option does
3. `predict()` — generating predictions
4. `plot()` — what each plot is for
5. `f_test()` — comparing nested models
6. Coefficient interpretation templates
7. Categorical predictors and dummy coding
8. Log-log and semi-log specifications
9. Residual diagnostics — what to look for
10. Omitted variable bias and model simplification
11. Common pitfalls

---

## 1. Constructor signature

```python
rsm.model.regress(
    data,           # polars/pandas DataFrame, OR {"name": df}
    rvar=None,      # response variable (str, must be numeric)
    evar=None,      # explanatory variables (list[str])
    ivar=None,      # interactions, e.g. ["Income:HH_size"]
    formula=None,   # alternative to rvar/evar, e.g. "Sales ~ Income * HH_size"
)
```

**Always pass data as `{"name": df}`** — the `name` becomes the dataset label in the printed summary. Plain `df` works, but the summary prints `"Not provided"`.

`pyrsm` accepts polars DataFrames natively; internally it stores polars and converts to pandas only when handing data to statsmodels. Categorical / string / Categorical / Enum columns are auto-converted to sorted Enums so dummy levels are stable across runs.

If both `formula` and (`rvar`,`evar`) are provided, `formula` wins.

## 2. `summary()` — what each option does

```python
reg.summary(
    vif=False,    # variance inflation factors (multicollinearity)
    ssq=False,    # sum-of-squares decomposition (Regression / Error / Total)
    rmse=False,   # root mean square error
    test=None,    # list of variable names to F-test against the null model
    ci=False,     # 95% confidence intervals for each coefficient
    dec=3,        # decimal places
    plain=True,   # plain-text vs styled output
)
```

**The default is plain `reg.summary()` — pass no flags unless the user explicitly asks.** The plain output already shows the coefficient table, F-statistic, R², adjusted R², n, df, and significance stars. Add flags only when the user asks: `vif=True` for multicollinearity, `ci=True` for coefficient confidence intervals, `ssq=True` for the sum-of-squares decomposition, `rmse=True` for residual spread, `test=["x1","x2"]` to F-test a subset of coefficients jointly against a nested model.

Output structure:

- **Header**: dataset name, response, explanatory variables, null/alt hypothesis statements.
- **Coefficient table**: index | coefficient | std.error | t.value | p.value | significance stars.
- **Significance code legend**: `***` < 0.001, `**` < 0.01, `*` < 0.05, `.` < 0.1.
- **Model fit**: R², adjusted R², F-statistic with df, p-value, n.
- Optional sections (VIF, SSQ, RMSE, F-test, CIs) appended in order.

## 3. `predict()` — generating predictions

```python
reg.predict(
    data=None,      # new data; if None, uses the training data
    cmd=None,       # dict of values to vary, e.g. {"Income": [50, 100, 150]}
    data_cmd=None,  # dict to override columns in `data` row-wise
    ci=True,        # add 2.5% / 97.5% confidence interval columns
    conf=0.95,      # confidence level
    dec=3,          # rounding for float columns
)
```

Two main modes:

- **Score new data**: pass `data=<new_df>`. The new DataFrame must contain all the `evar` columns. Categorical levels not seen in training will error; pyrsm reapplies the stored Enum types to keep things consistent.
- **Counterfactual / scenario**: pass `cmd={"Income": [50, 100, 150]}`. Other variables are held at their means (numeric) or modes (categorical). Useful for "predict sales for households earning $50K, $100K, $150K".

`ci=True` is mutually exclusive with `data_cmd`.

## 4. `plot()` — what each plot is for

```python
reg.plot(
    plots="dist",   # see options below
    nobs=1000,      # subsample for scatter plots; -1 for all
    incl=None,      # include only these variables (pred / coef plots)
    excl=None,      # exclude these variables
    incl_int=None,  # include these interactions (pred plots)
    fix=True,       # fix y-axis across pred panels
    hline=False,    # add response-mean horizontal line
    ice=False,      # add ICE lines (pdp plots only)
    ice_nobs=100,
    nnv=20,         # number of x grid points for pred / pdp
    minq=0.025,     # x range = quantiles [minq, maxq]
    maxq=0.975,
    figsize=None,
    ret=None,       # for "pip", return the importance DataFrame too
)
```

| `plots=` value | Purpose | Use when |
| --- | --- | --- |
| `"dist"` | Distribution of every variable in the model | First look, before fitting; spot skew |
| `"corr"` | Correlation matrix among response + predictors | Spot multicollinearity, detect non-linear / weak relations |
| `"scatter"` | Scatter of `rvar` vs each `evar` with smoothed line | Spot non-linearity |
| `"dashboard"` | Six-panel residual dashboard (predicted-vs-actual, residuals, Q-Q, residual histogram + density, residual vs row order, Cook's distance) | The single most important diagnostic |
| `"residual"` | Residuals vs each `evar` | Spot heteroscedasticity, missed non-linearity per variable |
| `"pred"` | Predicted-value plots across each predictor's range | Communicate what the model says |
| `"pdp"` | Partial dependence plots (add `ice=True` for ICE) | Effect of one predictor with others integrated out |
| `"pip"` | Permutation importance bar chart (`ret=True` returns scores) | Rank predictors by importance |
| `"coef"` | Coefficient plot with CIs | Visualize coefficient table |

## 5. `f_test()` — comparing nested models

```python
reg.f_test(test=["x1", "x2"], dec=3)
```

Tests the null hypothesis "the coefficients on the listed variables are jointly zero" by comparing the full model to the reduced model that omits them. Equivalent to passing `test=[...]` to `summary()`.

Use this when the user asks "do x1 and x2 *together* explain anything beyond the other variables?" — for example, a categorical variable with many levels (each level is a separate dummy, but the conceptual question is about the variable as a whole).

## 6. Coefficient interpretation templates

Use these templates verbatim (substituting variable names and units) when walking a student through results. **Always tie the units back to the sidecar description file** if available.

### Continuous predictor, linear response

> For a one-`<unit>` increase in `<predictor>` we expect, on average, to see a `<coefficient>`-`<unit-of-response>` `<increase|decrease>` in `<response>`, holding all other variables in the model constant.

Example: "For an increase in income of \$1,000 we expect, on average, to see an increase in sales of \$1.78, holding all other variables in the model constant."

### Significance verdict

> The coefficient on `<predictor>` is `<significant|not significant>` at the 5% level (p = `<p.value>`). We `<reject|fail to reject>` the null hypothesis that the coefficient on `<predictor>` is zero.

### F-test on the model as a whole

> H₀: All regression coefficients are equal to zero.
> Hₐ: At least one regression coefficient is not equal to zero.
> The F-statistic is `<F>` on (`<df1>`, `<df2>`) degrees of freedom with p-value `<p>`. We `<reject|fail to reject>` H₀ and conclude that the model as a whole `<does|does not>` explain a significant amount of variance in `<response>`. The R² of `<R²>` means the model explains `<R²×100>`% of the variance in `<response>`.

## 7. Categorical predictors and dummy coding

When an `evar` is a string / Categorical / Enum column, `pyrsm` dummy-codes it automatically:

- Each non-reference level becomes its own row in the coefficient table, formatted `<var>[<level>]`.
- The reference level is the alphabetically first level (because pyrsm sorts levels into an Enum before fitting). If the user wants a different reference level, they need to recode the column themselves before calling `regress`.
- A k-level categorical produces k−1 dummy rows in the table.

### Interpretation template for categorical levels

> Compared to a `<reference-level>` `<unit>`, a `<level>` `<unit>` is associated with a `<coefficient>`-`<unit-of-response>` `<higher|lower>` `<response>`, holding all other variables in the model constant.

Example: "Compared to an I1-clarity diamond, a VS1-clarity diamond is associated with a 0.792 higher log-price, holding carat constant."

### F-testing the variable as a whole

If the question is "does clarity matter?" rather than "does each clarity level matter?", run an F-test:

```python
reg.f_test(test=["clarity"])
```

(pyrsm understands the bare variable name and tests all of its dummies jointly.)

## 8. Log-log and semi-log specifications

Pyrsm doesn't have a "log" switch — log-transform the columns yourself in polars before fitting:

```python
df = df.with_columns(
    price_ln=pl.col("price").log(),
    carat_ln=pl.col("carat").log(),
)
reg = rsm.model.regress({"diamonds": df}, rvar="price_ln", evar=["carat_ln", "clarity"])
```

### Interpretation rules of thumb

| Spec | Coefficient on `x` interpreted as |
| --- | --- |
| `y ~ x` (linear-linear) | one-unit increase in `x` → β-unit increase in `y` |
| `ln(y) ~ x` (log-linear) | one-unit increase in `x` → 100·(exp(β)−1)% increase in `y` (for small β, ≈ 100β%) |
| `y ~ ln(x)` (linear-log) | 1% increase in `x` → β/100 increase in `y` |
| `ln(y) ~ ln(x)` (log-log) | 1% increase in `x` → β% increase in `y` (an elasticity) |

For categorical dummies in a `ln(y)` specification, the percentage interpretation is **100·(exp(β)−1)%**, not just 100·β%. The approximation is fine for small β (say, |β| < 0.1) but breaks down for larger coefficients — always use the exact form when reporting.

When to consider a log transform on `y`:

- The variable is strictly positive (price, sales, income, demand).
- The histogram is right-skewed.
- The residual dashboard shows fanning residuals (heteroscedasticity) or a curved predicted-vs-actual line.
- The economic interpretation is naturally multiplicative (a 1% change matters more than a $1 change).

## 9. Residual diagnostics — what to look for

Run `reg.plot("dashboard")` and read the six panels:

1. **Predicted vs Actual** — should be a tight, straight, diagonal cloud. Curvature → missing non-linearity (try a log transform or polynomial). Funnel shape → heteroscedasticity.
2. **Residuals vs Predicted** — should be a random horizontal band. Patterns = bad.
3. **Residuals vs Row order** — flat random scatter. A trend suggests autocorrelation (only really a concern for time-ordered data).
4. **Q-Q plot** — straight diagonal = normal residuals. Heavy tails or systematic curve = non-normal residuals (less critical for inference with large n thanks to CLT, but still informative).
5. **Histogram of residuals** — should look bell-shaped.
6. **Density plot of residuals vs theoretical normal** — same idea, just smoothed.

For a per-predictor view, add `reg.plot("residual")` — residuals vs each `evar` separately. Useful for spotting that one variable in particular is the source of trouble.

For multicollinearity: `reg.summary(vif=True)`. VIFs above 5 are suspicious; above 10 is a serious problem. High VIF inflates standard errors, can flip coefficient signs, and makes individual coefficients hard to interpret even when the model as a whole is fine.

## 10. Omitted variable bias and model simplification

### The intuition

When a regression includes correlated predictors and you drop one of them, the remaining predictors' coefficients change to absorb part of the dropped variable's effect. This is **omitted variable bias** (OVB), and it is the reason the skill is firm about not dropping variables on p-value alone.

The cleanest way to see it: imagine the "true" data-generating model is `y = β_x · x + β_z · z + ε`, and `x` and `z` are correlated. If you fit the (mis-)specified model `y ~ x` only, the estimated coefficient on `x` is approximately

```
β_x_estimated ≈ β_x + β_z · (cov(x,z) / var(x))
```

That second term — the **bias** — is not zero whenever `z` has a real effect (β_z ≠ 0) **and** `x` and `z` are correlated. Dropping `z` doesn't make `z`'s effect disappear; it reroutes that effect through whatever `x`s are still in the model.

In a regression on real data we don't know the "true" model, but the same mechanic applies: the coefficient of a kept predictor will move whenever a dropped predictor was both (a) correlated with it and (b) had a real partial effect on the response. The size and sign of the move tell you how much OVB you've introduced.

### When to be most worried

OVB is most dangerous in exactly the situations where students are most tempted to drop variables:

- The dropped variable is "non-significant" but **highly correlated** with another predictor in the model. Multicollinearity inflates standard errors and depresses individual significance — the variable looks irrelevant in isolation but is doing real work jointly.
- The dropped variable is a **confounder** (a common cause of both another predictor and the response). Dropping it shifts the coefficient of the predictor it confounded.
- The dataset is small (so power is low and a real effect can fail to clear the 5% bar).

If correlations among predictors are near zero (e.g., from a designed experiment), OVB risk is minimal — but in observational business data this is rare.

### The protocol Claude should walk students through

This restates Step 6 of `SKILL.md` for completeness. When simplification is genuinely warranted (not just because p > 0.05):

1. **Snapshot the current coefficients** of all retained predictors as a polars DataFrame. This is the baseline.
2. **Drop the highest-p-value predictor**, one at a time. For categoricals, drop the whole variable.
3. **Refit and compute**, for each remaining variable: `percent_shift = (new_coef − old_coef) / |old_coef| * 100`.
4. **OVB triggers** — investigate if **any** remaining variable shows:
   - `|percent_shift| > 10%`, **or**
   - the **sign** flips, **or**
   - the **significance status** flips across the 5% line.
5. **If a trigger fires, present all four remediation options** with rationales, and ask the student to pick:
   - **Re-include** the dropped variable. Default conservative choice.
   - **Relabel** a remaining variable to reflect the combined effect it now captures. ("Income now stands in for income-and-household-life-stage.") Honest but requires careful interpretation in the writeup.
   - **Combine** the correlated pair into a single composite predictor — a weighted average, simple average, index, principal component, or a domain-meaningful construct. Refit with the composite in place of the originals.
   - **Keep the dropped-variable model** with explicit acknowledgment of which kept coefficient is now biased and roughly in which direction. Only legitimate if the question being answered does not depend on that coefficient's interpretation.

### Example

In the catalog regression (`Sales ~ Income + HH_size + Age` with n = 200), `Age` has p = 0.559 — high. A naive student would drop it. The protocol says:

- Snapshot: Income = 1.775, HH_size = 22.122, Age = 0.449.
- Drop Age, refit `Sales ~ Income + HH_size`.
- Compare new Income and HH_size to baseline.
- If both are within ±10% of baseline and neither flips significance, OVB risk in this case turned out to be small — Age was approximately orthogonal to Income and HH_size. The student can drop Age **and document the OVB check as evidence that the simplification was safe**. (But also: with only three predictors, simplification is not necessary at all in this dataset — keeping Age in is fine.)

The teaching value is in the *check*, not in always reaching the same conclusion. Sometimes the check exonerates the drop; sometimes it forces a serious rethink. Either way the student now has evidence instead of a hunch.

### Helper snippet

```python
import polars as pl

# Step 1 — snapshot
baseline = reg.coef.select(["index", "coefficient", "p.value"]).clone()

# Step 2 — drop highest-p-value variable (excluding the intercept)
candidates = baseline.filter(pl.col("index") != "Intercept").sort("p.value", descending=True)
to_drop = candidates["index"][0]
print(f"Dropping {to_drop} (p = {candidates['p.value'][0]:.3f})")

# Step 3 — refit
new_evar = [v for v in reg.evar if v != to_drop]
reg2 = rsm.model.regress({reg.name: reg.data}, rvar=reg.rvar, evar=new_evar)

# Step 4 — compute percent shifts on the kept variables
new_coef = reg2.coef.select(["index", "coefficient", "p.value"]).rename({
    "coefficient": "new_coef", "p.value": "new_p"
})
comparison = baseline.rename({"coefficient": "old_coef", "p.value": "old_p"}).join(
    new_coef, on="index", how="inner"
).with_columns(
    pct_shift=((pl.col("new_coef") - pl.col("old_coef")) / pl.col("old_coef").abs() * 100).round(1),
    sign_flip=(pl.col("new_coef").sign() != pl.col("old_coef").sign()),
    sig_flip=((pl.col("old_p") < 0.05) != (pl.col("new_p") < 0.05)),
)
print(comparison)

# Step 5 — fire triggers
triggers = comparison.filter(
    (pl.col("pct_shift").abs() > 10) | pl.col("sign_flip") | pl.col("sig_flip")
)
if triggers.height > 0:
    print("\nOVB triggers fired on:")
    print(triggers)
else:
    print("\nNo OVB triggers — drop appears safe.")
```

This snippet is small enough to paste into the analysis script when the student gets to Step 6.

## 11. Common pitfalls

- **Passing a non-numeric `rvar`.** `regress` is OLS — the response must be numeric. For 0/1 outcomes use `rsm.model.logistic`; for multi-class use a different tool.
- **Forgetting the dict wrapper.** `regress(df, ...)` works but the summary header shows `"Not provided"`. Use `regress({"<name>": df}, ...)` for clean output.
- **Reading a coefficient sign without a residual check.** A regression on a model that fails the residual diagnostics can produce sensible-looking coefficients that are actually misleading. Always at least glance at `plot("dashboard")` before committing to an interpretation.
- **Confusing scatter-plot direction with regression direction.** A bivariate scatter shows the *marginal* relationship; the regression coefficient shows the relationship *holding other variables constant*. They can disagree (the diamonds-clarity example: clarity looks negatively associated with price in the scatter, but is positively associated with price in the regression once carat is controlled for). When you see this, explain it explicitly — students find it counter-intuitive at first.
- **Over-interpreting a tiny R².** A statistically significant model can still explain very little variance. Always report R² alongside the F-test.
- **Asking `predict` for confidence intervals while also using `data_cmd`.** Not supported — pyrsm raises a `ValueError`. Use `cmd` instead, or compute predictions and CIs separately.
- **Dropping a non-significant variable on the spot.** This is the single most common source of OVB in student work. Refuse to do it without walking the Step 6 protocol — snapshot, drop one at a time, compare percent shifts, and present the four remediation options (re-include / relabel / combine / acknowledge) when a trigger fires.
