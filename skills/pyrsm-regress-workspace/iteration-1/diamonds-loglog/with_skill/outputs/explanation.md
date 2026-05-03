# Diamonds: linear vs log-log regression of price on carat and clarity

You asked for a regression of **price** on **carat** and **clarity** using the diamonds dataset, and your TA suggested the residuals would look bad and that a log-log version would do better. Below I walk through both fits, look at the residuals from each, and translate the log-log coefficients into the kind of interpretation you can actually report.

The dataset (`diamonds.parquet`, n = 3,000) ships with a description file that confirms the units we care about:

- **price** — diamond price, in **US dollars** (\$338 to \$18,791)
- **carat** — weight of the stone (0.20 to 3.00)
- **clarity** — categorical, eight ordered grades from **I1** (worst) to **IF** (best): I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF.

Since clarity is a string column, `pyrsm.model.regress` will dummy-code it automatically. The reference level is the alphabetically first one, which here is **I1** — the worst-clarity grade. Every clarity coefficient should therefore be read as "compared to an I1-clarity diamond."

---

## 1. Linear specification: `price ~ carat + clarity`

### What it says

The linear fit is a strong model on paper:

- **F-test of the model as a whole** — p < 0.001, so we reject the null that all coefficients are zero. The model explains a significant amount of variation in price.
- **R-squared ≈ 0.89** — about 89% of the variance in dollar price is captured.
- **carat ≈ +\$8,400 per carat** (p < 0.001) — a one-carat heavier stone is worth roughly \$8,400 more, holding clarity constant.
- **clarity dummies (relative to I1)** — every level is positive and highly significant: SI2 ≈ +\$2,600, SI1 ≈ +\$3,500, VS2 ≈ +\$4,200, VS1 ≈ +\$4,500, VVS2 ≈ +\$5,000, VVS1 ≈ +\$5,200, IF ≈ +\$5,400, all p < 0.001.

A heads-up that always trips students up the first time: in a *bivariate* scatter, clarity looks negatively correlated with price (better-clarity stones are on average cheaper, because in this sample they tend to be small). Once we *control for carat* in the regression, the sign flips to the intuitive direction. This is the "regression direction is not the same as the scatter direction" point.

### What the residuals show — and why your TA is right

Look at `dashboard_linear.png` (the six-panel diagnostic):

1. **Predicted vs Actual** — the cloud is *curved*, not a straight 45-degree line. The model **systematically under-predicts price for big diamonds and over-predicts for small ones.** That's a sign of missing non-linearity.
2. **Residuals vs Predicted** — a textbook **funnel / fan shape**. Residual variance grows with fitted price. Cheap stones cluster within ±\$500 of the prediction; expensive stones can be \$3,000–\$5,000 off in either direction. This is **heteroskedasticity**.
3. **Q-Q plot** — heavy tails, especially on the upper end. Residuals are far from normal.
4. **Histogram of residuals** — right-skewed.

When the residual dashboard looks like that, the standard errors, p-values, and confidence intervals from the linear fit are no longer trustworthy. The point estimates aren't necessarily wrong, but the *uncertainty around them is mis-stated.* That's exactly what your TA was warning you about.

The economic intuition lines up too: a \$100 change in price is a huge deal for a \$400 stone but rounding error for an \$18,000 stone. Modeling price *additively* in dollars treats those two as the same kind of error, which is why the residuals fan out.

---

## 2. Log-log specification: `ln(price) ~ ln(carat) + clarity`

The fix is to put both price and carat on the log scale. In pyrsm we just create the new columns in polars and refit:

```python
df = df.with_columns(
    price_ln=pl.col("price").log(),
    carat_ln=pl.col("carat").log(),
)
reg_log = rsm.model.regress(
    {"diamonds": df},
    rvar="price_ln",
    evar=["carat_ln", "clarity"],
)
```

### What it says

- **R-squared ≈ 0.965** — explains about 96.5% of the variance in **log-price**, materially better than the 89% of the linear spec.
- **carat_ln ≈ +1.81** (p < 0.001).
- **Clarity dummies (vs I1)** — all positive, significant, and ordered correctly: SI2 ≈ +0.59, SI1 ≈ +0.79, VS2 ≈ +0.94, VS1 ≈ +1.02, VVS2 ≈ +1.13, VVS1 ≈ +1.18, IF ≈ +1.26.

### What the residuals show

`dashboard_loglog.png` looks dramatically better:

- **Predicted vs Actual** — tight, straight diagonal cloud.
- **Residuals vs Predicted** — a flat random band around zero. The funnel is gone.
- **Q-Q plot** — close to the 45-degree line, modest tails only.
- **Histogram of residuals** — roughly bell-shaped.

So the diagnostics back up the choice: the log-log spec respects the assumptions OLS needs (homoskedasticity, approximate normality of residuals), while the linear spec did not.

---

## 3. Reading the log-log coefficients in plain English

This is the part that pays off the work. Two interpretation rules apply here:

### a) The carat coefficient is an **elasticity**

When both sides are in logs, the slope is dimensionless and reads as a percentage-change ratio:

> A **1% increase in carat** is associated with a **β% increase in price**, holding clarity constant.

So with **carat_ln ≈ 1.81**, the carat-price elasticity is about **1.81**. A 1% heavier stone commands roughly a 1.81% higher price; a 10% heavier stone commands roughly an 18% higher price. The fact that the elasticity is *greater than 1* means the price–carat relationship is **super-linear in carat** — which matches our intuition that a 2-carat stone is worth more than twice a 1-carat stone.

### b) The clarity dummies translate via `100 · (exp(β) − 1)%`

When the **response** is logged but the **predictor** is a dummy (0/1), the percent-change interpretation is **not** simply 100·β%. The exact formula is:

> Compared to the reference level, this level is associated with a **100 · (exp(β) − 1)%** change in price, holding all other variables constant.

The 100·β% approximation is fine for small β (say |β| < 0.1) but breaks down here, where the clarity coefficients reach 1.0+. Using the exact form:

| Clarity (vs I1) | β       | 100·(exp(β) − 1)%  | Plain-English reading                                                    |
| --------------- | ------- | ------------------ | ------------------------------------------------------------------------ |
| SI2             | +0.59   | ≈ +80%             | An SI2-clarity stone sells for about 80% more than an I1, holding carat. |
| SI1             | +0.79   | ≈ +120%            | About 120% more than an I1.                                              |
| VS2             | +0.94   | ≈ +156%            | About 156% more than an I1.                                              |
| VS1             | +1.02   | ≈ +177%            | About 177% more than an I1.                                              |
| VVS2            | +1.13   | ≈ +210%            | About 210% more than an I1.                                              |
| VVS1            | +1.18   | ≈ +225%            | About 225% more than an I1.                                              |
| IF              | +1.26   | ≈ +252%            | About 252% more — the cleanest stones command roughly a 3.5× multiple.   |

The ordering is monotonic (each step up in clarity buys a bigger price multiplier), which is exactly what we'd expect from the GIA grade scale. All clarity coefficients are significant at the 5% level (and at 0.1%).

If you're tempted to write "100·β% ≈ 126% for IF" — don't. With β = 1.26 the linear approximation under-states the true effect by about a factor of 2. Always use the exact `exp(β) − 1` form for log-response models when you're reporting numbers.

---

## 4. Bottom line for the writeup

- The linear model has a high R² but **bad residuals** — fanning, curved, heavy-tailed. The TA is right.
- The log-log model fits better (R² ≈ 0.965), and its residuals look like what OLS expects.
- The carat coefficient in the log-log model is an **elasticity ≈ 1.81** — diamonds get more than proportionally more expensive as they get bigger.
- The clarity coefficients in the log-log model are best translated to percentages via **100 · (exp(β) − 1)%**, giving an intuitive interpretation: each step up the GIA clarity ladder is associated with a meaningfully higher price multiple over an I1-clarity reference, ranging from about +80% (SI2) up to about +250% (IF).

Use the log-log spec.
