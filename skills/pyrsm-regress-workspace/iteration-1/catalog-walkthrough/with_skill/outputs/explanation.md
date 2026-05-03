# Catalog sales regression — walkthrough

We fit a linear (OLS) regression with **Sales** as the response and **Income**, **HH_size**, and **Age** as the explanatory variables, on a sample of 200 households from the catalog company's customer database.

A reminder on units (from the dataset description):

- **Sales** — total dollars spent at the company in the past year
- **Income** — household income, measured in **thousands of dollars**
- **HH_size** — number of people in the household
- **Age** — age of the head of the household, in years

The fitted equation is:

```
Sales-hat = 45.359 + 1.775 * Income + 22.122 * HH_size + 0.449 * Age
```

---

## 1. Is the model doing anything at all? (the F-test)

This is always the first question. We are comparing two hypotheses about the model **as a whole**:

- **H0 (null):** all of the slope coefficients are zero — i.e., none of Income, HH_size, or Age helps predict Sales.
- **Ha (alternative):** at least one of the slope coefficients is non-zero.

The output gives us:

> F-statistic = **32.325** on (3, 196) degrees of freedom, p-value < 0.001.

Because the p-value is well below 0.05, we **reject H0** and conclude that the model as a whole explains a significant amount of variance in Sales. In plain English: yes, this set of predictors is doing real work — there is at least one variable in the model that is genuinely related to Sales.

## 2. How much variance does the model explain? (R²)

> R-squared = **0.331**, Adjusted R-squared = 0.321.

The model explains about **33.1%** of the variation in household sales. That's a meaningful chunk for messy customer-database data, but it also means roughly two-thirds of the variation in spending is driven by things we are not measuring here (preferences, prior purchase behavior, marketing exposure, etc.). Don't oversell the model — a significant F-test is **not** the same as a model that predicts perfectly.

The RMSE of **\$88.24** gives you another way to feel that — typical prediction errors for an individual household are on the order of \$88, against an average sales level of about \$265. So the model captures the average direction, but individual predictions are noisy.

## 3. Coefficient-by-coefficient interpretation

For each predictor, the null/alternative hypotheses are:

- **H0:** the coefficient on this variable is zero (it has no effect on Sales, holding the other variables fixed).
- **Ha:** the coefficient is not zero.

We test at the **5% significance level** (alpha = 0.05).

### Income — coefficient = 1.775, p < 0.001 ***

> For an increase in household income of **\$1,000**, we expect, on average, an increase in Sales of about **\$1.78**, holding HH_size and Age constant.

Why \$1,000 and not \$1? Because the description tells us Income is measured in thousands of dollars — so a "one-unit increase in Income" is really a \$1,000 jump in household income. Always anchor the interpretation to the actual unit of the variable.

This coefficient is **significant at the 5% level** (p < 0.001). We reject H0 and conclude that income has a real effect on sales. Direction makes sense: richer households spend more on apparel.

### HH_size — coefficient = 22.122, p < 0.001 ***

> For each **additional person** in the household, we expect, on average, an increase in Sales of about **\$22.12**, holding Income and Age constant.

This is **significant at the 5% level** (p < 0.001). We reject H0. Direction also makes intuitive sense — a bigger family buys more apparel from the catalog.

### Age — coefficient = 0.449, p = 0.559

> For each **additional year of age** of the head of the household, we expect, on average, an increase in Sales of about **\$0.45**, holding Income and HH_size constant.

This coefficient is **not significant at the 5% level** (p = 0.559 — well above 0.05). We **fail to reject H0**: we do not have evidence that age, on its own and after controlling for income and household size, has any effect on sales. Don't read into the small positive sign — a coefficient this far from significant is statistically indistinguishable from zero.

### Intercept — 45.359, p = 0.294

The intercept is the predicted Sales for a (hypothetical) household with Income = 0, HH_size = 0, and Age = 0. It's not really meaningful here — none of those values are inside the range of the data — and its p-value (0.294) tells us we can't distinguish it from zero either. That's fine. We don't usually interpret the intercept in problems like this; it's just there to anchor the regression line.

## 4. The "holding all other variables in the model constant" qualifier

This is the single most important phrase in regression interpretation, so it's worth saying again. When we say "for an increase in Income of \$1,000, Sales goes up by \$1.78 on average", we mean: comparing two households that differ in income by \$1,000 **but have the same household size and same age of the household head**, the one with higher income is predicted to spend \$1.78 more.

That's different from a simple scatter plot of Sales vs Income, which would also be picking up the indirect effect of income through correlated variables. Multiple regression isolates the partial effect of each predictor.

## 5. Bottom line

- The model overall is highly significant (F = 32.33, p < 0.001) and explains 33.1% of the variance in Sales.
- **Income** and **HH_size** are both highly significant predictors of household sales (p < 0.001 each), with the expected positive sign.
- **Age** is **not** a significant predictor once income and household size are accounted for. Knowing the age of the head of household doesn't add useful information for predicting how much they spend on apparel from this catalog.

---

## Want to go further?

The numbers above are only trustworthy if the regression's assumptions hold reasonably well. I haven't checked the residuals yet. If you want, I can run a quick set of diagnostic plots:

- `reg.plot("dist")` — distributions of every variable (look for skew that suggests a log transform, especially on Sales or Income).
- `reg.plot("corr")` — correlation matrix among Sales and the predictors.
- `reg.plot("dashboard")` — the six-panel residual dashboard. This is the single most important check: predicted-vs-actual, residuals vs predicted, Q-Q plot, etc. If this looks bad (fanning residuals, curvature), the conclusions above need to be reconsidered.
- `reg.plot("residual")` — residuals vs each predictor individually, useful for pinpointing where any problem is coming from.

I can also generate predictions for specific scenarios (e.g., predicted sales for a 4-person household with \$80K income at age 45) using `reg.predict(cmd={...})` if you want to use the model rather than just describe it.

Just let me know which (if any) you'd like to look at next.
