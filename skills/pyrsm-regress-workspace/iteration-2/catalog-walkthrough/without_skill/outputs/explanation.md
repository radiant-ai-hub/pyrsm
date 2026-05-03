# Catalog Sales Regression — Plain-English Walkthrough

You have 200 catalog customers. For each one we know how much they spent (`Sales`, in dollars), their household income (`Income`, in $1,000s), how many people live in their household (`HH_size`), and their `Age` (in years). The question is: **how much of customer spending can we explain using these three things, and what does each variable contribute?**

We use **ordinary least squares (OLS) regression** — the workhorse linear model. It fits a single straight-line equation:

```
Sales = b0 + b1*Income + b2*HH_size + b3*Age + error
```

OLS picks `b0, b1, b2, b3` to minimize the sum of squared prediction errors.

---

## The fitted equation

```
Sales = 45.36 + 1.78*Income + 22.12*HH_size + 0.45*Age
```

## What each coefficient means (in English)

A coefficient is the **expected change in Sales for a one-unit increase in that predictor, holding all other predictors fixed.** That "holding everything else fixed" part is the whole point of multiple regression — it's why we don't just look at simple correlations.

### Intercept = 45.36
This is the predicted Sales when Income = 0, HH_size = 0, and Age = 0. By itself it's not meaningful here (no one has income 0, household size 0, and age 0), so don't read anything into its sign or its non-significant p-value (0.294). It's just a mathematical anchor for the line.

### Income coefficient = 1.78  (p < 0.001) — **statistically significant**
For every additional **$1,000 of household income**, we expect this customer to spend about **$1.78 more in the catalog**, holding household size and age constant. The 95% confidence interval is [1.35, 2.20], so we're pretty confident the true effect is positive and somewhere in that range. The t-statistic is 8.18 — way past the rule-of-thumb threshold of 2 — so this is a strong, reliable effect.

Concretely: a customer making $100K is expected to spend about $1.78 × (100 - 50) = **$89 more** than an otherwise-identical customer making $50K.

### HH_size coefficient = 22.12  (p < 0.001) — **statistically significant**
Each **additional household member** is associated with about **$22.12 more in catalog Sales**, holding income and age constant. CI: [13.86, 30.39]. Also a strong effect (t = 5.28). Bigger households simply buy more — makes sense.

### Age coefficient = 0.45  (p = 0.559) — **NOT statistically significant**
The point estimate says each additional year of age is associated with about $0.45 more in Sales, but the standard error (0.77) is bigger than the estimate itself, and the p-value of 0.559 is nowhere near the conventional 0.05 cutoff. The 95% CI is [-1.07, 1.96] — it comfortably includes zero, which means we **cannot rule out that age has no effect at all** on Sales (after controlling for income and household size). Best read: age doesn't seem to matter once we know income and household size.

---

## How well does the model fit overall?

- **R-squared = 0.331.** Income, HH_size, and Age together explain about **33% of the variation** in customer Sales. Not amazing, but not bad — there's clearly more going on (product preferences, marketing exposure, time of year, etc.) we're not capturing.
- **Adjusted R-squared = 0.321.** Almost the same as R-squared, which means we're not just inflating fit by stuffing in junk variables.
- **F-statistic p-value = 5e-17.** This is the test that **at least one** of the predictors has a real (non-zero) effect. It's astronomically small, so yes — taken as a group, these predictors do explain Sales.

## What I'd actually report

> Income and household size are both strong, statistically significant predictors of catalog spending. Each extra $1,000 of household income is associated with about $1.78 more in Sales, and each additional household member with about $22 more, controlling for the other variables. Age does not appear to add explanatory power once income and household size are accounted for. The model explains about 33% of the variation in Sales (R² = 0.33, F-test p < 0.001, n = 200).

## A few things your prof will likely push you on

1. **"Statistically significant" ≠ "big".** Income's coefficient (1.78) is small per unit, but Income ranges from 20 to 119 — across that range it moves predicted Sales by ~$176, which is huge.
2. **"Holding everything else constant"** is the whole game in multiple regression. The Age coefficient could look very different in a simple Sales-vs-Age model — it's the *partial* effect that matters here.
3. **A non-significant variable is not "proven to have no effect"** — we just don't have enough evidence in this sample. The CI tells you the range of effects we can't rule out.
4. **R² of 0.33 is fine for cross-sectional consumer data** but always think about what's missing (omitted variables, non-linearities, interactions) before treating the coefficients as causal.
