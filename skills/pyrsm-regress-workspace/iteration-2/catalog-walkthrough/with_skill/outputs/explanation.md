# Catalog regression — plain-English walkthrough

We fit an ordinary least squares (OLS) regression on the `catalog` dataset
(200 customers, the past year of mail-order apparel sales). The model is:

```
Sales ~ Income + HH_size + Age
```

Units, straight from the dataset description file:

- **Sales** — total sales to a household in the past year, in **dollars**.
- **Income** — household income in **thousands of dollars** (so `Income = 75`
  means $75,000/year).
- **HH_size** — number of people in the household (a count).
- **Age** — age of the head of the household, in **years**.

---

## 1. Is the model doing anything at all? (the F-test)

> **H0:** All regression coefficients (on Income, HH_size, Age) are equal to zero.
> **Ha:** At least one of them is not zero.

The F-statistic is **32.325 on (3, 196) degrees of freedom**, and the
p-value is **< 0.001**. Because that p-value is well below 0.05, we
**reject H0** and conclude that the model as a whole **does** explain a
significant amount of variation in Sales. Translation: at least one of
Income, HH_size, and Age really is moving with Sales — this isn't a
collection of pure noise.

## 2. How much does the model explain? (R-squared)

R-squared is **0.331** and adjusted R-squared is **0.321**. So the model
explains roughly **33% of the variance in Sales** across these 200
households. Two-thirds of the variation in Sales is still down to things
the model can't see — promotions, individual taste, season, life events,
catalog received, and so on. RMSE is **$88.24**, which is the typical
size of a prediction error in dollars.

## 3. The coefficients, one at a time

The fitted equation is:

```
Sales = 45.36 + 1.775 * Income + 22.122 * HH_size + 0.449 * Age
```

Two reminders before we read the coefficients:

- Each coefficient is interpreted **holding all the other variables in
  the model constant**. That's the whole point of multiple regression —
  it isolates the partial effect of one variable from the others.
- Significance is judged at the **5% level** (p < 0.05). The `***` next
  to a row in the summary means p < 0.001.

### Intercept = 45.36 (p = 0.294)

The fitted Sales for a hypothetical household with Income = 0, HH_size =
0, Age = 0. That's not a real household, so we usually don't interpret
the intercept on its own — it's just where the regression plane crosses
the y-axis. Its p-value of 0.294 simply tells us we can't distinguish it
from zero given the noise.

### Income — coefficient 1.775, p < 0.001 (significant)

> For a **$1,000 increase in household income**, we expect, on average,
> sales to **rise by about $1.78**, holding HH_size and Age constant.

Because p < 0.001, we **reject** the null hypothesis that the Income
coefficient is zero. Income is a strong, statistically significant
predictor of catalog sales. Equivalently: a $10,000 income gap predicts
about a $17.75 difference in annual catalog spending, all else equal.

### HH_size — coefficient 22.122, p < 0.001 (significant)

> Comparing two households that differ by **one extra person**, we
> expect catalog sales to be about **$22.12 higher per year** in the
> larger household, holding Income and Age constant.

Again p < 0.001, so we **reject** the null. Household size has the
biggest dollar-per-unit effect of the three predictors — bigger
households simply buy more apparel.

### Age — coefficient 0.449, p = 0.559 (NOT significant at 5%)

> A **one-year-older** head of household is associated with about
> **$0.45 more in annual sales**, holding Income and HH_size constant.

But the p-value is 0.559 — far above 0.05 — so we **fail to reject**
the null hypothesis that the Age coefficient is zero. With this sample
of 200 customers, we **don't have enough evidence to say Age has an
effect on Sales** that's distinguishable from zero, once we've already
accounted for Income and household size.

## 4. Why we are KEEPING Age in the model (read this carefully)

The most common mistake at this point is to look at Age (p = 0.559) and
say "let's drop it." Don't. Here is why we are leaving it in:

- **Non-significance is not the same as "no effect."** All p = 0.559
  tells us is that, in *this* sample of 200, we don't have enough
  evidence to distinguish Age's coefficient from zero. It does **not**
  mean Age has no real effect on spending — only that we can't pin down
  its sign and size precisely with this data.
- **Dropping a variable can bias the coefficients you keep — that's
  omitted variable bias (OVB).** If Age happens to be correlated with
  Income or HH_size (and in real demographic data those correlations
  are almost never zero — older heads of household tend to have
  different incomes and household sizes than younger ones), then
  removing Age forces its lingering effect to be absorbed by Income
  and HH_size. The numbers and the story both shift, and not for the
  better.
- **The right time to think about simplifying is *after* the model is
  finalized**, and only when there's a real reason — deployment cost,
  communication, dimensionality. "p > 0.05" by itself is not a reason.
  With three predictors and 200 observations, simplification isn't
  needed here at all.

So: report the model as fit, with all three predictors. Note the
non-significance of Age in the writeup — that's an honest finding —
but do not drop it.

## 5. Putting the whole story in one paragraph

Catalog spending in this sample is meaningfully predicted by household
income and household size: every additional $1,000 of income is worth
about **$1.78** in extra annual sales, and each additional household
member is worth about **$22.12** in extra annual sales, holding the
other variables constant. Both effects are highly statistically
significant (p < 0.001). The estimated effect of Age (about $0.45 per
year of age) is small and not statistically distinguishable from zero in
this sample (p = 0.559) — but we keep Age in the model because removing
it could bias the Income and HH_size coefficients. Together the three
predictors explain roughly **33%** of the variation in annual catalog
sales (R-squared = 0.331, F(3, 196) = 32.33, p < 0.001).
