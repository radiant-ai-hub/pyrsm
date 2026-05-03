# Diamonds: linear vs log-log regression of price on carat and clarity

Dataset: `/Users/vnijs/gh/pyrsm/examples/data/data/diamonds.parquet`
n = 3,000 round-cut diamonds, price in US dollars (\$338--\$18,791), carat in
weight (0.20--3.00), clarity ordered worst-to-best as I1, SI2, SI1, VS2, VS1,
VVS2, VVS1, IF.

---

## 1. Linear specification: `price ~ carat + clarity`

Estimates (n = 3,000):

| variable        | coefficient | p.value |
| --------------- | ----------: | ------: |
| Intercept       |   -6,780.99 | < 0.001 |
| carat           |    8,438.03 | < 0.001 |
| clarity[SI2]    |    2,790.76 | < 0.001 |
| clarity[SI1]    |    3,608.53 | < 0.001 |
| clarity[VS2]    |    4,249.91 | < 0.001 |
| clarity[VS1]    |    4,461.96 | < 0.001 |
| clarity[VVS2]   |    5,109.48 | < 0.001 |
| clarity[VVS1]   |    5,027.67 | < 0.001 |
| clarity[IF]     |    5,265.17 | < 0.001 |

R^2 = 0.904, adjusted R^2 = 0.904, F(8, 2991) = 3,530, p < 0.001, RMSE = \$1,224.

The model as a whole is highly significant. Every coefficient is significant
at the 5% level, so **we keep them all in the model**. Reading the linear
table the obvious way:

- One additional carat is associated with about \$8,438 higher price, holding
  clarity constant.
- Compared to an I1 (worst) clarity diamond, every other clarity grade has a
  higher predicted price by between \$2,791 (SI2) and \$5,265 (IF), holding
  carat constant.

So far so good — the F-test passes, R^2 is high, every t-stat is huge. But
this is **exactly the situation the skill warns about**: a model can produce
sensible-looking coefficients while violating the assumptions that justify
their interpretation. We have to look at the residual dashboard before we
trust any of these numbers.

## 2. Why the residuals look bad in the linear model

See `dashboard_linear.png`. The TA was right — three things are visibly off:

1. **Predicted-vs-actual is curved, not a tight diagonal.** At low predicted
   prices the cloud is below the 45-degree line; at high predicted prices it
   fans upward and to the right. That curvature is the classic signature of a
   missed non-linearity — `price` is not a linear function of `carat`.
2. **Residuals vs predicted shows a megaphone (funnel).** Residual variance
   is small for cheap diamonds and grows dramatically for expensive ones —
   textbook **heteroscedasticity**. OLS is unbiased here, but the standard
   errors (and therefore every t-statistic and confidence interval in the
   table above) are wrong.
3. **Q-Q plot has heavy right-tail deviations.** Residuals are not normal —
   a small number of extremely large positive residuals dominate. Combined
   with the funnel, this confirms the linear specification is mis-shaped.

The economic intuition makes the same point. A 1-carat increase from
0.3 -> 1.3 carat does not add the same dollar amount as a 1-carat increase
from 1.5 -> 2.5 carat. Diamond pricing is inherently multiplicative: a higher
clarity grade or a heavier stone scales price up by a *factor*, not by a
fixed dollar amount. A linear model in dollar units is the wrong shape.

## 3. The log-log fix

To convert a multiplicative process into something linear we work in logs.
Two columns get a natural-log transform:

```python
df = df.with_columns(
    price_ln=pl.col("price").log(),
    carat_ln=pl.col("carat").log(),
)
```

`clarity` stays as a categorical — there's no log of a category. The dummy
coefficients now describe additive shifts in **log-price**, which translate
to multiplicative (% ) shifts in price.

### Log-log estimates

`ln(price) ~ ln(carat) + clarity`, n = 3,000:

| variable        | coefficient | p.value |
| --------------- | ----------: | ------: |
| Intercept       |       7.802 | < 0.001 |
| carat_ln        |       1.809 | < 0.001 |
| clarity[SI2]    |       0.444 | < 0.001 |
| clarity[SI1]    |       0.591 | < 0.001 |
| clarity[VS2]    |       0.749 | < 0.001 |
| clarity[VS1]    |       0.792 | < 0.001 |
| clarity[VVS2]   |       0.946 | < 0.001 |
| clarity[VVS1]   |       1.011 | < 0.001 |
| clarity[IF]     |       1.080 | < 0.001 |

R^2 = 0.966, adjusted R^2 = 0.966, F(8, 2991) = 10,688, p < 0.001, RMSE = 0.187
(in log-price units).

Caveat: R^2 from the two models is not directly comparable because the
response variables differ (price vs ln(price)). The right comparison is the
**residual dashboard** — see below.

### Why the dashboard is much better now

See `dashboard_loglog.png`:

- Predicted-vs-actual is now a tight, straight, diagonal cloud.
- Residuals vs predicted look like a horizontal band of roughly constant
  width — heteroscedasticity is largely gone.
- The Q-Q plot is much closer to a straight line.

So the log-log specification is a real improvement, not just cosmetic.

## 4. Interpreting the carat elasticity

For a `ln(y) ~ ln(x)` specification, the coefficient on `ln(x)` is an
**elasticity**: the percentage change in `y` per 1% change in `x`.

> **carat_ln coefficient = 1.809.**
>
> A 1% increase in carat weight is associated with approximately a **1.81%
> increase in price**, holding clarity constant. Doubling the carat (a 100%
> increase) multiplies price by 2^1.809 = 3.50 — i.e. roughly tripling and a
> half. A diamond cut from a stone twice as heavy is much more than twice
> as expensive, which is exactly what jewelers and the rare-large-stone
> economics tell you.

Note that the elasticity is **greater than 1** — price is *elastic* in
carat. That captures the well-known scarcity premium for large diamonds.

## 5. Interpreting the clarity dummies — exact %-change table

In a model where the response is `ln(price)`, a dummy coefficient β translates
into a percentage change in price of

> **100 * (exp(β) - 1) %**

This is the *exact* form. The shorthand "100 * β %" is only an approximation
that gets worse as β grows. With clarity coefficients reaching above 1.0, the
two diverge a lot. Reference level is **I1** (worst clarity). Holding carat
constant, the price premium relative to an I1 diamond is:

| clarity level | β       | exact 100*(exp(β)-1)% | shorthand 100*β% |
| ------------- | ------: | --------------------: | ---------------: |
| SI2           |  0.4444 |          **+55.96%**  |          +44.44% |
| SI1           |  0.5910 |          **+80.59%**  |          +59.10% |
| VS2           |  0.7488 |         **+111.44%**  |          +74.88% |
| VS1           |  0.7919 |         **+120.76%**  |          +79.19% |
| VVS2          |  0.9457 |         **+157.47%**  |          +94.57% |
| VVS1          |  1.0106 |         **+174.73%**  |         +101.06% |
| IF (best)     |  1.0804 |         **+194.60%**  |         +108.04% |

Read this as: "Compared to an I1-clarity diamond of the same carat, an
IF-clarity diamond is associated with a **194.6% higher price** — roughly
**2.95x** the price." (Multiplicative form: price ratio = exp(1.0804) = 2.946.)

Two pedagogical takeaways:

1. The exact form (55.96%) and the shorthand (44.44%) for SI2 differ by ~12
   percentage points. For β > ~0.1 the shorthand is misleading; always use
   the exponential form when reporting numbers.
2. The β's are monotone in clarity quality (I1 < SI2 < SI1 < ... < IF), as
   the description says they should be. The model didn't have that ordering
   built in — it discovered it from the data, which is reassuring.

## 6. Bottom line

- Linear `price ~ carat + clarity`: technically passes the F-test with high
  R^2, but the residual dashboard is **bad** (curved fit, fanning residuals,
  non-normal). The reported standard errors and t-stats cannot be trusted at
  face value, even though the point estimates are roughly in the right ball
  park.
- Log-log `ln(price) ~ ln(carat) + clarity`: dramatically better residuals;
  coefficients translate cleanly into a carat elasticity (~1.81) and clarity
  price premia ranging from +56% (SI2 vs I1) to +195% (IF vs I1). This is the
  specification to interpret and report.

Your TA was right to flag the linear residuals — and the log-log refit is the
standard fix for this exact pattern.
