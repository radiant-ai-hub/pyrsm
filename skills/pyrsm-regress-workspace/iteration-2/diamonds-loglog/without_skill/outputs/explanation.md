# Diamonds: price ~ carat + clarity (linear vs log-log)

## What I did

Loaded `/Users/vnijs/gh/pyrsm/examples/data/data/diamonds.parquet` (3,000 rows).
Treated `clarity` as a categorical variable with the natural order
`I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF` and used `I1` (worst) as the
reference level. Fit two OLS models with `statsmodels`:

1. **Linear:** `price ~ carat + clarity`
2. **Log-log:** `log(price) ~ log(carat) + clarity`

For each model I produced a 2x2 residual diagnostic dashboard (residuals vs
fitted, Normal Q-Q, scale-location, residual histogram) plus formal tests
(Breusch-Pagan for heteroskedasticity, Jarque-Bera for normality).

## Your TA was right

| Metric                       |     Linear |    Log-log |
| ---------------------------- | ---------: | ---------: |
| R-squared                    |     0.9042 |     0.9662 |
| Adj. R-squared               |     0.9040 |     0.9661 |
| Breusch-Pagan p-value        | 1.09e-108  |   2.90e-23 |
| Jarque-Bera p-value          |     ~0     |   1.13e-07 |
| Residual skewness            |       1.18 |      -0.24 |
| Residual excess kurtosis     |       6.02 |       0.17 |

### Linear model problems (see `dashboard_linear.png`)

- **Funnel-shaped residuals.** The residuals-vs-fitted and scale-location
  plots show a classic megaphone: dispersion grows sharply with fitted price.
  Breusch-Pagan rejects homoskedasticity overwhelmingly (LM stat ~ 1000+).
- **Heavy right tail.** Residual skewness 1.18 and excess kurtosis 6.0; the
  Q-Q plot bends sharply upward at the top. Jarque-Bera p-value is
  effectively zero.
- **Implied negative prices.** A linear specification on a strictly positive,
  right-skewed outcome will predict negative prices for small diamonds — a
  symptom that the functional form is wrong, not just the error distribution.

### Log-log fixes most of it (see `dashboard_loglog.png`)

- Residual variance is roughly constant across fitted values (the funnel is
  gone). Breusch-Pagan still rejects strict homoskedasticity, but the LM
  statistic drops from 1000+ to ~125 — a huge improvement.
- Q-Q plot is essentially on the line; skewness ~ -0.24, excess kurtosis
  ~ 0.17. Residuals are very close to normal.
- R-squared rises from 0.904 to 0.966 (note: not directly comparable across
  different DVs, but the diagnostic improvement is the real win).

## Interpretation of the log-log coefficients

- `log(carat)` coefficient = **1.81**. This is the **price elasticity of
  carat**: a 1% increase in carat is associated with about a 1.81% increase
  in price, holding clarity fixed. Diamond prices grow more than
  proportionally with size — consistent with the well-known "rarity
  premium" on big stones.
- Clarity dummies (relative to `I1`, the worst grade) all positive and
  monotonically increasing through the grades. The IF (internally flawless)
  premium over I1 is `exp(1.0804) - 1 ~ 195%`, holding carat fixed.

## Bottom line

Stick with the log-log specification. The linear model violates
homoskedasticity and normality so badly that its standard errors and
prediction intervals shouldn't be trusted; log-log is well-behaved and
gives a clean elasticity interpretation.

## Files

- `analysis.py` — full reproducible script
- `analysis_output.txt` — captured stdout (both regression tables, diagnostic
  tests, comparison summary)
- `dashboard_linear.png` — diagnostic dashboard for the linear model
- `dashboard_loglog.png` — diagnostic dashboard for the log-log model
