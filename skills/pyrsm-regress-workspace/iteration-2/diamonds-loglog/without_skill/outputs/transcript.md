# Transcript

## User

> can you regress price on carat and clarity for me? data is at
> /Users/vnijs/gh/pyrsm/examples/data/data/diamonds.parquet . my TA said the
> residuals will look bad and i should try a log-log version, can you check?

## Assistant

Loaded the parquet file (3,000 rows; `clarity` is an ordered categorical with
levels I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF). I set `I1` as the reference
level so each clarity coefficient reads as the premium over the worst grade.

Fit two OLS models with `statsmodels`:

1. `price ~ carat + clarity`
2. `log(price) ~ log(carat) + clarity`

For each I produced a 2x2 diagnostic dashboard (residuals vs fitted, Normal
Q-Q, scale-location, residual histogram) and ran Breusch-Pagan and
Jarque-Bera tests.

### Linear model

R-squared = 0.904. But the diagnostics are bad, exactly as your TA warned:

- Residuals fan out as fitted price grows (clear megaphone in
  residuals-vs-fitted and scale-location).
- Breusch-Pagan p-value ~ 1e-108 — heteroskedasticity is overwhelming.
- Residuals are right-skewed (skew = 1.18) with heavy tails (excess
  kurtosis = 6.0). Q-Q plot bends up sharply at the top end. Jarque-Bera
  p-value is effectively zero.
- The model also implies negative prices for small diamonds, a tell-tale
  sign the functional form is wrong on a strictly-positive, right-skewed
  outcome.

### Log-log model

R-squared = 0.966. Diagnostics are dramatically better:

- Residual spread is roughly constant across fitted values; the funnel is
  gone. Breusch-Pagan LM statistic drops from ~1000 to ~125 (still
  technically significant in a sample this large, but practically much
  closer to homoskedastic).
- Skew = -0.24, excess kurtosis = 0.17. Q-Q plot is essentially on the
  reference line.
- `log(carat)` coefficient = 1.81 — the price elasticity of carat. A 1%
  increase in carat is associated with a ~1.81% increase in price holding
  clarity fixed. Bigger diamonds are disproportionately more expensive
  (rarity premium).
- Clarity dummies (vs I1) are positive and monotone in clarity grade; IF
  vs I1 corresponds to roughly a 195% price premium at fixed carat
  (`exp(1.08) - 1`).

### Verdict

Your TA is right. Use the log-log specification — it's well-behaved,
interpretable as elasticities, and matches the underlying multiplicative
structure of diamond pricing. See `dashboard_linear.png` vs
`dashboard_loglog.png` for a visual side-by-side, and `analysis_output.txt`
for the full regression tables and tests.

## Files written

All in `/Users/vnijs/gh/pyrsm/skills/pyrsm-regress-workspace/iteration-2/diamonds-loglog/without_skill/outputs/`:

- `analysis.py`
- `analysis_output.txt`
- `dashboard_linear.png`
- `dashboard_loglog.png`
- `explanation.md`
- `transcript.md`
