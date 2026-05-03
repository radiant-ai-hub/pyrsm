# Flow Summary

1. **Located and inspected** `/Users/vnijs/gh/pyrsm/examples/data/model/catalog.parquet` — 200 rows, columns: `id, Sales, Income, HH_size, Age`. All numeric, no obvious cleaning needed.
2. **Looked at correlations** with Sales: Income (0.49) and HH_size (0.31) show notable positive associations; Age (0.04) is essentially flat.
3. **Fit an OLS regression** with statsmodels: `Sales ~ Income + HH_size + Age` on all 200 rows.
4. **Captured the standard regression output** (coefficients, std errors, t-stats, p-values, 95% CIs, R-squared, F-test) into `analysis_output.txt`.
5. **Verified** the model with a sanity-check prediction at the variable means — predicted Sales ($257.56) lands close to the sample mean ($265.48), as expected.
6. **Wrote a plain-English explanation** (`explanation.md`) walking through:
   - what each coefficient means with the "holding everything else constant" framing,
   - which predictors are significant (Income and HH_size, both p < 0.001) and which is not (Age, p = 0.56),
   - dollar-scale intuition (e.g. +$50K income → +~$89 in Sales),
   - overall fit (R² = 0.33, F-test highly significant),
   - common pitfalls a professor would push back on (significance vs. magnitude, non-significance vs. zero effect, partial vs. simple effects).

## Key results

| Predictor | Coefficient | p-value | Significant? |
|-----------|-------------|---------|--------------|
| Intercept | 45.36       | 0.294   | n/a          |
| Income    | 1.78        | <0.001  | yes          |
| HH_size   | 22.12       | <0.001  | yes          |
| Age       | 0.45        | 0.559   | no           |

R² = 0.331, Adj R² = 0.321, F p-value = 5.1e-17, n = 200.
