"""
Refit catalog Sales regression after considering whether to drop Age.

The student wants to drop Age because its p-value (0.559) is well above 0.05.
Before just dropping it, let's:
  1. Fit the full model (Sales ~ Income + HH_size + Age) to confirm the result.
  2. Fit the reduced model (Sales ~ Income + HH_size).
  3. Compare coefficients, R^2, and check whether Age is correlated with the
     other regressors (omitted-variable bias check).
  4. Print both results so the student can make an informed decision.
"""

import pandas as pd
import statsmodels.formula.api as smf

DATA_PATH = "/Users/vnijs/gh/pyrsm/examples/data/model/catalog.parquet"

df = pd.read_parquet(DATA_PATH)
print("Data shape:", df.shape)
print(df.head(), "\n")

# 1. Full model
full = smf.ols("Sales ~ Income + HH_size + Age", data=df).fit()
print("=" * 70)
print("FULL MODEL: Sales ~ Income + HH_size + Age")
print("=" * 70)
print(full.summary())
print()

# 2. Reduced model (drop Age)
reduced = smf.ols("Sales ~ Income + HH_size", data=df).fit()
print("=" * 70)
print("REDUCED MODEL: Sales ~ Income + HH_size")
print("=" * 70)
print(reduced.summary())
print()

# 3. Side-by-side coefficient comparison
print("=" * 70)
print("COEFFICIENT COMPARISON")
print("=" * 70)
comp = pd.DataFrame(
    {
        "full_coef": full.params,
        "full_p": full.pvalues,
        "reduced_coef": reduced.params,
        "reduced_p": reduced.pvalues,
    }
)
print(comp.round(4))
print()

print("R-squared full   :", round(full.rsquared, 4))
print("R-squared reduced:", round(reduced.rsquared, 4))
print("Adj R^2 full     :", round(full.rsquared_adj, 4))
print("Adj R^2 reduced  :", round(reduced.rsquared_adj, 4))
print()

# 4. Check correlations of Age with other predictors (OVB sanity check)
print("=" * 70)
print("CORRELATIONS AMONG PREDICTORS")
print("=" * 70)
print(df[["Sales", "Income", "HH_size", "Age"]].corr().round(3))
