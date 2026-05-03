"""
Catalog Sales Regression Walkthrough
------------------------------------
Predict Sales from Income, HH_size, and Age using OLS (ordinary least squares).
"""

import pandas as pd
import statsmodels.formula.api as smf

# 1. Load the data
df = pd.read_parquet("/Users/vnijs/gh/pyrsm/examples/data/model/catalog.parquet")

print("=" * 70)
print("STEP 1: PEEK AT THE DATA")
print("=" * 70)
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary statistics:")
print(df[["Sales", "Income", "HH_size", "Age"]].describe().round(2))

# 2. Quick correlation check
print("\n" + "=" * 70)
print("STEP 2: CORRELATIONS WITH SALES")
print("=" * 70)
print(df[["Sales", "Income", "HH_size", "Age"]].corr().round(3))

# 3. Fit the regression
print("\n" + "=" * 70)
print("STEP 3: FIT THE OLS REGRESSION")
print("=" * 70)
print("Model: Sales ~ Income + HH_size + Age")

model = smf.ols("Sales ~ Income + HH_size + Age", data=df).fit()
print(model.summary())

# 4. Pull out the key numbers in a tidy table
print("\n" + "=" * 70)
print("STEP 4: COEFFICIENT TABLE (TIDY)")
print("=" * 70)
coef_table = pd.DataFrame(
    {
        "coefficient": model.params,
        "std_error": model.bse,
        "t_stat": model.tvalues,
        "p_value": model.pvalues,
        "ci_low": model.conf_int()[0],
        "ci_high": model.conf_int()[1],
    }
).round(3)
print(coef_table)

print(f"\nR-squared:           {model.rsquared:.4f}")
print(f"Adjusted R-squared:  {model.rsquared_adj:.4f}")
print(f"F-statistic:         {model.fvalue:.2f}")
print(f"F p-value:           {model.f_pvalue:.3e}")
print(f"N (observations):    {int(model.nobs)}")

# 5. Predict for an example customer
print("\n" + "=" * 70)
print("STEP 5: SANITY-CHECK PREDICTION")
print("=" * 70)
example = pd.DataFrame({"Income": [70.0], "HH_size": [3], "Age": [48.0]})
pred = model.predict(example).iloc[0]
print(f"Predicted Sales for an 'average' customer (Income=70, HH_size=3, Age=48): "
      f"${pred:.2f}")
print(f"Actual mean Sales in data: ${df['Sales'].mean():.2f}")
