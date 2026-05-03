"""
Diamonds regression: price ~ carat + clarity, with linear vs log-log comparison.

Workflow:
  1. Load the parquet
  2. Fit linear model price ~ carat + clarity
  3. Inspect summary and residual dashboard
  4. Refit log-log: ln(price) ~ ln(carat) + clarity
  5. Inspect summary and residual dashboard
  6. Translate clarity dummies to exact %-change figures
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so plots save cleanly

import math
import polars as pl
import pyrsm as rsm

OUT = "/Users/vnijs/gh/pyrsm/skills/pyrsm-regress-workspace/iteration-2/diamonds-loglog/with_skill/outputs"
DATA_PATH = "/Users/vnijs/gh/pyrsm/examples/data/data/diamonds.parquet"

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
df = pl.read_parquet(DATA_PATH)
print("=" * 78)
print("DIAMONDS DATASET")
print("=" * 78)
print(f"Path : {DATA_PATH}")
print(f"Shape: {df.shape}")
print("Schema:")
for name, dtype in df.schema.items():
    print(f"  {name:10s} {dtype}")
print()
print("Price summary:")
print(df.select(
    pl.col("price").min().alias("min"),
    pl.col("price").max().alias("max"),
    pl.col("price").mean().alias("mean"),
    pl.col("price").median().alias("median"),
))
print("Carat summary:")
print(df.select(
    pl.col("carat").min().alias("min"),
    pl.col("carat").max().alias("max"),
    pl.col("carat").mean().alias("mean"),
    pl.col("carat").median().alias("median"),
))
print("Clarity levels (worst -> best): I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF")
print()

# ---------------------------------------------------------------------------
# 2. Linear fit: price ~ carat + clarity
# ---------------------------------------------------------------------------
print("=" * 78)
print("MODEL 1 — LINEAR: price ~ carat + clarity")
print("=" * 78)
reg_lin = rsm.model.regress(
    {"diamonds": df},
    rvar="price",
    evar=["carat", "clarity"],
)
reg_lin.summary(rmse=True, ssq=True)

# Residual dashboard for the linear spec
try:
    g_lin = reg_lin.plot("dashboard")
    g_lin.save(f"{OUT}/dashboard_linear.png", width=12, height=8, dpi=100)
    print(f"\nSaved: {OUT}/dashboard_linear.png")
except Exception as e:
    print(f"\n[WARN] could not save linear dashboard: {e}")

# ---------------------------------------------------------------------------
# 3. Build log-transformed columns
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("MODEL 2 — LOG-LOG: ln(price) ~ ln(carat) + clarity")
print("=" * 78)
df = df.with_columns(
    price_ln=pl.col("price").log(),
    carat_ln=pl.col("carat").log(),
)

reg_log = rsm.model.regress(
    {"diamonds": df},
    rvar="price_ln",
    evar=["carat_ln", "clarity"],
)
reg_log.summary(rmse=True, ssq=True)

# Residual dashboard for the log-log spec
try:
    g_log = reg_log.plot("dashboard")
    g_log.save(f"{OUT}/dashboard_loglog.png", width=12, height=8, dpi=100)
    print(f"\nSaved: {OUT}/dashboard_loglog.png")
except Exception as e:
    print(f"\n[WARN] could not save log-log dashboard: {e}")

# ---------------------------------------------------------------------------
# 4. Exact % interpretation of the clarity dummies in the log-log model
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("EXACT %-CHANGE TABLE FOR CLARITY DUMMIES — ln(price) model")
print("=" * 78)
print("For ln(y) ~ dummy: %-change in y vs reference = 100*(exp(beta) - 1)%")
print("Reference level for clarity = I1 (worst clarity)")
print()

coef = reg_log.coef  # polars DataFrame
clarity_rows = coef.filter(pl.col("index").str.starts_with("clarity["))
clarity_rows = clarity_rows.with_columns(
    pct_change_exact=(pl.col("coefficient").exp() - 1) * 100,
    pct_change_approx=pl.col("coefficient") * 100,
)
print(clarity_rows.select([
    "index", "coefficient", "p.value", "pct_change_exact", "pct_change_approx",
]))

print()
print("Interpretation example:")
row_if = clarity_rows.filter(pl.col("index") == "clarity[IF]")
if row_if.height > 0:
    b = row_if["coefficient"][0]
    pct = (math.exp(b) - 1) * 100
    print(f"  clarity[IF] beta = {b:.4f}")
    print(f"    -> exp({b:.4f}) - 1 = {math.exp(b)-1:.4f}")
    print(f"    -> {pct:.2f}% higher price than an I1-clarity diamond,")
    print(f"       holding ln(carat) (i.e. carat) constant.")

# Carat elasticity
carat_row = coef.filter(pl.col("index") == "carat_ln")
if carat_row.height > 0:
    b_carat = carat_row["coefficient"][0]
    print()
    print("Carat elasticity (log-log):")
    print(f"  beta on ln(carat) = {b_carat:.4f}")
    print(f"  Interpretation: a 1% increase in carat is associated with ")
    print(f"  approximately a {b_carat:.3f}% increase in price, holding clarity constant.")
    print(f"  Equivalently, doubling carat multiplies price by 2^{b_carat:.3f} = "
          f"{2**b_carat:.3f}.")

# Headline R^2 comparison
print()
print("=" * 78)
print("HEADLINE COMPARISON")
print("=" * 78)
print(f"Linear  model R^2: {reg_lin.fitted.rsquared:.4f}, "
      f"adj-R^2: {reg_lin.fitted.rsquared_adj:.4f}")
print(f"Log-log model R^2: {reg_log.fitted.rsquared:.4f}, "
      f"adj-R^2: {reg_log.fitted.rsquared_adj:.4f}")
print("Note: R^2 from the two models is not directly comparable (different "
      "response variables), but the residual dashboards are the right "
      "yardstick — and those tell a clear story.")
