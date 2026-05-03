"""Diamonds: linear vs log-log regression of price on carat and clarity.

Walks through the standard pyrsm regress workflow:
  1. Load the data
  2. Fit linear: price ~ carat + clarity
  3. Inspect summary + residual dashboard
  4. Refit log-log: ln(price) ~ ln(carat) + clarity
  5. Inspect summary + residual dashboard
"""

import matplotlib

matplotlib.use("Agg")  # headless backend so savefig works without a display

import polars as pl
import pyrsm as rsm

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
data_path = "/Users/vnijs/gh/pyrsm/examples/data/data/diamonds.parquet"
df = pl.read_parquet(data_path)

print("=" * 78)
print("DATA OVERVIEW")
print("=" * 78)
print(f"Shape: {df.shape}")
print("Schema:")
for name, dtype in df.schema.items():
    print(f"  {name}: {dtype}")
print()
print("Summary stats for price and carat:")
print(df.select(["price", "carat"]).describe())
print()
print("Clarity levels:")
print(df.group_by("clarity").len().sort("clarity"))
print()

# ---------------------------------------------------------------------------
# 2. Linear specification: price ~ carat + clarity
# ---------------------------------------------------------------------------
print("=" * 78)
print("LINEAR MODEL: price ~ carat + clarity")
print("=" * 78)

reg_lin = rsm.model.regress(
    {"diamonds": df},
    rvar="price",
    evar=["carat", "clarity"],
)
reg_lin.summary(rmse=True, ssq=True)

# Residual dashboard for the linear spec
print()
print("Saving linear-spec residual dashboard to dashboard_linear.png ...")
out_dir = "/Users/vnijs/gh/pyrsm/skills/pyrsm-regress-workspace/iteration-1/diamonds-loglog/with_skill/outputs"
try:
    g_lin = reg_lin.plot("dashboard")
    # plotnine ggplot exposes .save; matplotlib Figure exposes .savefig
    if hasattr(g_lin, "save"):
        g_lin.save(f"{out_dir}/dashboard_linear.png", width=10, height=8, dpi=100)
    else:
        g_lin.savefig(f"{out_dir}/dashboard_linear.png", dpi=100, bbox_inches="tight")
    print("  saved dashboard_linear.png")
except Exception as e:
    print(f"  failed to save linear dashboard: {e!r}")

# ---------------------------------------------------------------------------
# 3. Log-log specification: ln(price) ~ ln(carat) + clarity
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("LOG-LOG MODEL: ln(price) ~ ln(carat) + clarity")
print("=" * 78)

df_log = df.with_columns(
    price_ln=pl.col("price").log(),
    carat_ln=pl.col("carat").log(),
)

reg_log = rsm.model.regress(
    {"diamonds": df_log},
    rvar="price_ln",
    evar=["carat_ln", "clarity"],
)
reg_log.summary(rmse=True, ssq=True)

print()
print("Saving log-log residual dashboard to dashboard_loglog.png ...")
try:
    g_log = reg_log.plot("dashboard")
    if hasattr(g_log, "save"):
        g_log.save(f"{out_dir}/dashboard_loglog.png", width=10, height=8, dpi=100)
    else:
        g_log.savefig(f"{out_dir}/dashboard_loglog.png", dpi=100, bbox_inches="tight")
    print("  saved dashboard_loglog.png")
except Exception as e:
    print(f"  failed to save log-log dashboard: {e!r}")

print()
print("Done.")
