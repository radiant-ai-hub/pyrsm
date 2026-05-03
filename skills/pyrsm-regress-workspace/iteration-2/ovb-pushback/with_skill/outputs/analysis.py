"""OVB check for the catalog regression.

Walks the Step 6 protocol from the pyrsm-regress skill:
  1. Fit the full model: Sales ~ Income + HH_size + Age
  2. Snapshot baseline coefficients
  3. Drop Age (the variable the student wants to remove on p-value alone)
  4. Refit and compute percent shifts on the kept variables
  5. Check OVB triggers: |pct_shift| > 10%, sign flip, or significance flip
"""

import polars as pl
import pyrsm as rsm

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data_path = "/Users/vnijs/gh/pyrsm/examples/data/model/catalog.parquet"
df = pl.read_parquet(data_path)
print("=" * 72)
print("Data loaded:", data_path)
print("Shape:", df.shape)
print("Columns:", df.columns)
print()

# ---------------------------------------------------------------------------
# 2. Fit the full model
# ---------------------------------------------------------------------------
print("=" * 72)
print("STEP 1 - Full model: Sales ~ Income + HH_size + Age")
print("=" * 72)
reg_full = rsm.model.regress(
    {"catalog": df},
    rvar="Sales",
    evar=["Income", "HH_size", "Age"],
)
reg_full.summary(rmse=True, ssq=True)
print()

# ---------------------------------------------------------------------------
# 3. Snapshot baseline coefficients
# ---------------------------------------------------------------------------
print("=" * 72)
print("STEP 2 - Snapshot baseline coefficients")
print("=" * 72)
baseline = reg_full.coef.select(["index", "coefficient", "std.error", "p.value"]).clone()
print(baseline)
print()

# ---------------------------------------------------------------------------
# 4. Drop Age and refit
# ---------------------------------------------------------------------------
print("=" * 72)
print("STEP 3 - Drop Age and refit: Sales ~ Income + HH_size")
print("=" * 72)
new_evar = [v for v in reg_full.evar if v != "Age"]
reg_reduced = rsm.model.regress(
    {"catalog": df},
    rvar="Sales",
    evar=new_evar,
)
reg_reduced.summary(rmse=True, ssq=True)
print()

# ---------------------------------------------------------------------------
# 5. Compute percent shifts on the kept variables
# ---------------------------------------------------------------------------
print("=" * 72)
print("STEP 4 - OVB check: percent shifts on kept variables")
print("=" * 72)

new_coef = reg_reduced.coef.select(["index", "coefficient", "p.value"]).rename({
    "coefficient": "new_coef",
    "p.value": "new_p",
})

comparison = (
    baseline.rename({"coefficient": "old_coef", "p.value": "old_p"})
    .join(new_coef, on="index", how="inner")
    .with_columns(
        pct_shift=((pl.col("new_coef") - pl.col("old_coef")) / pl.col("old_coef").abs() * 100).round(2),
        sign_flip=(pl.col("new_coef").sign() != pl.col("old_coef").sign()),
        sig_flip=((pl.col("old_p") < 0.05) != (pl.col("new_p") < 0.05)),
    )
    .select(["index", "old_coef", "new_coef", "pct_shift", "old_p", "new_p", "sign_flip", "sig_flip"])
)
print(comparison)
print()

# ---------------------------------------------------------------------------
# 6. Trigger check (slope coefficients only — intercept shifts are mechanical
#    when a non-zero-mean predictor is dropped and are not OVB on the
#    interpreted effects)
# ---------------------------------------------------------------------------
print("=" * 72)
print("STEP 5 - Trigger evaluation (on kept slope coefficients)")
print("=" * 72)
slope_comparison = comparison.filter(pl.col("index") != "Intercept")
print("Slope-only view:")
print(slope_comparison)
print()
triggers = slope_comparison.filter(
    (pl.col("pct_shift").abs() > 10) | pl.col("sign_flip") | pl.col("sig_flip")
)
if triggers.height > 0:
    print("OVB triggers fired on the following variables:")
    print(triggers)
    print()
    print("==> A trigger fired. Recommended actions:")
    print("    1. Re-include Age (default conservative choice)")
    print("    2. Relabel a remaining variable to reflect the combined effect")
    print("    3. Combine Age with a correlated predictor into a composite")
    print("    4. Keep the reduced model with explicit acknowledgment")
else:
    print("No OVB triggers fired.")
    print("On the kept variables (Income, HH_size):")
    print("  - All percent shifts are within +/- 10%")
    print("  - No sign flips")
    print("  - No significance status flips across the 5% line")
    print()
    print("==> Age is approximately ORTHOGONAL to Income and HH_size in this sample.")
    print("    The drop appears OVB-safe based on the statistical check.")
    print()
    print("HOWEVER: with n=200 and only 3 predictors, there is no real reason")
    print("to simplify in the first place. Recommendation: KEEP Age in the")
    print("reported model. Non-significance does not mean 'no effect' - it")
    print("means we lack evidence in this sample. The OVB-safe finding is a")
    print("pedagogical confirmation, not a license to drop.")
print()

# ---------------------------------------------------------------------------
# 7. Sanity check: correlations among predictors (why the shifts are small)
# ---------------------------------------------------------------------------
print("=" * 72)
print("BONUS - Pairwise correlations among predictors")
print("=" * 72)
corr_df = df.select(["Income", "HH_size", "Age"]).to_pandas().corr()
print(corr_df.round(3))
print()
print("Small Age <-> Income and Age <-> HH_size correlations explain why")
print("dropping Age barely moves the kept coefficients - that is what")
print("'approximately orthogonal' means in practice.")
