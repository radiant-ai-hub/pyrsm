import polars as pl
import pyrsm as rsm

# 1. Load the catalog data
data_path = "/Users/vnijs/gh/pyrsm/examples/data/model/catalog.parquet"
df = pl.read_parquet(data_path)

print("Shape:", df.shape)
print("Schema:")
print(df.schema)
print()
print("Head:")
print(df.head())
print()

# 2. Fit the regression: Sales ~ Income + HH_size + Age
reg = rsm.model.regress(
    {"catalog": df},
    rvar="Sales",
    evar=["Income", "HH_size", "Age"],
)

# 3. Inspect the summary with RMSE and SSQ for context
reg.summary(rmse=True, ssq=True)
