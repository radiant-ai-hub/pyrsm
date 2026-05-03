import polars as pl
import pyrsm as rsm

# 1. Load the data (absolute path)
data_path = "/Users/vnijs/gh/pyrsm/examples/data/model/catalog.parquet"
df = pl.read_parquet(data_path)

# Quick look at the data
print("Shape:", df.shape)
print("Columns / dtypes:")
for name, dtype in zip(df.columns, df.dtypes):
    print(f"  {name}: {dtype}")
print()
print("First 5 rows:")
print(df.head())
print()

# 2. Fit the regression: predict Sales from Income, HH_size, and Age
reg = rsm.model.regress(
    {"catalog": df},
    rvar="Sales",
    evar=["Income", "HH_size", "Age"],
)

# 3. Inspect
reg.summary(rmse=True, ssq=True)
