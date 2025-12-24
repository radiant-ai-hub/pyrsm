# Model Examples

Regression and machine learning models using the `pyrsm.model` module.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [model-linear-regression.ipynb](model-linear-regression.ipynb) | OLS regression with statsmodels |
| [model-logistic-regression.ipynb](model-logistic-regression.ipynb) | Logistic regression with statsmodels |
| [model-rforest-classification.ipynb](model-rforest-classification.ipynb) | Random forest classifier |
| [model-rforest-regression.ipynb](model-rforest-regression.ipynb) | Random forest regressor |
| [model-mlp-classification.ipynb](model-mlp-classification.ipynb) | Neural network classifier |
| [model-mlp-regression.ipynb](model-mlp-regression.ipynb) | Neural network regressor |
| [model-xgboost-classification.ipynb](model-xgboost-classification.ipynb) | XGBoost classifier |
| [model-xgboost-regression.ipynb](model-xgboost-regression.ipynb) | XGBoost regressor |

## Installation

Machine learning models require the `ml` extra:

```bash
uv add "pyrsm[ml]"
```

## Usage

```python
import polars as pl
from pyrsm import model

df = pl.read_parquet("https://github.com/radiant-ai-hub/pyrsm/raw/refs/heads/main/examples/data/data/diamonds.parquet")

# Linear regression
reg = model.regress(df, rvar="price", evar=["carat", "clarity", "cut"])
reg.summary()
reg.plot(plots="dashboard")

# Random forest
rf = model.rforest(df, rvar="price", evar=["carat", "clarity", "cut"], mod_type="regression")
rf.summary()
rf.plot(plots="pdp")
```
