# PYRSM

Python functions and classes for Business Analytics at the Rady School of Management (RSM), University of California, San Diego (UCSD).

## Features

**Basics Module** - Statistical tests and analyses:
- `compare_means` - Compare means across groups (t-tests, ANOVA)
- `compare_props` - Compare proportions between groups
- `correlation` - Correlation analysis with significance tests
- `cross_tabs` - Cross-tabulation with chi-square tests
- `goodness` - Goodness of fit tests
- `single_mean` - Single sample mean tests
- `single_prop` - Single sample proportion tests
- `prob_calc` - Probability calculator for common distributions

**Model Module** - Regression and machine learning:
- `regress` - Linear regression with statsmodels
- `logistic` - Logistic regression with statsmodels
- `mlp` - Multi-layer perceptron (neural network) with sklearn
- `rforest` - Random forest with sklearn
- `xgboost` - XGBoost gradient boosting

**EDA Module** - Exploratory data analysis:
- `explore` - Data exploration and summary statistics
- `pivot` - Pivot tables
- `visualize` - Data visualization

All modules use [Polars](https://pola.rs/) DataFrames and [plotnine](https://plotnine.org/) for visualization.

## Installation

Requires Python 3.12+ and UV:

```bash
mkdir ~/project
cd ~/project
uv init .
uv venv --python 3.13
source .venv/bin/activate
uv add pyrsm
```

For machine learning models, install with extras:

```bash
uv add "pyrsm[ml]"
```

For all features:

```bash
uv add "pyrsm[all]"
```

## Quick Start

```python
import polars as pl
from pyrsm import basics, model

# Load data
df = pl.read_parquet("data.parquet")

# Statistical test
cm = basics.compare_means(df, var="price", byvar="category")
cm.summary()
cm.plot()

# Regression model
reg = model.regress(df, rvar="price", evar=["size", "age", "type"])
reg.summary()
reg.plot()
```

## Examples

Extensive example notebooks are available at: <https://github.com/radiant-ai-hub/pyrsm/tree/main/examples>

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).
