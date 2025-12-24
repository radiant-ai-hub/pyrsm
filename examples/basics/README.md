# Basics Examples

Statistical tests and analyses using the `pyrsm.basics` module.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [basics-single-mean.ipynb](basics-single-mean.ipynb) | One-sample t-test and z-test |
| [basics-single-proportion.ipynb](basics-single-proportion.ipynb) | One-sample proportion test |
| [basics-compare-means.ipynb](basics-compare-means.ipynb) | Two-sample t-test and ANOVA |
| [basics-compare-props.ipynb](basics-compare-props.ipynb) | Two-sample proportion test |
| [basics-cross-tabs.ipynb](basics-cross-tabs.ipynb) | Cross-tabulation with chi-square test |
| [basics-goodness.ipynb](basics-goodness.ipynb) | Goodness of fit test |
| [basics-correlation.ipynb](basics-correlation.ipynb) | Correlation analysis |
| [basics-probability-calculator.ipynb](basics-probability-calculator.ipynb) | Probability distributions |

## Usage

```python
import polars as pl
from pyrsm import basics

df = pl.read_parquet("https://github.com/radiant-ai-hub/pyrsm/raw/refs/heads/main/examples/data/data/titanic.parquet")

# Compare means across groups
cm = basics.compare_means(df, var1="pclass", var2="age")
cm.summary()
cm.plot()
```
