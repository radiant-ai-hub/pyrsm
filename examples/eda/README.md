# EDA Examples

Exploratory data analysis using the `pyrsm.eda` module.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [eda-explore.ipynb](eda-explore.ipynb) | Summary statistics for numeric columns |
| [eda-visualize.ipynb](eda-visualize.ipynb) | Data visualization (histograms, scatter, box plots) |
| [eda-pivot.ipynb](eda-pivot.ipynb) | Pivot tables for data aggregation |
| [eda-unpivot.ipynb](eda-unpivot.ipynb) | Reshape data from wide to long format |
| [eda-distr.ipynb](eda-distr.ipynb) | Distribution analysis |
| [eda-combine.ipynb](eda-combine.ipynb) | Join and combine datasets |

## Usage

```python
import polars as pl
from pyrsm import eda

df = pl.read_parquet("https://github.com/radiant-ai-hub/pyrsm/raw/refs/heads/main/examples/data/data/diamonds.parquet")

# Explore numeric columns
eda.explore(df, cols=["price", "carat"], by="cut")

# Visualize distributions
eda.visualize(df, x="price", geom="hist")
```
