# PYRSM Examples

Example Jupyter notebooks demonstrating the pyrsm package for Business Analytics.

## Folders

| Folder | Description |
|--------|-------------|
| [basics/](basics/) | Statistical tests (t-tests, ANOVA, chi-square, correlation) |
| [eda/](eda/) | Exploratory data analysis (explore, pivot, visualize, combine) |
| [model/](model/) | Regression and machine learning models |
| [data/](data/) | Example datasets used in notebooks |

## Running Notebooks

Run the below to get the relevant components of pyrsm locally.

```bash
# Install with notebook support
uv add "pyrsm[notebooks]"

# Or install all extras
uv add "pyrsm[all]"
```

If you already have pyrsm installed, you can download the example notebooks directly and run them.

## Data

All notebooks load data from online URLs for reproducibility:

```python
import polars as pl
df = pl.read_parquet("https://github.com/radiant-ai-hub/pyrsm/raw/refs/heads/main/examples/data/data/titanic.parquet")
```

Dataset descriptions are displayed using:

```python
import pyrsm as rsm
rsm.md("https://raw.githubusercontent.com/radiant-ai-hub/pyrsm/refs/heads/main/examples/data/data/titanic_description.md")
```
