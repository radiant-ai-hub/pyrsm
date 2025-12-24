# Example Data

Datasets used in pyrsm example notebooks.

## Datasets

| Dataset | Description |
|---------|-------------|
| `titanic.parquet` | Titanic passenger survival data |
| `diamonds.parquet` | Diamond prices and attributes |
| `superheroes.parquet` | Superhero names, alignment, and publishers |
| `publishers.parquet` | Comic book publishers and founding years |
| `avengers.parquet` | Avengers team members |

Additional example datasets are connected to each sub-folder in `examples/`

## Loading Data

```python
import polars as pl
import pyrsm as rsm

# Load from GitHub (recommended for reproducibility)
df = pl.read_parquet("https://github.com/radiant-ai-hub/pyrsm/raw/refs/heads/main/examples/data/data/titanic.parquet")

# Display dataset description
rsm.md("https://raw.githubusercontent.com/radiant-ai-hub/pyrsm/refs/heads/main/examples/data/data/titanic_description.md")
```

## Folder Structure

- `data/` - Parquet files and description markdown files
- `basics/` - Data specific to basics examples
- `design/` - Data specific to design examples
- `model/` - Data specific to model examples
- `multivariate/` - Data specific to model examples
