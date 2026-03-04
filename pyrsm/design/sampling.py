import numpy as np
import polars as pl

from pyrsm.utils import check_dataframe


class sampling:
    """
    Simple random sampling from a DataFrame.

    Parameters
    ----------
    data : pl.DataFrame or dict[str, pl.DataFrame]
        Dataset to sample from. If dict, uses first key as dataset name.
    vars : list[str] or None
        Variables to include. If None, all columns are used.
    sample_size : int
        Number of units to select
    seed : int
        Random seed for reproducibility

    Attributes
    ----------
    data : pl.DataFrame
        Full dataset with rnd_number column added
    selected : pl.DataFrame
        Selected sample rows
    sample_size : int
        Number of units selected
    seed : int
        Random seed used
    name : str
        Name of the dataset

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": range(100), "y": range(100)})
    >>> s = sampling(df, sample_size=5, seed=1234)
    >>> s.selected.shape[0]
    5
    """

    def __init__(
        self,
        data: pl.DataFrame | dict[str, pl.DataFrame],
        vars: list[str] | None = None,
        sample_size: int = 10,
        seed: int = 1234,
    ):
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            data = list(data.values())[0]
        else:
            self.name = "data"

        data = check_dataframe(data)

        if vars is not None:
            data = data.select(vars)

        self.vars = vars if vars is not None else data.columns
        self.sample_size = max(1, sample_size)
        self.seed = seed

        rng = np.random.default_rng(seed)
        rnd_numbers = rng.uniform(0, 1, size=data.height)

        self.data = data.with_columns(pl.Series("rnd_number", rnd_numbers))
        sorted_data = self.data.sort("rnd_number", descending=True)
        self.selected = sorted_data.head(self.sample_size)

    def summary(self, dec: int = 3) -> None:
        """Print summary of sampling results."""
        print("Sampling (simple random)")
        print(f"Data       : {self.name}")
        print(f"Variables  : {', '.join(self.vars)}")
        print(f"Random seed: {self.seed}")
        print(f"Sample size: {self.sample_size}")

        # Check for duplicates
        data_cols = [c for c in self.selected.columns if c != "rnd_number"]
        n_unique = self.selected.select(data_cols).unique().height
        if self.selected.height > n_unique:
            dup_msg = "Based on selected variables some duplicate rows exist"
        else:
            dup_msg = "Based on selected variables, no duplicate rows exist"
        print(f"Duplicates : {dup_msg}")
