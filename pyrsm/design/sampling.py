import numpy as np
import polars as pl

from pyrsm.data_scope import apply_scope
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
    data_filter : str, default ""
        Expression used to filter the data before the analysis, e.g.,
        "price > 5000 and cut in ['Ideal', 'Premium']". A bare name is a column
        and a call is an operation on one (``is_null(price)``). See
        ``pyrsm.data_scope`` for the full list.
    sort : str, default ""
        Column(s) to sort on, "-" in front for descending, e.g.,
        "clarity -price". Applied after the filter.
    slice : str, default ""
        Rows to keep, counting from 1, e.g., "1:100". Applied after the filter
        and the sort.

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
        data_filter: str = "",
        sort: str = "",
        slice: str = "",
    ):
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            data = list(data.values())[0]
        else:
            self.name = "data"

        data = check_dataframe(data)

        # The data scope, applied before anything is measured: filter, then
        # sort, then slice. Kept on the object so the output can say which
        # rows the numbers describe.
        self.data_filter = data_filter
        self.sort = sort
        self.slice = slice
        data = apply_scope(data, data_filter, sort, slice)

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
