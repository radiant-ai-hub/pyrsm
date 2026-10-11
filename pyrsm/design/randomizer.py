import numpy as np
import polars as pl

from pyrsm.data_scope import apply_scope, print_scope
from pyrsm.utils import check_dataframe


class randomizer:
    """
    Random assignment of cases to experimental conditions.

    Supports simple complete randomization and block randomization.

    Parameters
    ----------
    data : pl.DataFrame or dict[str, pl.DataFrame]
        Dataset to assign conditions to
    vars : list[str] or None
        Variables to include. If None, all columns are used.
    conditions : list[str] or None
        Condition labels. Default is ["A", "B"].
    blocks : str or None
        Column name to use for block randomization
    probs : list[float] or None
        Assignment probabilities for each condition. Default is equal probability.
    label : str
        Name for the generated condition column
    seed : int
        Random seed for reproducibility

    Attributes
    ----------
    data : pl.DataFrame
        Dataset with condition column added
    conditions : list[str]
        Condition labels
    blocks : str or None
        Block variable name
    probs : list[float]
        Assignment probabilities
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
    >>> df = pl.DataFrame({"name": [f"person_{i}" for i in range(20)]})
    >>> r = randomizer(df, conditions=["test", "control"], seed=1234)
    >>> r.data.shape[0]
    20
    """

    def __init__(
        self,
        data: pl.DataFrame | dict[str, pl.DataFrame],
        vars: list[str] | None = None,
        conditions: list[str] | None = None,
        blocks: str | None = None,
        probs: list[float] | None = None,
        label: str = ".conditions",
        seed: int = 1234,
        data_filter: str = "",
        sort: str = "",
        slice: str = "",
    ):
        if conditions is None:
            conditions = ["A", "B"]

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

        # Include blocks in vars if specified
        all_vars = list(vars) if vars is not None else list(data.columns)
        if blocks is not None and blocks not in all_vars:
            all_vars.append(blocks)

        data = data.select(all_vars)

        self.vars = [v for v in all_vars if v != blocks] if blocks else all_vars
        self.conditions = conditions
        self.blocks = blocks
        self.label = label
        self.seed = seed

        if probs is None:
            probs = [1.0 / len(conditions)] * len(conditions)
        elif len(probs) == 1:
            probs = probs * len(conditions)
        elif len(probs) != len(conditions):
            probs = [1.0 / len(conditions)] * len(conditions)

        self.probs = probs

        rng = np.random.default_rng(seed)

        if blocks is not None:
            # Block randomization
            block_col = data[blocks]
            cond_assignments = [""] * data.height

            for block_val in block_col.unique().sort().to_list():
                mask = block_col == block_val
                indices = [i for i, m in enumerate(mask.to_list()) if m]
                block_n = len(indices)
                assignments = _complete_ra(block_n, conditions, probs, rng)
                for idx, assignment in zip(indices, assignments):
                    cond_assignments[idx] = assignment
        else:
            # Simple complete randomization
            cond_assignments = _complete_ra(data.height, conditions, probs, rng)

        self.data = pl.DataFrame({label: cond_assignments}).hstack(data)

    def summary(self, dec: int = 3) -> None:
        """Print summary of randomization results."""
        if self.blocks is None:
            print("Random assignment (simple random)")
        else:
            print("Random assignment (blocking)")

        print(f"Data         : {self.name}")
        # The rows assigned, when a filter / sort / slice narrowed them.
        print_scope(self, width=13)

        if self.blocks is not None:
            print(f"Variables    : {', '.join(self.vars)}")
            print(f"Blocks       : {self.blocks}")
        else:
            print(f"Variables    : {', '.join(self.vars)}")

        print(f"Conditions   : {', '.join(self.conditions)}")
        print(f"Probabilities: {', '.join(str(round(p, dec)) for p in self.probs)}")
        print(f"Random seed  : {self.seed}")

        # Check for duplicates
        data_cols = [c for c in self.data.columns if c != self.label]
        n_unique = self.data.select(data_cols).unique().height
        if self.data.height > n_unique:
            dup_msg = "Based on selected variables some duplicate rows exist"
        else:
            dup_msg = "Based on selected variables, no duplicate rows exist"
        print(f"Duplicates   : {dup_msg}")

        # Assignment frequencies and proportions, as tables (one row per
        # condition and a Total; with blocks, one row per block).
        counts = self.data[self.label].value_counts().sort(self.label)
        total = int(counts["count"].sum())
        conds = [str(v) for v in counts[self.label].to_list()]
        with pl.Config(tbl_hide_dataframe_shape=True, tbl_rows=-1, tbl_hide_column_data_types=True):
            print("\nAssignment frequencies:")
            if self.blocks is None:
                print(pl.DataFrame({
                    self.label: conds + ["Total"],
                    "n": [int(v) for v in counts["count"].to_list()] + [total],
                }))
            else:
                cross = (
                    self.data.group_by([self.blocks, self.label])
                    .len()
                    .sort([self.blocks, self.label])
                    .pivot(on=self.label, index=self.blocks, values="len")
                    .fill_null(0)
                )
                cond_cols = [c for c in cross.columns if c != self.blocks]
                cross = cross.with_columns(
                    pl.sum_horizontal(*[pl.col(c) for c in cond_cols]).alias("Total")
                )
                print(cross)
            print("\nAssignment proportions:")
            print(pl.DataFrame({
                self.label: conds + ["Total"],
                "proportion": [round(int(v) / total, dec) for v in counts["count"].to_list()] + [1.0],
            }))


def _complete_ra(n, conditions, probs, rng):
    """Complete random assignment: allocate floor counts, then assign remainder randomly."""
    n_conds = len(conditions)
    # Floor allocation
    base_counts = [int(n * p) for p in probs]
    remainder = n - sum(base_counts)

    # Distribute remainder
    if remainder > 0:
        # Create probability weights for remainder allocation
        frac_parts = [n * p - int(n * p) for p in probs]
        frac_sum = sum(frac_parts)
        if frac_sum > 0:
            remainder_probs = [f / frac_sum for f in frac_parts]
        else:
            remainder_probs = [1 / n_conds] * n_conds

        # Assign remainder units to conditions based on fractional parts
        extra = rng.choice(n_conds, size=remainder, replace=False, p=remainder_probs)
        for idx in extra:
            base_counts[idx] += 1

    # Build assignment vector and shuffle
    assignments = []
    for cond, count in zip(conditions, base_counts):
        assignments.extend([cond] * count)

    rng.shuffle(assignments)
    return assignments
