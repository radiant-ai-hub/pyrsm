"""``rsm.register``: make a dataset made in report code available in Radiant.

In a Radiant report, ``rsm.register(sales_clean, "sales_clean")`` adds the frame
to the app's data list once the report (or the selected chunks) finish running.
Radiant installs its own implementation in the report kernel; this one is what
runs everywhere else -- a notebook, a script, a plain Quarto render -- where
there is no app to register with. It checks the arguments the same way, so code
that works here also works in Radiant, says plainly that nothing happened, and
returns ``None``.
"""

from __future__ import annotations

import keyword
from typing import Any


def _as_polars(data: Any):
    import polars as pl

    if isinstance(data, pl.LazyFrame):
        return data.collect()
    if isinstance(data, pl.DataFrame):
        return data
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a pyrsm dependency
        pd = None
    if pd is not None and isinstance(data, pd.DataFrame):
        return pl.from_pandas(data)
    raise TypeError(
        "rsm.register expects a polars or pandas DataFrame as its first argument, "
        f"not {type(data).__name__}."
    )


def _check_name(name: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        raise TypeError(
            'rsm.register needs the dataset name as its second argument, e.g. '
            'rsm.register(sales_clean, "sales_clean").'
        )
    name = name.strip()
    if not name.isidentifier() or keyword.iskeyword(name) or not name.isascii():
        raise ValueError(
            f"{name!r} cannot be a dataset name: use letters, digits and "
            "underscores, starting with a letter (e.g. sales_clean)."
        )
    return name


def register(data: Any, name: str, descr: str | None = None) -> None:
    """Register a DataFrame as a Radiant dataset.

    Parameters
    ----------
    data : polars.DataFrame | pandas.DataFrame
        The data to register. A pandas frame is converted to polars, a
        ``polars.LazyFrame`` is collected.
    name : str
        The name the dataset gets in Radiant's data list. Required, and a valid
        Python name, so the report can refer to it by that name later.
    descr : str, optional
        A short description shown with the dataset.

    Returns
    -------
    None

    Notes
    -----
    Outside Radiant this only checks the arguments and prints a note: there is
    no app to register with. In a Radiant report, the dataset appears in the data
    list after the run finishes. A name that belongs to data you imported is
    never overwritten.

    Examples
    --------
    >>> sales_clean = sales.filter(pl.col("price") > 0)
    >>> rsm.register(sales_clean, "sales_clean", descr="Sales without zero prices")
    """
    frame = _as_polars(data)
    name = _check_name(name)
    if descr is not None and not isinstance(descr, str):
        raise TypeError("rsm.register: descr must be a string.")
    print(
        f"rsm.register: `{name}` ({frame.height:,} rows × {frame.width} columns) "
        "is added to the data list when this code runs in a Radiant report; "
        "outside Radiant there is nothing to register."
    )
    return None
