"""Dataset descriptions stored with parquet files (``rsm.data.add_description``).

A description is markdown kept in two places, the same way Radiant saves data
and pyrsm ships its examples:

- inside the parquet file, under the ``description`` metadata key, so it cannot
  be separated from the data by copying or emailing the file;
- beside it, as ``<name>_description.md``, readable without a parquet reader.

When the two disagree, the copy inside the file wins.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

#: The parquet metadata key a description is stored under.
DESCRIPTION_KEY = "description"
#: The suffix of the markdown file written beside the parquet file.
DESCRIPTION_SUFFIX = "_description.md"

#: Metadata pyarrow writes for its own use; it describes the schema, and the
#: new file gets a fresh one.
_ARROW_SCHEMA_KEY = "ARROW:schema"
#: A string ending in one of these is taken for a file name.
_TEXT_SUFFIXES = {".md", ".markdown", ".qmd", ".txt"}


def description_path(path: str | Path) -> Path:
    """The markdown file beside ``path`` that holds its description."""
    path = Path(path)
    return path.with_name(f"{path.stem}{DESCRIPTION_SUFFIX}")


def _parquet_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError(
            "add_description expects the path to a parquet file as its first "
            f"argument, not {type(path).__name__}. Save a DataFrame first, "
            'e.g. df.write_parquet("data/sales.parquet").'
        )
    path = Path(path).expanduser()
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"{str(path)!r} is not a parquet file (.parquet).")
    if not path.is_file():
        raise FileNotFoundError(f"There is no parquet file at {str(path)!r}.")
    return path


def _description_text(descr: str | Path) -> str:
    """The markdown in ``descr``: the text itself, or read from the file it names."""
    if isinstance(descr, Path):
        source = descr.expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"There is no description file at {str(source)!r}.")
        text = source.read_text(encoding="utf-8")
    elif isinstance(descr, str):
        text = descr
        if "\n" not in descr.strip():
            source = Path(descr.strip()).expanduser()
            if source.is_file():
                text = source.read_text(encoding="utf-8")
            elif source.suffix.lower() in _TEXT_SUFFIXES:
                raise FileNotFoundError(
                    f"There is no description file at {descr.strip()!r}. Pass the "
                    "markdown text itself, or the path to an existing file."
                )
    else:
        raise TypeError(
            "add_description expects the description as markdown text or the "
            f"path to a markdown file, not {type(descr).__name__}."
        )
    if not text.strip():
        raise ValueError("The description is empty.")
    return text


def read_description(path: str | Path) -> str | None:
    """The description stored with a parquet file, or ``None`` when it has none.

    Parameters
    ----------
    path : str | Path
        Path to a parquet file.

    Returns
    -------
    str | None
        The markdown inside the file, else the one in ``<name>_description.md``
        beside it, else ``None``.

    Examples
    --------
    >>> rsm.data.read_description("data/sales.parquet")
    """
    path = Path(path).expanduser()
    metadata = pl.read_parquet_metadata(str(path)) or {}
    embedded = metadata.get(DESCRIPTION_KEY)
    if isinstance(embedded, str) and embedded.strip():
        return embedded
    sidecar = description_path(path)
    if sidecar.is_file():
        text = sidecar.read_text(encoding="utf-8")
        if text.strip():
            return text
    return None


def add_description(path: str | Path, descr: str | Path) -> Path:
    """Store a markdown description with a parquet file.

    Parameters
    ----------
    path : str | Path
        Path to the parquet file to describe. The file is rewritten with the
        same data and the description in its metadata.
    descr : str | Path
        The description as markdown text, or the path to a markdown file that
        holds it (e.g. ``"data/sales_description.md"``).

    Returns
    -------
    Path
        The parquet file.

    Notes
    -----
    The description is stored inside the file (metadata key ``description``)
    and written beside it as ``<name>_description.md``, the convention Radiant
    and pyrsm's example data use. It replaces any earlier description; other
    metadata in the file is kept.

    Examples
    --------
    >>> rsm.data.add_description("data/sales.parquet", "# Sales\\n\\nMonthly sales by region.")
    >>> rsm.data.add_description("data/sales.parquet", "notes/sales.md")
    >>> rsm.data.read_description("data/sales.parquet")
    """
    path = _parquet_path(path)
    text = _description_text(descr)

    metadata = {
        key: value
        for key, value in (pl.read_parquet_metadata(str(path)) or {}).items()
        if key != _ARROW_SCHEMA_KEY
    }
    metadata[DESCRIPTION_KEY] = text
    frame = pl.read_parquet(str(path))

    # Written beside the original and moved over it, so a failure part-way
    # leaves the original file as it was.
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        frame.write_parquet(str(partial), metadata=metadata)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)

    description_path(path).write_text(text, encoding="utf-8")
    return path
