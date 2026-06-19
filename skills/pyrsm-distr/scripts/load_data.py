"""Probe a data file: detect its format, load it with polars, find any sidecar
description markdown, and print a small JSON report.

Usage:
    python scripts/load_data.py "<absolute-path-to-data-file>"

The output JSON has these fields:
    path           absolute path to the data file
    exists         bool
    file_type      one of: parquet, csv, tsv, feather, arrow, xlsx, json, unknown
    shape          [n_rows, n_cols] (omitted if loading failed)
    columns        list of {name, dtype} (omitted if loading failed)
    sidecar_md     absolute path to the sidecar description .md, or null
    sidecar_md_candidates_searched   list of patterns we tried
    load_error     error message if loading failed (omitted on success)

We deliberately keep this script small and dependency-light: polars is the only
non-stdlib import. If polars cannot read the file (e.g., xlsx without the right
extra), we still report the metadata we can.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl


EXT_TO_TYPE = {
    ".parquet": "parquet",
    ".pq": "parquet",
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "tsv",
    ".feather": "feather",
    ".arrow": "arrow",
    ".ipc": "arrow",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
    ".json": "json",
    ".ndjson": "ndjson",
    ".jsonl": "ndjson",
}


def detect_file_type(path: Path) -> str:
    return EXT_TO_TYPE.get(path.suffix.lower(), "unknown")


def load_dataframe(path: Path, file_type: str) -> pl.DataFrame:
    """Load a file into a polars DataFrame. Raises on unsupported types."""
    if file_type == "parquet":
        return pl.read_parquet(path)
    if file_type == "csv":
        return pl.read_csv(path)
    if file_type == "tsv":
        return pl.read_csv(path, separator="\t")
    if file_type in ("feather", "arrow"):
        return pl.read_ipc(path)
    if file_type == "xlsx":
        return pl.read_excel(path)
    if file_type == "json":
        return pl.read_json(path)
    if file_type == "ndjson":
        return pl.read_ndjson(path)
    raise ValueError(
        f"Unsupported file extension '{path.suffix}'. "
        "Try one of: .parquet, .csv, .tsv, .feather, .arrow, .xlsx, .json, .ndjson."
    )


def find_sidecar_md(data_path: Path) -> tuple[Path | None, list[str]]:
    """Look for a markdown description file next to `data_path` whose filename
    is "very similar" to the data filename.

    The search proceeds in order of decreasing specificity. The first existing
    match wins. Returns (path or None, list of patterns we searched).
    """
    folder = data_path.parent
    stem = data_path.stem  # e.g. "catalog" from "catalog.parquet"

    patterns = [
        f"{stem}_description.md",   # pyrsm convention
        f"{stem}-description.md",
        f"{stem}.description.md",
        f"{stem}_readme.md",
        f"{stem}-readme.md",
        f"README_{stem}.md",
        f"readme_{stem}.md",
        f"{stem}.md",
    ]

    for name in patterns:
        candidate = folder / name
        if candidate.exists():
            return candidate, patterns

    # Case-insensitive fallback: glob the folder and look for any .md whose
    # stem starts with the data stem (e.g. "Catalog_DESCRIPTION.md").
    stem_lc = stem.lower()
    for md in folder.glob("*.md"):
        md_stem_lc = md.stem.lower()
        if md_stem_lc == stem_lc or md_stem_lc.startswith(stem_lc + "_") or md_stem_lc.startswith(stem_lc + "-"):
            return md, patterns

    return None, patterns


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python scripts/load_data.py <absolute-path-to-data-file>", file=sys.stderr)
        return 2

    raw = argv[1]
    path = Path(raw)
    report: dict = {
        "path": str(path),
        "exists": path.exists(),
    }

    if not path.is_absolute():
        report["load_error"] = (
            f"Path is not absolute: {raw!r}. Please pass an absolute path "
            "(starting with '/') so there is no ambiguity about the working directory."
        )
        print(json.dumps(report, indent=2))
        return 1

    if not path.exists():
        report["load_error"] = f"File does not exist: {raw}"
        print(json.dumps(report, indent=2))
        return 1

    file_type = detect_file_type(path)
    report["file_type"] = file_type

    try:
        df = load_dataframe(path, file_type)
        report["shape"] = list(df.shape)
        report["columns"] = [
            {"name": c, "dtype": str(df[c].dtype)} for c in df.columns
        ]
    except Exception as e:  # noqa: BLE001  surfacing the message is the point
        report["load_error"] = f"{type(e).__name__}: {e}"

    sidecar, patterns = find_sidecar_md(path)
    report["sidecar_md"] = str(sidecar) if sidecar else None
    report["sidecar_md_candidates_searched"] = patterns

    print(json.dumps(report, indent=2))
    return 0 if "load_error" not in report else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
