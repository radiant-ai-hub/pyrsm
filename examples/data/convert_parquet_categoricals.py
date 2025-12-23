#!/usr/bin/env python3
"""
Convert categorical columns in parquet files to Polars Enum type.

Usage:
    # Dry run (default) - shows what would be modified
    python convert_parquet_categoricals.py

    # Apply changes
    python convert_parquet_categoricals.py --apply

    # Specific directory
    python convert_parquet_categoricals.py --path /some/dir --apply
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import polars as pl


def get_backup_path(filepath: Path) -> Path:
    """Find next available backup filename with numbered suffixes."""
    bak = filepath.parent / f"{filepath.name}.bak"
    if not bak.exists():
        return bak
    i = 1
    while True:
        numbered = filepath.parent / f"{filepath.name}.bak.{i}"
        if not numbered.exists():
            return numbered
        i += 1


def process_parquet(filepath: Path, apply: bool) -> dict:
    """
    Process a single parquet file.

    Returns dict with:
        - path: filepath
        - cat_columns: dict of {col_name: [levels]}
        - status: 'success' | 'error'
        - error: error message if status == 'error'
        - backup_path: path to backup (if apply=True)
    """
    result = {
        "path": filepath,
        "cat_columns": {},
        "status": "success",
        "error": None,
        "backup_path": None,
    }

    try:
        # Load with pandas to get categorical metadata
        pdf = pd.read_parquet(filepath)

        # Extract categorical column info (preserving level order)
        # Check for whitespace issues in string-based levels
        cat_info = {}
        cols_needing_strip = []
        for col in pdf.columns:
            if pdf[col].dtype.name == "category":
                categories = pdf[col].cat.categories
                # Check if categories are string-based
                if categories.dtype == "object" or str(categories.dtype) == "string":
                    levels = [str(lvl) for lvl in categories]
                    stripped = [lvl.strip() for lvl in levels]
                    # Check if any level has leading/trailing whitespace
                    if levels != stripped:
                        cols_needing_strip.append(col)
                        cat_info[col] = stripped
                    else:
                        cat_info[col] = levels
                else:
                    # Non-string categories (numeric, etc.) - convert to string for Enum
                    cat_info[col] = [str(lvl) for lvl in categories]

        result["cat_columns"] = cat_info
        result["cols_stripped"] = cols_needing_strip

        if apply:
            # Only strip columns that need it
            for col in cols_needing_strip:
                categories = pdf[col].cat.categories
                stripped_cats = [str(c).strip() for c in categories]
                pdf[col] = pdf[col].cat.rename_categories(
                    dict(zip(categories, stripped_cats))
                )

            # Strip object columns only if they have whitespace issues
            for col in pdf.columns:
                if pdf[col].dtype == "object":
                    if pdf[col].dropna().apply(lambda x: isinstance(x, str)).all():
                        # Check if any value has leading/trailing whitespace
                        has_whitespace = pdf[col].dropna().apply(
                            lambda x: x != x.strip()
                        ).any()
                        if has_whitespace:
                            pdf[col] = pdf[col].str.strip()

            # Convert to polars
            df = pl.from_pandas(pdf)

            # Cast categorical columns to Enum with stripped level order
            for col, levels in cat_info.items():
                enum_dtype = pl.Enum(levels)
                df = df.with_columns(pl.col(col).cast(enum_dtype))

            # Create backup
            backup_path = get_backup_path(filepath)
            shutil.copy2(filepath, backup_path)
            result["backup_path"] = backup_path

            # Save back
            df.write_parquet(filepath)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert categorical columns in parquet files to Polars Enum type."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is dry-run)",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd(),
        help="Root directory to search (default: current directory)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=[".venv", "node_modules", "__pycache__", ".git"],
        help="Directories to exclude (default: .venv node_modules __pycache__ .git)",
    )
    args = parser.parse_args()

    root = args.path.resolve()
    if not root.exists():
        print(f"Error: Path does not exist: {root}")
        return 1

    # Find all parquet files (skip .bak files and excluded directories)
    exclude_set = set(args.exclude)
    parquet_files = [
        p for p in root.rglob("*.parquet")
        if ".bak" not in p.suffixes
        and not any(part in exclude_set for part in p.parts)
    ]

    if not parquet_files:
        print(f"No parquet files found in {root}")
        return 0

    mode = "APPLY MODE" if args.apply else "DRY RUN"
    print(f"\n{'=' * 60}")
    print(f"  Parquet Categorical → Enum Conversion ({mode})")
    print(f"{'=' * 60}")
    print(f"  Root: {root}")
    print(f"  Files found: {len(parquet_files)}")
    print(f"{'=' * 60}\n")

    results = []
    for filepath in sorted(parquet_files):
        result = process_parquet(filepath, apply=args.apply)
        results.append(result)

        rel_path = filepath.relative_to(root)

        if result["status"] == "error":
            print(f"ERROR: {rel_path}")
            print(f"       {result['error']}")
        else:
            cat_count = len(result["cat_columns"])
            cols_stripped = result.get("cols_stripped", [])
            if cat_count > 0:
                print(f"  {rel_path}")
                for col, levels in result["cat_columns"].items():
                    level_preview = str(levels[:5])
                    if len(levels) > 5:
                        level_preview = level_preview[:-1] + ", ...]"
                    stripped_marker = " [STRIPPED]" if col in cols_stripped else ""
                    print(f"    - {col}: {len(levels)} levels {level_preview}{stripped_marker}")
                if args.apply:
                    print(f"    → Backup: {result['backup_path'].name}")
                    print(f"    → Saved with Enum dtype")
            else:
                print(f"  {rel_path} (no categoricals)")
                if args.apply:
                    print(f"    → Backup: {result['backup_path'].name}")
                    print(f"    → Resaved")

    # Summary
    print(f"\n{'=' * 60}")
    success = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")
    with_cats = sum(1 for r in results if r["cat_columns"])
    cols_stripped = sum(len(r.get("cols_stripped", [])) for r in results)

    print(f"  Processed: {success}/{len(results)} files")
    print(f"  With categoricals: {with_cats}")
    if cols_stripped:
        print(f"  Columns stripped: {cols_stripped}")
    if errors:
        print(f"  Errors: {errors}")

    if not args.apply:
        print(f"\n  This was a DRY RUN. Use --apply to make changes.")
    print(f"{'=' * 60}\n")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    exit(main())
