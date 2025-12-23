#!/usr/bin/env python3
"""
Run all Jupyter notebooks and check for errors.

This script:
1. Finds all .ipynb files in the examples directory
2. Executes each notebook in-place (saves outputs back to the notebook)
3. Checks for error outputs in executed cells
4. Reports pass/fail status for each notebook

Usage:
    python scripts/run_notebooks.py [--path PATH] [--timeout SECONDS] [--stop-on-error]

Examples:
    python scripts/run_notebooks.py                    # Run all notebooks
    python scripts/run_notebooks.py --path examples/model  # Run only model notebooks
    python scripts/run_notebooks.py --stop-on-error    # Stop on first error
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def find_notebooks(base_path: Path) -> list[Path]:
    """Find all .ipynb files recursively, excluding checkpoints."""
    notebooks = []
    for nb in sorted(base_path.rglob("*.ipynb")):
        if ".ipynb_checkpoints" not in str(nb):
            notebooks.append(nb)
    return notebooks


def execute_notebook(notebook_path: Path, timeout: int = 600) -> tuple[bool, str]:
    """
    Execute a notebook in-place using jupyter nbconvert.

    Returns:
        (success: bool, message: str)
    """
    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(notebook_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 60  # Extra buffer for nbconvert overhead
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            return False, f"Execution failed: {error_msg[:500]}"

        return True, "Executed successfully"

    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout} seconds"
    except Exception as e:
        return False, f"Error: {str(e)}"


def check_notebook_errors(notebook_path: Path) -> list[dict]:
    """
    Check a notebook for error outputs in executed cells.

    Returns:
        List of error info dicts with cell_index and error details
    """
    errors = []

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        return [{"cell_index": -1, "error": f"Failed to read notebook: {e}"}]

    cells = nb.get("cells", [])
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        outputs = cell.get("outputs", [])
        for output in outputs:
            # Check for error output type
            if output.get("output_type") == "error":
                ename = output.get("ename", "Unknown")
                evalue = output.get("evalue", "")
                errors.append({
                    "cell_index": idx + 1,  # 1-indexed for user display
                    "error": f"{ename}: {evalue}"
                })

            # Check for stderr with error-like content
            if output.get("output_type") == "stream" and output.get("name") == "stderr":
                text = output.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                # Only flag if it looks like a real error (not just a warning)
                if "Error" in text or "Exception" in text or "Traceback" in text:
                    errors.append({
                        "cell_index": idx + 1,
                        "error": f"stderr: {text[:200]}"
                    })

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Run all Jupyter notebooks and check for errors"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="examples",
        help="Base path to search for notebooks (default: examples)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per notebook in seconds (default: 600)"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop execution on first error"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check for errors in already-executed notebooks (no execution)"
    )

    args = parser.parse_args()

    base_path = Path(args.path)
    if not base_path.exists():
        print(f"Error: Path '{base_path}' does not exist")
        sys.exit(1)

    notebooks = find_notebooks(base_path)
    if not notebooks:
        print(f"No notebooks found in '{base_path}'")
        sys.exit(0)

    print(f"Found {len(notebooks)} notebooks in '{base_path}'")
    print("=" * 70)

    results = {
        "passed": [],
        "failed": [],
        "errors": []
    }

    for i, nb_path in enumerate(notebooks, 1):
        rel_path = nb_path.relative_to(Path.cwd()) if nb_path.is_absolute() else nb_path
        print(f"\n[{i}/{len(notebooks)}] {rel_path}")

        if not args.check_only:
            # Execute the notebook
            print("  Executing...", end=" ", flush=True)
            success, message = execute_notebook(nb_path, args.timeout)

            if not success:
                print(f"FAILED")
                print(f"  Error: {message}")
                results["failed"].append((rel_path, message))

                if args.stop_on_error:
                    print("\nStopping due to --stop-on-error flag")
                    break
                continue

            print("OK", end="")

        # Check for errors in the notebook outputs
        errors = check_notebook_errors(nb_path)

        if errors:
            if not args.check_only:
                print(" (with errors in output)")
            else:
                print("  ERRORS FOUND")

            for err in errors:
                print(f"    Cell {err['cell_index']}: {err['error'][:100]}")

            results["errors"].append((rel_path, errors))

            if args.stop_on_error:
                print("\nStopping due to --stop-on-error flag")
                break
        else:
            if args.check_only:
                print("  OK - no errors")
            else:
                print()
            results["passed"].append(rel_path)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total = len(notebooks)
    passed = len(results["passed"])
    failed = len(results["failed"])
    with_errors = len(results["errors"])

    print(f"Total notebooks: {total}")
    print(f"  Passed:        {passed}")
    print(f"  Failed:        {failed}")
    print(f"  With errors:   {with_errors}")

    if results["failed"]:
        print("\nFailed notebooks:")
        for nb, msg in results["failed"]:
            print(f"  - {nb}")
            print(f"    {msg[:100]}")

    if results["errors"]:
        print("\nNotebooks with errors in output:")
        for nb, errors in results["errors"]:
            print(f"  - {nb}")
            for err in errors[:3]:  # Show first 3 errors
                print(f"      Cell {err['cell_index']}: {err['error'][:80]}")
            if len(errors) > 3:
                print(f"      ... and {len(errors) - 3} more errors")

    # Exit code
    if failed > 0 or with_errors > 0:
        sys.exit(1)

    print("\nAll notebooks passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()
