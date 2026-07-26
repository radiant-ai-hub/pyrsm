"""Smoke tests for the seven multivariate teaching notebooks.

These catch the cheap failures (missing files, code cells that no longer parse)
without a full render. Set ``PYRSM_RUN_NOTEBOOKS=1`` to additionally execute the
generated ``.ipynb`` end-to-end (slow; needs network for the data URLs).
"""

import ast
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
QMD_DIR = REPO / "examples" / "qmd" / "multivariate"
IPYNB_DIR = REPO / "examples" / "multivariate"

NOTEBOOKS = [
    "multivariate-pre-factor",
    "multivariate-full-factor",
    "multivariate-hclus",
    "multivariate-kclus",
    "multivariate-mds",
    "multivariate-prmap",
    "multivariate-conjoint",
    "multivariate-cbc",
]

_CODE_BLOCK = re.compile(r"```\{python\}(.*?)```", re.DOTALL)


def test_all_notebooks_present():
    missing = [
        n
        for n in NOTEBOOKS
        if not (QMD_DIR / f"{n}.qmd").exists() or not (IPYNB_DIR / f"{n}.ipynb").exists()
    ]
    assert not missing, f"missing notebooks: {missing}"


def test_no_legacy_combined_notebooks():
    for stale in (
        "multivariate-factor-analysis",
        "multivariate-cluster-analysis",
        "multivariate-maps",
    ):
        assert not (QMD_DIR / f"{stale}.qmd").exists()
        assert not (IPYNB_DIR / f"{stale}.ipynb").exists()


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_qmd_code_cells_parse(name):
    """Every ```{python}``` block must be syntactically valid Python.

    Magics (``%reload_ext`` etc.) are stripped before parsing.
    """
    text = (QMD_DIR / f"{name}.qmd").read_text()
    blocks = _CODE_BLOCK.findall(text)
    assert blocks, f"{name}: no python code cells found"
    for block in blocks:
        lines = [
            ln for ln in block.splitlines() if not ln.lstrip().startswith(("%", "!"))
        ]
        src = "\n".join(lines)
        ast.parse(src)  # raises SyntaxError on malformed cells


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_qmd_uses_pyrsm_multivariate(name):
    text = (QMD_DIR / f"{name}.qmd").read_text()
    assert "rsm.multivariate" in text


@pytest.mark.skipif(
    os.environ.get("PYRSM_RUN_NOTEBOOKS") != "1",
    reason="set PYRSM_RUN_NOTEBOOKS=1 to execute notebooks (slow, needs network)",
)
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_executes(name):
    import subprocess
    import sys

    nb = IPYNB_DIR / f"{name}.ipynb"
    env = {**os.environ, "MPLBACKEND": "Agg"}
    subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
            "--execute", "--stdout", "--ExecutePreprocessor.timeout=300", str(nb),
        ],
        check=True,
        capture_output=True,
        env=env,
    )
