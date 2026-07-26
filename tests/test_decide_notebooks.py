"""Smoke tests for Decide teaching notebooks."""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
QMD_DIR = REPO / "examples" / "qmd" / "decide"
IPYNB_DIR = REPO / "examples" / "decide"

NOTEBOOKS = [
    "decide-dtree",
    "decide-simulate",
]

_CODE_BLOCK = re.compile(r"```\{python\}(.*?)```", re.DOTALL)


def test_decide_notebooks_present():
    missing = [
        n
        for n in NOTEBOOKS
        if not (QMD_DIR / f"{n}.qmd").exists() or not (IPYNB_DIR / f"{n}.ipynb").exists()
    ]
    assert not missing, f"missing notebooks: {missing}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_decide_qmd_code_cells_parse(name):
    text = (QMD_DIR / f"{name}.qmd").read_text()
    blocks = _CODE_BLOCK.findall(text)
    assert blocks, f"{name}: no python code cells found"
    for block in blocks:
        lines = [
            ln for ln in block.splitlines() if not ln.lstrip().startswith(("%", "!"))
        ]
        src = "\n".join(lines)
        ast.parse(src)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_decide_qmd_uses_pyrsm_decide(name):
    text = (QMD_DIR / f"{name}.qmd").read_text()
    assert "rsm.decide" in text


@pytest.mark.skipif(
    os.environ.get("PYRSM_RUN_NOTEBOOKS") != "1",
    reason="set PYRSM_RUN_NOTEBOOKS=1 to execute notebooks",
)
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_decide_notebook_executes(name):
    import subprocess
    import sys

    nb = IPYNB_DIR / f"{name}.ipynb"
    env = {**os.environ, "MPLBACKEND": "Agg"}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--stdout",
            "--ExecutePreprocessor.timeout=300",
            str(nb),
        ],
        check=True,
        capture_output=True,
        env=env,
    )
