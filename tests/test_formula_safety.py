"""Model formulas are built only from terms that cannot run code.

patsy evaluates each factor of a formula as Python. A column name or an
interaction term chosen by a user (an uploaded file, a web form) must never
reach it unchecked.
"""

from pathlib import Path

import polars as pl
import pytest

import pyrsm as rsm
from pyrsm.model.model import build_formula, check_formula, formula_term


def _data(bad: str) -> pl.DataFrame:
    return pl.DataFrame({"y": [1.0, 2.0, 4.0, 3.0, 6.0, 5.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0, 7.0],
                         bad: [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]})


def _payload(marker: Path) -> str:
    return f"__import__('pathlib').Path({str(marker)!r}).touch() or x"


@pytest.mark.parametrize("fit", [rsm.model.regress, rsm.model.logistic])
def test_a_column_name_never_runs(tmp_path, fit):
    marker = tmp_path / "ran"
    bad = _payload(marker)
    df = _data(bad)
    if fit is rsm.model.logistic:
        df = df.with_columns((pl.col("y") > 3).cast(pl.Int64).alias("y"))
        kwargs = {"lev": 1}
    else:
        kwargs = {}
    with pytest.raises(ValueError, match="cannot be used in a model"):
        fit({"d": df}, rvar="y", evar=["x", bad], **kwargs)
    with pytest.raises(ValueError, match="cannot be used in a model"):
        fit({"d": df}, rvar=bad, evar=["x"], **kwargs)
    assert not marker.exists()


def test_an_interaction_term_never_runs(tmp_path):
    marker = tmp_path / "ran"
    df = _data("z")
    for term in (f"I(__import__('pathlib').Path({str(marker)!r}).touch())", "x:open", "x:z:I(x)", "C(x)", "x*z"):
        with pytest.raises(ValueError):
            rsm.model.regress({"d": df}, rvar="y", evar=["x", "z"], ivar=[term])
    assert not marker.exists()


def test_safe_terms_still_fit():
    df = _data("z")
    fit = rsm.model.regress({"d": df}, rvar="y", evar=["x", "z"], ivar=["x:z", "I(x^2)"])
    assert fit.formula == "y ~ x + z + x:z + I(x ** 2)"
    assert build_formula("y", [], None, df.columns) == "y ~ 1"
    assert formula_term("I(x ** 3)", df.columns) == "I(x ** 3)"


def test_a_typed_formula_is_checked_without_evaluating(tmp_path):
    marker = tmp_path / "ran"
    cols = ["y", "x", "z"]
    assert check_formula("y ~ x + z + x:z + I(x^2) - 1", cols) == "y ~ x + z + x:z + I(x ** 2) - 1"
    for formula in (f"y ~ {_payload(marker)}", "y ~ np.log(x)", "y ~ C(x)", "x", "y ~ w", "y ~ Q('x')"):
        with pytest.raises(ValueError):
            check_formula(formula, cols)
    assert not marker.exists()
