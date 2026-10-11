"""``rsm.register`` outside Radiant: same argument checks, a note, nothing else."""

import pandas as pd
import polars as pl
import pytest

import pyrsm as rsm


def test_register_is_a_top_level_function():
    assert callable(rsm.register)
    assert "register" in dir(rsm)


def test_register_outside_radiant_prints_a_note_and_returns_none(capsys):
    df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    assert rsm.register(df, "sales_clean", descr="Clean sales") is None
    out = capsys.readouterr().out
    assert "`sales_clean`" in out
    assert "3 rows × 2 columns" in out
    assert "outside Radiant" in out


def test_register_accepts_pandas_and_lazy_frames(capsys):
    rsm.register(pd.DataFrame({"x": [1, 2]}), "from_pandas")
    rsm.register(pl.LazyFrame({"x": [1]}), "from_lazy")
    out = capsys.readouterr().out
    assert "2 rows × 1 columns" in out
    assert "1 rows × 1 columns" in out


@pytest.mark.parametrize(
    "args, error",
    [
        (([1, 2], "x"), TypeError),
        ((pl.DataFrame({"x": [1]}),), TypeError),
        ((pl.DataFrame({"x": [1]}), ""), TypeError),
        ((pl.DataFrame({"x": [1]}), "2nd"), ValueError),
        ((pl.DataFrame({"x": [1]}), "sales clean"), ValueError),
        ((pl.DataFrame({"x": [1]}), "class"), ValueError),
    ],
)
def test_register_rejects_bad_arguments(args, error):
    with pytest.raises(error):
        rsm.register(*args)
