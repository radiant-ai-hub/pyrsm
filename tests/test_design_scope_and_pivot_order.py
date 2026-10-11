"""Sampling and random assignment say which rows they used; their tables print as
tables; a pivot lists rows and columns in the order of the variables' levels."""

import contextlib
import io

import polars as pl

import pyrsm as rsm


def _text(obj, **kw) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        obj.summary(**kw)
    return buf.getvalue()


DF = pl.DataFrame({"id": range(1, 101), "g": ["x", "y"] * 50})


def test_sampling_prints_its_rows():
    s = rsm.design.sampling({"d": DF}, vars=["id"], sample_size=5, data_filter="id > 10", sort="-id", slice="1:50")
    text = _text(s)
    assert "Data       : d\nFilter     : id > 10\nSort       : -id\nSlice      : 1:50\nVariables  : id" in text
    assert s.selected["id"].max() <= 100 and s.selected["id"].min() >= 51


def test_randomizer_prints_its_rows_and_tables():
    r = rsm.design.randomizer({"d": DF}, vars=["id"], conditions=["A", "B"], data_filter="id > 20")
    text = _text(r)
    assert "Data         : d\nFilter       : id > 20\nVariables    : id" in text
    assert r.data.height == 80
    assert "Assignment frequencies:\n┌" in text and "Assignment proportions:\n┌" in text
    assert "│ Total" in text


def test_pivot_rows_and_columns_follow_the_levels():
    df = pl.DataFrame({
        "a": pl.Series(["3rd", "1st", "2nd", "1st", "3rd"], dtype=pl.Enum(["1st", "2nd", "3rd"])),
        "b": pl.Series(["no", "yes", "no", "no", "yes"], dtype=pl.Enum(["yes", "no"])),
    })
    out = rsm.eda.pivot(df, rows="a", cols="b", totals=True)
    assert out["a"].to_list() == ["1st", "2nd", "3rd", "Total"]
    assert out.columns == ["a", "yes", "no", "Total"]
    freq = rsm.eda.pivot(df, rows="a")
    assert freq["a"].cast(pl.Utf8).to_list() == ["1st", "2nd", "3rd"]
