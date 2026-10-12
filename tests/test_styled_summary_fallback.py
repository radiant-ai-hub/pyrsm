"""``summary(plain=False)`` without great_tables prints the plain tables.

A report kernel or notebook that lacks the ``tables`` extra used to fail with
"great_tables is required for `.style`"; it now falls back to the text tables.
"""

import polars as pl
import pytest

import pyrsm as rsm
from pyrsm.basics import display_utils as du

salary = pl.DataFrame(
    {
        "rank": ["Prof", "AsstProf", "Prof", "AsstProf", "AssocProf", "AssocProf"] * 5,
        "salary": [150.0, 80.0, 140.0, 85.0, 100.0, 95.0] * 5,
    }
).with_columns(pl.col("salary") + pl.int_range(30).cast(pl.Float64))


@pytest.fixture
def notebook_without_great_tables(monkeypatch):
    monkeypatch.setattr(du, "is_notebook", lambda: True)
    monkeypatch.setattr(du, "has_great_tables", lambda: False)


def test_compare_means_falls_back_to_plain_text(notebook_without_great_tables, capsys):
    cm = rsm.basics.compare_means({"salary": salary}, var1="rank", var2="salary")
    cm.summary(extra=True, plain=False)
    out = capsys.readouterr().out
    assert "Pairwise mean comparisons" in out and "t.value" in out


def test_correlation_falls_back_to_plain_text(notebook_without_great_tables, capsys):
    data = salary.with_columns(other=pl.col("salary") * 2 + pl.int_range(30) % 7)
    rsm.basics.correlation({"salary": data}, vars=["salary", "other"]).summary(plain=False)
    assert "Correlation" in capsys.readouterr().out
