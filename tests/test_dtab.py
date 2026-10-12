"""rsm.dtab: Radiant's interactive result table in notebooks and HTML reports."""

import json
import re

import polars as pl
import pytest

import pyrsm as rsm
import pyrsm.interactive_table as it
from pyrsm.interactive_table import format_cell


@pytest.fixture(autouse=True)
def fresh_kernel(monkeypatch):
    """Each test is a new kernel: the first table carries Tabulator."""
    monkeypatch.setattr(it, "_LIB_SENT", False)


def _spec(html: str) -> dict:
    found = re.search(r'<script type="application/json" class="rsm-dtab-spec">(.*?)</script>', html, re.S)
    return json.loads(found.group(1).replace("<\\/", "</"))


@pytest.mark.parametrize(
    ("value", "kind", "dec", "shown"),
    [
        (None, "number", 3, ""),
        (5, "number", 3, "5"),
        (5.0, "number", 3, "5"),
        (1.23456, "number", 3, "1.235"),
        (0.0001234, "number", 3, "0.000123"),
        (0.2341, "percent", 2, "23.41%"),
        (float("nan"), "number", 3, "NaN"),
        (float("-inf"), "number", 3, "-Inf"),
        ("Fair", "text", 3, "Fair"),
    ],
)
def test_cells_read_as_in_radiant(value, kind, dec, shown):
    assert format_cell(value, kind, dec) == shown


def test_a_pivot_table_keeps_totals_below_and_spans_the_levels():
    pv = pl.DataFrame({"cut": ["Fair", "Good", "Total"], "D": [1, 2, 3], "E": [3, 4, 7], "Total": [4, 6, 10]})
    spec = rsm.dtab(pv, label_columns=["cut"], span="color", totals=True, format="heat").spec
    assert [r["cut"] for r in spec["rows"]] == ["Fair", "Good"]
    assert spec["totals"]["Total"] == 10 and spec["totalRows"] == 2
    assert spec["span"] == {"title": "color", "columns": ["D", "E"]}
    assert spec["boldColumns"] == ["Total"] and spec["conditionalColumns"] == ["D", "E"]


def test_percentage_strings_become_numbers_shown_as_percentages():
    pv = pl.DataFrame({"cut": ["Fair", "Good"], "D": ["12.50%", "87.50%"]})
    spec = rsm.dtab(pv, label_columns=["cut"]).spec
    assert {c["name"]: c["kind"] for c in spec["columns"]} == {"cut": "text", "D": "percent"}
    assert spec["rows"][0]["D"] == pytest.approx(0.125)


def test_nr_keeps_the_first_rows_and_says_how_many_there_are():
    df = pl.DataFrame({"x": list(range(250))})
    tab = rsm.dtab(df, nr=100)
    assert len(tab.spec["rows"]) == 100 and tab.spec["totalRows"] == 250
    assert "(the first 100 of 250)" in tab._repr_html_()


def test_the_first_table_carries_tabulator_and_later_ones_wait_for_it():
    df = pl.DataFrame({"x": [1.5, 2.5]})
    first, second = rsm.dtab(df)._repr_html_(), rsm.dtab(df)._repr_html_()
    assert "Tabulator" in first and "rsmDtabRun" in first
    assert len(second) < len(first) / 50, "Tabulator is sent once per kernel"
    assert "window.rsmDtabRun" in second  # retries until the library has loaded


def test_without_scripts_the_first_page_shows_as_a_plain_table():
    df = pl.DataFrame({"x": list(range(30)), "g": ["a", "b", "c"] * 10})
    html = rsm.dtab(df, page_length=10)._repr_html_()
    static = re.search(r'<table class="rsm-dtab-static">.*?</table>', html, re.S).group(0)
    assert static.count("<tr>") == 11  # header + one page
    assert _spec(html)["rows"][29] == {"x": 29, "g": "c"}


def test_markdown_for_pdf_and_word():
    df = pl.DataFrame({"g": ["a", "b"], "x": [1.23456, -2.0]})
    md = rsm.dtab(df, dec=2, caption="Means")._repr_markdown_()
    assert "**Means**" in md and "| a | 1.23 |" in md and "−2" in md


def test_a_script_tag_in_the_data_cannot_end_the_spec():
    df = pl.DataFrame({"t": ["</script><script>alert(1)</script>"]})
    html = rsm.dtab(df)._repr_html_()
    assert "</script><script>alert" not in html
    assert _spec(html)["rows"][0]["t"].startswith("</script>")
