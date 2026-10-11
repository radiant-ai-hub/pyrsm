"""Filter, sort and slice: one language, and every class takes it.

Radiant-for-R threads the scope through every analysis function, so a script
reproduces what was on the screen and the printed summary says which rows it
used. These tests hold the three properties that makes true here:

* the three strings select the rows they say they do, in that order;
* the polars source generated from a filter selects the same rows, so a
  snippet is a reproduction rather than a description;
* an analysis that was given a scope prints it, because a summary that does
  not say it was filtered is one a reader will take for the whole dataset.
"""

from __future__ import annotations

import datetime as dt
import io
from contextlib import redirect_stdout

import polars as pl
import pytest

import pyrsm as rsm
from pyrsm.data_scope import (
    ScopeError,
    apply_scope,
    filter_code,
    get_data,
    slice_code,
    sort_code,
    sort_spec,
)


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "price": [1000, 6000, 8000, None, 2000],
            "carat": [0.2, 1.1, 2.0, 0.5, 3.0],
            "cut": ["Ideal", "Fair", "Ideal", "Good", "Premium"],
            "date": [
                dt.date(2024, 1, 1),
                dt.date(2024, 6, 1),
                dt.date(2023, 3, 1),
                dt.date(2025, 2, 1),
                dt.date(2024, 9, 1),
            ],
        }
    )


# --- the filter language ---------------------------------------------------


@pytest.mark.parametrize(
    ("data_filter", "expected"),
    [
        ("price > 5000", 2),
        ("price > 5000 and cut == 'Ideal'", 1),
        ("price > 5000 or carat > 2", 3),
        # A null is neither greater nor not-greater, so `not` drops it too.
        ("not (price > 5000)", 2),
        ("cut in ['Ideal', 'Premium']", 3),
        ("cut not in ['Ideal']", 3),
        ("is_null(price)", 1),
        ("is_not_null(price)", 4),
        ("contains(cut, 'ood')", 1),
        ("starts_with(cut, 'I')", 2),
        ("ends_with(cut, 'al')", 2),
        ("lower(cut) == 'ideal'", 2),
        ("carat > mean(carat)", 2),
        ("round(carat) == 2", 1),
        ("year(date) == 2024", 3),
        ("date < '2024-06-01'", 2),
        ("date in ['2024-01-01', '2023-03-01']", 2),
    ],
)
def test_the_filter_selects_the_rows_it_says(data_filter, expected, df):
    assert apply_scope(df, data_filter).height == expected


@pytest.mark.parametrize(
    "data_filter",
    [
        "price > 5000",
        "price > 5000 and cut == 'Ideal'",
        "cut in ['Ideal', 'Premium']",
        "not is_null(price)",
        "carat > mean(carat)",
        "date < '2024-06-01'",
        "year(date) == 2024",
    ],
)
def test_the_generated_polars_selects_the_same_rows(data_filter, df):
    """A snippet that does not reproduce the result is worse than none,
    because it looks like it does."""
    namespace = {"pl": pl, "data": df}
    exec(f"data = data.filter({filter_code(data_filter, dict(df.schema))})", namespace)  # noqa: S102

    assert namespace["data"].equals(apply_scope(df, data_filter))


@pytest.mark.parametrize(
    ("bad", "says"),
    [
        ("price = 5000", "invalid syntax"),
        ("price.is_null()", "not methods"),
        ("1 < price < 10", "two comparisons"),
        ("wibble(price)", "Unknown function"),
        ("cut in 'Ideal'", "needs a list"),
        ("date < 'not-a-date'", "not a date"),
    ],
)
def test_what_it_refuses_it_explains(bad, says, df):
    with pytest.raises(ScopeError, match=says):
        apply_scope(df, bad)


def test_an_unknown_column_is_named(df):
    with pytest.raises(ScopeError, match="Unknown column in filter: nosuch"):
        apply_scope(df, "nosuch > 1")


def test_an_empty_filter_changes_nothing(df):
    assert apply_scope(df, "").equals(df)
    assert filter_code("") == ""


# --- sort ------------------------------------------------------------------


def test_a_minus_means_descending():
    assert sort_spec("clarity -price") == (["clarity", "price"], [False, True])


def test_sort_orders_the_frame(df):
    out = apply_scope(df, "", "-carat")

    assert out["carat"].to_list() == sorted(df["carat"].to_list(), reverse=True)


def test_the_generated_sort_matches(df):
    namespace = {"pl": pl, "data": df}
    exec(sort_code("cut -price", "data"), namespace)  # noqa: S102

    assert namespace["data"].equals(apply_scope(df, "", "cut -price"))


# --- slice -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("1:3", [1000, 6000, 8000]), ("2", [1000, 6000]), ("-2", [None, 2000]),
     ("1 3 5", [1000, 8000, 2000]), ("2 2", [6000, 8000])],
)
def test_slice_counts_rows_from_one(spec, expected, df):
    assert apply_scope(df, "", "", spec)["price"].to_list() == expected


def test_the_generated_slice_matches(df):
    for spec in ("1:3", "2", "-2", "1 3 5"):
        namespace = {"pl": pl, "data": df}
        exec(slice_code(spec, "data"), namespace)  # noqa: S102
        assert namespace["data"].equals(apply_scope(df, "", "", spec)), spec


# --- the three together ----------------------------------------------------


def test_the_order_is_filter_then_sort_then_slice(df):
    """A slice of unsorted rows and a slice of sorted rows are different
    rows, so the order cannot be left to taste."""
    out = apply_scope(df, "is_not_null(price)", "-price", "1:2")

    assert out["price"].to_list() == [8000, 6000]


def test_get_data_keeps_the_columns_asked_for(df):
    name, out = get_data({"d": df}, vars=["price"], data_filter="price > 5000")

    assert name == "d"
    assert out.columns == ["price"]
    assert out.height == 2


# --- every class takes it --------------------------------------------------

CLASSES = [
    (rsm.model.regress, {"rvar": "price", "evar": ["carat"]}),
    (rsm.basics.single_mean, {"var": "price"}),
    (rsm.basics.correlation, {"vars": ["price", "carat"]}),
]


@pytest.mark.parametrize(("build", "kwargs"), CLASSES)
def test_an_analysis_runs_on_the_scoped_rows(build, kwargs, df):
    scoped = build({"d": df}, data_filter="price > 1500", **kwargs)

    assert scoped.data.height == 3


@pytest.mark.parametrize(("build", "kwargs"), CLASSES)
def test_the_summary_says_it_was_filtered(build, kwargs, df):
    """The number under a filtered analysis is not a fact about the data
    unless the filter that produced it is on the page beside it."""
    scoped = build({"d": df}, data_filter="price > 1500", sort="-carat", **kwargs)

    printed = io.StringIO()
    with redirect_stdout(printed):
        scoped.summary()
    out = printed.getvalue()

    assert "Filter" in out and "price > 1500" in out
    assert "Sort" in out and "-carat" in out


def test_every_class_that_takes_data_takes_the_scope():
    """The three arguments are the contract, so a class that reads a dataset
    and cannot be scoped is the odd one out -- and the one a student will
    hit when their filter silently does not apply."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "pyrsm"
    missing = []
    for path in sorted(root.rglob("*.py")):
        if path.name.startswith("_"):
            continue
        for node in ast.parse(path.read_text()).body:
            if not (isinstance(node, ast.ClassDef) and node.name[0].islower()):
                continue
            for item in node.body:
                if not (isinstance(item, ast.FunctionDef) and item.name == "__init__"):
                    continue
                args = {a.arg for a in item.args.args}
                if "data" not in args:
                    continue
                doc = ast.get_docstring(node) or ""
                if not {"data_filter", "sort", "slice"} <= args:
                    missing.append(f"{path.name}:{node.name} (arguments)")
                elif "data_filter : str" not in doc or "sort : str" not in doc:
                    missing.append(f"{path.name}:{node.name} (docstring)")

    assert not missing, "not scoped: " + ", ".join(missing)


@pytest.mark.parametrize(("build", "kwargs"), CLASSES)
def test_no_scope_means_no_extra_lines(build, kwargs, df):
    plain = build({"d": df}, **kwargs)

    printed = io.StringIO()
    with redirect_stdout(printed):
        plain.summary()

    assert "Filter" not in printed.getvalue()


def test_a_column_name_with_dots_is_one_column():
    """R data often has dotted names (ketchup's ``price.heinz28``); a filter
    reads them as one column, and the generated polars code does too."""
    df = pl.DataFrame(
        {"price.heinz28": [3.5, 4.5, 5.0], "disp.heinz28": ["no", "yes", "yes"]}
    )
    expr = "price.heinz28 > 4 and disp.heinz28 == 'yes'"
    assert apply_scope(df, expr)["price.heinz28"].to_list() == [4.5, 5.0]
    code = filter_code(expr, df.schema)
    assert "pl.col('price.heinz28')" in code
    data = df  # noqa: F841 -- the generated code refers to ``data``
    assert eval(f"data.filter({code})", {"pl": pl, "data": df}).height == 2
    assert apply_scope(df, "is_null(price.heinz28)").height == 0


def test_a_method_call_on_a_dotted_name_is_still_refused():
    df = pl.DataFrame({"price.heinz28": [3.5, None]})
    with pytest.raises(ScopeError, match="functions, not methods"):
        apply_scope(df, "price.heinz28.is_null()")


# --- the eda functions take it too ---------------------------------------


def test_explore_summarizes_the_scoped_rows(df):
    out = rsm.eda.explore({"d": df}, cols=["carat"], agg=["n", "max"], data_filter="price > 1500")
    assert out["n"][0] == 3 and out["max"][0] == 3.0
    out = rsm.eda.explore({"d": df}, cols=["carat"], agg=["n"], sort="-carat", slice="1:2")
    assert out["n"][0] == 2


def test_pivot_counts_the_scoped_rows(df):
    out = rsm.eda.pivot({"d": df}, rows="cut", data_filter="price > 1500", totals=True)
    assert out.filter(pl.col("cut") == "Total")["count"][0] == 3
    lazy = rsm.eda.pivot(df.lazy(), rows="cut", data_filter="cut == 'Ideal'")
    assert lazy["count"].to_list() == [2]


def test_visualize_plots_the_scoped_rows(df):
    p = rsm.eda.visualize({"d": df}, x="carat", y="price", data_filter="price > 1500", slice="1:2")
    assert p.data.height == 2
