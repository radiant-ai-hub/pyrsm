"""pyrsm.data.compare_frames: strict equality plus actionable diagnostics."""

import pandas as pd
import polars as pl
import pytest

from pyrsm.data import compare_frames


@pytest.fixture
def expected():
    return pl.DataFrame(
        {
            "acctnum": ["10001", "10002", "10003", "10004"],
            "zip": ["02139", "92093", "10001", "60601"],
            "zip3": ["021", "920", "100", "606"],
            "first": pl.Series([10, 20, 30, 40], dtype=pl.Int32),
            "buyer": pl.Series(["yes", "no", "no", "yes"], dtype=pl.Categorical),
        }
    )


def test_identical_frames_are_equal(expected):
    result = compare_frames(expected, expected.clone())
    assert result.equal
    assert result.messages == []
    assert result.summary().startswith("The data frames are identical")


def test_missing_column_is_named(expected):
    result = compare_frames(expected, expected.drop("zip3"))
    assert not result.equal
    assert result.missing_columns == ["zip3"]
    assert any("`zip3` is missing" in m for m in result.messages)


def test_wrong_dtype_with_equal_values_is_reported_not_coerced(expected):
    actual = expected.with_columns(pl.col("first").cast(pl.Int64), pl.col("buyer").cast(pl.String))
    result = compare_frames(expected, actual)
    assert not result.equal
    assert result.polars_equals  # Polars' equals() ignores dtypes; strict does not
    by = {c.name: c for c in result.columns}
    assert by["first"].dtype_actual == "Int64" and by["first"].equal_as_text is True
    assert by["buyer"].dtype_expected == "Categorical" and by["buyer"].equal_as_text is True
    assert any("`first` is Int64, expected Int32" in m for m in result.messages)


def test_value_mismatches_are_counted_with_capped_examples(expected):
    actual = expected.with_columns(pl.Series("first", [10, 21, 31, 40], dtype=pl.Int32))
    result = compare_frames(expected, actual, max_examples=1)
    first = {c.name: c for c in result.columns}["first"]
    assert first.value_mismatches == 2
    assert first.examples == [{"row": 1, "expected": 20, "actual": 21}]
    assert compare_frames(expected, actual, show_values=False).columns[3].examples[0] == {"row": 1}


def test_column_order(expected):
    actual = expected.select(["zip", "acctnum", "zip3", "first", "buyer"])
    strict = compare_frames(expected, actual)
    assert not strict.equal and not strict.column_order_equal
    assert any("different order" in m for m in strict.messages)
    assert compare_frames(expected, actual, strict=False).equal


def test_keyed_comparison_reports_duplicates_and_unmatched(expected):
    actual = pl.concat([expected.head(3), expected.slice(2, 1)])  # 10003 twice, 10004 missing
    result = compare_frames(expected, actual, key="acctnum")
    assert result.key.duplicates_actual == 1
    assert result.key.examples_duplicate_actual == ["10003"]
    assert result.key.unmatched_expected == 1
    assert result.key.examples_unmatched_expected == ["10004"]
    assert any("appear more than once" in m for m in result.messages)


def test_keyed_comparison_matches_rows_regardless_of_order(expected):
    shuffled = expected.reverse().with_columns(
        pl.when(pl.col("acctnum") == "10002").then(99).otherwise(pl.col("first")).cast(pl.Int32).alias("first")
    )
    result = compare_frames(expected, shuffled, key="acctnum")
    first = {c.name: c for c in result.columns}["first"]
    assert first.value_mismatches == 1
    assert first.examples[0]["key"] == "10002"
    assert not result.equal  # row order and one value


def test_nulls_count_as_differences(expected):
    actual = expected.with_columns(pl.Series("zip3", ["021", None, "100", "606"]))
    zip3 = {c.name: c for c in compare_frames(expected, actual).columns}["zip3"]
    assert zip3.nulls_actual == 1 and zip3.value_mismatches == 1


def test_shape_difference_without_key(expected):
    result = compare_frames(expected, expected.head(2))
    assert any(m.startswith("Shape differs") for m in result.messages)
    assert all(c.value_mismatches is None for c in result.columns)


def test_pandas_input_is_converted_and_flagged(expected):
    result = compare_frames(expected.to_pandas(), expected.to_pandas())
    assert result.converted_from_pandas and result.equal


def test_to_dict_is_json_ready(expected):
    import json

    json.dumps(compare_frames(expected, expected.drop("zip")).to_dict())
