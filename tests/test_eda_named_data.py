"""EDA functions take ``{name: frame}``, explore takes categoricals by name,
and a mapped fill (e.g. combined variables) is not overwritten by a default."""

import polars as pl
import pytest

import pyrsm as rsm


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "age": [22.0, 38.0, 26.0, 35.0, 54.0, 2.0],
            "fare": [7.2, 71.3, 7.9, 53.1, 51.9, 21.1],
            "sex": pl.Series(["m", "f", "f", "f", "m", "m"]).cast(pl.Enum(["f", "m"])),
        }
    )


def test_eda_functions_accept_a_named_dataset(df):
    assert rsm.eda.explore({"titanic": df}, cols=["age"]).equals(rsm.eda.explore(df, cols=["age"]))
    named = rsm.eda.pivot({"titanic": df}, rows=["sex"]).sort("sex")
    assert named.equals(rsm.eda.pivot(df, rows=["sex"]).sort("sex"))
    rsm.eda.visualize({"titanic": df}, x="age")


def test_named_dataset_must_be_one_item(df):
    with pytest.raises(ValueError):
        rsm.eda.explore({"a": df, "b": df})


def test_explore_accepts_categorical_columns_by_name(df):
    out = rsm.eda.explore(df, cols=["age", "sex"])
    names = out[out.columns[0]].to_list()
    assert names[0] == "age"
    assert len(names) == 2 and names[1].startswith("sex_")


@pytest.mark.parametrize("geom", ["dist", "density"])
def test_combined_variables_keep_their_fill_mapping(df, geom):
    p = rsm.eda.visualize(df, x=["age", "fare"], geom=geom, combine="x")
    layer = p.layers[0]
    assert "fill" not in layer.geom.aes_params, "a default fill overrode the mapping"
    assert p.mapping["fill"] == "variable"


def test_a_fill_variable_is_not_overwritten_by_the_default(df):
    p = rsm.eda.visualize(df, x="age", geom="density", fill="sex")
    assert "fill" not in p.layers[0].geom.aes_params
