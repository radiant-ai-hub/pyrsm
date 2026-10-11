"""``rsm.data.add_description``: a description stored inside and beside a parquet file."""

from pathlib import Path

import polars as pl
import pytest

import pyrsm as rsm

MD = """# Sales

Monthly sales by region.

* price: price in USD"""


@pytest.fixture
def sales(tmp_path: Path) -> Path:
    path = tmp_path / "sales.parquet"
    pl.DataFrame({"price": [1.5, 2.0, None], "region": ["N", "S", "E"]}).write_parquet(path)
    return path


def test_text_is_stored_inside_the_file_and_beside_it(sales):
    frame = pl.read_parquet(sales)
    assert rsm.data.add_description(sales, MD) == sales
    assert pl.read_parquet_metadata(sales)["description"] == MD
    assert (sales.parent / "sales_description.md").read_text(encoding="utf-8") == MD
    assert rsm.data.read_description(sales) == MD
    assert pl.read_parquet(sales).equals(frame)


def test_a_markdown_file_path_is_read(sales, tmp_path):
    notes = tmp_path / "notes.md"
    notes.write_text(MD, encoding="utf-8")
    rsm.data.add_description(str(sales), str(notes))
    assert rsm.data.read_description(sales) == MD
    rsm.data.add_description(sales, notes)
    assert rsm.data.read_description(sales) == MD


def test_a_new_description_replaces_the_old_and_other_metadata_is_kept(tmp_path):
    path = tmp_path / "kept.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(path, metadata={"source": "survey"})
    rsm.data.add_description(path, "first")
    rsm.data.add_description(path, "second")
    metadata = pl.read_parquet_metadata(path)
    assert metadata["description"] == "second"
    assert metadata["source"] == "survey"
    assert not list(tmp_path.glob(".*partial"))


def test_the_sidecar_is_read_when_the_file_has_no_description(sales):
    (sales.parent / "sales_description.md").write_text(MD, encoding="utf-8")
    assert rsm.data.read_description(sales) == MD


def test_no_description_reads_as_none(sales):
    assert rsm.data.read_description(sales) is None


@pytest.mark.parametrize(
    "path, descr, error",
    [
        ("missing.parquet", MD, FileNotFoundError),
        ("sales.csv", MD, ValueError),
        (None, MD, TypeError),
        ("sales.parquet", "notes/missing.md", FileNotFoundError),
        ("sales.parquet", "   ", ValueError),
        ("sales.parquet", 3, TypeError),
    ],
)
def test_bad_arguments_are_refused_and_the_file_is_untouched(sales, tmp_path, path, descr, error):
    if isinstance(path, str):
        path = tmp_path / path
    before = sales.read_bytes()
    with pytest.raises(error):
        rsm.data.add_description(path, descr)
    assert sales.read_bytes() == before
    assert not (sales.parent / "sales_description.md").exists()
