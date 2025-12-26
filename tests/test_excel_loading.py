"""Tests for loading Excel files (.xls and .xlsx) with Polars."""

from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).parent.parent / "examples" / "data" / "data"


class TestExcelLoading:
    """Test loading Excel files with Polars."""

    def test_load_xlsx(self):
        """Test loading .xlsx file with Polars."""
        xlsx_path = DATA_DIR / "test_data.xlsx"
        df = pl.read_excel(xlsx_path)

        assert df.shape == (3, 3)
        assert df.columns == ["id", "name", "value"]
        assert df["id"].to_list() == [1, 2, 3]
        assert df["name"].to_list() == ["Alice", "Bob", "Charlie"]

    def test_load_xls(self):
        """Test loading .xls file with Polars."""
        xls_path = DATA_DIR / "test_data.xls"
        df = pl.read_excel(xls_path)

        assert df.shape == (3, 3)
        assert df.columns == ["id", "name", "value"]
        assert df["id"].to_list() == [1, 2, 3]
        assert df["name"].to_list() == ["Alice", "Bob", "Charlie"]
