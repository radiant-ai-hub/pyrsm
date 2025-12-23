from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

# Disable global string cache to prevent category contamination across tests
pl.disable_string_cache()

DATA_BASE = Path(__file__).resolve().parents[1] / "examples" / "data"


def _load_dataset(pkg: str, name: str, as_polars: bool = True):
    pq_path = DATA_BASE / pkg / f"{name}.parquet"
    md_path = DATA_BASE / pkg / f"{name}_description.md"
    # Always read with polars (handles Enum columns correctly)
    data = pl.read_parquet(pq_path)
    if not as_polars:
        data = data.to_pandas()
    description = md_path.read_text() if md_path.exists() else ""
    return data, description


@pytest.fixture(scope="session")
def load_basics_dataset():
    def _load(name: str):
        data, _ = _load_dataset("basics", name, as_polars=True)
        return data

    return _load


@pytest.fixture(scope="session")
def basics_plot_dir():
    out_dir = Path("tests/plot_comparisons/basics/compare_means")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture(scope="session")
def baseline_plot_dir():
    """Directory for baseline plot snapshots before conversion."""
    out_dir = Path("tests/plot_comparisons/basics/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture(scope="session")
def salary_data(load_basics_dataset):
    """Salary dataset for compare_means and correlation tests."""
    return load_basics_dataset("salary")


@pytest.fixture(scope="session")
def newspaper_data_session(load_basics_dataset):
    """Newspaper dataset for goodness/cross_tabs tests (session-scoped)."""
    return load_basics_dataset("newspaper")


@pytest.fixture(scope="session")
def diamonds_data(load_basics_dataset):
    """Diamonds dataset for correlation tests."""
    return load_basics_dataset("diamonds")


@pytest.fixture(scope="session")
def synthetic_group_frame():
    pdf = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b", "b", "c", "c", "c", "c"],
            "value": [1.0, np.nan, 0.5, 0.5, 0.5, 3.0, 3.0, 3.0, np.nan],
        }
    )
    return pdf, pl.from_pandas(pdf)


@pytest.fixture(scope="session")
def numeric_var1_frame():
    pdf = pd.DataFrame(
        {
            "measurement": [1, 2, 3, 4, 5, 6],
            "score_a": [10, 12, 11, 9, 10, 12],
            "score_b": [8, 7, 6, 8, 9, 7],
        }
    )
    return pdf, pl.from_pandas(pdf)
