from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

# Disable global string cache to prevent category contamination across tests
pl.disable_string_cache()

DATA_BASE = Path(__file__).resolve().parents[1] / "examples" / "data"
MV_REF_BASE = Path(__file__).resolve().parent / "reference" / "radiant_multivariate"


# ---------------------------------------------------------------------------
# multivariate helpers (reference fixtures generated from radiant.multivariate)
# ---------------------------------------------------------------------------


def mv_load_dataset(name: str) -> pl.DataFrame:
    """Load a multivariate example dataset as Polars (Enum order preserved)."""
    p = DATA_BASE / "multivariate" / f"{name}.parquet"
    if not p.exists():
        p = DATA_BASE / "model" / f"{name}.parquet"  # diamonds, etc.
    return pl.read_parquet(p)


def mv_load_ref(name: str) -> dict:
    """Load a reference JSON fixture produced by the R generator."""
    import json

    return json.loads((MV_REF_BASE / f"{name}.json").read_text())


def mv_mat(d: dict) -> np.ndarray:
    """Reshape a serialized (flat, row-major) matrix fixture into an ndarray."""
    return np.array(d["values"], dtype=float).reshape(d["nrow"], d["ncol"])


def mv_sign_align(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Flip columns of A so each aligns in sign with the matching column of B."""
    A = np.array(A, dtype=float, copy=True)
    for j in range(A.shape[1]):
        if float(A[:, j] @ B[:, j]) < 0:
            A[:, j] *= -1
    return A


def mv_to_ordered_enum(df: pl.DataFrame, cols) -> pl.DataFrame:
    """Cast integer columns to ordered Enum (numeric ascending) to mark ordinal."""
    for c in cols:
        levels = [str(x) for x in sorted(df[c].unique().to_list())]
        df = df.with_columns(pl.col(c).cast(pl.Utf8).cast(pl.Enum(levels)))
    return df


@pytest.fixture(scope="session")
def mv_data():
    return mv_load_dataset


@pytest.fixture(scope="session")
def mv_ref():
    return mv_load_ref


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
