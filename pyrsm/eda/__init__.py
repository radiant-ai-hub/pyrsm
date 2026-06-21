"""
rsm.eda - Exploratory Data Analysis functions for Polars DataFrames.
"""

from pyrsm.eda.associations import associations
from pyrsm.eda.combine import combine
from pyrsm.eda.distr import distr
from pyrsm.eda.explore import explore
from pyrsm.eda.missing import missing
from pyrsm.eda.outliers import outliers
from pyrsm.eda.pivot import pivot
from pyrsm.eda.profile import profile
from pyrsm.eda.unpivot import unpivot
from pyrsm.eda.visualize import visualize

__all__ = [
    "associations",
    "combine",
    "distr",
    "explore",
    "missing",
    "outliers",
    "pivot",
    "profile",
    "unpivot",
    "visualize",
]
