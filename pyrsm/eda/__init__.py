"""
rsm.eda - Exploratory Data Analysis functions for Polars DataFrames.
"""

from pyrsm.eda.combine import combine
from pyrsm.eda.distr import distr
from pyrsm.eda.explore import explore
from pyrsm.eda.pivot import pivot
from pyrsm.eda.unpivot import unpivot
from pyrsm.eda.visualize import visualize

__all__ = ["combine", "distr", "explore", "pivot", "unpivot", "visualize"]
