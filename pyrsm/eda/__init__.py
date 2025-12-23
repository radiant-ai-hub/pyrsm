"""
rsm.eda - Exploratory Data Analysis functions for Polars DataFrames.

Provides clean Python API for common EDA operations:
- explore(): Summary statistics for numeric columns
- pivot(): Pivot tables and crosstabs
- combine(): Combine datasets using joins, binds, or set operations
- visualize(): Create plots using plotnine
- distr(): Distribution analysis class
- distr_plot(): Standalone distribution plotting function
"""

from importlib import import_module

_SUBMODULES = {
    "distr",
    "explore",
    "pivot",
    "unpivot",
    "combine",
    "visualize",
}

# Functions to import from specific modules
_LAZY_FUNCTIONS = {
    "distr_plot": "distr",
}


def __getattr__(name):
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        attr = getattr(module, name, module)
        globals()[name] = attr
        return attr
    if name in _LAZY_FUNCTIONS:
        module_name = _LAZY_FUNCTIONS[name]
        module = import_module(f"{__name__}.{module_name}")
        attr = getattr(module, name)
        globals()[name] = attr
        # Also cache the main class/function from that module to avoid module shadowing
        if module_name in _SUBMODULES:
            main_attr = getattr(module, module_name, module)
            globals()[module_name] = main_attr
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(_SUBMODULES) + list(_LAZY_FUNCTIONS))


__all__ = list(_SUBMODULES) + list(_LAZY_FUNCTIONS)
