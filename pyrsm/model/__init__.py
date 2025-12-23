from importlib import import_module

_LAZY_SUBMODULES = {
    "logistic",
    "rforest",
    "mlp",
    "xgboost",
    "perf",
    "regress",
    "visualize",
}
_CORE_MODULE = "pyrsm.model.model"

# Functions to import from specific submodules
_LAZY_FUNCTIONS = {
    "gains_tab": "perf",
    "gains_plot": "perf",
    "lift_plot": "perf",
    "ROME_tab": "perf",
    "ROME_plot": "perf",
    "profit_tab": "perf",
    "profit_plot": "perf",
    "expected_plot": "perf",
    "uplift_tab": "perf",
    "uplift_plot": "perf",
    "uplift_profit_tab": "perf",
    "uplift_profit_plot": "perf",
    "inc_uplift_plot": "perf",
    "confusion": "perf",
    "evalbin": "perf",
    "auc": "perf",
    "profit_max": "perf",
}


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        attr = getattr(module, name, module)
        globals()[name] = attr
        return attr

    if name in _LAZY_FUNCTIONS:
        module = import_module(f"{__name__}.{_LAZY_FUNCTIONS[name]}")
        attr = getattr(module, name)
        globals()[name] = attr
        return attr

    core = import_module(_CORE_MODULE)
    if hasattr(core, name):
        attr = getattr(core, name)
        globals()[name] = attr
        return attr

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    core = import_module(_CORE_MODULE)
    core_exports = [n for n in dir(core) if not n.startswith("_")]
    return sorted(list(_LAZY_SUBMODULES) + list(_LAZY_FUNCTIONS) + core_exports)


__all__ = list(_LAZY_SUBMODULES) + list(_LAZY_FUNCTIONS)
