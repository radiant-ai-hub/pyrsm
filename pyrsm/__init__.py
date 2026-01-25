from importlib import import_module

__version__ = "2.2.0"

_LAZY_MODULES = {
    "basics": "pyrsm.basics",
    "model": "pyrsm.model",
    "multivariate": "pyrsm.multivariate",
    "eda": "pyrsm.eda",
    "bins": "pyrsm.bins",
    "notebook": "pyrsm.notebook",
    "props": "pyrsm.props",
    "stats": "pyrsm.stats",
    "utils": "pyrsm.utils",
}


# Create wrapper functions with deprecation errors
def _make_wrapper(func_name, module):
    def wrapper(*args, **kwargs):
        raise DeprecationWarning(
            f"{func_name}() is deprecated. Use {module}.{func_name}() instead."
        )

    return wrapper


# model functions that need wrappers (enforce use of .model)
model_functions = [
    "make_train",
    "cross_validation",
    "regress",
    "rforest",
    "logistic",
    "mlp",
    "xgboost",
    "gains_plot",
]

# basics classes that need wrappers (enforce use of .basics)
basics_functions = [
    "central_limit_theorem",
    "compare_means",
    "compare_props",
    "correlation",
    "cross_tabs",
    "goodness",
    "prob_calc",
    "single_mean",
    "single_prop",
]

# Create wrappers for model functions
for func_name in model_functions:
    globals()[func_name] = _make_wrapper(func_name, "model")

# Create wrappers for basics classes
for func_name in basics_functions:
    globals()[func_name] = _make_wrapper(func_name, "basics")

_WRAPPER_MAP = {
    **{f: globals()[f] for f in model_functions},
    **{f: globals()[f] for f in basics_functions},
}


_LAZY_FUNCTIONS = {
    "md": "pyrsm.utils",
}


def __getattr__(name):
    if name in _LAZY_MODULES:
        module = import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    if name in _LAZY_FUNCTIONS:
        module = import_module(_LAZY_FUNCTIONS[name])
        func = getattr(module, name)
        globals()[name] = func
        return func
    if name in _WRAPPER_MAP:
        return _WRAPPER_MAP[name]
    raise AttributeError(f"module 'pyrsm' has no attribute '{name}'")


def __dir__():
    return sorted(list(_LAZY_MODULES) + list(_WRAPPER_MAP) + list(_LAZY_FUNCTIONS))


__all__ = list(_LAZY_MODULES) + model_functions + basics_functions + list(_LAZY_FUNCTIONS)
