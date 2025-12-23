import sys
from importlib import import_module
from types import ModuleType

_EXPORTS = {
    "central_limit_theorem": "pyrsm.basics.central_limit_theorem",
    "compare_means": "pyrsm.basics.compare_means",
    "compare_props": "pyrsm.basics.compare_props",
    "correlation": "pyrsm.basics.correlation",
    "cross_tabs": "pyrsm.basics.cross_tabs",
    "goodness": "pyrsm.basics.goodness",
    "prob_calc": "pyrsm.basics.probability_calculator",
    "probability_calculator": "pyrsm.basics.probability_calculator",
    "single_mean": "pyrsm.basics.single_mean",
    "single_prop": "pyrsm.basics.single_prop",
}


def __getattr__(name):
    if name in _EXPORTS:
        module = import_module(_EXPORTS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __getattribute__(name):
    module = sys.modules[__name__]
    exports = ModuleType.__getattribute__(module, "_EXPORTS")
    if name in exports:
        existing = globals().get(name)
        if isinstance(existing, ModuleType):
            module = import_module(exports[name])
            attr = getattr(module, name)
            globals()[name] = attr
            return attr
    return ModuleType.__getattribute__(module, name)


def __dir__():
    return sorted(list(_EXPORTS))


__all__ = list(_EXPORTS)

# Ensure classes resolve correctly even if the submodule is imported directly
try:
    from pyrsm.basics.compare_means import compare_means as _compare_means
    from pyrsm.basics.compare_props import compare_props as _compare_props
    from pyrsm.basics.correlation import correlation as _correlation
    from pyrsm.basics.cross_tabs import cross_tabs as _cross_tabs
    from pyrsm.basics.goodness import goodness as _goodness
    from pyrsm.basics.single_mean import single_mean as _single_mean
    from pyrsm.basics.single_prop import single_prop as _single_prop

    globals()["compare_means"] = _compare_means
    globals()["compare_props"] = _compare_props
    globals()["correlation"] = _correlation
    globals()["cross_tabs"] = _cross_tabs
    globals()["goodness"] = _goodness
    globals()["single_mean"] = _single_mean
    globals()["single_prop"] = _single_prop
except Exception:
    pass
