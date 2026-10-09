"""Decision analysis and simulation tools.

This namespace holds the tools that mirror Radiant's Model > Decide menu:
decision-tree analysis, Monte Carlo simulation, and linear/integer
programming optimization.
"""

from importlib import import_module

_LAZY_SUBMODULES = {
    "dtree",
    "simulate",
    "optimize",
}

_LAZY_FUNCTIONS = {
    "DecisionNode": "dtree",
    "Diagnostic": "dtree",
    "SensitivityCell": "dtree",
    "SensitivityResult": "dtree",
    "SensitivitySpec": "dtree",
    "TornadoResult": "dtree",
    "UnknownSubtreeError": "dtree",
    "UnresolvedVariableError": "dtree",
    "UnsafeExpressionError": "dtree",
    "safe_eval": "dtree",
    "dtree_parser": "dtree",
    "dtree_examples": "dtree_examples",
    "dtree_example": "dtree_examples",
    "Constraint": "optimize",
    "DecisionVariable": "optimize",
    "RobustnessResult": "optimize",
    "Variable": "simulate",
    "Formula": "simulate",
    "SimulationSpec": "simulate",
    "SimulationResult": "simulate",
    "RepeatResult": "simulate",
    "FormulaError": "simulate",
    "compile_formula": "simulate",
    "repeat_simulate": "simulate",
}


def _load_submodule(name: str):
    module = import_module(f"{__name__}.{name}")
    if hasattr(module, name):
        globals()[name] = getattr(module, name)
    else:
        globals()[name] = module
    return module


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        module = _load_submodule(name)
        attr = getattr(module, name, module)
        globals()[name] = attr
        return attr

    if name in _LAZY_FUNCTIONS:
        module = _load_submodule(_LAZY_FUNCTIONS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(_LAZY_SUBMODULES) + list(_LAZY_FUNCTIONS))


__all__ = list(_LAZY_SUBMODULES) + list(_LAZY_FUNCTIONS)


# ``from pyrsm.<pkg>.<name> import <name>`` must not hide the class behind its
# submodule (see pyrsm/_package.py)
from pyrsm._package import keep_classes  # noqa: E402

keep_classes(__name__, _LAZY_SUBMODULES)
