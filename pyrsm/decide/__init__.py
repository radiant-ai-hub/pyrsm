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
    "UnknownSubtreeError": "dtree",
    "UnresolvedVariableError": "dtree",
    "UnsafeExpressionError": "dtree",
    "safe_eval": "dtree",
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
