"""Compatibility wrapper for :mod:`pyrsm.decide.simulate`.

Simulation now lives in ``pyrsm.decide``. This module keeps the old
``pyrsm.model.simulate`` import path working for existing notebooks and tests.
"""

from pyrsm.decide.simulate import (
    Diagnostic,
    Formula,
    FormulaError,
    RepeatResult,
    SimulationResult,
    SimulationSpec,
    Variable,
    compile_formula,
    repeat_simulate,
    simulate,
)

__all__ = [
    "Variable",
    "Formula",
    "SimulationSpec",
    "SimulationResult",
    "RepeatResult",
    "Diagnostic",
    "FormulaError",
    "compile_formula",
    "simulate",
    "repeat_simulate",
]

_simulate = simulate


def _make_module_callable() -> None:
    """Keep the legacy ``pyrsm.model.simulate(...)`` shorthand callable."""
    import sys
    import types

    class _CallableModule(types.ModuleType):
        def __call__(self, *args, **kwargs):
            return _simulate(*args, **kwargs)

    sys.modules[__name__].__class__ = _CallableModule


_make_module_callable()
