"""Compatibility wrapper for :mod:`pyrsm.decide.dtree`.

Decision analysis now lives in ``pyrsm.decide``. This module keeps the old
``pyrsm.model.dtree`` import path working for existing notebooks and tests.
"""

from pyrsm.decide.dtree import (
    DecisionNode,
    Diagnostic,
    SensitivityCell,
    SensitivityResult,
    SensitivitySpec,
    UnknownSubtreeError,
    UnresolvedVariableError,
    UnsafeExpressionError,
    dtree,
    safe_eval,
)

__all__ = [
    "DecisionNode",
    "Diagnostic",
    "SensitivityCell",
    "SensitivityResult",
    "SensitivitySpec",
    "UnknownSubtreeError",
    "UnresolvedVariableError",
    "UnsafeExpressionError",
    "dtree",
    "safe_eval",
]

_dtree = dtree


def _make_module_callable() -> None:
    """Keep the legacy ``pyrsm.model.dtree(...)`` shorthand callable."""
    import sys
    import types

    class _CallableModule(types.ModuleType):
        def __call__(self, *args, **kwargs):
            return _dtree(*args, **kwargs)

    sys.modules[__name__].__class__ = _CallableModule


_make_module_callable()
