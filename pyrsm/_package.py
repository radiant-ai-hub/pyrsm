"""Keep ``pyrsm.<package>.<name>`` pointing at the class, not the submodule.

Most pyrsm tools live in a submodule with the same name as their class
(``pyrsm/model/regress.py`` defines ``regress``), and the packages expose the
class lazily: ``pyrsm.model.regress`` is the class. But whenever the submodule
is imported directly -- ``from pyrsm.model.regress import regress``, often in
library code or tests -- Python binds the *module* on the package under that
name, so afterwards ``pyrsm.model.regress(...)`` fails with "'module' object is
not callable". ``keep_classes`` makes the package swap the module for the
class at the moment Python binds it.
"""

from __future__ import annotations

import sys
import types


def keep_classes(package: str, names) -> None:
    """Make ``sys.modules[package]`` bind classes instead of same-named submodules.

    Parameters
    ----------
    package : str
        The package's ``__name__``.
    names : iterable of str
        Submodule names whose same-named attribute (a class or function) should
        be what ``package.<name>`` refers to.
    """
    names = frozenset(names)
    module = sys.modules[package]

    class _Package(type(module)):
        def __setattr__(self, attr, value):
            if attr in names and isinstance(value, types.ModuleType):
                target = getattr(value, attr, None)
                if callable(target) and not isinstance(target, types.ModuleType):
                    value = target
            super().__setattr__(attr, value)

    module.__class__ = _Package
