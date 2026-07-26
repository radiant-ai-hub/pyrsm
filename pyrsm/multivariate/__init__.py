"""Multivariate analysis tools ported from R's ``radiant.multivariate``.

Public API (callable as ``pyrsm.multivariate.<name>``)::

    pre_factor, full_factor   # factor / PCA analysis
    hclus, kclus              # hierarchical & k-means clustering
    mds, prmap                # (dis)similarity & attribute perceptual maps
    conjoint                  # conjoint analysis

Each returns a result object with ``summary()`` and ``plot()`` methods, plus
``store()``/``predict()`` where the R package offers them.
"""

from importlib import import_module

_EXPORTS = {
    "pre_factor": "pyrsm.multivariate.pre_factor",
    "kmo": "pyrsm.multivariate.pre_factor",
    "full_factor": "pyrsm.multivariate.full_factor",
    "clean_loadings": "pyrsm.multivariate.full_factor",
    "hclus": "pyrsm.multivariate.hclus",
    "kclus": "pyrsm.multivariate.kclus",
    "mds": "pyrsm.multivariate.mds",
    "prmap": "pyrsm.multivariate.prmap",
    "conjoint": "pyrsm.multivariate.conjoint",
    "store_predictions": "pyrsm.multivariate.conjoint",
    "cbc": "pyrsm.multivariate.cbc",
}


def __getattr__(name):
    if name in _EXPORTS:
        mod_name = _EXPORTS[name]
        module = import_module(mod_name)
        # A class and its module share a name (e.g. ``full_factor``). Importing
        # the module auto-binds the *module* onto this package, shadowing the
        # *class*. Re-bind every public export from this module (the class wins)
        # so ``from pyrsm.multivariate import full_factor`` returns the class
        # regardless of access order.
        for export, src in _EXPORTS.items():
            if src == mod_name:
                globals()[export] = getattr(module, export)
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(_EXPORTS)


__all__ = sorted(_EXPORTS)
