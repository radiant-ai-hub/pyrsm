from importlib import import_module

_SUBMODULES = {"factor"}


def __getattr__(name):
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        attr = getattr(module, name, module)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(_SUBMODULES))


__all__ = list(_SUBMODULES)
