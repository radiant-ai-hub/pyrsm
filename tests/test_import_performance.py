import gc
import importlib
import sys
import time

FULL_IMPORT_BUDGET = 0.5  # seconds; tightened after lazy-loading refactor
SUBMODULE_IMPORT_BUDGET = 1.0


def _clear_pyrsm():
    """Remove pyrsm modules from sys.modules to force a cold import."""
    for mod in list(sys.modules):
        if mod == "pyrsm" or mod.startswith("pyrsm."):
            sys.modules.pop(mod, None)
    gc.collect()
    importlib.invalidate_caches()


def _cold_import(stmt: str, repeat: int = 3) -> float:
    """Execute an import statement and return the best-of-n timing."""
    timings: list[float] = []
    for _ in range(repeat):
        _clear_pyrsm()
        start = time.perf_counter()
        exec(stmt, {})
        timings.append(time.perf_counter() - start)
    return min(timings)


def test_full_import_is_fast_enough():
    duration = _cold_import("import pyrsm")
    assert duration < FULL_IMPORT_BUDGET, f"import pyrsm took {duration:.2f}s"


def test_basics_import_is_fast_enough_and_resolves_symbol():
    duration = _cold_import("from pyrsm.basics import compare_means")
    assert (
        duration < SUBMODULE_IMPORT_BUDGET
    ), f"from pyrsm.basics import compare_means took {duration:.2f}s"

    _clear_pyrsm()
    import pyrsm  # noqa: WPS433

    assert hasattr(pyrsm, "basics")
    assert hasattr(pyrsm.basics, "compare_means")


def test_model_import_is_fast_enough_and_resolves_symbol():
    duration = _cold_import("from pyrsm.model import regress")
    assert (
        duration < SUBMODULE_IMPORT_BUDGET
    ), f"from pyrsm.model import regress took {duration:.2f}s"

    _clear_pyrsm()
    import pyrsm  # noqa: WPS433

    assert hasattr(pyrsm, "model")
    assert hasattr(pyrsm.model, "regress")
