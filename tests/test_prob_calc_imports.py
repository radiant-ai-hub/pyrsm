import importlib

from pyrsm.basics import probability_calculator


def test_prob_calc_submodules_importable():
    modules = [
        "binomial",
        "chisq",
        "discrete",
        "exponential",
        "fdist",
        "lnorm",
        "normal",
        "poisson",
        "tdist",
        "uniform",
    ]
    for mod in modules:
        importlib.import_module(f"pyrsm.basics.prob_calc.{mod}")


def test_prob_calc_wrapper_summary():
    pc = probability_calculator.prob_calc("norm", mean=0, sd=1, lb=-1, ub=1)
    summary = pc.summary(ret=True)

    assert summary["Distribution"] == "Normal"
    # Ensure at least one probability line is produced
    prob_keys = [k for k in summary if k.startswith("P(")]
    assert prob_keys
