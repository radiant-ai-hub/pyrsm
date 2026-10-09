"""Every pyrsm plot is plotnine and uses the one pyrsm style.

The rule (``pyrsm.plot_style``): all plots are plotnine ``ggplot`` objects (or a
composition / list of them), every ggplot ends in ``theme_pyrsm()`` and takes
its single-series and reference colours from ``pyrsm.plot_style``.

Static checks scan the source so a new plot cannot pick its own theme, colour
literal or plotting library. Runtime checks call every plotting class and
function with each plot type it offers and inspect the returned objects.
"""

from __future__ import annotations

import ast
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from matplotlib.colors import is_color_like
from plotnine import ggplot
from plotnine.composition import Compose, plot_spacer

from pyrsm.plot_style import theme_pyrsm

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "pyrsm"
DATA = ROOT / "examples" / "data"
STYLE_FILE = PKG / "plot_style.py"

# ---------------------------------------------------------------------------
# static checks
# ---------------------------------------------------------------------------

OTHER_THEMES = {
    "theme_bw",
    "theme_classic",
    "theme_minimal",
    "theme_gray",
    "theme_grey",
    "theme_light",
    "theme_dark",
    "theme_void",
    "theme_538",
    "theme_seaborn",
    "theme_xkcd",
    "theme_matplotlib",
    "theme_tufte",
}
OTHER_PLOT_LIBS = ("matplotlib", "seaborn", "plotly", "altair", "bokeh")
# Colour literals a plot may still use directly: an outline that separates
# adjacent bars from each other is not part of the style.
ALLOWED_COLOUR_LITERALS = {"white"}
COLOUR_KEYWORDS = {"color", "colour", "fill"}


def _py_files():
    return sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


def _parse(path):
    return ast.parse(path.read_text(), filename=str(path))


def _rel(path):
    return str(path.relative_to(ROOT))


def test_only_plot_style_picks_a_theme():
    offenders = []
    for path in _py_files():
        if path == STYLE_FILE:
            continue
        for node in ast.walk(_parse(path)):
            names = []
            if isinstance(node, ast.Name):
                names = [node.id]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            elif isinstance(node, ast.ImportFrom):
                names = [a.name for a in node.names]
            for name in names:
                if name in OTHER_THEMES:
                    offenders.append(f"{_rel(path)}:{node.lineno} {name}")
    assert not offenders, (
        "use pyrsm.plot_style.theme_pyrsm() instead of:\n" + "\n".join(offenders)
    )


# Modules that use matplotlib only to measure and close a plotnine figure (no
# plotting): label placement draws the plot once to measure text and panel.
MEASURE_ONLY = {"pyrsm/multivariate/_label_layout.py"}


def test_no_other_plotting_library():
    offenders = []
    for path in _py_files():
        if _rel(path) in MEASURE_ONLY:
            continue
        for node in ast.walk(_parse(path)):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "plt":
                    mods = ["plt"]
            for m in mods:
                if m == "plt" or m.split(".")[0] in OTHER_PLOT_LIBS:
                    offenders.append(f"{_rel(path)}:{node.lineno} {m}")
    assert not offenders, "pyrsm plots are plotnine only:\n" + "\n".join(offenders)


def test_colour_literals_come_from_plot_style():
    """A fixed colour/fill passed to a geom or helper is a plot_style name."""
    offenders = []
    for path in _py_files():
        if path == STYLE_FILE:
            continue
        for node in ast.walk(_parse(path)):
            pairs = []
            if isinstance(node, ast.Call):
                pairs = [(k.arg, k.value) for k in node.keywords]
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args.args + node.args.kwonlyargs
                defaults = [None] * (len(node.args.args) - len(node.args.defaults))
                defaults += list(node.args.defaults) + list(node.args.kw_defaults)
                pairs = [(a.arg, d) for a, d in zip(args, defaults) if d is not None]
            elif isinstance(node, ast.Dict):
                pairs = [
                    (k.value, v)
                    for k, v in zip(node.keys, node.values)
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                ]
            for key, value in pairs:
                if (
                    key in COLOUR_KEYWORDS
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and value.value not in ALLOWED_COLOUR_LITERALS
                    and is_color_like(value.value)
                ):
                    offenders.append(f"{_rel(path)}:{value.lineno} {key}={value.value!r}")
    assert not offenders, (
        "use a pyrsm.plot_style constant (FILL, LINE, REF, COMP, MUTED, ...):\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# runtime checks: every plot class/function and plot type
# ---------------------------------------------------------------------------


def iter_ggplots(obj):
    """Yield every ggplot in a plot result; fail on anything that is not plotnine."""
    if isinstance(obj, plot_spacer):
        return
    if isinstance(obj, ggplot):
        yield obj
    elif isinstance(obj, Compose):
        for item in obj:
            yield from iter_ggplots(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from iter_ggplots(item)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from iter_ggplots(item)
    else:
        raise AssertionError(f"not a plotnine plot: {type(obj)!r}")


def _load(group, name):
    return pl.read_parquet(DATA / group / f"{name}.parquet")


@lru_cache(maxsize=None)
def _data(group, name):
    return _load(group, name)


def _titanic():
    return _data("model", "titanic").drop_nulls(subset=["age"])


def _salary():
    return _data("basics", "salary")


@lru_cache(maxsize=None)
def _model(kind, mod_type="classification"):
    # import each class from its own module: once a test imports e.g.
    # ``pyrsm.model.rforest`` the package attribute can be the submodule
    from pyrsm.model.crtree import crtree
    from pyrsm.model.logistic import logistic
    from pyrsm.model.mlp import mlp
    from pyrsm.model.regress import regress
    from pyrsm.model.rforest import rforest
    from pyrsm.model.xgboost import xgboost

    if kind == "regress":
        return regress(
            {"salary": _salary()}, rvar="salary", evar=["yrs_since_phd", "yrs_service", "rank"]
        )
    if kind == "logistic":
        return logistic(
            {"titanic": _titanic()}, rvar="survived", lev="Yes", evar=["age", "sex", "pclass"]
        )
    if mod_type == "classification":
        args = dict(
            data={"titanic": _titanic()},
            rvar="survived",
            lev="Yes",
            evar=["age", "sex"],
            mod_type="classification",
        )
    else:
        args = dict(
            data={"salary": _salary()},
            rvar="salary",
            evar=["yrs_since_phd", "rank"],
            mod_type="regression",
        )
    if kind == "crtree":
        return crtree(**args, cp=0.01)
    if kind == "rforest":
        return rforest(**args, n_estimators=10)
    if kind == "xgboost":
        return xgboost(**args, n_estimators=10)
    if kind == "mlp":
        return mlp(**args, hidden_layer_sizes=(3,), max_iter=200)
    raise ValueError(kind)


def _perf_df():
    rng = np.random.default_rng(1234)
    n = 200
    x1 = rng.uniform(0, 1, n)
    return pl.DataFrame(
        {
            "x1": x1,
            "x2": 1 - x1 + rng.normal(0, 0.1, n),
            "response": np.where(x1 + rng.normal(0, 0.3, n) > 0.6, "yes", "no"),
            "treatment": np.repeat(["treatment", "control"], n // 2),
        }
    )


def _production_mix():
    return {
        "meta": {"problem_name": "Production Mix", "objective_sense": "maximize"},
        "variables": {
            "a": {"label": "A", "type": "continuous", "min": 0},
            "b": {"label": "B", "type": "continuous", "min": 0},
        },
        "objective": {"a": 40.0, "b": 30.0},
        "constraints": [
            {"name": "labor", "coefficients": {"a": 2.0, "b": 1.0}, "sense": "L", "rhs": 400.0},
            {"name": "material", "coefficients": {"a": 1.0, "b": 2.0}, "sense": "L", "rhs": 500.0},
        ],
    }


CASES: dict[str, callable] = {}


def case(name):
    def register(fn):
        CASES[name] = fn
        return fn

    return register


# -- basics -------------------------------------------------------------------

for _p in ("scatter", "box", "density", "bar"):

    @case(f"basics.compare_means-{_p}")
    def _cm(p=_p):
        from pyrsm.basics.compare_means import compare_means

        return compare_means({"salary": _salary()}, var1="rank", var2="salary").plot(plots=p)


for _p in ("bar", "dodge"):

    @case(f"basics.compare_props-{_p}")
    def _cp(p=_p):
        from pyrsm.basics.compare_props import compare_props

        return compare_props({"salary": _salary()}, var1="rank", var2="sex", lev="Male").plot(
            plots=p
        )


@case("basics.correlation")
def _corr():
    from pyrsm.basics.correlation import correlation

    return correlation(
        {"salary": _salary()}, vars=["salary", "yrs_since_phd", "yrs_service"]
    ).plot()


@case("basics.correlation-categorical")
def _corr_cat():
    from pyrsm.basics.correlation import correlation

    d = _data("data", "diamonds").select(["price", "carat", "clarity", "cut"]).head(300)
    return correlation({"d": d}, vars=["price", "carat", "clarity", "cut"], method="polychoric").plot()


for _p in ("observed", "expected", "chisq", "dev_std", "perc_col", "perc_row", "perc"):

    @case(f"basics.cross_tabs-{_p}")
    def _ct(p=_p):
        from pyrsm.basics.cross_tabs import cross_tabs

        return cross_tabs(
            {"newspaper": _data("basics", "newspaper")}, "Income", "Newspaper"
        ).plot(plots=p)


for _p in ("observed", "expected", "chisq", "dev_std"):

    @case(f"basics.goodness-{_p}")
    def _gd(p=_p):
        from pyrsm.basics.goodness import goodness

        d = pl.DataFrame({"category": ["A"] * 10 + ["B"] * 20 + ["C"] * 30})
        return goodness(d, var="category", probs=[0.2, 0.3, 0.5]).plot(plots=p)


for _p in ("hist", "sim"):

    @case(f"basics.single_mean-{_p}")
    def _sm(p=_p):
        from pyrsm.basics.single_mean import single_mean

        return single_mean({"salary": _salary()}, var="salary", comp_value=100000).plot(plots=p)


@case("basics.single_prop-bar")
def _sp():
    from pyrsm.basics.single_prop import single_prop

    return single_prop({"salary": _salary()}, var="sex", lev="Male").plot(plots="bar")


@case("basics.single_prop-sim")
def _sp_sim():
    from pyrsm.basics.single_prop import single_prop

    return single_prop({"salary": _salary()}, var="sex", lev="Male").plot(plots="sim")


@case("basics.clt")
def _clt():
    from pyrsm.basics import clt

    return clt("Normal", n=50, m=40, seed=1).plot(bins=10)


_PROB = {
    "binom": dict(n=10, p=0.3),
    "chisq": dict(df=4),
    "disc": dict(v=[1, 2, 3, 4], p=[0.1, 0.2, 0.3, 0.4]),
    "expo": dict(rate=1),
    "fdist": dict(df1=5, df2=10),
    "lnorm": dict(meanlog=0, sdlog=1),
    "norm": dict(mean=0, sd=1),
    "pois": dict(lamb=3),
    "tdist": dict(df=10),
    "unif": dict(min=0, max=1),
}
_PROB_BOUNDS = {
    "binom": (dict(lb=2, ub=5), dict(plb=0.1, pub=0.9)),
    "chisq": (dict(lb=1, ub=6), dict(plb=0.1, pub=0.9)),
    "disc": (dict(lb=2, ub=3), dict(plb=0.1, pub=0.9)),
    "expo": (dict(lb=0.5, ub=2), dict(plb=0.1, pub=0.9)),
    "fdist": (dict(lb=0.5, ub=2), dict(plb=0.1, pub=0.9)),
    "lnorm": (dict(lb=0.5, ub=2), dict(plb=0.1, pub=0.9)),
    "norm": (dict(lb=-1, ub=1), dict(plb=0.1, pub=0.9)),
    "pois": (dict(lb=2, ub=5), dict(plb=0.1, pub=0.9)),
    "tdist": (dict(lb=-1, ub=1), dict(plb=0.1, pub=0.9)),
    "unif": (dict(lb=0.2, ub=0.8), dict(plb=0.1, pub=0.9)),
}
for _d in _PROB:
    for _kind, _bounds in zip(("values", "probs"), _PROB_BOUNDS[_d]):

        @case(f"basics.prob_calc-{_d}-{_kind}")
        def _pc(d=_d, bounds=_bounds):
            from pyrsm.basics.probability_calculator import prob_calc

            return prob_calc(d, **_PROB[d], **bounds).plot()


@case("props.prop_plot")
def _prop_plot():
    from pyrsm.props import prop_plot

    return prop_plot(_titanic(), "pclass", "survived", "Yes", breakeven=0.4)


# -- model --------------------------------------------------------------------

for _p in ("dist", "corr", "scatter", "dashboard", "residual", "pred", "pdp", "pip", "coef"):

    @case(f"model.regress-{_p}")
    def _reg(p=_p):
        return _model("regress").plot(plots=p)


for _p in ("pred", "pdp", "pip", "or", "dashboard", "dist", "corr"):

    @case(f"model.logistic-{_p}")
    def _lr(p=_p):
        return _model("logistic").plot(plots=p)


for _kind in ("rforest", "xgboost", "mlp"):
    for _p in ("pred", "pdp", "pdp_sklearn", "pip", "pip_sklearn"):

        @case(f"model.{_kind}-{_p}")
        def _ml(kind=_kind, p=_p):
            return _model(kind).plot(plots=p)

    @case(f"model.{_kind}-dashboard")
    def _ml_dash(kind=_kind):
        return _model(kind, "regression").plot(plots="dashboard")


for _p in ("prune", "imp", "pip", "pred", "pdp"):

    @case(f"model.crtree-{_p}")
    def _ct(p=_p):
        return _model("crtree").plot(plots=p)


@case("model.crtree-dashboard")
def _ct_dash():
    return _model("crtree", "regression").plot(plots="dashboard")


_PERF = {
    "gains_plot": lambda f, d: f(d, "response", "yes", ["x1", "x2"]),
    "lift_plot": lambda f, d: f(d, "response", "yes", ["x1", "x2"]),
    "profit_plot": lambda f, d: f(d, "response", "yes", "x1", prn=False),
    "expected_profit_plot": lambda f, d: f(d, "response", "yes", "x1", prn=False),
    "ROME_plot": lambda f, d: f(d, "response", "yes", "x1"),
    "uplift_plot": lambda f, d: f(d, "response", "yes", "x1", "treatment", "treatment"),
    "inc_uplift_plot": lambda f, d: f(d, "response", "yes", "x1", "treatment", "treatment"),
    "inc_profit_plot": lambda f, d: f(d, "response", "yes", "x1", "treatment", "treatment"),
}
for _fn in _PERF:

    @case(f"model.perf-{_fn}")
    def _perf(fn=_fn):
        import pyrsm.model.perf as perf

        return _PERF[fn](getattr(perf, fn), _perf_df())


# -- multivariate -------------------------------------------------------------

for _p in ("scree", "change"):

    @case(f"multivariate.pre_factor-{_p}")
    def _pf(p=_p):
        from pyrsm.multivariate.pre_factor import pre_factor

        return pre_factor({"shopping": _data("multivariate", "shopping")}, "v1:v6").plot(plots=p)


for _p in ("attr", "resp"):

    @case(f"multivariate.full_factor-{_p}")
    def _ff(p=_p):
        from pyrsm.multivariate.full_factor import full_factor

        return full_factor(
            {"shopping": _data("multivariate", "shopping")}, "v1:v6", nr_fact=2
        ).plot(plots=p)


for _p, _cut in (("dendro", 0.05), ("dendro", 0), ("scree", 0.05), ("change", 0.05)):

    @case(f"multivariate.hclus-{_p}-cutoff{_cut}")
    def _hc(p=_p, cut=_cut):
        from pyrsm.multivariate.hclus import hclus

        return hclus({"shopping": _data("multivariate", "shopping")}, "v1:v6").plot(
            plots=p, cutoff=cut
        )


for _p in ("density", "bar", "scatter"):

    @case(f"multivariate.kclus-{_p}")
    def _km(p=_p):
        from pyrsm.multivariate.kclus import kclus

        return kclus({"shopping": _data("multivariate", "shopping")}, "v1:v6", nr_clus=3).plot(
            plots=p
        )


@case("multivariate.kclus-categorical")
def _km_cat():
    from pyrsm.multivariate.kclus import kclus

    tp = _data("multivariate", "toothpaste")
    return kclus({"toothpaste": tp}, ["v1", "v2", "gender"], nr_clus=2).plot(plots="density")


@case("multivariate.mds")
def _mds():
    from pyrsm.multivariate.mds import mds

    return mds({"city": _data("multivariate", "city")}, "from", "to", "distance").plot()


for _p in ("brand", "attr", "pref"):

    @case(f"multivariate.prmap-{_p}")
    def _pm(p=_p):
        from pyrsm.multivariate.prmap import prmap

        pm = prmap(
            {"retailers": _data("multivariate", "retailers")},
            "retailer",
            "good_value:cluttered",
            pref=["segment1", "segment2"],
        )
        return pm.plot(plots=["brand", p] if p != "brand" else "brand")


for _p in ("pw", "iw"):

    @case(f"multivariate.conjoint-{_p}")
    def _cj(p=_p):
        from pyrsm.multivariate.conjoint import conjoint

        return conjoint({"mp3": _data("multivariate", "mp3")}, rvar="Rating", evar="Memory:Shape").plot(
            plots=p
        )

    @case(f"multivariate.cbc-{_p}")
    def _cbc(p=_p):
        from pyrsm.multivariate.cbc import cbc

        return cbc(
            {"choc": _data("multivariate", "choc")},
            rvar="chosen",
            evar=["brand", "chocolate", "price", "nuts"],
            id="choice_id",
            alt="alt",
        ).plot(plots=p)


# -- eda, design, decide ------------------------------------------------------


@case("eda.distr")
def _distr():
    from pyrsm.eda.distr import distr

    return distr(_data("data", "diamonds").select(["price", "carat", "cut"])).plot()


_VIS = {
    "dist": dict(x="price"),
    "hist": dict(x="price"),
    "density": dict(x="price"),
    "scatter": dict(x="carat", y="price", smooth="lm"),
    "bar": dict(x="cut"),
    "line": dict(x="carat", y="price"),
    "box": dict(x="cut", y="price"),
    "violin": dict(x="cut", y="price"),
}
for _g in _VIS:

    @case(f"eda.visualize-{_g}")
    def _vis(g=_g):
        from pyrsm.eda.visualize import visualize

        return visualize(_data("data", "diamonds").head(500), geom=g, **_VIS[g])


@case("eda.visualize-combined")
def _vis_comb():
    from pyrsm.eda.visualize import visualize

    return visualize(_data("data", "diamonds").head(500), x=["price", "carat"], combine="x")


@case("eda.visualize-scatter-categorical-agg")
def _vis_agg():
    from pyrsm.eda.visualize import visualize

    return visualize(_data("data", "diamonds").head(500), x="cut", y="price", agg="mean")


@case("design.sample_size_comp")
def _ssc():
    from pyrsm.design.sample_size_comp import sample_size_comp

    return sample_size_comp(type="mean", delta=2, sd=10, conf=0.95, power=0.8).plot()


@case("decide.optimize-lp")
def _opt():
    from pyrsm.decide.optimize import optimize

    return optimize(_production_mix()).plot(resolution=60)


@case("decide.optimize-robustness")
def _opt_rr():
    from pyrsm.decide.optimize import optimize

    return optimize(_production_mix()).robustness(trials=30, coef_noise=0.2, seed=1).plot()


# -- model > evaluate ---------------------------------------------------------


def _eval_data(name):
    import polars as pl
    from pathlib import Path

    return pl.read_csv(Path(__file__).parent / "data" / "evaluate" / name)


for _p in ("lift", "gains", "profit", "expected_profit", "rome"):

    @case(f"model.evaluate.evalbin-{_p}")
    def _eb(p=_p):
        from pyrsm.model.evaluate import evalbin

        return evalbin(_eval_data("dvd_preds.csv"), ["pred_logit", "pred_rnd"], "buy",
                       lev="yes", margin=6).plot(plots=p)


for _p in ("inc_uplift", "uplift", "inc_profit"):

    @case(f"model.evaluate.uplift-{_p}")
    def _up(p=_p):
        from pyrsm.model.evaluate import uplift

        return uplift(_eval_data("dvd_preds.csv"), "pred_logit", "buy", tvar="treat",
                      lev="yes", tlev="yes").plot(plots=p)


@case("model.evaluate.confusion")
def _cnf():
    from pyrsm.model.evaluate import confusion

    return confusion(_eval_data("dvd_preds.csv"), "pred_logit", "buy", lev="yes").plot()


@case("model.evaluate.evalreg")
def _er():
    from pyrsm.model.evaluate import evalreg

    return evalreg(_eval_data("diamonds_preds.csv"), ["pred_lm", "pred_rnd"], "price").plot()


@pytest.mark.parametrize("name", sorted(CASES))
def test_plot_is_plotnine_in_pyrsm_style(name, capsys):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = CASES[name]()
    capsys.readouterr()
    plots = list(iter_ggplots(out))
    assert plots, f"{name} returned no plot"
    expected = type(theme_pyrsm())
    for g in plots:
        assert type(g.theme) is expected, (
            f"{name}: theme is {type(g.theme).__name__}, expected theme_pyrsm() "
            f"({expected.__name__})"
        )


def test_theme_check_rejects_other_themes():
    """The runtime check tells theme_pyrsm apart from its theme_bw subclasses."""
    import pandas as pd
    from plotnine import aes, geom_point, theme_classic, theme_minimal

    g = ggplot(pd.DataFrame({"x": [1], "y": [1]}), aes("x", "y")) + geom_point()
    assert type((g + theme_pyrsm()).theme) is type(theme_pyrsm())
    for other in (theme_classic(), theme_minimal()):
        assert type((g + other).theme) is not type(theme_pyrsm())
    assert type(g.theme) is not type(theme_pyrsm())
