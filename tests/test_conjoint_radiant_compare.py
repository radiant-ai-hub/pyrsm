"""Parity tests: pyrsm.multivariate.conjoint vs radiant.multivariate::conjoint.

Extends tests/test_multivariate_conjoint.py (which checks PW / IW /
coefficients / std.errors / R-squared for mp3, carpet, movie, one interaction
model and coefficients for one by-group model).

Reference values come from tests/reference/radiant_multivariate/
conjoint_compare_*.json, written by scripts/generate_conjoint_compare_reference.R
(run ``Rscript scripts/generate_conjoint_compare_reference.R`` from the repo
root).

Tests marked ``xfail(strict=True)`` document a place where pyrsm currently
differs from radiant (a bug or a missing feature); they will start to XPASS -
and therefore fail - once pyrsm is fixed, so the marker must then be removed.
"""

from __future__ import annotations

import itertools
import re
import warnings

import numpy as np
import polars as pl
import pytest

from pyrsm.multivariate import conjoint, store_predictions

warnings.filterwarnings("ignore", category=RuntimeWarning)

COEF_TOL = 1e-8
PW_TOL = 1.0001e-3  # both sides round PW/IW to 3 decimals; ties may round apart
PRED_TOL = 5.0001e-4  # pyrsm.predict rounds to dec=3, R does not


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ref(mv_ref, name):
    return mv_ref(f"conjoint_compare_{name}")


def _col(ref_df: dict, col: str) -> list:
    return ref_df["data"][col]


def _assert_coeff(pyrsm_coeff: pl.DataFrame, ref_coeff: dict, rtol=1e-6):
    assert pyrsm_coeff["label"].to_list() == ref_coeff["label"]
    np.testing.assert_allclose(
        pyrsm_coeff["coefficient"].to_numpy(), ref_coeff["coefficient"], atol=COEF_TOL, rtol=0
    )
    np.testing.assert_allclose(
        pyrsm_coeff["std.error"].to_numpy(), ref_coeff["std.error"], atol=COEF_TOL, rtol=0
    )
    np.testing.assert_allclose(
        pyrsm_coeff["t.value"].to_numpy(), ref_coeff["t.value"], rtol=rtol, atol=1e-8
    )
    np.testing.assert_allclose(
        pyrsm_coeff["p.value"].to_numpy(), ref_coeff["p.value"], rtol=1e-5, atol=1e-12
    )


def _assert_fit(model, ref):
    """statsmodels OLS results vs broom::glance()."""
    assert model.rsquared == pytest.approx(ref["rsq"], abs=1e-10)
    assert model.rsquared_adj == pytest.approx(ref["rsq_adj"], abs=1e-10)
    assert model.fvalue == pytest.approx(ref["fstat"], rel=1e-8)
    assert model.f_pvalue == pytest.approx(ref["f_pvalue"], rel=1e-5, abs=1e-15)
    assert model.df_model == ref["df_model"]
    assert model.df_resid == ref["df_resid"]
    assert model.nobs == ref["nobs"]


def _assert_tables(m: dict, ref: dict):
    """PW / IW long tables and plot_ylim for one fitted model."""
    rpw = ref["PW"]
    keep = [i for i, a in enumerate(rpw["Attributes"]) if a is not None]  # drop R's NA rows
    assert m["PW"]["Attributes"].to_list() == [rpw["Attributes"][i] for i in keep]
    assert m["PW"]["Levels"].to_list() == [rpw["Levels"][i] for i in keep]
    np.testing.assert_allclose(
        m["PW"]["PW"].to_numpy(), [rpw["PW"][i] for i in keep], atol=PW_TOL, rtol=0
    )
    assert m["IW"]["Attributes"].to_list() == ref["IW"]["Attributes"]
    np.testing.assert_allclose(m["IW"]["IW"].to_numpy(), ref["IW"]["IW"], atol=PW_TOL, rtol=0)
    ylim = m["plot_ylim"]
    yl = ref["plot_ylim"]
    assert list(ylim) == yl["attr"]
    np.testing.assert_allclose([ylim[a][0] for a in yl["attr"]], yl["Min"], atol=1e-8)
    np.testing.assert_allclose([ylim[a][1] for a in yl["attr"]], yl["Max"], atol=1e-8)


def _norm(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _summary(cj, capsys, **kwargs) -> str:
    capsys.readouterr()
    cj.summary(**kwargs)
    return capsys.readouterr().out


def _header(text: str) -> list[str]:
    """Lines up to (not including) the first blank line, whitespace-normalized."""
    out = []
    for line in text.splitlines():
        if not line.strip():
            break
        out.append(_norm(line))
    return out


def _xtick_labels(g) -> list[str]:
    import matplotlib.pyplot as plt

    fig = g.draw()
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    plt.close(fig)
    return labels


# Datasets and model specifications. ``name`` is passed as the dataset name so
# summary headers match R (where df_name is ``dataset`` inside the generator).
FULL_CASES = {
    "full_mp3": dict(dataset="mp3", rvar="Rating", evar="Memory:Shape"),
    "full_carpet": dict(dataset="carpet", rvar="ranking", evar="design:money_back", reverse=True),
    "full_movie": dict(dataset="movie", rvar="Ranking", evar="price:food", reverse=True),
    "int_mp3": dict(dataset="mp3", rvar="Rating", evar="Memory:Shape", int="Memory:Shape"),
    "int_carpet_rev": dict(
        dataset="carpet", rvar="ranking", evar="design:money_back", int="price:seal", reverse=True
    ),
    "filter_mp3": dict(dataset="mp3", rvar="Rating", evar="Memory:Shape", data_filter="Rating > 30"),
    # pyrsm-only sort / slice; R reference built with mp3[1:16, ] and
    # arrange(mp3, desc(Rating)) %>% slice(1:14)
    "slice_mp3": dict(dataset="mp3", rvar="Rating", evar="Memory:Shape", slice="1:16"),
    "sortslice_mp3": dict(
        dataset="mp3", rvar="Rating", evar="Memory:Shape", sort="-Rating", slice="1:14"
    ),
}


def _fit(mv_data, spec, name="dataset"):
    spec = dict(spec)
    ds = spec.pop("dataset")
    return conjoint({name: mv_data(ds)}, **spec)


# ---------------------------------------------------------------------------
# 1 / 3. Full regression statistics, interactions, filter / sort / slice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", list(FULL_CASES))
def test_coefficients_t_p(mv_data, mv_ref, case):
    ref = _ref(mv_ref, case)
    cj = _fit(mv_data, FULL_CASES[case])
    assert cj.evar == ref["evar"]
    _assert_coeff(cj.coeff, ref["coeff"])


@pytest.mark.parametrize("case", list(FULL_CASES))
def test_fit_statistics(mv_data, mv_ref, case):
    ref = _ref(mv_ref, case)
    cj = _fit(mv_data, FULL_CASES[case])
    _assert_fit(cj.model_list["full"]["model"], ref)
    assert cj.nobs == ref["nobs"]
    assert cj.model_list["full"]["rsq_adj"] == pytest.approx(ref["rsq_adj"], abs=1e-10)


@pytest.mark.parametrize("case", list(FULL_CASES))
def test_pw_iw_plot_ylim(mv_data, mv_ref, case):
    # With interactions R's PW table also carries the interaction coefficients
    # as rows with Attributes/Levels = NA (and summary prints "NA NA -2.500");
    # pyrsm omits them. Main-effect rows, IW and plot_ylim must agree.
    ref = _ref(mv_ref, case)
    cj = _fit(mv_data, FULL_CASES[case])
    _assert_tables(cj.model_list["full"], ref)


def test_r_summary_df_and_nobs_are_off_by_one(mv_data, mv_ref):
    """Documents an R bug, not a pyrsm one.

    radiant prints ``df(glance$df - 1, df.residual)`` and
    ``Nr obs: glance$df + df.residual``, but broom >= 0.7 already reports the
    numerator df *without* the intercept. R therefore prints df(8,8) and
    "Nr obs: 17" for mp3 (18 rows, 9 slopes). pyrsm's model has the right values.
    """
    ref = _ref(mv_ref, "full_mp3")
    m = _fit(mv_data, FULL_CASES["full_mp3"]).model_list["full"]["model"]
    assert (m.df_model, m.df_resid, m.nobs) == (9, 8, 18)
    assert "df(8,8)" in ref["summary_additional"]
    assert "Nr obs: 17" in ref["summary_additional"]


# ---------------------------------------------------------------------------
# 2. by-group models
# ---------------------------------------------------------------------------

BY_CASES = {
    "by_mp3_radio": dict(dataset="mp3", rvar="Rating", evar=["Memory", "Price"], by="Radio"),
    "by_mp3_radio_rev": dict(
        dataset="mp3", rvar="Rating", evar=["Memory", "Price"], by="Radio", reverse=True
    ),
    "by_movie_sight_rev": dict(
        dataset="movie", rvar="Ranking", evar=["price", "comfort", "food"], by="sight",
        reverse=True,
    ),
}


@pytest.mark.parametrize("case", list(BY_CASES))
def test_by_group_models(mv_data, mv_ref, case):
    # reverse is applied within each group ((group max + 1) - x) on both sides
    ref = _ref(mv_ref, case)
    cj = _fit(mv_data, BY_CASES[case])
    assert cj.bylevs == ref["bylevs"]
    assert cj.evar == ref["evar"]
    for lev in ref["bylevs"]:
        g = ref["groups"][lev]
        m = cj.model_list[lev]
        _assert_coeff(m["coeff"], g["coeff"])
        _assert_fit(m["model"], g)
        _assert_tables(m, g)


@pytest.mark.parametrize("case", list(BY_CASES))
def test_by_wide_tables_values(mv_data, mv_ref, case):
    """The numbers in R's wide ``$PW`` / ``$IW`` (one row per group) are
    recoverable from pyrsm's per-group results."""
    ref = _ref(mv_ref, case)
    cj = _fit(mv_data, BY_CASES[case])
    pw, iw = ref["PW_wide"], ref["IW_wide"]
    for i, lev in enumerate(ref["bylevs"]):
        m = cj.model_list[lev]
        assert m["rsq"] == pytest.approx(_col(pw, ".Rsq")[i], abs=1e-10)
        np.testing.assert_allclose(
            m["coeff"]["coefficient"].to_numpy(),
            [_col(pw, c)[i] for c in pw["columns"][2:]],
            atol=COEF_TOL,
        )
        np.testing.assert_allclose(
            m["IW"]["IW"].to_numpy(), [_col(iw, c)[i] for c in iw["columns"][2:]], atol=PW_TOL
        )


def test_by_wide_tables_structure(mv_data, mv_ref):
    ref = _ref(mv_ref, "by_mp3_radio")
    cj = _fit(mv_data, BY_CASES["by_mp3_radio"])
    pw = cj.store("PW")
    assert pw.columns == ref["PW_wide"]["columns"]
    assert pw.height == len(ref["bylevs"])
    iw = cj.store("IW")
    assert iw.columns == ref["IW_wide"]["columns"]


def test_by_variable_inside_evar_is_dropped(mv_data, mv_ref):
    ref = _ref(mv_ref, "by_in_evar")
    cj = conjoint({"mp3": mv_data("mp3")}, rvar="Rating", evar="Memory:Shape", by="Radio")
    assert cj.evar == ref["evar"]


def test_by_aliased_terms(mv_data, mv_ref):
    # R: conjoint(mp3, "Rating", "Memory:Shape", by = "Radio") -> evar becomes
    # Memory, Size, Price, Shape (fixture by_in_evar)
    ref = _ref(mv_ref, "by_in_evar")
    cj = conjoint(
        {"mp3": mv_data("mp3")}, rvar="Rating", evar=["Memory", "Size", "Price", "Shape"],
        by="Radio",
    )
    for lev in ("No", "Yes"):
        rc = ref[f"coeff_{lev}"]
        assert cj.model_list[lev]["coeff"]["label"].to_list() == rc["label"]
        np.testing.assert_allclose(
            cj.model_list[lev]["coeff"]["coefficient"].to_numpy(), rc["coefficient"], atol=1e-6
        )
    iw = ref["IW_wide"]
    np.testing.assert_allclose(
        cj.model_list["No"]["IW"]["IW"].to_numpy(), [_col(iw, c)[0] for c in iw["columns"][2:]]
    )


def test_by_levels_for_string_by_variable(mv_data, mv_ref):
    """Deliberate difference from R (pyrsm policy: categorical data should be an
    Enum with a meaningful order; text columns use order of appearance, never an
    alphabetical sort). R's factor() sorts a character `by` to ['a', 'z']."""
    ref = _ref(mv_ref, "by_char")
    d = mv_data("mp3").with_columns(
        grp=pl.when(pl.col("Radio") == "No").then(pl.lit("z")).otherwise(pl.lit("a"))
    )
    cj = conjoint({"mp3": d}, rvar="Rating", evar=["Memory", "Price"], by="grp")
    assert cj.bylevs == d["grp"].unique(maintain_order=True).to_list()
    # with an Enum in R's order the groups come out as in R
    e = d.with_columns(pl.col("grp").cast(pl.Enum(ref["bylevs"])))
    cj = conjoint({"mp3": e}, rvar="Rating", evar=["Memory", "Price"], by="grp")
    assert cj.bylevs == ref["bylevs"]


# ---------------------------------------------------------------------------
# attribute types
# ---------------------------------------------------------------------------


def test_character_attribute_base_level(mv_data, mv_ref):
    """Deliberate difference from R (pyrsm policy, see above): a text attribute's
    base level is its first value to appear ('$50' for mp3 Price), where R's
    factor() takes the alphabetically first ('$100'). With Price as an Enum in
    R's level order the estimates equal R's."""
    ref = _ref(mv_ref, "char_attr")
    d = mv_data("mp3").with_columns(pl.col("Price").cast(pl.Utf8))
    cj = conjoint({"mp3": d}, rvar="Rating", evar="Memory:Shape")
    first = d["Price"].unique(maintain_order=True).to_list()
    assert cj.PW.filter(pl.col("Attributes") == "Price")["Levels"].to_list() == first
    r_order = sorted(first)
    e = d.with_columns(pl.col("Price").cast(pl.Enum(r_order)))
    cj = conjoint({"mp3": e}, rvar="Rating", evar="Memory:Shape")
    assert cj.coeff["label"].to_list() == ref["coeff"]["label"]
    np.testing.assert_allclose(cj.coeff["coefficient"].to_numpy(), ref["coeff"]["coefficient"])
    assert cj.PW["Levels"].to_list() == ref["PW"]["Levels"]


def test_numeric_attribute(mv_data, mv_ref):
    ref = _ref(mv_ref, "num_attr")
    d = mv_data("mp3").with_columns(
        pl.col("Price").cast(pl.Utf8).str.replace(r"\$", "").cast(pl.Float64)
    )
    cj = conjoint({"mp3": d}, rvar="Rating", evar="Memory:Shape")
    assert cj.coeff["label"].to_list() == ref["coeff"]["label"]


# ---------------------------------------------------------------------------
# 4. predict
# ---------------------------------------------------------------------------


def _safe_predict(model, *args, **kwargs):
    """Call model.predict, turning any error into a plain AssertionError.

    patsy errors carry a frame whose f_locals raise KeyError, which crashes
    pytest's traceback formatting (INTERNALERROR) in Python 3.13, so the
    original exception must not be chained.
    """
    err = None
    try:
        return model.predict(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    raise AssertionError(err)


@pytest.fixture
def mp3_model(mv_data):
    return conjoint({"mp3": mv_data("mp3")}, rvar="Rating", evar="Memory:Shape")


def test_predict_pred_data_values(mv_data, mv_ref, mp3_model):
    ref = _ref(mv_ref, "predict")
    df = mv_data("mp3")
    p = mp3_model.predict(df)
    np.testing.assert_allclose(
        p["Prediction"].to_numpy(), _col(ref["pred_data"], "Prediction"), atol=PRED_TOL
    )
    conf = mp3_model.predict(df, se=True)
    assert conf.columns == ref["pred_conf"]["columns"]
    for c in ["Prediction", "2.5%", "97.5%", "+/-"]:
        np.testing.assert_allclose(conf[c].to_numpy(), _col(ref["pred_conf"], c), atol=PRED_TOL)
    p90 = mp3_model.predict(df, se=True, interval="prediction", conf_lev=0.9)
    r90 = ref["pred_pred90"]
    assert p90.columns == r90["columns"]
    for c in r90["columns"][-4:]:
        np.testing.assert_allclose(p90[c].to_numpy(), _col(r90, c), atol=PRED_TOL)


def test_predict_interaction_model(mv_data, mv_ref):
    ref = _ref(mv_ref, "predict")
    df = mv_data("mp3")
    cj = conjoint({"mp3": df}, rvar="Rating", evar="Memory:Shape", int="Memory:Shape")
    p = cj.predict(df, se=True)
    assert p.columns == ref["pred_int"]["columns"]
    for c in ["Prediction", "2.5%", "97.5%", "+/-"]:
        np.testing.assert_allclose(p[c].to_numpy(), _col(ref["pred_int"], c), atol=PRED_TOL)


def test_predict_full_precision(mv_data, mv_ref, mp3_model):
    ref = _ref(mv_ref, "predict")
    p = mp3_model.predict(mv_data("mp3"))
    np.testing.assert_allclose(
        p["Prediction"].to_numpy(), _col(ref["pred_data"], "Prediction"), atol=1e-8
    )


def test_predict_output_columns(mv_data, mv_ref, mp3_model):
    ref = _ref(mv_ref, "predict")
    assert mp3_model.predict(mv_data("mp3"), se=True).columns == ref["pred_conf"]["columns"]


def test_predict_interval_labels_conf90(mv_data, mv_ref, mp3_model):
    ref = _ref(mv_ref, "predict")
    p = mp3_model.predict(mv_data("mp3"), se=True, interval="prediction", conf_lev=0.9)
    assert p.columns == ref["pred_pred90"]["columns"]


def test_predict_pred_cmd_fully_specified(mv_data, mv_ref, mp3_model):
    ref = _ref(mv_ref, "predict")
    p = mp3_model.predict(pred_cmd=ref["cmd_all"])
    np.testing.assert_allclose(
        p["Prediction"].to_numpy(), _col(ref["pred_cmd_all"], "Prediction"), atol=PRED_TOL
    )


def test_predict_pred_cmd_partial(mv_data, mv_ref, mp3_model):
    ref = _ref(mv_ref, "predict")
    p = _safe_predict(mp3_model, pred_cmd=ref["cmd_partial"])  # "Memory = c('4GB','8GB')"
    np.testing.assert_allclose(
        p["Prediction"].to_numpy(), _col(ref["pred_cmd_partial"], "Prediction"), atol=PRED_TOL
    )


def test_predict_pred_cmd_comma_separated(mv_data, mv_ref, mp3_model):
    ref = _ref(mv_ref, "predict")
    full = ref["cmd_comma"] + ", Radio = 'No', Size = 'Large', Shape = 'Circular'"
    p = _safe_predict(mp3_model, pred_cmd=full)
    np.testing.assert_allclose(
        p["Prediction"].to_numpy(), _col(ref["pred_cmd_comma"], "Prediction"), atol=PRED_TOL
    )


def test_predict_pred_data_plus_cmd(mv_data, mv_ref, mp3_model):
    ref = _ref(mv_ref, "predict")
    p = _safe_predict(mp3_model, mv_data("mp3"), pred_cmd=ref["cmd_mutate"])
    np.testing.assert_allclose(
        p["Prediction"].to_numpy(), _col(ref["pred_data_cmd"], "Prediction"), atol=PRED_TOL
    )


def test_predict_by_pred_data_values(mv_data, mv_ref):
    ref = _ref(mv_ref, "by_mp3_radio")
    cj = _fit(mv_data, BY_CASES["by_mp3_radio"], name="mp3")
    p = cj.predict(mv_data("mp3"))
    assert p["Radio"].to_list() == _col(ref["pred_data"], "Radio")
    np.testing.assert_allclose(
        p["Prediction"].to_numpy(), _col(ref["pred_data"], "Prediction"), atol=PRED_TOL
    )


def test_predict_by_columns(mv_data, mv_ref):
    ref = _ref(mv_ref, "by_mp3_radio")
    cj = _fit(mv_data, BY_CASES["by_mp3_radio"], name="mp3")
    assert cj.predict(mv_data("mp3")).columns == ref["pred_data"]["columns"]


def test_predict_by_pred_cmd_values(mv_data, mv_ref):
    # R: predict(conjoint(mp3, "Rating", c("Memory","Price"), by = "Radio"),
    #            pred_cmd = "Memory = c('4GB','8GB'); Price = c('$50','$150')")
    ref = _ref(mv_ref, "by_mp3_radio")
    cj = _fit(mv_data, BY_CASES["by_mp3_radio"], name="mp3")
    p = cj.predict(pred_cmd=ref["pred_cmd"])
    rp = ref["pred_cmd_result"]
    # same rows in the same order as R (the first attribute varies fastest)
    for c in ["Radio", "Memory", "Price"]:
        assert p[c].cast(pl.Utf8).to_list() == _col(rp, c)
    np.testing.assert_allclose(p["Prediction"].to_numpy(), _col(rp, "Prediction"), atol=PRED_TOL)


def test_predict_pred_cmd_row_order(mv_data, mv_ref):
    ref = _ref(mv_ref, "by_mp3_radio")
    cj = _fit(mv_data, BY_CASES["by_mp3_radio"], name="mp3")
    p = cj.predict(pred_cmd=ref["pred_cmd"])
    np.testing.assert_allclose(
        p["Prediction"].to_numpy(), _col(ref["pred_cmd_result"], "Prediction"), atol=PRED_TOL
    )


# ---------------------------------------------------------------------------
# 5. store
# ---------------------------------------------------------------------------


def test_store_prediction_values(mv_data, mv_ref, mp3_model):
    ref = _ref(mv_ref, "predict")
    df = mv_data("mp3")
    out = store_predictions(df, mp3_model.predict(df, se=True), name="pref, lb, ub")
    np.testing.assert_allclose(out["pref"].to_numpy(), ref["store_single"], atol=PRED_TOL)
    np.testing.assert_allclose(out["lb"].to_numpy(), ref["store_triple_lb"], atol=PRED_TOL)
    np.testing.assert_allclose(out["ub"].to_numpy(), ref["store_triple_ub"], atol=PRED_TOL)


def test_store_prediction_names(mv_data, mv_ref, mp3_model):
    ref = _ref(mv_ref, "predict")
    df = mv_data("mp3")
    pred = mp3_model.predict(df, se=True)
    assert store_predictions(df, pred, name="pref").columns == ref["store_single_cols"]
    assert store_predictions(df, pred, name="pref, lb, ub").columns == ref["store_triple_cols"]


def test_store_by_predictions_row_mismatch_raises(mv_data):
    """pyrsm refuses to store 36 by-group predictions into 18 rows. (radiant
    warns and silently stores the first group's 18 predictions - an R bug.)"""
    df = mv_data("mp3")
    cj = _fit(mv_data, BY_CASES["by_mp3_radio"], name="mp3")
    with pytest.raises(ValueError):
        store_predictions(df, cj.predict(df), name="p")


# ---------------------------------------------------------------------------
# 6. summary text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["full_mp3", "full_carpet", "filter_mp3", "int_mp3"])
def test_summary_header(mv_data, mv_ref, case, capsys):
    ref = _ref(mv_ref, case)
    out = _summary(_fit(mv_data, FULL_CASES[case]), capsys)
    assert _header(out) == _header(ref["summary"])


def test_summary_header_by(mv_data, mv_ref, capsys):
    ref = _ref(mv_ref, "by_mp3_radio")
    cj = _fit(mv_data, BY_CASES["by_mp3_radio"])
    for lev in ref["bylevs"]:
        out = _summary(cj, capsys, show=lev)
        assert _header(out) == _header(ref["groups"][lev]["summary"])


def test_summary_body_lines(mv_data, mv_ref, capsys):
    """Every part-worth, importance weight and coefficient row of Radiant's
    summary appears with the same label and the same 3-decimal value (0.000 and
    8.500, not 0.0 / 8.5). The table layout is pyrsm's (as in regress), so rows
    are compared on their cells rather than as identical lines."""
    ref = _ref(mv_ref, "full_mp3")
    out = _summary(_fit(mv_data, FULL_CASES["full_mp3"]), capsys)
    assert "shape:" not in out
    rows = {tuple(c.strip() for c in line.strip("│ ").split("┆")) for line in out.splitlines()}
    want = []
    for line in ref["summary"].splitlines():
        cells = line.split()
        if len(cells) >= 2 and re.fullmatch(r"-?\d+\.\d{3}", cells[-1]):
            want.append((cells[:-1], cells[-1]))
    assert len(want) == 15 + 5 + 10  # PW rows, IW rows, coefficients
    flat = [" ".join(r) for r in rows]
    for labels, value in want:
        assert any(all(lab in f for lab in labels) and f.endswith(value) for f in flat), (
            labels, value)


def test_summary_additional(mv_data, mv_ref, capsys):
    out = _summary(_fit(mv_data, FULL_CASES["full_mp3"]), capsys, additional=True)
    # same numbers as R; layout as pyrsm's regress ("R-squared: x, Adjusted ...")
    assert "R-squared: 0.999, Adjusted R-squared: 0.997" in out
    assert "F-statistic: 731.197" in out
    assert "Signif. codes:" in out


def test_mc_diag_gvif(mv_data, mv_ref):
    ref = _ref(mv_ref, "full_mp3")
    v = _fit(mv_data, FULL_CASES["full_mp3"]).vif()
    got = dict(zip(v["term"].to_list(), v["VIF"].to_list()))
    for name, val in zip(ref["vif"]["names"], ref["vif"]["VIF"]):
        assert got[name] == pytest.approx(val, abs=1e-3)


# ---------------------------------------------------------------------------
# 7. plots (data behind the plots, not pixels)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["full_mp3", "full_carpet", "full_movie", "int_mp3"])
def test_plot_data_and_titles(mv_data, mv_ref, case):
    ref = _ref(mv_ref, case)
    cj = _fit(mv_data, FULL_CASES[case])
    plots = cj.plot(plots=["pw", "iw"])
    names = list(ref["plots"])  # evar..., "iw"
    assert len(plots) == len(names)
    for g, nm in zip(plots, names):
        r = ref["plots"][nm]
        assert g.labels.get("title", None) == r["title"].strip()
        ycol = "IW" if nm == "iw" else "PW"
        # R bug: with interactions PW.df has NA-attribute rows, and
        # PW.df[PW.df$Attributes == var, ] turns them into NA rows, so every R
        # part-worth plot gets 4 missing points and an extra "NA" x-axis level.
        ry = [y for y in r["y"] if y is not None]
        np.testing.assert_allclose(g.data[ycol].to_numpy(), ry, atol=PW_TOL)


@pytest.mark.parametrize("case", ["full_mp3", "full_movie"])
def test_plot_scale_ylim(mv_data, mv_ref, case):
    """scale_plot=True: pyrsm uses coord_cartesian(ylim=plot_ylim[var]); R uses
    ylim(plot_ylim[var, 'Min'], plot_ylim[var, 'Max']). Same limits."""
    ref = _ref(mv_ref, case)
    cj = _fit(mv_data, FULL_CASES[case])
    plots = cj.plot(plots="pw", scale_plot=True)
    yl = ref["plot_ylim"]
    for g, a, lo, hi in zip(plots, yl["attr"], yl["Min"], yl["Max"]):
        assert g.coordinates.limits.y == pytest.approx((lo, hi), abs=1e-8)


def test_plot_iw_axis_order(mv_data, mv_ref):
    # R's IW x-axis is a character column -> alphabetical; same in pyrsm
    ref = _ref(mv_ref, "full_movie")
    g = _fit(mv_data, FULL_CASES["full_movie"]).plot(plots="iw")
    assert _xtick_labels(g) == ref["plots"]["iw"]["x_labels"]


@pytest.mark.parametrize("case", ["full_mp3", "full_carpet", "full_movie"])
def test_plot_pw_axis_level_order(mv_data, mv_ref, case):
    ref = _ref(mv_ref, case)
    cj = _fit(mv_data, FULL_CASES[case])
    for g, var in zip(cj.plot(plots="pw"), cj.evar):
        assert _xtick_labels(g) == ref["plots"][var]["x_labels"]


def test_plot_titles_by(mv_data, mv_ref):
    ref = _ref(mv_ref, "by_mp3_radio")
    cj = _fit(mv_data, BY_CASES["by_mp3_radio"])
    r = ref["groups"]["Yes"]["plots"]
    plots = cj.plot(plots=["pw", "iw"], show="Yes")
    assert [g.labels.get("title", None) for g in plots] == [r[k]["title"] for k in r]


def test_plot_values_by(mv_data, mv_ref):
    ref = _ref(mv_ref, "by_mp3_radio")
    cj = _fit(mv_data, BY_CASES["by_mp3_radio"])
    for lev in ref["bylevs"]:
        r = ref["groups"][lev]["plots"]
        plots = cj.plot(plots=["pw", "iw"], show=lev)
        for g, nm in zip(plots, r):
            ycol = "IW" if nm == "iw" else "PW"
            np.testing.assert_allclose(g.data[ycol].to_numpy(), r[nm]["y"], atol=PW_TOL)
