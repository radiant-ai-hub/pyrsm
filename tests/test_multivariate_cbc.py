"""Tests for pyrsm.multivariate.cbc (choice-based conjoint / conditional logit).

The example dataset ``choc`` is simulated from known part-worths
(``scripts/generate_cbc_example.py``), so the estimates are checked against the
data-generating values, and the conditional-logit fit is cross-validated against
statsmodels' ``ConditionalLogit`` on the same treatment-coded design.
"""

import numpy as np
import polars as pl
import pytest

from pyrsm.multivariate import cbc

EVAR = ["brand", "chocolate", "price", "nuts"]


def _fit(mv_data):
    return cbc(
        {"choc": mv_data("choc")}, rvar="chosen", evar=EVAR, id="choice_id", alt="alt"
    )


def test_cbc_construction(mv_data):
    m = _fit(mv_data)
    assert m.ntasks == 1600
    assert m.nobs == 4800
    # one coefficient per non-base level (brand 2, chocolate 2, nuts 1) + price
    assert m.coeff.height == 6
    assert m.fit["converged"]


def test_cbc_part_worths_base_zero_and_iw(mv_data):
    m = _fit(mv_data)
    # base (first) level of each categorical attribute has part-worth 0
    for v in ("brand", "chocolate", "nuts"):
        base = m.PW.filter(pl.col("Attributes") == v)["PW"].to_list()[0]
        assert base == 0.0
    # importance weights sum to 1 (within 3-decimal display rounding)
    assert abs(m.IW["IW"].sum() - 1.0) < 2e-3


def test_cbc_recovers_known_part_worths(mv_data):
    m = _fit(mv_data)
    co = dict(zip(m.coeff["label"].to_list(), m.coeff["coefficient"].to_list()))
    # data-generating values: Lindt 0.8, Godiva 1.3, Dark 0.5, White -0.4,
    # price -0.9, nuts 0.3 (recovered up to sampling noise)
    assert abs(co["brand|Lindt"] - 0.8) < 0.15
    assert abs(co["brand|Godiva"] - 1.3) < 0.15
    assert abs(co["chocolate|Dark"] - 0.5) < 0.15
    assert co["chocolate|White"] < 0
    assert abs(co["price"] - (-0.9)) < 0.15
    assert co["nuts|Yes"] > 0


def test_cbc_matches_statsmodels_conditional_logit(mv_data):
    """Cross-validate the internal MNL estimator against statsmodels."""
    from statsmodels.discrete.conditional_models import ConditionalLogit

    m = _fit(mv_data)
    sub = m._sub
    X, names = m._design(sub)
    y = sub["chosen"].cast(pl.Float64).to_numpy()
    g = sub["choice_id"].to_numpy()
    res = ConditionalLogit(y, X, groups=g).fit(disp=0)
    assert np.abs(np.array(m.fit["params"]) - res.params).max() < 1e-3
    assert abs(m.fit["llf"] - res.llf) < 1e-3


def test_cbc_predict_probabilities(mv_data):
    m = _fit(mv_data)
    pr = m.predict()
    assert {"choice_id", "alt", "utility", "probability"}.issubset(pr.columns)
    # probabilities sum to 1 within each choice task (3-decimal display rounding)
    sums = pr.group_by("choice_id").agg(pl.col("probability").sum())["probability"]
    assert np.allclose(sums.to_numpy(), 1.0, atol=2e-3)


def test_cbc_predict_expand_grid(mv_data):
    m = _fit(mv_data)
    grid = m.predict(
        pred_cmd={
            "brand": ["Hershey", "Lindt", "Godiva"],
            "chocolate": "Dark",
            "price": 3.99,
            "nuts": "Yes",
        }
    )
    assert grid.height == 3
    assert np.isclose(grid["probability"].sum(), 1.0, atol=2e-3)
    # rows are in the supplied order (Hershey, Lindt, Godiva); higher brand
    # part-worth -> higher choice probability, so probabilities increase
    probs = grid["probability"].to_numpy()
    assert probs[0] < probs[1] < probs[2]


def test_cbc_predict_cmd_string(mv_data):
    m = _fit(mv_data)
    g1 = m.predict(pred_cmd="brand = c('Hershey','Godiva'); chocolate = Dark; price = 3.99; nuts = No")
    g2 = m.predict(pred_cmd={"brand": ["Hershey", "Godiva"], "chocolate": "Dark",
                             "price": 3.99, "nuts": "No"})
    assert g1.height == 2
    assert np.allclose(g1["probability"].to_numpy(), g2["probability"].to_numpy())


def test_cbc_interactions(mv_data):
    base = _fit(mv_data)
    inter = cbc(
        {"choc": mv_data("choc")},
        rvar="chosen", evar=EVAR, id="choice_id", alt="alt", int="brand:nuts",
    )
    # interaction terms add coefficients (brand has 2 non-base x nuts 1 non-base)
    assert inter.coeff.height > base.coeff.height
    assert any(":" in lbl for lbl in inter.coeff["label"].to_list())


def test_cbc_store_and_summary_plot(mv_data, capsys):
    m = _fit(mv_data)
    assert "PW" in m.store("PW").columns
    assert "IW" in m.store("IW").columns
    m.summary()
    out = capsys.readouterr().out
    assert "Choice-based conjoint analysis" in out
    assert "part-worths" in out
    assert "pseudo-R-squared" in out
    assert m.plot(plots="pw") is not None
    assert m.plot(plots="iw") is not None
    multi = m.plot(plots=["pw", "iw"])
    assert isinstance(multi, list) and len(multi) == 4  # 3 categorical pw + 1 iw
    with pytest.raises(ValueError):
        m.plot(plots=["bogus"])


def test_cbc_data_filter(mv_data):
    df = mv_data("choc")
    m = cbc({"choc": df}, rvar="chosen", evar=EVAR, id="choice_id", alt="alt",
            data_filter="price <= 3.99")
    assert m.nobs < df.height
