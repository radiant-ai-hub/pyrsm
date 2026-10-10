"""conjoint_choice and cbc results may not change when their estimator is shared.

The conditional-logit estimator behind ``conjoint_choice`` (and the earlier
``cbc``) is also the engine of ``pyrsm.model.mnl``. These tests pin the
complete output of both classes -- printed summaries, coefficient tables,
part-worths, importance weights, willingness to pay, predictions and fit
statistics -- as they were before the estimator was shared
(tests/reference/conjoint_choice_golden.json, written at pyrsm da84b87).

Regenerate only for an intended change in results:
``python tests/test_conjoint_choice_golden.py --write``
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import pyrsm as rsm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data" / "multivariate"
GOLDEN = Path(__file__).resolve().parent / "reference" / "conjoint_choice_golden.json"
RTOL = 1e-9


def _printed(fn) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


def _frame(df: pl.DataFrame) -> dict:
    return {c: df[c].to_list() for c in df.columns}


def _cases() -> dict:
    minivan = pl.read_parquet(DATA / "minivan.parquet")
    market = pl.read_parquet(DATA / "minivan_market.parquet")
    choc = pl.read_parquet(DATA / "choc.parquet")
    attrs = ["seat", "cargo", "eng", "price"]
    out = {}

    def record(name, cc, wtp=None, preds=()):
        res = {
            "summary": _printed(cc.summary),
            "coef": _frame(cc.coef),
            "PW": _frame(cc.PW),
            "IW": _frame(cc.IW),
            "fit_stats": {k: float(v) for k, v in cc.fit_stats.items()},
            "converged": bool(cc.converged),
        }
        if wtp:
            res["wtp"] = _frame(cc.wtp(price=wtp))
        for i, p in enumerate(preds):
            res[f"predict_{i}"] = _frame(p(cc))
        out[name] = res

    cc = rsm.multivariate.conjoint_choice(
        {"minivan": minivan}, rvar="choice", lev="1", evar=attrs, id="task", alt="alt"
    )
    record(
        "minivan",
        cc,
        wtp="price",
        preds=[
            lambda m: m.predict(market),
            lambda m: m.predict(cmd={"eng": ["gas", "hyb", "elec"], "price": [30, 35, 40]}),
        ],
    )
    for c in ["yes", "no"]:
        record(
            f"minivan_carpool_{c}",
            rsm.multivariate.conjoint_choice(
                {"minivan": minivan},
                rvar="choice",
                lev="1",
                evar=attrs,
                id="task",
                alt="alt",
                data_filter=f"carpool == '{c}'",
            ),
            wtp="price",
        )
    record(
        "minivan_interaction",
        rsm.multivariate.conjoint_choice(
            {"minivan": minivan},
            rvar="choice",
            evar=attrs,
            ivar=["seat:price", "cargo:eng"],
            id="task",
        ),
        preds=[lambda m: m.predict(market)],
    )
    record(
        "choc",
        rsm.multivariate.conjoint_choice(
            {"choc": choc},
            rvar="chosen",
            evar=["brand", "chocolate", "price", "nuts"],
            id="choice_id",
            alt="alt",
        ),
        wtp="price",
        preds=[lambda m: m.predict(cmd={"brand": ["Hershey", "Lindt", "Godiva"], "price": [3.99]})],
    )

    ch = rsm.multivariate.cbc(
        {"choc": choc},
        rvar="chosen",
        evar=["brand", "chocolate", "price", "nuts"],
        id="choice_id",
        alt="alt",
    )
    out["cbc_choc"] = {
        "summary": _printed(ch.summary),
        "PW": _frame(ch.PW),
        "IW": _frame(ch.IW),
        "predict_0": _frame(
            ch.predict(
                pred_cmd={"brand": ["Hershey", "Lindt", "Godiva"], "chocolate": "Dark",
                          "price": 3.99, "nuts": "Yes"}
            )
        ),
        "predict_1": _frame(ch.predict()),
    }
    return out


def _compare(path: str, got, want) -> None:
    if isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(want), f"{path}: keys differ"
        for k in want:
            _compare(f"{path}.{k}", got[k], want[k])
    elif isinstance(want, list):
        assert len(got) == len(want), f"{path}: length differs"
        if want and all(isinstance(w, (int, float)) and not isinstance(w, bool) for w in want if w is not None):
            g = np.array([np.nan if v is None else v for v in got], dtype=float)
            w = np.array([np.nan if v is None else v for v in want], dtype=float)
            np.testing.assert_allclose(g, w, rtol=RTOL, atol=1e-12, equal_nan=True, err_msg=path)
        else:
            assert got == want, f"{path}: values differ"
    elif isinstance(want, float):
        np.testing.assert_allclose(got, want, rtol=RTOL, atol=1e-12, err_msg=path)
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"


@pytest.fixture(scope="module")
def current():
    return _cases()


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text())


@pytest.mark.parametrize(
    "case",
    ["minivan", "minivan_carpool_yes", "minivan_carpool_no", "minivan_interaction", "choc", "cbc_choc"],
)
def test_results_unchanged(case, current, golden):
    _compare(case, current[case], golden[case])


if __name__ == "__main__" and "--write" in sys.argv:
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(_cases(), indent=1))
    print(f"wrote {GOLDEN}")
