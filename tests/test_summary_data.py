"""The additive, JSON-safe structured-summary contract."""

from __future__ import annotations

import json

import polars as pl

from pyrsm import render_summary_html, render_summary_text
from pyrsm.basics.compare_means import compare_means
from pyrsm.model.logistic import logistic
from pyrsm.model.regress import regress


def _section(document, identifier):
    return next(section for section in document["sections"] if section["id"] == identifier)


def test_linear_summary_data_preserves_every_coefficient_and_requested_options(
    salary_data,
):
    model = regress(
        {"salary": salary_data},
        rvar="salary",
        evar=["yrs_since_phd", "yrs_service", "rank"],
    )

    document = model.summary_data(vif=True, ssq=True, rmse=True, ci=True, dec=4)
    coefficients = _section(document, "coefficients")

    assert document["schema_version"] == 1
    assert document["analysis_kind"] == "linear_regression"
    assert document["options"]["dec"] == 4
    assert len(coefficients["rows"]) == model.coef.height
    assert [column["key"] for column in coefficients["columns"]] == model.coef.columns
    assert _section(document, "model_fit")
    assert _section(document, "vif")
    assert _section(document, "sum_of_squares")
    assert _section(document, "rmse")
    assert _section(document, "confidence_intervals")
    json.dumps(document)


def test_logistic_summary_data_keeps_odds_ratio_columns_and_p_values():
    titanic_data = (
        pl.read_parquet("examples/data/model/titanic.parquet")
        .drop_nulls(subset=["age"])
        .to_pandas()
    )
    model = logistic(
        {"titanic": titanic_data},
        rvar="survived",
        lev="Yes",
        evar=["age", "sex", "pclass"],
    )

    document = model.summary_data(ci=True, dec=3)
    coefficients = _section(document, "coefficients")

    assert document["analysis_kind"] == "logistic_regression"
    assert len(coefficients["rows"]) == model.coef.height
    assert {column["key"] for column in coefficients["columns"]} >= {
        "OR",
        "OR%",
        "z.value",
        "p.value",
    }
    assert any(column["semantic_type"] == "p_value" for column in coefficients["columns"])
    assert _section(document, "confidence_intervals")


def test_compare_means_uses_the_same_document_and_safe_renderers(salary_data):
    result = compare_means({"salary": salary_data}, var1="rank", var2="salary")

    document = result.summary_data(extra=True, dec=2)
    text = render_summary_text(document)
    html = render_summary_html(document)

    assert document["analysis_kind"] == "compare_means"
    assert len(_section(document, "descriptive_statistics")["rows"]) == result.descriptive_stats.height
    assert len(_section(document, "pairwise_comparisons")["rows"]) == result.comp_stats.height
    assert "Pairwise mean comparisons" in text
    assert "pyrsm-summary" in html
    assert "<script" not in html.lower()
