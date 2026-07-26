"""Reference-fixture integrity and data-parity checks.

These guard against drift between the example parquet datasets shipped with
pyrsm and the data radiant.multivariate used to generate the reference
fixtures.
"""

import numpy as np
import pytest
from conftest import MV_REF_BASE, mv_load_dataset

EXPECTED_FIXTURES = [
    "pre_factor_shopping",
    "pre_factor_toothpaste",
    "pre_factor_diamonds",
    "pre_factor_toothpaste_hcor",
    "full_factor_shopping_2",
    "full_factor_shopping_2_none",
    "full_factor_shopping_3",
    "full_factor_toothpaste_2",
    "full_factor_diamonds_1",
    "full_factor_toothpaste_hcor",
    "hclus_shopping",
    "hclus_toothpaste",
    "hclus_toothpaste_id",
    "kclus_shopping_2",
    "kclus_shopping_3",
    "kclus_toothpaste_3",
    "mds_city_metric",
    "mds_city_nonmetric",
    "mds_tpbrands_metric",
    "mds_tpbrands_nonmetric",
    "prmap_computer",
    "prmap_retailers",
    "conjoint_mp3",
    "conjoint_carpet",
    "conjoint_movie",
    # expanded parity fixtures
    "pre_factor_toothpaste_mixed",
    "full_factor_shopping_2_quartimax",
    "full_factor_shopping_2_oblimin",
    "full_factor_shopping_2_simplimax",
    "full_factor_shopping_ml2",
    "full_factor_toothpaste_ml2",
    "hclus_toothpaste_gower",
    "kclus_toothpaste_kproto",
    "mds_city_metric_3d",
    "prmap_retailers_3d",
    "conjoint_mp3_int",
    "conjoint_mp3_by_radio",
]


def test_all_fixtures_present():
    missing = [f for f in EXPECTED_FIXTURES if not (MV_REF_BASE / f"{f}.json").exists()]
    assert not missing, f"missing reference fixtures: {missing}"


def test_session_info_present():
    assert (MV_REF_BASE / "sessionInfo.txt").exists()


@pytest.mark.parametrize(
    "dataset,shape",
    [
        ("shopping", (20, 7)),
        ("toothpaste", (60, 10)),
        ("city", (45, 3)),
        ("tpbrands", (45, 4)),
        ("computer", (5, 8)),
        ("retailers", (6, 10)),
        ("mp3", (18, 6)),
        ("carpet", (18, 6)),
        ("movie", (18, 6)),
    ],
)
def test_dataset_shapes(dataset, shape):
    assert mv_load_dataset(dataset).shape == shape


def test_conjoint_factor_levels_match_r():
    # the base level for treatment coding is the first Enum level; these must
    # match what radiant used (verified during fixture generation)

    mp3 = mv_load_dataset("mp3")
    assert mp3["Memory"].dtype.categories.to_list() == ["4GB", "6GB", "8GB"]
    assert mp3["Shape"].dtype.categories.to_list() == [
        "Circular",
        "Rectangular",
        "Square",
    ]


def test_diamonds_data_parity():
    # Rsq from the pre-factor math must match the value baked into the R test
    df = mv_load_dataset("diamonds")
    X = df.select(["price", "carat", "table"]).to_numpy().astype(float)
    cm = np.corrcoef(X, rowvar=False)
    r2 = 1 - 1 / np.diag(np.linalg.inv(cm))
    np.testing.assert_allclose(r2, [0.861258, 0.863566, 0.045071], atol=1e-5)
