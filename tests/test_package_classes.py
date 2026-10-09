"""``pyrsm.<pkg>.<name>`` stays the class after its submodule is imported directly."""

import polars as pl

import pyrsm as rsm


def test_hclus_stays_a_class_after_kclus_imports_its_module():
    data = pl.read_parquet("examples/data/multivariate/shopping.parquet")
    # kclus imports ``from .hclus import hclus`` while it runs (HC initialization)
    rsm.multivariate.kclus({"shopping": data}, vars=["v1", "v2", "v3"], hc_init=True)
    assert isinstance(rsm.multivariate.hclus, type)
    assert rsm.multivariate.hclus({"shopping": data}, vars=["v1", "v2"]).nobs > 0


def test_direct_submodule_import_keeps_the_class():
    from pyrsm.model.regress import regress  # noqa: F401

    assert isinstance(rsm.model.regress, type)
