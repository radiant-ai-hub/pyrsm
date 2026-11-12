"""pyrsm.multivariate.mds against radiant.multivariate::mds, number by number.

Fixtures: scripts/generate_mds_reference.R (radiant.multivariate 1.6.8, MASS
7.3.66). Non-metric MDS used sklearn's smacof, which rescales the map: the
stress was right but every recovered distance was around 1 (city: 0.9 for
distances of 200 to 3,300 miles). A Procrustes comparison of the coordinates
scales both maps first, so it passed. These tests compare the recovered
distances, the coordinates and the printed summary in the data's units.

Tolerances: from the same start pyrsm's isoMDS is R's to the last bit
(``test_isomds_is_mass_isomds_to_the_last_bit``). The start, cmdscale, comes
from numpy's LAPACK, which differs from R's reference LAPACK by about 1e-12;
on the city data isoMDS stops near its stress floor (0.01%), where that moves
the recovered distances by up to 4e-5 (0.014 miles). Hence rtol 1e-4 below,
a factor of 10,000 tighter than the bug (a factor of about 2,000).
"""

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from conftest import mv_mat, mv_sign_align

from pyrsm.multivariate import mds
from pyrsm.multivariate._isomds import _Kruskal, isomds, isotonic_fit, vmmin

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "reference" / "radiant_multivariate"
DATA = ROOT / "examples" / "data" / "multivariate"

COLS = {
    "city": ("from", "to", "distance"),
    "city2": ("from", "to", "distance"),
    "tpbrands": ("id1", "id2", "dissimilarity"),
}
CASES = [
    f"{name}_{method}{dim}"
    for name in COLS
    for method in ("metric", "nonmetric")
    for dim in ("", "_3d")
] + ["city2_nonmetric_filter"]


def ref(case: str) -> dict:
    return json.loads((REF / f"mds_{case}.json").read_text())


def dataset(case: str) -> str:
    return case.split("_")[0]


def run(case: str, **kw) -> mds:
    r = ref(case)
    name = dataset(case)
    i1, i2, dis = COLS[name]
    df = pl.read_parquet(DATA / f"{name}.parquet")
    return mds(
        {name: df}, i1, i2, dis, method=r["method"], nr_dim=r["nr_dim"],
        data_filter=r["data_filter"], **kw,
    )


@pytest.fixture(scope="module")
def fitted():
    return {case: run(case) for case in CASES}


def lower(M: np.ndarray) -> np.ndarray:
    return M[np.tril_indices_from(M, -1)]


# -- the fixtures cover what the tests claim -----------------------------------


def test_fixtures_cover_city_city2_tpbrands_both_methods_both_dims():
    assert len(CASES) == 13
    for case in CASES:
        r = ref(case)
        assert r["case"] == case
        assert r["method"] == ("non-metric" if "nonmetric" in case else "metric")
        assert r["nr_dim"] == (3 if case.endswith("_3d") else 2)


# -- data and setup ----------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_original_distances_and_labels_match_radiant(fitted, case):
    r, m = ref(case), fitted[case]
    assert m.labels == r["labels"]
    # tpbrands holds averages: parquet and R agree to the last digit or two
    np.testing.assert_allclose(m.dis_mat, mv_mat(r["orig_dist"]), rtol=1e-12, atol=0)
    assert m.nobs == r["nobs"]


# -- the numbers -------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_recovered_distances_match_radiant(fitted, case):
    """The regression: same distances in the data's units, not rescaled."""
    expected = lower(mv_mat(ref(case)["recovered_dist"]))
    got = lower(fitted[case].recovered_dist())
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=0)


@pytest.mark.parametrize("case", CASES)
def test_stress_matches_radiant(fitted, case):
    # near the floor (city: 0.00006) the stopping point moves the stress by 1e-6
    assert fitted[case].stress == pytest.approx(ref(case)["stress"], rel=1e-6, abs=2e-6)


@pytest.mark.parametrize("case", CASES)
def test_coordinates_match_radiant_up_to_the_sign_of_each_dimension(fitted, case):
    expected = mv_mat(ref(case)["points"])
    got = mv_sign_align(fitted[case].points, expected)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-4 * np.abs(expected).max())


# -- the printed summary ------------------------------------------------------


def _block(lines: list[str], title: str) -> list[str]:
    start = next(i for i, line in enumerate(lines) if line.startswith(title)) + 1
    end = next(
        (i for i in range(start, len(lines)) if lines[i].endswith("data:") or lines[i].startswith("Coordinates")),
        len(lines),
    )
    return lines[start:end]


def _r_rows(lines: list[str], labels: list[str]) -> dict[str, list[float]]:
    """Rows of an R matrix print (wrapped in blocks; labels may have spaces)."""
    rows: dict[str, list[float]] = {}
    by_length = sorted(labels, key=len, reverse=True)
    for line in lines:
        if not line or line[0] == " ":
            continue  # a header line of column names
        label = next(lab for lab in by_length if line.startswith(lab))
        rows.setdefault(label, []).extend(float(x) for x in line[len(label):].split())
    return rows


def _py_rows(lines: list[str]) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {}
    for line in lines:
        if "┆" not in line or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("│ ").split("┆")]
        if cells[0] in ("label", "str"):  # the header and the dtype row
            continue
        rows[cells[0]] = [float(c) for c in cells[1:] if c]
    return rows


@pytest.mark.parametrize("case", CASES)
def test_printed_summary_matches_radiant(fitted, case, capsys):
    r = ref(case)
    fitted[case].summary(dec=2)
    py = capsys.readouterr().out.splitlines()
    R = r["summary"]
    for title in ("Original distance data:", "Recovered distance data:"):
        want = _r_rows(_block(R, title), r["labels"])
        got = _py_rows(_block(py, title))
        assert list(got) == list(want), title
        for label in want:
            # both round to 2 decimals (a last-digit tie), plus the 4e-5 above
            np.testing.assert_allclose(
                got[label], want[label], rtol=5e-5, atol=0.0101, err_msg=f"{title} {label}"
            )
    stress = lambda lines: next(l for l in lines if l.startswith("Stress:")).split()[1]  # noqa: E731
    assert float(stress(py)) == pytest.approx(float(stress(R)), abs=1e-3)
    obs = lambda lines: re.search(r"Observations: *([\d,]+)", "\n".join(lines)).group(1)  # noqa: E731
    assert obs(py).replace(",", "") == obs(R)


# -- scale guards that need no R ------------------------------------------------


@pytest.mark.parametrize("name", list(COLS))
@pytest.mark.parametrize("nr_dim", [2, 3])
def test_nonmetric_recovered_distances_are_on_the_data_scale(name, nr_dim):
    i1, i2, dis = COLS[name]
    df = pl.read_parquet(DATA / f"{name}.parquet")
    m = mds({name: df}, i1, i2, dis, method="non-metric", nr_dim=nr_dim)
    orig, rec = lower(m.dis_mat), lower(m.recovered_dist())
    assert 0.8 < rec.mean() / orig.mean() < 1.25
    assert 0.8 < rec.max() / orig.max() < 1.25
    # and on the scale of the metric map of the same data
    metric = mds({name: df}, i1, i2, dis, method="metric", nr_dim=nr_dim)
    assert 0.8 < rec.mean() / lower(metric.recovered_dist()).mean() < 1.25


@pytest.mark.parametrize("factor", [1 / 1000, 1.609344, 1000])
def test_nonmetric_follows_the_units_of_the_data(factor):
    """Miles to thousands of miles, kilometers or yards: the map stays in the
    units of the data and keeps the order of the distances."""
    df = pl.read_parquet(DATA / "city2.parquet").with_columns(pl.col("distance") * factor)
    m = mds({"city2": df}, "from", "to", "distance", method="non-metric")
    orig, rec = lower(m.dis_mat), lower(m.recovered_dist())
    assert 0.8 < rec.mean() / orig.mean() < 1.25
    from scipy.stats import spearmanr

    assert spearmanr(orig, rec).statistic > 0.99


@pytest.mark.parametrize("name", ["city", "city2"])
def test_nonmetric_city_maps_reproduce_the_miles(name):
    """Real distances: the non-metric map is close to the miles themselves."""
    i1, i2, dis = COLS[name]
    m = mds({name: pl.read_parquet(DATA / f"{name}.parquet")}, i1, i2, dis, method="non-metric")
    orig, rec = lower(m.dis_mat), lower(m.recovered_dist())
    assert np.median(np.abs(rec - orig) / orig) < 0.1


def test_the_seed_does_not_change_nonmetric_mds():
    """isoMDS starts from the classical solution, so there is nothing random."""
    a, b = run("tpbrands_nonmetric"), run("tpbrands_nonmetric", seed=99)
    assert np.array_equal(a.points, b.points) and a.stress == b.stress


def test_nonmetric_does_not_use_sklearn_smacof(monkeypatch):
    """smacof normalizes the map to distances around 1; it must not come back."""
    import sklearn.manifold

    def boom(*args, **kwargs):
        raise AssertionError("non-metric MDS called sklearn's smacof")

    monkeypatch.setattr(sklearn.manifold, "smacof", boom)
    run("city_nonmetric")


def test_nonmetric_refuses_a_zero_distance_between_two_objects():
    df = pl.read_parquet(DATA / "city.parquet").with_columns(
        pl.when((pl.col("from") == "Boston") & (pl.col("to") == "NY"))
        .then(0).otherwise(pl.col("distance")).alias("distance")
    )
    with pytest.raises(ValueError, match="zero or negative distance"):
        mds({"city": df}, "from", "to", "distance", method="non-metric")
    mds({"city": df}, "from", "to", "distance", method="metric")  # metric still runs


def test_plotted_points_are_the_nonmetric_coordinates():
    m = run("city2_nonmetric")
    g = m.plot()
    np.testing.assert_allclose(np.sort(g.data["x"].to_numpy()), np.sort(m.points[:, 0]))


# -- the ported internals --------------------------------------------------------


def _fixture(name: str) -> dict:
    return json.loads((REF / f"{name}.json").read_text())


def test_vmmin_reproduces_r_bfgs_on_rosenbrock():
    """R's optim(method = "BFGS") is vmmin: same path, counts and result."""
    fr = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2  # noqa: E731
    grr = lambda x: np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])  # noqa: E731
    r = _fixture("vmmin_rosenbrock")
    for run_ in r["runs"]:
        abstol = run_["abstol"] if run_.get("abstol") is not None else -np.inf
        par, value, fncount, grcount, fail = vmmin(
            np.array(r["start"]), fr, grr, maxit=run_["maxit"], abstol=abstol, reltol=run_["reltol"]
        )
        np.testing.assert_allclose(par, run_["par"], rtol=1e-12, atol=1e-15)
        assert value == pytest.approx(run_["value"], rel=1e-9, abs=1e-20)
        assert (fncount, grcount, fail) == (run_["fncount"], run_["grcount"], run_["convergence"])


ISOMDS = sorted(f.stem for f in REF.glob("isomds_*.json"))


def test_isomds_fixtures_cover_radiants_setting_and_the_stopping_rules():
    assert {"isomds_city_radiant", "isomds_city2_radiant", "isomds_city_3d_radiant",
            "isomds_city2_3d_radiant", "isomds_tpbrands_radiant"} <= set(ISOMDS)
    caps = {_fixture(name)["maxit"] for name in ISOMDS}
    assert {1, 5, 50, 200} <= caps


@pytest.mark.parametrize("name", ISOMDS)
def test_isomds_is_mass_isomds_to_the_last_bit(name):
    """Same start, iteration cap and tolerance as MASS::isoMDS: the same map
    and stress (17-digit fixtures; sums in the C code's order)."""
    r = _fixture(name)
    points, stress = isomds(mv_mat(r["dis"]), mv_mat(r["init"]), maxit=r["maxit"], tol=r["tol"])
    expected = mv_mat(r["points"])
    np.testing.assert_allclose(points, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())
    assert stress == pytest.approx(r["stress_pct"], rel=1e-12)


def test_isomds_keeps_the_scale_of_its_start():
    r = _fixture("isomds_tpbrands_maxit200")
    init = mv_mat(r["init"])
    for factor in (0.5, 2.0):
        points, _ = isomds(mv_mat(r["dis"]) * factor, init * factor, maxit=r["maxit"], tol=r["tol"])
        assert 0.5 < np.abs(points).max() / np.abs(init * factor).max() < 2


@pytest.mark.parametrize("seed", range(6))
def test_isotonic_fit_is_the_least_squares_monotone_fit(seed):
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(seed)
    y = rng.normal(size=40).cumsum() + rng.normal(scale=3, size=40)
    if seed % 2:
        y[10:15] = y[10]  # ties
    expected = IsotonicRegression().fit_transform(np.arange(y.size), y)
    np.testing.assert_allclose(isotonic_fit(y), expected, atol=1e-12)
    assert np.all(np.diff(isotonic_fit(y)) >= -1e-12)


def test_isotonic_fit_leaves_an_increasing_vector_alone():
    y = np.array([0.5, 1.0, 1.0, 2.5, 7.0])
    np.testing.assert_array_equal(isotonic_fit(y), y)


@pytest.mark.parametrize("case", ["tpbrands_nonmetric", "city2_nonmetric_3d"])
def test_stress_gradient_matches_finite_differences(case):
    r = ref(case)
    D = mv_mat(r["orig_dist"])
    k = r["nr_dim"]
    obj = _Kruskal(D, k)
    x = mds._cmdscale(D, k).ravel(order="F")
    x = x + np.random.default_rng(1).normal(scale=0.01 * np.abs(x).max(), size=x.size)
    g = obj.gradient(x)
    h = 1e-6 * np.abs(x).max()
    num = np.array([
        (obj.stress(x + h * e) - obj.stress(x - h * e)) / (2 * h) for e in np.eye(x.size)
    ])
    np.testing.assert_allclose(g, num, rtol=1e-4, atol=1e-6 * np.abs(num).max())
