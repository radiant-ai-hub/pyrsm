# Plan to Port `radiant.multivariate` to `pyrsm`

This plan is intended as a handoff for implementing a clean, testable Python port of the R code in `../radiant.multivariate`. The target is not a Streamlit-first implementation. The target is a reusable `pyrsm.multivariate` API that can be validated numerically against `radiant.multivariate`, used from notebooks, and later wired into UI layers.

## Source Inventory

Relevant R package:

- R source: `../radiant.multivariate/R`
- R exported functions: `pre_factor`, `full_factor`, `hclus`, `kclus`, `mds`, `prmap`, `conjoint`
- R documentation and app examples: `/home/vnijs/gh/docs/multivariate/app`
- Figures for expected outputs and teaching examples: `/home/vnijs/gh/docs/multivariate/figures_multivariate`
- Example datasets: `/home/vnijs/gh/pyrsm/examples/data/multivariate`
- Draft Python modules: `../pyrsm_streamlit/modules`

Current `pyrsm` state:

- `pyrsm/multivariate/factor.py` is essentially empty.
- `pyrsm/multivariate/__init__.py` only lazy-loads `factor`.
- `pyrsm_streamlit/modules` contains useful drafts, but they are UI-oriented and sometimes depend on packages that are not currently `pyrsm` dependencies.
- `pyrsm.utils.polychoric_corr` and `pyrsm.utils.polychoric_matrix` already implement ordinal categorical correlations.
- `pyrsm.design.doe` already uses `polychoric_matrix` for design correlations. Reuse this path for factor-analysis cases with ordered categorical variables rather than adding a separate polychoric implementation.
- `pyrsm.basics.correlation` also exposes `method="polychoric"` and uses the same utility.

## Multivariate Menu Inventory

| Menu entry | R function | Documentation | Main datasets | Core outputs | Store/predict actions |
| --- | --- | --- | --- | --- | --- |
| Multivariate > Maps > `(Dis)similarity` | `mds` | `mds.md` | `city`, `city2`, `tpbrands` | original distance matrix, recovered distances, coordinates, stress, map plot | none in R |
| Multivariate > Maps > `Attributes` | `prmap` | `prmap.md` | `computer`, `retailers` | brand scores, attribute loadings, optional preference correlations, fit measures, map plot | none in R |
| Multivariate > Factor > `Pre-factor` | `pre_factor` | `pre_factor.md` | `shopping`, `toothpaste` | correlation matrix, Bartlett test, KMO/MSA, variable Rsq, eigenvalues, scree/change plots | none in R |
| Multivariate > Factor > `Factor` | `full_factor` | `full_factor.md` | `shopping`, `toothpaste` | loadings, communalities, eigenvalues, variance explained, factor scores, loadings/respondent plots | store factor scores |
| Multivariate > Cluster > `Hierarchical` | `hclus` | `hclus.md` | `shopping`, `toothpaste` | distance settings, linkage object, scree/change/dendrogram plots | store cluster assignments |
| Multivariate > Cluster > `K-clustering` | `kclus` | `kclus.md` | `shopping`, `toothpaste` | cluster sizes, cluster means/modes, within/between heterogeneity, density/bar/scatter plots | store cluster assignments |
| Multivariate > `Conjoint` | `conjoint` | `conjoint.md` | `mp3`, `carpet`, `movie` | part-worths, importance weights, regression coefficients, diagnostics, PW/IW plots | store PW/IW tables and predictions |

## Proposed Package Structure

Add real modules under `pyrsm/multivariate`:

```text
pyrsm/multivariate/
  __init__.py
  _utils.py
  _correlation.py
  pre_factor.py
  full_factor.py
  cluster.py
  maps.py
  conjoint.py
```

Public API:

```python
from pyrsm.multivariate import (
    pre_factor,
    full_factor,
    hclus,
    kclus,
    mds,
    prmap,
    conjoint,
)
```

Keep the R names as public names because the documentation, figures, and user mental model already use them. Internally, Python classes or dataclasses can hold results, but users should be able to call `rsm.multivariate.pre_factor(...)` and get an object with `.summary()` and `.plot()` methods, consistent with the rest of `pyrsm`.

## Shared Implementation Rules

Use these conventions across all modules:

- Accept Polars and pandas input, then convert through `pyrsm.utils.check_dataframe` or an equivalent shared helper.
- Preserve Polars `Enum` order when converting categorical variables. This is critical for conjoint base levels, ordinal correlations, and stable output.
- Support Radiant-style variable ranges such as `"v1:v6"` and lists like `["v1", "v2", "v3"]`.
- Store raw numeric tables on result objects for tests. Printed summaries should be secondary.
- Prefer `numpy`, `scipy`, `statsmodels`, `polars`, `pandas`, and existing `pyrsm` helpers. Avoid new hard dependencies unless parity requires them.
- Use `plotnine` where it gives ggplot-like parity. Matplotlib is acceptable for dendrograms if SciPy dendrogram support makes that cleaner.
- Keep UI state, Streamlit widgets, file upload logic, and app routing out of `pyrsm.multivariate`.

## Correlation Strategy for Factor Analysis

The R package uses `polycor::hetcor` for factor-analysis and perceptual-map paths when categorical variables are included and `hcor=TRUE`.

The Python plan should reuse existing `pyrsm` correlation code:

- For all ordered categorical variables, call `pyrsm.utils.polychoric_matrix`. This is already used by `pyrsm.design.doe` for design-menu correlations and should be the canonical implementation.
- For numeric-only variables, use Pearson correlation on standardized numeric columns.
- For mixed numeric plus ordered categorical variables, add a shared helper in `pyrsm/multivariate/_correlation.py` or move a general helper into `pyrsm.utils`:

```python
heterogeneous_corr(data, vars, ordered=None) -> np.ndarray
```

Pairwise rules for `heterogeneous_corr`:

- numeric/numeric: Pearson correlation
- ordinal/ordinal: existing `polychoric_corr`
- numeric/binary ordinal: point-biserial correlation, or polyserial if implemented
- numeric/multi-level ordinal: polyserial correlation for closer `polycor::hetcor` parity

Implementation recommendation:

1. First implement and test all-ordinal factor-analysis parity using existing `polychoric_matrix`.
2. Add `polyserial_corr` only when mixed numeric plus ordinal fixtures show it is needed for `radiant.multivariate` parity.
3. Use one shared correlation helper from `pre_factor`, `full_factor`, and `prmap`.
4. Document that categorical variables are treated as ordered when `hcor=True`, matching the note already shown in DOE summaries.

Do not copy the simplified `_hetcor_python` from `../pyrsm_streamlit/modules/maps/prmap.py` as-is. It is useful as a sketch, but the production implementation should be centralized, tested, and based on the existing `polychoric_corr` utility.

## Result Object Pattern

Each analysis should return an object with:

- input metadata: dataset name, selected variables, settings, filter if supported
- R-parity attributes where practical, for example `cmat`, `pre_r2`, `pre_kmo`, `pre_eigen`, `fres`, `floadings`, `scores`, `hc_out`, `km_out`, `stress`, `coordinates`, `part_worths`, `importance_weights`
- `summary(...)` method that prints Radiant-like text and returns or exposes table objects
- `plot(...)` method returning plot objects
- `store(...)` method only for tools that store results in R
- `predict(...)` method only for conjoint

Tests should assert the numeric attributes, not screen-formatted text, except for a few smoke tests that confirm summaries render.

## Module-Level Implementation Plan

### 1. Pre-Factor Analysis: `pre_factor`

R behavior to match:

- Inputs: `dataset`, `vars`, `hcor=False`, optional data filter.
- Builds a correlation matrix from selected variables.
- With `hcor=True`, R calls `polycor::hetcor`.
- Calculates:
  - Bartlett test of sphericity
  - KMO/MSA overall and per variable
  - eigenvalues of the correlation matrix
  - per-variable Rsq: `1 - 1 / diag(solve(cmat))` when the determinant is positive
- Plots:
  - scree plot with eigenvalue line at 1
  - eigenvalue-change plot

Python details:

- Use `heterogeneous_corr(..., hcor=True)` only when needed.
- For all ordered categorical variables, use `polychoric_matrix`.
- Implement Bartlett directly:

```text
chi_square = -(n - 1 - (2p + 5) / 6) * log(det(R))
df = p * (p - 1) / 2
p_value = scipy.stats.chi2.sf(chi_square, df)
```

- Implement KMO directly from the inverse correlation matrix, avoiding a new `factor_analyzer` dependency.
- Add tests for numeric-only and ordered-categorical cases.

### 2. Full Factor Analysis: `full_factor`

R behavior to match:

- Inputs: `dataset`, `vars`, `method="PCA"`, `hcor=False`, `nr_fact=1`, `rotation="varimax"`.
- Supports PCA and maximum likelihood factor analysis.
- Supports rotations such as none and varimax in the documented examples. R can also use rotations from `GPArotation`.
- Computes loadings, communalities, variance explained, and factor scores.
- Stores factor scores back to the original data.
- Plots attributes/loadings and respondents when at least two factors are requested.

Python details:

- Implement PCA on the correlation matrix first.
- Standardize numeric input using sample standard deviation (`ddof=1`) to match R's `scale`.
- For PCA scores, match the R formula:

```text
cscm = loadings @ inv(loadings.T @ loadings)
scores = standardized_data @ cscm
```

- Implement varimax locally or use a small internal utility. Keep sign orientation deterministic by aligning each component to the variable with the largest absolute loading.
- Reuse `heterogeneous_corr` for `hcor=True`.
- For ML factor analysis, evaluate `statsmodels.multivariate.factor.Factor` before adding a dependency. If exact parity is hard, implement PCA first and mark ML as phase 2 with reference fixtures.

### 3. Hierarchical Clustering: `hclus`

R behavior to match:

- Inputs: `dataset`, `vars`, `labels="none"`, `distance="sq.euclidian"`, `method="ward.D"`, `max_cases=5000`, `standardize=True`.
- If categorical variables are present and distance is not Gower, R switches to Gower.
- Numeric variables are standardized with R `scale`.
- Squared Euclidean distance is `dist(..., method="euclidean") ^ 2`.
- Uses R `hclust`.
- Plots scree, change, and dendrogram.
- Stores cluster assignments from `cutree`.

Python details:

- Use SciPy distance functions for numeric distances.
- Add a Gower implementation or optional dependency only after deciding whether mixed-type clustering is in phase 1.
- Be careful with `ward.D`: SciPy's `ward` is closer to R `ward.D2`, not necessarily R `ward.D`. Exact parity may require implementing R-compatible Lance-Williams updates for `ward.D`.
- Phase 1 should support the documented numeric examples. Phase 2 should add mixed-type Gower parity.

### 4. K-Clustering: `kclus`

R behavior to match:

- Inputs: `dataset`, `vars`, `fun="kmeans"`, `hc_init=True`, `distance="sq.euclidian"`, `method="ward.D"`, `seed=1234`, `nr_clus=2`, `standardize=True`, `lambda=NULL`.
- `kmeans` drops factor variables and warns.
- `kproto` supports mixed numeric/categorical data via `clustMixType::kproto`.
- Optional hierarchical clustering initialization.
- Outputs cluster sizes, original-scale means/modes, within-cluster heterogeneity, and between/total heterogeneity.
- Plots density, bar, and scatter views.
- Stores cluster assignments.

Python details:

- Build on `hclus` for hierarchical initialization.
- For numeric-only `kmeans`, decide whether scikit-learn parity is sufficient. R uses Hartigan-Wong by default, while scikit-learn uses Lloyd/Elkan. With fixed centers results may match common teaching examples, but this needs fixture testing.
- If exact parity is required, implement or vendor a small Hartigan-Wong-compatible routine.
- For mixed data, decide between adding an optional `kmodes` dependency for K-Prototypes or implementing the small subset needed for Radiant parity.

### 5. MDS From Dissimilarities: `mds`

R behavior to match:

- Inputs: `dataset`, `id1`, `id2`, `dis`, `method="metric"`, `nr_dim=2`, `seed=1234`.
- Builds a symmetric distance matrix from lower-triangle or lower-triangle-plus-diagonal data.
- Metric MDS uses `cmdscale`.
- Nonmetric MDS uses `MASS::isoMDS`.
- Stress for metric MDS is:

```text
sqrt(sum((dist(points) - d)^2) / sum(d^2))
```

- Plots dimension pairs and supports axis flipping.

Python details:

- Implement metric MDS exactly through double-centering and eigen-decomposition.
- Align coordinates only up to sign and rotation in tests unless exact R orientation is explicitly reproduced.
- For nonmetric MDS, scikit-learn SMACOF will not exactly match `MASS::isoMDS`. Treat nonmetric parity as phase 2 unless a custom `isoMDS` implementation is added.

### 6. Perceptual Maps From Attributes: `prmap`

R behavior to match:

- Inputs: `dataset`, `brand`, `attr`, `pref=""`, `nr_dim=2`, `hcor=False`.
- Runs a PCA/principal-components map from attribute correlations with varimax rotation.
- Computes brand-level factor scores by averaging respondent/item scores by brand.
- Computes optional preference correlations with map dimensions.
- Shows brand scores, attribute loadings, preference correlations, fit measures, and communalities.
- Plots brands, attribute arrows, and preference arrows.

Python details:

- Reuse `full_factor` PCA internals instead of duplicating factor-analysis code.
- Reuse `heterogeneous_corr` for `hcor=True`.
- For preference variables that are ordered categorical, use the same categorical-correlation strategy.
- Keep plotting data exposed for tests: brand coordinates, attribute arrows, preference arrows.

### 7. Conjoint Analysis: `conjoint`

R behavior to match:

- Inputs: `dataset`, `rvar`, `evar`, `int=""`, `by="none"`, `reverse=False`.
- Fits OLS models per full dataset or by-level.
- Treats factor/string/logical explanatory variables as categorical with the first level as base.
- `reverse=True` transforms the response as `(max(x) + 1) - x`.
- Builds part-worth tables and importance weights.
- Supports interactions, coefficient diagnostics, plots, prediction intervals, and store actions.

Python details:

- Use `statsmodels` and `patsy` for OLS.
- Build formulas with treatment coding and explicit first-level references from Polars `Enum` or observed order.
- Reproduce R coefficient labels such as `Memory|6GB`.
- Implement `predict(...)` through `statsmodels.get_prediction` for confidence and prediction intervals.
- Implement VIF diagnostics with `statsmodels.stats.outliers_influence.variance_inflation_factor`.

## Reference Fixture Strategy

Create an R fixture generator:

```text
scripts/generate_multivariate_reference.R
```

Responsibilities:

- Load `radiant.multivariate` and `radiant.data`.
- Load all parquet datasets from `/home/vnijs/gh/pyrsm/examples/data/multivariate`.
- Run representative examples for every menu entry.
- Write numeric outputs to `tests/reference/radiant_multivariate/` as CSV, JSON, or parquet.
- Write `sessionInfo()` and package versions beside the fixtures.

Recommended fixture cases:

- `pre_factor(shopping, "v1:v6")`
- `pre_factor(toothpaste, "v1:v6")`
- `pre_factor(...)` with ordered categorical variables and `hcor=TRUE`, using a small fixture that can validate reuse of `polychoric_matrix`
- `full_factor(shopping, "v1:v6", nr_fact=2, rotation="varimax")`
- `full_factor(toothpaste, "v1:v6", nr_fact=2, rotation="varimax")`
- `full_factor(...)` with ordered categorical variables and `hcor=TRUE`
- `hclus(shopping, "v1:v6")`
- `hclus(toothpaste, "v1:v6", labels="id")`
- `kclus(shopping, "v1:v6", nr_clus=2, seed=1234)`
- `kclus(toothpaste, "v1:v6", nr_clus=3, seed=1234)`
- `mds(city, id1="from", id2="to", dis="distance", method="metric")`
- `mds(tpbrands, id1="id1", id2="id2", dis="dissimilarity", method="metric")`
- `mds(tpbrands, id1="id1", id2="id2", dis="dissimilarity", method="nonmetric")`
- `prmap(computer, brand="brand", attr="high_end:business")`
- `prmap(retailers, brand="retailer", attr=<attribute vars>, pref="segment1:segment2")`
- `conjoint(mp3, rvar="Rating", evar="Memory:Shape")`
- `conjoint(carpet, rvar="ranking", evar="design:money_back", reverse=True)` if this matches docs
- `conjoint(movie, ...)` for a second conjoint teaching example
- `conjoint(..., by=<group var>)` using a small synthetic fixture if no bundled dataset has a natural `by`

Known R expected values from existing package tests should be added to Python tests:

- `mds(city, method="metric")` coordinates, compared up to sign.
- `prmap(computer, brand="brand", attr="high_end:business")` brand scores.
- `hclus(shopping, "v1:v6")` merge heights.
- `kclus(shopping, "v1:v6", nr_clus=2)` cluster means.
- `conjoint(mp3, rvar="Rating", evar="Memory:Shape")` part-worths and importance weights.

Comparison rules:

- Use tight tolerances (`1e-8`) for deterministic matrix algebra such as correlations, eigenvalues, metric MDS distances, part-worths, and PCA quantities when orientation is aligned.
- Use looser tolerances (`1e-6` to `1e-5`) for rotations, ML factor analysis, nonmetric MDS, and optimization-based methods.
- Compare PCA/factor loadings and MDS coordinates after deterministic sign alignment.
- Compare MDS recovered distances and stress even when coordinates differ by sign or rotation.
- Compare clustering assignments through cluster summaries when label switching is possible.
- Plot tests should assert plot data tables and required labels/arrows, not pixel-perfect image matches.

## Test Layout

Add focused tests:

```text
tests/test_multivariate_pre_factor.py
tests/test_multivariate_full_factor.py
tests/test_multivariate_cluster.py
tests/test_multivariate_maps.py
tests/test_multivariate_conjoint.py
tests/test_multivariate_reference.py
```

Each test file should include:

- API construction tests
- numeric parity tests against R fixtures
- summary smoke tests
- plot object smoke tests
- store/predict tests where relevant

Validation commands:

```bash
uv run pytest tests/test_multivariate_pre_factor.py
uv run pytest tests/test_multivariate_full_factor.py
uv run pytest tests/test_multivariate_cluster.py
uv run pytest tests/test_multivariate_maps.py
uv run pytest tests/test_multivariate_conjoint.py
uv run pytest tests/test_multivariate_reference.py
uv run pytest tests
```

## Notebook and Documentation Plan

After the API and tests are stable, add or expand teaching notebooks under `examples/qmd/multivariate` and regenerate `.ipynb` files.

Recommended notebooks:

- `multivariate-factor-analysis.qmd`
  - pre-factor diagnostics
  - choosing the number of factors
  - reading loadings and communalities
  - factor scores and stored scores
  - ordered categorical factor analysis using the shared polychoric path
- `multivariate-cluster-analysis.qmd`
  - hierarchical clustering
  - reading dendrograms and scree/change plots
  - choosing cluster counts
  - k-means cluster profiles
  - cluster assignment storage
- `multivariate-maps.qmd`
  - metric MDS from dissimilarity data
  - perceptual maps from attributes
  - interpreting brand points, attribute arrows, and preference arrows
- `multivariate-conjoint.qmd`
  - part-worths
  - importance weights
  - base utility
  - prediction for new product profiles
  - interactions and by-group models if implemented

Use the documentation figures in `/home/vnijs/gh/docs/multivariate/figures_multivariate` as a checklist for examples to reproduce, but generate the Python plots directly rather than embedding the old images as primary output.

## Dependency Decisions

Current dependencies already include the main scientific stack needed for phase 1:

- `numpy`
- `scipy`
- `pandas`
- `polars`
- `statsmodels`
- `scikit-learn`
- `plotnine`

Avoid adding hard dependencies initially. Reassess only for these specific needs:

- `factor_analyzer`: possibly useful, but avoid if KMO, Bartlett, PCA, and rotation can be implemented cleanly.
- `gower`: optional if a compact internal Gower distance implementation is not enough.
- `kmodes`: optional for K-Prototypes if mixed clustering parity is required.
- `fastcluster`: optional if it improves R `hclust` parity, but verify `ward.D` semantics first.

If any are added, prefer a `multivariate` optional extra rather than forcing all `pyrsm` users to install them.

## Phased Work Plan

### Phase 0: Reference Harness and Scaffolding

- Add package modules and public imports.
- Add the R fixture generator.
- Generate and commit reference outputs.
- Add shared dataset-loading helpers for tests.
- Add shared variable-range parsing and standardization helpers.

Deliverable: empty or minimal result objects can load data, parse variables, and run fixture generation.

### Phase 1: Correlation and Pre-Factor

- Add shared correlation helper.
- Reuse `polychoric_matrix` for ordered categorical matrices.
- Implement Pearson and all-ordinal `hcor=True` paths.
- Implement Bartlett, KMO, eigenvalues, Rsq, summaries, and plots.
- Add reference tests.

Deliverable: `pre_factor` matches R for numeric and all-ordinal categorical cases.

### Phase 2: PCA Factor Analysis and Attribute Maps

- Implement PCA factor analysis.
- Implement varimax and deterministic sign alignment.
- Implement factor scores and store method.
- Implement `prmap` on top of the PCA internals.
- Add `full_factor` and `prmap` reference tests.

Deliverable: PCA-based `full_factor` and `prmap` match R examples.

### Phase 3: Metric MDS

- Implement distance-matrix construction.
- Implement metric MDS via double-centering.
- Implement stress, summary, axis flipping, and plots.
- Add reference tests for `city` and `tpbrands`.

Deliverable: metric MDS examples match R up to orientation.

### Phase 4: Numeric Clustering

- Implement numeric `hclus` and `kclus`.
- Verify R-compatible standardization and squared Euclidean distances.
- Resolve or document `ward.D` differences.
- Implement cluster summary tables and store actions.
- Add tests for `shopping` and `toothpaste`.

Deliverable: documented numeric clustering examples match R summaries.

### Phase 5: Conjoint

- Implement OLS models with categorical treatment coding.
- Implement part-worth and importance-weight tables.
- Implement interactions, reverse response, by-group models, plots, prediction, and store actions.
- Add tests for `mp3`, `carpet`, and `movie`.

Deliverable: conjoint part-worths, importance weights, and predictions match R.

### Phase 6: Advanced Parity

- Add mixed numeric plus ordinal `heterogeneous_corr` with polyserial support if required by fixtures.
- Add ML factor analysis if still needed.
- Add additional rotations beyond none and varimax if required by docs.
- Add Gower distance and K-Prototypes for mixed clustering.
- Add nonmetric MDS parity or clearly document any accepted numerical differences.

Deliverable: advanced menu options either match R or have explicit test-backed limitations.

### Phase 7: Teaching Examples and Cleanup

- Expand multivariate qmd notebooks.
- Regenerate ipynb notebooks.
- Add API docs or README references.
- Run full tests.
- Review generated examples as teaching material, not just API demos.

Deliverable: students can learn each multivariate method from examples, and every menu entry has tested Python coverage.

## Risks and Decisions for Claude Code

Key risks:

- Exact R `ward.D` parity is not guaranteed with SciPy linkage.
- Exact R `kmeans` parity is not guaranteed with scikit-learn because R defaults to Hartigan-Wong.
- `MASS::isoMDS` and scikit-learn SMACOF are not the same nonmetric MDS algorithm.
- `polycor::hetcor` mixed numeric plus ordinal parity may require a true polyserial implementation.
- PCA, factor-analysis, and MDS signs are arbitrary. Tests must align orientation before comparison.

Recommended decision order:

1. Build the R fixtures first.
2. Implement the simplest dependency-free Python path.
3. Compare against fixtures.
4. Add optional dependencies or custom algorithms only where fixture comparisons show they are necessary.

## Definition of Done

The port is done when:

- Every multivariate menu entry has a public `pyrsm.multivariate` function.
- Numeric outputs for documented examples are tested against `radiant.multivariate`.
- Ordered categorical factor-analysis paths reuse the existing `polychoric_matrix` implementation from `pyrsm.utils`.
- Store and predict actions work where the R package offers them.
- Example notebooks teach the concepts and demonstrate the Python API.
- Full test suite passes with:

```bash
uv run pytest tests
```

