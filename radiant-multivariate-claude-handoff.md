# Claude Code Handoff: Complete `radiant.multivariate` Parity in `pyrsm`

Use this as the implementation brief. The current port has good scaffolding and passing tests for a subset, but the goal is full functionality parity with `radiant.multivariate`, not just the subset that is already tested.

## Goal

Replicate the functionality in `../radiant.multivariate` in Python under `pyrsm.multivariate`, with clean, testable APIs, one source file per key Radiant tool, Radiant-equivalent plots, expanded reference tests, and seven teaching notebooks in `examples/multivariate`.

The seven key tools are:

1. `pre_factor`
2. `full_factor`
3. `hclus`
4. `kclus`
5. `mds`
6. `prmap`
7. `conjoint`

## Current State

Current Python files:

- `pyrsm/multivariate/pre_factor.py`
- `pyrsm/multivariate/full_factor.py`
- `pyrsm/multivariate/cluster.py`
- `pyrsm/multivariate/maps.py`
- `pyrsm/multivariate/conjoint.py`
- `pyrsm/multivariate/_utils.py`
- `pyrsm/multivariate/_correlation.py`
- `pyrsm/multivariate/__init__.py`

Current reference fixtures and tests:

- `scripts/generate_multivariate_reference.R`
- `tests/reference/radiant_multivariate/`
- `tests/test_multivariate_pre_factor.py`
- `tests/test_multivariate_full_factor.py`
- `tests/test_multivariate_cluster.py`
- `tests/test_multivariate_maps.py`
- `tests/test_multivariate_conjoint.py`
- `tests/test_multivariate_reference.py`

Current multivariate test command passes:

```bash
uv run pytest tests/test_multivariate_pre_factor.py tests/test_multivariate_full_factor.py tests/test_multivariate_cluster.py tests/test_multivariate_maps.py tests/test_multivariate_conjoint.py tests/test_multivariate_reference.py
```

At the time of this handoff: `65 passed`.

Do not confuse those passing tests with full parity. Some tests currently encode missing functionality as expected behavior, for example:

- `full_factor(method="ml")` raises `NotImplementedError`.
- `hclus` with categorical variables raises `NotImplementedError`.
- nonmetric MDS is only checked with a loose stress tolerance because Python currently uses SMACOF rather than `MASS::isoMDS`.

These should be replaced with positive parity tests as implementation catches up.

## Source References

R package source:

- `../radiant.multivariate/R/pre_factor.R`
- `../radiant.multivariate/R/full_factor.R`
- `../radiant.multivariate/R/hclus.R`
- `../radiant.multivariate/R/kclus.R`
- `../radiant.multivariate/R/mds.R`
- `../radiant.multivariate/R/prmap.R`
- `../radiant.multivariate/R/conjoint.R`

Documentation text to include in notebooks:

- `/home/vnijs/gh/docs/multivariate/app/pre_factor.md`
- `/home/vnijs/gh/docs/multivariate/app/full_factor.md`
- `/home/vnijs/gh/docs/multivariate/app/hclus.md`
- `/home/vnijs/gh/docs/multivariate/app/kclus.md`
- `/home/vnijs/gh/docs/multivariate/app/mds.md`
- `/home/vnijs/gh/docs/multivariate/app/prmap.md`
- `/home/vnijs/gh/docs/multivariate/app/conjoint.md`

Figures referenced by the docs:

- `/home/vnijs/gh/docs/multivariate/figures_multivariate`

Datasets:

- `/home/vnijs/gh/pyrsm/examples/data/multivariate`

Drafts that may be useful but should not be copied directly:

- `../pyrsm_streamlit/modules`

## Non-Negotiable Requirements

1. Use one Python implementation file per key Radiant tool:
   - `pyrsm/multivariate/pre_factor.py`
   - `pyrsm/multivariate/full_factor.py`
   - `pyrsm/multivariate/hclus.py`
   - `pyrsm/multivariate/kclus.py`
   - `pyrsm/multivariate/mds.py`
   - `pyrsm/multivariate/prmap.py`
   - `pyrsm/multivariate/conjoint.py`

2. Keep shared helpers in separate private modules:
   - `_utils.py`
   - `_correlation.py`
   - add `_plotting.py` if useful
   - add `_labels.py` if useful for label placement

3. Preserve the public API:

```python
import pyrsm as rsm

rsm.multivariate.pre_factor(...)
rsm.multivariate.full_factor(...)
rsm.multivariate.hclus(...)
rsm.multivariate.kclus(...)
rsm.multivariate.mds(...)
rsm.multivariate.prmap(...)
rsm.multivariate.conjoint(...)
```

4. Add seven teaching notebooks, not four:
   - `examples/qmd/multivariate/multivariate-pre-factor.ipynb` or equivalent qmd/ipynb pair
   - `examples/qmd/multivariate/multivariate-full-factor.ipynb` or equivalent qmd/ipynb pair
   - `examples/qmd/multivariate/multivariate-hclus.ipynb` or equivalent qmd/ipynb pair
   - `examples/qmd/multivariate/multivariate-kclus.ipynb` or equivalent qmd/ipynb pair
   - `examples/qmd/multivariate/multivariate-mds.ipynb` or equivalent qmd/ipynb pair
   - `examples/qmd/multivariate/multivariate-prmap.ipynb` or equivalent qmd/ipynb pair
   - `examples/qmd/multivariate/multivariate-conjoint.ipynb` or equivalent qmd/ipynb pair

   Exact filenames can be adjusted to match repo conventions, but there must be one notebook per key tool.

5. Add the full text from each matching docs page into the matching notebook. Do not merely summarize the docs. Preserve the teaching explanations and exercises from `/home/vnijs/gh/docs/multivariate/app/*.md`, then add Python examples around that text.

6. Brand maps and factor maps must use square, origin-centered axes by default, matching Radiant:
   - compute `lim = max(abs(all plotted x/y coordinates))`
   - use `xlim=(-lim, lim)` and `ylim=(-lim, lim)`
   - use a fixed aspect ratio
   - draw horizontal and vertical zero lines

7. Labels attached to points and arrows need a much better approach than fixed `geom_text(..., nudge_y=...)`.
   - Radiant uses `ggrepel`.
   - In Python, implement a reusable label-placement helper or use a suitable dependency if already available.
   - Labels should remain visually attached to their point or arrow endpoint.
   - Arrow labels should be placed near the arrow endpoint and should not sit directly on top of arrowheads.

## Required File Split

Current problem:

- `pyrsm/multivariate/cluster.py` contains both `hclus` and `kclus`.
- `pyrsm/multivariate/maps.py` contains both `mds` and `prmap`.

Target:

```text
pyrsm/multivariate/
  __init__.py
  _correlation.py
  _plotting.py
  _utils.py
  conjoint.py
  full_factor.py
  hclus.py
  kclus.py
  mds.py
  pre_factor.py
  prmap.py
```

Implementation steps:

1. Move `hclus` code from `cluster.py` into `hclus.py`.
2. Move `kclus` code from `cluster.py` into `kclus.py`.
3. Move `mds` code from `maps.py` into `mds.py`.
4. Move `prmap` code from `maps.py` into `prmap.py`.
5. Delete `cluster.py` and `maps.py` after imports are updated, unless a temporary compatibility shim is needed.
6. Update `pyrsm/multivariate/__init__.py` exports:

```python
_EXPORTS = {
    "pre_factor": "pyrsm.multivariate.pre_factor",
    "kmo": "pyrsm.multivariate.pre_factor",
    "full_factor": "pyrsm.multivariate.full_factor",
    "clean_loadings": "pyrsm.multivariate.full_factor",
    "hclus": "pyrsm.multivariate.hclus",
    "kclus": "pyrsm.multivariate.kclus",
    "mds": "pyrsm.multivariate.mds",
    "prmap": "pyrsm.multivariate.prmap",
    "conjoint": "pyrsm.multivariate.conjoint",
}
```

7. Update tests and imports to point at the new module files.

## Shared Plotting and Labeling Work

Add shared plotting helpers before working on individual plot functions.

Recommended file:

- `pyrsm/multivariate/_plotting.py`

Recommended helpers:

```python
def symmetric_limit(*arrays, pad=1.05) -> float:
    """Return max absolute coordinate times pad."""

def square_origin_limits(x, y, *more_xy, pad=1.05) -> tuple[float, float]:
    """Return (-lim, lim) for both axes."""

def label_positions(points, labels, anchors=None, seed=1234, min_sep=0.04):
    """Return adjusted label positions in data coordinates."""

def arrow_label_positions(x, y, scale=1.08):
    """Return label positions just beyond arrow endpoints."""
```

Plot requirements:

- MDS maps: square, origin-centered, all dimension pairs, better labels.
- PR maps: square, origin-centered, brand labels, attribute labels, preference labels, arrows with heads.
- Full factor maps: square, origin-centered, attribute labels, respondent points when requested.
- K-cluster plots: support density, bar, and scatter, with categorical equivalents.

Testing approach:

- Do not rely on pixel-perfect plots.
- Test that plot data includes adjusted label positions.
- Test that x/y limits are symmetric around zero.
- Test that requested components are included in the plot data.
- Smoke test that plot objects render without exceptions.

## Tool-Specific Requirements

### 1. `pre_factor`

Current status:

- Numeric PCA diagnostics are implemented and tested.
- Polychoric all-ordinal path is implemented and tested against generated fixtures.

Missing or incomplete:

- `data_filter` support.
- No-variation checks matching R behavior.
- Full `polycor::hetcor` fallback semantics when heterogeneous correlation fails.
- Date variables should be treated as numeric, matching R.
- Plot cutoff behavior should match Radiant more closely.

R source references:

- `../radiant.multivariate/R/pre_factor.R`

Implementation tasks:

1. Add `data_filter=""` parameter.
2. Apply filter before selecting variables, using existing `pyrsm` filtering conventions if available.
3. Add no-variation checks and return/raise a useful error matching Radiant's message.
4. Ensure date-like columns are converted to numeric.
5. Keep existing `polychoric_matrix` reuse. This is important because design-menu correlations already use it.
6. Add mixed numeric plus ordinal tests if `heterogeneous_corr` supports polyserial.
7. Expand fixtures for factor variables where R `hetcor` succeeds.

Acceptance tests:

- Numeric fixtures still pass.
- Ordinal polychoric fixtures still pass.
- A mixed numeric plus ordered categorical fixture matches R `polycor::hetcor` within tolerance.
- Filtering leaves score/result rows aligned with the original data where relevant.

### 2. `full_factor`

Current status:

- PCA with varimax/none is implemented.
- PCA scores and store are implemented for complete rows.
- ML is explicitly not implemented.

Missing or incomplete:

- Maximum Likelihood factor analysis.
- Additional rotations imported by Radiant through `GPArotation`: `quartimax`, `oblimin`, `simplimax`.
- `clean_loadings`.
- `data_filter` support and storing scores into original row positions with `NA` for filtered-out rows.
- All-categorical `hcor=True` ML scores via the IRT scoring path.
- Respondent plot (`plots="resp"`).
- Square, origin-centered factor maps and repel labels.

R source references:

- `../radiant.multivariate/R/full_factor.R`

Implementation tasks:

1. Add `clean_loadings(floadings, cutoff=0, fsort=False, dec=8, repl=None)` and export it.
2. Implement `method="ML"` or `method="ml"` using `statsmodels` or a custom routine. Validate against R fixtures.
3. Support at least these rotations:
   - `none`
   - `varimax`
   - `quartimax`
   - `oblimin`
   - `simplimax`
4. If exact `simplimax` parity is too large for one pass, mark it in tests as a known gap only after confirming the UI/documentation exposes it.
5. Implement `plot(plots="attr")`, `plot(plots="resp")`, and `plot(plots=["attr", "resp"])`.
6. Use square origin-centered axes:
   - if respondent scores are shown, scale to respondent score coordinates
   - otherwise use `[-1, 1]`, matching loadings interpretation
7. Replace fixed label nudges with shared label-placement helper.
8. Update `store()` to match Radiant:
   - default prefix `factor`
   - generated names should map `RC1`/`PC1` to `factor1`, etc.
   - preserve original row count
   - insert missing values for filtered or dropped rows

Acceptance tests:

- Existing PCA fixtures still pass.
- New ML fixtures pass for at least `shopping` and `toothpaste`.
- Rotation fixtures pass for `none`, `varimax`, and any implemented additional rotations.
- `clean_loadings` output matches R for cutoff and sorting cases.
- `plot(plots="resp")` returns a valid plot and includes respondent points.
- Stored scores align with original rows under missing values and filters.

### 3. `hclus`

Current status:

- Numeric Ward/squared-Euclidean examples are implemented and tested.
- Categorical/mixed data currently raises `NotImplementedError`.

Missing or incomplete:

- Gower distance for mixed data.
- Automatic switch to Gower when categorical variables are present.
- Label uniqueness handling.
- More distance methods and linkage methods from R's `dist`/`hclust`.
- Radiant's normalized scree/change plots and cutoff behavior.
- Store default variable name should match Radiant conventions.

R source references:

- `../radiant.multivariate/R/hclus.R`

Implementation tasks:

1. Move into `hclus.py`.
2. Implement Gower distance for mixed numeric/categorical variables.
3. If any categorical variables are present and `distance != "gower"`, set distance to `"gower"` and record that setting.
4. Match Radiant standardization:
   - only numeric variables are standardized
   - categorical variables are left as categorical for Gower
5. Implement label behavior:
   - use labels only if unique
   - otherwise warn and fall back to row numbers
6. Normalize merge heights in scree/change plots by `max(height)` when plotting, matching Radiant.
7. Implement `cutoff` for scree/change/dendrogram plots.
8. Store cluster assignments with default `name = f"hclus{nr_clus}"` unless the existing Python API needs a compatibility alias.

Acceptance tests:

- Existing numeric fixtures still pass.
- New mixed-data fixture with Gower matches R `hclus`.
- A categorical case no longer raises `NotImplementedError`.
- `distance` is changed to `"gower"` when categorical variables are selected.
- Plot data for scree/change is normalized to `[0, 1]`.
- Store output has correct default name and row alignment.

### 4. `kclus`

Current status:

- Numeric k-means with hierarchical initialization is implemented and tested.
- `fun` parameter exists, but `kproto` is not implemented.
- Plotting only shows a simple means bar plot.

Missing or incomplete:

- K-Prototypes (`fun="kproto"`) for mixed numeric/categorical data.
- Gower-based hierarchical initialization for K-Prototypes.
- `lambda` handling for K-Prototypes.
- Categorical cluster modes and proportions in `clus_means`.
- Within-cluster heterogeneity proportions table.
- Density, bar, and scatter plots with categorical fallbacks.
- Store default variable name should match Radiant conventions.

R source references:

- `../radiant.multivariate/R/kclus.R`

Implementation tasks:

1. Move into `kclus.py`.
2. Preserve numeric k-means parity.
3. Implement `fun="kproto"`:
   - use a compact internal implementation or an optional dependency
   - match `clustMixType::kproto` enough to pass fixtures
   - support `lambda=None`
4. For `fun="kmeans"` with categorical variables:
   - drop categorical variables
   - print/warn that K-means cannot use them
   - suggest K-Prototypes
5. For `fun="kproto"` with no categorical variables:
   - fall back to K-means and message accordingly
6. Implement cluster centers:
   - numeric variables: means on original scale
   - categorical variables: modal level plus proportion, e.g. `"A (67%)"`
7. Implement summary output:
   - clustering method
   - HC init settings
   - lambda for K-Prototypes
   - cluster sizes
   - cluster means/modes
   - percentage of within-cluster heterogeneity per cluster
   - within/between/total heterogeneity
8. Implement plots:
   - `plots="density"`: density by cluster for numeric; proportional bar for categorical
   - `plots="bar"`: means with SE and margin-of-error bars for numeric; proportional bar for categorical
   - `plots="scatter"`: jitter/scatter by cluster for numeric; proportional bar for categorical
9. Store default variable name should be `kclus{nr_clus}`.

Acceptance tests:

- Existing numeric fixtures still pass.
- New K-Prototypes fixture with mixed variables matches R summaries.
- `plot("density")`, `plot("bar")`, and `plot("scatter")` work for numeric data.
- The same plot options work with at least one categorical variable.
- `clus_means` includes categorical modes/proportions for K-Prototypes.

### 5. `mds`

Current status:

- Metric MDS via classical scaling is implemented and tested.
- Nonmetric MDS uses scikit-learn SMACOF and is only loosely compared to R.
- Plot labels are fixed-nudge `geom_text`.

Missing or incomplete:

- More exact `MASS::isoMDS` parity for `method="non-metric"`.
- Plot all dimension pairs when `nr_dim > 2`.
- `rev_dim` plot parameter, not only mutation via `.flip()`.
- `fontsz`.
- Square origin-centered axes.
- Better label placement.
- Full Radiant summary tables: original dissimilarities, recovered distances, coordinates, stress.

R source references:

- `../radiant.multivariate/R/mds.R`

Implementation tasks:

1. Move into `mds.py`.
2. Keep metric MDS implementation.
3. Replace or improve nonmetric MDS:
   - target `MASS::isoMDS` parity
   - if exact parity is not feasible, document and test the precise accepted tolerance
   - do not leave only a loose smoke/stress test as the final state
4. Add `plot(rev_dim=None, fontsz=5, seed=1234, custom=False)`.
5. Plot all dimension pairs for `nr_dim > 2`.
6. Use square origin-centered axes.
7. Use shared label-placement helper.
8. Do not mutate object coordinates when `rev_dim` is passed to `plot`.
9. Keep `.flip()` if useful as an explicit state-changing helper, but plot should support Radiant's `rev_dim` behavior.

Acceptance tests:

- Metric fixtures still pass.
- Nonmetric fixtures have a real parity test against R output or a documented algorithmic tolerance.
- Plot data has symmetric limits.
- `plot(rev_dim=[1, 2])` flips only the rendered coordinates.
- `nr_dim=3` returns plots for `(1,2)`, `(1,3)`, and `(2,3)`.

### 6. `prmap`

Current status:

- Attribute PCA map is implemented and tested for `computer` and `retailers`.
- Preference correlations are computed for numeric preferences.
- Plot can show brands and attributes.

Missing or incomplete:

- `plots="pref"` preference arrows are not rendered.
- Labels use fixed nudges instead of repel/placement logic.
- Axes are not square and centered around origin.
- Arrow scaling does not match Radiant.
- Attribute and preference arrow segments should include arrowheads and slightly shortened endpoints.
- Preference variables of type factor should use heterogeneous correlations, matching Radiant.
- Plot all dimension pairs when `nr_dim > 2`.

R source references:

- `../radiant.multivariate/R/prmap.R`

Implementation tasks:

1. Move into `prmap.py`.
2. Reuse `full_factor`/PCA internals for loadings and scores.
3. Implement plot components:
   - `brand`
   - `attr`
   - `pref`
4. Build a single plotting data frame with:
   - x/y coordinates
   - label
   - type: `brand`, `attr`, `pref`
   - label positions from shared helper
5. Apply Radiant scaling:
   - multiply attr and pref vectors by `scaling`
   - shorten arrow endpoints by 0.9 for segment drawing
6. Use colors like Radiant:
   - brands black
   - attributes dark blue
   - preferences red
7. Use square origin-centered axes based on all selected plot components.
8. Use heterogeneous correlations for preference variables if they are categorical.

Acceptance tests:

- Existing `prmap` fixtures still pass.
- `plot(plots=["brand"])` includes only brand points/labels.
- `plot(plots=["brand", "attr"])` includes attribute arrows.
- `plot(plots=["brand", "attr", "pref"])` includes preference arrows.
- Plot limits are symmetric around zero.
- Preference factor-variable fixture matches R.

### 7. `conjoint`

Current status:

- Basic OLS conjoint is implemented and tested for `mp3`, `carpet`, and `movie`.
- Reverse rankings, by-group models, prediction, and basic plots exist.

Missing or incomplete:

- Prediction interface does not fully match Radiant:
  - `pred_data`
  - `pred_cmd`
  - `se`
  - `interval="confidence"` or `"prediction"`
  - `conf_lev`
  - by-group prediction outputs
- Prediction-store helper equivalent to `store.conjoint.predict`.
- `plot(scale_plot=True)` behavior.
- Per-attribute part-worth plots should match Radiant more closely.
- Full regression summary, diagnostics, and VIF behavior should be checked against R.
- Interactions need reference fixtures.
- `the_table` behavior and `plot_ylim` should be represented.

R source references:

- `../radiant.multivariate/R/conjoint.R`

Implementation tasks:

1. Keep categorical base-level ordering from Polars `Enum`.
2. Expand prediction API:

```python
predict(
    data=None,
    pred_cmd="",
    se=False,
    interval="confidence",
    conf_lev=0.95,
    dec=3,
)
```

3. Support prediction intervals as well as confidence intervals.
4. For by-group models, return predictions for each group in a structure equivalent to Radiant's `conjoint.predict.by`.
5. Add a store helper for predictions:

```python
store_predictions(dataset, prediction_result, name="prediction")
```

or another repo-consistent API that can append predicted values.
6. Implement `scale_plot=True` using `plot_ylim` logic from Radiant.
7. Add interaction fixtures and tests.
8. Add expand-grid teaching example from the docs:
   - generate all profiles
   - predict utilities
   - sort descending
   - identify best profile

Acceptance tests:

- Existing conjoint fixtures still pass.
- Prediction confidence interval and prediction interval fixtures match R.
- By-group predictions work.
- Interaction fixture matches R coefficients and part-worths.
- `scale_plot=True` applies shared y-limits per Radiant's `plot_ylim`.

## Notebook Requirements

There must be seven qmd notebooks and seven generated ipynb notebooks in `examples/multivariate`.

The source qmd files should be the canonical edited files. Generate ipynb from qmd after changes using the repo's existing notebook conversion workflow.

Each notebook must:

1. Include the full text from the corresponding docs page in `/home/vnijs/gh/docs/multivariate/app`.
2. Keep the Radiant teaching context, examples, and exercises.
3. Add Python code cells that reproduce the examples using `pyrsm.multivariate`.
4. Use datasets from `examples/data/multivariate` or GitHub raw URLs consistent with existing notebooks.
5. Include plots and summaries that correspond to the figures in `figures_multivariate`.
6. Include interpretation text for students.

Mapping:

| Docs page | Notebook |
| --- | --- |
| `pre_factor.md` | `multivariate-pre-factor.qmd` / `.ipynb` |
| `full_factor.md` | `multivariate-full-factor.qmd` / `.ipynb` |
| `hclus.md` | `multivariate-hclus.qmd` / `.ipynb` |
| `kclus.md` | `multivariate-kclus.qmd` / `.ipynb` |
| `mds.md` | `multivariate-mds.qmd` / `.ipynb` |
| `prmap.md` | `multivariate-prmap.qmd` / `.ipynb` |
| `conjoint.md` | `multivariate-conjoint.qmd` / `.ipynb` |

The current combined qmd files can be used as starting material, but they are not sufficient:

- `multivariate-factor-analysis.qmd` combines pre-factor and full-factor.
- `multivariate-cluster-analysis.qmd` combines hclus and kclus.
- `multivariate-maps.qmd` combines mds and prmap.
- `multivariate-conjoint.qmd` is short and omits parts of the docs.

## Reference Fixture Expansion

Update `scripts/generate_multivariate_reference.R` so the fixture set covers all key Radiant branches.

Add fixtures for:

### Pre-Factor

- numeric existing cases
- ordered categorical `hcor=TRUE`
- mixed numeric plus ordered categorical `hcor=TRUE` where `polycor::hetcor` succeeds
- filter case if filtering is implemented

### Full Factor

- PCA existing cases
- `method="ML"` with numeric data
- rotations:
  - `none`
  - `varimax`
  - `quartimax`
  - `oblimin`
  - `simplimax` if implemented
- all-ordinal hcor loadings and scores if implemented
- filter/store alignment case

### Hclus

- numeric existing cases
- mixed data with Gower
- non-default distances if exposed
- non-default linkage methods if exposed
- label duplicate behavior if feasible to test

### Kclus

- numeric existing cases
- `hc_init=False`
- mixed `fun="kproto"`
- categorical mode/proportion outputs
- `lambda` specified

### MDS

- metric existing cases
- nonmetric with exact or clearly bounded parity
- `nr_dim=3`
- axis flip plot-data fixture if useful

### PR Map

- existing computer and retailers
- preference arrows
- categorical preference variables with hcor
- `nr_dim=3`

### Conjoint

- existing mp3, carpet, movie
- interaction terms
- by-group model with fixture data
- predictions with confidence intervals
- predictions with prediction intervals
- expand-grid best-profile example

## Test Requirements

Tests should move from "implemented subset passes" to "Radiant parity is enforced."

Keep test files split by tool:

```text
tests/test_multivariate_pre_factor.py
tests/test_multivariate_full_factor.py
tests/test_multivariate_hclus.py
tests/test_multivariate_kclus.py
tests/test_multivariate_mds.py
tests/test_multivariate_prmap.py
tests/test_multivariate_conjoint.py
tests/test_multivariate_reference.py
```

If keeping combined tests temporarily, make sure final tests have coverage by tool.

Replace expected-missing tests:

- Replace `test_full_factor_ml_not_implemented` with ML parity tests.
- Replace `test_hclus_categorical_not_implemented` with Gower parity tests.
- Replace loose nonmetric MDS smoke test with a real parity test or a documented algorithm-specific tolerance.

Add plot-behavior tests:

- MDS square symmetric limits.
- PR map square symmetric limits.
- Full factor square symmetric limits.
- PR map `pref` arrows present.
- Kclus density/bar/scatter plot modes work.
- Labels have computed positions separate from raw points where needed.

Validation commands:

```bash
uv run pytest tests/test_multivariate_pre_factor.py
uv run pytest tests/test_multivariate_full_factor.py
uv run pytest tests/test_multivariate_hclus.py
uv run pytest tests/test_multivariate_kclus.py
uv run pytest tests/test_multivariate_mds.py
uv run pytest tests/test_multivariate_prmap.py
uv run pytest tests/test_multivariate_conjoint.py
uv run pytest tests/test_multivariate_reference.py
uv run pytest tests
```

## Implementation Order

Recommended order:

1. Split files and update imports.
2. Add shared plotting/label helpers.
3. Fix MDS and PR map plotting:
   - square origin axes
   - better labels
   - preference arrows
   - all dimension pairs
4. Fix full-factor plotting and add `clean_loadings`.
5. Add Gower to `hclus`.
6. Add K-Prototypes and missing `kclus` plots.
7. Add ML/rotations to `full_factor`.
8. Expand conjoint prediction, intervals, interactions, and scaled plots.
9. Expand R fixture generator.
10. Expand tests.
11. Replace four combined notebooks with seven tool-specific notebooks.
12. Regenerate ipynb notebooks from qmd.
13. Run full tests.

This order front-loads structure and plotting because those are explicit user requirements and will affect notebook outputs.

## Current Review Findings to Resolve

The following current code issues should be treated as implementation tasks:

1. `pyrsm/multivariate/__init__.py` routes `hclus` and `kclus` through `cluster.py` and `mds` and `prmap` through `maps.py`. Split these.
2. `pyrsm/multivariate/maps.py` uses fixed `geom_text` nudges for labels. Replace with shared label placement.
3. `pyrsm/multivariate/maps.py` does not use square origin-centered axes.
4. `pyrsm/multivariate/full_factor.py` rejects ML factor analysis.
5. `pyrsm/multivariate/cluster.py` rejects categorical `hclus` instead of switching to Gower.
6. `pyrsm/multivariate/cluster.py` does not implement K-Prototypes.
7. `pyrsm/multivariate/cluster.py` does not implement Radiant's density/bar/scatter plot modes for `kclus`.
8. `pyrsm/multivariate/maps.py` computes `pref_cor` but does not plot preference arrows.
9. `pyrsm/multivariate/full_factor.py` does not implement respondent plots.
10. `examples/qmd/multivariate` has four qmd files, not seven.
11. The current qmd files summarize docs instead of including full docs text.
12. Tests currently validate a subset. Expand tests to cover the missing Radiant functionality.

## Definition of Done

This work is complete when:

- There are seven implementation files, one per tool.
- `pyrsm.multivariate` exposes all seven tools at top level.
- R reference fixtures cover all major Radiant branches.
- Tests enforce parity for all seven tools.
- ML factor analysis is implemented or there is a documented, user-approved exception.
- Mixed-type hierarchical clustering with Gower is implemented.
- K-Prototypes is implemented.
- MDS and PR maps use square origin-centered axes and improved label placement.
- PR maps show brand, attribute, and preference components.
- Full factor plots show attributes and respondents.
- K-cluster plots support density, bar, and scatter views.
- Conjoint prediction supports confidence and prediction intervals and by-group models.
- There are seven qmd notebooks and seven generated ipynb notebooks in `examples/multivariate`.
- Each notebook contains the full corresponding docs text and Python examples.
- The full test suite passes:

```bash
uv run pytest tests
```

