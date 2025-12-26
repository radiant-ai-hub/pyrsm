# Justfile for pyrsm package testing and deployment
# Usage: just <recipe>

# Variables
rundev := "uv run --extra dev --extra ml --extra plot --extra tables"
runnb := "uv run --extra notebooks --extra plot --extra ml"
nbconvert := rundev + " jupyter nbconvert --to notebook --execute --inplace"

# Default recipe - show help
default:
    @just --list

# =============================================================================
# Testing
# =============================================================================

# Run all pytest tests
test:
    {{rundev}} pytest tests/ -v --tb=short

# Run basics module tests only
test-basics:
    {{rundev}} pytest tests/test_basics.py tests/test_single_*.py tests/test_compare_*.py tests/test_correlation*.py tests/test_goodness.py tests/test_cross_tabs.py tests/test_central_limit_theorem.py -v --tb=short

# Run model module tests only
test-model:
    {{rundev}} pytest tests/test_regression.py tests/test_logistic.py tests/test_perf.py tests/test_mlp.py tests/test_rforest.py tests/test_xgb.py tests/test_model_utils.py -v --tb=short

# Run eda module tests only
test-eda:
    {{rundev}} pytest tests/test_eda.py -v --tb=short

# Run all tests with verbose output
test-verbose:
    {{rundev}} pytest tests/ -v --tb=long

# Run tests with coverage report
test-coverage:
    {{rundev}} pytest tests/ -v --cov=pyrsm --cov-report=term-missing

# Quick smoke test (fast subset)
smoke:
    uv run pytest tests/test_basics.py tests/test_stats.py -v --tb=short -x

# =============================================================================
# Notebook Execution
# =============================================================================

# Execute basics notebooks only
notebooks-basics:
    @echo "=== Executing basics notebooks ==="
    {{nbconvert}} examples/basics/basics-compare-means.ipynb
    {{nbconvert}} examples/basics/basics-compare-props.ipynb
    {{nbconvert}} examples/basics/basics-correlation.ipynb
    {{nbconvert}} examples/basics/basics-cross-tabs.ipynb
    {{nbconvert}} examples/basics/basics-goodness.ipynb
    {{nbconvert}} examples/basics/basics-probability-calculator.ipynb
    {{nbconvert}} examples/basics/basics-single-proportion.ipynb
    @echo "=== All basics notebooks passed ==="

# Execute model notebooks only
notebooks-model:
    @echo "=== Executing model notebooks ==="
    {{nbconvert}} examples/model/model-linear-regression.ipynb
    {{nbconvert}} examples/model/model-logistic-regression.ipynb
    {{nbconvert}} examples/model/model-mlp-classification.ipynb
    {{nbconvert}} examples/model/model-mlp-regression.ipynb
    {{nbconvert}} examples/model/model-rforest-classification.ipynb
    {{nbconvert}} examples/model/model-rforest-regression.ipynb
    {{nbconvert}} examples/model/model-xgboost-classification.ipynb
    {{nbconvert}} examples/model/model-xgboost-regression.ipynb
    @echo "=== All model notebooks passed ==="

# Execute eda notebooks only
notebooks-eda:
    @echo "=== Executing eda notebooks ==="
    {{nbconvert}} examples/eda/eda-distr.ipynb
    {{nbconvert}} examples/eda/eda-explore.ipynb
    {{nbconvert}} examples/eda/eda-pivot.ipynb
    {{nbconvert}} examples/eda/eda-visualize.ipynb
    @echo "=== All eda notebooks passed ==="

# Execute data notebooks only
notebooks-data:
    @echo "=== Executing data notebooks ==="
    {{nbconvert}} examples/data/load-example-data.ipynb
    {{nbconvert}} examples/data/save-load-state.ipynb
    @echo "=== All data notebooks passed ==="

# Execute all example notebooks
notebooks: notebooks-basics notebooks-model notebooks-eda notebooks-data
    @echo "=== All notebooks executed successfully ==="

# Run all notebooks with detailed error checking
notebooks-run:
    @echo "=== Running all notebooks with error checking ==="
    {{runnb}} python scripts/run_notebooks.py --path examples
    @echo "=== All notebooks completed ==="

# Check notebooks for errors (no execution)
notebooks-check:
    @echo "=== Checking notebooks for errors (no execution) ==="
    {{runnb}} python scripts/run_notebooks.py --path examples --check-only
    @echo "=== Check complete ==="

# =============================================================================
# Code Quality
# =============================================================================

# Run ruff linter
lint:
    {{rundev}} ruff check pyrsm/ tests/

# Run ruff linter and fix issues
lint-fix:
    {{rundev}} ruff check pyrsm/ tests/ --fix

# Format code with black
format:
    {{rundev}} black pyrsm/ tests/

# Check code formatting with black
format-check:
    {{rundev}} black pyrsm/ tests/ --check

# Run ruff + black checks (run before committing)
pre-commit:
    @echo "=== Running ruff check ==="
    {{rundev}} ruff check pyrsm/ tests/
    @echo "=== Running black check ==="
    {{rundev}} black pyrsm/ tests/ --check
    @echo "=== All checks passed ==="

# Fix all ruff + black issues
pre-commit-fix:
    @echo "=== Fixing ruff issues ==="
    {{rundev}} ruff check pyrsm/ tests/ --fix
    @echo "=== Formatting with black ==="
    {{rundev}} black pyrsm/ tests/
    @echo "=== All fixes applied ==="

# =============================================================================
# Build & Deploy
# =============================================================================

# Remove build artifacts
clean:
    rm -rf dist/ build/ *.egg-info .pytest_cache .ruff_cache
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete

# Build package
build: clean
    uv build

# Publish to TestPyPI
publish-test: build
    @echo "Publishing to TestPyPI..."
    {{rundev}} twine upload --repository testpypi dist/*
    @echo ""
    @echo "Install from TestPyPI with:"
    @echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pyrsm"

# Publish to PyPI
[confirm("Are you sure you want to publish to PyPI?")]
publish: build
    @echo "Publishing to PyPI..."
    {{rundev}} twine upload dist/*
    @echo "Published to PyPI successfully!"

# =============================================================================
# Combined Targets
# =============================================================================

# Run all tests and notebooks
test-all: test notebooks
    @echo "=== All tests and notebooks passed ==="

# Run lint, tests, and notebooks
check: lint test notebooks
    @echo "=== All checks passed ==="

# =============================================================================
# Data Maintenance
# =============================================================================

# Dry run: show parquet categorical columns
parquet-check:
    @echo "=== Checking parquet files for categorical columns (dry run) ==="
    uv run python examples/data/convert_parquet_categoricals.py --path .

# Convert categoricals to Polars Enum (with backup)
parquet-convert:
    @echo "=== Converting parquet categoricals to Polars Enum ==="
    uv run python examples/data/convert_parquet_categoricals.py --path . --apply
