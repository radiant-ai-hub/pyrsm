"""Tests for pyrsm.eda module."""

import numpy as np
import polars as pl
import pytest

from pyrsm.eda import (
    associations,
    combine,
    explore,
    missing,
    outliers,
    pivot,
    profile,
    visualize,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return pl.DataFrame(
        {
            "price": np.random.uniform(100, 1000, 100).tolist(),
            "quantity": np.random.randint(1, 20, 100).tolist(),
            "category": np.random.choice(["A", "B", "C"], 100).tolist(),
            "region": np.random.choice(["North", "South"], 100).tolist(),
        }
    )


@pytest.fixture
def join_data():
    """Create data for join testing."""
    orders = pl.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "customer_id": [101, 102, 101, 103, 104],
            "amount": [100.0, 200.0, 150.0, 300.0, 250.0],
        }
    )
    customers = pl.DataFrame(
        {
            "customer_id": [101, 102, 103, 105],
            "name": ["Alice", "Bob", "Charlie", "Eve"],
        }
    )
    return orders, customers


class TestExplore:
    """Tests for explore function."""

    def test_explore_default(self, sample_data):
        """Test explore with default settings (variables as rows, to_dummies=True)."""
        result = explore(sample_data)
        assert isinstance(result, pl.DataFrame)
        # Default: variables as rows, functions as columns
        assert "variable" in result.columns
        assert "mean" in result.columns
        variables = result["variable"].to_list()
        assert "price" in variables
        assert "quantity" in variables
        # to_dummies=True by default, so dummy columns should appear
        assert any("category" in v for v in variables)

    def test_explore_specific_columns(self, sample_data):
        """Test explore with specific columns."""
        result = explore(sample_data, cols=["price"])
        assert "variable" in result.columns
        variables = result["variable"].to_list()
        assert "price" in variables
        assert "quantity" not in variables

    def test_explore_custom_functions(self, sample_data):
        """Test explore with custom functions."""
        result = explore(sample_data, cols=["price"], agg=["mean", "median", "min", "max"])
        assert "variable" in result.columns
        assert "mean" in result.columns
        assert "median" in result.columns
        assert "min" in result.columns
        assert "max" in result.columns
        assert "std" not in result.columns

    def test_explore_header_variable(self, sample_data):
        """Test explore with header='variable' (statistics as rows)."""
        result = explore(sample_data, cols=["price"], header="variable")
        assert "statistic" in result.columns
        assert "price" in result.columns

    def test_explore_grouped(self, sample_data):
        """Test explore with grouping."""
        result = explore(sample_data, cols=["price"], by="category")
        assert "category" in result.columns
        assert len(result) == 3  # A, B, C

    def test_explore_invalid_function(self, sample_data):
        """Test explore with invalid function raises error."""
        with pytest.raises(ValueError, match="Unknown aggregation function"):
            explore(sample_data, agg=["invalid_func"])

    def test_explore_lazyframe(self, sample_data):
        """Test explore with LazyFrame input."""
        result = explore(sample_data.lazy())
        assert isinstance(result, pl.DataFrame)

    def test_explore_all_functions(self, sample_data):
        """Test all supported functions."""
        agg = [
            "mean",
            "median",
            "sum",
            "std",
            "var",
            "min",
            "max",
            "count",
            "n_unique",
            "null_count",
        ]
        result = explore(sample_data, cols=["price"], agg=agg)
        # Default header="function": functions are columns
        for func in agg:
            assert func in result.columns

    def test_explore_to_dummies_false(self, sample_data):
        """Test explore with to_dummies=False skips categoricals."""
        result = explore(sample_data, to_dummies=False)
        variables = result["variable"].to_list()
        assert "price" in variables
        assert "quantity" in variables
        assert not any("category" in v for v in variables)
        assert not any("region" in v for v in variables)

    # --- Dummy-related tests ---

    def test_to_dummies_default_creates_dummy_columns(self, sample_data):
        """Dummies use {col}_{level} naming, drop_first removes one level, values are Float64."""
        result = explore(sample_data)
        variables = result["variable"].to_list()
        # category has 3 levels (A, B, C) → drop_first removes one → 2 dummies
        cat_dummies = [v for v in variables if v.startswith("category_")]
        assert len(cat_dummies) == 2
        # region has 2 levels (North, South) → drop_first removes one → 1 dummy
        region_dummies = [v for v in variables if v.startswith("region_")]
        assert len(region_dummies) == 1
        # Dummy columns should be Float64 (cast from UInt8)
        for col_name in cat_dummies + region_dummies:
            row = result.filter(pl.col("variable") == col_name)
            # mean of a dummy is a float between 0 and 1
            mean_val = row["mean"][0]
            assert 0.0 <= mean_val <= 1.0

    def test_to_dummies_with_cols_only_dummifies_listed_categoricals(self, sample_data):
        """When cols lists a categorical, only that one is dummified; others are excluded."""
        # cols=["price"] with to_dummies=True: no categoricals in cols → no dummies
        result = explore(sample_data, cols=["price"])
        variables = result["variable"].to_list()
        assert variables == ["price"]
        assert not any("category" in v for v in variables)
        assert not any("region" in v for v in variables)

    def test_to_dummies_with_by_excludes_grouping_column(self, sample_data):
        """The by column is NOT dummy-encoded; other categoricals still are."""
        result = explore(sample_data, by="category")
        cols = result.columns
        # category is the grouping column, not dummified
        assert "category" in cols
        assert not any(c.startswith("category_") for c in cols)
        # region dummies should appear in the grouped output
        assert any("region_" in c for c in cols)

    def test_to_dummies_false_with_by(self, sample_data):
        """Grouped + to_dummies=False: only numeric columns aggregated, no dummies."""
        result = explore(sample_data, by="category", to_dummies=False)
        cols = result.columns
        assert "category" in cols
        assert any("price_" in c for c in cols)
        assert any("quantity_" in c for c in cols)
        assert not any("region" in c for c in cols if c != "category")

    def test_to_dummies_with_multiple_categoricals(self, sample_data):
        """Both categoricals produce dummies with correct count after drop_first."""
        result = explore(sample_data)
        variables = result["variable"].to_list()
        cat_dummies = [v for v in variables if v.startswith("category_")]
        region_dummies = [v for v in variables if v.startswith("region_")]
        # 3 levels - 1 = 2 for category, 2 levels - 1 = 1 for region
        assert len(cat_dummies) == 2
        assert len(region_dummies) == 1
        # Total variables = 2 numeric + 2 cat dummies + 1 region dummy = 5
        assert len(variables) == 5

    # --- Header ("function" vs "variable") tests ---

    def test_header_function_shape_and_layout(self, sample_data):
        """header='function': one row per variable, one column per agg + 'variable' column."""
        result = explore(sample_data, cols=["price", "quantity"], to_dummies=False)
        assert result.columns[0] == "variable"
        # Default agg has 5 functions
        assert result.shape == (2, 6)  # 2 variables, 5 agg + 1 'variable' col
        assert set(result.columns) == {"variable", "mean", "median", "min", "max", "sd"}

    def test_header_variable_shape_and_layout(self, sample_data):
        """header='variable': one row per statistic, one column per variable + 'statistic' column."""
        result = explore(
            sample_data, cols=["price", "quantity"], header="variable", to_dummies=False
        )
        assert result.columns[0] == "statistic"
        # Default agg has 5 functions
        assert result.shape == (5, 3)  # 5 stats, 2 variables + 1 'statistic' col
        assert set(result.columns) == {"statistic", "price", "quantity"}
        assert result["statistic"].to_list() == ["mean", "median", "min", "max", "sd"]

    def test_header_function_values_match_variable_values(self, sample_data):
        """Both header layouts produce identical numeric values."""
        r_func = explore(
            sample_data, cols=["price", "quantity"], to_dummies=False, header="function"
        )
        r_var = explore(
            sample_data, cols=["price", "quantity"], to_dummies=False, header="variable"
        )
        # Compare price mean from both layouts
        price_mean_func = r_func.filter(pl.col("variable") == "price")["mean"][0]
        price_mean_var = r_var.filter(pl.col("statistic") == "mean")["price"][0]
        assert price_mean_func == price_mean_var
        # Compare quantity median from both layouts
        qty_median_func = r_func.filter(pl.col("variable") == "quantity")["median"][0]
        qty_median_var = r_var.filter(pl.col("statistic") == "median")["quantity"][0]
        assert qty_median_func == qty_median_var

    def test_header_ignored_when_grouped(self, sample_data):
        """When by is set, header has no effect on the output."""
        r1 = explore(sample_data, cols=["price"], by="category", header="function").sort("category")
        r2 = explore(sample_data, cols=["price"], by="category", header="variable").sort("category")
        assert r1.columns == r2.columns
        assert r1.equals(r2)

    # --- Combined / edge-case tests ---

    def test_to_dummies_true_header_variable(self, sample_data):
        """Dummy columns appear as column headers when header='variable'."""
        result = explore(sample_data, header="variable")
        cols = result.columns
        assert "statistic" in cols
        assert "price" in cols
        assert "quantity" in cols
        # Dummy columns should appear as column headers
        assert any(c.startswith("category_") for c in cols)
        assert any(c.startswith("region_") for c in cols)

    def test_to_dummies_false_header_variable(self, sample_data):
        """No dummies with to_dummies=False and header='variable'."""
        result = explore(sample_data, header="variable", to_dummies=False)
        cols = result.columns
        assert set(cols) == {"statistic", "price", "quantity"}

    def test_sd_alias_matches_std(self, sample_data):
        """agg=['sd'] and agg=['std'] produce identical values."""
        r_sd = explore(sample_data, cols=["price"], agg=["sd"])
        r_std = explore(sample_data, cols=["price"], agg=["std"])
        assert r_sd["sd"][0] == r_std["std"][0]

    def test_n_alias_matches_count(self, sample_data):
        """agg=['n'] and agg=['count'] produce identical values."""
        r_n = explore(sample_data, cols=["price"], agg=["n"])
        r_count = explore(sample_data, cols=["price"], agg=["count"])
        assert r_n["n"][0] == r_count["count"][0]

    def test_n_missing_alias_matches_null_count(self, sample_data):
        """agg=['n_missing'] and agg=['null_count'] produce identical values."""
        r_nm = explore(sample_data, cols=["price"], agg=["n_missing"])
        r_nc = explore(sample_data, cols=["price"], agg=["null_count"])
        assert r_nm["n_missing"][0] == r_nc["null_count"][0]


class TestProfile:
    """Tests for profile function."""

    def test_profile_returns_one_row_per_column(self, sample_data):
        """Profile includes type, missingness, unique counts, and summaries."""
        result = profile(sample_data)
        assert result.height == len(sample_data.columns)
        assert {"variable", "type", "n_missing", "pct_missing", "n_unique"}.issubset(
            result.columns
        )

        price = result.filter(pl.col("variable") == "price").row(0, named=True)
        category = result.filter(pl.col("variable") == "category").row(0, named=True)
        assert price["type"] == "numeric"
        assert price["mean"] is not None
        assert category["type"] == "categorical"
        assert category["top_values"]

    def test_profile_specific_columns_lazyframe(self, sample_data):
        """Profile supports selected columns and LazyFrame input."""
        result = profile(sample_data.lazy(), cols=["price"])
        assert result["variable"].to_list() == ["price"]


class TestMissing:
    """Tests for missing function."""

    @pytest.fixture
    def missing_data(self, sample_data):
        """Create sample data with null values."""
        return sample_data.with_columns(
            price=pl.when(pl.arange(0, pl.len()) < 3)
            .then(None)
            .otherwise(pl.col("price")),
            region=pl.when(pl.arange(0, pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("region")),
        )

    def test_missing_column_summary(self, missing_data):
        """Column summary reports counts and percentages."""
        result = missing(missing_data)
        price = result.filter(pl.col("variable") == "price").row(0, named=True)
        assert price["n_missing"] == 3
        assert price["pct_missing"] == pytest.approx(0.03)

    def test_missing_by_group(self, missing_data):
        """Grouped missingness returns wide per-group counts and rates."""
        result = missing(missing_data, cols=["price"], by="category")
        assert "category" in result.columns
        assert "price_n_missing" in result.columns
        assert "price_pct_missing" in result.columns

    def test_missing_patterns(self, missing_data):
        """Pattern table uses 1 for missing and 0 for observed."""
        result = missing(missing_data, cols=["price", "region"], patterns=True)
        assert {"price", "region", "n", "pct"}.issubset(result.columns)
        assert result["n"].sum() == missing_data.height


class TestAssociations:
    """Tests for associations function."""

    def test_associations_target(self, sample_data):
        """Target mode computes associations from each selected variable to target."""
        df = sample_data.with_columns(price_scaled=pl.col("price") * 2)
        result = associations(
            df,
            cols=["price_scaled", "category", "region"],
            target="price",
        )
        assert {"var1", "var2", "metric", "value", "p_value", "n"}.issubset(
            result.columns
        )
        assert set(result["var2"].to_list()) == {"price"}
        assert "pearson" in result["metric"].to_list()
        assert "eta_squared" in result["metric"].to_list()

    def test_associations_categorical_categorical(self):
        """Categorical pairs return Cramer's V."""
        df = pl.DataFrame(
            {
                "a": ["x", "x", "y", "y", "y", "x"],
                "b": ["u", "u", "v", "v", "u", "v"],
            }
        )
        result = associations(df)
        row = result.row(0, named=True)
        assert row["metric"] == "cramers_v"
        assert row["value"] is not None

    def test_associations_invalid_method(self, sample_data):
        """Invalid numeric correlation method raises."""
        with pytest.raises(ValueError, match="method must be"):
            associations(sample_data, method="kendall")


class TestOutliers:
    """Tests for outliers function."""

    def test_outliers_iqr_summary(self):
        """IQR summary flags a clear extreme value."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 100.0], "g": ["a", "a", "b", "b"]})
        result = outliers(df, cols="x")
        row = result.row(0, named=True)
        assert row["variable"] == "x"
        assert row["n_outliers"] == 1
        assert row["pct_outliers"] == pytest.approx(0.25)

    def test_outliers_flags(self):
        """ret='flags' returns row-level boolean indicators."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 100.0]})
        result = outliers(df, cols="x", ret="flags")
        assert result.columns == ["row_nr", "x_outlier"]
        assert result["x_outlier"].sum() == 1

    def test_outliers_invalid_method(self, sample_data):
        """Invalid outlier method raises."""
        with pytest.raises(ValueError, match="method must be"):
            outliers(sample_data, method="bad")


class TestPivot:
    """Tests for pivot function."""

    def test_pivot_frequency_table(self, sample_data):
        """Test single variable frequency table."""
        result = pivot(sample_data, rows="category")
        assert "category" in result.columns
        assert "count" in result.columns
        assert len(result) == 3  # A, B, C

    def test_pivot_crosstab(self, sample_data):
        """Test two-variable crosstab."""
        result = pivot(sample_data, rows="category", cols="region")
        assert "category" in result.columns
        assert "North" in result.columns or "South" in result.columns

    def test_pivot_with_values(self, sample_data):
        """Test pivot with value aggregation."""
        result = pivot(sample_data, rows="category", values="price", agg="mean")
        assert "category" in result.columns
        assert "price_mean" in result.columns

    def test_pivot_with_totals(self, sample_data):
        """Test pivot with totals."""
        result = pivot(sample_data, rows="category", totals=True)
        # Last row should be "Total"
        assert result["category"].to_list()[-1] == "Total"

    def test_pivot_normalize_row(self, sample_data):
        """Test pivot with row normalization."""
        result = pivot(sample_data, rows="category", cols="region", normalize="row", totals=True)
        # Row totals should be 100%
        total_col = result.filter(pl.col("category") != "Total")["Total"]
        assert all(abs(v - 100.0) < 0.01 for v in total_col.to_list())

    def test_pivot_invalid_agg(self, sample_data):
        """Test pivot with invalid aggregation raises error."""
        with pytest.raises(ValueError, match="Unknown aggregation"):
            pivot(sample_data, rows="category", agg="invalid")

    def test_pivot_lazyframe(self, sample_data):
        """Test pivot with LazyFrame input."""
        result = pivot(sample_data.lazy(), rows="category")
        assert isinstance(result, pl.DataFrame)

    # ---- Extended aggregation set (radiant.data parity) -----------

    def test_pivot_n_obs_alias(self, sample_data):
        """``n_obs`` is an alias of ``count`` for frequency tables."""
        a = pivot(sample_data, rows="category").sort("category")
        b = pivot(sample_data, rows="category", agg="n_obs").sort("category")
        # ``count`` and ``n_obs`` use the same underlying ``pl.len()``.
        assert a["count"].to_list() == b["count"].to_list()

    def test_pivot_n_distinct(self, sample_data):
        """``n_distinct`` counts unique values per group."""
        result = pivot(sample_data, rows="category", values="region", agg="n_distinct")
        # Two regions in the fixture: North + South.
        assert all(v in (1, 2) for v in result["region_n_distinct"].to_list())

    def test_pivot_n_missing(self, sample_data):
        """``n_missing`` returns the null count per group."""
        # Inject some nulls.
        df = sample_data.with_columns(
            pl.when(pl.col("price") < 200).then(None).otherwise(pl.col("price")).alias("price")
        )
        result = pivot(df, rows="category", values="price", agg="n_missing")
        assert (result["price_n_missing"] >= 0).all()

    def test_pivot_se_me_cv(self, sample_data):
        """``se``, ``me``, and ``cv`` populate without errors."""
        for agg in ("se", "me", "cv"):
            result = pivot(sample_data, rows="category", values="price", agg=agg)
            col = f"price_{agg}"
            assert col in result.columns
            # Non-trivial dataset → expect finite, non-null values.
            assert result[col].null_count() == 0

    def test_pivot_iqr(self, sample_data):
        """``IQR`` returns p75 - p25."""
        result = pivot(sample_data, rows="category", values="price", agg="IQR")
        # IQR for 100 uniform draws on [100, 1000] is positive in every group.
        assert (result["price_IQR"] > 0).all()

    def test_pivot_percentiles(self, sample_data):
        """Percentile keys ``p01`` … ``p99`` are accepted and ordered."""
        p10 = pivot(sample_data, rows="category", values="price", agg="p10")
        p90 = pivot(sample_data, rows="category", values="price", agg="p90")
        # 10th percentile must be ≤ 90th percentile per group.
        for cat, lo, hi in zip(
            p10["category"].to_list(),
            p10["price_p10"].to_list(),
            p90.sort("category")["price_p90"].to_list(),
        ):
            assert lo <= hi, f"p10 > p90 for category {cat}"

    def test_pivot_skew_kurtosis(self, sample_data):
        """``skew`` / ``kurtosis`` route through scipy and return Float64."""
        for agg in ("skew", "kurtosis"):
            result = pivot(sample_data, rows="category", values="price", agg=agg)
            assert result[f"price_{agg}"].dtype == pl.Float64

    def test_pivot_prop(self, sample_data):
        """``prop`` collapses a binary column to its share-of-1s."""
        df = sample_data.with_columns((pl.col("price") > 500).cast(pl.Int64).alias("is_high"))
        result = pivot(df, rows="category", values="is_high", agg="prop")
        # Proportions live in [0, 1].
        for v in result["is_high_prop"].to_list():
            assert 0.0 <= v <= 1.0


class TestCombine:
    """Tests for combine function."""

    def test_inner_join(self, join_data):
        """Test inner join."""
        orders, customers = join_data
        result = combine(orders, customers, on="customer_id", how="inner")
        assert "name" in result.columns
        # Only customers 101, 102, 103 are in both
        assert len(result) == 4  # orders 1, 2, 3, 4 have matching customers

    def test_left_join(self, join_data):
        """Test left join."""
        orders, customers = join_data
        result = combine(orders, customers, on="customer_id", how="left")
        assert len(result) == 5  # All orders kept

    def test_right_join(self, join_data):
        """Test right join."""
        orders, customers = join_data
        result = combine(orders, customers, on="customer_id", how="right")
        # All customers kept, including Eve (105) with no orders
        customer_ids = set(result["customer_id"].to_list())
        assert 105 in customer_ids  # Eve should be included

    def test_bind_rows(self, sample_data):
        """Test bind_rows."""
        df1 = sample_data.head(50)
        df2 = sample_data.tail(50)
        result = combine(df1, df2, how="bind_rows")
        assert len(result) == 100

    def test_bind_cols(self):
        """Test bind_cols."""
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"b": [4, 5, 6]})
        result = combine(df1, df2, how="bind_cols")
        assert "a" in result.columns
        assert "b" in result.columns

    def test_intersect(self):
        """Test intersect."""
        df1 = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        df2 = pl.DataFrame({"x": [2, 3, 4], "y": ["b", "c", "d"]})
        result = combine(df1, df2, how="intersect")
        assert len(result) == 2  # rows (2, "b") and (3, "c")

    def test_union(self):
        """Test union."""
        df1 = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        df2 = pl.DataFrame({"x": [2, 3], "y": ["b", "c"]})
        result = combine(df1, df2, how="union")
        assert len(result) == 3  # Unique rows

    def test_setdiff(self):
        """Test setdiff."""
        df1 = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        df2 = pl.DataFrame({"x": [2, 3], "y": ["b", "c"]})
        result = combine(df1, df2, how="setdiff")
        assert len(result) == 1  # Only (1, "a")

    def test_join_requires_on(self, join_data):
        """Test that join types require on parameter."""
        orders, customers = join_data
        with pytest.raises(ValueError, match="requires on="):
            combine(orders, customers, how="inner")

    def test_invalid_how(self, join_data):
        """Test invalid combine how raises error."""
        orders, customers = join_data
        with pytest.raises(ValueError, match="Unknown combine"):
            combine(orders, customers, how="invalid_how")

    def test_different_key_names(self, join_data):
        """Test join with different key names."""
        orders, customers = join_data
        # Rename customer_id in customers to cust_id
        customers_renamed = customers.rename({"customer_id": "cust_id"})
        result = combine(
            orders,
            customers_renamed,
            left_on="customer_id",
            right_on="cust_id",
            how="inner",
        )
        assert "name" in result.columns


class TestVisualize:
    """Tests for visualize function."""

    def test_visualize_histogram(self, sample_data):
        """Test histogram creation."""
        p = visualize(sample_data, x="price")
        assert p is not None

    def test_visualize_scatter(self, sample_data):
        """Test scatter plot creation."""
        p = visualize(sample_data, x="price", y="quantity")
        assert p is not None

    def test_visualize_multiple_x_composes_plots(self, sample_data):
        """Multiple x variables return a composed plot grid by default."""
        from plotnine.composition import Compose

        p = visualize(sample_data, x=["price", "quantity"], geom="hist")
        assert isinstance(p, Compose)

    def test_visualize_multiple_x_and_y_can_return_plot_list(self, sample_data):
        """ret='list' returns one ggplot per x/y pair in nested-loop order."""
        plots = visualize(
            sample_data,
            x=["price", "quantity"],
            y=["quantity", "price"],
            geom="scatter",
            ret="list",
        )

        assert len(plots) == 4
        assert [p.labels.x for p in plots] == [
            "price",
            "price",
            "quantity",
            "quantity",
        ]
        assert [p.labels.y for p in plots] == [
            "quantity",
            "price",
            "quantity",
            "price",
        ]

    def test_visualize_invalid_ret_raises(self, sample_data):
        """Invalid return mode raises a helpful error."""
        with pytest.raises(ValueError, match="ret must be"):
            visualize(sample_data, x=["price", "quantity"], ret="invalid")

    def test_visualize_bar(self, sample_data):
        """Test bar chart for categorical."""
        p = visualize(sample_data, x="category", geom="bar")
        assert p is not None

    def test_visualize_bar_with_y_groups_by_x_mean_by_default(self, sample_data):
        """Bar charts with y aggregate duplicate x values before plotting."""
        import matplotlib.pyplot as plt

        p = visualize(sample_data, x="category", y="price", geom="bar")
        expected = sample_data.group_by("category").agg(
            pl.col("price").mean().alias("price")
        )
        actual = p.data

        assert actual.height == expected.height
        assert actual.sort("category")["category"].to_list() == expected.sort(
            "category"
        )["category"].to_list()
        assert actual.sort("category")["price"].to_list() == pytest.approx(
            expected.sort("category")["price"].to_list()
        )
        assert type(p.layers[0].stat).__name__ == "stat_identity"
        assert p.labels.y == "Mean of price"

        p.draw()
        plt.close("all")

    def test_visualize_bar_with_y_respects_sum_agg(self, sample_data):
        """Bar-chart agg selection is computed with Polars group_by."""
        p = visualize(sample_data, x="category", y="price", geom="bar", agg="sum")
        expected = sample_data.group_by("category").agg(
            pl.col("price").sum().alias("price")
        )
        actual = p.data

        assert actual.height == expected.height
        assert actual.sort("category")["price"].to_list() == pytest.approx(
            expected.sort("category")["price"].to_list()
        )
        assert p.labels.y == "Sum of price"

    def test_visualize_box(self, sample_data):
        """Test box plot."""
        p = visualize(sample_data, x="category", y="price", geom="box")
        assert p is not None

    def test_visualize_with_color(self, sample_data):
        """Test plot with color aesthetic."""
        p = visualize(sample_data, x="price", y="quantity", color="category")
        assert p is not None

    def test_visualize_with_facet(self, sample_data):
        """Test faceted plot."""
        p = visualize(sample_data, x="price", facet="category")
        assert p is not None

    def test_visualize_smooth(self, sample_data):
        """Test scatter with smooth line."""
        p = visualize(sample_data, x="price", y="quantity", smooth="lm")
        assert p is not None

    def test_visualize_invalid_geom(self, sample_data):
        """Test invalid geom raises error."""
        with pytest.raises(ValueError, match="Unknown geom"):
            visualize(sample_data, x="price", geom="invalid")

    def test_visualize_missing_y(self, sample_data):
        """Test geom requiring y raises error when y missing."""
        with pytest.raises(ValueError, match="y is required"):
            visualize(sample_data, x="price", geom="scatter")

    def test_visualize_lazyframe(self, sample_data):
        """Test visualize with LazyFrame input."""
        p = visualize(sample_data.lazy(), x="price")
        assert p is not None

    def test_visualize_density(self, sample_data):
        """Test density plot."""
        p = visualize(sample_data, x="price", geom="density")
        assert p is not None

    def test_visualize_violin(self, sample_data):
        """Test violin plot."""
        p = visualize(sample_data, x="category", y="price", geom="violin")
        assert p is not None
