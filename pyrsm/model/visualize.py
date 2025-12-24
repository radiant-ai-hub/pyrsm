import math
import time
import warnings

import numpy as np
import pandas as pd
import polars as pl
import statsmodels as sm
from plotnine import (
    aes,
    coord_flip,
    geom_bar,
    geom_hline,
    geom_line,
    geom_point,
    geom_tile,
    ggplot,
    ggtitle,
    labs,
    scale_fill_gradient2,
    scale_x_discrete,
    scale_y_continuous,
    theme,
    theme_bw,
)
from sklearn.inspection import permutation_importance

from pyrsm.utils import expand_grid, ifelse, intersect, setdiff

# Suppress plotnine warnings about missing values in ICE lines
warnings.filterwarnings("ignore", message=".*Removed.*rows containing missing values.*")

from .model import (
    extract_evars,
    extract_rvar,
    get_dummies,
    sim_prediction,
)
from .perf import auc


def _is_numeric_with_many_unique(
    data: pl.DataFrame, col: str, threshold: int = 5
) -> bool:
    """Check if column is numeric with more than threshold unique values (polars version)."""
    dtype = data.schema.get(col)
    if dtype is None:
        return False
    if not dtype.is_numeric():
        return False
    # n_unique = data.select(pl.col(col).n_unique()).item()
    n_unique = data.get_column(col).n_unique()
    return n_unique > threshold


def _is_categorical(df: pl.DataFrame, col: str, nint: int = 25) -> bool:
    """Check if column should be treated as categorical."""
    dtype = df.schema.get(col)
    if dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Enum):
        return True
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        # n_unique = df.select(pl.col(col).n_unique()).item()
        n_unique = df.get_column(col).n_unique()
        return n_unique < nint
    return False


def _get_cat_order(df: pl.DataFrame, col: str) -> list | None:
    """Get category order from Enum column, or None if not Enum."""
    dtype = df[col].dtype
    if isinstance(dtype, pl.Enum):
        return list(dtype.categories)
    return None


def _to_pandas_for_predict(pred_data: pl.DataFrame) -> pd.DataFrame:
    """
    Convert polars DataFrame to pandas for statsmodels prediction.

    Polars Enum columns preserve all category levels during pandas conversion,
    which is required by patsy/statsmodels.

    Parameters
    ----------
    pred_data : pl.DataFrame
        The prediction data to convert (should have Enum columns for categoricals)

    Returns
    -------
    pd.DataFrame ready for statsmodels prediction
    """
    return pred_data.to_pandas()


def _calc_r_squared(y, yhat) -> float:
    """Calculate R² using polars correlation."""
    # Handle both Series and lists
    if isinstance(y, pl.Series):
        y_vals = y
    else:
        y_vals = pl.Series("y", list(y))

    if isinstance(yhat, pl.Series):
        yhat_vals = yhat
    else:
        yhat_vals = pl.Series("yhat", list(yhat))

    corr = (
        pl.DataFrame({"y": y_vals, "yhat": yhat_vals})
        .select(pl.corr("y", "yhat"))
        .item()
    )
    return corr**2 if corr is not None else 0.0


def _extract_model_and_scaling(model, data, rvar=None):
    """Extract fitted model and build scaled data dict if model has scaling info.

    For MLP models, scaling info (means/stds) is needed for correct predictions.
    This helper detects if a full model object (with .means, .stds, .fitted) was
    passed and extracts the scaling info into a data dict.

    Returns: (fitted_model, data_or_data_dict, rvar)
    """
    # Check if this is an mlp object (has means, stds, fitted attributes)
    if hasattr(model, "means") and hasattr(model, "stds") and hasattr(model, "fitted"):
        fitted = model.fitted
        evar = model.evar if hasattr(model, "evar") else []
        model_rvar = model.rvar if hasattr(model, "rvar") else rvar

        if data is None:
            # Use training data stored on model
            cols = evar + ([model_rvar] if model_rvar else [])
            data_dct = {
                "data": model.data.select(cols),
                "means": model.means,
                "stds": model.stds,
            }
        else:
            # Use provided data
            # data = check_dataframe(data) if not isinstance(data, pl.DataFrame) else data
            cols = evar + (
                [model_rvar] if model_rvar and model_rvar in data.columns else []
            )
            data_dct = {
                "data": data.select(cols) if cols else data,
                "means": model.means,
                "stds": model.stds,
            }
        return fitted, data_dct, model_rvar
    # Otherwise assume it's already a fitted sklearn model
    return model, data, rvar


def _compose_plots(plot_list: list, ncol: int = 2):
    """Compose a list of plots into a grid using plotnine's | and / operators."""
    if len(plot_list) == 0:
        return None
    if len(plot_list) == 1:
        return plot_list[0]

    nrow = math.ceil(len(plot_list) / ncol)

    # Build rows (side by side with |)
    rows = []
    for i in range(nrow):
        start_idx = i * ncol
        end_idx = min(start_idx + ncol, len(plot_list))
        row_plots = plot_list[start_idx:end_idx]

        if len(row_plots) == 1:
            rows.append(row_plots[0])
        else:
            row = row_plots[0]
            for p in row_plots[1:]:
                row = row | p
            rows.append(row)

    # Stack rows vertically with /
    if len(rows) == 1:
        combined = rows[0]
    else:
        combined = rows[0]
        for row in rows[1:]:
            combined = combined / row

    # Auto-adjust figure size
    height_per_row = 3
    width_per_col = 4
    fig_width = width_per_col * min(ncol, len(plot_list))
    fig_height = height_per_row * nrow
    combined = combined + theme(figure_size=(fig_width, fig_height))

    return combined


def _tranformation_info(fn, col_names, transformed):
    not_transformed = [c for c in col_names for f in fn if c == f]
    if transformed is None:
        # Use dict.fromkeys instead of set to preserve column order from data
        transformed = list(
            dict.fromkeys(
                c
                for c in col_names
                for f in fn
                if f"{c}_" in f and c != f and f"{c}_" not in col_names
            )
        )
    return not_transformed, transformed


# ============================================================================
# Shared Helper Functions (to reduce duplication)
# ============================================================================


def _calc_ylim(
    vals: list,
    min_max: tuple,
    fix: bool | tuple,
    plot_margin: float = 0.025,
) -> tuple:
    """Calculate y-axis limits for prediction plots.

    Parameters
    ----------
    vals : list
        Prediction values to consider for limits
    min_max : tuple
        Current (min, max) limits
    fix : bool or tuple
        If True, calculate limits from vals. If tuple, use as fixed limits.
    plot_margin : float
        Margin to add above/below extreme values

    Returns
    -------
    tuple of (min, max) y-axis limits
    """
    if isinstance(fix, bool) and fix:
        min_vals = min(vals)
        max_vals = max(vals)
        mmin = min(min_max[0], min_vals - plot_margin * abs(min_vals))
        mmax = max(min_max[1], max_vals + plot_margin * abs(max_vals))
        return (mmin, mmax)
    elif not isinstance(fix, bool) and len(fix) == 2:
        return fix
    else:
        return min_max


def _init_hline_and_minmax(
    data: pl.DataFrame,
    rvar: str | None,
    hline: bool | float,
    plot_margin: float = 0.025,
) -> tuple:
    """Initialize horizontal line value and initial min/max bounds.

    Parameters
    ----------
    data : pl.DataFrame
        Data containing the response variable
    rvar : str or None
        Response variable name
    hline : bool or float
        If True, compute mean of rvar. If float, use as hline value.
    plot_margin : float
        Margin to add above/below hline

    Returns
    -------
    tuple of (hline_val, min_max)
    """
    if isinstance(hline, bool):
        if hline and rvar is not None and rvar in data.columns:
            col = data[rvar]
            # For numeric columns, use mean; for categorical, compute proportion
            if col.dtype in [
                pl.Float64,
                pl.Float32,
                pl.Int64,
                pl.Int32,
                pl.Int16,
                pl.Int8,
            ]:
                hline_val = col.mean()
            elif col.dtype == pl.Boolean:
                hline_val = col.mean()
            else:
                # String/categorical: check for common positive labels
                positive_str = ["Yes", "yes", "YES", "1", "True", "true"]
                hline_val = col.cast(pl.Utf8).is_in(positive_str).mean()

            if hline_val is not None:
                min_max = (
                    hline_val - plot_margin * abs(hline_val),
                    hline_val + plot_margin * abs(hline_val),
                )
            else:
                min_max = (float("inf"), float("-inf"))
        else:
            hline_val = None
            min_max = (float("inf"), float("-inf"))
    else:
        hline_val = hline
        min_max = (hline - plot_margin * abs(hline), hline + plot_margin * abs(hline))

    return hline_val, min_max


def _dummify_for_sk(
    data: pl.DataFrame,
    transformed_cols: list[str],
    feature_names: list[str],
) -> pl.DataFrame:
    """Create dummy variables for sklearn model prediction.

    Parameters
    ----------
    data : pl.DataFrame
        Input data with categorical columns to dummify
    transformed_cols : list[str]
        List of column names that need dummy encoding
    feature_names : list[str]
        Feature names from fitted sklearn model (fitted.feature_names_in_)

    Returns
    -------
    pl.DataFrame with dummy columns matching model feature names
    """
    if len(transformed_cols) == 0:
        return data

    categories = {}
    for col in transformed_cols:
        prefix = f"{col}_"
        categories[col] = [
            f.replace(prefix, "") for f in feature_names if f.startswith(prefix)
        ]

    dummified = get_dummies(
        data.select(transformed_cols),
        drop_first=False,
        drop_nonvarying=False,
        categories=categories,
    )

    non_trs = [c for c in data.columns if c not in transformed_cols]
    if non_trs:
        dummified = pl.concat([dummified, data.select(non_trs)], how="horizontal")

    return dummified


def _scale_for_prediction(
    data: pl.DataFrame,
    not_transformed: list[str],
    means: dict | None,
    stds: dict | None,
) -> pl.DataFrame:
    """Scale numeric columns for sklearn MLP prediction.

    Parameters
    ----------
    data : pl.DataFrame
        Input data to scale
    not_transformed : list[str]
        Columns that were not dummy-encoded (numeric columns)
    means : dict or None
        Column means for scaling (from MLP model)
    stds : dict or None
        Column stds for scaling (from MLP model)

    Returns
    -------
    pl.DataFrame with scaled numeric columns
    """
    if means is None or stds is None:
        return data

    for col in not_transformed:
        if col in means and col in data.columns:
            data = data.with_columns(
                ((pl.col(col) - means[col]) / stds[col]).alias(col)
            )
    return data


def _prepare_incl_excl(
    incl: list | str | None,
    excl: list | str,
    incl_int: list | str,
    default_incl: list,
) -> tuple[list, list]:
    """Prepare include and exclude lists for plotting.

    Parameters
    ----------
    incl : list, str, or None
        Variables to include in plots
    excl : list or str
        Variables to exclude from plots
    incl_int : list or str
        Interaction terms to include
    default_incl : list
        Default list if incl is None

    Returns
    -------
    tuple of (incl, incl_int) after processing
    """
    if incl is None:
        incl = default_incl
    else:
        incl = ifelse(isinstance(incl, str), [incl], incl)

    excl = ifelse(isinstance(excl, str), [excl], excl)
    incl_int = ifelse(isinstance(incl_int, str), [incl_int], incl_int)

    if len(excl) > 0:
        incl = [i for i in incl if i not in excl]

    return incl, incl_int


def _create_single_var_plot(
    data: pl.DataFrame,
    var: str,
    is_numeric: bool,
    min_max: tuple,
    hline_val: float | None,
    cat_order: list | None = None,
    ice_data: list | None = None,
) -> ggplot:
    """Create a single-variable prediction plot.

    Parameters
    ----------
    data : pl.DataFrame
        Data with 'var' and 'prediction' columns
    var : str
        Variable name (x-axis)
    is_numeric : bool
        Whether the variable is numeric (line) or categorical (line+points)
    min_max : tuple
        Y-axis limits (min, max)
    hline_val : float or None
        Horizontal line value (mean of target), or None to skip
    cat_order : list or None
        Category order for x-axis (for Enum columns)
    ice_data : list or None
        ICE data: list of arrays, one per grid point, each containing predictions
        for all observations at that grid point

    Returns
    -------
    plotnine ggplot object
    """
    # Build ICE layer first (so PDP line is drawn on top)
    ice_layer = None
    if ice_data is not None and len(ice_data) > 0:
        # Get grid values from data
        grid_vals = data[var].to_list()

        # Build ICE DataFrame: one row per (observation, grid_point)
        ice_rows = []
        for grid_idx, preds in enumerate(ice_data):
            for obs_idx, pred in enumerate(preds):
                ice_rows.append(
                    {var: grid_vals[grid_idx], "prediction": pred, "obs": obs_idx}
                )
        ice_df = pl.DataFrame(ice_rows).drop_nulls()

        # Suppress plotnine warnings about missing values in ICE lines
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*missing values.*", category=UserWarning
            )
            if is_numeric:
                ice_layer = geom_line(
                    data=ice_df,
                    mapping=aes(x=var, y="prediction", group="obs"),
                    color="gray",
                    alpha=0.2,
                    size=0.3,
                )
            else:
                ice_layer = geom_line(
                    data=ice_df,
                    mapping=aes(x=var, y="prediction", group="obs"),
                    color="gray",
                    alpha=0.2,
                    size=0.3,
                )

    if is_numeric:
        p = ggplot(data, aes(x=var, y="prediction"))
        if ice_layer is not None:
            p = p + ice_layer
        p = p + geom_line(color="steelblue", size=1)
    else:
        p = ggplot(data, aes(x=var, y="prediction"))
        if ice_layer is not None:
            p = p + ice_layer
        p = (
            p
            + geom_line(color="steelblue", group=1, size=1)
            + geom_point(color="steelblue", size=3)
        )
        # Preserve category order for categorical variables
        if cat_order is not None:
            p = p + scale_x_discrete(limits=cat_order)

    p = p + labs(x="", y="Prediction") + ggtitle(var) + theme_bw()

    if isinstance(min_max, tuple) and min_max[0] != float("inf"):
        p = p + scale_y_continuous(limits=min_max)

    if hline_val is not None:
        p = p + geom_hline(yintercept=hline_val, linetype="dashed", color="gray")

    return p


def _create_interaction_plot(
    data: pl.DataFrame,
    var: str,
    var_list: list[str],
    is_num: list[bool],
    min_max: tuple,
    hline_val: float | None,
    is_pdp: bool = False,
) -> ggplot:
    """Create an interaction plot for two variables.

    Parameters
    ----------
    data : pl.DataFrame
        Data with var1, var2, and 'prediction' columns
    var : str
        Original interaction string (e.g., "x1:x2") for title
    var_list : list[str]
        [var1, var2] column names
    is_num : list[bool]
        [is_var1_numeric, is_var2_numeric]
    min_max : tuple
        Y-axis limits (min, max)
    hline_val : float or None
        Horizontal line value, or None to skip
    is_pdp : bool
        If True, use PDP style (num+num shows colored lines with rounded labels)
        If False, use pred_plot style (num+num shows heatmap)

    Returns
    -------
    plotnine ggplot object
    """
    vl = var_list.copy()

    if sum(is_num) == 2:
        if is_pdp:
            # PDP: Two numeric - line plot with color (sliced)
            data = data.with_columns(pl.col(vl[1]).round(2).cast(pl.Utf8).alias(vl[1]))
            p = (
                ggplot(data, aes(x=vl[0], y="prediction", color=vl[1]))
                + geom_line()
                + labs(x=vl[0], y="Prediction", color=vl[1])
                + ggtitle(var)
                + theme_bw()
            )
        else:
            # pred_plot: Heatmap for two numeric variables
            p = (
                ggplot(data, aes(x=vl[0], y=vl[1], fill="prediction"))
                + geom_tile()
                + scale_fill_gradient2(low="blue", mid="white", high="red")
                + labs(x=vl[0], y=vl[1], fill="Prediction")
                + ggtitle(var)
                + theme_bw()
            )
        # No y-limit or hline for heatmaps and 2-numeric PDP lines
        return p

    elif sum(is_num) == 1:
        # One numeric, one categorical - line plot with color grouping
        if is_num[1]:
            vl = [vl[1], vl[0]]
        data = data.with_columns(pl.col(vl[1]).cast(pl.Utf8))
        p = (
            ggplot(data, aes(x=vl[0], y="prediction", color=vl[1]))
            + geom_line()
            + labs(x=vl[0], y="Prediction", color=vl[1])
            + ggtitle(var)
            + theme_bw()
        )
    else:
        # Both categorical
        data = data.with_columns(pl.col(vl[1]).cast(pl.Utf8))
        p = (
            ggplot(data, aes(x=vl[0], y="prediction", color=vl[1]))
            + geom_line(aes(group=vl[1]))
            + geom_point(size=3)
            + labs(x=vl[0], y="Prediction", color=vl[1])
            + ggtitle(var)
            + theme_bw()
        )

    # Apply y-axis limits and hline for non-heatmap plots
    if isinstance(min_max, tuple) and min_max[0] != float("inf"):
        p = p + scale_y_continuous(limits=min_max)
    if hline_val is not None:
        p = p + geom_hline(yintercept=hline_val, linetype="dashed", color="gray")

    return p


def pred_plot_sm(
    fitted,
    data: pl.DataFrame | pd.DataFrame,
    incl=None,
    excl=[],
    incl_int=[],
    fix=True,
    hline=True,
    nnv=20,
    minq=0.025,
    maxq=0.975,
):
    """
    Generate prediction plots for statsmodels regression models (OLS and Logistic).
    A faster alternative to PDP plots.

    Parameters
    ----------
    fitted : A fitted (logistic) regression model
    data: Polars or Pandas DataFrame; dataset
    incl: List of strings; contains the names of the columns of data to use for prediction
          By default it will extract the names of all explanatory variables used in estimation
          Use [] to ensure no single-variable plots are created
    excl: List of strings; contains names of columns to exclude
    incl_int: List of strings; contains the names of the columns of data to be interacted for
          prediction plotting (e.g., ["x1:x2", "x2:x3"])
    fix : Logical or tuple
        Set the desired limited on yhat or have it calculated automatically.
        Set to FALSE to have y-axis limits set for each plot
    hline : Logical or float
        Add a horizontal line at the average of the target variable
    nnv: Integer: The number of values to simulate for numeric variables
    minq : float
        Quantile to use for the minimum value for simulation of numeric variables
    maxq : float
        Quantile to use for the maximum value for simulation of numeric variables

    Returns
    -------
    plotnine plot composition
    """
    if hasattr(fitted, "model"):
        model = fitted.model
    else:
        model = fitted

    # Convert pandas to polars if needed
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    rvar = model.endog_names

    # Initialize hline and y-limits
    hline_val, min_max = _init_hline_and_minmax(data, rvar, hline)

    # Prepare include/exclude lists
    default_incl = extract_evars(model, data.columns)
    incl, incl_int = _prepare_incl_excl(incl, excl, incl_int, default_incl)

    if len(incl) + len(incl_int) == 0:
        return None

    # Generate predictions for each variable
    pred_dict = {}
    for v in incl:
        iplot = sim_prediction(data, vary=v, nnv=nnv, minq=minq, maxq=maxq)
        predictions = fitted.predict(iplot.to_pandas())
        iplot = iplot.with_columns(prediction=pl.Series(predictions))
        min_max = _calc_ylim(predictions, min_max, fix)
        pred_dict[v] = iplot

    for v in incl_int:
        vl = v.split(":")
        is_num = [_is_numeric_with_many_unique(data, c, 5) for c in vl]
        iplot = sim_prediction(data, vary=vl, nnv=nnv, minq=minq, maxq=maxq)
        predictions = fitted.predict(iplot.to_pandas())
        iplot = iplot.with_columns(prediction=pl.Series(predictions))
        if sum(is_num) < 2:
            min_max = _calc_ylim(predictions, min_max, fix)
        pred_dict[v] = iplot

    # Create plots using helper functions
    plot_list = []
    for v in incl:
        is_num = _is_numeric_with_many_unique(data, v, 5)
        cat_order = None if is_num else _get_cat_order(data, v)
        p = _create_single_var_plot(
            pred_dict[v], v, is_num, min_max, hline_val, cat_order
        )
        plot_list.append(p)

    for v in incl_int:
        vl = v.split(":")
        is_num = [_is_numeric_with_many_unique(data, c, 5) for c in vl]
        p = _create_interaction_plot(
            pred_dict[v], v, vl, is_num, min_max, hline_val, is_pdp=False
        )
        plot_list.append(p)

    return _compose_plots(plot_list, ncol=2)


def pred_plot_sk(
    fitted,
    data: pl.DataFrame | pd.DataFrame,
    rvar=None,
    incl=None,
    excl=[],
    incl_int=[],
    transformed=None,
    fix=True,
    hline=True,
    nnv=20,
    minq=0.025,
    maxq=0.975,
    ret=False,
):
    """
    Generate prediction plots for sklearn models. A faster alternative to PDP plots
    that can handle interaction plots with categorical variables.

    Parameters
    ----------
    fitted : A fitted sklearn model
    data : Polars DataFrame with data used for estimation
    rvar : The column name for the response/target variable
    incl : A list of column names to generate prediction plots for
    excl : A list of column names to exclude from plotting
    incl_int : A list of ":" separated column names for interaction plots
    transformed : List of column names that were transformed using get_dummies
    fix : Logical or tuple for y-axis limits
    hline : Logical or float for horizontal line
    nnv: int - number of values for numeric variables
    minq : float - quantile for minimum value
    maxq : float - quantile for maximum value
    ret : Return the prediciton dictionary for testing purposes

    Returns
    -------
    plotnine plot composition
    """
    # Feature names from sklearn model
    if hasattr(fitted, "feature_names_in_"):
        fn = fitted.feature_names_in_
    else:
        raise Exception(
            "This function requires a fitted sklearn model with named features."
        )

    # Handle data dict with scaling info (MLP models)
    # Note: Do NOT scale here - scaling happens in pred_fun via _scale_for_prediction
    if isinstance(data, dict):
        means = data["means"]
        stds = data["stds"]
        data = data["data"]
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)
    else:
        means = None
        stds = None
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)

    # Identify transformed vs not-transformed columns
    not_transformed, transformed = _tranformation_info(fn, data.columns, transformed)

    # Validate transformation info
    ints = intersect(transformed, not_transformed)
    if len(ints) > 0:
        trn = '", "'.join(transformed)
        not_transformed = setdiff(not_transformed, transformed)
        raise Exception(
            f'Unclear which variables were transformed. Please specify transformed=["{trn}"].'
        )

    # Determine model type
    sk_type = "classification" if hasattr(fitted, "classes_") else "regression"

    # Prediction function with scaling
    def pred_fun(fitted, pred_data):
        pred_data = _scale_for_prediction(pred_data, not_transformed, means, stds)
        if sk_type == "classification":
            return fitted.predict_proba(pred_data.to_pandas())[:, 1]
        else:
            pred = fitted.predict(pred_data.to_pandas())
            if means is not None and stds is not None and rvar is not None:
                pred = pred * stds[rvar] + means[rvar]
            return pred

    # Prepare include/exclude lists
    default_incl = not_transformed + transformed
    incl, incl_int = _prepare_incl_excl(incl, excl, incl_int, default_incl)

    if len(incl) + len(incl_int) == 0:
        return None

    # Initialize hline and y-limits
    hline_val, min_max = _init_hline_and_minmax(data, rvar, hline)

    # Generate predictions
    pred_dict = {}
    base_cols = transformed + not_transformed
    base_data = data.select(base_cols).drop_nulls()

    for v in incl:
        iplot = sim_prediction(base_data, vary=v, nnv=nnv, minq=minq, maxq=maxq)
        iplot_dum = _dummify_for_sk(iplot, transformed, fn).select(fn)
        predictions = pred_fun(fitted, iplot_dum)
        iplot = iplot.with_columns(prediction=pl.Series(predictions))
        min_max = _calc_ylim(predictions, min_max, fix)
        pred_dict[v] = iplot

    for v in incl_int:
        vl = v.split(":")
        is_num = [_is_numeric_with_many_unique(data, c, 5) for c in vl]
        iplot = sim_prediction(base_data, vary=vl, nnv=nnv, minq=minq, maxq=maxq)
        iplot_dum = _dummify_for_sk(iplot, transformed, fn).select(fn)
        predictions = pred_fun(fitted, iplot_dum)
        iplot = iplot.with_columns(prediction=pl.Series(predictions))
        if sum(is_num) < 2:
            min_max = _calc_ylim(predictions, min_max, fix)
        pred_dict[v] = iplot

    if ret:
        return pred_dict

    # Create plots using helper functions
    plot_list = []
    for v in incl:
        is_num = _is_numeric_with_many_unique(data, v, 5)
        cat_order = None if is_num else _get_cat_order(data, v)
        p = _create_single_var_plot(
            pred_dict[v], v, is_num, min_max, hline_val, cat_order
        )
        plot_list.append(p)

    for v in incl_int:
        vl = v.split(":")
        is_num = [_is_numeric_with_many_unique(data, c, 5) for c in vl]
        p = _create_interaction_plot(
            pred_dict[v], v, vl, is_num, min_max, hline_val, is_pdp=False
        )
        plot_list.append(p)

    return _compose_plots(plot_list, ncol=2)


def pdp_sk(
    fitted,
    data: pl.DataFrame | pd.DataFrame,
    rvar=None,
    incl=None,
    excl=[],
    incl_int=[],
    transformed=None,
    mode="pdp",
    ice=False,
    ice_nobs=100,
    n_sample=2000,
    grid_resolution=50,
    minq=0.05,
    maxq=0.95,
    interaction_slices=5,
    fix=True,
    hline=True,
    ncol=2,
):
    """
    Generate Partial Dependence Plots (PDP) for sklearn models.

    Parameters
    ----------
    fitted : A fitted sklearn model
    data : Polars or Pandas DataFrame with data used for estimation
    rvar : The column name for the response/target variable
    incl : A list of column names to generate PDP plots for
    excl : A list of column names to exclude from plotting
    incl_int : A list of ":" separated column names for interaction plots
    transformed : List of column names that were transformed using get_dummies
    mode : str, "fast" or "pdp"
        "fast" - uses sim_prediction (mean/mode for other vars, like pred_plot_sk)
        "pdp" - true PDP: samples rows, replaces feature values, averages predictions
    ice : bool, default False
        If True, show Individual Conditional Expectation (ICE) lines behind the PDP.
        ICE lines show the prediction for each observation as the feature varies.
        Only works with mode="pdp".
    ice_nobs : int, default 100
        Number of observations to sample for ICE lines. Only used when ice=True.
        Limits the number of ICE lines to improve performance and readability.
        Use -1 to show ICE lines for all observations.
    n_sample : int
        Number of samples to use for PDP mode (caps at dataset size)
    grid_resolution : int
        Number of grid points for numeric variables
    minq : float
        Quantile for minimum value of numeric variables
    maxq : float
        Quantile for maximum value of numeric variables
    interaction_slices : int
        Number of slices for numeric-numeric interactions (line plot)
    fix : Logical or tuple for y-axis limits
    hline : Logical or float for horizontal line
    ncol : int
        Number of columns in plot grid

    Returns
    -------
    tuple: (plot, data_dict, runtime_seconds)
        - plot: plotnine plot composition
        - data_dict: dict of DataFrames with underlying PDP data
        - runtime_seconds: total computation time
    """
    start_time = time.time()

    # Extract scaling info if model object (e.g., mlp) was passed instead of fitted model
    fitted, data, rvar = _extract_model_and_scaling(fitted, data, rvar)

    # Handle data dict with scaling info (for MLP models)
    if isinstance(data, dict):
        means = data["means"]
        stds = data["stds"]
        data = data["data"]
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)
    else:
        means = None
        stds = None
        if isinstance(data, pd.DataFrame):
            data = pl.from_pandas(data)

    # Get feature names from sklearn model
    if hasattr(fitted, "feature_names_in_"):
        fn = fitted.feature_names_in_
    else:
        raise ValueError(
            "This function requires a fitted sklearn model with named features."
        )

    # Determine model type
    sk_type = "classification" if hasattr(fitted, "classes_") else "regression"

    # Identify transformed and not-transformed variables
    not_transformed, transformed = _tranformation_info(fn, data.columns, transformed)

    ints = intersect(transformed, not_transformed)
    if len(ints) > 0:
        trn = '", "'.join(transformed)
        raise ValueError(
            f'Unclear which variables were transformed. Please specify transformed=["{trn}"].'
        )

    # Prepare include/exclude lists
    default_incl = not_transformed + transformed
    incl, incl_int = _prepare_incl_excl(incl, excl, incl_int, default_incl)

    if len(incl) + len(incl_int) == 0:
        return None

    # Prediction function with scaling
    def pred_fun(fitted, data_for_pred):
        data_for_pred = _scale_for_prediction(
            data_for_pred, not_transformed, means, stds
        )
        data_pd = (
            data_for_pred.to_pandas()
            if isinstance(data_for_pred, pl.DataFrame)
            else data_for_pred
        )
        if sk_type == "classification":
            return fitted.predict_proba(data_pd)[:, 1]
        else:
            preds = fitted.predict(data_pd)
            # Unscale predictions for MLP regression
            if means is not None and stds is not None and rvar in means:
                preds = preds * stds[rvar] + means[rvar]
            return preds

    # Get base data for predictions
    base_data = data.select(transformed + not_transformed).drop_nulls()

    # Sample data for PDP mode
    n_obs = base_data.height
    if mode == "pdp":
        sample_size = min(n_sample, n_obs)
        sample_data = (
            base_data.sample(sample_size, seed=1234)
            if sample_size < n_obs
            else base_data
        )
    else:
        sample_data = base_data

    # Sample data for ICE lines (separate, smaller sample for performance)
    # ice_nobs=-1 or None means use all observations
    if ice and mode == "pdp":
        if ice_nobs is None or ice_nobs < 0:
            ice_sample_data = sample_data
        else:
            ice_sample_size = min(ice_nobs, sample_data.height)
            ice_sample_data = (
                sample_data.sample(ice_sample_size, seed=5678)
                if ice_sample_size < sample_data.height
                else sample_data
            )
    else:
        ice_sample_data = None

    # Helper to build grid for a variable - preserves Enum ordering
    def build_grid(var):
        col = data[var]
        if _is_categorical(data, var, 5):
            # Preserve Enum category order if available
            if isinstance(col.dtype, pl.Enum):
                return list(col.dtype.categories)
            return col.unique(maintain_order=True).drop_nulls().to_list()
        else:
            nu = col.n_unique()
            min_val = col.quantile(minq)
            max_val = col.quantile(maxq)
            return np.linspace(min_val, max_val, min(nu, grid_resolution)).tolist()

    # Compute PDP for single variable
    def compute_pdp_single(var):
        grid_vals = build_grid(var)
        predictions = []
        ice_predictions = [] if ice_sample_data is not None else None

        if mode == "pdp":
            for gv in grid_vals:
                # PDP: use full sample_data for mean prediction
                modified = sample_data.with_columns(
                    pl.lit(gv).cast(sample_data[var].dtype).alias(var)
                )
                modified_dum = _dummify_for_sk(modified, transformed, fn).select(fn)
                preds = pred_fun(fitted, modified_dum)
                predictions.append(np.mean(preds))

                # ICE: use smaller ice_sample_data for individual predictions
                if ice_predictions is not None:
                    ice_modified = ice_sample_data.with_columns(
                        pl.lit(gv).cast(ice_sample_data[var].dtype).alias(var)
                    )
                    ice_modified_dum = _dummify_for_sk(
                        ice_modified, transformed, fn
                    ).select(fn)
                    ice_preds = pred_fun(fitted, ice_modified_dum)
                    ice_predictions.append(ice_preds.tolist())
        else:
            iplot = sim_prediction(
                base_data, vary=var, nnv=grid_resolution, minq=minq, maxq=maxq
            )
            iplot_dum = _dummify_for_sk(iplot, transformed, fn).select(fn)
            preds = pred_fun(fitted, iplot_dum)
            grid_vals = iplot[var].to_list()
            predictions = preds.tolist()

        return (
            pl.DataFrame({var: grid_vals, "prediction": predictions}),
            ice_predictions,
        )

    # Compute PDP for interaction (two variables)
    def compute_pdp_interaction(var1, var2):
        is_num1 = _is_numeric_with_many_unique(data, var1, 5)
        is_num2 = _is_numeric_with_many_unique(data, var2, 5)

        grid1 = build_grid(var1)
        grid2 = build_grid(var2)

        if is_num1 and is_num2:
            grid2 = np.linspace(
                data[var2].quantile(minq), data[var2].quantile(maxq), interaction_slices
            ).tolist()

        schema = {var1: data[var1].dtype, var2: data[var2].dtype}
        grid_df = expand_grid({var1: grid1, var2: grid2}, schema)
        predictions = []

        if mode == "pdp":
            for row in grid_df.iter_rows(named=True):
                modified = sample_data.with_columns(
                    pl.lit(row[var1]).cast(sample_data[var1].dtype).alias(var1),
                    pl.lit(row[var2]).cast(sample_data[var2].dtype).alias(var2),
                )
                modified_dum = _dummify_for_sk(modified, transformed, fn).select(fn)
                preds = pred_fun(fitted, modified_dum)
                predictions.append(np.mean(preds))
        else:
            for row in grid_df.iter_rows(named=True):
                iplot = sim_prediction(
                    base_data, vary={var1: [row[var1]], var2: [row[var2]]}
                )
                iplot_dum = _dummify_for_sk(iplot, transformed, fn).select(fn)
                preds = pred_fun(fitted, iplot_dum)
                predictions.append(preds[0])

        return grid_df.with_columns(prediction=pl.Series(predictions))

    # Initialize hline and y-limits
    hline_val, min_max = _init_hline_and_minmax(data, rvar, hline)

    # Compute PDPs for all variables
    pred_dict = {}
    ice_dict = {}
    for v in incl:
        result, ice_data = compute_pdp_single(v)
        pred_dict[v] = result
        ice_dict[v] = ice_data
        min_max = _calc_ylim(result["prediction"].to_list(), min_max, fix)

    for v in incl_int:
        vl = v.split(":")
        is_num = [_is_numeric_with_many_unique(data, c, 5) for c in vl]
        result = compute_pdp_interaction(vl[0], vl[1])
        pred_dict[v] = result
        if sum(is_num) < 2:
            min_max = _calc_ylim(result["prediction"].to_list(), min_max, fix)

    # Create plots using helper functions
    plot_list = []
    for v in incl:
        is_num = _is_numeric_with_many_unique(data, v, 5)
        cat_order = None if is_num else _get_cat_order(data, v)
        p = _create_single_var_plot(
            pred_dict[v], v, is_num, min_max, hline_val, cat_order, ice_dict.get(v)
        )
        plot_list.append(p)

    for v in incl_int:
        vl = v.split(":")
        is_num = [_is_numeric_with_many_unique(data, c, 5) for c in vl]
        p = _create_interaction_plot(
            pred_dict[v], v, vl, is_num, min_max, hline_val, is_pdp=True
        )
        plot_list.append(p)

    time.time() - start_time
    plot = _compose_plots(plot_list, ncol=ncol)

    # if plot is not None:
    #     plot = plot + labs(caption=f"Runtime: {runtime:.2f}s | Mode: {mode}")

    # return plot, pred_dict, runtime
    return plot


def pdp_sm(
    fitted,
    data: pl.DataFrame | pd.DataFrame,
    incl=None,
    excl=[],
    incl_int=[],
    mode="pdp",
    ice=False,
    ice_nobs=100,
    n_sample=2000,
    grid_resolution=50,
    minq=0.05,
    maxq=0.95,
    interaction_slices=5,
    fix=True,
    hline=True,
    ncol=2,
):
    """
    Generate Partial Dependence Plots (PDP) for statsmodels regression models.

    Parameters
    ----------
    fitted : A fitted statsmodels model (OLS or Logistic)
    data : Polars or Pandas DataFrame with data used for estimation
    incl : A list of column names to generate PDP plots for
    excl : A list of column names to exclude from plotting
    incl_int : A list of ":" separated column names for interaction plots
    mode : str, "fast" or "pdp"
        "fast" - uses sim_prediction (mean/mode for other vars)
        "pdp" - true PDP: samples rows, replaces feature values, averages predictions
    ice : bool, default False
        If True, show Individual Conditional Expectation (ICE) lines behind the PDP.
        ICE lines show the prediction for each observation as the feature varies.
        Only works with mode="pdp".
    ice_nobs : int, default 100
        Number of observations to sample for ICE lines. Only used when ice=True.
        Limits the number of ICE lines to improve performance and readability.
        Use -1 to show ICE lines for all observations.
    n_sample : int
        Number of samples to use for PDP mode
    grid_resolution : int
        Number of grid points for numeric variables
    minq : float
        Quantile for minimum value of numeric variables
    maxq : float
        Quantile for maximum value of numeric variables
    interaction_slices : int
        Number of slices for numeric-numeric interactions
    fix : Logical or tuple for y-axis limits
    hline : Logical or float for horizontal line
    ncol : int
        Number of columns in plot grid

    Returns
    -------
    tuple: (plot, data_dict, runtime_seconds)
        - plot: plotnine plot composition
        - data_dict: dict of DataFrames with underlying PDP data
        - runtime_seconds: total computation time
    """
    start_time = time.time()

    if hasattr(fitted, "model"):
        model = fitted.model
    else:
        model = fitted

    # Convert to polars if needed
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    rvar = model.endog_names

    # Prepare include/exclude lists
    default_incl = extract_evars(model, data.columns)
    incl, incl_int = _prepare_incl_excl(incl, excl, incl_int, default_incl)

    if len(incl) + len(incl_int) == 0:
        return None, {}, 0.0

    # Get all evars for prediction
    all_evars = extract_evars(model, data.columns)

    # Get base data with all evars
    base_data = data.select(all_evars).drop_nulls()

    # Sample for PDP mode
    n_obs = base_data.height
    if mode == "pdp":
        sample_size = min(n_sample, n_obs)
        sample_data = (
            base_data.sample(sample_size, seed=1234)
            if sample_size < n_obs
            else base_data
        )
    else:
        sample_data = base_data

    # Sample data for ICE lines (separate, smaller sample for performance)
    # ice_nobs=-1 or None means use all observations
    if ice and mode == "pdp":
        if ice_nobs is None or ice_nobs < 0:
            ice_sample_data = sample_data
        else:
            ice_sample_size = min(ice_nobs, sample_data.height)
            ice_sample_data = (
                sample_data.sample(ice_sample_size, seed=5678)
                if ice_sample_size < sample_data.height
                else sample_data
            )
    else:
        ice_sample_data = None

    # Helper to build grid - preserves Enum ordering
    def build_grid(var):
        col = data[var]
        if _is_categorical(data, var, 5):
            # Preserve Enum category order if available
            if isinstance(col.dtype, pl.Enum):
                return list(col.dtype.categories)
            return col.unique(maintain_order=True).drop_nulls().to_list()
        else:
            nu = col.n_unique()
            min_val = col.quantile(minq)
            max_val = col.quantile(maxq)
            return np.linspace(min_val, max_val, min(nu, grid_resolution)).tolist()

    # Compute PDP for single variable
    def compute_pdp_single(var):
        grid_vals = build_grid(var)
        predictions = []
        ice_predictions = [] if ice_sample_data is not None else None

        if mode == "pdp":
            evars = extract_evars(model, data.columns)
            for gv in grid_vals:
                # PDP: use full sample_data for mean prediction
                modified = sample_data.select(evars).with_columns(
                    pl.lit(gv).cast(sample_data[var].dtype).alias(var)
                )
                modified_pd = _to_pandas_for_predict(modified)
                preds = fitted.predict(modified_pd)
                predictions.append(np.mean(preds))

                # ICE: use smaller ice_sample_data for individual predictions
                if ice_predictions is not None:
                    ice_modified = ice_sample_data.select(evars).with_columns(
                        pl.lit(gv).cast(ice_sample_data[var].dtype).alias(var)
                    )
                    ice_modified_pd = _to_pandas_for_predict(ice_modified)
                    ice_preds = fitted.predict(ice_modified_pd)
                    ice_predictions.append(list(ice_preds))
            return (
                pl.DataFrame({var: grid_vals, "prediction": predictions}),
                ice_predictions,
            )
        else:
            iplot = sim_prediction(
                data, vary=var, nnv=grid_resolution, minq=minq, maxq=maxq
            )
            preds = fitted.predict(iplot.to_pandas())
            return iplot.with_columns(prediction=pl.Series(list(preds))), None

    # Compute PDP for interaction
    def compute_pdp_interaction(var1, var2):
        is_num1 = _is_numeric_with_many_unique(data, var1, 5)
        is_num2 = _is_numeric_with_many_unique(data, var2, 5)

        grid1 = build_grid(var1)
        grid2 = build_grid(var2)

        if is_num1 and is_num2:
            grid2 = np.linspace(
                data[var2].quantile(minq),
                data[var2].quantile(maxq),
                interaction_slices,
            ).tolist()

        schema = {var1: data[var1].dtype, var2: data[var2].dtype}
        grid_df = expand_grid({var1: grid1, var2: grid2}, schema)
        predictions = []

        evars = extract_evars(model, data.columns)

        if mode == "pdp":
            for row in grid_df.iter_rows(named=True):
                modified = sample_data.select(evars).with_columns(
                    pl.lit(row[var1]).cast(sample_data[var1].dtype).alias(var1),
                    pl.lit(row[var2]).cast(sample_data[var2].dtype).alias(var2),
                )
                modified_pd = _to_pandas_for_predict(modified)
                preds = fitted.predict(modified_pd)
                predictions.append(np.mean(preds))
        else:
            for row in grid_df.iter_rows(named=True):
                iplot = sim_prediction(
                    data, vary={var1: [row[var1]], var2: [row[var2]]}
                )
                preds = fitted.predict(iplot.to_pandas())
                predictions.append(preds[0])

        return grid_df.with_columns(prediction=pl.Series(predictions))

    # Initialize hline and y-limits
    hline_val, min_max = _init_hline_and_minmax(data, rvar, hline)

    # Compute PDPs
    pred_dict = {}
    ice_dict = {}
    for v in incl:
        result, ice_data = compute_pdp_single(v)
        pred_dict[v] = result
        ice_dict[v] = ice_data
        min_max = _calc_ylim(result["prediction"].to_list(), min_max, fix)

    for v in incl_int:
        vl = v.split(":")
        is_num = [_is_numeric_with_many_unique(data, c, 5) for c in vl]
        result = compute_pdp_interaction(vl[0], vl[1])
        pred_dict[v] = result
        if sum(is_num) < 2:
            min_max = _calc_ylim(result["prediction"].to_list(), min_max, fix)

    # Create plots using helper functions
    plot_list = []
    for v in incl:
        is_num = _is_numeric_with_many_unique(data, v, 5)
        cat_order = None if is_num else _get_cat_order(data, v)
        p = _create_single_var_plot(
            pred_dict[v], v, is_num, min_max, hline_val, cat_order, ice_dict.get(v)
        )
        plot_list.append(p)

    for v in incl_int:
        vl = v.split(":")
        is_num = [_is_numeric_with_many_unique(data, c, 5) for c in vl]
        p = _create_interaction_plot(
            pred_dict[v], v, vl, is_num, min_max, hline_val, is_pdp=True
        )
        plot_list.append(p)

    time.time() - start_time
    plot = _compose_plots(plot_list, ncol=ncol)

    # if plot is not None:
    #     plot = plot + labs(caption=f"Runtime: {runtime:.2f}s | Mode: {mode}")

    # return plot, pred_dict, runtime
    return plot


def pip_plot_sm(fitted, data: pl.DataFrame | pd.DataFrame, rep=10, ret=False):
    """
    Creates permutation importance plots for models estimated using the
    statsmodels library.

    Parameters
    ----------
    fitted : A fitted statsmodels object
    data : Polars DataFrame with data used for estimation
    rep: int
        The number of times to resample and calculate the permutation importance
    ret: bool
        Return the variable importance table as a sorted DataFrame

    Returns
    -------
    plotnine plot or polars DataFrame if ret=True
    """
    fw = None
    if hasattr(fitted, "model"):
        model = fitted.model
        if hasattr(model, "_has_freq_weights") and model._has_freq_weights:
            fw = model.freq_weights
    else:
        return "This function requires a fitted linear or logistic regression"

    # Convert pandas to polars if needed
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    rvar = extract_rvar(model, data.columns)
    evars = extract_evars(model, data.columns)
    data = data.select([rvar] + evars).drop_nulls()

    if len(model.endog) != data.height:
        raise Exception(
            "The number of rows in the DataFrame should be the same as the number of rows in the data used to estimate the model"
        )

    def imp_calc_reg(base, pred):
        return base - _calc_r_squared(model.endog, pred)

    def imp_calc_logit(base, pred):
        return base - auc(model.endog, pred, weights=fw)

    # Calculate the baseline performance
    if isinstance(fitted, sm.genmod.generalized_linear_model.GLMResultsWrapper):
        baseline_fit = auc(model.endog, fitted.predict(data.to_pandas()), weights=fw)
        imp_calc = imp_calc_logit
        xlab = "Importance (AUC decrease)"
    elif isinstance(fitted, sm.regression.linear_model.RegressionResultsWrapper):
        baseline_fit = _calc_r_squared(model.endog, fitted.predict(data.to_pandas()))
        imp_calc = imp_calc_reg
        xlab = "Importance (R-square decrease)"
    else:
        return "This model type is not supported. For sklearn models use pip_plot_sk"

    # Initialize importance values
    importance_values = {v: 0.0 for v in evars}

    # Iterate over each feature
    for i in range(rep):
        for feature in evars:
            # Shuffle the feature column
            shuffled = data.with_columns(pl.col(feature).shuffle(seed=i).alias(feature))
            shuffled_pd = shuffled.select(evars).to_pandas()
            importance_values[feature] += imp_calc(
                baseline_fit, fitted.predict(shuffled_pd)
            )

    # Average importance
    importance_values = {k: v / rep for k, v in importance_values.items()}

    # Create DataFrame for plotting
    imp_df = pl.DataFrame(
        {
            "variable": list(importance_values.keys()),
            "importance": list(importance_values.values()),
        }
    ).sort("importance")

    # Create horizontal bar plot
    p = (
        ggplot(imp_df, aes(x="variable", y="importance"))
        + geom_bar(stat="identity", fill="slateblue")
        + coord_flip()
        + scale_x_discrete(limits=imp_df["variable"].to_list())
        + labs(x="", y=xlab)
        + ggtitle("Permutation Importance")
        + theme_bw()
        + theme(figure_size=(6, max(5, len(imp_df) * 0.4)))
    )

    if ret:
        return (imp_df.sort("importance", descending=True), p)

    return (None, p)


def pip_plot_sk(model, rep=5, ret=False):
    """
    Creates permutation importance plots for models estimated using the
    sklearn library. Handles categorical variables by shuffling the original
    variable rather than dummy-encoded versions.

    Parameters
    ----------
    model : A pyrsm model object with fitted sklearn model
    rep: int
        The number of times to resample and calculate permutation importance
    ret: bool
        Return the variable importance table as a sorted DataFrame

    Returns
    -------
    plotnine plot or polars DataFrame if ret=True
    """
    rvar = model.rvar
    evars = model.evar

    # Get data as polars
    data = model.data.select([rvar] + evars).drop_nulls()

    def imp_calc_reg(base, pred):
        return base - _calc_r_squared(data[rvar], pred)

    def imp_calc_clf(base, pred):
        return base - auc(data[rvar].to_list(), pred)

    # Calculate the baseline performance
    if hasattr(model.fitted, "classes_"):
        xlab = "Importance (AUC decrease)"
        baseline_pred = model.predict(data.select(evars))
        baseline_fit = auc(data[rvar].to_list(), baseline_pred["prediction"].to_list())
        imp_calc = imp_calc_clf
    else:
        baseline_pred = model.predict(data.select(evars))
        baseline_fit = _calc_r_squared(data[rvar], baseline_pred["prediction"])
        imp_calc = imp_calc_reg
        xlab = "Importance (R-square decrease)"

    # Initialize importance values
    importance_values = {v: 0.0 for v in evars}

    # Iterate over each feature
    for i in range(rep):
        for feature in evars:
            # Shuffle the feature column
            permuted = data.with_columns(pl.col(feature).shuffle(seed=i).alias(feature))
            pred_result = model.predict(permuted.select(evars))
            importance_values[feature] += imp_calc(
                baseline_fit, pred_result["prediction"].to_list()
            )

    # Average importance
    importance_values = {k: v / rep for k, v in importance_values.items()}

    # Create DataFrame for plotting
    imp_df = pl.DataFrame(
        {
            "variable": list(importance_values.keys()),
            "importance": list(importance_values.values()),
        }
    ).sort("importance")

    # Create horizontal bar plot
    p = (
        ggplot(imp_df, aes(x="variable", y="importance"))
        + geom_bar(stat="identity", fill="slateblue")
        + coord_flip()
        + scale_x_discrete(limits=imp_df["variable"].to_list())
        + labs(x="", y=xlab)
        + ggtitle("Permutation Importance")
        + theme_bw()
        + theme(figure_size=(6, max(5, len(imp_df) * 0.4)))
    )

    return (p, imp_df.sort("importance", descending=True))


def pip_plot_sklearn(fitted, X: pl.DataFrame, y, rep=5, ret=False):
    """
    Creates permutation importance plots for models estimated using the
    sklearn library using sklearn's built-in permutation_importance.

    Parameters
    ----------
    fitted : A fitted sklearn object
    X : Polars DataFrame with explanatory variables (features)
    y : Series or list with response variable (target)
    rep: int
        The number of times to resample and calculate permutation importance
    ret: bool
        Return the variable importance table as a sorted DataFrame

    Returns
    -------
    plotnine plot or polars DataFrame if ret=True
    """
    if hasattr(fitted, "classes_"):
        scoring = "roc_auc"
        xlab = "Importance (AUC decrease)"
    else:
        scoring = "r2"
        xlab = "Importance (R-square decrease)"

    # sklearn needs pandas/numpy
    if isinstance(X, pd.DataFrame):
        X_pd = X
    else:
        X_pd = X.to_pandas()
    y_list = y.to_list() if isinstance(y, pl.Series) else list(y)

    imp = permutation_importance(
        fitted, X_pd, y_list, scoring=scoring, n_repeats=rep, random_state=1234
    )

    # Create importance DataFrame
    imp_df = pl.DataFrame(
        {
            "variable": list(fitted.feature_names_in_),
            "importance": imp.importances_mean.tolist(),
        }
    ).sort("importance")

    # Create horizontal bar plot
    p = (
        ggplot(imp_df, aes(x="variable", y="importance"))
        + geom_bar(stat="identity", fill="slateblue")
        + coord_flip()
        + scale_x_discrete(limits=imp_df["variable"].to_list())
        + labs(x="", y=xlab)
        + ggtitle("Permutation Importance")
        + theme_bw()
        + theme(figure_size=(6, max(5, len(imp_df) * 0.4)))
    )

    if ret:
        return (p, imp_df.sort("importance", descending=True))
    else:
        return (p, None)
