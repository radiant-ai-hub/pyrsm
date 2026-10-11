import os
import re

import numpy as np
import pandas as pd
import polars as pl
import statsmodels as sm
import statsmodels.formula.api as smf
from plotnine import (
    aes,
    geom_abline,
    geom_density,
    geom_errorbarh,
    geom_histogram,
    geom_hline,
    geom_jitter,
    geom_line,
    geom_point,
    geom_smooth,
    geom_text,
    geom_vline,
    ggplot,
    ggtitle,
    labs,
    scale_x_continuous,
    scale_x_log10,
    scale_y_continuous,
    stat_function,
    stat_qq,
    stat_qq_line,
    stat_summary,
    theme,
)
from scipy import stats

from pyrsm.plot_style import ALPHA, FILL, LINE, REF, theme_pyrsm
from scipy.special import expit
from sklearn import metrics
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

from pyrsm.notebook import load_state, save_state
from pyrsm.stats import weighted_mean, weighted_sd
from pyrsm.utils import check_dataframe, expand_grid, ifelse

from .perf import auc


def to_pandas_with_categories(
    pred_data: pl.DataFrame, original_data: pl.DataFrame
) -> pd.DataFrame:
    """
    Convert polars DataFrame to pandas, preserving categorical levels from original data.

    Required for patsy/statsmodels which checks categorical levels match between
    training and prediction data.

    .. deprecated::
        Consider using Polars Enum types instead, which automatically preserve
        all category levels during pandas conversion. See convert_categoricals_to_enum().

    Parameters
    ----------
    pred_data : pl.DataFrame
        The prediction data to convert
    original_data : pl.DataFrame
        The original training data with full categorical levels

    Returns
    -------
    pd.DataFrame with categorical columns having correct levels
    """
    pred_pd = pred_data.to_pandas()
    orig_pd = original_data.to_pandas()

    for col in pred_pd.columns:
        if col in orig_pd.columns and hasattr(orig_pd[col], "cat"):
            pred_pd[col] = pd.Categorical(
                pred_pd[col], categories=orig_pd[col].cat.categories
            )

    return pred_pd


def convert_categoricals_to_enum(
    df: pl.DataFrame, columns: list[str] = None
) -> tuple[pl.DataFrame, dict]:
    """
    Convert categorical/string columns to Enum, preserving level ordering.

    - If column is already Enum: preserves its level order exactly
    - If column is String/Categorical: uses occurrence order (first-seen order)

    This ensures consistent ordering between training and prediction.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    columns : list[str], optional
        Specific columns to convert. If None, converts all string/categorical columns.

    Returns
    -------
    tuple[pl.DataFrame, dict]
        - DataFrame with Enum columns
        - Dictionary mapping column names to their Enum types (for reuse in prediction)

    Examples
    --------
    >>> df = pl.DataFrame({"cut": ["Ideal", "Fair", "Good"], "price": [500.0, 100.0, 200.0]})
    >>> converted, enum_types = convert_categoricals_to_enum(df, ["cut"])
    >>> list(enum_types["cut"].categories)
    ['Ideal', 'Fair', 'Good']
    """
    if columns is None:
        columns = [
            c for c in df.columns if df[c].dtype in (pl.Utf8, pl.String, pl.Categorical)
        ]

    enum_types = {}
    converted = df

    for col in columns:
        if col not in df.columns:
            continue

        col_data = df[col]
        if col_data.dtype in (pl.Utf8, pl.String, pl.Categorical, pl.Enum):
            # Preserve Enum order if already Enum, else use occurrence order
            if isinstance(col_data.dtype, pl.Enum):
                levels = list(col_data.dtype.categories)
            else:
                # Use occurrence order (maintain_order=True preserves first-seen order)
                levels = (
                    col_data.cast(pl.Utf8)
                    .unique(maintain_order=True)
                    .drop_nulls()
                    .to_list()
                )
            enum_type = pl.Enum(levels)
            enum_types[col] = enum_type
            converted = converted.with_columns(
                pl.col(col).cast(pl.Utf8).cast(enum_type).alias(col)
            )

    return converted, enum_types


def apply_enum_types(df: pl.DataFrame, enum_types: dict) -> pl.DataFrame:
    """
    Apply stored Enum types to a DataFrame for prediction.

    Validates that all values in the DataFrame exist in the Enum levels.
    Raises ValueError if unseen levels are encountered.

    Parameters
    ----------
    df : pl.DataFrame
        The prediction data
    enum_types : dict
        Dictionary mapping column names to pl.Enum types (from convert_categoricals_to_enum)

    Returns
    -------
    pl.DataFrame with Enum columns applied

    Raises
    ------
    ValueError
        If prediction data contains values not seen during training

    Examples
    --------
    >>> # First convert original data
    >>> original = pl.DataFrame({"cut": ["Fair", "Good", "Ideal"]})
    >>> _, enum_types = convert_categoricals_to_enum(original, ["cut"])
    >>> # Apply to new data
    >>> pred = pl.DataFrame({"cut": ["Fair", "Good"]})
    >>> result = apply_enum_types(pred, enum_types)
    """
    result = df

    for col, enum_type in enum_types.items():
        if col in df.columns:
            # Check for unseen levels
            current_values = set(df[col].cast(pl.Utf8).unique().drop_nulls().to_list())
            valid_levels = set(enum_type.categories)
            unseen = current_values - valid_levels

            if unseen:
                raise ValueError(
                    f"Column '{col}' contains values not seen during training: {unseen}. "
                    f"Valid levels are: {sorted(valid_levels)}"
                )

            result = result.with_columns(
                pl.col(col).cast(pl.Utf8).cast(enum_type).alias(col)
            )

    return result


def make_train(
    data: pl.DataFrame | pd.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
    strat_var: str | list[str] = None,
    test_size: float = 0.2,
    random_state: int = 1234,
):
    """
    If a stratification variable has been provide, will use stratified sampling on one or more variables to create a training variable for your dataset.
    """
    data = check_dataframe(data)

    if strat_var:
        training_var = np.zeros(data.height)
        strat_values = data.select(pl.concat_str(strat_var, separator="_")).to_series()
        splits = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        for train_index, test_index in splits.split(training_var, strat_values):
            training_var[train_index] = 1
    else:
        training_var = np.random.choice(
            [0, 1], size=data.height, p=[test_size, 1 - test_size]
        )

    return pl.Series(training_var).cast(int)


def extract_evars(model, cn):
    """
    Extract a list of the names of the explanatory variables in a statsmodels model
    """
    pattern = r"\b\w+\b"
    evars = re.findall(pattern, model.formula)[1:]
    evars = [v for v in evars if v in cn]
    return [v for i, v in enumerate(evars) if v not in evars[:i]]


def extract_rvar(model, cn):
    """
    Extract name of the response variable in a statsmodels model
    """
    pattern = r"\b\w+\b"
    return re.findall(pattern, model.formula)[0]


_R_POWER = re.compile(r"I\(\s*([A-Za-z_.][\w.]*)\s*\^\s*([0-9]+)\s*\)")
_PY_POWER = re.compile(r"I\(\s*([A-Za-z_.][\w.]*)\s*\*\*\s*([0-9]+)\s*\)")


def r_powers(term):
    """R's ``I(x^2)`` as the formula engine's ``I(x ** 2)``.

    In a Python formula ``^`` is bitwise XOR, so radiant.model's squared and
    cubed terms (``qterms``) must be rewritten before estimation. Applies to a
    single term, a list of terms, or a whole formula.
    """
    if term is None:
        return None
    if isinstance(term, (list, tuple)):
        return [r_powers(t) for t in term]
    return _R_POWER.sub(lambda m: f"I({m.group(1)} ** {m.group(2)})", str(term))


def r_label(term: str) -> str:
    """``I(x ** 2)`` back to R's ``I(x^2)``, for labels a student reads."""
    return _PY_POWER.sub(lambda m: f"I({m.group(1)}^{m.group(2)})", str(term))


def check_pred_vars(evar, data=None, cmd=None, data_cmd=None) -> None:
    """
    Refuse a prediction request that does not fit the model, with Radiant's message

    Parameters
    ----------
    evar : list[str]
        Explanatory variables in the model
    data : DataFrame, optional
        Prediction data. Every explanatory variable must be a column.
    cmd, data_cmd : dict, optional
        Prediction command(s). Every name must be an explanatory variable.

    Raises
    ------
    ValueError
        Naming the variables in the model and the ones that are missing
    """
    evar = list(evar or [])
    if data is not None:
        columns = list(data.columns)
        missing = [v for v in evar if v not in columns]
        if missing:
            raise ValueError(
                "All variables in the model must also be in the prediction data\n"
                f"Variables in the model: {', '.join(evar)}\n"
                f"Variables not available in prediction data: {', '.join(missing)}"
            )
    for command in (cmd, data_cmd):
        if not command or not isinstance(command, dict):
            continue
        extra = [k for k in command if k not in evar]
        if extra:
            raise ValueError(
                "The command entered contains variable names that are not in the model\n"
                f"Variables in the model: {', '.join(evar)}\n"
                f"Variables not available in prediction data: {', '.join(extra)}"
            )


def convert_to_list(v):
    if v is None:
        return []
    elif isinstance(v, list):
        return v
    elif isinstance(v, str):
        return [v]
    else:
        return list(v)


def convert_binary(data, rvar, lev):
    """Convert response variable to binary (0/1) based on level. Works with polars DataFrame."""
    cb = (
        pl.when(pl.col(rvar) == lev)
        .then(1)
        .when(pl.col(rvar).is_null())
        .then(None)
        .otherwise(0)
    )
    return data.with_columns(cb.alias(rvar))


def is_binary(series):
    """
    Efficiently check if a series has more than 2 unique values.
    Works with both polars Series and pandas Series.
    """
    if isinstance(series, pl.Series):
        return series.drop_nulls().n_unique() > 2
    else:
        # pandas Series
        return series.dropna().nunique() > 2


def check_binary(data, rvar, lev):
    """Check and convert binary response variable. Works with polars DataFrame."""
    unique_levels = data[rvar].unique().to_list()
    data = convert_binary(data, rvar, lev)
    rvar_sum = data[rvar].sum()
    if rvar_sum == 0 or rvar_sum == data.height:
        raise ValueError(
            f"All converted response values are {1 if rvar_sum == data.height else 0}. "
            f"Available levels in {rvar}: {unique_levels} and '{lev}' was selected."
        )
    else:
        return data


def get_dummies(
    df: pd.DataFrame | pl.DataFrame,
    drop_first: bool = True,
    drop_nonvarying: bool = True,
    categories: dict[str, list] | None = None,
) -> pl.DataFrame:
    """
    Create dummy variables preserving category ordering.

    - If column is Enum: uses Enum level order
    - If column is String/Categorical: uses occurrence order (first-seen order)

    Numeric columns are placed first, dummy columns appended at end.

    Parameters
    ----------
    df : DataFrame
        Input data (pandas or polars)
    drop_first : bool
        Drop first category (default True)
    drop_nonvarying : bool
        If False, preserve all categories even if not present in data.
        Requires categories dict to know which categories to preserve.
    categories : dict[str, list] | None
        Category lists per column from training. Used during prediction.

    Returns
    -------
    pl.DataFrame with dummy columns appended after non-categorical columns
    """
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    cat_cols = [
        c for c in df.columns if df[c].dtype in (pl.Utf8, pl.Categorical, pl.Enum)
    ]

    if not cat_cols:
        return df

    # Get non-categorical columns (keep at front, like pandas)
    non_cat_cols = [c for c in df.columns if c not in cat_cols]
    result = df.select(non_cat_cols) if non_cat_cols else pl.DataFrame()

    # When categories provided (prediction mode), create dummies directly
    if categories:
        for col in cat_cols:
            if col in categories:
                # Create dummy columns for each category in stored list
                for cat in categories[col]:
                    col_name = f"{col}_{cat}"
                    dummy_col = (
                        (df[col].cast(pl.Utf8) == cat).cast(pl.UInt8).alias(col_name)
                    )
                    result = result.with_columns(dummy_col)
            else:
                # Column not in categories - use Enum or occurrence order
                col_dtype = df[col].dtype
                if isinstance(col_dtype, pl.Enum):
                    cats = list(col_dtype.categories)
                else:
                    cats = (
                        df[col]
                        .cast(pl.Utf8)
                        .unique(maintain_order=True)
                        .drop_nulls()
                        .to_list()
                    )
                cats_to_use = cats[1:] if drop_first else cats
                for cat in cats_to_use:
                    col_name = f"{col}_{cat}"
                    dummy_col = (
                        (df[col].cast(pl.Utf8) == cat).cast(pl.UInt8).alias(col_name)
                    )
                    result = result.with_columns(dummy_col)
        return result

    # Training mode: use Enum or occurrence order
    for col in cat_cols:
        col_dtype = df[col].dtype
        if isinstance(col_dtype, pl.Enum):
            cats = list(col_dtype.categories)
        else:
            cats = (
                df[col].cast(pl.Utf8).unique(maintain_order=True).drop_nulls().to_list()
            )
        cats_to_use = cats[1:] if drop_first else cats
        for cat in cats_to_use:
            col_name = f"{col}_{cat}"
            dummy_col = (df[col].cast(pl.Utf8) == cat).cast(pl.UInt8).alias(col_name)
            result = result.with_columns(dummy_col)

    return result


def conditional_get_dummies(
    df: pd.DataFrame | pl.DataFrame,
    drop_nonvarying: bool = True,
    categories: dict[str, list] | None = None,
) -> pl.DataFrame:
    """
    Create dummy variables conditionally (drop_first only for 3+ categories).

    Parameters
    ----------
    df : DataFrame
        Input data (pandas or polars)
    drop_nonvarying : bool
        If False, preserve all categories even if not present in data.
        Requires categories dict to know which categories to preserve.
    categories : dict[str, list] | None
        Full category lists per column. Used when drop_nonvarying=False.

    Returns
    -------
    pl.DataFrame with dummy columns replacing categorical columns
    """
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    cat_cols = [
        c for c in df.columns if df[c].dtype in (pl.Utf8, pl.Categorical, pl.Enum)
    ]

    if not cat_cols:
        return df

    # Split into binary (drop first) and non-binary (keep all)
    binary_cols = []
    nonbinary_cols = []
    for col in cat_cols:
        n_cats = df[col].n_unique()
        if n_cats <= 2:
            binary_cols.append(col)
        else:
            nonbinary_cols.append(col)

    # Get non-categorical columns
    non_cat_cols = [c for c in df.columns if c not in cat_cols]
    result = df.select(non_cat_cols) if non_cat_cols else pl.DataFrame()

    # Process binary columns (keep all categories)
    if binary_cols:
        binary_result = get_dummies(
            df.select(binary_cols),
            drop_first=True,
            drop_nonvarying=drop_nonvarying,
            categories={
                k: v for k, v in (categories or {}).items() if k in binary_cols
            },
        )
        result = (
            pl.concat([result, binary_result], how="horizontal")
            if result.width > 0
            else binary_result
        )

    # Process non-binary columns (drop first)
    if nonbinary_cols:
        nonbinary_result = get_dummies(
            df.select(nonbinary_cols),
            drop_first=False,
            drop_nonvarying=drop_nonvarying,
            categories={
                k: v for k, v in (categories or {}).items() if k in nonbinary_cols
            },
        )
        result = (
            pl.concat([result, nonbinary_result], how="horizontal")
            if result.width > 0
            else nonbinary_result
        )

    return result


def sig_stars(pval) -> pl.Series:
    """
    pval: list of floats/None or a pl.Series
    returns: pl.Series of significance symbols (strings)
    """
    # Build a DataFrame so we can use the expression API
    df = pl.DataFrame({"pval": pval}).fill_nan(1)

    # NaN -> 1.0 (like np.nan_to_num(..., nan=1.0)), null stays null
    # Then map to significance stars using when/then/otherwise
    return df.with_columns(
        pl.when(pl.col("pval") < 0.001)
        .then(pl.lit("***"))
        .when(pl.col("pval") < 0.01)
        .then(pl.lit("**"))
        .when(pl.col("pval") < 0.05)
        .then(pl.lit("*"))
        .when(pl.col("pval") < 0.1)
        .then(pl.lit("."))
        .otherwise(pl.lit(" "))
        .alias("sig")
    )["sig"]


def get_mfit(fitted) -> tuple[dict | None, str | None]:
    """
    Get model fit statistics

    Parameters
    ----------
    fitted : A fitted linear or logistic 4egression model
    """

    mfit_dct = None
    model_type = None
    if isinstance(fitted, sm.genmod.generalized_linear_model.GLMResultsWrapper):
        fw = None
        if fitted.model._has_freq_weights:
            fw = fitted.model.freq_weights

        # gets same results as in R
        lrtest = -2 * (fitted.llnull - fitted.llf)
        mfit_dct = {
            "pseudo_rsq_mcf": [1 - fitted.llf / fitted.llnull],
            "pseudo_rsq_mcf_adj": [1 - (fitted.llf - fitted.df_model) / fitted.llnull],
            "AUC": [auc(fitted.model.endog, fitted.fittedvalues, weights=fw)],
            "log_likelihood": fitted.llf,
            "AIC": [fitted.aic],
            "BIC": [fitted.bic_llf],
            # "chisq": [fitted.pearson_chi2],
            "chisq": [lrtest],
            "chisq_df": [fitted.df_model],
            # "chisq_pval": [1 - stats.chi2.cdf(fitted.pearson_chi2, fitted.df_model)],
            "chisq_pval": [stats.chi2.sf(lrtest, fitted.df_model)],
            "nobs": [fitted.nobs],
        }

        model_type = "logistic"

    elif isinstance(fitted, sm.regression.linear_model.RegressionResultsWrapper):
        mfit_dct = {
            "rsq": [fitted.rsquared],
            "rsq_adj": [fitted.rsquared_adj],
            "fvalue": [fitted.fvalue],
            "ftest_df_model": [fitted.df_model],
            "ftest_df_resid": [fitted.df_resid],
            "ftest_pval": [fitted.f_pvalue],
            "nobs": [fitted.nobs],
        }

        model_type = "regression"

    return mfit_dct, model_type


def or_ci(fitted, alpha=0.05, intercept=False, importance=False, data=None, dec=3):
    """
    Confidence interval for Odds ratios

    Parameters
    ----------
    fitted : A fitted logistic regression model
    alpha : float
        Significance level
    intercept : bool
        Include intercept in output (True or False)
    importance : int
        Calculate variable importance. Only meaningful if data
        used in estimation was standardized prior to model
        estimation
    data : Pandas dataframe
        Unstandardized data used to calculate descriptive
        statistics
    dec : int
        Number of decimal places to use in rounding

    Returns
    -------
    Pandas dataframe with Odd-ratios and confidence intervals
    """

    df = pd.DataFrame(np.exp(fitted.params), columns=["OR"]).dropna()
    df["OR%"] = 100 * ifelse(df["OR"] < 1, -(1 - df["OR"]), df["OR"] - 1)

    low, high = [100 * alpha / 2, 100 * (1 - (alpha / 2))]
    df[[f"{low}%", f"{high}%"]] = np.exp(fitted.conf_int(alpha=alpha))
    df["p.values"] = ifelse(fitted.pvalues < 0.001, "< .001", fitted.pvalues.round(dec))
    df["  "] = sig_stars(fitted.pvalues)
    df["OR%"] = [f"{round(o, max(dec - 2, 0))}%" for o in df["OR%"]]
    df = df.reset_index()

    if importance:
        df["dummy"] = df["index"].str.contains("[T", regex=False)
        df["importance"] = (
            pd.DataFrame().assign(OR=df["OR"], ORinv=1 / df["OR"]).max(axis=1)
        )

    if isinstance(data, pd.DataFrame):
        # using a fake response variable variable
        data = data.assign(__rvar__=1).copy()
        form = "__rvar__ ~ " + fitted.model.formula.split("~", 1)[1]
        exog = pd.DataFrame(smf.logit(formula=form, data=data).exog)
        weights = fitted._freq_weights
        if sum(weights) > len(weights):

            def wmean(x):
                return weighted_mean(pd.DataFrame(x), weights)[0]

            def wstd(x):
                return weighted_sd(pd.DataFrame(x), weights)[0]

            df = pd.concat(
                [df, exog.apply([wmean, wstd, "min", "max"]).T],
                axis=1,
            )
        else:
            df = pd.concat([df, exog.apply(["mean", "std", "min", "max"]).T], axis=1)

    if intercept is False:
        df = df.loc[df["index"] != "Intercept"]

    return df.round(dec)


def or_plot(fitted, alpha=0.05, intercept=False, incl=None, excl=None, figsize=None):
    """
    Odds ratio plot

    Parameters
    ----------
    fitted : A fitted logistic regression model
    alpha : float
        Significance level
    intercept : bool
        Include intercept in odds-ratio plot (True or False)
    incl : str or list of strings
        Variables to include in the odds-ratio plot. All will be included by default
    excl : str or list of strings
        Variables to exclude from the odds-ratio plot. None are excluded by default

    Returns
    -------
    plotnine.ggplot
        Plot of Odds ratios
    """

    # iloc to reverse order
    df = or_ci(fitted, alpha=alpha, intercept=intercept, dec=100).dropna().iloc[::-1]

    if incl is not None:
        incl = ifelse(isinstance(incl, str), [incl], incl)
        rx = "(" + "|".join([f"^{v}$|^{v}\\[" for v in incl]) + ")"
        incl = df["index"].str.match(rf"{rx}")
        if intercept:
            incl[0] = True
        df = df[incl]

    if excl is not None and excl != []:
        excl = ifelse(isinstance(excl, str), [excl], excl)
        rx = "(" + "|".join([f"^{v}$|^{v}\\[" for v in excl]) + ")"
        excl = df["index"].str.match(rf"{rx}")
        if intercept:
            excl[0] = False
        df = df[~excl]

    low, high = [100 * alpha / 2, 100 * (1 - (alpha / 2))]

    # Add columns for error bar bounds
    df["xmin"] = df[f"{low}%"]
    df["xmax"] = df[f"{high}%"]
    df["index"] = df["index"].str.replace("[T.", "[")

    # Create ordered categorical for y-axis (maintain row order)
    df["index"] = pd.Categorical(
        df["index"], categories=df["index"].tolist(), ordered=True
    )

    return (
        ggplot(df, aes(x="OR", y="index"))
        + geom_vline(xintercept=1, linetype="dashdot", color=REF)
        + geom_errorbarh(aes(xmin="xmin", xmax="xmax"), height=0.2, color=LINE)
        + geom_point(color=LINE)
        + scale_x_log10()
        + labs(x="Odds-ratio", y="")
        + theme_pyrsm()
    )


def vif(fitted, dec=3):
    """
    Calculate the Variance Inflation Factor (VIF) associated with each
    exogenous variable

    Status
    ------
    WIP port of VIF calculation from R's car:::vif.default to Python

    Parameters
    ----------
    fitted : A fitted (logistic) regression model
    dec : int
        Number of decimal places to use in rounding

    Returns
    -------
    Pandas dataframe sorted by VIF score
    """

    if hasattr(fitted, "model"):
        model = fitted.model
    else:
        # legacy for when only an un-fitted model was accepted
        model = fitted

    is_categorical = {
        item.split("[T.")[0]: ("[T." in item) for item in model.exog_names
    }
    if sum(is_categorical.values()) > 0:
        evar = list(is_categorical.keys())[1:]
        exog = model.exog[:, 1:]
        Xcorr = np.linalg.det(np.corrcoef(exog, rowvar=False))
        df = pd.DataFrame(exog, columns=model.exog_names[1:])
        vif = []
        for col, cat in is_categorical.items():
            if col == "Intercept":
                continue
            elif cat:
                select = [f"{col}[T." in c for c in df.columns]
                Vcorr = np.linalg.det(df.loc[:, select].corr().values)
                drop = [f"{col}[T." not in c for c in df.columns]
                Ocorr = np.linalg.det(df.loc[:, drop].corr().values)
            else:
                Vcorr = 1
                Ocorr = np.linalg.det(df.drop(col, axis=1).corr().values)

            vif.append(Vcorr * Ocorr / Xcorr)
        df = pd.DataFrame(evar, columns=["index"])
    else:
        vif = [
            variance_inflation_factor(model.exog, i) for i in range(model.exog.shape[1])
        ]
        df = pd.DataFrame(model.exog_names, columns=["index"])

    df["vif"] = vif
    df["Rsq"] = 1 - 1 / df["vif"]

    if "Intercept" in model.exog_names:
        df = df[df["index"] != "Intercept"]

    df = df.sort_values("vif", ascending=False).set_index("index")
    df.index.name = None

    if dec is not None:
        df = df.round(dec)

    return df


def gvif(fitted, dec: int | None = 3) -> pd.DataFrame:
    """
    Generalized variance inflation factors as ``car::vif``

    Computed from the correlation matrix of the coefficient estimates (the
    model's covariance without the intercept), so it is right for weighted
    and GLM fits as well as OLS. One row per term: ``vif`` is the GVIF and
    ``Rsq`` = 1 - 1/GVIF, sorted from largest to smallest.
    """
    design = fitted.model.data.design_info
    names = list(fitted.params.index)
    cov = np.asarray(fitted.cov_params())
    keep = [i for i, n in enumerate(names) if n != "Intercept"]
    cov = cov[np.ix_(keep, keep)]
    sd = np.sqrt(np.diag(cov))
    R = cov / np.outer(sd, sd)
    det_all = np.linalg.det(R)
    position = {names[i]: j for j, i in enumerate(keep)}
    rows = []
    for term, sl in design.term_name_slices.items():
        if term == "Intercept":
            continue
        cols = [position[names[i]] for i in range(sl.start, sl.stop) if names[i] in position]
        rest = [j for j in range(len(keep)) if j not in cols]
        a = np.linalg.det(R[np.ix_(cols, cols)])
        b = np.linalg.det(R[np.ix_(rest, rest)]) if rest else 1.0
        rows.append((r_label(term), a * b / det_all))
    df = pd.DataFrame(rows, columns=["index", "vif"]).set_index("index")
    df["Rsq"] = 1 - 1 / df["vif"]
    df = df.sort_values("vif", ascending=False)
    df.index.name = None
    return df.round(dec) if dec is not None else df


def profile_ci(fitted, alpha: float = 0.05) -> pd.DataFrame:
    """
    Profile-likelihood confidence intervals for a GLM, as R's ``confint(glm)``

    For each coefficient the interval holds the values b at which the
    deviance of the model refit with that coefficient fixed at b (the other
    coefficients re-estimated) exceeds the minimum deviance by at most the
    chi-squared(1) quantile -- equivalently the signed root of the deviance
    difference lies within +/- z(1 - alpha/2).

    Returns
    -------
    pandas DataFrame with the lower and upper bound, indexed by coefficient
    """
    from scipy.optimize import brentq
    from statsmodels.genmod.generalized_linear_model import GLM

    model = fitted.model
    y, X = model.endog, model.exog
    w = getattr(model, "freq_weights", None)
    offset = getattr(model, "offset", None)
    base_offset = np.zeros(len(y)) if offset is None else np.asarray(offset)
    dev_hat = fitted.deviance
    q = stats.norm.ppf(1 - alpha / 2) ** 2
    params = np.asarray(fitted.params)
    bse = np.asarray(fitted.bse)

    def excess(j: int, b: float, start) -> float:
        others = np.delete(X, j, axis=1)
        off = base_offset + X[:, j] * b
        if others.shape[1] == 0:
            mu = model.family.link.inverse(off)
            dev = model.family.deviance(y, mu, freq_weights=w if w is not None else 1.0)
        else:
            fit = GLM(y, others, family=model.family, offset=off, freq_weights=w).fit(
                start_params=start
            )
            dev = fit.deviance
        return (dev - dev_hat) - q

    out = []
    for j in range(len(params)):
        start = np.delete(params, j)
        bounds = []
        for direction in (-1, 1):
            step = max(bse[j], 1e-8) * np.sqrt(q)
            edge = params[j] + direction * step
            for _ in range(60):
                if excess(j, edge, start) > 0:
                    break
                step *= 2
                edge = params[j] + direction * step
            lo, hi = sorted([params[j], edge])
            bounds.append(brentq(lambda b: excess(j, b, start), lo, hi, xtol=1e-10))
        out.append(bounds)
    return pd.DataFrame(out, index=fitted.params.index)


def ols_interval(fitted, df, conf=0.95, interval="confidence"):
    """
    Predictions of a linear regression with a confidence or prediction interval

    As R's ``predict.lm(..., interval=)``: the t distribution with the
    model's residual degrees of freedom.

    Parameters
    ----------
    fitted : OLS model fitted using the statsmodels formula interface
    df : Pandas dataframe with input data for prediction
    conf : float
        Confidence level (0-1). Default is 0.95
    interval : str
        "confidence": an interval around the mean of the response at these
        values of the explanatory variables. "prediction": an interval around a
        single new value of the response, which adds the residual variance

    Returns
    -------
    Pandas DataFrame with the prediction and the lower and upper bounds
    """
    if conf < 0 or conf > 1:
        raise ValueError(
            "Confidence level (conf) must be a numeric value between 0 and 1"
        )
    low, high = (1.0 - conf) / 2.0, 1.0 - (1.0 - conf) / 2.0
    frame = fitted.get_prediction(df).summary_frame(alpha=1 - conf)
    side = "obs" if interval == "prediction" else "mean"
    return pd.DataFrame(
        {
            "prediction": frame["mean"].to_numpy(),
            f"{low * 100:.2f}%": frame[f"{side}_ci_lower"].to_numpy(),
            f"{high * 100:.2f}%": frame[f"{side}_ci_upper"].to_numpy(),
        }
    )


def predict_ci(fitted, df, conf=0.95, alpha=None):
    """
    Compute predicted probabilities with confidence intervals based on a
    logistic regression model

    Parameters
    ----------
    fitted : Logistic regression model fitted using the statsmodels formula interface
    df : Pandas dataframe with input data for prediction
    conf : float
        Confidence level (0-1). Default is 0.95
    alpha : float
        Significance level (0-1). Default is 0.05

    Returns
    -------
    Pandas DataFrame with probability predictions and lower and upper confidence bounds

    Example
    -------
    import numpy as np
    import statsmodels.formula.api as smf
    import pandas as pd

    # simulate data
    np.random.seed(1)
    x1 = np.arange(100)
    x2 = pd.Series(["a", "b", "c", "a"], dtype="category").sample(100, replace=True)
    y = (x1 * 0.5 + np.random.normal(size=100, scale=10) > 30).astype(int)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    # estimate the model
    model = smf.logit(formula="y ~ x1 + x2", data=df).fit()
    model.summary()
    pred = predict_ci(model, df)

    from plotnine import aes, geom_line, ggplot
    pdat = pred.assign(x1=x1)
    (
        ggplot(pdat, aes(x="x1"))
        + geom_line(aes(y="prediction"))
        + geom_line(aes(y="2.5%"), linetype="dashed")
        + geom_line(aes(y="97.5%"), linetype="dashed")
    )
    """

    if conf < 0 or conf > 1:
        raise ValueError(
            "Confidence level (conf) must be a numeric value between 0 and 1"
        )

    if alpha is not None:
        raise ValueError(
            "The alpha argument has been deprecated. Use the confidence level (conf) instead (1-alpha)."
        )

    # generate predictions
    prediction = fitted.predict(df)

    # set up the data in df in the same was as the exog data
    # that is part of fitted.model.exog
    # use a fake response variable
    df = df.assign(__rvar__=1).copy()
    form = "__rvar__ ~ " + fitted.model.formula.split("~", 1)[1]
    exog = smf.logit(formula=form, data=df).exog

    low, high = [(1.0 - conf) / 2.0, (1.0 - (1.0 - conf) / 2.0)]
    Xb = np.dot(exog, fitted.params)
    se = np.sqrt((exog.dot(fitted.cov_params()) * exog).sum(-1))
    is_glm = isinstance(fitted, sm.genmod.generalized_linear_model.GLMResultsWrapper)
    if not is_glm and isinstance(fitted, sm.regression.linear_model.RegressionResultsWrapper):
        # R's predict.lm: t distribution with the residual df (a GLM's
        # wrapper subclasses the OLS one, so check it first)
        me = stats.t.ppf(high, fitted.df_resid) * se
    else:
        me = stats.norm.ppf(high) * se

    if isinstance(fitted, sm.genmod.generalized_linear_model.GLMResultsWrapper):
        return pd.DataFrame(
            {
                "prediction": prediction,
                f"{low * 100:.2f}%": expit(Xb - me),
                f"{high * 100:.2f}%": expit(Xb + me),
            }
        )
    elif isinstance(fitted, sm.regression.linear_model.RegressionResultsWrapper):
        return pd.DataFrame(
            {
                "prediction": prediction,
                f"{low * 100:.2f}%": Xb - me,
                f"{high * 100:.2f}%": Xb + me,
            }
        )


def nobs_dropped(obj):
    if hasattr(obj, "nobs_dropped") and obj.nobs_dropped > 0:
        return f" ({obj.nobs_dropped:,.0f} obs. dropped)"
    else:
        return ""


def model_fit(fitted, dec: int = 3, prn: bool = True) -> str | pd.DataFrame:
    """
    Compute various model fit statistics for a fitted linear or logistic regression model

    Parameters
    ----------
    fitted : statmodels ols or glm object
        Regression model fitted using statsmodels
    dec : int
        Number of decimal places to use in rounding
    prn : bool
        If True, print output, else return a Pandas dataframe with the results

    Returns
    -------
        If prn is True, print output, else return a Pandas dataframe with the results
    """
    if hasattr(fitted, "df_resid"):
        weighted_nobs = (
            fitted.df_resid + fitted.df_model + int(hasattr(fitted.params, "Intercept"))
        )
        if weighted_nobs > fitted.nobs:
            fitted.nobs = weighted_nobs
    mfit_dct, model_type = get_mfit(fitted)
    if not mfit_dct:
        return "Only linear and logistic regression models are currently supported."

    mfit = pd.DataFrame(mfit_dct)

    if prn:
        if model_type == "logistic":
            output = f"""Pseudo R-squared (McFadden): {mfit.pseudo_rsq_mcf.values[0].round(dec)}
Pseudo R-squared (McFadden adjusted): {mfit.pseudo_rsq_mcf_adj.values[0].round(dec)}
Area under the RO Curve (AUC): {mfit.AUC.values[0].round(dec)}
Log-likelihood: {mfit.log_likelihood.values[0].round(dec)}, AIC: {mfit.AIC.values[0].round(dec)}, BIC: {mfit.BIC.values[0].round(dec)}
Chi-squared: {mfit.chisq.values[0].round(dec)}, df({mfit.chisq_df.values[0]}), p.value {np.where(mfit.chisq_pval.values[0] < 0.001, "< 0.001", mfit.chisq_pval.values[0].round(dec))}
Nr obs: {mfit.nobs.values[0]:,.0f}{nobs_dropped(fitted)}"""
        elif model_type == "regression":
            output = f"""R-squared: {mfit.rsq.values[0].round(dec)}, Adjusted R-squared: {mfit.rsq_adj.values[0].round(dec)}
F-statistic: {mfit.fvalue[0].round(dec)} df({mfit.ftest_df_model.values[0]:.0f}, {mfit.ftest_df_resid.values[0]:.0f}), p.value {np.where(mfit.ftest_pval.values[0] < 0.001, "< 0.001", mfit.ftest_pval.values[0].round(dec))}
Nr obs: {mfit.nobs.values[0]:,.0f}{nobs_dropped(fitted)}"""
        else:
            output = "Model type not supported"
        return output
    else:
        return mfit


def coef_plot(
    fitted,
    alpha: float = 0.05,
    intercept: bool = False,
    incl: str = None,
    excl: list = [],
    figsize: tuple = None,
):
    """
    Coefficient plot

    Parameters
    ----------
    fitted : A fitted linear regression model
    alpha : float
        Significance level
    intercept : bool
        Include intercept in coefficient plot (True or False)
    incl : str or list of strings
        Variables to include in the coefficient plot. All will be included by default
    excl : str or list of strings
        Variables to exclude from the coefficient plot. None are excluded by default

    Returns
    -------
    plotnine.ggplot
        Coefficient plot
    """
    df = fitted.conf_int(alpha=alpha).reset_index().iloc[::-1]
    df["coefficient"] = fitted.params[df["index"]].dropna().values
    # columns dropped for perfect multicollinearity have no estimate (R: NA)
    df = df[~df["index"].isin(getattr(fitted, "aliased", []))]

    if not intercept:
        df = df.query('index != "Intercept"')

    if incl is not None:
        incl = ifelse(isinstance(incl, str), [incl], incl)
        rx = "(" + "|".join([f"^\b{v}|^{v}\\[" for v in incl]) + ")"
        incl = df["index"].str.match(rf"{rx}")
        if intercept:
            incl[0] = True
        df = df[incl]

    if len(excl) > 0 and excl is not None:
        excl = ifelse(isinstance(excl, str), [excl], excl)
        rx = "(" + "|".join([f"^\b{v}|^{v}\\[" for v in excl]) + ")"
        excl = df["index"].str.match(rf"{rx}")
        if intercept:
            excl[0] = False
        df = df[~excl]

    low, high = [100 * alpha / 2, 100 * (1 - (alpha / 2))]
    df.columns = ["index", f"{low}%", f"{high}%", "coefficient"]

    # Add columns for error bar bounds
    df["xmin"] = df[f"{low}%"]
    df["xmax"] = df[f"{high}%"]
    df["index"] = df["index"].str.replace("[T.", "[")

    # Create ordered categorical for y-axis (maintain row order)
    df["index"] = pd.Categorical(
        df["index"], categories=df["index"].tolist(), ordered=True
    )

    return (
        ggplot(df, aes(x="coefficient", y="index"))
        + geom_vline(xintercept=0, linetype="dashdot", color=REF)
        + geom_errorbarh(aes(xmin="xmin", xmax="xmax"), height=0.2, color=LINE)
        + geom_point(color=LINE)
        + labs(x="Coefficient", y="")
        + theme_pyrsm()
    )


def coef_ci(fitted, alpha: float = 0.05, intercept: bool = True, dec: int = 3):
    """
    Confidence interval for coefficient from linear regression

    Parameters
    ----------
    fitted : A fitted linear regression model
    alpha : float
        Significance level
    intercept : bool
        Include intercept in the output (True or False)
    dec : int
        Number of decimal places to use in rounding

    Returns
    -------
    Pandas dataframe with regression coefficients and confidence intervals
    """

    df = pd.DataFrame({"coefficient": fitted.params})
    aliased = list(getattr(fitted, "aliased", []))

    low, high = [round(100 * alpha / 2, 4), round(100 * (1 - (alpha / 2)), 4)]
    df[[f"{low:g}%", f"{high:g}%"]] = fitted.conf_int(alpha=alpha)

    if dec is None:
        df["p.values"] = ifelse(fitted.pvalues < 0.001, "< .001", fitted.pvalues)
    else:
        df = df.round(dec)
        df["p.values"] = ifelse(
            fitted.pvalues < 0.001, "< .001", fitted.pvalues.round(dec)
        )

    df["  "] = sig_stars(fitted.pvalues)
    if intercept is False:
        df = df[df.index != "Intercept"]

    return df[~df.index.isin(aliased)]


def evalreg(df, rvar: str, pred: str, dec: int = 3):
    """
    Evaluate regression models. Calculates R-squared, MSE, and MAE

    Parameters
    ----------
    df : DataFrame (polars or pandas) or a dictionary of DataFrames with keys to show results for
        multiple model predictions and datasets (training and test)
    rvar : str
        Name of the response variable column in df
    pred : str
        Name of the column, of list of column names, in df with model predictions
    dec : int
        Number of decimal places to use in rounding

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with evaluation metrics
    """

    dct = ifelse(isinstance(df, dict), df, {"All": df})
    pred = ifelse(isinstance(pred, str), [pred], pred)

    def calculate_metrics(key, dfm, pm):
        dfm = check_dataframe(dfm)
        n = dfm.height
        y_true = dfm[rvar].to_numpy()
        y_pred = dfm[pm].to_numpy()
        return pl.DataFrame(
            {
                "Type": [key],
                "predictor": [pm],
                "n": [n],
                "r2": [metrics.r2_score(y_true, y_pred)],
                "mse": [metrics.mean_squared_error(y_true, y_pred)],
                "mae": [metrics.mean_absolute_error(y_true, y_pred)],
            }
        )

    result = pl.concat(
        [calculate_metrics(key, val, p) for key, val in dct.items() for p in pred]
    )
    # Round numeric columns
    numeric_cols = [c for c in result.columns if result[c].dtype == pl.Float64]
    return result.with_columns([pl.col(c).round(dec) for c in numeric_cols])


def _fit_line(method: str):
    """A dashed fit line (``method`` is "lm" or "loess") in the reference color."""
    return geom_smooth(method=method, se=False, color=REF, linetype="dashdot", size=0.75)


def reg_dashboard(fitted, nobs: int = 1000, lines: list[str] | None = None):
    """
    Plot regression residual dashboard

    Parameters
    ----------
    fitted : Object with fitted values and residuals
    nobs: int
        Number of observations to use for plots.
        Set to None or -1 to plot all values.
        The Residuals vs Order plot will only be valid
        if all observations are plotted
    lines : list[str] or None
        None keeps the default linear fit lines. Otherwise the lines to add,
        as in Radiant: "line" adds a 45-degree line to the Actual vs Fitted
        and Q-Q plots and a linear fit to the residual plots; "loess" adds a
        local polynomial fit to the first three plots. ``[]`` adds none.
    """
    if hasattr(fitted.model, "endog"):
        endog = fitted.model.endog
    else:
        endog = fitted.model["endog"]

    data = pl.DataFrame(
        {
            "fitted": fitted.fittedvalues,
            "actual": endog,
            "resid": fitted.resid,
            "std_resid": fitted.resid / fitted.resid.std(),
            "order": np.arange(endog.shape[0]),
        }
    )

    if (nobs != -1 and nobs is not None) and (nobs < data.height):
        data = data.sample(nobs, seed=1234)

    default_lines = lines is None
    lines = [] if lines is None else list(lines)

    # Plot 1: Actual vs Fitted values
    p1 = (
        ggplot(data, aes(x="fitted", y="actual"))
        + geom_point(alpha=ALPHA, color=FILL)
        + labs(x="Fitted values", y="Actual values")
        + ggtitle("Actual vs Fitted values")
        + theme_pyrsm()
    )

    # Plot 2: Residuals vs Fitted values
    p2 = (
        ggplot(data, aes(x="fitted", y="resid"))
        + geom_point(alpha=ALPHA, color=FILL)
        + geom_hline(yintercept=0, linetype="dashed", color=REF)
        + labs(x="Fitted values", y="Residuals")
        + ggtitle("Residuals vs Fitted values")
        + theme_pyrsm()
    )

    # Plot 3: Residuals vs Row order
    p3 = (
        ggplot(data, aes(x="order", y="resid"))
        + geom_line(color=LINE)
        + geom_hline(yintercept=0, linetype="dashed", color=REF)
        + labs(x="Row order", y="Residuals")
        + ggtitle("Residuals vs Row order")
        + theme_pyrsm()
    )

    # Plot 4: Q-Q plot
    p4 = (
        ggplot(data, aes(sample="std_resid"))
        + stat_qq(color=LINE)
        + labs(x="Theoretical quantiles", y="Sample quantiles")
        + ggtitle("Normal Q-Q plot")
        + theme_pyrsm()
    )

    if default_lines:
        p1 = p1 + geom_smooth(method="lm", color=REF)
        p2 = p2 + geom_smooth(method="lm", color=REF)
        p4 = p4 + stat_qq_line(color=REF)
    else:
        if "line" in lines:
            p1 = p1 + geom_abline(linetype="dashdot", color=REF)
            p2 = p2 + _fit_line("lm")
            p3 = p3 + _fit_line("lm")
            p4 = p4 + stat_qq_line(color=REF, linetype="dashdot")
        if "loess" in lines:
            p1 = p1 + _fit_line("loess")
            p2 = p2 + _fit_line("loess")
            p3 = p3 + _fit_line("loess")

    # Plot 5: Histogram of residuals
    p5 = (
        ggplot(data, aes(x="resid"))
        + geom_histogram(fill=FILL, color="white", bins=30)
        + labs(x="Residuals", y="Count")
        + ggtitle("Histogram of residuals")
        + theme_pyrsm()
    )

    # Plot 6: Residuals vs Normal density
    p6 = (
        ggplot(data, aes(x="std_resid"))
        + geom_density(fill=FILL, alpha=ALPHA)
        + stat_function(fun=stats.norm.pdf, color=REF, size=1)
        + labs(x="Standardized residuals", y="Density")
        + ggtitle("Residuals vs Normal density")
        + theme_pyrsm()
    )

    # Compose into 3x2 grid and set figure size
    dashboard = (p1 | p2) / (p3 | p4) / (p5 | p6) + theme(figure_size=(10, 12))
    return dashboard


def logit_dashboard_data(fitted, qnt: int = 10) -> dict[str, pl.DataFrame | float]:
    """
    The numbers behind ``logit_dashboard``.

    Parameters
    ----------
    fitted : statsmodels GLM (Binomial) results, as stored on ``pyrsm.model.logistic``
    qnt : int
        Number of predicted-probability bins for the calibration and gains views.

    Returns
    -------
    dict with
        calibration : per bin, mean predicted probability, observed rate and n
        roc : false and true positive rates per threshold
        gains : cumulative share of observations vs cumulative share of positives
        predictions : predicted probability and actual outcome per observation
        auc : area under the ROC curve
    """
    if hasattr(fitted.model, "endog"):
        actual = np.asarray(fitted.model.endog, dtype=float)
    else:
        actual = np.asarray(fitted.model["endog"], dtype=float)
    pred = np.asarray(fitted.fittedvalues, dtype=float)
    df = pl.DataFrame({"pred": pred, "actual": actual})

    # Calibration: bins of (roughly) equal size by predicted probability.
    ranks = df["pred"].rank(method="ordinal").to_numpy()
    bins = np.minimum((ranks - 1) * qnt // len(ranks), qnt - 1).astype(int) + 1
    calibration = (
        df.with_columns(pl.Series("bin", bins))
        .group_by("bin")
        .agg(
            pl.col("pred").mean().alias("predicted"),
            pl.col("actual").mean().alias("observed"),
            pl.len().alias("n"),
        )
        .sort("bin")
    )

    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    roc = pl.DataFrame({"fpr": fpr, "tpr": tpr})
    auc_value = float(metrics.roc_auc_score(actual, pred))

    # Cumulative gains: contact the highest predicted probabilities first.
    order = np.argsort(-pred, kind="stable")
    cum_pos = np.cumsum(actual[order])
    cuts = np.ceil(np.arange(1, qnt + 1) * len(order) / qnt).astype(int) - 1
    gains = pl.concat(
        [
            pl.DataFrame({"cum_prop": [0.0], "cum_gains": [0.0]}),
            pl.DataFrame(
                {
                    "cum_prop": (cuts + 1) / len(order),
                    "cum_gains": cum_pos[cuts] / max(cum_pos[-1], 1.0),
                }
            ),
        ]
    )
    return {
        "calibration": calibration,
        "roc": roc,
        "gains": gains,
        "predictions": df,
        "auc": auc_value,
    }


def logit_dashboard(fitted, lev: str | None = None, qnt: int = 10, figsize=(10, 9)):
    """
    Diagnostic dashboard for a binary classifier (logistic regression)

    The four views applied analytics uses to decide whether a classifier can be
    relied on: are the probabilities calibrated, does the model rank well
    (ROC/AUC), how much of the response does targeting the top of the list
    capture (cumulative gains), and how well do the predicted probabilities
    separate the two outcomes.

    Parameters
    ----------
    fitted : statsmodels GLM (Binomial) results
    lev : str, optional
        Label for the positive outcome in the legend.
    qnt : int
        Number of predicted-probability bins for the calibration and gains views.
    figsize : tuple
        Figure size of the composed dashboard.
    """
    d = logit_dashboard_data(fitted, qnt=qnt)
    positive = str(lev) if lev is not None else "1"
    unit = scale_x_continuous(limits=(0, 1))
    unit_y = scale_y_continuous(limits=(0, 1))

    p1 = (
        ggplot(d["calibration"], aes(x="predicted", y="observed"))
        + geom_abline(intercept=0, slope=1, linetype="dashed", color=REF)
        + geom_line(color=LINE)
        + geom_point(color=LINE, size=2.5)
        + unit
        + unit_y
        + labs(x="Mean predicted probability", y="Observed rate")
        + ggtitle(f"Calibration ({qnt} bins)")
        + theme_pyrsm()
    )
    p2 = (
        ggplot(d["roc"], aes(x="fpr", y="tpr"))
        + geom_abline(intercept=0, slope=1, linetype="dashed", color=REF)
        + geom_line(color=LINE, size=1)
        + unit
        + unit_y
        + labs(x="False positive rate", y="True positive rate")
        + ggtitle(f"ROC curve (AUC = {d['auc']:.3f})")
        + theme_pyrsm()
    )
    p3 = (
        ggplot(d["gains"], aes(x="cum_prop", y="cum_gains"))
        + geom_abline(intercept=0, slope=1, linetype="dashed", color=REF)
        + geom_line(color=LINE)
        + geom_point(color=LINE)
        + unit
        + unit_y
        + labs(x="Share of observations targeted", y=f"Share of '{positive}' captured")
        + ggtitle("Cumulative gains")
        + theme_pyrsm()
    )
    preds = d["predictions"].with_columns(
        pl.when(pl.col("actual") == 1)
        .then(pl.lit(positive))
        .otherwise(pl.lit(f"not {positive}"))
        .alias("outcome")
    )
    p4 = (
        ggplot(preds, aes(x="pred", fill="outcome"))
        + geom_density(alpha=0.5)
        + unit
        + labs(x="Predicted probability", y="Density", fill="Actual")
        + ggtitle("Predicted probabilities by outcome")
        + theme_pyrsm()
        + theme(legend_position="bottom")
    )
    return (p1 | p2) / (p3 | p4) + theme(figure_size=figsize)


def sim_prediction(
    data: pd.DataFrame | pl.DataFrame,
    vary: list = [],
    nnv: int = 5,
    minq: float = 0,
    maxq: float = 1,
) -> pl.DataFrame:
    """
    Simulate data for prediction

    Parameters
    ----------
    data : DataFrame (polars or pandas)
    vary : List of column names or Dictionary with keys and values to use
    nnv : int
        Number of values to use to simulate the effect of a numeric variable
    minq : float
        Quantile to use for the minimum value of numeric variables
    maxq : float
        Quantile to use for the maximum value of numeric variables

    Returns:
    ----------
    pl.DataFrame with values to use for prediction
    """
    # Convert to polars
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    def fix_value(col_name):
        col = data[col_name]
        if col.dtype.is_numeric():
            return col.mean()
        else:
            # Get most common value
            return col.value_counts().sort("count", descending=True)[0, col_name]

    dct = {c: [fix_value(c)] for c in data.columns}
    schema = {c: data[c].dtype for c in data.columns}

    if isinstance(vary, dict):
        # user provided values and ranges
        for key, val in vary.items():
            dct[key] = val
    else:
        # auto derived values and ranges
        vary = ifelse(isinstance(vary, str), [vary], vary)
        for v in vary:
            col = data[v]
            if col.dtype.is_numeric():
                nu = col.n_unique()
                if nu > 2:
                    min_val = col.quantile(minq)
                    max_val = col.quantile(maxq)
                    dct[v] = np.linspace(min_val, max_val, min([nu, nnv])).tolist()
                else:
                    dct[v] = [col.min(), col.max()]
            else:
                dct[v] = col.unique().to_list()

    return expand_grid(dct, schema)


def scatter_plot(
    fitted,
    df,
    nobs: int = 1000,
    figsize: tuple = None,
    resid=False,
    lines: list[str] | None = None,
) -> None:
    """
    Scatter plot of explanatory and response variables from a fitted regression

    Parameters
    ----------
    fitted : A fitted linear regression model
    df : DataFrame (polars or pandas)
        Data frame with explanatory and response variables
    nobs : int
        Number of observations to use for the scatter plots. The default
        value is 1,000. To use all observations in the plots, use nobs=-1
    figsize : tuple
        A tuple that determines the figure size (ignored, kept for compatibility)
    resid : bool
        If True, use residuals as the response variable
    lines : list[str] or None
        For numeric explanatory variables: "line" adds a linear fit, "loess"
        a local polynomial fit, and "jitter" jitters the points
    """
    lines = list(lines or [])
    # Convert to polars
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    exog_names = extract_evars(fitted.model, df.columns)

    if resid:
        endog_name = "residuals"
        if df.height != len(fitted.resid):
            raise ValueError(
                "The number of observations in the fitted model and the data frame must be the same"
            )
        df = df.select(exog_names).with_columns(pl.Series(endog_name, fitted.resid))
    else:
        endog_name = extract_rvar(fitted.model, df.columns)
        df = df.select([endog_name] + exog_names)

    if nobs < df.height and nobs != np.inf and nobs != -1:
        df = df.sample(nobs, seed=1234)

    plots = []
    for exog_name in exog_names:
        col = df[exog_name]
        if col.dtype.is_numeric():
            # Scatter plot for numeric variables
            points = (
                geom_jitter(alpha=ALPHA, color=FILL, width=0.2, height=0)
                if "jitter" in lines
                else geom_point(alpha=ALPHA, color=FILL)
            )
            p = (
                ggplot(df, aes(x=exog_name, y=endog_name))
                + points
                + labs(x=exog_name, y=endog_name)
                + theme_pyrsm()
            )
            if "line" in lines:
                p = p + _fit_line("lm")
            if "loess" in lines:
                p = p + _fit_line("loess")
        else:
            # Jitter plot for categorical variables with mean lines
            # means_df = df.group_by(exog_name).agg(pl.col(endog_name).mean().alias("mean"))
            pl_mean = lambda x: x.mean()

            p = (
                ggplot(df, aes(x=exog_name, y=endog_name))
                + geom_jitter(alpha=ALPHA, width=0.2, color=FILL)
                + stat_summary(
                    fun_y=pl_mean,
                    fun_ymin=pl_mean,
                    fun_ymax=pl_mean,
                    geom="crossbar",
                    color=REF,
                    linetype="solid",
                    width=0.5,
                    size=0.8,
                    fatten=0,
                )
                + labs(x=exog_name, y=endog_name)
                + theme_pyrsm()
            )
        plots.append(p)

    # Compose plots into a grid (2 columns, 5 x 3 inches per plot)
    from pyrsm.plot_utils import compose_plots

    result = compose_plots(plots, ncol=2, width_per_col=5, height_per_row=3)

    return result


def residual_plot(
    fitted,
    df,
    nobs: int = 1000,
    figsize: tuple = None,
    lines: list[str] | None = None,
) -> None:
    """
    Plot of variables vs residuals

    Parameters
    ----------
    fitted : A fitted linear regression model
    nobs : int
        Number of observations to use for the scatter plots. The default
        value is 1,000. To use all observations in the plots, use nobs=-1
    figsize : tuple
        A tuple that determines the figure size. If None, size is
        determined based on the number of variables in the model
    """

    return scatter_plot(fitted, df, nobs=nobs, figsize=figsize, resid=True, lines=lines)


def influence_plot(fitted, figsize: tuple = None):
    """
    Influential observations: standardized residuals by observation, with the
    point size showing Cook's distance (as in Radiant)

    Observations with an absolute standardized residual below 1 are not shown,
    except the three with the largest Cook's distance, which are labeled with
    their row number. Dashed lines mark +/- 1 and +/- 3.

    Parameters
    ----------
    fitted : A fitted linear regression model

    Returns
    -------
    plotnine.ggplot
    """
    infl = fitted.get_influence()
    if hasattr(fitted, "resid_deviance"):
        # GLM: standardized deviance residuals (R's rstandard.glm)
        h = np.asarray(infl.hat_matrix_diag, dtype=float)
        std_resid = np.asarray(fitted.resid_deviance, dtype=float) / np.sqrt(1 - h)
    else:
        std_resid = np.asarray(infl.resid_studentized_internal, dtype=float)
    cooksd = np.asarray(infl.cooks_distance[0], dtype=float)
    index = np.arange(1, len(std_resid) + 1)
    top = set(np.argsort(-np.nan_to_num(cooksd, nan=-np.inf))[:3].tolist())
    is_top = np.array([i in top for i in range(len(index))])
    keep = (np.abs(std_resid) >= 1) | is_top
    df = pd.DataFrame(
        {
            "index": index[keep],
            "std_resid": std_resid[keep],
            "cooksd": cooksd[keep],
            "label": np.where(is_top[keep], index[keep].astype(str), ""),
        }
    )
    big = np.nanmax(np.abs(std_resid)) if len(std_resid) else 0
    lim = (min(-4.0, -big - 0.5), max(4.0, big + 0.7))
    return (
        ggplot(df, aes(x="index", y="std_resid"))
        + geom_hline(yintercept=[-3, -1, 1, 3], linetype="dashed", color=REF, size=0.4)
        + geom_point(aes(size="cooksd"), alpha=0.5, color=LINE)
        + geom_text(aes(label="label"), nudge_y=0.35, size=9)
        + scale_y_continuous(breaks=list(range(-4, 5)), limits=lim)
        + labs(
            title="Influential observations",
            x="Observation index",
            y="Standardized residuals",
            size="Cook's distance",
        )
        + theme_pyrsm()
    )


def cross_validation(
    mlobj,
    name,
    param_grid,
    scoring=None,
    prn=True,
    directory="cv-objects",
    n_splits=5,
    n_jobs=2,
    verbose=1,
    random_state=1234,
):
    """
    Use cross-validation to find the best hyper parameters for a machine learning model. This function can be used with the pyrsm package to automatically save the cross-validation object to a file and load it if it already exists.

    Parameters
    ----------
    mlobj: An object created by the pyrsm package with a fitted machine learning model
    name: str
        Name of the model to use in the file name for the cross-validation object. Most likely the same as the name of the model object
    param_grid: dict
        Dictionary with parameters to use for cross-validation
    scoring: dict
        Dictionary with scoring metrics to use for cross-validation. Defaults to {"r2": "r2", "rmse": "neg_root_mean_squared_error"} for regression and {"AUC": "roc_auc", "accuracy": "accuracy"} for classification
    prn : bool
        If True print output from cross-validation
    directory : str
        Directory to save the cross-validation object. Defaults to "cv-objects"
    n_splits : int
        Number of splits to use in cross validation. Defaults to 5.
    n_jobs : int
        Number of just to start to run the cross validation. Defaults to 2.
    random_state : int:
        Random seed to use. Defaults to 1234.

    Returns:
        cv: An sklearn GridSearchCV object
    """

    if not hasattr(mlobj, "mod_type"):
        raise ValueError("mlobj must be a pyrsm object")
    if not hasattr(mlobj, "fitted"):
        raise ValueError("mlobj must be a pyrsm object with a fitted model")
    if not hasattr(mlobj, "data_onehot"):
        raise ValueError("mlobj must be a pyrsm object with data_onehot")

    if mlobj.mod_type == "classification":
        kfolds = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        if scoring is None:
            scoring = {"AUC": "roc_auc", "accuracy": "accuracy"}
    elif mlobj.mod_type == "regression":
        kfolds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if scoring is None:
            scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error"}
    else:
        raise ValueError("mlobj must be a pyrsm object with a fitted model")

    refit = list(scoring.keys())[0]

    cv_file = f"{directory}/{name}-cross-validation-object.pkl"
    if os.path.exists(cv_file):
        cv = load_state(cv_file)["cv"]
    else:
        rvar = mlobj.rvar
        if hasattr(mlobj, "data_std"):
            # mlp: the raw data and a network that standardizes inside every fit,
            # so each fold is scaled by its own training rows and regression
            # scores are in the response's units
            from pyrsm.model.mlp import fold_scaled_data

            X, y, estimator = fold_scaled_data(mlobj)
        else:
            X = mlobj.data_onehot
            X = X.to_pandas() if hasattr(X, "to_pandas") else X
            y = mlobj.data.get_column(rvar).to_pandas() if hasattr(mlobj.data, "get_column") else mlobj.data[rvar]
            estimator = mlobj.fitted
        cv = GridSearchCV(
            estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=kfolds,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
        ).fit(X, y)
        if not os.path.exists(directory):
            os.mkdir(directory)
        save_state({"cv": cv}, cv_file)

    if prn:
        print("The best parameters are:", cv.best_params_)
        print("The best model fit score is:", cv.best_score_)
        print(
            "The GirdSearchCV model fit estimates:\n",
            pd.DataFrame(cv.cv_results_).sort_values(f"rank_test_{refit}").head(),
        )

    return cv


# %%


def _r_format_column(values: list, digits: int = 5) -> list[str]:
    """Format a numeric column the way R's ``print.anova`` does.

    ``zapsmall`` to ``digits``, then R's ``formatReal``: fixed notation unless
    it would be wider than scientific notation, at most ``digits``
    significant digits.
    """
    import math

    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return ["" for _ in values]
    # print.anova zaps each column first: zapsmall(x, digits)
    mx = max(abs(v) for v in finite)
    if mx > 0:
        # R (>= 4.4) zapsmall: round(x, max(0, digits - log10(mx))), the
        # fractional number of places itself rounded to the nearest integer
        places = int(math.floor(max(0.0, digits - math.log10(mx)) + 0.5))
        values = [None if v is None or not math.isfinite(v) else round(v, places) for v in values]
        finite = [v for v in values if v is not None]
    neg = any(v < 0 for v in finite)
    sig, left, right, wide_exp = [], [], [], False
    for v in finite:
        if v == 0:
            sig.append(1), left.append(1), right.append(0)
            continue
        mant, exp = f"{abs(v):.{digits - 1}e}".split("e")
        exp = int(exp)
        wide_exp = wide_exp or abs(exp) >= 100
        s = len(mant.replace(".", "").rstrip("0")) or 1
        sig.append(s)
        left.append(max(1, exp + 1))
        right.append(max(0, s - 1 - exp))
    rgt = max(right)
    fixed_width = neg + max(left) + (rgt + 1 if rgt > 0 else 0)
    mf = max(sig) - 1
    sci_width = neg + (mf + 1 if mf > 0 else 0) + 1 + (5 if wide_exp else 4)
    out = []
    for v in values:
        if v is None or not math.isfinite(v):
            out.append("")
        elif fixed_width <= sci_width:
            out.append(f"{v:.{rgt}f}")
        else:
            out.append(f"{v:.{mf}e}")
    return out


def _r_formula_lines(formula: str, width: int = 60) -> str:
    """Wrap a formula as R's ``deparse`` does: break after ``+`` past ``width``."""
    lhs, rhs = formula.split("~", 1)
    terms = [t.strip() for t in rhs.split("+")]
    lines, line = [], f"{lhs.strip()} ~ {terms[0]}"
    for term in terms[1:]:
        if len(line) + 3 > width:
            lines.append(line + " + ")
            line = "    " + term
        else:
            line += " + " + term
    lines.append(line)
    return "\n".join(lines)


def _step_block(label: str, aic: float, formula: str, columns: list[str], table: list | None) -> str:
    """One step of R's ``step()`` trace: header, formula and the AIC table.

    ``table`` rows are ``(label, *values, aic)`` with values for ``columns``;
    a ``Df`` column prints as integers, the rest as R's ``print.anova``.
    ``table=None`` is a step R takes without a table (dropping a term with
    zero df, i.e. an aliased one).
    """
    header = f"{label}:  AIC={float(f'{round(aic, 2):.7g}'):.7g}"
    if table is None:
        return f"{header}\n{_r_formula_lines(formula)}\n"
    labels = [row[0] for row in table]
    cols = []
    for i, name in enumerate([*columns, "AIC"]):
        values = [row[1 + i] for row in table]
        if name == "Df":
            cols.append((name, ["" if v is None else str(int(round(v))) for v in values]))
        else:
            cols.append((name, _r_format_column(values)))
    lw = max(len(x) for x in labels)
    widths = [max(len(h), *(len(v) for v in vals)) for h, vals in cols]
    out = [" " * lw + "".join(" " + h.rjust(w) for (h, _), w in zip(cols, widths))]
    for i, lab in enumerate(labels):
        out.append(
            lab.ljust(lw)
            + "".join(" " + vals[i].rjust(w) for (_, vals), w in zip(cols, widths))
        )
    return f"{header}\n{_r_formula_lines(formula)}\n\n" + "\n".join(out)


def _step_record(label: str, aic: float, formula: str, columns: list[str], table: list) -> dict:
    """One step of the stepwise trace as data: what ``_step_block`` prints.

    ``rows`` are ``{"term": "- carat", <column>: value, ..., "AIC": value}``
    with raw numbers (None where R prints nothing), ordered by AIC.
    """
    names = [*columns, "AIC"]
    return {
        "label": label,
        "aic": float(aic),
        "formula": formula,
        "columns": names,
        "rows": [
            {"term": row[0], **{n: (None if v is None else float(v)) for n, v in zip(names, row[1:])}}
            for row in table
        ],
    }


def step_select(
    *,
    full_fit,
    formula: str,
    rvar: str,
    direction: str,
    pdf,
    fit_formula,
    aic,
    stats,
    columns: list[str],
) -> dict:
    """
    Stepwise selection by AIC as R's ``step()``, with its printed trace

    Parameters
    ----------
    full_fit : statsmodels results
        The model with every candidate term (the upper scope)
    formula : str
        Its formula, in the order the student listed the terms
    direction : str
        "backward" (start from the full model), "forward" or "both" (start
        from the intercept-only model)
    pdf : pandas DataFrame
        Estimation data; every model is fit on the rows complete for the full model
    fit_formula : callable
        ``fit_formula(formula, pdf)`` -> fitted results
    aic : callable
        ``aic(fit)`` -> R's ``extractAIC`` for the model type
    stats : callable
        ``stats(fit, base)`` -> values for ``columns``; ``base`` is None for
        the ``<none>`` row, else the current model the change starts from
    columns : list[str]
        Table columns before AIC, e.g. ["Df", "Sum of Sq", "RSS"] (lm) or
        ["Df", "Deviance"] (glm)

    Returns
    -------
    dict with formula, fit, current terms, removed, added, the trace (text)
    and the steps (the trace as data, see ``_step_record``)
    """
    design = full_fit.model.data.design_info
    has_intercept = any(len(t.factors) == 0 for t in design.terms)

    # Terms in R's order: by degree, then as the student listed them.
    rhs = [t.strip() for t in formula.split("~", 1)[1].split("+")]
    terms = [t for t in design.terms if len(t.factors) > 0]

    def position(name: str) -> int:
        return rhs.index(name) if name in rhs else len(rhs)

    terms.sort(key=lambda t: (len(t.factors), position(t.name())))
    upper = [t.name() for t in terms]
    factors = {t.name(): {f.name() for f in t.factors} for t in terms}

    rows = full_fit.model.data.row_labels
    if rows is not None and len(rows) < len(pdf):
        pdf = pdf.loc[rows]

    degree = {t.name(): len(t.factors) for t in terms}

    def formula_for(keep: list[str]) -> str:
        # R's update() appends an added term; terms() then orders by degree
        keep = sorted(keep, key=lambda t: degree[t])
        rhs_ = " + ".join(keep) if keep else "1"
        if not has_intercept:
            rhs_ += " - 1"
        return f"{rvar} ~ {rhs_}"

    def droppable(t: str, current: list[str]) -> bool:
        return not any(factors[t] < factors[o] for o in current if o != t)

    def addable(t: str, current: list[str]) -> bool:
        return all(o in current for o in upper if factors[o] < factors[t])

    current = list(upper) if direction == "backward" else []
    fit = fit_formula(formula_for(current), pdf)
    fit_aic = aic(fit)
    removed: list[str] = []
    added: list[str] = []
    trace: list[str] = []
    steps: list[dict] = []
    label = "Start"
    while True:
        table = [("<none>", *stats(fit, None), fit_aic, current)]
        if direction in ("backward", "both"):
            for t in current:
                if droppable(t, current):
                    keep = [x for x in current if x != t]
                    f = fit_formula(formula_for(keep), pdf)
                    table.append((f"- {t}", *stats(f, fit), aic(f), keep))
            # R's step(): a term whose removal changes no df (its columns
            # were dropped as aliased) goes first, the last such term in
            # model order, without printing a table.
            zero = [r for r in table[1:] if r[1] is not None and round(r[1]) == 0]
            if zero:
                trace.append(_step_block(label, fit_aic, r_label(formula_for(current)), columns, None))
                label = "Step"
                removed.append(zero[-1][0][2:])
                current = zero[-1][-1]
                fit = fit_formula(formula_for(current), pdf)
                fit_aic = aic(fit)
                continue
        if direction in ("forward", "both"):
            for t in upper:
                if t not in current and addable(t, current):
                    keep = current + [t]
                    f = fit_formula(formula_for(keep), pdf)
                    table.append((f"+ {t}", *stats(f, fit), aic(f), keep))
        # R drops rows with zero df (adding an aliased term changes nothing)
        table = [table[0]] + [r for r in table[1:] if r[1] is None or round(r[1]) != 0]
        table.sort(key=lambda row: row[-2])
        rows = [(r_label(r[0]), *r[1:-1]) for r in table]
        trace.append(_step_block(label, fit_aic, r_label(formula_for(current)), columns, rows))
        steps.append(_step_record(label, fit_aic, r_label(formula_for(current)), columns, rows))
        label = "Step"
        best = table[0]
        if best[0] == "<none>" or best[-2] >= fit_aic - 1e-7:
            break
        changed = best[0][2:]
        (removed if best[0].startswith("-") else added).append(changed)
        current = best[-1]
        fit = fit_formula(formula_for(current), pdf)
        fit_aic = aic(fit)

    return {
        "formula": formula_for(current),
        "fit": fit,
        "current": current,
        "removed": removed,
        "added": added,
        "trace": "\n\n".join(trace),
        "steps": steps,
    }
