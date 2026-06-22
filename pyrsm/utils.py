import inspect
import json
from datetime import date, datetime
from functools import lru_cache
from itertools import product
from math import ceil
from sys import modules

import numpy as np
import polars as pl


@lru_cache(maxsize=1)
def _get_pandas():
    """Lazy load pandas to speed up import time."""
    import pandas as pd

    return pd


def add_description(df, md="", path=""):
    """
    Add a description to a Pandas DataFrame in markdown format

    Parameters
    ----------
    df : Pandas DataFrame
    md : str
        Data description in markdown format
    path : str
        Path to a text file with the data description in markdown format

    Returns
    -------
    Pandas DataFrame with added description
    """

    if path != "":
        f = open(path)
        md = f.read()
        f.close()
    elif md == "":
        print("Provide either text (markdown) or the path to a file")
        print("with the data description")

    df._metadata.append("description")
    df.description = md
    return df


def describe(df, prn=True):
    """
    Print out Pandas DataFrame description attribute if available. Else use Pandas'
    description method to provide summary statistics
    """
    if hasattr(df, "description"):
        if "ipykernel" in modules and prn:
            from IPython.display import Markdown, display

            display(Markdown(df.description))
        elif prn:
            print(df.description)
        else:
            return df.description
    else:
        print("No description attribute available")
        return df.describe()


def ifelse(cond, if_true, if_false):
    """
    Oneline if-else function like R

    Parameters
    ----------
    cond : List, pandas series, or numpy array of boolean values
    if_true : float, int, str, or list, pandas series, or numpy array
        Value to use if the condition is True
    if_false : float, int, str, or list, pandas series, or numpy array
        Value to use if the condition is False

    Returns
    -------
    Numpy array if the length of cond > 1. Else the same object type as either if_true or if_false

    Examples
    --------
    ifelse(2 > 3, "greater", "smaller")
    ifelse(np.array([2, 3, 4]) > 2, 1, 0)
    ifelse(np.array([2, 3, 4]) > 2, np.array([-1, -2, -3]), np.array([1, 2, 3]))
    """
    try:
        len(cond) > 1  # catching "TypeError: object of type 'bool' has no len()"
        return np.where(cond, if_true, if_false)
    except TypeError:
        if cond:
            return if_true
        else:
            return if_false


def format_nr(x, sym="", dec=2, perc=False):
    """
    Format a number or numeric vector with a specified number of decimal places,
    thousand sep, and a symbol

    Parameters
    ----------
    x : numeric (vector)
        Number or vector to format
    sym : str
        Symbol to use
    dec : int
        Number of decimal places to use in rounding
    perc : boolean
        Display numbers as a percentage

    Returns
    -------
    str
        Number(s) in the desired format
    """
    try:
        len(x) > 1  # catching "TypeError: object of type 'bool' has no len()"
        return [format_nr(i, sym, dec, perc) for i in x]
    except TypeError:
        if dec == 0:
            x = int(x)

        if perc:
            return sym + f"{round((100.0 * x), dec):,}" + "%"
        else:
            return sym + f"{round((x), dec):,}"


def levels_list(df):
    """
    Provide a DataFrame and get a dictionary back with the unique values for
    each column

    Parameters
    ----------
    df: Pandas or Polars DataFrame

    Returns
    -------
    dict
        Dictionary with unique values (levels) for each column

    Examples
    --------
    df = pl.DataFrame({
        "var1": ["a", "b", "a"],
        "var2": [1, 2, 1]
    })
    levels_list(df)
    """
    pd = _get_pandas()
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    return {str(col): df[col].unique().to_list() for col in df.columns}


def expand_grid(dct, schema=None):
    """
    Provide a dictionary and get a polars DataFrame back with all possible
    value combinations.

    Parameters
    ----------
    dct : dict
        Dictionary with value combinations to expand
    schema : dict of polars dtypes, optional
        Column types to cast the result to (must be polars dtypes)

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with all possible value combinations

    Example
    -------
    expand_grid({"var1": ["a", "b"], "var2": [1, 2]})
    """
    rows = list(product(*dct.values()))
    df = pl.DataFrame(rows, schema=list(dct.keys()), orient="row")

    if schema is not None:
        # Cast columns to match original schema if provided
        cast_exprs = []
        for col in df.columns:
            if col in schema:
                target_dtype = schema[col]
                current_dtype = df[col].dtype
                # Only cast if types differ and target is a valid polars dtype
                if current_dtype != target_dtype and isinstance(
                    target_dtype, pl.DataType
                ):
                    cast_exprs.append(pl.col(col).cast(target_dtype))
                else:
                    cast_exprs.append(pl.col(col))
            else:
                cast_exprs.append(pl.col(col))
        df = df.select(cast_exprs)

    return df


def table2data(df, freq):
    """
    Provide a DataFrame and get a DataFrame back with the total number of rows
    equal to the sum of the frequency variable

    Parameters
    ----------
    df: Pandas or Polars DataFrame
    freq: str
        String with the variable name of the frequency column in df

    Returns
    -------
    pl.DataFrame
        Polars DataFrame expanded in size based on the frequencies in selected column

    Examples
    --------
    df = pl.DataFrame({"var1": ["a", "b", "a"], "freq": [5, 2, 3]})
    table2data(df, "freq")
    """
    pd = _get_pandas()
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    # Get columns excluding the frequency column
    other_cols = [c for c in df.columns if c != freq]

    # Repeat each row by its frequency value
    return df.select(other_cols).select(pl.all().repeat_by(df[freq])).explode(pl.all())


def setdiff(x, y, sort=False):
    """
    Returns unique elements in x that are not in y

    Parameters
    ----------
    x : List or iterable
    y : List or iterable
    sort : boolean
        Sort the output

    Returns
    -------
    list
        Elements in x that are not in y

    Examples
    --------
    setdiff(["a", "b", "c"], ["b", "x"])
    """
    result = list(dict.fromkeys(item for item in x if item not in set(y)))
    return sorted(result) if sort else result


def union(x, y):
    """
    Return the unique, sorted list of values that are in either of the two input lists

    Parameters
    ----------
    x : List or iterable
    y : List or iterable

    Returns
    -------
    list
        Unique, sorted list of values that are in either input

    Examples
    --------
    union(["a", "b", "c"], ["b", "x"])
    """
    return sorted(set(x) | set(y))


def intersect(x, y):
    """
    Return the unique, sorted list of values that are in both input lists

    Parameters
    ----------
    x : List or iterable
    y : List or iterable

    Returns
    -------
    list
        Unique, sorted list of values that are in both inputs

    Examples
    --------
    intersect(["a", "b", "c"], ["b", "x"])
    """
    return sorted(set(x) & set(y))


def lag(arr, num=1, fill=None):
    """
    Shift data by a number of periods (lag)

    Parameters
    ----------
    arr : list, pl.Series, or array-like
        Data to shift
    num : int
        Number of periods to shift. Positive values lag (shift forward),
        negative values lead (shift backward)
    fill : Optional
        Value to use for missing values created by shifting. Default is None (null)

    Returns
    -------
    pl.Series
        Shifted data as a polars Series

    Examples
    --------
    lag([1, 2, 3, 4], num=1)  # [null, 1, 2, 3]
    lag([1, 2, 3, 4], num=-1)  # [2, 3, 4, null]
    """
    if isinstance(arr, pl.Series):
        s = arr
    else:
        s = pl.Series(arr)

    result = s.shift(num)
    if fill is not None:
        result = result.fill_null(fill)
    return result


def lead(arr, num=1, fill=None):
    """
    Shift data by a number of periods (lead)

    Convenience function for leading periods. Uses the lag function with negative num.

    Parameters
    ----------
    arr : list, pl.Series, or array-like
        Data to shift
    num : int
        Number of periods to lead (shift backward)
    fill : Optional
        Value to use for missing values. Default is None (null)

    Returns
    -------
    pl.Series
        Shifted data as a polars Series
    """
    return lag(arr, num=-num, fill=fill)


def months_abb(start=1, nr=12, year=datetime.today().year):
    """
    Create a list of abbreviated month labels

    Parameters
    ----------
    start : int
        Numeric value of the first month in the list (e.g., January is 1)
    nr : int
        Number of months to include in the list
    year : int
        Input to use to replace missing values created by shifting
        the values in arr forwards of backwards

    Returns
    -------
    list
        List of abbreviated month labels
    """

    rng = ceil((nr + (start - 1)) / 12)
    mnths = [
        date(year, m, 1).strftime("%B")[0:3]
        for i in range(1, rng + 1)
        for m in range(1, 13)
    ]
    start -= 1
    return mnths[start : (nr + start)]


def md(x: str) -> None:
    """
    Use in-line python code to generate markdown output

    Parameters
    ----------
    x : A python f-string, the path to a markdown file, or a URL to a markdown file

    Returns
    -------
    None - Markdown output is printed

    Examples
    --------
    md(f"### In-line code to markdown results")
    radius = 10
    md(f"The radius of the circle is {radius}.")
    md("./path-to-markdown-file.md")
    md("https://raw.githubusercontent.com/radiant-ai-hub/pyrsm/refs/heads/main/pyrsm/data/basics/salary_description.md")
    """
    import os
    from pathlib import Path

    from IPython.display import Markdown, display

    # Check if x is a URL
    if x.startswith("http://") or x.startswith("https://"):
        import ssl
        import urllib.request

        import certifi

        ssl_context = ssl.create_default_context(cafile=certifi.where())
        try:
            with urllib.request.urlopen(x, context=ssl_context) as response:
                content = response.read().decode("utf-8")
        except Exception as e:
            content = f"Error fetching URL: {e}"
        display(Markdown(content))
    # Check if x is a file path
    elif os.path.isfile(x) or Path(x).exists():
        with open(x) as f:
            content = f.read()
        display(Markdown(content))
    else:
        # Treat as raw markdown content
        display(Markdown(x))


def md_notebook(nb: str, type="python") -> None:
    """
    Print code from another notebook in markdown format
    This can be useful when you are using the %run magick
    in a notebook to source other notebooks. This way you
    both the notebook output and the code

    Parameters
    ----------
    nb : Path to a Jupyter Notebook file

    Returns
    -------
    None - Markdown output is printed

    Examples
    --------
    md("./path-to-notebook-file.ipynb")
    """
    with open(nb) as f:
        data = json.load(f)

    md_return = "\n"

    for cell in data["cells"]:
        if cell["cell_type"] == "code":
            md_return += f"```{type}\n" + "".join(cell["source"]) + "\n```\n"
        elif cell["cell_type"] == "markdown":
            md_return += "\n".join(cell["source"]) + "\n"
        else:
            md_return += "\n"

    md(md_return)


def odir(obj, private: bool = False) -> dict:
    """
    List an objects attributes and 'public' methods

    Parameters
    ----------
    obj : Any python object
    private : Boolean, default is false to exclude 'private' methods and attributes

    Returns
    -------
    Dictionary with names of attributes and methods

    Examples
    --------
    odir(["a"])
    """
    mth = []
    attr = []
    for i in inspect.getmembers(obj):
        if private or not i[0].startswith("_"):
            if inspect.ismethod(i[1]) or inspect.isbuiltin(i[1]):
                mth.append(i[0])
            else:
                attr.append(i[0])

    return {"methods": mth, "attributes": attr}


def check_dataframe(df):
    """Convert input to polars DataFrame."""
    if isinstance(df, pl.DataFrame):
        return df
    pd = _get_pandas()
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df.copy())
    return pl.DataFrame(df)


def check_series(s):
    """Convert input to polars Series."""
    if isinstance(s, pl.Series):
        return s
    pd = _get_pandas()
    if isinstance(s, pd.Series):
        return pl.from_pandas(s)
    return pl.Series(s)


def sig_stars(pval) -> pl.Series:
    """
    Convert p-values to significance stars.

    Parameters
    ----------
    pval : list of floats/None or a pl.Series
        P-values to convert

    Returns
    -------
    pl.Series
        Series of significance symbols (strings)
    """
    df = pl.DataFrame({"pval": pval}).fill_nan(1)
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


def polychoric_corr(x, y, inf=10):
    """Compute polychoric correlation between two ordinal variables.

    Estimates the correlation between two latent normal variables from
    their observed ordinal (discretized) versions using maximum likelihood.
    Equivalent to R's polycor::polychor().

    Parameters
    ----------
    x, y : array-like
        Ordinal variables (integer-coded).
    inf : float
        Value used for +/- infinity thresholds.

    Returns
    -------
    float
        Estimated polychoric correlation.
    """
    from scipy.optimize import minimize_scalar
    from scipy.stats import norm

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    n = x.size
    if n == 0:
        return 0.0

    # Integer-code the levels and build the contingency table in one vectorized
    # pass (np.add.at) instead of a Python loop over every observation.
    _, x_idx = np.unique(x, return_inverse=True)
    _, y_idx = np.unique(y, return_inverse=True)
    p, m = int(x_idx.max()) + 1, int(y_idx.max()) + 1
    if p < 2 or m < 2:
        return 0.0
    n_table = np.zeros((p, m))
    np.add.at(n_table, (x_idx, y_idx), 1.0)

    # Two-step thresholds from the marginals (held fixed while optimizing rho).
    x_thresh = np.concatenate(
        ([-inf], norm.ppf(np.cumsum(n_table.sum(axis=1))[:-1] / n), [inf])
    )
    y_thresh = np.concatenate(
        ([-inf], norm.ppf(np.cumsum(n_table.sum(axis=0))[:-1] / n), [inf])
    )

    # Bivariate-normal CDF over the whole threshold grid via Sheppard's
    # formula  Phi2(h,k,rho) = Phi(h)Phi(k) + int_0^rho phi2(h,k,t) dt,
    # the 1-D integral approximated with fixed Gauss-Legendre nodes. This is
    # fully vectorized in numpy (no per-point scipy CDF calls), which is what
    # makes the optimization fast.
    gx, gy = np.meshgrid(x_thresh, y_thresh, indexing="ij")
    h = gx.ravel()
    k = gy.ravel()
    nz = n_table > 0
    counts = n_table[nz]

    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(24)
    base = norm.cdf(h) * norm.cdf(k)
    h2k2 = h**2 + k**2
    hk = h * k

    def _phi2_grid(rho):
        if rho == 0.0:
            cdf = base
        else:
            t = 0.5 * rho * (gl_nodes + 1.0)
            w = 0.5 * rho * gl_weights
            omt2 = 1.0 - t**2
            expo = -(h2k2[:, None] - 2.0 * t[None, :] * hk[:, None]) / (
                2.0 * omt2[None, :]
            )
            dens = np.exp(expo) / (2.0 * np.pi * np.sqrt(omt2)[None, :])
            cdf = base + dens @ w
        return cdf.reshape(p + 1, m + 1)

    def _neg_ll(rho):
        cdf = _phi2_grid(rho)
        probs = cdf[1:, 1:] - cdf[:-1, 1:] - cdf[1:, :-1] + cdf[:-1, :-1]
        probs = np.clip(probs[nz], 1e-20, None)
        return -np.sum(counts * np.log(probs))

    result = minimize_scalar(_neg_ll, bounds=(-0.999, 0.999), method="bounded")
    return result.x


def polyserial_corr(x_num, y_codes):
    """Two-step polyserial correlation (Olsson, Drasgow & Dorans, 1982).

    Correlation between a continuous variable ``x_num`` and an ordinal variable
    given as integer ``y_codes``. Matches the default (non-ML) estimator used
    by ``polycor::polyserial`` and is the numeric-vs-factor cell of
    ``polycor::hetcor``. Closed-form, so it is fast.
    """
    from scipy.stats import norm

    x = np.asarray(x_num, dtype=float)
    y = np.asarray(y_codes, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if x.size < 2 or x.std(ddof=1) == 0 or y.std(ddof=1) == 0:
        return 0.0

    r_xy = np.corrcoef(x, y)[0, 1]
    sy = y.std(ddof=1)

    # sum of normal densities at the category thresholds
    levels = np.sort(np.unique(y))
    cum = 0.0
    dens_sum = 0.0
    for lev in levels[:-1]:
        cum += np.mean(y == lev)
        dens_sum += norm.pdf(norm.ppf(cum))
    if dens_sum == 0:
        return 0.0
    return float(np.clip(r_xy * sy / dens_sum, -1.0, 1.0))


def polychoric_matrix(data_matrix):
    """Compute polychoric correlation matrix for multiple ordinal variables.

    Parameters
    ----------
    data_matrix : np.ndarray
        n x k matrix where each column is an ordinal variable.

    Returns
    -------
    np.ndarray
        k x k polychoric correlation matrix.
    """
    k = data_matrix.shape[1]
    corr = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            r = polychoric_corr(data_matrix[:, i], data_matrix[:, j])
            corr[i, j] = r
            corr[j, i] = r
    return corr
