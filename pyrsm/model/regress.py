import re
from functools import lru_cache

import pandas as pd
import polars as pl

from pyrsm.basics import display_utils as du
from pyrsm.utils import check_dataframe, format_nr, ifelse, setdiff
from pyrsm.data_scope import apply_scope, print_scope


@lru_cache(maxsize=1)
def _model_utils():
    from pyrsm.model import model as model_mod

    return model_mod


@lru_cache(maxsize=1)
def _visualize_utils():
    from pyrsm.model import visualize as visualize_mod

    return visualize_mod


@lru_cache(maxsize=1)
def _get_visualize_plots():
    """Lazy load visualize plot functions."""
    from pyrsm.eda.distr import distr
    from pyrsm.model.visualize import pdp_sm, pip_plot_sm, pred_plot_sm

    return distr, pred_plot_sm, pdp_sm, pip_plot_sm


class regress:
    """
    A class to perform linear regression modeling

    Attributes
    ----------
    data : pl.DataFrame
        The data used in the analysis should be a Pandas or Polars DataFrame. Can also pass in a dictionary with the name of the DataFrame as the key and the DataFrame as the value.
    name : str
        The name of the dataset if provided as a dictionary.
    rvar : str
        The response variable.
    evar : list[str]
        List of column names of the explanatory variables.
    ivar : list[str]
        List of strings with the names of the columns included as explanatory variables (e.g., ["x1:x2", "x3:x4"])
    formula : str
        Model specification formula.
    fitted : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted model.
    coef : pd.DataFrame
        The estimated model coefficients with standard errors, p-values, etc.
    data_filter : str, default ""
        Expression used to filter the data before the analysis, e.g.,
        "price > 5000 and cut in ['Ideal', 'Premium']". A bare name is a column
        and a call is an operation on one (``is_null(price)``). See
        ``pyrsm.data_scope`` for the full list.
    sort : str, default ""
        Column(s) to sort on, "-" in front for descending, e.g.,
        "clarity -price". Applied after the filter.
    slice : str, default ""
        Rows to keep, counting from 1, e.g., "1:100". Applied after the filter
        and the sort.
    standardize : int
        Number of standard deviations used to standardize numeric variables
        (0 when the data were not standardized).
    center : bool
        Whether numeric variables were mean-centered before estimation.
    stepwise : bool
        Whether backward stepwise selection (AIC) was applied.
    robust : bool
        Whether heteroskedasticity-robust (HC1) standard errors are used.
    removed : list[str]
        Terms dropped by stepwise selection.

    Methods
    -------
    __init__(data, rvar=None, evar=None, ivar=None, form=None)
        Initialize the regress class with the provided data and parameters.
    summary(main=True, fit=True, ci=False, ssq=False, rmse=False, vif=False, test=None, dec=3)
        Summarize the model output.
    plot(plot_type, nobs=1000, incl=None, excl=None, incl_int=None, fix=True, hline=False, nnv=20, minq=0.025, maxq=0.975)
        Plots for the model.
    predict(data=None, cmd=None, data_cmd=None, ci=False, conf=0.95)
        Generate predictions using the fitted model.
    """

    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        rvar: str | None = None,
        evar: list[str] | None = None,
        ivar: list[str] | None = None,
        formula: str | None = None,
        data_filter: str = "",
        sort: str = "",
        slice: str = "",
        standardize: bool | int | float = False,
        center: bool = False,
        stepwise: bool = False,
        robust: bool = False,
    ) -> None:
        """
        Initialize the regress class to build a linear regression model with the provided data and parameters.

        Parameters
        ----------
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
            Dataset used for the analysis. If a Polars DataFrame is provided, it will be converted to a Pandas DataFrame. If a dictionary is provided, the key will be used as the name of the dataset.
        rvar : str, optional
            Name of the column in the data to be use as the response variable.
        evar : list[str], optional
            List of column names in the data to use as explanatory variables.
        ivar : list[str], optional
            List of interactions to add to the model as explanatory variables (e.g., ["x1:x2", "x3:x4])
        formula : str, optional
            Optional formula to use if rvar and evar are not provided.
        data_filter : str, default ""
            Expression used to filter the data before the analysis, e.g.,
            "price > 5000 and cut in ['Ideal', 'Premium']". A bare name is a
            column and a call is an operation on one (``is_null(price)``). See
            ``pyrsm.data_scope`` for the full list.
        sort : str, default ""
            Column(s) to sort on, "-" in front for descending, e.g.,
            "clarity -price". Applied after the filter.
        slice : str, default ""
            Rows to keep, counting from 1, e.g., "1:100". Applied after the
            filter and the sort.
        standardize : bool or int, default False
            Standardize numeric variables (response included) before
            estimation so coefficients are comparable: ``1`` divides by one
            standard deviation, ``2`` (or ``True``) by two standard deviations
            (Gelman, 2008), which puts numeric and binary explanatory variables
            on a similar scale. Predictions are returned in the original units.
        center : bool, default False
            Subtract the mean from numeric variables (response included) before
            estimation. Ignored when ``standardize`` is set.
        stepwise : bool, default False
            Backward stepwise selection of explanatory variables using AIC.
            A main effect is not dropped while an interaction that contains it
            remains in the model.
        robust : bool, default False
            Use heteroskedasticity-robust (HC1) standard errors for the
            coefficients, confidence intervals and F-tests.
        """
        if isinstance(data, dict):
            self.name = list(data.keys())[0]
            self.data = data[self.name]
        else:
            self.data = data
            self.name = "Not provided"

        # Store as polars internally
        self.data = check_dataframe(self.data)

        # The data scope, applied before anything is measured: filter, then
        # sort, then slice. Kept on the object so ``summary`` can say which
        # rows the numbers describe -- a reader who cannot see the filter has
        # no way to know the n is not the whole dataset.
        self.data_filter = data_filter
        self.sort = sort
        self.slice = slice
        self.data = apply_scope(self.data, data_filter, sort, slice)
        self.rvar = rvar
        model_utils = _model_utils()
        self.evar = model_utils.convert_to_list(evar)
        self.ivar = model_utils.convert_to_list(ivar)
        self.formula = formula

        # Convert categorical/string columns to sorted Enum BEFORE fitting
        # This ensures patsy learns alphabetically sorted levels
        self._enum_types = {}
        cols_to_convert = []
        for c in self.data.columns:
            dtype = self.data[c].dtype
            if isinstance(dtype, pl.Enum):
                self._enum_types[c] = dtype
            elif dtype in (pl.Utf8, pl.String, pl.Categorical):
                cols_to_convert.append(c)

        if cols_to_convert:
            self.data, enum_types = model_utils.convert_categoricals_to_enum(
                self.data, cols_to_convert
            )
            self._enum_types.update(enum_types)

        # Estimation options (radiant.model's "check" boxes)
        if standardize is True:
            standardize = 2
        self.standardize = float(standardize or 0)
        if self.standardize == int(self.standardize):
            self.standardize = int(self.standardize)
        self.center = bool(center) and not self.standardize
        self.stepwise = bool(stepwise)
        self.robust = bool(robust)
        self.removed: list[str] = []

        # The data in original units. Standardizing or centering changes what
        # is estimated, not what a student types for a prediction, so
        # ``predict`` starts from these values.
        self.data_raw = self.data
        self.scale_means: dict[str, float] = {}
        self.scale_sds: dict[str, float] = {}
        if self.standardize or self.center:
            self._calc_scaling()
            self.data = self._scale(self.data)

        # Convert to pandas only for statsmodels fitting
        if not self.formula:
            if self.evar is None or len(self.evar) == 0:
                self.formula = f"{self.rvar} ~ 1"
            else:
                self.formula = f"{self.rvar} ~ {' + '.join(self.evar)}"
            if self.ivar:
                self.formula += f" + {' + '.join(self.ivar)}"
            formula_given = False
        else:
            formula_given = True

        pdf = self.data.to_pandas()
        self.fitted = self._ols(self.formula, pdf)
        if formula_given:
            self.evar = model_utils.extract_evars(self.fitted.model, self.data.columns)
            self.rvar = model_utils.extract_rvar(self.fitted.model, self.data.columns)

        if self.stepwise:
            self._stepwise_backward(pdf)

        self.fitted.nobs_dropped = self.data.height - self.fitted.nobs

        # Build coefficient table as polars DataFrame
        self.coef = pl.DataFrame(
            {
                "index": self.fitted.params.index.str.replace(
                    "[T.", "[", regex=False
                ).tolist(),
                "coefficient": self.fitted.params.values,
                "std.error": (self.fitted.params / self.fitted.tvalues).values,
                "t.value": self.fitted.tvalues.values,
                "p.value": self.fitted.pvalues.values,
                "  ": model_utils.sig_stars(self.fitted.pvalues.values),
            }
        ).drop_nulls(subset=["coefficient"])

    def _ols(self, formula: str, pdf: pd.DataFrame):
        """Fit OLS, with HC1 standard errors when ``robust`` is set."""
        import statsmodels.formula.api as smf

        model = smf.ols(formula=formula, data=pdf)
        if self.robust:
            return model.fit(cov_type="HC1", use_t=True)
        return model.fit()

    def _calc_scaling(self) -> None:
        """Means and standard deviations of the numeric columns (R's ``sd``)."""
        for c in self.data.columns:
            dtype = self.data[c].dtype
            if not dtype.is_numeric() or dtype == pl.Boolean:
                continue
            col = self.data[c].cast(pl.Float64)
            mean = col.mean()
            if mean is None:
                continue
            self.scale_means[c] = float(mean)
            sd = col.std(ddof=1)
            self.scale_sds[c] = float(sd) if sd else 0.0

    def _scale(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply the estimation-data scaling to ``df`` (any subset of columns)."""
        exprs = []
        for c, mean in self.scale_means.items():
            if c not in df.columns:
                continue
            sd = self.scale_sds.get(c, 0.0)
            if self.standardize and sd > 0:
                exprs.append(
                    ((pl.col(c).cast(pl.Float64) - mean) / (self.standardize * sd)).alias(c)
                )
            else:
                exprs.append((pl.col(c).cast(pl.Float64) - mean).alias(c))
        return df.with_columns(exprs) if exprs else df

    def _unscale_response(self, values):
        """Convert predictions of a standardized/centered response to original units."""
        mean = self.scale_means.get(self.rvar)
        if mean is None:
            return values
        sd = self.scale_sds.get(self.rvar, 0.0)
        if self.standardize and sd > 0:
            return values * (self.standardize * sd) + mean
        return values + mean

    def _stepwise_backward(self, pdf: pd.DataFrame) -> None:
        """Backward elimination by AIC, respecting marginality (R's ``step``)."""
        design = self.fitted.model.data.design_info
        has_intercept = any(len(t.factors) == 0 for t in design.terms)
        terms = [t for t in design.terms if len(t.factors) > 0]
        names = [t.name() for t in terms]
        factors = {t.name(): {f.name() for f in t.factors} for t in terms}

        # Compare models on the same rows: those complete for the full model.
        rows = self.fitted.model.data.row_labels
        if rows is not None and len(rows) < len(pdf):
            pdf = pdf.loc[rows]

        def formula_for(keep: list[str]) -> str:
            rhs = " + ".join(keep) if keep else "1"
            if not has_intercept:
                rhs += " - 1"
            return f"{self.rvar} ~ {rhs}"

        current = list(names)
        current_aic = self.fitted.aic
        while current:
            # A term can go only if no remaining term contains it.
            candidates = [
                t
                for t in current
                if not any(factors[t] < factors[o] for o in current if o != t)
            ]
            best, best_aic = None, None
            for t in candidates:
                aic = self._ols(formula_for([x for x in current if x != t]), pdf).aic
                if best_aic is None or aic < best_aic:
                    best, best_aic = t, aic
            if best is None or best_aic >= current_aic - 1e-7:
                break
            current.remove(best)
            self.removed.append(best)
            current_aic = best_aic

        if self.removed:
            self.formula = formula_for(current)
            self.fitted = self._ols(self.formula, pdf)
            self.ivar = [t for t in self.ivar if t in current]

    def _model_evar(self) -> list[str]:
        """Explanatory variables still used by the (possibly stepwise) model."""
        if not self.removed:
            return list(self.evar)
        design = self.fitted.model.data.design_info
        used = " ".join(t.name() for t in design.terms)
        return [v for v in self.evar if re.search(rf"\b{re.escape(v)}\b", used)]

    def summary(
        self,
        vif=False,
        ssq=False,
        rmse=False,
        test: str | None = None,
        ci=False,
        conf: float = 0.95,
        dec: int = 3,
        plain: bool = True,
    ) -> None:
        """
        Summarize the linear regression model output

        Parameters
        ----------
        vif : bool, default False
            Print the generalized variance inflation factors.
        ssq : bool, default False
            Print the sum of squares.
        rmse : bool, default False
            Print the root mean square error.
        test : list[str] or None, optional
            List of variable names used in the model to test using a Chi-Square test or None if no tests are performed.
        ci : bool, default False
            Print the confidence intervals for the coefficients.
        conf : float, default 0.95
            Confidence level for the coefficient confidence intervals.
        dec : int, default 3
            Number of decimal places to round to.
        plain : bool, default False
            Force plain text output (useful for non-notebook environments like Shiny).
        """
        model_utils = _model_utils()

        self._summary_header()
        if not plain and du.is_notebook():
            self._summary_styled(dec)
        else:
            self._summary_plain(dec)
        du.print_sig_codes()

        print(f"\n{model_utils.model_fit(self.fitted, dec=dec)}")

        if vif:
            if self.evar is None or len(self.evar) < 2:
                print("\nVariance Inflation Factors cannot be calculated")
            else:
                print("\nVariance inflation factors:")
                print(f"\n{model_utils.vif(self.fitted).to_string()}")

        if ssq:
            print("\nSum of squares:")
            index = ["Regression", "Error", "Total"]
            sum_of_squares = [
                self.fitted.ess,
                self.fitted.ssr,
                self.fitted.centered_tss,
            ]
            sum_of_squares = pd.DataFrame(index=index).assign(
                df=format_nr(
                    [
                        self.fitted.df_model,
                        self.fitted.df_resid,
                        self.fitted.df_model + self.fitted.df_resid,
                    ],
                    dec=0,
                ),
                SS=format_nr(sum_of_squares, dec=0),
            )
            print(f"\n{sum_of_squares.to_string()}")

        if rmse:
            rmse_val, rsd_val = self.rmse()
            print(f"\nPrediction error (RMSE): {round(rmse_val, dec)}")
            print(f"Residual st.dev   (RSD): {round(rsd_val, dec)}")

        if test is not None and len(test) > 0:
            self.f_test(test=test, dec=dec)

        if ci:
            print("\nConfidence intervals:")
            df = model_utils.coef_ci(self.fitted, alpha=1 - conf, dec=dec)
            print(f"\n{df.to_string()}")

    def rmse(self) -> tuple[float, float]:
        """Root mean squared error and residual standard deviation."""
        rmse_val = (self.fitted.ssr / self.fitted.nobs) ** 0.5
        rsd_val = (self.fitted.ssr / self.fitted.df_resid) ** 0.5
        return rmse_val, rsd_val

    def _summary_header(self) -> None:
        """Print the summary header."""
        print("Linear regression (OLS)")
        print("Data                 :", self.name)
        print_scope(self, 21)
        print("Response variable    :", self.rvar)
        print("Explanatory variables:", ", ".join(self.evar))
        if self.stepwise:
            removed = ", ".join(self.removed) if self.removed else "none"
            print("Stepwise selection   : backward (AIC), removed", removed)
        print(f"Null hyp.: the effect of x on {self.rvar} is zero")
        print(f"Alt. hyp.: the effect of x on {self.rvar} is not zero")
        if self.standardize:
            print(f"**Standardized coefficients shown ({self.standardize} X SD)**")
        elif self.center:
            print("**Centered coefficients shown (x - mean(x))**")
        if self.robust:
            print("**Robust standard errors used**")
        print()

    def _summary_plain(self, dec: int = 3) -> None:
        """Print plain text coefficient table."""
        df = self.coef.clone()
        df = df.with_columns(
            [
                pl.col("coefficient").round(dec),
                pl.col("std.error").round(dec),
                pl.col("t.value").round(dec),
                pl.when(pl.col("p.value") < 0.001)
                .then(pl.lit("< .001"))
                .otherwise(pl.col("p.value").round(dec).cast(pl.Utf8))
                .alias("p.value"),
            ]
        )

        # pl.col("index").str.replace("[T.", "[", literal=True),
        du.print_plain_tables(df)

    def _summary_styled(self, dec: int = 3) -> None:
        """Display styled coefficient table in notebooks."""
        from IPython.display import display

        df = self.coef.clone()
        df = df.with_columns(
            [
                pl.col("index"),
                pl.col("p.value").map_elements(
                    lambda p: du.format_pval(p, dec), return_dtype=pl.Utf8
                ),
            ]
        )

        gt = du.style_table(
            df,
            title="Coefficient Estimates",
            subtitle=f"Response: {self.rvar}",
            number_cols=["coefficient", "std.error", "t.value"],
            dec=dec,
        )
        display(gt)

    def predict(
        self, data=None, cmd=None, data_cmd=None, ci=False, conf=0.95, dec=None
    ) -> pl.DataFrame:
        """
        Generate predictions using the fitted model.

        Parameters
        ----------
        data : pd.DataFrame | pl.DataFrame, optional
            Data used to generate predictions. If not provided, the estimation data will be used.
        cmd : dict[str, Union[int, list[int]]], optional
            Dictionary with the names of the columns to be used in the prediction and the values to be used.
        data_cmd : dict[str, Union[int, list[int]]], optional
            Dictionary with the names of the columns to be used in the prediction and the values to be used.
        ci : bool, default False
            Calculate confidence intervals for the predictions.
        conf : float, default 0.95
            Confidence level for the intervals.
        dec : int, optional
            Number of decimal places to round float columns in the output.
            If None (default), no rounding is applied.

        Returns
        -------
        pl.DataFrame
            DataFrame containing the predictions and the data used to make those predictions.
        """
        model_utils = _model_utils()
        if data is None:
            pred_data = self.data_raw.select(self.evar)
        else:
            data = check_dataframe(data)
            pred_data = data.select(self.evar)
            # Apply stored Enum types to ensure consistent categorical levels
            if self._enum_types:
                pred_data = model_utils.apply_enum_types(pred_data, self._enum_types)

        if data_cmd is not None:
            # Apply data_cmd values to pred_data
            pred_data = pred_data.with_columns(
                [pl.lit(v).alias(k) for k, v in data_cmd.items()]
            )
            # Re-apply Enum types after modifying columns
            if self._enum_types:
                pred_data = model_utils.apply_enum_types(pred_data, self._enum_types)
        elif cmd is not None:
            cmd = {k: ifelse(isinstance(v, str), [v], v) for k, v in cmd.items()}
            pred_data = model_utils.sim_prediction(data=pred_data, vary=cmd)
            # Re-apply Enum types after sim_prediction creates new data
            if self._enum_types:
                pred_data = model_utils.apply_enum_types(pred_data, self._enum_types)

        # A standardized/centered model is evaluated on scaled inputs; the
        # output shows the inputs and the prediction in original units.
        model_data = self._scale(pred_data) if self.scale_means else pred_data

        if ci:
            if data_cmd is not None:
                raise ValueError(
                    "Confidence intervals not available when using the Data & Command option"
                )
            else:
                # predict_ci returns pandas, convert result to polars
                ci_result = pl.from_pandas(
                    model_utils.predict_ci(self.fitted, model_data.to_pandas(), conf=conf)
                )
                if self.scale_means:
                    ci_result = ci_result.with_columns(
                        [self._unscale_response(pl.col(c)) for c in ci_result.columns]
                    )
                pred = pl.concat([pred_data, ci_result], how="horizontal")
        else:
            # Get predictions from statsmodels (needs pandas), add to polars DataFrame
            predictions = self.fitted.predict(model_data.to_pandas()).values
            if self.scale_means:
                predictions = self._unscale_response(predictions)
            pred = pred_data.with_columns(pl.lit(predictions).alias("prediction"))

        if dec is not None:
            pred = pred.with_columns(
                [
                    pl.col(c).round(dec)
                    for c in pred.columns
                    if pred[c].dtype in [pl.Float64, pl.Float32]
                ]
            )
        return pred

    def plot(
        self,
        plots="dist",
        data=None,
        alpha=0.05,
        nobs: int = 1000,
        intercept=False,
        incl=None,
        excl=None,
        incl_int=None,
        lines=None,
        fix=True,
        hline=False,
        ice=False,
        ice_nobs=100,
        nnv=20,
        minq=0.025,
        maxq=0.975,
        figsize=None,
        ret=None,
    ) -> None:
        """
        Plots for a linear regression model

        Parameters
        ----------
        plots : str or list[str], default 'dist'
            List of plot types to generate. Options include 'dist', 'corr', 'scatter', 'dashboard', 'residual', 'pred', 'pdp', 'pip', 'coef', 'influence'.
        alpha : float, default 0.05
            Significance level for the confidence intervals in a coefficient plot (coef).
        nobs : int, default 1000
            Number of observations to plot. Relevant for all plots that include a scatter of data points (i.e., corr, scatter, dashboard, residual).
        incl : list[str], optional
            Variables to include in the plot. Relevant for prediction plots (pred) and coefficient plots (coef).
        excl : list[str], optional
            Variables to exclude from the plot. Relevant for prediction plots (pred) and coefficient plots (coef).
        incl_int : list[str], optional
            Interaction terms to include in the plot. Relevant for prediction plots (pred).
        lines : list[str], optional
            Lines to add to 'scatter', 'residual' and 'dashboard' plots: "line"
            (linear fit), "loess" (local polynomial fit), and, for scatter and
            residual plots, "jitter" (jitter the points). For 'dashboard', None
            keeps the default linear fit lines.
        fix : bool, default True
            Fix the y-axis limits. Relevant for prediction plots (pred).
        hline : bool, default False
            Add a horizontal line to the plot at the mean of the response variable. Relevant for prediction plots (pred).
        ice : bool, default False
            Show Individual Conditional Expectation (ICE) lines behind the PDP.
            ICE lines show the prediction for each observation as the feature varies.
            Only applies to 'pdp' plots.
        ice_nobs : int, default 100
            Number of observations to sample for ICE lines. Only used when ice=True.
            Use -1 to show ICE lines for all observations.
        nnv : int, default 20
            Number of predicted values to calculate and to plot. Relevant for prediction plots.
        minq : float, default 0.025
            Minimum quantile of the explanatory variable values to use to calculate and plot predictions.
        maxq : float, default 0.975
            Maximum quantile of the explanatory variable values to use to calculate and plot predictions.
        figsize : tuple[int, int], default None
            Figure size for the plots in inches (e.g., "(3, 6)"). Relevant for 'corr', 'scatter', 'residual', and 'coef' plots.
        ret : bool, optional
            Whether to return the variable (permutation) importance scores for a "pip" plot.
        """
        model_utils = _model_utils()
        viz_utils = _visualize_utils()

        plots = model_utils.convert_to_list(
            plots
        )  # control for the case where a single string is passed
        excl = model_utils.convert_to_list(excl)
        incl = ifelse(incl is None, None, model_utils.convert_to_list(incl))
        incl_int = model_utils.convert_to_list(incl_int)

        if data is None:
            data = self.data
        else:
            data = check_dataframe(data)

        # Select relevant columns
        if self.rvar in data.columns:
            data = data.select([self.rvar] + self.evar)
        else:
            data = data.select(self.evar)

        # Apply stored Enum types to ensure consistent categorical levels
        if self._enum_types:
            data = model_utils.apply_enum_types(data, self._enum_types)

        if "dist" in plots:
            distr_plot, _, _, _ = _get_visualize_plots()
            return distr_plot(data).plot()
        if "corr" in plots:
            from pyrsm.basics.correlation import correlation

            cr = correlation(data)
            return cr.plot(nobs=nobs, figsize=figsize)
        if "scatter" in plots:
            return model_utils.scatter_plot(
                self.fitted, data, nobs=nobs, figsize=figsize, lines=lines
            )
        if "dashboard" in plots:
            return model_utils.reg_dashboard(self.fitted, nobs=nobs, lines=lines)
        if "residual" in plots:
            return model_utils.residual_plot(
                self.fitted, data, nobs=nobs, figsize=figsize, lines=lines
            )
        if "influence" in plots:
            return model_utils.influence_plot(self.fitted)
        if "pred" in plots:
            return viz_utils.pred_plot_sm(
                self.fitted,
                data=data,
                incl=incl,
                excl=excl,
                incl_int=incl_int,
                fix=fix,
                hline=hline,
                nnv=nnv,
                minq=minq,
                maxq=maxq,
            )
        if "pdp" in plots:
            _, _, pdp_sm, _ = _get_visualize_plots()
            return pdp_sm(
                self.fitted,
                data=data,
                incl=incl,
                excl=excl,
                incl_int=incl_int,
                ice=ice,
                ice_nobs=ice_nobs,
                grid_resolution=nnv,
                minq=minq,
                maxq=maxq,
                fix=fix,
                hline=hline,
            )

        if "pip" in plots or "vimp" in plots:
            return_pip, p = viz_utils.pip_plot_sm(
                self.fitted,
                data=data,
                rep=10,
                ret=ret,
            )
            if ret:
                return (p, return_pip)
            return p
        if "coef" in plots:
            return model_utils.coef_plot(
                self.fitted,
                alpha=alpha,
                intercept=intercept,
                incl=incl,
                excl=excl,
                figsize=figsize,
            )

    def f_test(self, test: str | None = None, dec: int = 3) -> None:
        """
        F-test for competing models

        Parameters
        ----------
        test : list[str] or None, optional
            List of variable names used in the model to test using an F-test or None if all variables are to be tested.
        dec : int, default 3
            Number of decimal places to round to.
        """
        if self.stepwise:
            print("\nModel comparisons are not conducted when Stepwise has been selected.")
            return
        if isinstance(test, str):
            test = [v.strip() for v in test.split(",") if v.strip()]
        test = list(test)
        evar = setdiff(self.evar, test)
        if self.ivar is not None and len(self.ivar) > 0:
            sint = setdiff(self.ivar, test)
            test += [s for t in test for s in sint if f"I({t}" not in s and t in s]
            sint = setdiff(sint, test)
        else:
            sint = []

        formula = f"{self.rvar} ~ "
        if len(evar) == 0 and len(sint) == 0:
            formula += "1"
        else:
            formula += f"{' + '.join(evar + sint)}"
        pattern = r"(\[T\.[^\]]*\])\:"

        # ensure constraints are unique
        hypothesis = list(
            set(
                [
                    f"({c} = 0)"
                    for c in self.fitted.model.exog_names
                    for v in test
                    if f"{v}:" in c
                    or f":{v}" in c
                    or f"{v}[T." in c
                    or v == c
                    or v == re.sub(pattern, ":", c)
                ]
            )
        )

        print(f"\nModel 1: {formula}")
        print(f"Model 2: {self.formula}")
        out = self.fitted.f_test(hypothesis)

        r2_sub = self.fitted.rsquared - (
            len(hypothesis) * out.fvalue * (1 - self.fitted.rsquared)
        ) / (self.fitted.nobs - self.fitted.df_model - 1)

        pvalue = ifelse(out.pvalue < 0.001, "< .001", round(out.pvalue, dec))
        print(f"R-squared, Model 1 vs 2: {r2_sub:.3f} vs {self.fitted.rsquared:.3f}")
        print(
            f"F-statistic: {round(out.fvalue, dec)} df ({out.df_num:.0f}, {out.df_denom:.0f}), p.value {pvalue}"
        )
