import re
from functools import lru_cache

import pandas as pd
import polars as pl

from pyrsm.basics import display_utils as du
from pyrsm.utils import check_dataframe, complete_rows, format_nr, ifelse, setdiff
from pyrsm.data_scope import apply_scope, print_scope


@lru_cache(maxsize=1)
def _fit_like_r(model, cov_type: str = "nonrobust"):
    from pyrsm.multivariate._lm import fit_like_r

    return fit_like_r(model, cov_type=cov_type)


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
    stepwise : str
        Stepwise selection direction ("backward", "forward", "both"), or ""
        when no stepwise selection was applied.
    robust : bool
        Whether heteroskedasticity-robust (HC1) standard errors are used.
    removed : list[str]
        Terms dropped by stepwise selection.
    step_trace : str
        The stepwise selection trace, laid out as R's ``step()`` prints it.
    step_steps : list[dict]
        The same trace as data: per step its label, AIC, formula and a table of
        the candidate changes (``term``, the statistics, ``AIC``).
    fitted_inference : statsmodels results
        The results used for coefficient inference: ``fitted`` itself, or its
        HC1 robust version when ``robust=True``.

    Methods
    -------
    __init__(data, rvar=None, evar=None, ivar=None, form=None)
        Initialize the regress class with the provided data and parameters.
    summary(main=True, fit=True, ci=False, ssq=False, rmse=False, vif=False, test=None, dec=3)
        Summarize the model output.
    plot(plot_type, nobs=1000, incl=None, excl=None, incl_int=None, fix=True, hline=True, nnv=20, minq=0.025, maxq=0.975)
        Plots for the model.
    predict(data=None, cmd=None, data_cmd=None, ci=False, conf=0.95, interval="confidence")
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
        stepwise: bool | str = False,
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
        stepwise : bool or str, default False
            Stepwise selection of explanatory variables by AIC, as R's
            ``step()``: ``"backward"`` (or ``True``) starts from the full model
            and drops terms, ``"forward"`` starts from an intercept-only model
            and adds terms, ``"both"`` starts from the intercept-only model and
            may add or drop at each step. AIC is ``n log(RSS/n) + 2k`` with
            ``k`` the number of estimated coefficients (R's ``extractAIC``).
            A main effect is not dropped while an interaction that contains it
            is in the model, and an interaction is not added before its main
            effects. ``summary()`` prints the selection trace.
        robust : bool, default False
            Heteroskedasticity-robust (HC1, White/MacKinnon) standard errors,
            as ``sandwich::vcovHC(type = "HC1")`` in Radiant. They are used
            for the coefficient table, confidence intervals, the coefficient
            plot and F-tests on ``test`` variables. The coefficients, R-squared,
            the overall F-statistic and prediction intervals are the classical
            OLS ones, as in Radiant.
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
        # radiant.model's squared / cubed terms, I(x^2), in Python's spelling
        self.ivar = model_utils.r_powers(model_utils.convert_to_list(ivar))
        self.formula = model_utils.r_powers(formula) if formula else formula

        # Convert categorical/string columns to Enum BEFORE fitting (Enum level
        # order is kept; text columns get their order of appearance) so patsy
        # uses the same level order for estimation and prediction
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
        if stepwise is True:
            stepwise = "backward"
        if stepwise and stepwise not in ("backward", "forward", "both"):
            raise ValueError("stepwise must be 'backward', 'forward' or 'both'")
        self.stepwise = stepwise or ""
        self.robust = bool(robust)
        self.removed: list[str] = []
        self.added: list[str] = []
        self.step_trace = ""
        self.aliased: list[str] = []
        self.step_steps: list[dict] = []

        import statsmodels.formula.api as smf

        if not self.formula:
            if self.evar is None or len(self.evar) == 0:
                self.formula = f"{self.rvar} ~ 1"
            else:
                self.formula = f"{self.rvar} ~ {' + '.join(self.evar)}"
            if self.ivar:
                self.formula += f" + {' + '.join(self.ivar)}"
            model = smf.ols(self.formula, data=model_utils.to_model_pandas(self.data))
        else:
            model = smf.ols(formula=self.formula, data=model_utils.to_model_pandas(self.data))
            self.evar = model_utils.extract_evars(model, self.data.columns)
            self.rvar = model_utils.extract_rvar(model, self.data.columns)

        # As R's lm(): levels that do not occur in the estimation rows are
        # dropped (the next level becomes the base), and collinear terms are
        # dropped rather than split by a pseudo-inverse (see ``_ols``)
        used = model_utils.formula_vars(model, self.data.columns)
        self.data = model_utils.drop_unused_levels(self.data, used)
        self._enum_types = {
            c: self.data[c].dtype for c in self._enum_types if c in self.data.columns
        }

        # The data in original units. Standardizing or centering changes what
        # is estimated, not what a student types for a prediction, so
        # ``predict`` starts from these values.
        self.data_raw = self.data
        self.scale_means: dict[str, float] = {}
        self.scale_sds: dict[str, float] = {}
        if self.standardize or self.center:
            self._calc_scaling()
            self.data = self._scale(self.data)

        pdf = model_utils.to_model_pandas(self.data)
        self.fitted = self._ols(self.formula, pdf)

        if self.stepwise:
            self._step(pdf)
        self.aliased = self.fitted.aliased

        self.fitted.nobs_dropped = self.data.height - self.fitted.nobs

        # Coefficient inference: robust (HC1) when asked, classical otherwise.
        if self.robust:
            # Same model, HC1 covariance; t distribution with n - k df as R.
            # Columns R drops as aliased stay dropped (sandwich::vcovHC).
            self.fitted_inference = _fit_like_r(self.fitted.model, cov_type="HC1")
        else:
            self.fitted_inference = self.fitted
        inf = self.fitted_inference
        #: Columns dropped because of perfect multicollinearity, as R's lm()
        #: drops them (NA coefficients). They are left out of the table.
        self.aliased = list(getattr(self.fitted, "aliased", []))
        keep = [n not in self.aliased for n in inf.params.index]
        params, bse, tvalues, pvalues = (
            inf.params[keep], inf.bse[keep], inf.tvalues[keep], inf.pvalues[keep]
        )

        # Build coefficient table as polars DataFrame
        self.coef = pl.DataFrame(
            {
                "index": [
                    model_utils.r_label(i)
                    for i in params.index.str.replace("[T.", "[", regex=False)
                ],
                "coefficient": params.values,
                "std.error": bse.values,
                "t.value": tvalues.values,
                "p.value": pvalues.values,
                "  ": model_utils.sig_stars(pvalues.values),
            }
        ).drop_nulls(subset=["coefficient"])

    def _ols(self, formula: str, pdf: pd.DataFrame):
        """Fit OLS (classical covariance; see ``fitted_inference`` for robust).

        Perfectly collinear columns are dropped as R's ``lm()`` drops them
        (``fitted.aliased``) instead of spreading the effect over them with a
        pseudo-inverse, so the coefficients, standard errors and stepwise
        steps are R's.
        """
        import statsmodels.formula.api as smf

        return _fit_like_r(smf.ols(formula=formula, data=pdf))

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

    def _step(self, pdf: pd.DataFrame) -> None:
        """Stepwise selection by AIC, as R's ``step()`` (and its printed trace)."""
        import numpy as np

        def aic(fit) -> float:
            # R's extractAIC for lm: n log(RSS/n) + 2 edf
            n = fit.nobs
            return n * np.log(fit.ssr / n) + 2 * (n - fit.df_resid)

        def stats(fit, base) -> list:
            # drop1 / add1 for lm: Df, Sum of Sq, RSS
            if base is None:
                return [None, None, fit.ssr]
            return [abs(fit.df_resid - base.df_resid), abs(fit.ssr - base.ssr), fit.ssr]

        out = _model_utils().step_select(
            full_fit=self.fitted,
            formula=self.formula,
            rvar=self.rvar,
            direction=self.stepwise,
            pdf=pdf,
            fit_formula=self._ols,
            aic=aic,
            stats=stats,
            columns=["Df", "Sum of Sq", "RSS"],
        )
        self.step_trace = out["trace"]
        self.step_steps = out["steps"]
        self.removed, self.added = out["removed"], out["added"]
        self.formula = out["formula"]
        self.fitted = out["fit"]
        self.ivar = [t for t in self.ivar if t in out["current"]]

    def _model_evar(self) -> list[str]:
        """Explanatory variables still used by the (possibly stepwise) model."""
        if not self.removed:
            return list(self.evar)
        design = self.fitted.model.data.design_info
        used = " ".join(t.name() for t in design.terms)
        return [v for v in self.evar if re.search(rf"\b{re.escape(v)}\b", used)]

    def summary_data(
        self,
        vif: bool = False,
        ssq: bool = False,
        rmse: bool = False,
        test: list[str] | None = None,
        ci: bool = False,
        dec: int = 3,
    ) -> dict:
        """Return the complete, JSON-safe linear-regression summary document.

        This is additive: :meth:`summary` retains its established printed output.
        Values and section selection live here so a browser never has to rebuild a
        partial regression summary from incidental result attributes.
        """
        from pyrsm.summary import column, document, key_value_section, table_rows, table_section

        model_utils = _model_utils()
        coefficient_columns = [
            column("index", semantic_type="text"),
            column("coefficient", semantic_type="number", decimals=dec),
            column("std.error", semantic_type="number", decimals=dec),
            column("t.value", semantic_type="number", decimals=dec),
            column("p.value", semantic_type="p_value", decimals=dec),
            column("  ", semantic_type="significance"),
        ]
        sections = [
            table_section(
                "coefficients",
                "Coefficients",
                table_rows(self.coef),
                coefficient_columns,
                footnotes=["Significance codes: *** < .001, ** < .01, * < .05, . < .1."],
            ),
            table_section(
                "model_fit",
                "Model fit",
                table_rows(model_utils.model_fit(self.fitted, dec=dec, prn=False)),
                [column(name, semantic_type="p_value" if "pval" in name else "number", decimals=dec)
                 for name in model_utils.model_fit(self.fitted, dec=dec, prn=False).columns],
            ),
        ]
        if vif:
            if self.evar is None or len(self.evar) < 2:
                sections.append(key_value_section("vif", "Variance inflation factors", {"available": False}))
            else:
                vif_table = model_utils.vif(self.fitted, dec=dec)
                sections.append(
                    table_section(
                        "vif",
                        "Variance inflation factors",
                        table_rows(vif_table, include_index=True),
                        [column(name, semantic_type="number", decimals=dec) for name in ["index", *vif_table.columns]],
                    )
                )
        if ssq:
            sections.append(
                table_section(
                    "sum_of_squares",
                    "Sum of squares",
                    [
                        {"index": "Regression", "df": self.fitted.df_model, "SS": self.fitted.ess},
                        {"index": "Error", "df": self.fitted.df_resid, "SS": self.fitted.ssr},
                        {"index": "Total", "df": self.fitted.df_model + self.fitted.df_resid, "SS": self.fitted.centered_tss},
                    ],
                    [column("index"), column("df", semantic_type="number", decimals=0), column("SS", semantic_type="number", decimals=0)],
                )
            )
        if rmse:
            sections.append(
                key_value_section(
                    "rmse",
                    "Root Mean Square Error (RMSE)",
                    {"RMSE": (self.fitted.ssr / self.fitted.nobs) ** 0.5},
                )
            )
        if ci:
            confidence_intervals = model_utils.coef_ci(self.fitted, dec=dec)
            sections.append(
                table_section(
                    "confidence_intervals",
                    "Confidence intervals",
                    table_rows(confidence_intervals, include_index=True),
                    [
                        column(name, semantic_type="p_value" if name == "p.values" else "number" if name != "index" and name != "  " else "text", decimals=dec)
                        for name in ["index", *confidence_intervals.columns]
                    ],
                )
            )
        if test:
            sections.append(
                key_value_section(
                    "requested_test",
                    "Requested nested-model test",
                    {"variables": ", ".join(test), "status": "Run with f_test() for the established printed comparison."},
                )
            )
        return document(
            producer="pyrsm.model.regress.summary_data",
            title="Linear regression (OLS)",
            analysis_kind="linear_regression",
            options={"vif": vif, "ssq": ssq, "rmse": rmse, "test": test or [], "ci": ci, "dec": dec},
            metadata={
                "Data": self.name,
                "Response variable": self.rvar,
                "Explanatory variables": ", ".join(self.evar),
                "Formula": self.formula,
                "Null hypothesis": f"the effect of x on {self.rvar} is zero",
                "Alternative hypothesis": f"the effect of x on {self.rvar} is not zero",
            },
            sections=sections,
        )


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
        plain : bool, default True
            If True (default), print plain text output. If False and running
            in a notebook (or a rendered Quarto report), show the formatted
            report Radiant shows.
        """
        if not plain and du.is_notebook():
            # Radiant's formatted report (pyrsm.report), as its Summary tab shows it
            from pyrsm.model.report import linear_report

            opts = dict(vif=vif, ssq=ssq, rmse=rmse, test=test, ci=ci, conf=conf, dec=dec)
            return du.show_report(self, linear_report(self, **opts), **opts)

        model_utils = _model_utils()

        if self.stepwise:
            # As Radiant's report: the selection trace, then the summary.
            print(self.step_trace)
            title = {
                "backward": "Backward",
                "forward": "Forward",
                "both": "Forward and Backward",
            }[self.stepwise]
            line = "-" * 52
            print(f"\n{line}\n{title} stepwise selection of variables\n{line}")

        self._summary_header()
        self._summary_plain(dec)
        du.print_sig_codes()

        print(f"\n{model_utils.model_fit(self.fitted, dec=dec)}")
        if self.aliased:
            print(f"\n{self.collinearity_note()}")

        if vif:
            if self.aliased:
                # R: car::vif() needs a full-rank model
                print("\nMulticollinearity diagnostics were not calculated.")
            elif self.evar is None or len(self.evar) < 2:
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
            if self.aliased:
                print("\nConfidence intervals were not calculated.")
            else:
                print("\nConfidence intervals:")
                df = model_utils.coef_ci(self.fitted_inference, alpha=1 - conf, dec=dec)
                print(f"\n{df.to_string()}")

    def collinearity_note(self) -> str:
        """R's perfect-multicollinearity message, naming the dropped columns
        (empty when nothing was dropped)."""
        if not self.aliased:
            return ""
        r_label = _model_utils().r_label
        names = ", ".join(r_label(a.replace("[T.", "[")) for a in self.aliased)
        return (
            "The set of explanatory variables exhibit perfect multicollinearity.\n"
            f"One or more variables were dropped from the estimation: {names}"
        )

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
            if self.stepwise == "backward":
                change = "removed: " + (", ".join(self.removed) or "none")
            else:
                kept = self.formula.split("~", 1)[1].strip()
                change = "selected: " + kept
            print(f"Stepwise selection   : {self.stepwise} (AIC), {change}")
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
                du.format_pval_expr("p.value", dec),
            ]
        )

        # pl.col("index").str.replace("[T.", "[", literal=True),
        du.print_plain_tables(df)

    def predict(
        self, data=None, cmd=None, data_cmd=None, ci=False, conf=0.95, dec=None,
        interval="confidence",
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
            Add an interval to each prediction (see ``interval``).
        conf : float, default 0.95
            Confidence level for the intervals.
        dec : int, optional
            Number of decimal places to round float columns in the output.
            If None (default), no rounding is applied.
        interval : str, default "confidence"
            With ``ci=True``: "confidence" for an interval around the mean of
            the response at these values of the explanatory variables, or
            "prediction" for an interval around a single new value of the
            response. A prediction interval also carries the residual variance,
            so it is always wider. Both use the t distribution, as R's
            ``predict.lm``.

        Returns
        -------
        pl.DataFrame
            DataFrame containing the predictions and the data used to make those predictions.
        """
        from pyrsm.model.model import check_pred_vars, ols_interval

        if interval not in ("confidence", "prediction"):
            raise ValueError("interval must be 'confidence' or 'prediction'")

        check_pred_vars(
            self.evar,
            data=None if data is None else check_dataframe(data),
            cmd=cmd if isinstance(cmd, dict) else None,
            data_cmd=data_cmd if isinstance(data_cmd, dict) else None,
        )
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
            ci_result = pl.from_pandas(
                ols_interval(
                    self.fitted, model_utils.to_model_pandas(model_data), conf=conf,
                    interval=interval,
                )
            )
            # Standardizing is linear, so the interval converts back to the
            # response's units exactly.
            if self.scale_means:
                ci_result = ci_result.with_columns(
                    [self._unscale_response(pl.col(c)) for c in ci_result.columns]
                )
            pred = pl.concat([pred_data, ci_result], how="horizontal")
        else:
            # Get predictions from statsmodels (needs pandas), add to polars DataFrame
            predictions = self.fitted.predict(model_utils.to_model_pandas(model_data)).values
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
        hline=True,
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
        hline : bool, default True
            Add a horizontal line at the mean of the response variable (the
            average response). Relevant for prediction plots (pred) and partial
            dependence plots (pdp), as for every other model.
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
        # The rows the model was estimated on: a row missing any model
        # variable has no fitted value or residual
        data = data.filter(complete_rows(data, data.columns))

        # Apply stored Enum types to ensure consistent categorical levels
        if self._enum_types:
            data = model_utils.apply_enum_types(data, self._enum_types)
        # dates enter the model as numbers (as in R's lm)
        data = model_utils.dates_to_numeric(data)

        if "dist" in plots:
            distr_plot, _, _, _ = _get_visualize_plots()
            return distr_plot(data).plot()
        if "corr" in plots:
            from pyrsm.model.model import corr_plot

            return corr_plot(data, nobs=nobs, figsize=figsize)
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
                self.fitted_inference,
                alpha=alpha,
                intercept=intercept,
                incl=incl,
                excl=excl,
                figsize=figsize,
            )

    def _f_test_aliased(self, formula: str, hypothesis: list[str], dec: int) -> None:
        """The model comparison when columns were dropped as aliased.

        A Wald test on the full model would test the dropped columns too, or
        count columns that only become estimable once others are removed. R
        compares the two fits instead (``anova(sub, full)``): the df are the
        difference in rank. With robust standard errors R tests the kept
        coefficients of the tested terms (car::linearHypothesis drops aliased
        ones).
        """
        sub = self._ols(formula, self.fitted.model.data.frame)
        full = self.fitted
        df_num = int(round(sub.df_resid - full.df_resid))
        r2_sub = 1 - sub.ssr / full.centered_tss
        if df_num <= 0:
            # Dropping the tested terms does not change the fit (their columns
            # were aliased, or others take their place): R's anova() gives no
            # p-value and nothing is printed.
            return
        if self.robust:
            # R stops here (linearHypothesis: "there are aliased coefficients
            # in the model"); test the kept coefficients of the tested terms.
            kept = [h for h in hypothesis if h[1:].split(" = ")[0] not in self.aliased]
            if not kept:
                return
            out = self.fitted_inference.f_test(kept)
            fvalue, pval = float(out.fvalue), float(out.pvalue)
            df_num = int(out.df_num)
        else:
            from scipy import stats

            fvalue = ((sub.ssr - full.ssr) / df_num) / (full.ssr / full.df_resid)
            pval = float(stats.f.sf(fvalue, df_num, full.df_resid))
        pvalue = ifelse(pval < 0.001, "< .001", round(pval, dec))
        r_label = _model_utils().r_label
        print(f"\nModel 1: {r_label(formula)}")
        print(f"Model 2: {r_label(self.formula)}")
        # + 0.0: an intercept-only model's R-squared rounds to 0.000, not -0.000
        print(f"R-squared, Model 1 vs 2: {round(r2_sub, 3) + 0.0:.3f} vs {full.rsquared:.3f}")
        print(
            f"F-statistic: {round(fvalue, dec)} df ({df_num:.0f}, {full.df_resid:.0f}), p.value {pvalue}"
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
        test = _model_utils().r_powers(list(test))
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

        r_label = _model_utils().r_label
        if self.aliased:
            self._f_test_aliased(formula, hypothesis, dec)
            return
        print(f"\nModel 1: {r_label(formula)}")
        print(f"Model 2: {r_label(self.formula)}")
        # The F-test uses the robust covariance when ``robust`` is set (R's
        # car::linearHypothesis with a vcov); the R-squared of the restricted
        # model comes from the classical F, which is exact for OLS.
        out = self.fitted_inference.f_test(hypothesis)
        classical = self.fitted.f_test(hypothesis)
        fvalue, fpval = float(pd.Series(out.fvalue).squeeze()), float(out.pvalue)
        df_num, df_denom = out.df_num, out.df_denom
        r2_sub = self.fitted.rsquared - (
            len(hypothesis)
            * float(pd.Series(classical.fvalue).squeeze())
            * (1 - self.fitted.rsquared)
        ) / (self.fitted.nobs - self.fitted.df_model - 1)

        pvalue = du.format_pval(fpval, dec)
        print(f"R-squared, Model 1 vs 2: {round(r2_sub, 3) + 0.0:.3f} vs {self.fitted.rsquared:.3f}")
        print(
            f"F-statistic: {round(fvalue, dec)} df ({df_num:.0f}, {df_denom:.0f}), p.value {pvalue}"
        )

