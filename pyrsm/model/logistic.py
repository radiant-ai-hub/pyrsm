from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit

from pyrsm.basics import display_utils as du
from pyrsm.model.model import (
    apply_enum_types,
    check_binary,
    convert_categoricals_to_enum,
    convert_to_list,
    model_fit,
    or_ci,
    or_plot,
    predict_ci,
    sig_stars,
    sim_prediction,
)
from pyrsm.model.model import vif as calc_vif
from pyrsm.model.visualize import (
    extract_evars,
    extract_rvar,
)
from pyrsm.utils import check_dataframe, ifelse, setdiff
from pyrsm.data_scope import apply_scope, print_scope


@lru_cache(maxsize=1)
def _get_scipy_stats():
    """Lazy load scipy.stats."""
    from scipy import stats

    return stats


@lru_cache(maxsize=1)
def _get_visualize_plots():
    """Lazy load visualize plot functions."""
    from pyrsm.eda.distr import distr
    from pyrsm.model.visualize import pdp_sm, pip_plot_sm, pred_plot_sm

    return distr, pred_plot_sm, pdp_sm, pip_plot_sm


@lru_cache(maxsize=1)
def _get_correlation():
    """Lazy load correlation class."""
    from pyrsm.basics.correlation import correlation

    return correlation


def _formula_model(formula: str, data: pl.DataFrame):
    """The design of ``formula`` (no fit), to read its explanatory variables."""
    return smf.glm(formula, data=data.to_pandas(), family=Binomial(link=Logit()))


class logistic:
    """
    A class to perform logistic regression modeling (binary classification)

    Attributes
    ----------
    data : pl.DataFrame
        Dataset used for the analysis (stored as polars DataFrame).
    name : str
        Name of the dataset if provided as a dictionary.
    rvar : str
        Name of the response variable in the data.
    lev: str
        Name of the level in the response variable the model will predict.
    evar : list[str]
        List of column names of the explanatory variables.
    ivar : list[str]
        List of strings with the names of the columns included as explanatory variables (e.g., ["x1:x2", "x3:x4"])
    formula : str
        Model specification formula.
    fitted : statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        The fitted model.
    coef : pl.DataFrame
        The estimated model coefficients with standard errors, p-values, etc.
    weights: pl.Series or None
        Frequency weights used in the model if provided.
    weights_name: str or None
        Column name for the variable in the data containing frequency weights if provided.
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
    """

    def __init__(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, pd.DataFrame | pl.DataFrame],
        rvar: str | None = None,
        lev: str | None = None,
        evar: list[str] | None = None,
        ivar: list[str] | None = None,
        formula: str | None = None,
        weights: str | None = None,
        data_filter: str = "",
        sort: str = "",
        slice: str = "",
        standardize: bool | int | float = False,
        center: bool = False,
        stepwise: bool | str = False,
        robust: bool = False,
        ci_type: str | None = None,
    ) -> None:
        """
        Logistic regression (GLM, binomial family, logit link)

        Parameters
        ----------
        standardize : bool or int, default False
            Standardize the numeric explanatory variables by 1 or 2 (``True``)
            standard deviations (weighted when ``weights`` is set), as
            radiant.model. The response is not scaled; odds ratios are then per
            1 or 2 standard deviations.
        center : bool, default False
            Mean-center the numeric explanatory variables. Ignored when
            ``standardize`` is set.
        stepwise : bool or str, default False
            Stepwise selection by AIC as R's ``step()``: ``"backward"`` (or
            ``True``), ``"forward"`` or ``"both"``. ``summary()`` prints the trace.
        robust : bool, default False
            Heteroskedasticity-consistent (HC1) standard errors, as
            ``sandwich::vcovHC(type = "HC1")``, with normal (z) tests. Used for
            the coefficient table, confidence intervals, odds ratios, the OR
            plot and tests on ``test`` variables. Turned on automatically, as
            in radiant.model, for sampling weights: non-integer weights with
            two values, the smallest below 1.
        ci_type : str, optional
            Coefficient confidence intervals: ``"profile"`` (profile
            likelihood, R's ``confint``) or ``"wald"`` (R's
            ``confint.default``). By default profile likelihood below 5,000
            observations and Wald otherwise, as radiant.model. Robust standard
            errors always use Wald intervals.
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
        self.lev = lev
        self.evar = convert_to_list(evar)
        # radiant.model's squared / cubed terms, I(x^2), in Python's spelling
        from pyrsm.model.model import r_powers

        self.ivar = r_powers(convert_to_list(ivar))
        self.formula = r_powers(formula) if formula else formula

        # The response as the student knows it (e.g. Yes / No), for plots
        self.rv = (
            self.data.get_column(self.rvar)
            if self.rvar is not None and self.rvar in self.data.columns
            else None
        )

        # Apply binary conversion on polars DataFrame
        if self.lev is not None and self.rvar is not None:
            self.data = check_binary(self.data, self.rvar, self.lev)

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

        # Handle weights
        if weights is not None and weights != "None":
            self.weights_name = weights
            self.weights = self.data.get_column(weights)
            w = self.weights.drop_nulls()
            # radiant.model: sampling weights (two values, non-integer, the
            # smallest below 1) call for robust standard errors.
            if (
                not w.dtype.is_integer()
                and w.n_unique() == 2
                and w.min() < 1
                and not bool((w == w.round(0)).all())
            ):
                self.robust = True
        else:
            self.weights_name = self.weights = None

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
            self.data, enum_types = convert_categoricals_to_enum(
                self.data, cols_to_convert
            )
            self._enum_types.update(enum_types)

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
            self.evar = extract_evars(_formula_model(self.formula, self.data), self.data.columns)

        # The data in original units, for prediction; numeric explanatory
        # variables are scaled for estimation when asked (not the response).
        self.data_raw = self.data
        self.scale_means: dict[str, float] = {}
        self.scale_sds: dict[str, float] = {}
        if self.standardize or self.center:
            self._calc_scaling()
            self.data = self._scale(self.data)

        pdf = self.data.to_pandas()
        self.fitted = self._glm(self.formula, pdf)
        if formula_given:
            self.rvar = extract_rvar(self.fitted.model, self.data.columns)
            if self.lev is None:
                self.lev = self.data[self.rvar][0]

        if self.stepwise:
            from pyrsm.model.model import step_select

            def stats(fit, base):
                # drop1 / add1 for glm: Df, Deviance
                if base is None:
                    return [None, fit.deviance]
                return [abs(fit.df_resid - base.df_resid), fit.deviance]

            out = step_select(
                full_fit=self.fitted,
                formula=self.formula,
                rvar=self.rvar,
                direction=self.stepwise,
                pdf=pdf,
                fit_formula=self._glm,
                aic=lambda fit: fit.aic,
                stats=stats,
                columns=["Df", "Deviance"],
            )
            self.step_trace = out["trace"]
            self.removed, self.added = out["removed"], out["added"]
            self.formula = out["formula"]
            self.fitted = out["fit"]
            self.ivar = [t for t in self.ivar if t in out["current"]]

        self.fitted.nobs_dropped = self.data.height - self.fitted.nobs

        # Confidence intervals: profile likelihood below 5,000 observations,
        # Wald otherwise (radiant.model); robust standard errors use Wald.
        if ci_type is None:
            ci_type = "profile" if self.fitted.nobs < 5000 else "wald"
        if ci_type not in ("profile", "wald"):
            raise ValueError("ci_type must be 'profile' or 'wald'")
        self.ci_type = "wald" if self.robust else ci_type

        # Coefficient inference: HC1 when robust. sandwich's HC1 is HC0 times
        # n / (n - k); statsmodels' GLM "HC1" uses a different correction.
        #: Columns dropped because of perfect multicollinearity, as R's glm()
        #: drops them (NA coefficients). They are left out of the table.
        self.aliased = list(getattr(self.fitted, "aliased", []))
        if self.robust:
            from pyrsm.multivariate._lm import glm_like_r

            self.fitted_inference = glm_like_r(self.fitted.model, robust=True)
        else:
            self.fitted_inference = self.fitted
        inf = self.fitted_inference
        keep = np.array([n not in self.aliased for n in self.fitted.params.index])

        # Build coefficient table as polars DataFrame
        from pyrsm.model.model import r_label as r_powers_label

        params = self.fitted.params[keep]
        or_vals = np.exp(params.values)
        or_pct = 100 * np.where(or_vals < 1, -(1 - or_vals), or_vals - 1)
        # No odds ratio for the intercept, as radiant.model
        is_intercept = np.array([i == "Intercept" for i in params.index])
        or_vals = np.where(is_intercept, np.nan, or_vals)
        or_pct = np.where(is_intercept, np.nan, or_pct)
        self.coef = pl.DataFrame(
            {
                "index": [
                    r_powers_label(i)
                    for i in params.index.str.replace("[T.", "[", regex=False)
                ],
                "OR": pl.Series(or_vals, dtype=pl.Float64).fill_nan(None),
                "OR%": pl.Series(or_pct, dtype=pl.Float64).fill_nan(None),
                "coefficient": params.values,
                "std.error": np.asarray(inf.bse)[keep],
                "z.value": np.asarray(inf.tvalues)[keep],
                "p.value": np.asarray(inf.pvalues)[keep],
                "  ": sig_stars(np.asarray(inf.pvalues)[keep]),
            }
        ).drop_nulls(subset=["coefficient"])

    def _glm(self, formula: str, pdf: pd.DataFrame):
        """Fit the logistic GLM (classical covariance; see ``fitted_inference``).

        Perfectly collinear columns are dropped as R's ``glm()`` drops them
        (``fitted.aliased``) instead of spreading the effect over them.
        """
        from pyrsm.multivariate._lm import glm_like_r

        weights = pdf[self.weights_name] if self.weights_name else None
        return glm_like_r(
            smf.glm(
                formula=formula,
                data=pdf,
                freq_weights=weights,
                family=Binomial(link=Logit()),
            )
        )

    def _calc_scaling(self) -> None:
        """Means and standard deviations of the numeric explanatory variables.

        R's ``scale_df``: ``sd`` without weights, a weighted population
        standard deviation with them.
        """
        w = None
        if self.weights is not None:
            w = self.weights.cast(pl.Float64).to_numpy()
        for c in self.evar:
            if c not in self.data.columns:
                continue
            dtype = self.data[c].dtype
            if not dtype.is_numeric() or dtype == pl.Boolean:
                continue
            x = self.data[c].cast(pl.Float64).to_numpy()
            ok = ~np.isnan(x)
            if w is not None:
                ok &= ~np.isnan(w)
                ww = w[ok] / w[ok].sum()
                mean = float(np.sum(ww * x[ok]))
                sd = float(np.sqrt(np.sum(ww * (x[ok] - mean) ** 2)))
            else:
                mean = float(np.mean(x[ok]))
                sd = float(np.std(x[ok], ddof=1)) if ok.sum() > 1 else 0.0
            self.scale_means[c] = mean
            self.scale_sds[c] = sd

    def _scale(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply the estimation-data scaling to ``df``."""
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

    def conf_int(self, conf: float = 0.95) -> pd.DataFrame:
        """Coefficient confidence intervals (``ci_type``), as radiant.model."""
        from pyrsm.model.model import profile_ci

        alpha = 1 - conf
        if self.ci_type == "profile":
            return profile_ci(self.fitted, alpha=alpha)
        ci = self.fitted_inference.conf_int(alpha=alpha)
        ci = pd.DataFrame(np.asarray(ci), index=self.fitted.params.index)
        return ci

    def summary(
        self,
        vif=False,
        test=None,
        ci=False,
        odds=False,
        conf: float = 0.95,
        dec=3,
        plain=True,
    ) -> None:
        """
        Summarize the logistic regression model output

        Parameters
        ----------
        vif : bool, default False
            Print the generalized variance inflation factors (as ``car::vif``).
        test : list[str] or None, optional
            Variables to test: likelihood-ratio Chi-squared test, or a robust
            Wald test when ``robust=True``.
        ci : bool, default False
            Print confidence intervals for the coefficients (``ci_type``).
        odds : bool, default False
            Print the odds ratios with confidence intervals.
        conf : float, default 0.95
            Confidence level for ``ci`` and ``odds``.
        dec : int, default 3
            Number of decimal places to round to.
        plain : bool, default False
            Force plain text output (useful for non-notebook environments like Shiny).
        """
        from pyrsm.model.model import gvif, r_label

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
        if not plain and du.is_notebook():
            self._summary_styled(dec)
        else:
            self._summary_plain(dec)
        du.print_sig_codes()

        print(f"\n{model_fit(self.fitted, dec=dec)}")
        if self.aliased:
            print(f"\n{self.collinearity_note()}")

        if vif:
            if self.aliased:
                print("\nMulticollinearity diagnostics were not calculated.")
            elif self.evar is None or len(self.evar) < 2:
                print("\nVariance Inflation Factors cannot be calculated")
            else:
                print("\nVariance inflation factors:")
                print(f"\n{gvif(self.fitted, dec=dec).to_string()}")

        if (ci or odds) and self.aliased:
            # R: confint() / exp(confint()) are skipped for an aliased model
            if ci:
                print("\nConfidence intervals were not calculated.")
            if odds:
                print("\nOdds ratios were not calculated")
        elif ci or odds:
            low, high = [f"{round(100 * p, 4):g}%" for p in ((1 - conf) / 2, 1 - (1 - conf) / 2)]
            cis = self.conf_int(conf=conf)
            coef = self.fitted.params
            labels = [r_label(i).replace("[T.", "[") for i in coef.index]
            if ci:
                tab = pd.DataFrame(
                    {
                        "coefficient": coef.values,
                        low: cis.iloc[:, 0].values,
                        high: cis.iloc[:, 1].values,
                        "+/-": cis.iloc[:, 1].values - coef.values,
                    },
                    index=labels,
                ).round(dec)
                print("\nConfidence intervals:")
                print(f"\n{tab.to_string()}")
            if odds:
                orlab = "std odds ratio" if self.standardize else "odds ratio"
                keep = [i != "Intercept" for i in coef.index]
                tab = pd.DataFrame(
                    {
                        orlab: np.exp(coef.values),
                        low: np.exp(cis.iloc[:, 0].values),
                        high: np.exp(cis.iloc[:, 1].values),
                    },
                    index=labels,
                )[keep].round(dec)
                print("\nOdds ratios:")
                print(f"\n{tab.to_string()}")

        if test is not None and len(test) > 0:
            self.chisq_test(test=test, dec=dec)

    def collinearity_note(self) -> str:
        """R's perfect-multicollinearity message, naming the dropped columns
        (empty when nothing was dropped)."""
        if not self.aliased:
            return ""
        from pyrsm.model.model import r_label

        names = ", ".join(r_label(a.replace("[T.", "[")) for a in self.aliased)
        return (
            "The set of explanatory variables exhibit perfect multicollinearity.\n"
            f"One or more variables were dropped from the estimation: {names}"
        )

    def _summary_header(self) -> None:
        """Print the summary header."""
        print("Logistic regression (GLM)")
        print(f"Data                 : {self.name}")
        print_scope(self, 21)
        print(f"Response variable    : {self.rvar}")
        print(f"Level                : {self.lev} in {self.rvar}")
        print(f"Explanatory variables: {', '.join(self.evar)}")
        if self.weights is not None:
            print(f"Weights used         : {self.weights_name}")
        if self.stepwise:
            if self.stepwise == "backward":
                change = "removed: " + (", ".join(self.removed) or "none")
            else:
                from pyrsm.model.model import r_label

                change = "selected: " + r_label(self.formula.split("~", 1)[1].strip())
            print(f"Stepwise selection   : {self.stepwise} (AIC), {change}")
        print(f"Null hyp.: there is no effect of x on {self.rvar}")
        print(f"Alt. hyp.: there is an effect of x on {self.rvar}")
        if self.standardize:
            print(
                f"**Standardized odds-ratios and coefficients shown ({self.standardize} X SD)**"
            )
        elif self.center:
            print("**Centered odds-ratios and coefficients shown (x - mean(x))**")
        if self.robust:
            print("**Robust standard errors used**")
        print()

    def _summary_plain(self, dec: int = 3) -> None:
        """Print plain text coefficient table."""
        df = self.coef.clone()
        df = df.with_columns(
            [
                pl.col("OR").round(dec).cast(pl.Utf8).fill_null(""),
                pl.col("coefficient").round(dec),
                pl.col("std.error").round(dec),
                pl.col("z.value").round(dec),
                pl.when(pl.col("p.value") < 0.001)
                .then(pl.lit("< .001"))
                .otherwise(pl.col("p.value").round(dec).cast(pl.Utf8))
                .alias("p.value"),
                (pl.col("OR%").round(max(dec - 2, 0)).cast(pl.Utf8) + "%")
                .fill_null("")
                .alias("OR%"),
                pl.col("index"),
            ]
        )
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
                (pl.col("OR%").round(max(dec - 2, 0)).cast(pl.Utf8) + "%").alias("OR%"),
            ]
        )

        gt = du.style_table(
            df,
            title="Coefficient Estimates",
            subtitle=f"Response: {self.rvar} (level: {self.lev})",
            number_cols=["OR", "coefficient", "std.error", "z.value"],
            dec=dec,
        )
        display(gt)

    def predict(
        self, data=None, cmd=None, data_cmd=None, ci=False, conf=0.95, dec=None
    ) -> pl.DataFrame:
        """
        Generate probability predictions using the fitted model.

        Parameters
        ----------
        dec : int, optional
            Number of decimal places to round float columns in the output.
            If None (default), no rounding is applied.

        Returns
        -------
        pl.DataFrame
            DataFrame containing the predictions and the data used to make those predictions.
        """
        from pyrsm.model.model import check_pred_vars

        check_pred_vars(
            self.evar,
            data=None if data is None else check_dataframe(data),
            cmd=cmd if isinstance(cmd, dict) else None,
            data_cmd=data_cmd if isinstance(data_cmd, dict) else None,
        )
        if data is None:
            pred_data = self.data_raw.select(self.evar)
        else:
            data = check_dataframe(data)
            pred_data = data.select(self.evar)
            # Apply stored Enum types to ensure consistent categorical levels
            if self._enum_types:
                pred_data = apply_enum_types(pred_data, self._enum_types)

        if data_cmd is not None:
            # Apply data_cmd values to pred_data
            pred_data = pred_data.with_columns(
                [pl.lit(v).alias(k) for k, v in data_cmd.items()]
            )
            # Re-apply Enum types after modifying columns
            if self._enum_types:
                pred_data = apply_enum_types(pred_data, self._enum_types)
        elif cmd is not None:
            cmd = {k: ifelse(isinstance(v, str), [v], v) for k, v in cmd.items()}
            pred_data = sim_prediction(data=pred_data, vary=cmd)
            # Re-apply Enum types after sim_prediction creates new data
            if self._enum_types:
                pred_data = apply_enum_types(pred_data, self._enum_types)

        # A standardized/centered model is evaluated on scaled inputs; the
        # output shows the inputs in original units.
        model_data = self._scale(pred_data) if self.scale_means else pred_data

        if ci:
            if data_cmd is not None:
                raise ValueError(
                    "Confidence intervals not available when using the Data & Command option"
                )
            else:
                # predict_ci returns pandas, convert result to polars
                ci_result = predict_ci(self.fitted, model_data.to_pandas(), conf=conf)
                pred = pl.concat(
                    [pred_data, pl.from_pandas(ci_result)], how="horizontal"
                )

        else:
            # Get predictions from statsmodels (needs pandas), add to polars DataFrame
            predictions = self.fitted.predict(model_data.to_pandas())
            pred = pred_data.with_columns(
                pl.lit(predictions.values).alias("prediction")
            )

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
        plots: Literal[
            "pred", "pdp", "pip", "or", "dashboard", "dist", "corr", "scatter", "fit", "influence"
        ] = "pred",
        data=None,
        incl=None,
        excl=None,
        incl_int=[],
        nobs: int = 1000,
        fix=True,
        hline=True,
        ice=False,
        ice_nobs=100,
        nnv=30,
        minq=0.025,
        maxq=0.975,
        ret=None,
        alpha=0.05,
        intercept=False,
        figsize=None,
    ) -> None:
        """
        Plots for a logistic regression model

        ``"dashboard"`` shows the diagnostics applied work uses for a binary
        classifier: calibration, ROC curve (AUC), cumulative gains and the
        distribution of predicted probabilities by outcome.

        As radiant.model: ``"fit"`` compares the actual and predicted
        probability in 30 bins of the predicted probability, ``"scatter"``
        shows each explanatory variable by response level (bar charts for
        categorical variables), ``"influence"`` the standardized deviance
        residuals with Cook's distance, and ``"or"`` (``"coef"``) the odds
        ratios with confidence intervals (``alpha``). ``nobs`` samples the
        rows for ``"corr"`` and ``"scatter"``.
        """
        plots = convert_to_list(plots)
        excl = convert_to_list(excl)
        incl = ifelse(incl is None, None, convert_to_list(incl))
        incl_int = convert_to_list(incl_int)

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
            data = apply_enum_types(data, self._enum_types)

        if "dashboard" in plots:
            from pyrsm.model.model import logit_dashboard

            return logit_dashboard(self.fitted, lev=self.lev, figsize=figsize or (10, 9))

        if "influence" in plots:
            from pyrsm.model.model import influence_plot

            return influence_plot(self.fitted)
        if "fit" in plots:
            return _fit_plot(self.fitted)
        if "scatter" in plots:
            raw = self.data_raw
            if self.rv is not None and len(self.rv) == raw.height:
                raw = raw.with_columns(self.rv.alias(self.rvar))
            return _scatter_plot(raw, self.rvar, self.evar, nobs=nobs)

        if "dist" in plots:
            (
                distr_plot,
                _,
                _,
                _,
            ) = _get_visualize_plots()
            return distr_plot(data).plot()
        elif "corr" in plots:
            correlation = _get_correlation()
            cr = correlation(data)
            return cr.plot(nobs=nobs, figsize=figsize)
        elif "or" in plots or "coef" in plots:
            return or_plot(
                self.fitted_inference,
                alpha=alpha,
                intercept=intercept,
                incl=incl,
                excl=excl,
                figsize=figsize,
            )
        elif "pred" in plots:
            _, pred_plot_sm, _, _ = _get_visualize_plots()
            # return pdp_sm(
            return pred_plot_sm(
                self.fitted,
                data=data,  # Pass polars DataFrame to preserve Enum types
                incl=incl,
                excl=excl,
                incl_int=incl_int,
                fix=fix,
                hline=hline,
                nnv=nnv,
                minq=minq,
                maxq=maxq,
            )
        elif "pdp" in plots:
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
        elif "pip" in plots or "vimp" in plots:
            _, _, _, pip_plot_sm = _get_visualize_plots()
            return_pip, p = pip_plot_sm(
                self.fitted,
                data=data,  # Pass polars DataFrame
                rep=10,
                ret=ret,
            )
            if ret:
                return (p, return_pip)
            return p

    def chisq_test(self, test=None, dec=3) -> None:
        """
        Chisq-test for competing models
        """
        if self.stepwise:
            print("\nModel comparisons are not conducted when Stepwise has been selected.")
            return
        if test is None:
            test = self.evar
        else:
            test = ifelse(isinstance(test, str), [test], test)
        from pyrsm.model.model import r_powers

        test = r_powers(list(test))

        evar = [c for c in self.evar if c not in test]
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

        from pyrsm.model.model import r_label

        print(f"\nModel 1: {r_label(formula)}")
        print(f"Model 2: {r_label(self.formula)}")

        pdf = self.data.to_pandas()
        sub_fitted = self._glm(formula, pdf.loc[self.fitted.model.data.row_labels])

        stats = _get_scipy_stats()
        if self.robust:
            # Wald test with the robust covariance, as car::linearHypothesis
            # with a vcov: every coefficient the restricted model leaves out
            kept = set(sub_fitted.params.index)
            names = list(self.fitted.params.index)
            # aliased columns have no estimate to test
            idx = [
                names.index(c) for c in names
                if c not in kept and c not in self.aliased
            ]
            L = np.zeros((len(idx), len(names)))
            for row, col in enumerate(idx):
                L[row, col] = 1.0
            wald = self.fitted_inference.wald_test(L, use_f=False, scalar=True)
            lrtest = float(wald.statistic)
            df = len(idx)
            pvalue = float(wald.pvalue)
        else:
            lrtest = -2 * (sub_fitted.llf - self.fitted.llf)
            df = self.fitted.df_model - sub_fitted.df_model
            pvalue = stats.chi2.sf(lrtest, df)

        pr2_full = 1 - self.fitted.llf / self.fitted.llnull
        pr2_sub = 1 - sub_fitted.llf / sub_fitted.llnull

        print(f"Pseudo R-squared, Model 1 vs 2: {pr2_sub:.3f} vs {pr2_full:.3f}")
        pvalue = ifelse(pvalue < 0.001, "< .001", round(pvalue, dec))
        print(f"Chi-squared: {round(lrtest, dec)} df ({df:.0f}), p.value {pvalue}")


def _fit_plot(fitted):
    """Actual vs predicted probability in 30 bins (radiant.model "Model fit")."""
    from plotnine import aes, geom_col, geom_line, ggplot, labs, scale_y_continuous

    from pyrsm.plot_style import FILL, REF, theme_pyrsm
    from pyrsm.bins import xtile

    actual = np.asarray(fitted.model.endog, dtype=float)
    pred = np.asarray(fitted.fittedvalues, dtype=float)
    if len(pred) < 30:
        raise ValueError("Insufficient observations to generate Model fit plot")
    bins = np.asarray(xtile(pred, 30))
    df = pd.DataFrame({"bin": bins, "actual": actual, "pred": pred})
    lo, hi = df["bin"].min(), df["bin"].max()
    if df.loc[df["bin"] == lo, "actual"].mean() < df.loc[df["bin"] == hi, "actual"].mean():
        df["bin"] = 1 + hi - df["bin"]
    agg = df.groupby("bin", as_index=False).agg(actual=("actual", "mean"), pred=("pred", "mean"))
    return (
        ggplot(agg, aes(x="bin"))
        + geom_col(aes(y="actual"), fill=FILL, alpha=0.6)
        + geom_line(aes(y="pred"), color=REF, size=1)
        + scale_y_continuous(limits=(0, 1))
        + labs(
            title="Actual vs Fitted values (binned)",
            x="Predicted probability bins",
            y="Probability",
        )
        + theme_pyrsm()
    )


def _scatter_plot(data: pl.DataFrame, rvar: str, evar: list[str], nobs: int = 1000):
    """Each explanatory variable by response level (radiant.model "Scatter")."""
    from plotnine import aes, geom_bar, geom_jitter, ggplot, labs, position_fill

    from pyrsm.plot_style import ALPHA, FILL, theme_pyrsm
    from pyrsm.plot_utils import compose_plots

    df = data.select([rvar, *evar])
    if nobs and nobs > 0 and nobs < df.height:
        df = df.sample(nobs, seed=1234)
    df = df.with_columns(pl.col(rvar).cast(pl.Utf8)).to_pandas()
    plots = []
    for v in evar:
        if pd.api.types.is_numeric_dtype(df[v]):
            p = (
                ggplot(df, aes(x=rvar, y=v))
                + geom_jitter(alpha=ALPHA, color=FILL, width=0.25, height=0)
                + labs(x=rvar, y=v)
                + theme_pyrsm()
            )
        else:
            p = (
                ggplot(df, aes(x=v, fill=rvar))
                + geom_bar(position=position_fill(), alpha=0.5)
                + labs(x=v, y="")
                + theme_pyrsm()
            )
        plots.append(p)
    return compose_plots(plots, ncol=1, width_per_col=7, height_per_row=3)
