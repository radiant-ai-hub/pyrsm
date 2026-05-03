"""
Regress price on carat and clarity for the diamonds dataset.
Compare a linear specification with a log-log specification and inspect residuals.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

DATA_PATH = "/Users/vnijs/gh/pyrsm/examples/data/data/diamonds.parquet"
OUT_DIR = (
    "/Users/vnijs/gh/pyrsm/skills/pyrsm-regress-workspace/"
    "iteration-1/diamonds-loglog/without_skill/outputs"
)


def diagnostic_plot(model, df, y, x_numeric, title, savepath):
    """4-panel residual diagnostic dashboard."""
    fitted = model.fittedvalues
    resid = model.resid
    influence = model.get_influence()
    std_resid = influence.resid_studentized_internal

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1) Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(fitted, resid, alpha=0.2, s=8, edgecolor="none")
    ax.axhline(0, color="red", lw=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")

    # 2) Normal Q-Q
    ax = axes[0, 1]
    sm.qqplot(resid, line="45", fit=True, ax=ax, markersize=3, alpha=0.3)
    ax.set_title("Normal Q-Q")

    # 3) Scale-Location
    ax = axes[1, 0]
    ax.scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.2, s=8, edgecolor="none")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("sqrt(|Std. residuals|)")
    ax.set_title("Scale-Location")

    # 4) Residuals vs main numeric predictor
    ax = axes[1, 1]
    ax.scatter(df[x_numeric], resid, alpha=0.2, s=8, edgecolor="none")
    ax.axhline(0, color="red", lw=1)
    ax.set_xlabel(x_numeric)
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals vs {x_numeric}")

    plt.tight_layout()
    fig.savefig(savepath, dpi=110)
    plt.close(fig)


def main():
    df = pd.read_parquet(DATA_PATH)
    print("Data shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head())
    print()
    print("Clarity levels (raw):", df["clarity"].unique())
    print()

    # Order clarity from worst to best (standard GIA scale)
    clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
    df["clarity"] = pd.Categorical(
        df["clarity"], categories=clarity_order, ordered=True
    )

    # ---------- Linear: price ~ carat + clarity ----------
    print("=" * 70)
    print("Linear specification: price ~ carat + clarity")
    print("=" * 70)
    m_lin = smf.ols("price ~ carat + C(clarity)", data=df).fit()
    print(m_lin.summary())
    print()

    # Breusch-Pagan test for heteroskedasticity
    bp_lin = sm.stats.diagnostic.het_breuschpagan(m_lin.resid, m_lin.model.exog)
    print(
        f"Breusch-Pagan (linear):  LM stat={bp_lin[0]:.2f}, p-value={bp_lin[1]:.3g}"
    )
    # Skewness/kurtosis of residuals
    print(
        f"Residual skew={stats.skew(m_lin.resid):.3f}, "
        f"kurtosis={stats.kurtosis(m_lin.resid):.3f}"
    )
    print()

    diagnostic_plot(
        m_lin,
        df,
        y="price",
        x_numeric="carat",
        title="Linear: price ~ carat + clarity",
        savepath=f"{OUT_DIR}/dashboard_linear.png",
    )

    # ---------- Log-Log: log(price) ~ log(carat) + clarity ----------
    print("=" * 70)
    print("Log-Log specification: log(price) ~ log(carat) + clarity")
    print("=" * 70)
    df["log_price"] = np.log(df["price"])
    df["log_carat"] = np.log(df["carat"])

    m_log = smf.ols("log_price ~ log_carat + C(clarity)", data=df).fit()
    print(m_log.summary())
    print()

    bp_log = sm.stats.diagnostic.het_breuschpagan(m_log.resid, m_log.model.exog)
    print(
        f"Breusch-Pagan (log-log): LM stat={bp_log[0]:.2f}, p-value={bp_log[1]:.3g}"
    )
    print(
        f"Residual skew={stats.skew(m_log.resid):.3f}, "
        f"kurtosis={stats.kurtosis(m_log.resid):.3f}"
    )
    print()

    diagnostic_plot(
        m_log,
        df,
        y="log_price",
        x_numeric="log_carat",
        title="Log-Log: log(price) ~ log(carat) + clarity",
        savepath=f"{OUT_DIR}/dashboard_loglog.png",
    )

    # ---------- Side-by-side summary ----------
    print("=" * 70)
    print("Comparison")
    print("=" * 70)
    print(f"{'Metric':<25}{'Linear':>15}{'Log-Log':>15}")
    print(f"{'R-squared':<25}{m_lin.rsquared:>15.4f}{m_log.rsquared:>15.4f}")
    print(
        f"{'Adj R-squared':<25}{m_lin.rsquared_adj:>15.4f}"
        f"{m_log.rsquared_adj:>15.4f}"
    )
    print(f"{'AIC':<25}{m_lin.aic:>15.1f}{m_log.aic:>15.1f}")
    print(f"{'Resid skew':<25}{stats.skew(m_lin.resid):>15.3f}"
          f"{stats.skew(m_log.resid):>15.3f}")
    print(f"{'Resid kurtosis':<25}{stats.kurtosis(m_lin.resid):>15.3f}"
          f"{stats.kurtosis(m_log.resid):>15.3f}")
    print(f"{'BP p-value (heterosk.)':<25}{bp_lin[1]:>15.3g}{bp_log[1]:>15.3g}")
    print()
    print(
        "Interpretation: in log-log, the log_carat coefficient "
        f"({m_log.params['log_carat']:.3f}) is the elasticity of price "
        "with respect to carat (a 1% increase in carat is associated with "
        f"about {m_log.params['log_carat']:.2f}% change in price)."
    )


if __name__ == "__main__":
    main()
