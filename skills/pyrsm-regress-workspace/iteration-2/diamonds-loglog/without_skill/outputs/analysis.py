"""
Regression analysis: price ~ carat + clarity (linear vs log-log)
Diamonds data from /Users/vnijs/gh/pyrsm/examples/data/data/diamonds.parquet
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy import stats

DATA_PATH = "/Users/vnijs/gh/pyrsm/examples/data/data/diamonds.parquet"
OUT_DIR = "/Users/vnijs/gh/pyrsm/skills/pyrsm-regress-workspace/iteration-2/diamonds-loglog/without_skill/outputs"


def diagnostic_dashboard(model, title, save_path):
    """Build a 2x2 residual diagnostic dashboard like R's plot.lm()."""
    fitted = model.fittedvalues
    resid = model.resid
    influence = model.get_influence()
    std_resid = influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(fitted, resid, alpha=0.3, s=10)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    # LOWESS-ish trend via binned means
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sm_line = lowess(resid, fitted, frac=0.3, return_sorted=True)
        ax.plot(sm_line[:, 0], sm_line[:, 1], color="red", linewidth=1.5)
    except Exception:
        pass
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")

    # 2. Normal Q-Q
    ax = axes[0, 1]
    sm.qqplot(std_resid, line="45", ax=ax, markersize=3, alpha=0.4)
    ax.set_title("Normal Q-Q (std. residuals)")

    # 3. Scale-Location
    ax = axes[1, 0]
    sqrt_abs_std = np.sqrt(np.abs(std_resid))
    ax.scatter(fitted, sqrt_abs_std, alpha=0.3, s=10)
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sm_line = lowess(sqrt_abs_std, fitted, frac=0.3, return_sorted=True)
        ax.plot(sm_line[:, 0], sm_line[:, 1], color="red", linewidth=1.5)
    except Exception:
        pass
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("sqrt(|std. residuals|)")
    ax.set_title("Scale-Location")

    # 4. Histogram of residuals with normal overlay
    ax = axes[1, 1]
    ax.hist(resid, bins=50, density=True, alpha=0.6, edgecolor="black")
    xs = np.linspace(resid.min(), resid.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, resid.mean(), resid.std()), "r-", linewidth=1.5)
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")
    ax.set_title("Residual histogram (normal overlay)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    df = pd.read_parquet(DATA_PATH)
    print("=" * 70)
    print("DATA OVERVIEW")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nHead:")
    print(df[["price", "carat", "clarity"]].head())
    print("\nClarity levels (worst -> best): I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF")
    print("Clarity counts:")
    print(df["clarity"].value_counts().sort_index())

    # Clarity is already an ordered category; statsmodels formula treats it as categorical.
    # Set "I1" (worst) as reference for interpretability.
    df["clarity"] = df["clarity"].astype(str)
    clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
    df["clarity"] = pd.Categorical(df["clarity"], categories=clarity_order, ordered=True)

    print("\nPrice summary:")
    print(df["price"].describe())
    print("\nCarat summary:")
    print(df["carat"].describe())

    # ---- Linear specification: price ~ carat + clarity ----
    print("\n" + "=" * 70)
    print("MODEL 1: LINEAR — price ~ carat + clarity")
    print("=" * 70)
    m_lin = smf.ols("price ~ carat + C(clarity)", data=df).fit()
    print(m_lin.summary())

    # Breusch-Pagan test for heteroskedasticity
    bp_lin = sm.stats.diagnostic.het_breuschpagan(m_lin.resid, m_lin.model.exog)
    print(f"\nBreusch-Pagan (linear): LM stat = {bp_lin[0]:.2f}, p-value = {bp_lin[1]:.3e}")
    # Jarque-Bera for normality (already in summary, but print clearly)
    jb_lin = sm.stats.stattools.jarque_bera(m_lin.resid)
    print(f"Jarque-Bera (linear): stat = {jb_lin[0]:.2f}, p-value = {jb_lin[1]:.3e}")

    diagnostic_dashboard(
        m_lin,
        "Linear model: price ~ carat + clarity",
        f"{OUT_DIR}/dashboard_linear.png",
    )

    # ---- Log-log specification: log(price) ~ log(carat) + clarity ----
    print("\n" + "=" * 70)
    print("MODEL 2: LOG-LOG — log(price) ~ log(carat) + clarity")
    print("=" * 70)
    m_log = smf.ols("np.log(price) ~ np.log(carat) + C(clarity)", data=df).fit()
    print(m_log.summary())

    bp_log = sm.stats.diagnostic.het_breuschpagan(m_log.resid, m_log.model.exog)
    print(f"\nBreusch-Pagan (log-log): LM stat = {bp_log[0]:.2f}, p-value = {bp_log[1]:.3e}")
    jb_log = sm.stats.stattools.jarque_bera(m_log.resid)
    print(f"Jarque-Bera (log-log): stat = {jb_log[0]:.2f}, p-value = {jb_log[1]:.3e}")

    diagnostic_dashboard(
        m_log,
        "Log-log model: log(price) ~ log(carat) + clarity",
        f"{OUT_DIR}/dashboard_loglog.png",
    )

    # ---- Side-by-side comparison ----
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<35}{'Linear':>15}{'Log-log':>15}")
    print("-" * 65)
    print(f"{'R-squared':<35}{m_lin.rsquared:>15.4f}{m_log.rsquared:>15.4f}")
    print(f"{'Adj. R-squared':<35}{m_lin.rsquared_adj:>15.4f}{m_log.rsquared_adj:>15.4f}")
    print(f"{'Residual std error':<35}{np.sqrt(m_lin.mse_resid):>15.4f}{np.sqrt(m_log.mse_resid):>15.4f}")
    print(f"{'Breusch-Pagan p-value':<35}{bp_lin[1]:>15.3e}{bp_log[1]:>15.3e}")
    print(f"{'Jarque-Bera p-value':<35}{jb_lin[1]:>15.3e}{jb_log[1]:>15.3e}")
    print(f"{'Resid skewness':<35}{stats.skew(m_lin.resid):>15.4f}{stats.skew(m_log.resid):>15.4f}")
    print(f"{'Resid kurtosis (excess)':<35}{stats.kurtosis(m_lin.resid):>15.4f}{stats.kurtosis(m_log.resid):>15.4f}")

    # Carat coefficient interpretation
    print("\nKey coefficients:")
    print(f"  Linear: carat = {m_lin.params['carat']:.2f}  "
          f"(a 1-carat increase predicts ${m_lin.params['carat']:.0f} higher price)")
    print(f"  Log-log: log(carat) = {m_log.params['np.log(carat)']:.4f}  "
          f"(a 1% increase in carat predicts ~{m_log.params['np.log(carat)']:.2f}% higher price; "
          f"this is the price elasticity of carat)")


if __name__ == "__main__":
    main()
