"""Descriptive statistics and econometric diagnostics used in the original notebook."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

def descriptive_statistics(returns, crisis_id, crisis_names):
    def row(x):
        x = np.asarray(x, dtype=float)
        return {"N": len(x), "Mean": x.mean(), "Std": x.std(ddof=1),
                "Min": x.min(), "Max": x.max(), "Skew": stats.skew(x),
                "ExcessKurt": stats.kurtosis(x, fisher=True)}
    rows = {"Overall": row(returns)}
    for name in crisis_names:
        rows[name] = row(returns[np.asarray(crisis_id) == name])
    return pd.DataFrame(rows).T

def stationarity_tests(x, label="series"):
    adf_stat, adf_p, *_ = adfuller(x, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto")
    return {"label": label, "adf_stat": adf_stat, "adf_p": adf_p,
            "kpss_stat": kpss_stat, "kpss_p": kpss_p}

def arch_diagnostics(x, label="series", lags=10):
    lb = acorr_ljungbox(np.asarray(x)**2, lags=[lags], return_df=True)
    lm_stat, lm_p, *_ = het_arch(x, nlags=lags)
    return {"label": label, "ljung_box_stat": float(lb["lb_stat"].iloc[0]),
            "ljung_box_p": float(lb["lb_pvalue"].iloc[0]),
            "arch_lm_stat": float(lm_stat), "arch_lm_p": float(lm_p)}
