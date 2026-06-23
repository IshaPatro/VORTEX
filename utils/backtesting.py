"""
backtesting.py — VaR/ES validation and regime predictive-content analysis.

Implements the standard regulatory backtests a risk team would run:
  • Kupiec POF      — unconditional coverage (is the breach rate correct?)
  • Christoffersen  — independence + conditional coverage (are breaches clustered?)
  • ES backtest     — is realized tail loss consistent with predicted CVaR?
  • Regime study    — do detected regimes carry forward-return information?
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import chi2


def var_exceedances(returns: pd.Series, var_series: pd.Series) -> pd.Series:
    """Boolean series: True where the realized return breached (fell below) VaR."""
    df = pd.concat([returns.rename("ret"), var_series.rename("var")], axis=1).dropna()
    return (df["ret"] < df["var"]).rename("exceedance")


def kupiec_pof(returns: pd.Series, var_series: pd.Series, confidence: float = 0.95) -> Dict[str, float]:
    """Kupiec Proportion-of-Failures test for unconditional coverage."""
    exc = var_exceedances(returns, var_series)
    n = int(len(exc))
    x = int(exc.sum())
    p = 1 - confidence
    if n == 0:
        return {"n": 0, "breaches": 0, "expected": 0, "rate": 0.0, "lr_pof": float("nan"), "p_value": float("nan")}
    pi = x / n
    # Likelihood ratio (guard against log(0)).
    if x == 0:
        lr = -2 * (n * np.log(1 - p))
    else:
        lr = -2 * (
            (n - x) * np.log(1 - p) + x * np.log(p)
            - ((n - x) * np.log(1 - pi) + x * np.log(pi))
        )
    return {
        "n": n,
        "breaches": x,
        "expected": round(n * p, 1),
        "rate": round(pi * 100, 2),
        "expected_rate": round(p * 100, 2),
        "lr_pof": round(float(lr), 3),
        "p_value": round(float(1 - chi2.cdf(lr, df=1)), 4),
    }


def christoffersen(returns: pd.Series, var_series: pd.Series, confidence: float = 0.95) -> Dict[str, float]:
    """Christoffersen independence + conditional-coverage tests (breach clustering)."""
    exc = var_exceedances(returns, var_series).astype(int).values
    if len(exc) < 2:
        return {"lr_ind": float("nan"), "lr_cc": float("nan"), "p_value_cc": float("nan")}

    # Transition counts n_ij (i = state at t-1, j = state at t).
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(exc[:-1], exc[1:]):
        if prev == 0 and cur == 0: n00 += 1
        elif prev == 0 and cur == 1: n01 += 1
        elif prev == 1 and cur == 0: n10 += 1
        else: n11 += 1

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _ll(p, k0, k1):
        if p <= 0 or p >= 1:
            return 0.0
        return k0 * np.log(1 - p) + k1 * np.log(p)

    lr_ind = -2 * (
        _ll(pi, n00 + n10, n01 + n11)
        - (_ll(pi01, n00, n01) + _ll(pi11, n10, n11))
    )
    kp = kupiec_pof(returns, var_series, confidence)
    lr_cc = kp["lr_pof"] + lr_ind  # conditional coverage = POF + independence
    return {
        "lr_ind": round(float(lr_ind), 3),
        "p_value_ind": round(float(1 - chi2.cdf(lr_ind, df=1)), 4),
        "lr_cc": round(float(lr_cc), 3),
        "p_value_cc": round(float(1 - chi2.cdf(lr_cc, df=2)), 4),
    }


def es_backtest(returns: pd.Series, var_series: pd.Series, es_series: pd.Series) -> Dict[str, float]:
    """Simple Expected-Shortfall backtest: compare the average realized loss on
    breach days to the predicted ES (Acerbi-Szekely style ratio)."""
    df = pd.concat(
        [returns.rename("ret"), var_series.rename("var"), es_series.rename("es")], axis=1
    ).dropna()
    breaches = df[df["ret"] < df["var"]]
    if breaches.empty:
        return {"n_breaches": 0, "avg_loss": float("nan"), "avg_es": float("nan"), "ratio": float("nan")}
    avg_loss = float(breaches["ret"].mean())
    avg_es = float(breaches["es"].mean())
    return {
        "n_breaches": int(len(breaches)),
        "avg_loss": round(avg_loss * 100, 3),
        "avg_es": round(avg_es * 100, 3),
        "ratio": round(avg_loss / avg_es, 3) if avg_es else float("nan"),
    }


def rolling_es(returns: pd.Series, confidence: float = 0.95, window: int = 252) -> pd.Series:
    """Rolling Expected Shortfall (mean of returns at/below the rolling quantile)."""
    q = 1 - confidence

    def _es(x):
        thr = np.quantile(x, q)
        tail = x[x <= thr]
        return tail.mean() if len(tail) else np.nan

    return returns.rolling(window).apply(_es, raw=True).rename("es")


def pinball_loss(returns: pd.Series, var_series: pd.Series, confidence: float = 0.95) -> float:
    """Quantile (pinball) loss for a VaR series at quantile level τ = 1-α.
    Lower is better; this is the proper scoring rule for quantile forecasts."""
    tau = 1 - confidence
    df = pd.concat([returns.rename("r"), var_series.rename("v")], axis=1).dropna()
    u = df["r"] - df["v"]
    loss = np.where(u >= 0, tau * u, (tau - 1) * u)
    return float(np.mean(loss))


def var_model_comparison(returns: pd.Series, var_df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """League table ranking every VaR method by coverage tests and pinball loss.

    A model 'passes' coverage if Kupiec p > 0.05. The winner is the
    coverage-passing model with the lowest pinball loss (else lowest pinball).
    """
    labels = {
        "historical_var": "Historical", "parametric_var": "Parametric",
        "ewma_var": "EWMA", "cornish_fisher_var": "Cornish-Fisher",
        "regime_adaptive_var": "Regime-Adaptive",
    }
    rows = []
    for col in var_df.columns:
        kp = kupiec_pof(returns, var_df[col], confidence)
        cc = christoffersen(returns, var_df[col], confidence)
        rows.append({
            "Method": labels.get(col, col),
            "Breaches": kp["breaches"],
            "Rate %": kp["rate"],
            "Kupiec p": kp["p_value"],
            "Christoffersen p": cc["p_value_cc"],
            "Pinball (bps)": round(pinball_loss(returns, var_df[col], confidence) * 1e4, 2),
        })
    df = pd.DataFrame(rows)
    # Winner: passing coverage, then lowest pinball.
    passing = df[df["Kupiec p"] > 0.05]
    pool = passing if not passing.empty else df
    winner = pool.sort_values("Pinball (bps)").iloc[0]["Method"]
    df = df.sort_values("Pinball (bps)").reset_index(drop=True)
    df["Rank"] = df.index + 1
    df["Verdict"] = df["Method"].apply(lambda m: "WINNER" if m == winner else "")
    return df.set_index("Method")


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series) -> Dict[str, float]:
    """Diebold-Mariano test for equal predictive accuracy (two loss series).

    H0: the two forecasts have equal expected loss. A significant negative stat
    means forecast A is better (lower loss). Uses a normal approximation.
    """
    d = (loss_a - loss_b).dropna()
    n = len(d)
    if n < 30 or d.std() == 0:
        return {"dm_stat": float("nan"), "p_value": float("nan"), "n": n}
    dm = float(d.mean() / (d.std(ddof=1) / np.sqrt(n)))
    from scipy.stats import norm
    p = float(2 * (1 - norm.cdf(abs(dm))))
    return {"dm_stat": round(dm, 3), "p_value": round(p, 4), "n": n}


def regime_forward_returns(
    regimes: pd.DataFrame, returns: pd.Series, horizon: int = 21
) -> pd.DataFrame:
    """Do regimes carry predictive content? Average *forward* h-day return and
    hit-rate conditional on the regime observed today."""
    df = pd.DataFrame({"regime": regimes["regime"], "ret": returns}).dropna()
    fwd = df["ret"].rolling(horizon).sum().shift(-horizon)
    df = df.assign(fwd=fwd).dropna()
    rows = []
    for name, grp in df.groupby("regime"):
        rows.append({
            "Regime": name,
            f"Avg Fwd {horizon}d (%)": round(grp["fwd"].mean() * 100, 2),
            "Hit Rate (%)": round((grp["fwd"] > 0).mean() * 100, 1),
            "Volatility (%)": round(grp["fwd"].std() * 100, 2),
            "N": int(len(grp)),
        })
    return pd.DataFrame(rows).set_index("Regime")
