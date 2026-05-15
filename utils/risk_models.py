"""
risk_models.py — Adaptive VaR engine + volatility forecasting.

VaR Methods: Historical, Parametric, EWMA, Regime-Adaptive
Volatility:  Rolling, EWMA (RiskMetrics), XGBoost (optional)
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm


# ── Volatility ──────────────────────────────────────────────────────────────

def rolling_volatility(returns: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    vol = returns.rolling(window).std()
    return (vol * np.sqrt(252) if annualize else vol).rename("rolling_vol")


def ewma_volatility(returns: pd.Series, lam: float = 0.94, annualize: bool = True) -> pd.Series:
    """RiskMetrics EWMA: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}"""
    r = returns.values
    var = np.zeros(len(r))
    var[0] = r[0] ** 2
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    vol = np.sqrt(var)
    return pd.Series(vol * (np.sqrt(252) if annualize else 1), index=returns.index, name="ewma_vol")


def xgboost_vol_forecast(returns: pd.Series, horizon: int = 5) -> Optional[pd.Series]:
    """XGBoost realized-vol forecast. Returns None if xgboost unavailable or data sparse."""
    try:
        import xgboost as xgb
    except ImportError:
        return None
    r = returns.dropna()
    df = pd.DataFrame({"ret": r})
    for lag in [1, 2, 3, 5, 10, 21]:
        df[f"r{lag}"] = r.shift(lag)
        df[f"v{lag}"] = r.rolling(lag).std().shift(1)
    df["target"] = r.rolling(horizon).std().shift(-horizon) * np.sqrt(252)
    df.dropna(inplace=True)
    if len(df) < 200:
        return None
    split = int(len(df) * 0.8)
    feats = [c for c in df.columns if c not in ("ret", "target")]
    m = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                          subsample=0.8, random_state=42, verbosity=0)
    m.fit(df.iloc[:split][feats], df.iloc[:split]["target"])
    return pd.Series(m.predict(df[feats]), index=df.index, name="xgb_vol")


# ── VaR ─────────────────────────────────────────────────────────────────────

def historical_var(returns: pd.Series, confidence: float = 0.95, window: int = 252) -> pd.Series:
    return returns.rolling(window).quantile(1 - confidence).rename("historical_var")


def parametric_var(returns: pd.Series, confidence: float = 0.95, window: int = 252) -> pd.Series:
    z = norm.ppf(1 - confidence)
    mu = returns.rolling(window).mean()
    sig = returns.rolling(window).std()
    return (mu + z * sig).rename("parametric_var")


def ewma_var(returns: pd.Series, confidence: float = 0.95, lam: float = 0.94) -> pd.Series:
    z = norm.ppf(1 - confidence)
    daily_vol = ewma_volatility(returns, lam=lam, annualize=False)
    return (z * daily_vol).rename("ewma_var")


def regime_adaptive_var(
    returns: pd.Series,
    regime_series: pd.Series,
    confidence: float = 0.95,
) -> pd.Series:
    """Per-regime empirical quantile stitched into a continuous VaR series."""
    q = 1 - confidence
    df = pd.DataFrame({"ret": returns, "regime": regime_series}).dropna()
    regime_q = df.groupby("regime")["ret"].quantile(q)
    mapped = df["regime"].map(regime_q)
    return mapped.rename("regime_adaptive_var")


def compute_all_var(
    returns: pd.Series,
    regime_series: Optional[pd.Series] = None,
    confidence: float = 0.95,
    window: int = 252,
) -> pd.DataFrame:
    parts = [
        historical_var(returns, confidence, window),
        parametric_var(returns, confidence, window),
        ewma_var(returns, confidence),
    ]
    if regime_series is not None:
        parts.append(regime_adaptive_var(returns, regime_series, confidence))
    return pd.concat(parts, axis=1).dropna(how="all")


# ── Portfolio helpers ────────────────────────────────────────────────────────

def portfolio_returns(returns_matrix: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    tickers = [t for t in weights if t in returns_matrix.columns]
    w = np.array([weights[t] for t in tickers], dtype=float)
    w /= w.sum()
    return (returns_matrix[tickers] @ w).rename("portfolio_return")


def compute_drawdown(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns).cumprod()
    return ((wealth / wealth.cummax()) - 1).rename("drawdown")


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / 252
    return float((excess.mean() / excess.std()) * np.sqrt(252)) if excess.std() != 0 else 0.0


def portfolio_metrics(returns: pd.Series) -> Dict[str, float]:
    dd = compute_drawdown(returns)
    var95 = float(returns.quantile(0.05))
    cvar95 = float(returns[returns <= var95].mean())
    return {
        "Ann. Return (%)": round(returns.mean() * 252 * 100, 2),
        "Ann. Volatility (%)": round(returns.std() * np.sqrt(252) * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio(returns), 3),
        "Max Drawdown (%)": round(dd.min() * 100, 2),
        "Skewness": round(float(returns.skew()), 3),
        "Excess Kurtosis": round(float(returns.kurtosis()), 3),
        "VaR 95% (daily %)": round(var95 * 100, 3),
        "CVaR 95% (daily %)": round(cvar95 * 100, 3),
    }
