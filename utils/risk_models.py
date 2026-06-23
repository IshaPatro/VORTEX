"""
risk_models.py — Adaptive VaR engine, volatility forecasting & risk attribution.

VaR Methods : Historical, Parametric (Gaussian), EWMA, Cornish-Fisher,
              Regime-Adaptive (causal / no look-ahead)
Volatility  : Rolling, EWMA (RiskMetrics), XGBoost walk-forward forecast
Attribution : Component / marginal VaR, risk-parity (ERC) weights
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
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


def _xgb_feature_frame(returns: pd.Series, horizon: int) -> Tuple[pd.DataFrame, list]:
    r = returns.dropna()
    df = pd.DataFrame({"ret": r})
    for lag in [1, 2, 3, 5, 10, 21]:
        df[f"r{lag}"] = r.shift(lag)
        if lag >= 2:  # rolling(1).std() is all-NaN
            df[f"v{lag}"] = r.rolling(lag).std().shift(1)
    df["target"] = r.rolling(horizon).std().shift(-horizon) * np.sqrt(252)
    df.dropna(inplace=True)
    feats = [c for c in df.columns if c not in ("ret", "target")]
    return df, feats


def xgboost_vol_forecast(
    returns: pd.Series, horizon: int = 5, refit_every: int = 63
) -> Optional[pd.Series]:
    """Walk-forward (expanding-window) XGBoost realized-vol forecast.

    Returns ONLY out-of-sample predictions (the test period), refitting the model
    every ``refit_every`` business days. Returns None if xgboost is unavailable or
    data is too sparse. See ``xgboost_vol_skill`` for OOS accuracy metrics.
    """
    try:
        import xgboost as xgb
    except ImportError:
        return None
    df, feats = _xgb_feature_frame(returns, horizon)
    if len(df) < 400:
        return None

    start = int(len(df) * 0.6)  # initial training span; predict the remaining 40%
    preds = pd.Series(index=df.index[start:], dtype=float)
    model = None
    for i in range(start, len(df)):
        if model is None or (i - start) % refit_every == 0:
            model = xgb.XGBRegressor(
                n_estimators=120, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
            )
            model.fit(df.iloc[:i][feats], df.iloc[:i]["target"])
        preds.iloc[i - start] = float(model.predict(df.iloc[[i]][feats])[0])
    return preds.rename("xgb_vol")


def xgboost_vol_skill(returns: pd.Series, horizon: int = 5) -> Optional[Dict[str, object]]:
    """Out-of-sample forecast skill for the XGBoost vol model.

    Reports RMSE and QLIKE vs. realized vol, benchmarked against a naive
    "tomorrow = today's rolling vol" forecast. Returns aligned series for plotting.
    """
    preds = xgboost_vol_forecast(returns, horizon=horizon)
    if preds is None:
        return None
    df, _ = _xgb_feature_frame(returns, horizon)
    realized = df["target"].reindex(preds.index)
    naive = (df["ret"].rolling(horizon).std() * np.sqrt(252)).reindex(preds.index)
    valid = preds.notna() & realized.notna() & naive.notna()
    p, a, n = preds[valid], realized[valid], naive[valid]
    if len(p) < 30:
        return None

    def _rmse(f):
        return float(np.sqrt(np.mean((f - a) ** 2)))

    def _qlike(f):  # robust vol loss (variance space)
        fv, av = (f / np.sqrt(252)) ** 2, (a / np.sqrt(252)) ** 2
        fv = fv.clip(lower=1e-10)
        return float(np.mean(av / fv - np.log(av / fv) - 1))

    return {
        "rmse_model": round(_rmse(p), 4),
        "rmse_naive": round(_rmse(n), 4),
        "qlike_model": round(_qlike(p), 4),
        "qlike_naive": round(_qlike(n), 4),
        "n_oos": int(len(p)),
        "predicted": p,
        "realized": a,
        "naive": n,
    }


# ── VaR ─────────────────────────────────────────────────────────────────────

def historical_var(returns: pd.Series, confidence: float = 0.95, window: int = 252) -> pd.Series:
    return returns.rolling(window).quantile(1 - confidence).rename("historical_var")


def parametric_var(returns: pd.Series, confidence: float = 0.95, window: int = 252) -> pd.Series:
    z = norm.ppf(1 - confidence)
    mu = returns.rolling(window).mean()
    sig = returns.rolling(window).std()
    return (mu + z * sig).rename("parametric_var")


def cornish_fisher_var(returns: pd.Series, confidence: float = 0.95, window: int = 252) -> pd.Series:
    """Modified (Cornish-Fisher) VaR that adjusts the Gaussian quantile for the
    sample skewness and excess kurtosis — far more honest for fat-tailed assets."""
    z = norm.ppf(1 - confidence)
    mu = returns.rolling(window).mean()
    sig = returns.rolling(window).std()
    s = returns.rolling(window).skew()
    k = returns.rolling(window).kurt()  # excess kurtosis
    z_cf = (
        z
        + (z ** 2 - 1) * s / 6.0
        + (z ** 3 - 3 * z) * k / 24.0
        - (2 * z ** 3 - 5 * z) * (s ** 2) / 36.0
    )
    return (mu + z_cf * sig).rename("cornish_fisher_var")


def ewma_var(returns: pd.Series, confidence: float = 0.95, lam: float = 0.94) -> pd.Series:
    z = norm.ppf(1 - confidence)
    daily_vol = ewma_volatility(returns, lam=lam, annualize=False)
    return (z * daily_vol).rename("ewma_var")


def regime_adaptive_var(
    returns: pd.Series,
    regime_series: pd.Series,
    confidence: float = 0.95,
    min_obs: int = 30,
) -> pd.Series:
    """Causal per-regime empirical quantile (NO look-ahead).

    For each day t, the VaR is the (1-α) quantile of all PAST returns that
    occurred in the same regime as day t. Until a regime has ``min_obs`` history
    it falls back to the expanding full-sample quantile.
    """
    q = 1 - confidence
    df = pd.DataFrame({"ret": returns, "regime": regime_series}).dropna()
    out = pd.Series(index=df.index, dtype=float)

    # Maintain a growing list of past returns per regime.
    history: Dict[object, list] = {}
    all_past: list = []
    for t, (r, reg) in enumerate(zip(df["ret"].values, df["regime"].values)):
        bucket = history.get(reg, [])
        if len(bucket) >= min_obs:
            out.iloc[t] = float(np.quantile(bucket, q))
        elif len(all_past) >= min_obs:
            out.iloc[t] = float(np.quantile(all_past, q))
        else:
            out.iloc[t] = np.nan
        # update history AFTER using it (strictly causal)
        history.setdefault(reg, []).append(r)
        all_past.append(r)
    return out.rename("regime_adaptive_var")


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
        cornish_fisher_var(returns, confidence, window),
    ]
    if regime_series is not None:
        parts.append(regime_adaptive_var(returns, regime_series, confidence))
    return pd.concat(parts, axis=1).dropna(how="all")


# ── Portfolio helpers ────────────────────────────────────────────────────────

def portfolio_returns(returns_matrix: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Daily portfolio log-returns under constant weights (implies daily rebalancing).

    Aggregation is done in SIMPLE-return space — a weighted sum of log-returns is
    only an approximation — then converted back to log space for downstream
    time-additive statistics. Only dates where every holding traded are used.
    """
    tickers = [t for t in weights if t in returns_matrix.columns]
    w = np.array([weights[t] for t in tickers], dtype=float)
    w /= w.sum()
    sub = returns_matrix[tickers].dropna()
    simple = np.expm1(sub.values)          # log -> simple per asset
    port_simple = simple @ w               # weighted simple portfolio return
    port_log = np.log1p(port_simple)       # back to log space
    return pd.Series(port_log, index=sub.index, name="portfolio_return")


def compute_drawdown(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns).cumprod()
    return ((wealth / wealth.cummax()) - 1).rename("drawdown")


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / 252
    return float((excess.mean() / excess.std()) * np.sqrt(252)) if excess.std() != 0 else 0.0


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Like Sharpe, but penalizes only downside (below-target) volatility."""
    excess = returns - rf / 252
    downside = excess[excess < 0]
    dd = downside.std()
    return float((excess.mean() / dd) * np.sqrt(252)) if dd and dd != 0 else 0.0


def portfolio_metrics(returns: pd.Series) -> Dict[str, float]:
    dd = compute_drawdown(returns)
    var95 = float(returns.quantile(0.05))
    cvar95 = float(returns[returns <= var95].mean())
    downside_dev = float(returns[returns < 0].std() * np.sqrt(252) * 100)
    return {
        "Ann. Return (%)": round(returns.mean() * 252 * 100, 2),
        "Ann. Volatility (%)": round(returns.std() * np.sqrt(252) * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio(returns), 3),
        "Sortino Ratio": round(sortino_ratio(returns), 3),
        "Max Drawdown (%)": round(dd.min() * 100, 2),
        "Downside Dev. (%)": round(downside_dev, 2),
        "Skewness": round(float(returns.skew()), 3),
        "Excess Kurtosis": round(float(returns.kurtosis()), 3),
        "VaR 95% (daily %)": round(var95 * 100, 3),
        "CVaR 95% (daily %)": round(cvar95 * 100, 3),
    }


# ── Risk attribution ─────────────────────────────────────────────────────────

def risk_contributions(
    returns_matrix: pd.DataFrame, weights: Dict[str, float]
) -> pd.DataFrame:
    """Euler/component risk decomposition for a Gaussian portfolio.

    Component VaR sums to total portfolio VaR. Columns: weight, marginal,
    component (CTR), and percent-of-risk per asset.
    """
    tickers = [t for t in weights if t in returns_matrix.columns]
    sub = returns_matrix[tickers].dropna()
    w = np.array([weights[t] for t in tickers], dtype=float)
    w /= w.sum()

    cov = sub.cov().values * 252  # annualized covariance
    port_var = float(w @ cov @ w)
    port_vol = float(np.sqrt(port_var))
    mctr = (cov @ w) / port_vol            # marginal contribution to risk
    ctr = w * mctr                         # component contribution (sums to port_vol)
    pct = ctr / port_vol

    return pd.DataFrame(
        {
            "Weight (%)": np.round(w * 100, 2),
            "Marginal Risk": np.round(mctr, 4),
            "Component Vol (%)": np.round(ctr * 100, 3),
            "Risk Contribution (%)": np.round(pct * 100, 2),
        },
        index=tickers,
    )


def diversification_ratio(returns_matrix: pd.DataFrame, weights: Dict[str, float]) -> float:
    """Weighted-average asset vol ÷ portfolio vol. >1 means diversification is
    working; =1 means no diversification benefit."""
    tickers = [t for t in weights if t in returns_matrix.columns]
    sub = returns_matrix[tickers].dropna()
    w = np.array([weights[t] for t in tickers], dtype=float)
    w /= w.sum()
    cov = sub.cov().values * 252
    asset_vol = np.sqrt(np.diag(cov))
    port_vol = float(np.sqrt(w @ cov @ w))
    return float((w @ asset_vol) / port_vol) if port_vol else float("nan")


def risk_concentration(contrib_df: pd.DataFrame) -> Dict[str, float]:
    """Herfindahl concentration of risk vs capital (0 = perfectly diversified,
    1 = single-name). Effective number of bets = 1/HHI."""
    rc = (contrib_df["Risk Contribution (%)"] / 100.0).values
    wt = (contrib_df["Weight (%)"] / 100.0).values
    hhi_risk = float(np.sum(rc ** 2))
    hhi_cap = float(np.sum(wt ** 2))
    return {
        "hhi_risk": round(hhi_risk, 3),
        "hhi_capital": round(hhi_cap, 3),
        "effective_bets_risk": round(1 / hhi_risk, 2) if hhi_risk else float("nan"),
        "effective_bets_capital": round(1 / hhi_cap, 2) if hhi_cap else float("nan"),
    }


def rolling_beta(port_ret: pd.Series, market_ret: pd.Series, window: int = 252) -> pd.Series:
    """Rolling market beta = Cov(port, mkt) / Var(mkt)."""
    df = pd.concat([port_ret.rename("p"), market_ret.rename("m")], axis=1).dropna()
    cov = df["p"].rolling(window).cov(df["m"])
    var = df["m"].rolling(window).var()
    return (cov / var).rename("beta")


def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """Rolling annualized Sharpe ratio."""
    mu = returns.rolling(window).mean()
    sig = returns.rolling(window).std()
    return ((mu / sig) * np.sqrt(252)).rename("rolling_sharpe")


def risk_parity_weights(returns_matrix: pd.DataFrame, tickers, max_iter: int = 500) -> Dict[str, float]:
    """Equal-Risk-Contribution (risk parity) weights via a simple fixed-point
    iteration on the covariance matrix."""
    sub = returns_matrix[list(tickers)].dropna()
    cov = sub.cov().values * 252
    n = len(tickers)
    w = np.ones(n) / n
    for _ in range(max_iter):
        mrc = cov @ w
        rc = w * mrc
        target = rc.mean()
        w = w * (target / (rc + 1e-12)) ** 0.5
        w = np.clip(w, 1e-6, None)
        w = w / w.sum()
    return {t: round(float(wi), 4) for t, wi in zip(tickers, w)}
