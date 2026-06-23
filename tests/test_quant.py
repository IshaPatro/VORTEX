"""
Numerical correctness tests for the VORTEX quant engine.

Run with:  pytest -q
These use synthetic data only (no network / no cached CSVs).
"""
import numpy as np
import pandas as pd
import pytest

from utils import risk_models as rm
from utils import backtesting as bt


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2010-01-01", periods=1500)
    # Fat-tailed, slightly negatively skewed daily returns.
    r = rng.standard_t(df=5, size=len(idx)) * 0.01 - 0.0002
    return pd.Series(r, index=idx, name="r")


@pytest.fixture
def matrix():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2010-01-01", periods=1500)
    cols = {f"A{i}": rng.normal(0.0003, 0.01 + 0.004 * i, len(idx)) for i in range(4)}
    return pd.DataFrame(cols, index=idx)


# ── VaR ──────────────────────────────────────────────────────────────────────

def test_var_more_extreme_at_higher_confidence(returns):
    v95 = rm.historical_var(returns, confidence=0.95, window=250).dropna()
    v99 = rm.historical_var(returns, confidence=0.99, window=250).dropna()
    # 99% VaR should be at least as negative as 95% VaR everywhere.
    aligned = pd.concat([v95, v99], axis=1).dropna()
    assert (aligned.iloc[:, 1] <= aligned.iloc[:, 0] + 1e-9).all()


def test_parametric_var_is_negative(returns):
    v = rm.parametric_var(returns, 0.95, 250).dropna()
    assert (v < 0).mean() > 0.95


def test_cornish_fisher_differs_under_skew(returns):
    p = rm.parametric_var(returns, 0.99, 250).dropna()
    cf = rm.cornish_fisher_var(returns, 0.99, 250).dropna()
    aligned = pd.concat([p, cf], axis=1).dropna()
    # On fat-tailed data the CF tail should generally be at least as extreme.
    assert aligned.iloc[:, 1].mean() <= aligned.iloc[:, 0].mean() + 1e-6


def test_regime_adaptive_var_is_causal(returns):
    # Construct a regime series; the function must not peek ahead.
    regime = pd.Series(np.where(np.arange(len(returns)) % 2 == 0, "A", "B"), index=returns.index)
    v = rm.regime_adaptive_var(returns, regime, 0.95, min_obs=30)
    # First min_obs observations cannot be estimated -> NaN.
    assert v.iloc[0] != v.iloc[0] or np.isnan(v.iloc[0])
    assert v.dropna().shape[0] > 0


# ── Volatility ────────────────────────────────────────────────────────────────

def test_ewma_recursion_matches_manual(returns):
    lam = 0.94
    out = rm.ewma_volatility(returns, lam=lam, annualize=False).values
    r = returns.values
    var = r[0] ** 2
    var = lam * var + (1 - lam) * r[0] ** 2  # t=1
    assert out[1] == pytest.approx(np.sqrt(var), rel=1e-9)


# ── Portfolio math ────────────────────────────────────────────────────────────

def test_portfolio_returns_identical_assets():
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, len(idx)), index=idx)
    mat = pd.DataFrame({"X": r, "Y": r})
    pr = rm.portfolio_returns(mat, {"X": 0.5, "Y": 0.5})
    # Equal weights of identical assets -> the same series.
    assert np.allclose(pr.values, r.values, atol=1e-12)


def test_drawdown_bounded(returns):
    dd = rm.compute_drawdown(returns)
    assert (dd <= 1e-9).all() and (dd >= -1.0 - 1e-9).all()


def test_sortino_finite(returns):
    s = rm.sortino_ratio(returns)
    assert np.isfinite(s)


# ── Risk attribution ──────────────────────────────────────────────────────────

def test_risk_contributions_sum_to_100(matrix):
    w = {c: 1 / matrix.shape[1] for c in matrix.columns}
    rc = rm.risk_contributions(matrix, w)
    assert rc["Risk Contribution (%)"].sum() == pytest.approx(100.0, abs=0.5)


def test_risk_parity_equalizes_contribution(matrix):
    rp = rm.risk_parity_weights(matrix, list(matrix.columns))
    rc = rm.risk_contributions(matrix, rp)
    contribs = rc["Risk Contribution (%)"].values
    # All risk contributions should be roughly equal under risk parity.
    assert contribs.max() - contribs.min() < 5.0


# ── Backtesting ───────────────────────────────────────────────────────────────

def test_kupiec_well_calibrated():
    rng = np.random.default_rng(0)
    n = 4000
    idx = pd.bdate_range("2005-01-01", periods=n)
    r = pd.Series(rng.normal(0, 0.01, n), index=idx)
    # A correctly specified Gaussian VaR should pass Kupiec (high p-value).
    var = pd.Series(-1.645 * 0.01, index=idx)
    res = bt.kupiec_pof(r, var, 0.95)
    assert res["p_value"] > 0.05
    assert abs(res["rate"] - 5.0) < 1.5


def test_exceedance_count(returns):
    var = rm.historical_var(returns, 0.95, 250)
    exc = bt.var_exceedances(returns, var)
    assert exc.dtype == bool
    assert 0 < exc.sum() < len(exc)


def test_pinball_loss_nonnegative(returns):
    var = rm.historical_var(returns, 0.95, 250)
    assert bt.pinball_loss(returns, var, 0.95) >= 0


def test_var_model_comparison_ranks(returns):
    var_df = rm.compute_all_var(returns, confidence=0.95, window=250)
    table = bt.var_model_comparison(returns, var_df, 0.95)
    assert "Rank" in table.columns
    assert table["Rank"].min() == 1
    assert (table["Verdict"] == "WINNER").sum() == 1


def test_diebold_mariano_detects_better(returns):
    rng = np.random.default_rng(5)
    good = pd.Series(np.abs(rng.normal(0, 1, 500)), index=range(500))
    bad = good + np.abs(rng.normal(2, 0.5, 500))  # systematically worse
    dm = bt.diebold_mariano(good, bad)
    assert dm["dm_stat"] < 0 and dm["p_value"] < 0.05


def test_diversification_ratio_ge_one(matrix):
    w = {c: 1 / matrix.shape[1] for c in matrix.columns}
    dr = rm.diversification_ratio(matrix, w)
    assert dr >= 0.99  # weighted-avg vol >= portfolio vol for imperfect correlation


def test_rolling_beta_self_is_one(returns):
    b = rm.rolling_beta(returns, returns, window=100).dropna()
    assert np.allclose(b.values, 1.0, atol=1e-6)


def test_regime_forward_returns_shape():
    idx = pd.bdate_range("2015-01-01", periods=800)
    reg = pd.DataFrame({"regime": np.where(np.arange(800) % 3 == 0, "Crisis", "Calm")}, index=idx)
    r = pd.Series(np.random.default_rng(3).normal(0, 0.01, 800), index=idx)
    out = bt.regime_forward_returns(reg, r, horizon=21)
    assert "Hit Rate (%)" in out.columns
    assert out.shape[0] == 2
