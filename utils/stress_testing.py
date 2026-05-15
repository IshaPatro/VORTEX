"""
stress_testing.py — Predefined macro stress scenarios.

Scenarios calibrated from historical drawdown analytics:
  • 2008 Financial Crisis
  • COVID-19 Crash (Feb-Mar 2020)
  • Fed Rate Shock (2022)
  • Tech Selloff (2022 Q1)
"""
from __future__ import annotations
from typing import Dict, List
import pandas as pd
import numpy as np

# Each scenario: per-asset shock (daily return equivalent of total drawdown)
SCENARIOS: Dict[str, Dict[str, float]] = {
    "2008 Financial Crisis": {
        "SPY":     -0.565,
        "QQQ":     -0.498,
        "TLT":      0.258,
        "GLD":      0.055,
        "XLF":     -0.778,
        "XLK":     -0.518,
        "BTC-USD":  0.000,   # Bitcoin didn't exist yet
    },
    "COVID Crash (Feb-Mar 2020)": {
        "SPY":     -0.340,
        "QQQ":     -0.287,
        "TLT":      0.195,
        "GLD":      0.017,
        "XLF":     -0.390,
        "XLK":     -0.267,
        "BTC-USD": -0.630,
    },
    "Fed Rate Shock (2022)": {
        "SPY":     -0.195,
        "QQQ":     -0.328,
        "TLT":     -0.310,
        "GLD":     -0.030,
        "XLF":     -0.120,
        "XLK":     -0.380,
        "BTC-USD": -0.650,
    },
    "Tech Selloff (Q1 2022)": {
        "SPY":     -0.130,
        "QQQ":     -0.215,
        "TLT":     -0.085,
        "GLD":      0.060,
        "XLF":      0.015,
        "XLK":     -0.225,
        "BTC-USD": -0.420,
    },
}

SCENARIO_DESCRIPTIONS: Dict[str, str] = {
    "2008 Financial Crisis":
        "Global financial meltdown triggered by subprime mortgage collapse. "
        "Equities shed 50%+, financials nearly wiped out. Flight to Treasuries.",
    "COVID Crash (Feb-Mar 2020)":
        "Fastest bear market in history — 34% peak-to-trough in 33 days. "
        "Crypto collapsed, bonds rallied briefly before the Fed backstop.",
    "Fed Rate Shock (2022)":
        "Fastest rate-hiking cycle since Volcker. Both equities AND bonds fell >20% "
        "— a rare dual-asset drawdown. Tech and duration most impacted.",
    "Tech Selloff (Q1 2022)":
        "Multiple compression in high-duration tech names as real yields surged. "
        "Value held; growth crushed. Bitcoin led risk-off rotation.",
}


def run_stress_test(
    weights: Dict[str, float],
    portfolio_value: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Apply each scenario's per-asset shocks to the weighted portfolio.

    Parameters
    ----------
    weights          : dict {ticker: weight}  (need not sum to 1)
    portfolio_value  : notional value in dollars

    Returns
    -------
    DataFrame with one row per scenario showing P&L impact and % loss.
    """
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers], dtype=float)
    w /= w.sum()

    rows: List[Dict] = []
    for scenario_name, shocks in SCENARIOS.items():
        weighted_loss = 0.0
        contributions: Dict[str, float] = {}
        for i, ticker in enumerate(tickers):
            shock = shocks.get(ticker, 0.0)
            contribution = w[i] * shock
            contributions[f"{ticker} contrib"] = round(contribution * 100, 2)
            weighted_loss += contribution

        pnl = weighted_loss * portfolio_value
        rows.append({
            "Scenario":           scenario_name,
            "Portfolio Loss (%)": round(weighted_loss * 100, 2),
            "P&L ($)":            round(pnl, 0),
            **contributions,
            "Description":        SCENARIO_DESCRIPTIONS[scenario_name],
        })

    return pd.DataFrame(rows).set_index("Scenario")


def scenario_bar_data(stress_df: pd.DataFrame) -> pd.DataFrame:
    """Extract just scenario name + portfolio loss for bar chart."""
    return stress_df[["Portfolio Loss (%)"]].copy()


def worst_case_scenario(stress_df: pd.DataFrame) -> str:
    return stress_df["Portfolio Loss (%)"].idxmin()
