"""
regime_detection.py
===================
Hidden Markov Model regime classification engine.

Detects four market regimes from return + volatility features:
  0 → Low Volatility   (calm, trending)
  1 → Crisis           (negative returns, extreme vol)
  2 → Recovery         (positive returns, declining vol)
  3 → Inflation Shock  (negative real returns, elevated vol)

The state labels are assigned post-hoc by inspecting the mean return
and mean volatility of each learned HMM state.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_STATES = 4
REGIME_LABELS = {
    0: "Low Volatility",
    1: "Crisis",
    2: "Recovery",
    3: "Inflation Shock",
}
REGIME_COLORS = {
    "Low Volatility":  "#00d4aa",   # teal
    "Crisis":          "#ff4444",   # red
    "Recovery":        "#44aaff",   # blue
    "Inflation Shock": "#ffaa00",   # amber
}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_hmm_features(returns: pd.Series) -> np.ndarray:
    """
    Construct a 2-column feature matrix [return, realized_vol] for HMM fitting.
    Realized vol = rolling 5-day std of returns.
    """
    r = returns.copy()
    vol = r.rolling(5).std().fillna(method="bfill").fillna(0)
    feat = np.column_stack([r.values, vol.values])
    return feat


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def fit_hmm(
    returns: pd.Series,
    n_states: int = N_STATES,
    n_iter: int = 200,
    random_state: int = 42,
) -> Tuple[GaussianHMM, np.ndarray, StandardScaler]:
    """
    Fit a Gaussian HMM to the return series.

    Returns
    -------
    model      : fitted GaussianHMM
    states     : raw state sequence (integers 0..n_states-1)
    scaler     : fitted StandardScaler (for inference on new data)
    """
    feat = _build_hmm_features(returns)
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-4,
    )
    model.fit(feat_scaled)
    states = model.predict(feat_scaled)
    return model, states, scaler


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------

def _assign_regime_labels(
    states: np.ndarray,
    returns: pd.Series,
    n_states: int = N_STATES,
) -> Dict[int, str]:
    """
    Map raw HMM state integers → semantic regime names.

    Strategy:
      - Compute mean return and mean |return| (proxy for vol) per state.
      - Sort states by (mean_return, mean_vol) to assign meaningful labels:
          lowest mean_return + highest vol → Crisis
          highest mean_return + decreasing vol → Recovery
          lowest vol + positive return → Low Volatility
          remainder → Inflation Shock
    """
    r = returns.values
    state_stats: Dict[int, Dict[str, float]] = {}
    for s in range(n_states):
        mask = states == s
        if mask.sum() == 0:
            state_stats[s] = {"mean_ret": 0.0, "mean_vol": 0.0}
            continue
        state_stats[s] = {
            "mean_ret": float(r[mask].mean()),
            "mean_vol": float(np.abs(r[mask]).mean()),
        }

    # Rank by vol ascending, then by return descending
    vol_sorted = sorted(state_stats.keys(), key=lambda s: state_stats[s]["mean_vol"])
    ret_sorted = sorted(state_stats.keys(), key=lambda s: state_stats[s]["mean_ret"])

    label_map: Dict[int, str] = {}

    # Crisis = worst mean return
    crisis = ret_sorted[0]
    label_map[crisis] = "Crisis"

    # Low Volatility = lowest vol among remaining
    remaining_vol = [s for s in vol_sorted if s not in label_map]
    low_vol = remaining_vol[0]
    label_map[low_vol] = "Low Volatility"

    # Recovery = best mean return among remaining
    remaining_ret = [s for s in ret_sorted if s not in label_map]
    recovery = remaining_ret[-1]
    label_map[recovery] = "Recovery"

    # Inflation Shock = whatever is left
    for s in range(n_states):
        if s not in label_map:
            label_map[s] = "Inflation Shock"

    return label_map


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_regimes(
    returns: pd.Series,
    n_states: int = N_STATES,
    n_iter: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run the full regime detection pipeline.

    Parameters
    ----------
    returns : pd.Series
        Daily log-returns indexed by date.

    Returns
    -------
    pd.DataFrame with columns:
        - return          : original returns
        - regime_id       : raw HMM state integer
        - regime          : semantic label string
        - prob_0 … prob_3 : posterior state probabilities
    """
    returns = returns.dropna()
    if len(returns) < 60:
        raise ValueError("Need at least 60 observations for regime detection.")

    model, states, scaler = fit_hmm(returns, n_states=n_states, n_iter=n_iter, random_state=random_state)
    label_map = _assign_regime_labels(states, returns, n_states=n_states)

    feat = _build_hmm_features(returns)
    feat_scaled = scaler.transform(feat)
    posteriors = model.predict_proba(feat_scaled)   # shape (T, n_states)

    result = pd.DataFrame(index=returns.index)
    result["return"] = returns.values
    result["regime_id"] = states
    result["regime"] = [label_map[s] for s in states]

    for i in range(n_states):
        semantic = label_map[i]
        # Store posterior prob under the semantic regime name for clarity
        result[f"prob_{semantic.replace(' ', '_')}"] = posteriors[:, i]

    return result


def regime_summary(regime_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary table: regime name, % of time, avg return, avg abs-return.
    """
    rows = []
    for regime_name, grp in regime_df.groupby("regime"):
        rows.append({
            "Regime": regime_name,
            "% Time": round(100 * len(grp) / len(regime_df), 1),
            "Avg Daily Return (%)": round(grp["return"].mean() * 100, 4),
            "Avg |Return| (%)": round(grp["return"].abs().mean() * 100, 4),
            "Days": len(grp),
        })
    return pd.DataFrame(rows).set_index("Regime")


def get_current_regime(regime_df: pd.DataFrame) -> Dict[str, object]:
    """Return the most recent detected regime and its posterior probabilities."""
    last = regime_df.iloc[-1]
    prob_cols = [c for c in regime_df.columns if c.startswith("prob_")]
    probs = {c.replace("prob_", "").replace("_", " "): round(float(last[c]), 4) for c in prob_cols}
    return {
        "regime": last["regime"],
        "date": regime_df.index[-1],
        "probabilities": probs,
    }
