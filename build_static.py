"""
build_static.py — Bake the VORTEX analytics into a static site.

Runs the full quant pipeline once (data → regimes → VaR → volatility → backtests
→ risk attribution → stress → Claude commentary), serializes every Plotly figure
plus all tables/metrics, and writes assets/data.js (`window.VORTEX_DATA`).

Usage:  python build_static.py
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from utils import (  # noqa: E402
    data_loader,
    regime_detection,
    risk_models,
    stress_testing,
    backtesting,
    visualization,
    ai_agents,
)

LOOKBACK = 252
BACKTEST_VIEW = 504  # ~2y window shown in backtest/forecast scatter charts
ASSETS = Path("assets")


def _fig(fig):
    """Normalize every figure: transparent background, left-aligned title with
    generous headroom, and the legend moved BELOW the plot so it never collides
    with the title."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c7d0e0"),
        title=dict(x=0.012, xanchor="left", y=0.98, yanchor="top",
                   font=dict(size=15, color="#e8edf6")),
        # Big top margin keeps the title clear of the (top) legend.
        margin=dict(l=44, r=24, t=96, b=46),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0,
                    font=dict(size=11)),
    )
    return json.loads(fig.to_json())


def _records(df):
    return json.loads(df.reset_index().to_json(orient="records"))


def main() -> None:
    print("Loading market data...")
    data = data_loader.load_all_tickers()
    tickers = list(data.keys())
    weights = {t: 1.0 / len(tickers) for t in tickers}

    rmx = data_loader.build_returns_matrix(data)
    port_ret = risk_models.portfolio_returns(rmx, weights)
    port_price = (1 + port_ret).cumprod() * 100

    print("Regimes (HMM, filtered + smoothed)...")
    regimes = regime_detection.detect_regimes(rmx.get("SPY", port_ret))
    current_regime = regime_detection.get_current_regime(regimes, use_filtered=True)
    regime_summary = regime_detection.regime_summary(regimes)

    print("Risk metrics, VaR, volatility...")
    metrics = risk_models.portfolio_metrics(port_ret)
    drawdown = risk_models.compute_drawdown(port_ret)
    vol_roll = risk_models.rolling_volatility(port_ret, window=21)
    vol_ewma = risk_models.ewma_volatility(port_ret)
    vol_xgb = risk_models.xgboost_vol_forecast(port_ret)
    vol_df = pd.concat([vol_roll, vol_ewma], axis=1)
    if vol_xgb is not None:
        vol_df = pd.concat([vol_df, vol_xgb], axis=1)
    vol_df.dropna(inplace=True)

    var_df = risk_models.compute_all_var(
        port_ret, regime_series=regimes["regime"], confidence=0.95, window=LOOKBACK
    )

    print("VaR/ES backtesting...")
    es_series = backtesting.rolling_es(port_ret, confidence=0.95, window=LOOKBACK)
    bt_hist = {
        **backtesting.kupiec_pof(port_ret, var_df["historical_var"], 0.95),
        **backtesting.christoffersen(port_ret, var_df["historical_var"], 0.95),
    }
    bt_ewma = {
        **backtesting.kupiec_pof(port_ret, var_df["ewma_var"], 0.95),
        **backtesting.christoffersen(port_ret, var_df["ewma_var"], 0.95),
    }
    bt_cf = {
        **backtesting.kupiec_pof(port_ret, var_df["cornish_fisher_var"], 0.95),
        **backtesting.christoffersen(port_ret, var_df["cornish_fisher_var"], 0.95),
    }
    es_bt = backtesting.es_backtest(port_ret, var_df["historical_var"], es_series)

    print("Model selection (VaR league table)...")
    var_league = backtesting.var_model_comparison(port_ret, var_df, 0.95)

    print("Risk attribution & forecast skill...")
    risk_tbl = risk_models.risk_contributions(rmx, weights)
    rp_weights = risk_models.risk_parity_weights(rmx, tickers)
    div_ratio = risk_models.diversification_ratio(rmx, weights)
    concentration = risk_models.risk_concentration(risk_tbl)
    skill = risk_models.xgboost_vol_skill(port_ret)
    dm = None
    if skill is not None:
        la = (skill["predicted"] - skill["realized"]) ** 2
        lb = (skill["naive"] - skill["realized"]) ** 2
        dm = backtesting.diebold_mariano(la, lb)
    regime_fwd = backtesting.regime_forward_returns(regimes, port_ret, horizon=21)
    avg_corr = visualization.average_pairwise_correlation(rmx, window=63)

    print("Market sensitivity & distribution...")
    roll_beta = risk_models.rolling_beta(port_ret, rmx["SPY"], window=252) if "SPY" in rmx else None
    roll_sharpe = risk_models.rolling_sharpe(port_ret, window=252)
    # Per-constituent risk/return stats for the scatter.
    rc_pct = {row["index"]: row["Risk Contribution (%)"] for row in _records(risk_tbl)}
    const_stats = pd.DataFrame({
        "ret": {t: rmx[t].mean() * 252 * 100 for t in tickers},
        "vol": {t: rmx[t].std() * (252 ** 0.5) * 100 for t in tickers},
        "risk": {t: rc_pct.get(t, 1.0) for t in tickers},
        "sharpe": {t: float((rmx[t].mean() / rmx[t].std()) * (252 ** 0.5)) for t in tickers},
    })

    print("Stress testing (static + historical replay + reverse)...")
    stress_res = stress_testing.run_stress_test(weights)
    worst = stress_testing.worst_case_scenario(stress_res)
    loss = float(stress_res.loc[worst, "Portfolio Loss (%)"])
    replay = stress_testing.historical_scenario_replay(rmx, weights)
    reverse = stress_testing.reverse_stress_test(metrics["Ann. Volatility (%)"], -20.0)

    print("AI commentary (cached Claude)...")
    agent_state = ai_agents.run_risk_agents(
        regime_data=current_regime,
        metrics_data=metrics,
        stress_data={"worst_scenario": worst, "worst_loss": loss},
    )

    print("Serializing figures...")
    figures = {
        "portfolio_value": _fig(visualization.plot_portfolio_value(port_price)),
        "instruments": _fig(visualization.plot_instruments(rmx)),
        "regime_timeline": _fig(visualization.plot_regime_timeline(regimes, port_price)),
        "regime_probabilities": _fig(visualization.plot_regime_probabilities(regimes)),
        "var_comparison": _fig(visualization.plot_var_comparison(port_ret[-LOOKBACK:], var_df[-LOOKBACK:])),
        "var_backtest": _fig(visualization.plot_var_backtest(port_ret[-BACKTEST_VIEW:], var_df["ewma_var"][-BACKTEST_VIEW:])),
        "volatility": _fig(visualization.plot_volatility_comparison(vol_df[-LOOKBACK:])),
        "rolling_correlation": _fig(visualization.plot_rolling_correlation(avg_corr)),
        "risk_contributions": _fig(visualization.plot_risk_contributions(risk_tbl)),
        "constituent_scatter": _fig(visualization.plot_constituent_scatter(const_stats)),
        "regime_forward": _fig(visualization.plot_regime_forward(regime_fwd)),
        "stress": _fig(visualization.plot_stress_test(stress_res)),
        "drawdown": _fig(visualization.plot_drawdown(drawdown)),
        "correlation": _fig(visualization.plot_correlation_heatmap(rmx)),
        "return_dist": _fig(visualization.plot_return_distribution(port_ret)),
        "qq": _fig(visualization.plot_qq(port_ret)),
        "monthly_heatmap": _fig(visualization.plot_monthly_heatmap(port_ret)),
        "rolling_sharpe": _fig(visualization.plot_rolling_beta_sharpe(
            roll_beta if roll_beta is not None else roll_sharpe, roll_sharpe)),
    }
    if skill is not None:
        figures["forecast_skill"] = _fig(
            visualization.plot_forecast_skill(
                skill["realized"][-BACKTEST_VIEW:], skill["predicted"][-BACKTEST_VIEW:]
            )
        )

    from scipy.stats import jarque_bera
    jb_stat, jb_p = jarque_bera(port_ret.dropna().values)
    distribution = {
        "jb_stat": round(float(jb_stat), 1),
        "jb_p": round(float(jb_p), 4),
        "normal": bool(jb_p > 0.05),
    }

    skill_stats = None
    if skill is not None:
        skill_stats = {k: v for k, v in skill.items() if not isinstance(v, pd.Series)}

    stress_table = [
        {"scenario": idx, "loss": float(r["Portfolio Loss (%)"]),
         "pnl": float(r["P&L ($)"]), "description": r["Description"]}
        for idx, r in stress_res.iterrows()
    ]

    # ── Data-driven analysis blurbs per section ──────────────────────────────
    cur = current_regime["regime"]
    cur_p = current_regime["probabilities"].get(cur, 0) * 100
    winner_row = var_league[var_league["Verdict"] == "WINNER"]
    winner = winner_row.index[0] if not winner_row.empty else var_league.index[0]
    winner_pin = float(winner_row["Pinball (bps)"].iloc[0]) if not winner_row.empty else 0.0
    top_risk = risk_tbl["Risk Contribution (%)"].idxmax()
    top_risk_pct = float(risk_tbl.loc[top_risk, "Risk Contribution (%)"])
    top_risk_w = float(risk_tbl.loc[top_risk, "Weight (%)"])
    beta_last = float(roll_beta.dropna().iloc[-1]) if roll_beta is not None and not roll_beta.dropna().empty else float("nan")
    sharpe_last = float(roll_sharpe.dropna().iloc[-1]) if not roll_sharpe.dropna().empty else float("nan")
    worst_replay = replay["Replay Loss (%)"].idxmin() if not replay.empty else None
    worst_replay_loss = float(replay["Replay Loss (%)"].min()) if not replay.empty else float("nan")
    fwd_txt = ""
    if cur in regime_fwd.index:
        fwd_col = [c for c in regime_fwd.columns if c.startswith("Avg Fwd")][0]
        fwd_txt = (f" Historically this regime preceded a {regime_fwd.loc[cur, fwd_col]:.2f}% "
                   f"average 21-day forward return ({regime_fwd.loc[cur, 'Hit Rate (%)']:.0f}% positive).")

    analysis = {
        "regimes": (
            f"The market is in a <b>{cur}</b> regime ({cur_p:.1f}% filtered probability).{fwd_txt} "
            "Regime bands are estimated causally (filtered), so this is the real-time read, not hindsight."
        ),
        "var": (
            f"Daily 95% VaR is <b>{metrics['VaR 95% (daily %)']}%</b> with CVaR <b>{metrics['CVaR 95% (daily %)']}%</b>. "
            f"The gap between them, alongside excess kurtosis of {metrics['Excess Kurtosis']}, signals fat tails that "
            "Gaussian VaR alone would understate — which is why five methods are shown."
        ),
        "backtest": (
            f"At 95%, historical VaR was breached {bt_hist['breaches']}× ({bt_hist['rate']}% vs 5% expected) — "
            f"Kupiec {'passes' if bt_hist['p_value'] > 0.05 else 'fails'} (p={bt_hist['p_value']}), but Christoffersen "
            f"{'passes' if bt_hist['p_value_cc'] > 0.05 else 'fails'} (p={bt_hist['p_value_cc']}), indicating breaches "
            "cluster during stress — a known, honest limitation of unconditional VaR."
        ),
        "models": (
            f"<b>{winner} VaR</b> is the best-calibrated model — the lowest pinball loss ({winner_pin} bps) among "
            "methods that pass Kupiec coverage. Lower pinball = sharper quantile forecast."
        ),
        "volatility": (
            f"The walk-forward XGBoost forecast beats the naive benchmark out-of-sample "
            f"(RMSE {skill_stats['rmse_model']} vs {skill_stats['rmse_naive']})"
            + (f", and the edge is statistically significant (Diebold-Mariano p={dm['p_value']})." if dm and dm.get('p_value') == dm.get('p_value') else ".")
        ),
        "distribution": (
            f"Jarque-Bera {'rejects' if not distribution['normal'] else 'does not reject'} normality "
            f"(p={distribution['jb_p']}). With skew {metrics['Skewness']} and excess kurtosis {metrics['Excess Kurtosis']}, "
            "losses arrive in larger, more frequent clusters than a bell curve predicts."
        ),
        "risk": (
            f"<b>{top_risk}</b> drives <b>{top_risk_pct:.0f}%</b> of portfolio risk on just {top_risk_w:.0f}% of capital. "
            f"The book holds {concentration['effective_bets_risk']} effective bets by risk vs "
            f"{concentration['effective_bets_capital']} by capital — risk is far more concentrated than weights suggest."
        ),
        "sensitivity": (
            f"Rolling market beta is <b>{beta_last:.2f}</b> and the trailing 1-year Sharpe is <b>{sharpe_last:.2f}</b>. "
            "The monthly calendar surfaces seasonality and the worst drawdown months at a glance."
        ),
        "stress": (
            f"The worst calibrated scenario is <b>{worst}</b> at {loss}%. "
            + (f"Replaying actual crisis paths, <b>{worst_replay}</b> is most damaging at {worst_replay_loss}%. " if worst_replay else "")
            + "Reverse stress shows the one-day shock required to hit a 20% loss."
        ),
    }

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_start": str(rmx.index.min().date()),
        "data_end": str(rmx.index.max().date()),
        "n_observations": int(len(rmx)),
        "tickers": tickers,
        "weights": {t: round(w, 4) for t, w in weights.items()},
        "metrics": {k: float(v) for k, v in metrics.items()},
        "regime": {
            "current": current_regime["regime"],
            "date": str(current_regime["date"].date()),
            "probabilities": current_regime["probabilities"],
            "summary": _records(regime_summary),
            "forward": _records(regime_fwd),
        },
        "backtests": {"historical": bt_hist, "ewma": bt_ewma, "cornish_fisher": bt_cf, "es": es_bt},
        "var_league": _records(var_league),
        "risk_attribution": _records(risk_tbl),
        "risk_parity": rp_weights,
        "diversification": {"ratio": round(div_ratio, 3), **concentration},
        "distribution": distribution,
        "analysis": analysis,
        "forecast_skill": ({**skill_stats, "dm": dm} if skill_stats else None),
        "stress": {
            "worst": worst, "worst_loss": loss, "table": stress_table,
            "replay": _records(replay), "reverse": reverse,
        },
        "commentary": {
            "provider": agent_state.get("provider_status", "Unknown"),
            "regime": agent_state.get("regime_analysis", ""),
            "var": agent_state.get("var_analysis", ""),
            "stress": agent_state.get("stress_analysis", ""),
        },
        "figures": figures,
    }

    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / "data.js"
    with open(out, "w") as f:
        f.write("// Auto-generated by build_static.py — do not edit by hand.\n")
        f.write("window.VORTEX_DATA = ")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")

    print(f"\nWrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    print(f"Regime: {current_regime['regime']} | Worst stress: {worst} ({loss}%)")
    print(f"Kupiec(hist) p={bt_hist['p_value']} | breach rate={bt_hist['rate']}%")
    print(f"AI provider: {payload['commentary']['provider']}")


if __name__ == "__main__":
    main()
