"""
visualization.py — Plotly-based charts for VORTEX dashboard.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict

# Use institutional dark theme colors
TEMPLATE = "plotly_dark"
COLORS = px.colors.qualitative.Plotly
REGIME_COLORS = {
    "Low Volatility":  "#00d4aa",
    "Crisis":          "#ff4444",
    "Recovery":        "#44aaff",
    "Inflation Shock": "#ffaa00",
}

def plot_regime_timeline(regime_df: pd.DataFrame, price_series: pd.Series) -> go.Figure:
    """Plots the portfolio price / cumulative returns with regime background highlights."""
    df = pd.DataFrame({"Price": price_series, "Regime": regime_df["regime"]}).dropna()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Price"], mode='lines', name='Portfolio Value',
                             line=dict(color='white', width=1.5)))

    # Find contiguous regime blocks to draw background shapes
    shapes = []
    df["regime_change"] = (df["Regime"] != df["Regime"].shift(1))
    change_indices = df[df["regime_change"]].index.tolist()
    
    if len(change_indices) == 0:
        change_indices = [df.index[0]]
    if change_indices[0] != df.index[0]:
         change_indices.insert(0, df.index[0])
    change_indices.append(df.index[-1])

    for i in range(len(change_indices) - 1):
        start_date = change_indices[i]
        end_date = change_indices[i+1]
        regime = df.loc[start_date, "Regime"]
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=start_date,
                y0=0,
                x1=end_date,
                y1=1,
                fillcolor=REGIME_COLORS.get(regime, "#888888"),
                opacity=0.2,
                layer="below",
                line_width=0,
            )
        )

    fig.update_layout(
        title="Market Regimes & Portfolio Trajectory",
        template=TEMPLATE,
        shapes=shapes,
        xaxis_title="",
        yaxis_title="Normalized Value",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
        showlegend=False
    )
    return fig

def plot_regime_probabilities(regime_df: pd.DataFrame) -> go.Figure:
    """Stacked area chart of HMM regime probabilities."""
    prob_cols = [c for c in regime_df.columns if c.startswith("prob_")]
    
    fig = go.Figure()
    for col in prob_cols:
        regime_name = col.replace("prob_", "").replace("_", " ")
        fig.add_trace(go.Scatter(
            x=regime_df.index, 
            y=regime_df[col],
            mode='lines',
            stackgroup='one',
            name=regime_name,
            line=dict(width=0.5, color=REGIME_COLORS.get(regime_name, "#888"))
        ))
        
    fig.update_layout(
        title="HMM State Probabilities",
        template=TEMPLATE,
        xaxis_title="",
        yaxis_title="Probability",
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_volatility_comparison(vol_df: pd.DataFrame) -> go.Figure:
    """Line chart comparing volatility models."""
    fig = go.Figure()
    for col in vol_df.columns:
        fig.add_trace(go.Scatter(x=vol_df.index, y=vol_df[col], mode='lines', name=col, line=dict(width=1.5)))
        
    fig.update_layout(
        title="Volatility Forecast Comparison (Annualized)",
        template=TEMPLATE,
        xaxis_title="",
        yaxis_title="Volatility",
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_var_comparison(returns: pd.Series, var_df: pd.DataFrame) -> go.Figure:
    """Plots daily returns as a scatter with VaR levels as lines."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=returns.index, y=returns, mode='markers', name='Daily Return',
        marker=dict(color='rgba(255,255,255,0.3)', size=3)
    ))
    
    for col in var_df.columns:
        fig.add_trace(go.Scatter(x=var_df.index, y=var_df[col], mode='lines', name=col, line=dict(width=1.5)))
        
    fig.update_layout(
        title="Value-at-Risk (VaR) Exceedances",
        template=TEMPLATE,
        xaxis_title="",
        yaxis_title="Return",
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_drawdown(drawdown_series: pd.Series) -> go.Figure:
    """Area chart for portfolio drawdown."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown_series.index, y=drawdown_series,
        mode='lines',
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='#ff4444', width=1)
    ))
    fig.update_layout(
        title="Portfolio Drawdown",
        template=TEMPLATE,
        xaxis_title="",
        yaxis_title="Drawdown",
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
    )
    return fig

def plot_correlation_heatmap(returns_df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap of asset returns."""
    corr = returns_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmid=0
    ))
    fig.update_layout(
        title="Asset Correlation (Returns)",
        template=TEMPLATE,
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    return fig

def plot_stress_test(stress_df: pd.DataFrame) -> go.Figure:
    """Bar chart for stress test results."""
    fig = go.Figure(go.Bar(
        x=stress_df.index,
        y=stress_df["Portfolio Loss (%)"],
        marker_color='#ff4444'
    ))
    fig.update_layout(
        title="Stress Test Scenarios (Projected Loss %)",
        template=TEMPLATE,
        xaxis_title="",
        yaxis_title="Loss (%)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=350
    )
    return fig


def plot_var_backtest(returns: pd.Series, var_series: pd.Series) -> go.Figure:
    """Returns vs a VaR line, with breaches (exceedances) highlighted in red."""
    df = pd.concat([returns.rename("ret"), var_series.rename("var")], axis=1).dropna()
    breaches = df[df["ret"] < df["var"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ret"], mode="markers", name="Daily Return",
        marker=dict(color="rgba(150,170,200,0.35)", size=3),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["var"], mode="lines", name="VaR (99%/95%)",
        line=dict(color="#00d4aa", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=breaches.index, y=breaches["ret"], mode="markers", name="Breach",
        marker=dict(color="#ff4444", size=5, symbol="x"),
    ))
    fig.update_layout(
        title="VaR Backtest — Exceedances", template=TEMPLATE,
        xaxis_title="", yaxis_title="Return",
        margin=dict(l=20, r=20, t=40, b=20), height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_rolling_correlation(corr_series: pd.Series) -> go.Figure:
    """Average pairwise rolling correlation over time (diversification monitor)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=corr_series.index, y=corr_series, mode="lines",
        fill="tozeroy", name="Avg pairwise ρ", line=dict(color="#ffaa00", width=1.5),
    ))
    fig.update_layout(
        title="Average Pairwise Correlation (63-day rolling)", template=TEMPLATE,
        xaxis_title="", yaxis_title="Correlation",
        margin=dict(l=20, r=20, t=40, b=20), height=350,
    )
    return fig


def plot_risk_contributions(contrib_df: pd.DataFrame) -> go.Figure:
    """Grouped bars: capital weight vs risk contribution per asset."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=contrib_df.index, y=contrib_df["Weight (%)"], name="Capital Weight %",
        marker_color="#44aaff",
    ))
    fig.add_trace(go.Bar(
        x=contrib_df.index, y=contrib_df["Risk Contribution (%)"], name="Risk Contribution %",
        marker_color="#ff4444",
    ))
    fig.update_layout(
        title="Capital Weight vs Risk Contribution", template=TEMPLATE,
        barmode="group", xaxis_title="", yaxis_title="%",
        margin=dict(l=20, r=20, t=40, b=20), height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_forecast_skill(realized: pd.Series, predicted: pd.Series) -> go.Figure:
    """Out-of-sample XGBoost volatility forecast vs realized volatility."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=realized.index, y=realized, mode="lines", name="Realized vol",
        line=dict(color="rgba(150,170,200,0.7)", width=1.2),
    ))
    fig.add_trace(go.Scatter(
        x=predicted.index, y=predicted, mode="lines", name="XGBoost forecast (OOS)",
        line=dict(color="#00d4aa", width=1.5),
    ))
    fig.update_layout(
        title="Volatility Forecast Skill (Out-of-Sample)", template=TEMPLATE,
        xaxis_title="", yaxis_title="Annualized Vol",
        margin=dict(l=20, r=20, t=40, b=20), height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_regime_forward(fwd_df: pd.DataFrame) -> go.Figure:
    """Average forward return per regime — regime predictive content."""
    col = [c for c in fwd_df.columns if c.startswith("Avg Fwd")][0]
    colors = [REGIME_COLORS.get(r, "#888") for r in fwd_df.index]
    fig = go.Figure(go.Bar(x=fwd_df.index, y=fwd_df[col], marker_color=colors))
    fig.update_layout(
        title="Forward 21-Day Return by Regime", template=TEMPLATE,
        xaxis_title="", yaxis_title="Avg forward return (%)",
        margin=dict(l=20, r=20, t=40, b=20), height=350,
    )
    return fig


def plot_portfolio_value(price_series: pd.Series) -> go.Figure:
    """Headline portfolio equity curve (rebased to 100)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_series.index, y=price_series, mode="lines", name="Portfolio",
        line=dict(color="#00d4aa", width=2),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.08)",
    ))
    fig.update_layout(
        title="Portfolio Value (rebased to 100)", template=TEMPLATE,
        xaxis_title="", yaxis_title="Value",
        margin=dict(l=20, r=20, t=40, b=20), height=360, showlegend=False,
    )
    return fig


def plot_instruments(returns_df: pd.DataFrame) -> go.Figure:
    """Growth of $100 in each instrument over the common history (log scale)."""
    common = returns_df.dropna()
    growth = np.exp(common.cumsum()) * 100.0  # log-returns -> cumulative growth
    fig = go.Figure()
    palette = px.colors.qualitative.Vivid
    for i, col in enumerate(growth.columns):
        fig.add_trace(go.Scatter(
            x=growth.index, y=growth[col], mode="lines", name=col,
            line=dict(width=1.6, color=palette[i % len(palette)]),
        ))
    fig.update_layout(
        title="Instrument Performance — Growth of $100 (log scale)", template=TEMPLATE,
        xaxis_title="", yaxis_title="Value ($, log)", yaxis_type="log",
        margin=dict(l=20, r=20, t=40, b=20), height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_return_distribution(returns: pd.Series) -> go.Figure:
    """Histogram of returns with a fitted Normal overlay (fat-tail evidence)."""
    from scipy.stats import norm
    r = returns.dropna().values
    mu, sig = r.mean(), r.std()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r, histnorm="probability density", nbinsx=120,
        name="Empirical", marker_color="rgba(0,212,170,0.55)",
    ))
    xs = np.linspace(r.min(), r.max(), 400)
    fig.add_trace(go.Scatter(
        x=xs, y=norm.pdf(xs, mu, sig), mode="lines", name="Normal fit",
        line=dict(color="#ff4d4d", width=2),
    ))
    fig.update_layout(
        title="Return Distribution vs Normal", template=TEMPLATE,
        xaxis_title="Daily return", yaxis_title="Density",
        margin=dict(l=20, r=20, t=40, b=20), height=350, bargap=0.02,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_qq(returns: pd.Series) -> go.Figure:
    """Normal Q-Q plot — points bending off the line at the ends reveal fat tails."""
    from scipy import stats
    r = returns.dropna().values
    (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=osm, y=osr, mode="markers", name="Quantiles",
        marker=dict(color="rgba(68,170,255,0.55)", size=4),
    ))
    line_x = np.array([osm.min(), osm.max()])
    fig.add_trace(go.Scatter(
        x=line_x, y=intercept + slope * line_x, mode="lines", name="Normal",
        line=dict(color="#ff4d4d", width=2),
    ))
    fig.update_layout(
        title="Normal Q-Q Plot", template=TEMPLATE,
        xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles",
        margin=dict(l=20, r=20, t=40, b=20), height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_constituent_scatter(stats_df: pd.DataFrame) -> go.Figure:
    """Risk/return map of constituents; bubble size = risk contribution."""
    fig = go.Figure(go.Scatter(
        x=stats_df["vol"], y=stats_df["ret"], mode="markers+text",
        text=stats_df.index, textposition="top center",
        marker=dict(
            size=stats_df["risk"], sizemode="area",
            sizeref=2.0 * stats_df["risk"].max() / (40.0 ** 2), sizemin=4,
            color=stats_df["sharpe"], colorscale="Tealgrn", showscale=True,
            colorbar=dict(title="Sharpe"), line=dict(width=1, color="#0b0e14"),
        ),
    ))
    fig.update_layout(
        title="Constituent Risk / Return (bubble = risk contribution)", template=TEMPLATE,
        xaxis_title="Annualized volatility (%)", yaxis_title="Annualized return (%)",
        margin=dict(l=20, r=20, t=40, b=20), height=380,
    )
    return fig


def plot_rolling_beta_sharpe(beta: pd.Series, sharpe: pd.Series) -> go.Figure:
    """Rolling market beta and rolling Sharpe on dual axes."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=beta.index, y=beta, mode="lines", name="Beta to SPY",
                             line=dict(color="#44aaff", width=1.5)))
    fig.add_trace(go.Scatter(x=sharpe.index, y=sharpe, mode="lines", name="Rolling Sharpe (1y)",
                             yaxis="y2", line=dict(color="#00d4aa", width=1.5)))
    fig.update_layout(
        title="Rolling Market Beta & Sharpe (252-day)", template=TEMPLATE,
        xaxis_title="", yaxis=dict(title="Beta"),
        yaxis2=dict(title="Sharpe", overlaying="y", side="right", showgrid=False),
        margin=dict(l=20, r=20, t=40, b=20), height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_monthly_heatmap(returns: pd.Series) -> go.Figure:
    """Calendar heatmap of monthly portfolio returns (year × month)."""
    r = returns.dropna()
    monthly = (np.expm1(r)).add(1).groupby([r.index.year, r.index.month]).prod() - 1
    monthly = monthly * 100
    monthly.index.names = ["year", "month"]
    pivot = monthly.unstack("month")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.reindex(columns=range(1, 13))
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=months, y=[str(y) for y in pivot.index],
        colorscale="RdYlGn", zmid=0,
        colorbar=dict(title="%"), hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Monthly Returns (%)", template=TEMPLATE,
        margin=dict(l=20, r=20, t=40, b=20), height=380,
    )
    return fig


def average_pairwise_correlation(returns_df: pd.DataFrame, window: int = 63) -> pd.Series:
    """Mean of the off-diagonal rolling correlation matrix at each step."""
    clean = returns_df.dropna()
    n = clean.shape[1]
    roll = clean.rolling(window).corr()
    dates = clean.index[window - 1:]
    vals = []
    for d in dates:
        try:
            sub = roll.loc[d].values
        except KeyError:
            vals.append(np.nan); continue
        mask = ~np.eye(n, dtype=bool)
        vals.append(np.nanmean(sub[mask]))
    return pd.Series(vals, index=dates, name="avg_corr").dropna()
