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
