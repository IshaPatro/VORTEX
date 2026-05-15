"""
VORTEX: Volatility-Oriented Regime Tracking & Explainability Engine
app.py — Main Streamlit application
"""
import streamlit as st
import pandas as pd
import numpy as np
import time

from utils import data_loader, regime_detection, risk_models, stress_testing, visualization, ai_agents

st.set_page_config(
    page_title="VORTEX Risk Engine",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS override
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    h1, h2, h3 {
        color: #FFFFFF;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4aa;
    }
    .stAlert {
        background-color: rgba(255, 68, 68, 0.1);
        border: 1px solid #ff4444;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State & Data Loading ──────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    return data_loader.load_all_tickers()

if "data" not in st.session_state:
    with st.spinner("Initializing Data Layer (fetching yfinance with caching)..."):
        st.session_state.data = load_data()

data = st.session_state.data

# ── Sidebar Configuration ───────────────────────────────────────────────────

st.sidebar.title("🌪️ VORTEX Engine")
st.sidebar.markdown("Institutional Market Risk Platform")
st.sidebar.divider()

st.sidebar.subheader("Portfolio Allocation")
tickers = list(data.keys())
weights = {}
for t in tickers:
    w = st.sidebar.slider(f"{t} Weight", 0.0, 1.0, 1.0/len(tickers), 0.05)
    weights[t] = w

var_confidence = st.sidebar.selectbox("VaR Confidence Level", [0.95, 0.99])
lookback_window = st.sidebar.slider("Lookback Window (Days)", 60, 500, 252)

st.sidebar.divider()
st.sidebar.caption("Powered by Hmmlearn, XGBoost, and Mistral 7B.")

st.sidebar.divider()
st.sidebar.subheader("LLM Provider Status")
provider_placeholder = st.sidebar.empty()


# ── Main Computations ────────────────────────────────────────────────────────

# Compute returns
with st.spinner("Crunching analytics..."):
    returns_matrix = data_loader.build_returns_matrix(data)
    price_matrix = data_loader.build_price_matrix(data)
    
    port_ret = risk_models.portfolio_returns(returns_matrix, weights)
    port_price = (1 + port_ret).cumprod() * 100 # Normalize to 100

    # 1. Regime Detection (use SPY as proxy for market regime)
    spy_ret = returns_matrix.get("SPY", port_ret)
    regimes = regime_detection.detect_regimes(spy_ret)
    current_regime = regime_detection.get_current_regime(regimes)

    # 2. Risk Metrics
    metrics = risk_models.portfolio_metrics(port_ret)
    drawdown = risk_models.compute_drawdown(port_ret)

    # 3. Volatility Forecasts
    vol_roll = risk_models.rolling_volatility(port_ret, window=21)
    vol_ewma = risk_models.ewma_volatility(port_ret)
    vol_xgb = risk_models.xgboost_vol_forecast(port_ret)
    vol_df = pd.concat([vol_roll, vol_ewma], axis=1)
    if vol_xgb is not None:
         vol_df = pd.concat([vol_df, vol_xgb], axis=1)
    vol_df.dropna(inplace=True)

    # 4. VaR Comparison
    var_df = risk_models.compute_all_var(
        port_ret, 
        regime_series=regimes["regime"], 
        confidence=var_confidence, 
        window=lookback_window
    )

    # 5. Stress Test
    stress_res = stress_testing.run_stress_test(weights)
    worst = stress_testing.worst_case_scenario(stress_res)
    loss = stress_res.loc[worst, "Portfolio Loss (%)"]

    # 6. Run LangGraph Multi-Agent Workflow
    agent_state = ai_agents.run_risk_agents(
        regime_data=current_regime,
        metrics_data=metrics,
        stress_data={"worst_scenario": worst, "worst_loss": loss}
    )
    
    # Update UI status
    status = agent_state.get("provider_status", "Unknown")
    if status == "Claude":
        provider_placeholder.success("🟢 Claude Active")
    elif status == "Hugging Face":
        provider_placeholder.warning("🟡 Hugging Face Fallback Active")
    else:
        provider_placeholder.error("🔴 Local Template Mode")

# ── Dashboard Layout ────────────────────────────────────────────────────────

st.title("Risk Overview")

# Top Metrics Row
cols = st.columns(4)
cols[0].metric("Current Regime", current_regime["regime"])
cols[1].metric("Ann. Volatility", f"{metrics['Ann. Volatility (%)']}%")
cols[2].metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']}")
cols[3].metric("Max Drawdown", f"{metrics['Max Drawdown (%)']}%")

st.divider()

# Row 2: AI Commentary & Timeline
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🤖 AI Intelligence")
    
    with st.expander("Regime Analysis", expanded=True):
        st.write(agent_state["regime_analysis"])
    
    with st.expander("Portfolio Health", expanded=True):
         st.write(agent_state["var_analysis"])
         
    with st.expander("Stress Alert", expanded=True):
         st.write(agent_state["stress_analysis"])

with col2:
    st.plotly_chart(visualization.plot_regime_timeline(regimes, port_price), use_container_width=True)
    st.plotly_chart(visualization.plot_regime_probabilities(regimes), use_container_width=True)


st.divider()

# Row 3: VaR & Volatility
col3, col4 = st.columns(2)

with col3:
    st.subheader("Value-at-Risk (VaR) Engine")
    st.plotly_chart(visualization.plot_var_comparison(port_ret[-lookback_window:], var_df[-lookback_window:]), use_container_width=True)

with col4:
    st.subheader("Volatility Forecasting")
    st.plotly_chart(visualization.plot_volatility_comparison(vol_df[-lookback_window:]), use_container_width=True)


st.divider()

# Row 4: Stress Testing & Drawdowns
col5, col6 = st.columns(2)

with col5:
    st.subheader("Historical Stress Testing")
    st.plotly_chart(visualization.plot_stress_test(stress_res), use_container_width=True)
    st.dataframe(stress_res[["Portfolio Loss (%)", "Description"]], use_container_width=True)

with col6:
    st.subheader("Drawdown & Correlation")
    st.plotly_chart(visualization.plot_drawdown(drawdown), use_container_width=True)
    st.plotly_chart(visualization.plot_correlation_heatmap(returns_matrix), use_container_width=True)
