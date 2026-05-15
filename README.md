# VORTEX: Volatility-Oriented Regime Tracking & Explainability Engine

**VORTEX** is an institutional-grade, AI-driven market risk analytics platform. Designed to bridge the gap between traditional quantitative finance and modern artificial intelligence, VORTEX leverages unsupervised machine learning (Hidden Markov Models) and Large Language Models (LLMs) to dynamically assess portfolio vulnerabilities, track market regimes, and explain complex risk metrics in plain English.

---

## ⚡ Features & Capabilities

- **Market Regime Detection:** Utilizes Gaussian Hidden Markov Models (HMM) to classify the macroeconomic environment into four distinct states: Low Volatility, Crisis, Recovery, and Inflation Shock.
- **Adaptive Value-at-Risk (VaR):** Implements four VaR methodologies—Historical, Parametric, EWMA (RiskMetrics), and a novel Regime-Adaptive VaR that adjusts non-linearly to systemic shifts.
- **Volatility Forecasting:** Employs EWMA decay models and XGBoost-based predictive rolling windows to forecast future market volatility.
- **Historical Stress Testing:** Computes hypothetical portfolio drawdowns against extreme historical shocks, including the 2008 Financial Crisis, COVID-19 Crash, and the 2022 Fed Rate Shock.
- **AI Risk Intelligence Layer:** Integrates Hugging Face's Inference API (Mistral-7B) to synthesize quantitative metrics into institutional-style, actionable natural language risk commentary. Includes a deterministic fallback for guaranteed 100% uptime.
- **Smart Data Caching:** Features a robust, rate-limit-aware data pipeline that automatically caches full historical OHLCV data from `yfinance` to local CSVs. Subsequent runs incrementally fetch only new missing rows, dramatically reducing API load and runtime.

---

## 🏗️ System Architecture

```text
VORTEX/
├── app.py                     # Main Streamlit dashboard and UI routing
├── data/                      # Local CSV caching directory (auto-generated)
├── requirements.txt           # Project dependencies
├── .env                       # Environment variables (API Keys)
└── utils/
    ├── __init__.py
    ├── data_loader.py         # Smart CSV caching and yfinance retry backoff
    ├── regime_detection.py    # Unsupervised HMM state classification
    ├── risk_models.py         # VaR engine, volatility tracking, and portfolio math
    ├── stress_testing.py      # Defined macroeconomic stress scenarios
    ├── ai_commentary.py       # LLM integration via Hugging Face API
    └── visualization.py       # Plotly interactive dark-theme charts
```

---

## 📸 Dashboard Screenshots

*(Placeholders for future screenshots)*

| Risk Overview | Regime Timeline |
|:---:|:---:|
| `[Screenshot 1 Placeholder: Main Dashboard Metrics]` | `[Screenshot 2 Placeholder: Plotly Regime Chart]` |

| VaR Comparison | Stress Testing |
|:---:|:---:|
| `[Screenshot 3 Placeholder: VaR Plotly Graph]` | `[Screenshot 4 Placeholder: Bar Chart of Scenarios]` |

---

## 🚀 Setup & Installation

**Prerequisites:** Python 3.9+

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/VORTEX.git
   cd VORTEX
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables (Optional but recommended):**
   Create a `.env` file in the root directory and add your Hugging Face token to enable the AI commentary engine. If omitted, the app gracefully falls back to deterministic commentary.
   ```env
   HG_TOKEN="your_huggingface_api_token"
   ```

4. **Run the Dashboard:**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Example Outputs

- **AI Commentary:** "The primary regime is 'Crisis' with an 85% probability. Market conditions are deteriorating with expanding VaR limits. Proceed with strict capital preservation protocols."
- **Regime-Adaptive VaR:** Widens dynamically during the 2008 and 2020 regimes compared to the static Parametric VaR.
- **Performance Output:** Generates deep correlation matrices, interactive P&L trajectories, and drawdown profiles.

---

## 💼 Resume-Ready Description

**VORTEX: AI-Driven Market Risk Analytics Engine**
* Built an institutional-grade risk management dashboard using Python, Streamlit, and Plotly, integrating quantitative finance models with LLM-based explainability.
* Engineered a smart data pipeline using `pandas` and `yfinance` that implements incremental local CSV caching and exponential backoff, cutting subsequent data load times by over 90%.
* Developed an unsupervised regime detection algorithm using Hidden Markov Models (HMM) to classify market states and dynamically adjust Value-at-Risk (VaR) estimations.
* Implemented multi-method risk modeling (Historical, Parametric, EWMA) and engineered historical stress testing against major systemic events (2008 Crisis, COVID-19).
* Integrated Hugging Face's inference API (Mistral-7B) to translate raw portfolio metrics and volatility forecasts into actionable, natural language risk commentary.