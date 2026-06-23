# VORTEX: Volatility-Oriented Regime Tracking & Explainability Engine

**VORTEX** is an institutional-grade, AI-driven market risk analytics platform. Designed to bridge the gap between traditional quantitative finance and modern artificial intelligence, VORTEX leverages unsupervised machine learning (Hidden Markov Models) and Large Language Models (LLMs) to dynamically assess portfolio vulnerabilities, track market regimes, and explain complex risk metrics in plain English.

---

## ⚡ Features & Capabilities

- **Market Regime Detection:** Gaussian Hidden Markov Models classify the environment into four states (Low Volatility, Crisis, Recovery, Inflation Shock). The live regime uses **filtered (causal) probabilities** — the honest real-time estimate, not the full-sample smoothed view.
- **Adaptive Value-at-Risk (VaR):** Five methodologies — Historical, Parametric, EWMA (RiskMetrics), **Cornish-Fisher** (skew/kurtosis-adjusted), and a **causal** Regime-Adaptive VaR (no look-ahead).
- **VaR / ES Backtesting:** Regulatory-style validation — **Kupiec POF**, **Christoffersen** (independence/conditional coverage), and an **Expected-Shortfall** test.
- **Volatility Forecasting:** Rolling, EWMA, and a **walk-forward** XGBoost model scored **out-of-sample** (RMSE / QLIKE) against a naive benchmark.
- **Risk Attribution:** Euler component/marginal VaR decomposition plus a **risk-parity (ERC)** weighting alternative.
- **Stress Testing:** Calibrated per-asset shocks, **historical path replay** (actual crisis returns), and **reverse stress testing**.
- **AI Risk Intelligence Layer:** A LangGraph multi-agent workflow (Regime Analyst, VaR/Risk, Stress-Testing agents) synthesizes quantitative metrics into institutional-style, actionable natural language risk commentary. Powered by **Claude** commentary (pre-generated and cached to disk for instant, token-free serving), backed by a **resilient provider chain**: cached Claude → **Hugging Face** Inference API (optional cloud fallback) → deterministic templates (guaranteed 100% uptime, never crashes).
- **Smart Data Caching:** Features a robust, rate-limit-aware data pipeline that automatically caches full historical OHLCV data from `yfinance` to local CSVs. Subsequent runs incrementally fetch only new missing rows, dramatically reducing API load and runtime.

---

## 🏗️ System Architecture

VORTEX ships as a **static website** (deployable to GitHub Pages). A Python pipeline
runs the analytics once and bakes the results into `assets/data.js`; the page renders
everything client-side with Plotly.js — no server required.

```text
VORTEX/
├── index.html                 # ★ Static quant dashboard (GitHub Pages entry point)
├── build_static.py            # Runs the pipeline & bakes results into assets/data.js
├── assets/
│   ├── dashboard.css          # Dashboard styling
│   ├── dashboard.js           # Client-side renderer (Plotly)
│   └── data.js                # Auto-generated analytics payload (committed)
├── site/                      # Plain-English explainer site (concepts guide)
│   ├── index.html
│   ├── styles.css
│   └── script.js
├── data/                      # Local CSV caching directory (auto-generated)
├── llm_cache.json             # Persistent LLM response cache (auto-generated)
├── requirements.txt           # Python dependencies (for regenerating data)
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         # Smart CSV caching and yfinance retry backoff
│   ├── regime_detection.py    # Unsupervised HMM state classification
│   ├── risk_models.py         # VaR engine, volatility tracking, and portfolio math
│   ├── stress_testing.py      # Stress scenarios, historical replay, reverse stress
│   ├── backtesting.py         # Kupiec, Christoffersen, ES test, regime forward-returns
│   ├── ai_agents.py           # LangGraph multi-agent risk-commentary workflow
│   └── visualization.py       # Plotly interactive dark-theme charts
├── tests/                     # pytest numerical-correctness suite
└── llm/
    ├── __init__.py
    ├── provider_router.py     # Resilient LangChain chat model (cache → HF → template)
    ├── huggingface_provider.py# Optional cloud fallback: Hugging Face Inference API
    ├── response_cache.py      # Persistent on-disk cache of Claude responses
    └── fallback_handler.py    # Deterministic templates + provider usage logging
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

## 🚀 Usage

### View the dashboard (no Python needed)
The dashboard is fully static. Just open `index.html` in any browser, or serve the
folder locally:
```bash
python -m http.server 8000   # then visit http://localhost:8000
```
The plain-English **concepts guide** lives at `site/index.html`.

### Deploy to GitHub Pages
Push the repo and enable Pages (Settings → Pages → deploy from branch, root). The
committed `index.html` + `assets/` are served directly — `https://<user>.github.io/VORTEX/`.

### Regenerate the analytics (optional)
To refresh the data, regime detection, charts and AI commentary with the latest market
data, re-run the builder:

**Prerequisites:** Python 3.9+

```bash
python -m venv .venvVortex
source .venvVortex/bin/activate   # Windows: .venvVortex\Scripts\activate
pip install -r requirements.txt

python build_static.py            # rebuilds assets/data.js
```

**AI commentary (no API keys required):** commentary is generated by **Claude** and
cached in `llm_cache.json`, then baked into `assets/data.js`. For prompts not in the
cache the builder falls back to an optional Hugging Face model (if `HG_TOKEN` is set)
and finally to deterministic templates, so it never fails.

---

## 📊 Example Outputs

- **AI Commentary:** "The primary regime is 'Crisis' with an 85% probability. Market conditions are deteriorating with expanding VaR limits. Proceed with strict capital preservation protocols."
- **Regime-Adaptive VaR:** Widens dynamically during the 2008 and 2020 regimes compared to the static Parametric VaR.
- **Performance Output:** Generates deep correlation matrices, interactive P&L trajectories, and drawdown profiles.

---

## 💼 Resume-Ready Description

**VORTEX: AI-Driven Market Risk Analytics Engine**
* Built an institutional-grade, statically-deployable (GitHub Pages) risk dashboard using Python and Plotly.js, integrating quantitative finance models with LLM-based explainability.
* Engineered a smart data pipeline using `pandas` and `yfinance` that implements incremental local CSV caching and exponential backoff, cutting subsequent data load times by over 90%.
* Developed an unsupervised regime detection algorithm using Hidden Markov Models (HMM) to classify market states and dynamically adjust Value-at-Risk (VaR) estimations.
* Implemented multi-method risk modeling (Historical, Parametric, EWMA, XGBoost volatility forecasting) and engineered historical stress testing against major systemic events (2008 Crisis, COVID-19).
* Built a LangGraph multi-agent commentary engine powered by Claude with a resilient, self-healing provider chain (cached Claude → Hugging Face → deterministic templates) to translate raw portfolio metrics into actionable natural-language risk commentary with guaranteed uptime.

