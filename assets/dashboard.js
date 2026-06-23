// VORTEX static dashboard renderer — reads window.VORTEX_DATA, draws with Plotly.

(function () {
  "use strict";

  const D = window.VORTEX_DATA;
  const $ = (id) => document.getElementById(id);

  const REGIME_COLORS = {
    "Low Volatility": "#00d4aa",
    Crisis: "#ff4d4d",
    Recovery: "#44aaff",
    "Inflation Shock": "#ffaa00",
  };

  if (!D) {
    document.body.insertAdjacentHTML(
      "afterbegin",
      '<p style="padding:40px;color:#ff4d4d;font-family:monospace">' +
        "VORTEX data not found. Run <b>python build_static.py</b> to generate assets/data.js." +
        "</p>"
    );
    return;
  }

  const fmt = (n, d = 2) =>
    typeof n === "number" ? n.toFixed(d) : n;
  const signClass = (n) => (n > 0 ? "pos" : n < 0 ? "neg" : "neutral");

  // ---------- Meta card ----------
  const rc = REGIME_COLORS[D.regime.current] || "#888";
  $("metaCard").innerHTML = `
    <h4>Snapshot</h4>
    <div class="meta-row"><span>Current regime</span>
      <span class="regime-pill" style="background:${rc}22;color:${rc}">
        <span class="dot" style="background:${rc}"></span>${D.regime.current}</span></div>
    <div class="meta-row"><span>As of</span><strong>${D.regime.date}</strong></div>
    <div class="meta-row"><span>Data window</span><strong>${D.data_start} → ${D.data_end}</strong></div>
    <div class="meta-row"><span>Observations</span><strong>${D.n_observations.toLocaleString()}</strong></div>
    <div class="meta-row"><span>Assets</span><strong>${D.tickers.length}</strong></div>
    <div class="meta-row"><span>AI provider</span><strong>${D.commentary.provider}</strong></div>
    <div class="meta-row"><span>Built</span><strong>${D.generated_at}</strong></div>`;

  // ---------- Portfolio line ----------
  $("portfolioLine").textContent =
    "Equal-weight portfolio: " + D.tickers.join(" · ");

  // ---------- KPI grid ----------
  const m = D.metrics;
  const kpis = [
    { label: "Current Regime", value: D.regime.current, cls: "neutral", raw: true },
    { label: "Ann. Return", value: m["Ann. Return (%)"], suffix: "%" },
    { label: "Ann. Volatility", value: m["Ann. Volatility (%)"], suffix: "%", neutral: true },
    { label: "Sharpe Ratio", value: m["Sharpe Ratio"], d: 2 },
    { label: "Sortino Ratio", value: m["Sortino Ratio"], d: 2 },
    { label: "Max Drawdown", value: m["Max Drawdown (%)"], suffix: "%" },
    { label: "Downside Dev.", value: m["Downside Dev. (%)"], suffix: "%", neutral: true },
    { label: "VaR 95% (daily)", value: m["VaR 95% (daily %)"], suffix: "%", d: 3 },
    { label: "CVaR 95% (daily)", value: m["CVaR 95% (daily %)"], suffix: "%", d: 3 },
    { label: "Skewness", value: m["Skewness"], d: 3 },
    { label: "Excess Kurtosis", value: m["Excess Kurtosis"], d: 2, neutral: true },
    { label: "Worst Stress", value: D.stress.worst_loss, suffix: "%" },
  ];
  $("kpiGrid").innerHTML = kpis
    .map((k) => {
      if (k.raw) {
        const c = REGIME_COLORS[k.value] || "#e8edf6";
        return `<div class="kpi"><div class="label">${k.label}</div>
          <div class="value" style="color:${c};font-size:18px">${k.value}</div></div>`;
      }
      const cls = k.neutral ? "neutral" : signClass(k.value);
      return `<div class="kpi"><div class="label">${k.label}</div>
        <div class="value ${cls}">${fmt(k.value, k.d || 2)}${k.suffix || ""}</div></div>`;
    })
    .join("");

  // ---------- Section analysis callouts ----------
  const A = D.analysis || {};
  const anMap = {
    anRegimes: A.regimes, anVar: A.var, anBacktest: A.backtest, anModels: A.models,
    anVol: A.volatility, anDist: A.distribution, anRisk: A.risk, anSens: A.sensitivity,
    anStress: A.stress,
  };
  Object.entries(anMap).forEach(([id, text]) => {
    const el = $(id);
    if (el && text) {
      el.innerHTML = `<span class="analysis-tag">Analysis</span><p>${text}</p>`;
    }
  });

  // ---------- AI commentary ----------
  $("providerBadge").textContent = "🟢 " + D.commentary.provider;
  $("aiRegime").textContent = D.commentary.regime;
  $("aiVar").textContent = D.commentary.var;
  $("aiStress").textContent = D.commentary.stress;

  // ---------- Regime summary table ----------
  if (Array.isArray(D.regime.summary) && D.regime.summary.length) {
    const cols = Object.keys(D.regime.summary[0]);
    const head = "<tr>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr>";
    const body = D.regime.summary
      .map((row) => {
        return (
          "<tr>" +
          cols
            .map((c) => {
              const v = row[c];
              if (c === "Regime") {
                const col = REGIME_COLORS[v] || "#888";
                return `<td><span class="regime-pill" style="background:${col}22;color:${col}"><span class="dot" style="background:${col}"></span>${v}</span></td>`;
              }
              return `<td class="num">${typeof v === "number" ? v.toLocaleString() : v}</td>`;
            })
            .join("") +
          "</tr>"
        );
      })
      .join("");
    $("regimeTable").innerHTML = head + body;
  }

  // ---------- Stress table ----------
  const stressHead =
    "<tr><th>Scenario</th><th>Loss</th><th>P&amp;L ($1M)</th><th>Description</th></tr>";
  const stressBody = D.stress.table
    .map(
      (r) =>
        `<tr><td>${r.scenario}</td>` +
        `<td class="num ${r.loss < 0 ? "neg" : "pos"}">${fmt(r.loss)}%</td>` +
        `<td class="num ${r.pnl < 0 ? "neg" : "pos"}">$${Math.round(r.pnl).toLocaleString()}</td>` +
        `<td>${r.description}</td></tr>`
    )
    .join("");
  $("stressTable").innerHTML = stressHead + stressBody;

  // ---------- generic records table ----------
  function renderTable(elId, records, negCols) {
    const el = $(elId);
    if (!el || !Array.isArray(records) || !records.length) return;
    negCols = negCols || [];
    const cols = Object.keys(records[0]).filter((c) => c !== "index");
    const head = "<tr>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr>";
    const body = records
      .map((row) => {
        return (
          "<tr>" +
          cols
            .map((c) => {
              const v = row[c];
              if (c === "Regime" && REGIME_COLORS[v]) {
                const col = REGIME_COLORS[v];
                return `<td><span class="regime-pill" style="background:${col}22;color:${col}"><span class="dot" style="background:${col}"></span>${v}</span></td>`;
              }
              if (typeof v === "number") {
                let cls = "num";
                if (negCols.includes(c)) cls += v < 0 ? " neg" : " pos";
                return `<td class="${cls}">${v.toLocaleString()}</td>`;
              }
              return `<td>${v}</td>`;
            })
            .join("") +
          "</tr>"
        );
      })
      .join("");
    el.innerHTML = head + body;
  }

  const pass = (p) => (p > 0.05 ? "pos" : "neg");

  // ---------- VaR backtest cards ----------
  const bt = D.backtests || {};
  const h = bt.historical || {};
  const es = bt.es || {};
  const btCards = [
    { label: "Breach Rate (95%)", val: `${h.rate}%`, sub: `expected ${h.expected_rate}%`, cls: Math.abs((h.rate || 0) - (h.expected_rate || 5)) < 1.5 ? "pos" : "neg" },
    { label: "Kupiec POF p-value", val: h.p_value, sub: h.p_value > 0.05 ? "coverage OK" : "reject", cls: pass(h.p_value) },
    { label: "Christoffersen CC p", val: h.p_value_cc, sub: h.p_value_cc > 0.05 ? "no clustering" : "breaches cluster", cls: pass(h.p_value_cc) },
    { label: "ES Realized / Predicted", val: es.ratio, sub: `${es.n_breaches} breaches`, cls: es.ratio && Math.abs(es.ratio - 1) < 0.15 ? "pos" : "neutral" },
  ];
  $("btCards").innerHTML = btCards
    .map(
      (c) =>
        `<div class="kpi"><div class="label">${c.label}</div><div class="value ${c.cls}" style="font-size:22px">${c.val}</div><div class="label" style="margin:6px 0 0;text-transform:none">${c.sub}</div></div>`
    )
    .join("");

  // ---------- Backtest coverage table ----------
  const methodMap = { historical: "Historical", ewma: "EWMA", cornish_fisher: "Cornish-Fisher" };
  const btRows = Object.keys(methodMap)
    .filter((k) => bt[k])
    .map((k) => ({
      Method: methodMap[k],
      Breaches: bt[k].breaches,
      Expected: bt[k].expected,
      "Rate %": bt[k].rate,
      "Kupiec p": bt[k].p_value,
      "Christoffersen p": bt[k].p_value_cc,
    }));
  renderTable("btTable", btRows);

  // ---------- Forecast skill ----------
  const sk = D.forecast_skill;
  if (sk) {
    let line = `XGBoost OOS skill — RMSE <strong>${sk.rmse_model}</strong> vs naive ${sk.rmse_naive} · QLIKE <strong>${sk.qlike_model}</strong> vs ${sk.qlike_naive} (${sk.n_oos} days)`;
    if (sk.dm && isFinite(sk.dm.dm_stat)) {
      const sig = sk.dm.p_value < 0.05 ? "significant" : "not significant";
      line += ` · Diebold-Mariano ${sk.dm.dm_stat} (p=${sk.dm.p_value}, ${sig})`;
    }
    $("skillLine").innerHTML = line;
  }

  // ---------- VaR model league table ----------
  if (Array.isArray(D.var_league) && D.var_league.length) {
    const winner = D.var_league.find((r) => r.Verdict === "WINNER");
    if (winner) {
      $("leagueVerdict").innerHTML =
        `<span class="verdict-tag">Best model</span><strong>${winner.Method} VaR</strong> — lowest pinball loss (${winner["Pinball (bps)"]} bps) among methods passing Kupiec coverage.`;
    }
    const cols = ["Rank", "Method", "Rate %", "Kupiec p", "Christoffersen p", "Pinball (bps)"];
    const head = "<tr>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr>";
    const order = [...D.var_league].sort((a, b) => a.Rank - b.Rank);
    const rowsHtml = order
      .map((r) => {
        const win = r.Verdict === "WINNER";
        const cells = cols
          .map((c) => {
            const v = c === "Method" ? (r.Method != null ? r.Method : r.index) : r[c];
            if (c === "Method") return `<td>${v}${win ? ' <span class="win-pill">WINNER</span>' : ""}</td>`;
            if (c === "Kupiec p" || c === "Christoffersen p") {
              const cls = v > 0.05 ? "pos" : "neg";
              return `<td class="num ${cls}">${v}</td>`;
            }
            return `<td class="num">${v}</td>`;
          })
          .join("");
        return `<tr class="${win ? "win-row" : ""}">${cells}</tr>`;
      })
      .join("");
    $("leagueTable").innerHTML = head + rowsHtml;
  }

  // ---------- Distribution (Jarque-Bera) ----------
  const dist = D.distribution;
  if (dist) {
    const verdict = dist.normal ? "consistent with normality" : "strongly non-normal (fat-tailed)";
    $("jbLine").innerHTML = `Jarque-Bera = <strong>${dist.jb_stat.toLocaleString()}</strong> (p=${dist.jb_p}) → returns are ${verdict}.`;
  }

  // ---------- Diversification cards ----------
  const dv = D.diversification;
  if (dv) {
    const dvCards = [
      { label: "Diversification Ratio", val: dv.ratio, cls: dv.ratio > 1.2 ? "pos" : "neutral" },
      { label: "Effective Bets (risk)", val: dv.effective_bets_risk, cls: "neutral" },
      { label: "Effective Bets (capital)", val: dv.effective_bets_capital, cls: "neutral" },
      { label: "Risk Concentration (HHI)", val: dv.hhi_risk, cls: dv.hhi_risk > 0.25 ? "neg" : "pos" },
    ];
    $("divCards").innerHTML = dvCards
      .map(
        (c) =>
          `<div class="kpi"><div class="label">${c.label}</div><div class="value ${c.cls}">${c.val}</div></div>`
      )
      .join("");
  }

  // ---------- Risk attribution table (+ risk parity) ----------
  const rp = D.risk_parity || {};
  const riskRows = (D.risk_attribution || []).map((r) => ({
    Asset: r.index,
    "Weight %": r["Weight (%)"],
    "Risk Contrib %": r["Risk Contribution (%)"],
    "Risk-Parity Wt %": rp[r.index] != null ? +(rp[r.index] * 100).toFixed(2) : null,
  }));
  renderTable("riskTable", riskRows);

  // ---------- Regime forward table ----------
  renderTable("regimeFwdTable", D.regime.forward, ["Avg Fwd 21d (%)"]);

  // ---------- Stress replay table ----------
  renderTable("replayTable", D.stress.replay, ["Replay Loss (%)"]);

  // ---------- Reverse stress ----------
  const rv = D.stress.reverse || {};
  $("reverseBox").innerHTML = `
    <p class="reverse-q">What would it take to lose <strong>${Math.abs(rv.target_loss_pct)}%</strong> in a single day?</p>
    <div class="reverse-stat"><span class="reverse-num">${rv.daily_sigma_move}σ</span><span>one-day move required</span></div>
    <p class="muted small">A ${Math.abs(rv.daily_sigma_move)}-sigma daily shock is extraordinarily rare under normal assumptions — but fat tails make it far more likely than a Gaussian model implies.</p>`;

  // ---------- Plotly figures ----------
  const cfg = { responsive: true, displayModeBar: false };
  const draw = (id, fig) => {
    if (fig && document.getElementById(id)) {
      Plotly.newPlot(id, fig.data, fig.layout, cfg);
    }
  };
  const F = D.figures;
  const plotMap = {
    plotPortfolio: F.portfolio_value,
    plotInstruments: F.instruments,
    plotRegimeTimeline: F.regime_timeline,
    plotRegimeProb: F.regime_probabilities,
    plotRegimeFwd: F.regime_forward,
    plotVar: F.var_comparison,
    plotDrawdown: F.drawdown,
    plotBacktest: F.var_backtest,
    plotVol: F.volatility,
    plotForecast: F.forecast_skill,
    plotRollCorr: F.rolling_correlation,
    plotCorr: F.correlation,
    plotDist: F.return_dist,
    plotQQ: F.qq,
    plotRisk: F.risk_contributions,
    plotConstituent: F.constituent_scatter,
    plotRollSharpe: F.rolling_sharpe,
    plotMonthly: F.monthly_heatmap,
    plotStress: F.stress,
  };
  Object.entries(plotMap).forEach(([id, fig]) => draw(id, fig));

  // Re-layout charts on resize for crispness
  let t;
  window.addEventListener("resize", () => {
    clearTimeout(t);
    t = setTimeout(() => {
      Object.keys(plotMap).forEach((id) => {
        const el = document.getElementById(id);
        if (el && el.data) Plotly.Plots.resize(el);
      });
    }, 150);
  });
})();
