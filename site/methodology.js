// VORTEX — methodology modal. Clicking any method card opens a popup with the
// full explanation and mathematics. Cards are matched to content by title slug,
// so no per-card markup changes are required.

(function () {
  "use strict";

  const slug = (s) =>
    s
      .toLowerCase()
      .replace(/\bvortex\b/g, "")
      .replace(/\bsignature\b/g, "")
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "");

  const F = (x) => `<div class="formula">${x}</div>`;
  const H = (t) => `<h4>${t}</h4>`;

  // ── Content ────────────────────────────────────────────────────────────────
  const METHODS = {
    "returns": {
      tag: "Foundation",
      title: "Returns & Portfolio Aggregation",
      html:
        H("Intuition") +
        "<p>A return is the percentage change in price. VORTEX uses <strong>log-returns</strong> because they are <em>time-additive</em>: the log-return over many days is simply the sum of daily log-returns, which makes statistics (means, volatility scaling) clean.</p>" +
        H("The mathematics") +
        F("r<sub>t</sub> = ln( P<sub>t</sub> / P<sub>t−1</sub> ) = ln(1 + R<sub>t</sub>)") +
        "<p>where R<sub>t</sub> is the ordinary (simple) return. Over a horizon, log-returns add: ln(P<sub>T</sub>/P<sub>0</sub>) = Σ<sub>t</sub> r<sub>t</sub>.</p>" +
        H("Portfolio aggregation (done right)") +
        "<p>A weighted sum of <em>log</em>-returns is only an approximation. A portfolio's return must be combined in <strong>simple-return space</strong>, then converted back:</p>" +
        F("R<sub>p,t</sub> = Σ<sub>i</sub> w<sub>i</sub> ( e<sup>r<sub>i,t</sub></sup> − 1 ) &nbsp;;&nbsp; r<sub>p,t</sub> = ln(1 + R<sub>p,t</sub>)") +
        "<p class='modal-note'>Constant weights imply <strong>daily rebalancing</strong>. VORTEX computes returns only on dates where every holding traded, avoiding calendar-mismatch artifacts (e.g. crypto trades weekends, equities don't).</p>",
    },

    "regime-detection-gaussian-hmm": {
      tag: "Regime Detection",
      title: "Gaussian Hidden Markov Model",
      html:
        H("Intuition") +
        "<p>The market's 'environment' is never observed directly. A Hidden Markov Model infers an unobservable <strong>state</strong> s<sub>t</sub> from observable evidence x<sub>t</sub>, and models how states transition over time. VORTEX fits 4 states on the feature vector [ daily return, 5-day realized volatility ].</p>" +
        H("The model") +
        "<p>Three parameter sets are learned: initial probabilities π, the transition matrix A, and Gaussian emissions per state.</p>" +
        F("Transition: &nbsp; P(s<sub>t</sub> = j | s<sub>t−1</sub> = i) = A<sub>ij</sub>") +
        F("Emission: &nbsp; x<sub>t</sub> | s<sub>t</sub> = s &nbsp;~&nbsp; 𝒩( μ<sub>s</sub>, Σ<sub>s</sub> )") +
        F("Joint likelihood: &nbsp; P(x<sub>1:T</sub>, s<sub>1:T</sub>) = π<sub>s<sub>1</sub></sub> · Π<sub>t</sub> A<sub>s<sub>t−1</sub>s<sub>t</sub></sub> · b<sub>s<sub>t</sub></sub>(x<sub>t</sub>)") +
        H("Estimation") +
        "<p>Parameters are fit by the <strong>Baum-Welch</strong> algorithm (Expectation-Maximization) which maximizes the marginal likelihood P(x<sub>1:T</sub>). The most-likely state path is recovered with the <strong>Viterbi</strong> algorithm.</p>" +
        "<p class='modal-note'>States carry no inherent meaning, so VORTEX labels them post-hoc by their mean return and volatility into Low-Volatility, Crisis, Recovery and Inflation Shock.</p>",
    },

    "historical-var": {
      tag: "Value-at-Risk",
      title: "Historical VaR",
      html:
        H("Intuition") +
        "<p>The most assumption-light VaR: just look at what actually happened. Sort the trailing window of returns and read off the loss that was only exceeded (1−α) of the time.</p>" +
        H("The mathematics") +
        F("VaR<sub>α</sub> = Quantile<sub>(1−α)</sub>( { r<sub>t−w+1</sub>, … , r<sub>t</sub> } )") +
        "<p>With α = 95% and a window w = 252 days, VaR is the 5th-percentile return — the threshold the worst ~1 day in 20 falls below.</p>" +
        H("Trade-offs") +
        "<p><strong>Pros:</strong> no distributional assumption; captures the real shape of the tail. <strong>Cons:</strong> assumes the future resembles the window, and reacts slowly (a calm window underestimates risk just before a shock).</p>",
    },

    "parametric-var": {
      tag: "Value-at-Risk",
      title: "Parametric (Gaussian) VaR",
      html:
        H("Intuition") +
        "<p>Assume returns are normally distributed, then the entire tail is described by just the mean and standard deviation — fast and smooth.</p>" +
        H("The mathematics") +
        F("VaR<sub>α</sub> = μ + z<sub>(1−α)</sub> · σ") +
        "<p>where z<sub>(1−α)</sub> is the standard-normal inverse CDF: z<sub>0.05</sub> ≈ −1.645 (95%), z<sub>0.01</sub> ≈ −2.326 (99%). μ and σ are estimated on a rolling window.</p>" +
        H("The catch") +
        "<p>Markets are <strong>not</strong> Gaussian — they have fat tails and negative skew. Parametric VaR therefore systematically <em>understates</em> the probability of extreme losses, which is exactly why VORTEX also computes Cornish-Fisher and Historical VaR.</p>",
    },

    "ewma-var-riskmetrics": {
      tag: "Value-at-Risk",
      title: "EWMA VaR (RiskMetrics)",
      html:
        H("Intuition") +
        "<p>Recent market action matters more than the distant past. EWMA weights observations with exponential decay, so the volatility estimate (and the VaR) reacts quickly when turbulence arrives.</p>" +
        H("The mathematics") +
        F("σ<sub>t</sub><sup>2</sup> = λ · σ<sub>t−1</sub><sup>2</sup> + (1 − λ) · r<sub>t−1</sub><sup>2</sup>") +
        F("VaR<sub>α</sub> = z<sub>(1−α)</sub> · σ<sub>t</sub>") +
        "<p>RiskMetrics uses λ = 0.94 for daily data. The decay implies an effective memory of roughly 1/(1−λ) ≈ 16 days — recent squared-returns dominate.</p>" +
        "<p class='modal-note'>This is a recursive filter: every past return contributes, but with geometrically shrinking weight (1−λ)λ<sup>k</sup>.</p>",
    },

    "regime-adaptive-var": {
      tag: "Value-at-Risk · Signature",
      title: "Regime-Adaptive VaR",
      html:
        H("Intuition") +
        "<p>Risk is not constant — a 'normal' day in a Crisis regime is far more dangerous than in a calm one. This method conditions the empirical tail on the regime the market is currently in, so the VaR band widens in a Crisis and tightens when calm.</p>" +
        H("The mathematics — and why it's causal") +
        F("VaR<sub>α</sub><sup>(t)</sup> = Quantile<sub>(1−α)</sub>( r<sub>τ</sub> : τ &lt; t , regime<sub>τ</sub> = s<sub>t</sub> )") +
        "<p>Crucially, the quantile for day t uses <strong>only past</strong> returns (τ &lt; t) drawn from the same regime — never future data. Until a regime has accumulated at least 30 observations, it falls back to the expanding full-sample quantile.</p>" +
        H("Why this matters") +
        "<p>A naive implementation computes one quantile per regime over the <em>entire</em> sample and maps it back — that uses future information (look-ahead bias) and would never be achievable live. VORTEX's strictly point-in-time construction is what makes the backtest honest.</p>",
    },

    "cornish-fisher-var": {
      tag: "Value-at-Risk",
      title: "Cornish-Fisher (Modified) VaR",
      html:
        H("Intuition") +
        "<p>Keep the speed of parametric VaR but correct the Gaussian quantile for the <strong>actual skewness and fat tails</strong> of the data, using an Edgeworth/Cornish-Fisher expansion.</p>" +
        H("The mathematics") +
        F("z<sub>cf</sub> = z + (z²−1)·S/6 + (z³−3z)·K/24 − (2z³−5z)·S²/36") +
        F("VaR<sub>α</sub> = μ + z<sub>cf</sub> · σ") +
        "<p>where S is skewness and K is excess kurtosis (both rolling). When S = K = 0 the expansion collapses to z and it reduces exactly to parametric VaR.</p>" +
        "<p class='modal-note'>For negatively-skewed, fat-tailed assets (skew ≈ −0.7, excess kurtosis ≈ 13 in this portfolio) z<sub>cf</sub> becomes more extreme than z, pushing VaR deeper into the tail — a more honest estimate.</p>",
    },

    "conditional-var-expected-shortfall": {
      tag: "Tail Risk",
      title: "Conditional VaR / Expected Shortfall",
      html:
        H("Intuition") +
        "<p>VaR tells you the threshold of a bad day; it says nothing about <em>how bad</em> things get once you cross it. CVaR (a.k.a. Expected Shortfall) answers that — the average loss in the tail beyond VaR.</p>" +
        H("The mathematics") +
        F("CVaR<sub>α</sub> = E[ r | r ≤ VaR<sub>α</sub> ] = (1/(1−α)) ∫<sub>0</sub><sup>1−α</sup> VaR<sub>u</sub> du") +
        H("Why regulators prefer it") +
        "<p>CVaR is a <strong>coherent</strong> risk measure — in particular it is sub-additive (diversification never increases it), which VaR can violate. Basel III / FRTB shifted bank capital from VaR to Expected Shortfall for exactly this reason.</p>",
    },

    "volatility-forecasting": {
      tag: "Forecasting",
      title: "Volatility Forecasting",
      html:
        H("Intuition") +
        "<p>Measuring today's risk is necessary; forecasting tomorrow's is more useful. VORTEX uses three lenses, all annualized by √252.</p>" +
        H("The models") +
        "<p><strong>Rolling:</strong> the standard deviation of the last 21 returns. <strong>EWMA:</strong> the RiskMetrics decay filter. <strong>XGBoost:</strong> a gradient-boosted tree trained on lagged returns and lagged realized vols to predict forward realized volatility.</p>" +
        F("target<sub>t</sub> = std( r<sub>t+1 … t+h</sub> ) · √252") +
        H("Honest evaluation") +
        "<p>The XGBoost model is <strong>walk-forward</strong>: trained on an expanding window and refit periodically, so every prediction is genuinely out-of-sample. It is scored against a naive 'tomorrow = today's rolling vol' benchmark using two losses:</p>" +
        F("RMSE = √ mean( (forecast − realized)² )") +
        F("QLIKE = mean( σ²<sub>real</sub>/σ²<sub>pred</sub> − ln(σ²<sub>real</sub>/σ²<sub>pred</sub>) − 1 )") +
        "<p class='modal-note'>QLIKE is robust to noise in the volatility proxy and penalizes under-prediction of risk more heavily — the right asymmetry for risk management.</p>",
    },

    "sharpe-ratio": {
      tag: "Performance",
      title: "Sharpe Ratio",
      html:
        H("Intuition") +
        "<p>How much return do you earn per unit of risk (volatility) taken? Higher is better; it lets you compare strategies on a risk-adjusted basis.</p>" +
        H("The mathematics") +
        F("S = ( E[r] − r<sub>f</sub> ) / σ<sub>r</sub> · √252") +
        "<p>The √252 annualizes a daily ratio. VORTEX uses r<sub>f</sub> = 0 for simplicity. As a rough guide: &gt;1 is solid, &gt;2 excellent.</p>" +
        "<p class='modal-note'>Sharpe penalizes upside and downside volatility equally — a limitation the Sortino ratio addresses.</p>",
    },

    "maximum-drawdown": {
      tag: "Performance",
      title: "Maximum Drawdown",
      html:
        H("Intuition") +
        "<p>The single most emotionally relevant risk metric: the largest peak-to-trough fall in portfolio value. It measures the worst loss an investor would have had to endure before recovery.</p>" +
        H("The mathematics") +
        F("W<sub>t</sub> = Π<sub>τ≤t</sub> (1 + r<sub>τ</sub>) &nbsp;(cumulative wealth)") +
        F("MDD = min<sub>t</sub> ( W<sub>t</sub> / max<sub>τ≤t</sub> W<sub>τ</sub> − 1 )") +
        "<p>It is <strong>path-dependent</strong> — two portfolios with identical mean and volatility can have very different drawdowns depending on the ordering of returns.</p>",
    },

    "stress-testing": {
      tag: "Stress Testing",
      title: "Scenario Stress Testing",
      html:
        H("Intuition") +
        "<p>A fire drill for the portfolio: if a known historical catastrophe struck the current holdings, what would the loss be?</p>" +
        H("The mathematics") +
        F("L<sub>scenario</sub> = Σ<sub>i</sub> w<sub>i</sub> · δ<sub>i</sub>") +
        "<p>where δ<sub>i</sub> is the calibrated peak-to-trough shock for asset i in that scenario (e.g. SPY −56.5% in 2008). Shocks are asset-specific, so the model captures that gold may rise while equities collapse.</p>" +
        "<p class='modal-note'>VORTEX complements these static shocks with <em>historical path replay</em> (using the actual realized returns over each crisis window) and <em>reverse stress testing</em>.</p>",
    },

    "higher-moments": {
      tag: "Distribution",
      title: "Skewness & Kurtosis",
      html:
        H("Intuition") +
        "<p>The mean and volatility only describe a distribution if it's symmetric and thin-tailed. The 3rd and 4th moments reveal the asymmetry and tail-fatness that drive real-world risk.</p>" +
        H("The mathematics") +
        F("Skewness: γ<sub>1</sub> = E[(r−μ)<sup>3</sup>] / σ<sup>3</sup>") +
        F("Excess kurtosis: γ<sub>2</sub> = E[(r−μ)<sup>4</sup>] / σ<sup>4</sup> − 3") +
        "<p>A normal distribution has γ<sub>1</sub> = 0 and γ<sub>2</sub> = 0. <strong>Negative skew</strong> means crashes are larger than rallies; <strong>positive excess kurtosis</strong> ('fat tails') means extreme moves happen far more often than a bell curve predicts.</p>",
    },

    "sortino-ratio": {
      tag: "Performance",
      title: "Sortino Ratio",
      html:
        H("Intuition") +
        "<p>Investors don't mind upside volatility — only downside. Sortino is Sharpe's smarter cousin: it divides excess return by <strong>downside deviation</strong> alone.</p>" +
        H("The mathematics") +
        F("σ<sub>d</sub> = √ E[ min(r − MAR, 0)² ]") +
        F("Sortino = ( E[r] − MAR ) / σ<sub>d</sub> · √252") +
        "<p>with the minimum-acceptable-return MAR = 0. A Sortino meaningfully higher than the Sharpe indicates the volatility is mostly to the upside.</p>",
    },

    "var-backtesting": {
      tag: "Validation",
      title: "VaR & ES Backtesting",
      html:
        H("Intuition") +
        "<p>A VaR model is a claim ('losses exceed this only 5% of the time'). Backtesting checks whether reality agreed — the question every risk committee asks first.</p>" +
        H("Kupiec POF — unconditional coverage") +
        "<p>Does the observed breach rate match the promised (1−α)? With x breaches in n days:</p>" +
        F("LR<sub>POF</sub> = −2 ln[ (1−p)<sup>n−x</sup> p<sup>x</sup> / ((1−π̂)<sup>n−x</sup> π̂<sup>x</sup> ) ] ~ χ²<sub>1</sub>") +
        "<p>where p = 1−α and π̂ = x/n. A high p-value means the breach rate is statistically acceptable.</p>" +
        H("Christoffersen — independence") +
        "<p>Breaches should be spread out, not clustered. Using first-order Markov transition counts n<sub>ij</sub> between breach/no-breach states:</p>" +
        F("LR<sub>ind</sub> ~ χ²<sub>1</sub> &nbsp;;&nbsp; LR<sub>cc</sub> = LR<sub>POF</sub> + LR<sub>ind</sub> ~ χ²<sub>2</sub>") +
        H("Expected-Shortfall test") +
        "<p>Beyond frequency, are the breach <em>magnitudes</em> right? VORTEX compares the average realized loss on breach days to the predicted CVaR (ratio ≈ 1 is well-calibrated).</p>" +
        "<p class='modal-note'>In this portfolio the historical VaR passes Kupiec (≈5.7% vs 5%) but fails Christoffersen independence — an honest finding that breaches cluster during crises.</p>",
    },

    "filtered-vs-smoothed-regimes": {
      tag: "Regime Detection",
      title: "Filtered vs Smoothed Probabilities",
      html:
        H("Intuition") +
        "<p>How confident are we about today's regime? There are two answers — one honest for real-time use, one that secretly peeks at the future.</p>" +
        H("Filtered (causal — what VORTEX reports)") +
        "<p>The <strong>forward algorithm</strong> gives the probability of state s at time t using only data up to t:</p>" +
        F("α<sub>t</sub>(s) ∝ b<sub>s</sub>(x<sub>t</sub>) · Σ<sub>i</sub> α<sub>t−1</sub>(i) A<sub>is</sub> &nbsp;⇒&nbsp; P(s<sub>t</sub> | x<sub>1:t</sub>)") +
        H("Smoothed (uses the whole sample)") +
        "<p>The <strong>forward-backward</strong> algorithm conditions on <em>all</em> data, including the future:</p>" +
        F("γ<sub>t</sub>(s) = P(s<sub>t</sub> = s | x<sub>1:T</sub>) ∝ α<sub>t</sub>(s) · β<sub>t</sub>(s)") +
        "<p class='modal-note'>Smoothed ribbons look cleaner but overstate how confidently you'd have known the regime live. VORTEX's current-regime readout uses the filtered estimate to avoid this hindsight bias.</p>",
    },

    "risk-attribution": {
      tag: "Attribution",
      title: "Risk Attribution (Euler / Component VaR)",
      html:
        H("Intuition") +
        "<p>Capital weights lie about risk. A 14% allocation to a volatile asset can drive a far larger share of total portfolio risk. Euler decomposition reveals where risk truly comes from.</p>" +
        H("The mathematics") +
        F("σ<sub>p</sub> = √( wᵀ Σ w )") +
        F("Marginal: MCTR<sub>i</sub> = ∂σ<sub>p</sub>/∂w<sub>i</sub> = (Σw)<sub>i</sub> / σ<sub>p</sub>") +
        F("Component: CTR<sub>i</sub> = w<sub>i</sub> · MCTR<sub>i</sub>") +
        "<p>Because σ<sub>p</sub> is homogeneous of degree 1 in w, Euler's theorem guarantees the components <strong>sum exactly to total risk</strong>: Σ<sub>i</sub> CTR<sub>i</sub> = σ<sub>p</sub>. Dividing by σ<sub>p</sub> gives each asset's percent-of-risk.</p>" +
        "<p class='modal-note'>In the equal-weight portfolio, BTC carries ~14% of capital but ~39% of risk — exactly the kind of hidden concentration this surfaces.</p>",
    },

    "risk-parity-erc": {
      tag: "Attribution",
      title: "Risk Parity (Equal Risk Contribution)",
      html:
        H("Intuition") +
        "<p>Instead of equalizing <em>capital</em>, equalize <em>risk</em>: choose weights so every asset contributes the same share of total portfolio volatility. This avoids a few volatile assets dominating.</p>" +
        H("The objective") +
        F("Find w such that &nbsp; w<sub>i</sub>·(Σw)<sub>i</sub> = w<sub>j</sub>·(Σw)<sub>j</sub> &nbsp; ∀ i, j") +
        H("How VORTEX solves it") +
        "<p>A simple fixed-point iteration scales each weight toward its target risk share until contributions converge:</p>" +
        F("w<sub>i</sub> ← w<sub>i</sub> · √( target / CTR<sub>i</sub> ) , &nbsp; then renormalize") +
        "<p class='modal-note'>The result naturally tilts toward low-volatility, diversifying assets (here, bonds and gold) and away from high-vol ones (crypto).</p>",
    },

    "historical-replay-reverse-stress": {
      tag: "Stress Testing",
      title: "Historical Replay & Reverse Stress",
      html:
        H("Historical path replay") +
        "<p>Rather than a single hand-set shock, replay the <strong>actual</strong> daily return paths of each asset over a crisis window and compound them against current weights — capturing the real correlation/contagion structure of that episode.</p>" +
        F("L<sub>replay</sub> = Π<sub>t∈window</sub> (1 + Σ<sub>i</sub> w<sub>i</sub> R<sub>i,t</sub>) − 1") +
        "<p class='modal-note'>Assets that didn't exist in a window (e.g. Bitcoin before 2014) are dropped and the weights renormalized over the survivors, so 2008 is evaluated on the 6 assets that traded then.</p>" +
        H("Reverse stress testing") +
        "<p>Flip the question: instead of 'what's the loss in scenario X?', ask 'what move would it take to lose Y%?'.</p>" +
        F("σ-move = L<sub>target</sub> / σ<sub>daily</sub>") +
        "<p>Expressing the required shock in standard deviations exposes how (im)plausible a target loss is — and how fat tails make large σ-moves more likely than a Gaussian implies.</p>",
    },
  };

  // ── Modal wiring ─────────────────────────────────────────────────────────
  const overlay = document.getElementById("methodModal");
  const body = document.getElementById("modalBody");
  const closeBtn = document.getElementById("modalClose");
  if (!overlay || !body) return;

  let lastFocus = null;

  function open(content) {
    body.innerHTML =
      `<span class="modal-tag">${content.tag}</span>` +
      `<h3 class="modal-title">${content.title}</h3>` +
      content.html;
    overlay.classList.add("open");
    overlay.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
    body.scrollTop = 0;
    closeBtn.focus();
  }

  function close() {
    overlay.classList.remove("open");
    overlay.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
    if (lastFocus) lastFocus.focus();
  }

  closeBtn.addEventListener("click", close);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && overlay.classList.contains("open")) close();
  });

  document.querySelectorAll(".method-card, .card.numbered").forEach((card) => {
    const h3 = card.querySelector("h3");
    if (!h3) return;
    const key = slug(h3.textContent);
    const content = METHODS[key];
    if (!content) return;

    card.classList.add("clickable");
    card.setAttribute("tabindex", "0");
    card.setAttribute("role", "button");
    const hint = document.createElement("span");
    hint.className = "method-more";
    hint.textContent = "View full derivation →";
    card.appendChild(hint);

    const trigger = () => { lastFocus = card; open(content); };
    card.addEventListener("click", trigger);
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); trigger(); }
    });
  });
})();
