// content3.js - Chapters 10-13
const H = require("./helpers");
const { h1, h2, body, r, bullet, tbl, tr, figure } = H;
const DIR = "/home/z/my-project/scripts/aurum/";

// ===================== CH 10 VALIDATION & RISK =====================
function ch10() {
  const out = [];
  out.push(h1("10. Validation Protocol and Risk Management Framework"));

  out.push(h2("10.1 The statistics that kill backtests"));
  out.push(body("Every quantitative claim in this section follows Bailey and Lopez de Prado's framework (The Deflated Sharpe Ratio, Journal of Portfolio Management 2014; The Probability of Backtest Overfitting, Journal of Computational Finance 2016). Three numbers anchor the design. First, minimum track length: for a strategy trading 250 times per year, an annualized Sharpe of 0.5 needs about 2,707 trades (roughly 10.8 years) before the observed Sharpe is statistically greater than zero at 95% confidence; even an excellent SR 1.0 needs 678 trades (~2.7 years). Second, the multiple-testing trap, shown in Figure 3: after trying 100 configurations on 1,500 trades, pure noise is expected to produce a best backtest Sharpe near 1.0. The 19,200-configuration case study in Chapter 8 is this trap in the wild - every Sharpe 1.5+ in-sample result was rejected by the Deflated Sharpe Ratio. Third, effective sample size: fifteen years of H1 bars is ~87,000 bars, but a 24-hour-horizon label leaves only ~3,600 effectively independent decision points, and the binding unit is trades, not bars. The protocol consequences are non-negotiable: combinatorial purged cross-validation with embargo (purgedcv) instead of a single walk-forward path; every trial counted - features, horizons, hyperparameters, losses, prompts included; Deflated Sharpe Ratio at or above 0.95 and probability of backtest overfitting below 0.5 as release gates; and pre-registration of the final configuration before the test window is touched."));
  out.push(...figure(DIR + "fig3_multiple_testing.png", "Figure 3: The multiple-testing trap. Expected maximum annualized Sharpe produced by pure luck as a function of configurations tried (1,500 trades assumed). After ~100 configurations, noise alone yields a backtest Sharpe near 1.0 - the reason every trial must be counted and the Deflated Sharpe Ratio applied."));
  out.push(...tbl({
    capText: "Table 11: Minimum trades required for statistical significance (95%, one-sided)",
    headers: ["Annualized Sharpe", "Min. trades (MinTRL)", "Years at 250 trades/yr"],
    widths: [34, 33, 33],
    align: ["c", "c", "c"],
    rows: [
      ["0.5 (good professional level)", "2,707", "~10.8"],
      ["1.0 (excellent; rarely verified live)", "678", "~2.7"],
      ["1.5 (rare; usually overfit)", "302", "~1.2"],
    ],
  }));

  out.push(h2("10.2 Data splits, contamination controls and calibration"));
  out.push(body("Data is split by regime blocks, never randomly: training on 2003-2018 (two full bull-bear cycles), validation on 2019-2022, and a locked test on 2023-2026 which contains both the mega-rally and the -27% crash, with embargoes at every boundary. Contamination from the pretrained backbone is treated as a first-class experimental variable, because a 2024-2025-pretrained model has literally read about gold's rally: all features are returns-based or binned (never raw price levels, which would date-stamp the observation); a from-scratch, randomly-initialized control model of identical architecture runs on identical data - if the pretrained init beats it specifically in the 2020-2026 window, that is leakage evidence; a pre-2020-only training run serves as the leakage-free control; and a zero-shot probe quantifies what the base model already knows about gold price history. Calibration is monitored, not assumed - RLHF demonstrably destroyed GPT-4's calibration (ECE 0.007 pre-training vs 0.074 post-RLHF, from the GPT-4 technical report), and the 17-day Jev experiment measured confidence running backwards in the field. The deployed protocol: expected calibration error and reliability diagrams per regime bucket, monthly temperature-scaling refits on a rolling window, and a pre-registered confidence-ablation test - confidence-weighted sizing must beat flat sizing on the same signals, or the confidence channel is noise and is switched off. Sizing follows the meta-labeling pattern (Lopez de Prado, Advances in Financial Machine Learning): the model proposes direction and probability; a secondary calibrated probability sizes the trade; miscalibrated probabilities are miscalibrated leverage."));

  out.push(h2("10.3 Risk management: sized for the edge we can prove"));
  out.push(...tbl({
    capText: "Table 12: Risk framework for AURUM (rules of the deterministic execution layer)",
    headers: ["Rule", "Setting", "Basis"],
    widths: [26, 38, 36],
    small: true,
    rows: [
      ["Risk per trade", "0.5-1.0% of equity (hard cap 1%)", "Fixed-fractional standard; below quarter-Kelly for p=0.55-class edges (full Kelly would be 10%)"],
      ["Position granularity check", "0.01 lot (1 oz); at $4,000 gold a $75 stop risks $75 = 1.5% of a $5,000 account", "Minimum position can exceed the risk budget on small accounts - account size must fit stop width"],
      ["Volatility targeting", "Scale gross exposure to ~15% annualized target", "Gold realized 29.5% in 2026 vs 15.3% in 2024 - fixed size would have tripled risk"],
      ["Stops", "ATR-based (1.5-3x ATR14, ~$128-255 at current vol)", "Adapts to the 2x vol regime shift; fixed-dollar stops fail across regimes"],
      ["Drawdown circuit breaker", "At -15% from equity high: flat, recalibrate on paper, human review to re-arm", "The real killer of retail systems is abandonment at -20%; pre-commit the shutdown rule"],
      ["Event and gap rules", "No new entries in the 30 minutes around FOMC/NFP/CPI (13:30/19:00 UTC); reduce or flat into event weekends; Friday risk budget halved", "Weekend gap p95 1.09%, max 3.52%; stops do not guarantee fills across gaps"],
      ["Leverage and account", "EU/UK 20:1 on gold; micro-lot live minimum ~$202 margin per 0.01 lot; realistic live account $2,000-7,500", "FCA PS19/18 margin floors; keeps 1%-risk budget consistent with stop width"],
      ["Go-live sequence", "CPCV backtest, then DSR/PBO gates, then 3-6 months and 100+ trades on OANDA practice with live calibration monitoring, then micro-lots at 0.25% risk; scale only after 300+ live trades with Sharpe CI excluding zero", "Every documented failure in Chapter 6 skipped at least one of these steps"],
    ],
  }));
  return out;
}

// ===================== CH 11 LIMITATIONS =====================
function ch11() {
  const out = [];
  out.push(h1("11. Master Limitations Table and Mitigations"));
  out.push(body("The user asked explicitly for the limitations of the plan, found before building. Twelve survive the research, each paired with its mitigation and the residual risk that remains after the mitigation. Items 5, 9 and 12 are the ones most likely to end the project in its original form, and they are the reason the build plan gates every phase."));
  out.push(...tbl({
    capText: "Table 13: Limitations, mitigations and residual risk",
    headers: ["#", "Limitation (with evidence)", "Mitigation (built into the plan)", "Residual risk"],
    widths: [5, 33, 36, 26],
    small: true,
    rows: [
      ["1", "Backtest overfitting: ~100 tried configs produce a lucky SR ~1.0 at achievable sample sizes (Ch. 10)", "CPCV + embargo; count every trial; DSR >= 0.95 and PBO < 0.5 release gates; pre-registration", "A single-instrument strategy may never gather 700+ clean post-selection trades - the honest edge may be unprovable"],
      ["2", "Regime shift: 2024-26 rally and -27% crash both in-window; models trained on any sub-period learn the wrong lesson (Ch. 7)", "Regime-blocked splits; per-regime reporting; vol targeting; weekly online learning (the verified winners' differentiator)", "The next regime is, by definition, not in the data"],
      ["3", "Pretraining contamination: the base model has read about 2024-26 gold (Ch. 10)", "Returns-based binned features only; from-scratch control; pre-2020 control run; zero-shot knowledge probe", "Leakage through subtle priors cannot be fully excluded"],
      ["4", "Calibration inversion: Jev's field confidence correlated -0.17 with outcomes; RLHF wrecked GPT-4 calibration (Ch. 3, 10)", "Calibration-penalized training objective; per-regime ECE monitoring; monthly refits; confidence-ablation kill switch", "Calibration under live regime shift may still decay between refits"],
      ["5", "GBDTs may simply win: trees beat deep models on medium numeric tabular data; LLM backbones added nothing in TS ablations (Ch. 5)", "Mandatory LightGBM/XGBoost and GRU baselines under identical CPCV; the looped model ships only if it beats them", "High: the honest outcome may be that the looped transformer loses - then the baseline system ships instead"],
      ["6", "Kaggle quota fragility: 30 GPU-h/week, 12h sessions, T4-only since Sep 2026 (Ch. 9)", "Resumable runs, checkpointing, CLI-pushed batch segments, pre-tokenized data; TPU pool as unproven upside; Colab fallback", "A multi-week conversion run disrupted mid-way costs calendar weeks, not just GPU-hours"],
      ["7", "Free data quality: roll gaps in futures, delisted Yahoo spot symbol, thin pre-2003 Dukascopy ticks, EST-timezone HistData files (Ch. 8)", "Dukascopy M1 primary from 2003; HistData cross-venue audit; bar-density checks; UTC normalization", "Silent gaps or bad ticks in any year could bias features; audits catch most, not all"],
      ["8", "Execution realism: spread widens at events; weekend gaps (p95 1.09%, max 3.52%); swap drag ~2.5bp for 3-day holds (Ch. 7)", "Full cost model in vectorbt; NautilusTrader bid/ask realism pass; event and Friday rules; swap as explicit cashflow", "Live slippage during vol spikes can exceed any modeled spread"],
      ["9", "Small effective sample: ~3,600 independent H1 decision points and 50-200 trades/yr at swing cadence (Ch. 10)", "20+ years of M1 data; multi-horizon experiment matrix (M15/H1/H4/D1) to find the best trade-count/edge balance", "High: at 100 trades/yr, a true SR 0.5 needs a decade to verify - paper trading is the only honest intermediate evidence"],
      ["10", "Single instrument: no diversification; gold-only systems have no verified public precedent (Ch. 6)", "Phase 1 is deliberately single-instrument for research cleanliness; architecture keeps a portfolio dimension for later FX/metal expansion", "Idiosyncratic gold regime risk is fully concentrated"],
      ["11", "Loop-specific fragilities: depth extrapolation degrades; retention collapse at unseen loop counts; naive quantization breaks looped models (Ch. 4)", "Train and serve at consistent depths; random loop-position supervision (99.8% retention in the source study); loop-aware quantization or none", "Looped-model serving is a young systems literature; T4 deployment may need engineering patience"],
      ["12", "No precedent, no proof: no looped-transformer trading model exists; nobody publishes a verified XAUUSD-only track record (Ch. 4, 6)", "Treat the project as research with a strict publication-grade evaluation protocol; baselines ship value even if the looped model loses; no live capital before the full gauntlet", "The idea itself may not work - which is exactly what the gates are for"],
    ],
  }));
  return out;
}

// ===================== CH 12 ARCHITECTURE & PLAN =====================
function ch12() {
  const out = [];
  out.push(h1("12. Recommended Architecture and Phased Build Plan"));

  out.push(h2("12.1 The AURUM-0.6B system"));
  out.push(body("AURUM (Latin for gold) is a System One decision model in the Jev mold, built open and self-hosted. Data plane: Dukascopy XAUUSD M1 bid/ask from 2003, resampled to H1 and D1; features are returns-based, volatility-normalized and Chronos-binned into the Qwen tokenizer's vocabulary - trend and momentum percentiles, ATR-relative stretch, session and event flags, funding-style proxies from futures basis, and correlated daily series (DXY, US10Y, VIX, WTI, EURUSD, silver, SPX from FRED and Yahoo). No raw price levels ever enter the state, and no free text enters at all, which closes Jev's prompt-injection failure mode by construction."));
  out.push(body("Model: Qwen3-0.6B-Base receives LoopUS-style surgery - encoder = first layers, a middle-block loop group behind a softplus selective gate, decoder = final layers - configured at B=8 loop steps with K=3 randomly supervised depths for memory-efficient training, plus a confidence head for early exit and the KL-between-successive-loops divergence as a free uncertainty signal. Heads are parallel and typed, not autoregressive: a direction choice (long / short / stand-aside), a conviction noul, a target-distance score, and a regime-noul; option symmetry is enforced architecturally (a single scalar per direction rather than ordered options) so the option-order bias measured at 26% in the Jev experiment cannot arise. Training runs in three stages: Stage A converts the backbone into a stable looped model on mixed data (~0.5-1B tokens: market states plus a small generic replay); Stage B trains the decision heads with an RLCD-style objective - proper scoring rules (Brier/log-loss) plus an explicit calibration-penalty term, class-balanced across regimes; Stage C is weekly online learning at low learning rate on the newest data with strict purging - the one differentiator that every verified individual winner shared. All training on Kaggle 2 x T4 via Accelerate DDP with resumable checkpoints; the model judges, and the deterministic execution layer of Table 12 decides and executes."));
  out.push(body("The evaluation gauntlet is the point of the whole system: mandatory baselines (buy-and-hold, time-series momentum, LightGBM, a small GRU) under identical combinatorial purged CV; the P1-P6 checklist; DSR at or above 0.95 and PBO below 0.5; full-cost vectorbt backtest and a NautilusTrader realism pass; then an OANDA practice phase with live calibration monitoring. The model ships only if it beats every baseline net of costs under the same splits - and if it does not, the baseline system ships and the looped model remains the research track."));

  out.push(h2("12.2 The seven phases"));
  out.push(...tbl({
    capText: "Table 14: Build plan - phases, deliverables and GO/NO-GO gates (Kaggle GPU-weeks are estimates from Chapter 9)",
    headers: ["Phase", "Weeks", "Deliverables", "GO/NO-GO gate"],
    widths: [22, 10, 38, 30],
    small: true,
    rows: [
      ["P0. Smoke tests", "1 (0.5 GPU-wk)", "Kaggle T4x2 throughput benchmark of the looped forward/backward pass; Dukascopy download test; resume-from-checkpoint test", "Measured looped throughput confirms >= 150M tokens/week; resume works after kill"],
      ["P1. Data infrastructure", "1-2", "Dukascopy M1 2003-2026 (~700 MB) via dukascopy-node; UTC-normalized H1/D1 panel; feature pipeline; HistData cross-audit; private Kaggle dataset; event calendar table", "Bar-density audit passes; cross-venue divergence quantified; dataset versioned on Kaggle"],
      ["P2. Baseline gauntlet", "2-3", "CPCV + DSR/PBO tooling (purgedcv, ml4t-diagnostic); buy-and-hold, TSM, LightGBM, GRU baselines on identical features", "Validation harness reproduces baseline scores; the bar to beat is registered and frozen"],
      ["P3. Looped conversion", "3-6 (2-4 GPU-wk)", "LoopUS surgery on Qwen3-0.6B-Base; Stage A conversion training with checkpointing; convergence and loop-stability diagnostics", "Looped model matches or beats the GRU baseline on validation proxy tasks at B=8; no divergence at trained depth"],
      ["P4. Decision training + calibration", "6-8 (1-2 GPU-wk)", "Stage B decision-head training; per-regime reliability diagrams; temperature scaling; confidence-ablation A/B", "ECE < 0.05 in every major regime bucket; calibrated probability beats LightGBM on the decision metric"],
      ["P5. Validation gauntlet", "8-10", "Full CPCV + DSR + PBO; full-cost vectorbt backtest; NautilusTrader realism pass; contamination ablations (from-scratch and pre-2020 controls)", "DSR >= 0.95; PBO < 0.5; >= 700 post-selection trades; net-of-cost Sharpe beats all baselines; P1-P6 checklist green"],
      ["P6. Paper trading", "10-16+", "OANDA practice deployment (hourly cadence); weekly online updates; live calibration and drift monitoring; trade log", ">= 100 paper trades; net paper Sharpe > 0; ECE stable; only then discuss micro-lot live at 0.25% risk"],
    ],
  }));

  out.push(h2("12.3 What success looks like - honestly"));
  out.push(body("Success is not a money printer; it is a system whose edge is either proven or refuted with publication-grade statistics, entirely on free infrastructure. The base case after 16 weeks: a validated backtest meeting the DSR/PBO gates at a net-of-cost Sharpe between 0.3 and 0.8, in paper trading, with calibration holding per regime. The stretch case: the looped transformer beats LightGBM decisively and the loop-depth signal demonstrably encodes regime difficulty - a first-of-kind result worth publishing either way. The failure case, which the gates make cheap: the looped model loses to gradient boosting, and the project still delivers a rigorously validated gold strategy stack plus a reusable looped-transformer training pipeline on Kaggle. Capital is not part of the success criteria; Chapter 6's evidence says that is the only honest ordering."));
  return out;
}

// ===================== CH 13 SOURCES =====================
function ch13() {
  const out = [];
  out.push(h1("13. Key Sources and Confidence Notes"));
  out.push(body("Sources are grouped by load; arXiv identifiers are given where a paper is the origin of a claim. Confidence follows the convention of Chapter 2: HIGH marks primary sources read in full during this research; MEDIUM marks secondary or multi-source agreement; LOW marks single unverified claims, all of which are listed in 13.5. The DeerFlow methodology reference is the repository cloned at the start of this research (github.com/bytedance/deer-flow, version 2.0, MIT license)."));

  out.push(h2("13.1 Architecture (looped transformers)"));
  out.push(bullet("Geiping et al., Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach, arXiv 2502.05171; HF: tomg-group-umd/huginn-0125 (Apache-2.0); code: github.com/seal-rg/recurrent-pretraining"));
  out.push(bullet("Park et al., LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models, arXiv 2605.11011; code: github.com/thrillcrazyer/LoopUS; HF: Thrillcrazyer/Qwen3_1.7B_LoopUS"));
  out.push(bullet("McLeish et al., Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence, arXiv 2511.07384; code: github.com/mcleish7/retrofitting-recurrence"));
  out.push(bullet("Zhu et al., Scaling Latent Reasoning via Looped Language Models (Ouro), arXiv 2510.25741; HF: ByteDance/Ouro-1.4B (Apache-2.0)"));
  out.push(bullet("ARLoop, arXiv 2611.05234; HF: osunlp/ARLoop-0.6B, ARLoop-1.3B. Skeptical and mechanistic: arXiv 2507.02199 (latent CoT critique), 2506.18233 (looping buys computation, not knowledge), 2610.19768 (loop-retention), 2610.40325 (how to loop), 2611.06206 (BabyLM data-budget result), ICLR 2027 OpenReview vJgG20sjea (weight tying and gradient pooling)"));
  out.push(bullet("Structured/time-series and RL evidence: arXiv 2605.11262 (latent CoT on structured data), 2609.36108 (LoopICL), 2610.41819 (ElTab), 2610.19552 (Test-Time Forecasting), 2609.37432 (Looped Actor), 2310.07820 (LLMTime), 2403.07815 (Chronos), 2310.01728 (Time-LLM)"));

  out.push(h2("13.2 Jev, System One models and AI-trading evidence"));
  out.push(bullet("TypeSafe AI: typesafe.ai blog, Introducing System One Models and Jev (15 Sep 2026); Jev 1.13 failure-modes documentation; LangChain integration post (langchain.com/blog/building-a-harness-with-jev); Wikipedia, Jev (AI model); TechCrunch coverage (18 Sep 2026)"));
  out.push(bullet("Verified by direct full-text read: the 17-day Jev trading experiment, Towards AI (19 Oct 2026); Miles Deutscher, The Most Powerful AI Trading Bot Possible (X article, 29 Sep 2026). Competitors: OpenAI Decisions (DevDay, 6 Oct 2026), Cloudflare Clef (open weights, Apache-2.0), Perplexity Decisions; community trackers madewithjev.com"));
  out.push(bullet("AI-trading evidence: TradingAgents arXiv 2412.20138 + repo history; Alpha Illusion arXiv 2605.16895; StockBench arXiv 2510.02209; FINSABER arXiv 2505.07078; TSFM blind spots arXiv 2610.02058 and 2607.04919; LLM-for-TS ablations arXiv 2406.16964; tabular GBDT evidence arXiv 2207.08815 and 2305.02997; Lopez-Lira and Tang arXiv 2304.07619; Red Queen's Trap arXiv 2512.15732"));

  out.push(h2("13.3 Individuals, statistics and risk"));
  out.push(bullet("Kaggle: Jane Street 2024-25 final leaderboard and discussion (2nd-place reproduction: kaggle.com/discussion/742687, github.com/cweill/jane-street-causal-forecasting); Optiver TAC 1st-place writeup (hyd); Jane Street 2021 1st-place writeup. Numerai leaderboard (numer.ai/leaderboard/data-scientists, live 6 Nov 2026). Rob Carver live performance 2014-2023 (qoppac.blogspot.com, fetched). Day-trader studies: Barber et al. (JFM 2014); Chague et al. (SSRN 3423101). Retail loss statistics: FCA CP16/40; AMF; OANDA site disclosure"));
  out.push(bullet("Validation and risk methodology: Bailey and Lopez de Prado (DSR, SSRN 2460551; PBO, Journal of Computational Finance 2016; PSR, SSRN 1821643); Advances in Financial Machine Learning (2018); Guo et al. arXiv 1706.04599 (calibration); GPT-4 Technical Report arXiv 2303.08774 (RLHF calibration loss). Tooling: purgedcv, ml4t-diagnostic, deflated-sharpe, pypbo, vectorbt 1.2.0, quantstats 0.0.86, NautilusTrader 1.231.0"));

  out.push(h2("13.4 Market data and infrastructure"));
  out.push(bullet("Gold market: World Gold Council Gold Demand Trends Q3 2026 (gold.org); COMEX GC=F daily/1h data (computed 6 Nov 2026); gold-api.com spot quote. Data sources: dukascopy-node v1.51.1 (dukascopy-node.app/instrument/xauusd), histdata.com coverage page, yfinance 1.8.0 (live-tested intraday limits; XAUUSD=X delisting), FRED series DGS10 / VIXCLS / DCOILWTICO / DEXUSEU / DTWEXBGS / SP500 / GOLDPMGBD228NLBM, federalreserve.gov FOMC calendar (verified), BLS release schedules, OANDA v20 practice API docs"));
  out.push(bullet("Kaggle and training: kaggle.com/docs (GPU quota), Kaggle staff discussion 361104 (T4x2 transition, Sep 2026), Unsloth requirements and HF T4 benchmarks (huggingface.co/blog/unsloth-trl), flash-attention README (Turing limitations), DDP benchmarks (github.com/nmquan22/multi-gpu-accelerate-vs-lightning)"));

  out.push(h2("13.5 Low-confidence items (flagged, not relied upon)"));
  out.push(bullet("Reddit r/hermesagent JEV trading-stack post: network-blocked during research; carried as context only. Jev's internal architecture and RLCD details: unpublished; the Brier-loss characterization is Wikipedia-editor inference (MEDIUM-LOW)."));
  out.push(bullet("Stooq long-history coverage and OANDA practice-account candle depth for XAUUSD: not verifiable from this environment; both are audited in Phase 1 before reliance. Exact 2026 broker swap rates: derived from rate math, to be confirmed against the target broker's table before live trading."));
  out.push(bullet("Reported TypeSafe AI late-2026 funding round (MEDIUM): from secondary press coverage only. Qwen3.5-0.8B quality figures: from the Artificial Analysis small-models writeup (MEDIUM); its Unsloth 4-bit fine-tuning support is expected but was not directly verified - one of Phase 0's smoke tests."));
  return out;
}

module.exports = { ch10, ch11, ch12, ch13 };
