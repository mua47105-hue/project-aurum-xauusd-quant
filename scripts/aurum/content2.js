// content2.js - Chapters 5-9
const H = require("./helpers");
const { h1, h2, body, r, bullet, tbl, tr, figure } = H;
const DIR = "/home/z/my-project/scripts/aurum/";

// ===================== CH 5 AI TRADING EVIDENCE =====================
function ch5() {
  const out = [];
  out.push(h1("5. AI Trading Systems: What the Evidence Says Works and Fails"));

  out.push(h2("5.1 The flagship claims do not survive scrutiny"));
  out.push(body("The most-cited LLM trading framework, TradingAgents (arXiv 2412.20138, Tauric Research; actively maintained through v0.7.1 in October 2026), simulates a trading firm with analyst, bull-bear debate, trader and risk agents. Its headline result - 26.62% cumulative return (30.5% annualized) on the best of five tech mega-caps - was measured over a three-month bull-market window (January to March 2024), on five stocks, with no transaction costs in the paper. More telling is the maintenance history: throughout 2026 the maintainers fixed look-ahead and point-in-time bugs across FRED macro data, social sentiment, decision memory and Alpha Vantage feeds, and the README now states plainly that the framework is a research scaffold, \u201Cnot a strategy with a fixed, replicable return,\u201D and that two runs of the same ticker and date can differ. The pattern generalizes. FinRL's own ecosystem papers concede the field's three killers - low signal-to-noise, survivorship bias and overfitting - and an AI4Finance paper states that DRL results which \u201Coptimistically reported increased profits in backtesting may suffer the false-positive issue due to overfitting.\u201D Contamination-free benchmarks deliver the verdict: on StockBench (arXiv 2510.02209), which scores sequential multi-month trading, most LLMs fail to beat buy-and-hold; FINSABER (arXiv 2505.07078) shows previously reported LLM advantages deteriorate across broader cross-sections and longer horizons, with LLM strategies underperforming in bull markets and losing heavily in bears."));

  out.push(h2("5.2 The evaluation protocol we adopt: Alpha Illusion, P1-P6"));
  out.push(body("The single most useful document found for this project is \u201CThe Alpha Illusion\u201D (arXiv 2605.16895, May 2026), which audits end-to-end LLM trading claims and prescribes a six-point screening protocol. Its verdict - \u201Creported alpha from end-to-end LLM trading agents should not be treated as deployment evidence\u201D - comes with a constructive alternative that matches our architecture: LLMs should enter as auditable information interfaces upstream of independent calibration, risk and execution modules, which satisfies the protocol by construction once final decision authority is moved out of the LLM. We adopt the protocol verbatim as our evaluation checklist."));
  out.push(...tbl({
    capText: "Table 4: The Alpha Illusion P1-P6 screening protocol, adopted for AURUM",
    headers: ["Gate", "Requirement", "How AURUM satisfies it"],
    widths: [16, 42, 42],
    small: true,
    rows: [
      ["P1. Temporal integrity", "No time-travel: training and features must not leak future information, including pretraining knowledge", "Returns-based features only; strict embargoes; from-scratch ablation and pre-2020 control runs (Chapter 10)"],
      ["P2. Dynamic universe", "No survivorship or static-universe bias", "Single instrument (XAUUSD) with full bear regimes in test; no delisting issue"],
      ["P3. Counterfactual robustness", "Performance must not flip under reversed evidence; no hidden style tilt", "Reverse-evidence direction-flip tests; per-regime reporting; regime-conditional calibration"],
      ["P4. Epistemic calibration", "Language confidence is not tradable probability; verify ECE and reliability per regime", "Calibration loss in training; monthly temperature-scaling refits; reliability monitoring per regime"],
      ["P5. Realistic implementation", "Gross-alpha illusion: token, latency, spread, swap and slippage costs must be netted", "Full cost model in backtest (Chapter 8); per-decision cost ~0.001 cents at 0.6B scale"],
      ["P6. Multi-agent disaggregation", "Consensus among correlated agents is not independent confirmation", "Single model, single decision policy; no echo chambers; deterministic execution"],
    ],
  }));

  out.push(h2("5.3 Foundation models for time series are not a free lunch"));
  out.push(body("Zero-shot time-series foundation models - the obvious alternative backbone - have documented blind spots. Chronos-2 is the most downloaded time-series model on Hugging Face (27.1M monthly downloads), yet SimpleTimeBench (arXiv 2610.02058, October 2026) shows Chronos-2, Moirai 2.0 and Toto frequently produce suboptimal forecasts even for trivial monotonic trends, periodic signals and leading-indicator covariates. A 30-dataset deployment study (arXiv 2607.04919) finds foundation models beat classical methods (naive, ETS, ARIMA, XGBoost) on only 15 of 30 datasets, with classical methods winning on some datasets using 2% of the data. And ablation work on LLM-based forecasters (arXiv 2406.16964) shows removing the language model or replacing it with a basic attention layer does not degrade performance. The tabular evidence is equally blunt: tree ensembles still beat deep models on medium-sized numeric tabular data (Grinsztajn et al., arXiv 2207.08815; McElfresh et al., arXiv 2305.02997). The consequence is a hard requirement, not a preference: AURUM's looped transformer must beat LightGBM/XGBoost and a small GRU trained on identical features under identical cross-validation, or it does not ship. The model earns its place only if the looped architecture's iterative computation adds measurable decision value over cheaper alternatives."));

  out.push(h2("5.4 The one differentiator that keeps appearing: online learning"));
  out.push(body("Across verified competition winners (detailed in Chapter 7), the recurring \u201Csecret sauce\u201D is not architecture but adaptation: updating model weights after deployment, at low learning rates, on newly arrived data. The second-place solution of the 2024-25 Jane Street competition derived nearly a third of its final score from intraday online gradient updates on top of a frozen 17-model ensemble; the solo winner of Optiver's 2023-24 competition applied online learning to every component of a CatBoost-plus-GRU-plus-Transformer blend. A static model decays; markets drift. AURUM's plan therefore includes a weekly online-learning stage on Kaggle with strict purging of overlap between the update window and the evaluation window. This is also where the looped architecture may pay a second dividend: loop-count and iteration-divergence signals give unusually cheap drift detectors for deciding when recalibration or retraining is needed."));
  return out;
}

// ===================== CH 6 INDIVIDUALS =====================
function ch6() {
  const out = [];
  out.push(h1("6. The Best Results Ever Achieved by Individuals"));
  out.push(body("The user's first research priority was the best ever verified from individual builders. The verified record, across four independent arenas, is remarkably consistent - and consistently modest."));

  out.push(h2("6.1 Kaggle finance competitions"));
  out.push(body("The Jane Street Real-Time Market Data Forecasting competition (October 2024 to July 2025; 25,703 registrations, 2,781 teams) is the largest recent test of individual skill on financial prediction. The private leaderboard's top ten is dominated by single-person teams (ms capital, first at 0.013890 weighted R-squared; Patrick Yam, second; hyd, fifth). Note the magnitude: the winning edge is an R-squared of 1.4% - the best individuals on earth, on a clean feature set, extract a sliver of signal. The verified second-place method is a 17-model neural ensemble with online learning: a reproduction by a third party measured the frozen ensemble at 0.01413 and the online-updated version at 0.01942, i.e., +0.005 absolute - a third of total performance - from post-deployment adaptation. The verified eighth-place solution used GRU sequence models and reported explicitly that \u201CMLP, time-series transformers, cross-symbol attention and embeddings didn't work for me.\u201D In Optiver's Trading at the Close (2023-24), the solo winner \u201Chyd\u201D - a repeat winner, also fifth at Jane Street - blended CatBoost (0.5), GRU (0.3) and Transformer (0.2) with online learning and post-processing. The historical Jane Street 2021 winner was a team, but its core was a supervised autoencoder-plus-MLP trained with a 31-gap purged 5-fold time-series split - validation hygiene, not exotic architecture, was the differentiator."));

  out.push(h2("6.2 Numerai, and the honest professionals"));
  out.push(body("Numerai's staking leaderboard is the largest set of verified, on-platform, real-money track records for individuals in quant. As of 6 November 2026: 568,149 NMR staked across 679 staked participants, with an average one-year return of +19.97%. The top of the all-time board: bpa at 52.1% one-year on 6,185 NMR staked; derrida at 47.2% on 6,417 NMR. These are real and verified - but the edge belongs to Numerai's metamodel; individual skill expresses itself as model selection and staking strategy, and self-selection means the average hides the dead. The cleanest professional benchmark remains Rob Carver, the ex-AHL manager who publishes his own live systematic results: 2014-2023 at a mean +14.1% per year on 24.3% annualized volatility - a live Sharpe ratio of 0.58 - against the SG CTA index at 0.38 over the same window, including a -8.7% year in 2022-23. That is what a world-class retail systematic trader with 114 instruments, a 50-year backtest and decades of experience actually achieves."));

  out.push(h2("6.3 The failure statistics"));
  out.push(body("The base rates are harsh and well documented. Regulator data: the FCA's own analysis found 82% of retail CFD clients lose money (average loss 2,200 pounds per client per year); France's AMF reports 89% losing (average 10,887 euros); OANDA's own website footer currently discloses 76.6%. Day-trading studies: fewer than 1% of Taiwanese day traders are persistently profitable net of fees (Barber, Lee, Liu and Odean, Journal of Financial Markets 2014), and 97% of Brazilian futures day traders who persisted past 300 days lost money (Chague, De-Losso and Giovannetti, SSRN 3423101). One 2025 paper documents the exact failure mode this project must avoid: a 500-agent crypto trading system that validated at 300%+ APY and then lost more than 70% of live capital (\u201CThe Red Queen's Trap,\u201D arXiv 2512.15732)."));
  out.push(...tbl({
    capText: "Table 5: Verified individual performance ceilings (net of costs)",
    headers: ["Arena", "Best verified individual result", "Sharpe / edge", "Confidence"],
    widths: [26, 40, 20, 14],
    small: true,
    rows: [
      ["Jane Street 2024-25 (Kaggle)", "Solo and near-solo teams fill the top 10; win = ensemble + online learning", "R-squared 0.014", "HIGH (leaderboard + writeups)"],
      ["Optiver TAC 2023-24 (Kaggle)", "Solo winner \u201Chyd\u201D: CatBoost+GRU+Transformer blend + online updates", "RMSPE 5.40 private", "HIGH (writeup)"],
      ["Numerai staking", "bpa: +52.1% one-year on 6,185 NMR staked; 679 staked average +19.97%", "n/a (payouts)", "HIGH (on-platform, live)"],
      ["Systematic futures (blog, live)", "Rob Carver 2014-2023: +14.1%/yr at 24.3% vol incl. -8.7% year", "SR 0.58 live", "HIGH (fetched)"],
      ["Day trading (studies)", "Under 1% persistently profitable (Taiwan); 97% of persistent Brazilian day traders lose", "n/a", "HIGH (published)"],
      ["XAUUSD-only systematic track record", "None found, by any individual or lab", "n/a", "HIGH (absence documented)"],
    ],
  }));
  out.push(body("The last row is the strategic summary: nobody, anywhere, publishes a verified live track record for a systematic XAUUSD-only strategy. There is no proof it cannot be done - only proof that if it is done, it will be done for the first time. The engineering conclusion is to set the success bar at a net-of-cost Sharpe of 0.5 to 1.0 (Carver-class), plan for 20%+ drawdowns, treat anything above SR 1.0 in backtest as a multiple-testing artifact until the Deflated Sharpe Ratio says otherwise (Chapter 10), and let the system prove itself in months of paper trading before a single dollar is risked."));
  return out;
}

// ===================== CH 7 GOLD 2026 =====================
function ch7() {
  const out = [];
  out.push(h1("7. Market Reality: Gold in November 2026"));

  out.push(h2("7.1 Where the market stands"));
  out.push(body("Gold closed this research day around $4,053-4,058 (spot $4,053.50 at 15:22 UTC, 6 November 2026; COMEX December futures $4,054). That number sits 27.4% below the all-time high of $5,586.20 printed on 29 January 2026, after a violent correction to $3,879.30 on 30 October - a max drawdown of 26.7% peak-to-trough that is still resolving. Realized volatility is 29.5% annualized in 2026, against 20.9% in 2025 and 15.3% in 2024. The average daily range is $80-120 (2.0-2.6% of price), and 2026 year-to-date stands at -6.5%. Demand-side, the World Gold Council's Q3 2026 report shows total demand of 1,369 tonnes (-1% year-on-year) but a record value of $188 billion, with central banks buying 321 tonnes in the quarter (+42%, the fastest since late 2024, 675 tonnes year-to-date) and ETF holdings at a record 4,256 tonnes after +211 tonnes of quarterly inflows. In short: the training and test data for this project contain a once-in-decades parabolic rally and a live crash - the regime stress test most trading research never gets."));
  out.push(...figure(DIR + "fig1_gold_milestones.png", "Figure 1: Verified gold price milestones, 2001-2026. Milestone-to-milestone path through verified COMEX/spot data points (research thread E); shaded band marks the 2026 drawdown regime in which the project begins."));

  out.push(h2("7.2 The full regime map"));
  out.push(...tbl({
    capText: "Table 6: Gold regimes 2001-2026 (verified price data, research threads D and E)",
    headers: ["Period", "Price path", "Regime character and driver"],
    widths: [16, 30, 54],
    small: true,
    rows: [
      ["2001-2011", "$255 to $1,912 (Sep 2011)", "Secular bull: USD bear, QE, collapsing real yields, crisis hedging"],
      ["2013", "$1,695 to $1,183 (-28%)", "Taper tantrum; ~500t ETF liquidation"],
      ["2014-2018", "$1,046 (Dec 2015) to $1,375 range", "Low-vol, range-bound grind; rising real yields, strong USD"],
      ["2018-2020", "$1,161 to $2,089 (Aug 2020)", "Fed pivot plus COVID: real yields below -1%, QE, ETF inflows"],
      ["2021-2022", "$1,614-2,079 chop", "Hiking cycle; Ukraine-invasion spike $2,079 (Mar 2022), USD spike low $1,614"],
      ["2023", "$1,810-2,063 range", "Range with geopolitical bid (Oct 2023)"],
      ["2024", "Breakout: $2,234 (Mar) to $2,802 (Oct)", "Regime change begins: central-bank buying above 1,000t/yr, de-dollarization"],
      ["2025", "$2,838 (Jan) to $4,398 (Oct)", "Accelerating mega-rally; vol steps up 15% to 21%"],
      ["2026", "ATH $5,586 (Jan 29), crash to $3,879 (Oct 30), now ~$4,053", "Parabolic blow-off then -27% correction; realized vol 29.5%; record CB and ETF holdings"],
    ],
  }));
  out.push(body("The machine-learning consequence is symmetric and brutal: a model trained on 2013-2018 learns a mean-reverting, range-bound asset and would be destroyed by 2024-2026; a model trained through 2025 learns that volatility ratchets and price only rises, and would be destroyed by the -27% drawdown that followed. Both failure modes are inside our data. Chapter 10 therefore mandates regime-blocked training splits and per-regime reporting: a strategy that only works in trending bull regimes is leveraged beta, not alpha."));

  out.push(h2("7.3 Costs, sessions and events - the numbers the backtest must charge"));
  out.push(...tbl({
    capText: "Table 7: XAUUSD microstructure facts (2025-2026, research thread D)",
    headers: ["Item", "Value", "Implication"],
    widths: [26, 36, 38],
    small: true,
    rows: [
      ["All-in round-turn cost, raw/ECN account", "$0.15-0.35 per oz (spread $0.05-0.25 plus ~$6-7/lot commission)", "About 1-3 basis points of price; negligible for multi-hour holds, fatal for scalping"],
      ["All-in cost, standard account", "$0.30-0.60 per oz (spread only)", "Use $0.30-0.35 as the base case in backtests; stress at $0.60"],
      ["Swap (rollover), long positions", "About -$0.35/oz per night (-$30 to -$45 per lot); triple on Wednesday", "A 3-day long hold costs ~$1/oz (~2.5bp) - must be modeled as explicit overnight cashflow"],
      ["Average daily range (2026)", "$80-120 (2.0-2.6%); ATR14 ~ $85", "Even 0.25x-ADR targets are ~100x the round-turn cost"],
      ["Session ranges (avg $ per hour, UTC)", "Asia $14.8, London $15.4, NY $17.7; peak 13:00-14:00 at $24.3 (0.67%)", "Peak = US data releases plus COMEX open; event-time features beat session dummies"],
      ["Weekend gaps (10y, 451 Mondays)", "Mean 0.32%, median 0.19%, p95 1.09%, max 3.52%", "Stops do not guarantee fills; Friday-risk rules and gap-aware risk budgeting required"],
      ["Event calendar", "FOMC statements 19:00 UTC (8/yr); NFP and CPI 13:30 UTC", "Free calendars: federalreserve.gov (verified), BLS schedule pages, FRED release API"],
      ["Retail leverage caps (EU/UK)", "Gold 20:1 (5% margin); 0.01-lot minimum margin ~$202 at $4,000 gold", "Minimum viable live account ~$500-2,000; ~$7,500 for strict 1% risk with $75 stops"],
    ],
  }));
  return out;
}

// ===================== CH 8 COMPONENTS =====================
function ch8() {
  const out = [];
  out.push(h1("8. Component Selection: Base Model, Data and Backtesting Stack"));

  out.push(h2("8.1 Base model"));
  out.push(...tbl({
    capText: "Table 8: Base model candidates at or under ~2B parameters (November 2026; parameter counts and licenses from Hugging Face configs, tokenizer behavior tested empirically)",
    headers: ["Model", "Params", "License", "Context", "Notes"],
    widths: [22, 11, 16, 13, 38],
    small: true,
    rows: [
      ["Qwen3-0.6B(-Base) - recommended", "0.75B", "Apache-2.0", "32k", "21.5M downloads/mo; digit-by-digit tokenizer (verified); LoopUS-proven family; best throughput per GPU-hour"],
      ["Qwen3.5-0.8B(-Base) - alternate", "0.87B", "Apache-2.0", "262k", "March 2026 release; AA Intelligence Index 9 vs 6.5 for Qwen3-0.6B (MEDIUM); 2-KV-head GQA gives tiny KV cache"],
      ["Qwen3-1.7B(-Base)", "2.03B", "Apache-2.0", "32k", "The exact backbone LoopUS converted; 2.5-3x slower than 0.6B on T4 - reserve as fallback"],
      ["Llama-3.2-1B", "1.24B", "Llama 3.2 (gated)", "131k", "3-digit-group tokenizer (verified) - worse for numeric features; license friction"],
      ["SmolLM2-1.7B", "1.71B", "Apache-2.0", "8k", "Digit tokenizer but weaker benchmarks and short context"],
      ["OLMo-2-1B", "1.49B", "Apache-2.0", "4k", "Fully open data; 3-digit groups; 4k context too short"],
      ["Gemma-3-1b-it", "1.00B", "Gemma terms", "32k", "Restrictive terms; fine tokenizer; instruction-tuned only"],
      ["Pythia-1.4b", "1.51B", "Apache-2.0", "2k", "2023-era quality; arbitrary digit merges - excluded"],
    ],
  }));
  out.push(body("Recommendation: Qwen3-0.6B-Base as the primary backbone, with Qwen3.5-0.8B-Base as the alternate if its Unsloth support matures. The decisive factors are empirical: Qwen tokenizers split numbers digit-by-digit (verified by direct tokenizer execution), which is the encoding LLMTime (arXiv 2310.07820) found materially better for numeric work than grouped-digit schemes; the Qwen3 family is the backbone LoopUS actually converted; and at 0.75B parameters it maximizes looped-training tokens per Kaggle GPU-hour (Chapter 9). Numeric features will enter as Chronos-style binned tokens (arXiv 2403.07815) - each feature scaled and quantized to a small integer vocabulary - keeping sequences short and robust; exact scaled-digit encoding remains an ablation option."));

  out.push(h2("8.2 Data"));
  out.push(...tbl({
    capText: "Table 9: Free data sources for XAUUSD and correlated series (verified 6 November 2026)",
    headers: ["Source", "XAUUSD coverage", "Granularity", "Role in the stack"],
    widths: [20, 28, 20, 32],
    small: true,
    rows: [
      ["Dukascopy (via dukascopy-node v1.51.1)", "From Jun 1999 (per tool table) / 2003 (official) - present", "Tick, M1-H4, D1", "PRIMARY: ~600-800 MB of M1 for 20 years, uploaded as a private Kaggle dataset"],
      ["HistData.com", "March 2009 - present", "M1 + tick (bid)", "Cross-check venue; note EST timezone in ASCII files"],
      ["Yahoo Finance (yfinance 1.8.0)", "GC=F futures only - XAUUSD=X is now DELISTED; 1h limited to last 730 days, 15m to 60 days, 1m to 8 days/request", "1m-1mo", "Recent-intraday features + cross-check; EURUSD=X, SI=F, DX-Y.NYB verified working at 1h"],
      ["FRED (free)", "GOLDPMGBD228NLBM (LBMA PM fix, 1968-); DGS10, VIXCLS, DCOILWTICO, DEXUSEU, DTWEXBGS, SP500", "Daily", "Macro and correlated daily features (US10Y, VIX, WTI, EURUSD, DXY, SPX)"],
      ["OANDA v20 practice (free)", "XAU_USD; 5,000 candles per request, paginated", "S5 to M", "Forward/paper trading environment and live feature parity"],
      ["Federal Reserve / BLS calendars", "FOMC dates (verified server-rendered HTML through 2027); NFP/CPI at 13:30 UTC", "Events", "Event filters and post-event volatility features"],
    ],
  }));
  out.push(body("Two data warnings carry engineering weight. First, Yahoo's spot symbol XAUUSD=X is delisted - any notebook or dataset built on it is silently broken - and COMEX futures (GC=F) are not back-adjusted, so their quarterly roll gaps contaminate long-horizon studies; futures data is used only for the recent-730-day feature panel, never for the primary series. Second, Dukascopy pre-2003 XAUUSD tick density is uncertain, so the primary panel starts conservatively at 2003 with a bar-density audit before any earlier data is admitted."));

  out.push(h2("8.3 Backtesting stack"));
  out.push(body("The research loop uses vectorbt 1.2.0 (released 19 October 2026; actively maintained): vectorized execution over ~100k H1 bars in seconds, parameter sweeps, and clean composition with quantstats 0.0.86 and the statistics tooling below. NautilusTrader 1.231.0 provides the final realism pass - true bid/ask fills from Dukascopy bid and ask series, latency and partial-fill models - before any paper-trading claim is made. Swap is modeled as an explicit overnight cashflow (-$0.35/oz/night long, triple Wednesday) and walk-forward as a simple outer loop. backtesting.py 0.6.6 is a acceptable minimal fallback; zipline-reloaded (equity-daily paradigm) and freqtrade (crypto venues) are wrong tools for gold. For the statistical layer: mlfinlab is dead (commercial since 2023), replaced by purgedcv (PyPI, updated 4 November 2026) for purged/embargoed and combinatorial CV, and ml4t-diagnostic for Deflated Sharpe Ratio, probability of backtest overfitting and White's Reality Check. One caution from the deflated-sharpe package's own case study: a grid search of 19,200 configurations on BTC 1-hour walk-forward produced many in-sample Sharpe 1.5+ strategies and the DSR rejected every one; the binding constraint was trade count, not search method."));
  return out;
}

// ===================== CH 9 KAGGLE FEASIBILITY =====================
function ch9() {
  const out = [];
  out.push(h1("9. Free-Stack Training Feasibility on Kaggle"));

  out.push(h2("9.1 What the free tier actually provides (November 2026)"));
  out.push(body("Kaggle retired the P100 on 15 September 2026: the free GPU offering is now a single machine shape, 2 x Tesla T4 (15.4 GB each), with 4 CPU cores and 29 GB RAM. The weekly quota is 30 GPU-hours from a shared pool - one hour on both T4s counts as one hour - resetting Saturdays at 00:00 UTC, with sessions capped at 12 hours, a 20-minute interactive idle timeout, 20 GB of persistent output in /kaggle/working, and internet access (phone-verified accounts). A separate TPU v3-8 pool offers roughly 20 additional hours per week; adapting the looped architecture to torch/xla is unproven and treated as upside, not plan. Two T4s with Accelerate DDP deliver a measured 1.8-2.4x speedup on sub-2B fine-tunes in public benchmarks. FlashAttention-2 does not run on Turing GPUs; PyTorch SDPA (used automatically by Unsloth, which officially supports T4's CUDA capability 7.0) is the attention backend."));

  out.push(h2("9.2 Throughput math, and the one big adjustment"));
  out.push(body("Published Unsloth benchmarks on a free-tier T4 show a 1.1B model at 4-bit QLoRA training 3.87x faster than plain Hugging Face (-73.8% VRAM); scaling to our sizes gives working estimates of 3,000-6,000 tokens/second for 0.6B and 1,500-2,500 for 1.7B, roughly doubling with both T4s. A 30-hour GPU-week therefore buys roughly 600M-1.2B tokens of standard QLoRA training on the 0.6B backbone. Looped training multiplies per-token cost: LoopUS at B=20 loop steps with K=5 supervised depths costs an estimated 4-8x per token (derived from the paper's mechanics - the single biggest uncertainty in this chapter). Figure 2 shows the resulting feasibility picture: the paper's exact recipe (3B tokens on a 1.7B backbone) would take 23-81 weeks of free quota and is out of reach; a scaled recipe on the 0.6B backbone at B=8, K=3 fits 160-475M tokens per week."));
  out.push(...figure(DIR + "fig2_kaggle_feasibility.png", "Figure 2: Tokens trainable per Kaggle GPU-week by scenario (midpoints with low-high ranges; Unsloth-relative throughput estimates, research thread C). LoopUS's 3B-token language-retention budget is out of free-tier reach, but AURUM's ~0.5-1B token budget is not."));
  out.push(body("The crucial adjustment: LoopUS needed 3B tokens because its goal was to add looping while retaining general language capability on web text. AURUM has no such obligation - the model only ever needs to read binned market-state features and score fixed actions. The conversion can therefore be trained directly on task data: hundreds of millions of tokens of market states (each hour of 2003-2026 history yields many overlapping training examples) plus a small generic replay to stabilize the loop. At 0.5-1B tokens on the 0.6B backbone, the scaled conversion fits inside 2-4 GPU-weeks with margin - provided a 30-minute throughput smoke test on Kaggle, run as Phase 0 of the build plan, confirms the estimated looped tokens/second before any multi-week commitment."));
  out.push(...tbl({
    capText: "Table 10: Kaggle free-tier facts and their engineering consequences",
    headers: ["Constraint", "Value (Nov 2026)", "Engineering consequence"],
    widths: [26, 26, 48],
    small: true,
    rows: [
      ["GPU shape", "2 x T4 16GB (P100 retired Sep 15, 2026)", "All VRAM math done for 15.4 GB; no FA2 - SDPA attention; fp16 (no bf16) on T4"],
      ["Weekly quota", "30 GPU-hours, shared pool, resets Sat 00:00 UTC", "Checkpoint every N steps to /kaggle/working and design all runs resumable - a crash must never cost a week"],
      ["Session cap", "12 hours; 20-min interactive idle timeout", "Long jobs via CLI push / Save & Run All in ~10.5h segments with resume"],
      ["RAM / CPU", "29 GB / 4 cores", "Pre-tokenize datasets once; avoid CPU-bound data loading with 2 DDP workers"],
      ["Persistent output", "20 GB /kaggle/working", "LoRA adapters (~100-500 MB) and final 0.6B fp16 checkpoint (~1.2 GB) fit comfortably; keep 2-3 checkpoints max"],
      ["DDP speedup", "1.8-2.4x measured on sub-2B models", "Training script as .py launched via accelerate, not in-notebook (CUDA fork issue)"],
      ["Inference latency", "0.6B looped (B=8) on T4: ~2-15 s per decision; CPU int4: 7-40 s (borderline)", "Hourly decision cadence needs under 60 s - T4 comfortably, CPU only as fallback"],
    ],
  }));
  return out;
}

module.exports = { ch5, ch6, ch7, ch8, ch9 };
