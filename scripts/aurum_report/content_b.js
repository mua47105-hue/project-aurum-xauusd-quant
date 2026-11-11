"use strict";
// Sections 8-14: Kaggle platform through References
const S = [];

// ══════════════════════════════ 8. KAGGLE PLATFORM ══════════════════════════════
S.push({ h1: "8. Kaggle as the Training and Automation Platform", blocks: [
  { t: "h2", x: "8.1 Verified Capabilities" },
  { t: "p", x: "During this research phase, the complete Kaggle API round-trip was verified live on the owner's account (tskdev): authentication with the provided access token; listing and pulling the two existing notebooks (both empty starter kernels); pushing a new private notebook with internet enabled; executing it on Kaggle infrastructure (status polled to COMPLETE in roughly one minute); and retrieving its output log programmatically. The probe also fingerprinted the Kaggle Python environment, which matters for dependency planning. No blocking issues were found; the only operational frictions were the absence of a pre-installed Kaggle CLI in the orchestration sandbox (trivially installed) and HTTP 429 rate limiting from Yahoo Finance inside the kernel (handled with retries and headers in the design)." },
  { t: "table",
    caption: "Table 6: Kaggle environment probe results (verified November 6, 2026)",
    headers: ["Probe", "Result"],
    widths: [45, 55],
    rows: [
      ["API authentication, kernels list/pull/push/status/output", "All functional end-to-end"],
      ["Python runtime", "3.13.18"],
      ["Pre-installed ML stack", "PyTorch 2.11.0 (CPU build in probe; GPU builds available on GPU kernels), pandas 2.3.3, NumPy 2.1.3, scikit-learn 1.6.1, XGBoost 3.4.1"],
      ["Outbound internet in kernels", "Enabled and reachable (Yahoo responded; rate-limited with HTTP 429)"],
      ["Existing notebooks on the account", "Two empty starter notebooks - the project starts from a clean slate"],
    ],
  },
  { t: "h2", x: "8.2 Quotas and Limits" },
  { t: "table",
    caption: "Table 7: Kaggle free-tier limits relevant to training operations",
    headers: ["Resource", "Limit", "Planning consequence"],
    widths: [26, 30, 44],
    rows: [
      ["GPU hours", "About 30 hours/week (floor; Kaggle states it may be higher depending on demand); weekly reset", "4-8 full AURUM training runs per week, plus ablations - the loop over r in {1,2,3} fits comfortably"],
      ["Session length", "12 hours maximum per interactive or committed run", "Every training kernel must checkpoint and be resumable; runs are chunked at 8-10 hours with warm restarts"],
      ["Concurrent GPU sessions", "1 interactive + 2 committed (background) GPU kernels", "Two training runs can execute in parallel from the API - the automation loop exploits this"],
      ["Working directory", "20 GB persisted via Save Version", "LoRA adapters (~100-200 MB each) plus tokenizer and config fit with wide margin; raw M1 archives (a few GB) live in attached datasets instead"],
      ["Datasets", "20 GB per dataset; private quota 20 GB total (public is free)", "The processed feature/label corpus (a few hundred MB Parquet) and model-checkpoint datasets both fit privately"],
      ["GPU types", "NVIDIA T4 x2 or P100 (16 GB each)", "Single-T4 LoRA training is the default; 2x T4 DDP is the reserve for the largest runs"],
    ],
  },
  { t: "h2", x: "8.3 Feasibility Math for a 0.45B Looped Trunk" },
  { t: "p", x: "Memory: the truncated trunk is roughly 470M parameters (about 314M in 20 non-embedding layers plus 155M tied embeddings); fp16 weights occupy about 0.95 GB. Trainable parameters - LoRA (rank 16) on the q/k/v/o and MLP projections of the last 12 layers, the injection adapter, and the scoring head - total roughly 12-15M (estimate); their AdamW states and fp32 master copies add well under 0.5 GB. With gradient checkpointing, activation memory at batch 8-16 and sequence lengths of 512-768 stays in the low single-digit GB. Total peak usage is approximately 3-5 GB against 16 GB on one T4 - comfortable headroom, and the reason 2x T4 is a reserve rather than a requirement. Throughput: at roughly 2 GFLOP per token for forward-plus-backward on a 0.35B-non-embedding model, a 120,000-example corpus at 2 epochs and average packed length near 600 tokens is of the order of 15,000-30,000 optimizer steps; on a single T4 at realistic efficiency this lands at roughly 3-7 GPU-hours per full run (estimate, to be calibrated in Phase 2). The weekly 30-hour budget therefore supports multiple full runs plus the smaller ablations that Phases 2-3 require." },
  { t: "h2", x: "8.4 The Automation Loop" },
  { t: "p", x: "The Bev walkthrough ran its training on a rented A100 with an autonomous agent monitoring checkpoints over SSH. Project AURUM replicates that loop with zero infrastructure cost, because the Kaggle API already exposes every primitive the agent used: the orchestrator (this assistant, on the owner's Kaggle token) generates a training notebook and pushes it; polls kernel status; pulls logs, metrics, and checkpoint artifacts from kernel output; analyzes results; and pushes the next iteration - hyperparameters, data-mixture updates, or code fixes - as a new kernel version. Checkpoint resume works through versioned Kaggle datasets: each training kernel saves its latest LoRA adapter and optimizer state to /working as output, and the orchestrator attaches the prior version as an input dataset for the next chunk. Data updates flow the same way, as versioned private datasets. Everything is scriptable today; nothing in this loop is speculative." },
  { t: "h2", x: "8.5 Fallback Ladder" },
  { t: "p", x: "If compute needs exceed the free tier (for example, a Qwen3-1.7B trunk or a large r sweep), the ladder is: (1) shift all CPU-bound work (data building, backtests, evaluation) to CPU kernels, which do not consume GPU quota; (2) use the two concurrent GPU commits to keep the queue full; (3) rent a single A100 by the hour on a spot platform (typical market rate roughly $1.0-1.5/hour at the time of writing; the entire Bev project - all iterations plus the final 24-hour run - cost $36 on Prime Intellect). The design keeps every artifact (datasets, adapters, eval suites) in Kaggle-portable form so a rented-GPU run is a drop-in replacement for the training kernel, not a new pipeline." },
]});

// ══════════════════════════════ 9. BACKTESTING ══════════════════════════════
S.push({ h1: "9. Backtesting and Evaluation Framework", blocks: [
  { t: "h2", x: "9.1 Validation Methodology" },
  { t: "p", x: "Financial labels overlap in time (one position's outcome window spans many other decision points), so ordinary k-fold cross-validation leaks. The framework uses walk-forward analysis as the outer loop - train 2000-2021, validate 2022-2023, test 2024-2026, then roll the windows forward monthly during operation - with purged k-fold cross-validation and an embargo inside the training window for hyperparameter selection, following Lopez de Prado. Every feature is computed strictly from bars that closed before the decision bar; execution is modeled at the next bar's open plus costs; and the test window is touched exactly once, at the end, by the frozen candidate model. No iteration after test-window results are seen - that rule is written into the roadmap's phase gates." },
  { t: "h2", x: "9.2 Cost Model" },
  { t: "note", x: "Planning assumption (to be calibrated against the owner's actual broker): XAUUSD all-in round-trip cost of $0.35 per ounce (typical raw-spread ECN pricing of $0.10-0.30 spread plus commission), slippage of $0.10 per ounce per side on stop orders, and overnight swap applied to positions held past 22:00 UTC. Backtests will be run at 1.0x, 1.5x, and 2.0x this cost assumption; a strategy whose edge dies at 1.5x assumed costs is treated as no edge." },
  { t: "h2", x: "9.3 Metrics and Gates" },
  { t: "table",
    caption: "Table 8: Core backtest metrics and Phase 4 gate thresholds",
    headers: ["Metric", "Definition", "Phase 4 gate (net of costs, OOS test window)"],
    widths: [24, 44, 32],
    rows: [
      ["Annualized Sharpe", "Mean daily return / std, annualized by sqrt(252)", "Above 1.0 (target 1.5 for paper-trading promotion)"],
      ["Max drawdown", "Peak-to-trough equity decline", "Below 20%"],
      ["Profit factor", "Gross profit / gross loss", "Above 1.25"],
      ["Expectancy", "Mean trade result in R multiples", "Above +0.15R"],
      ["Trade count", "Closed trades in test window", "At least 150 (statistical meaningfulness)"],
      ["Calibration", "Brier score and reliability curve on task-level probabilities", "Brier below 0.25 on T2 meta-gate; reliability slope 0.8-1.2"],
      ["Invariance", "Choice-order and paraphrase stability (Section 3.4)", "100% argmax stability under option permutation; probability shift under 0.02"],
    ],
  },
  { t: "h2", x: "9.4 Mandatory Baselines" },
  { t: "p", x: "The looped decision model is only credible if it beats, on the same test window and cost model: (1) buy-and-hold gold; (2) an EMA(50/200) trend-following rule; (3) a Donchian(20) breakout rule; (4) an XGBoost classifier on the identical tabular feature set with thresholded entries - the boring-ML control that isolates what the language-model trunk adds; and (5) the unlooped (r=1) version of itself - the ablation that isolates what looping adds. Baselines 4 and 5 are the two that can actually falsify the project's two core bets, and they are cheap: the XGBoost control trains in minutes on CPU kernels, and the r=1 ablation is the same checkpoint family." },
  { t: "h2", x: "9.5 Statistical Robustness" },
  { t: "p", x: "Three further guards against self-deception: deflated Sharpe ratio (correcting for the number of configurations tried - the multiple-testing problem that the automation loop makes worse by making experiments cheap); Monte Carlo bootstrap over trade sequences (5,000 resamples) to put a confidence interval on expectancy and drawdown; and regime slicing (bull/bear/chop years, high/low volatility halves, by-session) so that a headline Sharpe cannot hide a single-regime edge. Results tables will always show the bootstrap interval, never a point estimate alone." },
  { t: "h2", x: "9.6 Tooling" },
  { t: "p", x: "vectorbt (open source, NumPy/Numba-vectorized) runs parameter sweeps and baseline comparisons orders of magnitude faster than event-loop frameworks, which matters inside 12-hour kernels; a small custom event-driven harness executes the composed six-task policy (T1-T6 with the code-level risk wrapper) bar by bar, because the policy's conditional structure (gates, trailing logic, event guards) is not expressible as a pure vectorized signal. Both live in the repo as CPU kernels; neither consumes GPU quota." },
]});

// ══════════════════════════════ 10. SYSTEM DESIGN ══════════════════════════════
S.push({ h1: "10. Proposed System Design: AURUM Decider", blocks: [
  { t: "h2", x: "10.1 Architecture Overview" },
  { t: "p", x: "AURUM Decider is a 0.45B-class looped decision model. The trunk is Qwen3-0.6B truncated to 20 layers. Looping is applied Huginn-style but adapted for fine-tuning from a pretrained checkpoint: layers 1-12 form the prelude; layers 13-16 form the recurrent core, applied r times; layers 17-20 plus the scoring head form the coda. The packed input sequence is Question, then State, then Options (each with its own position-ID reset), then the answer prompt - the Bev ordering, which lets state tokens read the question first. Position IDs reset at every option boundary (each option's positions start at k+1, independent of its serial position), and the block-causal 4D float mask implements the visibility topology: options attend to question/state and themselves but never to each other; the answer prompt attends to everything. The mask and position IDs are computed once per example and reused identically on every loop iteration - looping mutates hidden states only." },
  { t: "p", x: "The injection adapter A projects the concatenation of the running state and the prelude output [s; e] from 2048 back to 1024 dimensions before each core pass. Following the risk analysis of Section 4.5, A is initialized so that iteration 1 approximately reproduces the unlooped trunk (identity-dominant initialization), which makes r=1 an exact baseline and lets the loop only improve on it during training. The core is deliberately blind to its iteration index. Training samples r from {1, 2, 3} (geometric weights 0.5/0.33/0.17); inference exposes r as a dial evaluated in {1, 2, 3, 4}." },
  { t: "h2", x: "10.2 Scoring Head and Calibration" },
  { t: "p", x: "The discarded language-modeling head is replaced by a two-layer self-attention scoring head over the extracted vectors: the final token embedding of each option plus the answer-prompt embedding, each augmented with a role embedding (option vs answer) and a task-type embedding (choice/noul/score, and the T1-T6 task identity). A linear projection maps each refined option vector to one scalar logit; unused slots (up to N_max=255) receive negative infinity before softmax. Following the Kev practice, a temperature is fitted post-hoc on a held-out calibration split per checkpoint and shipped with the model; calibration is re-verified monthly during operation (the calibration-drift risk in Section 12)." },
  { t: "h2", x: "10.3 Parameter Budget" },
  { t: "table",
    caption: "Table 9: AURUM Decider parameter budget (estimates; to be confirmed in Phase 2)",
    headers: ["Component", "Parameters", "Trainable", "Notes"],
    widths: [30, 18, 16, 36],
    rows: [
      ["Trunk layers 1-20 (Qwen3-0.6B, truncated)", "~314M", "No (frozen base)", "LoRA rank 16 on last 12 layers adds ~4-5M trainable low-rank weights"],
      ["Embeddings (tied, vocab 151,936 x 1024)", "~155M", "No", "Input embeddings only; the 2048-wide lm_head projection is discarded"],
      ["Injection adapter A (2048 to 1024)", "~2.1M", "Yes", "Identity-dominant initialization so r=1 reduces to baseline"],
      ["Scoring head (2 self-attention layers + roles + task types + linear)", "~5-8M", "Yes", "Includes task-type embeddings for T1-T6"],
      ["Total", "~0.47-0.48B", "~12-15M", "Trains on a single 16 GB T4 with checkpointing; ~3-7 GPU-hours per full run (estimate)"],
    ],
  },
  { t: "h2", x: "10.4 Training Recipe" },
  { t: "p", x: "Three stages, each with a go/no-go gate. Stage A (sanity, ~1 GPU-hour): freeze the trunk entirely, train only the head and adapter on a 20,000-example subset; success means the model beats random and class-prior baselines and passes the invariance suite - this isolates head-implementation bugs from everything else. Stage B (main, 3-7 GPU-hours): LoRA on the last 12 layers plus head and adapter, full 120,000-example corpus, cosine schedule, early stopping on the purged validation split; success means matching or beating the XGBoost tabular control on task-level accuracy and clearing the Brier gate. Stage C (optional): unfreeze the full trunk at low learning rate (8-bit optimizer, single T4) only if Stage B plateaus below its gate. The loop is present from Stage B onward; Stage A runs with r fixed at 1 to keep the sanity check minimal." },
  { t: "h2", x: "10.5 Dataset Budget and Mixture" },
  { t: "p", x: "Target corpus of roughly 120,000 examples: about 45% T1/T2 direction and gate tasks sampled from H1 decision points with triple-barrier and meta labels; 20% T3 exit-management tasks sampled from simulated open positions; 15% T4 stop-distance and 10% T5 size tasks; 10% T6 regime checks. Augmentation multiplies the base set: feature-perturbed counterfactual pairs (target 10% of corpus, label-flipped by construction) and 2-3 paraphrase templates per state. Class balance is enforced per task (no-action capped at 50% of T1 examples; T3 balanced across the four actions). The corpus, its generation code, and its version hash are published as a Kaggle dataset so every training run is exactly reproducible." },
]});

// ══════════════════════════════ 11. ROADMAP ══════════════════════════════
S.push({ h1: "11. Implementation Roadmap", blocks: [
  { t: "p", x: "The build is sequenced so that every phase produces a measurable artifact, consumes known compute, and ends at a gate that can stop the project cheaply if a core assumption fails. GPU-hour figures are planning estimates against the 30-hour weekly quota." },
  { t: "table",
    caption: "Table 10: Phased roadmap with artifacts, compute, and success gates",
    headers: ["Phase", "Deliverables (Kaggle artifacts)", "Est. GPU-hrs", "Success gate"],
    widths: [17, 38, 12, 33],
    rows: [
      ["1. Data pipeline (week 1)", "aurum-data-build CPU kernel; HistData ingestion + Dukascopy cross-check; multi-timeframe features; triple-barrier and meta labels; aurum-data-v1 dataset (~120k examples)", "0 (CPU)", "Zero look-ahead detected by automated audit; label distributions documented; Dukascopy sample comparison within tolerance"],
      ["2. Baseline decider (weeks 2-3)", "aurum-train-baseline kernel (r=1, Stages A/B); eval harness with invariance + calibration suites; XGBoost tabular control", "6-12", "100% choice-order invariance; task accuracy beats XGBoost control or within 2 points with better calibration (Brier gate); paraphrase flip rate under 10%"],
      ["3. Looped variant (weeks 3-4)", "aurum-train-looped kernel (looped core, r in {1,2,3}); depth sweep at inference r in {1,2,3,4}; A/B vs baseline on identical splits", "8-16", "Looped beats r=1 baseline on validation + 2022-2023 window on accuracy or Brier; no calibration degradation at operating r"],
      ["4. Full-policy backtest (weeks 4-5)", "aurum-backtest CPU kernel; composed T1-T6 policy with cost model; walk-forward + bootstrap + deflated Sharpe; baselines 1-5", "2-4 (inference only)", "Section 9.3 gates met on the untouched 2024-2026 test window: Sharpe above 1.0, max DD below 20%, PF above 1.25, 150+ trades, bootstrap CI positive"],
      ["5. Paper trading (weeks 5-8)", "aurum-live-daily scheduled kernel: refresh data, run policy, emit decisions file + risk report; orchestrator reports daily", "1-2/week", "4+ weeks of live-paper operation with decision latency under 1s, calibration stable (monthly Brier check), no risk-wrapper violations"],
      ["6. Extensions (post-MVP)", "Portfolio task over added instruments (XAGUSD, FX majors); optional halting head (adaptive r); Qwen3-1.7B trunk upgrade; MT5 bridge for semi-auto execution if the owner opts in", "per scope", "Defined per extension when Phase 5 evidence is in"],
    ],
  },
  { t: "p", x: "First two weeks in concrete terms: Phase 1 lands the data pipeline and publishes aurum-data-v1 with its audit report; Phase 2 Stage A lands the head-sanity run and the invariance suite. By the end of week two the owner reviews three artifacts - the dataset card, the invariance test report, and the Stage A metrics - before any large GPU spend begins. From Phase 2 onward the orchestrator loop of Section 8.4 runs all experiments: the owner never has to touch the Kaggle UI unless a quota decision is needed." },
]});

// ══════════════════════════════ 12. RISK REGISTER ══════════════════════════════
S.push({ h1: "12. Risk Register and Mitigations", blocks: [
  { t: "table",
    caption: "Table 11: Risk register (likelihood and impact rated High/Medium/Low)",
    headers: ["Risk", "L", "I", "Mitigation"],
    widths: [34, 7, 7, 52],
    rows: [
      ["Overfitting to limited gold history", "H", "H", "Purged CV + embargo; single touch of the test window; deflated Sharpe; mandatory baselines incl. XGBoost control; phase gates"],
      ["Regime dependence (edge is a bull-market artifact)", "M", "H", "Regime-sliced evaluation; walk-forward windows spanning 2022 bear and 2024-2026 bull; regime features in state; monthly refit cadence in operation"],
      ["Loop adds nothing at 0.45B scale (useful-r collapse)", "M", "M", "r=1 checkpoint is always the deployable fallback; loop is an ablation, not a dependency; success gate framed as beat-or-match"],
      ["Training instability of looped fine-tune", "M", "M", "Identity-init adapter; sandwich norms; depth-scaled init for new out-projections; Stage A isolates head bugs first"],
      ["Calibration drift in operation", "M", "H", "Per-checkpoint temperature; monthly Brier checks in paper trading; auto-flag (not auto-trade) on drift"],
      ["News gaps invalidate bar-based decisions", "H", "M", "Code-level event calendar guards (flat/hold around top-tier releases); no stop-free positions; weekend-flat default"],
      ["Class imbalance (no-trade dominates)", "H", "M", "Per-task balance caps; cost-weighted loss; threshold tuning on the precision-recall curve, gated by trade-count minimum"],
      ["Kaggle quota exhaustion mid-phase", "M", "M", "CPU kernels for all non-training work; 2 concurrent commits; chunked resumable runs; rented-A100 fallback ($20-40 covers a Bev-scale campaign)"],
      ["Live data source rate limits or gaps (Yahoo 429s)", "M", "L", "Retry with backoff and headers; HistData monthly re-anchor; Dukascopy secondary; missing-data means no decision (fail-safe flat)"],
      ["Execution realism gap (backtest flatters live)", "M", "H", "Conservative cost model with 1.5-2x stress runs; next-bar-open execution; slippage on stops; paper trading before any live consideration"],
      ["Rewording/format fragility (Laya lesson)", "M", "M", "Fixed versioned state format; paraphrase-augmented training; paraphrase invariance in the eval suite"],
      ["Choice-order leakage into live prompts", "L", "M", "Invariance by construction (mask + position reset) plus 100% permutation test gate every checkpoint"],
    ],
  },
]});

// ══════════════════════════════ 13. CONCLUSIONS ══════════════════════════════
S.push({ h1: "13. Conclusions and Next Steps", blocks: [
  { t: "p", x: "The research supports proceeding. Every component of the proposed system rests on a demonstrated precedent: the decision-model interface and its calibration discipline on the Jev ecosystem; the choice-order-invariance machinery and the 20-layer Qwen3-0.6B recipe on the Bev walkthrough; the looped transformer mechanics on Huginn and the 2026 literature; the labeling and validation methodology on the financial machine-learning canon; and the training-and-automation substrate on today's live verification of the owner's Kaggle access. The genuinely novel element - recurrent depth inside a decision model - is novel in the safe direction: the unlooped model is the r=1 special case of its own checkpoint, so the loop can be adopted or dropped on evidence rather than conviction." },
  { t: "p", x: "The recommended immediate next step is Phase 1: the data pipeline kernel and the aurum-data-v1 dataset, which costs no GPU quota and unblocks everything else. Three decisions from the owner are requested before or during Phase 1: (1) default risk parameters for the sizing task - proposed starting point 0.5% risk per trade with a 2.0% hard cap, owner-adjustable; (2) confirmation of the primary trading horizon - proposed H1 decision cadence with multi-day holds, matching the swing-style profile implied by the research; (3) explicit go-ahead for Phase 1. The owner should also regenerate the Kaggle API token shared in chat once the project's automation is handed over, as good hygiene for a credential that has passed through a conversation." },
]});

// ══════════════════════════════ 14. REFERENCES ══════════════════════════════
S.push({ h1: "14. References", blocks: [
  { t: "numbered", ref: "refs", items: [
    "TypeSafe AI - Introducing System One Models and Jev (blog), September 2026. https://typesafe.ai/blog/introducing-system-one-models-and-jev",
    "TypeSafe AI - homepage: RLCD, latency and cost claims, structured-output error table. https://typesafe.ai",
    "JevBench benchmark repository and leaderboard (v1.4.2.2). https://github.com/fstandhartinger/jevbench",
    "Kev: Jev-like family of small decision models (code and model cards). https://github.com/jaredpalmer/kev and https://huggingface.co/jaredpalmer/kev-4b",
    "Laya: multilingual System 1 decision model (ModernBERT-based), Apache 2.0. https://huggingface.co/convaiinnovations/laya",
    "AnyJev (Nokia Applied Research): turning frozen LLMs into decision models. https://github.com/nokia-applied-research/AnyJev",
    "Jev's Open Rivals Test the Decision Model Idea - Laya, Kev, AnyJev, DiffusionGemma compared. Trilogy AI Center of Excellence, September 2026. https://trilogyai.substack.com/p/jev-open-decision-models",
    "Building a Choice-Order Invariant Decision Language Model from Scratch (Bev Decider 0.4B walkthrough, project-provided document, 2026).",
    "Geiping et al. - Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn). arXiv 2502.05171. https://arxiv.org/abs/2502.05171",
    "Huginn-0125 model card (recurrent-depth 3.5B; depth modification at inference). https://huggingface.co/tomg-group-umd/huginn-0125",
    "What Are Looped Transformers? A Technical Deep Dive. saulius.io, September 2026. https://saulius.io/blog/what-are-looped-transformers-recurrent-depth",
    "Li, Liu, Zhou, Ma - Transformers are limited to AC0 (constant-depth circuit bounds). ICLR 2024, arXiv 2402.12875.",
    "Graves - Adaptive Computation Time for Recurrent Neural Networks, 2016 (per-position halting; Universal Transformer basis). arXiv 1603.08983",
    "Azizi et al. - Looped Language Models: A Survey and Meta-Analysis, 2026; Zhang - Recurrent Looped Transformer, 2026; LT2: Linear-Time Looped Transformers, arXiv 2605.20670; Looped Transformers under the Jacobian Lens, arXiv 2609.01924.",
    "Qwen3-0.6B model card (28 layers, GQA 16Q/8KV, Apache 2.0). https://huggingface.co/Qwen/Qwen3-0.6B and Qwen3 Technical Report",
    "Qwen3.5 hybrid Gated DeltaNet architecture analyses (GDN:full-attention ~3:1). Raschka LLM Architecture Gallery; ai.tekin.cn deep dive, 2026",
    "HistData.com - free forex/metals M1 and tick historical data (66 instruments incl. XAUUSD). https://www.histdata.com",
    "Dukascopy - free historical data feed (tick-to-monthly CSV export). https://www.dukascopy.com/swiss/english/marketwatch/historical",
    "Kaggle Notebooks documentation and community-verified limits (GPU quota, session length, concurrent sessions, storage). https://www.kaggle.com/docs/notebooks",
    "vectorbt - vectorized backtesting research engine. https://github.com/polakowo/vectorbt",
    "Lopez de Prado - Advances in Financial Machine Learning (triple-barrier labeling, meta-labeling, purged CV, deflated Sharpe). Wiley, 2018.",
    "Gold ML literature: Charugulla et al. 2025 (transformer gold futures, PubMed 41082458); arXiv 2512.22606 (LSTM/MLP + Gray Wolf Optimizer); Varela 2025 (FinBERT+LSTM real-time gold framework, ScienceDirect).",
  ]},
]});

module.exports = S;
