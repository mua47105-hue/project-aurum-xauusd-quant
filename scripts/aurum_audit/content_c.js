"use strict";
// Sections 8-13: Flaw 6, Flaw 7, Spec Delta, Phase 0/1 Gates, Residual Risks, References
const DIR = "/home/z/my-project/scripts/aurum_audit/";
const S = [];

// ══════════════════════════════ 8. FLAW 6 ══════════════════════════════
S.push({ h1: "8. Flaw 6: Earning the Right to Beat XGBoost - Cross-Asset and Micro-Sequence State", blocks: [
  { t: "h2", x: "8.1 Confirmation: on flat features, trees win" },
  { t: "p", x: "The audit's premise is the best-documented result in applied tabular machine learning: on medium-sized numeric tabular data, gradient-boosted trees remain state of the art (Grinsztajn, Oyallon and Varoquaux, NeurIPS 2022, arXiv 2207.08815 - 45 datasets, 20,000 GPU-hours of hyperparameter search). The first AURUM research report added the financial corroboration: the eighth-place Jane Street solution reported explicitly that MLPs, time-series transformers, cross-symbol attention and embeddings 'did not work for me', and time-series foundation models lose to classical methods on half of deployment datasets. If AURUM's state is a flat summary of roughly fifteen lagging indicators - RSI 63, ADX 31, distances, tags - the looped 450M-parameter model is being asked to process tabular data, the one regime where trees are favored, while paying a 450M-parameter compute tax. The audit is correct: either the state acquires structure that trees cannot see, or the transformer loses to its own control model." },
  { t: "h2", x: "8.2 Fix part one: the cross-asset panel (all free, all dimensionless)" },
  { t: "p", x: "Gold is not priced in a vacuum, and the drivers are precisely the ones the free data stack already covers. The academic anchor is Erb and Harvey's The Golden Dilemma (2013): the real price of gold has a long-run negative relationship with real interest rates - the single most-cited macro relationship in the gold literature - and gold's long-run real return is near zero with strong mean reversion, which is exactly the kind of slow-moving cross-asset context a state should carry. The dollar side is equally established: gold and the dollar index show a significant medium-to-long-term inverse relationship, regime-dependent and nonlinear (2026 SSRN work and the nonlinear-dynamics literature agree on the sign with regime caveats). Silver gives a relative-value channel: the gold-silver ratio (about 68 at the October 2026 quotes of $4,170 gold and $61 silver) mean-reverts on multi-month horizons and is computable from the same M1 feeds as gold. The panel below lands inside the Section 3 contract - every series enters as changes, z-scores, or ATR-normalized returns, never levels - and every source is already in the project's verified free stack." },
  { t: "table",
    caption: "Table 5: The cross-asset panel added to the state (sources verified in the first research phase)",
    headers: ["Series", "Source (free)", "Granularity", "Encoded as (dimensionless)"],
    widths: [22, 26, 14, 38],
    size: 18,
    rows: [
      ["XAGUSD (silver)", "HistData / Dukascopy M1", "H1", "ret24 in ATR units; XAU/XAG ratio z-score (24 and 120 bars)"],
      ["EURUSD, USDJPY", "Dukascopy M1", "H1", "ret24 in ATR units; synthetic dollar factor; rolling correlation to XAU (24/120 bars)"],
      ["DXY dollar index", "FRED DTWEXBGS / yfinance DX-Y.NYB", "Daily", "24-bar return, z-score; forward-filled with 1-day lag"],
      ["US 10Y nominal yield", "FRED DGS10", "Daily", "Change in basis points, z-score"],
      ["US 10Y real yield (proxy)", "FRED DGS10 minus T10YIE", "Daily", "Change in basis points, z-score (Erb-Harvey channel)"],
      ["VIX", "FRED VIXCLS", "Daily", "Level z-score, 1-day change"],
      ["S&P 500", "FRED SP500", "Daily", "24-bar return in ATR units (risk appetite)"],
      ["WTI crude", "FRED DCOILWTICO", "Daily", "24-bar return in ATR units (commodity complex)"],
    ],
  },
  { t: "p", x: "Daily macro series are joined to H1 bars with a strict one-day publication lag (the FRED value for day t enters states from day t+1 onward) and an explicit staleness tag, because macro data is the easiest place in the whole pipeline to hide look-ahead bias. The panel's purpose is not decoration: the divergence combinations it exposes - dollar up with gold up (real-bid regime) versus dollar down with gold up (weak-dollar rally), real yields spiking with silver breaking down - are genuinely different states with different follow-through, and they are interactions a tree can only see if a human engineers them, but which a shared trunk learns as aligned token groups." },
  { t: "h2", x: "8.3 Fix part two: ordered micro-sequence tokens" },
  { t: "p", x: "The second structural addition gives the transformer the one thing its architecture is built for and a flat feature vector is not: ordered sequences with local semantics. The design follows the patching principle validated by PatchTST (Nie et al., ICLR 2023, arXiv 2211.14730): subseries-level tokens retain local semantic information while cutting attention cost, and were the design that finally made transformers competitive on long-horizon forecasting. Concretely, the last 30 M1 bars before each H1 decision are serialized as an ordered token run: each M1 bar maps to one shape code from a 48-entry codebook fitted by k-means on training-fold normalized bar shapes (body position, range, direction), plus a signed return-quantile token and a tick-volume z-token. The last 8 H1 bars are likewise serialized as ordered tokens (shape code, return quantile, volume quantile, session id) rather than collapsed into aggregates. The recent sequence then carries run-lengths, compression and expansion, breakout retest patterns and momentum acceleration - structure that attention can read positionally and that a gradient-boosted tree ingesting the same fifteen summary floats cannot see without a human pre-computing every interaction. Codebook fitting is training-fold-only (rule SC-7), and the codebook is versioned with the dataset." },
  { t: "table",
    caption: "Table 6: Revised 600-token state budget (computed; base case with headroom for paraphrase variants)",
    headers: ["State block", "Tokens", "Content"],
    widths: [34, 12, 54],
    size: 18,
    rows: [
      ["XAU multi-timeframe core", "110", "M15/H1/H4/D1 indicators, all dimensionless (Table 2 replacements)"],
      ["H1 bar history (ordered)", "48", "Last 8 bars x 6 tokens: shape code, ret quantile, vol quantile, session"],
      ["M1 micro-sequence tail (ordered)", "60", "Last 30 M1 bars x 2 tokens: shape code + signed ret quantile"],
      ["Cross-asset panel", "36", "Table 5 series, ~4 tokens each, grouped by asset with id tags"],
      ["Macro block", "12", "Real yield, DXY, VIX, SPX, WTI deltas and z-scores"],
      ["Session / time / event", "14", "Hour sin-cos, day-of-week, session flags, minutes to next FOMC/NFP/CPI"],
      ["Position context (when in trade)", "12", "Side, unrealized R bucket, bars held, stop distance in ATR"],
      ["Header and task framing", "20", "Fixed versioned template tokens (schema hash)"],
      ["Base total", "312", "Grows to ~350-420 with paraphrase variants - inside the 600 budget"],
    ],
  },
  { t: "h2", x: "8.4 The honest control: race the transformer, gate on disagreement" },
  { t: "p", x: "The gradient-boosting control is promoted from baseline to permanent racing partner, for three reasons. First, as the ablation control it gets the identical flat features plus cheap derived sequence aggregates (run-length of directional bars, last-hour range percentile), so that if the looped model still wins out-of-sample, the win is attributable to the sequence structure and the trunk, not to information the control never saw. Second, as an ensemble member it is the cheap second opinion the verified individual winners all converged on - the Jane Street second place derived a third of its final score from a 17-model ensemble plus online updates, and the Optiver solo winner blended CatBoost, a GRU and a transformer. Third, as a gate: when the transformer and the GBDT control disagree on direction at the chosen cell, the state is one the two model families read differently, and the default action is stand-aside - disagreement is uncertainty, and uncertainty at 0.55-class edges is a no-trade. The loop-depth divergence signal (Section 5) and the GBDT disagreement score together form the OOD gate's two inputs." },
  { t: "note", x: "Acceptance gates for Flaw 6: looped model beats the identically-fed GBDT control on the decision metric (net EV capture and barrier-head Brier) on purged OOS; removing the sequence blocks degrades the looped model measurably (registered ablation proving the structure earns its tokens); cross-asset staleness audit shows zero same-day macro joins." },
]});

// ══════════════════════════════ 9. FLAW 7 ══════════════════════════════
S.push({ h1: "9. Flaw 7: Walk-Forward Inference Compute Architecture", blocks: [
  { t: "h2", x: "9.1 Corrected arithmetic: the GPU dissolves the problem, and exposes the real one" },
  { t: "p", x: "The audit's arithmetic is confirmed: 30,000 H1 bars x 6 task queries at loop depth r = 3 is 180,000 sequential forward passes, and at the 7-40 seconds per decision the first report measured for a 0.6B looped model on CPU, that is 15-83 days of wall clock - a timeout, not a backtest. The joint-head fix of Section 5 removes the factor of six at the source: one pass per bar for entry decisions plus a management pass on the roughly 30 percent of bars where a position is open. The remaining compute is then a batching problem, and batching is what GPUs do. The forward cost of the 20-layer trunk is about 0.63 GFLOP per token at r = 1, rising to 0.88 GFLOP per token at r = 3 (28 layer-passes); 30,000 states x 600 tokens is 18 million tokens, so a full r = 3 sweep is 15.9 PFLOP - 11 to 20 minutes on a single T4 at the conservative 13-23 TFLOPS effective throughput that batched fp16 SDPA inference achieves. Producing probability tables for all three loop depths r = 1, 2, 3, entry plus management, totals about 53 PFLOP, or 0.6 to 2.3 hours on one T4. Both T4s halve that. The complete out-of-sample inference layer of the project costs less GPU time than one percent of a weekly quota - provided it is engineered as the two-kernel design below rather than as per-bar calls." },
  { t: "h2", x: "9.2 The fix: the two-kernel probability-table architecture" },
  { t: "p", x: "Kernel A, the OOS table builder, runs on GPU and does nothing but turn features into probability tables. Its stages: load the versioned feature parquet; serialize all decision-point states once (orjson, schema-hashed); tokenize in batches with the fast tokenizer; run batched fp16 inference under torch inference mode with SDPA attention, batch size 128, pinned memory, two data-loader workers, and a resume checkpoint every 2,000 batches so a 12-hour session cap or a preemption never costs a rerun; collect the head outputs (48 barrier logits, quantile heads, vol, regime, the loop-divergence OOD score, and the r = 1, 2, 3 variants from the same batch) plus the GBDT control's probabilities computed on the same rows; and write one row per (timestamp, fold id, r) to a compressed parquet - the probability table - published as a versioned private Kaggle Dataset. No generation, no per-sample Python overhead, no model.generate anywhere: this is single-forward classification, which is the cheapest thing a transformer does." },
  { t: "p", x: "Kernel B, the backtest runner, is CPU-only and consumes the table without ever touching the model. The deterministic decision layer evaluates EV per direction and cell (costs, swap, blackout windows), applies the gate and the fractional-Kelly cap, and emits a trade list; vectorbt then runs the vectorized execution sweeps, and the NautilusTrader realism pass re-executes the final configuration on bid/ask data. The decisive property is that every policy variant - cost multipliers at 1.0x, 1.5x, 2.0x; loop depths r = 1, 2, 3; gate thresholds; Kelly fractions; each walk-forward fold's checkpoint - is a re-read of the same table with different post-processing parameters. Inference happens once; the entire experiment matrix of Phase 4 is then a CPU workload that costs zero GPU quota. This is the audit's requested batched GPU pre-computation, made concrete." },
  { t: "h2", x: "9.3 The real bottleneck: refits, not inference" },
  { t: "p", x: "Correcting the compute picture honestly exposes where the true constraint lives: walk-forward retraining. A single Stage B decision SFT over the 120,000-example corpus is 144 million tokens for two epochs - 4.2 to 14.8 GPU-hours on two T4s at the assumed 1,500-4,000 tokens per second per device and the measured 1.8-2.4x DDP speedup. Six cold refits for a full combinatorial purged CV would cost 26-92 GPU-hours before any ablation - not one quota week. The adopted schedule is therefore: one anchor model trained on the earliest regime blocks, three warm-start refits (each continuing from the previous checkpoint for half an epoch on newly arrived data, 1.0-3.7 GPU-hours each), probability tables per fold, and the registered ablations. The complete walk-forward campaign lands at 12-37 GPU-hours - one to one-and-a-half quota weeks, with the fold structure, embargo discipline, Deflated Sharpe Ratio and probability-of-backtest-overfitting gates of the original validation protocol unchanged; what changes is that fold count is honest about the free tier's budget, and every configuration trial is counted exactly as the multiple-testing chapter demands. Figure 2 shows the budget." },
  { t: "figure", path: DIR + "fig2_walkforward_budget.png", width: 560,
    caption: "Figure 2: The corrected walk-forward compute budget on Kaggle's free tier (2 x T4). The full campaign - Stage A-prime realignment, anchor SFT, three warm-start refits, all OOS probability tables at r = 1, 2, 3, and the ablation suite - totals 12-37 GPU-hours against the 30-hour weekly quota. Full CPCV with six cold refits (26-92 GPU-hours) is the infeasible alternative and is replaced by warm-start refits under the same statistical gates." },
  { t: "p", x: "Two acceleration options are registered as optional, not load-bearing. ONNX Runtime with fp16 has delivered roughly 2.9x throughput gains over eager PyTorch in production NLP serving, with the largest wins at small batch sizes; exporting the looped model with r unrolled is a Phase 4 optimization only if the table builder misses its two-hour gate. And torch.compile on the hand-rolled looped forward is worth one smoke-test measurement in Phase 0, with no commitment." },
  { t: "note", x: "Acceptance gates for Flaw 7: Phase 0 smoke test measures actual looped inference tokens per second on T4 (gate: full r = 3 table for 30,000 bars in at most 2 hours on one T4); resume-from-checkpoint verified by killing the kernel mid-run; probability-table schema versioned and frozen before Phase 4." },
]});

// ══════════════════════════════ 10. SPEC DELTA ══════════════════════════════
S.push({ h1: "10. Consolidated Specification Delta", blocks: [
  { t: "p", x: "The table below is the authoritative diff against the AURUM Decider Research Report and Build Plan (version 1.0). Sections not listed are unchanged. The intent is that this table, together with the acceptance gates of Section 11, is what gets frozen as version 1.1 of the build plan before Phase 0 begins." },
  { t: "table",
    caption: "Table 7: Section-by-section specification delta (version 1.0 to version 1.1)",
    headers: ["Plan section", "Version 1.0", "Version 1.1 (this pass)"],
    widths: [16, 38, 46],
    size: 18,
    rows: [
      ["6.4 Features and labels", "Triple-barrier labels at fixed m, n; hindsight exit and stop labels", "Barrier-matrix labels over 8 standardized cells x 3 outcomes x 2 directions; MFE/MAE and timeout-MTM quantile labels; FFD features for the tabular control"],
      ["6.5 Augmentation", "Counterfactual pairs with 'recomputed' labels (10 percent)", "Removed; replaced by matched-cohort contrastive pairs (8 percent) + bootstrap soft labels (B = 200) + paraphrase and feature dropout"],
      ["7.1 State serialization", "Nominal numbers allowed ('swing HH at 2612', 'ATR 6.2')", "Scale-invariance contract SC-1..SC-8; corrected dimensionless fragment; three-layer leakage audit (invariance test, adversarial validation, representation probe)"],
      ["7.2 Six-task decision map", "T1-T6 as six prompted tasks in cascade", "Barrier-matrix world model: joint heads (~85 outputs), one forward pass; tasks become heads or decision-layer functions (Table 3)"],
      ["7.3 Probabilities to positions", "Kelly on p and b from different tasks", "EV-argmax over cells net of ATR-unit costs; fractional Kelly f* = p - (1-p)/R on the chosen cell; quarter-Kelly, 1 percent cap"],
      ["10.1 Loop mechanics", "Frozen trunk + rank-16 LoRA; concat adapter 2048-to-1024", "Unfrozen 63M core; per-iteration sandwich RMSNorm; gated anchor re-injection with ReZero lambdas; per-iteration LoRA r=8"],
      ["10.3 Parameter budget", "~12-15M trainable", "75.1M trainable; peak VRAM ~3.7 GB of 15.4 GB on one T4 (Table 4)"],
      ["10.4 Training recipe", "Stages A (head sanity), B (SFT), C (full unfreeze escalation)", "Stage A unchanged; new Stage A-prime realignment CPT (150-300M tokens, KL anchor, random-r supervision); Stage B joint-head SFT; Stage C unchanged as escalation"],
      ["10.5 Dataset budget", "120k examples, 45 percent T1/T2, counterfactual 10 percent", "120k examples re-mixed (Section 4.3); labels per barrier cell + quantiles; matched cohorts 8 percent; paraphrase/dropout 10 percent"],
      ["State content (6.4/7.1)", "Single-asset, ~15 indicator summaries", "Cross-asset panel (Table 5) + ordered M1 and H1 sequence tokens (Table 6); base 312 tokens of 600"],
      ["9.6 / Phase 4 backtest", "Per-bar policy queries during backtest", "Two-kernel architecture: GPU probability tables once, CPU decision layer + vectorbt sweeps offline; all variants from one table"],
      ["Baselines (9.4)", "XGBoost as one of five baselines", "GBDT control promoted to racing partner + disagreement gate + r=1 ablation; ensemble reporting"],
    ],
  },
  { t: "p", x: "Two parameters are deliberately left open for the owner, as in version 1.0: the vertical barrier H (default 48 H1 bars, to be revisited against the multi-horizon experiment matrix of the first research report) and the Kelly multiplier k (default quarter-Kelly, 0.25, with the 1.0 percent hard cap). Everything else in this table is proposed for immediate freeze." },
]});

// ══════════════════════════════ 11. PHASE 0/1 GATES ══════════════════════════════
S.push({ h1: "11. Updated Phase 0 and Phase 1 Gate Checklist", blocks: [
  { t: "p", x: "Phase 0 (smoke tests) and Phase 1 (data infrastructure) absorb the new machinery. Every gate below is numeric and automatically checkable; the GO trigger for the build is that all Phase 0 gates pass and Phase 1 lands its dataset with every audit green. No GPU-week-scale commitment begins before that conjunction." },
  { t: "table",
    caption: "Table 8: Revised Phase 0 and Phase 1 gates",
    headers: ["Phase", "Gate", "Threshold", "Verifies"],
    widths: [10, 34, 26, 30],
    size: 18,
    rows: [
      ["P0", "Looped forward-backward throughput on T4 (hand-rolled loop, batch 16 x 600)", "at least 1,500 tok/s single T4; 2,700 tok/s on 2x T4", "Feasibility math of Sections 7 and 9"],
      ["P0", "Full r=3 probability-table build for a 30,000-bar sample", "at most 2 hours on one T4; resume works after kill", "Flaw 7 architecture"],
      ["P0", "Stage A-prime pilot (5M tokens)", "NLL within 5 percent of r=1 baseline at pilot scale; lambdas move off zero", "Realignment recipe trains stably"],
      ["P0", "Scale-invariance unit test on synthetic panel", "100 percent byte-identical under a in {0.25, 0.5, 2, 4}", "Flaw 1 contract is testable"],
      ["P1", "Cross-venue data audit (Dukascopy vs HistData, XAU + XAG + EUR + JPY)", "Divergence quantified and under registered tolerance; UTC normalization verified", "Data foundation incl. cross-asset panel"],
      ["P1", "Adversarial validation harness on train/test boundaries", "AUC at or below 0.55 at every boundary", "Flaw 1 gate L2"],
      ["P1", "Barrier-matrix labeler + bootstrap soft labels (B = 200) on CPU kernel", "All cells labeled; label distributions documented per regime block", "Flaws 3, 4 label schema"],
      ["P1", "Matched-cohort index built", "at least 70 percent of anchors matched within caliper; median distance reported", "Flaw 2 mechanism one"],
      ["P1", "M1/H1 shape codebooks fitted (training folds only) and versioned", "Reconstruction sanity check; schema hash registered", "Flaw 6 sequence tokens; rule SC-7"],
      ["P1", "aurum-data-v1 dataset published with audits green", "All P1 gates above green in the build log", "GO trigger for Phase 2 (baseline decider)"],
    ],
  },
  { t: "p", x: "Phase 2 onward inherits the original roadmap unchanged except where Table 7 rewrites the internals: the baseline decider of Phase 2 now trains the joint heads at r = 1, the looped variant of Phase 3 is the Stage A-prime plus Stage B pipeline, and the Phase 4 backtest consumes probability tables. The first report's statistical release gates - Deflated Sharpe Ratio at or above 0.95, probability of backtest overfitting below 0.5, every trial counted, single touch of the test window - remain the authority on whether anything ships." },
]});

// ══════════════════════════════ 12. RESIDUAL RISKS ══════════════════════════════
S.push({ h1: "12. Residual Risks and Open Questions", blocks: [
  { t: "p", x: "Fixing seven flaws does not make the project safe; it makes it internally consistent. Six residual risks remain, deliberately stated without mitigation theater. First, the block bootstrap soft labels preserve short-lag dependence but not long-horizon regime persistence, which is why the hard-vs-soft A/B is a registered gate rather than an assumption. Second, matched-cohort overlap can fail exactly when it is most wanted - in violent regimes, the flipped tag may not co-occur with comparable covariates, and the positivity audit will shrink the augmentation precisely then. Third, the fold-count honesty of Section 9 means the walk-forward evidence base is thinner than a full CPCV would give; the Deflated Sharpe and PBO machinery is designed for exactly this situation, but no statistic conjures information that six fewer refits did not produce. Fourth, T4 is fp16-only and Turing-old: the realignment recipe's stability on fp16 (loss scaling, no bf16 fallback) is a Phase 0 measurement, not a given. Fifth, the negative results in the looped-transformer literature still apply in full - looping may simply add nothing at 0.6B scale on this task, in which case the r=1 model ships and the loop remains a published negative ablation. Sixth, the Qwen3.5-0.8B alternate backbone remains unverified for Unsloth-style tooling, and the Jev reference video the owner mentioned has still not arrived; neither blocks the build." },
  { t: "p", x: "One open question is flagged for the owner's judgment rather than resolved here: whether the management-pass exit head (Section 5.3) should be in the MVP or deferred. The computed budget accommodates either; the recommendation is to ship the deterministic derived exits first and let the Phase 4 ablation decide whether the learned head earns its second pass." },
]});

// ══════════════════════════════ 13. REFERENCES ══════════════════════════════
S.push({ h1: "13. References", blocks: [
  { t: "p", x: "Sources are grouped by load. Entries marked with an arXiv identifier or URL were verified in this pass or in the two prior AURUM research phases; the prior reports' 180-source evidence base is referenced as a unit where its findings are reused. Confidence conventions follow Section 2." },
  { t: "h2", x: "13.1 Labeling, calibration, and distributional methods" },
  { t: "numbered", ref: "refs", items: [
    "Lopez de Prado, M. Advances in Financial Machine Learning. Wiley, 2018. (Triple-barrier method and meta-labeling, ch. 3; fractionally differentiated features, ch. 5; purged CV and the Sharpe statistics with Bailey.)",
    "mlfinpy documentation, Data Labelling: Triple-Barrier Method and Meta-Labeling. mlfinpy.readthedocs.io (meta-labeling quotation and F1 framing).",
    "Politis, D. N., and Romano, J. P. The Stationary Bootstrap. Journal of the American Statistical Association, 1994.",
    "Koenker, R., and Bassett, G. Regression Quantiles. Econometrica, 1978; Koenker, R. Quantile Regression (Cambridge, 2001 - pinball loss survey, 8,800+ citations).",
    "Muller, R., Kornblith, S., and Hinton, G. When Does Label Smoothing Help? NeurIPS 2019, arXiv 1906.02629 (label smoothing improves calibration).",
    "Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. On Calibration of Modern Neural Networks. ICML 2017, arXiv 1706.04599 (temperature scaling).",
    "Bellemare, M., Dabney, W., and Munos, R. A Distributional Perspective on Reinforcement Learning (C51). NeurIPS 2017; Dabney, W., et al. Distributional RL with Quantile Regression. arXiv 1710.10044.",
    "Ruder, S. An Overview of Multi-Task Learning in Deep Neural Networks, 2017; Abdelsamie, M., et al. Deep Multi-Task Learning: A Review. Springer, 2026 (hard parameter sharing).",
    "Kelly, J. A New Interpretation of Information Rate. Bell System Technical Journal, 1956 (fractional Kelly sizing, quarter-Kelly practice).",
    "Sweeney, J. Maximum Adverse Excursion. Wiley, 1997; TradesViz, MFE and MAE: The Complete Guide, 2026 (excursion analysis practice and R-multiple conventions).",
    "Morgan, S., and Harding, D. Matching Estimators of Causal Effects. Sociological Methods and Research, 2006; Stuart, E. Matching Methods for Causal Inference. Statistical Science, 2010; Abadie, A., and Imbens, G. Large Sample Properties of Matching Estimators. QJE, 2006.",
    "Lazarev, E. purgedcv: purged and combinatorial cross-validation for financial ML. 2026, github.com/eslazarev/purged-cross-validation; Bailey, D., and Lopez de Prado, M. The Deflated Sharpe Ratio (JPM 2014) and The Probability of Backtest Overfitting (JCF 2016).",
    "Kaggle community, Adversarial Validation (general/488194) and production drift-detection practice (apxml.com course chapter), as applied to train/test shift detection.",
  ]},
  { t: "h2", x: "13.2 Looped and recursive transformer architecture" },
  { t: "numbered", ref: "refs2", items: [
    "Bae, S., Fisch, A., Harutyunyan, H., Ji, Z., Kim, S., and Schuster, T. Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA. ICLR 2025, arXiv 2410.20672 (multi-LoRA per loop iteration, SVD initialization +6.5 points, 60B-token continued training + distillation, 58.4 vs 58.6 recovery).",
    "Geiping, J., et al. Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn). arXiv 2502.05171 (prelude/core/coda, input re-injection, 800B tokens).",
    "Park, et al. LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models. arXiv 2605.11011 (softplus gate, random deep supervision, 3B-token conversion).",
    "McLeish, et al. Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence. arXiv 2511.07384 (25-52B-token surgery).",
    "Zhu, et al. Scaling Latent Reasoning via Looped Language Models (Ouro). arXiv 2510.25741 (7.7T-token full-stack looping, per-step loss).",
    "Inner Loop Inference, arXiv 2602.14759 (zero-training looping of frozen models); loop-position retention study, arXiv 2610.19768 (99.8 percent retention with random loop supervision).",
    "Bachlechner, T., et al. ReZero is All You Need. arXiv 2003.04887 (zero-initialized learned residual gates); Bai, S., Kolter, J. Z., and Koltun, V. Deep Equilibrium Models. NeurIPS 2019, arXiv 1909.01377 (fixed-point stability).",
    "Residual-stream norm growth measurements (GPT-2 XL, ~1.045x per layer), LessWrong, 2023; An Analysis of Residual-Stream Geometry Across Depth, arXiv 2607.18348, 2026; Scaling Adaptive Depth with Norm-Agnostic Residual, 2026.",
    "Qwen3-0.6B model card (28 layers, GQA 16Q/8KV, 32k context, Apache-2.0) and Qwen3 Technical Report. huggingface.co/Qwen/Qwen3-0.6B.",
  ]},
  { t: "h2", x: "13.3 Market evidence, data, and platforms" },
  { t: "numbered", ref: "refs3", items: [
    "Erb, C., and Harvey, C. The Golden Dilemma. Financial Analysts Journal, 2013, SSRN 2078535 (gold and real interest rates; the golden constant).",
    "A Study on the Dynamic Relationship Between Gold Prices and the US Dollar Index. SSRN 6442031, 2026; He, Q., et al. Nonlinear Dynamics of Gold and the Dollar (regime-dependent inverse relationship).",
    "Grinsztajn, L., Oyallon, E., and Varoquaux, G. Why do tree-based models still outperform deep learning on tabular data? NeurIPS 2022 Datasets and Benchmarks, arXiv 2207.08815.",
    "Nie, Y., et al. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST). ICLR 2023, arXiv 2211.14730.",
    "Jane Street Real-Time Market Data Forecasting (Kaggle, 2024-25): final leaderboard; second-place 17-model ensemble with online learning (kaggle.com/discussion/742687 and reproduction); eighth-place writeup (transformers did not work). Optiver Trading at the Close (2023-24): first-place writeup by hyd (CatBoost + GRU + Transformer blend).",
    "The Alpha Illusion: An Audit of End-to-End LLM Trading Claims. arXiv 2605.16895 (P1-P6 screening protocol, adopted in the first AURUM report).",
    "Kaggle notebooks documentation and staff discussion 361104 (2 x T4, 15.4 GB, 30 GPU-hours/week, 12-hour sessions, P100 retired September 2026); Unsloth T4 benchmarks (huggingface.co/blog/unsloth-trl); Microsoft, Scaling up PyTorch inference with ONNX Runtime (2.88x fp16 throughput), 2022; Hugging Face batched-inference documentation and tutorial.",
    "Free data sources verified in the first research phase: Dukascopy historical feed; HistData.com (66 instruments including XAUUSD and XAGUSD M1); FRED series DGS10, T10YIE, VIXCLS, DTWEXBGS, SP500, DCOILWTICO, GOLDPMGBD228NLBM; yfinance (DX-Y.NYB, SI=F); federalreserve.gov FOMC calendar; vectorbt 1.2.0; NautilusTrader 1.231.0; Chronos tokenization arXiv 2403.07815; LLMTime digit tokenization arXiv 2310.07820.",
    "ByteDance DeerFlow repository and its deep-research skill (github.com/bytedance/deer-flow), the methodology reference for this pass and both prior AURUM research reports (Project AURUM Deep Research Report; AURUM Decider Research Report and Build Plan, v1.0, 6 November 2026).",
  ]},
]});

module.exports = S;
