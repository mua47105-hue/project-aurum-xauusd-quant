"use strict";
// ============================================================================
// AURUM Spec v2.0 Master Report — Content Part 1 (rev v2.0.2)
// Sections 1-3: Executive Summary, The Unified Codebase, Phase 1 (K1)
// ============================================================================
const H = require("./helpers");
const { h1, h2, body, bodyRuns, bullet, figure, mkTable, tableTitle, fmt, P } = H;

function build(D) {
  const cc = D.k6cc;
  const m7c = cc.M7_Core_Satellite || {};
  const m7a = cc.M7_Active_Grand_Hybrid || {};
  const m0 = cc.M0_Champ_v13 || {};
  const m1 = cc.M1_GBDT_v2 || {};
  const meta = D.k6stats._meta || {};
  const out = [];

  // ================================================================ 1
  out.push(h1("1. Executive Summary"));
  out.push(body(
    "Project AURUM Spec v2.0 mandated the synthesis of two institutional deep-research " +
    "blueprints — the AURUM v2 Quantitative Research Plan (48 primary sources) and the " +
    "AURUM Spec v2.0 Industrial Quant Blueprint (80+ primary sources, 13 self-tested " +
    "modules) — into a single unified codebase, and the autonomous end-to-end execution " +
    "of the complete pipeline on Kaggle's free tier, racing and blending the result " +
    "against the Spec v1.3 Champion (Flagship cap=3: OOS 2023-25 annualized Sharpe " +
    "2.444, MaxDD 13.06%, DSR 0.895 under its own barrier-trade accounting). This report " +
    "is the execution record and the tournament verdict, at code revision v2.0.2 — the " +
    "revision that repaired Engineering Flaw #8 at the forecast producer and measured, " +
    "for the first time, what the v2.0 active stack is actually worth."));
  out.push(bodyRuns([
    { text: "The headline question of the program — does the v2.0 continuous stack add " +
            "active skill on top of the champion? — now has a measured, honest answer: " },
    { text: "no. ", bold: true },
    { text: "The M7 Active Grand Hybrid (50% champion + 50% de-saturated active v2 " +
            "stack) runs at " + fmt(m7a.sharpe_ann, 3) + " OOS (" +
            fmt(m7a.cost2x_sharpe, 3) + " at 2.0x costs) against the champion's own " +
            fmt(m0.sharpe_ann, 3) + " — the active stack's true orthogonality gain is " +
            "negative (−0.67 Sharpe at the 50% weight). The de-saturated M2_RFF_Ridge, " +
            "rebuilt with a causal trailing standardization and gated at the producer " +
            "(G4.8: std 0.42, zero saturation, 558 round-turns/year), measures " +
            fmt(cc.M2_RFF_Ridge.sharpe_ann, 2) + " OOS — its within-fold prediction " +
            "oscillations carry no per-bar timing information. Every pre-fix M2 number " +
            "(the K2-vintage 1.87, the tournament 2.16) was passive gold exposure in " +
            "disguise." },
  ]));
  out.push(body(
    "The benchmark that survives is stated transparently. The M7 Core-Satellite — " +
    "50% champion + 50% passive gold sleeve, the explicit decomposition of what the " +
    "v2.0.1 M7 stream actually was in OOS (its v2 leg was a saturated constant +1.0, " +
    "bit-identical to Buy & Hold) — preserves the program's best numbers verbatim: " +
    fmt(m7c.sharpe_ann, 4) + " OOS, " + fmt(m7c.cost2x_sharpe, 4) + " at 2.0x costs, " +
    "the best cost survival the program has produced. But the v2.0.2 statistical " +
    "re-measurement withdraws its 95% DSR certification: repairing Flaw #8 replaced a " +
    "+0.35 artifact trial in the DSR family with the true −0.84 trial, widening the " +
    "honest trial dispersion, and the flagship's DSR moves from 0.958 (v2.0.1 artifact " +
    "panel) to " + fmt(m7c.oos_dsr, 3) + " primary / " +
    fmt(m7c.oos_dsr_conservative, 3) + " conservative — 0.9 and 2.5 points below the " +
    "pre-registered 0.95 bar. The gate fails and is reported as failed (Section 7.4): " +
    "the core-satellite result is borderline significant, not certified."));
  out.push(body(
    "Three further results carry the program's scientific weight. First, the v2.0.1 " +
    "causality audit: the frozen-model accounting originally credited the position " +
    "formed at the close of bar t with the return of bar t itself — a one-bar " +
    "lookahead. On the Channel_A baseline the same-bar formulation shows an " +
    "impossible Sharpe of 15.51 (OOS) / 14.43 (FIT); the corrected causal accounting " +
    "gives -0.13 / -0.51. The fix is regression-gated (G4.7) so the trap can never " +
    "silently return. Second, the neural tournament's honest autopsy: the contractive " +
    "ContinuumNet M5 governed its loop's spectral radius to rho = 0.5023 max across " +
    "all ten walk-forward folds (gate rho < 0.95), yet delivered -0.45 OOS Sharpe — " +
    "the architecture is stable and the signal is absent, a distinction this report " +
    "documents rather than buries. Third, the cost-viability audit that followed the " +
    "de-saturation: a genuinely active M2/M6 at H1 frequency trades 226-307 " +
    "round-turns per year and burns 2.5-6.2x the Carver one-third cost budget, so " +
    "the active stack is demoted to reference-only — the honest operational verdict " +
    "on the continuous program at this cost structure."));
  out.push(tableTitle("Table 1.1 — The final v2.0.2 tournament card (OOS 2023-25, net of costs, vol-target units)."));
  const cardRows = [];
  const cardOrder = [
    ["M7_Core_Satellite", "M7 Core-Satellite (champion + passive gold)", "ship"],
    ["M0_Champ_v13", "v1.3 Champion cap=3 (continuous)", "ship"],
    ["M0_Champ_v13_cap1", "v1.3 Champion cap=1 (continuous)", "ship"],
    ["M7_Active_Grand_Hybrid", "M7 Active Grand Hybrid (champion + active stack)", "ship"],
    ["BuyHold_Gold", "Buy & Hold Gold", "ship"],
    ["M1_GBDT_v2", "M1 GBDT v2 (engine)", "ref"],
    ["Carver_EWMAC", "Carver EWMAC baseline (engine)", "ref"],
    ["M6_AURUM_v2_Ensemble", "M6 v2 Stack (50/50 M1+M2, de-saturated)", "ref"],
    ["M3c_DAE", "M3c DAE (engine)", "ref"],
    ["M2_RFF_Ridge", "M2 RFF-Ridge (de-saturated)", "ref"],
    ["M5_ContinuumNet", "M5 ContinuumNet", "ref"],
  ];
  for (const [k, label, cls] of cardOrder) {
    const v = cc[k];
    if (!v || typeof v.sharpe_ann !== "number") continue;
    const st = (D.k6stats[k] || {});
    const ref = cls === "ref";
    cardRows.push([
      { t: label, left: 1, c: ref ? P.secondary : P.body },
      { t: fmt(v.sharpe_ann, 2), b: k === "M7_Core_Satellite", c: ref ? P.secondary : P.body },
      fmt(v.cost2x_sharpe),
      { t: fmt(v.oos_dsr, 3), c: (v.oos_dsr >= 0.95 ? P.green : P.secondary) },
      st.oos_bootstrap ? st.oos_bootstrap.sharpe_ci[0] : "—",
      st.oos_bootstrap ? st.oos_bootstrap.sharpe_ci[1] : "—",
      fmt(v.maxdd_volunits, 3),
    ]);
  }
  cardRows.push([
    { t: "v1.3 Flagship cap=3 (trade-based reference)", left: 1 },
    { t: "2.444", b: true }, "—", { t: "0.895", c: P.secondary }, "—", "—", { t: "13.06%", c: P.secondary },
  ]);
  out.push(mkTable(
    ["Strategy", "OOS Sharpe", "@2.0x", "DSR", "CI low", "CI high", "MaxDD"],
    cardRows, [30, 12, 11, 10, 12, 12, 13]));
  out.push(captionH(
    "'ship' rows are the speed-limit-gated shipping family; grey 'ref' rows are " +
    "reference-only (tournament also-rans, or — for M2/M6 — the de-saturated active " +
    "stack, which fails the Carver one-third cost budget by 2.5-6.2x and is demoted " +
    "from the shipping family with its audit flags exported verbatim). No strategy " +
    "remains quarantined under G4.8: the saturation was fixed at the producer in " +
    "v2.0.2 and the downstream audit now flags none. The v1.3 reference row is " +
    "barrier-trade accounting in per-trade R units and is not directly " +
    "unit-comparable with the continuous rows above it."));
  out.push(body(
    "The final gate register reads: eleven of twelve gates green, one honest failure. " +
    "G4.7 causality regression, G4.8 z-quality audit (producer-gated, zero flags), " +
    "G5.3 engine gauntlet (18 decisions), G5.4 Carver speed limit on the pruned " +
    "shipping family, G6.2 stack construction (fixed family, 90% shrink toward equal " +
    "weights), G6.6 export and G6_ALL's supporting checks all PASS; G6.1 statistics " +
    "FAILS — the flagship's DSR of 0.941 misses the pre-registered 0.95 bar by 0.9 " +
    "points, and the kernel reports the miss verbatim rather than widening the gate. " +
    "The live-side gates (K7) are green. The remainder of this report is the " +
    "evidence, including why that single red gate is itself a finding of the " +
    "program, not a failure of it."));

  // ================================================================ 2
  out.push(h1("2. The Unified v2.0 Codebase"));
  out.push(body(
    "Spec v2.0 consolidated everything the program had learned into one repository " +
    "tree, one feature store, one execution engine and one trial ledger. The codebase " +
    "is organized in three layers. The feature layer (aurum_v2/aurum_features) holds " +
    "the six feature families F1 through F6 and the builder that assembles the unified " +
    "95,008-bar H1 frame (2010-01-03 through 2025-12-31). The model layer " +
    "(aurum_v2/model) holds the shared kernel library v2_common.py (data loading, " +
    "purged walk-forward splits, portfolio metrics, the online execution engine runner, " +
    "the Master Trial Ledger and the block bootstrap), the OnlineExecutionEngine " +
    "components, the fusion losses, the denoising autoencoder / DMN architectures and " +
    "the contractive ContinuumNet with its JVP spectral governor. The kernel layer " +
    "(aurum_v2/kernels) holds the five pipeline stages K1 through K7 that ran on " +
    "Kaggle, each auto-bundled from the same sources by scripts/v2/bundle.py so that " +
    "the cloud kernels and the local runs execute literally identical code."));
  out.push(tableTitle("Table 2.1 — The five-stage v2.0 pipeline and its Kaggle execution record."));
  out.push(mkTable(
    ["Stage", "Kernel (tskdev/...)", "Purpose", "State"],
    [
      [{ t: "K1 Feature store", left: 1 }, "aurum-v2-features",
       { t: "375 causal features, Rank-IC audit, DST clock fixes", left: 1 }, "complete"],
      [{ t: "K2 CPU models", left: 1 }, "aurum-v2-cpu-models (+ranigain mirror)",
       { t: "M1 GBDT, M2 RFF-Ridge, RLSBank, engine gauntlet", left: 1 }, "complete (Flaw #8 found & FIXED v2.0.2)"],
      [{ t: "K3 Neural tournament", left: 1 }, "aurum-v2-neural-compact / -completion / ranigain m5-completion",
       { t: "M3c/M3m DAE, M4/M4b DMN, M5 ContinuumNet, rho governance", left: 1 }, "complete (autopsy)"],
      [{ t: "K6 Backtest", left: 1 }, "aurum-v2-backtest",
       { t: "18-strategy walk-forward, costs, bootstrap, DSR, PBO, gates", left: 1 }, "complete, 11/12 gates green (G6.1 honest miss)"],
      [{ t: "K7 Live daily", left: 1 }, "aurum-v2-live-daily",
       { t: "recomputable decision object, G6 live gates", left: 1 }, "complete (v2.0.2 re-run)"],
    ], [16, 28, 40, 16]));
  out.push(body(
    "Execution ran on two Kaggle accounts under a strict free-tier budget: tskdev for " +
    "CPU work and dataset hosting, ranigain for the T4x2 GPU hours (the M5 completion " +
    "folds 2023-2025). Every intermediate artifact is a versioned private dataset — " +
    "the feature store, the CPU forecasts (now at v3, the de-saturated vintage), the " +
    "neural forecasts, the v1.3 artifacts, the K6/K7 outputs, and a full workspace " +
    "backup — so any stage can be reproduced or audited from the cloud alone. The " +
    "backtest kernel adds a data-signature checkpoint system: each per-strategy result " +
    "is cached under a key derived from the code revision, the strategy name and the " +
    "md5 of the full forecast series, so replacing an input (for example M2's " +
    "de-saturated forecast column in v2.0.2) invalidates exactly the affected " +
    "strategies and nothing else — the full 18-strategy re-evaluation completed in " +
    "three and a half minutes."));
  out.push(h2("2.1 The v2.0.1 revision"));
  out.push(body(
    "Two changes were made during the final phase and both are regression-gated. " +
    "First, the frozen-model accounting fix: frozen_net originally computed " +
    "z[t] * r[t], crediting the position decided at the close of bar t with the return " +
    "of bar t — a one-bar lookahead. The corrected formulation holds w[t-1] during " +
    "bar t and charges the rebalance cost at bar t, exactly matching the event-loop " +
    "engine's already-causal net = z_prev * r[t]. Second, a performance rebuild of the " +
    "engine's vol-of-vol regime multiplier (the expanding 80th percentile is now " +
    "precomputed vectorized instead of re-estimated every bar, a 15x speedup with " +
    "bit-identical outputs), which made the full 18-strategy grid computable in " +
    "minutes on CPU. The K6 gate G4.7 re-derives the same-bar pathology on purpose " +
    "and fails loudly if it ever reappears."));
  out.push(h2("2.2 The v2.0.2 revision"));
  out.push(body(
    "The v2.0.2 revision executed the Engineering Flaw #8 remediation end-to-end " +
    "(Section 6.3). At the producer, K2's M2 position transform was replaced by a " +
    "causal trailing standardization — z = tanh((pred − EWMA_504(pred)) / (2 · " +
    "max(EWSTD_504(pred), 1e-9))), re-anchored at each yearly re-fit boundary — with " +
    "the saturation gate enforced inside K2 itself (G4.8 producer: std > 0.05, " +
    "|z| > 0.99 on under 5% of bars, OOS frozen turnover above 2 round-turns per " +
    "year); the raw model output pred_rff is byte-identical, so the forecast dataset " +
    "versioned cleanly (aurum-v2-cpu-preds v3). Downstream, the M6 stack was rebuilt " +
    "per Carver's shrinkage rule — the family is the fixed a-priori CPU pair " +
    "{M1_GBDT_v2, M2_RFF_Ridge} with no FIT-window membership filter, Sharpe inputs " +
    "are the buffered engine FIT estimates from the gauntlet (the shipped execution " +
    "configuration), and the estimates are shrunk 90% toward the equal-weight prior — " +
    "so M6 is now a genuinely diversified 50/50 blend of M1 and the de-saturated M2. " +
    "The old single M7 was split into two explicit, side-by-side benchmarks: the " +
    "M7 Active Grand Hybrid (the intended champion-plus-active-stack design, now " +
    "measurable for the first time) and the M7 Core-Satellite (champion plus passive " +
    "gold sleeve, the transparent accounting of what the v2.0.1 stream was in OOS)."));

  // ================================================================ 3
  out.push(h1("3. Phase 1 — The Unified Feature Store (K1)"));
  out.push(body(
    "The feature store is the program's single source of truth: 375 features on " +
    "95,008 H1 bars, 2010-2025, assembled by the K1 kernel from the six family " +
    "builders, with the two v1.x lookahead defects (the realized-measure shift and " +
    "the Kyle-lambda toxicity shift) fixed at the source and verified by assertion. " +
    "F1 is the market-clock family (DST transitions, session membership, LBMA/COMEX/SGE " +
    "fix windows, pre-fix drift); F2 the realized-measure family (HAR-style realized " +
    "volatility, jump shares, convexity at multiple horizons); F3 the path-signature " +
    "family (second-order signatures of intra-hour FX and metals paths); F4 the " +
    "microstructure-toxicity family (Amihud, Kyle lambda, spread proxies); F5 the " +
    "EW-RLS residual family; F6 the Carver EWMAC ladder, KMW drift, Blose seasonality " +
    "and change-point flags; plus the macro_mi block of FRED-derived macro surprises."));
  out.push(tableTitle("Table 3.1 — Feature families and their walk-forward Rank-IC audit (2016-2022 folds)."));
  const famRows = D.k1fam.map(f => [
    { t: f.family, left: 1 }, String(f.n),
    fmt(f.best_abs_t, 2), fmt(f.median_abs_t, 2), String(Math.round(f.n_abs_t_gt2)),
  ]);
  out.push(mkTable(
    ["Family", "Features", "Best |t|", "Median |t|", "|t| > 2"],
    famRows, [30, 16, 18, 18, 18]));
  out.push(captionH(
    "Rank-IC t-statistics are per-feature Spearman correlations between the feature at " +
    "bar t and the forward 24-bar vol-normalized return, evaluated per walk-forward " +
    "fold and aggregated; the audit covered all 375 features."));
  out.push(body(
    "The audit's message is sober and useful: clock effects (F1) and realized " +
    "microstructure (F2) carry the strongest and most numerous persistent signals " +
    "(best |t| 4.94 and 4.63, with 12 and 55 features above |t|=2 respectively), " +
    "while the trend family's information is thin at the raw-IC level — consistent " +
    "with the later finding that the continuous models extracted only modest edge " +
    "from this panel, and that the champion's barrier-conditional structure remains " +
    "the program's most efficient use of the same information."));
  out.push(...figure("fig3_rank_ic.png",
    "Figure 3.1 — Best |t| Rank-IC by family (left) and the top-22 individual features " +
    "with sign (right). Negative t-stats are reversal effects; clock and realized " +
    "families dominate.", 500));
  return out;
}

// local caption helper (distinct name to avoid clashing with H.caption)
function captionH(text) {
  const { Paragraph, TextRun, AlignmentType, FONTS, P } = H;
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 220, line: 280 },
    children: [new TextRun({ text, size: 18, italics: true, color: P.secondary, font: FONTS })] });
}

module.exports = { build };
