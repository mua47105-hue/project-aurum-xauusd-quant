"use strict";
// ============================================================================
// AURUM Spec v2.0 Master Report — Content Part 2
// Sections 4-6: Phase 2 (K2), Phase 3 (K3), Engineering Flaw #8 autopsy
// ============================================================================
const H = require("./helpers");
const { h1, h2, body, bodyRuns, bullet, figure, mkTable, tableTitle, fmt, safe, P, jload, path, K6 } = H;

function cap(text) {
  const { Paragraph, TextRun, AlignmentType, FONTS } = H;
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 220, line: 280 },
    children: [new TextRun({ text, size: 18, italics: true, color: P.secondary, font: FONTS })] });
}

function build(D) {
  const out = [];
  const em = D.k2engine;

  // ================================================================ 4
  out.push(h1("4. Phase 2 — CPU Models and the Execution Engine (K2)"));
  out.push(body(
    "Phase 2 trained the three CPU models in strict year-fold walk-forward (test years " +
    "2016 through 2025, train strictly before each test year with a 24-bar purge and a " +
    "15% chronological early-stop split, target y_ret24_norm — the forward 24-bar " +
    "vol-normalized XAUUSD return), and ran each through the v2.0 OnlineExecutionEngine " +
    "with explicit component toggles. M1_GBDT_v2 is a three-seed LightGBM ensemble " +
    "(depth 5, double-10 regularization, 0.06 learning rate, early stopping) whose " +
    "per-fold predictions are averaged before the position transform. M2_RFF_Ridge is " +
    "the random-features ridge (P=2048 features, gamma=sqrt(2), a ridge grid from 1e0 " +
    "to 1e4 selected on the early-stop slice, 10 seeds). RLSBank is the online " +
    "recursive-least-squares baseline whose sigma-warmup blank-forecast defect was fixed in the " +
    "final forecast vintage (94,288 of 95,008 bars covered)."));
  out.push(body(
    "Under the v2.0.1 causal accounting, M1_GBDT_v2 is the only CPU model with a " +
    "healthy, graded out-of-sample signal: FIT Sharpe -0.13 (engine) / -0.24 (frozen), " +
    "OOS Sharpe 0.94 through the full engine — the engine's online readout correction, " +
    "Carver buffer and regime multipliers lift a marginally negative in-sample model " +
    "to a positive OOS one, which is exactly the designed behavior of the execution " +
    "layer. Its cost survival is essentially flat (0.94 at 1.0x to 0.94 at 2.0x) " +
    "because the buffer holds turnover near 30 round-turns per year. M2_RFF_Ridge " +
    "is the v2.0.2 story: after the de-saturation rebuild (Section 6.3) its " +
    "genuinely active signal measures an engine FIT Sharpe of -0.74 and an engine " +
    "OOS Sharpe of -0.74 at 226-307 round-turns per year — the within-fold " +
    "oscillations of the heavy-ridge RFF predictions contain no per-bar timing " +
    "information. The pre-fix K2 vintage's reported 1.87 OOS with a 68% turnover " +
    "cut was the saturation artifact: the turnover fell because the signal " +
    "collapsed to a per-year constant, not because the buffer tamed a healthy " +
    "signal."));
  const bufRows = [];
  for (const k of ["M1_GBDT_v2__fit", "M2_RFF_Ridge__fit", "RLSBank__fit"]) {
    const d = em[k];
    if (!d || !d.full || !d.no_buffer) continue;
    bufRows.push([
      { t: k.replace("__fit", "").replace(/_/g, " "), left: 1 },
      fmt(d.no_buffer.turnover_py, 1), fmt(d.full.turnover_py, 1),
      fmt(d.no_buffer.metrics.sharpe_ann, 2), fmt(d.full.metrics.sharpe_ann, 2),
    ]);
  }
  if (bufRows.length) {
    out.push(tableTitle("Table 4.1 — Engine gauntlet on FIT 2016-22: the Carver +/-10% buffer's effect."));
    out.push(mkTable(
      ["Model", "Turnover no buffer", "Turnover buffered", "FIT Sharpe no buffer", "FIT Sharpe buffered"],
      bufRows, [24, 19, 19, 19, 19]));
    out.push(cap("K2 engine diagnostics (the same component classes K6 re-runs). The buffer cuts " +
      "turnover materially on every model; on M1 it converts a negative-FIT raw model " +
      "into the engine configuration that ships."));
  }
  out.push(body(
    "The engine itself is the phase's durable contribution: OnlineRidge delay-aware " +
    "readout correction (labels arrive 24 bars after their origin, never before), the " +
    "RLS bank, a river-exact Page-Hinkley change detector that freezes learning for 24 " +
    "bars on alarm, a Moreira-Muir volatility target at 15% annualized with a " +
    "vol-of-vol regime cut, the Carver +/-10% inertia buffer sized to the running mean " +
    "absolute position, and a Man-AHL-style drawdown governor. Every component is " +
    "causal by construction and toggleable, which is what makes the G5.3 gauntlet " +
    "meaningful: an engine configuration only ships if it beats its own frozen floor " +
    "on the FIT window."));
  out.push(...figure("fig6_buffer.png",
    "Figure 4.1 — Turnover (left) and FIT net Sharpe (right), no-buffer vs Carver-buffered, " +
    "per CPU model. The buffer is the engine's cost of doing business.", 500));

  // ================================================================ 5
  out.push(h1("5. Phase 3 — The Neural and Contractive-Loop Tournament (K3)"));
  out.push(body(
    "Phase 3 ran the neural tournament in three kernel generations on the ranigain T4x2 " +
    "budget: the compact run (M3c DAE, M3m search-only, M4 DMN-Lite under fp32 repair), " +
    "the completion run (M3m folds, M4 completion, M5 ContinuumNet folds 2020-2022), " +
    "and the m5-completion run that finished M5's 2023-2025 folds and trained the M4b " +
    "variant. All models target the same y_ret24_norm with the fused DeePMRobustSharpe " +
    "loss family; the ContinuumNet M5 carries the LoopBlock with a Kronecker-factored " +
    "Momentum damped by a governor, and its contraction is audited every fold by JVP " +
    "power iteration on the Jacobian's spectral radius."));
  const rhoRows = D.k3rho.map(r => [
    String(r.fold), fmt(r.rho, 4), String(r.km_alpha), safe(r.governor),
  ]);
  out.push(tableTitle("Table 5.1 — M5 LoopBlock spectral-radius audit by walk-forward fold."));
  out.push(mkTable(["Test-year fold", "rho (JVP)", "KM alpha", "Governor"], rhoRows, [25, 25, 25, 25]));
  out.push(cap("Acceptance gate rho < 0.95 every fold. Observed range 0.5012-0.5023 across all " +
    "ten folds 2016-2025: the loop is deeply contractive, an order of magnitude inside " +
    "the gate."));
  out.push(...figure("fig7_spectral.png",
    "Figure 5.1 — The audited spectral radius per fold against the 0.95 acceptance line.", 430));
  out.push(body(
    "The tournament's honest verdict is negative, and the evidence is threefold. " +
    "First, the denoising autoencoders: M3c reaches an OOS engine Sharpe of 0.99 and " +
    "M3m is negative (-1.20 frozen) — around the M1 benchmark but with nothing " +
    "commanding. Second, the DMN family: M4 and its M4b variant both land near -1.08 " +
    "OOS with coarse 144-level outputs (z-standard deviations of 0.17-0.19), triple " +
    "confirmed across the compact, completion and m5-completion runs. Third, the " +
    "ContinuumNet itself: the architecture the blueprints most wanted to work — " +
    "stable, contractive, honest — produces an OOS Sharpe of -0.45 with perfectly " +
    "graded forecasts (16,988 distinct OOS values, std 0.26). The loop-depth analysis " +
    "below shows the same story the v1.3 program already knew: additional iterations " +
    "do not create signal that the first pass does not find."));
  out.push(...figure("fig4_loop_depth.png",
    "Figure 5.2 — OOS Sharpe by loop iterate r=1..4 under the v2.0.1 causal accounting, " +
    "against the v1.3 unconstrained-loop expectation rescaled. Iteration depth adds " +
    "nothing; contraction made the loop safe but not profitable.", 440));
  out.push(body(
    "This is the correct scientific outcome of a tournament: the v2.0 neural program " +
    "spent its GPU budget, respected its stability gates, and lost to a three-seed " +
    "gradient-boosted ensemble on CPU. The autopsy matters because it closes the " +
    "question honestly — no neural model enters the shipped stack, and the GPU budget " +
    "is not spent chasing an architecture whose failure is now documented from three " +
    "independent directions."));

  // ================================================================ 6
  out.push(h1("6. Engineering Flaw #8 — The M2 RFF Fold-Scale Saturation"));
  out.push(body(
    "The final K6 audit added the eighth entry to the program's engineering-flaw " +
    "ledger, and it revises the reading of Phase 2's headline model. The trigger was " +
    "the new G4.8 z-quality audit, which checks every strategy's forecast column for " +
    "degeneracy (value diversity, dispersion, modal concentration) on both windows. " +
    "M2_RFF_Ridge failed it absolutely: its OOS z takes exactly one value (+1.0) on " +
    "all 16,991 bars of 2023-2025, and its FIT window is barely better — per-year, " +
    "the column is a sequence of per-fold constants."));
  const byyear = jload(path.join(K6, "z_quality_m2_byyear.json"), { by_year: [] });
  const yrRows = byyear.by_year.map(r => [
    String(r.year),
    String(r.z_nunique), fmt(r.z_std, 3), fmt(r.z_mean, 2),
    r.pred_std.toExponential(1), r.pred_mean.toExponential(2),
  ]);
  out.push(tableTitle("Table 6.1 — The M2 forecast column, year by year (the saturation fingerprint)."));
  out.push(mkTable(
    ["Year", "z distinct values", "z std", "z mean", "pred std", "pred mean"],
    yrRows, [12, 20, 14, 14, 20, 20], { cSize: 17 }));
  out.push(cap("Only the 2018 fold produced a graded z (std 0.29). Every other year collapses to a " +
    "constant sign: 2017 short, all other years long. The raw predictions are also " +
    "near-constant within each fold — the model is a per-year drift bet, not a per-bar signal."));
  out.push(h2("6.1 Root cause"));
  out.push(body(
    "The K2 export pipeline normalizes each fold's predictions with a fold-local " +
    "scale: z = tanh(pred / std(pred on the early-stop slice)). The design intent was " +
    "sound — vol-normalized return predictions are tiny, and without a scale the " +
    "position collapses toward zero. The defect is that the early-stop-slice " +
    "prediction standard deviation can be orders of magnitude smaller than the " +
    "test-fold prediction magnitude (down to the 1e-9 epsilon guard when the slice " +
    "predictions are constant). The ratio pred/scale then explodes past tanh's " +
    "saturation point (argument ~19), and z collapses to sign(pred) — a per-fold " +
    "directional bet. K2's coverage gate (G2.1b: 'RFF covers 58,504 bars') counted " +
    "covered rows, not value diversity, so a fully saturated column passed every " +
    "existing gate. The 68% turnover reduction that made M2 look like an efficiency " +
    "breakthrough was the same artifact: a per-year bet barely trades."));
  out.push(h2("6.2 Impact propagation and quarantine"));
  out.push(body(
    "The saturation propagated through the composites exactly as the pipeline was " +
    "wired: the M6 stack's positive-FIT membership filter (on the unbuffered " +
    "frozen_net FIT Sharpe) concentrated 100% weight on M2 — dropping M1_GBDT_v2, " +
    "the one CPU model with a genuine 0.94 OOS signal, for a -0.24 FIT estimate — " +
    "so M6, M6b and the OOS stream of the v2.0.1 M7 Grand Hybrid's v2 leg were all " +
    "the passive long-gold sleeve in disguise. Their OOS Sharpe (2.16) was " +
    "numerically identical to Buy & Hold's, which is itself the tell that surfaced " +
    "the flaw. The v2.0.1 G4.8 audit quarantined the four affected strategies while " +
    "still computing and exporting their results verbatim; nothing was withdrawn, " +
    "and the annotation traveled with every number. This report's v2.0.2 numbers " +
    "replace that quarantine with the repair itself: the column is rebuilt (Section " +
    "6.3), the composites are re-derived on the healthy inputs (Section 7), and the " +
    "downstream audit now flags nothing."));
  out.push(h2("6.3 Remediation — executed in v2.0.2"));
  out.push(body(
    "The remediation was executed in full at code revision v2.0.2, and this section " +
    "reports it as a completed change with its measured consequences. The position " +
    "transform in K2 was replaced by a causal trailing standardization of the raw " +
    "prediction: z_t = tanh( (p_t − EWMA_504(p)_t) / (2 · max(EWSTD_504(p)_t, 1e-9)) ), " +
    "re-anchored at each yearly re-fit boundary — the walk-forward replaces the " +
    "model, so its prediction scale is a new random variable, and carrying the dead " +
    "model's mean and standard deviation across the boundary would inject the very " +
    "drift artifact being removed. The first 63 bars of every fold emit no " +
    "position by design (no trailing history from the new model yet). The model " +
    "fit itself is " +
    "unchanged — pred_rff is byte-identical to the published run — so the forecast " +
    "dataset versioned cleanly and the data-signature checkpoints invalidated " +
    "exactly M2, M6, M6b and the M7 composites downstream."));
  out.push(body(
    "The saturation gate moved to the producer. K2 now enforces G4.8 at export " +
    "time — std(z) > 0.05 and P(|z| > 0.99) < 5% on both windows, plus a genuinely-" +
    "active OOS turnover floor above 2 round-turns per year — and a negative-control " +
    "variant (fold mean retained, robust expanding IQR scale) is exported alongside " +
    "to document why any fold-local scaling must fail under a drift/dispersion ratio " +
    "far from one."));
  const g48 = (D.k2meta && D.k2meta.audit_g48) || {};
  out.push(tableTitle("Table 6.2 — The G4.8 producer gate on the rebuilt z_rff (v2.0.2)."));
  const g48rows = ["fit", "oos"].filter(w => g48[w]).map(w => [
    { t: w === "fit" ? "FIT 2016-22" : "OOS 2023-25", left: 1 },
    String(g48[w].n), String(g48[w].nunique), fmt(g48[w].std, 3),
    String(g48[w].sat_frac_gt_0p99), String(g48[w].frozen_turnover_py),
  ]);
  if (g48rows.length) {
    out.push(mkTable(
      ["Window", "Bars", "Distinct z", "std(z)", "P(|z|>0.99)", "Frozen turn/yr"],
      g48rows, [20, 15, 16, 14, 18, 17], { cSize: 17 }));
    const negc = D.k2meta.negative_control_foldiqr || {};
    out.push(cap("The rebuilt column is genuinely active on every axis. The negative-control " +
      "IQR variant (fold mean retained) saturates exactly as predicted — " +
      (negc.fit ? Math.round(negc.fit.sat_frac_gt_0p99 * 100) + "%" : "—") + " of FIT and " +
      (negc.oos ? Math.round(negc.oos.sat_frac_gt_0p99 * 100) + "%" : "—") +
      " of OOS bars above |z| = 0.99, one distinct OOS value — the documented fate of " +
      "any fold-local scale under drift >> dispersion."));
  }
  out.push(...figure("fig11_z_desaturation.png",
    "Figure 6.1 — The Engineering Flaw #8 root cause and its fix in one picture. Top: " +
    "fold-level prediction drift (mu) versus within-fold dispersion (sigma), up to " +
    "100x apart on a log scale — the regime in which fold-local scaling collapses. " +
    "Middle: the v2.0.1 saturated z — per-fold sign bets, one unique OOS value. " +
    "Bottom: the v2.0.2 causal trailing standardization — genuinely active, " +
    "G4.8-gated at the producer.", 470));
  out.push(bodyRuns([
    { text: "The de-saturated measurement is the remediation's scientific payoff, and " +
            "it is negative: " },
    { text: "the honest active M2 signal is noise. ", bold: true },
    { text: "Engine FIT Sharpe -0.736, engine OOS Sharpe -0.744 — the heavy-ridge " +
            "RFF model's per-bar oscillations do not predict the next 24 hours of " +
            "gold. What the pre-fix M2 actually was, the audit now shows precisely: " +
            "a per-year directional bet (short 2017, long 2019-2025 except the graded " +
            "2018 fold) that happened to be long through a historic bull market. " +
            "Every pre-fix number built on that column — the K2-vintage 1.87, the " +
            "v2.0.1 tournament's 2.16, and the v2 leg of the old M7 headline — was " +
            "passive gold exposure attributed to a model. The v2.0.2 tournament " +
            "carries the corrected numbers, and the composites were rebuilt on top " +
            "of them (Section 7)." },
  ]));
  out.push(bullet(
    "Gate, then trust — DONE: the G4.8 saturation gate now runs inside K2 at export " +
    "time (std, saturation fraction, turnover floor), with the K6 downstream audit " +
    "as the independent second layer. The rebuilt column passes every check on both " +
    "windows, and the downstream audit now flags nothing."));
  out.push(bullet(
    "Robust scaling — DONE: the fold-local early-stop scale is replaced by the " +
    "causal trailing EWMA/EWSTD standardization, re-anchored per re-fit fold, with " +
    "an epsilon floor for degenerate constant stretches (which now map to z = 0, no " +
    "position, rather than a sign bet)."));
  out.push(bullet(
    "Re-run and re-issue — DONE: the forecasts shipped as aurum-v2-cpu-preds v3 " +
    "(pred_rff byte-identical, z_rff rebuilt, negative-control column included for " +
    "the record); the data-signature checkpoints invalidated exactly M2/M6/M6b/M7 " +
    "in K6 and the full re-evaluation ran in three and a half minutes."));
  out.push(bullet(
    "The verdict on M2's prior numbers — STATED: M2 must be read as what the audit " +
    "shows: a per-year gold drift bet that happened to be long through the 2019-2025 " +
    "bull market. Its K2 OOS 1.87 and every composite built on it inherit that " +
    "reading, and the de-saturated truth is that the RFF timing approach carries no " +
    "per-bar edge on this panel at this frequency."));
  return out;
}

module.exports = { build };
