"use strict";
// ============================================================================
// AURUM Spec v2.0 Master Report — Content Part 3 (rev v2.0.2)
// Sections 7-8: Phase 4 (K6 backtest + statistics), Phase 5 (K7 live)
// ============================================================================
const H = require("./helpers");
const { h1, h2, body, bodyRuns, bullet, figure, mkTable, tableTitle, fmt, safe, P } = H;

function cap(text) {
  const { Paragraph, TextRun, AlignmentType, FONTS } = H;
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 220, line: 280 },
    children: [new TextRun({ text, size: 18, italics: true, color: P.secondary, font: FONTS })] });
}

function build(D) {
  const out = [];
  const S = D.k6bt.strategies || {};
  const cc = D.k6cc;
  const meta = D.k6stats._meta || {};
  const flagged = new Set(D.k6zq.flagged_strategies || []);

  // ================================================================ 7
  out.push(h1("7. Phase 4 — The Full Walk-Forward Backtest and Statistical Gates (K6)"));

  out.push(h2("7.1 The v2.0.1 causality regression"));
  out.push(body(
    "Before any strategy is compared, the kernel proves its own accounting is causal. " +
    "The G4.7 regression deliberately re-creates the original one-bar-lookahead " +
    "formulation on the Channel_A baseline and requires the pathology to be visible: " +
    "same-bar Sharpe 15.51 (OOS) and 14.43 (FIT) — an impossible edge produced purely " +
    "by crediting the close-of-bar position with the same bar's return. The corrected " +
    "causal accounting on the identical series gives -0.13 and -0.51: the channel " +
    "carries no same-bar residual edge once the lookahead is removed. Both conditions " +
    "are gate requirements (the trap must be detectable, and the fix must leave no " +
    "residue), so a future regression in either direction fails the run loudly. This " +
    "discovery retroactively deflated every pre-v2.0.1 frozen-accounting number in the " +
    "program's working notes — most visibly the champion's continuous translation, " +
    "whose OOS figure moved from an inflated working estimate to the verified 2.611 " +
    "reported here."));

  out.push(h2("7.2 The tournament"));
  out.push(body(
    "Eighteen strategies ran through the identical harness: walk-forward forecasts " +
    "frozen at each fold boundary, the single-position causal event loop, the G5.3 " +
    "engine-vs-frozen gauntlet decided per strategy on FIT only, costs at 1.0x/1.5x/" +
    "2.0x (5e-5 per unit exposure change at 1.0x), and the Carver speed-limit audit " +
    "(annual cost drag at most one third of pre-cost Sharpe). The v1.3 champion " +
    "translations are evaluated under frozen accounting only — a pure translation of " +
    "the barrier policy's own walk-forward trades into continuous units, with no v2 " +
    "engine overlay, so the lineage comparison is like-for-like."));
  const order = [
    "M7_Core_Satellite", "M0_Champ_v13", "M0_Champ_v13_cap1",
    "M7_Active_Grand_Hybrid", "M6_AURUM_v2_Ensemble", "M6b_Ensemble_EQW",
    "M1_GBDT_v2", "Carver_EWMAC", "M3c_DAE", "RLSBank", "BuyHold_Gold",
    "M2_RFF_Ridge", "M5_ContinuumNet",
    "Channel_A", "M4_DMN_Lite", "M4b_DMN_Lite", "M3m_DAE", "Channel_B",
  ];
  const demoted = new Set(["M2_RFF_Ridge", "M6_AURUM_v2_Ensemble", "M6b_Ensemble_EQW"]);
  const rows = [];
  for (const name of order) {
    const e = S[name];
    if (!e) continue;
    const f = e.fit || {}, o = e.oos || {};
    const fm = f.metrics || {}, om = o.metrics || {};
    const ref = demoted.has(name);
    rows.push([
      { t: name + (flagged.has(name) ? " *" : ""), left: 1, c: ref ? P.secondary : P.body },
      { t: e.gauntlet_engine ? "engine" : "frozen" },
      fmt(fm.sharpe_ann, 2), fmt(om.sharpe_ann, 2),
      fmt((o.cost1_5 || o["cost1.5"] || {}).sharpe_ann, 2),
      fmt((o.cost2_0 || o["cost2.0"] || {}).sharpe_ann, 2),
      fmt(e.oos.turnover_py, 1),
      { t: fmt(om.maxdd_volunits, 3) },
      { t: o.speed_limit_ok === null || o.speed_limit_ok === undefined ? "—" : (o.speed_limit_ok ? "pass" : "FAIL"),
        c: o.speed_limit_ok ? P.green : P.crimson },
    ]);
  }
  out.push(tableTitle("Table 7.1 — The full K6 tournament grid (FIT 2016-22, OOS 2023-25, net of costs)."));
  out.push(mkTable(
    ["Strategy", "Mode", "FIT SR", "OOS SR", "OOS 1.5x", "OOS 2.0x", "Turn/py", "MaxDD", "Speed"],
    rows, [26, 9, 9, 9, 10, 10, 10, 9, 8], { cSize: 16, hSize: 16 }));
  out.push(cap("Vol-target units; Turn/py = round turns per year on the OOS window; MaxDD in vol units. " +
    "Grey rows are the de-saturated active stack, demoted to reference-only after failing the Carver " +
    "one-third cost budget. Speed is the Carver rule on the OOS window (shipping family gated; " +
    "also-rans reported)."));

  out.push(body(
    "Three readings of the grid matter. The champion lineage is confirmed: the cap=3 " +
    "flagship translation runs 2.61 OOS with the shallowest drawdown of the field " +
    "(0.096 vol units) and the cap=1 variant 2.54 — the continuous accounting agrees " +
    "with the barrier-trade verdict that made this policy the v1.3 champion. The " +
    "engine layer earns its keep exactly where designed: M1's gauntlet ships the " +
    "engine and lifts the model from negative FIT to +0.94 OOS, and the Carver EWMAC " +
    "baseline repeats the pattern (+1.23 OOS through the engine). And the composites " +
    "now behave as their honest ingredients dictate: the rebuilt M6 (50/50 M1 + " +
    "de-saturated M2) runs +0.73 OOS through the engine — M1's real edge diluted by " +
    "M2's measured noise — and the M7 Active Grand Hybrid lands at 1.95, while the " +
    "M7 Core-Satellite preserves the pre-fix headline at 2.64 because that is " +
    "arithmetically what the pre-fix headline always was: half champion, half " +
    "passive gold. The speed-limit column carries the v2.0.2 verdict on the active " +
    "stack: at 226-558 round-turns per year, M2 and the M6 pair burn 2.5-6.2x the " +
    "Carver cost budget and are demoted from the shipping family — a genuinely " +
    "active signal that cannot pay its own costs at H1 frequency."));
  out.push(...figure("fig1_oos_equity.png",
    "Figure 7.1 — OOS 2023-25 cumulative net P&L in vol-target units. The " +
    "Core-Satellite (gold) rides the champion (green) with the passive sleeve's " +
    "smooth drift; the Active Grand Hybrid (crimson) shows what the de-saturated " +
    "stack actually adds — drag, not edge.", 500));
  out.push(...figure("fig2_year_sharpe.png",
    "Figure 7.2 — Year-by-year annualized net Sharpe, FIT to the left of the red line, " +
    "OOS to the right. The champion family is positive in every OOS year; no v2.0 " +
    "model is.", 500));
  out.push(body(
    "The year-by-year view answers the mandate's two stress questions directly. In " +
    "the 2018 chop year the champion translation holds a small positive Sharpe while " +
    "every continuous model except M3c is negative — the barrier structure's " +
    "conditional entries are what survive sideways gold. In the 2023 transition year " +
    "(the first OOS year, and the year the champion's own v1.3 walk-forward had " +
    "never seen) the champion family posts a strong positive while the v2 models " +
    "do not; 2024-2025 extend the same ordering. The mandate hoped the continuous " +
    "program would fix 2018 and 2023; the honest answer is that it did not, and the " +
    "one leg that does move those years — the passive sleeve — does so with beta, " +
    "not skill."));
  out.push(...figure("fig5_cost_survival.png",
    "Figure 7.3 — Cost survival: OOS Sharpe at 1.0x / 1.5x / 2.0x costs. The champion " +
    "family's degradation is the real cost of an active policy; the Core-Satellite's " +
    "flatness is the passive sleeve barely trading; the de-saturated M2's collapse " +
    "is the honest cost of a fast, empty signal.", 470));
  out.push(...figure("fig9_drawdown.png",
    "Figure 7.4 — OOS drawdown paths. The Core-Satellite's deeper path (0.131 vol " +
    "units vs the champion's 0.096) is the passive sleeve's gold beta showing " +
    "through; the Active hybrid tracks the champion with the stack's added noise.", 470));

  out.push(h2("7.3 The two Grand Hybrids — active vs core-satellite"));
  out.push(body(
    "v2.0.2 replaces the single M7 with two fixed, selection-free 50/50 benchmarks, " +
    "both chosen a priori and both shipped against their FIT-grid optimum (the " +
    "diagnostic grid, recorded in the ledger, peaks at w_v2 = 0.00 — champion only, " +
    "FIT Sharpe 2.01 — on the active arm). The M7 Active Grand Hybrid is the design " +
    "the program always intended: half champion exposure, half M6 stack over " +
    "{M1_GBDT_v2, de-saturated M2_RFF_Ridge}. Measured for the first time on " +
    "genuinely active inputs it runs 1.95 OOS (1.63 at 2.0x costs) against the " +
    "champion's own 2.61 — the true active orthogonality gain of the v2.0 continuous " +
    "stack is negative, −0.67 Sharpe at the 50% weight. The M7 Core-Satellite is " +
    "the transparent decomposition of what the v2.0.1 headline actually was: half " +
    "champion, half passive gold sleeve (the saturated v2 leg was bit-identical to " +
    "Buy & Hold in OOS, so this arm reproduces the 2.6351 / 2.5012-at-2.0x stream " +
    "to the last digit). It tops the field because 2023-2025 gold drift pays the " +
    "passive leg while the champion leg supplies the active edge — a real portfolio " +
    "result about regime beta, not a modeling edge, and one whose statistical " +
    "certainty is now measured honestly in Section 7.4."));
  out.push(body(
    "The M6 stack itself was rebuilt to the specification this program should have " +
    "used from the start. The family is fixed a priori — {M1, M2}, no FIT-window " +
    "membership filter (the v2.0.1 filter dropped M1, the one model with a genuine " +
    "OOS edge, on a -0.24 unbuffered FIT estimate); the Sharpe inputs are the " +
    "buffered engine FIT estimates from the gauntlet, the selection-relevant metric; " +
    "and the estimates are shrunk 90% toward the equal-weight prior per Carver's " +
    "rule. With both engine-FIT estimates negative (M1 -0.13, M2 -0.74), the shrunk " +
    "mean-variance solution cannot rank the members and the equal-weight prior ships " +
    "unchanged: M6 is a genuinely diversified 50/50 active blend, its +0.73 OOS " +
    "being M1's real 0.94 diluted by M2's measured noise."));

  out.push(h2("7.4 Statistics — bootstrap, DSR, PBO, and one honest failed gate"));
  const sens = meta.dsr_sensitivity_m7 || {};
  out.push(body(
    "Uncertainty is quantified three ways. Block bootstrap (5,000 resamples, 24-bar " +
    "non-overlapping blocks matching the holding horizon) puts the Core-Satellite's " +
    "OOS Sharpe at 2.64 with a 95% CI of [1.49, 3.77] — wide, as an honest " +
    "three-year window on one instrument must be, and comfortably excluding zero. " +
    "The Deflated Sharpe Ratio follows Bailey & Lopez de Prado exactly: per-bar " +
    "Sharpe against a hurdle built from the variance of trial Sharpes and the " +
    "number of effective independent trials. The v2.0.2 trial family counts the " +
    "signal families plus one blend slot — the fixed a-priori portfolio " +
    "constructions over counted members are correlated re-weightings, not " +
    "independent attempts, so they contribute a single trial at their best-FIT " +
    "variant (N=12); a conservative accounting that counts every distinct blend " +
    "stream is exported alongside (N=14). The CSCV probability of backtest " +
    "overfitting over the FIT stream family is 0.0049."));
  out.push(tableTitle("Table 7.2 — Deflated Sharpe Ratio, OOS 2023-25 (primary family N=12; conservative N=14 in grey)."));
  const dsrRows = [];
  const dsrOrder = ["M7_Core_Satellite", "M0_Champ_v13", "M0_Champ_v13_cap1",
    "M7_Active_Grand_Hybrid", "BuyHold_Gold", "Carver_EWMAC", "M6_AURUM_v2_Ensemble",
    "M1_GBDT_v2", "M3c_DAE", "RLSBank", "M2_RFF_Ridge", "M5_ContinuumNet"];
  for (const k of dsrOrder) {
    const v = D.k6stats[k];
    if (!v || v.oos_dsr === undefined) continue;
    dsrRows.push([
      { t: k, left: 1, c: demoted.has(k) ? P.secondary : P.body },
      { t: fmt(v.oos_dsr, 3), c: v.oos_dsr >= 0.95 ? P.green : P.secondary },
      { t: fmt(v.oos_dsr_conservative, 3), c: P.secondary },
      v.dsr_passes_95 ? "pass" : "—",
    ]);
  }
  out.push(mkTable(["Strategy", "DSR (primary)", "DSR (conservative)", ">= 0.95"],
    dsrRows, [40, 20, 22, 18]));
  out.push(cap("No strategy clears the pre-registered 0.95 admission bar under the honest " +
    "v2.0.2 panel — the closest is the Core-Satellite flagship at 0.941 primary / " +
    "0.925 conservative, with the champion at 0.935. The v2.0.1 certification of " +
    "0.958 was measured on a panel whose M2 trial was the saturated artifact " +
    "(+0.35 FIT); replacing it with the true -0.84 widens the trial dispersion and " +
    "raises the Bailey-LdP null threshold. The 95% claim is withdrawn and the gate " +
    "fails, as pre-registered."));
  out.push(tableTitle("Table 7.3 — DSR sensitivity of the flagship to the trial-family definition."));
  out.push(mkTable(
    ["Trial family", "N", "Flagship DSR"],
    [
      [{ t: "Primary: signal families + one blend slot", left: 1 },
       String(sens.family_N ? sens.family_N.primary : meta.dsr_family_N), fmt(sens.primary, 3)],
      [{ t: "Conservative: every distinct blend stream counts", left: 1 },
       String(sens.family_N ? sens.family_N.conservative : "—"), fmt(sens.conservative, 3)],
      [{ t: "+ M7 blend grid as independent trials (diagnostic only)", left: 1 },
       String(sens.family_N ? sens.family_N.with_grid : "—"), fmt(sens.with_m7_grid, 3)],
      [{ t: "+ K2 ledger rows (pre-fix accounting vintage)", left: 1 },
       String(sens.family_N ? sens.family_N.with_k2 : "—"), fmt(sens.with_k2_prefix, 3)],
      [{ t: "All ledger rows, mixed units (not methodologically valid)", left: 1 },
       String(sens.family_N ? sens.family_N.all : "—"), fmt(sens.all_ledger_mixed_units, 3)],
    ], [56, 14, 30], { cSize: 17 }));
  out.push(cap("The full sensitivity ladder is disclosed rather than cherry-picked; the mixed-unit " +
    "row is shown only to document why unit-consistency is required. Under every " +
    "defensible accounting the flagship sits between 0.93 and 0.94 — borderline, " +
    "not certified."));
  out.push(...figure("fig8_dsr.png",
    "Figure 7.5 — DSR by strategy against the 0.95 admission gate: primary family " +
    "(bars) and conservative full panel (diamonds), with the trial count and PBO " +
    "in the title.", 470));
  out.push(body(
    "The gate register closes the phase with the program's first deliberately " +
    "reported failure: G4.7 causality regression PASS, G4.8 z-quality audit PASS " +
    "(producer-gated since v2.0.2; zero strategies flagged), G5.3 engine gauntlet " +
    "PASS (18 decisions recorded), G5.4 Carver speed limit PASS on the pruned " +
    "shipping family (the demoted active stack's FAIL flags exported verbatim as " +
    "reference), G6.2 stack construction PASS (fixed family, engine-FIT inputs, " +
    "90% shrink toward equal weights), G6.6 export PASS — and G6.1 statistics " +
    "FAIL: flagship DSR 0.941 against the pre-registered 0.95 bar, PBO 0.0049 " +
    "well inside its gate. G6_ALL therefore reads FAIL, by design of the gate and " +
    "choice of this report: the alternative — widening the bar after seeing the " +
    "number, or counting only the accounting vintage that passes — is exactly the " +
    "practice the DSR exists to prevent. The kernel's complete log, per-strategy " +
    "diagnostics, net series and the trial ledger ship alongside this report in " +
    "the artifacts bundle."));

  // ================================================================ 8
  out.push(h1("8. Phase 5 — The Live v2.0 Decision System (K7)"));
  const dec = D.k7decision || {};
  const v2s = dec.v2_state || {};
  out.push(body(
    "The K7 kernel turns the pipeline into a daily-recomputable decision object: every " +
    "field is derived from versioned datasets and the same code that produced the " +
    "backtest, so the live decision is auditable to the bar. At the as-of bar " +
    safe(dec.as_of_bar) + ", the system reports the stack target exposure z = " +
    fmt(v2s.z_target_exposure, 4) + " with the executed position at z = " +
    fmt(v2s.z_executed, 4) + " inside the Carver buffer (band " +
    fmt((v2s.carver_buffer || {}).band, 4) + ", " +
    safe((v2s.carver_buffer || {}).inside_band) + "), under a regime multiplier of " +
    fmt(((v2s.regime || {}).vol_target_mult), 3) + " from the volatility target " +
    "(vol-of-vol cut " + safe((v2s.regime || {}).vol_of_vol_mult) + ", drawdown " +
    "governor " + fmt((v2s.regime || {}).dd_governor_scale, 2) + ", PH freeze " +
    safe((v2s.regime || {}).ph_frozen) + "). The catastrophic-stop frame sits at 2.5x " +
    "ATR14 (" + fmt((v2s.catastrophic_stop || {}).atr14, 1) + " points), and the " +
    "cost budget reports " + fmt((v2s.cost_budget || {}).recent_turnover_py, 1) +
    " round turns per year against the one-third speed limit audited in K6."));
  out.push(body(
    "The v2.0.2 re-run embeds the tournament verdict directly in the decision " +
    "object (stack_context_v202): the live model is M1_GBDT_v2 — the only active " +
    "CPU model that passes the Carver speed limit — with the de-saturated M2 " +
    "annotated as measured noise (reference-only), the M6 family recorded at its " +
    "shipped 50/50 weights, and both M7 benchmarks carried with their OOS Sharpe, " +
    "2.0x-cost and DSR fields for the desk's morning read. The kernel's live gates " +
    "all pass, and the recommendation that follows from this report is now " +
    "measured rather than provisional: the v2.0 active stack does not add edge on " +
    "top of the champion, the champion's barrier reference remains the operational " +
    "policy, and the Core-Satellite is the only v2-era configuration worth " +
    "considering as an overlay — with its borderline DSR stated alongside."));
  return out;
}

module.exports = { build };
