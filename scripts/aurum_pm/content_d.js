"use strict";
// Section 6: Spec v1.2 delta, T4 re-validation list, Phase-2 plan
const S = [];

S.push({ h1: "6. Specification v1.2 Delta and the Phase-2 Execution Plan", blocks: [
  { t: "h2", x: "6.1 Consolidated spec changes, each with its evidence" },
  { t: "table",
    caption: "Table 8: Spec v1.1 -> v1.2 deltas (all evidence local unless marked)",
    headers: ["Spec section", "v1.1", "v1.2", "Evidence"],
    widths: [16, 26, 34, 24],
    size: 18,
    rows: [
      ["4.2 / 6 (labels)", "H1 head trains on cohort p_win (B=200 bootstrap)", "H1 trains on realized win + adjacent-cell triangular smoothing (own 1.0 / m,t +-1 0.5 / diag 0.25 / h +-1 0.5); cohort p_win/b_eff enter the state as compact input block; cohort auxiliary loss allowed at weight <= 0.3 behind the 4.2 kill-switch", "Table 3/4: realized AUC 0.523-0.532 vs cohort 0.495; walk-forward +0.0297R vs +0.0070R"],
      ["7 (evaluation)", "Brier/logloss vs realized as primary gate", "Same gates kept, plus mandatory yearly walk-forward table with canonical-long beta and k-NN references, prediction-level neutralization applied, and 2018-style chop years called out", "exp2b Table 4; exp3d neutralization keeps AUC (0.532->0.533, 0.541->0.542)"],
      ["3.4 (G1.4)", "joint adversarial AUC <= 0.55 (hard)", "Three-part gate: (a) single-feature |AUC-0.5| <= 0.15 hard; (b) joint AUC <= within-era reference + 0.05; (c) defense stack verified (walk-forward + neutralization + H5 per-era confusion)", "within-era reference 0.5969-0.7996; qmap+drop8 reaches 0.6166 only by killing signal (AUC 0.533->0.482)"],
      ["5 (macro/priors)", "macro + priors dropped as date fingerprints", "macro returns as inputs via innovation transforms (d20z/d5z/rank60 on raw FRED levels, kernel-side); priors return unchanged; both under the 3.4-c defense", "macro singles 0.47-0.53 (clean), joint 0.9955 (dependence drift -> model-level defense)"],
      ["8 (Stage A')", "full 151,936-vocab LM CE, chunked + checkpointed", "vocab_keep_ids slicing: 97-token frozen tied head, remapped targets, no chunking; masked-state realignment parked as Phase-4 ablation", "97/151,936 measured; CE/grad identity exact; grad cosine >= 0.945 worst-case toy; x1.32-x1.76 projected"],
      ["9 (compute)", "Stage A' ~14 h (based on 4,334 tok/s DDP figure)", "G0.5 accounting fixed (bs=8/GPU counted as 16); Stage A' 4.9-8.2 h on 2xT4 depending on ckpt; Phase 2 total fits 25.8 h remaining quota with >= 1 retry", "DDP bench source inspection; calibrated cost model 5.44-5.57 TFLOP/s"],
      ["14 (discipline)", "(none)", "push protocol: build --verify green + test_local green + metadata lint before any push; kernels edited only in src/lib + src/kernels", "byte-exact roundtrip of all 3 kernels; 40/40 checks in 13 s"],
    ],
  },
  { t: "h2", x: "6.2 The only items that need a T4 (one kernel, 30-40 minutes GPU)" },
  { t: "p", x: "Everything above was verified on CPU; four things genuinely cannot be, and they collapse into a single short re-validation kernel (aurum-p2a-check), pushed once under the new discipline:" },
  { t: "numbered", ref: "refs", items: [
    "Sliced-CE equivalence on real states: run the pilot's 5M-token Stage A' recipe for ~200 steps with the sliced head; report loss trajectory against the pilot's (0.5488 -> 0.3267 over 611 steps) and the kept-mass ratio Z_kept/Z_full on real checkpoints (expected > 0.9 given the pilot's 0.3267 full-vocab loss).",
    "Memory and throughput at the projected configs: r=3, bs=16, sliced, ckpt on and off; bs=32 ckpt on (model says OOM - confirm the boundary); corrected G0.5 DDP bench with honest token accounting.",
    "Head-step cost re-measure with the state-format addition (cohort input block): confirm the ~313-330 tokens/state estimate and the head-step overhead factor (pilot: 982/1138 = 0.86).",
    "GradScaler behaviour with the sliced head: initial scale and overflow dynamics should be strictly friendlier (smaller logit magnitudes); confirm no scaler collapse.",
  ] },
  { t: "p", x: "Budget: well under one hour on 2xT4, leaving the weekly quota essentially intact for the real Phase-2 run." },
  { t: "h2", x: "6.3 Phase-2 execution gates and budget" },
  { t: "bullets", items: [
    "Stage A' (sliced CE, 70M tokens, 3 epochs over 2012-2025 states): 4.9-8.2 h on 2xT4. Gate: loss reduction >= 30% (pilot achieved 40.5%), zero NaN skips, clamp rate < 5%.",
    "Stage B (joint head on realized + smoothed targets, cohort inputs, aux weight 0.3 with kill-switch at first checkpoint): 2.5-4 h. Gate: realized val-era AUC >= 0.53 on the 8-cell panel (the local XGBoost ceiling is 0.532 - the transformer must at least match the cheap model to justify itself), EV-argmax walk-forward meanR >= +0.02R with neutralization applied.",
    "Probability tables + walk-forward evaluation: ~1 h (G0.6 measured 62.6 rows/s on 2 GPUs; 88k bars x 3 refits ~ 45 min).",
    "Total: 8.5-13 h, one retry fits the remaining 25 h 50 m; two retries fit if the optimistic throughput bound holds.",
  ] },
  { t: "h2", x: "6.4 Risk register updates" },
  { t: "p", x: "Three risks are re-rated by this pass. The cohort memorization trap is closed as a design flaw but its residue becomes an evaluation risk: the +0.03R/year class of signal is small, so Phase-2 success is defined against the references (canonical beta, k-NN, XGBoost ceiling), not against zero - a Phase-2 that merely matches the XGBoost ceiling must trigger the Spec's model-class honesty clause. The era-drift risk is re-framed from 'pass a 0.55 gate' to 'bound the drift budget and neutralize', with the measured evidence that the signal is orthogonal to the drift direction. And the compute risk improves materially: the corrected accounting plus vocabulary slicing move Phase 2 from 'fits with little headroom' to 'fits with retries', which matters because the 2018-style regime year is exactly the case where a single training run's variance will tempt over-interpretation - the budget for reruns is the real insurance.", opts: { after: 200 } },
]});

module.exports = S;
