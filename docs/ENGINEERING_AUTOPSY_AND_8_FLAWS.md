# Engineering Autopsy — The 8 Quantitative Trap Doors of Project AURUM

> This document is the complete record of every material flaw caught during Project AURUM
> (Spec v1.0 → v2.0.2), with the exact math, the code-level fix, and the before/after
> numbers. Each flaw is also regression-gated so it can never silently return. Flaws 1-5
> belong to the v1.x lineage (looped-transformer world model); Flaws 6-8 to the v2 lineage
> (continuous engine). Sources: `kaggle_kernels/`, `aurum_v2/`, `artifacts/`, the six master
> reports in `reports/`, and `docs/EXECUTION_JOURNAL.md`.
>
> **Meta-lesson:** all eight flaws share one shape — *an information boundary was crossed by
> an innocent-looking convenience* (a `shift(-1)`, a fold-local scale, an ex-post overlay, a
> same-bar product). The defense that actually worked was not cleverness but **gates +
> ledgers + pre-registration**: every trap below is now a machine-checked gate (G-series) or
> a structural property of the engine, not a convention.

---

## Flaw 1 — Nominal price/date fingerprinting (the "AUC 1.000 model")

**Where:** v1.0/v1.1 data pipeline (`v1_3/kernels/p1_data.py`, serialization design).
**Symptom:** early cohort classifiers showed perfect discrimination (AUC = 1.000) between
training eras — the model was not learning market state, it was **fingerprinting the
calendar**. Nominal price levels (gold at $1,100 in 2010 vs $2,600 in 2025) and absolute
timestamps uniquely identify the era; any learner given them will "cheat" by era-matching
and then fail catastrophically in any new era.

**Fix (three layers, v1.1→v1.2):**
1. **Scale-invariance contract** — every feature is a function of *returns/spreads/ratios*,
   never raw price; enforced structurally by the serialization (v1.3 codebooks bucket
   standardized innovations, not levels; G0.3 gate verified scale-invariance to 2.04e-10 on
   features / 2.24e-13 on labels, including a ×137 price rescale of the real XAUUSD-2024
   sample).
2. **Macro innovations** — external information enters only as *innovations* (surprises),
   never levels: the `mi_*` FRED family (v2: 15 features, e.g. `mi_vix_d5z`, `mi_y2_d5z`).
3. **Era neutralization** — the `era_clf.json` audit + neutralization spec
   (`artifacts/` v1.x data: `neutralization_spec.json`, `era_clf.json`, 1.05 MB) verify the
   serialized state carries no era signal above chance.

**After:** era classifier AUC dropped to chance; OOS degradation became attributable to
regime, not memorized calendars.

---

## Flaw 2 — K=200 k-NN cohort soft-label memorization

**Where:** v1.1 label construction (cohort states → soft barrier-label targets). Kernels:
`v1_3/local_experiments/exp1_memorization.py`, `exp2_targets.py`.

**Symptom:** k-nearest-neighbour lookup over cohort states (K=200) produced "soft labels"
with a **soft Brier score of 0.0022** — implausibly perfect. Realized (out-of-sample)
Brier on the same barriers was **0.247**, a 100× gap. The k-NN was not smoothing noise, it
was **retrieving the training bar itself** (or its near-duplicates) and copying its outcome:
soft-label leakage.

**Fix:** soft labels were replaced by **triangular-smoothed realized targets** — the label
for a barrier is a causal, forward-looking function of realized price paths only
(`labels_cls_smoothed.parquet` in the phase-1 data; the smoothing kernel has no access to
neighbor identities). The experiment suite (`exp2`, `exp2b`) re-validated that train/OOS
Brier gaps collapsed to the statistically expected range.

**Gate:** cohort statistics + memorization audit (`cohort_std_stats.csv`, audit.json in the
phase-1 data release asset); the v1.2 postmortem report documents the discovery.

---

## Flaw 3 — The 151,936-vocab bottleneck

**Where:** v1.1/v1.2 tokenizer (market-state serialization). `v1_3/local_experiments/
exp4_vocab.py`, `exp4b_toy_ce.py`.

**Symptom:** the original serialization vocabulary had **151,936 distinct tokens**. A
6-layer decoder over 151k classes spends nearly all capacity modeling token identity, not
dynamics; training loss stalled and the sliced cross-entropy was dominated by the head.

**Fix:** the **97/111-token sliced vocabulary** — an *exact slicing* of the full softmax:
only the tokens that actually occur in the market-state language are kept, via
`vocab_keep_ids.npy` (1,016 bytes — the literal keep-list; also in `data/`). The sliced
cross-entropy objective computes loss only over kept ids with a KL-anchor leash to the full
distribution. Reconstruction fidelity **Z = 0.9977** (kept tokens explain 99.77% of the
probability mass).

**Result:** the looped core trained to convergence within Kaggle GPU budget; the same
tokenizer powers the v1.3 champion and (conceptually) the v2 label columns
(`y_ret24_norm`, `y_dir24`, `y_ret1_norm` in the v2 feature store).

---

## Flaw 4 — The v1.2 `H3` ex-post excursion overlay

**Where:** v1.2 backtest decision layer (H3 = MFE/MAE excursion head).
**Kernels:** `kaggle_kernels/v1_3/aurum-p4-backtest.py` (v1.2 lineage), ablation preserved
in `artifacts/v1_3/p4_backtest/overlay_ablation.json`.

**Symptom:** the H3 head's "path overlay" used the **ex-post realized excursion** (the
maximum favorable excursion actually achieved by the trade) to improve the exit decision —
i.e. the backtest was allowed to know how far the trade ran before deciding whether to hold
it. The overlay credited **+1.08R of synthetic expectancy**, and the GBDT audit block
appeared to add **+0.63R** on top of a weak base.

**Fix:** the causal **entry-order path walk** — the backtest replays the barrier order book
in strict timestamp order; the exit decision at bar *t* may use only H3 *quantile
forecasts* made at bars ≤ *t* (q50/q25 pinball heads), never realized excursions. The
`overlay_ablation.json` artifact records the before/after; the honest v1.3 GBDT parity
block contributes +0.0310R (vs the bogus +0.0599R), and the k-NN block −0.0155R — both
reported in the v1.3 master report.

**Gate:** G5.x family in `aurum-p4-backtest` (v1.3, 12 gates all green) — any overlay must
prove causality on a synthetic trap case before entering the backtest.

---

## Flaw 5 — Unbounded hourly position stacking

**Where:** v1.2→v1.3 execution layer (portfolio construction on H1 signals).

**Symptom:** with hourly signal updates and no position-resolution lock, overlapping
signals **stacked pyramids**: 7,207 trades with a **127.7R maximum drawdown** — the
"strategy" was an accidental leveraged trend bet whose equity curve had nothing to do with
the per-trade edge being measured.

**Fix (v1.3, two components):**
1. **Resolution-locked engine, cap=1 / cap=3** — a position resolves (fully or up to the
   concurrent-position cap) before its slot can be reused; non-overlapping portfolio
   accounting with 1% equity risk per trade and Quarter-Kelly sizing (k=0.25, H=48 bars).
2. **Carver ±10% trade-to-edge inertia buffer** (kept into v2 as the engine's buffer): a
   pending trade only executes if its edge exceeds the incumbent trade's edge by ±10% —
   the same buffer that in v2.0.2 cuts turnover by 32.0% (M1) / 38.6% (M2) at zero Sharpe
   cost (gate G2.3 / G5.3 engine-vs-frozen gauntlet).

**Result:** the v1.3 flagship (cap=3) measures 3,341 trades in the v2 translated frame,
OOS Sharpe 2.444, MaxDD 13.06%, DSR 0.895; in its native frame, cap=3 @1.0× cost:
9,819 trades, +1,160.7R total, 33.0% CAGR.

---

## Flaw 6 — Unconstrained loop expansion (the deeper-the-worse paradox)

**Where:** v1.x looped-transformer training (recurrent depth r ∈ {1,2,3} over looping
layers 13-16 of Qwen3-0.6B).

**Symptom:** validation loss ordered **r=1 > r=2 > r=3** (deeper looping strictly worse) —
the opposite of the "looped transformers get better with depth" literature. Diagnosis: the
unconstrained recurrent state accumulated spectral mass each pass; with 63M unfrozen
recurrent params the loop amplified its own drift and the residual stream exploded
logarithmically (residual clamp C=16384 fired constantly at r=3).

**Fix (v2, structural — ContinuumNet `aurum_v2/model/continuum.py`):**
1. **Non-residual spectral-capped loop:** the iterate contracts toward its fixed point,
   with the one-step Jacobian's spectral norm capped at **σ ≤ 0.80** by construction
   (projected weight reparametrization).
2. **KM damping:** Kilian-Mueller-style adaptive damping on the loop step.
3. **JVP governor:** the spectral radius ρ is *estimated online* via Jacobian-vector
   products and the damping auto-tunes (fold logs: `km_alpha=0.5`, governor `auto_tune_alpha`).

**Measured:** ρ = 0.5012 → 0.5014 across **all ten walk-forward folds 2016-2025**
(`artifacts/v2/k3_neural/rho_log.json`), always below the 0.95 gate. **And the honest
verdict:** M5 ContinuumNet still delivers **−0.45 OOS Sharpe** — the loop is now provably
stable, and the signal simply is not there. This is the program's cleanest "we fixed the
machinery, and the machinery then told us the truth" result.

---

## Flaw 7 — The v2.0.1 `frozen_net` same-bar lookahead (`z[t]·r[t]` vs `z[t-1]·r[t]`)

**Where:** v2.0.1 K6 accounting of the frozen (no-engine) model stream.

**Symptom:** the frozen-model P&L credited the position formed at the close of bar *t* with
the return of bar *t* itself — a one-bar lookahead. On the **Channel_A** baseline the
same-bar formulation shows an impossible **Sharpe of 15.51 (OOS) / 14.43 (FIT)**; the
corrected causal accounting gives **−0.13 (OOS) / −0.51 (FIT)**.

**The code shape of the bug:**
```python
# BEFORE (v2.0.1) — same-bar: earns the return that produced the signal
pnl = z.shift(0).mul(r)            # z[t] * r[t]

# AFTER (v2.0.2) — causal: a position at the close of t earns t+1
pnl = z.shift(1).mul(r)            # z[t-1] * r[t]
```

**Fix + gate:** all engine/frozen accounting paths route through the causal convention;
**G4.7** (`G4.7_causality_v201`) recomputes the same-bar trap Sharpe on every K6 run and
fails the run if it is no longer absurd (the 15.5 must *stay* absurd — if a "fixed" stream
ever shows same-bar Sharpe near it, causality has regressed). Confirmed green in the final
run: `same-bar (lookahead) Sharpe {"fit": 14.429, "oos": 15.508}` detected and the causal
stream reported instead (`artifacts/v2/k6_backtest/gates.json`).

**Fallout:** every v2.0.1 frozen-model Sharpe was restated in v2.0.2; no headline number
in this repo contains same-bar P&L.

---

## Flaw 8 — The v2.0.1 `M2_RFF_Ridge` fold-scale collapse (`z_rff` saturation)

**Where:** v2.0.1 K2 producer — the transform from RFF-Ridge regression predictions
`pred_rff` to positions `z_rff`. The most instructive flaw of the program: it produced a
**2.635 headline that was actually passive gold exposure**.

**Mechanism, step by step:**
1. `pred_rff` are cross-sectional ridge predictions with a heavy penalty (z ≈ 10⁴): 16,991
   unique OOS values but std ≈ 1e-4 — a tiny **positive mean drift μ_fold > 0** with
   micro-oscillations around it.
2. v2.0.1 normalized per fold: `z = tanh(pred / std(pred_in_fold))`. Dividing a
   mean-drifted signal by its *within-fold collapsed scale* makes the ratio ≫ 5 everywhere.
3. `tanh(≫5) → +1.0` constant: **z_rff saturated to +1.0** on 88.2% of FIT bars and **100%
   of OOS bars** (OOS: std 0.0, 1 unique value — see the negative-control audit in
   `artifacts/v2/k2_cpu_preds_v3/gates.json`).
4. A constant +1.0 position is Buy & Hold. So "M2_RFF_Ridge Sharpe 1.87-2.16" was gold
   beta; the v2.0.1 "M7 Grand Hybrid (2.635)" was 50% v1.3 champion + 50% *passive gold*;
   and the M6 ensemble and the DSR panel were contaminated by the same artifact.

**The fix (v2.0.2, producer-side, `aurum_v2/kernels/k2_cpu_models.py` → `rff_z_v202`):**
replace the fold-interior scale with a **causal trailing standardization**:

```python
RFF_Z_SPAN  = 504        # ~1 month of H1 bars, trailing only
ewma  = p.ewm(span=504, min_periods=63).mean()      # center: trailing EWMA
ewstd = p.ewm(span=504, min_periods=63).std()       # scale: trailing EWSTD
z_rff = np.tanh((p - ewma) / (2.0 * np.maximum(ewstd, 1e-9)))
# re-anchored at each yearly fold boundary; pred_rff itself byte-identical
```

i.e. `z_t = tanh((p_t − EWMA₅₀₄(p)_t) / (2·max(EWSTD₅₀₄(p)_t, ε)))` — the de-meaning kills
the fold drift, the trailing window makes it causal, and the ×2 soft-scales into the tanh's
linear region.

**Gate (G4.8, producer-side, runs inside K2 before export):**
- `std(z_rff) > 0.05` — measured: **0.4248 FIT / 0.4159 OOS** ✓
- fraction of `|z| > 0.99` < 5% — measured: **0.0** both windows ✓
- active OOS turnover > 2.0 — measured: **558 round-turns/year** ✓
- plus a **negative control**: the old fold-mean-retained IQR variant is computed alongside
  and must show saturation (it does: OOS std 0.0, 1 unique value) — proving the gate detects
  the disease it was built for.

**After (all numbers restated, v2.0.2):**

| Quantity | v2.0.1 (saturated artifact) | v2.0.2 (de-saturated truth) |
|---|---|---|
| z_rff OOS std / unique values | ~0 / 1 | 0.416 / 16,805 |
| M2_RFF_Ridge OOS Sharpe | +1.87…+2.16 (gold beta) | **−0.744** (noise) |
| M2 DSR | 0.8237 | 0.0 |
| M6 ensemble OOS Sharpe | (contaminated) | 0.728 |
| M7 benchmark | "Grand Hybrid 2.635" | split: **Core-Satellite 2.635** (passive sleeve, transparent) / **Active Grand Hybrid 1.946** (true active claim) |
| DSR panel N | 120 configs | 170 configs (artifact trial +0.3504 → true −0.8394) |
| Flagship DSR | 0.958 "pass" | **0.9407 — pre-registered 0.95 bar MISSED, reported as failed** |

**Downstream verification:** K6 independently re-audits every strategy's z-stream
(`z_quality.json`: nunique, std, top-value share per window; flagged saturated/degenerate:
**none**), and the per-year M2 audit (`z_quality_m2_byyear.json`) confirms no year regresses.
The signature chart `figures/v2_202/fig11_z_desaturation.png` shows before/after scatter.

---

## The gate register that now guards all eight flaws

| Gate | Stage | Guards against | Status in final v2.0.2 run |
|---|---|---|---|
| G0.3 | p0-core | scale/price fingerprinting (Flaw 1) | green (v1.3 chain) |
| G1.1-G1.7 | K1 features | feature-store shape/nan/units/coverage/causality | green |
| G2.1-G2.4 | K2 | fold coverage, RLS floor, engine-vs-frozen gauntlet, export | green |
| **G4.7** | K6 | same-bar P&L regression (Flaw 7) | green (trap detected: 15.51 absurd, as required) |
| **G4.8** | K2 producer + K6 audit | z saturation / fold-scale collapse (Flaw 8) | green (std 0.42, sat 0%, negative control saturated as required) |
| G5.3 | K6 | engine vs frozen (buffer must earn its complexity) | green |
| G5.4 | K6 | Carver speed limit (cost ≤ ⅓ pre-cost SR) | green |
| **G6.1** | K6 | DSR ≥ 0.95 over honest panel; PBO ≤ 0.20 | **FAILED 0.9407 (PBO 0.0049) — reported as failed** |
| G6.2 | K6 | ensemble construction a-priori + Carver S24 shrinkage | green |
| G6.4 | K7 | live decision consistency (d=0 recomputes) | green |
| G6.6 | K6 | artifact export completeness | green |

**Final register: 11 of 12 gates green, one honest failure.** The one failure is the
program's most valuable output: it is the difference between a backtest and a story.
