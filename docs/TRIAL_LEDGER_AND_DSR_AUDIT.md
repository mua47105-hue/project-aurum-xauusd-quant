# Trial Ledger & DSR Audit — Project AURUM v2.0.2

> The complete multiple-testing accounting for the v2 tournament: every evaluated
> configuration, how trial counts feed the Deflated Sharpe Ratio, and how CSCV-PBO was
> computed. Sources: `artifacts/v2/k2_cpu_preds_v3/trial_ledger.jsonl` (v2.0.2 rows),
> `artifacts/v2/k2_cpu_preds_v3/../../k2_out2` vintage (78 search rows, mirrored in the
> workspace backup), `artifacts/v2/k3_neural/trial_ledger.jsonl`,
> `artifacts/v2/k6_backtest/trial_ledger.jsonl`, and the implementation in
> `aurum_v2/model/online_engine.py` (`deflated_sharpe_ratio`, `evaluate_deflated_sharpe_gate`,
> `pbo_cscv`) invoked from `aurum_v2/kernels/k6_backtest.py` §7.

---

## 1. The ledger (what is recorded, and why)

Every model/config evaluation in the v2 program appends one JSON line:

```json
{"model": "M1_GBDT_v2", "config": {…full hyperparameters…}, "config_hash": "14f016ab1b",
 "fit_metric": -0.2391, "stage": "fit", "kernel": "K6", "ts": "2026-11-10 12:04:11",
 "notes": ""}
```

`config_hash` is the stable hash of the config dict — the dedup key. Stages:
`config` (a-priori construction), `search` (hyperparameter probes),
`fold` (per-fold results), `fit` (FIT-window evaluation),
`engine` (OnlineExecutionEngine variants), `gauntlet` (engine-vs-frozen decision rows),
`fit_eval` (blend-grid diagnostics), `regression` (causality/G4.7 trap rows).

### Ledger census (final v2.0.2 state)

| Kernel | Rows | Contents |
|---|---|---|
| K2 (search vintage, `k2_out2`) | 78 | M1 GBDT config/fold rows (31), M2 RFF rows (16), RLSBank (1), engine variant rows (30: M1/M2/RLSBank × {no_buffer, frozen_rls_off, no_regimes, cost ladders…}) |
| K2 (v2.0.2 rows, `k2_out3`) | 3 | the de-saturation re-anchor trial + G4.8 audit rows |
| K3 (neural) | 27 | search rows (7: M3c, M3m, M4, M4b, M5 hyperparameter probes) + fold rows (20: M5 × 10 folds, M4b × 10 folds) |
| K6 (tournament) | 446 | config rows (36), fit rows (40), engine rows (30), gauntlet rows (35), causality regression rows (26), M7 blend-grid diagnostics (252), M5/M4b/M6/M7 family rows |
| **Total** | **551** | **170 unique (model, config_hash) pairs** |

**N_trials = 170** is the number the DSR null is built from (§3).

---

## 2. The Deflated Sharpe Ratio (exact formula used)

Implementation: `aurum_v2/model/online_engine.py::deflated_sharpe_ratio` —
Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, Journal of Portfolio
Management — computed **per-bar** (SR and V[SR] in the same frequency; annualized figures
are derived by √5786 only for reporting):

```
DSR = Φ( (SR − SR₀)·√(T−1) / √(1 − γ₃·SR + (γ₄−1)/4·SR²) )

SR₀ = √V[SR_trials] · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]
     with γ = 0.5772156649 (Euler–Mascheroni), N = 170, e = 2.71828…
```

- `SR` — the strategy's per-bar Sharpe on the **OOS** window (16,991 bars);
- `γ₃, γ₄` — OOS return skewness and kurtosis (the non-normality correction);
- `V[SR_trials]` — cross-sectional variance of the **trial Sharpes** (the panel of §3);
- The pre-registered admission gate (**G6.1**) requires **DSR ≥ 0.95**.

### Why V[SR_trials] is the load-bearing quantity

The null threshold SR₀ scales with √V[SR_trials]: the more dispersed your trials' Sharpe
estimates, the higher the bar any single survivor must clear. This is exactly where
**Flaw #8 bit the statistics**: the v2.0.1 panel contained the saturated-M2 artifact trial
at **+0.3504 FIT Sharpe**. De-saturating M2 replaced it with the true trial at
**−0.8394** — widening the honest dispersion, raising SR₀, and dropping the flagship
M7_Core_Satellite from **DSR 0.958 (v2.0.1, "pass") to 0.9407 (v2.0.2, FAIL)**. The same
repair moved CSCV-PBO from 0.0152 to **0.0049** (the selection became *more* stable — the
panel now agrees on the same winners — while the significance bar simultaneously rose).

### The trial panel (what exactly counts as a trial)

v2.0.2 applies the Bailey-LdP requirement that **N counts independent trials**. Correlated
siblings are not independent, so the panel is reduced to *effective strategy families*:

1. **Exact stream duplicates collapse** (e.g. M6b is bit-identical to M6 when the stack
   weight hits the equal-weight fallback).
2. **Sibling groups count once, via their best-FIT member**: M3c/M3m → `M3_DAE`;
   M4/M4b → `M4_DMN`; M0_Champ_v13 (cap=3) / cap=1 → `M0_Champ`.
3. **Blend constructions count as ONE blend-slot trial**: M6/M6b/M7_Active/M7_Core are
   deterministic re-weightings of already-counted members (fixed a-priori 50/50 weights,
   shipped *against* their FIT optimum — the M7 blend grid found best w_v2 ≠ 0.50 and the
   ship used 0.50 anyway, so the grid's 252 rows are diagnostics, not selection trials).
4. **Excluded, but sensitivity-reported**: the pre-fix same-bar accounting vintage (K2/K3
   rows with unit-incomparable metrics).

**Primary family (N = 12):** M1_GBDT_v2, M2_RFF_Ridge, RLSBank, M3_DAE, M4_DMN,
M5_ContinuumNet, Carver_EWMAC, Channel_A, Channel_B, BuyHold_Gold, M0_Champ, + 1 blend slot.
**Conservative vintage (N = 14):** every distinct blend stream counts separately
(flagship DSR under this vintage: **0.9248**). Both are exported in
`stats.json → _meta`.

### The final DSR register (primary vintage, OOS 2023-25)

| Strategy | OOS Sharpe | DSR | Verdict |
|---|---|---|---|
| **M7 Core-Satellite** | 2.6351 | **0.9407** | **FAIL (bar 0.95) — borderline, certification withdrawn** |
| M0 Champ v1.3 (cap=3) | 2.6112 | 0.9352 | fail |
| M0 Champ v1.3 (cap=1) | 2.5442 | 0.9202 | fail |
| Buy & Hold Gold | 2.1604 | 0.7769 | fail |
| **M7 Active Grand Hybrid** | 1.9460 | **0.6502** | fail |
| Carver EWMAC | 1.228 | 0.2021 | fail |
| M3c DAE | 0.993 | 0.1087 | fail |
| M1 GBDT v2 | 0.938 | 0.0933 | fail |
| M6 Ensemble | 0.728 | 0.0465 | fail |
| RLSBank | 0.600 | 0.0275 | fail |
| M5 ContinuumNet | −0.453 | 0.0001 | fail |
| M2 RFF Ridge (de-saturated) | −0.744 | 0.0 | fail |
| M4/M4b DMN | −1.08 | 0.0 | fail |
| Channel A | −0.212 | 0.0005 | fail |

**No strategy clears the pre-registered 0.95 bar under the honest panel.** The v2.0.1
certification (0.958) was measured on a panel whose M2 trial was the saturated artifact;
v2.0.2 reports the gate as failed, verbatim, in `gates.json → G6.1` and in Section 7.4 of
the master report. The 5,000-sample block bootstrap (24-bar blocks) brackets the flagship
at OOS Sharpe CI [1.486, 3.765], p05 = 1.688 — the stream is real; the *selection
significance* is what fails.

---

## 3. CSCV-PBO (combinatorially symmetric cross-validation)

Implementation: `pbo_cscv` (in `online_engine.py`), applied to the **FIT-window net-stream
matrix of the selection family** with **16 blocks** (the standard S=16 CSCV of
Bailey-Borwein-López de Prado-Zhu 2017):

1. Divide the evaluation matrix (bars × strategies) into S=16 contiguous, equal blocks.
2. Form all C(16,8) = 12,870 combinations of 8 training / 8 testing blocks, symmetric
   under complement.
3. For each combination: rank strategies by Sharpe on the training half; record each
   strategy's rank on the testing half.
4. **PBO** = fraction of combinations where the best-training strategy's testing rank falls
   in the bottom half of the testing ranks (a rank violation), i.e. the probability that
   selection was driven by overfit.

**Measured: 0.0049** (v2.0.2 panel; 0.0152 on the v2.0.1 panel) — the selection among the
12 families is essentially never inverted out-of-sample. Gate: PBO ≤ 0.20 (green by a
wide margin). Interpretation: the tournament's *ranking* is stable; what fails is the
*absolute significance* of the flagship after honest trial-count inflation — the two
statistical lenses (PBO stable, DSR borderline) are exactly what a healthy-but-humble
program should show.

---

## 4. Bootstrap protocol (every OOS Sharpe in this repo)

- **Block bootstrap**, 5,000 samples, block length 24 bars (one trading day), on the net
  (post-cost) per-bar return stream; report CI [p05… hmm: lo, med, hi], and
  `sharpe_p05` = 5th percentile of the bootstrap Sharpe distribution.
- Applied uniformly to all 18 strategies (`stats.json`, `champion_compare.json`).
- v1.3 lineage additionally used a **daily** bootstrap (block 10 days, 110 blocks) on the
  portfolio equity stream — the 0.895 DSR of the v1.3 flagship was computed on daily
  returns with its own trial count (the v1.3 selection family: 8 strategies).

## 5. Engine-vs-frozen gauntlet (G5.3) — why engine variants are trials too

Each active strategy was evaluated both as a *frozen* signal (z from the model, direct
position) and through the **OnlineExecutionEngine** (Page-Hinkley breaks, Moreira-Muir vol
targeting, Carver ±10% buffer, AHL DD governor). The engine ships **iff it beats frozen on
FIT** (v1.3 translations are forced frozen — their engine is the v1.3 portfolio layer).
Measured buffer value: turnover cut **32.0%** (M1) / **38.6%** (M2) at equal-or-better FIT
Sharpe. Every engine variant row in the ledger is a trial — the +10 engine rows per model
in the K2 ledger are exactly these ablations (no_buffer / frozen_rls_off / no_regimes /
cost ladders).

## 6. Reproducing the audit

```bash
# panel + DSR + PBO, straight from the final K6 artifact
python3 - <<'PY'
import json
g = json.load(open('artifacts/v2/k6_backtest/gates.json'))
print(g['G6.1_stats']['detail'])          # N=170, DSR 0.9407, PBO 0.0049, verbatim
st = json.load(open('artifacts/v2/k6_backtest/stats.json'))
print(json.dumps(st['_meta'], indent=1))  # dsr_sensitivity_m7, pbo, vintages
PY

# recompute DSR from first principles
python3 - <<'PY'
import sys; sys.path.insert(0, 'aurum_v2/model')
from online_engine import deflated_sharpe_ratio
# SR per-bar of the flagship OOS stream, T=16991, skew/kurt from the stream,
# sr_trials = the 12-family FIT Sharpe panel (annualized/√5786), N=170
PY
```

The gate text is deliberately self-describing: `gates.json` carries the full reasoning
string, including the pre-registered-bar-missed finding, so any future agent re-reading
the artifact gets the complete statistical context without this document.
