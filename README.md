# Project AURUM — XAU/USD H1 Quantitative Trading System (v1.0 → v2.0.2)

> **Institutional archive.** This repository is the complete, self-contained record of Project
> AURUM: every kernel, every research blueprint, every engineering flaw we caught and fixed,
> every report, every chart, every prediction file and model head — from the first deep-research
> session (Spec v1.0) through the final honest statistical re-measurement (Spec v2.0.2).
> Cloning this repo plus running `python3 scripts/download_heavy_assets.py` gives you 100% of
> the program. Nothing else is needed — no chat logs, no external state.

---

## 1. Executive Summary

Project AURUM is an AI trading-decision system for XAU/USD (spot gold) on the H1 horizon. It
was built entirely on free Kaggle compute (2×T4 GPUs, account `tskdev`, with a secondary
account `ranigain` for quota-overflow reruns) through five specification generations in five
days (2026-11-06 → 2026-11-10). The program's defining discipline: **every headline number is
gate-checked, bootstrap-intervalled, and reported with its own failure modes** — including the
final one, where the flagship's 95% DSR certification was *withdrawn* because fixing a data
artifact widened the honest trial dispersion (Spec v2.0.2, Section 7.4 of the master report).

### The evolution

| Spec | Date | What it was | Defining result |
|---|---|---|---|
| **v1.0** | 2026-11-06 | Deep-research feasibility study + build plan: gold tradability, Kaggle GPU budget, multiple-testing discipline | `Project_AURUM_Deep_Research_Report` (31 pp), `AURUM_Decider_Research_Report_and_Build_Plan` |
| **v1.1** | 2026-11-06/07 | "48-cell world model": looped **Qwen3-0.6B** core (recurrent depth r∈{1,2,3}, looping layers 13–16, ~63M unfrozen recurrent params), serialized market-state language, 5 output heads (H1 48-barrier logits, H2 timeout MTM quantiles, H3 MFE/MAE pinball, H4 vol quantiles, H5 regime aux) | Phase 0 infrastructure proven: 1,051 tok/s (1×T4), 4,334 tok/s (2×T4 DDP); Phase 1 data pipeline |
| **v1.2** | 2026-11-07/08 | Postmortem generation: the **4-failure autopsy** (quota overrun 25h50m/30h; cohort memorization; 151,936-vocab bottleneck; ex-post overlay leak) and the fix plan — **97/111-token vocab slicing**, triangular-smoothed realized targets, causal path walks | `AURUM_Phase01_Postmortem_4_Failures_Verified_Fixes` (19 pp) + `AURUM_Engineering_Verification_7_Flaws_and_Fixes` (31 pp) |
| **v1.3** — *Champion* | 2026-11-09 | Decision-layer & realism upgrade: **LCB-EV** (lower-confidence-bound expected value) + **non-overlapping portfolio execution** (resolution-locked cap=1/cap=3), Quarter-Kelly k=0.25, 1% equity risk, live-daily kernel with 4-depth CPU forwards | **Flagship cap=3: OOS Sharpe 2.444, DSR 0.895** (v1.3 native frame); 3,341 trades in the v2 translated frame, CAGR 54.6%, MaxDD 13.1% |
| **v2.0.2** | 2026-11-09/10 | Ground-up continuous engine: **375 causal features** (path signatures, EW-RLS residualization, realized-jump geometry, microstructure toxicity), 6+2 model tournament (GBDT / de-saturated RFF-Ridge / DAE / DMN / contractive ContinuumNet), OnlineExecutionEngine (Page-Hinkley, Moreira-Muir vol targeting, Carver ±10% inertia buffer, AHL DD governor), Carver SR-shrinkage ensembling, dual M7 hybrid benchmarks | **M7 Core-Satellite: OOS 2.635 / 2.501 @2.0× costs, DSR 0.9407** (borderline, honestly reported); **M7 Active Grand Hybrid: 1.946 / 1.629, DSR 0.6502**; CSCV-PBO 0.0049 |

### The headline table (Spec v2.0.2 unified frame)

All numbers: annualized Sharpe of the net (post-cost) daily equity stream, 15% annualized
volatility target, `COST_R = 5e-5` per unit exposure change at 1.0×; 5,786 H1 bars/year.
**FIT = 2016-01 → 2022-12 walk-forward training window (41,513 bars). OOS = 2023-01 → 2025-12,
untouched (16,991 bars).** Bootstrap: 5,000 samples, 24-bar blocks.

| Strategy | FIT 1.0× | FIT 1.5× | FIT 2.0× | OOS 1.0× | OOS 1.5× | OOS 2.0× | OOS DSR | Cost 2.0× Sharpe |
|---|---|---|---|---|---|---|---|---|
| **M7 Core-Satellite** (50% v1.3 champion + 50% passive gold) | 1.449 | 1.375 | 1.300 | **2.635** | **2.568** | **2.501** | 0.9407† | 2.5012 |
| M0 Champ v1.3 (translated, cap=3) | 2.013 | 1.861 | 1.710 | 2.611 | 2.475 | 2.339 | 0.9352 | 2.3391 |
| M0 Champ v1.3 (cap=1) | 1.738 | 1.609 | 1.480 | 2.544 | 2.440 | 2.335 | 0.9202 | 2.3350 |
| Buy & Hold Gold (vol-targeted) | 0.699 | 0.684 | 0.670 | 2.160 | 2.147 | 2.135 | 0.7769 | 2.1345 |
| **M7 Active Grand Hybrid** (50% champion + 50% ACTIVE v2 stack) | 1.465 | 1.334 | 1.221 | **1.946** | 1.785 | 1.629 | 0.6502 | 1.6294 |
| Carver EWMAC (benchmark) | 0.372 | 0.283 | 0.086 | 1.228 | 1.169 | 1.100 | 0.2021 | 1.100 |
| M3c DAE (denoising autoencoder) | 0.684 | 0.370 | 0.111 | 0.993 | 0.929 | 0.853 | 0.1087 | 0.853 |
| M1 GBDT v2 (3-seed LightGBM) | −0.126 | −0.173 | −0.217 | 0.938 | 0.942 | 0.936 | 0.0933 | 0.9365 |
| M6 AURUM v2 Ensemble (M1+M2, Carver S24) | −0.407 | −0.475 | −0.495 | 0.728 | 0.592 | 0.482 | 0.0465 | 0.4816 |
| RLSBank (online ridge bank) | −1.144 | −1.496 | −1.869 | 0.600 | 0.206 | −0.157 | 0.0275 | −0.157 |
| M5 ContinuumNet (contractive loop) | −0.006 | −0.640 | −1.276 | −0.453 | −1.035 | −1.619 | 0.0001 | −1.6188 |
| M2 RFF Ridge (**de-saturated**, v2.0.2) | −0.736 | −0.934 | −0.964 | −0.744 | −0.790 | −1.036 | 0.0 | −1.0355 |
| Channel A (causality trap baseline) | −0.225 | −0.378 | −0.743 | −0.212 | −0.365 | −0.507 | 0.0005 | — |

† **The honest footnote that defines this program:** the flagship's DSR of 0.9407 *misses* the
pre-registered 0.95 admission bar. In v2.0.1 it measured 0.958 and "passed" — but that panel
contained the saturated-M2 artifact trial (+0.3504 FIT). Spec v2.0.2 replaced it with the true
de-saturated trial (−0.8394), widened the Bailey–López de Prado null threshold, and the
certification was withdrawn to borderline. **The gate is reported as FAILED (G6.1), exactly as
pre-registered.** No strategy clears the 0.95 bar under the honest panel; the best active
v2-native model (M1 GBDT, 0.938 OOS) fails DSR at 0.0933. See
`docs/TRIAL_LEDGER_AND_DSR_AUDIT.md` for the full multiple-testing accounting.

### What "Grand Hybrid" means (v2.0.2 dual benchmark)

- **M7 Core-Satellite** — the transparent decomposition: 50% v1.3 champion + 50% passive gold
  sleeve. This is what the v2.0.1 "M7 Grand Hybrid" *actually was* in OOS (its v2 leg had
  saturated to a constant +1.0, bit-identical to Buy & Hold). It preserves the program's best
  numbers verbatim: 2.6351 / 2.5012 @2.0×, the best cost survival produced.
- **M7 Active Grand Hybrid** — the true active claim: 50% champion + 50% *de-saturated* v2
  active stack (M6 = fixed a-priori {M1 GBDT, M2 RFF} 50/50 after Carver S24 SR-shrinkage 0.90).
  Honest number: 1.946 OOS / 1.629 @2.0× / DSR 0.6502.

### The v1.3 native frame (for reference)

The v1.3 flagship was measured in its own R-multiple portfolio frame before the v2 translation:
cap=3 @1.0× cost: 9,819 trades, E[R]=+0.1182R, total +1,160.7R, CAGR 33.0%; @2.0× cost: CAGR
12.5%. Daily bootstrap OOS Sharpe median 2.32 [1.37, 3.23]. Full detail in
`artifacts/v1_3/p4_backtest/backtest_main.json` and the v1.3 master report.

---

## 2. Repository Layout

```
project-aurum-xauusd-quant/
├── README.md                  <- you are here
├── AGENTS.md                  <- onboarding for AI coding agents (READ FIRST if you are an agent)
├── docs/
│   ├── ENGINEERING_AUTOPSY_AND_8_FLAWS.md   <- the 8 quantitative trap doors, with math & diffs
│   ├── DATA_AND_FEATURE_DICTIONARY.md       <- all 375 features: formulas, lags, OOS Rank-IC t-stats
│   ├── TRIAL_LEDGER_AND_DSR_AUDIT.md        <- N_trials, DSR, CSCV-PBO accounting
│   ├── AURUM_Current_State_Verified.md      <- the verified state audit (2026-11-07)
│   ├── EXECUTION_JOURNAL.md                 <- session-by-session execution log
│   └── blueprints/                          <- the two deep-research .md blueprints (v2 design inputs)
├── reports/                   <- all six master reports, .docx + .pdf each
│   ├── v1_0_research/         Project_AURUM_Deep_Research_Report            (31 pp)
│   ├── v1_0_build_plan/       AURUM_Decider_Research_Report_and_Build_Plan
│   ├── v1_2_postmortem/       AURUM_Phase01_Postmortem_4_Failures_Verified_Fixes (19 pp)
│   ├── v1_2_engineering_verification/  AURUM_Engineering_Verification_7_Flaws_and_Fixes (31 pp)
│   ├── v1_3_champion/         AURUM_Master_Execution_and_Backtest_Report    (21 pp)
│   └── v2_0_2_master/         AURUM_Spec_v2_Master_Execution_and_Tournament_Report (20 pp)
├── figures/                   <- all 32 charts (200 DPI PNG)
│   ├── v2_202/        (11)  fig1_oos_equity … fig11_z_desaturation
│   ├── v13_master/    (8)   v1.3 report figures (equity, realism, causality, DSR, live pathway)
│   ├── v13_report_figs/(8)  v1.2/v1.3-era report figures (cost sensitivity, depth sweep, folds)
│   ├── research/      (3)   gold milestones, Kaggle feasibility, multiple testing
│   └── audit/         (2)   hindsight bias, walk-forward budget
├── aurum_v2/                  <- v2.0.2 source code (the continuous engine)
│   ├── aurum_features/       f1_clock … f6_trend, build_v2_features.py
│   ├── kernels/              k1_features, k2_cpu_models, k3_gpu_neural_compact, k3b_completion,
│   │                         k6_backtest, k7_live_daily
│   └── model/                continuum, losses (DeePMRobustSharpe), online_engine, rff_ridge,
│                             ae_mlp_dmn, purge, train_continuum, v2_common
├── v1_3/                      <- v1.3 source code + institutional docs
│   ├── kernels/              p0_core.py, p0_pilot.py, p1_data.py (+ v6 copy)
│   ├── lib/                  aurum_data / aurum_feat / aurum_gates / aurum_label / aurum_model / aurum_ser
│   ├── local_experiments/    exp1_memorization … exp4b_toy_ce (the flaw-discovery experiments)
│   ├── CURRENT_STATE.md, execution_journal.md, build.py
├── kaggle_kernels/            <- the 16 pushed Kaggle kernel scripts (authoritative, as-run)
│   ├── v1_3/  aurum-p0-core v8, aurum-p0-pilot v6, aurum-p1-data v6, aurum-p2-train v5,
│   │          aurum-p3-tables v8, aurum-p4-backtest v4, aurum-live-daily v2
│   └── v2/    aurum-v2-features v4, aurum-v2-cpu-models v5, aurum-v2-cpu-models-rg v1 (K2RG),
│               ranigain-aurum-v2-cpu-models-rg v1, aurum-v2-neural-compact v2,
│               aurum-v2-neural-completion v1, aurum-v2-backtest v6, aurum-v2-backtest-and-stats v1,
│               aurum-v2-live-daily v6   (+ k2rg/k3b/k6/k7 staging dirs with kernel-metadata.json)
├── scripts/                   <- all build/report/chart tooling (creds removed; env-var pattern)
│   ├── v2/                    bundle.py, build_k*.py, make_charts.py, report/, generate_report.js
│   ├── aurum*/                v1.x report generators + figures
│   ├── download_heavy_assets.py  <- one command: fetch all >50MB Release assets + verify SHA-256
│   └── kgapi.py               Kaggle API helper (Bearer-token auth from env)
├── artifacts/                 <- every gate JSON, ledger, and compact result file
│   ├── v2/k1_features/        gates.json, feature_meta.json, ic_audit.parquet, ic_family_summary.csv
│   ├── v2/k2_cpu_preds_v3/    v2_cpu_forecasts.parquet (v3 = de-saturated z_rff) + gates + meta
│   ├── v2/k3_neural/          v2_neural_forecasts.parquet + rho_log + trial ledger
│   ├── v2/k6_backtest/        backtest_v2.json, champion_compare.json, stats.json, gates.json,
│   │                          gauntlet.json, net_series.parquet, year_table.csv, z_quality.json,
│   │                          trial_ledger.jsonl (446 rows), evalgrid/*.json, fitgauntlet/*.json
│   ├── v2/k7_live_daily/      decision_v2.json (final live decision), gates.json
│   ├── v2/v13_reference/      v1.3 trades_all.parquet, policy_fit.json, backtest_main.json
│   ├── v1_3/p0_core/          gates, bench, DDP bench, gentable reference tables
│   ├── v1_3/p2_train_meta/    fold_results, schedule, stageA/B results, temperatures
│   ├── v1_3/p3_tables/        prob-table gates + cell meta (the 63MB parquet is a Release asset)
│   ├── v1_3/p4_backtest/      backtest_main.json, equity_curves.parquet, trades_all.parquet, …
│   ├── v1_3/live_daily/       decision.json, risk_audit.json, live_state.txt
│   └── model_heads/           heads_fold2016.pt … heads_fold2025.pt, heads_main.pt (11 × ~5.1MB)
├── data/                      <- h1_xauusd.parquet (core price series), codebooks.npz, vocab_keep_ids.npy
├── tests/                     <- test_local.py (v1.3 + v2), live-logic tests T1-T7, smoke tests
└── .gitignore
```

**Heavy assets (>50MB, not in the git tree)** live in GitHub Release **`v2.0.2`** and are
restored by `python3 scripts/download_heavy_assets.py`:

| Asset | Size | What it is |
|---|---|---|
| `aurum_v2_features.parquet` | 253.7 MB | 95,008 H1 bars × 375 causal v2 features, 2010-01-03 → 2025-12-31 |
| `core_fp16.pt` | 125.9 MB | trained Looped Qwen3-0.6B core (fp16), Spec v1.3 |
| `aurum_oos_prob_tables.parquet` | 63.3 MB | 569,152-row × 4-depth OOS probability tables, Spec v1.3 |
| `h1_xauusd_and_cross_assets.tar.gz` | ~476 MB | the full `aurum-phase1-data` v6: H1+M1 parquets (XAUUSD/EURUSD/USDJPY/XAGUSD), labels (R actual/beff/evref/excursions/pwin), cohort_idx, codebooks, vocab, gates |
| `aurum_v2_workspace_backup.tar.gz` | ~77 MB | complete final workspace snapshot (sources + outputs, as uploaded to Kaggle `tskdev/aurum-v2-workspace-backup`) |

---

## 3. The Pipeline (how to reproduce everything)

### 3.1 The seven v2 kernels (Kaggle, in order)

| # | Kernel (slug under `tskdev/`) | Compute | Wall | Produces |
|---|---|---|---|---|
| K1 | `aurum-v2-features` | CPU | ~25 min | `aurum_v2_features.parquet` (375 features), IC audit, G1 gates |
| K2 | `aurum-v2-cpu-models` → v2.0.2 rerun via `aurum-v2-cpu-models-rg` (K2RG) | CPU | ~30 s (RG patch) | `v2_cpu_forecasts.parquet` v3 (z_gbdt, de-saturated z_rff, z_rlsbank), G2 + G4.8 gates |
| K3 | `aurum-v2-neural-compact` + `aurum-v2-neural-completion` | 2×T4 GPU | ~9 h (quota-spread over folds) | `v2_neural_forecasts.parquet` (M3c/M3m/M4/M5), rho_log, trial ledger |
| K6 | `aurum-v2-backtest` | CPU | ~3 min | `backtest_v2.json` (18 strategies × FIT/OOS × 1.0/1.5/2.0×), gates, gauntlet, DSR/PBO, net_series, year_table, z_quality |
| K7 | `aurum-v2-live-daily` | CPU | ~7 min | `decision_v2.json` (next-bar position, regime, stops, cost budget) |

Kernel sources: `kaggle_kernels/v2/*.py` (as-run standalone bundles) and `aurum_v2/` (the
modular source they are bundled from via `scripts/v2/bundle.py`).

### 3.2 The v1.3 chain

`aurum-p0-core` (infra, gates G0.1-G0.7) → `aurum-p0-pilot` → `aurum-p1-data` (serialization,
labels, cohorts, vocab) → `aurum-p2-train` (Stage A/B looped-core training, 10 walk-forward
folds 2016-2025) → `aurum-p3-tables` (569k-row OOS probability tables) → `aurum-p4-backtest`
(cap=1/cap=3 portfolio engine, 12 gates G5.0-G5.9) → `aurum-live-daily` (LCB-EV decision).

### 3.3 Local reproduction

```bash
# 0) restore heavy assets (features parquet, core weights, prob tables, base data)
python3 scripts/download_heavy_assets.py          # needs GITHUB_TOKEN env var (repo is private)

# 1) v2 features already built? verify integrity
python3 - <<'PY'
import pandas as pd; df = pd.read_parquet('data/aurum_v2_features.parquet')
assert df.shape == (95008, 375), df.shape
print('features OK:', df.index[0], '->', df.index[-1])
PY

# 2) unit tests (v2 source-level sanity: shapes, causality, purge/embargo)
python3 tests/test_local_v2.py

# 3) charts + report (v2.0.2, 11 figures, 20-page docx+pdf, 0 postcheck errors)
AURUM_CH_K6=artifacts/v2/k6_backtest python3 scripts/v2/make_charts.py
cd scripts/v2 && npm install && node report/generate.js && python3 report/postcheck.py
```

### 3.4 Kaggle push convention (how the kernels were operated)

```bash
export KAGGLE_API_TOKEN=$(cat ~/.aurum_token)     # NEVER hardcode; see AGENTS.md
python3 scripts/kgapi.py k-pull tskdev/aurum-v2-backtest /tmp/k6.py   # fetch latest
# edit aurum_v2/kernels/k6_backtest.py, then:
python3 scripts/v2/build_k6.py        # bundles modular source -> kaggle_work/v2/k6/kernel.py
kaggle kernels push -p kaggle_work/v2/k6
```

---

## 4. The Statistical Honesty Contract

Every number in this repo obeys five pre-registered rules (violating any of them produced at
least one of the 8 flaws — see `docs/ENGINEERING_AUTOPSY_AND_8_FLAWS.md`):

1. **Walk-forward only.** FIT 2016-2022, OOS 2023-2025, purge=24 bars + embargo=24 bars on
   every fold boundary (`aurum_v2/model/purge.py`).
2. **No same-bar information.** A position formed at the close of bar *t* may only earn the
   return of bar *t+1* (`z[t-1]·r[t]`, never `z[t]·r[t]`) — regression-gated by **G4.7**.
3. **No fold-scale normalization on predictions crossing a gate.** Every z-score transform
   must be causal (trailing window only) — regression-gated by **G4.8** (std > 0.05,
   saturation fraction < 5%, turnover > 2.0).
4. **Costs before Sharpe.** `COST_R = 5e-5` per unit exposure change; the Carver speed limit
   (annual cost ≤ ⅓ pre-cost SR) gates the shipping family (**G5.4**).
5. **Multiple-testing accounting.** Every trial is ledgered (K2/K3/K6 = 551 rows); DSR is
   computed against the *honest* panel (N=170 unique configs; the saturated-M2 artifact trial
   was replaced by the true one in v2.0.2) and CSCV-PBO is reported alongside (**G6.1**).

## 5. Provenance & License

- All market data: HistData M1 XAUUSD/EURUSD/USDJPY/XAGUSD (2010-2025) + FRED macro series,
  fetched inside Kaggle kernels; LBMA/WMR/SGE/COMEX timetables verified against primary
  sources (see `aurum_v2/aurum_features/f1_clock.py` header).
- Compute: Kaggle free tier (2×T4), accounts `tskdev` / `ranigain`. GPU quota was the binding
  constraint throughout (30 h/week); see `docs/AURUM_Current_State_Verified.md`.
- Reports were generated with the `docx` skill pipeline (docx → LibreOffice PDF), postchecked
  to 0 errors / 0 warnings.
- Internal use / research. Not investment advice. Past OOS performance does not guarantee
  future returns; the flagship's own DSR gate failure (0.9407 < 0.95) is the operative
  caution.
