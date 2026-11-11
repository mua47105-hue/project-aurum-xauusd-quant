# PROJECT AURUM — Verified Current State Audit
**Audit date:** 2026-11-07 (this session; credentials restored + live Kaggle state verified ~13:30 UTC same day) · **Auditor:** execution agent, under owner directive: *"don't just assume anything is done, verify that"*
**Method:** Every claim below was re-checked this session against artifacts on disk (pulled Kaggle outputs, gates JSONs, result files, code greps, live test re-runs) and, after credentials were restored, against the LIVE Kaggle API. Nothing is taken from memory or conversation summaries alone.

---

## 1. Project identity (one paragraph)

Project AURUM: an AI trading decision model for XAUUSD (gold) H1 trading. Core approach: refit **Qwen3-0.6B as a looped transformer** (recurrent depth r∈{1,2,3}, looping layers 13–16, ~63M unfrozen recurrent params), trained on a serialized market-state "language" (97-token sliced vocabulary, sliced cross-entropy objective with KL-anchor leash), with 5 output heads (H1 48-barrier logits, H2 timeout MTM quantiles, H3 MFE/MAE pinball, H4 vol quantiles, H5 regime aux), feeding a deterministic EV + Quarter-Kelly decision layer (k=0.25, 1% equity risk cap, H=48 bars). All compute free on **Kaggle 2×T4** via API automation, account **`tskdev`**.

**Governing documents:**
- **Spec v1.1** — frozen, IMPLEMENTED, deployed to Kaggle (verified byte-exact, §3).
- **Spec v1.2** — APPROVED by owner, **0% implemented** (verified, §5). It is the plan for all remaining work (6-step pipeline, §6).
- Spec text lives in conversation history only — **no spec file exists on disk** (verified: no *spec* file anywhere in the workspace).

**Deliverable reports on disk (verified):**
| File (in `download/`) | Pages | Content |
|---|---|---|
| `Project_AURUM_Deep_Research_Report.pdf/.docx` | 31 | Original feasibility research (gold tradability, Kaggle budget, multiple-testing discipline) |
| `AURUM_Engineering_Verification_7_Flaws_and_Fixes.pdf/.docx` | 31 | Engineering audit of the v1.1 design |
| `AURUM_Phase01_Postmortem_4_Failures_Verified_Fixes.pdf/.docx` | 19 | The 4-failure postmortem that produced Spec v1.2 (quota correction 25h50m/30h; Stage A' budget 4.9–8.2h) |

---

## 2. ✅ BLOCKER RESOLVED — Kaggle credentials restored & LIVE account state verified

**History:** the previous sandbox recycle had wiped the credentials (full-filesystem sweep found none; API returned "Authentication required"). **This session the owner supplied a new KGAT API token.** It is saved at `~/.kaggle/access_token` (chmod 600, not stored in the project tree) and authentication is verified working (username `tskdev`).

**LIVE-VERIFIED Kaggle account state (queried via API ~13:30 UTC, 2026-11-07):**

| Item | Verified state |
|---|---|
| Kernels (6 total) | `aurum-p0-core` COMPLETE (last run 2026-11-07 03:56) · `aurum-p0-pilot` COMPLETE (02:48) · `aurum-p1-data` COMPLETE (02:48) · `jarvis-env-test` COMPLETE (Nov 6) · 2 pre-project test notebooks (Oct 29 / Nov 1). Nothing running, nothing errored. |
| Datasets (1) | `tskdev/aurum-phase1-data` — status **ready**, 456.9 MB, updated 2026-11-07 03:36, content == local `dataset_p2/` (== p1_data v4 output) |
| **GPU quota (live)** | **Used 4h 9m 26s of 30h/week → 25h 50m 34s remaining** · 0s reserved · window resets **2026-11-14 00:00 UTC** (≈6.4 days). Matches owner's HOLD-time correction exactly; confirms **no GPU consumed since the HOLD** |
| TPU quota | Untouched (0 of 20h) |
| GPU access | `hasEverRun=true` — GPU enabled & proven on this account |

**Budget conclusion:** Stage A' needs 4.9–8.2h GPU; 25.8h remain → full Spec v1.2 pipeline fits with the 2.5h safety floor honored. Hard-stop rule stays active: if quota drops below 2.5h before a critical run, pause and report.

---

## 3. What WAS actually done — Phase 0 + Phase 1 (VERIFIED COMPLETE)

Evidence: 17 pulled result directories + 1 env-test = **18 kernel pushes with artifacts on disk** (matches the "18 pushes" claim). Local workspace: `/home/z/my-project/kaggle_work/aurum/` (2.9 GB restored from sandbox snapshot 2026-11-07 07:59 UTC).

### 3.1 Environment probe — `tskdev/jarvis-env-test` (PASSED)
Kaggle image verified: Python 3.13.18, torch 2.11.0, pandas 2.3.3, numpy 2.1.3, sklearn 1.6.1, xgboost 3.4.1. (Internet hit HTTP 429 on that probe run; later kernels' HistData/FRED fetches worked.)

### 3.2 `aurum-p0-core` — infrastructure kernel, v1→v8 (FINAL: ALL GATES PASS)
Final run v8 on Kaggle 2×T4, pulled and verified:
- **G0.1 env**: 2× Tesla T4, 15.64GB, torch 2.11.0+cu130, transformers 4.57.1 — PASS
- **G0.1 fp16 policy**: fp16 + GradScaler structural, residual clamp C=16384 — PASS
- **G0.2 loop correctness**: R=1 sandwich-off maxdiff 5.44e-05; identity-init; R=3 finite — PASS
- **G0.3 scale invariance**: features 2.04e-10, labels 2.24e-13, real XAUUSD-2024 ×137 scale 9.17e-11 — PASS
- **G0.4 throughput 1GPU**: r=3/bs=16/ckpt=on = **1051.4 tok/s** (full grid in bench.json; 6 OOM configs documented) — PASS
- **G0.5 DDP 2×T4**: **4334 tok/s aggregate**, step 3780ms, 10.98GB, grad_norm gap 0.0 — PASS
- **G0.6 table timing**: 30k rows × 85 outputs (R=3, 2 GPUs) in **479s wall** — PASS
- **G0.7 kill/resume**: checkpoint resume identical (maxdiff 0.0) — PASS

Bench highlights (from `results/p0_core_v8/bench.json`): r=1/bs=8/ckpt: 1674.6 tok/s @10.97GB; r=1/bs=16/ckpt: 1500.8 @14.73GB; r=2/bs=8: 1299.1 @10.97GB; r=1/bs=32: OOM. **Note: r=3/bs=16/ckpt = 14.73GB > 13.5GB budget** — this is why Step 2 (p2a-check) must re-verify VRAM with the sliced-CE head (projected to fit; see exp4 below) or fall back to bs=8.

### 3.3 `aurum-p0-pilot` — pilot training loop, v1→v6 (FINAL: ALL GATES PASS)
Final run v6 on Kaggle 2×T4, **real data** (`source: real`, world=2):
- 814 total steps / 611 LM steps / **5,005,312 tokens**
- **Loss 0.5488 → 0.3267 (−40.5%)** (gate ≥30%) — PASS
- **0 NaN skips**, clamp_rate 0.0, resid_max 7461.5 (C=16384), GradScaler final scale 16384 — PASS
- Throughput **1138 tok/s/GPU** LM-only, 982 incl. head-step (floor 700) — PASS
- Held-out ACTUAL-outcome **Brier 0.2473, AUC 0.5610** (true edge signal vs realized simple-stack outcomes, 500 bars EV-argmax) — PASS
- Held-out soft-label Brier 0.0022 (cohort estimator replication check) — PASS
- Pilot data: 17,000 states, mean 268.9 tokens/state

### 3.4 `aurum-p1-data` — dataset build kernel, v1→v4 (FINAL: 4/5 gates PASS — G1.4 FAIL is the known leakage finding)
Final run v4 on Kaggle CPU, pulled to `results/p1_data_v4/` == dataset `tskdev/aurum-phase1-data` (pulled copy at `dataset_p2/`, content verified identical):
- **G1.0 env** PASS · **G1.1 coverage** PASS: XAUUSD 16/16 years 2010–2025, 5,604,396 M1 rows, mean 93.2% minute-coverage; cross-assets XAGUSD 5.40M / EURUSD 5.86M / USDJPY 5.77M rows (UTC offsets estimated per asset)
- **G1.2 codebooks** PASS: M1 k=32 (99.5% coverage, inertia 0.2872), H1 k=32 (100%, 2.1524), fit <2021 only
- **G1.3 labels** PASS: **87,948 labeled H1 bars**, cohort≥180 for 94.9%, mean p_win 0.427, median b_eff 1.41
- **G1.4 adversarial AUC — FAIL (deliberate finding)**: RAW joint AUC **1.0000**; drop protocol removed 8 date-fingerprint features (m_spread, m_y10, m_y2, m_dxy, prior_w48, prior_l48, prior_l24, prior_w24); post-drop joint AUC **0.8055** vs gate ≤0.55 (in-train reference 0.7996). Worst surviving single feature ret48_atr |AUC−0.5| = 0.0438 (G1.4b PASS).
- **G1.5 serialization** PASS: 200 states, **mean 265 tokens/state** (31-feature post-drop set; v1.2 full format measured at 312.9 in exp4)

**This G1.4 failure is not a bug — it is the leakage evidence that triggered the owner's HOLD, the 4-failure postmortem, and Spec v1.2's redesign** (macro innovation transforms instead of raw levels; neutralization vector instead of feature dropping; within-era reference logic).

### 3.5 Local build system (RE-VERIFIED GREEN THIS SESSION)
- `python3 src/build.py --verify` → **3/3 BYTE-EXACT** roundtrips vs pulled Kaggle sources (p1_data 77ce874eb130 / 83,872 B; p0_pilot 8b6fe7c860f9 / 85,782 B; p0_core 811aa14493e2 / 89,087 B)
- `python3 src/test_local.py` → **40/40 PASS** (14s): labeler, cohort strict-past leakage asserts, scale invariance ×137, serialization, sliced-vs-gathered CE identity + gradient match, DDP gloo spawn/teardown, VRAM estimator (mean |err| 0.0%, est. 11.26GB @ sliced bs16), kill/resume equivalence
- **`src/` is the canonical tree.** Top-level `kernels/` + `lib/` are STALE pre-refactor copies (differ by explicit-import block) — superseded; do not edit those.

---

## 4. What WAS actually done — postmortem research (VERIFIED, all 9 experiments have real results)

All in `kaggle_work/aurum/local/` with result JSONs in `local/results/`. These are the experiments behind the approved Spec v1.2:

| Exp | Question | Verified result |
|---|---|---|
| **exp1_memorization** | Failure #1: does the model memorize cohort soft labels? | Yes: XGB predicts soft labels at val Brier 0.0017, but real-outcome test AUC only **0.5233** — cohort targets ≠ tradeable edge |
| **exp2_targets** | Which target is best? (T0 cohort / T1 realized / T2 smoothed / T3 mixes / T4 prior-in / T5 shallow) | T1 realized & T2 smoothed beat T0 on meanR (0.0453 / 0.0537 vs 0.0035); T5 shallow strongest overall — basis for v1.2 realized + triangular-smoothed H1 target |
| **exp2b_walkforward** | Robustness, expanding window 2016–2025 | T5: mean yearly R **0.0297**, 7/10 positive years (T0 cohort: 0.0070, 6/10); canonical long 0.0316 — targets choice holds out-of-sample |
| **exp3_era_audit** | Failure #2: era/date fingerprints | Raw 39-feat joint AUC 1.0; post-drop 31-feat 0.8048; residual drifters small (ret48_atr 0.5425) |
| **exp3b_frontier** | Which macro representation survives? | qmap / rankdiff arms measured with within-era references (e.g. M1_qmap AUC 0.7724 vs within-era ref 0.7803) — foundation of v1.2 G1.4(b) |
| **exp3c_final** | Innovation transforms individually clean? | mi_y10_d20z 0.476, mi_y2_d20z 0.471, mi_spread_d20z 0.527, mi_dxy_d20z 0.488, mi_vix_d5z 0.513 — **all \|AUC−0.5\| ≤ 0.03**; BUT dropping features to kill joint AUC destroys signal (test AUC 0.53→0.49) |
| **exp3d_neutralization** | Does neutralization preserve signal? | YES: canon_L AUC 0.5317→**0.5334**, canon_S 0.5412→**0.5421** after neutralization — basis of v1.2 G1.4(c) "save neutralization vector" |
| **exp4_vocab_cost** | Vocab + cost model | **97 unique token IDs** (min 10 / max 86128, saved as `vocab_keep_ids.npy`); full v1.2 format = **312.9 mean tokens/state** (min 183 / max 324, n=95,008 states); sliced-CE projections: r3/bs8/ckpt ≈ 4.82GB projected (fits), FLOP model calibrated vs observed (MFU 8.6%) |
| **exp4b_toy_ce** | Sliced-CE gradient quality | grad-cosine 0.945 at kept-mass 0.059; sliced objective improves held sliced CE (3.60→3.02) but not full CE — confirms the need for the **Z_kept/Z_full > 0.90** leash in real training |

---

## 5. What is NOT done — the entire Spec v1.2 pipeline (0% implemented) — VERIFIED

Verified by code search across `src/`, `kernels/`, `lib/`: **no v1.2 feature code exists in the canonical tree.** The innovation-panel computation exists ONLY as standalone experiment code (`local/exp3*.py`); the triangular-smoothed target logic only inside `local/exp2_targets.py`.

| Step | Item | Status |
|---|---|---|
| 1 | `aurum-p1-data` **v5** rebuild (macro innovation panel d20z/d5z/rank60, priors restored as inputs, cohort input summary with ≤ i−96 neighbors, triangular-smoothed H1 target, new G1.4 a/b/c + neutralization vector) | **NOT WRITTEN** — kernel source is still v4/Spec v1.1 |
| 1 | Dataset `tskdev/aurum-phase1-data` v5 versioning | **NOT DONE** (needs credentials) |
| 2 | `aurum-p2a-check` kernel (sliced-CE trajectory, Z ratio > 0.90, VRAM r=3 bs=16 < 13.5GB, token length 313–330, GradScaler stability) | **KERNEL DOES NOT EXIST** |
| 3 | `aurum-p2-train` (Stage A' CPT ≤2022-only states, Stage B 5-head SFT + cohort-aux kill-switch, causal walk-forward refits, temperature scaling) | **KERNEL DOES NOT EXIST** |
| 3 | Dataset `tskdev/aurum-models-v1` (checkpoints, LoRA deltas, sandwich norms, temperature scalars) | **DOES NOT EXIST** |
| 4 | `aurum-p3-tables` (batched inference → `aurum_oos_prob_tables.parquet`, 85 heads, OOD divergence, era-neutralized) | **KERNEL DOES NOT EXIST** |
| 5 | `aurum-p4-backtest` (vectorbt + bar-by-bar sim, baselines B&H/EMA/Donchian/k-NN/XGBoost/r-ablations, 5000-sample block bootstrap, DSR, cost stress ×1/1.5/2) | **KERNEL DOES NOT EXIST** — no backtest machinery anywhere locally either |
| 6 | `aurum-live-daily` schedulable paper-trading kernel | **DOES NOT EXIST** |
| 6 | Master Report (`AURUM_Master_Execution_and_Backtest_Report.docx/.pdf`) | **DOES NOT EXIST** |
| — | `execution_journal.md` | Exists, 2 entries (restore/blocker + preflight) — now updated with Entry 1 result + this audit as Entry 2 |

**No GPU hours have been consumed since the HOLD** (nothing was pushed — credentials were lost at the sandbox recycle).

---

## 6. Ready-to-execute plan for the next session (Spec v1.2, owner-approved autonomous mode)

The 6-step pipeline from the approved directive — all blockers now cleared:

1. **Step 1** — Build `aurum-p1-data` v5: port innovation transforms from `local/exp3c_final.py`, triangular-smoothed target from `local/exp2_targets.py`, add cohort input summary + priors to serializer; local gates (build --verify + test_local 40/40 + G1.4a/b/c offline check); push & version dataset. *(FRED fetch runs inside the Kaggle kernel — proven working there in v4.)*
2. **Step 2** — `aurum-p2a-check` (30–40min T4×2): sliced-CE trajectory, Z_kept/Z_full > 0.90, VRAM < 13.5GB @ r=3/bs=16/ckpt (fallback bs=8 @10.97GB proven), token length 313–330 vs 312.9 measured, GradScaler 0 NaNs. *(Re-check quota via `/api/v1/kernels/quota` immediately before the GPU push.)*
3. **Step 3** — `aurum-p2-train`: Stage A' CPT on **pre-test states only (≤2022 / ≤y−1)**, gates: loss −30%, 0 NaN, clamp <5%, ~60M tokens; Stage B joint-head SFT with λ≤0.3 cohort aux (kill-switch λ=0), variance discipline, early-stop on realized val AUC; per-head temperature scaling on purged split; publish `tskdev/aurum-models-v1`. Budget 4.9–8.2h GPU.
4. **Step 4** — `aurum-p3-tables`: batched fp16 inference → `aurum_oos_prob_tables.parquet`.
5. **Step 5** — `aurum-p4-backtest` (CPU): decision layer (EV + Quarter-Kelly k=0.25, 1% cap, MFE q50 partial TP, MAE q25 trailing stop, event blackouts, loop-OOD + GBDT disagreement gates), all baselines, 5000-sample stationary block bootstrap, DSR, cost stress.
6. **Step 6** — `aurum-live-daily` + **Master Report**.

Hard stops (owner-approved): gate failure after full fallback ladder · quota < 2.5h floor · unrecoverable Kaggle loss. Everything else self-heals (patch → verify → re-push).

---

## 7. Environment & constraints (verified this session)

- **Sandbox:** 2 CPU cores, 3 GB RAM, ~4.3 GB free disk. Python 3.12 venv with pandas 2.2.3, numpy 2.1.3, torch 2.15.0+cpu, transformers 4.57.1, xgboost 2.1.3, sklearn 1.5.2, pyarrow 26.0.1, kaggle CLI 2.2.5. Qwen3-0.6B tokenizer cached (vocab 151,643).
- **Network:** histdata.com reachable (raw M1 re-download possible); **fred.stlouisfed.org TIMES OUT** from sandbox → a fully-local v5 rebuild needs FRED via alternate route (assistant WebFetch tool, stooq mirror, or do the v5 fetch inside the Kaggle kernel, where FRED fetches are proven to work from the v4 run).
- **Raw data on disk:** all 4 symbols × 16 years of HistData M1 zips cached in `results/p1_data_v4/raw/` (and duplicated in other result dirs) — no re-download needed for rebuilds.
- **Data artifacts:** full v4 feature/label set in `dataset_p2/` (h1_features, labels_{pwin,beff,aux,cls_actual,R_actual,evref}, codebooks, cohort_idx, cohort_std_stats, coverage_report, audit, gates, sample_states).

## 8. Key numbers cheat-sheet (all verified from artifact files)

| Metric | Value | Source |
|---|---|---|
| Labeled H1 bars | 87,948 (94.9% with cohort≥180) | p1_data_v4/gates.json G1.3 |
| XAUUSD M1 rows | 5,604,396 (2010–2025, 93.2% cov.) | G1.1 |
| Pilot loss reduction | −40.5% (0.5488→0.3267) | pilot v6 |
| Pilot actual-outcome AUC | 0.5610 (Brier 0.2473) | pilot v6 |
| DDP throughput 2×T4 | 4,334 tok/s aggregate | p0_core v8 G0.5 |
| r3/bs16/ckpt 1GPU | 1,051.4 tok/s @ 14.73GB (full CE head) | bench.json |
| Sliced-CE projected mem (r3/bs8) | ~4.82GB → fits budget | exp4 |
| v1.2 state size | 312.9 tokens mean (target ~313–330) | exp4 |
| Sliced vocab | 97 token IDs (10 … 86128) | exp4 / vocab_keep_ids.npy |
| G1.4 v4 result | raw 1.0000 → post-drop 0.8055 (FAIL, by design finding) | p1_data_v4/gates.json |
| Innovation features single-AUC | all ≤ 0.03 from 0.5 | exp3c |
| Neutralization effect | signal preserved (AUC +0.002/+0.001) | exp3d |
| Walk-forward (T5 shallow) | mean yearly R 0.0297, 7/10 years positive | exp2b |
| GPU quota | **LIVE-VERIFIED: 4h9m26s used / 30h → 25h50m34s left, resets 2026-11-14 00:00 UTC** | Kaggle API `/api/v1/kernels/quota` (this session) |
| Stage A' budget | 4.9–8.2h GPU (floor rule 2.5h) | postmortem §6.3 |

---

## 9. Bottom line

**Actually done and verified:** complete Phase 0+1 infrastructure (18 Kaggle pushes, all gates green except the deliberate G1.4 leakage finding), the full v1.1 dataset on Kaggle, a proven DDP training loop with real-data pilot (−40.5% loss, genuine AUC 0.561), the entire postmortem research program (9 experiments with results), and three written reports. Local toolchain is green (byte-exact roundtrips, 40/40 tests).

**Not done at all:** the entire Spec v1.2 execution pipeline — Steps 1–6 (v5 data rebuild, pre-train checks, Stage A'/B training, inference tables, backtest, live harness, Master Report). 0% of that code exists.

**Blocking dependency: RESOLVED.** Credentials restored and live-verified (auth OK, all kernels COMPLETE, dataset ready, 25h50m/30h GPU quota remaining, reset Nov 14). **The project is fully unblocked and ready for the owner's GO prompt to execute the Spec v1.2 pipeline end-to-end.**

*Audit artifacts: `kaggle_work/aurum/execution_journal.md` (Entries 0–3), all gate/bench/result JSONs under `kaggle_work/aurum/results/` and `local/results/`.*
