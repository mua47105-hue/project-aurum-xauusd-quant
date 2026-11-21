# AURUM STATE — Live Checkpoint Ledger (Spec v2.1 Remediation)

> **This file is the single source of truth for resuming the v2.1 remediation.**
> Updated and pushed on every commit. Any future session can resume in 10 seconds:
> ```bash
> git clone https://github.com/mua47105-hue/project-aurum-xauusd-quant.git && cd project-aurum-xauusd-quant
> git checkout v2.1-remediation && bash scripts/install_secret_hook.sh
> pip install -r requirements.txt && python3 scripts/download_heavy_assets.py all
> ```

## Current position
- **Branch:** `v2.1-remediation` (base `6ae2565` = shipped v2.0.2 archive)
- **Step:** 1 COMPLETE (repro gaps #1–#3 + frozen policy) → next: STEP 2 (p4 v2.1 patch)
- **Last verified commit:** see `git log -1` (this file is committed together with each step)
- **Date:** 2026-11-21

## Verified numbers vs. regression targets (updated per step)
| Item | Target (audit / honest work report) | Status |
|---|---|---|
| Base clone | `6ae2565` | ✅ verified |
| Secret hook | blocks `KGAT_test` dummy | ✅ tested (FATAL abort observed) |
| Environment | xgb 3.4.1 / pd 2.3.3 / np 2.1.3 / sk 1.6.1 / numba 0.61.2 / lgbm 4.5.0 / torch 2.11.0+cpu / transformers 4.57.1 | ✅ installed & recorded (`environment.lock.json`) |
| Repro gap #1 — tests | `env -u PYTHONPATH` both suites | ✅ v13 16/16 PASS 0 fail · v2 all module self-tests PASS |
| Repro gap #3 — k2 standalone | pyflakes 0 undefined; smoke run | ✅ data 95008×378 loads, walk-forward folds run |
| Flaw #11 — frozen policy | `v1_3/config/frozen_policy.json` | ✅ created (w=[0.6,0.2,0.2] κ=0 λ=0.5 EV_min=0 overlay=path u=0.1; F6 margin 0.89555 vs 0.88429; v2.1 registry) |
| Downloader public-token docs | anonymous release path | ✅ README + docstring fixed, anonymous path active |
| features parquet SHA | `49cb6014…c48ba` | ✅ verified |
| prob_tables SHA | `03868101…54fea` | ✅ verified (release download) |
| phase1 bundle SHA | `dfd55109…17123b` | ✅ verified + extracted (1.1 GB) |
| core_fp16 SHA | `8c1556fa…ecb33` | ✅ verified (release download) |
| K2 CPU forecasts | Kaggle `tskdev/aurum-v2-cpu-preds` | ✅ downloaded |
| K3 neural forecasts | Kaggle `tskdev/aurum-v2-forecasts-neural` (== local survivor, hash `5741a86a…`) | ✅ downloaded |
| v1.3 artifacts (trades_all 4.5 MB) | Kaggle `tskdev/aurum-v13-artifacts` | ✅ downloaded |
| p4 flagship cap=3 (Step 2) | n=3,341, E[R]=+0.1478, SR **2.4443**, CAGR 54.56%, MaxDD 13.06% (bit-exact) | ⏳ pending |
| Un-throttled MaxDD | 23.92% | ⏳ |
| Pilot 0.5% un-throttled MaxDD | ~12.54% | ⏳ |
| Base-carry Sharpe | ~2.09 | ⏳ |
| r1 Fast Flagship cap=3 | +0.1466R, SR ~2.478 | ⏳ |
| r1 Clean EV cap=3 | ~+0.232R, SR ~2.611 | ⏳ |
| k6 bridge E0/E1/E2 | 2.6112 / ≈2.36–2.38 / partial-TP-halved (G5.10 honest spread) | ⏳ |
| U7 attribution (G5.12) | NW HAC(5) CAPM + L/S + 5-regime + DSR N=12..500, all finite | ⏳ |
| Red-team CI 7/7 | all executed with JSON verdicts | ⏳ |

## Open items / blockers
- None for Steps 1–7 (GitHub PAT supplied 2026-11-21; stored outside repo at `/tmp/.creds/`; pushes verified).
- Sandbox recycles between sessions: EVERY step must be committed+pushed before session end (Rule 1).

## Defect log (self-caught, Appendix-C culture)
- **D1 (Step 0, fixed):** hook self-test piped `git commit` through `tail`, masking the blocked-commit exit code; the `&&` chain then ran `git reset --soft HEAD~1` from `6ae2565`, silently regressing the branch to `07f1fd8`. Caught by push non-fast-forward + `git ls-remote` cross-check; fixed by `git rebase 6ae2565`. **Rule: never gate an `&&` chain on a piped `git commit`; check exit status directly.**

## Step log
- **Step 0 (2026-11-21):** clone @6ae2565 ✅ · branch `v2.1-remediation` pushed ✅ · pre-commit secret-blocker installed + self-tested (KGAT dummy → FATAL abort) ✅ · `.gitignore` extended for >25 MB v2.1 outputs ✅ · pinned env installed & recorded ✅ · all heavy assets restored & SHA-256 verified ✅ (features/prob_tables/phase1/core_fp16 + Kaggle K2/K3/v13). Push verified `29ece2e`.
- **Step 1 (2026-11-21):** repro gap #1 fixed (both test files resolve from repo root; `env -u PYTHONPATH`: v13 16/16 PASS, v2 all self-tests PASS) ✅ · repro gap #3 fixed (k2 standalone: `import sys as _sys`, sibling-path fallback, explicit imports, explicit FIT/OOS/ALL_FOLD years; pyflakes 0 undefined; smoke run reaches walk-forward) ✅ · Flaw #11 freeze `v1_3/config/frozen_policy.json` ✅ · downloader/README public-token docs ✅ · evidence `artifacts/v2_1/step1_tests.json` ✅.
