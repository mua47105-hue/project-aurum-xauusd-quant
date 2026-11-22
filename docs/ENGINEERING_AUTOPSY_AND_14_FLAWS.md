# Project AURUM — Engineering Autopsy: the 14 Flaws and 3 Reproducibility Gaps

> Spec v2.1 (Pilot Release v2.1.0) — companion document to
> `docs/reports/AURUM_Spec_v2_1_RedTeam_Remediated_Master_Report.docx`.
> Sources: the Independent Red-Team Forensic Audit (2026-11-11), the honest
> postmortem of the 2026-11-15 rollback, and the verified v2.1 remediation
> runs. Every "v2.1 status" below is backed by a committed, re-executable
> artifact.

## Part I — The 14 flaws

### Fixed in v2.0.2 and re-verified under v2.1 (Flaws #1–#8)

| # | Flaw | v2.0.2 fix | v2.1 re-verification |
|---|------|-----------|----------------------|
| 1 | Label leakage via absolute prices (serializer emitted raw price levels) | Serializer made dimensionless (ATR units, log-returns, z-scores, codebook ids) | rt1 prefix-invariance: 0/73,907 feature values changed by 500 garbage future bars; scale x1.37: 0 violations |
| 2 | Cohort memorization (k-NN cohort lookup could see same-cohort neighbours) | Strict >=96-bar cohort-gap mask before any use | G5.2a_cohort_gap green (used violations = 0) in local AND official Kaggle p4 runs |
| 3 | 151,936-vocab bottleneck | 97/111-token vocab slicing | G6.1_state: 338 tokens (280-380 band) in the live kernel |
| 4 | Ex-post overlay leak (overlay chosen with OOS knowledge) | Causal path-aware overlay, rule chosen on FIT only (F4) | Overlay ablation in every p4 run; policy frozen with overlay="path" |
| 5 | Same-bar position credit (z[t]·r[t] lookahead) | frozen_net lag convention (w[t] earns r[t+1]); G4.7 negative control | G4.7_causality_v201 green in k6 v2.1 |
| 6 | Fold-local z saturation of z_rff (per-fold constant ±1) | Causal trailing 504-bar standardization, re-anchored per fold (rff_z_v202) | rt7: z_rff strict-G4.8 PASS (fit std 0.425, oos 0.416) |
| 7 | DSR accounting with N=12 "effective families" vs the project's own N=170 ledger | Full-family + conservative DSR exported side by side | U7 honest N=12..500 panel (Section 6.3 of the master report) |
| 8 | Engine/tournament evaluation inconsistencies | OnlineExecutionEngine unification, Baseline-Gauntlet toggles | k6 v2.1 eval grid re-run end-to-end on the U2-fixed producers |

### Fixed by Spec v2.1 (Flaws #9–#14)

| # | Flaw | v2.1 remediation | Verified evidence |
|---|------|-----------------|-------------------|
| 9 | z_gbdt fold-local saturation: OOS 2023 mean z +0.954, 100% beyond \|z\|>0.9 — a constant long-gold position inflating M1_GBDT_v2 to +1.253 OOS | U2: the fold-local transform removed at the producer; rff_z_v202 applied to pred_gbdt; strict G4.8 extended to ALL producers | Collapse verified EXACT vs audit: 1.2534 → 0.1611; by-year -1.58/+0.002/+1.692 (audit -1.58/+0.00/+1.69); negative control fails as expected; Kaggle producer-side re-run k2 v8 |
| 10 | Bridge entry-bar credit: champion_exposure indexed from the decision bar, crediting the entry bar's close-to-close move (+0.23 OOS Sharpe) | U1: faithful E1 = pure shift z[i0+1:i1]; E2 adds partial-TP halving on [i_part+1, i1); i_part instrumentation exported per trade | E0 2.6248 / E1 2.3950 (audit family 2.36-2.40) / E2 2.5873; G5.10 PASS; defect D4 root-caused en route (see Part III) |
| 11 | Unpinned environment: unpinned xgboost flips the FIT grid (w=[0.5,0,0.5], lambda_var 0.25; cap1 OOS Sharpe swings 2.305→2.665) | requirements.txt + environment.lock.json + v1_3/config/frozen_policy.json loaded by default (--use-frozen-policy); G5.11; G6.0 in the live kernel | Flagship bit-exact under the frozen policy locally AND on Kaggle (float-identical 2.444339830427954); refit drift = 0 under the pins |
| 12 | 2023 OOS bar degradation (5,153 bars, -13.1% vs the 2016-22 mean) silently thinner than every other OOS year | G1.7_bar_coverage gate flags the degraded year; ex-2023 (2024-25) sensitivity rows exported with and without carry | G1.7 green in both official runs; ex-2023 Sharpe 3.18 (no carry) / 2.91 (base carry) vs 2.44 full-OOS |
| 13 | The v2.0.2 active model tournament contains zero OOS alpha (best NW t = -0.14) | U3: formal quarantine of M1-M6 + the M7 active arm with the top-60 pruning evidence recorded (-0.215 OOS); production registry = the v1.3 barrier engine + r=1 strategies | Quarantine registry in frozen_policy.json + the k2 ledger; rt7 verifies it |
| 14 | Decision-layer overfitting: the pooled posterior improves FIT Sharpe and delivers nothing OOS (the raw r=1 rule is better) | U4: AURUM_r1_Fast_Flagship and AURUM_r1_Clean_EV promoted to first-class registry strategies; --fast-r1 live path | r1_clean E[R] +0.2321 / SR 2.6114 / 718.6R vs flagship +0.1478 / 2.4443; rt2's permutation finding independently re-derives the mechanism (p↔R rank corr -0.046) |

## Part II — The 3 reproducibility gaps

| # | Gap (audit s2.2) | v2.1 fix | Verification |
|---|------------------|---------|--------------|
| 1 | tests/test_local_v2.py and test_local_v13.py could not run without an external PYTHONPATH (module dirs resolved against tests/) | Both resolve from the repo root | `env -u PYTHONPATH`: v13 16/16 PASS 0 fail; v2 all module self-tests PASS (artifacts/v2_1/step1_tests.json) |
| 2 | The pinned environment existed only in Kaggle's image | requirements.txt + environment.lock.json (python 3.12.16 sandbox vs Kaggle 3.13.18 recorded honestly; xgboost/pandas/numpy exact) | Bit-exact reproduction on BOTH the sandbox and Kaggle (the strongest possible pin verification) |
| 3 | aurum_v2/kernels/k2_cpu_models.py crashed standalone (undefined _sys; main() referenced symbols only the stitched Kaggle bundle defined) | import sys as _sys, sibling-path fallback, explicit v2_common/rff_ridge/online_engine imports + explicit FIT/OOS/ALL_FOLD years | pyflakes 0 undefined; smoke run reaches the M1_GBDT walk-forward on the restored 95,008×378 feature store |

## Part III — The v2.1 defect log (self-caught, Appendix-C culture)

The honest work report catalogued 13 defects from the pre-v2.1 era. The v2.1
remediation added six of its own, each self-caught, root-caused and fixed
within the cycle (see STATE.md for the full entries):

- **D1** — a piped `git commit` masked a blocked-commit exit code, letting a
  hook-test chain run `git reset --soft` from the base commit. Rule: never
  gate an `&&` chain on a piped git commit.
- **D2** — the M1 friction matrix indexed the unsorted resolve frame with
  execute()'s sorted row indices; every M1 row paired with the wrong trade.
  Caught by tracing one trade end-to-end against raw minute prices. Rule:
  execute()'s `sel` indexes `ex["sub"]`, never the resolve frame.
- **D3** — r1_clean subtracted cost inside the EV argmax and used the path
  overlay; the repo's v1.2 convention (aurum48_candidates) has neither. Rule:
  derive strategy conventions from the repo's own reference implementations.
- **D4** — the bridge collapse (E1/E2 at 0.67/0.51): 17.2% of flagship cap=3
  trades are one-bar trades, and the old clamp placed them on the decision
  bar while normal trades shifted. Also corrected en route: the literal
  close→open entry gap averages +4e-6 (≈0) — the artifact is the whole
  entry-bar close-to-close credit, not just the gap.
- **D5** — pandas-2.x Index comparisons return ndarrays; a `.values` call on
  one crashed the carry frame (the Nov-15 bug class recurring). Guard
  comments added; k6 re-run to completion.
- **D6** — the Kaggle bundle integration cluster: frozen-policy path
  resolution on /kaggle/src, stripped import blocks (traceback NameError),
  a missing phase1/cpu-preds dataset mount, an h3v use-before-definition and
  an array-in-boolean-context in the fast-r1 dual decision. Fixed across
  three push rounds (k2 v8, p4 v6, k6 v11, live v7); all fixes mirrored in
  the repo sources and the bundle builder.

## Part IV — The permanent operating law

1. GitHub is the only workspace: commit AND push after every step, verify
   `local HEAD == remote`.
2. Size tiering: <25 MB tracked in Git; heavy binaries live in GitHub
   Releases / Kaggle Datasets with SHA-256 manifests in
   scripts/download_heavy_assets.py.
3. The pre-commit secret blocker stays installed on every clone
   (`bash scripts/install_secret_hook.sh`); tokens never enter the repo.
4. STATE.md is the live checkpoint ledger — any session resumes in ten
   seconds from `git clone`.
5. Pre-stop emergency push whenever anything looks transient.
