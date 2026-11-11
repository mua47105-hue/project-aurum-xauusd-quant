# AGENTS.md — AI Agent Onboarding for Project AURUM

> **Read this first.** This file is written for future AI coding agents (Claude Code,
> OpenCode, Z.ai, Cursor, …) who inherit this repository. It encodes everything an agent
> needs to operate safely on this codebase: what exists, how it fits together, and the
> hard rules that must never be broken. Human engineers will find it useful too.

---

## 1. Project identity (30 seconds)

Project AURUM is a quantitative trading-decision system for XAU/USD (spot gold), H1 bars,
built 2026-11-06 → 2026-11-10 in five spec generations (v1.0 → v2.0.2). Everything ran on
free Kaggle compute. The final state (v2.0.2) is an 18-strategy walk-forward tournament with
a strict statistical-honesty contract. The current best transparent benchmark is
**M7 Core-Satellite (OOS 2023-25 Sharpe 2.635, 2.501 at 2.0× costs, DSR 0.9407 — the 0.95
gate honestly FAILED)**. Read `README.md` §1-§4 before touching anything.

**Where the truth lives:** gate JSONs and parquets in `artifacts/`, NOT prose. If a chat log,
a docx, and `gates.json` disagree, `gates.json` wins. If code and artifacts disagree, rerun
the code.

## 2. Directory map (what to touch, what to never touch)

| Path | Role | Agent policy |
|---|---|---|
| `aurum_v2/` | v2.0.2 modular source (features, kernels, model) | Editable after reading §4 rules |
| `v1_3/` | v1.3 source + experiments (the champion lineage) | **Frozen.** Read-only reference; the v1.3 champion must stay reproducible |
| `kaggle_kernels/` | As-run standalone Kaggle scripts (16) | Read-only historical record of what executed |
| `scripts/` | Build / bundle / report / chart tooling | Editable |
| `artifacts/` | Gates, ledgers, parquets, model heads | **Never edit by hand.** Regenerate via kernels only |
| `reports/`, `figures/` | Generated outputs (docx/pdf/png) | Regenerate via scripts; do not hand-edit |
| `docs/` | Institutional memory (this file, autopsies, dictionary, journal) | Append-only; keep history |
| `data/` | Small core data (h1 parquet, codebooks, vocab ids) | Read-only |
| `tests/` | `test_local.py` (v2 + v1.3), live-logic T1-T7, smokes | Run before every push; extend when adding logic |
| `~/.aurum_token` (outside repo) | Kaggle API token | See §5 — never commit |

## 3. The two build systems

### 3.1 v2: `scripts/v2/bundle.py` + `build_k*.py`

The v2 kernels exist in **two forms**:

1. **Modular source** — `aurum_v2/` (`aurum_features/`, `kernels/`, `model/`). This is what
   you edit. `aurum_v2/test_local.py` runs shape/causality sanity checks locally in seconds.
2. **Standalone bundles** — `scripts/v2/build_k6.py` etc. concatenate the modules (via
   `bundle.py`) into one self-contained `kernel.py` (e.g. `kaggle_work/v2/k6/kernel.py`,
   97 KB) plus a `kernel-metadata.json`. Kaggle only ever sees the bundle.

So the edit loop is:

```bash
python3 aurum_v2/test_local.py                 # 1) local sanity
python3 scripts/v2/build_k6.py                 # 2) rebuild bundle (writes kaggle_work/v2/k6/)
# 3) inspect the diff of kaggle_work/v2/k6/kernel.py — it must contain your change
kaggle kernels push -p kaggle_work/v2/k6       # 4) push (needs KAGGLE_API_TOKEN)
python3 scripts/kgapi.py k-status tskdev/aurum-v2-backtest   # 5) poll status
python3 scripts/kgapi.py k-download-all tskdev/aurum-v2-backtest kaggle_work/v2/k6_out1  # 6) pull outputs
```

**The bundle is generated, never hand-edited.** If you find yourself editing
`kaggle_work/v2/k6/kernel.py` directly, stop — edit `aurum_v2/` and re-bundle.

### 3.2 v1.3: single-file kernels

The v1.3 kernels are already self-contained single files (`kaggle_kernels/v1_3/*.py`,
50-130 KB each). They embed their own library code. Treat them as frozen binaries that
happen to be Python.

### 3.3 Reports and charts

- Charts: `AURUM_CH_K6=<k6 out dir> python3 scripts/v2/make_charts.py` → 11 PNGs (200 DPI).
  Env overrides: `AURUM_CH_K1/K2/K2_PRE/K3/K6`.
- v2 master report: `cd scripts/v2 && npm install && node report/generate.js` → docx
  (content lives in `report/content_1.js` … `content_4.js`; numbers are read live from the
  artifact JSONs via `report/helpers.js` `jload()` — never hardcode a number into prose).
  Then `python3 report/postcheck.py` — **the target is 0 errors, 0 warnings**; it checks
  headings, 11 inline images, placeholder leftovers, table integrity, TOC, PDF page count,
  and renders every PDF page to detect blank pages.
- PDF: postcheck itself runs `soffice --headless --convert-to pdf`.

## 4. The hard rules (break any and the program's numbers become fiction)

1. **Causality.** A feature may only use information through the close of bar *t* to predict
   bar *t+1*+ returns. Two shift(-1) bugs were caught and fixed in v2.0.0 (F2 tripower
   quarticity, F4 Kyle lambda — see module headers). Positions: `z[t-1]·r[t]`, never
   `z[t]·r[t]` (Flaw #7, gate **G4.7**). Any new predictor must pass a same-bar trap test.
2. **No unanchored normalization.** Never standardize a prediction stream by fold-interior
   statistics that leak scale across the gate (Flaw #8). Use trailing/causal windows only
   (see `rff_z_v202` in `aurum_v2/kernels/k2_cpu_models.py`), and gate with **G4.8**
   (std > 0.05, |z|>0.99 fraction < 5%, turnover > 2.0).
3. **Walk-forward discipline.** FIT 2016-2022 / OOS 2023-2025 with purge=24/embargo=24
   (`aurum_v2/model/purge.py`). Never evaluate on OOS during development; never tune on it.
4. **Ledger every trial.** Any model/config evaluation must append to the trial ledger
   (`trial_ledger.jsonl`: model, config, config_hash, fit_metric, stage, kernel, ts). DSR
   and PBO are computed over the ledger; an unledgered trial silently corrupts the
   multiple-testing accounting (see `docs/TRIAL_LEDGER_AND_DSR_AUDIT.md`).
5. **Gates are contracts.** `gates.json` files define pass/fail per stage (G0-G6). A failed
   gate ships as failed (see v2.0.2 G6.1). Never weaken a gate threshold to make a number
   pass; if the threshold must change, that is a new spec version with a documented reason.
6. **Costs before Sharpe.** All Sharpe numbers are net of `COST_R = 5e-5` per unit exposure
   change at 1.0×; 1.5× and 2.0× ladders are mandatory for any shipping candidate; the
   Carver speed limit (annual cost ≤ ⅓ pre-cost SR) gates shipping (G5.4).
7. **v1.3 artifacts are the reference.** `artifacts/v2/v13_reference/` (and the
   `tskdev/aurum-v13-artifacts` dataset on Kaggle) are the frozen v1.3 champion outputs.
   K6/K7 mount them read-only. Never regenerate them from modified code.
8. **Secrets stay out.** See §5. The pre-commit gate must stay green.

## 5. Credentials & secrets (read twice)

- Kaggle API tokens (accounts `tskdev`, `ranigain`) and the GitHub token are **never in this
  repo**. They live outside the tree (e.g. `~/.aurum_token`, `~/.kaggle/kaggle.json`, or env
  vars) and are read at runtime: `os.environ.get("KAGGLE_KEY")`,
  `os.environ.get("GITHUB_TOKEN")`, or `KAGGLE_API_TOKEN` for `scripts/kgapi.py`.
- Pattern used by helper scripts (keep it): a tiny shell file that `cat`s an out-of-repo
  token file into an env var — the script itself contains no secret.
- **Mandatory pre-commit gate** (already part of the release process; keep it):

  ```bash
  grep -rnE "KGAT[_][A-Za-z0-9]+|ghp[_][A-Za-z0-9]+|github_pat[_][A-Za-z0-9_]+" \
    --include="*" . && echo "SECRET LEAK — ABORT" || echo "clean"
  ```

  (Must print `clean` and zero matches. Note: historical Kaggle tokens were rotated/revoked
  after the program ended; treat any token found in old logs as dead, and never commit a new
  one.)
- The Kaggle datasets referenced by the kernels are private to `tskdev`/`ranigain`. A new
  agent needs the owner to re-issue a Kaggle API token and a GitHub fine-grained PAT with
  `Contents: read/write` on this repo to use `scripts/download_heavy_assets.py`.

## 6. Kaggle dataset & kernel mounts (how the kernels see data)

Inside a Kaggle kernel, attached datasets mount read-only at:

```
/kaggle/input/<dataset-slug>/...            # newest version
/kaggle/input/datasets/<owner>/<slug>/...   # some images use this layout
```

The v2 kernel attachment map (from `kernel-metadata.json` `dataset_sources`):

| Dataset | Used by | Content |
|---|---|---|
| `tskdev/aurum-phase1-data` (v6) | v1.3 kernels | M1/H1 parquets, labels, cohorts, vocab, codebooks |
| `tskdev/aurum-v2-featstore` | K2/K3/K6/K7 | `aurum_v2_features.parquet` (253.7 MB), feature_meta, IC audit |
| `tskdev/aurum-v2-cpu-preds` (v3) | K6/K7 | `v2_cpu_forecasts.parquet` (de-saturated z_rff) + gates |
| `tskdev/aurum-v2-forecasts-neural` | K6 | `v2_neural_forecasts.parquet` + rho_log + ledger |
| `tskdev/aurum-v13-artifacts` | K6/K7 | v1.3 champion: trades_all, policy_fit, backtest_main |
| `tskdev/aurum-v2-workspace-backup` | (backup only) | full workspace snapshot (302 files) |

**Checkpoint invalidation:** K6 caches per-strategy results in `ckpt/evalgrid/*.json` +
`*.parquet`, keyed by a data signature (hash of inputs + code rev). If you change an input
(e.g. re-push cpu-preds), the signature changes and stale checkpoints are **invalidated
automatically**; unchanged strategies are reused (that is why a full K6 rerun is ~3 min).
Never hand-delete `ckpt/` unless you want a cold rebuild (~40 min).

**GPU quota reality:** 30 h/week per account, 2×T4. The K3 neural tournament was split into
`aurum-v2-neural-compact` + `aurum-v2-neural-completion` specifically to spread quota across
the weekly reset. If you resume K3 work, check quota first
(`kaggle config view` / the API) and leave a 2.5 h safety floor.

## 7. Where every number in the docs comes from

| Claim | Source of truth |
|---|---|
| 18-strategy FIT/OOS Sharpe at 1.0/1.5/2.0× | `artifacts/v2/k6_backtest/backtest_v2.json` |
| Champion compare (2.6351 / 2.5012 / DSR 0.9407 / …) | `artifacts/v2/k6_backtest/champion_compare.json` |
| Gate register (11/12 green, G6.1 fail) | `artifacts/v2/k6_backtest/gates.json` |
| DSR/PBO panel (N=170, PBO 0.0049) | G6.1 detail in the same gates.json |
| z_rff de-saturation (std 0.42, sat 0%) | `artifacts/v2/k2_cpu_preds_v3/gates.json` + `k2_meta.json` |
| Spectral radius ρ=0.5012-0.5014 | `artifacts/v2/k3_neural/rho_log.json` |
| Feature families & IC t-stats | `artifacts/v2/k1_features/` (feature_meta.json, ic_audit.parquet, ic_family_summary.csv) |
| v1.3 flagship (2.444 / DSR 0.895) | `artifacts/v1_3/p4_backtest/backtest_main.json` + v2 translation in champion_compare.json |
| Live decision (as of 2025-12-31 14:00) | `artifacts/v2/k7_live_daily/decision_v2.json` |
| Trial counts (551 rows) | `artifacts/v2/*/trial_ledger.jsonl` |

If you are asked for any number: open the JSON, do not quote from memory.

## 8. Resuming the project — suggested first moves

1. `python3 tests/test_local_v2.py` — confirm the source tree is sane.
2. `python3 scripts/download_heavy_assets.py` — restore the 253.7 MB feature store + weights.
3. Read `docs/EXECUTION_JOURNAL.md` + `docs/AURUM_Current_State_Verified.md` for the session
   history, then `docs/ENGINEERING_AUTOPSY_AND_8_FLAWS.md` for the traps already solved.
4. Verify Kaggle state: `python3 scripts/kgapi.py k-status tskdev/aurum-v2-backtest`.
5. Anything new you build: new kernel version, ledgered trials, gates extended (G7.x), and
   the DSR panel recomputed with the new trials included — never compared against the old
   panel.

## 9. Known dead ends (do not re-litigate)

- **Neural nets on v2 features (M3/M4/M5):** all fail DSR (best: M3c 0.1087). The contractive
  loop was correctly implemented (ρ governed < 0.95) and still lost. The signal ceiling for
  v2 features on CPU-priced models is ~0.9-1.0 OOS Sharpe (M1 GBDT).
- **Saturated M2 RFF-Ridge as a "winner":** 1.87-2.16 pre-fix Sharpe was passive gold
  exposure in disguise (constant +1.0 position). De-saturated: −0.744. It stays in the
  tournament as the reference-only leg of M6/M7-Active and as the canonical Flaw #8 example.
- **Stacking hourly v1.3 signals without the resolution lock:** unbounded pyramiding
  (7,207 trades, 127.7R drawdown). The cap=1/cap=3 engine + Carver buffer is the fix; do not
  remove them.
- **k-NN soft labels on cohort states (K=200):** memorization, Brier 0.0022 soft vs 0.247
  realized. Realized triangular-smoothed targets only.
