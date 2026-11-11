# Data & Feature Dictionary — Project AURUM v2.0.2

> Canonical reference for the v2 feature store `aurum_v2_features.parquet`
> (Release asset, 253.7 MB): **95,008 H1 bars × 375 columns** spanning
> **2010-01-03 16:00 UTC → 2025-12-31 14:00 UTC** (16 years, 87,948 traded bars in the
> walk-forward era after warmup; coverage-by-year in `feature_meta.json`).
> 372 columns are model features; 3 are labels: `y_ret24_norm`, `y_dir24`, `y_ret1_norm`.
> Source of truth for everything below: `aurum_v2/aurum_features/*.py` (formulas),
> `artifacts/v2/k1_features/feature_meta.json` (groups), `ic_audit.parquet`
> (measured Rank-IC), `ic_family_summary.csv` (family rollups).
>
> **Measurement protocol for the IC column:** Spearman Rank-IC of each feature against
> the forward 24h normalized return `y_ret24_norm`, computed per walk-forward fold on the
> FIT window (2016-2022), n = 7 folds; t-stat = mean/std·√n. This is a *screening* audit —
> features were selected a-priori by family (Blueprint §3), not by IC mining, precisely so
> that the DSR/PBO accounting in `TRIAL_LEDGER_AND_DSR_AUDIT.md` stays honest. The
> appendix lists all 372 features with their measured t-stats.

---

## 0. The three label columns

| Column | Definition | Causal rule |
|---|---|---|
| `y_ret24_norm` | forward 24-hour log return of XAUUSD, normalized by trailing realized vol (the v2 regression target: `target y_ret24_norm`, position `z = tanh(2·pred)` at the engine) | uses bars *t+1 … t+24* only; never a feature |
| `y_dir24` | sign of the forward 24h return (auxiliary classification target) | same window |
| `y_ret1_norm` | next-bar (1h) normalized return (diagnostic / RLSBank target) | bar *t+1* only |

---

## 1. F1 — DST-Aware Exchange Clocks & Fixes (49 features)

**Module:** `aurum_v2/aurum_features/f1_clock.py` (F1 Clocks 24 + Sessions 25).

**Ground-truth timetable** (verified against LBMA/IBA auction rules, SGE benchmark rules
Nov-2019, WM/Refinitiv fix window, IANA tzdata; see module header):

- LBMA Gold AM auction 10:30 Europe/London · PM auction 15:00 London · Silver 12:00 London
- WM/Refinitiv FX fix 16:00 London (window 15:59:30-16:00)
- SGE Shanghai benchmark AM 10:15 / PM 15:15 Asia/Shanghai · Night session
- COMEX/Globex metals open/settle/break; Tokyo day session

**Feature groups and formulas:**
- `minutes_to_*` — signed minutes to the next occurrence of each fixture (LBMA AM/PM,
  Silver, WMR, SGE AM/PM, COMEX open/settle/break); DST-corrected via zoneinfo, so the
  "same" wall-clock auction is 23/25 hours apart across DST switches (the
  `days_to_dst_uk/us`, `dst_change_week_*` features encode the gap week explicitly).
- Session dummies & cluster flags: `sess_ldn_*`, `sess_ny_*`, `sess_ldn_ny_overlap`,
  `sess_tky_*`, `in_pm_fix_first4m`, … — Lou-Polk-Skouras session-split returns.
- **Pre-auction drift** (Blueprint 1 §5.1, S38/S39): 30-minute M1 drift into LBMA/COMEX
  fixtures (`pre_pm_fix_60m`, `pre_us_macro_drift`, …).
- **Post-fix reversal** (Krohn-Mueller-Whelan 2024 JF): signed reversal of the fix-window
  move over the following hour.
- **Blose et al. daytime asymmetry flags** (S40), **Tokyo gotobi** (5/10/20/25 day-of-month),
  **London month-end 4pm** fix flag.

**Causal rule:** all clock features are deterministic functions of the bar timestamp — no
lookahead by construction; the fix-window *return* features use only completed windows.

**Measured (2016-2022 FIT Rank-IC, 7 folds):** best `sess_ldn_pm_cluster` **t=+4.94**,
`sess_ny_day` +4.53, `sess_ldn_ny_overlap` +3.47, `minutes_to_LBMA_*` −2.72,
`pre_pm_fix_60m` +2.33. Family: median |t| 1.23, 12 features with |t|>2.

## 2. F2 — Realized Path Geometry, Jumps & Roughness (177 features)

**Module:** `aurum_v2/aurum_features/f2_realized.py`. Computed on M1 data inside each H1
bar and over trailing windows {1h, 4h, 24h, 120h (har)} for XAUUSD and the three context
assets (EURUSD, USDJPY, XAGUSD).

**Core estimators (all classical, constants verified against the primaries):**
- `RV` = Σ rᵢ² (Andersen-Bollerslev-Diebold-Labys 2001)
- `BV` = (π/2)(n/(n−1)) Σ|rᵢ₋₁||rᵢ| (Barndorff-Nielsen & Shephard 2004)
- `TQ` = n·μ₄/₃⁻³·(n/(n−2)) Σ|rᵢ₋₂|^{4/3}|rᵢ₋₁|^{4/3}|rᵢ|^{4/3} — **causally lagged** (see fix below)
- `MedRV`, `MinRV` (jump-robust), `RVpos`/`RVneg` (semivariance decomposition),
  `jumpShare = (RV−BV)/RV`, `logRV`, HAR aggregation `har120`
- Roughness: lag-1 autocorrelation of |r|, and the Hurst-style scaling residual
  `RV(1h)/RV(24h)` family.

**The v2.0 causality fix (Execution Mandate II.1):** the blueprint's tripower quarticity
and MedRV used the minute window [i−1, i, i+1] — at minute :59 of an H1 bar this peeks one
minute into the next hour. Fixed to the strictly trailing triple [i−2, i−1, i]; the
estimator is the same functional of a 3-term neighborhood, now fully inside the closed bar.

**Measured:** best `jumpShare_1h_har120` **t=+4.63**; the JPY realized block
(`BV_4h_jpy` −4.44, `MinRV_1h_jpy` −4.43, `RVpos_*_jpy` −4.47 …) is the strongest single
cluster in the store (safe-haven co-movement reversal). Family: 177 features, median |t|
1.49, **55 features |t|>2** — the workhorse family.

## 3. F3 — Depth-2/3 Multi-Asset Path Signatures & Lévy Areas (47 features)

**Module:** `aurum_v2/aurum_features/f3_signatures.py`. Theory: Chevyrev-Kormilitzin
(arXiv:1603.03788), Morrill et al. (arXiv:2006.00873), Reizenstein-Graham (arXiv:1802.08252).

**Definition:** for the d-dim path X over [a,b] (channels: normalized time — mandatory
time augmentation — plus log-returns of XAU/EUR/JPY/XAG and realized vol):
- Level 1: `S^{i}_{a,t} = X^i_t − X^i_a`
- Level 2: `S^{i,j}_{a,t} = ∫_a^t S^i_{a,s} dX^j_s`
- **Lévy area:** `A^{ij} = ½(S^{i,j} − S^{j,i})` — the signed enclosed area; `A>0` iff
  channel i *leads* channel j in the counter-clockwise sense; sign flips under path
  reversal — the who-moved-first detector that OHLC aggregates destroy.

Windows: 24h and 120h trailing; cross-asset pairs (xau_eur, xau_jpy, xau_xag, eur_jpy) plus
time-augmented self terms (`sig2_time_*`). `ll_*_area` are the log-Lévy areas.

**Causal rule:** signatures integrate the trailing window only; the level-2 integrals close
at the end of bar *t*.

**Measured:** best `sig2_eur_xau` **t=+3.53**, `sig2_time_eur` +2.19, `ll_xau_xag_area`
−2.17. Family: 47 features, median |t| 0.95, 4 with |t|>2 — a selective, high-conviction
family (order-of-events information, not redundant with F2 magnitudes).

## 4. F4 — Microstructural Toxicity & Liquidity (15 features)

**Module:** `aurum_v2/aurum_features/f4_toxicity.py`.

**Formulas:**
- **Bulk Volume Classification** (Easley, López de Prado & O'Hara 2012 RFS):
  `V_buy,i = V_i·Φ(ΔP_i/σ_dP)`, `V_sell = V_i − V_buy`; **VPIN** = volume-bucketed mean
  |V_buy−V_sell|/V_bucket (30 M1 bars per bucket, 48 buckets/24h). FX/CFD adaptation: tick
  volume (M1 price-update count) as the activity proxy (Bollershev-Domowitz 1993;
  Dacorogna et al. 2001 lineage).
- **Kyle's λ** (block version, rolling L with min_periods=240):
  `λ = cov(r_{t+1}, SV_t)/var(SV_t)` where SV is signed volume — **causality-fixed**:
  originally coded `r.shift(-1).rolling.cov(SV)` (peeks at minute :60 from :59); replaced
  by the mathematically identical `r.rolling.cov(SV.shift(1))` (same lag-1 covariance under
  stationarity of the lag operator, zero future peeking).
- **Amihud illiquidity** `|r|/dollar-volume` at 1h/4h; **CLV** (close location value)
  accumulation; order-flow imbalance proxies at both windows.

**Measured:** best `amihud_1h_xau` **t=−2.76**, `amihud_4h_xau` −2.54, `clv_xau` −2.09.
Family: 15 features, median |t| 1.55, 3 with |t|>2.

## 5. F5 — Online EW-RLS Cross-Asset Residualization (18 features)

**Module:** `aurum_v2/aurum_features/f5_residual.py`. State-space decomposition of the
XAUUSD hourly log return on the arbitrage/macro-hedging triangle (JPY, EUR, XAG):

```
y_t = α_t + β_JPY,t·x_JPY,t + β_EUR,t·x_EUR,t + β_XAG,t·x_XAG,t + ε_t
β_t = β_{t−1} + w_t        (random-walk coefficients)
```

estimated by **Recursive Least Squares with forgetting factor λ** (the information-form
Kalman filter for TVP regression; RLS(λ) ⇔ Kalman with Q/R tied to (1−λ)/λ):

```
a_t    = y_t − x_t'β_{t−1}                      (a-priori innovation — THE feature)
g_t    = P_{t−1}x_t / (λ + x_t'P_{t−1}x_t)      (gain)
P_t    = λ⁻¹P_{t−1} − g_t x_t'λ⁻¹P_{t−1}
β_t    = β_{t−1} + a_t·g_t
```

**Features (Channels A & B):** the innovation stream `chA_*` (gold return unexplained by
the triangle — the tradable residual), the exit/entry innovations `chB_exit`/`chB_entry`
(order-resolved variants), rolling β levels (`beta_jpy`, `beta_eur`, `beta_xag`), β
velocity, `spread_lvl` (rolling innovation dispersion), and RLS gain statistics.

**Causal rule:** the tradable residual is strictly the **a-priori** innovation (betas
update *after* observing) — the recursion is causal by construction; this survived every
audit untouched.

**Measured:** best `spread_lvl` **t=−4.21**, `beta_xag` −3.74, `chB_exit` +1.92. Family: 18
features, median |t| 0.95. (The innovation channels themselves are near-zero IC as raw
features — their value was as *inputs to the online engine*, where RLSBank consumes them.)

## 6. F6 — Multi-Horizon Saturating Trend Ladder & CPD Regime (50 features)

**Module:** `aurum_v2/aurum_features/f6_trend.py`. Synthesis of Carver's EWMAC ladder
(Blueprint 2 §3.6) with Blueprint 1 §5.2:

- **EWMAC ladder** s ∈ {2,4,8,16,32,64,128,256} h: `EWMAC_s = (EWMA_fast − EWMA_slow)/σ_s`
  with Carver scalar normalization (mean|F| ≈ 10, cap ±20), **CFM saturating response**
  `φ(x)=tanh(x/s*)`, s*=0.89, at 24/96/240h; **forecast diversification multiplier**
  (FDM) across the ladder; `cfm_onesided_504h_xau` — one-sided CFM momentum at the
  one-month horizon.
- **Man AHL damped MACD** (Baz et al. 2015) at (8,24)/(16,48)/(32,96) with the
  non-monotonic response `R(y) = y·exp(−y²/4)/0.89` + the `ahl_macd_composite`.
- **Lim et al. (2019)** multi-scale vol-normalized returns.
- **Wood-Roberts-Zohren (2021) fast causal changepoint detector** at lookbacks 21 and 126:
  `cpd_severity_21/126_xau` (severity direction per their Eq. 10) and `cpd_location_21/126`.
- Ex-ante vol `ex_ante_vol_63_xau` (the engine's vol-target input).

**Measured:** best `cfm_onesided_504h_xau` t=−1.81, `ex_ante_vol_63_xau` −1.61,
`ewmac_raw_64h_xau` −1.22. Family: 50 features, median |t| 0.53, none |t|>2 — **by
design**: the ladder's value is not cross-sectional IC but the *saturating* response and
cost control that let the engine trade it cheaply (the Carver_EWMAC benchmark strategy
posts 1.228 OOS Sharpe from this family, G5.4 speed-limit green).

## 7. mi_* — FRED Macro Innovations (15 features)

**Module:** v1.2 lineage carried over verbatim (Blueprint B1). External series (VIX,
DXY-related yields `y2`, `y10`, `y2y10` spread, real rates, CPI surprise proxy, initial
claims, …) enter **only as innovations**: each series is differenced at horizons d1/d5/d20
and z-scored with a long trailing window (`mi_vix_d5z` = 5-day change of VIX, z-scored).
Levels are banned (Flaw 1). FRED fetches run inside K1 with a 1-day publication lag
assumption; features are forward-filled at most 24h (stale macro is dropped to 0 by the
`mi_*` staleness mask).

**Measured:** best `mi_vix_d5z` t=+1.90, `mi_y2_d5z` −1.27, `mi_vix_d20z` +1.18. Family: 15
features, median |t| 0.27 — weak standalone, kept for regime conditioning and because the
v1.2 lineage showed them stabilizing fold-level variance.

---

## 8. Construction & integrity rules (the contract every feature obeys)

1. **Causality:** every feature at bar *t* uses information available at the close of bar
   *t*. The two historical violations (F2 TQ/MedRV triple, F4 Kyle λ) are fixed in the
   source and documented in `feature_meta.json → fixes_applied`; K1's G1.3 gate re-verifies
   causality by construction on every build.
2. **No raw prices, no raw dates** (Flaw 1): returns, ratios, innovations only; clocks are
   fixture-relative, never epoch.
3. **Coverage:** NaN policy = trailing-window warmup only; `coverage_by_year` audited
   (G1.6); no interpolation across gaps (missing hours stay missing).
4. **Units:** G1.5 verifies each family's unit contract (e.g. all realized measures in
   log-return² units; clocks in minutes; macro in z-scores).
5. **Freeze:** the store is content-hashed; K2/K3/K6/K7 mount `tskdev/aurum-v2-featstore`
   at a pinned version. Changing the store invalidates every downstream checkpoint (K6's
   data-signature checkpoint system) — by design.

## 9. Appendix — complete per-feature measured Rank-IC table

All 372 model features, grouped by family, sorted by |t| within family. Mean/std of the
per-fold Spearman Rank-IC against `y_ret24_norm`, FIT window 2016-2022, n=7 folds.


### F1 Clocks & Fixes (24 features)

| Feature | mean Rank-IC | std | t-stat (n=7 folds) |
|---|---|---|---|
| `minutes_to_LBMA_SILVER` | -0.0076 | 0.0074 | **-2.72** |
| `minutes_to_LBMA_AM` | -0.0076 | 0.0074 | **-2.72** |
| `minutes_to_LBMA_PM` | -0.0076 | 0.0074 | **-2.72** |
| `minutes_to_WMR_FIX` | -0.0076 | 0.0074 | **-2.72** |
| `pre_pm_fix_60m` | +0.0094 | 0.0107 | **+2.33** |
| `pre_us_macro_drift` | +0.0051 | 0.0087 | **+1.55** |
| `minutes_to_COMEX_BREAK` | +0.0036 | 0.0070 | **+1.36** |
| `minutes_to_COMEX_OPEN` | +0.0036 | 0.0070 | **+1.36** |
| `minutes_to_COMEX_SETTLE` | +0.0036 | 0.0070 | **+1.36** |
| `minutes_to_SGE_PM` | -0.0036 | 0.0077 | **-1.23** |
| `minutes_to_SGE_NIGHT` | -0.0036 | 0.0077 | **-1.23** |
| `minutes_to_SGE_AM` | -0.0036 | 0.0077 | **-1.23** |
| `pre_lbma_pm_drift` | -0.0071 | 0.0155 | **-1.22** |
| `wmr_post_fix_rev_signal` | +0.0079 | 0.0179 | **+1.17** |
| `dst_change_week_uk` | -0.0310 | 0.0767 | **-1.07** |
| `days_to_dst_uk` | -0.0123 | 0.0356 | **-0.91** |
| `vr_h1_m1_convexity` | -0.0039 | 0.0121 | **-0.85** |
| `dst_change_week_us` | +0.0166 | 0.0647 | **+0.68** |
| `in_pm_fix_first4m` | +0.0016 | 0.0090 | **+0.48** |
| `in_pm_fix_window` | +0.0016 | 0.0090 | **+0.48** |
| `days_to_dst_us` | -0.0072 | 0.0408 | **-0.47** |
| `pre_lbma_am_drift` | -0.0007 | 0.0075 | **-0.23** |
| `ny_ldn_offset_hours` | -0.0014 | 0.0485 | **-0.08** |
| `wmr_post_fix_ofi_signal` | +0.0000 | 0.0000 | **+nan** |

### F1 Sessions (25 features)

| Feature | mean Rank-IC | std | t-stat (n=7 folds) |
|---|---|---|---|
| `sess_ldn_pm_cluster` | +0.0072 | 0.0039 | **+4.94** |
| `sess_ny_day` | +0.0125 | 0.0073 | **+4.53** |
| `sess_ldn_ny_overlap` | +0.0074 | 0.0057 | **+3.47** |
| `sess_asia_night` | +0.0065 | 0.0053 | **+3.26** |
| `sess_globex_break` | -0.0036 | 0.0032 | **-3.01** |
| `is_asian_overnight` | -0.0058 | 0.0057 | **-2.72** |
| `is_us_daytime` | +0.0055 | 0.0057 | **+2.57** |
| `sess_london_day` | +0.0062 | 0.0084 | **+1.95** |
| `sess_asia_eu_handover` | -0.0010 | 0.0014 | **-1.93** |
| `ret6_london_am` | -0.0161 | 0.0246 | **-1.73** |
| `is_dst_us` | -0.0441 | 0.0748 | **-1.56** |
| `is_dst_uk` | -0.0426 | 0.0772 | **-1.46** |
| `is_london_month_end_4pm` | +0.0063 | 0.0124 | **+1.35** |
| `ret24_ldn_ny_overlap` | -0.0260 | 0.0534 | **-1.29** |
| `sess_shanghai_day` | -0.0018 | 0.0051 | **-0.92** |
| `ret6_ldn_ny_overlap` | -0.0121 | 0.0355 | **-0.90** |
| `ret24_london_am` | -0.0086 | 0.0285 | **-0.80** |
| `ret24_ny_pm` | +0.0062 | 0.0289 | **+0.57** |
| `ret24_asia` | +0.0044 | 0.0376 | **+0.31** |
| `is_tokyo_gotobi_pre_fix` | +0.0009 | 0.0160 | **+0.15** |
| `is_comex_first_hour` | -0.0002 | 0.0046 | **-0.14** |
| `sess_us_macro_hour` | -0.0002 | 0.0046 | **-0.14** |
| `is_dst_gap_week` | +0.0014 | 0.0485 | **+0.08** |
| `ret6_asia` | +0.0002 | 0.0202 | **+0.03** |
| `ret6_ny_pm` | -0.0002 | 0.0270 | **-0.02** |

### F2 Realized / Jumps / Roughness (138 features)

| Feature | mean Rank-IC | std | t-stat (n=7 folds) |
|---|---|---|---|
| `jumpShare_1h_har120` | +0.0541 | 0.0309 | **+4.63** |
| `BV_4h_jpy` | -0.0458 | 0.0273 | **-4.44** |
| `MinRV_1h_jpy` | -0.0376 | 0.0225 | **-4.43** |
| `BV_1h_jpy` | -0.0386 | 0.0231 | **-4.42** |
| `MedRV_1h_jpy` | -0.0382 | 0.0229 | **-4.41** |
| `MinRV_4h_jpy` | -0.0448 | 0.0271 | **-4.37** |
| `MedRV_4h_jpy` | -0.0455 | 0.0278 | **-4.34** |
| `logRV_4h_jpy` | -0.0445 | 0.0275 | **-4.29** |
| `RV_4h_jpy` | -0.0445 | 0.0275 | **-4.29** |
| `RV_1h_jpy` | -0.0384 | 0.0240 | **-4.23** |
| `logRV_1h_jpy` | -0.0384 | 0.0240 | **-4.23** |
| `logRV_24h_jpy` | -0.0543 | 0.0447 | **-3.21** |
| `RV_24h_jpy` | -0.0543 | 0.0447 | **-3.21** |
| `BV_24h_jpy` | -0.0553 | 0.0485 | **-3.02** |
| `jumpFlag_4h_jpy` | +0.0083 | 0.0075 | **+2.95** |
| `jumpZ_4h_jpy` | +0.0119 | 0.0111 | **+2.84** |
| `MinRV_24h_jpy` | -0.0531 | 0.0504 | **-2.78** |
| `MedRV_24h_jpy` | -0.0533 | 0.0509 | **-2.77** |
| `MedRV_1h_xau` | -0.0310 | 0.0302 | **-2.72** |
| `MinRV_1h_xau` | -0.0301 | 0.0299 | **-2.67** |
| `BV_1h_xau` | -0.0306 | 0.0305 | **-2.65** |
| `RV_1h_xau` | -0.0304 | 0.0310 | **-2.59** |
| `logRV_1h_xau` | -0.0304 | 0.0310 | **-2.59** |
| `MinRV_4h_xau` | -0.0353 | 0.0364 | **-2.57** |
| `MedRV_4h_xau` | -0.0356 | 0.0374 | **-2.52** |
| `BV_4h_xau` | -0.0354 | 0.0372 | **-2.51** |
| `garman_klass_xau` | -0.0292 | 0.0309 | **-2.50** |
| `rogers_satchell_xau` | -0.0282 | 0.0304 | **-2.46** |
| `MedRV_24h_eur` | -0.0293 | 0.0325 | **-2.39** |
| `logRV_4h_xau` | -0.0348 | 0.0387 | **-2.38** |
| `RV_4h_xau` | -0.0348 | 0.0387 | **-2.38** |
| `parkinson_xau` | -0.0280 | 0.0317 | **-2.34** |
| `jumpShare_4h_jpy` | +0.0077 | 0.0089 | **+2.30** |
| `MinRV_24h_eur` | -0.0278 | 0.0322 | **-2.29** |
| `jumpZ_1h_jpy` | +0.0079 | 0.0096 | **+2.17** |
| `BV_1h_xag` | -0.0289 | 0.0357 | **-2.14** |
| `BV_1h_har528` | -0.0357 | 0.0447 | **-2.11** |
| `BV_24h_eur` | -0.0268 | 0.0337 | **-2.10** |
| `gap_xau` | -0.0056 | 0.0072 | **-2.07** |
| `logRV_1h_xag` | -0.0274 | 0.0351 | **-2.06** |
| `RV_1h_xag` | -0.0274 | 0.0351 | **-2.06** |
| `MedRV_1h_xag` | -0.0280 | 0.0365 | **-2.03** |
| `jumpShare_1h_jpy` | +0.0079 | 0.0104 | **+2.02** |
| `MinRV_1h_xag` | -0.0272 | 0.0359 | **-2.01** |
| `jumpFlag_1h_xag` | +0.0159 | 0.0213 | **+1.97** |
| `MedRV_24h_xau` | -0.0393 | 0.0529 | **-1.96** |
| `YZ_528h_xau` | -0.0362 | 0.0506 | **-1.89** |
| `BV_4h_xag` | -0.0327 | 0.0459 | **-1.88** |
| `MinRV_24h_xau` | -0.0367 | 0.0521 | **-1.86** |
| `RV_24h_eur` | -0.0250 | 0.0363 | **-1.82** |
| `logRV_24h_eur` | -0.0250 | 0.0363 | **-1.82** |
| `MinRV_4h_xag` | -0.0305 | 0.0444 | **-1.81** |
| `MedRV_4h_xag` | -0.0307 | 0.0450 | **-1.81** |
| `jumpZ_24h_xag` | +0.0343 | 0.0508 | **+1.79** |
| `BV_24h_xau` | -0.0368 | 0.0552 | **-1.76** |
| `RV_4h_xag` | -0.0306 | 0.0460 | **-1.76** |
| `logRV_4h_xag` | -0.0306 | 0.0460 | **-1.76** |
| `YZ_24h_xau` | -0.0357 | 0.0537 | **-1.76** |
| `RV_24h_xau` | -0.0386 | 0.0583 | **-1.75** |
| `logRV_24h_xau` | -0.0386 | 0.0583 | **-1.75** |
| `RV_1h_har528` | -0.0306 | 0.0475 | **-1.70** |
| `BV_1h_har24` | -0.0366 | 0.0569 | **-1.70** |
| `RV_1h_har24` | -0.0384 | 0.0599 | **-1.70** |
| `YZ_120h_xau` | -0.0417 | 0.0652 | **-1.69** |
| `BV_1h_har120` | -0.0346 | 0.0557 | **-1.64** |
| `SJ_24h_xag` | +0.0249 | 0.0407 | **+1.62** |
| `SJ_1h_xau` | -0.0036 | 0.0061 | **-1.58** |
| `RV_1h_har120` | -0.0352 | 0.0592 | **-1.57** |
| `jumpFlag_4h_xag` | +0.0165 | 0.0279 | **+1.57** |
| `jumpFlag_1h_jpy` | +0.0073 | 0.0124 | **+1.56** |
| `SJ_4h_eur` | -0.0052 | 0.0096 | **-1.42** |
| `jumpShare_1h_xag` | +0.0142 | 0.0265 | **+1.42** |
| `SJ_4h_xau` | -0.0087 | 0.0165 | **-1.41** |
| `SJ_24h_eur` | -0.0148 | 0.0280 | **-1.39** |
| `jumpShare_1h_har528` | +0.0328 | 0.0695 | **+1.25** |
| `jumpZ_4h_xau` | +0.0092 | 0.0195 | **+1.25** |
| `jumpShare_24h_xag` | +0.0300 | 0.0657 | **+1.21** |
| `jumpZ_1h_xag` | +0.0126 | 0.0277 | **+1.20** |
| `SJ_4h_xag` | -0.0048 | 0.0112 | **-1.14** |
| `jumpShare_4h_xau` | +0.0092 | 0.0215 | **+1.14** |
| `BV_24h_xag` | -0.0311 | 0.0742 | **-1.11** |
| `MinRV_1h_eur` | -0.0079 | 0.0197 | **-1.06** |
| `RV_1h_eur` | -0.0079 | 0.0200 | **-1.05** |
| `logRV_1h_eur` | -0.0079 | 0.0200 | **-1.05** |
| `H_rough_vol_xau` | -0.0327 | 0.0828 | **-1.05** |
| `jumpShare_1h_har24` | +0.0105 | 0.0267 | **+1.04** |
| `MedRV_1h_eur` | -0.0078 | 0.0202 | **-1.02** |
| `BV_1h_eur` | -0.0077 | 0.0202 | **-1.01** |
| `MinRV_24h_xag` | -0.0267 | 0.0727 | **-0.97** |
| `jumpFlag_4h_eur` | +0.0064 | 0.0177 | **+0.96** |
| `RV_24h_xag` | -0.0267 | 0.0740 | **-0.96** |
| `logRV_24h_xag` | -0.0267 | 0.0740 | **-0.96** |
| `jumpShare_4h_xag` | +0.0156 | 0.0435 | **+0.95** |
| `jumpZ_1h_xau` | +0.0045 | 0.0128 | **+0.94** |
| `jumpZ_4h_xag` | +0.0148 | 0.0420 | **+0.93** |
| `gk_minus_rs_xau` | -0.0031 | 0.0090 | **-0.91** |
| `jumpFlag_24h_eur` | +0.0081 | 0.0237 | **+0.91** |
| `jumpFlag_24h_xag` | +0.0203 | 0.0596 | **+0.90** |
| `MinRV_4h_eur` | -0.0079 | 0.0233 | **-0.89** |
| `MedRV_4h_eur` | -0.0079 | 0.0237 | **-0.88** |
| `MedRV_24h_xag` | -0.0249 | 0.0747 | **-0.88** |
| `VRz8_xau` | +0.0177 | 0.0548 | **+0.86** |
| `jumpShare_24h_eur` | +0.0109 | 0.0343 | **+0.84** |
| `BV_4h_eur` | -0.0075 | 0.0244 | **-0.81** |
| `VRz4_xau` | +0.0162 | 0.0558 | **+0.77** |
| `jumpShare_4h_eur` | +0.0064 | 0.0221 | **+0.76** |
| `VR8_xau` | +0.0163 | 0.0597 | **+0.72** |
| `VR16_xau` | +0.0129 | 0.0483 | **+0.71** |
| `jumpZ_4h_eur` | +0.0062 | 0.0232 | **+0.71** |
| `SJ_1h_jpy` | -0.0032 | 0.0121 | **-0.70** |
| `jumpShare_1h_xau` | +0.0031 | 0.0117 | **+0.70** |
| `logRV_4h_eur` | -0.0064 | 0.0251 | **-0.68** |
| `RV_4h_eur` | -0.0064 | 0.0251 | **-0.68** |
| `VRz16_xau` | +0.0102 | 0.0456 | **+0.59** |
| `jumpZ_24h_eur` | +0.0071 | 0.0324 | **+0.58** |
| `jumpShare_1h_eur` | -0.0036 | 0.0164 | **-0.57** |
| `SJ_1h_har528` | -0.0158 | 0.0726 | **-0.57** |
| `hurst_VR_xau` | +0.0097 | 0.0475 | **+0.54** |
| `SJ_1h_xag` | -0.0011 | 0.0064 | **-0.47** |
| `VRz2_xau` | +0.0092 | 0.0565 | **+0.43** |
| `VR4_xau` | +0.0092 | 0.0580 | **+0.42** |
| `jumpZ_24h_xau` | +0.0052 | 0.0393 | **+0.35** |
| `SJ_24h_xau` | +0.0038 | 0.0294 | **+0.34** |
| `jumpFlag_1h_eur` | -0.0017 | 0.0137 | **-0.34** |
| `jumpFlag_24h_jpy` | +0.0058 | 0.0466 | **+0.33** |
| `jumpZ_1h_eur` | -0.0020 | 0.0161 | **-0.33** |
| `SJ_1h_har120` | -0.0076 | 0.0699 | **-0.29** |
| `jumpZ_24h_jpy` | +0.0033 | 0.0312 | **+0.28** |
| `jumpFlag_24h_xau` | +0.0047 | 0.0508 | **+0.24** |
| `SJ_1h_eur` | -0.0009 | 0.0117 | **-0.20** |
| `jumpFlag_1h_xau` | -0.0006 | 0.0082 | **-0.19** |
| `SJ_4h_jpy` | -0.0013 | 0.0222 | **-0.16** |
| `SJ_1h_har24` | +0.0014 | 0.0291 | **+0.13** |
| `VR2_xau` | +0.0026 | 0.0570 | **+0.12** |
| `jumpFlag_4h_xau` | -0.0010 | 0.0235 | **-0.11** |
| `jumpShare_24h_xau` | +0.0014 | 0.0452 | **+0.08** |
| `jumpShare_24h_jpy` | -0.0004 | 0.0211 | **-0.05** |
| `SJ_24h_jpy` | +0.0003 | 0.0526 | **+0.02** |

### F3 Depth-2 Signatures & Levy Areas (47 features)

| Feature | mean Rank-IC | std | t-stat (n=7 folds) |
|---|---|---|---|
| `sig2_eur_xau` | +0.0111 | 0.0083 | **+3.53** |
| `sig2_time_eur` | +0.0091 | 0.0110 | **+2.19** |
| `ll_xau_xag_area` | -0.0088 | 0.0108 | **-2.17** |
| `sig2_xau_jpy` | -0.0078 | 0.0096 | **-2.15** |
| `levy_xau_eur` | -0.0042 | 0.0056 | **-2.00** |
| `sig2_eur_xag` | +0.0045 | 0.0072 | **+1.67** |
| `sig2_xag_time` | -0.0061 | 0.0097 | **-1.67** |
| `sig2_xau_time` | -0.0055 | 0.0103 | **-1.41** |
| `sig1_xag` | -0.0042 | 0.0081 | **-1.36** |
| `ll_xau_xag_lag` | -0.0042 | 0.0081 | **-1.36** |
| `sig1_xau` | -0.0043 | 0.0085 | **-1.34** |
| `ll_xau_jpy_lead` | -0.0043 | 0.0085 | **-1.34** |
| `ll_xau_xag_lead` | -0.0043 | 0.0085 | **-1.34** |
| `ll_xau_xag_sym` | -0.0063 | 0.0127 | **-1.30** |
| `sig2_xag_jpy` | -0.0060 | 0.0126 | **-1.27** |
| `sig2_xag_eur` | +0.0046 | 0.0106 | **+1.16** |
| `sig2_eur_eur` | +0.0060 | 0.0141 | **+1.14** |
| `sig2_xag_xau` | -0.0055 | 0.0129 | **-1.12** |
| `ll_xau_jpy_d11` | -0.0057 | 0.0146 | **-1.03** |
| `ll_xau_xag_d11` | -0.0057 | 0.0146 | **-1.03** |
| `sig2_xau_xau` | -0.0057 | 0.0146 | **-1.03** |
| `ll_xau_xag_d22` | -0.0043 | 0.0111 | **-1.02** |
| `sig2_xag_xag` | -0.0043 | 0.0111 | **-1.02** |
| `sig2_time_xau` | +0.0023 | 0.0069 | **+0.88** |
| `ll_xau_jpy_area` | -0.0047 | 0.0147 | **-0.85** |
| `sig2_xau_xag` | -0.0057 | 0.0191 | **-0.78** |
| `levy_xau_jpy` | -0.0045 | 0.0153 | **-0.77** |
| `ll_xau_jpy_d22` | -0.0024 | 0.0098 | **-0.66** |
| `sig2_jpy_jpy` | -0.0024 | 0.0098 | **-0.66** |
| `sig2_xau_eur` | +0.0026 | 0.0113 | **+0.62** |
| `levy_xag_jpy` | -0.0032 | 0.0156 | **-0.55** |
| `sig2_jpy_xau` | +0.0027 | 0.0138 | **+0.51** |
| `sig2_jpy_eur` | -0.0023 | 0.0123 | **-0.50** |
| `sig2_time_jpy` | -0.0023 | 0.0145 | **-0.41** |
| `sig2_eur_time` | -0.0018 | 0.0119 | **-0.41** |
| `levy_eur_jpy` | +0.0027 | 0.0179 | **+0.40** |
| `sig1_eur` | +0.0017 | 0.0149 | **+0.30** |
| `sig2_eur_jpy` | +0.0015 | 0.0173 | **+0.23** |
| `levy_xag_eur` | -0.0006 | 0.0082 | **-0.21** |
| `sig2_time_xag` | +0.0008 | 0.0130 | **+0.16** |
| `ll_xau_jpy_sym` | -0.0008 | 0.0157 | **-0.14** |
| `sig2_jpy_xag` | +0.0008 | 0.0205 | **+0.11** |
| `sig1_jpy` | +0.0003 | 0.0088 | **+0.10** |
| `ll_xau_jpy_lag` | +0.0003 | 0.0088 | **+0.10** |
| `levy_xau_xag` | +0.0005 | 0.0133 | **+0.10** |
| `sig2_jpy_time` | +0.0002 | 0.0070 | **+0.08** |
| `sig2_time_time` | +0.0000 | 0.0000 | **+nan** |

### F4 VPIN / Kyle / Amihud (15 features)

| Feature | mean Rank-IC | std | t-stat (n=7 folds) |
|---|---|---|---|
| `amihud_1h_xau` | -0.0315 | 0.0302 | **-2.76** |
| `amihud_4h_xau` | -0.0356 | 0.0370 | **-2.54** |
| `clv_xau` | -0.0037 | 0.0047 | **-2.09** |
| `amihud_24h_xau` | -0.0358 | 0.0614 | **-1.55** |
| `clv_mean120_xau` | -0.0222 | 0.0464 | **-1.26** |
| `clv_mean6_xau` | -0.0030 | 0.0180 | **-0.43** |
| `clv_mean24_xau` | +0.0014 | 0.0337 | **+0.11** |
| `clv_tv_sum120_xau` | +0.0000 | 0.0000 | **+nan** |
| `clv_tv_sum24_xau` | +0.0000 | 0.0000 | **+nan** |
| `clv_tv_sum6_xau` | +0.0000 | 0.0000 | **+nan** |
| `kyle_lambda_lag_xau` | +0.0000 | 0.0000 | **+nan** |
| `kyle_lambda_xau` | +0.0000 | 0.0000 | **+nan** |
| `tick_volume_24h_xau` | +0.0000 | 0.0000 | **+nan** |
| `vpin_signed_xau` | +0.0000 | 0.0000 | **+nan** |
| `vpin_xau` | +0.0000 | 0.0000 | **+nan** |

### F5 EW-RLS Residual Channels (18 features)

| Feature | mean Rank-IC | std | t-stat (n=7 folds) |
|---|---|---|---|
| `spread_lvl` | -0.0334 | 0.0210 | **-4.21** |
| `beta_xag` | -0.0545 | 0.0385 | **-3.74** |
| `chB_exit` | +0.0222 | 0.0306 | **+1.92** |
| `pred` | -0.0035 | 0.0077 | **-1.19** |
| `R2_ew` | -0.0139 | 0.0330 | **-1.12** |
| `res_mom_A` | +0.0132 | 0.0365 | **+0.95** |
| `q_A` | +0.0132 | 0.0365 | **+0.95** |
| `eps` | +0.0053 | 0.0151 | **+0.93** |
| `sig_eq` | -0.0146 | 0.0569 | **-0.68** |
| `sig_eps` | -0.0146 | 0.0569 | **-0.68** |
| `beta_jpy` | +0.0118 | 0.0593 | **+0.53** |
| `spread_z` | -0.0125 | 0.0669 | **-0.50** |
| `beta_eur` | +0.0037 | 0.0433 | **+0.23** |
| `chB_active` | +0.0000 | 0.0000 | **+nan** |
| `conflict` | +0.0000 | 0.0000 | **+nan** |
| `ou_tau` | +0.0000 | 0.0000 | **+nan** |
| `ou_valid` | +0.0000 | 0.0000 | **+nan** |
| `q_B` | +0.0000 | 0.0000 | **+nan** |

### F6 Saturating EWMAC / CFM (46 features)

| Feature | mean Rank-IC | std | t-stat (n=7 folds) |
|---|---|---|---|
| `cfm_onesided_504h_xau` | -0.0155 | 0.0226 | **-1.81** |
| `ex_ante_vol_63_xau` | -0.0340 | 0.0560 | **-1.61** |
| `ewmac_raw_64h_xau` | -0.0187 | 0.0406 | **-1.22** |
| `ewmac_F_64h_xau` | -0.0187 | 0.0409 | **-1.21** |
| `norm_ret_252_xau` | -0.0204 | 0.0465 | **-1.16** |
| `ewmac_raw_128h_xau` | -0.0167 | 0.0412 | **-1.07** |
| `F_blend_xau` | -0.0132 | 0.0334 | **-1.05** |
| `ewmac_sat_256h_xau` | -0.0275 | 0.0730 | **-1.00** |
| `ewmac_F_128h_xau` | -0.0155 | 0.0414 | **-0.99** |
| `cfm_tanh_trend_240_xau` | -0.0147 | 0.0428 | **-0.91** |
| `ewmac_raw_256h_xau` | -0.0272 | 0.0832 | **-0.87** |
| `ewmac_F_256h_xau` | -0.0263 | 0.0834 | **-0.84** |
| `ewmac_sat_64h_xau` | -0.0160 | 0.0509 | **-0.83** |
| `ewmac_F_32h_xau` | -0.0184 | 0.0605 | **-0.80** |
| `ewmac_raw_32h_xau` | -0.0182 | 0.0612 | **-0.79** |
| `ewmac_sat_128h_xau` | -0.0088 | 0.0299 | **-0.78** |
| `fdm_xau` | -0.0224 | 0.0772 | **-0.77** |
| `ahl_macd_raw_32_96_xau` | -0.0179 | 0.0659 | **-0.72** |
| `norm_ret_1_xau` | -0.0023 | 0.0088 | **-0.70** |
| `norm_ret_8_xau` | -0.0056 | 0.0264 | **-0.56** |
| `norm_ret_126_xau` | -0.0145 | 0.0694 | **-0.55** |
| `cfm_onesided_120h_xau` | -0.0098 | 0.0488 | **-0.53** |
| `ewmac_sat_32h_xau` | -0.0161 | 0.0811 | **-0.52** |
| `ewmac_sat_16h_xau` | -0.0095 | 0.0494 | **-0.51** |
| `ewmac_F_2h_xau` | -0.0045 | 0.0257 | **-0.46** |
| `ewmac_raw_2h_xau` | -0.0045 | 0.0259 | **-0.46** |
| `ewmac_F_16h_xau` | -0.0077 | 0.0468 | **-0.44** |
| `ahl_macd_raw_16_48_xau` | -0.0099 | 0.0608 | **-0.43** |
| `ewmac_raw_16h_xau` | -0.0074 | 0.0471 | **-0.41** |
| `ahl_macd_composite_xau` | -0.0093 | 0.0599 | **-0.41** |
| `cfm_tanh_trend_96_xau` | -0.0071 | 0.0465 | **-0.41** |
| `ewmac_sat_2h_xau` | -0.0035 | 0.0248 | **-0.38** |
| `ahl_macd_damped_16_48_xau` | -0.0090 | 0.0755 | **-0.31** |
| `norm_ret_24_xau` | +0.0027 | 0.0316 | **+0.23** |
| `ewmac_F_4h_xau` | -0.0025 | 0.0318 | **-0.21** |
| `norm_ret_63_xau` | +0.0039 | 0.0518 | **+0.20** |
| `ewmac_raw_4h_xau` | -0.0023 | 0.0319 | **-0.19** |
| `ewmac_sat_4h_xau` | -0.0019 | 0.0283 | **-0.18** |
| `ahl_macd_damped_32_96_xau` | -0.0046 | 0.0699 | **-0.17** |
| `ewmac_F_8h_xau` | -0.0021 | 0.0381 | **-0.15** |
| `ewmac_raw_8h_xau` | -0.0021 | 0.0381 | **-0.15** |
| `ahl_macd_damped_8_24_xau` | -0.0013 | 0.0316 | **-0.11** |
| `ewmac_sat_8h_xau` | -0.0015 | 0.0355 | **-0.11** |
| `cfm_onesided_24h_xau` | +0.0007 | 0.0304 | **+0.06** |
| `cfm_tanh_trend_24_xau` | +0.0007 | 0.0304 | **+0.06** |
| `ahl_macd_raw_8_24_xau` | +0.0010 | 0.0430 | **+0.06** |

### F6 CPD Regime (4 features)

| Feature | mean Rank-IC | std | t-stat (n=7 folds) |
|---|---|---|---|
| `cpd_severity_21_xau` | -0.0134 | 0.0326 | **-1.09** |
| `cpd_severity_126_xau` | +0.0104 | 0.0312 | **+0.88** |
| `cpd_location_21_xau` | -0.0043 | 0.0141 | **-0.82** |
| `cpd_location_126_xau` | -0.0059 | 0.0291 | **-0.54** |

### FRED Macro Innovations (mi_*) (15 features)

| Feature | mean Rank-IC | std | t-stat (n=7 folds) |
|---|---|---|---|
| `mi_vix_d5z` | +0.0421 | 0.0586 | **+1.90** |
| `mi_y2_d5z` | -0.0148 | 0.0308 | **-1.27** |
| `mi_vix_d20z` | +0.0289 | 0.0651 | **+1.18** |
| `mi_dxy_rank60` | +0.0213 | 0.0645 | **+0.88** |
| `mi_y2_d20z` | -0.0170 | 0.0561 | **-0.80** |
| `mi_y10_d5z` | -0.0102 | 0.0407 | **-0.66** |
| `mi_vix_rank60` | +0.0099 | 0.0500 | **+0.52** |
| `mi_dxy_d20z` | +0.0063 | 0.0620 | **+0.27** |
| `mi_y2_rank60` | -0.0071 | 0.0788 | **-0.24** |
| `mi_y10_d20z` | +0.0025 | 0.0615 | **+0.11** |
| `mi_spread_d5z` | -0.0012 | 0.0453 | **-0.07** |
| `mi_dxy_d5z` | +0.0013 | 0.0575 | **+0.06** |
| `mi_y10_rank60` | +0.0013 | 0.0680 | **+0.05** |
| `mi_spread_d20z` | +0.0015 | 0.0838 | **+0.05** |
| `mi_spread_rank60` | +0.0003 | 0.0696 | **+0.01** |

### F2 Realized (extended) (40 features)

| Feature | mean Rank-IC | std | t-stat (n=7 folds) |
|---|---|---|---|
| `RVpos_1h_jpy` | -0.0378 | 0.0223 | **-4.47** |
| `RVpos_4h_jpy` | -0.0437 | 0.0259 | **-4.47** |
| `RVneg_1h_jpy` | -0.0361 | 0.0237 | **-4.04** |
| `RVneg_4h_jpy` | -0.0429 | 0.0285 | **-3.98** |
| `RVpos_24h_jpy` | -0.0529 | 0.0441 | **-3.18** |
| `RVneg_24h_jpy` | -0.0539 | 0.0492 | **-2.90** |
| `RVpos_4h_xau` | -0.0369 | 0.0360 | **-2.71** |
| `RVpos_1h_xau` | -0.0308 | 0.0303 | **-2.69** |
| `RVneg_1h_xau` | -0.0287 | 0.0302 | **-2.52** |
| `RVneg_4h_xau` | -0.0316 | 0.0398 | **-2.10** |
| `RVneg_1h_xag` | -0.0260 | 0.0328 | **-2.10** |
| `RVpos_1h_xag` | -0.0266 | 0.0361 | **-1.95** |
| `RVpos_24h_eur` | -0.0266 | 0.0369 | **-1.90** |
| `RVpos_24h_xau` | -0.0398 | 0.0566 | **-1.86** |
| `RVpos_4h_xag` | -0.0304 | 0.0435 | **-1.85** |
| `RVneg_4h_xag` | -0.0294 | 0.0456 | **-1.71** |
| `RVnegShare_24h_xag` | -0.0229 | 0.0391 | **-1.55** |
| `sig_ret_xau` | -0.0336 | 0.0578 | **-1.54** |
| `RVneg_24h_xau` | -0.0340 | 0.0599 | **-1.50** |
| `RVneg_24h_eur` | -0.0216 | 0.0382 | **-1.49** |
| `RVnegShare_4h_xau` | +0.0102 | 0.0186 | **+1.46** |
| `RVnegShare_1h_xau` | +0.0044 | 0.0092 | **+1.28** |
| `RVnegShare_24h_eur` | +0.0126 | 0.0287 | **+1.16** |
| `RVpos_1h_eur` | -0.0085 | 0.0208 | **-1.08** |
| `RVpos_24h_xag` | -0.0285 | 0.0724 | **-1.04** |
| `RVnegShare_4h_xag` | +0.0037 | 0.0103 | **+0.95** |
| `RVneg_24h_xag` | -0.0261 | 0.0730 | **-0.94** |
| `RVneg_1h_eur` | -0.0070 | 0.0198 | **-0.94** |
| `RVnegShare_4h_eur` | +0.0034 | 0.0110 | **+0.82** |
| `RVnegShare_1h_jpy` | +0.0028 | 0.0094 | **+0.80** |
| `RVpos_4h_eur` | -0.0073 | 0.0254 | **-0.76** |
| `RVnegShare_1h_xag` | +0.0026 | 0.0101 | **+0.68** |
| `RVnegShare_1h_har120` | +0.0146 | 0.0592 | **+0.65** |
| `RVneg_4h_eur` | -0.0057 | 0.0242 | **-0.62** |
| `RVnegShare_4h_jpy` | +0.0033 | 0.0178 | **+0.49** |
| `RVnegShare_1h_har528` | -0.0062 | 0.0399 | **-0.41** |
| `RVnegShare_1h_har24` | +0.0073 | 0.0479 | **+0.40** |
| `RVnegShare_1h_eur` | +0.0010 | 0.0118 | **+0.22** |
| `RVnegShare_24h_xau` | -0.0010 | 0.0271 | **-0.10** |
| `RVnegShare_24h_jpy` | +0.0001 | 0.0367 | **+0.01** |

---

*Generated from `artifacts/v2/k1_features/ic_audit.parquet` (372 rows) — regenerate the
appendix with `python3 scripts/gen_feature_table.py` (repo root context).*
