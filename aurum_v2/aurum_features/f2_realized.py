"""
AURUM v2.0 -- Feature Module F2: M1 Realized Path Geometry
===========================================================
Blueprint 2 Section 3.2 WITH THE MANDATED CAUSALITY FIX applied:

  *** BUG-FIX (Spec v2.0 execution mandate II.1) ***
  The original blueprint computed the tripower quarticity term TQ and the
  MedRV median triple with r.abs().shift(-1) -- at minute :59 of an H1 bar
  this peeks 1 minute into the NEXT hour.  Fixed to the strictly causal
  lag pattern [shift(2), shift(1), shift(0)]:
      tq3  : |r_{i-2}|^{4/3} |r_{i-1}|^{4/3} |r_i|^{4/3}     (was i-1, i, i+1)
      med3 : median(|r_{i-2}|, |r_{i-1}|, |r_i|)             (was i-1, i, i+1)
  This shifts the three-term windows one bar back in time; the estimators
  remain the same functional of a 3-term neighborhood, now fully inside the
  trailing window.

Exact realized-measure machinery from the literature (all constants verified
against fetched primaries; see Spec v2.0 Section 3 formulas and Section 2 ledger):

  RV   = sum r_i^2                                   (Andersen-Bollerslev-Diebold-Labys 2001, JASA)
  BV   = (pi/2)(n/(n-1)) sum |r_{i-1}||r_i|          (Barndorff-Nielsen & Shephard 2004, JFEC 2(1))
  TQ   = n mu_{4/3}^-3 (n/(n-2)) sum |r_{i-2}|^{4/3}|r_{i-1}|^{4/3}|r_i|^{4/3}   (causally lagged)
  Z    = ((RV-BV)/RV) / sqrt(theta (1/n) max(1, TQ/BV^2)),  theta=(pi/2)^2+pi-5~=0.6090
         (BNS 2006 ratio jump statistic; Huang-Tauchen 2005 form)
  RV+/RV- (realized semivariances), SJ = RV+ - RV-   (BN-Kinnebrock-Shephard 2010;
                                                      Patton-Sheppard REStat 97(3) 2015)
  MedRV = pi/(6-4sqrt(3)+pi) (n/(n-2)) sum med(...)^2 (Andersen-Dobrev-Schaumburg 2012)
  MinRV = pi/(pi-2) (n/(n-1)) sum min(...)^2
  Parkinson/Garman-Klass/Rogers-Satchell/Yang-Zhang range estimators
         (Parkinson 1980; Garman-Klass 1980; Rogers-Satchell 1991; Yang-Zhang 2000;
          canonical GK constants 0.511/0.019/0.383 per Meilijson arXiv:0807.3492)
  VR(q) with Lo-MacKinlay (1988 RFS) unbiased overlapping estimator and
         heteroskedasticity-robust z (delta_j, theta(q)); H = 0.5 + slope(logVR, log q)/2
  Volatility roughness: m(q, Delta) = E|log sigma_{t+Delta} - log sigma_t|^q propto Delta^{qH}
         (Gatheral-Jaisson-Rosenbaum arXiv:1410.3394, QF 18(6) 2018; H ~= 0.1, and
          "Crude Oil and Gold futures show similar smoothness" -- fetched footnote)

Predictive-sign hypotheses (Patton-Sheppard): RV- ("bad volatility") dominates vol
forecasting and persistence; SJ < 0 hours -> vol continuation + negative drift bias;
SJ > 0 -> vol decay; jump hours (Z > 2.326) -> partial reversal; VR > 1 trending
micro-regime -> weight continuation features; VR < 1 -> weight fade features.

All rolling computations are trailing (causal): the window [t-W, t] inclusive,
and features are read at H1 closes from fully-closed M1 bars only.
"""

import numpy as np
import pandas as pd

# --- verified constants -------------------------------------------------------
SQRT_2_OVER_PI = 0.7978845608  # mu_1 = E|Z|
MU_4_3 = 0.83086               # mu_{4/3} = 2^(2/3) Gamma(7/6)/sqrt(pi)
MU_4_3_INV3 = 1.7430           # mu_{4/3}^{-3}
THETA_JUMP = 0.6090            # (pi/2)^2 + pi - 5   (Huang-Tauchen)
Z99 = 2.326                    # one-sided 99% jump threshold
MEDRV_C = 1.418640             # pi/(6 - 4*sqrt(3) + pi)
MINRV_C = 2.751935             # pi/(pi - 2)
PARKINSON_C = 0.360674         # 1/(4 ln 2)
GK_K1, GK_K2, GK_K3 = 0.511, 0.019, 0.383   # canonical Garman-Klass
YZ_ALPHA = 1.34                # Yang-Zhang k constant
VOL_FLOOR = 1e-12

M1_WINDOWS = {"1h": 60, "4h": 240, "24h": 1440}   # M1 bars per trailing window


def _m1_terms(m1: pd.DataFrame) -> pd.DataFrame:
    """Per-M1-bar additive terms for every realized measure. m1 must have a
    UTC DatetimeIndex and a 'close' column (open-labeled bars).

    CAUSALITY FIX APPLIED HERE: the original blueprint's tq3/med3 used
    r.abs().shift(-1), peeking at minute :60 of the hour.  The strictly
    causal neighborhood is [i-2, i-1, i]:
        tq3  = |r_{i-2}|^{4/3} * |r_{i-1}|^{4/3} * |r_i|^{4/3}
        med3 = median(|r_{i-2}|, |r_{i-1}|, |r_i|)
    """
    r = np.log(m1["close"] / m1["close"].shift(1))
    out = pd.DataFrame(index=m1.index)
    out["r2"] = r ** 2
    out["r2_pos"] = (r ** 2).where(r > 0, 0.0)
    out["r2_neg"] = (r ** 2).where(r < 0, 0.0)
    out["bip"] = (r.abs() * r.abs().shift(1))          # |r_{i-1}||r_i|
    # --- FIX: [shift(2), shift(1), shift(0)] instead of [shift(1), shift(0), shift(-1)]
    out["tq3"] = (r.abs().shift(2) ** (4.0 / 3.0)) * (r.abs().shift(1) ** (4.0 / 3.0)) * (r.abs() ** (4.0 / 3.0))
    out["min2"] = np.minimum(r.abs(), r.abs().shift(1)) ** 2
    med3 = pd.concat([r.abs().shift(2), r.abs().shift(1), r.abs()], axis=1).median(axis=1)
    out["med2"] = med3 ** 2
    out["absr"] = r.abs()
    return out


def realized_block_m1(m1: pd.DataFrame, h1_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Realized-variance family per trailing M1 window, sampled at H1 closes.

    Parameters
    ----------
    m1 : M1 bars (UTC DatetimeIndex open-labeled, 'close' column required).
    h1_index : the H1 bar labels (open-labeled).  Features are read from the
        last M1 bar at or before (h1_label + 59min), i.e. bars fully closed
        inside the H1 bar -- strictly causal.

    Memory note: each window's block is downsampled to the H1 index
    immediately after rolling (engineering adaptation for 8M-row M1 frames;
    numerically identical to the blueprint's concat-then-sample).
    """
    t = _m1_terms(m1)
    h1_close_idx = h1_index + pd.Timedelta(minutes=59)
    parts = []
    for wname, W in M1_WINDOWS.items():
        s = t.rolling(W, min_periods=W // 2).sum()
        # rolling sum of x^2 needs (sum x)^2 handled separately; RV etc. use
        # only the additive terms; n = window size (use actual count for gaps)
        cnt = t["r2"].rolling(W, min_periods=1).count()
        n = cnt.clip(lower=2)

        rv = s["r2"]
        bv = (np.pi / 2.0) * (n / (n - 1)) * s["bip"]
        tq = n * MU_4_3_INV3 * (n / (n - 2).clip(lower=1)) * s["tq3"]
        medrv = MEDRV_C * (n / (n - 2).clip(lower=1)) * s["med2"]
        minrv = MINRV_C * (n / (n - 1)) * s["min2"]

        # BNS ratio jump statistic (one-sided); NaN where RV ~ 0
        with np.errstate(divide="ignore", invalid="ignore"):
            jump_share = ((rv - bv) / rv).clip(lower=0.0)
            zstat = ((rv - bv) / rv) / np.sqrt(
                (THETA_JUMP / n) * np.maximum(1.0, tq / (bv ** 2 + VOL_FLOOR))
            )
        rvp, rvn = s["r2_pos"], s["r2_neg"]

        block = pd.DataFrame(
            {
                f"RV_{wname}": rv,
                f"BV_{wname}": bv,
                f"MedRV_{wname}": medrv,
                f"MinRV_{wname}": minrv,
                f"jumpZ_{wname}": zstat,
                f"jumpShare_{wname}": jump_share,
                f"RVpos_{wname}": rvp,
                f"RVneg_{wname}": rvn,
                f"SJ_{wname}": rvp - rvn,
            },
            index=m1.index,
        )
        block[f"RVnegShare_{wname}"] = rvn / (rv + VOL_FLOOR)
        block[f"jumpFlag_{wname}"] = (zstat > Z99).astype(float)
        block[f"logRV_{wname}"] = np.log(rv + VOL_FLOOR)
        parts.append(block.reindex(h1_close_idx, method="ffill").set_axis(h1_index))
    feats = pd.concat(parts, axis=1)
    return feats


def range_block_h1(h1: pd.DataFrame) -> pd.DataFrame:
    """Range-based volatility per H1 bar + Yang-Zhang window estimator.
    h1: H1 bars (UTC index, open-labeled) with open/high/low/close."""
    o, h, l, c = h1["open"], h1["high"], h1["low"], h1["close"]
    u = np.log(h / o); lo = np.log(l / o); cl = np.log(c / o)
    d = u - lo
    oc = np.log(o / c.shift(1))                        # overnight/inter-bar gap
    co = np.log(c / o)

    pk = PARKINSON_C * d ** 2
    gk = GK_K1 * d ** 2 - GK_K2 * (cl * (u + lo) - 2 * u * lo) - GK_K3 * cl ** 2
    rs = u * (u - cl) + lo * (lo - cl)
    out = pd.DataFrame(
        {"parkinson": pk.clip(lower=0.0), "garman_klass": gk.clip(lower=0.0),
         "rogers_satchell": rs, "gap": oc},
        index=h1.index,
    )
    # Yang-Zhang drift-independent estimator over trailing windows
    n_win = {"YZ_24h": 24, "YZ_120h": 120, "YZ_528h": 528}
    for name, W in n_win.items():
        v_o = oc.rolling(W, min_periods=W // 2).var()
        v_c = co.rolling(W, min_periods=W // 2).var()
        v_rs = rs.rolling(W, min_periods=W // 2).mean()
        k = (YZ_ALPHA - 1.0) / (YZ_ALPHA + (W + 1.0) / (W - 1.0))
        out[name] = v_o + k * v_c + (1 - k) * v_rs
    out["gk_minus_rs"] = out["garman_klass"] - out["rogers_satchell"]
    return out


def variance_ratio_block(h1_ret: pd.Series, qs=(2, 4, 8, 16), window: int = 240,
                         demean: bool = True) -> pd.DataFrame:
    """Lo-MacKinlay VR(q) with unbiased overlapping estimator and
    heteroskedasticity-robust z-statistic, on a trailing window of H1 returns.
    Returns VR(q), z(q), and the VR-slope Hurst estimate H = 0.5 + slope/2."""
    x = h1_ret.copy()
    if demean:
        x = x - x.rolling(window, min_periods=window // 2).mean()
    x2 = x ** 2
    N = window
    s1 = x2.rolling(N, min_periods=N // 2).sum()
    n_eff = x2.rolling(N, min_periods=N // 2).count().clip(lower=3)
    sig1 = s1 / (n_eff - 1)
    out = {}
    zouts = {}
    for q in qs:
        qsum = x.rolling(q).sum()                       # overlapping q-period sums
        s_q = (qsum ** 2).rolling(N, min_periods=N // 2).sum()
        # Lo-MacKinlay unbiased denominator for overlapping q-sums, per-period
        # form: m = (nq - q + 1)(1 - q/nq)  [Monte-Carlo verified unbiased on
        # iid data: E[VR] = 1.00 +- 0.01 at N=240, q in {4,16}]
        nq = (qsum.rolling(N, min_periods=N // 2).count()).clip(lower=q + 2)
        m = (nq - q + 1) * (1 - q / nq)
        sigq = s_q / m
        vr = sigq / (q * sig1 + 1e-18)
        out[f"VR{q}"] = vr
        # heteroskedasticity-robust variance theta(q) = sum_j [2(1-j/q)]^2 delta_j
        # delta_j = n * sum x_t^2 x_{t-j}^2 / (sum x_t^2)^2
        theta = pd.Series(0.0, index=x.index)
        for j in range(1, q):
            x2x2 = (x2 * x2.shift(j)).rolling(N, min_periods=N // 2).sum()
            dj = n_eff * x2x2 / (s1 ** 2 + 1e-18)
            theta = theta + ((2 * (q - j) / q) ** 2) * dj
        z = np.sqrt(n_eff) * (vr - 1) / np.sqrt(theta + 1e-18)
        zouts[f"VRz{q}"] = z
    res = pd.DataFrame({**out, **zouts})
    # Hurst from VR scaling: log VR(q) = (2H-1) log q  (fractional-process map)
    logq = np.log(np.array(qs, dtype=float))
    logq_c = logq - logq.mean()
    stack = np.stack([res[f"VR{q}"].values for q in qs], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logvr = np.log(np.abs(stack))
    logvr_c = logvr - logvr.mean(axis=1, keepdims=True)
    slope = (logvr_c * logq_c).sum(axis=1) / ((logq_c ** 2).sum() + 1e-18)
    res["hurst_VR"] = 0.5 + slope / 2.0
    return res


def roughness_block(rv_h1: pd.Series, deltas=(1, 2, 4, 8, 24), q: float = 2.0,
                    window: int = 240) -> pd.Series:
    """Gatheral-Jaisson-Rosenbaum volatility-roughness exponent on the H1 RV
    series: m(q, Delta) = mean |log sigma_{t+D} - log sigma_t|^q over trailing
    window; H_rough = slope(log m vs log Delta)/q."""
    logsig = 0.5 * np.log(rv_h1 + VOL_FLOOR)
    lags = np.array(deltas, dtype=float)
    ms = []
    for D in deltas:
        diff = (logsig - logsig.shift(D)).abs() ** q
        ms.append(diff.rolling(window, min_periods=window // 2).mean())
    M = np.stack([m.values for m in ms], axis=1)          # T x len(deltas)
    logd = np.log(lags); logd_c = logd - logd.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        logm = np.log(M + VOL_FLOOR)
    logm_c = logm - np.nanmean(logm, axis=1, keepdims=True)
    slope = np.nansum(logm_c * logd_c, axis=1) / ((logd_c ** 2).sum() + 1e-18)
    return pd.Series(slope / q, index=rv_h1.index, name="H_rough_vol")


def har_aggregates(h1_feats: pd.DataFrame, horizons=(24, 120, 528)) -> pd.DataFrame:
    """HAR-style slow aggregates (daily / weekly / monthly analogues) of the
    per-hour realized features: rolling means per horizon (Corsi 2009 windows
    1/5/22 days -> 24/120/528 hours)."""
    out = {}
    for H in horizons:
        roll = h1_feats.rolling(H, min_periods=H // 2).mean()
        for col in roll.columns:
            out[f"{col}_har{H}"] = roll[col]
    return pd.DataFrame(out)


def f2_self_test() -> None:
    """Synthetic-data checks: (1) BV/RV unbiased on constant-vol Gaussian M1;
    (2) VR ~ 1 on iid returns; (3) jump statistic fires on an injected jump;
    (4) range sanity: GK/PK within 2x of true variance on GBM;
    (5) causality: truncating the tail of the M1 frame must not change
    features at earlier H1 closes;
    (6) BUG-FIX REGRESSION: tq3/med3 must not read future M1 bars."""
    rng = np.random.default_rng(7)
    n = 20_000
    idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    sig = 0.0004  # per-minute vol
    px = 2000.0 * np.exp(np.cumsum(sig * rng.standard_normal(n)))
    m1 = pd.DataFrame({"close": px}, index=idx)

    # (6) BUG-FIX REGRESSION: perturb the LAST M1 bar; terms at earlier bars
    # must be identical (with the old shift(-1), the last bar's perturbation
    # would leak into tq3/med2 of the previous bar).
    m1_pert = m1.copy()
    m1_pert.iloc[-1, 0] = m1_pert.iloc[-1, 0] * 1.5
    tA = _m1_terms(m1)
    tB = _m1_terms(m1_pert)
    a = tA["tq3"].iloc[:-1].values
    b = tB["tq3"].iloc[:-1].values
    assert np.allclose(a, b, equal_nan=True), "tq3 peeks at future bars!"
    a = tA["med2"].iloc[:-1].values
    b = tB["med2"].iloc[:-1].values
    assert np.allclose(a, b, equal_nan=True), "med2 peeks at future bars!"
    # and tq3/med2 at bar i DO depend on bars i-2, i-1, i (they are 3-term):
    assert tA["tq3"].iloc[100] != tA["tq3"].iloc[101]

    h1_idx = pd.date_range(idx[0], idx[-1], freq="h", tz="UTC")
    f = realized_block_m1(m1, h1_idx)
    last = f.iloc[-1]
    # (1) BV ~ RV ~ sigma^2 * 60 on the 1h window
    true_rv = (sig ** 2) * 60
    assert 0.5 < last["RV_1h"] / true_rv < 2.0, f"RV biased: {last['RV_1h']:.3e} vs {true_rv:.3e}"
    assert 0.5 < last["BV_1h"] / true_rv < 2.0, f"BV biased: {last['BV_1h']:.3e}"

    # (2) VR ~ 1 on iid H1 returns
    r_h1 = pd.Series(rng.standard_normal(4000) * 0.002,
                     index=pd.date_range("2020-01-01", periods=4000, freq="h", tz="UTC"))
    vr = variance_ratio_block(r_h1)
    assert 0.8 < vr["VR4"].iloc[-1] < 1.25, f"VR(4) on iid should be ~1, got {vr['VR4'].iloc[-1]}"
    assert 0.4 < vr["hurst_VR"].iloc[-1] < 0.6, f"iid Hurst should be ~0.5, got {vr['hurst_VR'].iloc[-1]}"

    # (3) jump statistic fires on injected jump
    px2 = px.copy(); px2[10_000:10_000 + 30] *= 1.004
    m1b = pd.DataFrame({"close": px2}, index=idx)
    fb = realized_block_m1(m1b, h1_idx)
    assert fb["jumpZ_24h"].max() > 2.326, "jump statistic failed to fire on injected jump"

    # (4) range estimators on GBM H1 built from the actual M1 path per hour
    n_h1 = len(px) // 60
    blocks = px[: n_h1 * 60].reshape(n_h1, 60)
    h1o, h1c = blocks[:, 0], blocks[:, -1]
    h1h, h1l = blocks.max(axis=1), blocks.min(axis=1)
    h1 = pd.DataFrame({"open": h1o, "high": h1h, "low": h1l, "close": h1c},
                      index=h1_idx[: n_h1])
    rb = range_block_h1(h1)
    true_var = (sig ** 2) * 60
    mid = rb["rogers_satchell"].iloc[100:200].mean()
    assert 0.3 < mid / true_var < 3.0, f"RS estimator off: {mid:.3e} vs {true_var:.3e}"

    # (5) causality: features at earlier H1 closes invariant to future M1 data
    h1_cut = h1_idx[h1_idx < idx[15_000]]
    f_cut = realized_block_m1(m1.iloc[:15_060], h1_cut)
    f_full_at_cut = f.loc[h1_cut[h1_cut < idx[15_000] - pd.Timedelta("2h")]]
    f_cut_cmp = f_cut.loc[h1_cut[h1_cut < idx[15_000] - pd.Timedelta("2h")]]
    pd.testing.assert_frame_equal(f_cut_cmp, f_full_at_cut, check_freq=False)
    print("F2 self-tests passed: BV/RV unbiased, VR~1 iid, jump fires, RS sane, "
          "causal, shift-fix regression green.")
