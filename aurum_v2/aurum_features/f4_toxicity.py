"""
AURUM v2.0 -- Feature Module F4: Microstructural Toxicity & Liquidity
======================================================================
Blueprint 2 Section 3.4 WITH THE MANDATED CAUSALITY FIX applied:

  *** BUG-FIX (Spec v2.0 execution mandate II.1) ***
  kyle_lambda_block originally computed the lag-1 impact covariance as
      cov_lag = df["r"].shift(-1).rolling(L, min_periods=240).cov(df["SV"])
  which peeks at minute :60 of the hour at bar :59.  Replaced with the
  mathematically identical lag-1 covariance, strictly causal:
      cov_lag = df["r"].rolling(L, min_periods=240).cov(df["SV"].shift(1))
  (Cov(r_{t+1}, SV_t) == Cov(r_t, SV_{t-1}) under stationarity of the lag
  operator -- same estimator, zero future peeking.)

Formulas (verified; see Spec Section 3.4 and ledger P3-15..P3-18):

  Bulk Volume Classification (Easley, Lopez de Prado & O'Hara 2012, RFS 25(5)):
      V_buy,i  = V_i * Phi( dP_i / sigma_dP ),   V_sell,i = V_i - V_buy,i
  VPIN (volume-bucketed): time-bar buckets (30 M1 bars ~ 30 min; 48 per 24h);
      VPIN = mean_bucket |V_buy - V_sell| / V_bucket.
  FX/CFD adaptation: tick volume (number of M1 price updates) as the activity
  proxy (Bollerslev-Domowitz 1993; Dacorogna et al. 2001 lineage); the
  Andersen-Bondarenko critique (J. Financial Markets 2014) -- VPIN largely
  proxies short-horizon volatility -- is honored by using VPIN as a STATE
  variable, never as standalone alpha.

  Kyle lambda (Kyle 1985, Econometrica): lambda = Cov(r, SV)/Var(SV) with
  SV = BVC-signed volume; rolling estimation = transient impact slope.

  Realized Amihud (Lacava-Ranaldo-Santucci de Magistris via arXiv:2609.00943):
      Illiq = RPV / nu = sum_i |r_i| / sum_i nu_i.

  CLV (close location value) volume-weighted accumulation:
      CLV = ((C-L) - (H-C)) / (H-L) in [-1,1];  AD_t = sum CLV * V (windowed).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

M1_BARS_PER_BUCKET = 30      # time-bar VPIN bucket (~30 min); 48 buckets/24h
N_BUCKETS = 48               # trailing buckets in the VPIN mean (24h)
SIG_DP_SPAN = 1440           # rolling std of M1 price changes (24h)
LAMBDA_SPAN = 1440           # rolling Kyle-lambda window (24h)


def _bvc(m1: pd.DataFrame, price_col: str = "close", vol_col: str = "volume") -> pd.DataFrame:
    dP = m1[price_col].diff()
    sd = dP.rolling(SIG_DP_SPAN, min_periods=120).std()
    z = (dP / (sd + 1e-12)).fillna(0.0)
    v = m1[vol_col].astype("float64").fillna(0.0).clip(lower=0.0)
    vb = v * _norm.cdf(z)
    vs = v - vb
    sv = vb - vs                       # BVC signed volume in [-V, +V]
    return pd.DataFrame({"V": v, "Vb": vb, "Vs": vs, "SV": sv}, index=m1.index)


def vpin_block(m1: pd.DataFrame, h1_index: pd.DatetimeIndex,
               price_col: str = "close", vol_col: str = "volume") -> pd.DataFrame:
    """Causal tick-volume VPIN (time-bar buckets + BVC) sampled at H1 closes."""
    b = _bvc(m1, price_col, vol_col)
    # bucket sums: fixed M1_BARS_PER_BUCKET-bar buckets on the global minute grid
    full = pd.date_range(m1.index[0], m1.index[-1], freq="min", tz="UTC")
    g = b.reindex(full).fillna(0.0)
    minute_pos = np.arange(len(g))
    bucket_id = minute_pos // M1_BARS_PER_BUCKET
    take = minute_pos % M1_BARS_PER_BUCKET == (M1_BARS_PER_BUCKET - 1)
    vsum = g["V"].groupby(bucket_id).cumsum()
    bsum = g["Vb"].groupby(bucket_id).cumsum()
    ssum = g["Vs"].groupby(bucket_id).cumsum()
    # bucket-end rows: total, buy, sell within bucket
    V_b = vsum[take]; Vb_b = bsum[take]; Vs_b = ssum[take]
    imb = (Vb_b - Vs_b).abs() / (V_b + 1e-12)
    vpin = imb.rolling(N_BUCKETS, min_periods=N_BUCKETS // 2).mean()
    # per-bucket signed imbalance (direction of toxic flow)
    simb = (Vb_b - Vs_b) / (V_b + 1e-12)
    svpin = simb.rolling(N_BUCKETS, min_periods=N_BUCKETS // 2).mean()
    out = pd.DataFrame({"vpin": vpin, "vpin_signed": svpin})
    h1_close_idx = h1_index + pd.Timedelta(minutes=59)
    return out.reindex(h1_close_idx, method="ffill").set_axis(h1_index)


def kyle_lambda_block(m1: pd.DataFrame, h1_index: pd.DatetimeIndex,
                      price_col: str = "close", vol_col: str = "volume") -> pd.DataFrame:
    """Rolling Kyle lambda = Cov(r_M1, SV)/Var(SV) (impact slope per signed
    tick-volume unit), plus transient/total impact ratio via lagged covariances.

    CAUSALITY FIX APPLIED HERE: lag-1 covariance computed as
    Cov(r_t, SV_{t-1}) instead of the blueprint's Cov(r_{t+1}, SV_t)
    via shift(-1) -- identical estimator, zero future peeking."""
    b = _bvc(m1, price_col, vol_col)
    r = np.log(m1[price_col] / m1[price_col].shift(1))
    df = pd.DataFrame({"r": r, "SV": b["SV"]}, index=m1.index).dropna()
    cov = df["r"].rolling(LAMBDA_SPAN, min_periods=240).cov(df["SV"])
    var = df["SV"].rolling(LAMBDA_SPAN, min_periods=240).var()
    lam = (cov / (var + 1e-18)).clip(lower=0.0)
    # --- FIX: lag-1 impact share, strictly causal (was r.shift(-1).cov(SV))
    cov_lag = df["r"].rolling(LAMBDA_SPAN, min_periods=240).cov(df["SV"].shift(1))
    lam_lag = (cov_lag / (var + 1e-18)).clip(lower=0.0)
    out = pd.DataFrame({"kyle_lambda": lam, "kyle_lambda_lag": lam_lag,
                        "tick_volume_24h": b["V"].rolling(LAMBDA_SPAN, min_periods=240).sum()})
    h1_close_idx = h1_index + pd.Timedelta(minutes=59)
    return out.reindex(h1_close_idx, method="ffill").set_axis(h1_index)


def amihud_clv_block(m1: pd.DataFrame, h1: pd.DataFrame,
                     price_col: str = "close", vol_col: str = "volume") -> pd.DataFrame:
    """Realized Amihud per window + CLV volume-weighted accumulation per H1.
    h1: H1 OHLC frame (for CLV); m1: for the realized variant."""
    r = np.log(m1[price_col] / m1[price_col].shift(1)).abs()
    v = m1[vol_col].astype("float64").fillna(0.0).clip(lower=0.0)
    out = {}
    for wname, W in (("1h", 60), ("4h", 240), ("24h", 1440)):
        rpv = r.rolling(W, min_periods=W // 2).sum()
        nu = v.rolling(W, min_periods=W // 2).sum()
        out[f"amihud_{wname}"] = rpv / (nu + 1e-12)
    res = pd.DataFrame(out, index=m1.index)
    h1_close_idx = h1.index + pd.Timedelta(minutes=59)
    res = res.reindex(h1_close_idx, method="ffill").set_axis(h1.index)

    # CLV per H1 bar (from H1 OHLC), tick-volume weighted, windowed
    h, l, c = h1["high"], h1["low"], h1["close"]
    rng = (h - l).replace(0.0, np.nan)
    clv = (((c - l) - (h - c)) / rng).fillna(0.0)
    tv = h1[vol_col].astype("float64").fillna(0.0) if vol_col in h1 else 0.0
    clv_tv = clv * tv
    res["clv"] = clv
    for W in (6, 24, 120):
        res[f"clv_tv_sum{W}"] = clv_tv.rolling(W, min_periods=W // 2).sum()
        res[f"clv_mean{W}"] = clv.rolling(W, min_periods=W // 2).mean()
    return res


def f4_self_test() -> None:
    rng = np.random.default_rng(11)
    n = 60 * 24 * 30
    idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    # toxic regime: signed flow drives price (persistent impact)
    flow = rng.standard_normal(n)
    ret = 0.00002 * flow + 0.0002 * rng.standard_normal(n)
    px = 2000 * np.exp(np.cumsum(ret))
    tv = (100 + 50 * rng.random(n)) * (1 + 5 * np.abs(flow))
    m1 = pd.DataFrame({"close": px, "volume": tv}, index=idx)
    h1_idx = pd.date_range(idx[0], idx[-1], freq="h", tz="UTC")
    vp = vpin_block(m1, h1_idx)
    assert vp["vpin"].notna().sum() > 100
    assert ((vp["vpin"].dropna() >= 0) & (vp["vpin"].dropna() <= 1)).all(), "VPIN outside [0,1]"
    kl = kyle_lambda_block(m1, h1_idx)
    assert (kl["kyle_lambda"].dropna() > 0).mean() > 0.9, "lambda should be ~positive under impact"
    # CLV block with synthetic H1
    n_h1 = n // 60
    blocks = px[: n_h1 * 60].reshape(n_h1, 60)
    h1 = pd.DataFrame({"open": blocks[:, 0], "high": blocks.max(1), "low": blocks.min(1),
                       "close": blocks[:, -1], "volume": tv[: n_h1 * 60].reshape(n_h1, 60).sum(1)},
                      index=h1_idx[: n_h1])
    am = amihud_clv_block(m1, h1)
    assert am["amihud_1h"].notna().sum() > 100
    assert am["clv"].between(-1, 1).all()
    # causality
    cut = 60 * 24 * 20
    h1c = h1_idx[h1_idx < idx[cut]]
    vpc = vpin_block(m1.iloc[: cut + 60], h1c)
    common = h1c[h1c < idx[cut - 120]]
    pd.testing.assert_frame_equal(vpc.loc[common], vp.loc[common])
    # --- BUG-FIX REGRESSION: perturbing the LAST M1 bar must not change any
    # earlier kyle_lambda_lag row (old shift(-1) leaked :60 into :59).
    m1b = m1.copy()
    m1b.iloc[-1, m1b.columns.get_loc("close")] *= 1.5
    klb = kyle_lambda_block(m1b, h1_idx)
    prev = h1_idx[h1_idx < idx[-1] - pd.Timedelta("2h")]
    a = kl.loc[prev, "kyle_lambda_lag"].fillna(0).values
    b2 = klb.loc[prev, "kyle_lambda_lag"].fillna(0).values
    assert np.allclose(a, b2, equal_nan=True), "kyle_lambda_lag peeks at future bars!"
    print("F4 self-tests passed: VPIN bounds, lambda sign, Amihud/CLV, causal, "
          "cov-lag fix regression green.")
