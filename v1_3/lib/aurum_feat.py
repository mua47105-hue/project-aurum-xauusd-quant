# ============================================================================
# AURUM dimensionless feature pipeline (Spec v1.1, audit fix #1 + #6).
# EVERY feature is a function of scale-invariant transforms only:
#   log-return ratios, ATR-normalized distances, rolling z-scores (past-only),
#   range positions (0..1), codebook ids, calendar cyclical ids.
# NO absolute price, NO nominal ATR, NO absolute date, NO raw volume level.
# Cross-asset features (XAG/EUR/JPY/DXY-proxy/Y10/VIX) give the sequence model
# non-tabular structure that a 15-lag XGBoost cannot trivially match (audit #6).
# ============================================================================
import numpy as np
import pandas as pd

EPS = 1e-12

# ---------- primitives (all scale-invariant by construction) ----------------

def wilder_atr(h, l, c, n=14):
    prev_c = c.shift(1)
    tr = pd.Series(np.maximum.reduce([h - l, (h - prev_c).abs(), (l - prev_c).abs()]), index=h.index)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

def roll_z(x, n, minp=None):
    minp = max(4, n // 2) if minp is None else minp
    m = x.rolling(n, min_periods=minp).mean()
    s = x.rolling(n, min_periods=minp).std()
    return (x - m) / (s + EPS)

def logret(c, h=1):
    return np.log(c / c.shift(h))

def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / (dn + EPS))

def stoch_k(h, l, c, n=14):
    ll = l.rolling(n, min_periods=n).min()
    hh = h.rolling(n, min_periods=n).max()
    return 100 * (c - ll) / (hh - ll + EPS)

def macd_atr(c, atr, fast=12, slow=26, sig=9):
    ef = c.ewm(span=fast, adjust=False).mean()
    es = c.ewm(span=slow, adjust=False).mean()
    line = (ef - es)
    sgnl = line.ewm(span=sig, adjust=False).mean()
    return (line - sgnl) / (atr + EPS)

def session_of_hour(hour_utc):
    """Cyclical session id from UTC hour (calendar feature, not a date proxy)."""
    if 0 <= hour_utc < 7:
        return "ASIA"
    if 7 <= hour_utc < 12:
        return "LONDON"
    if 12 <= hour_utc < 16:
        return "NY"
    if 16 <= hour_utc < 21:
        return "OVL"
    return "OFF"

DOW = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

# ---------- main feature builder --------------------------------------------

def compute_h1_features(df_h1, m1_code=None, h1_code=None, cross=None, fred=None,
                        priors=None):
    """
    df_h1: DataFrame indexed by UTC hourly DatetimeIndex with columns o,h,l,c,v.
    m1_code / h1_code: pd.Series of codebook codes aligned to df_h1.index.
    cross: dict asset -> df_h1 (o,h,l,c) for XAGUSD/EURUSD/USDJPY (own clocks aligned to UTC).
    fred: dict series -> pd.Series of daily values indexed by date.
    priors: DataFrame with prior_w24/l24/w48/l48 aligned to df_h1.index.
    Returns DataFrame of dimensionless features (float), same index.
    """
    o, h, l, c, v = df_h1["o"], df_h1["h"], df_h1["l"], df_h1["c"], df_h1["v"]
    atr = wilder_atr(h, l, c, 14)
    F = pd.DataFrame(index=df_h1.index)
    # family 1: multi-horizon moves in ATR units (price distance / ATR ->
    # scale invariant, and aligned with the ATR-unit barrier geometry)
    for hh in (1, 4, 12, 24, 48):
        F[f"ret{hh}_atr"] = (c - c.shift(hh)) / (atr + EPS)
    # family 2/3: vol regime
    F["atr_ratio"] = atr / atr.shift(48)
    F["vol_z"] = roll_z(atr, 252)
    # family 4/5: oscillators (z-scored, past-only)
    F["rsi_z"] = roll_z(rsi(c, 14), 252)
    F["stoch_z"] = roll_z(stoch_k(h, l, c, 14), 252)
    # family 6: MACD in ATR units
    F["macd_atr"] = macd_atr(c, atr)
    # family 7/8: bollinger & donchian positions
    sma20 = c.rolling(20, min_periods=20).mean()
    std20 = c.rolling(20, min_periods=20).std()
    F["bb_pos"] = (c - sma20) / (2 * std20 + EPS)
    F["bb_w_z"] = roll_z(2 * std20 / (sma20 + EPS), 252)
    F["d20"] = (c - l.rolling(20, min_periods=20).min()) / (h.rolling(20, min_periods=20).max() - l.rolling(20, min_periods=20).min() + EPS)
    F["d55"] = (c - l.rolling(55, min_periods=55).min()) / (h.rolling(55, min_periods=55).max() - l.rolling(55, min_periods=55).min() + EPS)
    # family 9: candle anatomy, last 3 bars (ATR units)
    for k in (1, 2, 3):
        ok_, hk_, lk_, ck_ = o.shift(k), h.shift(k), l.shift(k), c.shift(k)
        F[f"body{k}"] = (ck_ - ok_) / (atr + EPS)
        F[f"uwick{k}"] = (hk_ - np.maximum(ok_, ck_)) / (atr + EPS)
        F[f"lwick{k}"] = (np.minimum(ok_, ck_) - lk_) / (atr + EPS)
    # family 10: directional run length (normalized, signed)
    sgn = np.sign(c.diff()).fillna(0.0)
    run = sgn.groupby((sgn != sgn.shift()).cumsum()).cumcount() + 1
    F["run"] = (run * sgn) / 10.0
    # family 11: volume z (volume level is not price-scale dependent)
    F["volm_z"] = roll_z(np.log1p(v.astype(float)), 252)
    # family 12/13: shape codebook codes
    if m1_code is not None:
        F["m1_shape"] = m1_code.reindex(df_h1.index)
    if h1_code is not None:
        F["h1_shape"] = h1_code.reindex(df_h1.index)
    # family 14: cross-asset relative moves in their own ATR units + macro z
    if cross:
        for a, dfa in cross.items():
            aatr = wilder_atr(dfa["h"], dfa["l"], dfa["c"], 14)
            rel = ((dfa["c"] - dfa["c"].shift(4)) / (aatr + EPS)).reindex(df_h1.index).ffill(limit=6)
            F[f"x_{a}"] = rel
    if fred:
        for k, s in fred.items():
            daily = pd.to_numeric(s, errors="coerce").sort_index()
            if daily.index.tz is None:
                daily.index = daily.index.tz_localize("UTC")
            # z-score, then shift(1): use YESTERDAY's value intraday (FRED daily
            # series are published during/after the US session - no lookahead)
            z = roll_z(daily, 252, minp=60).shift(1)
            # fill daily gaps first (weekends/holidays, up to 10 days), then map
            # every hour of a calendar date to that date's value
            z_ff = z.ffill(limit=10)
            F[f"m_{k}"] = z_ff.reindex(df_h1.index.normalize()).values
    # family 15: past-only cohort base rates (merged later if provided)
    if priors is not None:
        for col in ("prior_w24", "prior_l24", "prior_w48", "prior_l48"):
            if col in priors.columns:
                F[col] = priors[col].reindex(df_h1.index)
    return F

# Feature columns used for cohort matching (excludes priors to avoid mild
# self-reference; all dimensionless).
COHORT_FEATURES = [
    "ret1_atr", "ret4_atr", "ret12_atr", "ret24_atr", "ret48_atr",
    "atr_ratio", "vol_z", "rsi_z", "stoch_z", "macd_atr", "bb_pos", "bb_w_z",
    "d20", "d55", "body1", "uwick1", "lwick1", "run", "volm_z",
    "m1_shape", "h1_shape",
]
COHORT_FEATURES += ["x_xagusd", "x_eurusd", "x_usdjpy", "m_dxy", "m_y10", "m_vix"]

# ---------- synthetic market generator (for scale-invariance & smoke tests) --

def synthetic_ohlc(n_bars=3000, seed=7, p0=1837.42, m1_per_bar=60):
    """Regime-switching GBM with vol clustering; returns (h1_df, m1_df).
    p0 is arbitrary - features must be invariant to it (that is the test)."""
    rng = np.random.default_rng(seed)
    # H1 regime path
    vol = np.empty(n_bars); vol[0] = 0.10
    drift = np.zeros(n_bars)
    regime = 0
    for i in range(1, n_bars):
        if rng.random() < 0.01:
            regime = rng.integers(0, 3)
        vol[i] = np.clip(0.92 * vol[i - 1] + 0.08 * (0.05 + 0.12 * regime / 2) + rng.normal() * 0.008, 0.02, 0.9)
        drift[i] = (regime - 1) * 0.0004
    # M1 sub-paths (60 per H1 bar) - random walk bridging hourly closes
    tot = n_bars * m1_per_bar
    steps = rng.normal(drift.repeat(m1_per_bar) / m1_per_bar, (vol.repeat(m1_per_bar) / np.sqrt(m1_per_bar * 252 * 24)))
    m1_close = p0 * np.exp(np.cumsum(steps))
    idx_h1 = pd.date_range("2019-01-01", periods=n_bars, freq="1h", tz="UTC")
    idx_m1 = pd.date_range("2019-01-01", periods=tot, freq="1min", tz="UTC")
    # intrabar noise for o/h/l
    wick = np.abs(rng.normal(0, 0.35, tot)) * vol.repeat(m1_per_bar) / np.sqrt(252 * 24 * 60) * m1_close
    m1 = pd.DataFrame({
        "o": np.roll(m1_close, 1) * (1 + rng.normal(0, 1e-4, tot)),
        "h": m1_close + wick * rng.random(tot),
        "l": m1_close - wick * rng.random(tot),
        "c": m1_close,
        "v": rng.integers(10, 500, tot),
    }, index=idx_m1)
    m1.iloc[0, m1.columns.get_loc("o")] = p0
    m1["h"] = m1[["h", "o", "c"]].max(axis=1)
    m1["l"] = m1[["l", "o", "c"]].min(axis=1)
    h1 = m1.resample("1h").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum")).dropna()
    return h1, m1

def scale_ohlc(df, factor, cols=("o", "h", "l", "c")):
    out = df.copy()
    for col in cols:
        out[col] = out[col] * factor
    return out
