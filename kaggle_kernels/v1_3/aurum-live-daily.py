# ============================================================================
# Auto-generated AURUM kernel: live_daily (Step 6a) = p3 lib + live tail
# ============================================================================
# ============================================================================
# AURUM kernel: lib/* (shared with p3_tables) — sliced for live_daily
# Spec v1.1 frozen. Do not edit on Kaggle - edit sources and rebuild.
# ============================================================================
import os as _os, sys as _sys
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _pin_transformers():
    """Qwen3 needs transformers>=4.51; 5.x changed model APIs and the Kaggle
    image ships a partially broken 5.20.2 (keras_nlp backend import error).
    Force-install 4.57.1 with --no-deps (bypasses the resolver refusing the
    downgrade), verify, and purge cached modules."""
    import subprocess as _sp
    ok = False
    try:
        import transformers
        from packaging.version import Version
        ok = Version("4.51.0") <= Version(transformers.__version__) < Version("5.0.0")
    except Exception:
        ok = False
    if ok:
        return
    try:
        _sp.run([_sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
                 "-q", "transformers==4.57.1"], check=True, timeout=900,
                capture_output=True)
        _sp.run([_sys.executable, "-m", "pip", "install", "--no-deps", "-q",
                 "tokenizers==0.22.0", "huggingface-hub==0.36.2"],
                check=False, timeout=900, capture_output=True)
    except Exception as _e:
        print("transformers pin install error:", _e, flush=True)
    for _mod in list(_sys.modules):
        if _mod.startswith(("transformers", "tokenizers", "huggingface_hub")):
            del _sys.modules[_mod]
    import transformers
    from packaging.version import Version as _V
    _v = Version(transformers.__version__)
    assert _V("4.51.0") <= _v < _V("5.0.0"), (
        f"transformers pin FAILED, got {_v}; Qwen3 layer API unverified on this version")

_pin_transformers()


# ==================== lib/aurum_gates.py ====================
# ============================================================================
# AURUM shared helpers: gates, logging, env report
# Spec v1.1 (frozen). All quantity-producing code paths must be dimensionless
# (ATR units / log-returns / z-scores / ratios / codebook ids) - see audit #1.
# ============================================================================
import json, os, sys, time, traceback, random
import numpy as np

GATES = {}
LOG_LINES = []

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)

def gate(gid, ok, detail=None, extra=None):
    g = {"gate": gid, "pass": bool(ok), "detail": str(detail)[:2000]}
    if extra:
        g.update(extra)
    GATES[gid] = g
    log(f"GATE {gid}: {'PASS' if ok else 'FAIL'} | {detail}")
    return ok

def save_gates(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(GATES, f, indent=2, default=str)
    with open(os.path.join(os.path.dirname(path), "run_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES))

def set_seeds(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def env_report():
    import platform
    info = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cwd": os.getcwd(),
        "argv": sys.argv,
    }
    try:
        import pandas as pd
        info["pandas"] = pd.__version__
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["capability"] = str(torch.cuda.get_device_capability(0))
            p = torch.cuda.get_device_properties(0)
            info["total_mem_gb"] = round(p.total_memory / 1e9, 2)
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception as e:
        info["torch_error"] = str(e)
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = "missing"
    return info

def is_kaggle():
    return os.path.exists("/kaggle/working")

def stage(name):
    """Decorator-ish context: log stage start/end, capture exceptions."""
    class C:
        def __init__(self):
            self.t0 = time.time()
            self.ok = None
            self.err = None
        def __enter__(self):
            log(f"===== STAGE {name} START =====")
            return self
        def __exit__(self, et, ev, tb):
            self.ok = et is None
            if et is not None:
                self.err = "".join(traceback.format_exception(et, ev, tb))[-3000:]
                log(f"===== STAGE {name} ERROR =====\n{self.err}")
            else:
                log(f"===== STAGE {name} DONE in {time.time()-self.t0:.1f}s =====")
            return False  # do not suppress
    return C()

# Frozen architecture constants (Spec v1.1) ---------------------------------
BARRIER_STOPS = [1.0, 1.5, 2.0, 2.5]      # x ATR14(H1)
BARRIER_TPS   = [1.5, 2.5, 4.0]           # x ATR14(H1)
BARRIER_HORS  = [24, 48]                  # H1 bars; default vertical barrier H=48
N_CELLS       = 2 * 4 * 3 * 2             # = 48 cells (dir x stop x tp x horizon)
KELLY_FRACTION = 0.25                     # quarter Kelly
HARD_EQUITY_RISK_CAP = 0.01               # 1.0% equity risk per trade
COHORT_K = 200                            # B=200 bootstrap cohorts
COHORT_GAP = 96                           # neighbour must end >=96 bars before target bar
READOUT_NAMES = ["H1_BARRIER", "H2_MTM", "H3_EXCUR", "H4_VOL", "H5_REGIME"]
HEAD_DIMS = {"h1": 48, "h2": 8, "h3": 12, "h4": 8, "h5": 9}  # 85 outputs total

def cell_axes():
    """Flat 48-cell layout: for d in (long,short) for m in stops for t in tps for h in hors."""
    dirs, mss, tps, hrs = [], [], [], []
    for d in ("L", "S"):
        for m in BARRIER_STOPS:
            for t in BARRIER_TPS:
                for h in BARRIER_HORS:
                    dirs.append(d); mss.append(m); tps.append(t); hrs.append(h)
    return dirs, mss, tps, hrs

def cell_names():
    dirs, mss, tps, hrs = cell_axes()
    return [f"{d}_m{m}_t{t}_h{h}" for d, m, t, h in zip(dirs, mss, tps, hrs)]


# ==================== lib/aurum_feat.py ====================
# ============================================================================
# AURUM dimensionless feature pipeline (Spec v1.2).
# EVERY feature is a function of scale-invariant transforms only:
#   log-return ratios, ATR-normalized distances, rolling z-scores (past-only),
#   range positions (0..1), codebook ids, calendar cyclical ids.
# NO absolute price, NO nominal ATR, NO absolute date, NO raw volume level.
# Cross-asset features (XAG/EUR/JPY + macro INNOVATIONS) give the sequence
# model non-tabular structure a 15-lag XGBoost cannot trivially match.
#
# v1.2 changes (Failure #1/#2 remediation):
#   - Macro panel switched from raw-level z-scores (date fingerprints,
#     G1.4 AUC 0.96-1.00) to INNOVATION transforms on raw FRED levels:
#     d20z / d5z (z-score of the daily innovation over a trailing window)
#     and rank60 (percentile rank of the level in a trailing 60-day window),
#     all with a strict 1-day publication lag (exp3c: |AUC-0.5| <= 0.03).
#   - The 4 base-rate priors enter the STATE as self-z-scores (era-safe
#     encoding; raw trailing win rates were date fingerprints at AUC 0.85-0.90).
#     Raw values stay in the parquet for the decision layer / baselines.
#   - Cohort MATCHING space is era-sanitized: raw m_dxy/m_y10 replaced by
#     their innovation forms (matched neighbors must not be era-concentrated).
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

# ---------- macro innovation panel (Spec v1.2) --------------------------------

def macro_innovations(fred, h1_index):
    """Exact daily-domain innovation transforms on RAW FRED levels.

    For each series s in fred (y10, y2, spread, dxy, vix), computed on the
    DAILY series then mapped onto H1 bars with a STRICT 1-day lag
    (yesterday's fully-published value; FRED dailies publish during/after
      the US session - no lookahead):
        mi_{s}_d20z   = roll_z(level.diff(), 20)   - 20-day z of daily innovation
        mi_{s}_d5z    = roll_z(level.diff(), 5)    - 5-day z of daily innovation
        mi_{s}_rank60 = level.rolling(60).rank(pct=True) - 60-day percentile rank
    Weekend/holiday gaps are forward-filled up to 10 days.
    Returns DataFrame aligned to h1_index (era-invariant by construction:
    exp3c measured |AUC-0.5| <= 0.03 for the d20z/d5z family on 2012-2025)."""
    F = pd.DataFrame(index=h1_index)
    for k, s in fred.items():
        daily = pd.to_numeric(s, errors="coerce").sort_index()
        if daily.index.tz is None:
            daily.index = daily.index.tz_localize("UTC")
        innov = daily.diff()
        d20z = roll_z(innov, 20, minp=10)
        d5z = roll_z(innov, 5, minp=3)
        rank60 = daily.rolling(60, min_periods=30).rank(pct=True)
        for nm, ser in ((f"mi_{k}_d20z", d20z), (f"mi_{k}_d5z", d5z),
                        (f"mi_{k}_rank60", rank60)):
            z = ser.shift(1)                       # strict 1-day lag
            F[nm] = z.ffill(limit=10).reindex(
                h1_index.normalize()).values
    return F

# ---------- self-z-scored priors (era-safe state encoding) -------------------

PRIOR_COLS = ("prior_w24", "prior_l24", "prior_w48", "prior_l48")

def zscore_priors(priors, n=504):
    """Z-score each trailing base-rate prior against its own trailing n-bar
    distribution (past-only). The raw trailing win rate is a date fingerprint
    (G1.4 AUC 0.85-0.90 in v4); its self-z ('is the current base rate unusually
    high vs recent history?') is era-invariant by the same construction as every
    other roll_z feature. Returns DataFrame with prior_zw24/... columns."""
    out = pd.DataFrame(index=priors.index)
    for c in PRIOR_COLS:
        if c in priors.columns:
            zname = c.replace("prior_", "prior_z")
            out[zname] = roll_z(priors[c], n)
    return out

def compute_h1_features(df_h1, m1_code=None, h1_code=None, cross=None, fred=None,
                        priors=None, innov=None):
    """
    df_h1: DataFrame indexed by UTC hourly DatetimeIndex with columns o,h,l,c,v.
    m1_code / h1_code: pd.Series of codebook codes aligned to df_h1.index.
    cross: dict asset -> df_h1 (o,h,l,c) for XAGUSD/EURUSD/USDJPY (own clocks aligned to UTC).
    fred: dict series -> pd.Series of daily values indexed by date.
    priors: DataFrame with prior_w24/l24/w48/l48 aligned to df_h1.index (RAW
            values; they are kept for the parquet/decision layer - the STATE
            uses their z-scores from zscore_priors).
    innov: DataFrame from macro_innovations() aligned to df_h1.index
            (mi_{s}_d20z/d5z/rank60 columns; if None and fred given, computed).
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
    # family 14b: macro INNOVATION panel (Spec v1.2: replaces raw macro levels
    # in the state; m_vix is kept - it passed G1.4-a at 0.456 in the v4 audit)
    if innov is None and fred:
        innov = macro_innovations(fred, df_h1.index)
    if innov is not None:
        for c in innov.columns:
            F[c] = innov[c]
    # family 15: past-only cohort base rates (merged later if provided)
    if priors is not None:
        for col in ("prior_w24", "prior_l24", "prior_w48", "prior_l48"):
            if col in priors.columns:
                F[col] = priors[col].reindex(df_h1.index)
    return F

# Feature columns used for cohort MATCHING (Spec v1.2: era-sanitized -
# raw m_dxy/m_y10 replaced by innovation forms so matched neighbors are not
# era-concentrated; m_vix kept, it passed G1.4-a). Excludes priors to avoid
# mild self-reference; all dimensionless.
COHORT_FEATURES = [
    "ret1_atr", "ret4_atr", "ret12_atr", "ret24_atr", "ret48_atr",
    "atr_ratio", "vol_z", "rsi_z", "stoch_z", "macd_atr", "bb_pos", "bb_w_z",
    "d20", "d55", "body1", "uwick1", "lwick1", "run", "volm_z",
    "m1_shape", "h1_shape",
]
COHORT_FEATURES += ["x_xagusd", "x_eurusd", "x_usdjpy", "m_vix",
                    "mi_dxy_d20z", "mi_y10_d20z"]

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


# ==================== lib/aurum_label.py ====================
# ============================================================================
# AURUM 48-cell barrier labeler + B=200 matched-cohort bootstrap (Spec v1.1).
#
# Audit fix #2 (counterfactual validity): NO synthetic price paths are ever
# fabricated. Soft labels are the realized outcome distribution over K=200
# regime-matched HISTORICAL episodes (matched on dimensionless state features,
# strictly from the past) - a mathematically valid empirical bootstrap.
# Audit fix #4 (single-path noise): p_win / b_eff / excursion quantiles are
# distributional targets, not one noisy path realization.
# Audit fix #3: one standardized barrier matrix, 48 cells, evaluated for both
# directions in a single pass; the joint head learns all cells at once.
# All outcomes are in ATR units -> scale invariant (audit fix #1).
# ============================================================================
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

EPS = 1e-12

DIRS = ("L", "S")

# ---------------------------------------------------------------------------
# 1) Actual-path deterministic outcomes per (bar, cell) - M1 execution
# ---------------------------------------------------------------------------

def barrier_actual_outcomes(m1, h1, i_states, chunk=512):
    """
    For each H1 state bar i (state = close of bar i), a hypothetical entry is
    taken at the OPEN of bar i+1. Outcomes are resolved on M1 data.

    Returns dict of arrays indexed by position in i_states:
      win    (N, 48) bool  : realized R > 0 under simple stack (TP-first=win,
                             SL-first=loss, timeout=win iff residual>0)
      R      (N, 48) float : realized P&L in R-multiples (R = m*ATR)
      cls    (N, 48) int8  : 0=SL first, 1=TP first, 2=timeout
      mfe    (N, 2, 2)     : max favorable excursion /ATR  [dir, horizon]
      mae    (N, 2, 2)     : max adverse  excursion /ATR  [dir, horizon]
      res    (N, 2, 2)     : timeout residual P&L /ATR     [dir, horizon]
    Entry uses next-bar open; ties inside the same minute -> SL first
    (conservative convention, applied uniformly).
    """
    m1 = m1.sort_index()
    m1_ts = np.asarray(m1.index.asi8, dtype=np.int64)  # ns epoch (UTC)
    m1_lo = m1["l"].values.astype(np.float64)
    m1_hi = m1["h"].values.astype(np.float64)
    m1_cl = m1["c"].values.astype(np.float64)

    h1 = h1.sort_index()
    h1_ts = np.asarray(h1.index.asi8, dtype=np.int64)
    h1_open = h1["o"].values.astype(np.float64)
    h1_hi = h1["h"].values.astype(np.float64)
    h1_lo = h1["l"].values.astype(np.float64)
    h1_cl = h1["c"].values.astype(np.float64)

    # ATR14 (Wilder) on H1, computed with past info only
    prev_c = np.roll(h1_cl, 1); prev_c[0] = h1_cl[0]
    tr = np.maximum(h1_hi - h1_lo, np.maximum(np.abs(h1_hi - prev_c), np.abs(h1_lo - prev_c)))
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().values

    n = len(i_states)
    n_m, n_t, n_h = len(BARRIER_STOPS), len(BARRIER_TPS), len(BARRIER_HORS)
    ns = BARRIER_STOPS; ts_ = BARRIER_TPS; hs = [h * 60 for h in BARRIER_HORS]  # minutes

    win = np.zeros((n, N_CELLS), dtype=bool)
    R = np.full((n, N_CELLS), np.nan)
    cls = np.full((n, N_CELLS), 255, dtype=np.int16)
    mfe = np.full((n, 2, 2), np.nan)
    mae = np.full((n, 2, 2), np.nan)
    res = np.full((n, 2, 2), np.nan)

    H1_NS = 3600 * 1_000_000_000
    MAX_WIN_NS = 48 * H1_NS

    for c0 in range(0, n, chunk):
        sl = slice(c0, min(c0 + chunk, n))
        rows = np.asarray(i_states[sl])
        # entry bar = state bar + 1; needs a next bar to exist
        valid = (rows + 1 < len(h1_ts)) & np.isfinite(atr[rows]) & (atr[rows] > 0)
        for ri, i in enumerate(rows):
            if not valid[ri]:
                continue
            e = h1_open[i + 1]
            a = atr[i]
            t0 = h1_ts[i + 1]              # entry time = open of next bar
            s = np.searchsorted(m1_ts, t0, side="left")
            e_idx = min(len(m1_ts), np.searchsorted(m1_ts, t0 + MAX_WIN_NS, side="left"))
            if e_idx - s < 30:             # not enough forward M1 data
                continue
            hi_w = m1_hi[s:e_idx]; lo_w = m1_lo[s:e_idx]; cl_w = m1_cl[s:e_idx]
            # first-touch helper: first index where condition holds, else BIG
            def first_touch(cond):
                idx = np.flatnonzero(cond)
                return idx[0] if len(idx) else 1 << 30
            lims = []
            for hmin in hs:
                lims.append(min(len(hi_w), max(1, int(np.searchsorted(m1_ts[s:e_idx], t0 + hmin * 60_000_000_000, side="left")))))
            rbase = 0
            for di, d in enumerate(DIRS):
                # long: TP above (hi >= e+t*a), SL below (lo <= e-m*a)
                if d == "L":
                    tps_hit = np.array([first_touch(hi_w >= e + t * a) for t in ts_])
                    sls_hit = np.array([first_touch(lo_w <= e - m * a) for m in ns])
                else:
                    tps_hit = np.array([first_touch(lo_w <= e - t * a) for t in ts_])
                    sls_hit = np.array([first_touch(hi_w >= e + m * a) for m in ns])
                for hi_, lim in enumerate(lims):
                    if d == "L":
                        mfe_v = (hi_w[:lim].max() - e) / a
                        mae_v = (e - lo_w[:lim].min()) / a
                        res_v = (cl_w[lim - 1] - e) / a
                    else:
                        mfe_v = (e - lo_w[:lim].min()) / a
                        mae_v = (hi_w[:lim].max() - e) / a
                        res_v = (e - cl_w[lim - 1]) / a
                    mfe[c0 + ri, di, hi_] = mfe_v
                    mae[c0 + ri, di, hi_] = mae_v
                    res[c0 + ri, di, hi_] = res_v
                for mi, m in enumerate(ns):
                    for ti, t in enumerate(ts_):
                        for hi_, lim in enumerate(lims):
                            tsl = sls_hit[mi]; ttp = tps_hit[ti]
                            cell = rbase
                            if tsl <= ttp and tsl < lim:      # SL first (ties->SL)
                                R[c0 + ri, cell] = -1.0
                                cls[c0 + ri, cell] = 0
                            elif ttp < lim:
                                R[c0 + ri, cell] = t / m
                                cls[c0 + ri, cell] = 1
                            else:
                                rv = res[c0 + ri, di, hi_]
                                R[c0 + ri, cell] = rv * a / (m * a) if np.isfinite(rv) else np.nan
                                cls[c0 + ri, cell] = 2
                            win[c0 + ri, cell] = R[c0 + ri, cell] > 0
                            rbase += 1
    return {"win": win, "R": R, "cls": cls, "mfe": mfe, "mae": mae, "res": res}

def expand_actual(actual, i_states, N):
    """Expand labeler outputs (indexed by i_states position) to full-length
    (N, ...) arrays so cohort gathering by original bar index works.
    win is float (1.0/0.0) with NaN outside i_states so rolling priors and
    cohort means ignore uncomputed bars."""
    out = {}
    w = np.full((N, N_CELLS), np.nan)
    w[i_states] = actual["win"].astype(np.float64)
    out["win"] = w
    full = np.full((N, N_CELLS), np.nan)
    full[i_states] = actual["R"]
    out["R"] = full
    for k in ("mfe", "mae", "res"):
        full = np.full((N, 2, 2), np.nan)
        full[i_states] = actual[k]
        out[k] = full
    out["cls"] = np.full((N, N_CELLS), 255, dtype=np.int16)
    out["cls"][i_states] = actual["cls"]
    return out

# ---------------------------------------------------------------------------
# 2) Matched-cohort index (past-only, leakage-safe)
# ---------------------------------------------------------------------------

def build_cohort_index(feat_std, label_valid, k=COHORT_K, gap=COHORT_GAP,
                       query_k=900, chunk=4000):
    """
    feat_std   : (N, F) standardized features; rows with label_valid=False may
                 contain NaN (they are never used as neighbours or queries).
    label_valid: (N,) bool - bar has valid features AND valid actual outcomes.
    Neighbour j is eligible for target i iff j <= i - gap (strictly in the
    past, forward windows cannot overlap) and label_valid[j].
    Returns int32 (N, k) cohort indices (-1 = missing) and per-bar counts.
    """
    N = feat_std.shape[0]
    valid_rows = np.flatnonzero(label_valid)
    nn = NearestNeighbors(n_neighbors=min(query_k, len(valid_rows)),
                          algorithm="kd_tree", metric="euclidean", n_jobs=-1)
    nn.fit(feat_std[valid_rows])
    cohort = np.full((N, k), -1, dtype=np.int32)
    nvalid = np.zeros(N, dtype=np.int32)
    for s in range(0, len(valid_rows), chunk):
        rows = valid_rows[s: s + chunk]
        _, idx_local = nn.kneighbors(feat_std[rows])
        for r in range(len(rows)):
            i = rows[r]
            cand = valid_rows[idx_local[r]]
            cand = cand[cand <= i - gap][:k]
            nvalid[i] = len(cand)
            if len(cand):
                cohort[i, : len(cand)] = cand
    return cohort, nvalid

# ---------------------------------------------------------------------------
# 3) Cohort bootstrap soft labels (B=200)
# ---------------------------------------------------------------------------

def cohort_soft_labels(cohort, nvalid, actual, k_min=120, chunk=4000):
    """
    From actual-path outcomes of matched episodes:
      p_win (N,48): mean win flag            (H1 training target)
      b_eff (N,48): mean win size / |mean loss size| in R units (Kelly b)
      ev_ref(N,48): mean realized R          (reference EV)
      q_mfe (N,2,2,3), q_mae (N,2,2,3): cohort q25/q50/q75 of excursions
                                          (H3 target; drives deterministic exits)
    Bars with < k_min valid neighbours -> NaN (excluded from training).
    """
    N = cohort.shape[0]
    win, R = actual["win"], actual["R"]
    mfe, mae = actual["mfe"], actual["mae"]
    p_win = np.full((N, N_CELLS), np.nan)
    b_eff = np.full((N, N_CELLS), np.nan)
    ev_ref = np.full((N, N_CELLS), np.nan)
    q_mfe = np.full((N, 2, 2, 3), np.nan)
    q_mae = np.full((N, 2, 2, 3), np.nan)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        for i in range(s, e):
            m = nvalid[i]
            if m < k_min:
                continue
            ids = cohort[i, :m]
            w = win[ids].astype(np.float64)      # (m,48)
            r = R[ids]                            # (m,48)
            p_win[i] = w.mean(0)
            with np.errstate(invalid="ignore"):
                pos = np.where(r > 0, r, np.nan)
                neg = np.where(r < 0, r, np.nan)
                mp = np.nanmean(pos, 0); mn = np.nanmean(neg, 0)
                b_eff[i] = mp / np.abs(mn)
                ev_ref[i] = np.nanmean(r, 0)
            for d in range(2):
                for hh in range(2):
                    q_mfe[i, d, hh] = np.nanquantile(mfe[ids, d, hh], [0.25, 0.5, 0.75]) if np.isfinite(mfe[ids, d, hh]).any() else np.nan
                    q_mae[i, d, hh] = np.nanquantile(mae[ids, d, hh], [0.25, 0.5, 0.75]) if np.isfinite(mae[ids, d, hh]).any() else np.nan
    return {"p_win": p_win, "b_eff": b_eff, "ev_ref": ev_ref, "q_mfe": q_mfe, "q_mae": q_mae}

# ---------------------------------------------------------------------------
# 3b) 48-cell adjacent triangular smoothing (Spec v1.2: primary H1 target =
#      realized hard outcomes + triangular smoothing over the cell grid)
# ---------------------------------------------------------------------------

def build_smooth_kernel():
    """Triangular kernel over the 48-cell grid (own 1.0, stop/TP +-1 -> 0.5,
    diagonal (m+-1,t+-1) -> 0.25, horizon +-1 (same m,t) -> 0.5; long and
    short directions are NEVER mixed). Rows sum to 1. Exact port of the
    postmortem aurum_common.build_smooth_kernel (exp2 T2_smoothed, the
    walk-forward winning target family)."""
    K = np.zeros((N_CELLS, N_CELLS))
    for i in range(N_CELLS):
        for j in range(N_CELLS):
            if i == j:
                K[i, j] = 1.0
                continue
            # decode flat index: d*24 + mi*6 + ti*2 + hi
            if (i // 24) != (j // 24):
                continue
            dmi = abs((i % 24) // 6 - (j % 24) // 6)
            dti = abs((i % 6) // 2 - (j % 6) // 2)
            dhi = abs((i % 2) - (j % 2))
            if dhi > 1 or dmi > 1 or dti > 1:
                continue
            if dhi == 1 and dmi == 0 and dti == 0:
                K[i, j] = 0.5
            elif dmi == 1 and dti == 0 and dhi == 0:
                K[i, j] = 0.5
            elif dti == 1 and dmi == 0 and dhi == 0:
                K[i, j] = 0.5
            elif dmi == 1 and dti == 1 and dhi == 0:
                K[i, j] = 0.25
    return K / K.sum(axis=1, keepdims=True)

def smoothed_targets(win_matrix):
    """win_matrix: (N, 48) float realized win flags (NaN where uncomputed).
    Returns (N, 48) triangular-smoothed targets in [0,1]; rows with any NaN
    cell stay NaN (label_valid bars have all 48 cells finite)."""
    K = build_smooth_kernel()
    W = win_matrix.copy()
    bad = ~np.isfinite(W).all(axis=1)
    W[bad] = 0.0
    out = W @ K.T
    out[bad] = np.nan
    return out

# ---------------------------------------------------------------------------
# 3c) Cohort INPUT summary (Spec v1.2: canonical long/short p_win, b_eff and
#     the cohort EV-argmax cell id, from neighbors <= i-96 only - these enter
#     the SERIALIZED STATE as compact inputs, never the targets)
# ---------------------------------------------------------------------------

CANON_L = 0 * 24 + 1 * 6 + 1 * 2 + 1      # L_m1.5_t2.5_h48  (index 9, pilot canon)
CANON_S = 1 * 24 + 1 * 6 + 1 * 2 + 1      # S_m1.5_t2.5_h48  (index 33)

def cohort_input_summary(soft):
    """From cohort_soft_labels output (p_win/b_eff computed over the SAME
    gap=96 past-only neighbor sets): per-bar canonical L/S p_win, b_eff and
    the EV-argmax cell id over all 48 cells. Returns dict of (N,) arrays."""
    p_win, b_eff = soft["p_win"], soft["b_eff"]
    with np.errstate(invalid="ignore"):
        ev = p_win * b_eff - (1.0 - p_win)          # (N, 48) EV in R units
        ok = np.isfinite(ev).all(axis=1)
        arg = np.full(len(ev), -1, dtype=np.int16)
        if ok.any():
            ev_safe = np.where(ok[:, None], ev, -np.inf)
            arg[ok] = np.argmax(ev_safe, axis=1)[ok]
    return {
        "coh_Lpw": p_win[:, CANON_L], "coh_Lb": b_eff[:, CANON_L],
        "coh_Spw": p_win[:, CANON_S], "coh_Sb": b_eff[:, CANON_S],
        "coh_arg": arg,
    }

# ---------------------------------------------------------------------------
# 4) Auxiliary head targets: H2 (MTM), H4 (vol), H5 (regime)
# ---------------------------------------------------------------------------

def aux_targets(h1, actual, atr):
    """H2: P(up) & magnitude bucket at h in {4,12,24,48}.
       H4: vol expansion prob at h in {4,12,24,48} + tercile class + q50 ratio.
       H5: 3x3 regime class (trend x vol) from past-only tercile edges."""
    c = h1["c"].values
    N = len(c)
    out = {}
    # H2 (moves in ATR units, matching the feature construction)
    for hh in (4, 12, 24, 48):
        fwd = np.full(N, np.nan)
        fwd[:-hh] = (c[hh:] - c[:-hh]) / atr[:-hh]
        sgn = (fwd > 0).astype(np.float64)
        mag = np.abs(fwd)
        med = pd.Series(mag).rolling(252, min_periods=60).median().shift(1).values
        out[f"h2_up{hh}"] = sgn
        out[f"h2_big{hh}"] = (mag > med).astype(np.float64)
    # H4: realized vol ratio fwd12/fwd48 (past window), expansion & terciles
    lr = np.log(c / np.roll(c, 1)); lr[0] = 0.0
    rv12 = pd.Series(lr).pow(2).rolling(12, min_periods=12).sum().pow(0.5).values
    rv48 = pd.Series(lr).pow(2).rolling(48, min_periods=48).sum().pow(0.5).values
    ratio = pd.Series(rv12 / (rv48 + EPS))
    for hh in (4, 12, 24, 48):
        fwd = ratio.shift(-hh).values
        out[f"h4_exp{hh}"] = (fwd > 1.0).astype(np.float64)
    q33 = ratio.rolling(252, min_periods=60).quantile(0.33).shift(1).values
    q66 = ratio.rolling(252, min_periods=60).quantile(0.66).shift(1).values
    fwd12 = ratio.shift(-12).values
    terc = np.full(N, 255, dtype=np.int16)
    terc[(fwd12 < q33)] = 0
    terc[(fwd12 >= q33) & (fwd12 <= q66)] = 1
    terc[(fwd12 > q66)] = 2
    out["h4_terc12"] = terc
    out["h4_q50_12"] = fwd12
    # H5 regime: trend = 24-bar move in ATR units vs past terciles; vol = atr_ratio terciles
    r24 = pd.Series((c - np.roll(c, 24)) / (atr + EPS))
    r24.iloc[0] = np.nan
    arr = r24.replace([np.inf, -np.inf], np.nan)
    t33 = arr.rolling(252, min_periods=60).quantile(0.33).shift(1).values
    t66 = arr.rolling(252, min_periods=60).quantile(0.66).shift(1).values
    trend = np.full(N, 1, dtype=np.int8)
    trend[arr.values < t33] = 0
    trend[arr.values > t66] = 2
    ar = pd.Series(atr / np.roll(atr, 48)); ar.iloc[0] = np.nan
    v33 = ar.rolling(252, min_periods=60).quantile(0.33).shift(1).values
    v66 = ar.rolling(252, min_periods=60).quantile(0.66).shift(1).values
    volc = np.full(N, 1, dtype=np.int8)
    volc[ar.values < v33] = 0
    volc[ar.values > v66] = 2
    out["h5_regime"] = (trend * 3 + volc).astype(np.int8)
    return out

# ---------------------------------------------------------------------------
# 5) Deterministic exit stack (MVP, Spec v1.1 Sec 12): q25 MAE trail,
#    q50 MFE partial TP, cell-m hard stop, H time stop.
# ---------------------------------------------------------------------------

def deterministic_exit_plan(p_win_row, b_eff_row, q_mfe_row, q_mae_row):
    """Given per-bar label rows, compute the trade plan the backtest will use:
    EV-argmax cell -> (direction, stop_m, tp_t, horizon, p, b), quarter-Kelly
    risk fraction with 1% hard cap, and exit levels (in ATR units):
    hard stop m, partial TP at q50 MFE (close half), trail q25 MAE below
    high-water mark, time stop at the cell horizon."""
    ev = p_win_row * b_eff_row - (1.0 - p_win_row)     # EV in R units
    m = int(np.nanargmax(ev))
    dirs, ms, ts, hs = cell_axes()
    p = float(p_win_row[m]); b = float(b_eff_row[m])
    if not np.isfinite(p) or not np.isfinite(b) or p <= 0 or b <= 0:
        return None
    kelly = p - (1 - p) / b
    risk = min(0.25 * max(kelly, 0.0), 0.01)           # quarter Kelly, 1% cap
    if risk <= 0:
        return None
    d = 0 if dirs[m] == "L" else 1
    hidx = 0 if hs[m] == 24 else 1
    return {
        "cell": m, "dir": dirs[m], "stop_atr": ms[m], "tp_atr": ts[m],
        "horizon": hs[m], "p": p, "b": b, "ev": float(ev[m]),
        "risk_frac": risk,
        "partial_tp_atr": float(q_mfe_row[d, hidx, 1]),   # q50 MFE
        "trail_mae_atr": float(q_mae_row[d, hidx, 0]),    # q25 MAE
    }


# ==================== lib/aurum_data.py ====================
# ============================================================================
# AURUM data ingestion: HistData M1 (primary), Dukascopy gap-fill (fallback),
# FRED daily macro (DTWEXBGS/DGS10/DGS2/T10Y2Y/VIXCLS). Free sources only.
# ============================================================================
import io, os, re, time, zipfile, requests
import numpy as np
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "Accept": "text/html,application/xhtml+xml,application/zip,*/*"}

FRED_SERIES = {"dxy": "DTWEXBGS", "y10": "DGS10", "y2": "DGS2", "spread": "T10Y2Y", "vix": "VIXCLS"}

# ---------------------------------------------------------------------------
# HistData: token-flow downloader for yearly ASCII M1 archives
# ---------------------------------------------------------------------------

def histdata_year_url(pair, year):
    return f"https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair.lower()}/{year}"

def histdata_download_year(pair, year, out_dir, session=None, tries=4, sleep_s=3.0, log=print):
    """Downloads the full-year ASCII M1 zip for pair (e.g. 'xauusd'). Returns
    path to zip or None. Parses the actual `file_down` form fields (tk token
    + platform/timeframe/fxpair controls) and POSTs them to /get.php."""
    path = os.path.join(out_dir, f"{pair.upper()}_M1_{year}.zip")
    if os.path.exists(path) and os.path.getsize(path) > 10_000:
        return path
    s = session or requests.Session()
    s.headers.update(UA)
    url = histdata_year_url(pair, year)
    for attempt in range(tries):
        try:
            r = s.get(url, timeout=60)
            if r.status_code != 200:
                log(f"  histdata {pair} {year}: GET {r.status_code}")
                time.sleep(sleep_s * (attempt + 1)); continue
            html = r.text
            form = re.search(r'<form id="file_down".*?</form>', html, re.S)
            if not form:
                log(f"  histdata {pair} {year}: no file_down form (len {len(html)})")
                time.sleep(sleep_s * (attempt + 1)); continue
            payload = {}
            for tag in re.findall(r"<input[^>]*>", form.group(0)):
                nm = re.search(r'name="([^"]*)"', tag)
                vl = re.search(r'value="([^"]*)"', tag)
                if nm:
                    payload[nm.group(1)] = vl.group(1) if vl else ""
            if "tk" not in payload or not payload["tk"]:
                log(f"  histdata {pair} {year}: no tk token in form")
                time.sleep(sleep_s * (attempt + 1)); continue
            payload["datemonth"] = str(year)   # full year
            pr = s.post("https://www.histdata.com/get.php", data=payload,
                        headers={"Referer": url}, timeout=600)
            ct = pr.headers.get("content-type", "")
            if pr.status_code == 200 and (pr.content[:2] == b"PK") and len(pr.content) > 10_000:
                os.makedirs(out_dir, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(pr.content)
                log(f"  histdata {pair} {year}: OK {len(pr.content)/1e6:.1f} MB")
                return path
            log(f"  histdata {pair} {year}: POST st={pr.status_code} ct={ct} len={len(pr.content)}")
        except Exception as e:
            log(f"  histdata {pair} {year}: exc {type(e).__name__}: {str(e)[:120]}")
        time.sleep(sleep_s * (attempt + 1))
    return None

def parse_histdata_zip(zip_path, log=print):
    """Returns M1 DataFrame (UTC tz-aware index, o/h/l/c/v) or None.
    Handles both flat csv-in-zip and nested monthly zips."""
    frames = []
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            for nm in names:
                if nm.lower().endswith(".csv"):
                    frames.append(_read_hist_csv(z.read(nm)))
                elif nm.lower().endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(z.read(nm))) as z2:
                        for nm2 in z2.namelist():
                            if nm2.lower().endswith(".csv"):
                                frames.append(_read_hist_csv(z2.read(nm2)))
    except Exception as e:
        log(f"  parse {os.path.basename(zip_path)}: {type(e).__name__} {str(e)[:150]}")
        return None
    if not frames:
        return None
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df

def _read_hist_csv(raw):
    txt = raw.decode("latin-1").replace("\x00", "")
    # header: GMT time;Open;High;Low;Close;Volume, rows "20240101 180000;2062.598;..."
    df = pd.read_csv(io.StringIO(txt), sep=";", header=0, dtype=str,
                     usecols=[0, 1, 2, 3, 4, 5])
    df.columns = ["ts", "o", "h", "l", "c", "v"]
    for col in ("o", "h", "l", "c"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["v"] = pd.to_numeric(df["v"], errors="coerce").fillna(0.0)
    ts = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S", utc=True)
    df.index = ts
    df = df.dropna(subset=["o", "h", "l", "c"])
    df = df[(df["o"] > 0) & (df["h"] >= df[["o", "c"]].min(axis=1) - 1e-9)
            & (df["l"] <= df[["o", "c"]].max(axis=1) + 1e-9)]
    return df[["o", "h", "l", "c", "v"]]

# ---------------------------------------------------------------------------
# Dukascopy fallback (gap filler): per-day BID M1 bi5 (LZMA) files
# ---------------------------------------------------------------------------

def dukascopy_day_url(pair, y, m, d):
    return f"https://datafeed.dukascopy.com/datafeed/{pair.upper()}/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"

def parse_bi5(raw):
    try:
        import lzma
        data = lzma.decompress(raw)
    except Exception:
        return None
    n = len(data) // 24
    if n == 0:
        return None
    arr = np.frombuffer(data[:n * 24], dtype=">i4").reshape(n, 6).astype(np.float64)
    # cols: sec-offset(ms?), open, close, low, high, volume  (point-scaled)
    # Dukascopy M1: [time_ms_offset, open, close, low, high, volume], prices *1e5 (fx) or *1e3 (metals)
    return arr

# ---------------------------------------------------------------------------
# FRED (fredgraph.csv endpoints - no API key needed)
# ---------------------------------------------------------------------------

def fred_fetch(series_code, log=print):
    """FRED daily series via fredgraph.csv (no API key). requests first,
    curl-subprocess fallback (some networks throttle python-requests UAs)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}"
    txt = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=45)
            if r.status_code == 200 and "observation_date" in r.text[:200]:
                txt = r.text
                break
            log(f"  FRED {series_code}: st={r.status_code}")
        except Exception as e:
            log(f"  FRED {series_code}: exc {type(e).__name__}")
        time.sleep(4 * (attempt + 1))
    if txt is None:
        try:
            import subprocess
            p = subprocess.run(["curl", "-sS", "--max-time", "90", url],
                               capture_output=True, text=True, timeout=120)
            if p.returncode == 0 and "observation_date" in p.stdout[:200]:
                txt = p.stdout
                log(f"  FRED {series_code}: OK via curl fallback")
        except Exception as e:
            log(f"  FRED {series_code}: curl exc {type(e).__name__}")
    if txt is None:
        return None
    try:
        df = pd.read_csv(io.StringIO(txt))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
        s = df.set_index("date")["value"].sort_index()
        log(f"  FRED {series_code}: OK {len(s)} rows ({s.index.min().date()}..{s.index.max().date()})")
        return s
    except Exception as e:
        log(f"  FRED {series_code}: parse exc {type(e).__name__}")
        return None

# ---------------------------------------------------------------------------
# UTC-offset estimation for HistData clocks (their ASCII export has run in a
# server-local clock historically). We correlate the hourly volatility profile
# against a canonical forex session template and pick the best integer shift.
# ---------------------------------------------------------------------------

def estimate_utc_offset(m1, log=print, candidates=(-3, -2, -1, 0, 1, 2, 3)):
    """m1: M1 df with tz-UTC index (as-parsed clock). Returns (offset_hours, corr)."""
    r = np.log(m1["c"] / m1["c"].shift(1)).abs()
    prof = r.groupby(m1.index.hour).mean().reindex(range(24), fill_value=0.0).values
    prof = (prof - prof.mean()) / (prof.std() + 1e-12)
    # canonical template: London open 07-08 UTC, NY open 12-13 UTC, overlap 15-16
    tpl = np.zeros(24)
    tpl[[7, 8, 12, 13, 15, 16]] = 1.6
    tpl[[3, 4, 5, 6, 9, 10, 17, 18]] = 0.8
    tpl[[21, 22, 23, 0, 1, 2]] = 0.4
    tpl = (tpl - tpl.mean()) / (tpl.std() + 1e-12)
    best_off, best_c = 0, -2
    for off in candidates:
        c = np.corrcoef(np.roll(prof, off), tpl)[0, 1]
        if c > best_c:
            best_c, best_off = c, off
    c0 = np.corrcoef(prof, tpl)[0, 1]
    if best_c < c0 * 1.2 or best_c < 0.35:
        best_off, best_c = 0, c0   # ambiguous -> keep as-is
    log(f"  utc-offset est: offset={best_off:+d}h corr={best_c:.2f} (null corr={c0:.2f})")
    return best_off, float(best_c)

def apply_utc_offset(df, offset_hours):
    if offset_hours == 0:
        return df
    return df.set_index(df.index + pd.Timedelta(hours=offset_hours))

# ---------------------------------------------------------------------------
# H1 aggregation + coverage report
# ---------------------------------------------------------------------------

def to_h1(m1):
    h1 = m1.resample("1h").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"),
                               c=("c", "last"), v=("v", "sum"))
    return h1.dropna(subset=["o", "c"])

def coverage_report(m1, years=range(2010, 2026)):
    rep = {}
    exp = 365.25 * 24 * 60 * (5 / 7)  # approx tradable minutes per year
    for y in years:
        sub = m1[(m1.index.year == y)]
        got = len(sub)
        rep[str(y)] = {"rows": int(got), "coverage": round(got / exp, 3) if exp else None}
    return rep


# ==================== lib/aurum_ser.py ====================
# ============================================================================
# AURUM state serialization (Spec v1.1 sec 7.1): each H1 state becomes a
# ~312-token dimensionless text block. No absolute prices, no absolute dates -
# only cyclical calendar (hour-of-day, dow, session), ATR-unit quantities,
# z-scores, ratios, codebook ids.
# ============================================================================
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


# ---------------------------------------------------------------------------
# Shape codebooks (fit on TRAIN window only, then applied everywhere)
# ---------------------------------------------------------------------------

def m1_path_features(m1, h1_index, n_pts=10):
    """For each H1 bar: normalized intrabar M1 path shape.
    Returns (N, n_pts+2) matrix: n_pts resampled normalized closes + relative
    time position of path high & low (all dimensionless)."""
    N = len(h1_index)
    X = np.full((N, n_pts + 2), np.nan)
    h1_starts = h1_index.asi8
    m1_idx = m1.index.asi8
    m1_c = m1["c"].values
    m1_o = m1["o"].values
    m1_h = m1["h"].values
    m1_l = m1["l"].values
    H1ns = 3_600_000_000_000
    for i in range(N):
        s = np.searchsorted(m1_idx, h1_starts[i], side="left")
        e = np.searchsorted(m1_idx, h1_starts[i] + H1ns, side="left")
        if e - s < 10:
            continue
        c = m1_c[s:e]; o = m1_o[s]
        rng = c.max() - c.min()
        if rng <= 0:
            continue
        norm = (c - c.min()) / rng
        # resample to n_pts points (indices, not values, to stay time-uniform)
        pos = np.linspace(0, len(norm) - 1, n_pts)
        X[i, :n_pts] = np.interp(pos, np.arange(len(norm)), norm)
        X[i, n_pts] = np.argmax(m1_h[s:e]) / (e - s)       # rel time of high
        X[i, n_pts + 1] = np.argmin(m1_l[s:e]) / (e - s)   # rel time of low
    return X

def h1_path_features(h1, n_bars=8):
    """Normalized shape of the last n_bars H1 closes (z-scored path)."""
    c = h1["c"].values
    N = len(c)
    X = np.full((N, n_bars), np.nan)
    for i in range(n_bars - 1, N):
        seg = c[i - n_bars + 1: i + 1]
        sd = seg.std()
        X[i] = (seg - seg.mean()) / (sd + 1e-12) if sd > 0 else 0.0
    return X

def fit_codebook(X, k=32, seed=13, sample=200_000):
    rows = X[np.isfinite(X).all(axis=1)]
    if len(rows) > sample:
        rows = rows[np.random.default_rng(seed).choice(len(rows), sample, replace=False)]
    km = MiniBatchKMeans(n_clusters=k, n_init=5, random_state=seed, batch_size=4096).fit(rows)
    return km

def apply_codebook(km, X, index=None):
    codes = np.full(len(X), -1, dtype=np.int32)
    ok = np.isfinite(X).all(axis=1)
    if ok.any():
        codes[ok] = km.predict(X[ok])
    return pd.Series(codes, index=index if index is not None else range(len(X)))

# ---------------------------------------------------------------------------
# State serialization
# ---------------------------------------------------------------------------

def _f(x, nd=2, signed=True):
    if x is None or not np.isfinite(x):
        return "na"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"

def _u(x, nd=2):
    if x is None or not np.isfinite(x):
        return "na"
    return f"{x:.{nd}f}"

def _r100(x):
    """rank60 in [0,1] -> integer 0..99 (compact, 1-2 tokens)."""
    if x is None or not np.isfinite(x):
        return "na"
    return f"{int(round(float(x) * 100)):02d}"

def _c(x, nd=1):
    """v1.2 compact number format: unsigned ('-' only for negatives), no
    leading zero before the decimal point ('.88' / '-.4' / '10.8'). Measured
    ~2 tokens/field vs ~5 for '+0.88' - this is what keeps the FULL 15-value
    macro panel inside the ~313-330-token state budget."""
    if x is None or not np.isfinite(x):
        return "na"
    s = f"{float(x):.{nd}f}"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s

# Spec v1.2 macro innovation panel: (state key, feature base name)
MINNOV_SERIES = (("y10", "mi_y10"), ("y2", "mi_y2"), ("sp", "mi_spread"),
                 ("dx", "mi_dxy"), ("vx", "mi_vix"))

def serialize_state(ts, row, with_cross=True, with_macro=True, with_priors=True,
                   with_minnov=False, with_zpriors=False, with_cohort=False,
                   trend3=None, vol3=None, drop=None, compact=False):
    """ts: bar timestamp (UTC, end of bar); row: dict/Series of features.
    trend3/vol3 may be passed directly or live in the row (0/1/2 class ids).
    drop: set of feature names excluded by the adversarial-validation audit -
    their fields/lines are omitted from the frozen state format.
    v1.2 additions (all default OFF so the frozen v1.1 call sites are
    byte-identical):
      with_minnov  - 'minnov' line: per FRED series d20z/d5z/rank60 triplet
                     (era-invariant macro innovations, strict 1-day lag)
      with_zpriors - 'prior' line carries SELF-Z-SCORED priors (era-safe) in
                     place of the raw trailing rates
      with_cohort  - 'coh' line: canonical L/S p_win, b_eff + EV-argmax cell id
                     (neighbors <= i-96 only)
      compact      - v1.2 number encoding (unsigned, no leading zero, 1dp on
                     low-information fields) -> full format fits ~313-330 tokens
    Returns the text block (~313-330 Qwen tokens in the v1.2 format)."""
    drop = drop or set()
    hour = ts.hour
    trend = row.get("trend3", np.nan) if trend3 is None else trend3
    volc = row.get("vol3", np.nan) if vol3 is None else vol3
    L = []
    L.append(f"[STATE h{hour:02d} dow={DOW[ts.weekday()]} sess={session_of_hour(hour)}]")
    if compact:
        L.append("ret_atr r1={r1} r4={r4} r12={r12} r24={r24} r48={r48}".format(
            r1=_c(row.get("ret1_atr"), 2), r4=_c(row.get("ret4_atr"), 2),
            r12=_c(row.get("ret12_atr"), 2), r24=_c(row.get("ret24_atr"), 2),
            r48=_c(row.get("ret48_atr"), 2)))
        L.append("vol_atr atrr={atrr} volz={volz}".format(
            atrr=_c(row.get("atr_ratio"), 2), volz=_c(row.get("vol_z"), 2)))
        L.append("osc rsi={rsi} stoch={st} macd={macd}".format(
            rsi=_c(row.get("rsi_z"), 2), st=_c(row.get("stoch_z"), 2), macd=_c(row.get("macd_atr"), 2)))
        L.append("range bb={bb} bbw={bbw} d20={d20} d55={d55} run={run}".format(
            bb=_c(row.get("bb_pos"), 2), bbw=_c(row.get("bb_w_z"), 2), d20=_c(row.get("d20"), 2),
            d55=_c(row.get("d55"), 2), run=_c(row.get("run"), 2)))
        L.append("candle c1=b{b1}/u{u1}/d{d1} c2=b{b2}/u{u2}/d{d2} c3=b{b3}/u{u3}/d{d3}".format(
            b1=_c(row.get("body1"), 2), u1=_c(row.get("uwick1"), 2), d1=_c(row.get("lwick1"), 2),
            b2=_c(row.get("body2"), 2), u2=_c(row.get("uwick2"), 2), d2=_c(row.get("lwick2"), 2),
            b3=_c(row.get("body3"), 2), u3=_c(row.get("uwick3"), 2), d3=_c(row.get("lwick3"), 2)))
        L.append("shape m1={m1} h1={h1} vlm={vlm}".format(
            m1=("na" if not np.isfinite(row.get("m1_shape", np.nan)) else f"{int(row['m1_shape']):02d}"),
            h1=("na" if not np.isfinite(row.get("h1_shape", np.nan)) else f"{int(row['h1_shape']):02d}"),
            vlm=_c(row.get("volm_z"), 2)))
    else:
        L.append("ret_atr r1={r1} r4={r4} r12={r12} r24={r24} r48={r48}".format(
            r1=_f(row.get("ret1_atr")), r4=_f(row.get("ret4_atr")), r12=_f(row.get("ret12_atr")),
            r24=_f(row.get("ret24_atr")), r48=_f(row.get("ret48_atr"))))
        L.append("vol_atr atrr={atrr} volz={volz}".format(
            atrr=_u(row.get("atr_ratio")), volz=_f(row.get("vol_z"))))
        L.append("osc rsi={rsi} stoch={st} macd={macd}".format(
            rsi=_f(row.get("rsi_z")), st=_f(row.get("stoch_z")), macd=_f(row.get("macd_atr"))))
        L.append("range bb={bb} bbw={bbw} d20={d20} d55={d55} run={run}".format(
            bb=_f(row.get("bb_pos")), bbw=_f(row.get("bb_w_z")), d20=_u(row.get("d20")),
            d55=_u(row.get("d55")), run=_f(row.get("run"))))
        L.append("candle c1=b{b1}/u{u1}/d{d1} c2=b{b2}/u{u2}/d{d2} c3=b{b3}/u{u3}/d{d3}".format(
            b1=_f(row.get("body1")), u1=_f(row.get("uwick1"), 2), d1=_f(row.get("lwick1")),
            b2=_f(row.get("body2")), u2=_f(row.get("uwick2")), d2=_f(row.get("lwick2")),
            b3=_f(row.get("body3")), u3=_f(row.get("uwick3")), d3=_f(row.get("lwick3"))))
        L.append("shape m1={m1} h1={h1} vlm={vlm}".format(
            m1=("na" if not np.isfinite(row.get("m1_shape", np.nan)) else f"{int(row['m1_shape']):02d}"),
            h1=("na" if not np.isfinite(row.get("h1_shape", np.nan)) else f"{int(row['h1_shape']):02d}"),
            vlm=_f(row.get("volm_z"))))
    if with_cross:
        xg = [("xag", "x_xagusd", row.get("x_xagusd")), ("eur", "x_eurusd", row.get("x_eurusd")),
              ("jpy", "x_usdjpy", row.get("x_usdjpy"))]
        xg = [(a, v) for a, k, v in xg if k not in drop]
        if xg:
            fx = (lambda v: _c(v, 2)) if compact else (lambda v: _f(v))
            L.append("xasset " + " ".join(f"{a}={fx(v)}" for a, v in xg))
    if with_macro:
        mc = [("dxy", "m_dxy", row.get("m_dxy")), ("y10", "m_y10", row.get("m_y10")),
              ("vix", "m_vix", row.get("m_vix"))]
        mc = [(a, v) for a, k, v in mc if k not in drop]
        if mc:
            fx = (lambda v: _c(v, 2)) if compact else (lambda v: _f(v))
            L.append("macro " + " ".join(f"{a}={fx(v)}" for a, v in mc))
    if with_minnov:
        # per series: d20z(1dp) d5z(1dp) rank60(int 0-99), space separated
        parts = []
        for short, base in MINNOV_SERIES:
            if f"{base}_d20z" in drop:
                continue
            parts.append(f"{short} {_c(row.get(base + '_d20z'))} "
                         f"{_c(row.get(base + '_d5z'))} "
                         f"{_r100(row.get(base + '_rank60'))}")
        if parts:
            L.append("minnov " + " ".join(parts))
    trend_v, volc_v = trend, volc
    try:
        if np.isfinite(trend_v) and np.isfinite(volc_v):
            tn = ["down", "flat", "up"][int(trend_v)]; vn = ["low", "mid", "high"][int(volc_v)]
            L.append(f"regime trend={tn} vol={vn}")
    except (TypeError, ValueError):
        pass
    if with_priors and not with_zpriors:
        pr = [("w24", "prior_w24", row.get("prior_w24")), ("l24", "prior_l24", row.get("prior_l24")),
              ("w48", "prior_w48", row.get("prior_w48")), ("l48", "prior_l48", row.get("prior_l48"))]
        pr = [(a, v) for a, k, v in pr if k not in drop]
        if pr:
            L.append("prior " + " ".join(f"{a}={_u(v)}" for a, v in pr))
    if with_zpriors:
        # Spec v1.2: self-z-scored base rates (era-safe encoding of the 4
        # priors; raw trailing win rates were date fingerprints at AUC .85-.90)
        zr = [("zw24", "prior_zw24"), ("zl24", "prior_zl24"),
              ("zw48", "prior_zw48"), ("zl48", "prior_zl48")]
        zr = [(a, row.get(c)) for a, c in zr if c not in drop]
        if zr:
            L.append("prior " + " ".join(f"{a}={_c(v, 2)}" for a, v in zr))
    if with_cohort:
        # Spec v1.2 cohort input summary (neighbors <= i-96 only)
        argv = row.get("coh_arg")
        arg_s = "na" if argv is None or not np.isfinite(argv) else f"{int(argv):02d}"
        cf = [("Lpw", "coh_Lpw", _c(row.get("coh_Lpw"), 2)),
              ("Lb", "coh_Lb", _c(row.get("coh_Lb"), 2)),
              ("Spw", "coh_Spw", _c(row.get("coh_Spw"), 2)),
              ("Sb", "coh_Sb", _c(row.get("coh_Sb"), 2)),
              ("arg", "coh_arg", arg_s)]
        cf = [(a, v) for a, k, v in cf if k not in drop]
        if cf:
            L.append("coh " + " ".join(f"{a}={v}" for a, v in cf))
    return "\n".join(L)

def serialize_states(df_feat, h1_index, **kw):
    """Serialize all states. Returns list of texts."""
    out = []
    trend = kw.pop("trend3", None); volc = kw.pop("vol3", None)
    for i, ts in enumerate(h1_index):
        row = df_feat.iloc[i]
        if trend is not None:
            row = row.copy()
            row["trend3"] = trend[i]; row["vol3"] = volc[i]
        out.append(serialize_state(ts, row, **kw))
    return out


# ==================== lib/aurum_model.py ====================
# ============================================================================
# AURUM LoopedQwen3 (Spec v1.1, frozen):
#   - Base: Qwen/Qwen3-0.6B (28 decoder layers, d=1024).
#   - Looped core: decoder layers 13..16 (inclusive) re-executed R times
#     (R in {1,2,3}); shared weights, per-iteration Sandwich RMSNorm with
#     identity init (audit fix #5: full unfreeze of the ~60M core params
#     instead of rank-16 LoRA; T4 16GB holds it).
#   - Sandwich design (per iteration r):
#         z = pre_r(h);  z = core(z);  h = clamp(post_r(z))
#     with pre/post = x + tanh(g)*(RMSNorm(x) - x), g=0 at init, so
#     R=1 is EXACTLY the pretrained model at initialization (verified by
#     the r=1 equivalence self-test).
#   - fp16 guardrails (user directive): torch.amp autocast(fp16) + GradScaler
#     everywhere; RMSNorm accumulates in fp32; pre-norm residual norms are
#     clamped (C=100) so the looped stream can never overflow fp16 on sm_75.
#   - Joint Multi-Task World-Model Head: 5 learned readout tokens appended at
#     the end of the state sequence (guardrail: no variable-option permutation
#     masks). Single forward pass -> 85 outputs:
#       H1_BARRIER 48 (per-cell P(win), sigmoid)
#       H2_MTM      8 (multi-horizon direction & magnitude, sigmoid)
#       H3_EXCUR   12 (MFE/MAE q25/q50/q75 in ATR units, softplus)
#       H4_VOL      8 (expansion probs + terciles + q50)
#       H5_REGIME   9 (trend x vol classes, softmax)
# ============================================================================
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint as tk_ckpt


BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
CORE_START, CORE_END = 13, 17          # decoder layers 13..16 inclusive
N_READOUT = 5
# Qwen3-0.6B residual-stream L2 norms reach ~7.5k on real activations (measured:
# mean ~200-700, max 7464). The clamp is a runaway guard for the looped core,
# NOT a normal-path operation: C=16384 sits 2x above the observed stock max and
# 4x below fp16 overflow (65504), so it never fires on healthy activations.
RESID_CLAMP = 16384.0

# ---------------------------------------------------------------------------


class Fp32RMSNorm(nn.Module):
    """RMSNorm with fp32 accumulation (weight kept fp32)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x32 = x.float()
        v = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (v * self.weight.float()).to(dt)


class SandwichNorm(nn.Module):
    """x + tanh(g) * (RMSNorm(x) - x); g=0 -> exact identity at init."""

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1))
        self.n = Fp32RMSNorm(dim)

    def forward(self, x):
        dt = x.dtype
        x32 = x.float()
        n32 = self.n(x).float()
        out = x32 + torch.tanh(self.g).float() * (n32 - x32)
        return out.to(dt)


def clamp_resid(x, C=RESID_CLAMP, stats=None):
    """Clamp the L2 norm of the residual stream (fp32 math, per token)."""
    n = x.float().norm(dim=-1, keepdim=True)
    s = torch.clamp(C / (n + 1e-6), max=1.0)
    if stats is not None and torch.is_grad_enabled() and n.numel() > 0:
        with torch.no_grad():
            nm = n.max()
            stats["resid_max"] = max(stats.get("resid_max", 0.0), float(nm))
            stats["clamp_count"] = stats.get("clamp_count", 0.0) + float((n > C).sum())
            stats["clamp_total"] = stats.get("clamp_total", 0.0) + float(n.numel())
    return (x.float() * s).to(x.dtype)


def build_causal_mask(T, dtype, device):
    m = torch.full((1, 1, T, T), torch.finfo(dtype).min, dtype=torch.float32, device=device)
    m = torch.triu(m, diagonal=1)
    return m.to(dtype)


def build_joint_mask(pad_mask, T, L, dtype, device):
    """Causal + key-padding mask. pad_mask: (B, L) True=real token.
    Readout positions (L..T-1) are always valid keys."""
    B = pad_mask.size(0)
    key_ok = torch.ones(B, T, dtype=torch.bool, device=device)
    key_ok[:, :L] = pad_mask
    neg = torch.finfo(dtype).min
    m = torch.zeros(B, 1, T, T, dtype=torch.float32, device=device)
    m.masked_fill_(~key_ok[:, None, None, :], neg)
    causal = torch.triu(torch.full((T, T), neg, dtype=torch.float32, device=device), diagonal=1)
    m = m + causal[None, None]
    return m.clamp(min=neg).to(dtype)


def call_layer(layer, h, pe, pos_ids, attn_mask):
    kw = dict(attention_mask=attn_mask, position_ids=pos_ids, use_cache=False,
              output_attentions=False)
    if pe is not None:
        kw["position_embeddings"] = pe
    try:
        out = layer(h, **kw)
    except TypeError:
        kw.pop("position_embeddings", None)
        kw.pop("position_ids", None)
        out = layer(h, attention_mask=attn_mask)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _mlp_head(d, hidden, out):
    return nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out))


class JointHead(nn.Module):
    def __init__(self, d_model, hidden=256):
        super().__init__()
        self.readout = nn.Embedding(N_READOUT, d_model)
        nn.init.normal_(self.readout.weight, std=0.02)
        self.h1 = _mlp_head(d_model, hidden, HEAD_DIMS["h1"])   # 48 sigmoid
        self.h2 = _mlp_head(d_model, hidden, HEAD_DIMS["h2"])   # 8 sigmoid
        self.h3 = _mlp_head(d_model, hidden, HEAD_DIMS["h3"])   # 12 softplus
        self.h4 = _mlp_head(d_model, hidden, HEAD_DIMS["h4"])   # 8 sigmoid(+)
        self.h5 = _mlp_head(d_model, hidden, HEAD_DIMS["h5"])   # 9 softmax

    def forward(self, ro_hidden):
        return {"h1": self.h1(ro_hidden[:, 0]), "h2": self.h2(ro_hidden[:, 1]),
                "h3": self.h3(ro_hidden[:, 2]), "h4": self.h4(ro_hidden[:, 3]),
                "h5": self.h5(ro_hidden[:, 4])}

    @torch.no_grad()
    def predict(self, logits):
        """Apply the frozen activations -> dict of probabilities/values.
        h4 layout: [0:4] expansion probs (sigmoid), [4:7] vol tercile (softmax),
        [7] fwd12/48 vol-ratio q50 (softplus)."""
        h4 = torch.cat([torch.sigmoid(logits["h4"][:, :4]),
                        torch.softmax(logits["h4"][:, 4:7], dim=-1),
                        Fn.softplus(logits["h4"][:, 7:8])], dim=1)
        out = {"h1": torch.sigmoid(logits["h1"]), "h2": torch.sigmoid(logits["h2"]),
               "h3": Fn.softplus(logits["h3"]), "h4": h4,
               "h5": torch.softmax(logits["h5"], dim=-1)}
        return out


class LoopedQwen(nn.Module):
    def __init__(self, base, n_loop=3, sandwich=True, clamp=RESID_CLAMP,
                 grad_ckpt=False, explicit_mask=False):
        super().__init__()
        self.base = base                       # AutoModelForCausalLM
        self.n_loop = n_loop
        self.sandwich = sandwich
        self.clamp = clamp
        self.grad_ckpt = grad_ckpt
        self.explicit_mask = explicit_mask
        d = base.config.hidden_size
        self.pre_norms = nn.ModuleList([SandwichNorm(d) for _ in range(n_loop)])
        self.post_norms = nn.ModuleList([SandwichNorm(d) for _ in range(n_loop)])
        self.head = JointHead(d)
        self.stats = {}
        # Spec v1.2 sliced-CE: active-vocab slice of the FROZEN lm_head.
        # keep_ids remaps full-vocab ids -> 0..K-1; the full logits tensor is
        # never materialized in the training path (151,936 -> ~112 logits).
        self.register_buffer("keep_ids", None, persistent=False)
        self.keep_pos = None                   # full-id -> sliced-id lookup

    def set_keep_ids(self, keep_ids):
        """keep_ids: LongTensor of active vocab ids (sorted)."""
        dev = next(self.base.parameters()).device
        self.keep_ids = torch.as_tensor(np.asarray(keep_ids, dtype=np.int64), device=dev)
        V = self.base.config.vocab_size
        pos = torch.full((V,), -1, dtype=torch.long, device=dev)
        pos[self.keep_ids] = torch.arange(len(self.keep_ids), device=dev)
        self.keep_pos = pos

    # ---------------- parameter management ----------------
    def set_trainable(self, train_lm_head=False):
        for p in self.base.parameters():
            p.requires_grad = False
        for i in range(CORE_START, CORE_END):
            for p in self.base.model.layers[i].parameters():
                p.requires_grad = True
        if train_lm_head and not self.base.config.tie_word_embeddings:
            for p in self.base.lm_head.parameters():
                p.requires_grad = True
        for p in self.pre_norms.parameters():
            p.requires_grad = True
        for p in self.post_norms.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---------------- core forward ----------------
    def _run_layer(self, layer, h, pe, pos_ids, mask):
        if self.grad_ckpt and self.training:
            return tk_ckpt(lambda hh: call_layer(layer, hh, pe, pos_ids, mask), h,
                           use_reentrant=False)
        return call_layer(layer, h, pe, pos_ids, mask)

    def forward(self, ids, add_readout=True, lm_output=True, pad_mask=None,
                z_sample=0, ro_embed=None):
        """lm_output: True=full-vocab logits | 'sliced'=K-logit slice (needs
        set_keep_ids) | False=none. z_sample: if >0, ALSO compute the kept-mass
        ratio Z on a subsample of positions (with grad) -> out['z'].
        ro_embed: (N_READOUT, d) tensor replacing the model's readout
        embeddings (Spec v1.2 dual-branch Stage B / fold heads) - gradients
        flow into the CALLER's tensor. out['ro_h'] exposes the raw readout
        hidden states for external head stacks."""
        m = self.base.model
        B, L = ids.shape
        te = m.embed_tokens(ids)
        if add_readout:
            if ro_embed is None:
                ro = self.head.readout.weight.unsqueeze(0).expand(B, -1, -1).to(te.dtype)
            else:
                ro = ro_embed.unsqueeze(0).expand(B, -1, -1).to(te.dtype)
            h = torch.cat([te, ro], dim=1)
            T = L + N_READOUT
        else:
            h = te
            T = L
        pos_ids = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, -1)
        pe = None
        if hasattr(m, "rotary_emb"):
            pe = m.rotary_emb(h.float(), pos_ids)
        mask = None
        if pad_mask is not None:
            mask = build_joint_mask(pad_mask, T, L, h.dtype, h.device)
        elif self.explicit_mask:
            mask = build_causal_mask(T, h.dtype, h.device)

        def core_pass(hh):
            for i, layer in enumerate(m.layers):
                if i == CORE_START:
                    # In-place recursion: h_{r+1} = clamp(post_r(core(pre_r(h_r)))).
                    # With identity-init norms (g=0) and R=1 this is EXACTLY the
                    # pretrained forward (verified by the r=1 equivalence test).
                    for r in range(self.n_loop):
                        if self.sandwich:
                            z = self.pre_norms[r](hh)
                            for j in range(CORE_START, CORE_END):
                                z = self._run_layer(m.layers[j], z, pe, pos_ids, mask)
                            z = self.post_norms[r](z)
                            hh = clamp_resid(z, self.clamp, self.stats)
                        else:
                            for j in range(CORE_START, CORE_END):
                                hh = self._run_layer(m.layers[j], hh, pe, pos_ids, mask)
                    continue
                if CORE_START < i < CORE_END:
                    continue
                hh = self._run_layer(layer, hh, pe, pos_ids, mask)
            return hh

        if self.grad_ckpt and self.training:
            h = tk_ckpt(core_pass, h, use_reentrant=False)
        else:
            h = core_pass(h)
        h = m.norm(h)

        out = {}
        if lm_output is True:
            out["logits"] = self.base.lm_head(h[:, :L] if add_readout else h)
        elif lm_output == "sliced":
            # Spec v1.2: full logits (B, L, 151936) are NEVER materialized;
            # only the K active columns of the frozen lm_head are used.
            hs = h[:, :L] if add_readout else h
            W = self.base.lm_head.weight                     # (V, d), frozen
            Wk = W.index_select(0, self.keep_ids)            # (K, d)
            out["logits_sliced"] = torch.nn.functional.linear(hs, Wk)
        if add_readout:
            out["ro_h"] = h[:, -N_READOUT:, :]
            out["head"] = self.head(h[:, -N_READOUT:, :])
        if z_sample > 0 and self.keep_ids is not None and lm_output is not False:
            hs = h[:, :L] if add_readout else h
            n = hs.shape[0] * hs.shape[1]
            flat_idx = torch.randint(0, n, (min(z_sample, n),), device=hs.device)
            hsub = hs.reshape(-1, hs.shape[-1])[flat_idx]    # (S, d) w/ grad
            zlogits = torch.nn.functional.linear(hsub, self.base.lm_head.weight)  # (S, V)
            zprob = torch.softmax(zlogits.float(), dim=-1)
            out["z"] = zprob.index_select(1, self.keep_ids).sum(-1).mean()
        return out

    # ---------------- losses ----------------
    @staticmethod
    def lm_loss(logits, ids, pad_mask=None, chunk_rows=2048):
        """Memory-safe chunked CE. The 151,936-vocab logits for (B=16, T=512)
        are ~2.5GB in fp16; a single fp32 cast + CE would add ~7.5GB and OOM a
        T4. Each chunk's fp32 cast + CE is recomputed in backward via
        checkpoint, so peak fp32 memory is one chunk (~1.2GB)."""
        tgt = ids[:, 1:].clone()
        if pad_mask is not None:
            tgt[~pad_mask[:, 1:]] = -100
        lg = logits[:, :-1].reshape(-1, logits.size(-1))
        tg = tgt.reshape(-1)
        n_rows = lg.shape[0]
        total = torch.zeros((), device=lg.device, dtype=torch.float32)
        n = 0
        for s in range(0, n_rows, chunk_rows):
            sl = slice(s, min(s + chunk_rows, n_rows))
            t = tg[sl]
            cnt = int((t != -100).sum())
            if cnt == 0:
                continue
            def _ce(l, tt):
                return Fn.cross_entropy(l.float(), tt, ignore_index=-100,
                                        reduction="sum")
            total = total + tk_ckpt(_ce, lg[sl], t, use_reentrant=False)
            n += cnt
        return total / max(n, 1)

    def sliced_lm_loss(self, logits_sliced, ids):
        """Spec v1.2 sliced-CE: cross-entropy over the K active-vocab logits
        only (targets remapped full-id -> sliced-id; off-vocab targets are
        impossible by construction - the stream is built from keep_ids).
        logits_sliced: (B, L, K); ids: (B, L) full-vocab ids."""
        pos = self.keep_pos                       # (V,) full -> sliced (-1 if off)
        tgt = pos[ids[:, 1:]]                     # (B, L-1)
        lg = logits_sliced[:, :-1].reshape(-1, logits_sliced.size(-1))
        tg = tgt.reshape(-1)
        ok = tg >= 0
        if not ok.any():
            return torch.tensor(0.0, device=ids.device)
        return Fn.cross_entropy(lg[ok].float(), tg[ok])

    def l2sp_leash(self, init_sd, lam=0.01):
        """Weight-space leash to the frozen base (L2-SP): lam * mean squared
        drift of the unfrozen core params from their pretrained init values.
        init_sd: dict name -> tensor snapshot taken at training start."""
        num, den = 0.0, 0.0
        for n_, p in self.base.model.layers.named_parameters():
            if not p.requires_grad or n_ not in init_sd:
                continue
            d = (p.float() - init_sd[n_].to(p.device).float())
            num = num + d.pow(2).sum()
            den = den + init_sd[n_].to(p.device).float().pow(2).sum()
        if den <= 0:
            return torch.tensor(0.0)
        return lam * num / den

    @staticmethod
    def head_loss(head_logits, tgt):
        """tgt: dict of tensors on the same device.
        h1/h2: (B,48|8) in [0,1]; h3: (B,12) values; h4: (B,7) probs + h4c (B,) long,
        h4q (B,) value; h5: (B,) long 0..8. NaNs allowed -> masked out.
        Spec v1.2: h1 targets are TRIANGULAR-SMOOTHED realized outcomes
        (soft [0,1] - BCE-with-logits handles them); h3 uses PINBALL loss at
        its quantile levels (H3_QUANTILES) instead of smooth-L1."""
        losses = []
        for k in ("h1", "h2"):
            if tgt.get(k) is not None:
                t = tgt[k].float()
                lg = head_logits[k]
                msk = torch.isfinite(t)
                if msk.any():
                    losses.append(Fn.binary_cross_entropy_with_logits(
                        lg[msk], t[msk]))
        if tgt.get("h3") is not None:
            t = tgt["h3"].float()
            msk = torch.isfinite(t) & (t.abs() < 1e4)
            if msk.any():
                # Spec v1.2: pinball (quantile) loss. H3 layout (12):
                # [L48 MFE q25/q50/q75, L48 MAE q25/q50/q75,
                #  S48 MFE q25/q50/q75, S48 MAE q25/q50/q75]
                qs = torch.tensor([0.25, 0.5, 0.75] * 4, device=t.device,
                                  dtype=torch.float32)
                pred = Fn.softplus(head_logits["h3"])
                d = (t - pred).float()
                pin = torch.maximum(qs * d, (qs - 1.0) * d)
                losses.append((pin * msk.float()).sum() / msk.float().sum())
        if tgt.get("h4") is not None:
            t = tgt["h4"].float()
            msk = torch.isfinite(t)
            if msk.any():
                losses.append(Fn.binary_cross_entropy_with_logits(
                    head_logits["h4"][:, :4][msk], t[msk]))
        if tgt.get("h4c") is not None:
            t = tgt["h4c"].long()
            ok = (t >= 0) & (t < 3)
            if ok.any():
                losses.append(Fn.cross_entropy(head_logits["h4"][:, 4:7][ok], t[ok]))
        if tgt.get("h4q") is not None:
            t = tgt["h4q"].float()
            msk = torch.isfinite(t) & (t > 0) & (t < 1e4)
            if msk.any():
                losses.append(Fn.smooth_l1_loss(Fn.softplus(head_logits["h4"][:, 7])[msk], t[msk]))
        if tgt.get("h5") is not None:
            t = tgt["h5"].long()
            ok = (t >= 0) & (t < 9)
            if ok.any():
                losses.append(Fn.cross_entropy(head_logits["h5"][ok], t[ok]))
        if not losses:
            return torch.tensor(0.0, device=head_logits["h1"].device)
        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Training utilities (fp16-strict, T4/sm_75 safe)
# ---------------------------------------------------------------------------

def make_optimizer(model, lr=1e-4, wd=0.01, betas=(0.9, 0.95)):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)


def autocast_ctx(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


def load_base(device=None, dtype=torch.float32, log=print):
    from transformers import AutoModelForCausalLM
    try:
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=dtype)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=dtype)
    if device is not None:
        base = base.to(device)
    base.eval()
    log(f"  loaded {BASE_MODEL_ID} ({sum(p.numel() for p in base.parameters())/1e6:.1f}M params, "
        f"{len(base.model.layers)} layers, d={base.config.hidden_size}, "
        f"tied={base.config.tie_word_embeddings})")
    return base


def load_tokenizer(log=print):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    log(f"  tokenizer vocab={tok.vocab_size}")
    return tok


# ==================== kernel p3_tables ====================
# ============================================================================
# AURUM Phase 3 - PROBABILITY TABLES KERNEL (2xT4, ~1.5h) - Spec v1.2
#   Mounts : tskdev/aurum-phase1-data (v5) + tskdev/aurum-models-v1
#   Step 1 : re-serialize all labeled states (byte-identical pipeline to
#            p2-train v6: same tokenizer flags, same drop set, same order)
#   Step 2 : forward the trained looped core at depths r in {1,2,3,4}
#            (2 workers, one per GPU, round-robin state shards; fp16 autocast
#            no-grad, bs=32 - the exact recipe of the p2-train cache stage).
#            r=4 extrapolation: the 4th loop iteration reuses the LAST trained
#            sandwich norm (weight-tied cell reuse; training only saw r<=3 via
#            random loop supervision).
#   Step 3 : heads = main (trained <=2022) + 10 walk-forward fold heads
#            (per-year, causal); per-head temperature scaling (h1/h5).
#   Step 4 : OOD loop-divergence vs the r=3 reference on calibrated H1 probs.
#   Step 5 : era-neutralized predictions (per-year orthogonalization against
#            the p1-data era classifier score f(x)) -> neutralized EV argmax.
#   Step 6 : aurum_oos_prob_tables.parquet (one row per VALID state x depth x
#            variant{main,fold}: 85 calibrated head outputs + OOD + EV fields;
#            fold variant only has 2016-2025 states) + p3_cell_meta.json +
#            p3_tables_summary.json + gates.json
#   Validation: recomputed fold-calibration Brier must match the training-time
#            fold_results.json values (same rows, same metric) within 0.01.
# ============================================================================
P3_DEPTHS = [1, 2, 3, 4]
P3_TIMEOUT_S = 3.5 * 3600


# ---------------------------------------------------------------------------
# Shared data/serialization/heads section - VERBATIM from kernels/p2_train.py
# (any edit here MUST be mirrored in p2_train.py; byte-identical tokenization
#  is what makes the p3 tables comparable to the p2-train cache/refits)
# ---------------------------------------------------------------------------
import argparse, json, os, shutil, subprocess, sys, time
import numpy as np

TRAIN_END_YEAR = 2022              # hygiene: Stage A' + main Stage B pre-test only
FOLD_YEARS = list(range(2016, 2026))
CAL_FRAC = 0.2                     # purged calibration tail of each fold window


def _find_ds():
    """Locate the phase-1 dataset mount robustly (mount names can differ;
    logs the /kaggle/input tree when the expected path is missing)."""
    import glob as _glob
    cands = []
    if os.environ.get("AURUM_DS"):
        cands.append(os.environ["AURUM_DS"])
    cands += ["/kaggle/input/datasets/tskdev/aurum-phase1-data",
              "/kaggle/input/aurum-phase1-data"]
    cands += sorted(_glob.glob("/kaggle/input/datasets/*/aurum-phase1-data*"))
    cands += sorted(_glob.glob("/kaggle/input/*/*/aurum-phase1-data*"))
    cands += sorted(_glob.glob("/kaggle/input/*/aurum-phase1-data*"))
    cands += sorted(_glob.glob("/kaggle/input/aurum*"))
    for c in cands:
        if os.path.isfile(os.path.join(c, "h1_features.parquet")):
            return c
    try:
        import subprocess as _sp
        tree = _sp.run(["find", "/kaggle/input", "-maxdepth", "3"],
                       capture_output=True, text=True, timeout=20).stdout
        print("[_find_ds] DATASET NOT FOUND. /kaggle/input tree:\n" + tree[:3000],
              flush=True)
    except Exception:
        pass
    return cands[0]

def _ds():
    DS = "/kaggle/input/aurum-phase1-data"
    if os.path.isfile(os.path.join(DS, "h1_features.parquet")):
        return DS
    found = _find_ds()
    return found if found != DS else os.environ.get("AURUM_DS", DS)

def _trend_volc(h1):
    a = wilder_atr(h1["h"], h1["l"], h1["c"], 14)
    r24 = (h1["c"] - h1["c"].shift(24)) / (a + 1e-12)
    t33 = r24.rolling(252, min_periods=60).quantile(0.33).shift(1)
    t66 = r24.rolling(252, min_periods=60).quantile(0.66).shift(1)
    trend = np.where(r24 < t33, 0, np.where(r24 > t66, 2, 1)).astype(float)
    trend[~np.isfinite(t33.values) | ~np.isfinite(r24.values)] = np.nan
    ar = a / a.shift(48)
    v33 = ar.rolling(252, min_periods=60).quantile(0.33).shift(1)
    v66 = ar.rolling(252, min_periods=60).quantile(0.66).shift(1)
    volc = np.where(ar < v33, 0, np.where(ar > v66, 2, 1)).astype(float)
    volc[~np.isfinite(v33.values) | ~np.isfinite(ar.values)] = np.nan
    return trend, volc

def _ser_drop(DS):
    drop = {"m_dxy", "m_y10", "m_y2", "m_spread"}
    try:
        ss = json.load(open(os.path.join(DS, "state_stats.json")))
        drop |= set(ss.get("state_drop", []))
    except Exception:
        pass
    return drop

# ----------------------------------------------------------------------------
# data prep (parent): all labeled states tokenized once; ragged pools saved
# ----------------------------------------------------------------------------

def prepare_data(WORK):
    import pandas as pd
    cache = os.path.join(WORK, "p2_train_data.npz")
    meta_f = os.path.join(WORK, "p2_train_data_meta.json")
    if os.path.exists(cache) and os.path.exists(meta_f):
        log("p2 data: using cache")
        return cache, json.load(open(meta_f))

    DS = _ds()
    feat = pd.read_parquet(os.path.join(DS, "h1_features.parquet"))
    if "ts" in feat.columns:
        feat = feat.drop(columns=["ts"])
    h1 = pd.read_parquet(os.path.join(DS, "h1_xauusd.parquet"))
    h1 = h1[~h1.index.duplicated()].sort_index()
    ts = feat.index
    trend, volc = _trend_volc(h1)
    ser_drop = _ser_drop(DS)

    R = pd.read_parquet(os.path.join(DS, "labels_R_actual.parquet"))
    cls_ = pd.read_parquet(os.path.join(DS, "labels_cls_actual.parquet"))
    W_real = (R.values > 0).astype(np.float64)
    W_real[~np.isfinite(R.values)] = np.nan
    valid_R = np.isfinite(R.values).all(1) & (cls_.values != 255).all(1)
    cols = [c for c in COHORT_FEATURES if c in feat.columns]
    label_valid = valid_R.copy()
    label_valid &= feat[cols].notna().all(axis=1).values
    label_valid &= np.isfinite(feat[cols].values).all(1)
    label_valid[:400] = False

    with stage("serialize+tokenize (all labeled states)"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        idx_all = np.flatnonzero(label_valid)
        B = 2000
        ids_concat, lens = [], []
        for s in range(0, len(idx_all), B):
            rows = idx_all[s:s + B]
            t_batch = [serialize_state(ts[i], feat.iloc[i], with_cross=True,
                                       with_macro=True, with_priors=False,
                                       with_minnov=True, with_zpriors=True,
                                       with_cohort=True, trend3=trend[i],
                                       vol3=volc[i], drop=ser_drop, compact=True)
                       for i in rows]
            enc = tok(t_batch, add_special_tokens=False)["input_ids"]
            for e_ in enc:
                ids_concat.append(np.asarray(e_, dtype=np.int32))
                lens.append(len(e_))
        lens = np.array(lens, dtype=np.int32)
        lab_flat = np.concatenate(ids_concat)
        years = np.asarray(ts[idx_all].year)

        # Stage A' stream: contiguous pre-test states (includes unlabeled bars
        # with valid serialization? spec: 'states' - use labeled pre-test set
        # + head-invisible unlabeled is unnecessary; labeled covers 2012-2022)
        pre = years <= TRAIN_END_YEAR
        ends = np.cumsum(lens); starts = ends - lens
        pre_flat = np.concatenate([lab_flat[starts[i]:ends[i]]
                                   for i in np.flatnonzero(pre)])
        meta = {"n_labeled_states": int(len(idx_all)),
                "n_pretest": int(pre.sum()), "n_oos": int((~pre).sum()),
                "pretest_tokens": int(pre_flat.size),
                "mean_tokens": round(float(lens.mean()), 1),
                "min_tokens": int(lens.min()), "max_tokens": int(lens.max())}
        log(f"p2 data: {json.dumps(meta)}")

        sm = pd.read_parquet(os.path.join(DS, "labels_cls_smoothed.parquet")).values
        aux = pd.read_parquet(os.path.join(DS, "labels_aux.parquet"))
        pw = pd.read_parquet(os.path.join(DS, "labels_pwin.parquet")).values
        try:
            exc = pd.read_parquet(os.path.join(DS, "labels_excursions.parquet"))
            h3 = np.repeat(exc[["mfe_L48", "mae_L48", "mfe_S48", "mae_S48"]].values, 3, 1)
        except Exception:
            q = lambda pre_, suf_: np.stack([aux[f"{pre_}{q_}{suf_}"].values
                                             for q_ in (25, 50, 75)], 1)
            h3 = np.concatenate([q("mfe_q", "_48"), q("mae_q", "_48"),
                                 q("mfe_q", "_48S"), q("mae_q", "_48S")], 1)
        np.savez_compressed(
            cache, stream_flat=pre_flat.astype(np.int32),
            lab_flat=lab_flat.astype(np.int32), lab_lens=lens,
            lab_idx=idx_all, lab_years=years.astype(np.int16),
            sm_h1=sm.astype(np.float32), pwin=pw.astype(np.float32),
            win_real=W_real.astype(np.float32), R_real=R.values.astype(np.float32),
            h2up=aux[["h2up4","h2up12","h2up24","h2up48"]].values.astype(np.float32),
            h2big=aux[["h2big4","h2big12","h2big24","h2big48"]].values.astype(np.float32),
            h4exp=aux[["h4exp4","h4exp12","h4exp24","h4exp48"]].values.astype(np.float32),
            h4terc=np.nan_to_num(aux["h4terc12"].values.astype(float), nan=-1).astype(np.int64),
            h4q=np.nan_to_num(aux["h4q50_12"].values.astype(float), nan=-1).astype(np.float32),
            h5=np.nan_to_num(aux["h5regime"].values.astype(float), nan=-1).astype(np.int64),
            h3=h3.astype(np.float32))
    with open(meta_f, "w") as f:
        json.dump(meta, f, indent=1)
    return cache, meta

# ----------------------------------------------------------------------------
# head-stack helpers (dual-branch Stage B + fold refits share these)
# ----------------------------------------------------------------------------

def _mk_head_module(model, dropout):
    """Fresh readout+heads stack with variance discipline, on the same device
    (core shared, NOT part of this module)."""
    import torch.nn as nn
    d = model.base.config.hidden_size
    class HeadStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.readout = nn.Embedding(5, d)
            nn.init.normal_(self.readout.weight, std=0.02)
            def mlp(out):
                return nn.Sequential(nn.Linear(d, 256), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(256, out))
            self.h1, self.h2, self.h3 = mlp(48), mlp(8), mlp(12)
            self.h4, self.h5 = mlp(8), mlp(9)
        def forward(self, ro_hidden):
            return {"h1": self.h1(ro_hidden[:, 0]), "h2": self.h2(ro_hidden[:, 1]),
                    "h3": self.h3(ro_hidden[:, 2]), "h4": self.h4(ro_hidden[:, 3]),
                    "h5": self.h5(ro_hidden[:, 4])}
    return HeadStack()




def _find_models_ds():
    """Locate the aurum-models-v1 dataset mount robustly."""
    import glob as _glob
    cands = []
    if os.environ.get("AURUM_MODELS"):
        cands.append(os.environ["AURUM_MODELS"])
    cands += ["/kaggle/input/datasets/tskdev/aurum-models-v1",
              "/kaggle/input/aurum-models-v1"]
    cands += sorted(_glob.glob("/kaggle/input/datasets/*/aurum-models-v1*"))
    cands += sorted(_glob.glob("/kaggle/input/*/aurum-models-v1*"))
    cands += sorted(_glob.glob("/kaggle/input/*aurum-models*"))
    for c in cands:
        if os.path.isfile(os.path.join(c, "core_fp16.pt")):
            return c
    try:
        import subprocess as _sp
        tree = _sp.run(["find", "/kaggle/input", "-maxdepth", "3"],
                       capture_output=True, text=True, timeout=20).stdout
        print("[_find_models_ds] DATASET NOT FOUND. /kaggle/input tree:\n"
              + tree[:3000], flush=True)
    except Exception:
        pass
    return None


def build_core(base, core_sd, depth, log=print):
    """LoopedQwen at inference depth `depth` with trained fp16 weights.
    Trained sandwich norms cover iterations 0..n_trained-1 (n_loop_train=3);
    iterations beyond that (r=4 extrapolation) reuse the LAST trained norm."""
    model = LoopedQwen(base, n_loop=depth, sandwich=True, grad_ckpt=False)
    for i in range(CORE_START, CORE_END):
        sd_i = {k: v.float() for k, v in core_sd["layers"][str(i)].items()}
        model.base.model.layers[i].load_state_dict(sd_i)
    pre_sd = {k: v.float() for k, v in core_sd["pre"].items()}
    post_sd = {k: v.float() for k, v in core_sd["post"].items()}
    n_trained = 1 + max(int(k.split(".")[0]) for k in pre_sd)
    model.pre_norms.load_state_dict(pre_sd, strict=False)
    model.post_norms.load_state_dict(post_sd, strict=False)
    if depth > n_trained:
        for r in range(n_trained, depth):
            model.pre_norms[r].load_state_dict(
                model.pre_norms[n_trained - 1].state_dict())
            model.post_norms[r].load_state_dict(
                model.post_norms[n_trained - 1].state_dict())
        log(f"  r={depth} extrapolation: iterations {n_trained}..{depth-1} "
            f"reuse trained norm index {n_trained-1}")
    model.eval()
    return model


def _p3_worker(rank, world, WORK, MODELS, depths):
    """One GPU's shard of the depth sweep (round-robin, cache-stage recipe)."""
    import torch
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and world > 1:
        dev = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    z = np.load(os.path.join(WORK, "p2_train_data.npz"))
    lab_flat, lens = z["lab_flat"], z["lab_lens"]
    N = len(lens)
    ends = np.cumsum(lens)
    starts = ends - lens
    my_rows = np.arange(rank, N, world)
    core_sd = torch.load(os.path.join(MODELS, "core_fp16.pt"), map_location="cpu",
                         weights_only=False)
    ro_w = core_sd["readout"]["readout.weight"].float().to(dev)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    base = load_base(device=dev, dtype=torch.float32,
                     log=lambda m: print(f"  [w{rank}] {m}", flush=True))
    for depth in depths:
        out_f = os.path.join(WORK, f"roh_r{depth}_part{rank}.npy")
        if os.path.exists(out_f):
            continue
        model = build_core(base, core_sd, depth)
        model.to(dev)
        H5 = np.zeros((len(my_rows), 5, model.base.config.hidden_size),
                      dtype=np.float16)
        t0 = time.time()
        bs_c = 32
        with torch.no_grad():
            for s0 in range(0, len(my_rows), bs_c):
                rows = my_rows[s0: s0 + bs_c]
                seqs = [lab_flat[starts[i]:ends[i]] for i in rows]
                L = max(len(s_) for s_ in seqs)
                # Build the padded batch in numpy, convert once (numpy int32 ->
                # torch LongTensor slice-assign raises TypeError on Kaggle's
                # numpy-2 stack; numpy->numpy is also faster per row).
                arr = np.full((len(seqs), L), pad_id, dtype=np.int64)
                msk = np.zeros((len(seqs), L), dtype=bool)
                for i, s_ in enumerate(seqs):
                    arr[i, :len(s_)] = s_
                    msk[i, :len(s_)] = True
                ids = torch.from_numpy(arr)
                pm = torch.from_numpy(msk)
                with autocast_ctx(dev):
                    o = model(ids.to(dev), add_readout=True, lm_output=False,
                              pad_mask=pm.to(dev), ro_embed=ro_w)
                H5[s0:s0 + len(rows)] = o["ro_h"].half().cpu().numpy()
                if rank == 0 and s0 % 6400 == 0:
                    rps = (s0 + len(rows)) * world / max(time.time() - t0, 1e-9)
                    print(f"  [w0] r{depth}: ~{s0 * world}/{N} "
                          f"({rps:.1f} rows/s agg)", flush=True)
        np.save(out_f, H5)
        print(f"  [w{rank}] depth r={depth} DONE {len(my_rows)} rows "
              f"in {time.time()-t0:.0f}s", flush=True)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()


def _p3_forward_all(depths, WORK, MODELS):
    """Forward all labeled states at each depth. Two subprocess workers (one
    per GPU) when >=2 CUDA devices; in-process single path otherwise."""
    import torch
    z = np.load(os.path.join(WORK, "p2_train_data.npz"))
    N = len(z["lab_lens"])
    world = max(1, torch.cuda.device_count())
    todo = [d for d in depths
            if not os.path.exists(os.path.join(WORK, f"roh_r{d}_part0.npy"))]
    if todo and world > 1:
        todo = [d for d in depths
                if not all(os.path.exists(os.path.join(WORK, f"roh_r{d}_part{rk}.npy"))
                           for rk in range(world))]
    if todo:
        if world > 1:
            import subprocess as sp
            env = dict(os.environ)
            env.update(AURUM_WORK=WORK, AURUM_MODELS=MODELS)
            procs, lfs = [], []
            for rk in range(world):
                lf = open(os.path.join(WORK, f"p3w{rk}.log"), "w")
                lfs.append(lf)
                procs.append(sp.Popen(
                    [sys.executable, os.path.abspath(__file__), "--mode", "worker",
                     "--rank", str(rk), "--world", str(world)],
                    env=env, stdout=lf, stderr=sp.STDOUT))
            t0 = time.time()
            while any(p.poll() is None for p in procs):
                if time.time() - t0 > P3_TIMEOUT_S:
                    for p in procs:
                        p.kill()
                    raise RuntimeError("p3 forward workers timed out")
                time.sleep(20)
            for i, (p, lf) in enumerate(zip(procs, lfs)):
                lf.close()
                if p.returncode != 0:
                    tail = open(os.path.join(WORK, f"p3w{i}.log")).read()[-2000:]
                    raise RuntimeError(f"worker {i} failed (rc={p.returncode}):\n{tail}")
        else:
            _p3_worker(0, 1, WORK, MODELS, todo)
    roh = {}
    for d in depths:
        if world > 1:
            parts = [np.load(os.path.join(WORK, f"roh_r{d}_part{rk}.npy"))
                     for rk in range(world)]
            full = np.zeros((N, *parts[0].shape[1:]), dtype=parts[0].dtype)
            for rk in range(world):
                full[rk::world] = parts[rk]
            roh[d] = full
        else:
            roh[d] = np.load(os.path.join(WORK, f"roh_r{d}_part0.npy"))
    return roh


class _Shim:
    """Duck-types LoopedQwen for _mk_head_module (needs .base.config.hidden_size)."""
    def __init__(self, d):
        import types
        self.base = types.SimpleNamespace(
            config=types.SimpleNamespace(hidden_size=d))


def _head_stack(d):
    return _mk_head_module(_Shim(d), 0.0)


def _head_activate(logits, T_h1=1.0, T_h5=1.0):
    """Training-time activations + temperature scaling (h1/h5 only)."""
    import torch
    import torch.nn.functional as Fn
    return {
        "h1": torch.sigmoid(logits["h1"] / T_h1),
        "h2": torch.sigmoid(logits["h2"]),
        "h3": Fn.softplus(logits["h3"]),
        "h4": torch.cat([torch.sigmoid(logits["h4"][:, :4]),
                         torch.softmax(logits["h4"][:, 4:7], dim=-1),
                         Fn.softplus(logits["h4"][:, 7:8])], dim=1),
        "h5": torch.softmax(logits["h5"] / T_h5, dim=-1),
    }


def _neutralize_per_year(P, f, years):
    """Per-year linear orthogonalization of each column of P against the era
    score f (exp3d protocol / neutralization_spec.json)."""
    Pn = P.copy()
    for yv in np.unique(years):
        m = years == yv
        if m.sum() < 30 or not np.isfinite(f[m]).all():
            continue
        ff = f[m]
        df_ = ff - ff.mean()
        var = float((df_ * df_).sum()) + 1e-12
        beta = (df_ @ (P[m] - P[m].mean(0))) / var          # (48,)
        Pn[m] = P[m] - np.outer(df_, beta)
    return np.clip(Pn, 0.0, 1.0)


def _ev_fields(P, b_geo):
    """EV per cell (R units): p*b - (1-p); argmax cell + its p/EV."""
    ev = P * b_geo[None, :] - (1.0 - P)
    cell = ev.argmax(1)
    n = np.arange(len(cell))
    return cell, P[n, cell], ev[n, cell]


# ============================================================================
# LIVE DAILY (Step 6a) — Spec v1.3 decision layer.  CPU-only.
#
# v1.2 -> v1.3: the decision is now the *exact* flagship policy fitted in
# aurum-p4-backtest v1.3 (policy_fit.json) on walk-forward fold years
# 2016-2022 ONLY (zero 2023-2025 peeking):
#   - forward at ALL FOUR depths r=1..4 (pool uses 1:3; r=4 for OOD);
#   - multi-depth pooled posterior  p_bar = .6 p1 + .2 p2 + .2 p3   (F6);
#   - sigma_loop = population std of (p1,p2,p3) per cell;
#   - shrinkage toward the causal base-rate prior (kappa, F3; 0 here);
#   - LCB-EV = p_sh*b - (1-p_sh) - lambda_var * b * sigma_loop - cost
#     (lambda_var=0.5, F3; cost = 0.05/stop_atr_mult);
#   - EV_min hurdle (F3; 0.0 — the LCB penalty supplies the effective
#     hurdle), GBDT directional agreement gate (fold model, train <= y-1),
#     NFP/VIX blackouts, loop-OOD gate;
#   - exit plan = "path" overlay (F4): half-position partial TP limit at
#     MFE-q50*ATR, tightened trail at MAE-q25*ATR, hard stop m*ATR,
#     TP t*ATR, time stop h bars — all orders placed at entry (causal);
#   - quarter-Kelly risk, 1.0% hard equity cap (cap=1 sizing; the cap=3
#     portfolio variant scales it by 1/3 per position).
# Reference account for sizing: 10,000 USD, 1.0% hard equity risk cap.
# ============================================================================

LIVE_DEPTHS     = [1, 2, 3, 4]      # forward depths (pool 1:3; 4 for OOD)
LIVE_POOL_W     = [0.6, 0.2, 0.2]   # F6 perf fit over (r1, r2, r3)
LIVE_KAPPA      = 0.0               # F3 perf (F2 Brier value was 0.4)
LIVE_LAMBDA_VAR = 0.5               # F3
LIVE_EV_MIN     = 0.0               # F3 (LCB penalty supplies the hurdle)
LIVE_COST_BASE_R = 0.05             # per-cell cost = 0.05 / stop_atr_mult
LIVE_OOD_TH     = 0.10
LIVE_OVERLAY_RULE = "path"          # F4 (fit ann-Sharpe 1.170 vs 0.896 clean)
KELLY_FRACTION_LIVE = 0.25
HARD_EQUITY_RISK_CAP_LIVE = 0.01
REF_ACCOUNT_USD = 10000.0
LIVE_POLICY_SRC = ("aurum-p4-backtest v1.3 policy_fit.json; "
                    "fit window 2016-2022; OOS 2023-25 never touched")


def _live_last_bar(DS):
    feat = pd.read_parquet(os.path.join(DS, "h1_features.parquet"))
    if "ts" in feat.columns:
        feat = feat.drop(columns=["ts"])
    h1 = pd.read_parquet(os.path.join(DS, "h1_xauusd.parquet"))
    h1 = h1[~h1.index.duplicated()].sort_index()
    trend, volc = _trend_volc(h1)
    ser_drop = _ser_drop(DS)
    i = len(feat) - 1
    # walk back to the last bar with a fully finite feature row
    fv = feat.values
    while i > 0 and not np.isfinite(fv[i]).all():
        i -= 1
    return feat, h1, trend, volc, ser_drop, i


def _live_era_score(DS, feat, i):
    """era_clf OOF-score at bar i (era-regime reference for the audit)."""
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier()
        clf.load_model(os.path.join(DS, "era_clf.json"))
        spec = json.load(open(os.path.join(DS, "neutralization_spec.json")))
        cols = [c for c in spec["features"] if c in feat.columns]
        X = feat[cols].iloc[[i]].values
        if not np.isfinite(X).all():
            return None
        return float(clf.predict_proba(X)[0, 1])
    except Exception as e:
        log(f"  era score unavailable: {e}")
        return None


def _live_blackout(feat, i, ts):
    v = feat["mi_vix_d20z"].values[i]
    shock = np.isfinite(v) and abs(v) >= 2.5
    dt = pd.Timestamp(ts)
    wd = pd.Timestamp(year=dt.year, month=dt.month, day=1).dayofweek
    ff = 1 + (4 - wd) % 7
    nfp = (dt.day == ff) and (dt.hour in (12, 13, 14))
    return {"nfp_window": bool(nfp), "vix_shock": bool(shock),
            "mi_vix_d20z": float(v) if np.isfinite(v) else None}


def _forward_state(model, ids, ro_w, dev, heads, T_h1, T_h5):
    import torch
    with torch.no_grad(), autocast_ctx(dev):
        o = model(ids.to(dev), add_readout=True, lm_output=False,
                  pad_mask=None, ro_embed=ro_w)
        H = o["ro_h"].float()
        A = _head_activate(heads(H), T_h1, T_h5)
        return {k: v[0].float().cpu().numpy() for k, v in A.items()}


def _live_base_rate_prior(cls, year_all, y_live):
    """Expanding per-cell base rate over rows with year <= y_live-1 (causal).
    Matches p4_backtest base_rate_priors: mean of (cls>0) per cell."""
    m = year_all <= (y_live - 1)
    W = (cls.values > 0).astype(np.float64)
    fin = np.isfinite(cls.values)
    Wf = np.where(fin, W, np.nan)
    with np.errstate(invalid="ignore"):
        pi = np.nanmean(Wf[m], axis=0)
    return pi


def _live_overlay_quantiles(d, h_, h3v, exc, y_live):
    """Overlay levels for the selected cell (ATR units).
    h=48 -> H3-head predictions (partial TP = MFE q50, trail = MAE q25);
    h=24 -> empirical quantiles over year <= y_live-1 (the h3 head only
    predicts 48h quantiles).  Matches p4 empirical_overlay_q (masked)."""
    if h_ == 48:
        q50 = float(h3v[1] if d == "L" else h3v[7])   # MFE q50
        q25 = float(h3v[3] if d == "L" else h3v[9])   # MAE q25
        return q50, q25, "h3_head_48h"
    m = (pd.DatetimeIndex(exc.index).year.values <= (y_live - 1))
    v50 = exc[f"mfe_{d}24"].values[m]
    v25 = exc[f"mae_{d}24"].values[m]
    q50 = float(np.nanquantile(v50, 0.50))
    q25 = float(np.nanquantile(v25, 0.25))
    return q50, q25, f"empirical_le{y_live - 1}_h24"


def _gbdt_live_fold(feat, cls, i, y_live, log):
    """GBDT fold model for the year containing the live bar, exactly as
    p4_backtest gbdt_walkforward (era-safe spec features, canonical cells,
    15% tail validation split, XGB params, seed 1337, train <= y-1)."""
    from xgboost import XGBClassifier
    spec = json.load(open(os.path.join(_ds(), "neutralization_spec.json")))
    spec_cols = [c for c in spec["features"] if c in feat.columns]
    X_all = feat[spec_cols].values.astype(np.float32)
    names = cell_names()
    ci_L = names.index("L_m1.0_t1.5_h48")
    ci_S = names.index("S_m1.0_t1.5_h48")
    yL_all = cls.values[:, ci_L]
    yS_all = cls.values[:, ci_S]
    year_all = pd.DatetimeIndex(feat.index).year.values.astype(np.int16)
    assert len(feat) == len(cls), "feat/cls row misalignment"
    finite = np.isfinite(X_all).all(1)
    lab_rows = np.flatnonzero(finite & (np.isfinite(yL_all) | np.isfinite(yS_all)))
    tr = lab_rows[(year_all[lab_rows] <= y_live - 1)]
    n_val = max(200, int(len(tr) * 0.15))
    tr_core, val = tr[:-n_val], tr[-n_val:]
    out = {}
    audit = []
    for tag, ycol in (("L", yL_all), ("S", yS_all)):
        m_tr = np.isfinite(ycol[tr_core])
        m_va = np.isfinite(ycol[val])
        xtr = X_all[tr_core[m_tr]]
        ytr = (ycol[tr_core[m_tr]] > 0).astype(int)
        xva = X_all[val[m_va]]
        yva = (ycol[val[m_va]] > 0).astype(int)
        clf = None
        try:
            clf = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                reg_lambda=10.0, subsample=0.7, colsample_bytree=0.7,
                min_child_weight=20, tree_method="hist", n_jobs=4,
                eval_metric="logloss", early_stopping_rounds=30,
                random_state=1337)
            clf.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)
        except TypeError:
            from xgboost import callback as _xgbc
            clf = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                reg_lambda=10.0, subsample=0.7, colsample_bytree=0.7,
                min_child_weight=20, tree_method="hist", n_jobs=4,
                eval_metric="logloss", random_state=1337)
            clf.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False,
                    callbacks=[_xgbc.EarlyStopping(rounds=30, save_best=True)])
        out["p" + tag] = float(clf.predict_proba(X_all[[i]])[0, 1])
        audit.append({"dir": tag, "n_train": int(m_tr.sum()),
                      "best_iter": int(clf.best_iteration or 300),
                      "max_train_year": int(year_all[tr].max())})
    out["audit"] = audit
    out["n_features"] = len(spec_cols)
    return out


def live_main(WORK):
    import torch
    t0 = time.time()
    log("ENV: " + json.dumps(env_report()))
    log("LIVE POLICY v1.3: " + LIVE_POLICY_SRC)
    MODELS = _find_models_ds()
    DS = _ds()
    ok = MODELS is not None and os.path.isfile(
        os.path.join(DS, "h1_features.parquet"))
    gate("G6.0_mounts", ok,
         f"phase1-data at {DS} "
         f"({'OK' if os.path.isfile(os.path.join(DS, 'h1_features.parquet')) else 'MISSING'}); "
         f"aurum-models-v1 at {MODELS} ({'OK' if MODELS else 'MISSING'}); "
         f"policy constants from {LIVE_POLICY_SRC}")
    if not ok:
        save_gates(os.path.join(WORK, "gates.json"))
        return

    # ---------- labels / excursions (GBDT targets + overlay + prior) -------
    cls = pd.read_parquet(os.path.join(DS, "labels_cls_actual.parquet"))
    exc = pd.read_parquet(os.path.join(DS, "labels_excursions.parquet"))

    # ---------------- latest bar + state serialization ----------------------
    feat, h1, trend, volc, ser_drop, i = _live_last_bar(DS)
    ts = feat.index[i]
    y_live = int(pd.Timestamp(ts).year)
    log(f"latest fully-featured bar: {ts} (row {i}/{len(feat)}); live fold year {y_live}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    state_txt = serialize_state(feat.index[i], feat.iloc[i], with_cross=True,
                                with_macro=True, with_priors=False,
                                with_minnov=True, with_zpriors=True,
                                with_cohort=True, trend3=trend[i],
                                vol3=volc[i], drop=ser_drop, compact=True)
    enc = np.asarray(tok(state_txt, add_special_tokens=False)["input_ids"],
                     dtype=np.int64)
    gate("G6.1_state", 280 <= len(enc) <= 380,
         f"serialized latest state: {len(enc)} tokens (expect 280-380)")
    with open(os.path.join(WORK, "live_state.txt"), "w") as f:
        f.write(state_txt)

    # ---------------- model forward at ALL FOUR depths ----------------------
    core_sd = torch.load(os.path.join(MODELS, "core_fp16.pt"),
                         map_location="cpu", weights_only=False)
    ro_w = core_sd["readout"]["readout.weight"].float()
    d_model = ro_w.shape[1]
    heads = _head_stack(d_model)
    heads.load_state_dict(torch.load(
        os.path.join(MODELS, "heads_main.pt"), map_location="cpu",
        weights_only=False))
    heads.eval()
    temps = json.load(open(os.path.join(MODELS, "temperatures_main.json")))
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"compute device: {dev.type} (CPU-only by design)")
    base = load_base(device=dev, dtype=torch.float32, log=log)
    ids = torch.from_numpy(enc)[None, :]
    out = {}
    for depth in LIVE_DEPTHS:
        model = build_core(base, core_sd, depth, log=log)
        model.to(dev)
        out[depth] = _forward_state(model, ids, ro_w.to(dev), dev,
                                    heads.to(dev),
                                    temps.get("h1", 1.0), temps.get("h5", 1.0))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        log(f"  forward r={depth}: h1 shape {out[depth]['h1'].shape}; "
            f"finite={bool(np.isfinite(out[depth]['h1']).all())}")
    fin = all(np.isfinite(v).all() for d in out.values() for v in d.values())
    gate("G6.2_forward", fin and sorted(out.keys()) == LIVE_DEPTHS,
         f"forwards r={LIVE_DEPTHS} on {dev.type}; all heads finite={fin}; "
         f"h1 shape {out[3]['h1'].shape}; temperatures {temps}")

    # ---------------- v1.3 decision policy ----------------------------------
    P = {r: np.clip(out[r]["h1"], 0.0, 1.0) for r in LIVE_DEPTHS}
    ood_mean = float(np.mean([np.abs(P[r] - P[3]).mean()
                              for r in LIVE_DEPTHS if r != 3]))
    ood_max = float(np.max([np.abs(P[r] - P[3]).max()
                            for r in LIVE_DEPTHS if r != 3]))
    w = np.asarray(LIVE_POOL_W, np.float64)
    p_bar = w[0] * P[1] + w[1] * P[2] + w[2] * P[3]
    m_pool = (P[1] + P[2] + P[3]) / 3.0
    sig_loop = np.sqrt(((P[1] - m_pool) ** 2 + (P[2] - m_pool) ** 2
                        + (P[3] - m_pool) ** 2) / 3.0)
    dirs_ax, mss, tps, hrs = cell_axes()
    b_geo = np.array([t / m for t, m in zip(tps, mss)])
    cost = np.asarray(LIVE_COST_BASE_R, np.float64) / np.asarray(mss, np.float64)
    year_all = pd.DatetimeIndex(feat.index).year.values.astype(np.int16)
    assert len(feat) == len(cls), "feat/cls row misalignment"
    pi48 = _live_base_rate_prior(cls, year_all, y_live)
    s = LIVE_KAPPA * b_geo
    p_sh = (p_bar + s * pi48) / (1.0 + s)
    EV_lcb = (p_sh * b_geo - (1.0 - p_sh)
              - LIVE_LAMBDA_VAR * b_geo * sig_loop - cost)
    cell = int(np.nanargmax(np.where(np.isfinite(EV_lcb), EV_lcb, -1e9)))
    d_cell = dirs_ax[cell]
    m_cell, t_cell, h_cell = mss[cell], tps[cell], hrs[cell]
    b_cell = float(b_geo[cell])
    cost_cell = float(cost[cell])
    p_bar_cell = float(p_bar[cell])
    p_sh_cell = float(p_sh[cell])
    sig_cell = float(sig_loop[cell])
    pi_cell = float(pi48[cell])
    ev_cell = float(EV_lcb[cell])

    # GBDT agreement (fold model for the live year, train <= y-1)
    gb = _gbdt_live_fold(feat, cls, i, y_live, log)
    pL, pS = gb["pL"], gb["pS"]
    gbdt_dir = "L" if pL >= pS else "S"
    agree = bool(gbdt_dir == d_cell)
    log(f"  GBDT fold {y_live}: train<= {y_live - 1} "
        f"(L {gb['audit'][0]['n_train']:,} / S {gb['audit'][1]['n_train']:,} rows, "
        f"{gb['n_features']} era-safe features); pL={pL:.4f} pS={pS:.4f} -> {gbdt_dir}")

    # overlay levels for the selected cell
    q50_atr, q25_atr, ov_src = _live_overlay_quantiles(
        d_cell, h_cell, out[3]["h3"], exc, y_live)

    # gates + risk sizing
    blackout = _live_blackout(feat, i, ts)
    era = _live_era_score(DS, feat, i)
    atr = float(wilder_atr(h1["h"], h1["l"], h1["c"], 14).values[i])
    c_usd = float(h1["c"].values[i])
    kelly = (p_sh_cell * b_cell - (1 - p_sh_cell)) / b_cell
    f_risk_cap1 = float(np.clip(KELLY_FRACTION_LIVE * kelly, 0.0,
                                HARD_EQUITY_RISK_CAP_LIVE))
    f_risk_cap3 = float(np.clip(KELLY_FRACTION_LIVE * kelly, 0.0,
                                HARD_EQUITY_RISK_CAP_LIVE / 3.0))
    ev_ok = bool(ev_cell >= LIVE_EV_MIN)
    prob_ok = bool(0.01 < p_bar_cell < 0.99)
    ood_ok = bool(ood_mean <= LIVE_OOD_TH)
    blackout_clear = not blackout["nfp_window"] and not blackout["vix_shock"]
    ov_ok = bool(np.isfinite(q50_atr) and np.isfinite(q25_atr) and q50_atr > 0)
    trade_ok = bool(ev_ok and prob_ok and ood_ok and blackout_clear and agree
                    and f_risk_cap1 > 0 and np.isfinite(atr) and atr > 0 and ov_ok)

    reason = None
    if not trade_ok:
        why = []
        if not ev_ok:
            why.append(f"EV_LCB<{LIVE_EV_MIN} ({ev_cell:+.4f})")
        if not prob_ok:
            why.append(f"p_bar out of (0.01,0.99) ({p_bar_cell:.4f})")
        if not ood_ok:
            why.append(f"loop-OOD>{LIVE_OOD_TH} ({ood_mean:.4f})")
        if not agree:
            why.append(f"GBDT disagreement (cell {d_cell} vs GBDT {gbdt_dir})")
        if blackout["nfp_window"]:
            why.append("NFP blackout window")
        if blackout["vix_shock"]:
            why.append("VIX-innovation shock")
        if not ov_ok:
            why.append("overlay levels invalid")
        if f_risk_cap1 <= 0:
            why.append("Kelly<=0")
        reason = "; ".join(why) or "unknown"
    gate("G6.3_calibration",
         (0.0 <= p_bar_cell <= 1.0) and (0.0 <= p_sh_cell <= 1.0)
         and ood_mean >= 0.0 and float(P[3].min()) >= 0.0
         and float(P[3].max()) <= 1.0,
         f"temperature scaling applied {temps}; pooled p_bar in [0,1]; "
         f"prod p_bar={p_bar_cell:.4f}; shrunk p={p_sh_cell:.4f}; "
         f"loop-OOD mean={ood_mean:.4f} max={ood_max:.4f}")

    # internal consistency of the policy formulas (G6.4b)
    ev_re = (p_sh_cell * b_cell - (1 - p_sh_cell)
             - LIVE_LAMBDA_VAR * b_cell * sig_cell - cost_cell)
    pbar_re = float(w[0] * P[1][cell] + w[1] * P[2][cell] + w[2] * P[3][cell])
    sig_re = float(np.std(np.array([P[1][cell], P[2][cell], P[3][cell]])))
    ood_re = float(np.mean([np.abs(P[r] - P[3]).mean()
                            for r in (1, 2, 4)]))
    cons = (abs(ev_re - ev_cell) < 1e-6 and abs(pbar_re - p_bar_cell) < 1e-6
            and abs(sig_re - sig_cell) < 1e-6 and abs(ood_re - ood_mean) < 1e-6)
    gate("G6.4b_policy_consistency", cons,
         f"EV_LCB reproduced from components (d={abs(ev_re - ev_cell):.2e}); "
         f"p_bar pooling reproduced (d={abs(pbar_re - p_bar_cell):.2e}); "
         f"sigma_loop population-std reproduced (d={abs(sig_re - sig_cell):.2e}); "
         f"ood reproduced (d={abs(ood_re - ood_mean):.2e})")
    gate("G6.4_decision", trade_ok or reason is not None,
         f"action={'TRADE' if trade_ok else 'NO-TRADE'}; "
         f"cell={cell_names()[cell]}; p_bar={p_bar_cell:.4f}; p_sh={p_sh_cell:.4f}; "
         f"EV_LCB={ev_cell:+.4f}R (>= {LIVE_EV_MIN}); sigma_loop={sig_cell:.4f}; "
         f"lambda_var*b*sig={LIVE_LAMBDA_VAR * b_cell * sig_cell:.4f}; "
         f"cost={cost_cell:.4f}R; GBDT {gbdt_dir} (pL={pL:.3f}/pS={pS:.3f}) "
         f"agrees={agree}; f_risk_cap1={f_risk_cap1:.5f} "
         f"(cap {HARD_EQUITY_RISK_CAP_LIVE}); "
         f"{'reason=' + reason if reason else 'all gates clear'}")

    # ---------------- heads detail + decision + audit -----------------------
    h5v = out[3]["h5"]
    h3v = out[3]["h3"]
    regime_names = ["trend_dn/low", "trend_dn/mid", "trend_dn/high",
                    "range/low", "range/mid", "range/high",
                    "trend_up/low", "trend_up/mid", "trend_up/high"]
    stop_usd = m_cell * atr
    risk_usd = REF_ACCOUNT_USD * f_risk_cap1
    size_oz = risk_usd / stop_usd if stop_usd > 0 else 0.0
    decision = {
        "as_of_bar": str(ts),
        "symbol": "XAUUSD",
        "policy": {
            "spec": "v1.3",
            "source": LIVE_POLICY_SRC,
            "w_pool": LIVE_POOL_W, "kappa": LIVE_KAPPA,
            "lambda_var": LIVE_LAMBDA_VAR, "ev_min": LIVE_EV_MIN,
            "cost_base_R": LIVE_COST_BASE_R, "overlay_rule": LIVE_OVERLAY_RULE,
            "ood_threshold": LIVE_OOD_TH,
            "kelly_fraction": KELLY_FRACTION_LIVE,
            "equity_risk_cap": HARD_EQUITY_RISK_CAP_LIVE},
        "model": {
            "core": "tskdev/aurum-models-v1/core_fp16.pt",
            "heads": "heads_main.pt", "forward_depths_r": LIVE_DEPTHS,
            "pool_depths_r": [1, 2, 3], "ood_reference_depth_r": 3,
            "temperatures": temps},
        "depths_h1_at_selected_cell": {
            f"r{r}": round(float(P[r][cell]), 4) for r in LIVE_DEPTHS},
        "pooling": {
            "w": LIVE_POOL_W, "p_bar_cell": round(p_bar_cell, 4),
            "sigma_loop_cell": round(sig_cell, 4),
            "ood_mean": round(ood_mean, 5), "ood_max": round(ood_max, 5)},
        "lcb_ev": {
            "b_geo": round(b_cell, 4), "cost_R": round(cost_cell, 4),
            "base_rate_prior_cell": round(pi_cell, 4),
            "p_shrunk_cell": round(p_sh_cell, 4),
            "var_penalty_R": round(LIVE_LAMBDA_VAR * b_cell * sig_cell, 4),
            "ev_lcb_R": round(ev_cell, 4), "ev_min_hurdle": LIVE_EV_MIN},
        "gbdt_agreement": {
            "fold_year": y_live, "train_through": y_live - 1,
            "n_features": gb["n_features"], "pL": round(pL, 4),
            "pS": round(pS, 4), "dir": gbdt_dir, "cell_dir": d_cell,
            "agrees": agree, "audit": gb["audit"]},
        "overlay_path": {
            "rule": LIVE_OVERLAY_RULE, "source": ov_src,
            "hard_stop_atr": float(m_cell), "take_profit_atr": float(t_cell),
            "partial_tp_atr": round(q50_atr, 3), "trail_mae_atr": round(q25_atr, 3),
            "time_stop_bars": int(h_cell),
            "orders_placed_at_entry": True,
            "partial_tp_active": bool(0 < q50_atr < t_cell),
            "tightened_stop_active": bool(0 < q25_atr < m_cell)},
        "decision": {
            "action": ("BUY" if d_cell == "L" else "SELL") if trade_ok else "NO TRADE",
            "cell": cell_names()[cell],
            "direction": "long" if d_cell == "L" else "short",
            "stop_atr_mult": float(m_cell), "tp_atr_mult": float(t_cell),
            "horizon_h1_bars": int(h_cell),
            "p_win_pooled": round(p_bar_cell, 4),
            "p_win_shrunk": round(p_sh_cell, 4),
            "b_geo": round(b_cell, 4),
            "ev_lcb_R": round(ev_cell, 4),
            "kelly_raw": round(float(kelly), 5),
            "kelly_fraction_quarter": round(float(KELLY_FRACTION_LIVE * kelly), 5),
            "risk_fraction_capped_cap1": round(f_risk_cap1, 5),
            "risk_fraction_capped_cap3": round(f_risk_cap3, 5),
            "no_trade_reason": reason},
        "sizing_reference": {
            "account_usd": REF_ACCOUNT_USD,
            "risk_usd_cap1": round(risk_usd, 2),
            "atr14_h1_usd": round(atr, 3),
            "close_usd": round(c_usd, 2),
            "stop_distance_usd": round(stop_usd, 3),
            "position_oz_cap1": round(size_oz, 3) if trade_ok else 0.0,
            "position_lots_cap1": round(size_oz / 100.0, 4) if trade_ok else 0.0},
        "gates": {
            "ev_lcb_ok": ev_ok, "prob_ok": prob_ok, "loop_ood_ok": ood_ok,
            "loop_ood_mean": round(ood_mean, 5), "loop_ood_max": round(ood_max, 5),
            "gbdt_agrees": agree, "blackout_clear": bool(blackout_clear),
            "blackout_detail": blackout, "overlay_ok": ov_ok,
            "era_score": era,
            "era_disagreement": (round(abs((era or 0.5) - p_bar_cell), 4)
                                 if era is not None else None)},
        "heads_detail": {
            "h2_up_prob": {k: round(float(out[3]["h2"][j]), 4)
                           for j, k in enumerate(
                        ["up4", "up12", "up24", "up48", "big4", "big12",
                         "big24", "big48"])},
            "h3_excursions_R_atr": {
                "L_mfe_q25": round(float(h3v[0]), 3),
                "L_mfe_q50": round(float(h3v[1]), 3),
                "L_mfe_q75": round(float(h3v[2]), 3),
                "L_mae_q25": round(float(h3v[3]), 3),
                "L_mae_q50": round(float(h3v[4]), 3),
                "L_mae_q75": round(float(h3v[5]), 3),
                "S_mfe_q25": round(float(h3v[6]), 3),
                "S_mfe_q50": round(float(h3v[7]), 3),
                "S_mfe_q75": round(float(h3v[8]), 3),
                "S_mae_q25": round(float(h3v[9]), 3),
                "S_mae_q50": round(float(h3v[10]), 3),
                "S_mae_q75": round(float(h3v[11]), 3)},
            "h4_vol": {"expansion_probs": [round(float(x), 4) for x in
                                            out[3]["h4"][:4]],
                       "vol_tercile_probs": [round(float(x), 4) for x in
                                             out[3]["h4"][4:7]],
                       "vol_q50_ratio": round(float(out[3]["h4"][7]), 3)},
            "h5_regime_probs": {regime_names[j]: round(float(h5v[j]), 4)
                                for j in range(9)},
            "regime_call": regime_names[int(np.argmax(h5v))]},
        "generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(WORK, "decision.json"), "w") as f:
        json.dump(decision, f, indent=1)

    audit = {
        "checklist": {
            "state_serialized_from_latest_fully_featured_bar": True,
            "tokens": int(len(enc)),
            "forward_depths_r": LIVE_DEPTHS,
            "cpu_only": dev.type == "cpu",
            "temperature_scaling_applied_h1_h5": True,
            "probabilities_bounded_0_1": bool(P[3].min() >= 0 and P[3].max() <= 1),
            "multi_depth_pool_w": LIVE_POOL_W,
            "lcb_ev_gate": {"hurdle": LIVE_EV_MIN, "value": round(ev_cell, 5),
                             "pass": ev_ok},
            "lambda_var": LIVE_LAMBDA_VAR,
            "sigma_loop_at_cell": round(sig_cell, 5),
            "loop_ood_gate": {"threshold": LIVE_OOD_TH,
                               "value": round(ood_mean, 5), "pass": ood_ok},
            "gbdt_agreement_gate": {"dir": gbdt_dir, "cell_dir": d_cell,
                                    "agrees": agree,
                                    "train_through": y_live - 1},
            "event_blackout_gate": blackout,
            "overlay_orders_placed_at_entry": True,
            "overlay_rule": LIVE_OVERLAY_RULE,
            "kelly_capped_at": HARD_EQUITY_RISK_CAP_LIVE,
            "quarter_kelly": KELLY_FRACTION_LIVE,
            "risk_fraction": round(f_risk_cap1, 5),
            "stop_distance_usd": round(stop_usd, 3),
            "position_oz_ref_account": round(size_oz, 3),
            "decision_emitted": decision["decision"]["action"],
            "no_lookahead": ("state uses bar features at/ before as_of_bar only; "
                             "GBDT fold trained <= y-1; overlay quantiles h48 from "
                             "H3 head, h24 empirical <= y-1; base-rate prior <= y-1"),
            "policy_source": LIVE_POLICY_SRC},
        "era_score": era,
        "n_gates_passed": int(sum(1 for g in GATES.values() if g["pass"])),
    }
    with open(os.path.join(WORK, "risk_audit.json"), "w") as f:
        json.dump(audit, f, indent=1)
    gate("G6.5_audit",
         os.path.isfile(os.path.join(WORK, "decision.json"))
         and os.path.isfile(os.path.join(WORK, "risk_audit.json"))
         and decision["decision"]["action"] is not None,
         "decision.json + risk_audit.json written; action="
         + decision["decision"]["action"])

    save_gates(os.path.join(WORK, "gates.json"))
    log("LIVE DAILY COMPLETE. GATES: "
        + json.dumps({k: v["pass"] for k, v in GATES.items()}))
    log(f"total wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    WORK = os.environ.get("AURUM_WORK",
                          "/kaggle/working" if os.path.exists("/kaggle/working")
                          else os.getcwd())
    set_seeds(1337)
    live_main(WORK)
