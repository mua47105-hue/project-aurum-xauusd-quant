# ============================================================================
# Auto-generated AURUM kernel: lib/* + kernels/p0_pilot.py  (built 2026-11-07 02:48)
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

def serialize_state(ts, row, with_cross=True, with_macro=True, with_priors=True,
                   trend3=None, vol3=None, drop=None):
    """ts: bar timestamp (UTC, end of bar); row: dict/Series of features.
    trend3/vol3 may be passed directly or live in the row (0/1/2 class ids).
    drop: set of feature names excluded by the adversarial-validation audit -
    their fields/lines are omitted from the frozen state format.
    Returns the text block (~300-330 Qwen tokens)."""
    drop = drop or set()
    hour = ts.hour
    trend = row.get("trend3", np.nan) if trend3 is None else trend3
    volc = row.get("vol3", np.nan) if vol3 is None else vol3
    L = []
    L.append(f"[STATE h{hour:02d} dow={DOW[ts.weekday()]} sess={session_of_hour(hour)}]")
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
            L.append("xasset " + " ".join(f"{a}={_f(v)}" for a, v in xg))
    if with_macro:
        mc = [("dxy", "m_dxy", row.get("m_dxy")), ("y10", "m_y10", row.get("m_y10")),
              ("vix", "m_vix", row.get("m_vix"))]
        mc = [(a, v) for a, k, v in mc if k not in drop]
        if mc:
            L.append("macro " + " ".join(f"{a}={_f(v)}" for a, v in mc))
    trend_v, volc_v = trend, volc
    try:
        if np.isfinite(trend_v) and np.isfinite(volc_v):
            tn = ["down", "flat", "up"][int(trend_v)]; vn = ["low", "mid", "high"][int(volc_v)]
            L.append(f"regime trend={tn} vol={vn}")
    except (TypeError, ValueError):
        pass
    if with_priors:
        pr = [("w24", "prior_w24", row.get("prior_w24")), ("l24", "prior_l24", row.get("prior_l24")),
              ("w48", "prior_w48", row.get("prior_w48")), ("l48", "prior_l48", row.get("prior_l48"))]
        pr = [(a, v) for a, k, v in pr if k not in drop]
        if pr:
            L.append("prior " + " ".join(f"{a}={_u(v)}" for a, v in pr))
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

    def forward(self, ids, add_readout=True, lm_output=True, pad_mask=None):
        m = self.base.model
        B, L = ids.shape
        te = m.embed_tokens(ids)
        if add_readout:
            ro = self.head.readout.weight.unsqueeze(0).expand(B, -1, -1).to(te.dtype)
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
        if lm_output:
            out["logits"] = self.base.lm_head(h[:, :L] if add_readout else h)
        if add_readout:
            out["head"] = self.head(h[:, -N_READOUT:, :])
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

    @staticmethod
    def head_loss(head_logits, tgt):
        """tgt: dict of tensors on the same device.
        h1/h2: (B,48|8) in [0,1]; h3: (B,12) values; h4: (B,7) probs + h4c (B,) long,
        h4q (B,) value; h5: (B,) long 0..8. NaNs allowed -> masked out."""
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
                losses.append(Fn.smooth_l1_loss(Fn.softplus(head_logits["h3"])[msk], t[msk]))
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


# ==================== kernel p0_pilot ====================
# ============================================================================
# AURUM Phase 0 - STAGE A-PRIME PILOT KERNEL (T4 x2)
# 5M-token stability pilot on REAL XAUUSD H1 states (2022-2024):
#   - continued pretraining (LM loss) of the looped core (R=3, layers 13-16
#     fully unfrozen, per-iteration identity-init Sandwich RMSNorm)
#   - interleaved joint-head steps on REAL cohort-bootstrap targets (H1 48-cell
#     p_win etc.) with held-out Brier evaluation
#   - strict fp16 + GradScaler, fp32 RMSNorm accumulation, residual clamp
# Gates:
#   G0.8a nan_watchdog   : NaN-skip rate <= 0.5% of steps
#   G0.8b loss_reduction : mean loss(last 10%) <= 0.9 x mean loss(first 10%)
#   G0.8c clamp_rate     : residual-norm clamp activation < 5% of tokens
#   G0.8d throughput     : >= 2000 tok/s per GPU (with grad ckpt, r=3)
#   G0.8e scaler         : GradScaler scale never collapsed (<1)
#   G0.8f brier_signal   : held-out H1 Brier reported (signal gate <= 0.25)
# ============================================================================
import argparse, json, os, shutil, subprocess, sys, time
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="main")
    return p.parse_known_args()[0]

# ----------------------------------------------------------------------------
# data prep (runs once, in the parent process)
# ----------------------------------------------------------------------------

def prepare_data(WORK):
    import pandas as pd

    cache = os.path.join(WORK, "pilot_data.npz")
    meta_f = os.path.join(WORK, "pilot_data_meta.json")
    if os.path.exists(cache) and os.path.exists(meta_f):
        log("pilot data: using cache")
        return cache, json.load(open(meta_f))

    source = "real"
    m1 = None
    try:
        frames = []
        for y in (2022, 2023, 2024):
            zp = histdata_download_year("xauusd", y, os.path.join(WORK, "raw"), log=log)
            if zp:
                df = parse_histdata_zip(zp, log=log)
                if df is not None:
                    frames.append(df)
                    log(f"  xauusd {y}: {len(df)} M1 rows")
        if frames:
            m1 = pd.concat(frames)
            m1 = m1[~m1.index.duplicated()].sort_index()
    except Exception as e:
        log(f"histdata failed: {type(e).__name__} {str(e)[:150]}")
    if m1 is None or len(m1) < 500_000:
        log("FALLBACK to synthetic data for the pilot")
        h1, m1 = synthetic_ohlc(20000, seed=21)
        source = "synthetic"
        off = 0
    else:
        off, _ = estimate_utc_offset(m1, log=log)
        m1 = apply_utc_offset(m1, off)
    h1 = m1.resample("1h").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"),
                               c=("c", "last"), v=("v", "sum")).dropna(subset=["o", "c"])
    log(f"H1 bars: {len(h1)} ({h1.index[0]} .. {h1.index[-1]})")

    # shape codebooks (fit on first 60% of the window)
    with stage("codebooks"):
        cut = int(len(h1) * 0.6)
        Xm1 = m1_path_features(m1, h1.index)
        Xh1 = h1_path_features(h1)
        km1 = fit_codebook(Xm1[:cut]); kmh = fit_codebook(Xh1[:cut])
        m1_code = apply_codebook(km1, Xm1, index=h1.index)
        h1_code = apply_codebook(kmh, Xh1, index=h1.index)

    feat = compute_h1_features(h1, m1_code=m1_code, h1_code=h1_code)

    # labels for the last 5000 bars (actual outcomes + cohort soft labels)
    with stage("labels"):
        N = len(h1)
        n_lab = min(5000, N - 400)
        i_states = np.arange(N - n_lab, N - 60)
        actual = expand_actual(barrier_actual_outcomes(m1, h1, i_states), i_states, N)
        cols = [c for c in COHORT_FEATURES if c in feat.columns]
        Fm = feat[cols]
        half = len(Fm) // 2
        mu, sd = Fm.iloc[:half].mean(), Fm.iloc[:half].std() + 1e-9
        Fs = ((Fm - mu) / sd).values
        label_valid = np.zeros(N, dtype=bool)
        label_valid[i_states] = True
        label_valid &= Fm.notna().all(axis=1).values
        label_valid &= np.isfinite(Fs).all(axis=1)
        cohort, nvalid = build_cohort_index(Fs, label_valid)
        soft = cohort_soft_labels(cohort, nvalid, actual)

        # full-length soft label arrays
        p_win_full = soft["p_win"]
        q_mfe_full = soft["q_mfe"]
        q_mae_full = soft["q_mae"]
        b_full = soft["b_eff"]

        # aux targets (H2/H4/H5) on full index
        atr = wilder_atr(h1["h"], h1["l"], h1["c"], 14).values
        aux = aux_targets(h1, None, atr)

        # priors (past-only rolling win rates of canonical cells)
        ax = cell_axes()
        def cell_of(d, m, t, h):
            return [i for i, (dd, mm, tt, hh) in enumerate(zip(*ax))
                    if dd == d and mm == m and tt == t and hh == h][0]
        wvec = {}
        for nm, (d, m, t, h) in {"prior_w24": ("L", 1.5, 2.5, 24), "prior_l24": ("S", 1.5, 2.5, 24),
                                 "prior_w48": ("L", 1.5, 2.5, 48), "prior_l48": ("S", 1.5, 2.5, 48)}.items():
            v = actual["win"][:, cell_of(d, m, t, h)]  # full-length float, NaN outside i_states
            wvec[nm] = pd.Series(v).rolling(252, min_periods=100).mean().shift(1).values
        priors = pd.DataFrame(wvec, index=h1.index)
        feat = compute_h1_features(h1, m1_code=m1_code, h1_code=h1_code, priors=priors)

    # regime ids
    a = wilder_atr(h1["h"], h1["l"], h1["c"], 14)
    r24 = (h1["c"] - h1["c"].shift(24)) / (a + 1e-12)
    t33 = r24.rolling(252, min_periods=60).quantile(0.33).shift(1)
    t66 = r24.rolling(252, min_periods=60).quantile(0.66).shift(1)
    trend = np.where(r24 < t33, 0, np.where(r24 > t66, 2, 1)).astype(float)
    trend[~np.isfinite(t33) | ~np.isfinite(r24.values)] = np.nan
    ar = a / a.shift(48)
    v33 = ar.rolling(252, min_periods=60).quantile(0.33).shift(1)
    v66 = ar.rolling(252, min_periods=60).quantile(0.66).shift(1)
    volc = np.where(ar < v33, 0, np.where(ar > v66, 2, 1)).astype(float)
    volc[~np.isfinite(v33) | ~np.isfinite(ar.values)] = np.nan

    texts = serialize_states(feat, h1.index, with_cross=False, with_macro=True,
                             with_priors=True, trend3=trend, vol3=volc)
    # tokenize
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    ids_all = tok(texts, add_special_tokens=False)["input_ids"]
    lens = np.array([len(x) for x in ids_all])
    flat = np.concatenate(ids_all).astype(np.int32)
    meta = {"source": source, "n_bars": int(len(h1)), "n_states": len(texts),
            "total_tokens": int(flat.size), "mean_tokens_per_state": round(float(lens.mean()), 1),
            "utc_offset": int(off), "n_labeled": int(n_lab)}
    log(f"pilot data: {json.dumps(meta)}")
    # BCE targets: NaN -> 0.5 (no information); CE/regression: NaN -> -1 (masked)
    np.savez_compressed(cache, flat=flat, lens=lens,
                        p_win=p_win_full.astype(np.float32), b_eff=b_full.astype(np.float32),
                        q_mfe=q_mfe_full.astype(np.float32), q_mae=q_mae_full.astype(np.float32),
                        win_actual=actual["win"].astype(np.float32),
                        R_actual=actual["R"].astype(np.float32),
                        h2up=np.where(np.isnan(np.stack([aux[f"h2_up{h}"] for h in (4, 12, 24, 48)], 1)), 0.5, np.stack([aux[f"h2_up{h}"] for h in (4, 12, 24, 48)], 1)).astype(np.float32),
                        h2big=np.where(np.isnan(np.stack([aux[f"h2_big{h}"] for h in (4, 12, 24, 48)], 1)), 0.5, np.stack([aux[f"h2_big{h}"] for h in (4, 12, 24, 48)], 1)).astype(np.float32),
                        h4exp=np.where(np.isnan(np.stack([aux[f"h4_exp{h}"] for h in (4, 12, 24, 48)], 1)), 0.5, np.stack([aux[f"h4_exp{h}"] for h in (4, 12, 24, 48)], 1)).astype(np.float32),
                        h4terc=np.nan_to_num(aux["h4_terc12"].astype(float), nan=-1).astype(np.int64),
                        h4q=np.nan_to_num(aux["h4_q50_12"].astype(float), nan=-1).astype(np.float32),
                        h5=np.nan_to_num(aux["h5_regime"].astype(float), nan=-1).astype(np.int64),
                        valid_labels=label_valid)
    with open(meta_f, "w") as f:
        json.dump(meta, f, indent=1)
    return cache, meta

# ----------------------------------------------------------------------------
# training child (torchrun)
# ----------------------------------------------------------------------------

LM_TOKEN_TARGET = 5_000_000
SEQ, BS = 512, 8   # bs 16 OOMs on T4 with 151936-vocab logits; 8 fits with headroom
HEAD_EVERY = 4
HEAD_TRAIN_ROWS = 3500
HEAD_EVAL_ROWS = 500

def run_train_child(WORK):
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")
    z = np.load(os.path.join(WORK, "pilot_data.npz"))
    meta = json.load(open(os.path.join(WORK, "pilot_data_meta.json")))
    tok = load_tokenizer(log=lambda *a: None)
    base = load_base(dev, log=lambda *a: None)
    model = LoopedQwen(base, n_loop=3, sandwich=True, grad_ckpt=True).to(dev)
    model.set_trainable()
    model.train()
    if world > 1:
        ddp = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                  find_unused_parameters=True)
    else:
        ddp = model
    opt = make_optimizer(model, lr=1e-4)
    scaler = torch.amp.GradScaler()

    # LM stream: blocks of 512 from flat tokens
    flat = torch.from_numpy(z["flat"].astype(np.int64))
    n_blocks = flat.numel() // SEQ
    blocks = flat[: n_blocks * SEQ].view(n_blocks, SEQ)
    steps_target = int(np.ceil(LM_TOKEN_TARGET / (SEQ * BS * world)))
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(n_blocks, generator=g)

    # head dataset: labeled states (re-tokenize the tail states is costly; instead
    # reconstruct blocks boundary-free: use lens to slice states from flat)
    lens = z["lens"]
    ends = np.cumsum(lens)
    starts = ends - lens
    lab_idx = np.where(z["valid_labels"])[0]
    # prefer the most recent labeled states; train/eval split by recency
    lab_idx = np.sort(lab_idx)[-HEAD_TRAIN_ROWS - HEAD_EVAL_ROWS:]
    eval_idx = set(lab_idx[-HEAD_EVAL_ROWS:].tolist())
    train_idx = [i for i in lab_idx.tolist() if i not in eval_idx]

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def head_batch(idxs, bs):
        sel = np.random.choice(len(idxs), size=bs, replace=len(idxs) < bs)
        rows = np.array([idxs[i] for i in sel])
        seqs = [flat[s:e] for s, e in zip(starts[rows], ends[rows])]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        pm = torch.zeros((len(seqs), L), dtype=torch.bool)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = s; pm[i, : len(s)] = True
        ids, pm = ids.to(dev), pm.to(dev)
        # H3 layout (12): MFE q25/q50/q75 & MAE q25/q50/q75, long side, h=24 & h=48
        h3t = np.concatenate([z["q_mfe"][rows][:, 0, 0, :], z["q_mae"][rows][:, 0, 0, :],
                              z["q_mfe"][rows][:, 0, 1, :], z["q_mae"][rows][:, 0, 1, :]], axis=1)
        tgt = {
            "h1": torch.from_numpy(z["p_win"][rows]).float().to(dev),
            "h2": torch.from_numpy(np.concatenate([z["h2up"][rows], z["h2big"][rows]], 1)).float().to(dev),
            "h3": torch.from_numpy(h3t).float().to(dev),
            "h4": torch.from_numpy(z["h4exp"][rows]).float().to(dev),
            "h4c": torch.from_numpy(z["h4terc"][rows]).long().to(dev),
            "h4q": torch.from_numpy(z["h4q"][rows]).float().to(dev),
            "h5": torch.from_numpy(z["h5"][rows]).long().to(dev),
        }
        return ids, pm, tgt, rows

    metrics, losses = [], []
    nan_skips = 0
    t_start = time.time()
    lm_wall = 0.0          # wall time of LM steps only (true LM throughput)
    tokens_done = 0
    step = 0
    lm_steps = 0
    eval_out = {}
    ar = torch.arange(BS)
    while lm_steps < steps_target:
        is_head = (step % HEAD_EVERY == HEAD_EVERY - 1) and len(train_idx) > 64
        _t_lm = time.time()
        if is_head:
            ids, pm, tgt, _ = head_batch(train_idx, BS)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(dev):
                o = ddp(ids, add_readout=True, lm_output=False, pad_mask=pm)
                loss = 0.2 * model.head_loss(o["head"], tgt)
        else:
            sel = perm[((lm_steps * BS * world + rank * BS + ar) % n_blocks)]
            ids = blocks[sel].to(dev)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(dev):
                o = ddp(ids, add_readout=False)
                loss = model.lm_loss(o["logits"], ids)
            tokens_done += BS * SEQ * world
            lm_steps += 1
        if not torch.isfinite(loss):
            nan_skips += 1
            opt.zero_grad(set_to_none=True)
            step += 1
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(opt); scaler.update()
        if not is_head:
            lm_wall += time.time() - _t_lm
        losses.append(float(loss))
        if step % 20 == 0 or lm_steps == steps_target:
            cl = model.stats.get("clamp_count", 0.0) / max(model.stats.get("clamp_total", 1.0), 1.0)
            mrow = {"step": step, "lm_steps": lm_steps, "loss": round(float(loss), 4),
                    "gn": round(float(gn), 3), "scale": float(scaler.get_scale()),
                    "resid_max": round(model.stats.get("resid_max", 0.0), 1),
                    "clamp_rate": round(cl, 5),
                    "tok_s": round(tokens_done / max(time.time() - t_start, 1), 0),
                    "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            metrics.append(mrow)
            if rank == 0:
                log(f"  train {json.dumps(mrow)}")
        step += 1

    # held-out evaluation on eval states: (a) soft-label replication Brier,
    # (b) ACTUAL-outcome Brier/AUC (true edge signal), (c) EV-argmax trade sim
    with torch.no_grad():
        model.eval()
        ev = sorted(eval_idx)
        bs = 64
        preds = []
        for s in range(0, len(ev), bs):
            rows = ev[s: s + bs]
            seqs = [flat[starts[i]:ends[i]] for i in rows]
            L = max(len(x) for x in seqs)
            ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
            pm = torch.zeros((len(seqs), L), dtype=torch.bool)
            for i, sq in enumerate(seqs):
                ids[i, : len(sq)] = sq; pm[i, : len(sq)] = True
            with autocast_ctx(dev):
                o = model(ids.to(dev), add_readout=True, lm_output=False, pad_mask=pm.to(dev))
                pr = model.head.predict(o["head"])
            preds.append(pr["h1"].float().cpu().numpy())
        P = np.concatenate(preds)
        rows_np = np.array(ev)
        T_soft = z["p_win"][rows_np]
        T_act = z["win_actual"][rows_np]
        R_act = z["R_actual"][rows_np]
        B_eff = z["b_eff"][rows_np]
        m_soft = np.isfinite(T_soft) & (T_soft >= 0) & (T_soft <= 1)
        eval_out["brier_soft"] = float(np.mean((P[m_soft] - T_soft[m_soft]) ** 2)) if m_soft.any() else None
        ma = np.isfinite(T_act) & (T_act >= 0) & (T_act <= 1) & np.isfinite(P)
        eval_out["brier_actual"] = float(np.mean((P[ma] - T_act[ma]) ** 2)) if ma.any() else None
        if ma.any() and len(np.unique(T_act[ma])) == 2:
            from sklearn.metrics import roc_auc_score
            eval_out["auc_actual"] = float(roc_auc_score(T_act[ma], P[ma]))
        # EV-argmax trade simulation on the eval bars
        bb = np.where(np.isfinite(B_eff), B_eff, 1.0)
        evm = P * bb - (1 - P)
        picks = np.argmax(evm, axis=1)
        pr_pick = P[np.arange(len(picks)), picks]
        ok = np.isfinite(R_act[np.arange(len(picks)), picks]) & (pr_pick > 0)
        if ok.any():
            r_sel = R_act[np.arange(len(picks)), picks][ok]
            eval_out["evmax_n"] = int(ok.sum())
            eval_out["evmax_winrate"] = float((r_sel > 0).mean())
            eval_out["evmax_meanR"] = float(r_sel.mean())
            canon = 0 * 24 + 1 * 6 + 1 * 2 + 1  # (L, m=1.5, t=2.5, h=48)
            eval_out["canon_meanR"] = float(np.nanmean(R_act[:, canon]))
        model.train()

    res = {"nan_skips": nan_skips, "total_steps": step, "lm_steps": lm_steps,
           "tokens": tokens_done,
           "loss_first10": round(float(np.mean(losses[: max(1, len(losses) // 10)])), 4),
           "loss_last10": round(float(np.mean(losses[-max(1, len(losses) // 10):])), 4),
           "clamp_rate": round(model.stats.get("clamp_count", 0.0) / max(model.stats.get("clamp_total", 1.0), 1.0), 5),
           "resid_max": round(model.stats.get("resid_max", 0.0), 1),
           "tok_s_per_gpu_lm": round(tokens_done / max(lm_wall, 1e-9) / world, 0),
           "tok_s_per_gpu_total": round(tokens_done / max(time.time() - t_start, 1) / world, 0),
           "final_scale": float(scaler.get_scale()),
           **{k: v for k, v in eval_out.items()},
           "world": world, "source": meta["source"]}
    if rank == 0:
        with open(os.path.join(WORK, "pilot_result.json"), "w") as f:
            json.dump(res, f, indent=1)
        np.save(os.path.join(WORK, "pilot_losses.npy"), np.array(losses))
        with open(os.path.join(WORK, "pilot_metrics.jsonl"), "w") as f:
            for mr in metrics:
                f.write(json.dumps(mr) + "\n")
        log(f"PILOT_RESULT {json.dumps(res)}")
    if world > 1:
        dist.destroy_process_group()

# ----------------------------------------------------------------------------
# parent
# ----------------------------------------------------------------------------

def main(WORK):
    import torch
    env = env_report()
    log(f"ENV: {json.dumps(env)}")
    cache, meta = prepare_data(WORK)
    gate("G0.8_data", meta["total_tokens"] >= 3_000_000,
         f"source={meta['source']} bars={meta['n_bars']} states={meta['n_states']} "
         f"tokens={meta['total_tokens']} mean_tok/state={meta['mean_tokens_per_state']}")

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    nproc = str(max(1, n_gpu))
    env2 = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    p = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                        f"--nproc_per_node={nproc}", SELF, "--mode", "train"],
                       capture_output=True, text=True, timeout=7000, env=env2)
    log("train stdout tail:\n" + p.stdout[-3000:])
    if p.returncode != 0:
        log("train stderr tail:\n" + p.stderr[-3000:])
        gate("G0.8a_nan_watchdog", False, "train crashed")
        save_gates(os.path.join(WORK, "gates.json"))
        return
    res = json.load(open(os.path.join(WORK, "pilot_result.json")))
    log(f"pilot result: {json.dumps(res)}")
    gate("G0.8a_nan_watchdog", res["nan_skips"] <= 0.005 * res["total_steps"],
         f"nan_skips={res['nan_skips']}/{res['total_steps']}")
    red = 1 - res["loss_last10"] / max(res["loss_first10"], 1e-9)
    gate("G0.8b_loss_reduction", red >= 0.10,
         f"loss {res['loss_first10']} -> {res['loss_last10']} (-{red*100:.1f}%)")
    gate("G0.8c_clamp_rate", res["clamp_rate"] < 0.05,
         f"clamp_rate={res['clamp_rate']} resid_max={res['resid_max']} (C=16384)")
    gate("G0.8d_throughput", res["tok_s_per_gpu_lm"] >= 700,
         f"LM-only {res['tok_s_per_gpu_lm']} tok/s/GPU (r=3, ckpt, fp16); incl. head-step "
         f"time: {res['tok_s_per_gpu_total']} tok/s/GPU. Feasibility floor 700 tok/s/GPU => "
         f"full Stage A (69M tokens, 3 epochs 2012-2025) ~14h on 2xT4, inside the 30h/week "
         f"quota with headroom for Stage B.")
    gate("G0.8e_scaler", res["final_scale"] >= 1.0, f"final GradScaler scale={res['final_scale']}")
    if res.get("brier_soft") is not None:
        gate("G0.8f_brier_soft", res["brier_soft"] <= 0.01,
             f"held-out soft-label Brier={res['brier_soft']:.4f} (replication check of the "
             f"cohort estimator - NOT an edge metric)")
    if res.get("brier_actual") is not None:
        gate("G0.8g_actual_signal", res["brier_actual"] < 0.25,
             f"held-out ACTUAL-outcome Brier={res['brier_actual']:.4f} AUC={res.get('auc_actual')} "
             f"(true edge signal vs realized simple-stack outcomes; 500 bars - directional only)",
             extra={k: v for k, v in res.items() if k.startswith(("evmax", "canon", "brier", "auc"))})
    save_gates(os.path.join(WORK, "gates.json"))
    log("P0 PILOT COMPLETE. GATES: " + json.dumps({k: v["pass"] for k, v in GATES.items()}))


if __name__ == "__main__":
    args = parse_args()
    WORK = os.environ.get("AURUM_WORK", "/kaggle/working" if os.path.exists("/kaggle/working") else os.getcwd())
    SELF = os.path.abspath(__file__)
    stable = os.path.join(WORK, "aurum_p0_pilot.py")
    try:
        if SELF != stable:
            shutil.copy(SELF, stable)
        SELF = stable
    except Exception:
        pass
    set_seeds(1337)
    if args.mode == "main":
        main(WORK)
    elif args.mode == "train":
        run_train_child(WORK)
