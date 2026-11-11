"""
AURUM v2.0 -- Shared kernel library (v2_common.py)
===================================================
Common machinery for the v2.0 execution kernels K2/K3/K6:
  * data loading (aurum_v2_features.parquet + phase1 H1 prices)
  * year-based purged walk-forward splits (v1.3-compatible semantics:
    fold head for year y trains on bars < y with a 24-bar purge at the
    boundary; 2016-2022 = FIT, 2023-2025 = untouched OOS)
  * portfolio metrics on the continuous per-bar net return stream
  * the OnlineExecutionEngine runner with component toggles (Baseline
    Gauntlet G5.3) and frozen-model comparison
  * the Master Trial Ledger (every configuration from Experiment #1)
  * block-bootstrap confidence intervals

All costs in vol-scaled return units: cost_r = 5e-5 per unit exposure change
(~1bp round turn on unit notional at the 15% vol target; v1.3's 0.05R at
1.0-ATR stop is the barrier-trade analogue -- documented approximation).
"""

import hashlib
import json
import math
import os
import time

import numpy as np
import pandas as pd
from online_engine import (OnlineRidge, RLSBank, PageHinkley, VolTarget,
                               CarverBuffer, DrawdownGovernor)

BARS_PER_YEAR = 5786          # empirical: 87,948 bars / 15.2 y (locked)
COST_R = 5e-5                 # per unit |dz| at cost multiplier 1.0
SIGMA_TGT_ANN = 0.15
FIT_YEARS = list(range(2016, 2023))     # 2016-2022: selection / stacking window
OOS_YEARS = [2023, 2024, 2025]          # untouched OOS
ALL_FOLD_YEARS = list(range(2016, 2026))


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------
def _find_dir(probe: str, *cands: str) -> str | None:
    for c in cands:
        if c and os.path.isfile(os.path.join(c, probe)):
            return c
    return None


def find_features_ds():
    ds = os.environ.get("AURUM_V2F")
    return ds or _find_dir(
        "aurum_v2_features.parquet",
        "/kaggle/input/aurum-v2-featstore",
        "/kaggle/input/aurum-v2-features",
        "/kaggle/input/datasets/tskdev/aurum-v2-featstore",
        *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v2-featstore*")),
        *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v2-features*")),
    )


def find_phase1_ds():
    ds = os.environ.get("AURUM_DS")
    return ds or _find_dir(
        "h1_xauusd.parquet",
        "/kaggle/input/aurum-phase1-data",
        "/kaggle/input/datasets/tskdev/aurum-phase1-data",
        *sorted(__import__("glob").glob("/kaggle/input/*/aurum-phase1-data*")),
    ) or find_features_ds()   # the featstore copies also carry h1_xauusd.parquet


def load_v2_data():
    """Returns (feats, price) where feats has extra causal columns:
    r_xau (per-bar log return), sigma (EW-96 std), close_xau."""
    fdir = find_features_ds()
    assert fdir, "aurum-v2-features dataset not found"
    feats = pd.read_parquet(os.path.join(fdir, "aurum_v2_features.parquet"))
    feats.index = pd.DatetimeIndex(feats.index)
    if feats.index.tz is None:
        feats.index = feats.index.tz_localize("UTC")
    pdir = find_phase1_ds()
    assert pdir, "aurum-phase1-data dataset not found"
    px = pd.read_parquet(os.path.join(pdir, "h1_xauusd.parquet"))
    px.index = pd.DatetimeIndex(px.index)
    if px.index.tz is None:
        px.index = px.index.tz_localize("UTC")
    close = px["c"].reindex(feats.index) if "c" in px.columns else px["close"].reindex(feats.index)
    feats["close_xau"] = close
    feats["r_xau"] = np.log(close / close.shift(1))
    feats["sigma"] = feats["r_xau"].ewm(span=96, min_periods=24).std()
    return feats, px


# ----------------------------------------------------------------------------
# year-based purged walk-forward
# ----------------------------------------------------------------------------
def fold_indices(feats: pd.DataFrame, year: int, purge: int = 24,
                 es_frac: float = 0.15):
    """Walk-forward split for test-year `year`:
    train = bars strictly before Jan-1 of `year` minus a `purge`-bar gap
            (label windows of train rows must not overlap test labels),
    split chronologically into (fit, early-stop-val) by es_frac,
    test = bars of `year`.
    Returns (fit_pos, es_pos, test_pos) as positional indices into feats."""
    idx = feats.index
    year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    t0 = idx.searchsorted(year_start)
    train_end = max(0, t0 - purge)          # purge: drop train rows whose
    test_pos = np.arange(t0, idx.searchsorted(pd.Timestamp(f"{year + 1}-01-01", tz="UTC")))
    train_pos = np.arange(0, train_end)
    if len(train_pos) < 2000 or len(test_pos) < 100:
        return None
    n_es = max(200, int(len(train_pos) * es_frac))
    return train_pos[:-n_es], train_pos[-n_es:], test_pos


# ----------------------------------------------------------------------------
# metrics on the per-bar net return stream (vol-scaled units)
# ----------------------------------------------------------------------------
def ann_metrics(net: np.ndarray, bars_per_year: int = BARS_PER_YEAR) -> dict:
    net = np.asarray(net, dtype="float64")
    net = net[np.isfinite(net)]
    if len(net) < 20:
        return {"n": len(net), "sharpe_ann": float("nan"), "maxdd": float("nan")}
    mu, sd = net.mean(), net.std(ddof=1)
    sharpe = mu / (sd + 1e-12) * math.sqrt(bars_per_year)
    eq = np.cumsum(net)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    maxdd = float(-dd.min())
    # annualized return in vol-target units (per-bar mean * bars/year)
    ret_ann = mu * bars_per_year
    ulcer = float(np.sqrt((dd ** 2).mean()))
    return {"n": int(len(net)), "sharpe_ann": round(float(sharpe), 4),
            "ret_ann_volunits": round(float(ret_ann), 5),
            "maxdd_volunits": round(maxdd, 5),
            "calmar": round(float(ret_ann / maxdd), 4) if maxdd > 1e-12 else None,
            "ulcer": round(ulcer, 6),
            "win_rate": round(float((net > 0).mean()), 4)}


def turnover_per_year(z: np.ndarray) -> float:
    dz = np.abs(np.diff(np.asarray(z, dtype="float64")))
    return float(dz.mean() * BARS_PER_YEAR / 2.0)   # round turns per year


# ----------------------------------------------------------------------------
# engine runner (Baseline Gauntlet toggles)
# ----------------------------------------------------------------------------
def run_online_engine(z_model: pd.Series, r: pd.Series, sigma: pd.Series,
                      cost_mult: float = 1.0,
                      use_buffer: bool = True, use_rls: bool = True,
                      use_regimes: bool = True,
                      label_horizon: int = 24,
                      readout_feats: pd.DataFrame | None = None,
                      vol_of_vol: pd.Series | None = None) -> dict:
    """Run the v2.0 execution stack over a per-bar model-score series, with
    explicit Baseline-Gauntlet (G5.3) component toggles.

    z_model : the frozen model's CONTINUOUS position z in [-1,1] per bar
              (the tanh output; the engine treats it as the score).
      use_buffer=False -> raw rebalancing every bar (no Carver band)
      use_rls=False    -> frozen model (no online readout correction)
      use_regimes=False-> no vol-target/PH/vol-of-vol/DD multipliers
    Uses the SAME component classes as OnlineExecutionEngine (formulas stay
    single-sourced); the explicit loop gives exact toggle control.
    """

    sigma_tgt_bar = SIGMA_TGT_ANN / math.sqrt(BARS_PER_YEAR)
    readout = OnlineRidge(3, lam=0.999)          # intercept + 2 features
    bank = RLSBank(3)
    ph = PageHinkley(lam=50.0)
    vt = VolTarget(sigma_tgt_bar)
    buffer = CarverBuffer()
    dd = DrawdownGovernor()

    if readout_feats is None:
        x1 = z_model.astype("float64")
        x2 = (r / (sigma + 1e-12)).shift(1).fillna(0.0)
        readout_feats = pd.DataFrame({"x1": x1, "x2": x2})
    if vol_of_vol is None:
        vol_of_vol = sigma.rolling(24 * 30, min_periods=24 * 10).std()

    idx = z_model.index
    z_arr = z_model.values.astype("float64")
    r_arr = r.reindex(idx).values
    s_arr = sigma.reindex(idx).values
    vov = vol_of_vol.reindex(idx).values
    X = readout_feats.reindex(idx).values
    # v2.0.1 perf: the vol-of-vol regime multiplier compares the current
    # vol-of-vol to the EXPANDING 80th percentile of its own history.  The
    # original per-bar np.quantile was O(T^2); precomputing the expanding
    # quantile is mathematically identical (same linear interpolation, same
    # NaN->0.0 fill, same t>=720 activation) and O(T log T).
    vov_filled = np.where(np.isfinite(vov), vov.astype("float64"), 0.0)
    vov_q80 = (pd.Series(vov_filled).expanding(min_periods=1)
               .quantile(0.8).values)

    pending: list[tuple[int, np.ndarray, float]] = []
    nets, zs = [], []
    alarms = 0
    regime_sum = 0.0
    z_cur, avg_abs_z, freeze_until, last_net = 0.0, 0.05, -1, 0.0
    for t in range(len(idx)):
        # ---- label-arrival updates (delay-aware; never before t+H)
        arrivals = [p for p in pending if p[0] == t]
        for _, px_, py in arrivals:
            if use_rls and t >= freeze_until:
                readout.update(px_, py)
                bank.update(px_, py)
        pending = [p for p in pending if p[0] != t]

        # ---- regime gates
        m_regime = 1.0
        if use_regimes:
            ph.update(r_arr[t] / (s_arr[t] + 1e-12))
            if ph.n == 0:                      # just reset => alarm fired
                alarms += 1
                freeze_until = t + 24
            vt.update(s_arr[t])
            m_vol = vt.multiplier(s_arr[t])
            m_volov = 1.0
            if t >= 24 * 30 and vov_filled[t] > vov_q80[t]:
                m_volov = 0.5
            m_dd = dd.step(last_net)
            m_regime = m_vol * m_volov * m_dd * (0.5 if t < freeze_until else 1.0)
        regime_sum += m_regime

        # ---- position with (optional) online correction
        xt = np.concatenate([[1.0], X[t]])
        d = readout.predict(xt) if use_rls else 0.0
        z_ideal = float(np.tanh(z_arr[t] + d)) * m_regime
        if use_buffer:
            z_new = buffer.apply(z_cur, z_ideal, avg_abs_z, cost_mult)
        else:
            z_new = float(np.clip(z_ideal, -2.0, 2.0))
        z_new = float(np.clip(z_new, -2.0, 2.0))
        cost = COST_R * cost_mult * abs(z_new - z_cur)
        net = z_cur * r_arr[t] - cost
        last_net = net
        avg_abs_z += (abs(z_new) - avg_abs_z) / max(1, t + 1)
        z_cur = z_new
        nets.append(net)
        zs.append(z_new)

        # ---- queue this origin's label (arrives at t + H)
        y = r_arr[t] / (s_arr[t] + 1e-12)
        pending.append((t + label_horizon, xt, y))

    z_out = np.asarray(zs)
    net_out = np.asarray(nets)
    return {
        "net": pd.Series(net_out, index=idx),
        "z": pd.Series(z_out, index=idx),
        "turnover_py": turnover_per_year(z_out),
        "metrics": ann_metrics(net_out),
        "diagnostics": {"ph_alarms": alarms,
                        "mean_regime_mult": round(regime_sum / max(1, len(idx)), 4),
                        "cost_drag_py": round(float(
                            COST_R * cost_mult * np.abs(np.diff(z_out)).mean()
                            * BARS_PER_YEAR), 6)},
    }


def frozen_net(z: pd.Series, r: pd.Series, sigma: pd.Series,
               cost_mult: float = 1.0) -> dict:
    """The no-engine floor: hold z exactly as the model outputs it, pay costs
    on every exposure change (the anti-buffer).

    v2.0.1 CAUSALITY FIX (regression-tested in K6 gate G4.7): the position
    z[t] is decided from information through the CLOSE of bar t, so it can
    only earn the return of bar t+1.  The original formulation credited
    z[t] * r[t] (same bar), a one-bar lookahead that inflated every frozen
    accounting number (Channel_A baseline: Sharpe 15.02 -> -0.13 after the
    fix).  Fixed convention, identical to the engine loop (net = z_cur *
    r_arr[t]): the exposure HELD during bar t is w[t-1]; the cost of moving
    to w[t] is charged at bar t.
    """
    idx = z.index
    zv = z.values.astype("float64")
    rv = r.reindex(idx).values
    sv = sigma.reindex(idx).values
    v = (SIGMA_TGT_ANN / math.sqrt(BARS_PER_YEAR)) / (sv + 1e-12)
    w = zv * v
    w_lag = np.concatenate([[0.0], w[:-1]])          # held during bar t
    dzv = np.abs(np.diff(w, prepend=0.0))            # trade at close of bar t
    net = w_lag * rv - COST_R * cost_mult * dzv
    return {"net": pd.Series(net, index=idx),
            "turnover_py": turnover_per_year(zv),
            "metrics": ann_metrics(net)}


# ----------------------------------------------------------------------------
# Master Trial Ledger
# ----------------------------------------------------------------------------
class TrialLedger:
    """Every configuration evaluated, from Experiment #1 (DSR accounting)."""

    def __init__(self, path: str):
        self.path = path
        self.rows = []
        if os.path.isfile(path):
            with open(path) as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        self.rows.append(json.loads(ln))

    def add(self, model: str, config: dict, fit_metric: float | None = None,
            stage: str = "search", kernel: str = "", notes: str = ""):
        cfg_hash = hashlib.md5(json.dumps(config, sort_keys=True,
                                          default=str).encode()).hexdigest()[:10]
        row = {"model": model, "config": config, "config_hash": cfg_hash,
               "fit_metric": fit_metric, "stage": stage, "kernel": kernel,
               "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "notes": notes}
        self.rows.append(row)
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return row

    def unique_configs(self, model: str | None = None) -> int:
        seen = set()
        for r in self.rows:
            if model is None or r["model"] == model:
                seen.add((r["model"], r["config_hash"]))
        return len(seen)

    def fit_sharpes(self) -> np.ndarray:
        return np.array([r["fit_metric"] for r in self.rows
                         if isinstance(r.get("fit_metric"), (int, float))
                         and np.isfinite(r["fit_metric"])])


# ----------------------------------------------------------------------------
# block bootstrap CI
# ----------------------------------------------------------------------------
def block_bootstrap_ci(net: np.ndarray, n_boot: int = 5000, block: int = 24,
                       seed: int = 1337) -> dict:
    """Moving-block bootstrap on the per-bar net return stream: CI for the
    annualized Sharpe (and mean).  Non-overlapping 24-bar blocks (1 trading
    day), resampled with replacement.

    v2.0.1: chunk-vectorized (identical math and RNG draw order to the
    original per-boot loop; rng.integers(size=(k, n_blocks)) consumes the
    PCG64 stream in the same row-major order as k consecutive
    rng.integers(0, n_blocks, n_blocks) calls)."""
    rng = np.random.default_rng(seed)
    net = np.asarray(net, dtype="float64")
    net = net[np.isfinite(net)]
    T = len(net)
    n_blocks = T // block
    if n_blocks < 4:
        return {"sharpe_ci": [None, None], "n_boot": 0}
    blocks = net[: n_blocks * block].reshape(n_blocks, block)
    stats = np.empty(n_boot)
    chunk = 50
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        sel = rng.integers(0, n_blocks, size=(k, n_blocks))
        sample = blocks[sel].reshape(k, n_blocks * block)
        mu = sample.mean(axis=1)
        sd = sample.std(axis=1, ddof=1)
        stats[done: done + k] = mu / (sd + 1e-12) * math.sqrt(BARS_PER_YEAR)
        done += k
    return {"sharpe_ci": [round(float(np.percentile(stats, 2.5)), 3),
                          round(float(np.percentile(stats, 97.5)), 3)],
            "sharpe_p05": round(float(np.percentile(stats, 5)), 3),
            "n_boot": n_boot, "block": block}


def self_test() -> None:
    rng = np.random.default_rng(7)
    # fold_indices sanity
    idx = pd.date_range("2014-01-01", "2026-01-01", freq="h", tz="UTC")
    df = pd.DataFrame(index=idx)
    for y in (2016, 2020, 2025):
        res = fold_indices(df, y)
        assert res is not None
        fit, es, test = res
        assert idx[test[0]].year == y and idx[test[-1]].year == y
        assert idx[fit[-1]] < idx[test[0]] - pd.Timedelta("24h")
        assert idx[es[0]] > idx[fit[-1]]
    # metrics
    net = rng.standard_normal(5000) * 0.002 + 0.0001
    m = ann_metrics(net)
    assert 2.0 < m["sharpe_ann"] < 6.0
    eq_path = np.concatenate([np.linspace(0, 1, 2500), np.linspace(1, 0.7, 2500)])
    m2 = ann_metrics(np.diff(eq_path) * 10)
    assert m2["maxdd_volunits"] > 0
    # bootstrap
    ci = block_bootstrap_ci(net, n_boot=300)
    assert ci["sharpe_ci"][0] < m["sharpe_ann"] < ci["sharpe_ci"][1] + 2
    # ledger
    led = TrialLedger("/tmp/_test_ledger.jsonl")
    led.add("m_test", {"a": 1}, 0.5)
    led.add("m_test", {"a": 1}, 0.6)
    led.add("m_test", {"a": 2}, 0.4)
    assert led.unique_configs() == 2
    assert len(led.fit_sharpes()) == 3
    os.remove("/tmp/_test_ledger.jsonl")
    # engine runner smoke (with toggles)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    T = 3000
    idx2 = pd.date_range("2020-01-01", periods=T, freq="h", tz="UTC")
    sig = pd.Series(0.001, index=idx2)
    r = pd.Series(0.001 * rng.standard_normal(T), index=idx2)
    # a score with mild foresight-free autocorrelation
    zmod = pd.Series(np.tanh(0.3 * np.convolve(r.values, np.ones(8) / 8, "same")),
                     index=idx2)
    out = run_online_engine(zmod, r, sig, cost_mult=1.0, use_buffer=True)
    assert np.isfinite(out["net"]).mean() > 0.99
    out_nb = run_online_engine(zmod, r, sig, use_buffer=False)
    assert out_nb["turnover_py"] >= out["turnover_py"], "buffer must cut turnover"
    out_frz = run_online_engine(zmod, r, sig, use_rls=False)
    assert np.isfinite(out_frz["net"]).all()
    fr = frozen_net(zmod, r, sig)
    assert np.isfinite(fr["net"]).all()
    print("v2_common self-test passed: folds, metrics, bootstrap, ledger, engine toggles.")
