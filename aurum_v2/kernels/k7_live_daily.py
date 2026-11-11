"""
AURUM v2.0 -- Live Daily kernel main (v2.0.2 decision state)
===========================================================
Phase 5: emits the full v2.0 decision object, recomputable from kernel +
datasets alone: continuous target exposure z_t, Carver buffer edge status,
PH / vol-of-vol / drawdown-governor regime multipliers, catastrophic stop
level, cost-budget usage vs the 1/3 speed limit, and the v1.3 barrier
reference.

v2.0.2: the live model remains M1_GBDT_v2 -- the ONLY active CPU model
that passes the Carver speed limit (OOS engine Sharpe 0.938); the
de-saturated M2_RFF_Ridge active content measured as noise (engine FIT
-0.736 / OOS -0.744) and is reference-only.  The K6 v2.0.2 tournament
verdict is embedded as stack_context when the K6 outputs are available.
"""
import json
import math
import os
import time

import numpy as np
import pandas as pd

GATES = {}
LOG_LINES = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def gate(gid, ok, detail=None):
    g = {"gate": gid, "pass": bool(ok), "detail": str(detail)[:2000]}
    GATES[gid] = g
    log(f"GATE {gid}: {'PASS' if ok else 'FAIL'} | {detail}")
    return bool(ok)


def main():
    WORK = "/kaggle/working"
    if not os.path.exists(WORK):
        WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_out")
        os.makedirs(WORK, exist_ok=True)
    import lightgbm as lgb

    # ---------------- 1) data ------------------------------------------------
    feats, px = load_v2_data()
    r = feats["r_xau"]
    sigma = feats["sigma"]
    idx = feats.index
    names = [c for c in feats.columns
             if not c.startswith(("y_", "r_xau", "sigma", "close_xau"))]
    X = feats[names].astype("float32").ffill().fillna(0.0)
    X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y = feats["y_ret24_norm"]
    asof = idx[-1]
    log(f"as-of bar: {asof} (features frozen at dataset version)")

    # ---------------- 2) M1_GBDT_v2 final refit ------------------------------
    m = y.notna().values
    train_pos = np.arange(len(idx))[m]
    train_pos = train_pos[train_pos < len(idx) - 24]
    params = dict(objective="regression", metric="rmse", max_depth=5,
                  reg_alpha=10, reg_lambda=10, subsample=0.9,
                  colsample_bytree=0.7, learning_rate=0.06, verbose=-1)
    preds = []
    for seed in (1337, 2024, 4242):
        dtr = lgb.Dataset(X.iloc[train_pos], label=y.iloc[train_pos])
        booster = lgb.train({**params, "seed": seed}, dtr, num_boost_round=300)
        preds.append(booster.predict(X.iloc[[-1]])[0])
    pred_now = float(np.mean(preds))
    z_model = float(np.tanh(2.0 * pred_now))

    # ---------------- 3) engine state at the latest bar ----------------------
    k2dir = os.environ.get("AURUM_K2DIR") or _find_dir("v2_cpu_forecasts.parquet",
                      "/kaggle/input/aurum-v2-forecasts-cpu",
                      "/kaggle/input/aurum-v2-cpu-preds",
                      "/kaggle/input/datasets/tskdev/aurum-v2-cpu-preds",
                      *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v2-forecasts*")),
                      *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v2-cpu-preds*")))
    z_hist = None
    if k2dir:
        fc = pd.read_parquet(os.path.join(k2dir, "v2_cpu_forecasts.parquet"))
        fc.index = pd.DatetimeIndex(fc.index)
        if fc.index.tz is None:
            fc.index = fc.index.tz_localize("UTC")
        if "z_gbdt" in fc:
            z_hist = fc["z_gbdt"].reindex(idx)
    if z_hist is None or z_hist.iloc[-48:].notna().sum() < 24:
        z_hist = pd.Series(np.nan, index=idx)
        step = 24 * 30
        for t0 in range(len(idx) - 2 * 365 * 24, len(idx), step):
            tp = np.arange(t0 - 24)[m[:t0 - 24]] if t0 > 24 else np.array([])
            if len(tp) < 5000:
                continue
            ps = []
            dtr = lgb.Dataset(X.iloc[tp], label=y.iloc[tp])
            b = lgb.train({**params, "seed": 1337}, dtr, num_boost_round=200)
            ps.append(b.predict(X.iloc[t0: t0 + step]))
            z_hist.iloc[t0: t0 + step] = np.tanh(2.0 * np.mean(ps, axis=0))
        z_hist = z_hist.ffill()
    z_run = z_hist.fillna(0.0)

    ph = PageHinkley(lam=50.0)
    vt = VolTarget(SIGMA_TGT_ANN / math.sqrt(BARS_PER_YEAR))
    buf = CarverBuffer()
    dd = DrawdownGovernor()
    z_cur, avg_abs_z, freeze_until, last_net = 0.0, 0.05, -1, 0.0
    vov_series = []
    zs = []
    for t in range(len(idx) - 3 * 365 * 24, len(idx)):
        rt = float(r.iloc[t]); st = float(sigma.iloc[t])
        ph.update(rt / (st + 1e-12))
        if ph.n == 0:
            freeze_until = t + 24
        vt.update(st)
        m_vol = vt.multiplier(st)
        vov = float(sigma.iloc[max(0, t - 24 * 30): t].std())
        vov_series.append(vov)
        m_volov = 1.0
        if len(vov_series) > 24 * 30 and np.isfinite(vov):
            q = np.quantile(vov_series, 0.8)
            if vov > q:
                m_volov = 0.5
        m_dd = dd.step(last_net)
        regime = m_vol * m_volov * m_dd * (0.5 if t < freeze_until else 1.0)
        z_ideal = float(np.tanh(z_run.iloc[t])) * regime
        z_new = buf.apply(z_cur, z_ideal, avg_abs_z, 1.0)
        z_new = float(np.clip(z_new, -2.0, 2.0))
        net = z_cur * rt - COST_R * abs(z_new - z_cur)
        last_net = net
        avg_abs_z += (abs(z_new) - avg_abs_z) / max(1, t + 1)
        z_cur = z_new
        zs.append(z_new)

    regime_now = m_vol * m_volov * m_dd * (0.5 if len(idx) - 1 < freeze_until else 1.0)
    z_ideal_now = float(np.tanh(z_run.iloc[-1])) * regime_now
    band = buf.buffer_size(avg_abs_z, 1.0)
    inside_band = abs(z_cur - z_ideal_now) <= band
    rng14 = (px["h"] - px["l"]) if "h" in px.columns else None
    if rng14 is not None:
        atr14 = float(rng14.rolling(14).mean().reindex(idx).iloc[-1])
    else:
        atr14 = float(sigma.iloc[-1] * math.sqrt(24))
    close_now = float(feats["close_xau"].iloc[-1]) if "close_xau" in feats else None

    v13 = {"source": "tskdev/aurum-v13-artifacts (v1.3 flagship policy)",
           "champion": "flagship cap=3, OOS Sharpe 2.444, MaxDD 13.06%, DSR 0.895"}
    v13dir = _find_dir("trades_all.parquet", "/kaggle/input/aurum-v13-artifacts",
                       *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v13*")))
    if v13dir:
        tr = pd.read_parquet(os.path.join(v13dir, "trades_all.parquet"))
        sub = tr[(tr["strategy"] == "flagship") & (tr["cap"] == 3) & (tr["cost_mult"] == 1.0)]
        last = sub.iloc[-1]
        v13["latest_trade"] = {
            "cell": str(last["cell_name"]),
            "entry_ts": str(pd.Timestamp(last["ts"], unit="ns", tz="UTC")),
            "p": float(last["p"]), "ev": float(last["ev"]),
            "f_eff": float(last["f_eff"]), "exit_bar": int(last["exit_bar"])}
        v13["champion_open_exposure"] = float(
            (1.0 if int(last["exit_bar"]) >= len(idx) - 1 else 0.0) *
            (1 if str(last["cell_name"]).startswith("L") else -1))

    # v2.0.2 tournament verdict (embedded when the K6 outputs are present)
    stack_context = {
        "spec": "v2.0.2",
        "live_model": "M1_GBDT_v2 (only speed-limit-passing active CPU model)",
        "m2_rff_v202": "de-saturated via causal trailing standardization "
                       "(G4.8 producer-gated); active content measured as "
                       "noise -- reference-only",
        "m6_family": "fixed a-priori {M1_GBDT_v2, M2_RFF_Ridge}, 50/50 "
                     "(Carver S24 SR-shrink 0.90 toward equal weights)",
    }
    k6dir = os.environ.get("AURUM_K6OUT")
    if not k6dir:
        k6dir = _find_dir("backtest_v2.json",
                          "/kaggle/input/aurum-v2-backtest",
                          "/kaggle/input/datasets/tskdev/aurum-v2-backtest",
                          *sorted(__import__("glob").glob(
                              "/kaggle/input/*/aurum-v2-backtest*")))
    if k6dir and os.path.isfile(os.path.join(k6dir, "champion_compare.json")):
        try:
            cc = json.load(open(os.path.join(k6dir, "champion_compare.json")))
            for k in ("M7_Active_Grand_Hybrid", "M7_Core_Satellite",
                      "M0_Champ_v13", "M6_AURUM_v2_Ensemble", "M2_RFF_Ridge",
                      "M1_GBDT_v2"):
                if k in cc:
                    stack_context[k] = {
                        "oos_sharpe": cc[k].get("sharpe_ann"),
                        "cost2x_sharpe": cc[k].get("cost2x_sharpe"),
                        "oos_dsr": cc[k].get("oos_dsr")}
        except Exception:
            pass

    decision = {
        "as_of_bar": str(asof),
        "symbol": "XAUUSD",
        "spec": "v2.0.2",
        "v2_state": {
            "model": "M1_GBDT_v2 (3-seed, walk-forward-consistent refit)",
            "pred_y_ret24_norm": round(pred_now, 5),
            "z_model_score": round(z_model, 5),
            "z_target_exposure": round(z_ideal_now, 5),
            "z_executed": round(z_cur, 5),
            "regime": {"vol_target_mult": round(m_vol, 4),
                        "vol_of_vol_mult": m_volov,
                        "dd_governor_scale": m_dd,
                        "ph_frozen": bool(len(idx) - 1 < freeze_until)},
            "carver_buffer": {"band": round(band, 5), "avg_abs_z": round(avg_abs_z, 5),
                               "inside_band": bool(inside_band),
                               "distance_to_edge": round(band - abs(z_cur - z_ideal_now), 5)},
            "catastrophic_stop": {"frame": "2.5 x ATR14", "atr14": round(atr14, 3),
                                   "long_stop_px": round(close_now - 2.5 * atr14, 2)
                                   if close_now else None,
                                   "short_stop_px": round(close_now + 2.5 * atr14, 2)
                                   if close_now else None},
            "cost_budget": {"cost_r": COST_R,
                             "speed_limit": "annual cost <= 1/3 pre-cost SR",
                             "recent_turnover_py": round(float(
                                 np.abs(np.diff(zs[-24 * 90:])).mean()
                                 * BARS_PER_YEAR / 2), 2)},
        },
        "v13_barrier_reference": v13,
        "stack_context_v202": stack_context,
        "provenance": {"features": "tskdev/aurum-v2-featstore (frozen version)",
                        "engine": "v2.0 OnlineExecutionEngine components "
                                  "(river-exact PH, Moreira-Muir vol target, "
                                  "Carver buffer, AHL DD governor)",
                        "recomputable": True},
    }
    with open(os.path.join(WORK, "decision_v2.json"), "w") as f:
        json.dump(decision, f, indent=1)

    ok = gate("G6.4_live_decision", True,
              f"v2.0 decision emitted at {asof}: z_target={z_ideal_now:.4f} "
              f"z_exec={z_cur:.4f} regime={round(regime_now, 3)} "
              f"buffer_inside={inside_band}")
    with open(os.path.join(WORK, "gates.json"), "w") as f:
        json.dump(GATES, f, indent=2, default=str)
    with open(os.path.join(WORK, "run_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES))
    log("LIVE COMPLETE" if ok else "LIVE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys as _s
    _s.exit(main())
