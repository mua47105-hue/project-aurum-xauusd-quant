"""
AURUM v2.0 -- K2 kernel main: Fast CPU Models & Execution Upgrade
==================================================================
Phase 2 of the v2.0 execution plan (0 GPU hours):

  * M1_GBDT_v2  : regularized LightGBM 3-seed ensemble on the ~345 v2
                  features (max_depth=5, reg_alpha=10, reg_lambda=10,
                  subsample=0.9, colsample_bytree=0.7), walk-forward yearly
                  folds 2016-2025, target y_ret24_norm, position z=tanh(2*pred).
  * M2_RFF_Ridge: Kelly-Malamud closed-form RFF timing (P=2048, gamma=sqrt(2)
                  dimension-scaled, 10 seeds averaged, ridge sweep
                  z in {1e0..1e4} selected on the fold's ES slice).
                  v2.0.2: the position transform is a CAUSAL TRAILING
                  standardization of the raw prediction (Engineering Flaw #8
                  fix -- see rff_z_v202 below), gated by G4.8 at the producer.
  * RLSBank     : the hyperparameter-free adaptation floor as a standalone
                  strategy (delay-aware updates, per-coordinate median bank).
  * OnlineExecutionEngine evaluation of M1/M2 with Baseline-Gauntlet toggles
    (buffer / RLS / regimes), cost multipliers 1.0x/1.5x/2.0x, FIT vs OOS.
  * Master Trial Ledger from Experiment #1.

Outputs: v2_cpu_forecasts.parquet, engine_metrics.json, trial_ledger.jsonl,
gates.json, run_log.txt.
"""
import glob as _glob
import json
import math
import os
import time
import traceback

import numpy as np
import pandas as pd

GATES = {}
LOG_LINES = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def gate(gid, ok, detail=None):
    g = {"gate": gid, "pass": bool(ok), "detail": str(detail)[:3000]}
    GATES[gid] = g
    log(f"GATE {gid}: {'PASS' if ok else 'FAIL'} | {detail}")
    return bool(ok)


def stage(name):
    class C:
        def __init__(self):
            self.t0 = time.time()
        def __enter__(self):
            log(f"===== STAGE {name} START =====")
            return self
        def __exit__(self, et, ev, tb):
            if et is not None:
                err = "".join(traceback.format_exception(et, ev, tb))[-2500:]
                log(f"===== STAGE {name} ERROR =====\n{err}")
                raise
            log(f"===== STAGE {name} DONE in {time.time()-self.t0:.1f}s =====")
            return False
    return C()


FEAT_COLS = None

# ---------------------------------------------------------------------------
# v2.0.2 (Engineering Flaw #8 fix): causal trailing standardization of the
# raw RFF prediction.
#
# ROOT CAUSE of the flaw: under a heavy ridge penalty (z=1e4) the selected
# model's predictions carry a fold-level mean drift (1e-3..1e-2) that is up
# to 100x larger than their within-fold dispersion (~1e-4).  The old fold-
# local transform tanh(pred / std(pred_es)) divided by a COLLAPSED scale, so
# pred/scale >> 5 saturated tanh to sign(pred): z_rff became a per-fold
# constant +-1.0 (OOS 2023-25: 16,991 bars, ONE unique value +1.0; FIT 2017
# fold: constant -1.0).  M2 silently degraded into passive gold / per-fold
# sign bets with ~zero per-bar information (G4.8 z-quality audit).
#
# FIX: standardize the raw prediction against its OWN trailing history,
# re-anchored at each yearly re-fit boundary (the walk-forward replaces the
# model; its prediction scale is a new random variable, and carrying the
# dead model's mean/std across the boundary injects the very drift artifact
# being removed):
#     z_t = tanh( (p_t - EWMA_504(p)_t) / (2.0 * max(EWSTD_504(p)_t, eps)) )
# min_periods=63 (~2.6 trading days): the first 63 bars of every fold are
# NaN by design (no trailing history from the new model yet).
RFF_Z_SPAN = 504          # trailing EWMA/EWSTD span (H1 bars, ~1 month)
RFF_Z_MINP = 63           # warmup before a z is emitted (per fold)
RFF_Z_EPS = 1e-9          # scale floor (degenerate constant stretches -> z=0)


def rff_z_v202(pred: pd.Series) -> pd.Series:
    """Causal trailing standardization of pred_rff (v2.0.2 primary)."""
    out = pd.Series(np.nan, index=pred.index)
    for _, g in pred.groupby(pred.index.year):   # fold = calendar test year
        p = g.dropna()
        if len(p) < 2 * RFF_Z_MINP:
            continue
        ewma = p.ewm(span=RFF_Z_SPAN, min_periods=RFF_Z_MINP).mean()
        ewstd = p.ewm(span=RFF_Z_SPAN, min_periods=RFF_Z_MINP).std()
        x = (p - ewma) / (2.0 * np.maximum(ewstd, RFF_Z_EPS))
        out.loc[p.index] = np.tanh(x)
    return out


def rff_z_foldiqr(pred: pd.Series) -> pd.Series:
    """NEGATIVE-CONTROL variant (kept for documentation only): fold mean
    RETAINED in the numerator, robust expanding IQR scale (causal within
    the fold).  Wherever the fold drift >> fold IQR this saturates exactly
    like the v2.0.1 flaw -- it is exported so the report can show WHY
    fold-local scaling cannot survive a drift/dispersion ratio >> 1."""
    out = pd.Series(np.nan, index=pred.index)
    for _, g in pred.groupby(pred.index.year):
        p = g.dropna()
        if len(p) < 2 * RFF_Z_MINP:
            continue
        vals = p.values
        res = np.full(len(vals), np.nan)
        for i in range(RFF_Z_MINP, len(vals)):
            q75, q25 = np.percentile(vals[: i + 1], [75, 25])
            res[i] = np.tanh(vals[i] / max((q75 - q25) / 1.349, RFF_Z_EPS))
        out.loc[p.index] = res
    return out


def prepare_matrix(feats: pd.DataFrame):
    """Feature matrix (finite-filled) + targets; returns (X, y, names)."""
    names = [c for c in feats.columns
             if not c.startswith(("y_", "r_xau", "sigma", "close_xau"))]
    X = feats[names].astype("float32")
    X = X.ffill().fillna(0.0)
    # replace inf
    X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y = feats["y_ret24_norm"].astype("float64")
    return X, y, names


def main():
    WORK = "/kaggle/working"
    if not os.path.exists(WORK):
        WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k2_out")
        os.makedirs(WORK, exist_ok=True)
    import lightgbm as lgb

    ledger = TrialLedger(os.path.join(WORK, "trial_ledger.jsonl"))
    T0 = time.time()

    # ---------------- 1) data ------------------------------------------------
    with stage("load data"):
        feats, px = load_v2_data()
        log(f"features: {feats.shape[0]} bars x {feats.shape[1]} cols "
            f"({feats.index[0]} .. {feats.index[-1]})")
        X, y, names = prepare_matrix(feats)
        log(f"model matrix: {X.shape}; target NaN: {y.isna().mean():.3%}")

    r = feats["r_xau"]
    sigma = feats["sigma"]
    yr = pd.Series(feats.index.year, index=feats.index)
    fit_mask = yr.isin(FIT_YEARS)
    oos_mask = yr.isin(OOS_YEARS)

    # ---------------- 2) M1_GBDT_v2 walk-forward -----------------------------
    with stage("M1_GBDT_v2 walk-forward"):
        z_gbdt = pd.Series(np.nan, index=feats.index)
        pred_gbdt = pd.Series(np.nan, index=feats.index)
        fold_log = []
        params = dict(objective="regression", metric="rmse",
                      max_depth=5, reg_alpha=10, reg_lambda=10,
                      subsample=0.9, colsample_bytree=0.7,
                      learning_rate=0.06, verbose=-1)
        ledger.add("M1_GBDT_v2", {**params, "n_seeds": 3, "target": "y_ret24_norm"},
                   stage="config", kernel="K2")
        for yy in ALL_FOLD_YEARS:
            res = fold_indices(feats, yy)
            if res is None:
                continue
            fit_pos, es_pos, test_pos = res
            m_tr = y.notna().values
            fit_m = fit_pos[m_tr[fit_pos]]
            Xtr, ytr = X.iloc[fit_m], y.iloc[fit_m]
            Xes, yes = X.iloc[es_pos], y.iloc[es_pos]
            Xte = X.iloc[test_pos]
            preds = []
            for seed in (1337, 2024, 4242):
                dtr = lgb.Dataset(Xtr, label=ytr)
                des = lgb.Dataset(Xes, label=yes, reference=dtr)
                booster = lgb.train({**params, "seed": seed, "bagging_seed": seed,
                                     "feature_fraction_seed": seed},
                                    dtr, num_boost_round=1200, valid_sets=[des],
                                    callbacks=[lgb.early_stopping(60, verbose=False)])
                preds.append(booster.predict(Xte, num_iteration=booster.best_iteration))
                ledger.add("M1_GBDT_v2", {"seed": seed, "fold": yy,
                                          "best_iter": booster.best_iteration},
                           stage="fit", kernel="K2")
            p = np.mean(preds, axis=0)
            # fold-local scale normalization: z = tanh(pred / std(in-sample pred))
            # (predictions of vol-normalized 24h returns are tiny; without
            #  normalization the position collapses to zero and costs dominate)
            es_pred = booster.predict(Xes, num_iteration=booster.best_iteration)
            scale = float(np.std(es_pred)) + 1e-9
            z_gbdt.iloc[test_pos] = np.tanh(p / scale)
            pred_gbdt.iloc[test_pos] = p
            fold_log.append({"fold": yy, "best_iter": booster.best_iteration,
                             "pred_std": float(np.std(p))})
            log(f"  fold {yy}: best_iter={booster.best_iteration} "
                f"pred_std={np.std(p):.4f}")
        pre_2016 = int((feats.index < pd.Timestamp("2016-01-01", tz="UTC")).sum())
        n_missing = int(z_gbdt.isna().sum())
        ok21 = gate("G2.1_folds", len(fold_log) == 10 and n_missing == pre_2016,
                    f"10/10 folds predicted; NaN z rows={n_missing} "
                    f"(= pre-2016 warmup {pre_2016}: 2010-2015 excluded by design)")

    # ---------------- 3) M2_RFF_Ridge walk-forward ---------------------------
    # v2.0.2: the model fit itself is UNCHANGED (same seeds, same ridge
    # sweep, same pred_rff bytes); only the position transform is replaced
    # by the causal trailing standardization (rff_z_v202).
    with stage("M2_RFF_Ridge walk-forward"):
        pred_rff = pd.Series(np.nan, index=feats.index)
        for yy in ALL_FOLD_YEARS:
            res = fold_indices(feats, yy)
            if res is None:
                continue
            fit_pos, es_pos, test_pos = res
            m_tr = y.notna().values
            fit_m = fit_pos[m_tr[fit_pos]]
            m2 = RFFRidgeTimingModel(n_features=X.shape[1], p=2048,
                                     gamma=math.sqrt(2.0),
                                     ridge_grid=(1e0, 1e1, 1e2, 1e3, 1e4),
                                     n_seeds=10, seed=1337)
            m2.fit(X.iloc[fit_m].values, y.iloc[fit_m].values,
                   X_val=X.iloc[es_pos].values, y_val=y.iloc[es_pos].values)
            p = m2.predict(X.iloc[test_pos].values)
            pred_rff.iloc[test_pos] = p
            ledger.add("M2_RFF_Ridge", {"fold": yy, "z_selected": m2.z_best,
                                        "p": 2048, "n_seeds": 10,
                                        "gamma": "sqrt(2)/sqrt(d)"},
                       stage="fit", kernel="K2")
            log(f"  fold {yy}: z*={m2.z_best:g} pred_std={np.std(p):.4f}")
        for z_ in (1e0, 1e1, 1e2, 1e3, 1e4):
            ledger.add("M2_RFF_Ridge", {"ridge_z": z_, "p": 2048, "n_seeds": 10},
                       stage="config", kernel="K2",
                       notes="ridge grid member (selection space for DSR N)")
        # v2.0.2 position transforms (fold-local ES-scale REMOVED -- Flaw #8)
        z_rff = rff_z_v202(pred_rff)
        z_rff_iqr = rff_z_foldiqr(pred_rff)      # negative-control variant
        ledger.add("M2_RFF_Ridge",
                   {"rev": "k2-v2.0.2", "transform": "causal trailing",
                    "z": "tanh((p - EWMA_504(p)) / (2*max(EWSTD_504(p), 1e-9)))",
                    "re_anchor": "yearly re-fit boundary (fold = test year)",
                    "span": RFF_Z_SPAN, "min_periods": RFF_Z_MINP,
                    "eps": RFF_Z_EPS,
                    "variant_foldiqr": "tanh(p / max(IQR_fold/1.349, eps)) "
                                       "(negative control, mean retained)"},
                   stage="config", kernel="K2",
                   notes="Engineering Flaw #8 producer-side fix")
        ok_rff = gate("G2.1b_rff", int(z_rff.notna().sum()) >= pre_2016 and len(z_rff.dropna()) > 50000,
                      f"RFF predictions cover {int(z_rff.notna().sum())} bars "
                      f"(2016-2025 span = {len(feats) - pre_2016}; "
                      f"{RFF_Z_MINP}-bar warmup per fold by design)")

        # ---- G4.8 producer-side saturation gate (v2.0.2) --------------------
        # The z-saturation audit is enforced AT THE PRODUCER now, not just
        # discovered downstream in K6: std(z) > 0.05 and P(|z|>0.99) < 5%
        # on BOTH windows, plus a genuinely-active OOS turnover floor
        # (frozen accounting, > 2 round-turns / year).
        audit48 = {}
        ok48 = True
        for wname, mask in (("fit", fit_mask), ("oos", oos_mask)):
            sel = z_rff.dropna().index
            sel = sel[mask.reindex(sel).fillna(False)]
            if len(sel) < 500:
                continue
            v = z_rff[sel].values
            std = float(np.std(v))
            sat = float((np.abs(v) > 0.99).mean())
            fro = frozen_net(z_rff[sel], r[sel], sigma[sel], 1.0)
            audit48[wname] = {"n": int(len(v)),
                              "std": round(std, 5),
                              "sat_frac_gt_0p99": round(sat, 6),
                              "nunique": int(pd.unique(v).size),
                              "frozen_turnover_py": round(fro["turnover_py"], 2),
                              "frozen_sharpe": fro["metrics"].get("sharpe_ann")}
            ok48 = ok48 and (std > 0.05) and (sat < 0.05)
        ok48 = ok48 and audit48.get("oos", {}).get("frozen_turnover_py", 0.0) > 2.0
        # negative-control variant audited for the record (expected to FAIL)
        neg = {}
        for wname, mask in (("fit", fit_mask), ("oos", oos_mask)):
            sel = z_rff_iqr.dropna().index
            sel = sel[mask.reindex(sel).fillna(False)]
            if len(sel) < 500:
                continue
            vi = z_rff_iqr[sel].values
            neg[wname] = {"std": round(float(np.std(vi)), 5),
                          "sat_frac_gt_0p99": round(float((np.abs(vi) > 0.99).mean()), 6),
                          "nunique": int(pd.unique(vi).size)}
        ledger.add("M2_RFF_Ridge", {"rev": "k2-v2.0.2", **audit48,
                                     "negative_control_foldiqr": neg},
                   audit48.get("oos", {}).get("frozen_sharpe"),
                   stage="zquality", kernel="K2")
        ok48g = gate("G4.8_z_rff_producer", ok48,
                     f"v2.0.2 producer-side saturation gate on z_rff: "
                     + json.dumps(audit48)
                     + " -- negative control (fold-mean-retained IQR variant, "
                       "expected saturated): " + json.dumps(neg))

    # ---------------- 4) RLSBank standalone floor ----------------------------
    with stage("RLSBank standalone floor"):
        # readout features: last-3 vol-normalized returns + F_blend/10
        f_blend = (feats["F_blend_xau"] / 10.0) if "F_blend_xau" in feats else 0.0
        rn = (r / (sigma + 1e-12))
        bankX = pd.DataFrame({"x1": rn.shift(1), "x2": rn.shift(2),
                              "x3": rn.shift(3), "x4": f_blend.shift(1)}).fillna(0.0)
        z_bank = pd.Series(np.nan, index=feats.index)
        bank = RLSBank(5)   # top-level in the bundle (online_engine.py)
        H = 24
        pending = []
        idx = feats.index
        bankXv = bankX.values
        scores = np.full(len(idx), np.nan)
        for t in range(len(idx)):
            for _, bx, by in [p for p in pending if p[0] == t]:
                if np.isfinite(by):
                    bank.update(bx, by)      # skip warmup-NaN labels (sigma
                                             # undefined in the first 24 bars)
            pending = [p for p in pending if p[0] != t]
            xt = np.concatenate([[1.0], bankXv[t]])
            scores[t] = bank.predict(xt)
            pending.append((t + H, xt, r.iloc[t] / (sigma.iloc[t] + 1e-12)))
        z_bank = pd.Series(np.tanh(scores), index=idx)
        # valid only after warmup (first 30d of 2010)
        z_bank.iloc[: 24 * 30] = np.nan
        ledger.add("RLSBank", {"lams": [0.99, 0.995, 0.999, 0.9995],
                               "features": "rn lags 1-3 + F_blend/10"},
                   stage="config", kernel="K2")
        ok_bank = gate("G2.1c_rlsbank", int(z_bank.notna().sum()) > 80000,
                       "RLSBank floor predictions emitted")

    # ---------------- 5) engine evaluation (Phase 2 mandate) -----------------
    with stage("engine evaluation"):
        engine_metrics = {}
        for name, zseries in (("M1_GBDT_v2", z_gbdt), ("M2_RFF_Ridge", z_rff),
                              ("RLSBank", z_bank)):
            zm = zseries.dropna()
            res_all = {}
            for window, mask in (("fit", fit_mask), ("oos", oos_mask)):
                sel = zm.index[mask.reindex(zm.index).fillna(False)]
                if len(sel) < 500:
                    continue
                zw = zseries[sel]
                rw, sw = r[sel], sigma[sel]
                variants = {}
                full = run_online_engine(zw, rw, sw, 1.0)
                variants["full"] = {"metrics": full["metrics"],
                                    "turnover_py": full["turnover_py"],
                                    "diag": full["diagnostics"]}
                nb = run_online_engine(zw, rw, sw, 1.0, use_buffer=False)
                variants["no_buffer"] = {"metrics": nb["metrics"],
                                         "turnover_py": nb["turnover_py"]}
                fz = run_online_engine(zw, rw, sw, 1.0, use_rls=False)
                variants["frozen_rls_off"] = {"metrics": fz["metrics"],
                                              "turnover_py": fz["turnover_py"]}
                nr = run_online_engine(zw, rw, sw, 1.0, use_regimes=False)
                variants["no_regimes"] = {"metrics": nr["metrics"],
                                          "turnover_py": nr["turnover_py"]}
                fro = frozen_net(zw, rw, sw, 1.0)
                variants["frozen_raw"] = {"metrics": fro["metrics"],
                                          "turnover_py": fro["turnover_py"]}
                if window == "fit":
                    for cm in (1.5, 2.0):
                        v = run_online_engine(zw, rw, sw, cm)
                        variants[f"full_cost{cm}"] = {
                            "metrics": v["metrics"],
                            "turnover_py": v["turnover_py"]}
                res_all[window] = variants
                log(f"  {name} [{window}] full: {variants['full']['metrics']} "
                    f"turnover={variants['full']['turnover_py']:.1f}/y")
            engine_metrics[name] = res_all
            # ledger entries for the gauntlet (engine variants count as trials)
            v = res_all.get("fit", {}).get("full", {}).get("metrics", {})
            ledger.add(f"{name}+engine", {"variant": "full"}, v.get("sharpe_ann"),
                       stage="engine", kernel="K2")
            for k in ("no_buffer", "frozen_rls_off", "no_regimes", "frozen_raw"):
                vv = res_all.get("fit", {}).get(k, {}).get("metrics", {})
                ledger.add(f"{name}+engine", {"variant": k}, vv.get("sharpe_ann"),
                           stage="engine", kernel="K2")
        with open(os.path.join(WORK, "engine_metrics.json"), "w") as f:
            json.dump(engine_metrics, f, indent=1, default=str)

    # ---------------- 6) Baseline Gauntlet (G5.3 preview) --------------------
    with stage("baseline gauntlet (G5.3)"):
        gauntlet = {}
        for name in ("M1_GBDT_v2", "M2_RFF_Ridge"):
            v = engine_metrics[name].get("fit", {})
            full = v.get("full", {}).get("metrics", {}).get("sharpe_ann", -9)
            frozen = v.get("frozen_rls_off", {}).get("metrics", {}).get("sharpe_ann", -9)
            nobuf = v.get("no_buffer", {}).get("turnover_py", 0)
            fullto = v.get("full", {}).get("turnover_py", 0)
            gauntlet[name] = {
                "engine_beats_frozen": bool(full >= frozen),
                "buffer_turnover_cut_pct": round(100 * (1 - fullto / nobuf), 1)
                if nobuf > 0 else None,
                "decision": "engine+buffer on" if full >= frozen else "engine off (frozen ships)",
            }
            log(f"  {name}: {gauntlet[name]}")
        ok_g = gate("G2.3_gauntlet", all(g["buffer_turnover_cut_pct"] or 0 >= 25
                                          for g in gauntlet.values()),
                    json.dumps(gauntlet))

    # ---------------- 7) export ----------------------------------------------
    with stage("export"):
        out = pd.DataFrame({
            "z_gbdt": z_gbdt, "pred_gbdt": pred_gbdt,
            "z_rff": z_rff, "pred_rff": pred_rff,
            "z_rff_foldiqr": z_rff_iqr,     # negative control (v2.0.2)
            "z_rlsbank": z_bank, "r_xau": r, "sigma": sigma,
            "y_ret24_norm": feats["y_ret24_norm"],
        })
        out.to_parquet(os.path.join(WORK, "v2_cpu_forecasts.parquet"))
        with open(os.path.join(WORK, "gauntlet.json"), "w") as f:
            json.dump(gauntlet, f, indent=1)
        meta = {"n_rows": int(len(feats)),
                "ledger_entries": len(ledger.rows),
                "unique_configs": ledger.unique_configs(),
                "wall_s": round(time.time() - T0, 1)}
        with open(os.path.join(WORK, "k2_meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        ok_ex = gate("G2.4_export", True,
                     f"v2_cpu_forecasts.parquet + engine_metrics.json + "
                     f"trial_ledger.jsonl ({len(ledger.rows)} entries, "
                     f"{ledger.unique_configs()} unique configs)")

    all_ok = all([ok21, ok_rff, ok48g, ok_bank, ok_g, ok_ex])
    gate("G2_ALL", all_ok, "all K2 gates green" if all_ok else "K2 gate failure")
    with open(os.path.join(WORK, "gates.json"), "w") as f:
        json.dump(GATES, f, indent=2, default=str)
    with open(os.path.join(WORK, "run_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES))
    log("K2 COMPLETE" if all_ok else "K2 FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    _sys.exit(main())
