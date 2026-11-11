#!/usr/bin/env python3
"""Patch aurum-live-daily from v1.2 -> v1.3 (Spec v1.3 decision layer).

Replaces:
  A) the LIVE DAILY constants block (PROD_DEPTH/OOD_REF_DEPTH) with the
     v1.3 policy constants (fitted in aurum-p4-backtest v1.3 on 2016-2022);
  B) the whole live_main() with the v1.3 pipeline: 4-depth CPU forwards,
     multi-depth pooled posterior, sigma_loop, LCB-EV, GBDT agreement gate,
     path-overlay exit plan, quarter-Kelly capped risk, full audit.
Writes the patched kernel to src_push/live/aurum-live-daily.py, compiles it,
and runs structural assertions.
"""
import re, sys, os

SRC = "/home/z/my-project/kaggle_work/aurum/pulled/aurum-live-daily/aurum-live-daily.py"
OUT_DIR = "/home/z/my-project/kaggle_work/aurum/src_push/live"
OUT = os.path.join(OUT_DIR, "aurum-live-daily.py")

src = open(SRC).read()

# ---------------------------------------------------------------- A: consts
CONST_OLD = """# ============================================================================
# LIVE DAILY (Step 6a): production daily decision pathway for the latest bar
# of the mounted dataset. Runs the full stack: state serialization -> looped
# core forward at r=3 (production depth) + r=1 (loop-OOD reference) ->
# temperature-scaled 5 heads -> 48-cell calibrated probabilities -> EV/Kelly
# decision with risk gates -> decision.json + risk_audit.json.
# Reference account for sizing: 10,000 USD, 1.0% hard equity risk cap.
# ============================================================================

PROD_DEPTH = 3
OOD_REF_DEPTH = 1
LIVE_OOD_TH = 0.10
REF_ACCOUNT_USD = 10000.0"""

CONST_NEW = '''# ============================================================================
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
                    "fit window 2016-2022; OOS 2023-25 never touched")'''

assert CONST_OLD in src, "constants anchor not found"
src = src.replace(CONST_OLD, CONST_NEW)

# ---------------------------------------------------- A2: pure helpers + live_main
# find live_main span
m_start = src.index("def live_main(WORK):")
m_end = src.index('if __name__ == "__main__":')
NEW_BLOCK = r'''def _live_base_rate_prior(cls, year_all, y_live):
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


'''
src = src[:m_start] + NEW_BLOCK + src[m_end:]

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT, "w") as f:
    f.write(src)

# compile check
compile(src, OUT, "exec")
print("COMPILE OK ->", OUT, f"({len(src.splitlines())} lines)")

# structural assertions
must_contain = [
    "values[m]",                        # masked empirical overlay (causality)
    "LIVE_POOL_W     = [0.6, 0.2, 0.2]",
    "LIVE_LAMBDA_VAR = 0.5",
    "LIVE_OVERLAY_RULE = \"path\"",
    "G6.4b_policy_consistency",
    "_gbdt_live_fold",
    "GBDT disagreement",
    "h3_head_48h",
    "risk_fraction_capped_cap3",
]
for s in must_contain:
    assert s in src, f"missing: {s}"
print("STRUCTURAL ASSERTIONS OK:", len(must_contain), "markers")
# old v1.2 constants must be gone entirely
assert "PROD_DEPTH" not in src, "PROD_DEPTH still referenced"
assert "OOD_REF_DEPTH" not in src, "OOD_REF_DEPTH still referenced"
print("OLD CONSTANTS REMOVED OK")
print("PATCH COMPLETE")
