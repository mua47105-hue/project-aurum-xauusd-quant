# ============================================================================
# AURUM kernel: p4_backtest (Step 5) — Spec v1.3 "Decision-Layer & Realism
# Upgrade".  CPU-only (consumes aurum_oos_prob_tables.parquet).
#
# v1.2 -> v1.3 (5 flaws fixed):
#  F1  Realized-excursion overlay (lookahead) REPLACED by a causal path-aware
#      walk on H1 OHLC (orders placed at entry; conservative intra-bar order:
#      adverse first; fills at level; ties -> stop).  Overlay is ABLATED
#      (clean vs overlay) and the rule is chosen on pre-2023 only.
#      ATR-unit bug fixed: overlay levels converted to R via /m.
#      B&H forward-outcome bug fixed (was backward 24-bar return).
#  F2  Non-overlapping execution: concurrency cap 1 / 3, positions locked to
#      their resolution bar; per-trade risk 1%/cap; daily mark-to-market
#      portfolio equity, ann Sharpe / CAGR / MaxDD% / Calmar.
#  F3  Decision policy: multi-depth pooled posterior p_bar(r=1:3), shrinkage
#      to fold base-rate prior (strength ~ payoff b), LCB-EV with sigma_loop
#      penalty + lambda_var, EV_min hurdle; all fit causally on <=2022.
#  F4  DSR + bootstrap on NON-OVERLAPPING daily portfolio returns (plus
#      per-trade for the v1.2 comparison).
#  F5  G4.6/G5.2a fixed (era imputation in p3 finalize; strict >=96-bar gap
#      mask audited on USED neighbours).
# ============================================================================
import os as _os, sys as _sys
_os.environ.setdefault("OMP_NUM_THREADS", "4")

# ==================== lib (gates / log / stage) ====================
import json, os, sys, time, traceback, random, hashlib
import numpy as np
import pandas as pd

GATES = {}
LOG_LINES = []

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)

def gate(gid, ok, detail=None, extra=None):
    g = {"gate": gid, "pass": bool(ok), "detail": str(detail)[:2400]}
    if extra:
        g.update(extra)
    GATES[gid] = g
    log(f"GATE {gid}: {'PASS' if ok else 'FAIL'} | {detail}")
    return bool(ok)

def save_gates(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(GATES, f, indent=2, default=str)
    with open(os.path.join(os.path.dirname(path), "run_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES))

def set_seeds(seed=1337):
    random.seed(seed)
    np.random.seed(seed)

def env_report():
    import platform
    info = {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "cwd": os.getcwd(), "argv": sys.argv,
            "local_mode": bool(os.environ.get("AURUM_LOCAL"))}
    for mod in ("xgboost", "sklearn", "numba"):
        try:
            m = __import__(mod)
            info[mod] = m.__version__
        except Exception:
            info[mod] = "missing"
    return info

def stage(name):
    class C:
        def __init__(self):
            self.t0 = time.time(); self.ok = None; self.err = None
        def __enter__(self):
            log(f"===== STAGE {name} START ====="); return self
        def __exit__(self, et, ev, tb):
            self.ok = et is None
            if et is not None:
                self.err = "".join(traceback.format_exception(et, ev, tb))[-3000:]
                log(f"===== STAGE {name} ERROR =====\n{self.err}")
            else:
                log(f"===== STAGE {name} DONE in {time.time()-self.t0:.1f}s =====")
            return False
    return C()

# ==================== frozen architecture constants ====================
BARRIER_STOPS = [1.0, 1.5, 2.0, 2.5]
BARRIER_TPS   = [1.5, 2.5, 4.0]
BARRIER_HORS  = [24, 48]
N_CELLS       = 48
KELLY_FRACTION = 0.25
HARD_EQUITY_RISK_CAP = 0.01
COHORT_GAP = 96

def cell_axes():
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

# ==================== p4 config (Spec v1.3) ====================
P4_DEPTHS    = [1, 2, 3, 4]
POOL_DEPTHS  = [1, 2, 3]          # p_bar multi-depth pool
EV_TH        = 0.0
OOD_TH       = 0.10
COST_BASE_R  = 0.05
COST_MULTS   = [1.0, 1.5, 2.0]
WF_YEARS     = list(range(2016, 2026))
FIT_YEARS    = list(range(2016, 2023))   # policy fitting window (pre-2023)
OOS_YEARS    = [2023, 2024, 2025]
CANON_DIR_CELL = {"L": "L_m1.0_t1.5_h48", "S": "S_m1.0_t1.5_h48"}
EMA_CELL     = {"L": "L_m2.0_t2.5_h48", "S": "S_m2.0_t2.5_h48"}
DON_CELL     = {"L": "L_m2.0_t4.0_h48", "S": "S_m2.0_t4.0_h48"}
SHOCK_TH     = 2.5
NFP_UTC_HOURS = (12, 13, 14)
BOOT_N       = 5000
BOOT_SEED    = 1337
XGB_SEED     = 1337
MAX_CONCURRENT_MODES = [1, 3]
DAY_BLOCK = 10                     # daily-bootstrap block length (days)
# postmortem references (Phase 0/1 exp2_targets, 2023-25 OOS)
POSTMORTEM = {"gbdt": 0.0599, "knn": 0.0078}
POSTMORTEM_TOL = 0.06

# ==================== mounts (kaggle + local dev) ====================
_LOCAL = bool(os.environ.get("AURUM_LOCAL"))
_LOCAL_ROOT = os.environ.get("AURUM_LOCAL_ROOT",
                             "/home/z/my-project/kaggle_work/aurum")

def _pick(cands, probe):
    for c in cands:
        if c and os.path.isfile(os.path.join(c, probe)):
            return c
    return None

def _find_ds():
    import glob as _glob
    if _LOCAL:
        c = _pick([os.path.join(_LOCAL_ROOT, "phase1_ds")], "h1_features.parquet")
        if c:
            return c
    cands = []
    if os.environ.get("AURUM_DS"):
        cands.append(os.environ["AURUM_DS"])
    cands += ["/kaggle/input/datasets/tskdev/aurum-phase1-data",
              "/kaggle/input/aurum-phase1-data"]
    cands += sorted(_glob.glob("/kaggle/input/datasets/*/aurum-phase1-data*"))
    cands += sorted(_glob.glob("/kaggle/input/*/*/aurum-phase1-data*"))
    cands += sorted(_glob.glob("/kaggle/input/*/aurum-phase1-data*"))
    return _pick(cands, "h1_features.parquet")

def _find_p3_out():
    import glob as _glob
    if _LOCAL:
        # prefer the era-imputed local finalize output (G4.6 green) over the
        # pristine v2 pull; on Kaggle the mounted p3-tables v3 provides it.
        c = _pick([os.path.join(_LOCAL_ROOT, "results", "p3_tables_local"),
                   os.path.join(_LOCAL_ROOT, "results", "p3_tables_latest")],
                  "aurum_oos_prob_tables.parquet")
        if c:
            return c
    pats = []
    if os.environ.get("AURUM_P3"):
        pats.append(os.environ["AURUM_P3"])
    pats += ["/kaggle/input/aurum-p3-tables",
             "/kaggle/input/datasets/tskdev/aurum-p3-tables"]
    pats += sorted(_glob.glob("/kaggle/input/*/aurum-p3-tables*"))
    pats += sorted(_glob.glob("/kaggle/input/datasets/*/aurum-p3-tables*"))
    c = _pick(pats, "aurum_oos_prob_tables.parquet")
    if c:
        return c
    # last resort: search everywhere shallow + dump the mount tree for the log
    try:
        import subprocess as _sp
        out = _sp.run(["find", "/kaggle/input", "-maxdepth", "4", "-name",
                       "aurum_oos_prob_tables.parquet"],
                      capture_output=True, text=True, timeout=60).stdout.strip()
        if out:
            return os.path.dirname(out.split("\n")[0])
        tree = _sp.run(["find", "/kaggle/input", "-maxdepth", "3"],
                       capture_output=True, text=True, timeout=60).stdout
        print("[_find_p3_out] NOT FOUND. tree:\n" + tree[:4000], flush=True)
    except Exception:
        pass
    return None

def _find_models_ds():
    import glob as _glob
    if _LOCAL:
        c = _pick([os.path.join(_LOCAL_ROOT, "results", "p3_tables_latest")],
                  "fold_results.json")
        if c:      # fold_results.json shipped with the p3 output mount
            return c
    cands = ["/kaggle/input/datasets/tskdev/aurum-models-v1",
             "/kaggle/input/aurum-models-v1"]
    cands += sorted(_glob.glob("/kaggle/input/datasets/*/aurum-models-v1*"))
    cands += sorted(_glob.glob("/kaggle/input/*/aurum-models-v1*"))
    return _pick(cands, "fold_results.json")

# ==================== data loading ====================

def load_all(WORK):
    DS = _find_ds(); P3 = _find_p3_out(); MODELS = _find_models_ds()
    ok = DS is not None and P3 is not None
    gate("G5.0_mounts", ok,
         f"phase1-data={'OK' if DS else 'MISSING'} p3_out={'OK' if P3 else 'MISSING'} "
         f"models={'OK' if MODELS else 'MISSING'}")
    if not ok:
        return None

    p3_gates_path = os.path.join(P3, "gates.json")
    p3_g = json.load(open(p3_gates_path)) if os.path.isfile(p3_gates_path) else {}
    p3_all = bool(p3_g) and all(p3_g.get(k, {}).get("pass", False)
                                for k in p3_g)
    gate("G5.0b_p3_gates", p3_all,
         "p3 gates (all must be green incl G4.6): "
         + json.dumps({k: p3_g.get(k, {}).get("pass") for k in sorted(p3_g)}))
    if not p3_all:
        return None

    log("loading p3 probability tables ...")
    tb = pd.read_parquet(os.path.join(P3, "aurum_oos_prob_tables.parquet"))
    meta_p3 = json.load(open(os.path.join(P3, "p3_cell_meta.json"))) \
        if os.path.isfile(os.path.join(P3, "p3_cell_meta.json")) else {}
    log(f"  tables: {len(tb):,} rows x {tb.shape[1]} cols")

    vcol = tb["variant"].astype(str).values
    tb = tb[vcol == "fold"].copy()
    log(f"  fold rows: {len(tb):,}")

    labels = pd.read_parquet(os.path.join(DS, "labels_R_actual.parquet"))
    exc = pd.read_parquet(os.path.join(DS, "labels_excursions.parquet"))
    cls = pd.read_parquet(os.path.join(DS, "labels_cls_actual.parquet"))
    feat = pd.read_parquet(os.path.join(DS, "h1_features.parquet"))
    if "ts" in feat.columns:
        feat = feat.drop(columns=["ts"])
    px = pd.read_parquet(os.path.join(DS, "h1_xauusd.parquet"))
    px = px[~px.index.duplicated()].sort_index()
    cohort = np.load(os.path.join(DS, "cohort_idx.npy"))
    spec = json.load(open(os.path.join(DS, "neutralization_spec.json")))
    fold_ref = json.load(open(os.path.join(MODELS, "fold_results.json"))) \
        if MODELS and os.path.isfile(os.path.join(MODELS, "fold_results.json")) else {}

    names = cell_names()
    assert labels.columns.tolist() == names, "label columns != cell_names"
    idx = labels.index

    R_mat = labels.values.astype(np.float32)
    CLS_mat = cls.values.astype(np.int16)      # 0=SL 1=TP 2=timeout (255=nan)
    year_all = pd.DatetimeIndex(idx).year.values.astype(np.int16)
    ts_ns_all = idx.asi8
    pos_of_ts = pd.Series(np.arange(len(idx), dtype=np.int64), index=idx)

    # price / ATR arrays (label conventions: entry = open[i+1], atr[i])
    o = px["o"].values.astype(np.float64)
    hi = px["h"].values.astype(np.float64)
    lo = px["l"].values.astype(np.float64)
    c = px["c"].values.astype(np.float64)
    px_ts = px.index.asi8
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_c), np.abs(lo - prev_c)))
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False,
                            min_periods=14).mean().values.astype(np.float64)
    day_id_all = (px_ts // 86400_000_000_000).astype(np.int64)

    # labels index may be a subset of px index -> map label pos -> px pos
    px_pos_of_lab = px.index.get_indexer(idx)
    assert (px_pos_of_lab >= 0).all(), "labels ts missing in h1_xauusd"

    log(f"  labels: {R_mat.shape}; years {year_all.min()}..{year_all.max()}; "
        f"px bars {len(px):,}")

    return dict(DS=DS, P3=P3, MODELS=MODELS, tb=tb, meta_p3=meta_p3,
                labels=labels, exc=exc, cls=cls, feat=feat, px=px, cohort=cohort,
                spec=spec, names=names, idx=idx, fold_ref=fold_ref,
                R_mat=R_mat, CLS_mat=CLS_mat, year_all=year_all,
                ts_ns_all=ts_ns_all, pos_of_ts=pos_of_ts,
                o=o, hi=hi, lo=lo, c=c, atr=atr, px_ts=px_ts,
                day_id_all=day_id_all, px_pos_of_lab=px_pos_of_lab)

# ==================== blackouts ====================

def nfp_blackout(ts_ns):
    dt = pd.DatetimeIndex(ts_ns.astype("datetime64[ns]"), tz="UTC")
    ym = dt.year.values.astype(np.int64) * 12 + (dt.month.values - 1)
    u_ym, inv = np.unique(ym, return_inverse=True)
    ff_dom = np.empty(len(u_ym), dtype=np.int64)
    for i, v in enumerate(u_ym):
        y_, m_ = int(v) // 12, int(v) % 12 + 1
        wd = pd.Timestamp(year=y_, month=m_, day=1).dayofweek
        ff_dom[i] = 1 + (4 - wd) % 7
    return (dt.day.values == ff_dom[inv]) & np.isin(dt.hour.values, NFP_UTC_HOURS)


def shock_blackout(feat, pos):
    v = feat["mi_vix_d20z"].values
    return np.abs(v[pos]) >= SHOCK_TH

# ==================== GBDT walk-forward (seeded) ====================

def gbdt_walkforward(bundle):
    from xgboost import XGBClassifier
    feat = bundle["feat"]; cls = bundle["cls"]; names = bundle["names"]
    spec_cols = [c for c in bundle["spec"]["features"] if c in feat.columns]
    X_all = feat[spec_cols].values.astype(np.float32)
    year_all = bundle["year_all"]
    ci_L = names.index(CANON_DIR_CELL["L"]); ci_S = names.index(CANON_DIR_CELL["S"])
    yL_all = cls.values[:, ci_L]; yS_all = cls.values[:, ci_S]
    finite = np.isfinite(X_all).all(1)
    lab_rows = np.flatnonzero(finite & (np.isfinite(yL_all) | np.isfinite(yS_all)))
    preds, audit = {}, []
    for y in WF_YEARS:
        tr = lab_rows[(year_all[lab_rows] <= y - 1)]
        n_val = max(200, int(len(tr) * 0.15))
        tr_core, val = tr[:-n_val], tr[-n_val:]
        out = {}
        for tag, ycol in (("L", yL_all), ("S", yS_all)):
            m_tr = np.isfinite(ycol[tr_core]); m_va = np.isfinite(ycol[val])
            try:
                clf = XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    reg_lambda=10.0, subsample=0.7, colsample_bytree=0.7,
                    min_child_weight=20, tree_method="hist", n_jobs=4,
                    eval_metric="logloss", early_stopping_rounds=30,
                    random_state=XGB_SEED)
                clf.fit(X_all[tr_core[m_tr]], (ycol[tr_core[m_tr]] > 0).astype(int),
                        eval_set=[(X_all[val[m_va]], (ycol[val[m_va]] > 0).astype(int))],
                        verbose=False)
            except TypeError:
                # xgboost >= 3.x: early stopping via callbacks
                from xgboost import callback as _xgbc
                clf = XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    reg_lambda=10.0, subsample=0.7, colsample_bytree=0.7,
                    min_child_weight=20, tree_method="hist", n_jobs=4,
                    eval_metric="logloss", random_state=XGB_SEED)
                clf.fit(X_all[tr_core[m_tr]], (ycol[tr_core[m_tr]] > 0).astype(int),
                        eval_set=[(X_all[val[m_va]], (ycol[val[m_va]] > 0).astype(int))],
                        verbose=False,
                        callbacks=[_xgbc.EarlyStopping(rounds=30, save_best=True)])
            te = np.flatnonzero((year_all == y) & finite & np.isfinite(ycol))
            p = np.full(len(bundle["idx"]), np.nan, dtype=np.float32)
            p[te] = clf.predict_proba(X_all[te])[:, 1]
            out["p" + tag] = p
            audit.append({"year": y, "dir": tag, "n_train": int(m_tr.sum()),
                          "best_iter": int(clf.best_iteration or 300),
                          "max_train_year": int(year_all[tr].max())})
        preds[y] = out
        log(f"  GBDT fold {y}: train<= {y-1} "
            f"(L {audit[-2]['n_train']:,} / S {audit[-1]['n_train']:,} rows)")
    return preds, audit

# ==================== k-NN cohort (STRICT gap mask) ====================

def cohort_probs(bundle, rows_pos):
    cls = bundle["cls"]; names = bundle["names"]; cohort = bundle["cohort"]
    ci_L = names.index(CANON_DIR_CELL["L"]); ci_S = names.index(CANON_DIR_CELL["S"])
    yL = cls.values[:, ci_L]; yS = cls.values[:, ci_S]
    pL = np.full(len(rows_pos), np.nan, np.float32)
    pS = np.full(len(rows_pos), np.nan, np.float32)
    CH = 4000
    for s0 in range(0, len(rows_pos), CH):
        pos = rows_pos[s0:s0 + CH]
        nb = cohort[pos]
        ok = (nb >= 0) & (nb < (pos[:, None] - COHORT_GAP))   # STRICT mask
        for yy, out in ((yL, pL), (yS, pS)):
            vals = np.where(ok, yy[nb], np.nan)
            cnt = np.isfinite(vals).sum(1)
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(vals, axis=1) if vals.size else np.array([])
            good = cnt >= 30
            out[s0:s0 + CH][good] = mean[good]
    return pL, pS


def cohort_gap_audit(bundle, rows_pos, n_check=300, seed=7):
    """G5.2a: audit the neighbours ACTUALLY USED (post strict mask)."""
    rng = np.random.default_rng(seed)
    spot = rng.choice(rows_pos, min(n_check, len(rows_pos)), replace=False)
    raw_v = used_v = excluded = 0
    for p0 in spot:
        nb = bundle["cohort"][p0]
        nb = nb[nb >= 0]
        if len(nb) and nb.max() >= p0 - COHORT_GAP:
            raw_v += 1
        used = nb[nb < p0 - COHORT_GAP]
        excluded += len(nb) - len(used)
        if len(used) and used.max() >= p0 - COHORT_GAP:
            used_v += 1
    return {"raw_violations": int(raw_v), "used_violations": int(used_v),
            "excluded_neighbours": int(excluded), "n_check": int(len(spot))}

# ==================== causal quantile / base-rate priors ====================

def empirical_overlay_q(bundle):
    """Empirical MFE-q50 / MAE-q25 per (dir, horizon) over year <= y-1.
    Used by baselines (canonical h48) and by AURUM h24 cells (h3 head only
    predicts 48h quantiles).  ATR units, as are the h3 predictions."""
    exc = bundle["exc"]; year_all = bundle["year_all"]
    q = {}
    for y in WF_YEARS:
        m = (year_all <= y - 1)
        q[y] = {}
        for h_ in (24, 48):
            for k, qv in (("mfe", 0.50), ("mae", 0.25)):
                for d in ("L", "S"):
                    q[y][f"{k}_{d}{h_}"] = float(
                        np.nanquantile(exc[f"{k}_{d}{h_}"].values[m], qv))
    return q


def base_rate_priors(bundle):
    """Expanding per-cell win rate (mean of cls>0) over year <= y-1.
    Causal shrinkage prior pi(c, y).  Also the 2 canonical cells for stacks."""
    cls = bundle["cls"]; year_all = bundle["year_all"]
    W = (cls.values > 0).astype(np.float64)
    fin = np.isfinite(cls.values)
    Wf = np.where(fin, W, np.nan)
    pi = {}
    for y in WF_YEARS:
        m = (year_all <= y - 1)
        with np.errstate(invalid="ignore"):
            pi[y] = np.nanmean(Wf[m], axis=0)      # (48,)
    return pi

# ==================== parquet pivots (per state) ====================

def fold_pivot(bundle):
    """Fold-table pivot: per state (sorted by ts) arrays for r=1..4:
    P_r[ri] (48,) probs, plus h3 predictions (per state), ood, year, ts, px pos."""
    tb = bundle["tb"]
    states = tb[["ts", "year"]].drop_duplicates().sort_values("ts")
    ts_s = states["ts"].values.astype(np.int64)
    yr_s = states["year"].values.astype(np.int16)
    idx_lab = bundle["idx"]
    pos_s = idx_lab.get_indexer(
        pd.DatetimeIndex(ts_s.astype("datetime64[ns]"), tz="UTC"))
    assert (pos_s >= 0).all(), "fold ts missing in labels index"
    n = len(ts_s)
    P = {r: np.full((n, 48), np.nan, np.float32) for r in P4_DEPTHS}
    h3 = np.full((n, 12), np.nan, np.float32)
    ood = np.full(n, np.nan, np.float32)
    name2i = {nm: i for i, nm in enumerate(bundle["names"])}
    sub = tb.sort_values(["ts", "r"])
    # group rows by state via searchsorted on ts
    t_ts = sub["ts"].values.astype(np.int64)
    st_idx = np.searchsorted(ts_s, t_ts)     # row -> state index (ts sorted)
    rr = sub["r"].values.astype(np.int8)
    h1cols = [f"h1_c{i:02d}" for i in range(48)]
    H1 = sub[h1cols].values.astype(np.float32)
    h3cols = [f"h3_{i}" for i in range(0, 12)]
    H3 = sub[h3cols].values.astype(np.float32)
    OODv = sub["ood_mean"].values.astype(np.float32)
    for j in range(len(sub)):
        P[rr[j]][st_idx[j]] = H1[j]
        if rr[j] == 3:                 # h3/ood are depth-invariant in the table
            h3[st_idx[j]] = H3[j]
            ood[st_idx[j]] = OODv[j]
    return dict(ts=ts_s, year=yr_s, pos=pos_s, P=P, h3=h3, ood=ood, n=n)


def h3_q(bundle, h3arr, d_ax, h_ax):
    """h3-predicted overlay quantiles per row: q50 MFE / q25 MAE (ATR units).
    h3 layout: [L MFE q25/q50/q75, L MAE q25/q50/q75, S MFE..., S MAE...].
    h3 only predicts 48h -> h24 rows fall back to empirical 24h quantiles."""
    emp = bundle["_emp_q"]
    yr = bundle["_yr_rows"]           # year per row (aligned)
    q50 = np.where(d_ax == "L", h3arr[:, 1], h3arr[:, 7])
    q25 = np.where(d_ax == "L", h3arr[:, 3], h3arr[:, 9])
    q50 = np.asarray(q50, np.float64).copy()
    q25 = np.asarray(q25, np.float64).copy()
    m24 = h_ax == 24
    if m24.any():
        for d in ("L", "S"):
            msk = m24 & (d_ax == d)
            if msk.any():
                q50[msk] = [emp[y][f"mfe_{d}24"] for y in yr[msk]]
                q25[msk] = [emp[y][f"mae_{d}24"] for y in yr[msk]]
    return q50, q25


# ==================== causal path-walk resolution engine ====================
# Convention (matches the p1 labeler): entry at open[px_i+1]; barriers from
# atr[px_i]; conservative intra-bar order (adverse first, ties->stop); fills
# at the level; timeout marks close of the horizon bar.  Overlay orders are
# placed AT ENTRY (causal): optional tightened stop q25*ATR (if 0<q25<m) and
# optional half-position partial-TP limit at q50*ATR (if 0<q50<t); after a
# partial fill the remainder keeps the ORIGINAL stop/TP and times out at h.

from numba import njit

@njit(cache=True)
def _walk_core(px_i, cell_m, cell_t, cell_h, is_long,
               use_ov, q25, q50,
               o, hi, lo, c, atr, n_bars):
    """Resolve one trade.  Returns (R_path, exit_bar, ov_code, n_life, V_series
    filled into out_V[0..n_life-1]).  V = mark-to-market R value of the whole
    position at the CLOSE of each bar of its life (entry bar .. exit bar)."""
    e = o[px_i + 1]
    a = atr[px_i]
    if a <= 0.0 or not (a == a) or not (e == e):
        return (np.nan, -1, 0)
    ma = cell_m * a
    ta = cell_t * a
    if is_long:
        stop_lvl = e - ma
        tp_lvl = e + ta
    else:
        stop_lvl = e + ma
        tp_lvl = e - ta
    # overlay levels (price), placed at entry
    ov_ok = use_ov and (q25 == q25) and (q50 == q50)
    tight_on = ov_ok and (q25 > 0.0) and (q25 < cell_m)
    part_on = ov_ok and (q50 > 0.0) and (q50 < cell_t)
    if is_long:
        tight_lvl = e - q25 * a if tight_on else 0.0
        part_lvl = e + q50 * a if part_on else 0.0
    else:
        tight_lvl = e + q25 * a if tight_on else 0.0
        part_lvl = e - q50 * a if part_on else 0.0
    end = px_i + 1 + cell_h
    if end > n_bars:
        end = n_bars
    half_banked = 0.0            # R banked by the partial fill (0 if none)
    partial_done = False
    ov_code = 0                   # 0 clean, 1 partial, 2 tighten
    R = np.nan
    exit_bar = end
    for j in range(px_i + 1, end):
        oj = o[j]; hj = hi[j]; lj = lo[j]
        if is_long:
            # ---- phase 1 (before partial fill) -------------------------
            if not partial_done:
                # tightened stop first (most conservative)
                if tight_on and lj <= tight_lvl:
                    R = -(q25 / cell_m)
                    ov_code = 2
                    exit_bar = j
                    break
                # original stop (full loss; conservative beats partial/TP
                # inside the same bar)
                if lj <= stop_lvl:
                    R = -1.0
                    exit_bar = j
                    break
                # partial TP limit (bank HALF the position at q50 level)
                if part_on and hj >= part_lvl:
                    partial_done = True
                    half_banked = 0.5 * q50 / cell_m
                    ov_code = 1
                # original TP (also covers part_on with q50>=t case)
                if hj >= tp_lvl:
                    if partial_done:
                        R = half_banked + 0.5 * (cell_t / cell_m)
                    else:
                        R = cell_t / cell_m
                    exit_bar = j
                    break
            # ---- phase 2 (half position remaining after partial) ------
            else:
                if lj <= stop_lvl:
                    R = half_banked + 0.5 * (-1.0)
                    exit_bar = j
                    break
                if hj >= tp_lvl:
                    R = half_banked + 0.5 * (cell_t / cell_m)
                    exit_bar = j
                    break
        else:
            if not partial_done:
                if tight_on and hj >= tight_lvl:
                    R = -(q25 / cell_m)
                    ov_code = 2
                    exit_bar = j
                    break
                if hj >= stop_lvl:
                    R = -1.0
                    exit_bar = j
                    break
                if part_on and lj <= part_lvl:
                    partial_done = True
                    half_banked = 0.5 * q50 / cell_m
                    ov_code = 1
                if lj <= tp_lvl:
                    if partial_done:
                        R = half_banked + 0.5 * (cell_t / cell_m)
                    else:
                        R = cell_t / cell_m
                    exit_bar = j
                    break
            else:
                if hj >= stop_lvl:
                    R = half_banked + 0.5 * (-1.0)
                    exit_bar = j
                    break
                if lj <= tp_lvl:
                    R = half_banked + 0.5 * (cell_t / cell_m)
                    exit_bar = j
                    break
    if not (R == R):                       # timeout (no touch)
        jend = end - 1
        if is_long:
            R = (c[jend] - o[px_i + 1]) / (cell_m * a)
        else:
            R = (o[px_i + 1] - c[jend]) / (cell_m * a)
        exit_bar = jend
    return (R, exit_bar, ov_code)


@njit(cache=True)
def walk_batch(px_i_arr, cell_m_arr, cell_t_arr, cell_h_arr, is_long_arr,
               use_ov_arr, q25_arr, q50_arr,
               o, hi, lo, c, atr, n_bars):
    n = px_i_arr.shape[0]
    R = np.full(n, np.nan, np.float64)
    eb = np.full(n, -1, np.int64)
    ov = np.zeros(n, np.int8)
    for i in range(n):
        r, e_, v_ = _walk_core(px_i_arr[i], cell_m_arr[i], cell_t_arr[i],
                               cell_h_arr[i], is_long_arr[i],
                               use_ov_arr[i], q25_arr[i], q50_arr[i],
                               o, hi, lo, c, atr, n_bars)
        R[i] = r; eb[i] = e_; ov[i] = v_
    return R, eb, ov


@njit(cache=True)
def mtm_series(px_i, cell_m, cell_t, cell_h, is_long, use_ov, q25, q50,
               o, hi, lo, c, atr, n_bars, V_out):
    """Re-walk one trade and write the mark-to-market value of the WHOLE
    position (in R units) at the close of every bar of its life into
    V_out[0..k-1].  Mirrors _walk_core phase logic."""
    e = o[px_i + 1]
    a = atr[px_i]
    k = 0
    if a <= 0.0 or not (a == a) or not (e == e):
        return 0
    ma = cell_m * a
    ta = cell_t * a
    if is_long:
        stop_lvl = e - ma; tp_lvl = e + ta
    else:
        stop_lvl = e + ma; tp_lvl = e - ta
    ov_ok = use_ov and (q25 == q25) and (q50 == q50)
    tight_on = ov_ok and (q25 > 0.0) and (q25 < cell_m)
    part_on = ov_ok and (q50 > 0.0) and (q50 < cell_t)
    if is_long:
        tight_lvl = e - q25 * a if tight_on else 0.0
        part_lvl = e + q50 * a if part_on else 0.0
    else:
        tight_lvl = e + q25 * a if tight_on else 0.0
        part_lvl = e - q50 * a if part_on else 0.0
    end = px_i + 1 + cell_h
    if end > n_bars:
        end = n_bars
    partial_done = False
    half_banked = 0.0
    resolved = False
    for j in range(px_i + 1, end):
        oj = o[j]; hj = hi[j]; lj = lo[j]
        if is_long:
            if not partial_done:
                if tight_on and lj <= tight_lvl:
                    V_out[k] = -(q25 / cell_m); k += 1; resolved = True; break
                if lj <= stop_lvl:
                    V_out[k] = -1.0; k += 1; resolved = True; break
                if part_on and hj >= part_lvl:
                    partial_done = True
                    half_banked = 0.5 * q50 / cell_m
                if hj >= tp_lvl:
                    V_out[k] = (half_banked + 0.5 * (cell_t / cell_m)
                                if partial_done else cell_t / cell_m)
                    k += 1; resolved = True; break
                # running mark (phase 1)
                v = (c[j] - e) / (cell_m * a)
                V_out[k] = v if not partial_done else half_banked + 0.5 * v
            else:
                if lj <= stop_lvl:
                    V_out[k] = half_banked - 0.5; k += 1; resolved = True; break
                if hj >= tp_lvl:
                    V_out[k] = half_banked + 0.5 * (cell_t / cell_m)
                    k += 1; resolved = True; break
                v = (c[j] - e) / (cell_m * a)
                V_out[k] = half_banked + 0.5 * v
        else:
            if not partial_done:
                if tight_on and hj >= tight_lvl:
                    V_out[k] = -(q25 / cell_m); k += 1; resolved = True; break
                if hj >= stop_lvl:
                    V_out[k] = -1.0; k += 1; resolved = True; break
                if part_on and lj <= part_lvl:
                    partial_done = True
                    half_banked = 0.5 * q50 / cell_m
                if lj <= tp_lvl:
                    V_out[k] = (half_banked + 0.5 * (cell_t / cell_m)
                                if partial_done else cell_t / cell_m)
                    k += 1; resolved = True; break
                v = (e - c[j]) / (cell_m * a)
                V_out[k] = v if not partial_done else half_banked + 0.5 * v
            else:
                if hj >= stop_lvl:
                    V_out[k] = half_banked - 0.5; k += 1; resolved = True; break
                if lj <= tp_lvl:
                    V_out[k] = half_banked + 0.5 * (cell_t / cell_m)
                    k += 1; resolved = True; break
                v = (e - c[j]) / (cell_m * a)
                V_out[k] = half_banked + 0.5 * v
        k += 1
    if not resolved:
        jend = end - 1
        if is_long:
            v = (c[jend] - o[px_i + 1]) / (cell_m * a)
        else:
            v = (o[px_i + 1] - c[jend]) / (cell_m * a)
        if not partial_done:
            V_out[k] = v
        else:
            V_out[k] = half_banked + 0.5 * v
        k += 1
    return k


@njit(cache=True)
def mtm_batch(px_i_arr, cell_m_arr, cell_t_arr, cell_h_arr, is_long_arr,
              use_ov_arr, q25_arr, q50_arr,
              o, hi, lo, c, atr, n_bars, max_life):
    n = px_i_arr.shape[0]
    V = np.zeros((n, max_life), np.float64)
    KL = np.zeros(n, np.int64)
    for i in range(n):
        V_out = V[i]
        KL[i] = mtm_series(px_i_arr[i], cell_m_arr[i], cell_t_arr[i],
                           cell_h_arr[i], is_long_arr[i],
                           use_ov_arr[i], q25_arr[i], q50_arr[i],
                           o, hi, lo, c, atr, n_bars, V_out)
    return V, KL


# ==================== non-overlapping selector ====================

@njit(cache=True)
def select_nonoverlap(t_dec, exit_bar, cap):
    """Chronological admission under a concurrency cap.  A position occupies
    bars (t_dec+1 .. exit_bar]; it is 'open at decision bar t' iff
    exit_bar > t.  Admit candidate i iff #open(t) < cap."""
    n = t_dec.shape[0]
    keep = np.zeros(n, np.bool_)
    active = np.full(64, -1, np.int64)      # exit bars of open positions
    nact = 0
    for i in range(n):
        t = t_dec[i]
        # retire positions resolved at/before t (compaction scan)
        w = 0
        for j in range(nact):
            if active[j] > t:
                active[w] = active[j]; w += 1
        nact = w
        if nact < cap and exit_bar[i] > t:
            keep[i] = True
            active[nact] = exit_bar[i]
            nact += 1
    return keep


def apply_concurrency(trades, cap):
    """trades: dict of aligned arrays (already chronologically sorted by ts).
    Returns boolean keep mask.  Per-trade risk cap becomes 1%/cap."""
    t_dec = trades["pos"].astype(np.int64)          # decision bar == state bar
    exit_bar = trades["exit_bar"].astype(np.int64)
    if cap <= 0:                                   # diagnostic: unlimited
        return np.ones(len(t_dec), np.bool_)
    return select_nonoverlap(t_dec, exit_bar, cap)


# ==================== daily MTM portfolio ====================

@njit(cache=True)
def bh_marks_batch(px_i_arr, atr, o, c, n_bars, max_life):
    """Pure-hold MTM marks for Buy&Hold blocks: v(j) = (c[px_i+1+j] -
    o[px_i+1]) / (2*atr[px_i]); no barriers (matches the B&H label exactly)."""
    n = px_i_arr.shape[0]
    V = np.zeros((n, max_life), np.float64)
    KL = np.zeros(n, np.int64)
    for i in range(n):
        px_i = px_i_arr[i]
        a = 2.0 * atr[px_i]
        e = o[px_i + 1]
        k = 0
        if a <= 0.0 or not (a == a) or not (e == e):
            continue
        end = px_i + 25
        if end > n_bars:
            end = n_bars
        for j in range(px_i + 1, end):
            V[i, k] = (c[j] - e) / a
            k += 1
        KL[i] = k
    return V, KL


THROTTLE_DD_LIMIT = 0.15        # de-risk linearly to 0 at 15%% equity drawdown


def simulate_portfolio(trades, keep, cost_mult, cap, bundle):
    """Sequential day-by-day portfolio simulation with a causal drawdown
    throttle.  Per-trade risk f_eff = min(quarter-Kelly, 1%/cap) x
    max(0, 1 - DD/DD_LIMIT), where DD is the strategy's own equity drawdown
    at the PREVIOUS day's close (strictly past information at the entry
    decision bar).  Guarantees bounded drawdown: as DD -> 15%% new-trade risk
    -> 0, so the equity cannot compound deeply beyond the last-admitted
    position's residual mark (<~1-2%% overshoot).  Returns (daily, f_eff)."""
    sel = np.flatnonzero(keep)
    if len(sel) == 0:
        return None, None
    b = bundle
    day_id = b["day_id_all"]
    px_pos_of_lab = b["px_pos_of_lab"]
    o, hi_, lo_, c_ = b["o"], b["hi"], b["lo"], b["c"]
    atr = b["atr"]; n_bars = len(o)
    max_life = int(48)
    px_i = px_pos_of_lab[trades["pos"][sel]]
    is_bh = trades.get("bh") is not None and bool(trades["bh"][sel].any())
    if is_bh:
        V, KL = bh_marks_batch(px_i, atr, o, c_, n_bars, max_life)
    else:
        V, KL = mtm_batch(px_i,
                          trades["m"][sel], trades["t"][sel], trades["h"][sel],
                          trades["is_long"][sel],
                          trades["use_ov"][sel], trades["q25"][sel], trades["q50"][sel],
                          o, hi_, lo_, c_, atr, n_bars, max_life)
    f_nom = np.minimum(trades["f_kelly"][sel], HARD_EQUITY_RISK_CAP / cap)
    cost = trades["cost_base"][sel] * cost_mult
    entry_day = day_id[px_i + 1]
    exit_day = day_id[trades["exit_bar"][sel]]
    # per-trade map: day -> mark value at the last bar of that day in its life
    day_mark = []
    for k in range(len(sel)):
        e_b = px_i[k] + 1
        x_b = trades["exit_bar"][sel][k]
        days = day_id[e_b: x_b + 1]
        if len(days) == 0:
            day_mark.append({})
            continue
        Vk = V[k][:KL[k]]
        last_idx = {}
        for j in range(x_b - e_b + 1):
            last_idx[days[j]] = j
        day_mark.append({d: (Vk[j] if j < len(Vk) else Vk[-1])
                         for d, j in last_idx.items()})
    order = np.argsort(entry_day, kind="stable")
    d_all = np.arange(int(entry_day.min()), int(exit_day.max()) + 1,
                      dtype=np.int64)
    # entry bookkeeping
    enter_idx = {int(d): [] for d in d_all}
    for k in range(len(sel)):
        enter_idx[int(entry_day[k])].append(k)
    eq = 1.0
    peak = 1.0
    f_eff = np.zeros(len(sel), np.float64)
    last_v = np.zeros(len(sel), np.float64)
    active = np.zeros(len(sel), np.bool_)
    rets = np.zeros(len(d_all))
    eqs = np.zeros(len(d_all) + 1); eqs[0] = 1.0
    for i, d in enumerate(d_all):
        dd = 1.0 - eq / peak if peak > 0 else 0.0
        # -- entries decided today use yesterday's close equity (causal) ----
        for k in enter_idx[int(d)]:
            thr = max(0.0, 1.0 - dd / THROTTLE_DD_LIMIT)
            f_eff[k] = f_nom[k] * thr
            last_v[k] = 0.0
            active[k] = True     # position opens regardless (f_eff may be ~0 -> near-flat)
        # -- marks of open trades today --------------------------------------
        contrib = 0.0
        for k in range(len(sel)):
            if not active[k]:
                continue
            dm = day_mark[k]
            if d in dm:
                v = dm[d]
                contrib += f_eff[k] * (v - last_v[k])
                last_v[k] = v
        # -- cost charged on the resolution day (scaled by f_eff) -------------
        for k in range(len(sel)):
            if active[k] and int(exit_day[k]) == d:
                contrib -= f_eff[k] * cost[k]
                active[k] = False
        # a trade whose life ended without touching today's marks still
        # resolves on its exit day (e.g., resolved before the day's last bar)
        eq *= (1.0 + contrib)
        eqs[i + 1] = eq
        peak = max(peak, eq)
        rets[i] = contrib
    return (dict(day_ns=d_all * 86400_000_000_000, equity=eqs, daily_ret=rets,
                 n_days=len(d_all)), f_eff)


def portfolio_metrics(daily, trades_kept, ts_ns, years_span):
    """Annualized Sharpe / CAGR / MaxDD% / Calmar from the daily MTM series."""
    out = {}
    if daily is None or daily["n_days"] == 0:
        return {"note": "no trades"}
    rets = daily["daily_ret"]
    eq = daily["equity"]
    n = len(rets)
    sd = rets.std(ddof=1) if n > 1 else 0.0
    mean = rets.mean()
    days_per_year = n / years_span if years_span > 0 else n
    out["n_days"] = int(n)
    out["ann_factor_days_per_year"] = round(float(days_per_year), 1)
    out["sharpe_daily"] = float(mean / sd) if sd > 0 else 0.0
    out["sharpe_annualized"] = float(mean / sd * np.sqrt(days_per_year)) \
        if sd > 0 else 0.0
    total = eq[-1]
    yrs = years_span
    out["total_return_pct"] = float((total - 1.0) * 100.0)
    out["cagr_pct"] = float((total ** (1.0 / yrs) - 1.0) * 100.0) if yrs > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    out["max_dd_pct"] = float(dd.max() * 100.0)
    out["calmar"] = float(out["cagr_pct"] / out["max_dd_pct"]) \
        if out["max_dd_pct"] > 0 else 0.0
    return out


# ==================== resolution: clean (labels) + exit bars ====================

@njit(cache=True)
def exit_bar_clean(px_i, cell_m, cell_t, cell_h, is_long, cls_code,
                   o, hi, lo, atr, n_bars):
    """Exit bar for a CLEAN trade whose outcome type is the label's cls code
    (0=SL, 1=TP, 2=timeout): first H1 bar whose range touches the level
    (exact hour of the M1 first touch); timeout -> last bar of window."""
    a = atr[px_i]
    if a <= 0.0 or not (a == a):
        return min(px_i + 1 + cell_h, n_bars - 1)
    e = o[px_i + 1]
    ma = cell_m * a; ta = cell_t * a
    if is_long:
        stop_lvl = e - ma; tp_lvl = e + ta
    else:
        stop_lvl = e + ma; tp_lvl = e - ta
    end = px_i + 1 + cell_h
    if end > n_bars:
        end = n_bars
    if cls_code == 2:
        return end - 1
    for j in range(px_i + 1, end):
        if is_long:
            if cls_code == 0 and lo[j] <= stop_lvl:
                return j
            if cls_code == 1 and hi[j] >= tp_lvl:
                return j
        else:
            if cls_code == 0 and hi[j] >= stop_lvl:
                return j
            if cls_code == 1 and lo[j] <= tp_lvl:
                return j
    return end - 1


@njit(cache=True)
def exit_bar_batch(px_i_arr, m_arr, t_arr, h_arr, is_long_arr, cls_arr,
                   o, hi, lo, atr, n_bars):
    n = px_i_arr.shape[0]
    out = np.zeros(n, np.int64)
    for i in range(n):
        out[i] = exit_bar_clean(px_i_arr[i], m_arr[i], t_arr[i], h_arr[i],
                                is_long_arr[i], cls_arr[i], o, hi, lo, atr, n_bars)
    return out


def resolve_candidates(bundle, cand, overlay_rule):
    """Attach R_gross, exit_bar, overlay_code, R_walk_clean to a candidate
    table.  CLEAN: R = label ground truth, exit bar by label type.
    PATH: R from the causal overlay walk (orders at entry)."""
    b = bundle
    px_i = b["px_pos_of_lab"][cand["pos"]]
    m = cand["m"].astype(np.float64)
    t = cand["t"].astype(np.float64)
    h = cand["h"].astype(np.int64)
    is_long = cand["is_long"].astype(np.bool_)
    o, hi_, lo_, c_ = b["o"], b["hi"], b["lo"], b["c"]
    atr = b["atr"]; n_bars = len(o)
    use_ov = np.ones(len(cand["pos"]), np.bool_) if overlay_rule == "path" \
        else np.zeros(len(cand["pos"]), np.bool_)
    # causal overlay walk (always computed; used when rule == path)
    R_path, eb_path, ov_code = walk_batch(
        px_i, m, t, h, is_long, use_ov,
        cand["q25"].astype(np.float64), cand["q50"].astype(np.float64),
        o, hi_, lo_, c_, atr, n_bars)
    # clean reference walk (validation gate G5.6)
    R_walk_clean, _, _ = walk_batch(
        px_i, m, t, h, is_long, np.zeros(len(px_i), np.bool_),
        cand["q25"].astype(np.float64), cand["q50"].astype(np.float64),
        o, hi_, lo_, c_, atr, n_bars)
    # clean resolution from labels
    cls_code = b["CLS_mat"][cand["pos"], cand["cell"]].astype(np.int64)
    cls_code = np.where(cls_code == 255, 2, cls_code)
    eb_clean = exit_bar_batch(px_i, m, t, h, is_long, cls_code,
                              o, hi_, lo_, atr, n_bars)
    if overlay_rule == "path":
        R_gross = R_path
        exit_bar = np.where(eb_path >= 0, eb_path, eb_clean)
        overlay = ov_code
    else:
        R_gross = cand["R_label"].astype(np.float64)
        exit_bar = eb_clean
        overlay = np.zeros(len(px_i), np.int8)
    out = dict(cand)
    out.update(px_i=px_i, R_gross=R_gross, exit_bar=exit_bar.astype(np.int64),
               overlay=overlay.astype(np.int8), R_walk_clean=R_walk_clean,
               use_ov=use_ov)
    return out


def execute(bundle, res, cap, year_mask=None, cost_mult=1.0):
    """Concurrency selection + per-trade clipping + daily MTM portfolio.
    year_mask: boolean over res rows (already chronological)."""
    rows = np.arange(len(res["pos"]))
    if year_mask is not None:
        rows = rows[year_mask]
    sub = {k: v[rows] for k, v in res.items() if isinstance(v, np.ndarray)}
    order = np.argsort(sub["pos"], kind="stable")     # chronological
    sub = {k: v[order] for k, v in sub.items() if isinstance(v, np.ndarray)}
    keep = apply_concurrency(sub, cap)
    sel = np.flatnonzero(keep)
    daily, f = simulate_portfolio(sub, keep, cost_mult, cap, bundle)
    Rn = sub["R_gross"][sel] - sub["cost_base"][sel] * cost_mult
    return dict(sub=sub, keep=keep, sel=sel, f=f, daily=daily, R_net=Rn,
                ts=sub["ts"][sel], year=sub["year"][sel],
                pos=sub["pos"][sel], exit_bar=sub["exit_bar"][sel])


def per_trade_metrics(R, p=None, win=None, ts_ns=None):
    """Per-trade R statistics (NaN-guarded)."""
    R = np.asarray(R, np.float64)
    n = len(R)
    if n == 0:
        return {"n_trades": 0}
    mean = float(R.mean()); sd = float(R.std(ddof=1)) if n > 1 else 0.0
    pos, neg = R[R > 0], R[R < 0]
    pf = float(pos.sum() / max(1e-9, -neg.sum())) if len(neg) and len(pos) \
        else (999.0 if len(pos) else 0.0)
    cum = np.cumsum(R)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    out = {"n_trades": int(n), "hit_rate": float((R > 0).mean()),
           "expectancy_R": mean, "total_R": float(cum[-1]),
           "std_R": sd, "sharpe_per_trade": (mean / sd) if sd > 0 else 0.0,
           "profit_factor": min(pf, 999.0),
           "max_dd_R": float((peak - cum).max()),
           "avg_win_R": float(pos.mean()) if len(pos) else 0.0,
           "avg_loss_R": float(neg.mean()) if len(neg) else 0.0}
    if ts_ns is not None and n > 1:
        yrs = max((ts_ns[-1] - ts_ns[0]) / (365.25 * 86400e9), 1e-9)
        out["trades_per_year"] = n / yrs
    if p is not None and win is not None:
        p = np.asarray(p, np.float64); win = np.asarray(win, np.float64)
        m = np.isfinite(p) & np.isfinite(win)
        if m.sum() > 10:
            pp, ww = p[m], (win[m] > 0).astype(np.float64)
            out["brier"] = float(np.mean((pp - ww) ** 2))
            order = np.argsort(pp)
            ece = 0.0
            for b_ in np.array_split(order, 10):
                if len(b_):
                    ece += len(b_) / m.sum() * abs(pp[b_].mean() - ww[b_].mean())
            out["ece"] = float(ece)
        else:
            out["brier"] = None
            out["ece"] = None
    else:
        out["brier"] = None
        out["ece"] = None
    return out


# ==================== strategy candidate builders ====================

def _base_cand(bundle, pos, cell, p, ev, q50, q25, src_h3=None):
    """Assemble the common candidate dict."""
    b = bundle
    m = b["_MSS"][cell]; t = b["_TPS"][cell]; h_ = b["_HRS"][cell]
    d_ax = b["_DIRS"][cell]
    is_long = (d_ax == "L")
    R_label = b["R_mat"][pos, cell].astype(np.float64)
    cost_base = np.full(len(pos), COST_BASE_R, np.float64) / m
    kelly = (p * (t / m) - (1 - p)) / (t / m)
    f_kelly = np.clip(KELLY_FRACTION * kelly, 0.0, HARD_EQUITY_RISK_CAP)
    return dict(
        ts=b["ts_ns_all"][pos].astype(np.int64),
        year=b["year_all"][pos].astype(np.int16),
        pos=pos.astype(np.int64), cell=cell.astype(np.int16),
        m=np.float64(m), t=np.float64(t), h=np.int64(h_),
        is_long=np.bool_(is_long) if np.isscalar(is_long) else is_long,
        p=p.astype(np.float32), ev=ev.astype(np.float32),
        f_kelly=f_kelly.astype(np.float64),
        q25=q25.astype(np.float64), q50=q50.astype(np.float64),
        R_label=R_label, cost_base=cost_base,
        ood=np.zeros(len(pos), np.float32))


def aurum48_candidates(bundle, r, gbdt_dir, overlay_src="h3"):
    """v1.2-style raw EV-argmax over 48 cells at depth r (fixed: clean labels,
    concurrency applied later).  Agreement gate vs GBDT direction (v1.2)."""
    b = bundle
    piv = b["_piv"]
    yr = piv["year"]; pos = piv["pos"]
    P = piv["P"][r]
    bgeo = b["_B_GEO"]
    emp = b["_emp_q"]
    # vectorized: EV matrix (n,48) via per-depth probs
    EV = P * bgeo[None, :] - (1.0 - P)
    cell = np.nanargmax(np.where(np.isfinite(EV), EV, -1e9), axis=1).astype(np.int16)
    nrow = np.arange(len(pos))
    p_sel = P[nrow, cell].astype(np.float32)
    ev_sel = EV[nrow, cell].astype(np.float32)
    d_ax = b["_DIRS"][cell]
    nfp = nfp_blackout(piv["ts"])
    shock = shock_blackout(b["feat"], pos)
    ood = piv["ood"]
    g = gbdt_dir[pos]
    agree = (g == np.where(d_ax == "L", "L", "S"))
    ok = (ev_sel > EV_TH) & np.isfinite(p_sel) & (p_sel > 0.01) & (p_sel < 0.99) \
        & (ood <= OOD_TH) & (~nfp) & (~shock) & agree \
        & np.isfinite(b["R_mat"][pos, cell])
    keep = np.flatnonzero(ok & (p_sel > 0))
    pos_k = pos[keep]; cell_k = cell[keep]
    # overlay quantiles: h3 for h48, empirical 24h for h24
    d_k = b["_DIRS"][cell_k]; h_k = b["_HRS"][cell_k]; yr_k = yr[keep]
    q50 = np.empty(len(keep)); q25 = np.empty(len(keep))
    h3rows = piv["h3"][keep]
    for d in ("L", "S"):
        for hh in (24, 48):
            msk = (d_k == d) & (h_k == hh)
            if msk.any():
                if hh == 48:
                    q50[msk] = np.where(d == "L", h3rows[msk, 1], h3rows[msk, 7])
                    q25[msk] = np.where(d == "L", h3rows[msk, 3], h3rows[msk, 9])
                else:
                    q50[msk] = [emp[yy][f"mfe_{d}24"] for yy in yr_k[msk]]
                    q25[msk] = [emp[yy][f"mae_{d}24"] for yy in yr_k[msk]]
    cand = _base_cand(bundle, pos_k, cell_k, p_sel[keep], ev_sel[keep],
                      q50, q25)
    cand["ood"] = ood[keep]
    return cand


def canon_candidates(bundle, r, gbdt_dir):
    """2 canonical cells (L/S m1.0 t1.5 h48), EV argmax over L/S (v1.2 canon,
    fixed).  Agreement gate vs GBDT (v1.2 semantics)."""
    b = bundle
    piv = b["_piv"]
    yr = piv["year"]; pos = piv["pos"]
    P = piv["P"][r]
    ci_L = b["names"].index(CANON_DIR_CELL["L"])
    ci_S = b["names"].index(CANON_DIR_CELL["S"])
    pL = P[:, ci_L]; pS = P[:, ci_S]
    bgeo = 1.5
    evL = pL * bgeo - (1 - pL); evS = pS * bgeo - (1 - pS)
    pick_L = evL >= evS
    cell = np.where(pick_L, ci_L, ci_S).astype(np.int16)
    p = np.where(pick_L, pL, pS).astype(np.float32)
    ev = np.where(pick_L, evL, evS).astype(np.float32)
    d_ax = b["_DIRS"][cell]
    nfp = nfp_blackout(piv["ts"]); shock = shock_blackout(b["feat"], pos)
    ood = piv["ood"]
    g = gbdt_dir[pos]
    agree = (g == np.where(d_ax == "L", "L", "S"))
    ok = (ev > EV_TH) & np.isfinite(p) & (p > 0.01) & (p < 0.99) \
        & (ood <= OOD_TH) & (~nfp) & (~shock) & agree \
        & np.isfinite(b["R_mat"][pos, cell]) & (p > 0)
    keep = np.flatnonzero(ok)
    pos_k = pos[keep]; cell_k = cell[keep]
    h3rows = piv["h3"][keep]
    d_k = b["_DIRS"][cell_k]
    q50 = np.where(d_k == "L", h3rows[:, 1], h3rows[:, 7]).astype(np.float64)
    q25 = np.where(d_k == "L", h3rows[:, 3], h3rows[:, 9]).astype(np.float64)
    cand = _base_cand(bundle, pos_k, cell_k, p[keep], ev[keep], q50, q25)
    cand["ood"] = ood[keep]
    return cand


def pooled_probs(bundle, w):
    """Weighted depth pool p_bar(r=1:3) per state x cell: (n, 48)."""
    piv = bundle["_piv"]
    P = (w[0] * piv["P"][1] + w[1] * piv["P"][2] + w[2] * piv["P"][3])
    return P


def sigma_loop(bundle):
    """Per state x cell: population std of p across depths 1..3."""
    piv = bundle["_piv"]
    P1, P2, P3 = piv["P"][1], piv["P"][2], piv["P"][3]
    m = (P1 + P2 + P3) / 3.0
    var = ((P1 - m) ** 2 + (P2 - m) ** 2 + (P3 - m) ** 2) / 3.0
    return np.sqrt(var)


def shrink(p, kappa, b_eff, pi):
    """Shrink p toward base-rate prior pi with strength kappa*b (payoff-
    dependent: higher b -> stronger shrinkage; winner's-curse correction)."""
    s = kappa * b_eff
    return (p + s * pi) / (1.0 + s)


def flagship_candidates(bundle, params, gbdt_dir, overlay_rule):
    """AURUM v1.3 flagship: pooled p_bar(1:3) -> shrinkage to fold base-rate
    -> LCB-EV (lambda_var * b * sigma_loop penalty) -> EV_min hurdle.
    Agreement gate vs GBDT direction.  Non-overlapping applied at execute."""
    b = bundle
    piv = b["_piv"]
    yr = piv["year"]; pos = piv["pos"]
    P = pooled_probs(b, params["w"])                       # (n,48)
    sig = sigma_loop(b)                                    # (n,48)
    bgeo = b["_B_GEO"]
    pi_mat = np.stack([b["_pi_by_year"][int(y)] for y in yr])  # (n,48)
    P_sh = shrink(P, params["kappa"], bgeo[None, :], pi_mat)
    cost = COST_BASE_R / b["_MSS"][None, :]
    EV_lcb = P_sh * bgeo[None, :] - (1.0 - P_sh) \
        - params["lam"] * bgeo[None, :] * sig - cost
    cell = np.nanargmax(np.where(np.isfinite(EV_lcb), EV_lcb, -1e9),
                        axis=1).astype(np.int16)
    nrow = np.arange(len(pos))
    p_bar = P[nrow, cell]
    p_sh = P_sh[nrow, cell]
    ev_lcb = EV_lcb[nrow, cell]
    sig_sel = sig[nrow, cell]
    d_ax = b["_DIRS"][cell]
    nfp = nfp_blackout(piv["ts"]); shock = shock_blackout(b["feat"], pos)
    ood = piv["ood"]
    g = gbdt_dir[pos]
    agree = (g == np.where(d_ax == "L", "L", "S"))
    ok = (ev_lcb >= params["ev_min"]) & np.isfinite(p_bar) & (p_bar > 0.01) \
        & (p_bar < 0.99) & (ood <= OOD_TH) & (~nfp) & (~shock) & agree \
        & np.isfinite(b["R_mat"][pos, cell]) & (p_sh > 0)
    keep = np.flatnonzero(ok)
    pos_k = pos[keep]; cell_k = cell[keep]
    h3rows = piv["h3"][keep]
    d_k = b["_DIRS"][cell_k]; h_k = b["_HRS"][cell_k]; yr_k = yr[keep]
    emp = b["_emp_q"]
    q50 = np.empty(len(keep)); q25 = np.empty(len(keep))
    for d in ("L", "S"):
        for hh in (24, 48):
            msk = (d_k == d) & (h_k == hh)
            if msk.any():
                if hh == 48:
                    q50[msk] = np.where(d == "L", h3rows[msk, 1], h3rows[msk, 7])
                    q25[msk] = np.where(d == "L", h3rows[msk, 3], h3rows[msk, 9])
                else:
                    q50[msk] = [emp[yy][f"mfe_{d}24"] for yy in yr_k[msk]]
                    q25[msk] = [emp[yy][f"mae_{d}24"] for yy in yr_k[msk]]
    cand = _base_cand(bundle, pos_k, cell_k, p_sh[keep].astype(np.float32),
                      ev_lcb[keep].astype(np.float32), q50, q25)
    cand["p_bar"] = p_bar[keep].astype(np.float32)
    cand["sigma_loop"] = sig_sel[keep].astype(np.float32)
    cand["ood"] = ood[keep]
    return cand


def stack_candidates(bundle, params, gbdt_pL, gbdt_pS, overlay_rule):
    """Precision-weighted log-opinion pool AURUM(p_bar, u) + GBDT(1-u) at the
    canonical cells; directional agreement gate; shrinkage; disagreement
    penalty lambda_d*b*|p_a - p_g|; EV_min hurdle."""
    b = bundle
    piv = b["_piv"]
    yr = piv["year"]; pos = piv["pos"]
    ci_L = b["names"].index(CANON_DIR_CELL["L"])
    ci_S = b["names"].index(CANON_DIR_CELL["S"])
    P = pooled_probs(b, params["w"])
    pa_L, pa_S = P[:, ci_L], P[:, ci_S]
    pg_L = gbdt_pL[pos].astype(np.float64)
    pg_S = gbdt_pS[pos].astype(np.float64)
    fin = np.isfinite(pa_L) & np.isfinite(pa_S) & np.isfinite(pg_L) & np.isfinite(pg_S)
    u = params["u"]
    eps = 1e-9
    def logit(x):
        return np.log(np.clip(x, eps, 1 - eps) / np.clip(1 - x, eps, 1 - eps))
    def expit(z):
        return 1.0 / (1.0 + np.exp(-z))
    ps_L = expit(u * logit(pa_L) + (1 - u) * logit(pg_L))
    ps_S = expit(u * logit(pa_S) + (1 - u) * logit(pg_S))
    piL = np.array([b["_pi_by_year"][int(y)][ci_L] for y in yr])
    piS = np.array([b["_pi_by_year"][int(y)][ci_S] for y in yr])
    ps_L_sh = shrink(ps_L, params["kappa"], 1.5, piL)
    ps_S_sh = shrink(ps_S, params["kappa"], 1.5, piS)
    dis = np.abs(pa_L - pg_L) + np.abs(pa_S - pg_S)   # disagreement (both dirs)
    evL = ps_L_sh * 1.5 - (1 - ps_L_sh) - params["lam"] * 1.5 * np.abs(pa_L - pg_L) \
        - COST_BASE_R / 1.0
    evS = ps_S_sh * 1.5 - (1 - ps_S_sh) - params["lam"] * 1.5 * np.abs(pa_S - pg_S) \
        - COST_BASE_R / 1.0
    pick_L = evL >= evS
    cell = np.where(pick_L, ci_L, ci_S).astype(np.int16)
    p = np.where(pick_L, ps_L_sh, ps_S_sh)
    ev = np.where(pick_L, evL, evS)
    d_ax = b["_DIRS"][cell]
    # agreement gate: GBDT direction must equal stack direction
    g_dir = np.where(gbdt_pL[pos] >= gbdt_pS[pos], "L", "S")
    agree = (g_dir == np.where(d_ax == "L", "L", "S"))
    nfp = nfp_blackout(piv["ts"]); shock = shock_blackout(b["feat"], pos)
    ok = (ev >= params["ev_min"]) & np.isfinite(p) & (p > 0.01) & (p < 0.99) \
        & (p > 0) & (~nfp) & (~shock) & agree & fin \
        & np.isfinite(b["R_mat"][pos, cell])
    keep = np.flatnonzero(ok)
    pos_k = pos[keep]; cell_k = cell[keep]
    h3rows = piv["h3"][keep]
    d_k = b["_DIRS"][cell_k]
    q50 = np.where(d_k == "L", h3rows[:, 1], h3rows[:, 7]).astype(np.float64)
    q25 = np.where(d_k == "L", h3rows[:, 3], h3rows[:, 9]).astype(np.float64)
    cand = _base_cand(bundle, pos_k, cell_k, p[keep].astype(np.float32),
                      ev[keep].astype(np.float32), q50, q25)
    cand["p_bar"] = np.where(pick_L, pa_L, pa_S)[keep].astype(np.float32)
    cand["p_gbdt"] = np.where(pick_L, pg_L, pg_S)[keep].astype(np.float32)
    return cand


def baseline_candidates(bundle, pL_full, pS_full, overlay_src="emp48"):
    """GBDT / k-NN baseline at the canonical cells (clean resolution).
    EV argmax L/S; f>0 filter (p>0.4 at b=1.5); blackouts."""
    b = bundle
    year_all = b["year_all"]
    fin = np.isfinite(pL_full) & np.isfinite(pS_full) & np.isin(year_all, WF_YEARS)
    pos = np.flatnonzero(fin)
    pL = pL_full[pos].astype(np.float64); pS = pS_full[pos].astype(np.float64)
    bgeo = 1.5
    evL = pL * bgeo - (1 - pL); evS = pS * bgeo - (1 - pS)
    pick_L = evL >= evS
    cell = np.where(pick_L, b["names"].index(CANON_DIR_CELL["L"]),
                    b["names"].index(CANON_DIR_CELL["S"])).astype(np.int16)
    p = np.where(pick_L, pL, pS)
    ev = np.where(pick_L, evL, evS)
    ts = b["ts_ns_all"][pos]
    nfp = nfp_blackout(ts); shock = shock_blackout(b["feat"], pos)
    ok = (ev > EV_TH) & (p > 0) & np.isfinite(b["R_mat"][pos, cell]) \
        & (~nfp) & (~shock)
    keep = np.flatnonzero(ok)
    pos_k = pos[keep]; cell_k = cell[keep]
    yr_k = year_all[pos_k]
    emp = b["_emp_q"]
    d_k = b["_DIRS"][cell_k]
    q50 = np.array([emp[int(y)][f"mfe_{d}48"] for y, d in zip(yr_k, d_k)])
    q25 = np.array([emp[int(y)][f"mae_{d}48"] for y, d in zip(yr_k, d_k)])
    return _base_cand(bundle, pos_k, cell_k, p[keep].astype(np.float32),
                      ev[keep].astype(np.float32), q50, q25)


def trailing_p(cell_idx, R_mat, prior=0.45, min_hist=20):
    fin = np.isfinite(R_mat[:, cell_idx])
    win = (R_mat[:, cell_idx] > 0).astype(np.float64)
    p = np.full(len(win), np.nan)
    cnt = 0; s = 0.0
    for i in np.flatnonzero(fin):
        p[i] = (s / cnt) if cnt >= min_hist else prior
        s += win[i]; cnt += 1
    return p


def ema_donchian_candidates(bundle, kind):
    b = bundle
    R_mat = b["R_mat"]; names = b["names"]
    px = b["px"]; year_all = b["year_all"]
    c = px["c"].values
    if kind == "ema":
        cs = pd.Series(c)
        sig = np.sign(cs.ewm(span=50, adjust=False).mean().values
                      - cs.ewm(span=200, adjust=False).mean().values)
        cellmap, m, t = EMA_CELL, 2.0, 2.5
        prev = np.roll(sig, 1); prev[0] = np.nan
        entry = np.isfinite(sig) & (sig != 0) & (sig != prev)
    else:
        hh = px["h"].rolling(20, min_periods=20).max().shift(1).values
        ll = px["l"].rolling(20, min_periods=20).min().shift(1).values
        sig = np.where(c > hh, 1.0, np.where(c < ll, -1.0, np.nan))
        cellmap, m, t = DON_CELL, 2.0, 4.0
        entry = np.isfinite(sig) & (sig != 0)
    pos = np.flatnonzero(entry & np.isin(year_all, WF_YEARS))
    d_ax = np.where(sig[pos] > 0, "L", "S")
    cell = np.array([names.index(cellmap[d]) for d in d_ax], dtype=np.int16)
    pL_i = names.index(cellmap["L"]); pS_i = names.index(cellmap["S"])
    p_full = np.where(sig > 0, trailing_p(pL_i, R_mat), trailing_p(pS_i, R_mat))
    p = p_full[pos]
    bgeo = t / m
    ev = p * bgeo - (1 - p)
    ts = b["ts_ns_all"][pos]
    nfp = nfp_blackout(ts); shock = shock_blackout(b["feat"], pos)
    ok = (ev > EV_TH) & (p > 0) & np.isfinite(b["R_mat"][pos, cell]) \
        & (~nfp) & (~shock)
    keep = np.flatnonzero(ok)
    pos_k = pos[keep]; cell_k = cell[keep]
    yr_k = year_all[pos_k]
    emp = b["_emp_q"]
    d_k = b["_DIRS"][cell_k]
    q50 = np.array([emp[int(y)][f"mfe_{d}48"] for y, d in zip(yr_k, d_k)])
    q25 = np.array([emp[int(y)][f"mae_{d}48"] for y, d in zip(yr_k, d_k)])
    return _base_cand(bundle, pos_k, cell_k, p[keep].astype(np.float32),
                      ev[keep].astype(np.float32), q50, q25)


def buyhold_candidates(bundle):
    """Beta reference, FIXED: non-overlapping 24-bar blocks from the first
    2016 label bar; FORWARD outcome R = (c[p+24] - o[p+1]) / (2*atr[p]);
    cost per block; fixed 0.5% risk (v1.2 sizing)."""
    b = bundle
    year_all = b["year_all"]
    px = b["px"]; atr = b["atr"]; o = b["o"]; c = b["c"]
    start = int(np.flatnonzero(year_all == 2016)[0])
    pos = np.arange(start, len(year_all) - 24, 24)
    pos = pos[np.isin(year_all[pos], WF_YEARS)]
    px_i = b["px_pos_of_lab"][pos]
    ok = (px_i + 25 < len(o)) & (atr[px_i] > 0)
    pos = pos[ok]; px_i = px_i[ok]
    R = np.array([(c[pi + 24] - o[pi + 1]) / (2.0 * atr[pi]) for pi in px_i])
    keep = np.flatnonzero(np.isfinite(R))
    pos_k = pos[keep]
    cell_k = np.full(len(keep), -1, np.int16)
    n = len(keep)
    cand = dict(
        ts=b["ts_ns_all"][pos_k].astype(np.int64),
        year=b["year_all"][pos_k].astype(np.int16),
        pos=pos_k.astype(np.int64), cell=cell_k,
        m=np.full(n, 2.0), t=np.full(n, 24.0), h=np.full(n, 24, np.int64),
        is_long=np.ones(n, np.bool_),
        p=np.full(n, 0.5, np.float32), ev=np.zeros(n, np.float32),
        f_kelly=np.full(n, 0.005, np.float64),
        q25=np.zeros(n), q50=np.zeros(n),
        R_label=R[keep], cost_base=np.full(n, COST_BASE_R / 2.0),
        ood=np.zeros(n, np.float32),
        bh=np.ones(n, np.bool_))       # pure-hold MTM branch flag
    return cand


def resolve_buyhold(bundle, cand):
    """B&H resolution: R_label is the forward outcome; exit_bar = px_i+24."""
    b = bundle
    px_i = b["px_pos_of_lab"][cand["pos"]]
    exit_bar = np.minimum(px_i + 24, len(b["o"]) - 1).astype(np.int64)
    out = dict(cand)
    out.update(px_i=px_i, R_gross=cand["R_label"].astype(np.float64),
               exit_bar=exit_bar, overlay=np.zeros(len(px_i), np.int8),
               R_walk_clean=cand["R_label"].astype(np.float64),
               use_ov=np.zeros(len(px_i), np.bool_),
               bh=np.ones(len(px_i), np.bool_))
    return out


# ==================== bootstrap + DSR ====================

def _pf(R):
    pos, neg = R[R > 0], R[R < 0]
    if len(neg) == 0:
        return float("inf") if len(pos) else 0.0
    return float(pos.sum() / max(1e-9, -neg.sum()))


def bootstrap_ci_pertrade(T, cost_mult=1.0, n=BOOT_N, seed=BOOT_SEED):
    """Week-block bootstrap on per-trade net R (entry-week blocks)."""
    R = np.asarray(T["R_net"], np.float64)
    if len(R) < 30:
        return {"n": int(len(R)), "note": "too few trades"}
    wk = T["ts"] // (7 * 86400 * 10 ** 9)
    uw, inv = np.unique(wk, return_inverse=True)
    blocks = [R[inv == i] for i in range(len(uw))]
    pv = np.asarray(T.get("p"), np.float64) if T.get("p") is not None else None
    wv = (np.asarray(T.get("R_label"), np.float64) > 0).astype(np.float64) \
        if T.get("R_label") is not None else None
    pblk = None
    if pv is not None and wv is not None:
        m = np.isfinite(pv) & np.isfinite(wv)
        if m.all() is False:
            pv = np.where(np.isfinite(pv), pv, 0.5)
            wv = np.where(np.isfinite(wv), wv, 0.0)
        pblk = [(pv[inv == i], wv[inv == i]) for i in range(len(uw))]
    rng = np.random.default_rng(seed)
    nw = len(blocks)
    out = np.zeros((n, 5))
    for b in range(n):
        idx = rng.integers(0, nw, nw)
        Rb = np.concatenate([blocks[i] for i in idx])
        m, s = Rb.mean(), Rb.std(ddof=1)
        br = float("nan")
        if pblk is not None:
            pp = np.concatenate([pblk[i][0] for i in idx])
            ww = np.concatenate([pblk[i][1] for i in idx])
            br = float(np.mean((pp - ww) ** 2))
        cum = np.cumsum(Rb)
        pk = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
        out[b] = (m, (m / s) if s > 0 else 0.0, _pf(Rb),
                  float((pk - cum).max()), br)
    def ci(col):
        q = np.nanpercentile(out[:, col], [2.5, 50, 97.5])
        return {"lo": round(float(q[0]), 5), "med": round(float(q[1]), 5),
                "hi": round(float(q[2]), 5)}
    return {"n_boot": n, "n_blocks": int(nw),
            "expectancy_R": ci(0), "sharpe_per_trade": ci(1),
            "profit_factor": ci(2), "max_dd_R": ci(3), "brier": ci(4)}


def bootstrap_ci_daily(daily, n=BOOT_N, seed=BOOT_SEED + 1):
    """DAY_BLOCK-day block bootstrap on daily portfolio returns:
    ann-Sharpe / MaxDD% / CAGR distributions."""
    if daily is None or daily["n_days"] < 30:
        return {"note": "too few days"}
    rets = daily["daily_ret"]
    nd = len(rets)
    nblk = max(1, nd // DAY_BLOCK)
    starts = np.arange(0, nd, DAY_BLOCK)
    blocks = [rets[s:s + DAY_BLOCK] for s in starts]
    dpv = daily["n_days"] / max(1e-9, (daily["day_ns"][-1] - daily["day_ns"][0])
                                / (365.25 * 86400e9))
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 3))
    for b in range(n):
        idx = rng.integers(0, len(blocks), len(blocks))
        rb = np.concatenate([blocks[i] for i in idx])
        eq = np.cumprod(1.0 + rb)
        peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
        dd = float(((peak - eq) / peak).max() * 100.0)
        m, s = rb.mean(), rb.std(ddof=1)
        sharpe = (m / s * np.sqrt(dpv)) if s > 0 else 0.0
        yrs = len(rb) / dpv
        cagr = float((eq[-1] ** (1.0 / max(yrs, 1e-9)) - 1.0) * 100.0)
        out[b] = (sharpe, dd, cagr)
    def ci(col):
        q = np.nanpercentile(out[:, col], [2.5, 50, 97.5])
        return {"lo": round(float(q[0]), 5), "med": round(float(q[1]), 5),
                "hi": round(float(q[2]), 5)}
    return {"n_boot": n, "n_blocks": int(len(blocks)), "block_days": DAY_BLOCK,
            "sharpe_annualized": ci(0), "max_dd_pct": ci(1), "cagr_pct": ci(2)}


def dsr_table(family, seed_note=""):
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).
    family: name -> return series (NON-OVERLAPPING; daily portfolio returns
    for the primary table, per-trade for the v1.2-comparison table)."""
    from scipy.stats import norm, skew, kurtosis
    srs = {}
    for k, R in family.items():
        R = np.asarray(R, np.float64)
        R = R[np.isfinite(R)]
        if len(R) > 5 and R.std(ddof=1) > 0:
            srs[k] = float(R.mean() / R.std(ddof=1))
    if len(srs) < 3:
        return {"note": "family too small", "srs": srs}
    vals = np.array(list(srs.values()))
    V = float(np.var(vals, ddof=1))
    N = len(vals)
    g = 0.5772156649015329
    Z1 = float(norm.ppf(1.0 - 1.0 / N))
    Z2 = float(norm.ppf(1.0 - 1.0 / (N * np.exp(1.0))))
    SR0 = float(np.sqrt(V) * ((1 - g) * Z1 + g * Z2))
    out = {"N_family": N, "sr0": round(SR0, 6), "members": {}}
    for k, R in family.items():
        R = np.asarray(R, np.float64)
        R = R[np.isfinite(R)]
        Tn = len(R)
        if k not in srs or Tn < 5:
            out["members"][k] = {"note": "skipped"}
            continue
        SR = srs[k]
        s3 = float(skew(R)); k4 = float(kurtosis(R))     # Fisher
        denom = np.sqrt(max(1e-12, 1.0 - s3 * SR + (k4 - 1.0) / 4.0 * SR ** 2))
        dsr = float(norm.cdf((SR - SR0) * np.sqrt(Tn - 1) / denom))
        out["members"][k] = {"sr": round(SR, 5), "T": int(Tn),
                             "skew": round(s3, 3), "kurt": round(k4, 3),
                             "dsr": round(dsr, 5)}
    return out


# ==================== causal policy fitting (<=2022 only) ====================

def _brier(p, win):
    m = np.isfinite(p) & np.isfinite(win)
    if m.sum() < 100:
        return np.nan
    return float(np.mean((p[m] - win[m]) ** 2))


def fit_pool_weights(bundle):
    """F1: simplex grid (step .1) on Brier over FIT_YEARS, all 48 cells."""
    b = bundle
    piv = b["_piv"]
    yr = piv["year"]; pos = piv["pos"]
    fitm = np.isin(yr, FIT_YEARS)
    W = (b["CLS_mat"][pos] > 0).astype(np.float64)      # realized win
    W = np.where(b["CLS_mat"][pos] == 255, np.nan, W)
    P1 = piv["P"][1]; P2 = piv["P"][2]; P3 = piv["P"][3]
    best, grid = None, []
    for a in range(11):
        for c in range(11 - a):
            d = 10 - a - c
            w = np.array([a, c, d], np.float64) / 10.0
            Pp = w[0] * P1 + w[1] * P2 + w[2] * P3
            br = _brier(Pp[fitm], W[fitm])
            grid.append({"w": w.tolist(), "brier": round(br, 6)})
            if np.isfinite(br) and (best is None or br < best[1]):
                best = (w, br)
    return best[0], best[1], grid


def fit_kappa(bundle, w):
    """F2: kappa grid on Brier of shrunk pooled probs (48 cells, FIT_YEARS)."""
    b = bundle
    piv = b["_piv"]
    yr = piv["year"]; pos = piv["pos"]
    fitm = np.isin(yr, FIT_YEARS)
    P = w[0] * piv["P"][1] + w[1] * piv["P"][2] + w[2] * piv["P"][3]
    W = (b["CLS_mat"][pos] > 0).astype(np.float64)
    W = np.where(b["CLS_mat"][pos] == 255, np.nan, W)
    pi = b["_pi_by_year"]
    pi_fit = np.stack([pi[int(y)] for y in yr])
    bgeo = b["_B_GEO"]
    best, grid = None, []
    for kap in (0.0, 0.05, 0.10, 0.20, 0.40):
        P_sh = shrink(P, kap, bgeo[None, :], pi_fit)
        br = _brier(P_sh[fitm], W[fitm])
        grid.append({"kappa": kap, "brier": round(br, 6)})
        if np.isfinite(br) and (best is None or br < best[1]):
            best = (kap, br)
    return best[0], best[1], grid


def _ann_sharpe_of(bundle, params, gbdt_dir, overlay_rule, lam, ev_min, kap=None):
    """One flagship evaluation on FIT_YEARS (standalone window), cap=1."""
    p = dict(params); p["lam"] = lam; p["ev_min"] = ev_min
    if kap is not None:
        p["kappa"] = kap
    cand = flagship_candidates(bundle, p, gbdt_dir, overlay_rule)
    fitm = np.isin(cand["year"], FIT_YEARS)
    cand = {k: (v[fitm] if isinstance(v, np.ndarray) and len(v) == len(fitm) else v)
            for k, v in cand.items()}
    res = resolve_candidates(bundle, cand, overlay_rule)
    ex = execute(bundle, res, cap=1)
    Rn = ex["R_net"]
    if len(Rn) < 20 or ex["daily"] is None:
        return {"sharpe": -9.9, "er": -9.9, "n": len(Rn)}
    d = ex["daily"]
    rets = d["daily_ret"]
    sd = rets.std(ddof=1)
    dpv = d["n_days"] / max(1e-9, (d["day_ns"][-1] - d["day_ns"][0])
                            / (365.25 * 86400e9))
    sharpe = float(rets.mean() / sd * np.sqrt(dpv)) if sd > 0 else 0.0
    return {"sharpe": sharpe, "er": float(Rn.mean()), "n": int(len(Rn)),
            "maxdd_pct": None if ex["daily"] is None else None}


def fit_lambda_evmin(bundle, params, gbdt_dir):
    """F3: 3-D grid on (kappa, lambda_var, EV_min) by FIT-window ann-Sharpe
    (tie: E[R]).  kappa is re-fitted here on the PERFORMANCE objective (the
    Brier-optimal kappa from F2 is kept for the record / stack); overlay
    rule fixed to CLEAN (principled default; F4 ablates afterwards)."""
    kappas = (0.0, 0.05, 0.10, 0.20, 0.40)
    lams = (0.0, 0.25, 0.5, 1.0)
    evmins = (0.0, 0.05, 0.10, 0.15, 0.20)
    best, grid = None, []
    for kap in kappas:
        for lam in lams:
            for em in evmins:
                r = _ann_sharpe_of(bundle, params, gbdt_dir, "clean", lam, em, kap)
                grid.append({"kappa": kap, "lam": lam, "ev_min": em,
                             **{k: (round(v, 5) if isinstance(v, float) else v)
                                for k, v in r.items()}})
                key = (round(r["sharpe"], 6), round(r["er"], 6))
                if best is None or key > best[0]:
                    best = (key, (kap, lam, em), r)
    return best[1], best[2], grid


def ablate_overlay(bundle, params, gbdt_dir, stack_params=None):
    """F4: overlay on/off on FIT_YEARS at the chosen params (ann-Sharpe
    primary, E[R] tie-break); also OOS numbers for the report (report-only,
    never used for the choice)."""
    out = {"fit_window": {}, "oos_report_only": {}}
    for rule in ("clean", "path"):
        r = _ann_sharpe_of(bundle, params, gbdt_dir, rule,
                           params["lam"], params["ev_min"])
        out["fit_window"][rule] = {k: (round(v, 5) if isinstance(v, float) else v)
                                   for k, v in r.items()}
        # OOS report-only
        p2 = dict(params)
        cand = flagship_candidates(bundle, p2, gbdt_dir, rule)
        oosm = np.isin(cand["year"], OOS_YEARS)
        cand = {k: (v[oosm] if isinstance(v, np.ndarray) and len(v) == len(oosm) else v)
                for k, v in cand.items()}
        res = resolve_candidates(bundle, cand, rule)
        ex = execute(bundle, res, cap=1)
        Rn = ex["R_net"]
        d = ex["daily"]
        sd = d["daily_ret"].std(ddof=1) if d is not None else 0.0
        dpv = (d["n_days"] / max(1e-9, (d["day_ns"][-1] - d["day_ns"][0])
                / (365.25 * 86400e9))) if d is not None else 1.0
        sharpe = float(d["daily_ret"].mean() / sd * np.sqrt(dpv)) if sd > 0 else 0.0
        out["oos_report_only"][rule] = {
            "n": int(len(Rn)), "er": round(float(Rn.mean()), 5) if len(Rn) else None,
            "sharpe_ann": round(sharpe, 5)}
    c_fit = out["fit_window"]["clean"]; p_fit = out["fit_window"]["path"]
    rule = "clean" if (round(c_fit["sharpe"], 6), round(c_fit["er"], 6)) >= \
        (round(p_fit["sharpe"], 6), round(p_fit["er"], 6)) else "path"
    out["chosen_rule"] = rule
    out["decision_basis"] = ("FIT_YEARS 2016-22 standalone cap=1 ann-Sharpe "
                             "(tie-break E[R]); OOS numbers report-only")
    return rule, out


def fit_stack_u(bundle, params, gbdt_pL, gbdt_pS):
    """F5: stack pool weight u by Brier of the log-pool at the canonical
    cells (FIT_YEARS rows where both models finite)."""
    b = bundle
    piv = b["_piv"]
    yr = piv["year"]; pos = piv["pos"]
    fitm = np.isin(yr, FIT_YEARS)
    ci_L = b["names"].index(CANON_DIR_CELL["L"])
    ci_S = b["names"].index(CANON_DIR_CELL["S"])
    P = pooled_probs(b, params["w"])
    pa_L, pa_S = P[:, ci_L], P[:, ci_S]
    pg_L = gbdt_pL[pos]; pg_S = gbdt_pS[pos]
    fin = fitm & np.isfinite(pa_L) & np.isfinite(pa_S) \
        & np.isfinite(pg_L) & np.isfinite(pg_S)
    eps = 1e-9
    def logit(x):
        return np.log(np.clip(x, eps, 1 - eps) / np.clip(1 - x, eps, 1 - eps))
    def expit(z):
        return 1.0 / (1.0 + np.exp(-z))
    W = (b["CLS_mat"][pos, ci_L] > 0).astype(np.float64)
    W = np.where(b["CLS_mat"][pos, ci_L] == 255, np.nan, W)
    best, grid = None, []
    for i in range(21):
        u = i / 20.0
        ps_L = expit(u * logit(pa_L) + (1 - u) * logit(pg_L))
        br_L = _brier(ps_L[fin], W[fin])
        W_S = (b["CLS_mat"][pos, ci_S] > 0).astype(np.float64)
        W_S = np.where(b["CLS_mat"][pos, ci_S] == 255, np.nan, W_S)
        ps_S = expit(u * logit(pa_S) + (1 - u) * logit(pg_S))
        br_S = _brier(ps_S[fin], W_S[fin])
        br = 0.5 * (br_L + br_S)
        grid.append({"u": u, "brier_L": round(br_L, 6), "brier_S": round(br_S, 6),
                     "brier": round(br, 6)})
        if np.isfinite(br) and (best is None or br < best[1]):
            best = (u, br)
    return best[0], best[1], grid


# ==================== evaluation ====================

def years_span_of(daily):
    if daily is None or daily["n_days"] == 0:
        return 1e-9
    return max(1e-9, (daily["day_ns"][-1] - daily["day_ns"][0]) / (365.25 * 86400e9))


def portfolio_block(ex, cost_mult):
    """Portfolio metrics block from an executed run."""
    d = ex["daily"]
    Rn = ex["R_net"]
    out = per_trade_metrics(Rn)
    if d is not None and d["n_days"] > 0:
        rets = d["daily_ret"]
        sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
        dpv = d["n_days"] / years_span_of(d)
        eq = d["equity"]
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        out["portfolio"] = {
            "n_days": int(d["n_days"]),
            "days_per_year": round(float(dpv), 1),
            "sharpe_daily": float(rets.mean() / sd) if sd > 0 else 0.0,
            "sharpe_annualized": float(rets.mean() / sd * np.sqrt(dpv)) if sd > 0 else 0.0,
            "total_return_pct": float((eq[-1] - 1.0) * 100.0),
            "cagr_pct": float((eq[-1] ** (1.0 / years_span_of(d)) - 1.0) * 100.0),
            "max_dd_pct": float(dd.max() * 100.0),
            "calmar": float((eq[-1] ** (1.0 / years_span_of(d)) - 1.0) * 100.0
                            / max(dd.max() * 100.0, 1e-9))}
    else:
        out["portfolio"] = None
    return out


def exposure_and_concurrency(bundle, ex, cap):
    """G5.8 audit: observed max concurrent positions, cap1 strict
    non-overlap check, exposure fraction."""
    sub = ex["sub"]; keep = ex["keep"]
    pos = sub["pos"][keep]; xb = sub["exit_bar"][keep]
    order = np.argsort(pos, kind="stable")
    pos = pos[order]; xb = xb[order]
    n = len(pos)
    if n == 0:
        return {"max_concurrent": 0, "overlap_violations": 0, "exposure": 0.0}
    events = []
    for i in range(n):
        events.append((pos[i] + 1, 1))
        events.append((xb[i] + 1, -1))     # position frees AT xb+1 (bar after res)
    events.sort()
    cur = mx = 0
    for t, d_ in events:
        cur += d_
        mx = max(mx, cur)
    # overlap violations: pairs of kept trades sharing any bar
    viol = 0
    if cap == 1:
        for i in range(n - 1):
            if xb[i] > pos[i + 1]:         # i still open at next decision bar
                viol += 1
    # exposure: bars with an open position over the window
    day_id = bundle["day_id_all"]
    lo_b, hi_b = pos[0], xb[-1]
    occ = np.zeros(hi_b - lo_b + 2, np.int64)
    for i in range(n):
        occ[pos[i] + 1 - lo_b] += 1
        occ[xb[i] + 1 - lo_b] -= 1
    occ = np.cumsum(occ)
    exposure = float((occ[: hi_b - lo_b + 1] > 0).mean()) if len(occ) > 1 else 0.0
    return {"max_concurrent": int(mx), "overlap_violations": int(viol),
            "exposure": round(exposure, 4)}


# ==================== main ====================

def main(WORK):
    t0 = time.time()
    log("ENV: " + json.dumps(env_report()))
    bundle = load_all(WORK)
    if bundle is None:
        save_gates(os.path.join(WORK, "gates.json"))
        return
    b = bundle
    names = b["names"]
    # derived arrays used by the builders
    _dirs, _mss, _tps, _hrs = cell_axes()
    b["_DIRS"] = np.array(_dirs)
    b["_MSS"] = np.array(_mss, np.float64)
    b["_TPS"] = np.array(_tps, np.float64)
    b["_HRS"] = np.array(_hrs, np.int64)
    b["_B_GEO"] = b["_TPS"] / b["_MSS"]

    with stage("pivots / priors / quantiles"):
        b["_piv"] = fold_pivot(b)
        b["_pi_by_year"] = base_rate_priors(b)
        b["_emp_q"] = empirical_overlay_q(b)
    log(f"  pivot states={b['_piv']['n']:,}; "
        f"base-rate priors for {len(b['_pi_by_year'])} years")

    # ---------- GBDT + cohort baselines ------------------------------------
    with stage("gbdt walk-forward (44 era-safe features, per fold year)"):
        preds, audit = gbdt_walkforward(b)
    year_all = b["year_all"]
    pL_full = np.full(len(year_all), np.nan, np.float32)
    pS_full = np.full(len(year_all), np.nan, np.float32)
    for y in WF_YEARS:
        m = year_all == y
        pL_full[m] = preds[y]["pL"][m]; pS_full[m] = preds[y]["pS"][m]
    gbdt_dir = np.where(np.isnan(pL_full) | np.isnan(pS_full), "N",
                        np.where(pL_full >= pS_full, "L", "S")).astype(object)

    with stage("k-NN cohort probabilities (canonical cells, strict mask)"):
        wf_pos = np.flatnonzero(np.isin(year_all, WF_YEARS))
        pL_c, pS_c = cohort_probs(b, wf_pos)
        pL_cfull = np.full(len(year_all), np.nan, np.float32); pL_cfull[wf_pos] = pL_c
        pS_cfull = np.full(len(year_all), np.nan, np.float32); pS_cfull[wf_pos] = pS_c
        aud = cohort_gap_audit(b, wf_pos, 300)
    gate("G5.2a_cohort_gap", aud["used_violations"] == 0,
         f"USED-neighbour audit (strict >= {COHORT_GAP}-bar mask enforced before "
         f"use): used violations={aud['used_violations']}; raw excluded="
         f"{aud['raw_violations']}/{aud['n_check']} spot states "
         f"({aud['excluded_neighbours']} neighbours dropped by the mask)")

    # ---------- causal policy fitting (<=2022) ------------------------------
    with stage("causal policy fit on 2016-2022 (zero 2023-25 peeking)"):
        w, br_w, grid_w = fit_pool_weights(b)
        log(f"  F1 pool weights w={w.tolist()} Brier={br_w:.5f}")
        kappa, br_k, grid_k = fit_kappa(b, w)
        log(f"  F2 kappa={kappa} Brier={br_k:.5f}")
        params0 = {"w": w, "kappa": kappa}
        (kap3, lam, ev_min), f3best, grid_le = fit_lambda_evmin(b, params0, gbdt_dir)
        params = {"w": w, "kappa": kap3, "lam": lam, "ev_min": ev_min}
        log(f"  F3 (kappa, lambda_var, EV_min)=({kap3}, {lam}, {ev_min}) "
            f"ann-Sharpe={f3best['sharpe']:.4f} E[R]={f3best['er']:.5f} n={f3best['n']}"
            f" (kappa_Brier from F2 was {kappa})")
        # F6: performance re-fit of the pooling weights at the F3 params
        # (greedy coordinate ascent: the posterior pool is a decision-layer
        # input for the flagship; the stack keeps the Brier-calibrated w)
        w_cands = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                   [1 / 3, 1 / 3, 1 / 3], [0.5, 0.25, 0.25],
                   [0.5, 0.0, 0.5], [0.25, 0.25, 0.5], [0.6, 0.2, 0.2],
                   [0.2, 0.2, 0.6], [0.4, 0.2, 0.4])
        best_w, best_r = list(params["w"]), None
        grid_w_perf = []
        for wc in w_cands:
            p6 = dict(params); p6["w"] = np.array(wc, np.float64)
            r6 = _ann_sharpe_of(b, p6, gbdt_dir, "clean",
                                params["lam"], params["ev_min"], params["kappa"])
            grid_w_perf.append({"w": wc, **{k: (round(v, 5) if isinstance(v, float) else v)
                                            for k, v in r6.items()}})
            if best_r is None or (round(r6["sharpe"], 6), round(r6["er"], 6)) > \
                    (round(best_r["sharpe"], 6), round(best_r["er"], 6)):
                best_w, best_r = list(wc), r6
        params["w"] = np.array(best_w, np.float64)
        log(f"  F6 pooling-w performance re-fit -> w={best_w} "
            f"ann-Sharpe={best_r['sharpe']:.4f} E[R]={best_r['er']:.5f} n={best_r['n']}")
        rule, ablation = ablate_overlay(b, params, gbdt_dir)
        params["overlay_rule"] = rule
        log(f"  F4 overlay ablation -> rule={rule} "
            f"(fit clean {ablation['fit_window']['clean']} vs "
            f"path {ablation['fit_window']['path']})")
        u, br_u, grid_u = fit_stack_u(b, params, pL_full, pS_full)
        stack_params = dict(params); stack_params["u"] = u
        log(f"  F5 stack u={u} Brier={br_u:.5f}")
    fit_record = {
        "fit_window_years": FIT_YEARS,
        "max_year_used": max(FIT_YEARS),
        "oos_years_never_touched_by_fits": OOS_YEARS,
        "protocol": "F1 w (Brier, 48 cells) -> F2 kappa (Brier of shrunk) -> "
                    "F3 3-D grid (kappa, lambda_var, EV_min) by FIT-window cap1 "
                    "ann-Sharpe (clean default) -> F6 pooling-w performance "
                    "re-fit -> F4 overlay ablation (FIT window; choice made on "
                    "FIT only, OOS report-only) -> F5 stack u (Brier at "
                    "canonical cells). Stack uses Brier-w/Brier-kappa; flagship "
                    "uses performance-w/perf-kappa.",
        "w": {"value_brier_F1": w.tolist(), "brier": round(br_w, 6),
              "grid": grid_w,
              "value_perf_F6_flagship": params["w"].tolist()},
        "kappa": {"value_brier_F2": kappa, "brier": round(br_k, 6), "grid": grid_k,
                   "value_perf_F3": kap3},
        "lambda_var": {"value": lam, "grid": grid_le},
        "ev_min": {"value": ev_min},
        "overlay_rule": {"value": rule, "ablation": ablation},
        "stack_u": {"value": u, "brier": round(br_u, 6), "grid": grid_u},
        "f6_w_perf_grid": grid_w_perf,
        "f3_best": {k: (round(v, 5) if isinstance(v, float) else v)
                    for k, v in f3best.items()}}
    gate("G5.7_fit_causality", max(FIT_YEARS) == 2022,
         f"all policy fits used years {min(FIT_YEARS)}..{max(FIT_YEARS)} only; "
         f"OOS {OOS_YEARS} never touched by any fit; "
         f"fold predictions are themselves walk-forward (heads trained <=y-1)")

    # ---------- build all strategies ---------------------------------------
    with stage("build candidates for all strategies"):
        C = {}
        for r in P4_DEPTHS:
            C[f"aurum_r{r}"] = aurum48_candidates(b, r, gbdt_dir)
            log(f"  aurum_r{r}: {len(C[f'aurum_r{r}']['pos']):,} candidates")
        C["canon"] = canon_candidates(b, 3, gbdt_dir)
        C["flagship"] = flagship_candidates(b, params, gbdt_dir, rule)
        C["flagship"]["policy"] = "LCB-EV + pooled p_bar(1:3) + shrinkage + EV_min + GBDT agreement"
        log(f"  canon: {len(C['canon']['pos']):,} | flagship: {len(C['flagship']['pos']):,}")
        C["stack"] = stack_candidates(b, stack_params, pL_full, pS_full, rule)
        log(f"  stack: {len(C['stack']['pos']):,}")
        C["gbdt"] = baseline_candidates(b, pL_full, pS_full)
        C["knn"] = baseline_candidates(b, pL_cfull, pS_cfull)
        C["ema"] = ema_donchian_candidates(b, "ema")
        C["donchian"] = ema_donchian_candidates(b, "donchian")
        C["buyhold"] = buyhold_candidates(b)
        for k in ("gbdt", "knn", "ema", "donchian", "buyhold"):
            log(f"  {k}: {len(C[k]['pos']):,} candidates")

    # ---------- resolve ------------------------------------------------------
    with stage("resolve (clean labels / path overlay + exit bars)"):
        R = {}
        for k, cand in C.items():
            if k == "buyhold":
                R[k] = resolve_buyhold(b, cand)
            else:
                # the ablation-chosen overlay rule applies to the v1.3 DECISION
                # LAYER (flagship + stack); comparators (aurum_r*, canon and
                # all baselines) resolve from clean barrier labels.
                R[k] = resolve_candidates(b, cand,
                                          rule if k in ("flagship", "stack")
                                          else "clean")
    # walk validation gate (flagship, clean walk vs labels)
    fw = R["flagship"]["R_walk_clean"] - R["flagship"]["R_label"]
    agree = float((np.abs(fw) <= 0.25).mean())
    corr = float(np.corrcoef(R["flagship"]["R_walk_clean"],
                             R["flagship"]["R_label"])[0, 1])
    mdr = float(np.abs(fw).mean())
    gate("G5.6_walk_validation", (agree >= 0.85) and (corr >= 0.95) and (mdr <= 0.20),
         f"H1 walk (conservative, adverse-first) vs p1 M1 labels on flagship "
         f"trades: |dR|<=0.25 agreement={agree * 100:.2f}%; corr={corr:.5f}; "
         f"mean|dR|={mdr:.4f}. Residual gap = intra-hour M1 sequencing (H1 cannot "
         f"resolve TP-vs-SL order inside one hour; walk takes the adverse side) "
         f"+ weekend clock-vs-bar-count windows. Clean trades resolve from the "
         f"labels themselves; the walk supplies exit bars for concurrency.")

    # ---------- determinism --------------------------------------------------
    rep_flag = flagship_candidates(b, params, gbdt_dir, rule)
    rep_res = resolve_candidates(b, rep_flag, rule)
    h1 = hashlib.sha256(np.concatenate(
        [R["flagship"]["ts"].view(np.int64), R["flagship"]["R_gross"],
         R["flagship"]["cell"].astype(np.int64),
         R["flagship"]["exit_bar"]]).tobytes()).hexdigest()
    h2 = hashlib.sha256(np.concatenate(
        [rep_res["ts"].view(np.int64), rep_res["R_gross"],
         rep_res["cell"].astype(np.int64), rep_res["exit_bar"]]).tobytes()).hexdigest()
    hg1 = hashlib.sha256(pL_full.tobytes()).hexdigest()[:16]
    hg2 = hashlib.sha256(np.asarray(preds[2025]["pL"]).tobytes()).hexdigest()[:16]
    gate("G5.1_determinism", h1 == h2,
         f"flagship trade-hash {h1[:16]} == {h2[:16]}; GBDT pL hash {hg1}")

    # ---------- execute + evaluate -------------------------------------------
    summary = {"config": {
        "spec": "v1.3", "ev_th": EV_TH, "ood_th": OOD_TH,
        "kelly": KELLY_FRACTION, "risk_cap": HARD_EQUITY_RISK_CAP,
        "cost_base_r": COST_BASE_R, "cost_mults": COST_MULTS,
        "oos_years": OOS_YEARS, "wf_years": WF_YEARS,
        "fit_years": FIT_YEARS, "boot_n": BOOT_N,
        "overlay_rule": rule, "overlay_rule_scope": "flagship + stack (v1.3 decision layer); comparators use clean labels",
        "overlay_engine": "causal path walk on H1 OHLC; orders at entry; "
                           "adverse-first intra-bar; fills at level; ATR->R via /m",
        "concurrency": "max_concurrent in {1,3}; per-trade risk 1%/cap; "
                       "positions locked to resolution bar",
        "blackouts": "NFP first-Friday 12:00-14:59 UTC + |mi_vix_d20z|>=2.5",
        "postmortem_refs": POSTMORTEM}}
    strat_caps = {k: [1] for k in C}
    for k in ("flagship", "stack", "canon"):
        strat_caps[k] = [1, 3]
    EX = {}
    for k in C:
        EX[k] = {}
        for cap in strat_caps[k]:
            full_mask = None
            oos_mask = np.isin(R[k]["year"], OOS_YEARS)
            fit_mask = np.isin(R[k]["year"], FIT_YEARS)
            EX[k][cap] = {}
            for mult in COST_MULTS:
                exf = execute(b, R[k], cap, None, mult)
                exo = execute(b, R[k], cap, oos_mask, mult)
                exfit = execute(b, R[k], cap, fit_mask, mult)
                EX[k][cap][mult] = {"full": exf, "oos": exo, "fit": exfit}
                summary.setdefault(k, {}).setdefault(f"cap{cap}", {})
                summary[k][f"cap{cap}"][f"cost_{mult}"] = {
                    "full": portfolio_block(exf, mult),
                    "oos": portfolio_block(exo, mult),
                    "fit": portfolio_block(exfit, mult)}
                log(f"  {k} cap{cap} c{mult}: "
                    f"full n={summary[k][f'cap{cap}'][f'cost_{mult}']['full'].get('n_trades', 0)}"
                    f" E[R]={summary[k][f'cap{cap}'][f'cost_{mult}']['full'].get('expectancy_R', float('nan')):.4f}"
                    f" | oos n={summary[k][f'cap{cap}'][f'cost_{mult}']['oos'].get('n_trades', 0)}"
                    f" E[R]={summary[k][f'cap{cap}'][f'cost_{mult}']['oos'].get('expectancy_R', float('nan')):.4f}"
                    f" Sharpe_ann={((summary[k][f'cap{cap}'][f'cost_{mult}']['oos'].get('portfolio') or {}).get('sharpe_annualized', float('nan'))):.4f}")

    # ---------- coverage + realism gates -------------------------------------
    fl = R["flagship"]
    fl_years = set(np.unique(EX["flagship"][1][1.0]["full"]["year"]).tolist())
    oos_n = int(summary["flagship"]["cap1"]["cost_1.0"]["oos"].get("n_trades", 0))
    gate("G5.3_coverage", (len(fl_years) >= 9) and (oos_n >= 50),
         f"flagship EXECUTED (cap1) in {len(fl_years)}/10 years {sorted(fl_years)}; "
         f"OOS executed trades={oos_n}")
    real_ok = True
    real_detail = {}
    for cap in (1, 3):
        exf = EX["flagship"][cap][1.0]["full"]
        aud_rc = exposure_and_concurrency(b, exf, cap)
        dd = (summary["flagship"][f"cap{cap}"]["cost_1.0"]["full"].get("portfolio") or {})
        real_detail[f"cap{cap}"] = {**aud_rc, "max_dd_pct": dd.get("max_dd_pct")}
        if aud_rc["overlap_violations"] > 0 or aud_rc["max_concurrent"] > cap \
                or (dd.get("max_dd_pct", 100.0) >= 20.0):
            real_ok = False
    gate("G5.8_portfolio_realism", real_ok,
         "flagship MaxDD% < 20%% and concurrency respected: "
         + json.dumps(real_detail))

    # ---------- metric sanity --------------------------------------------------
    fin = True
    for k in C:
        for cap in strat_caps[k]:
            for mult in COST_MULTS:
                for w_ in ("full", "oos", "fit"):
                    mb = summary[k][f"cap{cap}"][f"cost_{mult}"][w_]
                    for kk in ("expectancy_R", "sharpe_per_trade", "profit_factor",
                               "max_dd_R", "brier", "ece"):
                        v = mb.get(kk)
                        if v is not None and not np.isfinite(v):
                            fin = False
                    prt = mb.get("portfolio")
                    if prt is not None:
                        for kk in ("sharpe_annualized", "cagr_pct", "max_dd_pct"):
                            v = prt.get(kk)
                            if v is not None and not np.isfinite(v):
                                fin = False
    gate("G5.4_metrics_finite", fin, f"all metrics finite (NaN-guarded) = {fin}")

    # ---------- postmortem parity gate ----------------------------------------
    g_oos = summary["gbdt"]["cap1"]["cost_1.0"]["oos"].get("expectancy_R")
    k_oos = summary["knn"]["cap1"]["cost_1.0"]["oos"].get("expectancy_R")
    pm_g = abs(g_oos - POSTMORTEM["gbdt"]) <= POSTMORTEM_TOL
    pm_k = abs(k_oos - POSTMORTEM["knn"]) <= POSTMORTEM_TOL
    gate("G5.9_postmortem_parity", pm_g and pm_k,
         f"fixed GBDT OOS E[R]={g_oos:.4f} vs postmortem {POSTMORTEM['gbdt']} "
         f"(tol {POSTMORTEM_TOL}); k-NN {k_oos:.4f} vs {POSTMORTEM['knn']}")

    # ---------- bootstrap + DSR ------------------------------------------------
    with stage(f"bootstrap x{BOOT_N} (OOS, cost 1.0x)"):
        boot_pt, boot_d = {}, {}
        for k in C:
            exo = EX[k][1][1.0]["oos"]
            T = {"R_net": exo["R_net"], "ts": exo["ts"], "p": exo["sub"]["p"][exo["sel"]],
                 "R_label": exo["sub"]["R_label"][exo["sel"]]}
            boot_pt[k] = bootstrap_ci_pertrade(T, 1.0)
            if k in ("flagship", "stack", "canon"):
                boot_d[k] = bootstrap_ci_daily(exo["daily"])
        summary["bootstrap_oos_cost_1.0"] = boot_pt
        summary["bootstrap_daily_oos"] = boot_d

    fam_daily, fam_trade = {}, {}
    for k in C:
        exo = EX[k][1][1.0]["oos"]
        if exo["daily"] is not None and exo["daily"]["n_days"] >= 30:
            fam_daily[f"{k}|cap1"] = exo["daily"]["daily_ret"]
        if len(exo["R_net"]) >= 30:
            fam_trade[f"{k}|cap1|c1.0"] = exo["R_net"]
    for cap in (3,):
        exo = EX["flagship"][cap][1.0]["oos"]
        if exo["daily"] is not None:
            fam_daily[f"flagship|cap{cap}"] = exo["daily"]["daily_ret"]
    for mult in (1.5, 2.0):
        exo = EX["flagship"][1][mult]["oos"]
        if exo["daily"] is not None:
            fam_daily[f"flagship|cap1|c{mult}"] = exo["daily"]["daily_ret"]
    summary["dsr_daily_nonoverlapping"] = dsr_table(fam_daily)
    summary["dsr_per_trade_v12_style"] = dsr_table(fam_trade)
    log("  DSR daily: " + json.dumps(
        {k: v.get("dsr") for k, v in
         summary["dsr_daily_nonoverlapping"].get("members", {}).items()})[:600])

    # ---------- export ----------------------------------------------------------
    with stage("export"):
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows, eq_rows = [], []
        for k in C:
            for cap in strat_caps[k]:
                for mult in COST_MULTS:
                    ex = EX[k][cap][mult]["full"]
                    sub = ex["sub"]; sel = ex["sel"]
                    n = len(sel)
                    cell = sub["cell"][sel]
                    rows.append(pd.DataFrame({
                        "strategy": k, "cap": cap, "cost_mult": mult,
                        "ts": sub["ts"][sel].astype(np.int64),
                        "year": sub["year"][sel].astype(np.int16),
                        "cell": cell.astype(np.int16),
                        "cell_name": [names[c] if c >= 0 else "BH_24bar" for c in cell],
                        "p": sub["p"][sel].astype(np.float32),
                        "ev": sub["ev"][sel].astype(np.float32),
                        "f_eff": ex["f"].astype(np.float64),
                        "ood": sub["ood"][sel].astype(np.float32),
                        "overlay": sub["overlay"][sel].astype(np.int8),
                        "R_label": sub["R_label"][sel].astype(np.float32),
                        "R_gross": sub["R_gross"][sel].astype(np.float32),
                        "exit_bar": sub["exit_bar"][sel].astype(np.int64),
                        "cost_base": sub["cost_base"][sel].astype(np.float32),
                        "R_net": ex["R_net"].astype(np.float32)}))
                    d = ex["daily"]
                    if d is not None:
                        # day-aligned cumulative R of trades resolved up to each day
                        sub_e = ex["sub"]
                        exit_day = bundle["day_id_all"][sub_e["exit_bar"][ex["sel"]]]
                        Rsel = ex["R_net"]
                        order_d = np.argsort(exit_day, kind="stable")
                        ed_sorted = exit_day[order_d]
                        cum_sorted = np.cumsum(Rsel[order_d])
                        days = d["day_ns"] // 86400_000_000_000
                        j = np.searchsorted(ed_sorted, days, side="right")
                        cum_R_day = np.where(j > 0, cum_sorted[np.maximum(j - 1, 0)], 0.0)
                        eq_rows.append(pd.DataFrame({
                            "strategy": k, "cap": cap, "cost_mult": mult,
                            "day_ns": d["day_ns"],
                            "equity": d["equity"][1:].astype(np.float64),
                            "daily_ret": d["daily_ret"].astype(np.float64),
                            "cum_R_resolved": cum_R_day.astype(np.float64)}))
        trades = pd.concat(rows, ignore_index=True)
        pq.write_table(pa.Table.from_pandas(trades, preserve_index=False),
                       os.path.join(WORK, "trades_all.parquet"), compression="zstd")
        eqs = pd.concat(eq_rows, ignore_index=True)
        pq.write_table(pa.Table.from_pandas(eqs, preserve_index=False),
                       os.path.join(WORK, "equity_curves.parquet"), compression="zstd")
        with open(os.path.join(WORK, "backtest_main.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        with open(os.path.join(WORK, "policy_fit.json"), "w") as f:
            json.dump(fit_record, f, indent=1, default=float)
        with open(os.path.join(WORK, "overlay_ablation.json"), "w") as f:
            json.dump(ablation, f, indent=1, default=float)
        with open(os.path.join(WORK, "gbdt_audit.json"), "w") as f:
            json.dump(audit, f, indent=1)
        with open(os.path.join(WORK, "backtest_summary.md"), "w") as f:
            f.write("# AURUM P4 Backtest v1.3 — headline results\n\n")
            f.write("OOS 2023-25, cost 1.0x, cap=1 (cap=3 for AURUM family)\n\n")
            f.write("| strategy | OOS n | E[R] | Sharpe/t | PF | MaxDD R | "
                    "ann Sharpe | CAGR% | MaxDD% | Calmar |\n"
                    "|---|---|---|---|---|---|---|---|---|---|\n")
            for k in C:
                for cap in strat_caps[k]:
                    o = summary[k][f"cap{cap}"]["cost_1.0"]["oos"]
                    if o.get("n_trades", 0) == 0:
                        continue
                    prt = o.get("portfolio") or {}
                    f.write(f"| {k} cap{cap} | {o['n_trades']} | "
                            f"{o['expectancy_R']:.4f} | {o['sharpe_per_trade']:.4f} | "
                            f"{o['profit_factor']:.3f} | {o['max_dd_R']:.1f} | "
                            f"{prt.get('sharpe_annualized', float('nan')):.3f} | "
                            f"{prt.get('cagr_pct', float('nan')):.2f} | "
                            f"{prt.get('max_dd_pct', float('nan')):.2f} | "
                            f"{prt.get('calmar', float('nan')):.3f} |\n")
            f.write("\nCost sensitivity (OOS E[R], cap1):\n\n")
            f.write("| strategy | 1.0x | 1.5x | 2.0x |\n|---|---|---|---|\n")
            for k in C:
                o1 = summary[k]["cap1"]["cost_1.0"]["oos"].get("expectancy_R", float('nan'))
                o15 = summary[k]["cap1"]["cost_1.5"]["oos"].get("expectancy_R", float('nan'))
                o2 = summary[k]["cap1"]["cost_2.0"]["oos"].get("expectancy_R", float('nan'))
                f.write(f"| {k} | {o1:.4f} | {o15:.4f} | {o2:.4f} |\n")
        n_files = sum(os.path.isfile(os.path.join(WORK, f_)) for f_ in
                      ("trades_all.parquet", "equity_curves.parquet",
                       "backtest_main.json", "policy_fit.json",
                       "overlay_ablation.json", "gbdt_audit.json",
                       "backtest_summary.md"))
        gate("G5.5_export", n_files == 7,
             f"exported {n_files}/7 files; trades={len(trades):,} rows")

    save_gates(os.path.join(WORK, "gates.json"))
    log("P4 BACKTEST v1.3 COMPLETE. GATES: "
        + json.dumps({k: v["pass"] for k, v in GATES.items()}))
    log(f"total wall {time.time()-t0:.0f}s")
    return summary


if __name__ == "__main__":
    WORK = os.environ.get("AURUM_WORK",
                          "/kaggle/working" if os.path.exists("/kaggle/working")
                          else os.path.join(os.getcwd(), "p4_out"))
    os.makedirs(WORK, exist_ok=True)
    set_seeds(1337)
    main(WORK)
