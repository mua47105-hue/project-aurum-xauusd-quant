#!/usr/bin/env python3
"""
AURUM v1.3 DIAGNOSTIC — audit the v1.2 p4_backtest anomalies.

Reproduces on local data:
  1. GBDT / k-NN strategies under the v1.2 realized-excursion overlay
     -> expect E[R] ~ +0.63 / +0.53 on OOS 2023-25  (ANOMALY REPRODUCED)
  2. Same strategies with overlay OFF (clean barrier labels)
     -> expect ~ +0.06 / +0.01                      (postmortem parity)
  3. Decomposition of the synthetic payout.
  4. AURUM r=3 48-cell: overlay on/off.
  5. Ensemble OR-gate trade-count inflation.
Saves: kaggle_work/aurum/results/diag/diagnostic.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd

BASE = "/home/z/my-project/kaggle_work/aurum"
DS = f"{BASE}/phase1_ds"
P3 = f"{BASE}/results/p3_tables_latest"
OUT = f"{BASE}/results/diag"
os.makedirs(OUT, exist_ok=True)

BARRIER_STOPS = [1.0, 1.5, 2.0, 2.5]
BARRIER_TPS = [1.5, 2.5, 4.0]
BARRIER_HORS = [24, 48]
COHORT_GAP = 96
WF_YEARS = list(range(2016, 2026))
OOS_YEARS = [2023, 2024, 2025]
COST_BASE_R = 0.05
EV_TH = 0.0
OOD_TH = 0.10
KELLY_FRACTION = 0.25
HARD_EQUITY_RISK_CAP = 0.01
SHOCK_TH = 2.5
NFP_UTC_HOURS = (12, 13, 14)
CANON = {"L": "L_m1.0_t1.5_h48", "S": "S_m1.0_t1.5_h48"}


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


NAMES = cell_names()
DIRS, MSS, TPS, HRS = cell_axes()
MSS = np.array(MSS); TPS = np.array(TPS); HRS = np.array(HRS)
DIRS = np.array(DIRS)
B_GEO = TPS / MSS

t0 = time.time()
print("loading ...")
labels = pd.read_parquet(f"{DS}/labels_R_actual.parquet")
cls = pd.read_parquet(f"{DS}/labels_cls_actual.parquet")
exc = pd.read_parquet(f"{DS}/labels_excursions.parquet")
feat = pd.read_parquet(f"{DS}/h1_features.parquet")
if "ts" in feat.columns:
    feat = feat.drop(columns=["ts"])
cohort = np.load(f"{DS}/cohort_idx.npy")
tb = pd.read_parquet(f"{P3}/aurum_oos_prob_tables.parquet")
tb = tb[tb["variant"].astype(str) == "fold"].copy()

idx = labels.index
R_mat = labels.values.astype(np.float32)
year_all = pd.DatetimeIndex(idx).year.values.astype(np.int16)
ts_ns_all = idx.asi8
spec = json.load(open(f"{DS}/neutralization_spec.json"))
ci_L = NAMES.index(CANON["L"]); ci_S = NAMES.index(CANON["S"])
print(f"labels {R_mat.shape}, years {year_all.min()}-{year_all.max()}, fold rows {len(tb):,}")


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


def shock_blackout(pos):
    v = feat["mi_vix_d20z"].values
    return np.abs(v[pos]) >= SHOCK_TH


def empirical_overlay_q():
    q = {}
    for y in WF_YEARS:
        m = (year_all <= y - 1)
        q[y] = {f"{k}_{d}": float(np.nanquantile(exc[f"{k}_{d}48"].values[m], qv))
                for k, qv in (("mfe", 0.50), ("mae", 0.25)) for d in ("L", "S")}
    return q


EMP_Q = empirical_overlay_q()

# ---------------- GBDT walk-forward (v1.2 identical, seeded) ----------------
from xgboost import XGBClassifier

spec_cols = [c for c in spec["features"] if c in feat.columns]
X_all = feat[spec_cols].values.astype(np.float32)
yL_all = cls.values[:, ci_L]; yS_all = cls.values[:, ci_S]
finite = np.isfinite(X_all).all(1)
lab_rows = np.flatnonzero(finite & (np.isfinite(yL_all) | np.isfinite(yS_all)))
preds = {}
for y in WF_YEARS:
    tr = lab_rows[(year_all[lab_rows] <= y - 1)]
    n_val = max(200, int(len(tr) * 0.15))
    tr_core, val = tr[:-n_val], tr[-n_val:]
    out = {}
    for tag, ycol in (("L", yL_all), ("S", yS_all)):
        m_tr = np.isfinite(ycol[tr_core]); m_va = np.isfinite(ycol[val])
        clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            reg_lambda=10.0, subsample=0.7, colsample_bytree=0.7,
            min_child_weight=20, tree_method="hist", n_jobs=4,
            eval_metric="logloss", early_stopping_rounds=30,
            random_state=1337)
        clf.fit(X_all[tr_core[m_tr]], (ycol[tr_core[m_tr]] > 0).astype(int),
                eval_set=[(X_all[val[m_va]], (ycol[val[m_va]] > 0).astype(int))],
                verbose=False)
        te = np.flatnonzero((year_all == y) & finite & np.isfinite(ycol))
        p = np.full(len(idx), np.nan, dtype=np.float32)
        p[te] = clf.predict_proba(X_all[te])[:, 1]
        out["p" + tag] = p
    preds[y] = out
    print(f"  GBDT fold {y} done")
pL_full = np.full(len(idx), np.nan, np.float32)
pS_full = np.full(len(idx), np.nan, np.float32)
for y in WF_YEARS:
    m = year_all == y
    pL_full[m] = preds[y]["pL"][m]; pS_full[m] = preds[y]["pS"][m]

# ---------------- k-NN cohort (strict gap mask as v1.2 cohort_probs) --------
wf_pos = np.flatnonzero(np.isin(year_all, WF_YEARS))
yL = cls.values[:, ci_L]; yS = cls.values[:, ci_S]
CH = 4000
pL_c = np.full(len(wf_pos), np.nan, np.float32)
pS_c = np.full(len(wf_pos), np.nan, np.float32)
for s0 in range(0, len(wf_pos), CH):
    pos = wf_pos[s0:s0 + CH]
    nb = cohort[pos]
    ok = (nb >= 0) & (nb < (pos[:, None] - COHORT_GAP))
    for yy, out in ((yL, pL_c), (yS, pS_c)):
        vals = np.where(ok, yy[nb], np.nan)
        cnt = np.isfinite(vals).sum(1)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(vals, axis=1) if vals.size else np.array([])
        good = cnt >= 30
        out[s0:s0 + CH][good] = mean[good]
pL_cfull = np.full(len(idx), np.nan, np.float32); pL_cfull[wf_pos] = pL_c
pS_cfull = np.full(len(idx), np.nan, np.float32); pS_cfull[wf_pos] = pS_c


def cohort_gap_violations(n_check=200, seed=7):
    rng = np.random.default_rng(seed)
    spot = rng.choice(wf_pos, n_check, replace=False)
    raw_v = used_v = 0
    for p0 in spot:
        nb = cohort[p0]; nb = nb[nb >= 0]
        if len(nb) and nb.max() >= p0 - COHORT_GAP:
            raw_v += 1
        used = nb[nb < p0 - COHORT_GAP]
        if len(used) and used.max() >= p0 - COHORT_GAP:
            used_v += 1
    return raw_v, used_v


# ---------------- the v1.2 overlay engine (verbatim semantics) --------------
def v12_overlay(R_lab, mfe_r, mae_r, q50_mfe, q25_mae, m_arr):
    tighten = (mae_r >= q25_mae) & (q25_mae < m_arr)
    partial = (~tighten) & (mfe_r >= q50_mfe)
    R_g = R_lab.copy()
    R_g[tighten] = -q25_mae[tighten]
    R_g[partial] = 0.5 * q50_mfe[partial] + 0.5 * R_lab[partial]
    return R_g, tighten, partial


def metrics(R):
    R = np.asarray(R, np.float64)
    n = len(R)
    if n == 0:
        return {"n": 0}
    sd = R.std(ddof=1) if n > 1 else 0.0
    pos, neg = R[R > 0], R[R < 0]
    pf = float(pos.sum() / max(1e-9, -neg.sum())) if len(neg) and len(pos) else (999.0 if len(pos) else 0.0)
    cum = np.cumsum(R)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return {"n": int(n), "E[R]": round(float(R.mean()), 4),
            "hit": round(float((R > 0).mean()), 4),
            "sharpe_t": round(float(R.mean() / sd), 4) if sd > 0 else 0.0,
            "PF": round(min(pf, 999.0), 3),
            "maxDD_R": round(float((peak - cum).max()), 1)}


def baseline_flow(name, pL_full_arr, pS_full_arr, overlay_on):
    fin = np.isfinite(pL_full_arr) & np.isfinite(pS_full_arr) & np.isin(year_all, WF_YEARS)
    pos = np.flatnonzero(fin)
    ts = ts_ns_all[pos]; year = year_all[pos].astype(int)
    b = BARRIER_TPS[0] / BARRIER_STOPS[0]
    evL = pL_full_arr[pos] * b - (1 - pL_full_arr[pos])
    evS = pS_full_arr[pos] * b - (1 - pS_full_arr[pos])
    pick_L = evL >= evS
    cell = np.where(pick_L, ci_L, ci_S).astype(np.int16)
    p = np.where(pick_L, pL_full_arr[pos], pS_full_arr[pos]).astype(np.float32)
    ev = np.where(pick_L, evL, evS)
    R_lab = R_mat[pos, cell].astype(np.float64)
    d_ax = np.where(pick_L, "L", "S")
    mfe_r = np.array([exc[f"mfe_{d}48"].values[p0] for d, p0 in zip(d_ax, pos)])
    mae_r = np.array([exc[f"mae_{d}48"].values[p0] for d, p0 in zip(d_ax, pos)])
    q50 = np.array([EMP_Q[y][f"mfe_{d}"] for y, d in zip(year, d_ax)])
    q25 = np.array([EMP_Q[y][f"mae_{d}"] for y, d in zip(year, d_ax)])
    nfp = nfp_blackout(ts); shock = shock_blackout(pos)
    m_arr = np.float64(BARRIER_STOPS[0])
    if overlay_on:
        R_g, tig, par = v12_overlay(R_lab, mfe_r, mae_r, q50, q25, m_arr)
    else:
        R_g, tig, par = R_lab.copy(), np.zeros(len(pos), bool), np.zeros(len(pos), bool)
    kelly = ev / b
    f_k = np.clip(KELLY_FRACTION * kelly, 0.0, HARD_EQUITY_RISK_CAP)
    ok = (ev > EV_TH) & np.isfinite(R_lab) & np.isfinite(mfe_r) & np.isfinite(mae_r) \
        & (~nfp) & (~shock) & (f_k > 0)
    keep = np.flatnonzero(ok)
    R_net = R_g[keep] - (COST_BASE_R / m_arr)
    oos = np.isin(year[keep], OOS_YEARS)
    out = {
        "oos": metrics(R_net[oos]),
        "overlay_counts": {"n_tighten": int(tig[keep].sum()), "n_partial": int(par[keep].sum())},
        "partial_share_oos": float(par[keep][oos].mean()) if oos.sum() else None,
    }
    # synthetic payout accounting on OOS
    if overlay_on and oos.sum():
        no = par[keep][oos]
        Rl = R_lab[keep][oos]
        Rg = R_g[keep][oos]
        out["synthetic"] = {
            "mean_Rlab_on_partial": round(float(Rl[no].mean()), 4),
            "mean_Rg_on_partial": round(float(Rg[no].mean()), 4),
            "mean_gain_per_partial_trade": round(float((Rg[no] - Rl[no]).mean()), 4),
            "mean_Rlab_on_clean": round(float(Rl[~no].mean()), 4),
            "mean_Rg_on_clean": round(float(Rg[~no].mean()), 4),
        }
    return out


def aurum_flow(r, overlay_on, canonical_only=False, min_votes=1, use_vote=False):
    t = tb[tb["r"] == r].sort_values("ts").reset_index(drop=True)
    ts = t["ts"].values.astype(np.int64)
    pos = idx.get_indexer(pd.DatetimeIndex(ts.astype("datetime64[ns]"), tz="UTC"))
    year = t["year"].values.astype(int)
    if canonical_only:
        pL = t[f"h1_c{ci_L:02d}"].values.astype(np.float32)
        pS = t[f"h1_c{ci_S:02d}"].values.astype(np.float32)
        b = TPS[ci_L] / MSS[ci_L]
        evL, evS = pL * b - (1 - pL), pS * b - (1 - pS)
        pick_L = evL >= evS
        cell = np.where(pick_L, ci_L, ci_S).astype(np.int16)
        p = np.where(pick_L, pL, pS)
        ev = np.where(pick_L, evL, evS)
    else:
        cell = t["ev_cell_raw"].values.astype(np.int16)
        p = t["p_evcell_raw"].values.astype(np.float32)
        ev = t["ev_raw"].values.astype(np.float32)
    R_lab = R_mat[pos, cell].astype(np.float64)
    d_ax = DIRS[cell]
    mfe_r = np.empty(len(t)); mae_r = np.empty(len(t))
    for d_ in ("L", "S"):
        for h_ in (24, 48):
            msk = (d_ax == d_) & (HRS[cell] == h_)
            if msk.any():
                mfe_r[msk] = exc[f"mfe_{d_}{h_}"].values[pos[msk]]
                mae_r[msk] = exc[f"mae_{d_}{h_}"].values[pos[msk]]
    q50 = np.where(d_ax == "L", t["h3_1"].values, t["h3_7"].values).astype(np.float32)
    q25 = np.where(d_ax == "L", t["h3_3"].values, t["h3_9"].values).astype(np.float32)
    ood = t["ood_mean"].values.astype(np.float32)
    nfp = nfp_blackout(ts); shock = shock_blackout(pos)
    d_aurum = np.where(d_ax == "L", "L", "S")
    # v1.2 plain AURUM required GBDT agreement; ens used OR-gate min_votes=1
    g_dir = np.where(np.isnan(pL_full) | np.isnan(pS_full), "N",
                     np.where(pL_full * (TPS[ci_L]/MSS[ci_L]) - (1 - pL_full)
                              >= pS_full * (TPS[ci_S]/MSS[ci_S]) - (1 - pS_full), "L", "S")).astype(object)
    c_dir = np.where(np.isnan(pL_cfull) | np.isnan(pS_cfull), "N",
                     np.where(pL_cfull * (TPS[ci_L]/MSS[ci_L]) - (1 - pL_cfull)
                              >= pS_cfull * (TPS[ci_S]/MSS[ci_S]) - (1 - pS_cfull), "L", "S")).astype(object)
    g = g_dir[pos]; c = c_dir[pos]
    if use_vote:
        votes = (g == d_aurum).astype(int) + (c == d_aurum).astype(int)
        agree = votes >= min_votes
    else:
        agree = (g == d_aurum)
    ok = (ev > EV_TH) & np.isfinite(R_lab) & np.isfinite(mfe_r) & np.isfinite(mae_r) \
        & np.isfinite(p) & (p > 0.01) & (p < 0.99) & (ood <= OOD_TH) & (~nfp) & (~shock) & agree
    m_arr = MSS[cell]
    if overlay_on:
        R_g, tig, par = v12_overlay(R_lab, mfe_r, mae_r, q50, q25, m_arr)
    else:
        R_g, tig, par = R_lab.copy(), np.zeros(len(t), bool), np.zeros(len(t), bool)
    b_geo = TPS[cell] / MSS[cell]
    kelly = (p * b_geo - (1 - p)) / b_geo
    f_k = np.clip(KELLY_FRACTION * kelly, 0.0, HARD_EQUITY_RISK_CAP)
    keep = np.flatnonzero(ok & (f_k > 0))
    R_net = R_g[keep] - (COST_BASE_R / m_arr[keep])
    oos = np.isin(year[keep], OOS_YEARS)
    return {"oos": metrics(R_net[oos]),
            "overlay_counts": {"n_tighten": int(tig[keep].sum()),
                               "n_partial": int(par[keep].sum())}}


print("\n" + "=" * 78)
print("AUDIT 1: GBDT / k-NN under v1.2 overlay vs clean (OOS 2023-25, cost 1.0x)")
print("=" * 78)
res = {}
res["gbdt_overlayON"] = baseline_flow("gbdt", pL_full, pS_full, True)
res["gbdt_overlayOFF"] = baseline_flow("gbdt", pL_full, pS_full, False)
res["knn_overlayON"] = baseline_flow("knn", pL_cfull, pS_cfull, True)
res["knn_overlayOFF"] = baseline_flow("knn", pL_cfull, pS_cfull, False)
for k in ("gbdt_overlayON", "gbdt_overlayOFF", "knn_overlayON", "knn_overlayOFF"):
    print(f"{k:18s} {res[k]['oos']}  overlay={res[k]['overlay_counts']}")
    if "synthetic" in res[k]:
        print(f"    synthetic: {res[k]['synthetic']}")

print("\n" + "=" * 78)
print("AUDIT 2: AURUM r=3 48-cell / canonical — overlay on/off")
print("=" * 78)
res["aurum_r3_overlayON"] = aurum_flow(3, True)
res["aurum_r3_overlayOFF"] = aurum_flow(3, False)
res["aurum_canon_overlayON"] = aurum_flow(3, True, canonical_only=True)
res["aurum_canon_overlayOFF"] = aurum_flow(3, False, canonical_only=True)
for k in ("aurum_r3_overlayON", "aurum_r3_overlayOFF",
          "aurum_canon_overlayON", "aurum_canon_overlayOFF"):
    print(f"{k:24s} {res[k]['oos']}  overlay={res[k]['overlay_counts']}")

print("\n" + "=" * 78)
print("AUDIT 3: 'Ensemble' OR-gate trade-count inflation (v1.2 semantics)")
print("=" * 78)
plain = aurum_flow(3, True)                 # AND: GBDT must agree (v1.2 plain)
ens = aurum_flow(3, True, use_vote=True, min_votes=1)   # OR-gate
print(f"plain AURUM r3 (GBDT-agree)  oos={plain['oos']}")
print(f"ens  (GBDT-OR-cohort, v=1)   oos={ens['oos']}")
res["ens_inflation"] = {"plain": plain["oos"], "ens_or_gate": ens["oos"]}

print("\n" + "=" * 78)
print("AUDIT 4: cohort gap — raw vs masked (k-NN neighbour audit)")
print("=" * 78)
raw_v, used_v = cohort_gap_violations(300)
print(f"300-state spot check: raw violations={raw_v}, used (post-mask) violations={used_v}")
res["cohort_gap"] = {"raw_violations": raw_v, "used_violations": used_v, "n_check": 300}

res["_meta"] = {"wall_s": round(time.time() - t0, 1),
                "note": "v1.2 semantics reproduced from pulled Kaggle artifacts; "
                        "postmortem refs: kNN +0.0078R, GBDT +0.0599R (2023-25)"}
with open(f"{OUT}/diagnostic.json", "w") as f:
    json.dump(res, f, indent=1)
print(f"\nsaved {OUT}/diagnostic.json  (wall {time.time()-t0:.0f}s)")
