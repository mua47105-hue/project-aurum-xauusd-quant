#!/usr/bin/env python3
"""Local smoke test for p4_backtest.py with fully synthetic data.
Builds mini phase1-data + p3-output mounts, runs the kernel's main(),
checks all G5.x gates and output files."""
import os, sys, json, shutil
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/z/my-project/kaggle_work/aurum/src/kernels")

ROOT = "/home/z/my-project/kaggle_work/aurum"
SMOKE = f"{ROOT}/smoke_p4"
DS = f"{SMOKE}/phase1"
P3 = f"{SMOKE}/p3out"
WORK = f"{SMOKE}/work"

for d in (DS, P3, WORK):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)

rng = np.random.default_rng(42)
sys.path.insert(0, f"{ROOT}/src/kernels")
import p4_backtest as K

# ---------------- synthetic phase1-data ----------------
# 2015-01-01 .. 2025-12-31 hourly weekdays -> ~11 years
idx = pd.date_range("2015-01-01", "2025-12-31 23:00", freq="h", tz="UTC")
idx = idx[idx.dayofweek < 5]                      # drop weekends
N = len(idx)
print(f"synthetic bars: {N} ({idx[0]} .. {idx[-1]})")

names = K.cell_names()
# labels: 48 cells, R in [-1, 3], NaN for last 60 bars and randomly 2%
R = rng.uniform(-1, 3, size=(N, 48)).astype(np.float32)
R[rng.random((N, 48)) < 0.02] = np.nan
R[-60:] = np.nan
labels = pd.DataFrame(R, index=idx, columns=names)
labels.to_parquet(f"{DS}/labels_R_actual.parquet")

exc_cols = {}
for d in ("L", "S"):
    for h in (24, 48):
        exc_cols[f"mfe_{d}{h}"] = np.abs(rng.uniform(0, 4, N))
        exc_cols[f"mae_{d}{h}"] = np.abs(rng.uniform(0, 2.5, N))
        exc_cols[f"res_{d}{h}"] = rng.uniform(-1, 3, N)
exc_cols["mfe_L48"][-60:] = np.nan
pd.DataFrame(exc_cols, index=idx).to_parquet(f"{DS}/labels_excursions.parquet")

cls = (rng.random((N, 48)) < 0.43).astype(np.float32)
cls[R != R] = np.nan  # align NaN
pd.DataFrame(cls, index=idx, columns=names).to_parquet(f"{DS}/labels_cls_actual.parquet")

feat = {}
spec_feats = [f"f{i:02d}" for i in range(44)]
for c in spec_feats:
    feat[c] = rng.normal(0, 1, N)
feat["mi_vix_d20z"] = rng.normal(0, 1, N)          # some >= 2.5 shocks
feat["ret24_atr"] = rng.normal(0, 2, N)
feat["c"] = 2000 + np.cumsum(rng.normal(0, 2, N))
feat["h"] = feat["c"] + np.abs(rng.normal(0, 1, N))
feat["l"] = feat["c"] - np.abs(rng.normal(0, 1, N))
featdf = pd.DataFrame(feat, index=idx)
featdf.to_parquet(f"{DS}/h1_features.parquet")

# price parquet (ohlcv) for EMA/Donchian baselines
pd.DataFrame({"o": feat["c"], "h": feat["h"], "l": feat["l"], "c": feat["c"],
              "v": np.zeros(N)}, index=idx).to_parquet(f"{DS}/h1_xauusd.parquet")

# cohort: 200 neighbours, each a random bar >= 96 older (p1 protocol)
coh = np.full((N, 200), -1, dtype=np.int32)
for p in range(N):
    lo = max(0, p - 5000)
    hi = p - K.COHORT_GAP
    if hi > lo + 10:
        k = rng.integers(20, 200)
        coh[p, :k] = rng.integers(lo, hi, k)
np.save(f"{DS}/cohort_idx.npy", coh)

json.dump({"protocol": "test", "features": spec_feats},
          open(f"{DS}/neutralization_spec.json", "w"))

# ---------------- synthetic p3 output ----------------
years = pd.DatetimeIndex(idx).year.values
fold_mask = (years >= 2016) & (years <= 2025)
fold_idx = np.flatnonzero(fold_mask)
rows = []
for r in K.P4_DEPTHS:
    n = len(fold_idx)
    P48 = rng.uniform(0.05, 0.95, size=(n, 48)).astype(np.float16)
    b_geo = np.array([K.b_geo_of(j) for j in range(48)])
    ev = P48.astype(np.float32) * b_geo[None, :] - (1 - P48.astype(np.float32))
    cell = ev.argmax(1)
    pb = ev[np.arange(n), cell]
    tbl = {
        "ts": idx.asi8[fold_idx].astype(np.int64),
        "year": years[fold_idx].astype(np.int16),
        "r": np.full(n, r, np.int8),
        "variant": ["fold"] * n,
        "f_era": rng.uniform(0, 1, n).astype(np.float32),
        "ev_cell_raw": cell.astype(np.int16),
        "p_evcell_raw": P48.astype(np.float32)[np.arange(n), cell],
        "ev_raw": pb.astype(np.float32),
        "ev_cell_neut": cell.astype(np.int16),
        "p_evcell_neut": np.clip(
            P48.astype(np.float32)[np.arange(n), cell]
            + rng.normal(0, 0.02, n), 0.01, 0.99).astype(np.float32),
        "ev_neut": (pb + rng.normal(0, 0.01, n)).astype(np.float32),
        "ood_mean": (np.zeros(n, np.float32) if r == 3
                     else np.abs(rng.uniform(0, 0.2, n)).astype(np.float32)),
        "ood_max": rng.uniform(0, 0.4, n).astype(np.float32),
    }
    for j in range(48):
        tbl[f"h1_c{j:02d}"] = P48[:, j]
    for j in range(12):
        tbl[f"h3_{j}"] = np.abs(rng.uniform(0.1, 2.0, n)).astype(np.float16)
    rows.append(pd.DataFrame(tbl))
tb_all = pd.concat(rows, ignore_index=True)
# shuffle to test the ts-sort path
tb_all = tb_all.sample(frac=1.0, random_state=1).reset_index(drop=True)
tb_all.to_parquet(f"{P3}/aurum_oos_prob_tables.parquet")
json.dump({"G4.2_depth_forwards": {"pass": True, "detail": "synthetic"},
           "G4.4_tables": {"pass": True, "detail": "synthetic"}},
          open(f"{P3}/gates.json", "w"))
json.dump({"row_counts": {"fold": int(len(tb_all) / 4)}},
          open(f"{P3}/p3_cell_meta.json", "w"))

# ---------------- run the kernel ----------------
os.environ["AURUM_DS"] = DS
os.environ["AURUM_P3"] = P3
os.environ["AURUM_WORK"] = WORK
K.main(WORK)

# ---------------- checks ----------------
gates = json.load(open(f"{WORK}/gates.json"))
print("\n=== GATES ===")
allok = True
for k, v in gates.items():
    print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} | {str(v['detail'])[:110]}")
    allok &= v["pass"]
summ = json.load(open(f"{WORK}/backtest_main.json"))
strats = [k for k in summ if k.startswith(("aurum", "ens", "gbdt", "cohort",
                                           "ema", "donchian", "buyhold"))]
print(f"\nstrategies: {len(strats)}")
for s in strats:
    o = summ[s]["cost_1.0"]["oos"]
    print(f"  {s:14s} oos_n={o.get('n_trades', 0):5d} "
          f"E[R]={o.get('expectancy_R', float('nan')):+.4f} "
          f"PF={o.get('profit_factor', float('nan')):.3f}")
files = ["trades_all.parquet", "equity_curves.parquet", "backtest_main.json",
         "backtest_summary.md", "gbdt_audit.json"]
missing = [f for f in files if not os.path.isfile(f"{WORK}/{f}")]
print(f"\nmissing files: {missing}")
tr = pd.read_parquet(f"{WORK}/trades_all.parquet")
print(f"trades_all: {len(tr):,} rows, {tr.strategy.nunique()} strategies, "
      f"NaN in R_net_1x: {int(tr.R_net_1x.isna().sum())}")
assert allok, "SOME GATES FAILED"
assert not missing, "MISSING FILES"
assert tr.R_net_1x.isna().sum() == 0
print("\nSMOKE TEST PASSED")
