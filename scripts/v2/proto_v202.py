"""Spec v2.0.2 prototype: de-saturated z_rff + M6/M7 active-stack preview.

Validates, BEFORE touching the kernels:
  1. causal per-fold trailing standardization of pred_rff
     z = tanh((p - EWMA_504(p)) / (2 * max(EWSTD_504(p), eps)))
  2. G4.8 producer checks: std(z)>0.05, P(|z|>0.99)<5%, OOS frozen turnover>2.0
  3. resulting M2 engine/frozen FIT & OOS Sharpes
  4. M6 fixed-family shrunk-MV weights (engine-FIT SRs, 90% shrink to equal)
  5. M7_Active_Grand_Hybrid vs M7_Core_Satellite OOS preview
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/z/my-project/aurum_v2/model")
os.environ.setdefault("AURUM_V2F", "/home/z/my-project/kaggle_work/v2/k1_out4")
os.environ.setdefault("AURUM_DS", "/home/z/my-project/kaggle_work/aurum/dataset_p2")

from v2_common import (BARS_PER_YEAR, FIT_YEARS, OOS_YEARS, frozen_net,
                       load_v2_data, run_online_engine)

K2 = "/home/z/my-project/kaggle_work/v2/k2_out2"
EPS = 1e-9


def zrff_v202(pred: pd.Series) -> pd.Series:
    """Per-fold (calendar year = walk-forward test slice) causal trailing
    standardization.  The standardizer re-anchors at each yearly re-fit
    boundary: the model is replaced, its prediction scale is a new random
    variable, and carrying the dead model's mean/std across the boundary
    injects a known artifact (Engineering Flaw #8 root cause)."""
    out = pd.Series(np.nan, index=pred.index)
    for yr, g in pred.groupby(pred.index.year):
        p = g.dropna()
        if len(p) < 130:
            continue
        ewma = p.ewm(span=504, min_periods=63).mean()
        ewstd = p.ewm(span=504, min_periods=63).std()
        x = (p - ewma) / (2.0 * np.maximum(ewstd, EPS))
        out.loc[p.index] = np.tanh(x)
    return out


def zrff_foldiqr(pred: pd.Series) -> pd.Series:
    """Negative-control variant: fold mean RETAINED in the numerator, robust
    expanding IQR scale (causal within fold).  Expected to saturate wherever
    |fold drift| >> fold IQR -- kept as a documented diagnostic only."""
    out = pd.Series(np.nan, index=pred.index)
    for yr, g in pred.groupby(pred.index.year):
        p = g.dropna()
        if len(p) < 130:
            continue
        vals = p.values
        res = np.full(len(vals), np.nan)
        for i in range(len(vals)):
            if i < 63:
                continue
            w = vals[: i + 1]
            q75, q25 = np.percentile(w, [75, 25])
            scale = max((q75 - q25) / 1.349, EPS)
            res[i] = np.tanh(vals[i] / scale)
        out.loc[p.index] = res
    return out


def main():
    feats, px = load_v2_data()
    r = feats["r_xau"]
    sigma = feats["sigma"]
    yr = pd.Series(feats.index.year, index=feats.index)
    fit_mask = yr.isin(FIT_YEARS)
    oos_mask = yr.isin(OOS_YEARS)

    fc = pd.read_parquet(os.path.join(K2, "v2_cpu_forecasts.parquet"))
    fc.index = pd.DatetimeIndex(fc.index)
    if fc.index.tz is None:
        fc.index = fc.index.tz_localize("UTC")

    z_new = zrff_v202(fc["pred_rff"])
    z_iqr = zrff_foldiqr(fc["pred_rff"])

    print("=== G4.8 producer checks (primary z_rff v2.0.2) ===")
    for wname, mask in (("fit", fit_mask), ("oos", oos_mask)):
        sel = z_new.dropna().index
        sel = sel[mask.reindex(sel).fillna(False)]
        v = z_new[sel].values
        sat = float((np.abs(v) > 0.99).mean())
        fro = frozen_net(z_new[sel], r[sel], sigma[sel], 1.0)
        print(f"  [{wname}] n={len(v)} std={np.std(v):.4f} "
              f"P(|z|>0.99)={sat:.4%} nuniq={pd.unique(v).size} "
              f"frozen_to={fro['turnover_py']:.1f}/y "
              f"frozen_sh={fro['metrics']['sharpe_ann']:.3f}")
    print("=== variant z_rff_foldiqr (negative control) ===")
    for wname, mask in (("fit", fit_mask), ("oos", oos_mask)):
        sel = z_iqr.dropna().index
        sel = sel[mask.reindex(sel).fillna(False)]
        v = z_iqr[sel].values
        sat = float((np.abs(v) > 0.99).mean())
        print(f"  [{wname}] std={np.std(v):.4f} P(|z|>0.99)={sat:.4%} "
              f"nuniq={pd.unique(v).size}")

    print("=== M2_RFF_Ridge v2.0.2 evaluation ===")
    for wname, mask in (("fit", fit_mask), ("oos", oos_mask)):
        sel = z_new.dropna().index
        sel = sel[mask.reindex(sel).fillna(False)]
        zw, rw, sw = z_new[sel], r[sel], sigma[sel]
        eng = run_online_engine(zw, rw, sw, 1.0)
        fro = frozen_net(zw, rw, sw, 1.0)
        print(f"  [{wname}] engine_sh={eng['metrics']['sharpe_ann']:.3f} "
              f"to={eng['turnover_py']:.1f} | frozen_sh="
              f"{fro['metrics']['sharpe_ann']:.3f} to={fro['turnover_py']:.1f}")

    # ---- M6 fixed-family weights preview (engine-FIT SRs) -----------------
    # M1 numbers come from the cached K6 gauntlet (z unchanged);
    # M2 engine FIT Sharpe recomputed above.
    import json
    g = json.load(open("/home/z/my-project/kaggle_work/v2/k6_out1/gauntlet.json"))
    m1_eng = g["M1_GBDT_v2"]["engine_fit_sharpe"]
    sel = z_new.dropna().index
    sel = sel[fit_mask.reindex(sel).fillna(False)]
    m2_eng = run_online_engine(z_new[sel], r[sel], sigma[sel], 1.0)["metrics"]["sharpe_ann"]
    print(f"\n=== M6 weight preview: engine FIT SRs M1={m1_eng:.4f} M2={m2_eng:.4f}")

    # member FIT frozen net streams (weight-estimation streams)
    streams = {}
    for mname, zseries in (("M1_GBDT_v2", fc["z_gbdt"]),
                           ("M2_RFF_Ridge", z_new)):
        z = zseries.dropna()
        s = z.index[fit_mask.reindex(z.index).fillna(False)]
        streams[mname] = frozen_net(zseries[s], r[s], sigma[s], 1.0)["net"]
    sdf = pd.DataFrame(streams).dropna()
    sd = sdf.std()
    sr = pd.Series({"M1_GBDT_v2": m1_eng, "M2_RFF_Ridge": m2_eng})
    sr_shrunk = 0.10 * sr + 0.90 * sr.mean()   # 90% toward equal-weight prior
    mu_shrunk = sr_shrunk * sd * BARS_PER_YEAR
    C = sdf.corr()
    C_shrunk = 0.50 * C + 0.50 * np.eye(len(C))
    sdv = sd.values
    Cov = C_shrunk.values * np.outer(sdv, sdv) * BARS_PER_YEAR
    w_raw = np.linalg.solve(Cov + 1e-6 * np.eye(len(Cov)), mu_shrunk.values)
    w = np.clip(w_raw, 0.0, None)
    w = w / w.sum() if w.sum() > 1e-9 else np.ones(len(w)) / len(w)
    print(f"  shrunk SRs: {dict(sr_shrunk.round(4))}")
    print(f"  weights: M1={w[0]:.4f} M2={w[1]:.4f} (raw {w_raw.round(4)})")

    # ---- M7 preview --------------------------------------------------------
    z_champ = None
    import glob
    trades = pd.read_parquet(
        "/home/z/my-project/kaggle_work/v2/v13_art/trades_all.parquet")
    idx = feats.index
    zc = np.zeros(len(idx))
    sub = trades[(trades["strategy"] == "flagship") & (trades["cap"] == 3)
                 & (trades["cost_mult"] == 1.0)]
    for row in sub.itertuples():
        d = 1.0 if row.cell_name.startswith("L") else -1.0
        i0 = idx.searchsorted(pd.Timestamp(row.ts, unit="ns", tz="UTC"))
        i1 = min(len(idx), max(int(row.exit_bar), i0 + 1))
        if i0 < len(idx):
            zc[i0:i1] += d / 3.0
    z_champ = pd.Series(np.clip(zc, -1, 1), index=idx)

    z_m6 = (z_src := pd.concat(
        [fc["z_gbdt"].rename("M1"), z_new.rename("M2")], axis=1
    ).ffill().fillna(0.0).values @ w)
    z_m6 = pd.Series(z_m6, index=idx)
    z_active = 0.5 * z_champ + 0.5 * z_m6
    z_core = 0.5 * z_champ + 0.5 * 1.0
    print("\n=== M7 preview (frozen accounting, OOS 2023-25) ===")
    for name, z in (("M7_Active_Grand_Hybrid", z_active),
                    ("M7_Core_Satellite", z_core),
                    ("M0_Champ_v13", z_champ),
                    ("M6_AURUM_v2_Ensemble", z_m6)):
        sel = z.dropna().index
        sel = sel[oos_mask.reindex(sel).fillna(False)]
        fro = frozen_net(z[sel], r[sel], sigma[sel], 1.0)
        c2 = frozen_net(z[sel], r[sel], sigma[sel], 2.0)
        print(f"  {name:24s} OOS sh={fro['metrics']['sharpe_ann']:.4f} "
              f"cost2x={c2['metrics']['sharpe_ann']:.4f} "
              f"to={fro['turnover_py']:.1f}")


if __name__ == "__main__":
    main()
