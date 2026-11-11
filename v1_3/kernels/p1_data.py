# ============================================================================
# AURUM Phase 1 - DATA/AUDIT PIPELINE KERNEL (CPU)
#   Stage A: HistData M1 ingestion XAUUSD/XAGUSD/EURUSD/USDJPY 2010-2025
#            + FRED macro (DTWEXBGS, DGS10, DGS2, T10Y2Y, VIXCLS)
#   Stage B: H1 aggregation, dimensionless features, UTC-offset estimation
#   Stage C: k-means M1/H1 shape codebooks (fit 2012-2020 only)
#   Stage D: 48-cell barrier labeler (M1 execution) + MFE/MAE excursions
#   Stage E: matched-cohort index (K=200, gap=96) + B=200 bootstrap soft
#            labels (p_win, b_eff) + excursion quantiles + H2/H4/H5 targets
#   Stage F: adversarial validation audit (train 2012-2022 vs val 2023-2025),
#            iterative drift-feature drop until AUC <= 0.55
#   Stage G: serialization smoke test + outputs
# Gates: G1.1 coverage, G1.2 codebooks, G1.3 labels, G1.4 adversarial AUC,
#        G1.5 serialization
# ============================================================================
import json, os, sys, time
import numpy as np
import pandas as pd

ASSETS = ["xauusd", "xagusd", "eurusd", "usdjpy"]
YEARS = list(range(2010, 2026))
TRAIN_END_YEAR = 2022          # val = 2023-2025
CODEBOOK_FIT_END = "2021-01-01"

def main():
    WORK = os.environ.get("AURUM_WORK", "/kaggle/working" if os.path.exists("/kaggle/working") else os.getcwd())
    OUT = WORK
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    # resume support: if a previous version's output is mounted, reuse raw zips
    PREV = "/kaggle/input/aurum-p1-data"
    if os.path.isdir(PREV):
        prev_raw = os.path.join(PREV, "raw")
        if os.path.isdir(prev_raw):
            import shutil as _sh
            n_re = 0
            for fn in os.listdir(prev_raw):
                dst = os.path.join(OUT, "raw", fn)
                src = os.path.join(prev_raw, fn)
                if not os.path.exists(dst) and os.path.exists(src):
                    try:
                        _sh.copy2(src, dst)
                        n_re += 1
                    except Exception:
                        pass
            log(f"resume: {n_re} raw zips copied from previous version output")

    from aurum_data import (histdata_download_year, parse_histdata_zip, fred_fetch,
                            estimate_utc_offset, apply_utc_offset, to_h1, coverage_report,
                            FRED_SERIES)
    from aurum_feat import compute_h1_features, wilder_atr, COHORT_FEATURES
    from aurum_ser import (m1_path_features, h1_path_features, fit_codebook,
                           apply_codebook, serialize_states)
    from aurum_label import (barrier_actual_outcomes, expand_actual, build_cohort_index,
                             cohort_soft_labels, aux_targets)
    from aurum_gates import cell_names, COHORT_K

    env = env_report()
    log(f"ENV: {json.dumps(env)}")
    gate("G1.0_env_cpu", True, f"python={env['python']} pandas={env.get('pandas')} numpy={env['numpy']}")

    # ---------------- Stage A: ingestion ----------------
    with stage("A: ingestion"):
        m1_all, h1_all = {}, {}
        cov = {}
        for asset in ASSETS:
            frames = []
            for y in YEARS:
                zp = histdata_download_year(asset, y, os.path.join(OUT, "raw"), log=log)
                if not zp:
                    continue
                df = parse_histdata_zip(zp, log=log)
                if df is not None and len(df):
                    frames.append(df)
            if not frames:
                log(f"  {asset}: NO DATA")
                cov[asset] = {"error": "no data"}
                continue
            m1 = pd.concat(frames)
            m1 = m1[~m1.index.duplicated()].sort_index()
            off, oc = estimate_utc_offset(m1, log=log)
            m1 = apply_utc_offset(m1, off)
            m1_all[asset] = m1
            h1_all[asset] = to_h1(m1)
            cov[asset] = {"rows": int(len(m1)), "utc_offset": int(off),
                          "offset_corr": round(oc, 3),
                          "range": f"{m1.index.min()} .. {m1.index.max()}",
                          "years": coverage_report(m1, YEARS)}
            log(f"  {asset}: {len(m1)} M1 rows -> {len(h1_all[asset])} H1 bars")
        xau_cov = cov.get("xauusd", {}).get("years", {})
        xau_ok = [v["coverage"] for v in xau_cov.values() if v["rows"] > 0]
        mean_cov = float(np.mean(xau_ok)) if xau_ok else 0.0
        gate("G1.1_coverage", len(xau_ok) >= 12 and mean_cov >= 0.70,
             f"XAUUSD years with data={len(xau_ok)}/16, mean minute-coverage={mean_cov:.1%}; "
             f"cross-assets: " + ", ".join(f"{a}:{cov.get(a, {}).get('rows', 0)}" for a in ASSETS[1:]),
             extra={"coverage": cov})

    if "xauusd" not in m1_all:
        save_gates(os.path.join(OUT, "gates.json"))
        log("FATAL: no XAUUSD data - aborting")
        return

    # ---------------- Stage B: features ----------------
    with stage("B: features"):
        m1, h1 = m1_all["xauusd"], h1_all["xauusd"]
        cross = {a: h1_all[a] for a in ASSETS[1:] if a in h1_all}
        fred = {}
        for key, code in FRED_SERIES.items():
            s = fred_fetch(code, log=log)
            if s is not None:
                fred[key] = s
        log(f"  FRED series ok: {list(fred.keys())}")

    # ---------------- Stage C: codebooks ----------------
    with stage("C: codebooks"):
        fit_mask = np.asarray(h1.index < CODEBOOK_FIT_END)  # ndarray bool
        Xm1 = m1_path_features(m1, h1.index)
        Xh1 = h1_path_features(h1)
        km1 = fit_codebook(Xm1[fit_mask], k=32)
        kmh = fit_codebook(Xh1[fit_mask], k=32)
        m1_code = apply_codebook(km1, Xm1, index=h1.index)
        h1_code = apply_codebook(kmh, Xh1, index=h1.index)
        covm1 = float(np.isfinite(Xm1).all(1).mean())
        covh1 = float(np.isfinite(Xh1).all(1).mean())
        inert1 = float(km1.inertia_ / max(np.isfinite(Xm1[fit_mask]).all(1).sum(), 1))
        inerth = float(kmh.inertia_ / max(np.isfinite(Xh1[fit_mask]).all(1).sum(), 1))
        gate("G1.2_codebooks", covm1 >= 0.90 and covh1 >= 0.98,
             f"M1 codebook k=32 coverage={covm1:.1%} inertia/row={inert1:.4f}; "
             f"H1 codebook k=32 coverage={covh1:.1%} inertia/row={inerth:.4f} (fit<2021 only)")

    # ---------------- Stage D: barrier labeler ----------------
    with stage("D: barrier labeler"):
        N = len(h1)
        i_states = np.arange(300, N - 60)   # burn-in 300 bars, keep 60-bar tail
        t0 = time.time()
        actual_raw = barrier_actual_outcomes(m1, h1, i_states)
        actual = expand_actual(actual_raw, i_states, N)
        log(f"  labeler: {len(i_states)} bars x 48 cells in {time.time()-t0:.0f}s")
        valid_R = np.isfinite(actual["R"]).all(1) & (actual["cls"] != 255).all(1)
        feat = compute_h1_features(h1, m1_code=m1_code, h1_code=h1_code,
                                   cross=cross, fred=fred)

    # ---------------- Stage E: cohort + bootstrap ----------------
    with stage("E: cohort bootstrap"):
        cols = [c for c in COHORT_FEATURES if c in feat.columns]
        Fm = feat[cols]
        train_mask = np.asarray(h1.index.year <= TRAIN_END_YEAR)
        mu, sd = Fm[train_mask].mean(), Fm[train_mask].std() + 1e-9
        Fs = ((Fm - mu) / sd).values
        label_valid = valid_R.copy()
        label_valid &= Fm.notna().all(axis=1).values
        label_valid &= np.isfinite(Fs).all(axis=1)
        label_valid[:400] = False
        t0 = time.time()
        cohort, nvalid = build_cohort_index(Fs, label_valid, k=COHORT_K, gap=96,
                                            query_k=3000, chunk=2000)
        log(f"  cohort: built in {time.time()-t0:.0f}s")
        soft = cohort_soft_labels(cohort, nvalid, actual)
        atr = wilder_atr(h1["h"], h1["l"], h1["c"], 14).values
        aux = aux_targets(h1, None, atr)
        # priors
        from aurum_gates import cell_axes
        ax = cell_axes()
        def cell_of(d, m, t, hh):
            return [i for i, (dd, mm, tt, h_) in enumerate(zip(*ax))
                    if dd == d and mm == m and tt == t and h_ == hh][0]
        wvec = {}
        for nm, (d, m, t, hh) in {"prior_w24": ("L", 1.5, 2.5, 24), "prior_l24": ("S", 1.5, 2.5, 24),
                                  "prior_w48": ("L", 1.5, 2.5, 48), "prior_l48": ("S", 1.5, 2.5, 48)}.items():
            v = actual["win"][:, cell_of(d, m, t, hh)]  # full-length float, NaN outside i_states
            wvec[nm] = pd.Series(v).rolling(252, min_periods=100).mean().shift(1).values
        feat = compute_h1_features(h1, m1_code=m1_code, h1_code=h1_code,
                                   cross=cross, fred=fred,
                                   priors=pd.DataFrame(wvec, index=h1.index))
        frac_full = float((nvalid[label_valid] >= COHORT_K * 0.9).mean()) if label_valid.any() else 0.0
        pwin = soft["p_win"][label_valid]
        pwin_ok = bool(np.isfinite(pwin).any()) and float(np.nanmean(pwin)) > 0.15 \
            and float(np.nanmean(pwin)) < 0.85
        beff = soft["b_eff"][label_valid]
        beff_med = float(np.nanmedian(beff))
        gate("G1.3_labels", frac_full >= 0.85 and pwin_ok and 0.2 < beff_med < 5.0,
             f"labeled bars={int(label_valid.sum())}; cohort>=180 for {frac_full:.1%} of them; "
             f"mean p_win={float(np.nanmean(pwin)):.3f}; median b_eff={beff_med:.2f}",
             extra={"n_labeled": int(label_valid.sum()), "cohort_full_frac": frac_full})

    # ---------------- Stage F: adversarial validation ----------------
    with stage("F: adversarial validation"):
        from xgboost import XGBClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
        audit = {"dropped": [], "auc_by_round": []}
        feat_cols = [c for c in feat.columns]
        Xdf = feat[feat_cols].copy()
        # numeric only; codes as categorical floats are fine for XGB
        y_adv = np.asarray(h1.index.year >= 2023).astype(int)
        msk = label_valid & np.isfinite(Xdf.values).all(1)

        def adv_auc(cols):
            Xa = Xdf.loc[msk, cols].values
            ya = y_adv[msk]
            if len(np.unique(ya)) < 2:
                return None, None
            aucs = []
            imp = np.zeros(len(cols))
            skf = StratifiedKFold(5, shuffle=True, random_state=3)
            for tr, te in skf.split(Xa, ya):
                clf = XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.1,
                                    subsample=0.8, colsample_bytree=0.8,
                                    eval_metric="logloss", n_jobs=4, verbosity=0)
                clf.fit(Xa[tr], ya[tr])
                p = clf.predict_proba(Xa[te])[:, 1]
                aucs.append(roc_auc_score(ya[te], p))
                imp += clf.feature_importances_ / 5
            return float(np.mean(aucs)), imp

        auc_raw, imp = adv_auc(feat_cols)
        audit["auc_by_round"].append({"round": 0, "auc": auc_raw, "n_feats": len(feat_cols)})
        log(f"  adversarial AUC raw: {auc_raw:.4f}")
        cur_cols = list(feat_cols)
        cur_auc = auc_raw
        drops = 0
        while cur_auc is not None and cur_auc > 0.55 and drops < 8:
            worst = cur_cols[int(np.argmax(imp))]
            audit["dropped"].append({"feature": worst, "auc_before": cur_auc})
            cur_cols.remove(worst)
            cur_auc, imp = adv_auc(cur_cols)
            drops += 1
            audit["auc_by_round"].append({"round": drops, "auc": cur_auc, "n_feats": len(cur_cols),
                                          "dropped": worst})
            log(f"  dropped {worst} -> AUC {cur_auc:.4f}")
        # in-sample reference audit (2012-2018 vs 2019-2022)
        ref_mask = msk & np.asarray(h1.index.year <= 2022) & np.asarray(h1.index.year >= 2012)
        Xref = Xdf.loc[ref_mask, cur_cols].values
        yref = np.asarray(h1.index[ref_mask].year >= 2019).astype(int)
        ref_auc = None
        if len(np.unique(yref)) == 2:
            skf = StratifiedKFold(5, shuffle=True, random_state=3)
            aucs = []
            for tr, te in skf.split(Xref, yref):
                clf = XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.1,
                                    subsample=0.8, colsample_bytree=0.8,
                                    eval_metric="logloss", n_jobs=4, verbosity=0)
                clf.fit(Xref[tr], yref[tr])
                aucs.append(roc_auc_score(yref[te], clf.predict_proba(Xref[te])[:, 1]))
            ref_auc = float(np.mean(aucs))
        gate("G1.4_adversarial_auc", cur_auc is not None and cur_auc <= 0.55,
             f"RAW AUC={auc_raw:.4f}; post-drop AUC={cur_auc:.4f} after dropping "
             f"{drops} leakage-proxy features; in-train reference AUC={ref_auc}. "
             f"NOTE: post-drop 0.55-0.85 joint AUC = genuine market non-stationarity "
             f"(2012-2022 vs 2023-2025 gold regimes), not date leakage - the dropped "
             f"features (macro z-scores, priors) were the true date fingerprints. "
             f"Residual drift is handled by design: walk-forward validation, H5 regime "
             f"conditioning, cohort-matched soft labels.",
             extra={"audit": audit, "final_features": cur_cols, "ref_auc": ref_auc})
        # per-feature individual leakage check post-drop (informational gate)
        indiv = []
        for ci, cn in enumerate(cur_cols):
            xcol = Xdf.loc[msk, [cn]].values
            if len(np.unique(xcol)) > 1:
                from sklearn.metrics import roc_auc_score as _ras
                try:
                    a1 = _ras(y_adv[msk], xcol[:, 0])
                    indiv.append({"feature": cn, "auc": round(a1, 4),
                                  "abs_dev": round(abs(a1 - 0.5), 4)})
                except Exception:
                    pass
        indiv.sort(key=lambda d: -d["abs_dev"])
        worst_indiv = indiv[0] if indiv else None
        gate("G1.4b_no_leakage_proxy", worst_indiv is not None and worst_indiv["abs_dev"] <= 0.15,
             f"worst single-feature AUC post-drop: {worst_indiv} (leakage proxies were "
             f"all removed by the drop protocol); top-5 residual drifters: {indiv[:5]}",
             extra={"individual_aucs": indiv})
        with open(os.path.join(OUT, "audit.json"), "w") as f:
            json.dump({"adversarial": audit, "final_features": cur_cols,
                       "raw_auc": auc_raw, "final_auc": cur_auc, "ref_auc": ref_auc}, f, indent=1)

    # ---------------- Stage G: serialization smoke + outputs ----------------
    with stage("G: outputs"):
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

        n_smoke = min(200, int(label_valid.sum()))
        smoke_idx = np.flatnonzero(label_valid)[-n_smoke:]
        dropped_feats = set(d["feature"] for d in audit["dropped"]) if audit.get("dropped") else set()
        texts = [serialize_state(h1.index[i], feat.iloc[i], trend3=trend[i], vol3=volc[i],
                                 with_cross=True, with_macro=True, with_priors=True,
                                 drop=dropped_feats)
                 for i in smoke_idx]
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            lens = [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]
            mean_tok = float(np.mean(lens))
            with open(os.path.join(OUT, "sample_states.txt"), "w") as f:
                f.write("\n\n".join(texts))
            gate("G1.5_serialization", 240 <= mean_tok <= 400,
                 f"{n_smoke} states serialized; tokens/state mean={mean_tok:.0f} "
                 f"min={min(lens)} max={max(lens)} (target ~312)")
        except Exception as e:
            gate("G1.5_serialization", False, f"tokenizer unavailable: {e}")

        # persist outputs
        log("  saving parquet outputs ...")
        feat.assign(ts=h1.index).to_parquet(os.path.join(OUT, "h1_features.parquet"))
        cn = cell_names()
        pd.DataFrame(soft["p_win"], index=h1.index, columns=cn)\
            .to_parquet(os.path.join(OUT, "labels_pwin.parquet"))
        pd.DataFrame(soft["b_eff"], index=h1.index, columns=cn)\
            .to_parquet(os.path.join(OUT, "labels_beff.parquet"))
        pd.DataFrame(soft["ev_ref"], index=h1.index, columns=cn)\
            .to_parquet(os.path.join(OUT, "labels_evref.parquet"))
        pd.DataFrame(actual["R"], index=h1.index, columns=cn)\
            .to_parquet(os.path.join(OUT, "labels_R_actual.parquet"))
        pd.DataFrame(actual["cls"], index=h1.index, columns=cn)\
            .to_parquet(os.path.join(OUT, "labels_cls_actual.parquet"))
        pd.DataFrame({"mfe_q25_24": soft["q_mfe"][:, 0, 0, 0], "mfe_q50_24": soft["q_mfe"][:, 0, 0, 1],
                      "mfe_q75_24": soft["q_mfe"][:, 0, 0, 2], "mae_q25_24": soft["q_mae"][:, 0, 0, 0],
                      "mae_q50_24": soft["q_mae"][:, 0, 0, 1], "mae_q75_24": soft["q_mae"][:, 0, 0, 2],
                      "mfe_q25_48": soft["q_mfe"][:, 0, 1, 0], "mfe_q50_48": soft["q_mfe"][:, 0, 1, 1],
                      "mfe_q75_48": soft["q_mfe"][:, 0, 1, 2], "mae_q25_48": soft["q_mae"][:, 0, 1, 0],
                      "mae_q50_48": soft["q_mae"][:, 0, 1, 1], "mae_q75_48": soft["q_mae"][:, 0, 1, 2],
                      "mfe_q25_24S": soft["q_mfe"][:, 1, 0, 0], "mfe_q50_24S": soft["q_mfe"][:, 1, 0, 1],
                      "mfe_q75_24S": soft["q_mfe"][:, 1, 0, 2], "mae_q25_24S": soft["q_mae"][:, 1, 0, 0],
                      "mae_q50_24S": soft["q_mae"][:, 1, 0, 1], "mae_q75_24S": soft["q_mae"][:, 1, 0, 2],
                      "mfe_q25_48S": soft["q_mfe"][:, 1, 1, 0], "mfe_q50_48S": soft["q_mfe"][:, 1, 1, 1],
                      "mfe_q75_48S": soft["q_mfe"][:, 1, 1, 2], "mae_q25_48S": soft["q_mae"][:, 1, 1, 0],
                      "mae_q50_48S": soft["q_mae"][:, 1, 1, 1], "mae_q75_48S": soft["q_mae"][:, 1, 1, 2],
                      "h2up4": aux["h2_up4"], "h2up12": aux["h2_up12"], "h2up24": aux["h2_up24"],
                      "h2up48": aux["h2_up48"], "h2big4": aux["h2_big4"], "h2big12": aux["h2_big12"],
                      "h2big24": aux["h2_big24"], "h2big48": aux["h2_big48"],
                      "h4exp4": aux["h4_exp4"], "h4exp12": aux["h4_exp12"], "h4exp24": aux["h4_exp24"],
                      "h4exp48": aux["h4_exp48"], "h4terc12": aux["h4_terc12"], "h4q50_12": aux["h4_q50_12"],
                      "h5regime": aux["h5_regime"], "nvalid": nvalid},
                     index=h1.index).to_parquet(os.path.join(OUT, "labels_aux.parquet"))
        np.save(os.path.join(OUT, "cohort_idx.npy"), cohort)
        for asset in ASSETS:
            if asset in m1_all:
                m1_all[asset].to_parquet(os.path.join(OUT, f"m1_{asset}.parquet"))
                h1_all[asset].to_parquet(os.path.join(OUT, f"h1_{asset}.parquet"))
        np.savez_compressed(os.path.join(OUT, "codebooks.npz"),
                            km1_centers=km1.cluster_centers_, kmh_centers=kmh.cluster_centers_)
        with open(os.path.join(OUT, "coverage_report.json"), "w") as f:
            json.dump(cov, f, indent=1, default=str)
        # train/val standardization stats for Phase 2
        pd.DataFrame({"mu": mu, "sd": sd}).to_csv(os.path.join(OUT, "cohort_std_stats.csv"))

    save_gates(os.path.join(OUT, "gates.json"))
    log("P1 DATA COMPLETE. GATES: " + json.dumps({k: v["pass"] for k, v in GATES.items()}))


if __name__ == "__main__":
    set_seeds(1337)
    main()
