# ============================================================================
# AURUM Phase 3 - PROBABILITY TABLES KERNEL (2xT4, ~1.5h) - Spec v1.2
#   Mounts : tskdev/aurum-phase1-data (v5) + tskdev/aurum-models-v1
#   Step 1 : re-serialize all labeled states (byte-identical pipeline to
#            p2-train v6: same tokenizer flags, same drop set, same order)
#   Step 2 : forward the trained looped core at depths r in {1,2,3,4}
#            (2 workers, one per GPU, round-robin state shards; fp16 autocast
#            no-grad, bs=32 - the exact recipe of the p2-train cache stage).
#            r=4 extrapolation: the 4th loop iteration reuses the LAST trained
#            sandwich norm (weight-tied cell reuse; training only saw r<=3 via
#            random loop supervision).
#   Step 3 : heads = main (trained <=2022) + 10 walk-forward fold heads
#            (per-year, causal); per-head temperature scaling (h1/h5).
#   Step 4 : OOD loop-divergence vs the r=3 reference on calibrated H1 probs.
#   Step 5 : era-neutralized predictions (per-year orthogonalization against
#            the p1-data era classifier score f(x)) -> neutralized EV argmax.
#   Step 6 : aurum_oos_prob_tables.parquet (one row per VALID state x depth x
#            variant{main,fold}: 85 calibrated head outputs + OOD + EV fields;
#            fold variant only has 2016-2025 states) + p3_cell_meta.json +
#            p3_tables_summary.json + gates.json
#   Validation: recomputed fold-calibration Brier must match the training-time
#            fold_results.json values (same rows, same metric) within 0.01.
# ============================================================================
P3_DEPTHS = [1, 2, 3, 4]
P3_TIMEOUT_S = 3.5 * 3600


def _find_models_ds():
    """Locate the aurum-models-v1 dataset mount robustly."""
    import glob as _glob
    cands = []
    if os.environ.get("AURUM_MODELS"):
        cands.append(os.environ["AURUM_MODELS"])
    cands += ["/kaggle/input/datasets/tskdev/aurum-models-v1",
              "/kaggle/input/aurum-models-v1"]
    cands += sorted(_glob.glob("/kaggle/input/datasets/*/aurum-models-v1*"))
    cands += sorted(_glob.glob("/kaggle/input/*/aurum-models-v1*"))
    cands += sorted(_glob.glob("/kaggle/input/*aurum-models*"))
    for c in cands:
        if os.path.isfile(os.path.join(c, "core_fp16.pt")):
            return c
    try:
        import subprocess as _sp
        tree = _sp.run(["find", "/kaggle/input", "-maxdepth", "3"],
                       capture_output=True, text=True, timeout=20).stdout
        print("[_find_models_ds] DATASET NOT FOUND. /kaggle/input tree:\n"
              + tree[:3000], flush=True)
    except Exception:
        pass
    return None


def build_core(base, core_sd, depth, log=print):
    """LoopedQwen at inference depth `depth` with trained fp16 weights.
    Trained sandwich norms cover iterations 0..n_trained-1 (n_loop_train=3);
    iterations beyond that (r=4 extrapolation) reuse the LAST trained norm."""
    model = LoopedQwen(base, n_loop=depth, sandwich=True, grad_ckpt=False)
    for i in range(CORE_START, CORE_END):
        sd_i = {k: v.float() for k, v in core_sd["layers"][str(i)].items()}
        model.base.model.layers[i].load_state_dict(sd_i)
    pre_sd = {k: v.float() for k, v in core_sd["pre"].items()}
    post_sd = {k: v.float() for k, v in core_sd["post"].items()}
    n_trained = 1 + max(int(k.split(".")[0]) for k in pre_sd)
    model.pre_norms.load_state_dict(pre_sd, strict=False)
    model.post_norms.load_state_dict(post_sd, strict=False)
    if depth > n_trained:
        for r in range(n_trained, depth):
            model.pre_norms[r].load_state_dict(
                model.pre_norms[n_trained - 1].state_dict())
            model.post_norms[r].load_state_dict(
                model.post_norms[n_trained - 1].state_dict())
        log(f"  r={depth} extrapolation: iterations {n_trained}..{depth-1} "
            f"reuse trained norm index {n_trained-1}")
    model.eval()
    return model


def _p3_worker(rank, world, WORK, MODELS, depths):
    """One GPU's shard of the depth sweep (round-robin, cache-stage recipe)."""
    import torch
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and world > 1:
        dev = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    z = np.load(os.path.join(WORK, "p2_train_data.npz"))
    lab_flat, lens = z["lab_flat"], z["lab_lens"]
    N = len(lens)
    ends = np.cumsum(lens)
    starts = ends - lens
    my_rows = np.arange(rank, N, world)
    core_sd = torch.load(os.path.join(MODELS, "core_fp16.pt"), map_location="cpu")
    ro_w = core_sd["readout"]["readout.weight"].float().to(dev)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    base = load_base(device=dev, dtype=torch.float32,
                     log=lambda m: print(f"  [w{rank}] {m}", flush=True))
    for depth in depths:
        out_f = os.path.join(WORK, f"roh_r{depth}_part{rank}.npy")
        if os.path.exists(out_f):
            continue
        model = build_core(base, core_sd, depth)
        model.to(dev)
        H5 = np.zeros((len(my_rows), 5, model.base.config.hidden_size),
                      dtype=np.float16)
        t0 = time.time()
        bs_c = 32
        with torch.no_grad():
            for s0 in range(0, len(my_rows), bs_c):
                rows = my_rows[s0: s0 + bs_c]
                seqs = [lab_flat[starts[i]:ends[i]] for i in rows]
                L = max(len(s_) for s_ in seqs)
                ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
                pm = torch.zeros((len(seqs), L), dtype=torch.bool)
                for i, s_ in enumerate(seqs):
                    ids[i, :len(s_)] = s_
                    pm[i, :len(s_)] = True
                with autocast_ctx(dev):
                    o = model(ids.to(dev), add_readout=True, lm_output=False,
                              pad_mask=pm.to(dev), ro_embed=ro_w)
                H5[s0:s0 + len(rows)] = o["ro_h"].half().cpu().numpy()
                if rank == 0 and s0 % 6400 == 0:
                    rps = (s0 + len(rows)) * world / max(time.time() - t0, 1e-9)
                    print(f"  [w0] r{depth}: ~{s0 * world}/{N} "
                          f"({rps:.1f} rows/s agg)", flush=True)
        np.save(out_f, H5)
        print(f"  [w{rank}] depth r={depth} DONE {len(my_rows)} rows "
              f"in {time.time()-t0:.0f}s", flush=True)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()


def _p3_forward_all(depths, WORK, MODELS):
    """Forward all labeled states at each depth. Two subprocess workers (one
    per GPU) when >=2 CUDA devices; in-process single path otherwise."""
    import torch
    z = np.load(os.path.join(WORK, "p2_train_data.npz"))
    N = len(z["lab_lens"])
    world = max(1, torch.cuda.device_count())
    todo = [d for d in depths
            if not os.path.exists(os.path.join(WORK, f"roh_r{d}_part0.npy"))]
    if todo and world > 1:
        todo = [d for d in depths
                if not all(os.path.exists(os.path.join(WORK, f"roh_r{d}_part{rk}.npy"))
                           for rk in range(world))]
    if todo:
        if world > 1:
            import subprocess as sp
            env = dict(os.environ)
            env.update(AURUM_WORK=WORK, AURUM_MODELS=MODELS)
            procs, lfs = [], []
            for rk in range(world):
                lf = open(os.path.join(WORK, f"p3w{rk}.log"), "w")
                lfs.append(lf)
                procs.append(sp.Popen(
                    [sys.executable, os.path.abspath(__file__), "--mode", "worker",
                     "--rank", str(rk), "--world", str(world)],
                    env=env, stdout=lf, stderr=sp.STDOUT))
            t0 = time.time()
            while any(p.poll() is None for p in procs):
                if time.time() - t0 > P3_TIMEOUT_S:
                    for p in procs:
                        p.kill()
                    raise RuntimeError("p3 forward workers timed out")
                time.sleep(20)
            for i, (p, lf) in enumerate(zip(procs, lfs)):
                lf.close()
                if p.returncode != 0:
                    tail = open(os.path.join(WORK, f"p3w{i}.log")).read()[-2000:]
                    raise RuntimeError(f"worker {i} failed (rc={p.returncode}):\n{tail}")
        else:
            _p3_worker(0, 1, WORK, MODELS, todo)
    roh = {}
    for d in depths:
        if world > 1:
            parts = [np.load(os.path.join(WORK, f"roh_r{d}_part{rk}.npy"))
                     for rk in range(world)]
            full = np.zeros((N, *parts[0].shape[1:]), dtype=parts[0].dtype)
            for rk in range(world):
                full[rk::world] = parts[rk]
            roh[d] = full
        else:
            roh[d] = np.load(os.path.join(WORK, f"roh_r{d}_part0.npy"))
    return roh


class _Shim:
    """Duck-types LoopedQwen for _mk_head_module (needs .base.config.hidden_size)."""
    def __init__(self, d):
        import types
        self.base = types.SimpleNamespace(
            config=types.SimpleNamespace(hidden_size=d))


def _head_stack(d):
    return _mk_head_module(_Shim(d), 0.0)


def _head_activate(logits, T_h1=1.0, T_h5=1.0):
    """Training-time activations + temperature scaling (h1/h5 only)."""
    import torch
    import torch.nn.functional as Fn
    return {
        "h1": torch.sigmoid(logits["h1"] / T_h1),
        "h2": torch.sigmoid(logits["h2"]),
        "h3": Fn.softplus(logits["h3"]),
        "h4": torch.cat([torch.sigmoid(logits["h4"][:, :4]),
                         torch.softmax(logits["h4"][:, 4:7], dim=-1),
                         Fn.softplus(logits["h4"][:, 7:8])], dim=1),
        "h5": torch.softmax(logits["h5"] / T_h5, dim=-1),
    }


def _neutralize_per_year(P, f, years):
    """Per-year linear orthogonalization of each column of P against the era
    score f (exp3d protocol / neutralization_spec.json)."""
    Pn = P.copy()
    for yv in np.unique(years):
        m = years == yv
        if m.sum() < 30 or not np.isfinite(f[m]).all():
            continue
        ff = f[m]
        df_ = ff - ff.mean()
        var = float((df_ * df_).sum()) + 1e-12
        beta = (df_ @ (P[m] - P[m].mean(0))) / var          # (48,)
        Pn[m] = P[m] - np.outer(df_, beta)
    return np.clip(Pn, 0.0, 1.0)


def _ev_fields(P, b_geo):
    """EV per cell (R units): p*b - (1-p); argmax cell + its p/EV."""
    ev = P * b_geo[None, :] - (1.0 - P)
    cell = ev.argmax(1)
    n = np.arange(len(cell))
    return cell, P[n, cell], ev[n, cell]


def main(WORK):
    import torch
    import pandas as pd
    t0 = time.time()
    log("ENV: " + json.dumps(env_report()))
    MODELS = _find_models_ds()
    DS = _ds()
    ok_mount = MODELS is not None and os.path.isfile(
        os.path.join(DS, "h1_features.parquet"))
    gate("G4.0_mounts", ok_mount,
         f"phase1-data at {DS} "
         f"({'OK' if os.path.isfile(os.path.join(DS, 'h1_features.parquet')) else 'MISSING'}); "
         f"aurum-models-v1 at {MODELS} ({'OK' if MODELS else 'MISSING'})")
    if not ok_mount:
        save_gates(os.path.join(WORK, "gates.json"))
        return

    # ---------------- data (identical pipeline to p2-train v6) -------------
    cache, meta = prepare_data(WORK)
    z = np.load(cache)
    lab_idx, lab_years = z["lab_idx"], z["lab_years"]
    N = len(lab_idx)
    gate("G4.1_data", N == 87948,
         f"labeled states={N} (expect 87948); mean tokens={meta['mean_tokens']}; "
         f"pre-test={meta['n_pretest']} OOS={meta['n_oos']}")
    if N != 87948:
        save_gates(os.path.join(WORK, "gates.json"))
        return
    years = np.asarray(lab_years)

    # ---------------- depth sweep (one worker per GPU) ----------------------
    with stage("depth-sweep forward r in {1,2,3,4}"):
        _ = load_base(device=None, log=log)   # warm the HF cache once
        del _
        roh = _p3_forward_all(P3_DEPTHS, WORK, MODELS)
    shapes_ok = all(roh[d].shape == (N, 5, 1024) for d in P3_DEPTHS)
    gate("G4.2_depth_forwards", shapes_ok,
         "ro_h shapes " + json.dumps({d: list(roh[d].shape) for d in P3_DEPTHS}))
    if not shapes_ok:
        save_gates(os.path.join(WORK, "gates.json"))
        return

    # ---------------- heads + temperatures ---------------------------------
    core_sd = torch.load(os.path.join(MODELS, "core_fp16.pt"), map_location="cpu")
    d_model = core_sd["readout"]["readout.weight"].shape[1]
    heads = {"main": _head_stack(d_model)}
    heads["main"].load_state_dict(
        torch.load(os.path.join(MODELS, "heads_main.pt"), map_location="cpu"))
    heads["main"].eval()
    temps_main = json.load(open(os.path.join(MODELS, "temperatures_main.json")))
    temps_fold = json.load(open(os.path.join(MODELS, "temperatures.json")))
    for yv in FOLD_YEARS:
        st = _head_stack(d_model)
        st.load_state_dict(torch.load(
            os.path.join(MODELS, f"heads_fold{yv}.pt"), map_location="cpu"))
        st.eval()
        heads[f"fold{yv}"] = st
    fold_results_ref = json.load(open(os.path.join(MODELS, "fold_results.json")))
    stageB_ref = json.load(open(os.path.join(MODELS, "stageB_result.json")))

    # ---------------- era score for neutralization --------------------------
    f_era = np.full(N, np.nan)
    try:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            import subprocess as sp
            sp.run([sys.executable, "-m", "pip", "install", "-q", "xgboost"],
                   check=True, timeout=600)
            from xgboost import XGBClassifier
        clf = XGBClassifier()
        clf.load_model(os.path.join(DS, "era_clf.json"))
        spec = json.load(open(os.path.join(DS, "neutralization_spec.json")))
        feat_df = pd.read_parquet(os.path.join(DS, "h1_features.parquet"))
        cols = [c for c in spec["features"] if c in feat_df.columns]
        X = feat_df[cols].iloc[lab_idx].values
        ok_rows = np.isfinite(X).all(1)
        f_era[ok_rows] = clf.predict_proba(X[ok_rows])[:, 1]
        log(f"  era score coverage: {ok_rows.mean()*100:.2f}% "
            f"({ok_rows.sum()}/{N}); mean f={np.nanmean(f_era):.4f}")
    except Exception as e:
        log(f"  era-score FAILED ({e}); neutralized columns = raw")
    gate("G4.6_era_score", bool(np.isfinite(f_era).mean() > 0.98),
         f"era_clf coverage={np.isfinite(f_era).mean()*100:.2f}%")

    # ---------------- build the probability tables ---------------------------
    dirs_ax, mss, tps, hrs = cell_axes()
    b_geo = np.array([t / m for _, m, t, _ in zip(dirs_ax, mss, tps, hrs)])
    ts_ns = np.asarray(pd.read_parquet(
        os.path.join(DS, "h1_xauusd.parquet")).index.asi8)[lab_idx]

    valid = {"main": np.arange(N),
             "fold": np.flatnonzero((years >= 2016) & (years <= 2025))}
    blocks, summary = [], {}
    t_head = time.time()
    for variant in ("main", "fold"):
        rows_idx = valid[variant]
        n_v = len(rows_idx)
        yr_v = years[rows_idx]
        f_v = np.nan_to_num(f_era[rows_idx], nan=0.5)
        for r in P3_DEPTHS:
            H = torch.from_numpy(roh[r][rows_idx].astype(np.float32))
            P_v = np.zeros((n_v, 48), dtype=np.float32)
            O_v = {k: np.zeros((n_v, {"h2": 8, "h3": 12, "h4": 8, "h5": 9}[k]),
                                dtype=np.float32)
                   for k in ("h2", "h3", "h4", "h5")}
            if variant == "main":
                with torch.no_grad():
                    A = _head_activate(heads["main"](H),
                                       temps_main.get("h1", 1.0),
                                       temps_main.get("h5", 1.0))
                P_v = A["h1"].numpy().astype(np.float32)
                for k in ("h2", "h3", "h4", "h5"):
                    O_v[k] = A[k].numpy().astype(np.float32)
            else:
                for yv in FOLD_YEARS:
                    ry = np.flatnonzero(yr_v == yv)
                    if len(ry) == 0:
                        continue
                    tv = temps_fold.get(str(yv), {})
                    with torch.no_grad():
                        A = _head_activate(heads[f"fold{yv}"](H[ry]),
                                           tv.get("h1", 1.0), tv.get("h5", 1.0))
                    P_v[ry] = A["h1"].numpy()
                    for k in ("h2", "h3", "h4", "h5"):
                        O_v[k][ry] = A[k].numpy()
            Pn_v = _neutralize_per_year(P_v, f_v, yr_v)
            cell_r, p_r, ev_r = _ev_fields(P_v, b_geo)
            cell_n, p_n, ev_n = _ev_fields(Pn_v, b_geo)
            blk = {
                "ts": ts_ns[rows_idx], "year": yr_v.astype(np.int16),
                "r": np.full(n_v, r, dtype=np.int8),
                "f_era": f_era[rows_idx].astype(np.float32),
                "ev_cell_raw": cell_r.astype(np.int16),
                "p_evcell_raw": p_r.astype(np.float32),
                "ev_raw": ev_r.astype(np.float32),
                "ev_cell_neut": cell_n.astype(np.int16),
                "p_evcell_neut": p_n.astype(np.float32),
                "ev_neut": ev_n.astype(np.float32),
                "_P": P_v.astype(np.float16),
            }
            for j in range(48):
                blk[f"h1_c{j:02d}"] = P_v[:, j].astype(np.float16)
            for k, arr in O_v.items():
                for j in range(arr.shape[1]):
                    blk[f"{k}_{j}"] = arr[:, j].astype(np.float16)
            blocks.append({"variant": variant, "r": r, "blk": blk,
                           "rows_idx": rows_idx})
            log(f"  heads {variant} r={r}: {n_v} states "
                f"({time.time()-t_head:.0f}s)")
            t_head = time.time()

    # OOD divergence vs r=3 reference per variant (48-cell calibrated H1)
    for b in blocks:
        ref = next(x for x in blocks
                   if x["variant"] == b["variant"] and x["r"] == 3)
        if b["r"] == 3:
            b["blk"]["ood_mean"] = np.zeros(len(b["blk"]["ts"]), np.float32)
            b["blk"]["ood_max"] = np.zeros(len(b["blk"]["ts"]), np.float32)
        else:
            diff = np.abs(b["blk"]["_P"].astype(np.float32)
                          - ref["blk"]["_P"].astype(np.float32))
            b["blk"]["ood_mean"] = diff.mean(1).astype(np.float32)
            b["blk"]["ood_max"] = diff.max(1).astype(np.float32)

    # ---------------- validation vs training-time metrics --------------------
    val_detail = {}
    n_fold_ok = 0
    H3 = torch.from_numpy(roh[3].astype(np.float32))
    for yv in FOLD_YEARS:
        fr = fold_results_ref.get(str(yv), {})
        if not fr or fr.get("skipped"):
            continue
        fold_train_pos = np.flatnonzero(years <= yv - 1)
        n_cal = max(200, int(len(fold_train_pos) * CAL_FRAC))
        cal_pos = fold_train_pos[-n_cal:]
        with torch.no_grad():
            Pcal = torch.sigmoid(heads[f"fold{yv}"](H3[cal_pos])["h1"]).numpy()
        T_sm = z["sm_h1"][cal_pos]
        brier = float(np.nanmean((Pcal - np.nan_to_num(T_sm, nan=0.5)) ** 2))
        ref_b = fr.get("brier_sm")
        ok = ref_b is not None and abs(brier - ref_b) < 0.01
        n_fold_ok += int(ok)
        val_detail[str(yv)] = {"brier_recomputed": round(brier, 5),
                               "brier_ref": ref_b, "match": bool(ok)}
    v22 = np.flatnonzero(years == 2022)
    with torch.no_grad():
        P22 = torch.sigmoid(heads["main"](H3[v22])["h1"]).numpy()
    b22 = float(np.nanmean((P22 - np.nan_to_num(z["sm_h1"][v22], nan=0.5)) ** 2))
    b22_ref = stageB_ref.get("final_val", {}).get("brier_sm")
    main_ok = b22_ref is not None and abs(b22 - b22_ref) < 0.012
    gate("G4.3_heads_validated", n_fold_ok >= 8 and main_ok,
         f"fold-cal Brier matches training values for {n_fold_ok}/10 folds "
         f"(tol 0.01); main-head 2022 Brier {b22:.5f} vs training "
         f"{b22_ref} (tol 0.012); detail={json.dumps(val_detail)}")

    # ---------------- summary metrics per (variant, r) -----------------------
    from sklearn.metrics import roc_auc_score
    for b in blocks:
        blk, rows_idx = b["blk"], b["rows_idx"]
        n_ = np.arange(len(blk["ts"]))
        P32 = blk["_P"].astype(np.float32)
        p_sel = P32[n_, blk["ev_cell_raw"].astype(int)]
        yv_real = z["win_real"][rows_idx, blk["ev_cell_raw"].astype(int)]
        sm_t = z["sm_h1"][rows_idx, blk["ev_cell_raw"].astype(int)]
        fin = np.isfinite(yv_real)
        summary[f"{b['variant']}_r{b['r']}"] = {
            k: round(vv, 5) for k, vv in {
                "n_states": int(len(blk["ts"])),
                "brier_sm_argmax": float(np.nanmean(
                    (p_sel - np.nan_to_num(sm_t, nan=0.5)) ** 2)),
                "brier_re_argmax": float(np.mean(
                    (p_sel[fin] - yv_real[fin]) ** 2)),
                "auc_re_argmax": float(roc_auc_score(yv_real[fin], p_sel[fin])),
                "ev_mean": float(blk["ev_raw"].mean()),
                "ood_mean": float(blk["ood_mean"].mean()),
                "ood_p95": float(np.percentile(blk["ood_mean"], 95)),
                "neut_cell_agree": float((
                    blk["ev_cell_neut"] == blk["ev_cell_raw"]).mean()),
            }.items()}
    log("SUMMARY " + json.dumps(summary))

    # ---------------- write parquet ------------------------------------------
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema_order = ["ts", "year", "r", "variant", "f_era",
                    "ev_cell_raw", "p_evcell_raw", "ev_raw",
                    "ev_cell_neut", "p_evcell_neut", "ev_neut",
                    "ood_mean", "ood_max"] + \
                   [f"h1_c{j:02d}" for j in range(48)] + \
                   [f"h2_{j}" for j in range(8)] + \
                   [f"h3_{j}" for j in range(12)] + \
                   [f"h4_{j}" for j in range(8)] + \
                   [f"h5_{j}" for j in range(9)]
    for b in blocks:
        b["blk"]["variant_id"] = np.full(len(b["blk"]["ts"]),
                                         0 if b["variant"] == "fold" else 1,
                                         dtype=np.int8)
    cols = {}
    for col in schema_order:
        if col == "variant":
            vals = np.concatenate([b["blk"]["variant_id"] for b in blocks])
            cols[col] = pa.DictionaryArray.from_arrays(
                pa.array(vals, type=pa.int8()),
                pa.array(["fold", "main"], type=pa.string()))
        else:
            cols[col] = np.concatenate([b["blk"][col] for b in blocks])
    fields = []
    for col in schema_order:
        if col == "ts":
            t_ = pa.int64()
        elif col == "year":
            t_ = pa.int16()
        elif col == "r":
            t_ = pa.int8()
        elif col == "variant":
            t_ = pa.dictionary(pa.int8(), pa.string())
        elif col in ("ev_cell_raw", "ev_cell_neut"):
            t_ = pa.int16()
        elif col in ("f_era", "p_evcell_raw", "ev_raw", "p_evcell_neut",
                     "ev_neut", "ood_mean", "ood_max"):
            t_ = pa.float32()
        else:
            t_ = pa.float16()
        fields.append(pa.field(col, t_))
    tbl = pa.Table.from_arrays(
        [cols[c.name] if c.name == "variant"
         else pa.array(cols[c.name]).cast(c.type) for c in fields],
        schema=pa.schema(fields))
    out_pq = os.path.join(WORK, "aurum_oos_prob_tables.parquet")
    pq.write_table(tbl, out_pq, compression="zstd")
    n_rows = tbl.num_rows
    exp_rows = N * len(P3_DEPTHS) + len(valid["fold"]) * len(P3_DEPTHS)
    h1_bad = float(sum(int(np.isnan(b["blk"][f"h1_c{j}"]).any())
                       for b in blocks for j in range(48)))
    gate("G4.4_tables", n_rows == exp_rows and h1_bad == 0,
         f"parquet rows={n_rows:,} (expect {exp_rows:,}); "
         f"{len(schema_order)} columns; "
         f"{os.path.getsize(out_pq)/1e6:.1f} MB; h1 col-blocks with NaN={h1_bad}")

    ood_stats = {f"{b['variant']}_r{b['r']}":
                 {"mean": round(float(b["blk"]["ood_mean"].mean()), 6),
                  "p95": round(float(np.percentile(b["blk"]["ood_mean"], 95)), 6)}
                 for b in blocks}
    gate("G4.5_ood",
         all(ood_stats[f"{v}_r3"]["mean"] == 0.0 for v in ("main", "fold"))
         and all(ood_stats[f"{v}_r{r}"]["mean"] > 0
                 for v in ("main", "fold") for r in (1, 2, 4)),
         "loop-divergence |p_r - p_3| mean/p95: " + json.dumps(ood_stats))

    # ---------------- cell meta + summary + export ---------------------------
    json.dump({
        "cell_names": cell_names(),
        "cell_axes": {"dirs": list(dirs_ax), "stops": list(mss),
                      "tps": list(tps), "hors": list(hrs)},
        "b_geo": b_geo.tolist(),
        "r_win_R": b_geo.tolist(), "r_loss_R": [-1.0] * 48,
        "depths": P3_DEPTHS, "variants": ["fold", "main"],
        "row_counts": {"main": int(N), "fold": int(len(valid["fold"]))},
        "temperatures_main": temps_main,
        "temperatures_fold": temps_fold,
        "h_layout": {
            "h1": "48 cell P(win) calibrated; cells ordered dir(L,S) x "
                  "stop(1,1.5,2,2.5) x tp(1.5,2.5,4) x hor(24,48)",
            "h2": "8 sigmoid (up4/12/24/48, big4/12/24/48)",
            "h3": "12 softplus (mfe_L q25/50/75, mae_L ..., mfe_S ..., mae_S ...)",
            "h4": "8 (expansion probs 0-3, vol tercile 4-6, vol q50 ratio 7)",
            "h5": "9 softmax (trend3 x volc3)"},
        "provenance": {"models": "tskdev/aurum-models-v1",
                       "phase1": "tskdev/aurum-phase1-data (v5)",
                       "kernel": "tskdev/aurum-p3-tables"},
    }, open(os.path.join(WORK, "p3_cell_meta.json"), "w"), indent=1)
    json.dump({"summary": summary, "ood": ood_stats,
               "validation": {"folds": val_detail,
                              "main_2022": {"brier": round(b22, 5),
                                            "ref": b22_ref}},
               "wall_s": round(time.time() - t0, 1)},
              open(os.path.join(WORK, "p3_tables_summary.json"), "w"), indent=1)
    save_gates(os.path.join(WORK, "gates.json"))
    log("P3 TABLES COMPLETE. GATES: "
        + json.dumps({k: v["pass"] for k, v in GATES.items()}))
    log(f"total wall {time.time()-t0:.0f}s")


def _worker_main(rank, world):
    WORK = os.environ["AURUM_WORK"]
    MODELS = os.environ.get("AURUM_MODELS") or _find_models_ds()
    _p3_worker(rank, world, WORK, MODELS, P3_DEPTHS)
    print(f"[w{rank}] all depths done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="main")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    args = ap.parse_known_args()[0]
    WORK = os.environ.get("AURUM_WORK",
                          "/kaggle/working" if os.path.exists("/kaggle/working")
                          else os.getcwd())
    SELF = os.path.abspath(__file__)
    stable = os.path.join(WORK, "aurum_p3_tables.py")
    try:
        if SELF != stable and args.mode == "main":
            import shutil as _sh
            _sh.copy(SELF, stable)
            SELF = stable
    except Exception:
        pass
    set_seeds(1337)
    if args.mode == "worker":
        _worker_main(args.rank, args.world)
    else:
        main(WORK)
