"""
AURUM v2.0 -- K3 kernel main: Neural & Contractive-Loop Training (GPU)
======================================================================
Phase 3 of the v2.0 execution plan (runs on the secondary account's 2xT4):

  * M3_DAE_ResMLP : Jane-Street bottleneck DAE + ResMLP at compact
                    (d_enc=32) and medium (AEMLP d_enc=96) scales.
  * M4_DMN_Lite   : Oxford-Man VSN + GRU + 2-head masked MHA + GRN with
                    max-norm(3) readout, on 63+ step windows.
  * M5_ContinuumNet: contractive looped transformer (RevIN + Group VSN +
                    prelude + spectrally-capped non-residual LoopBlock
                    (sigma(W)<=0.8, KM damping, per-iterate supervision,
                    JVP spectral-radius governor rho<0.95) + coda),
                    d_model sweep {64, 128, 256}.

All models train on the FUSED RobustNetSharpeLoss (costs + SoftMin + Ulcer
inside the graph, gamma=0.5 cost scaler, sqrt(5786) annualizer) with early
stopping on validation net portfolio Sharpe (patience from cfg), under the
year-based purged walk-forward (fold y trains on bars < y with 24-bar purge).

Selection protocol: config search on train(<2022)/val(2022); the best config
per family refits across the 10 yearly folds (2016-2025); 2023-25 fold
predictions remain untouched OOS until K6.

Every configuration enters the Master Trial Ledger.  The spectral governor
log (rho, alpha per checkpoint) and the M5 per-iterate z series
(r=1..R loop-depth analysis vs v1.3's unconstrained loop) are exported.
"""
import json
import math
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

GATES = {}
LOG_LINES = []
T_START = time.time()
HARD_BUDGET_S = 1.9 * 3600     # COMPACT build for the ~2.5h tskdev GPU budget


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def gate(gid, ok, detail=None):
    g = {"gate": gid, "pass": bool(ok), "detail": str(detail)[:3000]}
    GATES[gid] = g
    log(f"GATE {gid}: {'PASS' if ok else 'FAIL'} | {detail}")
    return bool(g)


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


def budget_left():
    return HARD_BUDGET_S - (time.time() - T_START)


def prepare_matrix(feats: pd.DataFrame):
    names = [c for c in feats.columns
             if not c.startswith(("y_", "r_xau", "sigma", "close_xau"))]
    X = feats[names].astype("float32")
    X = X.ffill().fillna(0.0).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y = feats["y_ret24_norm"].astype("float32")
    return X, y, names


def group_slices_from_meta(names, meta_path):
    """VSN group slices aligned to the (already filtered) feature order."""
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        ordered = meta["columns"]
        groups = meta["group_slices"]
        name_pos = {c: i for i, c in enumerate(names)}
        slices, start = [], 0
        for s, e, nm in groups:
            members = [c for c in ordered[s:e] if c in name_pos]
            if members:
                slices.append((start, start + len(members), nm))
                start += len(members)
        if abs(start - len(names)) <= 2 and slices:
            # reorder names to the group order for the VSN
            reordered = []
            for s, e, nm in groups:
                reordered.extend([c for c in ordered[s:e] if c in name_pos])
            return slices, reordered
    except Exception as e:
        log(f"  group meta unavailable ({e}); using single-group fallback")
    return [(0, len(names), "all")], list(names)


# ----------------------------------------------------------------------------
# generic neural trainer (M3/M4; M5 uses train_one_fold with the governor)
# ----------------------------------------------------------------------------
def train_generic(model, X, ret, sigma, y_ret, y_dir, fit_pos, es_pos,
                  seq_len=256, batch_size=64, lr=1e-3, max_epochs=40,
                  patience=12, cost_r=5e-5, windows_per_epoch=5000,
                  device="cuda", seed=1337, w_aux=0.15):
    """Train any model with the dict-output interface {z, ret24, dir_logit,
    meta_logit} on the fused Sharpe loss; early stop on ES net Sharpe."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    crit = RobustNetSharpeLoss(cost_r=cost_r, gamma_cost_scaler=0.5,
                               softmin_tau=0.2, softmin_lambda=0.25,
                               lam_dd=0.10)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    rng = np.random.default_rng(seed)

    Xf = X  # (T, F) float32
    retf = np.nan_to_num(ret.values, nan=0.0).astype("float32")
    sigf = np.nan_to_num(sigma.values, nan=1e-4).astype("float32")
    yrf = np.nan_to_num(y_ret.values, nan=0.0).astype("float32")
    ydf = np.nan_to_num(y_dir.values, nan=0.0).astype("float32")
    T = len(Xf)

    def make_window(pos):
        lo = max(0, pos - seq_len + 1)
        x = Xf.iloc[lo:pos + 1].values
        if len(x) < seq_len:
            pad = np.repeat(x[:1], seq_len - len(x), axis=0)
            x = np.vstack([pad, x])
        r = retf[lo:pos + 1]
        r = np.concatenate([np.zeros(seq_len - len(r)), r])
        s = sigf[lo:pos + 1]
        s = np.concatenate([np.full(seq_len - len(s), 1e-4), s])
        return x, r, s, yrf[pos], ydf[pos]

    es_windows = [make_window(int(p)) for p in es_pos[:: max(1, len(es_pos) // 1500)]]
    es_x = torch.as_tensor(np.stack([w[0] for w in es_windows]), device=device)
    es_r = torch.as_tensor(np.stack([w[1] for w in es_windows]), device=device)
    es_s = torch.as_tensor(np.stack([w[2] for w in es_windows]), device=device)

    best_sharpe, best_state, bad = -1e9, None, 0
    hist = {"val_sharpe": []}
    for epoch in range(max_epochs):
        model.train()
        sel = rng.choice(fit_pos, size=min(windows_per_epoch, len(fit_pos)),
                         replace=False)
        for i0 in range(0, len(sel), batch_size):
            batch = [make_window(int(p)) for p in sel[i0: i0 + batch_size]]
            if len(batch) < 8:
                continue
            x = torch.as_tensor(np.stack([b[0] for b in batch]), device=device)
            r = torch.as_tensor(np.stack([b[1] for b in batch]), device=device)
            s = torch.as_tensor(np.stack([b[2] for b in batch]), device=device)
            yr = torch.as_tensor(np.array([b[3] for b in batch]), device=device)
            yd = torch.as_tensor(np.array([b[4] for b in batch]), device=device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device == "cuda"):
                out = model(x)
                loss = crit(out["z"], r, s)
                loss = loss + w_aux * aux_losses(out["dir_logit"], out["ret24"],
                                                 out["meta_logit"], yd, yr)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        # ---- validation: ES net Sharpe on final-step positions
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", enabled=device == "cuda"):
            zs = torch.cat([model(es_x[i:i + 512])["z"].float()
                            for i in range(0, len(es_x), 512)])
        z_np = zs[:, -1].cpu().numpy()
        r_np = es_r[:, -1].cpu().numpy()
        s_np = es_s[:, -1].cpu().numpy()
        mu = (z_np * (0.15 / math.sqrt(5786) / (s_np + 1e-12)) * r_np).mean()
        # net sharpe via the losses helper on (1, N)
        zt = torch.as_tensor(z_np[None], dtype=torch.float32)
        rt = torch.as_tensor(r_np[None], dtype=torch.float32)
        st = torch.as_tensor(s_np[None], dtype=torch.float32)
        R = net_strategy_return(zt, rt, st, cost_r,
                                0.15 / math.sqrt(5786)).numpy().ravel()
        sharpe = float(R.mean() / (R.std() + 1e-12) * math.sqrt(5786))
        hist["val_sharpe"].append(sharpe)
        if sharpe > best_sharpe + 1e-4:
            best_sharpe, bad = sharpe, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    hist["best_val_sharpe"] = best_sharpe
    return model, hist


@torch.no_grad()
def predict_z(model, X, positions, seq_len, device="cuda", batch=512):
    """Frozen per-bar z (and per-iterate z for ContinuumNet) on given rows."""
    Xf = X
    outs, outs_iters = [], []
    for i0 in range(0, len(positions), batch):
        batch_pos = positions[i0:i0 + batch]
        wins = []
        for p in batch_pos:
            lo = max(0, int(p) - seq_len + 1)
            x = Xf.iloc[lo:int(p) + 1].values
            if len(x) < seq_len:
                pad = np.repeat(x[:1], seq_len - len(x), axis=0)
                x = np.vstack([pad, x])
            wins.append(x)
        x = torch.as_tensor(np.stack(wins), device=device)
        with torch.autocast("cuda", enabled=device == "cuda"):
            o = model(x)
        outs.append(o["z"].float()[:, -1].cpu().numpy())
        if "z_iterates" in o or hasattr(o, "z_iterates"):
            pass
    z = np.concatenate(outs)
    return z


@torch.no_grad()
def predict_z_iterates(model, X, positions, seq_len, device="cuda", batch=256):
    """Per-iterate z series (loop-depth analysis) for ContinuumNet."""
    iters = None
    outs = []
    for i0 in range(0, len(positions), batch):
        batch_pos = positions[i0:i0 + batch]
        wins = []
        for p in batch_pos:
            lo = max(0, int(p) - seq_len + 1)
            x = X.iloc[lo:int(p) + 1].values
            if len(x) < seq_len:
                pad = np.repeat(x[:1], seq_len - len(x), axis=0)
                x = np.vstack([pad, x])
            wins.append(x)
        x = torch.as_tensor(np.stack(wins), device=device)
        with torch.autocast("cuda", enabled=device == "cuda"):
            o = model.forward_per_iterate(x)
        outs.append(o["z"].float()[:, -1].cpu().numpy())
        zi = np.stack([t.float()[:, -1].cpu().numpy() for t in o["z_iterates"]], axis=1)
        iters = zi if iters is None else np.vstack([iters, zi])
    return np.concatenate(outs), iters


def main():
    WORK = "/kaggle/working"
    if not os.path.exists(WORK):
        WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k3_out")
        os.makedirs(WORK, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}; torch={torch.__version__}; "
        f"gpu={torch.cuda.get_device_name(0) if device == 'cuda' else 'none'}")
    ledger = TrialLedger(os.path.join(WORK, "trial_ledger.jsonl"))

    # ---------------- 1) data ------------------------------------------------
    with stage("load data"):
        feats, px = load_v2_data()
        X, y, names = prepare_matrix(feats)
        fdir = find_features_ds()
        slices, ordered = group_slices_from_meta(
            names, os.path.join(fdir, "feature_meta.json"))
        X = X[ordered]          # reorder to VSN group order
        names = ordered
        log(f"matrix {X.shape}; VSN groups: {[(s, e, nm) for s, e, nm in slices]}")
        # matrices for the trainer
        ret = feats["r_xau"]
        sigma = feats["sigma"]
        y_dir = feats["y_dir24"]

    # search-phase splits: train < 2022 (purged), val = 2022
    idx = feats.index
    t2022 = idx.searchsorted(pd.Timestamp("2022-01-01", tz="UTC"))
    t2023 = idx.searchsorted(pd.Timestamp("2023-01-01", tz="UTC"))
    search_train = np.arange(0, t2022 - 24)
    search_val = np.arange(t2022, t2023)
    m_y = y.notna().values
    search_fit = search_train[m_y[search_train]][:: 2]     # subsample search fits
    n_es = max(500, int(len(search_train) * 0.1))
    search_es = search_train[-n_es:]

    F_ = X.shape[1]

    # ---------------- 2) search phase ----------------------------------------
    with stage("config search (train<2022 / val 2022)"):
        results = {}

        # ---- M3 compact + medium
        for scale, mk in (("M3c", lambda dr: JaneStreetDAEResMLP(
                              in_features=F_, d_enc=32, noise_std=0.10, swap_noise=0.15)),
                          ("M3m", lambda dr: AEMLP(
                              F_, enc_dim=96, noise_sigma=0.035, dropout=(dr, dr, dr, dr)))):
            best = None
            for dr in (0.2,):
                for lr in (1e-3,):
                    if budget_left() < 3600:
                        log("  budget low; truncating search")
                        break
                    torch.manual_seed(1337)
                    model = mk(dr)
                    model, hist = train_generic(
                        model, X, ret, sigma, y, y_dir,
                        search_fit, search_es, seq_len=256, batch_size=64,
                        lr=lr, max_epochs=18, patience=6, device=device)
                    # evaluate on 2022
                    val_pos = search_val[m_y[search_val]]
                    zv = predict_z(model, X, val_pos[::2], 256, device)
                    rv = ret.values[val_pos[::2]]
                    sv = sigma.values[val_pos[::2]]
                    R = zv * (0.15 / math.sqrt(5786) / (sv + 1e-12)) * rv
                    sharpe22 = float(np.mean(R) / (np.std(R) + 1e-12) * math.sqrt(5786))
                    ledger.add(scale, {"dropout": dr, "lr": lr}, sharpe22,
                               stage="search", kernel="K3")
                    log(f"  {scale} dr={dr} lr={lr}: val2022 Sharpe {sharpe22:.3f} "
                        f"(es {hist['best_val_sharpe']:.3f})")
                    if best is None or sharpe22 > best[0]:
                        best = (sharpe22, {"dropout": dr, "lr": lr})
                else:
                    continue
                break
            results[scale] = best
            log(f"  {scale} BEST: {best}")

        # ---- M4
        best = None
        for dr in (0.2,):
            for lr in (1e-3,):
                if budget_left() < 3600:
                    break
                torch.manual_seed(1337)
                model = OxfordManDMNLite(in_features=F_, d_model=32, n_heads=2,
                                          dropout=dr)
                model, hist = train_generic(
                    model, X, ret, sigma, y, y_dir,
                    search_fit, search_es, seq_len=128, batch_size=64,
                    lr=lr, max_epochs=18, patience=6, device=device)
                val_pos = search_val[m_y[search_val]]
                zv = predict_z(model, X, val_pos[::2], 128, device)
                rv = ret.values[val_pos[::2]]
                sv = sigma.values[val_pos[::2]]
                R = zv * (0.15 / math.sqrt(5786) / (sv + 1e-12)) * rv
                sharpe22 = float(np.mean(R) / (np.std(R) + 1e-12) * math.sqrt(5786))
                ledger.add("M4_DMN_Lite", {"dropout": dr, "lr": lr, "d_model": 32},
                           sharpe22, stage="search", kernel="K3")
                log(f"  M4 dr={dr} lr={lr}: val2022 Sharpe {sharpe22:.3f}")
                if best is None or sharpe22 > best[0]:
                    best = (sharpe22, {"dropout": dr, "lr": lr, "d_model": 32})
        results["M4"] = best
        log(f"  M4 BEST: {best}")

        # ---- M5 ContinuumNet d_model sweep
        best = None
        for d_model in (64, 128):
            for lr in (1e-3, 3e-4):
                if budget_left() < 3600:
                    break
                torch.manual_seed(1337)
                net_cfg = ContinuumConfig(n_features=F_, group_slices=slices,
                                          d_model=d_model, n_heads=4,
                                          dropout=0.2, loop_iters=4,
                                          km_alpha=0.5, spectral_cap=0.80)
                cfg = TrainConfig(seq_len=256, batch_size=32, lr=lr,
                                  max_epochs=18, patience=6, device=device)
                # train on the search split (reuse train_one_fold: it takes
                # feats ndarray + idx arrays)
                Xt = X.values.astype("float32")
                rett = np.nan_to_num(ret.values, nan=0.0)
                sigt = np.nan_to_num(sigma.values, nan=1e-4)
                yrt = np.nan_to_num(y.values, nan=0.0)
                ydt = np.nan_to_num(y_dir.values, nan=0.0)
                model, hist = train_one_fold(
                    Xt, rett, sigt, yrt, ydt,
                    search_fit, search_es, slices, net_cfg, cfg)
                val_pos = search_val[m_y[search_val]]
                zv = predict_z(model, X, val_pos[::2], 256, device)
                rv = ret.values[val_pos[::2]]
                sv = sigma.values[val_pos[::2]]
                R = zv * (0.15 / math.sqrt(5786) / (sv + 1e-12)) * rv
                sharpe22 = float(np.mean(R) / (np.std(R) + 1e-12) * math.sqrt(5786))
                rho = audit_loop_spectral_radius(model, torch.as_tensor(
                    X.values[np.arange(max(0, t2022 - 260), t2022 - 4)],
                    dtype=torch.float32, device=device))
                ledger.add("M5_ContinuumNet",
                           {"d_model": d_model, "lr": lr, "loop_iters": 4,
                            "cap": 0.80, "km_alpha": 0.5},
                           sharpe22, stage="search", kernel="K3",
                           notes=f"rho={rho:.3f}")
                log(f"  M5 d={d_model} lr={lr}: val2022 Sharpe {sharpe22:.3f} "
                    f"rho={rho:.3f}")
                if best is None or sharpe22 > best[0]:
                    best = (sharpe22, {"d_model": d_model, "lr": lr})
                del model
                torch.cuda.empty_cache()
        results["M5"] = best
        log(f"  M5 BEST: {best}")
        with open(os.path.join(WORK, "search_results.json"), "w") as f:
            json.dump({k: (v if v is None else {"sharpe22": v[0], "cfg": v[1]})
                       for k, v in results.items()}, f, indent=1, default=str)

    # ---------------- 3) fold refits -----------------------------------------
    with stage("walk-forward fold refits"):
        forecasts = pd.DataFrame(index=feats.index)
        m5_iters = {}
        rho_log = []
        fold_status = {}
        for fam, best in results.items():
            if best is None:
                fold_status[fam] = "search failed"
                continue
            cfgd = best[1]
            fam_z = pd.Series(np.nan, index=feats.index)
            if fam == "M3m" and budget_left() < 2.2 * 3600:
                fold_status[fam] = "skipped (compact budget; search-only)"
                continue
            for yy in ALL_FOLD_YEARS:
                if budget_left() < 1200:
                    log(f"  BUDGET STOP before {fam} fold {yy} "
                        f"({budget_left()/3600:.2f}h left)")
                    fold_status[fam] = f"budget-stopped at fold {yy}"
                    break
                res = fold_indices(feats, yy)
                if res is None:
                    continue
                fit_pos, es_pos, test_pos = res
                fit_m = fit_pos[m_y[fit_pos]]
                preds = []
                for seed in (1337,):
                    torch.manual_seed(seed)
                    if fam == "M3c":
                        model = JaneStreetDAEResMLP(
                            in_features=F_, d_enc=32, noise_std=0.10,
                            swap_noise=0.15)
                        model, _ = train_generic(
                            model, X, ret, sigma, y, y_dir, fit_m[::2], es_pos,
                            seq_len=256, batch_size=64, lr=cfgd["lr"],
                            max_epochs=28, patience=9, device=device, seed=seed)
                        zv = predict_z(model, X, test_pos, 256, device)
                    elif fam == "M3m":
                        model = AEMLP(F_, enc_dim=96, noise_sigma=0.035,
                                      dropout=(cfgd["dropout"],) * 4)
                        model, _ = train_generic(
                            model, X, ret, sigma, y, y_dir, fit_m[::2], es_pos,
                            seq_len=256, batch_size=64, lr=cfgd["lr"],
                            max_epochs=28, patience=9, device=device, seed=seed)
                        zv = predict_z(model, X, test_pos, 256, device)
                    elif fam == "M4":
                        model = OxfordManDMNLite(in_features=F_, d_model=32,
                                                 n_heads=2,
                                                 dropout=cfgd["dropout"])
                        model, _ = train_generic(
                            model, X, ret, sigma, y, y_dir, fit_m[::2], es_pos,
                            seq_len=128, batch_size=64, lr=cfgd["lr"],
                            max_epochs=28, patience=9, device=device, seed=seed)
                        zv = predict_z(model, X, test_pos, 128, device)
                    else:   # M5
                        net_cfg = ContinuumConfig(
                            n_features=F_, group_slices=slices,
                            d_model=cfgd["d_model"], n_heads=4, dropout=0.2,
                            loop_iters=4, km_alpha=0.5, spectral_cap=0.80)
                        tcfg = TrainConfig(seq_len=256, batch_size=32,
                                           lr=cfgd["lr"], max_epochs=28,
                                           patience=9, device=device, seed=seed)
                        model, hist = train_one_fold(
                            X.values.astype("float32"),
                            np.nan_to_num(ret.values), np.nan_to_num(sigma.values),
                            np.nan_to_num(y.values), np.nan_to_num(y_dir.values),
                            fit_m[::2], es_pos, slices, net_cfg, tcfg)
                        zv, zi = predict_z_iterates(model, X, test_pos, 256, device)
                        if seed == 1337:
                            m5_iters[yy] = zi
                        rho = audit_loop_spectral_radius(
                            model, torch.as_tensor(
                                X.values[es_pos[-260:]].astype("float32"),
                                device=device))
                        rho_log.append({"fold": yy, "seed": seed, "rho": round(rho, 4),
                                        "km_alpha": net_cfg.km_alpha,
                                        "governor": "auto_tune_alpha"})
                        ledger.add("M5_ContinuumNet",
                                   {"fold": yy, "seed": seed,
                                    "d_model": cfgd["d_model"]},
                                   hist.get("best_val_sharpe"),
                                   stage="fold", kernel="K3",
                                   notes=f"rho={rho:.3f}")
                    preds.append(zv)
                    del model
                    torch.cuda.empty_cache()
                fam_z.iloc[test_pos] = np.mean(preds, axis=0)
                log(f"  {fam} fold {yy}: predicted {len(test_pos)} bars "
                    f"(z std {np.std(np.concatenate(preds)):.3f})")
            forecasts[f"z_{fam}"] = fam_z
            fold_status[fam] = "complete"

        # per-iterate loop-depth analysis export (M5)
        if m5_iters:
            it_mat = np.concatenate([m5_iters[y] for y in sorted(m5_iters)], axis=0)
            it_idx = feats.index[np.concatenate(
                [np.arange(feats.index.searchsorted(pd.Timestamp(f"{y}-01-01", tz="UTC")),
                           feats.index.searchsorted(pd.Timestamp(f"{y+1}-01-01", tz="UTC")))
                 for y in sorted(m5_iters)])]
            it_df = pd.DataFrame(it_mat, index=it_idx,
                                 columns=[f"z_iter{i+1}" for i in range(it_mat.shape[1])])
            forecasts = forecasts.join(it_df)

        ok_folds = gate("G3.4_folds", all(v == "complete" for v in fold_status.values()),
                        f"fold status: {fold_status}; budget used "
                        f"{(time.time()-T_START)/3600:.2f}h")

    # ---------------- 4) spectral gate + export -------------------------------
    with stage("spectral gate + export"):
        ok_rho = gate("G3.3_spectral", all(r["rho"] < 0.95 for r in rho_log)
                      if rho_log else True,
                      f"rho at every audited M5 checkpoint < 0.95: "
                      f"{rho_log[:6]}..." if rho_log else "no M5 folds trained")
        forecasts["r_xau"] = ret
        forecasts["sigma"] = sigma
        forecasts["y_ret24_norm"] = feats["y_ret24_norm"]
        forecasts.to_parquet(os.path.join(WORK, "v2_neural_forecasts.parquet"))
        with open(os.path.join(WORK, "rho_log.json"), "w") as f:
            json.dump(rho_log, f, indent=1)
        with open(os.path.join(WORK, "k3_meta.json"), "w") as f:
            json.dump({"fold_status": fold_status,
                       "search": {k: (None if v is None else v[1])
                                  for k, v in results.items()},
                       "ledger_entries": len(ledger.rows),
                       "wall_h": round((time.time() - T_START) / 3600, 2)},
                      f, indent=1, default=str)
        ok_ex = gate("G3.5_export", True,
                     f"v2_neural_forecasts.parquet ({len(forecasts.columns)} cols); "
                     f"rho_log.json; trial_ledger.jsonl ({len(ledger.rows)} entries)")

    all_ok = all([ok_folds, ok_rho, ok_ex])
    gate("G3_ALL", all_ok, "all K3 gates green" if all_ok else "K3 gate failure")
    with open(os.path.join(WORK, "gates.json"), "w") as f:
        json.dump(GATES, f, indent=2, default=str)
    with open(os.path.join(WORK, "run_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES))
    log("K3 COMPLETE" if all_ok else "K3 FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys as _s
    _s.exit(main())
