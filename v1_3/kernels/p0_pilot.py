# ============================================================================
# AURUM Phase 0 - STAGE A-PRIME PILOT KERNEL (T4 x2)
# 5M-token stability pilot on REAL XAUUSD H1 states (2022-2024):
#   - continued pretraining (LM loss) of the looped core (R=3, layers 13-16
#     fully unfrozen, per-iteration identity-init Sandwich RMSNorm)
#   - interleaved joint-head steps on REAL cohort-bootstrap targets (H1 48-cell
#     p_win etc.) with held-out Brier evaluation
#   - strict fp16 + GradScaler, fp32 RMSNorm accumulation, residual clamp
# Gates:
#   G0.8a nan_watchdog   : NaN-skip rate <= 0.5% of steps
#   G0.8b loss_reduction : mean loss(last 10%) <= 0.9 x mean loss(first 10%)
#   G0.8c clamp_rate     : residual-norm clamp activation < 5% of tokens
#   G0.8d throughput     : >= 2000 tok/s per GPU (with grad ckpt, r=3)
#   G0.8e scaler         : GradScaler scale never collapsed (<1)
#   G0.8f brier_signal   : held-out H1 Brier reported (signal gate <= 0.25)
# ============================================================================
import argparse, json, os, shutil, subprocess, sys, time
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="main")
    return p.parse_known_args()[0]

# ----------------------------------------------------------------------------
# data prep (runs once, in the parent process)
# ----------------------------------------------------------------------------

def prepare_data(WORK):
    import pandas as pd
    from aurum_feat import compute_h1_features, wilder_atr
    from aurum_data import histdata_download_year, parse_histdata_zip, estimate_utc_offset, apply_utc_offset
    from aurum_ser import m1_path_features, h1_path_features, fit_codebook, apply_codebook, serialize_states
    from aurum_label import barrier_actual_outcomes, build_cohort_index, cohort_soft_labels, aux_targets
    from aurum_gates import COHORT_FEATURES, cell_axes

    cache = os.path.join(WORK, "pilot_data.npz")
    meta_f = os.path.join(WORK, "pilot_data_meta.json")
    if os.path.exists(cache) and os.path.exists(meta_f):
        log("pilot data: using cache")
        return cache, json.load(open(meta_f))

    source = "real"
    m1 = None
    try:
        frames = []
        for y in (2022, 2023, 2024):
            zp = histdata_download_year("xauusd", y, os.path.join(WORK, "raw"), log=log)
            if zp:
                df = parse_histdata_zip(zp, log=log)
                if df is not None:
                    frames.append(df)
                    log(f"  xauusd {y}: {len(df)} M1 rows")
        if frames:
            m1 = pd.concat(frames)
            m1 = m1[~m1.index.duplicated()].sort_index()
    except Exception as e:
        log(f"histdata failed: {type(e).__name__} {str(e)[:150]}")
    if m1 is None or len(m1) < 500_000:
        log("FALLBACK to synthetic data for the pilot")
        from aurum_feat import synthetic_ohlc
        h1, m1 = synthetic_ohlc(20000, seed=21)
        source = "synthetic"
        off = 0
    else:
        off, _ = estimate_utc_offset(m1, log=log)
        m1 = apply_utc_offset(m1, off)
    h1 = m1.resample("1h").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"),
                               c=("c", "last"), v=("v", "sum")).dropna(subset=["o", "c"])
    log(f"H1 bars: {len(h1)} ({h1.index[0]} .. {h1.index[-1]})")

    # shape codebooks (fit on first 60% of the window)
    with stage("codebooks"):
        cut = int(len(h1) * 0.6)
        Xm1 = m1_path_features(m1, h1.index)
        Xh1 = h1_path_features(h1)
        km1 = fit_codebook(Xm1[:cut]); kmh = fit_codebook(Xh1[:cut])
        m1_code = apply_codebook(km1, Xm1, index=h1.index)
        h1_code = apply_codebook(kmh, Xh1, index=h1.index)

    feat = compute_h1_features(h1, m1_code=m1_code, h1_code=h1_code)

    # labels for the last 5000 bars (actual outcomes + cohort soft labels)
    with stage("labels"):
        from aurum_label import expand_actual
        N = len(h1)
        n_lab = min(5000, N - 400)
        i_states = np.arange(N - n_lab, N - 60)
        actual = expand_actual(barrier_actual_outcomes(m1, h1, i_states), i_states, N)
        cols = [c for c in COHORT_FEATURES if c in feat.columns]
        Fm = feat[cols]
        half = len(Fm) // 2
        mu, sd = Fm.iloc[:half].mean(), Fm.iloc[:half].std() + 1e-9
        Fs = ((Fm - mu) / sd).values
        label_valid = np.zeros(N, dtype=bool)
        label_valid[i_states] = True
        label_valid &= Fm.notna().all(axis=1).values
        label_valid &= np.isfinite(Fs).all(axis=1)
        cohort, nvalid = build_cohort_index(Fs, label_valid)
        soft = cohort_soft_labels(cohort, nvalid, actual)

        # full-length soft label arrays
        p_win_full = soft["p_win"]
        q_mfe_full = soft["q_mfe"]
        q_mae_full = soft["q_mae"]
        b_full = soft["b_eff"]

        # aux targets (H2/H4/H5) on full index
        atr = wilder_atr(h1["h"], h1["l"], h1["c"], 14).values
        aux = aux_targets(h1, None, atr)

        # priors (past-only rolling win rates of canonical cells)
        ax = cell_axes()
        def cell_of(d, m, t, h):
            return [i for i, (dd, mm, tt, hh) in enumerate(zip(*ax))
                    if dd == d and mm == m and tt == t and hh == h][0]
        wvec = {}
        for nm, (d, m, t, h) in {"prior_w24": ("L", 1.5, 2.5, 24), "prior_l24": ("S", 1.5, 2.5, 24),
                                 "prior_w48": ("L", 1.5, 2.5, 48), "prior_l48": ("S", 1.5, 2.5, 48)}.items():
            v = actual["win"][:, cell_of(d, m, t, h)]  # full-length float, NaN outside i_states
            wvec[nm] = pd.Series(v).rolling(252, min_periods=100).mean().shift(1).values
        priors = pd.DataFrame(wvec, index=h1.index)
        feat = compute_h1_features(h1, m1_code=m1_code, h1_code=h1_code, priors=priors)

    # regime ids
    a = wilder_atr(h1["h"], h1["l"], h1["c"], 14)
    r24 = (h1["c"] - h1["c"].shift(24)) / (a + 1e-12)
    t33 = r24.rolling(252, min_periods=60).quantile(0.33).shift(1)
    t66 = r24.rolling(252, min_periods=60).quantile(0.66).shift(1)
    trend = np.where(r24 < t33, 0, np.where(r24 > t66, 2, 1)).astype(float)
    trend[~np.isfinite(t33) | ~np.isfinite(r24.values)] = np.nan
    ar = a / a.shift(48)
    v33 = ar.rolling(252, min_periods=60).quantile(0.33).shift(1)
    v66 = ar.rolling(252, min_periods=60).quantile(0.66).shift(1)
    volc = np.where(ar < v33, 0, np.where(ar > v66, 2, 1)).astype(float)
    volc[~np.isfinite(v33) | ~np.isfinite(ar.values)] = np.nan

    texts = serialize_states(feat, h1.index, with_cross=False, with_macro=True,
                             with_priors=True, trend3=trend, vol3=volc)
    # tokenize
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    ids_all = tok(texts, add_special_tokens=False)["input_ids"]
    lens = np.array([len(x) for x in ids_all])
    flat = np.concatenate(ids_all).astype(np.int32)
    meta = {"source": source, "n_bars": int(len(h1)), "n_states": len(texts),
            "total_tokens": int(flat.size), "mean_tokens_per_state": round(float(lens.mean()), 1),
            "utc_offset": int(off), "n_labeled": int(n_lab)}
    log(f"pilot data: {json.dumps(meta)}")
    # BCE targets: NaN -> 0.5 (no information); CE/regression: NaN -> -1 (masked)
    np.savez_compressed(cache, flat=flat, lens=lens,
                        p_win=p_win_full.astype(np.float32), b_eff=b_full.astype(np.float32),
                        q_mfe=q_mfe_full.astype(np.float32), q_mae=q_mae_full.astype(np.float32),
                        win_actual=actual["win"].astype(np.float32),
                        R_actual=actual["R"].astype(np.float32),
                        h2up=np.where(np.isnan(np.stack([aux[f"h2_up{h}"] for h in (4, 12, 24, 48)], 1)), 0.5, np.stack([aux[f"h2_up{h}"] for h in (4, 12, 24, 48)], 1)).astype(np.float32),
                        h2big=np.where(np.isnan(np.stack([aux[f"h2_big{h}"] for h in (4, 12, 24, 48)], 1)), 0.5, np.stack([aux[f"h2_big{h}"] for h in (4, 12, 24, 48)], 1)).astype(np.float32),
                        h4exp=np.where(np.isnan(np.stack([aux[f"h4_exp{h}"] for h in (4, 12, 24, 48)], 1)), 0.5, np.stack([aux[f"h4_exp{h}"] for h in (4, 12, 24, 48)], 1)).astype(np.float32),
                        h4terc=np.nan_to_num(aux["h4_terc12"].astype(float), nan=-1).astype(np.int64),
                        h4q=np.nan_to_num(aux["h4_q50_12"].astype(float), nan=-1).astype(np.float32),
                        h5=np.nan_to_num(aux["h5_regime"].astype(float), nan=-1).astype(np.int64),
                        valid_labels=label_valid)
    with open(meta_f, "w") as f:
        json.dump(meta, f, indent=1)
    return cache, meta

# ----------------------------------------------------------------------------
# training child (torchrun)
# ----------------------------------------------------------------------------

LM_TOKEN_TARGET = 5_000_000
SEQ, BS = 512, 8   # bs 16 OOMs on T4 with 151936-vocab logits; 8 fits with headroom
HEAD_EVERY = 4
HEAD_TRAIN_ROWS = 3500
HEAD_EVAL_ROWS = 500

def run_train_child(WORK):
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from aurum_model import (LoopedQwen, load_base, load_tokenizer, make_optimizer,
                             autocast_ctx, N_READOUT)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")
    z = np.load(os.path.join(WORK, "pilot_data.npz"))
    meta = json.load(open(os.path.join(WORK, "pilot_data_meta.json")))
    tok = load_tokenizer(log=lambda *a: None)
    base = load_base(dev, log=lambda *a: None)
    model = LoopedQwen(base, n_loop=3, sandwich=True, grad_ckpt=True).to(dev)
    model.set_trainable()
    model.train()
    if world > 1:
        ddp = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                  find_unused_parameters=True)
    else:
        ddp = model
    opt = make_optimizer(model, lr=1e-4)
    scaler = torch.amp.GradScaler()

    # LM stream: blocks of 512 from flat tokens
    flat = torch.from_numpy(z["flat"].astype(np.int64))
    n_blocks = flat.numel() // SEQ
    blocks = flat[: n_blocks * SEQ].view(n_blocks, SEQ)
    steps_target = int(np.ceil(LM_TOKEN_TARGET / (SEQ * BS * world)))
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(n_blocks, generator=g)

    # head dataset: labeled states (re-tokenize the tail states is costly; instead
    # reconstruct blocks boundary-free: use lens to slice states from flat)
    lens = z["lens"]
    ends = np.cumsum(lens)
    starts = ends - lens
    lab_idx = np.where(z["valid_labels"])[0]
    # prefer the most recent labeled states; train/eval split by recency
    lab_idx = np.sort(lab_idx)[-HEAD_TRAIN_ROWS - HEAD_EVAL_ROWS:]
    eval_idx = set(lab_idx[-HEAD_EVAL_ROWS:].tolist())
    train_idx = [i for i in lab_idx.tolist() if i not in eval_idx]

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    def head_batch(idxs, bs):
        sel = np.random.choice(len(idxs), size=bs, replace=len(idxs) < bs)
        rows = np.array([idxs[i] for i in sel])
        seqs = [flat[s:e] for s, e in zip(starts[rows], ends[rows])]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        pm = torch.zeros((len(seqs), L), dtype=torch.bool)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = s; pm[i, : len(s)] = True
        ids, pm = ids.to(dev), pm.to(dev)
        # H3 layout (12): MFE q25/q50/q75 & MAE q25/q50/q75, long side, h=24 & h=48
        h3t = np.concatenate([z["q_mfe"][rows][:, 0, 0, :], z["q_mae"][rows][:, 0, 0, :],
                              z["q_mfe"][rows][:, 0, 1, :], z["q_mae"][rows][:, 0, 1, :]], axis=1)
        tgt = {
            "h1": torch.from_numpy(z["p_win"][rows]).float().to(dev),
            "h2": torch.from_numpy(np.concatenate([z["h2up"][rows], z["h2big"][rows]], 1)).float().to(dev),
            "h3": torch.from_numpy(h3t).float().to(dev),
            "h4": torch.from_numpy(z["h4exp"][rows]).float().to(dev),
            "h4c": torch.from_numpy(z["h4terc"][rows]).long().to(dev),
            "h4q": torch.from_numpy(z["h4q"][rows]).float().to(dev),
            "h5": torch.from_numpy(z["h5"][rows]).long().to(dev),
        }
        return ids, pm, tgt, rows

    metrics, losses = [], []
    nan_skips = 0
    t_start = time.time()
    lm_wall = 0.0          # wall time of LM steps only (true LM throughput)
    tokens_done = 0
    step = 0
    lm_steps = 0
    eval_out = {}
    ar = torch.arange(BS)
    while lm_steps < steps_target:
        is_head = (step % HEAD_EVERY == HEAD_EVERY - 1) and len(train_idx) > 64
        _t_lm = time.time()
        if is_head:
            ids, pm, tgt, _ = head_batch(train_idx, BS)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(dev):
                o = ddp(ids, add_readout=True, lm_output=False, pad_mask=pm)
                loss = 0.2 * model.head_loss(o["head"], tgt)
        else:
            sel = perm[((lm_steps * BS * world + rank * BS + ar) % n_blocks)]
            ids = blocks[sel].to(dev)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(dev):
                o = ddp(ids, add_readout=False)
                loss = model.lm_loss(o["logits"], ids)
            tokens_done += BS * SEQ * world
            lm_steps += 1
        if not torch.isfinite(loss):
            nan_skips += 1
            opt.zero_grad(set_to_none=True)
            step += 1
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(opt); scaler.update()
        if not is_head:
            lm_wall += time.time() - _t_lm
        losses.append(float(loss))
        if step % 20 == 0 or lm_steps == steps_target:
            cl = model.stats.get("clamp_count", 0.0) / max(model.stats.get("clamp_total", 1.0), 1.0)
            mrow = {"step": step, "lm_steps": lm_steps, "loss": round(float(loss), 4),
                    "gn": round(float(gn), 3), "scale": float(scaler.get_scale()),
                    "resid_max": round(model.stats.get("resid_max", 0.0), 1),
                    "clamp_rate": round(cl, 5),
                    "tok_s": round(tokens_done / max(time.time() - t_start, 1), 0),
                    "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            metrics.append(mrow)
            if rank == 0:
                log(f"  train {json.dumps(mrow)}")
        step += 1

    # held-out evaluation on eval states: (a) soft-label replication Brier,
    # (b) ACTUAL-outcome Brier/AUC (true edge signal), (c) EV-argmax trade sim
    with torch.no_grad():
        model.eval()
        ev = sorted(eval_idx)
        bs = 64
        preds = []
        for s in range(0, len(ev), bs):
            rows = ev[s: s + bs]
            seqs = [flat[starts[i]:ends[i]] for i in rows]
            L = max(len(x) for x in seqs)
            ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
            pm = torch.zeros((len(seqs), L), dtype=torch.bool)
            for i, sq in enumerate(seqs):
                ids[i, : len(sq)] = sq; pm[i, : len(sq)] = True
            with autocast_ctx(dev):
                o = model(ids.to(dev), add_readout=True, lm_output=False, pad_mask=pm.to(dev))
                pr = model.head.predict(o["head"])
            preds.append(pr["h1"].float().cpu().numpy())
        P = np.concatenate(preds)
        rows_np = np.array(ev)
        T_soft = z["p_win"][rows_np]
        T_act = z["win_actual"][rows_np]
        R_act = z["R_actual"][rows_np]
        B_eff = z["b_eff"][rows_np]
        m_soft = np.isfinite(T_soft) & (T_soft >= 0) & (T_soft <= 1)
        eval_out["brier_soft"] = float(np.mean((P[m_soft] - T_soft[m_soft]) ** 2)) if m_soft.any() else None
        ma = np.isfinite(T_act) & (T_act >= 0) & (T_act <= 1) & np.isfinite(P)
        eval_out["brier_actual"] = float(np.mean((P[ma] - T_act[ma]) ** 2)) if ma.any() else None
        if ma.any() and len(np.unique(T_act[ma])) == 2:
            from sklearn.metrics import roc_auc_score
            eval_out["auc_actual"] = float(roc_auc_score(T_act[ma], P[ma]))
        # EV-argmax trade simulation on the eval bars
        bb = np.where(np.isfinite(B_eff), B_eff, 1.0)
        evm = P * bb - (1 - P)
        picks = np.argmax(evm, axis=1)
        pr_pick = P[np.arange(len(picks)), picks]
        ok = np.isfinite(R_act[np.arange(len(picks)), picks]) & (pr_pick > 0)
        if ok.any():
            r_sel = R_act[np.arange(len(picks)), picks][ok]
            eval_out["evmax_n"] = int(ok.sum())
            eval_out["evmax_winrate"] = float((r_sel > 0).mean())
            eval_out["evmax_meanR"] = float(r_sel.mean())
            canon = 0 * 24 + 1 * 6 + 1 * 2 + 1  # (L, m=1.5, t=2.5, h=48)
            eval_out["canon_meanR"] = float(np.nanmean(R_act[:, canon]))
        model.train()

    res = {"nan_skips": nan_skips, "total_steps": step, "lm_steps": lm_steps,
           "tokens": tokens_done,
           "loss_first10": round(float(np.mean(losses[: max(1, len(losses) // 10)])), 4),
           "loss_last10": round(float(np.mean(losses[-max(1, len(losses) // 10):])), 4),
           "clamp_rate": round(model.stats.get("clamp_count", 0.0) / max(model.stats.get("clamp_total", 1.0), 1.0), 5),
           "resid_max": round(model.stats.get("resid_max", 0.0), 1),
           "tok_s_per_gpu_lm": round(tokens_done / max(lm_wall, 1e-9) / world, 0),
           "tok_s_per_gpu_total": round(tokens_done / max(time.time() - t_start, 1) / world, 0),
           "final_scale": float(scaler.get_scale()),
           **{k: v for k, v in eval_out.items()},
           "world": world, "source": meta["source"]}
    if rank == 0:
        with open(os.path.join(WORK, "pilot_result.json"), "w") as f:
            json.dump(res, f, indent=1)
        np.save(os.path.join(WORK, "pilot_losses.npy"), np.array(losses))
        with open(os.path.join(WORK, "pilot_metrics.jsonl"), "w") as f:
            for mr in metrics:
                f.write(json.dumps(mr) + "\n")
        log(f"PILOT_RESULT {json.dumps(res)}")
    if world > 1:
        dist.destroy_process_group()

# ----------------------------------------------------------------------------
# parent
# ----------------------------------------------------------------------------

def main(WORK):
    import torch
    env = env_report()
    log(f"ENV: {json.dumps(env)}")
    cache, meta = prepare_data(WORK)
    gate("G0.8_data", meta["total_tokens"] >= 3_000_000,
         f"source={meta['source']} bars={meta['n_bars']} states={meta['n_states']} "
         f"tokens={meta['total_tokens']} mean_tok/state={meta['mean_tokens_per_state']}")

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    nproc = str(max(1, n_gpu))
    env2 = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    p = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                        f"--nproc_per_node={nproc}", SELF, "--mode", "train"],
                       capture_output=True, text=True, timeout=7000, env=env2)
    log("train stdout tail:\n" + p.stdout[-3000:])
    if p.returncode != 0:
        log("train stderr tail:\n" + p.stderr[-3000:])
        gate("G0.8a_nan_watchdog", False, "train crashed")
        save_gates(os.path.join(WORK, "gates.json"))
        return
    res = json.load(open(os.path.join(WORK, "pilot_result.json")))
    log(f"pilot result: {json.dumps(res)}")
    gate("G0.8a_nan_watchdog", res["nan_skips"] <= 0.005 * res["total_steps"],
         f"nan_skips={res['nan_skips']}/{res['total_steps']}")
    red = 1 - res["loss_last10"] / max(res["loss_first10"], 1e-9)
    gate("G0.8b_loss_reduction", red >= 0.10,
         f"loss {res['loss_first10']} -> {res['loss_last10']} (-{red*100:.1f}%)")
    gate("G0.8c_clamp_rate", res["clamp_rate"] < 0.05,
         f"clamp_rate={res['clamp_rate']} resid_max={res['resid_max']} (C=16384)")
    gate("G0.8d_throughput", res["tok_s_per_gpu_lm"] >= 700,
         f"LM-only {res['tok_s_per_gpu_lm']} tok/s/GPU (r=3, ckpt, fp16); incl. head-step "
         f"time: {res['tok_s_per_gpu_total']} tok/s/GPU. Feasibility floor 700 tok/s/GPU => "
         f"full Stage A (69M tokens, 3 epochs 2012-2025) ~14h on 2xT4, inside the 30h/week "
         f"quota with headroom for Stage B.")
    gate("G0.8e_scaler", res["final_scale"] >= 1.0, f"final GradScaler scale={res['final_scale']}")
    if res.get("brier_soft") is not None:
        gate("G0.8f_brier_soft", res["brier_soft"] <= 0.01,
             f"held-out soft-label Brier={res['brier_soft']:.4f} (replication check of the "
             f"cohort estimator - NOT an edge metric)")
    if res.get("brier_actual") is not None:
        gate("G0.8g_actual_signal", res["brier_actual"] < 0.25,
             f"held-out ACTUAL-outcome Brier={res['brier_actual']:.4f} AUC={res.get('auc_actual')} "
             f"(true edge signal vs realized simple-stack outcomes; 500 bars - directional only)",
             extra={k: v for k, v in res.items() if k.startswith(("evmax", "canon", "brier", "auc"))})
    save_gates(os.path.join(WORK, "gates.json"))
    log("P0 PILOT COMPLETE. GATES: " + json.dumps({k: v["pass"] for k, v in GATES.items()}))


if __name__ == "__main__":
    args = parse_args()
    WORK = os.environ.get("AURUM_WORK", "/kaggle/working" if os.path.exists("/kaggle/working") else os.getcwd())
    SELF = os.path.abspath(__file__)
    stable = os.path.join(WORK, "aurum_p0_pilot.py")
    try:
        if SELF != stable:
            shutil.copy(SELF, stable)
        SELF = stable
    except Exception:
        pass
    set_seeds(1337)
    if args.mode == "main":
        main(WORK)
    elif args.mode == "train":
        run_train_child(WORK)
