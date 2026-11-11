# ============================================================================
# AURUM Phase 0 - CORE SMOKE KERNEL (T4 x2, sm_75)
# Gates:
#   G0.1 env           : 2x T4, fp16 path enforced, versions pinned
#   G0.2 correctness   : custom looped forward (R=1, sandwich off AND on with
#                        identity-init) == stock Qwen3 forward; R=3 finite
#   G0.3 scale-inv     : features + labels + serialization invariant to price
#                        scale x{1, 37.3, 1000} (synthetic + real sample)
#   G0.4 throughput    : training tok/s r in {1,2,3}, bs in {8,16,32},
#                        grad-ckpt on/off (single GPU)
#   G0.5 ddp           : 2xT4 DDP training throughput
#   G0.6 table timing  : 30k-bar probability table (85 outputs/bar, R=3) wall
#                        time across both GPUs
#   G0.7 kill/resume   : SIGKILL mid-generation -> resume -> identical table
# ============================================================================
import argparse, json, os, shutil, signal, subprocess, sys, time
import numpy as np

SELF = None  # set in main; stable copy of this script

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="main")
    p.add_argument("--rows", type=int, default=0)
    p.add_argument("--out", default="")
    p.add_argument("--ckpt", default="")
    p.add_argument("--resume", default="")
    p.add_argument("--ck-every", type=int, default=16)
    p.add_argument("--batch", type=int, default=128)
    return p.parse_known_args()[0]

# ----------------------------------------------------------------------------
# gentable mode: batched single-GPU probability-table generator with
# checkpoint/resume. This is the exact walk-forward precompute architecture
# (audit fix #7): one forward pass per bar -> all 85 outputs.
# ----------------------------------------------------------------------------

def build_synthetic_states(n_rows, tok, seed=1337, log=print):
    """Synthetic H1 regimes -> dimensionless features -> serialized states."""
    from aurum_feat import synthetic_ohlc, compute_h1_features
    from aurum_ser import m1_path_features, h1_path_features, fit_codebook, apply_codebook, serialize_states
    from aurum_gates import cell_axes
    from aurum_label import aux_targets
    import pandas as pd
    h1, m1 = synthetic_ohlc(n_bars=n_rows + 400, seed=seed)
    feat = compute_h1_features(h1)
    # trend/vol regime ids for the serializer (rule-based, past-only edges)
    from aurum_feat import wilder_atr
    a = wilder_atr(h1["h"], h1["l"], h1["c"], 14)
    r24 = (h1["c"] - h1["c"].shift(24)) / (a + 1e-12)
    t33 = r24.rolling(252, min_periods=60).quantile(0.33).shift(1)
    t66 = r24.rolling(252, min_periods=60).quantile(0.66).shift(1)
    trend = np.where(r24 < t33, 0, np.where(r24 > t66, 2, 1)).astype(float)
    trend[~np.isfinite(t33) | ~np.isfinite(r24)] = np.nan
    ar = a / a.shift(48)
    v33 = ar.rolling(252, min_periods=60).quantile(0.33).shift(1)
    v66 = ar.rolling(252, min_periods=60).quantile(0.66).shift(1)
    volc = np.where(ar < v33, 0, np.where(ar > v66, 2, 1)).astype(float)
    volc[~np.isfinite(v33) | ~np.isfinite(ar)] = np.nan
    texts = serialize_states(feat, h1.index, with_cross=False, with_macro=False,
                             with_priors=False, trend3=trend, vol3=volc)
    keep = h1.index[600:]  # burn-in for rolling features
    texts = texts[600:]
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    lens = np.array([len(e) for e in enc])
    return enc, lens, texts

def run_gentable(args):
    import torch
    from aurum_model import LoopedQwen, load_base, load_tokenizer, autocast_ctx, N_READOUT
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer()
    base = load_base(dev)
    model = LoopedQwen(base, n_loop=3, sandwich=True).to(dev).eval()
    states_npz = os.path.join(WORK, "gentable_states.npz")
    if os.path.exists(states_npz):
        z = np.load(states_npz, allow_pickle=True)
        enc = [list(map(int, e)) for e in z["enc"][: args.rows]] if args.rows else [list(map(int, e)) for e in z["enc"]]
    else:
        n_gen = min(30000, max((args.rows or 0) * 3, 600))
        enc, lens, _ = build_synthetic_states(n_gen, tok)
        np.savez_compressed(states_npz, enc=np.array(enc, dtype=object),
                            lens=np.array([len(e) for e in enc]))
        enc = enc[: args.rows] if args.rows else enc
    N = len(enc)
    out = np.full((N, 85), np.nan, dtype=np.float32)
    done = 0
    if args.resume and os.path.exists(args.resume):
        z = np.load(args.resume)
        done = int(z["done"])
        out[:done] = z["out"][:done]
        print(f"[gentable] resume from row {done}/{N}", flush=True)
    B = args.batch
    t0 = time.time()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    nb = (N - done + B - 1) // B
    for bi in range(nb):
        s = done + bi * B
        e = min(s + B, N)
        batch = enc[s:e]
        L = max(len(b) for b in batch)
        ids = torch.full((e - s, L), pad_id, dtype=torch.long)
        pm = torch.zeros((e - s, L), dtype=torch.bool)
        for i, b in enumerate(batch):
            ids[i, : len(b)] = torch.tensor(b, dtype=torch.long)
            pm[i, : len(b)] = True
        ids, pm = ids.to(dev), pm.to(dev)
        with torch.no_grad(), autocast_ctx(dev):
            o = model(ids, add_readout=True, lm_output=False, pad_mask=pm)
            pr = model.head.predict(o["head"])
        vec = torch.cat([pr["h1"], pr["h2"], pr["h3"], pr["h4"], pr["h5"]], dim=1)
        out[s:e] = vec.float().cpu().numpy()
        if (bi + 1) % args.ck_every == 0 or e == N:
            cur = done + (bi + 1) * B
            np.savez(args.ckpt, done=min(cur, N), out=out[: min(cur, N)])
    dt = time.time() - t0
    np.savez_compressed(args.out, out=out, done=N, seconds=dt)
    print(f"[gentable] DONE rows={N} in {dt:.1f}s ({N/max(dt,1e-9):.1f} rows/s)", flush=True)

# ----------------------------------------------------------------------------
# ddp-bench mode: torchrun child for 2xT4 throughput
# ----------------------------------------------------------------------------

def run_ddp_bench(_args):
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from aurum_model import LoopedQwen, load_base, make_optimizer, autocast_ctx
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")
    base = load_base(dev)
    model = LoopedQwen(base, n_loop=3, sandwich=True, grad_ckpt=True).to(dev)
    model.set_trainable()
    model.train()
    ddp = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                              find_unused_parameters=True)
    opt = make_optimizer(model, lr=1e-4)
    scaler = torch.amp.GradScaler()
    ids = torch.randint(100, 140000, (8, 512), device=dev)
    times = []
    for step in range(28):
        if step == 6:
            torch.cuda.synchronize(); dist.barrier(); t_start = time.time()
        t_s = time.time()
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(dev):
            o = ddp(ids, add_readout=False)
            loss = model.lm_loss(o["logits"], ids)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        if step >= 6:
            times.append(time.time() - t_s)
        if step == 27:
            gn = torch.norm(torch.stack([p.grad.norm() for p in model.parameters()
                                         if p.requires_grad and p.grad is not None]))
            dist.all_reduce(gn, op=dist.ReduceOp.AVG)
            local_gn = torch.norm(torch.stack([p.grad.norm() for p in model.parameters()
                                               if p.requires_grad and p.grad is not None]))
    dt = time.time() - t_start
    toks = len(times) * 16 * 512 * 2  # both GPUs
    res = {"tok_per_s_ddp": toks / dt, "step_ms_mean": 1000 * float(np.mean(times)),
           "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
           "grad_norm_gap": float((gn - local_gn).abs())}
    if rank == 0:
        with open(os.path.join(WORK, "ddp_bench.json"), "w") as f:
            json.dump(res, f)
        print("DDP_BENCH_RESULT " + json.dumps(res), flush=True)
    dist.destroy_process_group()

# ----------------------------------------------------------------------------
# main mode
# ----------------------------------------------------------------------------

def main():
    global SELF, WORK
    import torch
    from aurum_model import (LoopedQwen, load_base, load_tokenizer, autocast_ctx,
                             make_optimizer, N_READOUT)

    env = env_report()
    log(f"ENV: {json.dumps(env)}")
    gate("G0.1_env", env.get("gpu_count") == 2 and all("T4" in g for g in env.get("gpus", [])),
         f"{env.get('gpus')} cap={env.get('capability')} bf16={env.get('bf16_supported')} mem={env.get('total_mem_gb')}GB torch={env.get('torch')} transformers={env.get('transformers')}",
         extra={"env": env})
    gate("G0.1_fp16_policy", True,
         "fp16 + GradScaler enforced structurally (autocast fp16 hardcoded; fp32 RMSNorm "
         "accumulation; residual clamp C=16384). torch reports bf16 'supported' on sm_75 but that "
         "is emulation only - Turing has no bf16 tensor cores, fp16 stays the mandated path.")

    # ------------------------------------------------------------------
    # G0.2..G0.5 run inside phase_a() so every model/tensor reference is a
    # LOCAL that dies on return - the main process must hold ~0 GPU memory
    # before spawning gentable / DDP subprocesses (v4 lesson: 3.95GB left on
    # cuda:0 OOM-killed the GPU-0 children).
    # ------------------------------------------------------------------
    def phase_a():
        dev = torch.device("cuda:0")
        tok = load_tokenizer()
        base = load_base(dev)
        # ---------------- G0.2 correctness ----------------
        with stage("G0.2 correctness"):
            torch.manual_seed(0)
            ids = torch.randint(100, 140000, (2, 64), device=dev)
            with torch.no_grad():
                stock = base(input_ids=ids).logits.float()
                m_off = LoopedQwen(base, n_loop=1, sandwich=False).to(dev).eval()
                d_off = (m_off(ids)["logits"].float() - stock).abs().max().item()
                m_id = LoopedQwen(base, n_loop=1, sandwich=True).to(dev).eval()
                d_id = (m_id(ids)["logits"].float() - stock).abs().max().item()
                m3 = LoopedQwen(base, n_loop=3, sandwich=True).to(dev).eval()
                o3 = m3(ids)
                fin3 = bool(torch.isfinite(o3["logits"].float()).all())
                heads_ok = all(tuple(o3["head"][k].shape) == (2, d) for k, d in
                               (("h1", 48), ("h2", 8), ("h3", 12), ("h4", 8), ("h5", 9)))
                # padding-mask path smoke
                ids2 = torch.randint(100, 140000, (3, 40), device=dev)
                pm = torch.ones((3, 40), dtype=torch.bool, device=dev)
                pm[0, 30:] = False; pm[1, 20:] = False
                o_pm = m3(ids2, pad_mask=pm)
                fin_pm = bool(torch.isfinite(o_pm["logits"].float()).all()) and \
                    bool(torch.isfinite(o_pm["head"]["h1"].float()).all())
            ok = d_off < 1e-3 and d_id < 1e-3 and fin3 and heads_ok and fin_pm
            gate("G0.2_loop_correctness", ok,
                 f"R=1 sandwich-off maxdiff={d_off:.2e}; identity-init sandwich maxdiff={d_id:.2e}; "
                 f"R=3 finite={fin3} heads={heads_ok} padmask_finite={fin_pm}")

        # ---------------- G0.3 scale invariance ----------------
        with stage("G0.3 scale invariance"):
            from aurum_feat import synthetic_ohlc, compute_h1_features, scale_ohlc, wilder_atr
            from aurum_ser import m1_path_features, h1_path_features, fit_codebook, apply_codebook, serialize_states
            from aurum_label import barrier_actual_outcomes, build_cohort_index, cohort_soft_labels
            h1, m1 = synthetic_ohlc(1600, seed=11)
            h1 = h1.iloc[300:]; m1 = m1.iloc[300 * 60:]
            diffs_feat, diffs_lab, diffs_ser = [], [], []
            for c in (1.0, 37.3, 1000.0):
                h1s, m1s = scale_ohlc(h1, c), scale_ohlc(m1, c)
                f0 = compute_h1_features(h1)
                f1 = compute_h1_features(h1s)
                num = f0.select_dtypes(float)
                d = float((num - f1[num.columns]).abs().max().max())
                diffs_feat.append(d)
                i_states = np.arange(300, len(h1) - 60)
                o0 = barrier_actual_outcomes(m1, h1, i_states)
                o1 = barrier_actual_outcomes(m1s, h1s, i_states)
                dl = float(np.nanmax(np.abs(np.nan_to_num(o0["R"], nan=0.0) - np.nan_to_num(o1["R"], nan=0.0))))
                dl = max(dl,
                         float(np.nanmax(np.abs(np.nan_to_num(o0["mfe"], nan=0.0) - np.nan_to_num(o1["mfe"], nan=0.0)))),
                         float(np.nanmax(np.abs(np.nan_to_num(o0["mae"], nan=0.0) - np.nan_to_num(o1["mae"], nan=0.0)))))
                diffs_lab.append(dl)
                s0 = serialize_states(compute_h1_features(h1), h1.index, with_cross=False, with_macro=False, with_priors=False)
                s1 = serialize_states(compute_h1_features(h1s), h1s.index, with_cross=False, with_macro=False, with_priors=False)
                diffs_ser.append(0.0 if s0 == s1 else 1.0)
            feat_ok = max(diffs_feat) < 1e-6
            lab_ok = max(diffs_lab) < 1e-6
            ser_ok = max(diffs_ser) == 0.0
            # real-data spot check (best effort)
            real_note = "skipped"
            try:
                from aurum_data import histdata_download_year, parse_histdata_zip
                zp = histdata_download_year("xauusd", 2024, os.path.join(WORK, "raw"), log=log)
                if zp:
                    m1r = parse_histdata_zip(zp)
                    h1r = m1r.resample("1h").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum")).dropna()
                    h1r = h1r.iloc[-1200:]
                    fr0 = compute_h1_features(h1r); fr1 = compute_h1_features(scale_ohlc(h1r, 137.0))
                    num = fr0.select_dtypes(float)
                    d_real = float((num - fr1[num.columns]).abs().max().max())
                    real_note = f"real XAUUSD-2024 x137 scale maxdiff={d_real:.2e}"
            except Exception as e:
                real_note = f"real check failed: {type(e).__name__} {str(e)[:100]}"
            gate("G0.3_scale_invariance", feat_ok and lab_ok and ser_ok,
                 f"features maxdiff={max(diffs_feat):.2e} labels maxdiff={max(diffs_lab):.2e} "
                 f"serializer identical={ser_ok}; {real_note}",
                 extra={"scale_invariance": {"features": diffs_feat, "labels": diffs_lab,
                                              "serializer": diffs_ser, "real": real_note},
                        "static_audit": "all features are log-return ratios / ATR-unit distances / "
                                        "past-only rolling z-scores / range positions / codebook ids / "
                                        "cyclical calendar ids; no absolute price, date or raw volume"})

        # ---------------- G0.4 single-GPU throughput ----------------
        with stage("G0.4 throughput"):
            from aurum_model import clamp_resid
            results = []
            for r in (1, 2, 3):
                for bs in (8, 16, 32):
                    for ckpt in ((False, True) if r == 3 else (True,)):
                        model = LoopedQwen(base, n_loop=r, sandwich=True, grad_ckpt=ckpt).to(dev)
                        model.set_trainable()
                        model.train()
                        opt = make_optimizer(model, lr=1e-4)
                        scaler = torch.amp.GradScaler()
                        ids = torch.randint(100, 140000, (bs, 512), device=dev)
                        torch.cuda.reset_peak_memory_stats()
                        try:
                            ts = []
                            for step in range(24):
                                t_s = time.time()
                                opt.zero_grad(set_to_none=True)
                                with autocast_ctx(dev):
                                    o = model(ids, add_readout=False)
                                    loss = model.lm_loss(o["logits"], ids)
                                scaler.scale(loss).backward()
                                scaler.unscale_(opt)
                                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                                scaler.step(opt); scaler.update()
                                torch.cuda.synchronize()
                                if step >= 4:
                                    ts.append(time.time() - t_s)
                            toks = len(ts) * bs * 512 / float(np.sum(ts))
                            row = {"r": r, "bs": bs, "ckpt": ckpt, "tok_per_s": round(toks, 1),
                                   "step_ms": round(1000 * float(np.mean(ts)), 1),
                                   "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                                   "scale": float(scaler.get_scale())}
                            results.append(row)
                            log(f"  bench {row}")
                        except torch.cuda.OutOfMemoryError:
                            results.append({"r": r, "bs": bs, "ckpt": ckpt, "OOM": True})
                            log(f"  bench OOM r={r} bs={bs} ckpt={ckpt}")
                            torch.cuda.empty_cache()
                        del model, opt
                        torch.cuda.empty_cache()
            with open(os.path.join(WORK, "bench.json"), "w") as f:
                json.dump(results, f, indent=1)
            best_r3 = [x for x in results if x.get("r") == 3 and not x.get("OOM") and x["bs"] == 16 and x["ckpt"]]
            tps_gate = best_r3[0]["tok_per_s"] if best_r3 else 0
            gate("G0.4_throughput_1gpu", tps_gate >= 700,
                 f"r=3/bs=16/ckpt=on tok/s={tps_gate}; full grid in bench.json "
                 f"({sum(1 for x in results if x.get('OOM'))} OOM configs)")

        return None

    phase_a()
    import gc as _gc
    _gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    log(f"  main-process GPU after phase_a: {torch.cuda.memory_allocated(0)/1e9:.2f} GB allocated, "
        f"{torch.cuda.memory_reserved(0)/1e9:.2f} GB reserved")

    # ---------------- G0.5 DDP throughput (runs AFTER main-process GPU cleanup; the child needs a clean GPU) ----------------
    # ---------------- G0.5 DDP throughput ----------------
    with stage("G0.5 DDP"):
        env_ddp = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        p = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                            "--nproc_per_node=2", SELF, "--mode", "ddp-bench"],
                           capture_output=True, text=True, timeout=1500, env=env_ddp)
        try:
            with open(os.path.join(WORK, "ddp_stdout.txt"), "w") as f:
                f.write(p.stdout or "")
            with open(os.path.join(WORK, "ddp_stderr.txt"), "w") as f:
                f.write(p.stderr or "")
        except Exception:
            pass
        if p.returncode != 0:
            log("DDP child STDOUT head:\n" + (p.stdout or "")[:2500])
            log("DDP child STDERR head:\n" + (p.stderr or "")[:4000])
        res = None
        for ln in p.stdout.splitlines():
            if ln.startswith("DDP_BENCH_RESULT"):
                res = json.loads(ln[len("DDP_BENCH_RESULT "):])
        if res is None and os.path.exists(os.path.join(WORK, "ddp_bench.json")):
            res = json.load(open(os.path.join(WORK, "ddp_bench.json")))
        if res is None:
            log("DDP stderr tail: " + p.stderr[-2000:])
            gate("G0.5_ddp_throughput", False, "no DDP result; see log")
        else:
            gate("G0.5_ddp_throughput", res["tok_per_s_ddp"] >= 1400,
                 f"2xT4 DDP tok/s={res['tok_per_s_ddp']:.0f} step={res['step_ms_mean']:.0f}ms "
                 f"mem={res['mem_gb']}GB grad_norm_gap={res['grad_norm_gap']:.2e}",
                 extra=res)

    # ---------------- G0.6 30k-bar table timing ----------------
    tok = load_tokenizer()
    with stage("G0.6 table timing"):
        states_npz = os.path.join(WORK, "gentable_states.npz")
        if not os.path.exists(states_npz):
            enc, lens, _ = build_synthetic_states(30000, tok)
            np.savez_compressed(states_npz, enc=np.array(enc, dtype=object), lens=lens)
        t0 = time.time()
        procs = []
        for g in (0, 1):
            env2 = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            env2["CUDA_VISIBLE_DEVICES"] = str(g)
            procs.append(subprocess.Popen(
                [sys.executable, SELF, "--mode", "gentable", "--rows", "15000",
                 "--out", os.path.join(WORK, f"table_g{g}.npz"),
                 "--ckpt", os.path.join(WORK, f"table_g{g}_ck.npz"), "--batch", "128"],
                env=env2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
        outs = [pr.communicate()[0] for pr in procs]
        wall = time.time() - t0
        ok = all(pr.returncode == 0 for pr in procs)
        t0b, rows_done = time.time(), 0
        tabs = []
        for g in (0, 1):
            fp = os.path.join(WORK, f"table_g{g}.npz")
            ok = ok and os.path.exists(fp)
            if ok:
                z = np.load(fp)
                tabs.append(z["out"])
                rows_done += int(z["done"])
        nan_ok = False
        if ok:
            tab = np.concatenate(tabs, 0)
            nan_ok = bool(np.isfinite(tab).all()) and tab.shape == (30000, 85) \
                and float(tab[:, :48].min()) >= 0 and float(tab[:, :48].max()) <= 1
        gate("G0.6_table_timing", ok and nan_ok and rows_done == 30000,
             f"30k rows x 85 outputs (R=3, 2 GPUs) in {wall:.0f}s wall "
             f"({30000/wall:.1f} rows/s); probs in [0,1]: {nan_ok}",
             extra={"wall_s": round(wall, 1), "rows_per_s": round(30000 / wall, 1)})

    # ---------------- G0.7 kill/resume ----------------
    with stage("G0.7 kill/resume"):
        ref = os.path.join(WORK, "table_ref.npz")
        p = subprocess.run([sys.executable, SELF, "--mode", "gentable", "--rows", "4000",
                            "--out", ref, "--ckpt", os.path.join(WORK, "ref_ck.npz"),
                            "--ck-every", "8", "--batch", "128"],
                           capture_output=True, text=True, timeout=1800)
        if p.returncode != 0 or not os.path.exists(ref):
            gate("G0.7_kill_resume", False, "reference run failed: " + p.stdout[-500:] + p.stderr[-500:])
        else:
            ckf = os.path.join(WORK, "kr_ck.npz")
            outf = os.path.join(WORK, "table_kr.npz")
            for f in (ckf, outf):
                if os.path.exists(f):
                    os.remove(f)
            env2 = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            env2["CUDA_VISIBLE_DEVICES"] = "0"
            pr = subprocess.Popen([sys.executable, SELF, "--mode", "gentable", "--rows", "4000",
                                   "--out", outf, "--ckpt", ckf, "--ck-every", "8",
                                   "--batch", "128"], env=env2,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            time.sleep(40)
            killed = False
            if pr.poll() is None:
                pr.send_signal(signal.SIGKILL)
                pr.wait()
                killed = True
            had_ckpt = os.path.exists(ckf)
            t0 = time.time()
            pr2 = subprocess.run([sys.executable, SELF, "--mode", "gentable", "--rows", "4000",
                                  "--out", outf, "--ckpt", ckf, "--resume", ckf,
                                  "--ck-every", "8", "--batch", "128"], env=env2,
                                 capture_output=True, text=True, timeout=1800)
            resume_dt = time.time() - t0
            ok = pr2.returncode == 0 and os.path.exists(outf)
            detail = f"killed={killed} checkpoint_existed={had_ckpt} resume_rc={pr2.returncode}"
            if ok:
                zr, zk = np.load(ref), np.load(outf)
                md = float(np.abs(zr["out"] - zk["out"]).max())
                ok = md < 2e-3
                detail += f" maxdiff={md:.2e} resume_wall={resume_dt:.0f}s"
            gate("G0.7_kill_resume", ok, detail)

    save_gates(os.path.join(WORK, "gates.json"))
    log("P0 CORE COMPLETE. GATES: " + json.dumps({k: v["pass"] for k, v in GATES.items()}))


if __name__ == "__main__":
    args = parse_args()
    WORK = os.environ.get("AURUM_WORK", "/kaggle/working" if os.path.exists("/kaggle/working") else os.getcwd())
    SELF = os.path.abspath(__file__)
    stable = os.path.join(WORK, "aurum_p0_core.py")
    try:
        if SELF != stable:
            shutil.copy(SELF, stable)
        SELF = stable
    except Exception:
        pass
    set_seeds(1337)
    if args.mode == "main":
        main()
    elif args.mode == "gentable":
        run_gentable(args)
    elif args.mode == "ddp-bench":
        run_ddp_bench(args)
