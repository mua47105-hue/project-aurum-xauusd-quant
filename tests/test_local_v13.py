#!/usr/bin/env python3
"""Local CPU pre-flight tests for the AURUM library (no GPU needed for most).
Run: python test_local.py [quick|model|data]"""
import os, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, ROOT)

FAIL = []
def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)

# ---------------------------------------------------------------- features
def test_features():
    print("== features: scale invariance ==")
    from aurum_feat import synthetic_ohlc, compute_h1_features, scale_ohlc
    h1, m1 = synthetic_ohlc(1500, seed=5)
    h1 = h1.iloc[300:]; m1 = m1.iloc[300 * 60:]
    f0 = compute_h1_features(h1)
    worst = 0.0
    for c in (37.3, 1000.0, 1e-4):
        f1 = compute_h1_features(scale_ohlc(h1, c))
        num = f0.select_dtypes(float)
        d = float((num - f1[num.columns]).abs().max().max())
        worst = max(worst, d)
    check("scale_invariance_features", worst < 1e-6, f"maxdiff={worst:.2e}")
    nan_frac = float(f0.select_dtypes(float).isna().mean().mean())
    check("features_finite_burnin", nan_frac < 0.35, f"nan_frac={nan_frac:.2f}")
    # cross/fred paths
    print("== features: cross-asset & fred merge ==")
    cross = {"xagusd": scale_ohlc(h1, 82.0).rename(columns=str)}
    f2 = compute_h1_features(h1, cross=cross)
    check("cross_merge", "x_xagusd" in f2.columns and f2["x_xagusd"].notna().mean() > 0.8)
    idx = pd.date_range("2015-01-01", "2025-01-01", freq="B")
    fred = {"dxy": pd.Series(np.random.default_rng(0).normal(0, 1, len(idx)), index=idx)}
    f3 = compute_h1_features(h1, fred=fred)
    # h1 index is 2019 synthetic; fred covers it
    check("fred_merge", "m_dxy" in f3.columns and f3["m_dxy"].notna().mean() > 0.7,
          f"notna={f3['m_dxy'].notna().mean():.2f}")

# ---------------------------------------------------------------- labeler
def naive_label(m1_win_hi, m1_win_lo, m1_win_cl, e, a, m, t, lim):
    """Independent brute-force: long entry, returns (cls, R). Tie -> SL first."""
    sl, tp = e - m * a, e + t * a
    for i in range(lim):
        lo, hi = m1_win_lo[i], m1_win_hi[i]
        if lo <= sl:
            return 0, -1.0
        if hi >= tp:
            return 1, t / m
    r = (m1_win_cl[lim - 1] - e) / (m * a)
    return 2, r

def test_labeler():
    print("== labeler: vectorized vs brute force ==")
    from aurum_feat import synthetic_ohlc
    from aurum_label import barrier_actual_outcomes, BARRIER_STOPS, BARRIER_TPS, BARRIER_HORS
    h1, m1 = synthetic_ohlc(700, seed=9)
    i_states = np.arange(320, len(h1) - 60)
    t0 = time.time()
    out = barrier_actual_outcomes(m1, h1, i_states)
    dt = time.time() - t0
    m1 = m1.sort_index()
    m1_hi, m1_lo, m1_cl = m1["h"].values, m1["l"].values, m1["c"].values
    m1_ts = m1.index.asi8
    h1o = h1["o"].values
    rng = np.random.default_rng(0)
    mism = 0; checked = 0
    for trial in range(60):
        bi = int(rng.integers(0, len(i_states)))
        i = i_states[bi]
        e = h1o[i + 1]
        s = np.searchsorted(m1_ts, h1.index.asi8[i + 1])
        H1NS = 3_600_000_000_000
        e_idx = np.searchsorted(m1_ts, h1.index.asi8[i + 1] + 48 * H1NS)
        if e_idx - s < 30:
            continue
        prev_c = np.roll(h1["c"].values, 1); prev_c[0] = h1["c"].values[0]
        tr = np.maximum(h1["h"].values - h1["l"].values,
                        np.maximum(np.abs(h1["h"].values - prev_c), np.abs(h1["l"].values - prev_c)))
        a = pd.Series(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().values[i]
        for mi, m in enumerate(BARRIER_STOPS):
            for ti, t in enumerate(BARRIER_TPS):
                for hi_, h in enumerate(BARRIER_HORS):
                    lim = int(np.searchsorted(m1_ts[s:e_idx], h1.index.asi8[i + 1] + h * H1NS))
                    lim = max(1, min(lim, e_idx - s))
                    cls_n, R_n = naive_label(m1_hi[s:e_idx], m1_lo[s:e_idx], m1_cl[s:e_idx],
                                             e, a, m, t, lim)
                    cell = (0 * 24) + mi * 6 + ti * 2 + hi_  # dir L block layout: m,t,h
                    cls_v = out["cls"][bi, cell]
                    R_v = out["R"][bi, cell]
                    checked += 1
                    if cls_n != cls_v or abs(R_n - R_v) > 1e-9:
                        mism += 1
    check("labeler_matches_bruteforce", mism == 0 and checked > 500,
          f"checked={checked} mismatches={mism} ({dt:.1f}s for {len(i_states)} bars)")
    # scale invariance of labels
    from aurum_feat import scale_ohlc
    out2 = barrier_actual_outcomes(scale_ohlc(m1, 51.7), scale_ohlc(h1, 51.7), i_states)
    dR = float(np.nanmax(np.abs(np.nan_to_num(out["R"], nan=0) - np.nan_to_num(out2["R"], nan=0))))
    dX = float(np.nanmax(np.abs(np.nan_to_num(out["mfe"], nan=0) - np.nan_to_num(out2["mfe"], nan=0))))
    check("labeler_scale_invariance", dR < 1e-9 and dX < 1e-9, f"dR={dR:.2e} dMFE={dX:.2e}")

def test_cohort():
    print("== cohort: leakage-safety + soft labels ==")
    from aurum_feat import synthetic_ohlc, compute_h1_features
    from aurum_label import (barrier_actual_outcomes, expand_actual, build_cohort_index,
                             cohort_soft_labels)
    h1, m1 = synthetic_ohlc(2600, seed=3)
    N = len(h1)
    i_states = np.arange(320, N - 60)
    actual = expand_actual(barrier_actual_outcomes(m1, h1, i_states), i_states, N)
    feat = compute_h1_features(h1)
    cols = ["ret1_atr", "ret4_atr", "vol_z", "rsi_z", "d20", "body1"]
    Fm = feat[cols]
    mu, sd = Fm.iloc[:1000].mean(), Fm.iloc[:1000].std() + 1e-9
    Fs = ((Fm - mu) / sd).values
    lv = np.zeros(N, bool); lv[i_states] = True
    lv &= np.isfinite(Fs).all(1)
    cohort, nvalid = build_cohort_index(Fs, lv, k=200, gap=96)
    # leakage check: every neighbour index <= i - 96
    ok_gap = True
    for i in np.flatnonzero(lv):
        ns = cohort[i][cohort[i] >= 0]
        if len(ns) and ns.max() > i - 96:
            ok_gap = False
            break
    check("cohort_gap_respected", ok_gap)
    mid = N // 2
    check("cohort_fills_midhistory", nvalid[mid] == 200, f"nvalid[mid]={nvalid[mid]}")
    check("cohort_early_sparse", nvalid[350] < 200, f"nvalid[350]={nvalid[350]}")
    soft = cohort_soft_labels(cohort, nvalid, actual)
    pw = soft["p_win"][mid]
    check("soft_labels_range", bool(np.isfinite(pw).all()) and pw.min() >= 0 and pw.max() <= 1,
          f"p_win[mid] in [{pw.min():.2f},{pw.max():.2f}]")
    be = soft["b_eff"][mid]
    check("b_eff_range", bool(np.isfinite(be).mean() > 0.8) and np.nanmedian(be) > 0.1,
          f"median b_eff={np.nanmedian(be):.2f}")

def test_serializer():
    print("== serializer ==")
    from aurum_feat import synthetic_ohlc, compute_h1_features
    from aurum_ser import serialize_state, m1_path_features, h1_path_features, fit_codebook, apply_codebook
    h1, m1 = synthetic_ohlc(900, seed=2)
    feat = compute_h1_features(h1)
    Xm1 = m1_path_features(m1, h1.index)
    Xh1 = h1_path_features(h1)
    km1 = fit_codebook(Xm1[:500], k=16); kmh = fit_codebook(Xh1[:500], k=16)
    feat["m1_shape"] = apply_codebook(km1, Xm1, index=h1.index)
    feat["h1_shape"] = apply_codebook(kmh, Xh1, index=h1.index)
    txt = serialize_state(h1.index[-1], feat.iloc[-1], with_cross=False, with_macro=False, with_priors=False)
    lines = txt.splitlines()
    check("serializer_lines", 7 <= len(lines) <= 11, f"{len(lines)} lines")
    abs_px = f"{h1['c'].iloc[-1]:.2f}"
    check("serializer_no_abs_price", not any(abs_px in ln for ln in lines))
    t2 = serialize_state(h1.index[-1], feat.iloc[-1], with_cross=False, with_macro=False, with_priors=False)
    check("serializer_deterministic", txt == t2)
    print("  sample:\n" + "\n".join("    " + l for l in lines[:4]))

def test_aux():
    print("== aux targets ==")
    from aurum_feat import synthetic_ohlc, wilder_atr
    from aurum_label import aux_targets
    h1, m1 = synthetic_ohlc(1200, seed=4)
    atr = wilder_atr(h1["h"], h1["l"], h1["c"], 14).values
    aux = aux_targets(h1, None, atr)
    ok = True
    bad = []
    for k, v in aux.items():
        arr = np.asarray(v)
        if not np.isfinite(arr[400:-60]).all():
            ok = False; bad.append(k)
    h5 = aux["h5_regime"]
    check("aux_targets_finite", ok, f"non-finite tail-60-excluded: {bad}")
    check("h5_classes", set(np.unique(h5[400:])).issubset(set(range(9))),
          f"classes={sorted(set(np.unique(h5[400:])))}")

def test_data_net():
    print("== data: network (HistData + FRED) ==")
    from aurum_data import histdata_download_year, parse_histdata_zip, fred_fetch
    zp = histdata_download_year("xauusd", 2024, "/tmp/aurum_raw", log=print)
    ok_dl = zp is not None
    df = parse_histdata_zip(zp, log=print) if zp else None
    check("histdata_2024_download", ok_dl and df is not None and len(df) > 300_000,
          f"rows={0 if df is None else len(df)}")
    if df is not None:
        check("histdata_sane", bool((df["h"] >= df[["o", "c"]].max(axis=1) - 1e-9).all()))
    s = fred_fetch("DGS10", log=print)
    check("fred_dgs10", s is not None and len(s) > 2000,
          f"rows={0 if s is None else len(s)}")

def test_model():
    print("== model: LoopedQwen correctness (CPU) ==")
    import torch
    from aurum_model import LoopedQwen, load_base, load_tokenizer, JointHead
    torch.manual_seed(0)
    base = load_base(None)  # CPU fp32
    ids = torch.randint(100, 140000, (2, 48))
    with torch.no_grad():
        stock = base(input_ids=ids).logits
        m = LoopedQwen(base, n_loop=1, sandwich=False)
        d_off = (m(ids)["logits"] - stock).abs().max().item()
        check("r1_equivalence_sandwich_off", d_off < 1e-3, f"maxdiff={d_off:.2e}")
        m_id = LoopedQwen(base, n_loop=1, sandwich=True)
        d_id = (m_id(ids)["logits"] - stock).abs().max().item()
        check("r1_equivalence_identity_init", d_id < 1e-3, f"maxdiff={d_id:.2e}")
        m3 = LoopedQwen(base, n_loop=3, sandwich=True)
        o3 = m3(ids)
        check("r3_finite", bool(torch.isfinite(o3["logits"]).all()))
        check("head_shapes", tuple(o3["head"]["h1"].shape) == (2, 48)
              and tuple(o3["head"]["h5"].shape) == (2, 9))
        pm = torch.ones((2, 48), dtype=torch.bool); pm[0, 30:] = False
        opm = m3(ids, pad_mask=pm)
        check("padmask_finite", bool(torch.isfinite(opm["logits"]).all())
              and bool(torch.isfinite(opm["head"]["h1"]).all()))
    # backward pass with head loss (synthetic targets)
    m3.set_trainable()
    n_tr = m3.trainable_param_count()
    check("trainable_params", 50e6 < n_tr < 80e6, f"{n_tr/1e6:.1f}M")
    ids = torch.randint(100, 140000, (2, 48))
    out = m3(ids)
    lm = m3.lm_loss(out["logits"], ids)
    tgt = {"h1": torch.rand(2, 48), "h2": torch.rand(2, 8),
           "h3": torch.rand(2, 12) * 2, "h4": torch.rand(2, 4),
           "h4c": torch.tensor([0, 2]), "h4q": torch.tensor([1.2, 0.8]),
           "h5": torch.tensor([3, 8])}
    hl = m3.head_loss(out["head"], tgt)
    (lm + hl).backward()
    g = sum(p.grad.abs().sum().item() for p in m3.parameters()
            if p.requires_grad and p.grad is not None)
    check("backward_finite", np.isfinite(g) and g > 0, f"sum|grad|={g:.1f}")
    tok = load_tokenizer()
    from aurum_ser import serialize_state
    from aurum_feat import synthetic_ohlc, compute_h1_features
    h1, m1 = synthetic_ohlc(900, seed=2)
    feat = compute_h1_features(h1)
    txt = serialize_state(h1.index[-1], feat.iloc[-1])
    n_tok = len(tok(txt, add_special_tokens=False)["input_ids"])
    check("token_budget", 240 <= n_tok <= 400, f"{n_tok} tokens/state (target ~312)")

def test_gentable_local():
    print("== gentable mechanics (CPU, tiny) ==")
    import subprocess, sys, json, os, signal, time
    import numpy as np
    out_dir = "/tmp/aurum_gt"
    os.makedirs(out_dir, exist_ok=True)
    from build import build
    bd = build("p0_core")
    env = dict(os.environ, AURUM_WORK=out_dir)
    # tiny rows so CPU finishes fast: rows=48, batch=16
    ref = os.path.join(out_dir, "ref.npz")
    p = subprocess.run([sys.executable, os.path.join(bd, "kernel.py"), "--mode", "gentable",
                        "--rows", "48", "--out", ref, "--ckpt", os.path.join(out_dir, "ck.npz"),
                        "--ck-every", "1", "--batch", "16"], env=env,
                       capture_output=True, text=True, timeout=1800)
    ok = p.returncode == 0 and os.path.exists(ref)
    if not ok:
        print(p.stdout[-2500:]); print(p.stderr[-2500:])
    check("gentable_ref_run", ok)
    if ok:
        z = np.load(ref)
        check("gentable_shape", z["out"].shape == (48, 85) and bool(np.isfinite(z["out"]).all()))
        # resume equivalence: run again with pre-seeded checkpoint of first 32 rows
        ck = os.path.join(out_dir, "ck2.npz")
        np.savez(ck, done=32, out=z["out"][:32])
        out2 = os.path.join(out_dir, "kr.npz")
        p2 = subprocess.run([sys.executable, os.path.join(bd, "kernel.py"), "--mode", "gentable",
                             "--rows", "48", "--out", out2, "--ckpt", ck, "--resume", ck,
                             "--ck-every", "1", "--batch", "16"], env=env,
                            capture_output=True, text=True, timeout=1800)
        if p2.returncode != 0:
            print(p2.stdout[-1500:]); print(p2.stderr[-1500:])
        z2 = np.load(out2) if os.path.exists(out2) else None
        md = float(np.abs(z["out"] - z2["out"]).max()) if z2 is not None else 9e9
        check("gentable_resume_identical", p2.returncode == 0 and md < 2e-3, f"maxdiff={md:.2e}")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    t0 = time.time()
    if mode in ("quick", "all"):
        test_features(); test_labeler(); test_cohort(); test_serializer(); test_aux()
    if mode in ("data", "all"):
        test_data_net()
    if mode in ("model", "all"):
        test_model()
    if mode in ("gentable", "all"):
        test_gentable_local()
    print(f"\n==== {len(FAIL)} failures ====" + ("" if not FAIL else ": " + ", ".join(FAIL)))
    print(f"total {time.time()-t0:.0f}s")
    sys.exit(1 if FAIL else 0)
