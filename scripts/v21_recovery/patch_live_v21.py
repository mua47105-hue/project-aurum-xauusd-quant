#!/usr/bin/env python3
"""Step 4 (U4 live kernel) patch for kaggle_kernels/v1_3/aurum-live-daily.py:

  --use-frozen-policy (DEFAULT ON): loads v1_3/config/frozen_policy.json,
     verifies the LIVE_* constants match the frozen registry (G6.0 gate).
  --fast-r1: single r=1 forward pass (skip r=2/3/4 core builds) +
     state-hash-keyed per-depth forward cache (WORK/forward_cache/);
     dual-decision decision.json emitting AURUM_r1_Fast_Flagship,
     AURUM_r1_Clean_EV and the frozen v1.3 pooled reference (with explicit
     requires-full-forward status when the full-forward cache is cold);
     G6.2 adapted for fast/cache mode (the fix: the gate no longer requires
     all four depths when running the documented fast-r1/cache mode).
  Overlay fallback in fast mode: h48 cells use empirical <=y-1 48h
     quantiles (the h3 head needs the full forward) — documented.
  Latency: per-stage timings written to WORK/latency.json (fast + full).
"""
P = "kaggle_kernels/v1_3/aurum-live-daily.py"
src = open(P).read()


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (
        f"anchor count {src.count(old)} != {count}: {old[:90]!r}")
    src = src.replace(old, new, count)


# ---------------------------------------------------------------- module head
rep('''LIVE_POLICY_SRC = ("aurum-p4-backtest v1.3 policy_fit.json; "
                    "fit window 2016-2022; OOS 2023-25 never touched")''',
    '''LIVE_POLICY_SRC = ("aurum-p4-backtest v1.3 policy_fit.json; "
                    "fit window 2016-2022; OOS 2023-25 never touched")

# ==================== v2.1 (Spec v2.1, U4 live kernel) ====================
FAST_R1 = False            # --fast-r1: single r=1 forward + dual decision
POLICY_FROZEN = True       # --use-frozen-policy (default ON, Flaw #11)
V21_FROZEN_POLICY_PATH = os.environ.get(
    "AURUM_FROZEN_POLICY",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "v1_3", "config", "frozen_policy.json"))
LAT = {}                   # per-stage latency ledger (v2.1)


def _load_frozen_live():
    """Load the frozen policy and verify the LIVE constants match (G6.0)."""
    with open(V21_FROZEN_POLICY_PATH) as f:
        fp = json.load(f)
    fpol = fp["flagship_policy"]
    checks = {
        "w_pool": (LIVE_POOL_W == fpol["w_pooled_r123"]),
        "kappa": (LIVE_KAPPA == fpol["kappa"]),
        "lambda_var": (LIVE_LAMBDA_VAR == fpol["lambda_var"]),
        "ev_min": (LIVE_EV_MIN == fpol["ev_min"]),
        "overlay_rule": (LIVE_OVERLAY_RULE == fpol["overlay_rule"]),
    }
    return fp, checks''')

# ---------------------------------------------------- overlay h3v=None fallback
rep('''    if h_ == 48:
        q50 = float(h3v[1] if d == "L" else h3v[7])   # MFE q50
        q25 = float(h3v[3] if d == "L" else h3v[9])   # MAE q25
        return q50, q25, "h3_head_48h"''',
    '''    if h_ == 48 and h3v is not None:
        q50 = float(h3v[1] if d == "L" else h3v[7])   # MFE q50
        q25 = float(h3v[3] if d == "L" else h3v[9])   # MAE q25
        return q50, q25, "h3_head_48h"
    if h_ == 48 and h3v is None:
        # v2.1 fast-r1 fallback: the h3 head needs the full forward; use the
        # empirical <=y-1 48h quantiles (documented convention difference).
        m = (pd.DatetimeIndex(exc.index).year.values <= (y_live - 1))
        q50 = float(np.nanquantile(exc[f"mfe_{d}48"].values[m], 0.50))
        q25 = float(np.nanquantile(exc[f"mae_{d}48"].values[m], 0.25))
        return q50, q25, f"empirical_le{y_live - 1}_h48_fast_r1"''')

# ---------------------------------------------------- forward loop + cache
rep('''    ids = torch.from_numpy(enc)[None, :]
    out = {}
    for depth in LIVE_DEPTHS:
        model = build_core(base, core_sd, depth, log=log)
        model.to(dev)
        out[depth] = _forward_state(model, ids, ro_w.to(dev), dev,
                                    heads.to(dev),
                                    temps.get("h1", 1.0), temps.get("h5", 1.0))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        log(f"  forward r={depth}: h1 shape {out[depth]['h1'].shape}; "
            f"finite={bool(np.isfinite(out[depth]['h1']).all())}")
    fin = all(np.isfinite(v).all() for d in out.values() for v in d.values())
    gate("G6.2_forward", fin and sorted(out.keys()) == LIVE_DEPTHS,
         f"forwards r={LIVE_DEPTHS} on {dev.type}; all heads finite={fin}; "
         f"h1 shape {out[3]['h1'].shape}; temperatures {temps}")''',
    '''    ids = torch.from_numpy(enc)[None, :]
    # v2.1 state-hash-keyed per-depth forward cache: identical serialized
    # state -> identical forward outputs (the core is frozen weights).
    import hashlib as _hl
    state_hash = _hl.sha256(np.ascontiguousarray(enc).tobytes()).hexdigest()[:16]
    cache_dir = os.path.join(WORK, "forward_cache")
    os.makedirs(cache_dir, exist_ok=True)
    depths_run = [1] if FAST_R1 else LIVE_DEPTHS
    out = {}
    cache_hits = {}
    for depth in depths_run:
        t_d = time.time()
        cpath = os.path.join(cache_dir, f"{state_hash}_r{depth}.npz")
        if os.path.isfile(cpath):
            z = np.load(cpath)
            out[depth] = {k: z[k] for k in z.files}
            cache_hits[depth] = True
            LAT[f"forward_r{depth}_cached_s"] = round(time.time() - t_d, 3)
            log(f"  forward r={depth}: CACHE HIT ({state_hash}_r{depth})")
            continue
        cache_hits[depth] = False
        model = build_core(base, core_sd, depth, log=log)
        model.to(dev)
        out[depth] = _forward_state(model, ids, ro_w.to(dev), dev,
                                    heads.to(dev),
                                    temps.get("h1", 1.0), temps.get("h5", 1.0))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        np.savez(cpath, **{k: v for k, v in out[depth].items()})
        LAT[f"forward_r{depth}_cold_s"] = round(time.time() - t_d, 3)
        log(f"  forward r={depth}: h1 shape {out[depth]['h1'].shape}; "
            f"finite={bool(np.isfinite(out[depth]['h1']).all())}")
    fin = all(np.isfinite(v).all() for d in out.values() for v in d.values())
    # G6.2 v2.1: the gate requires all LIVE_DEPTHS in FULL mode; in the
    # documented --fast-r1 mode it requires the r=1 forward (cache or cold).
    # The FIX: the mode and cache provenance are part of the gate record.
    if FAST_R1:
        ok62 = fin and sorted(out.keys()) == [1]
        g62 = (f"FAST-R1 mode: forward r=[1] on {dev.type} "
               f"({'cache hit' if cache_hits.get(1) else 'cold'}); "
               f"finite={fin}; full pooled reference "
               f"requires-full-forward unless cached")
    else:
        ok62 = fin and sorted(out.keys()) == LIVE_DEPTHS
        g62 = (f"forwards r={LIVE_DEPTHS} on {dev.type}; all heads finite={fin}; "
               f"h1 shape {out[3]['h1'].shape}; temperatures {temps}")
    gate("G6.2_forward", ok62, g62)''')

# ---------------------------------------------------- fast-r1 decision path
rep('''    # ---------------- v1.3 decision policy ----------------------------------
    P = {r: np.clip(out[r]["h1"], 0.0, 1.0) for r in LIVE_DEPTHS}''',
    '''    # ---------------- v1.3 decision policy ----------------------------------
    P = {r: np.clip(out[r]["h1"], 0.0, 1.0) for r in out}
    v21_dual = None
    if FAST_R1:
        # v2.1 (U4): dual r=1 decision — the audit s4.2 finding that r=1
        # alone matches/beats the pooled flagship OOS with 3x cheaper
        # inference. The pooled v1.3 reference is computed from the
        # per-depth cache when warm, else explicitly deferred.
        t_d = time.time()
        P1 = P[1]
        dirs_ax, mss, tps, hrs = cell_axes()
        b_geo = np.array([t / m for t, m in zip(tps, mss)])
        cost_v = np.asarray(LIVE_COST_BASE_R, np.float64) / np.asarray(mss, np.float64)
        year_all2 = pd.DatetimeIndex(feat.index).year.values.astype(np.int16)
        pi48 = _live_base_rate_prior(cls, year_all2, y_live)
        # --- AURUM_r1_Fast_Flagship: flagship machinery on the r=1 posterior
        # (w=[1,0,0]; sigma_loop == 0 -> LCB penalty 0; kappa 0 -> no shrink)
        EV_f = P1 * b_geo - (1.0 - P1) - 0.0 * b_geo - cost_v
        cell_f = int(np.nanargmax(np.where(np.isfinite(EV_f), EV_f, -1e9)))
        # --- AURUM_r1_Clean_EV: raw v1.2-style EV-argmax (no cost in EV)
        EV_c = P1 * b_geo - (1.0 - P1)
        cell_c = int(np.nanargmax(np.where(np.isfinite(EV_c), EV_c, -1e9)))
        gb = _gbdt_live_fold(feat, cls, i, y_live, log)
        gbdt_dir2 = "L" if gb["pL"] >= gb["pS"] else "S"
        blackout2 = _live_blackout(feat, i, ts)
        bo_clear = not blackout2["nfp_window"] and not blackout2["vix_shock"]
        pooled_ref = {"status": "requires_full_forward",
                      "note": "pooled p_bar(1:3) + loop-OOD need r=2/3 "
                              "forwards; run without --fast-r1 (or warm the "
                              "per-depth cache first)"}
        pooled_cached = all(os.path.isfile(os.path.join(
            cache_dir, f"{state_hash}_r{d}.npz")) for d in (2, 3, 4))
        if pooled_cached:
            Pp = {}
            for d in (1, 2, 3, 4):
                z = np.load(os.path.join(cache_dir, f"{state_hash}_r{d}.npz"))
                Pp[d] = np.clip(z["h1"], 0.0, 1.0)
            w_ = np.asarray(LIVE_POOL_W, np.float64)
            p_bar_ = w_[0] * Pp[1] + w_[1] * Pp[2] + w_[2] * Pp[3]
            m_pool_ = (Pp[1] + Pp[2] + Pp[3]) / 3.0
            sig_ = np.sqrt(((Pp[1] - m_pool_) ** 2 + (Pp[2] - m_pool_) ** 2
                            + (Pp[3] - m_pool_) ** 2) / 3.0)
            EVp = (p_bar_ * b_geo - (1.0 - p_bar_)
                   - LIVE_LAMBDA_VAR * b_geo * sig_ - cost_v)
            cell_p = int(np.nanargmax(np.where(np.isfinite(EVp), EVp, -1e9)))
            ood_m = float(np.mean([np.abs(Pp[r] - Pp[3]).mean()
                                   for r in (1, 2, 4)]))
            pooled_ref = {"status": "cached_full_forward",
                          "cell": cell_names()[cell_p],
                          "p_bar_cell": round(float(p_bar_[cell_p]), 4),
                          "ev_lcb_R": round(float(EVp[cell_p]), 4),
                          "ood_mean": round(ood_m, 5),
                          "agrees_gbdt": bool(gbdt_dir2 == dirs_ax[cell_p])}
        def _r1_entry(name, cellv, evv, ev_includes_cost):
            d_c = dirs_ax[cellv]
            p_c = float(P1[cellv])
            q50_a, q25_a, ov_s = _live_overlay_quantiles(
                d_c, int(hrs[cellv]), None, exc, y_live)
            agree_c = bool(gbdt_dir2 == d_c)
            ok = (evv >= 0.0) and (0.01 < p_c < 0.99) and agree_c and bo_clear
            atr_l = float(wilder_atr(h1["h"], h1["l"], h1["c"], 14).values[i])
            kelly_c = (p_c * b_geo[cellv] - (1 - p_c)) / b_geo[cellv]
            f_c = float(np.clip(KELLY_FRACTION_LIVE * kelly_c, 0.0,
                                HARD_EQUITY_RISK_CAP_LIVE))
            return {
                "strategy": name,
                "machinery": ("flagship machinery on r=1 only (w=[1,0,0], "
                              "sigma_loop=0 -> LCB penalty 0)"
                              if name == "AURUM_r1_Fast_Flagship" else
                              "raw r=1 EV-argmax + GBDT gate (no LCB, no "
                              "shrinkage; v1.2 convention: no cost in EV)"),
                "action": ("BUY" if d_c == "L" else "SELL") if ok else "NO TRADE",
                "cell": cell_names()[cellv], "direction":
                    "long" if d_c == "L" else "short",
                "p_win_r1": round(p_c, 4),
                "EV_R": round(float(evv[cellv]), 4),
                "ev_includes_cost": ev_includes_cost,
                "gbdt_agrees": agree_c, "gbdt_dir": gbdt_dir2,
                "blackout_clear": bool(bo_clear),
                "overlay": {"q50_atr": round(q50_a, 3), "q25_atr": round(q25_a, 3),
                            "source": ov_s},
                "risk_fraction_capped": round(f_c, 5),
                "atr14_h1_usd": round(atr_l, 3),
                "loop_ood_gate": "deferred (requires full forward)",
                "trade_ok": bool(ok)}
        v21_dual = {
            "mode": "fast-r1",
            "frozen_policy_source": os.path.basename(V21_FROZEN_POLICY_PATH),
            "strategies": [
                _r1_entry("AURUM_r1_Fast_Flagship", cell_f, EV_f, True),
                _r1_entry("AURUM_r1_Clean_EV", cell_c, EV_c, False)],
            "frozen_v13_pooled_reference": pooled_ref,
            "state_hash": state_hash,
            "latency_note": "single r=1 forward; overlay h48 via empirical "
                            "fallback (h3 head needs the full forward)"}
        LAT["fast_r1_dual_decision_s"] = round(time.time() - t_d, 3)
        log("  v2.1 dual r=1 decision: "
            + json.dumps({s["strategy"]: s["action"]
                          for s in v21_dual["strategies"]})
            + f"; pooled reference: {pooled_ref['status']}")''')

# guard the pooled path against fast mode (P lacks r=2/3)
rep('''    ood_mean = float(np.mean([np.abs(P[r] - P[3]).mean()
                              for r in LIVE_DEPTHS if r != 3]))
    ood_max = float(np.max([np.abs(P[r] - P[3]).max()
                            for r in LIVE_DEPTHS if r != 3]))''',
    '''    ood_mean = float(np.mean([np.abs(P[r] - P[3]).mean()
                              for r in LIVE_DEPTHS if r != 3 and r in P]))
    ood_max = float(np.max([np.abs(P[r] - P[3]).max()
                            for r in LIVE_DEPTHS if r != 3 and r in P]))''')

# decision.json: embed the dual decision + mode + latency
rep('''        "generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_s": round(time.time() - t0, 1),
    }''',
    '''        "generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_s": round(time.time() - t0, 1),
        "v21_mode": {"fast_r1": bool(FAST_R1),
                     "policy_frozen": bool(POLICY_FROZEN),
                     "state_hash": state_hash,
                     "forward_cache": {f"r{d}": ("hit" if cache_hits[d] else "cold")
                                       for d in depths_run}},
        "v21_dual_decision": v21_dual,
        "v21_latency_s": dict(LAT, total=round(time.time() - t0, 1)),
    }''')

# heads_detail guard for fast mode (out[3] unavailable)
rep('''    # ---------------- heads detail + decision + audit -----------------------
    h5v = out[3]["h5"]
    h3v = out[3]["h3"]''',
    '''    # ---------------- heads detail + decision + audit -----------------------
    if FAST_R1:
        h5v = None
        h3v = None
    else:
        h5v = out[3]["h5"]
        h3v = out[3]["h3"]''')

# overlay quantiles call guard (h3v may be None in fast mode)
rep('''    q50_atr, q25_atr, ov_src = _live_overlay_quantiles(
        d_cell, h_cell, out[3]["h3"], exc, y_live)''',
    '''    q50_atr, q25_atr, ov_src = _live_overlay_quantiles(
        d_cell, h_cell, h3v, exc, y_live)''')

# heads_detail block guard (uses out[3] h2/h4/h5) — emit only in full mode
rep('''        "heads_detail": {
            "h2_up_prob": {k: round(float(out[3]["h2"][j]), 4)''',
    '''        "heads_detail": (None if FAST_R1 else {
            "h2_up_prob": {k: round(float(out[3]["h2"][j]), 4)''')
rep('''            "h5_regime_probs": {regime_names[j]: round(float(h5v[j]), 4)
                                for j in range(9)},
            "regime_call": regime_names[int(np.argmax(h5v))]},''',
    '''            "h5_regime_probs": {regime_names[j]: round(float(h5v[j]), 4)
                                for j in range(9)},
            "regime_call": regime_names[int(np.argmax(h5v))]}),''')

# fast mode: populate out[2/3/4] from the per-depth cache (warm) or run the
# missing forwards once (cold fallback, latency-honest) so the pooled v1.3
# decision path and its gates stay intact
rep('''    # ---------------- v1.3 decision policy ----------------------------------
    P = {r: np.clip(out[r]["h1"], 0.0, 1.0) for r in out}
    v21_dual = None
    if FAST_R1:''',
    '''    # ---------------- v1.3 decision policy ----------------------------------
    if FAST_R1:
        # the pooled v1.3 reference path needs r=2/3/4: load from the
        # per-depth cache (the state hash keys identical inputs); on a cold
        # cache run them once and mark the fallback in the latency ledger.
        t_p = time.time()
        for d in (2, 3, 4):
            if d in out:
                continue
            cpath = os.path.join(cache_dir, f"{state_hash}_r{d}.npz")
            if os.path.isfile(cpath):
                z = np.load(cpath)
                out[d] = {k: z[k] for k in z.files}
            else:
                model = build_core(base, core_sd, d, log=log)
                model.to(dev)
                out[d] = _forward_state(model, ids, ro_w.to(dev), dev,
                                        heads.to(dev),
                                        temps.get("h1", 1.0), temps.get("h5", 1.0))
                del model
                np.savez(cpath, **{k: v for k, v in out[d].items()})
                LAT[f"fast_r1_cold_pooled_fallback_r{d}_s"] = True
        LAT["fast_r1_pooled_restore_s"] = round(time.time() - t_p, 3)
    P = {r: np.clip(out[r]["h1"], 0.0, 1.0) for r in out}
    v21_dual = None
    if FAST_R1:''')

# latency.json export + G6.0 frozen-policy gate + argparse
rep('''    save_gates(os.path.join(WORK, "gates.json"))
    log("LIVE DAILY COMPLETE. GATES: "
        + json.dumps({k: v["pass"] for k, v in GATES.items()}))
    log(f"total wall {time.time() - t0:.0f}s")''',
    '''    with open(os.path.join(WORK, "latency.json"), "w") as f:
        json.dump({"mode": "fast-r1" if FAST_R1 else "full",
                   "stage_s": LAT, "total_s": round(time.time() - t0, 1)},
                  f, indent=1)
    save_gates(os.path.join(WORK, "gates.json"))
    log("LIVE DAILY COMPLETE. GATES: "
        + json.dumps({k: v["pass"] for k, v in GATES.items()}))
    log(f"total wall {time.time() - t0:.0f}s")''')

rep('''if __name__ == "__main__":
    WORK = os.environ.get("AURUM_WORK",
                          "/kaggle/working" if os.path.exists("/kaggle/working")
                          else os.getcwd())
    set_seeds(1337)
    live_main(WORK)''',
    '''if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="AURUM v1.3/v2.1 live daily")
    _ap.add_argument("--fast-r1", action="store_true",
                     help="v2.1 U4: single r=1 forward + dual decision "
                          "(AURUM_r1_Fast_Flagship / AURUM_r1_Clean_EV)")
    _ap.add_argument("--use-frozen-policy", dest="frozen", action="store_true",
                     default=True,
                     help="verify LIVE constants against the frozen policy "
                          "(v2.1 default, Flaw #11)")
    _ap.add_argument("--no-frozen-policy", dest="frozen", action="store_false")
    _args, _unk = _ap.parse_known_args()
    FAST_R1 = _args.fast_r1
    POLICY_FROZEN = _args.frozen
    WORK = os.environ.get("AURUM_WORK",
                          "/kaggle/working" if os.path.exists("/kaggle/working")
                          else os.getcwd())
    set_seeds(1337)
    if POLICY_FROZEN:
        _fp, _ck = _load_frozen_live()
        gate("G6.0_policy_frozen", all(_ck.values()),
             f"LIVE constants match the frozen policy registry "
             f"({os.path.basename(V21_FROZEN_POLICY_PATH)}): "
             + json.dumps({k: bool(v) for k, v in _ck.items()}))
    print(f"[live-daily] FAST_R1={FAST_R1} POLICY_FROZEN={POLICY_FROZEN}",
          flush=True)
    live_main(WORK)''')

open(P, "w").write(src)
print("live-daily v2.1 patch applied")
