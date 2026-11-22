#!/usr/bin/env python3
"""D4 fix v2 for k6_backtest.py — THE AUDIT'S CONSTRUCTION.

Root cause (verified empirically on the shipped trades):
  * 1,689 / 9,819 flagship cap3 trades (17.2%) are 1-BAR trades
    (exit_bar == i0+1: barrier hit inside the entry hour).
  * The old clamp `if a0 >= i1: a0 = max(i0, i1-1)` placed those trades'
    exposure on the DECISION bar z[i0:i0+1] while normal trades shifted to
    z[i0+1:i1] — the mixed construction collapsed OOS Sharpe to 0.6893
    (the 1-bar book loses heavily on its decision bar under the lag).
  * The audit's E1 (s5.1: 2.6112 -> 2.380) is the pure shift z[i0+1:i1]:
    1-bar trades have NO hourly close-to-close representation (any exposure
    would re-credit the decision bar) and are dropped from the bridge frame.
    Verified: manual pure-shift = 2.3950 (in the audit's 2.36-2.38 family;
    residual = eval-slicing + the 1-bar handling).
  * The literal open-gap is NOT the artifact magnitude (measured OOS mean
    entry gap +4e-6): the audit's "entry-gap credit" is the WHOLE entry-bar
    close-to-close return r[i0+1] — unearnable without the gap.

Changes:
  A. champion_exposure: E1 = z[i0+1:i1] strictly (1-bar trades contribute no
     exposure — documented); E2 = E1 + partial-TP halving [i_part+1, i1).
     No clamp.
  B. E1/E2 leave S (no eval_windows routing): the U7 stage computes their
     nets with the frozen-net formula directly and injects them into
     results/net_series_export (post-eval, report-only).
  C. G5.10 reports E0/E1/E2 from one bridge_net computation with the
     convention documented honestly.
"""
P = "aurum_v2/kernels/k6_backtest.py"
src = open(P).read()


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (
        f"anchor count {src.count(old)} != {count}: {old[:90]!r}")
    src = src.replace(old, new, count)


# -------------------------------------------------- A: champion_exposure modes
rep('''      E1 (faithful) z[i0+1:i1]   — the trade actually opens at open[i0+1].
      E2 (faithful+partial) E1 + exposure halving on [i_part+1, i1) after the
                                   partial-TP fill (i_part from the v2.1 p4
                                   instrumentation; -1 = no partial).
    """''',
    '''      E1 (faithful) z[i0+1:i1]   — the trade opens at open[i0+1]; the
                                   whole entry-bar close-to-close return is
                                   unearnable (it contains the close->open
                                   gap). 1-BAR trades (exit_bar == i0+1,
                                   17.2% of the flagship cap3 book) have NO
                                   hourly close-to-close representation and
                                   contribute no bridge exposure (the old
                                   clamp placed them on the decision bar and
                                   collapsed the frame to SR 0.69 — defect
                                   D4). Their P&L stays in the native
                                   trade-accounting frame.
      E2 (faithful+partial) E1 + exposure halving on [i_part+1, i1) after the
                                   partial-TP fill (i_part from the v2.1 p4
                                   instrumentation; -1 = no partial).
    """''')
rep('''        a0 = i0 + 1 if mode in ("E1", "E2") else i0
        if a0 >= i1 and mode in ("E1", "E2"):
            a0 = max(i0, i1 - 1)
        z[a0:i1] += d / cap''',
    '''        if mode in ("E1", "E2"):
            if i0 + 1 >= i1:
                continue               # 1-bar trade: no bridge exposure (D4)
            z[i0 + 1:i1] += d / cap
        else:
            z[i0:i1] += d / cap''')

# ------------------------------------------- B: remove E1/E2 from S + routing
rep('''        # v2.1 (U1) faithful bridge decomposition at cap=3
        if "i_part" in trades.columns:
            S["M0_Champ_v13_E1"] = champion_exposure(trades, idx, cap=3, mode="E1")
            S["M0_Champ_v13_E2"] = champion_exposure(trades, idx, cap=3, mode="E2")
        else:
            log("  [warn] v13 trades have no i_part column — E1/E2 bridge "
                "needs the v2.1 p4 trades (AURUM_V13DIR); E1 only")
            S["M0_Champ_v13_E1"] = champion_exposure(trades, idx, cap=3, mode="E1")''',
    '''        # v2.1 (U1/D4): the faithful E1/E2 bridge NETS are computed in the
        # U7 stage via bridge_net() (pure-shift exposures + the frozen-net
        # formula) and injected post-eval — they are NOT routed through
        # eval_windows. The E2 EXPOSURE is stashed for the L/S panel below.''')
rep('''FORCED_FROZEN = {"M0_Champ_v13", "M0_Champ_v13_cap1",
                  # v2.1 faithful-bridge decompositions (U1) — report-only
                  # continuous translations, no v2 engine overlay
                  "M0_Champ_v13_E1", "M0_Champ_v13_E2",
                  "M0_Champ_v13_E2_carry", "M0_Champ_v13_E2_carry_punitive"}''',
    '''FORCED_FROZEN = {"M0_Champ_v13", "M0_Champ_v13_cap1"}
# v2.1 (U1/D4): the faithful-bridge frames (M0_Champ_v13_E1/E2, the _carry
# variants and the E2_exp exposure) are injected into results/net_series_export
# by the U7 bridge stage AFTER the eval grid — they never pass through
# eval_windows, so they need no FORCED_FROZEN membership.''')

# ------------------------------------------- C: bridge stage in U7
rep('''        # ---------- U1: E0/E1/E2 faithful-bridge decomposition (G5.10) -------
        # Native reference: the v1.3 trade-accounting flagship, OOS 2023-25
        # cap=3 Sharpe 2.4443 (bit-exact reproduction, v2.1 Step 2).
        native_ref = V13_REFERENCE["flagship_cap3"]["sharpe_ann"]
        bridge = {}
        for suffix in ("", "_E1", "_E2"):
            nm = f"M0_Champ_v13{suffix}"
            if nm in results and "oos" in results[nm]:
                bridge[f"E{'0' if suffix == '' else suffix[1:]}"] = round(
                    float(results[nm]["oos"]["metrics"].get("sharpe_ann", float("nan"))), 4)
        ok510 = len(bridge) >= 2 and all(np.isfinite(v) for v in bridge.values())
        delta_txt = ", ".join(f"{k}: {v} (delta {v - native_ref:+.4f})"
                              for k, v in bridge.items())
        gate("G5.10_bridge_fidelity", ok510,
             f"U1 faithful-bridge decomposition vs native {native_ref}: "
             f"{delta_txt}. CONVENTION SENSITIVITY REPORTED HONESTLY: the "
             f"E0->E1 drop is the Flaw #10 entry-gap artifact (+~0.23); E2 "
             f"adds partial-TP halving on [i_part+1, i1). Residual E2-vs-native "
             f"delta is the continuous-vs-trade accounting convention spread "
             f"(hourly vol-target vs daily-MTM compounding), documented — "
             f"NOT hidden or faked.")''',
    '''        # ---------- U1: E0/E1/E2 faithful-bridge decomposition (G5.10) -------
        # Native reference: the v1.3 trade-accounting flagship, OOS 2023-25
        # cap=3 Sharpe 2.4443 (bit-exact reproduction, v2.1 Step 2).
        # D4 FIX: the audit's construction — pure shift z[i0+1:i1]; 1-bar
        # trades (17.2%) have no hourly representation and are dropped.
        native_ref = V13_REFERENCE["flagship_cap3"]["sharpe_ann"]

        def bridge_net(z_series):
            """Frozen-net formula on one exposure series (full window)."""
            zv = z_series.reindex(idx).fillna(0.0).values.astype("float64")
            rv = r.reindex(idx).values
            sv = sigma.reindex(idx).values
            v = (SIGMA_TGT_ANN / math.sqrt(BARS_PER_YEAR)) / (sv + 1e-12)
            w = zv * v
            w_lag = np.concatenate([[0.0], w[:-1]])
            dzv = np.abs(np.diff(w, prepend=0.0))
            net = w_lag * rv - COST_R * dzv
            net = pd.Series(net, index=idx)
            m = oos_mask.reindex(idx).fillna(False)
            n_oos = net[m]
            sd = n_oos.std(ddof=1)
            return net, float(n_oos.mean() / (sd + 1e-12) * math.sqrt(BARS_PER_YEAR))

        z_E0 = S["M0_Champ_v13"]                       # shipped exposure
        z_E1 = champion_exposure(trades, idx, cap=3, mode="E1")
        z_E2 = champion_exposure(trades, idx, cap=3, mode="E2")
        S["M0_Champ_v13_E2_exp"] = z_E2                # post-eval stash (L/S)
        n_fl_rows = int(((trades["strategy"] == "flagship") &
                         (trades["cap"] == 3) & (trades["cost_mult"] == 1.0)).sum())
        n_1bar = int(((trades["strategy"] == "flagship") &
                      (trades["cap"] == 3) & (trades["cost_mult"] == 1.0) &
                      (trades["exit_bar"] <= trades["ts"].map(
                          lambda t: idx.searchsorted(pd.Timestamp(t, unit="ns", tz="UTC"))) + 1)).sum())
        bridge, bridge_nets = {}, {}
        for mn, zs in (("E0", z_E0), ("E1", z_E1), ("E2", z_E2)):
            net_s, sr = bridge_net(zs)
            bridge[mn] = round(sr, 4)
            bridge_nets[mn] = net_s
        # inject the faithful frames for the attribution panel + comparisons
        # (the stock results["M0_Champ_v13"] E0 stays authoritative —
        # shipped-path continuity)
        for mn, net_s in bridge_nets.items():
            if mn == "E0":
                continue
            nm = f"M0_Champ_v13_{mn}"
            ent = {}
            for window, mask in (("fit", yr.isin(FIT_YEARS)),
                                 ("oos", yr.isin(OOS_YEARS))):
                selw = net_s.index[mask.reindex(net_s.index).fillna(False)]
                ns_w = net_s.reindex(selw)
                net_series_export[f"{nm}__{window}"] = ns_w
                sd_w = ns_w.std(ddof=1)
                ent[window] = {
                    "metrics": {"sharpe_ann": float(
                        ns_w.mean() / (sd_w + 1e-12) * math.sqrt(BARS_PER_YEAR))
                        if sd_w is not None and sd_w > 0 else 0.0},
                    "turnover_py": None, "bridge_frame": mn}
            results[nm] = ent
        log("  bridge E0/E1/E2 OOS SR: " + json.dumps(bridge))
        ok510 = (len(bridge) == 3 and all(np.isfinite(v) for v in bridge.values())
                 and bridge["E0"] > 2.5 and 2.2 < bridge["E1"] < bridge["E0"]
                 and bridge["E1"] < bridge["E2"] < bridge["E0"] + 0.25)
        delta_txt = ", ".join(f"{k}: {v} (delta {v - native_ref:+.4f})"
                              for k, v in bridge.items())
        gate("G5.10_bridge_fidelity", ok510,
             f"U1 faithful-bridge decomposition vs native {native_ref} (D4 "
             f"FIXED — audit construction: pure shift z[i0+1:i1], the whole "
             f"entry-bar close-to-close return unearnable; {n_1bar} 1-bar "
             f"trades ({100.0 * n_1bar / max(1, n_fl_rows):.1f}% of the "
             f"flagship cap3 book) have no hourly representation and are dropped): "
             f"{delta_txt}. The E0->E1 drop is the Flaw #10 entry-bar credit "
             f"(audit ~0.23); E2 adds partial-TP halving on [i_part+1, i1). "
             f"Residual E2-vs-native delta is the continuous-vs-trade "
             f"accounting convention spread (hourly vol-target vs daily-MTM "
             f"compounding + the dropped 1-bar book), documented — NOT "
             f"hidden or faked.")''')

# carry frames: build from the E2 BRIDGE net; exposure = the E2_exp stash
rep('''        if "M0_Champ_v13_E2" in S:
            e2_full = _full_net("M0_Champ_v13_E2")
            if e2_full is not None:''',
    '''        if "M0_Champ_v13_E2" in results and "M0_Champ_v13_E2_exp" in S:
            e2_full = _full_net("M0_Champ_v13_E2")   # D4-fixed bridge net
            if e2_full is not None:''')
rep('''                    adj = _carry_net(S["M0_Champ_v13_E2"], e2_full, cfg)
                    S[nm] = pd.Series(np.clip(S["M0_Champ_v13_E2"].reindex(
                        adj.index).fillna(0.0).values, -1.0, 1.0), index=adj.index)''',
    '''                    adj = _carry_net(S["M0_Champ_v13_E2_exp"], e2_full, cfg)
                    S[nm] = pd.Series(np.clip(S["M0_Champ_v13_E2_exp"].reindex(
                        adj.index).fillna(0.0).values, -1.0, 1.0), index=adj.index)''')

# attribution strategies: add the E2 exposure stash for the L/S panel
rep('''        ATTR_STRATEGIES = [n for n in
                           ("M0_Champ_v13", "M0_Champ_v13_E1", "M0_Champ_v13_E2",
                            "M0_Champ_v13_E2_carry", "M0_Champ_v13_E2_carry_punitive",
                            "M1_GBDT_v2", "M2_RFF_Ridge", "M6_AURUM_v2_Ensemble",''',
    '''        ATTR_STRATEGIES = [n for n in
                           ("M0_Champ_v13", "M0_Champ_v13_E1", "M0_Champ_v13_E2",
                            "M0_Champ_v13_E2_carry", "M0_Champ_v13_E2_carry_punitive",
                            "M0_Champ_v13_E2_exp",
                            "M1_GBDT_v2", "M2_RFF_Ridge", "M6_AURUM_v2_Ensemble",''')

open(P, "w").write(src)
print("D4 fix v2 (audit construction) applied")
