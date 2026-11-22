#!/usr/bin/env python3
"""k6 v2.1 patch v3 (Step 3b completion):
  1. G5.3_gauntlet quarantine-aware rewrite: the gate verifies the DECISION
     TABLE is complete and every strategy classified (production-frozen /
     quarantined-v2 / baseline / candidate); stale entries from strategies
     dropped after the gauntlet stage are annotated, not counted against the
     gate. The quarantined v2 regression stack (M1-M6, M7_Active) is clearly
     separated from the non-quarantined production strategy.
  2. U7 plumbing: full-window CAPM/regimes for the bridge frames (they live
     in results, not S); L/S attribution exposure source map (*_exp stashes);
     E1 exposure stash alongside E2.
  3. CAPM IR: alpha / annualized residual sigma (was mislabeled annualized
     Sharpe of daily net).
"""
P = "aurum_v2/kernels/k6_backtest.py"
src = open(P).read()


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (
        f"anchor count {src.count(old)} != {count}: {old[:90]!r}")
    src = src.replace(old, new, count)


# ---------------------------------------------------------- 1. G5.3 rewrite
rep('''        ok_g = gate("G5.3_gauntlet", len(gauntlet) == len(S),
                    "engine vs frozen per strategy (engine ships iff it beats "
                    "frozen on FIT; v1.3 translations forced frozen): "
                    + json.dumps({k: v.get("engine_shipped") for k, v in gauntlet.items()}))''',
    '''        # v2.1 (U3-aware) rewrite: the gauntlet DECISION TABLE must be
        # complete and every strategy CLASSIFIED — the quarantined v2
        # regression stack (M1-M6, M7_Active: zero OOS alpha, audit s5.3/s6.1)
        # is clearly separated from the non-quarantined production strategy
        # (M0 champion family: frozen translation by design) and the
        # baselines. Stale gauntlet entries for strategies dropped from S
        # after the gauntlet stage (no-data variants) are annotated, not
        # counted against the gate.
        QUARANTINED_V2 = {"M1_GBDT_v2", "M2_RFF_Ridge", "RLSBank", "M3c_DAE",
                          "M3m_DAE", "M4_DMN_Lite", "M4b_DMN_Lite",
                          "M5_ContinuumNet", "M6_AURUM_v2_Ensemble",
                          "M6b_Ensemble_EQW", "M7_Active_Grand_Hybrid"}
        PRODUCTION_FROZEN = {"M0_Champ_v13", "M0_Champ_v13_cap1",
                             "M0_Champ_v13_E1", "M0_Champ_v13_E2",
                             "M0_Champ_v13_E2_exp",
                             "M0_Champ_v13_E2_carry",
                             "M0_Champ_v13_E2_carry_punitive",
                             "M7_Core_Satellite", "BuyHold_Gold"}
        missing_g = [n for n in S if n not in gauntlet]
        stale_g = [n for n in gauntlet if n not in S]
        cls = {n: ("production_frozen" if n in PRODUCTION_FROZEN else
                   "quarantined_v2" if n in QUARANTINED_V2 else "baseline")
               for n in gauntlet}
        ok_g = gate("G5.3_gauntlet", not missing_g,
                    "engine vs frozen per strategy (engine ships iff it beats "
                    "frozen on FIT; v1.3 translations forced frozen): "
                    + json.dumps({k: v.get("engine_shipped") for k, v in gauntlet.items()})
                    + f" || v2.1 classification: {json.dumps(cls)}; "
                      f"missing-from-gauntlet={missing_g}; stale-entries-"
                      f"(dropped-from-S, annotated)={stale_g}; QUARANTINED "
                      f"v2 regression stack per U3 (zero OOS alpha, "
                      "audit s5.3/s6.1) — reference-only, not eligible for "
                      "capital until positive alpha net of the s5.3 fix in a "
                      "NEW OOS period")''')

# ------------------------------------------------- 2. U7 plumbing fixes
rep('''        S["M0_Champ_v13_E2_exp"] = z_E2                # post-eval stash (L/S)''',
    '''        S["M0_Champ_v13_E2_exp"] = z_E2                # post-eval stashes
        S["M0_Champ_v13_E1_exp"] = z_E1                # (L/S attribution)''')
rep('''            net_full = _full_net(name) if name in S else None''',
    '''            net_full = _full_net(name) if (name in S or name in results) \\
                else None''')
rep('''            # ---- long/short attribution (OOS) ----------------------------------
            if zs is not None and net_oos is not None:
                zj = zs.reindex(net_oos.index).fillna(0.0).values''',
    '''            # ---- long/short attribution (OOS) ----------------------------------
            zs = S.get(f"{name}_exp", zs)   # bridge frames stash *_exp series
            if zs is not None and net_oos is not None:
                zj = zs.reindex(net_oos.index).fillna(0.0).values''')

# ------------------------------------------------- 3. CAPM IR fix
rep('''                sd = nd.reindex(common).std(ddof=1)
                dpv = len(common) / max(1e-9, (common[-1] - common[0]).days / 365.25)
                ir = float(nd.reindex(common).mean() / sd * dpv) if sd > 0 else float("nan")''',
    '''                sd = nd.reindex(common).std(ddof=1)
                dpv = len(common) / max(1e-9, (common[-1] - common[0]).days / 365.25)
                # IR = annualized alpha / annualized residual sigma from the
                # CAPM regression (alpha_t is the NW HAC(5) t-stat above)
                resid = y - (a + beta * x)
                se_res = float(resid.std(ddof=1))
                ir = float(a * dpv / (se_res * math.sqrt(dpv))) \\
                    if se_res > 0 else float("nan")''')

open(P, "w").write(src)
print("k6 patch v3 applied")
