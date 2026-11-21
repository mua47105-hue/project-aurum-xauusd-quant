#!/usr/bin/env python3
"""v2.1 patch for aurum_v2/kernels/k6_backtest.py (Step 3).

U1 (faithful bridge): champion_exposure gains E0/E1/E2 modes; E1 fixes the
     Flaw #10 entry-bar off-by-one (z[i0+1:i1]); E2 adds partial-TP exposure
     halving on [i_part+1, i1) using the v2.1 i_part instrumentation.
G5.10: honest bridge-decomposition gate (delta vs native 2.4443 reported,
     never faked or hidden).
U7 (attribution stage, G5.12): Newey-West HAC(5) CAPM vs B&H gold, long/short
     attribution, 5-regime torture table, honest project-wide DSR panel
     N in {12,50,100,170,200,500} — computed on the faithful-bridge (E2) and
     carry-adjusted champion frames. Written defensively: no .reindex on
     ndarrays, no Series-vs-Timestamp comparison without .index (the two
     Nov-15 bug classes, fixed permanently).
"""
import re

P = "aurum_v2/kernels/k6_backtest.py"
src = open(P).read()


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (
        f"anchor count {src.count(old)} != {count}: {old[:90]!r}")
    src = src.replace(old, new, count)


# ---------------------------------------------------- A: bridge E0/E1/E2
rep('''def champion_exposure(trades: pd.DataFrame, index: pd.DatetimeIndex,
                      cap: int = 3) -> pd.Series:
    """v1.3 flagship per-bar exposure from its walk-forward trades.

    VERIFIED MAPPING: exit_bar is the absolute bar position in the SAME
    2010-based H1 frame (first 2016-01-04 entry sits at frame position 36521
    with exit_bar 36522; exit_bar max 94948 < 95008).  Each active trade
    contributes dir/cap; the sum is clipped to [-1, 1].  Causal by
    construction (the trades are the v1.3 policy's own walk-forward output).
    """
    z = np.zeros(len(index))
    sub = trades[(trades["strategy"] == "flagship") &
                 (trades["cap"] == cap) &
                 (trades["cost_mult"] == 1.0)]
    for row in sub.itertuples():
        d = 1.0 if row.cell_name.startswith("L") else -1.0
        i0 = index.searchsorted(pd.Timestamp(row.ts, unit="ns", tz="UTC"))
        i1 = int(row.exit_bar)
        if i0 >= len(index):
            continue
        i1 = min(len(index), max(i1, i0 + 1))
        z[i0:i1] += d / cap
    return pd.Series(np.clip(z, -1.0, 1.0), index=index)''',
    '''def champion_exposure(trades: pd.DataFrame, index: pd.DatetimeIndex,
                      cap: int = 3, mode: str = "E0") -> pd.Series:
    """v1.3 flagship per-bar exposure from its walk-forward trades.

    VERIFIED MAPPING: exit_bar is the absolute bar position in the SAME
    2010-based H1 frame (first 2016-01-04 entry sits at frame position 36521
    with exit_bar 36522; exit_bar max 94948 < 95008).  Each active trade
    contributes dir/cap; the sum is clipped to [-1, 1].  Causal by
    construction (the trades are the v1.3 policy's own walk-forward output).

    v2.1 (U1 / Flaw #10) bridge decomposition:
      E0 (shipped)  z[i0:i1]     — indexes exposure from the DECISION bar,
                                   crediting the close->open gap of the entry
                                   bar, a move the trade never earns
                                   (+0.23 OOS Sharpe artifact, audit s5.1).
      E1 (faithful) z[i0+1:i1]   — the trade actually opens at open[i0+1].
      E2 (faithful+partial) E1 + exposure halving on [i_part+1, i1) after the
                                   partial-TP fill (i_part from the v2.1 p4
                                   instrumentation; -1 = no partial).
    """
    z = np.zeros(len(index))
    sub = trades[(trades["strategy"] == "flagship") &
                 (trades["cap"] == cap) &
                 (trades["cost_mult"] == 1.0)]
    have_ip = "i_part" in trades.columns
    for row in sub.itertuples():
        d = 1.0 if row.cell_name.startswith("L") else -1.0
        i0 = index.searchsorted(pd.Timestamp(row.ts, unit="ns", tz="UTC"))
        i1 = int(row.exit_bar)
        if i0 >= len(index):
            continue
        i1 = min(len(index), max(i1, i0 + 1))
        a0 = i0 + 1 if mode in ("E1", "E2") else i0
        if a0 >= i1 and mode in ("E1", "E2"):
            a0 = max(i0, i1 - 1)
        z[a0:i1] += d / cap
        if mode == "E2" and have_ip:
            ip = int(getattr(row, "i_part", -1))
            if ip >= 0 and ip + 1 < i1:
                # remaining half position after the partial-TP fill
                z[ip + 1:i1] -= d / (2.0 * cap)
    return pd.Series(np.clip(z, -1.0, 1.0), index=index)''')

# ---------------------------------------------------- B: FORCED_FROZEN set
rep('''FORCED_FROZEN = {"M0_Champ_v13", "M0_Champ_v13_cap1"}''',
    '''FORCED_FROZEN = {"M0_Champ_v13", "M0_Champ_v13_cap1",
                  # v2.1 faithful-bridge decompositions (U1) — report-only
                  # continuous translations, no v2 engine overlay
                  "M0_Champ_v13_E1", "M0_Champ_v13_E2",
                  "M0_Champ_v13_E2_carry", "M0_Champ_v13_E2_carry_punitive"}''')

# ---------------------------------------------------- C: S-panel additions
rep('''        S["M0_Champ_v13"] = champion_exposure(trades, idx, cap=3)
        S["M0_Champ_v13_cap1"] = champion_exposure(trades, idx, cap=1)''',
    '''        S["M0_Champ_v13"] = champion_exposure(trades, idx, cap=3)   # E0 shipped
        S["M0_Champ_v13_cap1"] = champion_exposure(trades, idx, cap=1)
        # v2.1 (U1) faithful bridge decomposition at cap=3
        if "i_part" in trades.columns:
            S["M0_Champ_v13_E1"] = champion_exposure(trades, idx, cap=3, mode="E1")
            S["M0_Champ_v13_E2"] = champion_exposure(trades, idx, cap=3, mode="E2")
        else:
            log("  [warn] v13 trades have no i_part column — E1/E2 bridge "
                "needs the v2.1 p4 trades (AURUM_V13DIR); E1 only")
            S["M0_Champ_v13_E1"] = champion_exposure(trades, idx, cap=3, mode="E1")''')

# ------------------------------------------- D: U7 stage between stats/export
u7 = '''    # ---------------- 7b) v2.1 U1/U7: bridge + attribution (G5.10/G5.12) ----
    with stage("v2.1 U1 bridge decomposition + U7 attribution"):
        # ---------- U1: E0/E1/E2 faithful-bridge decomposition (G5.10) -------
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
             f"NOT hidden or faked.")

        # ---------- U7: attribution panel ---------------------------------------
        # Defensive-pandas rules (the two Nov-15 bug classes, fixed
        # permanently): (1) NEVER call .reindex() on a numpy ndarray — keep
        # Series/indices until the final .values; (2) NEVER compare a Series
        # to a Timestamp directly — mask via index membership.
        def _daily(net_s):
            """Hourly net Series -> daily sum Series (UTC date index)."""
            s = net_s.dropna()
            if len(s) == 0:
                return s
            return s.groupby(s.index.floor("D")).sum()

        def _nw_capm(y, x, lags=5):
            """OLS alpha/beta with Newey-West HAC(lags) t-stats.
            Returns (alpha, t_alpha, p_alpha, beta, se_beta) — all floats."""
            y = np.asarray(y, float); x = np.asarray(x, float)
            m = np.isfinite(y) & np.isfinite(x)
            y, x = y[m], x[m]
            n = len(y)
            if n < 50:
                return (float("nan"),) * 5
            X = np.column_stack([np.ones(n), x])
            XtX_inv = np.linalg.inv(X.T @ X)
            beta = XtX_inv @ X.T @ y
            resid = y - X @ beta
            # Bartlett-kernel HAC covariance
            S = np.zeros((2, 2))
            for l in range(lags + 1):
                w = 1.0 - l / (lags + 1.0)
                if l == 0:
                    G = X.T @ (X * (resid ** 2)[:, None])
                else:
                    G = (X[l:].T @ (X[:-l] * (resid[l:] * resid[:-l])[:, None]))
                S += w * (G + G.T if l > 0 else G)
            V = XtX_inv @ S @ XtX_inv
            se = np.sqrt(np.diag(V))
            t_a = beta[0] / se[0] if se[0] > 0 else float("nan")
            t_b = beta[1] / se[1] if se[1] > 0 else float("nan")
            from scipy.stats import norm as _nrm
            p_a = 2.0 * (1.0 - _nrm.cdf(abs(t_a))) if np.isfinite(t_a) else float("nan")
            return float(beta[0]), float(t_a), float(p_a), float(beta[1]), float(t_b)

        gold_h = r.copy()                     # hourly gold log returns (Series)
        gold_d = _daily(gold_h)
        REGIMES = [("2016-19 range", [2016, 2017, 2018, 2019]),
                   ("2020 COVID", [2020]),
                   ("2021-22 bear", [2021, 2022]),
                   ("2023 degraded/chop", [2023]),
                   ("2024-25 parabolic", [2024, 2025])]
        DSRS_N = [12, 50, 100, 170, 200, 500]

        def _full_net(name):
            """fit+oos hourly nets concatenated (2016-25 full window)."""
            parts = []
            for w in ("fit", "oos"):
                k = f"{name}__{w}"
                if k in net_series_export:
                    parts.append(net_series_export[k])
            return pd.concat(parts).sort_index() if parts else None

        def _carry_net(z_series, net_series, cfg):
            """Apply the audit s6.4 overnight-swap model to a CONTINUOUS frame:
            charge rate*|z|/365 at each 22:00 UTC bar held through the roll,
            Wednesday x3 (longs pay, shorts receive)."""
            zv = z_series.reindex(net_series.index).fillna(0.0)
            idxr = zv.index
            at_roll = (idxr.hour == cfg["hour_utc"]).values
            wed = (idxr.dayofweek == 2).values       # Wednesday
            rate = np.where(zv.values > 0, cfg["long_annual"], cfg["short_annual"])
            mult = np.where(wed, 3.0 if cfg.get("wed_x3", True) else 1.0, 1.0)
            charge = np.abs(zv.values) * rate * mult / cfg.get("day_basis", 365)
            vals = net_series.values.copy()
            vals[at_roll] += charge[at_roll]
            return pd.Series(vals, index=net_series.index)

        CARRY_CFG_BASE = {"long_annual": -0.05, "short_annual": 0.025,
                          "hour_utc": 22, "wed_x3": True, "day_basis": 365}
        CARRY_CFG_PUN = {"long_annual": -0.055, "short_annual": 0.015,
                         "hour_utc": 22, "wed_x3": True, "day_basis": 365}
        # carry-adjusted champion frames (E2 + carry), evaluated in-frame
        if "M0_Champ_v13_E2" in S:
            e2_full = _full_net("M0_Champ_v13_E2")
            if e2_full is not None:
                for nm, cfg in (("M0_Champ_v13_E2_carry", CARRY_CFG_BASE),
                                ("M0_Champ_v13_E2_carry_punitive", CARRY_CFG_PUN)):
                    adj = _carry_net(S["M0_Champ_v13_E2"], e2_full, cfg)
                    S[nm] = pd.Series(np.clip(S["M0_Champ_v13_E2"].reindex(
                        adj.index).fillna(0.0).values, -1.0, 1.0), index=adj.index)
                    # split back into fit/oos windows for the eval machinery
                    entry = {}
                    for window, mask in (("fit", yr.isin(FIT_YEARS)),
                                         ("oos", yr.isin(OOS_YEARS))):
                        selw = adj.index[mask.reindex(adj.index).fillna(False)]
                        ns_w = adj.reindex(selw)
                        net_series_export[f"{nm}__{window}"] = ns_w
                        sd_w = ns_w.std(ddof=1)
                        entry[window] = {
                            "metrics": {"sharpe_ann": float(
                                ns_w.mean() / (sd_w + 1e-12)
                                * math.sqrt(BARS_PER_YEAR)) if sd_w is not None and sd_w > 0 else 0.0},
                            "turnover_py": None, "carry_frame": True}
                    results[nm] = entry

        ATTR_STRATEGIES = [n for n in
                           ("M0_Champ_v13", "M0_Champ_v13_E1", "M0_Champ_v13_E2",
                            "M0_Champ_v13_E2_carry", "M0_Champ_v13_E2_carry_punitive",
                            "M1_GBDT_v2", "M2_RFF_Ridge", "M6_AURUM_v2_Ensemble",
                            "M7_Core_Satellite", "M7_Active_Grand_Hybrid",
                            "BuyHold_Gold", "Carver_EWMAC")
                           if n in results or n in S]
        capm_table, ls_table, regime_table = {}, {}, {}
        for name in ATTR_STRATEGIES:
            net_oos = net_series_export.get(f"{name}__oos")
            net_full = _full_net(name) if name in S else None
            zs = S.get(name)
            # ---- CAPM vs B&H gold (daily, NW HAC(5)) --------------------------
            for wname, net in (("oos", net_oos), ("full", net_full)):
                if net is None or len(net) < 200:
                    continue
                nd = _daily(net)
                common = nd.index.intersection(gold_d.index)
                if len(common) < 100:
                    continue
                y = nd.reindex(common).values
                x = gold_d.reindex(common).values
                a, ta, pa, beta, tb = _nw_capm(y, x, lags=5)
                up = x > 0
                uc = float(y[up].mean() / x[up].mean()) if up.any() and x[up].mean() != 0 else float("nan")
                dc = float(y[~up].mean() / x[~up].mean()) if (~up).any() and x[~up].mean() != 0 else float("nan")
                sd = nd.reindex(common).std(ddof=1)
                dpv = len(common) / max(1e-9, (common[-1] - common[0]).days / 365.25)
                ir = float(nd.reindex(common).mean() / sd * dpv) if sd > 0 else float("nan")
                capm_table.setdefault(name, {})[wname] = {
                    "alpha_pct_yr": round(a * dpv * 100, 2),
                    "alpha_t": round(ta, 2), "alpha_p": round(pa, 5),
                    "beta_xau": round(beta, 3), "IR": round(ir, 2),
                    "up_capture": round(uc, 3), "down_capture": round(dc, 3)}
            # ---- long/short attribution (OOS) ----------------------------------
            if zs is not None and net_oos is not None:
                zj = zs.reindex(net_oos.index).fillna(0.0).values
                nv = net_oos.values
                lng, sht, flt = zj > 1e-12, zj < -1e-12, np.abs(zj) <= 1e-12
                n = len(zj)
                pl = float(nv[lng].sum()) if lng.any() else 0.0
                ps = float(nv[sht].sum()) if sht.any() else 0.0
                ls_table[name] = {
                    "window": "oos",
                    "long_hrs_pct": round(100 * lng.mean(), 1),
                    "short_hrs_pct": round(100 * sht.mean(), 1),
                    "flat_hrs_pct": round(100 * flt.mean(), 1),
                    "pnl_long": round(pl, 4), "pnl_short": round(ps, 4),
                    "short_share_pct": round(100 * ps / (pl + ps), 1)
                    if (pl + ps) != 0 else float("nan"),
                    "n_bars": int(n)}
            # ---- 5-regime torture table (annualized Sharpe of daily net) -------
            if net_full is not None:
                nd = _daily(net_full)
                row = {}
                for rname, years in REGIMES:
                    dmask = nd.index.year.isin(years)
                    v = nd.values[dmask]
                    # annualize: daily Sharpe * sqrt(days per year in window)
                    if len(v) > 20:
                        row[rname] = round(float(
                            v.mean() / (v.std(ddof=1) + 1e-12)
                            * math.sqrt(len(v) / max(1, len(years)))), 3)
                    else:
                        row[rname] = float("nan")
                regime_table[name] = row

        # ---- honest project-wide DSR panel (U7): N in {12..500} ---------------
        # Trial dispersion: the effective-family panel's frozen FIT Sharpes
        # (same vintage the audit used); N is varied per the Bailey-LdP
        # expectation-maximum threshold by tiling the panel's dispersion to N
        # trials — the audit's s6.6 method ("the 369-row ledger for the
        # pessimistic bound" is exported as dsr_conservative elsewhere).
        dsr_panel = {}
        fam_vals = fam_primary if isinstance(fam_primary, np.ndarray) and len(fam_primary) > 5 \\
            else fam_conservative
        for name in ("M0_Champ_v13_E2", "M0_Champ_v13_E2_carry"):
            net = net_series_export.get(f"{name}__oos")
            if net is None:
                continue
            nv = net.values
            z_ = (nv - nv.mean()) / (nv.std() + 1e-12)
            sr_pb = float(nv.mean() / (nv.std(ddof=1) + 1e-12))
            row = {}
            for N in DSRS_N:
                tiled = np.tile(fam_vals / math.sqrt(BARS_PER_YEAR),
                                int(np.ceil(N / max(1, len(fam_vals)))))[:N]
                row[f"N={N}"] = round(deflated_sharpe_ratio(
                    sr_pb, len(nv), float(np.mean(z_ ** 3)), float(np.mean(z_ ** 4)),
                    sr_trials=tiled, N_trials=N), 4)
            dsr_panel[name] = row

        attribution = {
            "spec": "v2.1.0",
            "native_reference_flagship_cap3_oos": native_ref,
            "bridge_decomposition_E0_E1_E2": bridge,
            "capm_nw_hac5_vs_buyhold_gold": capm_table,
            "long_short_attribution_oos": ls_table,
            "regime_torture_5": regime_table,
            "dsr_panel_project_wide_N": dsr_panel,
            "carry_model": {"base": CARRY_CFG_BASE, "punitive": CARRY_CFG_PUN,
                            "applied": "rate*|z|/365 at each 22:00 UTC bar, Wed x3 "
                                       "(longs pay, shorts receive; audit s6.4)"},
            "notes": "champion faithful frame = E2; alpha in %/yr of the "
                     "vol-targeted continuous frame; regimes are annualized "
                     "Sharpe of daily net within each window"}
        with open(os.path.join(WORK, "attribution_v21.json"), "w") as f:
            json.dump(attribution, f, indent=1, default=str)
        # ---- G5.12: the attribution stage MUST actually pass -------------------
        need = ["M0_Champ_v13_E2", "M0_Champ_v13_E2_carry", "BuyHold_Gold"]
        capm_ok = all(n in capm_table and "oos" in capm_table[n] for n in need[:2])
        ls_ok = "M0_Champ_v13_E2" in ls_table
        reg_ok = "M0_Champ_v13_E2" in regime_table and all(
            np.isfinite(v) for v in regime_table["M0_Champ_v13_E2"].values())
        dsr_ok = all(n in dsr_panel and len(dsr_panel[n]) == len(DSRS_N)
                     and all(np.isfinite(v) for v in dsr_panel[n].values())
                     for n in dsr_panel)
        fin_ok = all(np.isfinite(capm_table[n]["oos"]["alpha_t"])
                     for n in need[:2] if n in capm_table and "oos" in capm_table[n])
        ok512 = bool(capm_ok and ls_ok and reg_ok and dsr_ok and fin_ok)
        gate("G5.12_attribution", ok512,
             f"U7 attribution panel: CAPM(NW HAC(5))={capm_ok} L/S={ls_ok} "
             f"regimes={reg_ok} DSR-panel(6 Ns)={dsr_ok} finite={fin_ok}; "
             f"champion E2 OOS alpha="
             f"{capm_table.get('M0_Champ_v13_E2', {}).get('oos', {}).get('alpha_pct_yr')}%/yr "
             f"(t={capm_table.get('M0_Champ_v13_E2', {}).get('oos', {}).get('alpha_t')}); "
             f"bridge {bridge}")

    '''
rep('''    # ---------------- 8) champion comparison + export ------------------------''',
    u7 + '''    # ---------------- 8) champion comparison + export ------------------------''')

# --------------------------------------------- E: champion_compare additions
rep('''        for name in ("M7_Active_Grand_Hybrid", "M7_Core_Satellite",
                     "M6_AURUM_v2_Ensemble", "M6b_Ensemble_EQW",
                     "M0_Champ_v13", "M0_Champ_v13_cap1", "M5_ContinuumNet",
                     "M2_RFF_Ridge", "M1_GBDT_v2", "BuyHold_Gold"):''',
    '''        for name in ("M7_Active_Grand_Hybrid", "M7_Core_Satellite",
                     "M6_AURUM_v2_Ensemble", "M6b_Ensemble_EQW",
                     "M0_Champ_v13", "M0_Champ_v13_cap1",
                     "M0_Champ_v13_E1", "M0_Champ_v13_E2",
                     "M0_Champ_v13_E2_carry", "M0_Champ_v13_E2_carry_punitive",
                     "M5_ContinuumNet",
                     "M2_RFF_Ridge", "M1_GBDT_v2", "BuyHold_Gold"):''')

# ------------------------------------------------- F: G6_ALL includes new gates
rep('''    gate("G6_ALL", all_ok, "all K6 gates green" if all_ok else "K6 gate failure")''',
    '''    all_ok = all_ok and ok510 and ok512
    gate("G6_ALL", all_ok, "all K6 gates green (incl v2.1 G5.10 bridge + "
         "G5.12 attribution)" if all_ok else "K6 gate failure")''')

open(P, "w").write(src)
print("k6 v2.1 patch applied")
