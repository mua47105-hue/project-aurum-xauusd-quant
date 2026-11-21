#!/usr/bin/env python3
"""v2.1 main() patch for kaggle_kernels/v1_3/aurum-p4-backtest.py (Step 2).

Applies, with anchor-asserted replacements:
  A. frozen-policy branch (default) / refit-with-drift branch around the
     original causal-policy-fit stage  (+ G5.11_policy_frozen gate)
  B. first-class r=1 strategies (U4): r1_fast (flagship machinery, w=[1,0,0])
     + r1_clean (raw EV-argmax)
  C. resolve scope + strat_caps for the r1 strategies
  D. v2.1 evaluation panel: risk-tier x throttle x carry matrix (U5),
     ex-2023 sensitivity, M1 friction survival matrix (U6),
     G1.7 bar-coverage gate (Flaw #12)
  E. exports: i_part column in trades_all, summary['v21'], v21_summary.json,
     G5.5 count 7->8
  F. __main__ argparse (--use-frozen-policy default ON / --refit-policy)
"""
import re
import sys

P = "kaggle_kernels/v1_3/aurum-p4-backtest.py"
src = open(P).read()
orig = src


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (
        f"anchor count {src.count(old)} != {count} for: {old[:90]!r}")
    src = src.replace(old, new, count)


# ---------------------------------------------------------------- A: policy
# Replace the policy-fit stage (with-block) + fit_record + G5.7 with a
# frozen/refit branch structure. The original with-block body is re-indented
# +4 programmatically.
m = re.search(
    r"(    # ---------- causal policy fitting \(<=2022\) -+\n)"
    r"(    with stage\(\"causal policy fit on 2016-2022 \(zero 2023-25 peeking\)\"\):\n)"
    r"(.*?)"
    r"(    fit_record = \{\n)", src, re.S)
assert m, "policy-fit block not found"
body = m.group(3)
body_re = "\n".join(("    " + ln if ln.strip() else ln) for ln in body.split("\n"))

frozen_branch = '''    # ---------- policy: FROZEN (v2.1 default, Flaw #11) or refit ----------
    if POLICY_MODE == "frozen":
        # v2.1 (Flaw #11): the DECISION policy is loaded from the frozen
        # config (v1_3/config/frozen_policy.json) instead of being re-derived
        # at run time from near-tied FIT grids (F6 margin was 1.27%; an
        # unpinned xgboost flips the selection). No grid is evaluated.
        with stage("policy: FROZEN load (Flaw #11; zero grid re-derivation)"):
            fp = load_frozen_policy()
            fpol = fp["flagship_policy"]
            w = np.array(fpol["w_brier_F1_for_stack"], np.float64)
            kappa = fpol["kappa_brier_F2_for_stack"]
            kap3 = fpol["kappa"]; lam = fpol["lambda_var"]; ev_min = fpol["ev_min"]
            params = {"w": np.array(fpol["w_pooled_r123"], np.float64),
                      "kappa": kap3, "lam": lam, "ev_min": ev_min}
            rule = fpol["overlay_rule"]
            u = fpol["stack_u"]
            stack_params = dict(params); stack_params["u"] = u
            br_w = br_k = br_u = None
            grid_w = grid_k = grid_le = grid_w_perf = None
            ablation = {"chosen_rule": rule, "frozen": True}
            f3best = fp["fit_provenance"]["f3_best_clean"]
            log(f"  frozen policy loaded: w={fpol['w_pooled_r123']} kappa={kap3} "
                f"lam={lam} ev_min={ev_min} overlay={rule} u={u}")
            log(f"  frozen provenance: F6 margin "
                f"{fp['fit_provenance']['f6_margin']['winner']['fit_ann_sharpe']} vs "
                f"{fp['fit_provenance']['f6_margin']['runner_up']['fit_ann_sharpe']} "
                f"(razor-thin -> the policy is frozen, the env pinned)")
        fit_record = {
            "mode": "frozen", "frozen_source": os.path.basename(V21_FROZEN_POLICY),
            "fit_window_years": FIT_YEARS, "max_year_used": max(FIT_YEARS),
            "oos_years_never_touched_by_fits": OOS_YEARS,
            "protocol": fp["fit_provenance"]["protocol"],
            "w": {"value_brier_F1": w.tolist(),
                  "value_perf_F6_flagship": params["w"].tolist()},
            "kappa": {"value_brier_F2": kappa, "value_perf_F3": kap3},
            "lambda_var": {"value": lam},
            "ev_min": {"value": ev_min},
            "overlay_rule": {"value": rule, "ablation": ablation},
            "stack_u": {"value": u},
            "f6_w_perf_grid": fp["fit_provenance"]["f6_margin"],
            "f3_best": f3best}
        gate("G5.7_fit_causality", max(FIT_YEARS) == 2022,
             f"policy FROZEN from {os.path.basename(V21_FROZEN_POLICY)} "
             f"(originally fitted on {min(FIT_YEARS)}..{max(FIT_YEARS)} only; "
             f"OOS {OOS_YEARS} never touched by any fit)")
        gate("G5.11_policy_frozen", True,
             f"decision policy loaded from frozen config (Flaw #11 fix): "
             f"w={fpol['w_pooled_r123']} kappa={kap3} lambda_var={lam} "
             f"ev_min={ev_min} overlay={rule} stack_u={u}; integrity check passed")
        POLICY_DRIFT = {"mode": "frozen", "refit_performed": False}
    else:
        # --refit-policy: run the original causal fit machinery for DRIFT
        # TELEMETRY against the frozen registry. The decisions still use the
        # REFIT values ONLY in this diagnostic mode (never in production).
'''

# fit_record block + G5.7 gate in refit mode: capture from 'fit_record = {' to
# the end of the G5.7 gate call (index-based: the dict closes inline with '}}'),
# re-indent +4, and append drift telemetry.
_g7_tail = '         f"fold predictions are themselves walk-forward (heads trained <=y-1)")\n'
i0 = src.index("    fit_record = {\n")
i1 = src.index(_g7_tail, i0) + len(_g7_tail)
assert i1 > i0, "fit_record/G5.7 block not found"
fitrec_body = src[i0:i1]

refit_tail = '''        # v2.1 drift telemetry: compare the refit against the frozen registry
        fp = load_frozen_policy()
        fpol = fp["flagship_policy"]
        POLICY_DRIFT = {
            "mode": "refit",
            "w_refit": params["w"].tolist(), "w_frozen": fpol["w_pooled_r123"],
            "kappa_refit": kap3, "kappa_frozen": fpol["kappa"],
            "lambda_refit": lam, "lambda_frozen": fpol["lambda_var"],
            "ev_min_refit": ev_min, "ev_min_frozen": fpol["ev_min"],
            "overlay_refit": rule, "overlay_frozen": fpol["overlay_rule"],
            "u_refit": u, "u_frozen": fpol["stack_u"]}
        drift_zero = (params["w"].tolist() == fpol["w_pooled_r123"]
                      and kap3 == fpol["kappa"] and lam == fpol["lambda_var"]
                      and ev_min == fpol["ev_min"] and rule == fpol["overlay_rule"]
                      and u == fpol["stack_u"])
        POLICY_DRIFT["drift_zero_under_pinned_env"] = bool(drift_zero)
        gate("G5.11_policy_frozen", drift_zero,
             f"REFIT drift telemetry vs frozen registry: "
             f"{'ZERO drift (pinned env reproduces the frozen policy)' if drift_zero else 'DRIFT DETECTED — investigate environment pin'} "
             + json.dumps({k: v for k, v in POLICY_DRIFT.items() if k != 'mode'}))

'''

fitrec_ind = "\n".join(("    " + ln if ln.strip() else ln)
                       for ln in fitrec_body.split("\n"))
new_block = (m.group(1) + frozen_branch
             + '        with stage("causal policy REFIT on 2016-2022 (drift telemetry; not the production path)"):\n'
             + body_re + fitrec_ind + "\n" + refit_tail)
src = src[:m.start()] + new_block + src[i1:]

# ------------------------------------------------------- B: r1 strategies
rep('''        log(f"  stack: {len(C['stack']['pos']):,}")''',
    '''        log(f"  stack: {len(C['stack']['pos']):,}")
        # v2.1 (U4): first-class r=1 strategies — audit s4.2: r=1 alone matches
        # or beats the pooled flagship OOS (2.486/2.599 vs 2.453) with 3x
        # cheaper inference and +44% more total R under the raw rule.
        C["r1_fast"] = flagship_candidates(
            b, {**params, "w": np.array([1.0, 0.0, 0.0])}, gbdt_dir, rule)
        C["r1_fast"]["policy"] = ("r=1-only flagship machinery (w=[1,0,0]; "
                                  "LCB-EV + shrinkage + EV_min + GBDT gate)")
        log(f"  r1_fast: {len(C['r1_fast']['pos']):,}")
        C["r1_clean"] = clean_ev_r1_candidates(b, gbdt_dir, rule)
        C["r1_clean"]["policy"] = ("raw r=1 EV-argmax + GBDT gate "
                                   "(no LCB, no shrinkage, no depth pool)")
        log(f"  r1_clean: {len(C['r1_clean']['pos']):,}")''')

# ------------------------------------------------------- C: resolve scope
rep('''                R[k] = resolve_candidates(b, cand,
                                          rule if k in ("flagship", "stack")
                                          else "clean")''',
    '''                R[k] = resolve_candidates(
                    b, cand,
                    rule if k in ("flagship", "stack", "r1_fast", "r1_clean")
                    else "clean")''')

# ------------------------------------------------------- D: strat_caps
rep('''    strat_caps = {k: [1] for k in C}
    for k in ("flagship", "stack", "canon"):
        strat_caps[k] = [1, 3]''',
    '''    strat_caps = {k: [1] for k in C}
    for k in ("flagship", "stack", "canon", "r1_fast"):
        strat_caps[k] = [1, 3]
    strat_caps["r1_clean"] = [3]          # U4: Clean EV is a cap=3 strategy''')

# --------------------------------------------- E: v2.1 panel before export
v21_panel = '''    # ---------- v2.1 remediation panel (U4/U5/U6 + G1.7) ---------------------
    v21 = {"spec": "v2.1.0", "policy_drift": POLICY_DRIFT}
    with stage("v2.1 risk-tier x throttle x carry matrix (flagship cap3, OOS)"):
        res_fl = R["flagship"]
        oos_m = np.isin(res_fl["year"], OOS_YEARS)
        ex23_m = np.isin(res_fl["year"], [2024, 2025])   # ex-2023 sensitivity
        tier_rows = {}
        for tier_name, rc in RISK_TIERS.items():
            for thr_name, tdd in (("throttle15", THROTTLE_DD_LIMIT),
                                  ("unthrottled", -1.0)):
                for cr_name, cc in (("no_carry", None),
                                    ("carry_base", CARRY_BASE),
                                    ("carry_punitive", CARRY_PUNITIVE)):
                    ex = execute(b, res_fl, 3, oos_m, 1.0, risk_cap=rc,
                                 throttle_dd=tdd, carry=cc)
                    blk = portfolio_block(ex, 1.0)
                    pr = blk.get("portfolio") or {}
                    tier_rows[f"{tier_name}|{thr_name}|{cr_name}"] = {
                        "risk_cap": rc, "n": blk.get("n_trades", 0),
                        "E_R": blk.get("expectancy_R"),
                        "sharpe": pr.get("sharpe_annualized"),
                        "cagr_pct": pr.get("cagr_pct"),
                        "max_dd_pct": pr.get("max_dd_pct")}
                    log(f"  {tier_name}|{thr_name}|{cr_name}: "
                        f"n={blk.get('n_trades', 0)} "
                        f"SR={pr.get('sharpe_annualized', float('nan')):.4f} "
                        f"MaxDD={pr.get('max_dd_pct', float('nan')):.2f}%")
        v21["tier_matrix_flagship_cap3_oos"] = tier_rows
        # ex-2023 sensitivity (2024-25 only), with and without carry (Flaw #12)
        ex23 = {}
        for combo, (rc, tdd, cc) in {
                "standard|throttle15|no_carry": (RISK_TIERS["standard"], THROTTLE_DD_LIMIT, None),
                "standard|throttle15|carry_base": (RISK_TIERS["standard"], THROTTLE_DD_LIMIT, CARRY_BASE),
                "standard|unthrottled|no_carry": (RISK_TIERS["standard"], -1.0, None),
                "pilot|unthrottled|no_carry": (RISK_TIERS["institutional_pilot"], -1.0, None)}.items():
            ex = execute(b, res_fl, 3, ex23_m, 1.0, risk_cap=rc,
                         throttle_dd=tdd, carry=cc)
            blk = portfolio_block(ex, 1.0)
            pr = blk.get("portfolio") or {}
            ex23[combo] = {"n": blk.get("n_trades", 0),
                           "E_R": blk.get("expectancy_R"),
                           "sharpe": pr.get("sharpe_annualized"),
                           "cagr_pct": pr.get("cagr_pct"),
                           "max_dd_pct": pr.get("max_dd_pct")}
        v21["ex2023_sensitivity_2024_25"] = ex23

    with stage("v2.1 M1 path resolution + friction survival matrix (U6)"):
        m1d = load_m1(b)
        if m1d is not None:
            log(f"  M1 feed: {m1d['n']:,} bars "
                f"({pd.DatetimeIndex([m1d['ts'][0]]).tz_localize('UTC')[0]} .. "
                f"{pd.DatetimeIndex([m1d['ts'][-1]]).tz_localize('UTC')[0]})")
            atr_all = b["atr"]
            ok_atr = np.isfinite(atr_all) & (atr_all > 0)
            rv_all = (b["hi"] - b["lo"]) / np.where(ok_atr, atr_all, np.nan)
            rv_thr = float(np.nanquantile(rv_all, 0.9))
            exo3 = EX["flagship"][3][1.0]["oos"]
            fr = m1_resolve_matrix(b, m1d, res_fl, exo3["sel"], rv_thr)
            v21["m1_friction_matrix_flagship_cap3_oos"] = {
                k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                for k, v in fr.items() if not k.startswith("_")}
            v21["m1_friction_matrix_flagship_cap3_oos"]["n_hour_mapped"] = fr.get("_n_hour_mapped")
            v21["m1_top_decile_rv_threshold"] = rv_thr
            for k, v in v21["m1_friction_matrix_flagship_cap3_oos"].items():
                if isinstance(v, dict) and "E_R" in v:
                    log(f"  {k}: n={v['n']} E[R]={v['E_R']:.4f} "
                        f"(vs H1 walk {v['vs_h1_ER']:+.4f})")
            # keep minute-level arrays for the k6 faithful bridge (U1)
            np.save(os.path.join(WORK, "v21_m1_R_flagship_cap3_oos.npy"),
                    fr["m1_pure"]["_R"])
            np.save(os.path.join(WORK, "v21_m1_i_part_flagship_cap3_oos.npy"),
                    fr["m1_pure"]["_i_part_m1"])
        else:
            log("  [warn] M1 feed not found — friction matrix skipped")
            v21["m1_friction_matrix_flagship_cap3_oos"] = None

    with stage("v2.1 OOS bar-coverage audit (Flaw #12 -> G1.7)"):
        px_years = pd.DatetimeIndex(b["px"].index).year
        ycnt = pd.Series(px_years).value_counts().sort_index()
        base_years = [y for y in ycnt.index if 2016 <= y <= 2022]
        base_mean = float(ycnt[base_years].mean())
        cov = {}
        for y in OOS_YEARS:
            n_y = int(ycnt.get(y, 0))
            ratio = n_y / base_mean
            cov[int(y)] = {"h1_bars": n_y, "vs_2016_22_mean_pct": round((ratio - 1) * 100, 1),
                           "degraded": bool(ratio < COVERAGE_FLAG_RATIO)}
        v21["oos_bar_coverage"] = {
            "baseline_2016_22_mean_h1_bars": round(base_mean, 1),
            "per_year": cov}
        degraded = [y for y, d in cov.items() if d["degraded"]]
        g17_ok = len(cov) == len(OOS_YEARS) and (2023 in degraded)
        gate("G1.7_bar_coverage", g17_ok,
             f"OOS H1 bar coverage vs 2016-22 mean ({base_mean:.0f}): "
             + "; ".join(f"{y}: {d['h1_bars']} ({d['vs_2016_22_mean_pct']:+.1f}%)"
                         + (" DEGRADED" if d["degraded"] else "")
                         for y, d in cov.items())
             + f" — flagged years {degraded}; ex-2023 sensitivity exported "
               f"(2023 is thin: the entire v2 feature store is built on "
               f"~18%-gappy 2023 M1 data, audit s2.6)")
    summary["v21"] = v21

'''
rep('''    # ---------- export ----------------------------------------------------------''',
    v21_panel + '''    # ---------- export ----------------------------------------------------------''')

# ------------------------------------------------- F: trades i_part column
rep('''                        "exit_bar": sub["exit_bar"][sel].astype(np.int64),
                        "cost_base": sub["cost_base"][sel].astype(np.float32),''',
    '''                        "exit_bar": sub["exit_bar"][sel].astype(np.int64),
                        "i_part": sub["i_part"][sel].astype(np.int64),
                        "cost_base": sub["cost_base"][sel].astype(np.float32),''')

# ------------------------------------------------- G: G5.5 count 7 -> 8
rep('''        n_files = sum(os.path.isfile(os.path.join(WORK, f_)) for f_ in
                      ("trades_all.parquet", "equity_curves.parquet",
                       "backtest_main.json", "policy_fit.json",
                       "overlay_ablation.json", "gbdt_audit.json",
                       "backtest_summary.md"))
        gate("G5.5_export", n_files == 7,
             f"exported {n_files}/7 files; trades={len(trades):,} rows")''',
    '''        with open(os.path.join(WORK, "v21_summary.json"), "w") as f:
            json.dump(v21, f, indent=1, default=float)
        n_files = sum(os.path.isfile(os.path.join(WORK, f_)) for f_ in
                      ("trades_all.parquet", "equity_curves.parquet",
                       "backtest_main.json", "policy_fit.json",
                       "overlay_ablation.json", "gbdt_audit.json",
                       "backtest_summary.md", "v21_summary.json"))
        gate("G5.5_export", n_files == 8,
             f"exported {n_files}/8 files (v2.1 adds v21_summary.json); "
             f"trades={len(trades):,} rows")''')

# ------------------------------------------------- H: __main__ argparse
rep('''if __name__ == "__main__":
    WORK = os.environ.get("AURUM_WORK",''',
    '''POLICY_MODE = "frozen"          # v2.1 default (Flaw #11); "refit" = telemetry


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="AURUM v1.3/v2.1 p4 backtest")
    _ap.add_argument("--use-frozen-policy", dest="frozen", action="store_true",
                     default=True,
                     help="load the frozen policy (v2.1 default, Flaw #11)")
    _ap.add_argument("--refit-policy", dest="frozen", action="store_false",
                     help="re-run the causal FIT grids (drift telemetry only)")
    _args, _unknown = _ap.parse_known_args()
    if not _args.frozen:
        POLICY_MODE = "refit"
    print(f"[p4] POLICY_MODE={POLICY_MODE}", flush=True)
    WORK = os.environ.get("AURUM_WORK",''')

open(P, "w").write(src)
print("patch applied:", len(orig), "->", len(src), "bytes")
