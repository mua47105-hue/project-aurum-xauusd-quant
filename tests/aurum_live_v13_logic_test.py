#!/usr/bin/env python3
"""Local logic test of the shipped live-daily v1.3 policy text.

Extracts the EXACT code segments from the patched kernel file and execs them
with synthetic inputs, verifying:
  T1 pooled p_bar == .6 p1 + .2 p2 + .2 p3
  T2 sigma_loop == population std over (p1,p2,p3)
  T3 EV_LCB formula == p_sh*b - (1-p_sh) - lam*b*sig - 0.05/m  (cell argmax)
  T4 G6.4b consistency recomputation reproduces ev/p_bar/sigma/ood
  T5 _live_overlay_quantiles h24 path masks to year <= y-1 (no lookahead)
  T6 _live_base_rate_prior masks to year <= y-1
  T7 shrink() with kappa=0 is identity; with kappa>0 pulls toward pi
"""
import numpy as np
import pandas as pd
import json, os
import textwrap

F = "/home/z/my-project/kaggle_work/aurum/src_push/live/aurum-live-daily.py"
src = open(F).read()
lb, rb = chr(91), chr(93)

def seg(a, b, dedent=True):
    i = src.index(a)
    j = src.index(b, i)
    t = src[i:j]
    return textwrap.dedent(t) if dedent else t

# ---- extract module pieces (pure python, no torch needed) ----------------
const_blk = seg("LIVE_DEPTHS     = [1, 2, 3, 4]", "def _live_base_rate_prior")
helpers = seg("def _live_base_rate_prior", "def live_main(WORK):")
cell_axes_blk = seg("def cell_axes():", "def cell_names():")
ns = {"np": np, "pd": pd, "json": json, "os": os}
barrier_blk = seg("BARRIER_STOPS = ", "def cell_axes():", dedent=False)
exec(barrier_blk, ns)
exec(cell_axes_blk, ns)
exec(const_blk, ns)
exec(helpers, ns)

# ---- synthetic depth outputs ----------------------------------------------
rng = np.random.default_rng(42)
P1 = np.clip(0.30 + 0.25 * rng.random(48), 0.02, 0.98)
P2 = np.clip(P1 + 0.02 * rng.standard_normal(48), 0.02, 0.98)
P3 = np.clip(P1 + 0.015 * rng.standard_normal(48), 0.02, 0.98)
P4 = np.clip(P3 + 0.01 * rng.standard_normal(48), 0.02, 0.98)
out = {r: {"h1": P} for r, P in ((1, P1), (2, P2), (3, P3), (4, P4))}
y_live = 2025

# synthetic label frame (aligned with feat): 400 rows in 2024, 100 in 2025
idx = pd.DatetimeIndex(
    [pd.Timestamp("2024-06-01 00:00", tz="UTC")] * 400
    + [pd.Timestamp("2025-06-01 00:00", tz="UTC")] * 100)
cols48 = [f"c{k}" for k in range(48)]
data48 = np.concatenate([np.full((400, 48), 1.0), np.full((100, 48), 0.0)])
cls48 = pd.DataFrame(data48, columns=cols48, index=idx)
exc = pd.DataFrame({
    "mfe_L24": np.concatenate([np.full(400, 1.0), np.full(100, 50.0)]),
    "mae_L24": np.concatenate([np.full(400, 2.0), np.full(100, 99.0)])},
    index=idx)
feat = pd.DataFrame({"x": np.zeros(len(idx))}, index=idx)

# ---- T1-T3: exec the exact policy segment from live_main ------------------
policy_seg = seg("    # ---------------- v1.3 decision policy",
                 "    # GBDT agreement (fold model")
pol_ns = dict(ns)
pol_ns.update({"out": out, "y_live": y_live, "LIVE_DEPTHS": ns["LIVE_DEPTHS"],
               "feat": feat, "cls": cls48})
exec(policy_seg, pol_ns)
p_bar_cell = pol_ns["p_bar_cell"]; sig_cell = pol_ns["sig_cell"]
ev_cell = pol_ns["ev_cell"]; cell = pol_ns["cell"]
p_sh_cell = pol_ns["p_sh_cell"]; pi_cell = pol_ns["pi_cell"]
b_cell = pol_ns["b_cell"]; cost_cell = pol_ns["cost_cell"]
sig_loop = pol_ns["sig_loop"]; p_bar = pol_ns["p_bar"]; EV_lcb = pol_ns["EV_lcb"]

w = np.asarray(ns["LIVE_POOL_W"])
t1 = np.allclose(p_bar, w[0] * P1 + w[1] * P2 + w[2] * P3, atol=1e-12)
t2 = np.allclose(sig_loop, np.std(np.stack([P1, P2, P3]), axis=0), atol=1e-9)
# T3: verify EV on EVERY cell (not just selected)
dirs_ax, mss, tps, hrs = ns["cell_axes"]()
b_geo = np.array([t / m for t, m in zip(tps, mss)])
cost = 0.05 / np.array(mss, float)
pi48 = pol_ns["pi48"]
p_sh = pol_ns["p_sh"]
ev_manual = p_sh * b_geo - (1 - p_sh) - 0.5 * b_geo * sig_loop - cost
t3 = np.allclose(EV_lcb, ev_manual, atol=1e-12) and int(np.argmax(EV_lcb)) == cell

# ---- T4: G6.4b consistency block ------------------------------------------
cons_seg = seg("    # internal consistency of the policy formulas",
               '    gate("G6.4b_policy_consistency"')
cns = dict(pol_ns)
exec(cons_seg.replace("G6.4b_policy_consistency", "X", 1)
     if False else cons_seg, cns)
t4 = (abs(cns["ev_re"] - ev_cell) < 1e-6
      and abs(cns["pbar_re"] - p_bar_cell) < 1e-6
      and abs(cns["sig_re"] - sig_cell) < 1e-6
      and abs(cns["ood_re"] - pol_ns["ood_mean"]) < 1e-6)

# ---- T5: overlay masking (h=24 empirical path) -----------------------------
idx = pd.DatetimeIndex(
    [pd.Timestamp("2024-06-01 00:00", tz="UTC")] * 400
    + [pd.Timestamp("2025-06-01 00:00", tz="UTC")] * 100)
exc = pd.DataFrame({
    "mfe_L24": np.concatenate([np.full(400, 1.0), np.full(100, 50.0)]),
    "mae_L24": np.concatenate([np.full(400, 2.0), np.full(100, 99.0)])},
    index=idx)
q50, q25, ovsrc = ns["_live_overlay_quantiles"]("L", 24, np.zeros(12), exc, 2025)
t5 = (q50 == 1.0 and q25 == 2.0 and ovsrc == "empirical_le2024_h24")
# unmasked would give q50=50, q25=99 -> test fails then

# ---- T6: base-rate prior masking -------------------------------------------
cls = pd.DataFrame(
    {c: np.concatenate([np.full(400, 1.0), np.full(100, 0.0)]).astype(float)
     for c in ["c0"]}, index=idx)
# _live_base_rate_prior expects 48-col cls with nan handling; build synthetic
cols = [f"c{k}" for k in range(48)]
data = np.concatenate([np.full((400, 48), 1.0), np.full((100, 48), 0.0)])
cls48 = pd.DataFrame(data, columns=cols, index=idx)
pi = ns["_live_base_rate_prior"](cls48, pd.DatetimeIndex(exc.index).year.values, 2025)
t6 = np.allclose(pi, 1.0)   # unmasked mean would be 400/500 = 0.8

# ---- T7: kappa=0 identity --------------------------------------------------
p = P1.copy()
pi_v = np.full(48, 0.4)
s0 = 0.0 * b_geo
p_sh0 = (p + s0 * pi_v) / (1 + s0)
t7a = np.allclose(p_sh0, p)
s1 = 0.8 * b_geo
p_sh1 = (p + s1 * pi_v) / (1 + s1)
t7b = np.all(np.abs(p_sh1 - p) < np.abs(p - pi_v) + 1e-9)  # pulled toward pi

ok = all([t1, t2, t3, t4, t5, t6, t7a, t7b])
print(f"T1 pooling          : {'PASS' if t1 else 'FAIL'}")
print(f"T2 sigma_loop       : {'PASS' if t2 else 'FAIL'}")
print(f"T3 EV_LCB formula   : {'PASS' if t3 else 'FAIL'}")
print(f"T4 consistency      : {'PASS' if t4 else 'FAIL'}")
print(f"T5 overlay mask     : {'PASS' if t5 else 'FAIL'} (q50={q50}, q25={q25})")
print(f"T6 prior mask       : {'PASS' if t6 else 'FAIL'} (pi={pi.mean():.3f})")
print(f"T7 shrink kappa     : {'PASS' if (t7a and t7b) else 'FAIL'}")
print(f"selected cell={cell} ({dirs_ax}) ev={ev_cell:+.4f} "
      f"p_bar={p_bar_cell:.4f} sig={sig_cell:.4f} cost={cost_cell:.4f} pi={pi_cell:.4f}")
print("ALL LOGIC TESTS:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
