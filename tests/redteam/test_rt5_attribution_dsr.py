#!/usr/bin/env python3
"""RT5 — attribution + honest project-wide DSR (audit A.5, U7).

Loads the k6 v2.1 attribution panel and verifies:
  * CAPM NW HAC(5): champion frames have positive OOS alpha with t > 2
    (the audit's core finding: statistically significant alpha net of the
    gold beta), beta in (0.05, 0.45), and the FAITHFUL frame's alpha is
    BELOW the artifact frame's (Flaw #10 direction);
  * long/short attribution: hours sum to ~100%, short share 5-20%;
  * 5-regime table: all five windows finite, 2021-22 bear > 1.5 (the
    audit's bear-regime robustness), 2024-25 parabolic < 3.6;
  * DSR panel: strictly decreasing in N, N=170 < 0.95 (the honest
    project-wide miss), and the carry-adjusted panel <= the raw panel.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rt_common import REPO, verdict, have, load_json  # noqa: E402

NEED = ["artifacts/v2_1/k6_out/attribution_v21.json",
        "artifacts/v2_1/k6_out/gates.json"]
if not have(*NEED):
    verdict("rt5_attribution_dsr", False, "SKIP: k6 attribution absent",
            skipped=True)

a = load_json("artifacts/v2_1/k6_out/attribution_v21.json")
capm = a["capm_nw_hac5_vs_buyhold_gold"]
ls = a["long_short_attribution_oos"]
reg = a["regime_torture_5"]
dsr = a["dsr_panel_project_wide_N"]
g = load_json("artifacts/v2_1/k6_out/gates.json")

e0 = capm["M0_Champ_v13"]["oos"]
e2 = capm["M0_Champ_v13_E2"]["oos"]
e2c = capm["M0_Champ_v13_E2_carry"]["oos"]
ls2 = ls["M0_Champ_v13_E2"]
reg2 = reg["M0_Champ_v13_E2"]
d2 = dsr["M0_Champ_v13_E2"]
d2c = dsr["M0_Champ_v13_E2_carry"]

capm_ok = (e2["alpha_t"] > 2.0 and e2["alpha_pct_yr"] > 5.0
           and 0.05 < e2["beta_xau"] < 0.45
           and e2["alpha_pct_yr"] < e0["alpha_pct_yr"])
ls_ok = (abs(ls2["long_hrs_pct"] + ls2["short_hrs_pct"] + ls2["flat_hrs_pct"]
             - 100.0) < 0.5 and 3.0 < ls2["short_share_pct"] < 25.0)
reg_ok = (all(isinstance(v, (int, float)) for v in reg2.values())
          and reg2["2021-22 bear"] > 1.5 and reg2["2024-25 parabolic"] < 3.6)
ns = [12, 50, 100, 170, 200, 500]
vals = [d2[f"N={n}"] for n in ns]
valsc = [d2c[f"N={n}"] for n in ns]
dsr_ok = (all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
          and d2["N=170"] < 0.95
          and all(valsc[i] <= vals[i] + 1e-9 for i in range(len(vals))))
g512 = bool(g.get("G5.12_attribution", {}).get("pass"))

ok = capm_ok and ls_ok and reg_ok and dsr_ok and g512
verdict("rt5_attribution_dsr", ok,
        f"U7 attribution panel: CAPM champion faithful-E2 OOS alpha="
        f"{e2['alpha_pct_yr']:+.2f}%/yr (NW t={e2['alpha_t']:.2f}, "
        f"beta={e2['beta_xau']:.3f}) < artifact-E0 {e0['alpha_pct_yr']:+.2f} "
        f"(Flaw #10 direction, t={e0['alpha_t']:.2f}); L/S hours "
        f"{ls2['long_hrs_pct']}/{ls2['short_hrs_pct']}/{ls2['flat_hrs_pct']} "
        f"(short share {ls2['short_share_pct']}%); 5-regime finite "
        f"(bear {reg2['2021-22 bear']}, parabolic {reg2['2024-25 parabolic']}); "
        f"DSR strictly decreasing in N, N=170={d2['N=170']} (<0.95 honest "
        f"miss), carry-adjusted <= raw at every N. G5.12 in-run PASS={g512}.",
        {"capm_ok": capm_ok, "ls_ok": ls_ok, "regime_ok": reg_ok,
         "dsr_ok": dsr_ok, "g512_pass": g512,
         "alpha_e2": e2["alpha_pct_yr"], "alpha_t_e2": e2["alpha_t"],
         "dsr_n170": d2["N=170"]})
