#!/usr/bin/env python3
"""RT6 — overnight swap carry + risk-tier torture (audit A.6, U5).

Loads the p4 v2.1 tier matrix and verifies the accounting invariants and the
exact regression references:
  * un-throttled MaxDD 23.92% (exact audit reference) > throttled 13.06%;
  * pilot 0.5% un-throttled MaxDD ~12.54% (exact) < standard un-throttled;
  * base-carry Sharpe < no-carry (the audit's ~-0.3 SR carry drag), punitive
    < base;
  * carry cannot INFLATE Sharpe for the champion (long-biased book pays);
  * ex-2023 (2024-25) sensitivity rows present with and without carry.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rt_common import verdict, have, load_json  # noqa: E402

NEED = ["artifacts/v2_1/p4_out/v21_summary.json"]
if not have(*NEED):
    verdict("rt6_carry_and_tiers", False, "SKIP: p4 v21_summary absent",
            skipped=True)

v = load_json("artifacts/v2_1/p4_out/v21_summary.json")
tm = v["tier_matrix_flagship_cap3_oos"]
ex = v["ex2023_sensitivity_2024_25"]

std_thr = tm["standard|throttle15|no_carry"]
std_un = tm["standard|unthrottled|no_carry"]
std_cb = tm["standard|throttle15|carry_base"]
std_cp = tm["standard|throttle15|carry_punitive"]
pil_un = tm["institutional_pilot|unthrottled|no_carry"]

checks = {
    "unthrottled_maxdd_23p92_exact": abs(std_un["max_dd_pct"] - 23.92) < 0.05,
    "unthrottled_dd_gt_throttled": std_un["max_dd_pct"] > std_thr["max_dd_pct"],
    "throttled_13p06_exact": abs(std_thr["max_dd_pct"] - 13.06) < 0.05,
    "pilot_unthrottled_12p54_exact": abs(pil_un["max_dd_pct"] - 12.54) < 0.05,
    "pilot_dd_lt_standard": pil_un["max_dd_pct"] < std_un["max_dd_pct"],
    "carry_base_drag": 0.15 < (std_thr["sharpe"] - std_cb["sharpe"]) < 0.60,
    "carry_punitive_lt_base": std_cp["sharpe"] < std_cb["sharpe"],
    "ex2023_rows_with_and_without_carry": (
        "standard|throttle15|no_carry" in ex
        and "standard|throttle15|carry_base" in ex),
    "ex2023_sharpe_gt_full_oos": ex["standard|throttle15|no_carry"]["sharpe"]
    > std_thr["sharpe"],
}
ok = all(checks.values())
verdict("rt6_carry_and_tiers", ok,
        f"U5 tier/carry matrix: throttled {std_thr['max_dd_pct']:.2f}% vs "
        f"un-throttled {std_un['max_dd_pct']:.2f}% (audit 23.92% EXACT); "
        f"pilot-0.5% un-throttled {pil_un['max_dd_pct']:.2f}% (audit 12.54% "
        f"EXACT); carry drag {std_thr['sharpe'] - std_cb['sharpe']:+.3f} SR "
        f"(audit est. -0.3), punitive drag "
        f"{std_thr['sharpe'] - std_cp['sharpe']:+.3f}; ex-2023 (2024-25) "
        f"SR {ex['standard|throttle15|no_carry']['sharpe']:.3f} > full-OOS "
        f"{std_thr['sharpe']:.3f} (2023 is the degraded year, Flaw #12). "
        f"Checks: {json.dumps(checks)}",
        {"checks": checks,
         "unthrottled_maxdd": std_un["max_dd_pct"],
         "pilot_unthrottled_maxdd": pil_un["max_dd_pct"],
         "carry_drag_sharpe": round(std_thr["sharpe"] - std_cb["sharpe"], 3)})
