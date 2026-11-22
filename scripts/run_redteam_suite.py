#!/usr/bin/env python3
"""Master runner for the AURUM v2.1 Red-Team CI suite (Step 5).

Runs all 7 adversarial modules, captures their JSON verdicts from
artifacts/v2_1/redteam_ci/, and writes redteam_suite.json.  Exit code 1 iff
any module FAILS (SKIP does not fail the suite — it means the module's data
is absent in this environment; the suite is designed to run fully where the
restored data lives and to skip gracefully in data-less CI).
"""
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "artifacts", "v2_1", "redteam_ci")
MODULES = [
    ("rt1_causality_prefix", "test_rt1_causality_prefix.py"),
    ("rt2_label_shuffle", "test_rt2_label_shuffle.py"),
    ("rt3_m1_friction", "test_rt3_m1_friction.py"),
    ("rt4_bridge_fidelity", "test_rt4_bridge_fidelity.py"),
    ("rt5_attribution_dsr", "test_rt5_attribution_dsr.py"),
    ("rt6_carry_and_tiers", "test_rt6_carry_and_tiers.py"),
    ("rt7_z_quality_quarantine", "test_rt7_z_quality_quarantine.py"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {"suite": "redteam_ci", "spec": "v2.1.0",
               "written_utc": time.strftime("%Y-%m-%d %H:%M:%S"),
               "modules": {}, "n_pass": 0, "n_fail": 0, "n_skip": 0}
    for name, fn in MODULES:
        t0 = time.time()
        p = subprocess.run(
            [sys.executable, os.path.join(REPO, "tests", "redteam", fn)],
            capture_output=True, text=True, timeout=1800)
        vpath = os.path.join(OUT, name + ".json")
        v = json.load(open(vpath)) if os.path.exists(vpath) else {
            "verdict": "FAIL", "detail": p.stdout[-500:] + p.stderr[-500:]}
        v["elapsed_s"] = round(time.time() - t0, 1)
        v["exit_code"] = p.returncode
        with open(vpath, "w") as f:
            json.dump(v, f, indent=1)
        summary["modules"][name] = {"verdict": v["verdict"],
                                    "detail": v["detail"][:300],
                                    "elapsed_s": v["elapsed_s"]}
        vk = v["verdict"].lower()
        summary[f"n_{vk}"] = summary.get(f"n_{vk}", 0) + 1
        print(f"[{name}] {v['verdict'].upper()} ({v['elapsed_s']}s): "
              f"{v['detail'][:200]}")
    total = len(MODULES)
    ran = summary["n_pass"] + summary["n_fail"]
    summary["verdict"] = "FAIL" if summary["n_fail"] else (
        "PASS" if summary["n_pass"] == total else
        f"PASS_WITH_SKIPS ({summary['n_skip']}/{total} skipped — data absent)")
    with open(os.path.join(OUT, "redteam_suite.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nSUITE: {summary['verdict']}  "
          f"pass={summary['n_pass']} fail={summary['n_fail']} "
          f"skip={summary['n_skip']} (of {total})")
    sys.exit(1 if summary["n_fail"] else 0)


if __name__ == "__main__":
    main()
