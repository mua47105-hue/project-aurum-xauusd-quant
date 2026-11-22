"""Shared helpers for the AURUM v2.1 red-team CI modules.

Each module is standalone (python3 tests/redteam/test_rtN_*.py) and writes a
machine-readable verdict to artifacts/v2_1/redteam_ci/.  Modules that need
heavy data SKIP (verdict "skip", exit 0) when the data is absent so the suite
runs green in data-less CI contexts (GitHub Actions runs the unit tests and
the secret scan; the full red-team suite runs where the data lives).
"""
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, "artifacts", "v2_1", "redteam_ci")


def verdict(module, passed, detail, metrics=None, skipped=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    v = {"module": module,
         "verdict": "skip" if skipped else ("pass" if passed else "FAIL"),
         "detail": str(detail)[:4000],
         "metrics": metrics or {},
         "elapsed_s": 0,
         "written_utc": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = os.path.join(OUT_DIR, module + ".json")
    with open(path, "w") as f:
        json.dump(v, f, indent=1, default=float)
    tag = "SKIP" if skipped else ("PASS" if passed else "FAIL")
    print(f"[{module}] {tag}: {detail}")
    if not skipped and not passed:
        sys.exit(1)
    sys.exit(0)


def have(*relpaths):
    return all(os.path.exists(os.path.join(REPO, p)) for p in relpaths)


def load_json(path):
    with open(os.path.join(REPO, path)) as f:
        return json.load(f)
