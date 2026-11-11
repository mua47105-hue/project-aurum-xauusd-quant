#!/usr/bin/env python3
"""Local live-pathway smoke test: run live_daily.live_main against the REAL
model artifacts + real phase1-data parquets (CPU)."""
import os, sys, json

REF = "/home/z/my-project/kaggle_work/aurum/ref"
WORK = "/home/z/my-project/kaggle_work/aurum/smoke_live"
os.makedirs(WORK, exist_ok=True)
os.environ["AURUM_DS"] = REF
os.environ["AURUM_MODELS"] = os.path.join(REF, "models")
os.environ["AURUM_WORK"] = WORK

sys.path.insert(0, "/home/z/my-project/kaggle_work/aurum/src/kernels")
import live_daily as L

L.live_main(WORK)

print("\n=== GATES ===")
gates = json.load(open(f"{WORK}/gates.json"))
allok = True
for k, v in gates.items():
    print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} | {str(v['detail'])[:120]}")
    allok &= v["pass"]
dec = json.load(open(f"{WORK}/decision.json"))
print("\n=== DECISION ===")
print(json.dumps(dec["decision"], indent=1))
print(json.dumps(dec["sizing_reference"], indent=1))
print(json.dumps(dec["gates"], indent=1))
print("regime:", dec["heads_detail"]["regime_call"])
aud = json.load(open(f"{WORK}/risk_audit.json"))
print("\naudit keys:", list(aud["checklist"].keys())[:6], "...")
assert allok, "GATES FAILED"
print("\nLIVE SMOKE TEST PASSED")
