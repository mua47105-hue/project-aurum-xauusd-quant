#!/usr/bin/env python3
"""AURUM pipeline watcher: polls p3; on COMPLETE pulls output, verifies G4 gates,
pushes p4; polls p4; pulls p4 output. Logs everything. On ERROR pulls log tail."""
import json, os, subprocess, sys, time

ROOT = "/home/z/my-project/kaggle_work/aurum"
LOG = open(f"{ROOT}/watcher.log", "a", buffering=1)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, file=LOG, flush=True)

def sh(*args, timeout=900):
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip()

def status(kernel):
    rc, out = sh("kaggle", "kernels", "status", f"tskdev/{kernel}", timeout=60)
    return out

def pull(kernel, dest):
    os.makedirs(dest, exist_ok=True)
    rc, out = sh("kaggle", "kernels", "output", f"tskdev/{kernel}", "-p", dest, timeout=1800)
    return rc == 0, out

def push(path):
    rc, out = sh("kaggle", "kernels", "push", "-p", path, timeout=300)
    return rc == 0, out

def wait_complete(kernel, max_hours=6.0):
    t0 = time.time()
    last = ""
    while time.time() - t0 < max_hours * 3600:
        s = status(kernel)
        if s != last:
            log(f"{kernel}: {s}")
            last = s
        if "COMPLETE" in s:
            return "COMPLETE"
        if "ERROR" in s:
            return "ERROR"
        if "CANCEL" in s:
            return "CANCELLED"
        time.sleep(120)
    return "TIMEOUT"

# ---- stage 1: p3 ----
log("=== watching aurum-p3-tables ===")
st = wait_complete("aurum-p3-tables")
log(f"p3 final status: {st}")
if st != "COMPLETE":
    ok, out = pull("aurum-p3-tables", f"{ROOT}/logs/p3_v5_error")
    log(f"p3 error-log pull ok={ok}")
    sys.exit(1)

ok, out = pull("aurum-p3-tables", f"{ROOT}/results/p3_tables_latest")
log(f"p3 output pull ok={ok}: {out[-200:] if out else ''}")

# verify G4 gates
gates_path = f"{ROOT}/results/p3_tables_latest/gates.json"
try:
    g = json.load(open(gates_path))
    allpass = all(v.get("pass") for v in g.values())
    log(f"p3 GATES: {json.dumps({k: v.get('pass') for k, v in g.items()})}")
    if not allpass:
        log("p3 GATES FAILED - stopping for manual diagnosis")
        sys.exit(1)
except Exception as e:
    log(f"p3 gates.json unreadable: {e}")
    sys.exit(1)

# ---- stage 2: push p4 ----
log("=== pushing aurum-p4-backtest ===")
ok, out = push(f"{ROOT}/push/p4")
log(f"p4 push ok={ok}: {out[-200:]}")
if not ok:
    sys.exit(1)

st = wait_complete("aurum-p4-backtest", max_hours=3.0)
log(f"p4 final status: {st}")
if st != "COMPLETE":
    ok, out = pull("aurum-p4-backtest", f"{ROOT}/logs/p4_v1_error")
    log(f"p4 error-log pull ok={ok}")
    sys.exit(1)

ok, out = pull("aurum-p4-backtest", f"{ROOT}/results/p4_backtest_latest")
log(f"p4 output pull ok={ok}: {out[-300:] if out else ''}")
try:
    g = json.load(open(f"{ROOT}/results/p4_backtest_latest/gates.json"))
    log(f"p4 GATES: {json.dumps({k: v.get('pass') for k, v in g.items()})}")
except Exception as e:
    log(f"p4 gates unreadable: {e}")
log("=== PIPELINE COMPLETE - ready for report ===")
