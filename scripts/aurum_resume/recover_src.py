#!/usr/bin/env python3
"""Reconstruct canonical src tree from the pulled v5-era Kaggle kernels.
- src/lib/*.py        <- lib sections of aurum-p2-train.py (newest build)
- src/kernels/*.py    <- kernel bodies of p1_data / p2a_check / p2_train / ds_diag
- updates src/build.py PULLED map to the fresh pulls (v5 era); p0 kernels are
  frozen historical artifacts (old lib era) and drop out of --verify.
"""
import os, re, hashlib

ROOT = "/home/z/my-project/kaggle_work/aurum"
SRC = os.path.join(ROOT, "src")
LIB = os.path.join(SRC, "lib")
KER = os.path.join(SRC, "kernels")
PULL = os.path.join(ROOT, "results", "src_pull")
os.makedirs(LIB, exist_ok=True); os.makedirs(KER, exist_ok=True)

MARK = re.compile(r"\n\n\n# ==================== (lib/(\S+?)\.py|kernel (\S+?)) ====================\n")

def split(path):
    src = open(path).read()
    marks = list(MARK.finditer(src))
    assert marks, f"no markers in {path}"
    preamble = src[:marks[0].start()].rstrip("\n")
    sections, kernels = {}, {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(src)
        body = src[m.end():end]
        body = body.rstrip("\n")
        if m.group(2):
            sections[m.group(2)] = body
        else:
            kernels[m.group(3)] = body
    return preamble, sections, kernels

# 1) libs from p2_train (newest)
pre, libs, kers = split(os.path.join(PULL, "p2-train", "aurum-p2-train.py"))
print("p2_train libs:", list(libs), "kernels:", list(kers))
assert list(libs) == ["aurum_gates", "aurum_feat", "aurum_label",
                      "aurum_data", "aurum_ser", "aurum_model"], "unexpected lib set"
for name, body in libs.items():
    with open(os.path.join(LIB, name + ".py"), "w") as f:
        f.write(body + "\n")
    print(f"  lib/{name}.py: {len(body)} B")

# 2) kernel bodies from each pull
pulls = {
    "p1_data": os.path.join(PULL, "p1-data", "aurum-p1-data.py"),
    "p2a_check": os.path.join(PULL, "p2a-check", "aurum-p2a-check.py"),
    "p2_train": os.path.join(PULL, "p2-train", "aurum-p2-train.py"),
}
for kname, path in pulls.items():
    _, secs, kers_ = split(path)
    # lib consistency check vs canonical
    for lname, body in secs.items():
        canon = open(os.path.join(LIB, lname + ".py")).read().rstrip("\n")
        if canon != body:
            print(f"  !! lib {lname} differs between {kname} and p2_train")
    kb = kers_[kname]
    with open(os.path.join(KER, kname + ".py"), "w") as f:
        f.write(kb + "\n")
    print(f"  kernels/{kname}.py: {len(kb)} B")

# ds_diag: tiny standalone, no markers — whole file is the body
dd = open(os.path.join(PULL, "ds-diag", "aurum-ds-diag.py")).read()
with open(os.path.join(KER, "ds_diag.py"), "w") as f:
    f.write(dd)
print(f"  kernels/ds_diag.py: {len(dd)} B (standalone, no lib markers)")

# 3) patch src/build.py PULLED map
bp = os.path.join(SRC, "build.py")
b = open(bp).read()
new_pulled = '''PULLED = {
    "p1_data": os.path.join(ROOT, "results", "src_pull", "p1-data", "aurum-p1-data.py"),
    "p2a_check": os.path.join(ROOT, "results", "src_pull", "p2a-check", "aurum-p2a-check.py"),
    "p2_train": os.path.join(ROOT, "results", "src_pull", "p2-train", "aurum-p2-train.py"),
}
# p0_core / p0_pilot are frozen Phase-0 artifacts built from the pre-v5 lib
# set; they are historical and excluded from the roundtrip check.'''
b = re.sub(r"PULLED = \{.*?\n\}", new_pulled, b, flags=re.S)
open(bp, "w").write(b)
print("build.py PULLED map updated")

# 4) roundtrip verify with the fresh tree
import sys
sys.path.insert(0, SRC)
import importlib.util
spec = importlib.util.spec_from_file_location("aurum_build", bp)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ok = mod.verify()
print("VERIFY:", "ALL BYTE-EXACT" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
