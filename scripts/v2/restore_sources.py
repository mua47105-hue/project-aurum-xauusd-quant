#!/usr/bin/env python3
"""Extract module sections from pulled kernel bundles back into module files
(post-sandbox-recycle source restoration)."""
import os
import re
import sys

ROOT = "/home/z/my-project"

JOBS = [
    # (pulled kernel file, {section basename: output path})
    ("kaggle_work/v2/pull_aurum-v2-features/aurum-v2-features.py", {
        "f1_clock.py": "aurum_v2/aurum_features/f1_clock.py",
        "f2_realized.py": "aurum_v2/aurum_features/f2_realized.py",
        "f3_signatures.py": "aurum_v2/aurum_features/f3_signatures.py",
        "f4_toxicity.py": "aurum_v2/aurum_features/f4_toxicity.py",
        "f5_residual.py": "aurum_v2/aurum_features/f5_residual.py",
        "f6_trend.py": "aurum_v2/aurum_features/f6_trend.py",
        "build_v2_features.py": "aurum_v2/aurum_features/build_v2_features.py",
    }),
    ("kaggle_work/v2/pull_aurum-v2-cpu-models/aurum-v2-cpu-models.py", {
        "online_engine.py": "aurum_v2/model/online_engine.py",
        "rff_ridge.py": "aurum_v2/model/rff_ridge.py",
        "v2_common.py": "aurum_v2/model/v2_common.py",
    }),
]


def extract(kernel_path, mapping):
    src = open(kernel_path).read()
    lines = src.split("\n")
    sections = {}
    cur = None
    for ln in lines:
        m = re.match(r"# ==================== (\S+) ===================="
                     r"|# ==================== (\S+\.py) ====================", ln)
        if ln.startswith("# ==================== ") and ln.rstrip().endswith("==="):
            name = ln.strip("# ").strip("=").strip()
            if name in mapping:
                cur = name
                sections[cur] = []
                continue
            elif cur:
                cur = None
        if cur:
            sections[cur].append(ln)
    for name, out in mapping.items():
        if name not in sections:
            print(f"MISSING section {name} in {kernel_path}")
            continue
        body = "\n".join(sections[name]).strip() + "\n"
        os.makedirs(os.path.dirname(os.path.join(ROOT, out)), exist_ok=True)
        with open(os.path.join(ROOT, out), "w") as f:
            f.write(body)
        print(f"restored {out} ({len(body)//1024} KB)")


for kpath, mapping in JOBS:
    extract(os.path.join(ROOT, kpath), mapping)
print("done")
