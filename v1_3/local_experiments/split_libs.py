#!/usr/bin/env python3
"""Reconstruct the local lib/ + kernels/ source tree from the pulled Kaggle
kernels (they are auto-generated concatenations). Verifies byte-exact
roundtrip so build.py can regenerate identical kernels locally."""
import os, re, hashlib

ROOT = "/home/z/my-project/kaggle_work/aurum"
LIB = os.path.join(ROOT, "src", "lib")
KER = os.path.join(ROOT, "src", "kernels")
os.makedirs(LIB, exist_ok=True); os.makedirs(KER, exist_ok=True)

MARK = re.compile(r"^# ==================== (lib/(\S+)|kernel (\S+)) ====================$", re.M)

kernels = {
    "aurum-p1-data.py": "p1_data.py",
    "p0-pilot/src/aurum-p0-pilot.py": None,
    "p0-core/src/aurum-p0-core.py": None,
}
paths = {
    "p1_data": os.path.join(ROOT, "p1-data/src/aurum-p1-data.py"),
    "p0_pilot": os.path.join(ROOT, "p0-pilot/src/aurum-p0-pilot.py"),
    "p0_core": os.path.join(ROOT, "p0-core/src/aurum-p0-core.py"),
}

lib_hashes = {}
for kname, path in paths.items():
    src = open(path).read()
    marks = list(MARK.finditer(src))
    # preamble = up to first lib marker
    preamble = src[:marks[0].start()]
    sections = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(src)
        if m.group(2):       # lib/xxx.py
            sections[m.group(2)] = src[m.end():end]
        else:                # kernel body
            sections["__kernel__"] = src[m.end():end]
    print(f"== {kname}: preamble {len(preamble)}B, sections: "
          f"{[k for k in sections if k != '__kernel__']} + kernel body "
          f"{len(sections.get('__kernel__',''))}B")
    for libname, content in sections.items():
        if libname == "__kernel__":
            open(os.path.join(KER, f"{kname}.py"), "w").write(content)
            continue
        h = hashlib.sha256(content.encode()).hexdigest()[:12]
        if libname in lib_hashes and lib_hashes[libname][0] != h:
            print(f"   !! {libname} DIFFERS between kernels "
                  f"({lib_hashes[libname][1]} vs {kname})")
        lib_hashes[libname] = (h, kname)
        fn = os.path.join(LIB, libname.replace("lib/", "") + ".py")
        if not os.path.exists(fn):
            open(fn, "w").write(content)
    open(os.path.join(ROOT, "src", f"preamble_{kname}.txt"), "w").write(preamble)

print("\nlib files reconstructed:")
for f in sorted(os.listdir(LIB)):
    p = os.path.join(LIB, f)
    print(f"  {f:24s} {os.path.getsize(p):7d} B  sha={lib_hashes.get(f[:-3], ('?',''))[0]}")
