#!/usr/bin/env python3
"""Assemble src/kernels/p3_tables.py = p3 header + VERBATIM shared segments
from src/kernels/p2_train.py + the new p3 code (its __main__ ends the file).
Order matters: shared imports/constants/functions must precede the p3 code,
otherwise the __main__ block runs before names like argparse/prepare_data
exist (v1 lesson)."""
import os, py_compile, subprocess, sys, re

ROOT = "/home/z/my-project/kaggle_work/aurum"
P2 = os.path.join(ROOT, "src", "kernels", "p2_train.py")
NEW = "/home/z/my-project/scripts/aurum_resume/p3_new_code.py"
OUT = os.path.join(ROOT, "src", "kernels", "p3_tables.py")

lines = open(P2).read().splitlines(keepends=True)
def seg(a, b):
    return "".join(lines[a - 1:b])
# 1-based: 25-26 imports, 28 TRAIN_END_YEAR, 41 FOLD_YEARS, 43 CAL_FRAC,
#          51-228 _find_ds .. _mk_head_module
shared = seg(25, 26) + "\n" + seg(28, 28) + seg(41, 41) + seg(43, 43) + \
    "\n\n" + seg(51, 228)
for fn in ("def _find_ds", "def _ds", "def _trend_volc", "def _ser_drop",
           "def prepare_data", "def _mk_head_module"):
    assert fn in shared, fn
assert shared.rstrip().endswith("return HeadStack()")

new_code = open(NEW).read()
# split the p3 header comment (through the P3_TIMEOUT_S constant) from the rest
head_end = new_code.index("P3_TIMEOUT_S = ")
head_end = new_code.index("\n", head_end) + 1
p3_header, p3_body = new_code[:head_end], new_code[head_end:]
assert "if __name__" in p3_body

glue = (
    "\n\n# ---------------------------------------------------------------------------\n"
    "# Shared data/serialization/heads section - VERBATIM from kernels/p2_train.py\n"
    "# (any edit here MUST be mirrored in p2_train.py; byte-identical tokenization\n"
    "#  is what makes the p3 tables comparable to the p2-train cache/refits)\n"
    "# ---------------------------------------------------------------------------\n"
)
body = p3_header.rstrip("\n") + "\n" + glue + shared.rstrip("\n") + \
    "\n\n\n" + p3_body.rstrip("\n") + "\n"
with open(OUT, "w") as f:
    f.write(body)
py_compile.compile(OUT, doraise=True)
n = len(body.splitlines())
print(f"assembled {OUT}: {n} lines - syntax OK")

# __main__-order smoke: --help must resolve argparse and exit cleanly
r = subprocess.run([sys.executable, OUT, "--help"], capture_output=True, text=True,
                   timeout=120)
assert r.returncode == 0 and "--mode" in r.stdout, (r.returncode, r.stdout, r.stderr)
print("__main__ --help OK (names resolve before guard)")

r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "build.py"),
                    "--build", "p3_tables"], capture_output=True, text=True)
print(r.stdout.strip())
if r.returncode != 0:
    print(r.stderr[-3000:]); raise SystemExit(1)

# final built artifact: run --help too
r = subprocess.run([sys.executable, os.path.join(ROOT, "dist", "aurum-p3-tables.py"),
                    "--help"], capture_output=True, text=True, timeout=120)
assert r.returncode == 0 and "--mode" in r.stdout, (r.returncode, r.stderr[-500:])
print("built artifact --help OK")
