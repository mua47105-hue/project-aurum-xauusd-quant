#!/usr/bin/env python3
"""Build live_daily.py = p3 lib slice (everything before p3's main) + live tail."""
import re

SRC = "/home/z/my-project/kaggle_work/aurum/src/kernels/p3_tables.py"
TAIL = "/home/z/my-project/kaggle_work/aurum/src/kernels/_live_tail.py"
OUT = "/home/z/my-project/kaggle_work/aurum/src/kernels/live_daily.py"

src = open(SRC).read()
marker = "\ndef main(WORK):"
i = src.index(marker)
head = src[:i]
# strip the p3-only __main__ block is not in head; also drop the stale
# "Auto-generated AURUM kernel: lib/* + kernels/p3_tables.py" banner comment
head = head.replace(
    "# Auto-generated AURUM kernel: lib/* + kernels/p3_tables.py  (built 2026-11-08 14:50)",
    "# AURUM kernel: lib/* (shared with p3_tables) — sliced for live_daily", 1)
tail = open(TAIL).read()
banner = ("# ============================================================================\n"
          "# Auto-generated AURUM kernel: live_daily (Step 6a) = p3 lib + live tail\n"
          "# ============================================================================\n")
open(OUT, "w").write(banner + head + tail)
import py_compile
py_compile.compile(OUT, doraise=True)
n = len(open(OUT).readlines())
print(f"built {OUT}: {n} lines (head {len(head.splitlines())} + tail "
      f"{len(tail.splitlines())}); compile OK")
