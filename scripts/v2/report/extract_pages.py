#!/usr/bin/env python3
"""Extract page numbers for each report heading from the rendered PDF and
write page_map.json for the second generation pass."""
import json
import re
import sys

from pypdf import PdfReader

PDF = "/home/z/my-project/download/AURUM_Spec_v2_Master_Execution_and_Tournament_Report.pdf"
OUT = "/home/z/my-project/scripts/v2/report/page_map.json"

ENTRIES = [
    "1. Executive Summary",
    "2. The Unified v2.0 Codebase",
    "2.1 The v2.0.1 revision",
    "3. Phase 1 — The Unified Feature Store (K1)",
    "4. Phase 2 — CPU Models and the Execution Engine (K2)",
    "5. Phase 3 — The Neural and Contractive-Loop Tournament (K3)",
    "6. Engineering Flaw #8 — The M2 RFF Fold-Scale Saturation",
    "6.1 Root cause",
    "6.2 Impact propagation and quarantine",
    "6.3 Remediation",
    "7. Phase 4 — The Full Walk-Forward Backtest and Statistical Gates (K6)",
    "7.1 The v2.0.1 causality regression",
    "7.2 The tournament",
    "7.3 The Grand Hybrid, decomposed",
    "7.4 Statistics — bootstrap, DSR, PBO",
    "8. Phase 5 — The Live v2.0 Decision System (K7)",
    "9. The Verdict — v1.2 vs v1.3 vs v2.0",
    "10. Statistical Honesty, Reconciliation and Limitations",
    "10.1 What the numbers are, and are not",
    "10.2 Known limitations",
    "10.3 Reconciliation of quoted vintages",
    "11. Appendices",
    "Appendix A — Reproduction map",
    "Appendix B — Locked constants and gate register",
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


reader = PdfReader(PDF)
page_texts = [norm(p.extract_text() or "") for p in reader.pages]

page_map, missing = {}, []
for e in ENTRIES:
    key = norm(e)[:22]
    found = None
    for i, pt in enumerate(page_texts):
        if key in pt:
            found = i + 1
            break
    if found is None:
        missing.append(e)
    else:
        page_map[e] = found

json.dump(page_map, open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"mapped {len(page_map)}/{len(ENTRIES)} headings -> {OUT}")
if missing:
    print("MISSING:", missing)
    sys.exit(1)
