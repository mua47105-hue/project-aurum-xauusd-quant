#!/usr/bin/env python3
"""Generic AURUM v2.0 kernel bundler (post-recycle rebuild, tolerant version)."""
import json
import os
import time

ROOT = "/home/z/my-project"
V2 = os.path.join(ROOT, "aurum_v2")

IMPORT_PREFIXES = (
    "from f1_clock import", "from f2_realized import",
    "from f3_signatures import", "from f4_toxicity import",
    "from f5_residual import", "from f6_trend import",
    "from build_v2_features import", "from losses import",
    "from continuum import", "from ae_mlp_dmn import",
    "from purge import", "from online_engine import",
    "from rff_ridge import", "from v2_common import",
    "from train_continuum import",
)


def strip_module(src: str, keep_main_guard: bool = False) -> str:
    lines = src.split("\n")
    out, skip, skip_paren = [], False, False
    for ln in lines:
        if ln.startswith('if __name__ == "__main__":') and not keep_main_guard:
            skip = True
            continue
        if skip:
            if ln.startswith((" ", "\t")) or ln.strip() == "":
                continue
            skip = False
        if ln.startswith("from __future__ import"):
            continue
        stripped = ln.lstrip()
        if any(stripped.startswith(p) for p in IMPORT_PREFIXES):
            if "(" in ln and ")" not in ln:
                skip_paren = True
            continue
        if skip_paren:
            if ")" in ln:
                skip_paren = False
            continue
        out.append(ln)
    return "\n".join(out)


def bundle(title, kernel_id, modules, main_path, out_dir, dataset_sources,
           enable_gpu, kernel_sources=None):
    parts = [f'''# ============================================================================
# Auto-generated AURUM v2.0 kernel: {title}
# (rebuilt {time.strftime("%Y-%m-%d %H:%M")})
# Spec v2.0 frozen. Do not edit on Kaggle - edit sources and rebuild.
# ============================================================================
from __future__ import annotations

import os, sys as _sys

''']
    for mod in modules:
        path = mod if os.path.isabs(mod) else os.path.join(V2, mod)
        with open(path) as f:
            src = f.read()
        parts.append(f"\n\n# ==================== {os.path.basename(path)} "
                     f"====================\n")
        parts.append(strip_module(src))
    with open(main_path) as f:
        main_src = f.read()
    parts.append("\n\n# ==================== kernel main ====================\n"
                 + strip_module(main_src, keep_main_guard=True))
    out = "".join(parts)
    os.makedirs(out_dir, exist_ok=True)
    kp = os.path.join(out_dir, "kernel.py")
    with open(kp, "w") as f:
        f.write(out)
    compile(out, kp, "exec")
    meta = {
        "id": kernel_id, "title": title, "code_file": "kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": enable_gpu, "enable_internet": True,
        "dataset_sources": dataset_sources,
        "kernel_sources": kernel_sources or [], "competition_sources": [],
    }
    with open(os.path.join(out_dir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"built {kp}: {len(out)/1024:.0f} KB (syntax OK)")
    return kp
