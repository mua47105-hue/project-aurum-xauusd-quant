#!/usr/bin/env python3
"""Build AURUM Kaggle kernels: concat lib/* + kernels/<name>.py -> out/<name>/kernel.py
plus kernel-metadata.json. Strips `from aurum_* import` lines (flat namespace)."""
import json, os, re, sys, time, datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(ROOT, "lib")

KERNELS = {
    "p0_core":  {"slug": "aurum-p0-core",  "gpu": True,  "title": "AURUM P0 Core",  "timeout": 14400},
    "p0_pilot": {"slug": "aurum-p0-pilot", "gpu": True,  "title": "AURUM P0 Pilot", "timeout": 14400},
    "p1_data":  {"slug": "aurum-p1-data",  "gpu": False, "title": "AURUM P1 Data",  "timeout": 39600, "sources": ["tskdev/aurum-p1-data"]},
}
LIB_ORDER = ["aurum_gates.py", "aurum_feat.py", "aurum_label.py",
             "aurum_data.py", "aurum_ser.py", "aurum_model.py"]

HEADER = '''# ============================================================================
# Auto-generated AURUM kernel: lib/* + kernels/{name}.py  (built {ts})
# Spec v1.1 frozen. Do not edit on Kaggle - edit sources and rebuild.
# ============================================================================
import os as _os, sys as _sys
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _pin_transformers():
    """Qwen3 needs transformers>=4.51; 5.x changed model APIs and the Kaggle
    image ships a partially broken 5.20.2 (keras_nlp backend import error).
    Force-install 4.57.1 with --no-deps (bypasses the resolver refusing the
    downgrade), verify, and purge cached modules."""
    import subprocess as _sp
    ok = False
    try:
        import transformers
        from packaging.version import Version
        ok = Version("4.51.0") <= Version(transformers.__version__) < Version("5.0.0")
    except Exception:
        ok = False
    if ok:
        return
    try:
        _sp.run([_sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
                 "-q", "transformers==4.57.1"], check=True, timeout=900,
                capture_output=True)
        _sp.run([_sys.executable, "-m", "pip", "install", "--no-deps", "-q",
                 "tokenizers==0.22.0", "huggingface-hub==0.36.2"],
                check=False, timeout=900, capture_output=True)
    except Exception as _e:
        print("transformers pin install error:", _e, flush=True)
    for _mod in list(_sys.modules):
        if _mod.startswith(("transformers", "tokenizers", "huggingface_hub")):
            del _sys.modules[_mod]
    import transformers
    from packaging.version import Version as _V
    _v = Version(transformers.__version__)
    assert _V("4.51.0") <= _v < _V("5.0.0"), (
        f"transformers pin FAILED, got {_v}; Qwen3 layer API unverified on this version")

_pin_transformers()
'''

IMPORT_START = re.compile(r"^\s*(from\s+aurum_\w+\s+import|import\s+aurum_\w+)")

def strip_aurum_imports(src):
    """Remove complete (possibly multi-line, possibly indented) import
    statements that reference aurum_* modules. Handles parenthesized and
    backslash continuations."""
    lines = src.split("\n")
    out = []
    i = 0
    while i < len(lines):
        if IMPORT_START.match(lines[i]):
            depth = lines[i].count("(") - lines[i].count(")")
            cont = lines[i].rstrip().endswith("\\")
            while (depth > 0 or cont) and i + 1 < len(lines):
                i += 1
                depth += lines[i].count("(") - lines[i].count(")")
                cont = lines[i].rstrip().endswith("\\")
            i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)

def build(name):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [HEADER.replace("{name}", name).replace("{ts}", ts)]
    for lf in LIB_ORDER:
        src = open(os.path.join(LIB, lf)).read()
        src = strip_aurum_imports(src)
        parts.append(f"\n\n# ==================== lib/{lf} ====================\n{src}")
    k = open(os.path.join(ROOT, "kernels", name + ".py")).read()
    k = strip_aurum_imports(k)
    parts.append(f"\n\n# ==================== kernel {name} ====================\n{k}")
    out_dir = os.path.join(ROOT, "out", name)
    os.makedirs(out_dir, exist_ok=True)
    code = "".join(parts)
    with open(os.path.join(out_dir, "kernel.py"), "w") as f:
        f.write(code)
    cfg = KERNELS[name]
    meta = {"id": f"tskdev/{cfg['slug']}", "title": cfg["title"],
            "code_file": "kernel.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": cfg["gpu"], "enable_internet": True,
            "dataset_sources": [], "kernel_sources": cfg.get("sources", []),
            "competition_sources": []}
    with open(os.path.join(out_dir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"built {name}: {len(code.splitlines())} lines -> {out_dir}/kernel.py")
    return out_dir

if __name__ == "__main__":
    names = sys.argv[1:] or list(KERNELS)
    for nm in names:
        build(nm)
