#!/usr/bin/env python3
"""Build the v2.1 Kaggle bundles (k6 = aurum-v2-backtest, k2 = aurum-v2-cpu-models)
from the patched modular sources, using the pulled v2.0.2 bundles as
templates (header + online_engine + v2_common/rff_ridge sections retained
verbatim — verified same vintage; the kernel-main section regenerated).
"""
import os
import py_compile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
K6_BUNDLE = os.path.join(REPO, "kaggle_kernels", "v2", "aurum-v2-backtest.py")
K2_BUNDLE = os.path.join(REPO, "kaggle_kernels", "v2", "aurum-v2-cpu-models.py")
PULLED = "/tmp/kpull"


def kernel_main_body(mod_path, strip_until, code_rev_old, code_rev_new):
    """v2: keep the FULL modular body (the try/except import shim is
    Kaggle-safe by design — v2_common/online_engine are inlined above; the
    stripping round lost `import traceback` -> NameError on Kaggle)."""
    src = open(mod_path).read()
    body = src
    body = body.replace(code_rev_old, code_rev_new, 1)
    return body


def build(bundle_path, pulled_path, mod_path, strip_marker, rev_old, rev_new):
    pulled = open(pulled_path).read()
    cut = pulled.index("# ==================== kernel main")
    head = pulled[:cut]
    banner = ("# ==================== kernel main (v2.1 — regenerated from the "
              "patched modular source) ====================\n")
    body = kernel_main_body(mod_path, strip_marker, rev_old, rev_new)
    # the modular shim (stripped above) defined _HERE for the local fallback
    body = ("_HERE = os.path.dirname(os.path.abspath(__file__))\n\n" + body)
    out = head + banner + body
    open(bundle_path, "w").write(out)
    py_compile.compile(bundle_path, doraise=True)
    n = len(out.splitlines())
    print(f"built {os.path.basename(bundle_path)}: {n} lines; compile OK")
    return out


# ---- k6: strip the modular import/shim block; keep from CODE_REV onward ----
k6_out = build(
    K6_BUNDLE, os.path.join(PULLED, "aurum-v2-backtest", "aurum-v2-backtest.py"),
    os.path.join(REPO, "aurum_v2", "kernels", "k6_backtest.py"),
    'CODE_REV = "k6-v2.0.2"', 'CODE_REV = "k6-v2.0.2"', 'CODE_REV = "k6-v2.1.0"')
assert "bridge_net" in k6_out and "G5.10_bridge_fidelity" in k6_out \
    and "G5.12_attribution" in k6_out, "k6 v2.1 content missing"

# ---- k2: strip the import shim; keep from the GATES/LOG section onward -----
k2_src = open(os.path.join(REPO, "aurum_v2", "kernels", "k2_cpu_models.py")).read()
# v2: keep the FULL modular body (imports included; the try/except import
# shim is Kaggle-safe — v2_common/rff_ridge/online_engine are inlined above)
k2_body = k2_src
pulled2 = open(os.path.join(PULLED, "aurum-v2-cpu-models",
                            "aurum-v2-cpu-models.py")).read()
cut = pulled2.index("# ==================== kernel main")
head2 = pulled2[:cut]
banner2 = ("# ==================== kernel main (v2.1 — regenerated from the "
           "patched modular source) ====================\n"
           "# v2.1 (U2/Flaw #9): z_gbdt = rff_z_v202(pred_gbdt); strict G4.8\n"
           "# on ALL producers; U3 quarantine ledger. See STATE.md.\n")
out2 = head2 + banner2 + k2_body
open(K2_BUNDLE, "w").write(out2)
py_compile.compile(K2_BUNDLE, doraise=True)
print(f"built aurum-v2-cpu-models.py: {len(out2.splitlines())} lines; compile OK")
assert "rff_z_v202(pred_gbdt)" in out2 and "G4.8_v21_strict_all_producers" in out2 \
    and "QUARANTINE_V21" in out2, "k2 v2.1 content missing"
print("both v2.1 bundles built")
