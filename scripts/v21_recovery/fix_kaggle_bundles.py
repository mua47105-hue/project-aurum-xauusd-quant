#!/usr/bin/env python3
"""Kaggle bundle fixes (round 2):
1. p4 + live bundles: embed the frozen policy JSON as an in-kernel fallback
   (on Kaggle the script runs from /kaggle/src — the repo-relative
   v1_3/config/frozen_policy.json does not exist there; the file remains the
   source of truth in the repo, the embed is byte-identical).
2. build_v21_bundles.py: keep the FULL modular body (its try/except import
   shim is Kaggle-safe by design) — the stripped cut lost `import traceback`
   (NameError in the stage context manager).
"""
import json

POLICY = open("v1_3/config/frozen_policy.json").read()
# sanity: the embedded text must parse
json.loads(POLICY)


def rep_file(path, old, new):
    s = open(path).read()
    assert s.count(old) == 1, f"anchor mismatch in {path}: {old[:80]!r}"
    open(path, "w").write(s.replace(old, new))
    print(f"patched {path}")


# ---- p4: embed fallback in load_frozen_policy --------------------------------
rep_file(
    "kaggle_kernels/v1_3/aurum-p4-backtest.py",
    '''def load_frozen_policy(path=V21_FROZEN_POLICY):
    """Flaw #11 fix: load the frozen flagship policy instead of re-deriving it
    from near-tied FIT grids (F6 margin was 1.27%). Returns (fp, drift) where
    drift is filled later by refit telemetry."""
    with open(path) as f:
        fp = json.load(f)''',
    '''_EMBEDDED_FROZEN_POLICY = r"""{POLICY}"""


def load_frozen_policy(path=V21_FROZEN_POLICY):
    """Flaw #11 fix: load the frozen flagship policy instead of re-deriving it
    from near-tied FIT grids (F6 margin was 1.27%). On Kaggle the repo file
    is not mounted — the byte-identical embedded copy is the fallback."""
    import os as _os
    if not _os.path.isfile(path):
        path = None
    if path is None:
        fp = json.loads(_EMBEDDED_FROZEN_POLICY)
    else:
        with open(path) as f:
            fp = json.load(f)'''.replace("{POLICY}", POLICY))

# ---- live: same fallback in _load_frozen_live --------------------------------
rep_file(
    "kaggle_kernels/v1_3/aurum-live-daily.py",
    '''def _load_frozen_live():
    """Load the frozen policy and verify the LIVE constants match (G6.0)."""
    with open(V21_FROZEN_POLICY_PATH) as f:
        fp = json.load(f)''',
    '''_EMBEDDED_FROZEN_POLICY = r"""{POLICY}"""


def _load_frozen_live():
    """Load the frozen policy and verify the LIVE constants match (G6.0).
    On Kaggle the repo file is not mounted — the byte-identical embedded
    copy is the fallback."""
    if os.path.isfile(V21_FROZEN_POLICY_PATH):
        with open(V21_FROZEN_POLICY_PATH) as f:
            fp = json.load(f)
    else:
        fp = json.loads(_EMBEDDED_FROZEN_POLICY)'''.replace("{POLICY}", POLICY))

print("frozen-policy embeds installed")
