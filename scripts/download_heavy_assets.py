#!/usr/bin/env python3
"""Project AURUM — heavy asset downloader with SHA-256 verification.

Restores every >50 MB binary (feature store, model weights, probability tables,
base price data, workspace backup) that lives in GitHub Release `v2.0.2` (primary)
or on Kaggle (fallback) into its exact expected local path, verifying SHA-256.

Usage:
    export GITHUB_TOKEN=<your-PAT>          # fine-grained PAT, Contents:read
    python3 scripts/download_heavy_assets.py            # all assets
    python3 scripts/download_heavy_assets.py features core_fp16
    python3 scripts/download_heavy_assets.py --list

Kaggle fallback (used automatically when GITHUB_TOKEN is absent or the GitHub
download fails, for the assets that exist on Kaggle):
    export KAGGLE_API_TOKEN=...            # tskdev token (Bearer-auth)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
GITHUB_OWNER = "mua47105-hue"
GITHUB_REPO = "project-aurum-xauusd-quant"
RELEASE_TAG = "v2.0.2"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASSETS = {
    # key: (release filename, sha256, relative dest path, kaggle source or None)
    "features": {
        "file": "aurum_v2_features.parquet",
        "sha256": "49cb60140396df57ce97f8636a98fce8d57bfbf1435d0b6f1f0f5c524c8c48ba",
        "dest": os.path.join("data", "aurum_v2_features.parquet"),
        "size_mb": 253.7,
        "kaggle": ("dataset_zip", "tskdev/aurum-v2-featstore"),
        "desc": "95,008 H1 bars x 375 causal v2 features, 2010-2025",
    },
    "core_fp16": {
        "file": "core_fp16.pt",
        "sha256": "8c1556fa3777081a1ba69bbf108d5741e8d78f9cacaaad01fb47d1ded97ecb33",
        "dest": os.path.join("models", "core_fp16.pt"),
        "size_mb": 125.9,
        "kaggle": ("kernel_file", "tskdev/aurum-p2-train", "models/core_fp16.pt"),
        "desc": "trained Looped Qwen3-0.6B core (fp16), Spec v1.3",
    },
    "prob_tables": {
        "file": "aurum_oos_prob_tables.parquet",
        "sha256": "0386810185f83c9236b903d8fd21b288e65d2dc47ef5ba0408ed07121af54fea",
        "dest": os.path.join("models", "aurum_oos_prob_tables.parquet"),
        "size_mb": 63.3,
        "kaggle": ("kernel_file", "tskdev/aurum-p3-tables", "aurum_oos_prob_tables.parquet"),
        "desc": "569,152-row x 4-depth OOS probability tables, Spec v1.3",
    },
    "phase1_data": {
        "file": "h1_xauusd_and_cross_assets.tar.gz",
        "sha256": "dfd55109cd514fffe02c29f8f6ab3ed457b5c1df624372422fbe2f26ad17123b",
        "dest": os.path.join("data", "phase1", "h1_xauusd_and_cross_assets.tar.gz"),
        "size_mb": 476.8,
        "kaggle": ("dataset_zip", "tskdev/aurum-phase1-data"),
        "desc": "full aurum-phase1-data v6: H1/M1 parquets (XAUUSD/EURUSD/USDJPY/XAGUSD), "
                "labels (R_actual/beff/evref/excursions/pwin), cohort_idx, codebooks, vocab",
        "extract_into": os.path.join("data", "phase1"),
    },
    "workspace_backup": {
        "file": "aurum_v2_workspace_backup.tar.gz",
        "sha256": "2cc4fcd738b8aa0ee07fc5068fe167b91f0c72942fb7da48e043ab3e788502c4",
        "dest": os.path.join("backup", "aurum_v2_workspace_backup.tar.gz"),
        "size_mb": 307.7,
        "kaggle": ("dataset_zip", "tskdev/aurum-v2-workspace-backup"),
        "desc": "complete final workspace snapshot (sources + outputs), also mirrored at "
                "Kaggle dataset tskdev/aurum-v2-workspace-backup",
    },
}


def gh_token():
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def kg_token():
    return os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch(url, dest, headers=None, timeout=3600):
    """Stream download with curl (follows redirects), token passed via header file."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    args = ["curl", "-sS", "-L", "--retry", "3", "--max-time", str(timeout),
            "-o", tmp, url]
    hdr = None
    if headers:
        fd, hdr = tempfile.mkstemp(prefix=".hdr_")
        with os.fdopen(fd, "w") as f:
            for k, v in headers.items():
                f.write('header = "%s: %s"\n' % (k, v))
        args += ["-K", hdr]
    try:
        subprocess.run(args, check=True)
        if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            raise RuntimeError("empty download")
        os.replace(tmp, dest)
    finally:
        if hdr and os.path.exists(hdr):
            os.unlink(hdr)


def github_asset_url(session_file):
    """Resolve {filename: browser_download_url} for release RELEASE_TAG."""
    tok = gh_token()
    req = urllib.request.Request(
        f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/tags/{RELEASE_TAG}",
        headers={"Authorization": f"Bearer {tok}",
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(req, timeout=60) as r:
        rel = json.load(r)
    with open(session_file, "w") as f:
        json.dump({a["name"]: a for a in rel.get("assets", [])}, f)
    return rel


def kaggle_fetch(spec, dest):
    """Fetch an asset from Kaggle (dataset zip or kernel-output file)."""
    tok = kg_token()
    if not tok:
        raise RuntimeError("KAGGLE_API_TOKEN/KAGGLE_KEY not set")
    hdr = {"Authorization": "Bearer " + tok}
    kind = spec[0]
    if kind == "dataset_zip":
        ref = spec[1]
        # dataset zips contain all files; we need the one matching our target name
        zipd = tempfile.mkdtemp(prefix="kgzip_")
        zpath = os.path.join(zipd, "ds.zip")
        fetch(f"https://www.kaggle.com/api/v1/datasets/download/{ref}", zpath, hdr)
        wanted = os.path.basename(dest)
        with zipfile.ZipFile(zpath) as z:
            cand = [n for n in z.namelist() if os.path.basename(n) == wanted]
            if not cand:
                raise RuntimeError(f"{wanted} not inside dataset zip {ref}: {z.namelist()[:8]}...")
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            with z.open(cand[0]) as src, open(dest, "wb") as out:
                subprocess.run(["true"])  # noop; stream below
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
    elif kind == "kernel_file":
        _, kref, fname = spec
        owner, slug = kref.split("/")
        url = ("https://www.kaggle.com/api/v1/kernels/output?userName="
               + owner + "&kernelSlug=" + slug)
        req = urllib.request.Request(url, headers=hdr)
        with urllib.request.urlopen(req, timeout=120) as r:
            files = json.load(r).get("files") or []
        match = [f for f in files
                 if (f.get("fileNameNullable") or f.get("fileName")) == fname]
        if not match:
            raise RuntimeError(f"{fname} not in kernel output {kref}")
        signed = match[0].get("urlNullable") or match[0].get("url")
        fetch(signed, dest)  # signed URL: no auth header
    else:
        raise RuntimeError("unknown kaggle spec " + kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("assets", nargs="*", help="keys: " + " ".join(ASSETS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download even if hash matches")
    args = ap.parse_args()

    if args.list or not args.assets:
        print(f"Release {GITHUB_OWNER}/{GITHUB_REPO} {RELEASE_TAG} — heavy assets:")
        for k, a in ASSETS.items():
            print(f"  {k:18s} {a['size_mb']:7.1f} MB  -> {a['dest']}")
            print(f"  {'':18s} {a['desc']}")
        if not args.assets:
            print("\n(no keys given — nothing downloaded; pass keys or 'all')")
        return

    keys = list(ASSETS) if "all" in args.assets else args.assets
    unknown = [k for k in keys if k not in ASSETS]
    if unknown:
        sys.exit("unknown asset key(s): " + ", ".join(unknown))

    session_file = tempfile.mktemp(prefix=".gh_assets_")
    gh_map = {}
    if gh_token():
        try:
            rel = github_asset_url(session_file)
            print(f"GitHub release '{rel.get('name', RELEASE_TAG)}': "
                  f"{len(rel.get('assets', []))} assets visible")
            gh_map = json.load(open(session_file))
        except Exception as e:
            print(f"[warn] GitHub release lookup failed ({e}); will try Kaggle fallback")
    else:
        print("[info] GITHUB_TOKEN not set — Kaggle fallback only")

    failures = []
    for key in keys:
        a = ASSETS[key]
        dest = os.path.join(REPO_ROOT, a["dest"])
        if os.path.exists(dest) and not args.force:
            actual = sha256_of(dest)
            if actual == a["sha256"]:
                print(f"[skip] {key}: {dest} already present and verified")
                continue
            print(f"[warn]  {key}: exists but hash mismatch — re-downloading")
        print(f"[get ] {key}: {a['file']} ({a['size_mb']} MB) -> {a['dest']}")
        ok = False
        # --- primary: GitHub Release asset
        if a["file"] in gh_map and gh_token():
            url = gh_map[a["file"]].get("browser_download_url")
            if not url:  # private repo: use the API asset URL with octet-stream
                url = gh_map[a["file"]]["url"]
            try:
                fetch(url, dest, headers={
                    "Authorization": f"Bearer {gh_token()}",
                    "Accept": "application/octet-stream"})
                ok = True
            except Exception as e:
                print(f"[warn]  GitHub download failed: {e}")
        # --- fallback: Kaggle
        if not ok and a.get("kaggle") and kg_token():
            try:
                print(f"[info] falling back to Kaggle for {key}")
                kaggle_fetch(a["kaggle"], dest)
                ok = True
            except Exception as e:
                print(f"[warn]  Kaggle fallback failed: {e}")
        if not ok:
            failures.append(key)
            print(f"[FAIL] {key}: could not download from any source")
            continue
        # --- verify
        actual = sha256_of(dest)
        if actual != a["sha256"]:
            failures.append(key)
            print(f"[FAIL] {key}: SHA-256 mismatch\n"
                  f"       expected {a['sha256']}\n"
                  f"       actual   {actual}")
            os.remove(dest)
            continue
        print(f"[ ok ] {key}: SHA-256 verified ({a['sha256'][:16]}…)")
        # --- optional extraction (phase1 tarball stays as tar.gz; document contents)
        if "extract_into" in a:
            print(f"[info] {key}: tarball left at {a['dest']} — extract with: "
                  f"tar xzf {os.path.relpath(dest, REPO_ROOT)} -C "
                  f"{os.path.relpath(os.path.join(REPO_ROOT, a['extract_into']), REPO_ROOT)}")

    if os.path.exists(session_file):
        os.unlink(session_file)
    print()
    if failures:
        sys.exit("FAILED assets: " + ", ".join(failures))
    print("All requested heavy assets present and SHA-256 verified.")


if __name__ == "__main__":
    main()
