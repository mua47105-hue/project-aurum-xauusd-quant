#!/usr/bin/env python3
"""Kaggle API v1 helper with Bearer-token auth (no secrets stored here).

Usage: export KAGGLE_API_TOKEN=$(cat scripts/creds/token_tskdev)
       python3 scripts/kgapi.py <command> [args]

Commands:
  ds-files <owner>/<slug>              list all dataset files (paginated)
  ds-download <owner>/<slug> <file> <dest>
  ds-download-all <owner>/<slug> <destdir>
  k-files <owner>/<slug>               list kernel output files (paginated)
  k-download-all <owner>/<slug> <destdir>   via signed urls
  k-pull <owner>/<slug> <dest.py>      pull latest kernel source
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.parse

BASE = "https://www.kaggle.com/api/v1"


def token():
    t = os.environ.get("KAGGLE_API_TOKEN")
    if not t:
        sys.exit("KAGGLE_API_TOKEN env var not set")
    return t.strip()


def api(path, params=None, timeout=180):
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token()})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def k_files(ref):
    owner, slug = ref.split("/")
    out, page_tok = [], None
    while True:
        params = {"userName": owner, "kernelSlug": slug}
        if page_tok:
            params["pageToken"] = page_tok
        d = api("/kernels/output", params)
        for f in d.get("files") or []:
            out.append({"name": f.get("fileNameNullable") or f.get("fileName") or "",
                        "url": f.get("urlNullable") or f.get("url") or ""})
        page_tok = d.get("nextPageToken")
        if not page_tok:
            break
    return out


def ds_files(ref, page_size=100):
    owner, slug = ref.split("/")
    out, page_tok = [], None
    while True:
        params = {"pageToken": page_tok} if page_tok else None
        d = api(f"/datasets/list/{owner}/{slug}", params)
        for f in d.get("datasetFiles") or []:
            out.append({"name": f.get("nameNullable") or f.get("name"),
                        "size": f.get("totalBytesNullable") or f.get("totalBytes") or 0})
        page_tok = d.get("nextPageTokenNullable")
        if not page_tok:
            break
    return out


def download(url, dest, retries=4, use_auth=True):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  [skip exists] {dest}")
        return
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    hdr = os.path.join("/tmp", ".kg_hdr_" + str(os.getpid()))
    try:
        for i in range(retries):
            try:
                args = ["curl", "-sS", "-L", "--max-time", "3600", "--retry", "2",
                        "-o", dest + ".part", url]
                if use_auth:
                    with open(hdr, "w") as f:
                        f.write('header = "Authorization: Bearer %s"\n' % token())
                    args += ["-K", hdr]
                subprocess.run(args, check=True)
                if os.path.getsize(dest + ".part") == 0:
                    raise RuntimeError("empty download")
                os.replace(dest + ".part", dest)
                print(f"  downloaded {dest} ({os.path.getsize(dest):,} B)")
                return
            except Exception as e:
                print(f"  retry {i+1} for {dest}: {e}")
                time.sleep(5)
        raise RuntimeError(f"failed to download {url}")
    finally:
        if os.path.exists(hdr):
            os.unlink(hdr)


def main():
    cmd = sys.argv[1]
    if cmd == "ds-files":
        for f in ds_files(sys.argv[2]):
            print(f"{f['name']}\t{f['size']}")
    elif cmd == "ds-download":
        ref, fname, dest = sys.argv[2], sys.argv[3], sys.argv[4]
        download(f"{BASE}/datasets/download/{ref}/{urllib.parse.quote(fname)}", dest)
    elif cmd == "ds-download-all":
        ref, destdir = sys.argv[2], sys.argv[3]
        files = ds_files(ref)
        print(f"{ref}: {len(files)} files")
        for f in files:
            download(f"{BASE}/datasets/download/{ref}/{urllib.parse.quote(f['name'])}",
                     os.path.join(destdir, f["name"]))
    elif cmd == "ds-download-zip":
        # Download entire dataset as zip and extract to destdir
        ref, destdir = sys.argv[2], sys.argv[3]
        import zipfile
        zip_path = os.path.join("/tmp", ref.replace("/", "__") + ".zip")
        download(f"{BASE}/datasets/download/{ref}", zip_path)
        os.makedirs(destdir, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(destdir)
        print(f"extracted {len(os.listdir(destdir))} top-level entries -> {destdir}")
    elif cmd == "k-files":
        for f in k_files(sys.argv[2]):
            print(f["name"])
    elif cmd == "k-download-all":
        ref, destdir = sys.argv[2], sys.argv[3]
        files = k_files(ref)
        print(f"{ref}: {len(files)} output files")
        for f in files:
            if not f["url"]:
                print(f"  [no url] {f['name']}")
                continue
            download(f["url"], os.path.join(destdir, f["name"]), use_auth=False)
    elif cmd == "k-pull":
        ref, dest = sys.argv[2], sys.argv[3]
        owner, slug = ref.split("/")
        d = api("/kernels/pull", {"userName": owner, "kernelSlug": slug})
        src = d.get("blob", {}).get("source", "")
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        open(dest, "w").write(src)
        ver = d.get("metadata", {}).get("currentVersionNumberNullable")
        print(f"pulled {ref} v{ver} -> {dest} ({len(src)} bytes)")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
