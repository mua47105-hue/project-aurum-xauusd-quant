# Credentials pattern (READ BEFORE ADDING ANY SCRIPT THAT TALKS TO KAGGLE/GITHUB)

**No secret is ever committed to this repository.** The historical token files
(`token_tskdev`, `token_ranigain`) were intentionally excluded; the tokens they held were
rotated/revoked after the program concluded anyway.

## Pattern used everywhere in this repo

1. Tokens live **outside the repo**, e.g. `~/.aurum_token`, `~/.kaggle/kaggle.json`, or a
   shell env var.
2. Scripts read them from the environment at runtime:

   ```python
   token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
   gh    = os.environ.get("GITHUB_TOKEN")
   ```

3. The historical helper was a 4-line shell wrapper that cats an out-of-repo file:

   ```bash
   #!/bin/bash
   export KAGGLE_API_TOKEN=$(cat ~/.aurum_token)
   exec kaggle "$@"
   ```

## Kaggle API auth notes (2026-11 era)

- Basic auth (`-u user:token`) stopped working for some endpoints; **Bearer** auth works:
  `Authorization: Bearer $KAGGLE_API_TOKEN` (see `scripts/kgapi.py`).
- Kernel-output files are listed at `/api/v1/kernels/output?userName=…&kernelSlug=…`
  (paginated via `nextPageToken`); each file record carries a **signed URL** that needs no
  auth header.
- Dataset downloads redirect (`/api/v1/datasets/download/{owner}/{slug}`) — use
  `curl -L` and pass the token via a curl config file, not the command line.

## Mandatory pre-commit gate

```bash
grep -rnE "KGAT[_][A-Za-z0-9]+|ghp[_][A-Za-z0-9]+|github_pat[_][A-Za-z0-9_]+" \
  --include="*" . && echo "SECRET LEAK — ABORT" || echo "clean"
```

Must print `clean` with zero matches before every commit/push.
