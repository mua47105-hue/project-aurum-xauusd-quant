#!/bin/bash
# AURUM secret-blocker installer — run once after every fresh clone:
#   bash scripts/install_secret_hook.sh
# Installs .git/hooks/pre-commit:
#   * STRICT patterns (KGAT_|ghp_|github_pat_|kaggle.json|access_token|GITHUB_TOKEN|
#     KAGGLE_KEY|KAGGLE_USERNAME) hard-abort the commit in any file NOT listed in
#     .secretscan-allow.
#   * HARD patterns (realistic full-length tokens) hard-abort in EVERY file,
#     including allowlisted ones — allowlisting tolerates prose/prefix mentions
#     only, never an actual credential.
# This repository is PUBLIC: tokens must never enter history.
# Store credentials outside the repo (/tmp/.creds/, ~/.kaggle/).
set -e
cd "$(dirname "$0")/.."
mkdir -p .git/hooks
cat > .git/hooks/pre-commit <<'HOOK'
#!/bin/bash
# AURUM v2.1 secret-blocker
set -u
STRICT='KGAT_|ghp_|github_pat_|kaggle\.json|access_token|GITHUB_TOKEN|KAGGLE_KEY|KAGGLE_USERNAME'
HARD='ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{60,}|KGAT_[A-Za-z0-9_-]{30,}|"key"\s*:\s*"[A-Za-z0-9]{30,}"'
ALLOW=".secretscan-allow"
FILES=$(git diff --cached --name-only --diff-filter=ACMR | grep -v '^\.git/' || true)
[ -z "$FILES" ] && exit 0
# build allowlist set (comments/blank lines ignored)
declare -A OK=()
if [ -f "$ALLOW" ]; then
  while IFS= read -r a; do
    case "$a" in ''|\#*) continue;; esac
    OK["$a"]=1
  done < "$ALLOW"
fi
rc=0
for f in $FILES; do
  [ -f "$f" ] || continue
  # the allowlist file itself is definitionally pattern documentation
  [ "$f" = "$ALLOW" ] && continue
  if grep -En "$HARD" "$f" 2>/dev/null; then
    echo "FATAL: full-length credential in $f (even allowlisted files may not contain real tokens) — COMMIT ABORTED"; rc=1
  elif [ -n "${OK[$f]:-}" ]; then
    continue
  elif grep -En "$STRICT" "$f" 2>/dev/null; then
    echo "FATAL: secret pattern in $f — COMMIT ABORTED (add to $ALLOW ONLY if it is pattern documentation)"; rc=1
  fi
done
exit $rc
HOOK
chmod +x .git/hooks/pre-commit
echo "pre-commit secret-blocker installed (strict + hard-token layers, allowlist=.secretscan-allow)."
