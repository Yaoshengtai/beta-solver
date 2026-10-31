#!/usr/bin/env bash
# One-time setup: point this checkout at an empty GitHub repo and push main.
# Usage: scripts/first_push.sh <repo-url>
# Uses the git credentials already configured on this machine; stores none.
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 <repo-url>" >&2
  exit 1
fi

cd "$(dirname "$0")/.."
[ -d .git ] || git init -b main
git add -A
git diff --cached --quiet || git commit -m "Initial commit"
git remote get-url origin >/dev/null 2>&1 || git remote add origin "$1"
git push -u origin main --tags
echo "Pushed. Now enable GitHub Pages (Settings > Pages > Source: GitHub Actions)."
