#!/usr/bin/env bash
# Start of every unit: bring the local checkout up to date with GitHub,
# which is the single source of truth.
# Usage: scripts/sync.sh
set -euo pipefail

cd "$(dirname "$0")/.."
git fetch --tags origin
git pull --ff-only origin main
git log --oneline -1
git describe --tags --abbrev=0 2>/dev/null || echo "(no unit tag yet)"
