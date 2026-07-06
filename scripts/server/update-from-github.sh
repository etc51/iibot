#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOCK_FILE="/tmp/samosbor-updater.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "updater is already running"
  exit 0
fi

current_commit="$(git rev-parse HEAD)"
branch="$(git branch --show-current)"
if [[ -z "$branch" ]]; then
  branch="master"
fi
git fetch origin "$branch"
remote_commit="$(git rev-parse "origin/$branch")"

if [[ "$current_commit" == "$remote_commit" ]]; then
  echo "already up to date: $current_commit"
  exit 0
fi

git pull --ff-only origin "$branch"
source "$ROOT_DIR/.venv/bin/activate"
python -m pip install -r requirements-tbank.txt
python -m pip install -e .
python -m unittest discover -s tests -v
"$ROOT_DIR/scripts/server/install-server.sh"

echo "updated to $(git rev-parse --short HEAD)"
