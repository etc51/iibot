#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOCK_FILE="/tmp/samosbor-paper-cycle.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "paper-cycle is already running"
  exit 0
fi

source "$ROOT_DIR/.venv/bin/activate"
BASE_CONFIG="configs/server_tbank_stocks_intraday_300k_focused.toml"
EFFECTIVE_CONFIG="configs/server_tbank_stocks_intraday_300k_focused.effective.toml"

python -m samosbor.cli --config "$BASE_CONFIG" refresh-effective-config --output "$EFFECTIVE_CONFIG"
python -m samosbor.cli --config "$EFFECTIVE_CONFIG" paper-cycle
