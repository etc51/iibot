#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/.venv/bin/activate"
exec python -m samosbor.cli \
  --config configs/server_tbank_stocks_intraday_300k_focused.toml \
  collect-microstructure \
  --interval-sec "${SAMOSBOR_MICROSTRUCTURE_INTERVAL_SEC:-15}" \
  --depth "${SAMOSBOR_MICROSTRUCTURE_DEPTH:-10}"
