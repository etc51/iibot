#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT_DIR"
source .venv/bin/activate

BASE_CONFIG="configs/server_tbank_stocks_intraday_300k_focused.toml"
EFFECTIVE_CONFIG="configs/server_tbank_stocks_intraday_300k_focused.effective.toml"
AUTONOMY_CONFIG="configs/server_tbank_stocks_intraday_300k_focused.autonomy.toml"
AUTONOMY_PARQUET_DIR="data/server_moex_strategy_lab_data_processed"
ACTIVE_CONFIG="$EFFECTIVE_CONFIG"

python -m samosbor.cli --config "$BASE_CONFIG" refresh-effective-config --output "$EFFECTIVE_CONFIG"

python scripts/server/build-offline-autonomy-config.py \
  --source "$ACTIVE_CONFIG" \
  --output "$AUTONOMY_CONFIG" \
  --parquet-dir "$AUTONOMY_PARQUET_DIR"

python scripts/server/update-offline-parquet-cache.py \
  --config "$ACTIVE_CONFIG" \
  --parquet-dir "$AUTONOMY_PARQUET_DIR"

mkdir -p runs/logs
python -m samosbor.cli --config "$ACTIVE_CONFIG" daily-review --summary-only \
  > runs/logs/daily-review-summary.json \
  2>> runs/logs/daily-review.log
cat runs/logs/daily-review-summary.json

python -m samosbor.cli --config "$AUTONOMY_CONFIG" nightly-autonomy --base-config "$BASE_CONFIG" --effective-output "$EFFECTIVE_CONFIG"
