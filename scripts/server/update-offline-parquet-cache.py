from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from samosbor.config import load_config
from samosbor.data.tbank import TBankMarketDataProvider
from samosbor.offline_parquet_cache import (
    candles_to_frame,
    incremental_fetch_start,
    write_instrument_metadata,
    latest_candle_timestamp,
    load_candle_frame,
    merge_candle_frames,
    write_candle_frame,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Incrementally update local parquet candles for offline nightly autonomy."
    )
    parser.add_argument("--config", required=True, help="Source runtime config path.")
    parser.add_argument("--parquet-dir", required=True, help="Target parquet directory.")
    parser.add_argument(
        "--bootstrap-history-days",
        type=int,
        default=None,
        help=(
            "History window to fetch when a parquet file does not exist yet. "
            "Defaults to the config [data].history_days value."
        ),
    )
    parser.add_argument(
        "--overlap-hours",
        type=int,
        default=48,
        help="Overlap window to refetch around the latest cached candle.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    parquet_dir = config.resolve_path(args.parquet_dir)
    provider = TBankMarketDataProvider(config)
    instruments = provider.resolve_universe(config.data.instruments)
    metadata_path = write_instrument_metadata(parquet_dir, instruments)
    bootstrap_history_days = max(
        1,
        args.bootstrap_history_days
        if args.bootstrap_history_days is not None
        else config.data.history_days,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir = config.autotune_dir() / "offline-cache" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    missing_symbols: list[str] = []

    for instrument in instruments:
        path = parquet_dir / f"{instrument.symbol}_{config.data.timeframe.lower()}.parquet"
        existing = load_candle_frame(path)
        previous_latest = latest_candle_timestamp(existing)
        fetch_from = incremental_fetch_start(
            previous_latest,
            bootstrap_history_days=bootstrap_history_days,
            overlap_hours=args.overlap_hours,
        )
        try:
            fresh_candles = provider.get_candles_range(
                instrument,
                timeframe=config.data.timeframe,
                from_dt=fetch_from,
            )
        except Exception as exc:
            if not existing.empty:
                results.append(
                    {
                        "symbol": instrument.symbol,
                        "status": "stale-cache-reused",
                        "reason": str(exc),
                        "path": str(path),
                        "previous_latest": previous_latest.isoformat() if previous_latest else "",
                    }
                )
                continue
            results.append(
                {
                    "symbol": instrument.symbol,
                    "status": "bootstrap-failed",
                    "reason": str(exc),
                    "path": str(path),
                    "previous_latest": previous_latest.isoformat() if previous_latest else "",
                    "fetch_from": fetch_from.isoformat(),
                }
            )
            missing_symbols.append(instrument.symbol)
            continue

        fresh = candles_to_frame(fresh_candles)
        merged = merge_candle_frames(existing, fresh)
        write_candle_frame(path, merged)
        latest = latest_candle_timestamp(merged)
        results.append(
            {
                "symbol": instrument.symbol,
                "status": "updated",
                "path": str(path),
                "fetched_rows": len(fresh),
                "total_rows": len(merged),
                "previous_latest": previous_latest.isoformat() if previous_latest else "",
                "latest": latest.isoformat() if latest else "",
                "fetch_from": fetch_from.isoformat(),
            }
        )

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_config": str(Path(args.config).resolve()),
        "parquet_dir": str(parquet_dir),
        "instrument_metadata_path": str(metadata_path),
        "timeframe": config.data.timeframe,
        "bootstrap_history_days": bootstrap_history_days,
        "overlap_hours": args.overlap_hours,
        "safe_to_continue": not missing_symbols,
        "missing_symbols": missing_symbols,
        "results": results,
    }
    (output_dir / "offline_cache_update.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if missing_symbols else 0


if __name__ == "__main__":
    raise SystemExit(main())
