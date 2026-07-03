from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..domain import Candle, Instrument


class CSVMarketDataProvider:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path

    def resolve_universe(self, instruments: list[Instrument]) -> list[Instrument]:
        return instruments

    def load_history(self, instruments: list[Instrument]) -> dict[str, list[Candle]]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        requested = {instrument.symbol for instrument in instruments}
        candles: dict[str, list[Candle]] = defaultdict(list)

        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                symbol = row["symbol"].strip().upper()
                if symbol not in requested:
                    continue
                timestamp = datetime.fromisoformat(row["timestamp"])
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                candles[symbol].append(
                    Candle(
                        timestamp=timestamp,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                )

        return {symbol: sorted(items, key=lambda candle: candle.timestamp) for symbol, items in candles.items()}
