from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from samosbor.data.parquet_directory import ParquetDirectoryProvider
from samosbor.domain import Instrument, InstrumentType
from samosbor.offline_parquet_cache import write_instrument_metadata


class ParquetDirectoryProviderTest(unittest.TestCase):
    def test_provider_reads_symbol_timeframe_parquet_with_time_index(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            frame = pd.DataFrame(
                {
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.5, 100.5],
                    "close": [100.8, 101.5],
                    "volume": [1000, 1200],
                },
                index=pd.to_datetime(
                    [
                        datetime(2025, 1, 1, 7, 0, tzinfo=timezone.utc),
                        datetime(2025, 1, 1, 7, 30, tzinfo=timezone.utc),
                    ]
                ),
            )
            frame.index.name = "time"
            frame.to_parquet(root / "SBER_30min.parquet")

            provider = ParquetDirectoryProvider(root, timeframe="30min", history_days=10)
            candles = provider.get_candles(
                Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK)
            )

            self.assertEqual(len(candles), 2)
            self.assertEqual(candles[0].timestamp, datetime(2025, 1, 1, 7, 0, tzinfo=timezone.utc))
            self.assertEqual(candles[1].close, 101.5)

    def test_provider_reads_timestamp_utc_column(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pd.DataFrame(
                {
                    "timestamp_utc": pd.to_datetime(
                        [
                            datetime(2025, 1, 1, 7, 0, tzinfo=timezone.utc),
                            datetime(2025, 1, 1, 7, 30, tzinfo=timezone.utc),
                        ],
                        utc=True,
                    ),
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.5, 100.5],
                    "close": [100.8, 101.5],
                    "volume": [1000, 1200],
                }
            ).to_parquet(root / "SBER_30min.parquet")

            provider = ParquetDirectoryProvider(root, timeframe="30min", history_days=10)
            candles = provider.get_candles(
                Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK)
            )

            self.assertEqual(len(candles), 2)
            self.assertEqual(candles[0].timestamp, datetime(2025, 1, 1, 7, 0, tzinfo=timezone.utc))
            self.assertEqual(candles[1].close, 101.5)

    def test_provider_enriches_instrument_from_metadata_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            frame = pd.DataFrame(
                {
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.5],
                    "close": [100.8],
                    "volume": [1000],
                },
                index=pd.to_datetime([datetime(2025, 1, 1, 7, 0, tzinfo=timezone.utc)]),
            )
            frame.index.name = "time"
            frame.to_parquet(root / "SBER_30min.parquet")
            write_instrument_metadata(
                root,
                [
                    Instrument(
                        symbol="SBER",
                        instrument_type=InstrumentType.STOCK,
                        uid="uid-sber",
                        figi="figi-sber",
                        class_code="TQBR",
                        lot_size=10,
                        tick_size=0.01,
                    )
                ],
            )

            provider = ParquetDirectoryProvider(root, timeframe="30min", history_days=10)
            instrument = provider.resolve_instrument(
                Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK)
            )

            self.assertEqual(instrument.uid, "uid-sber")
            self.assertEqual(instrument.figi, "figi-sber")
            self.assertEqual(instrument.class_code, "TQBR")
            self.assertEqual(instrument.lot_size, 10)


if __name__ == "__main__":
    unittest.main()
