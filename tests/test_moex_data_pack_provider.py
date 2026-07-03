from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from samosbor.data.moex_data_pack import MoexDataPackProvider
from samosbor.domain import Instrument, InstrumentType


def _write_metadata(base_path: Path, *, uid: str) -> None:
    metadata_root = base_path / "data_pack" / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "ticker": "SBER",
                "root_symbol": "SBER",
                "display_code": "SBER",
                "expiration_date": "",
                "is_current": True,
                "class_code": "TQBR",
                "figi": "BBG004730N88",
                "instrument_uid": uid,
                "lot": 10,
                "tick_size": 0.01,
                "currency": "rub",
            }
        ]
    ).to_parquet(metadata_root / "instruments.parquet")


class MoexDataPackProviderTest(unittest.TestCase):
    def test_provider_accepts_timestamp_utc_and_plain_volume_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            uid = "uid-sber"
            _write_metadata(root, uid=uid)
            candles_root = root / "data_pack" / "candles_1m" / "SBER"
            candles_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "timestamp_utc": pd.to_datetime(
                        [
                            datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc),
                            datetime(2025, 1, 1, 10, 1, tzinfo=timezone.utc),
                        ],
                        utc=True,
                    ),
                    "open": [100.0, 101.0],
                    "high": [101.5, 102.0],
                    "low": [99.5, 100.5],
                    "close": [101.0, 101.8],
                    "volume": [100, 120],
                }
            ).to_parquet(candles_root / f"{uid}_part1.parquet")

            provider = MoexDataPackProvider(root, timeframe="1min", history_days=10)
            instrument = provider.resolve_instrument(
                Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK)
            )

            candles = provider.get_candles(instrument)

            self.assertEqual(len(candles), 2)
            self.assertEqual(candles[0].timestamp, datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc))
            self.assertEqual(candles[1].volume, 120.0)

    def test_provider_resamples_to_interval_end_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            uid = "uid-sber"
            _write_metadata(root, uid=uid)
            candles_root = root / "data_pack" / "candles_1m" / "SBER"
            candles_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "utc_ts": pd.to_datetime(
                        [
                            datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc),
                            datetime(2025, 1, 1, 10, 1, tzinfo=timezone.utc),
                            datetime(2025, 1, 1, 10, 14, tzinfo=timezone.utc),
                        ],
                        utc=True,
                    ),
                    "open": [100.0, 101.0, 102.0],
                    "high": [101.0, 102.0, 103.0],
                    "low": [99.5, 100.5, 101.5],
                    "close": [100.8, 101.7, 102.6],
                    "volume_lots": [100, 120, 140],
                }
            ).to_parquet(candles_root / f"{uid}_part1.parquet")

            provider = MoexDataPackProvider(root, timeframe="15min", history_days=10)
            instrument = provider.resolve_instrument(
                Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK)
            )

            candles = provider.get_candles(instrument)

            self.assertEqual(len(candles), 1)
            self.assertEqual(candles[0].timestamp, datetime(2025, 1, 1, 10, 15, tzinfo=timezone.utc))
            self.assertEqual(candles[0].open, 100.0)
            self.assertEqual(candles[0].close, 102.6)
            self.assertEqual(candles[0].volume, 360.0)

    def test_provider_deduplicates_overlapping_parquet_chunks_by_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            uid = "uid-sber"
            _write_metadata(root, uid=uid)
            candles_root = root / "data_pack" / "candles_1m" / "SBER"
            candles_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "utc_ts": pd.to_datetime(
                        [
                            datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc),
                            datetime(2025, 1, 1, 10, 1, tzinfo=timezone.utc),
                        ],
                        utc=True,
                    ),
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.0, 100.0],
                    "close": [100.5, 101.5],
                    "volume_lots": [10, 11],
                }
            ).to_parquet(candles_root / f"{uid}_part1.parquet")
            pd.DataFrame(
                {
                    "utc_ts": pd.to_datetime(
                        [
                            datetime(2025, 1, 1, 10, 1, tzinfo=timezone.utc),
                            datetime(2025, 1, 1, 10, 2, tzinfo=timezone.utc),
                        ],
                        utc=True,
                    ),
                    "open": [101.2, 102.0],
                    "high": [102.5, 103.0],
                    "low": [100.5, 101.5],
                    "close": [102.0, 102.8],
                    "volume_lots": [99, 12],
                }
            ).to_parquet(candles_root / f"{uid}_part2.parquet")

            provider = MoexDataPackProvider(root, timeframe="1min", history_days=10)
            instrument = provider.resolve_instrument(
                Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK)
            )

            candles = provider.get_candles(instrument)

            self.assertEqual(len(candles), 3)
            self.assertEqual(candles[1].timestamp, datetime(2025, 1, 1, 10, 1, tzinfo=timezone.utc))
            self.assertEqual(candles[1].open, 101.2)
            self.assertEqual(candles[1].volume, 99.0)

    def test_provider_keeps_full_first_bucket_when_history_cutoff_slices_interval(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            uid = "uid-sber"
            _write_metadata(root, uid=uid)
            candles_root = root / "data_pack" / "candles_1m" / "SBER"
            candles_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "utc_ts": pd.to_datetime(
                        [
                            datetime(2025, 1, 1, 12, 15, tzinfo=timezone.utc),
                            datetime(2025, 1, 1, 12, 20, tzinfo=timezone.utc),
                            datetime(2025, 1, 1, 12, 29, tzinfo=timezone.utc),
                            datetime(2025, 1, 2, 12, 20, tzinfo=timezone.utc),
                        ],
                        utc=True,
                    ),
                    "open": [100.0, 101.0, 102.0, 200.0],
                    "high": [101.0, 102.0, 103.0, 201.0],
                    "low": [99.0, 100.0, 101.0, 199.0],
                    "close": [100.5, 101.5, 102.5, 200.5],
                    "volume_lots": [10, 11, 12, 20],
                }
            ).to_parquet(candles_root / f"{uid}_part1.parquet")

            provider = MoexDataPackProvider(root, timeframe="15min", history_days=1)
            instrument = provider.resolve_instrument(
                Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK)
            )

            candles = provider.get_candles(instrument)

            self.assertEqual(len(candles), 2)
            self.assertEqual(candles[0].timestamp, datetime(2025, 1, 1, 12, 30, tzinfo=timezone.utc))
            self.assertEqual(candles[0].open, 100.0)
            self.assertEqual(candles[0].close, 102.5)
            self.assertEqual(candles[1].timestamp, datetime(2025, 1, 2, 12, 30, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
