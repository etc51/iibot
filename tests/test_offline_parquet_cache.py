from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from samosbor.domain import Candle, Instrument, InstrumentType
from samosbor.offline_parquet_cache import (
    candles_to_frame,
    incremental_fetch_start,
    instrument_metadata_path,
    latest_candle_timestamp,
    load_instrument_metadata,
    load_candle_frame,
    merge_candle_frames,
    write_instrument_metadata,
    write_candle_frame,
)


class OfflineParquetCacheTest(unittest.TestCase):
    def test_merge_candle_frames_replaces_overlap_and_sorts_index(self):
        base_time = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
        existing = candles_to_frame(
            [
                Candle(base_time, 100.0, 101.0, 99.0, 100.5, 10.0),
                Candle(base_time + timedelta(minutes=30), 101.0, 102.0, 100.0, 101.5, 11.0),
            ]
        )
        fresh = candles_to_frame(
            [
                Candle(base_time + timedelta(minutes=30), 101.5, 103.0, 101.0, 102.5, 12.0),
                Candle(base_time + timedelta(minutes=60), 102.5, 104.0, 102.0, 103.5, 13.0),
            ]
        )

        merged = merge_candle_frames(existing, fresh)

        self.assertEqual(len(merged), 3)
        self.assertEqual(float(merged.iloc[1]["close"]), 102.5)
        self.assertEqual(float(merged.iloc[2]["close"]), 103.5)
        self.assertEqual(merged.index.name, "time")

    def test_load_and_write_round_trip_preserves_latest_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "SBER_30min.parquet"
            candles = [
                Candle(
                    datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
                    100.0,
                    101.0,
                    99.0,
                    100.5,
                    10.0,
                ),
                Candle(
                    datetime(2026, 6, 1, 10, 30, tzinfo=timezone.utc),
                    100.5,
                    102.0,
                    100.0,
                    101.5,
                    12.0,
                ),
            ]
            frame = candles_to_frame(candles)
            write_candle_frame(path, frame)

            loaded = load_candle_frame(path)

            self.assertEqual(len(loaded), 2)
            self.assertEqual(
                latest_candle_timestamp(loaded),
                datetime(2026, 6, 1, 10, 30, tzinfo=timezone.utc),
            )

    def test_load_candle_frame_supports_timestamp_utc_column(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "SBER_30min.parquet"
            pd.DataFrame(
                {
                    "timestamp_utc": pd.to_datetime(
                        [
                            datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc),
                            datetime(2026, 6, 1, 10, 30, tzinfo=timezone.utc),
                        ],
                        utc=True,
                    ),
                    "open": [100.0, 100.5],
                    "high": [101.0, 102.0],
                    "low": [99.0, 100.0],
                    "close": [100.5, 101.5],
                    "volume": [10.0, 12.0],
                }
            ).to_parquet(path)

            loaded = load_candle_frame(path)

            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded.index.name, "time")
            self.assertEqual(
                latest_candle_timestamp(loaded),
                datetime(2026, 6, 1, 10, 30, tzinfo=timezone.utc),
            )

    def test_incremental_fetch_start_uses_overlap_when_cache_exists(self):
        latest = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
        fetch_from = incremental_fetch_start(
            latest,
            bootstrap_history_days=120,
            overlap_hours=48,
        )

        self.assertEqual(fetch_from, latest - timedelta(hours=48))

    def test_incremental_fetch_start_bootstraps_when_cache_missing(self):
        before = datetime.now(timezone.utc) - timedelta(days=3)
        fetch_from = incremental_fetch_start(
            None,
            bootstrap_history_days=3,
            overlap_hours=48,
        )
        after = datetime.now(timezone.utc) - timedelta(days=3, minutes=-1)

        self.assertGreaterEqual(fetch_from, before)
        self.assertLessEqual(fetch_from, after)

    def test_instrument_metadata_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            path = write_instrument_metadata(
                root,
                [
                    Instrument(
                        symbol="SBER",
                        instrument_type=InstrumentType.STOCK,
                        uid="uid-sber",
                        class_code="TQBR",
                        lot_size=10,
                        tick_size=0.01,
                    ),
                    Instrument(
                        symbol="CNYRUBF",
                        instrument_type=InstrumentType.FUTURE,
                        uid="uid-cny",
                        class_code="SPBFUT",
                        lot_size=1,
                        tick_size=0.0001,
                        initial_margin_buy=1234.0,
                        initial_margin_sell=1250.0,
                        tick_value=1.5,
                    ),
                ],
            )

            loaded = load_instrument_metadata(root)

            self.assertEqual(path, instrument_metadata_path(root))
            self.assertEqual(loaded["SBER"].uid, "uid-sber")
            self.assertEqual(loaded["SBER"].lot_size, 10)
            self.assertEqual(loaded["CNYRUBF"].instrument_type, InstrumentType.FUTURE)
            self.assertEqual(loaded["CNYRUBF"].tick_value, 1.5)


if __name__ == "__main__":
    unittest.main()
