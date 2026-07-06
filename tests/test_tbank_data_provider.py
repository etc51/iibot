from __future__ import annotations

import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from samosbor.data.tbank import (
    _candle_request_ranges,
    _pace_tbank_request,
    _secondary_timeframe_history_days,
    drop_incomplete_trailing_candle,
)
from samosbor.domain import Candle


def _candle(ts: datetime) -> Candle:
    return Candle(
        timestamp=ts,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1_000.0,
    )


class TBankDataProviderTest(unittest.TestCase):
    def test_drop_incomplete_trailing_candle_removes_forming_bar(self):
        candles = [
            _candle(datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)),
            _candle(datetime(2025, 1, 1, 10, 15, tzinfo=timezone.utc)),
        ]

        filtered = drop_incomplete_trailing_candle(
            candles,
            timeframe="15min",
            as_of=datetime(2025, 1, 1, 10, 20, tzinfo=timezone.utc),
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].timestamp, candles[0].timestamp)

    def test_drop_incomplete_trailing_candle_keeps_closed_bar(self):
        candles = [
            _candle(datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)),
            _candle(datetime(2025, 1, 1, 10, 15, tzinfo=timezone.utc)),
        ]

        filtered = drop_incomplete_trailing_candle(
            candles,
            timeframe="15min",
            as_of=datetime(2025, 1, 1, 10, 30, tzinfo=timezone.utc),
        )

        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[-1].timestamp, candles[-1].timestamp)

    def test_intraday_candle_ranges_are_split_by_day(self):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=2, hours=3)

        ranges = _candle_request_ranges(start, end, "15min")

        self.assertEqual(
            ranges,
            [
                (start, start + timedelta(days=1)),
                (start + timedelta(days=1), start + timedelta(days=2)),
                (start + timedelta(days=2), end),
            ],
        )

    def test_secondary_intraday_timeframe_uses_shorter_history(self):
        self.assertEqual(
            _secondary_timeframe_history_days(
                primary_timeframe="15min",
                secondary_timeframe="5min",
                configured_days=10,
            ),
            4,
        )
        self.assertEqual(
            _secondary_timeframe_history_days(
                primary_timeframe="15min",
                secondary_timeframe="30min",
                configured_days=10,
            ),
            10,
        )

    def test_tbank_request_pacing_uses_shared_state_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "tbank-rate-limit.state"
            with patch.dict(
                os.environ,
                {
                    "SAMOSBOR_TBANK_RATE_LIMIT_PATH": str(state_path),
                    "SAMOSBOR_TBANK_MIN_REQUEST_INTERVAL_SEC": "0.01",
                },
            ):
                started = time.perf_counter()
                _pace_tbank_request()
                _pace_tbank_request()
                elapsed = time.perf_counter() - started

            self.assertTrue(state_path.exists())
            self.assertGreaterEqual(elapsed, 0.005)


if __name__ == "__main__":
    unittest.main()
