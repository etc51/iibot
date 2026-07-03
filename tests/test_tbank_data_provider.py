from __future__ import annotations

import unittest
from datetime import datetime, timezone

from samosbor.data.tbank import drop_incomplete_trailing_candle
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


if __name__ == "__main__":
    unittest.main()
