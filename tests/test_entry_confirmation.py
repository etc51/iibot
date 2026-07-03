from __future__ import annotations

from datetime import datetime, timedelta, timezone

from samosbor.autonomy.entry_confirmation import build_entry_confirmation_context
from samosbor.domain import Candle


def _candle(ts: datetime, open_price: float, close_price: float) -> Candle:
    return Candle(
        timestamp=ts,
        open=open_price,
        high=max(open_price, close_price),
        low=min(open_price, close_price),
        close=close_price,
        volume=1_000_000,
    )


def test_5m_confirmation_blocks_severe_short_rebound():
    ts = datetime(2026, 7, 3, 10, 0, tzinfo=timezone.utc)
    candles = [
        _candle(ts, 100.0, 100.2),
        _candle(ts + timedelta(minutes=5), 100.2, 100.6),
        _candle(ts + timedelta(minutes=10), 100.6, 100.7),
    ]

    context = build_entry_confirmation_context(
        candles,
        "short",
        signal_timestamp=ts,
        primary_timeframe="15min",
        confirmation_timeframe="5min",
        min_bars=3,
        max_adverse_ret=0.005,
    )

    assert context["available"] is True
    assert context["against_direction"] is True
    assert context["reason"] == "5m rebound against short"


def test_5m_confirmation_allows_mild_short_rebound():
    ts = datetime(2026, 7, 3, 10, 0, tzinfo=timezone.utc)
    candles = [
        _candle(ts - timedelta(minutes=30), 99.0, 99.1),
        _candle(ts - timedelta(minutes=25), 99.1, 99.2),
        _candle(ts - timedelta(minutes=20), 99.2, 99.3),
        _candle(ts - timedelta(minutes=15), 99.3, 99.4),
        _candle(ts - timedelta(minutes=10), 99.4, 99.5),
        _candle(ts - timedelta(minutes=5), 99.5, 99.6),
        _candle(ts, 100.0, 100.1),
        _candle(ts + timedelta(minutes=5), 100.1, 100.2),
        _candle(ts + timedelta(minutes=10), 100.2, 100.3),
    ]

    context = build_entry_confirmation_context(
        candles,
        "short",
        signal_timestamp=ts,
        primary_timeframe="15min",
        confirmation_timeframe="5min",
        min_bars=3,
        max_adverse_ret=0.005,
    )

    assert context["available"] is True
    assert context["ret_window"] == 0.003
    assert context["adverse_bars"] == 3
    assert context["against_direction"] is False
    assert context["reason"] == "confirmed"
