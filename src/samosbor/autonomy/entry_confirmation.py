from __future__ import annotations

from datetime import datetime, timedelta

from ..analysis.indicators import ema
from ..domain import Candle


def build_entry_confirmation_context(
    candles: list[Candle],
    direction: str,
    *,
    signal_timestamp: datetime,
    primary_timeframe: str,
    confirmation_timeframe: str,
    min_bars: int = 3,
    max_adverse_ret: float = 0.005,
) -> dict[str, object]:
    if not confirmation_timeframe.strip():
        return {"available": False, "enabled": False}
    if not candles:
        return {
            "available": False,
            "enabled": True,
            "timeframe": confirmation_timeframe,
            "reason": "no confirmation candles",
        }

    primary_duration = _timeframe_duration(primary_timeframe)
    window = [
        candle
        for candle in candles
        if signal_timestamp <= candle.timestamp < signal_timestamp + primary_duration
    ]
    if len(window) < max(1, min_bars):
        window = [candle for candle in candles if candle.timestamp <= signal_timestamp][-max(1, min_bars) :]
    if len(window) < max(1, min_bars):
        return {
            "available": False,
            "enabled": True,
            "timeframe": confirmation_timeframe,
            "reason": "insufficient confirmation candles",
            "bars": len(window),
            "required_bars": max(1, min_bars),
        }

    normalized_direction = direction.strip().lower()
    latest = window[-1]
    previous = window[-2] if len(window) >= 2 else window[-1]
    first = window[0]
    ret_window = (latest.close / first.open - 1.0) if first.open > 0 else 0.0
    ret_last = (latest.close / previous.close - 1.0) if previous.close > 0 else 0.0
    adverse_bars = _adverse_bar_count(window[-max(1, min_bars) :], normalized_direction)
    closes = [candle.close for candle in candles if candle.timestamp <= latest.timestamp]
    ema_fast = ema(closes, 9)

    against_direction = False
    reason = ""
    if normalized_direction == "short":
        against_direction = ret_window > max_adverse_ret
        if against_direction:
            reason = "5m rebound against short"
    elif normalized_direction == "long":
        against_direction = ret_window < -max_adverse_ret
        if against_direction:
            reason = "5m pullback against long"

    return {
        "available": True,
        "enabled": True,
        "timeframe": confirmation_timeframe,
        "bars": len(window),
        "ret_window": round(ret_window, 6),
        "ret_last": round(ret_last, 6),
        "adverse_bars": adverse_bars,
        "ema_fast": round(ema_fast, 6) if ema_fast is not None else None,
        "latest_close": round(float(latest.close), 6),
        "against_direction": against_direction,
        "confirmation_ok": not against_direction,
        "reason": reason or "confirmed",
    }


def _adverse_bar_count(candles: list[Candle], direction: str) -> int:
    if direction == "short":
        return sum(1 for candle in candles if candle.close > candle.open)
    if direction == "long":
        return sum(1 for candle in candles if candle.close < candle.open)
    return 0


def _timeframe_duration(timeframe: str) -> timedelta:
    mapping = {
        "day": timedelta(days=1),
        "hour": timedelta(hours=1),
        "30min": timedelta(minutes=30),
        "15min": timedelta(minutes=15),
        "10min": timedelta(minutes=10),
        "5min": timedelta(minutes=5),
        "1min": timedelta(minutes=1),
    }
    return mapping.get(timeframe.strip().lower(), timedelta(minutes=15))
