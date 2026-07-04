from __future__ import annotations

from datetime import datetime, timedelta
from math import isfinite
from typing import Any

from ..analysis.indicators import atr
from ..config import StrategySection
from ..domain import Candle, Signal, SignalDirection

ADAPTIVE_WAIT_PREFIX = "entry waits for adaptive confirmation"
ADAPTIVE_SKIP_PREFIX = "entry skipped after adaptive wait"
CONFIRMATION_BLOCK_PREFIX = "entry blocked by 5min confirmation"


def build_adaptive_entry_context(
    signal: Signal,
    candles: list[Candle],
    strategy: StrategySection,
    *,
    prior_events: list[dict[str, Any]],
    timeframe: str,
) -> dict[str, object]:
    if not strategy.adaptive_entry_enabled:
        return {"available": False, "enabled": False, "reason": "disabled"}
    if not candles:
        return {"available": False, "enabled": True, "reason": "no candles"}

    entry_candle = signal.metadata.get("entry_candle", {})
    if not isinstance(entry_candle, dict) or not entry_candle.get("available", False):
        return {"available": False, "enabled": True, "reason": "missing entry candle context"}

    confirmation = signal.metadata.get("entry_confirmation", {})
    if not isinstance(confirmation, dict):
        confirmation = {}

    if confirmation.get("against_direction"):
        return {
            "available": True,
            "enabled": True,
            "grade": "D",
            "action": "defer-to-lower-timeframe-confirmation",
            "wait_bars": 0,
            "max_wait_bars": 0,
            "reason": "lower timeframe is against entry",
        }

    grade, reason = _entry_grade(signal.direction, entry_candle, confirmation)
    wait_bars = 0 if grade == "A" else 1
    max_wait_bars = 0 if grade == "A" else (1 if grade == "B" else 2)
    anchor = _find_wait_anchor(
        prior_events,
        symbol=signal.instrument.symbol,
        direction=signal.direction.value,
    )
    elapsed_bars = _elapsed_bars(anchor, candles[-1].timestamp, timeframe=timeframe) if anchor else 0
    atr_value = atr(candles, strategy.atr_window)
    favorable_move_atr = _favorable_move_atr(signal, anchor, atr_value)
    chase_limit = max(0.0, float(strategy.adaptive_entry_max_chase_atr))
    confirmed_after_wait = _confirmed_after_wait(signal.direction, entry_candle, confirmation)

    action = "enter-now"
    block_reason = ""
    if grade != "A":
        if favorable_move_atr is not None and favorable_move_atr > chase_limit:
            action = "skip-chase"
            block_reason = (
                f"{ADAPTIVE_SKIP_PREFIX} "
                f"({favorable_move_atr:.2f} ATR already moved, limit {chase_limit:.2f})"
            )
        elif anchor is None:
            action = "wait"
            block_reason = f"{ADAPTIVE_WAIT_PREFIX} ({grade} setup: {reason})"
        elif elapsed_bars < wait_bars:
            action = "wait"
            block_reason = (
                f"{ADAPTIVE_WAIT_PREFIX} "
                f"({grade} setup: {elapsed_bars}/{wait_bars} bar)"
            )
        elif confirmed_after_wait:
            action = "enter-after-confirmation"
        elif elapsed_bars < max_wait_bars:
            action = "wait"
            block_reason = f"{ADAPTIVE_WAIT_PREFIX} ({grade} setup still unconfirmed)"
        else:
            action = "skip-unconfirmed"
            block_reason = f"{ADAPTIVE_SKIP_PREFIX} ({grade} setup stayed unconfirmed)"

    return {
        "available": True,
        "enabled": True,
        "grade": grade,
        "reason": reason,
        "action": action,
        "block_reason": block_reason,
        "wait_bars": wait_bars,
        "max_wait_bars": max_wait_bars,
        "elapsed_bars": elapsed_bars,
        "confirmed_after_wait": confirmed_after_wait,
        "favorable_move_atr": round(favorable_move_atr, 4) if favorable_move_atr is not None else None,
        "max_chase_atr": chase_limit,
        "anchor_timestamp": anchor.get("timestamp") if anchor else None,
        "anchor_price": _anchor_price(anchor),
        "signal_price": round(float(signal.entry_price), 6),
    }


def adaptive_entry_block_reason(signal: Signal) -> str | None:
    context = signal.metadata.get("adaptive_entry", {})
    if not isinstance(context, dict):
        return None
    reason = str(context.get("block_reason", "")).strip()
    return reason or None


def tracked_alternative_plan(strategy: StrategySection) -> dict[str, object]:
    return {
        "enabled": bool(strategy.alternative_plan_enabled),
        "entry_offset_bars": int(strategy.alternative_plan_entry_offset_bars),
        "stop_multiple": float(strategy.alternative_plan_atr_stop_multiple),
        "reward_to_risk": float(strategy.alternative_plan_reward_to_risk),
        "mode": "observe-only",
    }


def _entry_grade(
    direction: SignalDirection,
    entry_candle: dict[str, object],
    confirmation: dict[str, object],
) -> tuple[str, str]:
    direction_confirmed = bool(entry_candle.get("direction_confirmed_by_close"))
    reversal = bool(entry_candle.get("reversal_against_direction"))
    close_position = _float(entry_candle.get("close_position"), 0.5)
    ret1 = _float(entry_candle.get("ret1"), 0.0)
    range_ratio = _float(entry_candle.get("range_ratio"), 1.0)
    adverse_bars = int(_float(confirmation.get("adverse_bars"), 0.0))

    favorable_close = close_position <= 0.2 if direction == SignalDirection.SHORT else close_position >= 0.8
    mild_reversal = _is_mild_reversal(direction, ret1, close_position, range_ratio)
    lower_timeframe_clean = adverse_bars <= 1

    if direction_confirmed and favorable_close and range_ratio <= 2.5 and lower_timeframe_clean:
        return "A", "direction confirmed with clean close"
    if direction_confirmed and not reversal and range_ratio <= 3.0:
        return "B", "direction confirmed but close/range needs one more check"
    if mild_reversal and lower_timeframe_clean and range_ratio <= 1.2:
        return "B", "minor rebound after signal"
    return "C", "reversal, wide candle, or weak close needs adaptive wait"


def _confirmed_after_wait(
    direction: SignalDirection,
    entry_candle: dict[str, object],
    confirmation: dict[str, object],
) -> bool:
    reversal = bool(entry_candle.get("reversal_against_direction"))
    direction_confirmed = bool(entry_candle.get("direction_confirmed_by_close"))
    ret_window = _float(confirmation.get("ret_window"), 0.0)
    adverse_bars = int(_float(confirmation.get("adverse_bars"), 0.0))
    lower_direction_ok = ret_window < 0 if direction == SignalDirection.SHORT else ret_window > 0
    return (direction_confirmed and not reversal) or (not reversal and lower_direction_ok and adverse_bars <= 1)


def _is_mild_reversal(
    direction: SignalDirection,
    ret1: float,
    close_position: float,
    range_ratio: float,
) -> bool:
    if range_ratio > 1.2:
        return False
    if direction == SignalDirection.SHORT:
        return 0.0 <= ret1 <= 0.0015 and close_position <= 0.65
    return -0.0015 <= ret1 <= 0.0 and close_position >= 0.35


def _find_wait_anchor(
    events: list[dict[str, Any]],
    *,
    symbol: str,
    direction: str,
) -> dict[str, Any] | None:
    wait_events: list[dict[str, Any]] = []
    for event in reversed(events):
        if event.get("symbol") != symbol or event.get("direction") != direction:
            continue
        if event.get("action") == "open" or event.get("approved") is True:
            break
        reason = str(event.get("reason", ""))
        if reason.startswith(ADAPTIVE_WAIT_PREFIX) or reason.startswith(CONFIRMATION_BLOCK_PREFIX):
            wait_events.append(event)
            continue
        if event.get("action") == "signal":
            break
    return wait_events[-1] if wait_events else None


def _elapsed_bars(anchor: dict[str, Any], timestamp: datetime, *, timeframe: str) -> int:
    try:
        anchor_timestamp = datetime.fromisoformat(str(anchor["timestamp"]))
    except Exception:
        return 0
    seconds = max(0.0, (timestamp - anchor_timestamp).total_seconds())
    duration = _timeframe_duration(timeframe).total_seconds()
    if duration <= 0:
        return 0
    return int(seconds // duration)


def _favorable_move_atr(signal: Signal, anchor: dict[str, Any] | None, atr_value: float | None) -> float | None:
    if anchor is None or atr_value is None or atr_value <= 0:
        return None
    anchor_price = _anchor_price(anchor)
    if anchor_price is None:
        return None
    if signal.direction == SignalDirection.SHORT:
        move = anchor_price - signal.entry_price
    else:
        move = signal.entry_price - anchor_price
    return max(0.0, move / atr_value)


def _anchor_price(anchor: dict[str, Any] | None) -> float | None:
    if not anchor:
        return None
    metadata = anchor.get("metadata", {})
    if isinstance(metadata, dict):
        adaptive = metadata.get("adaptive_entry", {})
        if isinstance(adaptive, dict):
            price = _optional_float(adaptive.get("signal_price"))
            if price is not None:
                return price
    return _optional_float(anchor.get("entry_price"))


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
    return mapping.get(timeframe.lower(), timedelta(minutes=15))


def _float(value: object, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _optional_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None
