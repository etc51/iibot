from __future__ import annotations

from ..domain import Candle, Position, SignalDirection


def runner_breakeven_stop(
    *,
    direction: SignalDirection,
    entry_price: float,
    buffer_bps: float,
) -> float:
    buffer = max(0.0, float(buffer_bps)) / 10_000.0
    if direction == SignalDirection.SHORT:
        return entry_price * (1.0 - buffer)
    return entry_price * (1.0 + buffer)


def runner_extreme_price(
    *,
    direction: SignalDirection,
    current_extreme: float,
    candle: Candle,
    activation_price: float,
) -> float:
    if direction == SignalDirection.SHORT:
        base = current_extreme if current_extreme > 0 else activation_price
        return min(base, candle.low)
    base = current_extreme if current_extreme > 0 else activation_price
    return max(base, candle.high)


def runner_trailing_stop(
    *,
    direction: SignalDirection,
    entry_price: float,
    current_stop: float,
    extreme_price: float,
    atr_value: float | None,
    atr_multiple: float,
    lock_ratio: float,
    breakeven_buffer_bps: float,
) -> float | None:
    if entry_price <= 0 or extreme_price <= 0:
        return None

    lock_ratio = max(0.0, min(1.0, float(lock_ratio)))
    candidates = [
        runner_breakeven_stop(
            direction=direction,
            entry_price=entry_price,
            buffer_bps=breakeven_buffer_bps,
        )
    ]
    if atr_value is not None and atr_value > 0 and atr_multiple > 0:
        if direction == SignalDirection.SHORT:
            candidates.append(extreme_price + atr_value * atr_multiple)
        else:
            candidates.append(extreme_price - atr_value * atr_multiple)

    favorable_move = (
        entry_price - extreme_price
        if direction == SignalDirection.SHORT
        else extreme_price - entry_price
    )
    if favorable_move > 0 and lock_ratio > 0:
        if direction == SignalDirection.SHORT:
            candidates.append(entry_price - favorable_move * lock_ratio)
        else:
            candidates.append(entry_price + favorable_move * lock_ratio)

    if direction == SignalDirection.SHORT:
        candidate = min(candidates)
        return candidate if candidate < current_stop else None
    candidate = max(candidates)
    return candidate if candidate > current_stop else None


def position_runner_trailing_stop(
    position: Position,
    *,
    atr_value: float | None,
    atr_multiple: float,
    lock_ratio: float,
    breakeven_buffer_bps: float,
) -> float | None:
    if not position.runner_active:
        return None
    extreme_price = position.runner_extreme_price or position.current_price
    return runner_trailing_stop(
        direction=position.direction,
        entry_price=position.entry_price,
        current_stop=position.stop_price,
        extreme_price=extreme_price,
        atr_value=atr_value,
        atr_multiple=atr_multiple,
        lock_ratio=lock_ratio,
        breakeven_buffer_bps=breakeven_buffer_bps,
    )
