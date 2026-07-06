from __future__ import annotations

from datetime import datetime
from typing import Any

from ..analysis.indicators import atr
from ..domain import Candle, Instrument, InstrumentType, Signal, SignalDirection


PENDING_PULLBACK_SHORT_STATE = "WAIT_PULLBACK_SHORT"


def record_pending_pullback_short(
    payload: dict[str, Any],
    signal: Signal,
    *,
    candles: list[Candle],
    timestamp: datetime,
    quantity_lots: int,
    policy_metadata: dict[str, object],
    market_regime: dict[str, object],
    expires_after_bars: int = 8,
    pullback_atr_fraction: float = 0.25,
    min_pullback_pct: float = 0.002,
) -> dict[str, object] | None:
    if signal.direction != SignalDirection.SHORT or quantity_lots < 1:
        return None

    pending_entries = payload.setdefault("pending_entries", [])
    symbol = signal.instrument.symbol
    if any(
        item.get("symbol") == symbol
        and item.get("state") == PENDING_PULLBACK_SHORT_STATE
        for item in pending_entries
    ):
        return None

    atr_value = atr(candles, 14)
    min_pullback = max(signal.entry_price * min_pullback_pct, 0.0)
    atr_pullback = (atr_value or 0.0) * max(0.0, pullback_atr_fraction)
    pullback_size = max(min_pullback, atr_pullback, signal.instrument.tick_size)
    item = {
        "id": f"{symbol}:{timestamp.isoformat()}:pullback-short",
        "state": PENDING_PULLBACK_SHORT_STATE,
        "symbol": symbol,
        "direction": SignalDirection.SHORT.value,
        "created_at": timestamp.isoformat(),
        "last_evaluated_at": timestamp.isoformat(),
        "expires_after_bars": max(1, int(expires_after_bars)),
        "bars_seen": 0,
        "entry_price": float(signal.entry_price),
        "stop_price": float(signal.stop_price),
        "take_profit": float(signal.take_profit),
        "pullback_trigger_price": round(signal.entry_price + pullback_size, 8),
        "failed_rebound_price": float(signal.entry_price),
        "rebound_seen": False,
        "rebound_high": float(signal.entry_price),
        "quantity_lots": max(1, int(quantity_lots)),
        "signal_strength": float(signal.strength),
        "reason": signal.reason,
        "context_score": float(signal.context_score),
        "metadata": {
            **_json_ready_dict(signal.metadata),
            "market_regime": market_regime,
            "regime_policy": policy_metadata,
            "entry_mode": "wait_pullback_short",
        },
        "instrument": _instrument_payload(signal.instrument),
        "trigger": {
            "type": "failed_rebound_short",
            "pullback_atr_fraction": float(pullback_atr_fraction),
            "min_pullback_pct": float(min_pullback_pct),
        },
    }
    pending_entries.append(item)
    return item


def evaluate_pending_entries(
    payload: dict[str, Any],
    history_by_symbol: dict[str, list[Candle]],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    remaining: list[dict[str, object]] = []

    for item in payload.get("pending_entries", []):
        if item.get("state") != PENDING_PULLBACK_SHORT_STATE:
            remaining.append(item)
            continue

        result = _evaluate_pullback_short_item(item, history_by_symbol.get(str(item.get("symbol", "")), []))
        if result["status"] == "waiting":
            remaining.append(dict(result["item"]))
        else:
            results.append(result)

    payload["pending_entries"] = remaining
    return results


def pending_entry_signal(
    item: dict[str, object],
    *,
    reward_to_risk: float,
) -> Signal:
    instrument = _instrument_from_payload(dict(item.get("instrument", {})), str(item["symbol"]))
    entry_price = float(item.get("trigger_price", item.get("entry_price", 0.0)))
    original_stop = float(item.get("stop_price", entry_price))
    rebound_high = float(item.get("rebound_high", entry_price))
    stop_price = max(original_stop, rebound_high + instrument.tick_size)
    if stop_price <= entry_price:
        stop_price = entry_price + max(instrument.tick_size, abs(entry_price) * 0.005)
    risk_per_unit = stop_price - entry_price
    take_profit = entry_price - risk_per_unit * max(0.1, float(reward_to_risk))
    metadata = dict(item.get("metadata", {}))
    metadata["entry_mode"] = "pullback_short"
    metadata["pending_entry"] = {
        "id": item.get("id", ""),
        "state": item.get("state", ""),
        "created_at": item.get("created_at", ""),
        "triggered_at": item.get("triggered_at", ""),
        "bars_seen": item.get("bars_seen", 0),
        "rebound_high": round(rebound_high, 6),
        "pullback_trigger_price": item.get("pullback_trigger_price", 0.0),
        "failed_rebound_price": item.get("failed_rebound_price", 0.0),
        "outcome": "triggered",
    }
    return Signal(
        instrument=instrument,
        direction=SignalDirection.SHORT,
        strength=float(item.get("signal_strength", 0.0)),
        entry_price=entry_price,
        stop_price=stop_price,
        take_profit=take_profit,
        reason=f"pullback-short trigger after {item.get('reason', 'deferred short')}",
        context_score=float(item.get("context_score", 0.0)),
        metadata=metadata,
    )


def pending_entry_quantity_lots(item: dict[str, object]) -> int:
    return max(1, int(item.get("quantity_lots", 1)))


def pending_entry_expired_event(result: dict[str, object]) -> dict[str, object]:
    item = dict(result.get("item", {}))
    return {
        "timestamp": result.get("timestamp", ""),
        "symbol": item.get("symbol", ""),
        "action": "pending-entry",
        "status": "expired",
        "state": item.get("state", ""),
        "reason": result.get("reason", "pending entry expired"),
        "metadata": _pending_event_metadata(item),
    }


def _evaluate_pullback_short_item(
    item: dict[str, object],
    candles: list[Candle],
) -> dict[str, object]:
    if not candles:
        return {"status": "waiting", "item": item}

    last_evaluated_at = _parse_timestamp(item.get("last_evaluated_at")) or _parse_timestamp(item.get("created_at"))
    created_at = _parse_timestamp(item.get("created_at"))
    if created_at is None:
        return {
            "status": "expired",
            "reason": "invalid pending entry timestamp",
            "timestamp": candles[-1].timestamp.isoformat(),
            "item": item,
        }

    new_candles = [
        candle
        for candle in candles
        if candle.timestamp > created_at
        and (last_evaluated_at is None or candle.timestamp > last_evaluated_at)
    ]
    if not new_candles:
        return {"status": "waiting", "item": item}

    updated = dict(item)
    entry_price = float(updated.get("entry_price", 0.0))
    pullback_trigger = float(updated.get("pullback_trigger_price", entry_price))
    failed_rebound_price = float(updated.get("failed_rebound_price", entry_price))
    bars_seen = int(updated.get("bars_seen", 0))
    expires_after_bars = max(1, int(updated.get("expires_after_bars", 8)))
    rebound_seen = bool(updated.get("rebound_seen", False))
    rebound_high = float(updated.get("rebound_high", entry_price))

    for candle in new_candles:
        bars_seen += 1
        rebound_high = max(rebound_high, float(candle.high))
        if candle.high >= pullback_trigger:
            rebound_seen = True

        updated.update(
            {
                "bars_seen": bars_seen,
                "last_evaluated_at": candle.timestamp.isoformat(),
                "rebound_seen": rebound_seen,
                "rebound_high": rebound_high,
            }
        )

        failed_rebound = (
            rebound_seen
            and candle.close <= failed_rebound_price
            and candle.close < candle.open
        )
        if failed_rebound:
            updated["triggered_at"] = candle.timestamp.isoformat()
            updated["trigger_price"] = float(candle.close)
            return {
                "status": "triggered",
                "timestamp": candle.timestamp.isoformat(),
                "reason": "failed rebound confirmed",
                "item": updated,
            }

        if bars_seen >= expires_after_bars:
            return {
                "status": "expired",
                "timestamp": candle.timestamp.isoformat(),
                "reason": "pullback-short trigger expired",
                "item": updated,
            }

    return {"status": "waiting", "item": updated}


def _pending_event_metadata(item: dict[str, object]) -> dict[str, object]:
    return {
        "id": item.get("id", ""),
        "created_at": item.get("created_at", ""),
        "bars_seen": item.get("bars_seen", 0),
        "rebound_seen": item.get("rebound_seen", False),
        "rebound_high": item.get("rebound_high", 0.0),
        "pullback_trigger_price": item.get("pullback_trigger_price", 0.0),
        "failed_rebound_price": item.get("failed_rebound_price", 0.0),
        "quantity_lots": item.get("quantity_lots", 0),
    }


def _instrument_payload(instrument: Instrument) -> dict[str, object]:
    return {
        "symbol": instrument.symbol,
        "instrument_type": instrument.instrument_type.value,
        "figi": instrument.figi,
        "uid": instrument.uid,
        "class_code": instrument.class_code,
        "lot_size": instrument.lot_size,
        "tick_size": instrument.tick_size,
        "currency": instrument.currency,
        "initial_margin_buy": instrument.initial_margin_buy,
        "initial_margin_sell": instrument.initial_margin_sell,
        "tick_value": instrument.tick_value,
    }


def _instrument_from_payload(payload: dict[str, object], fallback_symbol: str) -> Instrument:
    try:
        instrument_type = InstrumentType(str(payload.get("instrument_type", InstrumentType.STOCK.value)))
    except ValueError:
        instrument_type = InstrumentType.STOCK
    return Instrument(
        symbol=str(payload.get("symbol", fallback_symbol)),
        instrument_type=instrument_type,
        figi=str(payload.get("figi", "")),
        uid=str(payload.get("uid", "")),
        class_code=str(payload.get("class_code", "")),
        lot_size=max(1, int(payload.get("lot_size", 1) or 1)),
        tick_size=float(payload.get("tick_size", 0.01) or 0.01),
        currency=str(payload.get("currency", "rub")),
        initial_margin_buy=float(payload.get("initial_margin_buy", 0.0) or 0.0),
        initial_margin_sell=float(payload.get("initial_margin_sell", 0.0) or 0.0),
        tick_value=float(payload.get("tick_value", 0.0) or 0.0),
    )


def _json_ready_dict(payload: dict[str, Any]) -> dict[str, object]:
    ready = _json_ready(payload)
    return ready if isinstance(ready, dict) else {}


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return str(value)


def _parse_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None
