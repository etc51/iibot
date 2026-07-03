from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..domain import Candle, ExitReason, Instrument, InstrumentType, Signal, SignalDirection, TradeRecord
from ..execution.paper import LocalPaperBroker


def signal_feedback_path(state_path: Path) -> Path:
    suffix = state_path.suffix or ".json"
    return state_path.with_name(f"{state_path.stem}_signal_feedback{suffix}")


def load_signal_feedback(path: Path) -> dict[str, list[dict[str, object]]]:
    if not path.exists():
        return {"pending": [], "resolved": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "pending": list(payload.get("pending", [])),
        "resolved": list(payload.get("resolved", [])),
    }


def save_signal_feedback(path: Path, payload: dict[str, list[dict[str, object]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def record_shadow_signal(
    payload: dict[str, list[dict[str, object]]],
    signal: Signal,
    *,
    timestamp: datetime,
    horizon_bars: int,
    quantity_lots: int = 1,
    slippage_bps: float = 0.0,
    commission_bps: float = 0.0,
) -> None:
    if any(item["symbol"] == signal.instrument.symbol for item in payload.get("pending", [])):
        return
    signature = (
        signal.instrument.symbol,
        signal.direction.value,
        timestamp.isoformat(),
    )
    for item in payload.get("pending", []) + payload.get("resolved", []):
        existing_signature = (
            item["symbol"],
            item["direction"],
            item["created_at"],
        )
        if existing_signature == signature:
            return

    payload.setdefault("pending", []).append(
        _shadow_signal_item(
            signal,
            timestamp=timestamp,
            horizon_bars=horizon_bars,
            quantity_lots=quantity_lots,
            slippage_bps=slippage_bps,
            commission_bps=commission_bps,
        )
    )


def resolve_pending_signals(
    payload: dict[str, list[dict[str, object]]],
    history_by_symbol: dict[str, list[Candle]],
) -> list[dict[str, object]]:
    resolved_items: list[dict[str, object]] = []
    remaining: list[dict[str, object]] = []

    for item in payload.get("pending", []):
        candles = history_by_symbol.get(item["symbol"], [])
        resolved = _resolve_signal_item(item, candles)
        if resolved is None:
            remaining.append(item)
            continue
        resolved_items.append(resolved)

    payload["pending"] = remaining
    payload.setdefault("resolved", []).extend(resolved_items)
    return resolved_items


def resolved_feedback_to_trades(payload: dict[str, list[dict[str, object]]]) -> list[TradeRecord]:
    trades: list[TradeRecord] = []
    for item in payload.get("resolved", []):
        gross_pnl = float(item.get("gross_pnl", 0.0))
        net_pnl = float(item.get("net_pnl", gross_pnl))
        trades.append(
            TradeRecord(
                symbol=str(item["symbol"]),
                direction=SignalDirection(str(item["direction"])),
                quantity_lots=int(item.get("quantity_lots", 1)),
                entry_time=datetime.fromisoformat(str(item["created_at"])),
                exit_time=datetime.fromisoformat(str(item["resolved_at"])),
                entry_price=float(item.get("entry_fill_price", item["entry_price"])),
                exit_price=float(item.get("exit_fill_price", item["exit_price"])),
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                reason=str(item["outcome_reason"]),
                signal_strength=float(item.get("signal_strength", 0.0)),
                entry_reason=str(item.get("reason", "shadow-feedback")),
                entry_context_score=float(item.get("context_score", 0.0)),
                entry_metadata=dict(item.get("metadata", {})),
                initial_stop_price=float(item.get("stop_price", item.get("entry_price", 0.0))),
                initial_take_profit=float(item.get("take_profit", item.get("exit_price", 0.0))),
            )
        )
    return trades


def build_trade_evidence(
    closed_trades: list[TradeRecord],
    payload: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    feedback_trades = resolved_feedback_to_trades(payload)
    combined_trades = list(closed_trades)
    seen = {_trade_record_signature(trade) for trade in closed_trades}
    unique_feedback_trades: list[TradeRecord] = []
    duplicate_feedback_trades = 0

    for trade in feedback_trades:
        signature = _trade_record_signature(trade)
        if signature in seen:
            duplicate_feedback_trades += 1
            continue
        seen.add(signature)
        unique_feedback_trades.append(trade)
        combined_trades.append(trade)

    combined_trades.sort(
        key=lambda trade: (
            trade.exit_time,
            trade.entry_time,
            trade.symbol,
            trade.direction.value,
            trade.reason,
        )
    )
    if closed_trades and unique_feedback_trades:
        evidence_source = "closed-trades+signal-feedback"
    elif unique_feedback_trades:
        evidence_source = "signal-feedback"
    else:
        evidence_source = "closed-trades"

    return {
        "trades": combined_trades,
        "evidence_source": evidence_source,
        "counts": {
            "closed_trades": len(closed_trades),
            "feedback_trades": len(feedback_trades),
            "deduplicated_feedback_trades": len(unique_feedback_trades),
            "duplicate_feedback_trades": duplicate_feedback_trades,
            "combined_trades": len(combined_trades),
        },
    }


def simulate_signal_feedback(
    signal: Signal,
    *,
    timestamp: datetime,
    future_candles: list[Candle],
    horizon_bars: int,
    quantity_lots: int = 1,
    slippage_bps: float = 0.0,
    commission_bps: float = 0.0,
) -> dict[str, object] | None:
    item = _shadow_signal_item(
        signal,
        timestamp=timestamp,
        horizon_bars=horizon_bars,
        quantity_lots=quantity_lots,
        slippage_bps=slippage_bps,
        commission_bps=commission_bps,
    )
    return _resolve_signal_item(item, future_candles)


def backfill_signal_feedback_for_symbol(
    payload: dict[str, list[dict[str, object]]],
    *,
    instrument: Instrument,
    candles: list[Candle],
    strategy,
    warmup_bars: int,
    horizon_bars: int,
    max_signals: int = 0,
    quantity_lots: int = 1,
    slippage_bps: float = 0.0,
    commission_bps: float = 0.0,
) -> int:
    strategy.prepare_history(instrument, candles)
    existing_signatures = {
        (str(item["symbol"]), str(item["direction"]), str(item["created_at"]))
        for item in payload.get("pending", []) + payload.get("resolved", [])
    }
    generated = 0
    next_allowed_index = max(0, warmup_bars - 1)

    for index in range(max(0, warmup_bars - 1), len(candles) - 1):
        if index < next_allowed_index:
            continue
        history = candles[: index + 1]
        latest = candles[index]
        signal = strategy.generate_signal(instrument, history)
        if signal is None or not strategy.allows_entry_at(latest.timestamp):
            continue
        signature = (
            signal.instrument.symbol,
            signal.direction.value,
            latest.timestamp.isoformat(),
        )
        if signature in existing_signatures:
            continue

        resolved = simulate_signal_feedback(
            signal,
            timestamp=latest.timestamp,
            future_candles=candles[index + 1 :],
            horizon_bars=horizon_bars,
            quantity_lots=quantity_lots,
            slippage_bps=slippage_bps,
            commission_bps=commission_bps,
        )
        if resolved is None:
            continue

        payload.setdefault("resolved", []).append(resolved)
        existing_signatures.add(signature)
        generated += 1
        next_allowed_index = index + max(1, int(resolved.get("bars_held", 1)))
        if max_signals > 0 and generated >= max_signals:
            break

    payload["resolved"].sort(key=lambda item: str(item["created_at"]))
    return generated


def default_signal_horizon_bars(timeframe: str) -> int:
    mapping = {
        "day": 5,
        "hour": 24,
        "30min": 32,
        "15min": 48,
        "10min": 60,
        "5min": 96,
        "1min": 180,
    }
    return mapping.get(timeframe.lower(), 24)


def _trade_record_signature(trade: TradeRecord) -> tuple[object, ...]:
    return (
        trade.symbol,
        trade.direction.value,
        trade.entry_time.isoformat(),
        trade.exit_time.isoformat(),
        round(trade.entry_price, 8),
        round(trade.exit_price, 8),
        round(trade.gross_pnl, 8),
        round(trade.net_pnl, 8),
        trade.reason,
        round(trade.signal_strength, 8),
    )


def _shadow_signal_item(
    signal: Signal,
    *,
    timestamp: datetime,
    horizon_bars: int,
    quantity_lots: int,
    slippage_bps: float,
    commission_bps: float,
) -> dict[str, object]:
    instrument = signal.instrument
    return {
        "symbol": instrument.symbol,
        "direction": signal.direction.value,
        "created_at": timestamp.isoformat(),
        "entry_price": signal.entry_price,
        "stop_price": signal.stop_price,
        "take_profit": signal.take_profit,
        "signal_strength": signal.strength,
        "reason": signal.reason,
        "context_score": signal.context_score,
        "metadata": _json_ready(signal.metadata),
        "horizon_bars": max(1, horizon_bars),
        "quantity_lots": max(1, int(quantity_lots)),
        "lot_size": max(1, int(instrument.lot_size or 1)),
        "instrument_type": instrument.instrument_type.value,
        "tick_size": float(instrument.tick_size),
        "currency": str(instrument.currency),
        "initial_margin_buy": float(instrument.initial_margin_buy),
        "initial_margin_sell": float(instrument.initial_margin_sell),
        "tick_value": float(instrument.tick_value),
        "slippage_bps": float(slippage_bps),
        "commission_bps": float(commission_bps),
    }


def _resolve_signal_item(item: dict[str, object], candles: list[Candle]) -> dict[str, object] | None:
    created_at = datetime.fromisoformat(str(item["created_at"]))
    future_candles = [candle for candle in candles if candle.timestamp > created_at]
    if not future_candles:
        return None

    direction = SignalDirection(str(item["direction"]))
    stop_price = float(item["stop_price"])
    take_profit = float(item["take_profit"])
    horizon_bars = int(item.get("horizon_bars", 24))

    for index, candle in enumerate(future_candles, start=1):
        exit_price = None
        outcome_reason = None
        if direction == SignalDirection.LONG:
            if candle.low <= stop_price:
                exit_price = stop_price
                outcome_reason = "stop-loss"
            elif candle.high >= take_profit:
                exit_price = take_profit
                outcome_reason = "take-profit"
        else:
            if candle.high >= stop_price:
                exit_price = stop_price
                outcome_reason = "stop-loss"
            elif candle.low <= take_profit:
                exit_price = take_profit
                outcome_reason = "take-profit"

        if exit_price is None and index >= horizon_bars:
            exit_price = candle.close
            outcome_reason = "expired"

        if exit_price is None:
            continue

        trade = _resolve_shadow_trade(
            item,
            exit_price=exit_price,
            resolved_at=candle.timestamp,
            outcome_reason=outcome_reason,
        )
        if trade is None:
            return None
        return {
            **item,
            "resolved_at": candle.timestamp.isoformat(),
            "exit_price": exit_price,
            "entry_fill_price": round(trade.entry_price, 6),
            "exit_fill_price": round(trade.exit_price, 6),
            "gross_pnl": round(trade.gross_pnl, 6),
            "net_pnl": round(trade.net_pnl, 6),
            "entry_commission": round(trade.gross_pnl - trade.net_pnl - _exit_commission_from_item(item, trade), 6),
            "exit_commission": round(_exit_commission_from_item(item, trade), 6),
            "bars_held": index,
            "outcome_reason": outcome_reason,
        }

    return None


def _resolve_shadow_trade(
    item: dict[str, object],
    *,
    exit_price: float,
    resolved_at: datetime,
    outcome_reason: str,
) -> TradeRecord | None:
    instrument = _instrument_from_item(item)
    direction = SignalDirection(str(item["direction"]))
    signal = Signal(
        instrument=instrument,
        direction=direction,
        strength=float(item.get("signal_strength", 0.0)),
        entry_price=float(item["entry_price"]),
        stop_price=float(item["stop_price"]),
        take_profit=float(item["take_profit"]),
        reason="shadow-feedback",
        context_score=float(item.get("context_score", 0.0)),
        metadata=dict(item.get("metadata", {})),
    )
    quantity_lots = max(1, int(item.get("quantity_lots", 1)))
    broker = LocalPaperBroker.fresh(
        initial_cash=1_000_000_000.0,
        slippage_bps=float(item.get("slippage_bps", 0.0)),
        commission_bps=float(item.get("commission_bps", 0.0)),
    )
    broker.open_position(signal, quantity_lots, datetime.fromisoformat(str(item["created_at"])))
    close_reason = _close_reason_for_outcome(outcome_reason)
    trade = broker.close_position(
        instrument.symbol,
        price=exit_price,
        timestamp=resolved_at,
        reason=close_reason,
    )
    return trade


def _instrument_from_item(item: dict[str, object]) -> Instrument:
    instrument_type = str(item.get("instrument_type", InstrumentType.STOCK.value)).strip().lower()
    try:
        normalized_type = InstrumentType(instrument_type)
    except ValueError:
        normalized_type = InstrumentType.STOCK
    return Instrument(
        symbol=str(item["symbol"]),
        instrument_type=normalized_type,
        lot_size=max(1, int(item.get("lot_size", 1) or 1)),
        tick_size=float(item.get("tick_size", 0.01) or 0.01),
        currency=str(item.get("currency", "rub")),
        initial_margin_buy=float(item.get("initial_margin_buy", 0.0) or 0.0),
        initial_margin_sell=float(item.get("initial_margin_sell", 0.0) or 0.0),
        tick_value=float(item.get("tick_value", 0.0) or 0.0),
    )


def _close_reason_for_outcome(outcome_reason: str) -> ExitReason:
    if outcome_reason == "stop-loss":
        return ExitReason.STOP_LOSS
    if outcome_reason == "take-profit":
        return ExitReason.TAKE_PROFIT
    return ExitReason.END_OF_TEST


def _exit_commission_from_item(item: dict[str, object], trade: TradeRecord) -> float:
    quantity_lots = max(1, int(item.get("quantity_lots", 1)))
    lot_size = max(1, int(item.get("lot_size", 1) or 1))
    commission_bps = float(item.get("commission_bps", 0.0))
    exit_notional = abs(trade.exit_price * quantity_lots * lot_size)
    return exit_notional * commission_bps / 10_000


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return str(value)

