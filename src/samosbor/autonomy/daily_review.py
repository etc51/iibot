from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ..analysis.indicators import atr
from ..autonomy.runner import runner_breakeven_stop, runner_extreme_price, runner_trailing_stop
from ..config import AppConfig
from ..domain import Candle, Instrument, PortfolioState, Signal, SignalDirection, TradeRecord
from ..strategy.trend_following import TrendFollowingStrategy
from .entry_confirmation import build_entry_confirmation_context
from .ml_learning import assess_signal_learning, build_entry_candle_context


def daily_review_path(state_path: Path) -> Path:
    suffix = state_path.suffix or ".json"
    return state_path.with_name(f"{state_path.stem}_daily_review{suffix}")


def build_daily_review_payload(
    config: AppConfig,
    portfolio: PortfolioState,
    trades: list[TradeRecord],
    *,
    candles_by_symbol: dict[str, list[Candle]],
    instruments_by_symbol: dict[str, Instrument],
    confirmation_history_by_symbol: dict[str, list[Candle]] | None = None,
    feedback_payload: dict[str, list[dict[str, object]]] | None = None,
    report_date: date | None = None,
    days: int = 1,
    generated_at: datetime | None = None,
    max_signal_rows: int = 250,
    max_ml_candidates: int = 60,
    max_holding_bars: int = 32,
) -> dict[str, object]:
    if days < 1:
        raise ValueError("days must be >= 1")

    generated_at = generated_at or datetime.now(timezone.utc)
    timezone_info = ZoneInfo(config.app.timezone)
    anchor_date = report_date or generated_at.astimezone(timezone_info).date()
    start_at, end_at = _date_window(anchor_date, days=days, timezone_info=timezone_info)
    confirmation_history_by_symbol = confirmation_history_by_symbol or {}
    feedback_payload = feedback_payload or {"resolved": [], "pending": []}

    actual_opened = _select_trades_by_entry(trades, start_at=start_at, end_at=end_at, timezone_info=timezone_info)
    actual_closed = _select_trades_by_exit(trades, start_at=start_at, end_at=end_at, timezone_info=timezone_info)

    stop_multipliers = _review_stop_multipliers(config)
    reward_to_risk_values = _review_reward_to_risk_values(config)
    entry_offsets = [0, 1, 2]

    candidates = _scan_signal_candidates(
        config,
        candles_by_symbol=candles_by_symbol,
        instruments_by_symbol=instruments_by_symbol,
        confirmation_history_by_symbol=confirmation_history_by_symbol,
        feedback_payload=feedback_payload,
        start_at=start_at,
        end_at=end_at,
        timezone_info=timezone_info,
        stop_multipliers=stop_multipliers,
        reward_to_risk_values=reward_to_risk_values,
        entry_offsets=entry_offsets,
        max_holding_bars=max_holding_bars,
        max_ml_candidates=max_ml_candidates,
    )
    _mark_actual_matches(candidates, actual_opened, timeframe=config.data.timeframe)

    actual_reviews = [
        _review_actual_trade(
            trade,
            config,
            candles_by_symbol.get(trade.symbol, []),
            instruments_by_symbol.get(trade.symbol),
            timezone_info=timezone_info,
            stop_multipliers=stop_multipliers,
            reward_to_risk_values=reward_to_risk_values,
            entry_offsets=entry_offsets,
            max_holding_bars=max_holding_bars,
        )
        for trade in actual_opened
    ]

    tradable_candidates = [row for row in candidates if row["tradable"]]
    missed_opportunities = [
        row
        for row in tradable_candidates
        if not row["actual_match"] and float(row["best_plan"]["net_pnl_per_lot_rub"]) > 0.0
    ]
    missed_opportunities.sort(key=lambda row: float(row["best_plan"]["net_pnl_per_lot_rub"]), reverse=True)

    weak_candidates = [
        row
        for row in tradable_candidates
        if float(row["best_plan"]["net_pnl_per_lot_rub"]) < 0.0
    ]
    weak_candidates.sort(key=lambda row: float(row["best_plan"]["net_pnl_per_lot_rub"]))

    grid_summary = _grid_summary(tradable_candidates)
    recommendations = _daily_recommendations(
        actual_reviews,
        missed_opportunities,
        grid_summary,
        config=config,
    )
    training_examples = _training_examples(candidates, actual_reviews)

    return {
        "generated_at": generated_at.isoformat(),
        "period": {
            "timezone": config.app.timezone,
            "days": days,
            "report_date": anchor_date.isoformat(),
            "start_at": start_at.isoformat(),
            "end_at": end_at.isoformat(),
        },
        "parameters": {
            "timeframe": config.data.timeframe,
            "confirmation_timeframe": config.strategy.entry_confirmation_timeframe,
            "entry_offsets_bars": entry_offsets,
            "stop_multipliers": stop_multipliers,
            "reward_to_risk_values": reward_to_risk_values,
            "max_holding_bars": max_holding_bars,
            "slippage_bps": config.execution.slippage_bps,
            "commission_bps": config.execution.commission_bps,
            "ml_max_candidates": max_ml_candidates,
        },
        "portfolio": {
            "realized_pnl_rub": round(portfolio.realized_pnl, 2),
            "open_positions": len(portfolio.positions),
            "trading_halted": portfolio.trading_halted,
        },
        "actual_day": {
            "opened_trades": len(actual_opened),
            "closed_trades": len(actual_closed),
            "opened_summary": _trade_summary(actual_opened),
            "closed_summary": _trade_summary(actual_closed),
        },
        "signal_scan": {
            "symbols_scanned": len(candles_by_symbol),
            "candidate_signals": len(candidates),
            "tradable_candidates": len(tradable_candidates),
            "blocked_candidates": len(candidates) - len(tradable_candidates),
            "ml_assessed_candidates": sum(1 for row in candidates if row["ml_learning"].get("available")),
            "entry_confirmation_blocked": sum(
                1
                for row in candidates
                if "entry-confirmation" in row.get("block_tags", [])
            ),
            "ml_blocked_candidates": sum(1 for row in candidates if "ml-learning" in row.get("block_tags", [])),
            "matched_actual_entries": sum(1 for row in candidates if row["actual_match"]),
            "missed_positive_opportunities": len(missed_opportunities),
            "weak_tradable_candidates": len(weak_candidates),
        },
        "grid_summary": grid_summary,
        "missed_opportunities": [_public_candidate(row) for row in missed_opportunities[:20]],
        "weak_candidates": [_public_candidate(row) for row in weak_candidates[:20]],
        "actual_trade_reviews": actual_reviews,
        "recommendations": recommendations,
        "training_examples": training_examples[:500],
        "candidate_signals": [_public_candidate(row) for row in _rank_candidates(candidates)[:max_signal_rows]],
    }


def save_daily_review(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_daily_review(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_daily_review(output_dir / "daily_review.json", payload)
    (output_dir / "daily_review.md").write_text(_render_markdown(payload), encoding="utf-8")
    _write_training_csv(output_dir / "training_examples.csv", list(payload.get("training_examples", [])))


def _scan_signal_candidates(
    config: AppConfig,
    *,
    candles_by_symbol: dict[str, list[Candle]],
    instruments_by_symbol: dict[str, Instrument],
    confirmation_history_by_symbol: dict[str, list[Candle]],
    feedback_payload: dict[str, list[dict[str, object]]],
    start_at: datetime,
    end_at: datetime,
    timezone_info: ZoneInfo,
    stop_multipliers: list[float],
    reward_to_risk_values: list[float],
    entry_offsets: list[int],
    max_holding_bars: int,
    max_ml_candidates: int,
) -> list[dict[str, object]]:
    strategy = TrendFollowingStrategy(config.strategy, timeframe=config.data.timeframe)
    strategy.prepare_market_context(candles_by_symbol)
    for symbol, candles in candles_by_symbol.items():
        instrument = instruments_by_symbol.get(symbol)
        if instrument is not None:
            strategy.prepare_history(instrument, candles)

    rows: list[dict[str, object]] = []
    pending_ml: list[tuple[int, Signal]] = []
    for symbol, candles in sorted(candles_by_symbol.items()):
        instrument = instruments_by_symbol.get(symbol)
        if instrument is None:
            continue
        for index, candle in enumerate(candles):
            localized = candle.timestamp.astimezone(timezone_info)
            if not (start_at <= localized < end_at):
                continue
            history = candles[: index + 1]
            if len(history) < config.backtest.warmup_bars:
                continue
            signal = strategy.generate_signal(instrument, history)
            if signal is None:
                continue

            entry_confirmation = build_entry_confirmation_context(
                confirmation_history_by_symbol.get(symbol, []),
                signal.direction.value,
                signal_timestamp=candle.timestamp,
                primary_timeframe=config.data.timeframe,
                confirmation_timeframe=config.strategy.entry_confirmation_timeframe,
                min_bars=config.strategy.entry_confirmation_min_bars,
                max_adverse_ret=config.strategy.entry_confirmation_max_adverse_ret,
            )
            block_reasons: list[str] = []
            block_tags: list[str] = []
            schedule_block = strategy.entry_block_reason_for_instrument(instrument, candle.timestamp, signal.direction)
            if schedule_block:
                block_reasons.append(schedule_block)
                block_tags.append("strategy-rule")
            if entry_confirmation.get("available") and entry_confirmation.get("against_direction"):
                block_reasons.append(
                    f"entry blocked by {entry_confirmation.get('timeframe')} confirmation "
                    f"({entry_confirmation.get('reason')})"
                )
                block_tags.append("entry-confirmation")

            best_plan, configured_plan, grid = _best_plan_for_signal(
                signal,
                candles,
                index,
                instrument,
                config,
                stop_multipliers=stop_multipliers,
                reward_to_risk_values=reward_to_risk_values,
                entry_offsets=entry_offsets,
                max_holding_bars=max_holding_bars,
                end_at=end_at,
                timezone_info=timezone_info,
            )
            row = {
                "symbol": symbol,
                "direction": signal.direction.value,
                "timestamp": candle.timestamp.isoformat(),
                "local_time": localized.isoformat(),
                "entry_index": index,
                "entry_price": round(signal.entry_price, 6),
                "signal_strength": round(signal.strength, 4),
                "context_score": round(signal.context_score, 4),
                "reason": signal.reason,
                "entry_candle": build_entry_candle_context(history, signal.direction.value),
                "entry_confirmation": entry_confirmation,
                "ml_learning": {"available": False, "reason": "not assessed yet", "blocks_entry": False},
                "runtime_block_reasons": block_reasons,
                "block_tags": block_tags,
                "tradable": not block_reasons,
                "actual_match": False,
                "actual_trade_net_pnl_rub": None,
                "configured_plan": configured_plan,
                "best_plan": best_plan,
                "grid": grid,
            }
            rows.append(row)
            pending_ml.append((len(rows) - 1, signal))

    ml_indexes = _ml_candidate_indexes(rows, max_ml_candidates=max_ml_candidates)
    for row_index, signal in pending_ml:
        row = rows[row_index]
        if row_index not in ml_indexes:
            row["ml_learning"] = {
                "available": False,
                "reason": "skipped by daily review ml cap",
                "blocks_entry": False,
            }
            continue
        ml_learning = assess_signal_learning(
            signal,
            feedback_payload,
            timestamp=datetime.fromisoformat(str(row["timestamp"])),
            quantity_lots=1,
            timezone_name=config.app.timezone,
            slippage_bps=config.execution.slippage_bps,
            commission_bps=config.execution.commission_bps,
        )
        row["ml_learning"] = ml_learning
        if ml_learning.get("blocks_entry"):
            row["runtime_block_reasons"].append(str(ml_learning.get("reason", "entry blocked by ML learning")))
            row["block_tags"].append("ml-learning")
            row["tradable"] = False
    return rows


def _ml_candidate_indexes(rows: list[dict[str, object]], *, max_ml_candidates: int) -> set[int]:
    if max_ml_candidates <= 0:
        return set()
    if len(rows) <= max_ml_candidates:
        return set(range(len(rows)))
    ranked = sorted(
        enumerate(rows),
        key=lambda item: (
            float(item[1]["signal_strength"]),
            float(item[1]["best_plan"]["net_pnl_per_lot_rub"]),
        ),
        reverse=True,
    )
    return {index for index, _ in ranked[:max_ml_candidates]}


def _best_plan_for_signal(
    signal: Signal,
    candles: list[Candle],
    entry_index: int,
    instrument: Instrument,
    config: AppConfig,
    *,
    stop_multipliers: list[float],
    reward_to_risk_values: list[float],
    entry_offsets: list[int],
    max_holding_bars: int,
    end_at: datetime,
    timezone_info: ZoneInfo,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    base_atr = _signal_atr_estimate(signal, candles[: entry_index + 1], config)
    plans: list[dict[str, object]] = []
    for entry_offset in entry_offsets:
        delayed_index = entry_index + entry_offset
        if delayed_index >= len(candles):
            continue
        if candles[delayed_index].timestamp.astimezone(timezone_info) >= end_at:
            continue
        for stop_multiple in stop_multipliers:
            for reward_to_risk in reward_to_risk_values:
                plans.append(
                    _simulate_plan(
                        direction=signal.direction,
                        candles=candles,
                        entry_index=delayed_index,
                        entry_offset_bars=entry_offset,
                        entry_price=candles[delayed_index].close,
                        base_atr=base_atr,
                        stop_multiple=stop_multiple,
                        reward_to_risk=reward_to_risk,
                        lot_size=instrument.lot_size,
                        slippage_bps=config.execution.slippage_bps,
                        commission_bps=config.execution.commission_bps,
                        max_holding_bars=max_holding_bars,
                        end_at=end_at,
                        timezone_info=timezone_info,
                        runner_enabled=config.strategy.take_profit_activates_runner,
                        runner_breakeven_buffer_bps=config.strategy.runner_breakeven_buffer_bps,
                        runner_trailing_atr_multiple=config.strategy.runner_trailing_atr_multiple,
                        runner_profit_lock_ratio=config.strategy.runner_profit_lock_ratio,
                        runner_atr_window=config.strategy.atr_window,
                    )
                )
    if not plans:
        empty = _empty_plan(signal.entry_price)
        return empty, empty, []
    plans.sort(key=lambda row: float(row["net_pnl_per_lot_rub"]), reverse=True)
    configured_plan = min(
        plans,
        key=lambda row: (
            abs(int(row["entry_offset_bars"])),
            abs(float(row["stop_multiple"]) - config.strategy.atr_stop_multiple),
            abs(float(row["reward_to_risk"]) - config.strategy.reward_to_risk),
        ),
    )
    return plans[0], configured_plan, plans


def _review_actual_trade(
    trade: TradeRecord,
    config: AppConfig,
    candles: list[Candle],
    instrument: Instrument | None,
    *,
    timezone_info: ZoneInfo,
    stop_multipliers: list[float],
    reward_to_risk_values: list[float],
    entry_offsets: list[int],
    max_holding_bars: int,
) -> dict[str, object]:
    index = _entry_index_for_trade(candles, trade.entry_time, timeframe=config.data.timeframe)
    units = _trade_units(trade)
    actual_per_lot = trade.net_pnl / max(1, trade.quantity_lots)
    result = {
        "symbol": trade.symbol,
        "direction": trade.direction.value,
        "entry_time": trade.entry_time.isoformat(),
        "entry_time_local": trade.entry_time.astimezone(timezone_info).isoformat(),
        "exit_time": trade.exit_time.isoformat(),
        "net_pnl_rub": round(trade.net_pnl, 2),
        "net_pnl_per_lot_rub": round(actual_per_lot, 2),
        "entry_price": round(trade.entry_price, 6),
        "exit_price": round(trade.exit_price, 6),
        "exit_reason": trade.reason,
        "best_plan": None,
        "improvement_per_lot_rub": None,
        "diagnosis": [],
    }
    if index is None or instrument is None:
        result["diagnosis"] = ["missing candles for actual trade review"]
        return result

    base_atr = _actual_trade_atr_estimate(trade, candles[: index + 1], config)
    plans: list[dict[str, object]] = []
    end_at = trade.exit_time.astimezone(timezone_info).replace(hour=23, minute=59, second=59)
    for entry_offset in entry_offsets:
        delayed_index = index + entry_offset
        if delayed_index >= len(candles):
            continue
        for stop_multiple in stop_multipliers:
            for reward_to_risk in reward_to_risk_values:
                plans.append(
                    _simulate_plan(
                        direction=trade.direction,
                        candles=candles,
                        entry_index=delayed_index,
                        entry_offset_bars=entry_offset,
                        entry_price=candles[delayed_index].close,
                        base_atr=base_atr,
                        stop_multiple=stop_multiple,
                        reward_to_risk=reward_to_risk,
                        lot_size=instrument.lot_size,
                        slippage_bps=config.execution.slippage_bps,
                        commission_bps=config.execution.commission_bps,
                        max_holding_bars=max_holding_bars,
                        end_at=end_at,
                        timezone_info=timezone_info,
                        quantity_lots=max(1, trade.quantity_lots),
                        runner_enabled=config.strategy.take_profit_activates_runner,
                        runner_breakeven_buffer_bps=config.strategy.runner_breakeven_buffer_bps,
                        runner_trailing_atr_multiple=config.strategy.runner_trailing_atr_multiple,
                        runner_profit_lock_ratio=config.strategy.runner_profit_lock_ratio,
                        runner_atr_window=config.strategy.atr_window,
                    )
                )
    if not plans:
        result["diagnosis"] = ["no alternative exit plan could be simulated"]
        return result

    plans.sort(key=lambda row: float(row["net_pnl_per_lot_rub"]), reverse=True)
    best_plan = plans[0]
    improvement = float(best_plan["net_pnl_per_lot_rub"]) - actual_per_lot
    result["best_plan"] = best_plan
    result["improvement_per_lot_rub"] = round(improvement, 2)
    result["diagnosis"] = _actual_trade_diagnosis(trade, best_plan, improvement, units=units)
    return result


def _simulate_plan(
    *,
    direction: SignalDirection,
    candles: list[Candle],
    entry_index: int,
    entry_offset_bars: int,
    entry_price: float,
    base_atr: float,
    stop_multiple: float,
    reward_to_risk: float,
    lot_size: int,
    slippage_bps: float,
    commission_bps: float,
    max_holding_bars: int,
    end_at: datetime,
    timezone_info: ZoneInfo,
    quantity_lots: int = 1,
    runner_enabled: bool = False,
    runner_breakeven_buffer_bps: float = 0.0,
    runner_trailing_atr_multiple: float = 0.0,
    runner_profit_lock_ratio: float = 0.0,
    runner_atr_window: int = 14,
) -> dict[str, object]:
    distance = max(1e-9, base_atr * stop_multiple)
    if direction == SignalDirection.LONG:
        stop_price = entry_price - distance
        take_profit = entry_price + distance * reward_to_risk
    else:
        stop_price = entry_price + distance
        take_profit = entry_price - distance * reward_to_risk

    initial_stop_price = stop_price
    exit_price = candles[entry_index].close
    exit_time = candles[entry_index].timestamp
    exit_reason = "no-future-candle"
    exit_index = entry_index
    runner_active = False
    runner_activated_at = ""
    runner_activation_price = 0.0
    runner_extreme = 0.0
    end_index = min(len(candles) - 1, entry_index + max(1, max_holding_bars))
    for index in range(entry_index + 1, end_index + 1):
        candle = candles[index]
        if candle.timestamp.astimezone(timezone_info) >= end_at:
            break
        exit_index = index
        exit_time = candle.timestamp
        if direction == SignalDirection.LONG:
            if candle.low <= stop_price:
                exit_price = stop_price
                exit_reason = _simulated_stop_reason(direction, entry_price, stop_price)
                break
            if candle.high >= take_profit and not runner_active:
                if runner_enabled:
                    runner_active = True
                    runner_activated_at = candle.timestamp.isoformat()
                    runner_activation_price = take_profit
                    runner_extreme = runner_extreme_price(
                        direction=direction,
                        current_extreme=runner_extreme,
                        candle=candle,
                        activation_price=take_profit,
                    )
                    stop_price = runner_breakeven_stop(
                        direction=direction,
                        entry_price=entry_price,
                        buffer_bps=runner_breakeven_buffer_bps,
                    )
                else:
                    exit_price = take_profit
                    exit_reason = "take-profit"
                    break
        else:
            if candle.high >= stop_price:
                exit_price = stop_price
                exit_reason = _simulated_stop_reason(direction, entry_price, stop_price)
                break
            if candle.low <= take_profit and not runner_active:
                if runner_enabled:
                    runner_active = True
                    runner_activated_at = candle.timestamp.isoformat()
                    runner_activation_price = take_profit
                    runner_extreme = runner_extreme_price(
                        direction=direction,
                        current_extreme=runner_extreme,
                        candle=candle,
                        activation_price=take_profit,
                    )
                    stop_price = runner_breakeven_stop(
                        direction=direction,
                        entry_price=entry_price,
                        buffer_bps=runner_breakeven_buffer_bps,
                    )
                else:
                    exit_price = take_profit
                    exit_reason = "take-profit"
                    break
        if runner_active:
            runner_extreme = runner_extreme_price(
                direction=direction,
                current_extreme=runner_extreme,
                candle=candle,
                activation_price=runner_activation_price or take_profit,
            )
            new_stop = runner_trailing_stop(
                direction=direction,
                entry_price=entry_price,
                current_stop=stop_price,
                extreme_price=runner_extreme,
                atr_value=atr(candles[: index + 1], runner_atr_window),
                atr_multiple=runner_trailing_atr_multiple,
                lock_ratio=runner_profit_lock_ratio,
                breakeven_buffer_bps=runner_breakeven_buffer_bps,
            )
            if new_stop is not None:
                stop_price = new_stop
        exit_price = candle.close
        exit_reason = "time-exit"

    gross, net = _hypothetical_pnl(
        direction,
        entry_price=entry_price,
        exit_price=exit_price,
        lot_size=lot_size,
        quantity_lots=quantity_lots,
        slippage_bps=slippage_bps,
        commission_bps=commission_bps,
    )
    risk_per_lot = distance * max(1, lot_size)
    return {
        "entry_offset_bars": entry_offset_bars,
        "entry_index": entry_index,
        "exit_index": exit_index,
        "entry_time": candles[entry_index].timestamp.isoformat(),
        "exit_time": exit_time.isoformat(),
        "entry_price": round(entry_price, 6),
        "exit_price": round(exit_price, 6),
        "stop_price": round(initial_stop_price, 6),
        "final_stop_price": round(stop_price, 6),
        "take_profit": round(take_profit, 6),
        "stop_multiple": round(stop_multiple, 4),
        "reward_to_risk": round(reward_to_risk, 4),
        "runner_enabled": bool(runner_enabled),
        "runner_activated": runner_active or bool(runner_activated_at),
        "runner_activated_at": runner_activated_at,
        "runner_extreme_price": round(runner_extreme, 6),
        "exit_reason": exit_reason,
        "gross_pnl_per_lot_rub": round(gross / max(1, quantity_lots), 2),
        "net_pnl_per_lot_rub": round(net / max(1, quantity_lots), 2),
        "realized_r_net": round((net / max(1, quantity_lots)) / risk_per_lot, 4) if risk_per_lot > 0 else 0.0,
    }


def _simulated_stop_reason(direction: SignalDirection, entry_price: float, stop_price: float) -> str:
    if math.isclose(stop_price, entry_price, rel_tol=1e-9, abs_tol=1e-9):
        return "breakeven-stop"
    if direction == SignalDirection.LONG and stop_price > entry_price:
        return "profit-protect-stop"
    if direction == SignalDirection.SHORT and stop_price < entry_price:
        return "profit-protect-stop"
    return "stop-loss"


def _hypothetical_pnl(
    direction: SignalDirection,
    *,
    entry_price: float,
    exit_price: float,
    lot_size: int,
    quantity_lots: int,
    slippage_bps: float,
    commission_bps: float,
) -> tuple[float, float]:
    units = max(1, lot_size) * max(1, quantity_lots)
    if direction == SignalDirection.LONG:
        entry_fill = _slipped_price(entry_price, is_buy=True, slippage_bps=slippage_bps)
        exit_fill = _slipped_price(exit_price, is_buy=False, slippage_bps=slippage_bps)
        gross = (exit_fill - entry_fill) * units
    else:
        entry_fill = _slipped_price(entry_price, is_buy=False, slippage_bps=slippage_bps)
        exit_fill = _slipped_price(exit_price, is_buy=True, slippage_bps=slippage_bps)
        gross = (entry_fill - exit_fill) * units
    commission = (abs(entry_fill * units) + abs(exit_fill * units)) * commission_bps / 10_000
    return gross, gross - commission


def _slipped_price(price: float, *, is_buy: bool, slippage_bps: float) -> float:
    factor = 1 + slippage_bps / 10_000
    return price * factor if is_buy else price / factor


def _signal_atr_estimate(signal: Signal, candles: list[Candle], config: AppConfig) -> float:
    configured_multiple = max(1e-9, config.strategy.atr_stop_multiple)
    distance = abs(signal.entry_price - signal.stop_price)
    if distance > 0:
        return max(1e-9, distance / configured_multiple)
    value = atr(candles, config.strategy.atr_window)
    return max(1e-9, float(value or 0.0))


def _actual_trade_atr_estimate(trade: TradeRecord, candles: list[Candle], config: AppConfig) -> float:
    configured_multiple = max(1e-9, config.strategy.atr_stop_multiple)
    distance = abs(float(trade.entry_price) - float(trade.initial_stop_price or 0.0))
    if distance > 0:
        return max(1e-9, distance / configured_multiple)
    value = atr(candles, config.strategy.atr_window)
    return max(1e-9, float(value or 0.0))


def _mark_actual_matches(candidates: list[dict[str, object]], trades: list[TradeRecord], *, timeframe: str) -> None:
    duration = _timeframe_duration(timeframe)
    for row in candidates:
        timestamp = datetime.fromisoformat(str(row["timestamp"]))
        for trade in trades:
            if trade.symbol != row["symbol"]:
                continue
            if trade.direction.value != row["direction"]:
                continue
            if abs((trade.entry_time - timestamp).total_seconds()) <= duration.total_seconds():
                row["actual_match"] = True
                row["actual_trade_net_pnl_rub"] = round(trade.net_pnl, 2)
                break


def _grid_summary(candidates: list[dict[str, object]]) -> dict[str, object]:
    plans = [plan for row in candidates for plan in row.get("grid", [])]
    return {
        "best_entry_offset_bars": dict(Counter(str(row["best_plan"]["entry_offset_bars"]) for row in candidates)),
        "best_stop_multiple": dict(Counter(str(row["best_plan"]["stop_multiple"]) for row in candidates)),
        "best_reward_to_risk": dict(Counter(str(row["best_plan"]["reward_to_risk"]) for row in candidates)),
        "grid_rows": _grid_group_rows(plans),
    }


def _grid_group_rows(plans: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, object], list[dict[str, object]]] = defaultdict(list)
    for plan in plans:
        key = (plan["entry_offset_bars"], plan["stop_multiple"], plan["reward_to_risk"])
        grouped[key].append(plan)
    rows = []
    for (entry_offset, stop_multiple, reward_to_risk), group in grouped.items():
        pnl = [float(item["net_pnl_per_lot_rub"]) for item in group]
        rows.append(
            {
                "entry_offset_bars": entry_offset,
                "stop_multiple": stop_multiple,
                "reward_to_risk": reward_to_risk,
                "samples": len(group),
                "avg_net_pnl_per_lot_rub": round(sum(pnl) / len(pnl), 2),
                "win_rate_pct": round(sum(1 for value in pnl if value > 0) / len(pnl) * 100.0, 3),
            }
        )
    rows.sort(key=lambda row: float(row["avg_net_pnl_per_lot_rub"]), reverse=True)
    return rows


def _daily_recommendations(
    actual_reviews: list[dict[str, object]],
    missed_opportunities: list[dict[str, object]],
    grid_summary: dict[str, object],
    *,
    config: AppConfig,
) -> list[dict[str, object]]:
    recommendations: list[dict[str, object]] = []
    improvable_losses = [
        row
        for row in actual_reviews
        if float(row.get("net_pnl_rub", 0.0)) < 0 and float(row.get("improvement_per_lot_rub") or 0.0) > 0
    ]
    if improvable_losses:
        recommendations.append(
            {
                "action": "review-losing-entry-exits",
                "confidence": _confidence(len(improvable_losses)),
                "reason": f"{len(improvable_losses)} losing trade(s) had better stop/take/entry-offset alternatives in day review.",
                "mode": "learn",
                "blocks_entry": False,
            }
        )

    if missed_opportunities:
        recommendations.append(
            {
                "action": "learn-missed-positive-setups",
                "confidence": _confidence(len(missed_opportunities)),
                "reason": f"{len(missed_opportunities)} positive hindsight setup(s) were not matched by actual entries.",
                "top_symbols": [row["symbol"] for row in missed_opportunities[:5]],
                "mode": "observe",
            }
        )

    grid_rows = list(grid_summary.get("grid_rows", []))
    if grid_rows:
        best = grid_rows[0]
        current = min(
            grid_rows,
            key=lambda row: (
                abs(int(row["entry_offset_bars"])),
                abs(float(row["stop_multiple"]) - config.strategy.atr_stop_multiple),
                abs(float(row["reward_to_risk"]) - config.strategy.reward_to_risk),
            ),
        )
        if float(best["avg_net_pnl_per_lot_rub"]) > float(current["avg_net_pnl_per_lot_rub"]):
            recommendations.append(
                {
                    "action": "test-exit-grid-candidate",
                    "confidence": "low",
                    "reason": "Daily grid found a better average exit setup; keep it as evidence until repeated.",
                    "candidate": best,
                    "current_like": current,
                    "mode": "observe",
                }
            )

    if not recommendations:
        recommendations.append(
            {
                "action": "keep-current-rules",
                "confidence": "medium" if actual_reviews else "low",
                "reason": "Daily review did not find a repeated enough improvement pattern.",
            }
        )
    return recommendations


def _training_examples(
    candidates: list[dict[str, object]],
    actual_reviews: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in candidates:
        best = row["best_plan"]
        configured = row["configured_plan"]
        rows.append(
            {
                "source": "candidate_signal",
                "symbol": row["symbol"],
                "direction": row["direction"],
                "timestamp": row["timestamp"],
                "signal_strength": row["signal_strength"],
                "context_score": row["context_score"],
                "tradable": row["tradable"],
                "actual_match": row["actual_match"],
                "configured_net_pnl_per_lot_rub": configured["net_pnl_per_lot_rub"],
                "best_net_pnl_per_lot_rub": best["net_pnl_per_lot_rub"],
                "best_entry_offset_bars": best["entry_offset_bars"],
                "best_stop_multiple": best["stop_multiple"],
                "best_reward_to_risk": best["reward_to_risk"],
                "label_profitable_best": float(best["net_pnl_per_lot_rub"]) > 0,
                "label_profitable_configured": float(configured["net_pnl_per_lot_rub"]) > 0,
            }
        )
    for row in actual_reviews:
        best_plan = row.get("best_plan") or {}
        rows.append(
            {
                "source": "actual_trade",
                "symbol": row["symbol"],
                "direction": row["direction"],
                "timestamp": row["entry_time"],
                "actual_net_pnl_per_lot_rub": row["net_pnl_per_lot_rub"],
                "best_net_pnl_per_lot_rub": best_plan.get("net_pnl_per_lot_rub"),
                "best_entry_offset_bars": best_plan.get("entry_offset_bars"),
                "best_stop_multiple": best_plan.get("stop_multiple"),
                "best_reward_to_risk": best_plan.get("reward_to_risk"),
                "improvement_per_lot_rub": row.get("improvement_per_lot_rub"),
                "label_actual_profitable": float(row["net_pnl_per_lot_rub"]) > 0,
            }
        )
    return rows


def _public_candidate(row: dict[str, object]) -> dict[str, object]:
    return {
        key: row[key]
        for key in [
            "symbol",
            "direction",
            "timestamp",
            "local_time",
            "entry_price",
            "signal_strength",
            "context_score",
            "reason",
            "runtime_block_reasons",
            "block_tags",
            "tradable",
            "actual_match",
            "actual_trade_net_pnl_rub",
            "entry_candle",
            "entry_confirmation",
            "ml_learning",
            "configured_plan",
            "best_plan",
        ]
        if key in row
    }


def _rank_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        candidates,
        key=lambda row: (
            bool(row.get("tradable")),
            float(row["best_plan"]["net_pnl_per_lot_rub"]),
            float(row["signal_strength"]),
        ),
        reverse=True,
    )


def _actual_trade_diagnosis(
    trade: TradeRecord,
    best_plan: dict[str, object],
    improvement: float,
    *,
    units: float,
) -> list[str]:
    notes: list[str] = []
    if improvement <= 0:
        notes.append("actual execution was not worse than tested alternatives")
        return notes
    if trade.net_pnl < 0 and float(best_plan["net_pnl_per_lot_rub"]) > 0:
        notes.append("tested alternative could have turned this loss into profit")
    if int(best_plan["entry_offset_bars"]) > 0:
        notes.append(f"waiting {best_plan['entry_offset_bars']} bar(s) improved outcome")
    if best_plan["exit_reason"] == "take-profit":
        notes.append("alternative plan reached take-profit before deterioration")
    if best_plan["exit_reason"] == "stop-loss" and trade.net_pnl < 0:
        notes.append("even best tested plan still hit stop; setup quality needs review")
    if abs(float(best_plan["net_pnl_per_lot_rub"])) * max(1.0, units) > abs(trade.net_pnl) * 1.25:
        notes.append("stop/take geometry materially changed realized result")
    return notes or ["alternative improved result, but no dominant cause was isolated"]


def _review_stop_multipliers(config: AppConfig) -> list[float]:
    values = {1.0, 1.25, 1.5, 2.0, float(config.strategy.atr_stop_multiple)}
    values.update(float(value) for value in config.research.atr_stop_multipliers)
    return sorted(value for value in values if value > 0)


def _review_reward_to_risk_values(config: AppConfig) -> list[float]:
    values = {1.5, 2.0, 2.5, 3.0, float(config.strategy.reward_to_risk)}
    values.update(float(value) for value in config.research.reward_to_risk_values)
    return sorted(value for value in values if value > 0)


def _date_window(anchor_date: date, *, days: int, timezone_info: ZoneInfo) -> tuple[datetime, datetime]:
    start_date = anchor_date - timedelta(days=days - 1)
    return (
        datetime.combine(start_date, time.min, tzinfo=timezone_info),
        datetime.combine(anchor_date + timedelta(days=1), time.min, tzinfo=timezone_info),
    )


def _select_trades_by_entry(
    trades: list[TradeRecord],
    *,
    start_at: datetime,
    end_at: datetime,
    timezone_info: ZoneInfo,
) -> list[TradeRecord]:
    return [
        trade
        for trade in trades
        if start_at <= trade.entry_time.astimezone(timezone_info) < end_at
    ]


def _select_trades_by_exit(
    trades: list[TradeRecord],
    *,
    start_at: datetime,
    end_at: datetime,
    timezone_info: ZoneInfo,
) -> list[TradeRecord]:
    return [
        trade
        for trade in trades
        if start_at <= trade.exit_time.astimezone(timezone_info) < end_at
    ]


def _trade_summary(trades: list[TradeRecord]) -> dict[str, object]:
    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "net_pnl_rub": 0.0,
            "win_rate_pct": 0.0,
            "expectancy_rub": 0.0,
        }
    pnl = [trade.net_pnl for trade in trades]
    wins = [value for value in pnl if value > 0]
    losses = [value for value in pnl if value < 0]
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "net_pnl_rub": round(sum(pnl), 2),
        "win_rate_pct": round(len(wins) / len(trades) * 100.0, 3),
        "expectancy_rub": round(sum(pnl) / len(trades), 2),
        "gross_profit_rub": round(sum(wins), 2),
        "gross_loss_rub": round(abs(sum(losses)), 2),
    }


def _entry_index_for_trade(candles: list[Candle], entry_time: datetime, *, timeframe: str) -> int | None:
    if not candles:
        return None
    best_index = None
    best_lag = None
    for index, candle in enumerate(candles):
        lag = abs((candle.timestamp - entry_time).total_seconds())
        if best_lag is None or lag < best_lag:
            best_lag = lag
            best_index = index
    duration = _timeframe_duration(timeframe).total_seconds()
    if best_index is None or best_lag is None or best_lag > duration:
        return None
    return best_index


def _trade_units(trade: TradeRecord) -> float:
    if trade.direction == SignalDirection.LONG:
        price_delta = trade.exit_price - trade.entry_price
    else:
        price_delta = trade.entry_price - trade.exit_price
    if abs(price_delta) > 1e-12 and abs(trade.gross_pnl) > 1e-12:
        return abs(trade.gross_pnl / price_delta)
    return max(1.0, float(trade.quantity_lots))


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


def _empty_plan(entry_price: float) -> dict[str, object]:
    return {
        "entry_offset_bars": 0,
        "entry_index": 0,
        "exit_index": 0,
        "entry_time": "",
        "exit_time": "",
        "entry_price": round(entry_price, 6),
        "exit_price": round(entry_price, 6),
        "stop_price": round(entry_price, 6),
        "take_profit": round(entry_price, 6),
        "stop_multiple": 0.0,
        "reward_to_risk": 0.0,
        "exit_reason": "not-simulated",
        "gross_pnl_per_lot_rub": 0.0,
        "net_pnl_per_lot_rub": 0.0,
        "realized_r_net": 0.0,
    }


def _confidence(sample_size: int) -> str:
    if sample_size >= 8:
        return "high"
    if sample_size >= 4:
        return "medium"
    return "low"


def _write_training_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_markdown(payload: dict[str, object]) -> str:
    actual = payload["actual_day"]
    scan = payload["signal_scan"]
    opened = actual["opened_summary"]
    grid_rows = payload["grid_summary"].get("grid_rows", [])[:5]
    lines = [
        "# Daily Review",
        "",
        f"- Period: {payload['period']['start_at']} .. {payload['period']['end_at']} ({payload['period']['timezone']})",
        f"- Opened trades: {actual['opened_trades']}",
        f"- Net PnL opened: {opened['net_pnl_rub']} RUB",
        f"- Candidate signals: {scan['candidate_signals']}",
        f"- Tradable candidates: {scan['tradable_candidates']}",
        f"- Missed positive opportunities: {scan['missed_positive_opportunities']}",
        f"- Weak tradable candidates: {scan['weak_tradable_candidates']}",
        "",
        "## Best Exit Grid Rows",
    ]
    if grid_rows:
        for row in grid_rows:
            lines.append(
                "- offset={entry_offset_bars}, stop={stop_multiple}, rr={reward_to_risk}: "
                "{avg_net_pnl_per_lot_rub} RUB/lot, win {win_rate_pct}%, n={samples}".format(**row)
            )
    else:
        lines.append("- No signal grid rows")

    lines.append("")
    lines.append("## Recommendations")
    for item in payload.get("recommendations", []):
        lines.append(f"- {item.get('action')}: {item.get('reason')}")

    lines.append("")
    lines.append("## Top Missed Opportunities")
    missed = payload.get("missed_opportunities", [])[:10]
    if missed:
        for row in missed:
            best = row["best_plan"]
            lines.append(
                f"- {row['symbol']} {row['direction']} {row['local_time']}: "
                f"{best['net_pnl_per_lot_rub']} RUB/lot, exit={best['exit_reason']}"
            )
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)
