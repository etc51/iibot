from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from ..autonomy.ml_learning import build_post_close_learning_analysis
from ..domain import ExitReason, InstrumentType, PortfolioState, Position, Signal, SignalDirection, TradeRecord


class LocalPaperBroker:
    def __init__(
        self,
        portfolio: PortfolioState,
        *,
        slippage_bps: float,
        commission_bps: float,
        trades: list[TradeRecord] | None = None,
        events: list[dict[str, Any]] | None = None,
    ):
        self.portfolio = portfolio
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.trades = trades or []
        self.events = events or []

    @classmethod
    def fresh(
        cls,
        initial_cash: float,
        *,
        slippage_bps: float,
        commission_bps: float,
    ) -> "LocalPaperBroker":
        return cls(
            PortfolioState(cash=initial_cash, peak_equity=initial_cash),
            slippage_bps=slippage_bps,
            commission_bps=commission_bps,
        )

    @classmethod
    def load(
        cls,
        state_path: Path,
        *,
        initial_cash: float,
        slippage_bps: float,
        commission_bps: float,
    ) -> "LocalPaperBroker":
        if not state_path.exists():
            return cls.fresh(
                initial_cash,
                slippage_bps=slippage_bps,
                commission_bps=commission_bps,
            )
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        trades = [
            TradeRecord(
                symbol=item["symbol"],
                direction=SignalDirection(item["direction"]),
                quantity_lots=int(item["quantity_lots"]),
                entry_time=datetime.fromisoformat(item["entry_time"]),
                exit_time=datetime.fromisoformat(item["exit_time"]),
                entry_price=float(item["entry_price"]),
                exit_price=float(item["exit_price"]),
                gross_pnl=float(item["gross_pnl"]),
                net_pnl=float(item["net_pnl"]),
                reason=_normalized_loaded_trade_reason(
                    str(item["reason"]),
                    net_pnl=float(item["net_pnl"]),
                ),
                signal_strength=float(item.get("signal_strength", 0.0)),
                entry_reason=str(item.get("entry_reason", "")),
                entry_context_score=float(item.get("entry_context_score", 0.0)),
                entry_metadata=dict(item.get("entry_metadata", {})),
                initial_stop_price=float(item.get("initial_stop_price", item.get("entry_price", 0.0))),
                initial_take_profit=float(item.get("initial_take_profit", item.get("exit_price", 0.0))),
            )
            for item in payload.get("trades", [])
        ]
        return cls(
            portfolio=PortfolioState.from_dict(payload["portfolio"]),
            slippage_bps=slippage_bps,
            commission_bps=commission_bps,
            trades=trades,
            events=payload.get("events", []),
        )

    def save(self, state_path: Path) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "portfolio": self.portfolio.to_dict(),
            "trades": [
                {
                    "symbol": trade.symbol,
                    "direction": trade.direction.value,
                    "quantity_lots": trade.quantity_lots,
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat(),
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "gross_pnl": trade.gross_pnl,
                    "net_pnl": trade.net_pnl,
                    "reason": trade.reason,
                    "signal_strength": trade.signal_strength,
                    "entry_reason": trade.entry_reason,
                    "entry_context_score": trade.entry_context_score,
                    "entry_metadata": trade.entry_metadata,
                    "initial_stop_price": trade.initial_stop_price,
                    "initial_take_profit": trade.initial_take_profit,
                }
                for trade in self.trades
            ],
            "events": self.events,
        }
        state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def mark_to_market(self, marks: dict[str, float], timestamp: datetime) -> float:
        for symbol, position in self.portfolio.positions.items():
            if symbol in marks:
                position.current_price = marks[symbol]
                position.updated_at = timestamp
        equity = self.portfolio.equity(marks)
        self.portfolio.peak_equity = max(self.portfolio.peak_equity, equity)
        return equity

    def open_position(self, signal: Signal, quantity_lots: int, timestamp: datetime) -> Position | None:
        if quantity_lots < 1:
            return None

        is_buy = signal.direction == SignalDirection.LONG
        fill_price = self._slipped_price(signal.entry_price, is_buy=is_buy)
        units = quantity_lots * signal.instrument.lot_size
        notional = fill_price * units
        commission = self._commission(notional)
        margin_requirement = 0.0

        if signal.instrument.instrument_type == InstrumentType.FUTURE:
            if signal.direction == SignalDirection.LONG:
                margin_requirement = signal.instrument.initial_margin_buy * quantity_lots
            else:
                margin_requirement = signal.instrument.initial_margin_sell * quantity_lots
            self.portfolio.cash -= commission
        else:
            if signal.direction == SignalDirection.LONG:
                self.portfolio.cash -= notional + commission
            else:
                self.portfolio.cash += notional - commission

        position = Position(
            instrument=signal.instrument,
            direction=signal.direction,
            quantity_lots=quantity_lots,
            entry_price=fill_price,
            entry_commission=commission,
            margin_requirement=margin_requirement,
            current_price=fill_price,
            stop_price=signal.stop_price,
            take_profit=signal.take_profit,
            opened_at=timestamp,
            updated_at=timestamp,
            signal_strength=signal.strength,
            entry_reason=signal.reason,
            entry_context_score=signal.context_score,
            entry_metadata=dict(signal.metadata),
            initial_stop_price=signal.stop_price,
            initial_take_profit=signal.take_profit,
        )
        self.portfolio.positions[signal.instrument.symbol] = position
        self.events.append(
            {
                "timestamp": timestamp.isoformat(),
                "symbol": signal.instrument.symbol,
                "action": "open",
                "direction": signal.direction.value,
                "quantity_lots": quantity_lots,
                "fill_price": fill_price,
                "commission": commission,
                "margin_requirement": margin_requirement,
                "reason": signal.reason,
                "signal_strength": signal.strength,
                "context_score": signal.context_score,
                "metadata": dict(signal.metadata),
                "initial_stop_price": signal.stop_price,
                "initial_take_profit": signal.take_profit,
            }
        )
        return position

    def close_position(
        self,
        symbol: str,
        *,
        price: float,
        timestamp: datetime,
        reason: ExitReason,
    ) -> TradeRecord | None:
        position = self.portfolio.positions.get(symbol)
        if position is None:
            return None

        is_buy = position.direction == SignalDirection.SHORT
        fill_price = self._slipped_price(price, is_buy=is_buy)
        exit_reason = self._normalized_exit_reason(reason, position, trigger_price=price)
        units = position.quantity_units
        notional = fill_price * units
        commission = self._commission(notional)

        if position.instrument.instrument_type == InstrumentType.FUTURE:
            if position.direction == SignalDirection.LONG:
                gross_pnl = (fill_price - position.entry_price) * units
            else:
                gross_pnl = (position.entry_price - fill_price) * units
            self.portfolio.cash += gross_pnl - commission
        else:
            if position.direction == SignalDirection.LONG:
                self.portfolio.cash += notional - commission
                gross_pnl = (fill_price - position.entry_price) * units
            else:
                self.portfolio.cash -= notional + commission
                gross_pnl = (position.entry_price - fill_price) * units

        net_pnl = gross_pnl - position.entry_commission - commission
        self.portfolio.realized_pnl += net_pnl

        trade = TradeRecord(
            symbol=symbol,
            direction=position.direction,
            quantity_lots=position.quantity_lots,
            entry_time=position.opened_at,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=fill_price,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            reason=exit_reason.value,
            signal_strength=position.signal_strength,
            entry_reason=position.entry_reason,
            entry_context_score=position.entry_context_score,
            entry_metadata=dict(position.entry_metadata),
            initial_stop_price=position.initial_stop_price or position.stop_price,
            initial_take_profit=position.initial_take_profit or position.take_profit,
        )
        post_close_analysis = build_post_close_learning_analysis(trade)
        trade = replace(
            trade,
            entry_metadata={
                **trade.entry_metadata,
                "post_close_analysis": post_close_analysis,
            },
        )
        self.trades.append(trade)
        self.events.append(
            {
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "action": "close",
                "direction": position.direction.value,
                "quantity_lots": position.quantity_lots,
                "fill_price": fill_price,
                "commission": commission,
                "reason": exit_reason.value,
                "net_pnl": net_pnl,
                "gross_pnl": gross_pnl,
                "entry_time": position.opened_at.isoformat(),
                "entry_price": position.entry_price,
                "signal_strength": position.signal_strength,
                "post_close_analysis": post_close_analysis,
            }
        )
        del self.portfolio.positions[symbol]
        return trade

    def update_position_protection(
        self,
        symbol: str,
        *,
        timestamp: datetime,
        stop_price: float | None = None,
        take_profit: float | None = None,
        reason: str = "manual-protection-update",
    ) -> bool:
        position = self.portfolio.positions.get(symbol)
        if position is None:
            return False

        changed = False
        if stop_price is not None and stop_price != position.stop_price:
            position.stop_price = stop_price
            changed = True
        if take_profit is not None and take_profit != position.take_profit:
            position.take_profit = take_profit
            changed = True
        if not changed:
            return False

        position.updated_at = timestamp
        self.events.append(
            {
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "action": "protect",
                "direction": position.direction.value,
                "stop_price": position.stop_price,
                "take_profit": position.take_profit,
                "reason": reason,
            }
        )
        return True

    def activate_position_runner(
        self,
        symbol: str,
        *,
        timestamp: datetime,
        activation_price: float,
        stop_price: float,
        extreme_price: float,
        reason: str = "take-profit-runner-activation",
    ) -> bool:
        position = self.portfolio.positions.get(symbol)
        if position is None or position.runner_active:
            return False

        position.runner_active = True
        position.runner_activated_at = timestamp
        position.runner_activation_price = activation_price
        position.runner_extreme_price = extreme_price
        position.stop_price = stop_price
        position.updated_at = timestamp
        self.events.append(
            {
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "action": "runner-activate",
                "direction": position.direction.value,
                "activation_price": activation_price,
                "stop_price": stop_price,
                "take_profit": position.take_profit,
                "runner_extreme_price": position.runner_extreme_price,
                "reason": reason,
            }
        )
        return True

    def update_position_runner_extreme(
        self,
        symbol: str,
        *,
        timestamp: datetime,
        extreme_price: float,
    ) -> bool:
        position = self.portfolio.positions.get(symbol)
        if position is None or not position.runner_active:
            return False
        if extreme_price == position.runner_extreme_price:
            return False

        position.runner_extreme_price = extreme_price
        position.updated_at = timestamp
        return True

    def _slipped_price(self, price: float, *, is_buy: bool) -> float:
        factor = 1 + (self.slippage_bps / 10_000)
        return price * factor if is_buy else price / factor

    @staticmethod
    def _normalized_exit_reason(
        reason: ExitReason,
        position: Position,
        *,
        trigger_price: float,
    ) -> ExitReason:
        if reason != ExitReason.STOP_LOSS:
            return reason
        if math.isclose(trigger_price, position.entry_price, rel_tol=1e-9, abs_tol=1e-9):
            return ExitReason.BREAKEVEN_STOP
        if position.direction == SignalDirection.LONG:
            if trigger_price > position.entry_price:
                return ExitReason.PROFIT_PROTECT_STOP
            return ExitReason.STOP_LOSS
        if trigger_price < position.entry_price:
            return ExitReason.PROFIT_PROTECT_STOP
        return ExitReason.STOP_LOSS

    def _commission(self, notional: float) -> float:
        return abs(notional) * self.commission_bps / 10_000


def _normalized_loaded_trade_reason(reason: str, *, net_pnl: float) -> str:
    if reason == ExitReason.STOP_LOSS.value and net_pnl > 0:
        return ExitReason.PROFIT_PROTECT_STOP.value
    return reason
