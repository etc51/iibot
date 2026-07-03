from __future__ import annotations

import math
from datetime import datetime

from ..config import BacktestSection
from ..domain import BacktestResult, Candle, EquityPoint, ExitReason, SignalDirection
from ..execution.paper import LocalPaperBroker
from ..risk.manager import RiskManager
from ..strategy.trend_following import TrendFollowingStrategy


class BacktestEngine:
    def __init__(
        self,
        *,
        strategy: TrendFollowingStrategy,
        risk_manager: RiskManager,
        backtest: BacktestSection,
        slippage_bps: float,
        commission_bps: float,
    ):
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.backtest = backtest
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps

    def run_with_instruments(
        self,
        candles_by_symbol,
        instruments_by_symbol,
        *,
        trade_start_at=None,
    ):
        if hasattr(self.strategy, "prepare_market_context"):
            self.strategy.prepare_market_context(candles_by_symbol)
        for symbol, candles in candles_by_symbol.items():
            self.strategy.prepare_history(instruments_by_symbol[symbol], candles)

        broker = LocalPaperBroker.fresh(
            self.backtest.initial_cash,
            slippage_bps=self.slippage_bps,
            commission_bps=self.commission_bps,
        )
        events: list[dict[str, object]] = []
        history_by_symbol = {symbol: [] for symbol in candles_by_symbol}
        latest_marks: dict[str, float] = {}
        timeline = sorted(
            [
                (candle.timestamp, symbol, candle)
                for symbol, candles in candles_by_symbol.items()
                for candle in candles
            ],
            key=lambda item: item[0],
        )
        equity_curve: list[EquityPoint] = []

        for timestamp, symbol, candle in timeline:
            history = history_by_symbol[symbol]
            history.append(candle)
            latest_marks[symbol] = candle.close

            can_trade = trade_start_at is None or timestamp >= trade_start_at

            if can_trade:
                position = broker.portfolio.positions.get(symbol)
                if position is not None:
                    exit_price, reason = self._check_exit(
                        position.direction,
                        position.entry_price,
                        position.stop_price,
                        position.take_profit,
                        candle,
                    )
                    if reason is not None:
                        broker.close_position(
                            symbol,
                            price=exit_price,
                            timestamp=timestamp,
                            reason=reason,
                        )
                    elif self.strategy.should_force_flatten_at(timestamp):
                        broker.close_position(
                            symbol,
                            price=candle.close,
                            timestamp=timestamp,
                            reason=ExitReason.SESSION_FLAT,
                        )

                position = broker.portfolio.positions.get(symbol)
                if position is not None:
                    self._apply_trailing_stop(
                        broker=broker,
                        symbol=symbol,
                        position=position,
                        mark_price=candle.close,
                        timestamp=timestamp,
                    )

                if len(history) >= self.backtest.warmup_bars and not broker.portfolio.trading_halted:
                    signal = self.strategy.generate_signal(instruments_by_symbol[symbol], history)
                    if signal is not None:
                        if position and position.direction != signal.direction:
                            broker.close_position(
                                symbol,
                                price=candle.close,
                                timestamp=timestamp,
                                reason=ExitReason.SIGNAL_FLIP,
                            )
                            position = None
                        if position is None:
                            entry_block_reason = self.strategy.entry_block_reason_for_instrument(
                                instruments_by_symbol[symbol],
                                timestamp,
                                signal.direction,
                            )
                            if entry_block_reason is not None:
                                events.append(
                                    {
                                        "timestamp": timestamp.isoformat(),
                                        "symbol": symbol,
                                        "action": "signal",
                                        "approved": False,
                                        "reason": entry_block_reason,
                                        "direction": signal.direction.value,
                                        "strength": signal.strength,
                                        "quantity_lots": 0,
                                    }
                                )
                            else:
                                decision = self.risk_manager.approve(
                                    broker.portfolio,
                                    signal,
                                    {**latest_marks, symbol: candle.close},
                                    broker.trades,
                                )
                                events.append(
                                    {
                                        "timestamp": timestamp.isoformat(),
                                        "symbol": symbol,
                                        "action": "signal",
                                        "approved": decision.approved,
                                        "reason": decision.reason,
                                        "direction": signal.direction.value,
                                        "strength": signal.strength,
                                        "quantity_lots": decision.quantity_lots,
                                    }
                                )
                                if decision.approved:
                                    broker.open_position(signal, decision.quantity_lots, timestamp)

            equity = broker.mark_to_market(latest_marks, timestamp)
            equity_curve.append(
                EquityPoint(
                    timestamp=timestamp,
                    equity=equity,
                    cash=broker.portfolio.cash,
                    gross_exposure=broker.portfolio.gross_exposure(latest_marks),
                )
            )

        final_timestamp = timeline[-1][0] if timeline else datetime.utcnow()
        for symbol in list(broker.portfolio.positions):
            broker.close_position(
                symbol,
                price=latest_marks[symbol],
                timestamp=final_timestamp,
                reason=ExitReason.END_OF_TEST,
            )

        equity = broker.mark_to_market(latest_marks, final_timestamp)
        equity_curve.append(
            EquityPoint(
                timestamp=final_timestamp,
                equity=equity,
                cash=broker.portfolio.cash,
                gross_exposure=broker.portfolio.gross_exposure(latest_marks),
            )
        )
        return BacktestResult(
            portfolio=broker.portfolio,
            trades=broker.trades,
            equity_curve=equity_curve,
            events=events + broker.events,
        )

    def _apply_trailing_stop(
        self,
        *,
        broker: LocalPaperBroker,
        symbol: str,
        position,
        mark_price: float,
        timestamp,
    ) -> None:
        strategy_config = getattr(self.strategy, "config", None)
        if strategy_config is None:
            return

        new_stop = self.risk_manager.trailing_stop_price(position, mark_price, strategy_config)
        if new_stop is None:
            return

        broker.update_position_protection(
            symbol,
            timestamp=timestamp,
            stop_price=new_stop,
            reason="trailing-profit-protection",
        )

    @staticmethod
    def _check_exit(
        direction: SignalDirection,
        entry_price: float,
        stop_price: float,
        take_profit: float,
        candle: Candle,
    ):
        if direction == SignalDirection.LONG:
            if candle.low <= stop_price:
                return stop_price, BacktestEngine._stop_exit_reason(
                    direction,
                    entry_price,
                    stop_price,
                )
            if candle.high >= take_profit:
                return take_profit, ExitReason.TAKE_PROFIT
        else:
            if candle.high >= stop_price:
                return stop_price, BacktestEngine._stop_exit_reason(
                    direction,
                    entry_price,
                    stop_price,
                )
            if candle.low <= take_profit:
                return take_profit, ExitReason.TAKE_PROFIT
        return None, None

    @staticmethod
    def _stop_exit_reason(
        direction: SignalDirection,
        entry_price: float,
        stop_price: float,
    ) -> ExitReason:
        if math.isclose(stop_price, entry_price, rel_tol=1e-9, abs_tol=1e-9):
            return ExitReason.BREAKEVEN_STOP
        if direction == SignalDirection.LONG:
            if stop_price > entry_price:
                return ExitReason.PROFIT_PROTECT_STOP
            return ExitReason.STOP_LOSS
        if stop_price < entry_price:
            return ExitReason.PROFIT_PROTECT_STOP
        return ExitReason.STOP_LOSS
