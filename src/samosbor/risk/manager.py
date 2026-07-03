from __future__ import annotations

from statistics import mean

from ..config import RiskSection, StrategySection
from ..domain import (
    InstrumentType,
    PortfolioState,
    Position,
    RiskDecision,
    Signal,
    SignalDirection,
    TradeRecord,
)


class RiskManager:
    def __init__(self, config: RiskSection):
        self.config = config

    def update_drawdown_state(self, portfolio: PortfolioState, marks: dict[str, float]) -> float:
        equity = portfolio.equity(marks)
        if portfolio.peak_equity <= 0:
            portfolio.peak_equity = equity
        else:
            portfolio.peak_equity = max(portfolio.peak_equity, equity)
        if portfolio.peak_equity > 0:
            drawdown = 1 - (equity / portfolio.peak_equity)
            if drawdown >= self.config.max_drawdown:
                portfolio.trading_halted = True
        return equity

    def approve(
        self,
        portfolio: PortfolioState,
        signal: Signal,
        marks: dict[str, float],
        recent_trades: list[TradeRecord],
    ) -> RiskDecision:
        equity = self.update_drawdown_state(portfolio, marks)
        if portfolio.trading_halted:
            return RiskDecision(False, "trading halted by drawdown control")

        if signal.instrument.symbol in portfolio.positions:
            return RiskDecision(False, "position already open")

        if len(portfolio.positions) >= self.config.max_positions:
            return RiskDecision(False, "max positions reached")

        if equity <= 0:
            return RiskDecision(False, "equity depleted")

        remaining_slots = max(1, self.config.max_positions - len(portfolio.positions))

        stop_distance = abs(signal.entry_price - signal.stop_price)
        if stop_distance <= 0:
            return RiskDecision(False, "invalid stop distance")

        risk_fraction = self._dynamic_risk_fraction(recent_trades)
        risk_budget = equity * risk_fraction
        risk_per_lot = stop_distance * signal.instrument.lot_size
        quantity_lots = int(risk_budget // risk_per_lot)
        if quantity_lots < 1:
            return RiskDecision(False, "risk budget too small")

        quantity_lots = self._cap_single_position_exposure(quantity_lots, signal, equity)
        if quantity_lots < 1:
            return RiskDecision(False, "single position exposure cap reached")

        current_exposure = portfolio.gross_exposure(marks)
        notional_per_lot = signal.entry_price * signal.instrument.lot_size

        if (
            signal.instrument.instrument_type == InstrumentType.FUTURE
            and (
                signal.instrument.initial_margin_buy > 0
                or signal.instrument.initial_margin_sell > 0
            )
        ):
            margin_per_lot = (
                signal.instrument.initial_margin_buy
                if signal.direction == SignalDirection.LONG
                else signal.instrument.initial_margin_sell
            )
            if margin_per_lot <= 0:
                return RiskDecision(False, "futures margin data unavailable")

            current_margin = portfolio.margin_reserved()
            max_margin_rub = equity * self.config.max_gross_exposure
            max_extra_margin = max(0.0, max_margin_rub - current_margin)
            margin_capped = int(max_extra_margin // margin_per_lot)
            quantity_lots = min(quantity_lots, margin_capped)
            if quantity_lots < 1:
                return RiskDecision(False, "futures margin cap reached")

            reserved_cash = equity * self.config.cash_reserve_ratio
            free_cash = max(0.0, portfolio.cash - current_margin - reserved_cash)
            cash_capped = int(free_cash // margin_per_lot)
            quantity_lots = min(quantity_lots, cash_capped)
            if quantity_lots < 1:
                return RiskDecision(False, "cash reserve rule blocked entry")

            estimated_notional = quantity_lots * signal.entry_price * signal.instrument.lot_size
            return RiskDecision(
                True,
                "approved",
                quantity_lots=quantity_lots,
                risk_budget_rub=risk_budget,
                estimated_notional_rub=estimated_notional,
            )

        max_exposure_rub = equity * self.config.max_gross_exposure
        max_extra_notional = max(0.0, max_exposure_rub - current_exposure)
        exposure_capped = int(max_extra_notional // notional_per_lot)
        quantity_lots = min(quantity_lots, exposure_capped)
        if quantity_lots < 1:
            return RiskDecision(False, "gross exposure cap reached")

        deployable_stock_budget = max(0.0, equity * (1 - self.config.cash_reserve_ratio))
        quantity_lots = self._cap_remaining_slot_budget(
            quantity_lots,
            current_allocated_rub=0.0,
            total_budget_rub=deployable_stock_budget,
            per_lot_rub=notional_per_lot,
            remaining_slots=max(1, self.config.max_positions),
        )
        if quantity_lots < 1:
            return RiskDecision(False, "stock slot budget reached")

        if signal.direction == SignalDirection.LONG:
            reserved_cash = equity * self.config.cash_reserve_ratio
            free_cash = max(0.0, portfolio.cash - reserved_cash)
            cash_capped = int(free_cash // notional_per_lot)
            quantity_lots = min(quantity_lots, cash_capped)
            if quantity_lots < 1:
                return RiskDecision(False, "cash reserve rule blocked entry")

        estimated_notional = quantity_lots * notional_per_lot
        return RiskDecision(
            True,
            "approved",
            quantity_lots=quantity_lots,
            risk_budget_rub=risk_budget,
            estimated_notional_rub=estimated_notional,
        )

    def _cap_single_position_exposure(
        self,
        quantity_lots: int,
        signal: Signal,
        equity: float,
    ) -> int:
        if quantity_lots < 1:
            return 0

        position_cap_ratio = min(
            self.config.max_gross_exposure,
            max(0.0, self.config.max_position_exposure_ratio),
        )
        if position_cap_ratio <= 0:
            return 0

        notional_per_lot = signal.entry_price * signal.instrument.lot_size
        if notional_per_lot <= 0:
            return 0

        max_position_notional = equity * position_cap_ratio
        position_capped = int(max_position_notional // notional_per_lot)
        return min(quantity_lots, position_capped)

    def _cap_remaining_slot_budget(
        self,
        quantity_lots: int,
        *,
        current_allocated_rub: float,
        total_budget_rub: float,
        per_lot_rub: float,
        remaining_slots: int,
    ) -> int:
        if quantity_lots < 1 or per_lot_rub <= 0 or remaining_slots <= 0:
            return 0

        remaining_budget = max(0.0, total_budget_rub - current_allocated_rub)
        slot_budget = remaining_budget / remaining_slots
        slot_capped = int(slot_budget // per_lot_rub)
        return min(quantity_lots, slot_capped)

    def trailing_stop_price(
        self,
        position: Position,
        mark_price: float,
        strategy: StrategySection,
    ) -> float | None:
        breakeven_trigger_pct = max(0.0, strategy.breakeven_trigger_pct)
        trigger_rub = max(0.0, strategy.trailing_profit_trigger_rub)
        lock_ratio = max(0.0, min(1.0, strategy.trailing_profit_lock_ratio))
        if position.quantity_units <= 0:
            return None

        open_profit = position.unrealized_pnl(mark_price)
        candidates: list[float] = []
        if breakeven_trigger_pct > 0:
            trigger_profit_rub = (
                position.entry_price
                * position.quantity_units
                * breakeven_trigger_pct
                / 100.0
            )
            if open_profit >= trigger_profit_rub:
                trailing_profit_rub = max(0.0, open_profit - trigger_profit_rub)
                protected_move = (
                    trailing_profit_rub * lock_ratio / position.quantity_units
                    if lock_ratio > 0
                    else 0.0
                )
                if position.direction == SignalDirection.LONG:
                    candidate = position.entry_price + protected_move
                    if candidate > position.stop_price:
                        candidates.append(candidate)
                else:
                    candidate = position.entry_price - protected_move
                    if candidate < position.stop_price:
                        candidates.append(candidate)

        if trigger_rub > 0 and lock_ratio > 0 and open_profit >= trigger_rub:
            locked_profit = open_profit * lock_ratio
            protected_move = locked_profit / position.quantity_units
            if position.direction == SignalDirection.LONG:
                candidate = position.entry_price + protected_move
                if candidate > position.stop_price:
                    candidates.append(candidate)
            else:
                candidate = position.entry_price - protected_move
                if candidate < position.stop_price:
                    candidates.append(candidate)

        if not candidates:
            return None
        if position.direction == SignalDirection.LONG:
            return max(candidates)
        return min(candidates)

    def _dynamic_risk_fraction(self, recent_trades: list[TradeRecord]) -> float:
        if len(recent_trades) < self.config.min_trades_for_kelly:
            return self.config.max_risk_per_trade

        sample = recent_trades[-self.config.kelly_lookback_trades :]
        wins = [trade.net_pnl for trade in sample if trade.net_pnl > 0]
        losses = [-trade.net_pnl for trade in sample if trade.net_pnl < 0]
        if not wins or not losses:
            return self.config.max_risk_per_trade * 0.5

        win_rate = len(wins) / len(sample)
        avg_win = mean(wins)
        avg_loss = mean(losses)
        if avg_loss <= 0:
            return self.config.max_risk_per_trade * 0.5

        edge_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / edge_ratio)
        scale = max(0.25, min(1.0, 0.5 + kelly / 2))
        return self.config.max_risk_per_trade * scale
