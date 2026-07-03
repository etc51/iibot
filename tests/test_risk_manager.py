from __future__ import annotations

import unittest
from datetime import datetime, timezone

from samosbor.config import RiskSection, StrategySection
from samosbor.domain import (
    Instrument,
    InstrumentType,
    PortfolioState,
    Position,
    Signal,
    SignalDirection,
)
from samosbor.risk.manager import RiskManager


def _stock_signal(symbol: str, direction: SignalDirection = SignalDirection.LONG) -> Signal:
    return Signal(
        instrument=Instrument(symbol=symbol, instrument_type=InstrumentType.STOCK, lot_size=1),
        direction=direction,
        strength=0.8,
        entry_price=100.0,
        stop_price=99.0,
        take_profit=102.0,
        reason="test",
    )


class RiskManagerDiversificationTest(unittest.TestCase):
    def test_stock_position_is_capped_by_remaining_slot_budget(self):
        manager = RiskManager(
            RiskSection(
                max_risk_per_trade=0.02,
                max_gross_exposure=1.5,
                cash_reserve_ratio=0.0,
                max_positions=4,
                max_position_exposure_ratio=0.30,
            )
        )
        portfolio = PortfolioState(cash=300_000.0, peak_equity=300_000.0)

        decision = manager.approve(portfolio, _stock_signal("SBER"), {"SBER": 100.0}, [])

        self.assertTrue(decision.approved)
        self.assertEqual(decision.quantity_lots, 750)
        self.assertEqual(decision.estimated_notional_rub, 75_000.0)

    def test_position_cap_still_leaves_room_for_other_stocks(self):
        manager = RiskManager(
            RiskSection(
                max_risk_per_trade=0.02,
                max_gross_exposure=1.5,
                cash_reserve_ratio=0.10,
                max_positions=4,
                max_position_exposure_ratio=0.30,
            )
        )
        existing_instrument = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=1)
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        portfolio = PortfolioState(
            cash=210_000.0,
            peak_equity=300_000.0,
            positions={
                "SBER": Position(
                    instrument=existing_instrument,
                    direction=SignalDirection.LONG,
                    quantity_lots=900,
                    entry_price=100.0,
                    entry_commission=0.0,
                    margin_requirement=0.0,
                    current_price=100.0,
                    stop_price=95.0,
                    take_profit=110.0,
                    opened_at=now,
                    updated_at=now,
                )
            },
        )

        decision = manager.approve(portfolio, _stock_signal("GAZP"), {"SBER": 100.0, "GAZP": 100.0}, [])

        self.assertTrue(decision.approved)
        self.assertEqual(decision.quantity_lots, 675)
        self.assertEqual(decision.estimated_notional_rub, 67_500.0)

    def test_short_stock_position_uses_same_slot_budget(self):
        manager = RiskManager(
            RiskSection(
                max_risk_per_trade=0.02,
                max_gross_exposure=1.5,
                cash_reserve_ratio=0.10,
                max_positions=4,
                max_position_exposure_ratio=0.30,
            )
        )
        portfolio = PortfolioState(cash=300_000.0, peak_equity=300_000.0)

        decision = manager.approve(
            portfolio,
            _stock_signal("TATN", direction=SignalDirection.SHORT),
            {"TATN": 100.0},
            [],
        )

        self.assertTrue(decision.approved)
        self.assertEqual(decision.quantity_lots, 675)
        self.assertEqual(decision.estimated_notional_rub, 67_500.0)

    def test_futures_position_is_capped_by_single_position_exposure_ratio(self):
        manager = RiskManager(
            RiskSection(
                max_risk_per_trade=0.02,
                max_gross_exposure=2.0,
                cash_reserve_ratio=0.0,
                max_positions=4,
                max_position_exposure_ratio=0.30,
            )
        )
        portfolio = PortfolioState(cash=300_000.0, peak_equity=300_000.0)
        instrument = Instrument(
            symbol="RI",
            instrument_type=InstrumentType.FUTURE,
            lot_size=1,
            initial_margin_buy=500.0,
            initial_margin_sell=500.0,
        )
        signal = Signal(
            instrument=instrument,
            direction=SignalDirection.LONG,
            strength=0.7,
            entry_price=1_000.0,
            stop_price=990.0,
            take_profit=1_020.0,
            reason="test",
        )

        decision = manager.approve(portfolio, signal, {"RI": 1_000.0}, [])

        self.assertTrue(decision.approved)
        self.assertEqual(decision.quantity_lots, 90)
        self.assertEqual(decision.estimated_notional_rub, 90_000.0)

    def test_trailing_stop_moves_to_breakeven_after_half_percent_profit(self):
        manager = RiskManager(RiskSection())
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        position = Position(
            instrument=Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=1),
            direction=SignalDirection.LONG,
            quantity_lots=10,
            entry_price=100.0,
            entry_commission=0.0,
            margin_requirement=0.0,
            current_price=100.5,
            stop_price=95.0,
            take_profit=110.0,
            opened_at=now,
            updated_at=now,
        )

        candidate = manager.trailing_stop_price(
            position,
            100.5,
            StrategySection(breakeven_trigger_pct=0.5, trailing_profit_lock_ratio=0.5),
        )

        self.assertEqual(candidate, 100.0)

    def test_trailing_stop_locks_only_excess_profit_above_breakeven_trigger(self):
        manager = RiskManager(RiskSection())
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        position = Position(
            instrument=Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=1),
            direction=SignalDirection.LONG,
            quantity_lots=10,
            entry_price=100.0,
            entry_commission=0.0,
            margin_requirement=0.0,
            current_price=101.0,
            stop_price=95.0,
            take_profit=110.0,
            opened_at=now,
            updated_at=now,
        )

        candidate = manager.trailing_stop_price(
            position,
            101.0,
            StrategySection(breakeven_trigger_pct=0.5, trailing_profit_lock_ratio=0.5),
        )

        self.assertEqual(candidate, 100.25)

    def test_trailing_stop_uses_tighter_ruble_lock_when_both_triggers_are_enabled(self):
        manager = RiskManager(RiskSection())
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        position = Position(
            instrument=Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=1),
            direction=SignalDirection.LONG,
            quantity_lots=10,
            entry_price=100.0,
            entry_commission=0.0,
            margin_requirement=0.0,
            current_price=102.0,
            stop_price=95.0,
            take_profit=110.0,
            opened_at=now,
            updated_at=now,
        )

        candidate = manager.trailing_stop_price(
            position,
            102.0,
            StrategySection(
                breakeven_trigger_pct=0.5,
                trailing_profit_trigger_rub=10.0,
                trailing_profit_lock_ratio=0.5,
            ),
        )

        self.assertEqual(candidate, 101.0)

    def test_runner_trailing_stop_uses_atr_and_profit_lock_after_take_profit_activation(self):
        manager = RiskManager(RiskSection())
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        position = Position(
            instrument=Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=1),
            direction=SignalDirection.LONG,
            quantity_lots=10,
            entry_price=100.0,
            entry_commission=0.0,
            margin_requirement=0.0,
            current_price=106.0,
            stop_price=100.1,
            take_profit=104.0,
            opened_at=now,
            updated_at=now,
            runner_active=True,
            runner_activation_price=104.0,
            runner_extreme_price=106.0,
        )

        candidate = manager.runner_trailing_stop_price(
            position,
            atr_value=2.0,
            strategy=StrategySection(
                runner_breakeven_buffer_bps=10.0,
                runner_trailing_atr_multiple=1.3,
                runner_profit_lock_ratio=0.35,
            ),
        )

        self.assertAlmostEqual(candidate, 103.4)


if __name__ == "__main__":
    unittest.main()
