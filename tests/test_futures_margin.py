from __future__ import annotations

import unittest
from datetime import datetime, timezone

from samosbor.config import RiskSection
from samosbor.domain import ExitReason, Instrument, InstrumentType, PortfolioState, Signal, SignalDirection
from samosbor.execution.paper import LocalPaperBroker
from samosbor.risk.manager import RiskManager


class FuturesMarginTest(unittest.TestCase):
    def test_future_round_trip_uses_margin_style_cash_flow(self):
        broker = LocalPaperBroker.fresh(100_000, slippage_bps=0, commission_bps=0)
        instrument = Instrument(
            symbol="CNYRUBF",
            instrument_type=InstrumentType.FUTURE,
            lot_size=1,
            initial_margin_buy=1_000.0,
            initial_margin_sell=1_100.0,
        )
        signal = Signal(
            instrument=instrument,
            direction=SignalDirection.LONG,
            strength=0.8,
            entry_price=100.0,
            stop_price=95.0,
            take_profit=110.0,
            reason="test",
        )

        position = broker.open_position(signal, 2, datetime(2025, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(position.margin_requirement, 2_000.0)
        self.assertEqual(broker.portfolio.cash, 100_000.0)
        self.assertEqual(broker.portfolio.equity({"CNYRUBF": 100.0}), 100_000.0)

        trade = broker.close_position(
            "CNYRUBF",
            price=110.0,
            timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc),
            reason=ExitReason.TAKE_PROFIT,
        )
        self.assertIsNotNone(trade)
        self.assertEqual(round(broker.portfolio.cash, 2), 100_020.0)

    def test_future_risk_uses_margin_cap(self):
        manager = RiskManager(
            RiskSection(
                max_risk_per_trade=0.02,
                max_gross_exposure=1.0,
                cash_reserve_ratio=0.0,
            )
        )
        portfolio = PortfolioState(cash=10_000.0, peak_equity=10_000.0)
        instrument = Instrument(
            symbol="CNYRUBF",
            instrument_type=InstrumentType.FUTURE,
            lot_size=1,
            initial_margin_buy=2_000.0,
            initial_margin_sell=2_000.0,
        )
        signal = Signal(
            instrument=instrument,
            direction=SignalDirection.LONG,
            strength=0.5,
            entry_price=100.0,
            stop_price=99.0,
            take_profit=102.0,
            reason="test",
        )

        decision = manager.approve(portfolio, signal, {"CNYRUBF": 100.0}, [])

        self.assertTrue(decision.approved)
        self.assertEqual(decision.quantity_lots, 5)


if __name__ == "__main__":
    unittest.main()
