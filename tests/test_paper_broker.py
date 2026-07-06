from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from samosbor.domain import ExitReason, Instrument, InstrumentType, Signal, SignalDirection
from samosbor.execution.paper import LocalPaperBroker


class PaperBrokerTest(unittest.TestCase):
    def test_load_drops_zero_lot_positions_from_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "portfolio": {
                            "cash": 100_000.0,
                            "realized_pnl": 0.0,
                            "peak_equity": 100_000.0,
                            "positions": {
                                "ROSN": {
                                    "instrument": {
                                        "symbol": "ROSN",
                                        "instrument_type": "stock",
                                        "lot_size": 10,
                                    },
                                    "direction": "short",
                                    "quantity_lots": 0,
                                    "entry_price": 301.6293,
                                    "entry_commission": 0.0,
                                    "margin_requirement": 0.0,
                                    "current_price": 299.15,
                                    "stop_price": 305.6446,
                                    "take_profit": 292.0134,
                                    "opened_at": "2026-06-15T09:00:00+00:00",
                                    "updated_at": "2026-06-15T09:30:00+00:00",
                                },
                                "SBER": {
                                    "instrument": {
                                        "symbol": "SBER",
                                        "instrument_type": "stock",
                                        "lot_size": 1,
                                    },
                                    "direction": "short",
                                    "quantity_lots": 8,
                                    "entry_price": 297.9208,
                                    "entry_commission": 0.0,
                                    "margin_requirement": 0.0,
                                    "current_price": 297.97,
                                    "stop_price": 300.5611,
                                    "take_profit": 291.7373,
                                    "opened_at": "2026-06-15T09:00:00+00:00",
                                    "updated_at": "2026-06-15T09:30:00+00:00",
                                },
                            },
                        },
                        "trades": [],
                        "events": [],
                    }
                ),
                encoding="utf-8",
            )

            broker = LocalPaperBroker.load(
                state_path,
                initial_cash=100_000.0,
                slippage_bps=0,
                commission_bps=0,
            )

            self.assertEqual(list(broker.portfolio.positions), ["SBER"])

    def test_load_normalizes_profitable_stop_loss_trade_reason(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "portfolio": {
                            "cash": 100_000.0,
                            "realized_pnl": 12.5,
                            "peak_equity": 100_000.0,
                            "positions": {},
                        },
                        "trades": [
                            {
                                "symbol": "TRNFP",
                                "direction": "short",
                                "quantity_lots": 1,
                                "entry_time": "2026-07-06T10:00:00+00:00",
                                "exit_time": "2026-07-06T11:00:00+00:00",
                                "entry_price": 100.0,
                                "exit_price": 98.0,
                                "gross_pnl": 2.0,
                                "net_pnl": 1.69,
                                "reason": "stop-loss",
                            }
                        ],
                        "events": [],
                    }
                ),
                encoding="utf-8",
            )

            broker = LocalPaperBroker.load(
                state_path,
                initial_cash=100_000.0,
                slippage_bps=0,
                commission_bps=0,
            )

            self.assertEqual(broker.trades[0].reason, ExitReason.PROFIT_PROTECT_STOP.value)

    def test_round_trip_trade_updates_cash_and_records_trade(self):
        broker = LocalPaperBroker.fresh(100_000, slippage_bps=0, commission_bps=0)
        instrument = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=10)
        signal = Signal(
            instrument=instrument,
            direction=SignalDirection.LONG,
            strength=0.8,
            entry_price=100.0,
            stop_price=95.0,
            take_profit=110.0,
            reason="test",
            context_score=0.2,
            metadata={"trend_strength": 0.03},
        )

        opened = broker.open_position(signal, 2, datetime(2025, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(opened.quantity_units, 20)
        trade = broker.close_position(
            "SBER",
            price=110.0,
            timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc),
            reason=ExitReason.TAKE_PROFIT,
        )

        self.assertIsNotNone(trade)
        self.assertEqual(len(broker.trades), 1)
        self.assertGreater(broker.portfolio.cash, 100_000)
        self.assertEqual(opened.signal_strength, 0.8)
        self.assertEqual(trade.signal_strength, 0.8)
        self.assertEqual(trade.entry_reason, "test")
        self.assertEqual(trade.entry_context_score, 0.2)
        self.assertEqual(trade.entry_metadata["trend_strength"], 0.03)
        self.assertEqual(trade.initial_stop_price, 95.0)
        self.assertEqual(trade.initial_take_profit, 110.0)
        self.assertEqual(trade.entry_metadata["post_close_analysis"]["outcome"], "profit")
        self.assertFalse(trade.entry_metadata["post_close_analysis"]["is_error"])
        self.assertEqual(broker.events[-1]["post_close_analysis"]["outcome"], "profit")

    def test_open_position_rejects_zero_lots(self):
        broker = LocalPaperBroker.fresh(100_000, slippage_bps=0, commission_bps=0)
        instrument = Instrument(symbol="PLZL", instrument_type=InstrumentType.STOCK, lot_size=1)
        signal = Signal(
            instrument=instrument,
            direction=SignalDirection.SHORT,
            strength=0.8,
            entry_price=1857.0,
            stop_price=1885.0,
            take_profit=1792.0,
            reason="test",
        )

        opened = broker.open_position(signal, 0, datetime(2025, 1, 1, tzinfo=timezone.utc))

        self.assertIsNone(opened)
        self.assertEqual(broker.portfolio.cash, 100_000)
        self.assertEqual(broker.portfolio.positions, {})
        self.assertEqual(broker.events, [])

    def test_close_position_compares_result_with_entry_ml_learning(self):
        broker = LocalPaperBroker.fresh(100_000, slippage_bps=0, commission_bps=0)
        instrument = Instrument(symbol="OZON", instrument_type=InstrumentType.STOCK, lot_size=1)
        signal = Signal(
            instrument=instrument,
            direction=SignalDirection.SHORT,
            strength=0.8,
            entry_price=100.0,
            stop_price=105.0,
            take_profit=90.0,
            reason="test",
            metadata={
                "ml_learning": {
                    "available": True,
                    "probability_profit": 0.32,
                    "expected_pnl_position_rub": -120.0,
                    "learning_tags": ["low-quality-learning"],
                }
            },
        )

        broker.open_position(signal, 10, datetime(2025, 1, 1, tzinfo=timezone.utc))
        trade = broker.close_position(
            "OZON",
            price=105.0,
            timestamp=datetime(2025, 1, 1, 1, 0, tzinfo=timezone.utc),
            reason=ExitReason.STOP_LOSS,
        )

        self.assertIsNotNone(trade)
        analysis = trade.entry_metadata["post_close_analysis"]
        self.assertEqual(analysis["outcome"], "error")
        self.assertTrue(analysis["is_error"])
        self.assertEqual(analysis["ml_entry_bias"], "risk")
        self.assertEqual(analysis["ml_verdict"], "ml_warned_loss")
        self.assertIn("full-risk-error", analysis["tags"])

    def test_four_bps_per_side_is_0_08_percent_round_trip_cost(self):
        broker = LocalPaperBroker.fresh(100_000, slippage_bps=0, commission_bps=4)
        instrument = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=10)
        signal = Signal(
            instrument=instrument,
            direction=SignalDirection.LONG,
            strength=0.8,
            entry_price=100.0,
            stop_price=95.0,
            take_profit=110.0,
            reason="commission-check",
        )

        opened = broker.open_position(signal, 1, datetime(2025, 1, 1, tzinfo=timezone.utc))
        trade = broker.close_position(
            "SBER",
            price=100.0,
            timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc),
            reason=ExitReason.END_OF_TEST,
        )

        self.assertIsNotNone(trade)
        self.assertAlmostEqual(opened.entry_commission, 0.4, places=6)
        self.assertAlmostEqual(broker.events[-1]["commission"], 0.4, places=6)
        self.assertAlmostEqual(trade.gross_pnl, 0.0, places=6)
        self.assertAlmostEqual(trade.net_pnl, -0.8, places=6)
        self.assertAlmostEqual(broker.portfolio.realized_pnl, -0.8, places=6)
        self.assertAlmostEqual(broker.portfolio.cash, 99_999.2, places=6)

    def test_update_position_protection_records_protect_event(self):
        broker = LocalPaperBroker.fresh(100_000, slippage_bps=0, commission_bps=0)
        instrument = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=10)
        signal = Signal(
            instrument=instrument,
            direction=SignalDirection.LONG,
            strength=0.8,
            entry_price=100.0,
            stop_price=95.0,
            take_profit=110.0,
            reason="test",
        )

        broker.open_position(signal, 1, datetime(2025, 1, 1, tzinfo=timezone.utc))
        updated = broker.update_position_protection(
            "SBER",
            timestamp=datetime(2025, 1, 1, 1, 0, tzinfo=timezone.utc),
            stop_price=103.0,
            reason="trailing-profit-protection",
        )

        self.assertTrue(updated)
        self.assertEqual(broker.portfolio.positions["SBER"].stop_price, 103.0)
        self.assertEqual(broker.events[-1]["action"], "protect")
        self.assertEqual(broker.events[-1]["reason"], "trailing-profit-protection")
        self.assertEqual(broker.events[-1]["stop_price"], 103.0)

    def test_short_stop_below_entry_is_recorded_as_profit_protection(self):
        broker = LocalPaperBroker.fresh(100_000, slippage_bps=0, commission_bps=0)
        instrument = Instrument(symbol="X5", instrument_type=InstrumentType.STOCK, lot_size=1)
        signal = Signal(
            instrument=instrument,
            direction=SignalDirection.SHORT,
            strength=0.8,
            entry_price=100.0,
            stop_price=105.0,
            take_profit=90.0,
            reason="test",
        )

        broker.open_position(signal, 10, datetime(2025, 1, 1, tzinfo=timezone.utc))
        broker.update_position_protection(
            "X5",
            timestamp=datetime(2025, 1, 1, 1, 0, tzinfo=timezone.utc),
            stop_price=98.0,
            reason="trailing-profit-protection",
        )
        trade = broker.close_position(
            "X5",
            price=98.0,
            timestamp=datetime(2025, 1, 1, 2, 0, tzinfo=timezone.utc),
            reason=ExitReason.STOP_LOSS,
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade.reason, ExitReason.PROFIT_PROTECT_STOP.value)
        self.assertGreater(trade.net_pnl, 0.0)
        self.assertEqual(broker.events[-1]["reason"], ExitReason.PROFIT_PROTECT_STOP.value)

    def test_activate_position_runner_moves_stop_and_records_event(self):
        broker = LocalPaperBroker.fresh(100_000, slippage_bps=0, commission_bps=0)
        instrument = Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=10)
        signal = Signal(
            instrument=instrument,
            direction=SignalDirection.LONG,
            strength=0.8,
            entry_price=100.0,
            stop_price=95.0,
            take_profit=104.0,
            reason="test",
        )

        broker.open_position(signal, 1, datetime(2025, 1, 1, tzinfo=timezone.utc))
        activated = broker.activate_position_runner(
            "SBER",
            timestamp=datetime(2025, 1, 1, 1, 0, tzinfo=timezone.utc),
            activation_price=104.0,
            stop_price=100.1,
            extreme_price=106.0,
        )

        position = broker.portfolio.positions["SBER"]
        self.assertTrue(activated)
        self.assertTrue(position.runner_active)
        self.assertEqual(position.stop_price, 100.1)
        self.assertEqual(position.runner_activation_price, 104.0)
        self.assertEqual(position.runner_extreme_price, 106.0)
        self.assertEqual(broker.events[-1]["action"], "runner-activate")
        self.assertEqual(broker.events[-1]["reason"], "take-profit-runner-activation")


if __name__ == "__main__":
    unittest.main()
