from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

from samosbor.autonomy.signal_feedback import (
    backfill_signal_feedback_for_symbol,
    build_trade_evidence,
    default_signal_horizon_bars,
    load_signal_feedback,
    record_shadow_signal,
    resolve_pending_signals,
    resolved_feedback_to_trades,
    save_signal_feedback,
    signal_feedback_path,
)
from samosbor.domain import Candle, Instrument, InstrumentType, Signal, SignalDirection, TradeRecord


class SignalFeedbackTest(unittest.TestCase):
    class FakeStrategy:
        def prepare_history(self, instrument, candles):
            return None

        def generate_signal(self, instrument, history):
            if len(history) not in {3, 4, 6}:
                return None
            last = history[-1]
            return Signal(
                instrument=instrument,
                direction=SignalDirection.LONG,
                strength=0.6 + len(history) * 0.01,
                entry_price=last.close,
                stop_price=last.close - 1.0,
                take_profit=last.close + 1.0,
                reason="test",
            )

        def allows_entry_at(self, timestamp):
            return True

    def test_signal_feedback_path_uses_state_stem(self):
        path = signal_feedback_path(Path("state/server_state.json"))
        self.assertEqual(str(path).replace("\\", "/"), "state/server_state_signal_feedback.json")

    def test_record_and_resolve_shadow_signal(self):
        payload = {"pending": [], "resolved": []}
        signal = Signal(
            instrument=Instrument(symbol="CNYRUBF", instrument_type=InstrumentType.FUTURE, lot_size=1),
            direction=SignalDirection.LONG,
            strength=0.72,
            entry_price=100.0,
            stop_price=98.0,
            take_profit=104.0,
            reason="test",
        )
        opened_at = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        record_shadow_signal(payload, signal, timestamp=opened_at, horizon_bars=3)

        candles = [
            Candle(
                timestamp=opened_at + timedelta(hours=1),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1_000_000,
            ),
            Candle(
                timestamp=opened_at + timedelta(hours=2),
                open=100.5,
                high=104.5,
                low=100.0,
                close=104.2,
                volume=1_000_000,
            ),
        ]

        resolved = resolve_pending_signals(payload, {"CNYRUBF": candles})

        self.assertEqual(len(payload["pending"]), 0)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["outcome_reason"], "take-profit")
        trades = resolved_feedback_to_trades(payload)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].signal_strength, 0.72)
        self.assertEqual(trades[0].reason, "take-profit")

    def test_shadow_signal_can_expire(self):
        payload = {"pending": [], "resolved": []}
        signal = Signal(
            instrument=Instrument(symbol="CNYRUBF", instrument_type=InstrumentType.FUTURE, lot_size=1),
            direction=SignalDirection.SHORT,
            strength=0.51,
            entry_price=100.0,
            stop_price=102.0,
            take_profit=96.0,
            reason="test",
        )
        opened_at = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        record_shadow_signal(payload, signal, timestamp=opened_at, horizon_bars=2)
        candles = [
            Candle(
                timestamp=opened_at + timedelta(hours=1),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.2,
                volume=1_000_000,
            ),
            Candle(
                timestamp=opened_at + timedelta(hours=2),
                open=100.2,
                high=101.2,
                low=99.5,
                close=100.8,
                volume=1_000_000,
            ),
        ]
        resolve_pending_signals(payload, {"CNYRUBF": candles})
        trades = resolved_feedback_to_trades(payload)

        self.assertEqual(trades[0].reason, "expired")
        self.assertAlmostEqual(trades[0].net_pnl, -0.8, places=6)

    def test_record_shadow_signal_allows_multiple_pending_for_same_symbol(self):
        payload = {"pending": [], "resolved": []}
        instrument = Instrument(symbol="OZON", instrument_type=InstrumentType.STOCK, lot_size=1)
        first = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        second = first + timedelta(minutes=15)
        signal = Signal(
            instrument=instrument,
            direction=SignalDirection.SHORT,
            strength=0.7,
            entry_price=100.0,
            stop_price=101.0,
            take_profit=98.0,
            reason="test",
        )

        record_shadow_signal(payload, signal, timestamp=first, horizon_bars=3)
        record_shadow_signal(payload, signal, timestamp=second, horizon_bars=3)
        record_shadow_signal(payload, signal, timestamp=second, horizon_bars=3)

        self.assertEqual(len(payload["pending"]), 2)
        self.assertEqual(payload["pending"][0]["created_at"], first.isoformat())
        self.assertEqual(payload["pending"][1]["created_at"], second.isoformat())

    def test_shadow_signal_uses_rub_scaled_pnl_and_commissions(self):
        payload = {"pending": [], "resolved": []}
        signal = Signal(
            instrument=Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=10),
            direction=SignalDirection.LONG,
            strength=0.8,
            entry_price=100.0,
            stop_price=98.0,
            take_profit=104.0,
            reason="test",
        )
        opened_at = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        record_shadow_signal(
            payload,
            signal,
            timestamp=opened_at,
            horizon_bars=3,
            commission_bps=10.0,
        )
        candles = [
            Candle(
                timestamp=opened_at + timedelta(hours=1),
                open=100.0,
                high=104.2,
                low=99.8,
                close=104.0,
                volume=1_000_000,
            ),
        ]

        resolve_pending_signals(payload, {"SBER": candles})
        trades = resolved_feedback_to_trades(payload)

        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(trades[0].gross_pnl, 40.0, places=6)
        self.assertAlmostEqual(trades[0].net_pnl, 37.96, places=6)
        self.assertEqual(trades[0].quantity_lots, 1)

    def test_shadow_signal_uses_four_bps_as_0_08_round_trip_cost(self):
        payload = {"pending": [], "resolved": []}
        signal = Signal(
            instrument=Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=10),
            direction=SignalDirection.LONG,
            strength=0.8,
            entry_price=100.0,
            stop_price=98.0,
            take_profit=104.0,
            reason="commission-check",
        )
        opened_at = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        record_shadow_signal(
            payload,
            signal,
            timestamp=opened_at,
            horizon_bars=1,
            commission_bps=4.0,
        )
        candles = [
            Candle(
                timestamp=opened_at + timedelta(hours=1),
                open=100.0,
                high=100.5,
                low=99.5,
                close=100.0,
                volume=1_000_000,
            ),
        ]

        resolve_pending_signals(payload, {"SBER": candles})
        trades = resolved_feedback_to_trades(payload)

        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(trades[0].gross_pnl, 0.0, places=6)
        self.assertAlmostEqual(trades[0].net_pnl, -0.8, places=6)
        self.assertAlmostEqual(payload["resolved"][0]["entry_commission"], 0.4, places=6)
        self.assertAlmostEqual(payload["resolved"][0]["exit_commission"], 0.4, places=6)

    def test_shadow_signal_take_profit_can_activate_runner(self):
        payload = {"pending": [], "resolved": []}
        signal = Signal(
            instrument=Instrument(symbol="SBER", instrument_type=InstrumentType.STOCK, lot_size=1),
            direction=SignalDirection.LONG,
            strength=0.8,
            entry_price=100.0,
            stop_price=95.0,
            take_profit=104.0,
            reason="runner-check",
        )
        opened_at = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        record_shadow_signal(
            payload,
            signal,
            timestamp=opened_at,
            horizon_bars=3,
            runner_enabled=True,
            runner_breakeven_buffer_bps=10.0,
            runner_trailing_atr_multiple=1.3,
            runner_profit_lock_ratio=0.35,
        )
        candles = [
            Candle(
                timestamp=opened_at + timedelta(hours=1),
                open=100.0,
                high=106.0,
                low=100.2,
                close=105.0,
                volume=1_000_000,
            ),
            Candle(
                timestamp=opened_at + timedelta(hours=2),
                open=105.0,
                high=105.2,
                low=102.0,
                close=103.0,
                volume=1_000_000,
            ),
        ]

        resolve_pending_signals(payload, {"SBER": candles})
        trades = resolved_feedback_to_trades(payload)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].reason, "profit-protect-stop")
        self.assertTrue(payload["resolved"][0]["runner_activated"])
        self.assertGreater(payload["resolved"][0]["final_stop_price"], 100.0)

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "signal_feedback.json"
            payload = {"pending": [{"symbol": "CNYRUBF"}], "resolved": []}
            save_signal_feedback(path, payload)
            loaded = load_signal_feedback(path)
            self.assertEqual(loaded["pending"][0]["symbol"], "CNYRUBF")

    def test_default_signal_horizon_bars_has_hourly_default(self):
        self.assertEqual(default_signal_horizon_bars("hour"), 24)
        self.assertEqual(default_signal_horizon_bars("unknown"), 24)

    def test_backfill_signal_feedback_for_symbol_generates_resolved_items(self):
        instrument = Instrument(symbol="CNYRUBF", instrument_type=InstrumentType.FUTURE, lot_size=1)
        candles = []
        ts = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        prices = [100.0, 100.5, 101.0, 102.2, 102.8, 103.4, 104.5]
        for price in prices:
            candles.append(
                Candle(
                    timestamp=ts,
                    open=price - 0.2,
                    high=price + 0.6,
                    low=price - 0.6,
                    close=price,
                    volume=1_000_000,
                )
            )
            ts += timedelta(hours=1)

        payload = {"pending": [], "resolved": []}
        generated = backfill_signal_feedback_for_symbol(
            payload,
            instrument=instrument,
            candles=candles,
            strategy=self.FakeStrategy(),
            warmup_bars=3,
            horizon_bars=2,
        )

        self.assertEqual(generated, 3)
        self.assertEqual(len(payload["resolved"]), 3)
        self.assertTrue(all(item["outcome_reason"] in {"take-profit", "expired"} for item in payload["resolved"]))

    def test_build_trade_evidence_merges_and_deduplicates_feedback(self):
        opened = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        duplicate_trade = TradeRecord(
            symbol="CNYRUBF",
            direction=SignalDirection.LONG,
            quantity_lots=1,
            entry_time=opened,
            exit_time=opened + timedelta(hours=1),
            entry_price=100.0,
            exit_price=101.5,
            gross_pnl=1.5,
            net_pnl=1.5,
            reason="take-profit",
            signal_strength=0.72,
        )
        closed_trades = [duplicate_trade]
        payload = {
            "pending": [],
            "resolved": [
                {
                    "symbol": "CNYRUBF",
                    "direction": "long",
                    "created_at": opened.isoformat(),
                    "resolved_at": (opened + timedelta(hours=1)).isoformat(),
                    "entry_price": 100.0,
                    "exit_price": 101.5,
                    "gross_pnl": 1.5,
                    "outcome_reason": "take-profit",
                    "signal_strength": 0.72,
                },
                {
                    "symbol": "USDRUBF",
                    "direction": "short",
                    "created_at": (opened + timedelta(days=1)).isoformat(),
                    "resolved_at": (opened + timedelta(days=1, hours=2)).isoformat(),
                    "entry_price": 95.0,
                    "exit_price": 92.5,
                    "gross_pnl": 2.5,
                    "outcome_reason": "take-profit",
                    "signal_strength": 0.81,
                },
            ],
        }

        evidence = build_trade_evidence(closed_trades, payload)

        self.assertEqual(evidence["evidence_source"], "closed-trades+signal-feedback")
        self.assertEqual(evidence["counts"]["closed_trades"], 1)
        self.assertEqual(evidence["counts"]["feedback_trades"], 2)
        self.assertEqual(evidence["counts"]["deduplicated_feedback_trades"], 1)
        self.assertEqual(evidence["counts"]["duplicate_feedback_trades"], 1)
        self.assertEqual(evidence["counts"]["combined_trades"], 2)
        self.assertEqual([trade.symbol for trade in evidence["trades"]], ["CNYRUBF", "USDRUBF"])


if __name__ == "__main__":
    unittest.main()
