from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from samosbor.autonomy.entry_schedule import (
    build_entry_schedule_tuning_payload,
    write_entry_schedule_tuning,
)
from samosbor.autonomy.entry_symbols import (
    build_entry_symbol_tuning_payload,
    write_entry_symbol_tuning,
)
from samosbor.domain import PortfolioState, SignalDirection, TradeRecord
from samosbor.reporting.paper_report import build_paper_report_payload, write_paper_report


def _trade(
    *,
    symbol: str,
    entry_time: datetime,
    exit_time: datetime,
    net_pnl: float,
    reason: str = "take-profit",
    direction: SignalDirection = SignalDirection.LONG,
) -> TradeRecord:
    return TradeRecord(
        symbol=symbol,
        direction=direction,
        quantity_lots=1,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=100.0,
        exit_price=101.0,
        gross_pnl=net_pnl,
        net_pnl=net_pnl,
        reason=reason,
    )


class PaperReportTest(unittest.TestCase):
    def test_paper_report_filters_by_moscow_date_and_writes_files(self):
        portfolio = PortfolioState(cash=100_000.0, realized_pnl=50.0, peak_equity=100_100.0)
        trades = [
            _trade(
                symbol="CNYRUBF",
                entry_time=datetime(2025, 1, 14, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 14, 22, 15, tzinfo=timezone.utc),
                net_pnl=100.0,
            ),
            _trade(
                symbol="IMOEXF",
                entry_time=datetime(2025, 1, 14, 8, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 14, 21, 30, tzinfo=timezone.utc),
                net_pnl=-50.0,
                reason="stop-loss",
            ),
            _trade(
                symbol="USDRUBF",
                entry_time=datetime(2025, 1, 15, 18, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 15, 21, 10, tzinfo=timezone.utc),
                net_pnl=25.0,
            ),
        ]

        payload = build_paper_report_payload(
            portfolio,
            trades,
            timezone_name="Europe/Moscow",
            report_date=date(2025, 1, 15),
            days=1,
        )

        self.assertEqual(payload["summary"]["trades"], 2)
        self.assertEqual(payload["summary"]["net_pnl_rub"], 50.0)
        self.assertEqual(payload["summary"]["win_rate_pct"], 50.0)
        self.assertEqual(len(payload["closed_trades"]), 2)
        self.assertEqual(
            [row["entry_hour"] for row in payload["entry_hour_breakdown"]],
            [9, 11],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_paper_report(output_dir, payload)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "summary.md").exists())
            self.assertTrue((output_dir / "trades.csv").exists())


class EntryScheduleAutonomyTest(unittest.TestCase):
    def test_tuning_recommends_safe_add_remove_changes(self):
        portfolio = PortfolioState(cash=100_000.0, peak_equity=100_000.0)
        trades = [
            _trade(
                symbol="CNYRUBF",
                entry_time=datetime(2025, 1, 10, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 10, 8, 0, tzinfo=timezone.utc),
                net_pnl=-100.0,
            ),
            _trade(
                symbol="CNYRUBF",
                entry_time=datetime(2025, 1, 11, 6, 5, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 11, 8, 5, tzinfo=timezone.utc),
                net_pnl=-120.0,
            ),
            _trade(
                symbol="CNYRUBF",
                entry_time=datetime(2025, 1, 12, 6, 10, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 12, 8, 10, tzinfo=timezone.utc),
                net_pnl=-80.0,
            ),
            _trade(
                symbol="CNYRUBF",
                entry_time=datetime(2025, 1, 13, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 13, 9, 0, tzinfo=timezone.utc),
                net_pnl=-40.0,
            ),
            _trade(
                symbol="CNYRUBF",
                entry_time=datetime(2025, 1, 14, 7, 10, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 14, 9, 10, tzinfo=timezone.utc),
                net_pnl=-30.0,
            ),
            _trade(
                symbol="CNYRUBF",
                entry_time=datetime(2025, 1, 15, 9, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc),
                net_pnl=200.0,
            ),
            _trade(
                symbol="CNYRUBF",
                entry_time=datetime(2025, 1, 16, 9, 5, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 16, 10, 5, tzinfo=timezone.utc),
                net_pnl=150.0,
            ),
            _trade(
                symbol="CNYRUBF",
                entry_time=datetime(2025, 1, 17, 9, 10, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 17, 10, 10, tzinfo=timezone.utc),
                net_pnl=125.0,
            ),
        ]

        payload = build_entry_schedule_tuning_payload(
            portfolio,
            trades,
            timezone_name="Europe/Moscow",
            current_hours=[9, 10, 11],
            report_date=date(2025, 1, 31),
            lookback_days=45,
            min_trades_per_hour=3,
            max_hours_to_add=1,
            max_hours_to_remove=1,
        )

        self.assertTrue(payload["changed"])
        self.assertEqual(payload["evidence_source"], "closed-trades")
        self.assertEqual(payload["removals"], [9])
        self.assertEqual(payload["additions"], [12])
        self.assertEqual(payload["proposed_hours"], [10, 11, 12])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_entry_schedule_tuning(output_dir, payload)
            self.assertTrue((output_dir / "schedule_tuning.json").exists())
            self.assertTrue((output_dir / "schedule_patch.toml").exists())
            self.assertTrue((output_dir / "summary.md").exists())

    def test_tuning_keeps_schedule_when_runtime_window_would_become_too_narrow(self):
        portfolio = PortfolioState(cash=100_000.0, peak_equity=100_000.0)
        trades = [
            _trade(
                symbol="CNYRUBF",
                entry_time=datetime(2025, 1, 10, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 10, 8, 0, tzinfo=timezone.utc),
                net_pnl=-100.0,
            ),
            _trade(
                symbol="CNYRUBF",
                entry_time=datetime(2025, 1, 11, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 11, 9, 0, tzinfo=timezone.utc),
                net_pnl=-90.0,
            ),
            _trade(
                symbol="CNYRUBF",
                entry_time=datetime(2025, 1, 12, 8, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 12, 10, 0, tzinfo=timezone.utc),
                net_pnl=-80.0,
            ),
            _trade(
                symbol="CNYRUBF",
                entry_time=datetime(2025, 1, 13, 9, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 13, 11, 0, tzinfo=timezone.utc),
                net_pnl=-70.0,
            ),
        ]

        payload = build_entry_schedule_tuning_payload(
            portfolio,
            trades,
            timezone_name="Europe/Moscow",
            current_hours=[9, 10, 11, 12, 13, 14],
            report_date=date(2025, 1, 31),
            lookback_days=45,
            min_trades_per_hour=1,
            max_hours_to_remove=2,
            min_active_hours=6,
        )

        self.assertFalse(payload["changed"])
        self.assertEqual(payload["proposed_hours"], [9, 10, 11, 12, 13, 14])
        self.assertIn("over-narrow", payload["reason"])

    def test_tuning_keeps_empty_hours_as_trade_anytime_policy(self):
        portfolio = PortfolioState(cash=100_000.0, peak_equity=100_000.0)
        trades = [
            _trade(
                symbol="CNYRUBF",
                entry_time=datetime(2025, 1, 10, 9, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 10, 10, 0, tzinfo=timezone.utc),
                net_pnl=100.0,
            ),
            _trade(
                symbol="CNYRUBF",
                entry_time=datetime(2025, 1, 11, 9, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 11, 10, 0, tzinfo=timezone.utc),
                net_pnl=120.0,
            ),
        ]

        payload = build_entry_schedule_tuning_payload(
            portfolio,
            trades,
            timezone_name="Europe/Moscow",
            current_hours=[],
            report_date=date(2025, 1, 31),
            lookback_days=45,
            min_trades_per_hour=1,
        )

        self.assertFalse(payload["changed"])
        self.assertEqual(payload["proposed_hours"], [])
        self.assertEqual(payload["additions"], [])
        self.assertEqual(payload["removals"], [])
        self.assertIn("without hour filters", payload["reason"])


class EntrySymbolAutonomyTest(unittest.TestCase):
    def test_tuning_recommends_blocking_weak_symbols(self):
        portfolio = PortfolioState(cash=100_000.0, peak_equity=100_000.0)
        trades = [
            _trade(
                symbol="CNYRUBF",
                entry_time=datetime(2025, 1, 10, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 10, 8, 0, tzinfo=timezone.utc),
                net_pnl=200.0,
            ),
            _trade(
                symbol="CNYRUBF",
                entry_time=datetime(2025, 1, 11, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 11, 8, 0, tzinfo=timezone.utc),
                net_pnl=150.0,
            ),
            _trade(
                symbol="CNYRUBF",
                entry_time=datetime(2025, 1, 12, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 12, 8, 0, tzinfo=timezone.utc),
                net_pnl=100.0,
            ),
            _trade(
                symbol="IMOEXF",
                entry_time=datetime(2025, 1, 10, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 10, 9, 0, tzinfo=timezone.utc),
                net_pnl=-100.0,
            ),
            _trade(
                symbol="IMOEXF",
                entry_time=datetime(2025, 1, 11, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 11, 9, 0, tzinfo=timezone.utc),
                net_pnl=-90.0,
            ),
            _trade(
                symbol="IMOEXF",
                entry_time=datetime(2025, 1, 12, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 12, 9, 0, tzinfo=timezone.utc),
                net_pnl=-80.0,
            ),
            _trade(
                symbol="IMOEXF",
                entry_time=datetime(2025, 1, 13, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 13, 9, 0, tzinfo=timezone.utc),
                net_pnl=-70.0,
            ),
        ]

        payload = build_entry_symbol_tuning_payload(
            portfolio,
            trades,
            timezone_name="Europe/Moscow",
            current_blocked_symbols=[],
            current_blocked_long_symbols=[],
            current_blocked_short_symbols=[],
            evidence_source="signal-feedback",
            report_date=date(2025, 1, 31),
            lookback_days=45,
            min_trades_per_symbol=4,
            min_trades_per_direction_symbol=4,
            max_symbols_to_block=1,
            max_total_blocked_symbols=4,
        )

        self.assertTrue(payload["changed"])
        self.assertEqual(payload["evidence_source"], "signal-feedback")
        self.assertEqual(payload["additions"], ["IMOEXF"])
        self.assertEqual(payload["proposed_blocked_symbols"], ["IMOEXF"])
        self.assertEqual(payload["proposed_blocked_long_symbols"], [])
        self.assertEqual(payload["proposed_blocked_short_symbols"], [])
        payload["evidence_counts"] = {
            "combined_trades": 7,
            "closed_trades": 3,
            "deduplicated_feedback_trades": 4,
            "duplicate_feedback_trades": 1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_entry_symbol_tuning(output_dir, payload)
            self.assertTrue((output_dir / "symbol_restrictions.json").exists())
            self.assertTrue((output_dir / "symbol_restrictions_patch.toml").exists())
            self.assertTrue((output_dir / "summary.md").exists())
            self.assertIn(
                "Evidence counts: combined=7, closed=3, feedback=4, dupes=1",
                (output_dir / "summary.md").read_text(encoding="utf-8"),
            )

    def test_tuning_can_block_only_weak_direction_for_symbol(self):
        portfolio = PortfolioState(cash=100_000.0, peak_equity=100_000.0)
        trades = [
            _trade(
                symbol="SBER",
                direction=SignalDirection.LONG,
                entry_time=datetime(2025, 1, 10, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 10, 8, 0, tzinfo=timezone.utc),
                net_pnl=-100.0,
            ),
            _trade(
                symbol="SBER",
                direction=SignalDirection.LONG,
                entry_time=datetime(2025, 1, 11, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 11, 8, 0, tzinfo=timezone.utc),
                net_pnl=-90.0,
            ),
            _trade(
                symbol="SBER",
                direction=SignalDirection.LONG,
                entry_time=datetime(2025, 1, 12, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 12, 8, 0, tzinfo=timezone.utc),
                net_pnl=-80.0,
            ),
            _trade(
                symbol="SBER",
                direction=SignalDirection.LONG,
                entry_time=datetime(2025, 1, 13, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 13, 8, 0, tzinfo=timezone.utc),
                net_pnl=-70.0,
            ),
            _trade(
                symbol="SBER",
                direction=SignalDirection.SHORT,
                entry_time=datetime(2025, 1, 10, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 10, 9, 0, tzinfo=timezone.utc),
                net_pnl=170.0,
            ),
            _trade(
                symbol="SBER",
                direction=SignalDirection.SHORT,
                entry_time=datetime(2025, 1, 11, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 11, 9, 0, tzinfo=timezone.utc),
                net_pnl=160.0,
            ),
            _trade(
                symbol="SBER",
                direction=SignalDirection.SHORT,
                entry_time=datetime(2025, 1, 12, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 12, 9, 0, tzinfo=timezone.utc),
                net_pnl=150.0,
            ),
            _trade(
                symbol="SBER",
                direction=SignalDirection.SHORT,
                entry_time=datetime(2025, 1, 13, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 13, 9, 0, tzinfo=timezone.utc),
                net_pnl=140.0,
            ),
            _trade(
                symbol="GAZP",
                direction=SignalDirection.LONG,
                entry_time=datetime(2025, 1, 10, 8, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 10, 10, 0, tzinfo=timezone.utc),
                net_pnl=180.0,
            ),
            _trade(
                symbol="GAZP",
                direction=SignalDirection.LONG,
                entry_time=datetime(2025, 1, 11, 8, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 11, 10, 0, tzinfo=timezone.utc),
                net_pnl=170.0,
            ),
            _trade(
                symbol="GAZP",
                direction=SignalDirection.LONG,
                entry_time=datetime(2025, 1, 12, 8, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 12, 10, 0, tzinfo=timezone.utc),
                net_pnl=160.0,
            ),
            _trade(
                symbol="GAZP",
                direction=SignalDirection.LONG,
                entry_time=datetime(2025, 1, 13, 8, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 13, 10, 0, tzinfo=timezone.utc),
                net_pnl=150.0,
            ),
            _trade(
                symbol="GAZP",
                direction=SignalDirection.SHORT,
                entry_time=datetime(2025, 1, 10, 9, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 10, 11, 0, tzinfo=timezone.utc),
                net_pnl=-100.0,
            ),
            _trade(
                symbol="GAZP",
                direction=SignalDirection.SHORT,
                entry_time=datetime(2025, 1, 11, 9, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 11, 11, 0, tzinfo=timezone.utc),
                net_pnl=-90.0,
            ),
            _trade(
                symbol="GAZP",
                direction=SignalDirection.SHORT,
                entry_time=datetime(2025, 1, 12, 9, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 12, 11, 0, tzinfo=timezone.utc),
                net_pnl=-80.0,
            ),
            _trade(
                symbol="GAZP",
                direction=SignalDirection.SHORT,
                entry_time=datetime(2025, 1, 13, 9, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 13, 11, 0, tzinfo=timezone.utc),
                net_pnl=-70.0,
            ),
        ]

        payload = build_entry_symbol_tuning_payload(
            portfolio,
            trades,
            timezone_name="Europe/Moscow",
            current_blocked_symbols=[],
            current_blocked_long_symbols=[],
            current_blocked_short_symbols=[],
            evidence_source="signal-feedback",
            report_date=date(2025, 1, 31),
            lookback_days=45,
            min_trades_per_symbol=4,
            min_trades_per_direction_symbol=4,
            max_symbols_to_block=1,
            max_total_blocked_symbols=4,
        )

        self.assertTrue(payload["changed"])
        self.assertEqual(payload["proposed_blocked_symbols"], [])
        self.assertEqual(payload["long_additions"], ["SBER"])
        self.assertEqual(payload["short_additions"], ["GAZP"])
        self.assertEqual(payload["proposed_blocked_long_symbols"], ["SBER"])
        self.assertEqual(payload["proposed_blocked_short_symbols"], ["GAZP"])

    def test_directional_tuning_uses_unrounded_negative_expectancy(self):
        portfolio = PortfolioState(cash=100_000.0, peak_equity=100_000.0)
        trades = [
            _trade(
                symbol="VTBR",
                direction=SignalDirection.LONG,
                entry_time=datetime(2025, 1, 10, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 10, 8, 0, tzinfo=timezone.utc),
                net_pnl=0.10,
            ),
            _trade(
                symbol="VTBR",
                direction=SignalDirection.LONG,
                entry_time=datetime(2025, 1, 11, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 11, 8, 0, tzinfo=timezone.utc),
                net_pnl=0.09,
            ),
            _trade(
                symbol="VTBR",
                direction=SignalDirection.LONG,
                entry_time=datetime(2025, 1, 12, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 12, 8, 0, tzinfo=timezone.utc),
                net_pnl=0.08,
            ),
            _trade(
                symbol="VTBR",
                direction=SignalDirection.LONG,
                entry_time=datetime(2025, 1, 13, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 13, 8, 0, tzinfo=timezone.utc),
                net_pnl=0.07,
            ),
            _trade(
                symbol="VTBR",
                direction=SignalDirection.SHORT,
                entry_time=datetime(2025, 1, 10, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 10, 9, 0, tzinfo=timezone.utc),
                net_pnl=-0.004,
            ),
            _trade(
                symbol="VTBR",
                direction=SignalDirection.SHORT,
                entry_time=datetime(2025, 1, 11, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 11, 9, 0, tzinfo=timezone.utc),
                net_pnl=-0.004,
            ),
            _trade(
                symbol="VTBR",
                direction=SignalDirection.SHORT,
                entry_time=datetime(2025, 1, 12, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 12, 9, 0, tzinfo=timezone.utc),
                net_pnl=-0.004,
            ),
            _trade(
                symbol="VTBR",
                direction=SignalDirection.SHORT,
                entry_time=datetime(2025, 1, 13, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 13, 9, 0, tzinfo=timezone.utc),
                net_pnl=-0.004,
            ),
        ]

        payload = build_entry_symbol_tuning_payload(
            portfolio,
            trades,
            timezone_name="Europe/Moscow",
            current_blocked_symbols=[],
            current_blocked_long_symbols=[],
            current_blocked_short_symbols=[],
            evidence_source="signal-feedback",
            report_date=date(2025, 1, 31),
            lookback_days=45,
            min_trades_per_symbol=4,
            min_trades_per_direction_symbol=4,
        )

        self.assertEqual(payload["proposed_blocked_symbols"], [])
        self.assertEqual(payload["short_additions"], ["VTBR"])
        self.assertEqual(payload["proposed_blocked_short_symbols"], ["VTBR"])

    def test_tuning_can_unblock_recovered_symbol_and_replace_it_with_current_loser(self):
        portfolio = PortfolioState(cash=100_000.0, peak_equity=100_000.0)
        trades = [
            _trade(
                symbol="SBER",
                entry_time=datetime(2025, 1, 10, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 10, 8, 0, tzinfo=timezone.utc),
                net_pnl=120.0,
            ),
            _trade(
                symbol="SBER",
                entry_time=datetime(2025, 1, 11, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 11, 8, 0, tzinfo=timezone.utc),
                net_pnl=110.0,
            ),
            _trade(
                symbol="SBER",
                entry_time=datetime(2025, 1, 12, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 12, 8, 0, tzinfo=timezone.utc),
                net_pnl=100.0,
            ),
            _trade(
                symbol="SBER",
                entry_time=datetime(2025, 1, 13, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 13, 8, 0, tzinfo=timezone.utc),
                net_pnl=90.0,
            ),
            _trade(
                symbol="GAZP",
                entry_time=datetime(2025, 1, 10, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 10, 9, 0, tzinfo=timezone.utc),
                net_pnl=-100.0,
            ),
            _trade(
                symbol="GAZP",
                entry_time=datetime(2025, 1, 11, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 11, 9, 0, tzinfo=timezone.utc),
                net_pnl=-90.0,
            ),
            _trade(
                symbol="GAZP",
                entry_time=datetime(2025, 1, 12, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 12, 9, 0, tzinfo=timezone.utc),
                net_pnl=-80.0,
            ),
            _trade(
                symbol="GAZP",
                entry_time=datetime(2025, 1, 13, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 13, 9, 0, tzinfo=timezone.utc),
                net_pnl=-70.0,
            ),
        ]

        payload = build_entry_symbol_tuning_payload(
            portfolio,
            trades,
            timezone_name="Europe/Moscow",
            current_blocked_symbols=["SBER"],
            current_blocked_long_symbols=[],
            current_blocked_short_symbols=[],
            evidence_source="signal-feedback",
            report_date=date(2025, 1, 31),
            lookback_days=45,
            min_trades_per_symbol=4,
            max_symbols_to_block=1,
            max_total_blocked_symbols=4,
            max_symbols_to_unblock=1,
        )

        self.assertTrue(payload["changed"])
        self.assertEqual(payload["removals"], ["SBER"])
        self.assertEqual(payload["additions"], ["GAZP"])
        self.assertEqual(payload["proposed_blocked_symbols"], ["GAZP"])

    def test_tuning_can_unblock_recovered_directional_restrictions(self):
        portfolio = PortfolioState(cash=100_000.0, peak_equity=100_000.0)
        trades = [
            _trade(
                symbol="SBER",
                direction=SignalDirection.LONG,
                entry_time=datetime(2025, 1, 10, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 10, 8, 0, tzinfo=timezone.utc),
                net_pnl=120.0,
            ),
            _trade(
                symbol="SBER",
                direction=SignalDirection.LONG,
                entry_time=datetime(2025, 1, 11, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 11, 8, 0, tzinfo=timezone.utc),
                net_pnl=110.0,
            ),
            _trade(
                symbol="SBER",
                direction=SignalDirection.LONG,
                entry_time=datetime(2025, 1, 12, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 12, 8, 0, tzinfo=timezone.utc),
                net_pnl=100.0,
            ),
            _trade(
                symbol="SBER",
                direction=SignalDirection.LONG,
                entry_time=datetime(2025, 1, 13, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 13, 8, 0, tzinfo=timezone.utc),
                net_pnl=90.0,
            ),
            _trade(
                symbol="GAZP",
                direction=SignalDirection.SHORT,
                entry_time=datetime(2025, 1, 10, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 10, 9, 0, tzinfo=timezone.utc),
                net_pnl=130.0,
            ),
            _trade(
                symbol="GAZP",
                direction=SignalDirection.SHORT,
                entry_time=datetime(2025, 1, 11, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 11, 9, 0, tzinfo=timezone.utc),
                net_pnl=120.0,
            ),
            _trade(
                symbol="GAZP",
                direction=SignalDirection.SHORT,
                entry_time=datetime(2025, 1, 12, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 12, 9, 0, tzinfo=timezone.utc),
                net_pnl=110.0,
            ),
            _trade(
                symbol="GAZP",
                direction=SignalDirection.SHORT,
                entry_time=datetime(2025, 1, 13, 7, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 13, 9, 0, tzinfo=timezone.utc),
                net_pnl=100.0,
            ),
        ]

        payload = build_entry_symbol_tuning_payload(
            portfolio,
            trades,
            timezone_name="Europe/Moscow",
            current_blocked_symbols=[],
            current_blocked_long_symbols=["SBER"],
            current_blocked_short_symbols=["GAZP"],
            evidence_source="signal-feedback",
            report_date=date(2025, 1, 31),
            lookback_days=45,
            min_trades_per_symbol=4,
            min_trades_per_direction_symbol=4,
            max_symbols_to_block=0,
            max_long_symbols_to_block=0,
            max_short_symbols_to_block=0,
            max_long_symbols_to_unblock=1,
            max_short_symbols_to_unblock=1,
        )

        self.assertTrue(payload["changed"])
        self.assertEqual(payload["long_removals"], ["SBER"])
        self.assertEqual(payload["short_removals"], ["GAZP"])
        self.assertEqual(payload["proposed_blocked_long_symbols"], [])
        self.assertEqual(payload["proposed_blocked_short_symbols"], [])

    def test_tuning_skips_restrictions_when_runtime_universe_is_too_small(self):
        portfolio = PortfolioState(cash=100_000.0, peak_equity=100_000.0)
        trades = [
            _trade(
                symbol="SBER",
                entry_time=datetime(2025, 1, 10, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 10, 8, 0, tzinfo=timezone.utc),
                net_pnl=-100.0,
            ),
            _trade(
                symbol="SBER",
                entry_time=datetime(2025, 1, 11, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 11, 8, 0, tzinfo=timezone.utc),
                net_pnl=-90.0,
            ),
            _trade(
                symbol="SBER",
                entry_time=datetime(2025, 1, 12, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 12, 8, 0, tzinfo=timezone.utc),
                net_pnl=-80.0,
            ),
            _trade(
                symbol="SBER",
                entry_time=datetime(2025, 1, 13, 6, 0, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 13, 8, 0, tzinfo=timezone.utc),
                net_pnl=-70.0,
            ),
        ]

        payload = build_entry_symbol_tuning_payload(
            portfolio,
            trades,
            timezone_name="Europe/Moscow",
            current_blocked_symbols=[],
            current_blocked_long_symbols=[],
            current_blocked_short_symbols=[],
            evidence_source="signal-feedback",
            report_date=date(2025, 1, 31),
            lookback_days=45,
            min_trades_per_symbol=4,
            runtime_symbols=["SBER", "GAZP", "LKOH", "TATN"],
            min_runtime_symbols_for_restrictions=8,
        )

        self.assertFalse(payload["changed"])
        self.assertEqual(payload["proposed_blocked_symbols"], [])
        self.assertIn("too small", payload["reason"])


if __name__ == "__main__":
    unittest.main()
