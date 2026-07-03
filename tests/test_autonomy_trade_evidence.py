from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from samosbor.autonomy.signal_feedback import save_signal_feedback, signal_feedback_path
from samosbor.config import load_config
from samosbor.domain import PortfolioState, SignalDirection, TradeRecord
from samosbor.orchestrator import TradingOrchestrator


def _trade(
    *,
    symbol: str,
    direction: SignalDirection,
    opened: datetime,
    net_pnl: float,
    signal_strength: float,
) -> TradeRecord:
    return TradeRecord(
        symbol=symbol,
        direction=direction,
        quantity_lots=1,
        entry_time=opened,
        exit_time=opened + timedelta(hours=1),
        entry_price=100.0,
        exit_price=101.0,
        gross_pnl=net_pnl,
        net_pnl=net_pnl,
        reason="take-profit" if net_pnl >= 0 else "stop-loss",
        signal_strength=signal_strength,
    )


class FakeBroker:
    def __init__(self, trades: list[TradeRecord]):
        self.trades = trades
        self.portfolio = PortfolioState(cash=1_000_000.0)


class AutonomyTradeEvidenceTest(unittest.TestCase):
    def test_tune_entry_quality_uses_mixed_trade_evidence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            state_dir = root / "state"
            config_dir.mkdir(parents=True)
            state_dir.mkdir(parents=True)

            config_path = config_dir / "paper.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[app]",
                        'timezone = "Europe/Moscow"',
                        "",
                        "[data]",
                        'source = "csv"',
                        'csv_path = "data/demo.csv"',
                        "",
                        "[strategy]",
                        'style = "ema_adx_macd"',
                        "min_signal_strength = 0.0",
                        "",
                        "[execution]",
                        'mode = "local-paper"',
                        "allow_live_trading = false",
                        'state_path = "state/paper_state.json"',
                        "",
                        "[backtest]",
                        "",
                        "[reporting]",
                        'output_dir = "runs"',
                        "",
                        "[research]",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = load_config(config_path)
            orchestrator = TradingOrchestrator(config)
            broker_trade = _trade(
                symbol="CNYRUBF",
                direction=SignalDirection.LONG,
                opened=datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc),
                net_pnl=-120.0,
                signal_strength=0.2,
            )
            orchestrator._load_paper_broker = lambda: FakeBroker([broker_trade])  # type: ignore[method-assign]

            feedback_path = signal_feedback_path(config.resolve_path(config.execution.state_path))
            save_signal_feedback(
                feedback_path,
                {
                    "pending": [],
                    "resolved": [
                        {
                            "symbol": "USDRUBF",
                            "direction": "long",
                            "created_at": datetime(2025, 1, 2, 10, 0, tzinfo=timezone.utc).isoformat(),
                            "resolved_at": datetime(2025, 1, 2, 11, 0, tzinfo=timezone.utc).isoformat(),
                            "entry_price": 100.0,
                            "exit_price": 103.0,
                            "gross_pnl": 150.0,
                            "outcome_reason": "take-profit",
                            "signal_strength": 0.85,
                        }
                    ],
                },
            )

            payload = orchestrator.tune_entry_quality(
                lookback_trades=10,
                min_trades=2,
                min_trade_retention_ratio=0.0,
                min_expectancy_improvement_rub=0.0,
                bucket_step=0.05,
            )

            self.assertEqual(payload["evidence_source"], "closed-trades+signal-feedback")
            self.assertEqual(
                payload["evidence_counts"],
                {
                    "closed_trades": 1,
                    "feedback_trades": 1,
                    "deduplicated_feedback_trades": 1,
                    "duplicate_feedback_trades": 0,
                    "combined_trades": 2,
                },
            )
            self.assertEqual(payload["lookback"]["eligible_trades"], 2)


if __name__ == "__main__":
    unittest.main()
