from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from samosbor.config import BacktestSection, ResearchSection, RiskSection, StrategySection
from samosbor.domain import Candle, Instrument, InstrumentType
from samosbor.research.walk_forward import WalkForwardValidator


class WalkForwardValidatorTest(unittest.TestCase):
    def test_walk_forward_returns_fold_summary(self):
        instrument = Instrument(symbol="TEST", instrument_type=InstrumentType.STOCK, lot_size=1)
        candles = []
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        price = 100.0
        for _ in range(280):
            candles.append(
                Candle(
                    timestamp=ts,
                    open=price,
                    high=price * 1.01,
                    low=price * 0.995,
                    close=price * 1.004,
                    volume=5_000_000,
                )
            )
            ts += timedelta(days=1)
            price *= 1.003

        validator = WalkForwardValidator(
            base_strategy=StrategySection(min_liquidity_rub=1.0, fast_window=5, slow_window=20),
            risk=RiskSection(),
            backtest=BacktestSection(initial_cash=100_000, warmup_bars=20),
            research=ResearchSection(
                strategy_styles=["sma_breakout"],
                fast_windows=[5],
                slow_windows=[20],
                require_breakout_values=[True],
                atr_stop_multipliers=[1.5],
                reward_to_risk_values=[2.0],
                trend_strength_values=[0.002],
                subset_min_size=1,
                subset_max_size=1,
                top_n=3,
                min_trades=1,
                walk_forward_train_months=3,
                walk_forward_test_months=1,
                walk_forward_step_months=1,
            ),
            timeframe="day",
            slippage_bps=0,
            commission_bps=0,
        )

        payload = validator.run({"TEST": candles}, {"TEST": instrument})

        self.assertGreaterEqual(payload["summary"]["folds_evaluated"], 1)
        self.assertIn("average_test_total_return_pct", payload["summary"])
        self.assertIn("average_test_normalized_monthly_return_pct", payload["summary"])
        self.assertGreaterEqual(len(payload["folds"]), 1)
        self.assertIn("normalized_monthly_return_pct", payload["folds"][0]["test_summary"])


if __name__ == "__main__":
    unittest.main()
