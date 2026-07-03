from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from samosbor.config import BacktestSection, ResearchSection, RiskSection, StrategySection
from samosbor.domain import Candle, Instrument, InstrumentType
from samosbor.research.optimizer import ParameterOptimizer


class OptimizerTest(unittest.TestCase):
    def test_optimizer_returns_ranked_candidates(self):
        instrument = Instrument(symbol="TEST", instrument_type=InstrumentType.STOCK, lot_size=1)
        candles = []
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        price = 100.0
        for _ in range(120):
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
            ts += timedelta(hours=1)
            price *= 1.003

        optimizer = ParameterOptimizer(
            base_strategy=StrategySection(min_liquidity_rub=1.0),
            risk=RiskSection(),
            backtest=BacktestSection(initial_cash=100_000, warmup_bars=30),
            research=ResearchSection(
                fast_windows=[10],
                slow_windows=[30],
                atr_stop_multipliers=[1.5],
                reward_to_risk_values=[2.0],
                trend_strength_values=[0.002],
                subset_min_size=1,
                subset_max_size=1,
                top_n=3,
                min_trades=1,
            ),
            timeframe="hour",
            slippage_bps=0,
            commission_bps=0,
        )
        payload = optimizer.run({"TEST": candles}, {"TEST": instrument})

        self.assertEqual(payload["evaluated_candidates"], 1)
        self.assertIsNotNone(payload["best_candidate"])
        self.assertEqual(len(payload["top_candidates"]), 1)
        self.assertEqual(payload["best_candidate"]["style"], "sma_breakout")
        self.assertTrue(payload["best_candidate"]["require_breakout"])
        self.assertEqual(payload["best_candidate"]["adx_min"], 20.0)
        self.assertEqual(payload["best_candidate"]["breakeven_trigger_pct"], 0.0)
        self.assertEqual(payload["best_candidate"]["trailing_profit_trigger_rub"], 0.0)
        self.assertEqual(payload["best_candidate"]["trailing_profit_lock_ratio"], 0.0)

    def test_optimizer_can_evaluate_ta_style_candidates(self):
        instrument = Instrument(symbol="TEST", instrument_type=InstrumentType.STOCK, lot_size=1)
        candles = []
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        price = 100.0
        for _ in range(140):
            candles.append(
                Candle(
                    timestamp=ts,
                    open=price,
                    high=price * 1.012,
                    low=price * 0.994,
                    close=price * 1.006,
                    volume=5_000_000,
                )
            )
            ts += timedelta(hours=1)
            price *= 1.004

        optimizer = ParameterOptimizer(
            base_strategy=StrategySection(min_liquidity_rub=1.0),
            risk=RiskSection(),
            backtest=BacktestSection(initial_cash=100_000, warmup_bars=30),
            research=ResearchSection(
                strategy_styles=["ema_adx_macd"],
                fast_windows=[10],
                slow_windows=[30],
                require_breakout_values=[False],
                atr_stop_multipliers=[1.5],
                reward_to_risk_values=[2.0],
                trend_strength_values=[0.002],
                adx_min_values=[15.0],
                subset_min_size=1,
                subset_max_size=1,
                top_n=3,
                min_trades=1,
            ),
            timeframe="hour",
            slippage_bps=0,
            commission_bps=0,
        )

        payload = optimizer.run({"TEST": candles}, {"TEST": instrument})

        self.assertEqual(payload["evaluated_candidates"], 1)
        self.assertIsNotNone(payload["best_candidate"])
        self.assertEqual(payload["best_candidate"]["style"], "ema_adx_macd")
        self.assertFalse(payload["best_candidate"]["require_breakout"])
        self.assertEqual(payload["best_candidate"]["adx_min"], 15.0)

    def test_optimizer_can_evaluate_mean_reversion_style_candidates(self):
        instrument = Instrument(symbol="TEST", instrument_type=InstrumentType.STOCK, lot_size=1)
        candles = []
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        price = 100.0
        for index in range(180):
            close = price + (0.35 if index % 2 == 0 else -0.45)
            if index in {59, 89, 119, 149, 179}:
                close -= 4.0
            candles.append(
                Candle(
                    timestamp=ts,
                    open=price,
                    high=max(price, close) + 0.9,
                    low=min(price, close) - 0.9,
                    close=close,
                    volume=5_000_000,
                )
            )
            ts += timedelta(hours=1)
            price = close

        optimizer = ParameterOptimizer(
            base_strategy=StrategySection(min_liquidity_rub=1.0),
            risk=RiskSection(),
            backtest=BacktestSection(initial_cash=100_000, warmup_bars=30),
            research=ResearchSection(
                strategy_styles=["rsi_mean_reversion"],
                fast_windows=[10],
                slow_windows=[30],
                atr_stop_multipliers=[1.5],
                reward_to_risk_values=[2.0],
                trend_strength_values=[0.002],
                rsi_long_max_values=[65.0],
                rsi_short_min_values=[35.0],
                subset_min_size=1,
                subset_max_size=1,
                top_n=3,
                min_trades=1,
            ),
            timeframe="hour",
            slippage_bps=0,
            commission_bps=0,
        )

        payload = optimizer.run({"TEST": candles}, {"TEST": instrument})

        self.assertEqual(payload["evaluated_candidates"], 1)
        self.assertIsNotNone(payload["best_candidate"])
        self.assertEqual(payload["best_candidate"]["style"], "rsi_mean_reversion")
        self.assertEqual(payload["best_candidate"]["rsi_long_max"], 65.0)
        self.assertEqual(payload["best_candidate"]["rsi_short_min"], 35.0)

    def test_optimizer_can_evaluate_donchian_ta_style_candidates(self):
        instrument = Instrument(symbol="TEST", instrument_type=InstrumentType.STOCK, lot_size=1)
        candles = []
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        price = 100.0
        for _ in range(160):
            candles.append(
                Candle(
                    timestamp=ts,
                    open=price,
                    high=price * 1.015,
                    low=price * 0.993,
                    close=price * 1.007,
                    volume=5_000_000,
                )
            )
            ts += timedelta(hours=1)
            price *= 1.004

        optimizer = ParameterOptimizer(
            base_strategy=StrategySection(min_liquidity_rub=1.0),
            risk=RiskSection(),
            backtest=BacktestSection(initial_cash=100_000, warmup_bars=30),
            research=ResearchSection(
                strategy_styles=["ema_adx_donchian"],
                fast_windows=[10],
                slow_windows=[30],
                atr_stop_multipliers=[1.5],
                reward_to_risk_values=[2.0],
                trend_strength_values=[0.002],
                adx_min_values=[15.0],
                subset_min_size=1,
                subset_max_size=1,
                top_n=3,
                min_trades=1,
            ),
            timeframe="hour",
            slippage_bps=0,
            commission_bps=0,
        )

        payload = optimizer.run({"TEST": candles}, {"TEST": instrument})

        self.assertEqual(payload["evaluated_candidates"], 1)
        self.assertIsNotNone(payload["best_candidate"])
        self.assertEqual(payload["best_candidate"]["style"], "ema_adx_donchian")
        self.assertEqual(payload["best_candidate"]["adx_min"], 15.0)

    def test_optimizer_can_evaluate_hybrid_style_candidates(self):
        instrument = Instrument(symbol="TEST", instrument_type=InstrumentType.STOCK, lot_size=1)
        candles = []
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        price = 100.0
        for _ in range(180):
            candles.append(
                Candle(
                    timestamp=ts,
                    open=price,
                    high=price * 1.015,
                    low=price * 0.993,
                    close=price * 1.007,
                    volume=5_000_000,
                )
            )
            ts += timedelta(hours=1)
            price *= 1.004

        optimizer = ParameterOptimizer(
            base_strategy=StrategySection(min_liquidity_rub=1.0),
            risk=RiskSection(),
            backtest=BacktestSection(initial_cash=100_000, warmup_bars=30),
            research=ResearchSection(
                strategy_styles=["adx_regime_hybrid"],
                fast_windows=[10],
                slow_windows=[30],
                require_breakout_values=[False],
                atr_stop_multipliers=[1.5],
                reward_to_risk_values=[2.0],
                trend_strength_values=[0.002],
                adx_min_values=[15.0],
                rsi_long_max_values=[75.0],
                rsi_short_min_values=[25.0],
                subset_min_size=1,
                subset_max_size=1,
                top_n=3,
                min_trades=1,
            ),
            timeframe="hour",
            slippage_bps=0,
            commission_bps=0,
        )

        payload = optimizer.run({"TEST": candles}, {"TEST": instrument})

        self.assertEqual(payload["evaluated_candidates"], 1)
        self.assertIsNotNone(payload["best_candidate"])
        self.assertEqual(payload["best_candidate"]["style"], "adx_regime_hybrid")
        self.assertEqual(payload["best_candidate"]["adx_min"], 15.0)

    def test_optimizer_can_include_trailing_profit_parameters(self):
        instrument = Instrument(symbol="TEST", instrument_type=InstrumentType.STOCK, lot_size=1)
        candles = []
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        price = 100.0
        for _ in range(150):
            candles.append(
                Candle(
                    timestamp=ts,
                    open=price,
                    high=price * 1.014,
                    low=price * 0.994,
                    close=price * 1.006,
                    volume=5_000_000,
                )
            )
            ts += timedelta(hours=1)
            price *= 1.004

        optimizer = ParameterOptimizer(
            base_strategy=StrategySection(min_liquidity_rub=1.0),
            risk=RiskSection(),
            backtest=BacktestSection(initial_cash=100_000, warmup_bars=30),
            research=ResearchSection(
                strategy_styles=["ema_adx_macd"],
                fast_windows=[10],
                slow_windows=[30],
                require_breakout_values=[False],
                atr_stop_multipliers=[1.5],
                reward_to_risk_values=[2.0],
                trailing_profit_trigger_rub_values=[1_000.0],
                trailing_profit_lock_ratio_values=[0.5],
                trend_strength_values=[0.002],
                adx_min_values=[15.0],
                subset_min_size=1,
                subset_max_size=1,
                top_n=3,
                min_trades=1,
            ),
            timeframe="hour",
            slippage_bps=0,
            commission_bps=0,
        )

        payload = optimizer.run({"TEST": candles}, {"TEST": instrument})

        self.assertEqual(payload["evaluated_candidates"], 1)
        self.assertEqual(payload["best_candidate"]["trailing_profit_trigger_rub"], 1_000.0)
        self.assertEqual(payload["best_candidate"]["trailing_profit_lock_ratio"], 0.5)

    def test_strategy_from_candidate_payload_can_include_breakeven_trigger_parameter(self):
        optimizer = ParameterOptimizer(
            base_strategy=StrategySection(min_liquidity_rub=1.0),
            risk=RiskSection(),
            backtest=BacktestSection(initial_cash=100_000, warmup_bars=30),
            research=ResearchSection(
                strategy_styles=["ema_adx_macd"],
                fast_windows=[10],
                slow_windows=[30],
                require_breakout_values=[False],
                atr_stop_multipliers=[1.5],
                reward_to_risk_values=[2.0],
                breakeven_trigger_pct_values=[0.5],
                trend_strength_values=[0.002],
                adx_min_values=[15.0],
                subset_min_size=1,
                subset_max_size=1,
                top_n=3,
                min_trades=1,
            ),
            timeframe="hour",
            slippage_bps=0,
            commission_bps=0,
        )

        strategy = optimizer.strategy_from_candidate_payload(
            {
                "style": "ema_adx_macd",
                "fast_window": 10,
                "slow_window": 30,
                "require_breakout": False,
                "atr_stop_multiple": 1.5,
                "reward_to_risk": 2.0,
                "breakeven_trigger_pct": 0.5,
                "adx_min": 15.0,
                "min_trend_strength": 0.002,
            }
        )

        self.assertEqual(strategy.style, "ema_adx_macd")
        self.assertAlmostEqual(strategy.breakeven_trigger_pct, 0.5)

    def test_optimizer_can_evaluate_vwap_relvol_orb_style_candidates(self):
        instrument = Instrument(symbol="TEST", instrument_type=InstrumentType.STOCK, lot_size=1)
        candles = []
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        price = 100.0
        for index in range(180):
            close = price * (1.006 if index % 7 not in {0, 1} else 1.002)
            candles.append(
                Candle(
                    timestamp=ts,
                    open=price,
                    high=max(price, close) * 1.01,
                    low=min(price, close) * 0.995,
                    close=close,
                    volume=8_000_000 if index % 8 == 0 else 5_000_000,
                )
            )
            ts += timedelta(minutes=15)
            price = close

        optimizer = ParameterOptimizer(
            base_strategy=StrategySection(min_liquidity_rub=1.0),
            risk=RiskSection(),
            backtest=BacktestSection(initial_cash=100_000, warmup_bars=30),
            research=ResearchSection(
                strategy_styles=["vwap_relvol_orb"],
                fast_windows=[10],
                slow_windows=[30],
                require_breakout_values=[True],
                opening_range_bars_values=[2],
                rel_volume_threshold_values=[1.0],
                atr_stop_multipliers=[1.5],
                reward_to_risk_values=[2.0],
                trend_strength_values=[0.002],
                adx_min_values=[15.0],
                subset_min_size=1,
                subset_max_size=1,
                top_n=3,
                min_trades=1,
            ),
            timeframe="15min",
            slippage_bps=0,
            commission_bps=0,
        )

        payload = optimizer.run({"TEST": candles}, {"TEST": instrument})

        self.assertEqual(payload["evaluated_candidates"], 1)
        self.assertIsNotNone(payload["best_candidate"])
        self.assertEqual(payload["best_candidate"]["style"], "vwap_relvol_orb")
        self.assertEqual(payload["best_candidate"]["opening_range_bars"], 2)
        self.assertEqual(payload["best_candidate"]["rel_volume_threshold"], 1.0)


if __name__ == "__main__":
    unittest.main()
