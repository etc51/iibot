from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from samosbor.analysis.context import MarketBreadthContextProvider, StaticContextProvider
from samosbor.config import StrategySection
from samosbor.domain import Candle, Instrument, InstrumentType, SignalDirection
from samosbor.strategy.trend_following import TrendFollowingStrategy


def make_candles(*, trend: float, final_close: float | None = None) -> list[Candle]:
    candles: list[Candle] = []
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    price = 100.0
    for index in range(80):
        close = price + trend
        if final_close is not None and index == 79:
            close = final_close
        candles.append(
            Candle(
                timestamp=ts,
                open=price,
                high=max(price, close) + 0.5,
                low=min(price, close) - 0.5,
                close=close,
                volume=2_000_000,
            )
        )
        ts += timedelta(hours=1)
        price = close
    return candles


class TrendFollowingTATest(unittest.TestCase):
    def setUp(self):
        self.instrument = Instrument(symbol="CNYRUBF", instrument_type=InstrumentType.FUTURE, lot_size=1)

    def test_ema_adx_macd_can_open_long_without_breakout(self):
        candles = make_candles(trend=0.2, final_close=114.5)
        strategy = TrendFollowingStrategy(
            StrategySection(
                style="ema_adx_macd",
                require_breakout=False,
                min_liquidity_rub=1.0,
                min_trend_strength=0.002,
            ),
            timeframe="hour",
        )
        with patch.object(
            TrendFollowingStrategy,
            "_ta_features",
            return_value={
                "ema_fast": 113.0,
                "ema_slow": 110.0,
                "rsi": 61.0,
                "macd": 1.2,
                "macd_hist": 0.4,
                "macd_signal": 0.8,
                "adx": 24.0,
                "dmp": 27.0,
                "dmn": 15.0,
            },
        ):
            signal = strategy.generate_signal(self.instrument, candles)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, SignalDirection.LONG)
        self.assertGreater(signal.take_profit, signal.entry_price)

    def test_market_context_blocks_long_against_bearish_market(self):
        candles = make_candles(trend=0.2, final_close=114.5)
        strategy = TrendFollowingStrategy(
            StrategySection(
                style="ema_adx_macd",
                require_breakout=False,
                min_liquidity_rub=1.0,
                min_trend_strength=0.002,
                use_market_context=True,
                market_context_block_threshold=0.15,
            ),
            timeframe="hour",
            context_provider=StaticContextProvider({"CNYRUBF": -0.2}),
        )
        with patch.object(
            TrendFollowingStrategy,
            "_ta_features",
            return_value={
                "ema_fast": 113.0,
                "ema_slow": 110.0,
                "rsi": 61.0,
                "macd": 1.2,
                "macd_hist": 0.4,
                "macd_signal": 0.8,
                "adx": 24.0,
                "dmp": 27.0,
                "dmn": 15.0,
            },
        ):
            signal = strategy.generate_signal(self.instrument, candles)

        self.assertIsNone(signal)

    def test_market_context_blocks_short_against_bullish_market(self):
        candles = make_candles(trend=-0.2, final_close=84.5)
        strategy = TrendFollowingStrategy(
            StrategySection(
                style="ema_adx_macd",
                require_breakout=False,
                min_liquidity_rub=1.0,
                min_trend_strength=0.002,
                use_market_context=True,
                market_context_block_threshold=0.15,
            ),
            timeframe="hour",
            context_provider=StaticContextProvider({"CNYRUBF": 0.2}),
        )
        with patch.object(
            TrendFollowingStrategy,
            "_ta_features",
            return_value={
                "ema_fast": 84.0,
                "ema_slow": 88.0,
                "rsi": 39.0,
                "macd": -1.2,
                "macd_hist": -0.4,
                "macd_signal": -0.8,
                "adx": 24.0,
                "dmp": 15.0,
                "dmn": 27.0,
            },
        ):
            signal = strategy.generate_signal(self.instrument, candles)

        self.assertIsNone(signal)

    def test_market_breadth_context_provider_scores_bullish_market(self):
        histories = {f"SBER{i}": make_candles(trend=0.4) for i in range(8)}
        provider = MarketBreadthContextProvider(histories, min_symbols=8, max_score=0.25)

        score = provider.score(self.instrument, histories["SBER0"])

        self.assertGreater(score, 0.0)

    def test_ema_adx_macd_breakout_filter_can_block_long_entry(self):
        candles = make_candles(trend=0.2, final_close=114.5)
        strategy = TrendFollowingStrategy(
            StrategySection(
                style="ema_adx_macd",
                require_breakout=True,
                min_liquidity_rub=1.0,
                min_trend_strength=0.002,
            ),
            timeframe="hour",
        )
        with patch.object(
            TrendFollowingStrategy,
            "_ta_features",
            return_value={
                "ema_fast": 113.0,
                "ema_slow": 110.0,
                "rsi": 61.0,
                "macd": 1.2,
                "macd_hist": 0.4,
                "macd_signal": 0.8,
                "adx": 24.0,
                "dmp": 27.0,
                "dmn": 15.0,
            },
        ):
            signal = strategy.generate_signal(self.instrument, candles)

        self.assertIsNone(signal)

    def test_ema_adx_macd_can_open_short(self):
        candles = make_candles(trend=-0.2, final_close=85.5)
        strategy = TrendFollowingStrategy(
            StrategySection(
                style="ema_adx_macd",
                require_breakout=False,
                min_liquidity_rub=1.0,
                min_trend_strength=0.002,
            ),
            timeframe="hour",
        )
        with patch.object(
            TrendFollowingStrategy,
            "_ta_features",
            return_value={
                "ema_fast": 87.0,
                "ema_slow": 90.0,
                "rsi": 39.0,
                "macd": -1.2,
                "macd_hist": -0.4,
                "macd_signal": -0.8,
                "adx": 24.0,
                "dmp": 14.0,
                "dmn": 28.0,
            },
        ):
            signal = strategy.generate_signal(self.instrument, candles)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, SignalDirection.SHORT)
        self.assertLess(signal.take_profit, signal.entry_price)

    def test_ema_adx_donchian_can_open_long(self):
        candles = make_candles(trend=0.25, final_close=118.0)
        strategy = TrendFollowingStrategy(
            StrategySection(
                style="ema_adx_donchian",
                require_breakout=False,
                min_liquidity_rub=1.0,
                min_trend_strength=0.002,
            ),
            timeframe="hour",
        )
        with patch.object(
            TrendFollowingStrategy,
            "_ta_features",
            return_value={
                "ema_fast": 115.0,
                "ema_slow": 111.0,
                "rsi": 63.0,
                "adx": 26.0,
                "dmp": 29.0,
                "dmn": 14.0,
                "donchian_lower": 104.0,
                "donchian_upper": 117.0,
            },
        ):
            signal = strategy.generate_signal(self.instrument, candles)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, SignalDirection.LONG)
        self.assertGreater(signal.take_profit, signal.entry_price)

    def test_ema_adx_donchian_can_open_short(self):
        candles = make_candles(trend=-0.25, final_close=82.0)
        strategy = TrendFollowingStrategy(
            StrategySection(
                style="ema_adx_donchian",
                require_breakout=False,
                min_liquidity_rub=1.0,
                min_trend_strength=0.002,
            ),
            timeframe="hour",
        )
        with patch.object(
            TrendFollowingStrategy,
            "_ta_features",
            return_value={
                "ema_fast": 85.0,
                "ema_slow": 89.0,
                "rsi": 38.0,
                "adx": 26.0,
                "dmp": 13.0,
                "dmn": 30.0,
                "donchian_lower": 83.0,
                "donchian_upper": 96.0,
            },
        ):
            signal = strategy.generate_signal(self.instrument, candles)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, SignalDirection.SHORT)
        self.assertLess(signal.take_profit, signal.entry_price)

    def test_adx_regime_hybrid_can_open_trend_long(self):
        candles = make_candles(trend=0.25, final_close=118.0)
        strategy = TrendFollowingStrategy(
            StrategySection(
                style="adx_regime_hybrid",
                require_breakout=False,
                min_liquidity_rub=1.0,
                min_trend_strength=0.002,
            ),
            timeframe="hour",
        )
        with patch.object(
            TrendFollowingStrategy,
            "_ta_features",
            return_value={
                "ema_fast": 115.0,
                "ema_slow": 111.0,
                "rsi": 62.0,
                "macd": 1.3,
                "macd_hist": 0.5,
                "macd_signal": 0.8,
                "adx": 28.0,
                "dmp": 30.0,
                "dmn": 14.0,
                "donchian_lower": 104.0,
                "donchian_upper": 117.0,
            },
        ):
            signal = strategy.generate_signal(self.instrument, candles)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, SignalDirection.LONG)
        self.assertGreater(signal.take_profit, signal.entry_price)

    def test_adx_regime_hybrid_can_open_range_short(self):
        candles = make_candles(trend=0.8, final_close=162.0)
        strategy = TrendFollowingStrategy(
            StrategySection(
                style="adx_regime_hybrid",
                require_breakout=False,
                min_liquidity_rub=1.0,
                min_trend_strength=0.002,
                rsi_long_max=65.0,
            ),
            timeframe="hour",
        )
        with patch.object(
            TrendFollowingStrategy,
            "_ta_features",
            return_value={
                "ema_fast": 158.0,
                "ema_slow": 150.0,
                "rsi": 70.0,
                "macd": 0.2,
                "macd_hist": 0.05,
                "macd_signal": 0.15,
                "adx": 12.0,
                "dmp": 18.0,
                "dmn": 16.0,
                "donchian_lower": 140.0,
                "donchian_upper": 161.0,
            },
        ):
            signal = strategy.generate_signal(self.instrument, candles)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, SignalDirection.SHORT)
        self.assertLess(signal.take_profit, signal.entry_price)

    def test_rsi_mean_reversion_can_open_long(self):
        candles = make_candles(trend=-0.8, final_close=38.0)
        strategy = TrendFollowingStrategy(
            StrategySection(
                style="rsi_mean_reversion",
                min_liquidity_rub=1.0,
                min_trend_strength=0.002,
                rsi_short_min=35.0,
            ),
            timeframe="hour",
        )

        signal = strategy.generate_signal(self.instrument, candles)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, SignalDirection.LONG)
        self.assertGreater(signal.take_profit, signal.entry_price)

    def test_rsi_mean_reversion_can_open_short(self):
        candles = make_candles(trend=0.8, final_close=162.0)
        strategy = TrendFollowingStrategy(
            StrategySection(
                style="rsi_mean_reversion",
                min_liquidity_rub=1.0,
                min_trend_strength=0.002,
                rsi_long_max=65.0,
            ),
            timeframe="hour",
        )

        signal = strategy.generate_signal(self.instrument, candles)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, SignalDirection.SHORT)
        self.assertLess(signal.take_profit, signal.entry_price)

    def test_vwap_relvol_orb_can_open_long(self):
        candles = make_candles(trend=0.2, final_close=115.5)
        strategy = TrendFollowingStrategy(
            StrategySection(
                style="vwap_relvol_orb",
                require_breakout=True,
                opening_range_bars=2,
                rel_volume_threshold=1.05,
                min_liquidity_rub=1.0,
                min_trend_strength=0.002,
            ),
            timeframe="15min",
        )
        with patch.object(
            TrendFollowingStrategy,
            "_ta_features",
            return_value={
                "ema_fast": 114.2,
                "ema_slow": 112.0,
                "rsi": 61.0,
                "adx": 21.0,
                "dmp": 25.0,
                "dmn": 14.0,
                "session_vwap": 113.9,
                "relative_volume": 1.35,
                "opening_range_high": 115.0,
                "opening_range_low": 110.5,
                "session_bar_index": 4.0,
            },
        ):
            signal = strategy.generate_signal(self.instrument, candles)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, SignalDirection.LONG)
        self.assertGreater(signal.take_profit, signal.entry_price)

    def test_vwap_relvol_orb_can_open_short(self):
        candles = make_candles(trend=-0.2, final_close=84.5)
        strategy = TrendFollowingStrategy(
            StrategySection(
                style="vwap_relvol_orb",
                require_breakout=True,
                opening_range_bars=2,
                rel_volume_threshold=1.05,
                min_liquidity_rub=1.0,
                min_trend_strength=0.002,
            ),
            timeframe="15min",
        )
        with patch.object(
            TrendFollowingStrategy,
            "_ta_features",
            return_value={
                "ema_fast": 86.0,
                "ema_slow": 88.8,
                "rsi": 39.0,
                "adx": 21.0,
                "dmp": 14.0,
                "dmn": 26.0,
                "session_vwap": 85.3,
                "relative_volume": 1.4,
                "opening_range_high": 89.0,
                "opening_range_low": 84.8,
                "session_bar_index": 5.0,
            },
        ):
            signal = strategy.generate_signal(self.instrument, candles)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, SignalDirection.SHORT)
        self.assertLess(signal.take_profit, signal.entry_price)

    def test_min_signal_strength_can_block_entry(self):
        candles = make_candles(trend=0.2, final_close=114.5)
        strategy = TrendFollowingStrategy(
            StrategySection(
                style="ema_adx_macd",
                require_breakout=False,
                min_liquidity_rub=1.0,
                min_trend_strength=0.002,
                min_signal_strength=1.01,
            ),
            timeframe="hour",
        )
        with patch.object(
            TrendFollowingStrategy,
            "_ta_features",
            return_value={
                "ema_fast": 113.0,
                "ema_slow": 110.0,
                "rsi": 61.0,
                "macd": 1.2,
                "macd_hist": 0.4,
                "macd_signal": 0.8,
                "adx": 24.0,
                "dmp": 27.0,
                "dmn": 15.0,
            },
        ):
            signal = strategy.generate_signal(self.instrument, candles)

        self.assertIsNone(signal)

    def test_unknown_strategy_style_raises(self):
        with self.assertRaises(ValueError):
            TrendFollowingStrategy(
                StrategySection(style="unknown-style"),
                timeframe="hour",
            )

    def test_entry_schedule_uses_moscow_hours(self):
        strategy = TrendFollowingStrategy(
            StrategySection(
                allowed_entry_hours=[12],
                allowed_entry_weekdays=[2],
                schedule_timezone="Europe/Moscow",
            ),
            timeframe="hour",
        )

        allowed_timestamp = datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc)
        blocked_timestamp = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)

        self.assertTrue(strategy.allows_entry_at(allowed_timestamp))
        self.assertFalse(strategy.allows_entry_at(blocked_timestamp))

    def test_session_flat_window_blocks_new_entries_and_matches_moscow_hour(self):
        strategy = TrendFollowingStrategy(
            StrategySection(
                allowed_entry_hours=[10, 11, 12, 13, 14, 15, 16, 17],
                allowed_entry_weekdays=[0, 1, 2, 3, 4],
                forced_flat_hours=[18, 19, 20, 21, 22, 23],
                forced_flat_weekdays=[0, 1, 2, 3, 4],
                schedule_timezone="Europe/Moscow",
            ),
            timeframe="30min",
        )

        entry_timestamp = datetime(2025, 1, 1, 14, 0, tzinfo=timezone.utc)
        forced_flat_timestamp = datetime(2025, 1, 1, 15, 0, tzinfo=timezone.utc)

        self.assertTrue(strategy.allows_entry_at(entry_timestamp))
        self.assertTrue(strategy.should_force_flatten_at(forced_flat_timestamp))
        self.assertFalse(strategy.allows_entry_at(forced_flat_timestamp))
        self.assertEqual(
            strategy.entry_block_reason_at(forced_flat_timestamp),
            "entry blocked by session-flat window",
        )

    def test_intraday_session_labels_ignore_off_hours_and_restart_each_day(self):
        strategy = TrendFollowingStrategy(
            StrategySection(
                allowed_entry_hours=[10, 11],
                allowed_entry_weekdays=[2, 3],
                forced_flat_hours=[18, 19],
                forced_flat_weekdays=[2, 3],
                schedule_timezone="Europe/Moscow",
            ),
            timeframe="15min",
        )

        labels, indexes = strategy._intraday_session_labels(
            [
                datetime(2025, 1, 1, 6, 0, tzinfo=timezone.utc),   # 09:00 MSK off-hours
                datetime(2025, 1, 1, 7, 0, tzinfo=timezone.utc),   # 10:00 MSK first active
                datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc),   # 11:00 MSK second active
                datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc),   # 12:00 MSK off-hours
                datetime(2025, 1, 2, 7, 0, tzinfo=timezone.utc),   # next day first active
            ]
        )

        self.assertEqual(labels[0], None)
        self.assertEqual(indexes[0], None)
        self.assertEqual(indexes[1], 0)
        self.assertEqual(indexes[2], 1)
        self.assertEqual(labels[1], labels[2])
        self.assertEqual(labels[3], None)
        self.assertEqual(indexes[4], 0)
        self.assertNotEqual(labels[1], labels[4])

    def test_symbol_restriction_blocks_entry_for_specific_ticker(self):
        strategy = TrendFollowingStrategy(
            StrategySection(
                blocked_symbols=["CNYRUBF"],
                allowed_entry_hours=[12],
                allowed_entry_weekdays=[2],
                schedule_timezone="Europe/Moscow",
            ),
            timeframe="hour",
        )

        blocked_timestamp = datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc)

        self.assertEqual(
            strategy.entry_block_reason_for_instrument(self.instrument, blocked_timestamp),
            "entry blocked by symbol restriction (CNYRUBF)",
        )

    def test_directional_symbol_restrictions_can_block_only_short_entries(self):
        strategy = TrendFollowingStrategy(
            StrategySection(
                blocked_short_symbols=["CNYRUBF"],
                allowed_entry_hours=[12],
                allowed_entry_weekdays=[2],
                schedule_timezone="Europe/Moscow",
            ),
            timeframe="hour",
        )

        allowed_timestamp = datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc)

        self.assertIsNone(
            strategy.entry_block_reason_for_instrument(
                self.instrument,
                allowed_timestamp,
                SignalDirection.LONG,
            )
        )
        self.assertEqual(
            strategy.entry_block_reason_for_instrument(
                self.instrument,
                allowed_timestamp,
                SignalDirection.SHORT,
            ),
            "entry blocked by short restriction (CNYRUBF)",
        )

    def test_allowed_symbols_can_limit_runtime_universe(self):
        strategy = TrendFollowingStrategy(
            StrategySection(
                allowed_symbols=["USDRUBF"],
                allowed_entry_hours=[12],
                allowed_entry_weekdays=[2],
                schedule_timezone="Europe/Moscow",
            ),
            timeframe="hour",
        )

        allowed_timestamp = datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc)

        self.assertEqual(
            strategy.entry_block_reason_for_instrument(
                self.instrument,
                allowed_timestamp,
                SignalDirection.LONG,
            ),
            "entry blocked by allowed universe restriction (CNYRUBF)",
        )


if __name__ == "__main__":
    unittest.main()
