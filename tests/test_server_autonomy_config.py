from __future__ import annotations

import unittest

from samosbor.server_autonomy_config import (
    DAILY_AUTONOMY_RESEARCH_ARRAY_LIMITS,
    build_offline_autonomy_config_text,
)


class ServerAutonomyConfigTest(unittest.TestCase):
    def test_build_offline_autonomy_config_rewrites_data_source_and_path(self):
        source = "\n".join(
            [
                "[app]",
                'timezone = "Europe/Moscow"',
                "",
                "[data]",
                'source = "tbank"',
                'timeframe = "30min"',
                "history_days = 120",
                'csv_path = "data/demo.csv"',
                "",
                "[strategy]",
                'style = "ema_adx_macd"',
                "",
            ]
        )
        rendered = build_offline_autonomy_config_text(
            source,
            parquet_dir_path="data/server_moex_strategy_lab_data_processed",
        )

        self.assertIn('source = "parquet-directory"', rendered)
        self.assertIn(
            'parquet_dir_path = "data/server_moex_strategy_lab_data_processed"',
            rendered,
        )
        self.assertNotIn('source = "tbank"', rendered)
        self.assertNotIn('csv_path = "data/demo.csv"', rendered)
        self.assertIn('[strategy]\nstyle = "ema_adx_macd"', rendered)

    def test_build_offline_autonomy_config_accepts_utf8_bom(self):
        rendered = build_offline_autonomy_config_text(
            '\ufeff[data]\nsource = "tbank"\n',
            parquet_dir_path="data/offline",
        )

        self.assertTrue(rendered.startswith("[data]\n"))
        self.assertIn('source = "parquet-directory"', rendered)

    def test_build_offline_autonomy_config_reuses_existing_parquet_key(self):
        source = "\n".join(
            [
                "[data]",
                'source = "csv"',
                'parquet_dir_path = "old/path"',
                'local_data_pack_path = "unused"',
                "",
                "[execution]",
                'mode = "local-paper"',
                "",
            ]
        )

        rendered = build_offline_autonomy_config_text(
            source,
            parquet_dir_path="data/new_path",
        )

        self.assertIn('source = "parquet-directory"', rendered)
        self.assertIn('parquet_dir_path = "data/new_path"', rendered)
        self.assertNotIn('parquet_dir_path = "old/path"', rendered)
        self.assertNotIn('local_data_pack_path = "unused"', rendered)

    def test_build_offline_autonomy_config_limits_research_grid_for_wide_universe(self):
        source_lines = [
            "[data]",
            'source = "tbank"',
            "history_days = 30",
            "",
        ]
        for symbol in ["SBER", "GAZP", "LKOH", "NVTK", "ROSN", "TATN", "VTBR", "CHMF"]:
            source_lines.extend(
                [
                    "[[data.instruments]]",
                    f'symbol = "{symbol}"',
                    'instrument_type = "stock"',
                    "",
                ]
            )
        source_lines.extend(
            [
                "[strategy]",
                'style = "ema_adx_donchian"',
                "fast_window = 20",
                "slow_window = 50",
                "require_breakout = false",
                "atr_stop_multiple = 1.5",
                "reward_to_risk = 2.0",
                "breakeven_trigger_pct = 0.5",
                "trailing_profit_trigger_rub = 1200.0",
                "trailing_profit_lock_ratio = 0.5",
                "min_trend_strength = 0.002",
                "adx_min = 20.0",
                "rsi_long_max = 75.0",
                "rsi_short_min = 25.0",
                "",
                "[risk]",
                "max_positions = 6",
                "",
                "[research]",
                'strategy_styles = ["ema_adx_macd", "ema_adx_donchian", "adx_regime_hybrid"]',
                "fast_windows = [10, 15, 20]",
                "slow_windows = [30, 40, 50]",
                "require_breakout_values = [false, true]",
                "opening_range_bars_values = [2, 3]",
                "rel_volume_threshold_values = [1.0, 1.15, 1.3]",
                "atr_stop_multipliers = [1.25, 1.5]",
                "reward_to_risk_values = [1.5, 2.0]",
                "breakeven_trigger_pct_values = [0.5]",
                "trailing_profit_trigger_rub_values = [0.0, 1200.0]",
                "trailing_profit_lock_ratio_values = [0.5]",
                "trend_strength_values = [0.002, 0.004]",
                "adx_min_values = [15.0, 20.0]",
                "rsi_long_max_values = [70.0, 75.0]",
                "rsi_short_min_values = [25.0, 30.0]",
                "subset_min_size = 1",
                "subset_max_size = 2",
                "top_n = 12",
                "min_trades = 12",
                "walk_forward_train_months = 4",
                "",
            ]
        )
        source = "\n".join(source_lines)

        rendered = build_offline_autonomy_config_text(
            source,
            parquet_dir_path="data/server_moex_strategy_lab_data_processed",
        )

        self.assertIn('strategy_styles = ["ema_adx_donchian", "ema_adx_macd", "adx_regime_hybrid"]', rendered)
        self.assertIn("fast_windows = [20, 10]", rendered)
        self.assertIn("slow_windows = [50, 30]", rendered)
        self.assertIn("require_breakout_values = [false, true]", rendered)
        self.assertIn("opening_range_bars_values = [2, 3]", rendered)
        self.assertIn("rel_volume_threshold_values = [1.1, 1.0]", rendered)
        self.assertIn("atr_stop_multipliers = [1.5]", rendered)
        self.assertIn("reward_to_risk_values = [2.0, 1.5]", rendered)
        self.assertIn("breakeven_trigger_pct_values = [0.5]", rendered)
        self.assertIn("trailing_profit_trigger_rub_values = [1200.0, 0.0]", rendered)
        self.assertIn("trailing_profit_lock_ratio_values = [0.5]", rendered)
        self.assertIn("trend_strength_values = [0.002, 0.004]", rendered)
        self.assertIn("adx_min_values = [20.0, 15.0]", rendered)
        self.assertIn("rsi_long_max_values = [75.0, 70.0]", rendered)
        self.assertIn("rsi_short_min_values = [25.0, 30.0]", rendered)
        self.assertIn("subset_min_size = 8", rendered)
        self.assertIn("subset_max_size = 8", rendered)
        self.assertIn("top_n = 8", rendered)
        self.assertIn("min_trades = 12", rendered)
        self.assertIn("walk_forward_train_months = 4", rendered)

    def test_build_offline_autonomy_config_daily_profile_keeps_grid_small(self):
        source = "\n".join(
            [
                "[data]",
                'source = "tbank"',
                "",
                "[[data.instruments]]",
                'symbol = "SBER"',
                'instrument_type = "stock"',
                "",
                "[strategy]",
                'style = "ema_adx_macd"',
                "fast_window = 20",
                "slow_window = 50",
                "require_breakout = false",
                "reward_to_risk = 2.5",
                "breakeven_trigger_pct = 0.75",
                "trailing_profit_trigger_rub = 1200.0",
                "trailing_profit_lock_ratio = 0.35",
                "min_trend_strength = 0.002",
                "adx_min = 20.0",
                "",
                "[research]",
                'strategy_styles = ["ema_adx_macd", "ema_adx_donchian"]',
                "fast_windows = [10, 20]",
                "slow_windows = [30, 50]",
                "require_breakout_values = [false, true]",
                "reward_to_risk_values = [2.0, 2.5]",
                "breakeven_trigger_pct_values = [0.0, 0.75]",
                "trailing_profit_trigger_rub_values = [0.0, 1200.0]",
                "trailing_profit_lock_ratio_values = [0.2, 0.35]",
                "trend_strength_values = [0.0015, 0.002]",
                "adx_min_values = [15.0, 20.0]",
                "",
            ]
        )

        rendered = build_offline_autonomy_config_text(
            source,
            parquet_dir_path="data/offline",
            research_array_limits=DAILY_AUTONOMY_RESEARCH_ARRAY_LIMITS,
        )

        self.assertIn('strategy_styles = ["ema_adx_macd", "ema_adx_donchian"]', rendered)
        self.assertIn("fast_windows = [20, 10]", rendered)
        self.assertIn("slow_windows = [50, 30]", rendered)
        self.assertIn("require_breakout_values = [false]", rendered)
        self.assertIn("reward_to_risk_values = [2.5, 2.0]", rendered)
        self.assertIn("breakeven_trigger_pct_values = [0.75]", rendered)
        self.assertIn("trailing_profit_trigger_rub_values = [1200.0]", rendered)
        self.assertIn("trailing_profit_lock_ratio_values = [0.35]", rendered)
        self.assertIn("trend_strength_values = [0.002]", rendered)
        self.assertIn("adx_min_values = [20.0]", rendered)

    def test_build_offline_autonomy_config_injects_research_grid_when_blank(self):
        source = "\n".join(
            [
                "[data]",
                'source = "tbank"',
                "",
                "[[data.instruments]]",
                'symbol = "CNYRUBF"',
                'instrument_type = "future"',
                "",
                "[[data.instruments]]",
                'symbol = "USDRUBF"',
                'instrument_type = "future"',
                "",
                "[strategy]",
                'style = "ema_adx_macd"',
                "fast_window = 10",
                "slow_window = 40",
                "require_breakout = false",
                "atr_stop_multiple = 1.5",
                "reward_to_risk = 2.0",
                "breakeven_trigger_pct = 0.5",
                "trailing_profit_trigger_rub = 1200.0",
                "trailing_profit_lock_ratio = 0.5",
                "min_trend_strength = 0.002",
                "adx_min = 20.0",
                "rsi_long_max = 75.0",
                "rsi_short_min = 25.0",
                "",
                "[research]",
                "",
            ]
        )

        rendered = build_offline_autonomy_config_text(
            source,
            parquet_dir_path="data/server_moex_strategy_lab_data_processed",
        )

        self.assertIn('[research]\nstrategy_styles = ["ema_adx_macd"]', rendered)
        self.assertIn("fast_windows = [10]", rendered)
        self.assertIn("slow_windows = [40]", rendered)
        self.assertIn("require_breakout_values = [false]", rendered)
        self.assertIn("subset_min_size = 2", rendered)
        self.assertIn("subset_max_size = 2", rendered)
        self.assertIn("top_n = 8", rendered)

    def test_build_offline_autonomy_config_requires_data_section(self):
        with self.assertRaises(ValueError):
            build_offline_autonomy_config_text(
                "[app]\nname = \"samosbor\"\n",
                parquet_dir_path="data/new_path",
            )

    def test_build_offline_autonomy_config_uses_full_runtime_universe_for_narrow_universe_too(self):
        source = "\n".join(
            [
                "[data]",
                'source = "tbank"',
                'timeframe = "15min"',
                "",
                "[[data.instruments]]",
                'symbol = "SBER"',
                'instrument_type = "stock"',
                "",
                "[[data.instruments]]",
                'symbol = "GAZP"',
                'instrument_type = "stock"',
                "",
                "[[data.instruments]]",
                'symbol = "LKOH"',
                'instrument_type = "stock"',
                "",
                "[strategy]",
                'style = "ema_adx_macd"',
                "fast_window = 20",
                "slow_window = 50",
                "atr_stop_multiple = 1.5",
                "reward_to_risk = 2.0",
                "breakeven_trigger_pct = 0.5",
                "trailing_profit_trigger_rub = 1200.0",
                "trailing_profit_lock_ratio = 0.5",
                "min_trend_strength = 0.002",
                "adx_min = 20.0",
                "rsi_long_max = 75.0",
                "rsi_short_min = 25.0",
                "",
                "[research]",
                'strategy_styles = ["ema_adx_macd", "adx_regime_hybrid", "rsi_mean_reversion"]',
                "fast_windows = [10, 20]",
                "slow_windows = [30, 50, 70]",
                "reward_to_risk_values = [1.5, 2.0]",
                "subset_min_size = 1",
                "subset_max_size = 2",
                "top_n = 10",
                "",
            ]
        )

        rendered = build_offline_autonomy_config_text(
            source,
            parquet_dir_path="data/new_path",
        )

        self.assertIn("subset_min_size = 3", rendered)
        self.assertIn("subset_max_size = 3", rendered)
        self.assertIn('strategy_styles = ["ema_adx_macd", "adx_regime_hybrid", "rsi_mean_reversion"]', rendered)
        self.assertIn("top_n = 8", rendered)

    def test_build_offline_autonomy_config_preserves_subset_search_for_offline_sources(self):
        source = "\n".join(
            [
                "[data]",
                'source = "parquet-directory"',
                'parquet_dir_path = "data/cache"',
                "",
                "[[data.instruments]]",
                'symbol = "SBER"',
                'instrument_type = "stock"',
                "",
                "[[data.instruments]]",
                'symbol = "GAZP"',
                'instrument_type = "stock"',
                "",
                "[[data.instruments]]",
                'symbol = "LKOH"',
                'instrument_type = "stock"',
                "",
                "[[data.instruments]]",
                'symbol = "NVTK"',
                'instrument_type = "stock"',
                "",
                "[strategy]",
                'style = "ema_adx_macd"',
                "fast_window = 20",
                "slow_window = 50",
                "reward_to_risk = 2.0",
                "",
                "[research]",
                "subset_min_size = 1",
                "subset_max_size = 2",
                "top_n = 10",
                "",
            ]
        )

        rendered = build_offline_autonomy_config_text(
            source,
            parquet_dir_path="data/new_path",
        )

        self.assertIn("subset_min_size = 1", rendered)
        self.assertIn("subset_max_size = 2", rendered)
        self.assertIn("top_n = 8", rendered)


if __name__ == "__main__":
    unittest.main()
