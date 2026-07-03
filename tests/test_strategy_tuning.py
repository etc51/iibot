from __future__ import annotations

import unittest

from samosbor.autonomy.strategy_tuning import (
    adapt_strategy_tuning_research,
    build_strategy_tuning_payload,
)
from samosbor.config import BacktestSection, ResearchSection, StrategySection
from samosbor.research.targets import (
    effective_target_daily_profit_rub,
    effective_target_monthly_profit_rub,
    effective_target_monthly_return_pct,
)


class ResearchTargetTest(unittest.TestCase):
    def test_effective_target_prefers_profit_goal_when_present(self):
        backtest = BacktestSection(initial_cash=1_000_000)
        research = ResearchSection(target_monthly_return_pct=5.0, target_monthly_profit_rub=7_500.0)

        self.assertEqual(effective_target_monthly_profit_rub(research, backtest), 7_500.0)
        self.assertEqual(effective_target_monthly_return_pct(research, backtest), 0.75)

    def test_effective_target_prefers_daily_goal_when_present(self):
        backtest = BacktestSection(initial_cash=300_000)
        research = ResearchSection(
            trading_days_per_month=20,
            target_daily_profit_rub=3_000.0,
            target_monthly_return_pct=1.0,
            target_monthly_profit_rub=3_000.0,
        )

        self.assertEqual(effective_target_daily_profit_rub(research, backtest), 3_000.0)
        self.assertEqual(effective_target_monthly_profit_rub(research, backtest), 60_000.0)
        self.assertEqual(effective_target_monthly_return_pct(research, backtest), 20.0)


class StrategyTuningTest(unittest.TestCase):
    def test_adaptive_research_shrinks_walk_forward_window_to_fit_history(self):
        research = ResearchSection(
            walk_forward_train_months=6,
            walk_forward_test_months=1,
            walk_forward_step_months=1,
            subset_min_size=1,
            subset_max_size=2,
        )

        adjusted, meta = adapt_strategy_tuning_research(
            research,
            available_months=4,
            fixed_subset_size=1,
        )

        self.assertIsNotNone(adjusted)
        self.assertTrue(meta["history_was_adapted"])
        self.assertEqual(adjusted.walk_forward_train_months, 3)
        self.assertEqual(adjusted.walk_forward_test_months, 1)
        self.assertEqual(adjusted.subset_min_size, 1)
        self.assertEqual(adjusted.subset_max_size, 1)

    def test_strategy_tuning_accepts_candidate_when_guardrails_pass(self):
        current = StrategySection(
            style="ema_adx_macd",
            fast_window=10,
            slow_window=40,
            require_breakout=False,
            atr_stop_multiple=1.5,
            reward_to_risk=2.0,
            min_trend_strength=0.002,
            adx_min=20.0,
        )
        candidate = StrategySection(
            style="ema_adx_macd",
            fast_window=10,
            slow_window=40,
            require_breakout=True,
            atr_stop_multiple=1.5,
            reward_to_risk=2.0,
            min_trend_strength=0.002,
            adx_min=15.0,
        )
        payload = build_strategy_tuning_payload(
            current_strategy=current,
            candidate_strategy=candidate,
            baseline_latest_test_summary={
                "total_return_pct": 0.4,
                "normalized_monthly_return_pct": 0.4,
                "max_drawdown_pct": 1.8,
                "sharpe_ratio": 0.3,
            },
            candidate_latest_test_summary={
                "total_return_pct": 1.1,
                "normalized_monthly_return_pct": 1.1,
                "max_drawdown_pct": 2.1,
                "sharpe_ratio": 0.7,
            },
            walk_forward_summary={
                "average_test_normalized_monthly_return_pct": 0.8,
                "probability_positive_pct": 66.7,
            },
            walk_forward_config={
                "train_months": 3,
                "test_months": 1,
                "step_months": 1,
            },
            backtest=BacktestSection(initial_cash=1_000_000),
            research=ResearchSection(target_monthly_profit_rub=7_500.0),
            research_window={"available_months": 4, "usable": True},
        )

        self.assertTrue(payload["changed"])
        self.assertEqual(payload["target"]["daily_profit_rub"], 375.0)
        self.assertEqual(payload["patch_values"]["require_breakout"], True)
        self.assertEqual(payload["patch_values"]["adx_min"], 15.0)

    def test_strategy_tuning_rejects_candidate_when_drawdown_deteriorates_too_much(self):
        current = StrategySection(style="ema_adx_macd")
        candidate = StrategySection(style="ema_adx_macd", reward_to_risk=1.5)
        payload = build_strategy_tuning_payload(
            current_strategy=current,
            candidate_strategy=candidate,
            baseline_latest_test_summary={
                "total_return_pct": 0.6,
                "normalized_monthly_return_pct": 0.6,
                "max_drawdown_pct": 1.0,
                "sharpe_ratio": 0.5,
            },
            candidate_latest_test_summary={
                "total_return_pct": 1.2,
                "normalized_monthly_return_pct": 1.2,
                "max_drawdown_pct": 3.5,
                "sharpe_ratio": 0.8,
            },
            walk_forward_summary={
                "average_test_normalized_monthly_return_pct": 0.9,
                "probability_positive_pct": 75.0,
            },
            walk_forward_config={
                "train_months": 3,
                "test_months": 1,
                "step_months": 1,
            },
            backtest=BacktestSection(initial_cash=1_000_000),
            research=ResearchSection(target_monthly_profit_rub=7_500.0),
            research_window={"available_months": 4, "usable": True},
        )

        self.assertFalse(payload["changed"])
        self.assertEqual(payload["reason"], "candidate failed one or more safety guardrails")

    def test_strategy_tuning_can_apply_style_switch_with_rsi_thresholds(self):
        current = StrategySection(
            style="ema_adx_macd",
            fast_window=10,
            slow_window=40,
            require_breakout=False,
            min_trend_strength=0.002,
            adx_min=20.0,
            rsi_long_max=75.0,
            rsi_short_min=25.0,
        )
        candidate = StrategySection(
            style="rsi_mean_reversion",
            fast_window=12,
            slow_window=36,
            require_breakout=False,
            min_trend_strength=0.003,
            adx_min=20.0,
            rsi_long_max=68.0,
            rsi_short_min=32.0,
        )
        payload = build_strategy_tuning_payload(
            current_strategy=current,
            candidate_strategy=candidate,
            baseline_latest_test_summary={
                "total_return_pct": 0.4,
                "normalized_monthly_return_pct": 0.4,
                "max_drawdown_pct": 1.5,
                "sharpe_ratio": 0.3,
            },
            candidate_latest_test_summary={
                "total_return_pct": 1.0,
                "normalized_monthly_return_pct": 1.0,
                "max_drawdown_pct": 1.7,
                "sharpe_ratio": 0.8,
            },
            walk_forward_summary={
                "average_test_normalized_monthly_return_pct": 0.7,
                "probability_positive_pct": 66.7,
            },
            walk_forward_config={
                "train_months": 3,
                "test_months": 1,
                "step_months": 1,
            },
            backtest=BacktestSection(initial_cash=300_000),
            research=ResearchSection(target_daily_profit_rub=3_000.0),
            research_window={"available_months": 4, "usable": True},
        )

        self.assertTrue(payload["changed"])
        self.assertEqual(payload["patch_values"]["style"], "rsi_mean_reversion")
        self.assertEqual(payload["patch_values"]["rsi_long_max"], 68.0)
        self.assertEqual(payload["patch_values"]["rsi_short_min"], 32.0)

    def test_strategy_tuning_can_apply_style_switch_to_donchian_trend(self):
        current = StrategySection(
            style="ema_adx_macd",
            fast_window=10,
            slow_window=40,
            require_breakout=False,
            min_trend_strength=0.002,
            adx_min=20.0,
        )
        candidate = StrategySection(
            style="ema_adx_donchian",
            fast_window=12,
            slow_window=36,
            require_breakout=False,
            min_trend_strength=0.003,
            adx_min=18.0,
        )
        payload = build_strategy_tuning_payload(
            current_strategy=current,
            candidate_strategy=candidate,
            baseline_latest_test_summary={
                "total_return_pct": 0.4,
                "normalized_monthly_return_pct": 0.4,
                "max_drawdown_pct": 1.5,
                "sharpe_ratio": 0.3,
            },
            candidate_latest_test_summary={
                "total_return_pct": 1.0,
                "normalized_monthly_return_pct": 1.0,
                "max_drawdown_pct": 1.9,
                "sharpe_ratio": 0.8,
            },
            walk_forward_summary={
                "average_test_normalized_monthly_return_pct": 0.7,
                "probability_positive_pct": 66.7,
            },
            walk_forward_config={
                "train_months": 3,
                "test_months": 1,
                "step_months": 1,
            },
            backtest=BacktestSection(initial_cash=300_000),
            research=ResearchSection(target_daily_profit_rub=3_000.0),
            research_window={"available_months": 4, "usable": True},
        )

        self.assertTrue(payload["changed"])
        self.assertEqual(payload["patch_values"]["style"], "ema_adx_donchian")
        self.assertEqual(payload["patch_values"]["adx_min"], 18.0)

    def test_strategy_tuning_can_apply_style_switch_to_hybrid_regime(self):
        current = StrategySection(
            style="ema_adx_macd",
            fast_window=10,
            slow_window=40,
            require_breakout=False,
            min_trend_strength=0.002,
            adx_min=20.0,
            rsi_long_max=75.0,
            rsi_short_min=25.0,
        )
        candidate = StrategySection(
            style="adx_regime_hybrid",
            fast_window=12,
            slow_window=36,
            require_breakout=False,
            min_trend_strength=0.003,
            adx_min=18.0,
            rsi_long_max=68.0,
            rsi_short_min=32.0,
        )
        payload = build_strategy_tuning_payload(
            current_strategy=current,
            candidate_strategy=candidate,
            baseline_latest_test_summary={
                "total_return_pct": 0.4,
                "normalized_monthly_return_pct": 0.4,
                "max_drawdown_pct": 1.5,
                "sharpe_ratio": 0.3,
            },
            candidate_latest_test_summary={
                "total_return_pct": 1.0,
                "normalized_monthly_return_pct": 1.0,
                "max_drawdown_pct": 1.8,
                "sharpe_ratio": 0.8,
            },
            walk_forward_summary={
                "average_test_normalized_monthly_return_pct": 0.7,
                "probability_positive_pct": 66.7,
            },
            walk_forward_config={
                "train_months": 3,
                "test_months": 1,
                "step_months": 1,
            },
            backtest=BacktestSection(initial_cash=300_000),
            research=ResearchSection(target_daily_profit_rub=3_000.0),
            research_window={"available_months": 4, "usable": True},
        )

        self.assertTrue(payload["changed"])
        self.assertEqual(payload["patch_values"]["style"], "adx_regime_hybrid")
        self.assertEqual(payload["patch_values"]["adx_min"], 18.0)
        self.assertEqual(payload["patch_values"]["rsi_long_max"], 68.0)

    def test_strategy_tuning_can_apply_style_switch_to_vwap_relvol_orb(self):
        current = StrategySection(
            style="ema_adx_macd",
            fast_window=10,
            slow_window=40,
            require_breakout=False,
            min_trend_strength=0.002,
            adx_min=20.0,
        )
        candidate = StrategySection(
            style="vwap_relvol_orb",
            fast_window=10,
            slow_window=30,
            require_breakout=True,
            opening_range_bars=3,
            rel_volume_threshold=1.2,
            min_trend_strength=0.002,
            adx_min=15.0,
        )
        payload = build_strategy_tuning_payload(
            current_strategy=current,
            candidate_strategy=candidate,
            baseline_latest_test_summary={
                "normalized_monthly_return_pct": -0.5,
                "total_return_pct": -0.5,
                "max_drawdown_pct": 1.0,
                "sharpe_ratio": -0.2,
            },
            candidate_latest_test_summary={
                "normalized_monthly_return_pct": 0.2,
                "total_return_pct": 0.4,
                "max_drawdown_pct": 1.4,
                "sharpe_ratio": 0.1,
            },
            walk_forward_summary={
                "average_test_normalized_monthly_return_pct": 0.15,
                "probability_positive_pct": 65.0,
            },
            walk_forward_config={"folds_evaluated": 4},
            backtest=BacktestSection(initial_cash=100_000),
            research=ResearchSection(target_monthly_return_pct=5.0),
            research_window={"usable": True},
        )

        self.assertTrue(payload["changed"])
        self.assertEqual(payload["patch_values"]["style"], "vwap_relvol_orb")
        self.assertEqual(payload["patch_values"]["opening_range_bars"], 3)
        self.assertEqual(payload["patch_values"]["rel_volume_threshold"], 1.2)

    def test_strategy_tuning_can_apply_trailing_profit_change(self):
        current = StrategySection(
            style="ema_adx_macd",
            fast_window=10,
            slow_window=40,
            require_breakout=False,
            trailing_profit_trigger_rub=0.0,
            trailing_profit_lock_ratio=0.0,
            min_trend_strength=0.002,
            adx_min=20.0,
        )
        candidate = StrategySection(
            style="ema_adx_macd",
            fast_window=10,
            slow_window=40,
            require_breakout=False,
            trailing_profit_trigger_rub=1_200.0,
            trailing_profit_lock_ratio=0.5,
            min_trend_strength=0.002,
            adx_min=20.0,
        )
        payload = build_strategy_tuning_payload(
            current_strategy=current,
            candidate_strategy=candidate,
            baseline_latest_test_summary={
                "total_return_pct": 0.4,
                "normalized_monthly_return_pct": 0.4,
                "max_drawdown_pct": 1.5,
                "sharpe_ratio": 0.3,
            },
            candidate_latest_test_summary={
                "total_return_pct": 1.0,
                "normalized_monthly_return_pct": 1.0,
                "max_drawdown_pct": 1.8,
                "sharpe_ratio": 0.8,
            },
            walk_forward_summary={
                "average_test_normalized_monthly_return_pct": 0.7,
                "probability_positive_pct": 66.7,
            },
            walk_forward_config={
                "train_months": 3,
                "test_months": 1,
                "step_months": 1,
            },
            backtest=BacktestSection(initial_cash=300_000),
            research=ResearchSection(target_daily_profit_rub=3_000.0),
            research_window={"available_months": 4, "usable": True},
        )

        self.assertTrue(payload["changed"])
        self.assertEqual(payload["patch_values"]["trailing_profit_trigger_rub"], 1_200.0)
        self.assertEqual(payload["patch_values"]["trailing_profit_lock_ratio"], 0.5)

    def test_strategy_tuning_can_apply_breakeven_trigger_change(self):
        current = StrategySection(
            style="ema_adx_macd",
            fast_window=10,
            slow_window=40,
            require_breakout=False,
            breakeven_trigger_pct=0.0,
            min_trend_strength=0.002,
            adx_min=20.0,
        )
        candidate = StrategySection(
            style="ema_adx_macd",
            fast_window=10,
            slow_window=40,
            require_breakout=False,
            breakeven_trigger_pct=0.5,
            min_trend_strength=0.002,
            adx_min=20.0,
        )
        payload = build_strategy_tuning_payload(
            current_strategy=current,
            candidate_strategy=candidate,
            baseline_latest_test_summary={
                "total_return_pct": 0.4,
                "normalized_monthly_return_pct": 0.4,
                "max_drawdown_pct": 1.5,
                "sharpe_ratio": 0.3,
            },
            candidate_latest_test_summary={
                "total_return_pct": 1.0,
                "normalized_monthly_return_pct": 1.0,
                "max_drawdown_pct": 1.8,
                "sharpe_ratio": 0.8,
            },
            walk_forward_summary={
                "average_test_normalized_monthly_return_pct": 0.7,
                "probability_positive_pct": 66.7,
            },
            walk_forward_config={
                "train_months": 3,
                "test_months": 1,
                "step_months": 1,
            },
            backtest=BacktestSection(initial_cash=300_000),
            research=ResearchSection(target_daily_profit_rub=3_000.0),
            research_window={"available_months": 4, "usable": True},
        )

        self.assertTrue(payload["changed"])
        self.assertEqual(payload["patch_values"]["breakeven_trigger_pct"], 0.5)


if __name__ == "__main__":
    unittest.main()
