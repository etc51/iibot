from __future__ import annotations

import unittest
from datetime import datetime, timezone

from samosbor.autonomy.exit_tuning import (
    build_exit_reason_breakdown,
    build_exit_tuning_payload,
    specialize_exit_tuning_research,
)
from samosbor.config import BacktestSection, ResearchSection, StrategySection
from samosbor.domain import SignalDirection, TradeRecord


def _trade(net_pnl: float, reason: str) -> TradeRecord:
    return TradeRecord(
        symbol="CNYRUBF",
        direction=SignalDirection.LONG,
        quantity_lots=1,
        entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        exit_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
        entry_price=100.0,
        exit_price=101.0,
        gross_pnl=net_pnl,
        net_pnl=net_pnl,
        reason=reason,
    )


class ExitTuningResearchTest(unittest.TestCase):
    def test_specialized_exit_research_freezes_entry_dimensions(self):
        research = ResearchSection(
            strategy_styles=["ema_adx_macd", "sma_breakout"],
            fast_windows=[8, 10],
            slow_windows=[30, 40],
            require_breakout_values=[False, True],
            atr_stop_multipliers=[1.25, 1.5, 1.75],
            reward_to_risk_values=[1.5, 2.0, 2.5],
            trend_strength_values=[0.002, 0.004],
            adx_min_values=[15.0, 20.0],
        )
        strategy = StrategySection(
            style="ema_adx_macd",
            fast_window=10,
            slow_window=40,
            require_breakout=False,
            atr_stop_multiple=1.5,
            reward_to_risk=2.0,
            min_trend_strength=0.002,
            adx_min=20.0,
        )

        specialized = specialize_exit_tuning_research(research, strategy)

        self.assertEqual(specialized.strategy_styles, ["ema_adx_macd"])
        self.assertEqual(specialized.fast_windows, [10])
        self.assertEqual(specialized.slow_windows, [40])
        self.assertEqual(specialized.require_breakout_values, [False])
        self.assertEqual(specialized.atr_stop_multipliers, [1.25, 1.5, 1.75])
        self.assertEqual(specialized.reward_to_risk_values, [1.5, 2.0, 2.5])


class ExitTuningPayloadTest(unittest.TestCase):
    def test_exit_breakdown_groups_reasons(self):
        rows = build_exit_reason_breakdown(
            [
                _trade(-120.0, "stop-loss"),
                _trade(-80.0, "stop-loss"),
                _trade(140.0, "take-profit"),
            ]
        )

        self.assertEqual(rows[0]["reason"], "stop-loss")
        self.assertEqual(rows[0]["net_pnl_rub"], -200.0)
        self.assertEqual(rows[1]["reason"], "take-profit")
        self.assertEqual(rows[1]["net_pnl_rub"], 140.0)

    def test_exit_tuning_accepts_candidate_when_guardrails_pass(self):
        current = StrategySection(atr_stop_multiple=1.5, reward_to_risk=2.0)
        candidate = StrategySection(atr_stop_multiple=1.75, reward_to_risk=1.5)
        payload = build_exit_tuning_payload(
            current_strategy=current,
            candidate_strategy=candidate,
            baseline_latest_test_summary={
                "total_return_pct": 0.4,
                "normalized_monthly_return_pct": 0.4,
                "max_drawdown_pct": 1.2,
                "sharpe_ratio": 0.4,
            },
            candidate_latest_test_summary={
                "total_return_pct": 0.8,
                "normalized_monthly_return_pct": 0.8,
                "max_drawdown_pct": 1.5,
                "sharpe_ratio": 0.7,
            },
            baseline_exit_breakdown=build_exit_reason_breakdown(
                [_trade(-100.0, "stop-loss"), _trade(120.0, "take-profit")]
            ),
            candidate_exit_breakdown=build_exit_reason_breakdown(
                [_trade(-60.0, "stop-loss"), _trade(180.0, "take-profit")]
            ),
            walk_forward_summary={
                "average_test_normalized_monthly_return_pct": 0.5,
                "probability_positive_pct": 66.7,
            },
            walk_forward_config={"train_months": 4, "test_months": 1, "step_months": 1},
            backtest=BacktestSection(initial_cash=1_000_000),
            research=ResearchSection(target_monthly_profit_rub=7_500.0),
            research_window={"available_months": 5, "usable": True},
        )

        self.assertTrue(payload["changed"])
        self.assertEqual(payload["target"]["daily_profit_rub"], 375.0)
        self.assertEqual(payload["patch_values"]["atr_stop_multiple"], 1.75)
        self.assertEqual(payload["patch_values"]["reward_to_risk"], 1.5)

    def test_exit_tuning_accepts_ruble_exit_improvement_below_monthly_threshold(self):
        current = StrategySection(atr_stop_multiple=1.5, reward_to_risk=2.0)
        candidate = StrategySection(atr_stop_multiple=1.75, reward_to_risk=2.0)
        payload = build_exit_tuning_payload(
            current_strategy=current,
            candidate_strategy=candidate,
            baseline_latest_test_summary={
                "total_return_pct": 0.4,
                "normalized_monthly_return_pct": 0.4,
                "max_drawdown_pct": 1.2,
                "sharpe_ratio": 0.4,
            },
            candidate_latest_test_summary={
                "total_return_pct": 0.401,
                "normalized_monthly_return_pct": 0.401,
                "max_drawdown_pct": 1.0,
                "sharpe_ratio": 0.5,
            },
            baseline_exit_breakdown=build_exit_reason_breakdown(
                [_trade(-100.0, "stop-loss"), _trade(120.0, "take-profit")]
            ),
            candidate_exit_breakdown=build_exit_reason_breakdown(
                [_trade(-60.0, "stop-loss"), _trade(170.0, "take-profit")]
            ),
            walk_forward_summary={
                "average_test_normalized_monthly_return_pct": 0.41,
                "probability_positive_pct": 100.0,
            },
            walk_forward_config={"train_months": 4, "test_months": 1, "step_months": 1},
            backtest=BacktestSection(initial_cash=1_000_000),
            research=ResearchSection(target_monthly_profit_rub=7_500.0),
            research_window={"available_months": 5, "usable": True},
        )

        self.assertTrue(payload["changed"])
        self.assertEqual(payload["comparison"]["realized_net_pnl_delta_rub"], 90.0)

    def test_exit_tuning_rejects_candidate_when_returns_and_ruble_pnl_do_not_improve(self):
        current = StrategySection(atr_stop_multiple=1.5, reward_to_risk=2.0)
        candidate = StrategySection(atr_stop_multiple=1.75, reward_to_risk=2.0)
        payload = build_exit_tuning_payload(
            current_strategy=current,
            candidate_strategy=candidate,
            baseline_latest_test_summary={
                "total_return_pct": 0.4,
                "normalized_monthly_return_pct": 0.4,
                "max_drawdown_pct": 1.2,
                "sharpe_ratio": 0.4,
            },
            candidate_latest_test_summary={
                "total_return_pct": 0.4,
                "normalized_monthly_return_pct": 0.4,
                "max_drawdown_pct": 1.0,
                "sharpe_ratio": 0.5,
            },
            baseline_exit_breakdown=build_exit_reason_breakdown(
                [_trade(-100.0, "stop-loss"), _trade(120.0, "take-profit")]
            ),
            candidate_exit_breakdown=build_exit_reason_breakdown(
                [_trade(-90.0, "stop-loss"), _trade(110.0, "take-profit")]
            ),
            walk_forward_summary={
                "average_test_normalized_monthly_return_pct": 0.41,
                "probability_positive_pct": 100.0,
            },
            walk_forward_config={"train_months": 4, "test_months": 1, "step_months": 1},
            backtest=BacktestSection(initial_cash=1_000_000),
            research=ResearchSection(target_monthly_profit_rub=7_500.0),
            research_window={"available_months": 5, "usable": True},
        )

        self.assertFalse(payload["changed"])
        self.assertEqual(payload["reason"], "candidate exit settings failed one or more safety guardrails")

    def test_exit_tuning_can_apply_trailing_profit_patch(self):
        current = StrategySection(
            atr_stop_multiple=1.5,
            reward_to_risk=2.0,
            trailing_profit_trigger_rub=0.0,
            trailing_profit_lock_ratio=0.0,
        )
        candidate = StrategySection(
            atr_stop_multiple=1.5,
            reward_to_risk=2.0,
            trailing_profit_trigger_rub=1_200.0,
            trailing_profit_lock_ratio=0.5,
        )
        payload = build_exit_tuning_payload(
            current_strategy=current,
            candidate_strategy=candidate,
            baseline_latest_test_summary={
                "total_return_pct": 0.4,
                "normalized_monthly_return_pct": 0.4,
                "max_drawdown_pct": 1.2,
                "sharpe_ratio": 0.4,
            },
            candidate_latest_test_summary={
                "total_return_pct": 0.9,
                "normalized_monthly_return_pct": 0.9,
                "max_drawdown_pct": 1.4,
                "sharpe_ratio": 0.8,
            },
            baseline_exit_breakdown=build_exit_reason_breakdown(
                [_trade(-100.0, "stop-loss"), _trade(120.0, "take-profit")]
            ),
            candidate_exit_breakdown=build_exit_reason_breakdown(
                [_trade(-40.0, "stop-loss"), _trade(180.0, "take-profit")]
            ),
            walk_forward_summary={
                "average_test_normalized_monthly_return_pct": 0.6,
                "probability_positive_pct": 66.7,
            },
            walk_forward_config={"train_months": 4, "test_months": 1, "step_months": 1},
            backtest=BacktestSection(initial_cash=300_000),
            research=ResearchSection(target_daily_profit_rub=3_000.0),
            research_window={"available_months": 5, "usable": True},
        )

        self.assertTrue(payload["changed"])
        self.assertEqual(payload["patch_values"]["trailing_profit_trigger_rub"], 1_200.0)
        self.assertEqual(payload["patch_values"]["trailing_profit_lock_ratio"], 0.5)

    def test_exit_tuning_can_apply_breakeven_trigger_patch(self):
        current = StrategySection(
            atr_stop_multiple=1.5,
            reward_to_risk=2.0,
            breakeven_trigger_pct=0.0,
        )
        candidate = StrategySection(
            atr_stop_multiple=1.5,
            reward_to_risk=2.0,
            breakeven_trigger_pct=0.5,
        )
        payload = build_exit_tuning_payload(
            current_strategy=current,
            candidate_strategy=candidate,
            baseline_latest_test_summary={
                "total_return_pct": 0.4,
                "normalized_monthly_return_pct": 0.4,
                "max_drawdown_pct": 1.2,
                "sharpe_ratio": 0.4,
            },
            candidate_latest_test_summary={
                "total_return_pct": 0.9,
                "normalized_monthly_return_pct": 0.9,
                "max_drawdown_pct": 1.3,
                "sharpe_ratio": 0.8,
            },
            baseline_exit_breakdown=build_exit_reason_breakdown(
                [_trade(-100.0, "stop-loss"), _trade(120.0, "take-profit")]
            ),
            candidate_exit_breakdown=build_exit_reason_breakdown(
                [_trade(-40.0, "stop-loss"), _trade(170.0, "take-profit")]
            ),
            walk_forward_summary={
                "average_test_normalized_monthly_return_pct": 0.6,
                "probability_positive_pct": 66.7,
            },
            walk_forward_config={"train_months": 4, "test_months": 1, "step_months": 1},
            backtest=BacktestSection(initial_cash=300_000),
            research=ResearchSection(target_daily_profit_rub=3_000.0),
            research_window={"available_months": 5, "usable": True},
        )

        self.assertTrue(payload["changed"])
        self.assertEqual(payload["patch_values"]["breakeven_trigger_pct"], 0.5)


if __name__ == "__main__":
    unittest.main()
