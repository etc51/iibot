from __future__ import annotations

from samosbor.autonomy.ml_learning import LOW_QUALITY_PROBABILITY_THRESHOLD
from samosbor.config import load_config
from samosbor.domain import TradeMode
from samosbor.research.targets import (
    effective_target_daily_profit_rub,
    effective_target_monthly_profit_rub,
    effective_target_monthly_return_pct,
)


def test_focused_runtime_matches_project_goal():
    config = load_config("configs/server_tbank_stocks_intraday_300k_focused.toml")

    assert config.execution.mode == TradeMode.LOCAL_PAPER
    assert config.execution.allow_live_trading is False
    assert LOW_QUALITY_PROBABILITY_THRESHOLD >= 0.40
    assert config.execution.commission_bps == 4.0
    assert config.data.tbank_candle_source == "include-weekend"
    assert config.strategy.allowed_entry_hours == []
    assert config.strategy.allowed_entry_weekdays == []
    assert config.strategy.forced_flat_hours == []
    assert config.strategy.entry_confirmation_timeframe == "5min"
    assert config.strategy.entry_confirmation_max_adverse_ret == 0.005
    assert config.strategy.take_profit_activates_runner is True
    assert config.strategy.runner_breakeven_buffer_bps == 10.0
    assert config.strategy.runner_trailing_atr_multiple == 1.3
    assert config.strategy.runner_profit_lock_ratio == 0.35
    assert config.learning_mode.enabled is True
    assert config.learning_mode.profile == "relaxed_paper_learning"
    assert config.learning_risk.probe.max_positions == 2
    assert config.learning_risk.probe.max_trades_per_day == 6
    assert config.learning_risk.exploration.max_positions == 1
    assert config.learning_risk.exploration.max_trades_per_day == 3
    assert config.backtest.initial_cash == 300_000
    assert effective_target_daily_profit_rub(config.research, config.backtest) == 2_000.0
    assert effective_target_monthly_profit_rub(config.research, config.backtest) == 40_000.0
    assert round(effective_target_monthly_return_pct(config.research, config.backtest), 3) == 13.333
