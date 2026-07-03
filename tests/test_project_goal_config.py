from __future__ import annotations

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
    assert config.execution.commission_bps == 4.0
    assert config.data.tbank_candle_source == "include-weekend"
    assert config.strategy.allowed_entry_hours == []
    assert config.strategy.allowed_entry_weekdays == []
    assert config.strategy.forced_flat_hours == []
    assert config.backtest.initial_cash == 300_000
    assert effective_target_daily_profit_rub(config.research, config.backtest) == 2_000.0
    assert effective_target_monthly_profit_rub(config.research, config.backtest) == 40_000.0
    assert round(effective_target_monthly_return_pct(config.research, config.backtest), 3) == 13.333
