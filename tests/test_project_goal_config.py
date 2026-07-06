from __future__ import annotations

from pathlib import Path

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
    assert config.learning_mode.allow_probe_to_use_global_position_slots is True
    assert config.learning_mode.allow_exploration_to_use_global_position_slots is True
    assert config.learning_risk.probe.risk_multiplier == 0.25
    assert config.learning_risk.probe.max_positions == 12
    assert config.learning_risk.probe.max_trades_per_day == 40
    assert config.learning_risk.probe.max_new_trades_per_cycle == 6
    assert config.learning_risk.probe.max_same_symbol_trades_per_day == 3
    assert config.learning_risk.probe.max_same_entry_mode_trades_per_day == 15
    assert config.learning_risk.probe.max_same_regime_trades_per_day == 25
    assert config.learning_risk.exploration.risk_multiplier == 0.10
    assert config.learning_risk.exploration.max_positions == 12
    assert config.learning_risk.exploration.max_trades_per_day == 40
    assert config.learning_risk.exploration.max_new_trades_per_cycle == 6
    assert config.learning_risk.exploration.max_same_symbol_trades_per_day == 2
    assert config.learning_risk.exploration.max_same_entry_mode_trades_per_day == 15
    assert config.learning_risk.exploration.max_same_regime_trades_per_day == 25
    assert config.learning_caps.daily_cap_behavior == "warn_only"
    assert config.learning_caps.same_symbol_cap_behavior == "shadow_only"
    assert config.learning_caps.same_entry_mode_cap_behavior == "shadow_only"
    assert config.learning_caps.same_regime_cap_behavior == "reduce_size"
    assert config.learning_caps.same_regime_cap_multiplier == 0.50
    assert config.regime_policy.weak_down_choppy.short_direct_probe_enabled is True
    assert config.regime_policy.weak_down_choppy.short_direct_probe_min_signal_strength == 0.20
    assert config.regime_policy.weak_down_choppy.short_direct_probe_multiplier == 0.25
    assert config.regime_policy.weak_down_choppy.create_pullback_addon_after_direct_probe is True
    assert config.regime_policy.weak_down_choppy.pullback_addon_multiplier == 0.15
    assert config.regime_policy.weak_down_choppy.long.allow_normal_long is False
    assert config.side_policy.long.normal_enabled is True
    assert config.side_policy.long.full_size_long_requires_clean_uptrend is True
    assert config.side_policy.long.exploration_risk_multiplier == 0.05
    assert config.market_selloff_impulse.basket.enabled is True
    assert config.market_selloff_impulse.basket.max_new_shorts_per_cycle == 8
    assert config.market_selloff_impulse.basket.max_selloff_positions == 10
    assert config.market_selloff_impulse.basket.per_symbol_risk_multiplier == 0.15
    assert config.market_selloff_impulse.basket.max_total_selloff_risk == 0.015
    assert config.market_selloff_impulse.basket.min_symbols_to_trade == 2
    assert config.market_selloff_impulse.basket.max_symbols_to_trade == 10
    assert config.market_selloff_impulse.learning_caps.max_same_symbol_selloff_trades_per_day == 2
    assert config.market_selloff_impulse.learning_caps.max_same_entry_mode_selloff_trades_per_day == 20
    assert config.market_selloff_impulse.learning_caps.max_same_regime_selloff_trades_per_day == 35
    assert config.market_selloff_impulse.long.allow_normal_long is False
    assert config.backtest.initial_cash == 300_000
    assert effective_target_daily_profit_rub(config.research, config.backtest) == 2_000.0
    assert effective_target_monthly_profit_rub(config.research, config.backtest) == 40_000.0
    assert round(effective_target_monthly_return_pct(config.research, config.backtest), 3) == 13.333


def _write_config(tmp_path: Path, body: str) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "test.toml"
    config_path.write_text(body, encoding="utf-8")
    return config_path


def test_relaxed_probe_defaults_are_not_too_conservative(tmp_path: Path):
    config = load_config(_write_config(tmp_path, ""))

    assert config.learning_risk.probe.risk_multiplier == 0.25
    assert config.learning_risk.probe.max_positions == 12
    assert config.learning_risk.probe.max_trades_per_day == 40


def test_relaxed_exploration_defaults_are_not_too_conservative(tmp_path: Path):
    config = load_config(_write_config(tmp_path, ""))

    assert config.learning_risk.exploration.risk_multiplier == 0.10
    assert config.learning_risk.exploration.max_positions == 12
    assert config.learning_risk.exploration.max_trades_per_day == 40


def test_old_probe_cap_aliases_do_not_override_new_defaults(tmp_path: Path):
    config = load_config(
        _write_config(
            tmp_path,
            "\n".join(
                [
                    "[risk]",
                    "max_probe_positions = 5",
                    "max_probe_trades_per_day = 10",
                    "max_exploration_positions = 4",
                    "max_exploration_trades_per_day = 8",
                ]
            ),
        )
    )

    assert config.learning_risk.probe.max_positions == 12
    assert config.learning_risk.probe.max_trades_per_day == 40
    assert config.learning_risk.exploration.max_positions == 12
    assert config.learning_risk.exploration.max_trades_per_day == 40


def test_explicit_user_probe_cap_config_is_respected(tmp_path: Path):
    config = load_config(
        _write_config(
            tmp_path,
            "\n".join(
                [
                    "[risk.probe]",
                    "risk_multiplier = 0.2",
                    "max_positions = 7",
                    "max_trades_per_day = 21",
                    "max_same_symbol_trades_per_day = 4",
                ]
            ),
        )
    )

    assert config.learning_risk.probe.risk_multiplier == 0.2
    assert config.learning_risk.probe.max_positions == 7
    assert config.learning_risk.probe.max_trades_per_day == 21
    assert config.learning_risk.probe.max_same_symbol_trades_per_day == 4


def test_explicit_user_exploration_cap_config_is_respected(tmp_path: Path):
    config = load_config(
        _write_config(
            tmp_path,
            "\n".join(
                [
                    "[learning_risk.exploration]",
                    "risk_multiplier = 0.05",
                    "max_positions = 6",
                    "max_trades_per_day = 18",
                    "max_new_trades_per_cycle = 3",
                ]
            ),
        )
    )

    assert config.learning_risk.exploration.risk_multiplier == 0.05
    assert config.learning_risk.exploration.max_positions == 6
    assert config.learning_risk.exploration.max_trades_per_day == 18
    assert config.learning_risk.exploration.max_new_trades_per_cycle == 3
