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
    assert config.strategy.require_breakout is True
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
    assert config.learning_risk.probe.max_positions == 20
    assert config.learning_risk.probe.max_trades_per_day == 40
    assert config.learning_risk.probe.max_new_trades_per_cycle == 6
    assert config.learning_risk.probe.max_same_symbol_trades_per_day == 3
    assert config.learning_risk.probe.max_same_entry_mode_trades_per_day == 15
    assert config.learning_risk.probe.max_same_regime_trades_per_day == 25
    assert config.learning_risk.exploration.risk_multiplier == 0.10
    assert config.learning_risk.exploration.max_positions == 20
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
    assert config.paper_alpha_capture.enabled is True
    assert config.paper_alpha_capture.profile == "aggressive_paper_alpha"
    assert config.paper_alpha_capture.target_gross_exposure_selloff == 1.00
    assert config.golden_baseline.enabled is True
    assert config.golden_baseline.source_run == "20260707-105708"
    assert config.golden_baseline.source_commit == "56be2bda69876f917731f81d913fa32aa9aad8b5"
    assert config.golden_baseline.timeframes == ["15min", "5min", "1min"]
    assert config.golden_baseline.forbidden_timeframes == ["10min"]
    assert config.golden_baseline.early_5m.starter_size_multiplier == 0.25
    assert config.golden_baseline.early_5m.max_positions == 3
    assert config.golden_baseline.execution_1m.required_for_early_5m is True
    assert config.short_only.enabled is True
    assert config.short_only.disable_all_longs is True
    assert config.short_only.flatten_existing_longs is True
    assert config.short_only.no_trade_in_range_chop is True
    assert config.short_only.allow_shorts_only_in_regimes == [
        "market_selloff_impulse",
        "clean_downtrend",
        "weak_down_choppy",
        "mixed_bearish",
    ]
    assert config.short_only.strategy_signal_is_optional is False
    assert config.short_only.allow_synthetic_short_candidates is True
    assert config.short_only.synthetic.enabled is True
    assert config.short_only.synthetic.real_trading_enabled is False
    assert config.short_only.synthetic.shadow_only is True
    assert config.short_only.allow_existing_short_upsize is False
    assert config.short_only.upsize.enabled is False
    assert config.short_only.upsize.real_trading_enabled is False
    assert config.short_only.upsize.shadow_only is True
    assert config.short_only.paper_exposure_sizing_enabled is False
    assert config.short_only.paper_exposure_sizing.enabled is False
    assert config.short_only.real_trade_sources.strategy_short is True
    assert config.short_only.real_trade_sources.early_5m_starter is True
    assert config.short_only.real_trade_sources.synthetic is False
    assert config.short_only.real_trade_sources.ml_only is False
    assert config.short_only.real_trade_sources.price_action_fallback is False
    assert config.short_only.real_trade_sources.upsize is False
    assert config.short_only.real_trade_sources.mixed_bearish is False
    assert config.short_only.real_trade_sources.expanded_sizing is False
    assert config.short_only.mixed_bearish_override.enabled is True
    assert config.short_only.mixed_bearish_override.min_breadth_down == 0.70
    assert config.short_only.mixed_bearish_override.real_trading_enabled is False
    assert config.short_only.mixed_bearish_override.shadow_only is True
    assert config.short_only.edge.min_expected_net_edge_rub == 5.0
    assert config.short_only.edge.allow_price_action_edge_in_mixed_bearish is True
    assert config.short_only.sizing.market_selloff_impulse.target_gross_exposure == 1.00
    assert config.short_only.sizing.market_selloff_impulse.max_positions == 12
    assert config.short_only.sizing.market_selloff_impulse.max_new_shorts_per_cycle == 12
    assert config.short_only.sizing.market_selloff_impulse.per_symbol_exposure_target == 0.08
    assert config.short_only.sizing.market_selloff_impulse.per_symbol_exposure_max == 0.12
    assert config.short_only.sizing.clean_downtrend.target_gross_exposure == 0.70
    assert config.short_only.sizing.clean_downtrend.max_positions == 10
    assert config.short_only.sizing.market_selloff_impulse.max_risk_quantity_expansion == 1.0
    assert config.short_only.sizing.clean_downtrend.max_risk_quantity_expansion == 1.0
    assert config.short_only.sizing.weak_down_choppy.target_gross_exposure == 0.35
    assert config.short_only.sizing.weak_down_choppy.max_gross_exposure == 0.60
    assert config.short_only.sizing.weak_down_choppy.max_positions == 6
    assert config.short_only.sizing.weak_down_choppy.max_new_shorts_per_cycle == 6
    assert config.short_only.sizing.weak_down_choppy.per_symbol_exposure_target == 0.04
    assert config.short_only.sizing.weak_down_choppy.per_symbol_exposure_max == 0.07
    assert config.short_only.sizing.weak_down_choppy.max_risk_quantity_expansion == 1.0
    assert config.short_only.sizing.mixed_bearish.target_gross_exposure == 0.0
    assert config.short_only.sizing.mixed_bearish.max_positions == 0
    assert config.short_only.mixed_bearish_override.target_gross_exposure == 0.0
    assert config.short_only.mixed_bearish_override.max_positions == 0
    assert config.short_only.mixed_bearish_override.per_symbol_exposure_target == 0.0
    assert config.short_only.mixed_bearish_override.per_symbol_exposure_max == 0.0
    assert config.short_only.microstructure.hard_max_spread_bps == 40.0
    assert config.short_only.confirmation.strong_rebound_action == "no_trade"
    assert config.short_only.ml.positive_edge_is_required_but_not_sufficient is True
    assert config.short_only.ml.ml_positive_standalone_real_trading is False
    assert config.short_only.ml.missing_model_action == "no_trade"
    assert config.short_only.damage_guard.enabled is True
    assert config.short_only.damage_guard.daily_loss_limit_rub == 1500.0
    assert config.short_only.damage_guard.daily_loss_limit_pct == 0.005
    assert config.short_only.damage_guard.include_open_pnl is True
    assert config.short_only.exits.early_loss_guard_enabled is True
    assert config.short_ev_engine.enabled is True
    assert config.short_ev_engine.mode == "short_only_ev"
    assert config.short_ev_engine.allowed_setups == [
        "normal_15m_trend_short",
        "golden_15m_breakout_short",
        "early_5m_acceleration_short",
        "failed_rebound_short",
        "market_selloff_short",
    ]
    assert config.short_ev_engine.long_enabled is False
    assert config.short_ev_engine.range_chop_enabled is False
    assert config.short_ev_engine.live_enabled is False
    assert config.short_ev_engine.ev_gate.min_ev_net_rub == 10.0
    assert config.short_ev_engine.ev_gate.min_ev_per_risk == 0.05
    assert config.short_ev_engine.probe.max_size_multiplier == 0.10
    assert config.short_ev_engine.exits.breakeven.activation_mfe_pct == 0.0035
    assert config.short_ev_engine.exits.trailing.atr_timeframe == "5min"
    assert config.short_ev_engine.timeframes.primary == "15min"
    assert config.short_ev_engine.timeframes.trigger == "5min"
    assert config.short_ev_engine.timeframes.execution_guard == "1min"
    assert config.short_ev_engine.timeframes.forbidden == ["10min"]
    assert config.regime_policy.weak_down_choppy.short_direct_probe_enabled is True
    assert config.regime_policy.weak_down_choppy.short_direct_probe_min_signal_strength == 0.15
    assert config.regime_policy.weak_down_choppy.short_direct_exploration_min_signal_strength == 0.08
    assert config.regime_policy.weak_down_choppy.short_direct_probe_multiplier == 0.40
    assert config.regime_policy.weak_down_choppy.short_direct_exploration_multiplier == 0.25
    assert config.regime_policy.weak_down_choppy.short_direct_probe_max_soft_issues == 8
    assert config.regime_policy.weak_down_choppy.allow_ml_negative_edge_exploration is True
    assert config.regime_policy.weak_down_choppy.create_pullback_addon_after_direct_probe is True
    assert config.regime_policy.weak_down_choppy.pullback_addon_multiplier == 0.15
    assert config.regime_policy.weak_down_choppy.long.allow_normal_long is False
    assert config.side_policy.long.normal_enabled is False
    assert config.side_policy.long.probe_enabled is False
    assert config.side_policy.long.exploration_enabled is False
    assert config.side_policy.long.full_size_long_requires_clean_uptrend is True
    assert config.side_policy.long.exploration_risk_multiplier == 0.05
    assert config.market_selloff_impulse.basket.enabled is True
    assert config.market_selloff_impulse.basket.max_new_shorts_per_cycle == 20
    assert config.market_selloff_impulse.basket.max_selloff_positions == 20
    assert config.market_selloff_impulse.basket.per_symbol_risk_multiplier == 0.15
    assert config.market_selloff_impulse.basket.per_symbol_exposure_target == 0.12
    assert config.market_selloff_impulse.basket.per_symbol_exposure_max == 0.18
    assert config.market_selloff_impulse.basket.max_total_selloff_gross_exposure == 1.00
    assert config.market_selloff_impulse.basket.max_total_selloff_risk == 0.03
    assert config.market_selloff_impulse.basket.min_symbols_to_trade == 4
    assert config.market_selloff_impulse.basket.max_symbols_to_trade == 20
    assert config.market_selloff_impulse.risk.market_breakdown_short_multiplier == 0.60
    assert config.confirmation_5m.market_selloff_impulse.min_bars == 1
    assert config.confirmation_5m.market_selloff_impulse.neutral_confirmation_mode == "allow_reduced_short"
    assert config.learning_microstructure.market_selloff_impulse.max_entry_spread_bps_normal == 20.0
    assert config.market_selloff_impulse.learning_caps.max_same_symbol_selloff_trades_per_day == 2
    assert config.market_selloff_impulse.learning_caps.max_same_entry_mode_selloff_trades_per_day == 20
    assert config.market_selloff_impulse.learning_caps.max_same_regime_selloff_trades_per_day == 35
    assert config.market_selloff_impulse.long.allow_normal_long is False
    assert config.market_selloff_impulse.long.capitulation_bounce_probe_multiplier == 0.05
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


def test_aggressive_paper_alpha_disabled_outside_local_paper(tmp_path: Path):
    config = load_config(
        _write_config(
            tmp_path,
            "\n".join(
                [
                    "[paper_alpha_capture]",
                    "enabled = true",
                    "",
                    "[execution]",
                    'mode = "tbank-sandbox"',
                    "allow_live_trading = false",
                ]
            ),
        )
    )

    assert config.paper_alpha_capture.enabled is False
    assert config.execution.allow_live_trading is False


def test_short_only_disabled_outside_local_paper(tmp_path: Path):
    config = load_config(
        _write_config(
            tmp_path,
            "\n".join(
                [
                    "[short_only]",
                    "enabled = true",
                    "",
                    "[execution]",
                    'mode = "tbank-sandbox"',
                    "allow_live_trading = false",
                ]
            ),
        )
    )

    assert config.short_only.enabled is False
    assert config.execution.allow_live_trading is False


def test_short_ev_engine_disabled_outside_local_paper(tmp_path: Path):
    config = load_config(
        _write_config(
            tmp_path,
            "\n".join(
                [
                    "[short_ev_engine]",
                    "enabled = true",
                    "",
                    "[execution]",
                    'mode = "tbank-sandbox"',
                    "allow_live_trading = false",
                ]
            ),
        )
    )

    assert config.short_ev_engine.enabled is False
    assert config.short_ev_engine.live_enabled is False
    assert config.execution.allow_live_trading is False
