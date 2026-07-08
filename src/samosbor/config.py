from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .domain import Instrument, InstrumentType, TradeMode


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


@dataclass(frozen=True)
class AppSection:
    name: str = "samosbor"
    timezone: str = "Europe/Moscow"


@dataclass(frozen=True)
class TBankSection:
    token_env: str = "TBANK_INVEST_TOKEN"
    token_file: str = ""
    sandbox_token_env: str = "TBANK_SANDBOX_TOKEN"
    sandbox_token_file: str = ""
    account_id_env: str = "TBANK_ACCOUNT_ID"
    account_name: str = "Фьючерсы"
    app_name: str = "samosbor"
    ssl_verify_env: str = "SSL_TBANK_VERIFY"


@dataclass(frozen=True)
class DataSection:
    source: str = "tbank"
    timeframe: str = "hour"
    history_days: int = 120
    tbank_candle_source: str = "include-weekend"
    csv_path: str = ""
    parquet_dir_path: str = ""
    local_data_pack_path: str = ""
    instruments: list[Instrument] = field(default_factory=list)


@dataclass(frozen=True)
class StrategySection:
    style: str = "sma_breakout"
    fast_window: int = 20
    slow_window: int = 50
    atr_window: int = 14
    volume_window: int = 20
    breakout_window: int = 20
    opening_range_bars: int = 2
    rel_volume_threshold: float = 1.1
    require_breakout: bool = True
    atr_stop_multiple: float = 2.0
    reward_to_risk: float = 2.0
    breakeven_trigger_pct: float = 0.0
    trailing_profit_trigger_rub: float = 0.0
    trailing_profit_lock_ratio: float = 0.0
    take_profit_activates_runner: bool = False
    runner_breakeven_buffer_bps: float = 10.0
    runner_trailing_atr_multiple: float = 1.3
    runner_profit_lock_ratio: float = 0.35
    min_signal_strength: float = 0.0
    min_trend_strength: float = 0.004
    min_liquidity_rub: float = 50_000_000
    allow_shorts: bool = True
    adx_window: int = 14
    adx_min: float = 20.0
    rsi_window: int = 14
    rsi_long_min: float = 50.0
    rsi_long_max: float = 75.0
    rsi_short_min: float = 25.0
    rsi_short_max: float = 50.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    order_book_depth: int = 10
    require_order_book: bool = False
    max_entry_spread_bps: float = 0.0
    min_entry_liquidity_cover: float = 0.0
    min_entry_book_imbalance: float = -1.0
    use_market_context: bool = False
    market_context_fast_window: int = 20
    market_context_slow_window: int = 50
    market_context_return_window: int = 4
    market_context_min_symbols: int = 8
    market_context_max_score: float = 0.25
    market_context_block_threshold: float = 0.15
    schedule_timezone: str = "Europe/Moscow"
    allowed_entry_hours: list[int] = field(default_factory=list)
    allowed_entry_weekdays: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    allowed_symbols: list[str] = field(default_factory=list)
    blocked_symbols: list[str] = field(default_factory=list)
    blocked_long_symbols: list[str] = field(default_factory=list)
    blocked_short_symbols: list[str] = field(default_factory=list)
    forced_flat_hours: list[int] = field(default_factory=list)
    forced_flat_weekdays: list[int] = field(default_factory=list)
    entry_confirmation_timeframe: str = ""
    entry_confirmation_min_bars: int = 3
    entry_confirmation_max_adverse_ret: float = 0.005


@dataclass(frozen=True)
class RiskSection:
    max_risk_per_trade: float = 0.01
    max_gross_exposure: float = 1.25
    max_drawdown: float = 0.12
    cash_reserve_ratio: float = 0.15
    max_positions: int = 6
    max_position_exposure_ratio: float = 1.0
    kelly_lookback_trades: int = 20
    min_trades_for_kelly: int = 8


@dataclass(frozen=True)
class LearningModeSection:
    enabled: bool = False
    profile: str = "strict"
    record_strict_policy_shadow: bool = False
    record_rejected_shadow: bool = False
    allow_probe_trades: bool = True
    allow_exploration_trades: bool = True
    min_effective_risk_multiplier_to_trade: float = 0.05
    allow_choppy_trend_short_probe: bool = True
    choppy_trend_short_probe_min_signal_strength: float = 0.45
    choppy_trend_short_default_mode: str = "wait_pullback"
    allow_range_chop_exploration: bool = True
    allow_probe_to_use_global_position_slots: bool = True
    allow_exploration_to_use_global_position_slots: bool = True
    do_not_block_probe_due_to_daily_limit_unless_extreme: bool = True
    do_not_block_exploration_due_to_daily_limit_unless_extreme: bool = True


@dataclass(frozen=True)
class ModeRiskSection:
    risk_multiplier: float = 1.0
    max_positions: int = 0
    max_trades_per_day: int = 0
    max_new_trades_per_cycle: int = 0
    max_same_symbol_trades_per_day: int = 0
    max_same_entry_mode_trades_per_day: int = 0
    max_same_regime_trades_per_day: int = 0


@dataclass(frozen=True)
class LearningCapsSection:
    daily_cap_behavior: str = "warn_only"
    same_symbol_cap_behavior: str = "shadow_only"
    same_entry_mode_cap_behavior: str = "shadow_only"
    same_regime_cap_behavior: str = "reduce_size"
    same_regime_cap_multiplier: float = 0.50


@dataclass(frozen=True)
class LearningRiskSection:
    normal: ModeRiskSection = field(default_factory=lambda: ModeRiskSection(risk_multiplier=1.0))
    probe: ModeRiskSection = field(
        default_factory=lambda: ModeRiskSection(
            risk_multiplier=0.25,
            max_positions=12,
            max_trades_per_day=40,
            max_new_trades_per_cycle=6,
            max_same_symbol_trades_per_day=3,
            max_same_entry_mode_trades_per_day=15,
            max_same_regime_trades_per_day=25,
        )
    )
    exploration: ModeRiskSection = field(
        default_factory=lambda: ModeRiskSection(
            risk_multiplier=0.10,
            max_positions=12,
            max_trades_per_day=40,
            max_new_trades_per_cycle=6,
            max_same_symbol_trades_per_day=2,
            max_same_entry_mode_trades_per_day=15,
            max_same_regime_trades_per_day=25,
        )
    )


@dataclass(frozen=True)
class ModeSignalSection:
    min_signal_strength: float = 0.0
    min_trend_strength: float = 0.0
    adx_min: float = 0.0


@dataclass(frozen=True)
class LearningSignalsSection:
    normal: ModeSignalSection = field(
        default_factory=lambda: ModeSignalSection(
            min_signal_strength=0.30,
            min_trend_strength=0.002,
            adx_min=20.0,
        )
    )
    probe: ModeSignalSection = field(
        default_factory=lambda: ModeSignalSection(
            min_signal_strength=0.20,
            min_trend_strength=0.001,
            adx_min=16.0,
        )
    )
    exploration: ModeSignalSection = field(
        default_factory=lambda: ModeSignalSection(
            min_signal_strength=0.12,
            min_trend_strength=0.0005,
            adx_min=12.0,
        )
    )


@dataclass(frozen=True)
class ModeMicrostructureSection:
    max_entry_spread_bps: float = 0.0
    min_entry_liquidity_cover: float = 0.0
    min_entry_book_imbalance: float = -1.0


@dataclass(frozen=True)
class LearningMicrostructureSection:
    normal: ModeMicrostructureSection = field(
        default_factory=lambda: ModeMicrostructureSection(
            max_entry_spread_bps=12.0,
            min_entry_liquidity_cover=2.0,
            min_entry_book_imbalance=-0.35,
        )
    )
    probe: ModeMicrostructureSection = field(
        default_factory=lambda: ModeMicrostructureSection(
            max_entry_spread_bps=18.0,
            min_entry_liquidity_cover=1.2,
            min_entry_book_imbalance=-0.60,
        )
    )
    exploration: ModeMicrostructureSection = field(
        default_factory=lambda: ModeMicrostructureSection(
            max_entry_spread_bps=25.0,
            min_entry_liquidity_cover=0.8,
            min_entry_book_imbalance=-0.80,
        )
    )
    market_selloff_impulse: "MarketSelloffMicrostructureSection" = field(
        default_factory=lambda: MarketSelloffMicrostructureSection()
    )


@dataclass(frozen=True)
class MlLearningPolicySection:
    negative_edge_only_multiplier: float = 0.35
    negative_edge_plus_one_soft_issue_multiplier: float = 0.15
    negative_edge_plus_multiple_soft_issues_multiplier: float = 0.08
    negative_edge_plus_hard_execution_issue: str = "reject"


@dataclass(frozen=True)
class Confirmation5mMarketSelloffImpulseSection:
    min_bars: int = 1
    allow_same_bar_breakdown: bool = True
    neutral_confirmation_mode: str = "allow_reduced_short"
    mild_rebound_against_short_mode: str = "allow_reduced_short"
    strong_rebound_against_short_mode: str = "stop_chase_wait_pullback"
    extreme_adverse_mode: str = "shadow_or_reject"


@dataclass(frozen=True)
class Confirmation5mPolicySection:
    neutral_confirmation_mode: str = "probe"
    mild_rebound_against_short_mode: str = "exploration_or_wait"
    strong_rebound_against_short_mode: str = "wait_pullback"
    hard_block_rebound_against_short: bool = False
    mild_adverse_ret: float = 0.0025
    strong_adverse_ret: float = 0.005
    extreme_adverse_ret: float = 0.012
    market_selloff_impulse: Confirmation5mMarketSelloffImpulseSection = field(
        default_factory=Confirmation5mMarketSelloffImpulseSection
    )


@dataclass(frozen=True)
class WeakDownChoppyShortConfirmationSection:
    aligned_5m_mode: str = "probe"
    neutral_5m_mode: str = "probe"
    mild_rebound_5m_mode: str = "exploration_or_wait_addon"
    strong_rebound_5m_mode: str = "wait_pullback_only"
    extreme_adverse_5m_mode: str = "shadow_or_reject"


@dataclass(frozen=True)
class WeakDownChoppyRegimePolicySection:
    enable_probe_now_with_pending_addon: bool = True
    short_direct_probe_enabled: bool = True
    short_direct_exploration_enabled: bool = True
    short_direct_probe_min_signal_strength: float = 0.15
    short_direct_exploration_min_signal_strength: float = 0.08
    short_direct_probe_max_soft_issues: int = 8
    short_direct_probe_multiplier: float = 0.40
    short_direct_exploration_multiplier: float = 0.25
    create_pullback_addon_after_direct_probe: bool = True
    pullback_addon_multiplier: float = 0.15
    strict_policy_keeps_wait_pullback: bool = True
    allow_ml_negative_edge_exploration: bool = True
    short_confirmation: WeakDownChoppyShortConfirmationSection = field(
        default_factory=WeakDownChoppyShortConfirmationSection
    )
    long: WeakDownChoppyLongPolicySection = field(default_factory=lambda: WeakDownChoppyLongPolicySection())


@dataclass(frozen=True)
class CleanUptrendLongPolicySection:
    allow_direct_trend_long: bool = True
    long_direct_probe_min_signal_strength: float = 0.15
    long_direct_normal_min_signal_strength: float = 0.30
    long_probe_multiplier: float = 0.20
    long_normal_multiplier: float = 0.35


@dataclass(frozen=True)
class MixedLongPolicySection:
    allow_long_probe: bool = True
    allow_long_exploration: bool = True
    long_probe_multiplier: float = 0.10
    long_exploration_multiplier: float = 0.05
    require_5m_aligned_or_recovery: bool = True


@dataclass(frozen=True)
class RangeChopLongPolicySection:
    allow_mean_reversion_long_exploration: bool = True
    long_exploration_multiplier: float = 0.05
    require_failed_breakdown_or_reclaim: bool = True


@dataclass(frozen=True)
class WeakDownChoppyLongPolicySection:
    allow_normal_long: bool = False
    allow_rebound_probe_long: bool = True
    allow_rebound_exploration_long: bool = True
    long_probe_multiplier: float = 0.06
    long_exploration_multiplier: float = 0.04
    require_strong_rebound_or_failed_breakdown: bool = True
    default_long_mode: str = "shadow_only"


@dataclass(frozen=True)
class MarketSelloffLongPolicySection:
    allow_normal_long: bool = False
    allow_capitulation_bounce_probe: bool = True
    capitulation_bounce_probe_multiplier: float = 0.05
    default_long_mode: str = "shadow_or_tiny_probe"
    require_reclaim_confirmation: bool = True


@dataclass(frozen=True)
class LongRegimePolicySection:
    long: object


@dataclass(frozen=True)
class CleanUptrendRegimePolicySection:
    long: CleanUptrendLongPolicySection = field(default_factory=CleanUptrendLongPolicySection)


@dataclass(frozen=True)
class MixedRegimePolicySection:
    long: MixedLongPolicySection = field(default_factory=MixedLongPolicySection)


@dataclass(frozen=True)
class RangeChopRegimePolicySection:
    long: RangeChopLongPolicySection = field(default_factory=RangeChopLongPolicySection)


@dataclass(frozen=True)
class RegimePolicySection:
    weak_down_choppy: WeakDownChoppyRegimePolicySection = field(
        default_factory=WeakDownChoppyRegimePolicySection
    )
    clean_uptrend: CleanUptrendRegimePolicySection = field(default_factory=CleanUptrendRegimePolicySection)
    mixed: MixedRegimePolicySection = field(default_factory=MixedRegimePolicySection)
    range_chop: RangeChopRegimePolicySection = field(default_factory=RangeChopRegimePolicySection)


@dataclass(frozen=True)
class LongSidePolicySection:
    normal_enabled: bool = True
    normal_only_in_regimes: list[str] = field(default_factory=lambda: ["clean_uptrend"])
    probe_enabled: bool = True
    exploration_enabled: bool = True
    probe_risk_multiplier: float = 0.10
    exploration_risk_multiplier: float = 0.05
    max_probe_trades_per_day: int = 3
    max_long_probe_trades_per_day: int = 10
    max_long_exploration_trades_per_day: int = 10
    max_same_symbol_long_trades_per_day: int = 2
    full_size_long_requires_clean_uptrend: bool = True


@dataclass(frozen=True)
class SidePolicySection:
    long: LongSidePolicySection = field(default_factory=LongSidePolicySection)


@dataclass(frozen=True)
class SymbolHealthModeSection:
    risk_multiplier: float = 1.0
    max_trades_per_day: int = 0
    observe_only: bool = False


@dataclass(frozen=True)
class SymbolHealthPolicySection:
    probation: SymbolHealthModeSection = field(
        default_factory=lambda: SymbolHealthModeSection(
            risk_multiplier=0.25,
            max_trades_per_day=2,
        )
    )
    weak: SymbolHealthModeSection = field(
        default_factory=lambda: SymbolHealthModeSection(
            risk_multiplier=0.10,
            max_trades_per_day=1,
        )
    )
    observe_only: SymbolHealthModeSection = field(
        default_factory=lambda: SymbolHealthModeSection(
            risk_multiplier=0.0,
            max_trades_per_day=0,
            observe_only=True,
        )
    )


@dataclass(frozen=True)
class MarketSelloffBasketSection:
    enabled: bool = True
    max_new_shorts_per_cycle: int = 20
    max_selloff_positions: int = 20
    per_symbol_risk_multiplier: float = 0.15
    per_symbol_exposure_target: float = 0.12
    per_symbol_exposure_max: float = 0.18
    max_total_selloff_gross_exposure: float = 1.00
    max_total_selloff_risk: float = 0.03
    prefer_liquid_symbols: bool = True
    min_symbols_to_trade: int = 4
    max_symbols_to_trade: int = 20


@dataclass(frozen=True)
class MarketSelloffRiskSection:
    market_breakdown_short_multiplier: float = 0.60
    selloff_momentum_short_multiplier: float = 0.45
    panic_probe_short_multiplier: float = 0.25
    post_selloff_failed_rebound_short_multiplier: float = 0.35


@dataclass(frozen=True)
class MarketSelloffMicrostructureSection:
    max_entry_spread_bps_normal: float = 20.0
    max_entry_spread_bps_probe: float = 32.0
    max_entry_spread_bps_exploration: float = 40.0
    min_entry_liquidity_cover_normal: float = 1.0
    min_entry_liquidity_cover_probe: float = 0.6
    min_entry_liquidity_cover_exploration: float = 0.4
    min_entry_book_imbalance_normal: float = -0.60
    min_entry_book_imbalance_probe: float = -0.85
    min_entry_book_imbalance_exploration: float = -0.95


@dataclass(frozen=True)
class MarketSelloffLearningCapsSection:
    max_same_symbol_selloff_trades_per_day: int = 2
    max_same_entry_mode_selloff_trades_per_day: int = 20
    max_same_regime_selloff_trades_per_day: int = 35
    max_new_selloff_trades_per_cycle: int = 8
    daily_cap_behavior: str = "warn_only"
    same_symbol_cap_behavior: str = "shadow_only"
    same_entry_mode_cap_behavior: str = "reduce_size"


@dataclass(frozen=True)
class MarketSelloffImpulseSection:
    basket: MarketSelloffBasketSection = field(default_factory=MarketSelloffBasketSection)
    risk: MarketSelloffRiskSection = field(default_factory=MarketSelloffRiskSection)
    learning_caps: MarketSelloffLearningCapsSection = field(default_factory=MarketSelloffLearningCapsSection)
    long: MarketSelloffLongPolicySection = field(default_factory=MarketSelloffLongPolicySection)


@dataclass(frozen=True)
class PaperAlphaCaptureSection:
    enabled: bool = False
    profile: str = "aggressive_paper_alpha"
    use_full_paper_budget: bool = True
    target_gross_exposure_normal: float = 0.40
    target_gross_exposure_selloff: float = 1.00
    max_gross_exposure_selloff: float = 1.25
    min_cash_reserve_selloff: float = 0.03
    allow_budget_ramp: bool = True
    budget_ramp_step_per_cycle: float = 0.25
    do_not_wait_for_pullback_in_broad_selloff: bool = True


@dataclass(frozen=True)
class ShortOnlyEdgeSection:
    min_expected_net_edge_rub: float = 5.0
    min_expected_net_edge_per_lot_rub: float = 0.0
    required_edge_buffer_bps: float = 2.0
    allow_price_action_edge_in_selloff: bool = True
    allow_price_action_edge_in_clean_downtrend: bool = True
    allow_price_action_edge_in_weak_down_choppy: bool = True
    allow_price_action_edge_in_mixed_bearish: bool = True
    allow_ml_fallback_when_model_missing: bool = True
    allow_price_action_fallback_when_ml_stale: bool = True
    negative_ml_expected_edge_action: str = "no_trade"


@dataclass(frozen=True)
class ShortEvEngineEvGateSection:
    min_ev_net_rub: float = 10.0
    min_ev_per_risk: float = 0.05
    min_sample_count_for_real: int = 30
    min_confidence: float = 0.55
    allow_golden_baseline_prior: bool = True
    unknown_setup_action: str = "shadow_only"
    negative_ev_action: str = "no_trade"
    insufficient_sample_action: str = "shadow_or_probe"


@dataclass(frozen=True)
class ShortEvEngineProbeSection:
    enabled: bool = True
    max_size_multiplier: float = 0.10
    max_probe_trades_per_day: int = 10
    probe_requires_ev_positive: bool = True


@dataclass(frozen=True)
class ShortEvBreakevenExitSection:
    enabled: bool = True
    activation_mfe_pct: float = 0.0035
    activation_range_pct_min: float = 0.003
    activation_range_pct_max: float = 0.004
    buffer_bps: float = 2.0
    include_round_trip_costs: bool = True
    apply_to_all_short_setups: bool = True


@dataclass(frozen=True)
class ShortEvOrderBookTighteningSection:
    enabled: bool = True
    adverse_imbalance_threshold: float = -0.60
    spread_widen_bps: float = 18.0
    reclaim_5m_ema9_exit: bool = True
    tighten_multiplier: float = 0.7


@dataclass(frozen=True)
class ShortEvTrailingExitSection:
    enabled: bool = True
    activation_mfe_pct: float = 0.006
    atr_timeframe: str = "5min"
    atr_window: int = 14
    atr_multiple: float = 1.3
    use_local_5m_swing_high: bool = True
    use_order_book_tightening: bool = True


@dataclass(frozen=True)
class ShortEvExitsSection:
    breakeven: ShortEvBreakevenExitSection = field(default_factory=ShortEvBreakevenExitSection)
    trailing: ShortEvTrailingExitSection = field(default_factory=ShortEvTrailingExitSection)
    order_book_tightening: ShortEvOrderBookTighteningSection = field(
        default_factory=ShortEvOrderBookTighteningSection
    )


@dataclass(frozen=True)
class ShortEvTimeframesSection:
    primary: str = "15min"
    trigger: str = "5min"
    execution_guard: str = "1min"
    forbidden: list[str] = field(default_factory=lambda: ["10min"])


@dataclass(frozen=True)
class ShortEvDamageGuardSection:
    enabled: bool = True
    daily_loss_limit_rub: float = 1500.0
    daily_loss_limit_pct: float = 0.005
    include_open_pnl: bool = True
    action: str = "block_new_real_entries"
    allow_exits: bool = True
    allow_shadow_candidates: bool = True
    reset_next_session: bool = True


@dataclass(frozen=True)
class ShortEvEarly5mContextOverrideSection:
    enabled: bool = True
    allow_when_15m_ema_not_bearish: bool = True
    require_5m_ema9_slope_negative: bool = True
    require_5m_macd_hist_negative: bool = True
    require_5m_ret_window_negative: bool = True
    require_5m_close_below_ema9: bool = True
    require_order_book_strict: bool = True
    require_1m_guard: bool = True
    size_multiplier: float = 0.15
    shadow_if_ev_not_positive: bool = True


@dataclass(frozen=True)
class ShortEvEarly5mSetupSection:
    require_rolling_low_break: bool = False
    rolling_low_as_quality_bonus: bool = True
    size_multiplier_with_rolling_low: float = 0.25
    size_multiplier_without_rolling_low: float = 0.18
    context_override: ShortEvEarly5mContextOverrideSection = field(
        default_factory=ShortEvEarly5mContextOverrideSection
    )


@dataclass(frozen=True)
class ShortEvSetupsSection:
    early_5m_acceleration_short: ShortEvEarly5mSetupSection = field(
        default_factory=ShortEvEarly5mSetupSection
    )


@dataclass(frozen=True)
class ShortEvEngineSection:
    enabled: bool = False
    mode: str = "short_only_ev"
    allowed_setups: list[str] = field(
        default_factory=lambda: [
            "normal_15m_trend_short",
            "golden_15m_breakout_short",
            "early_5m_acceleration_short",
            "failed_rebound_short",
            "market_selloff_short",
        ]
    )
    unknown_setup_action: str = "shadow_only"
    long_enabled: bool = False
    range_chop_enabled: bool = False
    live_enabled: bool = False
    default_spread_cost_bps: float = 8.0
    ev_gate: ShortEvEngineEvGateSection = field(default_factory=ShortEvEngineEvGateSection)
    probe: ShortEvEngineProbeSection = field(default_factory=ShortEvEngineProbeSection)
    exits: ShortEvExitsSection = field(default_factory=ShortEvExitsSection)
    timeframes: ShortEvTimeframesSection = field(default_factory=ShortEvTimeframesSection)
    damage_guard: ShortEvDamageGuardSection = field(default_factory=ShortEvDamageGuardSection)
    setups: ShortEvSetupsSection = field(default_factory=ShortEvSetupsSection)


@dataclass(frozen=True)
class ShortOnlySizingRegimeSection:
    target_gross_exposure: float = 0.0
    max_gross_exposure: float = 0.0
    max_positions: int = 0
    max_new_shorts_per_cycle: int = 0
    per_symbol_exposure_target: float = 0.0
    per_symbol_exposure_max: float = 0.0
    max_risk_quantity_expansion: float = 1.0


@dataclass(frozen=True)
class ShortOnlyMixedBearishOverrideSection:
    enabled: bool = True
    min_breadth_down: float = 0.70
    min_confidence: float = 0.50
    min_symbols: int = 8
    target_gross_exposure: float = 0.0
    max_gross_exposure: float = 0.0
    max_positions: int = 0
    max_new_shorts_per_cycle: int = 0
    per_symbol_exposure_target: float = 0.0
    per_symbol_exposure_max: float = 0.0
    real_trading_enabled: bool = False
    shadow_only: bool = True
    min_shadow_trades_before_reenable: int = 100
    reenable_requires_positive_expectancy: bool = True


@dataclass(frozen=True)
class ShortOnlySizingSection:
    use_original_risk_manager: bool = True
    max_risk_per_trade: float = 0.01
    max_positions: int = 12
    max_gross_exposure: float = 1.5
    max_position_exposure_ratio: float = 0.18
    cash_reserve_ratio: float = 0.08
    disable_expansion: bool = True
    disable_one_lot_fallback: bool = True
    market_selloff_impulse: ShortOnlySizingRegimeSection = field(
        default_factory=lambda: ShortOnlySizingRegimeSection(
            target_gross_exposure=1.00,
            max_gross_exposure=1.25,
            max_positions=20,
            max_new_shorts_per_cycle=20,
            per_symbol_exposure_target=0.12,
            per_symbol_exposure_max=0.18,
            max_risk_quantity_expansion=1.0,
        )
    )
    clean_downtrend: ShortOnlySizingRegimeSection = field(
        default_factory=lambda: ShortOnlySizingRegimeSection(
            target_gross_exposure=1.00,
            max_gross_exposure=1.00,
            max_positions=20,
            max_new_shorts_per_cycle=20,
            per_symbol_exposure_target=0.12,
            per_symbol_exposure_max=0.18,
            max_risk_quantity_expansion=1.0,
        )
    )
    weak_down_choppy: ShortOnlySizingRegimeSection = field(
        default_factory=lambda: ShortOnlySizingRegimeSection(
            target_gross_exposure=0.15,
            max_gross_exposure=0.25,
            max_positions=3,
            max_new_shorts_per_cycle=2,
            per_symbol_exposure_target=0.03,
            per_symbol_exposure_max=0.05,
            max_risk_quantity_expansion=1.0,
        )
    )
    mixed_bearish: ShortOnlySizingRegimeSection = field(
        default_factory=lambda: ShortOnlySizingRegimeSection(
            target_gross_exposure=0.0,
            max_gross_exposure=0.0,
            max_positions=0,
            max_new_shorts_per_cycle=0,
            per_symbol_exposure_target=0.0,
            per_symbol_exposure_max=0.0,
            max_risk_quantity_expansion=1.0,
        )
    )
    range_chop: ShortOnlySizingRegimeSection = field(
        default_factory=lambda: ShortOnlySizingRegimeSection(
            target_gross_exposure=0.0,
            max_gross_exposure=0.0,
            max_positions=0,
            max_new_shorts_per_cycle=0,
            per_symbol_exposure_target=0.0,
            per_symbol_exposure_max=0.0,
            max_risk_quantity_expansion=1.0,
        )
    )


@dataclass(frozen=True)
class ShortOnlyMicrostructureSection:
    hard_max_spread_bps: float = 40.0
    hard_min_liquidity_cover: float = 0.4
    hard_min_book_imbalance: float = -0.95
    soft_spread_bps: float = 20.0
    soft_liquidity_cover: float = 1.0
    soft_book_imbalance: float = -0.60
    soft_multiplier: float = 0.75
    bad_but_allowed_multiplier: float = 0.50


@dataclass(frozen=True)
class ShortOnlyConfirmationSection:
    selloff_min_5m_bars: int = 1
    normal_min_5m_bars: int = 1
    neutral_5m_multiplier: float = 0.85
    mild_rebound_multiplier: float = 0.65
    strong_rebound_action: str = "no_trade"
    strong_rebound_multiplier: float = 0.35
    extreme_adverse_action: str = "no_trade"


@dataclass(frozen=True)
class ShortOnlyMlSection:
    allow_if_expected_net_edge_positive: bool = True
    positive_edge_is_required_but_not_sufficient: bool = True
    ml_positive_standalone_real_trading: bool = False
    negative_edge_action: str = "no_trade"
    low_quality_action: str = "no_trade"
    missing_model_action: str = "no_trade"
    positive_edge_multiplier: float = 1.0
    weak_positive_edge_multiplier: float = 0.5


@dataclass(frozen=True)
class ShortOnlyRealTradeSourcesSection:
    strategy_short: bool = True
    early_5m_starter: bool = True
    synthetic: bool = False
    ml_only: bool = False
    price_action_fallback: bool = False
    upsize: bool = False
    mixed_bearish: bool = False
    expanded_sizing: bool = False


@dataclass(frozen=True)
class ShortOnlySyntheticSection:
    enabled: bool = True
    real_trading_enabled: bool = False
    shadow_only: bool = True
    reason: str = "disabled_after_0_53_short_only_winners"
    min_shadow_trades_before_reenable: int = 100
    reenable_requires_positive_expectancy: bool = True
    reenable_requires_profit_factor: float = 1.15


@dataclass(frozen=True)
class ShortOnlyPaperExposureSizingSection:
    enabled: bool = False
    reason: str = "disabled_after_expanded_size_losses"


@dataclass(frozen=True)
class ShortOnlyUpsizeSection:
    enabled: bool = False
    real_trading_enabled: bool = False
    shadow_only: bool = True
    reason: str = "disabled_until_strategy_short_baseline_positive"


@dataclass(frozen=True)
class ShortOnlyDamageGuardSection:
    enabled: bool = True
    daily_loss_limit_rub: float = 1500.0
    daily_loss_limit_pct: float = 0.005
    include_open_pnl: bool = True
    action: str = "block_new_entries"
    allow_exits: bool = True
    allow_position_management: bool = True
    allow_shadow_candidates: bool = True
    reset_next_session: bool = True


@dataclass(frozen=True)
class ShortOnlyExitsSection:
    use_existing_atr_stop: bool = True
    use_existing_take_profit: bool = True
    use_existing_runner: bool = True
    early_loss_guard_enabled: bool = True
    early_loss_guard_bars: int = 2
    early_loss_guard_min_mfe_r: float = 0.25
    early_loss_guard_exit_if_negative: bool = True
    breakeven_after_mfe_r: float = 0.50
    breakeven_buffer_bps: float = 5.0


@dataclass(frozen=True)
class ShortOnlySection:
    enabled: bool = False
    disable_all_longs: bool = True
    flatten_existing_longs: bool = True
    no_trade_in_range_chop: bool = True
    allow_shorts_only_in_regimes: list[str] = field(
        default_factory=lambda: [
            "market_selloff_impulse",
            "clean_downtrend",
            "weak_down_choppy",
            "mixed_bearish",
        ]
    )
    allow_mixed_regime_shorts: bool = False
    pullback_is_addon_not_required: bool = True
    ml_is_edge_gate_not_blocker: bool = True
    microstructure_is_size_modifier_not_blocker: bool = True
    confirmation_is_size_modifier_not_blocker: bool = True
    strategy_signal_is_optional: bool = False
    allow_synthetic_short_candidates: bool = True
    allow_existing_short_upsize: bool = False
    paper_exposure_sizing_enabled: bool = False
    real_trade_sources: ShortOnlyRealTradeSourcesSection = field(
        default_factory=ShortOnlyRealTradeSourcesSection
    )
    mixed_bearish_override: ShortOnlyMixedBearishOverrideSection = field(
        default_factory=ShortOnlyMixedBearishOverrideSection
    )
    synthetic: ShortOnlySyntheticSection = field(default_factory=ShortOnlySyntheticSection)
    paper_exposure_sizing: ShortOnlyPaperExposureSizingSection = field(
        default_factory=ShortOnlyPaperExposureSizingSection
    )
    upsize: ShortOnlyUpsizeSection = field(default_factory=ShortOnlyUpsizeSection)
    damage_guard: ShortOnlyDamageGuardSection = field(default_factory=ShortOnlyDamageGuardSection)
    edge: ShortOnlyEdgeSection = field(default_factory=ShortOnlyEdgeSection)
    sizing: ShortOnlySizingSection = field(default_factory=ShortOnlySizingSection)
    microstructure: ShortOnlyMicrostructureSection = field(default_factory=ShortOnlyMicrostructureSection)
    confirmation: ShortOnlyConfirmationSection = field(default_factory=ShortOnlyConfirmationSection)
    ml: ShortOnlyMlSection = field(default_factory=ShortOnlyMlSection)
    exits: ShortOnlyExitsSection = field(default_factory=ShortOnlyExitsSection)


@dataclass(frozen=True)
class ExecutionSection:
    mode: TradeMode = TradeMode.LOCAL_PAPER
    slippage_bps: float = 5.0
    commission_bps: float = 4.0
    state_path: str = "state/paper_state.json"
    allow_live_trading: bool = False


@dataclass(frozen=True)
class BacktestSection:
    initial_cash: float = 1_000_000.0
    warmup_bars: int = 60


@dataclass(frozen=True)
class ReportingSection:
    output_dir: str = "runs"
    write_csv: bool = True


@dataclass(frozen=True)
class ResearchSection:
    strategy_styles: list[str] = field(default_factory=lambda: ["sma_breakout"])
    fast_windows: list[int] = field(default_factory=lambda: [10, 15, 20])
    slow_windows: list[int] = field(default_factory=lambda: [30, 40, 50])
    require_breakout_values: list[bool] = field(default_factory=lambda: [True])
    opening_range_bars_values: list[int] = field(default_factory=lambda: [2, 3])
    rel_volume_threshold_values: list[float] = field(default_factory=lambda: [1.0, 1.15, 1.3])
    atr_stop_multipliers: list[float] = field(default_factory=lambda: [1.5, 2.0])
    reward_to_risk_values: list[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])
    breakeven_trigger_pct_values: list[float] = field(default_factory=lambda: [0.0])
    trailing_profit_trigger_rub_values: list[float] = field(default_factory=lambda: [0.0])
    trailing_profit_lock_ratio_values: list[float] = field(default_factory=lambda: [0.0])
    trend_strength_values: list[float] = field(default_factory=lambda: [0.004, 0.006])
    adx_min_values: list[float] = field(default_factory=lambda: [20.0])
    rsi_long_max_values: list[float] = field(default_factory=lambda: [70.0, 75.0])
    rsi_short_min_values: list[float] = field(default_factory=lambda: [25.0, 30.0])
    subset_min_size: int = 1
    subset_max_size: int = 3
    top_n: int = 10
    min_trades: int = 6
    walk_forward_train_months: int = 6
    walk_forward_test_months: int = 1
    walk_forward_step_months: int = 1
    monte_carlo_iterations: int = 1000
    monte_carlo_horizon_months: int = 12
    trading_days_per_month: int = 20
    target_daily_profit_rub: float = 0.0
    target_monthly_return_pct: float = 5.0
    target_monthly_profit_rub: float = 0.0
    random_seed: int = 42


@dataclass(frozen=True)
class GoldenBaselineTimeframesSection:
    primary: str = "15min"
    early_trigger: str = "5min"
    execution_guard: str = "1min"


@dataclass(frozen=True)
class GoldenBaselineEarly5mTriggerSection:
    ema_window: int = 9
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_window: int = 14
    rsi_min: float = 25.0
    rsi_max: float = 55.0
    rolling_low_window_min: int = 6
    rolling_low_window_max: int = 12
    max_close_position: float = 0.35
    max_adverse_ret: float = 0.005


@dataclass(frozen=True)
class GoldenBaselineEarly5mPromotionSection:
    enabled: bool = True
    deadline_15m_bars: int = 1
    close_if_not_promoted: bool = True
    allow_size_add_on_promotion: bool = False
    promoted_entry_mode: str = "golden_15m_short_breakout_promoted"


@dataclass(frozen=True)
class GoldenBaselineEarly5mFailureExitSection:
    enabled: bool = True
    check_after_minutes: int = 5
    close_if_pnl_negative_and_no_continuation: bool = True
    reason: str = "early_5m_failed_fast"


@dataclass(frozen=True)
class GoldenBaselineEarly5mSection:
    enabled: bool = True
    real_trading_enabled: bool = True
    starter_size_multiplier: float = 0.25
    max_positions: int = 3
    max_new_entries_per_cycle: int = 2
    require_15m_bearish_context: bool = True
    require_5m_breakdown: bool = True
    require_1m_execution_guard: bool = True
    promotion_deadline_15m_bars: int = 1
    close_if_not_promoted: bool = True
    shadow_all_rejected: bool = True
    trigger: GoldenBaselineEarly5mTriggerSection = field(
        default_factory=GoldenBaselineEarly5mTriggerSection
    )
    promotion: GoldenBaselineEarly5mPromotionSection = field(
        default_factory=GoldenBaselineEarly5mPromotionSection
    )
    failure_exit: GoldenBaselineEarly5mFailureExitSection = field(
        default_factory=GoldenBaselineEarly5mFailureExitSection
    )


@dataclass(frozen=True)
class GoldenBaselineExecution1mSection:
    enabled: bool = True
    required_for_early_5m: bool = True
    required_for_15m: bool = False
    lookback_bars: int = 3
    block_if_rebound_bars: int = 2
    max_positive_rebound_ret: float = 0.003
    max_close_position_after_rebound: float = 0.65
    block_if_price_above_5m_ema9: bool = True
    block_if_order_book_deteriorates: bool = True
    fallback_for_15m: str = "allow_with_metadata"
    fallback_for_early_5m: str = "shadow_only"


@dataclass(frozen=True)
class GoldenBaselineSection:
    enabled: bool = False
    name: str = "ema_adx_macd_short_breakout_v1_3tf"
    source_run: str = "20260707-105708"
    source_commit: str = "56be2bda69876f917731f81d913fa32aa9aad8b5"
    real_trading_profile: bool = True
    timeframes: list[str] = field(default_factory=lambda: ["15min", "5min", "1min"])
    forbidden_timeframes: list[str] = field(default_factory=lambda: ["10min"])
    timeframes_config: GoldenBaselineTimeframesSection = field(
        default_factory=GoldenBaselineTimeframesSection
    )
    early_5m: GoldenBaselineEarly5mSection = field(default_factory=GoldenBaselineEarly5mSection)
    execution_1m: GoldenBaselineExecution1mSection = field(
        default_factory=GoldenBaselineExecution1mSection
    )


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    app: AppSection
    tbank: TBankSection
    data: DataSection
    strategy: StrategySection
    risk: RiskSection
    execution: ExecutionSection
    backtest: BacktestSection
    reporting: ReportingSection
    research: ResearchSection
    learning_mode: LearningModeSection = field(default_factory=LearningModeSection)
    learning_risk: LearningRiskSection = field(default_factory=LearningRiskSection)
    learning_caps: LearningCapsSection = field(default_factory=LearningCapsSection)
    learning_signals: LearningSignalsSection = field(default_factory=LearningSignalsSection)
    learning_microstructure: LearningMicrostructureSection = field(default_factory=LearningMicrostructureSection)
    ml_learning_policy: MlLearningPolicySection = field(default_factory=MlLearningPolicySection)
    confirmation_5m: Confirmation5mPolicySection = field(default_factory=Confirmation5mPolicySection)
    regime_policy: RegimePolicySection = field(default_factory=RegimePolicySection)
    side_policy: SidePolicySection = field(default_factory=SidePolicySection)
    symbol_health_policy: SymbolHealthPolicySection = field(default_factory=SymbolHealthPolicySection)
    market_selloff_impulse: MarketSelloffImpulseSection = field(default_factory=MarketSelloffImpulseSection)
    paper_alpha_capture: PaperAlphaCaptureSection = field(default_factory=PaperAlphaCaptureSection)
    short_only: ShortOnlySection = field(default_factory=ShortOnlySection)
    short_ev_engine: ShortEvEngineSection = field(default_factory=ShortEvEngineSection)
    golden_baseline: GoldenBaselineSection = field(default_factory=GoldenBaselineSection)

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.root_dir / path

    def runtime_profile_name(self) -> str:
        stem = Path(self.execution.state_path).stem.strip()
        if not stem:
            return "paper"
        if stem.endswith("_state"):
            stem = stem[: -len("_state")]
        elif stem.endswith("-state"):
            stem = stem[: -len("-state")]
        return stem or "paper"

    def autotune_dir(self) -> Path:
        return self.resolve_path(self.configured_autotune_dir())

    def configured_autotune_dir(self) -> str:
        return str(Path(self.reporting.output_dir) / "autotune" / self.runtime_profile_name())


def _parse_instrument(payload: dict[str, Any]) -> Instrument:
    return Instrument(
        symbol=payload["symbol"].strip().upper(),
        instrument_type=InstrumentType(payload["instrument_type"]),
        figi=payload.get("figi", ""),
        uid=payload.get("uid", ""),
        class_code=payload.get("class_code", ""),
        lot_size=int(payload.get("lot_size", 1)),
        tick_size=float(payload.get("tick_size", 0.01)),
        currency=payload.get("currency", "rub"),
        initial_margin_buy=float(payload.get("initial_margin_buy", 0.0)),
        initial_margin_sell=float(payload.get("initial_margin_sell", 0.0)),
        tick_value=float(payload.get("tick_value", 0.0)),
    )


def _dataclass_payload(cls, payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    allowed = set(cls.__dataclass_fields__)
    return {key: value for key, value in payload.items() if key in allowed}


def _parse_mode_risk(payload: dict[str, Any] | None, default: ModeRiskSection) -> ModeRiskSection:
    values = {**default.__dict__, **_dataclass_payload(ModeRiskSection, payload)}
    return ModeRiskSection(**values)


def _learning_risk_payload(
    learning_risk_raw: dict[str, Any] | None,
    risk_raw: dict[str, Any] | None,
    mode: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if isinstance(risk_raw, dict) and isinstance(risk_raw.get(mode), dict):
        payload.update(risk_raw[mode])
    if isinstance(learning_risk_raw, dict) and isinstance(learning_risk_raw.get(mode), dict):
        payload.update(learning_risk_raw[mode])
    aliases = {
        f"max_{mode}_positions": "max_positions",
        f"max_{mode}_trades_per_day": "max_trades_per_day",
    }
    for source in (risk_raw, learning_risk_raw):
        if not isinstance(source, dict):
            continue
        for old_name, new_name in aliases.items():
            if old_name in source and new_name not in payload and _legacy_cap_can_expand_default(
                mode,
                new_name,
                source[old_name],
            ):
                payload[new_name] = source[old_name]
    return payload


def _legacy_cap_can_expand_default(mode: str, field_name: str, value: object) -> bool:
    default_mode = getattr(LearningRiskSection(), mode, None)
    if default_mode is None:
        return False
    try:
        configured = float(value)
        default_value = float(getattr(default_mode, field_name))
    except (TypeError, ValueError):
        return False
    return configured > default_value


def _parse_mode_signal(payload: dict[str, Any] | None, default: ModeSignalSection) -> ModeSignalSection:
    values = {**default.__dict__, **_dataclass_payload(ModeSignalSection, payload)}
    return ModeSignalSection(**values)


def _parse_mode_microstructure(
    payload: dict[str, Any] | None,
    default: ModeMicrostructureSection,
) -> ModeMicrostructureSection:
    values = {**default.__dict__, **_dataclass_payload(ModeMicrostructureSection, payload)}
    return ModeMicrostructureSection(**values)


def _parse_confirmation_5m_policy(payload: dict[str, Any] | None) -> Confirmation5mPolicySection:
    default = Confirmation5mPolicySection()
    raw = payload if isinstance(payload, dict) else {}
    values = {
        **default.__dict__,
        **_dataclass_payload(Confirmation5mPolicySection, raw),
    }
    values["market_selloff_impulse"] = Confirmation5mMarketSelloffImpulseSection(
        **{
            **default.market_selloff_impulse.__dict__,
            **_dataclass_payload(
                Confirmation5mMarketSelloffImpulseSection,
                raw.get("market_selloff_impulse", {}) if isinstance(raw.get("market_selloff_impulse"), dict) else {},
            ),
        }
    )
    return Confirmation5mPolicySection(**values)


def _parse_weak_down_choppy_policy(payload: dict[str, Any] | None) -> WeakDownChoppyRegimePolicySection:
    default = WeakDownChoppyRegimePolicySection()
    raw = payload if isinstance(payload, dict) else {}
    short_confirmation = WeakDownChoppyShortConfirmationSection(
        **{
            **default.short_confirmation.__dict__,
            **_dataclass_payload(
                WeakDownChoppyShortConfirmationSection,
                raw.get("short_confirmation", {}) if isinstance(raw.get("short_confirmation"), dict) else {},
            ),
        }
    )
    long_policy = WeakDownChoppyLongPolicySection(
        **{
            **default.long.__dict__,
            **_dataclass_payload(
                WeakDownChoppyLongPolicySection,
                raw.get("long", {}) if isinstance(raw.get("long"), dict) else {},
            ),
        }
    )
    values = {
        **default.__dict__,
        **_dataclass_payload(WeakDownChoppyRegimePolicySection, raw),
        "short_confirmation": short_confirmation,
        "long": long_policy,
    }
    return WeakDownChoppyRegimePolicySection(**values)


def _parse_regime_long_policy(
    payload: dict[str, Any] | None,
    cls: type,
    default: object,
) -> object:
    raw = payload if isinstance(payload, dict) else {}
    long_raw = raw.get("long", {}) if isinstance(raw.get("long"), dict) else {}
    return cls(**{**default.__dict__, **_dataclass_payload(cls, long_raw)})


def _parse_short_only_sizing(payload: dict[str, Any] | None) -> ShortOnlySizingSection:
    default = ShortOnlySizingSection()
    raw = payload if isinstance(payload, dict) else {}
    values = {
        **default.__dict__,
        **_dataclass_payload(ShortOnlySizingSection, raw),
    }
    values.update(
        {
        "market_selloff_impulse": ShortOnlySizingRegimeSection(
            **{
                **default.market_selloff_impulse.__dict__,
                **_dataclass_payload(
                    ShortOnlySizingRegimeSection,
                    raw.get("market_selloff_impulse", {})
                    if isinstance(raw.get("market_selloff_impulse"), dict)
                    else {},
                ),
            }
        ),
        "clean_downtrend": ShortOnlySizingRegimeSection(
            **{
                **default.clean_downtrend.__dict__,
                **_dataclass_payload(
                    ShortOnlySizingRegimeSection,
                    raw.get("clean_downtrend", {}) if isinstance(raw.get("clean_downtrend"), dict) else {},
                ),
            }
        ),
        "weak_down_choppy": ShortOnlySizingRegimeSection(
            **{
                **default.weak_down_choppy.__dict__,
                **_dataclass_payload(
                    ShortOnlySizingRegimeSection,
                    raw.get("weak_down_choppy", {}) if isinstance(raw.get("weak_down_choppy"), dict) else {},
                ),
            }
        ),
        "mixed_bearish": ShortOnlySizingRegimeSection(
            **{
                **default.mixed_bearish.__dict__,
                **_dataclass_payload(
                    ShortOnlySizingRegimeSection,
                    raw.get("mixed_bearish", {}) if isinstance(raw.get("mixed_bearish"), dict) else {},
                ),
            }
        ),
        "range_chop": ShortOnlySizingRegimeSection(
            **{
                **default.range_chop.__dict__,
                **_dataclass_payload(
                    ShortOnlySizingRegimeSection,
                    raw.get("range_chop", {}) if isinstance(raw.get("range_chop"), dict) else {},
                ),
            }
        ),
        }
    )
    return ShortOnlySizingSection(**values)


def _parse_golden_baseline(payload: dict[str, Any] | None) -> GoldenBaselineSection:
    default = GoldenBaselineSection()
    raw = payload if isinstance(payload, dict) else {}
    values = {
        **default.__dict__,
        **_dataclass_payload(GoldenBaselineSection, raw),
    }
    timeframes_raw = raw.get("timeframes", {})
    if isinstance(timeframes_raw, dict):
        timeframes_config = GoldenBaselineTimeframesSection(
            **{
                **default.timeframes_config.__dict__,
                **_dataclass_payload(GoldenBaselineTimeframesSection, timeframes_raw),
            }
        )
        values["timeframes_config"] = timeframes_config
        values["timeframes"] = [
            timeframes_config.primary,
            timeframes_config.early_trigger,
            timeframes_config.execution_guard,
        ]
    elif isinstance(timeframes_raw, list):
        values["timeframes"] = [str(item) for item in timeframes_raw]
    early_raw = raw.get("early_5m", {}) if isinstance(raw.get("early_5m"), dict) else {}
    default_early = default.early_5m
    trigger = GoldenBaselineEarly5mTriggerSection(
        **{
            **default_early.trigger.__dict__,
            **_dataclass_payload(
                GoldenBaselineEarly5mTriggerSection,
                early_raw.get("trigger", {}) if isinstance(early_raw.get("trigger"), dict) else {},
            ),
        }
    )
    promotion = GoldenBaselineEarly5mPromotionSection(
        **{
            **default_early.promotion.__dict__,
            **_dataclass_payload(
                GoldenBaselineEarly5mPromotionSection,
                early_raw.get("promotion", {}) if isinstance(early_raw.get("promotion"), dict) else {},
            ),
        }
    )
    failure_exit = GoldenBaselineEarly5mFailureExitSection(
        **{
            **default_early.failure_exit.__dict__,
            **_dataclass_payload(
                GoldenBaselineEarly5mFailureExitSection,
                early_raw.get("failure_exit", {}) if isinstance(early_raw.get("failure_exit"), dict) else {},
            ),
        }
    )
    values["early_5m"] = GoldenBaselineEarly5mSection(
        **{
            **default_early.__dict__,
            **_dataclass_payload(GoldenBaselineEarly5mSection, early_raw),
            "trigger": trigger,
            "promotion": promotion,
            "failure_exit": failure_exit,
        }
    )
    values["execution_1m"] = GoldenBaselineExecution1mSection(
        **{
            **default.execution_1m.__dict__,
            **_dataclass_payload(
                GoldenBaselineExecution1mSection,
                raw.get("execution_1m", {}) if isinstance(raw.get("execution_1m"), dict) else {},
            ),
        }
    )
    return GoldenBaselineSection(**values)


def _parse_short_only(payload: dict[str, Any] | None, *, execution_mode: TradeMode) -> ShortOnlySection:
    default = ShortOnlySection()
    raw = payload if isinstance(payload, dict) else {}
    values = {
        **default.__dict__,
        **_dataclass_payload(ShortOnlySection, raw),
    }
    values["real_trade_sources"] = ShortOnlyRealTradeSourcesSection(
        **{
            **default.real_trade_sources.__dict__,
            **_dataclass_payload(
                ShortOnlyRealTradeSourcesSection,
                raw.get("real_trade_sources", {}) if isinstance(raw.get("real_trade_sources"), dict) else {},
            ),
        }
    )
    values["mixed_bearish_override"] = ShortOnlyMixedBearishOverrideSection(
        **{
            **default.mixed_bearish_override.__dict__,
            **_dataclass_payload(
                ShortOnlyMixedBearishOverrideSection,
                raw.get("mixed_bearish_override", {})
                if isinstance(raw.get("mixed_bearish_override"), dict)
                else {},
            ),
        }
    )
    values["synthetic"] = ShortOnlySyntheticSection(
        **{
            **default.synthetic.__dict__,
            **_dataclass_payload(
                ShortOnlySyntheticSection,
                raw.get("synthetic", {}) if isinstance(raw.get("synthetic"), dict) else {},
            ),
        }
    )
    values["paper_exposure_sizing"] = ShortOnlyPaperExposureSizingSection(
        **{
            **default.paper_exposure_sizing.__dict__,
            **_dataclass_payload(
                ShortOnlyPaperExposureSizingSection,
                raw.get("paper_exposure_sizing", {})
                if isinstance(raw.get("paper_exposure_sizing"), dict)
                else {},
            ),
        }
    )
    values["upsize"] = ShortOnlyUpsizeSection(
        **{
            **default.upsize.__dict__,
            **_dataclass_payload(
                ShortOnlyUpsizeSection,
                raw.get("upsize", {}) if isinstance(raw.get("upsize"), dict) else {},
            ),
        }
    )
    values["damage_guard"] = ShortOnlyDamageGuardSection(
        **{
            **default.damage_guard.__dict__,
            **_dataclass_payload(
                ShortOnlyDamageGuardSection,
                raw.get("damage_guard", {}) if isinstance(raw.get("damage_guard"), dict) else {},
            ),
        }
    )
    values["edge"] = ShortOnlyEdgeSection(
        **{
            **default.edge.__dict__,
            **_dataclass_payload(
                ShortOnlyEdgeSection,
                raw.get("edge", {}) if isinstance(raw.get("edge"), dict) else {},
            ),
        }
    )
    values["sizing"] = _parse_short_only_sizing(raw.get("sizing", {}) if isinstance(raw.get("sizing"), dict) else {})
    values["microstructure"] = ShortOnlyMicrostructureSection(
        **{
            **default.microstructure.__dict__,
            **_dataclass_payload(
                ShortOnlyMicrostructureSection,
                raw.get("microstructure", {}) if isinstance(raw.get("microstructure"), dict) else {},
            ),
        }
    )
    values["confirmation"] = ShortOnlyConfirmationSection(
        **{
            **default.confirmation.__dict__,
            **_dataclass_payload(
                ShortOnlyConfirmationSection,
                raw.get("confirmation", {}) if isinstance(raw.get("confirmation"), dict) else {},
            ),
        }
    )
    values["ml"] = ShortOnlyMlSection(
        **{
            **default.ml.__dict__,
            **_dataclass_payload(
                ShortOnlyMlSection,
                raw.get("ml", {}) if isinstance(raw.get("ml"), dict) else {},
            ),
        }
    )
    values["exits"] = ShortOnlyExitsSection(
        **{
            **default.exits.__dict__,
            **_dataclass_payload(
                ShortOnlyExitsSection,
                raw.get("exits", {}) if isinstance(raw.get("exits"), dict) else {},
            ),
        }
    )
    if execution_mode != TradeMode.LOCAL_PAPER:
        values["enabled"] = False
    return ShortOnlySection(**values)


def _parse_short_ev_engine(payload: dict[str, Any] | None, *, execution_mode: TradeMode) -> ShortEvEngineSection:
    default = ShortEvEngineSection()
    raw = payload if isinstance(payload, dict) else {}
    values = {
        **default.__dict__,
        **_dataclass_payload(ShortEvEngineSection, raw),
    }
    values["ev_gate"] = ShortEvEngineEvGateSection(
        **{
            **default.ev_gate.__dict__,
            **_dataclass_payload(
                ShortEvEngineEvGateSection,
                raw.get("ev_gate", {}) if isinstance(raw.get("ev_gate"), dict) else {},
            ),
        }
    )
    values["probe"] = ShortEvEngineProbeSection(
        **{
            **default.probe.__dict__,
            **_dataclass_payload(
                ShortEvEngineProbeSection,
                raw.get("probe", {}) if isinstance(raw.get("probe"), dict) else {},
            ),
        }
    )
    exits_raw = raw.get("exits", {}) if isinstance(raw.get("exits"), dict) else {}
    default_exits = default.exits
    values["exits"] = ShortEvExitsSection(
        breakeven=ShortEvBreakevenExitSection(
            **{
                **default_exits.breakeven.__dict__,
                **_dataclass_payload(
                    ShortEvBreakevenExitSection,
                    exits_raw.get("breakeven", {}) if isinstance(exits_raw.get("breakeven"), dict) else {},
                ),
            }
        ),
        trailing=ShortEvTrailingExitSection(
            **{
                **default_exits.trailing.__dict__,
                **_dataclass_payload(
                    ShortEvTrailingExitSection,
                    exits_raw.get("trailing", {}) if isinstance(exits_raw.get("trailing"), dict) else {},
                ),
            }
        ),
        order_book_tightening=ShortEvOrderBookTighteningSection(
            **{
                **default_exits.order_book_tightening.__dict__,
                **_dataclass_payload(
                    ShortEvOrderBookTighteningSection,
                    exits_raw.get("order_book_tightening", {})
                    if isinstance(exits_raw.get("order_book_tightening"), dict)
                    else {},
                ),
            }
        ),
    )
    values["timeframes"] = ShortEvTimeframesSection(
        **{
            **default.timeframes.__dict__,
            **_dataclass_payload(
                ShortEvTimeframesSection,
                raw.get("timeframes", {}) if isinstance(raw.get("timeframes"), dict) else {},
            ),
        }
    )
    values["damage_guard"] = ShortEvDamageGuardSection(
        **{
            **default.damage_guard.__dict__,
            **_dataclass_payload(
                ShortEvDamageGuardSection,
                raw.get("damage_guard", {}) if isinstance(raw.get("damage_guard"), dict) else {},
            ),
        }
    )
    setups_raw = raw.get("setups", {}) if isinstance(raw.get("setups"), dict) else {}
    early_raw = (
        setups_raw.get("early_5m_acceleration_short", {})
        if isinstance(setups_raw.get("early_5m_acceleration_short"), dict)
        else {}
    )
    override_raw = (
        early_raw.get("context_override", {})
        if isinstance(early_raw.get("context_override"), dict)
        else {}
    )
    default_early = default.setups.early_5m_acceleration_short
    early_values = {
        **default_early.__dict__,
        **_dataclass_payload(ShortEvEarly5mSetupSection, early_raw),
    }
    early_values["context_override"] = ShortEvEarly5mContextOverrideSection(
        **{
            **default_early.context_override.__dict__,
            **_dataclass_payload(ShortEvEarly5mContextOverrideSection, override_raw),
        }
    )
    values["setups"] = ShortEvSetupsSection(
        early_5m_acceleration_short=ShortEvEarly5mSetupSection(**early_values)
    )
    if execution_mode != TradeMode.LOCAL_PAPER:
        values["enabled"] = False
    if not bool(values.get("live_enabled", False)):
        values["live_enabled"] = False
    return ShortEvEngineSection(**values)


def load_config(config_path: str | Path) -> AppConfig:
    config_path = Path(config_path).resolve()
    root_dir = config_path.parent.parent
    load_dotenv(root_dir / ".env")

    raw = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))

    app = AppSection(**raw.get("app", {}))
    tbank = TBankSection(**raw.get("tbank", {}))

    data_raw = raw.get("data", {})
    instruments = [_parse_instrument(item) for item in data_raw.get("instruments", [])]
    data = DataSection(
        source=data_raw.get("source", "tbank"),
        timeframe=data_raw.get("timeframe", "hour"),
        history_days=int(data_raw.get("history_days", 120)),
        tbank_candle_source=str(data_raw.get("tbank_candle_source", "include-weekend")),
        csv_path=data_raw.get("csv_path", ""),
        parquet_dir_path=data_raw.get("parquet_dir_path", ""),
        local_data_pack_path=data_raw.get("local_data_pack_path", ""),
        instruments=instruments,
    )

    strategy_values = _dataclass_payload(StrategySection, raw.get("strategy", {}))
    entry_confirmation_raw = raw.get("entry_confirmation", {})
    if isinstance(entry_confirmation_raw, dict):
        entry_aliases = {
            "timeframe": "entry_confirmation_timeframe",
            "min_bars": "entry_confirmation_min_bars",
            "max_adverse_ret": "entry_confirmation_max_adverse_ret",
        }
        for raw_key, strategy_key in entry_aliases.items():
            if raw_key in entry_confirmation_raw:
                strategy_values[strategy_key] = entry_confirmation_raw[raw_key]
    order_book_raw = raw.get("order_book", {})
    if isinstance(order_book_raw, dict):
        order_book_aliases = {
            "depth": "order_book_depth",
            "require_order_book": "require_order_book",
            "max_entry_spread_bps": "max_entry_spread_bps",
            "min_entry_liquidity_cover": "min_entry_liquidity_cover",
            "min_entry_book_imbalance": "min_entry_book_imbalance",
        }
        for raw_key, strategy_key in order_book_aliases.items():
            if raw_key in order_book_raw:
                strategy_values[strategy_key] = order_book_raw[raw_key]
    strategy = StrategySection(**strategy_values)
    risk_raw = raw.get("risk", {})
    risk = RiskSection(**_dataclass_payload(RiskSection, risk_raw))

    learning_mode_raw: dict[str, Any] = {}
    if isinstance(risk_raw.get("learning_mode"), dict):
        learning_mode_raw.update(risk_raw["learning_mode"])
    if isinstance(raw.get("learning_mode"), dict):
        learning_mode_raw.update(raw["learning_mode"])
    learning_mode = LearningModeSection(**_dataclass_payload(LearningModeSection, learning_mode_raw))
    default_learning_risk = LearningRiskSection()
    learning_risk_raw = raw.get("learning_risk", {})
    learning_risk = LearningRiskSection(
        normal=_parse_mode_risk(
            _learning_risk_payload(learning_risk_raw, risk_raw, "normal"),
            default_learning_risk.normal,
        ),
        probe=_parse_mode_risk(
            _learning_risk_payload(learning_risk_raw, risk_raw, "probe"),
            default_learning_risk.probe,
        ),
        exploration=_parse_mode_risk(
            _learning_risk_payload(learning_risk_raw, risk_raw, "exploration"),
            default_learning_risk.exploration,
        ),
    )
    learning_caps_raw: dict[str, Any] = {}
    if isinstance(risk_raw.get("learning_caps"), dict):
        learning_caps_raw.update(risk_raw["learning_caps"])
    if isinstance(raw.get("learning_caps"), dict):
        learning_caps_raw.update(raw["learning_caps"])
    learning_caps = LearningCapsSection(
        **{
            **LearningCapsSection().__dict__,
            **_dataclass_payload(LearningCapsSection, learning_caps_raw),
        }
    )
    signals_raw = raw.get("signals", {})
    default_learning_signals = LearningSignalsSection()
    learning_signals = LearningSignalsSection(
        normal=_parse_mode_signal(
            signals_raw.get("normal", {}) if isinstance(signals_raw, dict) else {},
            default_learning_signals.normal,
        ),
        probe=_parse_mode_signal(
            signals_raw.get("probe", {}) if isinstance(signals_raw, dict) else {},
            default_learning_signals.probe,
        ),
        exploration=_parse_mode_signal(
            signals_raw.get("exploration", {}) if isinstance(signals_raw, dict) else {},
            default_learning_signals.exploration,
        ),
    )
    microstructure_raw = raw.get("microstructure", {})
    default_learning_microstructure = LearningMicrostructureSection()
    learning_microstructure = LearningMicrostructureSection(
        normal=_parse_mode_microstructure(
            microstructure_raw.get("normal", {}) if isinstance(microstructure_raw, dict) else {},
            default_learning_microstructure.normal,
        ),
        probe=_parse_mode_microstructure(
            microstructure_raw.get("probe", {}) if isinstance(microstructure_raw, dict) else {},
            default_learning_microstructure.probe,
        ),
        exploration=_parse_mode_microstructure(
            microstructure_raw.get("exploration", {}) if isinstance(microstructure_raw, dict) else {},
            default_learning_microstructure.exploration,
        ),
        market_selloff_impulse=MarketSelloffMicrostructureSection(
            **{
                **default_learning_microstructure.market_selloff_impulse.__dict__,
                **_dataclass_payload(
                    MarketSelloffMicrostructureSection,
                    microstructure_raw.get("market_selloff_impulse", {})
                    if isinstance(microstructure_raw, dict)
                    else {},
                ),
            }
        ),
    )
    ml_learning_policy = MlLearningPolicySection(
        **_dataclass_payload(MlLearningPolicySection, raw.get("ml_learning", {}))
    )
    confirmation_5m = _parse_confirmation_5m_policy(raw.get("confirmation_5m", {}))
    side_raw = raw.get("side", {})
    side_policy = SidePolicySection(
        long=LongSidePolicySection(
            **_dataclass_payload(
                LongSidePolicySection,
                side_raw.get("long", {}) if isinstance(side_raw, dict) else {},
            )
        )
    )
    regime_policy_raw = raw.get("regime_policy", {})
    regime_policy_raw = regime_policy_raw if isinstance(regime_policy_raw, dict) else {}
    regime_policy = RegimePolicySection(
        weak_down_choppy=_parse_weak_down_choppy_policy(
            regime_policy_raw.get("weak_down_choppy", {})
            if isinstance(regime_policy_raw.get("weak_down_choppy"), dict)
            else {}
        ),
        clean_uptrend=CleanUptrendRegimePolicySection(
            long=_parse_regime_long_policy(
                regime_policy_raw.get("clean_uptrend", {})
                if isinstance(regime_policy_raw.get("clean_uptrend"), dict)
                else {},
                CleanUptrendLongPolicySection,
                CleanUptrendLongPolicySection(),
            )
        ),
        mixed=MixedRegimePolicySection(
            long=_parse_regime_long_policy(
                regime_policy_raw.get("mixed", {})
                if isinstance(regime_policy_raw.get("mixed"), dict)
                else {},
                MixedLongPolicySection,
                MixedLongPolicySection(),
            )
        ),
        range_chop=RangeChopRegimePolicySection(
            long=_parse_regime_long_policy(
                regime_policy_raw.get("range_chop", {})
                if isinstance(regime_policy_raw.get("range_chop"), dict)
                else {},
                RangeChopLongPolicySection,
                RangeChopLongPolicySection(),
            )
        ),
    )
    symbol_health_raw = raw.get("symbol_health", {})
    default_symbol_health = SymbolHealthPolicySection()
    symbol_health_policy = SymbolHealthPolicySection(
        probation=SymbolHealthModeSection(
            **{
                **default_symbol_health.probation.__dict__,
                **_dataclass_payload(
                    SymbolHealthModeSection,
                    symbol_health_raw.get("probation", {}) if isinstance(symbol_health_raw, dict) else {},
                ),
            }
        ),
        weak=SymbolHealthModeSection(
            **{
                **default_symbol_health.weak.__dict__,
                **_dataclass_payload(
                    SymbolHealthModeSection,
                    symbol_health_raw.get("weak", {}) if isinstance(symbol_health_raw, dict) else {},
                ),
            }
        ),
        observe_only=SymbolHealthModeSection(
            **{
                **default_symbol_health.observe_only.__dict__,
                **_dataclass_payload(
                    SymbolHealthModeSection,
                    symbol_health_raw.get("observe_only", {}) if isinstance(symbol_health_raw, dict) else {},
                ),
            }
        ),
    )
    selloff_raw = raw.get("market_selloff_impulse", {})
    default_selloff = MarketSelloffImpulseSection()
    market_selloff_impulse = MarketSelloffImpulseSection(
        basket=MarketSelloffBasketSection(
            **{
                **default_selloff.basket.__dict__,
                **_dataclass_payload(
                    MarketSelloffBasketSection,
                    selloff_raw.get("basket", {}) if isinstance(selloff_raw, dict) else {},
                ),
            }
        ),
        risk=MarketSelloffRiskSection(
            **{
                **default_selloff.risk.__dict__,
                **_dataclass_payload(
                    MarketSelloffRiskSection,
                    selloff_raw.get("risk", {}) if isinstance(selloff_raw, dict) else {},
                ),
            }
        ),
        learning_caps=MarketSelloffLearningCapsSection(
            **{
                **default_selloff.learning_caps.__dict__,
                **_dataclass_payload(
                    MarketSelloffLearningCapsSection,
                    selloff_raw.get("learning_caps", {}) if isinstance(selloff_raw, dict) else {},
                ),
            }
        ),
        long=MarketSelloffLongPolicySection(
            **{
                **default_selloff.long.__dict__,
                **_dataclass_payload(
                    MarketSelloffLongPolicySection,
                    selloff_raw.get("long", {}) if isinstance(selloff_raw, dict) else {},
                ),
            }
        ),
    )

    execution_raw = raw.get("execution", {})
    execution = ExecutionSection(
        mode=TradeMode(execution_raw.get("mode", TradeMode.LOCAL_PAPER.value)),
        slippage_bps=float(execution_raw.get("slippage_bps", 5.0)),
        commission_bps=float(execution_raw.get("commission_bps", 4.0)),
        state_path=execution_raw.get("state_path", "state/paper_state.json"),
        allow_live_trading=bool(execution_raw.get("allow_live_trading", False)),
    )
    paper_alpha_raw = raw.get("paper_alpha_capture", {})
    default_paper_alpha = PaperAlphaCaptureSection()
    paper_alpha_values = {
        **default_paper_alpha.__dict__,
        **_dataclass_payload(PaperAlphaCaptureSection, paper_alpha_raw if isinstance(paper_alpha_raw, dict) else {}),
    }
    if execution.mode != TradeMode.LOCAL_PAPER:
        paper_alpha_values["enabled"] = False
    paper_alpha_capture = PaperAlphaCaptureSection(**paper_alpha_values)
    short_only = _parse_short_only(raw.get("short_only", {}), execution_mode=execution.mode)
    short_ev_engine = _parse_short_ev_engine(raw.get("short_ev_engine", {}), execution_mode=execution.mode)
    golden_baseline = _parse_golden_baseline(raw.get("golden_baseline", {}))
    if short_only.enabled:
        side_policy = SidePolicySection(
            long=LongSidePolicySection(
                **{
                    **side_policy.long.__dict__,
                    "normal_enabled": False,
                    "probe_enabled": False,
                    "exploration_enabled": False,
                }
            )
        )

    backtest = BacktestSection(**raw.get("backtest", {}))
    reporting = ReportingSection(**raw.get("reporting", {}))
    research = ResearchSection(**raw.get("research", {}))

    return AppConfig(
        root_dir=root_dir,
        app=app,
        tbank=tbank,
        data=data,
        strategy=strategy,
        risk=risk,
        learning_mode=learning_mode,
        learning_risk=learning_risk,
        learning_caps=learning_caps,
        learning_signals=learning_signals,
        learning_microstructure=learning_microstructure,
        ml_learning_policy=ml_learning_policy,
        confirmation_5m=confirmation_5m,
        regime_policy=regime_policy,
        side_policy=side_policy,
        symbol_health_policy=symbol_health_policy,
        market_selloff_impulse=market_selloff_impulse,
        paper_alpha_capture=paper_alpha_capture,
        short_only=short_only,
        short_ev_engine=short_ev_engine,
        golden_baseline=golden_baseline,
        execution=execution,
        backtest=backtest,
        reporting=reporting,
        research=research,
    )


def read_secret_from_env_or_file(env_name: str, file_path: str, *, label: str) -> str:
    value = os.environ.get(env_name, "")
    if value.strip():
        return _normalize_secret_value(value)

    if file_path:
        path = Path(os.path.expandvars(os.path.expanduser(file_path)))
        if not path.exists():
            raise RuntimeError(f"{label} file does not exist: {path}")
        return _normalize_secret_value(path.read_text(encoding="utf-8-sig"))

    raise RuntimeError(f"{label} is missing. Set {env_name} or configure a token file.")


def _normalize_secret_value(raw: str) -> str:
    text = raw.strip().strip("\"'")
    if not text:
        raise RuntimeError("Secret value is empty.")

    bearer_match = re.search(r"Bearer\s+([A-Za-z0-9._=-]{20,})", text, flags=re.IGNORECASE)
    if bearer_match:
        return bearer_match.group(1)

    for raw_line in text.splitlines():
        line = raw_line.strip().strip("\"'")
        if not line:
            continue
        if "=" in line:
            line = line.split("=", 1)[1].strip().strip("\"'")
        token_match = re.search(r"([A-Za-z0-9._=-]{20,})", line)
        if token_match:
            return token_match.group(1)

    if re.fullmatch(r"[A-Za-z0-9._=-]{20,}", text):
        return text
    raise RuntimeError("Secret file does not contain a valid-looking token.")
