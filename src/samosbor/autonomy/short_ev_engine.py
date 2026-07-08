from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from ..analysis.indicators import atr, average_turnover, rolling_low
from ..domain import Candle, SignalDirection, TradeRecord
from .golden_baseline import _baseline_features, _trigger_features

ALLOWED_SHORT_SETUPS = (
    "normal_15m_trend_short",
    "golden_15m_breakout_short",
    "early_5m_acceleration_short",
    "failed_rebound_short",
    "market_selloff_short",
)

LEGACY_ENTRY_MODE_TO_SETUP = {
    "golden_15m_short_breakout": "golden_15m_breakout_short",
    "golden_15m_short_breakout_promoted": "golden_15m_breakout_short",
    "early_5m_starter_short": "early_5m_acceleration_short",
    "market_breakdown_short": "market_selloff_short",
    "selloff_momentum_short": "market_selloff_short",
    "panic_probe_short": "market_selloff_short",
    "post_selloff_failed_rebound_short": "failed_rebound_short",
    "pullback_short": "failed_rebound_short",
    "trend_short": "normal_15m_trend_short",
}

GOLDEN_PRIOR_SETUPS = {"normal_15m_trend_short", "golden_15m_breakout_short"}


@dataclass(frozen=True)
class SetupVerdict:
    setup_id: str
    passed: bool
    reason: str
    failed_conditions: list[str]
    default_size_multiplier: float
    source: str
    indicators: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CostEstimate:
    entry_commission_rub: float
    exit_commission_rub: float
    spread_cost_rub: float
    slippage_cost_rub: float
    total_cost_rub: float
    total_cost_bps: float

    def to_dict(self) -> dict[str, float]:
        return {
            "entry_commission_rub": round(self.entry_commission_rub, 2),
            "exit_commission_rub": round(self.exit_commission_rub, 2),
            "spread_cost_rub": round(self.spread_cost_rub, 2),
            "slippage_cost_rub": round(self.slippage_cost_rub, 2),
            "total_cost_rub": round(self.total_cost_rub, 2),
            "total_cost_bps": round(self.total_cost_bps, 4),
        }


@dataclass(frozen=True)
class SetupStats:
    setup_id: str
    sample_count: int
    wins: int
    losses: int
    win_rate: float
    avg_win_net_rub: float
    avg_loss_net_rub: float
    avg_mfe_pct: float
    avg_mae_pct: float
    p_hit_breakeven_03_04: float
    p_stop: float
    p_runner: float
    values_are_net: bool
    source: str


@dataclass(frozen=True)
class EVResult:
    setup_id: str
    sample_count: int
    win_rate: float
    avg_win_net_rub: float
    avg_loss_net_rub: float
    avg_mfe_pct: float
    avg_mae_pct: float
    p_hit_breakeven_03_04: float
    p_stop: float
    p_runner: float
    expected_gross_rub: float
    estimated_commission_rub: float
    estimated_spread_cost_rub: float
    estimated_slippage_rub: float
    estimated_total_cost_rub: float
    ev_net_rub: float
    ev_per_risk: float
    confidence: float
    source: str
    decision: str
    reason: str
    values_are_net: bool
    ml_expected_net_edge_rub: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, float):
                payload[key] = round(value, 6)
        return payload


def canonical_setup_id(value: object) -> str:
    setup_id = str(value or "").strip()
    if setup_id in ALLOWED_SHORT_SETUPS:
        return setup_id
    return LEGACY_ENTRY_MODE_TO_SETUP.get(setup_id, setup_id)


def estimate_round_trip_costs(signal, quantity_lots: int, config) -> CostEstimate:
    quantity_units = max(1, int(quantity_lots)) * int(getattr(signal.instrument, "lot_size", 1) or 1)
    notional = max(0.0, float(signal.entry_price) * quantity_units)
    commission_bps = max(0.0, float(getattr(config.execution, "commission_bps", 4.0)))
    slippage_bps = max(0.0, float(getattr(config.execution, "slippage_bps", 5.0)))
    micro = signal.metadata.get("microstructure", {}) if isinstance(signal.metadata, dict) else {}
    spread_bps = _safe_float(micro.get("estimated_spread_cost_bps")) if isinstance(micro, dict) else None
    if spread_bps is None and isinstance(micro, dict):
        spread_bps = _safe_float(micro.get("spread_bps"))
    if spread_bps is None:
        spread_bps = max(
            0.0,
            float(getattr(getattr(config, "short_ev_engine", object()), "default_spread_cost_bps", 8.0)),
        )
    entry_commission = notional * commission_bps / 10_000.0
    exit_commission = notional * commission_bps / 10_000.0
    spread_cost = notional * max(0.0, float(spread_bps)) / 10_000.0
    slippage_cost = notional * (slippage_bps * 2.0) / 10_000.0
    total = entry_commission + exit_commission + spread_cost + slippage_cost
    total_bps = commission_bps * 2.0 + slippage_bps * 2.0 + max(0.0, float(spread_bps))
    return CostEstimate(
        entry_commission_rub=entry_commission,
        exit_commission_rub=exit_commission,
        spread_cost_rub=spread_cost,
        slippage_cost_rub=slippage_cost,
        total_cost_rub=total,
        total_cost_bps=total_bps,
    )


def build_setup_stats(trades: list[TradeRecord], setup_id: str) -> SetupStats:
    setup_id = canonical_setup_id(setup_id)
    sample = [
        trade
        for trade in trades
        if trade.direction == SignalDirection.SHORT
        and _trade_setup_id(trade) == setup_id
    ]
    wins = [float(trade.net_pnl) for trade in sample if float(trade.net_pnl) > 0.0]
    losses = [float(trade.net_pnl) for trade in sample if float(trade.net_pnl) < 0.0]
    mfe_pct = [_trade_mfe_pct(trade) for trade in sample]
    mae_pct = [_trade_mae_pct(trade) for trade in sample]
    mfe_pct = [value for value in mfe_pct if value is not None]
    mae_pct = [value for value in mae_pct if value is not None]
    stop_count = sum(1 for trade in sample if "stop" in str(trade.reason).lower())
    runner_count = sum(1 for trade in sample if "runner" in str(trade.reason).lower())
    be_count = sum(1 for value in mfe_pct if 0.003 <= value <= 0.004 or value >= 0.0035)
    total = len(sample)
    return SetupStats(
        setup_id=setup_id,
        sample_count=total,
        wins=len(wins),
        losses=len(losses),
        win_rate=(len(wins) / total if total else 0.0),
        avg_win_net_rub=(mean(wins) if wins else 0.0),
        avg_loss_net_rub=(mean(losses) if losses else 0.0),
        avg_mfe_pct=(mean(mfe_pct) if mfe_pct else 0.0),
        avg_mae_pct=(mean(mae_pct) if mae_pct else 0.0),
        p_hit_breakeven_03_04=(be_count / total if total else 0.0),
        p_stop=(stop_count / total if total else 0.0),
        p_runner=(runner_count / total if total else 0.0),
        values_are_net=True,
        source="empirical",
    )


def estimate_short_setup_ev(
    signal,
    *,
    setup_verdict: SetupVerdict,
    setup_stats: SetupStats,
    costs: CostEstimate,
    quantity_lots: int,
    config,
    ml_expected_net_edge_rub: float | None = None,
) -> EVResult:
    setup_id = canonical_setup_id(setup_verdict.setup_id)
    gate = config.short_ev_engine.ev_gate
    probe_cfg = config.short_ev_engine.probe
    planned_risk = _planned_risk_rub(signal, quantity_lots)
    if setup_id not in set(config.short_ev_engine.allowed_setups):
        return _ev_result(
            setup_id=setup_id,
            stats=setup_stats,
            costs=costs,
            planned_risk=planned_risk,
            ev_net=0.0,
            confidence=0.0,
            source="shadow_only_insufficient_data",
            decision="shadow_only",
            reason="unknown_setup_not_allowed_real",
            ml_expected_net_edge_rub=ml_expected_net_edge_rub,
        )
    if not setup_verdict.passed:
        return _ev_result(
            setup_id=setup_id,
            stats=setup_stats,
            costs=costs,
            planned_risk=planned_risk,
            ev_net=0.0,
            confidence=0.0,
            source="shadow_only_insufficient_data",
            decision="shadow_only",
            reason=setup_verdict.reason or "setup_conditions_failed",
            ml_expected_net_edge_rub=ml_expected_net_edge_rub,
        )
    if ml_expected_net_edge_rub is not None and ml_expected_net_edge_rub < 0.0:
        return _ev_result(
            setup_id=setup_id,
            stats=setup_stats,
            costs=costs,
            planned_risk=planned_risk,
            ev_net=float(ml_expected_net_edge_rub),
            confidence=0.0,
            source="empirical" if setup_stats.sample_count else "shadow_only_insufficient_data",
            decision="blocked_negative_ev",
            reason="fresh_ml_negative_edge_blocks_setup",
            ml_expected_net_edge_rub=ml_expected_net_edge_rub,
        )

    min_samples = int(gate.min_sample_count_for_real)
    stats = setup_stats
    source = "empirical"
    if stats.sample_count < min_samples:
        if bool(gate.allow_golden_baseline_prior) and setup_id in GOLDEN_PRIOR_SETUPS:
            stats = _golden_baseline_prior(setup_id)
            source = "golden_baseline_prior"
        else:
            ev_net = _price_action_ev(signal, quantity_lots, costs)
            decision = "probe_allowed" if bool(probe_cfg.enabled) and ev_net > 0.0 else "shadow_only"
            confidence = 0.35 if decision == "probe_allowed" else 0.0
            return _ev_result(
                setup_id=setup_id,
                stats=SetupStats(
                    setup_id=setup_id,
                    sample_count=setup_stats.sample_count,
                    wins=setup_stats.wins,
                    losses=setup_stats.losses,
                    win_rate=max(0.05, min(0.95, float(signal.strength))),
                    avg_win_net_rub=abs(signal.entry_price - signal.take_profit)
                    * max(1, int(quantity_lots))
                    * signal.instrument.lot_size,
                    avg_loss_net_rub=-abs(signal.stop_price - signal.entry_price)
                    * max(1, int(quantity_lots))
                    * signal.instrument.lot_size,
                    avg_mfe_pct=setup_stats.avg_mfe_pct,
                    avg_mae_pct=setup_stats.avg_mae_pct,
                    p_hit_breakeven_03_04=setup_stats.p_hit_breakeven_03_04,
                    p_stop=setup_stats.p_stop,
                    p_runner=setup_stats.p_runner,
                    values_are_net=False,
                    source="shadow_only_insufficient_data",
                ),
                costs=costs,
                planned_risk=planned_risk,
                ev_net=ev_net,
                confidence=confidence,
                source="shadow_only_insufficient_data",
                decision=decision,
                reason="insufficient_setup_samples_probe" if decision == "probe_allowed" else "insufficient_setup_samples",
                ml_expected_net_edge_rub=ml_expected_net_edge_rub,
            )

    ev_gross = _stats_ev(stats)
    ev_net = ev_gross if stats.values_are_net else ev_gross - costs.total_cost_rub
    confidence = min(1.0, 0.55 + max(0, stats.sample_count - min_samples) / max(1, min_samples) * 0.20)
    if source == "golden_baseline_prior":
        confidence = 0.60
    if ml_expected_net_edge_rub is not None and ml_expected_net_edge_rub > 0 and ev_net > 0:
        confidence = min(1.0, confidence + 0.05)
    ev_per_risk = ev_net / planned_risk if planned_risk > 0 else 0.0
    if ev_net < float(gate.min_ev_net_rub):
        decision = "blocked_negative_ev"
        reason = "setup_ev_below_min_after_costs"
    elif ev_per_risk < float(gate.min_ev_per_risk):
        decision = "blocked_negative_ev"
        reason = "setup_ev_per_risk_below_min"
    elif confidence < float(gate.min_confidence):
        decision = "shadow_only"
        reason = "setup_ev_confidence_below_min"
    else:
        decision = "real_allowed"
        reason = "setup_ev_positive_after_costs"
    return _ev_result(
        setup_id=setup_id,
        stats=stats,
        costs=costs,
        planned_risk=planned_risk,
        ev_net=ev_net,
        confidence=confidence,
        source=source,
        decision=decision,
        reason=reason,
        ml_expected_net_edge_rub=ml_expected_net_edge_rub,
    )


def classify_short_setup(
    signal,
    candles_15m: list[Candle],
    candles_5m: list[Candle],
    *,
    market_regime,
    config,
    real_trade_source: str,
    golden_3tf: dict[str, Any] | None,
    execution_guard: dict[str, Any] | None,
) -> SetupVerdict:
    allowed = set(config.short_ev_engine.allowed_setups)
    if signal.direction != SignalDirection.SHORT:
        return _setup("", False, "not_short_signal", ["not_short_signal"], 0.0, {}, "side")
    if getattr(market_regime, "regime", "") == "range_chop":
        return _setup("", False, "range_chop_not_tradable", ["range_chop"], 0.0, {}, "regime")
    golden_entry = canonical_setup_id((golden_3tf or {}).get("entry_mode", ""))
    if golden_entry == "golden_15m_breakout_short" and bool((golden_3tf or {}).get("passed", False)):
        return _setup(golden_entry, golden_entry in allowed, "passed", [], 1.0, golden_3tf or {}, "golden_3tf")
    if real_trade_source == "early_5m_starter" or golden_entry == "early_5m_acceleration_short":
        return _early_setup(
            candles_15m,
            candles_5m,
            execution_guard or {},
            config,
            allowed,
            golden_3tf=golden_3tf,
            market_regime=market_regime,
        )
    selloff = _market_selloff_setup(signal, candles_15m, market_regime, config, allowed)
    if selloff.passed:
        return selloff
    failed_rebound = _failed_rebound_setup(candles_15m, candles_5m, execution_guard or {}, config, allowed)
    if failed_rebound.passed:
        return failed_rebound
    normal = _normal_setup(candles_15m, config, allowed, require_breakout=False)
    if normal.passed:
        return normal
    golden = _normal_setup(candles_15m, config, allowed, require_breakout=True)
    if golden.passed:
        return _setup(
            "golden_15m_breakout_short",
            "golden_15m_breakout_short" in allowed,
            "passed",
            [],
            1.0,
            golden.indicators,
            "registry",
        )
    return _first_failed([selloff, failed_rebound, normal, golden])


def find_registry_setup_for_raw_signal(
    candles_15m: list[Candle],
    candles_5m: list[Candle],
    *,
    market_regime,
    config,
) -> SetupVerdict:
    allowed = set(config.short_ev_engine.allowed_setups)
    if getattr(market_regime, "regime", "") == "range_chop":
        return _setup("", False, "range_chop_not_tradable", ["range_chop"], 0.0, {}, "regime")
    selloff = _market_selloff_setup(None, candles_15m, market_regime, config, allowed)
    if selloff.passed:
        return selloff
    failed = _failed_rebound_setup(candles_15m, candles_5m, {}, config, allowed)
    if failed.passed:
        return failed
    normal = _normal_setup(candles_15m, config, allowed, require_breakout=False)
    if normal.passed:
        return normal
    return _first_failed([selloff, failed, normal])


def short_net_breakeven_stop(position, costs: dict[str, Any], *, buffer_bps: float) -> float:
    total_cost = _safe_float(costs.get("estimated_total_cost_rub", costs.get("total_cost_rub"))) or 0.0
    units = max(1, int(position.quantity_units))
    cost_per_share = total_cost / units
    buffer = float(position.entry_price) * max(0.0, float(buffer_bps)) / 10_000.0
    return float(position.entry_price) - cost_per_share - buffer


def short_mfe_pct(position) -> float:
    if float(position.entry_price) <= 0:
        return 0.0
    price = float(position.mfe_price or position.entry_price)
    return max(0.0, (float(position.entry_price) - price) / float(position.entry_price))


def short_trailing_stop(
    position,
    candles_5m: list[Candle],
    *,
    atr_window: int,
    atr_multiple: float,
    order_book: dict[str, Any] | None = None,
    tighten_multiplier: float = 1.0,
) -> tuple[float | None, str]:
    if not candles_5m:
        return None, "5m_trailing_unavailable"
    latest = candles_5m[-1]
    atr_value = atr(candles_5m, atr_window)
    recent = candles_5m[-max(2, min(len(candles_5m), atr_window)) :]
    swing_high = max(float(candle.high) for candle in recent)
    candidates = [swing_high]
    if atr_value is not None and atr_value > 0:
        candidates.append(float(latest.close) + atr_value * max(0.0, float(atr_multiple)) * max(0.1, tighten_multiplier))
    if order_book and _adverse_book_for_short(order_book):
        candidates.append(float(latest.close) + (float(position.entry_price) - float(latest.close)) * 0.35)
    stop = min(candidates)
    if stop < float(position.stop_price):
        return stop, "short_ev_trailing"
    return None, "short_stop_already_tighter"


def _normal_setup(
    candles_15m: list[Candle],
    config,
    allowed: set[str],
    *,
    require_breakout: bool,
) -> SetupVerdict:
    setup_id = "golden_15m_breakout_short" if require_breakout else "normal_15m_trend_short"
    failed: list[str] = []
    latest = candles_15m[-1] if candles_15m else None
    features = _baseline_features(candles_15m, config.strategy)
    indicators: dict[str, Any] = dict(features)
    indicators["turnover_rub"] = _round_or_none(average_turnover(candles_15m, config.strategy.volume_window))
    indicators["rolling_low_20"] = _round_or_none(rolling_low(candles_15m[:-1], config.strategy.breakout_window))
    if latest is None:
        failed.append("missing_15m_candles")
    if str(config.data.timeframe).lower() != "15min":
        failed.append("primary_timeframe_not_15min")
    if "10min" in {str(config.data.timeframe).lower(), str(config.strategy.entry_confirmation_timeframe).lower()}:
        failed.append("forbidden_10m_timeframe")
    if str(config.strategy.style).lower() != "ema_adx_macd":
        failed.append("strategy_style_not_ema_adx_macd")
    if not _feature_lt(features, "ema_fast", "ema_slow"):
        failed.append("ema20_not_below_ema50")
    if not _feature_gte(features, "adx", float(config.strategy.adx_min)):
        failed.append("adx_below_min")
    if not _feature_lt_value(features, "macd_hist", 0.0):
        failed.append("macd_hist_not_negative")
    if not _feature_between(features, "rsi", float(config.strategy.rsi_short_min), 55.0):
        failed.append("rsi_outside_short_band")
    if latest is not None and _safe_float(features.get("ema_fast")) is not None and latest.close > float(features["ema_fast"]):
        failed.append("close_above_ema20")
    turnover = _safe_float(indicators.get("turnover_rub"))
    if turnover is None or turnover < float(config.strategy.min_liquidity_rub):
        failed.append("liquidity_below_min")
    if require_breakout:
        breakout_low = _safe_float(indicators.get("rolling_low_20"))
        if latest is None or breakout_low is None or latest.close > breakout_low:
            failed.append("close_not_below_rolling_low20")
    return _setup(setup_id, not failed and setup_id in allowed, "passed" if not failed else "; ".join(failed), failed, 1.0 if require_breakout else 0.5, indicators, "registry")


def _early_setup(
    candles_15m: list[Candle],
    candles_5m: list[Candle],
    execution_guard: dict[str, Any],
    config,
    allowed: set[str],
    *,
    golden_3tf: dict[str, Any] | None = None,
    market_regime=None,
) -> SetupVerdict:
    failed: list[str] = []
    setup_id = "early_5m_acceleration_short"
    early_cfg = config.short_ev_engine.setups.early_5m_acceleration_short
    override_cfg = early_cfg.context_override
    normal_context = _normal_setup(candles_15m, config, allowed, require_breakout=False)
    original_context_failed: list[str] = []
    context_override_used = False
    context_override_reason = ""
    strict_acceleration_down = False
    rsi_below_25_selloff_acceleration = False
    order_book_failures = _early_order_book_failures(golden_3tf or {})
    if not candles_5m:
        failed.extend(
            [
                f"context_{item}"
                for item in normal_context.failed_conditions
                if item != "close_not_below_rolling_low20"
            ]
        )
        failed.append("missing_5m_candles")
        indicators: dict[str, Any] = {"context_15m": normal_context.indicators}
        rolling_low_broken = False
        setup_quality = "unknown"
        quality_flags: list[str] = []
        size_multiplier = float(early_cfg.size_multiplier_without_rolling_low)
        size_multiplier_reason = "missing_5m_candles"
    else:
        trigger = config.golden_baseline.early_5m.trigger
        features_5m = _trigger_features(candles_5m, trigger)
        indicators = {"context_15m": normal_context.indicators, "trigger_5m": features_5m}
        latest = candles_5m[-1]
        strict_acceleration_down = _early_strict_5m_acceleration_down(
            features_5m,
            latest,
            override_cfg,
        )
        rsi_value = _safe_float(features_5m.get("rsi"))
        rsi_below_25_selloff_acceleration = (
            rsi_value is not None
            and rsi_value < 25.0
            and strict_acceleration_down
            and getattr(market_regime, "regime", "") == "market_selloff_impulse"
        )
        context_override_eligible = (
            bool(override_cfg.enabled)
            and bool(override_cfg.allow_when_15m_ema_not_bearish)
            and strict_acceleration_down
            and getattr(market_regime, "regime", "") != "range_chop"
        )
        for item in normal_context.failed_conditions:
            if item == "close_not_below_rolling_low20":
                continue
            context_reason = f"context_{item}"
            if item == "ema20_not_below_ema50" and context_override_eligible:
                original_context_failed.append(context_reason)
                context_override_used = True
                context_override_reason = "strict_5m_acceleration_down"
                continue
            failed.append(context_reason)
        if not _feature_lt_value(features_5m, "ema9_slope", 0.0):
            failed.append("trigger_ema9_slope_not_negative")
        if _safe_float(features_5m.get("ema9")) is None or latest.close > float(features_5m["ema9"]):
            failed.append("trigger_close_above_ema9")
        if not _feature_lt_value(features_5m, "macd_hist", 0.0):
            failed.append("trigger_macd_hist_not_negative")
        if not _feature_between(features_5m, "rsi", 25.0, 55.0) and not rsi_below_25_selloff_acceleration:
            failed.append("trigger_rsi_outside_band")
        rolling_low_max = _safe_float(features_5m.get("rolling_low_max"))
        rolling_low_broken = rolling_low_max is not None and latest.close <= rolling_low_max
        setup_quality = "early_breakdown" if rolling_low_broken else "early_acceleration_no_breakout"
        quality_flags = []
        if not rolling_low_broken:
            quality_flags.append("trigger_close_not_below_rolling_low_quality_penalty")
            if bool(early_cfg.require_rolling_low_break):
                failed.append("trigger_close_not_below_rolling_low")
        if (_safe_float(features_5m.get("close_position")) or 1.0) > 0.35:
            failed.append("trigger_close_position_too_high")
        if context_override_used:
            setup_quality = "context_override_strict_5m_acceleration"
            size_multiplier = float(override_cfg.size_multiplier)
            size_multiplier_reason = "context_override_smaller_size"
        elif rolling_low_broken:
            size_multiplier = float(early_cfg.size_multiplier_with_rolling_low)
            size_multiplier_reason = "rolling_low_break_quality_bonus"
        else:
            size_multiplier = float(early_cfg.size_multiplier_without_rolling_low)
            size_multiplier_reason = "rolling_low_not_broken_smaller_size"
    failed.extend(order_book_failures)
    if not bool(execution_guard.get("available", False)):
        failed.append("execution_1m_unavailable")
    elif not bool(execution_guard.get("passed", False)):
        failed.append("execution_1m_guard_blocked")
    indicators["early_5m"] = {
        "rolling_low_broken": bool(rolling_low_broken),
        "rolling_low_required": bool(early_cfg.require_rolling_low_break),
        "setup_quality": setup_quality,
        "quality_flags": quality_flags,
        "size_multiplier_reason": size_multiplier_reason,
        "context_override_used": bool(context_override_used),
        "context_override_reason": context_override_reason,
        "original_15m_context_failed": original_context_failed,
        "rsi_below_25_selloff_acceleration": bool(rsi_below_25_selloff_acceleration),
        "strict_5m_acceleration_down": bool(strict_acceleration_down),
        "order_book_strict_passed": not bool(order_book_failures),
        "order_book_failures": list(order_book_failures),
        "blocked_by_1m_after_context_override": bool(
            context_override_used and "execution_1m_guard_blocked" in failed
        ),
    }
    return _setup(
        setup_id,
        not failed and setup_id in allowed,
        "passed" if not failed else "; ".join(failed),
        failed,
        size_multiplier,
        indicators,
        "registry",
    )


def _early_order_book_failures(golden_3tf: dict[str, Any]) -> list[str]:
    if not isinstance(golden_3tf, dict):
        return []
    conditions = golden_3tf.get("failed_conditions", [])
    if not isinstance(conditions, list):
        return []
    return [str(item) for item in conditions if str(item).startswith("order_book_")]


def _early_strict_5m_acceleration_down(features_5m: dict[str, Any], latest: Candle, override_cfg) -> bool:
    checks = []
    if bool(override_cfg.require_5m_ema9_slope_negative):
        checks.append(_feature_lt_value(features_5m, "ema9_slope", 0.0))
    if bool(override_cfg.require_5m_macd_hist_negative):
        checks.append(_feature_lt_value(features_5m, "macd_hist", 0.0))
    if bool(override_cfg.require_5m_ret_window_negative):
        checks.append(_feature_lt_value(features_5m, "ret_window", 0.0))
    if bool(override_cfg.require_5m_close_below_ema9):
        ema9 = _safe_float(features_5m.get("ema9"))
        checks.append(ema9 is not None and latest.close <= ema9)
    return all(checks)


def _failed_rebound_setup(
    candles_15m: list[Candle],
    candles_5m: list[Candle],
    execution_guard: dict[str, Any],
    config,
    allowed: set[str],
) -> SetupVerdict:
    setup_id = "failed_rebound_short"
    failed: list[str] = []
    normal_context = _normal_setup(candles_15m, config, allowed, require_breakout=False)
    context_failures = [
        item
        for item in normal_context.failed_conditions
        if item not in {"close_above_ema20", "close_not_below_rolling_low20"}
    ]
    failed.extend([f"context_{item}" for item in context_failures])
    indicators: dict[str, Any] = {"context_15m": normal_context.indicators}
    if len(candles_5m) < 4:
        failed.append("insufficient_5m_candles")
    else:
        latest = candles_5m[-1]
        previous = candles_5m[-2]
        before = candles_5m[-3]
        trigger = config.golden_baseline.early_5m.trigger
        features_5m = _trigger_features(candles_5m, trigger)
        indicators["trigger_5m"] = features_5m
        lower_high = latest.high < max(previous.high, before.high)
        close_below_ema9 = _safe_float(features_5m.get("ema9")) is not None and latest.close <= float(features_5m["ema9"])
        breaks_previous_low = latest.close <= previous.low
        if not lower_high:
            failed.append("5m_lower_high_missing")
        if not close_below_ema9:
            failed.append("5m_close_not_below_ema9")
        if not breaks_previous_low:
            failed.append("previous_5m_low_not_broken")
    if bool(execution_guard.get("available", False)) and not bool(execution_guard.get("passed", False)):
        failed.append("execution_1m_guard_blocked")
    elif execution_guard and not bool(execution_guard.get("available", False)):
        failed.append("execution_1m_unavailable")
    return _setup(setup_id, not failed and setup_id in allowed, "passed" if not failed else "; ".join(failed), failed, 0.40, indicators, "registry")


def _market_selloff_setup(
    signal,
    candles_15m: list[Candle],
    market_regime,
    config,
    allowed: set[str],
) -> SetupVerdict:
    setup_id = "market_selloff_short"
    failed: list[str] = []
    features = getattr(market_regime, "features", {}) if market_regime is not None else {}
    breadth_down = _safe_float(features.get("breadth_down")) or 0.0
    symbols = int(_safe_float(features.get("symbols")) or _safe_float(features.get("symbols_confirming")) or 0)
    ret15 = _safe_float(features.get("universe_ret_15m")) or _safe_float(features.get("ret_15m")) or 0.0
    ret30 = _safe_float(features.get("universe_ret_30m")) or _safe_float(features.get("ret_30m")) or 0.0
    ret60 = _safe_float(features.get("universe_ret_60m")) or _safe_float(features.get("ret_60m")) or 0.0
    if getattr(market_regime, "regime", "") != "market_selloff_impulse":
        failed.append("market_selloff_impulse_missing")
    if breadth_down < 0.60:
        failed.append("breadth_down_below_0_60")
    if symbols and symbols < 10:
        failed.append("symbols_confirming_below_10")
    if min(ret15, ret30, ret60) > -0.004 and getattr(market_regime, "regime", "") != "market_selloff_impulse":
        failed.append("universe_return_not_selloff")
    if len(candles_15m) < 2:
        failed.append("missing_symbol_momentum")
    else:
        latest = candles_15m[-1]
        previous = candles_15m[-2]
        local_low = rolling_low(candles_15m[:-1], min(10, max(1, len(candles_15m) - 1)))
        if latest.close >= previous.close and (local_low is None or latest.close > local_low):
            failed.append("symbol_negative_momentum_missing")
    multiplier = 0.75 if (signal is not None and float(signal.strength) >= 0.70) else 0.50
    indicators = {
        "breadth_down": breadth_down,
        "symbols_confirming": symbols,
        "universe_ret_15m": ret15,
        "universe_ret_30m": ret30,
        "universe_ret_60m": ret60,
    }
    return _setup(setup_id, not failed and setup_id in allowed, "passed" if not failed else "; ".join(failed), failed, multiplier, indicators, "registry")


def _setup(
    setup_id: str,
    passed: bool,
    reason: str,
    failed: list[str],
    multiplier: float,
    indicators: dict[str, Any],
    source: str,
) -> SetupVerdict:
    if setup_id and setup_id not in ALLOWED_SHORT_SETUPS:
        failed = [*failed, "unknown_setup_not_allowed_real"]
        passed = False
        reason = "unknown_setup_not_allowed_real"
    return SetupVerdict(
        setup_id=setup_id,
        passed=bool(passed),
        reason=reason,
        failed_conditions=list(dict.fromkeys(failed)),
        default_size_multiplier=float(multiplier),
        source=source,
        indicators=indicators,
    )


def _first_failed(verdicts: list[SetupVerdict]) -> SetupVerdict:
    for verdict in verdicts:
        if verdict.setup_id:
            return verdict
    return _setup("", False, "unknown_setup_not_allowed_real", ["unknown_setup_not_allowed_real"], 0.0, {}, "registry")


def _ev_result(
    *,
    setup_id: str,
    stats: SetupStats,
    costs: CostEstimate,
    planned_risk: float,
    ev_net: float,
    confidence: float,
    source: str,
    decision: str,
    reason: str,
    ml_expected_net_edge_rub: float | None,
) -> EVResult:
    return EVResult(
        setup_id=setup_id,
        sample_count=stats.sample_count,
        win_rate=stats.win_rate,
        avg_win_net_rub=stats.avg_win_net_rub,
        avg_loss_net_rub=stats.avg_loss_net_rub,
        avg_mfe_pct=stats.avg_mfe_pct,
        avg_mae_pct=stats.avg_mae_pct,
        p_hit_breakeven_03_04=stats.p_hit_breakeven_03_04,
        p_stop=stats.p_stop,
        p_runner=stats.p_runner,
        expected_gross_rub=_stats_ev(stats),
        estimated_commission_rub=costs.entry_commission_rub + costs.exit_commission_rub,
        estimated_spread_cost_rub=costs.spread_cost_rub,
        estimated_slippage_rub=costs.slippage_cost_rub,
        estimated_total_cost_rub=costs.total_cost_rub,
        ev_net_rub=ev_net,
        ev_per_risk=(ev_net / planned_risk if planned_risk > 0 else 0.0),
        confidence=confidence,
        source=source,
        decision=decision,
        reason=reason,
        values_are_net=stats.values_are_net,
        ml_expected_net_edge_rub=ml_expected_net_edge_rub,
    )


def _golden_baseline_prior(setup_id: str) -> SetupStats:
    return SetupStats(
        setup_id=setup_id,
        sample_count=56,
        wins=26,
        losses=30,
        win_rate=26 / 56,
        avg_win_net_rub=360.0,
        avg_loss_net_rub=-185.0,
        avg_mfe_pct=0.006,
        avg_mae_pct=0.003,
        p_hit_breakeven_03_04=0.55,
        p_stop=30 / 56,
        p_runner=0.0,
        values_are_net=True,
        source="golden_baseline_prior",
    )


def _stats_ev(stats: SetupStats) -> float:
    loss_rate = max(0.0, 1.0 - stats.win_rate)
    return stats.win_rate * stats.avg_win_net_rub + loss_rate * stats.avg_loss_net_rub


def _price_action_ev(signal, quantity_lots: int, costs: CostEstimate) -> float:
    units = max(1, int(quantity_lots)) * signal.instrument.lot_size
    planned_risk = abs(float(signal.stop_price) - float(signal.entry_price)) * units
    planned_reward = abs(float(signal.entry_price) - float(signal.take_profit)) * units
    win_probability = max(0.05, min(0.95, float(signal.strength)))
    return planned_reward * win_probability - planned_risk * (1.0 - win_probability) - costs.total_cost_rub


def _planned_risk_rub(signal, quantity_lots: int) -> float:
    return abs(float(signal.stop_price) - float(signal.entry_price)) * max(1, int(quantity_lots)) * signal.instrument.lot_size


def _trade_setup_id(trade: TradeRecord) -> str:
    metadata = trade.entry_metadata if isinstance(trade.entry_metadata, dict) else {}
    engine = metadata.get("short_ev_engine", {})
    if isinstance(engine, dict) and engine.get("setup_id"):
        return canonical_setup_id(engine.get("setup_id"))
    short_only = metadata.get("short_only", {})
    if isinstance(short_only, dict):
        for key in ("setup_id", "entry_mode"):
            if short_only.get(key):
                return canonical_setup_id(short_only.get(key))
    if metadata.get("entry_mode"):
        return canonical_setup_id(metadata.get("entry_mode"))
    return ""


def _trade_mfe_pct(trade: TradeRecord) -> float | None:
    metadata = trade.entry_metadata if isinstance(trade.entry_metadata, dict) else {}
    excursion = metadata.get("trade_excursion", {})
    if not isinstance(excursion, dict) or float(trade.entry_price) <= 0:
        return None
    price = _safe_float(excursion.get("mfe_price"))
    if price is None:
        return None
    if trade.direction == SignalDirection.SHORT:
        return max(0.0, (float(trade.entry_price) - price) / float(trade.entry_price))
    return max(0.0, (price - float(trade.entry_price)) / float(trade.entry_price))


def _trade_mae_pct(trade: TradeRecord) -> float | None:
    metadata = trade.entry_metadata if isinstance(trade.entry_metadata, dict) else {}
    excursion = metadata.get("trade_excursion", {})
    if not isinstance(excursion, dict) or float(trade.entry_price) <= 0:
        return None
    price = _safe_float(excursion.get("mae_price"))
    if price is None:
        return None
    if trade.direction == SignalDirection.SHORT:
        return max(0.0, (price - float(trade.entry_price)) / float(trade.entry_price))
    return max(0.0, (float(trade.entry_price) - price) / float(trade.entry_price))


def _adverse_book_for_short(order_book: dict[str, Any]) -> bool:
    spread = _safe_float(order_book.get("spread_bps")) or 0.0
    imbalance = _safe_float(order_book.get("side_imbalance", order_book.get("imbalance"))) or 0.0
    return spread >= 18.0 or imbalance <= -0.60


def _feature_lt(features: dict[str, Any], left: str, right: str) -> bool:
    left_value = _safe_float(features.get(left))
    right_value = _safe_float(features.get(right))
    return left_value is not None and right_value is not None and left_value < right_value


def _feature_lt_value(features: dict[str, Any], key: str, value: float) -> bool:
    current = _safe_float(features.get(key))
    return current is not None and current < value


def _feature_gte(features: dict[str, Any], key: str, value: float) -> bool:
    current = _safe_float(features.get(key))
    return current is not None and current >= value


def _feature_between(features: dict[str, Any], key: str, low: float, high: float) -> bool:
    current = _safe_float(features.get(key))
    return current is not None and low <= current <= high


def _safe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _round_or_none(value: object) -> float | None:
    numeric = _safe_float(value)
    return round(numeric, 6) if numeric is not None else None
